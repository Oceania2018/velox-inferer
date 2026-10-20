// Portable kernels for transformer/hybrid models: quantized-weight linear and embedding, sigmoid
// gating, causal depthwise convolution and the Gated DeltaNet recurrence. Simple and correct on
// every supported architecture; hardware branches replace them with tuned kernels.
#include <math.h>
#include <stdint.h>

#include "velox_common.cuh"

// Weight formats; must match velox_cuda::portable::FORMAT_*.
#define VELOX_FMT_F32 0
#define VELOX_FMT_BF16 1
#define VELOX_FMT_FP8_ROW_BF16 2
#define VELOX_FMT_NVFP4 3

namespace {

struct WDesc {
    const uint8_t* base;
    int format;
    long long n;  // parent rows
    long long k;
};

__device__ __forceinline__ long long align256(long long x) { return (x + 255) / 256 * 256; }

__device__ __forceinline__ float bf16_to_f32(uint16_t b) { return __uint_as_float(static_cast<uint32_t>(b) << 16); }

__device__ __forceinline__ float e4m3fn_to_f32(uint8_t b) {
    const int e = (b >> 3) & 0xf, m = b & 7;
    float v;
    if (e == 0) v = ldexpf(static_cast<float>(m), -9);
    else if (e == 15 && m == 7) v = __int_as_float(0x7fc00000);
    else v = ldexpf(1.0f + m * 0.125f, e - 7);
    return (b & 0x80) ? -v : v;
}

__device__ __forceinline__ float e2m1_to_f32(uint8_t c) {
    const float t[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v = t[c & 7];
    return (c & 8) ? -v : v;
}

__device__ __forceinline__ long long nvfp4_scale_offset(long long n, long long g, long long k) {
    const long long k_tiles = k / 64;
    const long long row_tile = n / 128, row_inner = n % 128, scale_tile = g / 4, scale_lane = g % 4;
    return (row_tile * k_tiles + scale_tile) * 512 + (row_inner % 32) * 16 + (row_inner / 32) * 4 + scale_lane;
}

// Decoded W[row, col] of the parent matrix.
__device__ float weight_at(const WDesc& d, long long row, long long col) {
    switch (d.format) {
        case VELOX_FMT_F32:
            return reinterpret_cast<const float*>(d.base)[row * d.k + col];
        case VELOX_FMT_BF16:
            return bf16_to_f32(reinterpret_cast<const uint16_t*>(d.base)[row * d.k + col]);
        case VELOX_FMT_FP8_ROW_BF16: {
            const uint8_t* s = d.base + align256(d.n * d.k) + 2 * row;
            return e4m3fn_to_f32(d.base[row * d.k + col]) * bf16_to_f32(static_cast<uint16_t>(s[0] | (s[1] << 8)));
        }
        default: {  // NVFP4
            const uint8_t byte = d.base[(row * d.k + col) / 2];
            const uint8_t code = (col & 1) ? (byte >> 4) : (byte & 0xf);
            const long long scale_plane = align256(d.n * d.k / 2);
            const float scale = e4m3fn_to_f32(d.base[scale_plane + nvfp4_scale_offset(row, col / 16, d.k)]);
            const float divisor = *reinterpret_cast<const float*>(d.base + scale_plane + d.n * d.k / 16);
            return e2m1_to_f32(code) * scale / divisor;
        }
    }
}

// y[m, rows] = x[m, k] . W[row_begin + r, :]^T, one warp per output element.
__global__ void linear_kernel(const float* __restrict__ x, WDesc w, long long row_begin, int rows, int m,
                              float* __restrict__ y) {
    const int r = blockIdx.x * (blockDim.x / VELOX_WARP) + threadIdx.x / VELOX_WARP;
    const int lane = threadIdx.x % VELOX_WARP;
    if (r >= rows) return;
    for (int i = blockIdx.y; i < m; i += gridDim.y) {
        const float* xr = x + static_cast<size_t>(i) * w.k;
        float acc = 0.0f;
        for (long long c = lane; c < w.k; c += VELOX_WARP) acc += xr[c] * weight_at(w, row_begin + r, c);
        acc = velox_warp_sum(acc);
        if (lane == 0) y[static_cast<size_t>(i) * rows + r] = acc;
    }
}

__global__ void embed_kernel(const uint32_t* __restrict__ ids, WDesc w, long long row_begin, float* __restrict__ out) {
    const long long row = row_begin + ids[blockIdx.x];
    for (long long c = threadIdx.x; c < w.k; c += blockDim.x) out[blockIdx.x * w.k + c] = weight_at(w, row, c);
}

__global__ void sigmoid_mul_kernel(float* __restrict__ x, const float* __restrict__ gate, long long len) {
    for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < len;
         i += static_cast<long long>(gridDim.x) * blockDim.x)
        x[i] = x[i] / (1.0f + expf(-gate[i]));
}

constexpr int kMaxConvWidth = 8;

// One thread per channel walks the tokens in order, carrying the window.
__global__ void conv1d_silu_kernel(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ state,
                                   float* __restrict__ out, int tokens, int channels, int width) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float win[kMaxConvWidth];
    float wc[kMaxConvWidth];
    for (int j = 0; j < width; ++j) wc[j] = w[static_cast<size_t>(j) * channels + c];
    for (int j = 0; j < width - 1; ++j) win[j] = state[static_cast<size_t>(j) * channels + c];
    for (int t = 0; t < tokens; ++t) {
        win[width - 1] = x[static_cast<size_t>(t) * channels + c];
        float acc = 0.0f;
        for (int j = 0; j < width; ++j) acc += wc[j] * win[j];
        out[static_cast<size_t>(t) * channels + c] = acc / (1.0f + expf(-acc));
        for (int j = 0; j < width - 1; ++j) win[j] = win[j + 1];
    }
    for (int j = 0; j < width - 1; ++j) state[static_cast<size_t>(j) * channels + c] = win[j];
}

// One block per value head; thread i < v_dim owns state row S[h][i][:]. q/k are normalised in
// shared memory each step.
__global__ void gated_delta_net_kernel(const float* __restrict__ qkv, const float* __restrict__ ab,
                                       const float* __restrict__ a_log, const float* __restrict__ dt_bias,
                                       float* __restrict__ state, float* __restrict__ out, int tokens, int k_heads,
                                       int v_heads, int k_dim, int v_dim) {
    extern __shared__ float smem[];
    float* qs = smem;          // k_dim
    float* ks = smem + k_dim;  // k_dim
    __shared__ float scratch[VELOX_WARP];
    __shared__ float s_alpha, s_beta;
    const int h = blockIdx.x, kh = h / (v_heads / k_heads), tid = threadIdx.x;
    const int width = 2 * k_heads * k_dim + v_heads * v_dim;
    float* S = state + static_cast<size_t>(h) * v_dim * k_dim;
    const float inv_sqrt_dk = rsqrtf(static_cast<float>(k_dim));

    for (int t = 0; t < tokens; ++t) {
        const float* row = qkv + static_cast<size_t>(t) * width;
        float qv = 0.0f, kv = 0.0f;
        if (tid < k_dim) {
            qv = row[kh * k_dim + tid];
            kv = row[(k_heads + kh) * k_dim + tid];
        }
        const float qq = velox_block_sum(qv * qv, scratch);
        const float kk = velox_block_sum(kv * kv, scratch);
        if (tid < k_dim) {
            qs[tid] = qv * rsqrtf(qq + 1e-6f);
            ks[tid] = kv * rsqrtf(kk + 1e-6f);
        }
        if (tid == 0) {
            const float a = ab[static_cast<size_t>(t) * 2 * v_heads + h] + dt_bias[h];
            const float b = ab[static_cast<size_t>(t) * 2 * v_heads + v_heads + h];
            const float sp = a > 20.0f ? a : log1pf(expf(a));
            s_alpha = expf(-expf(a_log[h]) * sp);
            s_beta = 1.0f / (1.0f + expf(-b));
        }
        __syncthreads();
        if (tid < v_dim) {
            float* Si = S + static_cast<size_t>(tid) * k_dim;
            const float alpha = s_alpha;
            float sk = 0.0f;
            for (int j = 0; j < k_dim; ++j) {
                const float sv = Si[j] * alpha;
                Si[j] = sv;
                sk += sv * ks[j];
            }
            const float delta = s_beta * (row[2 * k_heads * k_dim + h * v_dim + tid] - sk);
            float o = 0.0f;
            for (int j = 0; j < k_dim; ++j) {
                const float sv = Si[j] + delta * ks[j];
                Si[j] = sv;
                o += sv * qs[j];
            }
            out[(static_cast<size_t>(t) * v_heads + h) * v_dim + tid] = o * inv_sqrt_dk;
        }
        __syncthreads();
    }
}

}  // namespace

VELOX_API int velox_portable_linear(const float* x, const uint8_t* w, int format, long long n, long long k,
                                    long long row_begin, int rows, int m, float* y) {
    constexpr int kWarps = 8;
    const dim3 grid((rows + kWarps - 1) / kWarps, m < 65535 ? m : 65535);
    linear_kernel<<<grid, kWarps * VELOX_WARP>>>(x, WDesc{w, format, n, k}, row_begin, rows, m, y);
    return velox_launch_status();
}

VELOX_API int velox_portable_embed(const uint32_t* ids, int count, const uint8_t* w, int format, long long n,
                                   long long k, long long row_begin, float* out) {
    embed_kernel<<<count, 256>>>(ids, WDesc{w, format, n, k}, row_begin, out);
    return velox_launch_status();
}

VELOX_API int velox_portable_sigmoid_mul(float* x, const float* gate, long long len) {
    sigmoid_mul_kernel<<<velox_grid_1d(len, 256), 256>>>(x, gate, len);
    return velox_launch_status();
}

VELOX_API int velox_portable_conv1d_silu(const float* x, const float* w, float* state, float* out, int tokens,
                                         int channels, int width) {
    if (width > kMaxConvWidth) return VELOX_ERR_UNSUPPORTED;
    conv1d_silu_kernel<<<(channels + 127) / 128, 128>>>(x, w, state, out, tokens, channels, width);
    return velox_launch_status();
}

VELOX_API int velox_portable_gated_delta_net(const float* qkv, const float* ab, const float* a_log,
                                             const float* dt_bias, float* state, float* out, int tokens, int k_heads,
                                             int v_heads, int k_dim, int v_dim) {
    const int threads = ((k_dim > v_dim ? k_dim : v_dim) + VELOX_WARP - 1) / VELOX_WARP * VELOX_WARP;
    if (threads > 1024) return VELOX_ERR_UNSUPPORTED;
    gated_delta_net_kernel<<<v_heads, threads, 2 * k_dim * sizeof(float)>>>(qkv, ab, a_log, dt_bias, state, out, tokens,
                                                                         k_heads, v_heads, k_dim, v_dim);
    return velox_launch_status();
}
