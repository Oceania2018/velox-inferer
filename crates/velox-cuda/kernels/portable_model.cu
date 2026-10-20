// Portable kernels for transformer/hybrid models: quantized-weight linear and embedding, sigmoid
// gating, causal depthwise convolution and the Gated DeltaNet recurrence. Simple and correct on
// every supported architecture; hardware branches replace them with tuned kernels.
#include <math.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include "velox_common.cuh"

// Weight formats; must match velox_cuda::portable::FORMAT_*.
#define VELOX_FMT_F32 0
#define VELOX_FMT_BF16 1
#define VELOX_FMT_FP8_ROW_BF16 2
#define VELOX_FMT_NVFP4 3
#define VELOX_FMT_Q4 4
#define VELOX_FMT_Q5 5
#define VELOX_FMT_Q6 6
#define VELOX_FMT_Q8 7

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
        case VELOX_FMT_Q4:
        case VELOX_FMT_Q5:
        case VELOX_FMT_Q6:
        case VELOX_FMT_Q8: {  // row_split_k128_v1
            const int bits = d.format == VELOX_FMT_Q4 ? 4 : d.format == VELOX_FMT_Q5 ? 5 : d.format == VELOX_FMT_Q6 ? 6 : 8;
            const int group = bits == 8 ? 32 : 64;
            const int high = bits == 5 ? 8 : bits == 6 ? 16 : 0;
            const long long gpr = (d.k + 127) / 128 * 128 / group;
            const long long groups = d.n * gpr;
            const long long high_off = align256(groups * 32);
            const long long scale_off = high_off + align256(groups * high);
            const long long gi = row * gpr + col / group;
            const int lane = static_cast<int>(col % group);
            uint32_t u;
            if (bits == 8) {
                u = d.base[gi * 32 + lane];
            } else {
                const uint8_t byte = d.base[gi * 32 + lane / 2];
                u = (lane & 1) ? (byte >> 4) : (byte & 0xf);
            }
            const int hb = high > 0 ? bits - 4 : 0;  // Q5: 1, Q6: 2; Q4/Q8 have no high plane
            for (int j = 0; j < hb; ++j) {
                const int t = lane * hb + j;
                u |= static_cast<uint32_t>((d.base[high_off + gi * high + t / 8] >> (t % 8)) & 1) << (4 + j);
            }
            const int q = u >= (1u << (bits - 1)) ? static_cast<int>(u) - (1 << bits) : static_cast<int>(u);
            const uint8_t* s = d.base + scale_off + 2 * gi;
            return __half2float(__ushort_as_half(static_cast<unsigned short>(s[0] | (s[1] << 8)))) * static_cast<float>(q);
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

// Tiled y[m, rows] = x . W^T for larger m: each 64x64 output tile decodes its weight tile once
// into shared memory per K step, so weights are read m/64 times instead of m times.
constexpr int kTile = 64, kTileK = 16;

__global__ void __launch_bounds__(256)
linear_tiled_kernel(const float* __restrict__ x, WDesc w, long long row_begin, int rows, int m, float* __restrict__ y) {
    __shared__ float Xs[kTileK][kTile + 4];
    __shared__ float Ws[kTileK][kTile + 4];
    const int t = threadIdx.x;
    const int m0 = blockIdx.y * kTile, n0 = blockIdx.x * kTile;
    const int lr = t / 4, lk = (t % 4) * 4;
    const int ty = t / 16, tx = t % 16;
    float acc[4][4] = {};
    for (long long k0 = 0; k0 < w.k; k0 += kTileK) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const long long kk = k0 + lk + i;
            const int xm = m0 + lr, wn = n0 + lr;
            Xs[lk + i][lr] = (xm < m && kk < w.k) ? x[static_cast<size_t>(xm) * w.k + kk] : 0.0f;
            Ws[lk + i][lr] = (wn < rows && kk < w.k) ? weight_at(w, row_begin + wn, kk) : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < kTileK; ++kk) {
            float a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                a[i] = Xs[kk][ty * 4 + i];
                b[i] = Ws[kk][tx * 4 + i];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = m0 + ty * 4 + i;
        if (r >= m) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = n0 + tx * 4 + j;
            if (c < rows) y[static_cast<size_t>(r) * rows + c] = acc[i][j];
        }
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
    const WDesc d{w, format, n, k};
    if (m >= 16 && (m + kTile - 1) / kTile <= 65535) {
        const dim3 grid((rows + kTile - 1) / kTile, (m + kTile - 1) / kTile);
        linear_tiled_kernel<<<grid, 256>>>(x, d, row_begin, rows, m, y);
    } else {
        constexpr int kWarps = 8;
        const dim3 grid((rows + kWarps - 1) / kWarps, m < 65535 ? m : 65535);
        linear_kernel<<<grid, kWarps * VELOX_WARP>>>(x, d, row_begin, rows, m, y);
    }
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

// ---- Vision-tower ops ------------------------------------------------------------------------

namespace {

// Non-causal MHA: one warp per (token, head), keys visited in order with an online softmax.
template <int PER>
__global__ void attention_full_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                      const float* __restrict__ v, float* __restrict__ out, int tokens, int heads,
                                      int head_dim, float scale) {
    const int h = blockIdx.x, t = blockIdx.y, lane = threadIdx.x;
    const size_t q_off = (static_cast<size_t>(t) * heads + h) * head_dim;
    float qr[PER], acc[PER];
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int d = lane + p * VELOX_WARP;
        qr[p] = d < head_dim ? q[q_off + d] * scale : 0.0f;
        acc[p] = 0.0f;
    }
    float m = -INFINITY, l = 0.0f;
    for (int j = 0; j < tokens; ++j) {
        const size_t kv_off = (static_cast<size_t>(j) * heads + h) * head_dim;
        float s = 0.0f;
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            const int d = lane + p * VELOX_WARP;
            if (d < head_dim) s += qr[p] * k[kv_off + d];
        }
        s = velox_warp_sum(s);
        const float m_new = fmaxf(m, s);
        const float corr = expf(m - m_new), pj = expf(s - m_new);
        l = l * corr + pj;
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            const int d = lane + p * VELOX_WARP;
            if (d < head_dim) acc[p] = acc[p] * corr + pj * v[kv_off + d];
        }
        m = m_new;
    }
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int d = lane + p * VELOX_WARP;
        if (d < head_dim) out[q_off + d] = acc[p] / l;
    }
}

__global__ void layer_norm_kernel(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
                                  float* __restrict__ y, int dim, float eps) {
    __shared__ float scratch[VELOX_WARP];
    const float* xr = x + static_cast<size_t>(blockIdx.x) * dim;
    float* yr = y + static_cast<size_t>(blockIdx.x) * dim;
    float s = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) s += xr[i];
    const float mean = velox_block_sum(s, scratch) / dim;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float d = xr[i] - mean;
        ss += d * d;
    }
    const float inv = rsqrtf(velox_block_sum(ss, scratch) / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) yr[i] = (xr[i] - mean) * inv * w[i] + b[i];
}

__global__ void add_bias_kernel(float* __restrict__ x, const float* __restrict__ bias, long long len, int dim) {
    for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < len;
         i += static_cast<long long>(gridDim.x) * blockDim.x)
        x[i] += bias[i % dim];
}

__global__ void gelu_kernel(float* __restrict__ x, long long len, int erf_kind) {
    for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < len;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const float a = x[i];
        x[i] = erf_kind ? 0.5f * a * (1.0f + erff(a * 0.70710678118654752f))
                        : 0.5f * a * (1.0f + tanhf(0.79788456080286536f * (a + 0.044715f * a * a * a)));
    }
}

// layout 0: interleaved (axis i % axes, freq theta^(-2i/R)); 1: blocked (see RopeLayout).
__global__ void rope_positions_kernel(float* __restrict__ x, const uint32_t* __restrict__ pos, long long pairs_total,
                                      int tokens, int heads, int head_dim, int rotary_dim, float theta, int axes,
                                      int layout) {
    const int half = rotary_dim / 2;
    for (long long p = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; p < pairs_total;
         p += static_cast<long long>(gridDim.x) * blockDim.x) {
        const int i = static_cast<int>(p % half);
        const long long row = p / half;  // token * heads + head
        const int t = static_cast<int>(row / heads);
        int axis;
        double freq;
        if (layout == 0) {
            axis = i % axes;
            freq = pow(static_cast<double>(theta), -2.0 * i / rotary_dim);
        } else {
            const int per = half / axes;
            axis = i / per;
            freq = pow(static_cast<double>(theta), -2.0 * (i % per) / (static_cast<double>(rotary_dim) / axes));
        }
        double s, c;
        sincos(static_cast<double>(pos[static_cast<size_t>(axis) * tokens + t]) * freq, &s, &c);
        float* base = x + row * head_dim;
        const float a = base[i], b = base[i + half];
        base[i] = static_cast<float>(a * c - b * s);
        base[i + half] = static_cast<float>(a * s + b * c);
    }
}

}  // namespace

VELOX_API int velox_portable_attention_full(const float* q, const float* k, const float* v, float* out, int tokens,
                                            int heads, int head_dim) {
    if (tokens > 65535) return VELOX_ERR_UNSUPPORTED;
    const dim3 grid(heads, tokens);
    const float scale = rsqrtf(static_cast<float>(head_dim));
    if (head_dim <= 32) attention_full_kernel<1><<<grid, VELOX_WARP>>>(q, k, v, out, tokens, heads, head_dim, scale);
    else if (head_dim <= 64) attention_full_kernel<2><<<grid, VELOX_WARP>>>(q, k, v, out, tokens, heads, head_dim, scale);
    else if (head_dim <= 128) attention_full_kernel<4><<<grid, VELOX_WARP>>>(q, k, v, out, tokens, heads, head_dim, scale);
    else if (head_dim <= 256) attention_full_kernel<8><<<grid, VELOX_WARP>>>(q, k, v, out, tokens, heads, head_dim, scale);
    else return VELOX_ERR_UNSUPPORTED;
    return velox_launch_status();
}

VELOX_API int velox_portable_layer_norm(const float* x, const float* w, const float* b, float* y, int rows, int dim,
                                        float eps) {
    layer_norm_kernel<<<rows, 256>>>(x, w, b, y, dim, eps);
    return velox_launch_status();
}

VELOX_API int velox_portable_add_bias(float* x, const float* bias, long long len, int dim) {
    add_bias_kernel<<<velox_grid_1d(len, 256), 256>>>(x, bias, len, dim);
    return velox_launch_status();
}

VELOX_API int velox_portable_gelu(float* x, long long len, int erf_kind) {
    gelu_kernel<<<velox_grid_1d(len, 256), 256>>>(x, len, erf_kind);
    return velox_launch_status();
}

VELOX_API int velox_portable_rope_positions(float* x, const uint32_t* pos, int tokens, int heads, int head_dim,
                                            int rotary_dim, float theta, int axes, int layout) {
    const long long pairs = static_cast<long long>(tokens) * heads * (rotary_dim / 2);
    rope_positions_kernel<<<velox_grid_1d(pairs, 256), 256>>>(x, pos, pairs, tokens, heads, head_dim, rotary_dim, theta,
                                                              axes, layout);
    return velox_launch_status();
}
