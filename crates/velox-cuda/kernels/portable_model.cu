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
linear_tiled_kernel(const float* __restrict__ x, WDesc w, long long row_begin, int rows, int m, float* __restrict__ y,
                    long long k_split) {
    // Split-K: block z covers [z * k_split, (z + 1) * k_split) and writes its own [m, rows] slice.
    const long long k_begin = blockIdx.z * k_split;
    const long long k_end = k_begin + k_split < w.k ? k_begin + k_split : w.k;
    y += static_cast<size_t>(blockIdx.z) * m * rows;
    __shared__ float Xs[kTileK][kTile + 4];
    __shared__ float Ws[kTileK][kTile + 4];
    const int t = threadIdx.x;
    const int m0 = blockIdx.y * kTile, n0 = blockIdx.x * kTile;
    const int lr = t / 4, lk = (t % 4) * 4;
    const int ty = t / 16, tx = t % 16;
    float acc[4][4] = {};
    for (long long k0 = k_begin; k0 < k_end; k0 += kTileK) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const long long kk = k0 + lk + i;
            const int xm = m0 + lr, wn = n0 + lr;
            Xs[lk + i][lr] = (xm < m && kk < k_end) ? x[static_cast<size_t>(xm) * w.k + kk] : 0.0f;
            Ws[lk + i][lr] = (wn < rows && kk < k_end) ? weight_at(w, row_begin + wn, kk) : 0.0f;
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

// One thread per (token, channel): tap j reads the input at time t - (width - 1) + j, from x when
// that is >= 0 and otherwise from the rolling state (oldest first). The state is advanced by
// conv1d_state_kernel afterwards, so every read here sees the state from before the call.
__global__ void conv1d_silu_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                   const float* __restrict__ state, float* __restrict__ out, int tokens, int channels,
                                   int width) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<long long>(tokens) * channels) return;
    const int t = static_cast<int>(i / channels), c = static_cast<int>(i % channels);
    float acc = 0.0f;
    for (int j = 0; j < width; ++j) {
        const int s = t - (width - 1) + j;
        const float in = s >= 0 ? x[static_cast<size_t>(s) * channels + c] : state[static_cast<size_t>(t + j) * channels + c];
        acc += w[static_cast<size_t>(j) * channels + c] * in;
    }
    out[i] = acc / (1.0f + expf(-acc));
}

// Few tokens (decode, verify): one thread per channel runs the window over its inputs and advances
// the state itself (each thread reads its channel's state before writing it, so state may alias
// state_in).
constexpr int kConvSmallTokens = 32;
__global__ void conv1d_silu_small_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                         const float* state_in, float* state, float* __restrict__ out, int tokens,
                                         int channels, int width) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float win[kMaxConvWidth], wt[kMaxConvWidth];
#pragma unroll
    for (int j = 0; j < kMaxConvWidth; ++j) {
        wt[j] = j < width ? w[static_cast<size_t>(j) * channels + c] : 0.0f;
        win[j] = j < width - 1 ? state_in[static_cast<size_t>(j) * channels + c] : 0.0f;
    }
    for (int t = 0; t < tokens; ++t) {
        const float in = x[static_cast<size_t>(t) * channels + c];
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < kMaxConvWidth - 1; ++j)
            if (j < width - 1) acc += wt[j] * win[j];
#pragma unroll
        for (int j = 0; j < kMaxConvWidth; ++j)
            if (j == width - 1) acc += wt[j] * in;
        out[static_cast<size_t>(t) * channels + c] = acc / (1.0f + expf(-acc));
#pragma unroll
        for (int j = 0; j < kMaxConvWidth - 2; ++j)
            if (j < width - 2) win[j] = win[j + 1];
#pragma unroll
        for (int j = 0; j < kMaxConvWidth - 1; ++j)
            if (j == width - 2) win[j] = in;
    }
#pragma unroll
    for (int j = 0; j < kMaxConvWidth - 1; ++j)
        if (j < width - 1) state[static_cast<size_t>(j) * channels + c] = win[j];
}

// state[j] <- input at time tokens - (width - 1) + j. Ascending j only reads indices >= j + tokens
// of the old state, which are not yet overwritten.
__global__ void conv1d_state_kernel(const float* __restrict__ x, const float* state_in, float* state, int tokens,
                                    int channels, int width) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    for (int j = 0; j < width - 1; ++j) {
        const int s = tokens - (width - 1) + j;
        state[static_cast<size_t>(j) * channels + c] =
            s >= 0 ? x[static_cast<size_t>(s) * channels + c] : state_in[static_cast<size_t>(j + tokens) * channels + c];
    }
}

// Gated DeltaNet: state rows are independent, so one warp owns one row S[h][i][:] (lanes split
// k_dim, PER elements each, kept in registers for the whole call). `out` may be null (state only).
template <int PER>
__device__ __forceinline__ void gdn_rows(const float* __restrict__ qkv, const float* __restrict__ ab,
                                         const float* __restrict__ a_log, const float* __restrict__ dt_bias,
                                         const float* state_in, float* state, float* __restrict__ out, int tokens,
                                         int k_heads, int v_heads, int k_dim, int v_dim, int h, int i, int lane) {
    const int kh = h / (v_heads / k_heads);
    const int width = 2 * k_heads * k_dim + v_heads * v_dim;
    const float inv_sqrt_dk = rsqrtf(static_cast<float>(k_dim));
    const float neg_a = -expf(a_log[h]), bias = dt_bias[h];
    const size_t row_off = (static_cast<size_t>(h) * v_dim + i) * k_dim;
    float* row_state = state + row_off;  // written at the end; read from state_in (may alias)
    float s[PER];
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int d = lane + p * VELOX_WARP;
        s[p] = d < k_dim ? state_in[row_off + d] : 0.0f;
    }
    for (int t = 0; t < tokens; ++t) {
        const float* row = qkv + static_cast<size_t>(t) * width;
        float q[PER], k[PER], qq = 0.0f, kk = 0.0f;
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            const int d = lane + p * VELOX_WARP;
            q[p] = d < k_dim ? row[kh * k_dim + d] : 0.0f;
            k[p] = d < k_dim ? row[(k_heads + kh) * k_dim + d] : 0.0f;
            qq += q[p] * q[p];
            kk += k[p] * k[p];
        }
        const float qn = rsqrtf(velox_warp_sum(qq) + 1e-6f), kn = rsqrtf(velox_warp_sum(kk) + 1e-6f);
        const float a = ab[static_cast<size_t>(t) * 2 * v_heads + h] + bias;
        const float b = ab[static_cast<size_t>(t) * 2 * v_heads + v_heads + h];
        const float alpha = expf(neg_a * (a > 20.0f ? a : log1pf(expf(a))));
        const float beta = 1.0f / (1.0f + expf(-b));
        float sk = 0.0f;
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            s[p] *= alpha;
            sk += s[p] * k[p] * kn;
        }
        const float delta = beta * (row[2 * k_heads * k_dim + h * v_dim + i] - velox_warp_sum(sk));
        float o = 0.0f;
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            s[p] += delta * k[p] * kn;
            o += s[p] * q[p] * qn;
        }
        if (out != nullptr) {
            o = velox_warp_sum(o);
            if (lane == 0) out[(static_cast<size_t>(t) * v_heads + h) * v_dim + i] = o * inv_sqrt_dk;
        }
    }
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int d = lane + p * VELOX_WARP;
        if (d < k_dim) row_state[d] = s[p];
    }
}

// Grid: heads x row groups.
template <int PER>
__global__ void gated_delta_net_rows_kernel(const float* __restrict__ qkv, const float* __restrict__ ab,
                                            const float* __restrict__ a_log, const float* __restrict__ dt_bias,
                                            const float* state_in, float* state, float* __restrict__ out,
                                            int tokens, int k_heads, int v_heads, int k_dim, int v_dim) {
    const int warps = blockDim.x / VELOX_WARP;
    const int h = blockIdx.x, i = blockIdx.y * warps + threadIdx.x / VELOX_WARP, lane = threadIdx.x % VELOX_WARP;
    if (i >= v_dim) return;
    gdn_rows<PER>(qkv, ab, a_log, dt_bias, state_in, state, out, tokens, k_heads, v_heads, k_dim, v_dim, h, i, lane);
}

// gdn_replay: per-layer buffers, passed by value as a kernel parameter (3.6 KB; graph launches
// capture it with the node).
constexpr int kGdnReplayMaxLayers = 64;
struct GdnReplayTable {
    const float* qkv[kGdnReplayMaxLayers];
    const float* conv_out[kGdnReplayMaxLayers];
    const float* ab[kGdnReplayMaxLayers];
    const float* a_log[kGdnReplayMaxLayers];
    const float* dt_bias[kGdnReplayMaxLayers];
    float* conv_state[kGdnReplayMaxLayers];
    float* ssm[kGdnReplayMaxLayers];
};

// Grid: heads x row groups x layers; the recurrent state advances in place, no output.
template <int PER>
__global__ void gdn_replay_rows_kernel(const __grid_constant__ GdnReplayTable t, int tokens, int k_heads, int v_heads,
                                       int k_dim, int v_dim) {
    const int warps = blockDim.x / VELOX_WARP, l = blockIdx.z;
    const int h = blockIdx.x, i = blockIdx.y * warps + threadIdx.x / VELOX_WARP, lane = threadIdx.x % VELOX_WARP;
    if (i >= v_dim) return;
    gdn_rows<PER>(t.conv_out[l], t.ab[l], t.a_log[l], t.dt_bias[l], t.ssm[l], t.ssm[l], nullptr, tokens,
                  k_heads, v_heads, k_dim, v_dim, h, i, lane);
}

// Grid: channel blocks x layers. New conv state row j is row tokens + j of [old state; inputs];
// ascending j reads old rows at or above the ones it overwrites.
__global__ void gdn_replay_conv_kernel(const __grid_constant__ GdnReplayTable t, int tokens, int channels, int width) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, l = blockIdx.y;
    if (c >= channels) return;
    float* st = t.conv_state[l];
    const float* x = t.qkv[l];
    for (int j = 0; j < width - 1; ++j) {
        const int r = tokens + j;
        st[static_cast<size_t>(j) * channels + c] = r < width - 1 ? st[static_cast<size_t>(r) * channels + c]
                                                                  : x[static_cast<size_t>(r - (width - 1)) * channels + c];
    }
}


// Chunkwise-parallel Gated DeltaNet for prefill (the chunked delta rule in WY form). Within a chunk
// of C tokens let G be the in-chunk cumulative log decay and gamma = exp(G). The per-token deltas
// D_r = beta_r (v_r - S'_r k_r) (S'_r = decayed state before token r) solve (I + A) D = R with
//   A[r][s] = beta_r exp(G_r - G_s) k_r.k_s (s < r),   R_r = beta_r (v_r - gamma_r S0 k_r),
// so D = T R with T = (I + A)^-1, and then
//   o_r = gamma_r S0 q_r + sum_{s<=r} exp(G_r - G_s) (q_r.k_s) D_s
//   S_C = gamma_{C-1} S0 + sum_s exp(G_{C-1} - G_s) D_s k_s^T.
// The key-side quantities (gates, T = (I + A)^-1, P) do not depend on the state, so a prepare pass
// computes them for every (head, chunk) in parallel into scratch; the recurrence then owns head h
// and kGdnVt state rows (kept in registers across chunks; chunks run in order) and only does the
// state-dependent products.
constexpr int kGdnChunk = 16;
constexpr int kGdnVt = 16;
constexpr int kGdnThreads = 128;
// Below this many tokens the per-row recurrence is as fast (decode and MTP verify stay on it).
constexpr int kGdnChunkMinTokens = 16;

// Reduce-scatter over the 8 lanes of an aligned lane group: on return v[0] of lane g (g = lane % 8)
// holds the group sum of v[g].
__device__ __forceinline__ void gdn_reduce_scatter8(float (&v)[8], int g) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const bool hi = g & 4;
        const float send = hi ? v[k] : v[k + 4], keep = hi ? v[k + 4] : v[k];
        v[k] = keep + __shfl_xor_sync(0xffffffffu, send, 4);
    }
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        const bool hi = g & 2;
        const float send = hi ? v[k] : v[k + 2], keep = hi ? v[k + 2] : v[k];
        v[k] = keep + __shfl_xor_sync(0xffffffffu, send, 2);
    }
    const bool hi = g & 1;
    const float send = hi ? v[0] : v[1], keep = hi ? v[1] : v[0];
    v[0] = keep + __shfl_xor_sync(0xffffffffu, send, 1);
}

// Per chunk in scratch: T[C][C] (row-major), P[C][C], gamma[C], decay_last[C], beta[C].
constexpr int kGdnPrep = 2 * kGdnChunk * kGdnChunk + 3 * kGdnChunk;

// One block per (value head, chunk): L2-normalized q/k of the chunk, gates and their scan,
// A[r][s] = beta_r exp(G_r - G_s) k_r.k_s (s < r), P[r][s] = exp(G_r - G_s) q_r.k_s (s <= r), and
// T = (I + A)^-1 by forward substitution.
template <int DK>
__global__ void __launch_bounds__(kGdnThreads)
    gated_delta_net_prepare_kernel(const float* __restrict__ qkv, const float* __restrict__ ab,
                                   const float* __restrict__ a_log, const float* __restrict__ dt_bias,
                                   float* __restrict__ prep, int tokens, int k_heads, int v_heads, int v_dim) {
    constexpr int C = kGdnChunk, KS = DK + 4, AS = C + 4;
    __shared__ __align__(16) float ks[C * KS], qs[C * KS];
    __shared__ __align__(16) float at[C * AS];  // A transposed: at[s][r] = A[r][s]
    __shared__ float gs[C], betas[C];
    const int h = blockIdx.x, ci = blockIdx.y, t0 = ci * C, tid = threadIdx.x;
    const int lane = tid % VELOX_WARP, warp = tid / VELOX_WARP;
    const int kh = h / (v_heads / k_heads);
    const int n = tokens - t0 < C ? tokens - t0 : C;
    const size_t width = static_cast<size_t>(2 * k_heads * DK + v_heads * v_dim);
    float* out = prep + (static_cast<size_t>(h) * gridDim.y + ci) * kGdnPrep;

    // Rows: 8 threads per chunk row, each 4-wide pieces at dc * 4 + 32 j.
    {
        constexpr int NJ = DK / 32;
        const int r = tid / 8, dc = tid % 8;
        float q[NJ][4], k[NJ][4], qq = 0.0f, kk = 0.0f;
        const float* row = qkv + static_cast<size_t>(t0 + r) * width;
#pragma unroll
        for (int j = 0; j < NJ; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                q[j][e] = r < n ? row[kh * DK + dc * 4 + 32 * j + e] : 0.0f;
                k[j][e] = r < n ? row[(k_heads + kh) * DK + dc * 4 + 32 * j + e] : 0.0f;
                qq += q[j][e] * q[j][e];
                kk += k[j][e] * k[j][e];
            }
#pragma unroll
        for (int o = 1; o < 8; o <<= 1) {
            qq += __shfl_xor_sync(0xffffffffu, qq, o);
            kk += __shfl_xor_sync(0xffffffffu, kk, o);
        }
        const float qn = rsqrtf(qq + 1e-6f), kn = rsqrtf(kk + 1e-6f);
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            *reinterpret_cast<float4*>(&qs[r * KS + dc * 4 + 32 * j]) =
                make_float4(q[j][0] * qn, q[j][1] * qn, q[j][2] * qn, q[j][3] * qn);
            *reinterpret_cast<float4*>(&ks[r * KS + dc * 4 + 32 * j]) =
                make_float4(k[j][0] * kn, k[j][1] * kn, k[j][2] * kn, k[j][3] * kn);
        }
    }
    // Gates (padded rows: g = 0, beta = 0, so they change nothing) and their in-chunk scan.
    if (warp == 0) {
        float g = 0.0f, beta = 0.0f;
        if (lane < n) {
            const float* abr = ab + static_cast<size_t>(t0 + lane) * 2 * v_heads;
            const float a = abr[h] + dt_bias[h];
            g = -expf(a_log[h]) * (a > 20.0f ? a : log1pf(expf(a)));
            beta = 1.0f / (1.0f + expf(-abr[v_heads + h]));
        }
#pragma unroll
        for (int o = 1; o < VELOX_WARP; o <<= 1) {
            const float u = __shfl_up_sync(0xffffffffu, g, o);
            if (lane >= o) g += u;
        }
        const float last = __shfl_sync(0xffffffffu, g, C - 1);
        if (lane < C) {
            gs[lane] = g;
            betas[lane] = beta;
            out[2 * C * C + lane] = expf(g);
            out[2 * C * C + C + lane] = expf(last - g);
            out[2 * C * C + 2 * C + lane] = beta;
        }
    }
    __syncthreads();

    // A and P: one (r, s) pair per thread and step.
    for (int idx = tid; idx < C * C; idx += kGdnThreads) {
        const int r = idx / C, c = idx % C;
        float kk = 0.0f, qk = 0.0f;
        if (c <= r) {
#pragma unroll 8
            for (int d = 0; d < DK; d += 4) {
                const float4 kc = *reinterpret_cast<const float4*>(&ks[c * KS + d]);
                const float4 kr = *reinterpret_cast<const float4*>(&ks[r * KS + d]);
                const float4 qr = *reinterpret_cast<const float4*>(&qs[r * KS + d]);
                kk += kc.x * kr.x + kc.y * kr.y + kc.z * kr.z + kc.w * kr.w;
                qk += kc.x * qr.x + kc.y * qr.y + kc.z * qr.z + kc.w * qr.w;
            }
        }
        const float dec = c <= r ? expf(gs[r] - gs[c]) : 0.0f;
        at[c * AS + r] = c < r ? betas[r] * dec * kk : 0.0f;
        out[C * C + r * C + c] = dec * qk;
    }
    __syncthreads();

    // T = (I + A)^-1; lane j owns column j.
    if (warp == 0) {
        float tc[C];
#pragma unroll
        for (int r = 0; r < C; ++r) tc[r] = r == lane ? 1.0f : 0.0f;
#pragma unroll
        for (int c = 0; c < C - 1; ++c) {
#pragma unroll
            for (int r = (c + 1) & ~3; r < C; r += 4) {
                const float4 a4 = *reinterpret_cast<const float4*>(&at[c * AS + r]);
                tc[r] -= a4.x * tc[c];
                tc[r + 1] -= a4.y * tc[c];
                tc[r + 2] -= a4.z * tc[c];
                tc[r + 3] -= a4.w * tc[c];
            }
        }
#pragma unroll
        for (int r = 0; r < C; ++r)
            if (lane < C) out[r * C + lane] = tc[r];
    }
}

// The state-dependent part, chunks in order, from the prepare pass's per-chunk matrices.
template <int DK>
__global__ void __launch_bounds__(kGdnThreads, 3)
    gated_delta_net_chunk_kernel(const float* __restrict__ qkv, const float* __restrict__ prep,
                                 float* __restrict__ state, float* __restrict__ out, int tokens, int k_heads,
                                 int v_heads, int v_dim) {
    constexpr int C = kGdnChunk, VT = kGdnVt, KS = DK + 4, PS = C + 4, RS = VT + 1, NJ = DK / 32;
    constexpr int NV = C * VT / kGdnThreads;
    static_assert(kGdnThreads == VT * 8 && kGdnThreads == C * 8, "8 threads per state row and per chunk row");
    __shared__ __align__(16) float ks[C * KS];  // normalized keys
    __shared__ __align__(16) float qs[C * KS];  // normalized queries
    __shared__ __align__(16) float ts[C * PS];  // T
    __shared__ __align__(16) float ps[C * PS];  // P
    __shared__ float rs[C * RS];                // V, then R in place
    __shared__ float ds[C * RS];                // D
    __shared__ float gam[C], decay_last[C], betas[C];

    const int h = blockIdx.x, i0 = blockIdx.y * VT, tid = threadIdx.x;
    const int kh = h / (v_heads / k_heads);
    const int chunks = (tokens + C - 1) / C;
    const size_t width = static_cast<size_t>(2 * k_heads * DK + v_heads * v_dim);
    const float inv_sqrt_dk = rsqrtf(static_cast<float>(DK));

    // Thread (si, dc) owns state row si (and chunk row si for loads) over columns
    // dc * 4 + 32 * j + e; its 8-thread group is lane-aligned within the warp.
    const int si = tid / 8, dc = tid % 8;
    const bool row_ok = i0 + si < v_dim;
    float* row_state = state + (static_cast<size_t>(h) * v_dim + i0 + si) * DK;
    float s[NJ][4];
#pragma unroll
    for (int j = 0; j < NJ; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) s[j][e] = row_ok ? row_state[dc * 4 + 32 * j + e] : 0.0f;

    // Raw inputs of a chunk, loaded into registers while the previous chunk finishes.
    float rq[NJ][4], rk[NJ][4], rv[NV], rt[2], rp[2], rg = 0.0f;
    auto load = [&](int ci) {
        const int t0 = ci * C, n = tokens - t0 < C ? tokens - t0 : C;
        const float* row = qkv + static_cast<size_t>(t0 + si) * width;
#pragma unroll
        for (int j = 0; j < NJ; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                rq[j][e] = si < n ? row[kh * DK + dc * 4 + 32 * j + e] : 0.0f;
                rk[j][e] = si < n ? row[(k_heads + kh) * DK + dc * 4 + 32 * j + e] : 0.0f;
            }
#pragma unroll
        for (int m = 0; m < NV; ++m) {
            const int e = tid + kGdnThreads * m, c = e / VT, i = e % VT;
            const float* vrow = qkv + static_cast<size_t>(t0 + c) * width;
            rv[m] = c < n && i0 + i < v_dim ? vrow[2 * k_heads * DK + h * v_dim + i0 + i] : 0.0f;
        }
        const float* pr = prep + (static_cast<size_t>(h) * chunks + ci) * kGdnPrep;
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            rt[m] = pr[tid + kGdnThreads * m];
            rp[m] = pr[C * C + tid + kGdnThreads * m];
        }
        if (tid < 3 * C) rg = pr[2 * C * C + tid];
    };
    load(0);

    for (int ci = 0; ci < chunks; ++ci) {
        const int t0 = ci * C, n = tokens - t0 < C ? tokens - t0 : C;
        // L2-normalize q and k of chunk row si (8-lane group reduction) and publish the chunk.
        {
            float qq = 0.0f, kk = 0.0f;
#pragma unroll
            for (int j = 0; j < NJ; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    qq += rq[j][e] * rq[j][e];
                    kk += rk[j][e] * rk[j][e];
                }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) {
                qq += __shfl_xor_sync(0xffffffffu, qq, o);
                kk += __shfl_xor_sync(0xffffffffu, kk, o);
            }
            const float qn = rsqrtf(qq + 1e-6f), kn = rsqrtf(kk + 1e-6f);
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
                *reinterpret_cast<float4*>(&qs[si * KS + dc * 4 + 32 * j]) =
                    make_float4(rq[j][0] * qn, rq[j][1] * qn, rq[j][2] * qn, rq[j][3] * qn);
                *reinterpret_cast<float4*>(&ks[si * KS + dc * 4 + 32 * j]) =
                    make_float4(rk[j][0] * kn, rk[j][1] * kn, rk[j][2] * kn, rk[j][3] * kn);
            }
#pragma unroll
            for (int m = 0; m < NV; ++m) {
                const int e = tid + kGdnThreads * m;
                rs[(e / VT) * RS + e % VT] = rv[m];
            }
#pragma unroll
            for (int m = 0; m < 2; ++m) {
                const int e = tid + kGdnThreads * m;
                ts[(e / C) * PS + e % C] = rt[m];
                ps[(e / C) * PS + e % C] = rp[m];
            }
            if (tid < C) gam[tid] = rg;
            else if (tid < 2 * C) decay_last[tid - C] = rg;
            else if (tid < 3 * C) betas[tid - 2 * C] = rg;
        }
        __syncthreads();
        if (ci + 1 < chunks) load(ci + 1);

        // x = S0[si].k_c, y = S0[si].q_c for chunk rows c, 8 at a time (lane dc ends up owning
        // c = 8 g + dc): R[c][si] = beta_c (v_c - gamma_c x), and gamma_c y kept for o.
        float ys[C / 8];
#pragma unroll
        for (int g = 0; g < C / 8; ++g) {
            float x[8], y[8];
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const int c = 8 * g + u;
                x[u] = y[u] = 0.0f;
#pragma unroll
                for (int j = 0; j < NJ; ++j) {
                    const float4 kv = *reinterpret_cast<const float4*>(&ks[c * KS + dc * 4 + 32 * j]);
                    const float4 qv = *reinterpret_cast<const float4*>(&qs[c * KS + dc * 4 + 32 * j]);
                    x[u] += kv.x * s[j][0] + kv.y * s[j][1] + kv.z * s[j][2] + kv.w * s[j][3];
                    y[u] += qv.x * s[j][0] + qv.y * s[j][1] + qv.z * s[j][2] + qv.w * s[j][3];
                }
            }
            gdn_reduce_scatter8(x, dc);
            gdn_reduce_scatter8(y, dc);
            const int c = 8 * g + dc;
            rs[c * RS + si] = betas[c] * (rs[c * RS + si] - gam[c] * x[0]);
            ys[g] = gam[c] * y[0];
        }
        __syncthreads();

        // D = T R, o = gamma S0 q + P D, and the state update for state row si (T and P are lower
        // triangular with explicit zeros).
        {
            float acc[C / 8];
#pragma unroll
            for (int g = 0; g < C / 8; ++g) acc[g] = 0.0f;
#pragma unroll 2
            for (int u = 0; u < C; u += 4) {
                const float r0 = rs[u * RS + si], r1 = rs[(u + 1) * RS + si], r2 = rs[(u + 2) * RS + si],
                            r3 = rs[(u + 3) * RS + si];
#pragma unroll
                for (int g = 0; g < C / 8; ++g) {
                    const float4 t4 = *reinterpret_cast<const float4*>(&ts[(8 * g + dc) * PS + u]);
                    acc[g] += t4.x * r0 + t4.y * r1 + t4.z * r2 + t4.w * r3;
                }
            }
#pragma unroll
            for (int g = 0; g < C / 8; ++g) ds[(8 * g + dc) * RS + si] = acc[g];
        }
        __syncthreads();
        {
            float acc[C / 8];
#pragma unroll
            for (int g = 0; g < C / 8; ++g) acc[g] = ys[g];
#pragma unroll 2
            for (int u = 0; u < C; u += 4) {
                const float d0 = ds[u * RS + si], d1 = ds[(u + 1) * RS + si], d2 = ds[(u + 2) * RS + si],
                            d3 = ds[(u + 3) * RS + si];
#pragma unroll
                for (int g = 0; g < C / 8; ++g) {
                    const float4 p4 = *reinterpret_cast<const float4*>(&ps[(8 * g + dc) * PS + u]);
                    acc[g] += p4.x * d0 + p4.y * d1 + p4.z * d2 + p4.w * d3;
                }
            }
#pragma unroll
            for (int g = 0; g < C / 8; ++g) {
                const int c = 8 * g + dc;
                if (c < n && row_ok)
                    out[(static_cast<size_t>(t0 + c) * v_heads + h) * v_dim + i0 + si] = acc[g] * inv_sqrt_dk;
            }
        }
        {
            const float gl = gam[C - 1];
#pragma unroll
            for (int j = 0; j < NJ; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) s[j][e] *= gl;
            for (int c = 0; c < n; ++c) {
                const float dv = ds[c * RS + si] * decay_last[c];
#pragma unroll
                for (int j = 0; j < NJ; ++j) {
                    const float4 kv = *reinterpret_cast<const float4*>(&ks[c * KS + dc * 4 + 32 * j]);
                    s[j][0] += dv * kv.x;
                    s[j][1] += dv * kv.y;
                    s[j][2] += dv * kv.z;
                    s[j][3] += dv * kv.w;
                }
            }
        }
        __syncthreads();
    }
    if (row_ok) {
#pragma unroll
        for (int j = 0; j < NJ; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) row_state[dc * 4 + 32 * j + e] = s[j][e];
    }
}

// Small-m linear for FP8-row and NVFP4: one warp per output row, codes decoded a group at a time
// (16 bytes of FP8 codes, or one 16-element NVFP4 group sharing a scale) and applied to all m
// rows, so each weight is read once.
constexpr int kGemvMaxM = 8;

__device__ __forceinline__ float fp8_lut(uint8_t b) { return e4m3fn_to_f32(b); }

template <int FMT>
__global__ void linear_gemv_kernel(const float* __restrict__ x, WDesc w, long long row_begin, int rows, int m,
                                   float* __restrict__ y) {
    const int r = blockIdx.x * (blockDim.x / VELOX_WARP) + threadIdx.x / VELOX_WARP;
    const int lane = threadIdx.x % VELOX_WARP;
    if (r >= rows) return;
    const long long row = row_begin + r, k = w.k;
    float acc[kGemvMaxM];
#pragma unroll
    for (int i = 0; i < kGemvMaxM; ++i) acc[i] = 0.0f;
    if (FMT == VELOX_FMT_FP8_ROW_BF16) {
        const uint8_t* codes = w.base + row * k;
        for (long long c0 = lane * 16LL; c0 < k; c0 += 16LL * VELOX_WARP) {
            const int n = static_cast<int>(k - c0 < 16 ? k - c0 : 16);
            float wv[16];
#pragma unroll
            for (int j = 0; j < 16; ++j) wv[j] = j < n ? fp8_lut(codes[c0 + j]) : 0.0f;
#pragma unroll
            for (int i = 0; i < kGemvMaxM; ++i) {
                if (i < m) {
                    const float* xr = x + static_cast<size_t>(i) * k + c0;
                    float s = 0.0f;
#pragma unroll
                    for (int j = 0; j < 16; ++j)
                        if (j < n) s += wv[j] * xr[j];
                    acc[i] += s;
                }
            }
        }
        const uint8_t* sp = w.base + align256(w.n * k) + 2 * row;
        const float scale = bf16_to_f32(static_cast<uint16_t>(sp[0] | (sp[1] << 8)));
#pragma unroll
        for (int i = 0; i < kGemvMaxM; ++i) acc[i] *= scale;
    } else {  // NVFP4: k % 64 == 0, so groups of 16 never straddle the row end
        const uint8_t* codes = w.base + row * k / 2;
        const long long scale_plane = align256(w.n * k / 2);
        const float inv_div = 1.0f / *reinterpret_cast<const float*>(w.base + scale_plane + w.n * k / 16);
        for (long long g = lane; g < k / 16; g += VELOX_WARP) {
            const float scale = e4m3fn_to_f32(w.base[scale_plane + nvfp4_scale_offset(row, g, k)]);
            float wv[16];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint8_t byte = codes[g * 8 + j];
                wv[2 * j] = e2m1_to_f32(byte & 0xf) * scale;
                wv[2 * j + 1] = e2m1_to_f32(byte >> 4) * scale;
            }
#pragma unroll
            for (int i = 0; i < kGemvMaxM; ++i) {
                if (i < m) {
                    const float* xr = x + static_cast<size_t>(i) * k + g * 16;
                    float s = 0.0f;
#pragma unroll
                    for (int j = 0; j < 16; ++j) s += wv[j] * xr[j];
                    acc[i] += s;
                }
            }
        }
#pragma unroll
        for (int i = 0; i < kGemvMaxM; ++i) acc[i] *= inv_div;
    }
#pragma unroll
    for (int i = 0; i < kGemvMaxM; ++i) {
        const float s = velox_warp_sum(acc[i]);
        if (lane == 0 && i < m) y[static_cast<size_t>(i) * rows + r] = s;
    }
}

float* linear_split_scratch(size_t floats) {
    static VeloxScratch s;
    return static_cast<float*>(s.get(floats * sizeof(float)));
}

__global__ void sum_splits_kernel(const float* __restrict__ part, int splits, long long n, float* __restrict__ y) {
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        float s = 0.0f;
        for (int z = 0; z < splits; ++z) s += part[z * n + i];
        y[i] = s;
    }
}

// Vectorized small-m linear for FP8-row and NVFP4. Each warp owns R adjacent weight rows and
// streams them in 16-byte loads (16 FP8 or 32 NVFP4 codes per lane per row), so one x load feeds
// R rows and several loads per lane are in flight. Requires 16-byte aligned rows: k % 16 == 0
// for FP8; NVFP4 always qualifies (k % 64 == 0).
__device__ __forceinline__ uint4 velox_ld_stream(const void* p) {
    return __ldg(reinterpret_cast<const uint4*>(p));
}

// E4M3FN byte -> f32 by moving sign, exponent and mantissa into place and rebiasing (2^120 is
// exact, and covers the subnormals). NaN codes never occur in valid weights.
__device__ __forceinline__ float e4m3_bits(uint32_t word, int byte) {
    const uint32_t b = (word >> (8 * byte)) & 0xffu;
    return __uint_as_float(((b & 0x80u) << 24) | ((b & 0x7fu) << 20)) * 0x1p120f;
}

// The two E2M1 codes of the low byte of `t`, times 2^-14: each nibble's magnitude bits go to f16
// bits [11:9] and its sign to bit 15, so exponent e becomes 2^(e-15) and e = 0 a subnormal
// m * 2^-15, i.e. code value * 2^-14 exactly. Callers fold 2^14 into the block scale.
__device__ __forceinline__ float2 e2m1x2_scaled(uint32_t t) {
    const uint32_t h = ((t & 0x7u) << 9) | ((t & 0x8u) << 12) | ((t & 0x70u) << 21) | ((t & 0x80u) << 24);
    return __half22float2(*reinterpret_cast<const __half2*>(&h));
}

// KW warps share one row group and split its K (KW = 8 for matrices with few rows, where one warp
// per row would leave most SMs idle); their sums meet in shared memory.
template <int FMT, int R, int M, int KW>
__global__ void __launch_bounds__(256)
linear_gemv_vec_kernel(const float* __restrict__ x, WDesc w, long long row_begin, int rows, int m,
                       float* __restrict__ y) {
    __shared__ float red[KW][R][M];
    const int wid = blockIdx.x * (blockDim.x / VELOX_WARP) + threadIdx.x / VELOX_WARP;
    const int kw = (threadIdx.x / VELOX_WARP) % KW;
    const int lane = threadIdx.x % VELOX_WARP;
    const int r0 = (wid / KW) * R;
    if (r0 >= rows) return;  // whole row groups only, so KW > 1 blocks return together
    const long long k = w.k;
    // Groupwise Q4 (group 64) / Q8 (group 32), row_split_k128_v1: a row's base plane is gpr
    // groups of 32 bytes over K padded to 128; one 16-byte chunk is half a group either way.
    constexpr bool kRowSplit = FMT == VELOX_FMT_Q4 || FMT == VELOX_FMT_Q8;
    const long long gpr = (k + 127) / 128 * 128 / (FMT == VELOX_FMT_Q8 ? 32 : 64);
    const long long row_bytes = kRowSplit ? gpr * 32
                                : FMT == VELOX_FMT_NVFP4 ? k / 2
                                : FMT == VELOX_FMT_BF16  ? 2 * k
                                                         : k;
    constexpr int kElems = FMT == VELOX_FMT_NVFP4 || FMT == VELOX_FMT_Q4 ? 32 : FMT == VELOX_FMT_BF16 ? 8 : 16;
    // Only real columns are visited (the launcher requires k % kElems == 0 for row-split).
    const long long chunks = kRowSplit ? k / kElems : row_bytes / 16;
    const uint8_t* rp[R];
    bool live[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        live[r] = r0 + r < rows;
        rp[r] = w.base + (row_begin + (live[r] ? r0 + r : r0)) * row_bytes;
    }
    float acc[R][M];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int i = 0; i < M; ++i) acc[r][i] = 0.0f;
    const long long scale_plane = align256(w.n * k / 2);
    const uint8_t* sbase[R];
#pragma unroll
    for (int r = 0; r < R; ++r) sbase[r] = w.base + scale_plane + nvfp4_scale_offset(row_begin + r0 + (live[r] ? r : 0), 0, k);
    // Row-split FP16 group scales follow the base plane (no high plane for Q4/Q8).
    const uint16_t* gscale[R];
#pragma unroll
    for (int r = 0; r < R; ++r)
        gscale[r] = reinterpret_cast<const uint16_t*>(w.base + align256(w.n * gpr * 32)) + (row_begin + r0 + (live[r] ? r : 0)) * gpr;

    for (long long c = kw * VELOX_WARP + lane; c < chunks; c += VELOX_WARP * KW) {
        uint4 wv[R];
#pragma unroll
        for (int r = 0; r < R; ++r) wv[r] = velox_ld_stream(rp[r] + c * 16);
        float sc[R][2];
        if (FMT == VELOX_FMT_NVFP4) {
            // Groups 2c and 2c+1 are adjacent bytes of one 4-byte scale word: 512-byte tiles per
            // 64 columns, so chunk c sits at (c / 2) * 512 + (c % 2) * 2 past the row's base.
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint32_t s2 = *reinterpret_cast<const uint16_t*>(sbase[r] + (c >> 1) * 512 + (c & 1) * 2);
                sc[r][0] = e4m3_bits(s2, 0) * 16384.0f;  // e2m1x2_scaled is 2^-14 times the code
                sc[r][1] = e4m3_bits(s2, 1) * 16384.0f;
            }
        }
        if (kRowSplit) {
#pragma unroll
            for (int r = 0; r < R; ++r) sc[r][0] = __half2float(__ushort_as_half(gscale[r][c >> 1]));
        }
#pragma unroll
        for (int i = 0; i < M; ++i) {
            if (i >= m) break;
            const float4* xp = reinterpret_cast<const float4*>(x + static_cast<size_t>(i) * k + c * kElems);
            float xv[kElems];
#pragma unroll
            for (int q = 0; q < kElems / 4; ++q) {
                const float4 t = __ldg(xp + q);
                xv[4 * q] = t.x, xv[4 * q + 1] = t.y, xv[4 * q + 2] = t.z, xv[4 * q + 3] = t.w;
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint32_t words[4] = {wv[r].x, wv[r].y, wv[r].z, wv[r].w};
                if (FMT == VELOX_FMT_NVFP4) {
                    float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
                    for (int b = 0; b < 16; ++b) {
                        const float2 v = e2m1x2_scaled(words[b / 4] >> (8 * (b % 4)));
                        const float s = v.x * xv[2 * b] + v.y * xv[2 * b + 1];
                        if (b < 8) s0 += s; else s1 += s;
                    }
                    acc[r][i] += s0 * sc[r][0] + s1 * sc[r][1];
                } else if (FMT == VELOX_FMT_Q4) {
                    float s = 0.0f;
#pragma unroll
                    for (int b = 0; b < 16; ++b) {
                        const uint32_t byte = (words[b / 4] >> (8 * (b % 4))) & 0xffu;
                        const int lo = static_cast<int>(byte & 0xfu) - (byte & 0x8u ? 16 : 0);
                        const int hi = static_cast<int>(byte >> 4) - (byte & 0x80u ? 16 : 0);
                        s += static_cast<float>(lo) * xv[2 * b] + static_cast<float>(hi) * xv[2 * b + 1];
                    }
                    acc[r][i] += s * sc[r][0];
                } else if (FMT == VELOX_FMT_Q8) {
                    float s = 0.0f;
#pragma unroll
                    for (int b = 0; b < 16; ++b)
                        s += static_cast<float>(static_cast<int8_t>((words[b / 4] >> (8 * (b % 4))) & 0xffu)) * xv[b];
                    acc[r][i] += s * sc[r][0];
                } else if (FMT == VELOX_FMT_BF16) {
                    float s = 0.0f;
#pragma unroll
                    for (int b = 0; b < 4; ++b)
                        s += __uint_as_float(words[b] << 16) * xv[2 * b] + __uint_as_float(words[b] & 0xffff0000u) * xv[2 * b + 1];
                    acc[r][i] += s;
                } else {
                    float s = 0.0f;
#pragma unroll
                    for (int b = 0; b < 16; ++b) s += e4m3_bits(words[b / 4], b % 4) * xv[b];
                    acc[r][i] += s;
                }
            }
        }
    }
    if (KW > 1) {
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int i = 0; i < M; ++i) {
                const float s = velox_warp_sum(acc[r][i]);
                if (lane == 0) red[kw][r][i] = s;
            }
        __syncthreads();
        if (kw != 0) return;
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int i = 0; i < M; ++i) {
                float s = 0.0f;
                for (int j = 0; j < KW; ++j) s += red[j][r][i];
                acc[r][i] = lane == 0 ? s : 0.0f;  // the warp sum below then returns s
            }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (!live[r]) continue;
        const long long row = row_begin + r0 + r;
        float post = 1.0f;
        if (FMT == VELOX_FMT_BF16 || kRowSplit) {
        } else if (FMT == VELOX_FMT_NVFP4) {
            post = 1.0f / *reinterpret_cast<const float*>(w.base + scale_plane + w.n * k / 16);
        } else {
            const uint8_t* sp = w.base + align256(w.n * k) + 2 * row;
            post = bf16_to_f32(static_cast<uint16_t>(sp[0] | (sp[1] << 8)));
        }
#pragma unroll
        for (int i = 0; i < M; ++i) {
            const float s = velox_warp_sum(acc[r][i]);
            if (lane == 0 && i < m) y[static_cast<size_t>(i) * rows + r0 + r] = s * post;
        }
    }
}

template <int FMT, int M>
int launch_gemv_vec(const float* x, const WDesc& d, long long row_begin, int rows, int m, float* y) {
    constexpr int kWarps = 8;
    // Few rows: a whole block (8 warps) per row, splitting K. Otherwise prefer more rows per warp
    // (more x reuse) while keeping enough warps to fill the GPU.
    if (rows < 2048) {
        linear_gemv_vec_kernel<FMT, 1, M, kWarps><<<rows, kWarps * VELOX_WARP>>>(x, d, row_begin, rows, m, y);
        return velox_launch_status();
    }
    const int r = rows >= 4 * 4096 ? 4 : rows >= 2 * 2048 ? 2 : 1;
    const int warps = (rows + r - 1) / r;
    const dim3 grid((warps + kWarps - 1) / kWarps);
    if (r == 4) linear_gemv_vec_kernel<FMT, 4, M, 1><<<grid, kWarps * VELOX_WARP>>>(x, d, row_begin, rows, m, y);
    else if (r == 2) linear_gemv_vec_kernel<FMT, 2, M, 1><<<grid, kWarps * VELOX_WARP>>>(x, d, row_begin, rows, m, y);
    else linear_gemv_vec_kernel<FMT, 1, M, 1><<<grid, kWarps * VELOX_WARP>>>(x, d, row_begin, rows, m, y);
    return velox_launch_status();
}

template <int FMT>
int launch_gemv_vec_m(const float* x, const WDesc& d, long long row_begin, int rows, int m, float* y) {
    if (m == 1) return launch_gemv_vec<FMT, 1>(x, d, row_begin, rows, m, y);
    if (m == 2) return launch_gemv_vec<FMT, 2>(x, d, row_begin, rows, m, y);
    if (m <= 4) return launch_gemv_vec<FMT, 4>(x, d, row_begin, rows, m, y);
    return launch_gemv_vec<FMT, 8>(x, d, row_begin, rows, m, y);
}

}  // namespace

namespace {
// Few output rows (e.g. a 96-row gate projection) over a few tokens, bf16 weights, exact f32
// activations: a block computes kSnRows rows for every token, its threads splitting K, so the
// activations are read once per row group rather than once per row. Fixed-order reductions.
constexpr int kSnRows = 4, kSnThreads = 512, kSnMaxM = 16;

template <int M>
__global__ void __launch_bounds__(kSnThreads) linear_small_n_bf16_kernel(const float* __restrict__ x,
                                                                      const uint16_t* __restrict__ w, long long k,
                                                                      long long row_begin, int rows, int m,
                                                                      float* __restrict__ y) {
    __shared__ float red[kSnThreads / VELOX_WARP][kSnRows][M];
    const int r0 = blockIdx.x * kSnRows;
    float acc[kSnRows][M];
#pragma unroll
    for (int r = 0; r < kSnRows; ++r)
#pragma unroll
        for (int t = 0; t < M; ++t) acc[r][t] = 0.0f;
    const uint16_t* wr[kSnRows];
#pragma unroll
    for (int r = 0; r < kSnRows; ++r) wr[r] = w + (row_begin + min(r0 + r, rows - 1)) * k;
    // Four consecutive K values per step: one 8-byte weight load per row, one float4 per token.
    for (long long c = threadIdx.x; c < k / 4; c += kSnThreads) {
        float4 xv[M];
#pragma unroll
        for (int t = 0; t < M; ++t)
            xv[t] = t < m ? reinterpret_cast<const float4*>(x + t * k)[c] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (int r = 0; r < kSnRows; ++r) {
            const uint2 b = reinterpret_cast<const uint2*>(wr[r])[c];
            const float w0 = __uint_as_float(b.x << 16), w1 = __uint_as_float(b.x & 0xffff0000u);
            const float w2 = __uint_as_float(b.y << 16), w3 = __uint_as_float(b.y & 0xffff0000u);
#pragma unroll
            for (int t = 0; t < M; ++t) acc[r][t] += w0 * xv[t].x + w1 * xv[t].y + w2 * xv[t].z + w3 * xv[t].w;
        }
    }
    const int warp = threadIdx.x / VELOX_WARP, lane = threadIdx.x % VELOX_WARP;
#pragma unroll
    for (int r = 0; r < kSnRows; ++r)
#pragma unroll
        for (int t = 0; t < M; ++t) {
            const float v = velox_warp_sum(acc[r][t]);
            if (lane == 0) red[warp][r][t] = v;
        }
    __syncthreads();
    for (int i = threadIdx.x; i < kSnRows * M; i += kSnThreads) {
        const int r = i / M, t = i % M;
        if (r0 + r >= rows || t >= m) continue;
        float s = 0.0f;
        for (int wi = 0; wi < kSnThreads / VELOX_WARP; ++wi) s += red[wi][r][t];
        y[static_cast<long long>(t) * rows + r0 + r] = s;
    }
}

template <int M>
int launch_small_n(const float* x, const uint8_t* w, long long k, long long row_begin, int rows, int m, float* y) {
    linear_small_n_bf16_kernel<M><<<(rows + kSnRows - 1) / kSnRows, kSnThreads>>>(
        x, reinterpret_cast<const uint16_t*>(w), k, row_begin, rows, m, y);
    return velox_launch_status();
}
}  // namespace

VELOX_API int velox_portable_linear(const float* x, const uint8_t* w, int format, long long n, long long k,
                                    long long row_begin, int rows, int m, float* y) {
    const WDesc d{w, format, n, k};
    if (format == VELOX_FMT_BF16 && m >= 2 && m <= kSnMaxM && rows <= 4096 && k % 4 == 0 &&
        reinterpret_cast<uintptr_t>(x) % 16 == 0 && reinterpret_cast<uintptr_t>(w) % 8 == 0) {
        if (m <= 2) return launch_small_n<2>(x, w, k, row_begin, rows, m, y);
        if (m <= 4) return launch_small_n<4>(x, w, k, row_begin, rows, m, y);
        if (m <= 8) return launch_small_n<8>(x, w, k, row_begin, rows, m, y);
        return launch_small_n<16>(x, w, k, row_begin, rows, m, y);
    }
    // x rows are read as float4, so k must keep them 16-byte aligned too.
    if (m <= kGemvMaxM && k % 16 == 0 && reinterpret_cast<uintptr_t>(x) % 16 == 0) {
        if (format == VELOX_FMT_NVFP4) return launch_gemv_vec_m<VELOX_FMT_NVFP4>(x, d, row_begin, rows, m, y);
        if (format == VELOX_FMT_FP8_ROW_BF16)
            return launch_gemv_vec_m<VELOX_FMT_FP8_ROW_BF16>(x, d, row_begin, rows, m, y);
        if (format == VELOX_FMT_BF16) return launch_gemv_vec_m<VELOX_FMT_BF16>(x, d, row_begin, rows, m, y);
        if (format == VELOX_FMT_Q4 && k % 32 == 0) return launch_gemv_vec_m<VELOX_FMT_Q4>(x, d, row_begin, rows, m, y);
        if (format == VELOX_FMT_Q8) return launch_gemv_vec_m<VELOX_FMT_Q8>(x, d, row_begin, rows, m, y);
    }
    if (m <= kGemvMaxM && (format == VELOX_FMT_FP8_ROW_BF16 || format == VELOX_FMT_NVFP4)) {
        constexpr int kWarps = 8;
        const dim3 grid((rows + kWarps - 1) / kWarps);
        if (format == VELOX_FMT_FP8_ROW_BF16)
            linear_gemv_kernel<VELOX_FMT_FP8_ROW_BF16><<<grid, kWarps * VELOX_WARP>>>(x, d, row_begin, rows, m, y);
        else
            linear_gemv_kernel<VELOX_FMT_NVFP4><<<grid, kWarps * VELOX_WARP>>>(x, d, row_begin, rows, m, y);
        return velox_launch_status();
    }
    // The tiled kernel needs enough output tiles to fill the GPU; skinny outputs (a few dozen rows,
    // e.g. GDN gate projections) go to the warp-per-output kernel instead.
    const long long tiles = static_cast<long long>((rows + kTile - 1) / kTile) * ((m + kTile - 1) / kTile);
    if (m >= 16 && (m + kTile - 1) / kTile <= 65535) {
        // Too few tiles to fill the GPU (skinny outputs such as GDN gate projections): split K into
        // slices written separately, then summed in a fixed order, so results stay deterministic.
        int splits = 1;
        while (tiles * splits < 256 && k / (splits * 2) >= 256) splits *= 2;
        const long long k_split = (k + splits - 1) / splits;
        const dim3 grid((rows + kTile - 1) / kTile, (m + kTile - 1) / kTile, splits);
        if (splits == 1) {
            linear_tiled_kernel<<<grid, 256>>>(x, d, row_begin, rows, m, y, k_split);
        } else {
            float* part = linear_split_scratch(static_cast<size_t>(splits) * m * rows);
            if (part == nullptr) return static_cast<int>(cudaErrorMemoryAllocation);
            linear_tiled_kernel<<<grid, 256>>>(x, d, row_begin, rows, m, part, k_split);
            const long long n = static_cast<long long>(m) * rows;
            sum_splits_kernel<<<velox_grid_1d(n, 256), 256>>>(part, splits, n, y);
        }
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

// state_in: the rolling state before the call; the advanced state goes to `state` (may be the same buffer).
VELOX_API int velox_portable_conv1d_silu(const float* x, const float* w, const float* state_in, float* state, float* out,
                                         int tokens, int channels, int width) {
    if (width > kMaxConvWidth) return VELOX_ERR_UNSUPPORTED;
    if (tokens <= kConvSmallTokens) {
        conv1d_silu_small_kernel<<<(channels + 127) / 128, 128>>>(x, w, state_in, state, out, tokens, channels, width);
        return velox_launch_status();
    }
    conv1d_silu_kernel<<<velox_grid_1d(static_cast<long long>(tokens) * channels, 256), 256>>>(x, w, state_in, out,
                                                                                                tokens, channels, width);
    conv1d_state_kernel<<<(channels + 255) / 256, 256>>>(x, state_in, state, tokens, channels, width);
    return velox_launch_status();
}

// state_in: the state before the call; the updated state goes to `state` (may be the same buffer).
VELOX_API int velox_portable_gated_delta_net(const float* qkv, const float* ab, const float* a_log,
                                             const float* dt_bias, const float* state_in, float* state, float* out,
                                             int tokens, int k_heads, int v_heads, int k_dim, int v_dim) {
    if (tokens >= kGdnChunkMinTokens && (k_dim == 32 || k_dim == 64 || k_dim == 128)) {
        if (state_in != state) {
            // The chunked kernel updates in place.
            const size_t n = static_cast<size_t>(v_heads) * v_dim * k_dim;
            if (cudaError_t e = cudaMemcpyAsync(state, state_in, n * sizeof(float), cudaMemcpyDeviceToDevice, 0))
                return static_cast<int>(e);
        }
        const int chunks = (tokens + kGdnChunk - 1) / kGdnChunk;
        static VeloxScratch scratch;
        float* prep = static_cast<float*>(scratch.get(static_cast<size_t>(v_heads) * chunks * kGdnPrep * sizeof(float)));
        if (prep == nullptr) return static_cast<int>(cudaErrorMemoryAllocation);
        const dim3 pgrid(v_heads, chunks), grid(v_heads, (v_dim + kGdnVt - 1) / kGdnVt), block(kGdnThreads);
#define VELOX_GDN_CHUNK(DK)                                                                                         \
    gated_delta_net_prepare_kernel<DK><<<pgrid, block>>>(qkv, ab, a_log, dt_bias, prep, tokens, k_heads, v_heads,   \
                                                         v_dim);                                                    \
    gated_delta_net_chunk_kernel<DK><<<grid, block>>>(qkv, prep, state, out, tokens, k_heads, v_heads, v_dim)
        if (k_dim == 32) {
            VELOX_GDN_CHUNK(32);
        } else if (k_dim == 64) {
            VELOX_GDN_CHUNK(64);
        } else {
            VELOX_GDN_CHUNK(128);
        }
#undef VELOX_GDN_CHUNK
        return velox_launch_status();
    }
    constexpr int kWarps = 4;
    const dim3 grid(v_heads, (v_dim + kWarps - 1) / kWarps), block(kWarps * VELOX_WARP);
#define VELOX_GDN(PER)                                                                                     \
    gated_delta_net_rows_kernel<PER><<<grid, block>>>(qkv, ab, a_log, dt_bias, state_in, state, out, tokens, k_heads, \
                                                      v_heads, k_dim, v_dim)
    if (k_dim <= 32) VELOX_GDN(1);
    else if (k_dim <= 64) VELOX_GDN(2);
    else if (k_dim <= 128) VELOX_GDN(4);
    else if (k_dim <= 256) VELOX_GDN(8);
    else return VELOX_ERR_UNSUPPORTED;
#undef VELOX_GDN
    return velox_launch_status();
}

// Pointer arrays hold `layers` entries each: qkv, conv_out, ab, a_log, dt_bias, conv_state, ssm.
VELOX_API int velox_portable_gdn_replay(const float* const* qkv, const float* const* conv_out, const float* const* ab,
                                        const float* const* a_log, const float* const* dt_bias,
                                        float* const* conv_state, float* const* ssm, int layers, int tokens,
                                        int channels, int width, int k_heads, int v_heads, int k_dim, int v_dim) {
    if (layers <= 0) return 0;
    if (layers > kGdnReplayMaxLayers || width > kMaxConvWidth || k_dim > 256) return VELOX_ERR_UNSUPPORTED;
    GdnReplayTable host{};
    for (int l = 0; l < layers; ++l) {
        host.qkv[l] = qkv[l];
        host.conv_out[l] = conv_out[l];
        host.ab[l] = ab[l];
        host.a_log[l] = a_log[l];
        host.dt_bias[l] = dt_bias[l];
        host.conv_state[l] = conv_state[l];
        host.ssm[l] = ssm[l];
    }
    const GdnReplayTable& t = host;
    gdn_replay_conv_kernel<<<dim3((channels + 255) / 256, layers), 256>>>(t, tokens, channels, width);
    constexpr int kWarps = 4;
    const dim3 grid(v_heads, (v_dim + kWarps - 1) / kWarps, layers), block(kWarps * VELOX_WARP);
#define VELOX_GDN_REPLAY(PER) \
    gdn_replay_rows_kernel<PER><<<grid, block>>>(t, tokens, k_heads, v_heads, k_dim, v_dim)
    if (k_dim <= 32) VELOX_GDN_REPLAY(1);
    else if (k_dim <= 64) VELOX_GDN_REPLAY(2);
    else if (k_dim <= 128) VELOX_GDN_REPLAY(4);
    else VELOX_GDN_REPLAY(8);
#undef VELOX_GDN_REPLAY
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
// One block per token: the token's angles (double, slow on consumer GPUs) are computed once per pair
// into shared memory, then every head applies them.
__global__ void rope_positions_kernel(float* __restrict__ x, const uint32_t* __restrict__ pos, int tokens, int heads,
                                      int head_dim, int rotary_dim, float theta, int axes, int layout) {
    extern __shared__ float cs[];  // [half] cos, then [half] sin
    const int half = rotary_dim / 2, t = blockIdx.x;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
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
        cs[i] = static_cast<float>(c);
        cs[half + i] = static_cast<float>(s);
    }
    __syncthreads();
    for (int p = threadIdx.x; p < heads * half; p += blockDim.x) {
        const int h = p / half, i = p % half;
        float* base = x + (static_cast<size_t>(t) * heads + h) * head_dim;
        const float a = base[i], b = base[i + half], c = cs[i], s = cs[half + i];
        base[i] = a * c - b * s;
        base[i + half] = a * s + b * c;
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
    if (tokens <= 0) return 0;
    rope_positions_kernel<<<tokens, 256, rotary_dim * sizeof(float)>>>(x, pos, tokens, heads, head_dim, rotary_dim,
                                                                       theta, axes, layout);
    return velox_launch_status();
}
