// Tuned kernels for the RTX 5060 Ti (GB206, 36 SMs, 448 GB/s GDDR7), from rtx-5090-32g. Shapes outside each kernel's
// domain return VELOX_ERR_UNSUPPORTED and the Rust side falls back to the portable kernel.
#include <math.h>
#include <stdint.h>

#include "velox_common.cuh"

namespace {

// ---- MatMul, small M (decode and tiny batches) --------------------------------------------
// y[m, n] = x[m, k] . w[n, k]^T for m <= MT. One warp per output column streams its weight row
// once with 16-byte loads and applies it to all m rows, so weights are read once per call
// instead of once per row. Weights bypass L1 (__ldcs, read once); x stays cached.
constexpr int kGemvWarps = 8;

template <int MT>
__global__ void __launch_bounds__(kGemvWarps * VELOX_WARP)
gemv_tile_kernel(const float4* __restrict__ x, const float4* __restrict__ w, float* __restrict__ y, int m, int n,
                 int k4) {
    const int col = blockIdx.x * kGemvWarps + threadIdx.x / VELOX_WARP;
    const int lane = threadIdx.x % VELOX_WARP;
    if (col >= n) return;
    const float4* wr = w + static_cast<size_t>(col) * k4;
    float acc[MT];
#pragma unroll
    for (int r = 0; r < MT; ++r) acc[r] = 0.0f;
#pragma unroll 4
    for (int i = lane; i < k4; i += VELOX_WARP) {
        const float4 wv = __ldcs(wr + i);
#pragma unroll
        for (int r = 0; r < MT; ++r) {
            if (r < m) {
                const float4 xv = x[static_cast<size_t>(r) * k4 + i];
                acc[r] += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < MT; ++r) {
        const float s = velox_warp_sum(acc[r]);
        if (lane == 0 && r < m) y[static_cast<size_t>(r) * n + col] = s;
    }
}

// ---- MatMul, large M (prefill) -----------------------------------------------------------
// Shared-memory tiled SGEMM for the NT layout: 64x64 output tile per block, K step 16, 256
// threads each owning a 4x4 micro-tile. Any m, n, k (edges are guarded).
constexpr int kBM = 64, kBN = 64, kBK = 16;

__global__ void __launch_bounds__(256)
sgemm_nt_kernel(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y, int m, int n, int k) {
    __shared__ float As[kBK][kBM + 4];
    __shared__ float Bs[kBK][kBN + 4];
    const int t = threadIdx.x;
    const int row0 = blockIdx.y * kBM, col0 = blockIdx.x * kBN;
    const int load_r = t / 4, load_k = (t % 4) * 4;  // each thread loads 4 k-values of one row
    const int ty = t / 16, tx = t % 16;
    float acc[4][4] = {};

    for (int k0 = 0; k0 < k; k0 += kBK) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int kk = k0 + load_k + i;
            const int xr = row0 + load_r, wr = col0 + load_r;
            As[load_k + i][load_r] = (xr < m && kk < k) ? x[static_cast<size_t>(xr) * k + kk] : 0.0f;
            Bs[load_k + i][load_r] = (wr < n && kk < k) ? w[static_cast<size_t>(wr) * k + kk] : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < kBK; ++kk) {
            float a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                a[i] = As[kk][ty * 4 + i];
                b[i] = Bs[kk][tx * 4 + i];
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
        const int r = row0 + ty * 4 + i;
        if (r >= m) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = col0 + tx * 4 + j;
            if (c < n) y[static_cast<size_t>(r) * n + c] = acc[i][j];
        }
    }
}

// ---- Attention ---------------------------------------------------------------------------
// One block of kAttnWarps warps per (token, head). Warp w visits keys w, w + kAttnWarps, ...
// with an online softmax; the warps' partial (max, sum, acc) are merged through shared memory.
// Splitting the key loop keeps more loads in flight than the portable one-warp kernel.
constexpr int kAttnWarps = 4;

template <int PER>
__global__ void __launch_bounds__(kAttnWarps * VELOX_WARP)
attention_split_kernel(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                       float* __restrict__ out, int heads, int kv_heads, int head_dim, int pos0, float scale) {
    __shared__ float s_acc[kAttnWarps][PER * VELOX_WARP];
    __shared__ float s_m[kAttnWarps], s_l[kAttnWarps];
    const int h = blockIdx.x, t = blockIdx.y;
    const int warp = threadIdx.x / VELOX_WARP, lane = threadIdx.x % VELOX_WARP;
    const int kvh = h / (heads / kv_heads);
    const size_t q_off = (static_cast<size_t>(t) * heads + h) * head_dim;

    float qr[PER], acc[PER];
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int d = lane + p * VELOX_WARP;
        qr[p] = d < head_dim ? q[q_off + d] * scale : 0.0f;
        acc[p] = 0.0f;
    }
    float m = -INFINITY, l = 0.0f;
    const int visible = pos0 + t + 1;
    for (int j = warp; j < visible; j += kAttnWarps) {
        const size_t kv_off = (static_cast<size_t>(j) * kv_heads + kvh) * head_dim;
        float s = 0.0f;
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            const int d = lane + p * VELOX_WARP;
            if (d < head_dim) s += qr[p] * k[kv_off + d];
        }
        s = velox_warp_sum(s);
        const float m_new = fmaxf(m, s);
        const float corr = expf(m - m_new);
        const float pj = expf(s - m_new);
        l = l * corr + pj;
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            const int d = lane + p * VELOX_WARP;
            if (d < head_dim) acc[p] = acc[p] * corr + pj * v[kv_off + d];
        }
        m = m_new;
    }
#pragma unroll
    for (int p = 0; p < PER; ++p) s_acc[warp][lane + p * VELOX_WARP] = acc[p];
    if (lane == 0) {
        s_m[warp] = m;
        s_l[warp] = l;
    }
    __syncthreads();
    if (warp != 0) return;
    float gm = -INFINITY;
#pragma unroll
    for (int w = 0; w < kAttnWarps; ++w) gm = fmaxf(gm, s_m[w]);
    float scale_w[kAttnWarps], gl = 0.0f;
#pragma unroll
    for (int w = 0; w < kAttnWarps; ++w) {
        // A warp that saw no keys has m = -inf, l = 0 and contributes nothing.
        scale_w[w] = s_l[w] > 0.0f ? expf(s_m[w] - gm) : 0.0f;
        gl += s_l[w] * scale_w[w];
    }
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int d = lane + p * VELOX_WARP;
        if (d >= head_dim) continue;
        float o = 0.0f;
#pragma unroll
        for (int w = 0; w < kAttnWarps; ++w) o += s_acc[w][d] * scale_w[w];
        out[q_off + d] = o / gl;
    }
}

bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; }

}  // namespace

// Small-M path. Unsupported (fallback) when m > 8, k % 4 != 0 or pointers are not 16-byte aligned.
VELOX_API int velox_4090_matmul_small_m(const float* x, const float* w, float* y, int m, int n, int k) {
    if (m > 8 || k % 4 != 0 || !aligned16(x) || !aligned16(w)) return VELOX_ERR_UNSUPPORTED;
    const dim3 grid((n + kGemvWarps - 1) / kGemvWarps);
    const dim3 block(kGemvWarps * VELOX_WARP);
    const auto* x4 = reinterpret_cast<const float4*>(x);
    const auto* w4 = reinterpret_cast<const float4*>(w);
    const int k4 = k / 4;
    if (m == 1) gemv_tile_kernel<1><<<grid, block>>>(x4, w4, y, m, n, k4);
    else if (m == 2) gemv_tile_kernel<2><<<grid, block>>>(x4, w4, y, m, n, k4);
    else if (m <= 4) gemv_tile_kernel<4><<<grid, block>>>(x4, w4, y, m, n, k4);
    else gemv_tile_kernel<8><<<grid, block>>>(x4, w4, y, m, n, k4);
    return velox_launch_status();
}

VELOX_API int velox_4090_matmul_tiled(const float* x, const float* w, float* y, int m, int n, int k) {
    if ((m + kBM - 1) / kBM > 65535) return VELOX_ERR_UNSUPPORTED;
    const dim3 grid((n + kBN - 1) / kBN, (m + kBM - 1) / kBM);
    sgemm_nt_kernel<<<grid, 256>>>(x, w, y, m, n, k);
    return velox_launch_status();
}

VELOX_API int velox_4090_attention(const float* q, const float* k, const float* v, float* out, int tokens, int heads,
                                   int kv_heads, int head_dim, int pos0, float scale) {
    if (tokens > 65535) return VELOX_ERR_UNSUPPORTED;
    const dim3 grid(heads, tokens), block(kAttnWarps * VELOX_WARP);
    if (head_dim <= 64) {
        attention_split_kernel<2><<<grid, block>>>(q, k, v, out, heads, kv_heads, head_dim, pos0, scale);
    } else if (head_dim <= 128) {
        attention_split_kernel<4><<<grid, block>>>(q, k, v, out, heads, kv_heads, head_dim, pos0, scale);
    } else if (head_dim <= 256) {
        attention_split_kernel<8><<<grid, block>>>(q, k, v, out, heads, kv_heads, head_dim, pos0, scale);
    } else {
        return VELOX_ERR_UNSUPPORTED;
    }
    return velox_launch_status();
}
