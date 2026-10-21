// Portable kernels: correct on every supported NVIDIA architecture, simple rather than fast.
// They are the fallback for any op a hardware branch has not tuned yet, so a feature added on main
// works on every branch immediately. Shapes are validated on the Rust side before launch.
#include <math.h>

#include "velox_common.cuh"

namespace {

// y[m, n] = x[m, k] . w[n, k]^T. One warp per output element.
__global__ void matmul_warp_kernel(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y,
                                   int m, int n, int k) {
    const int col = blockIdx.x * (blockDim.x / VELOX_WARP) + threadIdx.x / VELOX_WARP;
    const int lane = threadIdx.x % VELOX_WARP;
    if (col >= n) return;
    const float* wr = w + static_cast<size_t>(col) * k;
    for (int row = blockIdx.y; row < m; row += gridDim.y) {
        const float* xr = x + static_cast<size_t>(row) * k;
        float acc = 0.0f;
        for (int i = lane; i < k; i += VELOX_WARP) acc += xr[i] * wr[i];
        acc = velox_warp_sum(acc);
        if (lane == 0) y[static_cast<size_t>(row) * n + col] = acc;
    }
}

__global__ void rms_norm_kernel(const float* __restrict__ x, const float* __restrict__ weight, float* __restrict__ y,
                                int dim, float eps, float offset) {
    __shared__ float scratch[VELOX_WARP];
    const float* xr = x + static_cast<size_t>(blockIdx.x) * dim;
    float* yr = y + static_cast<size_t>(blockIdx.x) * dim;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) ss += xr[i] * xr[i];
    ss = velox_block_sum(ss, scratch);
    const float inv = rsqrtf(ss / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) yr[i] = xr[i] * inv * (offset + weight[i]);
}

__global__ void rms_norm_warp_kernel(const float* __restrict__ x, const float* __restrict__ weight,
                                     float* __restrict__ y, int rows, int dim, float eps, float offset) {
    const int row = blockIdx.x * (blockDim.x / VELOX_WARP) + threadIdx.x / VELOX_WARP;
    const int lane = threadIdx.x % VELOX_WARP;
    if (row >= rows) return;
    const float* xr = x + static_cast<size_t>(row) * dim;
    float* yr = y + static_cast<size_t>(row) * dim;
    float ss = 0.0f;
    for (int i = lane; i < dim; i += VELOX_WARP) ss += xr[i] * xr[i];
    ss = velox_warp_sum(ss);
    const float inv = rsqrtf(ss / dim + eps);
    for (int i = lane; i < dim; i += VELOX_WARP) yr[i] = xr[i] * inv * (offset + weight[i]);
}

// The angle is formed in double: positions reach 1e5+, where a float angle loses ~1e-2 rad.
// One block per token: the token's rotary/2 angles are formed once (in double) into shared
// memory, then every head applies them. Double is slow on consumer GPUs, so it runs once per
// pair per token rather than once per pair per head.
__global__ void rope_kernel(float* __restrict__ x, int heads, int head_dim, int rotary_dim, int pos0, float theta) {
    extern __shared__ float cs[];  // [half] cos, then [half] sin
    const int half = rotary_dim / 2, t = blockIdx.x;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const double freq = pow(static_cast<double>(theta), -2.0 * i / rotary_dim);
        double s, c;
        sincos(static_cast<double>(pos0 + t) * freq, &s, &c);
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

__global__ void silu_mul_kernel(float* __restrict__ gate, const float* __restrict__ up, long long len) {
    for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < len;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const float g = gate[i];
        gate[i] = g / (1.0f + expf(-g)) * up[i];
    }
}

__global__ void add_kernel(float* __restrict__ acc, const float* __restrict__ x, long long len) {
    for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < len;
         i += static_cast<long long>(gridDim.x) * blockDim.x)
        acc[i] += x[i];
}

// One warp per (token, head); keys are visited sequentially with an online softmax. Each lane owns
// PER elements of the head dimension.
template <int PER>
__global__ void attention_warp_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                      const float* __restrict__ v, float* __restrict__ out, int heads, int kv_heads,
                                      int head_dim, int pos0, float scale) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int lane = threadIdx.x;
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
    for (int j = 0; j < visible; ++j) {
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
    for (int p = 0; p < PER; ++p) {
        const int d = lane + p * VELOX_WARP;
        if (d < head_dim) out[q_off + d] = acc[p] / l;
    }
}

}  // namespace

VELOX_API int velox_portable_matmul(const float* x, const float* w, float* y, int m, int n, int k) {
    constexpr int kWarps = 8;
    dim3 grid((n + kWarps - 1) / kWarps, m < 65535 ? m : 65535);
    matmul_warp_kernel<<<grid, kWarps * VELOX_WARP>>>(x, w, y, m, n, k);
    return velox_launch_status();
}

VELOX_API int velox_portable_rms_norm(const float* x, const float* weight, float* y, int rows, int dim, float eps,
                                      int offset) {
    if (dim <= 1024) {
        // Many short rows (per-head norms): one warp per row, 8 rows per block.
        rms_norm_warp_kernel<<<(rows + 7) / 8, 256>>>(x, weight, y, rows, dim, eps, offset ? 1.0f : 0.0f);
        return velox_launch_status();
    }
    // Wide rows: few of them per call (the residual stream), so give each the most threads.
    rms_norm_kernel<<<rows, 1024>>>(x, weight, y, dim, eps, offset ? 1.0f : 0.0f);
    return velox_launch_status();
}

VELOX_API int velox_portable_rope(float* x, int tokens, int heads, int head_dim, int rotary_dim, int pos0, float theta) {
    if (tokens <= 0) return 0;
    rope_kernel<<<tokens, 256, rotary_dim * sizeof(float)>>>(x, heads, head_dim, rotary_dim, pos0, theta);
    return velox_launch_status();
}

VELOX_API int velox_portable_silu_mul(float* gate, const float* up, long long len) {
    silu_mul_kernel<<<velox_grid_1d(len, 256), 256>>>(gate, up, len);
    return velox_launch_status();
}

VELOX_API int velox_portable_add(float* acc, const float* x, long long len) {
    add_kernel<<<velox_grid_1d(len, 256), 256>>>(acc, x, len);
    return velox_launch_status();
}

namespace {

// Split-KV (flash-decoding) attention for few queries over a long cache: each (token, head, split)
// warp reduces kSplitKeys keys to a partial (max, sum, acc), then one warp per (token, head)
// merges the splits.
constexpr int kSplitKeys = 256;  // widest split; short caches use narrower ones

template <int PER>
__global__ void attention_partial_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                         const float* __restrict__ v, float* __restrict__ part_acc,
                                         float* __restrict__ part_ml, int heads, int kv_heads, int head_dim, int pos0,
                                         float scale, int splits, int split_keys) {
    const int h = blockIdx.x, t = blockIdx.y, sp = blockIdx.z, lane = threadIdx.x;
    const int kvh = h / (heads / kv_heads);
    const size_t q_off = (static_cast<size_t>(t) * heads + h) * head_dim;
    const int visible = pos0 + t + 1;
    const int j0 = sp * split_keys, j1 = min(j0 + split_keys, visible);
    float qr[PER], acc[PER];
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int d = lane + p * VELOX_WARP;
        qr[p] = d < head_dim ? q[q_off + d] * scale : 0.0f;
        acc[p] = 0.0f;
    }
    float m = -INFINITY, l = 0.0f;
    for (int j = j0; j < j1; ++j) {
        const size_t kv_off = (static_cast<size_t>(j) * kv_heads + kvh) * head_dim;
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
    const size_t idx = (static_cast<size_t>(t) * heads + h) * splits + sp;
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int d = lane + p * VELOX_WARP;
        if (d < head_dim) part_acc[idx * head_dim + d] = acc[p];
    }
    if (lane == 0) {
        part_ml[2 * idx] = m;
        part_ml[2 * idx + 1] = l;
    }
}

template <int PER>
__global__ void attention_combine_kernel(const float* __restrict__ part_acc, const float* __restrict__ part_ml,
                                         float* __restrict__ out, int heads, int head_dim, int splits) {
    const int h = blockIdx.x, t = blockIdx.y, lane = threadIdx.x;
    const size_t base = (static_cast<size_t>(t) * heads + h) * splits;
    float gm = -INFINITY;
    for (int sp = 0; sp < splits; ++sp)
        if (part_ml[2 * (base + sp) + 1] > 0.0f) gm = fmaxf(gm, part_ml[2 * (base + sp)]);
    float acc[PER], gl = 0.0f;
#pragma unroll
    for (int p = 0; p < PER; ++p) acc[p] = 0.0f;
    for (int sp = 0; sp < splits; ++sp) {
        const float l = part_ml[2 * (base + sp) + 1];
        if (l <= 0.0f) continue;  // a split past the causal limit saw no keys
        const float w = expf(part_ml[2 * (base + sp)] - gm);
        gl += l * w;
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            const int d = lane + p * VELOX_WARP;
            if (d < head_dim) acc[p] += part_acc[(base + sp) * head_dim + d] * w;
        }
    }
    const size_t o_off = (static_cast<size_t>(t) * heads + h) * head_dim;
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int d = lane + p * VELOX_WARP;
        if (d < head_dim) out[o_off + d] = acc[p] / gl;
    }
}

// Grow-only scratch for split partials. Engine work is serialized, so one buffer suffices.
float* split_scratch(size_t floats) {
    static float* ptr = nullptr;
    static size_t cap = 0;
    if (floats > cap) {
        if (ptr != nullptr) cudaFree(ptr);
        if (cudaMalloc(&ptr, floats * sizeof(float)) != cudaSuccess) {
            ptr = nullptr;
            cap = 0;
            return nullptr;
        }
        cap = floats;
    }
    return ptr;
}

template <int PER>
int attention_split(const float* q, const float* k, const float* v, float* out, int tokens, int heads, int kv_heads,
                    int head_dim, int pos0, float scale) {
    // Aim for >= 512 partial warps so short caches still occupy the GPU, with 8..256 keys each.
    const int kv = pos0 + tokens;
    const int want = (512 + tokens * heads - 1) / (tokens * heads);
    int split_keys = (kv + want - 1) / want;
    split_keys = split_keys < 8 ? 8 : split_keys > kSplitKeys ? kSplitKeys : split_keys;
    const int splits = (kv + split_keys - 1) / split_keys;
    const size_t parts = static_cast<size_t>(tokens) * heads * splits;
    float* scratch = split_scratch(parts * (head_dim + 2));
    if (scratch == nullptr) return static_cast<int>(cudaErrorMemoryAllocation);
    float* part_acc = scratch;
    float* part_ml = scratch + parts * head_dim;
    attention_partial_kernel<PER><<<dim3(heads, tokens, splits), VELOX_WARP>>>(q, k, v, part_acc, part_ml, heads,
                                                                              kv_heads, head_dim, pos0, scale, splits, split_keys);
    attention_combine_kernel<PER><<<dim3(heads, tokens), VELOX_WARP>>>(part_acc, part_ml, out, heads, head_dim,
                                                                       splits);
    return velox_launch_status();
}

}  // namespace

VELOX_API int velox_portable_attention(const float* q, const float* k, const float* v, float* out, int tokens,
                                       int heads, int kv_heads, int head_dim, int pos0, float scale) {
    if (tokens > 65535) return VELOX_ERR_UNSUPPORTED;
    // Few queries over a long cache: split the keys so the whole GPU works on them.
    if (tokens * heads <= 2048 && head_dim <= 256) {
        if (head_dim <= 64) return attention_split<2>(q, k, v, out, tokens, heads, kv_heads, head_dim, pos0, scale);
        if (head_dim <= 128) return attention_split<4>(q, k, v, out, tokens, heads, kv_heads, head_dim, pos0, scale);
        return attention_split<8>(q, k, v, out, tokens, heads, kv_heads, head_dim, pos0, scale);
    }
    dim3 grid(heads, tokens);
    if (head_dim <= 32) {
        attention_warp_kernel<1><<<grid, VELOX_WARP>>>(q, k, v, out, heads, kv_heads, head_dim, pos0, scale);
    } else if (head_dim <= 64) {
        attention_warp_kernel<2><<<grid, VELOX_WARP>>>(q, k, v, out, heads, kv_heads, head_dim, pos0, scale);
    } else if (head_dim <= 128) {
        attention_warp_kernel<4><<<grid, VELOX_WARP>>>(q, k, v, out, heads, kv_heads, head_dim, pos0, scale);
    } else if (head_dim <= 256) {
        attention_warp_kernel<8><<<grid, VELOX_WARP>>>(q, k, v, out, heads, kv_heads, head_dim, pos0, scale);
    } else {
        return VELOX_ERR_UNSUPPORTED;
    }
    return velox_launch_status();
}

namespace {
// Block attention partials (DFlash drafts): token t's keys are its visible context positions
// [ctx_start(t), pos0), read from the ring at p % ring, then the block's own rows.
template <int PER>
__global__ void attention_block_partial_kernel(const float* __restrict__ q, const float* __restrict__ k_ctx,
                                               const float* __restrict__ v_ctx, const float* __restrict__ k_blk,
                                               const float* __restrict__ v_blk, float* __restrict__ part_acc,
                                               float* __restrict__ part_ml, int tokens, int heads, int kv_heads,
                                               int head_dim, long long pos0, long long ctx_lo, int window, int ring,
                                               float scale, int splits, int split_keys) {
    const int h = blockIdx.x, t = blockIdx.y, sp = blockIdx.z, lane = threadIdx.x;
    const int kvh = h / (heads / kv_heads);
    const size_t q_off = (static_cast<size_t>(t) * heads + h) * head_dim;
    long long start = pos0 + t + 1 - window;
    if (start < ctx_lo) start = ctx_lo;
    const int n_ctx = static_cast<int>(pos0 - start);
    const int total = n_ctx + tokens;
    const int j0 = sp * split_keys, j1 = min(j0 + split_keys, total);
    float qr[PER], acc[PER];
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int d = lane + p * VELOX_WARP;
        qr[p] = d < head_dim ? q[q_off + d] * scale : 0.0f;
        acc[p] = 0.0f;
    }
    float m = -INFINITY, l = 0.0f;
    for (int j = j0; j < j1; ++j) {
        const float* kr;
        const float* vr;
        if (j < n_ctx) {
            const size_t row = static_cast<size_t>((start + j) % ring);
            kr = k_ctx + (row * kv_heads + kvh) * head_dim;
            vr = v_ctx + (row * kv_heads + kvh) * head_dim;
        } else {
            const size_t row = static_cast<size_t>(j - n_ctx);
            kr = k_blk + (row * kv_heads + kvh) * head_dim;
            vr = v_blk + (row * kv_heads + kvh) * head_dim;
        }
        float s = 0.0f;
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            const int d = lane + p * VELOX_WARP;
            if (d < head_dim) s += qr[p] * kr[d];
        }
        s = velox_warp_sum(s);
        const float m_new = fmaxf(m, s);
        const float corr = expf(m - m_new), pj = expf(s - m_new);
        l = l * corr + pj;
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            const int d = lane + p * VELOX_WARP;
            if (d < head_dim) acc[p] = acc[p] * corr + pj * vr[d];
        }
        m = m_new;
    }
    const size_t idx = (static_cast<size_t>(t) * heads + h) * splits + sp;
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int d = lane + p * VELOX_WARP;
        if (d < head_dim) part_acc[idx * head_dim + d] = acc[p];
    }
    if (lane == 0) {
        part_ml[2 * idx] = m;
        part_ml[2 * idx + 1] = l;
    }
}

template <int PER>
int attention_block_split(const float* q, const float* k_ctx, const float* v_ctx, const float* k_blk,
                          const float* v_blk, float* out, int tokens, int heads, int kv_heads, int head_dim,
                          long long pos0, long long ctx_lo, int window, int ring) {
    long long most = pos0 - ctx_lo;  // token 0 sees the most context
    if (most > window - 1) most = window - 1;
    const int kv = static_cast<int>(most) + tokens;
    const int want = (1024 + tokens * heads - 1) / (tokens * heads);
    int split_keys = (kv + want - 1) / want;
    split_keys = split_keys < 16 ? 16 : split_keys > kSplitKeys ? kSplitKeys : split_keys;
    const int splits = (kv + split_keys - 1) / split_keys;
    const size_t parts = static_cast<size_t>(tokens) * heads * splits;
    float* scratch = split_scratch(parts * (head_dim + 2));
    if (scratch == nullptr) return static_cast<int>(cudaErrorMemoryAllocation);
    float* part_acc = scratch;
    float* part_ml = scratch + parts * head_dim;
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    attention_block_partial_kernel<PER><<<dim3(heads, tokens, splits), VELOX_WARP>>>(
        q, k_ctx, v_ctx, k_blk, v_blk, part_acc, part_ml, tokens, heads, kv_heads, head_dim, pos0, ctx_lo, window,
        ring, scale, splits, split_keys);
    attention_combine_kernel<PER><<<dim3(heads, tokens), VELOX_WARP>>>(part_acc, part_ml, out, heads, head_dim,
                                                                       splits);
    return velox_launch_status();
}

__global__ void dyn_conv_kernel(const float* __restrict__ x, const float* __restrict__ base,
                                const float* __restrict__ delta, float* __restrict__ out, int rows, int channels,
                                int group, int side, int accumulate) {
    const long long n = static_cast<long long>(rows) * channels;
    const int g_n = channels / group;
    for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
         idx += static_cast<long long>(gridDim.x) * blockDim.x) {
        const int i = static_cast<int>(idx / channels), h = static_cast<int>(idx % channels), g = h / group;
        const float* d = delta + static_cast<size_t>(i) * 4 * g_n;
        const int t0 = 2 * side, t1 = 2 * side + 1;
        float y = (base[t0 * channels + h] + d[t0 * g_n + g]) * x[idx];
        if (i > 0) y += (base[t1 * channels + h] + d[t1 * g_n + g]) * x[idx - channels];
        out[idx] = accumulate ? out[idx] + y : y;
    }
}
}  // namespace

VELOX_API int velox_portable_attention_block(const float* q, const float* k_ctx, const float* v_ctx,
                                             const float* k_blk, const float* v_blk, float* out, int tokens,
                                             int heads, int kv_heads, int head_dim, long long pos0, long long ctx_lo,
                                             int window, int ring) {
    if (tokens > 65535 || head_dim > 256) return VELOX_ERR_UNSUPPORTED;
    if (head_dim <= 64)
        return attention_block_split<2>(q, k_ctx, v_ctx, k_blk, v_blk, out, tokens, heads, kv_heads, head_dim, pos0,
                                        ctx_lo, window, ring);
    if (head_dim <= 128)
        return attention_block_split<4>(q, k_ctx, v_ctx, k_blk, v_blk, out, tokens, heads, kv_heads, head_dim, pos0,
                                        ctx_lo, window, ring);
    return attention_block_split<8>(q, k_ctx, v_ctx, k_blk, v_blk, out, tokens, heads, kv_heads, head_dim, pos0,
                                    ctx_lo, window, ring);
}

VELOX_API int velox_portable_dyn_conv(const float* x, const float* base, const float* delta, float* out, int rows,
                                      int channels, int group, int side, int accumulate) {
    const long long n = static_cast<long long>(rows) * channels;
    if (n == 0) return 0;
    dyn_conv_kernel<<<velox_grid_1d(n, 256), 256>>>(x, base, delta, out, rows, channels, group, side, accumulate);
    return velox_launch_status();
}

VELOX_API int velox_portable_copy2d(const float* src, long long src_stride, float* dst, long long dst_stride,
                                    long long rows, long long cols) {
    if (rows == 0 || cols == 0) return 0;
    return static_cast<int>(cudaMemcpy2DAsync(dst, dst_stride * sizeof(float), src, src_stride * sizeof(float),
                                              cols * sizeof(float), rows, cudaMemcpyDeviceToDevice, 0));
}

namespace {
__global__ void concat_cols_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out,
                                   long long rows, int a_cols, int b_cols) {
    const int w = a_cols + b_cols;
    const long long n = rows * w;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const long long r = i / w;
        const int c = static_cast<int>(i % w);
        out[i] = c < a_cols ? a[r * a_cols + c] : b[r * b_cols + (c - a_cols)];
    }
}
}  // namespace

VELOX_API int velox_portable_concat_cols(const float* a, const float* b, float* out, long long rows, int a_cols,
                                         int b_cols) {
    const long long n = rows * (a_cols + b_cols);
    if (n == 0) return 0;
    concat_cols_kernel<<<velox_grid_1d(n, 256), 256>>>(a, b, out, rows, a_cols, b_cols);
    return velox_launch_status();
}

namespace {
struct SplitOuts {
    float* ptr[4];
    int off[4];
    int cols[4];
    int n;
};

// One thread per source element; each element lands in at most one output.
__global__ void split_cols_kernel(const float* __restrict__ src, long long rows, int src_cols, SplitOuts o) {
    const long long n = rows * src_cols;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const long long r = i / src_cols;
        const int c = static_cast<int>(i % src_cols);
        for (int j = 0; j < o.n; ++j) {
            if (c >= o.off[j] && c < o.off[j] + o.cols[j]) {
                o.ptr[j][r * o.cols[j] + (c - o.off[j])] = src[i];
                break;
            }
        }
    }
}

__global__ void swiglu_kernel(const float* __restrict__ src, long long rows, int src_cols, int gate, int up, int width,
                              float* __restrict__ out) {
    const long long n = rows * width;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const long long r = i / width;
        const int c = static_cast<int>(i % width);
        const float g = src[r * src_cols + gate + c];
        out[i] = g / (1.0f + expf(-g)) * src[r * src_cols + up + c];
    }
}
}  // namespace

VELOX_API int velox_portable_split_cols(const float* src, long long rows, int src_cols, float* const* ptrs,
                                        const int* offs, const int* cols, int n) {
    if (n <= 0 || n > 4) return VELOX_ERR_UNSUPPORTED;
    SplitOuts o{};
    for (int j = 0; j < n; ++j) o.ptr[j] = ptrs[j], o.off[j] = offs[j], o.cols[j] = cols[j];
    o.n = n;
    const long long total = rows * src_cols;
    if (total == 0) return 0;
    split_cols_kernel<<<velox_grid_1d(total, 256), 256>>>(src, rows, src_cols, o);
    return velox_launch_status();
}

VELOX_API int velox_portable_swiglu(const float* src, long long rows, int src_cols, int gate, int up, int width,
                                    float* out) {
    const long long n = rows * width;
    if (n == 0) return 0;
    swiglu_kernel<<<velox_grid_1d(n, 256), 256>>>(src, rows, src_cols, gate, up, width, out);
    return velox_launch_status();
}
