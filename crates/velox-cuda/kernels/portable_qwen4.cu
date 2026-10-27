// Portable kernels for Qwen4-Exp (docs/qwen4_exp.md): hyper-connection streams, PLE gating and its
// dilated convolution, MoE routing, the QSA indexer's pooling and selection. Simple and correct on
// every supported architecture; hardware branches replace them with tuned kernels.
#include <math.h>
#include <stdint.h>

#include "velox_common.cuh"

namespace {

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float siluf_(float x) { return x * sigmoidf_(x); }

// ---- rms_norm_groups: one block per (row, group) ----

__global__ void rms_norm_groups_kernel(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y,
                                       int groups, int dim, float eps) {
    __shared__ float scratch[VELOX_WARP];
    const size_t base = static_cast<size_t>(blockIdx.x) * dim;
    const float* wg = w + static_cast<size_t>(blockIdx.x % groups) * dim;
    float ss = 0.0f;
    for (int c = threadIdx.x; c < dim; c += blockDim.x) ss += x[base + c] * x[base + c];
    ss = velox_block_sum(ss, scratch);
    const float inv = rsqrtf(ss / static_cast<float>(dim) + eps);
    for (int c = threadIdx.x; c < dim; c += blockDim.x) y[base + c] = x[base + c] * inv * wg[c];
}

__global__ void silu_kernel(float* __restrict__ x, long long len) {
    for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < len;
         i += static_cast<long long>(gridDim.x) * blockDim.x)
        x[i] = siluf_(x[i]);
}

// out[r][d] = mean_s xn[r][s][d] * sigmoid(gate[r][s][d])
__global__ void hc_mix_kernel(const float* __restrict__ xn, const float* __restrict__ gate, float* __restrict__ out,
                              long long rows, int streams, int dim) {
    const long long n = rows * dim;
    for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const long long r = i / dim, d = i % dim;
        float acc = 0.0f;
        for (int s = 0; s < streams; ++s) {
            const long long j = (r * streams + s) * dim + d;
            acc += xn[j] * sigmoidf_(gate[j]);
        }
        out[i] = acc / static_cast<float>(streams);
    }
}

// res[r][s][d] += y[r][d] * 2 sigmoid(inject[r * ld + col + s])
__global__ void hc_combine_kernel(float* __restrict__ res, const float* __restrict__ y, const float* __restrict__ inj,
                                  long long rows, int streams, int dim, int ld, int col) {
    const long long n = rows * streams * dim;
    for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const long long d = i % dim, rs = i / dim, s = rs % streams, r = rs / streams;
        res[i] += y[r * dim + d] * (2.0f * sigmoidf_(inj[r * ld + col + s]));
    }
}

// One block per (row, stream): the gate from the stream's key . query, then the gated value.
__global__ void ple_gate_kernel(const float* __restrict__ key, const float* __restrict__ q,
                                const float* __restrict__ value, float* __restrict__ out, int streams, int dim) {
    __shared__ float scratch[VELOX_WARP];
    const size_t at = static_cast<size_t>(blockIdx.x) * dim;
    const float* v = value + static_cast<size_t>(blockIdx.x / streams) * dim;
    float s = 0.0f;
    for (int c = threadIdx.x; c < dim; c += blockDim.x) s += key[at + c] * q[at + c];
    const float g = velox_block_sum(s, scratch) / sqrtf(static_cast<float>(dim));
    const float mag = sqrtf(fmaxf(fabsf(g), 1e-6f));
    const float gate = sigmoidf_(g > 0.0f ? mag : g < 0.0f ? -mag : 0.0f);
    for (int c = threadIdx.x; c < dim; c += blockDim.x) out[at + c] = v[c] * gate;
}

// out[t][c] = silu(sum_j w[j][c] u[t + j dil][c]) with u = [state; x].
__global__ void conv1d_dilated_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                      const float* __restrict__ state, float* __restrict__ out, int tokens,
                                      int channels, int width, int dil, const long long* __restrict__ pp) {
    if (pp != nullptr) state = reinterpret_cast<const float*>(pp[0]);
    const long long n = static_cast<long long>(tokens) * channels;
    const int hist = (width - 1) * dil;
    for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const int t = static_cast<int>(i / channels), c = static_cast<int>(i % channels);
        float acc = 0.0f;
        for (int j = 0; j < width; ++j) {
            const int u = t + j * dil;
            const float v = u < hist ? state[static_cast<size_t>(u) * channels + c]
                                     : x[static_cast<size_t>(u - hist) * channels + c];
            acc += w[static_cast<size_t>(j) * channels + c] * v;
        }
        out[i] = siluf_(acc);
    }
}

// New state row i is row tokens + i of u = [state; x]; ascending i only reads rows not yet written.
__global__ void conv1d_dilated_state_kernel(const float* __restrict__ x, float* __restrict__ state, int tokens,
                                            int channels, int hist, const long long* __restrict__ pp) {
    if (pp != nullptr) state = reinterpret_cast<float*>(pp[0]);
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    for (int i = 0; i < hist; ++i) {
        const int u = tokens + i;
        state[static_cast<size_t>(i) * channels + c] =
            u < hist ? state[static_cast<size_t>(u) * channels + c] : x[static_cast<size_t>(u - hist) * channels + c];
    }
}

__global__ void sigmoid_scale_rows_kernel(float* __restrict__ x, const float* __restrict__ gate, long long rows,
                                          int dim, long long goff, long long gstride) {
    const long long n = rows * dim;
    for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<long long>(gridDim.x) * blockDim.x)
        x[i] *= sigmoidf_(gate[goff + (i / dim) * gstride]);
}

// ---- moe_route: one warp per row; k rounds of a warp argmax (ties to the lower id) ----

constexpr int kMoeKMax = 32;

// Lane l holds logits l, l + 32, ... of its row in registers (PER <= 32 per lane) and a mask of
// the selected ones. Each of the k rounds is a register scan and a warp argmax (ties to the lower
// id).
template <int PER>
__global__ void moe_route_kernel(const float* __restrict__ logits, float* __restrict__ ids, float* __restrict__ wts,
                                 float* __restrict__ counts, int rows, int row_len, int experts, int k) {
    const int r = blockIdx.x * (blockDim.x / VELOX_WARP) + threadIdx.x / VELOX_WARP;
    const int lane = threadIdx.x % VELOX_WARP;
    if (r >= rows) return;
    const float* row = logits + static_cast<size_t>(r) * row_len;
    float v[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int e = lane + i * VELOX_WARP;
        const float x = e < experts ? row[e] : -INFINITY;
        v[i] = x != x ? -INFINITY : x;  // NaN ranks below every number
    }
    unsigned taken = 0;  // bit i: this lane's expert lane + 32 i is selected (or past `experts`)
#pragma unroll
    for (int i = 0; i < PER; ++i)
        if (lane + i * VELOX_WARP >= experts) taken |= 1u << i;
    int sel[kMoeKMax];
    float val[kMoeKMax];
    for (int j = 0; j < k; ++j) {
        float best = -INFINITY;
        int best_i = -1;
#pragma unroll
        for (int i = 0; i < PER; ++i)  // ascending i: ties keep the lower id
            if (!(taken >> i & 1u) && (best_i < 0 || v[i] > best)) best = v[i], best_i = i;
        int best_id = best_i < 0 ? 0x7fffffff : lane + best_i * VELOX_WARP;
        for (int o = VELOX_WARP / 2; o > 0; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, best_id, o);
            if (ob > best || (ob == best && oi < best_id)) best = ob, best_id = oi;
        }
        sel[j] = best_id;
        val[j] = best;
        if (best_id % VELOX_WARP == lane) taken |= 1u << (best_id / VELOX_WARP);
    }
    if (lane != 0) return;
    float sum = 0.0f;
    for (int j = 0; j < k; ++j) sum += expf(val[j] - val[0]);
    for (int j = 0; j < k; ++j) {
        ids[static_cast<size_t>(r) * k + j] = __uint_as_float(static_cast<unsigned>(sel[j]));
        wts[static_cast<size_t>(r) * k + j] = expf(val[j] - val[0]) / sum;
        if (counts != nullptr) atomicAdd(counts + sel[j], 1.0f);
    }
}

// Larger expert counts: logits re-read from memory each round.
__global__ void moe_route_wide_kernel(const float* __restrict__ logits, float* __restrict__ ids, float* __restrict__ wts,
                                      float* __restrict__ counts, int rows, int row_len, int experts, int k) {
    const int r = blockIdx.x * (blockDim.x / VELOX_WARP) + threadIdx.x / VELOX_WARP;
    const int lane = threadIdx.x % VELOX_WARP;
    if (r >= rows) return;
    const float* row = logits + static_cast<size_t>(r) * row_len;
    int sel[kMoeKMax];
    float val[kMoeKMax];
    for (int j = 0; j < k; ++j) {
        float best = -INFINITY;
        int best_id = 0x7fffffff;
        for (int e = lane; e < experts; e += VELOX_WARP) {
            bool taken = false;
            for (int i = 0; i < j; ++i) taken |= sel[i] == e;
            const float v = row[e];
            const float key = v != v ? -INFINITY : v;
            if (!taken && (key > best || (key == best && e < best_id))) {
                best = key;
                best_id = e;
            }
        }
        for (int o = VELOX_WARP / 2; o > 0; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, best_id, o);
            if (ob > best || (ob == best && oi < best_id)) {
                best = ob;
                best_id = oi;
            }
        }
        sel[j] = best_id;
        val[j] = best;
    }
    if (lane != 0) return;
    float sum = 0.0f;
    for (int j = 0; j < k; ++j) sum += expf(val[j] - val[0]);
    for (int j = 0; j < k; ++j) {
        ids[static_cast<size_t>(r) * k + j] = __uint_as_float(static_cast<unsigned>(sel[j]));
        wts[static_cast<size_t>(r) * k + j] = expf(val[j] - val[0]) / sum;
        if (counts != nullptr) atomicAdd(counts + sel[j], 1.0f);
    }
}

// ---- QSA pooling: one block per block row; dim <= 1024 ----

__global__ void qsa_pool_kernel(const float* __restrict__ raw, const float* __restrict__ w, float* __restrict__ pooled,
                                int block0, int ratio, int dim, int rotary, float theta, float eps,
                                const long long* __restrict__ pp) {
    __shared__ float scratch[VELOX_WARP];
    extern __shared__ float v[];
    if (pp != nullptr) {
        block0 = static_cast<int>(pp[0]);
        raw = reinterpret_cast<const float*>(pp[1]);
        pooled = reinterpret_cast<float*>(pp[2]);
    }
    const int b = block0 + blockIdx.x, c = threadIdx.x;
    float mean = 0.0f;
    if (c < dim) {
        for (int i = 0; i < ratio; ++i) mean += raw[(static_cast<size_t>(b) * ratio + i) * dim + c];
        mean /= static_cast<float>(ratio);
    }
    const float ss = velox_block_sum(c < dim ? mean * mean : 0.0f, scratch);
    const float inv = rsqrtf(ss / static_cast<float>(dim) + eps);
    if (c < dim) v[c] = mean * inv * w[c];
    __syncthreads();
    if (c >= dim) return;
    float out = v[c];
    const int half = rotary / 2;
    if (c < rotary) {
        const int i = c < half ? c : c - half;
        const float freq = powf(theta, -2.0f * static_cast<float>(i) / static_cast<float>(rotary));
        float sn, cs;
        sincosf(static_cast<float>(b * ratio) * freq, &sn, &cs);
        out = c < half ? v[c] * cs - v[c + half] * sn : v[c] * cs + v[c - half] * sn;
    }
    pooled[static_cast<size_t>(b) * dim + c] = out;
}

// ---- QSA selection ----

// scores[t][b] = sum_h relu(q[t][h] . pooled[b]) / sqrt(dim) for the blocks query t sees.
__global__ void qsa_score_kernel(const float* __restrict__ q, const float* __restrict__ pooled,
                                 float* __restrict__ scores, int tokens, int heads, int dim, int pos0, int ratio,
                                 int nblocks) {
    const long long n = static_cast<long long>(tokens) * nblocks;
    for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const int t = static_cast<int>(i / nblocks), b = static_cast<int>(i % nblocks);
        if (b >= (pos0 + t + 1) / ratio) continue;
        const float* pb = pooled + static_cast<size_t>(b) * dim;
        float sum = 0.0f;
        for (int h = 0; h < heads; ++h) {
            const float* qh = q + (static_cast<size_t>(t) * heads + h) * dim;
            float d = 0.0f;
            for (int c = 0; c < dim; ++c) d += qh[c] * pb[c];
            sum += fmaxf(d, 0.0f);
        }
        scores[i] = sum / sqrtf(static_cast<float>(dim));
    }
}

constexpr int kSelThreads = 256;

// Exclusive prefix sum of one value per thread over the block; returns the total in `total`.
__device__ __forceinline__ int block_scan(int v, int* sh, int* total) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int o = 1; o < kSelThreads; o <<= 1) {
        const int add = threadIdx.x >= o ? sh[threadIdx.x - o] : 0;
        __syncthreads();
        sh[threadIdx.x] += add;
        __syncthreads();
    }
    const int incl = sh[threadIdx.x];
    *total = sh[kSelThreads - 1];
    __syncthreads();
    return incl - v;
}

// One block per query: radix-select the threshold of the best `top` blocks (scores are >= 0, so
// their bits order as unsigned), then emit selected blocks' members in ascending order, then the
// tail, then -1 padding.
__global__ void __launch_bounds__(kSelThreads)
qsa_pick_kernel(const float* __restrict__ scores, float* __restrict__ sel, int nblocks, int pos0, int ratio, int top,
                int sel_len) {
    __shared__ int hist[256];
    __shared__ int sh[kSelThreads];
    __shared__ unsigned s_prefix;
    __shared__ int s_need;
    const int t = blockIdx.x;
    const int nc = (pos0 + t + 1) / ratio;
    const float* sc = scores + static_cast<size_t>(t) * nblocks;
    float* out = sel + static_cast<size_t>(t) * sel_len;
    const bool all = nc <= top;
    // Threshold bits and how many blocks equal to it still get in.
    unsigned prefix = 0;
    int need = top;
    if (!all) {
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (int i = threadIdx.x; i < 256; i += kSelThreads) hist[i] = 0;
            __syncthreads();
            const unsigned mask = shift == 24 ? 0u : 0xffffffffu << (shift + 8);
            for (int b = threadIdx.x; b < nc; b += kSelThreads) {
                const unsigned u = __float_as_uint(sc[b]);
                if ((u & mask) == (prefix & mask)) atomicAdd(&hist[(u >> shift) & 255u], 1);
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                int acc = 0, d = 255;
                for (; d > 0; --d) {
                    if (acc + hist[d] >= need) break;
                    acc += hist[d];
                }
                s_prefix = prefix | (static_cast<unsigned>(d) << shift);
                s_need = need - acc;
            }
            __syncthreads();
            prefix = s_prefix;
            need = s_need;
        }
    }
    // Emit selected block members in ascending block order.
    int written = 0, equal_seen = 0;
    for (int b0 = 0; b0 < nc; b0 += kSelThreads) {
        const int b = b0 + threadIdx.x;
        bool gt = false, eq = false;
        if (b < nc && !all) {
            const unsigned u = __float_as_uint(sc[b]);
            gt = u > prefix;
            eq = u == prefix;
        }
        int eq_total;
        const int eq_rank = block_scan(eq ? 1 : 0, sh, &eq_total);
        const bool take = b < nc && (all || gt || (eq && equal_seen + eq_rank < need));
        int taken;
        const int at = block_scan(take ? 1 : 0, sh, &taken);
        if (take)
            for (int i = 0; i < ratio; ++i)
                out[(written + at) * ratio + i] = __uint_as_float(static_cast<unsigned>(b * ratio + i));
        written += taken;
        equal_seen += eq_total;
    }
    // Tail, then padding.
    int n = written * ratio;
    for (int p = nc * ratio + threadIdx.x; p <= pos0 + t; p += kSelThreads)
        out[n + p - nc * ratio] = __uint_as_float(static_cast<unsigned>(p));
    n += pos0 + t + 1 - nc * ratio;
    for (int i = n + threadIdx.x; i < sel_len; i += kSelThreads) out[i] = __uint_as_float(0xffffffffu);
}

}  // namespace

VELOX_API int velox_portable_rms_norm_groups(const float* x, const float* w, float* y, int rows, int groups, int dim,
                                             float eps) {
    rms_norm_groups_kernel<<<static_cast<unsigned>(rows) * groups, 256>>>(x, w, y, groups, dim, eps);
    return velox_launch_status();
}

VELOX_API int velox_portable_silu(float* x, long long len) {
    if (len <= 0) return 0;
    silu_kernel<<<velox_grid_1d(len, 256), 256>>>(x, len);
    return velox_launch_status();
}

VELOX_API int velox_portable_hc_mix(const float* xn, const float* gate, float* out, long long rows, int streams,
                                    int dim) {
    hc_mix_kernel<<<velox_grid_1d(rows * dim, 256), 256>>>(xn, gate, out, rows, streams, dim);
    return velox_launch_status();
}

VELOX_API int velox_portable_hc_combine(float* res, const float* y, const float* inj, long long rows, int streams,
                                        int dim, int ld, int col) {
    hc_combine_kernel<<<velox_grid_1d(rows * streams * dim, 256), 256>>>(res, y, inj, rows, streams, dim, ld, col);
    return velox_launch_status();
}

VELOX_API int velox_portable_ple_gate(const float* key, const float* q, const float* value, float* out, int rows,
                                      int streams, int dim) {
    ple_gate_kernel<<<static_cast<unsigned>(rows) * streams, 256>>>(key, q, value, out, streams, dim);
    return velox_launch_status();
}

// params: null, or device [state] staged by a graph step (a recording serves any session).
VELOX_API int velox_portable_conv1d_dilated(const float* x, const float* w, float* state, float* out, int tokens,
                                            int channels, int width, int dil, const long long* params) {
    const long long n = static_cast<long long>(tokens) * channels;
    conv1d_dilated_kernel<<<velox_grid_1d(n, 256), 256>>>(x, w, state, out, tokens, channels, width, dil, params);
    if (const int e = velox_launch_status()) return e;
    conv1d_dilated_state_kernel<<<(channels + 255) / 256, 256>>>(x, state, tokens, channels, (width - 1) * dil,
                                                                  params);
    return velox_launch_status();
}

VELOX_API int velox_portable_sigmoid_scale_rows(float* x, const float* gate, long long rows, int dim,
                                                long long goff, long long gstride) {
    sigmoid_scale_rows_kernel<<<velox_grid_1d(rows * dim, 256), 256>>>(x, gate, rows, dim, goff, gstride);
    return velox_launch_status();
}

VELOX_API int velox_portable_moe_route(const float* logits, float* ids, float* wts, float* counts, int rows,
                                       int row_len, int experts, int k) {
    if (k > kMoeKMax) return VELOX_ERR_UNSUPPORTED;
    const int warps = 4;
    const unsigned grid = (rows + warps - 1) / warps;
    if (experts <= 16 * VELOX_WARP)
        moe_route_kernel<16><<<grid, warps * VELOX_WARP>>>(logits, ids, wts, counts, rows, row_len, experts, k);
    else if (experts <= 32 * VELOX_WARP)
        moe_route_kernel<32><<<grid, warps * VELOX_WARP>>>(logits, ids, wts, counts, rows, row_len, experts, k);
    else
        moe_route_wide_kernel<<<grid, warps * VELOX_WARP>>>(logits, ids, wts, counts, rows, row_len, experts, k);
    return velox_launch_status();
}

// params: null, or device [block0, raw, pooled] staged by a graph step.
VELOX_API int velox_portable_qsa_pool(const float* raw, const float* w, float* pooled, int block0, int blocks, int ratio,
                                      int dim, int rotary, float theta, float eps, const long long* params) {
    if (dim > 1024) return VELOX_ERR_UNSUPPORTED;
    const int threads = (dim + VELOX_WARP - 1) / VELOX_WARP * VELOX_WARP;
    qsa_pool_kernel<<<blocks, threads, dim * sizeof(float)>>>(raw, w, pooled, block0, ratio, dim, rotary, theta, eps,
                                                               params);
    return velox_launch_status();
}

VELOX_API int velox_portable_qsa_select(const float* q, const float* pooled, float* sel, float* scratch, int tokens,
                                        int heads, int dim, int pos0, int ratio, int budget) {
    const int nblocks = (pos0 + tokens) / ratio;
    const int sel_len = budget + ratio - 1;
    if (nblocks > 0) {
        const long long n = static_cast<long long>(tokens) * nblocks;
        qsa_score_kernel<<<velox_grid_1d(n, 256), 256>>>(q, pooled, scratch, tokens, heads, dim, pos0, ratio, nblocks);
        if (const int e = velox_launch_status()) return e;
    }
    qsa_pick_kernel<<<tokens, kSelThreads>>>(scratch, sel, nblocks, pos0, ratio, budget / ratio, sel_len);
    return velox_launch_status();
}
