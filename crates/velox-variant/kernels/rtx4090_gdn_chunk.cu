// RTX 5060 Ti chunked Gated DeltaNet at prefill widths on bf16 tensor cores: ninfer's mixed
// precision. The state S [v_dim, DK] of a head stays f32 in registers as mma accumulators; the four
// products of every 16-token chunk
//   Y = S [q | k]^T,   D^T = R^T T^T,   o^T = gamma y + D^T P^T,   S = gamma_last S + (D^T diag(w)) K
// run on mma.m16n8k16 with bf16 operands and f32 accumulation. The accumulator fragments of one
// product repack register-locally into the A operand of the next (the C layout's rows g / g + 8 at
// columns 2t, 2t + 1 are the A layout's k pairs), so only the chunk's inputs go through shared
// memory. The math is main's chunked kernel (see portable_model.cu: D = T R with T = (I + A)^-1
// from a prepare pass); the prepare pass here is that one, also emitting the normalized q/k as
// bf16 per key head. Main's exact f32 kernels serve decode widths and recorded steps.
//
// Qwen3.8 (48 heads, 128 x 128), 1024 tokens: 657 us exact f32 -> see kernels.rs.
#include <cuda_bf16.h>
#include <stdint.h>

#include "velox_common.cuh"

namespace {

constexpr int C = 16;       // tokens per chunk
constexpr int VT = 64;      // state rows per block: one 16-row m-tile per warp
constexpr int kThreads = 128;
constexpr int kPrep = 2 * C * C + 3 * C;  // per (head, chunk): T, P, gamma, decay_last, beta

__device__ __forceinline__ uint32_t bf16x2(float lo, float hi) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ void mma_bf16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Four 8x8 b16 tiles, transposed: from a row-major [k][n] tile at `p` (this lane's row address),
// the B fragments (b0, b1) of n-tiles n0 and n0 + 8.
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], const __nv_bfloat16* p) {
    const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}

// The C fragment (rows g / g + 8, columns 2t, 2t + 1) of two adjacent n-tiles as the A fragment
// of a 16-wide k step, optionally scaled per column.
__device__ __forceinline__ void frag_a(uint32_t (&a)[4], const float (&c0)[4], const float (&c1)[4]) {
    a[0] = bf16x2(c0[0], c0[1]);
    a[1] = bf16x2(c0[2], c0[3]);
    a[2] = bf16x2(c1[0], c1[1]);
    a[3] = bf16x2(c1[2], c1[3]);
}

// One block per (value head, chunk): L2-normalized q/k of the chunk (also stored as bf16 per key
// head for the recurrence), gates and their scan, A[r][s] = beta_r exp(G_r - G_s) k_r.k_s (s < r),
// P[r][s] = exp(G_r - G_s) q_r.k_s (s <= r), and T = (I + A)^-1 by forward substitution.
template <int DK>
__global__ void __launch_bounds__(kThreads)
    gdn_prepare_kernel(const float* __restrict__ qkv, const float* __restrict__ ab, const float* __restrict__ a_log,
                       const float* __restrict__ dt_bias, float* __restrict__ prep, __nv_bfloat16* __restrict__ qk16,
                       int tokens, int k_heads, int v_heads, int v_dim) {
    constexpr int KS = DK + 4, AS = C + 4;
    __shared__ __align__(16) float ks[C * KS], qs[C * KS];
    __shared__ __align__(16) float at[C * AS];  // A transposed: at[s][r] = A[r][s]
    __shared__ float gs[C], betas[C];
    const int h = blockIdx.x, ci = blockIdx.y, t0 = ci * C, tid = threadIdx.x;
    const int lane = tid % VELOX_WARP, warp = tid / VELOX_WARP;
    const int hv = v_heads / k_heads, kh = h / hv;
    const int n = tokens - t0 < C ? tokens - t0 : C;
    const size_t width = static_cast<size_t>(2 * k_heads * DK + v_heads * v_dim);
    float* out = prep + (static_cast<size_t>(h) * gridDim.y + ci) * kPrep;

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
        // The key head's first value head writes the bf16 copies (padded rows: zeros).
        __nv_bfloat16* q16 = h % hv == 0 ? qk16 + (static_cast<size_t>(kh) * gridDim.y + ci) * 2 * C * DK : nullptr;
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            const float4 qv = make_float4(q[j][0] * qn, q[j][1] * qn, q[j][2] * qn, q[j][3] * qn);
            const float4 kv = make_float4(k[j][0] * kn, k[j][1] * kn, k[j][2] * kn, k[j][3] * kn);
            *reinterpret_cast<float4*>(&qs[r * KS + dc * 4 + 32 * j]) = qv;
            *reinterpret_cast<float4*>(&ks[r * KS + dc * 4 + 32 * j]) = kv;
            if (q16 != nullptr) {
                *reinterpret_cast<uint2*>(q16 + r * DK + dc * 4 + 32 * j) = make_uint2(bf16x2(qv.x, qv.y), bf16x2(qv.z, qv.w));
                *reinterpret_cast<uint2*>(q16 + C * DK + r * DK + dc * 4 + 32 * j) =
                    make_uint2(bf16x2(kv.x, kv.y), bf16x2(kv.z, kv.w));
            }
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
    for (int idx = tid; idx < C * C; idx += kThreads) {
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

// The recurrence: a block owns 64 state rows of one head, chunks in order. Warp w holds rows
// 16 w .. 16 w + 15 as DK / 8 accumulator fragments; the chunk's bf16 q/k, f32 values and bf16
// T/P are staged in registers while the previous chunk computes, then published to shared memory.
template <int DK>
__global__ void __launch_bounds__(kThreads, 2)
    gdn_chunk_mma_kernel(const float* __restrict__ qkv, const float* __restrict__ prep,
                         const __nv_bfloat16* __restrict__ qk16, float* __restrict__ state, float* __restrict__ out,
                         int tokens, int k_heads, int v_heads, int v_dim) {
    constexpr int KS = DK + 8;  // bf16 row stride of the q/k tiles: ldmatrix rows on distinct banks
    constexpr int VS = VT + 1;  // f32 row stride of the value tile
    constexpr int TS = C + 8;   // bf16 row stride of T and P
    constexpr int NT = DK / 8;  // n-tiles over the state columns d
    constexpr int KK = DK / 16; // k steps over d
    constexpr int NQ = 2 * C * DK / 8 / kThreads;  // uint4 (8 bf16) of q | k per thread
    constexpr int NVV = C * VT / 4 / kThreads;     // float4 of values per thread
    static_assert(NQ >= 1 && NVV == 2 && C * C == 2 * kThreads, "staging split");
    __shared__ __align__(16) __nv_bfloat16 qs[C * KS], ks[C * KS];
    __shared__ __align__(16) __nv_bfloat16 ts[C * TS], ps[C * TS];
    __shared__ float vs[C * VS];
    __shared__ float gam[C], dl[C], bet[C];

    const int h = blockIdx.x, i0 = blockIdx.y * VT, tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int hv = v_heads / k_heads, kh = h / hv;
    const int chunks = (tokens + C - 1) / C;
    const size_t width = static_cast<size_t>(2 * k_heads * DK + v_heads * v_dim);
    const float inv_sqrt_dk = rsqrtf(static_cast<float>(DK));
    const int lr_a = 16 * warp + g, lr_b = lr_a + 8;  // this lane's rows within the block
    const int row_a = i0 + lr_a, row_b = i0 + lr_b;
    const bool ok_a = row_a < v_dim, ok_b = row_b < v_dim;

    // S as accumulator fragments: s[nt][0..1] = row g, d = 8 nt + 2t, 2t + 1; s[nt][2..3] = row g + 8.
    float s[NT][4];
    {
        const float* sa = state + (static_cast<size_t>(h) * v_dim + row_a) * DK + 2 * t;
        const float* sb = state + (static_cast<size_t>(h) * v_dim + row_b) * DK + 2 * t;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const float2 a = ok_a ? *reinterpret_cast<const float2*>(sa + 8 * nt) : make_float2(0.0f, 0.0f);
            const float2 b = ok_b ? *reinterpret_cast<const float2*>(sb + 8 * nt) : make_float2(0.0f, 0.0f);
            s[nt][0] = a.x, s[nt][1] = a.y, s[nt][2] = b.x, s[nt][3] = b.y;
        }
    }

    uint4 rqk[NQ];
    float4 rv[NVV];
    float rt[2], rp[2], rg = 0.0f;
    auto stage = [&](int ci) {
        const int t0 = ci * C, n = tokens - t0 < C ? tokens - t0 : C;
        const uint4* src = reinterpret_cast<const uint4*>(qk16 + (static_cast<size_t>(kh) * chunks + ci) * 2 * C * DK);
#pragma unroll
        for (int m = 0; m < NQ; ++m) rqk[m] = src[tid + kThreads * m];
#pragma unroll
        for (int m = 0; m < NVV; ++m) {
            const int e = tid + kThreads * m, c = e / (VT / 4), col = i0 + 4 * (e % (VT / 4));
            rv[m] = c < n && col + 3 < v_dim
                        ? *reinterpret_cast<const float4*>(qkv + static_cast<size_t>(t0 + c) * width +
                                                           2 * k_heads * DK + h * v_dim + col)
                        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        const float* pr = prep + (static_cast<size_t>(h) * chunks + ci) * kPrep;
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            rt[m] = pr[tid + kThreads * m];
            rp[m] = pr[C * C + tid + kThreads * m];
        }
        if (tid < 3 * C) rg = pr[2 * C * C + tid];
    };
    auto publish = [&]() {
#pragma unroll
        for (int m = 0; m < NQ; ++m) {
            const int e = tid + kThreads * m, per = C * DK / 8;
            const int which = e / per, rem = e % per, c = rem / (DK / 8), d8 = rem % (DK / 8);
            *reinterpret_cast<uint4*>((which ? ks : qs) + c * KS + 8 * d8) = rqk[m];
        }
#pragma unroll
        for (int m = 0; m < NVV; ++m) {
            const int e = tid + kThreads * m, c = e / (VT / 4), i4 = e % (VT / 4);
            float* d = vs + c * VS + 4 * i4;
            d[0] = rv[m].x, d[1] = rv[m].y, d[2] = rv[m].z, d[3] = rv[m].w;
        }
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            const int e = tid + kThreads * m, r = e / C, c = e % C;
            ts[r * TS + c] = __float2bfloat16_rn(rt[m]);
            ps[r * TS + c] = __float2bfloat16_rn(rp[m]);
        }
        if (tid < C) gam[tid] = rg;
        else if (tid < 2 * C) dl[tid - C] = rg;
        else if (tid < 3 * C) bet[tid - 2 * C] = rg;
    };
    stage(0);

    for (int ci = 0; ci < chunks; ++ci) {
        publish();
        __syncthreads();
        if (ci + 1 < chunks) stage(ci + 1);
        const int t0 = ci * C, n = tokens - t0 < C ? tokens - t0 : C;

        // Y = S [q | k]^T: y[ct] / x[ct] hold (row g | g + 8, c = 8 ct + 2t, 2t + 1).
        float y[2][4], x[2][4];
#pragma unroll
        for (int ct = 0; ct < 2; ++ct)
#pragma unroll
            for (int j = 0; j < 4; ++j) y[ct][j] = x[ct][j] = 0.0f;
#pragma unroll
        for (int kk = 0; kk < KK; ++kk) {
            uint32_t a[4];
            frag_a(a, s[2 * kk], s[2 * kk + 1]);
#pragma unroll
            for (int ct = 0; ct < 2; ++ct) {
                const __nv_bfloat16* qr = qs + (8 * ct + g) * KS + 16 * kk + 2 * t;
                const __nv_bfloat16* kr = ks + (8 * ct + g) * KS + 16 * kk + 2 * t;
                mma_bf16(y[ct], a[0], a[1], a[2], a[3], *reinterpret_cast<const uint32_t*>(qr),
                         *reinterpret_cast<const uint32_t*>(qr + 8));
                mma_bf16(x[ct], a[0], a[1], a[2], a[3], *reinterpret_cast<const uint32_t*>(kr),
                         *reinterpret_cast<const uint32_t*>(kr + 8));
            }
        }
        // R[c][row] = beta_c (v_c[row] - gamma_c x[row][c]), in x's layout.
        float rr[2][4];
#pragma unroll
        for (int ct = 0; ct < 2; ++ct)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = 8 * ct + 2 * t + (j & 1), lr = j < 2 ? lr_a : lr_b;
                rr[ct][j] = bet[c] * (vs[c * VS + lr] - gam[c] * x[ct][j]);
            }
        // D^T = R^T T^T (rows, k = c, n = s): dt[st] holds (row, s = 8 st + 2t, 2t + 1).
        float dt[2][4];
        {
            uint32_t a[4];
            frag_a(a, rr[0], rr[1]);
#pragma unroll
            for (int st = 0; st < 2; ++st) {
                dt[st][0] = dt[st][1] = dt[st][2] = dt[st][3] = 0.0f;
                const __nv_bfloat16* tr = ts + (8 * st + g) * TS + 2 * t;  // T[s][c' = 2t, 2t + 1]
                mma_bf16(dt[st], a[0], a[1], a[2], a[3], *reinterpret_cast<const uint32_t*>(tr),
                         *reinterpret_cast<const uint32_t*>(tr + 8));
            }
        }
        // o^T = gamma_r y + D^T P^T (rows, k = s, n = r), scaled by 1 / sqrt(DK).
        {
            uint32_t a[4];
            frag_a(a, dt[0], dt[1]);
#pragma unroll
            for (int rt2 = 0; rt2 < 2; ++rt2) {
                float o[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) o[j] = gam[8 * rt2 + 2 * t + (j & 1)] * y[rt2][j];
                const __nv_bfloat16* pr = ps + (8 * rt2 + g) * TS + 2 * t;  // P[r][s = 2t, 2t + 1]
                mma_bf16(o, a[0], a[1], a[2], a[3], *reinterpret_cast<const uint32_t*>(pr),
                         *reinterpret_cast<const uint32_t*>(pr + 8));
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int r = 8 * rt2 + 2 * t + (j & 1), row = j < 2 ? row_a : row_b;
                    if (r < n && (j < 2 ? ok_a : ok_b))
                        out[(static_cast<size_t>(t0 + r) * v_heads + h) * v_dim + row] = o[j] * inv_sqrt_dk;
                }
            }
        }
        // S = gamma_last S + (D^T diag(decay_last)) K (rows, k = s, n = d).
        {
            const float gl = gam[C - 1];
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                for (int j = 0; j < 4; ++j) s[nt][j] *= gl;
            const float w0 = dl[2 * t], w1 = dl[2 * t + 1], w8 = dl[8 + 2 * t], w9 = dl[9 + 2 * t];
            const uint32_t a0 = bf16x2(dt[0][0] * w0, dt[0][1] * w1), a1 = bf16x2(dt[0][2] * w0, dt[0][3] * w1);
            const uint32_t a2 = bf16x2(dt[1][0] * w8, dt[1][1] * w9), a3 = bf16x2(dt[1][2] * w8, dt[1][3] * w9);
            const int krow = (lane & 7) + 8 * ((lane >> 3) & 1), kcol = 8 * (lane >> 4);
#pragma unroll
            for (int np = 0; np < NT / 2; ++np) {
                uint32_t b[4];
                ldmatrix_x4_trans(b, ks + krow * KS + 16 * np + kcol);
                mma_bf16(s[2 * np], a0, a1, a2, a3, b[0], b[1]);
                mma_bf16(s[2 * np + 1], a0, a1, a2, a3, b[2], b[3]);
            }
        }
        __syncthreads();  // the tiles are free for the next publish
    }

    float* sa = state + (static_cast<size_t>(h) * v_dim + row_a) * DK + 2 * t;
    float* sb = state + (static_cast<size_t>(h) * v_dim + row_b) * DK + 2 * t;
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        if (ok_a) *reinterpret_cast<float2*>(sa + 8 * nt) = make_float2(s[nt][0], s[nt][1]);
        if (ok_b) *reinterpret_cast<float2*>(sb + 8 * nt) = make_float2(s[nt][2], s[nt][3]);
    }
}

template <int DK>
int run(const float* qkv, const float* ab, const float* a_log, const float* dt_bias, float* state, float* out,
        int tokens, int k_heads, int v_heads, int v_dim) {
    const int chunks = (tokens + C - 1) / C;
    static VeloxScratch scratch;
    const size_t prep_bytes = (static_cast<size_t>(v_heads) * chunks * kPrep * sizeof(float) + 255) / 256 * 256;
    const size_t qk_bytes = static_cast<size_t>(k_heads) * chunks * 2 * C * DK * sizeof(__nv_bfloat16);
    uint8_t* base = static_cast<uint8_t*>(scratch.get(prep_bytes + qk_bytes));
    if (base == nullptr) return static_cast<int>(cudaErrorMemoryAllocation);
    float* prep = reinterpret_cast<float*>(base);
    __nv_bfloat16* qk16 = reinterpret_cast<__nv_bfloat16*>(base + prep_bytes);
    gdn_prepare_kernel<DK><<<dim3(v_heads, chunks), kThreads>>>(qkv, ab, a_log, dt_bias, prep, qk16, tokens, k_heads,
                                                                v_heads, v_dim);
    gdn_chunk_mma_kernel<DK><<<dim3(v_heads, (v_dim + VT - 1) / VT), kThreads>>>(qkv, prep, qk16, state, out, tokens,
                                                                                 k_heads, v_heads, v_dim);
    return velox_launch_status();
}

}  // namespace

// Gated DeltaNet over `tokens` rows of qkv / ab (main's layouts), state updated in place from
// state_in (may alias state), outputs to out [tokens, v_heads, v_dim]. Prefill widths only
// (tokens >= 16), k_dim 32 / 64 / 128, v_dim a multiple of 4; VELOX_ERR_UNSUPPORTED otherwise.
VELOX_API int velox_4090_gdn_chunk(const float* qkv, const float* ab, const float* a_log, const float* dt_bias,
                                   const float* state_in, float* state, float* out, int tokens, int k_heads,
                                   int v_heads, int k_dim, int v_dim) {
    if (tokens < C || v_dim % 4 != 0 || v_dim <= 0 || k_heads <= 0 || v_heads % k_heads != 0 ||
        (k_dim != 32 && k_dim != 64 && k_dim != 128))
        return VELOX_ERR_UNSUPPORTED;
    if (state_in != state) {
        const size_t n = static_cast<size_t>(v_heads) * v_dim * k_dim;
        if (cudaError_t e = cudaMemcpyAsync(state, state_in, n * sizeof(float), cudaMemcpyDeviceToDevice, 0))
            return static_cast<int>(e);
    }
    if (k_dim == 32) return run<32>(qkv, ab, a_log, dt_bias, state, out, tokens, k_heads, v_heads, v_dim);
    if (k_dim == 64) return run<64>(qkv, ab, a_log, dt_bias, state, out, tokens, k_heads, v_heads, v_dim);
    return run<128>(qkv, ab, a_log, dt_bias, state, out, tokens, k_heads, v_heads, v_dim);
}
