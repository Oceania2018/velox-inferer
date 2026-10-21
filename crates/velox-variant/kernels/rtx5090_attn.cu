// RTX 5090 causal prefill attention (FlashAttention-2 schedule) for head_dim 256.
//
// Precision (velox_core::AttnPrecision::Fp8QkF16Pv): Q and K are rounded per head vector to E4M3
// (scale = amax / 448) and the scores use block-scaled FP8 MMA with unit UE8M0 scales, which runs
// at full rate with FP32 accumulation on sm_120a. V and the softmax weights are f16 for the value
// product. The caches stay f32; each call converts the prefix it reads.
//
// Block: one query head x 64 query tokens, 4 warps of 16 rows. Key tiles of 32 stream through a
// two-stage cp.async pipeline; K rows are 256 bytes and V rows 512 bytes in shared memory, with
// 16-byte chunks XOR-swizzled by (row & 7) so ldmatrix over 8 rows is conflict-free.
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#include <mutex>

#include "velox_common.cuh"

namespace {

constexpr int kD = 256;
constexpr int kBM = 64;
constexpr int kBN = 32;
constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src),
                 "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_u32(p)));
}

__device__ __forceinline__ void mma_fp8(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0x7fu), "h"((uint16_t)0),
          "h"((uint16_t)0), "r"(0x7fu), "h"((uint16_t)0), "h"((uint16_t)0));
}

__device__ __forceinline__ void mma_f16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&h);
}

// ---------------------------------------------------------------------------------------------
// Operand conversion. One warp per head vector of 256 values.

// codes[row] = e4m3(x * (1 / s)), s = amax / 448; scale[row] = s * mul.
__global__ void rows_to_fp8(const float* __restrict__ x, long long rows, float mul, uint8_t* __restrict__ codes,
                            float* __restrict__ scale) {
    const long long row = static_cast<long long>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (row >= rows) return;
    const float4* src = reinterpret_cast<const float4*>(x + row * kD) + lane * 2;
    const float4 a = src[0], b = src[1];
    float amax = fmaxf(fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(a.z), fabsf(a.w))),
                       fmaxf(fmaxf(fabsf(b.x), fabsf(b.y)), fmaxf(fabsf(b.z), fabsf(b.w))));
    amax = velox_warp_max(amax);
    const float s = __fdiv_rn(amax, 448.0f);
    const float inv = s > 0.0f ? __fdiv_rn(1.0f, s) : 0.0f;
    const float v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    uint32_t w[2] = {0, 0};
#pragma unroll
    for (int i = 0; i < 8; ++i)
        w[i / 4] |= static_cast<uint32_t>(__nv_cvt_float_to_fp8(__fmul_rn(v[i], inv), __NV_SATFINITE, __NV_E4M3))
                    << (8 * (i % 4));
    *reinterpret_cast<uint2*>(codes + row * kD + lane * 8) = make_uint2(w[0], w[1]);
    if (lane == 0) scale[row] = s * mul;
}

__global__ void to_f16(const float* __restrict__ x, long long n4, __half* __restrict__ y) {
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n4;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const float4 t = reinterpret_cast<const float4*>(x)[i];
        reinterpret_cast<__half2*>(y)[2 * i] = __floats2half2_rn(t.x, t.y);
        reinterpret_cast<__half2*>(y)[2 * i + 1] = __floats2half2_rn(t.z, t.w);
    }
}

// ---------------------------------------------------------------------------------------------

struct Smem {
    uint8_t k[2][kBN * kD];         // E4M3, 256 B per key
    __half v[2][kBN * kD];          // 512 B per key
    float ks[2][kBN];               // K row scales
};

__device__ __forceinline__ int k_off(int row, int byte) {  // 16 chunks per row
    return row * kD + ((((byte >> 4) ^ row) & 7) | ((byte >> 4) & 8)) * 16 + (byte & 15);
}
__device__ __forceinline__ int v_off(int row, int col) {  // col in halves; 32 chunks per row
    const int chunk = col >> 3;
    return row * kD + (((chunk ^ row) & 7) | (chunk & ~7)) * 8 + (col & 7);
}

struct AttnArgs {
    const uint8_t* q;   // [tokens, heads, D] E4M3
    const float* qs;    // [tokens * heads] scale * softmax_scale * log2(e)
    const uint8_t* k;   // [kv_len, kv_heads, D] E4M3
    const float* ks;    // [kv_len * kv_heads]
    const __half* v;    // [kv_len, kv_heads, D]
    float* out;         // [tokens, heads, D]
    int tokens, heads, kv_heads, pos0, kv_len;
};

__device__ __forceinline__ void load_tile(Smem& s, const AttnArgs& p, int stage, int kt, int hk) {
    const int t = threadIdx.x;
    const int key0 = kt * kBN;
#pragma unroll
    for (int i = 0; i < (kBN * kD / 16) / kThreads; ++i) {  // K: 512 chunks
        const int idx = t + i * kThreads, row = idx >> 4, c = idx & 15;
        const int key = key0 + row;
        const bool ok = key < p.kv_len;
        cp_async16(s.k[stage] + k_off(row, c * 16),
                   p.k + (static_cast<long long>(ok ? key : 0) * p.kv_heads + hk) * kD + c * 16, ok);
    }
#pragma unroll
    for (int i = 0; i < (kBN * kD * 2 / 16) / kThreads; ++i) {  // V: 1024 chunks
        const int idx = t + i * kThreads, row = idx >> 5, c = idx & 31;
        const int key = key0 + row;
        const bool ok = key < p.kv_len;
        cp_async16(s.v[stage] + v_off(row, c * 8),
                   p.v + (static_cast<long long>(ok ? key : 0) * p.kv_heads + hk) * kD + c * 8, ok);
    }
    if (t < kBN) {
        const int key = key0 + t;
        s.ks[stage][t] = key < p.kv_len ? p.ks[static_cast<long long>(key) * p.kv_heads + hk] : 0.0f;
    }
}

__global__ void __launch_bounds__(kThreads, 2) flash_prefill_kernel(AttnArgs p) {
    extern __shared__ __align__(128) uint8_t smem_raw[];
    Smem& s = *reinterpret_cast<Smem*>(smem_raw);
    const int h = blockIdx.y, hk = h / (p.heads / p.kv_heads);
    const int t0 = blockIdx.x * kBM;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, q4 = lane & 3;

    // Q fragments for this warp's 16 rows, all 8 k-steps of 32 bytes, straight from global.
    const int r0 = t0 + warp * 16 + g, r1 = r0 + 8;
    uint32_t qf[8][4];
    {
        const uint8_t* q0 = p.q + (static_cast<long long>(r0 < p.tokens ? r0 : 0) * p.heads + h) * kD;
        const uint8_t* q1 = p.q + (static_cast<long long>(r1 < p.tokens ? r1 : 0) * p.heads + h) * kD;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            qf[ks][0] = r0 < p.tokens ? *reinterpret_cast<const uint32_t*>(q0 + ks * 32 + 4 * q4) : 0u;
            qf[ks][1] = r1 < p.tokens ? *reinterpret_cast<const uint32_t*>(q1 + ks * 32 + 4 * q4) : 0u;
            qf[ks][2] = r0 < p.tokens ? *reinterpret_cast<const uint32_t*>(q0 + ks * 32 + 16 + 4 * q4) : 0u;
            qf[ks][3] = r1 < p.tokens ? *reinterpret_cast<const uint32_t*>(q1 + ks * 32 + 16 + 4 * q4) : 0u;
        }
    }
    const float qs0 = r0 < p.tokens ? p.qs[static_cast<long long>(r0) * p.heads + h] : 0.0f;
    const float qs1 = r1 < p.tokens ? p.qs[static_cast<long long>(r1) * p.heads + h] : 0.0f;
    const int qpos0 = p.pos0 + r0, qpos1 = p.pos0 + r1;

    float o[32][4];
#pragma unroll
    for (int i = 0; i < 32; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;

    const int last = min(p.tokens, t0 + kBM) - 1;
    const int n_keys = min(p.kv_len, p.pos0 + last + 1);
    const int n_tiles = (n_keys + kBN - 1) / kBN;

    load_tile(s, p, 0, 0, hk);
    cp_commit();
    for (int kt = 0; kt < n_tiles; ++kt) {
        const int st = kt & 1;
        if (kt + 1 < n_tiles) {
            load_tile(s, p, st ^ 1, kt + 1, hk);
            cp_commit();
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();

        // S = Q K^T for 16 rows x 32 keys.
        float sc[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i) sc[i][0] = sc[i][1] = sc[i][2] = sc[i][3] = 0.0f;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
#pragma unroll
            for (int np = 0; np < 2; ++np) {
                uint32_t b[4];
                const int row = np * 16 + (lane & 7) + ((lane >> 4) << 3);
                ldsm_x4(b, s.k[st] + k_off(row, ks * 32 + (((lane >> 3) & 1) << 4)));
                mma_fp8(sc[2 * np], qf[ks], b[0], b[1]);
                mma_fp8(sc[2 * np + 1], qf[ks], b[2], b[3]);
            }
        }

        // Scale, mask, online softmax (base 2). Rows g (c0, c1) and g + 8 (c2, c3).
        const int key0 = kt * kBN;
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int j = nt * 8 + 2 * q4 + e, key = key0 + j;
                const float kscale = s.ks[st][j];
                sc[nt][e] = key <= qpos0 ? sc[nt][e] * qs0 * kscale : -INFINITY;
                sc[nt][2 + e] = key <= qpos1 ? sc[nt][2 + e] * qs1 * kscale : -INFINITY;
                mx0 = fmaxf(mx0, sc[nt][e]);
                mx1 = fmaxf(mx1, sc[nt][2 + e]);
            }
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
        // A row with every key masked so far keeps m = -inf; use 0 as its reference.
        const float ref0 = mn0 == -INFINITY ? 0.0f : mn0, ref1 = mn1 == -INFINITY ? 0.0f : mn1;
        const float al0 = exp2f(m0 - ref0), al1 = exp2f(m1 - ref1);
        m0 = mn0, m1 = mn1;
        float ps0 = 0.0f, ps1 = 0.0f;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            sc[nt][0] = exp2f(sc[nt][0] - ref0);
            sc[nt][1] = exp2f(sc[nt][1] - ref0);
            sc[nt][2] = exp2f(sc[nt][2] - ref1);
            sc[nt][3] = exp2f(sc[nt][3] - ref1);
            ps0 += sc[nt][0] + sc[nt][1];
            ps1 += sc[nt][2] + sc[nt][3];
        }
        l0 = l0 * al0 + ps0;
        l1 = l1 * al1 + ps1;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            o[i][0] *= al0, o[i][1] *= al0;
            o[i][2] *= al1, o[i][3] *= al1;
        }

        // O += P V: two k-steps of 16 keys; P becomes the f16 A operand directly.
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const uint32_t a[4] = {pack_h2(sc[2 * j][0], sc[2 * j][1]), pack_h2(sc[2 * j][2], sc[2 * j][3]),
                                   pack_h2(sc[2 * j + 1][0], sc[2 * j + 1][1]),
                                   pack_h2(sc[2 * j + 1][2], sc[2 * j + 1][3])};
            const int vrow = j * 16 + (lane & 7) + (((lane >> 3) & 1) << 3);
#pragma unroll
            for (int dn = 0; dn < 16; ++dn) {
                uint32_t b[4];
                ldsm_x4_t(b, s.v[st] + v_off(vrow, dn * 16 + ((lane >> 4) << 3)));
                mma_f16(o[2 * dn], a, b[0], b[1]);
                mma_f16(o[2 * dn + 1], a, b[2], b[3]);
            }
        }
        __syncthreads();
    }

    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float inv0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, inv1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
#pragma unroll
    for (int nt = 0; nt < 32; ++nt) {
        const int d = nt * 8 + 2 * q4;
        if (r0 < p.tokens)
            *reinterpret_cast<float2*>(p.out + (static_cast<long long>(r0) * p.heads + h) * kD + d) =
                make_float2(o[nt][0] * inv0, o[nt][1] * inv0);
        if (r1 < p.tokens)
            *reinterpret_cast<float2*>(p.out + (static_cast<long long>(r1) * p.heads + h) * kD + d) =
                make_float2(o[nt][2] * inv1, o[nt][3] * inv1);
    }
}

// Grow-only and never freed (VeloxScratch): recorded CUDA graphs may still reference old buffers.
struct Scratch {
    VeloxScratch s;
    uint8_t* get(size_t need) { return static_cast<uint8_t*>(s.get(need)); }
};
Scratch g_attn_scratch;

size_t align_up(size_t x) { return (x + 255) / 256 * 256; }

}  // namespace

// Causal attention for head_dim 256 (see velox_core::backend::AttnShape). Returns
// VELOX_ERR_UNSUPPORTED for other geometries.
VELOX_API int velox_5090_attention_prefill(const float* q, const float* k, const float* v, float* out, int tokens,
                                           int heads, int kv_heads, int head_dim, int pos0, float scale) {
    if (head_dim != kD || tokens <= 0 || kv_heads <= 0 || heads % kv_heads != 0) return VELOX_ERR_UNSUPPORTED;
    const long long kv_len = static_cast<long long>(pos0) + tokens;
    const long long q_rows = static_cast<long long>(tokens) * heads, k_rows = kv_len * kv_heads;
    const size_t qb = align_up(q_rows * kD), qsb = align_up(q_rows * 4), kb = align_up(k_rows * kD),
                 ksb = align_up(k_rows * 4), vb = align_up(k_rows * kD * 2);
    uint8_t* base = g_attn_scratch.get(qb + qsb + kb + ksb + vb);
    if (!base) return static_cast<int>(cudaErrorMemoryAllocation);
    uint8_t* qc = base;
    float* qs = reinterpret_cast<float*>(base + qb);
    uint8_t* kc = base + qb + qsb;
    float* ks = reinterpret_cast<float*>(base + qb + qsb + kb);
    __half* vh = reinterpret_cast<__half*>(base + qb + qsb + kb + ksb);

    constexpr float kLog2e = 1.4426950408889634f;
    rows_to_fp8<<<static_cast<unsigned>((q_rows + 7) / 8), 256>>>(q, q_rows, scale * kLog2e, qc, qs);
    rows_to_fp8<<<static_cast<unsigned>((k_rows + 7) / 8), 256>>>(k, k_rows, 1.0f, kc, ks);
    to_f16<<<velox_grid_1d(k_rows * kD / 4, 256), 256>>>(v, k_rows * kD / 4, vh);
    if (const int e = velox_launch_status()) return e;

    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(flash_prefill_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Smem));
        configured = true;
    }
    AttnArgs p{qc, qs, kc, ks, vh, out, tokens, heads, kv_heads, pos0, static_cast<int>(kv_len)};
    const dim3 grid((tokens + kBM - 1) / kBM, heads);
    flash_prefill_kernel<<<grid, kThreads, sizeof(Smem)>>>(p);
    return velox_launch_status();
}
