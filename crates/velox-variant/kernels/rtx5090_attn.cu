// RTX 5090 causal prefill attention (FlashAttention-2 schedule) for head_dim 256, and the same
// schedule over f16 operands for System One scoring windows (at the end of the file).
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

__device__ __forceinline__ void mma_f16_h(uint32_t (&c)[2], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c[0]), "+r"(c[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// kTileH16: the value product accumulates each 32-key tile in f16 (full rate, 210 vs 105 TFLOPS
// on this card) and adds the tile's sum to the f32 output; P and V stay f16.
template <bool kTileH16>
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
        if constexpr (kTileH16) {
            uint32_t a[2][4];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                a[j][0] = pack_h2(sc[2 * j][0], sc[2 * j][1]);
                a[j][1] = pack_h2(sc[2 * j][2], sc[2 * j][3]);
                a[j][2] = pack_h2(sc[2 * j + 1][0], sc[2 * j + 1][1]);
                a[j][3] = pack_h2(sc[2 * j + 1][2], sc[2 * j + 1][3]);
            }
#pragma unroll
            for (int dn = 0; dn < 16; ++dn) {
                uint32_t h0[2] = {0u, 0u}, h1[2] = {0u, 0u};
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    uint32_t b[4];
                    ldsm_x4_t(b, s.v[st] + v_off(j * 16 + (lane & 7) + (((lane >> 3) & 1) << 3),
                                                 dn * 16 + ((lane >> 4) << 3)));
                    mma_f16_h(h0, a[j], b[0], b[1]);
                    mma_f16_h(h1, a[j], b[2], b[3]);
                }
                const float2 x0 = __half22float2(*reinterpret_cast<const __half2*>(&h0[0]));
                const float2 y0 = __half22float2(*reinterpret_cast<const __half2*>(&h0[1]));
                const float2 x1 = __half22float2(*reinterpret_cast<const __half2*>(&h1[0]));
                const float2 y1 = __half22float2(*reinterpret_cast<const __half2*>(&h1[1]));
                o[2 * dn][0] += x0.x, o[2 * dn][1] += x0.y, o[2 * dn][2] += y0.x, o[2 * dn][3] += y0.y;
                o[2 * dn + 1][0] += x1.x, o[2 * dn + 1][1] += x1.y;
                o[2 * dn + 1][2] += y1.x, o[2 * dn + 1][3] += y1.y;
            }
        } else {
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

// ---------------------------------------------------------------------------------------------
// FP8 value product (VELOX_ATTN_FP8PV=1). The value product above runs f16 x f16 -> f32, which is
// a quarter of the FP8 rate on this card (105 vs 417 TFLOPS, block-scaled e4m3 with f32
// accumulation). Here it is FP8 too:
//   P: exp2(s - m + 8), i.e. the softmax weights times 256 (max 256, so E4M3 keeps weights down
//      to ~2^-17 of the row maximum); the row sums use the same scaled values, so the scale cancels.
//   V: E4M3 per channel (kv head x dim) over the whole prefix, s_d = amax_d / 448, applied to the
//      output once at the end.
// Fragments: thread (g, q4) holds scores of keys {2q4, 2q4+1, 8+2q4, 9+2q4} (+16 for the second
// half) of each 32-key tile. The A fragment of m16n8k32 wants k = 4q4..4q4+3, so k slot s maps to
// key kperm(s) and V is stored transposed per tile in that key order: [kv_head][tile][dim][32].

__host__ __device__ __forceinline__ int kperm(int s) {
    const int h = s & 16, r = s & 15, q = r >> 2, i = r & 3;
    return h + (i < 2 ? 2 * q + i : 8 + 2 * q + (i - 2));
}

// amax per (kv_head, dim) over every key, as float bits (non-negative, so integer max works).
__global__ void v_channel_amax(const float* __restrict__ v, long long kv_len, int kv_heads,
                               unsigned* __restrict__ amax_bits) {
    const long long n4 = kv_len * kv_heads * (kD / 4);
    __shared__ unsigned local[8 * kD];  // up to 8 kv heads
    for (int i = threadIdx.x; i < kv_heads * kD; i += blockDim.x) local[i] = 0u;
    __syncthreads();
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n4;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const float4 t = reinterpret_cast<const float4*>(v)[i];
        const int c = static_cast<int>(i % (kv_heads * (kD / 4))) * 4;  // kv_head * kD + dim
        atomicMax(&local[c + 0], __float_as_uint(fabsf(t.x)));
        atomicMax(&local[c + 1], __float_as_uint(fabsf(t.y)));
        atomicMax(&local[c + 2], __float_as_uint(fabsf(t.z)));
        atomicMax(&local[c + 3], __float_as_uint(fabsf(t.w)));
    }
    __syncthreads();
    for (int i = threadIdx.x; i < kv_heads * kD; i += blockDim.x)
        if (local[i]) atomicMax(&amax_bits[i], local[i]);
}

// vt[hk][tile][d][s] = e4m3(v[tile * 32 + kperm(s)][hk][d] / s_d); keys past kv_len are 0.
// sv[hk * kD + d] = s_d. One block per (tile, kv_head).
__global__ void v_to_fp8_t(const float* __restrict__ v, long long kv_len, int kv_heads,
                           const unsigned* __restrict__ amax_bits, uint8_t* __restrict__ vt,
                           float* __restrict__ sv) {
    const long long tile = blockIdx.x;
    const int hk = blockIdx.y;
    __shared__ float inv[kD];
    for (int d = threadIdx.x; d < kD; d += blockDim.x) {
        const float sd = __fdiv_rn(__uint_as_float(amax_bits[hk * kD + d]), 448.0f);
        inv[d] = sd > 0.0f ? __fdiv_rn(1.0f, sd) : 0.0f;
        if (tile == 0) sv[hk * kD + d] = sd;
    }
    __syncthreads();
    uint8_t* dst = vt + (static_cast<long long>(hk) * gridDim.x + tile) * (kD * 32);
    // Thread w writes slots 4(w & 7) .. +3 of dim w >> 3 as one word; reads of a key are
    // coalesced across consecutive dims.
    for (int w = threadIdx.x; w < kD * 8; w += blockDim.x) {
        const int d = w & (kD - 1), s0 = (w >> 8) * 4;
        uint32_t word = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const long long key = tile * 32 + kperm(s0 + i);
            const float x = key < kv_len ? v[(key * kv_heads + hk) * kD + d] * inv[d] : 0.0f;
            word |= static_cast<uint32_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3)) << (8 * i);
        }
        *reinterpret_cast<uint32_t*>(dst + d * 32 + s0) = word;
    }
}

struct Smem8 {
    uint8_t k[2][kBN * kD];   // E4M3, 256 B per key
    uint8_t vt[2][kD * kBN];  // E4M3, V transposed: 32 B per dim (keys in kperm order)
    float ks[2][kBN];
};

// 32-byte rows of vt: the two 16-byte chunks swap on every fourth row, so ldmatrix over 8 rows
// (256 bytes) touches every bank once.
__device__ __forceinline__ int vt_off(int d, int byte) {
    return d * kBN + ((((byte >> 4) ^ (d >> 2)) & 1) << 4) + (byte & 15);
}

struct Attn8Args {
    const uint8_t* q;
    const float* qs;
    const uint8_t* k;
    const float* ks;
    const uint8_t* vt;  // [kv_heads][tiles][kD][32]
    const float* sv;    // [kv_heads * kD]
    float* out;
    int tokens, heads, kv_heads, pos0, kv_len, tiles;
};

__device__ __forceinline__ void load_tile8(Smem8& s, const Attn8Args& p, int stage, int kt, int hk) {
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
    const uint8_t* src = p.vt + (static_cast<long long>(hk) * p.tiles + kt) * (kD * kBN);
#pragma unroll
    for (int i = 0; i < (kD * kBN / 16) / kThreads; ++i) {  // V^T: 512 chunks
        const int idx = t + i * kThreads, d = idx >> 1, c = idx & 1;
        cp_async16(s.vt[stage] + vt_off(d, c * 16), src + d * kBN + c * 16, true);
    }
    if (t < kBN) {
        const int key = key0 + t;
        s.ks[stage][t] = key < p.kv_len ? p.ks[static_cast<long long>(key) * p.kv_heads + hk] : 0.0f;
    }
}

__device__ __forceinline__ uint32_t pack_e4m3x4(float a, float b, float c, float d) {
    const uint32_t lo = __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
    const uint32_t hi = __nv_cvt_float2_to_fp8x2(make_float2(c, d), __NV_SATFINITE, __NV_E4M3);
    return lo | (hi << 16);
}

__global__ void __launch_bounds__(kThreads, 2) flash_prefill_fp8pv_kernel(Attn8Args p) {
    extern __shared__ __align__(128) uint8_t smem_raw[];
    Smem8& s = *reinterpret_cast<Smem8*>(smem_raw);
    const int h = blockIdx.y, hk = h / (p.heads / p.kv_heads);
    const int t0 = blockIdx.x * kBM;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, q4 = lane & 3;

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

    load_tile8(s, p, 0, 0, hk);
    cp_commit();
    for (int kt = 0; kt < n_tiles; ++kt) {
        const int st = kt & 1;
        if (kt + 1 < n_tiles) {
            load_tile8(s, p, st ^ 1, kt + 1, hk);
            cp_commit();
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();

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
        const float ref0 = mn0 == -INFINITY ? 0.0f : mn0, ref1 = mn1 == -INFINITY ? 0.0f : mn1;
        const float al0 = exp2f(m0 - ref0), al1 = exp2f(m1 - ref1);
        m0 = mn0, m1 = mn1;
        // Weights times 256 (exponent + 8); the row sums carry the same factor.
        const float b0 = ref0 - 8.0f, b1 = ref1 - 8.0f;
        float ps0 = 0.0f, ps1 = 0.0f;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            sc[nt][0] = exp2f(sc[nt][0] - b0);
            sc[nt][1] = exp2f(sc[nt][1] - b0);
            sc[nt][2] = exp2f(sc[nt][2] - b1);
            sc[nt][3] = exp2f(sc[nt][3] - b1);
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

        // O += P V over the tile's 32 keys in one k-step: slots 4q4.. hold keys kperm(4q4..).
        const uint32_t pa[4] = {pack_e4m3x4(sc[0][0], sc[0][1], sc[1][0], sc[1][1]),
                                pack_e4m3x4(sc[0][2], sc[0][3], sc[1][2], sc[1][3]),
                                pack_e4m3x4(sc[2][0], sc[2][1], sc[3][0], sc[3][1]),
                                pack_e4m3x4(sc[2][2], sc[2][3], sc[3][2], sc[3][3])};
#pragma unroll
        for (int dn = 0; dn < 32; dn += 2) {
            uint32_t b[4];
            ldsm_x4(b, s.vt[st] + vt_off(dn * 8 + (lane & 7) + ((lane >> 4) << 3), ((lane >> 3) & 1) << 4));
            mma_fp8(o[dn], pa, b[0], b[1]);
            mma_fp8(o[dn + 1], pa, b[2], b[3]);
        }
        __syncthreads();
    }

    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float inv0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, inv1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
    const float* sv = p.sv + hk * kD;
#pragma unroll
    for (int nt = 0; nt < 32; ++nt) {
        const int d = nt * 8 + 2 * q4;
        const float s0 = sv[d], s1 = sv[d + 1];
        if (r0 < p.tokens)
            *reinterpret_cast<float2*>(p.out + (static_cast<long long>(r0) * p.heads + h) * kD + d) =
                make_float2(o[nt][0] * inv0 * s0, o[nt][1] * inv0 * s1);
        if (r1 < p.tokens)
            *reinterpret_cast<float2*>(p.out + (static_cast<long long>(r1) * p.heads + h) * kD + d) =
                make_float2(o[nt][2] * inv1 * s0, o[nt][3] * inv1 * s1);
    }
}

// VELOX_ATTN_FP8PV: unset or 0 = f16 x f16 -> f32 value product; 1 = FP8 (above); 2 = f16 with
// per-tile f16 accumulation (flash_prefill_kernel<true>).
int pv_mode() {
    static const int mode = [] {
        const char* e = getenv("VELOX_ATTN_FP8PV");
        return e == nullptr || e[0] == '\0' ? 0 : atoi(e);
    }();
    return mode;
}
bool fp8_pv_enabled() { return pv_mode() == 1; }

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
    if (fp8_pv_enabled() && kv_heads <= 8) {
        // q/k as below; V as transposed E4M3 tiles with per-channel scales (vb has room: 2 bytes
        // per value there, 1 here, plus the scales and channel maxima).
        const long long tiles = (kv_len + kBN - 1) / kBN;
        const size_t vtb = align_up(static_cast<size_t>(kv_heads) * tiles * kD * kBN),
                     svb = align_up(kv_heads * kD * 4);
        uint8_t* base8 = g_attn_scratch.get(qb + qsb + kb + ksb + vtb + 2 * svb);
        if (!base8) return static_cast<int>(cudaErrorMemoryAllocation);
        uint8_t* qc8 = base8;
        float* qs8 = reinterpret_cast<float*>(base8 + qb);
        uint8_t* kc8 = base8 + qb + qsb;
        float* ks8 = reinterpret_cast<float*>(base8 + qb + qsb + kb);
        uint8_t* vt = base8 + qb + qsb + kb + ksb;
        float* sv = reinterpret_cast<float*>(vt + vtb);
        unsigned* amax = reinterpret_cast<unsigned*>(vt + vtb + svb);
        rows_to_fp8<<<static_cast<unsigned>((q_rows + 7) / 8), 256>>>(q, q_rows, scale * kLog2e, qc8, qs8);
        rows_to_fp8<<<static_cast<unsigned>((k_rows + 7) / 8), 256>>>(k, k_rows, 1.0f, kc8, ks8);
        if (cudaError_t e = cudaMemsetAsync(amax, 0, kv_heads * kD * 4, 0)) return static_cast<int>(e);
        v_channel_amax<<<velox_grid_1d(k_rows * kD / 4, 256), 256>>>(v, kv_len, kv_heads, amax);
        v_to_fp8_t<<<dim3(static_cast<unsigned>(tiles), kv_heads), 256>>>(v, kv_len, kv_heads, amax, vt, sv);
        if (const int e = velox_launch_status()) return e;
        static bool configured8 = false;
        if (!configured8) {
            cudaFuncSetAttribute(flash_prefill_fp8pv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 sizeof(Smem8));
            configured8 = true;
        }
        Attn8Args p8{qc8, qs8, kc8, ks8, vt, sv, out, tokens, heads, kv_heads, pos0,
                     static_cast<int>(kv_len), static_cast<int>(tiles)};
        flash_prefill_fp8pv_kernel<<<dim3((tokens + kBM - 1) / kBM, heads), kThreads, sizeof(Smem8)>>>(p8);
        return velox_launch_status();
    }
    rows_to_fp8<<<static_cast<unsigned>((q_rows + 7) / 8), 256>>>(q, q_rows, scale * kLog2e, qc, qs);
    rows_to_fp8<<<static_cast<unsigned>((k_rows + 7) / 8), 256>>>(k, k_rows, 1.0f, kc, ks);
    to_f16<<<velox_grid_1d(k_rows * kD / 4, 256), 256>>>(v, k_rows * kD / 4, vh);
    if (const int e = velox_launch_status()) return e;

    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(flash_prefill_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Smem));
        cudaFuncSetAttribute(flash_prefill_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Smem));
        configured = true;
    }
    AttnArgs p{qc, qs, kc, ks, vh, out, tokens, heads, kv_heads, pos0, static_cast<int>(kv_len)};
    const dim3 grid((tokens + kBM - 1) / kBM, heads);
    if (pv_mode() == 2)
        flash_prefill_kernel<true><<<grid, kThreads, sizeof(Smem)>>>(p);
    else
        flash_prefill_kernel<false><<<grid, kThreads, sizeof(Smem)>>>(p);
    return velox_launch_status();
}

// ---------------------------------------------------------------------------------------------
// Scoring windows (velox_core::Backend::attention_windows), precision F16: Q, K, V and the softmax
// weights in f16 with f32 accumulation. A window's scores are read as a distribution over a few
// candidates, which E4M3 scores (flash_prefill_kernel) move visibly. Same schedule as
// flash_prefill_kernel; query t sees the keys below pos0 and its window's keys
// pos0 + t / window * window ..= pos0 + t, and a block skips the tiles of earlier windows only.

namespace {

struct SmemH {
    __half k[2][kBN * kD];  // 512 B per key, laid out as V (v_off)
    __half v[2][kBN * kD];
};

struct WinArgs {
    const __half* q;  // [tokens, heads, D], times softmax_scale * log2(e)
    const __half* k;  // [kv_len, kv_heads, D]
    const __half* v;
    float* out;  // [tokens, heads, D]
    int tokens, heads, kv_heads, pos0, kv_len, window;
};

__global__ void to_f16_scaled(const float* __restrict__ x, long long n4, float mul, __half* __restrict__ y) {
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n4;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const float4 t = reinterpret_cast<const float4*>(x)[i];
        reinterpret_cast<__half2*>(y)[2 * i] = __floats2half2_rn(t.x * mul, t.y * mul);
        reinterpret_cast<__half2*>(y)[2 * i + 1] = __floats2half2_rn(t.z * mul, t.w * mul);
    }
}

__device__ __forceinline__ void load_tile_h(SmemH& s, const WinArgs& p, int stage, int kt, int hk) {
    const int key0 = kt * kBN;
#pragma unroll
    for (int i = 0; i < (kBN * kD * 2 / 16) / kThreads; ++i) {  // 1024 chunks each of K and V
        const int idx = threadIdx.x + i * kThreads, row = idx >> 5, c = idx & 31;
        const int key = key0 + row;
        const bool ok = key < p.kv_len;
        const long long off = (static_cast<long long>(ok ? key : 0) * p.kv_heads + hk) * kD + c * 8;
        cp_async16(s.k[stage] + v_off(row, c * 8), p.k + off, ok);
        cp_async16(s.v[stage] + v_off(row, c * 8), p.v + off, ok);
    }
}

__global__ void __launch_bounds__(kThreads, 2) flash_windows_kernel(WinArgs p) {
    extern __shared__ __align__(128) uint8_t smem_raw[];
    SmemH& s = *reinterpret_cast<SmemH*>(smem_raw);
    const int h = blockIdx.y, hk = h / (p.heads / p.kv_heads);
    const int t0 = blockIdx.x * kBM;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, q4 = lane & 3;

    // Q fragments for this warp's 16 rows: 16 k-steps of 16 halves (32 bytes each, the byte
    // layout of the E4M3 fragments).
    const int r0 = t0 + warp * 16 + g, r1 = r0 + 8;
    uint32_t qf[16][4];
    {
        const uint8_t* q0 = reinterpret_cast<const uint8_t*>(
            p.q + (static_cast<long long>(r0 < p.tokens ? r0 : 0) * p.heads + h) * kD);
        const uint8_t* q1 = reinterpret_cast<const uint8_t*>(
            p.q + (static_cast<long long>(r1 < p.tokens ? r1 : 0) * p.heads + h) * kD);
#pragma unroll
        for (int ks = 0; ks < 16; ++ks) {
            qf[ks][0] = r0 < p.tokens ? *reinterpret_cast<const uint32_t*>(q0 + ks * 32 + 4 * q4) : 0u;
            qf[ks][1] = r1 < p.tokens ? *reinterpret_cast<const uint32_t*>(q1 + ks * 32 + 4 * q4) : 0u;
            qf[ks][2] = r0 < p.tokens ? *reinterpret_cast<const uint32_t*>(q0 + ks * 32 + 16 + 4 * q4) : 0u;
            qf[ks][3] = r1 < p.tokens ? *reinterpret_cast<const uint32_t*>(q1 + ks * 32 + 16 + 4 * q4) : 0u;
        }
    }
    const int qpos0 = p.pos0 + r0, qpos1 = p.pos0 + r1;
    // First key of each row's own window (keys from pos0 up to it belong to earlier windows).
    const int lo0 = p.pos0 + r0 / p.window * p.window, lo1 = p.pos0 + r1 / p.window * p.window;

    float o[32][4];
#pragma unroll
    for (int i = 0; i < 32; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;

    // The prefix's tiles, then from the first tile any of the block's windows sees.
    const int last = min(p.tokens, t0 + kBM) - 1;
    const int n_keys = min(p.kv_len, p.pos0 + last + 1);
    const int n_tiles = (n_keys + kBN - 1) / kBN;
    const int n_pre = (p.pos0 + kBN - 1) / kBN;
    const int k_start = max(n_pre, (p.pos0 + t0 / p.window * p.window) / kBN);
    const int n_iter = n_pre + max(0, n_tiles - k_start);
    auto tile = [&](int i) { return i < n_pre ? i : k_start + (i - n_pre); };

    load_tile_h(s, p, 0, tile(0), hk);
    cp_commit();
    for (int it = 0; it < n_iter; ++it) {
        const int st = it & 1, kt = tile(it);
        if (it + 1 < n_iter) {
            load_tile_h(s, p, st ^ 1, tile(it + 1), hk);
            cp_commit();
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();

        // S = Q K^T for 16 rows x 32 keys (K rows loaded as the column-major B operand).
        float sc[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i) sc[i][0] = sc[i][1] = sc[i][2] = sc[i][3] = 0.0f;
#pragma unroll
        for (int ks = 0; ks < 16; ++ks) {
#pragma unroll
            for (int np = 0; np < 2; ++np) {
                uint32_t b[4];
                const int row = np * 16 + (lane & 7) + ((lane >> 4) << 3);
                ldsm_x4(b, s.k[st] + v_off(row, ks * 16 + (((lane >> 3) & 1) << 3)));
                mma_f16(sc[2 * np], qf[ks], b[0], b[1]);
                mma_f16(sc[2 * np + 1], qf[ks], b[2], b[3]);
            }
        }

        // Mask, online softmax (base 2). Rows g (c0, c1) and g + 8 (c2, c3).
        const int key0 = kt * kBN;
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int key = key0 + nt * 8 + 2 * q4 + e;
                const bool see0 = key <= qpos0 && (key < p.pos0 || key >= lo0);
                const bool see1 = key <= qpos1 && (key < p.pos0 || key >= lo1);
                if (!see0) sc[nt][e] = -INFINITY;
                if (!see1) sc[nt][2 + e] = -INFINITY;
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

Scratch g_windows_scratch;

}  // namespace

// Scoring windows of `window` rows over a pos0 prefix, head_dim 256 (see
// velox_core::backend::Backend::attention_windows): f32 k and v hold the prefix, then window w at
// rows pos0 + w * window. Returns VELOX_ERR_UNSUPPORTED for other geometries.
VELOX_API int velox_5090_attention_windows(const float* q, const float* k, const float* v, float* out, int tokens,
                                           int window, int heads, int kv_heads, int head_dim, int pos0,
                                           float scale) {
    if (head_dim != kD || tokens <= 0 || window <= 0 || tokens % window != 0 || kv_heads <= 0 ||
        heads % kv_heads != 0)
        return VELOX_ERR_UNSUPPORTED;
    const long long kv_len = static_cast<long long>(pos0) + tokens;
    const long long q_rows = static_cast<long long>(tokens) * heads, k_rows = kv_len * kv_heads;
    const size_t qb = align_up(q_rows * kD * 2), kb = align_up(k_rows * kD * 2);
    uint8_t* base = g_windows_scratch.get(qb + 2 * kb);
    if (!base) return static_cast<int>(cudaErrorMemoryAllocation);
    __half* qh = reinterpret_cast<__half*>(base);
    __half* kh = reinterpret_cast<__half*>(base + qb);
    __half* vh = reinterpret_cast<__half*>(base + qb + kb);

    constexpr float kLog2e = 1.4426950408889634f;
    to_f16_scaled<<<velox_grid_1d(q_rows * kD / 4, 256), 256>>>(q, q_rows * kD / 4, scale * kLog2e, qh);
    to_f16<<<velox_grid_1d(k_rows * kD / 4, 256), 256>>>(k, k_rows * kD / 4, kh);
    to_f16<<<velox_grid_1d(k_rows * kD / 4, 256), 256>>>(v, k_rows * kD / 4, vh);
    if (const int e = velox_launch_status()) return e;

    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(flash_windows_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SmemH));
        configured = true;
    }
    WinArgs p{qh, kh, vh, out, tokens, heads, kv_heads, pos0, static_cast<int>(kv_len), window};
    const dim3 grid((tokens + kBM - 1) / kBM, heads);
    flash_windows_kernel<<<grid, kThreads, sizeof(SmemH)>>>(p);
    return velox_launch_status();
}
