// RTX 5060 Ti (from rtx-5090-32g) skinny linear (1..16 tokens) for weights whose uses are A16Only: NVFP4, groupwise
// q4_g64 / q5_g64 / q8_g32 (row_split_k128_v1). Activations are rounded to bf16 (ActQuant::Bf16); the
// weights stay exact:
//   NVFP4: E2M1 code x E4M3 group scale is bf16-exact, decoded per value; the tensor divisor is
//          applied to the result.
//   q4/q5/q8: integer codes are bf16-exact (q5: the nibble plus its bit from the high plane); each group accumulates into its own register tile, which
//          is scaled by the group's FP16 scale before it is added.
//
// Speculative verify and draft steps are weight-streaming, so the block streams raw weight bytes
// through a cp.async pipeline (64 rows x 128 K values per stage, 4 stages) and decodes B fragments
// straight from shared memory into registers for mma.m16n8k16 (bf16, f32 accumulate). Four warps
// own 16 weight rows each against all 16 (padded) tokens; split-K spreads small grids over the GPU.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#include <mutex>
#include <unordered_map>

#include "velox_common.cuh"

namespace {

enum Fmt { kNvfp4 = 0, kQ4 = 1, kQ8 = 2, kQ5 = 3 };

constexpr int kBN = 64, kKS = 128, kStages = 4, kThreads = 128, kTok = 16;
// Activation tile rows: 128 bf16 + 8 pad (272 bytes) so ldmatrix rows fall in distinct banks.
constexpr int kSA = kKS + 8;

template <int F>
struct Geo {
    // Weight bytes per row per stage, padded by 16 so the 8 rows a fragment reads hit 8 banks.
    static constexpr int kRowBytes = F == kQ8 ? kKS : kKS / 2;
    static constexpr int kStride = kRowBytes + 16;
    static constexpr int kGroup = F == kQ8 ? 32 : F == kNvfp4 ? 16 : 64;
    // q5 high-plane bytes per row per stage: one bit per value, 8 bytes per 64-group.
    static constexpr int kHighBytes = F == kQ5 ? kKS / 8 : 0;
    static constexpr int kHighStride = F == kQ5 ? kHighBytes + 16 : 1;
    // Scale bytes per row per stage: FP16 per group (q4: 2, q8: 4); NVFP4 uses 512-byte tiles.
    static constexpr int kScaleHalves = kKS / (F == kQ8 ? 32 : 64);
};

template <int F>
struct Smem {
    __nv_bfloat16 a[kStages][kTok * kSA];
    uint8_t b[kStages][kBN * Geo<F>::kStride];
    // q4/q8: [row][group] FP16 scales. NVFP4: the two 128x4 E4M3 tiles of the stage's k64 steps.
    uint8_t s[kStages][F == kNvfp4 ? 2 * 512 : kBN * Geo<F>::kScaleHalves * 2];
    uint8_t h[kStages][F == kQ5 ? kBN * Geo<F>::kHighStride : 16];
};

struct Args {
    const __nv_bfloat16* x;  // [16, kp] bf16, zero padded
    const uint8_t* w;        // parent payload
    long long parent_rows, row_begin;
    int rows, m, kp;         // kp: K padded to the format's row length (multiple of 128)
    long long gpr;           // q4/q8: groups per row
    long long scale_off;     // q4/q5/q8: FP16 scale plane; NVFP4: E4M3 scale tile plane
    long long high_off;      // q5: high-bit plane
    float alpha;             // NVFP4: 1 / weight divisor; else 1
    int kt_per_split;
    float* y;                // [m, rows], or [splits, m, rows] partials
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src),
                 "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_async8(void* dst, const void* src, bool valid) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(smem_u32(dst)), "l"(src),
                 "r"(valid ? 8 : 0));
}
__device__ __forceinline__ void cp_async4(void* dst, const void* src, bool valid) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(smem_u32(dst)), "l"(src),
                 "r"(valid ? 4 : 0));
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
__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
}

// E2M1 code -> f32 by building the bits (a table indexed at run time would live in local memory):
// magnitude codes 0..7 are 0, 0.5, 1, 1.5, 2, 3, 4, 6.
__device__ __forceinline__ float e2m1(uint32_t c) {
    const uint32_t e = (c >> 1) & 3, f = c & 1;
    const uint32_t mag = e ? ((e + 126u) << 23) | (f << 22) : f * (126u << 23);
    return __uint_as_float(mag | ((c & 8u) << 28));
}

// Four E2M1 codes (16 bits, code i in bits 4i..4i+3) -> bf16 pairs (codes 0,1) and (2,3). The
// magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6 are bf16 0x0000, 3F00, 3F80, 3FC0, 4000, 4040, 4080,
// 40C0: two byte-permute tables give their high and low bytes, then the sign bits are or-ed in.
__device__ __forceinline__ void e2m1x4_bf16(uint32_t c, uint32_t& p01, uint32_t& p23) {
    const uint32_t sel = c & 0x7777;
    const uint32_t hi = __byte_perm(0x3F3F3F00u, 0x40404040u, sel);
    const uint32_t lo = __byte_perm(0xC0800000u, 0xC0804000u, sel);
    p01 = __byte_perm(lo, hi, 0x5140) | ((c & 0x8u) << 12) | ((c & 0x80u) << 24);
    p23 = __byte_perm(lo, hi, 0x7362) | ((c & 0x800u) << 4) | ((c & 0x8000u) << 16);
}

// NVFP4 K order within a 32-value chunk: MMA position l = 16s + 8h + 2q + j (step s, fragment
// half h, lane quad q, element j) reads weight code p = 8q + 4s + 2h + j.
__host__ __device__ __forceinline__ int nvfp4_k_perm(int l) {
    const int s = (l >> 4) & 1, h = (l >> 3) & 1, q = (l >> 1) & 3, j = l & 1;
    return (l & ~31) + 8 * q + 4 * s + 2 * h + j;
}

__device__ __forceinline__ float e4m3(uint8_t b) {
    const __half_raw h = __nv_cvt_fp8_to_halfraw(b, __NV_E4M3);
    return __half2float(__half(h));
}

// Two adjacent K values (k, k+1) of row `r` as a bf16x2 register (k even; q5: `high` is the row's
// high-plane bytes for the stage).
template <int F>
__device__ __forceinline__ uint32_t decode2(const uint8_t* row, const uint8_t* high, int k, float nv_scale) {
    if (F == kQ8) {
        const uint16_t v = *reinterpret_cast<const uint16_t*>(row + k);
        return pack_bf16(static_cast<float>(static_cast<int8_t>(v & 0xff)), static_cast<float>(static_cast<int8_t>(v >> 8)));
    } else {
        const uint32_t v = row[k >> 1];
        if (F == kQ4) {
            const int lo = static_cast<int>(v & 0xf), hi = static_cast<int>(v >> 4);
            return pack_bf16(static_cast<float>(lo >= 8 ? lo - 16 : lo), static_cast<float>(hi >= 8 ? hi - 16 : hi));
        }
        if (F == kQ5) {
            // Bit t of a group's 8 high bytes is value t's bit 4 (k % 64 == t; k even, so both
            // bits sit in one byte).
            const uint32_t hb = high[k >> 3] >> (k & 7);
            const int lo = static_cast<int>((v & 0xf) | ((hb & 1) << 4)), hi = static_cast<int>((v >> 4) | ((hb & 2) << 3));
            return pack_bf16(static_cast<float>(lo >= 16 ? lo - 32 : lo), static_cast<float>(hi >= 16 ? hi - 32 : hi));
        }
        return pack_bf16(e2m1(v & 0xf) * nv_scale, e2m1(v >> 4) * nv_scale);
    }
}

template <int F>
__device__ __forceinline__ void load_stage(Smem<F>& s, const Args& p, int stage, int kt, int n0) {
    const int t = threadIdx.x;
    const int k0 = kt * kKS;
    // Activations: 16 rows x 256 bytes.
    for (int idx = t; idx < kTok * (kKS / 8); idx += kThreads) {
        const int r = idx / (kKS / 8), c = idx % (kKS / 8);
        cp_async16(&s.a[stage][r * kSA + c * 8], p.x + static_cast<long long>(r) * p.kp + k0 + c * 8, true);
    }
    // Weight rows: contiguous kRowBytes per row (row_split rows are gpr * 32 bytes, NVFP4 k / 2).
    constexpr int kChunks = Geo<F>::kRowBytes / 16;
    const long long row_bytes = F == kNvfp4 ? p.kp / 2 : p.gpr * 32;
    const long long kbyte = F == kQ8 ? k0 : k0 / 2;
    for (int idx = t; idx < kBN * kChunks; idx += kThreads) {
        const int r = idx / kChunks, c = idx % kChunks;
        const long long wrow = p.row_begin + n0 + r;
        const bool ok = wrow < p.parent_rows;
        cp_async16(&s.b[stage][r * Geo<F>::kStride + c * 16], p.w + (ok ? wrow : 0) * row_bytes + kbyte + c * 16, ok);
    }
    if (F == kNvfp4) {
        // Two k64 scale tiles of the 128-row tile holding these rows.
        const long long tile = (p.row_begin + n0) / 128;
        for (int idx = t; idx < 64; idx += kThreads) {
            const int step = idx >> 5, off = (idx & 31) * 16;
            cp_async16(&s.s[stage][step * 512 + off],
                       p.w + p.scale_off + (tile * (p.kp / 64) + kt * 2 + step) * 512 + off, true);
        }
    } else {
        if (F == kQ5) {
            // High bits of the stage's two groups: 16 contiguous bytes per row.
            for (int r = t; r < kBN; r += kThreads) {
                const long long wrow = p.row_begin + n0 + r;
                const bool ok = wrow < p.parent_rows;
                cp_async16(&s.h[stage][r * Geo<F>::kHighStride],
                           p.w + p.high_off + ((ok ? wrow : 0) * p.gpr + k0 / 64) * 8, ok);
            }
        }
        // FP16 scales of the stage's groups: q8 four (8 bytes), q4/q5 two (4 bytes) per row.
        constexpr int kHalves = Geo<F>::kScaleHalves;
        const long long g0 = k0 / Geo<F>::kGroup;
        for (int r = t; r < kBN; r += kThreads) {
            const long long wrow = p.row_begin + n0 + r;
            const bool ok = wrow < p.parent_rows;
            const uint8_t* src = p.w + p.scale_off + ((ok ? wrow : 0) * p.gpr + g0) * 2;
            if (kHalves == 4) cp_async8(&s.s[stage][r * 8], src, ok);
            else cp_async4(&s.s[stage][r * 4], src, ok);
        }
    }
}

template <int F>
__global__ void __launch_bounds__(kThreads) skinny16_kernel(Args p) {
    extern __shared__ __align__(128) uint8_t smem_raw[];
    Smem<F>& s = *reinterpret_cast<Smem<F>*>(smem_raw);
    const int n0 = blockIdx.x * kBN;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int kt_begin = blockIdx.z * p.kt_per_split;
    const int kt_end = min(p.kp / kKS, kt_begin + p.kt_per_split);
    const int kt_count = kt_end - kt_begin;
    float* y = p.y + static_cast<long long>(blockIdx.z) * p.m * p.rows;

#pragma unroll
    for (int st = 0; st < kStages - 1; ++st) {
        if (st < kt_count) load_stage<F>(s, p, st, kt_begin + st, n0);
        cp_commit();
    }

    float acc[2][4];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;

    const int a_row = (lane & 7) + (((lane >> 3) & 1) << 3), a_col = (lane >> 4) << 3;
    const int g = lane >> 2, q2 = (lane & 3) * 2;
    // This thread's B rows (one per n8 tile) within the block.
    const int br[2] = {warp * 16 + g, warp * 16 + 8 + g};

    for (int i = 0; i < kt_count; ++i) {
        cp_wait<kStages - 2>();
        __syncthreads();
        const int next = i + kStages - 1;
        if (next < kt_count) load_stage<F>(s, p, next % kStages, kt_begin + next, n0);
        cp_commit();
        const int st = i % kStages;
        const uint8_t* rowp[2] = {&s.b[st][br[0] * Geo<F>::kStride], &s.b[st][br[1] * Geo<F>::kStride]};
        const uint8_t* highp[2] = {&s.h[st][br[0] * Geo<F>::kHighStride], &s.h[st][br[1] * Geo<F>::kHighStride]};

        if (F == kNvfp4) {
            // K is permuted within each 32-value chunk (activations were stored to match, see
            // nvfp4_k_perm): thread q's B values for the chunk's two k16 steps are its 8
            // consecutive codes (one u32), all in 16-group q / 2.
            const int q = lane & 3;
#pragma unroll
            for (int ch = 0; ch < kKS / 32; ++ch) {
                uint32_t af[2][4];
                ldsm_x4(af[0], &s.a[st][a_row * kSA + ch * 32 + a_col]);
                ldsm_x4(af[1], &s.a[st][a_row * kSA + ch * 32 + 16 + a_col]);
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    const uint32_t w = *reinterpret_cast<const uint32_t*>(rowp[nt] + ch * 16 + 4 * q);
                    // Scale of (row, 16-group 2ch + q/2) in the stage's 128x4 tiles of k64 steps.
                    const int r = static_cast<int>((p.row_begin + n0) & 127) + br[nt];
                    const int gi = ch * 2 + (q >> 1);
                    const __nv_bfloat16 sb =
                        __float2bfloat16_rn(e4m3(s.s[st][(gi >> 2) * 512 + (r & 31) * 16 + (r >> 5) * 4 + (gi & 3)]));
                    const __nv_bfloat162 sc2 = __halves2bfloat162(sb, sb);
                    uint32_t v[4];
                    e2m1x4_bf16(w & 0xffff, v[0], v[1]);
                    e2m1x4_bf16(w >> 16, v[2], v[3]);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const __nv_bfloat162 x = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&v[j]), sc2);
                        v[j] = *reinterpret_cast<const uint32_t*>(&x);
                    }
                    mma_bf16(acc[nt], af[0], v[0], v[1]);
                    mma_bf16(acc[nt], af[1], v[2], v[3]);
                }
            }
        } else {
            constexpr int kSteps = Geo<F>::kGroup / 16;
#pragma unroll
            for (int gs = 0; gs < kKS / Geo<F>::kGroup; ++gs) {
                float part[2][4];
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) part[nt][0] = part[nt][1] = part[nt][2] = part[nt][3] = 0.0f;
#pragma unroll
                for (int kk = 0; kk < kSteps; ++kk) {
                    const int ks = gs * kSteps + kk;
                    uint32_t af[4];
                    ldsm_x4(af, &s.a[st][a_row * kSA + ks * 16 + a_col]);
#pragma unroll
                    for (int nt = 0; nt < 2; ++nt) {
                        const int k = ks * 16 + q2;
                        mma_bf16(part[nt], af, decode2<F>(rowp[nt], highp[nt], k, 0.0f),
                                 decode2<F>(rowp[nt], highp[nt], k + 8, 0.0f));
                    }
                }
                // C columns 2q, 2q+1 are rows g' = (lane & 3) * 2 (+1) of the n8 tile.
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    const int r0 = warp * 16 + nt * 8 + q2;
                    const __half* sh = reinterpret_cast<const __half*>(s.s[st]);
                    const float s0 = __half2float(sh[r0 * Geo<F>::kScaleHalves + gs]);
                    const float s1 = __half2float(sh[(r0 + 1) * Geo<F>::kScaleHalves + gs]);
                    acc[nt][0] += part[nt][0] * s0;
                    acc[nt][1] += part[nt][1] * s1;
                    acc[nt][2] += part[nt][2] * s0;
                    acc[nt][3] += part[nt][3] * s1;
                }
            }
        }
    }
    cp_wait<0>();

    // c0, c1: token g, rows 2q, 2q+1; c2, c3: token g + 8.
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        const int n = n0 + warp * 16 + nt * 8 + q2;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int t = g + 8 * h;
            if (t >= p.m) continue;
            float* out = y + static_cast<long long>(t) * p.rows + n;
            if (n < p.rows) out[0] = acc[nt][2 * h] * p.alpha;
            if (n + 1 < p.rows) out[1] = acc[nt][2 * h + 1] * p.alpha;
        }
    }
}

// x [m, k] f32 -> [16, kp] bf16 (RNE), zero padded; with `perm`, columns in nvfp4_k_perm order.
__global__ void to_bf16_16(const float* __restrict__ x, int m, int k, int kp, bool perm,
                           __nv_bfloat16* __restrict__ y) {
    const long long n = static_cast<long long>(kTok) * kp;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const int r = static_cast<int>(i / kp), l = static_cast<int>(i % kp);
        const int c = perm ? nvfp4_k_perm(l) : l;
        y[i] = __float2bfloat16_rn(r < m && c < k ? x[static_cast<long long>(r) * k + c] : 0.0f);
    }
}

__global__ void sum_splits16(const float* __restrict__ part, int splits, long long n, float* __restrict__ y) {
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        float s = 0.0f;
        for (int z = 0; z < splits; ++z) s += part[z * n + i];
        y[i] = s;
    }
}

struct Scratch {
    VeloxScratch s;
    uint8_t* get(size_t need) { return static_cast<uint8_t*>(s.get(need)); }
};
Scratch g_x16;     // bf16 activations
Scratch g_part16;  // split-K partials

constexpr int kSms = 82;  // RTX 5090 Laptop GPU (GB203)

long long align256(long long x) { return (x + 255) / 256 * 256; }

template <int F>
int launch(Args p, const float* x, int k) {
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(skinny16_kernel<F>, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Smem<F>));
        configured = true;
    }
    auto* xb = reinterpret_cast<__nv_bfloat16*>(g_x16.get(static_cast<size_t>(kTok) * p.kp * 2));
    if (!xb) return static_cast<int>(cudaErrorMemoryAllocation);
    to_bf16_16<<<velox_grid_1d(static_cast<long long>(kTok) * p.kp, 256), 256>>>(x, p.m, k, p.kp, F == kNvfp4, xb);
    p.x = xb;
    const int kt_total = p.kp / kKS;
    const long long blocks = (p.rows + kBN - 1) / kBN;
    // Split K until the grid covers the GPU twice, keeping at least 2 stages per slice.
    int splits = 1;
    while (blocks * splits < 2 * kSms && kt_total / (splits * 2) >= 2) splits *= 2;
    p.kt_per_split = (kt_total + splits - 1) / splits;
    float* y = p.y;
    if (splits > 1) {
        p.y = reinterpret_cast<float*>(g_part16.get(static_cast<size_t>(splits) * p.m * p.rows * sizeof(float)));
        if (!p.y) return static_cast<int>(cudaErrorMemoryAllocation);
    }
    skinny16_kernel<F><<<dim3(static_cast<unsigned>(blocks), 1, splits), kThreads, sizeof(Smem<F>)>>>(p);
    if (splits > 1) {
        const long long n = static_cast<long long>(p.m) * p.rows;
        sum_splits16<<<velox_grid_1d(n, 256), 256>>>(p.y, splits, n, y);
    }
    return velox_launch_status();
}

}  // namespace

// y[m, rows] = bf16(x) . W[row_begin.., :]^T for 1 <= m <= 16. format: 0 NVFP4 (needs k % 128 == 0
// and row_begin % 64 == 0), 4 q4_g64, 5 q5_g64 or 8 q8_g32 (row_split_k128_v1). Returns
// VELOX_ERR_UNSUPPORTED for anything else.
VELOX_API int velox_5060_linear_skinny16(const float* x, int m, const uint8_t* w, long long parent_rows, long long k,
                                         long long row_begin, int rows, int format, float* y) {
    if (m < 1 || m > kTok || rows <= 0 || reinterpret_cast<uintptr_t>(w) % 16 != 0) return VELOX_ERR_UNSUPPORTED;
    Args p{};
    p.w = w;
    p.parent_rows = parent_rows;
    p.row_begin = row_begin;
    p.rows = rows;
    p.m = m;
    p.y = y;
    p.alpha = 1.0f;
    if (format == 0) {
        if (k % 128 != 0 || row_begin % 64 != 0) return VELOX_ERR_UNSUPPORTED;
        p.kp = static_cast<int>(k);
        p.scale_off = align256(parent_rows * k / 2);
        // The tensor divisor is the FP32 word after the scale plane; read once per parent (a read
        // per call would synchronize).
        static std::mutex mu;
        static std::unordered_map<const uint8_t*, float> cache;
        std::lock_guard<std::mutex> lock(mu);
        auto it = cache.find(w);
        if (it == cache.end()) {
            float divisor = 0.0f;
            if (const cudaError_t e = cudaMemcpy(&divisor, w + p.scale_off + parent_rows * k / 16, 4, cudaMemcpyDeviceToHost))
                return static_cast<int>(e);
            it = cache.emplace(w, divisor).first;
        }
        p.alpha = 1.0f / it->second;
        return launch<kNvfp4>(p, x, static_cast<int>(k));
    }
    if (format != 4 && format != 5 && format != 8) return VELOX_ERR_UNSUPPORTED;
    const int group = format == 8 ? 32 : 64;
    p.kp = static_cast<int>((k + 127) / 128 * 128);
    p.gpr = p.kp / group;
    // row_split_k128_v1 planes: base codes, high bits (q5: 8 bytes per group), FP16 scales.
    const long long groups = parent_rows * p.gpr;
    p.high_off = align256(groups * 32);
    p.scale_off = p.high_off + (format == 5 ? align256(groups * 8) : 0);
    if (format == 5) return launch<kQ5>(p, x, static_cast<int>(k));
    return format == 8 ? launch<kQ8>(p, x, static_cast<int>(k)) : launch<kQ4>(p, x, static_cast<int>(k));
}
