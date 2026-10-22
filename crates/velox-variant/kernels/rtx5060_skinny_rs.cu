// RTX 5060 Ti skinny linear (1..16 tokens) for groupwise-integer weights (row_split_k128_v1: q4_g64,
// q5_g64, q8_g32), the same schedule as rtx5060_skinny_iq3.cu: weights are the 16-row A operand of
// mma.m16n8k16 (each warp owns 16 rows), tokens the B operand (8 per n-tile), 8 warps per block and
// two cp.async stages of 128 K values. Activations are rounded to bf16 (ActQuant::Bf16); integer
// codes are bf16-exact, each group accumulates into its own register tile and is scaled by its
// FP16 multiplier before it is added.
//
// Lane q of an MMA quad holds K positions 2q, 2q+1 (h = 0) and 2q+8, 2q+9 (h = 1) of each k16 step
// for rows g and g+8, so a 4/5-bit pair is one byte of the base plane. Nibbles become bf16 through
// the bias trick: bf16 0x4300 | n is exactly 128 + n (n < 128), and one paired subtract removes the
// bias and the code offset.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include "velox_common.cuh"

namespace {

enum Fmt { kQ4 = 4, kQ5 = 5, kQ8 = 8 };

constexpr int kWarps = 8, kBN = 16 * kWarps, kThreads = 32 * kWarps, kStages = 2, kKS = 128;
constexpr int kSA = kKS + 8;  // activation row: 128 bf16 + 8 pad

template <int F>
struct Geo {
    static constexpr int kGroup = F == kQ8 ? 32 : 64;
    static constexpr int kGroups = kKS / kGroup;               // groups per row per stage
    static constexpr int kBase = F == kQ8 ? kKS : kKS / 2;     // base-plane bytes per row per stage
    static constexpr int kStride = kBase + 16;                 // padded against bank conflicts
    static constexpr int kHigh = F == kQ5 ? kGroups * 8 : 0;   // q5 high-plane bytes per row per stage
};

template <int F, int NT>
struct Smem {
    __nv_bfloat16 a[kStages][NT * 8 * kSA];
    uint8_t b[kStages][kBN * Geo<F>::kStride];
    uint8_t h[kStages][F == kQ5 ? kBN * 16 : 16];
    __half s[kStages][kBN * Geo<F>::kGroups];
};

struct Args {
    const __nv_bfloat16* x;  // [m, kp] bf16, zero padded columns
    const uint8_t* w;        // parent payload
    long long parent_rows, row_begin, gpr;  // gpr: groups per physical row
    long long high_off, scale_off;
    int rows, m, kp;
    int kt_per_split;
    float* y;  // [m, rows], or [splits, m, rows] partials
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
template <int N>
__device__ __forceinline__ void cp_async(void* dst, const void* src, bool valid) {
    if (N == 16)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src),
                     "r"(valid ? 16 : 0));
    else
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_u32(dst)), "l"(src), "n"(N),
                     "r"(valid ? N : 0));
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

// bf16x2 (lo, hi) = (128 + a, 128 + b) - bias for 0 <= a, b < 128.
__device__ __forceinline__ uint32_t biased_pair(uint32_t a, uint32_t b, uint32_t bias2) {
    const uint32_t v = 0x43004300u | a | (b << 16);
    const __nv_bfloat162 r = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&v),
                                     *reinterpret_cast<const __nv_bfloat162*>(&bias2));
    return *reinterpret_cast<const uint32_t*>(&r);
}

// Values k, k+1 (k even, stage-local) of a row as bf16x2.
template <int F>
__device__ __forceinline__ uint32_t decode2(const uint8_t* base, const uint8_t* high, int k) {
    if (F == kQ8) {
        const uint16_t v = *reinterpret_cast<const uint16_t*>(base + k);
        // int8 s -> s + 128 in 0..255 does not fit 7 mantissa bits: convert directly.
        const __nv_bfloat162 r = __floats2bfloat162_rn(static_cast<float>(static_cast<int8_t>(v & 0xff)),
                                                       static_cast<float>(static_cast<int8_t>(v >> 8)));
        return *reinterpret_cast<const uint32_t*>(&r);
    }
    const uint32_t byte = base[k >> 1];
    if (F == kQ4) {
        // Signed nibble u (two's complement in 4 bits) = (u ^ 8) - 8.
        constexpr uint32_t kBias = 0x43084308u;  // bf16 136.0 x2
        return biased_pair((byte & 0xf) ^ 8u, (byte >> 4) ^ 8u, kBias);
    }
    // q5: bit t (t = k % 64) of the group's 8 high bytes is value t's bit 4; k is even, so both bits
    // sit in one byte. Signed 5-bit u = (u ^ 16) - 16.
    const uint32_t hb = high[k >> 3] >> (k & 7);
    constexpr uint32_t kBias = 0x43104310u;  // bf16 144.0 x2
    return biased_pair(((byte & 0xf) | ((hb & 1u) << 4)) ^ 16u, ((byte >> 4) | ((hb & 2u) << 3)) ^ 16u, kBias);
}

template <int F, int NT>
__device__ __forceinline__ void load_stage(Smem<F, NT>& s, const Args& p, int stage, int kt, int n0) {
    using G = Geo<F>;
    const int t = threadIdx.x;
    const int k0 = kt * kKS;
    for (int idx = t; idx < p.m * (kKS / 8); idx += kThreads) {
        const int r = idx / (kKS / 8), c = idx % (kKS / 8);
        cp_async<16>(&s.a[stage][r * kSA + c * 8], p.x + static_cast<long long>(r) * p.kp + k0 + c * 8, true);
    }
    const long long row_bytes = p.gpr * 32;  // base plane: 32 bytes per group for every format
    const long long kbyte = F == kQ8 ? k0 : k0 / 2;
    constexpr int kChunks = G::kBase / 16;
    for (int idx = t; idx < kBN * kChunks; idx += kThreads) {
        const int r = idx / kChunks, c = idx % kChunks;
        const long long wrow = p.row_begin + n0 + r;
        const bool ok = wrow < p.parent_rows;
        cp_async<16>(&s.b[stage][r * G::kStride + c * 16], p.w + (ok ? wrow * row_bytes + kbyte + c * 16 : 0), ok);
    }
    const long long g0 = k0 / G::kGroup;
    for (int r = t; r < kBN; r += kThreads) {
        const long long wrow = p.row_begin + n0 + r;
        const bool ok = wrow < p.parent_rows;
        const long long gi = (ok ? wrow : 0) * p.gpr + g0;
        if (F == kQ5) cp_async<16>(&s.h[stage][r * 16], p.w + p.high_off + gi * 8, ok);
        cp_async<G::kGroups * 2>(&s.s[stage][r * G::kGroups], p.w + p.scale_off + gi * 2, ok);
    }
}

template <int F, int NT>
__global__ void __launch_bounds__(kThreads) skinny_rs_kernel(Args p) {
    using G = Geo<F>;
    extern __shared__ __align__(128) uint8_t smem_raw[];
    Smem<F, NT>& s = *reinterpret_cast<Smem<F, NT>*>(smem_raw);
    const int n0 = blockIdx.x * kBN;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int stages = p.kp / kKS;
    const int kt_begin = blockIdx.z * p.kt_per_split;
    const int kt_end = min(stages, kt_begin + p.kt_per_split);
    const int kt_count = kt_end - kt_begin;
    float* y = p.y + static_cast<long long>(blockIdx.z) * p.m * p.rows;

    for (int i = p.m * kSA + threadIdx.x; i < NT * 8 * kSA; i += kThreads)
#pragma unroll
        for (int st = 0; st < kStages; ++st) s.a[st][i] = __float2bfloat16_rn(0.0f);

#pragma unroll
    for (int st = 0; st < kStages - 1; ++st) {
        if (st < kt_count) load_stage<F, NT>(s, p, st, kt_begin + st, n0);
        cp_commit();
    }

    float acc[NT][4];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;

    const int g = lane >> 2, q = lane & 3;
    const int ar[2] = {warp * 16 + g, warp * 16 + 8 + g};
    const int b_row = lane & 7, b_col = (lane >> 3) * 8;

    for (int i = 0; i < kt_count; ++i) {
        cp_wait<kStages - 2>();
        __syncthreads();
        const int next = i + kStages - 1;
        if (next < kt_count) load_stage<F, NT>(s, p, next % kStages, kt_begin + next, n0);
        cp_commit();
        const int st = i % kStages;
        const uint8_t* basep[2] = {&s.b[st][ar[0] * G::kStride], &s.b[st][ar[1] * G::kStride]};
        const uint8_t* highp[2] = {&s.h[st][(F == kQ5 ? ar[0] : 0) * 16], &s.h[st][(F == kQ5 ? ar[1] : 0) * 16]};
#pragma unroll
        for (int gs = 0; gs < G::kGroups; ++gs) {
            float part[NT][4];
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) part[nt][0] = part[nt][1] = part[nt][2] = part[nt][3] = 0.0f;
#pragma unroll
            for (int c32 = 0; c32 < G::kGroup / 32; ++c32) {
                const int kc = gs * G::kGroup + c32 * 32;
                uint32_t bf[NT][4];
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) ldsm_x4(bf[nt], &s.a[st][(nt * 8 + b_row) * kSA + kc + b_col]);
#pragma unroll
                for (int ks = 0; ks < 2; ++ks) {
                    const int k = kc + ks * 16 + 2 * q;
                    // A fragment: {row g h=0, row g+8 h=0, row g h=1, row g+8 h=1}.
                    const uint32_t af[4] = {decode2<F>(basep[0], highp[0], k), decode2<F>(basep[1], highp[1], k),
                                            decode2<F>(basep[0], highp[0], k + 8),
                                            decode2<F>(basep[1], highp[1], k + 8)};
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt) mma_bf16(part[nt], af, bf[nt][2 * ks], bf[nt][2 * ks + 1]);
                }
            }
            const float s0 = __half2float(s.s[st][ar[0] * G::kGroups + gs]);
            const float s1 = __half2float(s.s[st][ar[1] * G::kGroups + gs]);
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                acc[nt][0] += part[nt][0] * s0;
                acc[nt][1] += part[nt][1] * s0;
                acc[nt][2] += part[nt][2] * s1;
                acc[nt][3] += part[nt][3] * s1;
            }
        }
    }
    cp_wait<0>();

    // c0, c1: row g, tokens 2q, 2q+1; c2, c3: row g + 8.
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int n = n0 + ar[e];
            if (n >= p.rows) continue;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int t = nt * 8 + 2 * q + j;
                if (t < p.m) y[static_cast<long long>(t) * p.rows + n] = acc[nt][2 * e + j];
            }
        }
}

// x [m, k] f32 -> [m, kp] bf16 (RNE), zero padded columns.
__global__ void to_bf16_pad(const float* __restrict__ x, int m, int k, int kp, __nv_bfloat16* __restrict__ y) {
    const long long n = static_cast<long long>(m) * kp;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const long long r = i / kp;
        const int c = static_cast<int>(i % kp);
        y[i] = __float2bfloat16_rn(c < k ? x[r * k + c] : 0.0f);
    }
}

__global__ void sum_splits(const float* __restrict__ part, int splits, long long n, float* __restrict__ y) {
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        float s = 0.0f;
        for (int z = 0; z < splits; ++z) s += part[z * n + i];
        y[i] = s;
    }
}

VeloxScratch g_x;     // bf16 activations
VeloxScratch g_part;  // split-K partials

constexpr int kSms = 36;

long long align256(long long x) { return (x + 255) / 256 * 256; }

// Split-K factor (power of two, at least 4 stages per slice) whose grid best fills whole waves of
// `slots` resident blocks; the smallest split within 5% of the best wins.
int pick_splits(long long blocks, long long stages, int slots) {
    auto fill = [&](int sp) {
        const long long n = blocks * sp, waves = (n + slots - 1) / slots;
        return static_cast<double>(n) / static_cast<double>(waves * slots);
    };
    double best = 0.0;
    for (int sp = 1; sp <= 16 && stages / sp >= 4; sp *= 2) best = fill(sp) > best ? fill(sp) : best;
    for (int sp = 1; sp <= 16 && stages / sp >= 4; sp *= 2)
        if (fill(sp) >= 0.95 * best) return sp;
    return 1;
}

template <int F, int NT>
int launch(Args p) {
    static int per_sm = 0;
    if (per_sm == 0) {
        cudaFuncSetAttribute(skinny_rs_kernel<F, NT>, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Smem<F, NT>));
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, skinny_rs_kernel<F, NT>, kThreads, sizeof(Smem<F, NT>));
        per_sm = per_sm > 0 ? per_sm : 1;
    }
    const long long blocks = (p.rows + kBN - 1) / kBN, stages = p.kp / kKS;
    const int splits = pick_splits(blocks, stages, kSms * per_sm);
    p.kt_per_split = static_cast<int>((stages + splits - 1) / splits);
    float* y = p.y;
    if (splits > 1) {
        p.y = static_cast<float*>(g_part.get(static_cast<size_t>(splits) * p.m * p.rows * sizeof(float)));
        if (!p.y) return static_cast<int>(cudaErrorMemoryAllocation);
    }
    skinny_rs_kernel<F, NT><<<dim3(static_cast<unsigned>(blocks), 1, splits), kThreads, sizeof(Smem<F, NT>)>>>(p);
    if (splits > 1) {
        const long long n = static_cast<long long>(p.m) * p.rows;
        sum_splits<<<velox_grid_1d(n, 256), 256>>>(p.y, splits, n, y);
    }
    return velox_launch_status();
}

template <int F>
int launch_fmt(Args p) {
    return p.m <= 8 ? launch<F, 1>(p) : launch<F, 2>(p);
}

}  // namespace

// y[m, rows] = bf16(x) . W[row_begin.., :]^T for a row_split_k128_v1 parent [parent_rows, k] in
// format 4 (q4_g64), 5 (q5_g64) or 8 (q8_g32), 1 <= m <= 16. Returns VELOX_ERR_UNSUPPORTED for
// anything else.
VELOX_API int velox_5060_linear_skinny_rs(const float* x, int m, const uint8_t* w, long long parent_rows, long long k,
                                          long long row_begin, int rows, int format, float* y) {
    if (m < 1 || m > 16 || rows <= 0 || reinterpret_cast<uintptr_t>(w) % 16 != 0) return VELOX_ERR_UNSUPPORTED;
    if (format != kQ4 && format != kQ5 && format != kQ8) return VELOX_ERR_UNSUPPORTED;
    Args p{};
    p.w = w;
    p.parent_rows = parent_rows;
    p.row_begin = row_begin;
    p.rows = rows;
    p.m = m;
    p.y = y;
    p.kp = static_cast<int>((k + 127) / 128 * 128);
    p.gpr = p.kp / (format == kQ8 ? 32 : 64);
    const long long groups = parent_rows * p.gpr;
    p.high_off = align256(groups * 32);
    p.scale_off = p.high_off + (format == kQ5 ? align256(groups * 8) : 0);
    auto* xb = static_cast<__nv_bfloat16*>(g_x.get(static_cast<size_t>(m) * p.kp * 2));
    if (!xb) return static_cast<int>(cudaErrorMemoryAllocation);
    to_bf16_pad<<<velox_grid_1d(static_cast<long long>(m) * p.kp, 256), 256>>>(x, m, static_cast<int>(k), p.kp, xb);
    p.x = xb;
    if (format == kQ4) return launch_fmt<kQ4>(p);
    if (format == kQ5) return launch_fmt<kQ5>(p);
    return launch_fmt<kQ8>(p);
}
