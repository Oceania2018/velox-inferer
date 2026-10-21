// RTX 5090 skinny linear (2..64 tokens) for FP8 E4M3 row-scaled weights with bf16 activations
// (ActQuant::Bf16): one kernel per call, no activation quantizer and no split-K pass.
//
// A block owns 16 weight rows over the whole K and splits each stage's K across KW warps (64 K
// values per warp). Weight codes and f32 activations stream through a cp.async pipeline; E4M3
// codes widen exactly to bf16 (via f16) as the MMA A operand, activations round to bf16 (RNE) as
// the B operand of mma.m16n8k16 (f32 accumulate). The warps' partial sums meet in shared memory
// and the row's BF16 scale multiplies the complete dot product.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include "velox_common.cuh"

namespace {

constexpr int kRows = 16;     // weight rows per block
constexpr int kWarpK = 64;    // K values per warp per stage
constexpr int kStages = 2;

template <int TOK, int KW>
struct A16Geo {
    static constexpr int BK = KW * kWarpK;     // K per stage
    static constexpr int AS = BK + 8;          // activation row stride (floats): spreads banks
    static constexpr int CODE_BYTES = kRows * BK;
    static constexpr int ACT_FLOATS = TOK * AS;
    static constexpr int STAGE_BYTES = CODE_BYTES + ACT_FLOATS * 4;
    static constexpr int PARTIAL_BYTES = KW * (TOK / 8) * 32 * 4 * 4;
    static constexpr int SMEM = kStages * STAGE_BYTES > PARTIAL_BYTES ? kStages * STAGE_BYTES : PARTIAL_BYTES;
};

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

__device__ __forceinline__ void mma_bf16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Two E4M3 codes (low byte first) as a bf16x2 word: E4M3 -> f16 is exact, f16 -> bf16 too
// (every E4M3 value has at most 4 significant bits and lies in bf16's range).
__device__ __forceinline__ uint32_t e4m3x2_bf16x2(uint16_t x) {
    uint32_t h;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"(x));
    const __half2 hh = *reinterpret_cast<const __half2*>(&h);
    const __nv_bfloat162 b = __floats2bfloat162_rn(__low2float(hh), __high2float(hh));
    return *reinterpret_cast<const uint32_t*>(&b);
}

__device__ __forceinline__ uint32_t bf16x2_rn(float lo, float hi) {
    const __nv_bfloat162 b = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&b);
}

struct A16Args {
    const float* x;      // [m, k]
    const uint8_t* w;    // parent payload: [parent_rows, k] E4M3 codes, then BF16 row scales
    long long parent_rows, row_begin;
    int rows, m, k;
    float* y;            // [m, rows]
};

template <int TOK, int KW>
__global__ void __launch_bounds__(KW * 32) a16_fp8_kernel(A16Args p) {
    using G = A16Geo<TOK, KW>;
    extern __shared__ __align__(128) uint8_t smem[];
    auto codes = [&](int st) { return smem + st * G::STAGE_BYTES; };
    auto acts = [&](int st) { return reinterpret_cast<float*>(smem + st * G::STAGE_BYTES + G::CODE_BYTES); };

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int gid = lane >> 2, lid = lane & 3;
    const int row0 = blockIdx.x * kRows;
    const int groups = p.k / G::BK;
    const long long scale_plane = (p.parent_rows * p.k + 255) / 256 * 256;

    auto stage = [&](int st, int k0) {
        // Weight codes: 16 rows x BK bytes, 16-byte chunks XOR-swizzled by (row & 7).
        uint8_t* cs = codes(st);
        for (int i = tid; i < kRows * (G::BK / 16); i += KW * 32) {
            const int r = i / (G::BK / 16), c = i % (G::BK / 16);
            const bool ok = row0 + r < p.rows;
            const long long wr = p.row_begin + (ok ? row0 + r : 0);
            cp_async16(cs + r * G::BK + ((c ^ (r & 7)) * 16), p.w + wr * p.k + k0 + c * 16, ok);
        }
        // Activations: TOK rows x BK floats (rows past m zero-filled).
        float* as = acts(st);
        for (int i = tid; i < TOK * (G::BK / 4); i += KW * 32) {
            const int t = i / (G::BK / 4), c = i % (G::BK / 4);
            const bool ok = t < p.m;
            cp_async16(as + t * G::AS + c * 4, p.x + static_cast<long long>(ok ? t : 0) * p.k + k0 + c * 4, ok);
        }
    };

    float acc[TOK / 8][4] = {};
#pragma unroll
    for (int st = 0; st < kStages; ++st) {
        if (st < groups) stage(st, st * G::BK);
        cp_commit();
    }
    for (int g = 0; g < groups; ++g) {
        const int st = g % kStages;
        cp_wait<kStages - 1>();
        __syncthreads();
        const uint8_t* cs = codes(st);
        const float* as = acts(st);
#pragma unroll
        for (int ks = 0; ks < kWarpK / 16; ++ks) {
            const int col = warp * kWarpK + ks * 16 + 2 * lid;  // within the stage
            auto pair = [&](int r, int c) {
                const int chunk = c >> 4;
                return *reinterpret_cast<const uint16_t*>(cs + r * G::BK + ((chunk ^ (r & 7)) * 16) + (c & 15));
            };
            const uint32_t a0 = e4m3x2_bf16x2(pair(gid, col)), a1 = e4m3x2_bf16x2(pair(gid + 8, col));
            const uint32_t a2 = e4m3x2_bf16x2(pair(gid, col + 8)), a3 = e4m3x2_bf16x2(pair(gid + 8, col + 8));
#pragma unroll
            for (int nt = 0; nt < TOK / 8; ++nt) {
                const float* xr = as + (nt * 8 + gid) * G::AS + col;
                const float2 lo = *reinterpret_cast<const float2*>(xr);
                const float2 hi = *reinterpret_cast<const float2*>(xr + 8);
                mma_bf16(acc[nt], a0, a1, a2, a3, bf16x2_rn(lo.x, lo.y), bf16x2_rn(hi.x, hi.y));
            }
        }
        __syncthreads();
        if (g + kStages < groups) stage(st, (g + kStages) * G::BK);
        cp_commit();
    }
    cp_wait<0>();
    __syncthreads();

    // Cross-warp reduction (in warp order) and the epilogue by warp 0.
    float* part = reinterpret_cast<float*>(smem);
#pragma unroll
    for (int nt = 0; nt < TOK / 8; ++nt)
        *reinterpret_cast<float4*>(part + ((warp * (TOK / 8) + nt) * 32 + lane) * 4) =
            make_float4(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3]);
    __syncthreads();
    if (warp != 0) return;
    const uint16_t* sp = reinterpret_cast<const uint16_t*>(p.w + scale_plane);
    const int ra = row0 + gid, rb = row0 + gid + 8;
    const float sa = ra < p.rows ? __uint_as_float(static_cast<uint32_t>(sp[p.row_begin + ra]) << 16) : 0.0f;
    const float sb = rb < p.rows ? __uint_as_float(static_cast<uint32_t>(sp[p.row_begin + rb]) << 16) : 0.0f;
#pragma unroll
    for (int nt = 0; nt < TOK / 8; ++nt) {
        float4 s = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (int w = 0; w < KW; ++w) {
            const float4 v = *reinterpret_cast<const float4*>(part + ((w * (TOK / 8) + nt) * 32 + lane) * 4);
            s.x += v.x, s.y += v.y, s.z += v.z, s.w += v.w;
        }
        const int t = nt * 8 + 2 * lid;
        if (t < p.m) {
            if (ra < p.rows) p.y[static_cast<long long>(t) * p.rows + ra] = s.x * sa;
            if (rb < p.rows) p.y[static_cast<long long>(t) * p.rows + rb] = s.z * sb;
        }
        if (t + 1 < p.m) {
            if (ra < p.rows) p.y[static_cast<long long>(t + 1) * p.rows + ra] = s.y * sa;
            if (rb < p.rows) p.y[static_cast<long long>(t + 1) * p.rows + rb] = s.w * sb;
        }
    }
}

template <int TOK, int KW>
int launch_a16(const A16Args& a) {
    using G = A16Geo<TOK, KW>;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(a16_fp8_kernel<TOK, KW>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM);
        configured = true;
    }
    a16_fp8_kernel<TOK, KW><<<(a.rows + kRows - 1) / kRows, KW * 32, G::SMEM>>>(a);
    return velox_launch_status();
}

// ---- wide batches (17..64 tokens): batched speculative verify of several sequences ----
//
// The skinny kernel splits K across warps and keeps every token's f32 activations of a stage in
// shared memory; at 32-64 tokens that leaves one small block per SM and the MMAs starve. Here a
// block owns 32 weight rows (16 per warp, each over the whole K) and all tokens, activations are
// rounded to bf16 once per call (the same RNE rounding) and stream with the codes through a
// three-stage cp.async pipeline in ~34 KB, so several blocks share an SM.
constexpr int kWideK = 64;                 // K per stage
constexpr int kWideStages = 3;
constexpr int kWideWarps = 2;              // 16 weight rows each
constexpr int kWideRows = 16 * kWideWarps;
constexpr int kWideAS = kWideK + 8;        // bf16 activation row stride (elements): spreads banks

template <int TOK>
struct WideGeo {
    static constexpr int CODE = kWideRows * kWideK;  // bytes
    static constexpr int ACT = TOK * kWideAS * 2;    // bytes
    static constexpr int STAGE = CODE + ACT;
    static constexpr int SMEM = kWideStages * STAGE;
};

__global__ void to_bf16_kernel(const float2* __restrict__ x, __nv_bfloat162* __restrict__ y, long long pairs) {
    for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < pairs;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const float2 v = x[i];
        y[i] = __floats2bfloat162_rn(v.x, v.y);
    }
}

template <int TOK>
// Split-K: blockIdx.y takes K groups [y * gps, (y + 1) * gps); with several splits it writes unscaled
// partial sums to part[y][t][row] for wide_reduce_kernel, else the scaled outputs.
__global__ void __launch_bounds__(kWideWarps * 32)
    a16_fp8_wide_kernel(A16Args p, const __nv_bfloat16* __restrict__ xb, float* __restrict__ part, int gps) {
    using G = WideGeo<TOK>;
    extern __shared__ __align__(128) uint8_t smem[];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int gid = lane >> 2, lid = lane & 3;
    const int row0 = blockIdx.x * kWideRows;
    const int g0 = blockIdx.y * gps;
    const int groups = min(gps, p.k / kWideK - g0);
    const long long scale_plane = (p.parent_rows * p.k + 255) / 256 * 256;

    auto stage = [&](int st, int k0) {
        uint8_t* cs = smem + st * G::STAGE;
        // Codes: 32 rows x 64 bytes, 16-byte chunks XOR-swizzled by row pair.
        for (int i = tid; i < kWideRows * 4; i += kWideWarps * 32) {
            const int r = i >> 2, c = i & 3;
            const bool ok = row0 + r < p.rows;
            const long long wr = p.row_begin + (ok ? row0 + r : 0);
            cp_async16(cs + r * kWideK + ((c ^ ((r >> 1) & 3)) * 16), p.w + wr * p.k + k0 + c * 16, ok);
        }
        // Activations: TOK rows x 64 bf16 (rows past m zero-filled).
        uint8_t* as = cs + G::CODE;
        for (int i = tid; i < TOK * 8; i += kWideWarps * 32) {
            const int t = i >> 3, c = i & 7;
            const bool ok = t < p.m;
            cp_async16(as + t * kWideAS * 2 + c * 16, xb + static_cast<long long>(ok ? t : 0) * p.k + k0 + c * 8, ok);
        }
    };

    float acc[TOK / 8][4] = {};
#pragma unroll
    for (int st = 0; st < kWideStages - 1; ++st) {
        if (st < groups) stage(st, (g0 + st) * kWideK);
        cp_commit();
    }
    const int r = warp * 16 + gid;  // block-local rows r and r + 8
    for (int g = 0; g < groups; ++g) {
        cp_wait<kWideStages - 2>();
        __syncthreads();
        if (g + kWideStages - 1 < groups)
            stage((g + kWideStages - 1) % kWideStages, (g0 + g + kWideStages - 1) * kWideK);
        cp_commit();
        const uint8_t* cs = smem + (g % kWideStages) * G::STAGE;
        const __nv_bfloat16* as = reinterpret_cast<const __nv_bfloat16*>(cs + G::CODE);
        auto pair = [&](int rr, int c) {
            return *reinterpret_cast<const uint16_t*>(cs + rr * kWideK + (((c >> 4) ^ ((rr >> 1) & 3)) * 16) + (c & 15));
        };
#pragma unroll
        for (int ks = 0; ks < kWideK / 16; ++ks) {
            const int col = ks * 16 + 2 * lid;
            const uint32_t a0 = e4m3x2_bf16x2(pair(r, col)), a1 = e4m3x2_bf16x2(pair(r + 8, col));
            const uint32_t a2 = e4m3x2_bf16x2(pair(r, col + 8)), a3 = e4m3x2_bf16x2(pair(r + 8, col + 8));
#pragma unroll
            for (int nt = 0; nt < TOK / 8; ++nt) {
                const __nv_bfloat16* xr = as + (nt * 8 + gid) * kWideAS + col;
                mma_bf16(acc[nt], a0, a1, a2, a3, *reinterpret_cast<const uint32_t*>(xr),
                         *reinterpret_cast<const uint32_t*>(xr + 8));
            }
        }
    }
    cp_wait<0>();
    const int ra = row0 + r, rb = ra + 8;
    if (part != nullptr) {
        float* out = part + static_cast<long long>(blockIdx.y) * p.m * p.rows;
#pragma unroll
        for (int nt = 0; nt < TOK / 8; ++nt) {
            const int t = nt * 8 + 2 * lid;
            if (t < p.m) {
                if (ra < p.rows) out[static_cast<long long>(t) * p.rows + ra] = acc[nt][0];
                if (rb < p.rows) out[static_cast<long long>(t) * p.rows + rb] = acc[nt][2];
            }
            if (t + 1 < p.m) {
                if (ra < p.rows) out[static_cast<long long>(t + 1) * p.rows + ra] = acc[nt][1];
                if (rb < p.rows) out[static_cast<long long>(t + 1) * p.rows + rb] = acc[nt][3];
            }
        }
        return;
    }
    const uint16_t* sp = reinterpret_cast<const uint16_t*>(p.w + scale_plane);
    const float sa = ra < p.rows ? __uint_as_float(static_cast<uint32_t>(sp[p.row_begin + ra]) << 16) : 0.0f;
    const float sb = rb < p.rows ? __uint_as_float(static_cast<uint32_t>(sp[p.row_begin + rb]) << 16) : 0.0f;
#pragma unroll
    for (int nt = 0; nt < TOK / 8; ++nt) {
        const int t = nt * 8 + 2 * lid;
        if (t < p.m) {
            if (ra < p.rows) p.y[static_cast<long long>(t) * p.rows + ra] = acc[nt][0] * sa;
            if (rb < p.rows) p.y[static_cast<long long>(t) * p.rows + rb] = acc[nt][2] * sb;
        }
        if (t + 1 < p.m) {
            if (ra < p.rows) p.y[static_cast<long long>(t + 1) * p.rows + ra] = acc[nt][1] * sa;
            if (rb < p.rows) p.y[static_cast<long long>(t + 1) * p.rows + rb] = acc[nt][3] * sb;
        }
    }
}

// y[t][row] = scale(row) x the splits' partial sums, added in split order.
__global__ void wide_reduce_kernel(A16Args p, const float* __restrict__ part, int splits) {
    const long long n = static_cast<long long>(p.m) * p.rows;
    const long long scale_plane = (p.parent_rows * p.k + 255) / 256 * 256;
    const uint16_t* sp = reinterpret_cast<const uint16_t*>(p.w + scale_plane);
    for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        float s = 0.0f;
        for (int k = 0; k < splits; ++k) s += part[k * n + i];
        const int row = static_cast<int>(i % p.rows);
        p.y[i] = s * __uint_as_float(static_cast<uint32_t>(sp[p.row_begin + row]) << 16);
    }
}

// Row blocks below this many get their K split across blocks (about four per SM).
constexpr int kWideMinBlocks = 680;

template <int TOK>
int launch_wide(const A16Args& a) {
    using G = WideGeo<TOK>;
    // bf16 activations, grow-only and never freed (VeloxScratch): recorded graphs may hold them.
    static VeloxScratch scratch;
    const long long n = static_cast<long long>(a.m) * a.k;
    auto* xb = static_cast<__nv_bfloat16*>(scratch.get(static_cast<size_t>(n) * 2));
    if (xb == nullptr) return static_cast<int>(cudaErrorMemoryAllocation);
    to_bf16_kernel<<<velox_grid_1d(n / 2, 256), 256>>>(reinterpret_cast<const float2*>(a.x),
                                                       reinterpret_cast<__nv_bfloat162*>(xb), n / 2);
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(a16_fp8_wide_kernel<TOK>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM);
        configured = true;
    }
    const int blocks = (a.rows + kWideRows - 1) / kWideRows, groups = a.k / kWideK;
    int splits = (kWideMinBlocks + blocks - 1) / blocks;
    splits = splits < 1 ? 1 : splits > groups ? groups : splits > 8 ? 8 : splits;
    const int gps = (groups + splits - 1) / splits;
    splits = (groups + gps - 1) / gps;
    float* part = nullptr;
    if (splits > 1) {
        static VeloxScratch parts;
        part = static_cast<float*>(parts.get(static_cast<size_t>(splits) * a.m * a.rows * sizeof(float)));
        if (part == nullptr) return static_cast<int>(cudaErrorMemoryAllocation);
    }
    a16_fp8_wide_kernel<TOK><<<dim3(blocks, splits), kWideWarps * 32, G::SMEM>>>(a, xb, part, gps);
    if (splits > 1)
        wide_reduce_kernel<<<velox_grid_1d(static_cast<long long>(a.m) * a.rows, 256), 256>>>(a, part, splits);
    return velox_launch_status();
}

}  // namespace

// y[m, rows] = bf16(x) . W[row_begin.., :]^T x row scale, for an FP8 row-scaled weight. Requires
// 1 <= m <= 64, k % 256 == 0 and 16-byte aligned x rows; returns VELOX_ERR_UNSUPPORTED otherwise.
// Batched speculative verify (several sequences' windows) takes up to 64 rows in one pass over the
// weights (the wide kernel above).
VELOX_API int velox_5090_linear_a16_fp8(const float* x, int m, const uint8_t* w, long long parent_rows, long long k,
                                        long long row_begin, int rows, float* y) {
    if (m < 1 || m > 64 || k % 256 != 0 || k > 0x7fffffff || reinterpret_cast<uintptr_t>(x) % 16 != 0)
        return VELOX_ERR_UNSUPPORTED;
    const A16Args a{x, w, parent_rows, row_begin, rows, m, static_cast<int>(k), y};
    const bool k512 = k % 512 == 0;
    if (m <= 8) return k512 ? launch_a16<8, 8>(a) : launch_a16<8, 4>(a);
    if (m <= 16) return k512 ? launch_a16<16, 8>(a) : launch_a16<16, 4>(a);
    if (m <= 32) return launch_wide<32>(a);
    return launch_wide<64>(a);
}
