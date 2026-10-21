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

}  // namespace

// y[m, rows] = bf16(x) . W[row_begin.., :]^T x row scale, for an FP8 row-scaled weight. Requires
// 1 <= m <= 64, k % 256 == 0 and 16-byte aligned x rows; returns VELOX_ERR_UNSUPPORTED otherwise.
// Batched speculative verify (several sequences' windows) takes up to 64 rows in one pass over the
// weights; wider token tiles split K over fewer warps so both pipeline stages fit shared memory.
VELOX_API int velox_5090_linear_a16_fp8(const float* x, int m, const uint8_t* w, long long parent_rows, long long k,
                                        long long row_begin, int rows, float* y) {
    if (m < 1 || m > 64 || k % 256 != 0 || k > 0x7fffffff || reinterpret_cast<uintptr_t>(x) % 16 != 0)
        return VELOX_ERR_UNSUPPORTED;
    const A16Args a{x, w, parent_rows, row_begin, rows, m, static_cast<int>(k), y};
    const bool k512 = k % 512 == 0;
    if (m <= 8) return k512 ? launch_a16<8, 8>(a) : launch_a16<8, 4>(a);
    if (m <= 16) return k512 ? launch_a16<16, 8>(a) : launch_a16<16, 4>(a);
    if (m <= 32) return launch_a16<32, 4>(a);
    return launch_a16<64, 2>(a);
}
