// RTX 5060 Ti (from rtx-5090-32g) skinny linear (1..64 tokens) for FP8 E4M3 row-scaled weights with bf16 activations
// (ActQuant::Bf16): decode, speculative verify and batched verify of several sequences.
//
// A block owns R = 16 * RW weight rows and every token; each stage streams R x BK weight codes and
// TOK x BK bf16 activations (rounded once per call, RNE) through a cp.async pipeline. Warp w takes
// row group w % RW and K slice w / RW (64 K values) of every stage; E4M3 codes widen exactly to
// bf16 (via f16) as the A operand of mma.m16n8k16, activations are the B operand (f32 accumulate).
// The K-slice warps of a row group meet in shared memory, and the row's BF16 scale multiplies the
// complete dot product. Large row tiles keep the activations' L2 traffic below the weights'; matrices
// with few row tiles split K across blocks (partial sums reduced in split order).
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include "velox_common.cuh"

namespace {

constexpr int kWarpK = 64;  // K values per warp per stage

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

struct A16Args {
    const float* x;      // [m, k]
    const uint8_t* w;    // parent payload: [parent_rows, k] E4M3 codes, then BF16 row scales
    long long parent_rows, row_begin;
    int rows, m, k;
    float* y;            // [m, rows]
};

__device__ __forceinline__ float row_scale(const A16Args& p, int row) {
    const long long scale_plane = (p.parent_rows * p.k + 255) / 256 * 256;
    const uint16_t* sp = reinterpret_cast<const uint16_t*>(p.w + scale_plane);
    return __uint_as_float(static_cast<uint32_t>(sp[p.row_begin + row]) << 16);
}

template <int TOK, int RW, int KS, int ST>
struct TileGeo {
    static constexpr int R = 16 * RW;           // weight rows per block
    static constexpr int BK = KS * kWarpK;      // K per stage
    static constexpr int AS = BK + 8;           // bf16 activation row stride (elements): spreads banks
    static constexpr int CODE = R * BK;         // bytes
    static constexpr int STAGE = CODE + TOK * AS * 2;
    static constexpr int PART = KS > 1 ? RW * KS * (TOK / 8) * 32 * 16 : 0;  // cross-warp partials
    static constexpr int SMEM = ST * STAGE > PART ? ST * STAGE : PART;
    static constexpr int THREADS = RW * KS * 32;
};

// Split-K: blockIdx.y takes stages [y * sps, (y + 1) * sps); with several splits it writes unscaled
// partial sums to part[y][t][row] for reduce_kernel, else the scaled outputs.
template <int TOK, int RW, int KS, int ST>
__global__ void __launch_bounds__(RW * KS * 32)
    a16_fp8_tile_kernel(A16Args p, const __nv_bfloat16* __restrict__ xb, float* __restrict__ part, int sps) {
    using G = TileGeo<TOK, RW, KS, ST>;
    extern __shared__ __align__(128) uint8_t smem[];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int gid = lane >> 2, lid = lane & 3;
    const int rg = warp % RW, ks = warp / RW;
    const int row0 = blockIdx.x * G::R;
    const int s0 = blockIdx.y * sps;
    const int stages = min(sps, p.k / G::BK - s0);

    auto stage = [&](int st, int k0) {
        uint8_t* cs = smem + st * G::STAGE;
        // Codes: R rows x BK bytes, 16-byte chunks XOR-swizzled by (row & 7).
        for (int i = tid; i < G::R * (G::BK / 16); i += G::THREADS) {
            const int r = i / (G::BK / 16), c = i % (G::BK / 16);
            const bool ok = row0 + r < p.rows;
            const long long wr = p.row_begin + (ok ? row0 + r : 0);
            cp_async16(cs + r * G::BK + ((c ^ (r & 7)) * 16), p.w + wr * p.k + k0 + c * 16, ok);
        }
        // Activations: TOK rows x BK bf16 (rows past m zero-filled).
        __nv_bfloat16* as = reinterpret_cast<__nv_bfloat16*>(cs + G::CODE);
        for (int i = tid; i < TOK * (G::BK / 8); i += G::THREADS) {
            const int t = i / (G::BK / 8), c = i % (G::BK / 8);
            const bool ok = t < p.m;
            cp_async16(as + t * G::AS + c * 8, xb + static_cast<long long>(ok ? t : 0) * p.k + k0 + c * 8, ok);
        }
    };

    float acc[TOK / 8][4] = {};
#pragma unroll
    for (int st = 0; st < ST - 1; ++st) {
        if (st < stages) stage(st, (s0 + st) * G::BK);
        cp_commit();
    }
    const int r = rg * 16 + gid;  // block-local rows r and r + 8
    for (int g = 0; g < stages; ++g) {
        cp_wait<ST - 2>();
        __syncthreads();
        if (g + ST - 1 < stages) stage((g + ST - 1) % ST, (s0 + g + ST - 1) * G::BK);
        cp_commit();
        const uint8_t* cs = smem + (g % ST) * G::STAGE;
        const __nv_bfloat16* as = reinterpret_cast<const __nv_bfloat16*>(cs + G::CODE);
        auto pair = [&](int rr, int c) {
            return *reinterpret_cast<const uint16_t*>(cs + rr * G::BK + (((c >> 4) ^ (rr & 7)) * 16) + (c & 15));
        };
#pragma unroll
        for (int kk = 0; kk < kWarpK / 16; ++kk) {
            const int col = ks * kWarpK + kk * 16 + 2 * lid;  // within the stage
            const uint32_t a0 = e4m3x2_bf16x2(pair(r, col)), a1 = e4m3x2_bf16x2(pair(r + 8, col));
            const uint32_t a2 = e4m3x2_bf16x2(pair(r, col + 8)), a3 = e4m3x2_bf16x2(pair(r + 8, col + 8));
#pragma unroll
            for (int nt = 0; nt < TOK / 8; ++nt) {
                const __nv_bfloat16* xr = as + (nt * 8 + gid) * G::AS + col;
                mma_bf16(acc[nt], a0, a1, a2, a3, *reinterpret_cast<const uint32_t*>(xr),
                         *reinterpret_cast<const uint32_t*>(xr + 8));
            }
        }
    }
    cp_wait<0>();
    if constexpr (KS > 1) {
        // The K-slice warps of each row group add up in slice order (warp ks == 0 finishes).
        __syncthreads();
        float* pt = reinterpret_cast<float*>(smem);
#pragma unroll
        for (int nt = 0; nt < TOK / 8; ++nt)
            *reinterpret_cast<float4*>(pt + (((ks * RW + rg) * (TOK / 8) + nt) * 32 + lane) * 4) =
                make_float4(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3]);
        __syncthreads();
        if (ks != 0) return;
#pragma unroll
        for (int nt = 0; nt < TOK / 8; ++nt) {
            float4 s = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
            for (int q = 0; q < KS; ++q) {
                const float4 v = *reinterpret_cast<const float4*>(pt + (((q * RW + rg) * (TOK / 8) + nt) * 32 + lane) * 4);
                s.x += v.x, s.y += v.y, s.z += v.z, s.w += v.w;
            }
            acc[nt][0] = s.x, acc[nt][1] = s.y, acc[nt][2] = s.z, acc[nt][3] = s.w;
        }
    }
    const int ra = row0 + r, rb = ra + 8;
    float sa = 1.0f, sb = 1.0f;
    float* out = p.y;
    if (part != nullptr) {
        out = part + static_cast<long long>(blockIdx.y) * p.m * p.rows;
    } else {
        sa = ra < p.rows ? row_scale(p, ra) : 0.0f;
        sb = rb < p.rows ? row_scale(p, rb) : 0.0f;
    }
#pragma unroll
    for (int nt = 0; nt < TOK / 8; ++nt) {
        const int t = nt * 8 + 2 * lid;
        if (t < p.m) {
            if (ra < p.rows) out[static_cast<long long>(t) * p.rows + ra] = acc[nt][0] * sa;
            if (rb < p.rows) out[static_cast<long long>(t) * p.rows + rb] = acc[nt][2] * sb;
        }
        if (t + 1 < p.m) {
            if (ra < p.rows) out[static_cast<long long>(t + 1) * p.rows + ra] = acc[nt][1] * sa;
            if (rb < p.rows) out[static_cast<long long>(t + 1) * p.rows + rb] = acc[nt][3] * sb;
        }
    }
}

// y[t][row] = scale(row) x the splits' partial sums, added in split order.
__global__ void reduce_kernel(A16Args p, const float* __restrict__ part, int splits) {
    const long long n = static_cast<long long>(p.m) * p.rows;
    for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        float s = 0.0f;
        for (int k = 0; k < splits; ++k) s += part[k * n + i];
        p.y[i] = s * row_scale(p, static_cast<int>(i % p.rows));
    }
}

__global__ void to_bf16_kernel(const float2* __restrict__ x, __nv_bfloat162* __restrict__ y, long long pairs) {
    for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < pairs;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const float2 v = x[i];
        y[i] = __floats2bfloat162_rn(v.x, v.y);
    }
}

// x rounded to bf16 in grow-only scratch (never freed, VeloxScratch: recorded graphs may hold it).
const __nv_bfloat16* to_bf16(const A16Args& a) {
    static VeloxScratch scratch;
    const long long n = static_cast<long long>(a.m) * a.k;
    auto* xb = static_cast<__nv_bfloat16*>(scratch.get(static_cast<size_t>(n) * 2));
    if (xb != nullptr)
        to_bf16_kernel<<<velox_grid_1d(n / 2, 256), 256>>>(reinterpret_cast<const float2*>(a.x),
                                                           reinterpret_cast<__nv_bfloat162*>(xb), n / 2);
    return xb;
}

// Row tiles below this many get their K split across blocks (about two per SM).
constexpr int kMinBlocks = 340;

template <int TOK, int RW, int KS, int ST>
int launch_tile(const A16Args& a) {
    using G = TileGeo<TOK, RW, KS, ST>;
    if (a.k % G::BK != 0) return VELOX_ERR_UNSUPPORTED;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(a16_fp8_tile_kernel<TOK, RW, KS, ST>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             G::SMEM);
        configured = true;
    }
    const __nv_bfloat16* xb = to_bf16(a);
    if (xb == nullptr) return static_cast<int>(cudaErrorMemoryAllocation);
    const int blocks = (a.rows + G::R - 1) / G::R, total = a.k / G::BK;
    int splits = (kMinBlocks + blocks - 1) / blocks;
    splits = splits < 1 ? 1 : splits > 8 ? 8 : splits;
    // Each split keeps at least two pipeline depths of stages.
    while (splits > 1 && total / splits < 2 * ST) --splits;
    const int sps = (total + splits - 1) / splits;
    splits = (total + sps - 1) / sps;
    float* part = nullptr;
    if (splits > 1) {
        static VeloxScratch parts;
        part = static_cast<float*>(parts.get(static_cast<size_t>(splits) * a.m * a.rows * sizeof(float)));
        if (part == nullptr) return static_cast<int>(cudaErrorMemoryAllocation);
    }
    a16_fp8_tile_kernel<TOK, RW, KS, ST><<<dim3(blocks, splits), G::THREADS, G::SMEM>>>(a, xb, part, sps);
    if (splits > 1)
        reduce_kernel<<<velox_grid_1d(static_cast<long long>(a.m) * a.rows, 256), 256>>>(a, part, splits);
    return velox_launch_status();
}

}  // namespace

// y[m, rows] = bf16(x) . W[row_begin.., :]^T x row scale, for an FP8 row-scaled weight. Requires
// 1 <= m <= 64, k % 256 == 0 and 16-byte aligned x rows; returns VELOX_ERR_UNSUPPORTED otherwise.
VELOX_API int velox_5060_linear_a16_fp8(const float* x, int m, const uint8_t* w, long long parent_rows, long long k,
                                        long long row_begin, int rows, float* y) {
    if (m < 1 || m > 64 || k % 256 != 0 || k > 0x7fffffff || reinterpret_cast<uintptr_t>(x) % 16 != 0)
        return VELOX_ERR_UNSUPPORTED;
    const A16Args a{x, w, parent_rows, row_begin, rows, m, static_cast<int>(k), y};
    // Shapes measured on RTX 5090 (bench-linear --format fp8): 16 warps over two row groups and eight
    // K slices up to 16 tokens; wider batches trade K slices for row groups so stages fit.
    if (m <= 8) return launch_tile<8, 2, 8, 3>(a);
    if (m <= 16) return launch_tile<16, 2, 8, 3>(a);
    if (m <= 32) return launch_tile<32, 4, 4, 3>(a);
    return launch_tile<64, 4, 2, 3>(a);
}
