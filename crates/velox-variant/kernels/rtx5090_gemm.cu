// RTX 5090 tensor-core linear for prefill: activations are rounded on the fly to the precision the
// artifact allows (NVFP4 for AllowA4 + NVFP4 weights, E4M3 for AllowA8 + FP8 weights) and
// multiplied with block-scaled MMA on sm_120a:
//
//   A4 x W4: mma.sync.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64 (E2M1, UE4M3 per 16)
//   A8 x W8: mma.sync.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32 with unit UE8M0 scales,
//            which runs at full rate with FP32 accumulation; row scales are applied after.
//
// Both operands are 32 bytes of codes per row per MMA k-step, so one tile loop serves both. The
// host-side reference rounding is velox_core::quant; the quantizers below perform the same
// single-precision operations in the same order.
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>

#include <mutex>
#include <unordered_map>

#include "velox_common.cuh"

namespace {

// ---------------------------------------------------------------------------------------------
// Activation quantizers.

__device__ __forceinline__ uint8_t to_e4m3(float v) {
    return static_cast<uint8_t>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ float from_e4m3(uint8_t b) {
    const __half_raw h = __nv_cvt_fp8_to_halfraw(b, __NV_E4M3);
    return __half2float(__half(h));
}

// Two E2M1 codes in one byte: `lo` in bits 0-3, `hi` in bits 4-7 (RNE, saturating).
__device__ __forceinline__ uint32_t e2m1x2(float lo, float hi) {
    uint32_t r;
    asm("{\n.reg .b8 b;\ncvt.rn.satfinite.e2m1x2.f32 b, %1, %2;\ncvt.u32.u8 %0, b;\n}"
        : "=r"(r)
        : "f"(hi), "f"(lo));
    return r;
}

// One thread per 16-element group. Rows in [m, m_pad) are written as zeros so the GEMM can read
// whole tiles. codes: [m_pad, k/2], scales: [m_pad, k/16].
// KMAJOR (the TMA path): scales[(k64 * m_pad + row) * 4 + q] for group 4 * k64 + q, so a k64 step's
// words for consecutive rows are contiguous.
template <bool KMAJOR = false>
__global__ void quant_nvfp4_kernel(const float* __restrict__ x, int m, int m_pad, int k, float divisor,
                                   uint8_t* __restrict__ codes, uint8_t* __restrict__ scales) {
    const int groups = k / 16;
    const long long g = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= static_cast<long long>(m_pad) * groups) return;
    const int row = static_cast<int>(g / groups), grp = static_cast<int>(g % groups);
    uint2 packed = make_uint2(0, 0);
    uint8_t s = 0;
    if (row < m) {
        const float4* src = reinterpret_cast<const float4*>(x + static_cast<long long>(row) * k + grp * 16);
        float v[16];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float4 t = src[q];
            v[4 * q] = t.x, v[4 * q + 1] = t.y, v[4 * q + 2] = t.z, v[4 * q + 3] = t.w;
        }
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(v[i]));
        s = to_e4m3(__fdiv_rn(__fmul_rn(divisor, amax), 6.0f));
        if (s != 0) {
            const float ds = from_e4m3(s);
            uint32_t w[2] = {0, 0};
#pragma unroll
            for (int i = 0; i < 16; i += 2) {
                const float a = __fdiv_rn(__fmul_rn(v[i], divisor), ds);
                const float b = __fdiv_rn(__fmul_rn(v[i + 1], divisor), ds);
                w[i / 8] |= e2m1x2(a, b) << (8 * ((i / 2) % 4));
            }
            packed = make_uint2(w[0], w[1]);
        }
    }
    *reinterpret_cast<uint2*>(codes + g * 8) = packed;
    if (KMAJOR) scales[(static_cast<long long>(grp / 4) * m_pad + row) * 4 + grp % 4] = s;
    else scales[g] = s;
}

// One block per row: scale = amax / 448, codes = e4m3(x * (1 / scale)). codes: [m_pad, k].
__global__ void quant_fp8_kernel(const float* __restrict__ x, int m, int k, uint8_t* __restrict__ codes,
                                 float* __restrict__ row_scale) {
    __shared__ float scratch[VELOX_WARP];
    const int row = blockIdx.x;
    uint8_t* out = codes + static_cast<long long>(row) * k;
    if (row >= m) {
        for (int c = threadIdx.x * 4; c < k; c += blockDim.x * 4) *reinterpret_cast<uint32_t*>(out + c) = 0;
        if (threadIdx.x == 0) row_scale[row] = 0.0f;
        return;
    }
    const float* xr = x + static_cast<long long>(row) * k;
    float amax = 0.0f;
    for (int c = threadIdx.x * 4; c < k; c += blockDim.x * 4) {
        const float4 t = *reinterpret_cast<const float4*>(xr + c);
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(t.x), fabsf(t.y)), fmaxf(fabsf(t.z), fabsf(t.w))));
    }
    // Block max via the sum helper's pattern.
    amax = velox_warp_max(amax);
    if (threadIdx.x % VELOX_WARP == 0) scratch[threadIdx.x / VELOX_WARP] = amax;
    __syncthreads();
    if (threadIdx.x < VELOX_WARP) {
        float v = threadIdx.x < blockDim.x / VELOX_WARP ? scratch[threadIdx.x] : 0.0f;
        v = velox_warp_max(v);
        if (threadIdx.x == 0) scratch[0] = v;
    }
    __syncthreads();
    const float scale = __fdiv_rn(scratch[0], 448.0f);
    const float inv = scale > 0.0f ? __fdiv_rn(1.0f, scale) : 0.0f;
    for (int c = threadIdx.x * 4; c < k; c += blockDim.x * 4) {
        const float4 t = *reinterpret_cast<const float4*>(xr + c);
        const uint32_t w = to_e4m3(__fmul_rn(t.x, inv)) | (to_e4m3(__fmul_rn(t.y, inv)) << 8) |
                           (to_e4m3(__fmul_rn(t.z, inv)) << 16) | (static_cast<uint32_t>(to_e4m3(__fmul_rn(t.w, inv))) << 24);
        *reinterpret_cast<uint32_t*>(out + c) = w;
    }
    if (threadIdx.x == 0) row_scale[row] = scale;
}

// ---------------------------------------------------------------------------------------------
// GEMM. Tile configurations: Cfg<WM, WN, MT, .., NT> has WM x WN warps, each covering MT x 16
// tokens and NT x 8 weight rows, so a block is BM = WM * MT * 16 tokens by BN = WN * NT * 8 rows.
//   Wide:  256 x 128, 256 threads, 64 x 64 per warp (prefill grids of at least kWideBlocks)
//   Big:   128 x 128, 256 threads, 64 x 32 per warp (smaller prefill grids)
//   Small:  16 / 32 / 64 x 64, 64 threads (verify, drafts, short chunks: m <= 64), with split-K so a
//           handful of tokens still spreads the weight stream over every SM.


enum Kind { kFp4 = 0, kFp8 = 1 };

template <int WM_, int WN_, int MT_, int ST_, int KS_, int NT_ = 4>
struct Cfg {
    // ST: cp.async pipeline stages. KS: MMA k-steps (32 code bytes each) per stage, so each weight
    // row is read in BK-byte contiguous pieces; longer pieces keep DRAM pages open for skinny m.
    // NT: n8 tiles per warp (each warp covers MT x 16 tokens by NT x 8 weight rows).
    static constexpr int WM = WM_, WN = WN_, MT = MT_, ST = ST_, KS = KS_, BK = KS_ * 32, NT = NT_;
    static constexpr int BM = WM * MT * 16, BN = WN * NT * 8, THREADS = WM * WN * 32;
};
// Prefill. Wide's 64 x 64 warp tiles issue 32 MMAs per k-step per warp (Big: 16) and reuse each
// operand fragment more, but a quarter-size grid under-fills 170 SMs on narrow outputs. RTX 5090,
// m = 512 / 2048 (us): NVFP4 [34816,5120] Big 340 / 1300, Wide 293 / 946; [5120,17408] Big
// 114 / 468, Wide 201 / 485; FP8 [16384,5120] Big 247 / 733, Wide 221 / 725.
using Wide = Cfg<4, 2, 4, 3, 2, 8>;
using Big = Cfg<2, 4, 4, 3, 2>;
// Skinny m (decode, verify, drafts) is weight-streaming: 64 rows per block, 128 contiguous bytes
// per row per stage and four stages measured best on the RTX 5090 (m = 8: lm_head 1.05 -> 1.53
// TB/s, MLP gate|up 0.96 -> 1.29 TB/s).
// The tile is as tall as the batch (16 / 32 / 64 tokens), so a batched verify of several lanes
// still streams the weights once (16-row tiles took two or four passes at m = 17..64).
template <int MT>
using SmallT = Cfg<1, 2, MT, 4, 4>;
template <int MT>
using Small64T = Cfg<1, 2, MT, 3, 2>;  // K whose code bytes are not a multiple of SmallT::BK
using Small = SmallT<1>;

template <class C, bool SW>
struct Smem {
    uint8_t a[C::ST][C::BM * C::BK];
    uint8_t b[C::ST][C::BN * C::BK];
    uint32_t a_scale[C::ST][C::BM][C::KS];  // FP4: per row, the scale word of each of the stage's k64 steps
    // FP4: the weight's swizzled 128x4 scale tiles, as stored (SwiGLU: gate's, then up's).
    uint8_t b_scale[C::ST][C::KS][SW ? 2 : 1][512];
};

// 16-byte chunk c of row r lives at chunk c ^ ((r >> 1) & 3) for 64-byte rows and c ^ (r & 7) for
// 128 bytes and wider: ldmatrix over 8 rows is conflict-free either way.
template <int BK>
__device__ __forceinline__ int swz(int row, int byte) {
    constexpr int chunks = BK / 16;
    const int x = BK == 64 ? (row >> 1) & 3 : row & 7;
    return row * BK + ((((byte >> 4) ^ x) & (chunks - 1)) << 4) + (byte & 15);
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src),
                 "r"(valid ? 16 : 0));
}

__device__ __forceinline__ void cp_async8(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(smem_u32(dst)), "l"(src));
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_u32(p)));
}

__device__ __forceinline__ void mma_fp4(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1,
                                        uint32_t sfa, uint32_t sfb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "h"((uint16_t)0),
          "h"((uint16_t)0), "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
}

__device__ __forceinline__ void mma_fp8(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0x7fu), "h"((uint16_t)0),
          "h"((uint16_t)0), "r"(0x7fu), "h"((uint16_t)0), "h"((uint16_t)0));
}

struct GemmArgs {
    const uint8_t* a;         // [m_pad, kb] activation codes
    const uint8_t* a_scales;  // FP4: [m_pad, k/16] UE4M3;  FP8: unused
    const float* a_row;       // FP8: [m_pad] row scales
    const uint8_t* w;         // weight payload (parent)
    long long parent_rows;
    long long row_begin;
    int rows;                 // view rows
    int m;
    int k;                    // logical K
    int kb;                   // code bytes per row
    float alpha;              // FP4: 1 / (input divisor * weight divisor)
    int kt_per_split;         // stages per blockIdx.z slice
    float* y;                 // [m, rows], or [splits, m, rows] partials when split
    // SwiGLU (y = silu(x . gate^T) * (x . up^T)): w / parent_rows / row_begin / alpha are gate's,
    // these up's; rows counts output columns.
    const uint8_t* w2;
    long long parent_rows2;
    long long row_begin2;
    float alpha2;
    int m_pad;                // TMA path: activation rows as quantized (its scales are k64-major)
};

__device__ __forceinline__ const uint8_t* fp4_scale_plane(const uint8_t* w, long long parent_rows, int kb) {
    return w + (parent_rows * kb + 255) / 256 * 256;
}

// SwiGLU tiles: a block computes BN / 2 output columns from n0. Each warp's NT x 8 weight rows are
// the gate rows of its NT x 4 columns followed by the up rows of the same columns, so a thread
// holds both accumulators of every output it writes. B row `row` of the tile: (up?, column).
template <class C>
__device__ __forceinline__ void swiglu_row(int row, int& half, int& col) {
    constexpr int per_warp = C::NT * 8, half_rows = C::NT * 4;
    const int within = row % per_warp;
    half = within / half_rows;
    col = (row / per_warp) * half_rows + within % half_rows;
}

template <class C, int KIND, bool SW>
__device__ __forceinline__ void load_stage(Smem<C, SW>& s, const GemmArgs& p, int stage, int kt, int m0, int n0) {
    const int t = threadIdx.x;
    constexpr int chunks = C::BK / 16;
    const int kbyte = kt * C::BK;
#pragma unroll
    for (int idx = t; idx < C::BM * chunks; idx += C::THREADS) {
        const int row = idx / chunks, chunk = idx % chunks;
        cp_async16(s.a[stage] + swz<C::BK>(row, chunk * 16),
                   p.a + static_cast<long long>(m0 + row) * p.kb + kbyte + chunk * 16, true);
    }
#pragma unroll
    for (int idx = t; idx < C::BN * chunks; idx += C::THREADS) {
        const int row = idx / chunks, chunk = idx % chunks;
        const uint8_t* w = p.w;
        long long wrow;
        bool ok;
        if (SW) {
            int half, col;
            swiglu_row<C>(row, half, col);
            ok = n0 + col < p.rows;
            w = half ? p.w2 : p.w;
            wrow = (half ? p.row_begin2 : p.row_begin) + n0 + col;
        } else {
            wrow = p.row_begin + n0 + row;
            ok = wrow < p.parent_rows;
        }
        cp_async16(s.b[stage] + swz<C::BK>(row, chunk * 16), w + (ok ? wrow : 0) * p.kb + kbyte + chunk * 16, ok);
    }
    if (KIND == kFp4) {
        // KS k64 steps per stage. A: 8-byte copies of two scale words per row. B: KS contiguous
        // 512-byte tiles (SwiGLU: gate's and up's; a block's BN / 2 columns lie in one tile).
        const int k64 = kt * C::KS;
        for (int idx = t; idx < C::BM * (C::KS / 2); idx += C::THREADS) {
            const int row = idx / (C::KS / 2), pair = idx % (C::KS / 2);
            cp_async8(&s.a_scale[stage][row][2 * pair],
                      p.a_scales + static_cast<long long>(m0 + row) * (p.k / 16) + (k64 + 2 * pair) * 4);
        }
        constexpr int tiles = SW ? 2 : 1;
        for (int idx = t; idx < C::KS * tiles * 32; idx += C::THREADS) {
            const int step = idx / (tiles * 32), half = (idx >> 5) % tiles, off = (idx & 31) * 16;
            const uint8_t* plane = half ? fp4_scale_plane(p.w2, p.parent_rows2, p.kb) : fp4_scale_plane(p.w, p.parent_rows, p.kb);
            const long long tile = ((half ? p.row_begin2 : p.row_begin) + n0) / 128;
            cp_async16(&s.b_scale[stage][step][half][off], plane + (tile * (p.k / 64) + k64 + step) * 512 + off, true);
        }
    }
}

// Writes a block's tile from its warps' accumulators.
// Epilogue: c0,c1 at (row g, cols 2q, 2q+1); c2,c3 at row g+8. Scales are linear in the
// accumulator, so split-K partials can be scaled before they are summed.
template <class C, int KIND, bool SW>
__device__ __forceinline__ void store_tile(const float (&acc)[C::MT][C::NT][4], const GemmArgs& p, float* y, int m0,
                                           int n0, int wm, int wn, int lane) {
    constexpr int WROWS = C::MT * 16;
    const int g = lane >> 2, q = lane & 3;
    if (SW) {
        // out = silu(gate) * up, as the portable swiglu kernel computes it from the sums scaled as
        // the plain epilogue below scales them (FP4: alpha; FP8: weight row scale, then token's).
        const uint16_t* gs = reinterpret_cast<const uint16_t*>(p.w + (p.parent_rows * p.kb + 255) / 256 * 256);
        const uint16_t* us = reinterpret_cast<const uint16_t*>(p.w2 + (p.parent_rows2 * p.kb + 255) / 256 * 256);
#pragma unroll
        for (int nt = 0; nt < C::NT / 2; ++nt) {
            const int n = n0 + wn * (C::NT * 4) + nt * 8 + 2 * q;
            float fg[2] = {p.alpha, p.alpha}, fu[2] = {p.alpha2, p.alpha2};
            if (KIND == kFp8) {
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const bool in = n + e < p.rows;
                    fg[e] = in ? __uint_as_float(static_cast<uint32_t>(gs[p.row_begin + n + e]) << 16) : 0.0f;
                    fu[e] = in ? __uint_as_float(static_cast<uint32_t>(us[p.row_begin2 + n + e]) << 16) : 0.0f;
                }
            }
#pragma unroll
            for (int mt = 0; mt < C::MT; ++mt) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int t = m0 + wm * WROWS + mt * 16 + g + h * 8;
                    if (t >= p.m) continue;
                    const float xs = KIND == kFp8 ? p.a_row[t] : 1.0f;
                    float v[2];
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const float gv = acc[mt][nt][2 * h + e] * fg[e] * xs;
                        const float uv = acc[mt][nt + C::NT / 2][2 * h + e] * fu[e] * xs;
                        v[e] = gv / (1.0f + expf(-gv)) * uv;
                    }
                    float* out = y + static_cast<long long>(t) * p.rows + n;
                    if (n + 1 < p.rows && (p.rows & 1) == 0) {
                        *reinterpret_cast<float2*>(out) = make_float2(v[0], v[1]);
                    } else {
                        if (n < p.rows) out[0] = v[0];
                        if (n + 1 < p.rows) out[1] = v[1];
                    }
                }
            }
        }
        return;
    }
    const uint8_t* wscale = p.w + (p.parent_rows * p.kb + 255) / 256 * 256;  // FP8 BF16 row scales
#pragma unroll
    for (int nt = 0; nt < C::NT; ++nt) {
        const int n = n0 + wn * (C::NT * 8) + nt * 8 + 2 * q;
        float f0 = p.alpha, f1 = p.alpha;
        if (KIND == kFp8) {
            const long long r0 = p.row_begin + n;
            const uint16_t* sp = reinterpret_cast<const uint16_t*>(wscale);
            f0 = n < p.rows ? __uint_as_float(static_cast<uint32_t>(sp[r0]) << 16) : 0.0f;
            f1 = n + 1 < p.rows ? __uint_as_float(static_cast<uint32_t>(sp[r0 + 1]) << 16) : 0.0f;
        }
#pragma unroll
        for (int mt = 0; mt < C::MT; ++mt) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int t = m0 + wm * WROWS + mt * 16 + g + h * 8;
                if (t >= p.m) continue;
                const float xs = KIND == kFp8 ? p.a_row[t] : 1.0f;
                float* out = y + static_cast<long long>(t) * p.rows + n;
                const float v0 = acc[mt][nt][2 * h] * f0 * xs, v1 = acc[mt][nt][2 * h + 1] * f1 * xs;
                if (n + 1 < p.rows && (p.rows & 1) == 0) {
                    *reinterpret_cast<float2*>(out) = make_float2(v0, v1);
                } else {
                    if (n < p.rows) out[0] = v0;
                    if (n + 1 < p.rows) out[1] = v1;
                }
            }
        }
    }
}

template <class C, int KIND, bool SW>
__global__ void __launch_bounds__(C::THREADS, 1) gemm_kernel(GemmArgs p) {
    extern __shared__ __align__(128) uint8_t smem_raw[];
    Smem<C, SW>& s = *reinterpret_cast<Smem<C, SW>*>(smem_raw);
    const int n0 = blockIdx.x * (SW ? C::BN / 2 : C::BN), m0 = blockIdx.y * C::BM;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int wm = warp / C::WN, wn = warp % C::WN;
    const int kt_begin = blockIdx.z * p.kt_per_split;
    const int kt_end = min(p.kb / C::BK, kt_begin + p.kt_per_split);
    const int kt_count = kt_end - kt_begin;

#pragma unroll
    for (int st = 0; st < C::ST - 1; ++st) {
        if (st < kt_count) load_stage<C, KIND, SW>(s, p, st, kt_begin + st, m0, n0);
        cp_commit();
    }

    float acc[C::MT][C::NT][4];
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int j = 0; j < C::NT; ++j)
#pragma unroll
            for (int v = 0; v < 4; ++v) acc[i][j][v] = 0.0f;

    // ldmatrix lane roles. A x4: matrices (rows 0-7 | 8-15) x (bytes 0-15 | 16-31).
    const int a_row = (lane & 7) + (((lane >> 3) & 1) << 3), a_byte = (lane >> 4) << 4;
    // B x4 covers two n8 tiles: matrices (tile0 lo, tile0 hi, tile1 lo, tile1 hi).
    const int b_row = (lane & 7) + ((lane >> 4) << 3), b_byte = ((lane >> 3) & 1) << 4;
    const int sfa_row = ((lane & 1) << 3) | (lane >> 2);
    const int sfb_row = lane >> 2;
    constexpr int WROWS = C::MT * 16;

    for (int i = 0; i < kt_count; ++i) {
        cp_wait<C::ST - 2>();
        __syncthreads();
        const int next = i + C::ST - 1;
        if (next < kt_count) load_stage<C, KIND, SW>(s, p, next % C::ST, kt_begin + next, m0, n0);
        cp_commit();
        const int st = i % C::ST;
#pragma unroll
        for (int ks = 0; ks < C::KS; ++ks) {
            uint32_t af[C::MT][4], bf[C::NT][2], sfa[C::MT], sfb[C::NT];
#pragma unroll
            for (int mt = 0; mt < C::MT; ++mt) {
                const int r = wm * WROWS + mt * 16 + a_row;
                ldmatrix_x4(af[mt], s.a[st] + swz<C::BK>(r, ks * 32 + a_byte));
                if (KIND == kFp4) sfa[mt] = s.a_scale[st][wm * WROWS + mt * 16 + sfa_row][ks];
            }
#pragma unroll
            for (int nt = 0; nt < C::NT; nt += 2) {
                const int r = wn * (C::NT * 8) + nt * 8 + b_row;
                uint32_t t4[4];
                ldmatrix_x4(t4, s.b[st] + swz<C::BK>(r, ks * 32 + b_byte));
                bf[nt][0] = t4[0], bf[nt][1] = t4[1], bf[nt + 1][0] = t4[2], bf[nt + 1][1] = t4[3];
            }
            if (KIND == kFp4) {
#pragma unroll
                for (int nt = 0; nt < C::NT; ++nt) {
                    int half = 0, r;
                    if (SW) {
                        int col;
                        swiglu_row<C>(wn * (C::NT * 8) + nt * 8 + sfb_row, half, col);
                        r = static_cast<int>(((half ? p.row_begin2 : p.row_begin) + n0 + col) & 127);
                    } else {
                        r = static_cast<int>((p.row_begin + n0) & 127) + wn * (C::NT * 8) + nt * 8 + sfb_row;
                    }
                    sfb[nt] = *reinterpret_cast<const uint32_t*>(&s.b_scale[st][ks][half][(r & 31) * 16 + (r >> 5) * 4]);
                }
            }
#pragma unroll
            for (int mt = 0; mt < C::MT; ++mt)
#pragma unroll
                for (int nt = 0; nt < C::NT; ++nt) {
                    if (KIND == kFp4) mma_fp4(acc[mt][nt], af[mt], bf[nt][0], bf[nt][1], sfa[mt], sfb[nt]);
                    else mma_fp8(acc[mt][nt], af[mt], bf[nt][0], bf[nt][1]);
                }
        }
    }
    cp_wait<0>();
    store_tile<C, KIND, SW>(acc, p, p.y + static_cast<long long>(blockIdx.z) * p.m * p.rows, m0, n0, wm, wn, lane);
}

// ---------------------------------------------------------------------------------------------
// TMA prefill GEMM (Wide and Big tiles): thread 0 feeds the stages with TMA (2D tensor maps for the code
// tiles, whose 64-byte swizzle is swz<64>; 1D bulk copies for the scales) and mbarriers; every warp
// only runs ldmatrix and MMA. In the cp.async kernel every thread computed and issued its share of
// each stage's copies, and the copies kept the shared-memory pipe the top stall (ncu, NVFP4
// [34816,5120] m = 1536: MIO throttle 28%). A separate producer warp capped the 288-thread kernel
// at 168 registers and spilled the accumulators.

// Stages per tile: Wide fills three within the shared-memory limit; Big tiles finish a stage in half
// the time, so they keep more in flight.
template <class C>
__host__ __device__ constexpr int tma_stages() {
    return C::BM >= 256 ? 3 : 5;
}

template <class C, bool SW, int kTmaStages = tma_stages<C>()>
struct TmaSmem {
    // Code tiles first, each a multiple of 1024 bytes from a 1024-aligned base (the swizzle atom).
    uint8_t a[kTmaStages][C::BM * C::BK];
    uint8_t b[kTmaStages][C::BN * C::BK];
    uint32_t a_scale[kTmaStages][C::KS][C::BM];           // FP4: k64-major activation scale words
    uint8_t b_scale[kTmaStages][C::KS][SW ? 2 : 1][512];  // FP4: the weight's 128x4 tiles, as stored
    uint64_t full[kTmaStages], empty[kTmaStages];
};

__device__ __forceinline__ void mbar_init(uint64_t* b, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(b)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(b)), "r"(bytes)
                 : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, int parity) {
    asm volatile(
        "{\n.reg .pred p;\nVELOX_MBAR_WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@!p bra VELOX_MBAR_WAIT_%=;\n}\n" ::"r"(smem_u32(b)),
        "r"(parity)
        : "memory");
}
__device__ __forceinline__ void tma_2d(void* dst, const CUtensorMap* map, int x, int y, uint64_t* bar) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(
            smem_u32(dst)),
        "l"(reinterpret_cast<uint64_t>(map)), "r"(x), "r"(y), "r"(smem_u32(bar))
        : "memory");
}
__device__ __forceinline__ void bulk_1d(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(
                     smem_u32(dst)),
                 "l"(src), "r"(bytes), "r"(smem_u32(bar))
                 : "memory");
}

// tm_a: activation codes [m_pad, kb], box BK x BM. tm_b / tm_b2: weight (gate / up) codes
// [parent_rows, kb], box BK x BN (SwiGLU: BK x NT * 4, one per warp column group and half).
template <class C, int KIND, bool SW>
__global__ void __launch_bounds__(C::THREADS, 1)
    gemm_tma_kernel(const __grid_constant__ CUtensorMap tm_a, const __grid_constant__ CUtensorMap tm_b,
                    const __grid_constant__ CUtensorMap tm_b2, GemmArgs p) {
    static_assert(C::BK == 64, "the 64-byte TMA swizzle is swz<64>");
    constexpr int kTmaStages = tma_stages<C>();
    extern __shared__ uint8_t smem_raw[];
    TmaSmem<C, SW>& s =
        *reinterpret_cast<TmaSmem<C, SW>*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
    const int n0 = blockIdx.x * (SW ? C::BN / 2 : C::BN), m0 = blockIdx.y * C::BM;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    constexpr int kConsumers = C::WM * C::WN;  // every warp computes; thread 0 also issues the copies
    // Split-K: blockIdx.z takes stages [z * kt_per_split, ...) and writes its partial sums.
    const int kt_begin = blockIdx.z * p.kt_per_split;
    const int kt_count = min(p.kb / C::BK, kt_begin + p.kt_per_split) - kt_begin;
    float* y = p.y + static_cast<long long>(blockIdx.z) * p.m * p.rows;
    if (threadIdx.x == 0) {
        for (int i = 0; i < kTmaStages; ++i) {
            mbar_init(&s.full[i], 1);
            mbar_init(&s.empty[i], kConsumers);
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();

    // One thread issues a stage: the code tiles by tensor map, the FP4 scales by bulk copy.
    auto issue = [&](int i) {
        const int st = i % kTmaStages;
        constexpr uint32_t kTiles = C::BM * C::BK + C::BN * C::BK;
        constexpr uint32_t kScales = KIND == kFp4 ? C::KS * (C::BM * 4 + (SW ? 2 : 1) * 512) : 0;
        mbar_expect_tx(&s.full[st], kTiles + kScales);
        const int kbyte = (kt_begin + i) * C::BK;
        tma_2d(s.a[st], &tm_a, kbyte, m0, &s.full[st]);
        if (SW) {
            constexpr int half_rows = C::NT * 4;
#pragma unroll
            for (int g = 0; g < C::WN; ++g)
#pragma unroll
                for (int half = 0; half < 2; ++half)
                    tma_2d(s.b[st] + (g * C::NT * 8 + half * half_rows) * C::BK, half ? &tm_b2 : &tm_b, kbyte,
                           static_cast<int>((half ? p.row_begin2 : p.row_begin) + n0 + g * half_rows), &s.full[st]);
        } else {
            tma_2d(s.b[st], &tm_b, kbyte, static_cast<int>(p.row_begin + n0), &s.full[st]);
        }
        if (KIND == kFp4) {
            const int k64 = (kt_begin + i) * C::KS;
#pragma unroll
            for (int ks = 0; ks < C::KS; ++ks) {
                bulk_1d(s.a_scale[st][ks], p.a_scales + (static_cast<long long>(k64 + ks) * p.m_pad + m0) * 4,
                        C::BM * 4, &s.full[st]);
#pragma unroll
                for (int half = 0; half < (SW ? 2 : 1); ++half) {
                    const uint8_t* plane = half ? fp4_scale_plane(p.w2, p.parent_rows2, p.kb)
                                                : fp4_scale_plane(p.w, p.parent_rows, p.kb);
                    const long long tile = ((half ? p.row_begin2 : p.row_begin) + n0) / 128;
                    bulk_1d(s.b_scale[st][ks][half], plane + (tile * (p.k / 64) + k64 + ks) * 512, 512, &s.full[st]);
                }
            }
        }
    };
    if (threadIdx.x == 0)
        for (int i = 0; i < kTmaStages - 1 && i < kt_count; ++i) issue(i);

    const int wm = warp / C::WN, wn = warp % C::WN;
    float acc[C::MT][C::NT][4];
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int j = 0; j < C::NT; ++j)
#pragma unroll
            for (int v = 0; v < 4; ++v) acc[i][j][v] = 0.0f;
    const int a_row = (lane & 7) + (((lane >> 3) & 1) << 3), a_byte = (lane >> 4) << 4;
    const int b_row = (lane & 7) + ((lane >> 4) << 3), b_byte = ((lane >> 3) & 1) << 4;
    const int sfa_row = ((lane & 1) << 3) | (lane >> 2);
    const int sfb_row = lane >> 2;
    constexpr int WROWS = C::MT * 16;

    for (int i = 0; i < kt_count; ++i) {
        const int st = i % kTmaStages;
        // Refill the buffer stage i - 1 used once every warp has released it.
        if (threadIdx.x == 0 && i + kTmaStages - 1 < kt_count) {
            if (i > 0) mbar_wait(&s.empty[(i - 1) % kTmaStages], ((i - 1) / kTmaStages) & 1);
            issue(i + kTmaStages - 1);
        }
        mbar_wait(&s.full[st], (i / kTmaStages) & 1);
#pragma unroll
        for (int ks = 0; ks < C::KS; ++ks) {
            uint32_t af[C::MT][4], bf[C::NT][2], sfa[C::MT], sfb[C::NT];
#pragma unroll
            for (int mt = 0; mt < C::MT; ++mt) {
                const int r = wm * WROWS + mt * 16 + a_row;
                ldmatrix_x4(af[mt], s.a[st] + swz<C::BK>(r, ks * 32 + a_byte));
                if (KIND == kFp4) sfa[mt] = s.a_scale[st][ks][wm * WROWS + mt * 16 + sfa_row];
            }
#pragma unroll
            for (int nt = 0; nt < C::NT; nt += 2) {
                const int r = wn * (C::NT * 8) + nt * 8 + b_row;
                uint32_t t4[4];
                ldmatrix_x4(t4, s.b[st] + swz<C::BK>(r, ks * 32 + b_byte));
                bf[nt][0] = t4[0], bf[nt][1] = t4[1], bf[nt + 1][0] = t4[2], bf[nt + 1][1] = t4[3];
            }
            if (KIND == kFp4) {
#pragma unroll
                for (int nt = 0; nt < C::NT; ++nt) {
                    int half = 0, r;
                    if (SW) {
                        int col;
                        swiglu_row<C>(wn * (C::NT * 8) + nt * 8 + sfb_row, half, col);
                        r = static_cast<int>(((half ? p.row_begin2 : p.row_begin) + n0 + col) & 127);
                    } else {
                        r = static_cast<int>((p.row_begin + n0) & 127) + wn * (C::NT * 8) + nt * 8 + sfb_row;
                    }
                    sfb[nt] =
                        *reinterpret_cast<const uint32_t*>(&s.b_scale[st][ks][half][(r & 31) * 16 + (r >> 5) * 4]);
                }
            }
#pragma unroll
            for (int mt = 0; mt < C::MT; ++mt)
#pragma unroll
                for (int nt = 0; nt < C::NT; ++nt) {
                    if (KIND == kFp4) mma_fp4(acc[mt][nt], af[mt], bf[nt][0], bf[nt][1], sfa[mt], sfb[nt]);
                    else mma_fp8(acc[mt][nt], af[mt], bf[nt][0], bf[nt][1]);
                }
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(&s.empty[st]);
    }
    store_tile<C, KIND, SW>(acc, p, y, m0, n0, wm, wn, lane);
}

__global__ void sum_splits(const float* __restrict__ part, int splits, long long n, float* __restrict__ y) {
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        float s = 0.0f;
        for (int z = 0; z < splits; ++z) s += part[z * n + i];
        y[i] = s;
    }
}

// Grow-only and never freed (VeloxScratch): recorded CUDA graphs may still reference old buffers.
struct Scratch {
    VeloxScratch s;
    uint8_t* get(size_t need) { return static_cast<uint8_t*>(s.get(need)); }
};
Scratch g_scratch;  // quantized activations
Scratch g_part;     // split-K partial sums

constexpr int kSms = 170;
constexpr int kSmallMaxM = 64;  // skinny tiles up to here; prefill tiles (and fused SwiGLU) above

// Wide tiles when their grid fills the GPU about one and a half times over, and for at least two
// tile rows of tokens (m = 128 padded to one Wide tile: [34816,5120] 110 -> 150 us).
constexpr long long kWideBlocks = 256;
constexpr int kWideMinM = 2 * Wide::BM;

bool use_wide(int m, int rows) {
    return m >= kWideMinM &&
           static_cast<long long>((rows + Wide::BN - 1) / Wide::BN) * ((m + Wide::BM - 1) / Wide::BM) >= kWideBlocks;
}

// Token rows the quantizers pad to: the tile height of the configuration launch_gemm picks.
int m_tile(int m, int rows) {
    if (m <= kSmallMaxM) return m <= 16 ? 16 : m <= 32 ? 32 : 64;
    return use_wide(m, rows) ? Wide::BM : Big::BM;
}

template <class C, int KIND, bool SW = false>
int launch_cfg(GemmArgs p) {
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(gemm_kernel<C, KIND, SW>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             sizeof(Smem<C, SW>));
        configured = true;
    }
    const int kt_total = p.kb / C::BK;
    const int bn = SW ? C::BN / 2 : C::BN;  // output columns per block
    const long long blocks = static_cast<long long>((p.rows + bn - 1) / bn) * ((p.m + C::BM - 1) / C::BM);
    // Split K until the grid covers the GPU twice, keeping at least 8 stages per slice.
    int splits = 1;
    // The small tile always may split; the large one only when its grid is badly under-filled,
    // since the partial pass costs more than it saves on prefill-sized grids. SwiGLU never splits
    // (its epilogue needs the whole sums).
    while (!SW && (C::WM == 1 || blocks < 64) && blocks * splits < 2 * kSms && kt_total * C::BK / (splits * 2) >= 512) splits *= 2;
    p.kt_per_split = (kt_total + splits - 1) / splits;
    float* y = p.y;
    if (splits > 1) {
        p.y = reinterpret_cast<float*>(g_part.get(static_cast<size_t>(splits) * p.m * p.rows * sizeof(float)));
        if (!p.y) return static_cast<int>(cudaErrorMemoryAllocation);
    }
    const dim3 grid((p.rows + bn - 1) / bn, (p.m + C::BM - 1) / C::BM, splits);
    gemm_kernel<C, KIND, SW><<<grid, C::THREADS, sizeof(Smem<C, SW>)>>>(p);
    if (splits > 1) {
        const long long n = static_cast<long long>(p.m) * p.rows;
        sum_splits<<<velox_grid_1d(n, 256), 256>>>(p.y, splits, n, y);
    }
    return velox_launch_status();
}

template <int KIND>
int launch_gemm(const GemmArgs& p) {
    if (use_wide(p.m, p.rows)) return launch_cfg<Wide, KIND>(p);
    if (p.m > kSmallMaxM) return launch_cfg<Big, KIND>(p);
    if (p.kb % Small::BK != 0) {
        if (p.m <= 16) return launch_cfg<Small64T<1>, KIND>(p);
        return p.m <= 32 ? launch_cfg<Small64T<2>, KIND>(p) : launch_cfg<Small64T<4>, KIND>(p);
    }
    if (p.m <= 16) return launch_cfg<SmallT<1>, KIND>(p);
    return p.m <= 32 ? launch_cfg<SmallT<2>, KIND>(p) : launch_cfg<SmallT<4>, KIND>(p);
}

// Quantizes x to NVFP4 codes and scales padded to the tile height mt; returns the codes (scales
// follow at codes + m_pad * k / 2), or null when the scratch could not grow.
uint8_t* quantize_a4(const float* x, int m, int mt, long long k, float input_divisor, bool kmajor = false) {
    const int m_pad = (m + mt - 1) / mt * mt;
    const size_t code_bytes = static_cast<size_t>(m_pad) * k / 2;
    uint8_t* buf = g_scratch.get(code_bytes + static_cast<size_t>(m_pad) * k / 16);
    if (!buf) return nullptr;
    const long long groups = static_cast<long long>(m_pad) * (k / 16);
    if (kmajor)
        quant_nvfp4_kernel<true><<<velox_grid_1d(groups, 256), 256>>>(x, m, m_pad, static_cast<int>(k), input_divisor,
                                                                      buf, buf + code_bytes);
    else
        quant_nvfp4_kernel<<<velox_grid_1d(groups, 256), 256>>>(x, m, m_pad, static_cast<int>(k), input_divisor, buf,
                                                            buf + code_bytes);
    return buf;
}

// The weight divisor is the FP32 word after the scale plane; weights are immutable, so it is read
// from the device once per parent and cached (a read per call would synchronize).
int weight_divisor(const uint8_t* w, long long parent_rows, long long k, float& out) {
    static std::mutex mu;
    static std::unordered_map<const uint8_t*, float> cache;
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(w);
    if (it == cache.end()) {
        const long long scale_plane = (parent_rows * k / 2 + 255) / 256 * 256;
        float d;
        if (const cudaError_t e = cudaMemcpy(&d, w + scale_plane + parent_rows * k / 16, 4, cudaMemcpyDeviceToHost))
            return static_cast<int>(e);
        it = cache.emplace(w, d).first;
    }
    out = it->second;
    return 0;
}

// TMA tensor maps. The encoder is the driver's cuTensorMapEncodeTiled, looked up at run time (no
// link-time dependency on libcuda).
using EncodeTiled = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                 const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                 CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

EncodeTiled encoder() {
    static const EncodeTiled fn = []() -> EncodeTiled {
        void* f = nullptr;
        cudaDriverEntryPointQueryResult q;
        if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &f, 12000, cudaEnableDefault, &q) !=
                cudaSuccess ||
            q != cudaDriverEntryPointSuccess)
            return nullptr;
        return reinterpret_cast<EncodeTiled>(f);
    }();
    return fn;
}

// VELOX_NO_TMA=1 keeps the cp.async kernel for every tile (A/B).
bool tma_enabled() {
    static const bool on = getenv("VELOX_NO_TMA") == nullptr && encoder() != nullptr;
    return on;
}

// A [rows, row_bytes] byte matrix read in box_bytes x box_rows boxes with the 64-byte swizzle; rows
// past the end read as zeros.
bool encode_map(CUtensorMap* map, const void* base, long long rows, long long row_bytes, int box_bytes,
                int box_rows) {
    const cuuint64_t dims[2] = {static_cast<cuuint64_t>(row_bytes), static_cast<cuuint64_t>(rows)};
    const cuuint64_t strides[1] = {static_cast<cuuint64_t>(row_bytes)};
    const cuuint32_t box[2] = {static_cast<cuuint32_t>(box_bytes), static_cast<cuuint32_t>(box_rows)};
    const cuuint32_t elem[2] = {1, 1};
    return encoder()(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(base), dims, strides, box, elem,
                     CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                     CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

// Weights are immutable: one map per (parent, box height), encoded once.
bool weight_map(CUtensorMap* out, const uint8_t* w, long long parent_rows, int kb, int box_rows) {
    struct Key {
        const uint8_t* w;
        long long rows;
        int kb, box;
        bool operator==(const Key& o) const { return w == o.w && rows == o.rows && kb == o.kb && box == o.box; }
    };
    struct Hash {
        size_t operator()(const Key& k) const {
            return std::hash<const void*>()(k.w) ^ (static_cast<size_t>(k.rows) << 1) ^
                   (static_cast<size_t>(k.box) << 40);
        }
    };
    static std::mutex mu;
    static std::unordered_map<Key, CUtensorMap, Hash> cache;
    std::lock_guard<std::mutex> lock(mu);
    const Key key{w, parent_rows, kb, box_rows};
    auto it = cache.find(key);
    if (it == cache.end()) {
        CUtensorMap map;
        if (!encode_map(&map, w, parent_rows, kb, 64, box_rows)) return false;
        it = cache.emplace(key, map).first;
    }
    *out = it->second;
    return true;
}

// A prefill tile (Wide or Big) through the TMA kernel. p.a: [m_pad, kb] codes (FP4: k64-major scales
// at p.a_scales); returns VELOX_ERR_UNSUPPORTED when a map cannot be encoded (the caller falls back).
template <class C, int KIND, bool SW>
int launch_tma(GemmArgs p) {
    CUtensorMap ta, tb, tb2;
    const int box_b = SW ? C::NT * 4 : C::BN;
    if (!encode_map(&ta, p.a, p.m_pad, p.kb, C::BK, C::BM) || !weight_map(&tb, p.w, p.parent_rows, p.kb, box_b))
        return VELOX_ERR_UNSUPPORTED;
    if (SW) {
        if (!weight_map(&tb2, p.w2, p.parent_rows2, p.kb, box_b)) return VELOX_ERR_UNSUPPORTED;
    } else {
        tb2 = tb;
    }
    constexpr size_t smem = sizeof(TmaSmem<C, SW>) + 1024;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(gemm_tma_kernel<C, KIND, SW>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        configured = true;
    }
    const int bn = SW ? C::BN / 2 : C::BN;
    const int kt_total = p.kb / C::BK;
    const long long blocks = static_cast<long long>((p.rows + bn - 1) / bn) * ((p.m + C::BM - 1) / C::BM);
    // Split K as launch_cfg does for an under-filled grid (never for SwiGLU).
    int splits = 1;
    while (!SW && blocks < 64 && blocks * splits < 2 * kSms && kt_total * C::BK / (splits * 2) >= 512) splits *= 2;
    p.kt_per_split = (kt_total + splits - 1) / splits;
    float* y = p.y;
    if (splits > 1) {
        p.y = reinterpret_cast<float*>(g_part.get(static_cast<size_t>(splits) * p.m * p.rows * sizeof(float)));
        if (!p.y) return static_cast<int>(cudaErrorMemoryAllocation);
    }
    const dim3 grid((p.rows + bn - 1) / bn, (p.m + C::BM - 1) / C::BM, splits);
    gemm_tma_kernel<C, KIND, SW><<<grid, C::THREADS, smem>>>(ta, tb, tb2, p);
    if (splits > 1) {
        const long long n = static_cast<long long>(p.m) * p.rows;
        sum_splits<<<velox_grid_1d(n, 256), 256>>>(p.y, splits, n, y);
    }
    return velox_launch_status();
}

}  // namespace

// y[m, rows] = nvfp4(x) . W[row_begin.., :]^T for an NVFP4 weight. Requires k % 128 == 0,
// row_begin % 128 == 0 and 16-byte aligned x rows; returns VELOX_ERR_UNSUPPORTED otherwise.
VELOX_API int velox_5090_linear_a4(const float* x, int m, const uint8_t* w, long long parent_rows, long long k,
                                   long long row_begin, int rows, float input_divisor, float* y) {
    if (k % 128 != 0 || row_begin % 128 != 0 || m <= 0 || reinterpret_cast<uintptr_t>(x) % 16 != 0)
        return VELOX_ERR_UNSUPPORTED;
    const int mt = m_tile(m, rows);
    const int m_pad = (m + mt - 1) / mt * mt;
    float wd;
    if (const int e = weight_divisor(w, parent_rows, k, wd)) return e;
    // NVFP4 on 128 x 128 tiles stays on cp.async: the long-K down projection measured 5-9% slower
    // through TMA there (RTX 5090, [5120,17408] m = 256 / 1536: 101 / 354 -> 110 / 377 us).
    const bool tma = tma_enabled() && mt == Wide::BM;
    uint8_t* codes = quantize_a4(x, m, mt, k, input_divisor, tma);
    if (!codes) return static_cast<int>(cudaErrorMemoryAllocation);
    GemmArgs p{codes, codes + static_cast<size_t>(m_pad) * k / 2, nullptr, w, parent_rows, row_begin, rows, m,
               static_cast<int>(k), static_cast<int>(k / 2), 1.0f / (input_divisor * wd), 0, y};
    p.m_pad = m_pad;
    if (const int e = velox_launch_status()) return e;
    if (tma) {
        if (const int e = launch_tma<Wide, kFp4, false>(p); e != VELOX_ERR_UNSUPPORTED) return e;
        quantize_a4(x, m, mt, k, input_divisor);  // row-major scales for the cp.async kernel
    }
    return launch_gemm<kFp4>(p);
}

// y[m, rows] = silu(nvfp4(x) . G^T) * (nvfp4(x) . U^T) for NVFP4 gate / up views of `rows` rows
// (possibly of one parent), prefill widths only (m > 32). Each product is exactly what
// velox_5090_linear_a4 computes for that view, and the epilogue applies the portable swiglu's
// arithmetic, so the result matches linear_a4 + swiglu bit for bit without the [m, 2 * rows]
// intermediate. Requires k % 128 == 0 and row_begin % 128 == 0 for both views.
VELOX_API int velox_5090_linear_swiglu_a4(const float* x, int m, const uint8_t* wg, long long parent_rows_g,
                                          long long row_begin_g, const uint8_t* wu, long long parent_rows_u,
                                          long long row_begin_u, long long k, int rows, float input_divisor,
                                          float* y) {
    if (k % 128 != 0 || row_begin_g % 128 != 0 || row_begin_u % 128 != 0 || m <= kSmallMaxM ||
        reinterpret_cast<uintptr_t>(x) % 16 != 0)
        return VELOX_ERR_UNSUPPORTED;
    // The tile the composed path would pick for the fused [2 * rows] view.
    const bool wide = use_wide(m, 2 * rows);
    const int mt = wide ? Wide::BM : Big::BM;
    const int m_pad = (m + mt - 1) / mt * mt;
    const bool tma = tma_enabled();
    uint8_t* codes = quantize_a4(x, m, mt, k, input_divisor, tma);
    if (!codes) return static_cast<int>(cudaErrorMemoryAllocation);
    float dg, du;
    if (const int e = weight_divisor(wg, parent_rows_g, k, dg)) return e;
    if (const int e = weight_divisor(wu, parent_rows_u, k, du)) return e;
    GemmArgs p{codes, codes + static_cast<size_t>(m_pad) * k / 2, nullptr, wg, parent_rows_g, row_begin_g, rows, m,
               static_cast<int>(k), static_cast<int>(k / 2), 1.0f / (input_divisor * dg), 0, y,
               wu, parent_rows_u, row_begin_u, 1.0f / (input_divisor * du), m_pad};
    if (const int e = velox_launch_status()) return e;
    if (tma) {
        const int e = wide ? launch_tma<Wide, kFp4, true>(p) : launch_tma<Big, kFp4, true>(p);
        if (e != VELOX_ERR_UNSUPPORTED) return e;
        quantize_a4(x, m, mt, k, input_divisor);
    }
    return wide ? launch_cfg<Wide, kFp4, true>(p) : launch_cfg<Big, kFp4, true>(p);
}

// y[m, rows] = e4m3(x) . W^T for an FP8 row-scaled weight. Requires k % 64 == 0.
VELOX_API int velox_5090_linear_a8(const float* x, int m, const uint8_t* w, long long parent_rows, long long k,
                                   long long row_begin, int rows, float* y) {
    if (k % 64 != 0 || m <= 0 || reinterpret_cast<uintptr_t>(x) % 16 != 0) return VELOX_ERR_UNSUPPORTED;
    const int mt = m_tile(m, rows);
    const int m_pad = (m + mt - 1) / mt * mt;
    const size_t code_bytes = static_cast<size_t>(m_pad) * k;
    uint8_t* buf = g_scratch.get(code_bytes + static_cast<size_t>(m_pad) * 4);
    if (!buf) return static_cast<int>(cudaErrorMemoryAllocation);
    float* row_scale = reinterpret_cast<float*>(buf + code_bytes);
    quant_fp8_kernel<<<m_pad, 256>>>(x, m, static_cast<int>(k), buf, row_scale);
    if (const int e = velox_launch_status()) return e;
    GemmArgs p{buf, nullptr, row_scale, w, parent_rows, row_begin, rows, m, static_cast<int>(k),
               static_cast<int>(k), 1.0f, 0, y};
    p.m_pad = m_pad;
    if (tma_enabled() && m > kSmallMaxM) {
        const int e = mt == Wide::BM ? launch_tma<Wide, kFp8, false>(p) : launch_tma<Big, kFp8, false>(p);
        if (e != VELOX_ERR_UNSUPPORTED) return e;
    }
    return launch_gemm<kFp8>(p);
}

// y[m, rows] = silu(e4m3(x) . G^T) * (e4m3(x) . U^T) for FP8 row-scaled gate / up views of `rows`
// rows (possibly of one parent), prefill widths only (m > 32); equals linear_a8 + swiglu (see
// velox_5090_linear_swiglu_a4). Requires k % 64 == 0.
VELOX_API int velox_5090_linear_swiglu_a8(const float* x, int m, const uint8_t* wg, long long parent_rows_g,
                                          long long row_begin_g, const uint8_t* wu, long long parent_rows_u,
                                          long long row_begin_u, long long k, int rows, float* y) {
    if (k % 64 != 0 || m <= kSmallMaxM || reinterpret_cast<uintptr_t>(x) % 16 != 0) return VELOX_ERR_UNSUPPORTED;
    const bool wide = use_wide(m, 2 * rows);
    const int mt = wide ? Wide::BM : Big::BM;
    const int m_pad = (m + mt - 1) / mt * mt;
    const size_t code_bytes = static_cast<size_t>(m_pad) * k;
    uint8_t* buf = g_scratch.get(code_bytes + static_cast<size_t>(m_pad) * 4);
    if (!buf) return static_cast<int>(cudaErrorMemoryAllocation);
    float* row_scale = reinterpret_cast<float*>(buf + code_bytes);
    quant_fp8_kernel<<<m_pad, 256>>>(x, m, static_cast<int>(k), buf, row_scale);
    if (const int e = velox_launch_status()) return e;
    GemmArgs p{buf, nullptr, row_scale, wg, parent_rows_g, row_begin_g, rows, m, static_cast<int>(k),
               static_cast<int>(k), 1.0f, 0, y, wu, parent_rows_u, row_begin_u, 1.0f, m_pad};
    if (tma_enabled()) {
        const int e = wide ? launch_tma<Wide, kFp8, true>(p) : launch_tma<Big, kFp8, true>(p);
        if (e != VELOX_ERR_UNSUPPORTED) return e;
    }
    return wide ? launch_cfg<Wide, kFp8, true>(p) : launch_cfg<Big, kFp8, true>(p);
}
