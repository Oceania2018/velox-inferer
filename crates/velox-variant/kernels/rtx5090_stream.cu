// RTX 5090 weight-streaming linear for decode and speculative verify (1..16 tokens): NVFP4 and FP8
// E4M3 row-scaled weights, bf16 activations (ActQuant::Bf16), one launch.
//
// These widths are bound by reading the weights once, so the kernel keeps as many 16-byte weight
// loads in flight as it can and does nothing else between them: no shared-memory staging, no
// activation pre-pass, no separate split-K reduction. A warp owns a 16-row tile and a K range;
// each lane streams 16 contiguous bytes of rows g and g + 8 (g = lane / 4) per step, decodes them
// exactly to bf16 (E4M3 widens exactly; E2M1 times its E4M3 block scale needs at most 5
// significant bits) and feeds mma.m16n8k16 with them as the A operand.
//
// The dot product does not care about the order of K, so the MMA's logical K is a permutation of
// the physical one: lane (g, c) supplies logical K {2c, 2c+1, 2c+8, 2c+9} of MMA j from physical
// K base + 16c + 4j + {0, 1, 2, 3} (FP8; NVFP4 uses 32c), i.e. from the 16 bytes it loaded, and
// reads the activations (B operand, token g) at the same physical K as one float4. Matrices with
// few row tiles split K over the warps of a block, which sum their partials in shared memory in a
// fixed order, so results are deterministic.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#include "velox_common.cuh"

namespace {

constexpr int kFp8 = 0, kNvfp4 = 1;
constexpr int kWarps = 8;
constexpr int kUnroll = 4;  // steps whose weight loads are issued together

struct StreamArgs {
    const float* x;
    int m;
    const uint8_t* w;
    long long parent_rows, k, row_begin;
    int rows;
    float* y;
};

__device__ __forceinline__ long long align256(long long v) { return (v + 255) / 256 * 256; }

// Same tiling as velox_core::format's block_scale_k16_m128x4_v1 (portable nvfp4_scale_offset).
__device__ __forceinline__ long long nvfp4_scale_offset(long long n, long long g, long long k) {
    const long long k_tiles = k / 64;
    const long long row_tile = n / 128, row_inner = n % 128, scale_tile = g / 4, scale_lane = g % 4;
    return (row_tile * k_tiles + scale_tile) * 512 + (row_inner % 32) * 16 + (row_inner / 32) * 4 + scale_lane;
}

__device__ __forceinline__ uint4 ld_stream(const uint8_t* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ uint32_t pack_bf16(float a, float b) {
    const __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
}

__device__ __forceinline__ float e4m3_f32(uint32_t b) {
    const __half_raw h = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b & 0xffu), __NV_E4M3);
    return __half2float(__half(h));
}

// The two E4M3 codes in the low 16 bits of t as a bf16 pair.
__device__ __forceinline__ uint32_t fp8x2_bf16(uint32_t t) {
    const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(t & 0xffffu), __NV_E4M3);
    const float2 f = __half22float2(__half2(h));
    return pack_bf16(f.x, f.y);
}

// The two E2M1 codes in the low byte of t times s as a bf16 pair. Each nibble's magnitude bits go
// to f16 bits [11:9] and its sign to bit 15, giving the code value times 2^-14; s carries 2^14.
__device__ __forceinline__ uint32_t fp4x2_bf16(uint32_t t, float s) {
    const uint32_t h = ((t & 0x7u) << 9) | ((t & 0x8u) << 12) | ((t & 0x70u) << 21) | ((t & 0x80u) << 24);
    const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h));
    return pack_bf16(f.x * s, f.y * s);
}

__device__ __forceinline__ void mma_bf16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t word(const uint4& v, int i) {
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// NT token columns (8 or 16), KS warps per row tile splitting K.
template <int FMT, int NT, int KS>
__global__ void __launch_bounds__(kWarps * VELOX_WARP) stream_kernel(StreamArgs p) {
    constexpr int kStepK = FMT == kFp8 ? 64 : 128;  // K per step (a quad's 64 bytes of a row)
    constexpr int kLaneK = kStepK / 4;              // K per lane per step
    constexpr int kMmas = kLaneK / 4;               // MMAs per step
    constexpr int kTiles = kWarps / KS;
    __shared__ float red[KS > 1 ? kWarps : 1][16][NT];

    const int warp = threadIdx.x / VELOX_WARP, lane = threadIdx.x % VELOX_WARP;
    const int g = lane / 4, c = lane % 4;
    const int r0 = (blockIdx.x * kTiles + warp / KS) * 16;
    const int ks = warp % KS;
    const long long k = p.k;
    const int steps = static_cast<int>(k / kStepK);
    const int s_begin = steps * ks / KS, s_end = r0 < p.rows ? steps * (ks + 1) / KS : s_begin;

    const int ra = min(r0 + g, p.rows - 1), rb = min(r0 + g + 8, p.rows - 1);
    const long long pa = p.row_begin + ra, pb = p.row_begin + rb;
    const long long row_bytes = FMT == kFp8 ? k : k / 2;
    const uint8_t* wa = p.w + pa * row_bytes + c * 16;
    const uint8_t* wb = p.w + pb * row_bytes + c * 16;
    const long long scale_plane = align256(p.parent_rows * k / 2);
    const float* xa = p.x + static_cast<size_t>(g) * k + c * kLaneK;
    const float* xb = p.x + static_cast<size_t>(g + 8) * k + c * kLaneK;
    const bool ta = g < p.m, tb = NT > 8 && g + 8 < p.m;

    float acc[NT / 8][4];
#pragma unroll
    for (int t = 0; t < NT / 8; ++t)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[t][i] = 0.0f;

    for (int s0 = s_begin; s0 < s_end; s0 += kUnroll) {
        uint4 va[kUnroll], vb[kUnroll];
        float sa[kUnroll][2], sb[kUnroll][2];
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            if (s0 + u < s_end) {
                va[u] = ld_stream(wa + static_cast<long long>(s0 + u) * 64);
                vb[u] = ld_stream(wb + static_cast<long long>(s0 + u) * 64);
            }
        }
        if (FMT == kNvfp4) {
#pragma unroll
            for (int u = 0; u < kUnroll; ++u) {
                if (s0 + u < s_end) {
                    // This lane's 32 values are 16-value groups gi and gi + 1 (gi even): adjacent bytes.
                    const long long gi = static_cast<long long>(s0 + u) * 8 + 2 * c;
                    const uint32_t qa = __ldg(reinterpret_cast<const unsigned short*>(
                        p.w + scale_plane + nvfp4_scale_offset(pa, gi, k)));
                    const uint32_t qb = __ldg(reinterpret_cast<const unsigned short*>(
                        p.w + scale_plane + nvfp4_scale_offset(pb, gi, k)));
                    sa[u][0] = e4m3_f32(qa) * 16384.0f, sa[u][1] = e4m3_f32(qa >> 8) * 16384.0f;
                    sb[u][0] = e4m3_f32(qb) * 16384.0f, sb[u][1] = e4m3_f32(qb >> 8) * 16384.0f;
                }
            }
        }
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            if (s0 + u >= s_end) break;
            const long long kb = static_cast<long long>(s0 + u) * kStepK;
#pragma unroll
            for (int j = 0; j < kMmas; ++j) {
                uint32_t a0, a1, a2, a3;
                if (FMT == kFp8) {
                    const uint32_t ta4 = word(va[u], j), tb4 = word(vb[u], j);
                    a0 = fp8x2_bf16(ta4), a2 = fp8x2_bf16(ta4 >> 16);
                    a1 = fp8x2_bf16(tb4), a3 = fp8x2_bf16(tb4 >> 16);
                } else {
                    const uint32_t ta2 = word(va[u], j / 2) >> (16 * (j & 1));
                    const uint32_t tb2 = word(vb[u], j / 2) >> (16 * (j & 1));
                    const float sca = sa[u][j / 4], scb = sb[u][j / 4];
                    a0 = fp4x2_bf16(ta2, sca), a2 = fp4x2_bf16(ta2 >> 8, sca);
                    a1 = fp4x2_bf16(tb2, scb), a3 = fp4x2_bf16(tb2 >> 8, scb);
                }
                {
                    const float4 v = ta ? __ldg(reinterpret_cast<const float4*>(xa + kb + 4 * j))
                                        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    mma_bf16(acc[0], a0, a1, a2, a3, pack_bf16(v.x, v.y), pack_bf16(v.z, v.w));
                }
                if (NT > 8) {
                    const float4 v = tb ? __ldg(reinterpret_cast<const float4*>(xb + kb + 4 * j))
                                        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    mma_bf16(acc[NT / 8 - 1], a0, a1, a2, a3, pack_bf16(v.x, v.y), pack_bf16(v.z, v.w));
                }
            }
        }
    }

    // Output scale of a parent row: FP8's bf16 row scale, NVFP4's 1 / global divisor.
    auto post = [&](long long prow) {
        if (FMT == kFp8) {
            const unsigned short s =
                *reinterpret_cast<const unsigned short*>(p.w + align256(p.parent_rows * k) + 2 * prow);
            return __uint_as_float(static_cast<uint32_t>(s) << 16);
        }
        return 1.0f / *reinterpret_cast<const float*>(p.w + scale_plane + p.parent_rows * k / 16);
    };
    // C fragment: acc[t][0..1] = rows g, tokens 8t + 2c, +1; acc[t][2..3] = row g + 8.
    if (KS == 1) {
#pragma unroll
        for (int t = 0; t < NT / 8; ++t)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int row = r0 + g + (i >= 2 ? 8 : 0), tok = 8 * t + 2 * c + (i & 1);
                if (row < p.rows && tok < p.m)
                    p.y[static_cast<size_t>(tok) * p.rows + row] = acc[t][i] * post(p.row_begin + row);
            }
        return;
    }
#pragma unroll
    for (int t = 0; t < NT / 8; ++t)
#pragma unroll
        for (int i = 0; i < 4; ++i) red[warp][g + (i >= 2 ? 8 : 0)][8 * t + 2 * c + (i & 1)] = acc[t][i];
    __syncthreads();
    for (int e = threadIdx.x; e < kTiles * 16 * NT; e += kWarps * VELOX_WARP) {
        const int tile = e / (16 * NT), r = (e / NT) % 16, tok = e % NT;
        const int row = (blockIdx.x * kTiles + tile) * 16 + r;
        if (row >= p.rows || tok >= p.m) continue;
        float s = 0.0f;
#pragma unroll
        for (int j = 0; j < KS; ++j) s += red[tile * KS + j][r][tok];
        p.y[static_cast<size_t>(tok) * p.rows + row] = s * post(p.row_begin + row);
    }
}

template <int FMT, int NT>
int launch_nt(const StreamArgs& p, int ks, int tiles) {
    const int blocks = (tiles + kWarps / ks - 1) / (kWarps / ks);
    const int threads = kWarps * VELOX_WARP;
    switch (ks) {
        case 1: stream_kernel<FMT, NT, 1><<<blocks, threads>>>(p); break;
        case 2: stream_kernel<FMT, NT, 2><<<blocks, threads>>>(p); break;
        case 4: stream_kernel<FMT, NT, 4><<<blocks, threads>>>(p); break;
        default: stream_kernel<FMT, NT, 8><<<blocks, threads>>>(p); break;
    }
    return static_cast<int>(cudaGetLastError());
}

}  // namespace

// y[m, rows] = bf16(x[m, k]) . W[row_begin + r, :]^T for 1 <= m <= 16. fmt 0: FP8 E4M3 row-scaled
// (K % 64 == 0), 1: NVFP4 (K % 128 == 0). Returns VELOX_ERR_UNSUPPORTED for other shapes.
VELOX_API int velox_5090_linear_stream(const float* x, int m, const uint8_t* w, long long parent_rows, long long k,
                                       long long row_begin, int rows, int fmt, float* y) {
    const int step = fmt == kFp8 ? 64 : 128;
    if (m < 1 || m > 16 || rows < 1 || k % step != 0 || reinterpret_cast<uintptr_t>(w) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(x) % 16 != 0)
        return VELOX_ERR_UNSUPPORTED;
    const StreamArgs p{x, m, w, parent_rows, k, row_begin, rows, y};
    const int tiles = (rows + 15) / 16;
    // Enough warps in flight to cover the memory latency (about 12 per SM on 170 SMs), while each
    // keeps at least two unrolled batches of K.
    const int steps = static_cast<int>(k / step);
    int ks = 1;
    while (ks < 8 && tiles * ks < 2048 && steps / (ks * 2) >= 2 * kUnroll) ks *= 2;
    if (fmt == kFp8) return m <= 8 ? launch_nt<kFp8, 8>(p, ks, tiles) : launch_nt<kFp8, 16>(p, ks, tiles);
    return m <= 8 ? launch_nt<kNvfp4, 8>(p, ks, tiles) : launch_nt<kNvfp4, 16>(p, ks, tiles);
}
