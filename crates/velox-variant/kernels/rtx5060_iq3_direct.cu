// RTX 5060 Ti IQ3_XXS linear for 1..16 tokens: register-direct tensor-core kernel.
//
// Adapted from ninfer's iq3xxs_mma_direct_kernel (src/ops/common/iq3xxs_matmul.cuh, Apache-2.0).
// Weights are the A operand of mma.m16n8k16 (bf16, f32 accumulate) and tokens the B operand. K is
// permuted inside every 32-value sub-block (the dot product is order independent): for half h, MMA
// k-index 2q + {0,1} is value 8q + 4h + {0,1} and k-index 2q + 8 + {0,1} is value 8q + 4h + {2,3},
// where q is the lane's position in its quad. A lane's eight values per row are then exactly grid
// entries 2q and 2q+1, which iq3_xxs_planar_v1 stores contiguously per quad lane, and its eight
// activations per token are one 16-byte load (bf16, natural order, converted with RNE up front).
//
// Each lane loads its 24 code bytes of a row-block straight into registers, double-buffered across
// the warp's strided K blocks: W warps split K, MT 16-row tiles per block, NT 8-token tiles. The
// only block barrier is the final cross-warp K reduction. Signed grid magnitudes are exact bf16
// integers; each sub-block's scale d (0.25 + 0.5 ls) is applied to its f32 partial.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include "velox_common.cuh"

namespace {

__constant__ uint32_t kGrid[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};

#ifndef IQ3_DIRECT_W_SHORT
#define IQ3_DIRECT_W_SHORT 2
#endif
#ifndef IQ3_DIRECT_W_LONG
#define IQ3_DIRECT_W_LONG 8  // measured on this card: down 111 us with 8 warps vs 118 with 4
#endif
#ifndef IQ3_DIRECT_LONG_BLOCKS
#define IQ3_DIRECT_LONG_BLOCKS 32
#endif

struct Tables {
    uint2 grid[256];      // bf16x2 magnitudes (0,1) and (2,3) of every grid entry
    uint2 sign_mask[16];  // bf16 sign masks of a 4-bit nibble: bits (0,1) -> .x, (2,3) -> .y

    __device__ __forceinline__ void load() {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) {
            const uint32_t g = kGrid[i];
            uint32_t b[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = __float_as_uint(static_cast<float>((g >> (8 * j)) & 0xffu)) >> 16;
            grid[i] = make_uint2(b[0] | (b[1] << 16), b[2] | (b[3] << 16));
        }
        if (threadIdx.x < 16) {
            const uint32_t t = threadIdx.x;
            auto mask = [](uint32_t bits) { return ((bits & 1u) << 15) | ((bits & 2u) << 30); };
            sign_mask[t] = make_uint2(mask(t & 3u), mask(t >> 2));
        }
    }
};

struct Raw {
    uint4 index;  // index bytes of entries (2q, 2q+1) for sub-blocks 0..7
    uint2 signs;  // bits [7s, 7s+7): sign index q of sub-block s; top byte: ls of 2q, 2q+1
    float d;
};

__device__ __forceinline__ void mma_bf16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t bf16x2(float lo, float hi) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
}

// KB: IQ3 blocks per row as a compile-time constant (the model's K = 5120 and 17408), which lets the
// compiler unroll the block loop and issue loads further ahead; 0 reads it from k.
// XT: the activations' type. f32 (converted to bf16 in registers) saves the conversion launch
// and costs the same for 1..2 tokens; bf16 (pre-converted) is faster for more tokens.
// PAIR: a fused SwiGLU projection. Each 16-row tile is 8 gate rows (from w) and the matching 8 up
// rows (from up.w, possibly another parent); out[t][r] = silu(gate . x_t) * (up . x_t).
struct Up {
    const uint8_t* w;
    long long parent_rows, row_begin;
};

template <int W, int MT, int NT, int KB, typename XT, bool PAIR>
__global__ void __launch_bounds__(W * 32)
iq3_direct_kernel(const XT* __restrict__ x, const uint8_t* __restrict__ w, long long parent_rows, int k_rt,
                  long long row_begin, int rows, int m, float* __restrict__ y, Up up) {
    const int k = KB ? KB * 256 : k_rt;
    __shared__ Tables tables;
    __shared__ float4 partial[W > 1 ? W * NT * 32 : 1];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int gid = lane >> 2, q = lane & 3;
    const int blocks = KB ? KB : k / 256;
    constexpr int kTileRows = PAIR ? 8 : 16;  // output rows per 16-row MMA tile
    const int r0 = blockIdx.x * kTileRows * MT;
    auto dplane_of = [&](const uint8_t* base, long long n) {
        return reinterpret_cast<const __half*>(base + (n * blocks * 96 + 255) / 256 * 256);
    };

    const uint8_t* row_codes[MT][2];
    const __half* row_d[MT][2];
#pragma unroll
    for (int mi = 0; mi < MT; ++mi)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int r = PAIR ? r0 + mi * 8 + gid : r0 + mi * 16 + gid + 8 * e;
            const int rr = r < rows ? r : 0;  // rows past the view: output dropped
            const bool second = PAIR && e == 1;
            const uint8_t* base = second ? up.w : w;
            const long long row = (second ? up.row_begin : row_begin) + rr;
            row_codes[mi][e] = base + row * blocks * 96;
            row_d[mi][e] = dplane_of(base, second ? up.parent_rows : parent_rows) + row * blocks;
        }
    auto load = [&](int b, Raw (&raw)[MT][2]) {
#pragma unroll
        for (int mi = 0; mi < MT; ++mi)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const uint8_t* base = row_codes[mi][e] + b * 96;
                raw[mi][e].index = __ldg(reinterpret_cast<const uint4*>(base + q * 16));
                raw[mi][e].signs = __ldg(reinterpret_cast<const uint2*>(base + 64 + q * 8));
                raw[mi][e].d = __half2float(__ldg(row_d[mi][e] + b));
            }
    };
    // This lane's token rows (one per token tile) and whether they are real.
    const XT* xrow[NT];
    bool valid[NT];
#pragma unroll
    for (int ni = 0; ni < NT; ++ni) {
        const int t = ni * 8 + gid;
        valid[ni] = t < m;
        xrow[ni] = x + static_cast<long long>(valid[ni] ? t : 0) * k + q * 8;
    }

    tables.load();
    Raw cur[MT][2], next[MT][2];
    if (warp < blocks) load(warp, cur);
    __syncthreads();

    float acc[MT][NT][4];
#pragma unroll
    for (int mi = 0; mi < MT; ++mi)
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.0f;

#pragma unroll
    for (int b = warp; b < blocks; b += W) {
        const bool more = b + W < blocks;
        if (more) load(b + W, next);
        // The quad's eight sub-block scales: lane j's top sign byte holds ls of 2j (low), 2j+1.
        uint32_t scales[MT][2];
        uint64_t signs[MT][2];
#pragma unroll
        for (int mi = 0; mi < MT; ++mi)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const uint32_t own = cur[mi][e].signs.y >> 24;
                uint32_t s = 0;
#pragma unroll
                for (int j = 0; j < 4; ++j) s |= __shfl_sync(0xffffffffu, own, (lane & ~3) | j) << (8 * j);
                scales[mi][e] = s;
                signs[mi][e] = (static_cast<uint64_t>(cur[mi][e].signs.y) << 32) | cur[mi][e].signs.x;
            }
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            // Activations: values 8q..8q+7 of sub-block s for each token tile, as bf16 pairs.
            uint32_t xb[NT][4];
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                if (valid[ni]) {
                    if constexpr (sizeof(XT) == 4) {
                        const float4* xp = reinterpret_cast<const float4*>(xrow[ni] + b * 256 + s * 32);
                        const float4 a = __ldg(xp), c = __ldg(xp + 1);
                        xb[ni][0] = bf16x2(a.x, a.y), xb[ni][1] = bf16x2(a.z, a.w);
                        xb[ni][2] = bf16x2(c.x, c.y), xb[ni][3] = bf16x2(c.z, c.w);
                    } else {
                        const uint4 v = __ldg(reinterpret_cast<const uint4*>(xrow[ni] + b * 256 + s * 32));
                        xb[ni][0] = v.x, xb[ni][1] = v.y, xb[ni][2] = v.z, xb[ni][3] = v.w;
                    }
                } else {
                    xb[ni][0] = xb[ni][1] = xb[ni][2] = xb[ni][3] = 0u;
                }
            }
            // A fragments: f[mi][e][h] = entry 2q + h of row e, signed.
            uint2 f[MT][2][2];
#pragma unroll
            for (int mi = 0; mi < MT; ++mi)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const uint32_t word = s < 2 ? cur[mi][e].index.x : s < 4 ? cur[mi][e].index.y
                                        : s < 6 ? cur[mi][e].index.z : cur[mi][e].index.w;
                    const uint32_t pair = word >> (16 * (s & 1));
                    const uint32_t s7 = static_cast<uint32_t>(signs[mi][e] >> (7 * s)) & 127u;
                    const uint32_t s8 = s7 | ((__popc(s7) & 1u) << 7);
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const uint2 g = tables.grid[(pair >> (8 * h)) & 0xffu];
                        // bf16 sign bits of the nibble's four values, from ALU ops: the kernel is
                        // bound by the shared-memory queue (MIO), not by issue.
                        const uint32_t n = s8 >> (4 * h);
                        const uint2 mk = make_uint2(((n & 1u) << 15) | ((n & 2u) << 30), ((n & 4u) << 13) | ((n & 8u) << 28));
                        f[mi][e][h] = make_uint2(g.x ^ mk.x, g.y ^ mk.y);
                    }
                }
            float part[MT][NT][4];
#pragma unroll
            for (int mi = 0; mi < MT; ++mi)
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) part[mi][ni][0] = part[mi][ni][1] = part[mi][ni][2] = part[mi][ni][3] = 0.0f;
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int mi = 0; mi < MT; ++mi)
#pragma unroll
                    for (int ni = 0; ni < NT; ++ni)
                        mma_bf16(part[mi][ni], f[mi][0][h].x, f[mi][1][h].x, f[mi][0][h].y, f[mi][1][h].y,
                                 xb[ni][2 * h], xb[ni][2 * h + 1]);
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const float s0 = cur[mi][0].d * (0.25f + 0.5f * static_cast<float>((scales[mi][0] >> (4 * s)) & 15u));
                const float s8v = cur[mi][1].d * (0.25f + 0.5f * static_cast<float>((scales[mi][1] >> (4 * s)) & 15u));
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    acc[mi][ni][0] = fmaf(part[mi][ni][0], s0, acc[mi][ni][0]);
                    acc[mi][ni][1] = fmaf(part[mi][ni][1], s0, acc[mi][ni][1]);
                    acc[mi][ni][2] = fmaf(part[mi][ni][2], s8v, acc[mi][ni][2]);
                    acc[mi][ni][3] = fmaf(part[mi][ni][3], s8v, acc[mi][ni][3]);
                }
            }
        }
        if (more) {
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) cur[mi][0] = next[mi][0], cur[mi][1] = next[mi][1];
        }
    }

    // Cross-warp K reduction, then c0, c1: row gid, tokens 2q, 2q+1; c2, c3: row gid + 8.
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
        float c[NT][4];
        if (W > 1) {
            __syncthreads();
#pragma unroll
            for (int ni = 0; ni < NT; ++ni)
                partial[(warp * NT + ni) * 32 + lane] = make_float4(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3]);
            __syncthreads();
            if (warp != 0) continue;
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                c[ni][0] = c[ni][1] = c[ni][2] = c[ni][3] = 0.0f;
#pragma unroll
                for (int wi = 0; wi < W; ++wi) {
                    const float4 p = partial[(wi * NT + ni) * 32 + lane];
                    c[ni][0] += p.x, c[ni][1] += p.y, c[ni][2] += p.z, c[ni][3] += p.w;
                }
            }
        } else {
#pragma unroll
            for (int ni = 0; ni < NT; ++ni)
#pragma unroll
                for (int j = 0; j < 4; ++j) c[ni][j] = acc[mi][ni][j];
        }
        if (PAIR) {
            // c0, c1: gate row, tokens 2q, 2q+1; c2, c3: the matching up row.
            const int r = r0 + mi * 8 + gid;
            if (r < rows) {
#pragma unroll
                for (int ni = 0; ni < NT; ++ni)
#pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        const int t = ni * 8 + 2 * q + j;
                        const float g = c[ni][j];
                        if (t < m) y[static_cast<long long>(t) * rows + r] = g / (1.0f + expf(-g)) * c[ni][2 + j];
                    }
            }
            continue;
        }
#pragma unroll
        for (int ni = 0; ni < NT; ++ni)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int r = r0 + mi * 16 + gid + 8 * e;
                if (r >= rows) continue;
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int t = ni * 8 + 2 * q + j;
                    if (t < m) y[static_cast<long long>(t) * rows + r] = c[ni][2 * e + j];
                }
            }
    }
}

// x [m, k] f32 -> bf16 (RNE), natural order.
__global__ void to_bf16(const float* __restrict__ x, long long n, __nv_bfloat16* __restrict__ y) {
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<long long>(gridDim.x) * blockDim.x)
        y[i] = __float2bfloat16_rn(x[i]);
}

VeloxScratch g_x;

template <int W, int MT, int NT, int KB, typename XT, bool PAIR>
int launch_kb(const XT* x, const uint8_t* w, long long n, int k, long long rb, int rows, int m, float* y, Up up) {
    constexpr int kTileRows = (PAIR ? 8 : 16) * MT;
    const unsigned grid = static_cast<unsigned>((rows + kTileRows - 1) / kTileRows);
    iq3_direct_kernel<W, MT, NT, KB, XT, PAIR><<<grid, W * 32>>>(x, w, n, k, rb, rows, m, y, up);
    return velox_launch_status();
}

template <int W, int MT, int NT, bool PAIR, typename XT>
int launch(const XT* x, const uint8_t* w, long long n, int k, long long rb, int rows, int m, float* y, Up up) {
    if (k == 5120) return launch_kb<W, MT, NT, 20, XT, PAIR>(x, w, n, k, rb, rows, m, y, up);
    if (k == 17408) return launch_kb<W, MT, NT, 68, XT, PAIR>(x, w, n, k, rb, rows, m, y, up);
    return launch_kb<W, MT, NT, 0, XT, PAIR>(x, w, n, k, rb, rows, m, y, up);
}

template <bool PAIR>
int run(const float* x, int m, const uint8_t* w, long long parent_rows, long long k, long long row_begin, int rows,
        float* y, Up up) {
    if (m < 1 || m > 16 || rows <= 0 || k % 256 != 0 || k > (1ll << 30) || reinterpret_cast<uintptr_t>(x) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(w) % 16 != 0 || reinterpret_cast<uintptr_t>(up.w) % 16 != 0)
        return VELOX_ERR_UNSUPPORTED;
    const int ki = static_cast<int>(k);
    const bool long_rows = k / 256 > IQ3_DIRECT_LONG_BLOCKS;
    if (m <= 2) {
        return long_rows ? launch<IQ3_DIRECT_W_LONG, 1, 1, PAIR>(x, w, parent_rows, ki, row_begin, rows, m, y, up)
                         : launch<IQ3_DIRECT_W_SHORT, 1, 1, PAIR>(x, w, parent_rows, ki, row_begin, rows, m, y, up);
    }
    auto* xb = static_cast<__nv_bfloat16*>(g_x.get(static_cast<size_t>(m) * k * 2));
    if (!xb) return static_cast<int>(cudaErrorMemoryAllocation);
    to_bf16<<<velox_grid_1d(static_cast<long long>(m) * k, 256), 256>>>(x, static_cast<long long>(m) * k, xb);
    const __nv_bfloat16* x16 = xb;
    if (m <= 8) {
        return long_rows ? launch<IQ3_DIRECT_W_LONG, 1, 1, PAIR>(x16, w, parent_rows, ki, row_begin, rows, m, y, up)
                         : launch<IQ3_DIRECT_W_SHORT, 1, 1, PAIR>(x16, w, parent_rows, ki, row_begin, rows, m, y, up);
    }
    return long_rows ? launch<IQ3_DIRECT_W_LONG, 1, 2, PAIR>(x16, w, parent_rows, ki, row_begin, rows, m, y, up)
                     : launch<IQ3_DIRECT_W_SHORT, 2, 2, PAIR>(x16, w, parent_rows, ki, row_begin, rows, m, y, up);
}

}  // namespace

// y[m, rows] = bf16(x) . W[row_begin.., :]^T for an iq3_xxs_planar_v1 parent [parent_rows, k],
// 1 <= m <= 16; x must be 16-byte aligned. Warps split K: 2 for few blocks per row (gate/up,
// K = 5120), 8 for long rows (down, K = 17408). Up to 2 tokens read the f32 activations directly;
// more are converted to bf16 first. Returns VELOX_ERR_UNSUPPORTED for anything else.
VELOX_API int velox_5060_linear_iq3_direct(const float* x, int m, const uint8_t* w, long long parent_rows,
                                           long long k, long long row_begin, int rows, float* y) {
    return run<false>(x, m, w, parent_rows, k, row_begin, rows, y, Up{w, parent_rows, row_begin});
}

// out[m, rows] = silu(bf16(x) . G^T) * (bf16(x) . U^T) for `rows` gate rows of the parent at `w`
// from `gate_begin` and as many up rows of the parent at `up_w` from `up_begin` (same k).
VELOX_API int velox_5060_linear_swiglu_iq3_direct(const float* x, int m, const uint8_t* w, long long parent_rows,
                                                  long long gate_begin, const uint8_t* up_w,
                                                  long long up_parent_rows, long long up_begin, long long k, int rows,
                                                  float* out) {
    return run<true>(x, m, w, parent_rows, k, gate_begin, rows, out, Up{up_w, up_parent_rows, up_begin});
}
