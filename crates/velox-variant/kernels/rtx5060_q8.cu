// RTX 5060 Ti prefill linear on int8 tensor cores (velox_core::ActQuant::Int8G256): IQ3_XXS
// (plain and fused SwiGLU) and row_split_k128_v1 q4_g64 / q5_g64 weights.
//
// Adapted from ninfer's int8 prefill path (src/ops/common/q8_act.cuh, iq3xxs_matmul.cuh,
// rowsplit_q8_mma.cuh; Apache-2.0), which ninfer runs on this card from 33 (IQ3) / 64 (Q4/Q5)
// tokens. Activations are rounded to bf16, then quantized per token and 256-value group
// (scale = amax / 127, codes rint(x * 127 / amax)); weights are the A operand of
// mma.m16n8k32.s8 and exact int8 (IQ3 signed grid magnitudes <= 62, Q4/Q5 codes), tokens the B
// operand. Integer partials are scaled by the weight scale, then by the token's group scale.
//
// IQ3: K is permuted inside every 32-value sub-block so that a quad lane's k-positions [4q, 4q+4)
// and [16+4q, 16+4q+4) are grid entries 2q and 2q+1, which iq3_xxs_planar_v1 stores per quad lane,
// and its activations are values 8q..8q+7 in natural order. The sub-block scale d (0.25 + 0.5 ls)
// is (d / 4)(1 + 2 ls): partials of a block stay int32 and are converted once.
//
// Q4/Q5: inside every 64-value group, quad lane q owns code bytes 8q..8q+7 (values 16q..16q+15);
// the activations are stored in the matching lane order (quantize layout 1), so a lane's B operands
// for a group are one 16-byte load. Both k32 steps of a group share its scale and accumulate in
// the MMA, starting at the bits of 1.5 * 2^23 so the int32 sum converts without I2F.
//
// Unlike ninfer, row views may start anywhere and end mid-tile (loads clamp to the view's first
// row, stores are guarded), gate and up rows of the SwiGLU may come from different parents, and
// the output is f32 [m, rows].
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

constexpr int kGroup = 256;   // activation values per scale
constexpr int kTokenPad = 128;  // sx stride granularity (largest token tile)
constexpr int kBlockCode = 96;  // IQ3 code bytes per 256-value block
constexpr int kIndexBytes = 64;

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
__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0, unsigned a1, unsigned a2,
                                       unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// ---------------------------------------------------------------------------------------------
// Activations: x [m, k] f32 -> xq [m, k] int8 and sx [k / 256, stride] f32 (group-major, tokens
// padded to kTokenPad). 8 values per thread, one warp per 256-value group. Layout 0: natural order;
// layout 1 (Q4/Q5): a thread's 8 values are one half of a lane's 16, evens first, then odds.
__global__ void __launch_bounds__(256) quantize_act(const float* __restrict__ x, int m, int k,
                                                    int8_t* __restrict__ xq, float* __restrict__ sx, int stride,
                                                    int layout) {
    const long long item = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int chunks = k / 8;
    const long long token = item / chunks;
    if (token >= m) return;  // whole warps: k / 8 is a multiple of 32
    const int chunk = static_cast<int>(item % chunks);
    const float4* src = reinterpret_cast<const float4*>(x + token * k) + chunk * 2;
    const float4 a = src[0], b = src[1];
    float v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        v[i] = __bfloat162float(__float2bfloat16_rn(v[i]));
        amax = fmaxf(amax, fabsf(v[i]));
    }
    amax = velox_warp_max(amax);
    const float inv = amax > 0.0f ? __fdiv_rn(127.0f, amax) : 0.0f;
    unsigned q[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) q[i] = static_cast<unsigned>(__float2int_rn(__fmul_rn(v[i], inv))) & 0xffu;
    const uint2 packed = layout == 0 ? make_uint2(q[0] | q[1] << 8 | q[2] << 16 | q[3] << 24,
                                                  q[4] | q[5] << 8 | q[6] << 16 | q[7] << 24)
                                     : make_uint2(q[0] | q[2] << 8 | q[4] << 16 | q[6] << 24,
                                                  q[1] | q[3] << 8 | q[5] << 16 | q[7] << 24);
    *reinterpret_cast<uint2*>(xq + token * k + chunk * 8) = packed;
    if ((chunk & 31) == 0) sx[static_cast<long long>(chunk / 32) * stride + token] = __fdiv_rn(amax, 127.0f);
}

int stride_of(int m) { return (m + kTokenPad - 1) / kTokenPad * kTokenPad; }

VeloxScratch g_act;

// Quantized activations in the shared scratch (grow-only: recorded graphs may hold it).
struct Act {
    const int8_t* xq;
    const float* sx;
    int stride;
};
int quantize(const float* x, int m, int k, int layout, Act& out) {
    const int stride = stride_of(m);
    const size_t xb = (static_cast<size_t>(m) * k + 255) / 256 * 256;
    auto* base = static_cast<uint8_t*>(g_act.get(xb + static_cast<size_t>(k / kGroup) * stride * 4));
    if (!base) return static_cast<int>(cudaErrorMemoryAllocation);
    auto* xq = reinterpret_cast<int8_t*>(base);
    auto* sx = reinterpret_cast<float*>(base + xb);
    const long long threads = static_cast<long long>(m) * (k / 8);
    quantize_act<<<static_cast<unsigned>((threads + 255) / 256), 256>>>(x, m, k, xq, sx, stride, layout);
    out = {xq, sx, stride};
    return velox_launch_status();
}

// ---------------------------------------------------------------------------------------------
// IQ3_XXS x int8.

__device__ __forceinline__ unsigned sign8(unsigned index7) { return index7 | ((__popc(index7) & 1u) << 7); }

// Signed int8x4 value of every (grid index, 4-bit sign nibble): table[index * 16 + nibble].
__device__ __forceinline__ void load_signed_grid(uint32_t* table) {
    for (int i = threadIdx.x; i < 4096; i += blockDim.x) {
        const unsigned nib = static_cast<unsigned>(i) & 15u;
        // 0xff in every byte whose sign bit is set; magnitudes <= 62 so ~m + 1 never carries.
        const unsigned msk = ((nib & 1u) * 0xffu) | ((nib & 2u) * 0x7f80u) | ((nib & 4u) * 0x3fc000u) |
                             ((nib & 8u) * 0x1fe00000u);
        table[i] = (kGrid[i >> 4] ^ msk) + (msk & 0x01010101u);
    }
}

constexpr int kXRow = 256;  // staged token row bytes; chunk c of token t at c ^ ((t & 7) << 1)

template <int BN>
struct Iq3Smem {
    static constexpr int kTable = 4096 * 4;
    static constexpr int kX = BN * kXRow;
    static constexpr int kStage = kX + BN * 4;
    static constexpr int kBytes = kTable + 2 * kStage;
};

struct Iq3Args {
    const int8_t* xq;
    const float* sx;
    int stride, m, k;
    const uint8_t* w;   // gate (or only) parent
    const uint8_t* w2;  // up parent (PAIR)
    long long parent_rows, parent_rows2, row_begin, row_begin2;
    int rows;           // output rows
    float* y;           // [m, rows]
};

// CTA = WM warps x 16 MT weight rows over BN = 8 NT tokens and the full K. Each warp loads its rows'
// codes straight into registers two blocks ahead; the token tile's int8 activations and scales
// stream through a 2-stage cp.async ring. grid = (ceil(m / BN), row tiles); token tiles vary
// fastest so a weight tile is re-read from L2 by neighbouring CTAs. PAIR: every 16-row MMA tile is
// 8 gate rows and the matching 8 up rows; out = silu(gate) * up.
template <int MT, int NT, int WM, int MB, bool PAIR>
__global__ void __launch_bounds__(WM * 32, MB) iq3_q8_kernel(Iq3Args p) {
    constexpr int kThreads = WM * 32, BM = 16 * MT * WM, BN = 8 * NT;
    using L = Iq3Smem<BN>;
    extern __shared__ __align__(16) uint8_t smem[];
    auto* table = reinterpret_cast<uint32_t*>(smem);
    const int blocks = p.k / 256;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, lid = lane & 3;
    const int token0 = blockIdx.x * BN;
    const int tile0 = blockIdx.y * (BM / 16) + warp * MT;  // warp's first 16-row MMA tile
    auto stage_base = [&](int slot) { return smem + L::kTable + slot * L::kStage; };

    auto stage = [&](int slot, int block) {
        uint8_t* xs = stage_base(slot);
        for (int i = tid; i < BN * 16; i += kThreads) {
            const int t = i >> 4, chunk = i & 15;
            const bool valid = token0 + t < p.m;
            const int8_t* src = p.xq + static_cast<long long>(valid ? token0 + t : 0) * p.k + block * 256 + chunk * 16;
            cp_async16(xs + t * kXRow + ((chunk ^ ((t & 7) << 1)) * 16), src, valid);
        }
        uint8_t* ss = xs + L::kX;
        for (int i = tid; i < BN / 4; i += kThreads)
            cp_async16(ss + i * 16, p.sx + static_cast<long long>(block) * p.stride + token0 + i * 4, true);
        cp_commit();
    };

    const uint8_t* row_codes[MT][2];
    const __half* row_d[MT][2];
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int tile = tile0 + mi;
            const bool up = PAIR && r == 1;
            const int o = PAIR ? tile * 8 + gid : tile * 16 + gid + 8 * r;
            const int oc = o < p.rows ? o : 0;  // rows past the view: output dropped
            const uint8_t* base = up ? p.w2 : p.w;
            const long long n = up ? p.parent_rows2 : p.parent_rows;
            const long long row = (up ? p.row_begin2 : p.row_begin) + oc;
            row_codes[mi][r] = base + row * blocks * kBlockCode;
            row_d[mi][r] = reinterpret_cast<const __half*>(base + (n * blocks * kBlockCode + 255) / 256 * 256) +
                           row * blocks;
        }
    }
    struct Raw {
        uint4 index;
        uint2 signs;
        __half d;
    };
    auto load = [&](int block, Raw (&raw)[MT][2]) {
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const uint8_t* b = row_codes[mi][r] + block * kBlockCode;
                raw[mi][r].index = __ldg(reinterpret_cast<const uint4*>(b + lid * 16));
                raw[mi][r].signs = __ldg(reinterpret_cast<const uint2*>(b + kIndexBytes + lid * 8));
                raw[mi][r].d = __ldg(row_d[mi][r] + block);
            }
        }
    };

    stage(0, 0);
    Raw cur[MT][2], next[MT][2], after[MT][2];
    load(0, cur);
    if (blocks > 1) load(1, next);
    load_signed_grid(table);

    float acc[MT][NT][4] = {};
#pragma unroll 1
    for (int block = 0; block < blocks; ++block) {
        const bool more = block + 1 < blocks;
        if (block + 2 < blocks) load(block + 2, after);
        if (more) {
            stage((block + 1) & 1, block + 1);
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();

        uint4 index[MT][2];
        uint64_t signs[MT][2];
        unsigned scales[MT][2];
        float dq[MT][2];  // d / 4
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                // The quad's eight sub-block scales (two per lane).
                const unsigned own = cur[mi][r].signs.y >> 24;
                unsigned s = 0;
#pragma unroll
                for (int q = 0; q < 4; ++q) s |= __shfl_sync(0xffffffffu, own, (lane & ~3) | q) << (8 * q);
                index[mi][r] = cur[mi][r].index;
                signs[mi][r] = (static_cast<uint64_t>(cur[mi][r].signs.y) << 32) | cur[mi][r].signs.x;
                scales[mi][r] = s;
                dq[mi][r] = 0.25f * __half2float(cur[mi][r].d);
            }
        }
        const uint8_t* xs = stage_base(block & 1);
        const float* ss = reinterpret_cast<const float*>(xs + L::kX);
        const uint8_t* xrow = xs + gid * kXRow + (lid >> 1) * 16 + (lid & 1) * 8;
        const float* srow = ss + lid * 2;
        // Software-pipelined: each MMA of sub-block s is issued next to the integer scaling of
        // sub-block s - 1 (whose products wait in prev), so the tensor pipe is fed while the IMADs run.
        int iacc[MT][NT][4];
        int prev[MT][NT][4];
        int fprev[MT][2];
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            unsigned a[MT][4];
            int f[MT][2];
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    const unsigned word = s < 2 ? index[mi][r].x : s < 4 ? index[mi][r].y : s < 6 ? index[mi][r].z
                                                                                              : index[mi][r].w;
                    const unsigned pair = word >> (16 * (s & 1));
                    const unsigned sg = sign8(static_cast<unsigned>(signs[mi][r] >> (7 * s)) & 127u);
                    a[mi][r] = table[((pair & 0xffu) << 4) | (sg & 15u)];
                    a[mi][r + 2] = table[((pair >> 4) & 0xff0u) | (sg >> 4)];
                    f[mi][r] = 1 + 2 * static_cast<int>((scales[mi][r] >> (4 * s)) & 15u);
                }
            }
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                // Sub-block s is chunks 2s and 2s+1 of the row, swizzled by the row's low bits (gid).
                const uint2 xb = *reinterpret_cast<const uint2*>(xrow + ni * 8 * kXRow + ((s ^ gid) << 5));
#pragma unroll
                for (int mi = 0; mi < MT; ++mi) {
                    int p0 = 0, p1 = 0, p2 = 0, p3 = 0;
                    mma_s8(p0, p1, p2, p3, a[mi][0], a[mi][1], a[mi][2], a[mi][3], xb.x, xb.y);
                    if (s == 1) {
                        iacc[mi][ni][0] = prev[mi][ni][0] * fprev[mi][0];
                        iacc[mi][ni][1] = prev[mi][ni][1] * fprev[mi][0];
                        iacc[mi][ni][2] = prev[mi][ni][2] * fprev[mi][1];
                        iacc[mi][ni][3] = prev[mi][ni][3] * fprev[mi][1];
                    } else if (s > 1) {
                        iacc[mi][ni][0] += prev[mi][ni][0] * fprev[mi][0];
                        iacc[mi][ni][1] += prev[mi][ni][1] * fprev[mi][0];
                        iacc[mi][ni][2] += prev[mi][ni][2] * fprev[mi][1];
                        iacc[mi][ni][3] += prev[mi][ni][3] * fprev[mi][1];
                    }
                    prev[mi][ni][0] = p0;
                    prev[mi][ni][1] = p1;
                    prev[mi][ni][2] = p2;
                    prev[mi][ni][3] = p3;
                }
            }
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) fprev[mi][0] = f[mi][0], fprev[mi][1] = f[mi][1];
        }
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            const float2 xs2 = *reinterpret_cast<const float2*>(srow + ni * 8);
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int v0 = iacc[mi][ni][0] + prev[mi][ni][0] * fprev[mi][0];
                const int v1 = iacc[mi][ni][1] + prev[mi][ni][1] * fprev[mi][0];
                const int v2 = iacc[mi][ni][2] + prev[mi][ni][2] * fprev[mi][1];
                const int v3 = iacc[mi][ni][3] + prev[mi][ni][3] * fprev[mi][1];
                acc[mi][ni][0] = fmaf(static_cast<float>(v0), dq[mi][0] * xs2.x, acc[mi][ni][0]);
                acc[mi][ni][1] = fmaf(static_cast<float>(v1), dq[mi][0] * xs2.y, acc[mi][ni][1]);
                acc[mi][ni][2] = fmaf(static_cast<float>(v2), dq[mi][1] * xs2.x, acc[mi][ni][2]);
                acc[mi][ni][3] = fmaf(static_cast<float>(v3), dq[mi][1] * xs2.y, acc[mi][ni][3]);
            }
        }
        if (more) {
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                cur[mi][0] = next[mi][0], cur[mi][1] = next[mi][1];
                next[mi][0] = after[mi][0], next[mi][1] = after[mi][1];
            }
        }
        __syncthreads();
    }

    // c0, c1: row gid, tokens 2 lid, 2 lid + 1; c2, c3: row gid + 8 (PAIR: the matching up row).
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
        const int tile = tile0 + mi;
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            const int t = token0 + ni * 8 + lid * 2;
            const float* v = acc[mi][ni];
            if (PAIR) {
                const int o = tile * 8 + gid;
                if (o >= p.rows) continue;
                if (t < p.m) p.y[static_cast<long long>(t) * p.rows + o] = v[0] / (1.0f + expf(-v[0])) * v[2];
                if (t + 1 < p.m) p.y[static_cast<long long>(t + 1) * p.rows + o] = v[1] / (1.0f + expf(-v[1])) * v[3];
            } else {
                const int o = tile * 16 + gid;
                if (o < p.rows) {
                    if (t < p.m) p.y[static_cast<long long>(t) * p.rows + o] = v[0];
                    if (t + 1 < p.m) p.y[static_cast<long long>(t + 1) * p.rows + o] = v[1];
                }
                if (o + 8 < p.rows) {
                    if (t < p.m) p.y[static_cast<long long>(t) * p.rows + o + 8] = v[2];
                    if (t + 1 < p.m) p.y[static_cast<long long>(t + 1) * p.rows + o + 8] = v[3];
                }
            }
        }
    }
}

template <int MT, int NT, int WM, int MB, bool PAIR>
int launch_iq3(const Iq3Args& p) {
    constexpr int BM = 16 * MT * WM, BN = 8 * NT, kSmem = Iq3Smem<BN>::kBytes;
    auto kernel = iq3_q8_kernel<MT, NT, WM, MB, PAIR>;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
        configured = true;
    }
    const int rows_per_cta = PAIR ? BM / 2 : BM;
    const dim3 grid((p.m + BN - 1) / BN, (p.rows + rows_per_cta - 1) / rows_per_cta);
    kernel<<<grid, WM * 32, kSmem>>>(p);
    return velox_launch_status();
}

// ninfer's choice on this card: 64 rows x 64 tokens (4 warps, two CTAs per SM) below 1024 tokens,
// 128 x 128 (8 warps) from there.
template <bool PAIR>
int run_iq3(const float* x, Iq3Args p) {
    if (p.m < 1 || p.rows <= 0 || p.k % 256 != 0 || reinterpret_cast<uintptr_t>(p.w) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(p.w2) % 16 != 0 || reinterpret_cast<uintptr_t>(x) % 16 != 0)
        return VELOX_ERR_UNSUPPORTED;
    Act a;
    if (const int e = quantize(x, p.m, p.k, 0, a)) return e;
    p.xq = a.xq, p.sx = a.sx, p.stride = a.stride;
    return p.m >= 1024 ? launch_iq3<1, 16, 8, 1, PAIR>(p) : launch_iq3<1, 8, 4, 2, PAIR>(p);
}

// ---------------------------------------------------------------------------------------------
// q4_g64 / q5_g64 (row_split_k128_v1) x int8.

constexpr int kMagicInt = 0x4B400000;
constexpr float kMagicF32 = 12582912.0f;

// Spreads bits 0, 2, 4, 6 of x to bit 0 of bytes 0..3.
__device__ __forceinline__ unsigned spread_even(unsigned x) { return ((x & 0x55u) * 0x41041u) & 0x01010101u; }
__device__ __forceinline__ unsigned sext4(unsigned v) { return v | ((v & 0x08080808u) * 0x1fu); }
__device__ __forceinline__ unsigned sext5(unsigned v) { return v | ((v & 0x10101010u) * 0x0fu); }

struct RsArgs {
    const int8_t* xq;
    const float* sx;
    int stride, m, k;
    const uint8_t* w;
    long long parent_rows, row_begin;
    int rows;
    float* y;
};

// CTA = WM warps x 16 MT rows over 8 NT tokens; each warp loads its rows' codes straight into
// registers one 256-value stage ahead, the token tile's activations are shared through a 2-stage
// cp.async ring.
template <bool Q5, int MT, int NT, int WM, int MB>
__global__ void __launch_bounds__(WM * 32, MB) rs_q8_kernel(RsArgs p) {
    constexpr int kThreads = WM * 32, BM = 16 * MT * WM, BN = 8 * NT;
    constexpr int kXBytes = BN * kGroup, kStageB = kXBytes + BN * 4;
    __shared__ __align__(16) uint8_t smem[2][kStageB];
    const int gpr = p.k / 64, stages = p.k / kGroup;
    const long long n_groups = p.parent_rows * gpr;
    const uint8_t* codes = p.w;
    const uint8_t* high = p.w + (n_groups * 32 + 255) / 256 * 256;
    const uint8_t* scales = high + (Q5 ? (n_groups * 8 + 255) / 256 * 256 : 0);

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, lid = lane & 3;
    const int token0 = blockIdx.x * BN;
    const int row0 = blockIdx.y * BM + warp * 16 * MT;  // view-relative

    auto stage = [&](int slot, int s) {
        uint8_t* xs = smem[slot];
        for (int i = tid; i < BN * 16; i += kThreads) {
            const int t = i >> 4, chunk = i & 15;
            const bool valid = token0 + t < p.m;
            const int8_t* src = p.xq + static_cast<long long>(valid ? token0 + t : 0) * p.k + s * kGroup + chunk * 16;
            cp_async16(xs + t * kGroup + ((chunk ^ ((t & 1) << 2)) * 16), src, valid);
        }
        for (int i = tid; i < BN / 4; i += kThreads)
            cp_async16(xs + kXBytes + i * 16, p.sx + static_cast<long long>(s) * p.stride + token0 + i * 4, true);
        cp_commit();
    };

    struct Raw {
        uint2 codes[4];
        unsigned high[4];
        uint2 scales;
    };
    const uint8_t* row_codes[MT][2];
    const uint8_t* row_high[MT][2];
    const uint8_t* row_scales[MT][2];
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int o = row0 + mi * 16 + gid + r * 8;
            const long long row = p.row_begin + (o < p.rows ? o : 0);  // rows past the view: output dropped
            row_codes[mi][r] = codes + row * gpr * 32 + lid * 8;
            row_high[mi][r] = high + row * gpr * 8 + lid * 2;
            row_scales[mi][r] = scales + row * gpr * 2;
        }
    }
    auto load = [&](int s, Raw (&raw)[MT][2]) {
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
            for (int r = 0; r < 2; ++r) {
#pragma unroll
                for (int g = 0; g < 4; ++g) {
                    raw[mi][r].codes[g] = __ldg(reinterpret_cast<const uint2*>(row_codes[mi][r] + (4 * s + g) * 32));
                    raw[mi][r].high[g] =
                        Q5 ? __ldg(reinterpret_cast<const unsigned short*>(row_high[mi][r] + (4 * s + g) * 8)) : 0u;
                }
                raw[mi][r].scales = __ldg(reinterpret_cast<const uint2*>(row_scales[mi][r] + s * 8));
            }
        }
    };

    stage(0, 0);
    Raw cur[MT][2], next[MT][2];
    load(0, cur);

    float acc[MT][NT][4] = {};
#pragma unroll 1
    for (int s = 0; s < stages; ++s) {
        const bool more = s + 1 < stages;
        if (more) {
            load(s + 1, next);
            stage((s + 1) & 1, s + 1);
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();

        const uint8_t* xs = smem[s & 1];
        const uint8_t* xrow = xs + gid * kGroup;
        const float* srow = reinterpret_cast<const float*>(xs + kXBytes) + lid * 2;
        float gacc[MT][NT][4];
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            unsigned a0[MT][4], a1[MT][4];  // first and second k32 step
            float ws[MT][2];
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    const uint2 c = cur[mi][r].codes[g];
                    unsigned lo0 = c.x & 0x0f0f0f0fu, hi0 = (c.x >> 4) & 0x0f0f0f0fu;
                    unsigned lo1 = c.y & 0x0f0f0f0fu, hi1 = (c.y >> 4) & 0x0f0f0f0fu;
                    if (Q5) {
                        const unsigned h = cur[mi][r].high[g];
                        lo0 = sext5(lo0 | (spread_even(h) << 4));
                        hi0 = sext5(hi0 | (spread_even(h >> 1) << 4));
                        lo1 = sext5(lo1 | (spread_even(h >> 8) << 4));
                        hi1 = sext5(hi1 | (spread_even(h >> 9) << 4));
                    } else {
                        lo0 = sext4(lo0), hi0 = sext4(hi0), lo1 = sext4(lo1), hi1 = sext4(hi1);
                    }
                    a0[mi][r] = lo0, a0[mi][r + 2] = hi0;
                    a1[mi][r] = lo1, a1[mi][r + 2] = hi1;
                    const unsigned sbits = (g < 2 ? cur[mi][r].scales.x : cur[mi][r].scales.y) >> (16 * (g & 1));
                    ws[mi][r] = __half2float(__ushort_as_half(static_cast<unsigned short>(sbits)));
                }
            }
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int t = ni * 8 + gid;
                const uint4 xb =
                    *reinterpret_cast<const uint4*>(xrow + ni * 8 * kGroup + (((g * 4 + lid) ^ ((t & 1) << 2)) * 16));
#pragma unroll
                for (int mi = 0; mi < MT; ++mi) {
                    int p0 = kMagicInt, p1 = kMagicInt, p2 = kMagicInt, p3 = kMagicInt;
                    mma_s8(p0, p1, p2, p3, a0[mi][0], a0[mi][1], a0[mi][2], a0[mi][3], xb.x, xb.y);
                    mma_s8(p0, p1, p2, p3, a1[mi][0], a1[mi][1], a1[mi][2], a1[mi][3], xb.z, xb.w);
                    const float v0 = (__int_as_float(p0) - kMagicF32) * ws[mi][0];
                    const float v1 = (__int_as_float(p1) - kMagicF32) * ws[mi][0];
                    const float v2 = (__int_as_float(p2) - kMagicF32) * ws[mi][1];
                    const float v3 = (__int_as_float(p3) - kMagicF32) * ws[mi][1];
                    if (g == 0) {
                        gacc[mi][ni][0] = v0, gacc[mi][ni][1] = v1, gacc[mi][ni][2] = v2, gacc[mi][ni][3] = v3;
                    } else {
                        gacc[mi][ni][0] += v0, gacc[mi][ni][1] += v1, gacc[mi][ni][2] += v2, gacc[mi][ni][3] += v3;
                    }
                }
            }
        }
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            const float2 xs2 = *reinterpret_cast<const float2*>(srow + ni * 8);
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                acc[mi][ni][0] = fmaf(gacc[mi][ni][0], xs2.x, acc[mi][ni][0]);
                acc[mi][ni][1] = fmaf(gacc[mi][ni][1], xs2.y, acc[mi][ni][1]);
                acc[mi][ni][2] = fmaf(gacc[mi][ni][2], xs2.x, acc[mi][ni][2]);
                acc[mi][ni][3] = fmaf(gacc[mi][ni][3], xs2.y, acc[mi][ni][3]);
            }
        }
        if (more) {
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) cur[mi][0] = next[mi][0], cur[mi][1] = next[mi][1];
        }
        __syncthreads();
    }

#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
        const int r0 = row0 + mi * 16 + gid;
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            const int t = token0 + ni * 8 + lid * 2;
            auto put = [&](int token, int row, float value) {
                if (token < p.m && row < p.rows) p.y[static_cast<long long>(token) * p.rows + row] = value;
            };
            put(t, r0, acc[mi][ni][0]);
            put(t + 1, r0, acc[mi][ni][1]);
            put(t, r0 + 8, acc[mi][ni][2]);
            put(t + 1, r0 + 8, acc[mi][ni][3]);
        }
    }
}

// ninfer's choice on this card: MT=1, NT=8, 4 warps (64 rows x 64 tokens), two CTAs per SM.
template <bool Q5>
int launch_rs(const RsArgs& p) {
    constexpr int MT = 1, NT = 8, WM = 4, BM = 16 * MT * WM, BN = 8 * NT;
    const dim3 grid((p.m + BN - 1) / BN, (p.rows + BM - 1) / BM);
    rs_q8_kernel<Q5, MT, NT, WM, 2><<<grid, WM * 32>>>(p);
    return velox_launch_status();
}

}  // namespace

// y[m, rows] = int8(x) . W[row_begin.., :]^T for an iq3_xxs_planar_v1 parent [parent_rows, k]
// (ActQuant::Int8G256); x 16-byte aligned. Returns VELOX_ERR_UNSUPPORTED for anything else.
VELOX_API int velox_5060_linear_iq3_q8(const float* x, int m, const uint8_t* w, long long parent_rows, long long k,
                                       long long row_begin, int rows, float* y) {
    if (k > (1ll << 30)) return VELOX_ERR_UNSUPPORTED;
    Iq3Args p{};
    p.m = m, p.k = static_cast<int>(k), p.rows = rows, p.y = y;
    p.w = p.w2 = w;
    p.parent_rows = p.parent_rows2 = parent_rows;
    p.row_begin = p.row_begin2 = row_begin;
    return run_iq3<false>(x, p);
}

// out[m, rows] = silu(int8(x) . G^T) * (int8(x) . U^T) for `rows` gate rows of the parent at `w`
// from `gate_begin` and as many up rows of the parent at `up_w` from `up_begin` (same k).
VELOX_API int velox_5060_linear_swiglu_iq3_q8(const float* x, int m, const uint8_t* w, long long parent_rows,
                                              long long gate_begin, const uint8_t* up_w, long long up_parent_rows,
                                              long long up_begin, long long k, int rows, float* out) {
    if (k > (1ll << 30)) return VELOX_ERR_UNSUPPORTED;
    Iq3Args p{};
    p.m = m, p.k = static_cast<int>(k), p.rows = rows, p.y = out;
    p.w = w, p.parent_rows = parent_rows, p.row_begin = gate_begin;
    p.w2 = up_w, p.parent_rows2 = up_parent_rows, p.row_begin2 = up_begin;
    return run_iq3<true>(x, p);
}

// y[m, rows] = int8(x) . W[row_begin.., :]^T for a row_split_k128_v1 parent [parent_rows, k] in
// format 4 (q4_g64) or 5 (q5_g64) with k % 256 == 0 (ActQuant::Int8G256).
VELOX_API int velox_5060_linear_rs_q8(const float* x, int m, const uint8_t* w, long long parent_rows, long long k,
                                      long long row_begin, int rows, int format, float* y) {
    if (m < 1 || rows <= 0 || k % 256 != 0 || k > (1ll << 30) || (format != 4 && format != 5) ||
        reinterpret_cast<uintptr_t>(w) % 16 != 0 || reinterpret_cast<uintptr_t>(x) % 16 != 0)
        return VELOX_ERR_UNSUPPORTED;
    Act a;
    if (const int e = quantize(x, m, static_cast<int>(k), 1, a)) return e;
    RsArgs p{a.xq, a.sx, a.stride, m, static_cast<int>(k), w, parent_rows, row_begin, rows, y};
    return format == 5 ? launch_rs<true>(p) : launch_rs<false>(p);
}
