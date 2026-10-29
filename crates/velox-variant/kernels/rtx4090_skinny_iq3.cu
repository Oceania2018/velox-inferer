// RTX 5060 Ti skinny linear (1..16 tokens) for IQ3_XXS weights (iq3_xxs_planar_v1, see
// velox_core::format::iq3_xxs_at). Activations are rounded to bf16 (ActQuant::Bf16); the weights
// stay exact: signed grid magnitudes are integers <= 62 (bf16-exact), and each 32-value
// sub-block accumulates into its own register tile, scaled by d * (0.25 + 0.5 ls) before it is
// added.
//
// Decode and verify are bound by streaming the weights (and on this card by warps in flight:
// 8 warps x 2 stages measured best). Weights are the A operand of
// mma.m16n8k16 (each warp owns 16 rows) and tokens the B operand (8 per n-tile), so up to 8 tokens
// cost one MMA per k16 step. Each cp.async stage reads kBlk consecutive IQ3 blocks (kBlk * 96
// contiguous bytes) of each of the block's 64 rows. Activations are stored in k_perm order within
// each 32-value chunk, so the four lanes q of an MMA quad read exactly the values ninfer's layout
// gives them: grid entries 2q, 2q+1 of each sub-block (index bytes 16q + 2s + {0,1}) and sign
// group q (lane word q); A and B fragments cover the same K positions.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include "velox_common.cuh"

namespace {

#ifndef IQ3_BLK
#define IQ3_BLK 1
#endif
#ifndef IQ3_STAGES
#define IQ3_STAGES 2
#endif
constexpr int kBlk = IQ3_BLK;       // IQ3 blocks per row per stage
constexpr int kStages = IQ3_STAGES;
#ifndef IQ3_WARPS
#define IQ3_WARPS 8
#endif
// Each warp owns 16 weight rows.
constexpr int kWarps = IQ3_WARPS, kBN = 16 * kWarps, kThreads = 32 * kWarps, kCode = 96;
constexpr int kKS = kBlk * 256;     // K values per stage
constexpr int kRow = kBlk * kCode;  // code bytes per row per stage
constexpr int kStride = kRow + 16;  // padded so the 8 rows a fragment reads spread over banks
constexpr int kSA = kKS + 8;        // activation row: kKS bf16 + 8 pad

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

template <int NT>
struct Smem {
    __nv_bfloat16 a[kStages][NT * 8 * kSA];
    uint8_t b[kStages][kBN * kStride];
    // Grid entries as bf16 pairs: .x = magnitudes (0, 1), .y = (2, 3).
    uint2 grid[256];
};

struct Args {
    const __nv_bfloat16* x;                    // [m, k] bf16 in k_perm order
    const uint8_t* w;                          // parent payload
    long long parent_rows, row_begin, blocks;  // blocks: IQ3 blocks per row (k / 256)
    long long scale_off;                       // binary16 d plane
    int rows, m, k;
    int kt_per_split;  // stages per split-K slice
    float* y;          // [m, rows], or [splits, m, rows] partials
    // Fused SwiGLU (pair != 0): rows are output rows; each warp's 16 tile rows are 8 gate rows
    // (w from row_begin) and the matching 8 up rows (w2 from row_begin2), and
    // y[t][r] = silu(gate . x_t) * (up . x_t). No split-K.
    int pair;
    const uint8_t* w2;
    long long parent_rows2, row_begin2, scale_off2;
};

// Parent payload and row of shared tile row r (0..kBN) for block output rows from n0.
struct TileRow {
    const uint8_t* w;
    long long row;
    bool ok;
    int second;
};
__device__ __forceinline__ TileRow tile_row(const Args& p, int n0, int r) {
    if (!p.pair) {
        const long long row = p.row_begin + n0 + r;
        return {p.w, row, n0 + r < p.rows && row < p.parent_rows, 0};
    }
    const int e = (r & 15) >> 3, o = n0 + (r >> 4) * 8 + (r & 7);
    const long long row = (e ? p.row_begin2 : p.row_begin) + o;
    return {e ? p.w2 : p.w, row, o < p.rows && row < (e ? p.parent_rows2 : p.parent_rows), e};
}

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
__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// MMA position l = 16s + 8h + 2q + j within a 32-value chunk reads weight value 8q + 4s + 2h + j.
__host__ __device__ __forceinline__ int k_perm(int l) {
    const int s = (l >> 4) & 1, h = (l >> 3) & 1, q = (l >> 1) & 3, j = l & 1;
    return (l & ~31) + 8 * q + 4 * s + 2 * h + j;
}

// BF16 sign bits for the pair selected by bits 0 and 1 of `bits`.
__device__ __forceinline__ uint32_t sign_mask(uint32_t bits) {
    return ((bits & 1u) << 15) | ((bits & 2u) << 30);
}

template <int NT>
__device__ __forceinline__ void load_stage(Smem<NT>& s, const Args& p, int stage, int kt, int n0) {
    const int t = threadIdx.x;
    const long long k0 = static_cast<long long>(kt) * kKS;
    // Only the m real token rows (padding rows were zeroed once); a partial last stage zero-fills.
    for (int idx = t; idx < p.m * (kKS / 8); idx += kThreads) {
        const int r = idx / (kKS / 8), c = idx % (kKS / 8);
        const bool ok = k0 + c * 8 < p.k;
        cp_async16(&s.a[stage][r * kSA + c * 8], p.x + static_cast<long long>(r) * p.k + (ok ? k0 + c * 8 : 0), ok);
    }
    constexpr int kChunks = kRow / 16;
    for (int idx = t; idx < kBN * kChunks; idx += kThreads) {
        const int r = idx / kChunks, c = idx % kChunks;
        const TileRow tr = tile_row(p, n0, r);
        const long long blk = static_cast<long long>(kt) * kBlk + c / (kCode / 16);
        const bool ok = tr.ok && blk < p.blocks;
        cp_async16(&s.b[stage][r * kStride + c * 16],
                   tr.w + (ok ? (tr.row * p.blocks + kt * static_cast<long long>(kBlk)) * kCode + c * 16 : 0), ok);
    }
}

template <int NT>
__global__ void __launch_bounds__(kThreads) skinny_iq3_kernel(Args p) {
    // Prefill: blockIdx.y picks this block's tile of NT * 8 tokens (weights decoded once per tile).
    {
        const int t0 = blockIdx.y * NT * 8;
        p.x += static_cast<long long>(t0) * p.k;
        p.y += static_cast<long long>(t0) * p.rows;
        p.m = min(NT * 8, p.m - t0);
    }
    extern __shared__ __align__(128) uint8_t smem_raw[];
    Smem<NT>& s = *reinterpret_cast<Smem<NT>*>(smem_raw);
    const int n0 = blockIdx.x * (p.pair ? kBN / 2 : kBN);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int stages = static_cast<int>((p.blocks + kBlk - 1) / kBlk);
    const int kt_begin = blockIdx.z * p.kt_per_split;
    const int kt_end = min(stages, kt_begin + p.kt_per_split);
    const int kt_count = kt_end - kt_begin;
    float* y = p.y + static_cast<long long>(blockIdx.z) * p.m * p.rows;

    for (int i = threadIdx.x; i < 256; i += kThreads) {
        const uint32_t g = kGrid[i];
        uint32_t b[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) b[j] = __float_as_uint(static_cast<float>((g >> (8 * j)) & 0xffu)) >> 16;
        s.grid[i] = make_uint2(b[0] | (b[1] << 16), b[2] | (b[3] << 16));
    }
    for (int i = p.m * kSA + threadIdx.x; i < NT * 8 * kSA; i += kThreads)
#pragma unroll
        for (int st = 0; st < kStages; ++st) s.a[st][i] = __float2bfloat16_rn(0.0f);

#pragma unroll
    for (int st = 0; st < kStages - 1; ++st) {
        if (st < kt_count) load_stage<NT>(s, p, st, kt_begin + st, n0);
        cp_commit();
    }

    float acc[NT][4];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;

    const int g = lane >> 2, q = lane & 3;
    // A rows (and C rows) of this lane: g and g + 8 of the warp's 16.
    const int ar[2] = {warp * 16 + g, warp * 16 + 8 + g};
    long long drow[2];
    const __half* dps[2];
#pragma unroll
    for (int e = 0; e < 2; ++e) {
        const TileRow tr = tile_row(p, n0, ar[e]);
        drow[e] = (tr.ok ? tr.row : 0) * p.blocks;  // rows past the view: output dropped
        dps[e] = reinterpret_cast<const __half*>(tr.w + (tr.second ? p.scale_off2 : p.scale_off));
    }
    // ldmatrix x4 over 8 token rows: matrix j covers K columns 8j..8j+7 of a 32-value chunk.
    const int b_row = lane & 7, b_col = (lane >> 3) * 8;

    for (int i = 0; i < kt_count; ++i) {
        cp_wait<kStages - 2>();
        __syncthreads();
        const int next = i + kStages - 1;
        if (next < kt_count) load_stage<NT>(s, p, next % kStages, kt_begin + next, n0);
        cp_commit();
        const int st = i % kStages;
        const long long blk0 = static_cast<long long>(kt_begin + i) * kBlk;
#pragma unroll
        for (int bk = 0; bk < kBlk; ++bk) {
            if (blk0 + bk >= p.blocks) break;  // partial last stage
            const uint8_t* rowp[2] = {&s.b[st][ar[0] * kStride + bk * kCode], &s.b[st][ar[1] * kStride + bk * kCode]};
            uint64_t sw[2];
            float d[2];
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                sw[e] = *reinterpret_cast<const uint64_t*>(rowp[e] + 64 + 8 * q);
                d[e] = __half2float(dps[e][drow[e] + blk0 + bk]);
            }
#pragma unroll
            for (int sb = 0; sb < 8; ++sb) {
                const int kc = bk * 256 + sb * 32;  // stage column of this 32-value chunk
                uint32_t bf[NT][4];
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) ldsm_x4(bf[nt], &s.a[st][(nt * 8 + b_row) * kSA + kc + b_col]);
                // A fragment per k16 step: {row g h=0, row g+8 h=0, row g h=1, row g+8 h=1}.
                uint32_t af[2][4];
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const uint16_t idx = *reinterpret_cast<const uint16_t*>(rowp[e] + 16 * q + 2 * sb);
                    const uint32_t s7 = static_cast<uint32_t>(sw[e] >> (7 * sb)) & 127u;
                    const uint32_t s8 = s7 | ((__popc(s7) & 1u) << 7);
#pragma unroll
                    for (int st2 = 0; st2 < 2; ++st2) {
                        const uint2 mg = s.grid[(idx >> (8 * st2)) & 0xff];
                        const uint32_t bits = s8 >> (4 * st2);
                        af[st2][e] = mg.x | sign_mask(bits);
                        af[st2][2 + e] = mg.y | sign_mask(bits >> 2);
                    }
                }
                float part[NT][4];
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    part[nt][0] = part[nt][1] = part[nt][2] = part[nt][3] = 0.0f;
                    mma_bf16(part[nt], af[0], bf[nt][0], bf[nt][1]);
                    mma_bf16(part[nt], af[1], bf[nt][2], bf[nt][3]);
                }
                // Sub-block scales d (0.25 + 0.5 ls), ls in byte 7 of lane word sb / 2.
                float sc[2];
#pragma unroll
                for (int e = 0; e < 2; ++e)
                    sc[e] = d[e] *
                            (0.25f + 0.5f * static_cast<float>((rowp[e][64 + 8 * (sb >> 1) + 7] >> (4 * (sb & 1))) & 15));
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    acc[nt][0] += part[nt][0] * sc[0];
                    acc[nt][1] += part[nt][1] * sc[0];
                    acc[nt][2] += part[nt][2] * sc[1];
                    acc[nt][3] += part[nt][3] * sc[1];
                }
            }
        }
    }
    cp_wait<0>();

    if (p.pair) {
        // c0, c1: gate row, tokens 2q, 2q+1; c2, c3: the matching up row.
        const int o = n0 + warp * 8 + g;
        if (o < p.rows) {
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int t = nt * 8 + 2 * q + j;
                    const float gv = acc[nt][j];
                    if (t < p.m) y[static_cast<long long>(t) * p.rows + o] = gv / (1.0f + expf(-gv)) * acc[nt][2 + j];
                }
        }
        return;
    }
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

// x [m, k] f32 -> [m, k] bf16 (RNE) in k_perm order.
__global__ void to_bf16_perm(const float* __restrict__ x, int m, int k, __nv_bfloat16* __restrict__ y) {
    const long long n = static_cast<long long>(m) * k;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const long long r = i / k;
        y[i] = __float2bfloat16_rn(x[r * k + k_perm(static_cast<int>(i % k))]);
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

constexpr int kSms = 128;

long long align256(long long x) { return (x + 255) / 256 * 256; }

// Split-K factor (power of two, at least 4 stages per slice) whose grid best fills whole waves of
// `slots` resident blocks; the smallest split within 5% of the best wins (fewer partials to sum).
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

template <int NT>
int launch(Args p) {
    static int per_sm = 0;
    if (per_sm == 0) {
        cudaFuncSetAttribute(skinny_iq3_kernel<NT>, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Smem<NT>));
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, skinny_iq3_kernel<NT>, kThreads, sizeof(Smem<NT>));
        per_sm = per_sm > 0 ? per_sm : 1;
    }
    const int rows_per_block = p.pair ? kBN / 2 : kBN;
    const long long blocks = (p.rows + rows_per_block - 1) / rows_per_block, stages = (p.blocks + kBlk - 1) / kBlk;
    const int tiles = (p.m + NT * 8 - 1) / (NT * 8);
    const int splits = p.pair || tiles > 1 ? 1 : pick_splits(blocks, stages, kSms * per_sm);
    p.kt_per_split = static_cast<int>((stages + splits - 1) / splits);
    float* y = p.y;
    if (splits > 1) {
        p.y = static_cast<float*>(g_part.get(static_cast<size_t>(splits) * p.m * p.rows * sizeof(float)));
        if (!p.y) return static_cast<int>(cudaErrorMemoryAllocation);
    }
    skinny_iq3_kernel<NT><<<dim3(static_cast<unsigned>(blocks), tiles, splits), kThreads, sizeof(Smem<NT>)>>>(p);
    if (splits > 1) {
        const long long n = static_cast<long long>(p.m) * p.rows;
        sum_splits<<<velox_grid_1d(n, 256), 256>>>(p.y, splits, n, y);
    }
    return velox_launch_status();
}

}  // namespace

namespace {
int run(const float* x, Args p, int m, long long k) {
    if (m < 1 || (p.pair && m > 16) || p.rows <= 0 || k % 256 != 0 || reinterpret_cast<uintptr_t>(p.w) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(p.w2) % 16 != 0)
        return VELOX_ERR_UNSUPPORTED;
    p.blocks = k / 256;
    p.scale_off = align256(p.parent_rows * p.blocks * kCode);
    p.scale_off2 = align256(p.parent_rows2 * p.blocks * kCode);
    p.m = m;
    p.k = static_cast<int>(k);
    auto* xb = static_cast<__nv_bfloat16*>(g_x.get(static_cast<size_t>(m) * k * 2));
    if (!xb) return static_cast<int>(cudaErrorMemoryAllocation);
    to_bf16_perm<<<velox_grid_1d(static_cast<long long>(m) * k, 256), 256>>>(x, m, p.k, xb);
    p.x = xb;
    // Decode / verify: one tile of 8 or 16 tokens. Prefill: 64-token tiles.
    if (m <= 8) return launch<1>(p);
    return m <= 16 ? launch<2>(p) : launch<8>(p);
}
}  // namespace

// y[m, rows] = bf16(x) . W[row_begin.., :]^T for an iq3_xxs_planar_v1 parent [parent_rows, k],
// m >= 1 (more than 16 tokens run in 64-token tiles). Returns VELOX_ERR_UNSUPPORTED otherwise.
VELOX_API int velox_4090_linear_skinny_iq3(const float* x, int m, const uint8_t* w, long long parent_rows, long long k,
                                           long long row_begin, int rows, float* y) {
    Args p{};
    p.w = p.w2 = w;
    p.parent_rows = p.parent_rows2 = parent_rows;
    p.row_begin = p.row_begin2 = row_begin;
    p.rows = rows;
    p.y = y;
    return run(x, p, m, k);
}

// out[m, rows] = silu(bf16(x) . G^T) * (bf16(x) . U^T) for `rows` gate rows of the parent at `w`
// from `gate_begin` and as many up rows of the parent at `up_w` from `up_begin` (same k).
VELOX_API int velox_4090_linear_swiglu_skinny_iq3(const float* x, int m, const uint8_t* w, long long parent_rows,
                                                  long long gate_begin, const uint8_t* up_w,
                                                  long long up_parent_rows, long long up_begin, long long k, int rows,
                                                  float* out) {
    Args p{};
    p.pair = 1;
    p.w = w;
    p.parent_rows = parent_rows;
    p.row_begin = gate_begin;
    p.w2 = up_w;
    p.parent_rows2 = up_parent_rows;
    p.row_begin2 = up_begin;
    p.rows = rows;
    p.y = out;
    return run(x, p, m, k);
}
