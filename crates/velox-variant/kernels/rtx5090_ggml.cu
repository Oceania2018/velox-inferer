// RTX 5090 linear for GGML block weights (ggml_block_v1, GGUF imports) at decode and verify widths:
// bf16 tensor cores (mma.m16n8k16), weights as the A operand, up to 64 tokens as B.
//
// Every GGML type here decodes exactly into 16-value pieces of small integers (|q| <= 127, exact in
// bf16) times one f32 scale, minus one f32 min (Q2_K / Q4_K only):
//   q2_k   q in 0..3,  scale d*sc,        min dmin*m       (per 16)
//   q4_k   q in 0..15, scale d*sc,        min dmin*m       (per 32)
//   iq1_m  8g +- 1,    scale dl / 8                        (per 16)
//   iq2_*  grid*sign,  scale d(0.5+ls)/4                   (per 32 or 16)
//   iq3_*  grid*sign,  scale d(0.5+ls)/2 or d(1+2ls)       (per 32)
//   iq4_xs kvalues,    scale d(ls-32)                      (per 32)
// so one k16 MMA step covers one piece and its partial sum is scaled once: y = sum_p s_p (q_p . x)
// - m_p (sum x). K is permuted inside each piece so that quad lane t's A operands are the four
// consecutive values [4t, 4t+4) (one IQ3 grid entry, half an IQ2 one) for its two rows, and its B
// operands are the same four activations: one float4 load per token.
//
// A lane loads its two rows' whole super-block with aligned 8-byte loads and funnel-shifts it into
// registers (block sizes are even, so a block starts 0, 2, 4 or 6 bytes into its 8-byte word);
// pieces then decode at compile-time byte offsets. Lane-dependent offsets only take the forms
// C + t, C + t / 2 and C + 4t, which a 4-byte window or a 4-way select resolves.
//
// Activations are rounded to bf16 (ActQuant::Bf16). Sums for the mins are of the rounded values.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include "ggml_tables.cuh"
#include "rtx5090_ggml_bf16.cuh"
#include "velox_common.cuh"

namespace {

__host__ __device__ constexpr int block_bytes(int t) {
    return t == 0 ? 84 : t == 1 ? 144 : t == 2 ? 56 : t == 3 ? 66 : t == 4 ? 74 : t == 5 ? 82 : t == 6 ? 98 : t == 7 ? 110 : 136;
}
__host__ __device__ constexpr bool has_min(int t) { return t <= 1; }
// Registers holding a normalized block (one spare word for 4-byte windows at the end).
__host__ __device__ constexpr int block_words(int t) { return (block_bytes(t) + 3) / 4 + 1; }

// Bytes [c, c + 4) of a normalized block (c a compile-time constant after unrolling).
__device__ __forceinline__ uint32_t W(const uint32_t* r, int c) {
    return c % 4 == 0 ? r[c / 4] : __funnelshift_r(r[c / 4], r[c / 4 + 1], 8 * (c % 4));
}
__device__ __forceinline__ uint32_t B(const uint32_t* r, int c) { return (r[c / 4] >> (8 * (c % 4))) & 0xffu; }
__device__ __forceinline__ float H(uint32_t bits) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(bits & 0xffffu)));
}
// Word t (0..3) of bytes [c, c + 16).
__device__ __forceinline__ uint32_t W4(const uint32_t* r, int c, int t) {
    const uint32_t a = W(r, c), b = W(r, c + 4), d = W(r, c + 8), e = W(r, c + 12);
    const uint32_t lo = t & 1 ? b : a, hi = t & 1 ? e : d;
    return t & 2 ? hi : lo;
}

// Loads the block at `p` (2-byte aligned) into r[0 .. block_words(T)), byte i of the block at bits
// 8 (i % 4) of r[i / 4]. Words past `end` (the parent end, 8-byte aligned) read as zero.
template <int T>
__device__ __forceinline__ void load_block(const uint8_t* __restrict__ p, const uint8_t* end, uint32_t* r) {
    constexpr int NR = block_words(T), N64 = (NR + 2) / 2 + 1;
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    const uint2* base = reinterpret_cast<const uint2*>(a & ~static_cast<uintptr_t>(7));
    const int mis = static_cast<int>(a & 7);  // 0, 2, 4 or 6
    uint32_t raw[2 * N64];
#pragma unroll
    for (int i = 0; i < N64; ++i) {
        const uint2 v = reinterpret_cast<const uint8_t*>(base + i) < end ? __ldg(base + i) : make_uint2(0, 0);
        raw[2 * i] = v.x;
        raw[2 * i + 1] = v.y;
    }
    const bool word = mis >= 4;
    const int sh = 8 * (mis & 3);
#pragma unroll
    for (int i = 0; i < NR; ++i) {
        const uint32_t lo = word ? raw[i + 1] : raw[i], hi = word ? raw[i + 2] : raw[i + 1];
        r[i] = __funnelshift_r(lo, hi, sh);
    }
}

// Two signed integers as a bf16x2 word (exact for |v| <= 256).
__device__ __forceinline__ uint32_t bf16x2_of(int lo, int hi) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(static_cast<float>(lo), static_cast<float>(hi));
    return *reinterpret_cast<const uint32_t*>(&v);
}

// Sign masks of four values (sign bits s & 1 .. s & 8) for their two bf16x2 words.
__device__ __align__(16) const uint2 kSignMask[16] = {
    {0x00000000u, 0x00000000u}, {0x00008000u, 0x00000000u}, {0x80000000u, 0x00000000u}, {0x80008000u, 0x00000000u},
    {0x00000000u, 0x00008000u}, {0x00008000u, 0x00008000u}, {0x80000000u, 0x00008000u}, {0x80008000u, 0x00008000u},
    {0x00000000u, 0x80000000u}, {0x00008000u, 0x80000000u}, {0x80000000u, 0x80000000u}, {0x80008000u, 0x80000000u},
    {0x00000000u, 0x80008000u}, {0x00008000u, 0x80008000u}, {0x80000000u, 0x80008000u}, {0x80008000u, 0x80008000u},
};
__device__ __forceinline__ void apply_signs(uint2 g, uint32_t s, uint32_t& lo, uint32_t& hi) {
    const uint2 m = __ldg(&kSignMask[s]);
    lo = g.x ^ m.x, hi = g.y ^ m.y;
}

// Grid entry words: n = words per entry (2 for 4-value grids, 4 for 8-value ones), half = which
// 4 values of an 8-value entry.
__device__ __forceinline__ uint2 grid_words(const uint32_t* table, uint32_t idx, int n, int half) {
    return __ldg(reinterpret_cast<const uint2*>(table + idx * n) + half);
}

// Nibble-pair words of the four codes in byte lanes of `b` (low or high nibbles).
__device__ __forceinline__ void nibble_words(const uint32_t* table, uint32_t b, bool high, uint32_t& lo, uint32_t& hi) {
    const uint32_t v = high ? b >> 4 : b;
    lo = __ldg(table + ((v & 0xf) | ((v >> 4) & 0xf0)));
    hi = __ldg(table + (((v >> 16) & 0xf) | ((v >> 20) & 0xf0)));
}

// Piece P (values [16P, 16P + 16)) of a normalized block: values 16P + 4t .. 16P + 4t + 3 as two
// bf16x2 words (lo: the first two), and the piece's scale and min.
template <int T, int P>
__device__ __forceinline__ void piece(const uint32_t* r, int t, uint32_t& lo, uint32_t& hi, float& scale, float& mn) {
    mn = 0.0f;
    constexpr int ib = P / 2, h = P % 2;
    if (T == 0) {  // Q2_K: scales[16] | qs[64] | d | dmin
        constexpr int n = P / 8, sh = 2 * ((P % 8) / 2);
        const uint32_t sc = B(r, P), dd = W(r, 80);
        scale = H(dd) * static_cast<float>(sc & 0xf);
        mn = H(dd >> 16) * static_cast<float>(sc >> 4);
        const uint32_t b = (W4(r, 16 + 32 * n + 16 * h, t) >> sh) & 0x03030303u;
        nibble_words(kBfIdPair, b, false, lo, hi);
    } else if (T == 1) {  // Q4_K: d | dmin | scales[12] | qs[128]
        constexpr int c = P / 4, hb = (P % 4) / 2, s = 2 * c + hb;
        uint32_t a, m;
        if (s < 4) {
            a = B(r, 4 + s) & 63, m = B(r, 8 + s) & 63;
        } else {
            a = (B(r, 8 + s) & 0xf) | ((B(r, s) >> 6) << 4);
            m = (B(r, 8 + s) >> 4) | ((B(r, 4 + s) >> 6) << 4);
        }
        const uint32_t dd = W(r, 0);
        scale = H(dd) * static_cast<float>(a);
        mn = H(dd >> 16) * static_cast<float>(m);
        nibble_words(kBfIdPair, W4(r, 16 + 32 * c + 16 * h, t), hb, lo, hi);
    } else if (T == 2) {  // IQ1_M: qs[32] | qh[16] | scales[8]
        const uint32_t s01 = W(r, 48), s23 = W(r, 52);
        const uint32_t dbits = ((s01 & 0xffff) >> 12) | ((s01 >> 24) & 0x00f0) | (((s23 & 0xffff) >> 4) & 0x0f00) |
                               ((s23 >> 16) & 0xf000);
        const uint32_t w = (ib / 2 < 2 ? s01 : s23) >> (16 * ((ib / 2) % 2));
        scale = H(dbits) * static_cast<float>(2 * ((w >> (6 * (ib % 2) + 3 * h)) & 7) + 1) * 0.125f;
        const int l1 = t / 2;  // l = 2h + l1
        const uint32_t qh = B(r, 32 + 2 * ib + h);
        const uint32_t idx = ((W(r, 4 * ib + 2 * h) >> (8 * l1)) & 0xff) | ((qh << (l1 ? 4 : 8)) & 0x700);
        const int delta = qh & (l1 ? 0x80 : 0x08) ? -1 : 1;
        const uint64_t g = kIq1SGrid[idx] >> (32 * (t % 2));
        int q[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) q[e] = 8 * static_cast<int8_t>((g >> (8 * e)) & 0xff) + delta;
        lo = bf16x2_of(q[0], q[1]);
        hi = bf16x2_of(q[2], q[3]);
    } else if (T == 3) {  // IQ2_XXS: d | per 32: 4 grid bytes, u32 of 4 x 7 sign bits + scale
        const uint32_t a1 = W(r, 6 + 8 * ib);
        const int l = 2 * h + t / 2;
        scale = H(W(r, 0)) * (0.5f + static_cast<float>(a1 >> 28)) * 0.25f;
        const uint32_t idx = (W(r, 2 + 8 * ib + 2 * h) >> (8 * (t / 2))) & 0xff;
        const uint2 g = grid_words(kBfIq2Xxs, idx, 4, t % 2);
        const uint32_t s = (kKsignsIq2xs[(a1 >> (7 * l)) & 127] >> (4 * (t % 2))) & 0xf;
        apply_signs(g, s, lo, hi);
    } else if (T == 4) {  // IQ2_XS: d | qs[32] u16 | scales[8]
        const uint32_t sc = B(r, 66 + ib);
        scale = H(W(r, 0)) * (0.5f + static_cast<float>(h ? sc >> 4 : sc & 0xf)) * 0.25f;
        const uint32_t w = (W(r, 2 + 8 * ib + 4 * h) >> (16 * (t / 2))) & 0xffff;
        const uint2 g = grid_words(kBfIq2Xs, w & 511, 4, t % 2);
        const uint32_t s = (kKsignsIq2xs[w >> 9] >> (4 * (t % 2))) & 0xf;
        apply_signs(g, s, lo, hi);
    } else if (T == 5) {  // IQ2_S: d | qs[32] | signs[32] | qh[8] | scales[8]
        const uint32_t sc = B(r, 74 + ib);
        scale = H(W(r, 0)) * (0.5f + static_cast<float>(h ? sc >> 4 : sc & 0xf)) * 0.25f;
        const int l = 2 * h + t / 2;
        const uint32_t idx = ((W(r, 2 + 4 * ib + 2 * h) >> (8 * (t / 2))) & 0xff) | ((B(r, 66 + ib) << (8 - 2 * l)) & 0x300);
        const uint2 g = grid_words(kBfIq2S, idx, 4, t % 2);
        const uint32_t s = (W(r, 34 + 4 * ib + 2 * h) >> (8 * (t / 2) + 4 * (t % 2))) & 0xf;
        apply_signs(g, s, lo, hi);
    } else if (T == 6) {  // IQ3_XXS: d | qs[64] | 8 u32 of 4 x 7 sign bits + scale
        const uint32_t aux = W(r, 66 + 4 * ib);
        const int l = 2 * h + t / 2;
        scale = H(W(r, 0)) * (0.5f + static_cast<float>(aux >> 28)) * 0.5f;
        const uint2 g = grid_words(kBfIq3Xxs, (W(r, 2 + 8 * ib + 4 * h) >> (8 * t)) & 0xff, 2, 0);
        const uint32_t s = (kKsignsIq2xs[(aux >> (7 * l)) & 127] >> (4 * (t % 2))) & 0xf;
        apply_signs(g, s, lo, hi);
    } else if (T == 7) {  // IQ3_S: d | qs[64] | qh[8] | signs[32] | scales[4]
        const uint32_t sc = B(r, 106 + ib / 2);
        scale = H(W(r, 0)) * static_cast<float>(1 + 2 * (ib % 2 ? sc >> 4 : sc & 0xf));
        const int l = 2 * h + t / 2;
        const uint32_t qh = B(r, 66 + ib);
        const uint32_t idx = ((W(r, 2 + 8 * ib + 4 * h) >> (8 * t)) & 0xff) | ((qh << ((t % 2 ? 7 : 8) - 2 * l)) & 256);
        const uint2 g = grid_words(kBfIq3S, idx, 2, 0);
        const uint32_t s = (W(r, 74 + 4 * ib + 2 * h) >> (8 * (t / 2) + 4 * (t % 2))) & 0xf;
        apply_signs(g, s, lo, hi);
    } else {  // IQ4_XS: d | scales_h u16 | scales_l[4] | qs[128]
        const uint32_t d0 = W(r, 0);
        const int ls = static_cast<int>((B(r, 4 + ib / 2) >> (4 * (ib % 2))) & 0xf) |
                       (static_cast<int>(((d0 >> 16) >> (2 * ib)) & 3) << 4);
        scale = H(d0) * static_cast<float>(ls - 32);
        nibble_words(kBfIq4Pair, W4(r, 8 + 16 * ib, t), h, lo, hi);
    }
}

__device__ __forceinline__ uint32_t bf16x2_rn(float a, float b) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ void mma_bf16(float c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Sums of each 16-value piece of the bf16-rounded activations: xs[token][k / 16].
__global__ void piece_sums(const float* __restrict__ x, int m, int k, float* __restrict__ xs) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int pieces = k / 16;
    if (i >= static_cast<long long>(m) * pieces) return;
    const float4* p = reinterpret_cast<const float4*>(x + (i / pieces) * k + (i % pieces) * 16);
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float4 v = p[j];
        s += __bfloat162float(__float2bfloat16_rn(v.x)) + __bfloat162float(__float2bfloat16_rn(v.y)) +
             __bfloat162float(__float2bfloat16_rn(v.z)) + __bfloat162float(__float2bfloat16_rn(v.w));
    }
    xs[i] = s;
}


template <int T, int MT, int P>
struct Pieces {
    // Applies pieces P .. 15 of the normalized blocks ra (rows g) and rb (rows g + 8) of super-block blk.
    static __device__ __forceinline__ void run(const uint32_t* ra, const uint32_t* rb, int t, int g, int blk,
                                               const float* __restrict__ x, int m, int k, int pieces,
                                               const float* __restrict__ xs, float (&acc)[MT][4]) {
        // A rows g (a0: k 2t, 2t+1 = values 4t, 4t+1; a2: k 2t+8, 2t+9 = values 4t+2, 4t+3), g + 8.
        uint32_t a0, a1, a2, a3;
        float sa, sb, ma, mb;
        piece<T, P>(ra, t, a0, a2, sa, ma);
        piece<T, P>(rb, t, a1, a3, sb, mb);
        const int pc = blk * 16 + P;
#pragma unroll
        for (int i = 0; i < MT; ++i) {
            if (i * 8 >= m) break;
            const int tok = i * 8 + g;
            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (tok < m) v = __ldg(reinterpret_cast<const float4*>(x + static_cast<long long>(tok) * k + pc * 16 + 4 * t));
            float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            mma_bf16(c, a0, a1, a2, a3, bf16x2_rn(v.x, v.y), bf16x2_rn(v.z, v.w));
            acc[i][0] += c[0] * sa;
            acc[i][1] += c[1] * sa;
            acc[i][2] += c[2] * sb;
            acc[i][3] += c[3] * sb;
            if (has_min(T)) {
                const int n0 = i * 8 + 2 * t;  // C columns (tokens) 2t, 2t + 1 of this tile
                const float x0 = n0 < m ? xs[static_cast<long long>(n0) * pieces + pc] : 0.0f;
                const float x1 = n0 + 1 < m ? xs[static_cast<long long>(n0 + 1) * pieces + pc] : 0.0f;
                acc[i][0] -= ma * x0;
                acc[i][1] -= ma * x1;
                acc[i][2] -= mb * x0;
                acc[i][3] -= mb * x1;
            }
        }
        Pieces<T, MT, P + 1>::run(ra, rb, t, g, blk, x, m, k, pieces, xs, acc);
    }
};
template <int T, int MT>
struct Pieces<T, MT, 16> {
    static __device__ __forceinline__ void run(const uint32_t*, const uint32_t*, int, int, int, const float* __restrict__,
                                               int, int, int, const float* __restrict__, float (&)[MT][4]) {}
};

constexpr int kWarps = 8;

// One warp: 16 weight rows x MT tiles of 8 tokens over every KW-th super-block of its K split. The
// KW warps of a row group meet in shared memory. Lane (g, t) = (lane / 4, lane % 4) owns rows g and
// g + 8 of the tile.
template <int T, int MT, int KW>
__global__ void __launch_bounds__(kWarps * 32)
ggml_skinny(const float* __restrict__ x, int m, const uint8_t* __restrict__ w, long long parent_rows, long long row_begin,
            int rows, int k, const float* __restrict__ xs, float* __restrict__ y, int blocks_per_split) {
    constexpr int BB = block_bytes(T), NR = block_words(T);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    const int kw = warp % KW;
    const int r0 = (blockIdx.x * (kWarps / KW) + warp / KW) * 16;
    const int nblk = k / 256, pieces = k / 16;
    const int b0 = blockIdx.y * blocks_per_split, b1 = min(nblk, b0 + blocks_per_split);
    y += static_cast<long long>(blockIdx.y) * m * rows;
    float acc[MT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f;
    if (r0 < rows) {
        const long long row_bytes = static_cast<long long>(nblk) * BB;
        const uint8_t* end = w + parent_rows * row_bytes;
        const uint8_t* wa = w + (row_begin + min(r0 + g, rows - 1)) * row_bytes;
        const uint8_t* wb = w + (row_begin + min(r0 + g + 8, rows - 1)) * row_bytes;
        for (int blk = b0 + kw; blk < b1; blk += KW) {
            uint32_t ra[NR], rb[NR];
            load_block<T>(wa + static_cast<long long>(blk) * BB, end, ra);
            load_block<T>(wb + static_cast<long long>(blk) * BB, end, rb);
            Pieces<T, MT, 0>::run(ra, rb, t, g, blk, x, m, k, pieces, xs, acc);
        }
    }
    if (KW > 1) {
        __shared__ float part[kWarps][MT * 4][32];
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) part[warp][i * 4 + j][lane] = acc[i][j];
        __syncthreads();
        if (kw != 0) return;
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float s = 0.0f;
                for (int z = 0; z < KW; ++z) s += part[warp + z][i * 4 + j][lane];
                acc[i][j] = s;
            }
    }
    if (r0 >= rows) return;
#pragma unroll
    for (int i = 0; i < MT; ++i) {
        const int n0 = i * 8 + 2 * t;
        const int rA = r0 + g, rB = r0 + g + 8;
        if (n0 < m) {
            if (rA < rows) y[static_cast<long long>(n0) * rows + rA] = acc[i][0];
            if (rB < rows) y[static_cast<long long>(n0) * rows + rB] = acc[i][2];
        }
        if (n0 + 1 < m) {
            if (rA < rows) y[static_cast<long long>(n0 + 1) * rows + rA] = acc[i][1];
            if (rB < rows) y[static_cast<long long>(n0 + 1) * rows + rB] = acc[i][3];
        }
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

VeloxScratch g_split;
VeloxScratch g_sums;

// About this many warps keep enough loads in flight on 170 SMs.
constexpr long long kTargetWarps = 8192;

template <int T, int MT>
int launch_mt(const float* x, int m, const uint8_t* w, long long parent_rows, long long row_begin, int rows, int k,
              const float* xs, float* y) {
    const long long groups = (rows + 15) / 16;
    const int nblk = k / 256;
    int kw = 1;
    while (kw < kWarps && groups * kw < kTargetWarps && nblk / (kw * 2) >= 2) kw *= 2;
    int splits = 1;
    while (groups * kw * splits < kTargetWarps && nblk / (splits * 2) >= 2 * kw) splits *= 2;
    const int per = (nblk + splits - 1) / splits;
    splits = (nblk + per - 1) / per;
    const int per_block = kWarps / kw;
    const dim3 grid(static_cast<unsigned>((groups + per_block - 1) / per_block), splits);
    float* out = y;
    const long long n = static_cast<long long>(m) * rows;
    if (splits > 1) {
        out = static_cast<float*>(g_split.get(static_cast<size_t>(splits) * n * sizeof(float)));
        if (out == nullptr) return static_cast<int>(cudaErrorMemoryAllocation);
    }
    switch (kw) {
        case 1: ggml_skinny<T, MT, 1><<<grid, kWarps * 32>>>(x, m, w, parent_rows, row_begin, rows, k, xs, out, per); break;
        case 2: ggml_skinny<T, MT, 2><<<grid, kWarps * 32>>>(x, m, w, parent_rows, row_begin, rows, k, xs, out, per); break;
        case 4: ggml_skinny<T, MT, 4><<<grid, kWarps * 32>>>(x, m, w, parent_rows, row_begin, rows, k, xs, out, per); break;
        default: ggml_skinny<T, MT, 8><<<grid, kWarps * 32>>>(x, m, w, parent_rows, row_begin, rows, k, xs, out, per); break;
    }
    if (splits > 1) sum_splits<<<velox_grid_1d(n, 256), 256>>>(out, splits, n, y);
    return velox_launch_status();
}

template <int T>
int launch_t(const float* x, int m, const uint8_t* w, long long parent_rows, long long row_begin, int rows, int k, float* y) {
    const float* xs = nullptr;
    if (has_min(T)) {
        const long long n = static_cast<long long>(m) * (k / 16);
        float* buf = static_cast<float*>(g_sums.get(n * sizeof(float)));
        if (buf == nullptr) return static_cast<int>(cudaErrorMemoryAllocation);
        piece_sums<<<velox_grid_1d(n, 256), 256>>>(x, m, k, buf);
        xs = buf;
    }
    if (m <= 8) return launch_mt<T, 1>(x, m, w, parent_rows, row_begin, rows, k, xs, y);
    if (m <= 16) return launch_mt<T, 2>(x, m, w, parent_rows, row_begin, rows, k, xs, y);
    if (m <= 32) return launch_mt<T, 4>(x, m, w, parent_rows, row_begin, rows, k, xs, y);
    return launch_mt<T, 8>(x, m, w, parent_rows, row_begin, rows, k, xs, y);
}

}  // namespace

// y[m, rows] = bf16(x)[m, k] . W[row_begin + r]^T for GGML type `type` (position in
// velox_core::ggml::GgmlType::ALL), m <= 64, k % 1024 == 0, x 16-byte aligned, weights 8-byte aligned.
VELOX_API int velox_5090_linear_ggml(const float* x, int m, const uint8_t* w, long long parent_rows, long long row_begin,
                                     int rows, long long k, int type, float* y) {
    if (m < 1 || m > 64 || k % 1024 != 0 || k > (1LL << 30) || reinterpret_cast<uintptr_t>(x) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(w) % 8 != 0)
        return static_cast<int>(cudaErrorNotSupported);
    const int ki = static_cast<int>(k);
    switch (type) {
        case 0: return launch_t<0>(x, m, w, parent_rows, row_begin, rows, ki, y);
        case 1: return launch_t<1>(x, m, w, parent_rows, row_begin, rows, ki, y);
        case 2: return launch_t<2>(x, m, w, parent_rows, row_begin, rows, ki, y);
        case 3: return launch_t<3>(x, m, w, parent_rows, row_begin, rows, ki, y);
        case 4: return launch_t<4>(x, m, w, parent_rows, row_begin, rows, ki, y);
        case 5: return launch_t<5>(x, m, w, parent_rows, row_begin, rows, ki, y);
        case 6: return launch_t<6>(x, m, w, parent_rows, row_begin, rows, ki, y);
        case 7: return launch_t<7>(x, m, w, parent_rows, row_begin, rows, ki, y);
        case 8: return launch_t<8>(x, m, w, parent_rows, row_begin, rows, ki, y);
        default: return static_cast<int>(cudaErrorNotSupported);
    }
}
