// Portable kernels for transformer/hybrid models: quantized-weight linear and embedding, sigmoid
// gating, causal depthwise convolution and the Gated DeltaNet recurrence. Simple and correct on
// every supported architecture; hardware branches replace them with tuned kernels.
#include <math.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include "ggml_tables.cuh"
#include "velox_common.cuh"

// Weight formats; must match velox_cuda::portable::FORMAT_*.
#define VELOX_FMT_F32 0
#define VELOX_FMT_BF16 1
#define VELOX_FMT_FP8_ROW_BF16 2
#define VELOX_FMT_NVFP4 3
#define VELOX_FMT_Q4 4
#define VELOX_FMT_Q5 5
#define VELOX_FMT_Q6 6
#define VELOX_FMT_Q8 7
#define VELOX_FMT_IQ3_XXS 8
// GGML block types: base + position in velox_core::ggml::GgmlType::ALL
// (q2_k, q4_k, iq1_m, iq2_xxs, iq2_xs, iq2_s, iq3_xxs, iq3_s, iq4_xs, q2_0, iq4_nl, q6_k, q8_0).
#define VELOX_FMT_GGML_BASE 16
#define VELOX_FMT_GGML_END 29

namespace {

struct WDesc {
    const uint8_t* base;
    int format;
    long long n;  // parent rows
    long long k;
};

__device__ __forceinline__ long long align256(long long x) { return (x + 255) / 256 * 256; }

__device__ __forceinline__ float bf16_to_f32(uint16_t b) { return __uint_as_float(static_cast<uint32_t>(b) << 16); }

__device__ __forceinline__ float e4m3fn_to_f32(uint8_t b) {
    const int e = (b >> 3) & 0xf, m = b & 7;
    float v;
    if (e == 0) v = ldexpf(static_cast<float>(m), -9);
    else if (e == 15 && m == 7) v = __int_as_float(0x7fc00000);
    else v = ldexpf(1.0f + m * 0.125f, e - 7);
    return (b & 0x80) ? -v : v;
}

__device__ __forceinline__ float e2m1_to_f32(uint8_t c) {
    const float t[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v = t[c & 7];
    return (c & 8) ? -v : v;
}

__device__ __forceinline__ long long nvfp4_scale_offset(long long n, long long g, long long k) {
    const long long k_tiles = k / 64;
    const long long row_tile = n / 128, row_inner = n % 128, scale_tile = g / 4, scale_lane = g % 4;
    return (row_tile * k_tiles + scale_tile) * 512 + (row_inner % 32) * 16 + (row_inner / 32) * 4 + scale_lane;
}

// llama.cpp iq3xxs_grid (ggml-common.h, MIT): four magnitudes per entry, little-endian bytes.
__constant__ uint32_t kIq3XxsGrid[256] = {
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

// iq3_xxs_planar_v1, as velox_core::format::iq3_xxs_at documents: a [n, k/256, 96] code plane, then
// the 256-byte aligned binary16 block scales [n, k/256].
__device__ float iq3_xxs_at(const WDesc& d, long long row, long long col) {
    const long long blocks = d.k / 256;
    const long long bi = row * blocks + col / 256;
    const uint8_t* code = d.base + bi * 96;
    const uint8_t* sp = d.base + align256(d.n * blocks * 96) + 2 * bi;
    const float scale = __half2float(__ushort_as_half(static_cast<unsigned short>(sp[0] | (sp[1] << 8))));
    const int s = static_cast<int>(col % 256) / 32, j = static_cast<int>(col % 32), e = j / 4;
    const uint32_t g = kIq3XxsGrid[code[16 * (e / 2) + 2 * s + (e & 1)]];
    const float magnitude = static_cast<float>((g >> (8 * (j & 3))) & 0xff);
    // Little-endian u64 lane words, read byte-wise: the view's base need not be 8-byte aligned.
    auto lane = [code](int q) {
        uint64_t w = 0;
        for (int b = 7; b >= 0; --b) w = (w << 8) | code[64 + 8 * q + b];
        return w;
    };
    const uint32_t sign7 = static_cast<uint32_t>(lane(j / 8) >> (7 * s)) & 127u;
    const uint32_t sign8 = sign7 | ((__popc(sign7) & 1u) << 7);
    const uint32_t ls = static_cast<uint32_t>(lane(s / 2) >> (56 + 4 * (s & 1))) & 15u;
    const float v = magnitude * scale * (0.25f + 0.5f * static_cast<float>(ls));
    return (sign8 >> (j & 7)) & 1u ? -v : v;
}

// GGML blocks (ggml_block_v1): a row is k / ggml_block_values(t) consecutive blocks of one type,
// as velox_core::ggml documents; decoding follows ggml's dequantize_row_* operation for operation.
// Block sizes are not multiples of 4, so multi-byte fields are assembled from bytes.
__host__ __device__ __forceinline__ int ggml_block_bytes(int t) {
    switch (t) {
        case 0: return 84;
        case 1: return 144;
        case 2: return 56;
        case 3: return 66;
        case 4: return 74;
        case 5: return 82;
        case 6: return 98;
        case 7: return 110;
        case 8: return 136;
        case 9: return 18;
        case 10: return 18;
        case 11: return 210;
        default: return 34;
    }
}

// Values per block: 256 (K-quant and i-quant super-blocks), 64 (Q2_0) or 32 (IQ4_NL, Q8_0).
__host__ __device__ __forceinline__ int ggml_block_values(int t) {
    switch (t) {
        case 9: return 64;
        case 10:
        case 12: return 32;
        default: return 256;
    }
}

__device__ __forceinline__ float ggml_f16(const uint8_t* p) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(p[0] | (p[1] << 8))));
}
__device__ __forceinline__ uint32_t ggml_u16(const uint8_t* p) { return p[0] | (static_cast<uint32_t>(p[1]) << 8); }
__device__ __forceinline__ uint32_t ggml_u32(const uint8_t* p) {
    return p[0] | (static_cast<uint32_t>(p[1]) << 8) | (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}
__device__ __forceinline__ uint32_t ggml_byte(uint64_t e, int j) { return static_cast<uint32_t>(e >> (8 * j)) & 0xffu; }
__device__ __forceinline__ float ggml_sign(uint32_t signs, int j) { return (signs >> j) & 1u ? -1.0f : 1.0f; }

// Value v (0 .. ggml_block_values(T) - 1) of one block x of type T (VELOX_FMT_GGML_BASE + T).
template <int T>
__device__ __forceinline__ float ggml_value(const uint8_t* __restrict__ x, int v) {
    const int ib = v / 32, j = v % 32, l = j / 8, jj = j % 8;
    if (T == 0) {  // Q2_K: scales[16] | qs[64] | d | dmin
        const int n = v / 128, w = v % 128, sh = 2 * (w / 32), h = (w % 32) / 16;
        const uint32_t sc = x[n * 8 + (w / 32) * 2 + h];
        const float dl = ggml_f16(x + 80) * static_cast<float>(sc & 0xf);
        const float ml = ggml_f16(x + 82) * static_cast<float>(sc >> 4);
        return dl * static_cast<float>((x[16 + 32 * n + (w % 32)] >> sh) & 3) - ml;
    } else if (T == 1) {  // Q4_K: d | dmin | scales[12] | qs[128]
        const int c = v / 64, hi = (v % 64) / 32, s = 2 * c + hi;
        const uint8_t* q = x + 4;
        uint32_t sc, m;
        if (s < 4) {
            sc = q[s] & 63, m = q[s + 4] & 63;
        } else {
            sc = (q[s + 4] & 0xf) | ((q[s - 4] >> 6) << 4);
            m = (q[s + 4] >> 4) | ((q[s] >> 6) << 4);
        }
        const uint32_t b = x[16 + 32 * c + v % 32];
        const float d1 = ggml_f16(x) * static_cast<float>(sc), m1 = ggml_f16(x + 2) * static_cast<float>(m);
        return d1 * static_cast<float>(hi ? b >> 4 : b & 0xf) - m1;
    } else if (T == 2) {  // IQ1_M: qs[32] | qh[16] | scales[8]
        const uint32_t s0 = ggml_u16(x + 48), s1 = ggml_u16(x + 50), s2 = ggml_u16(x + 52), s3 = ggml_u16(x + 54);
        const uint32_t dbits = (s0 >> 12) | ((s1 >> 8) & 0x00f0) | ((s2 >> 4) & 0x0f00) | (s3 & 0xf000);
        const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(dbits)));
        const uint32_t w = ggml_u16(x + 48 + 2 * (ib / 2));
        const float dl = d * static_cast<float>(2 * ((w >> (6 * (ib % 2) + (l < 2 ? 0 : 3))) & 7) + 1);
        const uint32_t qh = x[32 + 2 * ib + l / 2];
        const uint32_t idx = x[4 * ib + l] | ((qh << (l % 2 ? 4 : 8)) & 0x700);
        const float delta = qh & (l % 2 ? 0x80 : 0x08) ? -0.125f : 0.125f;
        const float g = static_cast<float>(static_cast<int8_t>(ggml_byte(kIq1SGrid[idx], jj)));
        return dl * (g + delta);
    } else if (T == 3) {  // IQ2_XXS: d | per 32: 4 grid bytes, u32 of 4 x 7 sign bits + scale
        const uint32_t a1 = ggml_u32(x + 6 + 8 * ib);
        const float db = ggml_f16(x) * (0.5f + static_cast<float>(a1 >> 28)) * 0.25f;
        const uint64_t grid = kIq2XxsGrid[x[2 + 8 * ib + l]];
        return db * static_cast<float>(ggml_byte(grid, jj)) * ggml_sign(kKsignsIq2xs[(a1 >> (7 * l)) & 127], jj);
    } else if (T == 4) {  // IQ2_XS: d | qs[32] u16 | scales[8]
        const uint32_t s = x[66 + ib];
        const float db = ggml_f16(x) * (0.5f + static_cast<float>(l < 2 ? s & 0xf : s >> 4)) * 0.25f;
        const uint32_t q = ggml_u16(x + 2 + 2 * (4 * ib + l));
        return db * static_cast<float>(ggml_byte(kIq2XsGrid[q & 511], jj)) * ggml_sign(kKsignsIq2xs[q >> 9], jj);
    } else if (T == 5) {  // IQ2_S: d | qs[32] | signs[32] | qh[8] | scales[8]
        const uint32_t s = x[74 + ib];
        const float db = ggml_f16(x) * (0.5f + static_cast<float>(l < 2 ? s & 0xf : s >> 4)) * 0.25f;
        const uint32_t idx = x[2 + 4 * ib + l] | ((static_cast<uint32_t>(x[66 + ib]) << (8 - 2 * l)) & 0x300);
        return db * static_cast<float>(ggml_byte(kIq2SGrid[idx], jj)) * ggml_sign(x[34 + 4 * ib + l], jj);
    } else if (T == 6) {  // IQ3_XXS: d | qs[64] | 8 u32 of 4 x 7 sign bits + scale
        const uint32_t aux = ggml_u32(x + 66 + 4 * ib);
        const float db = ggml_f16(x) * (0.5f + static_cast<float>(aux >> 28)) * 0.5f;
        const uint32_t grid = kIq3XxsGrid[x[2 + 8 * ib + 2 * l + jj / 4]];
        return db * static_cast<float>(ggml_byte(grid, jj % 4)) * ggml_sign(kKsignsIq2xs[(aux >> (7 * l)) & 127], jj);
    } else if (T == 7) {  // IQ3_S: d | qs[64] | qh[8] | signs[32] | scales[4]
        const uint32_t s = x[106 + ib / 2];
        const float db = ggml_f16(x) * static_cast<float>(1 + 2 * (ib % 2 ? s >> 4 : s & 0xf));
        const uint32_t qh = x[66 + ib];
        const uint32_t idx = jj < 4 ? x[2 + 8 * ib + 2 * l] | ((qh << (8 - 2 * l)) & 256)
                                    : x[3 + 8 * ib + 2 * l] | ((qh << (7 - 2 * l)) & 256);
        return db * static_cast<float>(ggml_byte(kIq3SGrid[idx], jj % 4)) * ggml_sign(x[74 + 4 * ib + l], jj);
    } else if (T == 8) {  // IQ4_XS: d | scales_h u16 | scales_l[4] | qs[128]
        const int ls = static_cast<int>((x[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) |
                       (static_cast<int>((ggml_u16(x + 2) >> (2 * ib)) & 3) << 4);
        const float dl = ggml_f16(x) * static_cast<float>(ls - 32);
        const uint32_t b = x[8 + 16 * ib + j % 16];
        return dl * static_cast<float>(kValuesIq4nl[j < 16 ? b & 0xf : b >> 4]);
    } else if (T == 9) {  // Q2_0 (64 values): d | qs[16], codes 0..3 meaning -1, 0, +1, +2
        const int q = (x[2 + v / 4] >> (2 * (v % 4))) & 3;
        return static_cast<float>(q - 1) * ggml_f16(x);
    } else if (T == 10) {  // IQ4_NL (32 values): d | qs[16]
        const uint32_t b = x[2 + v % 16];
        return ggml_f16(x) * static_cast<float>(kValuesIq4nl[v < 16 ? b & 0xf : b >> 4]);
    } else if (T == 11) {  // Q6_K: ql[128] | qh[64] | scales[16] (int8) | d
        const int n = v / 128, quarter = (v % 128) / 32, l2 = v % 32;
        const uint32_t lo = x[64 * n + 32 * (quarter & 1) + l2];
        const uint32_t hi = (x[128 + 32 * n + l2] >> (2 * quarter)) & 3;
        const int q = static_cast<int>((quarter < 2 ? lo & 0xf : lo >> 4) | (hi << 4)) - 32;
        const float sc = static_cast<float>(static_cast<int8_t>(x[192 + 8 * n + l2 / 16 + 2 * quarter]));
        return ggml_f16(x + 208) * sc * static_cast<float>(q);
    } else {  // Q8_0 (32 values): d | qs[32] (int8)
        return static_cast<float>(static_cast<int8_t>(x[2 + v])) * ggml_f16(x);
    }
}

__device__ float ggml_at(const WDesc& d, long long row, long long col) {
    const int t = d.format - VELOX_FMT_GGML_BASE;
    const long long bb = ggml_block_bytes(t), bv = ggml_block_values(t);
    const uint8_t* x = d.base + (row * (d.k / bv) + col / bv) * bb;
    const int v = static_cast<int>(col % bv);
    switch (t) {
        case 0: return ggml_value<0>(x, v);
        case 1: return ggml_value<1>(x, v);
        case 2: return ggml_value<2>(x, v);
        case 3: return ggml_value<3>(x, v);
        case 4: return ggml_value<4>(x, v);
        case 5: return ggml_value<5>(x, v);
        case 6: return ggml_value<6>(x, v);
        case 7: return ggml_value<7>(x, v);
        case 8: return ggml_value<8>(x, v);
        case 9: return ggml_value<9>(x, v);
        case 10: return ggml_value<10>(x, v);
        case 11: return ggml_value<11>(x, v);
        default: return ggml_value<12>(x, v);
    }
}

// Decoded W[row, col] of the parent matrix.
__device__ float weight_at(const WDesc& d, long long row, long long col) {
    if (d.format >= VELOX_FMT_GGML_BASE) return ggml_at(d, row, col);
    switch (d.format) {
        case VELOX_FMT_F32:
            return reinterpret_cast<const float*>(d.base)[row * d.k + col];
        case VELOX_FMT_BF16:
            return bf16_to_f32(reinterpret_cast<const uint16_t*>(d.base)[row * d.k + col]);
        case VELOX_FMT_FP8_ROW_BF16: {
            const uint8_t* s = d.base + align256(d.n * d.k) + 2 * row;
            return e4m3fn_to_f32(d.base[row * d.k + col]) * bf16_to_f32(static_cast<uint16_t>(s[0] | (s[1] << 8)));
        }
        case VELOX_FMT_Q4:
        case VELOX_FMT_Q5:
        case VELOX_FMT_Q6:
        case VELOX_FMT_Q8: {  // row_split_k128_v1
            const int bits = d.format == VELOX_FMT_Q4 ? 4 : d.format == VELOX_FMT_Q5 ? 5 : d.format == VELOX_FMT_Q6 ? 6 : 8;
            const int group = bits == 8 ? 32 : 64;
            const int high = bits == 5 ? 8 : bits == 6 ? 16 : 0;
            const long long gpr = (d.k + 127) / 128 * 128 / group;
            const long long groups = d.n * gpr;
            const long long high_off = align256(groups * 32);
            const long long scale_off = high_off + align256(groups * high);
            const long long gi = row * gpr + col / group;
            const int lane = static_cast<int>(col % group);
            uint32_t u;
            if (bits == 8) {
                u = d.base[gi * 32 + lane];
            } else {
                const uint8_t byte = d.base[gi * 32 + lane / 2];
                u = (lane & 1) ? (byte >> 4) : (byte & 0xf);
            }
            const int hb = high > 0 ? bits - 4 : 0;  // Q5: 1, Q6: 2; Q4/Q8 have no high plane
            for (int j = 0; j < hb; ++j) {
                const int t = lane * hb + j;
                u |= static_cast<uint32_t>((d.base[high_off + gi * high + t / 8] >> (t % 8)) & 1) << (4 + j);
            }
            const int q = u >= (1u << (bits - 1)) ? static_cast<int>(u) - (1 << bits) : static_cast<int>(u);
            const uint8_t* s = d.base + scale_off + 2 * gi;
            return __half2float(__ushort_as_half(static_cast<unsigned short>(s[0] | (s[1] << 8)))) * static_cast<float>(q);
        }
        case VELOX_FMT_IQ3_XXS:
            return iq3_xxs_at(d, row, col);
        default: {  // NVFP4
            const uint8_t byte = d.base[(row * d.k + col) / 2];
            const uint8_t code = (col & 1) ? (byte >> 4) : (byte & 0xf);
            const long long scale_plane = align256(d.n * d.k / 2);
            const float scale = e4m3fn_to_f32(d.base[scale_plane + nvfp4_scale_offset(row, col / 16, d.k)]);
            const float divisor = *reinterpret_cast<const float*>(d.base + scale_plane + d.n * d.k / 16);
            return e2m1_to_f32(code) * scale / divisor;
        }
    }
}

// y[m, rows] = x[m, k] . W[row_begin + r, :]^T, one warp per output element.
__global__ void linear_kernel(const float* __restrict__ x, WDesc w, long long row_begin, int rows, int m,
                              float* __restrict__ y) {
    const int r = blockIdx.x * (blockDim.x / VELOX_WARP) + threadIdx.x / VELOX_WARP;
    const int lane = threadIdx.x % VELOX_WARP;
    if (r >= rows) return;
    for (int i = blockIdx.y; i < m; i += gridDim.y) {
        const float* xr = x + static_cast<size_t>(i) * w.k;
        float acc = 0.0f;
        for (long long c = lane; c < w.k; c += VELOX_WARP) acc += xr[c] * weight_at(w, row_begin + r, c);
        acc = velox_warp_sum(acc);
        if (lane == 0) y[static_cast<size_t>(i) * rows + r] = acc;
    }
}

// Tiled y[m, rows] = x . W^T for larger m: each 64x64 output tile decodes its weight tile once
// into shared memory per K step, so weights are read m/64 times instead of m times.
constexpr int kTile = 64, kTileK = 16;

__global__ void __launch_bounds__(256)
linear_tiled_kernel(const float* __restrict__ x, WDesc w, long long row_begin, int rows, int m, float* __restrict__ y,
                    long long k_split) {
    // Split-K: block z covers [z * k_split, (z + 1) * k_split) and writes its own [m, rows] slice.
    const long long k_begin = blockIdx.z * k_split;
    const long long k_end = k_begin + k_split < w.k ? k_begin + k_split : w.k;
    y += static_cast<size_t>(blockIdx.z) * m * rows;
    __shared__ float Xs[kTileK][kTile + 4];
    __shared__ float Ws[kTileK][kTile + 4];
    const int t = threadIdx.x;
    const int m0 = blockIdx.y * kTile, n0 = blockIdx.x * kTile;
    const int lr = t / 4, lk = (t % 4) * 4;
    const int ty = t / 16, tx = t % 16;
    float acc[4][4] = {};
    for (long long k0 = k_begin; k0 < k_end; k0 += kTileK) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const long long kk = k0 + lk + i;
            const int xm = m0 + lr, wn = n0 + lr;
            Xs[lk + i][lr] = (xm < m && kk < k_end) ? x[static_cast<size_t>(xm) * w.k + kk] : 0.0f;
            Ws[lk + i][lr] = (wn < rows && kk < k_end) ? weight_at(w, row_begin + wn, kk) : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < kTileK; ++kk) {
            float a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                a[i] = Xs[kk][ty * 4 + i];
                b[i] = Ws[kk][tx * 4 + i];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = m0 + ty * 4 + i;
        if (r >= m) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = n0 + tx * 4 + j;
            if (c < rows) y[static_cast<size_t>(r) * rows + c] = acc[i][j];
        }
    }
}

// Row-split (q4/q5/q6/q8) rows: a block stages its row's three byte ranges (codes, high plane,
// scales; each contiguous) in shared memory with 16-byte loads, then decodes from there. The
// table is often host-mapped, where the cost is per PCIe transaction: one round of wide loads per
// row instead of several narrow ones per element.
__global__ void embed_rs_kernel(const uint32_t* __restrict__ ids, WDesc d, long long row_begin, float* __restrict__ out) {
    extern __shared__ __align__(16) uint8_t em_smem[];
    const int bits = d.format == VELOX_FMT_Q4 ? 4 : d.format == VELOX_FMT_Q5 ? 5 : d.format == VELOX_FMT_Q6 ? 6 : 8;
    const int group = bits == 8 ? 32 : 64;
    const int hb = bits == 5 ? 1 : bits == 6 ? 2 : 0;  // high-plane bits per element
    const int high = hb * group / 8;                   // high-plane bytes per group
    const long long gpr = (d.k + 127) / 128 * 128 / group;
    const long long groups = d.n * gpr;
    const long long high_off = align256(groups * 32), scale_off = high_off + align256(groups * high);
    const long long row = row_begin + ids[blockIdx.x];
    if (row < 0 || row >= d.n) {  // ids written on the device are not checked by the host
        for (long long c = threadIdx.x; c < d.k; c += blockDim.x) out[blockIdx.x * d.k + c] = 0.0f;
        return;
    }
    const int rg = static_cast<int>((d.k + group - 1) / group);  // groups of this row
    uint8_t* sc = em_smem;                                           // [rg * 32]
    uint8_t* sh = em_smem + (rg * 32 + 15) / 16 * 16;                // [rg * high]
    uint8_t* ss = sh + (rg * high + 15) / 16 * 16;                   // [rg * 2]
    const uint8_t* codes = d.base + row * gpr * 32;
    const uint8_t* hp = d.base + high_off + row * gpr * high;
    const uint8_t* scp = d.base + scale_off + row * gpr * 2;
    for (int i = threadIdx.x; i < rg * 2; i += blockDim.x)
        reinterpret_cast<uint4*>(sc)[i] = reinterpret_cast<const uint4*>(codes)[i];
    for (int i = threadIdx.x; i < rg * high / 8; i += blockDim.x)
        reinterpret_cast<uint2*>(sh)[i] = reinterpret_cast<const uint2*>(hp)[i];
    for (int i = threadIdx.x; i < rg; i += blockDim.x)
        reinterpret_cast<uint16_t*>(ss)[i] = reinterpret_cast<const uint16_t*>(scp)[i];
    __syncthreads();
    float* o = out + static_cast<long long>(blockIdx.x) * d.k;
    for (int e = threadIdx.x; e < d.k; e += blockDim.x) {
        const int g = e / group, le = e % group;
        uint32_t u;
        if (bits == 8) {
            u = sc[g * 32 + le];
        } else {
            const uint8_t byte = sc[g * 32 + le / 2];
            u = (le & 1) ? (byte >> 4) : (byte & 0xf);
        }
        for (int b = 0; b < hb; ++b) {
            const int t = le * hb + b;
            u |= static_cast<uint32_t>((sh[g * high + t / 8] >> (t % 8)) & 1) << (4 + b);
        }
        const int q = u >= (1u << (bits - 1)) ? static_cast<int>(u) - (1 << bits) : static_cast<int>(u);
        const float scale = __half2float(__ushort_as_half(static_cast<unsigned short>(ss[2 * g] | (ss[2 * g + 1] << 8))));
        o[e] = scale * static_cast<float>(q);
    }
}

// Block (token, column slice): a few tokens still spread over the whole GPU.
__global__ void embed_kernel(const uint32_t* __restrict__ ids, WDesc w, long long row_begin, float* __restrict__ out) {
    const long long row = row_begin + ids[blockIdx.x];
    const bool valid = row >= 0 && row < w.n;  // ids written on the device are not checked by the host
    for (long long c = static_cast<long long>(blockIdx.y) * blockDim.x + threadIdx.x; c < w.k;
         c += static_cast<long long>(gridDim.y) * blockDim.x)
        out[blockIdx.x * w.k + c] = valid ? weight_at(w, row, c) : 0.0f;
}

// xb: null, or the result's bf16 twin.
__global__ void sigmoid_mul_kernel(float* __restrict__ x, const float* __restrict__ gate, long long len,
                                   unsigned short* __restrict__ xb) {
    for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < len;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const float v = x[i] / (1.0f + expf(-gate[i]));
        x[i] = v;
        if (xb != nullptr) xb[i] = velox_bf16_bits(v);
    }
}

constexpr int kMaxConvWidth = 8;

// One thread per (token, channel): tap j reads the input at time t - (width - 1) + j, from x when
// that is >= 0 and otherwise from the rolling state (oldest first). The state is advanced by
// conv1d_state_kernel afterwards, so every read here sees the state from before the call.
__global__ void conv1d_silu_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                   const float* __restrict__ state, float* __restrict__ out, int tokens, int channels,
                                   int width, const long long* __restrict__ pp) {
    if (pp != nullptr) state = reinterpret_cast<const float*>(pp[0]);  // staged [state_in, state_out]
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<long long>(tokens) * channels) return;
    const int t = static_cast<int>(i / channels), c = static_cast<int>(i % channels);
    float acc = 0.0f;
    for (int j = 0; j < width; ++j) {
        const int s = t - (width - 1) + j;
        const float in = s >= 0 ? x[static_cast<size_t>(s) * channels + c] : state[static_cast<size_t>(t + j) * channels + c];
        acc += w[static_cast<size_t>(j) * channels + c] * in;
    }
    out[i] = acc / (1.0f + expf(-acc));
}

// Few tokens (decode, verify): one thread per channel runs the window over its inputs and advances
// the state itself (each thread reads its channel's state before writing it, so state may alias
// state_in).
constexpr int kConvSmallTokens = 32;
__global__ void conv1d_silu_small_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                         const float* state_in, float* state, float* __restrict__ out, int tokens,
                                         int channels, int width, const long long* __restrict__ pp) {
    if (pp != nullptr) {  // staged by a graph step: [state_in, state_out]
        state_in = reinterpret_cast<const float*>(pp[0]);
        state = reinterpret_cast<float*>(pp[1]);
    }
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float win[kMaxConvWidth], wt[kMaxConvWidth];
#pragma unroll
    for (int j = 0; j < kMaxConvWidth; ++j) {
        wt[j] = j < width ? w[static_cast<size_t>(j) * channels + c] : 0.0f;
        win[j] = j < width - 1 ? state_in[static_cast<size_t>(j) * channels + c] : 0.0f;
    }
    for (int t = 0; t < tokens; ++t) {
        const float in = x[static_cast<size_t>(t) * channels + c];
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < kMaxConvWidth - 1; ++j)
            if (j < width - 1) acc += wt[j] * win[j];
#pragma unroll
        for (int j = 0; j < kMaxConvWidth; ++j)
            if (j == width - 1) acc += wt[j] * in;
        out[static_cast<size_t>(t) * channels + c] = acc / (1.0f + expf(-acc));
#pragma unroll
        for (int j = 0; j < kMaxConvWidth - 2; ++j)
            if (j < width - 2) win[j] = win[j + 1];
#pragma unroll
        for (int j = 0; j < kMaxConvWidth - 1; ++j)
            if (j == width - 2) win[j] = in;
    }
#pragma unroll
    for (int j = 0; j < kMaxConvWidth - 1; ++j)
        if (j < width - 1) state[static_cast<size_t>(j) * channels + c] = win[j];
}

// state[j] <- input at time tokens - (width - 1) + j. Ascending j only reads indices >= j + tokens
// of the old state, which are not yet overwritten.
__global__ void conv1d_state_kernel(const float* __restrict__ x, const float* state_in, float* state, int tokens,
                                    int channels, int width, const long long* __restrict__ pp) {
    if (pp != nullptr) {  // staged by a graph step: [state_in, state_out]
        state_in = reinterpret_cast<const float*>(pp[0]);
        state = reinterpret_cast<float*>(pp[1]);
    }
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    for (int j = 0; j < width - 1; ++j) {
        const int s = tokens - (width - 1) + j;
        state[static_cast<size_t>(j) * channels + c] =
            s >= 0 ? x[static_cast<size_t>(s) * channels + c] : state_in[static_cast<size_t>(j + tokens) * channels + c];
    }
}

// Gated DeltaNet: state rows are independent, so one warp owns one row S[h][i][:] (lanes split
// k_dim, PER elements each, kept in registers for the whole call). `out` may be null (state only).
template <int PER>
__device__ __forceinline__ void gdn_rows(const float* __restrict__ qkv, const float* __restrict__ ab,
                                         const float* __restrict__ a_log, const float* __restrict__ dt_bias,
                                         const float* state_in, float* state, float* __restrict__ out, int tokens,
                                         int k_heads, int v_heads, int k_dim, int v_dim, int h, int i, int lane) {
    const int kh = h / (v_heads / k_heads);
    const int width = 2 * k_heads * k_dim + v_heads * v_dim;
    const float inv_sqrt_dk = rsqrtf(static_cast<float>(k_dim));
    const float neg_a = -expf(a_log[h]), bias = dt_bias[h];
    const size_t row_off = (static_cast<size_t>(h) * v_dim + i) * k_dim;
    // Written at the end (unless state is null: outputs only); read from state_in (may alias).
    float* row_state = state != nullptr ? state + row_off : nullptr;
    float s[PER];
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int d = lane + p * VELOX_WARP;
        s[p] = d < k_dim ? state_in[row_off + d] : 0.0f;
    }
    for (int t = 0; t < tokens; ++t) {
        const float* row = qkv + static_cast<size_t>(t) * width;
        float q[PER], k[PER], qq = 0.0f, kk = 0.0f;
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            const int d = lane + p * VELOX_WARP;
            q[p] = d < k_dim ? row[kh * k_dim + d] : 0.0f;
            k[p] = d < k_dim ? row[(k_heads + kh) * k_dim + d] : 0.0f;
            qq += q[p] * q[p];
            kk += k[p] * k[p];
        }
        const float qn = rsqrtf(velox_warp_sum(qq) + 1e-6f), kn = rsqrtf(velox_warp_sum(kk) + 1e-6f);
        const float a = ab[static_cast<size_t>(t) * 2 * v_heads + h] + bias;
        const float b = ab[static_cast<size_t>(t) * 2 * v_heads + v_heads + h];
        const float alpha = expf(neg_a * (a > 20.0f ? a : log1pf(expf(a))));
        const float beta = 1.0f / (1.0f + expf(-b));
        float sk = 0.0f;
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            s[p] *= alpha;
            sk += s[p] * k[p] * kn;
        }
        const float delta = beta * (row[2 * k_heads * k_dim + h * v_dim + i] - velox_warp_sum(sk));
        float o = 0.0f;
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            s[p] += delta * k[p] * kn;
            o += s[p] * q[p] * qn;
        }
        if (out != nullptr) {
            o = velox_warp_sum(o);
            if (lane == 0) out[(static_cast<size_t>(t) * v_heads + h) * v_dim + i] = o * inv_sqrt_dk;
        }
    }
    if (row_state == nullptr) return;
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int d = lane + p * VELOX_WARP;
        if (d < k_dim) row_state[d] = s[p];
    }
}

// Windows up to this many tokens (decode, verify, replay) share their per-token head values
// through shared memory (`GdnHead`); longer ones recompute them per row warp (`gdn_rows`).
constexpr int kGdnRowsMaxT = 16;
constexpr int kGdnRowsWarps = 8;  // state rows per block of the per-row kernels

// One head's per-token values every row of it uses: q and k L2-normalized, alpha and beta.
template <int PER>
struct GdnHead {
    float q[kGdnRowsMaxT][PER * VELOX_WARP];
    float k[kGdnRowsMaxT][PER * VELOX_WARP];
    float alpha[kGdnRowsMaxT], beta[kGdnRowsMaxT];
};

// Fills `hd` for head h, one warp per token (the values `gdn_rows` computes in every row warp).
template <int PER>
__device__ __forceinline__ void gdn_head_prep(const float* __restrict__ qkv, const float* __restrict__ ab,
                                              const float* __restrict__ a_log, const float* __restrict__ dt_bias,
                                              int tokens, int k_heads, int v_heads, int k_dim, int v_dim, int h,
                                              GdnHead<PER>& hd) {
    const int warps = blockDim.x / VELOX_WARP, warp = threadIdx.x / VELOX_WARP, lane = threadIdx.x % VELOX_WARP;
    const int kh = h / (v_heads / k_heads);
    const int width = 2 * k_heads * k_dim + v_heads * v_dim;
    const float neg_a = -expf(a_log[h]), bias = dt_bias[h];
    for (int t = warp; t < tokens; t += warps) {
        const float* row = qkv + static_cast<size_t>(t) * width;
        float q[PER], k[PER], qq = 0.0f, kk = 0.0f;
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            const int d = lane + p * VELOX_WARP;
            q[p] = d < k_dim ? row[kh * k_dim + d] : 0.0f;
            k[p] = d < k_dim ? row[(k_heads + kh) * k_dim + d] : 0.0f;
            qq += q[p] * q[p];
            kk += k[p] * k[p];
        }
        const float qn = rsqrtf(velox_warp_sum(qq) + 1e-6f), kn = rsqrtf(velox_warp_sum(kk) + 1e-6f);
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            hd.q[t][lane + p * VELOX_WARP] = q[p] * qn;
            hd.k[t][lane + p * VELOX_WARP] = k[p] * kn;
        }
        if (lane == 0) {
            const float a = ab[static_cast<size_t>(t) * 2 * v_heads + h] + bias;
            const float b = ab[static_cast<size_t>(t) * 2 * v_heads + v_heads + h];
            hd.alpha[t] = expf(neg_a * (a > 20.0f ? a : log1pf(expf(a))));
            hd.beta[t] = 1.0f / (1.0f + expf(-b));
        }
    }
}

// Lanes per state row of `gdn_rows_shared`: a warp updates 8 rows, and a row's dot products
// reduce over 4 lanes (2 shuffles) instead of a whole warp (5).
constexpr int kGdnRowLanes = 4;
constexpr int kGdnRowsPerBlock = kGdnRowsWarps * (VELOX_WARP / kGdnRowLanes);

// `gdn_rows` over a window of at most kGdnRowsMaxT tokens whose head values are in `hd`, for row
// i with kGdnRowLanes lanes; lane j of the row holds elements j, j + 4, ... (conflict-free
// shared reads: the row's lanes hit distinct banks, the warp's 8 rows broadcast).
template <int PER>
__device__ __forceinline__ void gdn_rows_shared(const float* __restrict__ qkv, const float* state_in, float* state,
                                                float* __restrict__ out, int tokens, int k_heads, int v_heads,
                                                int k_dim, int v_dim, int h, int i, int j,
                                                const GdnHead<PER>& hd) {
    constexpr int V = PER * VELOX_WARP / kGdnRowLanes;  // values per lane
    const int width = 2 * k_heads * k_dim + v_heads * v_dim;
    const float inv_sqrt_dk = rsqrtf(static_cast<float>(k_dim));
    const bool live = i < v_dim;  // dead lanes still join the shuffles
    const size_t row_off = (static_cast<size_t>(h) * v_dim + (live ? i : 0)) * k_dim;
    float s[V];
#pragma unroll
    for (int p = 0; p < V; ++p) {
        const int d = j + p * kGdnRowLanes;
        s[p] = live && d < k_dim ? state_in[row_off + d] : 0.0f;
    }
    auto row_sum = [](float x) {
        x += __shfl_xor_sync(0xffffffffu, x, 1);
        return x + __shfl_xor_sync(0xffffffffu, x, 2);
    };
    for (int t = 0; t < tokens; ++t) {
        const float v = live ? qkv[static_cast<size_t>(t) * width + 2 * k_heads * k_dim + h * v_dim + i] : 0.0f;
        const float alpha = hd.alpha[t];
        float sk = 0.0f;
#pragma unroll
        for (int p = 0; p < V; ++p) {
            s[p] *= alpha;
            sk += s[p] * hd.k[t][j + p * kGdnRowLanes];
        }
        const float delta = hd.beta[t] * (v - row_sum(sk));
        float o = 0.0f;
#pragma unroll
        for (int p = 0; p < V; ++p) {
            s[p] += delta * hd.k[t][j + p * kGdnRowLanes];
            o += s[p] * hd.q[t][j + p * kGdnRowLanes];
        }
        if (out != nullptr) {
            o = row_sum(o);
            if (live && j == 0) out[(static_cast<size_t>(t) * v_heads + h) * v_dim + i] = o * inv_sqrt_dk;
        }
    }
    // Written at the end (unless state is null: outputs only); read from state_in (may alias).
    if (state == nullptr || !live) return;
#pragma unroll
    for (int p = 0; p < V; ++p) {
        const int d = j + p * kGdnRowLanes;
        if (d < k_dim) state[row_off + d] = s[p];
    }
}

// Grid: heads x row groups.
template <int PER>
__global__ void gated_delta_net_rows_kernel(const float* __restrict__ qkv, const float* __restrict__ ab,
                                            const float* __restrict__ a_log, const float* __restrict__ dt_bias,
                                            const float* state_in, float* state, float* __restrict__ out,
                                            int tokens, int k_heads, int v_heads, int k_dim, int v_dim,
                                            const long long* __restrict__ pp) {
    if (pp != nullptr) {  // staged by a graph step: [state_in, state_out]
        state_in = reinterpret_cast<const float*>(pp[0]);
        state = reinterpret_cast<float*>(pp[1]);
    }
    const int warps = blockDim.x / VELOX_WARP;
    const int h = blockIdx.x, i = blockIdx.y * warps + threadIdx.x / VELOX_WARP, lane = threadIdx.x % VELOX_WARP;
    if (tokens <= kGdnRowsMaxT) {
        // Grid y counts blocks of kGdnRowsPerBlock rows here (see the launch).
        __shared__ GdnHead<PER> hd;
        gdn_head_prep<PER>(qkv, ab, a_log, dt_bias, tokens, k_heads, v_heads, k_dim, v_dim, h, hd);
        __syncthreads();
        const int row = blockIdx.y * kGdnRowsPerBlock + threadIdx.x / kGdnRowLanes;
        gdn_rows_shared<PER>(qkv, state_in, state, out, tokens, k_heads, v_heads, k_dim, v_dim, h, row,
                             threadIdx.x % kGdnRowLanes, hd);
        return;
    }
    if (i >= v_dim) return;
    gdn_rows<PER>(qkv, ab, a_log, dt_bias, state_in, state, out, tokens, k_heads, v_heads, k_dim, v_dim, h, i, lane);
}

// gdn_replay: per-layer buffers, passed by value as a kernel parameter (3.6 KB; graph launches
// capture it with the node).
constexpr int kGdnReplayMaxLayers = 64;
struct GdnReplayTable {
    const float* qkv[kGdnReplayMaxLayers];
    const float* conv_out[kGdnReplayMaxLayers];
    const float* ab[kGdnReplayMaxLayers];
    const float* a_log[kGdnReplayMaxLayers];
    const float* dt_bias[kGdnReplayMaxLayers];
    float* conv_state[kGdnReplayMaxLayers];
    float* ssm[kGdnReplayMaxLayers];
};

// Grid: heads x row groups x layers; the recurrent state advances in place, no output.
template <int PER>
__global__ void gdn_replay_rows_kernel(const __grid_constant__ GdnReplayTable t, int tokens, int k_heads, int v_heads,
                                       int k_dim, int v_dim) {
    const int warps = blockDim.x / VELOX_WARP, l = blockIdx.z;
    const int h = blockIdx.x, i = blockIdx.y * warps + threadIdx.x / VELOX_WARP, lane = threadIdx.x % VELOX_WARP;
    if (tokens <= kGdnRowsMaxT) {
        __shared__ GdnHead<PER> hd;
        gdn_head_prep<PER>(t.conv_out[l], t.ab[l], t.a_log[l], t.dt_bias[l], tokens, k_heads, v_heads, k_dim, v_dim, h,
                           hd);
        __syncthreads();
        const int row = blockIdx.y * kGdnRowsPerBlock + threadIdx.x / kGdnRowLanes;
        gdn_rows_shared<PER>(t.conv_out[l], t.ssm[l], t.ssm[l], nullptr, tokens, k_heads, v_heads, k_dim, v_dim, h,
                             row, threadIdx.x % kGdnRowLanes, hd);
        return;
    }
    if (i >= v_dim) return;
    gdn_rows<PER>(t.conv_out[l], t.ab[l], t.a_log[l], t.dt_bias[l], t.ssm[l], t.ssm[l], nullptr, tokens,
                  k_heads, v_heads, k_dim, v_dim, h, i, lane);
}

// Grid: channel blocks x layers. New conv state row j is row tokens + j of [old state; inputs];
// ascending j reads old rows at or above the ones it overwrites.
__global__ void gdn_replay_conv_kernel(const __grid_constant__ GdnReplayTable t, int tokens, int channels, int width) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, l = blockIdx.y;
    if (c >= channels) return;
    float* st = t.conv_state[l];
    const float* x = t.qkv[l];
    for (int j = 0; j < width - 1; ++j) {
        const int r = tokens + j;
        st[static_cast<size_t>(j) * channels + c] = r < width - 1 ? st[static_cast<size_t>(r) * channels + c]
                                                                  : x[static_cast<size_t>(r - (width - 1)) * channels + c];
    }
}


// Chunkwise-parallel Gated DeltaNet for prefill (the chunked delta rule in WY form). Within a chunk
// of C tokens let G be the in-chunk cumulative log decay and gamma = exp(G). The per-token deltas
// D_r = beta_r (v_r - S'_r k_r) (S'_r = decayed state before token r) solve (I + A) D = R with
//   A[r][s] = beta_r exp(G_r - G_s) k_r.k_s (s < r),   R_r = beta_r (v_r - gamma_r S0 k_r),
// so D = T R with T = (I + A)^-1, and then
//   o_r = gamma_r S0 q_r + sum_{s<=r} exp(G_r - G_s) (q_r.k_s) D_s
//   S_C = gamma_{C-1} S0 + sum_s exp(G_{C-1} - G_s) D_s k_s^T.
// The key-side quantities (gates, T = (I + A)^-1, P) do not depend on the state, so a prepare pass
// computes them for every (head, chunk) in parallel into scratch; the recurrence then owns head h
// and kGdnVt state rows (kept in registers across chunks; chunks run in order) and only does the
// state-dependent products.
constexpr int kGdnChunk = 16;
constexpr int kGdnVt = 16;
constexpr int kGdnThreads = 128;
// Below this many tokens the per-row recurrence is as fast (decode and MTP verify stay on it).
constexpr int kGdnChunkMinTokens = 16;

// Reduce-scatter over the 8 lanes of an aligned lane group: on return v[0] of lane g (g = lane % 8)
// holds the group sum of v[g].
__device__ __forceinline__ void gdn_reduce_scatter8(float (&v)[8], int g) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const bool hi = g & 4;
        const float send = hi ? v[k] : v[k + 4], keep = hi ? v[k + 4] : v[k];
        v[k] = keep + __shfl_xor_sync(0xffffffffu, send, 4);
    }
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        const bool hi = g & 2;
        const float send = hi ? v[k] : v[k + 2], keep = hi ? v[k + 2] : v[k];
        v[k] = keep + __shfl_xor_sync(0xffffffffu, send, 2);
    }
    const bool hi = g & 1;
    const float send = hi ? v[0] : v[1], keep = hi ? v[1] : v[0];
    v[0] = keep + __shfl_xor_sync(0xffffffffu, send, 1);
}

// Per chunk in scratch: T[C][C] (row-major), P[C][C], gamma[C], decay_last[C], beta[C].
constexpr int kGdnPrep = 2 * kGdnChunk * kGdnChunk + 3 * kGdnChunk;

// One block per (value head, chunk): L2-normalized q/k of the chunk, gates and their scan,
// A[r][s] = beta_r exp(G_r - G_s) k_r.k_s (s < r), P[r][s] = exp(G_r - G_s) q_r.k_s (s <= r), and
// T = (I + A)^-1 by forward substitution.
template <int DK>
__global__ void __launch_bounds__(kGdnThreads)
    gated_delta_net_prepare_kernel(const float* __restrict__ qkv, const float* __restrict__ ab,
                                   const float* __restrict__ a_log, const float* __restrict__ dt_bias,
                                   float* __restrict__ prep, int tokens, int k_heads, int v_heads, int v_dim) {
    constexpr int C = kGdnChunk, KS = DK + 4, AS = C + 4;
    __shared__ __align__(16) float ks[C * KS], qs[C * KS];
    __shared__ __align__(16) float at[C * AS];  // A transposed: at[s][r] = A[r][s]
    __shared__ float gs[C], betas[C];
    const int h = blockIdx.x, ci = blockIdx.y, t0 = ci * C, tid = threadIdx.x;
    const int lane = tid % VELOX_WARP, warp = tid / VELOX_WARP;
    const int kh = h / (v_heads / k_heads);
    const int n = tokens - t0 < C ? tokens - t0 : C;
    const size_t width = static_cast<size_t>(2 * k_heads * DK + v_heads * v_dim);
    float* out = prep + (static_cast<size_t>(h) * gridDim.y + ci) * kGdnPrep;

    // Rows: 8 threads per chunk row, each 4-wide pieces at dc * 4 + 32 j.
    {
        constexpr int NJ = DK / 32;
        const int r = tid / 8, dc = tid % 8;
        float q[NJ][4], k[NJ][4], qq = 0.0f, kk = 0.0f;
        const float* row = qkv + static_cast<size_t>(t0 + r) * width;
#pragma unroll
        for (int j = 0; j < NJ; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                q[j][e] = r < n ? row[kh * DK + dc * 4 + 32 * j + e] : 0.0f;
                k[j][e] = r < n ? row[(k_heads + kh) * DK + dc * 4 + 32 * j + e] : 0.0f;
                qq += q[j][e] * q[j][e];
                kk += k[j][e] * k[j][e];
            }
#pragma unroll
        for (int o = 1; o < 8; o <<= 1) {
            qq += __shfl_xor_sync(0xffffffffu, qq, o);
            kk += __shfl_xor_sync(0xffffffffu, kk, o);
        }
        const float qn = rsqrtf(qq + 1e-6f), kn = rsqrtf(kk + 1e-6f);
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            *reinterpret_cast<float4*>(&qs[r * KS + dc * 4 + 32 * j]) =
                make_float4(q[j][0] * qn, q[j][1] * qn, q[j][2] * qn, q[j][3] * qn);
            *reinterpret_cast<float4*>(&ks[r * KS + dc * 4 + 32 * j]) =
                make_float4(k[j][0] * kn, k[j][1] * kn, k[j][2] * kn, k[j][3] * kn);
        }
    }
    // Gates (padded rows: g = 0, beta = 0, so they change nothing) and their in-chunk scan.
    if (warp == 0) {
        float g = 0.0f, beta = 0.0f;
        if (lane < n) {
            const float* abr = ab + static_cast<size_t>(t0 + lane) * 2 * v_heads;
            const float a = abr[h] + dt_bias[h];
            g = -expf(a_log[h]) * (a > 20.0f ? a : log1pf(expf(a)));
            beta = 1.0f / (1.0f + expf(-abr[v_heads + h]));
        }
#pragma unroll
        for (int o = 1; o < VELOX_WARP; o <<= 1) {
            const float u = __shfl_up_sync(0xffffffffu, g, o);
            if (lane >= o) g += u;
        }
        const float last = __shfl_sync(0xffffffffu, g, C - 1);
        if (lane < C) {
            gs[lane] = g;
            betas[lane] = beta;
            out[2 * C * C + lane] = expf(g);
            out[2 * C * C + C + lane] = expf(last - g);
            out[2 * C * C + 2 * C + lane] = beta;
        }
    }
    __syncthreads();

    // A and P: one (r, s) pair per thread and step.
    for (int idx = tid; idx < C * C; idx += kGdnThreads) {
        const int r = idx / C, c = idx % C;
        float kk = 0.0f, qk = 0.0f;
        if (c <= r) {
#pragma unroll 8
            for (int d = 0; d < DK; d += 4) {
                const float4 kc = *reinterpret_cast<const float4*>(&ks[c * KS + d]);
                const float4 kr = *reinterpret_cast<const float4*>(&ks[r * KS + d]);
                const float4 qr = *reinterpret_cast<const float4*>(&qs[r * KS + d]);
                kk += kc.x * kr.x + kc.y * kr.y + kc.z * kr.z + kc.w * kr.w;
                qk += kc.x * qr.x + kc.y * qr.y + kc.z * qr.z + kc.w * qr.w;
            }
        }
        const float dec = c <= r ? expf(gs[r] - gs[c]) : 0.0f;
        at[c * AS + r] = c < r ? betas[r] * dec * kk : 0.0f;
        out[C * C + r * C + c] = dec * qk;
    }
    __syncthreads();

    // T = (I + A)^-1; lane j owns column j.
    if (warp == 0) {
        float tc[C];
#pragma unroll
        for (int r = 0; r < C; ++r) tc[r] = r == lane ? 1.0f : 0.0f;
#pragma unroll
        for (int c = 0; c < C - 1; ++c) {
#pragma unroll
            for (int r = (c + 1) & ~3; r < C; r += 4) {
                const float4 a4 = *reinterpret_cast<const float4*>(&at[c * AS + r]);
                tc[r] -= a4.x * tc[c];
                tc[r + 1] -= a4.y * tc[c];
                tc[r + 2] -= a4.z * tc[c];
                tc[r + 3] -= a4.w * tc[c];
            }
        }
#pragma unroll
        for (int r = 0; r < C; ++r)
            if (lane < C) out[r * C + lane] = tc[r];
    }
}

// The state-dependent part, chunks in order, from the prepare pass's per-chunk matrices. A block
// owns R * 16 state rows of one head; each thread holds R of them (rows si + 16 r) over 16 columns,
// so every shared key/query/T/P/D value it reads feeds R rows (the loop is bound by shared-memory
// reads, not FMAs).
template <int DK, int R>
__global__ void __launch_bounds__(kGdnThreads, R > 1 ? 2 : 3)
    gated_delta_net_chunk_kernel(const float* __restrict__ qkv, const float* __restrict__ prep,
                                 float* __restrict__ state, float* __restrict__ out, int tokens, int k_heads,
                                 int v_heads, int v_dim) {
    constexpr int C = kGdnChunk, VT = kGdnVt * R, KS = DK + 4, PS = C + 4, RS = VT + 1, NJ = DK / 32;
    constexpr int NV = C * VT / kGdnThreads;
    static_assert(kGdnThreads == kGdnVt * 8 && kGdnThreads == C * 8, "8 threads per state row group and chunk row");
    __shared__ __align__(16) float ks[C * KS];  // normalized keys
    __shared__ __align__(16) float qs[C * KS];  // normalized queries
    __shared__ __align__(16) float ts[C * PS];  // T
    __shared__ __align__(16) float ps[C * PS];  // P
    __shared__ float rs[C * RS];                // V, then R in place
    __shared__ float ds[C * RS];                // D
    __shared__ float gam[C], decay_last[C], betas[C];

    const int h = blockIdx.x, i0 = blockIdx.y * VT, tid = threadIdx.x;
    const int kh = h / (v_heads / k_heads);
    const int chunks = (tokens + C - 1) / C;
    const size_t width = static_cast<size_t>(2 * k_heads * DK + v_heads * v_dim);
    const float inv_sqrt_dk = rsqrtf(static_cast<float>(DK));

    // Thread (si, dc) owns state rows si + 16 r (and chunk row si for loads) over columns
    // dc * 4 + 32 * j + e; its 8-thread group is lane-aligned within the warp.
    const int si = tid / 8, dc = tid % 8;
    bool row_ok[R];
    float s[R][NJ][4];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int col = si + kGdnVt * r;
        row_ok[r] = i0 + col < v_dim;
        const float* row_state = state + (static_cast<size_t>(h) * v_dim + i0 + col) * DK;
#pragma unroll
        for (int j = 0; j < NJ; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) s[r][j][e] = row_ok[r] ? row_state[dc * 4 + 32 * j + e] : 0.0f;
    }

    // Raw inputs of a chunk, loaded into registers while the previous chunk finishes.
    float rq[NJ][4], rk[NJ][4], rv[NV], rt[2], rp[2], rg = 0.0f;
    auto load = [&](int ci) {
        const int t0 = ci * C, n = tokens - t0 < C ? tokens - t0 : C;
        const float* row = qkv + static_cast<size_t>(t0 + si) * width;
#pragma unroll
        for (int j = 0; j < NJ; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                rq[j][e] = si < n ? row[kh * DK + dc * 4 + 32 * j + e] : 0.0f;
                rk[j][e] = si < n ? row[(k_heads + kh) * DK + dc * 4 + 32 * j + e] : 0.0f;
            }
#pragma unroll
        for (int m = 0; m < NV; ++m) {
            const int e = tid + kGdnThreads * m, c = e / VT, i = e % VT;
            const float* vrow = qkv + static_cast<size_t>(t0 + c) * width;
            rv[m] = c < n && i0 + i < v_dim ? vrow[2 * k_heads * DK + h * v_dim + i0 + i] : 0.0f;
        }
        const float* pr = prep + (static_cast<size_t>(h) * chunks + ci) * kGdnPrep;
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            rt[m] = pr[tid + kGdnThreads * m];
            rp[m] = pr[C * C + tid + kGdnThreads * m];
        }
        if (tid < 3 * C) rg = pr[2 * C * C + tid];
    };
    load(0);

    for (int ci = 0; ci < chunks; ++ci) {
        const int t0 = ci * C, n = tokens - t0 < C ? tokens - t0 : C;
        // L2-normalize q and k of chunk row si (8-lane group reduction) and publish the chunk.
        {
            float qq = 0.0f, kk = 0.0f;
#pragma unroll
            for (int j = 0; j < NJ; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    qq += rq[j][e] * rq[j][e];
                    kk += rk[j][e] * rk[j][e];
                }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) {
                qq += __shfl_xor_sync(0xffffffffu, qq, o);
                kk += __shfl_xor_sync(0xffffffffu, kk, o);
            }
            const float qn = rsqrtf(qq + 1e-6f), kn = rsqrtf(kk + 1e-6f);
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
                *reinterpret_cast<float4*>(&qs[si * KS + dc * 4 + 32 * j]) =
                    make_float4(rq[j][0] * qn, rq[j][1] * qn, rq[j][2] * qn, rq[j][3] * qn);
                *reinterpret_cast<float4*>(&ks[si * KS + dc * 4 + 32 * j]) =
                    make_float4(rk[j][0] * kn, rk[j][1] * kn, rk[j][2] * kn, rk[j][3] * kn);
            }
#pragma unroll
            for (int m = 0; m < NV; ++m) {
                const int e = tid + kGdnThreads * m;
                rs[(e / VT) * RS + e % VT] = rv[m];
            }
#pragma unroll
            for (int m = 0; m < 2; ++m) {
                const int e = tid + kGdnThreads * m;
                ts[(e / C) * PS + e % C] = rt[m];
                ps[(e / C) * PS + e % C] = rp[m];
            }
            if (tid < C) gam[tid] = rg;
            else if (tid < 2 * C) decay_last[tid - C] = rg;
            else if (tid < 3 * C) betas[tid - 2 * C] = rg;
        }
        __syncthreads();
        if (ci + 1 < chunks) load(ci + 1);

        // x = S0[row].k_c, y = S0[row].q_c for chunk rows c, 8 at a time (lane dc ends up owning
        // c = 8 g + dc): R[c][row] = beta_c (v_c - gamma_c x), and gamma_c y kept for o.
        float ys[R][C / 8];
#pragma unroll
        for (int g = 0; g < C / 8; ++g) {
            float x[R][8], y[R][8];
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const int c = 8 * g + u;
#pragma unroll
                for (int r = 0; r < R; ++r) x[r][u] = y[r][u] = 0.0f;
#pragma unroll
                for (int j = 0; j < NJ; ++j) {
                    const float4 kv = *reinterpret_cast<const float4*>(&ks[c * KS + dc * 4 + 32 * j]);
                    const float4 qv = *reinterpret_cast<const float4*>(&qs[c * KS + dc * 4 + 32 * j]);
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        x[r][u] += kv.x * s[r][j][0] + kv.y * s[r][j][1] + kv.z * s[r][j][2] + kv.w * s[r][j][3];
                        y[r][u] += qv.x * s[r][j][0] + qv.y * s[r][j][1] + qv.z * s[r][j][2] + qv.w * s[r][j][3];
                    }
                }
            }
            const int c = 8 * g + dc;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                gdn_reduce_scatter8(x[r], dc);
                gdn_reduce_scatter8(y[r], dc);
                const int col = si + kGdnVt * r;
                rs[c * RS + col] = betas[c] * (rs[c * RS + col] - gam[c] * x[r][0]);
                ys[r][g] = gam[c] * y[r][0];
            }
        }
        __syncthreads();

        // D = T R, o = gamma S0 q + P D, and the state update for this thread's rows (T and P are
        // lower triangular with explicit zeros).
        {
            float acc[R][C / 8];
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
                for (int g = 0; g < C / 8; ++g) acc[r][g] = 0.0f;
#pragma unroll 2
            for (int u = 0; u < C; u += 4) {
                float rr[R][4];
#pragma unroll
                for (int r = 0; r < R; ++r)
#pragma unroll
                    for (int v = 0; v < 4; ++v) rr[r][v] = rs[(u + v) * RS + si + kGdnVt * r];
#pragma unroll
                for (int g = 0; g < C / 8; ++g) {
                    const float4 t4 = *reinterpret_cast<const float4*>(&ts[(8 * g + dc) * PS + u]);
#pragma unroll
                    for (int r = 0; r < R; ++r)
                        acc[r][g] += t4.x * rr[r][0] + t4.y * rr[r][1] + t4.z * rr[r][2] + t4.w * rr[r][3];
                }
            }
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
                for (int g = 0; g < C / 8; ++g) ds[(8 * g + dc) * RS + si + kGdnVt * r] = acc[r][g];
        }
        __syncthreads();
        {
            float acc[R][C / 8];
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
                for (int g = 0; g < C / 8; ++g) acc[r][g] = ys[r][g];
#pragma unroll 2
            for (int u = 0; u < C; u += 4) {
                float dd[R][4];
#pragma unroll
                for (int r = 0; r < R; ++r)
#pragma unroll
                    for (int v = 0; v < 4; ++v) dd[r][v] = ds[(u + v) * RS + si + kGdnVt * r];
#pragma unroll
                for (int g = 0; g < C / 8; ++g) {
                    const float4 p4 = *reinterpret_cast<const float4*>(&ps[(8 * g + dc) * PS + u]);
#pragma unroll
                    for (int r = 0; r < R; ++r)
                        acc[r][g] += p4.x * dd[r][0] + p4.y * dd[r][1] + p4.z * dd[r][2] + p4.w * dd[r][3];
                }
            }
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
                for (int g = 0; g < C / 8; ++g) {
                    const int c = 8 * g + dc, col = si + kGdnVt * r;
                    if (c < n && row_ok[r])
                        out[(static_cast<size_t>(t0 + c) * v_heads + h) * v_dim + i0 + col] = acc[r][g] * inv_sqrt_dk;
                }
        }
        {
            const float gl = gam[C - 1];
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
                for (int j = 0; j < NJ; ++j)
#pragma unroll
                    for (int e = 0; e < 4; ++e) s[r][j][e] *= gl;
            for (int c = 0; c < n; ++c) {
                float dv[R];
#pragma unroll
                for (int r = 0; r < R; ++r) dv[r] = ds[c * RS + si + kGdnVt * r] * decay_last[c];
#pragma unroll
                for (int j = 0; j < NJ; ++j) {
                    const float4 kv = *reinterpret_cast<const float4*>(&ks[c * KS + dc * 4 + 32 * j]);
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        s[r][j][0] += dv[r] * kv.x;
                        s[r][j][1] += dv[r] * kv.y;
                        s[r][j][2] += dv[r] * kv.z;
                        s[r][j][3] += dv[r] * kv.w;
                    }
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (!row_ok[r]) continue;
        float* row_state = state + (static_cast<size_t>(h) * v_dim + i0 + si + kGdnVt * r) * DK;
#pragma unroll
        for (int j = 0; j < NJ; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) row_state[dc * 4 + 32 * j + e] = s[r][j][e];
    }
}

// Small-m linear for FP8-row and NVFP4: one warp per output row, codes decoded a group at a time
// (16 bytes of FP8 codes, or one 16-element NVFP4 group sharing a scale) and applied to all m
// rows, so each weight is read once.
constexpr int kGemvMaxM = 8;

__device__ __forceinline__ float fp8_lut(uint8_t b) { return e4m3fn_to_f32(b); }

template <int FMT>
__global__ void linear_gemv_kernel(const float* __restrict__ x, WDesc w, long long row_begin, int rows, int m,
                                   float* __restrict__ y) {
    const int r = blockIdx.x * (blockDim.x / VELOX_WARP) + threadIdx.x / VELOX_WARP;
    const int lane = threadIdx.x % VELOX_WARP;
    if (r >= rows) return;
    const long long row = row_begin + r, k = w.k;
    float acc[kGemvMaxM];
#pragma unroll
    for (int i = 0; i < kGemvMaxM; ++i) acc[i] = 0.0f;
    if (FMT == VELOX_FMT_FP8_ROW_BF16) {
        const uint8_t* codes = w.base + row * k;
        for (long long c0 = lane * 16LL; c0 < k; c0 += 16LL * VELOX_WARP) {
            const int n = static_cast<int>(k - c0 < 16 ? k - c0 : 16);
            float wv[16];
#pragma unroll
            for (int j = 0; j < 16; ++j) wv[j] = j < n ? fp8_lut(codes[c0 + j]) : 0.0f;
#pragma unroll
            for (int i = 0; i < kGemvMaxM; ++i) {
                if (i < m) {
                    const float* xr = x + static_cast<size_t>(i) * k + c0;
                    float s = 0.0f;
#pragma unroll
                    for (int j = 0; j < 16; ++j)
                        if (j < n) s += wv[j] * xr[j];
                    acc[i] += s;
                }
            }
        }
        const uint8_t* sp = w.base + align256(w.n * k) + 2 * row;
        const float scale = bf16_to_f32(static_cast<uint16_t>(sp[0] | (sp[1] << 8)));
#pragma unroll
        for (int i = 0; i < kGemvMaxM; ++i) acc[i] *= scale;
    } else {  // NVFP4: k % 64 == 0, so groups of 16 never straddle the row end
        const uint8_t* codes = w.base + row * k / 2;
        const long long scale_plane = align256(w.n * k / 2);
        const float inv_div = 1.0f / *reinterpret_cast<const float*>(w.base + scale_plane + w.n * k / 16);
        for (long long g = lane; g < k / 16; g += VELOX_WARP) {
            const float scale = e4m3fn_to_f32(w.base[scale_plane + nvfp4_scale_offset(row, g, k)]);
            float wv[16];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint8_t byte = codes[g * 8 + j];
                wv[2 * j] = e2m1_to_f32(byte & 0xf) * scale;
                wv[2 * j + 1] = e2m1_to_f32(byte >> 4) * scale;
            }
#pragma unroll
            for (int i = 0; i < kGemvMaxM; ++i) {
                if (i < m) {
                    const float* xr = x + static_cast<size_t>(i) * k + g * 16;
                    float s = 0.0f;
#pragma unroll
                    for (int j = 0; j < 16; ++j) s += wv[j] * xr[j];
                    acc[i] += s;
                }
            }
        }
#pragma unroll
        for (int i = 0; i < kGemvMaxM; ++i) acc[i] *= inv_div;
    }
#pragma unroll
    for (int i = 0; i < kGemvMaxM; ++i) {
        const float s = velox_warp_sum(acc[i]);
        if (lane == 0 && i < m) y[static_cast<size_t>(i) * rows + r] = s;
    }
}

float* linear_split_scratch(size_t floats) {
    static VeloxScratch s;
    return static_cast<float*>(s.get(floats * sizeof(float)));
}

__global__ void sum_splits_kernel(const float* __restrict__ part, int splits, long long n, float* __restrict__ y) {
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        float s = 0.0f;
        for (int z = 0; z < splits; ++z) s += part[z * n + i];
        y[i] = s;
    }
}

// Vectorized small-m linear for FP8-row and NVFP4. Each warp owns R adjacent weight rows and
// streams them in 16-byte loads (16 FP8 or 32 NVFP4 codes per lane per row), so one x load feeds
// R rows and several loads per lane are in flight. Requires 16-byte aligned rows: k % 16 == 0
// for FP8; NVFP4 always qualifies (k % 64 == 0).
__device__ __forceinline__ uint4 velox_ld_stream(const void* p) {
    return __ldg(reinterpret_cast<const uint4*>(p));
}

// E4M3FN byte -> f32 by moving sign, exponent and mantissa into place and rebiasing (2^120 is
// exact, and covers the subnormals). NaN codes never occur in valid weights.
__device__ __forceinline__ float e4m3_bits(uint32_t word, int byte) {
    const uint32_t b = (word >> (8 * byte)) & 0xffu;
    return __uint_as_float(((b & 0x80u) << 24) | ((b & 0x7fu) << 20)) * 0x1p120f;
}

// The two E2M1 codes of the low byte of `t`, times 2^-14: each nibble's magnitude bits go to f16
// bits [11:9] and its sign to bit 15, so exponent e becomes 2^(e-15) and e = 0 a subnormal
// m * 2^-15, i.e. code value * 2^-14 exactly. Callers fold 2^14 into the block scale.
__device__ __forceinline__ float2 e2m1x2_scaled(uint32_t t) {
    const uint32_t h = ((t & 0x7u) << 9) | ((t & 0x8u) << 12) | ((t & 0x70u) << 21) | ((t & 0x80u) << 24);
    return __half22float2(*reinterpret_cast<const __half2*>(&h));
}

// KW warps share one row group and split its K (KW = 8 for matrices with few rows, where one warp
// per row would leave most SMs idle); their sums meet in shared memory.
template <int FMT, int R, int M, int KW>
__global__ void __launch_bounds__(256)
linear_gemv_vec_kernel(const float* __restrict__ x, WDesc w, long long row_begin, int rows, int m,
                       float* __restrict__ y) {
    __shared__ float red[KW][R][M];
    const int wid = blockIdx.x * (blockDim.x / VELOX_WARP) + threadIdx.x / VELOX_WARP;
    const int kw = (threadIdx.x / VELOX_WARP) % KW;
    const int lane = threadIdx.x % VELOX_WARP;
    const int r0 = (wid / KW) * R;
    if (r0 >= rows) return;  // whole row groups only, so KW > 1 blocks return together
    const long long k = w.k;
    // Groupwise Q4 (group 64) / Q8 (group 32), row_split_k128_v1: a row's base plane is gpr
    // groups of 32 bytes over K padded to 128; one 16-byte chunk is half a group either way.
    constexpr bool kRowSplit = FMT == VELOX_FMT_Q4 || FMT == VELOX_FMT_Q8;
    const long long gpr = (k + 127) / 128 * 128 / (FMT == VELOX_FMT_Q8 ? 32 : 64);
    const long long row_bytes = kRowSplit ? gpr * 32
                                : FMT == VELOX_FMT_NVFP4 ? k / 2
                                : FMT == VELOX_FMT_BF16  ? 2 * k
                                                         : k;
    constexpr int kElems = FMT == VELOX_FMT_NVFP4 || FMT == VELOX_FMT_Q4 ? 32 : FMT == VELOX_FMT_BF16 ? 8 : 16;
    // Only real columns are visited (the launcher requires k % kElems == 0 for row-split).
    const long long chunks = kRowSplit ? k / kElems : row_bytes / 16;
    const uint8_t* rp[R];
    bool live[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        live[r] = r0 + r < rows;
        rp[r] = w.base + (row_begin + (live[r] ? r0 + r : r0)) * row_bytes;
    }
    float acc[R][M];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int i = 0; i < M; ++i) acc[r][i] = 0.0f;
    const long long scale_plane = align256(w.n * k / 2);
    const uint8_t* sbase[R];
#pragma unroll
    for (int r = 0; r < R; ++r) sbase[r] = w.base + scale_plane + nvfp4_scale_offset(row_begin + r0 + (live[r] ? r : 0), 0, k);
    // Row-split FP16 group scales follow the base plane (no high plane for Q4/Q8).
    const uint16_t* gscale[R];
#pragma unroll
    for (int r = 0; r < R; ++r)
        gscale[r] = reinterpret_cast<const uint16_t*>(w.base + align256(w.n * gpr * 32)) + (row_begin + r0 + (live[r] ? r : 0)) * gpr;

    for (long long c = kw * VELOX_WARP + lane; c < chunks; c += VELOX_WARP * KW) {
        uint4 wv[R];
#pragma unroll
        for (int r = 0; r < R; ++r) wv[r] = velox_ld_stream(rp[r] + c * 16);
        float sc[R][2];
        if (FMT == VELOX_FMT_NVFP4) {
            // Groups 2c and 2c+1 are adjacent bytes of one 4-byte scale word: 512-byte tiles per
            // 64 columns, so chunk c sits at (c / 2) * 512 + (c % 2) * 2 past the row's base.
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint32_t s2 = *reinterpret_cast<const uint16_t*>(sbase[r] + (c >> 1) * 512 + (c & 1) * 2);
                sc[r][0] = e4m3_bits(s2, 0) * 16384.0f;  // e2m1x2_scaled is 2^-14 times the code
                sc[r][1] = e4m3_bits(s2, 1) * 16384.0f;
            }
        }
        if (kRowSplit) {
#pragma unroll
            for (int r = 0; r < R; ++r) sc[r][0] = __half2float(__ushort_as_half(gscale[r][c >> 1]));
        }
#pragma unroll
        for (int i = 0; i < M; ++i) {
            if (i >= m) break;
            const float4* xp = reinterpret_cast<const float4*>(x + static_cast<size_t>(i) * k + c * kElems);
            float xv[kElems];
#pragma unroll
            for (int q = 0; q < kElems / 4; ++q) {
                const float4 t = __ldg(xp + q);
                xv[4 * q] = t.x, xv[4 * q + 1] = t.y, xv[4 * q + 2] = t.z, xv[4 * q + 3] = t.w;
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint32_t words[4] = {wv[r].x, wv[r].y, wv[r].z, wv[r].w};
                if (FMT == VELOX_FMT_NVFP4) {
                    float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
                    for (int b = 0; b < 16; ++b) {
                        const float2 v = e2m1x2_scaled(words[b / 4] >> (8 * (b % 4)));
                        const float s = v.x * xv[2 * b] + v.y * xv[2 * b + 1];
                        if (b < 8) s0 += s; else s1 += s;
                    }
                    acc[r][i] += s0 * sc[r][0] + s1 * sc[r][1];
                } else if (FMT == VELOX_FMT_Q4) {
                    float s = 0.0f;
#pragma unroll
                    for (int b = 0; b < 16; ++b) {
                        const uint32_t byte = (words[b / 4] >> (8 * (b % 4))) & 0xffu;
                        const int lo = static_cast<int>(byte & 0xfu) - (byte & 0x8u ? 16 : 0);
                        const int hi = static_cast<int>(byte >> 4) - (byte & 0x80u ? 16 : 0);
                        s += static_cast<float>(lo) * xv[2 * b] + static_cast<float>(hi) * xv[2 * b + 1];
                    }
                    acc[r][i] += s * sc[r][0];
                } else if (FMT == VELOX_FMT_Q8) {
                    float s = 0.0f;
#pragma unroll
                    for (int b = 0; b < 16; ++b)
                        s += static_cast<float>(static_cast<int8_t>((words[b / 4] >> (8 * (b % 4))) & 0xffu)) * xv[b];
                    acc[r][i] += s * sc[r][0];
                } else if (FMT == VELOX_FMT_BF16) {
                    float s = 0.0f;
#pragma unroll
                    for (int b = 0; b < 4; ++b)
                        s += __uint_as_float(words[b] << 16) * xv[2 * b] + __uint_as_float(words[b] & 0xffff0000u) * xv[2 * b + 1];
                    acc[r][i] += s;
                } else {
                    float s = 0.0f;
#pragma unroll
                    for (int b = 0; b < 16; ++b) s += e4m3_bits(words[b / 4], b % 4) * xv[b];
                    acc[r][i] += s;
                }
            }
        }
    }
    if (KW > 1) {
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int i = 0; i < M; ++i) {
                const float s = velox_warp_sum(acc[r][i]);
                if (lane == 0) red[kw][r][i] = s;
            }
        __syncthreads();
        if (kw != 0) return;
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int i = 0; i < M; ++i) {
                float s = 0.0f;
                for (int j = 0; j < KW; ++j) s += red[j][r][i];
                acc[r][i] = lane == 0 ? s : 0.0f;  // the warp sum below then returns s
            }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (!live[r]) continue;
        const long long row = row_begin + r0 + r;
        float post = 1.0f;
        if (FMT == VELOX_FMT_BF16 || kRowSplit) {
        } else if (FMT == VELOX_FMT_NVFP4) {
            post = 1.0f / *reinterpret_cast<const float*>(w.base + scale_plane + w.n * k / 16);
        } else {
            const uint8_t* sp = w.base + align256(w.n * k) + 2 * row;
            post = bf16_to_f32(static_cast<uint16_t>(sp[0] | (sp[1] << 8)));
        }
#pragma unroll
        for (int i = 0; i < M; ++i) {
            const float s = velox_warp_sum(acc[r][i]);
            if (lane == 0 && i < m) y[static_cast<size_t>(i) * rows + r0 + r] = s * post;
        }
    }
}

template <int FMT, int M>
int launch_gemv_vec(const float* x, const WDesc& d, long long row_begin, int rows, int m, float* y) {
    constexpr int kWarps = 8;
    // Few rows: a whole block (8 warps) per row, splitting K. Otherwise prefer more rows per warp
    // (more x reuse) while keeping enough warps to fill the GPU.
    if (rows < 2048) {
        linear_gemv_vec_kernel<FMT, 1, M, kWarps><<<rows, kWarps * VELOX_WARP>>>(x, d, row_begin, rows, m, y);
        return velox_launch_status();
    }
    const int r = rows >= 4 * 4096 ? 4 : rows >= 2 * 2048 ? 2 : 1;
    const int warps = (rows + r - 1) / r;
    const dim3 grid((warps + kWarps - 1) / kWarps);
    if (r == 4) linear_gemv_vec_kernel<FMT, 4, M, 1><<<grid, kWarps * VELOX_WARP>>>(x, d, row_begin, rows, m, y);
    else if (r == 2) linear_gemv_vec_kernel<FMT, 2, M, 1><<<grid, kWarps * VELOX_WARP>>>(x, d, row_begin, rows, m, y);
    else linear_gemv_vec_kernel<FMT, 1, M, 1><<<grid, kWarps * VELOX_WARP>>>(x, d, row_begin, rows, m, y);
    return velox_launch_status();
}

// Small-m linear for GGML blocks: one warp per output row, each lane decoding whole 32-value
// sub-blocks (one scale each; a 32-value block is one sub-block) and applying them to all m rows, so each weight is read once.
template <int T, int M>
__global__ void __launch_bounds__(256)
linear_gemv_ggml_kernel(const float* __restrict__ x, WDesc w, long long row_begin, int rows, int m,
                        float* __restrict__ y) {
    const int r = blockIdx.x * (blockDim.x / VELOX_WARP) + threadIdx.x / VELOX_WARP;
    const int lane = threadIdx.x % VELOX_WARP;
    if (r >= rows) return;
    const long long k = w.k, subs = k / 32;
    const int spb = ggml_block_values(T) / 32;  // 32-value sub-blocks per block
    const uint8_t* row = w.base + (row_begin + r) * (k / ggml_block_values(T)) * ggml_block_bytes(T);
    float acc[M];
#pragma unroll
    for (int i = 0; i < M; ++i) acc[i] = 0.0f;
    for (long long u = lane; u < subs; u += VELOX_WARP) {
        const uint8_t* blk = row + (u / spb) * ggml_block_bytes(T);
        const int v0 = static_cast<int>(u % spb) * 32;
        float wv[32];
#pragma unroll
        for (int j = 0; j < 32; ++j) wv[j] = ggml_value<T>(blk, v0 + j);
#pragma unroll
        for (int i = 0; i < M; ++i) {
            if (i >= m) break;
            const float4* xp = reinterpret_cast<const float4*>(x + static_cast<size_t>(i) * k + u * 32);
            float s = 0.0f;
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                const float4 t = __ldg(xp + q);
                s += wv[4 * q] * t.x + wv[4 * q + 1] * t.y + wv[4 * q + 2] * t.z + wv[4 * q + 3] * t.w;
            }
            acc[i] += s;
        }
    }
#pragma unroll
    for (int i = 0; i < M; ++i) {
        const float s = velox_warp_sum(acc[i]);
        if (lane == 0 && i < m) y[static_cast<size_t>(i) * rows + r] = s;
    }
}

template <int T>
int launch_gemv_ggml_t(const float* x, const WDesc& d, long long row_begin, int rows, int m, float* y) {
    constexpr int kWarps = 8;
    const dim3 grid((rows + kWarps - 1) / kWarps);
    if (m == 1) linear_gemv_ggml_kernel<T, 1><<<grid, kWarps * VELOX_WARP>>>(x, d, row_begin, rows, m, y);
    else if (m == 2) linear_gemv_ggml_kernel<T, 2><<<grid, kWarps * VELOX_WARP>>>(x, d, row_begin, rows, m, y);
    else if (m <= 4) linear_gemv_ggml_kernel<T, 4><<<grid, kWarps * VELOX_WARP>>>(x, d, row_begin, rows, m, y);
    else linear_gemv_ggml_kernel<T, 8><<<grid, kWarps * VELOX_WARP>>>(x, d, row_begin, rows, m, y);
    return velox_launch_status();
}

int launch_gemv_ggml(const float* x, const WDesc& d, long long row_begin, int rows, int m, float* y) {
    switch (d.format - VELOX_FMT_GGML_BASE) {
        case 0: return launch_gemv_ggml_t<0>(x, d, row_begin, rows, m, y);
        case 1: return launch_gemv_ggml_t<1>(x, d, row_begin, rows, m, y);
        case 2: return launch_gemv_ggml_t<2>(x, d, row_begin, rows, m, y);
        case 3: return launch_gemv_ggml_t<3>(x, d, row_begin, rows, m, y);
        case 4: return launch_gemv_ggml_t<4>(x, d, row_begin, rows, m, y);
        case 5: return launch_gemv_ggml_t<5>(x, d, row_begin, rows, m, y);
        case 6: return launch_gemv_ggml_t<6>(x, d, row_begin, rows, m, y);
        case 7: return launch_gemv_ggml_t<7>(x, d, row_begin, rows, m, y);
        case 8: return launch_gemv_ggml_t<8>(x, d, row_begin, rows, m, y);
        case 9: return launch_gemv_ggml_t<9>(x, d, row_begin, rows, m, y);
        case 10: return launch_gemv_ggml_t<10>(x, d, row_begin, rows, m, y);
        case 11: return launch_gemv_ggml_t<11>(x, d, row_begin, rows, m, y);
        default: return launch_gemv_ggml_t<12>(x, d, row_begin, rows, m, y);
    }
}

template <int FMT>
int launch_gemv_vec_m(const float* x, const WDesc& d, long long row_begin, int rows, int m, float* y) {
    if (m == 1) return launch_gemv_vec<FMT, 1>(x, d, row_begin, rows, m, y);
    if (m == 2) return launch_gemv_vec<FMT, 2>(x, d, row_begin, rows, m, y);
    if (m <= 4) return launch_gemv_vec<FMT, 4>(x, d, row_begin, rows, m, y);
    return launch_gemv_vec<FMT, 8>(x, d, row_begin, rows, m, y);
}

}  // namespace

namespace {
// Few output rows (e.g. a 96-row gate projection) over a few tokens, bf16 weights, exact f32
// activations: a block computes kSnRows rows for every token, its threads splitting K, so the
// activations are read once per row group rather than once per row. Fixed-order reductions.
constexpr int kSnRows = 4, kSnThreads = 512, kSnMaxM = 16;

template <int M>
__global__ void __launch_bounds__(kSnThreads) linear_small_n_bf16_kernel(const float* __restrict__ x,
                                                                      const uint16_t* __restrict__ w, long long k,
                                                                      long long row_begin, int rows, int m,
                                                                      float* __restrict__ y) {
    __shared__ float red[kSnThreads / VELOX_WARP][kSnRows][M];
    const int r0 = blockIdx.x * kSnRows;
    float acc[kSnRows][M];
#pragma unroll
    for (int r = 0; r < kSnRows; ++r)
#pragma unroll
        for (int t = 0; t < M; ++t) acc[r][t] = 0.0f;
    const uint16_t* wr[kSnRows];
#pragma unroll
    for (int r = 0; r < kSnRows; ++r) wr[r] = w + (row_begin + min(r0 + r, rows - 1)) * k;
    // Four consecutive K values per step: one 8-byte weight load per row, one float4 per token.
    for (long long c = threadIdx.x; c < k / 4; c += kSnThreads) {
        float4 xv[M];
#pragma unroll
        for (int t = 0; t < M; ++t)
            xv[t] = t < m ? reinterpret_cast<const float4*>(x + t * k)[c] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (int r = 0; r < kSnRows; ++r) {
            const uint2 b = reinterpret_cast<const uint2*>(wr[r])[c];
            const float w0 = __uint_as_float(b.x << 16), w1 = __uint_as_float(b.x & 0xffff0000u);
            const float w2 = __uint_as_float(b.y << 16), w3 = __uint_as_float(b.y & 0xffff0000u);
#pragma unroll
            for (int t = 0; t < M; ++t) acc[r][t] += w0 * xv[t].x + w1 * xv[t].y + w2 * xv[t].z + w3 * xv[t].w;
        }
    }
    const int warp = threadIdx.x / VELOX_WARP, lane = threadIdx.x % VELOX_WARP;
#pragma unroll
    for (int r = 0; r < kSnRows; ++r)
#pragma unroll
        for (int t = 0; t < M; ++t) {
            const float v = velox_warp_sum(acc[r][t]);
            if (lane == 0) red[warp][r][t] = v;
        }
    __syncthreads();
    for (int i = threadIdx.x; i < kSnRows * M; i += kSnThreads) {
        const int r = i / M, t = i % M;
        if (r0 + r >= rows || t >= m) continue;
        float s = 0.0f;
        for (int wi = 0; wi < kSnThreads / VELOX_WARP; ++wi) s += red[wi][r][t];
        y[static_cast<long long>(t) * rows + r0 + r] = s;
    }
}

template <int M>
int launch_small_n(const float* x, const uint8_t* w, long long k, long long row_begin, int rows, int m, float* y) {
    linear_small_n_bf16_kernel<M><<<(rows + kSnRows - 1) / kSnRows, kSnThreads>>>(
        x, reinterpret_cast<const uint16_t*>(w), k, row_begin, rows, m, y);
    return velox_launch_status();
}
}  // namespace

VELOX_API int velox_portable_linear(const float* x, const uint8_t* w, int format, long long n, long long k,
                                    long long row_begin, int rows, int m, float* y) {
    const WDesc d{w, format, n, k};
    if (format == VELOX_FMT_BF16 && m >= 2 && m <= kSnMaxM && rows <= 4096 && k % 4 == 0 &&
        reinterpret_cast<uintptr_t>(x) % 16 == 0 && reinterpret_cast<uintptr_t>(w) % 8 == 0) {
        if (m <= 2) return launch_small_n<2>(x, w, k, row_begin, rows, m, y);
        if (m <= 4) return launch_small_n<4>(x, w, k, row_begin, rows, m, y);
        if (m <= 8) return launch_small_n<8>(x, w, k, row_begin, rows, m, y);
        return launch_small_n<16>(x, w, k, row_begin, rows, m, y);
    }
    // x rows are read as float4, so k must keep them 16-byte aligned too.
    if (m <= kGemvMaxM && k % 16 == 0 && reinterpret_cast<uintptr_t>(x) % 16 == 0) {
        if (format == VELOX_FMT_NVFP4) return launch_gemv_vec_m<VELOX_FMT_NVFP4>(x, d, row_begin, rows, m, y);
        if (format == VELOX_FMT_FP8_ROW_BF16)
            return launch_gemv_vec_m<VELOX_FMT_FP8_ROW_BF16>(x, d, row_begin, rows, m, y);
        if (format == VELOX_FMT_BF16) return launch_gemv_vec_m<VELOX_FMT_BF16>(x, d, row_begin, rows, m, y);
        if (format == VELOX_FMT_Q4 && k % 32 == 0) return launch_gemv_vec_m<VELOX_FMT_Q4>(x, d, row_begin, rows, m, y);
        if (format == VELOX_FMT_Q8) return launch_gemv_vec_m<VELOX_FMT_Q8>(x, d, row_begin, rows, m, y);
        if (format >= VELOX_FMT_GGML_BASE && format < VELOX_FMT_GGML_END)
            return launch_gemv_ggml(x, d, row_begin, rows, m, y);
    }
    if (m <= kGemvMaxM && (format == VELOX_FMT_FP8_ROW_BF16 || format == VELOX_FMT_NVFP4)) {
        constexpr int kWarps = 8;
        const dim3 grid((rows + kWarps - 1) / kWarps);
        if (format == VELOX_FMT_FP8_ROW_BF16)
            linear_gemv_kernel<VELOX_FMT_FP8_ROW_BF16><<<grid, kWarps * VELOX_WARP>>>(x, d, row_begin, rows, m, y);
        else
            linear_gemv_kernel<VELOX_FMT_NVFP4><<<grid, kWarps * VELOX_WARP>>>(x, d, row_begin, rows, m, y);
        return velox_launch_status();
    }
    // The tiled kernel needs enough output tiles to fill the GPU; skinny outputs (a few dozen rows,
    // e.g. GDN gate projections) go to the warp-per-output kernel instead.
    const long long tiles = static_cast<long long>((rows + kTile - 1) / kTile) * ((m + kTile - 1) / kTile);
    if (m >= 16 && (m + kTile - 1) / kTile <= 65535) {
        // Too few tiles to fill the GPU (skinny outputs such as GDN gate projections): split K into
        // slices written separately, then summed in a fixed order, so results stay deterministic.
        int splits = 1;
        while (tiles * splits < 256 && k / (splits * 2) >= 256) splits *= 2;
        const long long k_split = (k + splits - 1) / splits;
        const dim3 grid((rows + kTile - 1) / kTile, (m + kTile - 1) / kTile, splits);
        if (splits == 1) {
            linear_tiled_kernel<<<grid, 256>>>(x, d, row_begin, rows, m, y, k_split);
        } else {
            float* part = linear_split_scratch(static_cast<size_t>(splits) * m * rows);
            if (part == nullptr) return static_cast<int>(cudaErrorMemoryAllocation);
            linear_tiled_kernel<<<grid, 256>>>(x, d, row_begin, rows, m, part, k_split);
            const long long n = static_cast<long long>(m) * rows;
            sum_splits_kernel<<<velox_grid_1d(n, 256), 256>>>(part, splits, n, y);
        }
    } else {
        constexpr int kWarps = 8;
        const dim3 grid((rows + kWarps - 1) / kWarps, m < 65535 ? m : 65535);
        linear_kernel<<<grid, kWarps * VELOX_WARP>>>(x, d, row_begin, rows, m, y);
    }
    return velox_launch_status();
}

VELOX_API int velox_portable_embed(const uint32_t* ids, int count, const uint8_t* w, int format, long long n,
                                   long long k, long long row_begin, float* out) {
    if (count <= 0) return 0;
    if (format == VELOX_FMT_Q4 || format == VELOX_FMT_Q5 || format == VELOX_FMT_Q6 || format == VELOX_FMT_Q8) {
        const int group = format == VELOX_FMT_Q8 ? 32 : 64, high = format == VELOX_FMT_Q5 ? 8 : format == VELOX_FMT_Q6 ? 16 : 0;
        const long long rg = (k + group - 1) / group;
        const size_t smem = (rg * 32 + 15) / 16 * 16 + (rg * high + 15) / 16 * 16 + rg * 2;
        if (smem <= 48 * 1024) {
            embed_rs_kernel<<<count, 256, smem>>>(ids, WDesc{w, format, n, k}, row_begin, out);
            return velox_launch_status();
        }
    }
    const unsigned slices = static_cast<unsigned>((k + 255) / 256 < 65535 ? (k + 255) / 256 : 65535);
    embed_kernel<<<dim3(count, slices), 256>>>(ids, WDesc{w, format, n, k}, row_begin, out);
    return velox_launch_status();
}

VELOX_API int velox_portable_sigmoid_mul(float* x, const float* gate, long long len) {
    sigmoid_mul_kernel<<<velox_grid_1d(len, 256), 256>>>(x, gate, len, nullptr);
    return velox_launch_status();
}

// velox_portable_sigmoid_mul that also writes the result rounded to bf16 into xb.
VELOX_API int velox_portable_sigmoid_mul_twin(float* x, const float* gate, unsigned short* xb, long long len) {
    sigmoid_mul_kernel<<<velox_grid_1d(len, 256), 256>>>(x, gate, len, xb);
    return velox_launch_status();
}

// state_in: the rolling state before the call; the advanced state goes to `state` (may be the same buffer).
// pp: null, or device [state_in, state_out] staged by a graph step (the states are per-call values).
VELOX_API int velox_portable_conv1d_silu(const float* x, const float* w, const float* state_in, float* state, float* out,
                                         int tokens, int channels, int width, const long long* pp) {
    if (width > kMaxConvWidth) return VELOX_ERR_UNSUPPORTED;
    if (tokens <= kConvSmallTokens) {
        conv1d_silu_small_kernel<<<(channels + 127) / 128, 128>>>(x, w, state_in, state, out, tokens, channels, width,
                                                                  pp);
        return velox_launch_status();
    }
    conv1d_silu_kernel<<<velox_grid_1d(static_cast<long long>(tokens) * channels, 256), 256>>>(x, w, state_in, out,
                                                                                                tokens, channels, width,
                                                                                                pp);
    conv1d_state_kernel<<<(channels + 255) / 256, 256>>>(x, state_in, state, tokens, channels, width, pp);
    return velox_launch_status();
}

// state_in: the state before the call; the updated state goes to `state` (may be the same buffer;
// null: outputs only, per-row kernel). pp: null, or device [state_in, state_out] staged by a graph
// step; staged calls take the per-row kernel, which reads the states through it.
VELOX_API int velox_portable_gated_delta_net(const float* qkv, const float* ab, const float* a_log,
                                             const float* dt_bias, const float* state_in, float* state, float* out,
                                             int tokens, int k_heads, int v_heads, int k_dim, int v_dim,
                                             const long long* pp) {
    if (pp == nullptr && state != nullptr && tokens >= kGdnChunkMinTokens &&
        (k_dim == 32 || k_dim == 64 || k_dim == 128)) {
        if (state_in != state) {
            // The chunked kernel updates in place.
            const size_t n = static_cast<size_t>(v_heads) * v_dim * k_dim;
            if (cudaError_t e = cudaMemcpyAsync(state, state_in, n * sizeof(float), cudaMemcpyDeviceToDevice, 0))
                return static_cast<int>(e);
        }
        const int chunks = (tokens + kGdnChunk - 1) / kGdnChunk;
        static VeloxScratch scratch;
        float* prep = static_cast<float*>(scratch.get(static_cast<size_t>(v_heads) * chunks * kGdnPrep * sizeof(float)));
        if (prep == nullptr) return static_cast<int>(cudaErrorMemoryAllocation);
        // State rows per thread: 4 (64 per block) unless the heads are narrower. Qwen3.8 (48 heads,
        // 128 x 128), 1024 tokens with the prepare pass: 1381 us at 1, 996 at 2, 805 at 4.
        const int r = v_dim >= 4 * kGdnVt ? 4 : v_dim >= 2 * kGdnVt ? 2 : 1;
        const dim3 pgrid(v_heads, chunks), grid(v_heads, (v_dim + kGdnVt * r - 1) / (kGdnVt * r)), block(kGdnThreads);
#define VELOX_GDN_CHUNK_R(DK, R) \
    gated_delta_net_chunk_kernel<DK, R><<<grid, block>>>(qkv, prep, state, out, tokens, k_heads, v_heads, v_dim)
#define VELOX_GDN_CHUNK(DK)                                                                                       \
    gated_delta_net_prepare_kernel<DK><<<pgrid, block>>>(qkv, ab, a_log, dt_bias, prep, tokens, k_heads, v_heads, \
                                                         v_dim);                                                  \
    if (r == 4) VELOX_GDN_CHUNK_R(DK, 4);                                                                         \
    else if (r == 2) VELOX_GDN_CHUNK_R(DK, 2);                                                                    \
    else VELOX_GDN_CHUNK_R(DK, 1)
        if (k_dim == 32) {
            VELOX_GDN_CHUNK(32);
        } else if (k_dim == 64) {
            VELOX_GDN_CHUNK(64);
        } else {
            VELOX_GDN_CHUNK(128);
        }
#undef VELOX_GDN_CHUNK
#undef VELOX_GDN_CHUNK_R
        return velox_launch_status();
    }
    constexpr int kWarps = kGdnRowsWarps;
    // Short windows: kGdnRowsPerBlock rows per block (4 lanes each); longer: a warp per row.
    const int rows_per_block = tokens <= kGdnRowsMaxT ? kGdnRowsPerBlock : kWarps;
    const dim3 grid(v_heads, (v_dim + rows_per_block - 1) / rows_per_block), block(kWarps * VELOX_WARP);
#define VELOX_GDN(PER)                                                                                     \
    gated_delta_net_rows_kernel<PER><<<grid, block>>>(qkv, ab, a_log, dt_bias, state_in, state, out, tokens, k_heads, \
                                                      v_heads, k_dim, v_dim, pp)
    if (k_dim <= 32) VELOX_GDN(1);
    else if (k_dim <= 64) VELOX_GDN(2);
    else if (k_dim <= 128) VELOX_GDN(4);
    else if (k_dim <= 256) VELOX_GDN(8);
    else return VELOX_ERR_UNSUPPORTED;
#undef VELOX_GDN
    return velox_launch_status();
}

// Pointer arrays hold `layers` entries each: qkv, conv_out, ab, a_log, dt_bias, conv_state, ssm.
VELOX_API int velox_portable_gdn_replay(const float* const* qkv, const float* const* conv_out, const float* const* ab,
                                        const float* const* a_log, const float* const* dt_bias,
                                        float* const* conv_state, float* const* ssm, int layers, int tokens,
                                        int channels, int width, int k_heads, int v_heads, int k_dim, int v_dim) {
    if (layers <= 0) return 0;
    if (layers > kGdnReplayMaxLayers || width > kMaxConvWidth || k_dim > 256) return VELOX_ERR_UNSUPPORTED;
    GdnReplayTable host{};
    for (int l = 0; l < layers; ++l) {
        host.qkv[l] = qkv[l];
        host.conv_out[l] = conv_out[l];
        host.ab[l] = ab[l];
        host.a_log[l] = a_log[l];
        host.dt_bias[l] = dt_bias[l];
        host.conv_state[l] = conv_state[l];
        host.ssm[l] = ssm[l];
    }
    const GdnReplayTable& t = host;
    gdn_replay_conv_kernel<<<dim3((channels + 255) / 256, layers), 256>>>(t, tokens, channels, width);
    constexpr int kWarps = kGdnRowsWarps;
    const int rows_per_block = tokens <= kGdnRowsMaxT ? kGdnRowsPerBlock : kWarps;
    const dim3 grid(v_heads, (v_dim + rows_per_block - 1) / rows_per_block, layers), block(kWarps * VELOX_WARP);
#define VELOX_GDN_REPLAY(PER) \
    gdn_replay_rows_kernel<PER><<<grid, block>>>(t, tokens, k_heads, v_heads, k_dim, v_dim)
    if (k_dim <= 32) VELOX_GDN_REPLAY(1);
    else if (k_dim <= 64) VELOX_GDN_REPLAY(2);
    else if (k_dim <= 128) VELOX_GDN_REPLAY(4);
    else VELOX_GDN_REPLAY(8);
#undef VELOX_GDN_REPLAY
    return velox_launch_status();
}

// ---- Vision-tower ops ------------------------------------------------------------------------

namespace {

// Non-causal MHA: one warp per (token, head), keys visited in order with an online softmax.
template <int PER>
__global__ void attention_full_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                      const float* __restrict__ v, float* __restrict__ out, int tokens, int heads,
                                      int head_dim, float scale) {
    const int h = blockIdx.x, t = blockIdx.y, lane = threadIdx.x;
    const size_t q_off = (static_cast<size_t>(t) * heads + h) * head_dim;
    float qr[PER], acc[PER];
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int d = lane + p * VELOX_WARP;
        qr[p] = d < head_dim ? q[q_off + d] * scale : 0.0f;
        acc[p] = 0.0f;
    }
    float m = -INFINITY, l = 0.0f;
    for (int j = 0; j < tokens; ++j) {
        const size_t kv_off = (static_cast<size_t>(j) * heads + h) * head_dim;
        float s = 0.0f;
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            const int d = lane + p * VELOX_WARP;
            if (d < head_dim) s += qr[p] * k[kv_off + d];
        }
        s = velox_warp_sum(s);
        const float m_new = fmaxf(m, s);
        const float corr = expf(m - m_new), pj = expf(s - m_new);
        l = l * corr + pj;
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            const int d = lane + p * VELOX_WARP;
            if (d < head_dim) acc[p] = acc[p] * corr + pj * v[kv_off + d];
        }
        m = m_new;
    }
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int d = lane + p * VELOX_WARP;
        if (d < head_dim) out[q_off + d] = acc[p] / l;
    }
}

__global__ void layer_norm_kernel(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
                                  float* __restrict__ y, int dim, float eps) {
    __shared__ float scratch[VELOX_WARP];
    const float* xr = x + static_cast<size_t>(blockIdx.x) * dim;
    float* yr = y + static_cast<size_t>(blockIdx.x) * dim;
    float s = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) s += xr[i];
    const float mean = velox_block_sum(s, scratch) / dim;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float d = xr[i] - mean;
        ss += d * d;
    }
    const float inv = rsqrtf(velox_block_sum(ss, scratch) / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) yr[i] = (xr[i] - mean) * inv * w[i] + b[i];
}

__global__ void add_bias_kernel(float* __restrict__ x, const float* __restrict__ bias, long long len, int dim) {
    for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < len;
         i += static_cast<long long>(gridDim.x) * blockDim.x)
        x[i] += bias[i % dim];
}

__global__ void gelu_kernel(float* __restrict__ x, long long len, int erf_kind) {
    for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < len;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const float a = x[i];
        x[i] = erf_kind ? 0.5f * a * (1.0f + erff(a * 0.70710678118654752f))
                        : 0.5f * a * (1.0f + tanhf(0.79788456080286536f * (a + 0.044715f * a * a * a)));
    }
}

// layout 0: interleaved (axis i % axes, freq theta^(-2i/R)); 1: blocked (see RopeLayout).
// One block per token: the token's angles (double, slow on consumer GPUs) are computed once per pair
// into shared memory, then every head applies them.
__global__ void rope_positions_kernel(float* __restrict__ x, const uint32_t* __restrict__ pos, int tokens, int heads,
                                      int head_dim, int rotary_dim, float theta, int axes, int layout) {
    extern __shared__ float cs[];  // [half] cos, then [half] sin
    const int half = rotary_dim / 2, t = blockIdx.x;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        int axis;
        double freq;
        if (layout == 0) {
            axis = i % axes;
            freq = pow(static_cast<double>(theta), -2.0 * i / rotary_dim);
        } else {
            const int per = half / axes;
            axis = i / per;
            freq = pow(static_cast<double>(theta), -2.0 * (i % per) / (static_cast<double>(rotary_dim) / axes));
        }
        double s, c;
        sincos(static_cast<double>(pos[static_cast<size_t>(axis) * tokens + t]) * freq, &s, &c);
        cs[i] = static_cast<float>(c);
        cs[half + i] = static_cast<float>(s);
    }
    __syncthreads();
    for (int p = threadIdx.x; p < heads * half; p += blockDim.x) {
        const int h = p / half, i = p % half;
        float* base = x + (static_cast<size_t>(t) * heads + h) * head_dim;
        const float a = base[i], b = base[i + half], c = cs[i], s = cs[half + i];
        base[i] = a * c - b * s;
        base[i + half] = a * s + b * c;
    }
}

}  // namespace

VELOX_API int velox_portable_attention_full(const float* q, const float* k, const float* v, float* out, int tokens,
                                            int heads, int head_dim) {
    if (tokens > 65535) return VELOX_ERR_UNSUPPORTED;
    const dim3 grid(heads, tokens);
    const float scale = rsqrtf(static_cast<float>(head_dim));
    if (head_dim <= 32) attention_full_kernel<1><<<grid, VELOX_WARP>>>(q, k, v, out, tokens, heads, head_dim, scale);
    else if (head_dim <= 64) attention_full_kernel<2><<<grid, VELOX_WARP>>>(q, k, v, out, tokens, heads, head_dim, scale);
    else if (head_dim <= 128) attention_full_kernel<4><<<grid, VELOX_WARP>>>(q, k, v, out, tokens, heads, head_dim, scale);
    else if (head_dim <= 256) attention_full_kernel<8><<<grid, VELOX_WARP>>>(q, k, v, out, tokens, heads, head_dim, scale);
    else return VELOX_ERR_UNSUPPORTED;
    return velox_launch_status();
}

VELOX_API int velox_portable_layer_norm(const float* x, const float* w, const float* b, float* y, int rows, int dim,
                                        float eps) {
    layer_norm_kernel<<<rows, 256>>>(x, w, b, y, dim, eps);
    return velox_launch_status();
}

VELOX_API int velox_portable_add_bias(float* x, const float* bias, long long len, int dim) {
    add_bias_kernel<<<velox_grid_1d(len, 256), 256>>>(x, bias, len, dim);
    return velox_launch_status();
}

VELOX_API int velox_portable_gelu(float* x, long long len, int erf_kind) {
    gelu_kernel<<<velox_grid_1d(len, 256), 256>>>(x, len, erf_kind);
    return velox_launch_status();
}

VELOX_API int velox_portable_rope_positions(float* x, const uint32_t* pos, int tokens, int heads, int head_dim,
                                            int rotary_dim, float theta, int axes, int layout) {
    if (tokens <= 0) return 0;
    rope_positions_kernel<<<tokens, 256, rotary_dim * sizeof(float)>>>(x, pos, tokens, heads, head_dim, rotary_dim,
                                                                       theta, axes, layout);
    return velox_launch_status();
}

// ---- MoE experts (Backend::moe_experts) ----
//
// Pairs p = r * k + j (row r, its j-th expert) are grouped by expert with a counting sort, then each
// expert runs tiled GEMMs over its gathered rows, so its weights are read once per 64 rows whether
// they sit in a device slot or in mapped system RAM. Every pair writes its own scratch row and the
// final sum runs over j in order, so results do not depend on the grouping order.
namespace {

struct ExpertDesc {
    const uint8_t* records;
    const uint8_t* slots;
    const float* map;  // i32 bits: slot of expert e, or -1
    long long record_bytes, slot_bytes;
    int gate_fmt, up_fmt, down_fmt;
    long long gate_off, up_off, down_off;
    int width, hidden;
};

__device__ __forceinline__ const uint8_t* expert_record(const ExpertDesc& d, int e) {
    const int slot = static_cast<int>(__float_as_uint(d.map[e]));
    return slot >= 0 ? d.slots + static_cast<long long>(slot) * d.slot_bytes
                     : d.records + static_cast<long long>(e) * d.record_bytes;
}

__global__ void moe_count_kernel(const float* __restrict__ ids, int pairs, int experts, int* __restrict__ counts) {
    for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < pairs; p += gridDim.x * blockDim.x) {
        const unsigned e = __float_as_uint(ids[p]);
        if (e < static_cast<unsigned>(experts)) atomicAdd(counts + e, 1);
    }
}

// Single thread: offsets[e] = sum of counts below e; cursors reset.
__global__ void moe_scan_kernel(const int* __restrict__ counts, int* __restrict__ offsets, int* __restrict__ cursor,
                                int experts) {
    if (threadIdx.x != 0) return;
    int acc = 0;
    for (int e = 0; e < experts; ++e) {
        offsets[e] = acc;
        cursor[e] = 0;
        acc += counts[e];
    }
    offsets[experts] = acc;
}

__global__ void moe_fill_kernel(const float* __restrict__ ids, int pairs, int experts, const int* __restrict__ offsets,
                                int* __restrict__ cursor, int* __restrict__ list) {
    for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < pairs; p += gridDim.x * blockDim.x) {
        const unsigned e = __float_as_uint(ids[p]);
        if (e < static_cast<unsigned>(experts)) list[offsets[e] + atomicAdd(cursor + e, 1)] = p;
    }
}

// Gate and up of expert blockIdx.y for output rows [64 blockIdx.x, +64) over its pairs, 64 at a time:
// h[p][i] = silu(gate_i . x[p / k]) * (up_i . x[p / k]).
__global__ void __launch_bounds__(256)
moe_up_kernel(const float* __restrict__ x, ExpertDesc d, const int* __restrict__ offsets,
              const int* __restrict__ list, int k, float* __restrict__ h) {
    const int e = blockIdx.y;
    const int begin = offsets[e], count = offsets[e + 1] - begin;
    if (count == 0) return;
    const uint8_t* rec = expert_record(d, e);
    const WDesc wg{rec + d.gate_off, d.gate_fmt, d.width, d.hidden};
    const WDesc wu{rec + d.up_off, d.up_fmt, d.width, d.hidden};
    __shared__ float Xs[kTileK][kTile + 4];
    __shared__ float Gs[kTileK][kTile + 4];
    __shared__ float Us[kTileK][kTile + 4];
    const int t = threadIdx.x, n0 = blockIdx.x * kTile;
    const int lr = t / 4, lk = (t % 4) * 4;
    const int ty = t / 16, tx = t % 16;
    for (int p0 = 0; p0 < count; p0 += kTile) {
        float ag[4][4] = {}, au[4][4] = {};
        const int xp = p0 + lr < count ? list[begin + p0 + lr] : -1;
        for (long long k0 = 0; k0 < d.hidden; k0 += kTileK) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const long long kk = k0 + lk + i;
                const int wn = n0 + lr;
                const bool kin = kk < d.hidden;
                Xs[lk + i][lr] = (xp >= 0 && kin) ? x[static_cast<size_t>(xp / k) * d.hidden + kk] : 0.0f;
                Gs[lk + i][lr] = (wn < d.width && kin) ? weight_at(wg, wn, kk) : 0.0f;
                Us[lk + i][lr] = (wn < d.width && kin) ? weight_at(wu, wn, kk) : 0.0f;
            }
            __syncthreads();
#pragma unroll
            for (int kk = 0; kk < kTileK; ++kk) {
                float a[4], g[4], u[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    a[i] = Xs[kk][ty * 4 + i];
                    g[i] = Gs[kk][tx * 4 + i];
                    u[i] = Us[kk][tx * 4 + i];
                }
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        ag[i][j] += a[i] * g[j];
                        au[i][j] += a[i] * u[j];
                    }
            }
            __syncthreads();
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (p0 + ty * 4 + i >= count) continue;
            const int p = list[begin + p0 + ty * 4 + i];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = n0 + tx * 4 + j;
                if (c < d.width) {
                    const float g = ag[i][j];
                    h[static_cast<size_t>(p) * d.width + c] = g / (1.0f + expf(-g)) * au[i][j];
                }
            }
        }
    }
}

// Down of expert blockIdx.y for output columns [64 blockIdx.x, +64): o[p][c] = down_c . h[p].
__global__ void __launch_bounds__(256)
moe_down_kernel(const float* __restrict__ h, ExpertDesc d, const int* __restrict__ offsets,
                const int* __restrict__ list, float* __restrict__ o) {
    const int e = blockIdx.y;
    const int begin = offsets[e], count = offsets[e + 1] - begin;
    if (count == 0) return;
    const WDesc wd{expert_record(d, e) + d.down_off, d.down_fmt, d.hidden, d.width};
    __shared__ float Xs[kTileK][kTile + 4];
    __shared__ float Ws[kTileK][kTile + 4];
    const int t = threadIdx.x, n0 = blockIdx.x * kTile;
    const int lr = t / 4, lk = (t % 4) * 4;
    const int ty = t / 16, tx = t % 16;
    for (int p0 = 0; p0 < count; p0 += kTile) {
        float acc[4][4] = {};
        const int xp = p0 + lr < count ? list[begin + p0 + lr] : -1;
        for (long long k0 = 0; k0 < d.width; k0 += kTileK) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const long long kk = k0 + lk + i;
                const int wn = n0 + lr;
                const bool kin = kk < d.width;
                Xs[lk + i][lr] = (xp >= 0 && kin) ? h[static_cast<size_t>(xp) * d.width + kk] : 0.0f;
                Ws[lk + i][lr] = (wn < d.hidden && kin) ? weight_at(wd, wn, kk) : 0.0f;
            }
            __syncthreads();
#pragma unroll
            for (int kk = 0; kk < kTileK; ++kk) {
                float a[4], b[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    a[i] = Xs[kk][ty * 4 + i];
                    b[i] = Ws[kk][tx * 4 + i];
                }
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
            }
            __syncthreads();
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (p0 + ty * 4 + i >= count) continue;
            const int p = list[begin + p0 + ty * 4 + i];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = n0 + tx * 4 + j;
                if (c < d.hidden) o[static_cast<size_t>(p) * d.hidden + c] = acc[i][j];
            }
        }
    }
}

__global__ void moe_combine_kernel(const float* __restrict__ o, const float* __restrict__ w, int rows, int k,
                                   int hidden, float* __restrict__ y) {
    const long long n = static_cast<long long>(rows) * hidden;
    for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const long long r = i / hidden, c = i % hidden;
        float acc = 0.0f;
        for (int j = 0; j < k; ++j) acc += w[r * k + j] * o[(r * k + j) * hidden + c];
        y[i] = acc;
    }
}

}  // namespace

// f32 slots of scratch velox_portable_moe_experts needs.
VELOX_API long long velox_portable_moe_scratch(int experts, int pairs, int width, int hidden) {
    return 3LL * experts + 1 + pairs + static_cast<long long>(pairs) * (width + hidden);
}

VELOX_API int velox_portable_moe_experts(const float* x, const float* ids, const float* w, const uint8_t* records,
                                         const uint8_t* slots, const float* map, long long record_bytes,
                                         long long slot_bytes, int experts,
                                         int gate_fmt, long long gate_off, int up_fmt, long long up_off, int down_fmt,
                                         long long down_off, int width, int hidden, int rows, int k, float* y,
                                         float* scratch) {
    const int pairs = rows * k;
    int* counts = reinterpret_cast<int*>(scratch);
    int* offsets = counts + experts;
    int* cursor = offsets + experts + 1;
    int* list = cursor + experts;
    float* h = scratch + 3LL * experts + 1 + pairs;
    float* o = h + static_cast<size_t>(pairs) * width;
    if (cudaError_t e = cudaMemsetAsync(counts, 0, sizeof(int) * experts, 0)) return static_cast<int>(e);
    moe_count_kernel<<<velox_grid_1d(pairs, 256), 256>>>(ids, pairs, experts, counts);
    moe_scan_kernel<<<1, 32>>>(counts, offsets, cursor, experts);
    moe_fill_kernel<<<velox_grid_1d(pairs, 256), 256>>>(ids, pairs, experts, offsets, cursor, list);
    const ExpertDesc d{records, slots,  map,     record_bytes, slot_bytes, gate_fmt, up_fmt,
                       down_fmt, gate_off, up_off, down_off, width,      hidden};
    moe_up_kernel<<<dim3((width + kTile - 1) / kTile, experts), 256>>>(x, d, offsets, list, k, h);
    moe_down_kernel<<<dim3((hidden + kTile - 1) / kTile, experts), 256>>>(h, d, offsets, list, o);
    moe_combine_kernel<<<velox_grid_1d(static_cast<long long>(rows) * hidden, 256), 256>>>(o, w, rows, k, hidden, y);
    return velox_launch_status();
}
