// RTX 5060 Ti GGML block linears (GGUF imports, ggml_<type> / ggml_block_v1) at decode and verify
// widths, llama.cpp's MMVQ design: activations quantized once per call to int8 per 32 values
// (ActQuant::Int8G32, llama.cpp's q8_1), each lane decoding one 32-value sub-block of a weight row
// at a time into eight int8x4 words plus its scales (per 8, 16 or 32 values) and, for the types
// with a minimum or a delta, offsets, then dp4a against every token row. A weight byte is read once
// for up to 16 tokens, and the per-element work of main's portable kernel (which decodes each
// value separately, in f32) becomes a few integer ops per 4 values.
//
// Sub-block decoding follows ggml's dequantize_row_* (velox_core::ggml is the reference; main's
// portable ggml_value is the same decoder per element). Tables: ggml_tables.cuh (llama.cpp, MIT).
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdlib.h>

#include "ggml_tables.cuh"
#include "velox_common.cuh"

namespace {

// velox_core::ggml::GgmlType::ALL order (main's VELOX_FMT_GGML_BASE + T).
enum : int { kQ2K = 0, kQ4K, kIq1M, kIq2Xxs, kIq2Xs, kIq2S, kIq3Xxs, kIq3S, kIq4Xs, kQ2_0, kIq4Nl, kQ6K, kQ8_0, kIq1S };

__host__ __device__ constexpr int block_bytes(int t) {
    return t == kQ2K      ? 84
           : t == kQ4K    ? 144
           : t == kIq1M   ? 56
           : t == kIq2Xxs ? 66
           : t == kIq2Xs  ? 74
           : t == kIq2S   ? 82
           : t == kIq3Xxs ? 98
           : t == kIq3S   ? 110
           : t == kQ2_0   ? 18
           : t == kIq4Nl  ? 18
           : t == kIq1S   ? 50
                          : 136;
}

// Values per stored block (the K granularity) and 32-value sub-blocks per block.
__host__ __device__ constexpr int block_values(int t) { return t == kQ2_0 ? 64 : t == kIq4Nl ? 32 : 256; }
template <int T>
__host__ __device__ constexpr int sub_per_block() {
    return block_values(T) / 32;
}

// Scale groups per 32-value sub-block (32 / G values each) and whether values carry an offset
// (value = scale * code + offset).
template <int T>
struct Traits {
    static constexpr int G = (T == kQ2K || T == kIq2Xs || T == kIq2S) ? 2 : T == kIq1M ? 4 : 1;
    static constexpr bool kOff = T == kQ2K || T == kQ4K || T == kIq1M || T == kQ2_0 || T == kIq1S;
};

template <int T>
struct Sub {
    int w[8];  // 32 int8 codes in value order
    float s[Traits<T>::G];
    float o[Traits<T>::kOff ? Traits<T>::G : 1];
};

__device__ __forceinline__ float f16_at(const uint8_t* p) {
    return __half2float(__ushort_as_half(*reinterpret_cast<const unsigned short*>(p)));
}
__device__ __forceinline__ uint32_t u16_at(const uint8_t* p) { return *reinterpret_cast<const unsigned short*>(p); }
// 4 bytes at a 2-byte aligned address.
__device__ __forceinline__ uint32_t u32_at2(const uint8_t* p) { return u16_at(p) | (u16_at(p + 2) << 16); }

// Byte b of the 4 magnitudes negated where sign bit b of `sb` (bits 0..3) is set. Grid magnitudes
// are 1..127, so a negated byte (~g + 1) never carries into the next one and a plain add does.
__device__ __forceinline__ int signed4(uint32_t mag, uint32_t sb) {
    const uint32_t b = (sb * 0x00204081u) & 0x01010101u;
    return static_cast<int>((mag ^ (b * 0xffu)) + b);
}

// llama.cpp's get_int_from_table_16 (MIT): the 8 nibbles of q looked up in a 16-entry int8 table;
// .x holds the low nibbles' values (byte order), .y the high nibbles'.
__device__ __forceinline__ int2 table16(uint32_t q, const uint32_t (&t)[4]) {
    const uint32_t sel = 0x32103210u | ((q & 0x88888888u) >> 1);
    uint32_t tmp[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t sh = 16 * i;
        const uint32_t lo = __byte_perm(t[0], t[1], q >> sh);
        const uint32_t hi = __byte_perm(t[2], t[3], q >> sh);
        tmp[i] = __byte_perm(lo, hi, sel >> sh);
    }
    return make_int2(static_cast<int>(__byte_perm(tmp[0], tmp[1], 0x6420)),
                     static_cast<int>(__byte_perm(tmp[0], tmp[1], 0x7531)));
}

// Sub-block ib (0..7) of super-block x.
template <int T>
__device__ __forceinline__ Sub<T> decode(const uint8_t* __restrict__ x, int ib) {
    Sub<T> r;
    if constexpr (T == kIq3S) {  // d | qs[64] | qh[8] | signs[32] | scales[4]
        const uint32_t s = x[106 + ib / 2];
        r.s[0] = f16_at(x) * static_cast<float>(1 + 2 * (ib % 2 ? s >> 4 : s & 0xf));
        const uint32_t qh = x[66 + ib];
        const uint8_t* qs = x + 2 + 8 * ib;
        const uint32_t sg = u32_at2(x + 74 + 4 * ib);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t a = qs[2 * l] | ((qh << (8 - 2 * l)) & 256);
            const uint32_t b = qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256);
            const uint32_t sb = sg >> (8 * l);
            r.w[2 * l] = signed4(__ldg(&kIq3SGrid[a]), sb & 0xf);
            r.w[2 * l + 1] = signed4(__ldg(&kIq3SGrid[b]), (sb >> 4) & 0xf);
        }
    } else if constexpr (T == kIq3Xxs) {  // d | qs[64] | 8 x u32 (4 x 7 sign bits, scale)
        const uint32_t aux = u32_at2(x + 66 + 4 * ib);
        r.s[0] = f16_at(x) * (0.5f + static_cast<float>(aux >> 28)) * 0.5f;
        const uint8_t* qs = x + 2 + 8 * ib;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t sb = __ldg(&kKsignsIq2xs[(aux >> (7 * l)) & 127]);
            r.w[2 * l] = signed4(__ldg(&kIq3XxsGridG[qs[2 * l]]), sb & 0xf);
            r.w[2 * l + 1] = signed4(__ldg(&kIq3XxsGridG[qs[2 * l + 1]]), sb >> 4);
        }
    } else if constexpr (T == kIq4Xs) {  // d | scales_h u16 | scales_l[4] | qs[128]
        const int ls = static_cast<int>((x[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) |
                       (static_cast<int>((u16_at(x + 2) >> (2 * ib)) & 3) << 4);
        r.s[0] = f16_at(x) * static_cast<float>(ls - 32);
        // kValuesIq4nl packed little-endian, four values per word.
        const uint32_t tab[4] = {0xbfad9881u, 0xf6eaddcfu, 0x26190d01u, 0x71594535u};
        const uint8_t* qs = x + 8 + 16 * ib;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int2 v = table16(u32_at2(qs + 4 * i), tab);
            r.w[i] = v.x;      // values 4i .. 4i+3 (low nibbles)
            r.w[4 + i] = v.y;  // values 16+4i .. (high nibbles)
        }
    } else if constexpr (T == kIq4Nl) {  // 32-value block: d | qs[16]; ib is 0
        r.s[0] = f16_at(x);
        const uint32_t tab[4] = {0xbfad9881u, 0xf6eaddcfu, 0x26190d01u, 0x71594535u};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int2 v = table16(u32_at2(x + 2 + 4 * i), tab);
            r.w[i] = v.x;      // values 4i .. 4i+3 (low nibbles)
            r.w[4 + i] = v.y;  // values 16+4i .. (high nibbles)
        }
    } else if constexpr (T == kQ4K) {  // d | dmin | scales[12] | qs[128]
        const int c = ib / 2, hi = ib % 2;
        const uint8_t* q = x + 4;
        uint32_t sc, m;
        if (ib < 4) {
            sc = q[ib] & 63, m = q[ib + 4] & 63;
        } else {
            sc = (q[ib + 4] & 0xf) | ((q[ib - 4] >> 6) << 4);
            m = (q[ib + 4] >> 4) | ((q[ib] >> 6) << 4);
        }
        r.s[0] = f16_at(x) * static_cast<float>(sc);
        r.o[0] = -f16_at(x + 2) * static_cast<float>(m);
        const uint32_t* qs = reinterpret_cast<const uint32_t*>(x + 16 + 32 * c);  // 16-byte aligned
#pragma unroll
        for (int i = 0; i < 8; ++i) r.w[i] = static_cast<int>((qs[i] >> (4 * hi)) & 0x0f0f0f0fu);
    } else if constexpr (T == kQ2K) {  // scales[16] | qs[64] | d | dmin
        const int n = ib / 4, sh = 2 * (ib % 4);
        const float d = f16_at(x + 80), dmin = f16_at(x + 82);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint32_t sc = x[n * 8 + (ib % 4) * 2 + h];
            r.s[h] = d * static_cast<float>(sc & 0xf);
            r.o[h] = -dmin * static_cast<float>(sc >> 4);
        }
        const uint32_t* qs = reinterpret_cast<const uint32_t*>(x + 16 + 32 * n);  // 4-byte aligned
#pragma unroll
        for (int i = 0; i < 8; ++i) r.w[i] = static_cast<int>((qs[i] >> sh) & 0x03030303u);
    } else if constexpr (T == kIq2Xxs) {  // d | per 32: 4 grid bytes, u32 (4 x 7 sign bits, scale)
        const uint32_t a1 = u32_at2(x + 6 + 8 * ib);
        r.s[0] = f16_at(x) * (0.5f + static_cast<float>(a1 >> 28)) * 0.25f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint64_t g = __ldg(&kIq2XxsGrid[x[2 + 8 * ib + l]]);
            const uint32_t sb = __ldg(&kKsignsIq2xs[(a1 >> (7 * l)) & 127]);
            r.w[2 * l] = signed4(static_cast<uint32_t>(g), sb & 0xf);
            r.w[2 * l + 1] = signed4(static_cast<uint32_t>(g >> 32), sb >> 4);
        }
    } else if constexpr (T == kIq2Xs) {  // d | qs u16[32] | scales[8]
        const uint32_t s = x[66 + ib];
        const float d = f16_at(x);
        r.s[0] = d * (0.5f + static_cast<float>(s & 0xf)) * 0.25f;
        r.s[1] = d * (0.5f + static_cast<float>(s >> 4)) * 0.25f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t q = u16_at(x + 2 + 2 * (4 * ib + l));
            const uint64_t g = __ldg(&kIq2XsGrid[q & 511]);
            const uint32_t sb = __ldg(&kKsignsIq2xs[q >> 9]);
            r.w[2 * l] = signed4(static_cast<uint32_t>(g), sb & 0xf);
            r.w[2 * l + 1] = signed4(static_cast<uint32_t>(g >> 32), sb >> 4);
        }
    } else if constexpr (T == kIq2S) {  // d | qs[32] | signs[32] | qh[8] | scales[8]
        const uint32_t s = x[74 + ib], qh = x[66 + ib];
        const float d = f16_at(x);
        r.s[0] = d * (0.5f + static_cast<float>(s & 0xf)) * 0.25f;
        r.s[1] = d * (0.5f + static_cast<float>(s >> 4)) * 0.25f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t idx = x[2 + 4 * ib + l] | ((qh << (8 - 2 * l)) & 0x300);
            const uint64_t g = __ldg(&kIq2SGrid[idx]);
            const uint32_t sb = x[34 + 4 * ib + l];
            r.w[2 * l] = signed4(static_cast<uint32_t>(g), sb & 0xf);
            r.w[2 * l + 1] = signed4(static_cast<uint32_t>(g >> 32), sb >> 4);
        }
    } else if constexpr (T == kQ2_0) {  // 64-value block: d | qs[16]; ib (0, 1) is a half
        // value = d * q - d with 2-bit q, four consecutive values per byte: one byte per word.
        const float d = f16_at(x);
        r.s[0] = d;
        r.o[0] = -d;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t two = u16_at(x + 2 + 8 * ib + 2 * i);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const uint32_t b = (two >> (8 * h)) & 0xffu;
                r.w[2 * i + h] = static_cast<int>((b & 0x03u) | ((b & 0x0cu) << 6) | ((b & 0x30u) << 12) |
                                                  ((b & 0xc0u) << 18));
            }
        }
    } else if constexpr (T == kIq1S) {  // d | qs[32] | qh u16[8]: one scale and one delta per 32 values
        const uint32_t qh = u16_at(x + 34 + 2 * ib);
        const float dl = f16_at(x) * static_cast<float>(2 * ((qh >> 12) & 7) + 1);
        r.s[0] = dl;
        r.o[0] = dl * (qh & 0x8000 ? -0.125f : 0.125f);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t idx = x[2 + 4 * ib + l] | (((qh >> (3 * l)) & 7) << 8);
            const uint64_t g = __ldg(&kIq1SGrid[idx]);  // int8 values -1, 0, 1
            r.w[2 * l] = static_cast<int>(static_cast<uint32_t>(g));
            r.w[2 * l + 1] = static_cast<int>(static_cast<uint32_t>(g >> 32));
        }
    } else {  // IQ1_M: qs[32] | qh[16] | scales[8]
        const uint32_t s0 = u16_at(x + 48), s1 = u16_at(x + 50), s2 = u16_at(x + 52), s3 = u16_at(x + 54);
        const uint32_t dbits = (s0 >> 12) | ((s1 >> 8) & 0x00f0) | ((s2 >> 4) & 0x0f00) | (s3 & 0xf000);
        const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(dbits)));
        const uint32_t w = u16_at(x + 48 + 2 * (ib / 2));
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const float dl = d * static_cast<float>(2 * ((w >> (6 * (ib % 2) + (l < 2 ? 0 : 3))) & 7) + 1);
            const uint32_t qh = x[32 + 2 * ib + l / 2];
            const uint32_t idx = x[4 * ib + l] | ((qh << (l % 2 ? 4 : 8)) & 0x700);
            const float delta = qh & (l % 2 ? 0x80 : 0x08) ? -0.125f : 0.125f;
            const uint64_t g = __ldg(&kIq1SGrid[idx]);  // int8 values -1, 0, 1
            r.w[2 * l] = static_cast<int>(static_cast<uint32_t>(g));
            r.w[2 * l + 1] = static_cast<int>(static_cast<uint32_t>(g >> 32));
            r.s[l] = dl;
            r.o[l] = dl * delta;
        }
    }
    return r;
}

// sum over the sub-block of value * code (codes of one token's 32 activations).
template <int T>
__device__ __forceinline__ float sub_dot(const Sub<T>& s, const int (&a)[8]) {
    constexpr int G = Traits<T>::G, P = 8 / G;
    float acc = 0.0f;
#pragma unroll
    for (int g = 0; g < G; ++g) {
        int d = 0;
#pragma unroll
        for (int i = 0; i < P; ++i) d = __dp4a(s.w[g * P + i], a[g * P + i], d);
        acc += s.s[g] * static_cast<float>(d);
        if constexpr (Traits<T>::kOff) {
            int z = 0;
#pragma unroll
            for (int i = 0; i < P; ++i) z = __dp4a(0x01010101, a[g * P + i], z);
            acc += s.o[g] * static_cast<float>(z);
        }
    }
    return acc;
}

constexpr int kWarps = 4;

// Output store: mode 0 writes v; mode 1 (the up projection of a SwiGLU whose gate projection
// already wrote y) writes silu(y) * v.
__device__ __forceinline__ void put(float* y, float v, int mode) {
    if (mode) {
        const float gte = *y;
        v *= gte / (1.0f + __expf(-gte));
    }
    *y = v;
}

// Warp per R output rows; lanes stride over the row's 32-value sub-blocks.
template <int T, int M, int R>
__global__ void __launch_bounds__(kWarps * VELOX_WARP)
    ggml_mmvq_kernel(const uint8_t* __restrict__ w, long long row_bytes, long long row_begin, int rows, int subs,
                     const int8_t* __restrict__ xq, const float* __restrict__ xd, int m, float* __restrict__ y,
                     long long ldy, int mode) {
    constexpr int BB = block_bytes(T), SPB = sub_per_block<T>();
    const int lane = threadIdx.x % VELOX_WARP;
    const int r0 = (blockIdx.x * kWarps + threadIdx.x / VELOX_WARP) * R;
    if (r0 >= rows) return;
    float acc[R][M];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < M; ++t) acc[r][t] = 0.0f;
    const long long k = static_cast<long long>(subs) * 32;
    for (int u = lane; u < subs; u += VELOX_WARP) {
        Sub<T> s[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = min(r0 + r, rows - 1);
            s[r] = decode<T>(w + (row_begin + row) * row_bytes + static_cast<long long>(u / SPB) * BB, u % SPB);
        }
#pragma unroll
        for (int t = 0; t < M; ++t) {
            if (t >= m) break;
            const int4* ap = reinterpret_cast<const int4*>(xq + t * k + static_cast<long long>(u) * 32);
            const int4 a0 = __ldg(ap), a1 = __ldg(ap + 1);
            const int a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float dx = __ldg(xd + static_cast<long long>(t) * subs + u);
#pragma unroll
            for (int r = 0; r < R; ++r) acc[r][t] += dx * sub_dot<T>(s[r], a);
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < M; ++t) {
            if (t >= m) break;
            const float v = velox_warp_sum(acc[r][t]);
            if (lane == 0 && r0 + r < rows) put(y + t * ldy + r0 + r, v, mode);
        }
}

template <int T, int M, int R>
int launch_r(const uint8_t* w, long long row_bytes, long long rb, int rows, int subs, const int8_t* xq,
             const float* xd, int m, float* y, long long ldy, int mode) {
    const int warps = (rows + R - 1) / R;
    ggml_mmvq_kernel<T, M, R><<<(warps + kWarps - 1) / kWarps, kWarps * VELOX_WARP>>>(w, row_bytes, rb, rows, subs,
                                                                                     xq, xd, m, y, ldy, mode);
    return velox_launch_status();
}

// Rows per warp: VELOX_GGML_R (1, 2 or 4) for tuning, else the default below.
int rows_per_warp(int m) {
    static const int forced = [] {
        const char* e = getenv("VELOX_GGML_R");
        return e ? atoi(e) : 0;
    }();
    if (forced == 1 || forced == 2 || forced == 4) return forced;
    return m <= 2 ? 2 : 4;
}

template <int T, int M>
int launch_m(const uint8_t* w, long long row_bytes, long long rb, int rows, int subs, const int8_t* xq,
             const float* xd, int m, float* y, long long ldy, int mode) {
    // Rows per warp share each activation load (from L1): at 8 and 16 tokens the activation reads,
    // not the weights, bound a one-row warp.
    switch (rows_per_warp(m)) {
        case 1: return launch_r<T, M, 1>(w, row_bytes, rb, rows, subs, xq, xd, m, y, ldy, mode);
        case 2: return launch_r<T, M, 2>(w, row_bytes, rb, rows, subs, xq, xd, m, y, ldy, mode);
        default: return launch_r<T, M, 4>(w, row_bytes, rb, rows, subs, xq, xd, m, y, ldy, mode);
    }
}

template <int T>
int launch_t(const uint8_t* w, long long row_bytes, long long rb, int rows, int subs, const int8_t* xq,
             const float* xd, int m, float* y, long long ldy, int mode) {
    if (m == 1) return launch_m<T, 1>(w, row_bytes, rb, rows, subs, xq, xd, m, y, ldy, mode);
    if (m == 2) return launch_m<T, 2>(w, row_bytes, rb, rows, subs, xq, xd, m, y, ldy, mode);
    if (m <= 4) return launch_m<T, 4>(w, row_bytes, rb, rows, subs, xq, xd, m, y, ldy, mode);
    if (m <= 8) return launch_m<T, 8>(w, row_bytes, rb, rows, subs, xq, xd, m, y, ldy, mode);
    return launch_m<T, 16>(w, row_bytes, rb, rows, subs, xq, xd, m, y, ldy, mode);
}

// ---- Tensor-core path (5..16 tokens): mma.m16n8k32 / k16 on int8 ----------------------------

// The activation blob of velox_5060_ggml_act for [m, k]: natural codes, scales and per-16 sums
// (the dp4a kernel), then the same in mma fragment order per (sub-block u, 8-token tile n): codes
// as 32 lanes x int2 (lane (g, t): token 8n + g, words t and t + 4), scales as 8 floats and sums as
// 16 floats (token 8n + j, halves 0 and 1), zero past m, so a lane's operands are one coalesced
// load each.
struct ActLayout {
    long long codes, d, sums, frag, fragp, fd, fs, bytes;
    int tiles;
    __host__ __device__ ActLayout(int m, long long k) {
        const long long subs = k / 32;
        tiles = (m + 7) / 8;
        auto up = [](long long v) { return (v + 255) / 256 * 256; };
        codes = 0;
        d = up(static_cast<long long>(m) * k);
        sums = d + up(m * subs * 4);
        frag = sums + up(m * subs * 8);
        fragp = frag + up(subs * tiles * 256);
        fd = fragp + up(subs * tiles * 256);
        fs = fd + up(subs * tiles * 32);
        bytes = fs + up(subs * tiles * 64);
    }
};

// One warp per (token, sub-block) for tiles * 8 tokens: the natural codes, scale and per-16 sums
// (tokens < m) and the same in the fragment orders of the tensor-core path, entry e = u * tiles + n
// for token 8n + g. `frag` puts words t and t + 4 in lane t's k-slots (the natural order); `fragp`
// words 2t and 2t + 1 (the paired order of the types with 32-value scales). Tokens past m are zero.
__global__ void quantize_act_kernel(const float* __restrict__ x, int m, int subs, int tiles,
                                    int8_t* __restrict__ q, float* __restrict__ d, float* __restrict__ sums,
                                    int2* __restrict__ frag, int2* __restrict__ fragp, float* __restrict__ fd,
                                    float* __restrict__ fs) {
    const long long gi = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) / VELOX_WARP;
    const int lane = threadIdx.x % VELOX_WARP;
    if (gi >= static_cast<long long>(tiles) * 8 * subs) return;
    const int tok = static_cast<int>(gi / subs), u = static_cast<int>(gi % subs);
    const bool live = tok < m;
    const long long k = static_cast<long long>(subs) * 32;
    const float v = live ? x[tok * k + static_cast<long long>(u) * 32 + lane] : 0.0f;
    const float amax = velox_warp_max(fabsf(v));
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    const int c = __float2int_rn(v * inv);
    int h = c;
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) h += __shfl_xor_sync(0xffffffffu, h, o);  // within each 16 lanes
    const float dv = amax / 127.0f;
    // Word w (codes 4w .. 4w + 3) assembled at lane 4w.
    const uint32_t b = static_cast<uint32_t>(c) & 0xffu;
    uint32_t word = b | (__shfl_down_sync(0xffffffffu, b, 1) << 8) | (__shfl_down_sync(0xffffffffu, b, 2) << 16) |
                    (__shfl_down_sync(0xffffffffu, b, 3) << 24);
    const int wt = __shfl_sync(0xffffffffu, static_cast<int>(word), 4 * (lane % 4));        // word t
    const int wt4 = __shfl_sync(0xffffffffu, static_cast<int>(word), 4 * (lane % 4 + 4));   // word t + 4
    const int w2t = __shfl_sync(0xffffffffu, static_cast<int>(word), 8 * (lane % 4));       // word 2t
    const int w2t1 = __shfl_sync(0xffffffffu, static_cast<int>(word), 8 * (lane % 4) + 4);  // word 2t + 1
    const int hs1 = __shfl_sync(0xffffffffu, h, 16);
    if (live) {
        q[tok * k + static_cast<long long>(u) * 32 + lane] = static_cast<int8_t>(c);
        if (lane == 0) {
            d[static_cast<long long>(tok) * subs + u] = dv;
            sums[(static_cast<long long>(tok) * subs + u) * 2] = static_cast<float>(h);
            sums[(static_cast<long long>(tok) * subs + u) * 2 + 1] = static_cast<float>(hs1);
        }
    }
    const int n = tok / 8, g = tok % 8;
    const long long e = static_cast<long long>(u) * tiles + n;
    if (lane < 4) {
        frag[e * 32 + g * 4 + lane] = make_int2(wt, wt4);
        fragp[e * 32 + g * 4 + lane] = make_int2(w2t, w2t1);
    }
    if (lane == 0) {
        fd[e * 8 + g] = dv;
        fs[e * 16 + g * 2] = static_cast<float>(h);
        fs[e * 16 + g * 2 + 1] = static_cast<float>(hs1);
    }
}

// llama.cpp's iq1s_grid (ggml-common.h, MIT) at 2 bits per value (0, 1 or 3 for -1), value j at
// bits 2j: 4 KiB, small enough for the v2 kernel's shared memory (kIq1SGrid is 16 KiB).
__device__ const uint16_t kIq1SGridP[2048] = {
    0xffff, 0xfffd, 0xfff0, 0xfff7, 0xfff5, 0xffcc, 0xffc0, 0xffdf, 0xffdd, 0xffd7, 0xffd5, 0xff30, 0xff0c, 0xff03, 0xff01, 0xff10,
    0xff7f, 0xff7d, 0xff77, 0xff75, 0xff40, 0xff5f, 0xff5d, 0xff57, 0xff55, 0xfcf3, 0xfcf0, 0xfccc, 0xfcc3, 0xfcc1, 0xfcc4, 0xfcc5,
    0xfcd0, 0xfc3c, 0xfc31, 0xfc34, 0xfc0d, 0xfc00, 0xfc05, 0xfc1c, 0xfc13, 0xfc11, 0xfc17, 0xfc70, 0xfc4c, 0xfc43, 0xfc41, 0xfc50,
    0xfdff, 0xfdfd, 0xfdf7, 0xfdf5, 0xfdc0, 0xfddf, 0xfddd, 0xfdd7, 0xfdd5, 0xfd30, 0xfd0c, 0xfd04, 0xfd13, 0xfd14, 0xfd7f, 0xfd7d,
    0xfd77, 0xfd75, 0xfd4c, 0xfd40, 0xfd44, 0xfd5f, 0xfd5d, 0xfd57, 0xfd55, 0xf3cc, 0xf3c3, 0xf3c1, 0xf3d0, 0xf33c, 0xf334, 0xf300,
    0xf305, 0xf313, 0xf310, 0xf34c, 0xf344, 0xf350, 0xf0fc, 0xf0f3, 0xf0f0, 0xf0f1, 0xf0c0, 0xf0c7, 0xf0c5, 0xf0d4, 0xf03f, 0xf030,
    0xf035, 0xf00f, 0xf00c, 0xf003, 0xf000, 0xf001, 0xf004, 0xf01f, 0xf01d, 0xf010, 0xf017, 0xf015, 0xf07c, 0xf04c, 0xf040, 0xf047,
    0xf045, 0xf05c, 0xf053, 0xf050, 0xf051, 0xf054, 0xf1c3, 0xf1c4, 0xf13c, 0xf133, 0xf10f, 0xf10d, 0xf100, 0xf107, 0xf11f, 0xf11c,
    0xf111, 0xf114, 0xf170, 0xf14c, 0xf143, 0xf144, 0xf7ff, 0xf7fd, 0xf7f7, 0xf7f5, 0xf7c0, 0xf7df, 0xf7dd, 0xf7d7, 0xf7d5, 0xf730,
    0xf70c, 0xf701, 0xf710, 0xf77f, 0xf77d, 0xf777, 0xf775, 0xf740, 0xf75f, 0xf75d, 0xf757, 0xf755, 0xf4f0, 0xf4cc, 0xf4c3, 0xf4c4,
    0xf4d3, 0xf4d0, 0xf43c, 0xf40f, 0xf40c, 0xf400, 0xf41c, 0xf413, 0xf414, 0xf44c, 0xf443, 0xf441, 0xf444, 0xf450, 0xf5ff, 0xf5fd,
    0xf5f7, 0xf5f5, 0xf5c0, 0xf5df, 0xf5dd, 0xf5d7, 0xf5d5, 0xf530, 0xf50c, 0xf504, 0xf51c, 0xf510, 0xf57f, 0xf57d, 0xf570, 0xf577,
    0xf575, 0xf540, 0xf55f, 0xf55d, 0xf557, 0xf555, 0xcfcf, 0xcfcc, 0xcfc3, 0xcfc4, 0xcfd3, 0xcfd0, 0xcf3c, 0xcf33, 0xcf0f, 0xcf00,
    0xcf07, 0xcf1c, 0xcf13, 0xcf10, 0xcf14, 0xcf4c, 0xcf43, 0xcf41, 0xcf5c, 0xcf50, 0xccfc, 0xccf3, 0xccf1, 0xccf4, 0xcccf, 0xcccd,
    0xccc0, 0xccc7, 0xccdc, 0xccd3, 0xccd4, 0xcc30, 0xcc35, 0xcc0f, 0xcc0c, 0xcc0d, 0xcc03, 0xcc00, 0xcc01, 0xcc04, 0xcc1f, 0xcc10,
    0xcc73, 0xcc4d, 0xcc40, 0xcc5c, 0xcc53, 0xcdcc, 0xcdc3, 0xcdc1, 0xcdd0, 0xcd3f, 0xcd31, 0xcd34, 0xcd0d, 0xcd00, 0xcd07, 0xcd05,
    0xcd13, 0xcd11, 0xcd70, 0xcd4c, 0xcd43, 0xcd41, 0xcd50, 0xc3fc, 0xc3f1, 0xc3f4, 0xc3c3, 0xc3c0, 0xc3c7, 0xc3c4, 0xc3dc, 0xc3d1,
    0xc33c, 0xc330, 0xc331, 0xc337, 0xc335, 0xc30c, 0xc303, 0xc300, 0xc301, 0xc304, 0xc31d, 0xc310, 0xc317, 0xc373, 0xc374, 0xc34f,
    0xc343, 0xc340, 0xc347, 0xc344, 0xc345, 0xc35c, 0xc353, 0xc350, 0xc354, 0xc0fd, 0xc0f0, 0xc0f5, 0xc0cc, 0xc0c3, 0xc0c0, 0xc0c1,
    0xc0c4, 0xc0df, 0xc0dd, 0xc0d0, 0xc0d7, 0xc0d5, 0xc03c, 0xc033, 0xc030, 0xc031, 0xc00c, 0xc00d, 0xc003, 0xc000, 0xc001, 0xc004,
    0xc005, 0xc01c, 0xc013, 0xc010, 0xc011, 0xc014, 0xc07f, 0xc07d, 0xc073, 0xc070, 0xc077, 0xc075, 0xc04f, 0xc04c, 0xc043, 0xc040,
    0xc041, 0xc044, 0xc045, 0xc05f, 0xc05d, 0xc050, 0xc1fc, 0xc1f3, 0xc1f0, 0xc1f1, 0xc1c0, 0xc1c1, 0xc1c7, 0xc1c5, 0xc1dc, 0xc1d1,
    0xc13f, 0xc13d, 0xc133, 0xc130, 0xc137, 0xc135, 0xc10c, 0xc100, 0xc101, 0xc107, 0xc104, 0xc11c, 0xc113, 0xc110, 0xc117, 0xc114,
    0xc115, 0xc171, 0xc175, 0xc14d, 0xc140, 0xc153, 0xc154, 0xc7cc, 0xc7c1, 0xc7d0, 0xc73c, 0xc733, 0xc731, 0xc734, 0xc70f, 0xc700,
    0xc707, 0xc705, 0xc71f, 0xc71c, 0xc713, 0xc711, 0xc714, 0xc770, 0xc74c, 0xc743, 0xc750, 0xc4cf, 0xc4cd, 0xc4c0, 0xc4c5, 0xc4dc,
    0xc4d0, 0xc43d, 0xc433, 0xc430, 0xc437, 0xc40c, 0xc403, 0xc400, 0xc401, 0xc404, 0xc405, 0xc41f, 0xc410, 0xc415, 0xc474, 0xc44c,
    0xc44d, 0xc440, 0xc447, 0xc45c, 0xc451, 0xc454, 0xc5f4, 0xc5c1, 0xc5d3, 0xc5d1, 0xc533, 0xc531, 0xc534, 0xc50f, 0xc50d, 0xc500,
    0xc507, 0xc51c, 0xc511, 0xc514, 0xc570, 0xc54c, 0xc541, 0xc545, 0xdfff, 0xdffd, 0xdff7, 0xdff5, 0xdfc0, 0xdfdf, 0xdfdd, 0xdfd0,
    0xdfd7, 0xdfd5, 0xdf30, 0xdf0c, 0xdf04, 0xdf1c, 0xdf10, 0xdf7f, 0xdf7d, 0xdf77, 0xdf75, 0xdf40, 0xdf5f, 0xdf5d, 0xdf50, 0xdf57,
    0xdf55, 0xdcf0, 0xdccc, 0xdcc3, 0xdcc4, 0xdcd0, 0xdc3d, 0xdc33, 0xdc34, 0xdc00, 0xdc07, 0xdc05, 0xdc1c, 0xdc13, 0xdc10, 0xdc11,
    0xdc70, 0xdc4f, 0xdc41, 0xdc44, 0xdc50, 0xddfc, 0xddf7, 0xddf5, 0xddcc, 0xddc0, 0xdddf, 0xdddd, 0xddd7, 0xddd5, 0xdd30, 0xdd0c,
    0xdd01, 0xdd04, 0xdd10, 0xdd7c, 0xdd77, 0xdd75, 0xdd4c, 0xdd40, 0xdd5f, 0xdd5d, 0xdd57, 0xdd55, 0xd3f0, 0xd3c3, 0xd3c1, 0xd3c4,
    0xd3d0, 0xd333, 0xd330, 0xd331, 0xd334, 0xd30d, 0xd300, 0xd307, 0xd305, 0xd311, 0xd370, 0xd34c, 0xd343, 0xd344, 0xd35c, 0xd350,
    0xd0f4, 0xd0c0, 0xd0dc, 0xd0d4, 0xd03f, 0xd030, 0xd037, 0xd00c, 0xd003, 0xd000, 0xd004, 0xd01d, 0xd010, 0xd017, 0xd074, 0xd04f,
    0xd043, 0xd040, 0xd047, 0xd045, 0xd05c, 0xd053, 0xd051, 0xd054, 0xd1f0, 0xd1cf, 0xd1cd, 0xd1c4, 0xd1d0, 0xd13c, 0xd134, 0xd100,
    0xd11f, 0xd11c, 0xd114, 0xd173, 0xd171, 0xd14f, 0xd145, 0xd7ff, 0xd7fd, 0xd7f7, 0xd7f5, 0xd7c0, 0xd7df, 0xd7dd, 0xd7d7, 0xd7d5,
    0xd730, 0xd70c, 0xd703, 0xd710, 0xd77f, 0xd77d, 0xd777, 0xd775, 0xd75f, 0xd75d, 0xd757, 0xd755, 0xd4f4, 0xd4cc, 0xd4c3, 0xd4c4,
    0xd4d0, 0xd431, 0xd434, 0xd40d, 0xd400, 0xd41c, 0xd413, 0xd411, 0xd414, 0xd470, 0xd44f, 0xd441, 0xd444, 0xd453, 0xd450, 0xd5ff,
    0xd5fd, 0xd5f7, 0xd5f5, 0xd5df, 0xd5dd, 0xd5d7, 0xd5d5, 0xd530, 0xd50c, 0xd501, 0xd504, 0xd510, 0xd57f, 0xd57d, 0xd577, 0xd575,
    0xd540, 0xd55f, 0xd55d, 0xd557, 0xd555, 0x3ff0, 0x3fcc, 0x3fc1, 0x3fd0, 0x3f34, 0x3f0d, 0x3f00, 0x3f07, 0x3f05, 0x3f1c, 0x3f13,
    0x3f11, 0x3f43, 0x3f44, 0x3f5c, 0x3f51, 0x3cff, 0x3cfc, 0x3cf3, 0x3cf1, 0x3cf4, 0x3ccd, 0x3cc0, 0x3cc1, 0x3cc7, 0x3cc5, 0x3cdc,
    0x3cd1, 0x3cd4, 0x3c30, 0x3c37, 0x3c35, 0x3c0c, 0x3c03, 0x3c00, 0x3c01, 0x3c04, 0x3c05, 0x3c10, 0x3c17, 0x3c15, 0x3c7c, 0x3c73,
    0x3c71, 0x3c4f, 0x3c4d, 0x3c40, 0x3c5f, 0x3c5c, 0x3c5d, 0x3df0, 0x3dcc, 0x3dc3, 0x3dc1, 0x3dd0, 0x3d3c, 0x3d0d, 0x3d00, 0x3d05,
    0x3d13, 0x3d14, 0x3d74, 0x3d43, 0x3d50, 0x33fc, 0x33c0, 0x33c4, 0x33d4, 0x3330, 0x3337, 0x3335, 0x330c, 0x3303, 0x3300, 0x3301,
    0x331c, 0x331d, 0x3310, 0x3317, 0x3315, 0x337c, 0x3371, 0x3374, 0x334f, 0x334d, 0x3340, 0x335f, 0x335c, 0x3354, 0x30fc, 0x30fd,
    0x30f0, 0x30f5, 0x30cc, 0x30c3, 0x30c0, 0x30c1, 0x30c4, 0x30df, 0x30d0, 0x30d5, 0x303c, 0x3033, 0x3030, 0x3031, 0x3034, 0x300f,
    0x300c, 0x3003, 0x3000, 0x3001, 0x3007, 0x3004, 0x301c, 0x3013, 0x3010, 0x3011, 0x3014, 0x307d, 0x3073, 0x3070, 0x3077, 0x304c,
    0x3043, 0x3040, 0x3041, 0x3044, 0x3045, 0x3050, 0x3057, 0x3055, 0x31fc, 0x31f0, 0x31f4, 0x31c3, 0x31c0, 0x31c7, 0x31c5, 0x31dc,
    0x31d3, 0x31d4, 0x313f, 0x313d, 0x3130, 0x3137, 0x310f, 0x310c, 0x310d, 0x3100, 0x3101, 0x3104, 0x311d, 0x3110, 0x3117, 0x317c,
    0x3170, 0x3175, 0x3143, 0x3140, 0x315c, 0x3153, 0x3151, 0x37f0, 0x37cc, 0x37c0, 0x37c5, 0x37d0, 0x373d, 0x3734, 0x370f, 0x3700,
    0x3707, 0x371c, 0x3713, 0x3711, 0x3714, 0x3770, 0x374c, 0x3743, 0x3741, 0x3744, 0x3750, 0x34fc, 0x34f0, 0x34f1, 0x34f5, 0x34cf,
    0x34c3, 0x34c0, 0x34c7, 0x34dc, 0x34d3, 0x34d1, 0x343f, 0x3430, 0x3435, 0x340c, 0x340d, 0x3403, 0x3400, 0x3401, 0x3404, 0x341f,
    0x341d, 0x3410, 0x3411, 0x3415, 0x3471, 0x3474, 0x344d, 0x3440, 0x3441, 0x3447, 0x345c, 0x3453, 0x3451, 0x3454, 0x35c1, 0x3533,
    0x3531, 0x3534, 0x3500, 0x3507, 0x3505, 0x3513, 0x3514, 0x3543, 0x3550, 0x0ffc, 0x0ff3, 0x0ff0, 0x0ff1, 0x0ff4, 0x0fcd, 0x0fc0,
    0x0fc5, 0x0fdc, 0x0fd3, 0x0fd4, 0x0f3f, 0x0f30, 0x0f37, 0x0f0c, 0x0f03, 0x0f00, 0x0f01, 0x0f04, 0x0f10, 0x0f17, 0x0f71, 0x0f74,
    0x0f40, 0x0f47, 0x0f5f, 0x0f5c, 0x0f51, 0x0f54, 0x0cf0, 0x0cf7, 0x0cf4, 0x0cf5, 0x0ccc, 0x0cc3, 0x0cc0, 0x0cc1, 0x0cc7, 0x0cc4,
    0x0cdf, 0x0cd0, 0x0cd1, 0x0cd7, 0x0cd5, 0x0c3c, 0x0c33, 0x0c30, 0x0c31, 0x0c34, 0x0c0f, 0x0c0c, 0x0c0d, 0x0c03, 0x0c00, 0x0c01,
    0x0c07, 0x0c04, 0x0c05, 0x0c1c, 0x0c13, 0x0c10, 0x0c11, 0x0c14, 0x0c7d, 0x0c70, 0x0c4c, 0x0c43, 0x0c40, 0x0c41, 0x0c44, 0x0c5f,
    0x0c50, 0x0c55, 0x0dfc, 0x0df1, 0x0dcd, 0x0dc0, 0x0dc5, 0x0ddc, 0x0dd3, 0x0d3d, 0x0d30, 0x0d35, 0x0d0c, 0x0d03, 0x0d00, 0x0d01,
    0x0d04, 0x0d1d, 0x0d10, 0x0d70, 0x0d4f, 0x0d4d, 0x0d40, 0x0d44, 0x0d45, 0x0d53, 0x03f3, 0x03f0, 0x03cc, 0x03c3, 0x03c0, 0x03c1,
    0x03c7, 0x03c4, 0x03dc, 0x03d0, 0x03d7, 0x03d5, 0x033c, 0x0333, 0x0330, 0x0331, 0x0334, 0x0335, 0x030f, 0x030c, 0x0303, 0x0300,
    0x0301, 0x0307, 0x0304, 0x0305, 0x031c, 0x031d, 0x0313, 0x0310, 0x0311, 0x0314, 0x037f, 0x0377, 0x0375, 0x034c, 0x0343, 0x0340,
    0x0341, 0x0344, 0x035c, 0x0353, 0x0350, 0x0355, 0x00fc, 0x00fd, 0x00f3, 0x00f0, 0x00f1, 0x00f4, 0x00cf, 0x00cc, 0x00cd, 0x00c3,
    0x00c0, 0x00c1, 0x00c4, 0x00c5, 0x00dc, 0x00d3, 0x00d0, 0x00d1, 0x00d4, 0x003f, 0x003c, 0x003d, 0x0033, 0x0030, 0x0031, 0x0037,
    0x0034, 0x000f, 0x000c, 0x000d, 0x0003, 0x0000, 0x0001, 0x0007, 0x0004, 0x0005, 0x001f, 0x001c, 0x0013, 0x0010, 0x0011, 0x0017,
    0x0014, 0x0015, 0x007c, 0x0073, 0x0070, 0x0074, 0x0075, 0x004f, 0x004c, 0x0043, 0x0040, 0x0041, 0x0047, 0x0044, 0x005c, 0x0053,
    0x0050, 0x0051, 0x0054, 0x01ff, 0x01fc, 0x01fd, 0x01f3, 0x01f1, 0x01f7, 0x01f4, 0x01cc, 0x01c3, 0x01c0, 0x01c7, 0x01c4, 0x01df,
    0x01dc, 0x01dd, 0x01d3, 0x01d0, 0x01d1, 0x01d7, 0x01d4, 0x013c, 0x0130, 0x0131, 0x0137, 0x0134, 0x0135, 0x010f, 0x010c, 0x010d,
    0x0103, 0x0100, 0x0101, 0x0107, 0x0104, 0x0105, 0x011c, 0x0113, 0x0110, 0x0114, 0x017d, 0x0170, 0x0171, 0x0177, 0x0174, 0x0175,
    0x014c, 0x0140, 0x0145, 0x015d, 0x0150, 0x0151, 0x0157, 0x0154, 0x07f3, 0x07f0, 0x07f1, 0x07f4, 0x07cf, 0x07c0, 0x07c7, 0x07dc,
    0x07d5, 0x0730, 0x0737, 0x0735, 0x070c, 0x0703, 0x0700, 0x0701, 0x0707, 0x0704, 0x071f, 0x071d, 0x0713, 0x0710, 0x077d, 0x0774,
    0x074f, 0x074d, 0x0740, 0x0747, 0x075c, 0x0754, 0x04fc, 0x04fd, 0x04f0, 0x04f5, 0x04cc, 0x04c3, 0x04c0, 0x04c1, 0x04c4, 0x04d0,
    0x043c, 0x0433, 0x0430, 0x0431, 0x0434, 0x040f, 0x040c, 0x040d, 0x0403, 0x0400, 0x0401, 0x0407, 0x0404, 0x0405, 0x041c, 0x0413,
    0x0410, 0x0411, 0x0414, 0x047c, 0x0470, 0x0474, 0x044c, 0x0443, 0x0440, 0x0441, 0x0447, 0x0444, 0x0450, 0x05f3, 0x05f7, 0x05c0,
    0x05c5, 0x05df, 0x05d0, 0x05d1, 0x05d4, 0x0530, 0x0537, 0x0534, 0x050c, 0x0500, 0x0501, 0x0507, 0x0504, 0x051d, 0x0510, 0x0517,
    0x0515, 0x057c, 0x0575, 0x054d, 0x0540, 0x0541, 0x0547, 0x0545, 0x055c, 0x1ff0, 0x1fc3, 0x1fc1, 0x1fc4, 0x1fd0, 0x1f33, 0x1f0f,
    0x1f00, 0x1f01, 0x1f07, 0x1f05, 0x1f1c, 0x1f13, 0x1f11, 0x1f14, 0x1f7c, 0x1f41, 0x1f50, 0x1cfc, 0x1cf3, 0x1cf1, 0x1cf4, 0x1ccd,
    0x1cc0, 0x1cdc, 0x1cdd, 0x1cd1, 0x1cd4, 0x1c30, 0x1c34, 0x1c0c, 0x1c00, 0x1c01, 0x1c04, 0x1c10, 0x1c11, 0x1c15, 0x1c73, 0x1c75,
    0x1c4d, 0x1c40, 0x1c5c, 0x1c51, 0x1c54, 0x1dcc, 0x1dc1, 0x1dc4, 0x1d3f, 0x1d3c, 0x1d31, 0x1d00, 0x1d01, 0x1d07, 0x1d1f, 0x1d70,
    0x1d4c, 0x1d41, 0x1d50, 0x13cc, 0x13cd, 0x13c0, 0x13c1, 0x13c5, 0x13dc, 0x13d1, 0x13d4, 0x133f, 0x133d, 0x1330, 0x1337, 0x1335,
    0x130c, 0x1303, 0x1300, 0x1301, 0x1304, 0x1305, 0x131f, 0x131d, 0x1310, 0x1373, 0x1370, 0x1374, 0x134f, 0x134d, 0x1343, 0x1340,
    0x1341, 0x1347, 0x1345, 0x135c, 0x1353, 0x1354, 0x10f0, 0x10f7, 0x10f5, 0x10cc, 0x10c0, 0x10c1, 0x10c4, 0x1033, 0x1030, 0x1031,
    0x1034, 0x100f, 0x100c, 0x1003, 0x1000, 0x1001, 0x1004, 0x101c, 0x1013, 0x1010, 0x1011, 0x1014, 0x1071, 0x1074, 0x1075, 0x104c,
    0x1040, 0x1041, 0x1044, 0x1045, 0x105d, 0x1050, 0x1051, 0x1057, 0x11fd, 0x11f4, 0x11c0, 0x11df, 0x11d1, 0x11d7, 0x11d4, 0x113f,
    0x1130, 0x1137, 0x1135, 0x110c, 0x1103, 0x1100, 0x1101, 0x1107, 0x1105, 0x111f, 0x1110, 0x1117, 0x117f, 0x117d, 0x1170, 0x1175,
    0x1143, 0x1141, 0x1147, 0x1144, 0x115f, 0x1153, 0x1151, 0x1155, 0x17c1, 0x17c4, 0x17d0, 0x173c, 0x170d, 0x1700, 0x1705, 0x171c,
    0x1714, 0x1770, 0x174c, 0x1747, 0x1751, 0x14fc, 0x14f3, 0x14cf, 0x14c0, 0x14dc, 0x14d3, 0x14d1, 0x14d4, 0x143f, 0x143c, 0x1430,
    0x1431, 0x1437, 0x140c, 0x1403, 0x1400, 0x1401, 0x1404, 0x141f, 0x1410, 0x1415, 0x147d, 0x1473, 0x1475, 0x1440, 0x145c, 0x1453,
    0x1450, 0x1454, 0x15cc, 0x15c1, 0x15c7, 0x153c, 0x1533, 0x1534, 0x150f, 0x1500, 0x1507, 0x1505, 0x1513, 0x1510, 0x1514, 0x1571,
    0x1543, 0x1547, 0x1545, 0x1551, 0x7fff, 0x7ffd, 0x7ff7, 0x7ff5, 0x7fdf, 0x7fdd, 0x7fd7, 0x7fd5, 0x7f30, 0x7f0f, 0x7f0c, 0x7f03,
    0x7f01, 0x7f04, 0x7f10, 0x7f7f, 0x7f7d, 0x7f77, 0x7f75, 0x7f40, 0x7f5f, 0x7f5d, 0x7f57, 0x7f55, 0x7cf0, 0x7ccc, 0x7cc3, 0x7cc1,
    0x7cc4, 0x7cd0, 0x7c3c, 0x7c33, 0x7c34, 0x7c0f, 0x7c0d, 0x7c00, 0x7c01, 0x7c07, 0x7c04, 0x7c13, 0x7c11, 0x7c14, 0x7c70, 0x7c74,
    0x7c43, 0x7c41, 0x7c44, 0x7c50, 0x7dff, 0x7dfd, 0x7df7, 0x7df5, 0x7dc0, 0x7ddf, 0x7ddd, 0x7dd7, 0x7dd5, 0x7d0c, 0x7d03, 0x7d04,
    0x7d10, 0x7d7f, 0x7d7d, 0x7d77, 0x7d75, 0x7d40, 0x7d5f, 0x7d5d, 0x7d57, 0x7d55, 0x73c3, 0x73c4, 0x733c, 0x7333, 0x730c, 0x7300,
    0x7305, 0x731c, 0x7313, 0x7314, 0x7343, 0x7344, 0x70fc, 0x70f4, 0x70cd, 0x70c0, 0x70c5, 0x70d1, 0x70d4, 0x703f, 0x703c, 0x7030,
    0x7037, 0x700c, 0x7003, 0x7000, 0x7001, 0x7004, 0x7005, 0x7010, 0x7011, 0x7017, 0x7015, 0x707c, 0x7073, 0x7071, 0x7074, 0x704f,
    0x704d, 0x7040, 0x7047, 0x7051, 0x71cc, 0x71c1, 0x71c4, 0x71d0, 0x713c, 0x7133, 0x7134, 0x7135, 0x710f, 0x7100, 0x7104, 0x7105,
    0x711c, 0x7111, 0x7115, 0x7170, 0x714c, 0x7145, 0x7153, 0x77ff, 0x77fd, 0x77f7, 0x77f5, 0x77c0, 0x77df, 0x77dd, 0x77d7, 0x77d5,
    0x773c, 0x7730, 0x770c, 0x7703, 0x7704, 0x7710, 0x7714, 0x777f, 0x777d, 0x7777, 0x7775, 0x7740, 0x775f, 0x775d, 0x7757, 0x7755,
    0x74f0, 0x74f1, 0x74cc, 0x74c3, 0x74c1, 0x74d0, 0x743c, 0x7433, 0x7431, 0x7434, 0x740f, 0x740d, 0x7400, 0x7405, 0x741c, 0x7413,
    0x7470, 0x7441, 0x7444, 0x7450, 0x75ff, 0x75fd, 0x75f7, 0x75f5, 0x75c0, 0x75df, 0x75dd, 0x75d7, 0x75d5, 0x7530, 0x750c, 0x7503,
    0x7501, 0x757f, 0x757d, 0x7577, 0x7575, 0x7540, 0x755f, 0x755d, 0x7557, 0x7555, 0x4ff0, 0x4fcc, 0x4fc1, 0x4fc7, 0x4fc4, 0x4fd0,
    0x4f3c, 0x4f31, 0x4f34, 0x4f00, 0x4f07, 0x4f05, 0x4f14, 0x4f15, 0x4f70, 0x4f4c, 0x4f43, 0x4f41, 0x4f44, 0x4f50, 0x4cfc, 0x4cf3,
    0x4cf1, 0x4cf4, 0x4ccf, 0x4cc0, 0x4cc7, 0x4cc5, 0x4cdc, 0x4cd3, 0x4cd1, 0x4cd4, 0x4c3f, 0x4c30, 0x4c0f, 0x4c0c, 0x4c03, 0x4c00,
    0x4c01, 0x4c04, 0x4c1d, 0x4c10, 0x4c73, 0x4c71, 0x4c4d, 0x4c40, 0x4c47, 0x4c5c, 0x4c53, 0x4c51, 0x4c54, 0x4df0, 0x4dcc, 0x4dc3,
    0x4dc4, 0x4dd0, 0x4d33, 0x4d31, 0x4d34, 0x4d0f, 0x4d0d, 0x4d00, 0x4d07, 0x4d11, 0x4d14, 0x4d70, 0x4d43, 0x4d41, 0x4d54, 0x43fc,
    0x43f3, 0x43f1, 0x43cf, 0x43c0, 0x43c7, 0x43d1, 0x433f, 0x4335, 0x430c, 0x4303, 0x4300, 0x4301, 0x4307, 0x4304, 0x431f, 0x431c,
    0x431d, 0x4310, 0x4373, 0x4371, 0x434d, 0x4343, 0x4340, 0x4347, 0x435c, 0x4354, 0x40ff, 0x40f0, 0x40f7, 0x40f5, 0x40cf, 0x40cc,
    0x40c3, 0x40c0, 0x40c1, 0x40c4, 0x40dc, 0x40d0, 0x40d4, 0x40d5, 0x403c, 0x4033, 0x4030, 0x4031, 0x4034, 0x400f, 0x400c, 0x400d,
    0x4003, 0x4000, 0x4001, 0x4007, 0x4004, 0x4005, 0x401c, 0x4013, 0x4010, 0x4011, 0x4014, 0x407c, 0x4070, 0x4077, 0x404c, 0x404d,
    0x4043, 0x4040, 0x4041, 0x4044, 0x4045, 0x405f, 0x405d, 0x4050, 0x4057, 0x4055, 0x41fc, 0x41f3, 0x41cf, 0x41c0, 0x41c4, 0x41df,
    0x41d1, 0x41d4, 0x4130, 0x4137, 0x4134, 0x410c, 0x410d, 0x4100, 0x4101, 0x4104, 0x4110, 0x4117, 0x417d, 0x4173, 0x4174, 0x4175,
    0x414d, 0x4143, 0x4140, 0x4153, 0x4151, 0x4154, 0x47f0, 0x47c1, 0x47c4, 0x47d0, 0x473c, 0x4731, 0x470f, 0x470d, 0x4700, 0x4701,
    0x4705, 0x4713, 0x4710, 0x4770, 0x474c, 0x4741, 0x4744, 0x4750, 0x44f3, 0x44f1, 0x44f4, 0x44cf, 0x44cd, 0x44c0, 0x44c7, 0x44c5,
    0x44df, 0x44dc, 0x44d3, 0x44d1, 0x443f, 0x443d, 0x4430, 0x4437, 0x4435, 0x440c, 0x4403, 0x4400, 0x4401, 0x4404, 0x441d, 0x4410,
    0x4411, 0x4415, 0x447c, 0x4473, 0x444f, 0x444d, 0x4440, 0x4445, 0x445c, 0x4451, 0x45f0, 0x45c0, 0x45d0, 0x4533, 0x4531, 0x4534,
    0x450f, 0x4500, 0x4507, 0x451c, 0x4570, 0x454c, 0x4543, 0x4540, 0x4541, 0x5fff, 0x5ffd, 0x5ff7, 0x5ff5, 0x5fc0, 0x5fdf, 0x5fdd,
    0x5fd7, 0x5fd5, 0x5f30, 0x5f0c, 0x5f03, 0x5f01, 0x5f04, 0x5f7f, 0x5f7d, 0x5f77, 0x5f75, 0x5f40, 0x5f5f, 0x5f5d, 0x5f57, 0x5f55,
    0x5cf0, 0x5cf4, 0x5ccc, 0x5cc3, 0x5cc1, 0x5cc4, 0x5cc5, 0x5c31, 0x5c34, 0x5c0c, 0x5c00, 0x5c07, 0x5c05, 0x5c1c, 0x5c13, 0x5c70,
    0x5c4f, 0x5c4d, 0x5c41, 0x5c44, 0x5dfd, 0x5df7, 0x5df5, 0x5dcf, 0x5dc4, 0x5ddd, 0x5dd7, 0x5dd5, 0x5d30, 0x5d0c, 0x5d01, 0x5d04,
    0x5d10, 0x5d7f, 0x5d7d, 0x5d77, 0x5d75, 0x5d40, 0x5d5f, 0x5d5d, 0x5d57, 0x5d55, 0x53c4, 0x53d0, 0x533c, 0x5333, 0x530f, 0x5303,
    0x5300, 0x5307, 0x5305, 0x531c, 0x5310, 0x5311, 0x5317, 0x5314, 0x5370, 0x50f1, 0x50f4, 0x50cf, 0x50cd, 0x50c0, 0x50c7, 0x50d1,
    0x50d4, 0x503d, 0x5030, 0x500c, 0x5003, 0x5000, 0x5001, 0x5004, 0x5010, 0x5015, 0x507c, 0x5073, 0x5070, 0x5071, 0x5074, 0x504d,
    0x5040, 0x5047, 0x51f0, 0x51cc, 0x51c1, 0x51c5, 0x51dc, 0x51d0, 0x5133, 0x5131, 0x5135, 0x510d, 0x5100, 0x5101, 0x5107, 0x511f,
    0x511d, 0x5171, 0x514f, 0x5140, 0x5141, 0x5144, 0x515c, 0x5153, 0x5151, 0x57ff, 0x57fd, 0x57f7, 0x57f5, 0x57df, 0x57dd, 0x57d7,
    0x57d5, 0x570c, 0x5703, 0x5701, 0x5704, 0x577f, 0x577d, 0x5777, 0x5775, 0x5740, 0x575f, 0x575d, 0x5757, 0x5755, 0x54f0, 0x54c3,
    0x54c4, 0x54dc, 0x54d0, 0x543c, 0x540f, 0x5400, 0x5405, 0x541c, 0x5411, 0x5414, 0x544f, 0x5441, 0x55ff, 0x55fd, 0x55f7, 0x55f5,
    0x55df, 0x55dd, 0x55d7, 0x55d5, 0x550c, 0x5503, 0x5501, 0x557f, 0x557d, 0x5577, 0x5575, 0x5540, 0x555f, 0x555d, 0x5557, 0x5555,
};

// Shared-memory copies of a type's grid (u32 words; u64 entries as two) and of the 128 sign
// patterns: a lookup by 32 lanes at random indices is a few bank conflicts there, but a dozen L1
// wavefronts from global memory, which bounded the kernel.
template <int T>
struct Grid {
    static constexpr int kWords = T == kIq3S     ? 512
                                  : T == kIq3Xxs ? 256
                                  : T == kIq2Xxs ? 512
                                  : T == kIq2Xs  ? 1024
                                  : T == kIq2S   ? 2048
                                  : T == kIq1S || T == kIq1M ? 1024
                                                 : 1;
    static constexpr bool kSigns = T == kIq3Xxs || T == kIq2Xxs || T == kIq2Xs;
    __device__ static const uint32_t* source() {
        if constexpr (T == kIq3S) return kIq3SGrid;
        else if constexpr (T == kIq3Xxs) return kIq3XxsGridG;
        else if constexpr (T == kIq2Xxs) return reinterpret_cast<const uint32_t*>(kIq2XxsGrid);
        else if constexpr (T == kIq2Xs) return reinterpret_cast<const uint32_t*>(kIq2XsGrid);
        else if constexpr (T == kIq2S) return reinterpret_cast<const uint32_t*>(kIq2SGrid);
        else if constexpr (T == kIq1S || T == kIq1M) return reinterpret_cast<const uint32_t*>(kIq1SGridP);
        else return nullptr;
    }
};

__device__ __forceinline__ uint64_t grid64(const uint32_t* sg, uint32_t i) {
    const uint2 v = reinterpret_cast<const uint2*>(sg)[i];
    return static_cast<uint64_t>(v.x) | (static_cast<uint64_t>(v.y) << 32);
}

// IQ1 codes 8 g + dl of a grid word (bytes g of -1, 0, 1) and a per-byte delta word (dl = +-1).
__device__ __forceinline__ uint32_t iq1_codes(uint32_t gw, uint32_t dlw) {
    return __vsub4(__vadd4((gw & 0x01010101u) << 3, dlw), ((gw >> 7) & 0x01010101u) << 4);
}

// IQ1_S / IQ1_M word wi of sub-block ib as codes 8 g + dl (|code| <= 9), from the shared 2-bit
// grid copy (kIq1SGridP).
template <int T>
__device__ __forceinline__ uint32_t iq1_word(const uint8_t* __restrict__ x, int ib, int wi, const uint32_t* sg) {
    const int l = wi / 2, half = wi % 2;
    uint32_t idx, dlw;
    if constexpr (T == kIq1S) {  // d | qs[32] | qh u16[8]
        const uint32_t qh = u16_at(x + 34 + 2 * ib);
        idx = x[2 + 4 * ib + l] | (((qh >> (3 * l)) & 7) << 8);
        dlw = qh & 0x8000 ? 0xffffffffu : 0x01010101u;
    } else {  // IQ1_M: qs[32] | qh[16] | scales[8]
        const uint32_t qh = x[32 + 2 * ib + l / 2];
        idx = x[4 * ib + l] | ((qh << (l % 2 ? 4 : 8)) & 0x700);
        dlw = qh & (l % 2 ? 0x80 : 0x08) ? 0xffffffffu : 0x01010101u;
    }
    const uint32_t v = (reinterpret_cast<const uint16_t*>(sg)[idx] >> (8 * half)) & 0xffu;
    const uint32_t b = (v & 0x03u) | ((v & 0x0cu) << 6) | ((v & 0x30u) << 12) | ((v & 0xc0u) << 18);
    // b's bit 0 per byte: |g| = 1; bit 1: g = -1 (iq1_codes reads them from a byte 0x01 / 0xff)
    return __vsub4(__vadd4((b & 0x01010101u) << 3, dlw), ((b >> 1) & 0x01010101u) << 4);
}

// Word wi (values 4wi .. 4wi+3) of sub-block ib as 4 int8 codes, for the types whose scales cover
// 16 or 32 values (all but IQ1_M); grids and sign patterns from the shared copies.
template <int T>
__device__ __forceinline__ int word(const uint8_t* __restrict__ x, int ib, int wi, const uint32_t* sg,
                                    const uint8_t* sks) {
    const int l = wi / 2, half = wi % 2;
    if constexpr (T == kIq3S) {
        const uint32_t qh = x[66 + ib];
        const uint32_t idx = x[2 + 8 * ib + 2 * l + half] | ((qh << (8 - 2 * l - half)) & 256);
        return signed4(sg[idx], (x[74 + 4 * ib + l] >> (4 * half)) & 0xf);
    } else if constexpr (T == kIq3Xxs) {
        const uint32_t aux = u32_at2(x + 66 + 4 * ib);
        const uint32_t sb = sks[(aux >> (7 * l)) & 127];
        return signed4(sg[x[2 + 8 * ib + 2 * l + half]], (sb >> (4 * half)) & 0xf);
    } else if constexpr (T == kIq4Xs) {
        const uint32_t tab[4] = {0xbfad9881u, 0xf6eaddcfu, 0x26190d01u, 0x71594535u};
        const int2 v = table16(u32_at2(x + 8 + 16 * ib + 4 * (wi % 4)), tab);
        return wi < 4 ? v.x : v.y;
    } else if constexpr (T == kQ4K) {
        const uint32_t q = reinterpret_cast<const uint32_t*>(x + 16 + 32 * (ib / 2))[wi];
        return static_cast<int>((q >> (4 * (ib % 2))) & 0x0f0f0f0fu);
    } else if constexpr (T == kQ2K) {
        const uint32_t q = reinterpret_cast<const uint32_t*>(x + 16 + 32 * (ib / 4))[wi];
        return static_cast<int>((q >> (2 * (ib % 4))) & 0x03030303u);
    } else if constexpr (T == kIq2Xxs) {
        const uint32_t a1 = u32_at2(x + 6 + 8 * ib);
        const uint32_t g = sg[2 * x[2 + 8 * ib + l] + half];
        const uint32_t sb = sks[(a1 >> (7 * l)) & 127];
        return signed4(g, (sb >> (4 * half)) & 0xf);
    } else if constexpr (T == kIq2Xs) {
        const uint32_t q = u16_at(x + 2 + 2 * (4 * ib + l));
        const uint32_t g = sg[2 * (q & 511) + half];
        const uint32_t sb = sks[q >> 9];
        return signed4(g, (sb >> (4 * half)) & 0xf);
    } else if constexpr (T == kIq1S) {
        return static_cast<int>(iq1_word<T>(x, ib, wi, sg));
    } else {  // IQ2_S
        const uint32_t idx = x[2 + 4 * ib + l] | ((static_cast<uint32_t>(x[66 + ib]) << (8 - 2 * l)) & 0x300);
        const uint64_t g = grid64(sg, idx);
        return signed4(static_cast<uint32_t>(g >> (32 * half)), (x[34 + 4 * ib + l] >> (4 * half)) & 0xf);
    }
}

// IQ2_XS / IQ2_S word wi of sub-block ib as its 4 grid magnitudes (bytes 8, 25 or 43) and sign bits.
template <int T>
__device__ __forceinline__ uint2 mag_sign(const uint8_t* __restrict__ x, int ib, int wi, const uint32_t* sg,
                                          const uint8_t* sks) {
    const int l = wi / 2, half = wi % 2;
    if constexpr (T == kIq1M) {  // |8 g + dl| and the code's signs
        const uint32_t c = iq1_word<T>(x, ib, wi, sg);
        const uint32_t sb = ((c >> 7) & 1u) | ((c >> 14) & 2u) | ((c >> 21) & 4u) | ((c >> 28) & 8u);
        return make_uint2(__vabsss4(c), sb);
    } else if constexpr (T == kIq2Xs) {
        const uint32_t q = u16_at(x + 2 + 2 * (4 * ib + l));
        return make_uint2(sg[2 * (q & 511) + half], (sks[q >> 9] >> (4 * half)) & 0xf);
    } else {  // IQ2_S
        const uint32_t idx = x[2 + 4 * ib + l] | ((static_cast<uint32_t>(x[66 + ib]) << (8 - 2 * l)) & 0x300);
        return make_uint2(static_cast<uint32_t>(grid64(sg, idx) >> (32 * half)), (x[34 + 4 * ib + l] >> (4 * half)) & 0xf);
    }
}

// Codes magnitude * sc (<= 43 * 31) of one word split as 16 hi + lo: hi <= 83 and lo <= 15 are
// int8 planes, each signed per byte (a zero byte too, so per-byte adds instead of signed4's carry).
__device__ __forceinline__ void split_planes(uint2 ms, uint32_t sc, int& hi, int& lo) {
    const uint32_t p02 = (ms.x & 0x00ff00ffu) * sc;         // values 0 and 2 as 16-bit lanes
    const uint32_t p13 = ((ms.x >> 8) & 0x00ff00ffu) * sc;  // values 1 and 3
    const uint32_t h = ((p02 >> 4) & 0x00ff00ffu) | (((p13 >> 4) & 0x00ff00ffu) << 8);
    const uint32_t l = (p02 & 0x000f000fu) | ((p13 & 0x000f000fu) << 8);
    const uint32_t b = (ms.y * 0x00204081u) & 0x01010101u, mask = b * 0xffu;
    hi = static_cast<int>(__vadd4(h ^ mask, b));
    lo = static_cast<int>(__vadd4(l ^ mask, b));
}

// Whether a type's lanes take words 2t and 2t + 1 (one 8-value group) in their k-slots t and t + 4.
// K may be permuted freely as long as A and B agree; 16-value scales need the natural order.
template <int T>
constexpr bool kPaired = T == kIq3S || T == kIq3Xxs || T == kIq2Xxs || T == kQ4K;

// The two words lane t puts in k-slots t and t + 4 of sub-block ib.
template <int T>
__device__ __forceinline__ int2 pair(const uint8_t* __restrict__ x, int ib, int t, const uint32_t* sg,
                                     const uint8_t* sks) {
    if constexpr (T == kIq3S) {  // group l = t: qs[2t], qs[2t + 1], one sign byte, qh bits 2t, 2t + 1
        const uint32_t qh = x[66 + ib], qs = u16_at(x + 2 + 8 * ib + 2 * t), sb = x[74 + 4 * ib + t];
        const uint32_t ia = (qs & 0xff) | ((qh << (8 - 2 * t)) & 256);
        const uint32_t ibb = (qs >> 8) | ((qh << (7 - 2 * t)) & 256);
        return make_int2(signed4(sg[ia], sb & 0xf), signed4(sg[ibb], sb >> 4));
    } else if constexpr (T == kIq3Xxs) {
        const uint32_t sb = sks[(u32_at2(x + 66 + 4 * ib) >> (7 * t)) & 127];
        const uint32_t qs = u16_at(x + 2 + 8 * ib + 2 * t);
        return make_int2(signed4(sg[qs & 0xff], sb & 0xf), signed4(sg[qs >> 8], sb >> 4));
    } else if constexpr (T == kIq2Xxs) {
        const uint32_t a1 = u32_at2(x + 6 + 8 * ib);
        const uint2 g = reinterpret_cast<const uint2*>(sg)[x[2 + 8 * ib + t]];
        const uint32_t sb = sks[(a1 >> (7 * t)) & 127];
        return make_int2(signed4(g.x, sb & 0xf), signed4(g.y, sb >> 4));
    } else if constexpr (T == kQ4K) {
        const uint32_t* q = reinterpret_cast<const uint32_t*>(x + 16 + 32 * (ib / 2));
        const int sh = 4 * (ib % 2);
        return make_int2(static_cast<int>((q[2 * t] >> sh) & 0x0f0f0f0fu),
                         static_cast<int>((q[2 * t + 1] >> sh) & 0x0f0f0f0fu));
    } else if constexpr (T == kIq4Xs) {  // words t, t + 4: low and high nibbles of the same 4 bytes
        const uint32_t tab[4] = {0xbfad9881u, 0xf6eaddcfu, 0x26190d01u, 0x71594535u};
        return table16(u32_at2(x + 8 + 16 * ib + 4 * t), tab);
    } else {
        return make_int2(word<T>(x, ib, t, sg, sks), word<T>(x, ib, t + 4, sg, sks));
    }
}

// Scales and offsets of the two 16-value halves of sub-block ib (equal halves for 32-value scales).
template <int T>
__device__ __forceinline__ void half_scales(const uint8_t* __restrict__ x, int ib, float (&s)[2], float (&o)[2]) {
    o[0] = o[1] = 0.0f;
    if constexpr (T == kIq3S) {
        const uint32_t v = x[106 + ib / 2];
        s[0] = s[1] = f16_at(x) * static_cast<float>(1 + 2 * (ib % 2 ? v >> 4 : v & 0xf));
    } else if constexpr (T == kIq3Xxs) {
        s[0] = s[1] = f16_at(x) * (0.5f + static_cast<float>(u16_at(x + 68 + 4 * ib) >> 12)) * 0.5f;
    } else if constexpr (T == kIq4Xs) {
        const int ls = static_cast<int>((x[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) |
                       (static_cast<int>((u16_at(x + 2) >> (2 * ib)) & 3) << 4);
        s[0] = s[1] = f16_at(x) * static_cast<float>(ls - 32);
    } else if constexpr (T == kQ4K) {
        const uint8_t* q = x + 4;
        uint32_t sc, m;
        if (ib < 4) {
            sc = q[ib] & 63, m = q[ib + 4] & 63;
        } else {
            sc = (q[ib + 4] & 0xf) | ((q[ib - 4] >> 6) << 4);
            m = (q[ib + 4] >> 4) | ((q[ib] >> 6) << 4);
        }
        s[0] = s[1] = f16_at(x) * static_cast<float>(sc);
        o[0] = o[1] = -f16_at(x + 2) * static_cast<float>(m);
    } else if constexpr (T == kQ2K) {
        const float d = f16_at(x + 80), dmin = f16_at(x + 82);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint32_t sc = x[(ib / 4) * 8 + (ib % 4) * 2 + h];
            s[h] = d * static_cast<float>(sc & 0xf);
            o[h] = -dmin * static_cast<float>(sc >> 4);
        }
    } else if constexpr (T == kIq2Xxs) {
        s[0] = s[1] = f16_at(x) * (0.5f + static_cast<float>(u16_at(x + 8 + 8 * ib) >> 12)) * 0.25f;
    } else {  // IQ2_XS (scales at 66), IQ2_S (scales at 74)
        const uint32_t v = x[(T == kIq2Xs ? 66 : 74) + ib];
        const float d = f16_at(x);
        s[0] = d * (0.5f + static_cast<float>(v & 0xf)) * 0.25f;
        s[1] = d * (0.5f + static_cast<float>(v >> 4)) * 0.25f;
    }
}

__device__ __forceinline__ void mma_k32(int (&c)[4], int a0, int a1, int a2, int a3, int b0, int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_f16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t half2_bits(int lo, int hi) {
    const __half2 h = __halves2half2(__int2half_rn(lo), __int2half_rn(hi));
    return *reinterpret_cast<const uint32_t*>(&h);
}

__device__ __forceinline__ void mma_k16(int (&c)[4], int a0, int a1, int b0) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a0), "r"(a1), "r"(b0));
}

constexpr int kMmaWarps = 4;
#ifndef VELOX_GGML_UNROLL
#define VELOX_GGML_UNROLL 4
#endif
constexpr int kSubUnroll = VELOX_GGML_UNROLL;

__device__ __forceinline__ void cp_async4(void* smem, const void* gmem) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(a), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait1() { asm volatile("cp.async.wait_group 1;\n" ::); }

// Warp per (16 weight rows, K split): the mma's M and NT tiles of 8 tokens (its N); a block's 8 warps
// cover 8 / KS row tiles with KS splits of K each (KS chosen per shape so a projection of a few
// thousand rows still fills the GPU), and the splits meet in shared memory. Each step a warp copies
// its rows' next two blocks (2 * BB bytes, 4-byte aligned since K % 1024 == 0) into shared memory
// with coalesced 4-byte loads, then decodes from there: in the mma fragment layout a lane reads rows
// g and g + 8, so decoding straight from global memory made every load touch 8 rows' cache lines.
// The staged row stride is odd in words, so those 8 rows hit 8 banks. Every sub-block is one k32
// mma (two k16 for 16-value scales), whose int32 result each lane scales for its rows and tokens
// 2t, 2t + 1. Lane (g, t) decodes only the words its A fragment holds: rows g and g + 8, words t and
// t + 4.
template <int T, int NT>
__global__ void __launch_bounds__(kMmaWarps * VELOX_WARP)
    ggml_mma_kernel(const uint8_t* __restrict__ w, long long row_bytes, long long row_begin, int rows, int subs,
                    const int2* __restrict__ frag, const float2* __restrict__ fd, const float4* __restrict__ fs,
                    int m, int tiles, float* __restrict__ y, long long ldy, int ks, int mode) {
    // frag points at the order this type's k-slots use (natural or paired).
    constexpr int BB = block_bytes(T);
    constexpr int W32 = 2 * BB / 4;                                // words of one row's step
    constexpr int SW = W32 | 1;                                    // staged stride (odd)
    constexpr int PER = (16 * W32 + VELOX_WARP - 1) / VELOX_WARP;  // copy words per lane
    constexpr bool kHalves = Traits<T>::G == 2;
    __shared__ uint32_t stage[kMmaWarps][2][16][SW];
    __shared__ uint32_t sg[Grid<T>::kWords];
    __shared__ uint8_t sks[128];
    // Scales (and offsets) of the staged step's 16 rows x 16 sub-blocks, decoded once per step: per
    // lane per sub-block they cost two rows' worth of byte unpacking, four lanes per row. Row stride
    // 17 entries keeps the 8 rows a load touches in different banks. 32-value-scale types without
    // offsets keep the scale only; 16-value-scale types or offsets keep (s0, s1, o0, o1) or (s, o).
    // Only the types whose scales take real unpacking use it (K-quants' 6-bit scale/min pairs, the
    // 16-value-scale IQ2s); a 4-bit scale is cheaper to decode in place than to stage.
    constexpr bool kUseTab = kHalves || Traits<T>::kOff;
    constexpr int kTab = kHalves ? 4 : 2;
    __shared__ float sct[kUseTab ? kMmaWarps : 1][16][17][kTab];
    if constexpr (Grid<T>::kWords > 1) {
        const uint32_t* src = Grid<T>::source();
        for (int i = threadIdx.x; i < Grid<T>::kWords; i += blockDim.x) sg[i] = __ldg(src + i);
    }
    if constexpr (Grid<T>::kSigns) {
        if (threadIdx.x < 128) sks[threadIdx.x] = kKsignsIq2xs[threadIdx.x];
    }
    __syncthreads();
    const int lane = threadIdx.x % VELOX_WARP, g = lane / 4, t = lane % 4, warp = threadIdx.x / VELOX_WARP;
    const int split = warp % ks;
    // grid.x: token chunks of NT tiles (consecutive blocks share their weight rows in L2); grid.y:
    // row blocks.
    const int n0 = blockIdx.x * NT;
    const int r0 = (blockIdx.y * (kMmaWarps / ks) + warp / ks) * 16;
    const bool live = r0 < rows;
    const int steps = subs / 16;
    float acc[NT][4];
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[n][i] = 0.0f;
    // Double-buffered staging: the next step's copy (cp.async, no registers) runs while this step
    // decodes and multiplies, so the warp's own global latency overlaps its work.
    // One 64-bit tile base and 32-bit offsets (rows r0.. are consecutive; the last tile repeats its
    // final row). The copy loop stays unrolled: issuing the copies back to back is what hides their
    // latency, and 32-bit offsets halve the registers 64-bit addresses held.
    const uint8_t* tile = w + (row_begin + r0) * row_bytes;
    const int last = rows - 1 - r0;
    auto issue = [&](int st, int buf) {
        const uint8_t* base = tile + static_cast<long long>(st) * 2 * BB;
#pragma unroll
        for (int i = 0; i < PER; ++i) {
            const int e = lane + i * VELOX_WARP, r = e / W32, c = e % W32;
            const uint32_t off = static_cast<uint32_t>(min(r, last)) * static_cast<uint32_t>(row_bytes) + 4u * c;
            if (e < 16 * W32) cp_async4(&stage[warp][buf][r][c], base + off);
        }
        cp_async_commit();
    };
    const int first = live ? split : steps;
    if (first < steps) issue(first, 0);
    int buf = 0;
    for (int st = first; st < steps; st += ks, buf ^= 1) {
        if (st + ks < steps) issue(st + ks, buf ^ 1);
        else cp_async_commit();  // an empty group keeps wait_group 1 meaning "this step's copy"
        cp_async_wait1();
        __syncwarp();
        const uint8_t* xa = reinterpret_cast<const uint8_t*>(stage[warp][buf][g]);
        const uint8_t* xb = reinterpret_cast<const uint8_t*>(stage[warp][buf][g + 8]);
#pragma unroll
        for (int i = 0; i < (kUseTab ? 8 : 0); ++i) {  // 256 (row, sub-block) entries, 8 per lane
            const int e = lane + i * VELOX_WARP, r = e / 16, jj = e % 16;
            float s2[2], o2[2];
            half_scales<T>(reinterpret_cast<const uint8_t*>(stage[warp][buf][r]) + (jj / 8) * BB, jj % 8, s2, o2);
            float* d = sct[warp][r][jj];
            d[0] = s2[0];
            if constexpr (kTab == 4) {
                d[1] = s2[1];
                d[2] = o2[0];
                d[3] = o2[1];
            } else if constexpr (kTab == 2) {
                d[1] = o2[0];
            }
        }
        __syncwarp();
#pragma unroll kSubUnroll
        for (int j = 0; j < 16; ++j) {
            const int u = st * 16 + j, ib = j % 8, off = (j / 8) * BB;
            const int2 pa = pair<T>(xa + off, ib, t, sg, sks), pb = pair<T>(xb + off, ib, t, sg, sks);
            const int a0 = pa.x, a1 = pb.x, a2 = pa.y, a3 = pb.y;
            float sa[2], oa[2], sb[2], ob[2];
            if constexpr (!kUseTab) {
                half_scales<T>(xa + off, ib, sa, oa);
                half_scales<T>(xb + off, ib, sb, ob);
            } else {
                const float* ta = sct[warp][g][j];
                const float* tb = sct[warp][g + 8][j];
                if constexpr (kTab == 4) {
                    sa[0] = ta[0], sa[1] = ta[1], oa[0] = ta[2], oa[1] = ta[3];
                    sb[0] = tb[0], sb[1] = tb[1], ob[0] = tb[2], ob[1] = tb[3];
                } else {
                    sa[0] = sa[1] = ta[0];
                    sb[0] = sb[1] = tb[0];
                    oa[0] = oa[1] = kTab == 2 ? ta[1] : 0.0f;
                    ob[0] = ob[1] = kTab == 2 ? tb[1] : 0.0f;
                }
            }
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                // Only the 8-tile instance can run past the last token tile (a partial chunk); a
                // runtime branch in the 1- and 2-tile instances kept the compiler from hoisting
                // the next sub-block's loads (decode widths ran 30% slower).
                if constexpr (NT == 8) {
                    if (n0 + n >= tiles) break;
                }
                const long long e = static_cast<long long>(u) * tiles + n0 + n;  // (sub-block, tile)
                const int2 b = __ldg(frag + e * 32 + lane);
                const float2 d2 = __ldg(fd + e * 4 + t);  // tokens 2t, 2t + 1
                int c0[4] = {0, 0, 0, 0}, c1[4] = {0, 0, 0, 0};
                if constexpr (kHalves) {
                    mma_k16(c0, a0, a1, b.x);
                    mma_k16(c1, a2, a3, b.y);
                } else {
                    mma_k32(c0, a0, a1, a2, a3, b.x, b.y);
                }
                float4 z = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if constexpr (Traits<T>::kOff) z = __ldg(fs + e * 4 + t);
#pragma unroll
                for (int q = 0; q < 2; ++q) {  // token columns 2t, 2t + 1 (zero scale past m)
                    const float dx = q ? d2.y : d2.x;
                    float va, vb;
                    if constexpr (kHalves) {
                        va = sa[0] * static_cast<float>(c0[q]) + sa[1] * static_cast<float>(c1[q]);
                        vb = sb[0] * static_cast<float>(c0[2 + q]) + sb[1] * static_cast<float>(c1[2 + q]);
                    } else {
                        va = sa[0] * static_cast<float>(c0[q]);
                        vb = sb[0] * static_cast<float>(c0[2 + q]);
                    }
                    if constexpr (Traits<T>::kOff) {
                        const float z0 = q ? z.z : z.x, z1 = q ? z.w : z.y;
                        va += oa[0] * z0 + oa[1] * z1;
                        vb += ob[0] * z0 + ob[1] * z1;
                    }
                    acc[n][q] += dx * va;
                    acc[n][2 + q] += dx * vb;
                }
            }
        }
        __syncwarp();  // every lane is done with this buffer before the step after next refills it
    }
    if (ks > 1) {
        // The K splits meet in the stage buffer (every warp is past its last step after the barrier).
        static_assert(sizeof(float) * kMmaWarps * NT * 4 * VELOX_WARP <= sizeof(stage), "stage holds the parts");
        float(*part)[NT * 4][VELOX_WARP] = reinterpret_cast<float(*)[NT * 4][VELOX_WARP]>(&stage[0][0][0][0]);
        __syncthreads();
#pragma unroll
        for (int n = 0; n < NT; ++n)
#pragma unroll
            for (int i = 0; i < 4; ++i) part[warp][n * 4 + i][lane] = acc[n][i];
        __syncthreads();
        if (split != 0) return;
        for (int v = 1; v < ks; ++v)
#pragma unroll
            for (int n = 0; n < NT; ++n)
#pragma unroll
                for (int i = 0; i < 4; ++i) acc[n][i] += part[warp + v][n * 4 + i][lane];
    }
    if (!live) return;
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int tok = (n0 + n) * 8 + 2 * t + q;
            if (tok >= m) continue;
            if (r0 + g < rows) put(y + tok * ldy + r0 + g, acc[n][q], mode);
            if (r0 + g + 8 < rows) put(y + tok * ldy + r0 + g + 8, acc[n][2 + q], mode);
        }
}

// From this many tokens the tensor-core kernel serves GGML weights (except IQ1_M);
// VELOX_GGML_MMA_MIN overrides it for tuning.
int mma_min_m() {
    static const int v = [] {
        const char* e = getenv("VELOX_GGML_MMA_MIN");
        return e ? atoi(e) : 5;
    }();
    return v;
}

template <int T>
int launch_mma(const uint8_t* w, long long row_bytes, long long rb, int rows, int subs, const uint8_t* blob,
               int m, float* y, long long ldy, int mode) {
    const ActLayout L(m, static_cast<long long>(subs) * 32);
    const auto* frag = reinterpret_cast<const int2*>(blob + (kPaired<T> ? L.fragp : L.frag));
    const auto* fd = reinterpret_cast<const float2*>(blob + L.fd);
    const auto* fs = reinterpret_cast<const float4*>(blob + L.fs);
    const int row_tiles = (rows + 15) / 16, steps = subs / 16, tok_tiles = L.tiles;
    const int nt = m <= 8 ? 1 : m <= 16 ? 2 : 8;
    const int chunks = (tok_tiles + nt - 1) / nt;
    // K splits: enough warps to fill the GPU (about 1024), at most one step per split.
    int ks = 1;
    while (ks < kMmaWarps && row_tiles * chunks * ks < 1024 && ks * 2 <= steps) ks *= 2;
    const int per_block = kMmaWarps / ks;
    const dim3 grid(chunks, (row_tiles + per_block - 1) / per_block);
    if (grid.y > 65535) return VELOX_ERR_UNSUPPORTED;
    if (nt == 1)
        ggml_mma_kernel<T, 1><<<grid, kMmaWarps * VELOX_WARP>>>(w, row_bytes, rb, rows, subs, frag, fd, fs, m,
                                                                tok_tiles, y, ldy, ks, mode);
    else if (nt == 2)
        ggml_mma_kernel<T, 2><<<grid, kMmaWarps * VELOX_WARP>>>(w, row_bytes, rb, rows, subs, frag, fd, fs, m,
                                                                tok_tiles, y, ldy, ks, mode);
    else
        ggml_mma_kernel<T, 8><<<grid, kMmaWarps * VELOX_WARP>>>(w, row_bytes, rb, rows, subs, frag, fd, fs, m,
                                                                tok_tiles, y, ldy, ks, mode);
    return velox_launch_status();
}

template <int T>
int launch_any(const uint8_t* w, long long row_bytes, long long rb, int rows, int subs, const int8_t* xq,
               const float* xd, int m, float* y, long long ldy, int mode) {
    if constexpr (T != kIq1M && T != kIq4Nl && T != kIq1S) {
        if (m >= mma_min_m())
            return launch_mma<T>(w, row_bytes, rb, rows, subs, reinterpret_cast<const uint8_t*>(xq), m, y, ldy,
                                 mode);
    }
    const long long k = static_cast<long long>(subs) * 32;
    for (int t0 = 0; t0 < m; t0 += 16) {
        const int mc = m - t0 < 16 ? m - t0 : 16;
        if (const int e = launch_t<T>(w, row_bytes, rb, rows, subs, xq + t0 * k, xd + static_cast<long long>(t0) * subs,
                                      mc, y + t0 * ldy, ldy, mode))
            return e;
    }
    return 0;
}


// ---- Prefill: int8 tensor cores on Int8G256 activations (layout 0) ---------------------------
//
// Every GGML type's per-32 (or per-16) scale is an integer multiple of its super-block's f16 d:
// (1 + 2s) for IQ3_S, (2s + 1) / 4 for IQ3_XXS, (ls - 32) for IQ4_XS, the 6-bit sc and m of Q4_K
// (d sc q - dmin m), (2s + 1) / 8 for IQ2_*, IQ1_M's (2s + 1) (g + delta) as codes 8g +- 1 over 8.
// So within a super-block each k32 (k16) product accumulates in int32 times that integer, and the
// float scaling (d, and the activation scale of one 256-value group) happens once per super-block.
// Block: kPM tokens x kPN weight rows, warps of 32 x 32 (2 along tokens x 2 along rows); K
// advances one super-block per step: activations by cp.async, weights decoded by the block into
// int8 codes in shared memory (rows of 272 bytes: ldmatrix rows 68 words apart, conflict-free).

// 64 tokens x 64 rows, 4 warps (2 x 2): 47 KB of shared memory, so two blocks share an SM and one
// computes while the other waits on its step's loads (one 8-warp 128-token block per SM left the
// tensor cores idle through every step's weight decode and copy).
constexpr int kPM = 64, kPN = 64, kPThreads = 128, kPS = 256 + 16;

struct PreSmem {
    int8_t a[kPM * kPS];
    int8_t b[kPN * kPS];
    int sc[kPN][16];      // per row: integer scale per half sub-block (16 values)
    int mn[kPN][16];      // per row: integer minimum per half sub-block (offset types)
    float d[kPN][2];      // per row: the super-block's d (with the type's constant) and dmin
    int xs[kPM][16];      // per token: code sums per 16 values (offset types)
};

// Codes of sub-block ib (eight int8x4 words), the integer factors of its halves and the block's
// float multipliers, so that value = dm.x * sc * code + dm.y * mn (dm.y = -dmin, 0 without minima).
template <int T>
__device__ __forceinline__ void qdecode(const uint8_t* __restrict__ x, int ib, int (&wd)[8], int (&sc)[2], int (&mn)[2],
                                        float2& dm) {
    mn[0] = mn[1] = 0;
    dm.y = 0.0f;
    if constexpr (T == kIq1S) {  // codes 8 g +- 1 over 8, one integer scale (2 s + 1) per 32 values
        dm.x = f16_at(x) * 0.125f;
        const uint32_t qh = u16_at(x + 34 + 2 * ib);
        sc[0] = sc[1] = 2 * static_cast<int>((qh >> 12) & 7) + 1;
        const int dl = qh & 0x8000 ? -1 : 1;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t idx = x[2 + 4 * ib + l] | (((qh >> (3 * l)) & 7) << 8);
            const uint64_t g = __ldg(&kIq1SGrid[idx]);
#pragma unroll
            for (int hw = 0; hw < 2; ++hw) {
                uint32_t v = 0;
#pragma unroll
                for (int bb = 0; bb < 4; ++bb) {
                    const int gv = static_cast<int8_t>((g >> (8 * (4 * hw + bb))) & 0xff);
                    v |= (static_cast<uint32_t>(8 * gv + dl) & 0xffu) << (8 * bb);
                }
                wd[2 * l + hw] = static_cast<int>(v);
            }
        }
        return;
    }
    if constexpr (T == kIq1M) {
        const uint32_t s0 = u16_at(x + 48), s1 = u16_at(x + 50), s2 = u16_at(x + 52), s3 = u16_at(x + 54);
        const uint32_t dbits = (s0 >> 12) | ((s1 >> 8) & 0x00f0) | ((s2 >> 4) & 0x0f00) | (s3 & 0xf000);
        dm.x = __half2float(__ushort_as_half(static_cast<unsigned short>(dbits))) * 0.125f;
        const uint32_t w = u16_at(x + 48 + 2 * (ib / 2));
#pragma unroll
        for (int h = 0; h < 2; ++h) sc[h] = 2 * static_cast<int>((w >> (6 * (ib % 2) + 3 * h)) & 7) + 1;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t qh = x[32 + 2 * ib + l / 2];
            const uint32_t idx = x[4 * ib + l] | ((qh << (l % 2 ? 4 : 8)) & 0x700);
            const int dl = qh & (l % 2 ? 0x80 : 0x08) ? -1 : 1;
            const uint64_t g = __ldg(&kIq1SGrid[idx]);
#pragma unroll
            for (int hw = 0; hw < 2; ++hw) {
                uint32_t v = 0;
#pragma unroll
                for (int bb = 0; bb < 4; ++bb) {
                    const int gv = static_cast<int8_t>((g >> (8 * (4 * hw + bb))) & 0xff);
                    v |= (static_cast<uint32_t>(8 * gv + dl) & 0xffu) << (8 * bb);
                }
                wd[2 * l + hw] = static_cast<int>(v);
            }
        }
        return;
    }
    const Sub<T> sub = decode<T>(x, ib);
#pragma unroll
    for (int i = 0; i < 8; ++i) wd[i] = sub.w[i];
    if constexpr (T == kIq3S) {
        const uint32_t v = x[106 + ib / 2];
        sc[0] = sc[1] = 1 + 2 * static_cast<int>(ib % 2 ? v >> 4 : v & 0xf);
        dm.x = f16_at(x);
    } else if constexpr (T == kIq3Xxs) {
        sc[0] = sc[1] = 2 * static_cast<int>(u16_at(x + 68 + 4 * ib) >> 12) + 1;
        dm.x = f16_at(x) * 0.25f;
    } else if constexpr (T == kIq4Xs) {
        const int ls = static_cast<int>((x[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) |
                       (static_cast<int>((u16_at(x + 2) >> (2 * ib)) & 3) << 4);
        sc[0] = sc[1] = ls - 32;
        dm.x = f16_at(x);
    } else if constexpr (T == kQ4K) {
        const uint8_t* q = x + 4;
        int a, m;
        if (ib < 4) {
            a = q[ib] & 63, m = q[ib + 4] & 63;
        } else {
            a = (q[ib + 4] & 0xf) | ((q[ib - 4] >> 6) << 4);
            m = (q[ib + 4] >> 4) | ((q[ib] >> 6) << 4);
        }
        sc[0] = sc[1] = a;
        mn[0] = mn[1] = m;
        dm.x = f16_at(x);
        dm.y = -f16_at(x + 2);
    } else if constexpr (T == kQ2K) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint32_t v = x[(ib / 4) * 8 + (ib % 4) * 2 + h];
            sc[h] = static_cast<int>(v & 0xf);
            mn[h] = static_cast<int>(v >> 4);
        }
        dm.x = f16_at(x + 80);
        dm.y = -f16_at(x + 82);
    } else if constexpr (T == kIq2Xxs) {
        sc[0] = sc[1] = 2 * static_cast<int>(u16_at(x + 8 + 8 * ib) >> 12) + 1;
        dm.x = f16_at(x) * 0.125f;
    } else {  // IQ2_XS (scales at 66), IQ2_S (scales at 74)
        const uint32_t v = x[(T == kIq2Xs ? 66 : 74) + ib];
        sc[0] = 2 * static_cast<int>(v & 0xf) + 1;
        sc[1] = 2 * static_cast<int>(v >> 4) + 1;
        dm.x = f16_at(x) * 0.125f;
    }
}

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(a), "l"(gmem));
}
__device__ __forceinline__ void cp_async_wait0() { asm volatile("cp.async.wait_group 0;\n" ::); }

template <int T>
__global__ void __launch_bounds__(kPThreads)
    ggml_prefill_kernel(const int8_t* __restrict__ xq, const float* __restrict__ sx, int stride, int m,
                        const uint8_t* __restrict__ w, long long row_bytes, long long row_begin, int rows, int k,
                        float* __restrict__ y, long long ldy, int mode) {
    constexpr int BB = block_bytes(T);
    constexpr bool kHalves = Traits<T>::G != 1;  // IQ1_M too: its scales cover 16 values
    constexpr bool kOff = T == kQ2K || T == kQ4K;
    extern __shared__ __align__(128) uint8_t smem_raw[];
    PreSmem& s = *reinterpret_cast<PreSmem*>(smem_raw);
    const int m0 = blockIdx.x * kPM, n0 = blockIdx.y * kPN;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, wm = warp >> 1, wn = warp & 1;
    const int g = lane >> 2, q4 = lane & 3;
    float acc[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
    // ldmatrix lane addressing: A (tokens x k) and B (weight rows x k), 16 rows x 32 bytes per x4.
    const int a_row = (lane & 7) + (((lane >> 3) & 1) << 3), a_col = (lane >> 4) << 4;
    const int b_row = (lane & 7) + ((lane >> 4) << 3), b_col = ((lane >> 3) & 1) << 4;
    const int blocks = k / 256;
    for (int kb = 0; kb < blocks; ++kb) {
        // Activations: 128 tokens x 256 bytes (tokens past m repeat the last one; never written).
        for (int idx = threadIdx.x; idx < kPM * 16; idx += kPThreads) {
            const int r = idx >> 4, c = idx & 15;
            const int tok = min(m0 + r, m - 1);
            cp_async16(s.a + r * kPS + c * 16, xq + static_cast<long long>(tok) * k + kb * 256 + c * 16);
        }
        cp_async_commit();
        // Weights: kPN rows x 8 sub-blocks.
#pragma unroll
        for (int i = 0; i < kPN * 8 / kPThreads; ++i) {
            const int e = threadIdx.x + i * kPThreads, r = e >> 3, ib = e & 7;
            int wd[8] = {0, 0, 0, 0, 0, 0, 0, 0}, sc[2] = {0, 0}, mn[2] = {0, 0};
            float2 dm = make_float2(0.0f, 0.0f);
            if (n0 + r < rows)
                qdecode<T>(w + (row_begin + n0 + r) * row_bytes + static_cast<long long>(kb) * BB, ib, wd, sc, mn, dm);
            int4* dst = reinterpret_cast<int4*>(s.b + r * kPS + ib * 32);
            dst[0] = make_int4(wd[0], wd[1], wd[2], wd[3]);
            dst[1] = make_int4(wd[4], wd[5], wd[6], wd[7]);
            s.sc[r][2 * ib] = sc[0], s.sc[r][2 * ib + 1] = sc[1];
            if constexpr (kOff) s.mn[r][2 * ib] = mn[0], s.mn[r][2 * ib + 1] = mn[1];
            if (ib == 0) s.d[r][0] = dm.x, s.d[r][1] = dm.y;
        }
        cp_async_wait0();
        __syncthreads();
        if constexpr (kOff) {  // code sums per (token, 16 values)
            for (int e = threadIdx.x; e < kPM * 16; e += kPThreads) {
                const int r = e >> 4, h = e & 15;
                const int4 v = *reinterpret_cast<const int4*>(s.a + r * kPS + h * 16);
                int z = __dp4a(v.x, 0x01010101, 0);
                z = __dp4a(v.y, 0x01010101, z);
                z = __dp4a(v.z, 0x01010101, z);
                s.xs[r][h] = __dp4a(v.w, 0x01010101, z);
            }
            __syncthreads();
        }
        int ia[2][4][4];
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) ia[i][j][0] = ia[i][j][1] = ia[i][j][2] = ia[i][j][3] = 0;
        int im[2][4][4];
        if constexpr (kOff) {
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) im[i][j][0] = im[i][j][1] = im[i][j][2] = im[i][j][3] = 0;
        }
#pragma unroll 2
        for (int sb = 0; sb < 8; ++sb) {
            uint32_t af[2][4], bfr[2][4];
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) ldsm_x4(af[mt], s.a + (wm * 32 + mt * 16 + a_row) * kPS + sb * 32 + a_col);
#pragma unroll
            for (int np = 0; np < 2; ++np) ldsm_x4(bfr[np], s.b + (wn * 32 + np * 16 + b_row) * kPS + sb * 32 + b_col);
            // bfr[np]: n-tile 2np (regs 0, 1 = k 0-15, 16-31) and 2np + 1 (regs 2, 3).
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const uint32_t b0 = bfr[nt >> 1][(nt & 1) * 2], b1 = bfr[nt >> 1][(nt & 1) * 2 + 1];
                const int n = wn * 32 + nt * 8 + 2 * q4;  // this lane's rows n, n + 1
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    if constexpr (kHalves) {
                        int c0[4] = {0, 0, 0, 0}, c1[4] = {0, 0, 0, 0};
                        mma_k16(c0, af[mt][0], af[mt][1], b0);
                        mma_k16(c1, af[mt][2], af[mt][3], b1);
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const int nn = n + (j & 1);
                            ia[mt][nt][j] += c0[j] * s.sc[nn][2 * sb] + c1[j] * s.sc[nn][2 * sb + 1];
                            if constexpr (kOff) {
                                const int tk = wm * 32 + mt * 16 + g + 8 * (j >> 1);
                                im[mt][nt][j] += s.mn[nn][2 * sb] * s.xs[tk][2 * sb] +
                                                 s.mn[nn][2 * sb + 1] * s.xs[tk][2 * sb + 1];
                            }
                        }
                    } else {
                        int c[4] = {0, 0, 0, 0};
                        mma_k32(c, af[mt][0], af[mt][1], af[mt][2], af[mt][3], b0, b1);
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const int nn = n + (j & 1);
                            ia[mt][nt][j] += c[j] * s.sc[nn][2 * sb];
                            if constexpr (kOff) {
                                const int tk = wm * 32 + mt * 16 + g + 8 * (j >> 1);
                                im[mt][nt][j] += s.mn[nn][2 * sb] * (s.xs[tk][2 * sb] + s.xs[tk][2 * sb + 1]);
                            }
                        }
                    }
                }
            }
        }
        // Super-block epilogue: d per row, the 256-value activation scale per token.
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
            for (int j2 = 0; j2 < 2; ++j2) {
                const int tok = min(m0 + wm * 32 + mt * 16 + g + 8 * j2, m - 1);
                const float a = sx[static_cast<long long>(kb) * stride + tok];
#pragma unroll
                for (int nt = 0; nt < 4; ++nt)
#pragma unroll
                    for (int j1 = 0; j1 < 2; ++j1) {
                        const int j = 2 * j2 + j1, nn = wn * 32 + nt * 8 + 2 * q4 + j1;
                        float v = s.d[nn][0] * static_cast<float>(ia[mt][nt][j]);
                        if constexpr (kOff) v += s.d[nn][1] * static_cast<float>(im[mt][nt][j]);
                        acc[mt][nt][j] += a * v;
                    }
            }
        __syncthreads();
    }
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int tok = m0 + wm * 32 + mt * 16 + g + 8 * (j >> 1);
                const int n = n0 + wn * 32 + nt * 8 + 2 * q4 + (j & 1);
                if (tok < m && n < rows) put(y + static_cast<long long>(tok) * ldy + n, acc[mt][nt][j], mode);
            }
}

template <int T>
int launch_pre(const int8_t* xq, const float* sx, int stride, int m, const uint8_t* w, long long row_bytes,
               long long rb, int rows, int k, float* y, long long ldy, int mode) {
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(ggml_prefill_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(PreSmem));
        configured = true;
    }
    const dim3 grid((m + kPM - 1) / kPM, (rows + kPN - 1) / kPN);
    if (grid.y > 65535) return VELOX_ERR_UNSUPPORTED;
    ggml_prefill_kernel<T><<<grid, kPThreads, sizeof(PreSmem)>>>(xq, sx, stride, m, w, row_bytes, rb, rows, k, y,
                                                                 ldy, mode);
    return velox_launch_status();
}


// ---- Prefill v2: weights decoded straight into the A fragments, activations through a ring ------
//
// The design of the IQ3_XXS int8 kernel (rtx5060_q8.cu): each warp owns 16 weight rows and decodes
// the words its lanes' A fragments hold (rows g, g + 8; natural k order) from global memory, which
// a 64-token tile amortizes; the token tile's int8 activations stream through a two-stage cp.async
// ring and reach the mma as B through ldmatrix. Per sub-block the int32 product times the row's
// integer scale accumulates in int32; per super-block that times d and the token's activation
// scale goes to f32 (see the integer factors above). Block: 8 warps = 128 rows x 64 tokens.

constexpr int kQ2Warps = 8, kQ2Tokens = 64, kQ2S = 256 + 16;
#ifndef VELOX_GGML_PRE_UNROLL
#define VELOX_GGML_PRE_UNROLL 8
#endif
constexpr int kPreSbUnroll = VELOX_GGML_PRE_UNROLL;

// The integer factors of sub-block ib's halves and the super-block's float multipliers (codes come
// from word<T>): value = dm.x * sc * code + dm.y * mn.
template <int T>
__device__ __forceinline__ void int_scales(const uint8_t* __restrict__ x, int ib, int (&sc)[2], int (&mn)[2],
                                           float2& dm) {
    mn[0] = mn[1] = 0;
    dm.y = 0.0f;
    if constexpr (T == kIq3S) {
        const uint32_t v = x[106 + ib / 2];
        sc[0] = sc[1] = 1 + 2 * static_cast<int>(ib % 2 ? v >> 4 : v & 0xf);
        dm.x = f16_at(x);
    } else if constexpr (T == kIq3Xxs) {
        sc[0] = sc[1] = 2 * static_cast<int>(u16_at(x + 68 + 4 * ib) >> 12) + 1;
        dm.x = f16_at(x) * 0.25f;
    } else if constexpr (T == kIq4Xs) {
        const int ls = static_cast<int>((x[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) |
                       (static_cast<int>((u16_at(x + 2) >> (2 * ib)) & 3) << 4);
        sc[0] = sc[1] = ls - 32;
        dm.x = f16_at(x);
    } else if constexpr (T == kQ4K) {
        const uint8_t* q = x + 4;
        int a, m;
        if (ib < 4) {
            a = q[ib] & 63, m = q[ib + 4] & 63;
        } else {
            a = (q[ib + 4] & 0xf) | ((q[ib - 4] >> 6) << 4);
            m = (q[ib + 4] >> 4) | ((q[ib] >> 6) << 4);
        }
        sc[0] = sc[1] = a;
        mn[0] = mn[1] = m;
        dm.x = f16_at(x);
        dm.y = -f16_at(x + 2);
    } else if constexpr (T == kQ2K) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint32_t v = x[(ib / 4) * 8 + (ib % 4) * 2 + h];
            sc[h] = static_cast<int>(v & 0xf);
            mn[h] = static_cast<int>(v >> 4);
        }
        dm.x = f16_at(x + 80);
        dm.y = -f16_at(x + 82);
    } else if constexpr (T == kIq2Xxs) {
        sc[0] = sc[1] = 2 * static_cast<int>(u16_at(x + 8 + 8 * ib) >> 12) + 1;
        dm.x = f16_at(x) * 0.125f;
    } else if constexpr (T == kIq1S) {  // codes 8 g +- 1 over 8, one scale per 32 values
        sc[0] = sc[1] = 2 * static_cast<int>((u16_at(x + 34 + 2 * ib) >> 12) & 7) + 1;
        dm.x = f16_at(x) * 0.125f;
    } else if constexpr (T == kIq1M) {  // codes 8 g +- 1 over 8, scales per 16 values
        const uint32_t s0 = u16_at(x + 48), s1 = u16_at(x + 50), s2 = u16_at(x + 52), s3 = u16_at(x + 54);
        const uint32_t dbits = (s0 >> 12) | ((s1 >> 8) & 0x00f0) | ((s2 >> 4) & 0x0f00) | (s3 & 0xf000);
        dm.x = __half2float(__ushort_as_half(static_cast<unsigned short>(dbits))) * 0.125f;
        const uint32_t w = u16_at(x + 48 + 2 * (ib / 2));
        sc[0] = 2 * static_cast<int>((w >> (6 * (ib % 2))) & 7) + 1;
        sc[1] = 2 * static_cast<int>((w >> (6 * (ib % 2) + 3)) & 7) + 1;
    } else {  // IQ2_XS (scales at 66), IQ2_S (scales at 74)
        const uint32_t v = x[(T == kIq2Xs ? 66 : 74) + ib];
        sc[0] = 2 * static_cast<int>(v & 0xf) + 1;
        sc[1] = 2 * static_cast<int>(v >> 4) + 1;
        dm.x = f16_at(x) * 0.125f;
    }
}

template <int T>
__global__ void __launch_bounds__(kQ2Warps * VELOX_WARP)
    ggml_pre2_kernel(const int8_t* __restrict__ xq, const float* __restrict__ sx, int stride, int m,
                     const uint8_t* __restrict__ w, long long row_bytes, long long row_begin, int rows, int k,
                     float* __restrict__ y, long long ldy, int mode) {
    constexpr int BB = block_bytes(T);
    constexpr int NT = kQ2Tokens / 8;
    // Q2_K: code q <= 3 times its 4-bit scale stays <= 45, so the per-16 scales fold into the
    // codes and a sub-block is one k32 mma accumulating straight into the super-block's int32
    // sums (two k16 mmas issue at the k32 rate on this GPU, and the per-tile rescaling goes).
    constexpr bool kFold = T == kQ2K;
    // IQ2_XS / IQ2_S: magnitude * scale reaches 1333, so the scaled codes go out as two int8 planes
    // (16 hi + lo), each one k32 mma per sub-block into its own super-block sums: the tensor-core
    // time of the two k16 mmas, without the per-tile rescale.
    constexpr bool kSplit = T == kIq2S || T == kIq2Xs || T == kIq1M;  // IQ1_M: |code| 9 * 15
    constexpr bool kHalves = Traits<T>::G != 1 && !kFold && !kSplit;
    constexpr bool kOff = T == kQ2K || T == kQ4K;
    __shared__ __align__(16) int8_t act[2][kQ2Tokens * kQ2S];
    __shared__ float sxs[2][kQ2Tokens];
    // Each warp's 16 rows of the super-block, copied from the 4-byte aligned address at or below it
    // (blocks are 2-byte aligned; row ends are 4-byte aligned, so the copy stays inside the row).
    // Odd word stride: the 8 rows a decode load touches hit 8 banks.
    constexpr int kWW = (BB + 2 + 3) / 4 | 1;
    __shared__ uint32_t wst[kQ2Warps][2][16][kWW];
    // Code sums per (token, 16 values) as f16 (exact: |sum| <= 2032), row stride 9 words: the B
    // operand of the minimum term's f16 mma.
    __shared__ __align__(16) __half xs[kOff ? kQ2Tokens : 1][18];
    __shared__ uint32_t sg[Grid<T>::kWords];
    __shared__ uint8_t sks[128];
    if constexpr (Grid<T>::kWords > 1) {
        const uint32_t* src = Grid<T>::source();
        for (int i = threadIdx.x; i < Grid<T>::kWords; i += blockDim.x) sg[i] = __ldg(src + i);
    }
    if constexpr (Grid<T>::kSigns) {
        if (threadIdx.x < 128) sks[threadIdx.x] = kKsignsIq2xs[threadIdx.x];
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int t0 = blockIdx.x * kQ2Tokens;
    const int r0 = blockIdx.y * (16 * kQ2Warps) + warp * 16;
    // A warp past the last row stages the last row (its results are not stored): its own rows
    // would be outside the weights.
    const uint8_t* tile = w + (row_begin + min(r0, rows - 1)) * row_bytes;
    const int last = max(rows - 1 - r0, 0);
    const int b_row = (lane & 7) + ((lane >> 4) << 3), b_col = ((lane >> 3) & 1) << 4;
    const int blocks = k / 256;
    auto stage = [&](int buf, int kb) {
        for (int i = threadIdx.x; i < kQ2Tokens * 16; i += blockDim.x) {
            const int r = i >> 4, c = i & 15;
            const int tok = min(t0 + r, m - 1);
            cp_async16(&act[buf][r * kQ2S + c * 16], xq + static_cast<long long>(tok) * k + kb * 256 + c * 16);
        }
        // This warp's rows (one commit group with the activations).
        const long long boff = static_cast<long long>(kb) * BB;
        const int words = static_cast<int>(((boff & 3) + BB + 3) / 4);
        const uint8_t* src = tile + (boff & ~3ll);
        for (int e = lane; e < 16 * words; e += VELOX_WARP) {
            const int r = e / words, c = e % words;
            cp_async4(&wst[warp][buf][r][c],
                      src + static_cast<uint32_t>(min(r, last)) * static_cast<uint32_t>(row_bytes) + 4u * c);
        }
        cp_async_commit();
        if (threadIdx.x < kQ2Tokens)
            sxs[buf][threadIdx.x] = sx[static_cast<long long>(kb) * stride + min(t0 + static_cast<int>(threadIdx.x), m - 1)];
    };
    float acc[NT][4];
#pragma unroll
    for (int n = 0; n < NT; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;
    stage(0, 0);
    for (int kb = 0; kb < blocks; ++kb) {
        const int cur = kb & 1;
        if (kb + 1 < blocks) stage(cur ^ 1, kb + 1);
        else cp_async_commit();
        cp_async_wait1();
        __syncthreads();
        if constexpr (kOff) {  // code sums per (token, 16 values) of this super-block
            for (int e = threadIdx.x; e < kQ2Tokens * 16; e += blockDim.x) {
                const int r = e >> 4, h = e & 15;
                const int4 v = *reinterpret_cast<const int4*>(&act[cur][r * kQ2S + h * 16]);
                int z = __dp4a(v.x, 0x01010101, 0);
                z = __dp4a(v.y, 0x01010101, z);
                z = __dp4a(v.z, 0x01010101, z);
                xs[r][h] = __int2half_rn(__dp4a(v.w, 0x01010101, z));
            }
            __syncthreads();
        }
        const int off = static_cast<int>((static_cast<long long>(kb) * BB) & 3);
        const uint8_t* xa = reinterpret_cast<const uint8_t*>(wst[warp][cur][g]) + off;
        const uint8_t* xb = reinterpret_cast<const uint8_t*>(wst[warp][cur][g + 8]) + off;
        int ia[NT][4];
#pragma unroll
        for (int n = 0; n < NT; ++n) ia[n][0] = ia[n][1] = ia[n][2] = ia[n][3] = 0;
        int il[kSplit ? NT : 1][4];
#pragma unroll
        for (int n = 0; n < (kSplit ? NT : 1); ++n) il[n][0] = il[n][1] = il[n][2] = il[n][3] = 0;
        // The minimum term sum_h mn[row][h] * xs[token][h] over the super-block's 16 halves is a
        // 16 x 16 x 8 product per token tile: one f16 mma (exact: mins <= 63, sums <= 2032, f32
        // accumulation). Lane (g, t) holds the A operand's halves 2t, 2t + 1 (sub-block t) and
        // 8 + 2t, 9 + 2t (sub-block 4 + t) of rows g, g + 8.
        uint32_t ma[4] = {0u, 0u, 0u, 0u};
        float2 dma, dmb;
#pragma unroll kPreSbUnroll
        for (int sb = 0; sb < 8; ++sb) {
            int2 pa, pb, la, lb;
            if constexpr (kSplit) {  // filled from the scales below
            } else if constexpr (T == kIq4Xs) {  // words t, t + 4 from one lookup
                pa = pair<T>(xa, sb, t, sg, sks);
                pb = pair<T>(xb, sb, t, sg, sks);
            } else {
                pa = make_int2(word<T>(xa, sb, t, sg, sks), word<T>(xa, sb, t + 4, sg, sks));
                pb = make_int2(word<T>(xb, sb, t, sg, sks), word<T>(xb, sb, t + 4, sg, sks));
            }
            int sca[2], mna[2], scb[2], mnb[2];
            int_scales<T>(xa, sb, sca, mna, dma);
            int_scales<T>(xb, sb, scb, mnb, dmb);
            if constexpr (kSplit) {  // pa / pb take the hi planes, la / lb the lo planes
                split_planes(mag_sign<T>(xa, sb, t, sg, sks), sca[0], pa.x, la.x);
                split_planes(mag_sign<T>(xa, sb, t + 4, sg, sks), sca[1], pa.y, la.y);
                split_planes(mag_sign<T>(xb, sb, t, sg, sks), scb[0], pb.x, lb.x);
                split_planes(mag_sign<T>(xb, sb, t + 4, sg, sks), scb[1], pb.y, lb.y);
            }
            if constexpr (kFold) {  // bytes q * sc <= 45: one multiply per word, no carries
                pa.x = static_cast<int>(static_cast<uint32_t>(pa.x) * static_cast<uint32_t>(sca[0]));
                pa.y = static_cast<int>(static_cast<uint32_t>(pa.y) * static_cast<uint32_t>(sca[1]));
                pb.x = static_cast<int>(static_cast<uint32_t>(pb.x) * static_cast<uint32_t>(scb[0]));
                pb.y = static_cast<int>(static_cast<uint32_t>(pb.y) * static_cast<uint32_t>(scb[1]));
            }
            if constexpr (kOff) {
                if (sb == t) ma[0] = half2_bits(mna[0], mna[1]), ma[1] = half2_bits(mnb[0], mnb[1]);
                if (sb == t + 4) ma[2] = half2_bits(mna[0], mna[1]), ma[3] = half2_bits(mnb[0], mnb[1]);
            }
#pragma unroll
            for (int np = 0; np < NT / 2; ++np) {
                uint32_t bf[4];
                ldsm_x4(bf, &act[cur][(np * 16 + b_row) * kQ2S + sb * 32 + b_col]);
#pragma unroll
                for (int h2 = 0; h2 < 2; ++h2) {
                    const int n = 2 * np + h2;
                    const uint32_t b0 = bf[2 * h2], b1 = bf[2 * h2 + 1];
                    if constexpr (kSplit) {
                        mma_k32(ia[n], pa.x, pb.x, pa.y, pb.y, b0, b1);
                        mma_k32(il[n], la.x, lb.x, la.y, lb.y, b0, b1);
                    } else if constexpr (kFold) {
                        mma_k32(ia[n], pa.x, pb.x, pa.y, pb.y, b0, b1);
                    } else if constexpr (kHalves) {
                        int c0[4] = {0, 0, 0, 0}, c1[4] = {0, 0, 0, 0};
                        mma_k16(c0, pa.x, pb.x, b0);
                        mma_k16(c1, pa.y, pb.y, b1);
                        ia[n][0] += c0[0] * sca[0] + c1[0] * sca[1];
                        ia[n][1] += c0[1] * sca[0] + c1[1] * sca[1];
                        ia[n][2] += c0[2] * scb[0] + c1[2] * scb[1];
                        ia[n][3] += c0[3] * scb[0] + c1[3] * scb[1];
                    } else {
                        int c[4] = {0, 0, 0, 0};
                        mma_k32(c, pa.x, pb.x, pa.y, pb.y, b0, b1);
                        ia[n][0] += c[0] * sca[0];
                        ia[n][1] += c[1] * sca[0];
                        ia[n][2] += c[2] * scb[0];
                        ia[n][3] += c[3] * scb[0];
                    }
                }
            }
        }
        if constexpr (kSplit) {
#pragma unroll
            for (int n = 0; n < NT; ++n)
#pragma unroll
                for (int j = 0; j < 4; ++j) ia[n][j] = 16 * ia[n][j] + il[n][j];
        }
        // Super-block epilogue: d per row (rows g, g + 8), the activation scale per token.
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            float off[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if constexpr (kOff) {
                const uint32_t* xr = reinterpret_cast<const uint32_t*>(xs[n * 8 + g]);
                mma_f16(off, ma[0], ma[1], ma[2], ma[3], xr[t], xr[4 + t]);
            }
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const float a = sxs[cur][n * 8 + 2 * t + q];
                float va = dma.x * static_cast<float>(ia[n][q]), vb = dmb.x * static_cast<float>(ia[n][2 + q]);
                if constexpr (kOff) {
                    va += dma.y * off[q];
                    vb += dmb.y * off[2 + q];
                }
                acc[n][q] += a * va;
                acc[n][2 + q] += a * vb;
            }
        }
        __syncthreads();  // every warp is done with this stage before it is refilled
    }
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int tok = t0 + n * 8 + 2 * t + q;
            if (tok >= m) continue;
            if (r0 + g < rows) put(y + static_cast<long long>(tok) * ldy + r0 + g, acc[n][q], mode);
            if (r0 + g + 8 < rows) put(y + static_cast<long long>(tok) * ldy + r0 + g + 8, acc[n][2 + q], mode);
        }
}

template <int T>
int launch_pre2(const int8_t* xq, const float* sx, int stride, int m, const uint8_t* w, long long row_bytes,
                long long rb, int rows, int k, float* y, long long ldy, int mode) {
    const dim3 grid((m + kQ2Tokens - 1) / kQ2Tokens, (rows + 16 * kQ2Warps - 1) / (16 * kQ2Warps));
    if (grid.y > 65535) return VELOX_ERR_UNSUPPORTED;
    ggml_pre2_kernel<T><<<grid, kQ2Warps * VELOX_WARP>>>(xq, sx, stride, m, w, row_bytes, rb, rows, k, y, ldy, mode);
    return velox_launch_status();
}

VeloxScratch g_act;

}  // namespace

// Quantizes x [m, k] (k % 32 == 0, 1 <= m <= 16) as ActQuant::Int8G32 into this file's scratch and
// returns the codes ([m, k] int8, 16-byte aligned rows; the blob's base, see ActLayout) and scales
// ([m, k / 32] f32).
VELOX_API int velox_5060_ggml_act(const float* x, int m, long long k, const int8_t** q, const float** d) {
    if (m < 1 || m > 65536 || k <= 0 || k % 32 != 0) return VELOX_ERR_UNSUPPORTED;
    const long long codes = static_cast<long long>(m) * k, groups = codes / 32;
    const ActLayout L(m, k);
    uint8_t* base = static_cast<uint8_t*>(g_act.get(L.bytes));
    if (!base) return static_cast<int>(cudaErrorMemoryAllocation);
    int8_t* qp = reinterpret_cast<int8_t*>(base);
    float* dp = reinterpret_cast<float*>(base + L.d);
    float* sp = reinterpret_cast<float*>(base + L.sums);
    const long long warps = static_cast<long long>(L.tiles) * 8 * (k / 32);
    quantize_act_kernel<<<static_cast<unsigned>((warps * VELOX_WARP + 255) / 256), 256>>>(
        x, m, static_cast<int>(k / 32), L.tiles, qp, dp, sp, reinterpret_cast<int2*>(base + L.frag),
        reinterpret_cast<int2*>(base + L.fragp), reinterpret_cast<float*>(base + L.fd),
        reinterpret_cast<float*>(base + L.fs));
    *q = qp;
    *d = dp;
    return velox_launch_status();
}

// y[t * ldy + r] (r < rows) = sum_c W[row_begin + r, c] * xq[t, c] for 1 <= m <= 16 token rows of
// activations from velox_5060_ggml_act, W of GGML type `type` (0..8, GgmlType::ALL order) with k %
// 1024 == 0, or IQ4_NL (10) with k % 32 == 0 (dp4a only). mode 1 instead writes silu(y) * that
// (the up projection of a SwiGLU after its gate). VELOX_ERR_UNSUPPORTED otherwise.
VELOX_API int velox_5060_linear_ggml_from(const int8_t* xq, const float* xd, int m, const uint8_t* w, int type,
                                          long long k, long long row_begin, int rows, float* y, long long ldy,
                                          int mode) {
    // IQ4_NL (32-value blocks, 2-byte aligned): dp4a at every width, any k % 32 == 0.
    if (type == kIq4Nl) {
        if (m < 1 || m > 65536 || rows <= 0 || k <= 0 || k % 32 != 0 || reinterpret_cast<uintptr_t>(w) % 16 != 0 ||
            mode < 0 || mode > 1)
            return VELOX_ERR_UNSUPPORTED;
        return launch_any<kIq4Nl>(w, k / 32 * block_bytes(kIq4Nl), row_begin, rows, static_cast<int>(k / 32), xq, xd,
                                  m, y, ldy, mode);
    }
    if (m < 1 || m > 65536 || rows <= 0 || k % 256 != 0 || type < 0 || (type > 8 && type != kIq1S))
        return VELOX_ERR_UNSUPPORTED;
    const long long row_bytes = k / 256 * block_bytes(type);
    // K % 1024 for the tensor-core path; dp4a (fewer tokens) needs only 16-byte rows.
    const bool dp4a_rows = m < mma_min_m() && row_bytes % 16 == 0;
    if (row_bytes % 4 != 0 || (k % 1024 != 0 && !dp4a_rows) || reinterpret_cast<uintptr_t>(w) % 16 != 0 || mode < 0 ||
        mode > 1)
        return VELOX_ERR_UNSUPPORTED;
    const int subs = static_cast<int>(k / 32);
    switch (type) {
        case kQ2K: return launch_any<kQ2K>(w, row_bytes, row_begin, rows, subs, xq, xd, m, y, ldy, mode);
        case kQ4K: return launch_any<kQ4K>(w, row_bytes, row_begin, rows, subs, xq, xd, m, y, ldy, mode);
        case kIq1M: return launch_any<kIq1M>(w, row_bytes, row_begin, rows, subs, xq, xd, m, y, ldy, mode);
        case kIq2Xxs: return launch_any<kIq2Xxs>(w, row_bytes, row_begin, rows, subs, xq, xd, m, y, ldy, mode);
        case kIq2Xs: return launch_any<kIq2Xs>(w, row_bytes, row_begin, rows, subs, xq, xd, m, y, ldy, mode);
        case kIq2S: return launch_any<kIq2S>(w, row_bytes, row_begin, rows, subs, xq, xd, m, y, ldy, mode);
        case kIq3Xxs: return launch_any<kIq3Xxs>(w, row_bytes, row_begin, rows, subs, xq, xd, m, y, ldy, mode);
        case kIq3S: return launch_any<kIq3S>(w, row_bytes, row_begin, rows, subs, xq, xd, m, y, ldy, mode);
        case kIq1S: return launch_any<kIq1S>(w, row_bytes, row_begin, rows, subs, xq, xd, m, y, ldy, mode);
        default: return launch_any<kIq4Xs>(w, row_bytes, row_begin, rows, subs, xq, xd, m, y, ldy, mode);
    }
}

extern "C" int velox_5060_q8_view(const void* blob, int m, long long k, int layout, const int8_t** xq,
                                  const float** sx, int* stride);

// y[t * ldy + r] = W[row_begin + r, :] . x[t, :] on int8 tensor cores for a GGML weight of `type`,
// from int8 activations (ActQuant::Int8G256, layout 0): the last velox_5060_q8_quantize when `blob`
// is null, else a caller blob (an int8 twin). mode 1 writes silu(y) * that. k % 1024 == 0.
VELOX_API int velox_5060_linear_ggml_q8(const void* blob, int m, const uint8_t* w, int type, long long k,
                                        long long row_begin, int rows, float* y, long long ldy, int mode) {
    // Rows of a 4-byte multiple: blocks are copied from the aligned address at or below them.
    if (m < 1 || rows <= 0 || k % 256 != 0 || k > (1ll << 30) || type < 0 || (type > 8 && type != kIq1S) || mode < 0 || mode > 1 ||
        (k % 1024 != 0 && (k / 256 * block_bytes(type)) % 4 != 0) || reinterpret_cast<uintptr_t>(w) % 16 != 0)
        return VELOX_ERR_UNSUPPORTED;
    const int8_t* xq;
    const float* sx;
    int stride;
    if (const int e = velox_5060_q8_view(blob, m, k, 0, &xq, &sx, &stride)) return e;
    const long long row_bytes = k / 256 * block_bytes(type);
    const int ki = static_cast<int>(k);
    switch (type) {
        case kQ2K: return launch_pre2<kQ2K>(xq, sx, stride, m, w, row_bytes, row_begin, rows, ki, y, ldy, mode);
        case kQ4K: return launch_pre2<kQ4K>(xq, sx, stride, m, w, row_bytes, row_begin, rows, ki, y, ldy, mode);
        case kIq1M: return launch_pre2<kIq1M>(xq, sx, stride, m, w, row_bytes, row_begin, rows, ki, y, ldy, mode);
        case kIq2Xxs: return launch_pre2<kIq2Xxs>(xq, sx, stride, m, w, row_bytes, row_begin, rows, ki, y, ldy, mode);
        case kIq2Xs: return launch_pre2<kIq2Xs>(xq, sx, stride, m, w, row_bytes, row_begin, rows, ki, y, ldy, mode);
        case kIq2S: return launch_pre2<kIq2S>(xq, sx, stride, m, w, row_bytes, row_begin, rows, ki, y, ldy, mode);
        case kIq3Xxs: return launch_pre2<kIq3Xxs>(xq, sx, stride, m, w, row_bytes, row_begin, rows, ki, y, ldy, mode);
        case kIq3S: return launch_pre2<kIq3S>(xq, sx, stride, m, w, row_bytes, row_begin, rows, ki, y, ldy, mode);
        case kIq1S: return launch_pre2<kIq1S>(xq, sx, stride, m, w, row_bytes, row_begin, rows, ki, y, ldy, mode);
        default: return launch_pre2<kIq4Xs>(xq, sx, stride, m, w, row_bytes, row_begin, rows, ki, y, ldy, mode);
    }
}

// ---- Routed experts at decode and verify widths (Backend::moe_experts) ----------------------
//
// Int8 activations per 32 values (ActQuant::Int8G32) for both the input and the SwiGLU product.
// Gate and up run as one GEMV per (row, expert) pair, warp per R rows over the K sub-blocks; down
// runs warp per 8 output rows of one token row, four lanes per output row, over that row's experts
// in order, so the weighted sum is deterministic and a warp reads one contiguous run of each
// expert's down rows (records may sit in mapped system RAM, where every PCIe transaction counts).
namespace {

struct MoeDesc {
    const uint8_t* records;
    const uint8_t* slots;
    const float* map;  // i32 bits: slot of expert e, or -1
    long long record_bytes, slot_bytes;
    long long gate_off, up_off, down_off;
    int width, hidden;
};

__device__ __forceinline__ const uint8_t* moe_record(const MoeDesc& d, int e) {
    const int slot = static_cast<int>(__float_as_uint(__ldg(d.map + e)));
    return slot >= 0 ? d.slots + static_cast<long long>(slot) * d.slot_bytes
                     : d.records + static_cast<long long>(e) * d.record_bytes;
}

// Warp per (row, sub-block): int8 codes and scale per 32 values, ActQuant::Int8G32 bit for bit.
__global__ void moe_quant_kernel(const float* __restrict__ x, int n, int subs, int8_t* __restrict__ q,
                                 float* __restrict__ d) {
    const long long gi = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) / VELOX_WARP;
    const int lane = threadIdx.x % VELOX_WARP;
    if (gi >= static_cast<long long>(n) * subs) return;
    const float v = x[gi * 32 + lane];
    const float amax = velox_warp_max(fabsf(v));
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    q[gi * 32 + lane] = static_cast<int8_t>(__float2int_rn(v * inv));
    if (lane == 0) d[gi] = amax / 127.0f;
}

constexpr int kMoeWarps = 4;
constexpr int kMoeUpRows = 2;
constexpr int kMoeDownRows = 8;  // per warp; 4 lanes each

// Words of a type's lookup grid kept in shared memory (u64 entries as two words), 0 for none.
template <int T>
__host__ __device__ constexpr int moe_grid_words() {
    return T == kIq1M ? 4096 : Grid<T>::kWords > 1 ? Grid<T>::kWords : 0;
}

// Sub-block ib of the stored block at x (shared memory), grids and sign patterns from the shared
// copies; the same values as decode<T>.
template <int T>
__device__ __forceinline__ Sub<T> decode_smem(const uint8_t* x, int ib, const uint32_t* sg, const uint8_t* sks) {
    if constexpr (T == kIq2S || T == kIq2Xxs || T == kIq2Xs || T == kIq3S || T == kIq3Xxs) {
        Sub<T> r;
        const float d = f16_at(x);
        if constexpr (T == kIq2S || T == kIq2Xs) {
            const uint32_t s = x[(T == kIq2S ? 74 : 66) + ib];
            r.s[0] = d * (0.5f + static_cast<float>(s & 0xf)) * 0.25f;
            r.s[1] = d * (0.5f + static_cast<float>(s >> 4)) * 0.25f;
        } else if constexpr (T == kIq2Xxs) {
            r.s[0] = d * (0.5f + static_cast<float>(u32_at2(x + 6 + 8 * ib) >> 28)) * 0.25f;
        } else if constexpr (T == kIq3Xxs) {
            r.s[0] = d * (0.5f + static_cast<float>(u32_at2(x + 66 + 4 * ib) >> 28)) * 0.5f;
        } else {
            const uint32_t s = x[106 + ib / 2];
            r.s[0] = d * static_cast<float>(1 + 2 * (ib % 2 ? s >> 4 : s & 0xf));
        }
#pragma unroll
        for (int wi = 0; wi < 8; ++wi) r.w[wi] = word<T>(x, ib, wi, sg, sks);
        return r;
    } else if constexpr (T == kIq1M) {
        Sub<T> r;
        const uint32_t s0 = u16_at(x + 48), s1 = u16_at(x + 50), s2 = u16_at(x + 52), s3 = u16_at(x + 54);
        const uint32_t dbits = (s0 >> 12) | ((s1 >> 8) & 0x00f0) | ((s2 >> 4) & 0x0f00) | (s3 & 0xf000);
        const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(dbits)));
        const uint32_t w = u16_at(x + 48 + 2 * (ib / 2));
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const float dl = d * static_cast<float>(2 * ((w >> (6 * (ib % 2) + (l < 2 ? 0 : 3))) & 7) + 1);
            const uint32_t qh = x[32 + 2 * ib + l / 2];
            const uint32_t idx = x[4 * ib + l] | ((qh << (l % 2 ? 4 : 8)) & 0x700);
            const float delta = qh & (l % 2 ? 0x80 : 0x08) ? -0.125f : 0.125f;
            const uint64_t g = grid64(sg, idx);
            r.w[2 * l] = static_cast<int>(static_cast<uint32_t>(g));
            r.w[2 * l + 1] = static_cast<int>(static_cast<uint32_t>(g >> 32));
            r.s[l] = dl;
            r.o[l] = dl * delta;
        }
        return r;
    } else {
        return decode<T>(x, ib);
    }
}

// The grid of T and the sign patterns into shared memory (every thread of the block takes part).
template <int T>
__device__ __forceinline__ void moe_load_tables(uint32_t* sg, uint8_t* sks) {
    constexpr int W = moe_grid_words<T>();
    if constexpr (W > 0) {
        const uint32_t* src;
        if constexpr (T == kIq1M) src = reinterpret_cast<const uint32_t*>(kIq1SGrid);
        else src = Grid<T>::source();
        for (int i = threadIdx.x; i < W; i += blockDim.x) sg[i] = __ldg(src + i);
        for (int i = threadIdx.x; i < 128; i += blockDim.x) sks[i] = __ldg(kKsignsIq2xs + i);
    }
}

// A byte span widened to 16-byte alignment, read by one warp as up to 32 N uint4 (lane l takes
// words l, l + 32, ...): every load is issued before any is used, so a warp keeps N 512-byte
// requests in flight (mapped system RAM is latency-bound otherwise). The widening stays inside
// the 16-byte aligned record holding the span.
template <int N>
struct Span {
    uint4 v[N];
    int lead;  // offset of the span start in the copy
    int n;     // uint4 words
};

template <int N>
__device__ __forceinline__ void span_load(const uint8_t* src, int bytes, int lane, Span<N>& s) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(src);
    const uint4* p = reinterpret_cast<const uint4*>(a & ~static_cast<uintptr_t>(15));
    s.lead = static_cast<int>(a & 15);
    s.n = (s.lead + bytes + 15) / 16;
#pragma unroll
    for (int i = 0; i < N; ++i) {
        const int w = lane + VELOX_WARP * i;
        if (w < s.n) s.v[i] = p[w];
    }
}

template <int N>
__device__ __forceinline__ void span_store(uint4* dst, int lane, const Span<N>& s) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
        const int w = lane + VELOX_WARP * i;
        if (w < s.n) dst[w] = s.v[i];
    }
}

// Most uint4 per lane a staged span takes, and the bytes that bounds a span to.
// IQ4_NL (4.5 bits per value) takes wider spans: two gate rows of K = 2560 are 2880 bytes, as are
// eight down rows of K = 640 (Qwen3.8-Flash-Next's MTP experts).
constexpr int kSpanUp = 4, kSpanDown = 3;
template <int T>
__host__ __device__ constexpr int span_up() {
    return T == kIq4Nl ? 6 : kSpanUp;
}
template <int T>
__host__ __device__ constexpr int span_down() {
    return T == kIq4Nl ? 6 : kSpanDown;
}
__host__ __device__ constexpr int span_cap(int n) { return n * VELOX_WARP * 16 - 16; }

// h[p][i] = silu(gate_i . x) * (up_i . x) for pair p = row * k + j (expert ids[p]): warp per R
// consecutive rows, whose gate rows and up rows (one span each) are staged into shared memory.
template <int T>
__global__ void __launch_bounds__(kMoeWarps* VELOX_WARP)
    moe_gate_up_kernel(MoeDesc d, const float* __restrict__ ids, int k, const int8_t* __restrict__ xq,
                       const float* __restrict__ xd, float* __restrict__ h) {
    constexpr int R = kMoeUpRows, SPB = sub_per_block<T>(), BB = block_bytes(T), N = span_up<T>();
    constexpr int GW = moe_grid_words<T>() > 0 ? moe_grid_words<T>() : 1;
    __shared__ uint32_t sg[GW];
    __shared__ uint8_t sks[128];
    __shared__ uint4 stage[kMoeWarps][2][N * VELOX_WARP];
    moe_load_tables<T>(sg, sks);
    __syncthreads();
    const int p = blockIdx.y, lane = threadIdx.x % VELOX_WARP, warp = threadIdx.x / VELOX_WARP;
    const int r0 = (blockIdx.x * kMoeWarps + warp) * R;
    if (r0 >= d.width) return;
    const int rows = min(R, d.width - r0);
    const int tok = p / k, subs = d.hidden / 32;
    const uint8_t* rec = moe_record(d, static_cast<int>(__float_as_uint(ids[p])));
    const int row_bytes = subs / SPB * BB;
    const long long first = static_cast<long long>(r0) * row_bytes;
    Span<N> sgate, sup;
    span_load(rec + d.gate_off + first, rows * row_bytes, lane, sgate);
    span_load(rec + d.up_off + first, rows * row_bytes, lane, sup);
    span_store(stage[warp][0], lane, sgate);
    span_store(stage[warp][1], lane, sup);
    __syncwarp();
    const uint8_t* gm = reinterpret_cast<const uint8_t*>(stage[warp][0]) + sgate.lead;
    const uint8_t* um = reinterpret_cast<const uint8_t*>(stage[warp][1]) + sup.lead;
    float ag[R] = {}, au[R] = {};
    for (int u = lane; u < subs; u += VELOX_WARP) {
        const int4* ap = reinterpret_cast<const int4*>(xq + (static_cast<long long>(tok) * subs + u) * 32);
        const int4 a0 = __ldg(ap), a1 = __ldg(ap + 1);
        const int a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
        const float dx = __ldg(xd + static_cast<long long>(tok) * subs + u);
        const int off = u / SPB * BB;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int at = min(r, rows - 1) * row_bytes + off;
            ag[r] += dx * sub_dot<T>(decode_smem<T>(gm + at, u % SPB, sg, sks), a);
            au[r] += dx * sub_dot<T>(decode_smem<T>(um + at, u % SPB, sg, sks), a);
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float g = velox_warp_sum(ag[r]), v = velox_warp_sum(au[r]);
        if (lane == 0 && r < rows) h[static_cast<long long>(p) * d.width + r0 + r] = g / (1.0f + __expf(-g)) * v;
    }
}

// y[tok][c] = sum_j w[p] * (down_c . h[p]) over the token's pairs p = tok * k + j in order: warp
// per 8 consecutive output rows (one span of each expert's down rows), four lanes per row. The
// next expert's span is loaded into registers while the current one is used.
template <int T>
__global__ void __launch_bounds__(kMoeWarps* VELOX_WARP)
    moe_down_kernel(MoeDesc d, const float* __restrict__ ids, const float* __restrict__ wts, int k,
                    const int8_t* __restrict__ hq, const float* __restrict__ hd, float* __restrict__ y) {
    constexpr int SPB = sub_per_block<T>(), BB = block_bytes(T), L = VELOX_WARP / kMoeDownRows, N = span_down<T>();
    constexpr int GW = moe_grid_words<T>() > 0 ? moe_grid_words<T>() : 1;
    __shared__ uint32_t sg[GW];
    __shared__ uint8_t sks[128];
    __shared__ uint4 stage[kMoeWarps][N * VELOX_WARP];
    moe_load_tables<T>(sg, sks);
    __syncthreads();
    const int tok = blockIdx.y, lane = threadIdx.x % VELOX_WARP, warp = threadIdx.x / VELOX_WARP;
    const int c0 = (blockIdx.x * kMoeWarps + warp) * kMoeDownRows;
    if (c0 >= d.hidden) return;
    const int rows = min(kMoeDownRows, d.hidden - c0);
    const int c = c0 + lane / L, sl = lane % L, subs = d.width / 32;
    const int row_bytes = subs / SPB * BB;
    const int mine = min(lane / L, rows - 1) * row_bytes;
    auto span_of = [&](int j) {
        return moe_record(d, static_cast<int>(__float_as_uint(ids[tok * k + j]))) + d.down_off +
               static_cast<long long>(c0) * row_bytes;
    };
    Span<N> next;
    span_load(span_of(0), rows * row_bytes, lane, next);
    float acc = 0.0f;
    for (int j = 0; j < k; ++j) {
        const int p = tok * k + j;
        __syncwarp();
        span_store(stage[warp], lane, next);
        const int lead = next.lead;
        __syncwarp();
        if (j + 1 < k) span_load(span_of(j + 1), rows * row_bytes, lane, next);
        const uint8_t* sm = reinterpret_cast<const uint8_t*>(stage[warp]) + lead + mine;
        float part = 0.0f;
        for (int u = sl; u < subs; u += L) {
            const int4* ap = reinterpret_cast<const int4*>(hq + (static_cast<long long>(p) * subs + u) * 32);
            const int4 a0 = __ldg(ap), a1 = __ldg(ap + 1);
            const int a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            part += __ldg(hd + static_cast<long long>(p) * subs + u) *
                    sub_dot<T>(decode_smem<T>(sm + u / SPB * BB, u % SPB, sg, sks), a);
        }
        acc += __ldg(wts + p) * part;
    }
#pragma unroll
    for (int o = L / 2; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (sl == 0 && c < d.hidden) y[static_cast<long long>(tok) * d.hidden + c] = acc;
}

template <int T>
int moe_up_launch(const MoeDesc& d, const float* ids, int pairs, int k, const int8_t* xq, const float* xd, float* h) {
    const int per_block = kMoeWarps * kMoeUpRows;
    const int row_bytes = d.hidden / 32 / sub_per_block<T>() * block_bytes(T);
    if (kMoeUpRows * row_bytes > span_cap(span_up<T>())) return VELOX_ERR_UNSUPPORTED;
    const dim3 grid((d.width + per_block - 1) / per_block, pairs);
    moe_gate_up_kernel<T><<<grid, kMoeWarps * VELOX_WARP>>>(d, ids, k, xq, xd, h);
    return velox_launch_status();
}

template <int T>
int moe_down_launch(const MoeDesc& d, const float* ids, const float* wts, int rows, int k, const int8_t* hq,
                    const float* hd, float* y) {
    const int per_block = kMoeWarps * kMoeDownRows;
    const int row_bytes = d.width / 32 / sub_per_block<T>() * block_bytes(T);
    if (kMoeDownRows * row_bytes > span_cap(span_down<T>())) return VELOX_ERR_UNSUPPORTED;
    const dim3 grid((d.hidden + per_block - 1) / per_block, rows);
    moe_down_kernel<T><<<grid, kMoeWarps * VELOX_WARP>>>(d, ids, wts, k, hq, hd, y);
    return velox_launch_status();
}

#define MOE_TYPES(F, ...)                             \
    switch (type) {                                   \
        case kQ2K: return F<kQ2K>(__VA_ARGS__);       \
        case kQ4K: return F<kQ4K>(__VA_ARGS__);       \
        case kIq1M: return F<kIq1M>(__VA_ARGS__);     \
        case kIq2Xxs: return F<kIq2Xxs>(__VA_ARGS__); \
        case kIq2Xs: return F<kIq2Xs>(__VA_ARGS__);   \
        case kIq2S: return F<kIq2S>(__VA_ARGS__);     \
        case kIq3Xxs: return F<kIq3Xxs>(__VA_ARGS__); \
        case kIq3S: return F<kIq3S>(__VA_ARGS__);     \
        case kIq4Xs: return F<kIq4Xs>(__VA_ARGS__);   \
        case kQ2_0: return F<kQ2_0>(__VA_ARGS__);             case kIq4Nl: return F<kIq4Nl>(__VA_ARGS__);   \
        default: return VELOX_ERR_UNSUPPORTED;        \
    }

int moe_up_any(int type, const MoeDesc& d, const float* ids, int pairs, int k, const int8_t* xq, const float* xd,
               float* h) {
    MOE_TYPES(moe_up_launch, d, ids, pairs, k, xq, xd, h)
}

int moe_down_any(int type, const MoeDesc& d, const float* ids, const float* wts, int rows, int k, const int8_t* hq,
                 const float* hd, float* y) {
    MOE_TYPES(moe_down_launch, d, ids, wts, rows, k, hq, hd, y)
}

#undef MOE_TYPES

// Region sizes of the scratch, each rounded to 64 bytes (int4 loads need 16-byte alignment).
long long moe_round(long long bytes) { return (bytes + 63) / 64 * 64; }

}  // namespace

// Bytes of scratch velox_5060_moe_experts needs for `rows` rows of `k` experts.
VELOX_API long long velox_5060_moe_scratch(int rows, int k, int width, int hidden) {
    const long long xs = static_cast<long long>(rows) * hidden, hw = static_cast<long long>(rows) * k * width;
    return moe_round(xs) + moe_round(4 * (xs / 32)) + moe_round(4 * hw) + moe_round(hw) + moe_round(4 * (hw / 32));
}

// Types are GgmlType::ALL positions; gate and up share one. VELOX_ERR_UNSUPPORTED for geometry
// the kernels do not cover (the caller falls back to the portable kernels).
VELOX_API int velox_5060_moe_experts(const float* x, const float* ids, const float* wts, const uint8_t* records,
                                     const uint8_t* slots, const float* map, long long record_bytes,
                                     long long slot_bytes, int up_type, long long gate_off, long long up_off,
                                     int down_type, long long down_off, int width, int hidden, int rows, int k,
                                     float* y, void* scratch) {
    if (rows < 1 || k < 1 || up_type < 0 || up_type > kIq4Nl || down_type < 0 || down_type > kIq4Nl ||
        hidden % block_values(up_type) != 0 || width % block_values(down_type) != 0 ||
        reinterpret_cast<uintptr_t>(scratch) % 16 != 0 || (gate_off | up_off | down_off | record_bytes | slot_bytes) % 4 != 0)
        return VELOX_ERR_UNSUPPORTED;
    const MoeDesc d{records, slots, map, record_bytes, slot_bytes, gate_off, up_off, down_off, width, hidden};
    const long long pairs = static_cast<long long>(rows) * k;
    const long long xs = static_cast<long long>(rows) * hidden, hw = pairs * width;
    uint8_t* p = static_cast<uint8_t*>(scratch);
    int8_t* xq = reinterpret_cast<int8_t*>(p);
    float* xd = reinterpret_cast<float*>(p += moe_round(xs));
    float* h = reinterpret_cast<float*>(p += moe_round(4 * (xs / 32)));
    int8_t* hq = reinterpret_cast<int8_t*>(p += moe_round(4 * hw));
    float* hd = reinterpret_cast<float*>(p += moe_round(hw));
    const int threads = 256;
    moe_quant_kernel<<<static_cast<unsigned>((xs / 32 * VELOX_WARP + threads - 1) / threads), threads>>>(
        x, rows, hidden / 32, xq, xd);
    if (const int e = moe_up_any(up_type, d, ids, static_cast<int>(pairs), k, xq, xd, h)) return e;
    moe_quant_kernel<<<static_cast<unsigned>((hw / 32 * VELOX_WARP + threads - 1) / threads), threads>>>(
        h, static_cast<int>(pairs), width / 32, hq, hd);
    return moe_down_any(down_type, d, ids, wts, rows, k, hq, hd, y);
}

// ---- Routed experts at prefill widths: grouped by expert ------------------------------------
//
// Pairs are sorted by expert (counting sort). A block owns one expert and a tile of its weight
// rows: it stages the rows once (device slot or mapped system RAM, 16-byte loads), decodes them
// into int8 codes and per-32 scales in shared memory, then runs every pair routed to the expert
// against that tile with dp4a. Each expert's weights are read once per call, so the cost is
// the experts touched, not the pairs. Down writes one output row per pair; the weighted sum over
// a row's experts runs in order afterwards (deterministic).
namespace {

__global__ void moe_count_kernel(const float* __restrict__ ids, int pairs, int experts, int* __restrict__ counts) {
    for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < pairs; p += gridDim.x * blockDim.x) {
        const unsigned e = __float_as_uint(ids[p]);
        if (e < static_cast<unsigned>(experts)) atomicAdd(counts + e, 1);
    }
}

// One block: exclusive scan of the counts (cursors reset).
__global__ void moe_scan_kernel(const int* __restrict__ counts, int* __restrict__ offsets, int* __restrict__ cursor,
                                int experts) {
    __shared__ int part[256];
    const int t = threadIdx.x, per = (experts + 255) / 256;
    int s = 0;
    for (int i = 0; i < per; ++i) {
        const int e = t * per + i;
        if (e < experts) s += counts[e];
    }
    part[t] = s;
    __syncthreads();
    if (t == 0) {
        int acc = 0;
        for (int i = 0; i < 256; ++i) {
            const int v = part[i];
            part[i] = acc;
            acc += v;
        }
        offsets[experts] = acc;
    }
    __syncthreads();
    int acc = part[t];
    for (int i = 0; i < per; ++i) {
        const int e = t * per + i;
        if (e < experts) {
            offsets[e] = acc;
            cursor[e] = 0;
            acc += counts[e];
        }
    }
}

__global__ void moe_fill_kernel(const float* __restrict__ ids, int pairs, int experts, const int* __restrict__ offsets,
                                int* __restrict__ cursor, int* __restrict__ list) {
    for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < pairs; p += gridDim.x * blockDim.x) {
        const unsigned e = __float_as_uint(ids[p]);
        if (e < static_cast<unsigned>(experts)) list[offsets[e] + atomicAdd(cursor + e, 1)] = p;
    }
}

// A decoded weight tile in shared memory: RT rows x S sub-blocks of 32 int8 codes (rows padded by
// 16 bytes against bank conflicts), G scales and (offset types) G offsets per sub-block.
template <int T>
struct Tile {
    static constexpr int G = Traits<T>::G;
    static constexpr bool kOff = Traits<T>::kOff;
    __host__ __device__ static int row_stride(int subs) { return subs * 32 + 16; }
    __host__ __device__ static size_t bytes(int rt, int subs, int raw) {
        const size_t codes = static_cast<size_t>(rt) * row_stride(subs);
        const size_t sc = static_cast<size_t>(rt) * subs * G * 4 * (kOff ? 2 : 1);
        return codes + sc + (raw + 31) / 16 * 16;
    }
};

// Stages rows [r0, r0 + rt) of a matrix at `w` (rows of row_bytes) into `raw` (16-byte loads over
// the widened span) and decodes them into the tile; returns after a block barrier.
template <int T>
__device__ __forceinline__ void tile_build(const uint8_t* __restrict__ w, int row_bytes, int r0, int rt, int subs,
                                           uint8_t* codes, float* sc, float* of, uint8_t* raw, const uint32_t* sg,
                                           const uint8_t* sks) {
    constexpr int SPB = sub_per_block<T>(), BB = block_bytes(T), G = Tile<T>::G;
    const uint8_t* src = w + static_cast<long long>(r0) * row_bytes;
    const uintptr_t a = reinterpret_cast<uintptr_t>(src);
    const uint4* p = reinterpret_cast<const uint4*>(a & ~static_cast<uintptr_t>(15));
    const int lead = static_cast<int>(a & 15), n = (lead + rt * row_bytes + 15) / 16;
    uint4* r = reinterpret_cast<uint4*>(raw);
    for (int i = threadIdx.x; i < n; i += 4 * blockDim.x) {
        uint4 v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (i + j * blockDim.x < n) v[j] = p[i + j * blockDim.x];
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (i + j * blockDim.x < n) r[i + j * blockDim.x] = v[j];
    }
    __syncthreads();
    const int stride = Tile<T>::row_stride(subs);
    for (int idx = threadIdx.x; idx < rt * subs; idx += blockDim.x) {
        const int row = idx / subs, u = idx % subs;
        const Sub<T> s = decode_smem<T>(raw + lead + row * row_bytes + u / SPB * BB, u % SPB, sg, sks);
        int4* c = reinterpret_cast<int4*>(codes + row * stride + u * 32);
        c[0] = make_int4(s.w[0], s.w[1], s.w[2], s.w[3]);
        c[1] = make_int4(s.w[4], s.w[5], s.w[6], s.w[7]);
#pragma unroll
        for (int g = 0; g < G; ++g) {
            sc[idx * G + g] = s.s[g];
            if constexpr (Tile<T>::kOff) of[idx * G + g] = s.o[g];
        }
    }
    __syncthreads();
}

// sub_dot of a decoded tile entry.
template <int T>
__device__ __forceinline__ float tile_dot(const uint8_t* codes, const float* sc, const float* of, int stride, int subs,
                                          int row, int u, const int (&a)[8]) {
    constexpr int G = Tile<T>::G;
    Sub<T> s;
    const int4* c = reinterpret_cast<const int4*>(codes + row * stride + u * 32);
    const int4 c0 = c[0], c1 = c[1];
    s.w[0] = c0.x, s.w[1] = c0.y, s.w[2] = c0.z, s.w[3] = c0.w;
    s.w[4] = c1.x, s.w[5] = c1.y, s.w[6] = c1.z, s.w[7] = c1.w;
#pragma unroll
    for (int g = 0; g < G; ++g) {
        s.s[g] = sc[(row * subs + u) * G + g];
        if constexpr (Tile<T>::kOff) s.o[g] = of[(row * subs + u) * G + g];
    }
    return sub_dot<T>(s, a);
}

template <int T>
__host__ __device__ constexpr int up_tile_rows() {
    return T == kIq1M ? 8 : 16;
}

// Gate (mode 0: h[p][i] = gate_i . x) or up (mode 1: h[p][i] = silu(h[p][i]) * (up_i . x)) for
// every pair of expert blockIdx.y, rows [RT blockIdx.x, + RT). A warp per pair, lanes over K.
template <int T>
__global__ void __launch_bounds__(128)
    moe_grouped_up_kernel(MoeDesc d, long long mat_off, const int* __restrict__ offsets, const int* __restrict__ list,
                          int k, const int8_t* __restrict__ xq, const float* __restrict__ xd, float* __restrict__ h,
                          int mode) {
    constexpr int RT = up_tile_rows<T>(), G = Tile<T>::G;
    constexpr int GW = moe_grid_words<T>() > 0 ? moe_grid_words<T>() : 1;
    __shared__ uint32_t sg[GW];
    __shared__ uint8_t sks[128];
    extern __shared__ __align__(16) uint8_t smem[];
    const int e = blockIdx.y, begin = offsets[e], count = offsets[e + 1] - begin;
    if (count == 0) return;
    moe_load_tables<T>(sg, sks);
    const int subs = d.hidden / 32, r0 = blockIdx.x * RT, rt = min(RT, d.width - r0);
    const int row_bytes = subs / sub_per_block<T>() * block_bytes(T), stride = Tile<T>::row_stride(subs);
    uint8_t* codes = smem;
    float* sc = reinterpret_cast<float*>(codes + RT * stride);
    float* of = sc + RT * subs * G;
    uint8_t* raw = reinterpret_cast<uint8_t*>(of + (Tile<T>::kOff ? RT * subs * G : 0));
    __syncthreads();
    tile_build<T>(moe_record(d, e) + mat_off, row_bytes, r0, rt, subs, codes, sc, of, raw, sg, sks);
    const int lane = threadIdx.x % VELOX_WARP;
    for (int i = threadIdx.x / VELOX_WARP; i < count; i += blockDim.x / VELOX_WARP) {
        const int p = list[begin + i], tok = p / k;
        float acc[RT];
#pragma unroll
        for (int r = 0; r < RT; ++r) acc[r] = 0.0f;
        for (int u = lane; u < subs; u += VELOX_WARP) {
            const int4* ap = reinterpret_cast<const int4*>(xq + (static_cast<long long>(tok) * subs + u) * 32);
            const int4 a0 = __ldg(ap), a1 = __ldg(ap + 1);
            const int a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float dx = __ldg(xd + static_cast<long long>(tok) * subs + u);
#pragma unroll
            for (int r = 0; r < RT; ++r) acc[r] += dx * tile_dot<T>(codes, sc, of, stride, subs, min(r, rt - 1), u, a);
        }
#pragma unroll
        for (int r = 0; r < RT; ++r) {
            const float v = velox_warp_sum(acc[r]);
            if (lane == r && r < rt) {
                float* o = h + static_cast<long long>(p) * d.width + r0 + r;
                if (mode) {
                    const float g = *o;
                    *o = g / (1.0f + __expf(-g)) * v;
                } else {
                    *o = v;
                }
            }
        }
    }
}

constexpr int kDownTileRows = 64;

// o[p][c] = down_c . h[p] for every pair of expert blockIdx.y, rows [64 blockIdx.x, + 64): a warp
// per pair, two output rows per lane.
template <int T>
__global__ void __launch_bounds__(128)
    moe_grouped_down_kernel(MoeDesc d, const int* __restrict__ offsets, const int* __restrict__ list,
                            const int8_t* __restrict__ hq, const float* __restrict__ hd, float* __restrict__ o) {
    constexpr int RT = kDownTileRows;
    constexpr int GW = moe_grid_words<T>() > 0 ? moe_grid_words<T>() : 1;
    __shared__ uint32_t sg[GW];
    __shared__ uint8_t sks[128];
    extern __shared__ __align__(16) uint8_t smem[];
    const int e = blockIdx.y, begin = offsets[e], count = offsets[e + 1] - begin;
    if (count == 0) return;
    moe_load_tables<T>(sg, sks);
    constexpr int G = Tile<T>::G;
    const int subs = d.width / 32, r0 = blockIdx.x * RT, rt = min(RT, d.hidden - r0);
    const int row_bytes = subs / sub_per_block<T>() * block_bytes(T), stride = Tile<T>::row_stride(subs);
    uint8_t* codes = smem;
    float* sc = reinterpret_cast<float*>(codes + RT * stride);
    float* of = sc + RT * subs * G;
    uint8_t* raw = reinterpret_cast<uint8_t*>(of + (Tile<T>::kOff ? RT * subs * G : 0));
    __syncthreads();
    tile_build<T>(moe_record(d, e) + d.down_off, row_bytes, r0, rt, subs, codes, sc, of, raw, sg, sks);
    const int lane = threadIdx.x % VELOX_WARP;
    for (int i = threadIdx.x / VELOX_WARP; i < count; i += blockDim.x / VELOX_WARP) {
        const int p = list[begin + i];
        float acc[2] = {0.0f, 0.0f};
        for (int u = 0; u < subs; ++u) {
            const int4* ap = reinterpret_cast<const int4*>(hq + (static_cast<long long>(p) * subs + u) * 32);
            const int4 a0 = __ldg(ap), a1 = __ldg(ap + 1);
            const int a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float dh = __ldg(hd + static_cast<long long>(p) * subs + u);
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int r = lane + q * VELOX_WARP;
                acc[q] += dh * tile_dot<T>(codes, sc, of, stride, subs, min(r, rt - 1), u, a);
            }
        }
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int r = lane + q * VELOX_WARP;
            if (r < rt) o[static_cast<long long>(p) * d.hidden + r0 + r] = acc[q];
        }
    }
}

__global__ void moe_sum_kernel(const float* __restrict__ o, const float* __restrict__ w, int rows, int k, int hidden,
                               float* __restrict__ y) {
    const long long n = static_cast<long long>(rows) * hidden;
    for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const long long r = i / hidden, c = i % hidden;
        float acc = 0.0f;
        for (int j = 0; j < k; ++j) acc += w[r * k + j] * o[(r * k + j) * hidden + c];
        y[i] = acc;
    }
}

template <int T>
int grouped_up_launch(const MoeDesc& d, long long mat_off, int experts, const int* offsets, const int* list, int k,
                      const int8_t* xq, const float* xd, float* h, int mode) {
    constexpr int RT = up_tile_rows<T>();
    const int subs = d.hidden / 32, row_bytes = subs / sub_per_block<T>() * block_bytes(T);
    const size_t smem = Tile<T>::bytes(RT, subs, RT * row_bytes);
    if (smem + 4 * moe_grid_words<T>() + 128 > 99 * 1024) return VELOX_ERR_UNSUPPORTED;
    static const cudaError_t attr = cudaFuncSetAttribute(moe_grouped_up_kernel<T>,
                                                         cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024 - 4 * moe_grid_words<T>() - 128);
    if (attr != cudaSuccess) return static_cast<int>(attr);
    const dim3 grid((d.width + RT - 1) / RT, experts);
    moe_grouped_up_kernel<T><<<grid, 128, smem>>>(d, mat_off, offsets, list, k, xq, xd, h, mode);
    return velox_launch_status();
}

template <int T>
int grouped_down_launch(const MoeDesc& d, int experts, const int* offsets, const int* list, const int8_t* hq,
                        const float* hd, float* o) {
    constexpr int RT = kDownTileRows;
    const int subs = d.width / 32, row_bytes = subs / sub_per_block<T>() * block_bytes(T);
    const size_t smem = Tile<T>::bytes(RT, subs, RT * row_bytes);
    if (smem + 4 * moe_grid_words<T>() + 128 > 99 * 1024) return VELOX_ERR_UNSUPPORTED;
    static const cudaError_t attr = cudaFuncSetAttribute(moe_grouped_down_kernel<T>,
                                                         cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024 - 4 * moe_grid_words<T>() - 128);
    if (attr != cudaSuccess) return static_cast<int>(attr);
    const dim3 grid((d.hidden + RT - 1) / RT, experts);
    moe_grouped_down_kernel<T><<<grid, 128, smem>>>(d, offsets, list, hq, hd, o);
    return velox_launch_status();
}

#define MOE_GROUPED(F, ...)                           \
    switch (type) {                                   \
        case kQ2K: return F<kQ2K>(__VA_ARGS__);       \
        case kQ4K: return F<kQ4K>(__VA_ARGS__);       \
        case kIq1M: return F<kIq1M>(__VA_ARGS__);     \
        case kIq2Xxs: return F<kIq2Xxs>(__VA_ARGS__); \
        case kIq2Xs: return F<kIq2Xs>(__VA_ARGS__);   \
        case kIq2S: return F<kIq2S>(__VA_ARGS__);     \
        case kIq3Xxs: return F<kIq3Xxs>(__VA_ARGS__); \
        case kIq3S: return F<kIq3S>(__VA_ARGS__);     \
        case kIq4Xs: return F<kIq4Xs>(__VA_ARGS__);   \
        case kQ2_0: return F<kQ2_0>(__VA_ARGS__);             case kIq4Nl: return F<kIq4Nl>(__VA_ARGS__);   \
        default: return VELOX_ERR_UNSUPPORTED;        \
    }

int grouped_up_any(int type, const MoeDesc& d, long long mat_off, int experts, const int* offsets, const int* list,
                   int k, const int8_t* xq, const float* xd, float* h, int mode) {
    MOE_GROUPED(grouped_up_launch, d, mat_off, experts, offsets, list, k, xq, xd, h, mode)
}

int grouped_down_any(int type, const MoeDesc& d, int experts, const int* offsets, const int* list, const int8_t* hq,
                     const float* hd, float* o) {
    MOE_GROUPED(grouped_down_launch, d, experts, offsets, list, hq, hd, o)
}

#undef MOE_GROUPED

}  // namespace

// ---- Routed experts at prefill widths on int8 tensor cores ----------------------------------
//
// The grouped scheme above with mma.m16n8k32 (k16 per half for 16-value scales): a block owns one
// expert and 64 weight rows (4 warps x 16) and walks K in steps; each step stages the rows' bytes
// and the gathered activations of up to 64 of the expert's pairs, decodes the weights into int8
// codes with per-32 (or per-16) scales, and every warp multiplies its 16 rows by the 8-pair tiles.
// Products accumulate in int32 per sub-block and in f32 with both scales (plus the offset types'
// offset times the activation code sums).
namespace {

constexpr int kGmRows = 64, kGmTok = 32;
// Shared memory the block's 64 full weight rows may take (staged once, 16-byte loads).
constexpr int kGmRawMax = 56 * 1024;

template <int T>
struct Gm {
    static constexpr int G = Traits<T>::G;
    static constexpr bool kOff = Traits<T>::kOff;
    static constexpr int kStep = T == kQ2_0 || T == kIq4Nl ? 128 : 256;  // values per K step
    static constexpr int kSu = kStep / 32;               // sub-blocks per step
    static constexpr int kStride = kStep + 16;           // code / activation row stride (bytes)
    struct Smem {
        int8_t codes[kGmRows * kStride];
        int8_t act[kGmTok * kStride];
        float ws[kGmRows][kSu][G];
        float wo[kOff ? kGmRows : 1][kSu][G];
        float xd[kGmTok][kSu];
        int xsum[kOff ? kGmTok : 1][kSu][G];
        // followed by the staged rows
    };
};

// mode 0: out[p][r] = W_r . x[p / k]; mode 1: out[p][r] = silu(out[p][r]) * (W_r . x[p / k]);
// mode 2: out[p][r] = W_r . a[p]. W: the expert's matrix at mat_off, `rows` x K; activations aq
// (int8 per 32 values) with scales ad.
template <int T>
__global__ void __launch_bounds__(128)
    moe_mma_kernel(MoeDesc d, long long mat_off, int rows, int K, const int* __restrict__ offsets,
                   const int* __restrict__ list, int k, const int8_t* __restrict__ aq, const float* __restrict__ ad,
                   int mode, float* __restrict__ out) {
    using C = Gm<T>;
    constexpr int G = C::G, SU = C::kSu, ST = C::kStride, SPB = sub_per_block<T>(), BB = block_bytes(T);
    constexpr int GW = moe_grid_words<T>() > 0 ? moe_grid_words<T>() : 1;
    __shared__ uint32_t sg[GW];
    __shared__ uint8_t sks[128];
    extern __shared__ __align__(16) uint8_t smem[];
    auto& S = *reinterpret_cast<typename C::Smem*>(smem);
    const int e = blockIdx.y, begin = offsets[e], count = offsets[e + 1] - begin;
    if (count == 0) return;
    moe_load_tables<T>(sg, sks);
    const int warp = threadIdx.x / VELOX_WARP, lane = threadIdx.x % VELOX_WARP, g = lane / 4, t = lane % 4;
    const int r0 = blockIdx.x * kGmRows;
    const uint8_t* wbase = moe_record(d, e) + mat_off;
    const long long row_bytes = static_cast<long long>(K / (SPB * 32)) * BB;
    const int subs = K / 32, steps = K / C::kStep;
    // The block's rows (contiguous in the record) staged once: one span widened to 16 bytes.
    const int live_rows = min(kGmRows, rows - r0);
    uint8_t* raw = smem + sizeof(typename C::Smem);
    {
        const uintptr_t a = reinterpret_cast<uintptr_t>(wbase + r0 * row_bytes);
        const uint4* src = reinterpret_cast<const uint4*>(a & ~static_cast<uintptr_t>(15));
        const int n = static_cast<int>(((a & 15) + live_rows * row_bytes + 15) / 16);
        for (int i = threadIdx.x; i < n; i += blockDim.x) cp_async16(raw + 16 * i, src + i);
        cp_async_commit();
        raw += a & 15;
    }
    cp_async_wait0();
    __syncthreads();
    for (int tg = 0; tg < count; tg += kGmTok) {
        const int nt = min(kGmTok, count - tg);
        float acc[kGmTok / 8][4];
#pragma unroll
        for (int n = 0; n < kGmTok / 8; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;
        for (int step = 0; step < steps; ++step) {
            __syncthreads();  // the previous step's mma is done with act, codes and scales
            for (int i = threadIdx.x; i < kGmTok * (C::kStep / 16); i += blockDim.x) {
                const int tk = i / (C::kStep / 16), c = i % (C::kStep / 16);
                int4 v = make_int4(0, 0, 0, 0);
                if (tk < nt) {
                    const int p = list[begin + tg + tk], row = mode < 2 ? p / k : p;
                    v = *reinterpret_cast<const int4*>(aq + static_cast<long long>(row) * K + step * C::kStep + c * 16);
                }
                *reinterpret_cast<int4*>(&S.act[tk * ST + c * 16]) = v;
            }
            for (int i = threadIdx.x; i < kGmTok * SU; i += blockDim.x) {
                const int tk = i / SU, u = i % SU;
                float v = 0.0f;
                if (tk < nt) {
                    const int p = list[begin + tg + tk], row = mode < 2 ? p / k : p;
                    v = ad[static_cast<long long>(row) * subs + step * SU + u];
                }
                S.xd[tk][u] = v;
            }
            __syncthreads();
            for (int i = threadIdx.x; i < kGmRows * SU; i += blockDim.x) {
                const int rr = i / SU, u = i % SU;
                const int su = step * SU + u;  // sub-block within the row
                const uint8_t* x = raw + min(rr, live_rows - 1) * row_bytes + (su / SPB) * BB;
                const Sub<T> s = decode_smem<T>(x, su % SPB, sg, sks);
                int4* cp = reinterpret_cast<int4*>(&S.codes[rr * ST + u * 32]);
                cp[0] = make_int4(s.w[0], s.w[1], s.w[2], s.w[3]);
                cp[1] = make_int4(s.w[4], s.w[5], s.w[6], s.w[7]);
#pragma unroll
                for (int q = 0; q < G; ++q) {
                    S.ws[rr][u][q] = s.s[q];
                    if constexpr (C::kOff) S.wo[rr][u][q] = s.o[q];
                }
            }
            if constexpr (C::kOff) {
                for (int i = threadIdx.x; i < kGmTok * SU * G; i += blockDim.x) {
                    const int tk = i / (SU * G), u = (i / G) % SU, q = i % G;
                    const int* w4 = reinterpret_cast<const int*>(&S.act[tk * ST + u * 32 + q * (32 / G)]);
                    int z = 0;
#pragma unroll
                    for (int j = 0; j < 8 / G; ++j) z = __dp4a(w4[j], 0x01010101, z);
                    S.xsum[tk][u][q] = z;
                }
            }
            __syncthreads();
            const int ra = warp * 16 + g, rb = ra + 8;
#pragma unroll
            for (int u = 0; u < SU; ++u) {
                const int* ca = reinterpret_cast<const int*>(&S.codes[ra * ST + u * 32]);
                const int* cb = reinterpret_cast<const int*>(&S.codes[rb * ST + u * 32]);
                const int a0 = ca[t], a1 = cb[t], a2 = ca[t + 4], a3 = cb[t + 4];
#pragma unroll
                for (int n = 0; n < kGmTok / 8; ++n) {
                    if (n * 8 >= nt) break;
                    const int* xb = reinterpret_cast<const int*>(&S.act[(n * 8 + g) * ST + u * 32]);
                    const int b0 = xb[t], b1 = xb[t + 4];
                    float va[2], vb[2];  // per token 2t, 2t + 1: rows ra, rb
                    if constexpr (G == 1) {
                        int c[4] = {0, 0, 0, 0};
                        mma_k32(c, a0, a1, a2, a3, b0, b1);
#pragma unroll
                        for (int q = 0; q < 2; ++q) {
                            va[q] = S.ws[ra][u][0] * static_cast<float>(c[q]);
                            vb[q] = S.ws[rb][u][0] * static_cast<float>(c[2 + q]);
                        }
                    } else {
                        int lo[4] = {0, 0, 0, 0}, hi[4] = {0, 0, 0, 0};
                        mma_k16(lo, a0, a1, b0);
                        mma_k16(hi, a2, a3, b1);
#pragma unroll
                        for (int q = 0; q < 2; ++q) {
                            va[q] = S.ws[ra][u][0] * static_cast<float>(lo[q]) + S.ws[ra][u][1] * static_cast<float>(hi[q]);
                            vb[q] = S.ws[rb][u][0] * static_cast<float>(lo[2 + q]) +
                                    S.ws[rb][u][1] * static_cast<float>(hi[2 + q]);
                        }
                    }
#pragma unroll
                    for (int q = 0; q < 2; ++q) {
                        const int tk = n * 8 + 2 * t + q;
                        if constexpr (C::kOff) {
#pragma unroll
                            for (int h = 0; h < G; ++h) {
                                const float z = static_cast<float>(S.xsum[tk][u][h]);
                                va[q] += S.wo[ra][u][h] * z;
                                vb[q] += S.wo[rb][u][h] * z;
                            }
                        }
                        const float xs = S.xd[tk][u];
                        acc[n][q] += xs * va[q];
                        acc[n][2 + q] += xs * vb[q];
                    }
                }
            }
        }
#pragma unroll
        for (int n = 0; n < kGmTok / 8; ++n) {
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int tk = n * 8 + 2 * t + q;
                if (tk >= nt) continue;
                const long long p = list[begin + tg + tk];
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int r = r0 + warp * 16 + g + 8 * h;
                    if (r >= rows) continue;
                    float* o = out + p * rows + r;
                    const float v = acc[n][2 * h + q];
                    if (mode == 1) {
                        const float gt = *o;
                        *o = gt / (1.0f + __expf(-gt)) * v;
                    } else {
                        *o = v;
                    }
                }
            }
        }
    }
}

template <int T>
int moe_mma_launch(const MoeDesc& d, long long mat_off, int rows, int K, int experts, const int* offsets,
                   const int* list, int k, const int8_t* aq, const float* ad, int mode, float* out) {
    using C = Gm<T>;
    if constexpr (C::G > 2) {
        return VELOX_ERR_UNSUPPORTED;
    } else {
        if (K % C::kStep != 0) return VELOX_ERR_UNSUPPORTED;
        const long long row_bytes = static_cast<long long>(K / (sub_per_block<T>() * 32)) * block_bytes(T);
        const long long raw = (kGmRows * row_bytes + 31) / 16 * 16;
        if (raw > kGmRawMax) return VELOX_ERR_UNSUPPORTED;
        const size_t smem = sizeof(typename C::Smem) + static_cast<size_t>(raw);
        if (smem + 4 * moe_grid_words<T>() + 128 > 99 * 1024) return VELOX_ERR_UNSUPPORTED;
        static const cudaError_t attr =
            cudaFuncSetAttribute(moe_mma_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(sizeof(typename Gm<T>::Smem)) + kGmRawMax);
        if (attr != cudaSuccess) return static_cast<int>(attr);
        const dim3 grid((rows + kGmRows - 1) / kGmRows, experts);
        moe_mma_kernel<T><<<grid, 128, smem>>>(d, mat_off, rows, K, offsets, list, k, aq, ad, mode, out);
        return velox_launch_status();
    }
}

#define MOE_MMA(F, ...)                               \
    switch (type) {                                   \
        case kQ2K: return F<kQ2K>(__VA_ARGS__);       \
        case kQ4K: return F<kQ4K>(__VA_ARGS__);       \
        case kIq2Xxs: return F<kIq2Xxs>(__VA_ARGS__); \
        case kIq2Xs: return F<kIq2Xs>(__VA_ARGS__);   \
        case kIq2S: return F<kIq2S>(__VA_ARGS__);     \
        case kIq3Xxs: return F<kIq3Xxs>(__VA_ARGS__); \
        case kIq3S: return F<kIq3S>(__VA_ARGS__);     \
        case kIq4Xs: return F<kIq4Xs>(__VA_ARGS__);   \
        case kQ2_0: return F<kQ2_0>(__VA_ARGS__);             case kIq4Nl: return F<kIq4Nl>(__VA_ARGS__);   \
        default: return VELOX_ERR_UNSUPPORTED;        \
    }

int moe_mma_any(int type, const MoeDesc& d, long long mat_off, int rows, int K, int experts, const int* offsets,
                const int* list, int k, const int8_t* aq, const float* ad, int mode, float* out) {
    MOE_MMA(moe_mma_launch, d, mat_off, rows, K, experts, offsets, list, k, aq, ad, mode, out)
}

#undef MOE_MMA

}  // namespace

// Bytes of scratch velox_5060_moe_grouped needs.
VELOX_API long long velox_5060_moe_grouped_scratch(int rows, int k, int width, int hidden, int experts) {
    const long long pairs = static_cast<long long>(rows) * k;
    const long long xs = static_cast<long long>(rows) * hidden, hw = pairs * width;
    return moe_round(xs) + moe_round(4 * (xs / 32)) + moe_round(4 * hw) + moe_round(hw) + moe_round(4 * (hw / 32)) +
           moe_round(4 * (3LL * experts + 1)) + moe_round(4 * pairs) + moe_round(4 * pairs * hidden);
}

// moe_experts grouped by expert, for many rows; arguments as velox_5060_moe_experts plus the bank
// size. VELOX_ERR_UNSUPPORTED for geometry the kernels do not cover.
VELOX_API int velox_5060_moe_grouped(const float* x, const float* ids, const float* wts, const uint8_t* records,
                                     const uint8_t* slots, const float* map, long long record_bytes,
                                     long long slot_bytes, int up_type, long long gate_off, long long up_off,
                                     int down_type, long long down_off, int width, int hidden, int rows, int k,
                                     int experts, float* y, void* scratch) {
    if (rows < 1 || k < 1 || experts < 1 || up_type < 0 || up_type > kIq4Nl || down_type < 0 || down_type > kIq4Nl ||
        hidden % block_values(up_type) != 0 || width % block_values(down_type) != 0 ||
        reinterpret_cast<uintptr_t>(scratch) % 16 != 0)
        return VELOX_ERR_UNSUPPORTED;
    const MoeDesc d{records, slots, map, record_bytes, slot_bytes, gate_off, up_off, down_off, width, hidden};
    const long long pairs = static_cast<long long>(rows) * k;
    const long long xs = static_cast<long long>(rows) * hidden, hw = pairs * width;
    uint8_t* p = static_cast<uint8_t*>(scratch);
    int8_t* xq = reinterpret_cast<int8_t*>(p);
    float* xd = reinterpret_cast<float*>(p += moe_round(xs));
    float* h = reinterpret_cast<float*>(p += moe_round(4 * (xs / 32)));
    int8_t* hq = reinterpret_cast<int8_t*>(p += moe_round(4 * hw));
    float* hd = reinterpret_cast<float*>(p += moe_round(hw));
    int* counts = reinterpret_cast<int*>(p += moe_round(4 * (hw / 32)));
    int* offsets = counts + experts;
    int* cursor = offsets + experts + 1;
    int* list = reinterpret_cast<int*>(p += moe_round(4 * (3LL * experts + 1)));
    float* o = reinterpret_cast<float*>(p += moe_round(4 * pairs));
    const int threads = 256;
    if (cudaError_t e = cudaMemsetAsync(counts, 0, sizeof(int) * experts, 0)) return static_cast<int>(e);
    moe_count_kernel<<<velox_grid_1d(pairs, threads), threads>>>(ids, static_cast<int>(pairs), experts, counts);
    moe_scan_kernel<<<1, 256>>>(counts, offsets, cursor, experts);
    moe_fill_kernel<<<velox_grid_1d(pairs, threads), threads>>>(ids, static_cast<int>(pairs), experts, offsets, cursor,
                                                               list);
    moe_quant_kernel<<<static_cast<unsigned>((xs / 32 * VELOX_WARP + threads - 1) / threads), threads>>>(
        x, rows, hidden / 32, xq, xd);
    // Tensor cores where the type's scales cover 16 or 32 values; dp4a otherwise (IQ1_M).
    for (int mode = 0; mode < 2; ++mode) {
        const long long off = mode ? up_off : gate_off;
        int e = moe_mma_any(up_type, d, off, width, hidden, experts, offsets, list, k, xq, xd, mode, h);
        if (e == VELOX_ERR_UNSUPPORTED) e = grouped_up_any(up_type, d, off, experts, offsets, list, k, xq, xd, h, mode);
        if (e) return e;
    }
    moe_quant_kernel<<<static_cast<unsigned>((hw / 32 * VELOX_WARP + threads - 1) / threads), threads>>>(
        h, static_cast<int>(pairs), width / 32, hq, hd);
    int e = moe_mma_any(down_type, d, down_off, hidden, width, experts, offsets, list, k, hq, hd, 2, o);
    if (e == VELOX_ERR_UNSUPPORTED) e = grouped_down_any(down_type, d, experts, offsets, list, hq, hd, o);
    if (e) return e;
    moe_sum_kernel<<<velox_grid_1d(static_cast<long long>(rows) * hidden, threads), threads>>>(o, wts, rows, k, hidden,
                                                                                             y);
    return velox_launch_status();
}
