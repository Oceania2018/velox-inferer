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
enum : int { kQ2K = 0, kQ4K, kIq1M, kIq2Xxs, kIq2Xs, kIq2S, kIq3Xxs, kIq3S, kIq4Xs, kQ2_0 };

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
                          : 136;
}

// Scale groups per 32-value sub-block (32 / G values each) and whether values carry an offset
// (value = scale * code + offset).
template <int T>
struct Traits {
    static constexpr int G = (T == kQ2K || T == kIq2Xs || T == kIq2S) ? 2 : T == kIq1M ? 4 : 1;
    static constexpr bool kOff = T == kQ2K || T == kQ4K || T == kIq1M || T == kQ2_0;
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
    constexpr int BB = block_bytes(T);
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
            s[r] = decode<T>(w + (row_begin + row) * row_bytes + static_cast<long long>(u / 8) * BB, u % 8);
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
                                                 : 1;
    static constexpr bool kSigns = T == kIq3Xxs || T == kIq2Xxs || T == kIq2Xs;
    __device__ static const uint32_t* source() {
        if constexpr (T == kIq3S) return kIq3SGrid;
        else if constexpr (T == kIq3Xxs) return kIq3XxsGridG;
        else if constexpr (T == kIq2Xxs) return reinterpret_cast<const uint32_t*>(kIq2XxsGrid);
        else if constexpr (T == kIq2Xs) return reinterpret_cast<const uint32_t*>(kIq2XsGrid);
        else if constexpr (T == kIq2S) return reinterpret_cast<const uint32_t*>(kIq2SGrid);
        else return nullptr;
    }
};

__device__ __forceinline__ uint64_t grid64(const uint32_t* sg, uint32_t i) {
    const uint2 v = reinterpret_cast<const uint2*>(sg)[i];
    return static_cast<uint64_t>(v.x) | (static_cast<uint64_t>(v.y) << 32);
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
    } else {  // IQ2_S
        const uint32_t idx = x[2 + 4 * ib + l] | ((static_cast<uint32_t>(x[66 + ib]) << (8 - 2 * l)) & 0x300);
        const uint64_t g = grid64(sg, idx);
        return signed4(static_cast<uint32_t>(g >> (32 * half)), (x[34 + 4 * ib + l] >> (4 * half)) & 0xf);
    }
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
    if constexpr (T != kIq1M) {
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
    constexpr bool kHalves = Traits<T>::G != 1;
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
    const uint8_t* tile = w + (row_begin + r0) * row_bytes;
    const int last = rows - 1 - r0;
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
        // The minimum term sum_h mn[row][h] * xs[token][h] over the super-block's 16 halves is a
        // 16 x 16 x 8 product per token tile: one f16 mma (exact: mins <= 63, sums <= 2032, f32
        // accumulation). Lane (g, t) holds the A operand's halves 2t, 2t + 1 (sub-block t) and
        // 8 + 2t, 9 + 2t (sub-block 4 + t) of rows g, g + 8.
        uint32_t ma[4] = {0u, 0u, 0u, 0u};
        float2 dma, dmb;
#pragma unroll kPreSbUnroll
        for (int sb = 0; sb < 8; ++sb) {
            int2 pa, pb;
            if constexpr (T == kIq4Xs) {  // words t, t + 4 from one lookup
                pa = pair<T>(xa, sb, t, sg, sks);
                pb = pair<T>(xb, sb, t, sg, sks);
            } else {
                pa = make_int2(word<T>(xa, sb, t, sg, sks), word<T>(xa, sb, t + 4, sg, sks));
                pb = make_int2(word<T>(xb, sb, t, sg, sks), word<T>(xb, sb, t + 4, sg, sks));
            }
            int sca[2], mna[2], scb[2], mnb[2];
            int_scales<T>(xa, sb, sca, mna, dma);
            int_scales<T>(xb, sb, scb, mnb, dmb);
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
                    if constexpr (kHalves) {
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
// 1024 == 0. mode 1 instead writes silu(y) * that (the up projection of a SwiGLU after its gate).
// VELOX_ERR_UNSUPPORTED otherwise.
VELOX_API int velox_5060_linear_ggml_from(const int8_t* xq, const float* xd, int m, const uint8_t* w, int type,
                                          long long k, long long row_begin, int rows, float* y, long long ldy,
                                          int mode) {
    if (m < 1 || m > 65536 || rows <= 0 || k % 256 != 0 || type < 0 || type > 8) return VELOX_ERR_UNSUPPORTED;
    const long long row_bytes = k / 256 * block_bytes(type);
    if (row_bytes % 4 != 0 || k % 1024 != 0 || reinterpret_cast<uintptr_t>(w) % 16 != 0 || mode < 0 || mode > 1)
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
    if (m < 1 || rows <= 0 || k % 1024 != 0 || k > (1ll << 30) || type < 0 || type > 8 || mode < 0 || mode > 1 ||
        reinterpret_cast<uintptr_t>(w) % 16 != 0)
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
        case kIq1M: return launch_pre<kIq1M>(xq, sx, stride, m, w, row_bytes, row_begin, rows, ki, y, ldy, mode);
        case kIq2Xxs: return launch_pre2<kIq2Xxs>(xq, sx, stride, m, w, row_bytes, row_begin, rows, ki, y, ldy, mode);
        case kIq2Xs: return launch_pre2<kIq2Xs>(xq, sx, stride, m, w, row_bytes, row_begin, rows, ki, y, ldy, mode);
        case kIq2S: return launch_pre2<kIq2S>(xq, sx, stride, m, w, row_bytes, row_begin, rows, ki, y, ldy, mode);
        case kIq3Xxs: return launch_pre2<kIq3Xxs>(xq, sx, stride, m, w, row_bytes, row_begin, rows, ki, y, ldy, mode);
        case kIq3S: return launch_pre2<kIq3S>(xq, sx, stride, m, w, row_bytes, row_begin, rows, ki, y, ldy, mode);
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

// 32-value sub-blocks per stored block.
template <int T>
__host__ __device__ constexpr int sub_per_block() {
    return T == kQ2_0 ? 2 : 8;
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

// h[p][i] = silu(gate_i . x) * (up_i . x) for pair p = row * k + j (expert ids[p]).
template <int T>
__global__ void __launch_bounds__(kMoeWarps* VELOX_WARP)
    moe_gate_up_kernel(MoeDesc d, const float* __restrict__ ids, int k, const int8_t* __restrict__ xq,
                       const float* __restrict__ xd, float* __restrict__ h) {
    constexpr int R = kMoeUpRows, SPB = sub_per_block<T>(), BB = block_bytes(T);
    const int p = blockIdx.y, lane = threadIdx.x % VELOX_WARP;
    const int r0 = (blockIdx.x * kMoeWarps + threadIdx.x / VELOX_WARP) * R;
    if (r0 >= d.width) return;
    const int tok = p / k, subs = d.hidden / 32;
    const uint8_t* rec = moe_record(d, static_cast<int>(__float_as_uint(ids[p])));
    const long long row_bytes = static_cast<long long>(subs / SPB) * BB;
    float ag[R] = {}, au[R] = {};
    for (int u = lane; u < subs; u += VELOX_WARP) {
        const int4* ap = reinterpret_cast<const int4*>(xq + (static_cast<long long>(tok) * subs + u) * 32);
        const int4 a0 = __ldg(ap), a1 = __ldg(ap + 1);
        const int a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
        const float dx = __ldg(xd + static_cast<long long>(tok) * subs + u);
        const long long off = static_cast<long long>(u / SPB) * BB;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const long long row = min(r0 + r, d.width - 1) * row_bytes + off;
            ag[r] += dx * sub_dot<T>(decode<T>(rec + d.gate_off + row, u % SPB), a);
            au[r] += dx * sub_dot<T>(decode<T>(rec + d.up_off + row, u % SPB), a);
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float g = velox_warp_sum(ag[r]), v = velox_warp_sum(au[r]);
        if (lane == 0 && r0 + r < d.width)
            h[static_cast<long long>(p) * d.width + r0 + r] = g / (1.0f + __expf(-g)) * v;
    }
}

constexpr int kMoeDownRows = 8;  // per warp; 4 lanes each

// y[tok][c] = sum_j w[p] * (down_c . h[p]) over the token's pairs p = tok * k + j in order.
template <int T>
__global__ void __launch_bounds__(kMoeWarps* VELOX_WARP)
    moe_down_kernel(MoeDesc d, const float* __restrict__ ids, const float* __restrict__ wts, int k,
                    const int8_t* __restrict__ hq, const float* __restrict__ hd, float* __restrict__ y) {
    constexpr int SPB = sub_per_block<T>(), BB = block_bytes(T), L = VELOX_WARP / kMoeDownRows;
    const int tok = blockIdx.y, lane = threadIdx.x % VELOX_WARP;
    const int c = (blockIdx.x * kMoeWarps + threadIdx.x / VELOX_WARP) * kMoeDownRows + lane / L;
    const int sl = lane % L, subs = d.width / 32;
    const long long row_bytes = static_cast<long long>(subs / SPB) * BB;
    const int cr = min(c, d.hidden - 1);
    float acc = 0.0f;
    for (int j = 0; j < k; ++j) {
        const int p = tok * k + j;
        const uint8_t* w = moe_record(d, static_cast<int>(__float_as_uint(ids[p]))) + d.down_off + cr * row_bytes;
        float part = 0.0f;
        for (int u = sl; u < subs; u += L) {
            const int4* ap = reinterpret_cast<const int4*>(hq + (static_cast<long long>(p) * subs + u) * 32);
            const int4 a0 = __ldg(ap), a1 = __ldg(ap + 1);
            const int a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            part += __ldg(hd + static_cast<long long>(p) * subs + u) *
                    sub_dot<T>(decode<T>(w + static_cast<long long>(u / SPB) * BB, u % SPB), a);
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
    const dim3 grid((d.width + per_block - 1) / per_block, pairs);
    moe_gate_up_kernel<T><<<grid, kMoeWarps * VELOX_WARP>>>(d, ids, k, xq, xd, h);
    return velox_launch_status();
}

template <int T>
int moe_down_launch(const MoeDesc& d, const float* ids, const float* wts, int rows, int k, const int8_t* hq,
                    const float* hd, float* y) {
    const int per_block = kMoeWarps * kMoeDownRows;
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
        case kQ2_0: return F<kQ2_0>(__VA_ARGS__);     \
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

// Values per stored block of `type`: the K granularity the kernels need.
int block_values(int type) { return type == kQ2_0 ? 64 : 256; }

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
    if (rows < 1 || k < 1 || up_type < 0 || up_type > kQ2_0 || down_type < 0 || down_type > kQ2_0 ||
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
    const int threads = 256;
    moe_quant_kernel<<<static_cast<unsigned>((xs / 32 * VELOX_WARP + threads - 1) / threads), threads>>>(
        x, rows, hidden / 32, xq, xd);
    if (const int e = moe_up_any(up_type, d, ids, static_cast<int>(pairs), k, xq, xd, h)) return e;
    moe_quant_kernel<<<static_cast<unsigned>((hw / 32 * VELOX_WARP + threads - 1) / threads), threads>>>(
        h, static_cast<int>(pairs), width / 32, hq, hd);
    return moe_down_any(down_type, d, ids, wts, rows, k, hq, hd, y);
}
