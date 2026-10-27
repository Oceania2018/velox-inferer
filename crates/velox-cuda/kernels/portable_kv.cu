// Portable kernels for quantized KV caches: `kv_store` (encode K/V rows) and causal attention that
// reads packed rows directly. Formats and their exact encoding are specified in
// velox-core/src/kv.rs; the writers here are bit-exact with `kv::encode_row`.
//
// Row layout (one (position, kv_head) row, rows contiguous, row stride kv_row_bytes, padded to 16
// bytes for the packed formats):
//   bf16  : head_dim bf16 words
//   fp8   : head_dim E4M3 codes, f32 scale (an f16 value); K rows Hadamard-rotated
//   nvfp4 : head_dim/2 E2M1 code bytes (even element in the low nibble), head_dim/16 E4M3 scales;
//           K and V rows Hadamard-rotated
// Element d of a row lives in lane d % 32, register d / 32 (the attention kernels' layout), so the
// warp-wide Hadamard transform needs no shared memory. Attention rotates the query instead of the
// keys ((Hq).k' = q.(Hk')) and the accumulated output instead of every value row.
#include <math.h>

#include <cuda_fp16.h>

#include "velox_common.cuh"

#define VELOX_KV_F32 0
#define VELOX_KV_BF16 1
#define VELOX_KV_FP8 2
#define VELOX_KV_NVFP4 3

namespace {

__host__ __device__ inline int kv_row_bytes(int fmt, int d) {
    switch (fmt) {
        case VELOX_KV_BF16: return (2 * d + 15) & ~15;
        case VELOX_KV_FP8: return (d + 4 + 15) & ~15;
        case VELOX_KV_NVFP4: return (d / 2 + d / 16 + 15) & ~15;
        default: return 4 * d;
    }
}

// ---- scalar codecs (match velox_core::quant / format bit for bit) ----

__device__ __forceinline__ float kv_bf16_to_f32(unsigned b) { return __uint_as_float(b << 16); }

__device__ __forceinline__ unsigned short kv_f32_to_bf16(float v) {
    const unsigned b = __float_as_uint(v);
    if (isnan(v)) return static_cast<unsigned short>((b >> 16) | 0x40);
    return static_cast<unsigned short>((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);
}

__device__ __forceinline__ float kv_e4m3_to_f32(unsigned b) {
    const unsigned e = (b >> 3) & 15u, m = b & 7u;
    if ((b & 0x7fu) == 0x7fu) return NAN;
    const float v = e == 0 ? static_cast<float>(m) * 0x1p-9f : __uint_as_float(((e + 120u) << 23) | (m << 20));
    return (b & 0x80u) ? -v : v;
}

// f32 -> E4M3FN, round to nearest even, saturating to 448; NaN -> 0x7f.
__device__ __forceinline__ unsigned kv_f32_to_e4m3(float x) {
    if (isnan(x)) return 0x7fu;
    const unsigned sign = (__float_as_uint(x) >> 24) & 0x80u;
    const float a = fabsf(x);
    unsigned code;
    if (a < 0x1p-6f) {
        code = static_cast<unsigned>(__float2int_rn(a * 512.0f));  // subnormal step 2^-9
    } else if (a > 448.0f) {
        code = 0x7eu;
    } else {
        const unsigned ua = __float_as_uint(a);
        int e = static_cast<int>(ua >> 23) - 127;
        unsigned q = (ua >> 20) & 7u;
        const unsigned rem = ua & 0xfffffu;
        if (rem > 0x80000u || (rem == 0x80000u && (q & 1u))) ++q;
        if (q == 8u) {
            q = 0;
            ++e;
        }
        code = (static_cast<unsigned>(e + 7) << 3) | q;
        if (code > 0x7eu) code = 0x7eu;
    }
    return sign | code;
}

__device__ __forceinline__ float kv_e2m1_to_f32(unsigned c) {
    const unsigned e = (c >> 1) & 3u, m = c & 1u;
    const float v = e == 0 ? (m ? 0.5f : 0.0f) : __uint_as_float(((e + 126u) << 23) | (m << 22));
    return (c & 8u) ? -v : v;
}

// f32 -> E2M1 (low nibble), round to nearest even, saturating to 6.
__device__ __forceinline__ unsigned kv_f32_to_e2m1(float x) {
    const unsigned sign = (__float_as_uint(x) >> 28) & 8u;
    const float a = fabsf(x);
    unsigned c;
    if (isnan(a)) {
        c = 7;
    } else if (a < 1.0f) {
        c = static_cast<unsigned>(__float2int_rn(a * 2.0f));
    } else if (a < 2.0f) {
        c = 2u + static_cast<unsigned>(__float2int_rn((a - 1.0f) * 2.0f));
    } else if (a < 4.0f) {
        c = 4u + static_cast<unsigned>(__float2int_rn(a - 2.0f));
    } else {
        const float r = rintf((a - 4.0f) * 0.5f);
        c = r >= 1.0f ? 7u : 6u;
    }
    return sign | c;
}

// f32 -> f16 -> f32 (RNE) for a positive value in [2^-24, 65504].
__device__ __forceinline__ float kv_round_f16(float x) {
    if (x < 0x1p-14f) return rintf(x * 0x1p24f) * 0x1p-24f;
    const unsigned u = __float_as_uint(x);
    return __uint_as_float((u + 0xfffu + ((u >> 13) & 1u)) & ~0x1fffu);
}

// Normalized Sylvester-Hadamard transform of one 32*PER row held as x[p] = row[lane + 32p], in
// ascending stride order (lane bits, then register bits), then one multiply by `norm`. Every lane
// of the warp must call it.
template <int PER>
__device__ __forceinline__ void kv_hadamard(float (&x)[PER], int lane, float norm) {
#pragma unroll
    for (int s = 1; s < VELOX_WARP; s <<= 1) {
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            const float peer = __shfl_xor_sync(0xffffffffu, x[p], s);
            x[p] = (lane & s) ? __fsub_rn(peer, x[p]) : __fadd_rn(x[p], peer);
        }
    }
#pragma unroll
    for (int span = 1; span < PER; span <<= 1) {
#pragma unroll
        for (int base = 0; base < PER; base += 2 * span) {
#pragma unroll
            for (int o = 0; o < span; ++o) {
                const float lo = x[base + o], hi = x[base + o + span];
                x[base + o] = __fadd_rn(lo, hi);
                x[base + o + span] = __fsub_rn(lo, hi);
            }
        }
    }
#pragma unroll
    for (int p = 0; p < PER; ++p) x[p] = __fmul_rn(x[p], norm);
}

template <int FMT>
struct KvRot {
    static constexpr bool k = FMT == VELOX_KV_FP8 || FMT == VELOX_KV_NVFP4;
    static constexpr bool v = FMT == VELOX_KV_NVFP4;
};

// ---- kv_store ----

// Encodes one row (`src`, head_dim values) into `dst`. One warp per row.
template <int FMT, int PER>
__device__ __forceinline__ void kv_store_row(const float* __restrict__ src, unsigned char* __restrict__ dst, int d,
                                             int lane, float norm, bool rotate) {
    float x[PER];
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int i = lane + p * VELOX_WARP;
        x[p] = i < d ? src[i] : 0.0f;
    }
    if (rotate) kv_hadamard<PER>(x, lane, norm);  // warp-uniform
    if (FMT == VELOX_KV_F32) {
        float* w = reinterpret_cast<float*>(dst);
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            const int i = lane + p * VELOX_WARP;
            if (i < d) w[i] = x[p];
        }
    } else if (FMT == VELOX_KV_BF16) {
        unsigned short* w = reinterpret_cast<unsigned short*>(dst);
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            const int i = lane + p * VELOX_WARP;
            if (i < d) w[i] = kv_f32_to_bf16(x[p]);
        }
    } else if (FMT == VELOX_KV_FP8) {
        // d == 32 * PER (checked on the host).
        float amax = 0.0f;
#pragma unroll
        for (int p = 0; p < PER; ++p) amax = fmaxf(amax, fabsf(x[p]));
        amax = velox_warp_max(amax);
        float scale = 0.0f, inv = 0.0f;
        if (amax > 0.0f) {
            scale = kv_round_f16(fminf(65504.0f, fmaxf(0x1p-24f, __fdiv_rn(amax, 448.0f))));
            inv = __fdiv_rn(1.0f, scale);
        }
#pragma unroll
        for (int p = 0; p < PER; ++p)
            dst[lane + p * VELOX_WARP] =
                inv == 0.0f ? 0 : static_cast<unsigned char>(kv_f32_to_e4m3(__fmul_rn(x[p], inv)));
        if (lane == 0) *reinterpret_cast<float*>(dst + d) = scale;
    } else {
        // NVFP4: register p holds groups 2p (lanes 0..15) and 2p+1 (lanes 16..31).
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            float amax = fabsf(x[p]);
#pragma unroll
            for (int o = 1; o < 16; o <<= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
            unsigned sc = 0, code = 0;
            if (amax > 0.0f) {
                sc = kv_f32_to_e4m3(fminf(448.0f, fmaxf(0x1p-9f, __fdiv_rn(amax, 6.0f))));
                code = kv_f32_to_e2m1(__fdiv_rn(x[p], kv_e4m3_to_f32(sc)));
            }
            const unsigned hi = __shfl_down_sync(0xffffffffu, code, 1);
            const int i = lane + p * VELOX_WARP;
            if ((lane & 1) == 0) dst[i >> 1] = static_cast<unsigned char>(code | (hi << 4));
            if ((lane & 15) == 0) dst[d / 2 + (i >> 4)] = static_cast<unsigned char>(sc);
        }
    }
}

constexpr int kStoreWarps = 8;

template <int FMT, int PER>
__global__ void __launch_bounds__(kStoreWarps* VELOX_WARP)
    kv_store_kernel(const float* __restrict__ k, const float* __restrict__ v, unsigned char* __restrict__ kc,
                    unsigned char* __restrict__ vc, int units, int kv_heads, int head_dim, long long pos0, float norm,
                    const long long* __restrict__ pp) {
    if (pp != nullptr) {  // staged by a graph step: [pos0, k_cache, v_cache]
        pos0 = pp[0];
        kc = reinterpret_cast<unsigned char*>(pp[1]);
        vc = reinterpret_cast<unsigned char*>(pp[2]);
    }
    const int warp = threadIdx.x / VELOX_WARP, lane = threadIdx.x % VELOX_WARP;
    const int unit = blockIdx.x * kStoreWarps + warp;  // token * kv_heads + kv_head
    if (unit >= units) return;                         // warp-uniform
    const size_t src = static_cast<size_t>(unit) * head_dim;
    const size_t row = static_cast<size_t>(pos0) * kv_heads + unit;
    const size_t rb = static_cast<size_t>(kv_row_bytes(FMT, head_dim));
    kv_store_row<FMT, PER>(k + src, kc + row * rb, head_dim, lane, norm, KvRot<FMT>::k);
    kv_store_row<FMT, PER>(v + src, vc + row * rb, head_dim, lane, norm, KvRot<FMT>::v);
}

template <int FMT>
int kv_store_launch(const float* k, const float* v, unsigned char* kc, unsigned char* vc, int tokens, int kv_heads,
                    int head_dim, long long pos0, float norm, const long long* pp) {
    const int units = tokens * kv_heads;
    const unsigned grid = static_cast<unsigned>((units + kStoreWarps - 1) / kStoreWarps);
    const dim3 block(kStoreWarps * VELOX_WARP);
    if (head_dim <= 32)
        kv_store_kernel<FMT, 1><<<grid, block>>>(k, v, kc, vc, units, kv_heads, head_dim, pos0, norm, pp);
    else if (head_dim <= 64)
        kv_store_kernel<FMT, 2><<<grid, block>>>(k, v, kc, vc, units, kv_heads, head_dim, pos0, norm, pp);
    else if (head_dim <= 128)
        kv_store_kernel<FMT, 4><<<grid, block>>>(k, v, kc, vc, units, kv_heads, head_dim, pos0, norm, pp);
    else
        kv_store_kernel<FMT, 8><<<grid, block>>>(k, v, kc, vc, units, kv_heads, head_dim, pos0, norm, pp);
    return velox_launch_status();
}

// ---- row loaders ----

// Lane `lane`'s elements x[p] = row[lane + 32p] (zero past head_dim).
template <int FMT, int PER>
__device__ __forceinline__ void kv_load_lane(const unsigned char* __restrict__ row, int d, int lane,
                                             float (&x)[PER]) {
    if (FMT == VELOX_KV_F32) {
        const float* w = reinterpret_cast<const float*>(row);
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            const int i = lane + p * VELOX_WARP;
            x[p] = i < d ? w[i] : 0.0f;
        }
    } else if (FMT == VELOX_KV_BF16) {
        const unsigned short* w = reinterpret_cast<const unsigned short*>(row);
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            const int i = lane + p * VELOX_WARP;
            x[p] = i < d ? kv_bf16_to_f32(w[i]) : 0.0f;
        }
    } else if (FMT == VELOX_KV_FP8) {
        const float s = *reinterpret_cast<const float*>(row + d);
#pragma unroll
        for (int p = 0; p < PER; ++p) x[p] = kv_e4m3_to_f32(row[lane + p * VELOX_WARP]) * s;
    } else {
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            const int i = lane + p * VELOX_WARP;
            const unsigned byte = row[i >> 1];
            x[p] = kv_e2m1_to_f32((lane & 1) ? byte >> 4 : byte & 15u) * kv_e4m3_to_f32(row[d / 2 + (i >> 4)]);
        }
    }
}

// Elements of one 16-byte code chunk, per format.
template <int FMT>
struct KvChunk {
    static constexpr int elems = FMT == VELOX_KV_F32 ? 4 : FMT == VELOX_KV_BF16 ? 8 : FMT == VELOX_KV_FP8 ? 16 : 32;
};

// Scale bits chunk `c` of a row needs (fp8: the row's f32 scale; nvfp4: its two E4M3 group scales).
template <int FMT>
__device__ __forceinline__ unsigned kv_chunk_scale(const unsigned char* __restrict__ row, int d, int c) {
    if (FMT == VELOX_KV_FP8) return *reinterpret_cast<const unsigned*>(row + d);
    if (FMT == VELOX_KV_NVFP4) return *reinterpret_cast<const unsigned short*>(row + d / 2 + 2 * c);
    return 0;
}

// Two E4M3 codes (low 16 bits of x) as f32 values times 2^-8: the code bits placed into f16
// exponent/mantissa fields (subnormals included), then converted. Finite codes only.
__device__ __forceinline__ float2 kv_e4m3x2_f16_scaled(unsigned x) {
    const unsigned h = ((x & 0x7fu) << 7) | ((x & 0x80u) << 8) | ((x & 0x7f00u) << 15) | ((x & 0x8000u) << 16);
    return __half22float2(*reinterpret_cast<const __half2*>(&h));
}

// Two E2M1 codes (low 8 bits of x) as f32 values times 2^-14, as above.
__device__ __forceinline__ float2 kv_e2m1x2_f16_scaled(unsigned x) {
    const unsigned h = ((x & 7u) << 9) | ((x & 8u) << 12) | ((x & 0x70u) << 21) | ((x & 0x80u) << 24);
    return __half22float2(*reinterpret_cast<const __half2*>(&h));
}

// Dequantizes one chunk (16 code bytes `w`, scale bits `sc`) into dst[0 .. elems] (16-byte aligned).
// Products of a code and its scale are exact in f32, so this equals the reference decode.
template <int FMT>
__device__ __forceinline__ void kv_dequant_chunk(uint4 w, unsigned sc, float* __restrict__ dst) {
    float4* o = reinterpret_cast<float4*>(dst);
    const unsigned u[4] = {w.x, w.y, w.z, w.w};
    if (FMT == VELOX_KV_F32) {
        o[0] = make_float4(__uint_as_float(u[0]), __uint_as_float(u[1]), __uint_as_float(u[2]), __uint_as_float(u[3]));
    } else if (FMT == VELOX_KV_BF16) {
#pragma unroll
        for (int j = 0; j < 2; ++j)
            o[j] = make_float4(__uint_as_float(u[2 * j] << 16), __uint_as_float(u[2 * j] & 0xffff0000u),
                               __uint_as_float(u[2 * j + 1] << 16), __uint_as_float(u[2 * j + 1] & 0xffff0000u));
    } else if (FMT == VELOX_KV_FP8) {
        const float s = __uint_as_float(sc) * 256.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 a = kv_e4m3x2_f16_scaled(u[j]), b = kv_e4m3x2_f16_scaled(u[j] >> 16);
            o[j] = make_float4(a.x * s, a.y * s, b.x * s, b.y * s);
        }
    } else {
        // Words 0-1: group 0 (elements 0..15); words 2-3: group 1.
        const float s0 = kv_e4m3_to_f32(sc & 0xffu) * 16384.0f, s1 = kv_e4m3_to_f32((sc >> 8) & 0xffu) * 16384.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float s = j < 2 ? s0 : s1;
            const float2 a = kv_e2m1x2_f16_scaled(u[j]), b = kv_e2m1x2_f16_scaled(u[j] >> 8),
                         c = kv_e2m1x2_f16_scaled(u[j] >> 16), d = kv_e2m1x2_f16_scaled(u[j] >> 24);
            o[2 * j] = make_float4(a.x * s, a.y * s, b.x * s, b.y * s);
            o[2 * j + 1] = make_float4(c.x * s, c.y * s, d.x * s, d.y * s);
        }
    }
}

// Sums x[0..8] over the warp and scatters the result: lane l returns the total of x[l & 7].
__device__ __forceinline__ float kv_reduce8(const float (&x)[8], int lane) {
    const unsigned full = 0xffffffffu;
    float a[4], c[2];
    const bool b2 = lane & 4, b1 = lane & 2, b0 = lane & 1;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float send = b2 ? x[i] : x[i + 4];
        a[i] = (b2 ? x[i + 4] : x[i]) + __shfl_xor_sync(full, send, 4);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const float send = b1 ? a[i] : a[i + 2];
        c[i] = (b1 ? a[i + 2] : a[i]) + __shfl_xor_sync(full, send, 2);
    }
    float s = (b0 ? c[1] : c[0]) + __shfl_xor_sync(full, b0 ? c[0] : c[1], 1);
    s += __shfl_xor_sync(full, s, 8);
    s += __shfl_xor_sync(full, s, 16);
    return s;
}

// ---- attention ----

// Grow-only scratch for split partials (engine work is serialized).
float* kvq_scratch(size_t floats) {
    static VeloxScratch s;
    return static_cast<float*>(s.get(floats * sizeof(float)));
}

// Merges split partials; with ROT the merged row is un-rotated (H applied) before it is written.
template <int PER, bool ROT>
__global__ void kvq_combine_kernel(const float* __restrict__ part_acc, const float* __restrict__ part_ml,
                                   float* __restrict__ out, int heads, int head_dim, int splits, float norm) {
    const int h = blockIdx.x, t = blockIdx.y, lane = threadIdx.x;
    const size_t base = (static_cast<size_t>(t) * heads + h) * splits;
    float gm = -INFINITY;
    for (int sp = 0; sp < splits; ++sp)
        if (part_ml[2 * (base + sp) + 1] > 0.0f) gm = fmaxf(gm, part_ml[2 * (base + sp)]);
    float acc[PER], gl = 0.0f;
#pragma unroll
    for (int p = 0; p < PER; ++p) acc[p] = 0.0f;
    for (int sp = 0; sp < splits; ++sp) {
        const float l = part_ml[2 * (base + sp) + 1];
        if (l <= 0.0f) continue;  // a split past the causal limit saw no keys
        const float w = expf(part_ml[2 * (base + sp)] - gm);
        gl += l * w;
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            const int d = lane + p * VELOX_WARP;
            if (d < head_dim) acc[p] += part_acc[(base + sp) * head_dim + d] * w;
        }
    }
#pragma unroll
    for (int p = 0; p < PER; ++p) acc[p] /= gl;
    if (ROT) kv_hadamard<PER>(acc, lane, norm);
    const size_t o_off = (static_cast<size_t>(t) * heads + h) * head_dim;
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int d = lane + p * VELOX_WARP;
        if (d < head_dim) out[o_off + d] = acc[p];
    }
}

// Query row `qp` (head_dim values) scaled, rotated when the keys are.
template <int FMT, int PER>
__device__ __forceinline__ void kvq_load_query(const float* __restrict__ qp, bool valid, int d, int lane, float scale,
                                               float norm, float (&qr)[PER]) {
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int i = lane + p * VELOX_WARP;
        qr[p] = valid && i < d ? qp[i] * scale : 0.0f;
    }
    if (KvRot<FMT>::k) kv_hadamard<PER>(qr, lane, norm);
}

// One warp per (token, head, split): online softmax over its keys, rows dequantized in registers.
template <int FMT, int PER>
__global__ void kvq_partial_kernel(const float* __restrict__ q, const unsigned char* __restrict__ k,
                                   const unsigned char* __restrict__ v, float* __restrict__ part_acc,
                                   float* __restrict__ part_ml, int heads, int kv_heads, int head_dim, int pos0,
                                   float scale, int splits, int split_keys, float norm) {
    const int h = blockIdx.x, t = blockIdx.y, sp = blockIdx.z, lane = threadIdx.x;
    const int kvh = h / (heads / kv_heads);
    const size_t rb = static_cast<size_t>(kv_row_bytes(FMT, head_dim));
    const int visible = pos0 + t + 1;
    const int j0 = sp * split_keys, j1 = min(j0 + split_keys, visible);
    float qr[PER], acc[PER];
    kvq_load_query<FMT, PER>(q + (static_cast<size_t>(t) * heads + h) * head_dim, true, head_dim, lane, scale, norm,
                             qr);
#pragma unroll
    for (int p = 0; p < PER; ++p) acc[p] = 0.0f;
    float m = -INFINITY, l = 0.0f;
    for (int j = j0; j < j1; ++j) {
        const size_t row = (static_cast<size_t>(j) * kv_heads + kvh) * rb;
        float kx[PER], vx[PER];
        kv_load_lane<FMT, PER>(k + row, head_dim, lane, kx);
        kv_load_lane<FMT, PER>(v + row, head_dim, lane, vx);
        float s = 0.0f;
#pragma unroll
        for (int p = 0; p < PER; ++p) s += qr[p] * kx[p];
        s = velox_warp_sum(s);
        const float m_new = fmaxf(m, s);
        const float corr = expf(m - m_new), pj = expf(s - m_new);
        l = l * corr + pj;
#pragma unroll
        for (int p = 0; p < PER; ++p) acc[p] = acc[p] * corr + pj * vx[p];
        m = m_new;
    }
    const size_t idx = (static_cast<size_t>(t) * heads + h) * splits + sp;
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int d = lane + p * VELOX_WARP;
        if (d < head_dim) part_acc[idx * head_dim + d] = acc[p];
    }
    if (lane == 0) {
        part_ml[2 * idx] = m;
        part_ml[2 * idx + 1] = l;
    }
}

// Verify / decode attention (few tokens, GQA): one block owns a KV head, a key split and up to
// 8 * RW query rows (token x head of the group). Each K/V tile is read from global memory once with
// 16-byte loads, dequantized into shared memory, and served to every row. Scores avoid per-key
// warp reductions: a warp keeps RW query rows in registers (lane holds elements lane + 32p),
// forms partial dot products for 8 keys at a time and reduce-scatters them, so lane l ends with the
// score of key l % 8.
constexpr int kVerifyWarps = 8;

template <int FMT>
struct KvTile {
    // 16 keys: K+V tiles take 32 KB of the ~100 KB of shared memory an SM has (sm_89, sm_120),
    // leaving room for three resident blocks.
    static constexpr int keys = 16;
    // 16-byte chunks each thread loads per tile (K and V) at head_dim 256.
    static constexpr int loads = (2 * keys * (256 / KvChunk<FMT>::elems) + kVerifyWarps * VELOX_WARP - 1) /
                                 (kVerifyWarps * VELOX_WARP);
};

// Scores are kept in the log2 domain (the query is pre-scaled by log2(e)) so softmax uses exp2.
// Resident blocks per SM each row count is compiled for (bounds registers per thread).
template <int RW>
struct kVerifyMinBlocks {
    static constexpr int value = RW <= 2 ? 3 : RW <= 3 ? 2 : 1;
};

constexpr float kLog2e = 1.4426950408889634f, kLn2 = 0.6931471805599453f;

template <int FMT, int PER, int RW>
__global__ void __launch_bounds__(kVerifyWarps* VELOX_WARP, kVerifyMinBlocks<RW>::value)
    kvq_verify_kernel(const float* __restrict__ q, const unsigned char* __restrict__ k,
                      const unsigned char* __restrict__ v, float* __restrict__ part_acc, float* __restrict__ part_ml,
                      int tokens, int heads, int kv_heads, int pos0, float scale, int splits, int split_keys,
                      float norm, const long long* __restrict__ pp) {
    if (pp != nullptr) {  // staged by a graph step: [pos0, k_cache, v_cache]
        pos0 = static_cast<int>(pp[0]);
        k = reinterpret_cast<const unsigned char*>(pp[1]);
        v = reinterpret_cast<const unsigned char*>(pp[2]);
    }
    constexpr int D = PER * VELOX_WARP;  // head_dim
    constexpr int TK = KvTile<FMT>::keys, EPC = KvChunk<FMT>::elems, CPR = D / EPC;
    constexpr int THREADS = kVerifyWarps * VELOX_WARP;
    constexpr int LOADS = (2 * TK * CPR + THREADS - 1) / THREADS;
    extern __shared__ float4 kvq_tile4[];
    float* ks = reinterpret_cast<float*>(kvq_tile4);
    float* vs = ks + TK * D;
    const int rg = blockIdx.x, kvh = blockIdx.y, sp = blockIdx.z;
    const int group = heads / kv_heads, rows = tokens * group;
    const int warp = threadIdx.x / VELOX_WARP, lane = threadIdx.x % VELOX_WARP;
    const size_t rb = static_cast<size_t>(kv_row_bytes(FMT, D));
    float qr[RW][PER], acc[RW][PER], m[RW], l[RW];
    int last[RW], wlast = -1;
#pragma unroll
    for (int i = 0; i < RW; ++i) {
        const int r = (rg * kVerifyWarps + warp) * RW + i;
        const int t = r / group, h = kvh * group + r % group;
        last[i] = r < rows ? pos0 + t : -1;
        wlast = max(wlast, last[i]);
        m[i] = -INFINITY;
        l[i] = 0.0f;
        kvq_load_query<FMT, PER>(q + (static_cast<size_t>(t) * heads + h) * D, r < rows, D, lane, scale * kLog2e,
                                 norm, qr[i]);
#pragma unroll
        for (int p = 0; p < PER; ++p) acc[i][p] = 0.0f;
    }
    const int j0 = sp * split_keys, j1 = min(j0 + split_keys, pos0 + tokens);
    // The next tile's packed bytes are loaded into registers (several 16-byte loads in flight per
    // thread) while the current tile is computed, then dequantized into shared memory. Keys past
    // `n` are zero-filled so the compute loops need no bounds checks.
    uint4 raw[LOADS];
    unsigned sc[LOADS];
    auto load_tile = [&](int base, int n) {
#pragma unroll
        for (int f = 0; f < LOADS; ++f) {
            const int idx = threadIdx.x + f * THREADS;
            const int which = idx / (TK * CPR), key = idx / CPR % TK, c = idx % CPR;
            raw[f] = make_uint4(0, 0, 0, 0);
            sc[f] = 0;
            if (which < 2 && key < n) {
                const unsigned char* row = (which ? v : k) + (static_cast<size_t>(base + key) * kv_heads + kvh) * rb;
                raw[f] = *reinterpret_cast<const uint4*>(row + 16 * c);
                sc[f] = kv_chunk_scale<FMT>(row, D, c);
            }
        }
    };
    if (j0 < j1) load_tile(j0, min(TK, j1 - j0));
    for (int base = j0; base < j1; base += TK) {
        const int n = min(TK, j1 - base);
        __syncthreads();
#pragma unroll
        for (int f = 0; f < LOADS; ++f) {
            const int idx = threadIdx.x + f * THREADS;
            const int which = idx / (TK * CPR), key = idx / CPR % TK, c = idx % CPR;
            if (which < 2) kv_dequant_chunk<FMT>(raw[f], sc[f], (which ? vs : ks) + key * D + c * EPC);
        }
        __syncthreads();
        if (base + TK < j1) load_tile(base + TK, min(TK, j1 - base - TK));
        const int stop = min(n, wlast - base + 1);  // warp-uniform
        for (int kc = 0; kc < stop; kc += 8) {
            float part[RW][8];
#pragma unroll
            for (int jj = 0; jj < 8; ++jj) {
                const float* kr = ks + (kc + jj) * D + lane;
                float kx[PER];
#pragma unroll
                for (int p = 0; p < PER; ++p) kx[p] = kr[p * VELOX_WARP];
#pragma unroll
                for (int i = 0; i < RW; ++i) {
                    float s = 0.0f;
#pragma unroll
                    for (int p = 0; p < PER; ++p) s = fmaf(qr[i][p], kx[p], s);
                    part[i][jj] = s;
                }
            }
            const int key = base + kc + (lane & 7);
            const bool in_tile = kc + (lane & 7) < n;
            float pj[RW];
#pragma unroll
            for (int i = 0; i < RW; ++i) {
                float s = kv_reduce8(part[i], lane);
                s = in_tile && key <= last[i] ? s : -INFINITY;
                float cm = fmaxf(s, __shfl_xor_sync(0xffffffffu, s, 1));
                cm = fmaxf(cm, __shfl_xor_sync(0xffffffffu, cm, 2));
                cm = fmaxf(cm, __shfl_xor_sync(0xffffffffu, cm, 4));
                if (cm > m[i]) {  // warp-uniform: rescale only when the running max grows
                    const float corr = exp2f(m[i] - cm);
                    l[i] *= corr;
#pragma unroll
                    for (int p = 0; p < PER; ++p) acc[i][p] *= corr;
                    m[i] = cm;
                }
                pj[i] = s == -INFINITY ? 0.0f : exp2f(s - m[i]);
                float ls = pj[i] + __shfl_xor_sync(0xffffffffu, pj[i], 1);
                ls += __shfl_xor_sync(0xffffffffu, ls, 2);
                ls += __shfl_xor_sync(0xffffffffu, ls, 4);
                l[i] += ls;
            }
#pragma unroll
            for (int jj = 0; jj < 8; ++jj) {
                const float* vr = vs + (kc + jj) * D + lane;
                float vx[PER];
#pragma unroll
                for (int p = 0; p < PER; ++p) vx[p] = vr[p * VELOX_WARP];
#pragma unroll
                for (int i = 0; i < RW; ++i) {
                    const float w = __shfl_sync(0xffffffffu, pj[i], jj);
#pragma unroll
                    for (int p = 0; p < PER; ++p) acc[i][p] = fmaf(w, vx[p], acc[i][p]);
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < RW; ++i) {
        const int r = (rg * kVerifyWarps + warp) * RW + i;
        if (r >= rows) continue;
        const int t = r / group, h = kvh * group + r % group;
        const size_t idx = (static_cast<size_t>(t) * heads + h) * splits + sp;
#pragma unroll
        for (int p = 0; p < PER; ++p) part_acc[idx * D + lane + p * VELOX_WARP] = acc[i][p];
        if (lane == 0) {
            part_ml[2 * idx] = m[i] * kLn2;  // natural-log domain for the combine kernel
            part_ml[2 * idx + 1] = l[i];
        }
    }
}

int kvq_sm_count() {
    static int sms = 0;
    if (sms == 0) {
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || sms <= 0)
            sms = 128;
    }
    return sms;
}

template <int FMT, int PER, int RW>
int kvq_verify(const float* q, const unsigned char* k, const unsigned char* v, float* out, int tokens, int heads,
               int kv_heads, int head_dim, int pos0, float scale, float norm, const long long* pp, int kv_grid) {
    constexpr int TK = KvTile<FMT>::keys;
    // Graph steps size the grid for their KV bucket; the kernel reads the actual position.
    // Sized for the KV bucket in every mode, so eager and recorded steps reduce in the same order.
    const int kv = kv_grid >= pos0 + tokens ? kv_grid : pos0 + tokens;
    const int rows = tokens * (heads / kv_heads);
    const int rgs = (rows + kVerifyWarps * RW - 1) / (kVerifyWarps * RW);
    // One wave of resident blocks; splits of whole tiles.
    const int want = (kVerifyMinBlocks<RW>::value * kvq_sm_count() + kv_heads * rgs - 1) / (kv_heads * rgs);
    int split_keys = (kv + want - 1) / want;
    split_keys = (split_keys + TK - 1) / TK * TK;
    const int splits = (kv + split_keys - 1) / split_keys;
    const size_t parts = static_cast<size_t>(tokens) * heads * splits;
    float* scratch = kvq_scratch(parts * (head_dim + 2));
    if (scratch == nullptr) return static_cast<int>(cudaErrorMemoryAllocation);
    float* part_acc = scratch;
    float* part_ml = scratch + parts * head_dim;
    const size_t smem = static_cast<size_t>(2) * TK * head_dim * sizeof(float);
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(kvq_verify_kernel<FMT, PER, RW>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(2 * TK * 256 * sizeof(float)));
        // Without a full carveout the driver may leave room for only one block per SM.
        cudaFuncSetAttribute(kvq_verify_kernel<FMT, PER, RW>, cudaFuncAttributePreferredSharedMemoryCarveout,
                             static_cast<int>(cudaSharedmemCarveoutMaxShared));
        configured = true;
    }
    kvq_verify_kernel<FMT, PER, RW><<<dim3(rgs, kv_heads, splits), kVerifyWarps * VELOX_WARP, smem>>>(
        q, k, v, part_acc, part_ml, tokens, heads, kv_heads, pos0, scale, splits, split_keys, norm, pp);
    kvq_combine_kernel<PER, KvRot<FMT>::v><<<dim3(heads, tokens), VELOX_WARP>>>(part_acc, part_ml, out, heads,
                                                                               head_dim, splits, norm);
    return velox_launch_status();
}

// One warp per (token, head) over all its keys (large prefill chunks).
template <int FMT, int PER>
__global__ void kvq_warp_kernel(const float* __restrict__ q, const unsigned char* __restrict__ k,
                                const unsigned char* __restrict__ v, float* __restrict__ out, int heads, int kv_heads,
                                int head_dim, int pos0, float scale, float norm) {
    const int h = blockIdx.x, t = blockIdx.y, lane = threadIdx.x;
    const int kvh = h / (heads / kv_heads);
    const size_t q_off = (static_cast<size_t>(t) * heads + h) * head_dim;
    const size_t rb = static_cast<size_t>(kv_row_bytes(FMT, head_dim));
    float qr[PER], acc[PER];
    kvq_load_query<FMT, PER>(q + q_off, true, head_dim, lane, scale, norm, qr);
#pragma unroll
    for (int p = 0; p < PER; ++p) acc[p] = 0.0f;
    float m = -INFINITY, l = 0.0f;
    const int visible = pos0 + t + 1;
    for (int j = 0; j < visible; ++j) {
        const size_t row = (static_cast<size_t>(j) * kv_heads + kvh) * rb;
        float kx[PER], vx[PER];
        kv_load_lane<FMT, PER>(k + row, head_dim, lane, kx);
        kv_load_lane<FMT, PER>(v + row, head_dim, lane, vx);
        float s = 0.0f;
#pragma unroll
        for (int p = 0; p < PER; ++p) s += qr[p] * kx[p];
        s = velox_warp_sum(s);
        const float m_new = fmaxf(m, s);
        const float corr = expf(m - m_new), pj = expf(s - m_new);
        l = l * corr + pj;
#pragma unroll
        for (int p = 0; p < PER; ++p) acc[p] = acc[p] * corr + pj * vx[p];
        m = m_new;
    }
#pragma unroll
    for (int p = 0; p < PER; ++p) acc[p] /= l;
    if (KvRot<FMT>::v) kv_hadamard<PER>(acc, lane, norm);
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int d = lane + p * VELOX_WARP;
        if (d < head_dim) out[q_off + d] = acc[p];
    }
}

// Most query rows a verify warp keeps in registers (6: a whole 8-token x 6-head window per block).
constexpr int kVerifyMaxRw = 6;

template <int FMT, int PER>
int kvq_attention(const float* q, const unsigned char* k, const unsigned char* v, float* out, int tokens, int heads,
                  int kv_heads, int head_dim, int pos0, float scale, float norm, const long long* pp, int kv_grid) {
    const int kv = pos0 + tokens;
    const int group = heads / kv_heads;
    const bool verify = tokens <= 16 && head_dim == 32 * PER && group > 1;
    if (pp != nullptr && !verify) return VELOX_ERR_UNSUPPORTED;  // only the verify path stages
    if (verify) {
        // Fewest row groups (each re-reads K/V), then the fewest rows per warp that cover them.
        const int rows = tokens * group;
        const int rgs = (rows + kVerifyWarps * kVerifyMaxRw - 1) / (kVerifyWarps * kVerifyMaxRw);
        const int rw = (rows + kVerifyWarps * rgs - 1) / (kVerifyWarps * rgs);
        switch (rw) {
            case 1: return kvq_verify<FMT, PER, 1>(q, k, v, out, tokens, heads, kv_heads, head_dim, pos0, scale, norm, pp, kv_grid);
            case 2: return kvq_verify<FMT, PER, 2>(q, k, v, out, tokens, heads, kv_heads, head_dim, pos0, scale, norm, pp, kv_grid);
            case 3: return kvq_verify<FMT, PER, 3>(q, k, v, out, tokens, heads, kv_heads, head_dim, pos0, scale, norm, pp, kv_grid);
            case 4: return kvq_verify<FMT, PER, 4>(q, k, v, out, tokens, heads, kv_heads, head_dim, pos0, scale, norm, pp, kv_grid);
            case 5: return kvq_verify<FMT, PER, 5>(q, k, v, out, tokens, heads, kv_heads, head_dim, pos0, scale, norm, pp, kv_grid);
            default:
                return kvq_verify<FMT, PER, 6>(q, k, v, out, tokens, heads, kv_heads, head_dim, pos0, scale, norm, pp, kv_grid);
        }
    }
    if (tokens * heads > 2048) {
        kvq_warp_kernel<FMT, PER><<<dim3(heads, tokens), VELOX_WARP>>>(q, k, v, out, heads, kv_heads, head_dim, pos0,
                                                                      scale, norm);
        return velox_launch_status();
    }
    // ~4096 partial warps, 16..256 keys each.
    const int want = (4096 + tokens * heads - 1) / (tokens * heads);
    int split_keys = (kv + want - 1) / want;
    split_keys = split_keys < 16 ? 16 : split_keys > 256 ? 256 : split_keys;
    const int splits = (kv + split_keys - 1) / split_keys;
    const size_t parts = static_cast<size_t>(tokens) * heads * splits;
    float* scratch = kvq_scratch(parts * (head_dim + 2));
    if (scratch == nullptr) return static_cast<int>(cudaErrorMemoryAllocation);
    float* part_acc = scratch;
    float* part_ml = scratch + parts * head_dim;
    kvq_partial_kernel<FMT, PER><<<dim3(heads, tokens, splits), VELOX_WARP>>>(
        q, k, v, part_acc, part_ml, heads, kv_heads, head_dim, pos0, scale, splits, split_keys, norm);
    kvq_combine_kernel<PER, KvRot<FMT>::v><<<dim3(heads, tokens), VELOX_WARP>>>(part_acc, part_ml, out, heads,
                                                                               head_dim, splits, norm);
    return velox_launch_status();
}

template <int FMT>
int kvq_dispatch(const float* q, const unsigned char* k, const unsigned char* v, float* out, int tokens, int heads,
                 int kv_heads, int head_dim, int pos0, float scale, float norm, const long long* pp,
                 int kv_grid) {
    if (head_dim <= 32) return kvq_attention<FMT, 1>(q, k, v, out, tokens, heads, kv_heads, head_dim, pos0, scale, norm, pp, kv_grid);
    if (head_dim <= 64) return kvq_attention<FMT, 2>(q, k, v, out, tokens, heads, kv_heads, head_dim, pos0, scale, norm, pp, kv_grid);
    if (head_dim <= 128)
        return kvq_attention<FMT, 4>(q, k, v, out, tokens, heads, kv_heads, head_dim, pos0, scale, norm, pp, kv_grid);
    return kvq_attention<FMT, 8>(q, k, v, out, tokens, heads, kv_heads, head_dim, pos0, scale, norm, pp, kv_grid);
}

}  // namespace

// Rotated formats need head_dim a power of two in 32..256 (validated by velox_core::check).
// params: null, or device [pos0, k_cache, v_cache] staged by a graph step.
VELOX_API int velox_portable_kv_store(const float* k, const float* v, unsigned char* k_cache, unsigned char* v_cache,
                                      int tokens, int kv_heads, int head_dim, long long pos0, int format, float norm,
                                      const long long* params) {
    if (tokens <= 0) return 0;
    if (head_dim > 256) return VELOX_ERR_UNSUPPORTED;
    switch (format) {
        case VELOX_KV_F32:
            return kv_store_launch<VELOX_KV_F32>(k, v, k_cache, v_cache, tokens, kv_heads, head_dim, pos0, norm, params);
        case VELOX_KV_BF16:
            return kv_store_launch<VELOX_KV_BF16>(k, v, k_cache, v_cache, tokens, kv_heads, head_dim, pos0, norm, params);
        case VELOX_KV_FP8:
            return kv_store_launch<VELOX_KV_FP8>(k, v, k_cache, v_cache, tokens, kv_heads, head_dim, pos0, norm, params);
        case VELOX_KV_NVFP4:
            return kv_store_launch<VELOX_KV_NVFP4>(k, v, k_cache, v_cache, tokens, kv_heads, head_dim, pos0, norm, params);
        default: return VELOX_ERR_UNSUPPORTED;
    }
}

// params: null, or device [pos0, k_cache, v_cache] staged by a graph step, with kv_grid >= pos0 + tokens
// sizing the grid.
VELOX_API int velox_portable_attention_kv(const float* q, const unsigned char* k, const unsigned char* v, float* out,
                                          int tokens, int heads, int kv_heads, int head_dim, int pos0, float scale,
                                          int format, float norm, const long long* params, int kv_grid) {
    if (tokens > 65535 || head_dim > 256) return VELOX_ERR_UNSUPPORTED;
    switch (format) {
        case VELOX_KV_F32:
            return kvq_dispatch<VELOX_KV_F32>(q, k, v, out, tokens, heads, kv_heads, head_dim, pos0, scale, norm, params, kv_grid);
        case VELOX_KV_BF16:
            return kvq_dispatch<VELOX_KV_BF16>(q, k, v, out, tokens, heads, kv_heads, head_dim, pos0, scale, norm, params, kv_grid);
        case VELOX_KV_FP8:
            return kvq_dispatch<VELOX_KV_FP8>(q, k, v, out, tokens, heads, kv_heads, head_dim, pos0, scale, norm, params, kv_grid);
        case VELOX_KV_NVFP4:
            return kvq_dispatch<VELOX_KV_NVFP4>(q, k, v, out, tokens, heads, kv_heads, head_dim, pos0, scale, norm, params, kv_grid);
        default: return VELOX_ERR_UNSUPPORTED;
    }
}

// ---- kv_load ----

namespace {
// One warp per (row, kv head): the represented row (rotated formats rotate back; the normalized
// Hadamard transform is its own inverse).
template <int FMT, int PER>
__global__ void __launch_bounds__(kStoreWarps* VELOX_WARP)
    kv_load_kernel(const unsigned char* __restrict__ kc, const unsigned char* __restrict__ vc, float* __restrict__ k,
                   float* __restrict__ v, int units, int kv_heads, int head_dim, long long pos0, float norm) {
    const int warp = threadIdx.x / VELOX_WARP, lane = threadIdx.x % VELOX_WARP;
    const int unit = blockIdx.x * kStoreWarps + warp;  // token * kv_heads + kv_head
    if (unit >= units) return;                         // warp-uniform
    const size_t rb = static_cast<size_t>(kv_row_bytes(FMT, head_dim));
    const size_t row = static_cast<size_t>(pos0) * kv_heads + unit;
#pragma unroll
    for (int side = 0; side < 2; ++side) {
        float x[PER];
        kv_load_lane<FMT, PER>((side ? vc : kc) + row * rb, head_dim, lane, x);
        if (side ? KvRot<FMT>::v : KvRot<FMT>::k) kv_hadamard<PER>(x, lane, norm);
        float* dst = (side ? v : k) + static_cast<size_t>(unit) * head_dim;
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            const int i = lane + p * VELOX_WARP;
            if (i < head_dim) dst[i] = x[p];
        }
    }
}

template <int FMT>
int kv_load_launch(const unsigned char* kc, const unsigned char* vc, float* k, float* v, int tokens, int kv_heads,
                   int head_dim, long long pos0, float norm) {
    const int units = tokens * kv_heads;
    const unsigned grid = static_cast<unsigned>((units + kStoreWarps - 1) / kStoreWarps);
    const dim3 block(kStoreWarps * VELOX_WARP);
    if (head_dim <= 32)
        kv_load_kernel<FMT, 1><<<grid, block>>>(kc, vc, k, v, units, kv_heads, head_dim, pos0, norm);
    else if (head_dim <= 64)
        kv_load_kernel<FMT, 2><<<grid, block>>>(kc, vc, k, v, units, kv_heads, head_dim, pos0, norm);
    else if (head_dim <= 128)
        kv_load_kernel<FMT, 4><<<grid, block>>>(kc, vc, k, v, units, kv_heads, head_dim, pos0, norm);
    else
        kv_load_kernel<FMT, 8><<<grid, block>>>(kc, vc, k, v, units, kv_heads, head_dim, pos0, norm);
    return velox_launch_status();
}
}  // namespace

VELOX_API int velox_portable_kv_load(const unsigned char* k_cache, const unsigned char* v_cache, float* k, float* v,
                                     int tokens, int kv_heads, int head_dim, long long pos0, int format, float norm) {
    if (tokens <= 0) return 0;
    if (head_dim > 256) return VELOX_ERR_UNSUPPORTED;
    switch (format) {
        case VELOX_KV_F32:
            return kv_load_launch<VELOX_KV_F32>(k_cache, v_cache, k, v, tokens, kv_heads, head_dim, pos0, norm);
        case VELOX_KV_BF16:
            return kv_load_launch<VELOX_KV_BF16>(k_cache, v_cache, k, v, tokens, kv_heads, head_dim, pos0, norm);
        case VELOX_KV_FP8:
            return kv_load_launch<VELOX_KV_FP8>(k_cache, v_cache, k, v, tokens, kv_heads, head_dim, pos0, norm);
        case VELOX_KV_NVFP4:
            return kv_load_launch<VELOX_KV_NVFP4>(k_cache, v_cache, k, v, tokens, kv_heads, head_dim, pos0, norm);
        default: return VELOX_ERR_UNSUPPORTED;
    }
}

// ---- attention_sparse ----

namespace {
// One warp per (token, head) over the cache positions its selection lists (negative entries skipped).
template <int FMT, int PER>
__global__ void kvq_sparse_kernel(const float* __restrict__ q, const unsigned char* __restrict__ k,
                                  const unsigned char* __restrict__ v, const float* __restrict__ sel, int sel_len,
                                  float* __restrict__ out, int heads, int kv_heads, int head_dim, float scale,
                                  float norm) {
    const int h = blockIdx.x, t = blockIdx.y, lane = threadIdx.x;
    const int kvh = h / (heads / kv_heads);
    const size_t q_off = (static_cast<size_t>(t) * heads + h) * head_dim;
    const size_t rb = static_cast<size_t>(kv_row_bytes(FMT, head_dim));
    float qr[PER], acc[PER];
    kvq_load_query<FMT, PER>(q + q_off, true, head_dim, lane, scale, norm, qr);
#pragma unroll
    for (int p = 0; p < PER; ++p) acc[p] = 0.0f;
    float m = -INFINITY, l = 0.0f;
    const float* list = sel + static_cast<size_t>(t) * sel_len;
    for (int i = 0; i < sel_len; ++i) {
        const int j = static_cast<int>(__float_as_uint(list[i]));
        if (j < 0) continue;
        const size_t row = (static_cast<size_t>(j) * kv_heads + kvh) * rb;
        float kx[PER], vx[PER];
        kv_load_lane<FMT, PER>(k + row, head_dim, lane, kx);
        kv_load_lane<FMT, PER>(v + row, head_dim, lane, vx);
        float s = 0.0f;
#pragma unroll
        for (int p = 0; p < PER; ++p) s += qr[p] * kx[p];
        s = velox_warp_sum(s);
        const float m_new = fmaxf(m, s);
        const float corr = expf(m - m_new), pj = expf(s - m_new);
        l = l * corr + pj;
#pragma unroll
        for (int p = 0; p < PER; ++p) acc[p] = acc[p] * corr + pj * vx[p];
        m = m_new;
    }
#pragma unroll
    for (int p = 0; p < PER; ++p) acc[p] = l > 0.0f ? acc[p] / l : 0.0f;
    if (KvRot<FMT>::v) kv_hadamard<PER>(acc, lane, norm);
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int d = lane + p * VELOX_WARP;
        if (d < head_dim) out[q_off + d] = acc[p];
    }
}

template <int FMT>
int kvq_sparse(const float* q, const unsigned char* k, const unsigned char* v, const float* sel, int sel_len,
               float* out, int tokens, int heads, int kv_heads, int head_dim, float scale, float norm) {
    const dim3 grid(heads, tokens);
    if (head_dim <= 32)
        kvq_sparse_kernel<FMT, 1><<<grid, VELOX_WARP>>>(q, k, v, sel, sel_len, out, heads, kv_heads, head_dim, scale, norm);
    else if (head_dim <= 64)
        kvq_sparse_kernel<FMT, 2><<<grid, VELOX_WARP>>>(q, k, v, sel, sel_len, out, heads, kv_heads, head_dim, scale, norm);
    else if (head_dim <= 128)
        kvq_sparse_kernel<FMT, 4><<<grid, VELOX_WARP>>>(q, k, v, sel, sel_len, out, heads, kv_heads, head_dim, scale, norm);
    else
        kvq_sparse_kernel<FMT, 8><<<grid, VELOX_WARP>>>(q, k, v, sel, sel_len, out, heads, kv_heads, head_dim, scale, norm);
    return velox_launch_status();
}
}  // namespace

VELOX_API int velox_portable_attention_sparse(const float* q, const unsigned char* k, const unsigned char* v,
                                              const float* sel, int sel_len, float* out, int tokens, int heads,
                                              int kv_heads, int head_dim, float scale, int format, float norm) {
    if (tokens > 65535 || head_dim > 256) return VELOX_ERR_UNSUPPORTED;
    switch (format) {
        case VELOX_KV_F32:
            return kvq_sparse<VELOX_KV_F32>(q, k, v, sel, sel_len, out, tokens, heads, kv_heads, head_dim, scale, norm);
        case VELOX_KV_BF16:
            return kvq_sparse<VELOX_KV_BF16>(q, k, v, sel, sel_len, out, tokens, heads, kv_heads, head_dim, scale, norm);
        case VELOX_KV_FP8:
            return kvq_sparse<VELOX_KV_FP8>(q, k, v, sel, sel_len, out, tokens, heads, kv_heads, head_dim, scale, norm);
        case VELOX_KV_NVFP4:
            return kvq_sparse<VELOX_KV_NVFP4>(q, k, v, sel, sel_len, out, tokens, heads, kv_heads, head_dim, scale, norm);
        default: return VELOX_ERR_UNSUPPORTED;
    }
}

// ---- recurrent-state checkpoints: `state_pack` / `state_unpack` ----
// Formats and their exact encoding are specified in velox-core/src/state.rs; these are bit-exact
// with `state::pack` / `state::unpack`. Either side may be mapped host memory (a context-store
// checkpoint): the kernels then read or write system RAM over PCIe directly. n is a multiple of
// 128 (one row of an S[h] matrix); each thread handles 4 consecutive values.

#define VELOX_STATE_F32 0
#define VELOX_STATE_BF16 1
#define VELOX_STATE_NVFP4 2

namespace {

__device__ __forceinline__ long long state_tid() { return blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; }
__device__ __forceinline__ long long state_stride() { return static_cast<long long>(gridDim.x) * blockDim.x; }

__global__ void state_copy_kernel(const float4* __restrict__ src, float4* __restrict__ dst, long long quads) {
    for (long long q = state_tid(); q < quads; q += state_stride()) dst[q] = src[q];
}

__global__ void state_pack_bf16_kernel(const float4* __restrict__ src, uint2* __restrict__ dst, long long quads) {
    for (long long q = state_tid(); q < quads; q += state_stride()) {
        const float4 x = src[q];
        dst[q] = make_uint2(static_cast<unsigned>(kv_f32_to_bf16(x.x)) | static_cast<unsigned>(kv_f32_to_bf16(x.y)) << 16,
                            static_cast<unsigned>(kv_f32_to_bf16(x.z)) | static_cast<unsigned>(kv_f32_to_bf16(x.w)) << 16);
    }
}

__global__ void state_unpack_bf16_kernel(const uint2* __restrict__ src, float4* __restrict__ dst, long long quads) {
    for (long long q = state_tid(); q < quads; q += state_stride()) {
        const uint2 w = src[q];
        dst[q] = make_float4(kv_bf16_to_f32(w.x & 0xffffu), kv_bf16_to_f32(w.x >> 16), kv_bf16_to_f32(w.y & 0xffffu),
                             kv_bf16_to_f32(w.y >> 16));
    }
}

// One warp per row of 128: lane l holds values 4l .. 4l + 3, so a group of 16 is 4 lanes.
__global__ void state_pack_nvfp4_kernel(const float* __restrict__ src, unsigned char* __restrict__ dst, long long n) {
    const long long rows = n / 128;
    unsigned short* codes = reinterpret_cast<unsigned short*>(dst);
    unsigned char* scales = dst + n / 2;
    float* divs = reinterpret_cast<float*>(dst + n / 2 + n / 16);
    const int lane = threadIdx.x % VELOX_WARP;
    for (long long r = state_tid() / VELOX_WARP; r < rows; r += state_stride() / VELOX_WARP) {
        const float4 x = reinterpret_cast<const float4*>(src + r * 128)[lane];
        const float a = fmaxf(fmaxf(fabsf(x.x), fabsf(x.y)), fmaxf(fabsf(x.z), fabsf(x.w)));
        float ga = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 1));
        ga = fmaxf(ga, __shfl_xor_sync(0xffffffffu, ga, 2));
        const float amax = velox_warp_max(a);
        float d = amax > 0.0f ? __fdiv_rn(2688.0f, amax) : 0.0f;
        if (!isfinite(d)) d = 0.0f;
        const unsigned sc = kv_f32_to_e4m3(__fdiv_rn(__fmul_rn(d, ga), 6.0f));
        const float s = kv_e4m3_to_f32(sc);
        const float v[4] = {x.x, x.y, x.z, x.w};
        unsigned packed = 0;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const unsigned c = s == 0.0f ? 0u : kv_f32_to_e2m1(__fdiv_rn(__fmul_rn(v[p], d), s));
            packed |= c << (4 * p);
        }
        codes[r * 32 + lane] = static_cast<unsigned short>(packed);  // 32 code words (64 bytes) per row
        if (lane % 4 == 0) scales[r * 8 + lane / 4] = static_cast<unsigned char>(sc);
        if (lane == 0) divs[r] = d;
    }
}

__global__ void state_unpack_nvfp4_kernel(const unsigned char* __restrict__ src, float4* __restrict__ dst, long long n) {
    const unsigned short* codes = reinterpret_cast<const unsigned short*>(src);
    const unsigned char* scales = src + n / 2;
    const float* divs = reinterpret_cast<const float*>(src + n / 2 + n / 16);
    for (long long q = state_tid(); q < n / 4; q += state_stride()) {
        const float d = divs[q / 32];
        const float s = kv_e4m3_to_f32(scales[q / 4]);
        const unsigned w = codes[q];
        float v[4];
#pragma unroll
        for (int p = 0; p < 4; ++p)
            v[p] = (d == 0.0f || s == 0.0f) ? 0.0f
                                            : __fdiv_rn(__fmul_rn(kv_e2m1_to_f32((w >> (4 * p)) & 15u), s), d);
        dst[q] = make_float4(v[0], v[1], v[2], v[3]);
    }
}
}  // namespace

VELOX_API int velox_portable_state_pack(const float* src, unsigned char* dst, long long n, int format) {
    if (n <= 0) return 0;
    if (n % 128 != 0) return VELOX_ERR_UNSUPPORTED;
    const long long quads = n / 4;
    switch (format) {
        case VELOX_STATE_F32:
            state_copy_kernel<<<velox_grid_1d(quads, 256), 256>>>(reinterpret_cast<const float4*>(src),
                                                                  reinterpret_cast<float4*>(dst), quads);
            break;
        case VELOX_STATE_BF16:
            state_pack_bf16_kernel<<<velox_grid_1d(quads, 256), 256>>>(reinterpret_cast<const float4*>(src),
                                                                       reinterpret_cast<uint2*>(dst), quads);
            break;
        case VELOX_STATE_NVFP4:
            state_pack_nvfp4_kernel<<<velox_grid_1d(n / 128 * VELOX_WARP, 256), 256>>>(src, dst, n);
            break;
        default: return VELOX_ERR_UNSUPPORTED;
    }
    return velox_launch_status();
}

VELOX_API int velox_portable_state_unpack(const unsigned char* src, float* dst, long long n, int format) {
    if (n <= 0) return 0;
    if (n % 128 != 0) return VELOX_ERR_UNSUPPORTED;
    const long long quads = n / 4;
    switch (format) {
        case VELOX_STATE_F32:
            state_copy_kernel<<<velox_grid_1d(quads, 256), 256>>>(reinterpret_cast<const float4*>(src),
                                                                  reinterpret_cast<float4*>(dst), quads);
            break;
        case VELOX_STATE_BF16:
            state_unpack_bf16_kernel<<<velox_grid_1d(quads, 256), 256>>>(reinterpret_cast<const uint2*>(src),
                                                                         reinterpret_cast<float4*>(dst), quads);
            break;
        case VELOX_STATE_NVFP4:
            state_unpack_nvfp4_kernel<<<velox_grid_1d(quads, 256), 256>>>(src, reinterpret_cast<float4*>(dst), n);
            break;
        default: return VELOX_ERR_UNSUPPORTED;
    }
    return velox_launch_status();
}
