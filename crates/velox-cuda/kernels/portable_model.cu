// Portable kernels for transformer/hybrid models: quantized-weight linear and embedding, sigmoid
// gating, causal depthwise convolution and the Gated DeltaNet recurrence. Simple and correct on
// every supported architecture; hardware branches replace them with tuned kernels.
#include <math.h>
#include <cuda_fp16.h>
#include <stdint.h>

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

// Decoded W[row, col] of the parent matrix.
__device__ float weight_at(const WDesc& d, long long row, long long col) {
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
linear_tiled_kernel(const float* __restrict__ x, WDesc w, long long row_begin, int rows, int m, float* __restrict__ y) {
    __shared__ float Xs[kTileK][kTile + 4];
    __shared__ float Ws[kTileK][kTile + 4];
    const int t = threadIdx.x;
    const int m0 = blockIdx.y * kTile, n0 = blockIdx.x * kTile;
    const int lr = t / 4, lk = (t % 4) * 4;
    const int ty = t / 16, tx = t % 16;
    float acc[4][4] = {};
    for (long long k0 = 0; k0 < w.k; k0 += kTileK) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const long long kk = k0 + lk + i;
            const int xm = m0 + lr, wn = n0 + lr;
            Xs[lk + i][lr] = (xm < m && kk < w.k) ? x[static_cast<size_t>(xm) * w.k + kk] : 0.0f;
            Ws[lk + i][lr] = (wn < rows && kk < w.k) ? weight_at(w, row_begin + wn, kk) : 0.0f;
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

__global__ void embed_kernel(const uint32_t* __restrict__ ids, WDesc w, long long row_begin, float* __restrict__ out) {
    const long long row = row_begin + ids[blockIdx.x];
    for (long long c = threadIdx.x; c < w.k; c += blockDim.x) out[blockIdx.x * w.k + c] = weight_at(w, row, c);
}

__global__ void sigmoid_mul_kernel(float* __restrict__ x, const float* __restrict__ gate, long long len) {
    for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < len;
         i += static_cast<long long>(gridDim.x) * blockDim.x)
        x[i] = x[i] / (1.0f + expf(-gate[i]));
}

constexpr int kMaxConvWidth = 8;

// One thread per channel walks the tokens in order, carrying the window.
__global__ void conv1d_silu_kernel(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ state,
                                   float* __restrict__ out, int tokens, int channels, int width) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float win[kMaxConvWidth];
    float wc[kMaxConvWidth];
    for (int j = 0; j < width; ++j) wc[j] = w[static_cast<size_t>(j) * channels + c];
    for (int j = 0; j < width - 1; ++j) win[j] = state[static_cast<size_t>(j) * channels + c];
    for (int t = 0; t < tokens; ++t) {
        win[width - 1] = x[static_cast<size_t>(t) * channels + c];
        float acc = 0.0f;
        for (int j = 0; j < width; ++j) acc += wc[j] * win[j];
        out[static_cast<size_t>(t) * channels + c] = acc / (1.0f + expf(-acc));
        for (int j = 0; j < width - 1; ++j) win[j] = win[j + 1];
    }
    for (int j = 0; j < width - 1; ++j) state[static_cast<size_t>(j) * channels + c] = win[j];
}

// Gated DeltaNet: state rows are independent, so one warp owns one row S[h][i][:] (lanes split
// k_dim, PER elements each, kept in registers for the whole call). Grid: heads x row groups.
template <int PER>
__global__ void gated_delta_net_rows_kernel(const float* __restrict__ qkv, const float* __restrict__ ab,
                                            const float* __restrict__ a_log, const float* __restrict__ dt_bias,
                                            float* __restrict__ state, float* __restrict__ out, int tokens,
                                            int k_heads, int v_heads, int k_dim, int v_dim) {
    const int warps = blockDim.x / VELOX_WARP;
    const int h = blockIdx.x, i = blockIdx.y * warps + threadIdx.x / VELOX_WARP, lane = threadIdx.x % VELOX_WARP;
    if (i >= v_dim) return;
    const int kh = h / (v_heads / k_heads);
    const int width = 2 * k_heads * k_dim + v_heads * v_dim;
    const float inv_sqrt_dk = rsqrtf(static_cast<float>(k_dim));
    const float neg_a = -expf(a_log[h]), bias = dt_bias[h];
    float* row_state = state + (static_cast<size_t>(h) * v_dim + i) * k_dim;
    float s[PER];
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int d = lane + p * VELOX_WARP;
        s[p] = d < k_dim ? row_state[d] : 0.0f;
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
        o = velox_warp_sum(o);
        if (lane == 0) out[(static_cast<size_t>(t) * v_heads + h) * v_dim + i] = o * inv_sqrt_dk;
    }
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int d = lane + p * VELOX_WARP;
        if (d < k_dim) row_state[d] = s[p];
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

template <int FMT, int R, int M>
__global__ void __launch_bounds__(256)
linear_gemv_vec_kernel(const float* __restrict__ x, WDesc w, long long row_begin, int rows, int m,
                       float* __restrict__ y) {
    const int warp = blockIdx.x * (blockDim.x / VELOX_WARP) + threadIdx.x / VELOX_WARP;
    const int lane = threadIdx.x % VELOX_WARP;
    const int r0 = warp * R;
    if (r0 >= rows) return;
    const long long k = w.k;
    const long long row_bytes = FMT == VELOX_FMT_NVFP4 ? k / 2 : k;
    const long long chunks = row_bytes / 16;
    constexpr int kElems = FMT == VELOX_FMT_NVFP4 ? 32 : 16;
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

    for (long long c = lane; c < chunks; c += VELOX_WARP) {
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
                } else {
                    float s = 0.0f;
#pragma unroll
                    for (int b = 0; b < 16; ++b) s += e4m3_bits(words[b / 4], b % 4) * xv[b];
                    acc[r][i] += s;
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (!live[r]) continue;
        const long long row = row_begin + r0 + r;
        float post;
        if (FMT == VELOX_FMT_NVFP4) {
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
    // Prefer more rows per warp (more x reuse) while keeping enough warps to fill the GPU.
    const int r = rows >= 4 * 4096 ? 4 : rows >= 2 * 2048 ? 2 : 1;
    const int warps = (rows + r - 1) / r;
    const dim3 grid((warps + kWarps - 1) / kWarps);
    if (r == 4) linear_gemv_vec_kernel<FMT, 4, M><<<grid, kWarps * VELOX_WARP>>>(x, d, row_begin, rows, m, y);
    else if (r == 2) linear_gemv_vec_kernel<FMT, 2, M><<<grid, kWarps * VELOX_WARP>>>(x, d, row_begin, rows, m, y);
    else linear_gemv_vec_kernel<FMT, 1, M><<<grid, kWarps * VELOX_WARP>>>(x, d, row_begin, rows, m, y);
    return velox_launch_status();
}

template <int FMT>
int launch_gemv_vec_m(const float* x, const WDesc& d, long long row_begin, int rows, int m, float* y) {
    if (m == 1) return launch_gemv_vec<FMT, 1>(x, d, row_begin, rows, m, y);
    if (m == 2) return launch_gemv_vec<FMT, 2>(x, d, row_begin, rows, m, y);
    if (m <= 4) return launch_gemv_vec<FMT, 4>(x, d, row_begin, rows, m, y);
    return launch_gemv_vec<FMT, 8>(x, d, row_begin, rows, m, y);
}

}  // namespace

VELOX_API int velox_portable_linear(const float* x, const uint8_t* w, int format, long long n, long long k,
                                    long long row_begin, int rows, int m, float* y) {
    const WDesc d{w, format, n, k};
    // x rows are read as float4, so k must keep them 16-byte aligned too.
    if (m <= kGemvMaxM && k % 16 == 0 && reinterpret_cast<uintptr_t>(x) % 16 == 0) {
        if (format == VELOX_FMT_NVFP4) return launch_gemv_vec_m<VELOX_FMT_NVFP4>(x, d, row_begin, rows, m, y);
        if (format == VELOX_FMT_FP8_ROW_BF16)
            return launch_gemv_vec_m<VELOX_FMT_FP8_ROW_BF16>(x, d, row_begin, rows, m, y);
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
    if (m >= 16 && (m + kTile - 1) / kTile <= 65535) {
        const dim3 grid((rows + kTile - 1) / kTile, (m + kTile - 1) / kTile);
        linear_tiled_kernel<<<grid, 256>>>(x, d, row_begin, rows, m, y);
    } else {
        constexpr int kWarps = 8;
        const dim3 grid((rows + kWarps - 1) / kWarps, m < 65535 ? m : 65535);
        linear_kernel<<<grid, kWarps * VELOX_WARP>>>(x, d, row_begin, rows, m, y);
    }
    return velox_launch_status();
}

VELOX_API int velox_portable_embed(const uint32_t* ids, int count, const uint8_t* w, int format, long long n,
                                   long long k, long long row_begin, float* out) {
    embed_kernel<<<count, 256>>>(ids, WDesc{w, format, n, k}, row_begin, out);
    return velox_launch_status();
}

VELOX_API int velox_portable_sigmoid_mul(float* x, const float* gate, long long len) {
    sigmoid_mul_kernel<<<velox_grid_1d(len, 256), 256>>>(x, gate, len);
    return velox_launch_status();
}

VELOX_API int velox_portable_conv1d_silu(const float* x, const float* w, float* state, float* out, int tokens,
                                         int channels, int width) {
    if (width > kMaxConvWidth) return VELOX_ERR_UNSUPPORTED;
    conv1d_silu_kernel<<<(channels + 127) / 128, 128>>>(x, w, state, out, tokens, channels, width);
    return velox_launch_status();
}

VELOX_API int velox_portable_gated_delta_net(const float* qkv, const float* ab, const float* a_log,
                                             const float* dt_bias, float* state, float* out, int tokens, int k_heads,
                                             int v_heads, int k_dim, int v_dim) {
    constexpr int kWarps = 4;
    const dim3 grid(v_heads, (v_dim + kWarps - 1) / kWarps), block(kWarps * VELOX_WARP);
#define VELOX_GDN(PER)                                                                                     \
    gated_delta_net_rows_kernel<PER><<<grid, block>>>(qkv, ab, a_log, dt_bias, state, out, tokens, k_heads, \
                                                      v_heads, k_dim, v_dim)
    if (k_dim <= 32) VELOX_GDN(1);
    else if (k_dim <= 64) VELOX_GDN(2);
    else if (k_dim <= 128) VELOX_GDN(4);
    else if (k_dim <= 256) VELOX_GDN(8);
    else return VELOX_ERR_UNSUPPORTED;
#undef VELOX_GDN
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
__global__ void rope_positions_kernel(float* __restrict__ x, const uint32_t* __restrict__ pos, long long pairs_total,
                                      int tokens, int heads, int head_dim, int rotary_dim, float theta, int axes,
                                      int layout) {
    const int half = rotary_dim / 2;
    for (long long p = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; p < pairs_total;
         p += static_cast<long long>(gridDim.x) * blockDim.x) {
        const int i = static_cast<int>(p % half);
        const long long row = p / half;  // token * heads + head
        const int t = static_cast<int>(row / heads);
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
        float* base = x + row * head_dim;
        const float a = base[i], b = base[i + half];
        base[i] = static_cast<float>(a * c - b * s);
        base[i + half] = static_cast<float>(a * s + b * c);
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
    const long long pairs = static_cast<long long>(tokens) * heads * (rotary_dim / 2);
    rope_positions_kernel<<<velox_grid_1d(pairs, 256), 256>>>(x, pos, pairs, tokens, heads, head_dim, rotary_dim, theta,
                                                              axes, layout);
    return velox_launch_status();
}
