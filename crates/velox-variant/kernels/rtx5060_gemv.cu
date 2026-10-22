// RTX 5060 Ti register-streaming GEMV (1..M tokens, exact f32 activations) for IQ3_XXS and
// q5_g64 weights. On this card single-token decode is bound by keeping enough loads in flight, and
// warps that stream weights straight into registers (no shared-memory staging, no block-wide
// synchronisation) reach more of the 426 GB/s than staged tensor-core tiles.
//
// IQ3_XXS (iq3_xxs_planar_v1): the four lanes q of a quad share one 256-value block. Lane q loads
// index bytes [16q, 16q+16) (grid entries 2q, 2q+1 of all eight sub-blocks) and lane word q (the
// sign indices of values 8q..8q+7 of every sub-block), so it owns values s*32 + 8q .. +7. The 4-bit
// sub-block scales sit in the top byte of each lane word and are exchanged within the quad. A warp
// walks (row, block) pairs of its R rows, eight blocks per step.
//
// q5_g64 (row_split_k128_v1): each lane streams 16 base bytes (32 values, half a group), their 4
// high-plane bytes and the group's FP16 scale.
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

constexpr int kWarps = 8;
// Rows per warp, measured: IQ3 1 (decode-bound, fewer registers), q5 2 (shares x loads).
#define GEMV_IQ3_R 1
#define GEMV_Q5_R 2

__device__ __forceinline__ uint4 ld16(const void* p) { return __ldg(reinterpret_cast<const uint4*>(p)); }
__device__ __forceinline__ uint2 ld8(const void* p) { return __ldg(reinterpret_cast<const uint2*>(p)); }

template <int R, int M>
__global__ void __launch_bounds__(kWarps * 32)
gemv_iq3_kernel(const float* __restrict__ x, const uint8_t* __restrict__ w, long long parent_rows, long long k,
                long long row_begin, int rows, int m, float* __restrict__ y) {
    __shared__ uint32_t grid[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) grid[i] = kGrid[i];
    __syncthreads();
    const int lane = threadIdx.x & 31, q = lane & 3, quad = lane >> 2;
    const int r0 = (blockIdx.x * kWarps + (threadIdx.x >> 5)) * R;
    if (r0 >= rows) return;
    const long long blocks = k / 256;
    const uint8_t* codes = w;
    const __half* dplane = reinterpret_cast<const __half*>(w + (parent_rows * blocks * 96 + 255) / 256 * 256);
    float acc[R][M];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int i = 0; i < M; ++i) acc[r][i] = 0.0f;

    const long long pairs = R * blocks;
    // Warp-uniform trip count: every lane takes part in the shuffles; quads past the end skip work.
    for (long long it0 = 0; it0 < pairs; it0 += 8) {
        const long long it = it0 + quad;
        const bool live = it < pairs;
        const int r = live ? static_cast<int>(it / blocks) : 0;
        const long long b = live ? it % blocks : 0;
        const bool row_ok = live && r0 + r < rows;
        const long long row = row_begin + r0 + (row_ok ? r : 0);
        const uint8_t* blk = codes + (row * blocks + b) * 96;
        const uint4 idx = ld16(blk + 16 * q);
        const uint2 sw = ld8(blk + 64 + 8 * q);
        const float d = __half2float(dplane[row * blocks + b]);
        // Top bytes of the quad's four lane words: ls of sub-blocks 2j (low nibble), 2j+1.
        const uint32_t top = sw.y >> 24;
        uint32_t ls4 = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) ls4 |= (__shfl_sync(0xffffffffu, top, (lane & ~3) + j) & 0xffu) << (8 * j);
        if (!row_ok) continue;
        const uint64_t sign_word = (static_cast<uint64_t>(sw.y) << 32) | sw.x;
        const uint32_t ib[4] = {idx.x, idx.y, idx.z, idx.w};
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            // Index bytes 2s, 2s+1 of this lane's 16: entries 2q, 2q+1 of sub-block s.
            const uint32_t pair = (ib[s >> 1] >> (16 * (s & 1))) & 0xffffu;
            const uint32_t g0 = grid[pair & 0xff], g1 = grid[pair >> 8];
            const uint32_t s7 = static_cast<uint32_t>(sign_word >> (7 * s)) & 127u;
            const uint32_t s8 = s7 | ((__popc(s7) & 1u) << 7);
            float wv[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t mag = ((j < 4 ? g0 : g1) >> (8 * (j & 3))) & 0xffu;
                wv[j] = __uint_as_float(__float_as_uint(static_cast<float>(mag)) | (((s8 >> j) & 1u) << 31));
            }
            const float sc = d * (0.25f + 0.5f * static_cast<float>((ls4 >> (4 * s)) & 15u));
            const long long col = b * 256 + s * 32 + 8 * q;
#pragma unroll
            for (int i = 0; i < M; ++i) {
                if (i >= m) break;
                const float4* xp = reinterpret_cast<const float4*>(x + i * k + col);
                const float4 a = __ldg(xp), c = __ldg(xp + 1);
                const float p = wv[0] * a.x + wv[1] * a.y + wv[2] * a.z + wv[3] * a.w + wv[4] * c.x + wv[5] * c.y +
                                wv[6] * c.z + wv[7] * c.w;
                acc[r][i] += p * sc;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int i = 0; i < M; ++i) {
            const float s = velox_warp_sum(acc[r][i]);
            if (lane == 0 && i < m && r0 + r < rows) y[static_cast<long long>(i) * rows + r0 + r] = s;
        }
}

template <int R, int M>
__global__ void __launch_bounds__(kWarps * 32)
gemv_q5_kernel(const float* __restrict__ x, const uint8_t* __restrict__ w, long long parent_rows, long long k,
               long long row_begin, int rows, int m, float* __restrict__ y) {
    const int lane = threadIdx.x & 31;
    const int r0 = (blockIdx.x * kWarps + (threadIdx.x >> 5)) * R;
    if (r0 >= rows) return;
    const long long gpr = (k + 127) / 128 * 128 / 64, groups = parent_rows * gpr;
    const long long high_off = (groups * 32 + 255) / 256 * 256;
    const long long scale_off = high_off + (groups * 8 + 255) / 256 * 256;
    const __half* scales = reinterpret_cast<const __half*>(w + scale_off);
    float acc[R][M];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int i = 0; i < M; ++i) acc[r][i] = 0.0f;
    // Chunk c: 32 values (16 base bytes) of half-group c; only real columns (k % 32 == 0).
    const long long chunks = k / 32;
    for (long long c = lane; c < chunks; c += 32) {
        uint4 base[R];
        uint32_t high[R];
        float sc[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const long long row = row_begin + r0 + (r0 + r < rows ? r : 0);
            const long long gi = row * gpr + (c >> 1);
            base[r] = ld16(w + row * gpr * 32 + c * 16);
            high[r] = __ldg(reinterpret_cast<const uint32_t*>(w + high_off + gi * 8 + (c & 1) * 4));
            sc[r] = __half2float(scales[gi]);
        }
#pragma unroll
        for (int i = 0; i < M; ++i) {
            if (i >= m) break;
            const float4* xp = reinterpret_cast<const float4*>(x + i * k + c * 32);
            float xv[32];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float4 t = __ldg(xp + j);
                xv[4 * j] = t.x, xv[4 * j + 1] = t.y, xv[4 * j + 2] = t.z, xv[4 * j + 3] = t.w;
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint32_t words[4] = {base[r].x, base[r].y, base[r].z, base[r].w};
                float s = 0.0f;
#pragma unroll
                for (int v = 0; v < 32; ++v) {
                    const uint32_t nib = (words[v >> 3] >> (4 * (v & 7))) & 0xfu;
                    const uint32_t u = nib | (((high[r] >> v) & 1u) << 4);
                    s += static_cast<float>(static_cast<int>(u) - (u & 16u ? 32 : 0)) * xv[v];
                }
                acc[r][i] += s * sc[r];
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int i = 0; i < M; ++i) {
            const float s = velox_warp_sum(acc[r][i]);
            if (lane == 0 && i < m && r0 + r < rows) y[static_cast<long long>(i) * rows + r0 + r] = s;
        }
}

template <int M>
int launch_iq3(const float* x, const uint8_t* w, long long n, long long k, long long rb, int rows, int m, float* y) {
    constexpr int R = GEMV_IQ3_R;
    const int warps = (rows + R - 1) / R;
    gemv_iq3_kernel<R, M><<<(warps + kWarps - 1) / kWarps, kWarps * 32>>>(x, w, n, k, rb, rows, m, y);
    return velox_launch_status();
}

template <int M>
int launch_q5(const float* x, const uint8_t* w, long long n, long long k, long long rb, int rows, int m, float* y) {
    constexpr int R = GEMV_Q5_R;
    const int warps = (rows + R - 1) / R;
    gemv_q5_kernel<R, M><<<(warps + kWarps - 1) / kWarps, kWarps * 32>>>(x, w, n, k, rb, rows, m, y);
    return velox_launch_status();
}

}  // namespace

// y[m, rows] = x . W[row_begin.., :]^T with exact f32 activations for 1 <= m <= 8. format 3:
// iq3_xxs_planar_v1 (k % 256 == 0); 5: q5_g64 row_split_k128_v1 (k % 32 == 0). x must be 16-byte
// aligned. Returns VELOX_ERR_UNSUPPORTED for anything else.
VELOX_API int velox_5060_gemv(const float* x, int m, const uint8_t* w, long long parent_rows, long long k,
                              long long row_begin, int rows, int format, float* y) {
    if (m < 1 || m > 8 || rows <= 0 || reinterpret_cast<uintptr_t>(x) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(w) % 16 != 0)
        return VELOX_ERR_UNSUPPORTED;
    if (format == 3 && k % 256 == 0) {
        if (m == 1) return launch_iq3<1>(x, w, parent_rows, k, row_begin, rows, m, y);
        if (m == 2) return launch_iq3<2>(x, w, parent_rows, k, row_begin, rows, m, y);
        if (m <= 4) return launch_iq3<4>(x, w, parent_rows, k, row_begin, rows, m, y);
        return launch_iq3<8>(x, w, parent_rows, k, row_begin, rows, m, y);
    }
    if (format == 5 && k % 32 == 0) {
        if (m == 1) return launch_q5<1>(x, w, parent_rows, k, row_begin, rows, m, y);
        if (m == 2) return launch_q5<2>(x, w, parent_rows, k, row_begin, rows, m, y);
        if (m <= 4) return launch_q5<4>(x, w, parent_rows, k, row_begin, rows, m, y);
        return launch_q5<8>(x, w, parent_rows, k, row_begin, rows, m, y);
    }
    return VELOX_ERR_UNSUPPORTED;
}
