// RTX 5060 Ti register-streaming GEMV (1..8 tokens, exact f32 activations) for q5_g64 weights
// (row_split_k128_v1). On this card single-token decode is bound by keeping enough loads in flight,
// and warps that stream weights straight into registers (no shared-memory staging, no block-wide
// synchronisation) reach more of the 426 GB/s than staged tensor-core tiles. Each lane streams 16
// base bytes (32 values, half a group), their 4 high-plane bytes and the group's FP16 scale.
#include <cuda_fp16.h>
#include <stdint.h>

#include "velox_common.cuh"

namespace {


constexpr int kWarps = 8;
// Rows per warp, measured: 2 (shares x loads; 1: 170 us, 4: 118 us vs 112 us on [12288,5120]).
#define GEMV_Q5_R 2

__device__ __forceinline__ uint4 ld16(const void* p) { return __ldg(reinterpret_cast<const uint4*>(p)); }
__device__ __forceinline__ uint2 ld8(const void* p) { return __ldg(reinterpret_cast<const uint2*>(p)); }

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
int launch_q5(const float* x, const uint8_t* w, long long n, long long k, long long rb, int rows, int m, float* y) {
    constexpr int R = GEMV_Q5_R;
    const int warps = (rows + R - 1) / R;
    gemv_q5_kernel<R, M><<<(warps + kWarps - 1) / kWarps, kWarps * 32>>>(x, w, n, k, rb, rows, m, y);
    return velox_launch_status();
}

}  // namespace

// y[m, rows] = x . W[row_begin.., :]^T with exact f32 activations for 1 <= m <= 8 and format 5
// (q5_g64 row_split_k128_v1, k % 32 == 0). x must be 16-byte aligned. Returns
// VELOX_ERR_UNSUPPORTED for anything else.
VELOX_API int velox_5060_gemv(const float* x, int m, const uint8_t* w, long long parent_rows, long long k,
                              long long row_begin, int rows, int format, float* y) {
    if (m < 1 || m > 8 || rows <= 0 || reinterpret_cast<uintptr_t>(x) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(w) % 16 != 0)
        return VELOX_ERR_UNSUPPORTED;
    if (format == 5 && k % 32 == 0) {
        if (m == 1) return launch_q5<1>(x, w, parent_rows, k, row_begin, rows, m, y);
        if (m == 2) return launch_q5<2>(x, w, parent_rows, k, row_begin, rows, m, y);
        if (m <= 4) return launch_q5<4>(x, w, parent_rows, k, row_begin, rows, m, y);
        return launch_q5<8>(x, w, parent_rows, k, row_begin, rows, m, y);
    }
    return VELOX_ERR_UNSUPPORTED;
}
