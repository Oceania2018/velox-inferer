// RTX 5060 Ti linear for bf16 weights with few output rows (the GDN gate projections a/b, 2 x 48
// rows) at prefill widths, on bf16 tensor cores (ActQuant::Bf16).
//
// Tokens are the A operand of mma.m16n8k16 (16 per block) and weight rows the B operand (up to 128
// per block, 16 n-tiles). Fragments come straight from global memory: K is permuted inside every
// 32-value step so a lane's values are contiguous (MMA k-indices 2t, 2t+1, 2t+8, 2t+9 of half h
// are values 8t + 4h .. 8t + 4h + 3; the dot product is order independent), which makes a lane's
// share one 16-byte weight load and two float4 activation loads per row, rounded to bf16 (RNE) in
// registers. Warps split K and reduce in a fixed order, so results are deterministic.
#include <cuda_bf16.h>
#include <stdint.h>

#include "velox_common.cuh"

namespace {

constexpr int kW = 8;         // warps per block, splitting K
constexpr int kTokens = 16;   // tokens per block (one MMA tile)
constexpr int kRows = 128;    // weight rows per block (16 n-tiles)

__device__ __forceinline__ void mma_bf16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t bf16x2(float lo, float hi) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
}

// NT: n-tiles of 8 rows this block computes (rows of the last block may be fewer; they are
// skipped warp-uniformly).
template <int NT>
__global__ void __launch_bounds__(kW * 32) bf16_skinny_kernel(const float* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                                                              int m, int k, int rows, float* __restrict__ y,
                                                              long long ldy) {
    __shared__ float sum[kTokens][kRows];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int n0 = blockIdx.x * kRows, t0 = blockIdx.y * kTokens;
    const int tiles = min(NT, (rows - n0 + 7) / 8);
    // This lane's token rows g and g + 8 (zero past m) and weight rows n0 + 8 nt + g.
    const bool ok0 = t0 + g < m, ok1 = t0 + g + 8 < m;
    const float* x0 = x + static_cast<long long>(ok0 ? t0 + g : 0) * k + 8 * t;
    const float* x1 = x + static_cast<long long>(ok1 ? t0 + g + 8 : 0) * k + 8 * t;
    const __nv_bfloat16* wr[NT];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        const int r = n0 + 8 * nt + g;
        wr[nt] = w + static_cast<long long>(r < rows ? r : 0) * k + 8 * t;
    }
    float acc[NT][4];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;

    const int steps = k / 32;
#pragma unroll 2
    for (int s = warp; s < steps; s += kW) {
        const int k0 = s * 32;
        float4 a[2][2];
        const float4 z = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        a[0][0] = ok0 ? __ldg(reinterpret_cast<const float4*>(x0 + k0)) : z;
        a[0][1] = ok0 ? __ldg(reinterpret_cast<const float4*>(x0 + k0) + 1) : z;
        a[1][0] = ok1 ? __ldg(reinterpret_cast<const float4*>(x1 + k0)) : z;
        a[1][1] = ok1 ? __ldg(reinterpret_cast<const float4*>(x1 + k0) + 1) : z;
        uint4 b[NT];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
            if (nt < tiles) b[nt] = __ldg(reinterpret_cast<const uint4*>(wr[nt] + k0));
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const float4 r0 = a[0][h], r1 = a[1][h];
            const uint32_t a0 = bf16x2(r0.x, r0.y), a1 = bf16x2(r1.x, r1.y);
            const uint32_t a2 = bf16x2(r0.z, r0.w), a3 = bf16x2(r1.z, r1.w);
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
                if (nt < tiles)
                    mma_bf16(acc[nt], a0, a1, a2, a3, h ? b[nt].z : b[nt].x, h ? b[nt].w : b[nt].y);
        }
    }

    // Fixed-order cross-warp sum: c0, c1 = token g, rows 2t, 2t + 1 of the tile; c2, c3 = token g + 8.
    for (int wi = 0; wi < kW; ++wi) {
        if (warp == wi) {
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                if (nt >= tiles) continue;
                const int c = 8 * nt + 2 * t;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float& d = sum[g + 8 * (j >> 1)][c + (j & 1)];
                    d = wi == 0 ? acc[nt][j] : d + acc[nt][j];
                }
            }
        }
        __syncthreads();
    }
    const int n_here = min(kRows, rows - n0);
    for (int i = threadIdx.x; i < kTokens * n_here; i += kW * 32) {
        const int tt = i / n_here, r = i % n_here;
        if (t0 + tt < m) y[static_cast<long long>(t0 + tt) * ldy + n0 + r] = sum[tt][r];
    }
}

}  // namespace

// y[t * ldy + r] = bf16(x) . W[row_begin + r, :] for a contiguous bf16 weight [parent_rows, k],
// r < rows <= ldy; k % 32 == 0, x and the weight 16-byte aligned. VELOX_ERR_UNSUPPORTED otherwise.
VELOX_API int velox_5060_linear_bf16_skinny(const float* x, int m, const uint8_t* w, long long k, long long row_begin,
                                            int rows, float* y, long long ldy) {
    if (m <= 0 || rows <= 0 || ldy < rows || k % 32 != 0 || k > (1ll << 30) || reinterpret_cast<uintptr_t>(x) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(w) % 16 != 0)
        return VELOX_ERR_UNSUPPORTED;
    const auto* wb = reinterpret_cast<const __nv_bfloat16*>(w) + row_begin * k;
    const dim3 grid((rows + kRows - 1) / kRows, (m + kTokens - 1) / kTokens);
    bf16_skinny_kernel<kRows / 8><<<grid, kW * 32>>>(x, wb, m, static_cast<int>(k), rows, y, ldy);
    return velox_launch_status();
}
