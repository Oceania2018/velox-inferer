// RTX 5060 Ti bf16 linear for prefill widths: y[m, rows] = bf16(x) . W^T with W stored bf16, on
// mma.m16n8k16 (bf16 inputs, f32 accumulation). Activations are rounded to bf16 (RNE) as the tile
// is staged, which every activation policy permits (ActQuant::Bf16). Qwen3.8-Flash-Next's
// hyper-connection, PLE and router projections are bf16 [320, 10240], [10240, 320], [10240, 2560].
//
// Block: 128 tokens x 64 weight rows, 8 warps (4 along tokens x 2 along rows, 32 x 32 each), K in
// steps of 32 double-buffered: weights by cp.async, activations converted through registers. A K
// that is a multiple of 16 but not 32 (Qwen3-VL's ViT fc2, K = 4304) zero-fills the last step's
// upper half, which adds exact zeros.
#include <cuda_bf16.h>
#include <stdint.h>

#include "velox_common.cuh"

namespace {

constexpr int kBM = 128, kBN = 64, kBK = 32, kPad = 8, kThreads = 256;

__device__ __forceinline__ void cp16(void* smem, const void* gmem) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(a), "l"(gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_wait0() { asm volatile("cp.async.wait_group 0;\n" ::); }

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}

__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

struct Smem {
    __nv_bfloat16 a[2][kBM][kBK + kPad];
    __nv_bfloat16 b[2][kBN][kBK + kPad];
};

__global__ void __launch_bounds__(kThreads)
    bf16_gemm_kernel(const float* __restrict__ x, int m, const __nv_bfloat16* __restrict__ w, int k, int rows,
                     float* __restrict__ y, long long ldy) {
    __shared__ __align__(16) Smem s;
    const int m0 = blockIdx.x * kBM, n0 = blockIdx.y * kBN;
    const int warp = threadIdx.x / VELOX_WARP, lane = threadIdx.x % VELOX_WARP;
    const int wm = (warp % 4) * 32, wn = (warp / 4) * 32;
    // Weights: 64 rows x 32 values = 256 chunks of 16 bytes, one per thread.
    const int br = threadIdx.x / 4, bc = (threadIdx.x % 4) * 8;
    const __nv_bfloat16* wsrc = w + static_cast<long long>(min(n0 + br, rows - 1)) * k + bc;
    auto load_b = [&](int buf, int k0) {
        if (k0 + bc < k)
            cp16(&s.b[buf][br][bc], wsrc + k0);
        else
            *reinterpret_cast<int4*>(&s.b[buf][br][bc]) = make_int4(0, 0, 0, 0);
    };
    // Activations: 128 rows x 32 floats = 1024 float4, four per thread.
    float4 ra[4];
    auto load_a = [&](int k0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int idx = threadIdx.x + i * kThreads, r = idx / 8, c = (idx % 8) * 4;
            ra[i] = m0 + r < m && k0 + c < k ? *reinterpret_cast<const float4*>(x + static_cast<long long>(m0 + r) * k + k0 + c)
                               : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    };
    auto store_a = [&](int buf) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int idx = threadIdx.x + i * kThreads, r = idx / 8, c = (idx % 8) * 4;
            __nv_bfloat162* d = reinterpret_cast<__nv_bfloat162*>(&s.a[buf][r][c]);
            d[0] = __floats2bfloat162_rn(ra[i].x, ra[i].y);
            d[1] = __floats2bfloat162_rn(ra[i].z, ra[i].w);
        }
    };
    // Tensor-core accumulation is not IEEE-exact over long K: the mma accumulators restart every
    // 1024 values and fold into `tot` with ordinary f32 adds.
    float acc[2][4][4] = {}, tot[2][4][4] = {};
    const int steps = (k + kBK - 1) / kBK;
    load_a(0);
    load_b(0, 0);
    cp_commit();
    store_a(0);
    for (int st = 0; st < steps; ++st) {
        const int cur = st & 1;
        cp_wait0();
        __syncthreads();
        if (st + 1 < steps) {
            load_b(cur ^ 1, (st + 1) * kBK);
            cp_commit();
            load_a((st + 1) * kBK);
        }
#pragma unroll
        for (int kk = 0; kk < kBK; kk += 16) {
            uint32_t af[2][4], bfr[2][4];
#pragma unroll
            for (int i = 0; i < 2; ++i)
                ldsm_x4(af[i], &s.a[cur][wm + i * 16 + (lane % 16)][kk + (lane / 16) * 8]);
            // B (stored [n][k]): x4 covers two n8 tiles x k16.
#pragma unroll
            for (int j = 0; j < 2; ++j)
                ldsm_x4(bfr[j], &s.b[cur][wn + j * 16 + (lane % 8) + (lane / 16) * 8][kk + ((lane / 8) % 2) * 8]);
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    mma_bf16(acc[i][j], af[i], bfr[j / 2][(j % 2) * 2], bfr[j / 2][(j % 2) * 2 + 1]);
        }
        if (st + 1 < steps) store_a(cur ^ 1);
        if ((st + 1) % (1024 / kBK) == 0 || st + 1 == steps) {
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j)
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        tot[i][j][q] += acc[i][j][q];
                        acc[i][j][q] = 0.0f;
                    }
        }
    }
    const int g = lane / 4, t = lane % 4;
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = m0 + wm + i * 16 + g + h * 8;
                const int c = n0 + wn + j * 8 + 2 * t;
                if (r >= m) continue;
                if (c < rows) y[static_cast<long long>(r) * ldy + c] = tot[i][j][2 * h];
                if (c + 1 < rows) y[static_cast<long long>(r) * ldy + c + 1] = tot[i][j][2 * h + 1];
            }
}

}  // namespace

// y[t * ldy + r] = bf16(x[t]) . W[row_begin + r] for bf16 weights of K values per row (k % 16 == 0,
// 16-byte aligned rows). VELOX_ERR_UNSUPPORTED otherwise.
VELOX_API int velox_4090_linear_bf16_mma(const float* x, int m, const uint16_t* w, long long k, long long row_begin,
                                         int rows, float* y, long long ldy) {
    if (m < 1 || rows < 1 || k % 16 != 0 || k > (1 << 30) || reinterpret_cast<uintptr_t>(x) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(w) % 16 != 0)
        return VELOX_ERR_UNSUPPORTED;
    const dim3 grid((m + kBM - 1) / kBM, (rows + kBN - 1) / kBN);
    if (grid.y > 65535) return VELOX_ERR_UNSUPPORTED;
    bf16_gemm_kernel<<<grid, kThreads>>>(x, m, reinterpret_cast<const __nv_bfloat16*>(w) + row_begin * k,
                                         static_cast<int>(k), rows, y, ldy);
    return velox_launch_status();
}
