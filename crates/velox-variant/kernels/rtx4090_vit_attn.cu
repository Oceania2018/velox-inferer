// RTX 5060 Ti (from rtx-5090-32g) non-causal attention for the vision tower (attention_full), FlashAttention-2 schedule
// on f16 tensor cores (AttnPrecision::F16): Q, K, V are converted to f16 with head_dim padded to
// DP (64, 80 or 128; Qwen3.5 ViT uses 72 -> 80), scores and the value product use
// mma.m16n8k16 f16 with FP32 accumulation, softmax runs in FP32.
//
// Block: one head x 64 queries, 4 warps of 16 rows; keys stream in tiles of 64 through a
// two-stage cp.async pipeline. Shared-memory rows are DP + 8 halves, an odd number of 16-byte
// chunks, so ldmatrix over 8 consecutive rows hits 8 distinct bank groups without swizzling.
#include <cuda_fp16.h>
#include <stdint.h>

#include <mutex>

#include "velox_common.cuh"

namespace {

constexpr int kBM = 64, kBN = 64, kWarps = 4, kThreads = kWarps * 32;

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src),
                 "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma_f16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&h);
}

// x [rows, d] f32 -> y [rows, dp] f16, zero padded.
__global__ void pad_to_f16(const float* __restrict__ x, long long rows, int d, int dp, __half* __restrict__ y) {
    const long long n = rows * dp;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const long long r = i / dp;
        const int c = static_cast<int>(i % dp);
        y[i] = __float2half_rn(c < d ? x[r * d + c] : 0.0f);
    }
}

template <int DP>
struct VitSmem {
    static constexpr int SD = DP + 8;
    __half k[2][kBN * SD];
    __half v[2][kBN * SD];
};

struct VitArgs {
    const __half* q;  // [tokens, heads, DP]
    const __half* k;
    const __half* v;
    float* out;       // [tokens, heads, d]
    int tokens, heads, d;
    float scale_log2;  // softmax scale * log2(e)
};

template <int DP>
__device__ __forceinline__ void vit_load(VitSmem<DP>& s, const VitArgs& p, int stage, int kt, int h) {
    constexpr int SD = DP + 8, CH = DP / 8;  // 16-byte chunks per row
    for (int idx = threadIdx.x; idx < kBN * CH; idx += kThreads) {
        const int row = idx / CH, c = idx % CH;
        const int key = kt * kBN + row;
        const bool ok = key < p.tokens;
        const long long g = (static_cast<long long>(ok ? key : 0) * p.heads + h) * DP + c * 8;
        cp_async16(s.k[stage] + row * SD + c * 8, p.k + g, ok);
        cp_async16(s.v[stage] + row * SD + c * 8, p.v + g, ok);
    }
}

template <int DP>
__global__ void __launch_bounds__(kThreads) vit_flash_kernel(VitArgs p) {
    constexpr int SD = DP + 8, KS = DP / 16, NO = DP / 8;
    extern __shared__ __align__(128) uint8_t smem_raw[];
    VitSmem<DP>& s = *reinterpret_cast<VitSmem<DP>*>(smem_raw);
    const int h = blockIdx.y, t0 = blockIdx.x * kBM;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, q4 = lane & 3;
    const int r0 = t0 + warp * 16 + g, r1 = r0 + 8;

    uint32_t qf[KS][4];
    {
        const __half* q0 = p.q + (static_cast<long long>(r0 < p.tokens ? r0 : 0) * p.heads + h) * DP;
        const __half* q1 = p.q + (static_cast<long long>(r1 < p.tokens ? r1 : 0) * p.heads + h) * DP;
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
            const int c = ks * 16 + 2 * q4;
            qf[ks][0] = r0 < p.tokens ? *reinterpret_cast<const uint32_t*>(q0 + c) : 0u;
            qf[ks][1] = r1 < p.tokens ? *reinterpret_cast<const uint32_t*>(q1 + c) : 0u;
            qf[ks][2] = r0 < p.tokens ? *reinterpret_cast<const uint32_t*>(q0 + c + 8) : 0u;
            qf[ks][3] = r1 < p.tokens ? *reinterpret_cast<const uint32_t*>(q1 + c + 8) : 0u;
        }
    }
    float o[NO][4];
#pragma unroll
    for (int i = 0; i < NO; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    const int n_tiles = (p.tokens + kBN - 1) / kBN;

    vit_load<DP>(s, p, 0, 0, h);
    cp_commit();
    for (int kt = 0; kt < n_tiles; ++kt) {
        const int st = kt & 1;
        if (kt + 1 < n_tiles) {
            vit_load<DP>(s, p, st ^ 1, kt + 1, h);
            cp_commit();
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();

        float sc[8][4];
#pragma unroll
        for (int i = 0; i < 8; ++i) sc[i][0] = sc[i][1] = sc[i][2] = sc[i][3] = 0.0f;
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
#pragma unroll
            for (int np = 0; np < 4; ++np) {
                uint32_t b[4];
                const int row = np * 16 + (lane & 7) + ((lane >> 4) << 3);
                ldsm_x4(b, s.k[st] + row * SD + ks * 16 + (((lane >> 3) & 1) << 3));
                mma_f16(sc[2 * np], qf[ks], b[0], b[1]);
                mma_f16(sc[2 * np + 1], qf[ks], b[2], b[3]);
            }
        }

        const int key0 = kt * kBN;
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const bool ok = key0 + nt * 8 + 2 * q4 + e < p.tokens;
                sc[nt][e] = ok ? sc[nt][e] * p.scale_log2 : -INFINITY;
                sc[nt][2 + e] = ok ? sc[nt][2 + e] * p.scale_log2 : -INFINITY;
                mx0 = fmaxf(mx0, sc[nt][e]);
                mx1 = fmaxf(mx1, sc[nt][2 + e]);
            }
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
        const float ref0 = mn0 == -INFINITY ? 0.0f : mn0, ref1 = mn1 == -INFINITY ? 0.0f : mn1;
        const float al0 = exp2f(m0 - ref0), al1 = exp2f(m1 - ref1);
        m0 = mn0, m1 = mn1;
        float ps0 = 0.0f, ps1 = 0.0f;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            sc[nt][0] = exp2f(sc[nt][0] - ref0);
            sc[nt][1] = exp2f(sc[nt][1] - ref0);
            sc[nt][2] = exp2f(sc[nt][2] - ref1);
            sc[nt][3] = exp2f(sc[nt][3] - ref1);
            ps0 += sc[nt][0] + sc[nt][1];
            ps1 += sc[nt][2] + sc[nt][3];
        }
        l0 = l0 * al0 + ps0;
        l1 = l1 * al1 + ps1;
#pragma unroll
        for (int i = 0; i < NO; ++i) {
            o[i][0] *= al0, o[i][1] *= al0;
            o[i][2] *= al1, o[i][3] *= al1;
        }

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t a[4] = {pack_h2(sc[2 * j][0], sc[2 * j][1]), pack_h2(sc[2 * j][2], sc[2 * j][3]),
                                   pack_h2(sc[2 * j + 1][0], sc[2 * j + 1][1]),
                                   pack_h2(sc[2 * j + 1][2], sc[2 * j + 1][3])};
            const int vrow = j * 16 + (lane & 7) + (((lane >> 3) & 1) << 3);
#pragma unroll
            for (int dn = 0; dn < DP / 16; ++dn) {
                uint32_t b[4];
                ldsm_x4_t(b, s.v[st] + vrow * SD + dn * 16 + ((lane >> 4) << 3));
                mma_f16(o[2 * dn], a, b[0], b[1]);
                mma_f16(o[2 * dn + 1], a, b[2], b[3]);
            }
        }
        __syncthreads();
    }

    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float inv0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, inv1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
#pragma unroll
    for (int nt = 0; nt < NO; ++nt) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int c = nt * 8 + 2 * q4 + e;
            if (c >= p.d) continue;
            if (r0 < p.tokens) p.out[(static_cast<long long>(r0) * p.heads + h) * p.d + c] = o[nt][e] * inv0;
            if (r1 < p.tokens) p.out[(static_cast<long long>(r1) * p.heads + h) * p.d + c] = o[nt][2 + e] * inv1;
        }
    }
}

// Grow-only and never freed (VeloxScratch): recorded CUDA graphs may still reference old buffers.
struct Scratch {
    VeloxScratch s;
    uint8_t* get(size_t need) { return static_cast<uint8_t*>(s.get(need)); }
};
Scratch g_vit;

template <int DP>
int launch_vit(const float* q, const float* k, const float* v, float* out, int tokens, int heads, int d) {
    const long long rows = static_cast<long long>(tokens) * heads;
    const size_t each = static_cast<size_t>(rows) * DP;
    uint8_t* base = g_vit.get(3 * each * sizeof(__half));
    if (!base) return static_cast<int>(cudaErrorMemoryAllocation);
    __half* qh = reinterpret_cast<__half*>(base);
    __half* kh = qh + each;
    __half* vh = kh + each;
    const unsigned grid = velox_grid_1d(rows * DP, 256);
    pad_to_f16<<<grid, 256>>>(q, rows, d, DP, qh);
    pad_to_f16<<<grid, 256>>>(k, rows, d, DP, kh);
    pad_to_f16<<<grid, 256>>>(v, rows, d, DP, vh);
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(vit_flash_kernel<DP>, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(VitSmem<DP>));
        configured = true;
    }
    const VitArgs a{qh, kh, vh, out, tokens, heads, d, 1.4426950408889634f / sqrtf(static_cast<float>(d))};
    vit_flash_kernel<DP><<<dim3((tokens + kBM - 1) / kBM, heads), kThreads, sizeof(VitSmem<DP>)>>>(a);
    return velox_launch_status();
}

}  // namespace

// Non-causal attention (see velox_core::backend::FullAttnShape) for head_dim <= 128.
VELOX_API int velox_4090_attention_full(const float* q, const float* k, const float* v, float* out, int tokens,
                                        int heads, int head_dim) {
    if (tokens <= 0 || heads <= 0 || heads > 65535) return VELOX_ERR_UNSUPPORTED;
    if (head_dim <= 64) return launch_vit<64>(q, k, v, out, tokens, heads, head_dim);
    if (head_dim <= 80) return launch_vit<80>(q, k, v, out, tokens, heads, head_dim);
    if (head_dim <= 128) return launch_vit<128>(q, k, v, out, tokens, heads, head_dim);
    return VELOX_ERR_UNSUPPORTED;
}
