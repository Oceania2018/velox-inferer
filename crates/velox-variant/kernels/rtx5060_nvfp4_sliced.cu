// RTX 5060 Ti NVFP4 linear (block_scale_k16_m128x4_v1) for 1..16 tokens with bf16 activations:
// sliced-K tensor-core kernel.
//
// Adapted from ninfer's nvfp4_a16_sliced_k_mma_kernel (src/ops/linear/nvfp4/
// nvfp4_a16_sliced_k_mma.cuh, Apache-2.0), with ninfer's launch shape for up to 16 tokens: a block
// owns 16 weight rows and 8 warps, each warp one 64-value slice of every 512-value K tile, staged
// through two cp.async stages (codes, the tile's E4M3 group scales, a swizzled bf16 activation
// tile). E2M1 code times its E4M3 scale is exact in bf16, so weights are exact; the tensor
// divisor is applied to the f32 result. Warps' partial sums meet in shared memory.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#include <mutex>
#include <unordered_map>

#include "velox_common.cuh"

namespace {

constexpr int kW = 8, kRows = 16, kTileK = 64, kBlockK = kW * kTileK, kStages = 2, kThreads = kW * 32;

template <int T>
union Smem {
    struct {
        alignas(16) uint8_t codes[kStages][kRows][kBlockK / 2];
        alignas(16) uint8_t scales[kStages][kRows][kBlockK / 16];
        alignas(16) __nv_bfloat16 x[kStages][kW][T * kTileK];
    } st;
    float4 partial[kW][T / 8][32];
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
template <int N>
__device__ __forceinline__ void cp_async(void* dst, const void* src, bool valid) {
    if (N == 16)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src),
                     "r"(valid ? 16 : 0));
    else
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_u32(dst)), "l"(src), "n"(N),
                     "r"(valid ? N : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void ldsm_x2(uint32_t& b0, uint32_t& b1, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(b0), "=r"(b1) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma_bf16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Activation tile [token][64] with 8-value chunks XOR-swizzled by token.
__device__ __forceinline__ int swz(int row, int col) { return col ^ ((row & 7) << 3); }

// E2M1 magnitude codes 0..7 are 0, 0.5, 1, 1.5, 2, 3, 4, 6; built from the bits.
__device__ __forceinline__ float e2m1(uint32_t c) {
    const uint32_t e = (c >> 1) & 3, f = c & 1;
    const uint32_t mag = e ? ((e + 126u) << 23) | (f << 22) : f * (126u << 23);
    return __uint_as_float(mag | ((c & 8u) << 28));
}
__device__ __forceinline__ float e4m3(uint8_t b) {
    const __half_raw h = __nv_cvt_fp8_to_halfraw(b, __NV_E4M3);
    return __half2float(__half(h));
}
// Values 2j, 2j+1 (one code byte) times their group scale, as bf16x2 (exact).
__device__ __forceinline__ uint32_t scaled_pair(uint8_t code, uint8_t scale) {
    const float s = e4m3(scale);
    const __nv_bfloat162 v = __floats2bfloat162_rn(e2m1(code & 0xfu) * s, e2m1(code >> 4) * s);
    return *reinterpret_cast<const uint32_t*>(&v);
}
__device__ __forceinline__ long long scale_offset(long long row, long long group, long long k) {
    return ((row / 128) * (k / 64) + group / 4) * 512 + (row & 31) * 16 + ((row & 127) >> 5) * 4 + (group & 3);
}

template <int T>
__global__ void __launch_bounds__(kThreads, 2)
nvfp4_sliced_kernel(const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ w, long long parent_rows,
                    int k, long long view_begin, int rows, int m, float alpha, float* __restrict__ y) {
    constexpr int NT = T / 8;
    __shared__ Smem<T> sm;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, lid = lane & 3;
    const int r0 = blockIdx.x * kRows;
    const int tiles = k / kBlockK;
    const uint8_t* codes = w;
    const uint8_t* scales = w + (parent_rows * k / 2 + 255) / 256 * 256;

    auto stage = [&](int slot, int tile) {
        const int k0 = tile * kBlockK;
        // Each warp loads 2 rows: 16 code chunks of 16 bytes (chunk-swizzled by row) and the
        // row's 8 scale words (4 groups each).
#pragma unroll
        for (int ri = 0; ri < kRows / kW; ++ri) {
            const int r = warp * (kRows / kW) + ri;
            const long long row = view_begin + (r0 + r < rows ? r0 + r : 0);
            if (lane < kBlockK / 32) {
                cp_async<16>(&sm.st.codes[slot][r][(lane ^ (r & 7)) * 16], codes + row * (k / 2) + k0 / 2 + lane * 16,
                             true);
            } else if (lane < kBlockK / 32 + kBlockK / 64) {
                const int t4 = lane - kBlockK / 32;
                cp_async<4>(&sm.st.scales[slot][r][t4 * 4], scales + scale_offset(row, k0 / 16 + t4 * 4, k), true);
            }
        }
        // This warp's 64 activations of every token row (rows past m are zero).
        for (int item = lane; item < T * (kTileK / 8); item += 32) {
            const int t = item / (kTileK / 8), c8 = (item % (kTileK / 8)) * 8;
            const bool ok = t < m;
            cp_async<16>(&sm.st.x[slot][warp][t * kTileK + swz(t, c8)],
                         x + (ok ? static_cast<long long>(t) * k + k0 + warp * kTileK + c8 : 0), ok);
        }
        cp_commit();
    };

    float acc[NT][4];
#pragma unroll
    for (int ni = 0; ni < NT; ++ni) acc[ni][0] = acc[ni][1] = acc[ni][2] = acc[ni][3] = 0.0f;
#pragma unroll
    for (int s = 0; s < kStages; ++s)
        if (s < tiles) stage(s, s);
    const int b_row = lane & 7, b_k = ((lane >> 3) & 1) << 3, wk0 = warp * kTileK;
    for (int tile = 0; tile < tiles; ++tile) {
        const int slot = tile % kStages;
        if (tile + kStages - 1 < tiles) cp_wait<kStages - 1>();
        else cp_wait<0>();
        __syncthreads();
#pragma unroll
        for (int ks = 0; ks < kTileK / 16; ++ks) {
            const int col = ks * 16 + lid * 2;
            auto pair = [&](int r, int c) {
                const int byte = (wk0 + c) / 2;
                return scaled_pair(sm.st.codes[slot][r][((byte / 16) ^ (r & 7)) * 16 + (byte & 15)],
                                   sm.st.scales[slot][r][(wk0 + c) / 16]);
            };
            const uint32_t a0 = pair(gid, col), a1 = pair(gid + 8, col);
            const uint32_t a2 = pair(gid, col + 8), a3 = pair(gid + 8, col + 8);
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int br = ni * 8 + b_row;
                uint32_t b0, b1;
                ldsm_x2(b0, b1, &sm.st.x[slot][warp][br * kTileK + swz(br, ks * 16 + b_k)]);
                mma_bf16(acc[ni], a0, a1, a2, a3, b0, b1);
            }
        }
        __syncthreads();
        if (tile + kStages < tiles) stage(slot, tile + kStages);
    }
    __syncthreads();
#pragma unroll
    for (int ni = 0; ni < NT; ++ni) sm.partial[warp][ni][lane] = make_float4(acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3]);
    __syncthreads();
    if (warp != 0) return;
#pragma unroll
    for (int ni = 0; ni < NT; ++ni) {
        float4 s = sm.partial[0][ni][lane];
#pragma unroll
        for (int wi = 1; wi < kW; ++wi) {
            const float4 v = sm.partial[wi][ni][lane];
            s.x += v.x, s.y += v.y, s.z += v.z, s.w += v.w;
        }
        // x, y: row gid, tokens 2 lid, 2 lid + 1; z, w: row gid + 8.
        const int ra = r0 + gid, rb = ra + 8, t = ni * 8 + 2 * lid;
        if (ra < rows) {
            if (t < m) y[static_cast<long long>(t) * rows + ra] = s.x * alpha;
            if (t + 1 < m) y[static_cast<long long>(t + 1) * rows + ra] = s.y * alpha;
        }
        if (rb < rows) {
            if (t < m) y[static_cast<long long>(t) * rows + rb] = s.z * alpha;
            if (t + 1 < m) y[static_cast<long long>(t + 1) * rows + rb] = s.w * alpha;
        }
    }
}

__global__ void to_bf16(const float* __restrict__ x, long long n, __nv_bfloat16* __restrict__ y) {
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<long long>(gridDim.x) * blockDim.x)
        y[i] = __float2bfloat16_rn(x[i]);
}

VeloxScratch g_x;

}  // namespace

// y[m, rows] = bf16(x) . W[row_begin.., :]^T / divisor for an NVFP4 parent [parent_rows, k],
// 1 <= m <= 16, k % 512 == 0. Returns VELOX_ERR_UNSUPPORTED for anything else.
VELOX_API int velox_5060_linear_nvfp4_sliced(const float* x, int m, const uint8_t* w, long long parent_rows,
                                             long long k, long long row_begin, int rows, float* y) {
    if (m < 1 || m > 16 || rows <= 0 || k % kBlockK != 0 || k > (1ll << 30) ||
        reinterpret_cast<uintptr_t>(w) % 16 != 0)
        return VELOX_ERR_UNSUPPORTED;
    // The tensor divisor is the FP32 word after the scale plane; read once per parent (a read per
    // call would synchronize).
    static std::mutex mu;
    static std::unordered_map<const uint8_t*, float> cache;
    float divisor;
    {
        std::lock_guard<std::mutex> lock(mu);
        auto it = cache.find(w);
        if (it == cache.end()) {
            float d = 0.0f;
            const long long off = (parent_rows * k / 2 + 255) / 256 * 256 + parent_rows * k / 16;
            if (const cudaError_t e = cudaMemcpy(&d, w + off, 4, cudaMemcpyDeviceToHost)) return static_cast<int>(e);
            it = cache.emplace(w, d).first;
        }
        divisor = it->second;
    }
    auto* xb = static_cast<__nv_bfloat16*>(g_x.get(static_cast<size_t>(m) * k * 2));
    if (!xb) return static_cast<int>(cudaErrorMemoryAllocation);
    to_bf16<<<velox_grid_1d(static_cast<long long>(m) * k, 256), 256>>>(x, static_cast<long long>(m) * k, xb);
    const unsigned grid = static_cast<unsigned>((rows + kRows - 1) / kRows);
    const int ki = static_cast<int>(k);
    if (m <= 8)
        nvfp4_sliced_kernel<8><<<grid, kThreads>>>(xb, w, parent_rows, ki, row_begin, rows, m, 1.0f / divisor, y);
    else
        nvfp4_sliced_kernel<16><<<grid, kThreads>>>(xb, w, parent_rows, ki, row_begin, rows, m, 1.0f / divisor, y);
    return velox_launch_status();
}
