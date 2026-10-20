// RTX 5090 tensor-core linear for prefill: activations are rounded on the fly to the precision the
// artifact allows (NVFP4 for AllowA4 + NVFP4 weights, E4M3 for AllowA8 + FP8 weights) and
// multiplied with block-scaled MMA on sm_120a:
//
//   A4 x W4: mma.sync.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64 (E2M1, UE4M3 per 16)
//   A8 x W8: mma.sync.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32 with unit UE8M0 scales,
//            which runs at full rate with FP32 accumulation; row scales are applied after.
//
// Both operands are 32 bytes of codes per row per MMA k-step, so one tile loop serves both. The
// host-side reference rounding is velox_core::quant; the quantizers below perform the same
// single-precision operations in the same order.
#include <cuda_fp8.h>
#include <stdint.h>

#include <mutex>
#include <unordered_map>

#include "velox_common.cuh"

namespace {

// ---------------------------------------------------------------------------------------------
// Activation quantizers.

__device__ __forceinline__ uint8_t to_e4m3(float v) {
    return static_cast<uint8_t>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ float from_e4m3(uint8_t b) {
    const __half_raw h = __nv_cvt_fp8_to_halfraw(b, __NV_E4M3);
    return __half2float(__half(h));
}

// Two E2M1 codes in one byte: `lo` in bits 0-3, `hi` in bits 4-7 (RNE, saturating).
__device__ __forceinline__ uint32_t e2m1x2(float lo, float hi) {
    uint32_t r;
    asm("{\n.reg .b8 b;\ncvt.rn.satfinite.e2m1x2.f32 b, %1, %2;\ncvt.u32.u8 %0, b;\n}"
        : "=r"(r)
        : "f"(hi), "f"(lo));
    return r;
}

// One thread per 16-element group. Rows in [m, m_pad) are written as zeros so the GEMM can read
// whole tiles. codes: [m_pad, k/2], scales: [m_pad, k/16].
__global__ void quant_nvfp4_kernel(const float* __restrict__ x, int m, int m_pad, int k, float divisor,
                                   uint8_t* __restrict__ codes, uint8_t* __restrict__ scales) {
    const int groups = k / 16;
    const long long g = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= static_cast<long long>(m_pad) * groups) return;
    const int row = static_cast<int>(g / groups), grp = static_cast<int>(g % groups);
    uint2 packed = make_uint2(0, 0);
    uint8_t s = 0;
    if (row < m) {
        const float4* src = reinterpret_cast<const float4*>(x + static_cast<long long>(row) * k + grp * 16);
        float v[16];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float4 t = src[q];
            v[4 * q] = t.x, v[4 * q + 1] = t.y, v[4 * q + 2] = t.z, v[4 * q + 3] = t.w;
        }
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(v[i]));
        s = to_e4m3(__fdiv_rn(__fmul_rn(divisor, amax), 6.0f));
        if (s != 0) {
            const float ds = from_e4m3(s);
            uint32_t w[2] = {0, 0};
#pragma unroll
            for (int i = 0; i < 16; i += 2) {
                const float a = __fdiv_rn(__fmul_rn(v[i], divisor), ds);
                const float b = __fdiv_rn(__fmul_rn(v[i + 1], divisor), ds);
                w[i / 8] |= e2m1x2(a, b) << (8 * ((i / 2) % 4));
            }
            packed = make_uint2(w[0], w[1]);
        }
    }
    *reinterpret_cast<uint2*>(codes + g * 8) = packed;
    scales[g] = s;
}

// One block per row: scale = amax / 448, codes = e4m3(x * (1 / scale)). codes: [m_pad, k].
__global__ void quant_fp8_kernel(const float* __restrict__ x, int m, int k, uint8_t* __restrict__ codes,
                                 float* __restrict__ row_scale) {
    __shared__ float scratch[VELOX_WARP];
    const int row = blockIdx.x;
    uint8_t* out = codes + static_cast<long long>(row) * k;
    if (row >= m) {
        for (int c = threadIdx.x * 4; c < k; c += blockDim.x * 4) *reinterpret_cast<uint32_t*>(out + c) = 0;
        if (threadIdx.x == 0) row_scale[row] = 0.0f;
        return;
    }
    const float* xr = x + static_cast<long long>(row) * k;
    float amax = 0.0f;
    for (int c = threadIdx.x * 4; c < k; c += blockDim.x * 4) {
        const float4 t = *reinterpret_cast<const float4*>(xr + c);
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(t.x), fabsf(t.y)), fmaxf(fabsf(t.z), fabsf(t.w))));
    }
    // Block max via the sum helper's pattern.
    amax = velox_warp_max(amax);
    if (threadIdx.x % VELOX_WARP == 0) scratch[threadIdx.x / VELOX_WARP] = amax;
    __syncthreads();
    if (threadIdx.x < VELOX_WARP) {
        float v = threadIdx.x < blockDim.x / VELOX_WARP ? scratch[threadIdx.x] : 0.0f;
        v = velox_warp_max(v);
        if (threadIdx.x == 0) scratch[0] = v;
    }
    __syncthreads();
    const float scale = __fdiv_rn(scratch[0], 448.0f);
    const float inv = scale > 0.0f ? __fdiv_rn(1.0f, scale) : 0.0f;
    for (int c = threadIdx.x * 4; c < k; c += blockDim.x * 4) {
        const float4 t = *reinterpret_cast<const float4*>(xr + c);
        const uint32_t w = to_e4m3(__fmul_rn(t.x, inv)) | (to_e4m3(__fmul_rn(t.y, inv)) << 8) |
                           (to_e4m3(__fmul_rn(t.z, inv)) << 16) | (static_cast<uint32_t>(to_e4m3(__fmul_rn(t.w, inv))) << 24);
        *reinterpret_cast<uint32_t*>(out + c) = w;
    }
    if (threadIdx.x == 0) row_scale[row] = scale;
}

// ---------------------------------------------------------------------------------------------
// GEMM.

constexpr int kBM = 128;      // tokens per block
constexpr int kBN = 128;      // weight rows per block
constexpr int kBK = 64;       // code bytes per row per stage (two MMA k-steps)
constexpr int kStages = 3;
constexpr int kThreads = 256;  // 8 warps: 2 along tokens x 4 along rows, 64 x 32 each

enum Kind { kFp4 = 0, kFp8 = 1 };

struct Smem {
    uint8_t a[kStages][kBM * kBK];
    uint8_t b[kStages][kBN * kBK];
    uint32_t a_scale[kStages][kBM][2];  // FP4: per row, the scale words of the stage's two k64 steps
    uint8_t b_scale[kStages][2][512];   // FP4: the weight's swizzled 128x4 scale tiles, as stored
};

// 16-byte chunk c of row r lives at chunk c ^ ((r >> 1) & 3): ldmatrix over 8 rows is
// conflict-free with 64-byte rows.
__device__ __forceinline__ int swz(int row, int byte) {
    return row * kBK + ((((byte >> 4) ^ (row >> 1)) & 3) << 4) + (byte & 15);
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src),
                 "r"(valid ? 16 : 0));
}

__device__ __forceinline__ void cp_async8(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(smem_u32(dst)), "l"(src));
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_u32(p)));
}

__device__ __forceinline__ void mma_fp4(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1,
                                        uint32_t sfa, uint32_t sfb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "h"((uint16_t)0),
          "h"((uint16_t)0), "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
}

__device__ __forceinline__ void mma_fp8(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0x7fu), "h"((uint16_t)0),
          "h"((uint16_t)0), "r"(0x7fu), "h"((uint16_t)0), "h"((uint16_t)0));
}

struct GemmArgs {
    const uint8_t* a;         // [m_pad, kb] activation codes
    const uint8_t* a_scales;  // FP4: [m_pad, k/16] UE4M3;  FP8: unused
    const float* a_row;       // FP8: [m_pad] row scales
    const uint8_t* w;         // weight payload (parent)
    long long parent_rows;
    long long row_begin;
    int rows;                 // view rows
    int m;
    int k;                    // logical K
    int kb;                   // code bytes per row
    float alpha;              // FP4: 1 / (input divisor * weight divisor)
    float* y;                 // [m, rows]
};

template <int KIND>
__device__ __forceinline__ void load_stage(Smem& s, const GemmArgs& p, int stage, int kt, int m0, int n0,
                                           const uint8_t* w_scale_plane) {
    const int t = threadIdx.x;
    const int kbyte = kt * kBK;
    // A and B: 128 rows x 4 chunks each, two chunks per thread per operand.
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int idx = t + i * kThreads;
        const int row = idx >> 2, chunk = idx & 3;
        cp_async16(s.a[stage] + swz(row, chunk * 16),
                   p.a + static_cast<long long>(m0 + row) * p.kb + kbyte + chunk * 16, true);
        const long long wrow = p.row_begin + n0 + row;
        const bool ok = wrow < p.parent_rows;
        cp_async16(s.b[stage] + swz(row, chunk * 16), p.w + (ok ? wrow : 0) * p.kb + kbyte + chunk * 16, ok);
    }
    if (KIND == kFp4) {
        // Two k64 steps per stage. A: one 8-byte copy per row. B: two contiguous 512-byte tiles.
        const int k64 = kt * 2;
        if (t < kBM)
            cp_async8(&s.a_scale[stage][t][0], p.a_scales + static_cast<long long>(m0 + t) * (p.k / 16) + k64 * 4);
        if (t < 64) {
            const int half = t >> 5, off = (t & 31) * 16;
            const long long tile = (p.row_begin + n0) / 128;
            cp_async16(&s.b_scale[stage][half][off],
                       w_scale_plane + (tile * (p.k / 64) + k64 + half) * 512 + off, true);
        }
    }
}

template <int KIND>
__global__ void __launch_bounds__(kThreads, 2) gemm_kernel(GemmArgs p) {
    extern __shared__ __align__(128) uint8_t smem_raw[];
    Smem& s = *reinterpret_cast<Smem*>(smem_raw);
    const int n0 = blockIdx.x * kBN, m0 = blockIdx.y * kBM;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int wm = warp >> 2, wn = warp & 3;  // 2 x 4 warps
    const int kt_count = p.kb / kBK;
    const uint8_t* w_scale_plane =
        p.w + (KIND == kFp4 ? ((p.parent_rows * p.kb + 255) / 256 * 256) : 0);

#pragma unroll
    for (int st = 0; st < kStages - 1; ++st) {
        if (st < kt_count) load_stage<KIND>(s, p, st, st, m0, n0, w_scale_plane);
        cp_commit();
    }

    float acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int v = 0; v < 4; ++v) acc[i][j][v] = 0.0f;

    // ldmatrix lane roles. A x4: matrices (rows 0-7 | 8-15) x (bytes 0-15 | 16-31).
    const int a_row = (lane & 7) + (((lane >> 3) & 1) << 3), a_byte = (lane >> 4) << 4;
    // B x4 covers two n8 tiles: matrices (tile0 lo, tile0 hi, tile1 lo, tile1 hi).
    const int b_row = (lane & 7) + ((lane >> 4) << 3), b_byte = ((lane >> 3) & 1) << 4;
    const int sfa_row = ((lane & 1) << 3) | (lane >> 2);
    const int sfb_row = lane >> 2;

    for (int kt = 0; kt < kt_count; ++kt) {
        cp_wait<kStages - 2>();
        __syncthreads();
        const int next = kt + kStages - 1;
        if (next < kt_count) load_stage<KIND>(s, p, next % kStages, next, m0, n0, w_scale_plane);
        cp_commit();
        const int st = kt % kStages;
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            uint32_t af[4][4], bf[4][2], sfa[4], sfb[4];
#pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
                const int r = wm * 64 + mt * 16 + a_row;
                ldmatrix_x4(af[mt], s.a[st] + swz(r, ks * 32 + a_byte));
                if (KIND == kFp4) sfa[mt] = s.a_scale[st][wm * 64 + mt * 16 + sfa_row][ks];
            }
#pragma unroll
            for (int nt = 0; nt < 4; nt += 2) {
                const int r = wn * 32 + nt * 8 + b_row;
                uint32_t t4[4];
                ldmatrix_x4(t4, s.b[st] + swz(r, ks * 32 + b_byte));
                bf[nt][0] = t4[0], bf[nt][1] = t4[1], bf[nt + 1][0] = t4[2], bf[nt + 1][1] = t4[3];
            }
            if (KIND == kFp4) {
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    const int r = wn * 32 + nt * 8 + sfb_row;
                    sfb[nt] = *reinterpret_cast<const uint32_t*>(&s.b_scale[st][ks][(r & 31) * 16 + (r >> 5) * 4]);
                }
            }
#pragma unroll
            for (int mt = 0; mt < 4; ++mt)
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    if (KIND == kFp4) mma_fp4(acc[mt][nt], af[mt], bf[nt][0], bf[nt][1], sfa[mt], sfb[nt]);
                    else mma_fp8(acc[mt][nt], af[mt], bf[nt][0], bf[nt][1]);
                }
        }
    }
    cp_wait<0>();

    // Epilogue: c0,c1 at (row g, cols 2q, 2q+1); c2,c3 at row g+8.
    const int g = lane >> 2, q = lane & 3;
    const uint8_t* wscale = p.w + (p.parent_rows * p.kb + 255) / 256 * 256;  // FP8 BF16 row scales
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
        const int n = n0 + wn * 32 + nt * 8 + 2 * q;
        float f0 = p.alpha, f1 = p.alpha;
        if (KIND == kFp8) {
            const long long r0 = p.row_begin + n;
            const uint16_t* sp = reinterpret_cast<const uint16_t*>(wscale);
            f0 = n < p.rows ? __uint_as_float(static_cast<uint32_t>(sp[r0]) << 16) : 0.0f;
            f1 = n + 1 < p.rows ? __uint_as_float(static_cast<uint32_t>(sp[r0 + 1]) << 16) : 0.0f;
        }
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int t = m0 + wm * 64 + mt * 16 + g + h * 8;
                if (t >= p.m) continue;
                const float xs = KIND == kFp8 ? p.a_row[t] : 1.0f;
                float* out = p.y + static_cast<long long>(t) * p.rows + n;
                const float v0 = acc[mt][nt][2 * h] * f0 * xs, v1 = acc[mt][nt][2 * h + 1] * f1 * xs;
                if (n + 1 < p.rows && (p.rows & 1) == 0) {
                    *reinterpret_cast<float2*>(out) = make_float2(v0, v1);
                } else {
                    if (n < p.rows) out[0] = v0;
                    if (n + 1 < p.rows) out[1] = v1;
                }
            }
        }
    }
}

// Grow-only device scratch for quantized activations. The engine issues kernels from one thread;
// the mutex only guards against concurrent first use.
struct Scratch {
    std::mutex mu;
    uint8_t* ptr = nullptr;
    size_t bytes = 0;
    uint8_t* get(size_t need) {
        std::lock_guard<std::mutex> lock(mu);
        if (need > bytes) {
            if (ptr) cudaFree(ptr);
            ptr = nullptr;
            bytes = 0;
            if (cudaMalloc(&ptr, need) != cudaSuccess) return nullptr;
            bytes = need;
        }
        return ptr;
    }
};
Scratch g_scratch;

template <int KIND>
int launch_gemm(const GemmArgs& p) {
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(gemm_kernel<KIND>, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Smem));
        configured = true;
    }
    const dim3 grid((p.rows + kBN - 1) / kBN, (p.m + kBM - 1) / kBM);
    gemm_kernel<KIND><<<grid, kThreads, sizeof(Smem)>>>(p);
    return velox_launch_status();
}

}  // namespace

// y[m, rows] = nvfp4(x) . W[row_begin.., :]^T for an NVFP4 weight. Requires k % 128 == 0,
// row_begin % 128 == 0 and 16-byte aligned x rows; returns VELOX_ERR_UNSUPPORTED otherwise.
VELOX_API int velox_5090_linear_a4(const float* x, int m, const uint8_t* w, long long parent_rows, long long k,
                                   long long row_begin, int rows, float input_divisor, float* y) {
    if (k % 128 != 0 || row_begin % 128 != 0 || m <= 0 || reinterpret_cast<uintptr_t>(x) % 16 != 0)
        return VELOX_ERR_UNSUPPORTED;
    const int m_pad = (m + kBM - 1) / kBM * kBM;
    const size_t code_bytes = static_cast<size_t>(m_pad) * k / 2;
    uint8_t* buf = g_scratch.get(code_bytes + static_cast<size_t>(m_pad) * k / 16);
    if (!buf) return static_cast<int>(cudaErrorMemoryAllocation);
    uint8_t* codes = buf;
    uint8_t* scales = buf + code_bytes;
    const long long groups = static_cast<long long>(m_pad) * (k / 16);
    quant_nvfp4_kernel<<<velox_grid_1d(groups, 256), 256>>>(x, m, m_pad, static_cast<int>(k), input_divisor, codes, scales);
    // The weight divisor is the FP32 word after the scale plane; weights are immutable, so it is
    // read from the device once per parent and cached (a read per call would synchronize).
    float weight_divisor;
    {
        static std::mutex mu;
        static std::unordered_map<const uint8_t*, float> cache;
        std::lock_guard<std::mutex> lock(mu);
        auto it = cache.find(w);
        if (it == cache.end()) {
            const long long scale_plane = (parent_rows * k / 2 + 255) / 256 * 256;
            if (const cudaError_t e = cudaMemcpy(&weight_divisor, w + scale_plane + parent_rows * k / 16, 4,
                                                 cudaMemcpyDeviceToHost))
                return static_cast<int>(e);
            it = cache.emplace(w, weight_divisor).first;
        }
        weight_divisor = it->second;
    }
    GemmArgs p{codes, scales, nullptr, w, parent_rows, row_begin, rows, m, static_cast<int>(k),
               static_cast<int>(k / 2), 1.0f / (input_divisor * weight_divisor), y};
    if (const int e = velox_launch_status()) return e;
    return launch_gemm<kFp4>(p);
}

// y[m, rows] = e4m3(x) . W^T for an FP8 row-scaled weight. Requires k % 64 == 0.
VELOX_API int velox_5090_linear_a8(const float* x, int m, const uint8_t* w, long long parent_rows, long long k,
                                   long long row_begin, int rows, float* y) {
    if (k % 64 != 0 || m <= 0 || reinterpret_cast<uintptr_t>(x) % 16 != 0) return VELOX_ERR_UNSUPPORTED;
    const int m_pad = (m + kBM - 1) / kBM * kBM;
    const size_t code_bytes = static_cast<size_t>(m_pad) * k;
    uint8_t* buf = g_scratch.get(code_bytes + static_cast<size_t>(m_pad) * 4);
    if (!buf) return static_cast<int>(cudaErrorMemoryAllocation);
    float* row_scale = reinterpret_cast<float*>(buf + code_bytes);
    quant_fp8_kernel<<<m_pad, 256>>>(x, m, static_cast<int>(k), buf, row_scale);
    if (const int e = velox_launch_status()) return e;
    GemmArgs p{buf, nullptr, row_scale, w, parent_rows, row_begin, rows, m, static_cast<int>(k),
               static_cast<int>(k), 1.0f, y};
    return launch_gemm<kFp8>(p);
}
