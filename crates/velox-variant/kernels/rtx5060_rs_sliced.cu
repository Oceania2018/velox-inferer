// RTX 5060 Ti groupwise-integer linear (row_split_k128_v1: q4_g64, q5_g64) for 2..16 tokens:
// sliced-K tensor-core kernel.
//
// Adapted from ninfer's q5_a16_sliced_k_mma_kernel (src/ops/linear/q5/q5_a16_sliced_k_mma.cuh,
// Apache-2.0), which ninfer selects on this card for the DFlash2 verify width. A block owns RT
// 16-row tiles of weight rows; its four warps each own one 64-value group of every 256-value K
// tile, staged through two cp.async stages (codes, q5 high bits, FP16 scales and the bf16
// activations of the tile). Weights are the A operand of mma.m16n8k16, tokens the B operand
// (ldmatrix from a swizzled activation tile, reused by the block's RT row tiles: every block
// stages the whole activation tile, so more rows per block means less L2 traffic per weight byte).
// Integer codes are bf16-exact; each group's scale is applied to its f32 partial. The warps'
// partial sums meet in shared memory at the end.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include "velox_common.cuh"

namespace {

enum Fmt { kQ4 = 4, kQ5 = 5 };

constexpr int kW = 4, kS = 2, kThreads = kW * 32;

// Staged rows are padded by 16 bytes so the 8 rows a warp reads at once (lane gid) fall in
// distinct banks: unpadded, the 128-byte code rows all hit the same four banks (75% of the shared
// accesses were bank conflicts, q4 [8192, 5120] m=8).
constexpr int kCodeRow = kW * 32 + 16, kHighRow = kW * 8 + 16;

template <int T, int RT>
union Smem {
    static constexpr int R = 16 * RT;
    struct {
        alignas(16) uint8_t codes[kS][R][kCodeRow];
        alignas(16) uint8_t high[kS][R][kHighRow];
        alignas(16) uint16_t scales[kS][R][kW];
        alignas(16) __nv_bfloat16 x[kS][kW][T][64];
    } st;
    float4 partial[kW][RT][T / 8][32];
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

// Activation tile [token][64] with 16-byte chunks XOR-swizzled by token, so ldmatrix rows hit
// distinct banks.
__device__ __forceinline__ int swz(int row, int col) { return (((col >> 3) ^ (row & 7)) << 3) | (col & 7); }

// bf16x2 (lo, hi) = (128 + a, 128 + b) - bias for 0 <= a, b < 128.
__device__ __forceinline__ uint32_t biased_pair(uint32_t a, uint32_t b, uint32_t bias2) {
    const uint32_t v = 0x43004300u | a | (b << 16);
    const __nv_bfloat162 r = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&v),
                                     *reinterpret_cast<const __nv_bfloat162*>(&bias2));
    return *reinterpret_cast<const uint32_t*>(&r);
}

// Values 2b, 2b+1 of a group (code byte b; q5: high bits 2b, 2b+1) as bf16x2.
template <int F>
__device__ __forceinline__ uint32_t pair_of(uint32_t code, uint32_t high8, int b) {
    if (F == kQ4) return biased_pair((code & 0xfu) ^ 8u, (code >> 4) ^ 8u, 0x43084308u);
    const uint32_t hb = high8 >> (2 * (b & 3));
    return biased_pair(((code & 0xfu) | ((hb & 1u) << 4)) ^ 16u, ((code >> 4) | ((hb & 2u) << 3)) ^ 16u, 0x43104310u);
}

// Blocks per SM the launch bounds ask for: fewer as the block's staged rows grow.
__host__ __device__ constexpr int min_blocks(int rt) { return rt == 1 ? 6 : rt == 2 ? 4 : 3; }

template <int F, int T, int GPR, int RT>
__global__ void __launch_bounds__(kThreads, min_blocks(RT))
rs_sliced_kernel(const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ w, long long parent_rows,
                 int k_rt, int kp_rt, long long view_begin, int rows, int m, float* __restrict__ y,
                 long long ldy) {
    constexpr int NT = T / 8, R = 16 * RT;
    __shared__ Smem<T, RT> sm;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, lid = lane & 3;
    const int kp = GPR ? GPR * 64 : kp_rt, k = GPR ? kp : k_rt;
    const int gpr = kp / 64, groups = (k + 63) / 64, tiles = (groups + kW - 1) / kW;
    const int r0 = blockIdx.x * R;
    const long long n_groups = parent_rows * gpr;
    const uint8_t* codes = w;
    const uint8_t* high = w + (n_groups * 32 + 255) / 256 * 256;
    const uint8_t* scales = high + (F == kQ5 ? (n_groups * 8 + 255) / 256 * 256 : 0);

    auto stage = [&](int slot, int tile) {
        const int g0 = tile * kW;
        for (int item = tid; item < R * kW * 2; item += kThreads) {
            const int half = item & 1, g = (item >> 1) % kW, r = item / (kW * 2);
            const bool ok = r0 + r < rows && g0 + g < groups;
            const long long gi = ok ? (view_begin + r0 + r) * gpr + g0 + g : 0;
            cp_async<16>(&sm.st.codes[slot][r][g * 32 + half * 16], codes + gi * 32 + half * 16, ok);
        }
        for (int item = tid; item < R * (kW / 2); item += kThreads) {
            const int r = item / (kW / 2), g = (item % (kW / 2)) * 2;
            const bool ok = r0 + r < rows && g0 + g < groups;
            const long long gi = ok ? (view_begin + r0 + r) * gpr + g0 + g : 0;
            if (F == kQ5) cp_async<16>(&sm.st.high[slot][r][g * 8], high + gi * 8, ok);
            // A row's scale pair (groups g, g+1) is 4-byte aligned: gpr and g0 + g are even.
            cp_async<4>(&sm.st.scales[slot][r][g], scales + gi * 2, ok);
        }
        // This warp's 64 activations of each token: 8 chunks of 16 bytes per token.
        for (int item = lane; item < T * 8; item += 32) {
            const int t = item / 8, c8 = (item % 8) * 8;
            const bool ok = t < m && (g0 + warp) * 64 + c8 < kp;
            const long long xi = ok ? static_cast<long long>(t) * kp + (g0 + warp) * 64 + c8 : 0;
            cp_async<16>(&sm.st.x[slot][warp][t][swz(t, c8)], x + xi, ok);
        }
        cp_commit();
    };

    float acc[RT][NT][4];
#pragma unroll
    for (int ri = 0; ri < RT; ++ri)
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) acc[ri][ni][0] = acc[ri][ni][1] = acc[ri][ni][2] = acc[ri][ni][3] = 0.0f;
#pragma unroll
    for (int slot = 0; slot < kS; ++slot) {
        if (slot < tiles) stage(slot, slot);
        else cp_commit();
    }
    constexpr int kUnroll = GPR ? (GPR + kW - 1) / kW : 1;
#pragma unroll kUnroll
    for (int tile = 0; tile < tiles; ++tile) {
        const int slot = tile % kS;
        cp_wait<kS - 1>();
        __syncthreads();
        float part[RT][NT][4];
#pragma unroll
        for (int ri = 0; ri < RT; ++ri)
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) part[ri][ni][0] = part[ri][ni][1] = part[ri][ni][2] = part[ri][ni][3] = 0.0f;
        // The group's 32 code bytes of rows gid and gid + 8 as words (two 16-byte loads each), and
        // its q5 high bits.
        uint32_t c_top[RT][8], c_bot[RT][8], h_top[RT][2], h_bot[RT][2];
#pragma unroll
        for (int ri = 0; ri < RT; ++ri) {
            const uint4* t4 = reinterpret_cast<const uint4*>(&sm.st.codes[slot][ri * 16 + gid][warp * 32]);
            const uint4* b4 = reinterpret_cast<const uint4*>(&sm.st.codes[slot][ri * 16 + gid + 8][warp * 32]);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const uint4 t = t4[h], b = b4[h];
                c_top[ri][4 * h] = t.x, c_top[ri][4 * h + 1] = t.y, c_top[ri][4 * h + 2] = t.z, c_top[ri][4 * h + 3] = t.w;
                c_bot[ri][4 * h] = b.x, c_bot[ri][4 * h + 1] = b.y, c_bot[ri][4 * h + 2] = b.z, c_bot[ri][4 * h + 3] = b.w;
            }
            h_top[ri][0] = h_top[ri][1] = h_bot[ri][0] = h_bot[ri][1] = 0u;
            if (F == kQ5) {
                const uint2 a = *reinterpret_cast<const uint2*>(&sm.st.high[slot][ri * 16 + gid][warp * 8]);
                const uint2 b = *reinterpret_cast<const uint2*>(&sm.st.high[slot][ri * 16 + gid + 8][warp * 8]);
                h_top[ri][0] = a.x, h_top[ri][1] = a.y, h_bot[ri][0] = b.x, h_bot[ri][1] = b.y;
            }
        }
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            // k16 step ks: MMA k-index 2 lid + {0,1} is value 16 ks + 2 lid + {0,1} (code byte
            // 8 ks + lid), k-index 2 lid + 8 + {0,1} is byte 8 ks + 4 + lid.
            const int b0 = ks * 8 + lid, b1 = b0 + 4;
            uint32_t a[RT][4];
#pragma unroll
            for (int ri = 0; ri < RT; ++ri) {
                // Byte b0 is byte lid of word 2 ks, b1 byte lid of word 2 ks + 1.
                const uint32_t t0 = (c_top[ri][2 * ks] >> (8 * lid)) & 0xffu, t1 = (c_top[ri][2 * ks + 1] >> (8 * lid)) & 0xffu;
                const uint32_t u0 = (c_bot[ri][2 * ks] >> (8 * lid)) & 0xffu, u1 = (c_bot[ri][2 * ks + 1] >> (8 * lid)) & 0xffu;
                a[ri][0] = pair_of<F>(t0, (h_top[ri][b0 >> 4] >> (8 * ((b0 >> 2) & 3))) & 0xffu, b0);
                a[ri][1] = pair_of<F>(u0, (h_bot[ri][b0 >> 4] >> (8 * ((b0 >> 2) & 3))) & 0xffu, b0);
                a[ri][2] = pair_of<F>(t1, (h_top[ri][b1 >> 4] >> (8 * ((b1 >> 2) & 3))) & 0xffu, b1);
                a[ri][3] = pair_of<F>(u1, (h_bot[ri][b1 >> 4] >> (8 * ((b1 >> 2) & 3))) & 0xffu, b1);
            }
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int br = ni * 8 + (lane & 7), bk = ks * 16 + ((lane >> 3) & 1) * 8;
                uint32_t x0, x1;
                ldsm_x2(x0, x1, &sm.st.x[slot][warp][br][swz(br, bk)]);
#pragma unroll
                for (int ri = 0; ri < RT; ++ri) mma_bf16(part[ri][ni], a[ri][0], a[ri][1], a[ri][2], a[ri][3], x0, x1);
            }
        }
        const bool live = tile * kW + warp < groups;
#pragma unroll
        for (int ri = 0; ri < RT; ++ri) {
            const float top = live ? __half2float(__ushort_as_half(sm.st.scales[slot][ri * 16 + gid][warp])) : 0.0f;
            const float bot = live ? __half2float(__ushort_as_half(sm.st.scales[slot][ri * 16 + gid + 8][warp])) : 0.0f;
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                acc[ri][ni][0] = fmaf(part[ri][ni][0], top, acc[ri][ni][0]);
                acc[ri][ni][1] = fmaf(part[ri][ni][1], top, acc[ri][ni][1]);
                acc[ri][ni][2] = fmaf(part[ri][ni][2], bot, acc[ri][ni][2]);
                acc[ri][ni][3] = fmaf(part[ri][ni][3], bot, acc[ri][ni][3]);
            }
        }
        if (tile + kS < tiles) {
            __syncthreads();  // the stage is reused
            stage(slot, tile + kS);
        } else {
            cp_commit();
        }
    }
    cp_wait<0>();
    __syncthreads();
#pragma unroll
    for (int ri = 0; ri < RT; ++ri)
#pragma unroll
        for (int ni = 0; ni < NT; ++ni)
            sm.partial[warp][ri][ni][lane] = make_float4(acc[ri][ni][0], acc[ri][ni][1], acc[ri][ni][2], acc[ri][ni][3]);
    __syncthreads();
    // Warp ri (mod kW) reduces and stores row tiles ri, ri + kW, ...
#pragma unroll
    for (int ri = 0; ri < RT; ++ri) {
        if (ri % kW != warp) continue;
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            float4 s = sm.partial[0][ri][ni][lane];
#pragma unroll
            for (int wi = 1; wi < kW; ++wi) {
                const float4 v = sm.partial[wi][ri][ni][lane];
                s.x += v.x, s.y += v.y, s.z += v.z, s.w += v.w;
            }
            // x, y: row gid, tokens 2 lid, 2 lid + 1; z, w: row gid + 8.
            const int ra = r0 + ri * 16 + gid, rb = ra + 8, t = ni * 8 + 2 * lid;
            if (ra < rows) {
                if (t < m) y[static_cast<long long>(t) * ldy + ra] = s.x;
                if (t + 1 < m) y[static_cast<long long>(t + 1) * ldy + ra] = s.y;
            }
            if (rb < rows) {
                if (t < m) y[static_cast<long long>(t) * ldy + rb] = s.z;
                if (t + 1 < m) y[static_cast<long long>(t + 1) * ldy + rb] = s.w;
            }
        }
    }
}

// x [m, k] f32 -> [m, kp] bf16 (RNE), zero padded columns.
__global__ void to_bf16_pad(const float* __restrict__ x, int m, int k, int kp, __nv_bfloat16* __restrict__ y) {
    const long long n = static_cast<long long>(m) * kp;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const long long r = i / kp;
        const int c = static_cast<int>(i % kp);
        y[i] = __float2bfloat16_rn(c < k ? x[r * k + c] : 0.0f);
    }
}

VeloxScratch g_x;

template <int F, int T, int GPR, int RT>
int launch(const __nv_bfloat16* x, const uint8_t* w, long long n, int k, int kp, long long rb, int rows, int m,
           float* y, long long ldy) {
    constexpr int R = 16 * RT;
    rs_sliced_kernel<F, T, GPR, RT><<<(rows + R - 1) / R, kThreads>>>(x, w, n, k, kp, rb, rows, m, y, ldy);
    return velox_launch_status();
}

template <int F, int T, int RT>
int launch_k(const __nv_bfloat16* x, const uint8_t* w, long long n, int k, int kp, long long rb, int rows, int m,
             float* y, long long ldy) {
    // The model's input widths as compile-time constants (the tile loop unrolls).
    if (k == 5120 && kp == 5120) return launch<F, T, 80, RT>(x, w, n, k, kp, rb, rows, m, y, ldy);
    if (k == 6144 && kp == 6144) return launch<F, T, 96, RT>(x, w, n, k, kp, rb, rows, m, y, ldy);
    return launch<F, T, 0, RT>(x, w, n, k, kp, rb, rows, m, y, ldy);
}

// Up to 8 tokens one row tile per block, 16 tokens two (measured, verify at 16k keys: batched
// 2 x 8 rows 38.4 -> 37.7 ms per round; one sequence slower with two).
template <int F, int T>
int launch_rt(const __nv_bfloat16* x, const uint8_t* w, long long n, int k, int kp, long long rb, int rows, int m,
              float* y, long long ldy) {
    return launch_k<F, T, T == 16 ? 2 : 1>(x, w, n, k, kp, rb, rows, m, y, ldy);
}

int run(const __nv_bfloat16* xb, int m, const uint8_t* w, long long parent_rows, int ki, int kp, long long row_begin,
        int rows, int format, float* y, long long ldy) {
    if (format == kQ5)
        return m <= 8 ? launch_rt<kQ5, 8>(xb, w, parent_rows, ki, kp, row_begin, rows, m, y, ldy)
                      : launch_rt<kQ5, 16>(xb, w, parent_rows, ki, kp, row_begin, rows, m, y, ldy);
    return m <= 8 ? launch_rt<kQ4, 8>(xb, w, parent_rows, ki, kp, row_begin, rows, m, y, ldy)
                  : launch_rt<kQ4, 16>(xb, w, parent_rows, ki, kp, row_begin, rows, m, y, ldy);
}

VeloxScratch g_shared;

}  // namespace

// y[m, rows] = bf16(x) . W[row_begin.., :]^T for a row_split_k128_v1 parent [parent_rows, k] in
// format 4 (q4_g64) or 5 (q5_g64), 1 <= m <= 16. Returns VELOX_ERR_UNSUPPORTED for anything else.
VELOX_API int velox_5060_linear_rs_sliced(const float* x, int m, const uint8_t* w, long long parent_rows, long long k,
                                          long long row_begin, int rows, int format, float* y) {
    if (m < 1 || m > 16 || rows <= 0 || k > (1ll << 30) || reinterpret_cast<uintptr_t>(w) % 16 != 0)
        return VELOX_ERR_UNSUPPORTED;
    if (format != kQ4 && format != kQ5) return VELOX_ERR_UNSUPPORTED;
    const int ki = static_cast<int>(k), kp = static_cast<int>((k + 127) / 128 * 128);
    auto* xb = static_cast<__nv_bfloat16*>(g_x.get(static_cast<size_t>(m) * kp * 2));
    if (!xb) return static_cast<int>(cudaErrorMemoryAllocation);
    to_bf16_pad<<<velox_grid_1d(static_cast<long long>(m) * kp, 256), 256>>>(x, m, ki, kp, xb);
    return run(xb, m, w, parent_rows, ki, kp, row_begin, rows, format, y, rows);
}

// x [m, k] f32 -> bf16 (RNE) in a scratch that velox_5060_linear_rs_sliced_bf16 calls then read:
// one conversion for several projections of the same input. k % 128 == 0 (no column padding).
// The scratch is reused by the next call (stream order keeps earlier readers correct).
VELOX_API int velox_5060_bf16_act(const float* x, int m, long long k, const void** xb_out) {
    if (m < 1 || m > 16 || k % 128 != 0 || k > (1ll << 30)) return VELOX_ERR_UNSUPPORTED;
    const int ki = static_cast<int>(k);
    auto* xb = static_cast<__nv_bfloat16*>(g_shared.get(static_cast<size_t>(m) * ki * 2));
    if (!xb) return static_cast<int>(cudaErrorMemoryAllocation);
    to_bf16_pad<<<velox_grid_1d(static_cast<long long>(m) * ki, 256), 256>>>(x, m, ki, ki, xb);
    *xb_out = xb;
    return velox_launch_status();
}

// velox_5060_linear_rs_sliced on activations converted by velox_5060_bf16_act, writing
// y[t * ldy + r] (rows <= ldy).
VELOX_API int velox_5060_linear_rs_sliced_bf16(const void* xb, int m, const uint8_t* w, long long parent_rows,
                                               long long k, long long row_begin, int rows, int format, float* y,
                                               long long ldy) {
    if (m < 1 || m > 16 || rows <= 0 || ldy < rows || k % 128 != 0 || k > (1ll << 30) ||
        reinterpret_cast<uintptr_t>(w) % 16 != 0 || (format != kQ4 && format != kQ5))
        return VELOX_ERR_UNSUPPORTED;
    const int ki = static_cast<int>(k);
    return run(static_cast<const __nv_bfloat16*>(xb), m, w, parent_rows, ki, ki, row_begin, rows, format, y, ldy);
}
