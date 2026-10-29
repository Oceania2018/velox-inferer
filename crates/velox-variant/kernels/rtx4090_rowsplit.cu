// RTX 5060 Ti (from rtx-5090-32g) tensor-core linear for groupwise-integer weights (q4/q5/q6_g64, q8_g32;
// row_split_k128_v1) with bf16 activations (ActQuant::Bf16). Used by the vision tower and the
// draft models, whose uses are A16Only.
//
// The weight stays exact: integer codes (|q| <= 128) are bf16-exact, so they go into the MMA as
// they are; each group of `group` K values accumulates into its own register tile, which is then
// multiplied by that group's FP16 scale (one per output row) and added to the result.
//
// Block: 128 tokens x 64 weight rows, 8 warps (4 along tokens x 2 along rows, 32 x 32 each);
// K advances 64 values per step: activations by cp.async, weights decoded by the block into
// shared memory. Rows are 72 bf16 (9 chunks of 16 bytes, odd), so ldmatrix is conflict-free.
#include <cuda_bf16.h>
#include <stdint.h>

#include <mutex>

#include "velox_common.cuh"

namespace {

constexpr int kBM = 128, kBN = 64, kBK = 64, kThreads = 256, kSA = kBK + 8;

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_u32(dst)), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::); }
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// x [m, k] f32 -> [m_pad, kp] bf16 (RNE), zero padded in both dimensions.
__global__ void to_bf16_padded(const float* __restrict__ x, int m, int k, int m_pad, int kp,
                               __nv_bfloat16* __restrict__ y) {
    const long long n = static_cast<long long>(m_pad) * kp;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const int r = static_cast<int>(i / kp), c = static_cast<int>(i % kp);
        y[i] = __float2bfloat16_rn(r < m && c < k ? x[static_cast<long long>(r) * k + c] : 0.0f);
    }
}

struct RsArgs {
    const __nv_bfloat16* x;  // [m_pad, kp]
    const uint8_t* w;        // parent payload
    long long parent_rows, row_begin;
    int rows, m, kp;
    int bits, group, high;   // high-plane bytes per group (q5: 8, q6: 16)
    long long gpr, high_off, scale_off;
    float* y;                // [m, rows]
};

struct RsSmem {
    __nv_bfloat16 a[kBM * kSA];
    __nv_bfloat16 b[kBN * kSA];
    float scale[kBN][2];  // per row: the step's group scales (two groups when group == 32)
};

// Thread t decodes 16 consecutive K values of one weight row: row t / 4, values (t % 4) * 16.
__device__ __forceinline__ void decode_step(RsSmem& s, const RsArgs& p, int n0, int k0) {
    const int t = threadIdx.x, row = t >> 2, part = t & 3;
    const long long wrow = p.row_begin + n0 + row;
    const int e0 = k0 + part * 16;  // first K index of this thread
    int q[16];
    if (wrow < p.parent_rows) {
        const long long gi = wrow * p.gpr + e0 / p.group;
        const int lane = e0 % p.group;
        const uint8_t* base = p.w + gi * 32;
        if (p.bits == 8) {
            const uint4 v = *reinterpret_cast<const uint4*>(base + lane);
            const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int i = 0; i < 16; ++i) q[i] = static_cast<int8_t>((w[i / 4] >> (8 * (i % 4))) & 0xff);
        } else {
            const uint2 v = *reinterpret_cast<const uint2*>(base + lane / 2);
            const uint32_t w[2] = {v.x, v.y};
            uint32_t hi = 0;
            const int hb = p.bits - 4;
            if (hb > 0) {
                // hb bits per value; 16 values start at bit lane * hb, a multiple of 16.
                const uint8_t* hp = p.w + p.high_off + gi * p.high + lane * hb / 8;
                hi = hb == 1 ? *reinterpret_cast<const uint16_t*>(hp) : *reinterpret_cast<const uint32_t*>(hp);
            }
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                uint32_t u = (w[i / 8] >> (4 * (i % 8))) & 0xf;
                if (hb > 0) u |= ((hi >> (i * hb)) & ((1u << hb) - 1)) << 4;
                q[i] = u >= (1u << (p.bits - 1)) ? static_cast<int>(u) - (1 << p.bits) : static_cast<int>(u);
            }
        }
        if (part == 0 || (p.group == 32 && part == 2)) {
            const uint16_t h = *reinterpret_cast<const uint16_t*>(p.w + p.scale_off + 2 * gi);
            s.scale[row][part == 0 ? 0 : 1] = __half2float(__ushort_as_half(h));
        }
    } else {
#pragma unroll
        for (int i = 0; i < 16; ++i) q[i] = 0;
        if (part == 0) s.scale[row][0] = s.scale[row][1] = 0.0f;
    }
    __nv_bfloat16* dst = s.b + row * kSA + part * 16;
#pragma unroll
    for (int i = 0; i < 16; i += 2)
        *reinterpret_cast<__nv_bfloat162*>(dst + i) =
            __floats2bfloat162_rn(static_cast<float>(q[i]), static_cast<float>(q[i + 1]));
}

__global__ void __launch_bounds__(kThreads) rowsplit_bf16_kernel(RsArgs p) {
    extern __shared__ __align__(128) uint8_t smem_raw[];
    RsSmem& s = *reinterpret_cast<RsSmem*>(smem_raw);
    const int n0 = blockIdx.x * kBN, m0 = blockIdx.y * kBM;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int wm = warp >> 1, wn = warp & 1;
    const int g = lane >> 2, q4 = lane & 3;
    const int groups_per_step = kBK / p.group;  // 1 (group 64) or 2 (group 32)
    const int ks_per_group = p.group / 16;

    float acc[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;

    const int a_row = (lane & 7) + (((lane >> 3) & 1) << 3), a_col = (lane >> 4) << 3;
    const int b_row = (lane & 7) + ((lane >> 4) << 3), b_col = ((lane >> 3) & 1) << 3;

    for (int k0 = 0; k0 < p.kp; k0 += kBK) {
        for (int idx = threadIdx.x; idx < kBM * (kBK / 8); idx += kThreads) {
            const int r = idx >> 3, c = idx & 7;
            cp_async16(s.a + r * kSA + c * 8, p.x + static_cast<long long>(m0 + r) * p.kp + k0 + c * 8);
        }
        cp_commit();
        decode_step(s, p, n0, k0);
        cp_wait_all();
        __syncthreads();
        for (int gs = 0; gs < groups_per_step; ++gs) {
            float part[2][4][4];
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) part[i][j][0] = part[i][j][1] = part[i][j][2] = part[i][j][3] = 0.0f;
            for (int kk = 0; kk < ks_per_group; ++kk) {
                const int col = (gs * ks_per_group + kk) * 16;
                uint32_t af[2][4], bf[4][2];
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) ldsm_x4(af[mt], s.a + (wm * 32 + mt * 16 + a_row) * kSA + col + a_col);
#pragma unroll
                for (int np = 0; np < 2; ++np) {
                    uint32_t t4[4];
                    ldsm_x4(t4, s.b + (wn * 32 + np * 16 + b_row) * kSA + col + b_col);
                    bf[2 * np][0] = t4[0], bf[2 * np][1] = t4[1], bf[2 * np + 1][0] = t4[2], bf[2 * np + 1][1] = t4[3];
                }
#pragma unroll
                for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                    for (int nt = 0; nt < 4; ++nt) mma_bf16(part[mt][nt], af[mt], bf[nt][0], bf[nt][1]);
            }
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int n = wn * 32 + nt * 8 + 2 * q4;
                const float s0 = s.scale[n][gs], s1 = s.scale[n + 1][gs];
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    acc[mt][nt][0] += part[mt][nt][0] * s0;
                    acc[mt][nt][1] += part[mt][nt][1] * s1;
                    acc[mt][nt][2] += part[mt][nt][2] * s0;
                    acc[mt][nt][3] += part[mt][nt][3] * s1;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const int n = n0 + wn * 32 + nt * 8 + 2 * q4;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int t = m0 + wm * 32 + mt * 16 + g + 8 * h;
                if (t >= p.m) continue;
                float* out = p.y + static_cast<long long>(t) * p.rows + n;
                if (n < p.rows) out[0] = acc[mt][nt][2 * h];
                if (n + 1 < p.rows) out[1] = acc[mt][nt][2 * h + 1];
            }
        }
}

// Grow-only and never freed (VeloxScratch): recorded CUDA graphs may still reference old buffers.
struct Scratch {
    VeloxScratch s;
    uint8_t* get(size_t need) { return static_cast<uint8_t*>(s.get(need)); }
};
Scratch g_rs;

long long align256(long long x) { return (x + 255) / 256 * 256; }

}  // namespace

// y[m, rows] = bf16(x) . W[row_begin.., :]^T for a row_split_k128_v1 weight of `bits` (4, 5, 6, 8).
VELOX_API int velox_4090_linear_rowsplit(const float* x, int m, const uint8_t* w, long long parent_rows, long long k,
                                         long long row_begin, int rows, int bits, float* y) {
    if (m <= 0 || (bits != 4 && bits != 5 && bits != 6 && bits != 8)) return VELOX_ERR_UNSUPPORTED;
    const int group = bits == 8 ? 32 : 64;
    const int high = bits == 5 ? 8 : bits == 6 ? 16 : 0;
    const long long kp = (k + 127) / 128 * 128;
    const long long gpr = kp / group, groups = parent_rows * gpr;
    const long long high_off = align256(groups * 32), scale_off = high_off + align256(groups * high);
    const int m_pad = (m + kBM - 1) / kBM * kBM;
    auto* xb = reinterpret_cast<__nv_bfloat16*>(g_rs.get(static_cast<size_t>(m_pad) * kp * 2));
    if (!xb) return static_cast<int>(cudaErrorMemoryAllocation);
    to_bf16_padded<<<velox_grid_1d(static_cast<long long>(m_pad) * kp, 256), 256>>>(
        x, m, static_cast<int>(k), m_pad, static_cast<int>(kp), xb);
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(rowsplit_bf16_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(RsSmem));
        configured = true;
    }
    const RsArgs a{xb, w, parent_rows, row_begin, rows, m, static_cast<int>(kp), bits, group, high, gpr, high_off,
                   scale_off, y};
    rowsplit_bf16_kernel<<<dim3((rows + kBN - 1) / kBN, m_pad / kBM), kThreads, sizeof(RsSmem)>>>(a);
    return velox_launch_status();
}
