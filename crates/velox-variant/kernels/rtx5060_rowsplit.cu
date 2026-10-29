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

// Pipelined form for f32 activations whose rows are float4 aligned (K % 4 == 0): the bf16 rounding
// happens as each tile is staged, so there is no padded copy of x, and K advances 32 values per step
// through double-buffered shared memory. The next step's activations and weight codes load into
// registers while the current step's MMAs run; one barrier per step. The same MMA order within
// each output and the same per-group scale folds as rowsplit_bf16_kernel, on a 128 x 128 block
// (42 KB of shared memory). ViT shapes at 4032 tokens: [1152, 1152] 238 -> 206 us, [1152, 4304]
// 923 -> 895 us, [4304, 1152] 775 -> 794 us (bench-linear, includes the old kernel's bf16 copy). The
// tensor pipe is busy ~60% of the time: every warp meets the barrier, so MMA bursts and staging
// alternate rather than overlap.
//
// Templated on the code width so group arithmetic is compile time. 4/5/6-bit codes decode two at a
// time without int-to-float conversions: bf16 0x4300 | u is exactly 128 + u for u < 128, so with
// the sign bit flipped first, one packed bf16 subtraction of 128 + 2^(bits-1) gives the signed code.
constexpr int kPK = 32, kPS = kPK + 8;  // 80-byte rows: odd multiples of 16 bytes, conflict-free ldmatrix
// Block 128 tokens x 128 weight rows; 8 warps, 2 along tokens x 4 along rows, 64 x 32 each.
constexpr int kPBN = 128, kPMT = 4, kPNT = 4;

struct PipeSmem {
    __nv_bfloat16 a[2][kBM * kPS];
    __nv_bfloat16 b[2][kPBN * kPS];
    float scale[2][kPBN];  // by group parity
};

struct PipeArgs {
    const float* x;  // [m, k]
    const uint8_t* w;
    long long parent_rows, row_begin;
    int rows, m, k, kp;
    long long gpr, high_off, scale_off;
    float* y;
};

__device__ __forceinline__ void mma_bf16_zero(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%10,%10,%10,%10};\n"
                 : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(0.0f));
}

// Codes of the 8 consecutive K values a thread decodes per step: weight row t / 4, values (t % 4) * 8.
struct PipeCodes {
    uint2 lo;     // eight nibbles (lo.x) or eight int8 codes
    uint32_t hi;  // high-plane bits: q5 one per value, q6 two
};

template <int BITS>
__device__ __forceinline__ void pipe_store_w(__nv_bfloat16* dst, const PipeCodes& c) {
    uint4 o;
    uint32_t* op = &o.x;
    if constexpr (BITS == 8) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t w = i < 2 ? c.lo.x : c.lo.y;
            const int sh = 16 * (i % 2);
            const __nv_bfloat162 h = __floats2bfloat162_rn(static_cast<float>(static_cast<int8_t>(w >> sh)),
                                                           static_cast<float>(static_cast<int8_t>(w >> (sh + 8))));
            op[i] = *reinterpret_cast<const uint32_t*>(&h);
        }
    } else {
        constexpr int hb = BITS - 4;
        constexpr uint32_t bias = 0x4300u + (1u << (BITS - 1));  // bf16 of 128 + 2^(bits-1)
        const uint32_t y = BITS == 4 ? c.lo.x ^ 0x88888888u : c.lo.x;
        const uint32_t h = c.hi ^ (hb == 1 ? 0xffu : 0xaaaau);  // flip each value's top bit
        const uint32_t y4 = y >> 4;
        const __nv_bfloat162 sub = __halves2bfloat162(__ushort_as_bfloat16(static_cast<unsigned short>(bias)),
                                                     __ushort_as_bfloat16(static_cast<unsigned short>(bias)));
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            // Byte p holds values 2p (low nibble) and 2p + 1 (high nibble).
            uint32_t t = (__byte_perm(y, y4, p | ((4 + p) << 8)) & 0x000f000fu) | 0x43004300u;
            if constexpr (hb == 1) {
                const uint32_t hp = h >> (2 * p);
                t |= ((hp & 1u) << 4) | ((hp & 2u) << 19);
            } else if constexpr (hb == 2) {
                const uint32_t hp = h >> (4 * p);
                t |= ((hp & 3u) << 4) | ((hp & 0xcu) << 18);
            }
            const __nv_bfloat162 v = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&t), sub);
            op[p] = *reinterpret_cast<const uint32_t*>(&v);
        }
    }
    *reinterpret_cast<uint4*>(dst) = o;
}

template <int BITS>
__global__ void __launch_bounds__(kThreads) rowsplit_bf16_pipe(PipeArgs p) {
    constexpr int kGroup = BITS == 8 ? 32 : 64, kSpg = kGroup / kPK;
    constexpr int kHigh = BITS == 5 ? 8 : BITS == 6 ? 16 : 0;
    extern __shared__ __align__(128) uint8_t smem_raw[];
    PipeSmem& s = *reinterpret_cast<PipeSmem*>(smem_raw);
    const int n0 = blockIdx.x * kPBN, m0 = blockIdx.y * kBM;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp >> 2, wn = warp & 3;
    const int g = lane >> 2, q4 = lane & 3;
    const int a_row = (lane & 7) + (((lane >> 3) & 1) << 3), a_col = (lane >> 4) << 3;
    const int b_row = (lane & 7) + ((lane >> 4) << 3), b_col = ((lane >> 3) & 1) << 3;

    // Activations: rows tid / 8 + 32 i, columns (tid % 8) * 4 of each 32-value step.
    const int ar = tid >> 3, ac = (tid & 7) * 4;
    const float* xp = p.x + static_cast<long long>(m0 + ar) * p.k + ac;
    const long long xstep = 32LL * p.k;
    bool arow[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) arow[i] = m0 + ar + 32 * i < p.m;
    float4 ra[4];
    auto load_a = [&](int k0) {
        const bool kin = k0 + ac < p.k;
#pragma unroll
        for (int i = 0; i < 4; ++i)
            ra[i] = arow[i] && kin ? __ldg(reinterpret_cast<const float4*>(xp + i * xstep + k0))
                                   : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    };
    auto store_a = [&](int buf) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const __nv_bfloat162 lo = __floats2bfloat162_rn(ra[i].x, ra[i].y);
            const __nv_bfloat162 hi = __floats2bfloat162_rn(ra[i].z, ra[i].w);
            *reinterpret_cast<uint2*>(s.a[buf] + (ar + 32 * i) * kPS + ac) =
                make_uint2(*reinterpret_cast<const uint32_t*>(&lo), *reinterpret_cast<const uint32_t*>(&hi));
        }
    };

    // Weights: this thread's row and its 16 values within each step (two runs of 8).
    const int wr = tid >> 1, wv = (tid & 1) * 16;
    const long long wrow = p.row_begin + n0 + wr;
    const bool wvalid = wrow < p.parent_rows && n0 + wr < p.rows;
    const long long g0 = (wvalid ? wrow : 0) * p.gpr;  // this row's first group
    PipeCodes wc[2] = {{make_uint2(0, 0), 0}, {make_uint2(0, 0), 0}};
    float wscale = 0.0f;
    auto load_w = [&](int st) {
        if (!wvalid) return;
        const long long gi = g0 + st / kSpg;
        const int ln = (st % kSpg) * kPK + wv;  // value index within the group
        const uint8_t* base = p.w + gi * 32;
        if constexpr (BITS == 8) {
            const uint4 v = *reinterpret_cast<const uint4*>(base + ln);
            wc[0].lo = make_uint2(v.x, v.y);
            wc[1].lo = make_uint2(v.z, v.w);
        } else {
            const uint2 v = *reinterpret_cast<const uint2*>(base + ln / 2);
            wc[0].lo.x = v.x;
            wc[1].lo.x = v.y;
            if constexpr (BITS == 5) {
                const uint32_t h = *reinterpret_cast<const uint16_t*>(p.w + p.high_off + gi * kHigh + ln / 8);
                wc[0].hi = h & 0xffu;
                wc[1].hi = h >> 8;
            }
            if constexpr (BITS == 6) {
                const uint32_t h = *reinterpret_cast<const uint32_t*>(p.w + p.high_off + gi * kHigh + ln / 4);
                wc[0].hi = h & 0xffffu;
                wc[1].hi = h >> 16;
            }
        }
        if ((tid & 1) == 0 && st % kSpg == 0)
            wscale = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(p.w + p.scale_off + 2 * gi)));
    };
    auto store_w = [&](int buf, int st) {
        pipe_store_w<BITS>(s.b[buf] + wr * kPS + wv, wc[0]);
        pipe_store_w<BITS>(s.b[buf] + wr * kPS + wv + 8, wc[1]);
        if ((tid & 1) == 0 && st % kSpg == 0) s.scale[(st / kSpg) & 1][wr] = wscale;
    };

    float acc[kPMT][kPNT][4], part[kPMT][kPNT][4];
#pragma unroll
    for (int i = 0; i < kPMT; ++i)
#pragma unroll
        for (int j = 0; j < kPNT; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.0f;

    const int steps = p.kp / kPK;
    load_a(0);
    load_w(0);
    store_a(0);
    store_w(0, 0);
    __syncthreads();
    for (int st0 = 0; st0 < steps; st0 += kSpg) {
        // Scales double-buffer by group (the next group's are stored before this one's fold has
        // passed a barrier when a group spans two steps), tiles by step.
        const int sbuf = (st0 / kSpg) & 1;
#pragma unroll
        for (int u = 0; u < kSpg; ++u) {
            const int st = st0 + u, cur = st & 1;
            const bool more = st + 1 < steps;
            if (more) {
                load_a((st + 1) * kPK);
                load_w(st + 1);
            }
#pragma unroll
            for (int kk = 0; kk < kPK; kk += 16) {
                uint32_t af[kPMT][4], bf[kPNT][2];
#pragma unroll
                for (int mt = 0; mt < kPMT; ++mt)
                    ldsm_x4(af[mt], s.a[cur] + (wm * 16 * kPMT + mt * 16 + a_row) * kPS + kk + a_col);
#pragma unroll
                for (int np = 0; np < kPNT / 2; ++np) {
                    uint32_t t4[4];
                    ldsm_x4(t4, s.b[cur] + (wn * 8 * kPNT + np * 16 + b_row) * kPS + kk + b_col);
                    bf[2 * np][0] = t4[0], bf[2 * np][1] = t4[1], bf[2 * np + 1][0] = t4[2], bf[2 * np + 1][1] = t4[3];
                }
#pragma unroll
                for (int mt = 0; mt < kPMT; ++mt)
#pragma unroll
                    for (int nt = 0; nt < kPNT; ++nt) {
                        if (u == 0 && kk == 0)
                            mma_bf16_zero(part[mt][nt], af[mt], bf[nt][0], bf[nt][1]);
                        else
                            mma_bf16(part[mt][nt], af[mt], bf[nt][0], bf[nt][1]);
                    }
                // The next tiles' stores between the MMA halves: warps issue in order, so this
                // work issues while the queued MMAs run instead of after the warp's last one.
                if (more && kk == 0) store_a(cur ^ 1);
                if (more && kk == 16) store_w(cur ^ 1, st + 1);
            }
            if (u == kSpg - 1) {
#pragma unroll
                for (int nt = 0; nt < kPNT; ++nt) {
                    const int n = wn * 8 * kPNT + nt * 8 + 2 * q4;
                    const float s0 = s.scale[sbuf][n], s1 = s.scale[sbuf][n + 1];
#pragma unroll
                    for (int mt = 0; mt < kPMT; ++mt) {
                        acc[mt][nt][0] += part[mt][nt][0] * s0;
                        acc[mt][nt][1] += part[mt][nt][1] * s1;
                        acc[mt][nt][2] += part[mt][nt][2] * s0;
                        acc[mt][nt][3] += part[mt][nt][3] * s1;
                    }
                }
            }
            __syncthreads();
        }
    }

#pragma unroll
    for (int mt = 0; mt < kPMT; ++mt)
#pragma unroll
        for (int nt = 0; nt < kPNT; ++nt) {
            const int n = n0 + wn * 8 * kPNT + nt * 8 + 2 * q4;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int t = m0 + wm * 16 * kPMT + mt * 16 + g + 8 * h;
                if (t >= p.m) continue;
                float* out = p.y + static_cast<long long>(t) * p.rows + n;
                if (n < p.rows) out[0] = acc[mt][nt][2 * h];
                if (n + 1 < p.rows) out[1] = acc[mt][nt][2 * h + 1];
            }
        }
}

template <int BITS>
void launch_pipe(const PipeArgs& a) {
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(rowsplit_bf16_pipe<BITS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(PipeSmem));
        configured = true;
    }
    rowsplit_bf16_pipe<BITS><<<dim3((a.rows + kPBN - 1) / kPBN, (a.m + kBM - 1) / kBM), kThreads, sizeof(PipeSmem)>>>(a);
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
VELOX_API int velox_5060_linear_rowsplit(const float* x, int m, const uint8_t* w, long long parent_rows, long long k,
                                         long long row_begin, int rows, int bits, float* y) {
    if (m <= 0 || (bits != 4 && bits != 5 && bits != 6 && bits != 8)) return VELOX_ERR_UNSUPPORTED;
    const int group = bits == 8 ? 32 : 64;
    const int high = bits == 5 ? 8 : bits == 6 ? 16 : 0;
    const long long kp = (k + 127) / 128 * 128;
    const long long gpr = kp / group, groups = parent_rows * gpr;
    const long long high_off = align256(groups * 32), scale_off = high_off + align256(groups * high);
    if (k % 4 == 0 && reinterpret_cast<uintptr_t>(x) % 16 == 0 && k <= (1 << 30)) {
        const PipeArgs a{x, w, parent_rows, row_begin, rows, m, static_cast<int>(k), static_cast<int>(kp),
                         gpr, high_off, scale_off, y};
        switch (bits) {
            case 4: launch_pipe<4>(a); break;
            case 5: launch_pipe<5>(a); break;
            case 6: launch_pipe<6>(a); break;
            default: launch_pipe<8>(a); break;
        }
        return velox_launch_status();
    }
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
