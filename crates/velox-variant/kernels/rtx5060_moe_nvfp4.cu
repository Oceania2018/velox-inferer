// RTX 5060 Ti routed experts (Backend::moe_experts) stored as NVFP4 (block_scale_k16_m128x4_v1):
// gate, up and down are each an NVFP4 matrix of a [gate | up | down] record, the record in mapped
// system RAM or in a device slot (Qwen3.6-35B-A3B: 256 experts, top-8, [512, 2048] gate / up).
//
// A matrix [n, K] is E2M1 codes [n, K / 2] (low nibble = even column), padded to 256 bytes, then
// one E4M3 scale per 16 values in 128x4 tiles (row tile r / 128, k64 tile; within a 512-byte tile
// row r % 32 owns 16 bytes, of which quarter (r % 128) / 32 owns 4, one per 16-group), then the f32
// tensor divisor. A block therefore takes weight rows {128 t + 32 q + s} for all four quarters q
// and a run of slots s: the scales it needs are then whole contiguous bytes of each tile, and the
// codes four contiguous runs. Bytes are staged into shared memory in 16-byte pieces over the runs
// widened to 16-byte boundaries (matrices inside a record are only 4-byte aligned in general):
// cp.async from device slots, plain loads held in registers from mapped system RAM (cp.async
// reads of mapped host memory measured ~6x slower; there every PCIe transaction counts).
//
// Decode and verify widths (rows <= 16): activations stay exact (f32, ActQuant::Exact). One row:
// gate and up are one GEMV per (row, expert) pair, SwiGLU fused; down runs per token row over its
// experts (deterministic weighted sum). 2..16 rows: the same GEMVs run per routed expert over the
// pairs that chose it (an expert shared by several rows of a verify window is read once), down
// writes one output row per pair and a final pass sums them per token in order. Codes become f32
// by bit construction, scales apply per 16; the kernels are bound by the weight stream.
//
// Prefill widths: pairs are grouped by expert (each expert's weights are read once per call) and
// multiplied on bf16 tensor cores (mma.m16n8k16, f32 accumulate) with activations rounded to bf16
// (ActQuant::Bf16): code x scale is bf16-exact, the divisor is applied to the f32 result.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#include "velox_common.cuh"

namespace {

struct NvDesc {
    const uint8_t* records;
    const uint8_t* slots;
    const float* map;  // i32 bits: slot of expert e, or -1
    long long record_bytes, slot_bytes;
    long long off[3];  // gate, up, down matrix offsets inside a record
    int width, hidden, experts;
};

__device__ __forceinline__ const uint8_t* nv_record(const NvDesc& d, int e, bool& device) {
    const int slot = static_cast<int>(__float_as_uint(__ldg(d.map + e)));
    device = slot >= 0;
    return slot >= 0 ? d.slots + static_cast<long long>(slot) * d.slot_bytes
                     : d.records + static_cast<long long>(e) * d.record_bytes;
}

__device__ __forceinline__ const uint8_t* nv_record_ptr(const NvDesc& d, int e) {
    bool dev;
    return nv_record(d, e, dev);
}

__host__ __device__ __forceinline__ long long nv_align256(long long x) { return (x + 255) / 256 * 256; }

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_u32(dst)), "l"(src));
}
// Zero-fills the 16 bytes instead of reading when !valid.
__device__ __forceinline__ void cp16z(void* dst, const void* src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src),
                 "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Copies 16-byte pieces i = tid, tid + nt, ... < total; at(i, dst, src) names piece i. Device
// memory goes through cp.async (the caller commits and waits); mapped system RAM through plain
// 16-byte loads, a batch held in registers before the stores: cp.async reads of mapped host memory
// stream ~6x slower over PCIe (2.9 vs 17.8 GB/s measured at decode).
template <class At>
__device__ __forceinline__ void copy16(int total, int tid, int nt, bool device, At at) {
    if (device) {
        for (int i = tid; i < total; i += nt) {
            uint8_t* dst;
            const uint8_t* src;
            at(i, dst, src);
            cp16(dst, src);
        }
        return;
    }
    constexpr int B = 8;
    for (int i0 = tid; i0 < total; i0 += B * nt) {
        uint4 v[B];
#pragma unroll
        for (int b = 0; b < B; ++b) {
            uint8_t* dst;
            const uint8_t* src;
            if (i0 + b * nt < total) {
                at(i0 + b * nt, dst, src);
                v[b] = *reinterpret_cast<const uint4*>(src);
            }
        }
#pragma unroll
        for (int b = 0; b < B; ++b) {
            uint8_t* dst;
            const uint8_t* src;
            if (i0 + b * nt < total) {
                at(i0 + b * nt, dst, src);
                *reinterpret_cast<uint4*>(dst) = v[b];
            }
        }
    }
}

// Four E2M1 codes (code i in bits 4i..4i+3) -> bf16 pairs (codes 0,1) and (2,3): the magnitudes
// 0, 0.5, 1, 1.5, 2, 3, 4, 6 are bf16 0x0000, 3F00, 3F80, 3FC0, 4000, 4040, 4080, 40C0; two
// byte-permute tables give their high and low bytes, then the sign bits are or-ed in.
__device__ __forceinline__ void e2m1x4_bf16(uint32_t c, uint32_t& p01, uint32_t& p23) {
    const uint32_t sel = c & 0x7777;
    const uint32_t hi = __byte_perm(0x3F3F3F00u, 0x40404040u, sel);
    const uint32_t lo = __byte_perm(0xC0800000u, 0xC0804000u, sel);
    p01 = __byte_perm(lo, hi, 0x5140) | ((c & 0x8u) << 12) | ((c & 0x80u) << 24);
    p23 = __byte_perm(lo, hi, 0x7362) | ((c & 0x800u) << 4) | ((c & 0x8000u) << 16);
}

__device__ __forceinline__ float e4m3(uint8_t b) {
    const __half_raw h = __nv_cvt_fp8_to_halfraw(b, __NV_E4M3);
    return __half2float(__half(h));
}

// sum_i code_i * x_i over 8 codes (one u32, code i in bits 4i..) and x0 = x[0..3], x1 = x[4..7].
__device__ __forceinline__ float dot8(uint32_t w, const float4& x0, const float4& x1) {
    uint32_t a, b, c, e;
    e2m1x4_bf16(w & 0xffff, a, b);
    e2m1x4_bf16(w >> 16, c, e);
    float s = __uint_as_float(a << 16) * x0.x;
    s = fmaf(__uint_as_float(a & 0xffff0000u), x0.y, s);
    s = fmaf(__uint_as_float(b << 16), x0.z, s);
    s = fmaf(__uint_as_float(b & 0xffff0000u), x0.w, s);
    s = fmaf(__uint_as_float(c << 16), x1.x, s);
    s = fmaf(__uint_as_float(c & 0xffff0000u), x1.y, s);
    s = fmaf(__uint_as_float(e << 16), x1.z, s);
    return fmaf(__uint_as_float(e & 0xffff0000u), x1.w, s);
}

// ---- Decode and verify widths: exact activations ------------------------------------------

constexpr int kDecThreads = 256;  // warp w: quarter w % 4, half w / 4 of the work
constexpr int kDownBatch = 8;     // experts staged at once by the per-token down kernel

// Shared-memory geometry of R slots of one [n, K] matrix: four code runs (one per quarter), one
// scale run per k64 tile, then the 16 bytes holding the divisor; runs keep the source's lead.
struct DecGeo {
    int cs, ss, kt, bytes;
    __host__ __device__ DecGeo(int K, int R) {
        cs = R * K / 2 + 16;
        ss = R * 16 + 16;
        kt = K / 64;
        bytes = 4 * cs + kt * ss + 16;
    }
};

// Staging plan of slots [s0, s0 + R) of row tile t of the [n, K] matrix whose 16-byte rounded-down
// base is `a` (`lead` bytes before the matrix): 16-byte pieces of the four code runs, the scale
// runs and the divisor.
struct DecPlan {
    const uint8_t* a;
    int lead, pc, ps, pieces;
    long long half, so;
    __device__ DecPlan(const uint8_t* base, const DecGeo& g, int n, int K, int R) {
        lead = static_cast<int>(reinterpret_cast<uintptr_t>(base) & 15);
        a = base - lead;
        pc = (lead + R * K / 2 + 15) / 16;
        ps = (lead + R * 16 + 15) / 16;
        pieces = 4 * pc + g.kt * ps + 1;
        half = K / 2;
        so = nv_align256(static_cast<long long>(n) * half);
    }
    __device__ __forceinline__ void piece(int i, uint8_t* sm, const DecGeo& g, int n, int t, int s0, uint8_t*& dst,
                                          const uint8_t*& src) const {
        if (i < 4 * pc) {
            const int r = i / pc, c = i % pc;
            dst = sm + r * g.cs + 16 * c;
            src = a + (static_cast<long long>(t) * 128 + r * 32 + s0) * half + 16 * c;
        } else if ((i -= 4 * pc) < g.kt * ps) {
            const int r = i / ps, c = i % ps;
            dst = sm + 4 * g.cs + r * g.ss + 16 * c;
            src = a + so + (static_cast<long long>(t) * g.kt + r) * 512 + s0 * 16 + 16 * c;
        } else {  // the divisor sits `lead` bytes into this piece (n * K / 16 is a multiple of 16)
            dst = sm + 4 * g.cs + g.kt * g.ss;
            src = a + so + n * half / 8;
        }
    }
    __device__ __forceinline__ float divisor(const uint8_t* sm, const DecGeo& g) const {
        return *reinterpret_cast<const float*>(sm + 4 * g.cs + g.kt * g.ss + lead);
    }
};

// Warp q's dot products of its R rows (slots s0.. of quarter q) of a staged matrix with x (f32,
// K values), lane-partial (not yet reduced): acc[i] += sum over the lane's 8-value chunks.
template <int R>
__device__ __forceinline__ void dec_dot(const uint8_t* sm, const DecGeo& g, int lead, int K, int q, int lane,
                                        const float* __restrict__ x, float (&acc)[R]) {
    const uint8_t* codes = sm + q * g.cs + lead;
    const uint8_t* scales = sm + 4 * g.cs + lead + q * 4;
    for (int c = lane; c < K / 8; c += VELOX_WARP) {
        const float4 x0 = __ldg(reinterpret_cast<const float4*>(x + 8 * c));
        const float4 x1 = __ldg(reinterpret_cast<const float4*>(x + 8 * c + 4));
        const int grp = c >> 1;
        const uint8_t* sp = scales + (grp >> 2) * g.ss + (grp & 3);
#pragma unroll
        for (int i = 0; i < R; ++i) {
            const uint32_t w = *reinterpret_cast<const uint32_t*>(codes + i * (K / 2) + 4 * c);
            acc[i] = fmaf(dot8(w, x0, x1), e4m3(sp[i * 16]), acc[i]);
        }
    }
}

// h[p][r] = silu(gate_r . x[p / k]) * (up_r . x[p / k]), f32. G = false: block per pair p
// (expert ids[p]); G = true: block per expert, over the pairs routed to it (offsets / list), so
// an expert's rows are read once however many rows of the window chose it.
template <int R, bool G>
__global__ void __launch_bounds__(kDecThreads)
    nv_gate_up_kernel(NvDesc d, const float* __restrict__ ids, const int* __restrict__ offsets,
                      const int* __restrict__ list, int k, const float* __restrict__ x, float* __restrict__ h) {
    extern __shared__ __align__(16) uint8_t sm[];
    const int K = d.hidden, n = d.width, per_tile = 32 / R;
    const int t = blockIdx.x / per_tile, s0 = (blockIdx.x % per_tile) * R;
    __shared__ float red[2][2][4 * R];  // [pair parity][gate, up][row]
    const int lane = threadIdx.x % VELOX_WARP, q = (threadIdx.x / VELOX_WARP) % 4, m = threadIdx.x / (4 * VELOX_WARP);
    int e, begin = 0, count = 1;
    if (G) {
        const int* act = offsets + d.experts + 1;
        const int gi = blockIdx.y, nd = act[0];
        if (gi >= nd + act[1]) return;
        e = act[gi < act[1] ? 2 + d.experts + gi : 2 + gi - act[1]];
        begin = offsets[e];
        count = offsets[e + 1] - begin;
        if (count == 0) return;
    } else {
        const unsigned u = __float_as_uint(ids[blockIdx.y]);
        if (u >= static_cast<unsigned>(d.experts)) {  // not routed (portable skips such ids too)
            if (threadIdx.x < 4 * R)
                h[static_cast<long long>(blockIdx.y) * n + t * 128 + threadIdx.x / R * 32 + s0 + threadIdx.x % R] = 0.0f;
            return;
        }
        e = static_cast<int>(u);
    }
    const DecGeo g(K, R);
    bool dev;
    const uint8_t* rec = nv_record(d, e, dev);
    const DecPlan pg(rec + d.off[0], g, n, K, R), pu(rec + d.off[1], g, n, K, R);
    copy16(pg.pieces + pu.pieces, threadIdx.x, blockDim.x, dev, [&](int i, uint8_t*& dst, const uint8_t*& src) {
        if (i < pg.pieces) pg.piece(i, sm, g, n, t, s0, dst, src);
        else pu.piece(i - pg.pieces, sm + g.bytes, g, n, t, s0, dst, src);
    });
    cp_commit();
    cp_wait<0>();
    __syncthreads();
    // Warps 0-3 take gate, 4-7 up; the SwiGLU meets in shared memory.
    const float dm = m ? pu.divisor(sm + g.bytes, g) : pg.divisor(sm, g);
    for (int i = 0; i < count; ++i) {
        const int p = G ? list[begin + i] : blockIdx.y;
        float a[R] = {};
        dec_dot<R>(sm + m * g.bytes, g, m ? pu.lead : pg.lead, K, q, lane, x + static_cast<long long>(p / k) * K, a);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float v = velox_warp_sum(a[r]);
            if (lane == r) red[i & 1][m][q * R + r] = v / dm;
        }
        __syncthreads();
        if (threadIdx.x < 4 * R) {
            const float gv = red[i & 1][0][threadIdx.x], uv = red[i & 1][1][threadIdx.x];
            h[static_cast<long long>(p) * n + t * 128 + threadIdx.x / R * 32 + s0 + threadIdx.x % R] =
                gv / (1.0f + __expf(-gv)) * uv;
        }
    }
}

// Single decode rows: y[tok][c] = sum_j w[p] * (down_c . h[p]) over the token's pairs
// p = tok * k + j in order (deterministic), up to kDownBatch experts' rows staged at once.
template <int R>
__global__ void __launch_bounds__(kDecThreads)
    nv_down_kernel(NvDesc d, const float* __restrict__ ids, const float* __restrict__ wts, int k,
                   const float* __restrict__ h, float* __restrict__ y) {
    extern __shared__ __align__(16) uint8_t sm[];
    __shared__ const uint8_t* sbase[kDownBatch];
    __shared__ int sidx[2][kDownBatch], scnt[2];
    const int K = d.width, n = d.hidden, per_tile = 32 / R;
    const int tok = blockIdx.y, t = blockIdx.x / per_tile, s0 = (blockIdx.x % per_tile) * R;
    const int lane = threadIdx.x % VELOX_WARP, q = (threadIdx.x / VELOX_WARP) % 4, m = threadIdx.x / (4 * VELOX_WARP);
    __shared__ float part1[4 * R];
    const DecGeo g(K, R);
    float acc[R] = {};
    for (int j0 = 0; j0 < k; j0 += kDownBatch) {
        const int nb = min(kDownBatch, k - j0);
        __syncthreads();  // the previous batch is consumed
        if (threadIdx.x < VELOX_WARP) {  // lane b resolves expert b; ballots list them by memory
            const int b = threadIdx.x;
            const unsigned e = b < nb ? __float_as_uint(ids[tok * k + j0 + b]) : ~0u;
            const bool ok = e < static_cast<unsigned>(d.experts);
            bool dev = false;
            const uint8_t* base = ok ? nv_record(d, static_cast<int>(e), dev) + d.off[2] : nullptr;
            if (b < nb) sbase[b] = base;
            const unsigned below = (1u << b) - 1;
#pragma unroll
            for (int dv = 0; dv < 2; ++dv) {
                const unsigned m = __ballot_sync(0xffffffffu, ok && dev == (dv == 1));
                if (ok && dev == (dv == 1)) sidx[dv][__popc(m & below)] = b;
                if (b == 0) scnt[dv] = __popc(m);
            }
        }
        __syncthreads();
        // Device slots through cp.async, then the system RAM experts in one batch of loads.
        // Every expert's matrix has the same lead (records and slots are 16-byte multiples).
        if (scnt[0] + scnt[1] == 0) continue;
        const DecPlan any(sbase[sidx[scnt[1] ? 1 : 0][0]], g, n, K, R);
        for (int dv = 1; dv >= 0; --dv) {
            copy16(scnt[dv] * any.pieces, threadIdx.x, blockDim.x, dv, [&](int i, uint8_t*& dst, const uint8_t*& src) {
                const int b = sidx[dv][i / any.pieces];
                const DecPlan pl(sbase[b], g, n, K, R);
                pl.piece(i % any.pieces, sm + b * g.bytes, g, n, t, s0, dst, src);
            });
        }
        cp_commit();
        cp_wait<0>();
        __syncthreads();
        for (int b = m; b < nb; b += 2) {  // warps 0-3 even experts, 4-7 odd ones
            if (!sbase[b]) continue;
            const int p = tok * k + j0 + b;
            const uint8_t* s = sm + b * g.bytes;
            float part[R] = {};
            dec_dot<R>(s, g, any.lead, K, q, lane, h + static_cast<long long>(p) * K, part);
            const float wd = __ldg(wts + p) / any.divisor(s, g);
#pragma unroll
            for (int i = 0; i < R; ++i) acc[i] = fmaf(wd, part[i], acc[i]);
        }
    }
    float sum[R];
#pragma unroll
    for (int i = 0; i < R; ++i) {
        sum[i] = velox_warp_sum(acc[i]);
        if (m == 1 && lane == i) part1[q * R + i] = sum[i];
    }
    __syncthreads();
    float* yp = y + static_cast<long long>(tok) * n + t * 128 + q * 32 + s0;
#pragma unroll
    for (int i = 0; i < R; ++i)
        if (m == 0 && lane == i) yp[i] = sum[i] + part1[q * R + i];
}

// Small verify windows (2..4 rows): o[p][c] = down_c . h[p] for the pairs routed to expert blockIdx.y (each
// expert's rows read once); nv_sum_kernel weights and sums them per token in order.
template <int R>
__global__ void __launch_bounds__(kDecThreads)
    nv_down_grouped_kernel(NvDesc d, const int* __restrict__ offsets, const int* __restrict__ list,
                           const float* __restrict__ h, float* __restrict__ o) {
    extern __shared__ __align__(16) uint8_t sm[];
    const int K = d.width, n = d.hidden, per_tile = 32 / R;
    const int* act = offsets + d.experts + 1;
    const int gi = blockIdx.y, nd = act[0];
    if (gi >= nd + act[1]) return;
    const int e = act[gi < act[1] ? 2 + d.experts + gi : 2 + gi - act[1]], begin = offsets[e], count = offsets[e + 1] - begin;
    const int t = blockIdx.x / per_tile, s0 = (blockIdx.x % per_tile) * R;
    const int lane = threadIdx.x % VELOX_WARP, q = (threadIdx.x / VELOX_WARP) % 4, m = threadIdx.x / (4 * VELOX_WARP);
    const DecGeo g(K, R);
    bool dev;
    const DecPlan pl(nv_record(d, e, dev) + d.off[2], g, n, K, R);
    copy16(pl.pieces, threadIdx.x, blockDim.x, dev,
           [&](int i, uint8_t*& dst, const uint8_t*& src) { pl.piece(i, sm, g, n, t, s0, dst, src); });
    cp_commit();
    cp_wait<0>();
    __syncthreads();
    const float dv = pl.divisor(sm, g);
    for (int i = m; i < count; i += 2) {  // warps 0-3 even pairs, 4-7 odd ones
        const int p = list[begin + i];
        float part[R] = {};
        dec_dot<R>(sm, g, pl.lead, K, q, lane, h + static_cast<long long>(p) * K, part);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float v = velox_warp_sum(part[r]);
            if (lane == r) o[static_cast<long long>(p) * n + t * 128 + q * 32 + s0 + r] = v / dv;
        }
    }
}

// ---- Prefill widths: grouped by expert on bf16 tensor cores --------------------------------

// One block: counts per expert, exclusive offsets (offsets[experts] = routed pairs), the pairs
// listed by expert, and after the offsets the routed experts by placement: act = offsets +
// experts + 1 holds act[0] in device slots at act[2 ..], act[1] in system RAM at act[2 + experts
// ..]. Orders within an expert and among experts are arbitrary; every pair's result is
// independent of them. Grids over the routed experts take the system RAM ones first, so their
// PCIe streams start early and overlap the device slots' blocks.
__global__ void __launch_bounds__(1024)
    nv_sort_kernel(const float* __restrict__ ids, const float* __restrict__ map, int pairs, int experts,
                   int* __restrict__ offsets, int* __restrict__ list) {
    extern __shared__ int cnt[];  // counts, then cursors
    __shared__ int wsum[32];
    __shared__ int nact[2];
    int* cur = cnt + experts;
    int* act = offsets + experts + 1;
    if (threadIdx.x < 2) nact[threadIdx.x] = 0;
    for (int e = threadIdx.x; e < experts; e += blockDim.x) cnt[e] = 0;
    __syncthreads();
    for (int p = threadIdx.x; p < pairs; p += blockDim.x) {
        const unsigned e = __float_as_uint(ids[p]);
        if (e < static_cast<unsigned>(experts)) atomicAdd(cnt + e, 1);
    }
    __syncthreads();
    // Thread t sums a contiguous chunk; a block scan of the chunk sums gives each chunk's start.
    const int per = (experts + blockDim.x - 1) / blockDim.x, e0 = threadIdx.x * per;
    int s = 0;
    for (int i = 0; i < per && e0 + i < experts; ++i) s += cnt[e0 + i];
    const int lane = threadIdx.x % VELOX_WARP, warp = threadIdx.x / VELOX_WARP;
    int incl = s;
#pragma unroll
    for (int o = 1; o < VELOX_WARP; o <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, incl, o);
        if (lane >= o) incl += v;
    }
    if (lane == VELOX_WARP - 1) wsum[warp] = incl;
    __syncthreads();
    if (warp == 0) {
        const int nw = blockDim.x / VELOX_WARP;
        int v = lane < nw ? wsum[lane] : 0;
#pragma unroll
        for (int o = 1; o < VELOX_WARP; o <<= 1) {
            const int u = __shfl_up_sync(0xffffffffu, v, o);
            if (lane >= o) v += u;
        }
        if (lane < nw) wsum[lane] = v;  // inclusive warp totals
    }
    __syncthreads();
    int acc = incl - s + (warp > 0 ? wsum[warp - 1] : 0);
    for (int i = 0; i < per && e0 + i < experts; ++i) {
        offsets[e0 + i] = acc;
        cur[e0 + i] = acc;
        if (cnt[e0 + i] > 0) {
            const bool dev = static_cast<int>(__float_as_uint(__ldg(map + e0 + i))) >= 0;
            act[2 + (dev ? 0 : experts) + atomicAdd(&nact[dev ? 0 : 1], 1)] = e0 + i;
        }
        acc += cnt[e0 + i];
    }
    if (threadIdx.x == blockDim.x - 1) offsets[experts] = wsum[(blockDim.x - 1) / VELOX_WARP];
    __syncthreads();
    if (threadIdx.x < 2) act[threadIdx.x] = nact[threadIdx.x];
    for (int p = threadIdx.x; p < pairs; p += blockDim.x) {
        const unsigned e = __float_as_uint(ids[p]);
        if (e < static_cast<unsigned>(experts)) list[atomicAdd(cur + e, 1)] = p;
    }
}

// NVFP4 K order within a 32-value chunk: MMA position l = 16s + 8h + 2q + j (step s, fragment
// half h, lane quad q, element j) reads weight code c = 8q + 4s + 2h + j, so a lane's B values for
// a chunk's two k16 steps are its 8 consecutive codes (one u32, one 16-group).
__host__ __device__ __forceinline__ int nv_perm(int l) {
    const int s = (l >> 4) & 1, h = (l >> 3) & 1, q = (l >> 1) & 3, j = l & 1;
    return (l & ~31) + 8 * q + 4 * s + 2 * h + j;
}
__host__ __device__ __forceinline__ int nv_unperm(int c) {
    const int q = (c >> 3) & 3, s = (c >> 2) & 1, h = (c >> 1) & 1, j = c & 1;
    return (c & ~31) + 16 * s + 8 * h + 2 * q + j;
}

// x [rows, K] f32 -> bf16 (RNE) in nv_perm column order.
__global__ void nv_to_bf16_perm(const float* __restrict__ x, long long n, int K, __nv_bfloat16* __restrict__ y) {
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const long long r = i / K;
        y[i] = __float2bfloat16_rn(x[r * K + nv_perm(static_cast<int>(i % K))]);
    }
}

constexpr int kKS = 128;      // K values per pipeline stage
constexpr int kStages = 2;  // 2 keeps two blocks per SM (3: one, 30% slower resident)
constexpr int kTok = 32;  // routed pairs per pass (m16 tiles of tokens)
constexpr int kMT = kTok / 16;
constexpr int kRowB = 80;     // staged code bytes per weight row: 64 + 16 for the lead
constexpr int kScB = 272;     // staged scale bytes per k64 tile: 256 + 16
constexpr int kSA = kKS + 8;  // activation row stride (bf16): ldmatrix rows in distinct banks

template <int NM>
struct MmaSmem {
    __nv_bfloat16 a[kStages][kTok * kSA];
    uint8_t w[kStages][NM][64 * kRowB];
    uint8_t s[kStages][NM][2 * kScB];
    float div[NM];
};

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

// NM = 2: gate and up of expert blockIdx.y against x rows (act rows p / k), h[p] = silu(g) * u as
// bf16 in nv_perm order (down's activations). NM = 1: down against h rows p, o[p][r] f32.
// A block owns rows {128 t + 32 q + 16 j + i} (warp q, i < 16) of an [n, K] matrix.
template <int NM>
__global__ void __launch_bounds__(128)
    nv_mma_kernel(NvDesc d, int n, int K, const int* __restrict__ offsets, const int* __restrict__ list, int k,
                  const __nv_bfloat16* __restrict__ act, __nv_bfloat16* __restrict__ hb, float* __restrict__ o) {
    extern __shared__ __align__(128) uint8_t smem_raw[];
    auto& S = *reinterpret_cast<MmaSmem<NM>*>(smem_raw);
    const int* routed = offsets + d.experts + 1;
    const int gi = blockIdx.y, nd = routed[0];
    if (gi >= nd + routed[1]) return;
    const int e = routed[gi < routed[1] ? 2 + d.experts + gi : 2 + gi - routed[1]], begin = offsets[e], count = offsets[e + 1] - begin;
    const int t = blockIdx.x / 2, j = blockIdx.x % 2;
    const int lane = threadIdx.x % VELOX_WARP, warp = threadIdx.x / VELOX_WARP, g = lane / 4, q4 = lane % 4;
    bool dev;
    const uint8_t* rec = nv_record(d, e, dev);
    const long long half = K / 2, so = nv_align256(static_cast<long long>(n) * half);
    const int kt = K / 64, steps = K / kKS;
    const uint8_t* base[NM];
    int lead[NM];
#pragma unroll
    for (int m = 0; m < NM; ++m) {
        base[m] = rec + d.off[NM == 2 ? m : 2];
        lead[m] = static_cast<int>(reinterpret_cast<uintptr_t>(base[m]) & 15);
    }
    if (threadIdx.x < NM) S.div[threadIdx.x] = *reinterpret_cast<const float*>(base[threadIdx.x] + so + n * half / 8);

    for (int tg = 0; tg < count; tg += kTok) {
        const int nt = min(kTok, count - tg), ar = (nt + 15) / 16 * 16;
        auto load = [&](int stage, int ks) {
            // Activations: nt gathered rows x 128 bf16 (16 pieces each); rows past nt (up to the
            // m16 tiles in use) zero-filled.
            for (int i = threadIdx.x; i < ar * 16; i += 128) {
                const int r = i / 16, c = i % 16;
                const bool ok = r < nt;
                const int p = ok ? list[begin + tg + r] : 0;
                const long long row = NM == 2 ? p / k : p;
                cp16z(&S.a[stage][r * kSA + c * 8], act + row * K + ks * kKS + c * 8, ok);
            }
#pragma unroll
            for (int m = 0; m < NM; ++m) {
                // Codes: 4 quarters x 16 rows, 64 bytes each at this step.
                const uint8_t* a = base[m] - lead[m] + ks * 64;
                const int pieces = lead[m] ? 5 : 4;
                copy16(64 * pieces, threadIdx.x, 128, dev, [&](int i, uint8_t*& dst, const uint8_t*& src) {
                    const int r = i / pieces, c = i % pieces;
                    const long long row = static_cast<long long>(t) * 128 + (r / 16) * 32 + j * 16 + r % 16;
                    dst = &S.w[stage][m][r * kRowB + 16 * c];
                    src = a + row * half + 16 * c;
                });
                // Scales: the step's two k64 tiles, slots 16 j .. 16 j + 15 (256 bytes each).
                const uint8_t* sa = base[m] - lead[m] + so + (static_cast<long long>(t) * kt + ks * 2) * 512 + j * 256;
                const int sp = lead[m] ? 17 : 16;
                copy16(2 * sp, threadIdx.x, 128, dev, [&](int i, uint8_t*& dst, const uint8_t*& src) {
                    const int tile = i / sp, c = i % sp;
                    dst = &S.s[stage][m][tile * kScB + 16 * c];
                    src = sa + tile * 512 + 16 * c;
                });
            }
        };
        __syncthreads();  // the previous pass is done with every stage
#pragma unroll
        for (int st = 0; st < kStages - 1; ++st) {
            if (st < steps) load(st, st);
            cp_commit();
        }
        float acc[NM][kMT][2][4];
#pragma unroll
        for (int m = 0; m < NM; ++m)
#pragma unroll
            for (int mt = 0; mt < kMT; ++mt)
#pragma unroll
                for (int bt = 0; bt < 2; ++bt) acc[m][mt][bt][0] = acc[m][mt][bt][1] = acc[m][mt][bt][2] = acc[m][mt][bt][3] = 0.0f;
        const int a_row = (lane & 7) + (((lane >> 3) & 1) << 3), a_col = (lane >> 4) << 3;
        for (int i = 0; i < steps; ++i) {
            cp_wait<kStages - 2>();
            __syncthreads();
            const int next = i + kStages - 1;
            if (next < steps) load(next % kStages, next);
            cp_commit();
            const int st = i % kStages;
#pragma unroll
            for (int ch = 0; ch < kKS / 32; ++ch) {
                uint32_t af[kMT][2][4];
#pragma unroll
                for (int mt = 0; mt < kMT; ++mt) {
                    if (mt * 16 >= nt) break;
                    ldsm_x4(af[mt][0], &S.a[st][(mt * 16 + a_row) * kSA + ch * 32 + a_col]);
                    ldsm_x4(af[mt][1], &S.a[st][(mt * 16 + a_row) * kSA + ch * 32 + 16 + a_col]);
                }
                const int gi = ch * 2 + (q4 >> 1);  // 16-group within the step
#pragma unroll
                for (int m = 0; m < NM; ++m) {
#pragma unroll
                    for (int bt = 0; bt < 2; ++bt) {
                        const int r = warp * 16 + bt * 8 + g;  // staged row (quarter warp)
                        const uint32_t w =
                            *reinterpret_cast<const uint32_t*>(&S.w[st][m][r * kRowB + lead[m] + ch * 16 + 4 * q4]);
                        const uint8_t sb = S.s[st][m][(gi >> 2) * kScB + lead[m] + (bt * 8 + g) * 16 + warp * 4 + (gi & 3)];
                        const __nv_bfloat16 sc = __float2bfloat16_rn(e4m3(sb));
                        const __nv_bfloat162 sc2 = __halves2bfloat162(sc, sc);
                        uint32_t v[4];
                        e2m1x4_bf16(w & 0xffff, v[0], v[1]);
                        e2m1x4_bf16(w >> 16, v[2], v[3]);
#pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const __nv_bfloat162 xv = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&v[u]), sc2);
                            v[u] = *reinterpret_cast<const uint32_t*>(&xv);
                        }
#pragma unroll
                        for (int mt = 0; mt < kMT; ++mt) {
                            if (mt * 16 >= nt) break;
                            mma_bf16(acc[m][mt][bt], af[mt][0], v[0], v[1]);
                            mma_bf16(acc[m][mt][bt], af[mt][1], v[2], v[3]);
                        }
                    }
                }
            }
        }
        cp_wait<0>();
        // C: c0, c1 = token g, weight rows 2 q4, 2 q4 + 1 of the n8 tile; c2, c3 = token g + 8.
#pragma unroll
        for (int mt = 0; mt < kMT; ++mt) {
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const int tk = mt * 16 + g + 8 * hh;
                if (tk >= nt) continue;
                const long long p = list[begin + tg + tk];
#pragma unroll
                for (int bt = 0; bt < 2; ++bt) {
#pragma unroll
                    for (int u = 0; u < 2; ++u) {
                        const int r = t * 128 + warp * 32 + j * 16 + bt * 8 + 2 * q4 + u;
                        const float v0 = acc[0][mt][bt][2 * hh + u] / S.div[0];
                        if constexpr (NM == 2) {
                            const float v1 = acc[NM - 1][mt][bt][2 * hh + u] / S.div[NM - 1];
                            hb[p * n + nv_unperm(r)] = __float2bfloat16_rn(v0 / (1.0f + __expf(-v0)) * v1);
                        } else {
                            o[p * n + r] = v0;
                        }
                    }
                }
            }
        }
    }
}

__global__ void nv_sum_kernel(const float* __restrict__ o, const float* __restrict__ w, int rows, int k, int hidden,
                              const float* __restrict__ ids, int experts, float* __restrict__ y) {
    const long long n = static_cast<long long>(rows) * hidden;
    for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const long long r = i / hidden, c = i % hidden;
        float acc = 0.0f;
        for (int j = 0; j < k; ++j) {
            if (__float_as_uint(ids[r * k + j]) >= static_cast<unsigned>(experts)) continue;
            acc += w[r * k + j] * o[(r * k + j) * hidden + c];
        }
        y[i] = acc;
    }
}

template <int NM>
int mma_launch(const NvDesc& d, int groups, int n, int K, const int* offsets, const int* list, int k, const __nv_bfloat16* act,
               __nv_bfloat16* hb, float* o) {
    static const cudaError_t attr = cudaFuncSetAttribute(nv_mma_kernel<NM>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                         static_cast<int>(sizeof(MmaSmem<NM>)));
    if (attr != cudaSuccess) return static_cast<int>(attr);
    const dim3 grid(n / 64, static_cast<unsigned>(groups));
    nv_mma_kernel<NM><<<grid, 128, sizeof(MmaSmem<NM>)>>>(d, n, K, offsets, list, k, act, hb, o);
    return velox_launch_status();
}

// ---- Verify windows (2..16 rows): persistent tensor-core streams over routed experts --------
//
// A job is (routed expert, 16 * MT weight rows); jobs of experts in device slots and of experts in
// system RAM form two lists (nv_sort_kernel). The first `gh` blocks take the system RAM jobs
// (plain loads; PCIe is the limit), the others the device jobs, so both memories stream at
// once. A job's K is cut into S stages that flow through an NBUF-deep cp.async ring: the next
// stages' bytes are in flight while the current one is multiplied, and every weight byte is read
// once for all the pairs routed to its expert (at most 16 per pass: k distinct experts per row).
//
// Weights are the mma A operand (16 rows), the expert's pairs the B operand (two n8 tiles). In
// nv_perm order a lane's A values for a 32-value chunk are its 8 consecutive codes (one u32, one
// 16-group), and its B values are the same 8 consecutive activations in natural order: one
// 16-byte load. Activations are split exactly into NSPLIT bf16 planes (x = hi + mid + lo for
// NSPLIT = 3, f32 products of bf16-exact code x scale weights accumulate in f32), so the result
// is the exact-activation product (ActQuant::Exact) at tensor-core cost.

// bf16 planes of x [n, K]: plane i at y + i * plane, x = sum of planes (exact for NSPLIT = 3).
template <int NSPLIT>
__device__ __forceinline__ void split_store(float v, __nv_bfloat16* y, long long plane) {
#pragma unroll
    for (int i = 0; i < NSPLIT; ++i) {
        const __nv_bfloat16 b = __float2bfloat16_rn(v);
        y[i * plane] = b;
        v -= __bfloat162float(b);
    }
}

template <int NSPLIT>
__global__ void nv_split_kernel(const float* __restrict__ x, long long n, __nv_bfloat16* __restrict__ y) {
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<long long>(gridDim.x) * blockDim.x)
        split_store<NSPLIT>(x[i], y + i, n);
}

constexpr int kVKw = 4;  // warps splitting a stage's K

template <int NM, int MT, int S>
struct VGeo {
    static constexpr int R = 4 * MT;  // slots per quarter: a job owns 16 * MT rows
    int ks, run, sc, mat, buf;
    __host__ __device__ VGeo(int K) {
        ks = K / S;               // values per stage
        run = ks / 2 + 16;        // one row's codes of a stage (+ lead)
        sc = R * 16 + 16;         // one k64 tile's scales of the job's slots (+ lead)
        mat = 4 * R * run + ks / 64 * sc + 16;  // + the divisor's 16 bytes
        buf = NM * mat;
    }
    // Shared memory: NBUF stage buffers, then the cross-warp reduction.
    __host__ __device__ int smem(int nbuf) const { return nbuf * buf + NM * MT * kVKw * 2 * 128 * 4; }
};

template <int NM, int MT, int S, int NSPLIT, int NBUF>
__global__ void __launch_bounds__(256, 2)
    nv_vmma_kernel(NvDesc d, int n, int K, const int* __restrict__ offsets, const int* __restrict__ list,
                   const int* __restrict__ act, int k, const __nv_bfloat16* __restrict__ xin, long long xplane,
                   __nv_bfloat16* __restrict__ hout, long long hplane, float* __restrict__ o, int gh) {
    using G = VGeo<NM, MT, S>;
    constexpr int R = G::R;
    extern __shared__ __align__(16) uint8_t sm[];
    const G geo(K);
    float* red = reinterpret_cast<float*>(sm + NBUF * geo.buf);
    const int lane = threadIdx.x % VELOX_WARP, w = threadIdx.x / VELOX_WARP, g = lane / 4, t4 = lane % 4;
    const int kw = w % kVKw, mt = (w / kVKw) % MT, m = w / (kVKw * MT);
    const int nd = act[0], nh = act[1];
    const int jpe = n / 128 * (32 / R);  // jobs per expert
    const bool split = nh > 0 && nd > 0 && static_cast<int>(gridDim.x) > gh;
    const bool host = nh > 0 && (nd == 0 || (split && static_cast<int>(blockIdx.x) < gh));
    const int role_blocks = !split ? gridDim.x : host ? gh : gridDim.x - gh;
    const int role_id = split && !host ? blockIdx.x - gh : blockIdx.x;
    const int jobs = (host ? nh : nd) * jpe;
    const int* elist = act + 2 + (host ? d.experts : 0);
    const long long half = K / 2, so = nv_align256(static_cast<long long>(n) * half);
    const int ktiles = K / 64, kts = geo.ks / 64;

    // Item iterator: job, token group of 16 pairs, stage.
    struct It {
        int job, tg, ks, e, begin, count;
    };
    auto load_job = [&](It& it) {
        if (it.job < jobs) {
            it.e = elist[it.job / jpe];
            it.begin = offsets[it.e];
            it.count = offsets[it.e + 1] - it.begin;
        }
    };
    auto next = [&](It& it) {
        if (++it.ks < S) return;
        it.ks = 0;
        if (++it.tg * 16 < it.count) return;
        it.tg = 0;
        it.job += role_blocks;
        load_job(it);
    };
    // Stages item `it` into `buf`: per matrix the 4R code runs of the stage (one row each, warps
    // over runs, lanes over 16-byte pieces), its k64 tiles' scale runs and, at stage 0, the divisor.
    auto stage = [&](const It& it, uint8_t* buf) {
        constexpr int RPW = 4 * R / 8;  // code runs per warp
        const int rg = it.job % jpe, tt = rg / (32 / R), s0 = (rg % (32 / R)) * R;
        const uint8_t* rec = nv_record_ptr(d, it.e);
#pragma unroll
        for (int mm = 0; mm < NM; ++mm) {
            const uint8_t* base = rec + d.off[NM == 2 ? mm : 2];
            const int lead = static_cast<int>(reinterpret_cast<uintptr_t>(base) & 15);
            const uint8_t* a = base - lead;
            const int pc = (lead + geo.ks / 2 + 15) / 16, ps = (lead + R * 16 + 15) / 16;
            uint8_t* mb = buf + mm * geo.mat;
            const uint8_t* src0 = a + (static_cast<long long>(tt) * 128 + s0) * half + static_cast<long long>(it.ks) * (geo.ks / 2);
            const uint8_t* ssrc = a + so + (static_cast<long long>(tt) * ktiles + it.ks * kts) * 512 + s0 * 16;
            uint8_t* sdst = mb + 4 * R * geo.run;
            if (!host) {
#pragma unroll
                for (int j = 0; j < RPW; ++j) {
                    const int r = w + 8 * j;
                    const uint8_t* src = src0 + ((r / R) * 32 + r % R) * half;
                    for (int c = lane; c < pc; c += VELOX_WARP) cp16(mb + r * geo.run + 16 * c, src + 16 * c);
                }
                for (int i = threadIdx.x; i < kts * ps; i += blockDim.x) {
                    const int r = i / ps, c = i % ps;
                    cp16(sdst + r * geo.sc + 16 * c, ssrc + r * 512 + 16 * c);
                }
                if (it.ks == 0 && threadIdx.x == blockDim.x - 1) cp16(sdst + kts * geo.sc, a + so + n * half / 8);
            } else {
                // System RAM: every load of the stage in flight before the stores.
                uint4 v[RPW][2];
#pragma unroll
                for (int j = 0; j < RPW; ++j) {
                    const int r = w + 8 * j;
                    const uint8_t* src = src0 + ((r / R) * 32 + r % R) * half;
#pragma unroll
                    for (int u = 0; u < 2; ++u)
                        if (lane + 32 * u < pc) v[j][u] = *reinterpret_cast<const uint4*>(src + 16 * (lane + 32 * u));
                }
                uint4 sv = make_uint4(0, 0, 0, 0), dv = sv;
                const int si = threadIdx.x;
                if (si < kts * ps) sv = *reinterpret_cast<const uint4*>(ssrc + si / ps * 512 + 16 * (si % ps));
                if (it.ks == 0 && threadIdx.x == blockDim.x - 1) dv = *reinterpret_cast<const uint4*>(a + so + n * half / 8);
#pragma unroll
                for (int j = 0; j < RPW; ++j) {
                    const int r = w + 8 * j;
#pragma unroll
                    for (int u = 0; u < 2; ++u)
                        if (lane + 32 * u < pc) *reinterpret_cast<uint4*>(mb + r * geo.run + 16 * (lane + 32 * u)) = v[j][u];
                }
                if (si < kts * ps) *reinterpret_cast<uint4*>(sdst + si / ps * geo.sc + 16 * (si % ps)) = sv;
                if (it.ks == 0 && threadIdx.x == blockDim.x - 1) *reinterpret_cast<uint4*>(sdst + kts * geo.sc) = dv;
            }
        }
    };

    float acc[2][4];
    float dvs[NM];
    auto compute = [&](const It& it, const uint8_t* buf) {
        const int lead_m = static_cast<int>(reinterpret_cast<uintptr_t>(nv_record_ptr(d, it.e) + d.off[NM == 2 ? m : 2]) & 15);
        if (it.ks == 0) {
#pragma unroll
            for (int mm = 0; mm < NM; ++mm) {
                const int l = static_cast<int>(reinterpret_cast<uintptr_t>(nv_record_ptr(d, it.e) + d.off[NM == 2 ? mm : 2]) & 15);
                dvs[mm] = *reinterpret_cast<const float*>(buf + mm * geo.mat + 4 * R * geo.run + kts * geo.sc + l);
            }
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;
        }
        const uint8_t* mb = buf + m * geo.mat;
        const int left = it.count - it.tg * 16, ntl = left > 8 ? 2 : 1;
        // Activation rows of this lane's two tokens (n-tile 0 and 1, token g), clamped.
        const __nv_bfloat16* xr[2];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int tk = min(nt * 8 + g, left - 1);
            const int p = list[it.begin + it.tg * 16 + tk];
            xr[nt] = xin + static_cast<long long>(NM == 2 ? p / k : p) * K + static_cast<long long>(it.ks) * geo.ks + 8 * t4;
        }
        // Rows mt * 16 + g (+ 8) of the job: quarter ri % 4, slot ri / 4.
        const int ri0 = mt * 16 + g, ri1 = ri0 + 8;
        const uint8_t* c0 = mb + ((ri0 % 4) * R + ri0 / 4) * geo.run + lead_m + 4 * t4;
        const uint8_t* c1 = mb + ((ri1 % 4) * R + ri1 / 4) * geo.run + lead_m + 4 * t4;
        const uint8_t* sbase = mb + 4 * R * geo.run + lead_m;
        const int so0 = (ri0 / 4) * 16 + (ri0 % 4) * 4, so1 = (ri1 / 4) * 16 + (ri1 % 4) * 4;
        const int chunks = geo.ks / 32 / kVKw;
#pragma unroll 2
        for (int ci = 0; ci < chunks; ++ci) {
            const int c = kw * chunks + ci;
            const int gl = c * 2 + (t4 >> 1);
            const uint8_t* sp = sbase + (gl >> 2) * geo.sc + (gl & 3);
            uint4 bx[2][NSPLIT];
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                for (int sp_i = 0; sp_i < NSPLIT; ++sp_i)
                    if (nt < ntl) bx[nt][sp_i] = __ldg(reinterpret_cast<const uint4*>(xr[nt] + sp_i * xplane + c * 32));
            uint32_t v0[4], v1[4];
            {
                const uint32_t w0 = *reinterpret_cast<const uint32_t*>(c0 + c * 16);
                const uint32_t w1 = *reinterpret_cast<const uint32_t*>(c1 + c * 16);
                e2m1x4_bf16(w0 & 0xffff, v0[0], v0[1]);
                e2m1x4_bf16(w0 >> 16, v0[2], v0[3]);
                e2m1x4_bf16(w1 & 0xffff, v1[0], v1[1]);
                e2m1x4_bf16(w1 >> 16, v1[2], v1[3]);
                const __nv_bfloat16 s0b = __float2bfloat16_rn(e4m3(sp[so0])), s1b = __float2bfloat16_rn(e4m3(sp[so1]));
                const __nv_bfloat162 sc0 = __halves2bfloat162(s0b, s0b), sc1 = __halves2bfloat162(s1b, s1b);
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const __nv_bfloat162 a = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&v0[u]), sc0);
                    const __nv_bfloat162 b = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&v1[u]), sc1);
                    v0[u] = *reinterpret_cast<const uint32_t*>(&a);
                    v1[u] = *reinterpret_cast<const uint32_t*>(&b);
                }
            }
            const uint32_t a_s0[4] = {v0[0], v1[0], v0[1], v1[1]};
            const uint32_t a_s1[4] = {v0[2], v1[2], v0[3], v1[3]};
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                if (nt >= ntl) break;
#pragma unroll
                for (int sp_i = 0; sp_i < NSPLIT; ++sp_i) {
                    mma_bf16(acc[nt], a_s0, bx[nt][sp_i].x, bx[nt][sp_i].y);
                    mma_bf16(acc[nt], a_s1, bx[nt][sp_i].z, bx[nt][sp_i].w);
                }
            }
        }
        if (it.ks != S - 1) return;
        // Sum the K warps' tiles, then the epilogue (one output element per thread and n-tile).
        float* rp = red + ((m * MT + mt) * kVKw + kw) * 2 * 128;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
            *reinterpret_cast<float4*>(rp + nt * 128 + lane * 4) = make_float4(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3]);
        __syncthreads();
        const int rg = it.job % jpe, tt = rg / (32 / R), s0 = (rg % (32 / R)) * R;
        for (int idx = threadIdx.x; idx < MT * 2 * 128; idx += blockDim.x) {
            const int emt = idx / 256, ent = (idx / 128) % 2, el = (idx / 4) % 32, ec = idx % 4;
            const int tk = ent * 8 + 2 * (el % 4) + (ec & 1);
            if (tk >= left) continue;
            const int ri = emt * 16 + el / 4 + (ec >= 2 ? 8 : 0);
            const int row = tt * 128 + (ri % 4) * 32 + s0 + ri / 4;
            const long long p = list[it.begin + it.tg * 16 + tk];
            float v[NM];
#pragma unroll
            for (int mm = 0; mm < NM; ++mm) {
                float s = 0.0f;
#pragma unroll
                for (int q = 0; q < kVKw; ++q) s += red[(((mm * MT + emt) * kVKw + q) * 2 + ent) * 128 + el * 4 + ec];
                v[mm] = s / dvs[mm];
            }
            if constexpr (NM == 2) split_store<NSPLIT>(v[0] / (1.0f + __expf(-v[0])) * v[1], hout + p * n + row, hplane);
            else o[p * n + row] = v[0];
        }
    };

    It cur{role_id, 0, 0, 0, 0, 0};
    load_job(cur);
    It pre = cur;
    for (int b = 0; b < NBUF - 1; ++b) {
        if (pre.job < jobs) {
            stage(pre, sm + b * geo.buf);
            next(pre);
        }
        cp_commit();
    }
    for (int slot = 0; cur.job < jobs; slot = (slot + 1) % NBUF) {
        if (pre.job < jobs) {
            stage(pre, sm + ((slot + NBUF - 1) % NBUF) * geo.buf);
            next(pre);
        }
        cp_commit();
        cp_wait<NBUF - 1>();
        __syncthreads();
        compute(cur, sm + slot * geo.buf);
        __syncthreads();
        next(cur);
    }
    cp_wait<0>();
}

long long nv_round(long long bytes) { return (bytes + 63) / 64 * 64; }

// Blocks of a persistent stream kernel resident at once (all SMs).
template <class F>
int stream_grid(F kernel, int smem) {
    int sms = 0, per = 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess) return 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, kernel, kDecThreads, smem) != cudaSuccess) return 0;
    return sms * per;
}

// System RAM streams: blocks of the verify kernels that take the system RAM experts' items.
constexpr int kHostBlocks = 24;

// bf16 planes of the verify activations: 1 (ActQuant::Bf16). 3 would be exact, ~1.6x slower.
constexpr int kSplit = 1;

template <int RU, int RD, int RG>
int dec_launch(const NvDesc& d, const float* ids, const float* wts, const float* x, float* y, uint8_t* p, int rows,
               int k, int mode, unsigned groups) {
    const int width = d.width, hidden = d.hidden, experts = d.experts, threads = 256;
    const long long pairs = static_cast<long long>(rows) * k, xs = static_cast<long long>(rows) * hidden;
    if (pairs > 65535) return VELOX_ERR_UNSUPPORTED;
    if (mode == 0) {
        const DecGeo gu(hidden, RU), dn(width, RD);
        const int up_smem = 2 * gu.bytes, down_smem = min(k, kDownBatch) * dn.bytes;
        if (up_smem > 96 * 1024 || down_smem > 96 * 1024) return VELOX_ERR_UNSUPPORTED;
        static const cudaError_t attr[2] = {
            cudaFuncSetAttribute(nv_gate_up_kernel<RU, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024),
            cudaFuncSetAttribute(nv_down_kernel<RD>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024)};
        for (const cudaError_t a : attr)
            if (a != cudaSuccess) return static_cast<int>(a);
        float* h = reinterpret_cast<float*>(p);
        nv_gate_up_kernel<RU, false><<<dim3(width / 128 * (32 / RU), static_cast<unsigned>(pairs)), kDecThreads, up_smem>>>(
            d, ids, nullptr, nullptr, k, x, h);
        nv_down_kernel<RD><<<dim3(hidden / 128 * (32 / RD), rows), kDecThreads, down_smem>>>(d, ids, wts, k, h, y);
        return velox_launch_status();
    }
    if (mode == 1) {  // small windows: the decode GEMVs once per routed expert, exact
        const DecGeo gu(hidden, RU), dg(width, RG);
        const int up_smem = 2 * gu.bytes;
        if (up_smem > 96 * 1024 || dg.bytes > 96 * 1024) return VELOX_ERR_UNSUPPORTED;
        static const cudaError_t attr[2] = {
            cudaFuncSetAttribute(nv_gate_up_kernel<RU, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024),
            cudaFuncSetAttribute(nv_down_grouped_kernel<RG>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024)};
        for (const cudaError_t a : attr)
            if (a != cudaSuccess) return static_cast<int>(a);
        float* h = reinterpret_cast<float*>(p);
        int* offsets = reinterpret_cast<int*>(p += nv_round(4 * pairs * width));
        int* list = reinterpret_cast<int*>(p += nv_round(4LL * (3 * experts + 3)));
        float* o = reinterpret_cast<float*>(p += nv_round(4 * pairs));
        nv_sort_kernel<<<1, 1024, 2 * experts * sizeof(int)>>>(ids, d.map, static_cast<int>(pairs), experts, offsets, list);
        nv_gate_up_kernel<RU, true><<<dim3(width / 128 * (32 / RU), groups), kDecThreads, up_smem>>>(d, ids, offsets, list,
                                                                                                    k, x, h);
        nv_down_grouped_kernel<RG><<<dim3(hidden / 128 * (32 / RG), groups), kDecThreads, dg.bytes>>>(d, offsets, list,
                                                                                                      h, o);
        nv_sum_kernel<<<velox_grid_1d(xs, threads), threads>>>(o, wts, rows, k, hidden, ids, experts, y);
        return velox_launch_status();
    }
    // Mode 3, verify windows: gate/up jobs of 16 rows in 2 K stages (2 buffers), down jobs of 32
    // rows in one stage (3 buffers), two blocks per SM.
    constexpr int kS1 = 2, kB1 = 2, kB2 = 3;
    auto* k1 = nv_vmma_kernel<2, 1, kS1, kSplit, kB1>;
    auto* k2 = nv_vmma_kernel<1, 2, 1, kSplit, kB2>;
    const int sm1 = VGeo<2, 1, kS1>(hidden).smem(kB1), sm2 = VGeo<1, 2, 1>(width).smem(kB2);
    // A stage's K is a multiple of 128 (four warps of 32-value chunks) and at most 2016 values (a
    // lane stages two 16-byte pieces of a code run).
    if (sm1 > 96 * 1024 || sm2 > 96 * 1024 || hidden % (128 * kS1) != 0 || hidden / kS1 > 2016 || width % 128 != 0 ||
        width > 2016)
        return VELOX_ERR_UNSUPPORTED;
    static const cudaError_t a1 = cudaFuncSetAttribute(k1, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
    static const cudaError_t a2 = cudaFuncSetAttribute(k2, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
    if (a1 != cudaSuccess) return static_cast<int>(a1);
    if (a2 != cudaSuccess) return static_cast<int>(a2);
    static const int g1 = stream_grid(k1, sm1), g2 = stream_grid(k2, sm2);
    if (g1 <= 0 || g2 <= 0) return VELOX_ERR_UNSUPPORTED;
    const int gh = kHostBlocks;
    const long long hw = pairs * width;
    auto* xb = reinterpret_cast<__nv_bfloat16*>(p);
    auto* hb = reinterpret_cast<__nv_bfloat16*>(p += nv_round(2LL * kSplit * xs));
    int* offsets = reinterpret_cast<int*>(p += nv_round(2LL * kSplit * hw));
    int* list = reinterpret_cast<int*>(p += nv_round(4LL * (3 * experts + 3)));
    float* o = reinterpret_cast<float*>(p += nv_round(4 * pairs));
    const int* act = offsets + experts + 1;
    nv_sort_kernel<<<1, 1024, 2 * experts * sizeof(int)>>>(ids, d.map, static_cast<int>(pairs), experts, offsets, list);
    nv_split_kernel<kSplit><<<velox_grid_1d(xs, threads), threads>>>(x, xs, xb);
    k1<<<g1, 256, sm1>>>(d, width, hidden, offsets, list, act, k, xb, xs, hb, hw, nullptr, gh);
    k2<<<g2, 256, sm2>>>(d, hidden, width, offsets, list, act, k, hb, hw, nullptr, 0, o, gh);
    nv_sum_kernel<<<velox_grid_1d(xs, threads), threads>>>(o, wts, rows, k, hidden, ids, experts, y);
    return velox_launch_status();
}

}  // namespace

// Bytes of scratch velox_5060_moe_nvfp4 needs in `mode` (see there).
VELOX_API long long velox_5060_moe_nvfp4_scratch(int rows, int k, int width, int hidden, int experts, int mode) {
    const long long pairs = static_cast<long long>(rows) * k;
    const long long route = nv_round(4LL * (3 * experts + 3)) + nv_round(4 * pairs);
    if (mode == 0) return nv_round(4 * pairs * width);
    if (mode == 1) return nv_round(4 * pairs * width) + route + nv_round(4 * pairs * hidden);
    if (mode == 3)
        return nv_round(2LL * kSplit * rows * hidden) + nv_round(2LL * kSplit * pairs * width) + route +
               nv_round(4 * pairs * hidden);
    return nv_round(2LL * rows * hidden) + nv_round(2 * pairs * width) + route + nv_round(4 * pairs * hidden);
}

// moe_experts for NVFP4 gate, up and down matrices at the given record offsets. mode 0: a GEMV
// per routed pair and a down pass per row (decode; exact activations); 1: the same kernels run
// once per routed expert over its pairs (small verify windows; exact); 3: persistent tensor-core
// streams over the routed experts (verify windows; bf16 activations); 2: grouped by expert on
// tensor cores (prefill; bf16 activations). VELOX_ERR_UNSUPPORTED for geometry the kernels do
// not cover (the caller falls back).
VELOX_API int velox_5060_moe_nvfp4(const float* x, const float* ids, const float* wts, const uint8_t* records,
                                   const uint8_t* slots, const float* map, long long record_bytes, long long slot_bytes,
                                   long long gate_off, long long up_off, long long down_off, int width, int hidden,
                                   int rows, int k, int experts, int mode, float* y, void* scratch) {
    if (rows < 1 || k < 1 || experts < 1 || experts > 6144 || mode < 0 || mode > 3 || width % 128 != 0 ||
        hidden % 128 != 0 || (gate_off | up_off | down_off) % 4 != 0 || (record_bytes | slot_bytes) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(scratch) % 16 != 0 || reinterpret_cast<uintptr_t>(x) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(records) % 16 != 0 || reinterpret_cast<uintptr_t>(slots) % 16 != 0)
        return VELOX_ERR_UNSUPPORTED;
    const NvDesc d{records, slots, map, record_bytes, slot_bytes, {gate_off, up_off, down_off}, width, hidden, experts};
    const long long pairs = static_cast<long long>(rows) * k, xs = static_cast<long long>(rows) * hidden;
    // Blocks along y: one per routed expert at most.
    const unsigned groups = static_cast<unsigned>(pairs < experts ? pairs : experts);
    const int threads = 256;
    uint8_t* p = static_cast<uint8_t*>(scratch);
    if (mode != 2) {
        // Tiles measured on the 5060 Ti (35B-A3B geometry): gate/up 16 rows, down 16 rows per
        // block at decode (fewer, longer system RAM runs), 32 rows grouped (more work per block).
        return dec_launch<4, 4, 8>(d, ids, wts, x, y, p, rows, k, mode, groups);
    }
    auto* xb = reinterpret_cast<__nv_bfloat16*>(p);
    auto* hb = reinterpret_cast<__nv_bfloat16*>(p += nv_round(2LL * rows * hidden));
    int* offsets = reinterpret_cast<int*>(p += nv_round(2 * pairs * width));
    int* list = reinterpret_cast<int*>(p += nv_round(4LL * (3 * experts + 3)));
    float* o = reinterpret_cast<float*>(p += nv_round(4 * pairs));
    nv_sort_kernel<<<1, 1024, 2 * experts * sizeof(int)>>>(ids, d.map, static_cast<int>(pairs), experts, offsets, list);
    nv_to_bf16_perm<<<velox_grid_1d(xs, threads), threads>>>(x, xs, hidden, xb);
    if (int e = mma_launch<2>(d, static_cast<int>(groups), width, hidden, offsets, list, k, xb, hb, nullptr)) return e;
    if (int e = mma_launch<1>(d, static_cast<int>(groups), hidden, width, offsets, list, k, hb, nullptr, o)) return e;
    nv_sum_kernel<<<velox_grid_1d(xs, threads), threads>>>(o, wts, rows, k, hidden, ids, experts, y);
    return velox_launch_status();
}
