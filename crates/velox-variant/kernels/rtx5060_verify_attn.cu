// RTX 5060 Ti (from rtx-5090-32g) attention over an NVFP4 KV cache (head_dim 256, GQA): decode, verify and
// prefill.
//
// Precision (velox_core::AttnPrecision::F16): the query is rotated (the cache's Hadamard basis),
// scaled and rounded to f16; cached K/V values (E2M1 x E4M3) are exact in f16; the softmax weights
// are rounded to f16 for the value product; accumulation is f32.
//
// Block: one KV head x one key split x one block of 128 query rows, eight warps; warp w owns rows
// [16w, 16w + 16) of the block (row r = token * group + head-in-group). Keys past the block's last
// token are never read (causal). Key tiles of 32 rows are
// fetched as packed bytes two tiles ahead into registers, decoded by every thread (native
// E2M1/E4M3 conversions) into f16 shared memory once and served to the row warps through
// ldmatrix + mma.sync m16n8k16. Each split writes an unnormalized partial
// (log2-domain max, sum, accumulator); a combine kernel merges splits and rotates the result back.
#include <cuda_fp16.h>
#include <stdint.h>

#include "velox_common.cuh"

namespace {

constexpr int kD = 256;
constexpr int kTK = 32;                    // keys per tile
constexpr int kRowBytes = kD / 2 + kD / 16;  // 128 code bytes + 16 E4M3 scales = 144
constexpr int kChunks = kD / 32;           // 16-byte code chunks per row (32 elements each)
constexpr float kLog2e = 1.4426950408889634f;

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
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

// Half offset of (row, col) in a [rows][256] f16 tile: 32 chunks of 8 halves per row, the low 3
// chunk bits XOR-swizzled by (row & 7) so ldmatrix over 8 consecutive rows is conflict-free.
__device__ __forceinline__ int sw(int row, int col) {
    const int chunk = col >> 3;
    return row * kD + (((chunk ^ row) & 7) | (chunk & ~7)) * 8 + (col & 7);
}

// Eight E2M1 codes (even element in the low nibble) as four f16x2 words, in element order.
__device__ __forceinline__ void e2m1x8_h(uint32_t x, uint32_t (&h)[4]) {
    asm("{\n.reg .b8 b0, b1, b2, b3;\nmov.b32 {b0, b1, b2, b3}, %4;\n"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\ncvt.rn.f16x2.e2m1x2 %1, b1;\n"
        "cvt.rn.f16x2.e2m1x2 %2, b2;\ncvt.rn.f16x2.e2m1x2 %3, b3;\n}"
        : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3])
        : "r"(x));
}

// Two E4M3 codes (low byte first) as an f16x2 word (exact).
__device__ __forceinline__ uint32_t e4m3x2_h(uint16_t x) {
    uint32_t h;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"(x));
    return h;
}

__device__ __forceinline__ uint32_t hmul2_u(uint32_t a, uint32_t b) {
    const __half2 r = __hmul2(*reinterpret_cast<const __half2*>(&a), *reinterpret_cast<const __half2*>(&b));
    return *reinterpret_cast<const uint32_t*>(&r);
}

// Normalized Sylvester-Hadamard transform of a 256-row held as x[p] = row[lane + 32p].
__device__ __forceinline__ void hadamard8(float (&x)[8], int lane, float norm) {
#pragma unroll
    for (int s = 1; s < 32; s <<= 1) {
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            const float peer = __shfl_xor_sync(0xffffffffu, x[p], s);
            x[p] = (lane & s) ? peer - x[p] : x[p] + peer;
        }
    }
#pragma unroll
    for (int span = 1; span < 8; span <<= 1) {
#pragma unroll
        for (int base = 0; base < 8; base += 2 * span) {
#pragma unroll
            for (int o = 0; o < span; ++o) {
                const float lo = x[base + o], hi = x[base + o + span];
                x[base + o] = lo + hi;
                x[base + o + span] = lo - hi;
            }
        }
    }
#pragma unroll
    for (int p = 0; p < 8; ++p) x[p] *= norm;
}


// One warp per (token, head) query row: rotate into the cache basis, scale (softmax scale x
// log2 e), round to f16. Output rows are grouped per KV head: row (kvh, t * group + head-in-group).
constexpr int kPrepWarps = 8;
__global__ void __launch_bounds__(kPrepWarps * 32)
    prep_query(const float* __restrict__ q, __half* __restrict__ q16, int tokens, int heads, int kv_heads,
               float qscale, float norm) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * kPrepWarps + (threadIdx.x >> 5);  // t * heads + h
    if (row >= tokens * heads) return;
    const int t = row / heads, h = row % heads, group = heads / kv_heads;
    float x[8];
    const float* src = q + static_cast<size_t>(row) * kD;
#pragma unroll
    for (int i = 0; i < 8; ++i) x[i] = src[lane + 32 * i];
    hadamard8(x, lane, norm);
    __half* dst = q16 + (static_cast<size_t>(h / group) * tokens * group + t * group + h % group) * kD;
#pragma unroll
    for (int i = 0; i < 8; ++i) dst[lane + 32 * i] = __float2half_rn(x[i] * qscale);
}

struct VerifyArgs {
    const __half* q16;    // [kv_heads][tokens * group][256], from prep_query
    const uint8_t* k;     // [kv rows, kv_heads] x 144 B
    const uint8_t* v;
    float* part_acc;      // [tokens * heads, splits, 256]
    float* part_ml;       // [tokens * heads, splits, 2] (log2-domain max, sum)
    const long long* pp;  // null, or device [pos0, k_cache, v_cache] staged by a graph step
    int tokens, heads, kv_heads, pos0, splits, split_keys;
};

constexpr int kBlockRows = 128;  // query rows per block (eight warps of 16)

constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kItems = 2 * kTK * kChunks;  // (K|V, key, chunk) per tile
constexpr int kPer = kItems / kThreads;
static_assert(kItems % kThreads == 0, "whole items per thread");

struct Raw {
    uint4 w[kPer];
    uint16_t s[kPer];
};

// Warps [0, mt) each own 16 query rows; all eight fetch and decode. Packed bytes are fetched two
// tiles ahead into registers.
__global__ void __launch_bounds__(kThreads) verify_kernel(VerifyArgs p) {
    extern __shared__ __align__(128) __half smem[];
    const int group = p.heads / p.kv_heads, row0 = blockIdx.z * kBlockRows;
    const int rows = min(kBlockRows, p.tokens * group - row0), mt = (rows + 15) / 16;
    __half* ks = smem;           // [kTK][256]
    __half* vs = ks + kTK * kD;  // [kTK][256]
    __half* qs = vs + kTK * kD;  // [mt * 16][256]

    const int pos0 = p.pp != nullptr ? static_cast<int>(p.pp[0]) : p.pos0;
    const uint8_t* kc = p.pp != nullptr ? reinterpret_cast<const uint8_t*>(p.pp[1]) : p.k;
    const uint8_t* vc = p.pp != nullptr ? reinterpret_cast<const uint8_t*>(p.pp[2]) : p.v;
    const int sp = blockIdx.x, kvh = blockIdx.y;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, q4 = lane & 3;
    const int j0 = sp * p.split_keys, j1 = min(j0 + p.split_keys, pos0 + (row0 + rows - 1) / group + 1);

    auto fetch = [&](Raw& r, int base) {
#pragma unroll
        for (int f = 0; f < kPer; ++f) {
            const int idx = threadIdx.x + f * kThreads;
            const int which = idx / (kTK * kChunks), key = idx / kChunks % kTK, c = idx % kChunks;
            r.w[f] = make_uint4(0, 0, 0, 0);
            r.s[f] = 0;
            if (base + key < j1) {
                const uint8_t* row =
                    (which ? vc : kc) + (static_cast<size_t>(base + key) * p.kv_heads + kvh) * kRowBytes;
                r.w[f] = __ldcs(reinterpret_cast<const uint4*>(row + 16 * c));
                r.s[f] = *reinterpret_cast<const uint16_t*>(row + kD / 2 + 2 * c);
            }
        }
    };
    auto decode = [&](const Raw& r) {
#pragma unroll
        for (int f = 0; f < kPer; ++f) {
            const int idx = threadIdx.x + f * kThreads;
            const int which = idx / (kTK * kChunks), key = idx / kChunks % kTK, c = idx % kChunks;
            // Group scales of elements 0..15 and 16..31 of the chunk, each broadcast to both halves.
            const uint32_t s01 = e4m3x2_h(r.s[f]);
            const uint32_t sa = __byte_perm(s01, 0u, 0x1010u), sb = __byte_perm(s01, 0u, 0x3232u);
            const uint32_t u[4] = {r.w[f].x, r.w[f].y, r.w[f].z, r.w[f].w};
            __half* dst = which ? vs : ks;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t h[4];
                e2m1x8_h(u[j], h);
                const uint32_t sc = j < 2 ? sa : sb;
                *reinterpret_cast<uint4*>(dst + sw(key, c * 32 + j * 8)) =
                    make_uint4(hmul2_u(h[0], sc), hmul2_u(h[1], sc), hmul2_u(h[2], sc), hmul2_u(h[3], sc));
            }
        }
    };

    Raw ra, rb;
    if (j0 < j1) fetch(ra, j0);
    if (j0 + kTK < j1) fetch(rb, j0 + kTK);

    // Query tile (16-byte chunks, swizzled like K/V).
    const __half* qsrc = p.q16 + (static_cast<size_t>(kvh) * p.tokens * group + row0) * kD;
    for (int i = threadIdx.x; i < mt * 16 * (kD / 8); i += kThreads) {
        const int r = i / (kD / 8), c = i % (kD / 8);
        const uint4 x = r < rows ? *reinterpret_cast<const uint4*>(qsrc + static_cast<size_t>(r) * kD + c * 8)
                                 : make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(qs + sw(r, c * 8)) = x;
    }

    const bool active = warp < mt;
    const int r0 = warp * 16 + g, r1 = r0 + 8;
    const int last0 = r0 < rows ? pos0 + (row0 + r0) / group : -1;
    const int last1 = r1 < rows ? pos0 + (row0 + r1) / group : -1;
    // Last key any row of this warp sees (warp-uniform; -1 for helper warps).
    const int wlast = active ? pos0 + (row0 + min(rows - 1, warp * 16 + 15)) / group : -1;
    float o[32][4];
#pragma unroll
    for (int i = 0; i < 32; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;

    auto compute = [&](int base) {
        // S = Q K^T: 16 rows x 32 keys, 16 k-steps.
        float sc[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i) sc[i][0] = sc[i][1] = sc[i][2] = sc[i][3] = 0.0f;
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            uint32_t a[4];
            ldsm_x4(a, qs + sw(warp * 16 + (lane & 15), kk * 16 + ((lane >> 4) << 3)));
#pragma unroll
            for (int np = 0; np < 2; ++np) {
                uint32_t b[4];
                ldsm_x4(b, ks + sw(np * 16 + (lane & 7) + ((lane >> 4) << 3), kk * 16 + (((lane >> 3) & 1) << 3)));
                mma_f16(sc[2 * np], a, b[0], b[1]);
                mma_f16(sc[2 * np + 1], a, b[2], b[3]);
            }
        }

        // Mask and online softmax (base 2). Rows g (c0, c1) and g + 8 (c2, c3).
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int key = base + nt * 8 + 2 * q4 + e;
                sc[nt][e] = key <= last0 ? sc[nt][e] : -INFINITY;
                sc[nt][2 + e] = key <= last1 ? sc[nt][2 + e] : -INFINITY;
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
        for (int nt = 0; nt < 4; ++nt) {
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
        for (int i = 0; i < 32; ++i) {
            o[i][0] *= al0, o[i][1] *= al0;
            o[i][2] *= al1, o[i][3] *= al1;
        }

        // O += P V: two k-steps of 16 keys; P becomes the f16 A operand directly.
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const uint32_t a[4] = {pack_h2(sc[2 * j][0], sc[2 * j][1]), pack_h2(sc[2 * j][2], sc[2 * j][3]),
                                   pack_h2(sc[2 * j + 1][0], sc[2 * j + 1][1]),
                                   pack_h2(sc[2 * j + 1][2], sc[2 * j + 1][3])};
            const int vrow = j * 16 + (lane & 7) + (((lane >> 3) & 1) << 3);
#pragma unroll
            for (int dn = 0; dn < 16; ++dn) {
                uint32_t b[4];
                ldsm_x4_t(b, vs + sw(vrow, dn * 16 + ((lane >> 4) << 3)));
                mma_f16(o[2 * dn], a, b[0], b[1]);
                mma_f16(o[2 * dn + 1], a, b[2], b[3]);
            }
        }
    };
    auto step = [&](Raw& r, int base) {
        __syncthreads();  // the previous tile is consumed (first time: the query tile is written)
        decode(r);
        __syncthreads();
        if (base + 2 * kTK < j1) fetch(r, base + 2 * kTK);
        if (base <= wlast) compute(base);
    };
    for (int base = j0; base < j1; base += 2 * kTK) {
        step(ra, base);
        if (base + kTK < j1) step(rb, base + kTK);
    }
    if (!active) return;

    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int r = half ? r1 : r0;
        if (r >= rows) continue;
        const int t = (row0 + r) / group, h = kvh * group + (row0 + r) % group;
        const size_t idx = (static_cast<size_t>(t) * p.heads + h) * p.splits + sp;
        float* dst = p.part_acc + idx * kD;
#pragma unroll
        for (int nt = 0; nt < 32; ++nt)
            *reinterpret_cast<float2*>(dst + nt * 8 + 2 * q4) =
                half ? make_float2(o[nt][2], o[nt][3]) : make_float2(o[nt][0], o[nt][1]);
        if (q4 == 0) {
            p.part_ml[2 * idx] = half ? m1 : m0;
            p.part_ml[2 * idx + 1] = half ? l1 : l0;
        }
    }
}

// One block per (token, head) row: warps take interleaved splits, the block sums them, warp 0
// normalizes and rotates back (V is stored rotated). Splits that saw no keys (sum 0) are skipped.
constexpr int kCombineWarps = 8;
__global__ void __launch_bounds__(kCombineWarps * 32)
    combine_kernel(const float* __restrict__ part_acc, const float* __restrict__ part_ml, float* __restrict__ out,
                   int splits, float norm) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, row = blockIdx.x;
    const float* ml = part_ml + static_cast<size_t>(row) * splits * 2;
    float gm = -INFINITY;
    for (int s = lane; s < splits; s += 32)
        if (ml[2 * s + 1] > 0.0f) gm = fmaxf(gm, ml[2 * s]);
    gm = velox_warp_max(gm);
    float gl = 0.0f;
    for (int s = lane; s < splits; s += 32) {
        const float l = ml[2 * s + 1];
        if (l > 0.0f) gl += l * exp2f(ml[2 * s] - gm);
    }
    gl = velox_warp_sum(gl);
    float acc[8] = {};
    const float* pa = part_acc + static_cast<size_t>(row) * splits * kD;
    for (int s = warp; s < splits; s += kCombineWarps) {
        const float l = ml[2 * s + 1];
        if (l <= 0.0f) continue;
        const float w = exp2f(ml[2 * s] - gm);
        const float4 a = reinterpret_cast<const float4*>(pa + s * kD)[lane * 2];
        const float4 b = reinterpret_cast<const float4*>(pa + s * kD)[lane * 2 + 1];
        acc[0] += a.x * w, acc[1] += a.y * w, acc[2] += a.z * w, acc[3] += a.w * w;
        acc[4] += b.x * w, acc[5] += b.y * w, acc[6] += b.z * w, acc[7] += b.w * w;
    }
    __shared__ float part[kCombineWarps][kD];
#pragma unroll
    for (int i = 0; i < 8; ++i) part[warp][8 * lane + i] = acc[i];
    __syncthreads();
    if (warp != 0) return;
    // The Hadamard helper wants element lane + 32p in x[p].
    const float inv = gl > 0.0f ? 1.0f / gl : 0.0f;
    float x[8];
#pragma unroll
    for (int p = 0; p < 8; ++p) {
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < kCombineWarps; ++w) s += part[w][lane + 32 * p];
        x[p] = s * inv;
    }
    hadamard8(x, lane, norm);
    float* dst = out + static_cast<size_t>(row) * kD;
#pragma unroll
    for (int p = 0; p < 8; ++p) dst[lane + 32 * p] = x[p];
}

// Grow-only and never freed: recorded CUDA graphs keep referencing the buffer.
uint8_t* verify_scratch(size_t bytes) {
    static VeloxScratch s;
    return static_cast<uint8_t*>(s.get(bytes));
}

int sm_count() {
    static int sms = 0;
    if (sms == 0) {
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || sms <= 0)
            sms = 36;
    }
    return sms;
}


size_t verify_smem(int mt) { return static_cast<size_t>(2 * kTK + mt * 16) * kD * sizeof(__half); }

// Resident verify blocks per SM for `mt` row tiles (cached per mt).
int verify_occupancy(int mt) {
    static int occ[kBlockRows / 16 + 1] = {};
    if (occ[mt] == 0) {
        static bool configured = false;
        if (!configured) {
            cudaFuncSetAttribute(verify_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(verify_smem(kBlockRows / 16)));
            cudaFuncSetAttribute(verify_kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                 static_cast<int>(cudaSharedmemCarveoutMaxShared));
            configured = true;
        }
        int n = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, verify_kernel, kThreads, verify_smem(mt)) !=
                cudaSuccess ||
            n <= 0)
            n = 1;
        occ[mt] = n;
    }
    return occ[mt];
}

size_t align_up(size_t x) { return (x + 255) / 256 * 256; }

}  // namespace

// Any number of tokens. params: null, or device [pos0, k_cache, v_cache] staged by a graph step,
// with kv_grid >= pos0 + tokens sizing the grid (the grid never depends on the staged value).
VELOX_API int velox_5060_attention_verify_nvfp4(const float* q, const unsigned char* k, const unsigned char* v,
                                                float* out, int tokens, int heads, int kv_heads, int head_dim,
                                                int pos0, float scale, float norm, const long long* params,
                                                int kv_grid) {
    if (head_dim != kD || tokens <= 0 || kv_heads <= 0 || heads % kv_heads != 0) return VELOX_ERR_UNSUPPORTED;
    const long long rows = static_cast<long long>(tokens) * (heads / kv_heads);
    const int row_blocks = static_cast<int>((rows + kBlockRows - 1) / kBlockRows);
    const int mt = static_cast<int>(rows < kBlockRows ? (rows + 15) / 16 : kBlockRows / 16);
    const int kv = kv_grid >= pos0 + tokens ? kv_grid : pos0 + tokens;
    // One wave of resident blocks, whole tiles per split, at least two tiles.
    const int want = (verify_occupancy(mt) * sm_count() + kv_heads * row_blocks - 1) / (kv_heads * row_blocks);
    int split_keys = (kv + want - 1) / want;
    split_keys = (split_keys + kTK - 1) / kTK * kTK;
    if (split_keys < 2 * kTK) split_keys = 2 * kTK;
    const int splits = (kv + split_keys - 1) / split_keys;
    const size_t n_rows = static_cast<size_t>(tokens) * heads;
    const size_t qb = align_up(n_rows * kD * sizeof(__half)), ab = align_up(n_rows * splits * kD * sizeof(float));
    uint8_t* base = verify_scratch(qb + ab + n_rows * splits * 2 * sizeof(float));
    if (base == nullptr) return static_cast<int>(cudaErrorMemoryAllocation);
    __half* q16 = reinterpret_cast<__half*>(base);
    float* part_acc = reinterpret_cast<float*>(base + qb);
    float* part_ml = reinterpret_cast<float*>(base + qb + ab);
    prep_query<<<static_cast<unsigned>((n_rows + kPrepWarps - 1) / kPrepWarps), kPrepWarps * 32>>>(
        q, q16, tokens, heads, kv_heads, scale * kLog2e, norm);
    VerifyArgs a{q16, k, v, part_acc, part_ml, params, tokens, heads, kv_heads, pos0, splits, split_keys};
    verify_kernel<<<dim3(splits, kv_heads, row_blocks), kThreads, verify_smem(mt)>>>(a);
    combine_kernel<<<static_cast<unsigned>(n_rows), kCombineWarps * 32>>>(part_acc, part_ml, out, splits, norm);
    return velox_launch_status();
}
