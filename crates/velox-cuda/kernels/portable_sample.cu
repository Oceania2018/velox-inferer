// Top-k over logits rows for sampling: the k best (value desc, index asc; NaN lowest) of the first
// `limit` values of each row, so the host only receives a few candidates instead of the vocabulary.
//
// Two passes. Pass 1: blocks of 256 threads take 4096-value slices of a row and keep the top k of
// each. Pass 2: one 1024-thread block per row keeps the final top k of all slice candidates. Each
// pass selects by k block reductions for k <= kSelectMax (each picks the best value ranked after
// the previous pick), else by a bitonic sort in shared memory; both give the same top k.
#include <math.h>
#include <stdint.h>
#include <string.h>

#include <mutex>

#include "velox_common.cuh"

namespace {

constexpr int kSlicePer = 16, kMergePer = 4;
constexpr int kSlice = 256 * kSlicePer;     // values per pass-1 block (61 slices for a 248k vocabulary)
constexpr int kMergeCap = 1024 * kMergePer;  // candidates one pass-2 block can hold
constexpr int kSelectMax = 32;                // largest k selected by reductions

__device__ __forceinline__ float rank_key(float v) { return isnan(v) ? -INFINITY : v; }

// a ranks before b. Index -1 marks padding, which ranks after everything.
__device__ __forceinline__ bool before(float va, int ia, float vb, int ib) {
    if (ia < 0) return false;
    if (ib < 0) return true;
    return va > vb || (va == vb && ia < ib);
}

// Selects the top k of the n (value, index) pairs in shared memory sv/si (THREADS * PER slots,
// entries past n are padding) into out_v/out_i by a bitonic sort of all slots, best first:
// log2(N)^2 / 2 compare-exchange passes, each a single sweep by the block.
template <int THREADS, int PER>
__device__ void block_top_k(float* sv, int* si, int n, int k, float* out_v, int* out_i) {
    constexpr int N = THREADS * PER;
    for (int j = n + static_cast<int>(threadIdx.x); j < N; j += THREADS) sv[j] = -INFINITY, si[j] = -1;
    __syncthreads();
    for (int size = 2; size <= N; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int t = threadIdx.x; t < N / 2; t += THREADS) {
                const int i = 2 * t - (t & (stride - 1)), j = i + stride;
                const bool best_first = (i & size) == 0;
                if (best_first != before(sv[i], si[i], sv[j], si[j])) {
                    const float v = sv[i];
                    const int x = si[i];
                    sv[i] = sv[j], si[i] = si[j];
                    sv[j] = v, si[j] = x;
                }
            }
            __syncthreads();
        }
    }
    for (int r = threadIdx.x; r < k; r += THREADS) out_v[r] = sv[r], out_i[r] = si[r];
}

__device__ __forceinline__ void keep_best(float& v, int& i, float ov, int oi) {
    if (before(ov, oi, v, i)) v = ov, i = oi;
}

// The block-wide best of every thread's (v, i), returned to all threads.
template <int THREADS>
__device__ void block_best(float& v, int& i) {
    __shared__ float wv[THREADS / 32];
    __shared__ int wi[THREADS / 32];
    for (int o = 16; o > 0; o >>= 1) keep_best(v, i, __shfl_xor_sync(0xffffffffu, v, o), __shfl_xor_sync(0xffffffffu, i, o));
    if ((threadIdx.x & 31) == 0) wv[threadIdx.x / 32] = v, wi[threadIdx.x / 32] = i;
    __syncthreads();
    v = wv[0], i = wi[0];
    for (int w = 1; w < THREADS / 32; ++w) keep_best(v, i, wv[w], wi[w]);
    __syncthreads();
}

// The top k (k <= kSelectMax) of the pairs each thread holds in registers (index -1: padding)
// into out_v/out_i, best first: round r takes the best pair ranked after round r - 1's pick.
template <int THREADS, int PER>
__device__ void block_select(const float (&v)[PER], const int (&ix)[PER], int k, float* out_v, int* out_i) {
    float pv = INFINITY;
    int pi = -1;
    for (int r = 0; r < k; ++r) {
        float bv = -INFINITY;
        int bi = -1;
#pragma unroll
        for (int p = 0; p < PER; ++p)
            if (ix[p] >= 0 && (r == 0 || before(pv, pi, v[p], ix[p]))) keep_best(bv, bi, v[p], ix[p]);
        block_best<THREADS>(bv, bi);
        if (threadIdx.x == 0) out_v[r] = bv, out_i[r] = bi;
        pv = bv, pi = bi;
    }
}

__global__ void __launch_bounds__(256) top_k_select_slices(const float* __restrict__ x, long long row_len, int limit,
                                                           int k, float* __restrict__ cand_v, int* __restrict__ cand_i) {
    const int row = blockIdx.y, slice = blockIdx.x, slices = gridDim.x;
    const int lo = slice * kSlice;
    const int n = min(kSlice, limit - lo);
    const float* xr = x + row * row_len + lo;
    float v[kSlicePer];
    int ix[kSlicePer];
#pragma unroll
    for (int p = 0; p < kSlicePer; ++p) {
        const int j = p * 256 + threadIdx.x;
        v[p] = j < n ? rank_key(xr[j]) : -INFINITY;
        ix[p] = j < n ? lo + j : -1;
    }
    const long long out = (static_cast<long long>(row) * slices + slice) * k;
    block_select<256, kSlicePer>(v, ix, k, cand_v + out, cand_i + out);
}

__global__ void __launch_bounds__(1024) top_k_select_merge(const float* __restrict__ cand_v,
                                                           const int* __restrict__ cand_i, int n, int k,
                                                           float* __restrict__ out_v, int* __restrict__ out_i) {
    const int row = blockIdx.x;
    float v[kMergePer];
    int ix[kMergePer];
#pragma unroll
    for (int p = 0; p < kMergePer; ++p) {
        const int j = p * 1024 + threadIdx.x;
        v[p] = j < n ? cand_v[static_cast<long long>(row) * n + j] : -INFINITY;
        ix[p] = j < n ? cand_i[static_cast<long long>(row) * n + j] : -1;
    }
    block_select<1024, kMergePer>(v, ix, k, out_v + static_cast<long long>(row) * k, out_i + static_cast<long long>(row) * k);
}

__global__ void __launch_bounds__(256) top_k_slices(const float* __restrict__ x, long long row_len, int limit, int k,
                                                    float* __restrict__ cand_v, int* __restrict__ cand_i) {
    __shared__ float sv[kSlice];
    __shared__ int si[kSlice];
    const int row = blockIdx.y, slice = blockIdx.x, slices = gridDim.x;
    const int lo = slice * kSlice;
    const int n = min(kSlice, limit - lo);
    const float* xr = x + row * row_len + lo;
    for (int p = 0; p < kSlicePer; ++p) {
        const int j = p * 256 + threadIdx.x;
        if (j < n) sv[j] = rank_key(xr[j]), si[j] = lo + j;
    }
    __syncthreads();
    const long long out = (static_cast<long long>(row) * slices + slice) * k;
    block_top_k<256, kSlicePer>(sv, si, n, k, cand_v + out, cand_i + out);
}

__global__ void __launch_bounds__(1024) top_k_merge(const float* __restrict__ cand_v, const int* __restrict__ cand_i,
                                                    int n, int k, float* __restrict__ out_v, int* __restrict__ out_i) {
    __shared__ float sv[kMergeCap];
    __shared__ int si[kMergeCap];
    const int row = blockIdx.x;
    for (int p = 0; p < kMergePer; ++p) {
        const int j = p * 1024 + threadIdx.x;
        if (j < n) sv[j] = cand_v[static_cast<long long>(row) * n + j], si[j] = cand_i[static_cast<long long>(row) * n + j];
    }
    __syncthreads();
    block_top_k<1024, kMergePer>(sv, si, n, k, out_v + static_cast<long long>(row) * k, out_i + static_cast<long long>(row) * k);
}

// k = 1 (greedy argmax, e.g. speculative verify): a reduction instead of sorting each slice.
// Same order as `before`: value desc, then index asc; NaN ranks lowest; index -1 is padding.

template <int THREADS>
__device__ void block_argmax(float v, int i, float* out_v, int* out_i) {
    __shared__ float wv[THREADS / 32];
    __shared__ int wi[THREADS / 32];
    for (int o = 16; o > 0; o >>= 1) keep_best(v, i, __shfl_xor_sync(0xffffffffu, v, o), __shfl_xor_sync(0xffffffffu, i, o));
    if ((threadIdx.x & 31) == 0) wv[threadIdx.x / 32] = v, wi[threadIdx.x / 32] = i;
    __syncthreads();
    if (threadIdx.x < 32) {
        v = threadIdx.x < THREADS / 32 ? wv[threadIdx.x] : -INFINITY;
        i = threadIdx.x < THREADS / 32 ? wi[threadIdx.x] : -1;
        for (int o = 16; o > 0; o >>= 1) keep_best(v, i, __shfl_xor_sync(0xffffffffu, v, o), __shfl_xor_sync(0xffffffffu, i, o));
        if (threadIdx.x == 0) *out_v = v, *out_i = i;
    }
}

__global__ void __launch_bounds__(256) argmax_slices(const float* __restrict__ x, long long row_len, int limit,
                                                     float* __restrict__ cand_v, int* __restrict__ cand_i) {
    const int row = blockIdx.y, slice = blockIdx.x;
    const int lo = slice * kSlice, n = min(kSlice, limit - lo);
    const float* xr = x + row * row_len + lo;
    float v = -INFINITY;
    int i = -1;
    for (int j = threadIdx.x; j < n; j += 256) keep_best(v, i, rank_key(xr[j]), lo + j);
    const long long out = static_cast<long long>(row) * gridDim.x + slice;
    block_argmax<256>(v, i, cand_v + out, cand_i + out);
}

__global__ void __launch_bounds__(256) argmax_merge(const float* __restrict__ cand_v, const int* __restrict__ cand_i,
                                                    int n, float* __restrict__ out_v, int* __restrict__ out_i) {
    const int row = blockIdx.x;
    float v = -INFINITY;
    int i = -1;
    for (int j = threadIdx.x; j < n; j += 256)
        keep_best(v, i, cand_v[static_cast<long long>(row) * n + j], cand_i[static_cast<long long>(row) * n + j]);
    block_argmax<256>(v, i, out_v + row, out_i + row);
}

struct Scratch {
    std::mutex mu;
    uint8_t* ptr = nullptr;
    size_t bytes = 0;
    uint8_t* get(size_t need) {
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
Scratch g_topk;

// Pinned staging for the results: one async copy and one stream synchronization per call (a
// pageable cudaMemcpy pays a full host round trip per array).
struct HostStage {
    uint8_t* ptr = nullptr;
    size_t bytes = 0;
    uint8_t* get(size_t need) {
        if (need > bytes) {
            if (ptr) cudaFreeHost(ptr);
            ptr = nullptr;
            bytes = 0;
            if (cudaMallocHost(&ptr, need) != cudaSuccess) return nullptr;
            bytes = need;
        }
        return ptr;
    }
};
HostStage g_topk_host;

}  // namespace

// Writes rows * k (index, value) pairs to host arrays; index -1 marks a missing value. Returns
// VELOX_ERR_UNSUPPORTED when the candidates of one row do not fit the merge block.
VELOX_API int velox_portable_top_k(const float* x, int rows, long long row_len, int limit, int k, int* host_idx,
                                   float* host_val) {
    const int slices = (limit + kSlice - 1) / kSlice;
    if (slices * k > kMergeCap || rows <= 0 || rows > 65535) return VELOX_ERR_UNSUPPORTED;
    std::lock_guard<std::mutex> lock(g_topk.mu);
    const size_t cand = static_cast<size_t>(rows) * slices * k, outn = static_cast<size_t>(rows) * k;
    uint8_t* base = g_topk.get((cand + outn) * 8);
    if (!base) return static_cast<int>(cudaErrorMemoryAllocation);
    float* cand_v = reinterpret_cast<float*>(base);
    int* cand_i = reinterpret_cast<int*>(base + cand * 4);
    float* out_v = reinterpret_cast<float*>(base + cand * 8);
    int* out_i = reinterpret_cast<int*>(base + cand * 8 + outn * 4);
    if (k == 1) {
        argmax_slices<<<dim3(slices, rows), 256>>>(x, row_len, limit, cand_v, cand_i);
        argmax_merge<<<rows, 256>>>(cand_v, cand_i, slices, out_v, out_i);
    } else if (k <= kSelectMax) {
        top_k_select_slices<<<dim3(slices, rows), 256>>>(x, row_len, limit, k, cand_v, cand_i);
        top_k_select_merge<<<rows, 1024>>>(cand_v, cand_i, slices * k, k, out_v, out_i);
    } else {
        top_k_slices<<<dim3(slices, rows), 256>>>(x, row_len, limit, k, cand_v, cand_i);
        top_k_merge<<<rows, 1024>>>(cand_v, cand_i, slices * k, k, out_v, out_i);
    }
    if (const int e = velox_launch_status()) return e;
    // out_v and out_i are adjacent: one copy brings both.
    uint8_t* stage = g_topk_host.get(outn * 8);
    if (!stage) return static_cast<int>(cudaErrorMemoryAllocation);
    if (cudaError_t e = cudaMemcpyAsync(stage, out_v, outn * 8, cudaMemcpyDeviceToHost, 0)) return static_cast<int>(e);
    if (cudaError_t e = cudaStreamSynchronize(0)) return static_cast<int>(e);
    memcpy(host_val, stage, outn * 4);
    memcpy(host_idx, stage + outn * 4, outn * 4);
    return 0;
}

namespace {

// argmax_merge writing the best index as a u32 (the row's best is always found: limit >= 1).
__global__ void __launch_bounds__(256) argmax_merge_to(const float* __restrict__ cand_v, const int* __restrict__ cand_i,
                                                       int n, unsigned int* __restrict__ dst) {
    float v = -INFINITY;
    int i = -1;
    for (int j = threadIdx.x; j < n; j += 256) keep_best(v, i, cand_v[j], cand_i[j]);
    block_best<256>(v, i);
    if (threadIdx.x == 0) *dst = static_cast<unsigned int>(i < 0 ? 0 : i);
}

Scratch g_argmax;  // its own scratch: a captured step must keep its addresses

// Pinned slots the device writes u32 values into, each with an event the host waits on.
constexpr int kFetchSlots = 8;
struct FetchRing {
    std::mutex mu;
    unsigned int* host = nullptr;
    cudaEvent_t ev[kFetchSlots] = {};
    int init() {
        if (host) return 0;
        if (cudaError_t e = cudaHostAlloc(reinterpret_cast<void**>(&host), kFetchSlots * sizeof(unsigned int),
                                          cudaHostAllocDefault))
            return static_cast<int>(e);
        for (int s = 0; s < kFetchSlots; ++s)
            if (cudaError_t e = cudaEventCreateWithFlags(&ev[s], cudaEventDisableTiming)) return static_cast<int>(e);
        return 0;
    }
} g_fetch;

}  // namespace

// The index of the best of the first `limit` values of one row (ranked as top_k: value desc,
// index asc, NaN last), written to dst as a u32 on the device, so the next step can read it there.
VELOX_API int velox_portable_argmax_to(const float* x, long long row_len, int limit, unsigned int* dst) {
    const int slices = (limit + kSlice - 1) / kSlice;
    if (limit <= 0 || row_len < limit) return VELOX_ERR_UNSUPPORTED;
    std::lock_guard<std::mutex> lock(g_argmax.mu);
    uint8_t* base = g_argmax.get(static_cast<size_t>(slices) * 8);
    if (!base) return static_cast<int>(cudaErrorMemoryAllocation);
    float* cand_v = reinterpret_cast<float*>(base);
    int* cand_i = reinterpret_cast<int*>(base + static_cast<size_t>(slices) * 4);
    argmax_slices<<<dim3(slices, 1), 256>>>(x, row_len, limit, cand_v, cand_i);
    argmax_merge_to<<<1, 256>>>(cand_v, cand_i, slices, dst);
    return velox_launch_status();
}

// Queues a copy of the u32 at src into pinned slot `slot` (0 <= slot < 8) behind the work queued
// so far, with an event; `velox_portable_fetch_u32_wait` waits for that event and reads the slot.
VELOX_API int velox_portable_fetch_u32_begin(const unsigned int* src, int slot) {
    if (slot < 0 || slot >= kFetchSlots) return VELOX_ERR_UNSUPPORTED;
    std::lock_guard<std::mutex> lock(g_fetch.mu);
    if (const int e = g_fetch.init()) return e;
    if (cudaError_t e = cudaMemcpyAsync(g_fetch.host + slot, src, sizeof(unsigned int), cudaMemcpyDeviceToHost, 0))
        return static_cast<int>(e);
    return static_cast<int>(cudaEventRecord(g_fetch.ev[slot], 0));
}

VELOX_API int velox_portable_fetch_u32_wait(int slot, unsigned int* out) {
    if (slot < 0 || slot >= kFetchSlots) return VELOX_ERR_UNSUPPORTED;
    std::lock_guard<std::mutex> lock(g_fetch.mu);
    if (const int e = g_fetch.init()) return e;
    if (cudaError_t e = cudaEventSynchronize(g_fetch.ev[slot])) return static_cast<int>(e);
    *out = g_fetch.host[slot];
    return 0;
}
