// Top-k over logits rows for sampling: the k best (value desc, index asc; NaN lowest) of the first
// `limit` values of each row, so the host only receives a few candidates instead of the vocabulary.
//
// Two passes. Pass 1: blocks of 256 threads take 4096-value slices of a row and select their
// top-k by repeated block argmax; every thread keeps the best of its own values, so a round
// is one block reduction plus a rescan by the single thread that owned the winner. Pass 2: one
// 1024-thread block per row selects the final top-k from all slice candidates the same way.
#include <math.h>
#include <stdint.h>

#include <mutex>

#include "velox_common.cuh"

namespace {

constexpr int kSlicePer = 16, kMergePer = 4;
constexpr int kSlice = 256 * kSlicePer;     // values per pass-1 block (61 slices for a 248k vocabulary)
constexpr int kMergeCap = 1024 * kMergePer;  // candidates one pass-2 block can hold

__device__ __forceinline__ float rank_key(float v) { return isnan(v) ? -INFINITY : v; }

// a ranks before b.
__device__ __forceinline__ bool before(float va, int ia, float vb, int ib) {
    return va > vb || (va == vb && static_cast<unsigned>(ia) < static_cast<unsigned>(ib));
}

// Selects the top k of the n (value, index) pairs in shared memory sv/si (strided per thread) into
// out_v/out_i; positions past the available values get index -1. Destroys sv.
template <int THREADS, int PER>
__device__ void block_top_k(float* sv, int* si, int n, int k, float* out_v, int* out_i) {
    __shared__ float wv[THREADS / 32];
    __shared__ int wi[THREADS / 32];
    __shared__ int winner;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float bv = -INFINITY;
    int bi = -1;
    auto rescan = [&]() {
        bv = -INFINITY;
        bi = -1;
        for (int p = 0; p < PER; ++p) {
            const int j = p * THREADS + tid;
            if (j < n && si[j] >= 0 && (bi < 0 || before(sv[j], si[j], bv, bi))) bv = sv[j], bi = si[j];
        }
    };
    rescan();
    for (int r = 0; r < k; ++r) {
        float v = bv;
        int i = bi;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, o);
            const int oi = __shfl_xor_sync(0xffffffffu, i, o);
            if (oi >= 0 && (i < 0 || before(ov, oi, v, i))) v = ov, i = oi;
        }
        if (lane == 0) wv[warp] = v, wi[warp] = i;
        __syncthreads();
        if (warp == 0) {
            v = lane < THREADS / 32 ? wv[lane] : -INFINITY;
            i = lane < THREADS / 32 ? wi[lane] : -1;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, v, o);
                const int oi = __shfl_xor_sync(0xffffffffu, i, o);
                if (oi >= 0 && (i < 0 || before(ov, oi, v, i))) v = ov, i = oi;
            }
            if (lane == 0) {
                out_v[r] = v;
                out_i[r] = i;
                winner = i;
            }
        }
        __syncthreads();
        // Indices are unique, so exactly one thread owns the winner (none once values run out).
        if (winner >= 0 && bi == winner) {
            for (int p = 0; p < PER; ++p) {
                const int j = p * THREADS + tid;
                if (j < n && si[j] == winner) si[j] = -1;
            }
            rescan();
        }
    }
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
    top_k_slices<<<dim3(slices, rows), 256>>>(x, row_len, limit, k, cand_v, cand_i);
    top_k_merge<<<rows, 1024>>>(cand_v, cand_i, slices * k, k, out_v, out_i);
    if (const int e = velox_launch_status()) return e;
    if (cudaError_t e = cudaMemcpy(host_idx, out_i, outn * 4, cudaMemcpyDeviceToHost)) return static_cast<int>(e);
    return static_cast<int>(cudaMemcpy(host_val, out_v, outn * 4, cudaMemcpyDeviceToHost));
}
