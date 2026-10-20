// Shared device helpers for portable kernels (this crate) and tuned kernels (velox-variant on each
// hardware branch). Variant build scripts get this directory as an include path.
//
// Keep portable code inside the CUDA subset that hipify translates (no inline PTX, no
// architecture-specific intrinsics) so the future HIP backend can reuse it. Lane math uses
// VELOX_WARP; RDNA GPUs run wave32, so 32 holds there too.
#pragma once

#include <cuda_runtime.h>

#define VELOX_API extern "C"
#define VELOX_WARP 32

// Launcher return codes: 0 success, >0 a cudaError_t, <0 a velox code.
#define VELOX_ERR_UNSUPPORTED (-1)

__device__ __forceinline__ float velox_warp_sum(float v) {
#pragma unroll
    for (int o = VELOX_WARP / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float velox_warp_max(float v) {
#pragma unroll
    for (int o = VELOX_WARP / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// Block-wide sum for blockDim.x a multiple of VELOX_WARP and at most 1024. Every thread gets the
// result. `scratch` must hold VELOX_WARP floats.
__device__ __forceinline__ float velox_block_sum(float v, float* scratch) {
    const int lane = threadIdx.x % VELOX_WARP;
    const int warp = threadIdx.x / VELOX_WARP;
    v = velox_warp_sum(v);
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    const int warps = blockDim.x / VELOX_WARP;
    v = lane < warps ? scratch[lane] : 0.0f;
    v = velox_warp_sum(v);
    __syncthreads();
    return v;
}

static inline int velox_launch_status() { return static_cast<int>(cudaGetLastError()); }

static inline unsigned velox_grid_1d(long long n, int block) {
    long long g = (n + block - 1) / block;
    return static_cast<unsigned>(g < 65535LL * 32 ? g : 65535LL * 32);
}
