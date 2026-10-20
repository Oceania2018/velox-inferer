// C ABI over the CUDA runtime. Rust binds only to these functions, never to cudaDeviceProp,
// whose layout changes between toolkit versions.
#include <cstring>

#include "velox_common.cuh"

VELOX_API int velox_cuda_device_count(int* count) { return static_cast<int>(cudaGetDeviceCount(count)); }

VELOX_API int velox_cuda_device_info(int device, char* name, int name_len, int* major, int* minor, int* sm_count,
                                     size_t* total_mem) {
    cudaDeviceProp p;
    cudaError_t e = cudaGetDeviceProperties(&p, device);
    if (e != cudaSuccess) return static_cast<int>(e);
    std::strncpy(name, p.name, static_cast<size_t>(name_len - 1));
    name[name_len - 1] = '\0';
    *major = p.major;
    *minor = p.minor;
    *sm_count = p.multiProcessorCount;
    *total_mem = p.totalGlobalMem;
    return 0;
}

VELOX_API int velox_cuda_set_device(int device) { return static_cast<int>(cudaSetDevice(device)); }

VELOX_API int velox_cuda_mem_info(size_t* free_bytes, size_t* total_bytes) {
    return static_cast<int>(cudaMemGetInfo(free_bytes, total_bytes));
}

VELOX_API int velox_cuda_malloc_zeroed(void** ptr, size_t bytes) {
    cudaError_t e = cudaMalloc(ptr, bytes);
    if (e != cudaSuccess) return static_cast<int>(e);
    return static_cast<int>(cudaMemset(*ptr, 0, bytes));
}

VELOX_API int velox_cuda_free(void* ptr) { return static_cast<int>(cudaFree(ptr)); }

VELOX_API int velox_cuda_htod(void* dst, const void* src, size_t bytes) {
    return static_cast<int>(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
}

VELOX_API int velox_cuda_dtoh(void* dst, const void* src, size_t bytes) {
    return static_cast<int>(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
}

VELOX_API int velox_cuda_dtod(void* dst, const void* src, size_t bytes) {
    return static_cast<int>(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, 0));
}

VELOX_API int velox_cuda_synchronize() { return static_cast<int>(cudaDeviceSynchronize()); }

VELOX_API const char* velox_cuda_error_string(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}

VELOX_API int velox_cuda_runtime_version(int* v) { return static_cast<int>(cudaRuntimeGetVersion(v)); }

VELOX_API int velox_cuda_driver_version(int* v) { return static_cast<int>(cudaDriverGetVersion(v)); }
