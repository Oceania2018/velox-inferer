// C ABI over the CUDA runtime. Rust binds only to these functions, never to cudaDeviceProp,
// whose layout changes between toolkit versions.
#include <cstdio>
#include <cstdlib>
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

// Small per-call uploads (token ids, rope positions) without cudaMalloc/cudaFree, whose free
// synchronizes the device. Bytes go through a pinned ring and an async copy queued on the default
// stream ahead of the kernel that reads them. When the ring wraps, the stream is synchronized once
// so no queued copy still reads a slot being overwritten.
#include <mutex>

namespace {
struct StageRing {
    std::mutex mu;
    unsigned char* host = nullptr;
    unsigned char* dev = nullptr;
    size_t size = 0, off = 0;
};
StageRing g_stage;
constexpr size_t kStageBytes = 8u << 20;
}  // namespace

VELOX_API int velox_cuda_stage(const void* src, size_t bytes, void** dev_out) {
    std::lock_guard<std::mutex> lock(g_stage.mu);
    if (bytes > kStageBytes / 2) return VELOX_ERR_UNSUPPORTED;
    if (!g_stage.host) {
        if (cudaError_t e = cudaMallocHost(&g_stage.host, kStageBytes)) return static_cast<int>(e);
        if (cudaError_t e = cudaMalloc(&g_stage.dev, kStageBytes)) return static_cast<int>(e);
        g_stage.size = kStageBytes;
    }
    const size_t n = (bytes + 255) / 256 * 256;
    if (g_stage.off + n > g_stage.size) {
        if (cudaError_t e = cudaStreamSynchronize(0)) return static_cast<int>(e);
        g_stage.off = 0;
    }
    std::memcpy(g_stage.host + g_stage.off, src, bytes);
    void* d = g_stage.dev + g_stage.off;
    if (cudaError_t e = cudaMemcpyAsync(d, g_stage.host + g_stage.off, bytes, cudaMemcpyHostToDevice, 0))
        return static_cast<int>(e);
    g_stage.off += n;
    *dev_out = d;
    return 0;
}

// CUDA events for op timing (VELOX_GPU_TIMING): recorded on the default stream around each op.
VELOX_API int velox_cuda_event_create(void** e) {
    return static_cast<int>(cudaEventCreate(reinterpret_cast<cudaEvent_t*>(e)));
}
VELOX_API int velox_cuda_event_record(void* e) { return static_cast<int>(cudaEventRecord(static_cast<cudaEvent_t>(e), 0)); }
VELOX_API int velox_cuda_event_elapsed(void* start, void* end, float* ms) {
    if (cudaError_t err = cudaEventSynchronize(static_cast<cudaEvent_t>(end))) return static_cast<int>(err);
    return static_cast<int>(cudaEventElapsedTime(ms, static_cast<cudaEvent_t>(start), static_cast<cudaEvent_t>(end)));
}
VELOX_API int velox_cuda_event_destroy(void* e) { return static_cast<int>(cudaEventDestroy(static_cast<cudaEvent_t>(e))); }

// CUDA graphs over the (per-thread) default stream. A step is captured, then either patched into
// the executable graph of its slot (same topology, new parameters: cudaGraphExecUpdate) or
// instantiated afresh, and launched as one submission instead of one per kernel.
namespace {
constexpr int kGraphSlots = 256;
cudaGraphExec_t g_exec[kGraphSlots] = {};
}  // namespace

// Waits for queued work (so the staging ring can restart) and starts capturing. Relaxed mode lets
// grow-only scratch allocate during capture.
VELOX_API int velox_cuda_graph_begin() {
    if (cudaError_t e = cudaStreamSynchronize(cudaStreamPerThread)) return static_cast<int>(e);
    {
        std::lock_guard<std::mutex> lock(g_stage.mu);
        g_stage.off = 0;
    }
    return static_cast<int>(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeRelaxed));
}

// Ends the capture and launches it as slot `slot`'s graph.
VELOX_API int velox_cuda_graph_end_launch(int slot) {
    if (slot < 0 || slot >= kGraphSlots) return VELOX_ERR_UNSUPPORTED;
    cudaGraph_t g = nullptr;
    if (cudaError_t e = cudaStreamEndCapture(cudaStreamPerThread, &g)) return static_cast<int>(e);
    cudaGraphExec_t& exec = g_exec[slot];
    if (exec != nullptr) {
        cudaGraphExecUpdateResultInfo info;
        if (cudaGraphExecUpdate(exec, g, &info) != cudaSuccess) {
            (void)cudaGetLastError();
            if (getenv("VELOX_GRAPH_DEBUG"))
                fprintf(stderr, "[graph] slot %d update failed (result %d, node %p)\n", slot,
                        static_cast<int>(info.result), static_cast<void*>(info.errorNode));
            cudaGraphExecDestroy(exec);
            exec = nullptr;
        }
    }
    if (exec == nullptr) {
        if (cudaError_t e = cudaGraphInstantiate(&exec, g, 0)) {
            cudaGraphDestroy(g);
            exec = nullptr;
            return static_cast<int>(e);
        }
    }
    cudaGraphDestroy(g);
    return static_cast<int>(cudaGraphLaunch(exec, cudaStreamPerThread));
}

// Ends a capture whose recording failed, discarding it.
VELOX_API int velox_cuda_graph_abort() {
    cudaGraph_t g = nullptr;
    cudaStreamEndCapture(cudaStreamPerThread, &g);
    if (g != nullptr) cudaGraphDestroy(g);
    return static_cast<int>(cudaGetLastError());
}
