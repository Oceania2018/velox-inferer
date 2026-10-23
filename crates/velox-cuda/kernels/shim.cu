// C ABI over the CUDA runtime. Rust binds only to these functions, never to cudaDeviceProp,
// whose layout changes between toolkit versions.
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_profiler_api.h>

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

// Device memory in use as nvidia-smi reports it (NVML, loaded at run time: no link dependency).
// Under WDDM cudaMemGetInfo also counts ~1 GiB the driver reserves, which nvidia-smi leaves out;
// budgets people set are in nvidia-smi's terms.
#ifdef _WIN32
VELOX_API int velox_nvml_used(size_t*, size_t*) { return VELOX_ERR_UNSUPPORTED; }
#else
#include <dlfcn.h>
namespace {
struct NvmlMemoryV1 {
    unsigned long long total, free, used;
};
struct NvmlMemoryV2 {
    unsigned int version;
    unsigned long long total, reserved, free, used;
};
using NvmlInit = int (*)();
using NvmlByPci = int (*)(const char*, void**);
using NvmlMemory = int (*)(void*, void*);
}  // namespace

VELOX_API int velox_nvml_used(size_t* used, size_t* total) {
    static void* lib = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if (lib == nullptr) return VELOX_ERR_UNSUPPORTED;
    static const auto init = reinterpret_cast<NvmlInit>(dlsym(lib, "nvmlInit_v2"));
    static const auto by_pci = reinterpret_cast<NvmlByPci>(dlsym(lib, "nvmlDeviceGetHandleByPciBusId_v2"));
    static const auto mem_v2 = reinterpret_cast<NvmlMemory>(dlsym(lib, "nvmlDeviceGetMemoryInfo_v2"));
    static const auto mem_v1 = reinterpret_cast<NvmlMemory>(dlsym(lib, "nvmlDeviceGetMemoryInfo"));
    static const int inited = init != nullptr ? init() : -1;
    if (inited != 0 || by_pci == nullptr) return VELOX_ERR_UNSUPPORTED;
    int dev = 0;
    char bus[32] = {0};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetPCIBusId(bus, sizeof bus, dev) != cudaSuccess)
        return VELOX_ERR_UNSUPPORTED;
    void* handle = nullptr;
    if (by_pci(bus, &handle) != 0) return VELOX_ERR_UNSUPPORTED;
    if (mem_v2 != nullptr) {
        NvmlMemoryV2 m{};
        m.version = static_cast<unsigned int>(sizeof(NvmlMemoryV2)) | (2u << 24);
        if (mem_v2(handle, &m) == 0) {
            *used = m.used;
            *total = m.total;
            return 0;
        }
    }
    if (mem_v1 != nullptr) {
        NvmlMemoryV1 m{};
        if (mem_v1(handle, &m) == 0) {
            *used = m.used;
            *total = m.total;
            return 0;
        }
    }
    return VELOX_ERR_UNSUPPORTED;
}
#endif

// Brackets the region an external profiler (ncu --profile-from-start off) collects.
VELOX_API int velox_cuda_profiler(int on) {
    return static_cast<int>(on ? cudaProfilerStart() : cudaProfilerStop());
}

// Zeroed pinned host memory mapped into the device address space (the same address on both
// sides under unified addressing, which is required).
VELOX_API int velox_cuda_host_alloc(void** ptr, size_t bytes) {
    void* h = nullptr;
    if (cudaError_t e = cudaHostAlloc(&h, bytes, cudaHostAllocMapped | cudaHostAllocPortable))
        return static_cast<int>(e);
    void* d = nullptr;
    if (cudaError_t e = cudaHostGetDevicePointer(&d, h, 0)) {
        cudaFreeHost(h);
        return static_cast<int>(e);
    }
    if (d != h) {
        cudaFreeHost(h);
        return VELOX_ERR_UNSUPPORTED;
    }
    memset(h, 0, bytes);
    *ptr = h;
    return 0;
}

VELOX_API int velox_cuda_free_host(void* ptr) { return static_cast<int>(cudaFreeHost(ptr)); }

// Virtual memory (driver cuMem* API, fetched through the runtime so nothing links libcuda).
namespace {
struct Vmm {
    PFN_cuMemGetAllocationGranularity_v10020 granularity = nullptr;
    PFN_cuMemAddressReserve_v10020 reserve = nullptr;
    PFN_cuMemAddressFree_v10020 address_free = nullptr;
    PFN_cuMemCreate_v10020 create = nullptr;
    PFN_cuMemRelease_v10020 release = nullptr;
    PFN_cuMemMap_v10020 map = nullptr;
    PFN_cuMemUnmap_v10020 unmap = nullptr;
    PFN_cuMemSetAccess_v10020 set_access = nullptr;
    bool ok = false;
};
template <typename F>
bool vmm_entry(const char* name, F& f) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q;
    if (cudaGetDriverEntryPointByVersion(name, &p, 12000, cudaEnableDefault, &q) != cudaSuccess ||
        q != cudaDriverEntryPointSuccess)
        return false;
    f = reinterpret_cast<F>(p);
    return true;
}
const Vmm& vmm() {
    static const Vmm v = [] {
        Vmm v;
        v.ok = vmm_entry("cuMemGetAllocationGranularity", v.granularity) &&
               vmm_entry("cuMemAddressReserve", v.reserve) && vmm_entry("cuMemAddressFree", v.address_free) &&
               vmm_entry("cuMemCreate", v.create) && vmm_entry("cuMemRelease", v.release) &&
               vmm_entry("cuMemMap", v.map) && vmm_entry("cuMemUnmap", v.unmap) &&
               vmm_entry("cuMemSetAccess", v.set_access);
        return v;
    }();
    return v;
}
CUmemAllocationProp vmm_prop() {
    int dev = 0;
    cudaGetDevice(&dev);
    CUmemAllocationProp p = {};
    p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    p.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    p.location.id = dev;
    return p;
}
// Driver codes share the runtime's numbering for the failures that matter here (2: out of memory).
int vmm_status(CUresult r) { return static_cast<int>(r); }
}  // namespace

// Chunk size in bytes (0: no virtual memory management on this device or driver).
VELOX_API int velox_vmm_granularity(size_t* bytes) {
    *bytes = 0;
    const Vmm& v = vmm();
    if (!v.ok) return 0;
    // Devices without virtual memory management fail the granularity query.
    const CUmemAllocationProp p = vmm_prop();
    return vmm_status(v.granularity(bytes, &p, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
}

VELOX_API int velox_vmm_reserve(void** va, size_t bytes) {
    CUdeviceptr p = 0;
    const int e = vmm_status(vmm().reserve(&p, bytes, 0, 0, 0));
    *va = reinterpret_cast<void*>(p);
    return e;
}

VELOX_API int velox_vmm_free(void* va, size_t bytes) {
    return vmm_status(vmm().address_free(reinterpret_cast<CUdeviceptr>(va), bytes));
}

VELOX_API int velox_vmm_create(unsigned long long* handle, size_t bytes) {
    const CUmemAllocationProp p = vmm_prop();
    CUmemGenericAllocationHandle h = 0;
    const int e = vmm_status(vmm().create(&h, bytes, &p, 0));
    *handle = h;
    return e;
}

VELOX_API int velox_vmm_release(unsigned long long handle) { return vmm_status(vmm().release(handle)); }

VELOX_API int velox_vmm_map(void* va, size_t bytes, unsigned long long handle) {
    const Vmm& v = vmm();
    const CUdeviceptr p = reinterpret_cast<CUdeviceptr>(va);
    if (const int e = vmm_status(v.map(p, bytes, 0, handle, 0))) return e;
    CUmemAccessDesc a = {};
    a.location = vmm_prop().location;
    a.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (const int e = vmm_status(v.set_access(p, bytes, &a, 1))) {
        v.unmap(p, bytes);
        return e;
    }
    return 0;
}

VELOX_API int velox_vmm_unmap(void* va, size_t bytes) {
    return vmm_status(vmm().unmap(reinterpret_cast<CUdeviceptr>(va), bytes));
}

VELOX_API int velox_cuda_malloc_zeroed(void** ptr, size_t bytes) {
    cudaError_t e = cudaMalloc(ptr, bytes);
    if (e != cudaSuccess) return static_cast<int>(e);
    return static_cast<int>(cudaMemset(*ptr, 0, bytes));
}

VELOX_API int velox_cuda_free(void* ptr) { return static_cast<int>(cudaFree(ptr)); }

VELOX_API int velox_cuda_htod(void* dst, const void* src, size_t bytes) {
    return static_cast<int>(cudaMemcpy(dst, src, bytes, cudaMemcpyDefault));
}

VELOX_API int velox_cuda_dtoh(void* dst, const void* src, size_t bytes) {
    return static_cast<int>(cudaMemcpy(dst, src, bytes, cudaMemcpyDefault));
}

VELOX_API int velox_cuda_dtod(void* dst, const void* src, size_t bytes) {
    // Default: either side may be mapped host memory (unified addressing).
    return static_cast<int>(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, 0));
}

// Many independent copies in one submission (CUDA 12.8+ batch API), on the per-thread stream.
VELOX_API int velox_cuda_copy_batch(void* const* dsts, const void* const* srcs, const size_t* sizes, size_t count) {
    if (count == 0) return 0;
    cudaMemcpyAttributes attr = {};
    attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    size_t idx = 0;
    return static_cast<int>(
        cudaMemcpyBatchAsync(dsts, srcs, sizes, count, &attr, &idx, 1, cudaStreamPerThread));
}

VELOX_API int velox_cuda_synchronize() { return static_cast<int>(cudaDeviceSynchronize()); }

VELOX_API const char* velox_cuda_error_string(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}

VELOX_API int velox_cuda_runtime_version(int* v) { return static_cast<int>(cudaRuntimeGetVersion(v)); }

VELOX_API int velox_cuda_driver_version(int* v) { return static_cast<int>(cudaDriverGetVersion(v)); }

// Small per-call uploads (token ids, positions, offsets) without cudaMalloc/cudaFree, whose free
// synchronizes the device. Outside graphs, bytes go through a pinned ring and an async copy queued
// on the default stream ahead of the kernel that reads them; when the ring wraps, the stream is
// synchronized once so no queued copy still reads a slot being overwritten.
//
// Inside a graph (capture or replay) bytes go to that graph's own pinned region instead, at the
// same offsets on every run, and no copy is queued: the graph's first node copies the whole region
// to the device when it executes. Replaying a graph therefore only rewrites the region on the host.
#include <mutex>
#include <utility>
#include <vector>

namespace {
struct StageRing {
    std::mutex mu;
    unsigned char* host = nullptr;
    unsigned char* dev = nullptr;
    size_t size = 0, off = 0;
};
StageRing g_stage;
constexpr size_t kStageBytes = 8u << 20;

constexpr size_t kRegionBytes = 64u << 10;
constexpr int kGraphSlots = 4096;
struct GraphSlot {
    cudaGraphExec_t exec = nullptr;
    unsigned char* host = nullptr;  // pinned staging region
    unsigned char* dev = nullptr;
    cudaEvent_t done = nullptr;     // recorded after each launch
    bool launched = false;
    size_t bytes = 0;               // device memory its executable graph took (measured)
};
std::vector<GraphSlot> g_slots(kGraphSlots);
size_t g_graph_bytes = 0;  // sum of the slots' bytes
int g_region = -1;  // slot whose region stage() fills, -1 outside graphs
size_t g_region_off = 0;

// Allocates the slot's region and event on first use and waits until its previous launch has
// consumed the region, so the host may rewrite it.
int prepare_slot(int slot) {
    GraphSlot& s = g_slots[slot];
    if (!s.host) {
        if (cudaError_t e = cudaMallocHost(&s.host, kRegionBytes)) return static_cast<int>(e);
        std::memset(s.host, 0, kRegionBytes);
        if (cudaError_t e = cudaMalloc(&s.dev, kRegionBytes)) return static_cast<int>(e);
        if (cudaError_t e = cudaEventCreateWithFlags(&s.done, cudaEventDisableTiming)) return static_cast<int>(e);
    }
    if (s.launched) {
        if (cudaError_t e = cudaEventSynchronize(s.done)) return static_cast<int>(e);
    }
    return 0;
}
}  // namespace

VELOX_API int velox_cuda_stage(const void* src, size_t bytes, void** dev_out) {
    std::lock_guard<std::mutex> lock(g_stage.mu);
    if (g_region >= 0) {
        GraphSlot& s = g_slots[g_region];
        const size_t n = (bytes + 15) / 16 * 16;
        if (g_region_off + n > kRegionBytes) return VELOX_ERR_UNSUPPORTED;
        std::memcpy(s.host + g_region_off, src, bytes);
        *dev_out = s.dev + g_region_off;
        g_region_off += n;
        return 0;
    }
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

// 1 while a graph is being captured or replayed (its region is the staging target).
VELOX_API int velox_cuda_in_graph() {
    std::lock_guard<std::mutex> lock(g_stage.mu);
    return g_region >= 0 ? 1 : 0;
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

// CUDA graphs over the (per-thread) default stream, one per slot. A step is captured (again
// whenever the caller cannot promise an identical recording), patched into the slot's executable
// graph (cudaGraphExecUpdate) or instantiated, and launched as one submission. A static step is
// afterwards replayed: the caller re-runs its host code with launches skipped, so staged inputs
// land in the slot's region, and the executable graph is launched as it is.

// Starts capturing slot `slot`; its region becomes the staging target and is copied to the device
// by the graph's first node. Relaxed mode lets grow-only scratch allocate during capture.
VELOX_API int velox_cuda_graph_begin(int slot) {
    if (slot < 0 || slot >= kGraphSlots) return VELOX_ERR_UNSUPPORTED;
    std::lock_guard<std::mutex> lock(g_stage.mu);
    if (const int e = prepare_slot(slot)) return e;
    if (cudaError_t e = cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeRelaxed))
        return static_cast<int>(e);
    GraphSlot& s = g_slots[slot];
    if (cudaError_t e = cudaMemcpyAsync(s.dev, s.host, kRegionBytes, cudaMemcpyHostToDevice, cudaStreamPerThread)) {
        cudaGraph_t g = nullptr;
        cudaStreamEndCapture(cudaStreamPerThread, &g);
        if (g != nullptr) cudaGraphDestroy(g);
        return static_cast<int>(e);
    }
    g_region = slot;
    g_region_off = 0;
    return 0;
}

// Starts a replay of slot `slot` (captured before): staging goes to its region again.
VELOX_API int velox_cuda_graph_replay_begin(int slot) {
    if (slot < 0 || slot >= kGraphSlots || g_slots[slot].exec == nullptr) return VELOX_ERR_UNSUPPORTED;
    std::lock_guard<std::mutex> lock(g_stage.mu);
    if (const int e = prepare_slot(slot)) return e;
    g_region = slot;
    g_region_off = 0;
    return 0;
}

namespace {
// VELOX_GAP_TRACE=1: GPU time between consecutive graph launches (the stream idles while the host
// prepares the next one), reported every 2000 launches; gaps over 20 ms (between requests) are
// left out.
struct GapTrace {
    bool on = getenv("VELOX_GAP_TRACE") != nullptr;
    std::vector<std::pair<cudaEvent_t, cudaEvent_t>> ev;
    void report() {
        double busy = 0, gaps = 0;
        int counted = 0;
        for (size_t i = 0; i < ev.size(); ++i) {
            float ms = 0;
            cudaEventSynchronize(ev[i].second);
            cudaEventElapsedTime(&ms, ev[i].first, ev[i].second);
            busy += ms;
            if (i + 1 < ev.size()) {
                float g = 0;
                cudaEventSynchronize(ev[i + 1].first);
                cudaEventElapsedTime(&g, ev[i].second, ev[i + 1].first);
                if (g < 20.0f) gaps += g, ++counted;
            }
        }
        fprintf(stderr, "[gaps] %zu launches: busy %.1f ms, gaps %.1f ms (%.1f us per gap)\n", ev.size(), busy, gaps,
                counted ? 1000.0 * gaps / counted : 0.0);
        for (auto& p : ev) cudaEventDestroy(p.first), cudaEventDestroy(p.second);
        ev.clear();
    }
};
GapTrace g_gaps;

int launch_slot(int slot) {
    GraphSlot& s = g_slots[slot];
    cudaEvent_t t0 = nullptr, t1 = nullptr;
    if (g_gaps.on) {
        cudaEventCreate(&t0);
        cudaEventRecord(t0, cudaStreamPerThread);
    }
    if (cudaError_t e = cudaGraphLaunch(s.exec, cudaStreamPerThread)) return static_cast<int>(e);
    if (g_gaps.on) {
        cudaEventCreate(&t1);
        cudaEventRecord(t1, cudaStreamPerThread);
        g_gaps.ev.emplace_back(t0, t1);
        if (g_gaps.ev.size() >= 2000) g_gaps.report();
    }
    s.launched = true;
    if (cudaError_t e = cudaEventRecord(s.done, cudaStreamPerThread)) return static_cast<int>(e);
    // WDDM (Windows, WSL) batches submissions until something flushes them; a stream query does,
    // so a step's first segment starts while the host still submits the rest.
    const cudaError_t q = cudaStreamQuery(cudaStreamPerThread);
    return q == cudaErrorNotReady ? 0 : static_cast<int>(q);
}
}  // namespace

// Ends the capture and launches it as slot `slot`'s graph.
VELOX_API int velox_cuda_graph_end_launch(int slot) {
    if (slot < 0 || slot >= kGraphSlots) return VELOX_ERR_UNSUPPORTED;
    std::lock_guard<std::mutex> lock(g_stage.mu);
    g_region = -1;
    cudaGraph_t g = nullptr;
    if (cudaError_t e = cudaStreamEndCapture(cudaStreamPerThread, &g)) return static_cast<int>(e);
    cudaGraphExec_t& exec = g_slots[slot].exec;
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
        // Executable graphs take device memory the caller budgets (velox_cuda_graph_bytes): the
        // free-memory drop across instantiation is this graph's share (driver pages, so coarse).
        GraphSlot& s = g_slots[slot];
        g_graph_bytes -= s.bytes;
        s.bytes = 0;
        size_t before = 0, after = 0, total = 0;
        cudaMemGetInfo(&before, &total);
        if (cudaError_t e = cudaGraphInstantiate(&exec, g, 0)) {
            cudaGraphDestroy(g);
            exec = nullptr;
            return static_cast<int>(e);
        }
        cudaMemGetInfo(&after, &total);
        s.bytes = before > after ? before - after : 0;
        g_graph_bytes += s.bytes;
    }
    cudaGraphDestroy(g);
    return launch_slot(slot);
}

// Launches slot `slot`'s graph as captured, its region rewritten by the replayed host code.
VELOX_API int velox_cuda_graph_replay_launch(int slot) {
    if (slot < 0 || slot >= kGraphSlots || g_slots[slot].exec == nullptr) return VELOX_ERR_UNSUPPORTED;
    std::lock_guard<std::mutex> lock(g_stage.mu);
    g_region = -1;
    return launch_slot(slot);
}

// Releases slot `slot`'s executable graph (after its last launch completes); its staging region
// stays for the slot's next capture.
VELOX_API int velox_cuda_graph_release(int slot) {
    if (slot < 0 || slot >= kGraphSlots) return VELOX_ERR_UNSUPPORTED;
    std::lock_guard<std::mutex> lock(g_stage.mu);
    GraphSlot& s = g_slots[slot];
    if (s.launched) {
        if (cudaError_t e = cudaEventSynchronize(s.done)) return static_cast<int>(e);
        s.launched = false;
    }
    if (s.exec != nullptr) {
        if (cudaError_t e = cudaGraphExecDestroy(s.exec)) return static_cast<int>(e);
        s.exec = nullptr;
    }
    g_graph_bytes -= s.bytes;
    s.bytes = 0;
    return 0;
}

// Device bytes the slots' executable graphs hold.
VELOX_API size_t velox_cuda_graph_bytes() {
    std::lock_guard<std::mutex> lock(g_stage.mu);
    return g_graph_bytes;
}

// Ends a capture or replay whose host code failed, discarding a capture.
VELOX_API int velox_cuda_graph_abort() {
    std::lock_guard<std::mutex> lock(g_stage.mu);
    g_region = -1;
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(cudaStreamPerThread, &status) == cudaSuccess && status != cudaStreamCaptureStatusNone) {
        cudaGraph_t g = nullptr;
        cudaStreamEndCapture(cudaStreamPerThread, &g);
        if (g != nullptr) cudaGraphDestroy(g);
    }
    return static_cast<int>(cudaGetLastError());
}
