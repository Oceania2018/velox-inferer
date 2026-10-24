//! CUDA backend.
//!
//! [`CudaBackend`] validates every call and forwards it to a [`CudaKernels`] implementation. Each
//! `CudaKernels` method defaults to the portable kernel, so a hardware branch overrides only the
//! ops it has tuned and inherits everything new that lands on main.

use std::collections::{HashMap, HashSet};
use std::ffi::{c_char, c_void, CStr};
use std::sync::{Arc, Mutex};

use velox_core::backend::{
    ActQuant, AttnPrecision, AttnShape, Backend, BackendKind, BlockAttnShape, ConvShape,
    Copy2dShape, CopySpec, DilatedConvShape, DynConvShape, ExpertBank, FullAttnShape,
    GdnReplayLayer, GdnShape, GeluKind, HcCombineShape, KvStoreShape, LinearDst, MoeRouteShape,
    NormKind, QsaPoolShape, QsaSelectShape, RopePosShape, RopeShape, RowGateShape, WeightView,
    GRAPH_STATIC,
};
use velox_core::{
    check, profile, DeviceProfile, Error, GpuArch, KvFormat, Op, Result, VariantInfo,
};

pub mod portable;
mod timing;

mod ffi {
    use super::*;

    extern "C" {
        pub fn velox_cuda_device_count(count: *mut i32) -> i32;
        pub fn velox_cuda_device_info(
            device: i32,
            name: *mut c_char,
            name_len: i32,
            major: *mut i32,
            minor: *mut i32,
            sm_count: *mut i32,
            total_mem: *mut usize,
        ) -> i32;
        pub fn velox_cuda_set_device(device: i32) -> i32;
        pub fn velox_cuda_mem_info(free: *mut usize, total: *mut usize) -> i32;
        pub fn velox_nvml_used(used: *mut usize, total: *mut usize) -> i32;
        pub fn velox_cuda_profiler(on: i32) -> i32;
        pub fn velox_cuda_malloc_zeroed(ptr: *mut *mut c_void, bytes: usize) -> i32;
        pub fn velox_cuda_host_alloc(ptr: *mut *mut c_void, bytes: usize) -> i32;
        pub fn velox_cuda_copy_batch(
            dsts: *const *mut c_void,
            srcs: *const *const c_void,
            sizes: *const usize,
            count: usize,
        ) -> i32;
        pub fn velox_cuda_free_host(ptr: *mut c_void) -> i32;
        pub fn velox_vmm_granularity(bytes: *mut usize) -> i32;
        pub fn velox_vmm_reserve(va: *mut *mut c_void, bytes: usize) -> i32;
        pub fn velox_vmm_free(va: *mut c_void, bytes: usize) -> i32;
        pub fn velox_vmm_create(handle: *mut u64, bytes: usize) -> i32;
        pub fn velox_vmm_release(handle: u64) -> i32;
        pub fn velox_vmm_map(va: *mut c_void, bytes: usize, handle: u64) -> i32;
        pub fn velox_vmm_unmap(va: *mut c_void, bytes: usize) -> i32;
        pub fn velox_cuda_free(ptr: *mut c_void) -> i32;
        pub fn velox_cuda_htod(dst: *mut c_void, src: *const c_void, bytes: usize) -> i32;
        pub fn velox_cuda_dtoh(dst: *mut c_void, src: *const c_void, bytes: usize) -> i32;
        pub fn velox_cuda_htod_async(dst: *mut c_void, src: *const c_void, bytes: usize) -> i32;
        pub fn velox_cuda_stage(src: *const c_void, bytes: usize, dev: *mut *mut c_void) -> i32;
        pub fn velox_cuda_event_create(e: *mut *mut c_void) -> i32;
        pub fn velox_cuda_event_record(e: *mut c_void) -> i32;
        pub fn velox_cuda_event_elapsed(a: *mut c_void, b: *mut c_void, ms: *mut f32) -> i32;
        pub fn velox_cuda_event_destroy(e: *mut c_void) -> i32;
        pub fn velox_cuda_dtod(dst: *mut c_void, src: *const c_void, bytes: usize) -> i32;
        pub fn velox_cuda_synchronize() -> i32;
        pub fn velox_cuda_error_string(code: i32) -> *const c_char;
        pub fn velox_cuda_runtime_version(v: *mut i32) -> i32;
        pub fn velox_cuda_driver_version(v: *mut i32) -> i32;
        pub fn velox_cuda_graph_begin(slot: i32) -> i32;
        pub fn velox_cuda_graph_end_launch(slot: i32) -> i32;
        pub fn velox_cuda_graph_replay_begin(slot: i32) -> i32;
        pub fn velox_cuda_graph_replay_launch(slot: i32) -> i32;
        pub fn velox_cuda_graph_abort() -> i32;
        pub fn velox_cuda_graph_release(slot: i32) -> i32;
        pub fn velox_cuda_graph_bytes() -> usize;
    }
}

/// Launcher return codes below zero (see velox_common.cuh).
const VELOX_ERR_UNSUPPORTED: i32 = -1;

/// Converts a CUDA/launcher return code into a Result. Public for variant launchers.
pub fn status(code: i32, what: &str) -> Result<()> {
    match code {
        0 => Ok(()),
        VELOX_ERR_UNSUPPORTED => Err(Error::Unsupported(format!(
            "{what}: shape not supported by this kernel"
        ))),
        c => {
            // SAFETY: cudaGetErrorString returns a static NUL-terminated string for any code.
            let msg = unsafe { CStr::from_ptr(ffi::velox_cuda_error_string(c)) };
            Err(Error::Backend(format!(
                "{what}: CUDA error {c}: {}",
                msg.to_string_lossy()
            )))
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceInfo {
    pub index: i32,
    pub name: String,
    pub arch: GpuArch,
    pub sm_count: u32,
    pub total_mem: u64,
}

impl DeviceInfo {
    pub fn profile(&self) -> Option<&'static DeviceProfile> {
        profile::detect(self.arch, self.sm_count, self.total_mem)
    }
}

pub fn devices() -> Result<Vec<DeviceInfo>> {
    let mut count = 0;
    // SAFETY: out-pointer to a live local.
    status(
        unsafe { ffi::velox_cuda_device_count(&mut count) },
        "cudaGetDeviceCount",
    )?;
    (0..count)
        .map(|index| {
            let mut name = [0 as c_char; 256];
            let (mut major, mut minor, mut sms, mut total) = (0, 0, 0, 0usize);
            // SAFETY: all out-pointers are live locals; name length matches the buffer.
            status(
                unsafe {
                    ffi::velox_cuda_device_info(
                        index,
                        name.as_mut_ptr(),
                        256,
                        &mut major,
                        &mut minor,
                        &mut sms,
                        &mut total,
                    )
                },
                "cudaGetDeviceProperties",
            )?;
            // SAFETY: the shim always NUL-terminates within the buffer.
            let name = unsafe { CStr::from_ptr(name.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            Ok(DeviceInfo {
                index,
                name,
                arch: GpuArch::Cuda {
                    major: major as u32,
                    minor: minor as u32,
                },
                sm_count: sms as u32,
                total_mem: total as u64,
            })
        })
        .collect()
}

/// (runtime, driver) versions as reported by CUDA, e.g. (13010, 13020).
pub fn versions() -> Result<(i32, i32)> {
    let (mut rt, mut drv) = (0, 0);
    // SAFETY: out-pointers to live locals.
    status(
        unsafe { ffi::velox_cuda_runtime_version(&mut rt) },
        "cudaRuntimeGetVersion",
    )?;
    status(
        unsafe { ffi::velox_cuda_driver_version(&mut drv) },
        "cudaDriverGetVersion",
    )?;
    Ok((rt, drv))
}

/// (free, total) bytes on the current device.
pub fn mem_info() -> Result<(u64, u64)> {
    let (mut free, mut total) = (0usize, 0usize);
    // SAFETY: out-pointers to live locals.
    status(
        unsafe { ffi::velox_cuda_mem_info(&mut free, &mut total) },
        "cudaMemGetInfo",
    )?;
    Ok((free as u64, total as u64))
}

/// Starts or stops collection by an external profiler started with collection off
/// (`ncu --profile-from-start off`).
pub fn profiler(on: bool) -> Result<()> {
    // SAFETY: no arguments besides the flag.
    status(
        unsafe { ffi::velox_cuda_profiler(i32::from(on)) },
        "cudaProfiler",
    )
}

/// Refuses to run a variant's arch-specific kernels on a device they were not compiled for.
/// `VELOX_ALLOW_ARCH_MISMATCH=1` lets PTX-carrying kernels JIT on another architecture; it exists
/// for correctness testing of one branch on another card, never for measurements.
pub fn require_arch(dev: &DeviceInfo, variant: &VariantInfo) -> Result<()> {
    if variant.runs_on(dev.arch)
        || std::env::var("VELOX_ALLOW_ARCH_MISMATCH").is_ok_and(|v| v == "1")
    {
        return Ok(());
    }
    let built: Vec<String> = variant.archs.iter().map(|a| a.to_string()).collect();
    let hint = dev
        .profile()
        .and_then(|p| p.tuned_branch)
        .map(|b| format!("check out branch '{b}'"))
        .unwrap_or_else(|| "use main (generic portable kernels)".into());
    Err(Error::Unsupported(format!(
        "this build is variant '{}' (branch '{}') compiled for {}; device {} is {}: {hint}",
        variant.id,
        variant.branch,
        built.join(", "),
        dev.name,
        dev.arch
    )))
}

/// Device allocation of f32 elements, freed on drop.
pub struct CudaBuffer {
    ptr: *mut f32,
    len: usize,
    /// Pinned, mapped system memory (see `Backend::alloc_host`) rather than device memory.
    host: bool,
    /// A reserved address range backed chunk by chunk (see `Backend::alloc_virtual`).
    vm: Option<Box<VmRange>>,
}

/// The chunks of a virtual [`CudaBuffer`]: which are mapped, and the range's size.
struct VmRange {
    chunk_bytes: usize,
    mapped: Vec<bool>,
}

/// Physical device memory of one virtual-memory chunk (released on drop; the memory stays alive
/// while a buffer still maps it).
pub struct CudaChunk {
    handle: u64,
}

impl Drop for CudaChunk {
    fn drop(&mut self) {
        // SAFETY: the handle came from cuMemCreate and is released exactly once.
        unsafe { ffi::velox_vmm_release(self.handle) };
        CHUNK_BYTES.fetch_sub(vm_chunk_bytes(), std::sync::atomic::Ordering::Relaxed);
    }
}

// SAFETY: a device pointer is a plain address; synchronisation is CUDA's stream ordering.
unsafe impl Send for CudaBuffer {}
unsafe impl Sync for CudaBuffer {}

/// Bytes of device buffers (cudaMalloc) alive and at their peak, and of physical chunks alive:
/// what the engine's memory log reports, to tell its own growth from the driver's.
static BUFFER_BYTES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static BUFFER_PEAK: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static CHUNK_BYTES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

impl CudaBuffer {
    fn new(len: usize) -> Result<Self> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        if len > 0 {
            let bytes = len
                .checked_mul(4)
                .ok_or_else(|| Error::Shape("allocation size overflows".into()))?;
            // SAFETY: out-pointer to a live local.
            status(
                unsafe { ffi::velox_cuda_malloc_zeroed(&mut ptr, bytes) },
                "cudaMalloc",
            )?;
            use std::sync::atomic::Ordering::Relaxed;
            let live = BUFFER_BYTES.fetch_add(bytes, Relaxed) + bytes;
            BUFFER_PEAK.fetch_max(live, Relaxed);
        }
        Ok(CudaBuffer {
            ptr: ptr.cast(),
            len,
            host: false,
            vm: None,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_ptr(&self) -> *const f32 {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.ptr
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        if let Some(vm) = self.vm.take() {
            // SAFETY: the range came from cuMemAddressReserve; mapped chunks are unmapped before
            // the range is freed. Callers synchronize before dropping a buffer work still uses.
            unsafe {
                for (i, _) in vm.mapped.iter().enumerate().filter(|(_, m)| **m) {
                    let va = self.ptr.cast::<u8>().add(i * vm.chunk_bytes);
                    ffi::velox_vmm_unmap(va.cast(), vm.chunk_bytes);
                }
                ffi::velox_vmm_free(self.ptr.cast(), vm.mapped.len() * vm.chunk_bytes);
            }
            return;
        }
        if !self.ptr.is_null() {
            // SAFETY: ptr came from cudaMalloc (or cudaHostAlloc) and is freed exactly once.
            // Errors at teardown (e.g. after a sticky kernel fault) cannot be reported from Drop.
            unsafe {
                if self.host {
                    ffi::velox_cuda_free_host(self.ptr.cast());
                } else {
                    ffi::velox_cuda_free(self.ptr.cast());
                    BUFFER_BYTES.fetch_sub(self.len * 4, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
    }
}

impl CudaBuffer {
    /// Zeroed pinned host memory the device addresses directly.
    fn new_host(len: usize) -> Result<Self> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        if len > 0 {
            let bytes = len
                .checked_mul(4)
                .ok_or_else(|| Error::Shape("allocation size overflows".into()))?;
            // SAFETY: out-pointer to a live local.
            status(
                unsafe { ffi::velox_cuda_host_alloc(&mut ptr, bytes) },
                "cudaHostAlloc (mapped)",
            )?;
        }
        Ok(CudaBuffer {
            ptr: ptr.cast(),
            len,
            host: true,
            vm: None,
        })
    }
}

/// Kernel set behind [`CudaBackend`]. Shapes are validated before these are called.
///
/// Every method defaults to the portable implementation. A hardware variant overrides the ops it
/// has tuned, may fall back to `portable::*` for shapes its kernel does not cover, and lists the
/// ops it overrides in [`CudaKernels::native_ops`].
#[allow(clippy::too_many_arguments)]
pub trait CudaKernels: Send + Sync {
    fn native_ops(&self) -> &'static [Op] {
        &[]
    }

    fn matmul(
        &self,
        x: &CudaBuffer,
        w: &CudaBuffer,
        y: &mut CudaBuffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        portable::matmul(x, w, y, m, n, k)
    }

    fn linear(
        &self,
        x: &CudaBuffer,
        w: &WeightView<'_, CudaBuffer>,
        y: &mut CudaBuffer,
        m: usize,
    ) -> Result<()> {
        portable::linear(x, w, y, m)
    }

    /// The activation rounding `linear` applies; see [`Backend::linear_act`].
    fn linear_act(&self, _w: &WeightView<'_, CudaBuffer>, _m: usize) -> ActQuant {
        ActQuant::Exact
    }

    /// Whether `linears` serves these destinations at `m` rows (every part rounded as `linear`
    /// rounds it). When it does not, the backend composes the op from `linear` and `copy2d`.
    fn serves_linears(&self, _dsts: &[LinearDst<'_, '_, '_, CudaBuffer>], _m: usize) -> bool {
        false
    }

    /// [`Backend::linears`] without scratch, for destinations `serves_linears` accepted (`xb`:
    /// `x`'s bf16 twin when the caller has one).
    fn linears(
        &self,
        _x: &CudaBuffer,
        _xb: Option<&CudaBuffer>,
        _xq: Option<(&CudaBuffer, u32)>,
        _dsts: &mut [LinearDst<'_, '_, '_, CudaBuffer>],
        _m: usize,
    ) -> Result<()> {
        Err(Error::Unsupported("linears".into()))
    }

    /// See [`Backend::takes_bf16`].
    fn takes_bf16(&self, _w: &WeightView<'_, CudaBuffer>, _m: usize) -> bool {
        false
    }

    /// See [`Backend::linear_bf16`].
    fn linear_bf16(
        &self,
        _xb: &CudaBuffer,
        _w: &WeightView<'_, CudaBuffer>,
        _y: &mut CudaBuffer,
        _m: usize,
    ) -> Result<()> {
        Err(Error::Unsupported("linear_bf16".into()))
    }

    /// See [`Backend::swiglu_twin`].
    fn swiglu_twin(
        &self,
        _gate: &WeightView<'_, CudaBuffer>,
        _up: &WeightView<'_, CudaBuffer>,
        _m: usize,
    ) -> bool {
        false
    }

    /// See [`Backend::linear_swiglu_twin`].
    #[allow(clippy::too_many_arguments)]
    fn linear_swiglu_twin(
        &self,
        _xb: &CudaBuffer,
        _gate: &WeightView<'_, CudaBuffer>,
        _up: &WeightView<'_, CudaBuffer>,
        _out: &mut CudaBuffer,
        _outb: &mut CudaBuffer,
        _m: usize,
    ) -> Result<()> {
        Err(Error::Unsupported("linear_swiglu_twin".into()))
    }

    /// See [`Backend::int8_twin_layout`].
    fn int8_twin_layout(&self, _w: &WeightView<'_, CudaBuffer>, _m: usize) -> Option<u32> {
        None
    }
    /// See [`Backend::int8_twin_slots`].
    fn int8_twin_slots(&self, _m: usize, _k: usize, _layout: u32) -> usize {
        0
    }
    /// See [`Backend::add_rms_norm_q8`].
    #[allow(clippy::too_many_arguments)]
    fn add_rms_norm_q8(
        &self,
        _acc: &mut CudaBuffer,
        _x: &CudaBuffer,
        _weight: &CudaBuffer,
        _y: &mut CudaBuffer,
        _yq: &mut CudaBuffer,
        _layout: u32,
        _rows: usize,
        _dim: usize,
        _eps: f32,
        _kind: NormKind,
    ) -> Result<()> {
        Err(Error::Unsupported("add_rms_norm_q8".into()))
    }
    /// See [`Backend::linear_q8`].
    fn linear_q8(
        &self,
        _xq: &CudaBuffer,
        _layout: u32,
        _w: &WeightView<'_, CudaBuffer>,
        _y: &mut CudaBuffer,
        _m: usize,
    ) -> Result<()> {
        Err(Error::Unsupported("linear_q8".into()))
    }
    /// See [`Backend::linear_swiglu_q8`].
    fn linear_swiglu_q8(
        &self,
        _xq: &CudaBuffer,
        _layout: u32,
        _gate: &WeightView<'_, CudaBuffer>,
        _up: &WeightView<'_, CudaBuffer>,
        _out: &mut CudaBuffer,
        _m: usize,
    ) -> Result<()> {
        Err(Error::Unsupported("linear_swiglu_q8".into()))
    }

    /// See [`Backend::add_rms_norm_twin`].
    #[allow(clippy::too_many_arguments)]
    fn add_rms_norm_twin(
        &self,
        acc: &mut CudaBuffer,
        x: &CudaBuffer,
        weight: &CudaBuffer,
        y: &mut CudaBuffer,
        yb: &mut CudaBuffer,
        rows: usize,
        dim: usize,
        eps: f32,
        kind: NormKind,
    ) -> Result<()> {
        portable::add_rms_norm_twin(acc, x, weight, y, yb, rows, dim, eps, kind)
    }

    /// See [`Backend::rms_norm_gated_twin`].
    #[allow(clippy::too_many_arguments)]
    fn rms_norm_gated_twin(
        &self,
        x: &CudaBuffer,
        weight: &CudaBuffer,
        gate: &mut CudaBuffer,
        gb: &mut CudaBuffer,
        rows: usize,
        dim: usize,
        eps: f32,
        kind: NormKind,
    ) -> Result<()> {
        portable::rms_norm_gated_twin(x, weight, gate, gb, rows, dim, eps, kind)
    }

    /// See [`Backend::sigmoid_mul_twin`].
    fn sigmoid_mul_twin(
        &self,
        x: &mut CudaBuffer,
        gate: &CudaBuffer,
        xb: &mut CudaBuffer,
        len: usize,
    ) -> Result<()> {
        portable::sigmoid_mul_twin(x, gate, xb, len)
    }

    /// A fused [`Backend::linear_swiglu`] (no scratch). The default reports `Unsupported`, and the
    /// backend composes the op from `linear` and `swiglu`.
    fn linear_swiglu(
        &self,
        _x: &CudaBuffer,
        _gate: &WeightView<'_, CudaBuffer>,
        _up: &WeightView<'_, CudaBuffer>,
        _out: &mut CudaBuffer,
        _m: usize,
    ) -> Result<()> {
        Err(Error::Unsupported("fused linear_swiglu".into()))
    }

    /// The activation rounding a fused `linear_swiglu` applies, `None` when it is not fused at
    /// these rows (the backend then reports the composed op's).
    fn linear_swiglu_act(
        &self,
        _gate: &WeightView<'_, CudaBuffer>,
        _up: &WeightView<'_, CudaBuffer>,
        _m: usize,
    ) -> Option<ActQuant> {
        None
    }

    fn embed(
        &self,
        ids: &[u32],
        table: &WeightView<'_, CudaBuffer>,
        out: &mut CudaBuffer,
    ) -> Result<()> {
        portable::embed(ids, table, out)
    }

    /// See [`Backend::embed_from`].
    fn embed_from(
        &self,
        ids: &CudaBuffer,
        offset: usize,
        count: usize,
        table: &WeightView<'_, CudaBuffer>,
        out: &mut CudaBuffer,
    ) -> Result<()> {
        portable::embed_from(ids, offset, count, table, out)
    }

    fn rms_norm(
        &self,
        x: &CudaBuffer,
        weight: &CudaBuffer,
        y: &mut CudaBuffer,
        rows: usize,
        dim: usize,
        eps: f32,
        kind: NormKind,
    ) -> Result<()> {
        portable::rms_norm(x, weight, y, rows, dim, eps, kind)
    }

    #[allow(clippy::too_many_arguments)]
    fn rms_norm_gated(
        &self,
        x: &CudaBuffer,
        weight: &CudaBuffer,
        gate: &mut CudaBuffer,
        rows: usize,
        dim: usize,
        eps: f32,
        kind: NormKind,
    ) -> Result<()> {
        portable::rms_norm_gated(x, weight, gate, rows, dim, eps, kind)
    }

    /// Residual add then RMS norm in one pass; see [`Backend::add_rms_norm`].
    #[allow(clippy::too_many_arguments)]
    fn add_rms_norm(
        &self,
        acc: &mut CudaBuffer,
        x: &CudaBuffer,
        weight: &CudaBuffer,
        y: &mut CudaBuffer,
        rows: usize,
        dim: usize,
        eps: f32,
        kind: NormKind,
    ) -> Result<()> {
        portable::add_rms_norm(acc, x, weight, y, rows, dim, eps, kind)
    }

    fn rope(&self, x: &mut CudaBuffer, s: RopeShape) -> Result<()> {
        portable::rope(x, s)
    }

    fn silu_mul(&self, gate: &mut CudaBuffer, up: &CudaBuffer, len: usize) -> Result<()> {
        portable::silu_mul(gate, up, len)
    }

    fn sigmoid_mul(&self, x: &mut CudaBuffer, gate: &CudaBuffer, len: usize) -> Result<()> {
        portable::sigmoid_mul(x, gate, len)
    }

    fn add(&self, acc: &mut CudaBuffer, x: &CudaBuffer, len: usize) -> Result<()> {
        portable::add(acc, x, len)
    }

    fn copy(
        &self,
        src: &CudaBuffer,
        src_offset: usize,
        dst: &mut CudaBuffer,
        dst_offset: usize,
        len: usize,
    ) -> Result<()> {
        portable::copy(src, src_offset, dst, dst_offset, len)
    }

    fn attention(
        &self,
        q: &CudaBuffer,
        k: &CudaBuffer,
        v: &CudaBuffer,
        out: &mut CudaBuffer,
        s: AttnShape,
    ) -> Result<()> {
        portable::attention(q, k, v, out, s)
    }

    /// The operand precision of `attention`; see [`Backend::attention_precision`].
    fn attention_precision(&self, _s: AttnShape) -> AttnPrecision {
        AttnPrecision::Exact
    }

    /// Encodes K/V rows into caches; see [`KvStoreShape`]. Must be bit-exact with
    /// `velox_core::kv::encode_row`.
    fn kv_store(
        &self,
        k: &CudaBuffer,
        v: &CudaBuffer,
        k_cache: &mut CudaBuffer,
        v_cache: &mut CudaBuffer,
        s: KvStoreShape,
    ) -> Result<()> {
        portable::kv_store(k, v, k_cache, v_cache, s)
    }

    /// Decodes cache rows to f32; see [`Backend::kv_load`].
    fn kv_load(
        &self,
        k_cache: &CudaBuffer,
        v_cache: &CudaBuffer,
        k: &mut CudaBuffer,
        v: &mut CudaBuffer,
        s: KvStoreShape,
    ) -> Result<()> {
        portable::kv_load(k_cache, v_cache, k, v, s)
    }

    /// `state_in`: advance from this state instead of `state` (which receives the result).
    fn conv1d_silu(
        &self,
        x: &CudaBuffer,
        weight: &CudaBuffer,
        state_in: Option<&CudaBuffer>,
        state: &mut CudaBuffer,
        out: &mut CudaBuffer,
        s: ConvShape,
    ) -> Result<()> {
        portable::conv1d_silu(x, weight, state_in, state, out, s)
    }

    fn gated_delta_net(
        &self,
        qkv: &CudaBuffer,
        ab: &CudaBuffer,
        a_log: &CudaBuffer,
        dt_bias: &CudaBuffer,
        state_in: Option<&CudaBuffer>,
        state: &mut CudaBuffer,
        out: &mut CudaBuffer,
        s: GdnShape,
    ) -> Result<()> {
        portable::gated_delta_net(qkv, ab, a_log, dt_bias, state_in, state, out, s)
    }

    /// [`Backend::gated_delta_net_out`]: outputs from `state_in`, no state written.
    #[allow(clippy::too_many_arguments)]
    fn gated_delta_net_out(
        &self,
        qkv: &CudaBuffer,
        ab: &CudaBuffer,
        a_log: &CudaBuffer,
        dt_bias: &CudaBuffer,
        state_in: &CudaBuffer,
        out: &mut CudaBuffer,
        s: GdnShape,
    ) -> Result<()> {
        portable::gated_delta_net_out(qkv, ab, a_log, dt_bias, state_in, out, s)
    }

    fn gdn_replay(
        &self,
        layers: &mut [GdnReplayLayer<'_, CudaBuffer>],
        conv: ConvShape,
        gdn: GdnShape,
    ) -> Result<()> {
        portable::gdn_replay(layers, conv, gdn)
    }

    fn attention_full(
        &self,
        q: &CudaBuffer,
        k: &CudaBuffer,
        v: &CudaBuffer,
        out: &mut CudaBuffer,
        s: FullAttnShape,
    ) -> Result<()> {
        portable::attention_full(q, k, v, out, s)
    }

    /// The operand precision of `attention_full`; see [`Backend::attention_full_precision`].
    fn attention_full_precision(&self, _s: FullAttnShape) -> AttnPrecision {
        AttnPrecision::Exact
    }

    fn layer_norm(
        &self,
        x: &CudaBuffer,
        weight: &CudaBuffer,
        bias: &CudaBuffer,
        y: &mut CudaBuffer,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<()> {
        portable::layer_norm(x, weight, bias, y, rows, dim, eps)
    }

    fn add_bias(
        &self,
        x: &mut CudaBuffer,
        bias: &CudaBuffer,
        rows: usize,
        dim: usize,
    ) -> Result<()> {
        portable::add_bias(x, bias, rows, dim)
    }

    fn gelu(&self, x: &mut CudaBuffer, len: usize, kind: GeluKind) -> Result<()> {
        portable::gelu(x, len, kind)
    }

    fn rope_positions(&self, x: &mut CudaBuffer, positions: &[u32], s: RopePosShape) -> Result<()> {
        portable::rope_positions(x, positions, s)
    }

    fn top_k(
        &self,
        x: &CudaBuffer,
        rows: usize,
        row_len: usize,
        limit: usize,
        k: usize,
    ) -> Result<Vec<Vec<(u32, f32)>>> {
        portable::top_k(x, rows, row_len, limit, k)
    }

    fn concat_cols(
        &self,
        a: &CudaBuffer,
        b: &CudaBuffer,
        out: &mut CudaBuffer,
        rows: usize,
        a_cols: usize,
        b_cols: usize,
    ) -> Result<()> {
        portable::concat_cols(a, b, out, rows, a_cols, b_cols)
    }

    fn split_cols(
        &self,
        src: &CudaBuffer,
        rows: usize,
        src_cols: usize,
        outs: &mut [(&mut CudaBuffer, usize, usize)],
    ) -> Result<()> {
        portable::split_cols(src, rows, src_cols, outs)
    }

    #[allow(clippy::too_many_arguments)]
    fn swiglu(
        &self,
        src: &CudaBuffer,
        rows: usize,
        src_cols: usize,
        gate: usize,
        up: usize,
        width: usize,
        out: &mut CudaBuffer,
    ) -> Result<()> {
        portable::swiglu(src, rows, src_cols, gate, up, width, out)
    }

    fn copy2d(&self, src: &CudaBuffer, dst: &mut CudaBuffer, shape: Copy2dShape) -> Result<()> {
        portable::copy2d(src, dst, shape)
    }

    fn transpose_heads(
        &self,
        x: &CudaBuffer,
        y: &mut CudaBuffer,
        rows: usize,
        a: usize,
        b: usize,
        dim: usize,
    ) -> Result<()> {
        portable::transpose_heads(x, y, rows, a, b, dim)
    }

    fn dyn_conv(
        &self,
        x: &CudaBuffer,
        base: &CudaBuffer,
        delta: &CudaBuffer,
        out: &mut CudaBuffer,
        shape: DynConvShape,
    ) -> Result<()> {
        portable::dyn_conv(x, base, delta, out, shape)
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_block(
        &self,
        q: &CudaBuffer,
        k_ctx: &CudaBuffer,
        v_ctx: &CudaBuffer,
        k_blk: &CudaBuffer,
        v_blk: &CudaBuffer,
        out: &mut CudaBuffer,
        shape: BlockAttnShape,
    ) -> Result<()> {
        portable::attention_block(q, k_ctx, v_ctx, k_blk, v_blk, out, shape)
    }

    #[allow(clippy::too_many_arguments)]
    fn rms_norm_groups(
        &self,
        x: &CudaBuffer,
        weight: &CudaBuffer,
        y: &mut CudaBuffer,
        rows: usize,
        groups: usize,
        dim: usize,
        eps: f32,
    ) -> Result<()> {
        portable::rms_norm_groups(x, weight, y, rows, groups, dim, eps)
    }

    fn silu(&self, x: &mut CudaBuffer, len: usize) -> Result<()> {
        portable::silu(x, len)
    }

    fn hc_mix(
        &self,
        xn: &CudaBuffer,
        gate: &CudaBuffer,
        out: &mut CudaBuffer,
        rows: usize,
        streams: usize,
        dim: usize,
    ) -> Result<()> {
        portable::hc_mix(xn, gate, out, rows, streams, dim)
    }

    fn hc_combine(
        &self,
        res: &mut CudaBuffer,
        y: &CudaBuffer,
        inject: &CudaBuffer,
        shape: HcCombineShape,
    ) -> Result<()> {
        portable::hc_combine(res, y, inject, shape)
    }

    #[allow(clippy::too_many_arguments)]
    fn ple_gate(
        &self,
        key_n: &CudaBuffer,
        q_n: &CudaBuffer,
        value: &CudaBuffer,
        out: &mut CudaBuffer,
        rows: usize,
        streams: usize,
        dim: usize,
    ) -> Result<()> {
        portable::ple_gate(key_n, q_n, value, out, rows, streams, dim)
    }

    fn conv1d_dilated_silu(
        &self,
        x: &CudaBuffer,
        weight: &CudaBuffer,
        state: &mut CudaBuffer,
        out: &mut CudaBuffer,
        shape: DilatedConvShape,
    ) -> Result<()> {
        portable::conv1d_dilated_silu(x, weight, state, out, shape)
    }

    fn sigmoid_scale_rows(
        &self,
        x: &mut CudaBuffer,
        gate: &CudaBuffer,
        shape: RowGateShape,
    ) -> Result<()> {
        portable::sigmoid_scale_rows(x, gate, shape)
    }

    fn moe_route(
        &self,
        logits: &CudaBuffer,
        ids: &mut CudaBuffer,
        weights: &mut CudaBuffer,
        counts: Option<&mut CudaBuffer>,
        shape: MoeRouteShape,
    ) -> Result<()> {
        portable::moe_route(logits, ids, weights, counts, shape)
    }

    /// See [`Backend::moe_scratch_len`]; an override must cover its own kernels' needs.
    fn moe_scratch_len(&self, bank: &ExpertBank<'_, CudaBuffer>, rows: usize, k: usize) -> usize {
        portable::moe_scratch_len(bank, rows, k)
    }

    #[allow(clippy::too_many_arguments)]
    fn moe_experts(
        &self,
        x: &CudaBuffer,
        ids: &CudaBuffer,
        weights: &CudaBuffer,
        bank: &ExpertBank<'_, CudaBuffer>,
        y: &mut CudaBuffer,
        scratch: &mut CudaBuffer,
        rows: usize,
        k: usize,
    ) -> Result<()> {
        portable::moe_experts(x, ids, weights, bank, y, scratch, rows, k)
    }

    /// See [`Backend::moe_experts_act`].
    fn moe_experts_act(&self, _bank: &ExpertBank<'_, CudaBuffer>, _rows: usize) -> ActQuant {
        ActQuant::Exact
    }

    fn qsa_pool(
        &self,
        raw: &CudaBuffer,
        weight: &CudaBuffer,
        pooled: &mut CudaBuffer,
        shape: QsaPoolShape,
    ) -> Result<()> {
        portable::qsa_pool(raw, weight, pooled, shape)
    }

    fn qsa_select(
        &self,
        q: &CudaBuffer,
        pooled: &CudaBuffer,
        sel: &mut CudaBuffer,
        scratch: &mut CudaBuffer,
        shape: QsaSelectShape,
    ) -> Result<()> {
        portable::qsa_select(q, pooled, sel, scratch, shape)
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_sparse(
        &self,
        q: &CudaBuffer,
        k: &CudaBuffer,
        v: &CudaBuffer,
        sel: &CudaBuffer,
        sel_len: usize,
        out: &mut CudaBuffer,
        shape: AttnShape,
    ) -> Result<()> {
        portable::attention_sparse(q, k, v, sel, sel_len, out, shape)
    }
}

/// Every op on the portable kernels; the CUDA kernel set of the generic variant on main.
pub struct PortableKernels;

impl CudaKernels for PortableKernels {}

pub struct CudaBackend {
    device: DeviceInfo,
    profile: Option<&'static DeviceProfile>,
    variant: &'static VariantInfo,
    kernels: Arc<dyn CudaKernels>,
    timing: Option<timing::Timing>,
    /// CUDA graph slots by key, and keys seen once (run eagerly as warm-up); `None` when graphs
    /// are off (op timing, or `VELOX_NO_GRAPHS`).
    graphs: Option<Mutex<GraphKeys>>,
}

#[derive(Default)]
struct GraphKeys {
    /// Slot of each segment key (a step's key, then `key | segment << 56`).
    slots: HashMap<u64, i32>,
    seen: HashSet<u64>,
    /// Static keys recorded at least once: later runs replay.
    recorded: HashSet<u64>,
    /// Each step's segment slots and when it last ran: a budget releases whole steps, so a
    /// replay never meets a missing segment.
    steps: HashMap<u64, GraphStep>,
    /// Released slots, reused before new ones.
    free: Vec<i32>,
    next: i32,
    clock: u64,
    /// Device bytes recorded graphs may hold (0: unbounded).
    budget: usize,
    released: usize,
}

#[derive(Default)]
struct GraphStep {
    segs: Vec<i32>,
    used: u64,
}

impl GraphKeys {
    /// The slot of segment `seg_key` of step `step`, taking a free one (releasing the least
    /// recently run other step when all are taken). `None`: no slot.
    fn slot_for(&mut self, seg_key: u64, step: u64) -> Option<i32> {
        if let Some(&s) = self.slots.get(&seg_key) {
            return Some(s);
        }
        let s = match self.free.pop() {
            Some(s) => s,
            None if self.next < GRAPH_SLOTS => {
                self.next += 1;
                self.next - 1
            }
            None => {
                let victim = self.lru(step)?;
                self.release(victim).ok()?;
                self.free.pop()?
            }
        };
        self.slots.insert(seg_key, s);
        self.steps.entry(step).or_default().segs.push(s);
        Some(s)
    }

    fn touch(&mut self, step: u64) {
        self.clock += 1;
        let clock = self.clock;
        if let Some(st) = self.steps.get_mut(&step) {
            st.used = clock;
        }
    }

    /// The least recently run step other than `keep`.
    fn lru(&self, keep: u64) -> Option<u64> {
        self.steps
            .iter()
            .filter(|(&k, _)| k != keep)
            .min_by_key(|(_, st)| st.used)
            .map(|(&k, _)| k)
    }

    /// Releases every segment graph of `step`; its next run records again.
    fn release(&mut self, step: u64) -> Result<()> {
        let Some(st) = self.steps.remove(&step) else {
            return Ok(());
        };
        self.slots.retain(|_, s| !st.segs.contains(s));
        self.recorded.remove(&step);
        for s in st.segs {
            // SAFETY: `s` is a slot this map handed out (in the shim's range).
            status(unsafe { ffi::velox_cuda_graph_release(s) }, "graph release")?;
            self.free.push(s);
        }
        self.released += 1;
        Ok(())
    }

    /// Releases least recently run steps other than `keep` while graphs hold over the budget.
    fn enforce(&mut self, keep: u64) -> Result<()> {
        if self.budget == 0 {
            return Ok(());
        }
        // SAFETY: plain query.
        while unsafe { ffi::velox_cuda_graph_bytes() } > self.budget {
            let Some(victim) = self.lru(keep) else { break };
            self.release(victim)?;
        }
        Ok(())
    }
}

/// Graph slots the shim keeps (`kGraphSlots`).
const GRAPH_SLOTS: i32 = 4096;

/// What this thread is doing inside [`Backend::graph`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphMode {
    None,
    /// Recording: ops launch into the capture and stage per-call values in the step's region.
    Capture,
    /// Re-running a static step's host code: ops only stage their per-call values; the recorded
    /// graph is launched afterwards.
    Replay,
}

/// The KV bucket (in keys) a static graph step's key carries, while one is recorded or replayed.
/// Bytes of one virtual-memory chunk on the current device (0: none).
fn vm_chunk_bytes() -> usize {
    static BYTES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BYTES.get_or_init(|| {
        if std::env::var_os("VELOX_NO_VMM").is_some() {
            return 0;
        }
        let mut b = 0usize;
        // SAFETY: out-pointer to a live local.
        match unsafe { ffi::velox_vmm_granularity(&mut b) } {
            0 => b,
            _ => 0,
        }
    })
}

pub(crate) fn static_kv_bucket() -> Option<usize> {
    if graph_mode() == GraphMode::None {
        return None;
    }
    let key = GRAPH_SEG.with(std::cell::Cell::get).0;
    (key & GRAPH_STATIC != 0).then(|| ((key >> 24) & 0xffff) as usize * velox_core::GRAPH_KV_BUCKET)
}

thread_local! {
    static GRAPH_MODE: std::cell::Cell<GraphMode> = const { std::cell::Cell::new(GraphMode::None) };
    /// The step being recorded or replayed: its key, current segment and that segment's slot.
    static GRAPH_SEG: std::cell::Cell<(u64, u64, i32)> = const { std::cell::Cell::new((0, 0, -1)) };
}

pub fn graph_mode() -> GraphMode {
    GRAPH_MODE.with(std::cell::Cell::get)
}

fn set_graph_mode(m: GraphMode) {
    GRAPH_MODE.with(|c| c.set(m));
}

/// Ops whose launches a replay skips return this.
trait Skipped {
    fn skipped() -> Self;
}

impl Skipped for Result<()> {
    fn skipped() -> Self {
        Ok(())
    }
}

impl<T> Skipped for Result<Vec<T>> {
    fn skipped() -> Self {
        Err(Error::Backend(
            "host readback inside a replayed graph step".into(),
        ))
    }
}

impl CudaBackend {
    /// Makes `device` current and binds the variant's kernels to it.
    pub fn new(
        device: DeviceInfo,
        profile: Option<&'static DeviceProfile>,
        variant: &'static VariantInfo,
        kernels: Arc<dyn CudaKernels>,
    ) -> Result<Self> {
        // SAFETY: plain value argument.
        status(
            unsafe { ffi::velox_cuda_set_device(device.index) },
            "cudaSetDevice",
        )?;
        let timing = timing::Timing::from_env();
        Ok(CudaBackend {
            device,
            profile,
            variant,
            kernels,
            graphs: (timing.is_none() && std::env::var_os("VELOX_NO_GRAPHS").is_none())
                .then(|| Mutex::new(GraphKeys::default())),
            timing,
        })
    }

    pub fn device(&self) -> &DeviceInfo {
        &self.device
    }

    /// Runs an op; a replay skips it (its launches are in the recorded graph).
    fn timed<T: Skipped>(&self, op: &str, f: impl FnOnce() -> T) -> T {
        if graph_mode() == GraphMode::Replay {
            return T::skipped();
        }
        self.staged(op, f)
    }

    /// Runs an op that stages per-call values: a replay runs it too (it returns after staging).
    fn staged<T>(&self, op: &str, f: impl FnOnce() -> T) -> T {
        match &self.timing {
            Some(t) => t.time(op, f),
            None => f(),
        }
    }
}

/// Queues `data` for upload through a pinned ring and returns its device address. The address
/// stays valid for kernels queued right after this call (the ring is recycled only after a stream
/// synchronize), so it suits per-call inputs such as token ids, not persistent data.
pub(crate) fn stage(data: &[u8]) -> Result<*const c_void> {
    let mut d: *mut c_void = std::ptr::null_mut();
    // SAFETY: the shim copies `data` into pinned memory before returning.
    status(
        unsafe { ffi::velox_cuda_stage(data.as_ptr().cast(), data.len(), &mut d) },
        "stage upload",
    )?;
    Ok(d)
}

/// Uploads raw bytes into a fresh buffer of `ceil(len / 4)` f32 slots (tail zeroed).
pub(crate) fn upload_raw(data: &[u8]) -> Result<CudaBuffer> {
    let buf = CudaBuffer::new(data.len().div_ceil(4))?;
    if !data.is_empty() {
        // SAFETY: buf holds at least data.len() bytes.
        status(
            unsafe { ffi::velox_cuda_htod(buf.ptr.cast(), data.as_ptr().cast(), data.len()) },
            "upload",
        )?;
    }
    Ok(buf)
}

/// An int8 twin must hold the `slots` its layout takes (0: no such twin here).
fn check_q8_twin(what: &str, twin: &CudaBuffer, slots: usize) -> Result<()> {
    if slots == 0 || twin.len < slots {
        return Err(Error::Shape(format!(
            "{what}: int8 twin of {} slots, layout needs {slots}",
            twin.len
        )));
    }
    Ok(())
}

impl Backend for CudaBackend {
    fn top_k(
        &self,
        x: &CudaBuffer,
        rows: usize,
        row_len: usize,
        limit: usize,
        k: usize,
    ) -> Result<Vec<Vec<(u32, f32)>>> {
        self.timed("top_k", || self.kernels.top_k(x, rows, row_len, limit, k))
    }

    fn concat_cols(
        &self,
        a: &CudaBuffer,
        b: &CudaBuffer,
        out: &mut CudaBuffer,
        rows: usize,
        a_cols: usize,
        b_cols: usize,
    ) -> Result<()> {
        check::concat_cols(a.len, b.len, out.len, rows, a_cols, b_cols)?;
        self.timed("concat_cols", || {
            self.kernels.concat_cols(a, b, out, rows, a_cols, b_cols)
        })
    }

    fn split_cols(
        &self,
        src: &CudaBuffer,
        rows: usize,
        src_cols: usize,
        outs: &mut [(&mut CudaBuffer, usize, usize)],
    ) -> Result<()> {
        self.timed("split_cols", || {
            self.kernels.split_cols(src, rows, src_cols, outs)
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn swiglu(
        &self,
        src: &CudaBuffer,
        rows: usize,
        src_cols: usize,
        gate: usize,
        up: usize,
        width: usize,
        out: &mut CudaBuffer,
    ) -> Result<()> {
        check::swiglu(src.len, rows, src_cols, gate, up, width, out.len)?;
        self.timed("swiglu", || {
            self.kernels
                .swiglu(src, rows, src_cols, gate, up, width, out)
        })
    }

    fn linears(
        &self,
        x: &CudaBuffer,
        xb: Option<&CudaBuffer>,
        xq: Option<(&CudaBuffer, u32)>,
        dsts: &mut [LinearDst<'_, '_, '_, CudaBuffer>],
        scratch: &mut CudaBuffer,
        m: usize,
    ) -> Result<()> {
        if !self.kernels.serves_linears(dsts, m) {
            return velox_core::backend::linears_composed(self, x, dsts, scratch, m);
        }
        for d in dsts.iter() {
            for p in d.parts {
                // A column range is checked as a view of `ld` rows over the destination.
                check::linear(x.len, &p.w, p.w.buf.len, d.y.len, m)?;
                if p.col + p.w.rows > d.ld || d.y.len < m * d.ld {
                    return Err(Error::Shape(format!(
                        "linears: columns {}..{} of {} x {} in {} values",
                        p.col,
                        p.col + p.w.rows,
                        m,
                        d.ld,
                        d.y.len
                    )));
                }
            }
        }
        let op = format!("linears m={}", if m <= 16 { "<=16" } else { ">16" });
        let k = dsts
            .iter()
            .flat_map(|d| d.parts.iter())
            .map(|p| p.w.k)
            .max()
            .unwrap_or(0);
        if let Some(xb) = xb {
            check::elementwise("linears bf16 twin", &[xb.len * 2], m * k)?;
        }
        if let Some((xq, layout)) = xq {
            check_q8_twin("linears", xq, self.kernels.int8_twin_slots(m, k, layout))?;
        }
        self.timed(&op, || self.kernels.linears(x, xb, xq, dsts, m))
    }

    fn bf16_twins(&self) -> bool {
        true
    }

    #[allow(clippy::too_many_arguments)]
    fn add_rms_norm_twin(
        &self,
        acc: &mut CudaBuffer,
        x: &CudaBuffer,
        weight: &CudaBuffer,
        y: &mut CudaBuffer,
        yb: &mut CudaBuffer,
        rows: usize,
        dim: usize,
        eps: f32,
        kind: NormKind,
    ) -> Result<()> {
        check::rms_norm(acc.len, weight.len, y.len, rows, dim)?;
        check::elementwise("add_rms_norm", &[x.len], rows * dim)?;
        check::elementwise("add_rms_norm twin", &[yb.len * 2], rows * dim)?;
        self.timed("add_rms_norm", || {
            self.kernels
                .add_rms_norm_twin(acc, x, weight, y, yb, rows, dim, eps, kind)
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn rms_norm_gated_twin(
        &self,
        x: &CudaBuffer,
        weight: &CudaBuffer,
        gate: &mut CudaBuffer,
        gb: &mut CudaBuffer,
        rows: usize,
        dim: usize,
        eps: f32,
        kind: NormKind,
    ) -> Result<()> {
        check::rms_norm(x.len, weight.len, gate.len, rows, dim)?;
        check::elementwise("rms_norm_gated twin", &[gb.len * 2], rows * dim)?;
        self.timed("rms_norm_gated", || {
            self.kernels
                .rms_norm_gated_twin(x, weight, gate, gb, rows, dim, eps, kind)
        })
    }

    fn sigmoid_mul_twin(
        &self,
        x: &mut CudaBuffer,
        gate: &CudaBuffer,
        xb: &mut CudaBuffer,
        len: usize,
    ) -> Result<()> {
        check::elementwise("sigmoid_mul", &[x.len, gate.len], len)?;
        check::elementwise("sigmoid_mul twin", &[xb.len * 2], len)?;
        self.timed("sigmoid_mul", || {
            self.kernels.sigmoid_mul_twin(x, gate, xb, len)
        })
    }

    fn takes_bf16(&self, w: &WeightView<'_, CudaBuffer>, m: usize) -> bool {
        self.kernels.takes_bf16(w, m)
    }

    fn linear_bf16(
        &self,
        xb: &CudaBuffer,
        w: &WeightView<'_, CudaBuffer>,
        y: &mut CudaBuffer,
        m: usize,
    ) -> Result<()> {
        if !self.kernels.takes_bf16(w, m) {
            return Err(Error::Unsupported("linear_bf16 view".into()));
        }
        check::linear(xb.len * 2, w, w.buf.len, y.len, m)?;
        let op = format!(
            "linear {} m={} bf16 in",
            w.format.name(),
            if m <= 16 { "<=16" } else { ">16" }
        );
        self.timed(&op, || self.kernels.linear_bf16(xb, w, y, m))
    }

    fn swiglu_twin(
        &self,
        gate: &WeightView<'_, CudaBuffer>,
        up: &WeightView<'_, CudaBuffer>,
        m: usize,
    ) -> bool {
        self.kernels.swiglu_twin(gate, up, m)
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_swiglu_twin(
        &self,
        xb: &CudaBuffer,
        gate: &WeightView<'_, CudaBuffer>,
        up: &WeightView<'_, CudaBuffer>,
        out: &mut CudaBuffer,
        outb: &mut CudaBuffer,
        m: usize,
    ) -> Result<()> {
        if !self.kernels.swiglu_twin(gate, up, m) {
            return Err(Error::Unsupported("linear_swiglu_twin views".into()));
        }
        check::linear(xb.len * 2, gate, gate.buf.len, out.len, m)?;
        check::linear(xb.len * 2, up, up.buf.len, out.len, m)?;
        check::elementwise("linear_swiglu twin", &[outb.len * 2], m * gate.rows)?;
        let op = format!(
            "linear_swiglu {} m={m} [{}x{}] twin",
            gate.format.name(),
            gate.rows,
            gate.k
        );
        self.timed(&op, || {
            self.kernels.linear_swiglu_twin(xb, gate, up, out, outb, m)
        })
    }

    fn int8_twin_layout(&self, w: &WeightView<'_, CudaBuffer>, m: usize) -> Option<u32> {
        self.kernels.int8_twin_layout(w, m)
    }

    fn int8_twin_slots(&self, m: usize, k: usize, layout: u32) -> usize {
        self.kernels.int8_twin_slots(m, k, layout)
    }

    fn add_rms_norm_q8(
        &self,
        acc: &mut CudaBuffer,
        x: &CudaBuffer,
        weight: &CudaBuffer,
        y: &mut CudaBuffer,
        yq: &mut CudaBuffer,
        layout: u32,
        rows: usize,
        dim: usize,
        eps: f32,
        kind: NormKind,
    ) -> Result<()> {
        check::rms_norm(acc.len, weight.len, y.len, rows, dim)?;
        check::elementwise("add_rms_norm", &[x.len], rows * dim)?;
        check_q8_twin(
            "add_rms_norm",
            yq,
            self.kernels.int8_twin_slots(rows, dim, layout),
        )?;
        self.timed("add_rms_norm", || {
            self.kernels
                .add_rms_norm_q8(acc, x, weight, y, yq, layout, rows, dim, eps, kind)
        })
    }

    fn linear_q8(
        &self,
        xq: &CudaBuffer,
        layout: u32,
        w: &WeightView<'_, CudaBuffer>,
        y: &mut CudaBuffer,
        m: usize,
    ) -> Result<()> {
        if self.kernels.int8_twin_layout(w, m) != Some(layout) {
            return Err(Error::Unsupported("linear_q8 view".into()));
        }
        check_q8_twin("linear", xq, self.kernels.int8_twin_slots(m, w.k, layout))?;
        check::linear(m * w.k, w, w.buf.len, y.len, m)?;
        let op = format!("linear {} m=>16 q8 in", w.format.name());
        self.timed(&op, || self.kernels.linear_q8(xq, layout, w, y, m))
    }

    fn linear_swiglu_q8(
        &self,
        xq: &CudaBuffer,
        layout: u32,
        gate: &WeightView<'_, CudaBuffer>,
        up: &WeightView<'_, CudaBuffer>,
        out: &mut CudaBuffer,
        m: usize,
    ) -> Result<()> {
        if self.kernels.int8_twin_layout(gate, m) != Some(layout) {
            return Err(Error::Unsupported("linear_swiglu_q8 views".into()));
        }
        check_q8_twin(
            "linear_swiglu",
            xq,
            self.kernels.int8_twin_slots(m, gate.k, layout),
        )?;
        check::linear(m * gate.k, gate, gate.buf.len, out.len, m)?;
        check::linear(m * up.k, up, up.buf.len, out.len, m)?;
        let op = format!(
            "linear_swiglu {} m={m} [{}x{}] q8 twin",
            gate.format.name(),
            gate.rows,
            gate.k
        );
        self.timed(&op, || {
            self.kernels.linear_swiglu_q8(xq, layout, gate, up, out, m)
        })
    }

    fn linear_swiglu(
        &self,
        x: &CudaBuffer,
        gate: &WeightView<'_, CudaBuffer>,
        up: &WeightView<'_, CudaBuffer>,
        out: &mut CudaBuffer,
        m: usize,
        scratch: &mut CudaBuffer,
    ) -> Result<()> {
        check::linear(x.len, gate, gate.buf.len, out.len, m)?;
        check::linear(x.len, up, up.buf.len, out.len, m)?;
        if self.kernels.linear_swiglu_act(gate, up, m).is_some() {
            let op = format!(
                "linear_swiglu {} m={m} [{}x{}]",
                gate.format.name(),
                gate.rows,
                gate.k
            );
            return self.timed(&op, || self.kernels.linear_swiglu(x, gate, up, out, m));
        }
        velox_core::backend::linear_swiglu_composed(self, x, gate, up, out, m, scratch)
    }

    fn fuses_swiglu(
        &self,
        gate: &WeightView<'_, CudaBuffer>,
        up: &WeightView<'_, CudaBuffer>,
        m: usize,
    ) -> bool {
        self.kernels.linear_swiglu_act(gate, up, m).is_some()
    }

    fn linear_swiglu_act(
        &self,
        gate: &WeightView<'_, CudaBuffer>,
        up: &WeightView<'_, CudaBuffer>,
        m: usize,
    ) -> ActQuant {
        self.kernels
            .linear_swiglu_act(gate, up, m)
            .unwrap_or_else(|| match velox_core::backend::swiglu_pair(gate, up) {
                Some((both, _, _)) => self.linear_act(&both, m),
                None => self.linear_act(gate, m),
            })
    }

    fn graph(&self, key: u64, f: &mut dyn FnMut() -> Result<()>) -> Result<()> {
        let Some(graphs) = &self.graphs else {
            return f();
        };
        if graph_mode() != GraphMode::None {
            // Nested steps run inside the enclosing recording.
            return f();
        }
        // The first run of a key is eager: lazy allocations and caches settle outside capture.
        let (slot, replay) = {
            let mut g = graphs
                .lock()
                .map_err(|_| Error::Backend("graph keys poisoned".into()))?;
            if g.seen.insert(key) {
                (None, false)
            } else {
                let slot = g.slot_for(key, key);
                g.touch(key);
                (slot, key & GRAPH_STATIC != 0 && g.recorded.contains(&key))
            }
        };
        let Some(slot) = slot else {
            return f();
        };
        let mode = if replay {
            GraphMode::Replay
        } else {
            GraphMode::Capture
        };
        // SAFETY: slot is in the shim's range (checked above).
        let begun = unsafe {
            if replay {
                ffi::velox_cuda_graph_replay_begin(slot)
            } else {
                ffi::velox_cuda_graph_begin(slot)
            }
        };
        status(begun, "graph begin")?;
        set_graph_mode(mode);
        GRAPH_SEG.with(|c| c.set((key, 0, slot)));
        let r = f();
        set_graph_mode(GraphMode::None);
        if let Err(e) = r {
            // SAFETY: ends the capture or replay begun above.
            unsafe { ffi::velox_cuda_graph_abort() };
            if replay {
                // Forget the recording: the next run records again.
                if let Ok(mut g) = graphs.lock() {
                    g.recorded.remove(&key);
                }
            }
            return Err(e);
        }
        // The last segment (see `graph_break`) is the one still open.
        let slot = GRAPH_SEG.with(std::cell::Cell::get).2;
        // SAFETY: slot is in the shim's range and its graph was begun above.
        let launched = unsafe {
            if replay {
                ffi::velox_cuda_graph_replay_launch(slot)
            } else {
                ffi::velox_cuda_graph_end_launch(slot)
            }
        };
        status(launched, "graph launch")?;
        if !replay {
            let mut g = graphs
                .lock()
                .map_err(|_| Error::Backend("graph keys poisoned".into()))?;
            if key & GRAPH_STATIC != 0 {
                g.recorded.insert(key);
            }
            g.enforce(key)?;
        }
        Ok(())
    }

    fn graph_break(&self) -> Result<()> {
        let (Some(graphs), mode) = (&self.graphs, graph_mode()) else {
            return Ok(());
        };
        if mode == GraphMode::None {
            return Ok(());
        }
        let (key, seg, slot) = GRAPH_SEG.with(std::cell::Cell::get);
        if seg + 1 >= 128 {
            return Ok(()); // the step stays one graph from here
        }
        let next_key = key | (seg + 1) << 56;
        let next = graphs
            .lock()
            .map_err(|_| Error::Backend("graph keys poisoned".into()))?
            .slot_for(next_key, key);
        let Some(next) = next else {
            return Ok(()); // out of slots: the step stays one graph
        };
        // SAFETY: both slots are in the shim's range; the current one was begun by `graph`.
        let r = unsafe {
            if mode == GraphMode::Replay {
                status(ffi::velox_cuda_graph_replay_launch(slot), "graph launch")
                    .and_then(|()| status(ffi::velox_cuda_graph_replay_begin(next), "graph begin"))
            } else {
                status(ffi::velox_cuda_graph_end_launch(slot), "graph launch")
                    .and_then(|()| status(ffi::velox_cuda_graph_begin(next), "graph begin"))
            }
        };
        GRAPH_SEG.with(|c| c.set((key, seg + 1, next)));
        r
    }

    fn set_graph_budget(&self, bytes: usize) {
        if let Some(Ok(mut g)) = self.graphs.as_ref().map(Mutex::lock) {
            g.budget = bytes;
        }
    }

    fn graph_stats(&self) -> (usize, usize, usize) {
        match self.graphs.as_ref().map(Mutex::lock) {
            // SAFETY: plain query.
            Some(Ok(g)) => (
                g.steps.len(),
                unsafe { ffi::velox_cuda_graph_bytes() },
                g.released,
            ),
            _ => (0, 0, 0),
        }
    }

    fn mem_info(&self) -> Option<(usize, usize)> {
        mem_info().ok().map(|(f, t)| (f as usize, t as usize))
    }

    fn device_allocs(&self) -> Option<(usize, usize, usize)> {
        use std::sync::atomic::Ordering::Relaxed;
        Some((
            BUFFER_BYTES.load(Relaxed),
            BUFFER_PEAK.load(Relaxed),
            CHUNK_BYTES.load(Relaxed),
        ))
    }

    fn device_in_use(&self) -> Option<usize> {
        let (mut used, mut total) = (0usize, 0usize);
        // SAFETY: out-pointers to live locals.
        match unsafe { ffi::velox_nvml_used(&mut used, &mut total) } {
            0 => Some(used),
            _ => self.mem_info().map(|(free, total)| total - free),
        }
    }

    fn copy2d(&self, src: &CudaBuffer, dst: &mut CudaBuffer, shape: Copy2dShape) -> Result<()> {
        check::copy2d(src.len, dst.len, &shape)?;
        self.staged("copy2d", || self.kernels.copy2d(src, dst, shape))
    }

    fn transpose_heads(
        &self,
        x: &CudaBuffer,
        y: &mut CudaBuffer,
        rows: usize,
        a: usize,
        b: usize,
        dim: usize,
    ) -> Result<()> {
        check::transpose_heads(x.len, y.len, rows, a, b, dim)?;
        self.staged("transpose_heads", || {
            self.kernels.transpose_heads(x, y, rows, a, b, dim)
        })
    }

    fn dyn_conv(
        &self,
        x: &CudaBuffer,
        base: &CudaBuffer,
        delta: &CudaBuffer,
        out: &mut CudaBuffer,
        shape: DynConvShape,
    ) -> Result<()> {
        check::dyn_conv(x.len, base.len, delta.len, out.len, &shape)?;
        self.timed("dyn_conv", || {
            self.kernels.dyn_conv(x, base, delta, out, shape)
        })
    }

    fn attention_block(
        &self,
        q: &CudaBuffer,
        k_ctx: &CudaBuffer,
        v_ctx: &CudaBuffer,
        k_blk: &CudaBuffer,
        v_blk: &CudaBuffer,
        out: &mut CudaBuffer,
        shape: BlockAttnShape,
    ) -> Result<()> {
        check::attention_block(
            q.len, k_ctx.len, v_ctx.len, k_blk.len, v_blk.len, out.len, &shape,
        )?;
        self.staged("attention block", || {
            self.kernels
                .attention_block(q, k_ctx, v_ctx, k_blk, v_blk, out, shape)
        })
    }

    fn write_at(&self, dst: &mut CudaBuffer, offset: usize, data: &[f32]) -> Result<()> {
        check::copy(data.len(), 0, dst.len, offset, data.len())?;
        if data.is_empty() {
            return Ok(());
        }
        // SAFETY: the destination range was validated; the driver stages pageable sources before
        // returning, so `data` may be reused right away.
        status(
            unsafe {
                ffi::velox_cuda_htod_async(
                    dst.ptr.add(offset).cast(),
                    data.as_ptr().cast(),
                    data.len() * 4,
                )
            },
            "write_at",
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn rms_norm_groups(
        &self,
        x: &CudaBuffer,
        weight: &CudaBuffer,
        y: &mut CudaBuffer,
        rows: usize,
        groups: usize,
        dim: usize,
        eps: f32,
    ) -> Result<()> {
        check::rms_norm_groups(x.len, weight.len, y.len, rows, groups, dim)?;
        self.timed("rms_norm_groups", || {
            self.kernels
                .rms_norm_groups(x, weight, y, rows, groups, dim, eps)
        })
    }

    fn silu(&self, x: &mut CudaBuffer, len: usize) -> Result<()> {
        check::elementwise("silu", &[x.len], len)?;
        self.timed("silu", || self.kernels.silu(x, len))
    }

    fn hc_mix(
        &self,
        xn: &CudaBuffer,
        gate: &CudaBuffer,
        out: &mut CudaBuffer,
        rows: usize,
        streams: usize,
        dim: usize,
    ) -> Result<()> {
        check::hc_mix(xn.len, gate.len, out.len, rows, streams, dim)?;
        self.timed("hc_mix", || {
            self.kernels.hc_mix(xn, gate, out, rows, streams, dim)
        })
    }

    fn hc_combine(
        &self,
        res: &mut CudaBuffer,
        y: &CudaBuffer,
        inject: &CudaBuffer,
        shape: HcCombineShape,
    ) -> Result<()> {
        check::hc_combine(res.len, y.len, inject.len, &shape)?;
        self.timed("hc_combine", || {
            self.kernels.hc_combine(res, y, inject, shape)
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn ple_gate(
        &self,
        key_n: &CudaBuffer,
        q_n: &CudaBuffer,
        value: &CudaBuffer,
        out: &mut CudaBuffer,
        rows: usize,
        streams: usize,
        dim: usize,
    ) -> Result<()> {
        check::ple_gate(key_n.len, q_n.len, value.len, out.len, rows, streams, dim)?;
        self.timed("ple_gate", || {
            self.kernels
                .ple_gate(key_n, q_n, value, out, rows, streams, dim)
        })
    }

    fn conv1d_dilated_silu(
        &self,
        x: &CudaBuffer,
        weight: &CudaBuffer,
        state: &mut CudaBuffer,
        out: &mut CudaBuffer,
        shape: DilatedConvShape,
    ) -> Result<()> {
        check::conv1d_dilated(x.len, weight.len, state.len, out.len, &shape)?;
        self.timed("conv1d_dilated_silu", || {
            self.kernels
                .conv1d_dilated_silu(x, weight, state, out, shape)
        })
    }

    fn sigmoid_scale_rows(
        &self,
        x: &mut CudaBuffer,
        gate: &CudaBuffer,
        shape: RowGateShape,
    ) -> Result<()> {
        check::row_gate(x.len, gate.len, &shape)?;
        self.timed("sigmoid_scale_rows", || {
            self.kernels.sigmoid_scale_rows(x, gate, shape)
        })
    }

    fn moe_route(
        &self,
        logits: &CudaBuffer,
        ids: &mut CudaBuffer,
        weights: &mut CudaBuffer,
        counts: Option<&mut CudaBuffer>,
        shape: MoeRouteShape,
    ) -> Result<()> {
        check::moe_route(
            logits.len,
            ids.len,
            weights.len,
            counts.as_ref().map(|c| c.len),
            &shape,
        )?;
        self.timed("moe_route", || {
            self.kernels.moe_route(logits, ids, weights, counts, shape)
        })
    }

    fn moe_scratch_len(&self, bank: &ExpertBank<'_, CudaBuffer>, rows: usize, k: usize) -> usize {
        self.kernels.moe_scratch_len(bank, rows, k)
    }

    #[allow(clippy::too_many_arguments)]
    fn moe_experts(
        &self,
        x: &CudaBuffer,
        ids: &CudaBuffer,
        weights: &CudaBuffer,
        bank: &ExpertBank<'_, CudaBuffer>,
        y: &mut CudaBuffer,
        scratch: &mut CudaBuffer,
        rows: usize,
        k: usize,
    ) -> Result<()> {
        let bufs = (bank.records.len, bank.slots.len, bank.map.len);
        check::moe_experts(x.len, ids.len, weights.len, bank, bufs, y.len, rows, k)?;
        self.timed("moe_experts", || {
            self.kernels
                .moe_experts(x, ids, weights, bank, y, scratch, rows, k)
        })
    }

    fn moe_experts_act(&self, bank: &ExpertBank<'_, CudaBuffer>, rows: usize) -> ActQuant {
        self.kernels.moe_experts_act(bank, rows)
    }

    fn qsa_pool(
        &self,
        raw: &CudaBuffer,
        weight: &CudaBuffer,
        pooled: &mut CudaBuffer,
        shape: QsaPoolShape,
    ) -> Result<()> {
        check::qsa_pool(raw.len, weight.len, pooled.len, &shape)?;
        self.timed("qsa_pool", || {
            self.kernels.qsa_pool(raw, weight, pooled, shape)
        })
    }

    fn qsa_select(
        &self,
        q: &CudaBuffer,
        pooled: &CudaBuffer,
        sel: &mut CudaBuffer,
        scratch: &mut CudaBuffer,
        shape: QsaSelectShape,
    ) -> Result<()> {
        check::qsa_select(q.len, pooled.len, sel.len, scratch.len, &shape)?;
        self.timed("qsa_select", || {
            self.kernels.qsa_select(q, pooled, sel, scratch, shape)
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_sparse(
        &self,
        q: &CudaBuffer,
        k: &CudaBuffer,
        v: &CudaBuffer,
        sel: &CudaBuffer,
        sel_len: usize,
        out: &mut CudaBuffer,
        shape: AttnShape,
    ) -> Result<()> {
        check::attention_sparse(q.len, k.len, v.len, sel.len, sel_len, out.len, &shape)?;
        self.timed("attention_sparse", || {
            self.kernels
                .attention_sparse(q, k, v, sel, sel_len, out, shape)
        })
    }

    type Buffer = CudaBuffer;
    type Chunk = CudaChunk;

    fn kind(&self) -> BackendKind {
        BackendKind::Cuda
    }

    fn describe(&self) -> String {
        let profile = self
            .profile
            .map_or("unknown".to_string(), |p| p.id.to_string());
        format!(
            "cuda:{} {} ({}, profile {profile}) variant {}",
            self.device.index, self.device.name, self.device.arch, self.variant.id
        )
    }

    fn profile(&self) -> Option<&'static DeviceProfile> {
        self.profile
    }

    fn native_ops(&self) -> Vec<Op> {
        self.kernels.native_ops().to_vec()
    }

    fn alloc(&self, len: usize) -> Result<CudaBuffer> {
        CudaBuffer::new(len)
    }

    fn upload(&self, data: &[f32]) -> Result<CudaBuffer> {
        // SAFETY: an f32 slice viewed as its bytes.
        upload_raw(unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), data.len() * 4) })
    }

    fn upload_bytes(&self, data: &[u8]) -> Result<CudaBuffer> {
        upload_raw(data)
    }

    fn alloc_host(&self, len: usize) -> Result<CudaBuffer> {
        CudaBuffer::new_host(len)
    }

    fn vm_chunk_len(&self) -> usize {
        vm_chunk_bytes() / 4
    }

    fn alloc_virtual(&self, len: usize) -> Result<CudaBuffer> {
        let chunk = vm_chunk_bytes();
        if chunk == 0 {
            return Err(Error::Unsupported("virtual memory on this device".into()));
        }
        let chunks = (len.max(1) * 4).div_ceil(chunk);
        let mut va: *mut c_void = std::ptr::null_mut();
        // SAFETY: out-pointer to a live local.
        status(
            unsafe { ffi::velox_vmm_reserve(&mut va, chunks * chunk) },
            "cuMemAddressReserve",
        )?;
        Ok(CudaBuffer {
            ptr: va.cast(),
            len,
            host: false,
            vm: Some(Box::new(VmRange {
                chunk_bytes: chunk,
                mapped: vec![false; chunks],
            })),
        })
    }

    fn alloc_chunk(&self) -> Result<CudaChunk> {
        let chunk = vm_chunk_bytes();
        if chunk == 0 {
            return Err(Error::Unsupported("virtual memory on this device".into()));
        }
        let mut handle = 0u64;
        // SAFETY: out-pointer to a live local.
        status(
            unsafe { ffi::velox_vmm_create(&mut handle, chunk) },
            "cuMemCreate",
        )?;
        CHUNK_BYTES.fetch_add(chunk, std::sync::atomic::Ordering::Relaxed);
        Ok(CudaChunk { handle })
    }

    fn map_chunk(&self, buf: &mut CudaBuffer, index: usize, chunk: &CudaChunk) -> Result<()> {
        let ptr = buf.ptr;
        let vm = buf
            .vm
            .as_mut()
            .ok_or_else(|| Error::Shape("map_chunk on a buffer that is not virtual".into()))?;
        match vm.mapped.get(index) {
            None => {
                return Err(Error::Shape(format!(
                    "map_chunk: chunk {index} of {}",
                    vm.mapped.len()
                )))
            }
            Some(true) => return Err(Error::Shape(format!("chunk {index} is already mapped"))),
            Some(false) => {}
        }
        // SAFETY: the address lies inside the reserved range (checked index); the handle is live.
        let va = unsafe { ptr.cast::<u8>().add(index * vm.chunk_bytes) };
        status(
            unsafe { ffi::velox_vmm_map(va.cast(), vm.chunk_bytes, chunk.handle) },
            "cuMemMap",
        )?;
        vm.mapped[index] = true;
        Ok(())
    }

    fn unmap_chunk(&self, buf: &mut CudaBuffer, index: usize) -> Result<()> {
        let ptr = buf.ptr;
        let vm = buf
            .vm
            .as_mut()
            .ok_or_else(|| Error::Shape("unmap_chunk on a buffer that is not virtual".into()))?;
        if !vm.mapped.get(index).copied().unwrap_or(false) {
            return Err(Error::Shape(format!("chunk {index} is not mapped")));
        }
        // SAFETY: a mapped chunk inside the reserved range; the caller synchronized its users.
        let va = unsafe { ptr.cast::<u8>().add(index * vm.chunk_bytes) };
        status(
            unsafe { ffi::velox_vmm_unmap(va.cast(), vm.chunk_bytes) },
            "cuMemUnmap",
        )?;
        vm.mapped[index] = false;
        Ok(())
    }

    fn host_alloc(&self) -> Option<velox_core::backend::HostAlloc<CudaBuffer>> {
        let device = self.device.index;
        Some(std::sync::Arc::new(move |len| {
            // SAFETY: plain runtime call; the calling thread then allocates for this device.
            status(
                unsafe { ffi::velox_cuda_set_device(device) },
                "cudaSetDevice",
            )?;
            CudaBuffer::new_host(len)
        }))
    }

    fn upload_bytes_host(&self, data: &[u8]) -> Result<CudaBuffer> {
        let buf = CudaBuffer::new_host(data.len().div_ceil(4))?;
        if !data.is_empty() {
            // SAFETY: buf is host memory of at least data.len() bytes, not yet used by the device.
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), buf.ptr.cast::<u8>(), data.len())
            };
        }
        Ok(buf)
    }

    fn copy_batch(
        &self,
        dsts: &mut [&mut CudaBuffer],
        ops: &[CopySpec<'_, CudaBuffer>],
    ) -> Result<()> {
        if graph_mode() != GraphMode::None {
            // Recorded steps keep per-copy nodes (and staged offsets).
            for op in ops {
                let dst = dsts.get_mut(op.dst).ok_or_else(|| {
                    Error::Shape(format!("copy_batch: no destination {}", op.dst))
                })?;
                self.copy(op.src, op.src_offset, dst, op.dst_offset, op.len)?;
            }
            return Ok(());
        }
        let mut dp = Vec::with_capacity(ops.len());
        let mut sp = Vec::with_capacity(ops.len());
        let mut sizes = Vec::with_capacity(ops.len());
        for op in ops {
            let dst = dsts
                .get(op.dst)
                .ok_or_else(|| Error::Shape(format!("copy_batch: no destination {}", op.dst)))?;
            check::copy(op.src.len, op.src_offset, dst.len, op.dst_offset, op.len)?;
            if op.len == 0 {
                continue;
            }
            // SAFETY: offsets validated above.
            unsafe {
                dp.push(dst.ptr.add(op.dst_offset).cast::<c_void>());
                sp.push(op.src.ptr.add(op.src_offset).cast::<c_void>().cast_const());
            }
            sizes.push(op.len * 4);
        }
        self.timed("copy_batch", || {
            // SAFETY: arrays of equal length; every range validated above.
            status(
                unsafe {
                    ffi::velox_cuda_copy_batch(dp.as_ptr(), sp.as_ptr(), sizes.as_ptr(), dp.len())
                },
                "cudaMemcpyBatchAsync",
            )
        })
    }

    fn host_bytes<'a>(&self, buf: &'a CudaBuffer) -> Option<&'a [u8]> {
        // SAFETY: mapped host memory of len f32 slots, owned by buf for its lifetime.
        (buf.host && !buf.ptr.is_null())
            .then(|| unsafe { std::slice::from_raw_parts(buf.ptr.cast::<u8>(), buf.len * 4) })
    }

    fn host_bytes_mut<'a>(&self, buf: &'a mut CudaBuffer) -> Option<&'a mut [u8]> {
        // SAFETY: as above, borrowed mutably.
        (buf.host && !buf.ptr.is_null())
            .then(|| unsafe { std::slice::from_raw_parts_mut(buf.ptr.cast::<u8>(), buf.len * 4) })
    }

    fn download(&self, buf: &CudaBuffer, out: &mut [f32]) -> Result<()> {
        check::elementwise("download", &[buf.len], out.len())?;
        if !out.is_empty() {
            // SAFETY: buf holds at least out.len() elements (checked above).
            status(
                unsafe {
                    ffi::velox_cuda_dtoh(out.as_mut_ptr().cast(), buf.ptr.cast(), out.len() * 4)
                },
                "download",
            )?;
        }
        Ok(())
    }

    fn len(&self, buf: &CudaBuffer) -> usize {
        buf.len
    }

    fn synchronize(&self) -> Result<()> {
        // SAFETY: no arguments.
        status(
            unsafe { ffi::velox_cuda_synchronize() },
            "cudaDeviceSynchronize",
        )
    }

    fn bind_thread(&self) -> Result<()> {
        // SAFETY: plain value argument.
        status(
            unsafe { ffi::velox_cuda_set_device(self.device.index) },
            "cudaSetDevice",
        )
    }

    fn matmul(
        &self,
        x: &CudaBuffer,
        w: &CudaBuffer,
        y: &mut CudaBuffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        check::matmul(x.len, w.len, y.len, m, n, k)?;
        self.timed("matmul", || self.kernels.matmul(x, w, y, m, n, k))
    }

    fn linear(
        &self,
        x: &CudaBuffer,
        w: &WeightView<'_, CudaBuffer>,
        y: &mut CudaBuffer,
        m: usize,
    ) -> Result<()> {
        check::linear(x.len, w, w.buf.len, y.len, m)?;
        // Verify-sized calls carry their shape: per-matrix efficiency is what decode tuning needs.
        let op = match m {
            1 => format!("linear {} m=1", w.format.name()),
            2..=16 => format!("linear {} m<=16 [{}x{}]", w.format.name(), w.rows, w.k),
            _ => format!("linear {} m>16", w.format.name()),
        };
        self.timed(&op, || self.kernels.linear(x, w, y, m))
    }

    fn linear_act(&self, w: &WeightView<'_, CudaBuffer>, m: usize) -> ActQuant {
        self.kernels.linear_act(w, m)
    }

    fn embed(
        &self,
        ids: &[u32],
        table: &WeightView<'_, CudaBuffer>,
        out: &mut CudaBuffer,
    ) -> Result<()> {
        check::embed(ids, table, table.buf.len, out.len)?;
        self.staged("embed", || self.kernels.embed(ids, table, out))
    }

    fn embed_from(
        &self,
        ids: &CudaBuffer,
        offset: usize,
        count: usize,
        table: &WeightView<'_, CudaBuffer>,
        out: &mut CudaBuffer,
    ) -> Result<()> {
        check::embed_from(ids.len, offset, count, table, table.buf.len, out.len)?;
        self.timed("embed_from", || {
            self.kernels.embed_from(ids, offset, count, table, out)
        })
    }

    fn argmax_to(
        &self,
        x: &CudaBuffer,
        limit: usize,
        dst: &mut CudaBuffer,
        dst_offset: usize,
    ) -> Result<()> {
        self.timed("argmax_to", || {
            portable::argmax_to(x, limit, dst, dst_offset)
        })
    }

    fn fetch_u32_begin(&self, src: &CudaBuffer, offset: usize, slot: usize) -> Result<()> {
        portable::fetch_u32_begin(src, offset, slot)
    }

    fn fetch_u32(&self, _src: &CudaBuffer, _offset: usize, slot: usize) -> Result<u32> {
        portable::fetch_u32(slot)
    }

    fn rms_norm(
        &self,
        x: &CudaBuffer,
        weight: &CudaBuffer,
        y: &mut CudaBuffer,
        rows: usize,
        dim: usize,
        eps: f32,
        kind: NormKind,
    ) -> Result<()> {
        check::rms_norm(x.len, weight.len, y.len, rows, dim)?;
        self.timed("rms_norm", || {
            self.kernels.rms_norm(x, weight, y, rows, dim, eps, kind)
        })
    }

    fn rms_norm_gated(
        &self,
        x: &CudaBuffer,
        weight: &CudaBuffer,
        gate: &mut CudaBuffer,
        rows: usize,
        dim: usize,
        eps: f32,
        kind: NormKind,
    ) -> Result<()> {
        check::rms_norm(x.len, weight.len, gate.len, rows, dim)?;
        self.timed("rms_norm_gated", || {
            self.kernels
                .rms_norm_gated(x, weight, gate, rows, dim, eps, kind)
        })
    }

    fn add_rms_norm(
        &self,
        acc: &mut CudaBuffer,
        x: &CudaBuffer,
        weight: &CudaBuffer,
        y: &mut CudaBuffer,
        rows: usize,
        dim: usize,
        eps: f32,
        kind: NormKind,
    ) -> Result<()> {
        check::rms_norm(acc.len, weight.len, y.len, rows, dim)?;
        check::elementwise("add_rms_norm", &[x.len], rows * dim)?;
        self.timed("add_rms_norm", || {
            self.kernels
                .add_rms_norm(acc, x, weight, y, rows, dim, eps, kind)
        })
    }

    fn rope(&self, x: &mut CudaBuffer, s: RopeShape) -> Result<()> {
        check::rope(x.len, &s)?;
        self.staged("rope", || self.kernels.rope(x, s))
    }

    fn silu_mul(&self, gate: &mut CudaBuffer, up: &CudaBuffer, len: usize) -> Result<()> {
        check::elementwise("silu_mul", &[gate.len, up.len], len)?;
        self.timed("silu_mul", || self.kernels.silu_mul(gate, up, len))
    }

    fn sigmoid_mul(&self, x: &mut CudaBuffer, gate: &CudaBuffer, len: usize) -> Result<()> {
        check::elementwise("sigmoid_mul", &[x.len, gate.len], len)?;
        self.timed("sigmoid_mul", || self.kernels.sigmoid_mul(x, gate, len))
    }

    fn add(&self, acc: &mut CudaBuffer, x: &CudaBuffer, len: usize) -> Result<()> {
        check::elementwise("add", &[acc.len, x.len], len)?;
        self.timed("add", || self.kernels.add(acc, x, len))
    }

    fn copy(
        &self,
        src: &CudaBuffer,
        src_offset: usize,
        dst: &mut CudaBuffer,
        dst_offset: usize,
        len: usize,
    ) -> Result<()> {
        check::copy(src.len, src_offset, dst.len, dst_offset, len)?;
        self.staged("copy", || {
            self.kernels.copy(src, src_offset, dst, dst_offset, len)
        })
    }

    fn attention(
        &self,
        q: &CudaBuffer,
        k: &CudaBuffer,
        v: &CudaBuffer,
        out: &mut CudaBuffer,
        s: AttnShape,
    ) -> Result<()> {
        check::attention(q.len, k.len, v.len, out.len, &s)?;
        let op = match (s.tokens, s.kv) {
            (1, KvFormat::F32) => "attention decode".to_string(),
            (2..=16, KvFormat::F32) => "attention verify".to_string(),
            (_, KvFormat::F32) => "attention prefill".to_string(),
            (1, f) => format!("attention decode kv {f}"),
            (2..=16, f) => format!("attention verify kv {f}"),
            (_, f) => format!("attention prefill kv {f}"),
        };
        if graph_mode() == GraphMode::Replay && s.tokens >= 16 {
            // Only the portable GQA paths stage their position; tuned prefill kernels do not.
            return Err(Error::Backend(
                "attention over 16+ tokens in a replayed graph step".into(),
            ));
        }
        self.staged(&op, || self.kernels.attention(q, k, v, out, s))
    }

    fn attention_precision(&self, s: AttnShape) -> AttnPrecision {
        self.kernels.attention_precision(s)
    }

    fn kv_store(
        &self,
        k: &CudaBuffer,
        v: &CudaBuffer,
        k_cache: &mut CudaBuffer,
        v_cache: &mut CudaBuffer,
        s: KvStoreShape,
    ) -> Result<()> {
        check::kv_store(k.len, v.len, k_cache.len, v_cache.len, &s)?;
        self.staged("kv_store", || {
            self.kernels.kv_store(k, v, k_cache, v_cache, s)
        })
    }

    fn kv_load(
        &self,
        k_cache: &CudaBuffer,
        v_cache: &CudaBuffer,
        k: &mut CudaBuffer,
        v: &mut CudaBuffer,
        s: KvStoreShape,
    ) -> Result<()> {
        check::kv_load(k_cache.len, v_cache.len, k.len, v.len, &s)?;
        self.timed("kv_load", || {
            self.kernels.kv_load(k_cache, v_cache, k, v, s)
        })
    }

    fn conv1d_silu(
        &self,
        x: &CudaBuffer,
        weight: &CudaBuffer,
        state: &mut CudaBuffer,
        out: &mut CudaBuffer,
        s: ConvShape,
    ) -> Result<()> {
        check::conv1d(x.len, weight.len, state.len, out.len, &s)?;
        self.staged("conv1d_silu", || {
            self.kernels.conv1d_silu(x, weight, None, state, out, s)
        })
    }

    fn conv1d_silu_from(
        &self,
        x: &CudaBuffer,
        weight: &CudaBuffer,
        state_in: &CudaBuffer,
        state_out: &mut CudaBuffer,
        out: &mut CudaBuffer,
        s: ConvShape,
    ) -> Result<()> {
        check::conv1d(x.len, weight.len, state_in.len, out.len, &s)?;
        self.staged("conv1d_silu", || {
            self.kernels
                .conv1d_silu(x, weight, Some(state_in), state_out, out, s)
        })
    }

    fn gated_delta_net(
        &self,
        qkv: &CudaBuffer,
        ab: &CudaBuffer,
        a_log: &CudaBuffer,
        dt_bias: &CudaBuffer,
        state: &mut CudaBuffer,
        out: &mut CudaBuffer,
        s: GdnShape,
    ) -> Result<()> {
        check::gdn(
            qkv.len,
            ab.len,
            a_log.len,
            dt_bias.len,
            state.len,
            out.len,
            &s,
        )?;
        let op = match s.tokens {
            1 => "gated_delta_net decode",
            2..=16 => "gated_delta_net verify",
            _ => "gated_delta_net prefill",
        };
        // Staged: graph steps stage the state (see `portable::gated_delta_net`).
        self.staged(op, || {
            self.kernels
                .gated_delta_net(qkv, ab, a_log, dt_bias, None, state, out, s)
        })
    }

    fn gdn_replay(
        &self,
        layers: &mut [GdnReplayLayer<'_, CudaBuffer>],
        conv: ConvShape,
        gdn: GdnShape,
    ) -> Result<()> {
        for l in layers.iter() {
            check::gdn_replay_layer(
                l.qkv.len,
                l.conv_out.len,
                l.ab.len,
                l.a_log.len,
                l.dt_bias.len,
                l.conv_state.len,
                l.ssm.len,
                &conv,
                &gdn,
            )?;
        }
        self.timed("gdn_replay", || self.kernels.gdn_replay(layers, conv, gdn))
    }

    fn gated_delta_net_from(
        &self,
        qkv: &CudaBuffer,
        ab: &CudaBuffer,
        a_log: &CudaBuffer,
        dt_bias: &CudaBuffer,
        state_in: &CudaBuffer,
        state_out: &mut CudaBuffer,
        out: &mut CudaBuffer,
        s: GdnShape,
    ) -> Result<()> {
        self.staged("gated_delta_net verify", || {
            self.kernels
                .gated_delta_net(qkv, ab, a_log, dt_bias, Some(state_in), state_out, out, s)
        })
    }

    fn gated_delta_net_out(
        &self,
        qkv: &CudaBuffer,
        ab: &CudaBuffer,
        a_log: &CudaBuffer,
        dt_bias: &CudaBuffer,
        state_in: &CudaBuffer,
        _scratch: &mut CudaBuffer,
        out: &mut CudaBuffer,
        s: GdnShape,
    ) -> Result<()> {
        self.staged("gated_delta_net verify", || {
            self.kernels
                .gated_delta_net_out(qkv, ab, a_log, dt_bias, state_in, out, s)
        })
    }

    fn attention_full(
        &self,
        q: &CudaBuffer,
        k: &CudaBuffer,
        v: &CudaBuffer,
        out: &mut CudaBuffer,
        s: FullAttnShape,
    ) -> Result<()> {
        check::attention_full(q.len, k.len, v.len, out.len, &s)?;
        self.timed("attention_full", || {
            self.kernels.attention_full(q, k, v, out, s)
        })
    }

    fn attention_full_precision(&self, s: FullAttnShape) -> AttnPrecision {
        self.kernels.attention_full_precision(s)
    }

    fn layer_norm(
        &self,
        x: &CudaBuffer,
        weight: &CudaBuffer,
        bias: &CudaBuffer,
        y: &mut CudaBuffer,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<()> {
        check::layer_norm(x.len, weight.len, bias.len, y.len, rows, dim)?;
        self.timed("layer_norm", || {
            self.kernels.layer_norm(x, weight, bias, y, rows, dim, eps)
        })
    }

    fn add_bias(
        &self,
        x: &mut CudaBuffer,
        bias: &CudaBuffer,
        rows: usize,
        dim: usize,
    ) -> Result<()> {
        check::add_bias(x.len, bias.len, rows, dim)?;
        self.timed("add_bias", || self.kernels.add_bias(x, bias, rows, dim))
    }

    fn gelu(&self, x: &mut CudaBuffer, len: usize, kind: GeluKind) -> Result<()> {
        check::elementwise("gelu", &[x.len], len)?;
        self.timed("gelu", || self.kernels.gelu(x, len, kind))
    }

    fn rope_positions(&self, x: &mut CudaBuffer, positions: &[u32], s: RopePosShape) -> Result<()> {
        check::rope_positions(x.len, positions, &s)?;
        self.staged("rope_positions", || {
            self.kernels.rope_positions(x, positions, s)
        })
    }
}

pub(crate) fn dtod(dst: *mut f32, src: *const f32, len: usize) -> Result<()> {
    // SAFETY: callers pass validated ranges inside live allocations.
    status(
        unsafe { ffi::velox_cuda_dtod(dst.cast(), src.cast(), len * 4) },
        "copy",
    )
}
