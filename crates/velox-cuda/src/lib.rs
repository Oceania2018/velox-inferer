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
    Copy2dShape, DynConvShape, FullAttnShape, GdnShape, GeluKind, KvStoreShape, NormKind,
    RopePosShape, RopeShape, WeightView, GRAPH_STATIC,
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
        pub fn velox_cuda_malloc_zeroed(ptr: *mut *mut c_void, bytes: usize) -> i32;
        pub fn velox_cuda_free(ptr: *mut c_void) -> i32;
        pub fn velox_cuda_htod(dst: *mut c_void, src: *const c_void, bytes: usize) -> i32;
        pub fn velox_cuda_dtoh(dst: *mut c_void, src: *const c_void, bytes: usize) -> i32;
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
}

// SAFETY: a device pointer is a plain address; synchronisation is CUDA's stream ordering.
unsafe impl Send for CudaBuffer {}
unsafe impl Sync for CudaBuffer {}

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
        }
        Ok(CudaBuffer {
            ptr: ptr.cast(),
            len,
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
        if !self.ptr.is_null() {
            // SAFETY: ptr came from cudaMalloc and is freed exactly once. Errors at teardown
            // (e.g. after a sticky kernel fault) cannot be reported from Drop.
            unsafe { ffi::velox_cuda_free(self.ptr.cast()) };
        }
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

    fn embed(
        &self,
        ids: &[u32],
        table: &WeightView<'_, CudaBuffer>,
        out: &mut CudaBuffer,
    ) -> Result<()> {
        portable::embed(ids, table, out)
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

    fn conv1d_silu(
        &self,
        x: &CudaBuffer,
        weight: &CudaBuffer,
        state: &mut CudaBuffer,
        out: &mut CudaBuffer,
        s: ConvShape,
    ) -> Result<()> {
        portable::conv1d_silu(x, weight, state, out, s)
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
        portable::gated_delta_net(qkv, ab, a_log, dt_bias, state, out, s)
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
    slots: HashMap<u64, i32>,
    seen: HashSet<u64>,
    /// Static keys recorded at least once: later runs replay.
    recorded: HashSet<u64>,
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
                let next = g.slots.len() as i32;
                let slot = *g.slots.entry(key).or_insert(next);
                (
                    Some(slot),
                    key & GRAPH_STATIC != 0 && g.recorded.contains(&key),
                )
            }
        };
        let Some(slot) = slot.filter(|&s| s < GRAPH_SLOTS) else {
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
        if !replay && key & GRAPH_STATIC != 0 {
            graphs
                .lock()
                .map_err(|_| Error::Backend("graph keys poisoned".into()))?
                .recorded
                .insert(key);
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
        let next_key = key | (seg + 1) << 56;
        let next = {
            let mut g = graphs
                .lock()
                .map_err(|_| Error::Backend("graph keys poisoned".into()))?;
            let n = g.slots.len() as i32;
            *g.slots.entry(next_key).or_insert(n)
        };
        if next >= GRAPH_SLOTS || seg + 1 >= 128 {
            return Ok(()); // out of slots: the step stays one graph
        }
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

    fn copy2d(&self, src: &CudaBuffer, dst: &mut CudaBuffer, shape: Copy2dShape) -> Result<()> {
        check::copy2d(src.len, dst.len, &shape)?;
        self.staged("copy2d", || self.kernels.copy2d(src, dst, shape))
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

    type Buffer = CudaBuffer;

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
        self.timed("conv1d_silu", || {
            self.kernels.conv1d_silu(x, weight, state, out, s)
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
        self.timed(op, || {
            self.kernels
                .gated_delta_net(qkv, ab, a_log, dt_bias, state, out, s)
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
