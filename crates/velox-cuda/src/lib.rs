//! CUDA backend.
//!
//! [`CudaBackend`] validates every call and forwards it to a [`CudaKernels`] implementation. Each
//! `CudaKernels` method defaults to the portable kernel, so a hardware branch overrides only the
//! ops it has tuned and inherits everything new that lands on main.

use std::ffi::{c_char, c_void, CStr};
use std::sync::Arc;

use velox_core::backend::{AttnShape, Backend, BackendKind, RopeShape};
use velox_core::{check, profile, DeviceProfile, Error, GpuArch, Op, Result, VariantInfo};

pub mod portable;

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
        pub fn velox_cuda_dtod(dst: *mut c_void, src: *const c_void, bytes: usize) -> i32;
        pub fn velox_cuda_synchronize() -> i32;
        pub fn velox_cuda_error_string(code: i32) -> *const c_char;
        pub fn velox_cuda_runtime_version(v: *mut i32) -> i32;
        pub fn velox_cuda_driver_version(v: *mut i32) -> i32;
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

    fn rms_norm(
        &self,
        x: &CudaBuffer,
        weight: &CudaBuffer,
        y: &mut CudaBuffer,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<()> {
        portable::rms_norm(x, weight, y, rows, dim, eps)
    }

    fn rope(&self, x: &mut CudaBuffer, s: RopeShape) -> Result<()> {
        portable::rope(x, s)
    }

    fn silu_mul(&self, gate: &mut CudaBuffer, up: &CudaBuffer, len: usize) -> Result<()> {
        portable::silu_mul(gate, up, len)
    }

    fn add(&self, acc: &mut CudaBuffer, x: &CudaBuffer, len: usize) -> Result<()> {
        portable::add(acc, x, len)
    }

    fn copy(
        &self,
        src: &CudaBuffer,
        dst: &mut CudaBuffer,
        dst_offset: usize,
        len: usize,
    ) -> Result<()> {
        portable::copy(src, dst, dst_offset, len)
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
}

/// Every op on the portable kernels; the CUDA kernel set of the generic variant on main.
pub struct PortableKernels;

impl CudaKernels for PortableKernels {}

pub struct CudaBackend {
    device: DeviceInfo,
    profile: Option<&'static DeviceProfile>,
    variant: &'static VariantInfo,
    kernels: Arc<dyn CudaKernels>,
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
        Ok(CudaBackend {
            device,
            profile,
            variant,
            kernels,
        })
    }

    pub fn device(&self) -> &DeviceInfo {
        &self.device
    }
}

impl Backend for CudaBackend {
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
        let buf = CudaBuffer::new(data.len())?;
        if !data.is_empty() {
            // SAFETY: buf holds data.len() f32 elements.
            status(
                unsafe {
                    ffi::velox_cuda_htod(buf.ptr.cast(), data.as_ptr().cast(), data.len() * 4)
                },
                "upload",
            )?;
        }
        Ok(buf)
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
        self.kernels.matmul(x, w, y, m, n, k)
    }

    fn rms_norm(
        &self,
        x: &CudaBuffer,
        weight: &CudaBuffer,
        y: &mut CudaBuffer,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<()> {
        check::rms_norm(x.len, weight.len, y.len, rows, dim)?;
        self.kernels.rms_norm(x, weight, y, rows, dim, eps)
    }

    fn rope(&self, x: &mut CudaBuffer, s: RopeShape) -> Result<()> {
        check::rope(x.len, &s)?;
        self.kernels.rope(x, s)
    }

    fn silu_mul(&self, gate: &mut CudaBuffer, up: &CudaBuffer, len: usize) -> Result<()> {
        check::elementwise("silu_mul", &[gate.len, up.len], len)?;
        self.kernels.silu_mul(gate, up, len)
    }

    fn add(&self, acc: &mut CudaBuffer, x: &CudaBuffer, len: usize) -> Result<()> {
        check::elementwise("add", &[acc.len, x.len], len)?;
        self.kernels.add(acc, x, len)
    }

    fn copy(
        &self,
        src: &CudaBuffer,
        dst: &mut CudaBuffer,
        dst_offset: usize,
        len: usize,
    ) -> Result<()> {
        check::copy(src.len, dst.len, dst_offset, len)?;
        self.kernels.copy(src, dst, dst_offset, len)
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
        self.kernels.attention(q, k, v, out, s)
    }
}

pub(crate) fn dtod(dst: *mut f32, src: *const f32, len: usize) -> Result<()> {
    // SAFETY: callers pass validated ranges inside live allocations.
    status(
        unsafe { ffi::velox_cuda_dtod(dst.cast(), src.cast(), len * 4) },
        "copy",
    )
}
