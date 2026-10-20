//! Backend and profile selection: explicit flags, then VELOX_PROFILE, then device detection.

use velox_core::{profile, Backend, DeviceProfile, Error, GpuArch, Result};
use velox_cpu::CpuBackend;

use crate::args::Args;

pub enum Selected {
    Cpu(CpuBackend),
    #[cfg(feature = "cuda")]
    Cuda(velox_cuda::CudaBackend),
}

impl Selected {
    /// Runs a closure over the selected backend's object-safe surface.
    pub fn with<R>(&self, f: impl Fn(&dyn Describe) -> R) -> R {
        match self {
            Selected::Cpu(b) => f(b),
            #[cfg(feature = "cuda")]
            Selected::Cuda(b) => f(b),
        }
    }
}

/// The parts of [`Backend`] that do not depend on its buffer type.
pub trait Describe {
    fn describe(&self) -> String;
    fn native_ops(&self) -> Vec<velox_core::Op>;
}

impl<B: Backend> Describe for B {
    fn describe(&self) -> String {
        Backend::describe(self)
    }
    fn native_ops(&self) -> Vec<velox_core::Op> {
        Backend::native_ops(self)
    }
}

/// (index, name, arch, SMs, total bytes, profile) of one visible CUDA device.
pub type DeviceRow = (
    i32,
    String,
    GpuArch,
    u32,
    u64,
    Option<&'static DeviceProfile>,
);

pub fn cuda_devices() -> Vec<DeviceRow> {
    #[cfg(feature = "cuda")]
    {
        velox_cuda::devices()
            .unwrap_or_default()
            .into_iter()
            .map(|d| {
                let p = d.profile();
                (d.index, d.name, d.arch, d.sm_count, d.total_mem, p)
            })
            .collect()
    }
    #[cfg(not(feature = "cuda"))]
    Vec::new()
}

pub fn profile(a: &Args) -> Result<Option<&'static DeviceProfile>> {
    let explicit = a
        .get("profile")
        .map(String::from)
        .or_else(|| std::env::var("VELOX_PROFILE").ok());
    if let Some(id) = explicit {
        return profile::by_id(&id).map(Some).ok_or_else(|| {
            Error::Config(format!("unknown profile '{id}' (see `velox profiles`)"))
        });
    }
    let index: i32 = a.parse("device", 0)?;
    Ok(cuda_devices()
        .into_iter()
        .find(|d| d.0 == index)
        .and_then(|d| d.5))
}

pub fn backend(a: &Args) -> Result<Selected> {
    let want = a.get("backend").unwrap_or("auto");
    match want {
        "cpu" => Ok(Selected::Cpu(CpuBackend)),
        "cuda" | "auto" => {
            #[cfg(feature = "cuda")]
            {
                let index: i32 = a.parse("device", 0)?;
                let dev = velox_cuda::devices()
                    .unwrap_or_default()
                    .into_iter()
                    .find(|d| d.index == index);
                match dev {
                    Some(dev) => {
                        let profile = profile(a)?.or_else(|| dev.profile());
                        if let Some(advice) = profile.and_then(|p| velox_variant::INFO.advice(p)) {
                            eprintln!("note: {advice}");
                        }
                        let kernels = velox_variant::cuda_kernels(&dev)?;
                        let b = velox_cuda::CudaBackend::new(
                            dev,
                            profile,
                            &velox_variant::INFO,
                            kernels,
                        )?;
                        return Ok(Selected::Cuda(b));
                    }
                    None if want == "cuda" => {
                        return Err(Error::Unsupported(format!("no CUDA device {index}")))
                    }
                    None => {}
                }
            }
            if want == "cuda" && !cfg!(feature = "cuda") {
                return Err(Error::Unsupported(
                    "built without CUDA; rebuild with --features cuda".into(),
                ));
            }
            Ok(Selected::Cpu(CpuBackend))
        }
        "hip" => Err(Error::Unsupported(
            "the HIP (AMD) backend is not implemented yet; see docs/adding-amd.md".into(),
        )),
        other => Err(Error::Config(format!("unknown backend '{other}'"))),
    }
}
