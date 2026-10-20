//! Hardware variant of this build: **rtx-4090** (branch `rtx-4090`).
//!
//! RTX 4090 24 GB (AD102, 128 SMs, sm_89): portable kernels; no TMA, no FP4
//!
//! Same contract as every branch (see main's version of this file): `INFO`, `plan_policy`,
//! and with the `cuda` feature `cuda_kernels`. No op is tuned here yet: every op runs the
//! portable kernel. Tuned kernels are added under `kernels/` (with `kernels/gencode.txt`) and
//! listed in `native_ops`, following rtx-5090-32g; see docs/branching.md.

use velox_core::{DeviceProfile, GpuArch, PlanPolicy, VariantInfo};

pub static INFO: VariantInfo = VariantInfo {
    id: "rtx-4090",
    branch: "rtx-4090",
    description: "RTX 4090 24 GB (AD102, 128 SMs, sm_89): portable kernels; no TMA, no FP4",
    profiles: &["rtx-4090"],
    archs: &[GpuArch::Cuda { major: 8, minor: 9 }],
};

pub fn plan_policy(profile: &DeviceProfile) -> PlanPolicy {
    PlanPolicy::for_profile(profile)
}

#[cfg(feature = "cuda")]
pub fn cuda_kernels(
    device: &velox_cuda::DeviceInfo,
) -> velox_core::Result<std::sync::Arc<dyn velox_cuda::CudaKernels>> {
    velox_cuda::require_arch(device, &INFO)?;
    Ok(std::sync::Arc::new(velox_cuda::PortableKernels))
}
