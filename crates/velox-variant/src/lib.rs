//! Hardware variant of this build: **rtx-5060-ti** (branch `rtx-5060-ti`).
//!
//! Tuned for the RTX 5060 Ti 16 GB (GB206, 36 SMs, 448 GB/s, sm_120a). Same contract as every
//! branch (see main's version of this file):
//!
//! - `INFO: VariantInfo`;
//! - `plan_policy(&DeviceProfile) -> PlanPolicy`;
//! - with the `cuda` feature, `cuda_kernels(&DeviceInfo) -> Result<Arc<dyn CudaKernels>>`.
//!
//! The kernel set starts from rtx-5090-32g's (same architecture), with grids sized for 36 SMs.
//! Every op, and every shape a tuned kernel does not cover, runs the portable kernel.

use velox_core::{DeviceProfile, GpuArch, PlanPolicy, VariantInfo};

#[cfg(feature = "cuda")]
mod kernels;

pub static INFO: VariantInfo = VariantInfo {
    id: "rtx-5060-ti",
    branch: "rtx-5060-ti",
    description:
        "RTX 5060 Ti 16 GB (GB206, 36 SMs): NVFP4/FP8 tensor-core linear, skinny verify kernels",
    profiles: &["rtx-5060-ti-16g"],
    archs: &[GpuArch::Cuda {
        major: 12,
        minor: 0,
    }],
};

pub fn plan_policy(profile: &DeviceProfile) -> PlanPolicy {
    PlanPolicy::for_profile(profile)
}

#[cfg(feature = "cuda")]
pub fn cuda_kernels(
    device: &velox_cuda::DeviceInfo,
) -> velox_core::Result<std::sync::Arc<dyn velox_cuda::CudaKernels>> {
    velox_cuda::require_arch(device, &INFO)?;
    Ok(std::sync::Arc::new(kernels::Rtx5060Kernels))
}
