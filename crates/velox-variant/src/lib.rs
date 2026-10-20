//! Hardware variant of this build: **generic** (branch `main`).
//!
//! This crate is the only code that differs between hardware branches. Every branch exposes the
//! same contract, which the CLI and conformance suite compile against:
//!
//! - `INFO: VariantInfo`: id, branch, the profiles it tunes and the architectures its kernels run on;
//! - `plan_policy(&DeviceProfile) -> PlanPolicy`: VRAM reserve and workspace per card;
//! - with the `cuda` feature, `cuda_kernels(&DeviceInfo) -> Result<Arc<dyn CudaKernels>>`.
//!
//! The generic variant runs the portable kernels on any device. See docs/branching.md for how a
//! hardware branch replaces this crate.

use velox_core::{DeviceProfile, PlanPolicy, VariantInfo};

pub static INFO: VariantInfo = VariantInfo {
    id: "generic",
    branch: "main",
    description: "portable kernels for every supported GPU; correct everywhere, tuned nowhere",
    profiles: &[],
    archs: &[],
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
