//! Hardware variant of this build: **rtx-5090-24g** (branch `rtx-5090-24g`).
//!
//! Tuned for the RTX 5090 Laptop GPU 24 GB (GB203, 82 SMs, 896 GB/s, sm_120a). Same contract as
//! every branch (see main's version of this file):
//!
//! - `INFO: VariantInfo`;
//! - `plan_policy(&DeviceProfile) -> PlanPolicy`;
//! - with the `cuda` feature, `cuda_kernels(&DeviceInfo) -> Result<Arc<dyn CudaKernels>>`.
//!
//! The kernel set is rtx-5060-ti's (same architecture; IQ3_XXS, NVFP4 and FP8 linears, chunked
//! Gated DeltaNet), since this card runs the same 24 GB-friendly IQ3_XXS artifact. File and
//! symbol names stay `rtx5060` / `velox_5060` so rtx-5060-ti kernel commits apply here
//! unchanged; grid sizing (split-K, verify splits) is for 82 SMs. Every op, and every shape a
//! tuned kernel does not cover, runs the portable kernel.

use velox_core::{DeviceProfile, GpuArch, PlanPolicy, VariantInfo};

#[cfg(feature = "cuda")]
mod kernels;

pub static INFO: VariantInfo = VariantInfo {
    id: "rtx-5090-24g",
    branch: "rtx-5090-24g",
    description:
        "RTX 5090 Laptop GPU 24 GB (GB203, 82 SMs): IQ3_XXS/NVFP4/FP8 tensor-core linear, skinny verify kernels",
    profiles: &["rtx-5090-laptop"],
    archs: &[GpuArch::Cuda {
        major: 12,
        minor: 0,
    }],
};

pub fn plan_policy(profile: &DeviceProfile) -> PlanPolicy {
    let mut p = PlanPolicy::for_profile(profile);
    if INFO.tunes(profile) {
        // Laptops often share the GPU with the desktop compositor and run power-limited; keep
        // a larger reserve and the default 2048-token prefill chunk.
        p.reserved_mib = p.reserved_mib.max(1536);
    }
    p
}

#[cfg(feature = "cuda")]
pub fn cuda_kernels(
    device: &velox_cuda::DeviceInfo,
) -> velox_core::Result<std::sync::Arc<dyn velox_cuda::CudaKernels>> {
    velox_cuda::require_arch(device, &INFO)?;
    Ok(std::sync::Arc::new(kernels::Rtx5060Kernels))
}
