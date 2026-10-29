//! Hardware variant of this build: **rtx-4090** (branch `rtx-4090`).
//!
//! Tuned for the RTX 4090 24 GB (AD102, 128 SMs, 1008 GB/s, sm_89). Same contract as every branch
//! (see main's version of this file):
//!
//! - `INFO: VariantInfo`;
//! - `plan_policy(&DeviceProfile) -> PlanPolicy`;
//! - with the `cuda` feature, `cuda_kernels(&DeviceInfo) -> Result<Arc<dyn CudaKernels>>`.
//!
//! The kernel set starts from rtx-5060-ti's (the GGML / IQ3_XXS kernels of the GSQ IQ2_S
//! package), with grids sized for 128 SMs. Ada has int8, bf16 and FP8 (E4M3) tensor cores but no
//! FP4 ones and no block-scaled `mma`: NVFP4 weights decode to bf16 in software, FP8 operands run
//! the plain `e4m3` `mma.sync`, and nothing takes 4-bit activations. Every op, and every shape a
//! tuned kernel does not cover, runs the portable kernel.

use velox_core::{DeviceProfile, GpuArch, PlanPolicy, VariantInfo};

#[cfg(feature = "cuda")]
mod kernels;

pub static INFO: VariantInfo = VariantInfo {
    id: "rtx-4090",
    branch: "rtx-4090",
    description: "RTX 4090 24 GB (AD102, 128 SMs): int8 / bf16 / FP8 tensor-core GGML and IQ3_XXS linear, no FP4",
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
    Ok(std::sync::Arc::new(kernels::Rtx4090Kernels))
}
