//! Hardware variant of this build: **rtx-5090-32g** (branch `rtx-5090-32g`).
//!
//! Tuned for the 32 GB desktop RTX 5090 (GB202, 170 SMs, sm_120a). Same contract as every
//! branch (see main's version of this file):
//!
//! - `INFO: VariantInfo`;
//! - `plan_policy(&DeviceProfile) -> PlanPolicy`;
//! - with the `cuda` feature, `cuda_kernels(&DeviceInfo) -> Result<Arc<dyn CudaKernels>>`.
//!
//! The kernel set combines rtx-5090-24g's (rtx-5060-ti lineage: IQ3_XXS, int8 prefill for
//! IQ3_XXS / q4 / q5, A16 NVFP4, chunked Gated DeltaNet, MoE, NVFP4 verify attention; grids for 170
//! SMs) with this branch's own kernels where they measure faster here (NVFP4 A4 prefill and fused
//! SwiGLU, FP8, f32-cache flash attention, matmul, groupwise bf16 prefill); `kernels.rs`
//! documents each choice. Every other op, and every shape a tuned kernel does not cover, runs the
//! portable kernel.

use velox_core::{DeviceProfile, GpuArch, PlanPolicy, VariantInfo};

#[cfg(feature = "cuda")]
mod kernels;

pub static INFO: VariantInfo = VariantInfo {
    id: "rtx-5090-32g",
    branch: "rtx-5090-32g",
    description: "RTX 5090 32 GB: NVFP4/FP8/IQ3_XXS tensor-core linear, chunked GDN, skinny verify kernels",
    profiles: &["rtx-5090"],
    archs: &[GpuArch::Cuda { major: 12, minor: 0 }],
};

pub fn plan_policy(profile: &DeviceProfile) -> PlanPolicy {
    let mut p = PlanPolicy::for_profile(profile);
    if INFO.tunes(profile) {
        // 32 GB leaves room for larger prefill chunks, which cut per-chunk overhead.
        p.workspace_tokens = 4096;
    }
    p
}

#[cfg(feature = "cuda")]
pub fn cuda_kernels(
    device: &velox_cuda::DeviceInfo,
) -> velox_core::Result<std::sync::Arc<dyn velox_cuda::CudaKernels>> {
    velox_cuda::require_arch(device, &INFO)?;
    Ok(std::sync::Arc::new(kernels::Rtx5090Kernels))
}
