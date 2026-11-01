//! Hardware variant of this build: **rtx-pro-6000** (branch `rtx-pro-6000`).
//!
//! Tuned for the 96 GB RTX PRO 6000 Blackwell (GB202, 188 SMs, sm_120a). The kernel set is the
//! RTX 5090's (same die); this branch differs in the cards it claims. Same contract as every
//! branch (see main's version of this file):
//!
//! - `INFO: VariantInfo`;
//! - `plan_policy(&DeviceProfile) -> PlanPolicy`;
//! - with the `cuda` feature, `cuda_kernels(&DeviceInfo) -> Result<Arc<dyn CudaKernels>>`.
//!
//! Native ops: MatMul (small-M GEMV tile, tiled SGEMM) and Linear (NVFP4 x NVFP4 and FP8 x FP8
//! block-scaled MMA for prefill, where the artifact allows rounded activations). Every other op,
//! and every shape a tuned kernel does not cover, runs the portable kernel.

use velox_core::{DeviceProfile, GpuArch, PlanPolicy, VariantInfo};

#[cfg(feature = "cuda")]
mod kernels;

pub static INFO: VariantInfo = VariantInfo {
    id: "rtx-pro-6000",
    branch: "rtx-pro-6000",
    description: "RTX PRO 6000 96 GB: NVFP4/FP8 tensor-core prefill linear, tiled SGEMM matmul",
    profiles: &["rtx-pro-6000"],
    archs: &[GpuArch::Cuda { major: 12, minor: 0 }],
};

pub fn plan_policy(profile: &DeviceProfile) -> PlanPolicy {
    let mut p = PlanPolicy::for_profile(profile);
    if INFO.tunes(profile) {
        // 96 GB leaves room for larger prefill chunks, which cut per-chunk overhead.
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
