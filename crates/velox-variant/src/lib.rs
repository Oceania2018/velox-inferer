//! Hardware variant of this build: **rtx-5060-ti** (branch `rtx-5060-ti`).
//!
//! RTX 5060 Ti 16 GB / 8 GB (GB206, 36 SMs): portable kernels, small-VRAM policy
//!
//! Same contract as every branch (see main's version of this file): `INFO`, `plan_policy`,
//! and with the `cuda` feature `cuda_kernels`. No op is tuned here yet: every op runs the
//! portable kernel. Tuned kernels are added under `kernels/` (with `kernels/gencode.txt`) and
//! listed in `native_ops`, following rtx-5090-32g; see docs/branching.md.

use velox_core::{DeviceProfile, GpuArch, PlanPolicy, VariantInfo};

pub static INFO: VariantInfo = VariantInfo {
    id: "rtx-5060-ti",
    branch: "rtx-5060-ti",
    description: "RTX 5060 Ti 16 GB / 8 GB (GB206, 36 SMs): portable kernels, small-VRAM policy",
    profiles: &["rtx-5060-ti-16g", "rtx-5060-ti-8g"],
    archs: &[GpuArch::Cuda {
        major: 12,
        minor: 0,
    }],
};

pub fn plan_policy(profile: &DeviceProfile) -> PlanPolicy {
    let mut p = PlanPolicy::for_profile(profile);
    if profile.id == "rtx-5060-ti-8g" {
        // 8 GB cannot spare a 2048-token f32 workspace next to weights and KV.
        p.workspace_tokens = 1024;
    }
    p
}

#[cfg(feature = "cuda")]
pub fn cuda_kernels(
    device: &velox_cuda::DeviceInfo,
) -> velox_core::Result<std::sync::Arc<dyn velox_cuda::CudaKernels>> {
    velox_cuda::require_arch(device, &INFO)?;
    Ok(std::sync::Arc::new(velox_cuda::PortableKernels))
}
