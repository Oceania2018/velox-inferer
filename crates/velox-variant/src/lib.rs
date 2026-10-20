//! Hardware variant of this build: **rtx-5090-24g** (branch `rtx-5090-24g`).
//!
//! RTX 5090 Laptop GPU 24 GB (GB203, 82 SMs): portable kernels, laptop VRAM policy
//!
//! Same contract as every branch (see main's version of this file): `INFO`, `plan_policy`,
//! and with the `cuda` feature `cuda_kernels`. No op is tuned here yet: every op runs the
//! portable kernel. Tuned kernels are added under `kernels/` (with `kernels/gencode.txt`) and
//! listed in `native_ops`, following rtx-5090-32g; see docs/branching.md.

use velox_core::{DeviceProfile, GpuArch, PlanPolicy, VariantInfo};

pub static INFO: VariantInfo = VariantInfo {
    id: "rtx-5090-24g",
    branch: "rtx-5090-24g",
    description: "RTX 5090 Laptop GPU 24 GB (GB203, 82 SMs): portable kernels, laptop VRAM policy",
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
    Ok(std::sync::Arc::new(velox_cuda::PortableKernels))
}
