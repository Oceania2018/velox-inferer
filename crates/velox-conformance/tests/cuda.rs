//! The suite on every visible CUDA device with this branch's variant kernels.
//! Requires `--features cuda`; passes vacuously when no device is visible.
#![cfg(feature = "cuda")]

use velox_conformance::run;
use velox_cuda::CudaBackend;

#[test]
fn variant_kernels_match_reference() {
    let devices = match velox_cuda::devices() {
        Ok(d) if !d.is_empty() => d,
        other => {
            eprintln!("skipping: no CUDA device ({other:?})");
            return;
        }
    };
    for dev in devices {
        let profile = dev.profile();
        let kernels = velox_variant::cuda_kernels(&dev).unwrap();
        let backend = CudaBackend::new(dev, profile, &velox_variant::INFO, kernels).unwrap();
        let failed: Vec<_> = run(&backend)
            .unwrap()
            .into_iter()
            .filter(|r| !r.passed())
            .collect();
        assert!(
            failed.is_empty(),
            "{}: {failed:#?}",
            velox_core::Backend::describe(&backend)
        );
    }
}
