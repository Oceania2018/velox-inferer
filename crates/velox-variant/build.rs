// Compiles kernels/*.cu with the nvcc flags in kernels/gencode.txt when the `cuda` feature is on.
// main has no kernels/ directory; hardware branches add one. This file is identical on every
// branch so merges from main never conflict here.
use std::path::PathBuf;

fn main() {
    let dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("kernels");
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() || !dir.is_dir() {
        return;
    }
    let sources = velox_build::cu_sources(&dir);
    if sources.is_empty() {
        return;
    }
    let common =
        std::env::var("DEP_VELOX_CUDA_INCLUDE").expect("velox-cuda exports its kernel include dir");
    velox_build::CudaBuild::new("velox_variant_kernels")
        .sources(sources)
        .include(&dir)
        .include(common)
        .flags(velox_build::flags_from_file(&dir.join("gencode.txt")))
        .compile();
}
