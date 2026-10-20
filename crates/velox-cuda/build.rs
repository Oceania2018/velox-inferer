use std::path::PathBuf;

use velox_build::{cu_sources, gencode_flags, CudaBuild};

fn main() {
    let dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("kernels");
    // Portable kernels cover every supported card: Ada (sm_89) and consumer Blackwell (sm_120),
    // plus PTX of the newest so future NVIDIA cards can JIT them.
    let archs = std::env::var("VELOX_CUDA_ARCHS").unwrap_or_else(|_| "89;120".into());
    CudaBuild::new("velox_cuda_portable")
        .sources(cu_sources(&dir))
        .include(&dir)
        .flags(gencode_flags(&archs))
        .compile();
    // Exposed to dependents' build scripts as DEP_VELOX_CUDA_INCLUDE (velox-variant kernels).
    println!("cargo:include={}", dir.display());
}
