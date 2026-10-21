//! Build-script helper that compiles CUDA sources with nvcc into a static library and emits the
//! cargo link directives. Shared by `velox-cuda` and every branch's `velox-variant`, so hardware
//! branches add `.cu` files without adding build dependencies.
//!
//! Environment:
//! - `NVCC`: nvcc path (default: `$CUDA_PATH/bin/nvcc`, then `nvcc` on PATH).
//! - `CUDA_PATH` / `CUDA_HOME`: toolkit root used to locate cudart.
//! - `VELOX_CUDA_ARCHS`: portable-kernel architectures, e.g. `89;120` (see [`gencode_flags`]).

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

pub fn cuda_root() -> Option<PathBuf> {
    ["CUDA_PATH", "CUDA_HOME"]
        .iter()
        .find_map(env::var_os)
        .map(PathBuf::from)
        .or_else(|| Some(PathBuf::from("/usr/local/cuda")).filter(|p| p.exists()))
}

pub fn nvcc() -> PathBuf {
    if let Some(p) = env::var_os("NVCC") {
        return PathBuf::from(p);
    }
    let exe = if cfg!(windows) { "nvcc.exe" } else { "nvcc" };
    cuda_root()
        .map(|r| r.join("bin").join(exe))
        .filter(|p| p.exists())
        .unwrap_or_else(|| PathBuf::from(exe))
}

/// `-gencode` flags for a `;`-separated list such as `89;120a`: SASS for each entry plus PTX of
/// the last one, so newer cards can still JIT the portable kernels.
pub fn gencode_flags(archs: &str) -> Vec<String> {
    let list: Vec<&str> = archs
        .split(';')
        .map(str::trim)
        .filter(|a| !a.is_empty())
        .collect();
    let mut flags = Vec::new();
    for a in &list {
        flags.push("-gencode".into());
        flags.push(format!("arch=compute_{a},code=sm_{a}"));
    }
    if let Some(last) = list.last() {
        flags.push("-gencode".into());
        flags.push(format!("arch=compute_{last},code=compute_{last}"));
    }
    flags
}

/// Reads whitespace-separated nvcc flags from a file, ignoring `#` comments.
pub fn flags_from_file(path: &Path) -> Vec<String> {
    println!("cargo:rerun-if-changed={}", path.display());
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .map(|l| l.split('#').next().unwrap_or(""))
        .flat_map(str::split_whitespace)
        .map(String::from)
        .collect()
}

/// `.cu` files directly inside `dir`, sorted for reproducible builds.
pub fn cu_sources(dir: &Path) -> Vec<PathBuf> {
    println!("cargo:rerun-if-changed={}", dir.display());
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|e| e == "cu"))
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

pub struct CudaBuild {
    lib: String,
    sources: Vec<PathBuf>,
    includes: Vec<PathBuf>,
    flags: Vec<String>,
}

impl CudaBuild {
    pub fn new(lib: &str) -> Self {
        CudaBuild {
            lib: lib.into(),
            sources: Vec::new(),
            includes: Vec::new(),
            flags: Vec::new(),
        }
    }

    pub fn sources(mut self, s: impl IntoIterator<Item = PathBuf>) -> Self {
        self.sources.extend(s);
        self
    }

    pub fn include(mut self, dir: impl Into<PathBuf>) -> Self {
        self.includes.push(dir.into());
        self
    }

    pub fn flags(mut self, f: impl IntoIterator<Item = String>) -> Self {
        self.flags.extend(f);
        self
    }

    /// Compiles every source and links the resulting static library plus cudart. Panics with the
    /// nvcc command line on failure, which is the build-script way to report errors.
    pub fn compile(self) {
        for v in ["NVCC", "CUDA_PATH", "CUDA_HOME", "VELOX_CUDA_ARCHS"] {
            println!("cargo:rerun-if-env-changed={v}");
        }
        let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by cargo"));
        let nvcc = nvcc();
        let mut objects = Vec::new();
        for src in &self.sources {
            println!("cargo:rerun-if-changed={}", src.display());
            let stem = src.file_stem().unwrap().to_string_lossy();
            let obj = out.join(format!("{}_{stem}.o", self.lib));
            let mut cmd = Command::new(&nvcc);
            // Per-thread default stream: kernels launched without a stream can be graph-captured.
            cmd.args([
                "-c",
                "-O3",
                "-std=c++17",
                "-lineinfo",
                "--default-stream",
                "per-thread",
            ])
            .args(&self.flags);
            if !cfg!(windows) {
                cmd.args(["-Xcompiler", "-fPIC"]);
            }
            for inc in &self.includes {
                println!("cargo:rerun-if-changed={}", inc.display());
                cmd.arg("-I").arg(inc);
            }
            cmd.arg(src).arg("-o").arg(&obj);
            run(&mut cmd);
            objects.push(obj);
        }
        let lib_file = if cfg!(windows) {
            format!("{}.lib", self.lib)
        } else {
            format!("lib{}.a", self.lib)
        };
        let mut ar = Command::new(&nvcc);
        ar.arg("-lib")
            .args(&objects)
            .arg("-o")
            .arg(out.join(lib_file));
        run(&mut ar);

        println!("cargo:rustc-link-search=native={}", out.display());
        println!("cargo:rustc-link-lib=static={}", self.lib);
        if let Some(root) = cuda_root() {
            let sub = if cfg!(windows) { "lib/x64" } else { "lib64" };
            println!(
                "cargo:rustc-link-search=native={}",
                root.join(sub).display()
            );
            println!(
                "cargo:rustc-link-search=native={}",
                root.join("targets/x86_64-linux/lib").display()
            );
        }
        println!("cargo:rustc-link-lib=dylib=cudart");
        if !cfg!(windows) {
            println!("cargo:rustc-link-lib=dylib=stdc++");
        }
    }
}

fn run(cmd: &mut Command) {
    let status = cmd.status().unwrap_or_else(|e| {
        panic!(
            "failed to start {cmd:?}: {e}. Is the CUDA toolkit installed (set NVCC or CUDA_PATH)?"
        )
    });
    if !status.success() {
        panic!("nvcc failed ({status}): {cmd:?}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gencode_adds_ptx_for_last_arch() {
        assert_eq!(
            gencode_flags("89; 120"),
            [
                "-gencode",
                "arch=compute_89,code=sm_89",
                "-gencode",
                "arch=compute_120,code=sm_120",
                "-gencode",
                "arch=compute_120,code=compute_120"
            ]
        );
        assert!(gencode_flags("").is_empty());
    }
}
