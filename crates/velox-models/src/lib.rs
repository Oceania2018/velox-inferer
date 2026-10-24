//! Model architectures. Each model is written only against `velox_core::Backend`, so the same
//! code runs on the CPU oracle, portable CUDA kernels and every hardware branch's tuned kernels.

pub mod qwen3_5;
pub mod qwen4_exp;
