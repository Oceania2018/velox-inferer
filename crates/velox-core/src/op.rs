use std::fmt;

/// Every device op in the [`crate::Backend`] contract.
///
/// A new op is added on main in one change: this enum, the Backend method, the CPU reference,
/// the portable CUDA kernel with its `CudaKernels` default method, and a conformance case.
/// Hardware branches then list it in their native ops once they override it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Op {
    MatMul,
    RmsNorm,
    Rope,
    SiluMul,
    Add,
    Copy,
    Attention,
}

impl Op {
    pub const ALL: &'static [Op] = &[
        Op::MatMul,
        Op::RmsNorm,
        Op::Rope,
        Op::SiluMul,
        Op::Add,
        Op::Copy,
        Op::Attention,
    ];

    pub fn name(&self) -> &'static str {
        match self {
            Op::MatMul => "matmul",
            Op::RmsNorm => "rms_norm",
            Op::Rope => "rope",
            Op::SiluMul => "silu_mul",
            Op::Add => "add",
            Op::Copy => "copy",
            Op::Attention => "attention",
        }
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}
