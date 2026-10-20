use std::fmt;

/// Every device op in the [`crate::Backend`] contract.
///
/// A new op is added on main in one change: this enum, the Backend method, the CPU reference,
/// the portable CUDA kernel with its `CudaKernels` default method, and a conformance case.
/// Hardware branches then list it in their native ops once they override it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Op {
    MatMul,
    Linear,
    Embed,
    RmsNorm,
    Rope,
    SiluMul,
    SigmoidMul,
    Add,
    Copy,
    Attention,
    Conv1dSilu,
    GatedDeltaNet,
    AttentionFull,
    LayerNorm,
    AddBias,
    Gelu,
    RopePositions,
    TopK,
}

impl Op {
    pub const ALL: &'static [Op] = &[
        Op::MatMul,
        Op::Linear,
        Op::Embed,
        Op::RmsNorm,
        Op::Rope,
        Op::SiluMul,
        Op::SigmoidMul,
        Op::Add,
        Op::Copy,
        Op::Attention,
        Op::Conv1dSilu,
        Op::GatedDeltaNet,
        Op::AttentionFull,
        Op::LayerNorm,
        Op::AddBias,
        Op::Gelu,
        Op::RopePositions,
        Op::TopK,
    ];

    pub fn name(&self) -> &'static str {
        match self {
            Op::MatMul => "matmul",
            Op::Linear => "linear",
            Op::Embed => "embed",
            Op::RmsNorm => "rms_norm",
            Op::Rope => "rope",
            Op::SiluMul => "silu_mul",
            Op::SigmoidMul => "sigmoid_mul",
            Op::Add => "add",
            Op::Copy => "copy",
            Op::Attention => "attention",
            Op::Conv1dSilu => "conv1d_silu",
            Op::GatedDeltaNet => "gated_delta_net",
            Op::AttentionFull => "attention_full",
            Op::LayerNorm => "layer_norm",
            Op::AddBias => "add_bias",
            Op::Gelu => "gelu",
            Op::RopePositions => "rope_positions",
            Op::TopK => "top_k",
        }
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}
