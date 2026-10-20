//! The device op contract every backend implements (CPU, CUDA, and later HIP).
//!
//! Layouts are row-major f32. Implementations must validate shapes with [`crate::check`] so that
//! every backend rejects the same malformed calls. Buffers may be longer than an op needs, which
//! lets a model reuse workspaces sized for its largest batch.

use std::fmt;

use crate::error::Result;
use crate::op::Op;
use crate::profile::DeviceProfile;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,
    Cuda,
    /// AMD ROCm/HIP. Reserved: no implementation yet (docs/adding-amd.md).
    Hip,
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            BackendKind::Cpu => "cpu",
            BackendKind::Cuda => "cuda",
            BackendKind::Hip => "hip",
        })
    }
}

/// Causal grouped-query attention over a KV cache.
///
/// `q` is `[tokens, heads, head_dim]` for positions `pos0 .. pos0 + tokens`. `k` and `v` are
/// `[capacity, kv_heads, head_dim]` caches whose first `pos0 + tokens` rows are valid. Query head
/// `h` reads KV head `h / (heads / kv_heads)`. Token `t` attends to keys `0 ..= pos0 + t`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttnShape {
    pub tokens: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub pos0: usize,
}

impl AttnShape {
    pub fn kv_len(&self) -> usize {
        self.pos0 + self.tokens
    }

    pub fn scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }
}

/// Rotary embedding (rotate-half / NeoX layout) applied in place to `[tokens, heads, head_dim]`.
/// Pair `(i, i + head_dim/2)` of token `t` rotates by `(pos0 + t) * theta^(-2i/head_dim)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeShape {
    pub tokens: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub pos0: usize,
    pub theta: f32,
}

pub trait Backend {
    type Buffer;

    fn kind(&self) -> BackendKind;
    /// Human-readable description, e.g. "cuda:0 NVIDIA GeForce RTX 5090 [variant rtx-5090]".
    fn describe(&self) -> String;
    fn profile(&self) -> Option<&'static DeviceProfile>;
    /// Ops served by hardware-tuned code; everything else runs the portable implementation.
    fn native_ops(&self) -> Vec<Op>;

    /// Zero-initialised buffer of `len` f32 elements.
    fn alloc(&self, len: usize) -> Result<Self::Buffer>;
    fn upload(&self, data: &[f32]) -> Result<Self::Buffer>;
    fn download(&self, buf: &Self::Buffer, out: &mut [f32]) -> Result<()>;
    fn len(&self, buf: &Self::Buffer) -> usize;
    /// Waits for all queued work; timing and downloads rely on it.
    fn synchronize(&self) -> Result<()>;

    /// `y[m, n] = x[m, k] · w[n, k]ᵀ` (weights stored as output rows, like PyTorch Linear).
    fn matmul(
        &self,
        x: &Self::Buffer,
        w: &Self::Buffer,
        y: &mut Self::Buffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()>;
    /// `y[r] = x[r] / sqrt(mean(x[r]²) + eps) * weight` for each of `rows` rows of width `dim`.
    fn rms_norm(
        &self,
        x: &Self::Buffer,
        weight: &Self::Buffer,
        y: &mut Self::Buffer,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<()>;
    fn rope(&self, x: &mut Self::Buffer, shape: RopeShape) -> Result<()>;
    /// `gate[i] = silu(gate[i]) * up[i]` for the first `len` elements.
    fn silu_mul(&self, gate: &mut Self::Buffer, up: &Self::Buffer, len: usize) -> Result<()>;
    /// `acc[i] += x[i]` for the first `len` elements.
    fn add(&self, acc: &mut Self::Buffer, x: &Self::Buffer, len: usize) -> Result<()>;
    /// `dst[dst_offset ..][.. len] = src[.. len]`.
    fn copy(
        &self,
        src: &Self::Buffer,
        dst: &mut Self::Buffer,
        dst_offset: usize,
        len: usize,
    ) -> Result<()>;
    /// Writes `[tokens, heads, head_dim]` into `out`; see [`AttnShape`].
    fn attention(
        &self,
        q: &Self::Buffer,
        k: &Self::Buffer,
        v: &Self::Buffer,
        out: &mut Self::Buffer,
        shape: AttnShape,
    ) -> Result<()>;
}
