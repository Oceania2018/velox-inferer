//! The device op contract every backend implements (CPU, CUDA, and later HIP).
//!
//! Layouts are row-major f32. Implementations must validate shapes with [`crate::check`] so that
//! every backend rejects the same malformed calls. Buffers may be longer than an op needs, which
//! lets a model reuse workspaces sized for its largest batch.

use std::fmt;

use crate::error::Result;
use crate::format::WeightFormat;
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

/// Rotary embedding (rotate-half / NeoX layout) applied in place to `[tokens, heads, head_dim]`,
/// over the first `rotary_dim` dims of each head (the rest pass through). With `R = rotary_dim`,
/// pair `(i, i + R/2)` of token `t` rotates by `(pos0 + t) * theta^(-2i/R)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeShape {
    pub tokens: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub pos0: usize,
    pub theta: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormKind {
    /// `y = x * rsqrt(mean(x²) + eps) * w`
    Plain,
    /// `y = x * rsqrt(mean(x²) + eps) * (1 + w)` (zero-centred weights)
    Offset,
}

/// Causal depthwise convolution over `tokens` rows of `channels`, kernel `width`, then SiLU:
/// with `u = [state; x]` (`state` holds the previous `width - 1` input rows, oldest first),
/// `out[t, c] = silu(Σ_j weight[j, c] · u[t + j, c])`; tap `width - 1` is the current token.
/// `state` is updated to the last `width - 1` rows of `u`. `weight` is `[width, channels]` f32.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvShape {
    pub tokens: usize,
    pub channels: usize,
    pub width: usize,
}

/// Gated DeltaNet over `tokens` steps.
///
/// `qkv` rows are `[q (k_heads·k_dim) | k (k_heads·k_dim) | v (v_heads·v_dim)]`; `ab` rows are
/// `[a (v_heads) | b (v_heads)]`. Value head `h` uses key head `h / (v_heads / k_heads)`.
/// `state` is `[v_heads, v_dim, k_dim]` f32, updated in place; `out` is `[tokens, v_heads, v_dim]`.
/// Per step: `g = -exp(a_log)·softplus(a + dt_bias)`, `β = sigmoid(b)`, q and k L2-normalised
/// (`x / sqrt(Σx² + 1e-6)`), `S = e^g·S`, `S += (β(v − S k)) ⊗ k`, `o = S q / sqrt(k_dim)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnShape {
    pub tokens: usize,
    pub k_heads: usize,
    pub v_heads: usize,
    pub k_dim: usize,
    pub v_dim: usize,
}

impl GdnShape {
    pub fn qkv_width(&self) -> usize {
        2 * self.k_heads * self.k_dim + self.v_heads * self.v_dim
    }
}

/// Rows `[row_begin, row_begin + rows)` of a stored weight matrix `[parent_rows, k]`, kept in its
/// encoded form. Views keep the parent geometry because plane offsets and NVFP4 scale swizzles
/// depend on it.
pub struct WeightView<'a, Buf> {
    pub buf: &'a Buf,
    pub format: WeightFormat,
    pub parent_rows: usize,
    pub k: usize,
    pub row_begin: usize,
    pub rows: usize,
}

impl<Buf> Clone for WeightView<'_, Buf> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Buf> Copy for WeightView<'_, Buf> {}

impl<'a, Buf> WeightView<'a, Buf> {
    /// The whole parent matrix.
    pub fn full(buf: &'a Buf, format: WeightFormat, n: usize, k: usize) -> Self {
        WeightView {
            buf,
            format,
            parent_rows: n,
            k,
            row_begin: 0,
            rows: n,
        }
    }

    /// Rows `[begin, begin + rows)` of this view.
    pub fn rows(&self, begin: usize, rows: usize) -> Self {
        WeightView {
            row_begin: self.row_begin + begin,
            rows,
            ..*self
        }
    }
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
    /// Uploads raw bytes (an encoded weight). The buffer holds `ceil(bytes / 4)` f32 slots; the
    /// tail of the last slot is zero.
    fn upload_bytes(&self, data: &[u8]) -> Result<Self::Buffer>;
    /// Waits for all queued work; timing and downloads rely on it.
    fn synchronize(&self) -> Result<()>;

    /// `y[m, rows] = x[m, k] · Wᵀ` with `W` the view's rows decoded from their stored format.
    fn linear(
        &self,
        x: &Self::Buffer,
        w: &WeightView<'_, Self::Buffer>,
        y: &mut Self::Buffer,
        m: usize,
    ) -> Result<()>;
    /// Embedding lookup: `out[t, :] = W[ids[t], :]` decoded, for a `[vocab, k]` table view.
    fn embed(
        &self,
        ids: &[u32],
        table: &WeightView<'_, Self::Buffer>,
        out: &mut Self::Buffer,
    ) -> Result<()>;
    /// RMS norm of `rows` rows of width `dim`; see [`NormKind`] for the weighting.
    #[allow(clippy::too_many_arguments)]
    fn rms_norm(
        &self,
        x: &Self::Buffer,
        weight: &Self::Buffer,
        y: &mut Self::Buffer,
        rows: usize,
        dim: usize,
        eps: f32,
        kind: NormKind,
    ) -> Result<()>;
    /// `x[i] *= sigmoid(gate[i])` for the first `len` elements.
    fn sigmoid_mul(&self, x: &mut Self::Buffer, gate: &Self::Buffer, len: usize) -> Result<()>;
    /// Causal depthwise convolution + SiLU with a rolling state; see [`ConvShape`].
    fn conv1d_silu(
        &self,
        x: &Self::Buffer,
        weight: &Self::Buffer,
        state: &mut Self::Buffer,
        out: &mut Self::Buffer,
        shape: ConvShape,
    ) -> Result<()>;
    /// Gated DeltaNet gating, L2 norm and recurrence over `tokens` steps; see [`GdnShape`].
    #[allow(clippy::too_many_arguments)]
    fn gated_delta_net(
        &self,
        qkv: &Self::Buffer,
        ab: &Self::Buffer,
        a_log: &Self::Buffer,
        dt_bias: &Self::Buffer,
        state: &mut Self::Buffer,
        out: &mut Self::Buffer,
        shape: GdnShape,
    ) -> Result<()>;

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
