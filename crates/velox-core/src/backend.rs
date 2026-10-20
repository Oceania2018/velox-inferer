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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FullAttnShape {
    pub tokens: usize,
    pub heads: usize,
    pub head_dim: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeluKind {
    /// `0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3)))`
    Tanh,
    /// `0.5x(1 + erf(x / sqrt 2))`
    Erf,
}

/// Which position axis and frequency each rotary pair uses, for `R = rotary_dim` and
/// `P = R / 2` pairs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeLayout {
    /// Pair `i` uses axis `i % axes` and frequency `theta^(-2i/R)` (Qwen MRoPE).
    Interleaved,
    /// Pairs split into `axes` equal blocks; pair `i` uses axis `i / (P/axes)` and frequency
    /// `theta^(-2(i % (P/axes)) / (R/axes))` (Qwen vision 2D RoPE).
    Blocked,
}

/// Rotate-half rotary over the first `rotary_dim` dims of each head of `[tokens, heads, head_dim]`,
/// pair `(i, i + R/2)` rotated by `positions[axis(i)][t] * freq(i)`; see [`RopeLayout`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopePosShape {
    pub tokens: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub theta: f32,
    pub axes: usize,
    pub layout: RopeLayout,
}

impl RopePosShape {
    /// (axis, inverse frequency) of rotary pair `i`.
    pub fn pair(&self, i: usize) -> (usize, f64) {
        let r = self.rotary_dim as f64;
        match self.layout {
            RopeLayout::Interleaved => (
                i % self.axes,
                f64::from(self.theta).powf(-2.0 * i as f64 / r),
            ),
            RopeLayout::Blocked => {
                let per = self.rotary_dim / 2 / self.axes;
                (
                    i / per,
                    f64::from(self.theta).powf(-2.0 * (i % per) as f64 / (r / self.axes as f64)),
                )
            }
        }
    }
}

impl GdnShape {
    pub fn qkv_width(&self) -> usize {
        2 * self.k_heads * self.k_dim + self.v_heads * self.v_dim
    }
}

/// The lowest activation precision a weight's use permits (artifact `uses[].activation_policy`).
/// The model author decides this per projection; a kernel may round activations no further.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum ActPolicy {
    /// Activations stay exact (f32 here, BF16 in the source model).
    #[default]
    A16Only,
    /// Per-token E4M3 activations are allowed.
    AllowA8,
    /// NVFP4 activations are allowed with this static input divisor.
    AllowA4 { divisor: f32 },
}

impl ActPolicy {
    /// The stricter of two policies, for a view that serves several uses.
    pub fn meet(self, other: ActPolicy) -> ActPolicy {
        use ActPolicy::*;
        match (self, other) {
            (A16Only, _) | (_, A16Only) => A16Only,
            (AllowA4 { divisor: a }, AllowA4 { divisor: b }) if a == b => self,
            _ => AllowA8,
        }
    }
}

/// How a `linear` kernel rounds its input activations before the product. The CPU oracle applies
/// the same rounding ([`crate::quant::quantize_activations`]), so low-precision kernels are still
/// checked against an exact reference.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ActQuant {
    Exact,
    /// Per row: `scale = amax / 448`, codes `e4m3(x * (1 / scale))` (RNE, saturating).
    Fp8Row,
    /// Per 16 elements: `s = e4m3(divisor * amax / 6)`, codes `e2m1(x * divisor / s)`.
    Nvfp4 {
        divisor: f32,
    },
}

/// The operand precision of a causal `attention` kernel. Like [`ActQuant`], the CPU oracle applies
/// the same input rounding before its exact computation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnPrecision {
    Exact,
    /// Q and K rounded per row (one `head_dim` vector) as [`ActQuant::Fp8Row`] for the score
    /// product; V rounded to f16 and the softmax weights to f16 for the value product.
    Fp8QkF16Pv,
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
    pub act: ActPolicy,
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
            act: ActPolicy::A16Only,
        }
    }

    pub fn with_act(self, act: ActPolicy) -> Self {
        WeightView { act, ..self }
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
    /// Makes this backend's device current on the calling thread. Servers call it before using
    /// a backend from a thread other than the one that created it (CUDA devices are per-thread).
    fn bind_thread(&self) -> Result<()> {
        Ok(())
    }

    /// `y[m, rows] = x[m, k] · Wᵀ` with `W` the view's rows decoded from their stored format.
    fn linear(
        &self,
        x: &Self::Buffer,
        w: &WeightView<'_, Self::Buffer>,
        y: &mut Self::Buffer,
        m: usize,
    ) -> Result<()>;
    /// The activation rounding `linear` applies to this view at `m` rows: `Exact`, or a rounding
    /// that `w.act` permits.
    fn linear_act(&self, _w: &WeightView<'_, Self::Buffer>, _m: usize) -> ActQuant {
        ActQuant::Exact
    }
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
    /// `dst[dst_offset ..][.. len] = src[src_offset ..][.. len]`.
    fn copy(
        &self,
        src: &Self::Buffer,
        src_offset: usize,
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
    /// The operand precision `attention` uses for this shape.
    fn attention_precision(&self, _shape: AttnShape) -> AttnPrecision {
        AttnPrecision::Exact
    }
    /// Non-causal multi-head attention: every token attends to all `shape.tokens` keys.
    /// `q`, `k`, `v`, `out` are `[tokens, heads, head_dim]`.
    fn attention_full(
        &self,
        q: &Self::Buffer,
        k: &Self::Buffer,
        v: &Self::Buffer,
        out: &mut Self::Buffer,
        shape: FullAttnShape,
    ) -> Result<()>;
    /// `y = (x - mean) / sqrt(var + eps) * w + b` per row (population variance).
    #[allow(clippy::too_many_arguments)]
    fn layer_norm(
        &self,
        x: &Self::Buffer,
        weight: &Self::Buffer,
        bias: &Self::Buffer,
        y: &mut Self::Buffer,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<()>;
    /// `x[r, c] += bias[c]` for `rows` rows of width `dim`.
    fn add_bias(
        &self,
        x: &mut Self::Buffer,
        bias: &Self::Buffer,
        rows: usize,
        dim: usize,
    ) -> Result<()>;
    /// `x[i] = gelu(x[i])` for the first `len` elements.
    fn gelu(&self, x: &mut Self::Buffer, len: usize, kind: GeluKind) -> Result<()>;
    /// The `k` largest of the first `limit` values of each row of `x` (`[rows, row_len]`), returned
    /// to the host in descending order, ties to the lower index, as `(index, value)`. NaN ranks
    /// below every number and is reported as -inf. Rows with fewer than `k` values return all of them.
    fn top_k(
        &self,
        x: &Self::Buffer,
        rows: usize,
        row_len: usize,
        limit: usize,
        k: usize,
    ) -> Result<Vec<Vec<(u32, f32)>>>;
    /// Rotary embedding with explicit per-axis positions (`[axes, tokens]`, host); see
    /// [`RopePosShape`].
    fn rope_positions(
        &self,
        x: &mut Self::Buffer,
        positions: &[u32],
        shape: RopePosShape,
    ) -> Result<()>;
}

/// Host reference for [`Backend::top_k`] on one row: the `k` largest of `row`, descending, ties to
/// the lower index, NaN below every number and reported as -inf.
pub fn top_k_row(row: &[f32], k: usize) -> Vec<(u32, f32)> {
    let key = |v: f32| if v.is_nan() { f32::NEG_INFINITY } else { v };
    let mut idx: Vec<u32> = (0..row.len() as u32).collect();
    let k = k.min(row.len());
    let better = |a: &u32, b: &u32| {
        key(row[*b as usize])
            .total_cmp(&key(row[*a as usize]))
            .then(a.cmp(b))
    };
    if k < idx.len() {
        idx.select_nth_unstable_by(k, better);
        idx.truncate(k);
    }
    idx.sort_unstable_by(better);
    idx.into_iter().map(|i| (i, key(row[i as usize]))).collect()
}
