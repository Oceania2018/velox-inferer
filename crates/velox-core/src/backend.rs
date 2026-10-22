//! The device op contract every backend implements (CPU, CUDA, and later HIP).
//!
//! Layouts are row-major f32. Implementations must validate shapes with [`crate::check`] so that
//! every backend rejects the same malformed calls. Buffers may be longer than an op needs, which
//! lets a model reuse workspaces sized for its largest batch.

use std::fmt;

use crate::error::{Error, Result};
use crate::format::WeightFormat;
use crate::kv::KvFormat;
use crate::op::Op;
use crate::profile::DeviceProfile;

/// [`Backend::alloc_host`] callable from any thread: pinning system RAM is slow (about 1 ms per
/// MiB), so a background thread can prepare it while the device works.
pub type HostAlloc<T> = std::sync::Arc<dyn Fn(usize) -> Result<T> + Send + Sync>;

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
/// `[capacity, kv_heads]` caches of rows stored in format `kv` whose first `pos0 + tokens`
/// positions are valid. Query head `h` reads KV head `h / (heads / kv_heads)`. Token `t` attends to
/// keys `0 ..= pos0 + t`. The result is exact softmax attention over the rows the cache represents
/// (see [`crate::kv`]; for `F32` the stored values themselves).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttnShape {
    pub tokens: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub pos0: usize,
    pub kv: KvFormat,
}

impl AttnShape {
    pub fn kv_len(&self) -> usize {
        self.pos0 + self.tokens
    }

    pub fn scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }
}

/// Stores `tokens` rows of K and V (`[tokens, kv_heads, head_dim]` f32, from row 0) into caches of
/// format `format` at positions `pos0 .. pos0 + tokens`, encoded as [`crate::kv::encode_row`]
/// specifies (bit-exact). Other cache rows are left untouched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvStoreShape {
    pub tokens: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub pos0: usize,
    pub format: KvFormat,
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
/// Only rows `row0 .. row0 + tokens` of `x` and `out` are read and written, so a chunk can be
/// processed in segments (recurrent state snapshots inside one prefill chunk).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvShape {
    pub tokens: usize,
    pub channels: usize,
    pub width: usize,
    pub row0: usize,
}

/// Gated DeltaNet over `tokens` steps.
///
/// `qkv` rows are `[q (k_heads·k_dim) | k (k_heads·k_dim) | v (v_heads·v_dim)]`; `ab` rows are
/// `[a (v_heads) | b (v_heads)]`. Value head `h` uses key head `h / (v_heads / k_heads)`.
/// `state` is `[v_heads, v_dim, k_dim]` f32, updated in place; `out` is `[tokens, v_heads, v_dim]`.
/// Per step: `g = -exp(a_log)·softplus(a + dt_bias)`, `β = sigmoid(b)`, q and k L2-normalised
/// (`x / sqrt(Σx² + 1e-6)`), `S = e^g·S`, `S += (β(v − S k)) ⊗ k`, `o = S q / sqrt(k_dim)`.
/// Only rows `row0 .. row0 + tokens` of `qkv`, `ab` and `out` are read and written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnShape {
    pub tokens: usize,
    pub k_heads: usize,
    pub v_heads: usize,
    pub k_dim: usize,
    pub v_dim: usize,
    pub row0: usize,
}

/// One copy of [`Backend::copy_batch`]:
/// `dsts[dst][dst_offset ..][.. len] = src[src_offset ..][.. len]`.
pub struct CopySpec<'a, Buf> {
    pub src: &'a Buf,
    pub src_offset: usize,
    pub dst: usize,
    pub dst_offset: usize,
    pub len: usize,
}

/// One Gated DeltaNet layer of [`Backend::gdn_replay`].
pub struct GdnReplayLayer<'a, Buf> {
    /// `[tokens, channels]` conv1d inputs of the window.
    pub qkv: &'a Buf,
    /// `[tokens, channels]` conv1d_silu outputs of the window (from the same starting state).
    pub conv_out: &'a Buf,
    /// `[tokens, 2 * v_heads]` gate inputs.
    pub ab: &'a Buf,
    pub a_log: &'a Buf,
    pub dt_bias: &'a Buf,
    /// `[width - 1, channels]` conv state, advanced in place.
    pub conv_state: &'a mut Buf,
    /// `[v_heads, v_dim, k_dim]` recurrent state, advanced in place.
    pub ssm: &'a mut Buf,
}

/// [`Backend::graph`] key flag: the recorded work is static (see there).
pub const GRAPH_STATIC: u64 = 1 << 63;

/// Static graph steps that attend over a KV cache are recorded per bucket of this many keys: the
/// backend sizes attention grids for the bucket and reads the actual length on the device.
pub const GRAPH_KV_BUCKET: usize = 1024;

/// Non-causal attention of a query block over a sliding-window context (DFlash drafting).
///
/// `q` is `[tokens, heads, head_dim]` for positions `pos0 .. pos0 + tokens`. Context keys live in
/// ring caches `[ring, kv_heads, head_dim]`, position `p` at row `p % ring`, valid for
/// `ctx_lo <= p < pos0` (`pos0 - ctx_lo <= ring`). The block's own keys are `[tokens, kv_heads,
/// head_dim]`. Token `i` attends to context positions `p` with `pos0 + i - p < window` and to every
/// block row (`tokens <= window`). Query head `h` reads KV head `h / (heads / kv_heads)`; scores are
/// scaled by `1 / sqrt(head_dim)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockAttnShape {
    pub tokens: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub pos0: usize,
    pub ctx_lo: usize,
    pub window: usize,
    pub ring: usize,
}

impl BlockAttnShape {
    /// First context position token `i` sees.
    pub fn ctx_start(&self, i: usize) -> usize {
        (self.pos0 + i + 1)
            .saturating_sub(self.window)
            .max(self.ctx_lo)
    }
}

/// Dynamic grouped convolution of DFlash2, side `side` (0: branch input, 1: branch output), over
/// `rows` consecutive positions: blocks of `block` rows (several sequences' draft blocks; 0: one
/// block of all rows), each convolved on its own. With `G = channels / group`, `base` is `[2 sides,
/// 2 taps, channels]` and `delta` is `[rows, 4 G]` (column `(2 side + tap) G + g`). For channel
/// `h` of group `g = h / group`:
///
/// `y[i][h] = (base[side][0][h] + delta[i][2 side G + g]) x[i][h]
///          + [i mod block > 0] (base[side][1][h] + delta[i][(2 side + 1) G + g]) x[i - 1][h]`
///
/// and `out = y`, or `out += y` with `accumulate`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DynConvShape {
    pub rows: usize,
    pub channels: usize,
    pub group: usize,
    pub side: usize,
    pub accumulate: bool,
    pub block: usize,
}

impl DynConvShape {
    /// Rows per independent block.
    pub fn block_rows(&self) -> usize {
        if self.block == 0 {
            self.rows.max(1)
        } else {
            self.block
        }
    }
}

/// A strided 2-D copy: `dst[dst_offset + r dst_stride + c] = src[src_offset + r src_stride + c]`
/// for `r < rows`, `c < cols`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Copy2dShape {
    pub rows: usize,
    pub cols: usize,
    pub src_offset: usize,
    pub src_stride: usize,
    pub dst_offset: usize,
    pub dst_stride: usize,
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
    /// Rounded to bf16 (RNE): the source model's own activation precision, so every policy
    /// (including `A16Only`) permits it.
    Bf16,
    /// Per row: `scale = amax / 448`, codes `e4m3(x * (1 / scale))` (RNE, saturating).
    Fp8Row,
    /// Per 16 elements: `s = e4m3(divisor * amax / 6)`, codes `e2m1(x * divisor / s)`.
    Nvfp4 {
        divisor: f32,
    },
}

/// The operand precision of an attention kernel (`attention`, `attention_full`). Like
/// [`ActQuant`], the CPU oracle applies the same input rounding before its exact computation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnPrecision {
    Exact,
    /// Q and K rounded per row (one `head_dim` vector) as [`ActQuant::Fp8Row`] for the score
    /// product; V rounded to f16 and the softmax weights to f16 for the value product.
    Fp8QkF16Pv,
    /// Q, K, V and the softmax weights rounded to f16 for both products.
    F16,
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
    /// Physical device memory that virtual buffers map (see [`Self::alloc_virtual`]).
    type Chunk;

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
    /// A SwiGLU FFN's projections: `out[r][i] = silu((x_r . gate_i)) * (x_r . up_i)` for `m` rows
    /// of `x`, where `gate` and `up` are views of the same width, format and K. Backends fuse it
    /// where they can; the default ([`linear_swiglu_composed`]) runs one GEMM into `scratch`
    /// (`[m, 2 * width]`) when the views are adjacent rows of one parent, else two, then
    /// `swiglu` / `silu_mul`.
    #[allow(clippy::too_many_arguments)]
    fn linear_swiglu(
        &self,
        x: &Self::Buffer,
        gate: &WeightView<'_, Self::Buffer>,
        up: &WeightView<'_, Self::Buffer>,
        out: &mut Self::Buffer,
        m: usize,
        scratch: &mut Self::Buffer,
    ) -> Result<()> {
        linear_swiglu_composed(self, x, gate, up, out, m, scratch)
    }
    /// The activation rounding `linear_swiglu` applies at `m` rows (that of the projection the
    /// default runs).
    fn linear_swiglu_act(
        &self,
        gate: &WeightView<'_, Self::Buffer>,
        up: &WeightView<'_, Self::Buffer>,
        m: usize,
    ) -> ActQuant {
        match swiglu_pair(gate, up) {
            Some((both, _, _)) => self.linear_act(&both, m),
            None => self.linear_act(gate, m),
        }
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
    /// Residual add then RMS norm: `acc[..rows * dim] += x`, then `y = rms_norm(acc)` (see
    /// [`Self::rms_norm`]). One pass where a backend fuses it.
    #[allow(clippy::too_many_arguments)]
    fn add_rms_norm(
        &self,
        acc: &mut Self::Buffer,
        x: &Self::Buffer,
        weight: &Self::Buffer,
        y: &mut Self::Buffer,
        rows: usize,
        dim: usize,
        eps: f32,
        kind: NormKind,
    ) -> Result<()> {
        self.add(acc, x, rows * dim)?;
        self.rms_norm(acc, weight, y, rows, dim, eps, kind)
    }
    /// Gated RMS norm: `gate[i] = silu(gate[i]) * rms_norm(x)[i]` over `rows * dim` elements (see
    /// [`Self::rms_norm`]). One pass where a backend fuses it.
    #[allow(clippy::too_many_arguments)]
    fn rms_norm_gated(
        &self,
        x: &Self::Buffer,
        weight: &Self::Buffer,
        gate: &mut Self::Buffer,
        rows: usize,
        dim: usize,
        eps: f32,
        kind: NormKind,
    ) -> Result<()> {
        let mut y = self.alloc(rows * dim)?;
        self.rms_norm(x, weight, &mut y, rows, dim, eps, kind)?;
        self.silu_mul(gate, &y, rows * dim)
    }
    /// Zeroed storage in system RAM that kernels read and write directly (pinned and mapped):
    /// no device memory, slower access. Backends without such memory return device storage.
    fn alloc_host(&self, len: usize) -> Result<Self::Buffer> {
        self.alloc(len)
    }
    /// f32 slots of one physical chunk of virtual memory; 0 when the backend has none (callers
    /// then allocate whole buffers).
    fn vm_chunk_len(&self) -> usize {
        0
    }
    /// A buffer of `len` f32 slots (rounded up to whole chunks) whose address range is reserved
    /// but backed by nothing: chunk `i` (slots `i * vm_chunk_len()` onward) must be mapped
    /// ([`Self::map_chunk`]) before an op touches it. The address never moves, so recorded
    /// graphs stay valid across remaps.
    fn alloc_virtual(&self, _len: usize) -> Result<Self::Buffer> {
        Err(crate::Error::Unsupported("virtual memory".into()))
    }
    /// One chunk of physical memory (contents undefined).
    fn alloc_chunk(&self) -> Result<Self::Chunk> {
        Err(crate::Error::Unsupported("virtual memory".into()))
    }
    /// Backs chunk `index` of the virtual buffer `buf` with `chunk`, which may back other
    /// buffers' chunks at the same time (they then share contents). The index must be unmapped.
    fn map_chunk(
        &self,
        _buf: &mut Self::Buffer,
        _index: usize,
        _chunk: &Self::Chunk,
    ) -> Result<()> {
        Err(crate::Error::Unsupported("virtual memory".into()))
    }
    /// Removes chunk `index`'s backing. Queued work touching it must have completed
    /// ([`Self::synchronize`]).
    fn unmap_chunk(&self, _buf: &mut Self::Buffer, _index: usize) -> Result<()> {
        Err(crate::Error::Unsupported("virtual memory".into()))
    }
    /// [`Self::alloc_host`] for another thread, when the backend has pinned memory (else `None`).
    fn host_alloc(&self) -> Option<HostAlloc<Self::Buffer>> {
        None
    }
    /// [`Self::upload_bytes`] into storage from [`Self::alloc_host`].
    fn upload_bytes_host(&self, data: &[u8]) -> Result<Self::Buffer> {
        self.upload_bytes(data)
    }
    /// The bytes of `buf` when the host can address them directly (storage from
    /// [`Self::alloc_host`] on a backend with such memory), else `None`. Callers synchronize
    /// with pending device work on `buf` first.
    fn host_bytes<'a>(&self, buf: &'a Self::Buffer) -> Option<&'a [u8]> {
        let _ = buf;
        None
    }
    /// Mutable [`Self::host_bytes`].
    fn host_bytes_mut<'a>(&self, buf: &'a mut Self::Buffer) -> Option<&'a mut [u8]> {
        let _ = buf;
        None
    }
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
    /// [`Self::conv1d_silu`] advancing from `state_in` into `state_out` (distinct buffers), leaving
    /// `state_in` as it was: speculative windows keep the state they may roll back to.
    fn conv1d_silu_from(
        &self,
        x: &Self::Buffer,
        weight: &Self::Buffer,
        state_in: &Self::Buffer,
        state_out: &mut Self::Buffer,
        out: &mut Self::Buffer,
        shape: ConvShape,
    ) -> Result<()> {
        self.copy(state_in, 0, state_out, 0, self.len(state_in))?;
        self.conv1d_silu(x, weight, state_out, out, shape)
    }
    /// [`Self::gated_delta_net`] updating from `state_in` into `state_out` (distinct buffers),
    /// leaving `state_in` as it was.
    #[allow(clippy::too_many_arguments)]
    fn gated_delta_net_from(
        &self,
        qkv: &Self::Buffer,
        ab: &Self::Buffer,
        a_log: &Self::Buffer,
        dt_bias: &Self::Buffer,
        state_in: &Self::Buffer,
        state_out: &mut Self::Buffer,
        out: &mut Self::Buffer,
        shape: GdnShape,
    ) -> Result<()> {
        self.copy(state_in, 0, state_out, 0, self.len(state_in))?;
        self.gated_delta_net(qkv, ab, a_log, dt_bias, state_out, out, shape)
    }
    /// Advances several GDN layers over the first `gdn.tokens` rows of a window whose conv outputs
    /// are known (a speculative window replayed after a partial accept): each layer's conv state
    /// as [`Self::conv1d_silu`] and its recurrent state as [`Self::gated_delta_net`] would leave
    /// them; nothing else is written. `conv.tokens == gdn.tokens` and `conv.row0 == gdn.row0`: the
    /// window's rows start at `row0` of `qkv`, `conv_out` and `ab` (several windows can share
    /// those buffers).
    fn gdn_replay(
        &self,
        layers: &mut [GdnReplayLayer<'_, Self::Buffer>],
        conv: ConvShape,
        gdn: GdnShape,
    ) -> Result<()> {
        let (ch, w1) = (conv.channels, conv.width - 1);
        let mut rows = self.alloc(w1 * ch)?;
        let mut out = self.alloc((gdn.row0 + gdn.tokens) * gdn.v_heads * gdn.v_dim)?;
        for l in layers.iter_mut() {
            crate::check::gdn_replay_layer(
                self.len(l.qkv),
                self.len(l.conv_out),
                self.len(l.ab),
                self.len(l.a_log),
                self.len(l.dt_bias),
                self.len(l.conv_state),
                self.len(l.ssm),
                &conv,
                &gdn,
            )?;
            // New state row j is row tokens + j of [old state; inputs].
            for j in 0..w1 {
                let r = conv.tokens + j;
                if r < w1 {
                    self.copy(l.conv_state, r * ch, &mut rows, j * ch, ch)?;
                } else {
                    self.copy(l.qkv, (conv.row0 + r - w1) * ch, &mut rows, j * ch, ch)?;
                }
            }
            self.copy(&rows, 0, l.conv_state, 0, w1 * ch)?;
            self.gated_delta_net(l.conv_out, l.ab, l.a_log, l.dt_bias, l.ssm, &mut out, gdn)?;
        }
        Ok(())
    }
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
    /// Several independent copies into `dsts` (written ranges must not overlap each other or a
    /// source), in stream order with other work but in any order among themselves: one submission
    /// for many small copies, such as a checkpoint's per-layer buffers going to system RAM.
    fn copy_batch(
        &self,
        dsts: &mut [&mut Self::Buffer],
        ops: &[CopySpec<'_, Self::Buffer>],
    ) -> Result<()> {
        for op in ops {
            let dst = dsts.get_mut(op.dst).ok_or_else(|| {
                crate::Error::Shape(format!("copy_batch: no destination {}", op.dst))
            })?;
            self.copy(op.src, op.src_offset, dst, op.dst_offset, op.len)?;
        }
        Ok(())
    }
    /// Writes `[tokens, heads, head_dim]` into `out`; see [`AttnShape`].
    fn attention(
        &self,
        q: &Self::Buffer,
        k: &Self::Buffer,
        v: &Self::Buffer,
        out: &mut Self::Buffer,
        shape: AttnShape,
    ) -> Result<()>;
    /// Encodes K/V rows into caches; see [`KvStoreShape`].
    fn kv_store(
        &self,
        k: &Self::Buffer,
        v: &Self::Buffer,
        k_cache: &mut Self::Buffer,
        v_cache: &mut Self::Buffer,
        shape: KvStoreShape,
    ) -> Result<()>;
    /// Decodes cache rows `pos0 .. pos0 + tokens` (see [`KvStoreShape`]) into the f32 rows they
    /// represent, `[tokens, kv_heads, head_dim]` each: lets f32 kernels (long prefill chunks) run
    /// over a quantized cache.
    fn kv_load(
        &self,
        k_cache: &Self::Buffer,
        v_cache: &Self::Buffer,
        k: &mut Self::Buffer,
        v: &mut Self::Buffer,
        shape: KvStoreShape,
    ) -> Result<()>;
    /// The operand precision `attention` uses for this shape. For a quantized cache the rounding
    /// applies to the rows the cache represents.
    fn attention_precision(&self, _shape: AttnShape) -> AttnPrecision {
        AttnPrecision::Exact
    }
    /// The operand precision `attention_full` uses for this shape.
    fn attention_full_precision(&self, _shape: FullAttnShape) -> AttnPrecision {
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
    /// Column blocks of `src` (`[rows, src_cols]`) into separate buffers: `out.0[r] =
    /// src[r][out.1 .. out.1 + out.2]` for each `(buffer, column offset, columns)`.
    fn split_cols(
        &self,
        src: &Self::Buffer,
        rows: usize,
        src_cols: usize,
        outs: &mut [(&mut Self::Buffer, usize, usize)],
    ) -> Result<()>;
    /// SwiGLU over a fused gate|up projection: `out[r][i] = silu(src[r][gate + i]) * src[r][up + i]`
    /// for `i < width`; `src` is `[rows, src_cols]`, `out` is `[rows, width]`.
    #[allow(clippy::too_many_arguments)]
    fn swiglu(
        &self,
        src: &Self::Buffer,
        rows: usize,
        src_cols: usize,
        gate: usize,
        up: usize,
        width: usize,
        out: &mut Self::Buffer,
    ) -> Result<()>;
    /// `out[r] = a[r] ++ b[r]` for `rows` rows: `a` is `[rows, a_cols]`, `b` is `[rows, b_cols]`,
    /// `out` is `[rows, a_cols + b_cols]`.
    fn concat_cols(
        &self,
        a: &Self::Buffer,
        b: &Self::Buffer,
        out: &mut Self::Buffer,
        rows: usize,
        a_cols: usize,
        b_cols: usize,
    ) -> Result<()>;
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
    /// Runs `f`, which must only queue device work: no host readback and no synchronization. A
    /// backend may record the work once per `key` and replay it as one submission (CUDA graphs);
    /// work under one key must keep the same structure (the same ops in the same order) for
    /// replays to stay cheap. `f` runs exactly once either way, so its host-side effects happen
    /// once.
    ///
    /// With [`GRAPH_STATIC`] set in `key`, the caller promises more: every run under the key
    /// issues the same ops on the same buffers with the same shapes, differing only in the
    /// per-call values ops take from the host (token ids, positions, copy offsets). A backend
    /// may then skip recording and launch the recorded work with those values refreshed.
    ///
    /// Per-call values under a static key also include the KV caches of `kv_store` and
    /// `attention` and the recurrent states of `conv1d_silu*` and `gated_delta_net*`: one
    /// recording serves any sequence (a batch of sequences is one step whatever sequences it holds).
    /// A static key carries a KV bucket in bits 24..40 (keys = bucket x [`GRAPH_KV_BUCKET`]); every
    /// attention call of the step covers at most that many keys, and grids are sized for it.
    fn graph(&self, _key: u64, f: &mut dyn FnMut() -> Result<()>) -> Result<()> {
        f()
    }
    /// Inside [`Self::graph`]: ends the recording here and continues in a new one, launched after
    /// it. A long step split this way starts executing while the rest is still being submitted.
    /// A no-op elsewhere.
    fn graph_break(&self) -> Result<()> {
        Ok(())
    }
    /// Strided 2-D copy between (or within non-overlapping parts of) buffers; see [`Copy2dShape`].
    fn copy2d(&self, src: &Self::Buffer, dst: &mut Self::Buffer, shape: Copy2dShape) -> Result<()>;
    /// DFlash2 dynamic grouped convolution; see [`DynConvShape`]. `out` must not alias `x`.
    fn dyn_conv(
        &self,
        x: &Self::Buffer,
        base: &Self::Buffer,
        delta: &Self::Buffer,
        out: &mut Self::Buffer,
        shape: DynConvShape,
    ) -> Result<()>;
    /// Writes `[tokens, heads, head_dim]` into `out`; see [`BlockAttnShape`].
    #[allow(clippy::too_many_arguments)]
    fn attention_block(
        &self,
        q: &Self::Buffer,
        k_ctx: &Self::Buffer,
        v_ctx: &Self::Buffer,
        k_blk: &Self::Buffer,
        v_blk: &Self::Buffer,
        out: &mut Self::Buffer,
        shape: BlockAttnShape,
    ) -> Result<()>;
}

/// Host reference for [`Backend::top_k`] on one row: the `k` largest of `row`, descending, ties to
/// the lower index, NaN below every number and reported as -inf.
/// The gate and up views as one view of adjacent rows of one parent, with the gate and up column
/// offsets of its output, when they are.
pub fn swiglu_pair<'a, Buf>(
    gate: &WeightView<'a, Buf>,
    up: &WeightView<'a, Buf>,
) -> Option<(WeightView<'a, Buf>, usize, usize)> {
    let w = gate.rows;
    let same = std::ptr::eq(gate.buf, up.buf)
        && gate.format == up.format
        && gate.parent_rows == up.parent_rows
        && gate.k == up.k
        && up.rows == w;
    let act = gate.act.meet(up.act);
    if same && up.row_begin == gate.row_begin + w {
        Some((
            WeightView {
                rows: 2 * w,
                act,
                ..*gate
            },
            0,
            w,
        ))
    } else if same && gate.row_begin == up.row_begin + w {
        Some((
            WeightView {
                rows: 2 * w,
                act,
                ..*up
            },
            w,
            0,
        ))
    } else {
        None
    }
}

/// [`Backend::linear_swiglu`] from `linear` and `swiglu` / `silu_mul`.
#[allow(clippy::too_many_arguments)]
pub fn linear_swiglu_composed<B: Backend + ?Sized>(
    b: &B,
    x: &B::Buffer,
    gate: &WeightView<'_, B::Buffer>,
    up: &WeightView<'_, B::Buffer>,
    out: &mut B::Buffer,
    m: usize,
    scratch: &mut B::Buffer,
) -> Result<()> {
    let w = gate.rows;
    if up.rows != w || up.k != gate.k {
        return Err(Error::Shape(format!(
            "linear_swiglu: gate [{w},{}] and up [{},{}] differ",
            gate.k, up.rows, up.k
        )));
    }
    match swiglu_pair(gate, up) {
        Some((both, g, u)) => {
            b.linear(x, &both, scratch, m)?;
            b.swiglu(scratch, m, 2 * w, g, u, w, out)
        }
        None => {
            b.linear(x, gate, out, m)?;
            b.linear(x, up, scratch, m)?;
            b.silu_mul(out, scratch, m * w)
        }
    }
}

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
