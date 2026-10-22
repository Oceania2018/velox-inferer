//! Qwen3.5 / Qwen3.6 / Qwen3.8 text model (`Qwen3_5ForCausalLM` Dense, `Qwen3_5MoeForCausalLM` sparse
//! MoE): Gated DeltaNet + gated attention hybrid.
//! The mathematics is specified in docs/qwen3_5-text.md.

mod config;
mod dflash;
mod mtp;
mod paging;
mod vision;
mod weights;

use velox_core::backend::{
    AttnShape, Backend, ConvShape, Copy2dShape, GdnReplayLayer, GdnShape, NormKind, RopeLayout,
    RopePosShape, RopeShape, WeightView,
};
use velox_core::{CopySpec, Error, KvFormat, KvStoreShape, Result, GRAPH_KV_BUCKET, GRAPH_STATIC};

pub use config::{
    FfnConfig, LayerKind, MoeConfig, TextConfig, DENSE_ARCHITECTURE, MOE_ARCHITECTURE,
};
pub use dflash::{
    BlockCandidates, DFlashConfig, DFlashModel, DFlashState, DFlashWorkspace, DraftContext,
    DRAFT_PAGE_ROWS,
};
pub use mtp::{MtpNext, MtpRope, MtpSession, MtpWorkspace};
pub use paging::{KvChunkSet, KvShare};
pub use vision::{Qwen35Vision, VisionConfig};
pub use weights::{DenseFfn, ExpertBank, Ffn, Layer, Mixer, MoeFfn, Runs, TextWeights, WDesc};

/// Per-sequence state: attention KV caches, GDN convolution and recurrent states, position.
pub struct Session<B: Backend> {
    pub pos: usize,
    pub capacity: usize,
    /// Rotary position minus cache position for tokens after the prompt. MRoPE compresses each
    /// image run, so this is negative once a prompt contains an image.
    pub rope_delta: i64,
    /// Storage format of the attention KV caches.
    pub kv_format: KvFormat,
    kv: Vec<(B::Buffer, B::Buffer)>,
    /// f32 slots of one cache position (all KV heads) in each attention layer's K or V cache.
    kv_row: usize,
    conv: Vec<B::Buffer>,
    ssm: Vec<B::Buffer>,
    /// Speculative-verify state, allocated on first `verify`.
    spec: Option<Spec<B>>,
    /// Distinguishes this session's graph recordings from other sessions' (they bind its buffers).
    id: u64,
    /// Flips whenever a committed verify swaps the recurrent state with the speculative buffers.
    parity: bool,
    /// Virtual-memory backing of the caches (see `paging`); `None`: allocated whole.
    paging: Option<paging::KvPaging<B>>,
}

/// Up to this many tokens (decode, verify) adjacent projections run as one fused linear; larger
/// chunks keep them separate, where per-matrix GEMM tiles are faster than the split copy saves.
const FUSE_MAX_TOKENS: usize = 16;

/// `y[t, runs.rows()] = h . W^T` for a projection stored as `runs`. With several runs, each is
/// computed into `scratch` (at least `t x` its rows) and copied to its columns of `y`.
fn linear_runs<'w, B: Backend>(
    b: &B,
    view: impl Fn(&WDesc) -> WeightView<'w, B::Buffer>,
    h: &B::Buffer,
    runs: &Runs,
    y: &mut B::Buffer,
    scratch: &mut B::Buffer,
    t: usize,
) -> Result<()>
where
    B::Buffer: 'w,
{
    if let Some(d) = runs.single() {
        return b.linear(h, &view(d), y, t);
    }
    let (total, mut col) = (runs.rows(), 0);
    for d in &runs.0 {
        b.linear(h, &view(d), scratch, t)?;
        b.copy2d(
            scratch,
            y,
            Copy2dShape {
                rows: t,
                cols: d.rows,
                src_offset: 0,
                src_stride: d.rows,
                dst_offset: col,
                dst_stride: total,
            },
        )?;
        col += d.rows;
    }
    Ok(())
}

/// Most tokens one `verify` call may feed (the anchor plus drafts).
pub const VERIFY_MAX: usize = 16;

/// Graph keys (`Backend::graph`) of the model's repeated steps: bits 0..16 carry the shape, 16..24
/// the kind, 24..40 the KV bucket (static steps attend over a bucket-sized grid) and 40..56 the
/// session.
const GRAPH_ADVANCE: u64 = 1 << 16;
const GRAPH_VERIFY: u64 = 2 << 16;
const GRAPH_COMMIT: u64 = 3 << 16;
const GRAPH_VERIFY_BATCH: u64 = 4 << 16;

/// Static steps (see [`GRAPH_STATIC`]) stay under this many tokens, where attention takes the
/// path that stages its position.
const STATIC_MAX_TOKENS: usize = 15;

/// Layers where a recorded step starts a new graph segment (see [`Backend::graph_break`]).
/// Submission time grows with a segment's size and the GPU waits for the first one, so it holds
/// one layer; each later segment is submitted while the GPU runs the ones before it.
const GRAPH_SEGMENT_STARTS: [usize; 5] = [1, 4, 16, 32, 48];

fn graph_key(kind: u64, shape: u64, s: &Session<impl Backend>, kv_len: Option<usize>) -> u64 {
    let mut key = kind | shape | u64::from(s.parity) << 15 | (s.id & 0xffff) << 40;
    if let Some(kv) = kv_len {
        key |= GRAPH_STATIC | (kv.div_ceil(GRAPH_KV_BUCKET) as u64 & 0xffff) << 24;
    }
    key
}

static NEXT_SESSION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// What `commit` needs to keep only a prefix of a verified window: the recurrent state before
/// the window, and per GDN layer the raw projections each token fed to the convolution and the
/// recurrence. Replaying a prefix runs the same kernels on the same inputs, so the committed
/// state is bit-identical to the verify trajectory at that point.
struct Spec<B: Backend> {
    conv: Vec<B::Buffer>,
    ssm: Vec<B::Buffer>,
    pos0: usize,
    tokens: usize,
    recording: bool,
    /// Set by [`Qwen35Text::verify_batch`]: the window's replay inputs are rows `row0 ..` of the
    /// batch workspace rather than the workspace's [`SpecRows`].
    batch_row0: Option<usize>,
}

/// Per GDN layer: `[VERIFY_MAX, gdn_qkv]` raw qkv and conv outputs and `[VERIFY_MAX, 2 * v_heads]`
/// a/b rows of the latest verify window (what commit replays a partially accepted window from).
/// One set in the workspace serves every session: a commit follows its verify.
struct SpecRows<B: Backend> {
    qkv: Vec<B::Buffer>,
    conv_out: Vec<B::Buffer>,
    ab: Vec<B::Buffer>,
}

/// Buffers for [`Qwen35Text::verify_batch`]: several sequences' windows (at most
/// [`VERIFY_MAX`] rows each) share one forward pass, so every weight is read once per round.
pub struct BatchWorkspace<B: Backend> {
    max_seqs: usize,
    /// Per GDN layer, `[max_seqs * VERIFY_MAX, *]` rows: what each sequence's commit replays.
    qkv: Vec<B::Buffer>,
    conv_out: Vec<B::Buffer>,
    ab: Vec<B::Buffer>,
    /// One sequence's attention rows (queries, keys, values, output), staged for the per-sequence
    /// cache.
    q: B::Buffer,
    k: B::Buffer,
    v: B::Buffer,
    attn: B::Buffer,
    /// Final-normed rows and their logits.
    hidden: B::Buffer,
    logits: B::Buffer,
    /// GDN outputs of the verify rows when a prefill chunk occupies the workspace's first rows.
    o: B::Buffer,
}

/// The prefill part of a [`Qwen35Text::mixed_step`]: one chunk of a prompt, fed to `session`
/// from its current position.
pub struct PrefillChunk<'a, 'b, B: Backend> {
    pub session: &'a mut Session<B>,
    pub ids: &'a [u32],
    /// Explicit per-token MRoPE positions of the chunk, else cache position + rope_delta.
    pub positions: Option<&'a [[u32; 3]]>,
    /// Embedding rows replacing tokens of the chunk (relative to its first token).
    pub inserts: &'a [EmbedInsert<'b, B::Buffer>],
    /// Each is filled with the recurrent state at its `pos` when that lies inside the chunk.
    pub snaps: &'a mut [Checkpoint<B>],
    /// Whether the candidates after the chunk's last token are wanted (the prompt's last chunk).
    pub candidates: bool,
}

/// The prefill rows of a mixed step inside the forward pass.
struct PrefillRows<'a, B: Backend> {
    s: &'a mut Session<B>,
    t: usize,
    axis_positions: Option<Vec<u32>>,
    rope_pos0: usize,
    snaps: &'a mut [Checkpoint<B>],
}

impl<B: Backend> Workspace<B> {
    /// Most rows (tokens) one forward pass takes.
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }
}

impl<B: Backend> BatchWorkspace<B> {
    /// Most sequences one batch takes.
    pub fn max_seqs(&self) -> usize {
        self.max_seqs
    }
}

/// The recurrent (GDN) state of a session after `pos` tokens. Attention KV needs no copy: rows
/// below `pos` are never rewritten, so restoring a checkpoint only rewinds the position and the
/// recurrent state.
pub struct Checkpoint<B: Backend> {
    pub pos: usize,
    pub rope_delta: i64,
    conv: Vec<B::Buffer>,
    ssm: Vec<B::Buffer>,
    host: bool,
}

impl<B: Backend> Checkpoint<B> {
    /// Whether the storage is in system RAM ([`Backend::alloc_host`]).
    pub fn on_host(&self) -> bool {
        self.host
    }

    /// Copies this checkpoint into `dst` (storage of the same session shape).
    pub fn copy_into(&self, b: &B, dst: &mut Checkpoint<B>) -> Result<()> {
        copy_all(
            b,
            self.conv.iter().chain(&self.ssm),
            dst.conv.iter_mut().chain(dst.ssm.iter_mut()),
        )?;
        dst.pos = self.pos;
        dst.rope_delta = self.rope_delta;
        Ok(())
    }
}

/// Cache positions per [`KvSegment`] page.
pub const KV_PAGE_ROWS: usize = 256;

/// Attention KV cache rows `lo..hi` of every attention layer, kept outside a session so the
/// prefix they belong to can be put back into a session whose cache now holds other tokens.
/// Stored in pages of [`KV_PAGE_ROWS`] positions, each one buffer of
/// [`Session::kv_page_len`] slots holding every layer's K then V rows, so the storage of any
/// segment can be recycled through a pool of equal pages.
pub struct KvSegment<B: Backend> {
    pub lo: usize,
    pub hi: usize,
    pages: Vec<B::Buffer>,
}

impl<B: Backend> KvSegment<B> {
    /// Storage size in bytes.
    pub fn bytes(&self, b: &B) -> usize {
        self.pages.iter().map(|x| b.len(x) * 4).sum()
    }

    /// Storage buffers in a fixed order (for serialization).
    pub fn buffers(&self) -> impl Iterator<Item = &B::Buffer> {
        self.pages.iter()
    }

    /// Mutable [`Self::buffers`], same order.
    pub fn buffers_mut(&mut self) -> impl Iterator<Item = &mut B::Buffer> {
        self.pages.iter_mut()
    }

    /// The pages, for reuse.
    pub fn into_pages(self) -> Vec<B::Buffer> {
        self.pages
    }
}

impl<B: Backend> Checkpoint<B> {
    /// Storage size in bytes.
    pub fn bytes(&self, b: &B) -> usize {
        self.conv
            .iter()
            .chain(&self.ssm)
            .map(|x| b.len(x) * 4)
            .sum()
    }

    /// Storage buffers in a fixed order (for serialization).
    pub fn buffers(&self) -> impl Iterator<Item = &B::Buffer> {
        self.conv.iter().chain(&self.ssm)
    }

    /// Mutable [`Self::buffers`], same order.
    pub fn buffers_mut(&mut self) -> impl Iterator<Item = &mut B::Buffer> {
        self.conv.iter_mut().chain(self.ssm.iter_mut())
    }
}

impl<B: Backend> Session<B> {
    /// f32 slots of one [`KvSegment`] page of this session's shape.
    pub fn kv_page_len(&self) -> usize {
        2 * self.kv.len() * KV_PAGE_ROWS * self.kv_row
    }

    /// Storage for attention KV rows `lo..hi`, with pages of [`Self::kv_page_len`] slots from
    /// `page` (a pool, or [`Backend::alloc_host`]); filled by [`Self::save_kv`] or through
    /// [`KvSegment::buffers_mut`].
    pub fn blank_kv(
        &self,
        b: &B,
        lo: usize,
        hi: usize,
        page: &mut dyn FnMut() -> Result<B::Buffer>,
    ) -> Result<KvSegment<B>> {
        if lo > hi || hi > self.capacity {
            return Err(Error::Config(format!(
                "KV rows {lo}..{hi} outside session capacity {}",
                self.capacity
            )));
        }
        let mut pages = Vec::with_capacity((hi - lo).div_ceil(KV_PAGE_ROWS));
        for _ in 0..(hi - lo).div_ceil(KV_PAGE_ROWS) {
            let p = page()?;
            if b.len(&p) < self.kv_page_len() {
                return Err(Error::Config("KV page smaller than kv_page_len".into()));
            }
            pages.push(p);
        }
        Ok(KvSegment { lo, hi, pages })
    }

    /// Copies attention KV rows `lo..hi` out of the session into pages from `page`.
    pub fn save_kv(
        &self,
        b: &B,
        lo: usize,
        hi: usize,
        page: &mut dyn FnMut() -> Result<B::Buffer>,
    ) -> Result<KvSegment<B>> {
        let mut seg = self.blank_kv(b, lo, hi, page)?;
        let row = self.kv_row;
        // One submission: every layer's K and V rows of every page.
        let mut ops = Vec::with_capacity(seg.pages.len() * 2 * self.kv.len());
        for p in 0..seg.pages.len() {
            let r0 = lo + p * KV_PAGE_ROWS;
            let n = (hi - r0).min(KV_PAGE_ROWS) * row;
            for (l, (kc, vc)) in self.kv.iter().enumerate() {
                for (side, src) in [kc, vc].into_iter().enumerate() {
                    ops.push(CopySpec {
                        src,
                        src_offset: r0 * row,
                        dst: p,
                        dst_offset: (2 * l + side) * KV_PAGE_ROWS * row,
                        len: n,
                    });
                }
            }
        }
        let mut dsts: Vec<&mut B::Buffer> = seg.pages.iter_mut().collect();
        b.copy_batch(&mut dsts, &ops)?;
        Ok(seg)
    }

    /// Device copy of the recurrent state at the current position.
    pub fn checkpoint(&self, b: &B) -> Result<Checkpoint<B>> {
        self.checkpoint_in(b, false)
    }

    /// Copy of the recurrent state at the current position, in system RAM when `host`.
    pub fn checkpoint_in(&self, b: &B, host: bool) -> Result<Checkpoint<B>> {
        let mut cp = self.blank_checkpoint_in(b, self.pos, host)?;
        self.snapshot_into(b, &mut cp)?;
        Ok(cp)
    }

    /// Checkpoint storage for this session's state, to be filled at `pos` by an advance (see
    /// [`Qwen35Text::advance_rows`]) or by [`Self::snapshot_into`].
    pub fn blank_checkpoint(&self, b: &B, pos: usize) -> Result<Checkpoint<B>> {
        self.blank_checkpoint_in(b, pos, false)
    }

    /// [`Self::blank_checkpoint`] in system RAM ([`Backend::alloc_host`]) when `host`: no device
    /// memory per checkpoint, at the cost of PCIe copies when one is taken or restored.
    /// Lengths of a checkpoint's buffers (see [`Self::host_checkpoint_from`]).
    pub fn state_lens(&self, b: &B) -> Vec<usize> {
        self.conv
            .iter()
            .chain(&self.ssm)
            .map(|s| b.len(s))
            .collect()
    }

    /// A system-RAM checkpoint over `bufs` (from [`Backend::alloc_host`], one per
    /// [`Self::state_lens`] entry).
    pub fn host_checkpoint_from(&self, bufs: Vec<B::Buffer>) -> Result<Checkpoint<B>> {
        if bufs.len() != self.conv.len() + self.ssm.len() {
            return Err(Error::Shape(format!(
                "checkpoint of {} buffers (state has {})",
                bufs.len(),
                self.conv.len() + self.ssm.len()
            )));
        }
        let mut conv = bufs;
        let ssm = conv.split_off(self.conv.len());
        Ok(Checkpoint {
            pos: 0,
            rope_delta: self.rope_delta,
            conv,
            ssm,
            host: true,
        })
    }

    pub fn blank_checkpoint_in(&self, b: &B, pos: usize, host: bool) -> Result<Checkpoint<B>> {
        let like = |bufs: &[B::Buffer]| -> Result<Vec<B::Buffer>> {
            bufs.iter()
                .map(|s| {
                    if host {
                        b.alloc_host(b.len(s))
                    } else {
                        b.alloc(b.len(s))
                    }
                })
                .collect()
        };
        Ok(Checkpoint {
            pos,
            rope_delta: self.rope_delta,
            conv: like(&self.conv)?,
            ssm: like(&self.ssm)?,
            host,
        })
    }

    /// Copies the current recurrent state into `cp` (reused storage) and stamps its position.
    pub fn snapshot_into(&self, b: &B, cp: &mut Checkpoint<B>) -> Result<()> {
        copy_all(
            b,
            self.conv.iter().chain(&self.ssm),
            cp.conv.iter_mut().chain(cp.ssm.iter_mut()),
        )?;
        cp.pos = self.pos;
        cp.rope_delta = self.rope_delta;
        Ok(())
    }

    /// Rewinds to `cp`, which must have been taken from this session (or a session with the
    /// same token prefix up to `cp.pos`).
    pub fn restore(&mut self, b: &B, cp: &Checkpoint<B>) -> Result<()> {
        if cp.pos > self.capacity {
            return Err(Error::Config("checkpoint beyond session capacity".into()));
        }
        copy_all(
            b,
            cp.conv.iter().chain(&cp.ssm),
            self.conv.iter_mut().chain(self.ssm.iter_mut()),
        )?;
        self.pos = cp.pos;
        self.rope_delta = cp.rope_delta;
        Ok(())
    }
}

/// Copies each of `srcs` whole into the matching buffer of `dsts`, in one submission.
fn copy_all<'a, B: Backend + 'a>(
    b: &B,
    srcs: impl Iterator<Item = &'a B::Buffer>,
    dsts: impl Iterator<Item = &'a mut B::Buffer>,
) -> Result<()> {
    let ops: Vec<CopySpec<'_, B::Buffer>> = srcs
        .enumerate()
        .map(|(i, src)| CopySpec {
            src,
            src_offset: 0,
            dst: i,
            dst_offset: 0,
            len: b.len(src),
        })
        .collect();
    let mut dsts: Vec<&mut B::Buffer> = dsts.collect();
    if dsts.len() != ops.len() {
        return Err(Error::Shape("copy_all: buffer counts differ".into()));
    }
    b.copy_batch(&mut dsts, &ops)
}

/// Embedding rows `buf[row_offset .. row_offset + rows]` (`[*, hidden]`) that replace the token
/// embeddings at tokens `token .. token + rows` of a `forward_with` call (image placeholders).
pub struct EmbedInsert<'a, Buf> {
    pub token: usize,
    pub rows: usize,
    pub buf: &'a Buf,
    pub row_offset: usize,
}

/// Activations for chunks of up to `max_tokens` tokens.
pub struct Workspace<B: Backend> {
    max_tokens: usize,
    x: B::Buffer,
    h: B::Buffer,
    y: B::Buffer,
    qkv: B::Buffer,
    conv_out: B::Buffer,
    z: B::Buffer,
    ab: B::Buffer,
    o: B::Buffer,
    q: B::Buffer,
    g: B::Buffer,
    k: B::Buffer,
    v: B::Buffer,
    qn: B::Buffer,
    kn: B::Buffer,
    attn: B::Buffer,
    mg: B::Buffer,
    mu: B::Buffer,
    /// Fused projection outputs before the split (GDN qkv|z, attention q|k|gate|v) and the fused
    /// MLP gate|up output.
    fused: B::Buffer,
    gu: B::Buffer,
    last: B::Buffer,
    last_n: B::Buffer,
    /// Final-normed hidden rows of the latest `verify` (`[VERIFY_MAX, hidden]`), which condition
    /// draft models.
    pub hidden: B::Buffer,
    logits_rows: B::Buffer,
    logits: B::Buffer,
    /// When set, each chunk leaves its composed input embeddings in `emb` and its final-normed
    /// hidden rows in `rows_hidden` (draft-model prefill reads them through `advance_rows`).
    pub keep_rows: bool,
    pub emb: B::Buffer,
    pub rows_hidden: B::Buffer,
    /// Row of the last token of the latest `advance`, inside the final chunk.
    last_row: usize,
    /// Target layers whose residual outputs (after the FFN) each chunk copies into `features`
    /// (`[max_tokens, capture.len() * hidden]`, layer-major columns): DFlash conditioning.
    capture: Vec<usize>,
    features: Option<B::Buffer>,
    /// f32 copies of a quantized cache's visible rows for long chunks (grown on demand), so they
    /// attend with the f32 prefill kernels instead of decoding packed rows per query.
    kv_f32: Option<(B::Buffer, B::Buffer)>,
    /// Verify replay rows (allocated with the first verify).
    spec_rows: Option<SpecRows<B>>,
    /// `emb` / `rows_hidden` hold `max_tokens` rows (MTP drafting reads them); without, they are
    /// placeholders and `keep_rows` does not fill them.
    rows: bool,
}

pub struct Qwen35Text<B: Backend> {
    pub cfg: TextConfig,
    pub w: TextWeights<B>,
}

impl<B: Backend> Qwen35Text<B> {
    pub fn new(cfg: TextConfig, w: TextWeights<B>) -> Self {
        Qwen35Text { cfg, w }
    }

    pub fn session(&self, b: &B, capacity: usize) -> Result<Session<B>> {
        self.session_with(b, capacity, KvFormat::F32)
    }

    /// A session whose attention KV caches store rows in `kv_format`.
    pub fn session_with(&self, b: &B, capacity: usize, kv_format: KvFormat) -> Result<Session<B>> {
        self.session_in(b, capacity, kv_format, false)
    }

    /// [`Self::session_with`] whose attention caches are paged when the backend has virtual
    /// memory: backed as the sequence grows, prefix chunks shareable between sessions (see
    /// [`KvShare`]).
    pub fn session_paged(&self, b: &B, capacity: usize, kv_format: KvFormat) -> Result<Session<B>> {
        self.session_in(b, capacity, kv_format, b.vm_chunk_len() > 0)
    }

    fn session_in(
        &self,
        b: &B,
        capacity: usize,
        kv_format: KvFormat,
        paged: bool,
    ) -> Result<Session<B>> {
        let c = &self.cfg;
        kv_format.check_head_dim(c.head_dim)?;
        let kv_len = kv_format
            .cache_slots(capacity, c.kv_heads, c.head_dim)
            .ok_or_else(|| Error::Config("KV cache size overflows".into()))?;
        if capacity == 0 || capacity > c.max_positions {
            return Err(Error::Config(format!(
                "session capacity {capacity} outside 1..={}",
                c.max_positions
            )));
        }
        let mut kv = Vec::new();
        let mut conv = Vec::new();
        let mut ssm = Vec::new();
        for kind in &c.layers {
            match kind {
                LayerKind::FullAttention if paged => {
                    kv.push((b.alloc_virtual(kv_len)?, b.alloc_virtual(kv_len)?))
                }
                LayerKind::FullAttention => kv.push((b.alloc(kv_len)?, b.alloc(kv_len)?)),
                LayerKind::Gdn => {
                    conv.push(b.alloc((c.conv_width - 1) * c.gdn_qkv())?);
                    ssm.push(b.alloc(c.gdn_v_heads * c.gdn_v_dim * c.gdn_k_dim)?);
                }
            }
        }
        Ok(Session {
            pos: 0,
            capacity,
            rope_delta: 0,
            kv_format,
            kv,
            kv_row: c.kv_heads * kv_format.row_slots(c.head_dim),
            conv,
            ssm,
            spec: None,
            id: NEXT_SESSION.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            parity: false,
            paging: paged.then(|| {
                let chunk = b.vm_chunk_len();
                paging::KvPaging::new(chunk, kv_len.div_ceil(chunk))
            }),
        })
    }

    pub fn workspace(&self, b: &B, max_tokens: usize) -> Result<Workspace<B>> {
        self.workspace_with(b, max_tokens, true)
    }

    /// [`Self::workspace`]; `rows`: whether chunks can keep their embeddings and final hidden rows
    /// (`Workspace::keep_rows`, which MTP drafting needs and DFlash does not).
    pub fn workspace_with(&self, b: &B, max_tokens: usize, rows: bool) -> Result<Workspace<B>> {
        let c = &self.cfg;
        let t = max_tokens;
        // Fused projections and the fused MLP gate|up only run up to FUSE_MAX_TOKENS rows.
        let tf = t.min(FUSE_MAX_TOKENS);
        let tr = if rows { t } else { 1 };
        Ok(Workspace {
            max_tokens,
            x: b.alloc(t * c.hidden)?,
            h: b.alloc(t * c.hidden)?,
            y: b.alloc(t * c.hidden)?,
            qkv: b.alloc(t * c.gdn_qkv())?,
            conv_out: b.alloc(t * c.gdn_qkv())?,
            z: b.alloc(t * c.gdn_v())?,
            ab: b.alloc(t * 2 * c.gdn_v_heads)?,
            o: b.alloc(t * c.gdn_v())?,
            q: b.alloc(t * c.q_dim())?,
            g: b.alloc(t * c.q_dim())?,
            k: b.alloc(t * c.kv_dim())?,
            v: b.alloc(t * c.kv_dim())?,
            qn: b.alloc(t * c.q_dim())?,
            kn: b.alloc(t * c.kv_dim())?,
            attn: b.alloc(t * c.q_dim())?,
            // The dense SwiGLU (a MoE block's shared expert); routed experts will need their own
            // scratch once the MoE ops land.
            mg: b.alloc(t * c.mlp_width())?,
            mu: b.alloc(t * c.mlp_width())?,
            fused: b.alloc(tf * (c.gdn_qkv() + c.gdn_v()).max(2 * c.q_dim() + 2 * c.kv_dim()))?,
            gu: b.alloc(tf * 2 * c.mlp_width())?,
            last: b.alloc(c.hidden)?,
            last_n: b.alloc(c.hidden)?,
            logits: b.alloc(c.vocab)?,
            hidden: b.alloc(VERIFY_MAX * c.hidden)?,
            logits_rows: b.alloc(VERIFY_MAX * c.vocab)?,
            last_row: 0,
            keep_rows: false,
            emb: b.alloc(tr * c.hidden)?,
            rows_hidden: b.alloc(tr * c.hidden)?,
            capture: Vec::new(),
            kv_f32: None,
            features: None,
            spec_rows: None,
            rows,
        })
    }

    /// Makes every chunk of `ws` keep the residual outputs of target `layers` (block indices) in
    /// its feature buffer, for a draft model conditioned on them.
    pub fn enable_capture(&self, b: &B, ws: &mut Workspace<B>, layers: &[usize]) -> Result<()> {
        if let Some(&bad) = layers.iter().find(|&&l| l >= self.cfg.layers.len()) {
            return Err(Error::Config(format!("capture layer {bad} out of range")));
        }
        ws.features = Some(b.alloc(ws.max_tokens * layers.len() * self.cfg.hidden)?);
        ws.capture = layers.to_vec();
        Ok(())
    }

    /// Feeds `ids` (any length, processed in workspace-sized chunks) and returns the logits after
    /// the last one. Rotary positions are `cache position + session.rope_delta`.
    pub fn forward(
        &self,
        b: &B,
        s: &mut Session<B>,
        ws: &mut Workspace<B>,
        ids: &[u32],
    ) -> Result<Vec<f32>> {
        self.forward_with(b, s, ws, ids, None, &[])
    }

    /// `forward` with explicit MRoPE positions (one (t, h, w) triple per id) and embedding rows
    /// that replace image placeholder embeddings.
    pub fn forward_with(
        &self,
        b: &B,
        s: &mut Session<B>,
        ws: &mut Workspace<B>,
        ids: &[u32],
        positions: Option<&[[u32; 3]]>,
        inserts: &[EmbedInsert<'_, B::Buffer>],
    ) -> Result<Vec<f32>> {
        self.advance(b, s, ws, ids, positions, inserts)?;
        self.logits(b, ws)
    }

    /// Feeds `ids` like [`Self::forward_with`] without producing logits; follow with
    /// [`Self::logits`] or [`Self::candidates`] when the next token is needed.
    pub fn advance(
        &self,
        b: &B,
        s: &mut Session<B>,
        ws: &mut Workspace<B>,
        ids: &[u32],
        positions: Option<&[[u32; 3]]>,
        inserts: &[EmbedInsert<'_, B::Buffer>],
    ) -> Result<()> {
        self.advance_snap(b, s, ws, ids, positions, inserts, &mut [])
    }

    /// [`Self::advance`] that also fills each of `snaps` with the recurrent state at its `pos`
    /// (inside `s.pos ..= s.pos + ids.len()`), without splitting the chunk's projections there.
    #[allow(clippy::too_many_arguments)]
    pub fn advance_snap(
        &self,
        b: &B,
        s: &mut Session<B>,
        ws: &mut Workspace<B>,
        ids: &[u32],
        positions: Option<&[[u32; 3]]>,
        inserts: &[EmbedInsert<'_, B::Buffer>],
        snaps: &mut [Checkpoint<B>],
    ) -> Result<()> {
        if positions.is_some_and(|p| p.len() != ids.len()) {
            return Err(Error::Shape(
                "one position triple per token is required".into(),
            ));
        }
        if ids.is_empty() {
            return Err(Error::Config("forward needs at least one token".into()));
        }
        if s.pos + ids.len() > s.capacity {
            return Err(Error::Config(format!(
                "{} + {} tokens exceed session capacity {}",
                s.pos,
                ids.len(),
                s.capacity
            )));
        }
        let t = ids.len();
        s.prepare_kv(b, s.pos, s.pos + t)?;
        if t <= FUSE_MAX_TOKENS
            && t <= ws.max_tokens
            && positions.is_none()
            && inserts.is_empty()
            && snaps.is_empty()
        {
            // Decode and verify steps repeat with the same structure: one recorded graph per shape.
            let recording = s.spec.as_ref().is_some_and(|sp| sp.recording);
            let shape = t as u64
                | u64::from(recording) << 8
                | u64::from(!ws.capture.is_empty()) << 9
                | u64::from(ws.keep_rows) << 10;
            let kv = (t <= STATIC_MAX_TOKENS).then_some(s.pos + t);
            let key = graph_key(GRAPH_ADVANCE, shape, s, kv);
            b.graph(key, &mut || {
                self.chunk(b, s, ws, ids, 0, None, &[], &mut [])
            })?;
            ws.last_row = t - 1;
            return Ok(());
        }
        for cp in snaps.iter_mut().filter(|cp| cp.pos == s.pos) {
            s.snapshot_into(b, cp)?;
        }
        for (i, chunk) in ids.chunks(ws.max_tokens).enumerate() {
            let base = i * ws.max_tokens;
            let pos = positions.map(|p| &p[base..base + chunk.len()]);
            self.chunk(b, s, ws, chunk, base, pos, inserts, snaps)?;
            ws.last_row = chunk.len() - 1;
        }
        Ok(())
    }

    /// One chunk of `ids`, which start at offset `base` of the `forward_with` call.
    #[allow(clippy::too_many_arguments)]
    fn chunk(
        &self,
        b: &B,
        s: &mut Session<B>,
        ws: &mut Workspace<B>,
        ids: &[u32],
        base: usize,
        positions: Option<&[[u32; 3]]>,
        inserts: &[EmbedInsert<'_, B::Buffer>],
        snaps: &mut [Checkpoint<B>],
    ) -> Result<()> {
        let c = &self.cfg;
        let w = &self.w;
        let t = ids.len();
        let pos0 = s.pos;
        // GDN segments end at each snapshot inside this chunk, then at its end.
        let mut cuts: Vec<usize> = snaps
            .iter()
            .filter(|cp| cp.pos > pos0 && cp.pos <= pos0 + t)
            .map(|cp| cp.pos - pos0)
            .chain(std::iter::once(t))
            .collect();
        cuts.sort_unstable();
        cuts.dedup();
        let hd = c.hidden;
        b.embed(ids, &w.view(&w.embed), &mut ws.x)?;
        for ins in inserts {
            let (lo, hi) = (ins.token.max(base), (ins.token + ins.rows).min(base + t));
            if lo < hi {
                let src = (ins.row_offset + lo - ins.token) * hd;
                b.copy(ins.buf, src, &mut ws.x, (lo - base) * hd, (hi - lo) * hd)?;
            }
        }
        if ws.keep_rows && ws.rows {
            b.copy(&ws.x, 0, &mut ws.emb, 0, t * hd)?;
        }
        // Axis-major [3, t] positions for explicit MRoPE, else cache position + rope_delta.
        let axis_positions: Option<Vec<u32>> = positions.map(|p| {
            (0..3)
                .flat_map(|a| p.iter().map(move |tri| tri[a]))
                .collect()
        });
        let rope_pos0 = match axis_positions {
            Some(_) => 0, // unused: explicit positions
            None => usize::try_from(pos0 as i64 + s.rope_delta)
                .map_err(|_| Error::Shape("negative rotary position".into()))?,
        };
        let (mut ai, mut gi) = (0, 0);
        for (li, l) in w.layers.iter().enumerate() {
            if GRAPH_SEGMENT_STARTS.contains(&li) {
                // Recorded steps launch in segments, so the GPU starts while the rest is submitted.
                b.graph_break()?;
            }
            if li == 0 {
                b.rms_norm(
                    &ws.x,
                    w.param(l.input_norm),
                    &mut ws.h,
                    t,
                    hd,
                    c.eps,
                    NormKind::Offset,
                )?;
            } else {
                // The previous layer's MLP residual, fused with this layer's input norm.
                b.add_rms_norm(
                    &mut ws.x,
                    &ws.y,
                    w.param(l.input_norm),
                    &mut ws.h,
                    t,
                    hd,
                    c.eps,
                    NormKind::Offset,
                )?;
                self.capture_features(b, ws, li - 1, t)?;
            }
            match &l.mixer {
                Mixer::Gdn {
                    qkv,
                    z,
                    qkvz,
                    ab,
                    out,
                    conv,
                    a_log,
                    dt_bias,
                    norm,
                } => {
                    // While a verify window is recorded, its qkv and a/b rows land directly in the
                    // spec buffers commit replays from, and the recurrent state advances into the
                    // spec state, so the session keeps the state to roll back to (no copies).
                    let Session {
                        spec,
                        conv: s_conv,
                        ssm: s_ssm,
                        ..
                    } = &mut *s;
                    let (qkv_buf, ab_buf, mut state_out) = match (
                        spec.as_mut().filter(|sp| sp.recording),
                        ws.spec_rows.as_mut(),
                    ) {
                        (
                            Some(Spec {
                                conv: sc, ssm: ss, ..
                            }),
                            Some(SpecRows {
                                qkv: sq,
                                conv_out: so,
                                ab: sa,
                            }),
                        ) => (
                            &mut sq[gi],
                            &mut sa[gi],
                            Some((&mut sc[gi], &mut ss[gi], &mut so[gi])),
                        ),
                        _ => (&mut ws.qkv, &mut ws.ab, None),
                    };
                    match qkvz.as_ref().filter(|_| t <= FUSE_MAX_TOKENS) {
                        Some(f) => {
                            b.linear(&ws.h, &w.view(f), &mut ws.fused, t)?;
                            b.split_cols(
                                &ws.fused,
                                t,
                                f.rows,
                                &mut [
                                    (&mut *qkv_buf, qkv.0[0].row_begin - f.row_begin, qkv.rows()),
                                    (&mut ws.z, z.row_begin - f.row_begin, z.rows),
                                ],
                            )?;
                        }
                        None => {
                            // conv_out ([t, gdn_qkv]) is only written after the projections.
                            linear_runs(
                                b,
                                |d| w.view(d),
                                &ws.h,
                                qkv,
                                qkv_buf,
                                &mut ws.conv_out,
                                t,
                            )?;
                            b.linear(&ws.h, &w.view(z), &mut ws.z, t)?;
                        }
                    }
                    linear_runs(b, |d| w.view(d), &ws.h, ab, ab_buf, &mut ws.conv_out, t)?;
                    let mut row0 = 0;
                    for &end in &cuts {
                        let cs = ConvShape {
                            tokens: end - row0,
                            channels: c.gdn_qkv(),
                            width: c.conv_width,
                            row0,
                        };
                        let gs = GdnShape {
                            tokens: end - row0,
                            k_heads: c.gdn_k_heads,
                            v_heads: c.gdn_v_heads,
                            k_dim: c.gdn_k_dim,
                            v_dim: c.gdn_v_dim,
                            row0,
                        };
                        let (cw, al, dt) = (w.param(*conv), w.param(*a_log), w.param(*dt_bias));
                        match &mut state_out {
                            Some((sc, ss, so)) => {
                                b.conv1d_silu_from(qkv_buf, cw, &s_conv[gi], sc, so, cs)?;
                                b.gated_delta_net_from(
                                    so, ab_buf, al, dt, &s_ssm[gi], ss, &mut ws.o, gs,
                                )?;
                            }
                            None => {
                                b.conv1d_silu(qkv_buf, cw, &mut s_conv[gi], &mut ws.conv_out, cs)?;
                                b.gated_delta_net(
                                    &ws.conv_out,
                                    ab_buf,
                                    al,
                                    dt,
                                    &mut s_ssm[gi],
                                    &mut ws.o,
                                    gs,
                                )?;
                            }
                        }
                        for cp in snaps.iter_mut().filter(|cp| cp.pos == pos0 + end) {
                            b.copy(&s_conv[gi], 0, &mut cp.conv[gi], 0, b.len(&s_conv[gi]))?;
                            b.copy(&s_ssm[gi], 0, &mut cp.ssm[gi], 0, b.len(&s_ssm[gi]))?;
                        }
                        row0 = end;
                    }
                    // z = silu(z) * plain_rmsnorm(o)
                    b.rms_norm_gated(
                        &ws.o,
                        w.param(*norm),
                        &mut ws.z,
                        t * c.gdn_v_heads,
                        c.gdn_v_dim,
                        c.eps,
                        NormKind::Plain,
                    )?;
                    b.linear(&ws.z, &w.view(out), &mut ws.y, t)?;
                    gi += 1;
                }
                Mixer::Attention { .. } => {
                    let rope = match &axis_positions {
                        Some(p) => Rope::Axes(p),
                        None => Rope::At(rope_pos0),
                    };
                    let kv_format = s.kv_format;
                    self.attention_mixer(b, ws, &l.mixer, &mut s.kv[ai], kv_format, t, pos0, rope)?;
                    ai += 1;
                }
            }
            b.add_rms_norm(
                &mut ws.x,
                &ws.y,
                w.param(l.post_norm),
                &mut ws.h,
                t,
                hd,
                c.eps,
                NormKind::Offset,
            )?;
            self.mlp(b, ws, l, t)?;
            // The MLP residual is added by the next layer's fused input norm.
        }
        b.add(&mut ws.x, &ws.y, t * hd)?;
        self.capture_features(b, ws, w.layers.len() - 1, t)?;
        if ws.keep_rows && ws.rows {
            b.rms_norm(
                &ws.x,
                w.param(w.final_norm),
                &mut ws.rows_hidden,
                t,
                hd,
                c.eps,
                NormKind::Offset,
            )?;
        }
        s.pos += t;
        Ok(())
    }

    /// Next-token logits after the last `advance`d token (downloaded: the whole vocabulary).
    pub fn logits(&self, b: &B, ws: &mut Workspace<B>) -> Result<Vec<f32>> {
        self.head(b, ws)?;
        let mut out = vec![0.0; self.cfg.vocab];
        b.synchronize()?;
        b.download(&ws.logits, &mut out)?;
        Ok(out)
    }

    /// The `k` best next-token candidates among the first `limit` ids, selected on the device:
    /// only `k` (id, logit) pairs cross to the host.
    pub fn candidates(
        &self,
        b: &B,
        ws: &mut Workspace<B>,
        k: usize,
        limit: usize,
    ) -> Result<Vec<(u32, f32)>> {
        self.head(b, ws)?;
        let v = self.cfg.vocab;
        Ok(b.top_k(&ws.logits, 1, v, limit.min(v), k)?
            .pop()
            .unwrap_or_default())
    }

    /// Gated attention mixer of one layer: `ws.h` (normed input, `t` rows) to `ws.y`, appending
    /// K/V at cache rows `pos0..` of `kv`.
    #[allow(clippy::too_many_arguments)]
    fn attention_mixer(
        &self,
        b: &B,
        ws: &mut Workspace<B>,
        mixer: &Mixer,
        kv: &mut (B::Buffer, B::Buffer),
        kv_format: KvFormat,
        t: usize,
        pos0: usize,
        rope: Rope<'_>,
    ) -> Result<()> {
        let (c, w) = (&self.cfg, &self.w);
        let Mixer::Attention {
            query,
            gate,
            key,
            value,
            fused,
            output,
            q_norm,
            k_norm,
        } = mixer
        else {
            return Err(Error::Config("attention_mixer on a GDN layer".into()));
        };
        match fused.as_ref().filter(|_| t <= FUSE_MAX_TOKENS) {
            Some(f) => {
                b.linear(&ws.h, &w.view(f), &mut ws.fused, t)?;
                let col = |d: &WDesc| d.row_begin - f.row_begin;
                b.split_cols(
                    &ws.fused,
                    t,
                    f.rows,
                    &mut [
                        (&mut ws.q, col(query), query.rows),
                        (&mut ws.g, col(gate), gate.rows),
                        (&mut ws.k, col(key), key.rows),
                        (&mut ws.v, col(value), value.rows),
                    ],
                )?;
            }
            None => {
                b.linear(&ws.h, &w.view(query), &mut ws.q, t)?;
                b.linear(&ws.h, &w.view(gate), &mut ws.g, t)?;
                b.linear(&ws.h, &w.view(key), &mut ws.k, t)?;
                b.linear(&ws.h, &w.view(value), &mut ws.v, t)?;
            }
        }
        b.rms_norm(
            &ws.q,
            w.param(*q_norm),
            &mut ws.qn,
            t * c.heads,
            c.head_dim,
            c.eps,
            NormKind::Offset,
        )?;
        b.rms_norm(
            &ws.k,
            w.param(*k_norm),
            &mut ws.kn,
            t * c.kv_heads,
            c.head_dim,
            c.eps,
            NormKind::Offset,
        )?;
        self.attend_rows(b, ws, kv, kv_format, t, pos0, rope)?;
        b.sigmoid_mul(&mut ws.attn, &ws.g, t * c.q_dim())?;
        b.linear(&ws.attn, &w.view(output), &mut ws.y, t)?;
        Ok(())
    }

    /// Rotates rows `0 .. t` of `ws.qn` / `ws.kn`, appends them with `ws.v` to `kv` at cache rows
    /// `pos0 ..` and attends, into rows `0 .. t` of `ws.attn`.
    #[allow(clippy::too_many_arguments)]
    fn attend_rows(
        &self,
        b: &B,
        ws: &mut Workspace<B>,
        kv: &mut (B::Buffer, B::Buffer),
        kv_format: KvFormat,
        t: usize,
        pos0: usize,
        rope: Rope<'_>,
    ) -> Result<()> {
        let c = &self.cfg;
        match rope {
            Rope::Axes(p) => {
                let rope = |heads| RopePosShape {
                    tokens: t,
                    heads,
                    head_dim: c.head_dim,
                    rotary_dim: c.rotary_dim,
                    theta: c.rope_theta,
                    axes: 3,
                    layout: RopeLayout::Interleaved,
                };
                b.rope_positions(&mut ws.qn, p, rope(c.heads))?;
                b.rope_positions(&mut ws.kn, p, rope(c.kv_heads))?;
            }
            Rope::At(rope_pos0) => {
                let rope = |heads| RopeShape {
                    tokens: t,
                    heads,
                    head_dim: c.head_dim,
                    rotary_dim: c.rotary_dim,
                    pos0: rope_pos0,
                    theta: c.rope_theta,
                };
                b.rope(&mut ws.qn, rope(c.heads))?;
                b.rope(&mut ws.kn, rope(c.kv_heads))?;
            }
        }
        let (kc, vc) = kv;
        let store = KvStoreShape {
            tokens: t,
            kv_heads: c.kv_heads,
            head_dim: c.head_dim,
            pos0,
            format: kv_format,
        };
        b.kv_store(&ws.kn, &ws.v, kc, vc, store)?;
        let shape = AttnShape {
            tokens: t,
            heads: c.heads,
            kv_heads: c.kv_heads,
            head_dim: c.head_dim,
            pos0,
            kv: kv_format,
        };
        if kv_format != KvFormat::F32 && t > FUSE_MAX_TOKENS {
            // Long chunk over a quantized cache: decode the visible rows once per layer (a few
            // ms at 16k positions) and run the f32 prefill kernels on them.
            let rows = (pos0 + t) * c.kv_dim();
            if ws.kv_f32.as_ref().is_none_or(|(k, _)| b.len(k) < rows) {
                let cap = rows.next_power_of_two();
                ws.kv_f32 = Some((b.alloc(cap)?, b.alloc(cap)?));
            }
            let (kf, vf) = ws.kv_f32.as_mut().expect("allocated above");
            let all = KvStoreShape {
                tokens: pos0 + t,
                pos0: 0,
                ..store
            };
            b.kv_load(kc, vc, kf, vf, all)?;
            let f32_shape = AttnShape {
                kv: KvFormat::F32,
                ..shape
            };
            b.attention(&ws.qn, kf, vf, &mut ws.attn, f32_shape)?;
        } else {
            b.attention(&ws.qn, kc, vc, &mut ws.attn, shape)?;
        }
        Ok(())
    }

    /// Copies the residual stream after layer `li` into the DFlash features when it is captured.
    fn capture_features(&self, b: &B, ws: &mut Workspace<B>, li: usize, t: usize) -> Result<()> {
        let hd = self.cfg.hidden;
        if let Some(j) = ws.capture.iter().position(|&x| x == li) {
            let cols = ws.capture.len() * hd;
            let f = ws.features.as_mut().expect("allocated with capture");
            b.copy2d(
                &ws.x,
                f,
                Copy2dShape {
                    rows: t,
                    cols: hd,
                    src_offset: 0,
                    src_stride: hd,
                    dst_offset: j * hd,
                    dst_stride: cols,
                },
            )?;
        }
        Ok(())
    }

    /// FFN of one layer: `ws.h` to `ws.y`. Every forward path (chunks, verify, batched verify, MTP)
    /// goes through here, so a new FFN kind only needs an arm in this match.
    fn mlp(&self, b: &B, ws: &mut Workspace<B>, l: &Layer, t: usize) -> Result<()> {
        match &l.ffn {
            Ffn::Dense(d) => self.swiglu_ffn(b, ws, d, t),
            Ffn::Moe(_) => Err(Error::Unsupported(
                "sparse MoE FFN execution (planned ops moe_route / moe_experts, see \
                 docs/qwen3_5-text.md) is not implemented yet"
                    .into(),
            )),
        }
    }

    /// `ws.y = down(silu(gate(ws.h)) * up(ws.h))`.
    fn swiglu_ffn(&self, b: &B, ws: &mut Workspace<B>, d: &DenseFfn, t: usize) -> Result<()> {
        let w = &self.w;
        let width = d.gate.rows;
        match d.gate_up.as_ref().filter(|_| t <= FUSE_MAX_TOKENS) {
            Some(f) => {
                b.linear(&ws.h, &w.view(f), &mut ws.gu, t)?;
                let (g, u) = (d.gate.row_begin - f.row_begin, d.up.row_begin - f.row_begin);
                b.swiglu(&ws.gu, t, f.rows, g, u, width, &mut ws.mg)?;
            }
            None => {
                b.linear(&ws.h, &w.view(&d.gate), &mut ws.mg, t)?;
                b.linear(&ws.h, &w.view(&d.up), &mut ws.mu, t)?;
                b.silu_mul(&mut ws.mg, &ws.mu, t * width)?;
            }
        }
        b.linear(&ws.mg, &w.view(&d.down), &mut ws.y, t)
    }

    /// [`Self::advance`] that also hands every chunk's rows to `on_chunk` (with `ws.keep_rows`
    /// set): composed embeddings in `ws.emb`, final-normed hidden rows in `ws.rows_hidden`, and
    /// the chunk's offset into `ids` and its token count. Each of `snaps` is filled with the recurrent state at its
    /// `pos` (within `s.pos ..= s.pos + ids.len()`; positions outside are left untouched).
    #[allow(clippy::too_many_arguments)]
    pub fn advance_rows(
        &self,
        b: &B,
        s: &mut Session<B>,
        ws: &mut Workspace<B>,
        ids: &[u32],
        positions: Option<&[[u32; 3]]>,
        inserts: &[EmbedInsert<'_, B::Buffer>],
        snaps: &mut [Checkpoint<B>],
        on_chunk: &mut dyn FnMut(&Workspace<B>, usize, usize) -> Result<()>,
    ) -> Result<()> {
        ws.keep_rows = true;
        let mut r = Ok(());
        for (i, chunk) in ids.chunks(ws.max_tokens).enumerate() {
            let base = i * ws.max_tokens;
            let pos = positions.map(|p| &p[base..base + chunk.len()]);
            r = self
                .advance_snap(b, s, ws, chunk, pos, &shift_inserts(inserts, base), snaps)
                .and_then(|()| on_chunk(ws, base, chunk.len()));
            if r.is_err() {
                break;
            }
        }
        ws.keep_rows = false;
        r
    }

    /// Speculative verify: feeds `ids` (the pending token, then drafts; at most [`VERIFY_MAX`] and
    /// one workspace chunk) at rotary positions `cache position + rope_delta`, recording what
    /// [`Self::commit`] needs, and returns the `k` best candidates (ids below `limit`) after each
    /// of them. The final-normed hidden rows stay in `ws.hidden` for draft models.
    #[allow(clippy::too_many_arguments)]
    pub fn verify(
        &self,
        b: &B,
        s: &mut Session<B>,
        ws: &mut Workspace<B>,
        ids: &[u32],
        k: usize,
        limit: usize,
    ) -> Result<Vec<Vec<(u32, f32)>>> {
        let t = ids.len();
        if t == 0 || t > VERIFY_MAX || t > ws.max_tokens {
            return Err(Error::Shape(format!(
                "verify takes 1..={} tokens, got {t}",
                VERIFY_MAX.min(ws.max_tokens)
            )));
        }
        self.ensure_spec(b, s)?;
        self.ensure_spec_rows(b, ws)?;
        if s.pos + t > s.capacity {
            return Err(Error::Config(format!(
                "{} + {t} tokens exceed session capacity {}",
                s.pos, s.capacity
            )));
        }
        s.prepare_kv(b, s.pos, s.pos + t)?;
        let (c, w) = (&self.cfg, &self.w);
        // Snapshot, forward and output head are one recorded step per window length.
        let shape = t as u64 | u64::from(!ws.capture.is_empty()) << 9;
        let kv = (t <= STATIC_MAX_TOKENS).then_some(s.pos + t);
        let key = graph_key(GRAPH_VERIFY, shape, s, kv);
        b.graph(key, &mut || {
            {
                // The window's GDN layers advance from the session's state into the spec buffers
                // (see `chunk`), so the state to roll back to stays in place.
                let Session { spec, pos, .. } = &mut *s;
                let sp = spec.as_mut().expect("allocated above");
                sp.pos0 = *pos;
                sp.tokens = t;
                sp.recording = true;
            }
            let r = self.chunk(b, s, ws, ids, 0, None, &[], &mut []);
            if let Some(sp) = s.spec.as_mut() {
                sp.recording = false;
            }
            r?;
            ws.last_row = t - 1;
            b.rms_norm(
                &ws.x,
                w.param(w.final_norm),
                &mut ws.hidden,
                t,
                c.hidden,
                c.eps,
                NormKind::Offset,
            )?;
            b.linear(&ws.hidden, &w.view(&w.head), &mut ws.logits_rows, t)
        })?;
        b.top_k(&ws.logits_rows, t, c.vocab, limit.min(c.vocab), k)
    }

    /// Keeps the first `keep` tokens of the last `verify` window and discards the rest: the
    /// session ends at `window start + keep`. Attention rows past that are simply overwritten
    /// later; the recurrent state is restored and the kept prefix replayed from the recorded
    /// projections.
    pub fn commit(
        &self,
        b: &B,
        s: &mut Session<B>,
        ws: &mut Workspace<B>,
        keep: usize,
    ) -> Result<()> {
        let c = &self.cfg;
        // Static, and no attention (bucket 0).
        let key = graph_key(GRAPH_COMMIT, keep as u64, s, Some(0));
        let Session {
            spec,
            conv,
            ssm,
            pos,
            parity,
            ..
        } = &mut *s;
        let sp = spec
            .as_mut()
            .ok_or_else(|| Error::Config("commit without verify".into()))?;
        if sp.batch_row0.is_some() {
            return Err(Error::Config(
                "commit after verify_batch (use commit_batched)".into(),
            ));
        }
        if keep > sp.tokens {
            return Err(Error::Shape(format!(
                "commit {keep} of a {}-token window",
                sp.tokens
            )));
        }
        if keep == sp.tokens {
            // The whole window stays: verify left its state in the spec buffers, which become
            // the session's (graph keys carry the parity, as recordings bind buffers).
            std::mem::swap(conv, &mut sp.conv);
            std::mem::swap(ssm, &mut sp.ssm);
            *parity = !*parity;
        } else {
            // The session still holds the state before the window: advance it over the kept
            // prefix from the recorded inputs, every GDN layer in one step per kept length.
            let rows = ws
                .spec_rows
                .as_ref()
                .ok_or_else(|| Error::Config("commit without verify rows".into()))?;
            b.graph(key, &mut || {
                if keep == 0 {
                    return Ok(());
                }
                let gdn_params = self.w.layers.iter().filter_map(|l| match &l.mixer {
                    Mixer::Gdn { a_log, dt_bias, .. } => Some((*a_log, *dt_bias)),
                    _ => None,
                });
                let mut layers: Vec<GdnReplayLayer<'_, B::Buffer>> = conv
                    .iter_mut()
                    .zip(ssm.iter_mut())
                    .zip(gdn_params)
                    .enumerate()
                    .map(|(gi, ((cs, ss), (a_log, dt_bias)))| GdnReplayLayer {
                        qkv: &rows.qkv[gi],
                        conv_out: &rows.conv_out[gi],
                        ab: &rows.ab[gi],
                        a_log: self.w.param(a_log),
                        dt_bias: self.w.param(dt_bias),
                        conv_state: cs,
                        ssm: ss,
                    })
                    .collect();
                b.gdn_replay(
                    &mut layers,
                    ConvShape {
                        tokens: keep,
                        channels: c.gdn_qkv(),
                        width: c.conv_width,
                        row0: 0,
                    },
                    GdnShape {
                        tokens: keep,
                        k_heads: c.gdn_k_heads,
                        v_heads: c.gdn_v_heads,
                        k_dim: c.gdn_k_dim,
                        v_dim: c.gdn_v_dim,
                        row0: 0,
                    },
                )
            })?;
        }
        *pos = sp.pos0 + keep;
        Ok(())
    }

    fn ensure_spec(&self, b: &B, s: &mut Session<B>) -> Result<()> {
        if s.spec.is_some() {
            return Ok(());
        }
        let like = |bufs: &[B::Buffer]| -> Result<Vec<B::Buffer>> {
            bufs.iter().map(|x| b.alloc(b.len(x))).collect()
        };
        s.spec = Some(Spec {
            conv: like(&s.conv)?,
            ssm: like(&s.ssm)?,
            pos0: 0,
            tokens: 0,
            recording: false,
            batch_row0: None,
        });
        Ok(())
    }

    fn ensure_spec_rows(&self, b: &B, ws: &mut Workspace<B>) -> Result<()> {
        if ws.spec_rows.is_some() {
            return Ok(());
        }
        let c = &self.cfg;
        let gdn = c.layers.iter().filter(|l| **l == LayerKind::Gdn).count();
        let per_layer = |w: usize| -> Result<Vec<B::Buffer>> {
            (0..gdn).map(|_| b.alloc(VERIFY_MAX * w)).collect()
        };
        ws.spec_rows = Some(SpecRows {
            qkv: per_layer(c.gdn_qkv())?,
            conv_out: per_layer(c.gdn_qkv())?,
            ab: per_layer(2 * c.gdn_v_heads)?,
        });
        Ok(())
    }

    /// Allocates what speculative verify needs for `s` and `ws` now rather than at the first
    /// verify, so a server's memory is committed at startup.
    pub fn reserve_verify(&self, b: &B, s: &mut Session<B>, ws: &mut Workspace<B>) -> Result<()> {
        self.ensure_spec(b, s)?;
        self.ensure_spec_rows(b, ws)
    }

    /// Buffers for batches of up to `max_seqs` sequences (see [`Self::verify_batch`]).
    pub fn batch_workspace(&self, b: &B, max_seqs: usize) -> Result<BatchWorkspace<B>> {
        let c = &self.cfg;
        let rows = max_seqs * VERIFY_MAX;
        let gdn = c.layers.iter().filter(|l| **l == LayerKind::Gdn).count();
        let per_layer =
            |w: usize| -> Result<Vec<B::Buffer>> { (0..gdn).map(|_| b.alloc(rows * w)).collect() };
        Ok(BatchWorkspace {
            max_seqs,
            qkv: per_layer(c.gdn_qkv())?,
            conv_out: per_layer(c.gdn_qkv())?,
            ab: per_layer(2 * c.gdn_v_heads)?,
            q: b.alloc(VERIFY_MAX * c.q_dim())?,
            k: b.alloc(VERIFY_MAX * c.kv_dim())?,
            v: b.alloc(VERIFY_MAX * c.kv_dim())?,
            attn: b.alloc(VERIFY_MAX * c.q_dim())?,
            hidden: b.alloc(rows * c.hidden)?,
            logits: b.alloc(rows * c.vocab)?,
            o: b.alloc(rows * c.gdn_v())?,
        })
    }

    /// [`Self::verify`] for several sequences at once: `windows[i]` (at most [`VERIFY_MAX`]
    /// tokens) is fed to `sessions[i]`. Every projection runs once over all rows; attention and
    /// the recurrences run per sequence on its own cache and state. Returns, per sequence, the `k`
    /// best candidates after each of its tokens. Each session must then be committed with
    /// [`Self::commit_batched`] before `bw` or `ws` is used again. DFlash features of sequence
    /// `i` are left in `ws` at rows starting at the sum of the earlier windows' lengths.
    #[allow(clippy::too_many_arguments)]
    pub fn verify_batch(
        &self,
        b: &B,
        sessions: &mut [&mut Session<B>],
        ws: &mut Workspace<B>,
        bw: &mut BatchWorkspace<B>,
        windows: &[&[u32]],
        k: usize,
        limit: usize,
    ) -> Result<Vec<Vec<Vec<(u32, f32)>>>> {
        let n = windows.len();
        if n != sessions.len() || n == 0 || n > bw.max_seqs {
            return Err(Error::Shape(format!(
                "verify_batch: {n} windows for {} sessions (at most {})",
                sessions.len(),
                bw.max_seqs
            )));
        }
        let mut rows = Vec::with_capacity(n);
        let mut ids = Vec::new();
        for (s, w) in sessions.iter_mut().zip(windows) {
            if w.is_empty() || w.len() > VERIFY_MAX {
                return Err(Error::Shape(format!(
                    "verify takes 1..={VERIFY_MAX} tokens per sequence"
                )));
            }
            if s.pos + w.len() > s.capacity {
                return Err(Error::Config(
                    "verify window exceeds session capacity".into(),
                ));
            }
            self.ensure_spec(b, s)?;
            s.prepare_kv(b, s.pos, s.pos + w.len())?;
            rows.push((ids.len(), w.len()));
            ids.extend_from_slice(w);
        }
        let t = ids.len();
        if t > ws.max_tokens {
            return Err(Error::Shape("verify batch exceeds the workspace".into()));
        }
        let (c, w) = (&self.cfg, &self.w);
        let run = |sessions: &mut [&mut Session<B>],
                   ws: &mut Workspace<B>,
                   bw: &mut BatchWorkspace<B>|
         -> Result<()> {
            self.chunk_batch(b, None, &[], sessions, ws, bw, &ids, &rows)?;
            b.rms_norm(
                &ws.x,
                w.param(w.final_norm),
                &mut bw.hidden,
                t,
                c.hidden,
                c.eps,
                NormKind::Offset,
            )?;
            b.linear(&bw.hidden, &w.view(&w.head), &mut bw.logits, t)
        };
        // Equal windows: one recorded step per (sequences, window, largest KV bucket). Caches,
        // states and positions are per-call values (see `Backend::graph`), so the recording
        // serves whichever sequences the batch holds.
        let m = windows[0].len();
        if windows.iter().all(|w| w.len() == m) {
            let kv = sessions.iter().map(|s| s.pos + m).max().unwrap_or(0);
            let key = GRAPH_VERIFY_BATCH
                | GRAPH_STATIC
                | n as u64
                | (m as u64) << 4
                | u64::from(!ws.capture.is_empty()) << 9
                | (kv.div_ceil(GRAPH_KV_BUCKET) as u64 & 0xffff) << 24;
            b.graph(key, &mut || run(sessions, ws, bw))?;
        } else {
            run(sessions, ws, bw)?;
        }
        let mut all = b.top_k(&bw.logits, t, c.vocab, limit.min(c.vocab), k)?;
        let mut out = Vec::with_capacity(n);
        for &(_, m) in rows.iter().rev() {
            out.push(all.split_off(all.len() - m));
        }
        out.reverse();
        for (s, &(r0, m)) in sessions.iter_mut().zip(&rows) {
            let pos = s.pos;
            let sp = s.spec.as_mut().expect("allocated above");
            sp.pos0 = pos;
            sp.tokens = m;
            sp.batch_row0 = Some(r0);
            s.pos += m;
        }
        Ok(out)
    }

    /// [`Self::commit`] for a session verified in a batch (replaying from `bw`).
    pub fn commit_batched(
        &self,
        b: &B,
        s: &mut Session<B>,
        bw: &BatchWorkspace<B>,
        keep: usize,
    ) -> Result<()> {
        let c = &self.cfg;
        let Session {
            spec,
            conv,
            ssm,
            pos,
            parity,
            ..
        } = &mut *s;
        let sp = spec
            .as_mut()
            .ok_or_else(|| Error::Config("commit without verify".into()))?;
        let row0 = sp
            .batch_row0
            .take()
            .ok_or_else(|| Error::Config("commit_batched after a single verify".into()))?;
        if keep > sp.tokens {
            return Err(Error::Shape(format!(
                "commit {keep} of a {}-token window",
                sp.tokens
            )));
        }
        if keep == sp.tokens {
            std::mem::swap(conv, &mut sp.conv);
            std::mem::swap(ssm, &mut sp.ssm);
            *parity = !*parity;
        } else if keep > 0 {
            let gdn_params = self.w.layers.iter().filter_map(|l| match &l.mixer {
                Mixer::Gdn { a_log, dt_bias, .. } => Some((*a_log, *dt_bias)),
                _ => None,
            });
            let mut layers: Vec<GdnReplayLayer<'_, B::Buffer>> = conv
                .iter_mut()
                .zip(ssm.iter_mut())
                .zip(gdn_params)
                .enumerate()
                .map(|(gi, ((cs, ss), (a_log, dt_bias)))| GdnReplayLayer {
                    qkv: &bw.qkv[gi],
                    conv_out: &bw.conv_out[gi],
                    ab: &bw.ab[gi],
                    a_log: self.w.param(a_log),
                    dt_bias: self.w.param(dt_bias),
                    conv_state: cs,
                    ssm: ss,
                })
                .collect();
            b.gdn_replay(
                &mut layers,
                ConvShape {
                    tokens: keep,
                    channels: c.gdn_qkv(),
                    width: c.conv_width,
                    row0,
                },
                GdnShape {
                    tokens: keep,
                    k_heads: c.gdn_k_heads,
                    v_heads: c.gdn_v_heads,
                    k_dim: c.gdn_k_dim,
                    v_dim: c.gdn_v_dim,
                    row0,
                },
            )?;
        }
        *pos = sp.pos0 + keep;
        Ok(())
    }

    /// One forward pass carrying a prefill chunk of one sequence and speculative verify windows of
    /// others (see [`Self::verify_batch`]): the chunk takes rows `0 .. T` of the workspace and
    /// window `i` the rows after it (plus the earlier windows' lengths). Projections and MLPs run
    /// once over every row, so the windows ride along a compute-bound prefill almost for free.
    /// Returns the candidates after the chunk's last token (when asked) and, per window, after
    /// each of its tokens. The verified sessions are committed with [`Self::commit_batched`];
    /// their DFlash features sit at rows `T + ...` of `ws`, the chunk's at rows `0 .. T`.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn mixed_step(
        &self,
        b: &B,
        prefill: PrefillChunk<'_, '_, B>,
        sessions: &mut [&mut Session<B>],
        ws: &mut Workspace<B>,
        bw: &mut BatchWorkspace<B>,
        windows: &[&[u32]],
        k: usize,
        limit: usize,
    ) -> Result<(Option<Vec<(u32, f32)>>, Vec<Vec<Vec<(u32, f32)>>>)> {
        let (c, w) = (&self.cfg, &self.w);
        let n = windows.len();
        let tp = prefill.ids.len();
        if n != sessions.len() || n > bw.max_seqs || tp == 0 {
            return Err(Error::Shape(format!(
                "mixed_step: {tp} prefill tokens, {n} windows for {} sessions (at most {})",
                sessions.len(),
                bw.max_seqs
            )));
        }
        let ps = prefill.session;
        if ps.pos + tp > ps.capacity {
            return Err(Error::Config(format!(
                "{} + {tp} tokens exceed session capacity {}",
                ps.pos, ps.capacity
            )));
        }
        ps.prepare_kv(b, ps.pos, ps.pos + tp)?;
        let mut ids = prefill.ids.to_vec();
        let mut rows = Vec::with_capacity(n);
        for (s, win) in sessions.iter_mut().zip(windows) {
            if win.is_empty() || win.len() > VERIFY_MAX {
                return Err(Error::Shape(format!(
                    "verify takes 1..={VERIFY_MAX} tokens per sequence"
                )));
            }
            if s.pos + win.len() > s.capacity {
                return Err(Error::Config(
                    "verify window exceeds session capacity".into(),
                ));
            }
            self.ensure_spec(b, s)?;
            s.prepare_kv(b, s.pos, s.pos + win.len())?;
            rows.push((ids.len() - tp, win.len()));
            ids.extend_from_slice(win);
        }
        let t = ids.len();
        if t > ws.max_tokens {
            return Err(Error::Shape("mixed step exceeds the workspace".into()));
        }
        let axis_positions: Option<Vec<u32>> = prefill.positions.map(|p| {
            (0..3)
                .flat_map(|a| p.iter().map(move |tri| tri[a]))
                .collect()
        });
        let rope_pos0 = match axis_positions {
            Some(_) => 0,
            None => usize::try_from(ps.pos as i64 + ps.rope_delta)
                .map_err(|_| Error::Shape("negative rotary position".into()))?,
        };
        let mut pre = PrefillRows {
            s: ps,
            t: tp,
            axis_positions,
            rope_pos0,
            snaps: prefill.snaps,
        };
        self.chunk_batch(
            b,
            Some(&mut pre),
            prefill.inserts,
            sessions,
            ws,
            bw,
            &ids,
            &rows,
        )?;
        pre.s.pos += tp;
        let cands = if prefill.candidates {
            ws.last_row = tp - 1;
            Some(self.candidates(b, ws, k, limit)?)
        } else {
            None
        };
        let mut out = Vec::with_capacity(n);
        let v = t - tp;
        if v > 0 {
            // The windows' rows, final-normed from row 0 of a scratch copy.
            b.copy(&ws.x, tp * c.hidden, &mut ws.h, 0, v * c.hidden)?;
            b.rms_norm(
                &ws.h,
                w.param(w.final_norm),
                &mut bw.hidden,
                v,
                c.hidden,
                c.eps,
                NormKind::Offset,
            )?;
            b.linear(&bw.hidden, &w.view(&w.head), &mut bw.logits, v)?;
            let mut all = b.top_k(&bw.logits, v, c.vocab, limit.min(c.vocab), k)?;
            for &(_, m) in rows.iter().rev() {
                out.push(all.split_off(all.len() - m));
            }
            out.reverse();
        }
        for (s, &(r0, m)) in sessions.iter_mut().zip(&rows) {
            let pos = s.pos;
            let sp = s.spec.as_mut().expect("allocated above");
            sp.pos0 = pos;
            sp.tokens = m;
            sp.batch_row0 = Some(r0);
            s.pos += m;
        }
        Ok((cands, out))
    }

    /// The forward pass of [`Self::verify_batch`] and [`Self::mixed_step`]: `ids` holds an
    /// optional prefill chunk (rows `0 .. T`, `inserts` relative to it) then every window, window
    /// `i` at verify rows `rows[i]` (start, length) after `T`. A prefill advances its session's
    /// state in place (snapshotting on the way); windows advance into each session's spec buffers
    /// (the state to roll back to stays in place) and keep their replay inputs in `bw` at their
    /// verify rows; attention appends to each session's cache.
    #[allow(clippy::too_many_arguments)]
    fn chunk_batch(
        &self,
        b: &B,
        mut pre: Option<&mut PrefillRows<'_, B>>,
        inserts: &[EmbedInsert<'_, B::Buffer>],
        sessions: &mut [&mut Session<B>],
        ws: &mut Workspace<B>,
        bw: &mut BatchWorkspace<B>,
        ids: &[u32],
        rows: &[(usize, usize)],
    ) -> Result<()> {
        let (c, w) = (&self.cfg, &self.w);
        let t = ids.len();
        let tp = pre.as_ref().map_or(0, |p| p.t);
        let v = t - tp;
        let hd = c.hidden;
        b.embed(ids, &w.view(&w.embed), &mut ws.x)?;
        for ins in inserts {
            let (lo, hi) = (ins.token, (ins.token + ins.rows).min(tp));
            if lo < hi {
                b.copy(
                    ins.buf,
                    ins.row_offset * hd,
                    &mut ws.x,
                    lo * hd,
                    (hi - lo) * hd,
                )?;
            }
        }
        // GDN segments of the prefill end at each snapshot inside the chunk, then at its end.
        let cuts: Vec<usize> = match pre.as_ref() {
            Some(p) => {
                let pos0 = p.s.pos;
                let mut v: Vec<usize> = p
                    .snaps
                    .iter()
                    .filter(|cp| cp.pos > pos0 && cp.pos <= pos0 + p.t)
                    .map(|cp| cp.pos - pos0)
                    .chain(std::iter::once(p.t))
                    .collect();
                v.sort_unstable();
                v.dedup();
                v
            }
            None => Vec::new(),
        };
        let (mut ai, mut gi) = (0, 0);
        for (li, l) in w.layers.iter().enumerate() {
            if GRAPH_SEGMENT_STARTS.contains(&li) {
                b.graph_break()?;
            }
            if li == 0 {
                b.rms_norm(
                    &ws.x,
                    w.param(l.input_norm),
                    &mut ws.h,
                    t,
                    hd,
                    c.eps,
                    NormKind::Offset,
                )?;
            } else {
                b.add_rms_norm(
                    &mut ws.x,
                    &ws.y,
                    w.param(l.input_norm),
                    &mut ws.h,
                    t,
                    hd,
                    c.eps,
                    NormKind::Offset,
                )?;
                self.capture_features(b, ws, li - 1, t)?;
            }
            match &l.mixer {
                Mixer::Gdn {
                    qkv,
                    z,
                    qkvz,
                    ab,
                    out,
                    conv,
                    a_log,
                    dt_bias,
                    norm,
                } => {
                    let (cw, al, dt) = (w.param(*conv), w.param(*a_log), w.param(*dt_bias));
                    {
                        // Without a prefill the projections land in the replay buffers directly.
                        let (qkv_buf, ab_buf) = if tp > 0 {
                            (&mut ws.qkv, &mut ws.ab)
                        } else {
                            (&mut bw.qkv[gi], &mut bw.ab[gi])
                        };
                        match qkvz.as_ref().filter(|_| t <= FUSE_MAX_TOKENS) {
                            Some(f) => {
                                b.linear(&ws.h, &w.view(f), &mut ws.fused, t)?;
                                b.split_cols(
                                    &ws.fused,
                                    t,
                                    f.rows,
                                    &mut [
                                        (
                                            &mut *qkv_buf,
                                            qkv.0[0].row_begin - f.row_begin,
                                            qkv.rows(),
                                        ),
                                        (&mut ws.z, z.row_begin - f.row_begin, z.rows),
                                    ],
                                )?;
                            }
                            None => {
                                // conv_out ([t, gdn_qkv]) is only written after the projections.
                                linear_runs(
                                    b,
                                    |d| w.view(d),
                                    &ws.h,
                                    qkv,
                                    qkv_buf,
                                    &mut ws.conv_out,
                                    t,
                                )?;
                                b.linear(&ws.h, &w.view(z), &mut ws.z, t)?;
                            }
                        }
                        linear_runs(b, |d| w.view(d), &ws.h, ab, ab_buf, &mut ws.conv_out, t)?;
                    }
                    if let Some(p) = pre.as_deref_mut() {
                        if v > 0 {
                            let (qw, abw) = (c.gdn_qkv(), 2 * c.gdn_v_heads);
                            b.copy(&ws.qkv, tp * qw, &mut bw.qkv[gi], 0, v * qw)?;
                            b.copy(&ws.ab, tp * abw, &mut bw.ab[gi], 0, v * abw)?;
                        }
                        let pos0 = p.s.pos;
                        let Session {
                            conv: s_conv,
                            ssm: s_ssm,
                            ..
                        } = &mut *p.s;
                        let mut row0 = 0;
                        for &end in &cuts {
                            let cs = ConvShape {
                                tokens: end - row0,
                                channels: c.gdn_qkv(),
                                width: c.conv_width,
                                row0,
                            };
                            let gs = GdnShape {
                                tokens: end - row0,
                                k_heads: c.gdn_k_heads,
                                v_heads: c.gdn_v_heads,
                                k_dim: c.gdn_k_dim,
                                v_dim: c.gdn_v_dim,
                                row0,
                            };
                            b.conv1d_silu(&ws.qkv, cw, &mut s_conv[gi], &mut ws.conv_out, cs)?;
                            b.gated_delta_net(
                                &ws.conv_out,
                                &ws.ab,
                                al,
                                dt,
                                &mut s_ssm[gi],
                                &mut ws.o,
                                gs,
                            )?;
                            for cp in p.snaps.iter_mut().filter(|cp| cp.pos == pos0 + end) {
                                b.copy(&s_conv[gi], 0, &mut cp.conv[gi], 0, b.len(&s_conv[gi]))?;
                                b.copy(&s_ssm[gi], 0, &mut cp.ssm[gi], 0, b.len(&s_ssm[gi]))?;
                            }
                            row0 = end;
                        }
                    }
                    for (s, &(r0, m)) in sessions.iter_mut().zip(rows) {
                        let Session {
                            spec,
                            conv: s_conv,
                            ssm: s_ssm,
                            ..
                        } = &mut **s;
                        let sp = spec.as_mut().expect("ensure_spec ran");
                        let cs = ConvShape {
                            tokens: m,
                            channels: c.gdn_qkv(),
                            width: c.conv_width,
                            row0: r0,
                        };
                        let gs = GdnShape {
                            tokens: m,
                            k_heads: c.gdn_k_heads,
                            v_heads: c.gdn_v_heads,
                            k_dim: c.gdn_k_dim,
                            v_dim: c.gdn_v_dim,
                            row0: r0,
                        };
                        b.conv1d_silu_from(
                            &bw.qkv[gi],
                            cw,
                            &s_conv[gi],
                            &mut sp.conv[gi],
                            &mut bw.conv_out[gi],
                            cs,
                        )?;
                        // Window outputs go to the workspace's rows after the prefill.
                        let o = if tp > 0 { &mut bw.o } else { &mut ws.o };
                        b.gated_delta_net_from(
                            &bw.conv_out[gi],
                            &bw.ab[gi],
                            al,
                            dt,
                            &s_ssm[gi],
                            &mut sp.ssm[gi],
                            o,
                            gs,
                        )?;
                    }
                    if tp > 0 && v > 0 {
                        let ow = c.gdn_v();
                        b.copy(&bw.o, 0, &mut ws.o, tp * ow, v * ow)?;
                    }
                    b.rms_norm_gated(
                        &ws.o,
                        w.param(*norm),
                        &mut ws.z,
                        t * c.gdn_v_heads,
                        c.gdn_v_dim,
                        c.eps,
                        NormKind::Plain,
                    )?;
                    b.linear(&ws.z, &w.view(out), &mut ws.y, t)?;
                    gi += 1;
                }
                Mixer::Attention { .. } => {
                    self.attention_batch(
                        b,
                        pre.as_deref_mut(),
                        sessions,
                        ws,
                        bw,
                        &l.mixer,
                        ai,
                        rows,
                    )?;
                    ai += 1;
                }
            }
            b.add_rms_norm(
                &mut ws.x,
                &ws.y,
                w.param(l.post_norm),
                &mut ws.h,
                t,
                hd,
                c.eps,
                NormKind::Offset,
            )?;
            self.mlp(b, ws, l, t)?;
        }
        b.add(&mut ws.x, &ws.y, t * hd)?;
        self.capture_features(b, ws, w.layers.len() - 1, t)?;
        Ok(())
    }

    /// The attention mixer of layer `ai` (attention-layer index) for a batch: projections and
    /// norms over all rows; the prefill rows (first) rotate, append and attend in place; then per
    /// window its rows are staged, rotated at its positions, appended to its cache and attended,
    /// and the outputs gathered back.
    #[allow(clippy::too_many_arguments)]
    fn attention_batch(
        &self,
        b: &B,
        pre: Option<&mut PrefillRows<'_, B>>,
        sessions: &mut [&mut Session<B>],
        ws: &mut Workspace<B>,
        bw: &mut BatchWorkspace<B>,
        mixer: &Mixer,
        ai: usize,
        rows: &[(usize, usize)],
    ) -> Result<()> {
        let (c, w) = (&self.cfg, &self.w);
        let tp = pre.as_ref().map_or(0, |p| p.t);
        let t: usize = tp + rows.iter().map(|r| r.1).sum::<usize>();
        let Mixer::Attention {
            query,
            gate,
            key,
            value,
            fused,
            output,
            q_norm,
            k_norm,
        } = mixer
        else {
            return Err(Error::Config("attention_batch on a GDN layer".into()));
        };
        match fused.as_ref().filter(|_| t <= FUSE_MAX_TOKENS) {
            Some(f) => {
                b.linear(&ws.h, &w.view(f), &mut ws.fused, t)?;
                let col = |d: &WDesc| d.row_begin - f.row_begin;
                b.split_cols(
                    &ws.fused,
                    t,
                    f.rows,
                    &mut [
                        (&mut ws.q, col(query), query.rows),
                        (&mut ws.g, col(gate), gate.rows),
                        (&mut ws.k, col(key), key.rows),
                        (&mut ws.v, col(value), value.rows),
                    ],
                )?;
            }
            None => {
                b.linear(&ws.h, &w.view(query), &mut ws.q, t)?;
                b.linear(&ws.h, &w.view(gate), &mut ws.g, t)?;
                b.linear(&ws.h, &w.view(key), &mut ws.k, t)?;
                b.linear(&ws.h, &w.view(value), &mut ws.v, t)?;
            }
        }
        b.rms_norm(
            &ws.q,
            w.param(*q_norm),
            &mut ws.qn,
            t * c.heads,
            c.head_dim,
            c.eps,
            NormKind::Offset,
        )?;
        b.rms_norm(
            &ws.k,
            w.param(*k_norm),
            &mut ws.kn,
            t * c.kv_heads,
            c.head_dim,
            c.eps,
            NormKind::Offset,
        )?;
        let (qd, kd) = (c.q_dim(), c.kv_dim());
        if let Some(p) = pre {
            // The prefill rows rotate and attend in place; attend_rows touches only rows 0 .. tp,
            // so the windows' rows after them stay as they are.
            let rope = match &p.axis_positions {
                Some(a) => Rope::Axes(a),
                None => Rope::At(p.rope_pos0),
            };
            let kv_format = p.s.kv_format;
            let pos0 = p.s.pos;
            self.attend_rows(b, ws, &mut p.s.kv[ai], kv_format, tp, pos0, rope)?;
        }
        for (s, &(r0, m)) in sessions.iter_mut().zip(rows) {
            let r = tp + r0;
            b.copy(&ws.qn, r * qd, &mut bw.q, 0, m * qd)?;
            b.copy(&ws.kn, r * kd, &mut bw.k, 0, m * kd)?;
            b.copy(&ws.v, r * kd, &mut bw.v, 0, m * kd)?;
            let pos0 = s.pos;
            let rope_pos0 = usize::try_from(pos0 as i64 + s.rope_delta)
                .map_err(|_| Error::Shape("negative rotary position".into()))?;
            let rope = |heads| RopeShape {
                tokens: m,
                heads,
                head_dim: c.head_dim,
                rotary_dim: c.rotary_dim,
                pos0: rope_pos0,
                theta: c.rope_theta,
            };
            b.rope(&mut bw.q, rope(c.heads))?;
            b.rope(&mut bw.k, rope(c.kv_heads))?;
            let kv_format = s.kv_format;
            let (kc, vc) = &mut s.kv[ai];
            let store = KvStoreShape {
                tokens: m,
                kv_heads: c.kv_heads,
                head_dim: c.head_dim,
                pos0,
                format: kv_format,
            };
            b.kv_store(&bw.k, &bw.v, kc, vc, store)?;
            let shape = AttnShape {
                tokens: m,
                heads: c.heads,
                kv_heads: c.kv_heads,
                head_dim: c.head_dim,
                pos0,
                kv: kv_format,
            };
            b.attention(&bw.q, kc, vc, &mut bw.attn, shape)?;
            b.copy(&bw.attn, 0, &mut ws.attn, r * qd, m * qd)?;
        }
        b.sigmoid_mul(&mut ws.attn, &ws.g, t * qd)?;
        b.linear(&ws.attn, &w.view(output), &mut ws.y, t)?;
        Ok(())
    }

    /// Final norm and output head for the last advanced row, into `ws.logits`.
    fn head(&self, b: &B, ws: &mut Workspace<B>) -> Result<()> {
        let (c, w) = (&self.cfg, &self.w);
        let row = ws.last_row;
        b.copy(&ws.x, row * c.hidden, &mut ws.last, 0, c.hidden)?;
        b.rms_norm(
            &ws.last,
            w.param(w.final_norm),
            &mut ws.last_n,
            1,
            c.hidden,
            c.eps,
            NormKind::Offset,
        )?;
        b.linear(&ws.last_n, &w.view(&w.head), &mut ws.logits, 1)
    }
}

/// Index of the largest logit among the first `limit` (the tokenizer's id range).
pub fn argmax(logits: &[f32], limit: usize) -> u32 {
    let mut best = 0;
    for (i, &v) in logits[..limit.min(logits.len())].iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best as u32
}

/// Rotary positions of an attention call: consecutive from a start, or explicit per-axis
/// (`[3, t]`, MRoPE) positions.
#[derive(Clone, Copy)]
enum Rope<'a> {
    At(usize),
    Axes(&'a [u32]),
}

/// `inserts` relative to a sub-call starting at token `base` of the original call (rows before
/// `base` are dropped; the chunk loop clips the rest).
pub fn shift_inserts<'a, Buf>(
    inserts: &[EmbedInsert<'a, Buf>],
    base: usize,
) -> Vec<EmbedInsert<'a, Buf>> {
    inserts
        .iter()
        .filter(|ins| ins.token + ins.rows > base)
        .map(|ins| {
            let skip = base.saturating_sub(ins.token);
            EmbedInsert {
                token: ins.token + skip - base,
                rows: ins.rows - skip,
                buf: ins.buf,
                row_offset: ins.row_offset + skip,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use velox_core::backend::ActPolicy;
    use velox_core::format::WeightFormat;
    use velox_cpu::CpuBackend;

    #[test]
    fn split_projection_matches_its_runs() -> Result<()> {
        // Rows 1..5 of parent 0 ([6, k]) then rows 2..5 of parent 1 ([5, k]), as the IQ3_XXS
        // artifact stores GDN q|k and v in two parents.
        let (b, k, t) = (CpuBackend, 8, 3);
        let p0: Vec<f32> = (0..6 * k).map(|i| (i as f32 * 0.37).sin()).collect();
        let p1: Vec<f32> = (0..5 * k).map(|i| (i as f32 * 0.11).cos()).collect();
        let x: Vec<f32> = (0..t * k).map(|i| i as f32 * 0.05 - 0.4).collect();
        let parents = [b.upload(&p0)?, b.upload(&p1)?];
        let h = b.upload(&x)?;
        let d = |parent: usize, n, row_begin, rows| WDesc {
            parent,
            format: WeightFormat::F32,
            n,
            k,
            row_begin,
            rows,
            act: ActPolicy::A16Only,
        };
        let runs = Runs(vec![d(0, 6, 1, 4), d(1, 5, 2, 3)]);
        let view = |d: &WDesc| WeightView {
            buf: &parents[d.parent],
            format: d.format,
            parent_rows: d.n,
            k: d.k,
            row_begin: d.row_begin,
            rows: d.rows,
            act: d.act,
        };
        let (mut y, mut scratch) = (b.alloc(t * 7)?, b.alloc(t * 4)?);
        linear_runs(&b, view, &h, &runs, &mut y, &mut scratch, t)?;
        let mut got = vec![0.0; t * 7];
        b.download(&y, &mut got)?;
        let rows: Vec<&[f32]> = (1..5)
            .map(|r| &p0[r * k..(r + 1) * k])
            .chain((2..5).map(|r| &p1[r * k..(r + 1) * k]))
            .collect();
        for i in 0..t {
            for (j, w) in rows.iter().enumerate() {
                let want: f32 = (0..k).map(|c| x[i * k + c] * w[c]).sum();
                assert!((got[i * 7 + j] - want).abs() < 1e-5, "[{i},{j}]");
            }
        }
        Ok(())
    }
}
