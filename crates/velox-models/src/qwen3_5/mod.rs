//! Qwen3.5 / Qwen3.6 / Qwen3.8 text model (`Qwen3_5ForCausalLM` Dense, `Qwen3_5MoeForCausalLM` sparse
//! MoE): Gated DeltaNet + gated attention hybrid.
//! The mathematics is specified in docs/qwen3_5-text.md.

mod config;
mod dflash;
pub mod gguf;
pub mod hf;
mod mtp;
mod paging;
mod vision;
mod weights;

use std::sync::Mutex;

use velox_core::backend::{
    AttnShape, Backend, ConvShape, Copy2dShape, GdnReplayLayer, GdnShape, LinearDst, LinearPart,
    NormKind, RopeLayout, RopePosShape, RopeShape, WeightView,
};
use velox_core::{
    CopySpec, Error, KvFormat, KvStoreShape, MoeRouteShape, Result, RowGateShape, GRAPH_KV_BUCKET,
    GRAPH_STATIC,
};

use crate::experts::ExpertCache;

pub use crate::experts::CacheReport;
pub use config::{
    FfnConfig, LayerKind, MoeConfig, TextConfig, DENSE_ARCHITECTURE, MOE_ARCHITECTURE,
};
pub use dflash::{
    BlockCandidates, DFlashConfig, DFlashModel, DFlashState, DFlashWorkspace, DraftContext,
    DRAFT_PAGE_ROWS,
};
pub use mtp::{MtpNext, MtpRope, MtpSession, MtpWorkspace};
pub use paging::{KvChunkSet, KvPool, KvShare};
pub use vision::{Qwen35Vision, VisionConfig};
pub use weights::{DenseFfn, Ffn, Layer, Mixer, MoeFfn, Runs, TextWeights, WDesc};

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
    spec: Option<Spec>,
    /// Distinguishes this session's graph recordings from other sessions' (they bind its buffers).
    id: u64,
    /// Virtual-memory backing of the caches (see `paging`); `None`: allocated whole.
    paging: Option<paging::KvPaging<B>>,
}

/// Up to this many tokens (decode, verify) adjacent projections run as one fused linear; larger
/// chunks keep them separate, where per-matrix GEMM tiles are faster than the split copy saves.
const FUSE_MAX_TOKENS: usize = 16;

/// The views of a projection's runs, at their columns of the concatenated output.
fn run_parts<'w, Buf>(
    view: &impl Fn(&WDesc) -> WeightView<'w, Buf>,
    runs: &Runs,
) -> Vec<LinearPart<'w, Buf>> {
    let mut col = 0;
    runs.0
        .iter()
        .map(|d| {
            let part = LinearPart { w: view(d), col };
            col += d.rows;
            part
        })
        .collect()
}

/// `y = h . W^T` for a projection stored as runs: each run lands in its columns of `y`
/// ([`Backend::linears`]; `scratch` holds `[t, rows]` of a run when the backend copies).
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
    let parts = run_parts(&view, runs);
    let ld = runs.rows();
    b.linears(
        h,
        None,
        None,
        &mut [LinearDst {
            y,
            ld,
            parts: &parts,
        }],
        scratch,
        t,
    )
}

/// The GDN input projections of `h` as one [`Backend::linears`] call (backends quantize `h`
/// once; `hb`: its bf16 twin when there is one): q | k | v into `qkv_y`, z into `z_y` and a | b
/// into `ab_y`.
#[allow(clippy::too_many_arguments)]
fn gdn_projections<'w, B: Backend>(
    b: &B,
    view: impl Fn(&WDesc) -> WeightView<'w, B::Buffer>,
    h: &B::Buffer,
    hb: Option<&B::Buffer>,
    hq: Option<(&B::Buffer, u32)>,
    (qkv, qkv_y): (&Runs, &mut B::Buffer),
    (z, z_y): (&WDesc, &mut B::Buffer),
    (ab, ab_y): (&Runs, &mut B::Buffer),
    scratch: &mut B::Buffer,
    t: usize,
) -> Result<()>
where
    B::Buffer: 'w,
{
    let (qkv_parts, ab_parts) = (run_parts(&view, qkv), run_parts(&view, ab));
    let z_parts = [LinearPart { w: view(z), col: 0 }];
    let (qkv_ld, ab_ld) = (qkv.rows(), ab.rows());
    b.linears(
        h,
        hb,
        hq,
        &mut [
            LinearDst {
                y: qkv_y,
                ld: qkv_ld,
                parts: &qkv_parts,
            },
            LinearDst {
                y: z_y,
                ld: z.rows,
                parts: &z_parts,
            },
            LinearDst {
                y: ab_y,
                ld: ab_ld,
                parts: &ab_parts,
            },
        ],
        scratch,
        t,
    )
}

/// `x += y; h = norm(x)` (the residual add fused with the next norm), also into `h`'s int8 twin
/// in `q8`'s layout when the workspace has one (its consumer reads int8 at `t`), else into its
/// bf16 twin when `twin` and the workspace has twins. Returns which twins now mirror `h`.
#[allow(clippy::too_many_arguments)]
fn add_norm_h<B: Backend>(
    b: &B,
    ws: &mut Workspace<B>,
    weight: &B::Buffer,
    t: usize,
    hd: usize,
    eps: f32,
    twin: bool,
    q8: Option<u32>,
) -> Result<HTwin> {
    let Workspace { x, y, h, twins, .. } = ws;
    if let (Some(layout), Some(hq)) = (q8, twins.as_mut().and_then(|tw| tw.hq.as_mut())) {
        b.add_rms_norm_q8(x, y, weight, h, hq, layout, t, hd, eps, NormKind::Offset)?;
        return Ok(HTwin {
            bf16: false,
            q8: Some(layout),
        });
    }
    match twins.as_mut().filter(|_| twin) {
        Some(tw) => {
            b.add_rms_norm_twin(x, y, weight, h, &mut tw.h, t, hd, eps, NormKind::Offset)?;
            Ok(HTwin {
                bf16: true,
                q8: None,
            })
        }
        None => {
            b.add_rms_norm(x, y, weight, h, t, hd, eps, NormKind::Offset)?;
            Ok(HTwin::NONE)
        }
    }
}

/// `y = x . W^T` from `x`'s bf16 twin when the caller has it and the backend takes it.
fn linear_from<B: Backend>(
    b: &B,
    x: &B::Buffer,
    xb: Option<&B::Buffer>,
    w: &WeightView<'_, B::Buffer>,
    y: &mut B::Buffer,
    t: usize,
) -> Result<()> {
    match xb.filter(|_| b.takes_bf16(w, t)) {
        Some(xb) => b.linear_bf16(xb, w, y, t),
        None => b.linear(x, w, y, t),
    }
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
/// one layer; each later segment is submitted while the GPU runs the ones before it. Every
/// boundary also idles the GPU between segments (~30 us under WSL2), so after the first few
/// layers the rest is one segment: 12 layers of GPU time cover its submission.
const GRAPH_SEGMENT_STARTS: [usize; 3] = [1, 4, 16];

fn graph_key(kind: u64, shape: u64, s: &Session<impl Backend>, kv_len: Option<usize>) -> u64 {
    let mut key = kind | shape | (s.id & 0xffff) << 40;
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
struct Spec {
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

/// Buffers for [`Qwen35Text::verify_batch`]: several sequences' windows (at most `window`
/// rows each) share one forward pass, so every weight is read once per round.
pub struct BatchWorkspace<B: Backend> {
    max_seqs: usize,
    /// Rows of one sequence's window, at most (the drafter's drafts plus the pending token).
    window: usize,
    /// Per GDN layer, `[max_seqs * window, *]` rows: what each sequence's commit replays.
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

    /// Rows of one sequence's window, at most.
    pub fn window(&self) -> usize {
        self.window
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

/// bf16 twins (Backend::bf16_twins) of `h`, `z`, `attn` and `mg`.
struct Twins<B: Backend> {
    h: B::Buffer,
    z: B::Buffer,
    attn: B::Buffer,
    mg: B::Buffer,
    /// `h`'s int8 twin ([`Backend::int8_twin_layout`]), when the backend's int8 linears take one.
    hq: Option<B::Buffer>,
}

/// Which twins of `ws.h` are current: its bf16 twin, and its int8 twin's layout.
#[derive(Clone, Copy)]
struct HTwin {
    bf16: bool,
    q8: Option<u32>,
}

impl HTwin {
    const NONE: Self = Self {
        bf16: false,
        q8: None,
    };
}

/// Activations for chunks of up to `max_tokens` tokens.
/// Where a chunk's token ids come from: the host, or `count` ids in the workspace's device slot
/// (`next_id`, written by `argmax_into`).
#[derive(Clone, Copy)]
enum Ids<'a> {
    Host(&'a [u32]),
    Slot(usize),
}

impl Ids<'_> {
    fn len(&self) -> usize {
        match self {
            Ids::Host(ids) => ids.len(),
            Ids::Slot(n) => *n,
        }
    }
}

/// Puts the GDN outputs of `t` tokens (`ws.o`, value heads grouped by key head) into the tiled
/// order of the gate rows and output-projection columns, when the artifact stores those tiled
/// ([`TextConfig::gdn_v_tiled`]). `ws.conv_out` is free once the recurrence has run.
fn tile_v_heads<B: Backend>(b: &B, c: &TextConfig, ws: &mut Workspace<B>, t: usize) -> Result<()> {
    if !c.gdn_v_tiled {
        return Ok(());
    }
    let per = c.gdn_v_heads / c.gdn_k_heads;
    b.transpose_heads(&ws.o, &mut ws.conv_out, t, c.gdn_k_heads, per, c.gdn_v_dim)?;
    b.copy(&ws.conv_out, 0, &mut ws.o, 0, t * c.gdn_v())
}

/// A MoE FFN's buffers: router logits (`[t, experts + 1]` with the shared gate fused), the
/// shared gate alone otherwise, routes, `moe_experts` scratch and the gated shared output.
struct MoeBufs<B: Backend> {
    logits: B::Buffer,
    score: B::Buffer,
    ids: B::Buffer,
    wts: B::Buffer,
    scratch: B::Buffer,
    shared: B::Buffer,
}

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
    /// bf16 twins of the GEMM inputs for up to FUSE_MAX_TOKENS rows, when the backend writes them
    /// (Backend::bf16_twins).
    twins: Option<Twins<B>>,
    last: B::Buffer,
    last_n: B::Buffer,
    /// One slot: the id `argmax_into` selects, which `advance_from` embeds on the device.
    next_id: B::Buffer,
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
    /// Routing and routed-expert scratch of MoE models.
    moe: Option<MoeBufs<B>>,
    /// Verify replay rows (allocated with the first verify).
    spec_rows: Option<SpecRows<B>>,
    /// Where verify's recurrences write their final state: one GDN layer's conv and SSM state,
    /// shared by every layer and session and never read (commit replays the kept rows from the
    /// recorded inputs instead, as ninfer's ReplaySSM). Saves a full state copy per lane. The SSM
    /// part is only `gated_delta_net_out`'s working state (backends may skip writing it).
    spec_state: Option<(B::Buffer, B::Buffer)>,
    /// `emb` / `rows_hidden` hold `max_tokens` rows (MTP drafting reads them); without, they are
    /// placeholders and `keep_rows` does not fill them.
    rows: bool,
    /// Physical memory the per-phase buffers share (see [`overlay`]); empty without virtual
    /// memory. Declared last, so the buffers mapping it drop first.
    _overlay: Vec<B::Chunk>,
}

/// Allocates `groups` of buffers (f32 lengths) whose groups share physical memory: each group's
/// buffers are disjoint, and every group starts at the same physical chunk. For buffers of
/// phases that are never live together (a layer's GDN mixer, attention mixer and MLP scratch),
/// this takes the largest group instead of the sum. `None` without virtual memory (or with
/// `VELOX_NO_OVERLAY`): the caller allocates them separately. Contents start zeroed.
fn overlay<B: Backend>(
    b: &B,
    groups: &[&[usize]],
) -> Result<Option<(Vec<Vec<B::Buffer>>, Vec<B::Chunk>)>> {
    let chunk = b.vm_chunk_len();
    if chunk == 0 || std::env::var_os("VELOX_NO_OVERLAY").is_some() {
        return Ok(None);
    }
    let span = |len: usize| len.max(1).div_ceil(chunk);
    let need = groups
        .iter()
        .map(|g| g.iter().map(|&l| span(l)).sum::<usize>())
        .max()
        .unwrap_or(0);
    let chunks = (0..need)
        .map(|_| b.alloc_chunk())
        .collect::<Result<Vec<_>>>()?;
    let zero = b.alloc(chunk)?;
    let mut out = Vec::with_capacity(groups.len());
    for g in groups {
        let mut at = 0;
        let mut bufs = Vec::with_capacity(g.len());
        for &len in g.iter() {
            let mut buf = b.alloc_virtual(len)?;
            for j in 0..span(len) {
                b.map_chunk(&mut buf, j, &chunks[at + j])?;
            }
            at += span(len);
            bufs.push(buf);
        }
        out.push(bufs);
    }
    // Chunk contents are undefined; the first group's buffers cover every chunk (it is not
    // necessarily the largest, so zero through each group).
    for bufs in &mut out {
        for buf in bufs.iter_mut() {
            let len = b.len(buf);
            let mut off = 0;
            while off < len {
                let n = chunk.min(len - off);
                b.copy(&zero, 0, buf, off, n)?;
                off += n;
            }
        }
    }
    Ok(Some((out, chunks)))
}

pub struct Qwen35Text<B: Backend> {
    pub cfg: TextConfig,
    pub w: TextWeights<B>,
    /// Routed experts' device cache and its update policy (MoE models, after
    /// [`Self::enable_experts`]).
    experts: Option<Mutex<Experts<B>>>,
}

/// When the expert cache moves experts: every `interval` tokens at most `moves` of them, and
/// without a limit after a prompt (`prompt` tokens or more since the last update). `reads`
/// scores weight reads (an expert once per forward step) instead of selections (once per row).
#[derive(Clone, Copy, Debug)]
pub struct ExpertPolicy {
    pub interval: usize,
    pub moves: usize,
    pub prompt: usize,
    pub reads: bool,
}

impl Default for ExpertPolicy {
    fn default() -> Self {
        // As tuned on Flash-Next (docs/qwen4_exp.md#expert-offload): fast decay, large budget.
        ExpertPolicy {
            interval: 8,
            moves: 512,
            prompt: 64,
            reads: true,
        }
    }
}

struct Experts<B: Backend> {
    /// The stack's MoE layers (record sets `0..main`, the MTP layer's too when its records fit
    /// the slots), with device slots.
    cache: ExpertCache<B>,
    main: usize,
    /// Sets past the stack's (the MTP layer's, often stored wider): read from system RAM, so
    /// they do not size the stack's slots.
    rest: ExpertCache<B>,
    policy: ExpertPolicy,
    /// Tokens and forward steps run since the last update.
    pending: usize,
    steps: u64,
    /// Totals over every update.
    report: CacheReport,
}

impl<B: Backend> Qwen35Text<B> {
    pub fn new(cfg: TextConfig, w: TextWeights<B>) -> Self {
        Qwen35Text {
            cfg,
            w,
            experts: None,
        }
    }

    /// Routed experts per MoE layer (0 for a dense model).
    pub fn cfg_experts(&self) -> usize {
        match self.cfg.ffn {
            FfnConfig::Moe(m) => m.experts,
            FfnConfig::Dense { .. } => 0,
        }
    }

    /// Record sets of the stack's MoE layers (the first ones; the MTP layer's come after).
    fn main_expert_sets(&self) -> usize {
        self.w
            .layers
            .iter()
            .filter(|l| matches!(l.ffn, Ffn::Moe(_)))
            .count()
    }

    /// Bytes of one expert slot: the largest record of the stack's MoE layers (0 for a dense
    /// model).
    pub fn expert_slot_bytes(&self) -> usize {
        self.w.experts[..self.main_expert_sets()]
            .iter()
            .map(|r| r.record_bytes)
            .max()
            .unwrap_or(0)
    }

    /// The largest record of any set (the MTP layer's included).
    fn max_record_bytes(&self) -> usize {
        self.w
            .experts
            .iter()
            .map(|r| r.record_bytes)
            .max()
            .unwrap_or(0)
    }

    /// Readies the routed experts of a MoE model (required before running one): `slots` device
    /// slots hold copies of the most selected experts (0: every expert is read from system RAM).
    pub fn enable_experts(&mut self, b: &B, slots: usize, policy: ExpertPolicy) -> Result<()> {
        let FfnConfig::Moe(m) = self.cfg.ffn else {
            return Ok(());
        };
        // The MTP layer's sets share the stack's slots when their records fit them.
        let slot = self.expert_slot_bytes();
        let main = if self.w.experts.iter().all(|r| r.record_bytes <= slot) {
            self.w.experts.len()
        } else {
            self.main_expert_sets()
        };
        let cache = ExpertCache::new(b, main, m.experts, self.expert_slot_bytes().max(16), slots)?;
        let rest = ExpertCache::new(
            b,
            self.w.experts.len() - main,
            m.experts,
            self.max_record_bytes().max(16),
            0,
        )?;
        self.experts = Some(Mutex::new(Experts {
            cache,
            main,
            rest,
            policy,
            pending: 0,
            steps: 0,
            report: CacheReport::default(),
        }));
        Ok(())
    }

    /// Expert weight reads served from device slots and from system RAM so far (an expert once
    /// per forward step that selects it), and experts moved.
    pub fn expert_report(&self) -> Option<CacheReport> {
        self.experts
            .as_ref()
            .map(|e| e.lock().expect("expert cache lock").report)
    }

    /// Before a step of `tokens`: updates the expert cache when the policy says so. Runs outside
    /// any graph recording (it waits for the device).
    fn expert_tick(&self, b: &B, tokens: usize) -> Result<()> {
        let Some(e) = &self.experts else {
            return Ok(());
        };
        let mut e = e.lock().expect("expert cache lock");
        let e = &mut *e;
        if e.pending >= e.policy.interval {
            let moves = if e.pending >= e.policy.prompt {
                usize::MAX
            } else {
                e.policy.moves
            };
            let recs: Vec<_> = self.w.experts[..e.main].iter().collect();
            let r = e
                .cache
                .update_steps(b, &recs, moves, e.steps, e.policy.reads)?;
            e.report.hits += r.hits;
            e.report.misses += r.misses;
            e.report.moves += r.moves;
            e.report.resident = r.resident;
            e.pending = 0;
            e.steps = 0;
        }
        e.pending += tokens;
        e.steps += 1;
        Ok(())
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
            paging: paged.then(|| {
                let chunk = b.vm_chunk_len();
                paging::KvPaging::new(chunk, kv_len.div_ceil(chunk))
            }),
        })
    }

    pub fn workspace(&self, b: &B, max_tokens: usize) -> Result<Workspace<B>> {
        self.workspace_with(b, max_tokens, true)
    }

    /// f32 slots of an int8 twin of `h` at `t` rows: the most any layout the backend's int8
    /// linears over `h` (a mixer's first projection, the dense MLP's gate) read takes; 0 when none.
    fn int8_twin_slots(&self, b: &B, t: usize) -> usize {
        let w = &self.w;
        let mut slots = 0;
        for l in &w.layers {
            let first = match &l.mixer {
                Mixer::Gdn { qkv, .. } => qkv.0.first(),
                Mixer::Attention { query, .. } => Some(query),
            };
            let gate = match &l.ffn {
                Ffn::Dense(d) => Some(&d.gate),
                Ffn::Moe(_) => None,
            };
            for d in first.into_iter().chain(gate) {
                if let Some(layout) = b.int8_twin_layout(&w.view(d), t) {
                    slots = slots.max(b.int8_twin_slots(t, d.k, layout));
                }
            }
        }
        slots
    }

    /// [`Self::workspace`]; `rows`: whether chunks can keep their embeddings and final hidden rows
    /// (`Workspace::keep_rows`, which MTP drafting needs and DFlash does not).
    pub fn workspace_with(&self, b: &B, max_tokens: usize, rows: bool) -> Result<Workspace<B>> {
        let c = &self.cfg;
        let t = max_tokens;
        // Fused projections and the fused MLP gate|up only run up to FUSE_MAX_TOKENS rows.
        let tf = t.min(FUSE_MAX_TOKENS);
        let q8_slots = self.int8_twin_slots(b, t);
        let tr = if rows { t } else { 1 };
        // Scratch of a layer's three phases: the GDN mixer, the attention mixer and the MLP (the
        // dense SwiGLU, a MoE block's shared expert; routed experts will need their own scratch
        // once the MoE ops land). Each phase finishes into `y` before the next starts, so with
        // virtual memory the phases overlay one another.
        let gdn = [
            t * c.gdn_qkv(),
            t * c.gdn_qkv(),
            t * c.gdn_v(),
            t * 2 * c.gdn_v_heads,
            t * c.gdn_v(),
        ];
        let attn = [
            t * c.q_dim(),
            t * c.q_dim(),
            t * c.kv_dim(),
            t * c.kv_dim(),
            t * c.q_dim(),
            t * c.kv_dim(),
            t * c.q_dim(),
        ];
        let mlp = [t * c.mlp_width(), t * c.mlp_width()];
        let (bufs, backing) = match overlay(b, &[&gdn, &attn, &mlp])? {
            Some(o) => o,
            None => (
                [&gdn[..], &attn[..], &mlp[..]]
                    .iter()
                    .map(|g| g.iter().map(|&l| b.alloc(l)).collect::<Result<Vec<_>>>())
                    .collect::<Result<Vec<_>>>()?,
                Vec::new(),
            ),
        };
        let mut bufs = bufs.into_iter();
        let mut next = || bufs.next().expect("three groups").into_iter();
        let (mut g_gdn, mut g_attn, mut g_mlp) = (next(), next(), next());
        let take = |g: &mut std::vec::IntoIter<B::Buffer>| g.next().expect("group sized above");
        Ok(Workspace {
            max_tokens,
            x: b.alloc(t * c.hidden)?,
            h: b.alloc(t * c.hidden)?,
            y: b.alloc(t * c.hidden)?,
            qkv: take(&mut g_gdn),
            conv_out: take(&mut g_gdn),
            z: take(&mut g_gdn),
            ab: take(&mut g_gdn),
            o: take(&mut g_gdn),
            q: take(&mut g_attn),
            g: take(&mut g_attn),
            k: take(&mut g_attn),
            v: take(&mut g_attn),
            qn: take(&mut g_attn),
            kn: take(&mut g_attn),
            attn: take(&mut g_attn),
            mg: take(&mut g_mlp),
            mu: take(&mut g_mlp),
            fused: b.alloc(tf * (c.gdn_qkv() + c.gdn_v()).max(2 * c.q_dim() + 2 * c.kv_dim()))?,
            gu: b.alloc(tf * 2 * c.mlp_width())?,
            // z and attn at every width (their output projections take bf16 at prefill widths
            // too); h and mg only at the fused widths that consume them.
            twins: match b.bf16_twins() {
                true => Some(Twins {
                    h: b.alloc(tf * c.hidden)?,
                    z: b.alloc(t * c.gdn_v())?,
                    attn: b.alloc(t * c.q_dim())?,
                    mg: b.alloc(tf * c.mlp_width())?,
                    hq: match q8_slots {
                        0 => None,
                        n => Some(b.alloc(n)?),
                    },
                }),
                false => None,
            },
            last: b.alloc(c.hidden)?,
            last_n: b.alloc(c.hidden)?,
            next_id: b.alloc(1)?,
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
            moe: self.moe_bufs(b, t)?,
            spec_rows: None,
            spec_state: None,
            rows,
            _overlay: backing,
        })
    }

    /// MoE buffers for `t` rows (None for a dense model).
    fn moe_bufs(&self, b: &B, t: usize) -> Result<Option<MoeBufs<B>>> {
        let FfnConfig::Moe(mc) = self.cfg.ffn else {
            return Ok(None);
        };
        let (e, k, sets) = (mc.experts, mc.top_k, self.w.experts.len());
        // Scratch for the largest any layer's experts need (a probe cache without slots).
        let probe = ExpertCache::new(b, sets, e, self.max_record_bytes().max(16), 0)?;
        let scratch = (0..sets)
            .map(|i| b.moe_scratch_len(&probe.bank(i, &self.w.experts[i]), t, k))
            .max()
            .unwrap_or(1);
        Ok(Some(MoeBufs {
            logits: b.alloc(t * (e + 1))?,
            score: b.alloc(t)?,
            ids: b.alloc(t * k)?,
            wts: b.alloc(t * k)?,
            scratch: b.alloc(scratch.max(1))?,
            shared: b.alloc(t * self.cfg.hidden)?,
        }))
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

    /// [`Self::advance`] by the one token whose id is in the workspace's device slot (the one
    /// [`Self::argmax_into`] last selected), so the step queues behind the previous one without
    /// a host round trip. Its graph recording (one per session, like a decode step's) embeds
    /// from that slot every run.
    pub fn advance_from(&self, b: &B, s: &mut Session<B>, ws: &mut Workspace<B>) -> Result<()> {
        self.expert_tick(b, 1)?;
        if s.pos >= s.capacity {
            return Err(Error::Config(format!(
                "{} + 1 tokens exceed session capacity {}",
                s.pos, s.capacity
            )));
        }
        s.prepare_kv(b, s.pos, s.pos + 1)?;
        let recording = s.spec.as_ref().is_some_and(|sp| sp.recording);
        let shape = 1
            | u64::from(recording) << 8
            | u64::from(!ws.capture.is_empty()) << 9
            | u64::from(ws.keep_rows) << 10
            | 1 << 11;
        let key = graph_key(GRAPH_ADVANCE, shape, s, Some(s.pos + 1));
        b.graph(key, &mut || {
            self.chunk(b, s, ws, Ids::Slot(1), 0, None, &[], &mut [])
        })?;
        ws.last_row = 0;
        Ok(())
    }

    /// Greedy selection on the device: the best next id among the first `limit` goes to the
    /// workspace's slot (for [`Self::advance_from`]) and a copy of it is queued to host fetch
    /// slot `slot` (0..8), which [`Self::next_id`] returns. Queue the next step before fetching
    /// and the device never waits for the host.
    pub fn argmax_into(
        &self,
        b: &B,
        ws: &mut Workspace<B>,
        limit: usize,
        slot: usize,
    ) -> Result<()> {
        self.head(b, ws)?;
        let v = self.cfg.vocab;
        b.argmax_to(&ws.logits, limit.min(v), &mut ws.next_id, 0)?;
        b.fetch_u32_begin(&ws.next_id, 0, slot)
    }

    /// The id an [`Self::argmax_into`] with fetch slot `slot` selected (waits for that step).
    pub fn next_id(&self, b: &B, ws: &Workspace<B>, slot: usize) -> Result<u32> {
        b.fetch_u32(&ws.next_id, 0, slot)
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
        self.expert_tick(b, ids.len())?;
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
                self.chunk(b, s, ws, Ids::Host(ids), 0, None, &[], &mut [])
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
            self.chunk(b, s, ws, Ids::Host(chunk), base, pos, inserts, snaps)?;
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
        ids: Ids<'_>,
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
        match ids {
            Ids::Host(ids) => b.embed(ids, &w.view(&w.embed), &mut ws.x)?,
            Ids::Slot(n) => b.embed_from(&ws.next_id, 0, n, &w.view(&w.embed), &mut ws.x)?,
        }
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
            // Decode and verify widths write bf16 twins of the GEMM inputs (Backend::bf16_twins);
            // the mixers' output projections read a twin at any width the backend takes it.
            let twin = t <= FUSE_MAX_TOKENS;
            let h_twin = if li == 0 {
                b.rms_norm(
                    &ws.x,
                    w.param(l.input_norm),
                    &mut ws.h,
                    t,
                    hd,
                    c.eps,
                    NormKind::Offset,
                )?;
                HTwin::NONE
            } else {
                // The previous layer's MLP residual, fused with this layer's input norm, into the
                // twins the mixer's projections read.
                let mixer_q8 = match &l.mixer {
                    Mixer::Gdn { qkv, .. } => qkv
                        .0
                        .first()
                        .and_then(|d| b.int8_twin_layout(&w.view(d), t)),
                    Mixer::Attention { query, .. } => b.int8_twin_layout(&w.view(query), t),
                };
                let h_twin =
                    add_norm_h(b, ws, w.param(l.input_norm), t, hd, c.eps, twin, mixer_q8)?;
                self.capture_features(b, ws, li - 1, t)?;
                h_twin
            };
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
                    // shared scratch state, so the session keeps the state to roll back to.
                    let Session {
                        spec,
                        conv: s_conv,
                        ssm: s_ssm,
                        ..
                    } = &mut *s;
                    let (qkv_buf, ab_buf, mut state_out) = match (
                        spec.as_ref().filter(|sp| sp.recording),
                        ws.spec_rows.as_mut(),
                        ws.spec_state.as_mut(),
                    ) {
                        (
                            Some(_),
                            Some(SpecRows {
                                qkv: sq,
                                conv_out: so,
                                ab: sa,
                            }),
                            Some((sc, ss)),
                        ) => (&mut sq[gi], &mut sa[gi], Some((sc, ss, &mut so[gi]))),
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
                            linear_runs(b, |d| w.view(d), &ws.h, ab, ab_buf, &mut ws.conv_out, t)?;
                        }
                        None => {
                            // conv_out ([t, gdn_qkv]) is only written after the projections.
                            gdn_projections(
                                b,
                                |d| w.view(d),
                                &ws.h,
                                ws.twins.as_ref().filter(|_| h_twin.bf16).map(|tw| &tw.h),
                                ws.twins
                                    .as_ref()
                                    .and_then(|tw| tw.hq.as_ref())
                                    .zip(h_twin.q8),
                                (qkv, &mut *qkv_buf),
                                (z, &mut ws.z),
                                (ab, &mut *ab_buf),
                                &mut ws.conv_out,
                                t,
                            )?;
                        }
                    }
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
                                b.gated_delta_net_out(
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
                    tile_v_heads(b, c, ws, t)?;
                    // z = silu(z) * plain_rmsnorm(o)
                    let rows = t * c.gdn_v_heads;
                    match ws.twins.as_mut().filter(|_| b.takes_bf16(&w.view(out), t)) {
                        Some(tw) => {
                            b.rms_norm_gated_twin(
                                &ws.o,
                                w.param(*norm),
                                &mut ws.z,
                                &mut tw.z,
                                rows,
                                c.gdn_v_dim,
                                c.eps,
                                NormKind::Plain,
                            )?;
                            linear_from(b, &ws.z, Some(&tw.z), &w.view(out), &mut ws.y, t)?;
                        }
                        None => {
                            b.rms_norm_gated(
                                &ws.o,
                                w.param(*norm),
                                &mut ws.z,
                                rows,
                                c.gdn_v_dim,
                                c.eps,
                                NormKind::Plain,
                            )?;
                            b.linear(&ws.z, &w.view(out), &mut ws.y, t)?;
                        }
                    }
                    gi += 1;
                }
                Mixer::Attention { .. } => {
                    let rope = match &axis_positions {
                        Some(p) => Rope::Axes(p),
                        None => Rope::At(rope_pos0),
                    };
                    let kv_format = s.kv_format;
                    self.attention_mixer(
                        b,
                        ws,
                        &l.mixer,
                        &mut s.kv[ai],
                        kv_format,
                        (t, pos0),
                        rope,
                        (h_twin, true),
                    )?;
                    ai += 1;
                }
            }
            let mlp_q8 = match &l.ffn {
                Ffn::Dense(d) => b.int8_twin_layout(&w.view(&d.gate), t),
                Ffn::Moe(_) => None,
            };
            let h_twin = add_norm_h(b, ws, w.param(l.post_norm), t, hd, c.eps, twin, mlp_q8)?;
            self.mlp(b, ws, l, t, h_twin)?;
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
    /// `(h_twin, twin)`: whether `ws.h`'s bf16 twin is current, and whether the gated output's
    /// twin may be written for the output projection (it is, when the backend takes it at `t`).
    #[allow(clippy::too_many_arguments)]
    fn attention_mixer(
        &self,
        b: &B,
        ws: &mut Workspace<B>,
        mixer: &Mixer,
        kv: &mut (B::Buffer, B::Buffer),
        kv_format: KvFormat,
        (t, pos0): (usize, usize),
        rope: Rope<'_>,
        (h_twin, twin): (HTwin, bool),
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
                // One linears call: backends quantize ws.h once for the four projections.
                let part = |d: &WDesc| {
                    [LinearPart {
                        w: w.view(d),
                        col: 0,
                    }]
                };
                let (pq, pg, pk, pv) = (part(query), part(gate), part(key), part(value));
                b.linears(
                    &ws.h,
                    ws.twins.as_ref().filter(|_| h_twin.bf16).map(|tw| &tw.h),
                    ws.twins
                        .as_ref()
                        .and_then(|tw| tw.hq.as_ref())
                        .zip(h_twin.q8),
                    &mut [
                        LinearDst {
                            y: &mut ws.q,
                            ld: query.rows,
                            parts: &pq,
                        },
                        LinearDst {
                            y: &mut ws.g,
                            ld: gate.rows,
                            parts: &pg,
                        },
                        LinearDst {
                            y: &mut ws.k,
                            ld: key.rows,
                            parts: &pk,
                        },
                        LinearDst {
                            y: &mut ws.v,
                            ld: value.rows,
                            parts: &pv,
                        },
                    ],
                    &mut ws.fused,
                    t,
                )?;
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
        match ws
            .twins
            .as_mut()
            .filter(|_| twin && b.takes_bf16(&w.view(output), t))
        {
            Some(tw) => {
                b.sigmoid_mul_twin(&mut ws.attn, &ws.g, &mut tw.attn, t * c.q_dim())?;
                linear_from(b, &ws.attn, Some(&tw.attn), &w.view(output), &mut ws.y, t)?;
            }
            None => {
                b.sigmoid_mul(&mut ws.attn, &ws.g, t * c.q_dim())?;
                b.linear(&ws.attn, &w.view(output), &mut ws.y, t)?;
            }
        }
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
    /// `h_twin`: which twins of `ws.h` are current.
    fn mlp(&self, b: &B, ws: &mut Workspace<B>, l: &Layer, t: usize, h_twin: HTwin) -> Result<()> {
        match &l.ffn {
            Ffn::Dense(d) => self.swiglu_ffn(b, ws, d, t, h_twin),
            Ffn::Moe(m) => self.moe_ffn(b, ws, m, t, h_twin),
        }
    }

    /// `ws.y = sum_j w_j expert_j(ws.h) + sigmoid(shared_score . ws.h) shared(ws.h)`.
    fn moe_ffn(
        &self,
        b: &B,
        ws: &mut Workspace<B>,
        m: &MoeFfn,
        t: usize,
        h_twin: HTwin,
    ) -> Result<()> {
        let FfnConfig::Moe(mc) = self.cfg.ffn else {
            return Err(Error::Config("MoE layer in a dense config".into()));
        };
        let (e, k, hidden) = (mc.experts, mc.top_k, self.cfg.hidden);
        let experts = self.experts.as_ref().ok_or_else(|| {
            Error::Config("MoE model: call Qwen35Text::enable_experts before running it".into())
        })?;
        // The shared expert first (it uses the dense FFN's scratch and writes ws.y).
        self.swiglu_ffn(b, ws, &m.shared, t, h_twin)?;
        let mb = ws
            .moe
            .as_mut()
            .ok_or_else(|| Error::Config("workspace has no MoE buffers".into()))?;
        b.copy(&ws.y, 0, &mut mb.shared, 0, t * hidden)?;
        // Router logits, with the shared expert's gate logit as one more column when fused.
        let fused = m.router_score.is_some();
        match &m.router_score {
            Some(d) => b.linear(&ws.h, &self.w.view(d), &mut mb.logits, t)?,
            None => {
                b.linear(&ws.h, &self.w.view(&m.router), &mut mb.logits, t)?;
                b.linear(&ws.h, &self.w.view(&m.shared_score), &mut mb.score, t)?;
            }
        }
        let row_len = if fused { e + 1 } else { e };
        let (gate, gate_offset, gate_stride) = if fused {
            (&mb.logits, e, e + 1)
        } else {
            (&mb.score, 0, 1)
        };
        b.sigmoid_scale_rows(
            &mut mb.shared,
            gate,
            RowGateShape {
                rows: t,
                dim: hidden,
                gate_offset,
                gate_stride,
            },
        )?;
        let mut ex = experts.lock().expect("expert cache lock");
        let ex = &mut *ex;
        // Sets in the slotted cache count selections for it; the rest are not counted.
        let (cache, l, counts) = if m.experts < ex.main {
            (&mut ex.cache, m.experts, true)
        } else {
            (&mut ex.rest, m.experts - ex.main, false)
        };
        b.moe_route(
            &mb.logits,
            &mut mb.ids,
            &mut mb.wts,
            if counts {
                Some(cache.counts_mut(l))
            } else {
                None
            },
            MoeRouteShape {
                rows: t,
                row_len,
                experts: e,
                k,
            },
        )?;
        let bank = cache.bank(l, &self.w.experts[m.experts]);
        b.moe_experts(
            &ws.h,
            &mb.ids,
            &mb.wts,
            &bank,
            &mut ws.y,
            &mut mb.scratch,
            t,
            k,
        )?;
        b.add(&mut ws.y, &mb.shared, t * hidden)
    }

    /// `ws.y = down(silu(gate(ws.h)) * up(ws.h))`.
    fn swiglu_ffn(
        &self,
        b: &B,
        ws: &mut Workspace<B>,
        d: &DenseFfn,
        t: usize,
        h_twin: HTwin,
    ) -> Result<()> {
        let w = &self.w;
        let width = d.gate.rows;
        let (gate, up) = (w.view(&d.gate), w.view(&d.up));
        let down = w.view(&d.down);
        if let Some(tw) = ws
            .twins
            .as_mut()
            .filter(|_| h_twin.bf16 && b.swiglu_twin(&gate, &up, t))
        {
            // bf16 twins in and out: no activation conversions for gate|up or down.
            let Twins { h, mg, .. } = tw;
            b.linear_swiglu_twin(h, &gate, &up, &mut ws.mg, mg, t)?;
            return linear_from(b, &ws.mg, Some(mg), &down, &mut ws.y, t);
        }
        if let (Some(layout), Some(hq)) =
            (h_twin.q8, ws.twins.as_ref().and_then(|tw| tw.hq.as_ref()))
        {
            if b.int8_twin_layout(&gate, t) == Some(layout) && b.fuses_swiglu(&gate, &up, t) {
                // The int8 twin feeds the fused gate|up directly: no quantization pass.
                b.linear_swiglu_q8(hq, layout, &gate, &up, &mut ws.mg, t)?;
                return b.linear(&ws.mg, &down, &mut ws.y, t);
            }
        }
        if t > FUSE_MAX_TOKENS && b.fuses_swiglu(&gate, &up, t) {
            // Prefill on a backend that fuses this width: no scratch is touched.
            b.linear_swiglu(&ws.h, &gate, &up, &mut ws.mg, t, &mut ws.mu)?;
        } else if t <= FUSE_MAX_TOKENS && (d.gate_up.is_some() || b.fuses_swiglu(&gate, &up, t)) {
            // Decode and verify: the fused projection (ws.gu is the composed op's scratch), from
            // the concatenated gate_up weight or from separate parents the backend fuses (GGUF).
            b.linear_swiglu(&ws.h, &gate, &up, &mut ws.mg, t, &mut ws.gu)?;
        } else {
            b.linear(&ws.h, &gate, &mut ws.mg, t)?;
            b.linear(&ws.h, &up, &mut ws.mu, t)?;
            b.silu_mul(&mut ws.mg, &ws.mu, t * width)?;
        }
        b.linear(&ws.mg, &down, &mut ws.y, t)
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
        self.expert_tick(b, ids.len())?;
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
        self.expert_tick(b, ids.len())?;
        let t = ids.len();
        if t == 0 || t > VERIFY_MAX || t > ws.max_tokens {
            return Err(Error::Shape(format!(
                "verify takes 1..={} tokens, got {t}",
                VERIFY_MAX.min(ws.max_tokens)
            )));
        }
        self.ensure_spec(s)?;
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
            let r = self.chunk(b, s, ws, Ids::Host(ids), 0, None, &[], &mut []);
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
        {
            // The session still holds the state before the window (verify wrote its own to
            // scratch): advance it over the kept prefix from the recorded inputs, every GDN layer
            // in one step per kept length.
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

    fn ensure_spec(&self, s: &mut Session<B>) -> Result<()> {
        if s.spec.is_some() {
            return Ok(());
        }
        s.spec = Some(Spec {
            pos0: 0,
            tokens: 0,
            recording: false,
            batch_row0: None,
        });
        Ok(())
    }

    fn ensure_spec_state(&self, b: &B, ws: &mut Workspace<B>) -> Result<()> {
        if ws.spec_state.is_none() {
            let c = &self.cfg;
            ws.spec_state = Some((
                b.alloc((c.conv_width - 1) * c.gdn_qkv())?,
                b.alloc(c.gdn_v_heads * c.gdn_v_dim * c.gdn_k_dim)?,
            ));
        }
        Ok(())
    }

    fn ensure_spec_rows(&self, b: &B, ws: &mut Workspace<B>) -> Result<()> {
        self.ensure_spec_state(b, ws)?;
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
        self.ensure_spec(s)?;
        self.ensure_spec_rows(b, ws)
    }

    /// Buffers for batches of up to `max_seqs` sequences of up to `window` tokens each (at most
    /// [`VERIFY_MAX`]; see [`Self::verify_batch`]). Every buffer scales with `max_seqs * window`, so
    /// the window should be what the drafter proposes (drafts + 1), not the limit.
    pub fn batch_workspace(
        &self,
        b: &B,
        max_seqs: usize,
        window: usize,
    ) -> Result<BatchWorkspace<B>> {
        let c = &self.cfg;
        let window = window.clamp(1, VERIFY_MAX);
        let rows = max_seqs * window;
        let gdn = c.layers.iter().filter(|l| **l == LayerKind::Gdn).count();
        let per_layer =
            |w: usize| -> Result<Vec<B::Buffer>> { (0..gdn).map(|_| b.alloc(rows * w)).collect() };
        Ok(BatchWorkspace {
            max_seqs,
            window,
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
        self.expert_tick(b, windows.iter().map(|w| w.len()).sum())?;
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
            if w.is_empty() || w.len() > bw.window {
                return Err(Error::Shape(format!(
                    "batched verify takes 1..={} tokens per sequence",
                    bw.window
                )));
            }
            if s.pos + w.len() > s.capacity {
                return Err(Error::Config(
                    "verify window exceeds session capacity".into(),
                ));
            }
            self.ensure_spec(s)?;
            self.ensure_spec_state(b, ws)?;
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
        if keep > 0 {
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
        self.expert_tick(
            b,
            prefill.ids.len() + windows.iter().map(|w| w.len()).sum::<usize>(),
        )?;
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
            if win.is_empty() || win.len() > bw.window {
                return Err(Error::Shape(format!(
                    "batched verify takes 1..={} tokens per sequence",
                    bw.window
                )));
            }
            if s.pos + win.len() > s.capacity {
                return Err(Error::Config(
                    "verify window exceeds session capacity".into(),
                ));
            }
            self.ensure_spec(s)?;
            self.ensure_spec_state(b, ws)?;
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
                                linear_runs(
                                    b,
                                    |d| w.view(d),
                                    &ws.h,
                                    ab,
                                    ab_buf,
                                    &mut ws.conv_out,
                                    t,
                                )?;
                            }
                            None => {
                                // conv_out ([t, gdn_qkv]) is only written after the projections.
                                gdn_projections(
                                    b,
                                    |d| w.view(d),
                                    &ws.h,
                                    None,
                                    None,
                                    (qkv, &mut *qkv_buf),
                                    (z, &mut ws.z),
                                    (ab, &mut *ab_buf),
                                    &mut ws.conv_out,
                                    t,
                                )?;
                            }
                        }
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
                            conv: s_conv,
                            ssm: s_ssm,
                            ..
                        } = &mut **s;
                        let (sc, ss) = ws.spec_state.as_mut().expect("ensure_spec_state ran");
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
                            sc,
                            &mut bw.conv_out[gi],
                            cs,
                        )?;
                        // Window outputs go to the workspace's rows after the prefill.
                        let o = if tp > 0 { &mut bw.o } else { &mut ws.o };
                        b.gated_delta_net_out(
                            &bw.conv_out[gi],
                            &bw.ab[gi],
                            al,
                            dt,
                            &s_ssm[gi],
                            ss,
                            o,
                            gs,
                        )?;
                    }
                    if tp > 0 && v > 0 {
                        let ow = c.gdn_v();
                        b.copy(&bw.o, 0, &mut ws.o, tp * ow, v * ow)?;
                    }
                    tile_v_heads(b, c, ws, t)?;
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
            self.mlp(b, ws, l, t, HTwin::NONE)?;
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
                // One linears call: backends quantize ws.h once for the four projections.
                let part = |d: &WDesc| {
                    [LinearPart {
                        w: w.view(d),
                        col: 0,
                    }]
                };
                let (pq, pg, pk, pv) = (part(query), part(gate), part(key), part(value));
                b.linears(
                    &ws.h,
                    None,
                    None,
                    &mut [
                        LinearDst {
                            y: &mut ws.q,
                            ld: query.rows,
                            parts: &pq,
                        },
                        LinearDst {
                            y: &mut ws.g,
                            ld: gate.rows,
                            parts: &pg,
                        },
                        LinearDst {
                            y: &mut ws.k,
                            ld: key.rows,
                            parts: &pk,
                        },
                        LinearDst {
                            y: &mut ws.v,
                            ld: value.rows,
                            parts: &pv,
                        },
                    ],
                    &mut ws.fused,
                    t,
                )?;
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
