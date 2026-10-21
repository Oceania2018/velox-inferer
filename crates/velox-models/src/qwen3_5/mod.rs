//! Qwen3.5 / Qwen3.8 text model (`Qwen3_5ForCausalLM`): Gated DeltaNet + gated attention hybrid.
//! The mathematics is specified in docs/qwen3_5-text.md.

mod config;
mod dflash;
mod mtp;
mod vision;
mod weights;

use velox_core::backend::{
    AttnShape, Backend, ConvShape, Copy2dShape, GdnShape, NormKind, RopeLayout, RopePosShape,
    RopeShape,
};
use velox_core::{Error, KvFormat, KvStoreShape, Result, GRAPH_KV_BUCKET, GRAPH_STATIC};

pub use config::{LayerKind, TextConfig};
pub use dflash::{BlockCandidates, DFlashConfig, DFlashModel, DFlashState, DFlashWorkspace};
pub use mtp::{MtpNext, MtpRope, MtpSession, MtpWorkspace};
pub use vision::{Qwen35Vision, VisionConfig};
pub use weights::{Layer, Mixer, TextWeights, WDesc};

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
    conv: Vec<B::Buffer>,
    ssm: Vec<B::Buffer>,
    /// Speculative-verify state, allocated on first `verify`.
    spec: Option<Spec<B>>,
    /// Distinguishes this session's graph recordings from other sessions' (they bind its buffers).
    id: u64,
    /// Flips whenever a committed verify swaps the recurrent state with the speculative buffers.
    parity: bool,
}

/// Up to this many tokens (decode, verify) adjacent projections run as one fused linear; larger
/// chunks keep them separate, where per-matrix GEMM tiles are faster than the split copy saves.
const FUSE_MAX_TOKENS: usize = 16;

/// Most tokens one `verify` call may feed (the anchor plus drafts).
pub const VERIFY_MAX: usize = 16;

/// Graph keys (`Backend::graph`) of the model's repeated steps: bits 0..16 carry the shape, 16..24
/// the kind, 24..40 the KV bucket (static steps attend over a bucket-sized grid) and 40..56 the
/// session.
const GRAPH_ADVANCE: u64 = 1 << 16;
const GRAPH_VERIFY: u64 = 2 << 16;
const GRAPH_COMMIT: u64 = 3 << 16;

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
    /// Per GDN layer: `[VERIFY_MAX, gdn_qkv]` raw qkv and `[VERIFY_MAX, 2 * v_heads]` a/b rows.
    qkv: Vec<B::Buffer>,
    ab: Vec<B::Buffer>,
    pos0: usize,
    tokens: usize,
    recording: bool,
}

/// The recurrent (GDN) state of a session after `pos` tokens. Attention KV needs no copy: rows
/// below `pos` are never rewritten, so restoring a checkpoint only rewinds the position and the
/// recurrent state.
pub struct Checkpoint<B: Backend> {
    pub pos: usize,
    pub rope_delta: i64,
    conv: Vec<B::Buffer>,
    ssm: Vec<B::Buffer>,
}

impl<B: Backend> Session<B> {
    /// Device copy of the recurrent state at the current position.
    pub fn checkpoint(&self, b: &B) -> Result<Checkpoint<B>> {
        let mut cp = self.blank_checkpoint(b, self.pos)?;
        self.snapshot_into(b, &mut cp)?;
        Ok(cp)
    }

    /// Checkpoint storage for this session's state, to be filled at `pos` by an advance (see
    /// [`Qwen35Text::advance_rows`]) or by [`Self::snapshot_into`].
    pub fn blank_checkpoint(&self, b: &B, pos: usize) -> Result<Checkpoint<B>> {
        let like = |bufs: &[B::Buffer]| -> Result<Vec<B::Buffer>> {
            bufs.iter().map(|s| b.alloc(b.len(s))).collect()
        };
        Ok(Checkpoint {
            pos,
            rope_delta: self.rope_delta,
            conv: like(&self.conv)?,
            ssm: like(&self.ssm)?,
        })
    }

    /// Copies the current recurrent state into `cp` (reused storage) and stamps its position.
    pub fn snapshot_into(&self, b: &B, cp: &mut Checkpoint<B>) -> Result<()> {
        for (src, dst) in self
            .conv
            .iter()
            .zip(cp.conv.iter_mut())
            .chain(self.ssm.iter().zip(cp.ssm.iter_mut()))
        {
            b.copy(src, 0, dst, 0, b.len(src))?;
        }
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
        for (dst, src) in self
            .conv
            .iter_mut()
            .zip(&cp.conv)
            .chain(self.ssm.iter_mut().zip(&cp.ssm))
        {
            let n = b.len(src);
            b.copy(src, 0, dst, 0, n)?;
        }
        self.pos = cp.pos;
        self.rope_delta = cp.rope_delta;
        Ok(())
    }
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
    on: B::Buffer,
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
            conv,
            ssm,
            spec: None,
            id: NEXT_SESSION.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            parity: false,
        })
    }

    pub fn workspace(&self, b: &B, max_tokens: usize) -> Result<Workspace<B>> {
        let c = &self.cfg;
        let t = max_tokens;
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
            on: b.alloc(t * c.gdn_v())?,
            q: b.alloc(t * c.q_dim())?,
            g: b.alloc(t * c.q_dim())?,
            k: b.alloc(t * c.kv_dim())?,
            v: b.alloc(t * c.kv_dim())?,
            qn: b.alloc(t * c.q_dim())?,
            kn: b.alloc(t * c.kv_dim())?,
            attn: b.alloc(t * c.q_dim())?,
            mg: b.alloc(t * c.intermediate)?,
            mu: b.alloc(t * c.intermediate)?,
            fused: b.alloc(t * (c.gdn_qkv() + c.gdn_v()).max(2 * c.q_dim() + 2 * c.kv_dim()))?,
            gu: b.alloc(t * 2 * c.intermediate)?,
            last: b.alloc(c.hidden)?,
            last_n: b.alloc(c.hidden)?,
            logits: b.alloc(c.vocab)?,
            hidden: b.alloc(VERIFY_MAX * c.hidden)?,
            logits_rows: b.alloc(VERIFY_MAX * c.vocab)?,
            last_row: 0,
            keep_rows: false,
            emb: b.alloc(t * c.hidden)?,
            rows_hidden: b.alloc(t * c.hidden)?,
            capture: Vec::new(),
            kv_f32: None,
            features: None,
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
        if ws.keep_rows {
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
                    let (qkv_buf, ab_buf, mut state_out) =
                        match spec.as_mut().filter(|sp| sp.recording) {
                            Some(Spec {
                                qkv: sq,
                                ab: sa,
                                conv: sc,
                                ssm: ss,
                                ..
                            }) => (&mut sq[gi], &mut sa[gi], Some((&mut sc[gi], &mut ss[gi]))),
                            None => (&mut ws.qkv, &mut ws.ab, None),
                        };
                    match qkvz.as_ref().filter(|_| t <= FUSE_MAX_TOKENS) {
                        Some(f) => {
                            b.linear(&ws.h, &w.view(f), &mut ws.fused, t)?;
                            b.split_cols(
                                &ws.fused,
                                t,
                                f.rows,
                                &mut [
                                    (&mut *qkv_buf, qkv.row_begin - f.row_begin, qkv.rows),
                                    (&mut ws.z, z.row_begin - f.row_begin, z.rows),
                                ],
                            )?;
                        }
                        None => {
                            b.linear(&ws.h, &w.view(qkv), qkv_buf, t)?;
                            b.linear(&ws.h, &w.view(z), &mut ws.z, t)?;
                        }
                    }
                    b.linear(&ws.h, &w.view(ab), ab_buf, t)?;
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
                            Some((sc, ss)) => {
                                b.conv1d_silu_from(
                                    qkv_buf,
                                    cw,
                                    &s_conv[gi],
                                    sc,
                                    &mut ws.conv_out,
                                    cs,
                                )?;
                                b.gated_delta_net_from(
                                    &ws.conv_out,
                                    ab_buf,
                                    al,
                                    dt,
                                    &s_ssm[gi],
                                    ss,
                                    &mut ws.o,
                                    gs,
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
                    b.rms_norm(
                        &ws.o,
                        w.param(*norm),
                        &mut ws.on,
                        t * c.gdn_v_heads,
                        c.gdn_v_dim,
                        c.eps,
                        NormKind::Plain,
                    )?;
                    // z = silu(z) * plain_rmsnorm(o)
                    b.silu_mul(&mut ws.z, &ws.on, t * c.gdn_v())?;
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
        if ws.keep_rows {
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
        b.sigmoid_mul(&mut ws.attn, &ws.g, t * c.q_dim())?;
        b.linear(&ws.attn, &w.view(output), &mut ws.y, t)?;
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

    /// Dense FFN of one layer: `ws.h` to `ws.y`.
    fn mlp(&self, b: &B, ws: &mut Workspace<B>, l: &Layer, t: usize) -> Result<()> {
        let (c, w) = (&self.cfg, &self.w);
        match l.mlp_gate_up.as_ref().filter(|_| t <= FUSE_MAX_TOKENS) {
            Some(f) => {
                b.linear(&ws.h, &w.view(f), &mut ws.gu, t)?;
                let (g, u) = (
                    l.mlp_gate.row_begin - f.row_begin,
                    l.mlp_up.row_begin - f.row_begin,
                );
                b.swiglu(&ws.gu, t, f.rows, g, u, c.intermediate, &mut ws.mg)?;
            }
            None => {
                b.linear(&ws.h, &w.view(&l.mlp_gate), &mut ws.mg, t)?;
                b.linear(&ws.h, &w.view(&l.mlp_up), &mut ws.mu, t)?;
                b.silu_mul(&mut ws.mg, &ws.mu, t * c.intermediate)?;
            }
        }
        b.linear(&ws.mg, &w.view(&l.mlp_down), &mut ws.y, t)
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
        if s.pos + t > s.capacity {
            return Err(Error::Config(format!(
                "{} + {t} tokens exceed session capacity {}",
                s.pos, s.capacity
            )));
        }
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
            // The session still holds the state before the window: replay the kept prefix onto
            // it, one recorded step per kept length.
            let sp = &*sp;
            b.graph(key, &mut || {
                if keep > 0 {
                    let mut gi = 0;
                    for l in &self.w.layers {
                        if let Mixer::Gdn {
                            conv: cw,
                            a_log,
                            dt_bias,
                            ..
                        } = &l.mixer
                        {
                            let cs = ConvShape {
                                tokens: keep,
                                channels: c.gdn_qkv(),
                                width: c.conv_width,
                                row0: 0,
                            };
                            b.conv1d_silu(
                                &sp.qkv[gi],
                                self.w.param(*cw),
                                &mut conv[gi],
                                &mut ws.conv_out,
                                cs,
                            )?;
                            let gs = GdnShape {
                                tokens: keep,
                                k_heads: c.gdn_k_heads,
                                v_heads: c.gdn_v_heads,
                                k_dim: c.gdn_k_dim,
                                v_dim: c.gdn_v_dim,
                                row0: 0,
                            };
                            b.gated_delta_net(
                                &ws.conv_out,
                                &sp.ab[gi],
                                self.w.param(*a_log),
                                self.w.param(*dt_bias),
                                &mut ssm[gi],
                                &mut ws.o,
                                gs,
                            )?;
                            gi += 1;
                        }
                    }
                }
                Ok(())
            })?;
        }
        *pos = sp.pos0 + keep;
        Ok(())
    }

    fn ensure_spec(&self, b: &B, s: &mut Session<B>) -> Result<()> {
        if s.spec.is_some() {
            return Ok(());
        }
        let c = &self.cfg;
        let like = |bufs: &[B::Buffer]| -> Result<Vec<B::Buffer>> {
            bufs.iter().map(|x| b.alloc(b.len(x))).collect()
        };
        let gdn = s.ssm.len();
        s.spec = Some(Spec {
            conv: like(&s.conv)?,
            ssm: like(&s.ssm)?,
            qkv: (0..gdn)
                .map(|_| b.alloc(VERIFY_MAX * c.gdn_qkv()))
                .collect::<Result<_>>()?,
            ab: (0..gdn)
                .map(|_| b.alloc(VERIFY_MAX * 2 * c.gdn_v_heads))
                .collect::<Result<_>>()?,
            pos0: 0,
            tokens: 0,
            recording: false,
        });
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
fn shift_inserts<'a, Buf>(
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
