//! Qwen3.5 / Qwen3.8 text model (`Qwen3_5ForCausalLM`): Gated DeltaNet + gated attention hybrid.
//! The mathematics is specified in docs/qwen3_5-text.md.

mod config;
mod mtp;
mod vision;
mod weights;

use velox_core::backend::{
    AttnShape, Backend, ConvShape, GdnShape, NormKind, RopeLayout, RopePosShape, RopeShape,
};
use velox_core::{Error, Result};

pub use config::{LayerKind, TextConfig};
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
    kv: Vec<(B::Buffer, B::Buffer)>,
    conv: Vec<B::Buffer>,
    ssm: Vec<B::Buffer>,
    /// Speculative-verify state, allocated on first `verify`.
    spec: Option<Spec<B>>,
}

/// Most tokens one `verify` call may feed (the anchor plus drafts).
pub const VERIFY_MAX: usize = 16;

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
        let dup = |bufs: &[B::Buffer]| -> Result<Vec<B::Buffer>> {
            bufs.iter()
                .map(|s| {
                    let n = b.len(s);
                    let mut d = b.alloc(n)?;
                    b.copy(s, 0, &mut d, 0, n)?;
                    Ok(d)
                })
                .collect()
        };
        Ok(Checkpoint {
            pos: self.pos,
            rope_delta: self.rope_delta,
            conv: dup(&self.conv)?,
            ssm: dup(&self.ssm)?,
        })
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
        let c = &self.cfg;
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
                LayerKind::FullAttention => kv.push((
                    b.alloc(capacity * c.kv_dim())?,
                    b.alloc(capacity * c.kv_dim())?,
                )),
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
            kv,
            conv,
            ssm,
            spec: None,
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
            last: b.alloc(c.hidden)?,
            last_n: b.alloc(c.hidden)?,
            logits: b.alloc(c.vocab)?,
            hidden: b.alloc(VERIFY_MAX * c.hidden)?,
            logits_rows: b.alloc(VERIFY_MAX * c.vocab)?,
            last_row: 0,
            keep_rows: false,
            emb: b.alloc(t * c.hidden)?,
            rows_hidden: b.alloc(t * c.hidden)?,
        })
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
        for (i, chunk) in ids.chunks(ws.max_tokens).enumerate() {
            let base = i * ws.max_tokens;
            let pos = positions.map(|p| &p[base..base + chunk.len()]);
            self.chunk(b, s, ws, chunk, base, pos, inserts)?;
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
    ) -> Result<()> {
        let c = &self.cfg;
        let w = &self.w;
        let t = ids.len();
        let pos0 = s.pos;
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
        for l in &w.layers {
            b.rms_norm(
                &ws.x,
                w.param(l.input_norm),
                &mut ws.h,
                t,
                hd,
                c.eps,
                NormKind::Offset,
            )?;
            match &l.mixer {
                Mixer::Gdn {
                    qkv,
                    z,
                    ab,
                    out,
                    conv,
                    a_log,
                    dt_bias,
                    norm,
                } => {
                    b.linear(&ws.h, &w.view(qkv), &mut ws.qkv, t)?;
                    b.linear(&ws.h, &w.view(z), &mut ws.z, t)?;
                    b.linear(&ws.h, &w.view(ab), &mut ws.ab, t)?;
                    if let Some(sp) = s.spec.as_mut().filter(|sp| sp.recording) {
                        b.copy(&ws.qkv, 0, &mut sp.qkv[gi], 0, t * c.gdn_qkv())?;
                        b.copy(&ws.ab, 0, &mut sp.ab[gi], 0, t * 2 * c.gdn_v_heads)?;
                    }
                    let cs = ConvShape {
                        tokens: t,
                        channels: c.gdn_qkv(),
                        width: c.conv_width,
                    };
                    b.conv1d_silu(
                        &ws.qkv,
                        w.param(*conv),
                        &mut s.conv[gi],
                        &mut ws.conv_out,
                        cs,
                    )?;
                    let gs = GdnShape {
                        tokens: t,
                        k_heads: c.gdn_k_heads,
                        v_heads: c.gdn_v_heads,
                        k_dim: c.gdn_k_dim,
                        v_dim: c.gdn_v_dim,
                    };
                    b.gated_delta_net(
                        &ws.conv_out,
                        &ws.ab,
                        w.param(*a_log),
                        w.param(*dt_bias),
                        &mut s.ssm[gi],
                        &mut ws.o,
                        gs,
                    )?;
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
                    self.attention_mixer(b, ws, &l.mixer, &mut s.kv[ai], t, pos0, rope)?;
                    ai += 1;
                }
            }
            b.add(&mut ws.x, &ws.y, t * hd)?;
            b.rms_norm(
                &ws.x,
                w.param(l.post_norm),
                &mut ws.h,
                t,
                hd,
                c.eps,
                NormKind::Offset,
            )?;
            self.mlp(b, ws, l, t)?;
            b.add(&mut ws.x, &ws.y, t * hd)?;
        }
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
            output,
            q_norm,
            k_norm,
        } = mixer
        else {
            return Err(Error::Config("attention_mixer on a GDN layer".into()));
        };
        b.linear(&ws.h, &w.view(query), &mut ws.q, t)?;
        b.linear(&ws.h, &w.view(gate), &mut ws.g, t)?;
        b.linear(&ws.h, &w.view(key), &mut ws.k, t)?;
        b.linear(&ws.h, &w.view(value), &mut ws.v, t)?;
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
        let kvd = c.kv_dim();
        b.copy(&ws.kn, 0, kc, pos0 * kvd, t * kvd)?;
        b.copy(&ws.v, 0, vc, pos0 * kvd, t * kvd)?;
        let shape = AttnShape {
            tokens: t,
            heads: c.heads,
            kv_heads: c.kv_heads,
            head_dim: c.head_dim,
            pos0,
        };
        b.attention(&ws.qn, kc, vc, &mut ws.attn, shape)?;
        b.sigmoid_mul(&mut ws.attn, &ws.g, t * c.q_dim())?;
        b.linear(&ws.attn, &w.view(output), &mut ws.y, t)?;
        Ok(())
    }

    /// Dense FFN of one layer: `ws.h` to `ws.y`.
    fn mlp(&self, b: &B, ws: &mut Workspace<B>, l: &Layer, t: usize) -> Result<()> {
        let (c, w) = (&self.cfg, &self.w);
        b.linear(&ws.h, &w.view(&l.mlp_gate), &mut ws.mg, t)?;
        b.linear(&ws.h, &w.view(&l.mlp_up), &mut ws.mu, t)?;
        b.silu_mul(&mut ws.mg, &ws.mu, t * c.intermediate)?;
        b.linear(&ws.mg, &w.view(&l.mlp_down), &mut ws.y, t)
    }

    /// [`Self::advance`] that also hands every chunk's rows to `on_chunk` (with `ws.keep_rows`
    /// set): composed embeddings in `ws.emb`, final-normed hidden rows in `ws.rows_hidden`, and
    /// the chunk's offset into `ids` and its token count.
    #[allow(clippy::too_many_arguments)]
    pub fn advance_rows(
        &self,
        b: &B,
        s: &mut Session<B>,
        ws: &mut Workspace<B>,
        ids: &[u32],
        positions: Option<&[[u32; 3]]>,
        inserts: &[EmbedInsert<'_, B::Buffer>],
        on_chunk: &mut dyn FnMut(&Workspace<B>, usize, usize) -> Result<()>,
    ) -> Result<()> {
        ws.keep_rows = true;
        let mut r = Ok(());
        for (i, chunk) in ids.chunks(ws.max_tokens).enumerate() {
            let base = i * ws.max_tokens;
            let pos = positions.map(|p| &p[base..base + chunk.len()]);
            r = self
                .advance(b, s, ws, chunk, pos, &shift_inserts(inserts, base))
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
        {
            let Session {
                spec,
                conv,
                ssm,
                pos,
                ..
            } = &mut *s;
            let sp = spec.as_mut().expect("allocated above");
            for (dst, src) in sp
                .conv
                .iter_mut()
                .zip(conv.iter())
                .chain(sp.ssm.iter_mut().zip(ssm.iter()))
            {
                b.copy(src, 0, dst, 0, b.len(src))?;
            }
            sp.pos0 = *pos;
            sp.tokens = t;
            sp.recording = true;
        }
        let r = self.advance(b, s, ws, ids, None, &[]);
        if let Some(sp) = s.spec.as_mut() {
            sp.recording = false;
        }
        r?;
        let (c, w) = (&self.cfg, &self.w);
        b.rms_norm(
            &ws.x,
            w.param(w.final_norm),
            &mut ws.hidden,
            t,
            c.hidden,
            c.eps,
            NormKind::Offset,
        )?;
        b.linear(&ws.hidden, &w.view(&w.head), &mut ws.logits_rows, t)?;
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
        let Session {
            spec,
            conv,
            ssm,
            pos,
            ..
        } = &mut *s;
        let sp = spec
            .as_ref()
            .ok_or_else(|| Error::Config("commit without verify".into()))?;
        if keep > sp.tokens {
            return Err(Error::Shape(format!(
                "commit {keep} of a {}-token window",
                sp.tokens
            )));
        }
        if keep < sp.tokens {
            for (dst, src) in conv
                .iter_mut()
                .zip(&sp.conv)
                .chain(ssm.iter_mut().zip(&sp.ssm))
            {
                b.copy(src, 0, dst, 0, b.len(src))?;
            }
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
