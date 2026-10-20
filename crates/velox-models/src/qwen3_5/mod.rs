//! Qwen3.5 / Qwen3.8 text model (`Qwen3_5ForCausalLM`): Gated DeltaNet + gated attention hybrid.
//! The mathematics is specified in docs/qwen3_5-text.md.

mod config;
mod vision;
mod weights;

use velox_core::backend::{
    AttnShape, Backend, ConvShape, GdnShape, NormKind, RopeLayout, RopePosShape, RopeShape,
};
use velox_core::{Error, Result};

pub use config::{LayerKind, TextConfig};
pub use vision::{Qwen35Vision, VisionConfig};
pub use weights::{Mixer, TextWeights, WDesc};

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
    logits: B::Buffer,
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
            last_row: 0,
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
                Mixer::Attention {
                    query,
                    gate,
                    key,
                    value,
                    output,
                    q_norm,
                    k_norm,
                } => {
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
                    match &axis_positions {
                        Some(p) => {
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
                        None => {
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
                    let (kc, vc) = &mut s.kv[ai];
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
            b.linear(&ws.h, &w.view(&l.mlp_gate), &mut ws.mg, t)?;
            b.linear(&ws.h, &w.view(&l.mlp_up), &mut ws.mu, t)?;
            b.silu_mul(&mut ws.mg, &ws.mu, t * c.intermediate)?;
            b.linear(&ws.mg, &w.view(&l.mlp_down), &mut ws.y, t)?;
            b.add(&mut ws.x, &ws.y, t * hd)?;
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
