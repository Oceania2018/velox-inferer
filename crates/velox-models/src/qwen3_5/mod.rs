//! Qwen3.5 / Qwen3.8 text model (`Qwen3_5ForCausalLM`): Gated DeltaNet + gated attention hybrid.
//! The mathematics is specified in docs/qwen3_5-text.md.

mod config;
mod weights;

use velox_core::backend::{AttnShape, Backend, ConvShape, GdnShape, NormKind, RopeShape};
use velox_core::{Error, Result};

pub use config::{LayerKind, TextConfig};
pub use weights::{Mixer, TextWeights, WDesc};

/// Per-sequence state: attention KV caches, GDN convolution and recurrent states, position.
pub struct Session<B: Backend> {
    pub pos: usize,
    pub capacity: usize,
    kv: Vec<(B::Buffer, B::Buffer)>,
    conv: Vec<B::Buffer>,
    ssm: Vec<B::Buffer>,
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
        })
    }

    /// Feeds `ids` (any length, processed in workspace-sized chunks) and returns the logits after
    /// the last one.
    pub fn forward(
        &self,
        b: &B,
        s: &mut Session<B>,
        ws: &mut Workspace<B>,
        ids: &[u32],
    ) -> Result<Vec<f32>> {
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
        let mut t_last = 0;
        for chunk in ids.chunks(ws.max_tokens) {
            self.chunk(b, s, ws, chunk)?;
            t_last = chunk.len() - 1;
        }
        self.logits(b, ws, t_last)
    }

    fn chunk(&self, b: &B, s: &mut Session<B>, ws: &mut Workspace<B>, ids: &[u32]) -> Result<()> {
        let c = &self.cfg;
        let w = &self.w;
        let t = ids.len();
        let pos0 = s.pos;
        let hd = c.hidden;
        b.embed(ids, &w.view(&w.embed), &mut ws.x)?;
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
                    let rope = |heads| RopeShape {
                        tokens: t,
                        heads,
                        head_dim: c.head_dim,
                        rotary_dim: c.rotary_dim,
                        pos0,
                        theta: c.rope_theta,
                    };
                    b.rope(&mut ws.qn, rope(c.heads))?;
                    b.rope(&mut ws.kn, rope(c.kv_heads))?;
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

    fn logits(&self, b: &B, ws: &mut Workspace<B>, row: usize) -> Result<Vec<f32>> {
        let (c, w) = (&self.cfg, &self.w);
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
        b.linear(&ws.last_n, &w.view(&w.head), &mut ws.logits, 1)?;
        let mut out = vec![0.0; c.vocab];
        b.synchronize()?;
        b.download(&ws.logits, &mut out)?;
        Ok(out)
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
