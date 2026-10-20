//! A Llama-style decoder stack written only against [`Backend`], so the same model code runs on
//! every backend and variant. Used by the benchmark and as the end-to-end conformance case.
//!
//! Embedding lookup, the LM head and sampling are not part of this stack yet; it takes and
//! returns hidden states.

use crate::backend::{AttnShape, Backend, RopeShape};
use crate::error::{Error, Result};
use crate::plan::ModelSpec;
use crate::rng::Rng;

/// Host copy of one decoder layer's weights; projection weights are `[out, in]`.
#[derive(Clone)]
pub struct HostLayer {
    pub attn_norm: Vec<f32>,
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub mlp_norm: Vec<f32>,
    pub w_gate: Vec<f32>,
    pub w_up: Vec<f32>,
    pub w_down: Vec<f32>,
}

impl HostLayer {
    /// Synthetic weights scaled by 1/sqrt(fan_in) so activations stay O(1) through the stack.
    pub fn random(spec: &ModelSpec, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let h = spec.hidden;
        let lin =
            |rng: &mut Rng, out: usize, inp: usize| rng.vec(out * inp, 1.0 / (inp as f32).sqrt());
        let norm = |rng: &mut Rng| (0..h).map(|_| 1.0 + 0.1 * rng.uniform()).collect();
        HostLayer {
            attn_norm: norm(&mut rng),
            wq: lin(&mut rng, spec.q_dim(), h),
            wk: lin(&mut rng, spec.kv_dim(), h),
            wv: lin(&mut rng, spec.kv_dim(), h),
            wo: lin(&mut rng, h, spec.q_dim()),
            mlp_norm: norm(&mut rng),
            w_gate: lin(&mut rng, spec.intermediate, h),
            w_up: lin(&mut rng, spec.intermediate, h),
            w_down: lin(&mut rng, h, spec.intermediate),
        }
    }

    pub fn bytes(&self) -> usize {
        4 * [
            &self.attn_norm,
            &self.wq,
            &self.wk,
            &self.wv,
            &self.wo,
            &self.mlp_norm,
            &self.w_gate,
            &self.w_up,
            &self.w_down,
        ]
        .iter()
        .map(|v| v.len())
        .sum::<usize>()
    }
}

struct Layer<B: Backend> {
    attn_norm: B::Buffer,
    wq: B::Buffer,
    wk: B::Buffer,
    wv: B::Buffer,
    wo: B::Buffer,
    mlp_norm: B::Buffer,
    w_gate: B::Buffer,
    w_up: B::Buffer,
    w_down: B::Buffer,
    k_cache: B::Buffer,
    v_cache: B::Buffer,
}

/// Activations for up to `max_tokens` tokens, shared by all layers.
struct Workspace<B: Backend> {
    h: B::Buffer,
    q: B::Buffer,
    k: B::Buffer,
    v: B::Buffer,
    attn: B::Buffer,
    proj: B::Buffer,
    gate: B::Buffer,
    up: B::Buffer,
}

pub struct Stack<B: Backend> {
    spec: ModelSpec,
    layers: Vec<Layer<B>>,
    ws: Workspace<B>,
    max_tokens: usize,
    max_seq: usize,
}

impl<B: Backend> Stack<B> {
    pub fn new(
        b: &B,
        spec: &ModelSpec,
        host: &[HostLayer],
        max_seq: usize,
        max_tokens: usize,
    ) -> Result<Self> {
        if host.is_empty() || max_tokens == 0 || max_tokens > max_seq {
            return Err(Error::Config(
                "stack needs layers and 0 < max_tokens <= max_seq".into(),
            ));
        }
        let kv = max_seq * spec.kv_dim();
        let layers = host
            .iter()
            .map(|l| {
                Ok(Layer {
                    attn_norm: b.upload(&l.attn_norm)?,
                    wq: b.upload(&l.wq)?,
                    wk: b.upload(&l.wk)?,
                    wv: b.upload(&l.wv)?,
                    wo: b.upload(&l.wo)?,
                    mlp_norm: b.upload(&l.mlp_norm)?,
                    w_gate: b.upload(&l.w_gate)?,
                    w_up: b.upload(&l.w_up)?,
                    w_down: b.upload(&l.w_down)?,
                    k_cache: b.alloc(kv)?,
                    v_cache: b.alloc(kv)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let t = max_tokens;
        let ws = Workspace {
            h: b.alloc(t * spec.hidden)?,
            q: b.alloc(t * spec.q_dim())?,
            k: b.alloc(t * spec.kv_dim())?,
            v: b.alloc(t * spec.kv_dim())?,
            attn: b.alloc(t * spec.q_dim())?,
            proj: b.alloc(t * spec.hidden)?,
            gate: b.alloc(t * spec.intermediate)?,
            up: b.alloc(t * spec.intermediate)?,
        };
        Ok(Stack {
            spec: spec.clone(),
            layers,
            ws,
            max_tokens,
            max_seq,
        })
    }

    /// Runs all layers in place on hidden states `x` (`[tokens, hidden]`) at positions
    /// `pos0 .. pos0 + tokens`, appending their keys and values to the caches.
    pub fn forward(&mut self, b: &B, x: &mut B::Buffer, tokens: usize, pos0: usize) -> Result<()> {
        if tokens == 0 || tokens > self.max_tokens || pos0 + tokens > self.max_seq {
            return Err(Error::Shape(format!(
                "forward of {tokens} tokens at {pos0} exceeds workspace {} / cache {}",
                self.max_tokens, self.max_seq
            )));
        }
        let s = &self.spec;
        let ws = &mut self.ws;
        let (hd, qd, kvd, inter) = (s.hidden, s.q_dim(), s.kv_dim(), s.intermediate);
        let rope = |heads| RopeShape {
            tokens,
            heads,
            head_dim: s.head_dim,
            pos0,
            theta: s.rope_theta,
        };
        let attn_shape = AttnShape {
            tokens,
            heads: s.heads,
            kv_heads: s.kv_heads,
            head_dim: s.head_dim,
            pos0,
        };

        for l in &mut self.layers {
            b.rms_norm(x, &l.attn_norm, &mut ws.h, tokens, hd, s.norm_eps)?;
            b.matmul(&ws.h, &l.wq, &mut ws.q, tokens, qd, hd)?;
            b.matmul(&ws.h, &l.wk, &mut ws.k, tokens, kvd, hd)?;
            b.matmul(&ws.h, &l.wv, &mut ws.v, tokens, kvd, hd)?;
            b.rope(&mut ws.q, rope(s.heads))?;
            b.rope(&mut ws.k, rope(s.kv_heads))?;
            b.copy(&ws.k, &mut l.k_cache, pos0 * kvd, tokens * kvd)?;
            b.copy(&ws.v, &mut l.v_cache, pos0 * kvd, tokens * kvd)?;
            b.attention(&ws.q, &l.k_cache, &l.v_cache, &mut ws.attn, attn_shape)?;
            b.matmul(&ws.attn, &l.wo, &mut ws.proj, tokens, hd, qd)?;
            b.add(x, &ws.proj, tokens * hd)?;

            b.rms_norm(x, &l.mlp_norm, &mut ws.h, tokens, hd, s.norm_eps)?;
            b.matmul(&ws.h, &l.w_gate, &mut ws.gate, tokens, inter, hd)?;
            b.matmul(&ws.h, &l.w_up, &mut ws.up, tokens, inter, hd)?;
            b.silu_mul(&mut ws.gate, &ws.up, tokens * inter)?;
            b.matmul(&ws.gate, &l.w_down, &mut ws.proj, tokens, hd, inter)?;
            b.add(x, &ws.proj, tokens * hd)?;
        }
        Ok(())
    }
}
