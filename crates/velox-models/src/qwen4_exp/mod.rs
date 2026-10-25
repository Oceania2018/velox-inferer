//! Qwen4-Exp text model (Qwen3.8-Flash-Next), written against the `Backend` op contract; see
//! docs/qwen4_exp.md for the mathematics and the weight layout.
//!
//! A step runs `T` tokens of one sequence through the stack and leaves the logits of the last one
//! in the workspace. Routed experts are read through an [`ExpertCache`], which the caller updates
//! between steps.

mod config;
mod experts;
mod load;
mod ple;

pub use config::{LayerKind, PleConfig, Qwen4Config};
pub use experts::{CacheReport, ExpertCache};
pub use load::{load, Qwen4Weights, Shards};
pub use ple::{hash_rows, PleTable};

use velox_core::{
    AttnShape, Backend, ConvShape, DilatedConvShape, Error, GdnShape, HcCombineShape, KvFormat,
    KvStoreShape, MoeRouteShape, NormKind, QsaPoolShape, QsaSelectShape, Result, RopeShape,
    RowGateShape, GRAPH_KV_BUCKET, GRAPH_STATIC,
};

/// Graph key kind of single-token decode steps.
const GRAPH_DECODE: u64 = 1 << 16;

use load::{GdnWeights, HcWeights, Mat, Mixer, MoeWeights, PleWeights, QsaWeights};

enum LayerState<B: Backend> {
    Gdn {
        /// `[width - 1, channels]`.
        conv: B::Buffer,
        /// `[v_heads, v_dim, k_dim]`.
        ssm: B::Buffer,
    },
    Qsa {
        k: B::Buffer,
        v: B::Buffer,
        /// Raw index keys `[capacity, qsa_dim]`.
        raw: B::Buffer,
        /// Pooled index keys `[capacity / ratio, qsa_dim]`.
        pooled: B::Buffer,
    },
}

/// One sequence's state: recurrent states, KV and index caches, the PLE history.
pub struct Session<B: Backend> {
    pub pos: usize,
    pub capacity: usize,
    pub kv: KvFormat,
    layers: Vec<LayerState<B>>,
    /// PLE convolution history `[(width - 1) * dilation, streams * hidden]`.
    ple_conv: Option<B::Buffer>,
    /// The last `ngram - 1` tokens (PLE hashing context).
    history: Vec<u32>,
}

/// Intermediate buffers for steps of up to `max_tokens` tokens.
pub struct Workspace<B: Backend> {
    pub max_tokens: usize,
    r: B::Buffer,
    xn: B::Buffer,
    gate: B::Buffer,
    lo: B::Buffer,
    inj: B::Buffer,
    mixed: B::Buffer,
    y: B::Buffer,
    // Gated DeltaNet
    qkv: B::Buffer,
    z: B::Buffer,
    ab: B::Buffer,
    conv: B::Buffer,
    o: B::Buffer,
    ot: B::Buffer,
    // QSA attention
    q: B::Buffer,
    g: B::Buffer,
    k: B::Buffer,
    v: B::Buffer,
    att: B::Buffer,
    iq: B::Buffer,
    ik: B::Buffer,
    sel: B::Buffer,
    qsa_scratch: B::Buffer,
    // MoE
    logits_moe: B::Buffer,
    ids: B::Buffer,
    wts: B::Buffer,
    moe_scratch: B::Buffer,
    sh: B::Buffer,
    sh_scratch: B::Buffer,
    sh_out: B::Buffer,
    // PLE
    ple_e: B::Buffer,
    ple_key: B::Buffer,
    ple_val: B::Buffer,
    ple_kn: B::Buffer,
    ple_qn: B::Buffer,
    ple_gv: B::Buffer,
    ple_gn: B::Buffer,
    ple_c: B::Buffer,
    // Head
    last: B::Buffer,
    pub logits: B::Buffer,
    best: B::Buffer,
}

pub struct Qwen4Exp<B: Backend> {
    pub cfg: Qwen4Config,
    pub w: Qwen4Weights<B>,
    /// Threads reading PLE table rows (random reads of a file of tens of GB: queue depth wins).
    pub ple_threads: usize,
}

fn lin<B: Backend>(b: &B, x: &B::Buffer, m: &Mat<B>, y: &mut B::Buffer, t: usize) -> Result<()> {
    b.linear(x, &m.view(), y, t)
}

impl<B: Backend> Qwen4Exp<B> {
    pub fn new(cfg: Qwen4Config, w: Qwen4Weights<B>) -> Self {
        Qwen4Exp {
            cfg,
            w,
            ple_threads: 32,
        }
    }

    pub fn session(&self, b: &B, capacity: usize, kv: KvFormat) -> Result<Session<B>> {
        let c = &self.cfg;
        kv.check_head_dim(c.head_dim)?;
        let capacity = capacity.div_ceil(c.qsa_ratio) * c.qsa_ratio;
        let layers = c
            .layers
            .iter()
            .map(|k| -> Result<LayerState<B>> {
                Ok(match k {
                    LayerKind::Gdn => LayerState::Gdn {
                        conv: b.alloc((c.conv_width - 1) * c.gdn_qkv())?,
                        ssm: b.alloc(c.gdn_v_heads * c.gdn_v_dim * c.gdn_k_dim)?,
                    },
                    LayerKind::Qsa => {
                        let slots = kv
                            .cache_slots(capacity, c.kv_heads, c.head_dim)
                            .ok_or_else(|| Error::Shape("KV cache size overflows".into()))?;
                        LayerState::Qsa {
                            k: b.alloc(slots)?,
                            v: b.alloc(slots)?,
                            raw: b.alloc(capacity * c.qsa_dim)?,
                            pooled: b.alloc(capacity / c.qsa_ratio * c.qsa_dim)?,
                        }
                    }
                })
            })
            .collect::<Result<_>>()?;
        let ple_conv = match &c.ple {
            Some(p) => Some(b.alloc((p.conv_width - 1) * p.dilation * c.wide())?),
            None => None,
        };
        Ok(Session {
            pos: 0,
            capacity,
            kv,
            layers,
            ple_conv,
            history: Vec::new(),
        })
    }

    pub fn workspace(
        &self,
        b: &B,
        cache: &ExpertCache<B>,
        max_tokens: usize,
        capacity: usize,
    ) -> Result<Workspace<B>> {
        let c = &self.cfg;
        let t = max_tokens.max(1);
        let a = |n: usize| b.alloc(n.max(1));
        let moe_scratch = {
            let rec = &self.w.layers[0].moe.experts;
            b.moe_scratch_len(&cache.bank(0, rec), t, c.top_k)
        };
        let blocks = capacity.div_ceil(c.qsa_ratio);
        Ok(Workspace {
            max_tokens: t,
            r: a(t * c.wide())?,
            xn: a(t * c.wide())?,
            gate: a(t * c.wide())?,
            lo: a(t * c.hc_rank)?,
            inj: a(t * c.streams)?,
            mixed: a(t * c.hidden)?,
            y: a(t * c.hidden)?,
            qkv: a(t * c.gdn_qkv())?,
            z: a(t * c.gdn_v())?,
            ab: a(t * 2 * c.gdn_v_heads)?,
            conv: a(t * c.gdn_qkv())?,
            o: a(t * c.gdn_v())?,
            ot: a(t * c.gdn_v())?,
            q: a(t * c.q_dim())?,
            g: a(t * c.q_dim())?,
            k: a(t * c.kv_dim())?,
            v: a(t * c.kv_dim())?,
            att: a(t * c.q_dim())?,
            iq: a(t * c.qsa_heads * c.qsa_dim)?,
            ik: a(t * c.qsa_dim)?,
            sel: a(t * (c.qsa_budget + c.qsa_ratio - 1))?,
            qsa_scratch: a(t * blocks)?,
            logits_moe: a(t * (c.experts + 1))?,
            ids: a(t * c.top_k)?,
            wts: a(t * c.top_k)?,
            moe_scratch: a(moe_scratch)?,
            sh: a(t * c.shared_width)?,
            sh_scratch: a(t * 2 * c.shared_width)?,
            sh_out: a(t * c.hidden)?,
            ple_e: a(t * c.hidden)?,
            ple_key: a(t * c.wide())?,
            ple_val: a(t * c.hidden)?,
            ple_kn: a(t * c.wide())?,
            ple_qn: a(t * c.wide())?,
            ple_gv: a(t * c.wide())?,
            ple_gn: a(t * c.wide())?,
            ple_c: a(t * c.wide())?,
            last: a(c.wide())?,
            logits: a(c.vocab)?,
            best: a(1)?,
        })
    }

    /// Runs `ids` (the next tokens of `s`) through the stack; the last token's logits end up in
    /// `ws.logits`.
    pub fn forward(
        &self,
        b: &B,
        s: &mut Session<B>,
        ws: &mut Workspace<B>,
        cache: &mut ExpertCache<B>,
        ids: &[u32],
    ) -> Result<()> {
        let c = &self.cfg;
        let t = ids.len();
        if t == 0 || t > ws.max_tokens {
            return Err(Error::Shape(format!(
                "step of {t} tokens (workspace holds {})",
                ws.max_tokens
            )));
        }
        if s.pos + t > s.capacity {
            return Err(Error::Shape(format!(
                "sequence of {} tokens past its capacity {}",
                s.pos + t,
                s.capacity
            )));
        }
        // PLE rows are read on the host before the step: nothing in it waits for the host.
        let ple_e = match (&c.ple, &self.w.ple_table) {
            (Some(_), Some(table)) => {
                let rows = table.rows(&s.history, ids);
                Some(table.embed(&rows, self.ple_threads)?)
            }
            _ => None,
        };
        let pos0 = s.pos;
        let dense = QsaSelectShape {
            tokens: t,
            heads: c.qsa_heads,
            dim: c.qsa_dim,
            pos0,
            ratio: c.qsa_ratio,
            budget: c.qsa_budget,
        }
        .dense();
        if t == 1 && dense {
            // A decode step on dense attention is the same work every time but for the position
            // (staged by the ops) and whether it completes a QSA block: one recorded graph per
            // KV bucket and block flag.
            let pool = (pos0 + 1) % c.qsa_ratio == 0;
            let key = GRAPH_STATIC
                | GRAPH_DECODE
                | u64::from(pool)
                | ((pos0 + 1).div_ceil(GRAPH_KV_BUCKET) as u64 & 0xffff) << 24;
            let mut step = || self.step(b, s, ws, cache, ids, ple_e.as_deref());
            b.graph(key, &mut step)?;
        } else {
            self.step(b, s, ws, cache, ids, ple_e.as_deref())?;
        }
        s.pos += t;
        if let Some(p) = &c.ple {
            s.history.extend_from_slice(ids);
            let keep = p.ngram - 1;
            if s.history.len() > keep {
                s.history.drain(..s.history.len() - keep);
            }
        }
        Ok(())
    }

    /// The device work of a step of `ids` at `s.pos` (no host waits: it may be recorded).
    fn step(
        &self,
        b: &B,
        s: &mut Session<B>,
        ws: &mut Workspace<B>,
        cache: &mut ExpertCache<B>,
        ids: &[u32],
        ple_e: Option<&[f32]>,
    ) -> Result<()> {
        let c = &self.cfg;
        let t = ids.len();
        let (h, st, wide) = (c.hidden, c.streams, c.wide());
        // The residual starts as `streams` copies of the embedding.
        b.embed(ids, &self.w.embedding.view(), &mut ws.y)?;
        for k in 0..st {
            b.copy2d(
                &ws.y,
                &mut ws.r,
                velox_core::backend::Copy2dShape {
                    rows: t,
                    cols: h,
                    src_offset: 0,
                    src_stride: h,
                    dst_offset: k * h,
                    dst_stride: wide,
                },
            )?;
        }
        let combine = HcCombineShape {
            rows: t,
            streams: st,
            dim: h,
            inject_ld: st,
            inject_col: 0,
        };
        for (li, layer) in self.w.layers.iter().enumerate() {
            if let (Some(pw), Some(e)) = (&layer.ple, ple_e) {
                b.write_at(&mut ws.ple_e, 0, e)?;
                self.ple(b, pw, s, ws, t)?;
            }
            self.hc_pre(b, &layer.hc_attn, ws, t)?;
            match (&layer.mixer, &mut s.layers[li]) {
                (Mixer::Gdn(gw), LayerState::Gdn { conv, ssm }) => {
                    self.gdn(b, gw, conv, ssm, ws, t)?
                }
                (Mixer::Qsa(qw), LayerState::Qsa { k, v, raw, pooled }) => {
                    self.qsa(b, qw, (k, v, raw, pooled), s.kv, s.pos, ws, t)?
                }
                _ => return Err(Error::Backend(format!("layer {li}: state kind mismatch"))),
            }
            b.hc_combine(&mut ws.r, &ws.y, &ws.inj, combine)?;
            self.hc_pre(b, &layer.hc_ffn, ws, t)?;
            self.moe(b, li, &layer.moe, cache, ws, t)?;
            b.hc_combine(&mut ws.r, &ws.y, &ws.inj, combine)?;
        }
        // The head mixer is the final norm: the last token only.
        b.copy(&ws.r, (t - 1) * wide, &mut ws.last, 0, wide)?;
        let hc = &self.w.hc_head;
        b.rms_norm_groups(&ws.last, &hc.norm, &mut ws.xn, 1, st, h, c.eps)?;
        lin(b, &ws.xn, &hc.down, &mut ws.lo, 1)?;
        b.silu(&mut ws.lo, c.hc_rank)?;
        lin(b, &ws.lo, &hc.up, &mut ws.gate, 1)?;
        b.hc_mix(&ws.xn, &ws.gate, &mut ws.mixed, 1, st, h)?;
        lin(b, &ws.mixed, &self.w.output, &mut ws.logits, 1)
    }

    /// The last step's logits.
    pub fn logits(&self, b: &B, ws: &Workspace<B>) -> Result<Vec<f32>> {
        let mut out = vec![0.0; self.cfg.vocab];
        b.synchronize()?;
        b.download(&ws.logits, &mut out)?;
        Ok(out)
    }

    /// The best of the first `limit` logits of the last step (greedy), chosen on the device.
    pub fn argmax(&self, b: &B, ws: &mut Workspace<B>, limit: usize) -> Result<u32> {
        b.argmax_to(&ws.logits, limit.min(self.cfg.vocab), &mut ws.best, 0)?;
        b.fetch_u32_begin(&ws.best, 0, 0)?;
        b.fetch_u32(&ws.best, 0, 0)
    }

    /// `ws.mixed` and `ws.inj` from `ws.r` through one hyper-connection mixer.
    fn hc_pre(&self, b: &B, hc: &HcWeights<B>, ws: &mut Workspace<B>, t: usize) -> Result<()> {
        let c = &self.cfg;
        b.rms_norm_groups(&ws.r, &hc.norm, &mut ws.xn, t, c.streams, c.hidden, c.eps)?;
        lin(b, &ws.xn, &hc.down, &mut ws.lo, t)?;
        b.silu(&mut ws.lo, t * c.hc_rank)?;
        lin(b, &ws.lo, &hc.up, &mut ws.gate, t)?;
        b.hc_mix(&ws.xn, &ws.gate, &mut ws.mixed, t, c.streams, c.hidden)?;
        if let Some(inj) = &hc.inject {
            lin(b, &ws.xn, inj, &mut ws.inj, t)?;
        }
        Ok(())
    }

    /// PLE: `ws.r += gated values + their convolution`, from the embedding in `ws.ple_e`.
    fn ple(
        &self,
        b: &B,
        pw: &PleWeights<B>,
        s: &mut Session<B>,
        ws: &mut Workspace<B>,
        t: usize,
    ) -> Result<()> {
        let c = &self.cfg;
        let p = c.ple.as_ref().expect("PLE layer has a PLE config");
        let (st, h, wide) = (c.streams, c.hidden, c.wide());
        lin(b, &ws.ple_e, &pw.key, &mut ws.ple_key, t)?;
        lin(b, &ws.ple_e, &pw.value, &mut ws.ple_val, t)?;
        b.rms_norm_groups(&ws.ple_key, &pw.norm_key, &mut ws.ple_kn, t, st, h, c.eps)?;
        b.rms_norm_groups(&ws.r, &pw.norm_query, &mut ws.ple_qn, t, st, h, c.eps)?;
        b.ple_gate(
            &ws.ple_kn,
            &ws.ple_qn,
            &ws.ple_val,
            &mut ws.ple_gv,
            t,
            st,
            h,
        )?;
        b.rms_norm_groups(&ws.ple_gv, &pw.norm_conv, &mut ws.ple_gn, t, st, h, c.eps)?;
        let state = s
            .ple_conv
            .as_mut()
            .ok_or_else(|| Error::Backend("session has no PLE state".into()))?;
        b.conv1d_dilated_silu(
            &ws.ple_gn,
            &pw.conv,
            state,
            &mut ws.ple_c,
            DilatedConvShape {
                tokens: t,
                channels: wide,
                width: p.conv_width,
                dilation: p.dilation,
            },
        )?;
        b.add(&mut ws.r, &ws.ple_gv, t * wide)?;
        b.add(&mut ws.r, &ws.ple_c, t * wide)
    }

    /// Gated DeltaNet on `ws.mixed` into `ws.y`.
    fn gdn(
        &self,
        b: &B,
        gw: &GdnWeights<B>,
        conv: &mut B::Buffer,
        ssm: &mut B::Buffer,
        ws: &mut Workspace<B>,
        t: usize,
    ) -> Result<()> {
        let c = &self.cfg;
        lin(b, &ws.mixed, &gw.qkv, &mut ws.qkv, t)?;
        lin(b, &ws.mixed, &gw.z, &mut ws.z, t)?;
        lin(b, &ws.mixed, &gw.ab, &mut ws.ab, t)?;
        b.conv1d_silu(
            &ws.qkv,
            &gw.conv,
            conv,
            &mut ws.conv,
            ConvShape {
                tokens: t,
                channels: c.gdn_qkv(),
                width: c.conv_width,
                row0: 0,
            },
        )?;
        b.gated_delta_net(
            &ws.conv,
            &ws.ab,
            &gw.a_log,
            &gw.dt_bias,
            ssm,
            &mut ws.o,
            GdnShape {
                tokens: t,
                k_heads: c.gdn_k_heads,
                v_heads: c.gdn_v_heads,
                k_dim: c.gdn_k_dim,
                v_dim: c.gdn_v_dim,
                row0: 0,
            },
        )?;
        // Grouped value heads to llama.cpp's tiled order, which z and the output projection use.
        let per = c.gdn_v_heads / c.gdn_k_heads;
        b.transpose_heads(&ws.o, &mut ws.ot, t, c.gdn_k_heads, per, c.gdn_v_dim)?;
        b.rms_norm(
            &ws.ot,
            &gw.norm,
            &mut ws.o,
            t * c.gdn_v_heads,
            c.gdn_v_dim,
            c.eps,
            NormKind::Plain,
        )?;
        b.sigmoid_mul(&mut ws.o, &ws.z, t * c.gdn_v())?;
        lin(b, &ws.o, &gw.out, &mut ws.y, t)
    }

    /// QSA attention on `ws.mixed` into `ws.y`.
    #[allow(clippy::too_many_arguments)]
    fn qsa(
        &self,
        b: &B,
        qw: &QsaWeights<B>,
        (kc, vc, raw, pooled): (
            &mut B::Buffer,
            &mut B::Buffer,
            &mut B::Buffer,
            &mut B::Buffer,
        ),
        kv: KvFormat,
        pos0: usize,
        ws: &mut Workspace<B>,
        t: usize,
    ) -> Result<()> {
        let c = &self.cfg;
        let (q_dim, d) = (c.q_dim(), c.head_dim);
        let qv = qw.q.view();
        b.linear(&ws.mixed, &qv.rows(0, q_dim), &mut ws.q, t)?;
        b.linear(&ws.mixed, &qv.rows(q_dim, q_dim), &mut ws.g, t)?;
        lin(b, &ws.mixed, &qw.k, &mut ws.k, t)?;
        lin(b, &ws.mixed, &qw.v, &mut ws.v, t)?;
        b.rms_norm(
            &ws.q,
            &qw.q_norm,
            &mut ws.att,
            t * c.heads,
            d,
            c.eps,
            NormKind::Plain,
        )?;
        b.rms_norm(
            &ws.k,
            &qw.k_norm,
            &mut ws.o,
            t * c.kv_heads,
            d,
            c.eps,
            NormKind::Plain,
        )?;
        let rope = |heads, head_dim| RopeShape {
            tokens: t,
            heads,
            head_dim,
            rotary_dim: c.rotary_dim,
            pos0,
            theta: c.rope_theta,
        };
        b.rope(&mut ws.att, rope(c.heads, d))?;
        b.rope(&mut ws.o, rope(c.kv_heads, d))?;
        b.kv_store(
            &ws.o,
            &ws.v,
            kc,
            vc,
            KvStoreShape {
                tokens: t,
                kv_heads: c.kv_heads,
                head_dim: d,
                pos0,
                format: kv,
            },
        )?;
        // Index keys: raw rows cached, blocks this step completes pooled.
        let di = c.qsa_dim;
        lin(b, &ws.mixed, &qw.index_k, &mut ws.ik, t)?;
        b.copy(&ws.ik, 0, raw, pos0 * di, t * di)?;
        let (b0, b1) = (pos0 / c.qsa_ratio, (pos0 + t) / c.qsa_ratio);
        if b1 > b0 {
            b.qsa_pool(
                raw,
                &qw.index_k_norm,
                pooled,
                QsaPoolShape {
                    block0: b0,
                    blocks: b1 - b0,
                    ratio: c.qsa_ratio,
                    dim: di,
                    rotary_dim: c.rotary_dim,
                    theta: c.rope_theta,
                    eps: c.eps,
                },
            )?;
        }
        let shape = AttnShape {
            tokens: t,
            heads: c.heads,
            kv_heads: c.kv_heads,
            head_dim: d,
            pos0,
            kv,
        };
        let sel = QsaSelectShape {
            tokens: t,
            heads: c.qsa_heads,
            dim: di,
            pos0,
            ratio: c.qsa_ratio,
            budget: c.qsa_budget,
        };
        // q is normalised and rotated in ws.att; the output goes to ws.q.
        if sel.dense() {
            b.attention(&ws.att, kc, vc, &mut ws.q, shape)?;
        } else {
            // The normalised index queries go to ws.ot (unused by attention layers).
            lin(b, &ws.mixed, &qw.index_q, &mut ws.iq, t)?;
            b.rms_norm(
                &ws.iq,
                &qw.index_q_norm,
                &mut ws.ot,
                t * c.qsa_heads,
                di,
                c.eps,
                NormKind::Plain,
            )?;
            b.rope(&mut ws.ot, rope(c.qsa_heads, di))?;
            b.qsa_select(&ws.ot, pooled, &mut ws.sel, &mut ws.qsa_scratch, sel)?;
            b.attention_sparse(&ws.att, kc, vc, &ws.sel, sel.sel_len(), &mut ws.q, shape)?;
        }
        b.sigmoid_mul(&mut ws.q, &ws.g, t * q_dim)?;
        lin(b, &ws.q, &qw.o, &mut ws.y, t)
    }

    /// Routed experts plus the gated shared expert on `ws.mixed` into `ws.y`.
    fn moe(
        &self,
        b: &B,
        li: usize,
        mw: &MoeWeights<B>,
        cache: &mut ExpertCache<B>,
        ws: &mut Workspace<B>,
        t: usize,
    ) -> Result<()> {
        let c = &self.cfg;
        let row_len = c.experts + 1;
        lin(b, &ws.mixed, &mw.router, &mut ws.logits_moe, t)?;
        b.moe_route(
            &ws.logits_moe,
            &mut ws.ids,
            &mut ws.wts,
            Some(cache.counts_mut(li)),
            MoeRouteShape {
                rows: t,
                row_len,
                experts: c.experts,
                k: c.top_k,
            },
        )?;
        let bank = cache.bank(li, &mw.experts);
        b.moe_experts(
            &ws.mixed,
            &ws.ids,
            &ws.wts,
            &bank,
            &mut ws.y,
            &mut ws.moe_scratch,
            t,
            c.top_k,
        )?;
        b.linear_swiglu(
            &ws.mixed,
            &mw.shared_gate.view(),
            &mw.shared_up.view(),
            &mut ws.sh,
            t,
            &mut ws.sh_scratch,
        )?;
        lin(b, &ws.sh, &mw.shared_down, &mut ws.sh_out, t)?;
        b.sigmoid_scale_rows(
            &mut ws.sh_out,
            &ws.logits_moe,
            RowGateShape {
                rows: t,
                dim: c.hidden,
                gate_offset: c.experts,
                gate_stride: row_len,
            },
        )?;
        b.add(&mut ws.y, &ws.sh_out, t * c.hidden)
    }
}
