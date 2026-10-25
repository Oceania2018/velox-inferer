//! MTP drafting for Qwen4-Exp (llama.cpp's qwen4exp `graph_mtp`).
//!
//! MTP entry `i` pairs the main model's residual streams at position `i - 1` with the token at
//! position `i`: `r_s = fc_e(norm(embed(x_i))) + fc_h(norm_s(h_{i-1,s}))` per stream, then one QSA
//! layer with its own KV and index caches, then the head's mixer and the shared output head. The
//! last entry's logits draft the token at `i + 1`. Entries are written for every position the
//! sequence keeps (prompt chunks shifted by one token, then each round's committed tokens), so the
//! MTP layer attends over the same history the main model does.

use velox_core::backend::Copy2dShape;
use velox_core::{Backend, Error, HcCombineShape, KvFormat, NormKind, Result};

use super::load::MtpWeights;
use super::{lin, ExpertCache, LayerState, Qwen4Exp, Workspace};

pub struct Mtp<B: Backend> {
    pub w: MtpWeights<B>,
    /// The MTP layer's experts are always read from system RAM (no device slots).
    cache: ExpertCache<B>,
}

pub struct MtpSession<B: Backend> {
    state: LayerState<B>,
    /// The main residual rows an `mtp_forward` reads (copied before the MTP layer reuses the
    /// workspace).
    h: B::Buffer,
    rows: usize,
    /// MoE scratch for the MTP experts' bank (swapped into the workspace for its MoE).
    moe_scratch: B::Buffer,
    kv: KvFormat,
    capacity: usize,
}

impl<B: Backend> Qwen4Exp<B> {
    pub fn mtp(&self, b: &B, w: MtpWeights<B>) -> Result<Mtp<B>> {
        let slot_bytes = w.moe.experts.record_bytes.div_ceil(16) * 16;
        Ok(Mtp {
            cache: ExpertCache::new(b, 1, self.cfg.experts, slot_bytes, 0)?,
            w,
        })
    }

    /// An MTP session of `capacity` positions taking up to `rows` entries per call.
    pub fn mtp_session(
        &self,
        b: &B,
        m: &Mtp<B>,
        capacity: usize,
        kv: KvFormat,
        rows: usize,
    ) -> Result<MtpSession<B>> {
        let c = &self.cfg;
        let rows = rows.max(1);
        let scratch = b.moe_scratch_len(&m.cache.bank(0, &m.w.moe.experts), rows, c.top_k);
        let capacity = capacity.div_ceil(c.qsa_ratio) * c.qsa_ratio;
        let slots = kv
            .cache_slots(capacity, c.kv_heads, c.head_dim)
            .ok_or_else(|| Error::Shape("KV cache size overflows".into()))?;
        Ok(MtpSession {
            state: LayerState::Qsa {
                k: b.alloc(slots)?,
                v: b.alloc(slots)?,
                raw: b.alloc(capacity * c.qsa_dim)?,
                pooled: b.alloc(capacity / c.qsa_ratio * c.qsa_dim)?,
            },
            h: b.alloc(rows * c.wide())?,
            rows,
            moe_scratch: b.alloc(scratch.max(1))?,
            kv,
            capacity,
        })
    }

    /// Writes MTP entries `pos0 .. pos0 + ids.len()`: entry `pos0 + j` from row `j` of the main
    /// residual in `ws` (the last step's or verify's) and token `ids[j]`. The last entry's draft
    /// logits end up in `ws.logits` (row 0).
    pub fn mtp_forward(
        &self,
        b: &B,
        m: &mut Mtp<B>,
        ms: &mut MtpSession<B>,
        ws: &mut Workspace<B>,
        pos0: usize,
        ids: &[u32],
    ) -> Result<()> {
        let c = &self.cfg;
        let t = ids.len();
        if t == 0 || t > ms.rows || t > ws.max_tokens || pos0 + t > ms.capacity {
            return Err(Error::Shape(format!(
                "MTP entries {pos0}+{t} (rows {}, capacity {})",
                ms.rows, ms.capacity
            )));
        }
        let (h, st, wide, eps) = (c.hidden, c.streams, c.wide(), c.eps);
        b.copy(&ws.r, 0, &mut ms.h, 0, t * wide)?;
        // r_s = fc_h(norm_s(h_s)) over the t * streams rows, plus fc_e(norm(embed(x))) per stream.
        b.rms_norm_groups(&ms.h, &m.w.hnorm, &mut ws.xn, t, st, h, eps)?;
        lin(b, &ws.xn, &m.w.fc_h, &mut ws.r, t * st)?;
        b.embed(ids, &self.w.embedding.view(), &mut ws.y)?;
        b.rms_norm(&ws.y, &m.w.enorm, &mut ws.mixed, t, h, eps, NormKind::Plain)?;
        lin(b, &ws.mixed, &m.w.fc_e, &mut ws.y, t)?;
        for k in 0..st {
            b.copy2d(
                &ws.y,
                &mut ws.gate,
                Copy2dShape {
                    rows: t,
                    cols: h,
                    src_offset: 0,
                    src_stride: h,
                    dst_offset: k * h,
                    dst_stride: wide,
                },
            )?;
        }
        b.add(&mut ws.r, &ws.gate, t * wide)?;
        let combine = HcCombineShape {
            rows: t,
            streams: st,
            dim: h,
            inject_ld: st,
            inject_col: 0,
        };
        self.hc_pre(b, &m.w.hc_attn, ws, t)?;
        let LayerState::Qsa { k, v, raw, pooled } = &mut ms.state else {
            return Err(Error::Backend("MTP state is not an attention layer".into()));
        };
        self.qsa(b, &m.w.attn, (k, v, raw, pooled), ms.kv, pos0, ws, t)?;
        b.hc_combine(&mut ws.r, &ws.y, &ws.inj, combine)?;
        self.hc_pre(b, &m.w.hc_ffn, ws, t)?;
        std::mem::swap(&mut ws.moe_scratch, &mut ms.moe_scratch);
        let moe = self.moe(b, 0, &m.w.moe, &mut m.cache, ws, t);
        std::mem::swap(&mut ws.moe_scratch, &mut ms.moe_scratch);
        moe?;
        b.hc_combine(&mut ws.r, &ws.y, &ws.inj, combine)?;
        // The MTP head mixer on the last entry, then the shared output head.
        b.copy(&ws.r, (t - 1) * wide, &mut ws.last, 0, wide)?;
        let hc = &m.w.hc_head;
        b.rms_norm_groups(&ws.last, &hc.norm, &mut ws.xn, 1, st, h, eps)?;
        lin(b, &ws.xn, &hc.down, &mut ws.lo, 1)?;
        b.silu(&mut ws.lo, c.hc_rank)?;
        lin(b, &ws.lo, &hc.up, &mut ws.gate, 1)?;
        b.hc_mix(&ws.xn, &ws.gate, &mut ws.mixed, 1, st, h)?;
        lin(b, &ws.mixed, &self.w.output, &mut ws.logits, 1)
    }
}
