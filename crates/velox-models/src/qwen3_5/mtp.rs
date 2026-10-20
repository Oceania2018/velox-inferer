//! MTP draft predictor (`Qwen3_5MTP`): one attention block conditioned on the target's
//! final-normed hidden state `h_t` and the embedding of the following token `x_(t+1)`:
//!
//! ```text
//! u = input_projection(concat(norm_e(embed(x_(t+1))), norm_h(h_t)))
//! u = block(u)                      // the target's attention + FFN geometry, private weights/KV
//! draft_hidden = final_norm(u)      // predicts x_(t+2); feeds the next MTP step recursively
//! ```
//!
//! MTP row `r` therefore pairs target row `r` with token `r + 1`, at row `r`'s rotary position,
//! and writes MTP KV row `r`. Draft logits come from the proposal head (a vocabulary subset)
//! when the artifact has one, otherwise from the target's output head.

use velox_core::backend::{Backend, NormKind, WeightView};
use velox_core::{Error, Result};

use super::weights::MtpWeights;
use super::{Qwen35Text, Rope, Workspace};

/// MTP attention cache, one row per target cache row.
pub struct MtpSession<B: Backend> {
    kv: (B::Buffer, B::Buffer),
    pub capacity: usize,
}

/// Activations of MTP calls of up to `rows` rows, plus a block workspace of its own so that the
/// target's workspace (whose last row feeds the next target step) is never touched.
pub struct MtpWorkspace<B: Backend> {
    rows: usize,
    block: Workspace<B>,
    e_in: B::Buffer,
    e: B::Buffer,
    h_in: B::Buffer,
    h: B::Buffer,
    cat: B::Buffer,
    /// Draft hidden rows of the latest `mtp_rows` (`[rows, hidden]`).
    pub out: B::Buffer,
    last: B::Buffer,
    logits: B::Buffer,
}

/// The token embeddings an MTP call pairs with its hidden rows.
pub enum MtpNext<'a, Buf> {
    /// Token ids, embedded here (decode: drafts, accepted tokens).
    Ids(&'a [u32]),
    /// Rows of already composed embeddings (prefill: image placeholders carry vision rows).
    Rows { buf: &'a Buf, row0: usize },
}

impl<B: Backend> Qwen35Text<B> {
    pub fn has_mtp(&self) -> bool {
        self.w.mtp.is_some()
    }

    pub fn mtp_session(&self, b: &B, capacity: usize) -> Result<MtpSession<B>> {
        let kv = self.cfg.kv_dim();
        Ok(MtpSession {
            kv: (b.alloc(capacity * kv)?, b.alloc(capacity * kv)?),
            capacity,
        })
    }

    pub fn mtp_workspace(&self, b: &B, rows: usize) -> Result<MtpWorkspace<B>> {
        let h = self.cfg.hidden;
        let vocab = self
            .w
            .proposal
            .as_ref()
            .map_or(self.cfg.vocab, |p| p.token_ids.len());
        Ok(MtpWorkspace {
            rows,
            block: self.workspace(b, rows)?,
            e_in: b.alloc(rows * h)?,
            e: b.alloc(rows * h)?,
            h_in: b.alloc(rows * h)?,
            h: b.alloc(rows * h)?,
            cat: b.alloc(rows * 2 * h)?,
            out: b.alloc(rows * h)?,
            last: b.alloc(h)?,
            logits: b.alloc(vocab)?,
        })
    }

    /// Runs `t` MTP rows at cache rows `pos0..pos0 + t`: hidden rows `hidden[hidden_row0..]`
    /// paired with `next`. Draft hidden rows land in `mw.out`.
    #[allow(clippy::too_many_arguments)]
    pub fn mtp_rows(
        &self,
        b: &B,
        ms: &mut MtpSession<B>,
        mw: &mut MtpWorkspace<B>,
        hidden: &B::Buffer,
        hidden_row0: usize,
        next: MtpNext<'_, B::Buffer>,
        t: usize,
        pos0: usize,
        rope: MtpRope<'_>,
    ) -> Result<()> {
        let m: &MtpWeights = self
            .w
            .mtp
            .as_ref()
            .ok_or_else(|| Error::Config("artifact has no MTP component".into()))?;
        let (c, w) = (&self.cfg, &self.w);
        let hd = c.hidden;
        if t == 0 {
            return Ok(());
        }
        if t > mw.rows || pos0 + t > ms.capacity {
            return Err(Error::Shape(format!(
                "mtp_rows: {t} rows at {pos0} (workspace {}, capacity {})",
                mw.rows, ms.capacity
            )));
        }
        match next {
            MtpNext::Ids(ids) => {
                if ids.len() != t {
                    return Err(Error::Shape("mtp_rows: one next token per row".into()));
                }
                b.embed(ids, &w.view(&w.embed), &mut mw.e_in)?;
            }
            MtpNext::Rows { buf, row0 } => b.copy(buf, row0 * hd, &mut mw.e_in, 0, t * hd)?,
        }
        b.copy(hidden, hidden_row0 * hd, &mut mw.h_in, 0, t * hd)?;
        b.rms_norm(
            &mw.e_in,
            w.param(m.embedding_norm),
            &mut mw.e,
            t,
            hd,
            c.eps,
            NormKind::Offset,
        )?;
        b.rms_norm(
            &mw.h_in,
            w.param(m.hidden_norm),
            &mut mw.h,
            t,
            hd,
            c.eps,
            NormKind::Offset,
        )?;
        b.concat_cols(&mw.e, &mw.h, &mut mw.cat, t, hd, hd)?;
        let ws = &mut mw.block;
        b.linear(&mw.cat, &w.view(&m.input_projection), &mut ws.x, t)?;
        let l = &m.layer;
        b.rms_norm(
            &ws.x,
            w.param(l.input_norm),
            &mut ws.h,
            t,
            hd,
            c.eps,
            NormKind::Offset,
        )?;
        let axes: Vec<u32>;
        let rope = match rope {
            MtpRope::At(p) => Rope::At(p),
            MtpRope::Positions(p) => {
                axes = (0..3)
                    .flat_map(|a| p.iter().map(move |tri| tri[a]))
                    .collect();
                Rope::Axes(&axes)
            }
        };
        self.attention_mixer(b, ws, &l.mixer, &mut ms.kv, t, pos0, rope)?;
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
        b.rms_norm(
            &ws.x,
            w.param(m.final_norm),
            &mut mw.out,
            t,
            hd,
            c.eps,
            NormKind::Offset,
        )
    }

    /// The `k` best draft candidates (token id, logit) from draft hidden row `row` of `mw.out`,
    /// among vocabulary ids below `limit`.
    pub fn mtp_draft(
        &self,
        b: &B,
        mw: &mut MtpWorkspace<B>,
        row: usize,
        k: usize,
        limit: usize,
    ) -> Result<Vec<(u32, f32)>> {
        let w = &self.w;
        let hd = self.cfg.hidden;
        b.copy(&mw.out, row * hd, &mut mw.last, 0, hd)?;
        let (head, map): (WeightView<'_, B::Buffer>, Option<&[u32]>) = match &w.proposal {
            Some(p) => (w.view(&p.head), Some(&p.token_ids)),
            None => (w.view(&w.head), None),
        };
        b.linear(&mw.last, &head, &mut mw.logits, 1)?;
        let n = head.rows;
        // With a proposal map, ids above `limit` are filtered after mapping, so fetch a margin.
        let cands = b.top_k(
            &mw.logits,
            1,
            n,
            if map.is_some() { n } else { limit.min(n) },
            k + 8,
        )?;
        Ok(cands
            .into_iter()
            .next()
            .unwrap_or_default()
            .into_iter()
            .map(|(i, v)| (map.map_or(i, |m| m[i as usize]), v))
            .filter(|&(id, _)| (id as usize) < limit)
            .take(k)
            .collect())
    }
}

/// Rotary positions of MTP rows: consecutive from a start (`cache row + rope_delta`), or one
/// (t, h, w) triple per row (prompts with images).
#[derive(Clone, Copy)]
pub enum MtpRope<'a> {
    At(usize),
    Positions(&'a [[u32; 3]]),
}
