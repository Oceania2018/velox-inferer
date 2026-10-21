//! Speculative decoding with the MTP draft model.
//!
//! Between rounds (target session at cache position `P`, token `c` sampled but not yet fed):
//! MTP KV holds rows `0..P` computed from the target's own hidden states, except right after a
//! prefill, when row `P - 1` is the *carry*: its pairing token `c` was not known yet. A round
//! verifies `[c, drafts..]` on the target, keeps the matching prefix (`commit`), then re-runs MTP
//! over the kept rows with the target's hidden states (repairing the KV rows that recursive
//! drafting filled with draft states) and drafts the next round from the last of them.

use velox_core::backend::Backend;
use velox_core::rng::Rng;
use velox_core::Result;
use velox_models::qwen3_5::{
    DFlashModel, DFlashState, DFlashWorkspace, MtpNext, MtpRope, MtpSession, MtpWorkspace,
    Qwen35Text, Session, Workspace,
};

/// The last prompt row, waiting for the first sampled token.
struct Carry {
    row: usize,
    pos: Option<[u32; 3]>,
}

pub struct MtpDrafter<B: Backend> {
    /// Drafts per round right now; adapts to the acceptance rate within `2..=max_drafts`.
    pub draft_len: usize,
    pub max_drafts: usize,
    /// Moving average of the per-draft acceptance rate.
    accept_ema: f64,
    session: MtpSession<B>,
    mw: MtpWorkspace<B>,
    /// Hidden input of the next MTP step: the carry row, or the previous draft hidden row.
    prev: B::Buffer,
    carry: Option<Carry>,
}

impl<B: Backend> MtpDrafter<B> {
    pub fn new(
        b: &B,
        model: &Qwen35Text<B>,
        capacity: usize,
        rows: usize,
        draft_len: usize,
    ) -> Result<Self> {
        Ok(MtpDrafter {
            draft_len: draft_len.min(3).max(1),
            max_drafts: draft_len.max(1),
            accept_ema: 0.6,
            session: model.mtp_session(b, capacity)?,
            mw: model.mtp_workspace(b, rows.max(velox_models::qwen3_5::VERIFY_MAX))?,
            prev: b.alloc(model.cfg.hidden)?,
            carry: None,
        })
    }

    /// Records a round's outcome and picks the next draft length: with per-draft acceptance `a`,
    /// the expected accepted run is about `a / (1 - a)`, so drafting one more than that wastes
    /// little when drafts are poor and lets long runs through when they are good (structured
    /// output often accepts > 90%).
    pub fn record(&mut self, accepted: usize, proposed: usize) {
        if proposed == 0 {
            return;
        }
        let rate = accepted as f64 / proposed as f64;
        self.accept_ema = 0.7 * self.accept_ema + 0.3 * rate;
        let a = self.accept_ema.min(0.98);
        let want = (a / (1.0 - a)).ceil() as usize + 1;
        self.draft_len = want.clamp(2.min(self.max_drafts), self.max_drafts);
    }

    /// Forgets the carry (a new prompt is about to be prefilled from `row`; MTP KV rows below it
    /// are still valid when the prefix was reused).
    pub fn reset(&mut self) {
        self.carry = None;
    }

    /// Feeds the MTP rows of one target prefill chunk: `ws.emb` / `ws.rows_hidden` hold `t` rows
    /// starting at cache row `row0`, whose rotary positions are `positions` (one triple per row)
    /// or `row + rope_delta`.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_chunk(
        &mut self,
        b: &B,
        model: &Qwen35Text<B>,
        ws: &Workspace<B>,
        row0: usize,
        t: usize,
        rope_delta: i64,
        positions: Option<&[[u32; 3]]>,
    ) -> Result<()> {
        if t == 0 {
            return Ok(());
        }
        let hd = model.cfg.hidden;
        let at = |row: usize| MtpRope::At((row as i64 + rope_delta).max(0) as usize);
        // The previous chunk's last row pairs with this chunk's first token.
        if let Some(c) = self.carry.take() {
            let pos = [c.pos.unwrap_or_default()];
            let rope = if c.pos.is_some() {
                MtpRope::Positions(&pos)
            } else {
                at(c.row)
            };
            let next = MtpNext::Rows {
                buf: &ws.emb,
                row0: 0,
            };
            model.mtp_rows(
                b,
                &mut self.session,
                &mut self.mw,
                &self.prev,
                0,
                next,
                1,
                c.row,
                rope,
            )?;
        }
        if t > 1 {
            let rope = match positions {
                Some(p) => MtpRope::Positions(&p[..t - 1]),
                None => at(row0),
            };
            let next = MtpNext::Rows {
                buf: &ws.emb,
                row0: 1,
            };
            model.mtp_rows(
                b,
                &mut self.session,
                &mut self.mw,
                &ws.rows_hidden,
                0,
                next,
                t - 1,
                row0,
                rope,
            )?;
        }
        b.copy(&ws.rows_hidden, (t - 1) * hd, &mut self.prev, 0, hd)?;
        self.carry = Some(Carry {
            row: row0 + t - 1,
            pos: positions.map(|p| p[t - 1]),
        });
        Ok(())
    }

    /// Drafts after `pending`, the token just sampled. First pairs the rows that still lack MTP
    /// state with their next tokens: the prefill carry, or `kept` hidden rows of the last verify
    /// (`ws.hidden`, cache rows from `row0`, followed by `next[..]` then `pending`).
    #[allow(clippy::too_many_arguments)]
    pub fn draft(
        &mut self,
        b: &B,
        model: &Qwen35Text<B>,
        s: &Session<B>,
        ws: &Workspace<B>,
        row0: usize,
        kept_next: &[u32],
        pending: u32,
        limit: usize,
    ) -> Result<Vec<u32>> {
        let hd = model.cfg.hidden;
        let delta = s.rope_delta;
        let at = |row: usize| MtpRope::At((row as i64 + delta).max(0) as usize);
        // Catch-up rows; the last one pairs with `pending` and yields the first draft. `row` is the
        // cache row of the next MTP step, `last` the output row the next draft comes from.
        let (mut row, mut last) = match self.carry.take() {
            Some(c) => {
                let pos = [c.pos.unwrap_or_default()];
                let rope = if c.pos.is_some() {
                    MtpRope::Positions(&pos)
                } else {
                    at(c.row)
                };
                let ids = [pending];
                model.mtp_rows(
                    b,
                    &mut self.session,
                    &mut self.mw,
                    &self.prev,
                    0,
                    MtpNext::Ids(&ids),
                    1,
                    c.row,
                    rope,
                )?;
                (c.row + 1, 0)
            }
            None => {
                let mut ids = kept_next.to_vec();
                ids.push(pending);
                let t = ids.len();
                model.mtp_rows(
                    b,
                    &mut self.session,
                    &mut self.mw,
                    &ws.hidden,
                    0,
                    MtpNext::Ids(&ids),
                    t,
                    row0,
                    at(row0),
                )?;
                (row0 + t, t - 1)
            }
        };
        // Recursive drafts: each step pairs the previous draft hidden row with the draft it chose.
        let mut drafts = Vec::with_capacity(self.draft_len);
        while drafts.len() < self.draft_len {
            let Some(&(d, _)) = model.mtp_draft(b, &mut self.mw, last, 1, limit)?.first() else {
                break;
            };
            drafts.push(d);
            if drafts.len() == self.draft_len || row >= self.session.capacity {
                break;
            }
            b.copy(&self.mw.out, last * hd, &mut self.prev, 0, hd)?;
            model.mtp_rows(
                b,
                &mut self.session,
                &mut self.mw,
                &self.prev,
                0,
                MtpNext::Ids(&[d]),
                1,
                row,
                at(row),
            )?;
            last = 0;
            row += 1;
        }
        Ok(drafts)
    }
}

/// Speculative decoding with the DFlash2 block drafter: one masked-block forward proposes
/// `drafts` tokens, each with the proposal distribution it was drawn from, so verification can
/// use min(1, p / q) and the residual max(p - q, 0).
/// The short DFlash block (drafts per round) every request starts with; see [`DFlashDrafter::observe`].
const ADAPT_LOW: usize = 7;

pub struct DFlashDrafter<B: Backend> {
    /// Drafts in the next block (adapted by [`Self::observe`] between `low` and `max`).
    pub drafts: usize,
    max: usize,
    low: usize,
    /// Running mean of accepted drafts per round.
    accepted: f32,
    state: DFlashState<B>,
    dw: DFlashWorkspace<B>,
    rng: Rng,
}

impl<B: Backend> DFlashDrafter<B> {
    pub fn new(
        b: &B,
        dm: &DFlashModel<B>,
        target: &Qwen35Text<B>,
        ctx_rows: usize,
        drafts: usize,
    ) -> Result<Self> {
        let max = drafts.clamp(1, velox_models::qwen3_5::VERIFY_MAX - 1);
        let low = max.min(ADAPT_LOW);
        Ok(DFlashDrafter {
            drafts: low,
            max,
            low,
            accepted: low as f32,
            state: dm.state(b)?,
            dw: dm.workspace(b, target, ctx_rows, max + 1)?,
            rng: Rng::new(0x5eed_df1a),
        })
    }

    /// Drops context rows from `pos` on (the target state there was replaced).
    pub fn forget(&mut self, dm: &DFlashModel<B>, pos: usize) {
        dm.rewind(&mut self.state, pos);
    }

    /// Starts a request at the short block.
    pub fn reset(&mut self) {
        self.drafts = self.low;
        self.accepted = self.low as f32;
    }

    /// Adapts the block to how many drafts the last round accepted: the long block while nearly
    /// every draft of the short one would be accepted (predictable text: more tokens per round for
    /// about the same verify cost), else the short one (a longer window then only costs more).
    pub fn observe(&mut self, accepted: usize) {
        self.accepted = 0.5 * self.accepted + 0.5 * accepted as f32;
        self.drafts = if self.accepted >= self.low as f32 - 0.5 {
            self.max
        } else {
            self.low
        };
    }

    /// Context for one target prefill chunk: `t` rows at cache rows `row0 ..` whose features the
    /// target captured in `ws`.
    pub fn prefill_chunk(
        &mut self,
        b: &B,
        dm: &DFlashModel<B>,
        ws: &Workspace<B>,
        row0: usize,
        t: usize,
    ) -> Result<()> {
        if t > dm.cfg.window {
            return Err(velox_core::Error::Shape(format!(
                "prefill chunk of {t} rows exceeds the DFlash window"
            )));
        }
        dm.materialize(b, &mut self.state, &mut self.dw, ws, row0, t)
    }

    /// Materializes the `kept` rows of the last verify (cache rows from `row0`, features in `ws`),
    /// aligns the context with the target `frontier`, and drafts after `anchor`. Returns the drafts
    /// and the proposal distribution of each.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn draft(
        &mut self,
        b: &B,
        dm: &DFlashModel<B>,
        target: &Qwen35Text<B>,
        ws: &Workspace<B>,
        row0: usize,
        kept: usize,
        frontier: usize,
        anchor: u32,
        limit: usize,
        temperature: f32,
    ) -> Result<(Vec<u32>, Vec<Vec<(u32, f32)>>)> {
        if kept > 0 {
            dm.materialize(b, &mut self.state, &mut self.dw, ws, row0, kept)?;
        }
        let st = &mut self.state;
        if st.end > frontier {
            dm.rewind(st, frontier);
        } else if st.end < frontier {
            // Rows without features (should not happen): continue with an empty context.
            st.lo = frontier;
            st.end = frontier;
        }
        let bc = dm.propose(b, target, st, &mut self.dw, anchor, self.drafts, limit)?;
        let rng = &mut self.rng;
        Ok(dm.select(&bc, anchor, temperature, &mut || rng.uniform() * 0.5 + 0.5))
    }
}

/// The main lane's draft model.
pub enum Drafter<B: Backend> {
    Mtp(MtpDrafter<B>),
    DFlash(DFlashDrafter<B>),
}
