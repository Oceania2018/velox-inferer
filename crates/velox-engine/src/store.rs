//! Context store: prompt prefixes kept in system RAM so a lane can resume any of them after other
//! conversations have overwritten its state (several agents, task types and side requests sharing
//! one server).
//!
//! An entry is a recurrent-state checkpoint plus the attention KV rows below it. KV is held as
//! segments shared between entries: an entry that extends another reuses its segments and adds
//! one for the rows after it, so a chain of agent steps costs one state and one step of KV each.
//!
//! Retention follows ninfer's context cache (docs/maintainer/cua-context-checkpoints.md §4):
//! - LRFU ranking (Lee et al., 2001): `CRF(t) = Σ 2^(-λ (t - t_i))` over an entry's references,
//!   stored as the demand `t_last + log2(CRF(t_last)) / λ` (λ = 0 ranks by recency alone). The
//!   clock advances once per request.
//! - Ancestor references: a request references every entry on its prefix, so a shared system
//!   prompt or task stays hot while each run resumes from its own deeper history.
//! - Roots (system prompt, task) are never superseded.
//! - Head protection: a chain's new head inherits the reference history of the point its request
//!   resumed from, so one-off requests cannot evict it as the coldest entry.
//! - Supersession: that point, if not a root and adopted only once, is evicted before anything
//!   else (its KV stays shared with the new head).
//!
//! An optional NVMe tier ([`DiskTier`]) keeps copies of evicted entries that can still be
//! resumed.

use std::collections::HashSet;
use std::sync::Arc;

use velox_core::{Backend, Error as CoreError};
use velox_models::qwen3_5::{Checkpoint, KvSegment, Session};

use crate::disk::DiskTier;

pub(crate) struct Entry<B: Backend> {
    pub id: u64,
    /// Keys of tokens `0..pos`.
    pub keys: Vec<u32>,
    /// Recurrent state at `pos`, in system RAM.
    pub state: Checkpoint<B>,
    /// Contiguous KV segments covering rows `0..pos`.
    pub kv: Vec<Arc<KvSegment<B>>>,
    root: bool,
    adopted: u32,
    superseded: bool,
    last: u64,
    crf: f64,
}

impl<B: Backend> Entry<B> {
    pub fn pos(&self) -> usize {
        self.keys.len()
    }

    fn prefix_of(&self, keys: &[u32]) -> bool {
        keys.len() >= self.keys.len() && keys[..self.keys.len()] == self.keys[..]
    }
}

pub(crate) struct ContextStore<B: Backend> {
    /// Declared before `inflight`: dropping it joins the writer that reads those entries.
    disk: Option<DiskTier>,
    /// Evicted entries being copied to disk, by record id (their memory must stay put).
    inflight: Vec<(u64, Entry<B>)>,
    entries: Vec<Entry<B>>,
    /// Storage of evicted states and KV pages, reused for new ones: pinning memory costs about
    /// 0.4 ms per MB, ten times the copy.
    spare: Vec<Checkpoint<B>>,
    pages: Vec<B::Buffer>,
    budget: usize,
    lambda: f64,
    now: u64,
    next_id: u64,
}

/// Evicted state storage kept for reuse.
const MAX_SPARE: usize = 4;

impl<B: Backend> ContextStore<B> {
    /// A store of at most `budget` bytes ranking entries by LRFU with decay `lambda` per request.
    pub fn new(budget: usize, lambda: f64, disk: Option<DiskTier>) -> Self {
        ContextStore {
            disk,
            inflight: Vec::new(),
            entries: Vec::new(),
            spare: Vec::new(),
            pages: Vec::new(),
            budget,
            lambda: lambda.max(0.0),
            now: 0,
            next_id: 1,
        }
    }

    pub fn enabled(&self) -> bool {
        self.budget > 0
    }

    /// Advances the LRFU clock (once per request) and releases entries whose disk copy is done.
    pub fn tick(&mut self) {
        self.now += 1;
        let Some(disk) = self.disk.as_mut() else {
            return;
        };
        for id in disk.completed() {
            if let Some(i) = self.inflight.iter().position(|(r, _)| *r == id) {
                let (_, e) = self.inflight.swap_remove(i);
                Self::release(&mut self.spare, &mut self.pages, e);
            }
        }
    }

    /// Reads the deepest disk record on the prefix of `keys` (at most `limit` tokens) into the
    /// store when it covers at least `gain` tokens more than `resident`, the best position
    /// available without it. Returns whether it did.
    pub fn promote(
        &mut self,
        b: &B,
        session: &Session<B>,
        keys: &[u32],
        limit: usize,
        resident: usize,
        gain: usize,
    ) -> Result<bool, CoreError> {
        let Some(disk) = self.disk.as_mut() else {
            return Ok(false);
        };
        let Some(r) = disk.best(keys, limit) else {
            return Ok(false);
        };
        if r.keys.len() < resident + gain {
            return Ok(false);
        }
        let (id, rkeys, rope_delta, segs, root) =
            (r.id, r.keys.clone(), r.rope_delta, r.segs.clone(), r.root);
        let pos = rkeys.len();
        let mut state = match self.spare.pop() {
            Some(s) => s,
            None => session.blank_checkpoint_in(b, pos, true)?,
        };
        let mut kv = Vec::with_capacity(segs.len());
        let (pool, page_len) = (&mut self.pages, session.kv_page_len());
        let mut page = || pool.pop().map_or_else(|| b.alloc_host(page_len), Ok);
        for &(lo, hi) in &segs {
            kv.push(session.blank_kv(b, lo, hi, &mut page)?);
        }
        let read = {
            let mut bufs: Vec<&mut [u8]> = Vec::new();
            for buf in state
                .buffers_mut()
                .chain(kv.iter_mut().flat_map(|s| s.buffers_mut()))
            {
                match b.host_bytes_mut(buf) {
                    Some(x) => bufs.push(x),
                    None => return Ok(false),
                }
            }
            disk.read(id, &mut bufs, self.now)
        };
        if let Err(e) = read {
            eprintln!("disk context: read failed: {e}");
            self.spare.push(state);
            return Ok(false);
        }
        state.pos = pos;
        state.rope_delta = rope_delta;
        let id = self.next_id;
        self.next_id += 1;
        self.entries.push(Entry {
            id,
            keys: rkeys,
            state,
            kv: kv.into_iter().map(Arc::new).collect(),
            root,
            adopted: 0,
            superseded: false,
            last: self.now,
            crf: 1.0,
        });
        self.evict(b, id)?;
        Ok(true)
    }

    /// The deepest entry position on the prefix of `keys` at or below `limit` (0 when none).
    pub fn best_pos(&self, keys: &[u32], limit: usize) -> usize {
        self.best(keys, limit).map_or(0, |e| e.pos())
    }

    /// Disk records ready to be read back.
    pub fn disk_len(&self) -> usize {
        self.disk.as_ref().map_or(0, |d| d.len())
    }

    /// The deepest entry on the prefix of `keys` at or below `limit` tokens.
    pub fn best(&self, keys: &[u32], limit: usize) -> Option<&Entry<B>> {
        self.entries
            .iter()
            .filter(|e| e.pos() <= limit && e.prefix_of(keys))
            .max_by_key(|e| e.pos())
    }

    fn demand(&self, e: &Entry<B>) -> f64 {
        if self.lambda == 0.0 {
            e.last as f64
        } else {
            e.last as f64 + e.crf.log2() / self.lambda
        }
    }

    fn reference(lambda: f64, now: u64, e: &mut Entry<B>) {
        let decay = (-(lambda * (now - e.last) as f64)).exp2();
        e.crf = 1.0 + e.crf * decay;
        e.last = now;
    }

    /// A request over `keys` resumed at `start`: references every live entry on that prefix
    /// (ancestor references), counts an adoption of `adopted` (the entry it was restored from),
    /// and returns the deepest such entry, the parent of what the request captures next.
    pub fn resumed(&mut self, keys: &[u32], start: usize, adopted: Option<u64>) -> Option<u64> {
        let (lambda, now) = (self.lambda, self.now);
        let mut deepest: Option<(usize, u64)> = None;
        for e in &mut self.entries {
            if e.pos() <= start && !e.superseded && e.prefix_of(keys) {
                Self::reference(lambda, now, e);
                if Some(e.id) == adopted {
                    e.adopted += 1;
                }
                if deepest.is_none_or(|(p, _)| e.pos() > p) {
                    deepest = Some((e.pos(), e.id));
                }
            }
        }
        deepest.map(|(_, id)| id)
    }

    /// Stores `cp` (state of `session` at `cp.pos`, whose KV rows below it are intact) under
    /// `keys[..cp.pos]`. A `root` is a shared anchor (system prompt, task); otherwise `chain` is
    /// the entry the request that produced `cp` resumed from: the new head inherits its demand
    /// and supersedes it. Returns the entry id.
    pub fn insert(
        &mut self,
        b: &B,
        session: &Session<B>,
        keys: &[u32],
        cp: &Checkpoint<B>,
        root: bool,
        chain: Option<u64>,
    ) -> Result<u64, CoreError> {
        let pos = cp.pos;
        let keys = &keys[..pos];
        let (lambda, now) = (self.lambda, self.now);
        if let Some(e) = self.entries.iter_mut().find(|e| e.keys == keys) {
            Self::reference(lambda, now, e);
            e.root |= root;
            return Ok(e.id);
        }
        // KV below the deepest stored prefix is shared with it.
        let parent = self
            .entries
            .iter()
            .filter(|e| e.pos() < pos && e.prefix_of(keys))
            .max_by_key(|e| e.pos());
        let mut kv = parent.map(|p| p.kv.clone()).unwrap_or_default();
        let lo = parent.map_or(0, |p| p.pos());
        let (pool, page_len) = (&mut self.pages, session.kv_page_len());
        let mut page = || pool.pop().map_or_else(|| b.alloc_host(page_len), Ok);
        kv.push(Arc::new(session.save_kv(b, lo, pos, &mut page)?));
        let mut state = match self.spare.pop() {
            Some(s) => s,
            None => session.blank_checkpoint_in(b, pos, true)?,
        };
        cp.copy_into(b, &mut state)?;
        let id = self.next_id;
        self.next_id += 1;
        let mut e = Entry {
            id,
            keys: keys.to_vec(),
            state,
            kv,
            root,
            adopted: 0,
            superseded: false,
            last: now,
            crf: 1.0,
        };
        if !root {
            if let Some(p) = chain.and_then(|c| self.entries.iter_mut().find(|e| e.id == c)) {
                // Head protection: continue the resumed point's reference history.
                e.crf = p.crf;
                e.last = p.last;
                Self::reference(lambda, now, &mut e);
                if !p.root && p.adopted <= 1 {
                    p.superseded = true;
                }
            }
        }
        self.entries.push(e);
        self.evict(b, id)?;
        Ok(id)
    }

    /// Bytes held: states plus distinct KV segments.
    pub fn bytes(&self, b: &B) -> usize {
        let mut seen = HashSet::new();
        let mut total = 0;
        for e in &self.entries {
            total += e.state.bytes(b);
            for s in &e.kv {
                if seen.insert(Arc::as_ptr(s)) {
                    total += s.bytes(b);
                }
            }
        }
        total
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Evicts superseded entries first, then the lowest demand, until the store fits its budget.
    /// `keep` (the entry just stored) is never evicted.
    fn evict(&mut self, b: &B, keep: u64) -> Result<(), CoreError> {
        let mut synced = false;
        while self.bytes(b) > self.budget {
            let victim = self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, e)| e.id != keep)
                .min_by(|(_, x), (_, y)| {
                    (!x.superseded, self.demand(x))
                        .partial_cmp(&(!y.superseded, self.demand(y)))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i);
            let Some(i) = victim else { break };
            if !synced {
                // Pending copies may still read the storage about to be freed.
                b.synchronize()?;
                synced = true;
            }
            let e = self.entries.swap_remove(i);
            if let Some(e) = self.spill(b, e) {
                Self::release(&mut self.spare, &mut self.pages, e);
            }
        }
        Ok(())
    }

    /// Returns an entry's storage to the pools (KV pages once no other entry shares them).
    fn release(spare: &mut Vec<Checkpoint<B>>, pages: &mut Vec<B::Buffer>, e: Entry<B>) {
        if spare.len() < MAX_SPARE {
            spare.push(e.state);
        }
        for seg in e.kv {
            if let Ok(seg) = Arc::try_unwrap(seg) {
                pages.extend(seg.into_pages());
            }
        }
    }

    /// Queues an evicted entry that can still be resumed for the disk tier; returns it back when
    /// it is not written (no tier, superseded, already on disk, no room, device storage).
    fn spill(&mut self, b: &B, e: Entry<B>) -> Option<Entry<B>> {
        let Some(disk) = self.disk.as_mut() else {
            return Some(e);
        };
        if e.superseded || disk.holds(&e.keys) {
            return Some(e);
        }
        let parts: Option<Vec<(*const u8, usize)>> = e
            .state
            .buffers()
            .chain(e.kv.iter().flat_map(|s| s.buffers()))
            .map(|buf| b.host_bytes(buf).map(|x| (x.as_ptr(), x.len())))
            .collect();
        let Some(parts) = parts else {
            return Some(e);
        };
        let segs = e.kv.iter().map(|s| (s.lo, s.hi)).collect();
        match disk.write(&e.keys, e.state.rope_delta, segs, e.root, parts, self.now) {
            Some(id) => {
                self.inflight.push((id, e));
                None
            }
            None => Some(e),
        }
    }

    /// Summary for debug logs: `pos:demand` per entry, `*` for roots, `~` for superseded.
    pub fn describe(&self) -> String {
        let mut v: Vec<&Entry<B>> = self.entries.iter().collect();
        v.sort_by_key(|e| e.pos());
        v.iter()
            .map(|e| {
                format!(
                    "{}{}:{:.0}",
                    if e.root {
                        "*"
                    } else if e.superseded {
                        "~"
                    } else {
                        ""
                    },
                    e.pos(),
                    self.demand(e)
                )
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn lrfu_demand_matches_crf_ranking() {
        // Two references 10 events apart at λ = 0.1 rank above one reference at the later time.
        let (lambda, t1, t2) = (0.1f64, 0u64, 10u64);
        let crf = 1.0 + (-(lambda * (t2 - t1) as f64)).exp2();
        let demand = t2 as f64 + crf.log2() / lambda;
        assert!(demand > t2 as f64);
        // ... and below a single reference 10 events later still (bounded bonus).
        assert!(demand < (t2 + 10) as f64);
    }
}
