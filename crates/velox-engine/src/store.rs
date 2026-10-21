//! Context store: prompt prefixes kept in system RAM so a lane can resume any of them after other
//! conversations have overwritten its state (several agents, task types and side requests sharing
//! one server).
//!
//! An entry is a recurrent-state checkpoint plus the attention KV rows below it, held as pages of
//! [`KV_PAGE_ROWS`] positions. A full page is addressed by the keys of the whole prefix up to its
//! end and shared by every entry on that prefix whatever order they were stored in, so a chain of
//! agent steps costs one state and about one step of KV each.
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

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Weak};

use velox_core::{Backend, Error as CoreError};
use velox_models::qwen3_5::{
    Checkpoint, DFlashModel, DFlashState, DraftContext, KvSegment, Session, KV_PAGE_ROWS,
};

use crate::disk::DiskTier;

pub(crate) struct Entry<B: Backend> {
    pub id: u64,
    /// Keys of tokens `0..pos`.
    pub keys: Vec<u32>,
    /// Recurrent state at `pos`, in system RAM.
    pub state: Checkpoint<B>,
    /// KV pages covering rows `0..pos` in order (page `i` holds rows `i * KV_PAGE_ROWS ..`).
    pub kv: Vec<Arc<KvSegment<B>>>,
    /// The DFlash drafter's context rows ending at `pos`, when it had them.
    pub draft: Option<DraftContext<B>>,
    root: bool,
    /// A prompt's end (see `ContextStore::mark_tail`).
    tail: bool,
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
    draft_pages: Vec<B::Buffer>,
    /// Full KV pages by the hash of their whole prefix (see [`page_hashes`]).
    shared: HashMap<u64, Weak<KvSegment<B>>>,
    budget: usize,
    lambda: f64,
    now: u64,
    next_id: u64,
}

/// Evicted state storage kept for reuse.
const MAX_SPARE: usize = 4;

/// Storage [`ContextStore::top_up`] keeps ready: two states (a chain head and an anchor) and
/// 8K positions of KV pages.
const RESERVE_STATES: usize = 2;
const RESERVE_PAGES: usize = 32;

impl<B: Backend> ContextStore<B> {
    /// A store of at most `budget` bytes ranking entries by LRFU with decay `lambda` per request.
    pub fn new(budget: usize, lambda: f64, disk: Option<DiskTier>) -> Self {
        ContextStore {
            disk,
            inflight: Vec::new(),
            entries: Vec::new(),
            spare: Vec::new(),
            pages: Vec::new(),
            draft_pages: Vec::new(),
            shared: HashMap::new(),
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
                self.release(e);
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
        dm: Option<&DFlashModel<B>>,
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
        let (id, rkeys, rope_delta, segs, root, draft_rows) = (
            r.id,
            r.keys.clone(),
            r.rope_delta,
            r.segs.clone(),
            r.root,
            r.draft,
        );
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
        // A record written with drafter context is read whole, so it needs a DFlash drafter.
        let mut draft = match (draft_rows, dm) {
            (Some((lo, hi)), Some(dm)) => {
                let (pool, len) = (&mut self.draft_pages, dm.context_page_len());
                let mut page = || pool.pop().map_or_else(|| b.alloc_host(len), Ok);
                Some(dm.blank_context(b, lo, hi, &mut page)?)
            }
            (Some(_), None) => return Ok(false),
            _ => None,
        };
        let read = {
            let mut bufs: Vec<&mut [u8]> = Vec::new();
            for buf in state
                .buffers_mut()
                .chain(kv.iter_mut().flat_map(|s| s.buffers_mut()))
                .chain(draft.iter_mut().flat_map(|d| d.buffers_mut()))
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
        // Pages another entry already holds replace the copies just read.
        let hashes = page_hashes(&rkeys);
        let mut pages = Vec::with_capacity(kv.len());
        for seg in kv {
            let i = seg.lo / KV_PAGE_ROWS;
            match hashes.get(i).filter(|_| seg.hi - seg.lo == KV_PAGE_ROWS) {
                Some(&h) => match self.shared.get(&h).and_then(Weak::upgrade) {
                    Some(old) => {
                        self.pages.extend(seg.into_pages());
                        pages.push(old);
                    }
                    None => {
                        let seg = Arc::new(seg);
                        self.shared.insert(h, Arc::downgrade(&seg));
                        pages.push(seg);
                    }
                },
                None => pages.push(Arc::new(seg)),
            }
        }
        let id = self.next_id;
        self.next_id += 1;
        self.entries.push(Entry {
            id,
            keys: rkeys,
            state,
            kv: pages,
            draft,
            root,
            tail: false,
            adopted: 0,
            superseded: false,
            last: self.now,
            crf: 1.0,
        });
        self.evict(b, id)?;
        Ok(true)
    }

    /// One step of preparing pinned storage while the server is idle: keeps a reserve of
    /// [`RESERVE_STATES`] states and [`RESERVE_PAGES`] KV pages within the budget, so captures
    /// do not pin memory on the request path. Returns whether there may be more to do.
    pub fn top_up(&mut self, b: &B, session: &Session<B>) -> Result<bool, CoreError> {
        if !self.enabled() {
            return Ok(false);
        }
        let page_len = session.kv_page_len();
        let pinned = self.bytes(b)
            + self.spare.iter().map(|s| s.bytes(b)).sum::<usize>()
            + self.pages.len() * page_len * 4;
        if pinned >= self.budget {
            return Ok(false);
        }
        if self.spare.len() < RESERVE_STATES {
            self.spare.push(session.blank_checkpoint_in(b, 0, true)?);
            return Ok(true);
        }
        if self.pages.len() < RESERVE_PAGES {
            for _ in 0..4 {
                self.pages.push(b.alloc_host(page_len)?);
            }
            return Ok(true);
        }
        Ok(false)
    }

    /// Whether an entry (resident or being copied to disk) holds exactly `keys`.
    pub fn holds(&self, keys: &[u32]) -> bool {
        self.entries
            .iter()
            .chain(self.inflight.iter().map(|(_, e)| e))
            .any(|e| e.keys == keys)
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
    /// Entries being copied to disk still qualify: their memory stays intact until the copy
    /// completes, and restoring only reads it.
    pub fn best(&self, keys: &[u32], limit: usize) -> Option<&Entry<B>> {
        self.entries
            .iter()
            .chain(self.inflight.iter().map(|(_, e)| e))
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
        // Resuming from a conversation's own history point (not a shared root) retires the private
        // end of the step that published it: the prompt-end entries this request diverges from
        // after that point (ninfer retires private endpoints beyond the divergence the same way).
        let from_root = self
            .entries
            .iter()
            .any(|e| e.root && e.pos() == start && e.prefix_of(keys));
        if start > 0 && !from_root {
            let dead: Vec<usize> = (0..self.entries.len())
                .filter(|&i| {
                    let e = &self.entries[i];
                    e.tail
                        && !e.prefix_of(keys)
                        && e.keys.iter().zip(keys).take_while(|(a, b)| a == b).count() >= start
                })
                .collect();
            for i in dead.into_iter().rev() {
                let e = self.entries.swap_remove(i);
                self.release(e);
            }
        }
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

    /// Marks entry `id` as a prompt's end, kept for an exact retry until the conversation moves on
    /// (see `Self::resumed`).
    pub fn mark_tail(&mut self, id: u64) {
        if let Some(e) = self.entries.iter_mut().find(|e| e.id == id) {
            e.tail = true;
        }
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
        drafter: Option<(&DFlashModel<B>, &DFlashState<B>)>,
    ) -> Result<u64, CoreError> {
        let pos = cp.pos;
        let keys = &keys[..pos];
        let (lambda, now) = (self.lambda, self.now);
        if let Some(e) = self.entries.iter_mut().find(|e| e.keys == keys) {
            Self::reference(lambda, now, e);
            e.root |= root;
            return Ok(e.id);
        }
        let hashes = page_hashes(keys);
        let mut kv = Vec::with_capacity(pos.div_ceil(KV_PAGE_ROWS));
        let (pool, page_len) = (&mut self.pages, session.kv_page_len());
        let mut page = || pool.pop().map_or_else(|| b.alloc_host(page_len), Ok);
        for i in 0..pos.div_ceil(KV_PAGE_ROWS) {
            let (lo, hi) = (i * KV_PAGE_ROWS, ((i + 1) * KV_PAGE_ROWS).min(pos));
            let full = hashes.get(i);
            if let Some(seg) = full
                .and_then(|h| self.shared.get(h))
                .and_then(Weak::upgrade)
            {
                kv.push(seg);
                continue;
            }
            let seg = Arc::new(session.save_kv(b, lo, hi, &mut page)?);
            if let Some(&h) = full {
                self.shared.insert(h, Arc::downgrade(&seg));
            }
            kv.push(seg);
        }
        // Drafter context is kept only when it reaches the entry's position: it is a window that
        // ends there, and context ending earlier would leave the restored drafter without one.
        let draft = match drafter {
            Some((dm, st)) => match dm.context_rows(st, pos) {
                Some((lo, hi)) if hi == pos => {
                    let (pool, len) = (&mut self.draft_pages, dm.context_page_len());
                    let mut page = || pool.pop().map_or_else(|| b.alloc_host(len), Ok);
                    Some(dm.save_context(b, st, lo, hi, &mut page)?)
                }
                _ => None,
            },
            None => None,
        };
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
            draft,
            root,
            tail: false,
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
            total += e.draft.as_ref().map_or(0, |d| d.bytes(b));
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
                self.release(e);
            }
        }
        self.shared.retain(|_, w| w.strong_count() > 0);
        Ok(())
    }

    /// Returns an entry's storage to the pools (KV pages once no other entry shares them).
    fn release(&mut self, e: Entry<B>) {
        if self.spare.len() < MAX_SPARE {
            self.spare.push(e.state);
        }
        for seg in e.kv {
            if let Ok(seg) = Arc::try_unwrap(seg) {
                self.pages.extend(seg.into_pages());
            }
        }
        if let Some(d) = e.draft {
            self.draft_pages.extend(d.into_pages());
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
            .chain(e.draft.iter().flat_map(|d| d.buffers()))
            .map(|buf| b.host_bytes(buf).map(|x| (x.as_ptr(), x.len())))
            .collect();
        let Some(parts) = parts else {
            return Some(e);
        };
        let segs = e.kv.iter().map(|s| (s.lo, s.hi)).collect();
        let draft = e.draft.as_ref().map(|d| (d.lo, d.hi));
        match disk.write(
            &e.keys,
            e.state.rope_delta,
            segs,
            draft,
            e.root,
            parts,
            self.now,
        ) {
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

/// Hash of the keys of every full KV page's whole prefix: entry `i` covers keys
/// `0 .. (i + 1) * KV_PAGE_ROWS` (chained, so equal hashes mean equal prefixes).
fn page_hashes(keys: &[u32]) -> Vec<u64> {
    let mut h = 0u64;
    keys.chunks_exact(KV_PAGE_ROWS)
        .map(|c| {
            let mut s = std::collections::hash_map::DefaultHasher::new();
            h.hash(&mut s);
            c.hash(&mut s);
            h = s.finish();
            h
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn page_hashes_chain_the_prefix() {
        let a: Vec<u32> = (0..3 * super::KV_PAGE_ROWS as u32 + 5).collect();
        let mut b = a.clone();
        b[1] = 7;
        let (ha, hb) = (super::page_hashes(&a), super::page_hashes(&b));
        assert_eq!(ha.len(), 3);
        // A change in the first page changes every later page's address.
        assert!(ha.iter().zip(&hb).all(|(x, y)| x != y));
        assert_eq!(ha, super::page_hashes(&a[..3 * super::KV_PAGE_ROWS]));
    }

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
