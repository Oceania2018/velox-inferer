//! Paged attention caches: a session's K/V caches in virtual memory ([`Backend::alloc_virtual`]),
//! backed chunk by chunk as the sequence grows, with chunks of an identical prefix shared between
//! sessions.
//!
//! Every cache of a session has the same row layout, so chunk index `i` covers the same rows in
//! all of them; one [`KvChunkSet`] backs index `i` of every cache. The rows a chunk covers depend
//! only on the tokens up to its last row, so sessions holding the same prefix can map the same
//! set ([`KvShare`]). Shared sets are read-only: a write below a shared set's end first copies it
//! (copy on write). Addresses never move, so kernels and recorded graphs are unchanged.
//!
//! With a [`KvPool`] the physical chunks are created once, at startup, and sets go back to the
//! pool when their last holder unmaps them: requests allocate no device memory for their caches.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};

use velox_core::{Backend, CopySpec, Error, Result};

use super::{KvSegment, Session, KV_PAGE_ROWS};

/// Rows backed past the last row a step writes: tiled attention kernels may load a whole tile
/// past the valid length (masking it), which must not touch unmapped memory.
const BACK_MARGIN_ROWS: usize = 256;

/// Physical chunks backing one chunk index of every cache of a session (layer-major, K then V).
pub struct KvChunkSet<B: Backend> {
    chunks: Vec<B::Chunk>,
    /// The key it is published under in a [`KvShare`] (0: unpublished). Cleared when its only
    /// holder writes into it.
    key: AtomicU64,
    /// Where its chunks go back when the last holder drops it (every holder unmaps first).
    pool: Option<Arc<KvPool<B>>>,
}

impl<B: Backend> Drop for KvChunkSet<B> {
    fn drop(&mut self) {
        if let Some(pool) = &self.pool {
            pool.give(std::mem::take(&mut self.chunks));
        }
    }
}

/// Chunk sets for the paged caches of every session, created at startup and sized from the
/// device memory left after weights, workspaces and the graph budget (see the engine). A set is
/// one chunk per cache (2 x attention layers); sessions take and return whole sets.
pub struct KvPool<B: Backend> {
    free: Mutex<Vec<Vec<B::Chunk>>>,
    sets: usize,
    /// Sets created outside the pool because it was empty (freed, not pooled, when dropped).
    overflow: AtomicUsize,
}

impl<B: Backend> KvPool<B> {
    /// Creates `sets` sets of `per_set` chunks.
    pub fn new(b: &B, sets: usize, per_set: usize) -> Result<Arc<Self>> {
        let mut free = Vec::with_capacity(sets);
        for _ in 0..sets {
            free.push(
                (0..per_set)
                    .map(|_| b.alloc_chunk())
                    .collect::<Result<Vec<_>>>()?,
            );
        }
        Ok(Arc::new(KvPool {
            free: Mutex::new(free),
            sets,
            overflow: AtomicUsize::new(0),
        }))
    }

    /// Sets in the pool.
    pub fn sets(&self) -> usize {
        self.sets
    }

    /// Sets not held by any session.
    pub fn free_sets(&self) -> usize {
        self.free.lock().map_or(0, |f| f.len())
    }

    /// Sets ever created outside the pool because it was empty.
    pub fn overflow(&self) -> usize {
        self.overflow.load(Ordering::Relaxed)
    }

    fn take(&self) -> Option<Vec<B::Chunk>> {
        self.free.lock().ok()?.pop()
    }

    fn give(&self, chunks: Vec<B::Chunk>) {
        if let Ok(mut f) = self.free.lock() {
            f.push(chunks);
        }
    }
}

/// Paging state of a session whose caches are virtual.
pub(super) struct KvPaging<B: Backend> {
    /// f32 slots per chunk.
    chunk: usize,
    /// Backing of each chunk index.
    sets: Vec<Option<Arc<KvChunkSet<B>>>>,
    /// One chunk per cache of address space, for copy on write (allocated on first use).
    scratch: Option<B::Buffer>,
    /// Where new sets come from (none: created and freed on demand).
    pool: Option<Arc<KvPool<B>>>,
}

impl<B: Backend> KvPaging<B> {
    pub(super) fn new(chunk: usize, chunks: usize) -> Self {
        KvPaging {
            chunk,
            sets: (0..chunks).map(|_| None).collect(),
            scratch: None,
            pool: None,
        }
    }
}

impl<B: Backend> KvPaging<B> {
    /// A private, unpublished set of `n` chunks: from the pool, or created when there is none
    /// (or it is empty, which [`KvPool::overflow`] counts).
    fn new_set(&self, b: &B, n: usize) -> Result<Arc<KvChunkSet<B>>> {
        let (chunks, pool) = match self.pool.as_ref().map(|p| (p.take(), p)) {
            Some((Some(c), p)) => (c, Some(p.clone())),
            other => {
                if let Some((_, p)) = other {
                    p.overflow.fetch_add(1, Ordering::Relaxed);
                }
                let c = (0..n)
                    .map(|_| b.alloc_chunk())
                    .collect::<Result<Vec<_>>>()?;
                (c, None)
            }
        };
        Ok(Arc::new(KvChunkSet {
            chunks,
            key: AtomicU64::new(0),
            pool,
        }))
    }
}

/// Published prefix chunk sets, by prefix key: sessions on the same prefix map one copy.
pub struct KvShare<B: Backend> {
    sets: HashMap<u64, Weak<KvChunkSet<B>>>,
}

impl<B: Backend> Default for KvShare<B> {
    fn default() -> Self {
        KvShare {
            sets: HashMap::new(),
        }
    }
}

impl<B: Backend> KvShare<B> {
    /// Live published sets.
    pub fn len(&self) -> usize {
        self.sets.values().filter(|w| w.strong_count() > 0).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn get(&self, key: u64) -> Option<Arc<KvChunkSet<B>>> {
        self.sets
            .get(&key)
            .and_then(Weak::upgrade)
            .filter(|s| s.key.load(Ordering::Relaxed) == key)
    }
}

/// Key of chunk `index` whose rows end at `keys.len()`: the prefix and the index.
fn chunk_key(keys: &[u32], index: usize) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    keys.hash(&mut h);
    index.hash(&mut h);
    h.finish().max(1)
}

impl<B: Backend> Session<B> {
    /// Whether the attention caches are paged (virtual memory).
    pub fn kv_paged(&self) -> bool {
        self.paging.is_some()
    }

    /// Bytes of device memory backing this session's caches (paged), or their full size.
    pub fn kv_bytes(&self, b: &B) -> usize {
        match &self.paging {
            Some(pg) => pg.sets.iter().flatten().count() * pg.chunk * 4 * 2 * self.kv.len(),
            None => self.kv.iter().map(|(k, v)| (b.len(k) + b.len(v)) * 4).sum(),
        }
    }

    /// Takes new chunk sets from `pool` from now on.
    pub fn set_kv_pool(&mut self, pool: Arc<KvPool<B>>) {
        if let Some(pg) = self.paging.as_mut() {
            pg.pool = Some(pool);
        }
    }

    /// Chunk sets this session alone holds (what trimming it returns to the pool).
    pub fn kv_private_sets(&self) -> usize {
        self.paging.as_ref().map_or(0, |pg| {
            pg.sets
                .iter()
                .flatten()
                .filter(|s| Arc::strong_count(s) == 1)
                .count()
        })
    }

    /// Chunk indices covering rows `0..rows` (with the margin).
    pub fn kv_sets_for(&self, rows: usize) -> usize {
        self.paging.as_ref().map_or(0, |pg| {
            ((rows + BACK_MARGIN_ROWS).min(self.capacity) * self.kv_row)
                .div_ceil(pg.chunk)
                .min(pg.sets.len())
        })
    }

    /// New sets backing rows `0..rows` may take: unbacked indices, plus shared ones a write may
    /// copy (an upper bound).
    pub fn kv_need(&self, rows: usize) -> usize {
        let end = self.kv_sets_for(rows);
        self.paging.as_ref().map_or(0, |pg| {
            pg.sets[..end]
                .iter()
                .filter(|s| s.as_ref().is_none_or(|s| Arc::strong_count(s) > 1))
                .count()
        })
    }

    /// Paged caches: (chunks in one set, rows one set covers).
    pub fn kv_set_shape(&self) -> Option<(usize, usize)> {
        self.paging
            .as_ref()
            .map(|pg| (2 * self.kv.len(), pg.chunk / self.kv_row))
    }

    /// Chunk indices backed by a set other sessions also map.
    pub fn kv_shared_chunks(&self) -> usize {
        self.paging.as_ref().map_or(0, |pg| {
            pg.sets
                .iter()
                .flatten()
                .filter(|s| Arc::strong_count(s) > 1)
                .count()
        })
    }

    /// Rows `lo..hi` of chunk index `i`: the rows any of its slots belong to.
    fn chunk_rows(&self, chunk: usize, i: usize) -> (usize, usize) {
        let lo = i * chunk / self.kv_row;
        let hi = ((i + 1) * chunk).div_ceil(self.kv_row).min(self.capacity);
        (lo, hi)
    }

    /// Before writing rows `lo..hi`: backs rows up to `hi` (and a margin; attention reads every
    /// row below) and makes every chunk the write touches
    /// private, copying shared ones. Cheap when nothing changes.
    pub fn prepare_kv(&mut self, b: &B, lo: usize, hi: usize) -> Result<()> {
        let Some(pg) = self.paging.as_ref() else {
            return Ok(());
        };
        let chunk = pg.chunk;
        let first = lo * self.kv_row / chunk;
        let end = ((hi + BACK_MARGIN_ROWS).min(self.capacity) * self.kv_row)
            .div_ceil(chunk)
            .min(pg.sets.len());
        let mut copy = Vec::new();
        for i in first..end {
            match &pg.sets[i] {
                None => {}
                Some(s) if Arc::strong_count(s) > 1 => copy.push(i),
                Some(s) => s.key.store(0, Ordering::Relaxed),
            }
        }
        self.copy_on_write(b, &copy)?;
        self.back_chunks(b, 0, end)
    }

    /// Maps fresh private sets at the unmapped indices in `first..end`.
    fn back_chunks(&mut self, b: &B, first: usize, end: usize) -> Result<()> {
        let n = 2 * self.kv.len();
        let Some(pg) = self.paging.as_mut() else {
            return Ok(());
        };
        for i in first..end {
            if pg.sets[i].is_some() {
                continue;
            }
            let set = pg.new_set(b, n)?;
            for (c, (k, v)) in self.kv.iter_mut().enumerate() {
                b.map_chunk(k, i, &set.chunks[2 * c])?;
                b.map_chunk(v, i, &set.chunks[2 * c + 1])?;
            }
            pg.sets[i] = Some(set);
        }
        Ok(())
    }

    /// Replaces the sets at `indices` with private copies of their contents.
    fn copy_on_write(&mut self, b: &B, indices: &[usize]) -> Result<()> {
        if indices.is_empty() {
            return Ok(());
        }
        let n = 2 * self.kv.len();
        let cache_len = self.kv.first().map_or(0, |(k, _)| b.len(k));
        let pg = self.paging.as_mut().expect("paged");
        let chunk = pg.chunk;
        if pg.scratch.is_none() {
            pg.scratch = Some(b.alloc_virtual(n * chunk)?);
        }
        for &i in indices {
            let set = pg.new_set(b, n)?;
            let scratch = pg.scratch.as_mut().expect("allocated above");
            for (j, ch) in set.chunks.iter().enumerate() {
                b.map_chunk(scratch, j, ch)?;
            }
            let len = chunk.min(cache_len - i * chunk);
            for (c, (k, v)) in self.kv.iter().enumerate() {
                b.copy(k, i * chunk, scratch, 2 * c * chunk, len)?;
                b.copy(v, i * chunk, scratch, (2 * c + 1) * chunk, len)?;
            }
            // The copies and every queued step reading the old mapping must finish first.
            b.synchronize()?;
            for j in 0..n {
                b.unmap_chunk(scratch, j)?;
            }
            for (c, (k, v)) in self.kv.iter_mut().enumerate() {
                b.unmap_chunk(k, i)?;
                b.unmap_chunk(v, i)?;
                b.map_chunk(k, i, &set.chunks[2 * c])?;
                b.map_chunk(v, i, &set.chunks[2 * c + 1])?;
            }
            pg.sets[i] = Some(set);
        }
        Ok(())
    }

    /// Before a request on prefix `keys` restores rows below `start`: shared sets that do not
    /// hold this prefix (another conversation's), or that the request will write into, become
    /// private copies, so [`Self::load_kv`] may skip every shared set that remains.
    pub fn retarget_kv(&mut self, b: &B, keys: &[u32], start: usize) -> Result<()> {
        let Some(pg) = self.paging.as_ref() else {
            return Ok(());
        };
        let chunk = pg.chunk;
        let mut copy = Vec::new();
        for (i, s) in pg.sets.iter().enumerate() {
            let Some(s) = s.as_ref().filter(|s| Arc::strong_count(s) > 1) else {
                continue;
            };
            let (_, hi) = self.chunk_rows(chunk, i);
            let held = hi <= start
                && hi <= keys.len()
                && s.key.load(Ordering::Relaxed) == chunk_key(&keys[..hi], i);
            if !held {
                copy.push(i);
            }
        }
        self.copy_on_write(b, &copy)
    }

    /// Maps published sets of prefix `keys` at unmapped chunk indices wholly below `start` (the
    /// rows the request restores), so they need no memory or copy of their own. Returns how
    /// many it mapped.
    pub fn adopt_kv(
        &mut self,
        b: &B,
        share: &KvShare<B>,
        keys: &[u32],
        start: usize,
    ) -> Result<usize> {
        let Some(pg) = self.paging.as_ref() else {
            return Ok(0);
        };
        let chunk = pg.chunk;
        let mut found = Vec::new();
        for i in 0..pg.sets.len() {
            let (_, hi) = self.chunk_rows(chunk, i);
            if hi > start || hi > keys.len() {
                break;
            }
            if pg.sets[i].is_none() {
                if let Some(s) = share.get(chunk_key(&keys[..hi], i)) {
                    found.push((i, s));
                }
            }
        }
        let n = found.len();
        for (i, set) in found {
            for (c, (k, v)) in self.kv.iter_mut().enumerate() {
                b.map_chunk(k, i, &set.chunks[2 * c])?;
                b.map_chunk(v, i, &set.chunks[2 * c + 1])?;
            }
            self.paging.as_mut().expect("paged").sets[i] = Some(set);
        }
        Ok(n)
    }

    /// Publishes the sets wholly inside rows `0..keys.len()` (written for prefix `keys`) so other
    /// sessions on that prefix can map them. A prefix already published keeps its set.
    pub fn publish_kv(&self, share: &mut KvShare<B>, keys: &[u32]) {
        let Some(pg) = self.paging.as_ref() else {
            return;
        };
        let rows = keys.len().min(self.pos);
        share.sets.retain(|_, w| w.strong_count() > 0);
        for (i, s) in pg.sets.iter().enumerate() {
            let (_, hi) = self.chunk_rows(pg.chunk, i);
            if hi > rows {
                break;
            }
            let Some(s) = s else { continue };
            let key = chunk_key(&keys[..hi], i);
            if s.key.load(Ordering::Relaxed) == key || share.get(key).is_some() {
                continue;
            }
            s.key.store(key, Ordering::Relaxed);
            share.sets.insert(key, Arc::downgrade(s));
        }
    }

    /// Unbacks private chunks wholly past row `keep` (with the margin), returning their memory.
    /// Synchronizes first when anything is unmapped.
    pub fn trim_kv(&mut self, b: &B, keep: usize) -> Result<usize> {
        let Some(pg) = self.paging.as_ref() else {
            return Ok(0);
        };
        let from = ((keep + BACK_MARGIN_ROWS).min(self.capacity) * self.kv_row).div_ceil(pg.chunk);
        let drop: Vec<usize> = (from..pg.sets.len())
            .filter(|&i| pg.sets[i].is_some())
            .collect();
        if drop.is_empty() {
            return Ok(0);
        }
        b.synchronize()?;
        for &i in &drop {
            for (k, v) in self.kv.iter_mut() {
                b.unmap_chunk(k, i)?;
                b.unmap_chunk(v, i)?;
            }
            self.paging.as_mut().expect("paged").sets[i] = None;
        }
        Ok(drop.len())
    }

    /// Writes the rows of `seg` at or after `from` back into the caches, skipping slots of shared
    /// sets (which hold this prefix already; see [`Self::retarget_kv`]) and backing the rest.
    pub fn load_kv(&mut self, b: &B, seg: &KvSegment<B>, from: usize) -> Result<()> {
        if seg.hi > self.capacity {
            return Err(Error::Config("KV segment beyond session capacity".into()));
        }
        let row = self.kv_row;
        let lo_row = from.max(seg.lo);
        if lo_row >= seg.hi {
            return Ok(());
        }
        // Back the rows written (shared sets stay as they are).
        if let Some(pg) = self.paging.as_ref() {
            let (first, end) = (lo_row * row / pg.chunk, (seg.hi * row).div_ceil(pg.chunk));
            for i in first..end.min(pg.sets.len()) {
                if let Some(s) = pg.sets[i].as_ref().filter(|s| Arc::strong_count(s) == 1) {
                    s.key.store(0, Ordering::Relaxed);
                }
            }
            self.back_chunks(
                b,
                first,
                end.min(self.paging.as_ref().expect("paged").sets.len()),
            )?;
        }
        // Slot ranges of shared sets, which are skipped.
        let skip: Vec<(usize, usize)> = self.paging.as_ref().map_or(Vec::new(), |pg| {
            pg.sets
                .iter()
                .enumerate()
                .filter(|(_, s)| s.as_ref().is_some_and(|s| Arc::strong_count(s) > 1))
                .map(|(i, _)| (i * pg.chunk, (i + 1) * pg.chunk))
                .collect()
        });
        // One submission; destinations are the caches in layer order, K then V.
        let mut ops = Vec::new();
        for (p, buf) in seg.pages.iter().enumerate() {
            let r0 = seg.lo + p * KV_PAGE_ROWS;
            let r1 = (r0 + KV_PAGE_ROWS).min(seg.hi);
            let a = r0.max(from);
            if a >= r1 {
                continue;
            }
            if b.len(buf) < self.kv_page_len() {
                return Err(Error::Config("KV segment does not fit this session".into()));
            }
            for (s0, s1) in subtract((a * row, r1 * row), &skip) {
                for l in 0..self.kv.len() {
                    for side in 0..2 {
                        ops.push(CopySpec {
                            src: buf,
                            src_offset: (2 * l + side) * KV_PAGE_ROWS * row + s0 - r0 * row,
                            dst: 2 * l + side,
                            dst_offset: s0,
                            len: s1 - s0,
                        });
                    }
                }
            }
        }
        let mut dsts: Vec<&mut B::Buffer> = self.kv.iter_mut().flat_map(|(k, v)| [k, v]).collect();
        b.copy_batch(&mut dsts, &ops)
    }
}

/// `range` minus the sorted, disjoint `holes`.
fn subtract(range: (usize, usize), holes: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut lo = range.0;
    for &(h0, h1) in holes {
        if h1 <= lo || h0 >= range.1 {
            continue;
        }
        if h0 > lo {
            out.push((lo, h0));
        }
        lo = lo.max(h1);
    }
    if lo < range.1 {
        out.push((lo, range.1));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::subtract;

    #[test]
    fn subtract_holes() {
        assert_eq!(subtract((0, 100), &[]), vec![(0, 100)]);
        assert_eq!(
            subtract((0, 100), &[(10, 20), (50, 60)]),
            vec![(0, 10), (20, 50), (60, 100)]
        );
        assert_eq!(subtract((15, 55), &[(10, 20), (50, 60)]), vec![(20, 50)]);
        assert_eq!(subtract((0, 100), &[(0, 200)]), vec![]);
    }
}
