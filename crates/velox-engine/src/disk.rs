//! NVMe tier below the context store (off by default): the hot entries of the store, roots
//! (system prompt, task) a request resumed from and branch points several did, are copied to disk by a
//! background thread while they are still in system RAM, and a request whose deepest match is only
//! on disk reads it back when that covers enough more than every resident source. That serves a
//! cold start (a restarted server finds its prompts on disk instead of prefilling them) and hot
//! prefixes that system RAM cannot keep; one-off history points never go to disk.
//!
//! With a model identity ([`DiskTier::new`] `tag`), records live in a directory named after it
//! and the storage layout and outlive the process: each file starts with a header describing it
//! (keys, KV ranges) and is indexed again at startup. Files are named by their keys, so a prefix
//! already on disk is never written again. Without one, records live in a process-private
//! directory removed on shutdown. Matching never reads the disk. Capacity is reclaimed in
//! least-recently-used order (a write or a read counts as a use; records found at startup rank by
//! file age, below anything used since).

use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread::JoinHandle;
use std::time::{Duration, SystemTime};

/// Bytes of pinned host memory the writer copies from. The store keeps the owning entry alive,
/// and unchanged, until the write completes.
struct Part(*const u8, usize);
// SAFETY: see `Part`; the memory outlives the job and nothing writes it meanwhile.
unsafe impl Send for Part {}

struct Job {
    id: u64,
    path: PathBuf,
    header: Vec<u8>,
    parts: Vec<Part>,
}

pub(crate) struct Record {
    pub id: u64,
    pub keys: Vec<u32>,
    pub rope_delta: i64,
    /// KV segment ranges, in file order after the state.
    pub segs: Vec<(usize, usize)>,
    /// Drafter context rows, in file order after the KV.
    pub draft: Option<(usize, usize)>,
    pub root: bool,
    path: PathBuf,
    /// Header bytes before the payload.
    offset: u64,
    /// File size.
    bytes: u64,
    last: u64,
    /// Written completely (in flight until then).
    pub ready: bool,
}

pub(crate) struct DiskTier {
    dir: PathBuf,
    /// Records outlive the process (see the module docs).
    persistent: bool,
    layout: u64,
    budget: u64,
    records: Vec<Record>,
    jobs: Option<Sender<Job>>,
    done: Receiver<(u64, std::io::Result<()>)>,
    writer: Option<JoinHandle<()>>,
    next_id: u64,
    /// Reads and writes since start (the other [`DiskStats`] fields are left at zero here).
    io: DiskStats,
}

/// What the tier holds and what it has moved since start, for `GET /metrics`.
#[derive(Clone, Copy, Default)]
pub(crate) struct DiskStats {
    /// Bytes of ready records.
    pub used: u64,
    pub budget: u64,
    pub records: u64,
    /// Records read back, their payload bytes and the prompt tokens they covered.
    pub reads: u64,
    pub read_bytes: u64,
    pub read_tokens: u64,
    /// Records written completely and their bytes.
    pub writes: u64,
    pub written_bytes: u64,
}

const MAGIC: &[u8; 8] = b"VXCTX\0\0\x01";
/// Magic, layout, header length.
const FIXED: usize = 20;
/// Partial files older than this are left over from a crash (a live writer renames its own
/// within seconds).
const STALE_TMP: Duration = Duration::from_secs(3600);

pub(crate) const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

/// FNV-1a: stable across processes and toolchains (file names, layout and header checks).
pub(crate) fn fnv64(seed: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(seed, |h, &x| {
        (h ^ x as u64).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn key_bytes(keys: &[u32]) -> Vec<u8> {
    keys.iter().flat_map(|k| k.to_le_bytes()).collect()
}

fn invalid(m: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, m.into())
}

impl DiskTier {
    /// A tier of at most `budget` bytes under `root`: in `<tag>-<layout>` when `tag` (the model's
    /// identity) is given, holding the records an earlier process left there, else in a new
    /// private directory. `layout` identifies the storage shapes the records must match.
    pub fn new(
        root: PathBuf,
        budget: u64,
        tag: Option<&str>,
        layout: u64,
    ) -> std::io::Result<Self> {
        let dir = match tag {
            Some(t) => {
                let t: String = t
                    .chars()
                    .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
                    .collect();
                root.join(format!("{t}-{layout:016x}"))
            }
            None => root.join(format!("velox-{}", std::process::id())),
        };
        fs::create_dir_all(&dir)?;
        let (jobs, rx) = channel::<Job>();
        let (tx, done) = channel();
        let writer = std::thread::Builder::new()
            .name("velox-disk-context".into())
            .spawn(move || {
                for job in rx {
                    // Written aside and renamed: a crash or another process never sees a partial
                    // record under its final name.
                    let tmp = job
                        .path
                        .with_extension(format!("tmp{}", std::process::id()));
                    let r = File::create(&tmp)
                        .and_then(|mut f| {
                            f.write_all(&job.header)?;
                            for Part(p, n) in &job.parts {
                                // SAFETY: see `Part`.
                                f.write_all(unsafe { std::slice::from_raw_parts(*p, *n) })?;
                            }
                            Ok(())
                        })
                        .and_then(|()| fs::rename(&tmp, &job.path));
                    if r.is_err() {
                        let _ = fs::remove_file(&tmp);
                    }
                    if tx.send((job.id, r)).is_err() {
                        break;
                    }
                }
            })?;
        let mut tier = DiskTier {
            dir,
            persistent: tag.is_some(),
            layout,
            budget,
            records: Vec::new(),
            jobs: Some(jobs),
            done,
            writer: Some(writer),
            next_id: 1,
            io: DiskStats::default(),
        };
        if tier.persistent {
            tier.scan();
        }
        Ok(tier)
    }

    /// Whether records outlive the process.
    pub fn persistent(&self) -> bool {
        self.persistent
    }

    /// Indexes the records an earlier process left, oldest first (evicted first), dropping files
    /// that do not match this layout and leftovers of interrupted writes.
    fn scan(&mut self) {
        let Ok(dir) = fs::read_dir(&self.dir) else {
            return;
        };
        let mut found = Vec::new();
        for ent in dir.flatten() {
            let path = ent.path();
            let mtime = ent
                .metadata()
                .and_then(|m| m.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            match path.extension().and_then(|e| e.to_str()) {
                Some("ctx") => match self.read_header(&path) {
                    Ok(r) => found.push((mtime, r)),
                    Err(e) => {
                        eprintln!("disk context: dropping {}: {e}", path.display());
                        let _ = fs::remove_file(&path);
                    }
                },
                Some(e)
                    if e.starts_with("tmp") && mtime.elapsed().is_ok_and(|age| age > STALE_TMP) =>
                {
                    let _ = fs::remove_file(&path);
                }
                _ => {}
            }
        }
        found.sort_by_key(|(t, _)| *t);
        for (_, mut r) in found {
            r.id = self.next_id;
            self.next_id += 1;
            self.records.push(r);
        }
        // The budget may be smaller than last time.
        self.make_room(0);
        if !self.records.is_empty() {
            let bytes: u64 = self.records.iter().map(|r| r.bytes).sum();
            eprintln!(
                "disk context: {} records ({:.2} GB) in {}",
                self.records.len(),
                bytes as f64 / 1e9,
                self.dir.display()
            );
        }
    }

    /// Magic, layout, header length, payload length, rope delta, root flag, keys, KV ranges,
    /// drafter range, then a checksum of all of it.
    fn header(
        &self,
        keys: &[u32],
        rope_delta: i64,
        segs: &[(usize, usize)],
        draft: Option<(usize, usize)>,
        root: bool,
        payload: u64,
    ) -> Vec<u8> {
        let mut h = Vec::with_capacity(80 + keys.len() * 4 + segs.len() * 16);
        h.extend_from_slice(MAGIC);
        h.extend_from_slice(&self.layout.to_le_bytes());
        h.extend_from_slice(&0u32.to_le_bytes());
        h.extend_from_slice(&payload.to_le_bytes());
        h.extend_from_slice(&rope_delta.to_le_bytes());
        h.push(root as u8);
        h.extend_from_slice(&(keys.len() as u32).to_le_bytes());
        h.extend_from_slice(&key_bytes(keys));
        h.extend_from_slice(&(segs.len() as u32).to_le_bytes());
        for &(lo, hi) in segs {
            h.extend_from_slice(&(lo as u64).to_le_bytes());
            h.extend_from_slice(&(hi as u64).to_le_bytes());
        }
        let (lo, hi) = draft.unwrap_or((0, 0));
        h.push(draft.is_some() as u8);
        h.extend_from_slice(&(lo as u64).to_le_bytes());
        h.extend_from_slice(&(hi as u64).to_le_bytes());
        let total = (h.len() + 8) as u32;
        h[16..FIXED].copy_from_slice(&total.to_le_bytes());
        let sum = fnv64(FNV_OFFSET, &h);
        h.extend_from_slice(&sum.to_le_bytes());
        h
    }

    /// The record a file holds, checked against this layout and the file's length.
    fn read_header(&self, path: &Path) -> std::io::Result<Record> {
        let mut f = File::open(path)?;
        let len = f.metadata()?.len();
        let mut h = vec![0u8; FIXED];
        f.read_exact(&mut h)?;
        if &h[..8] != MAGIC {
            return Err(invalid("not a context record"));
        }
        if u64::from_le_bytes(h[8..16].try_into().unwrap()) != self.layout {
            return Err(invalid("another storage layout"));
        }
        let total = u32::from_le_bytes(h[16..FIXED].try_into().unwrap()) as usize;
        if total < FIXED + 8 || total as u64 > len || total > 64 << 20 {
            return Err(invalid("bad header length"));
        }
        h.resize(total, 0);
        f.read_exact(&mut h[FIXED..])?;
        let body = total - 8;
        if fnv64(FNV_OFFSET, &h[..body]).to_le_bytes()[..] != h[body..] {
            return Err(invalid("header checksum"));
        }
        let mut at = FIXED;
        let mut take = |n: usize| -> std::io::Result<&[u8]> {
            let s = h[..body]
                .get(at..at + n)
                .ok_or_else(|| invalid("short header"))?;
            at += n;
            Ok(s)
        };
        let u64_of = |s: &[u8]| u64::from_le_bytes(s.try_into().unwrap());
        let u32_of = |s: &[u8]| u32::from_le_bytes(s.try_into().unwrap()) as usize;
        let payload = u64_of(take(8)?);
        let rope_delta = u64_of(take(8)?) as i64;
        let root = take(1)?[0] != 0;
        let n = u32_of(take(4)?);
        let keys = take(n * 4)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| u32::from_le_bytes(c))
            .collect();
        let n = u32_of(take(4)?);
        let mut segs = Vec::with_capacity(n.min(1 << 16));
        for _ in 0..n {
            let (lo, hi) = (u64_of(take(8)?), u64_of(take(8)?));
            segs.push((lo as usize, hi as usize));
        }
        let has_draft = take(1)?[0] != 0;
        let (lo, hi) = (u64_of(take(8)?), u64_of(take(8)?));
        if total as u64 + payload != len {
            return Err(invalid("truncated"));
        }
        Ok(Record {
            id: 0,
            keys,
            rope_delta,
            segs,
            draft: has_draft.then_some((lo as usize, hi as usize)),
            root,
            path: path.to_path_buf(),
            offset: total as u64,
            bytes: len,
            last: 0,
            ready: true,
        })
    }

    fn path_for(&self, id: u64, keys: &[u32]) -> PathBuf {
        if self.persistent {
            // 128 bits of the keys: a prefix has one file whichever process wrote it.
            let k = key_bytes(keys);
            let (a, b) = (fnv64(FNV_OFFSET, &k), fnv64(!FNV_OFFSET, &k));
            self.dir.join(format!("{a:016x}{b:016x}.ctx"))
        } else {
            self.dir.join(format!("{id}.ctx"))
        }
    }

    /// Whether a record (ready or being written) holds exactly `keys`.
    pub fn holds(&self, keys: &[u32]) -> bool {
        self.records.iter().any(|r| r.keys == keys)
    }

    /// Queues a copy of `parts` (state, KV, then drafter-context buffers) as a record of `keys`;
    /// returns its id, or `None` when it cannot fit. The caller keeps the memory alive until
    /// [`Self::completed`] reports the id.
    #[allow(clippy::too_many_arguments)]
    pub fn write(
        &mut self,
        keys: &[u32],
        rope_delta: i64,
        segs: Vec<(usize, usize)>,
        draft: Option<(usize, usize)>,
        root: bool,
        parts: Vec<(*const u8, usize)>,
        now: u64,
    ) -> Option<u64> {
        let payload: u64 = parts.iter().map(|&(_, n)| n as u64).sum();
        let header = self.header(keys, rope_delta, &segs, draft, root, payload);
        let offset = header.len() as u64;
        let bytes = offset + payload;
        if bytes > self.budget || !self.make_room(bytes) {
            return None;
        }
        let id = self.next_id;
        self.next_id += 1;
        let path = self.path_for(id, keys);
        let job = Job {
            id,
            path: path.clone(),
            header,
            parts: parts.into_iter().map(|(p, n)| Part(p, n)).collect(),
        };
        self.jobs.as_ref()?.send(job).ok()?;
        self.records.push(Record {
            id,
            keys: keys.to_vec(),
            rope_delta,
            segs,
            draft,
            root,
            path,
            offset,
            bytes,
            last: now,
            ready: false,
        });
        Some(id)
    }

    /// Evicts ready records, least recently used first, until `bytes` more fit.
    fn make_room(&mut self, bytes: u64) -> bool {
        loop {
            let used: u64 = self.records.iter().map(|r| r.bytes).sum();
            if used + bytes <= self.budget {
                return true;
            }
            let Some(i) = self
                .records
                .iter()
                .enumerate()
                .filter(|(_, r)| r.ready)
                .min_by_key(|(_, r)| (r.last, r.id))
                .map(|(i, _)| i)
            else {
                return false;
            };
            let r = self.records.swap_remove(i);
            let _ = fs::remove_file(&r.path);
        }
    }

    /// Ids of finished writes; failed ones are forgotten.
    pub fn completed(&mut self) -> Vec<u64> {
        let mut ids = Vec::new();
        while let Ok((id, r)) = self.done.try_recv() {
            match r {
                Ok(()) => {
                    if let Some(rec) = self.records.iter_mut().find(|r| r.id == id) {
                        rec.ready = true;
                        self.io.writes += 1;
                        self.io.written_bytes += rec.bytes;
                    }
                }
                Err(e) => {
                    eprintln!("disk context: write failed: {e}");
                    self.records.retain(|r| r.id != id);
                }
            }
            ids.push(id);
        }
        ids
    }

    /// The deepest ready record on the prefix of `keys` at or below `limit` tokens.
    pub fn best(&self, keys: &[u32], limit: usize) -> Option<&Record> {
        self.records
            .iter()
            .filter(|r| {
                r.ready
                    && r.keys.len() <= limit
                    && keys.len() >= r.keys.len()
                    && keys[..r.keys.len()] == r.keys[..]
            })
            .max_by_key(|r| r.keys.len())
    }

    /// Reads record `id` into `bufs` (in the order written), which must cover its payload
    /// exactly. A failed read drops the record.
    pub fn read(&mut self, id: u64, bufs: &mut [&mut [u8]], now: u64) -> std::io::Result<()> {
        let Some(i) = self.records.iter().position(|r| r.id == id) else {
            return Err(std::io::ErrorKind::NotFound.into());
        };
        let (path, offset, payload) = {
            let r = &self.records[i];
            (r.path.clone(), r.offset, r.bytes - r.offset)
        };
        let want: u64 = bufs.iter().map(|b| b.len() as u64).sum();
        let r = if want != payload {
            Err(invalid(format!(
                "record holds {payload} bytes, storage {want}"
            )))
        } else {
            File::open(&path).and_then(|mut f| {
                f.seek(SeekFrom::Start(offset))?;
                for b in bufs.iter_mut() {
                    f.read_exact(b)?;
                }
                Ok(())
            })
        };
        match &r {
            Ok(()) => {
                self.records[i].last = now;
                self.io.reads += 1;
                self.io.read_bytes += payload;
                self.io.read_tokens += self.records[i].keys.len() as u64;
            }
            Err(_) => {
                self.records.swap_remove(i);
                let _ = fs::remove_file(path);
            }
        }
        r
    }

    pub fn len(&self) -> usize {
        self.records.iter().filter(|r| r.ready).count()
    }

    pub fn stats(&self) -> DiskStats {
        let ready = self.records.iter().filter(|r| r.ready);
        DiskStats {
            used: ready.clone().map(|r| r.bytes).sum(),
            budget: self.budget,
            records: ready.count() as u64,
            ..self.io
        }
    }
}

impl Drop for DiskTier {
    fn drop(&mut self) {
        // Queued writes finish first: a persistent tier keeps them for the next start.
        self.jobs = None;
        if let Some(w) = self.writer.take() {
            let _ = w.join();
        }
        if !self.persistent {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tier(dir: &Path, budget: u64) -> DiskTier {
        DiskTier::new(dir.to_path_buf(), budget, Some("test"), 42).unwrap()
    }

    fn wait(t: &mut DiskTier) {
        while t.records.iter().any(|r| !r.ready) {
            t.completed();
            std::thread::yield_now();
        }
    }

    #[test]
    fn records_outlive_the_process_and_are_not_rewritten() {
        let dir = std::env::temp_dir().join(format!("velox-disk-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let keys: Vec<u32> = (0..300).collect();
        let (state, kv) = (vec![7u8; 64], vec![9u8; 128]);
        {
            let mut t = tier(&dir, 1 << 20);
            let parts = vec![(state.as_ptr(), state.len()), (kv.as_ptr(), kv.len())];
            let id = t.write(&keys, -3, vec![(0, 256), (256, 300)], None, true, parts, 1);
            assert!(id.is_some());
            wait(&mut t);
            assert!(t.holds(&keys));
        }
        // A new process (tier) indexes it again and reads it back.
        let mut t = tier(&dir, 1 << 20);
        assert_eq!(t.len(), 1);
        let r = t.best(&[keys.clone(), vec![1, 2]].concat(), 1000).unwrap();
        let (id, segs, rope, root) = (r.id, r.segs.clone(), r.rope_delta, r.root);
        assert_eq!((segs, rope, root), (vec![(0, 256), (256, 300)], -3, true));
        let (mut a, mut b) = (vec![0u8; 64], vec![0u8; 128]);
        t.read(id, &mut [&mut a[..], &mut b[..]], 2).unwrap();
        assert_eq!((a, b), (state, kv));
        // Storage of another size is refused (and the record dropped).
        let mut c = [0u8; 10];
        assert!(t.read(id, &mut [&mut c[..]], 3).is_err());
        assert_eq!(t.len(), 0);
        drop(t);
        // Another layout never sees the first one's directory.
        let other = DiskTier::new(dir.clone(), 1 << 20, Some("test"), 43).unwrap();
        assert_eq!(other.len(), 0);
        drop(other);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn stats_count_complete_writes_and_successful_reads() {
        let dir = std::env::temp_dir().join(format!("velox-disk-stats-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let keys: Vec<u32> = (0..40).collect();
        let data = [5u8; 100];
        let mut t = tier(&dir, 1 << 20);
        let id = t
            .write(
                &keys,
                0,
                vec![(0, 40)],
                None,
                true,
                vec![(data.as_ptr(), data.len())],
                1,
            )
            .unwrap();
        wait(&mut t);
        let s = t.stats();
        assert_eq!((s.records, s.writes, s.reads), (1, 1, 0));
        assert_eq!((s.used, s.budget), (s.written_bytes, 1 << 20));
        assert!(s.used > 100, "header and payload");
        let mut back = [0u8; 100];
        t.read(id, &mut [&mut back[..]], 2).unwrap();
        let s = t.stats();
        assert_eq!((s.reads, s.read_bytes, s.read_tokens), (1, 100, 40));
        // A failed read counts nothing (and drops the record).
        let mut short = [0u8; 1];
        assert!(t.read(id, &mut [&mut short[..]], 3).is_err());
        let s = t.stats();
        assert_eq!((s.reads, s.records, s.used), (1, 0, 0));
        drop(t);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_smaller_budget_drops_the_oldest_records_at_startup() {
        let dir = std::env::temp_dir().join(format!("velox-disk-budget-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let data = vec![1u8; 1000];
        {
            let mut t = tier(&dir, 1 << 20);
            for i in 0..3u32 {
                t.write(
                    &[i],
                    0,
                    vec![],
                    None,
                    true,
                    vec![(data.as_ptr(), data.len())],
                    1,
                );
                wait(&mut t);
                // Distinct modification times.
                std::thread::sleep(Duration::from_millis(20));
            }
        }
        let t = tier(&dir, 2500);
        assert_eq!(t.len(), 2);
        assert!(!t.holds(&[0]) && t.holds(&[1]) && t.holds(&[2]));
        drop(t);
        let _ = fs::remove_dir_all(&dir);
    }
}
