//! NVMe tier below the context store (ninfer's disk tier, off by default): entries the store
//! evicts while they can still be resumed are copied to disk by a background thread, and a
//! request whose deepest match is only on disk reads it back into system RAM when that covers
//! enough more than every resident source.
//!
//! Records live in one process-private directory removed on shutdown; their identity (keys,
//! KV ranges) stays in memory, so matching never reads the disk. Disk capacity is reclaimed in
//! least-recently-used order (a write or a read counts as a use).

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread::JoinHandle;

/// Bytes of pinned host memory the writer copies from. The store keeps the owning entry alive,
/// and unchanged, until the write completes.
struct Part(*const u8, usize);
// SAFETY: see `Part`; the memory outlives the job and nothing writes it meanwhile.
unsafe impl Send for Part {}

struct Job {
    id: u64,
    path: PathBuf,
    parts: Vec<Part>,
}

pub(crate) struct Record {
    pub id: u64,
    pub keys: Vec<u32>,
    pub rope_delta: i64,
    /// KV segment ranges, in file order after the state.
    pub segs: Vec<(usize, usize)>,
    pub root: bool,
    bytes: u64,
    last: u64,
    /// Written completely (in flight until then).
    pub ready: bool,
}

pub(crate) struct DiskTier {
    dir: PathBuf,
    budget: u64,
    records: Vec<Record>,
    jobs: Option<Sender<Job>>,
    done: Receiver<(u64, std::io::Result<()>)>,
    writer: Option<JoinHandle<()>>,
    next_id: u64,
}

impl DiskTier {
    /// A tier of at most `budget` bytes in a new private directory under `root`.
    pub fn new(root: PathBuf, budget: u64) -> std::io::Result<Self> {
        let dir = root.join(format!("velox-{}", std::process::id()));
        fs::create_dir_all(&dir)?;
        let (jobs, rx) = channel::<Job>();
        let (tx, done) = channel();
        let writer = std::thread::Builder::new()
            .name("velox-disk-context".into())
            .spawn(move || {
                for job in rx {
                    let r = File::create(&job.path).and_then(|mut f| {
                        for Part(p, n) in &job.parts {
                            // SAFETY: see `Part`.
                            f.write_all(unsafe { std::slice::from_raw_parts(*p, *n) })?;
                        }
                        Ok(())
                    });
                    if tx.send((job.id, r)).is_err() {
                        break;
                    }
                }
            })?;
        Ok(DiskTier {
            dir,
            budget,
            records: Vec::new(),
            jobs: Some(jobs),
            done,
            writer: Some(writer),
            next_id: 1,
        })
    }

    fn path(&self, id: u64) -> PathBuf {
        self.dir.join(format!("{id}.ctx"))
    }

    pub fn holds(&self, keys: &[u32]) -> bool {
        self.records.iter().any(|r| r.keys == keys)
    }

    /// Queues a copy of `parts` (state then KV buffers) as a record of `keys`; returns its id, or
    /// `None` when it cannot fit. The caller keeps the memory alive until [`Self::completed`]
    /// reports the id.
    #[allow(clippy::too_many_arguments)]
    pub fn write(
        &mut self,
        keys: &[u32],
        rope_delta: i64,
        segs: Vec<(usize, usize)>,
        root: bool,
        parts: Vec<(*const u8, usize)>,
        now: u64,
    ) -> Option<u64> {
        let bytes: u64 = parts.iter().map(|&(_, n)| n as u64).sum();
        if bytes > self.budget || !self.make_room(bytes) {
            return None;
        }
        let id = self.next_id;
        self.next_id += 1;
        let job = Job {
            id,
            path: self.path(id),
            parts: parts.into_iter().map(|(p, n)| Part(p, n)).collect(),
        };
        self.jobs.as_ref()?.send(job).ok()?;
        self.records.push(Record {
            id,
            keys: keys.to_vec(),
            rope_delta,
            segs,
            root,
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
                .min_by_key(|(_, r)| r.last)
                .map(|(i, _)| i)
            else {
                return false;
            };
            let r = self.records.swap_remove(i);
            let _ = fs::remove_file(self.path(r.id));
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
                    }
                }
                Err(e) => {
                    eprintln!("disk context: write failed: {e}");
                    self.records.retain(|r| r.id != id);
                    let _ = fs::remove_file(self.path(id));
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

    /// Reads record `id` into `bufs` (state then KV buffers, as written). A failed read drops
    /// the record.
    pub fn read(&mut self, id: u64, bufs: &mut [&mut [u8]], now: u64) -> std::io::Result<()> {
        let path = self.path(id);
        let r = File::open(&path).and_then(|mut f| {
            for b in bufs.iter_mut() {
                f.read_exact(b)?;
            }
            Ok(())
        });
        match &r {
            Ok(()) => {
                if let Some(rec) = self.records.iter_mut().find(|r| r.id == id) {
                    rec.last = now;
                }
            }
            Err(_) => {
                self.records.retain(|r| r.id != id);
                let _ = fs::remove_file(path);
            }
        }
        r
    }

    pub fn len(&self) -> usize {
        self.records.iter().filter(|r| r.ready).count()
    }
}

impl Drop for DiskTier {
    fn drop(&mut self) {
        self.jobs = None;
        if let Some(w) = self.writer.take() {
            let _ = w.join();
        }
        let _ = fs::remove_dir_all(&self.dir);
    }
}
