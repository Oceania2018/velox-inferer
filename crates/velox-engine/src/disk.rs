//! NVMe tier below the context store (off by default) for a cold start: the context store copies
//! its most common roots (system prompts) to disk on a background thread while they are still in
//! system RAM, and a request whose deepest match is only on disk reads it back when that covers
//! enough more than every resident source, so a restarted server finds its prompts on disk instead
//! of prefilling them. It holds at most `max_records` records.
//!
//! A record is two kinds of file. Its own (`.ctx`) holds a header describing it (keys, KV ranges)
//! then the recurrent state and drafter context. Its attention KV lives in page files (`.kvp`),
//! one per KV segment, named by the keys of the segment's whole prefix and its first row, like the
//! context store's shared pages: KV rows depend on nothing else, so records on one prefix (a
//! chain's root and its later branch points) share the pages they have in common and a new record
//! writes only the pages no other record holds. Pages are reference-counted and deleted with the
//! last record that uses them.
//!
//! With a model identity ([`DiskTier::new`] `tag`), files live in a directory named after it, the
//! build and the storage layout and outlive the process: they are indexed again at startup. A new
//! build (other kernels or code, so other numerics) starts empty: it deletes the directories of
//! every other build. Files are named by their keys, so a prefix already on disk is never written
//! again. Without one, files live in a process-private directory removed on shutdown. Matching
//! never reads the disk. Capacity is reclaimed in least-recently-used order of records (a write or
//! a read counts as a use; records found at startup rank by file age, below anything used since).

use std::collections::HashMap;
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

/// One file a job writes: a header then the parts.
struct Out {
    path: PathBuf,
    header: Vec<u8>,
    parts: Vec<Part>,
}

/// A record's new page files, then its own file (written last: a record file on disk means its
/// pages are there too).
struct Job {
    id: u64,
    files: Vec<Out>,
}

/// Host memory a record is written from: `(address, length)` buffers of the recurrent state, of
/// each KV segment (with its rows) and of the drafter context (with its rows).
pub(crate) struct Payload {
    pub state: Buffers,
    pub kv: Vec<Rows>,
    pub draft: Option<Rows>,
}

/// `(address, length)` of host buffers.
pub(crate) type Buffers = Vec<(*const u8, usize)>;
/// Rows `lo..hi` held in host buffers.
pub(crate) type Rows = ((usize, usize), Buffers);

pub(crate) struct Record {
    pub id: u64,
    pub keys: Vec<u32>,
    pub rope_delta: i64,
    /// KV segment ranges, in payload order after the state.
    pub segs: Vec<(usize, usize)>,
    /// Drafter context rows, in payload order after the KV.
    pub draft: Option<(usize, usize)>,
    pub root: bool,
    path: PathBuf,
    /// Header bytes before the state.
    offset: u64,
    /// State bytes (the drafter context follows them).
    state_bytes: u64,
    /// File size of the record's own file.
    bytes: u64,
    /// Page of each KV segment, in `segs` order.
    pages: Vec<u128>,
    /// Pages this record's write creates (forgotten again if it fails).
    created: Vec<u128>,
    /// Payload bytes its pages already on disk spared the write.
    shared_bytes: u64,
    last: u64,
    /// Written completely (in flight until then).
    pub ready: bool,
}

impl Record {
    fn payload(&self) -> u64 {
        self.bytes - self.offset
    }
}

/// A KV page file: rows `lo..hi` of a prefix.
struct Page {
    lo: usize,
    hi: usize,
    /// Header bytes before the rows.
    offset: u64,
    /// File size.
    bytes: u64,
    /// Records that use it.
    refs: u32,
    /// Written completely.
    ready: bool,
}

pub(crate) struct DiskTier {
    dir: PathBuf,
    /// Files outlive the process (see the module docs).
    persistent: bool,
    layout: u64,
    budget: u64,
    /// Records kept at most (least recently used ones go first).
    max_records: usize,
    records: Vec<Record>,
    pages: HashMap<u128, Page>,
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
    /// Bytes of complete records and their pages.
    pub used: u64,
    pub budget: u64,
    pub records: u64,
    /// KV page files of complete records.
    pub pages: u64,
    /// Records read back, their payload bytes and the prompt tokens they covered.
    pub reads: u64,
    pub read_bytes: u64,
    pub read_tokens: u64,
    /// Records written completely and the bytes written for them (record and new page files).
    pub writes: u64,
    pub written_bytes: u64,
    /// Of `writes`, roots (system prompt, task).
    pub root_writes: u64,
    /// Payload bytes the written records did not write because their pages were already on disk.
    pub shared_bytes: u64,
}

const MAGIC: &[u8; 8] = b"VXCTX\0\0\x02";
const PAGE_MAGIC: &[u8; 8] = b"VXKVP\0\0\x01";
/// Magic, layout, header length.
const FIXED: usize = 20;
/// Partial files older than this are left over from a crash (a live writer renames its own
/// within seconds).
const STALE_TMP: Duration = Duration::from_secs(3600);

pub(crate) const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

/// Identity of this build: a hash of the running executable (the kernels are linked into it).
/// 0 when it cannot be read.
pub(crate) fn executable_id() -> u64 {
    let Ok(mut f) = std::env::current_exe().and_then(File::open) else {
        return 0;
    };
    let mut buf = vec![0u8; 1 << 20];
    let mut h = FNV_OFFSET;
    loop {
        match f.read(&mut buf) {
            Ok(0) => return h,
            Ok(n) => h = fnv64(h, &buf[..n]),
            Err(_) => return 0,
        }
    }
}

/// FNV-1a: stable across processes and toolchains (file names, layout and header checks).
pub(crate) fn fnv64(seed: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(seed, |h, &x| {
        (h ^ x as u64).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// 128 bits of `bytes` (two FNV-1a streams) for file names.
fn fnv128(bytes: &[u8]) -> u128 {
    let (a, b) = (fnv64(FNV_OFFSET, bytes), fnv64(!FNV_OFFSET, bytes));
    ((a as u128) << 64) | b as u128
}

fn key_bytes(keys: &[u32]) -> Vec<u8> {
    keys.iter().flat_map(|k| k.to_le_bytes()).collect()
}

/// The page of KV rows `lo..hi` on the prefix whose key bytes are `kb` (at least `hi` keys).
fn page_id(kb: &[u8], lo: usize, hi: usize) -> u128 {
    let mut b = kb[..hi * 4].to_vec();
    b.extend_from_slice(&(lo as u64).to_le_bytes());
    fnv128(&b)
}

fn invalid(m: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, m.into())
}

/// Magic, layout, rows, the keys of those rows (a check against a name collision), payload length,
/// then a checksum of all of it.
fn page_header(layout: u64, keys: &[u32], lo: usize, hi: usize, payload: u64) -> Vec<u8> {
    let mut h = Vec::with_capacity(48 + (hi - lo) * 4);
    h.extend_from_slice(PAGE_MAGIC);
    h.extend_from_slice(&layout.to_le_bytes());
    h.extend_from_slice(&(lo as u64).to_le_bytes());
    h.extend_from_slice(&(hi as u64).to_le_bytes());
    h.extend_from_slice(&key_bytes(&keys[lo..hi]));
    h.extend_from_slice(&payload.to_le_bytes());
    let sum = fnv64(FNV_OFFSET, &h);
    h.extend_from_slice(&sum.to_le_bytes());
    h
}

/// Fills `bufs` in order, from `cursor` (buffer, byte) on, with `len` bytes of `f`.
fn fill(
    f: &mut File,
    bufs: &mut [&mut [u8]],
    cursor: &mut (usize, usize),
    mut len: u64,
) -> std::io::Result<()> {
    while len > 0 {
        let (i, at) = *cursor;
        let buf = bufs
            .get_mut(i)
            .ok_or_else(|| invalid("storage too small"))?;
        let n = ((buf.len() - at) as u64).min(len) as usize;
        f.read_exact(&mut buf[at..at + n])?;
        len -= n as u64;
        *cursor = if at + n == buf.len() {
            (i + 1, 0)
        } else {
            (i, at + n)
        };
    }
    Ok(())
}

impl DiskTier {
    /// A tier of at most `budget` bytes and `max_records` records under `root`: in `<tag>-b<build>-<layout>` when `tag` (the
    /// model's identity) is given, holding the records an earlier process of the same build left
    /// there, else in a new private directory. `build` identifies the code and kernels that
    /// computed the records; a persistent tier deletes the directories other builds left under
    /// `root` (a process of another build sharing `root` loses its records). `layout` identifies
    /// the storage shapes the records must match.
    pub fn new(
        root: PathBuf,
        budget: u64,
        max_records: usize,
        tag: Option<&str>,
        build: u64,
        layout: u64,
    ) -> std::io::Result<Self> {
        let dir = match tag {
            Some(t) => {
                let t: String = t
                    .chars()
                    .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
                    .collect();
                purge_other_builds(&root, build);
                root.join(format!("{t}-b{build:016x}-{layout:016x}"))
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
                    let r = job.files.iter().try_for_each(write_file);
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
            max_records,
            records: Vec::new(),
            pages: HashMap::new(),
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
    /// that do not match this layout, records missing a page, pages no record uses and leftovers
    /// of interrupted writes.
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
            let ext = path.extension().and_then(|e| e.to_str());
            let parsed = match ext {
                Some("ctx") => self.read_header(&path).map(|r| found.push((mtime, r))),
                Some("kvp") => self.read_page_header(&path).map(|(id, p)| {
                    self.pages.insert(id, p);
                }),
                Some(e) => {
                    if e.starts_with("tmp") && mtime.elapsed().is_ok_and(|age| age > STALE_TMP) {
                        let _ = fs::remove_file(&path);
                    }
                    Ok(())
                }
                None => Ok(()),
            };
            if let Err(e) = parsed {
                eprintln!("disk context: dropping {}: {e}", path.display());
                let _ = fs::remove_file(&path);
            }
        }
        found.sort_by_key(|(t, _)| *t);
        for (_, mut r) in found {
            let complete = r.segs.iter().zip(&r.pages).all(|(&(lo, hi), id)| {
                self.pages.get(id).is_some_and(|p| (p.lo, p.hi) == (lo, hi))
            });
            if !complete {
                eprintln!("disk context: dropping {}: missing pages", r.path.display());
                let _ = fs::remove_file(&r.path);
                continue;
            }
            for id in &r.pages {
                self.pages.get_mut(id).expect("checked").refs += 1;
            }
            r.id = self.next_id;
            self.next_id += 1;
            self.records.push(r);
        }
        let dir = &self.dir;
        self.pages.retain(|id, p| {
            if p.refs == 0 {
                let _ = fs::remove_file(page_path(dir, *id));
            }
            p.refs > 0
        });
        // The budget may be smaller than last time.
        self.make_room(0, false);
        if !self.records.is_empty() {
            eprintln!(
                "disk context: {} records, {} KV pages ({:.2} GB) in {}",
                self.records.len(),
                self.pages.len(),
                self.used(false) as f64 / 1e9,
                self.dir.display()
            );
        }
    }

    /// Magic, layout, header length, payload length, rope delta, root flag, keys, KV ranges,
    /// drafter range, state length, then a checksum of all of it.
    #[allow(clippy::too_many_arguments)]
    fn header(
        &self,
        keys: &[u32],
        rope_delta: i64,
        segs: &[(usize, usize)],
        draft: Option<(usize, usize)>,
        root: bool,
        state_bytes: u64,
        payload: u64,
    ) -> Vec<u8> {
        let mut h = Vec::with_capacity(88 + keys.len() * 4 + segs.len() * 16);
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
        h.extend_from_slice(&state_bytes.to_le_bytes());
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
        let keys: Vec<u32> = take(n * 4)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| u32::from_le_bytes(c))
            .collect();
        let n = u32_of(take(4)?);
        let mut segs = Vec::with_capacity(n.min(1 << 16));
        for _ in 0..n {
            let (lo, hi) = (u64_of(take(8)?) as usize, u64_of(take(8)?) as usize);
            if lo >= hi || hi > keys.len() {
                return Err(invalid("bad KV range"));
            }
            segs.push((lo, hi));
        }
        let has_draft = take(1)?[0] != 0;
        let (lo, hi) = (u64_of(take(8)?), u64_of(take(8)?));
        let state_bytes = u64_of(take(8)?);
        if total as u64 + payload != len || state_bytes > payload {
            return Err(invalid("truncated"));
        }
        let kb = key_bytes(&keys);
        let pages = segs.iter().map(|&(lo, hi)| page_id(&kb, lo, hi)).collect();
        Ok(Record {
            id: 0,
            keys,
            rope_delta,
            segs,
            draft: has_draft.then_some((lo as usize, hi as usize)),
            root,
            path: path.to_path_buf(),
            offset: total as u64,
            state_bytes,
            bytes: len,
            pages,
            created: Vec::new(),
            shared_bytes: 0,
            last: 0,
            ready: true,
        })
    }

    /// The page a file holds, checked against this layout, its name and the file's length.
    fn read_page_header(&self, path: &Path) -> std::io::Result<(u128, Page)> {
        let id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| u128::from_str_radix(s, 16).ok())
            .ok_or_else(|| invalid("not a page name"))?;
        let mut f = File::open(path)?;
        let len = f.metadata()?.len();
        let mut h = [0u8; 32];
        f.read_exact(&mut h)?;
        if &h[..8] != PAGE_MAGIC {
            return Err(invalid("not a KV page"));
        }
        let u64_of = |s: &[u8]| u64::from_le_bytes(s.try_into().unwrap());
        if u64_of(&h[8..16]) != self.layout {
            return Err(invalid("another storage layout"));
        }
        let (lo, hi) = (u64_of(&h[16..24]) as usize, u64_of(&h[24..32]) as usize);
        if lo >= hi || hi - lo > 1 << 20 {
            return Err(invalid("bad KV range"));
        }
        let offset = (32 + (hi - lo) * 4 + 16) as u64;
        if offset > len {
            return Err(invalid("truncated"));
        }
        let mut rest = vec![0u8; offset as usize - 32];
        f.read_exact(&mut rest)?;
        let body = [&h[..], &rest[..rest.len() - 8]].concat();
        if fnv64(FNV_OFFSET, &body).to_le_bytes()[..] != rest[rest.len() - 8..] {
            return Err(invalid("header checksum"));
        }
        if offset + u64_of(&body[body.len() - 8..]) != len {
            return Err(invalid("truncated"));
        }
        Ok((
            id,
            Page {
                lo,
                hi,
                offset,
                bytes: len,
                refs: 0,
                ready: true,
            },
        ))
    }

    fn path_for(&self, id: u64, keys: &[u32]) -> PathBuf {
        if self.persistent {
            // 128 bits of the keys: a prefix has one file whichever process wrote it.
            self.dir
                .join(format!("{:032x}.ctx", fnv128(&key_bytes(keys))))
        } else {
            self.dir.join(format!("{id}.ctx"))
        }
    }

    /// Whether a record (ready or being written) holds exactly `keys`.
    pub fn holds(&self, keys: &[u32]) -> bool {
        self.records.iter().any(|r| r.keys == keys)
    }

    /// Bytes of records and pages, complete ones only or also those being written.
    fn used(&self, ready_only: bool) -> u64 {
        let records = self.records.iter().filter(|r| r.ready || !ready_only);
        let pages = self.pages.values().filter(|p| p.ready || !ready_only);
        records.map(|r| r.bytes).sum::<u64>() + pages.map(|p| p.bytes).sum::<u64>()
    }

    /// Queues a copy of `payload` as a record of `keys`, writing only the KV pages not on disk
    /// yet; returns its id, or `None` when it cannot fit. The caller keeps the memory alive until
    /// [`Self::completed`] reports the id.
    pub fn write(
        &mut self,
        keys: &[u32],
        rope_delta: i64,
        root: bool,
        payload: Payload,
        now: u64,
    ) -> Option<u64> {
        let sum = |p: &[(*const u8, usize)]| p.iter().map(|&(_, n)| n as u64).sum::<u64>();
        let kb = key_bytes(keys);
        let segs: Vec<(usize, usize)> = payload.kv.iter().map(|(r, _)| *r).collect();
        if segs.iter().any(|&(lo, hi)| lo >= hi || hi > keys.len()) {
            return None;
        }
        let pages: Vec<u128> = segs.iter().map(|&(lo, hi)| page_id(&kb, lo, hi)).collect();
        // Pages on disk (or being written) that this record shares, and those it writes.
        let mut shared = Vec::new();
        let mut new = Vec::new();
        let mut shared_bytes = 0;
        for (((lo, hi), parts), &id) in payload.kv.into_iter().zip(&pages) {
            let len = sum(&parts);
            match self.pages.get(&id) {
                Some(p) if (p.lo, p.hi, p.bytes - p.offset) == (lo, hi, len) => {
                    shared.push(id);
                    shared_bytes += len;
                }
                // Same name, other rows or size: a collision, or storage of another shape.
                Some(_) => return None,
                None if new.iter().any(|(n, _, _)| *n == id) => return None,
                None => new.push((id, page_header(self.layout, keys, lo, hi, len), parts)),
            }
        }
        let draft = payload.draft.as_ref().map(|(r, _)| *r);
        let state_bytes = sum(&payload.state);
        let own = state_bytes + payload.draft.as_ref().map_or(0, |(_, p)| sum(p));
        let header = self.header(keys, rope_delta, &segs, draft, root, state_bytes, own);
        let offset = header.len() as u64;
        let bytes = offset + own;
        let need = bytes
            + new
                .iter()
                .map(|(_, h, p)| h.len() as u64 + sum(p))
                .sum::<u64>();
        if need > self.budget {
            return None;
        }
        // Held first, so making room never deletes them.
        for id in &shared {
            self.pages.get_mut(id).expect("checked").refs += 1;
        }
        if !self.make_room(need, true) {
            for id in &shared {
                self.pages.get_mut(id).expect("held").refs -= 1;
            }
            return None;
        }
        let id = self.next_id;
        self.next_id += 1;
        let path = self.path_for(id, keys);
        let mut files = Vec::with_capacity(new.len() + 1);
        let mut created = Vec::with_capacity(new.len());
        for (pid, h, parts) in new {
            let (lo, hi) = segs[pages.iter().position(|&x| x == pid).expect("own page")];
            let page = Page {
                lo,
                hi,
                offset: h.len() as u64,
                bytes: h.len() as u64 + sum(&parts),
                refs: 1,
                ready: false,
            };
            self.pages.insert(pid, page);
            created.push(pid);
            files.push(Out {
                path: page_path(&self.dir, pid),
                header: h,
                parts: parts.into_iter().map(|(p, n)| Part(p, n)).collect(),
            });
        }
        files.push(Out {
            path: path.clone(),
            header,
            parts: payload
                .state
                .into_iter()
                .chain(payload.draft.into_iter().flat_map(|(_, p)| p))
                .map(|(p, n)| Part(p, n))
                .collect(),
        });
        self.records.push(Record {
            id,
            keys: keys.to_vec(),
            rope_delta,
            segs,
            draft,
            root,
            path,
            offset,
            state_bytes,
            bytes,
            pages,
            created,
            shared_bytes,
            last: now,
            ready: false,
        });
        if self
            .jobs
            .as_ref()
            .and_then(|j| j.send(Job { id, files }).ok())
            .is_none()
        {
            self.forget(id);
            return None;
        }
        Some(id)
    }

    /// Evicts ready records, least recently used first, until `bytes` more fit, and one more
    /// record when `adding`.
    fn make_room(&mut self, bytes: u64, adding: bool) -> bool {
        loop {
            let records = self.records.len() + adding as usize;
            if self.used(false) + bytes <= self.budget && records <= self.max_records {
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
            self.remove(i);
        }
    }

    /// Removes record `i` and its file, and the pages no other record uses.
    fn remove(&mut self, i: usize) {
        let r = self.records.swap_remove(i);
        let _ = fs::remove_file(&r.path);
        for id in &r.pages {
            if let Some(p) = self.pages.get_mut(id) {
                p.refs -= 1;
                if p.refs == 0 {
                    self.pages.remove(id);
                    let _ = fs::remove_file(page_path(&self.dir, *id));
                }
            }
        }
    }

    /// Drops record `id` whose write failed or never started, and the pages it was creating
    /// (records sharing those find them missing when their own write completes).
    fn forget(&mut self, id: u64) {
        let Some(i) = self.records.iter().position(|r| r.id == id) else {
            return;
        };
        for pid in std::mem::take(&mut self.records[i].created) {
            if self.pages.get(&pid).is_some_and(|p| !p.ready) {
                self.pages.remove(&pid);
                let _ = fs::remove_file(page_path(&self.dir, pid));
            }
        }
        self.remove(i);
    }

    /// Ids of finished writes; failed ones are forgotten.
    pub fn completed(&mut self) -> Vec<u64> {
        let mut ids = Vec::new();
        while let Ok((id, r)) = self.done.try_recv() {
            let Some(i) = self.records.iter().position(|r| r.id == id) else {
                ids.push(id);
                continue;
            };
            // A page it shares may have belonged to an earlier write that failed.
            let r = r.and_then(|()| {
                let rec = &self.records[i];
                if rec.pages.iter().all(|p| self.pages.contains_key(p)) {
                    Ok(())
                } else {
                    Err(invalid("a shared KV page was not written"))
                }
            });
            match r {
                Ok(()) => {
                    let rec = &mut self.records[i];
                    rec.ready = true;
                    let created = std::mem::take(&mut rec.created);
                    let (bytes, shared, root) = (rec.bytes, rec.shared_bytes, rec.root);
                    let mut written = bytes;
                    for pid in created {
                        let p = self.pages.get_mut(&pid).expect("checked");
                        p.ready = true;
                        written += p.bytes;
                    }
                    self.io.writes += 1;
                    self.io.written_bytes += written;
                    self.io.shared_bytes += shared;
                    self.io.root_writes += root as u64;
                }
                Err(e) => {
                    eprintln!("disk context: write failed: {e}");
                    self.forget(id);
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

    /// Reads record `id` into `bufs` (state, KV segments, then drafter context, as written),
    /// which must cover its payload exactly. A failed read drops the record.
    pub fn read(&mut self, id: u64, bufs: &mut [&mut [u8]], now: u64) -> std::io::Result<()> {
        let Some(i) = self.records.iter().position(|r| r.id == id) else {
            return Err(std::io::ErrorKind::NotFound.into());
        };
        let r = self.read_into(&self.records[i], bufs);
        match &r {
            Ok(payload) => {
                let rec = &mut self.records[i];
                rec.last = now;
                self.io.reads += 1;
                self.io.read_bytes += payload;
                self.io.read_tokens += rec.keys.len() as u64;
            }
            Err(_) => self.remove(i),
        }
        r.map(|_| ())
    }

    /// Reads a record's payload into `bufs`; returns its bytes.
    fn read_into(&self, r: &Record, bufs: &mut [&mut [u8]]) -> std::io::Result<u64> {
        let pages: Vec<&Page> = r
            .pages
            .iter()
            .map(|id| self.pages.get(id).ok_or_else(|| invalid("missing KV page")))
            .collect::<Result<_, _>>()?;
        let payload = r.payload() + pages.iter().map(|p| p.bytes - p.offset).sum::<u64>();
        let want: u64 = bufs.iter().map(|b| b.len() as u64).sum();
        if want != payload {
            return Err(invalid(format!(
                "record holds {payload} bytes, storage {want}"
            )));
        }
        let mut cursor = (0, 0);
        let mut own = File::open(&r.path)?;
        own.seek(SeekFrom::Start(r.offset))?;
        fill(&mut own, bufs, &mut cursor, r.state_bytes)?;
        for (id, p) in r.pages.iter().zip(&pages) {
            let mut f = File::open(page_path(&self.dir, *id))?;
            let mut h = vec![0u8; p.offset as usize];
            f.read_exact(&mut h)?;
            let len = p.bytes - p.offset;
            if h != page_header(self.layout, &r.keys, p.lo, p.hi, len) {
                return Err(invalid("KV page of another prefix"));
            }
            fill(&mut f, bufs, &mut cursor, len)?;
        }
        fill(&mut own, bufs, &mut cursor, r.payload() - r.state_bytes)?;
        Ok(payload)
    }

    pub fn len(&self) -> usize {
        self.records.iter().filter(|r| r.ready).count()
    }

    pub fn stats(&self) -> DiskStats {
        DiskStats {
            used: self.used(true),
            budget: self.budget,
            records: self.len() as u64,
            pages: self.pages.values().filter(|p| p.ready).count() as u64,
            ..self.io
        }
    }
}

fn page_path(dir: &Path, id: u128) -> PathBuf {
    dir.join(format!("{id:032x}.kvp"))
}

/// Writes a file aside and renames it: a crash or another process never sees a partial file under
/// its final name.
fn write_file(out: &Out) -> std::io::Result<()> {
    let tmp = out
        .path
        .with_extension(format!("tmp{}", std::process::id()));
    let r = File::create(&tmp)
        .and_then(|mut f| {
            f.write_all(&out.header)?;
            for Part(p, n) in &out.parts {
                // SAFETY: see `Part`.
                f.write_all(unsafe { std::slice::from_raw_parts(*p, *n) })?;
            }
            Ok(())
        })
        .and_then(|()| fs::rename(&tmp, &out.path));
    if r.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    r
}

/// Deletes the persistent directories under `root` that another build wrote (and those of the
/// naming before builds were part of it), leaving this build's and private ones.
fn purge_other_builds(root: &Path, build: u64) {
    let Ok(dir) = fs::read_dir(root) else {
        return;
    };
    let ours = format!("-b{build:016x}-");
    for ent in dir.flatten() {
        let name = ent.file_name().to_string_lossy().into_owned();
        let private = name
            .strip_prefix("velox-")
            .is_some_and(|pid| pid.bytes().all(|c| c.is_ascii_digit()));
        if private || name.contains(&ours) || !ent.file_type().is_ok_and(|t| t.is_dir()) {
            continue;
        }
        match fs::remove_dir_all(ent.path()) {
            Ok(()) => eprintln!("disk context: deleted {name} (another build)"),
            Err(e) => eprintln!("disk context: cannot delete {name}: {e}"),
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
        DiskTier::new(dir.to_path_buf(), budget, 100, Some("test"), 7, 42).unwrap()
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("velox-disk-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn wait(t: &mut DiskTier) {
        while t.records.iter().any(|r| !r.ready) {
            t.completed();
            std::thread::yield_now();
        }
    }

    fn part(b: &[u8]) -> Vec<(*const u8, usize)> {
        vec![(b.as_ptr(), b.len())]
    }

    /// A record of `state`, one KV segment per `(range, bytes)` and no drafter context.
    fn payload(state: &[u8], kv: &[((usize, usize), &[u8])]) -> Payload {
        Payload {
            state: part(state),
            kv: kv.iter().map(|&(r, b)| (r, part(b))).collect(),
            draft: None,
        }
    }

    /// Files with extension `ext` in the tier directories under `dir`.
    fn files(dir: &Path, ext: &str) -> usize {
        fs::read_dir(dir)
            .unwrap()
            .flatten()
            .flat_map(|e| fs::read_dir(e.path()).unwrap().flatten())
            .filter(|e| e.path().extension().is_some_and(|x| x == ext))
            .count()
    }

    #[test]
    fn records_outlive_the_process_and_are_not_rewritten() {
        let dir = scratch("test");
        let keys: Vec<u32> = (0..300).collect();
        let (state, kv0, kv1, draft) = (vec![7u8; 64], vec![9u8; 128], vec![8u8; 32], [6u8; 16]);
        {
            let mut t = tier(&dir, 1 << 20);
            let mut p = payload(&state, &[((0, 256), &kv0), ((256, 300), &kv1)]);
            p.draft = Some(((200, 300), part(&draft)));
            assert!(t.write(&keys, -3, true, p, 1).is_some());
            wait(&mut t);
            assert!(t.holds(&keys));
        }
        // A new process (tier) indexes it again and reads it back.
        let mut t = tier(&dir, 1 << 20);
        assert_eq!(t.len(), 1);
        let r = t.best(&[keys.clone(), vec![1, 2]].concat(), 1000).unwrap();
        let (id, segs, rope, root, d) = (r.id, r.segs.clone(), r.rope_delta, r.root, r.draft);
        assert_eq!((segs, rope, root), (vec![(0, 256), (256, 300)], -3, true));
        assert_eq!(d, Some((200, 300)));
        let (mut a, mut b, mut c) = (vec![0u8; 64], vec![0u8; 128], vec![0u8; 32]);
        let mut e = [0u8; 16];
        t.read(id, &mut [&mut a[..], &mut b[..], &mut c[..], &mut e[..]], 2)
            .unwrap();
        assert_eq!((a, b, c, e), (state, kv0, kv1, draft));
        // Storage of another size is refused (and the record dropped).
        let mut c = [0u8; 10];
        assert!(t.read(id, &mut [&mut c[..]], 3).is_err());
        assert_eq!(t.len(), 0);
        drop(t);
        // Its pages went with it.
        assert_eq!(files(&dir, "kvp"), 0);
        // Another layout never sees the first one's directory.
        let other = DiskTier::new(dir.clone(), 1 << 20, 100, Some("test"), 7, 43).unwrap();
        assert_eq!(other.len(), 0);
        drop(other);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn records_on_one_prefix_share_their_kv_pages() {
        let dir = scratch("share");
        let keys: Vec<u32> = (0..600).collect();
        let (s1, s2) = ([1u8; 64], [2u8; 64]);
        let (p0, p1, tail1, p2) = ([10u8; 100], [11u8; 100], [12u8; 40], [13u8; 100]);
        let mut t = tier(&dir, 1 << 20);
        // A root, then a deeper branch point on it that shares its two full pages.
        let root = payload(
            &s1,
            &[((0, 256), &p0), ((256, 512), &p1), ((512, 520), &tail1)],
        );
        t.write(&keys[..520], 0, true, root, 1).unwrap();
        wait(&mut t);
        let before = t.stats();
        let branch = payload(
            &s2,
            &[((0, 256), &p0), ((256, 512), &p1), ((512, 600), &p2)],
        );
        let id = t.write(&keys, 0, false, branch, 2).unwrap();
        wait(&mut t);
        let s = t.stats();
        assert_eq!((s.records, s.pages, s.writes, s.root_writes), (2, 4, 2, 1));
        assert_eq!(s.shared_bytes, 200);
        let written = s.written_bytes - before.written_bytes;
        assert!(
            written < 64 + 100 + 4096,
            "state, one new page and headers: {written}"
        );
        // The branch reads back the pages the root wrote.
        let (mut a, mut b, mut c, mut d) = ([0u8; 64], [0u8; 100], [0u8; 100], [0u8; 100]);
        t.read(id, &mut [&mut a[..], &mut b[..], &mut c[..], &mut d[..]], 3)
            .unwrap();
        assert_eq!((a, b, c, d), (s2, p0, p1, p2));
        drop(t);
        // A new process finds the sharing again; the root's eviction keeps the shared pages.
        let mut t = tier(&dir, 1 << 20);
        assert_eq!((t.len(), t.stats().pages), (2, 4));
        let i = t.records.iter().position(|r| r.root).unwrap();
        t.remove(i);
        assert_eq!((t.len(), t.stats().pages), (1, 3));
        let id = t.best(&keys, 1000).unwrap().id;
        let (mut a, mut b, mut c, mut d) = ([0u8; 64], [0u8; 100], [0u8; 100], [0u8; 100]);
        t.read(id, &mut [&mut a[..], &mut b[..], &mut c[..], &mut d[..]], 4)
            .unwrap();
        assert_eq!((a, b, c, d), (s2, p0, p1, p2));
        let i = t.records.iter().position(|r| r.id == id).unwrap();
        t.remove(i);
        assert_eq!(t.stats().pages, 0);
        drop(t);
        assert_eq!(files(&dir, "kvp") + files(&dir, "ctx"), 0);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_failed_write_drops_the_records_sharing_its_pages() {
        let dir = scratch("fail");
        let keys: Vec<u32> = (0..300).collect();
        let (s, p0, p1) = ([1u8; 8], [2u8; 100], [3u8; 50]);
        let mut t = tier(&dir, 1 << 20);
        // Two records queued with the writer stopped: the second shares the first's page.
        let writer = t.jobs.take();
        let (tx, rx) = channel::<Job>();
        t.jobs = Some(tx);
        let a = t
            .write(&keys[..256], 0, true, payload(&s, &[((0, 256), &p0)]), 1)
            .unwrap();
        let b = t
            .write(
                &keys,
                0,
                false,
                payload(&s, &[((0, 256), &p0), ((256, 300), &p1)]),
                1,
            )
            .unwrap();
        assert_eq!(
            t.records.iter().find(|r| r.id == b).unwrap().shared_bytes,
            100
        );
        assert_eq!(
            t.records.iter().find(|r| r.id == b).unwrap().created.len(),
            1
        );
        // The first fails; the second's own files are written.
        t.forget(a);
        let job_b = rx.try_iter().find(|j| j.id == b).unwrap();
        writer.as_ref().unwrap().send(job_b).unwrap();
        while t.records.iter().any(|r| !r.ready) {
            t.completed();
            std::thread::yield_now();
        }
        assert_eq!((t.len(), t.stats().pages), (0, 0));
        assert_eq!(t.stats().writes, 0);
        t.jobs = writer;
        drop(t);
        assert_eq!(files(&dir, "kvp") + files(&dir, "ctx"), 0);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_new_build_deletes_the_records_of_other_builds() {
        let dir = scratch("build");
        let keys: Vec<u32> = (0..50).collect();
        let data = [3u8; 64];
        {
            let mut t = tier(&dir, 1 << 20);
            t.write(&keys, 0, true, payload(&data, &[((0, 50), &data)]), 1);
            wait(&mut t);
        }
        // Another model of the same build, the old unversioned naming and a live private tier.
        drop(DiskTier::new(dir.clone(), 1 << 20, 100, Some("other"), 7, 42).unwrap());
        fs::create_dir_all(dir.join("test-000000000000002a")).unwrap();
        fs::create_dir_all(dir.join("velox-12345")).unwrap();
        assert_eq!(tier(&dir, 1 << 20).len(), 1, "same build keeps its records");
        let next = DiskTier::new(dir.clone(), 1 << 20, 100, Some("test"), 8, 42).unwrap();
        assert_eq!(next.len(), 0);
        drop(next);
        let mut left: Vec<String> = fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        left.sort();
        assert_eq!(
            left,
            ["test-b0000000000000008-000000000000002a", "velox-12345"]
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn stats_count_complete_writes_and_successful_reads() {
        let dir = scratch("stats");
        let keys: Vec<u32> = (0..40).collect();
        let (state, kv) = ([5u8; 60], [4u8; 40]);
        let mut t = tier(&dir, 1 << 20);
        let id = t
            .write(&keys, 0, false, payload(&state, &[((0, 40), &kv)]), 1)
            .unwrap();
        wait(&mut t);
        let s = t.stats();
        assert_eq!(
            (s.records, s.pages, s.writes, s.root_writes, s.reads),
            (1, 1, 1, 0, 0)
        );
        assert_eq!(
            (s.used, s.budget, s.shared_bytes),
            (s.written_bytes, 1 << 20, 0)
        );
        assert!(s.used > 100, "headers and payload");
        let (mut a, mut b) = ([0u8; 60], [0u8; 40]);
        t.read(id, &mut [&mut a[..], &mut b[..]], 2).unwrap();
        let s = t.stats();
        assert_eq!((s.reads, s.read_bytes, s.read_tokens), (1, 100, 40));
        // A failed read counts nothing (and drops the record).
        let mut short = [0u8; 1];
        assert!(t.read(id, &mut [&mut short[..]], 3).is_err());
        let s = t.stats();
        assert_eq!((s.reads, s.records, s.pages, s.used), (1, 0, 0, 0));
        drop(t);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_smaller_budget_drops_the_oldest_records_at_startup() {
        let dir = scratch("budget");
        let data = vec![1u8; 1000];
        {
            let mut t = tier(&dir, 1 << 20);
            for i in 0..3u32 {
                t.write(&[i], 0, true, payload(&data, &[]), 1);
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

    #[test]
    fn the_tier_keeps_at_most_max_records() {
        let dir = scratch("max");
        let data = [1u8; 100];
        {
            let mut t = DiskTier::new(dir.clone(), 1 << 20, 2, Some("test"), 7, 42).unwrap();
            for i in 0..3u32 {
                t.write(&[i], 0, true, payload(&data, &[]), i as u64);
                wait(&mut t);
            }
            // The least recently used one made room.
            assert!(!t.holds(&[0]) && t.holds(&[1]) && t.holds(&[2]));
        }
        // A smaller limit at startup drops the oldest files.
        let t = DiskTier::new(dir.clone(), 1 << 20, 1, Some("test"), 7, 42).unwrap();
        assert_eq!(t.len(), 1);
        drop(t);
        assert_eq!(files(&dir, "ctx"), 1);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_record_missing_a_page_is_dropped_at_startup() {
        let dir = scratch("orphan");
        let keys: Vec<u32> = (0..10).collect();
        let (s, kv) = ([1u8; 8], [2u8; 8]);
        {
            let mut t = tier(&dir, 1 << 20);
            t.write(&keys, 0, true, payload(&s, &[((0, 10), &kv)]), 1);
            wait(&mut t);
        }
        for e in fs::read_dir(&dir).unwrap().flatten() {
            for f in fs::read_dir(e.path()).unwrap().flatten() {
                if f.path().extension().is_some_and(|x| x == "kvp") {
                    fs::remove_file(f.path()).unwrap();
                }
            }
        }
        let t = tier(&dir, 1 << 20);
        assert_eq!(t.len(), 0);
        drop(t);
        assert_eq!(files(&dir, "ctx"), 0);
        let _ = fs::remove_dir_all(&dir);
    }
}
