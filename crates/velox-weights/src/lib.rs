//! Weight containers.
//!
//! - `.vi` v1 is velox's native format (docs/weights.md). It deliberately shares the directory
//!   model of ninfer's v3 container (objects, bindings, uses, components, multi-file payload) so
//!   that importing a `.ninfer` is lossless: a header rewrite plus a payload copy.
//! - `.ninfer` v3 is read directly for compatibility, following ninfer's container spec.
//!
//! [`Artifact::open`] performs the generic structural checks (framing, file table, object
//! ranges, binding/use/resource references). Interpreting configs, formats and layouts belongs to
//! the model loader and ops.

use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

pub const NINFER_ENTRY_MAGIC: [u8; 8] = *b"NINFER\0\x03";
pub const NINFER_PART_MAGIC: [u8; 8] = *b"NINPRT\0\x03";
pub const VI_ENTRY_MAGIC: [u8; 8] = *b"VELOXVI\x01";
pub const VI_PART_MAGIC: [u8; 8] = *b"VELOXPT\x01";
const HEADER: u64 = 32;
const PAGE: u64 = 4096;
/// Default single-file size limit (decimal 32 GB), as in the ninfer writer.
pub const DEFAULT_PART_LIMIT: u64 = 32_000_000_000;

#[derive(Debug)]
pub enum Error {
    Io(PathBuf, io::Error),
    Format(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(p, e) => write!(f, "{}: {e}", p.display()),
            Error::Format(m) => write!(f, "invalid artifact: {m}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

fn bad<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::Format(msg.into()))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Container {
    NinferV3,
    VeloxV1,
}

impl Container {
    fn magics(self) -> ([u8; 8], [u8; 8]) {
        match self {
            Container::NinferV3 => (NINFER_ENTRY_MAGIC, NINFER_PART_MAGIC),
            Container::VeloxV1 => (VI_ENTRY_MAGIC, VI_PART_MAGIC),
        }
    }
}

impl fmt::Display for Container {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Container::NinferV3 => "ninfer v3",
            Container::VeloxV1 => "velox vi v1",
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ObjectKind {
    Tensor {
        shape: Vec<u64>,
        format: String,
        layout: String,
    },
    Resource {
        encoding: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Object {
    pub id: String,
    pub kind: ObjectKind,
    /// Byte offset in the logical payload.
    pub offset: u64,
    pub bytes: u64,
}

impl Object {
    /// Logical elements of a tensor (1 for a scalar); None for resources.
    pub fn elements(&self) -> Option<u64> {
        match &self.kind {
            ObjectKind::Tensor { shape, .. } => Some(shape.iter().product()),
            ObjectKind::Resource { .. } => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FileEntry {
    pub path: PathBuf,
    /// Byte offset of this file's payload within the file.
    pub payload_start: u64,
    pub payload_bytes: u64,
    /// Start of this file's slice of the logical payload.
    pub logical_start: u64,
}

pub struct Artifact {
    pub path: PathBuf,
    pub container: Container,
    pub artifact_id: [u8; 16],
    /// The complete JSON directory, kept verbatim for conversion and inspection.
    pub directory: Map<String, Value>,
    pub files: Vec<FileEntry>,
    pub objects: Vec<Object>,
    index: HashMap<String, usize>,
}

fn u64_field(v: &Value, what: &str) -> Result<u64> {
    v.as_u64()
        .map_or_else(|| bad(format!("{what} must be a non-negative integer")), Ok)
}

fn str_field<'a>(o: &'a Map<String, Value>, key: &str, what: &str) -> Result<&'a str> {
    match o.get(key).and_then(Value::as_str) {
        Some(s) if !s.is_empty() && !s.contains('\0') => Ok(s),
        _ => bad(format!("{what}.{key} must be a non-empty string")),
    }
}

fn read_header(path: &Path) -> Result<(File, [u8; 32])> {
    let mut f = File::open(path).map_err(|e| Error::Io(path.into(), e))?;
    let mut h = [0u8; 32];
    f.read_exact(&mut h)
        .map_err(|e| Error::Io(path.into(), e))?;
    Ok((f, h))
}

fn le_u64(b: &[u8]) -> u64 {
    u64::from_le_bytes(b.try_into().unwrap())
}

pub fn align_up(x: u64, a: u64) -> u64 {
    x.div_ceil(a) * a
}

pub fn hex(id: &[u8]) -> String {
    id.iter().map(|b| format!("{b:02x}")).collect()
}

impl Artifact {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let (mut f, h) = read_header(&path)?;
        let container = match &h[..8] {
            m if m == NINFER_ENTRY_MAGIC => Container::NinferV3,
            m if m == VI_ENTRY_MAGIC => Container::VeloxV1,
            m => return bad(format!("unknown magic {}", hex(m))),
        };
        let json_bytes = le_u64(&h[8..16]);
        if json_bytes == 0 || json_bytes > 1 << 32 {
            return bad(format!("json_bytes {json_bytes} out of range"));
        }
        let artifact_id: [u8; 16] = h[16..32].try_into().unwrap();
        let mut json = vec![0u8; json_bytes as usize];
        f.read_exact(&mut json)
            .map_err(|e| Error::Io(path.clone(), e))?;
        let directory = match serde_json::from_slice::<Value>(&json) {
            Ok(Value::Object(m)) => m,
            Ok(_) => return bad("directory is not a JSON object"),
            Err(e) => return bad(format!("directory JSON: {e}")),
        };
        let entry_payload_start = align_up(HEADER + json_bytes, PAGE);

        let allowed: &[&str] = match container {
            Container::NinferV3 => &[
                "components",
                "objects",
                "bindings",
                "uses",
                "files",
                "metadata",
                "provenance",
            ],
            Container::VeloxV1 => &[
                "components",
                "objects",
                "bindings",
                "uses",
                "files",
                "metadata",
                "provenance",
                "velox",
            ],
        };
        if let Some(k) = directory.keys().find(|k| !allowed.contains(&k.as_str())) {
            return bad(format!("unknown root field '{k}'"));
        }

        let files = Self::parse_files(
            &path,
            container,
            &artifact_id,
            &directory,
            entry_payload_start,
        )?;
        let payload_bytes = files.iter().map(|f| f.payload_bytes).sum::<u64>();
        let objects = Self::parse_objects(&directory, payload_bytes)?;
        let index = objects
            .iter()
            .enumerate()
            .map(|(i, o)| (o.id.clone(), i))
            .collect();
        let a = Artifact {
            path,
            container,
            artifact_id,
            directory,
            files,
            objects,
            index,
        };
        a.check_references()?;
        Ok(a)
    }

    fn parse_files(
        path: &Path,
        container: Container,
        id: &[u8; 16],
        dir: &Map<String, Value>,
        entry_payload_start: u64,
    ) -> Result<Vec<FileEntry>> {
        let list = match dir.get("files").and_then(Value::as_array) {
            Some(l) if !l.is_empty() => l,
            _ => return bad("files must be a non-empty array"),
        };
        let base = path.parent().unwrap_or(Path::new("."));
        let (_, part_magic) = container.magics();
        let mut out = Vec::with_capacity(list.len());
        let mut logical = 0u64;
        let mut seen = Vec::new();
        for (i, item) in list.iter().enumerate() {
            let o = item
                .as_object()
                .ok_or_else(|| Error::Format(format!("files[{i}] is not an object")))?;
            let payload_bytes = u64_field(
                o.get("payload_bytes").unwrap_or(&Value::Null),
                &format!("files[{i}].payload_bytes"),
            )?;
            if payload_bytes == 0 {
                return bad(format!("files[{i}].payload_bytes must be positive"));
            }
            let (file_path, payload_start) = if i == 0 {
                if !o.get("path").is_some_and(Value::is_null) {
                    return bad("files[0].path must be null");
                }
                (path.to_path_buf(), entry_payload_start)
            } else {
                let name = str_field(o, "path", &format!("files[{i}]"))?;
                if name.contains(['/', '\\']) || name == "." || name == ".." || seen.contains(&name)
                {
                    return bad(format!(
                        "files[{i}].path '{name}' is not a unique sibling file name"
                    ));
                }
                seen.push(name);
                let p = base.join(name);
                let (_, h) = read_header(&p)?;
                if h[..8] != part_magic || le_u64(&h[8..16]) != i as u64 || &h[16..32] != id {
                    return bad(format!("{} is not part {i} of this artifact", p.display()));
                }
                (p, PAGE)
            };
            let len = std::fs::metadata(&file_path)
                .map_err(|e| Error::Io(file_path.clone(), e))?
                .len();
            if len != payload_start + payload_bytes {
                return bad(format!(
                    "{} is {len} bytes, directory says {}",
                    file_path.display(),
                    payload_start + payload_bytes
                ));
            }
            out.push(FileEntry {
                path: file_path,
                payload_start,
                payload_bytes,
                logical_start: logical,
            });
            logical = logical
                .checked_add(payload_bytes)
                .ok_or_else(|| Error::Format("payload size overflows".into()))?;
        }
        Ok(out)
    }

    fn parse_objects(dir: &Map<String, Value>, payload_bytes: u64) -> Result<Vec<Object>> {
        let list = match dir.get("objects").and_then(Value::as_array) {
            Some(l) if !l.is_empty() => l,
            _ => return bad("objects must be a non-empty array"),
        };
        let mut out: Vec<Object> = Vec::with_capacity(list.len());
        let mut ids = std::collections::HashSet::new();
        for (i, item) in list.iter().enumerate() {
            let what = format!("objects[{i}]");
            let o = item
                .as_object()
                .ok_or_else(|| Error::Format(format!("{what} is not an object")))?;
            let id = str_field(o, "id", &what)?.to_string();
            let offset = u64_field(
                o.get("offset").unwrap_or(&Value::Null),
                &format!("{what}.offset"),
            )?;
            let bytes = u64_field(
                o.get("bytes").unwrap_or(&Value::Null),
                &format!("{what}.bytes"),
            )?;
            let kind = match o.get("kind").and_then(Value::as_str) {
                Some("tensor") => {
                    let shape = o
                        .get("shape")
                        .and_then(Value::as_array)
                        .filter(|s| s.len() <= 16)
                        .ok_or_else(|| {
                            Error::Format(format!("{what}.shape must be an array of rank 0..16"))
                        })?
                        .iter()
                        .map(|d| {
                            d.as_u64().filter(|&d| d > 0).ok_or_else(|| {
                                Error::Format(format!("{what}.shape has a non-positive dimension"))
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let format = str_field(o, "format", &what)?.to_string();
                    let layout = str_field(o, "layout", &what)?.to_string();
                    if offset % 256 != 0 {
                        return bad(format!(
                            "tensor {id} offset {offset} is not 256-byte aligned"
                        ));
                    }
                    ObjectKind::Tensor {
                        shape,
                        format,
                        layout,
                    }
                }
                Some("resource") => ObjectKind::Resource {
                    encoding: str_field(o, "encoding", &what)?.to_string(),
                },
                other => return bad(format!("{what}.kind {other:?} is not tensor or resource")),
            };
            if bytes == 0
                || offset
                    .checked_add(bytes)
                    .is_none_or(|end| end > payload_bytes)
            {
                return bad(format!(
                    "object {id} [{offset}, +{bytes}) is outside the payload"
                ));
            }
            if let Some(prev) = out.last() {
                if offset < prev.offset + prev.bytes {
                    return bad(format!("object {id} overlaps or precedes {}", prev.id));
                }
            }
            if !ids.insert(id.clone()) {
                return bad(format!("duplicate object id {id}"));
            }
            out.push(Object {
                id,
                kind,
                offset,
                bytes,
            });
        }
        Ok(out)
    }

    fn tensor(&self, id: &str, what: &str) -> Result<&Object> {
        match self.object(id) {
            Some(o) if matches!(o.kind, ObjectKind::Tensor { .. }) => Ok(o),
            Some(_) => bad(format!(
                "{what} references resource '{id}' where a tensor is required"
            )),
            None => bad(format!("{what} references missing object '{id}'")),
        }
    }

    /// Checks one Binding: `{object}` or `{parts: [{object, range}]}`.
    fn check_binding(&self, b: &Value, what: &str) -> Result<()> {
        let o = b
            .as_object()
            .ok_or_else(|| Error::Format(format!("{what} is not an object")))?;
        match (o.get("object"), o.get("parts")) {
            (Some(id), None) => {
                self.tensor(id.as_str().unwrap_or(""), what)?;
            }
            (None, Some(Value::Array(parts))) if !parts.is_empty() => {
                for (j, p) in parts.iter().enumerate() {
                    let pw = format!("{what}.parts[{j}]");
                    let id = p.get("object").and_then(Value::as_str).unwrap_or("");
                    let elements = self.tensor(id, &pw)?.elements().unwrap();
                    let r = p
                        .get("range")
                        .and_then(Value::as_array)
                        .filter(|r| r.len() == 2);
                    let (b, e) = match r.map(|r| (r[0].as_u64(), r[1].as_u64())) {
                        Some((Some(b), Some(e))) => (b, e),
                        _ => return bad(format!("{pw}.range must be [begin, end)")),
                    };
                    if !(b < e && e <= elements) {
                        return bad(format!("{pw}.range [{b}, {e}) outside 0..{elements}"));
                    }
                }
            }
            _ => {
                return bad(format!(
                    "{what} must have exactly one of object / non-empty parts"
                ))
            }
        }
        Ok(())
    }

    fn check_references(&self) -> Result<()> {
        let d = &self.directory;
        let bindings = d
            .get("bindings")
            .and_then(Value::as_object)
            .ok_or_else(|| Error::Format("bindings must be an object".into()))?;
        for (name, b) in bindings {
            self.check_binding(b, &format!("binding '{name}'"))?;
        }
        let uses = d
            .get("uses")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::Format("uses must be an array".into()))?;
        let mut seen = std::collections::HashSet::new();
        for (i, u) in uses.iter().enumerate() {
            let what = format!("uses[{i}]");
            let o = u
                .as_object()
                .ok_or_else(|| Error::Format(format!("{what} is not an object")))?;
            let param = str_field(o, "parameter", &what)?;
            let input = str_field(o, "input", &what)?;
            if !bindings.contains_key(param) {
                return bad(format!("{what} names unknown parameter '{param}'"));
            }
            if !seen.insert((param, input)) {
                return bad(format!("{what} repeats ({param}, {input})"));
            }
            if let Some(p) = o.get("activation_policy") {
                if !matches!(p.as_str(), Some("A16Only" | "AllowA8" | "AllowA4")) {
                    return bad(format!(
                        "{what}.activation_policy {p} is not A16Only/AllowA8/AllowA4"
                    ));
                }
            }
            if let Some(aux) = o.get("auxiliaries").and_then(Value::as_object) {
                for (role, b) in aux {
                    self.check_binding(b, &format!("{what}.auxiliaries.{role}"))?;
                }
            }
        }
        let comps = d
            .get("components")
            .and_then(Value::as_object)
            .ok_or_else(|| Error::Format("components must be an object".into()))?;
        if !comps.contains_key("text") {
            return bad("components must include 'text'");
        }
        for (cid, c) in comps {
            if !c.get("config").is_some_and(Value::is_object) {
                return bad(format!("component '{cid}' needs a config object"));
            }
            if let Some(t) = c.get("target").and_then(Value::as_str) {
                if !comps.contains_key(t) {
                    return bad(format!("component '{cid}' targets missing component '{t}'"));
                }
            }
            for (role, r) in c
                .get("resources")
                .and_then(Value::as_object)
                .into_iter()
                .flatten()
            {
                let rid = r.as_str().unwrap_or("");
                if !matches!(
                    self.object(rid).map(|o| &o.kind),
                    Some(ObjectKind::Resource { .. })
                ) {
                    return bad(format!(
                        "component '{cid}' resource '{role}' is not a resource object"
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn object(&self, id: &str) -> Option<&Object> {
        self.index.get(id).map(|&i| &self.objects[i])
    }

    pub fn payload_bytes(&self) -> u64 {
        self.files.iter().map(|f| f.payload_bytes).sum()
    }

    pub fn components(&self) -> &Map<String, Value> {
        self.directory["components"].as_object().unwrap()
    }

    pub fn bindings_len(&self) -> usize {
        self.directory["bindings"].as_object().map_or(0, Map::len)
    }

    pub fn uses_len(&self) -> usize {
        self.directory["uses"].as_array().map_or(0, Vec::len)
    }

    pub fn name(&self) -> Option<&str> {
        self.directory.get("metadata")?.get("name")?.as_str()
    }

    /// Reads logical payload bytes `[offset, offset + buf.len())`, crossing files as needed.
    pub fn read_range(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        let end = offset
            .checked_add(buf.len() as u64)
            .filter(|&e| e <= self.payload_bytes())
            .ok_or_else(|| {
                Error::Format(format!("read [{offset}, +{}) outside payload", buf.len()))
            })?;
        for f in &self.files {
            let (a, b) = (
                offset.max(f.logical_start),
                end.min(f.logical_start + f.payload_bytes),
            );
            if a >= b {
                continue;
            }
            let mut file = File::open(&f.path).map_err(|e| Error::Io(f.path.clone(), e))?;
            file.seek(SeekFrom::Start(f.payload_start + (a - f.logical_start)))
                .map_err(|e| Error::Io(f.path.clone(), e))?;
            file.read_exact(&mut buf[(a - offset) as usize..(b - offset) as usize])
                .map_err(|e| Error::Io(f.path.clone(), e))?;
        }
        Ok(())
    }

    pub fn read_object(&self, id: &str) -> Result<Vec<u8>> {
        let o = self
            .object(id)
            .ok_or_else(|| Error::Format(format!("no object '{id}'")))?;
        let mut buf = vec![0u8; o.bytes as usize];
        self.read_range(o.offset, &mut buf)?;
        Ok(buf)
    }

    /// Bytes per (format, layout) over tensor objects, largest first.
    pub fn format_summary(&self) -> Vec<(String, usize, u64)> {
        let mut m: HashMap<String, (usize, u64)> = HashMap::new();
        for o in &self.objects {
            let key = match &o.kind {
                ObjectKind::Tensor { format, layout, .. } => format!("{format} / {layout}"),
                ObjectKind::Resource { encoding } => format!("resource / {encoding}"),
            };
            let e = m.entry(key).or_default();
            e.0 += 1;
            e.1 += o.bytes;
        }
        let mut v: Vec<_> = m.into_iter().map(|(k, (n, b))| (k, n, b)).collect();
        v.sort_by_key(|e| std::cmp::Reverse(e.2));
        v
    }
}

/// Writes a container whose payload is copied from `src` (any container kind) as `target`.
/// Objects, bindings, uses and components are preserved byte-for-byte; only framing, the
/// files table and (for `.vi`) the `velox` provenance record change. Files are written under
/// temporary names and renamed into place last, entry after parts.
pub fn convert(
    src: &Artifact,
    out: &Path,
    target: Container,
    part_limit: u64,
    artifact_id: [u8; 16],
) -> Result<()> {
    if out.exists() {
        return bad(format!("{} already exists", out.display()));
    }
    let mut dir = src.directory.clone();
    match target {
        Container::VeloxV1 => {
            let mut v = Map::new();
            v.insert("format_version".into(), 1.into());
            v.insert("source_container".into(), src.container.to_string().into());
            v.insert("source_artifact_id".into(), hex(&src.artifact_id).into());
            dir.insert("velox".into(), Value::Object(v));
        }
        Container::NinferV3 => {
            dir.remove("velox");
        }
    }
    let payload = src.payload_bytes();
    let name = out
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| Error::Format("output needs a UTF-8 file name".into()))?;

    // Reserve JSON space, lay out files, and grow the reservation until the directory fits.
    let mut reserve = PAGE - HEADER;
    let (json, sizes) = loop {
        let entry_start = align_up(HEADER + reserve, PAGE);
        if part_limit <= entry_start + PAGE {
            return bad("part limit too small");
        }
        let entry_cap = (part_limit - entry_start) / PAGE * PAGE;
        let part_cap = (part_limit - PAGE) / PAGE * PAGE;
        let mut sizes = vec![payload.min(entry_cap)];
        let mut left = payload - sizes[0];
        while left > 0 {
            sizes.push(left.min(part_cap));
            left -= sizes.last().unwrap();
        }
        let files: Vec<Value> = sizes
            .iter()
            .enumerate()
            .map(|(i, &b)| {
                let path = if i == 0 {
                    Value::Null
                } else {
                    Value::from(format!("{name}.part-{i:04}"))
                };
                serde_json::json!({"path": path, "payload_bytes": b})
            })
            .collect();
        dir.insert("files".into(), Value::Array(files));
        let json = serde_json::to_vec(&Value::Object(dir.clone()))
            .expect("JSON serialisation cannot fail");
        if json.len() as u64 <= reserve {
            break (json, sizes);
        }
        reserve = align_up(json.len() as u64 + HEADER, PAGE) - HEADER;
    };

    let (entry_magic, part_magic) = target.magics();
    let base = out.parent().unwrap_or(Path::new("."));
    let mut written = Vec::new();
    let result = (|| -> Result<()> {
        let mut logical = 0u64;
        for (i, &bytes) in sizes.iter().enumerate() {
            let final_path = if i == 0 {
                out.to_path_buf()
            } else {
                base.join(format!("{name}.part-{i:04}"))
            };
            let mut tmp_name = final_path.clone().into_os_string();
            tmp_name.push(".tmp-velox");
            let tmp = PathBuf::from(tmp_name);
            let io = |e| Error::Io(tmp.clone(), e);
            let mut f = io::BufWriter::with_capacity(8 << 20, File::create(&tmp).map_err(io)?);
            written.push((tmp.clone(), final_path));
            let mut header = Vec::with_capacity(PAGE as usize);
            if i == 0 {
                header.extend_from_slice(&entry_magic);
                header.extend_from_slice(&reserve.to_le_bytes());
                header.extend_from_slice(&artifact_id);
                header.extend_from_slice(&json);
                header.resize((HEADER + reserve) as usize, b' ');
                header.resize(align_up(HEADER + reserve, PAGE) as usize, 0);
            } else {
                header.extend_from_slice(&part_magic);
                header.extend_from_slice(&(i as u64).to_le_bytes());
                header.extend_from_slice(&artifact_id);
                header.resize(PAGE as usize, 0);
            }
            f.write_all(&header).map_err(io)?;
            let mut chunk = vec![0u8; 64 << 20];
            let mut done = 0;
            while done < bytes {
                let n = (bytes - done).min(chunk.len() as u64) as usize;
                src.read_range(logical + done, &mut chunk[..n])?;
                f.write_all(&chunk[..n]).map_err(io)?;
                done += n as u64;
            }
            f.flush().map_err(io)?;
            logical += bytes;
        }
        // Parts first, entry last: an entry on disk always has its parts.
        for (tmp, fin) in written.iter().rev() {
            std::fs::rename(tmp, fin).map_err(|e| Error::Io(fin.clone(), e))?;
        }
        Ok(())
    })();
    if result.is_err() {
        for (tmp, _) in &written {
            let _ = std::fs::remove_file(tmp);
        }
    }
    result
}

/// A new artifact id from time and process entropy (not cryptographic; it only pairs files).
pub fn new_artifact_id() -> [u8; 16] {
    use std::hash::{BuildHasher, Hasher};
    let mut id = [0u8; 16];
    for half in id.chunks_mut(8) {
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u128(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos(),
        );
        half.copy_from_slice(&h.finish().to_le_bytes());
    }
    id
}

#[cfg(test)]
mod tests;
