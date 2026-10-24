//! GGUF (v2 / v3) reader: metadata and the tensor table, plus raw reads of tensor data. Interpreting
//! tensor types and names belongs to the model importers.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

use crate::{bad, Error, Result};

/// One entry of the tensor table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GgufTensor {
    pub name: String,
    /// ggml order: `dims[0]` is the contiguous (innermost) dimension.
    pub dims: Vec<u64>,
    /// `ggml_type` id.
    pub ggml_type: u32,
    /// Absolute file offset of the tensor's data.
    pub offset: u64,
}

pub struct Gguf {
    pub path: PathBuf,
    pub version: u32,
    /// Metadata; arrays longer than [`ARRAY_KEEP`] are summarised as `{"array_len": n}`.
    pub metadata: Map<String, Value>,
    pub tensors: Vec<GgufTensor>,
    file: File,
}

/// Arrays up to this length are kept in [`Gguf::metadata`] (vocabularies are not).
pub const ARRAY_KEEP: u64 = 1024;

struct Cursor<R: Read> {
    r: R,
    path: PathBuf,
    pos: u64,
}

impl<R: Read> Cursor<R> {
    fn bytes(&mut self, n: usize) -> Result<Vec<u8>> {
        let mut b = vec![0u8; n];
        self.r
            .read_exact(&mut b)
            .map_err(|e| Error::Io(self.path.clone(), e))?;
        self.pos += n as u64;
        Ok(b)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()?;
        if n > 1 << 30 {
            return bad(format!("gguf string of {n} bytes"));
        }
        String::from_utf8(self.bytes(n as usize)?).or_else(|_| bad("gguf string is not UTF-8"))
    }
    fn scalar(&mut self, t: u32) -> Result<Value> {
        Ok(match t {
            0 => Value::from(self.bytes(1)?[0]),
            1 => Value::from(self.bytes(1)?[0] as i8),
            2 => Value::from(u16::from_le_bytes(self.bytes(2)?.try_into().unwrap())),
            3 => Value::from(i16::from_le_bytes(self.bytes(2)?.try_into().unwrap())),
            4 => Value::from(self.u32()?),
            5 => Value::from(self.u32()? as i32),
            6 => Value::from(f32::from_le_bytes(self.bytes(4)?.try_into().unwrap())),
            7 => Value::from(self.bytes(1)?[0] != 0),
            8 => Value::from(self.string()?),
            10 => Value::from(self.u64()?),
            11 => Value::from(self.u64()? as i64),
            12 => Value::from(f64::from_le_bytes(self.bytes(8)?.try_into().unwrap())),
            other => return bad(format!("gguf value type {other}")),
        })
    }
}

impl Gguf {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path).map_err(|e| Error::Io(path.clone(), e))?;
        let mut c = Cursor {
            r: BufReader::with_capacity(
                1 << 20,
                file.try_clone().map_err(|e| Error::Io(path.clone(), e))?,
            ),
            path: path.clone(),
            pos: 0,
        };
        if c.bytes(4)? != b"GGUF" {
            return bad(format!("{} is not a GGUF file", path.display()));
        }
        let version = c.u32()?;
        if !(2..=3).contains(&version) {
            return bad(format!("GGUF version {version} is not supported"));
        }
        let (n_tensors, n_kv) = (c.u64()?, c.u64()?);
        let mut metadata = Map::new();
        for _ in 0..n_kv {
            let key = c.string()?;
            let t = c.u32()?;
            let v = if t == 9 {
                let (et, n) = (c.u32()?, c.u64()?);
                let mut items = Vec::new();
                for _ in 0..n {
                    let v = c.scalar(et)?;
                    if n <= ARRAY_KEEP {
                        items.push(v);
                    }
                }
                if n <= ARRAY_KEEP {
                    Value::Array(items)
                } else {
                    serde_json::json!({ "array_len": n })
                }
            } else {
                c.scalar(t)?
            };
            metadata.insert(key, v);
        }
        let mut tensors = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = c.string()?;
            let nd = c.u32()?;
            if nd > 4 {
                return bad(format!("gguf tensor {name} has {nd} dimensions"));
            }
            let dims = (0..nd).map(|_| c.u64()).collect::<Result<Vec<_>>>()?;
            let (ggml_type, offset) = (c.u32()?, c.u64()?);
            tensors.push(GgufTensor {
                name,
                dims,
                ggml_type,
                offset,
            });
        }
        let align = metadata
            .get("general.alignment")
            .and_then(Value::as_u64)
            .unwrap_or(32);
        let data = c.pos.div_ceil(align) * align;
        for t in &mut tensors {
            t.offset += data;
        }
        Ok(Gguf {
            path,
            version,
            metadata,
            tensors,
            file,
        })
    }

    pub fn tensor(&self, name: &str) -> Option<&GgufTensor> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Reads `buf.len()` bytes at absolute file offset `at`. Positional: concurrent reads from
    /// several threads do not disturb each other.
    pub fn read_at(&self, at: u64, buf: &mut [u8]) -> Result<()> {
        read_exact_at(&self.file, at, buf).map_err(|e| Error::Io(self.path.clone(), e))
    }
}

#[cfg(unix)]
fn read_exact_at(f: &File, at: u64, buf: &mut [u8]) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(f, buf, at)
}

#[cfg(windows)]
fn read_exact_at(f: &File, mut at: u64, mut buf: &mut [u8]) -> std::io::Result<()> {
    while !buf.is_empty() {
        match std::os::windows::fs::FileExt::seek_read(f, buf, at)? {
            0 => return Err(std::io::ErrorKind::UnexpectedEof.into()),
            n => {
                buf = &mut buf[n..];
                at += n as u64;
            }
        }
    }
    Ok(())
}

impl Gguf {
    pub fn meta_u64(&self, key: &str) -> Option<u64> {
        self.metadata.get(key).and_then(Value::as_u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn put_str(b: &mut Vec<u8>, s: &str) {
        b.extend_from_slice(&(s.len() as u64).to_le_bytes());
        b.extend_from_slice(s.as_bytes());
    }

    #[test]
    fn reads_metadata_and_tensor_table() {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes()); // tensors
        b.extend_from_slice(&2u64.to_le_bytes()); // kv
        put_str(&mut b, "qwen35.block_count");
        b.extend_from_slice(&4u32.to_le_bytes());
        b.extend_from_slice(&64u32.to_le_bytes());
        put_str(&mut b, "list");
        b.extend_from_slice(&9u32.to_le_bytes());
        b.extend_from_slice(&5u32.to_le_bytes());
        b.extend_from_slice(&2u64.to_le_bytes());
        b.extend_from_slice(&(-3i32).to_le_bytes());
        b.extend_from_slice(&7i32.to_le_bytes());
        put_str(&mut b, "w");
        b.extend_from_slice(&2u32.to_le_bytes());
        b.extend_from_slice(&4u64.to_le_bytes());
        b.extend_from_slice(&2u64.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes()); // F32
        b.extend_from_slice(&0u64.to_le_bytes());
        let data = b.len().div_ceil(32) * 32;
        b.resize(data, 0);
        for i in 0..8u32 {
            b.extend_from_slice(&(i as f32).to_le_bytes());
        }
        let dir = std::env::temp_dir().join(format!("velox-gguf-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.gguf");
        File::create(&path).unwrap().write_all(&b).unwrap();
        let g = Gguf::open(&path).unwrap();
        assert_eq!(g.meta_u64("qwen35.block_count"), Some(64));
        assert_eq!(g.metadata["list"], serde_json::json!([-3, 7]));
        let t = g.tensor("w").unwrap();
        assert_eq!(
            (t.dims.as_slice(), t.ggml_type, t.offset),
            (&[4u64, 2][..], 0, data as u64)
        );
        let mut v = [0u8; 4];
        g.read_at(t.offset + 12, &mut v).unwrap();
        assert_eq!(f32::from_le_bytes(v), 3.0);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
