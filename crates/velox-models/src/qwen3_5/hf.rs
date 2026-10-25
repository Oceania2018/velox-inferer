//! Imports a Hugging Face checkpoint of Qwen3.5 / 3.6 (`Qwen3_5ForConditionalGeneration` /
//! `Qwen3_5MoeForConditionalGeneration`, text model `qwen3_5_text` / `qwen3_5_moe_text`) stored as
//! safetensors, plain BF16 or compressed-tensors quantized, as a `.vi`.
//!
//! Every value keeps its stored encoding; the import only moves whole rows, which is exact:
//!
//! - BF16 tensors copy as `bf16`, F32 ones as `fp32`.
//! - compressed-tensors FP8 (`float-quantized`, channel strategy: E4M3 `weight` `[n, k]` plus a BF16
//!   `weight_scale` `[n, 1]`) is `fp8_e4m3fn_row_bf16` byte for byte (codes, padding, row scales).
//! - compressed-tensors NVFP4 (`nvfp4-pack-quantized`: `weight_packed` `[n, k/2]` in velox's nibble
//!   order, E4M3 `weight_scale` `[n, k/16]`, F32 `weight_global_scale`) is `nvfp4`: the codes copy,
//!   the scales are placed in the `block_scale_k16_m128x4_v1` swizzle and the global scale is the
//!   divisor. `input_global_scale` becomes the `AllowA4` activation divisor.
//!
//! HF already stores what velox wants for the parts the GGUF import has to undo: zero-centred norm
//! weights (velox applies `1 + w`; the GDN norm is plain), `A_log`, and GDN value heads in grouped
//! order (`g * (v_heads / k_heads) + j`). What does change:
//!
//! - `conv1d.weight` `[channels, 1, width]` is transposed to velox's `[width, channels]`.
//! - Attention `q_proj` interleaves each head's query and gate rows: split into query rows, then
//!   gate rows (row scales move with their rows).
//! - Projections that read the same input are stacked into one object where the encoding allows
//!   (row-scaled FP8 and BF16 always; NVFP4 only with equal global scales), so the loader runs them
//!   as one GEMM: attention q | gate | k | v, GDN qkv | z and a | b, router | shared-expert gate,
//!   shared-expert gate | up.
//! - Routed experts are one object per expert and projection (NVFP4 global scales differ per
//!   expert); the MTP head's fused BF16 `experts.gate_up_proj` `[E, 2I, H]` is split the same way
//!   (gate rows `[0, I)`, up rows `[I, 2I)` per expert, as transformers' `chunk(2)` reads it).
//!
//! The vision tower (`model.visual.*`) is not imported yet.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};
use velox_core::format::{align256, bf16_to_f32, encode_nvfp4, nvfp4_scale_offset};
use velox_core::{Error, Result};
use velox_weights::{Artifact, Container};

use super::config::{FfnConfig, LayerKind, TextConfig, DENSE_ARCHITECTURE, MOE_ARCHITECTURE};

fn werr(e: velox_weights::Error) -> Error {
    Error::Config(e.to_string())
}

fn ioerr(p: &Path, e: std::io::Error) -> Error {
    Error::Config(format!("{}: {e}", p.display()))
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

/// Bytes per element of a safetensors dtype.
fn dtype_bytes(d: &str) -> Option<u64> {
    match d {
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" => Some(1),
        "BF16" | "F16" | "I16" | "U16" => Some(2),
        "F32" | "I32" | "U32" => Some(4),
        "F64" | "I64" | "U64" => Some(8),
        _ => None,
    }
}

/// One tensor of a safetensors file.
#[derive(Clone, Debug)]
pub struct StTensor {
    file: usize,
    pub dtype: String,
    pub shape: Vec<u64>,
    /// Absolute file offset of the data.
    at: u64,
    pub bytes: u64,
}

impl StTensor {
    /// (rows, columns) of the tensor read as a matrix: the last dimension is `k`, the others flatten.
    fn matrix(&self) -> (u64, u64) {
        match self.shape.split_last() {
            Some((&k, rest)) => (rest.iter().product(), k),
            None => (1, 1),
        }
    }
}

/// The tensors of one or more safetensors files (8-byte LE header length, JSON header
/// `{name: {dtype, shape, data_offsets}}`, data), read positionally.
pub struct SafeTensors {
    paths: Vec<PathBuf>,
    files: Vec<File>,
    tensors: HashMap<String, StTensor>,
}

impl SafeTensors {
    /// The checkpoint in `dir`: the shards listed by `model.safetensors.index.json`, else
    /// `model.safetensors`.
    pub fn open_dir(dir: &Path) -> Result<Self> {
        let index = dir.join("model.safetensors.index.json");
        let names: Vec<String> = if index.exists() {
            let v = read_json(&index)?;
            let mut files: Vec<String> = v["weight_map"]
                .as_object()
                .ok_or_else(|| Error::Config(format!("{}: no weight_map", index.display())))?
                .values()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect();
            files.sort();
            files.dedup();
            files
        } else {
            vec!["model.safetensors".into()]
        };
        Self::open(names.iter().map(|n| dir.join(n)).collect())
    }

    pub fn open(paths: Vec<PathBuf>) -> Result<Self> {
        let mut files = Vec::with_capacity(paths.len());
        let mut tensors = HashMap::new();
        for (i, p) in paths.iter().enumerate() {
            let f = File::open(p).map_err(|e| ioerr(p, e))?;
            let len = f.metadata().map_err(|e| ioerr(p, e))?.len();
            let mut h = [0u8; 8];
            read_exact_at(&f, 0, &mut h).map_err(|e| ioerr(p, e))?;
            let n = u64::from_le_bytes(h);
            if n > 256 << 20 || 8 + n > len {
                return Err(Error::Config(format!(
                    "{}: header of {n} bytes in a {len}-byte file",
                    p.display()
                )));
            }
            let mut j = vec![0u8; n as usize];
            read_exact_at(&f, 8, &mut j).map_err(|e| ioerr(p, e))?;
            let v: Value = serde_json::from_slice(&j)
                .map_err(|e| Error::Config(format!("{}: header: {e}", p.display())))?;
            let base = 8 + n;
            for (name, t) in v.as_object().into_iter().flatten() {
                if name == "__metadata__" {
                    continue;
                }
                let bad = |m: &str| Error::Config(format!("{}: {name}: {m}", p.display()));
                let dtype = t["dtype"].as_str().ok_or_else(|| bad("no dtype"))?;
                let shape = t["shape"]
                    .as_array()
                    .ok_or_else(|| bad("no shape"))?
                    .iter()
                    .map(|d| d.as_u64().ok_or_else(|| bad("bad shape")))
                    .collect::<Result<Vec<u64>>>()?;
                let (a, b) = match t["data_offsets"].as_array().map(Vec::as_slice) {
                    Some([a, b]) => (
                        a.as_u64().ok_or_else(|| bad("bad data_offsets"))?,
                        b.as_u64().ok_or_else(|| bad("bad data_offsets"))?,
                    ),
                    _ => return Err(bad("no data_offsets")),
                };
                if a > b || base + b > len {
                    return Err(bad("data outside the file"));
                }
                if let Some(eb) = dtype_bytes(dtype) {
                    if shape.iter().product::<u64>() * eb != b - a {
                        return Err(bad("data size disagrees with shape and dtype"));
                    }
                }
                let st = StTensor {
                    file: i,
                    dtype: dtype.to_string(),
                    shape,
                    at: base + a,
                    bytes: b - a,
                };
                if tensors.insert(name.clone(), st).is_some() {
                    return Err(bad("tensor appears in more than one file"));
                }
            }
            files.push(f);
        }
        Ok(SafeTensors {
            paths,
            files,
            tensors,
        })
    }

    pub fn tensor(&self, name: &str) -> Option<&StTensor> {
        self.tensors.get(name)
    }

    /// Adds `other`'s tensors under `prefix` + name (a draft model beside the target).
    pub fn merge(&mut self, other: SafeTensors, prefix: &str) {
        let base = self.files.len();
        self.paths.extend(other.paths);
        self.files.extend(other.files);
        for (name, mut t) in other.tensors {
            t.file += base;
            self.tensors.insert(format!("{prefix}{name}"), t);
        }
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }

    /// Reads `buf.len()` bytes at absolute offset `at` of file `file`.
    fn read_at(&self, file: usize, at: u64, buf: &mut [u8]) -> Result<()> {
        read_exact_at(&self.files[file], at, buf).map_err(|e| ioerr(&self.paths[file], e))
    }

    fn read_all(&self, t: &StTensor) -> Result<Vec<u8>> {
        let mut v = vec![0u8; t.bytes as usize];
        self.read_at(t.file, t.at, &mut v)?;
        Ok(v)
    }
}

fn read_json(p: &Path) -> Result<Value> {
    let bytes = std::fs::read(p).map_err(|e| ioerr(p, e))?;
    serde_json::from_slice(&bytes).map_err(|e| Error::Config(format!("{}: {e}", p.display())))
}

/// A run of an output object's bytes.
enum Piece {
    /// `len` bytes of a checkpoint file from absolute offset `at`.
    File {
        file: usize,
        at: u64,
        len: u64,
    },
    /// Rows of `row_bytes` from a tensor at `at`: output row `i` is input row `perm[i]`.
    Rows {
        file: usize,
        at: u64,
        row_bytes: u64,
        perm: Vec<u32>,
    },
    Zero(u64),
    /// Bytes computed here (small).
    Data(Vec<u8>),
    /// The NVFP4 scale plane of the row-stacked `[rows, k / 16]` E4M3 tensors `(file, at, rows)`,
    /// swizzled by [`nvfp4_scale_offset`].
    Nvfp4Scales {
        srcs: Vec<(usize, u64, u64)>,
        k: u64,
    },
}

impl Piece {
    fn len(&self) -> u64 {
        match self {
            Piece::File { len, .. } => *len,
            Piece::Rows {
                row_bytes, perm, ..
            } => row_bytes * perm.len() as u64,
            Piece::Zero(n) => *n,
            Piece::Data(v) => v.len() as u64,
            Piece::Nvfp4Scales { srcs, k } => srcs.iter().map(|s| s.2).sum::<u64>() * k / 16,
        }
    }
}

struct Out {
    /// The directory entry without `offset` / `bytes`.
    entry: Map<String, Value>,
    bytes: u64,
    pieces: Vec<Piece>,
    offset: u64,
}

/// How a binding's input activations may be rounded (`uses[].activation_policy`).
enum Policy {
    A16,
    A8,
    /// `AllowA4` with the activation divisor in this fp32 object.
    A4(String),
}

/// (binding, first row, rows) of a matrix object.
type Part<'s> = (&'s str, u64, u64);

/// Rows of one checkpoint tensor (read as a matrix) for a stacked object.
struct RowSrc {
    weight: String,
    sel: Sel,
}

enum Sel {
    All,
    /// Rows `[begin, end)`.
    Range(u64, u64),
    /// These rows, in this order.
    Perm(Vec<u32>),
}

impl RowSrc {
    fn all(weight: String) -> Self {
        RowSrc {
            weight,
            sel: Sel::All,
        }
    }
}

/// Summary of an import.
pub struct ImportReport {
    /// (format, objects, bytes) of every tensor object.
    pub formats: Vec<(String, usize, u64)>,
    pub objects: usize,
    pub bindings: usize,
    pub uses: usize,
    pub payload_bytes: u64,
    pub mtp: bool,
}

struct Builder<'a> {
    st: &'a SafeTensors,
    outs: Vec<Out>,
    bindings: Map<String, Value>,
    uses: Vec<Value>,
    /// Activation-divisor objects by value bits (many experts share one).
    divisors: HashMap<u32, String>,
}

fn tensor_entry(shape: Value, format: &str, layout: &str) -> Map<String, Value> {
    let mut e = Map::new();
    e.insert("kind".into(), "tensor".into());
    e.insert("shape".into(), shape);
    e.insert("format".into(), format.into());
    e.insert("layout".into(), layout.into());
    e
}

impl<'a> Builder<'a> {
    fn new(st: &'a SafeTensors) -> Self {
        Builder {
            st,
            outs: Vec::new(),
            bindings: Map::new(),
            uses: Vec::new(),
            divisors: HashMap::new(),
        }
    }

    /// Rows `[r0, r0 + rows)` of BF16 tensor `weight` (read as a matrix) encoded as one `nvfp4`
    /// object `id`, bound whole to `binding` (A16: no activation divisor is known for them).
    fn nvfp4_from_bf16(
        &mut self,
        id: &str,
        weight: &str,
        r0: u64,
        rows: u64,
        binding: &str,
    ) -> Result<()> {
        self.nvfp4_stack(id, &[(weight, r0, rows)], &[(binding, 0, rows)])
    }

    /// Row ranges `(weight, r0, rows)` of BF16 tensors stacked and encoded as one `nvfp4` object
    /// `id` (one divisor for the stack), bound as `parts` (A16).
    fn nvfp4_stack(&mut self, id: &str, srcs: &[(&str, u64, u64)], parts: &[Part]) -> Result<()> {
        let mut x: Vec<f32> = Vec::new();
        let mut k0 = None;
        for &(weight, r0, rows) in srcs {
            let t = self.get(weight)?;
            let (_, k) = t.matrix();
            if t.dtype != "BF16" || *k0.get_or_insert(k) != k {
                return Err(Error::Unsupported(format!(
                    "{weight}: {} [.., {k}] does not stack as BF16 with {id}",
                    t.dtype
                )));
            }
            let mut raw = vec![0u8; (rows * k * 2) as usize];
            self.st.read_at(t.file, t.at + r0 * k * 2, &mut raw)?;
            x.extend(
                raw.chunks_exact(2)
                    .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))),
            );
        }
        let k = k0.unwrap_or(0);
        let n = x.len() as u64 / k.max(1);
        let payload = encode_nvfp4(&x, n as usize, k as usize);
        let id = self.push(
            format!("hf/{id}"),
            tensor_entry(json!([n, k]), "nvfp4", "block_scale_k16_m128x4_v1"),
            vec![Piece::Data(payload)],
        );
        for &part in parts {
            self.bind(&id, (n, k), part, &Policy::A16);
        }
        Ok(())
    }

    fn get(&self, name: &str) -> Result<&'a StTensor> {
        self.st
            .tensor(name)
            .ok_or_else(|| Error::Config(format!("checkpoint has no tensor {name}")))
    }

    fn has(&self, name: &str) -> bool {
        self.st.tensor(name).is_some()
    }

    fn push(&mut self, id: String, entry: Map<String, Value>, pieces: Vec<Piece>) -> String {
        let mut e = entry;
        e.insert("id".into(), id.clone().into());
        self.outs.push(Out {
            entry: e,
            bytes: pieces.iter().map(Piece::len).sum(),
            pieces,
            offset: 0,
        });
        id
    }

    /// Binds rows `[r0, r0 + rows)` of matrix object `id` (`[n, k]`) as `binding`, with an
    /// activation policy.
    fn bind(&mut self, id: &str, (n, k): (u64, u64), (binding, r0, rows): Part, p: &Policy) {
        let b = if r0 == 0 && rows == n {
            json!({ "object": id })
        } else {
            json!({ "parts": [{ "object": id, "range": [r0 * k, (r0 + rows) * k] }] })
        };
        self.bindings.insert(binding.into(), b);
        let mut u = json!({ "parameter": binding, "input": "activations" });
        let (policy, aux) = match p {
            Policy::A16 => ("A16Only", None),
            Policy::A8 => ("AllowA8", None),
            Policy::A4(d) => ("AllowA4", Some(d)),
        };
        u["activation_policy"] = policy.into();
        if let Some(d) = aux {
            u["auxiliaries"] = json!({ "activation_input_divisor": { "object": d } });
        }
        self.uses.push(u);
    }

    /// A small tensor (norm, `A_log`, ...) copied as a flat `bf16` / `fp32` vector of `len` values.
    fn vector(&mut self, binding: &str, name: &str, len: usize) -> Result<()> {
        let t = self.get(name)?;
        let format = match t.dtype.as_str() {
            "BF16" => "bf16",
            "F32" => "fp32",
            other => {
                return Err(Error::Unsupported(format!(
                    "{name}: small tensor of dtype {other}"
                )))
            }
        };
        let n: u64 = t.shape.iter().product();
        if n != len as u64 {
            return Err(Error::Config(format!(
                "{name}: {n} values, config implies {len}"
            )));
        }
        let pieces = vec![Piece::File {
            file: t.file,
            at: t.at,
            len: t.bytes,
        }];
        let id = self.push(
            format!("hf/{name}"),
            tensor_entry(json!([len]), format, "contiguous_le_v1"),
            pieces,
        );
        self.bindings
            .insert(binding.into(), json!({ "object": id }));
        Ok(())
    }

    /// A GDN `conv1d.weight` `[channels, 1, width]` as velox's `[width, channels]`.
    fn conv(&mut self, binding: &str, name: &str, channels: usize, width: usize) -> Result<()> {
        let t = self.get(name)?;
        let (format, eb) = match t.dtype.as_str() {
            "BF16" => ("bf16", 2),
            "F32" => ("fp32", 4),
            other => return Err(Error::Unsupported(format!("{name}: dtype {other}"))),
        };
        if t.shape.iter().product::<u64>() != (channels * width) as u64
            || t.shape.first() != Some(&(channels as u64))
        {
            return Err(Error::Config(format!(
                "{name}: shape {:?}, config implies [{channels}, 1, {width}]",
                t.shape
            )));
        }
        let src = self.st.read_all(t)?;
        let mut out = vec![0u8; src.len()];
        for c in 0..channels {
            for j in 0..width {
                let (s, d) = ((c * width + j) * eb, (j * channels + c) * eb);
                out[d..d + eb].copy_from_slice(&src[s..s + eb]);
            }
        }
        let id = self.push(
            format!("hf/{name}"),
            tensor_entry(json!([width, channels]), format, "contiguous_le_v1"),
            vec![Piece::Data(out)],
        );
        self.bindings
            .insert(binding.into(), json!({ "object": id }));
        Ok(())
    }

    /// Rows of BF16 or row-scaled FP8 tensors stacked into one object `id`, bound as `parts`.
    fn stacked(&mut self, id: &str, srcs: &[RowSrc], parts: &[Part]) -> Result<()> {
        let mut fp8: Option<bool> = None;
        let mut k0: Option<u64> = None;
        let mut code_pieces = Vec::new();
        let mut scales: Vec<u8> = Vec::new();
        let mut n = 0u64;
        for s in srcs {
            let t = self.get(&s.weight)?;
            let is_fp8 = match t.dtype.as_str() {
                "BF16" => false,
                "F8_E4M3" => true,
                other => {
                    return Err(Error::Unsupported(format!(
                        "{}: matrix of dtype {other}",
                        s.weight
                    )))
                }
            };
            let (rows_total, k) = t.matrix();
            if *fp8.get_or_insert(is_fp8) != is_fp8 || *k0.get_or_insert(k) != k {
                return Err(Error::Config(format!(
                    "{id}: {} differs from the other stacked tensors in dtype or columns",
                    s.weight
                )));
            }
            let row_bytes = if is_fp8 { k } else { 2 * k };
            let rows: Vec<u64> = match &s.sel {
                Sel::All => (0..rows_total).collect(),
                Sel::Range(a, b) => (*a..*b).collect(),
                Sel::Perm(p) => p.iter().map(|&r| u64::from(r)).collect(),
            };
            if rows.iter().any(|&r| r >= rows_total) {
                return Err(Error::Config(format!(
                    "{id}: row selection past {} [{rows_total}]",
                    s.weight
                )));
            }
            code_pieces.push(match &s.sel {
                Sel::All => Piece::File {
                    file: t.file,
                    at: t.at,
                    len: t.bytes,
                },
                Sel::Range(a, b) => Piece::File {
                    file: t.file,
                    at: t.at + a * row_bytes,
                    len: (b - a) * row_bytes,
                },
                Sel::Perm(p) => Piece::Rows {
                    file: t.file,
                    at: t.at,
                    row_bytes,
                    perm: p.clone(),
                },
            });
            if is_fp8 {
                let sname = format!("{}_scale", s.weight);
                let st = self.get(&sname)?;
                if st.dtype != "BF16" || st.shape.iter().product::<u64>() != rows_total {
                    return Err(Error::Unsupported(format!(
                        "{sname}: {} {:?}, expected BF16 [{rows_total}, 1] (per-channel FP8)",
                        st.dtype, st.shape
                    )));
                }
                let all = self.st.read_all(st)?;
                for &r in &rows {
                    scales.extend_from_slice(&all[2 * r as usize..2 * r as usize + 2]);
                }
            }
            n += rows.len() as u64;
        }
        let (fp8, k) = (fp8.unwrap_or(false), k0.unwrap_or(0));
        let mut pieces = code_pieces;
        let entry = if fp8 {
            pieces.push(Piece::Zero(align256((n * k) as usize) as u64 - n * k));
            pieces.push(Piece::Data(scales));
            tensor_entry(json!([n, k]), "fp8_e4m3fn_row_bf16", "row_scale_v1")
        } else {
            tensor_entry(json!([n, k]), "bf16", "contiguous_le_v1")
        };
        let id = self.push(format!("hf/{id}"), entry, pieces);
        let policy = if fp8 { Policy::A8 } else { Policy::A16 };
        for &(binding, r0, rows) in parts {
            if r0 + rows > n {
                return Err(Error::Config(format!(
                    "{binding}: rows {r0}+{rows} past {id} [{n}]"
                )));
            }
            self.bind(&id, (n, k), (binding, r0, rows), &policy);
        }
        Ok(())
    }

    /// The fp32 object holding divisor `v` (shared by every use with the same value).
    fn divisor(&mut self, v: f32) -> String {
        if let Some(id) = self.divisors.get(&v.to_bits()) {
            return id.clone();
        }
        let id = format!("hf/divisor/{:08x}", v.to_bits());
        self.push(
            id.clone(),
            tensor_entry(json!([1]), "fp32", "contiguous_le_v1"),
            vec![Piece::Data(v.to_le_bytes().to_vec())],
        );
        self.divisors.insert(v.to_bits(), id.clone());
        id
    }

    /// The single f32 value of tensor `name`.
    fn scalar(&self, name: &str) -> Result<f32> {
        let t = self.get(name)?;
        if t.dtype != "F32" || t.bytes != 4 {
            return Err(Error::Unsupported(format!(
                "{name}: {} {:?}, expected one F32",
                t.dtype, t.shape
            )));
        }
        let mut b = [0u8; 4];
        self.st.read_at(t.file, t.at, &mut b)?;
        Ok(f32::from_le_bytes(b))
    }

    /// compressed-tensors NVFP4 linears `bases` (`<base>.weight_packed` etc.) stacked into one
    /// `nvfp4` object `id`, bound as `parts`. Their weight global scales must be equal.
    fn nvfp4(&mut self, id: &str, bases: &[&str], parts: &[Part]) -> Result<()> {
        let mut k0 = None;
        let mut divisor = None;
        let mut codes = Vec::new();
        let mut srcs = Vec::new();
        // (first row, rows, activation divisor) per base.
        let mut spans = Vec::new();
        let mut n = 0u64;
        for base in bases {
            let packed = self.get(&format!("{base}.weight_packed"))?;
            let scale = self.get(&format!("{base}.weight_scale"))?;
            let (rows, half) = packed.matrix();
            let k = 2 * half;
            if packed.dtype != "U8"
                || packed.shape.len() != 2
                || scale.dtype != "F8_E4M3"
                || scale.shape != [rows, k / 16]
            {
                return Err(Error::Unsupported(format!(
                    "{base}: packed {} {:?} / scale {} {:?} is not NVFP4 (group 16)",
                    packed.dtype, packed.shape, scale.dtype, scale.shape
                )));
            }
            let gs = self.scalar(&format!("{base}.weight_global_scale"))?;
            if *k0.get_or_insert(k) != k || divisor.get_or_insert(gs).to_bits() != gs.to_bits() {
                return Err(Error::Config(format!(
                    "{id}: {base} differs from the other stacked tensors in columns or global scale"
                )));
            }
            if !(gs.is_finite() && gs > 0.0) {
                return Err(Error::Config(format!("{base}: weight_global_scale {gs}")));
            }
            let input = format!("{base}.input_global_scale");
            let act = if self.has(&input) {
                let v = self.scalar(&input)?;
                (v.is_finite() && v > 0.0).then_some(v)
            } else {
                None
            };
            codes.push(Piece::File {
                file: packed.file,
                at: packed.at,
                len: packed.bytes,
            });
            srcs.push((scale.file, scale.at, rows));
            spans.push((n, rows, act));
            n += rows;
        }
        let (k, divisor) = (k0.unwrap_or(0), divisor.unwrap_or(1.0));
        if !n.is_multiple_of(128) || !k.is_multiple_of(64) {
            return Err(Error::Unsupported(format!(
                "{id}: nvfp4 needs N % 128 == 0 and K % 64 == 0, got [{n}, {k}]"
            )));
        }
        let mut pieces = codes;
        pieces.push(Piece::Zero(
            align256((n * k / 2) as usize) as u64 - n * k / 2,
        ));
        pieces.push(Piece::Nvfp4Scales { srcs, k });
        pieces.push(Piece::Data(divisor.to_le_bytes().to_vec()));
        let id = self.push(
            format!("hf/{id}"),
            tensor_entry(json!([n, k]), "nvfp4", "block_scale_k16_m128x4_v1"),
            pieces,
        );
        for &(binding, r0, rows) in parts {
            if r0 + rows > n {
                return Err(Error::Config(format!(
                    "{binding}: rows {r0}+{rows} past {id} [{n}]"
                )));
            }
            // AllowA4 with one divisor when every base under the part agrees, else A8.
            let acts: Vec<Option<f32>> = spans
                .iter()
                .filter(|(s, len, _)| *s < r0 + rows && r0 < s + len)
                .map(|x| x.2)
                .collect();
            let policy = match acts.first() {
                Some(Some(d))
                    if acts
                        .iter()
                        .all(|a| a.map(f32::to_bits) == Some(d.to_bits())) =>
                {
                    Policy::A4(self.divisor(*d))
                }
                _ => Policy::A8,
            };
            self.bind(&id, (n, k), (binding, r0, rows), &policy);
        }
        Ok(())
    }

    fn is_nvfp4(&self, base: &str) -> bool {
        self.has(&format!("{base}.weight_packed"))
    }

    /// Linears `bases` (`<base>.weight`, BF16 / FP8 / NVFP4) stacked as one object when the encoding
    /// allows, else one object each; `parts[i]` covers all of `bases[i]`'s rows.
    fn linears(&mut self, id: &str, bases: &[String], parts: &[&str]) -> Result<()> {
        assert_eq!(bases.len(), parts.len());
        let rows: Vec<u64> = bases
            .iter()
            .map(|b| {
                Ok(if self.is_nvfp4(b) {
                    self.get(&format!("{b}.weight_packed"))?.matrix().0
                } else {
                    self.get(&format!("{b}.weight"))?.matrix().0
                })
            })
            .collect::<Result<_>>()?;
        let mut spans = Vec::new();
        let mut r0 = 0;
        for (p, r) in parts.iter().zip(&rows) {
            spans.push((*p, r0, *r));
            r0 += r;
        }
        let nv: Vec<bool> = bases.iter().map(|b| self.is_nvfp4(b)).collect();
        if nv.iter().all(|&x| !x) {
            let srcs: Vec<RowSrc> = bases
                .iter()
                .map(|b| RowSrc::all(format!("{b}.weight")))
                .collect();
            return self.stacked(id, &srcs, &spans);
        }
        if nv.iter().all(|&x| x) {
            let gs = bases
                .iter()
                .map(|b| {
                    self.scalar(&format!("{b}.weight_global_scale"))
                        .map(f32::to_bits)
                })
                .collect::<Result<Vec<_>>>()?;
            if gs.windows(2).all(|w| w[0] == w[1]) {
                let b: Vec<&str> = bases.iter().map(String::as_str).collect();
                return self.nvfp4(id, &b, &spans);
            }
        }
        // Separate objects.
        for (b, p) in bases.iter().zip(parts) {
            if self.is_nvfp4(b) {
                let rows = self.get(&format!("{b}.weight_packed"))?.matrix().0;
                self.nvfp4(b, &[b.as_str()], &[(p, 0, rows)])?;
            } else {
                let rows = self.get(&format!("{b}.weight"))?.matrix().0;
                self.stacked(b, &[RowSrc::all(format!("{b}.weight"))], &[(p, 0, rows)])?;
            }
        }
        Ok(())
    }

    fn linear(&mut self, base: &str, binding: &str) -> Result<()> {
        self.linears(base, &[base.to_string()], &[binding])
    }
}

/// Checks the matrix bound as `binding` has `k` columns (the loader checks rows).
fn check_k(b: &Builder, binding: &str, k: u64) -> Result<()> {
    let v = &b.bindings[binding];
    let id = v["object"]
        .as_str()
        .or_else(|| v["parts"][0]["object"].as_str())
        .unwrap_or("");
    let o = b
        .outs
        .iter()
        .rev()
        .find(|o| o.entry["id"] == id)
        .ok_or_else(|| Error::Config(format!("{binding}: no object")))?;
    if o.entry["shape"][1].as_u64() != Some(k) {
        return Err(Error::Config(format!(
            "{binding}: {} columns, config implies {k}",
            o.entry["shape"][1]
        )));
    }
    Ok(())
}

/// One decoder block: `src` is the checkpoint prefix (`model.language_model.layers.3`), `dst` the
/// binding prefix (`text/layers/3`).
fn block(b: &mut Builder, c: &TextConfig, kind: LayerKind, src: &str, dst: &str) -> Result<()> {
    let t = |s: &str| format!("{src}.{s}");
    let p = |s: &str| format!("{dst}/{s}");
    let hidden = c.hidden as u64;
    b.vector(&p("input_norm"), &t("input_layernorm.weight"), c.hidden)?;
    b.vector(
        &p("post_attention_norm"),
        &t("post_attention_layernorm.weight"),
        c.hidden,
    )?;
    match kind {
        LayerKind::Gdn => {
            let kd = (c.gdn_k_heads * c.gdn_k_dim) as u64;
            let v = c.gdn_v() as u64;
            let nv = c.gdn_v_heads as u64;
            // q | k | v (grouped value heads, as HF stores them) | z: one object.
            b.stacked(
                &t("linear_attn.in_proj_qkvz"),
                &[
                    RowSrc::all(t("linear_attn.in_proj_qkv.weight")),
                    RowSrc::all(t("linear_attn.in_proj_z.weight")),
                ],
                &[
                    (&p("gdn/query"), 0, kd),
                    (&p("gdn/key"), kd, kd),
                    (&p("gdn/value"), 2 * kd, v),
                    (&p("gdn/z"), 2 * kd + v, v),
                ],
            )?;
            b.stacked(
                &t("linear_attn.in_proj_ab"),
                &[
                    RowSrc::all(t("linear_attn.in_proj_a.weight")),
                    RowSrc::all(t("linear_attn.in_proj_b.weight")),
                ],
                &[
                    (&p("gdn/a_projection"), 0, nv),
                    (&p("gdn/b_projection"), nv, nv),
                ],
            )?;
            b.linear(&t("linear_attn.out_proj"), &p("gdn/output"))?;
            check_k(b, &p("gdn/query"), hidden)?;
            check_k(b, &p("gdn/a_projection"), hidden)?;
            check_k(b, &p("gdn/output"), v)?;
            b.conv(
                &p("gdn/convolution"),
                &t("linear_attn.conv1d.weight"),
                c.gdn_qkv(),
                c.conv_width,
            )?;
            b.vector(&p("gdn/a_log"), &t("linear_attn.A_log"), c.gdn_v_heads)?;
            b.vector(&p("gdn/dt_bias"), &t("linear_attn.dt_bias"), c.gdn_v_heads)?;
            b.vector(&p("gdn/norm"), &t("linear_attn.norm.weight"), c.gdn_v_dim)?;
        }
        LayerKind::FullAttention => {
            // q_proj rows: head h's query at [2h d, (2h + 1) d), its gate right after.
            let (heads, d) = (c.heads, c.head_dim);
            let mut perm = Vec::with_capacity(2 * heads * d);
            for gate in 0..2 {
                for h in 0..heads {
                    perm.extend((0..d).map(|e| ((2 * h + gate) * d + e) as u32));
                }
            }
            let (q, kv) = (c.q_dim() as u64, c.kv_dim() as u64);
            let qw = t("self_attn.q_proj.weight");
            if b.get(&qw)?.matrix().0 != 2 * q {
                return Err(Error::Config(format!(
                    "{qw}: {:?}, config implies {} rows (query and gate)",
                    b.get(&qw)?.shape,
                    2 * q
                )));
            }
            b.stacked(
                &t("self_attn.qkv"),
                &[
                    RowSrc {
                        weight: qw,
                        sel: Sel::Perm(perm),
                    },
                    RowSrc::all(t("self_attn.k_proj.weight")),
                    RowSrc::all(t("self_attn.v_proj.weight")),
                ],
                &[
                    (&p("attention/query"), 0, q),
                    (&p("attention/gate"), q, q),
                    (&p("attention/key"), 2 * q, kv),
                    (&p("attention/value"), 2 * q + kv, kv),
                ],
            )?;
            b.linear(&t("self_attn.o_proj"), &p("attention/output"))?;
            check_k(b, &p("attention/query"), hidden)?;
            check_k(b, &p("attention/output"), q)?;
            b.vector(
                &p("attention/query_norm"),
                &t("self_attn.q_norm.weight"),
                c.head_dim,
            )?;
            b.vector(
                &p("attention/key_norm"),
                &t("self_attn.k_norm.weight"),
                c.head_dim,
            )?;
        }
    }
    match &c.ffn {
        FfnConfig::Dense { .. } => {
            b.linears(
                &t("mlp.gate_up"),
                &[t("mlp.gate_proj"), t("mlp.up_proj")],
                &[&p("mlp/gate"), &p("mlp/up")],
            )?;
            b.linear(&t("mlp.down_proj"), &p("mlp/down"))?;
            check_k(b, &p("mlp/gate"), hidden)?;
        }
        FfnConfig::Moe(m) => {
            let e = m.experts as u64;
            // Router rows, then the shared-expert gate row: one GEMM.
            b.stacked(
                &t("mlp.router_shared_gate"),
                &[
                    RowSrc::all(t("mlp.gate.weight")),
                    RowSrc::all(t("mlp.shared_expert_gate.weight")),
                ],
                &[(&p("moe/router"), 0, e), (&p("moe/shared_score"), e, 1)],
            )?;
            check_k(b, &p("moe/router"), hidden)?;
            let (i, h) = (m.intermediate as u64, hidden);
            let fused = t("mlp.experts.gate_up_proj");
            if b.has(&fused) {
                // transformers' fused experts: gate_up_proj [E, 2I, H] (gate rows, then up rows),
                // down_proj [E, H, I].
                let gu = b.get(&fused)?;
                let down = t("mlp.experts.down_proj");
                let dn = b.get(&down)?;
                if gu.shape != [e, 2 * i, h] || dn.shape != [e, h, i] {
                    return Err(Error::Config(format!(
                        "{fused} {:?} / {down} {:?}, config implies [{e}, {}, {h}] / [{e}, {h}, {i}]",
                        gu.shape,
                        dn.shape,
                        2 * i
                    )));
                }
                // NVFP4 like the stack's experts when the shapes allow (the MTP layer's experts
                // then share their slot size and cost 3.5x less to read), else BF16 as stored.
                let quantize = i.is_multiple_of(128)
                    && h.is_multiple_of(128)
                    && i.is_multiple_of(64)
                    && h.is_multiple_of(64);
                for x in (0..e).filter(|_| quantize) {
                    let q = |r: &str| p(&format!("moe/experts/{x}/{r}"));
                    let id = |r: &str| format!("{}.{x}.{r}_proj", t("mlp.experts"));
                    let r0 = 2 * i * x;
                    b.nvfp4_from_bf16(&id("gate"), &fused, r0, i, &q("gate"))?;
                    b.nvfp4_from_bf16(&id("up"), &fused, r0 + i, i, &q("up"))?;
                    b.nvfp4_from_bf16(&id("down"), &down, h * x, h, &q("down"))?;
                }
                for x in (0..e).filter(|_| !quantize) {
                    let q = |r: &str| p(&format!("moe/experts/{x}/{r}"));
                    let r0 = 2 * i * x;
                    for (role, a) in [("gate", r0), ("up", r0 + i)] {
                        b.stacked(
                            &format!("{}.{x}.{role}_proj", t("mlp.experts")),
                            &[RowSrc {
                                weight: fused.clone(),
                                sel: Sel::Range(a, a + i),
                            }],
                            &[(&q(role), 0, i)],
                        )?;
                    }
                    b.stacked(
                        &format!("{}.{x}.down_proj", t("mlp.experts")),
                        &[RowSrc {
                            weight: down.clone(),
                            sel: Sel::Range(h * x, h * (x + 1)),
                        }],
                        &[(&q("down"), 0, h)],
                    )?;
                }
            } else {
                for x in 0..e {
                    let q = |r: &str| p(&format!("moe/experts/{x}/{r}"));
                    for role in ["gate", "up", "down"] {
                        b.linear(&t(&format!("mlp.experts.{x}.{role}_proj")), &q(role))?;
                    }
                }
            }
            check_k(b, &p("moe/experts/0/gate"), hidden)?;
            check_k(b, &p("moe/experts/0/down"), i)?;
            b.linears(
                &t("mlp.shared_expert.gate_up"),
                &[
                    t("mlp.shared_expert.gate_proj"),
                    t("mlp.shared_expert.up_proj"),
                ],
                &[&p("moe/shared/gate"), &p("moe/shared/up")],
            )?;
            b.linear(&t("mlp.shared_expert.down_proj"), &p("moe/shared/down"))?;
            check_k(b, &p("moe/shared/gate"), hidden)?;
        }
    }
    Ok(())
}

/// The text config velox reads, from an HF `config.json` (its `text_config` when present).
pub fn text_config(cfg: &Value) -> Result<Value> {
    let mut t = cfg
        .get("text_config")
        .unwrap_or(cfg)
        .as_object()
        .cloned()
        .ok_or_else(|| Error::Config("config.json is not an object".into()))?;
    let moe = t.contains_key("num_experts");
    let arch = if moe {
        MOE_ARCHITECTURE
    } else {
        DENSE_ARCHITECTURE
    };
    t.insert("architectures".into(), json!([arch]));
    t.remove("quantization_config");
    if moe {
        // A MoE config must not carry the Dense width (transformers may fill a default).
        t.remove("intermediate_size");
    }
    let v = Value::Object(t);
    TextConfig::from_json(&v)?;
    Ok(v)
}

/// Text resources: role -> bytes.
fn text_resources(dir: &Path) -> Result<Vec<(String, Vec<u8>)>> {
    let read = |name: &str| std::fs::read(dir.join(name)).map_err(|e| ioerr(&dir.join(name), e));
    let mut out = Vec::new();
    for name in ["tokenizer.json", "generation_config.json"] {
        out.push((name.to_string(), read(name)?));
    }
    let tc = dir.join("tokenizer_config.json");
    let tc_bytes = if tc.exists() {
        Some(read("tokenizer_config.json")?)
    } else {
        None
    };
    let template = if dir.join("chat_template.jinja").exists() {
        read("chat_template.jinja")?
    } else {
        tc_bytes
            .as_deref()
            .and_then(|b| serde_json::from_slice::<Value>(b).ok())
            .and_then(|v| v["chat_template"].as_str().map(|s| s.as_bytes().to_vec()))
            .ok_or_else(|| {
                Error::Config(format!(
                    "{}: no chat_template.jinja and no chat_template in tokenizer_config.json",
                    dir.display()
                ))
            })?
    };
    out.push(("chat_template.jinja".into(), template));
    if let Some(b) = tc_bytes {
        out.push(("tokenizer_config.json".into(), b));
    }
    Ok(out)
}

/// The prefix of the text stack's tensors (`model.language_model.` for the multimodal checkpoints,
/// `model.` for text-only ones).
fn text_prefix(st: &SafeTensors) -> Result<&'static str> {
    for p in ["model.language_model.", "model."] {
        if st.tensor(&format!("{p}embed_tokens.weight")).is_some() {
            return Ok(p);
        }
    }
    Err(Error::Config(
        "checkpoint has no embed_tokens.weight".into(),
    ))
}

/// Writes `out` from the HF checkpoint in `dir`: text stack, MTP head (when present) and the
/// tokenizer, chat template and generation config. `progress` gets a line per written gigabyte.
pub fn import(dir: &Path, out: &Path, progress: &mut dyn FnMut(&str)) -> Result<ImportReport> {
    import_with(dir, out, None, false, progress)
}

/// Tensor-name prefix of the DFlash2 drafter's checkpoint inside the merged tensor set.
const DFLASH: &str = "dflash::";

/// [`import`], adding the DFlash2 drafter checkpoint in `dflash` (a `DFlash2DraftModel`
/// conditioned on this target) as the `dflash2` component when given; `dflash_nvfp4` encodes
/// its MLP as NVFP4 (drafts only steer speed, never results; quantising attention, the conv
/// kernels or the selector too collapses acceptance).
pub fn import_with(
    dir: &Path,
    out: &Path,
    dflash: Option<&Path>,
    dflash_nvfp4: bool,
    progress: &mut dyn FnMut(&str),
) -> Result<ImportReport> {
    let hf_cfg = read_json(&dir.join("config.json"))?;
    let cfg_json = text_config(&hf_cfg)?;
    let c = TextConfig::from_json(&cfg_json)?;
    let mut st = SafeTensors::open_dir(dir)?;
    let dflash_cfg = match dflash {
        Some(d) => {
            let v = read_json(&d.join("config.json"))?;
            st.merge(SafeTensors::open_dir(d)?, DFLASH);
            Some(v)
        }
        None => None,
    };
    let mut b = Builder::new(&st);
    let pre = text_prefix(&st)?;

    // Text stack.
    let embed = format!("{pre}embed_tokens.weight");
    let vocab = c.vocab as u64;
    b.stacked(
        &format!("{pre}embed_tokens"),
        &[RowSrc::all(embed)],
        &[("text/token_embedding", 0, vocab)],
    )?;
    if !b.has("lm_head.weight") && !b.has("lm_head.weight_packed") {
        return Err(Error::Unsupported(
            "checkpoint has no lm_head (tied embeddings)".into(),
        ));
    }
    b.linear("lm_head", "text/output_head")?;
    check_k(&b, "text/token_embedding", c.hidden as u64)?;
    check_k(&b, "text/output_head", c.hidden as u64)?;
    b.vector("text/final_norm", &format!("{pre}norm.weight"), c.hidden)?;
    for (i, kind) in c.layers.iter().enumerate() {
        block(
            &mut b,
            &c,
            *kind,
            &format!("{pre}layers.{i}"),
            &format!("text/layers/{i}"),
        )?;
        progress(&format!("layer {}/{} planned", i + 1, c.layers.len()));
    }

    // MTP head: input projection over [norm_e(embed), norm_h(hidden)], one attention block.
    let mtp_layers = cfg_json
        .get("mtp_num_hidden_layers")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let mtp = mtp_layers > 0 && b.has("mtp.fc.weight");
    if mtp {
        if mtp_layers > 1 {
            return Err(Error::Unsupported(format!(
                "MTP with {mtp_layers} layers (velox runs one)"
            )));
        }
        let h = c.hidden as u64;
        b.stacked(
            "mtp.fc",
            &[RowSrc::all("mtp.fc.weight".into())],
            &[("mtp/input_projection", 0, h)],
        )?;
        check_k(&b, "mtp/input_projection", 2 * h)?;
        b.vector(
            "mtp/embedding_norm",
            "mtp.pre_fc_norm_embedding.weight",
            c.hidden,
        )?;
        b.vector("mtp/hidden_norm", "mtp.pre_fc_norm_hidden.weight", c.hidden)?;
        b.vector("mtp/final_norm", "mtp.norm.weight", c.hidden)?;
        block(
            &mut b,
            &c,
            LayerKind::FullAttention,
            "mtp.layers.0",
            "mtp/layers/0",
        )?;
    }

    if let Some(v) = &dflash_cfg {
        dflash2(&mut b, &c, v, dflash_nvfp4)?;
    }

    // Resources (text). The vision tower (`model.visual.*`, a `vision` component with
    // `preprocessor_config.json`) would be added here.
    let mut resources = Map::new();
    for (role, bytes) in text_resources(dir)? {
        let mut e = Map::new();
        e.insert("kind".into(), "resource".into());
        e.insert("encoding".into(), "raw_bytes_v1".into());
        let id = b.push(format!("resource/text/{role}"), e, vec![Piece::Data(bytes)]);
        resources.insert(role, id.into());
    }
    let Builder {
        mut outs,
        bindings,
        uses,
        ..
    } = b;

    // Layout: objects back to back on 256-byte boundaries.
    let mut at = 0u64;
    let mut objects = Vec::with_capacity(outs.len());
    for o in &mut outs {
        o.offset = at;
        let mut e = o.entry.clone();
        e.insert("offset".into(), at.into());
        e.insert("bytes".into(), o.bytes.into());
        objects.push(Value::Object(e));
        at = (at + o.bytes).div_ceil(256) * 256;
    }
    let payload = outs.last().map_or(0, |o| o.offset + o.bytes);

    let name = dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_string();
    let mut components = Map::new();
    components.insert(
        "text".into(),
        json!({ "config": cfg_json, "resources": resources }),
    );
    if let Some(v) = dflash_cfg {
        components.insert("dflash2".into(), json!({ "config": v }));
    }
    let report_counts = (objects.len(), bindings.len(), uses.len());
    let mut d = Map::new();
    d.insert("components".into(), Value::Object(components));
    d.insert("objects".into(), Value::Array(objects));
    d.insert("bindings".into(), Value::Object(bindings));
    d.insert("uses".into(), Value::Array(uses));
    d.insert("metadata".into(), json!({ "name": name }));
    d.insert(
        "provenance".into(),
        json!({
            "converter": "velox import-hf",
            "hf": {
                "dir": name,
                "architectures": hf_cfg.get("architectures"),
                "quantization": hf_cfg
                    .get("quantization_config")
                    .map(|q| json!({ "quant_method": q.get("quant_method"), "format": q.get("format") })),
            },
        }),
    );
    d.insert(
        "velox".into(),
        json!({ "format_version": 1, "source_container": "safetensors" }),
    );

    let mut formats: Vec<(String, usize, u64)> = Vec::new();
    for o in &outs {
        let f = o
            .entry
            .get("format")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| "resource".into());
        match formats.iter_mut().find(|x| x.0 == f) {
            Some(x) => {
                x.1 += 1;
                x.2 += o.bytes;
            }
            None => formats.push((f, 1, o.bytes)),
        }
    }
    formats.sort_by_key(|x| std::cmp::Reverse(x.2));

    let mut next_report = 1u64 << 30;
    // The last swizzled NVFP4 scale plane (object index, bytes).
    let mut plane: Option<(usize, Vec<u8>)> = None;
    let mut fill = |at: u64, buf: &mut [u8]| -> velox_weights::Result<()> {
        buf.fill(0);
        let end = at + buf.len() as u64;
        let first = outs.partition_point(|o| o.offset + o.bytes <= at);
        for (oi, o) in outs.iter().enumerate().skip(first) {
            if o.offset >= end {
                break;
            }
            let mut p0 = o.offset;
            for piece in &o.pieces {
                let len = piece.len();
                let (lo, hi) = (at.max(p0), end.min(p0 + len));
                if lo < hi {
                    let dst = &mut buf[(lo - at) as usize..(hi - at) as usize];
                    let rel = lo - p0;
                    read_piece(&st, piece, oi, rel, dst, &mut plane)
                        .map_err(|e| velox_weights::Error::Format(e.to_string()))?;
                }
                p0 += len;
            }
        }
        if end >= next_report {
            progress(&format!(
                "{:.1} / {:.1} GB",
                end as f64 / 1e9,
                payload as f64 / 1e9
            ));
            next_report += 1 << 30;
        }
        Ok(())
    };
    velox_weights::write_container(
        out,
        d,
        Container::VeloxV1,
        velox_weights::DEFAULT_PART_LIMIT,
        velox_weights::new_artifact_id(),
        payload,
        &mut fill,
    )
    .map_err(werr)?;
    // The written artifact must pass the reader's structural checks.
    Artifact::open(out).map_err(werr)?;
    Ok(ImportReport {
        formats,
        objects: report_counts.0,
        bindings: report_counts.1,
        uses: report_counts.2,
        payload_bytes: payload,
        mtp,
    })
}

/// Bytes `[rel, rel + dst.len())` of `piece` (of output object `oi`).
fn read_piece(
    st: &SafeTensors,
    piece: &Piece,
    oi: usize,
    rel: u64,
    dst: &mut [u8],
    plane: &mut Option<(usize, Vec<u8>)>,
) -> Result<()> {
    match piece {
        Piece::File { file, at, .. } => st.read_at(*file, at + rel, dst),
        Piece::Zero(_) => {
            dst.fill(0);
            Ok(())
        }
        Piece::Data(v) => {
            dst.copy_from_slice(&v[rel as usize..rel as usize + dst.len()]);
            Ok(())
        }
        Piece::Rows {
            file,
            at,
            row_bytes,
            perm,
        } => {
            let mut done = 0;
            while done < dst.len() {
                let p = rel + done as u64;
                let (r, col) = (p / row_bytes, p % row_bytes);
                let n = ((row_bytes - col) as usize).min(dst.len() - done);
                let src = at + u64::from(perm[r as usize]) * row_bytes + col;
                st.read_at(*file, src, &mut dst[done..done + n])?;
                done += n;
            }
            Ok(())
        }
        Piece::Nvfp4Scales { srcs, k } => {
            if plane.as_ref().map(|p| p.0) != Some(oi) {
                let k = *k as usize;
                let groups = k / 16;
                let n: u64 = srcs.iter().map(|s| s.2).sum();
                let mut out = vec![0u8; n as usize * groups];
                let mut r0 = 0usize;
                for &(file, at, rows) in srcs {
                    let mut raw = vec![0u8; rows as usize * groups];
                    st.read_at(file, at, &mut raw)?;
                    for r in 0..rows as usize {
                        for g in 0..groups {
                            out[nvfp4_scale_offset(r0 + r, g, k)] = raw[r * groups + g];
                        }
                    }
                    r0 += rows as usize;
                }
                *plane = Some((oi, out));
            }
            let v = &plane.as_ref().unwrap().1;
            dst.copy_from_slice(&v[rel as usize..rel as usize + dst.len()]);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests;

/// The DFlash2 drafter (`dflash2/`): HF `DFlash2DraftModel` names to velox's bindings. All BF16
/// as stored; norms are plain multipliers (Qwen3 RMSNorm). The context key / value projections
/// share the block's key / value rows.
fn dflash2(b: &mut Builder<'_>, c: &TextConfig, v: &Value, nvfp4: bool) -> Result<()> {
    let dc = super::dflash::DFlashConfig::from_json(v)?;
    let h = c.hidden as u64;
    if dc.target_layers.iter().any(|&l| l >= c.layers.len()) {
        return Err(Error::Config(format!(
            "dflash2 target layers {:?} past the target's {} layers",
            dc.target_layers,
            c.layers.len()
        )));
    }
    let t = |s: &str| format!("{DFLASH}{s}");
    // Whole tensors `srcs` stacked into one matrix `id`, bound as `parts`: NVFP4 when asked and
    // the shape allows, else as stored.
    let mat =
        |b: &mut Builder<'_>, id: &str, srcs: &[String], parts: &[Part], q: bool| -> Result<()> {
            let shapes: Vec<(u64, u64)> = srcs
                .iter()
                .map(|n| b.get(n).map(|t| t.matrix()))
                .collect::<Result<_>>()?;
            let n: u64 = shapes.iter().map(|s| s.0).sum();
            let k = shapes.first().map_or(0, |s| s.1);
            if q && nvfp4 && n.is_multiple_of(128) && k.is_multiple_of(64) {
                let rows: Vec<(&str, u64, u64)> = srcs
                    .iter()
                    .zip(&shapes)
                    .map(|(n, s)| (n.as_str(), 0, s.0))
                    .collect();
                b.nvfp4_stack(id, &rows, parts)
            } else {
                let rs: Vec<RowSrc> = srcs.iter().map(|n| RowSrc::all(n.clone())).collect();
                b.stacked(id, &rs, parts)
            }
        };
    let one = |b: &mut Builder<'_>, name: &str, binding: &str, rows: u64| {
        mat(b, name, &[name.to_string()], &[(binding, 0, rows)], false)
    };
    one(b, &t("fc.weight"), "dflash2/feature_projection", h)?;
    check_k(
        b,
        "dflash2/feature_projection",
        h * dc.target_layers.len() as u64,
    )?;
    b.vector("dflash2/context_norm", &t("hidden_norm.weight"), c.hidden)?;
    b.vector("dflash2/final_norm", &t("norm.weight"), c.hidden)?;
    let (qd, kvd) = (
        (dc.heads * dc.head_dim) as u64,
        (dc.kv_heads * dc.head_dim) as u64,
    );
    let (inter, groups) = (dc.intermediate as u64, h / dc.conv_group as u64);
    for i in 0..dc.layers {
        let l = |s: &str| t(&format!("layers.{i}.{s}"));
        let q = |s: &str| format!("dflash2/layers/{i}/{s}");
        b.vector(&q("input_norm"), &l("input_layernorm.weight"), c.hidden)?;
        b.vector(
            &q("post_attention_norm"),
            &l("post_attention_layernorm.weight"),
            c.hidden,
        )?;
        b.vector(
            &q("attention/query_norm"),
            &l("self_attn.q_norm.weight"),
            dc.head_dim,
        )?;
        b.vector(
            &q("attention/key_norm"),
            &l("self_attn.k_norm.weight"),
            dc.head_dim,
        )?;
        mat(
            b,
            &l("self_attn.qkv"),
            &[
                l("self_attn.q_proj.weight"),
                l("self_attn.k_proj.weight"),
                l("self_attn.v_proj.weight"),
            ],
            &[
                (&q("attention/query"), 0, qd),
                (&q("attention/key"), qd, kvd),
                (&q("attention/value"), qd + kvd, kvd),
                (&q("attention/context_key"), qd, kvd),
                (&q("attention/context_value"), qd + kvd, kvd),
            ],
            false,
        )?;
        one(b, &l("self_attn.o_proj.weight"), &q("attention/output"), h)?;
        mat(
            b,
            &l("mlp.gate_up"),
            &[l("mlp.gate_proj.weight"), l("mlp.up_proj.weight")],
            &[(&q("mlp/gate"), 0, inter), (&q("mlp/up"), inter, inter)],
            true,
        )?;
        let down = l("mlp.down_proj.weight");
        mat(
            b,
            &down,
            std::slice::from_ref(&down),
            &[(&q("mlp/down"), 0, h)],
            true,
        )?;
        for conv in ["attention_conv", "mlp_conv"] {
            b.vector(
                &q(&format!("{conv}/base_kernel")),
                &l(&format!("{conv}.base_kernel")),
                4 * c.hidden,
            )?;
            one(
                b,
                &l(&format!("{conv}.kernel_projection.weight")),
                &q(&format!("{conv}/kernel_projection")),
                4 * groups,
            )?;
        }
    }
    let rank = dc.rank as u64;
    one(
        b,
        &t("candidate_selector.hidden_projection.weight"),
        "dflash2/candidate_selector/hidden_projection",
        rank,
    )?;
    let vocab = c.vocab as u64;
    for book in ["predecessor_codebook", "successor_codebook"] {
        let name = t(&format!("candidate_selector.{book}"));
        b.stacked(
            &name,
            &[RowSrc::all(name.clone())],
            &[(&format!("dflash2/candidate_selector/{book}"), 0, vocab)],
        )?;
    }
    Ok(())
}
