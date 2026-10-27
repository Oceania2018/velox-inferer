//! Loads Qwen4-Exp weights straight from llama.cpp GGUF shards (docs/qwen4_exp.md#gguf-mapping).
//!
//! Matrices keep their GGML blocks; only whole rows move, to undo llama.cpp's tiled GDN value
//! heads and interleaved query/gate rows exactly as the Qwen3.5 import does. Routed experts become
//! per-expert records `[gate | up | down]` in system RAM the device reads directly; the PLE table
//! stays in the file.

use std::path::{Path, PathBuf};

use velox_core::format::{bf16_to_f32, f16_to_f32, f32_to_bf16};
use velox_core::ggml::GgmlType;
use velox_core::{ActPolicy, Backend, Error, ExpertMatrix, Result, WeightFormat, WeightView};
use velox_weights::gguf::{Gguf, GgufTensor};

use super::config::{LayerKind, Qwen4Config};
use super::ple::PleTable;
use crate::experts::ExpertRecords;

fn werr(e: velox_weights::Error) -> Error {
    Error::Config(e.to_string())
}

/// A stored matrix `[rows, k]` and the activation rounding its use permits.
pub struct Mat<B: Backend> {
    pub buf: B::Buffer,
    pub format: WeightFormat,
    pub rows: usize,
    pub k: usize,
    pub act: ActPolicy,
}

impl<B: Backend> Mat<B> {
    pub fn view(&self) -> WeightView<'_, B::Buffer> {
        WeightView::full(&self.buf, self.format, self.rows, self.k).with_act(self.act)
    }
}

/// A hyper-connection mixer: the per-stream norm, the low-rank gate and (except the head's) the
/// injection logits. `down` and `inject` hold the source weights times `1 / streams`.
pub struct HcWeights<B: Backend> {
    pub norm: B::Buffer,
    pub down: Mat<B>,
    pub up: Mat<B>,
    pub inject: Option<Mat<B>>,
}

pub struct GdnWeights<B: Backend> {
    /// q | k | v rows, value heads grouped.
    pub qkv: Mat<B>,
    /// Tiled value heads.
    pub z: Mat<B>,
    /// a rows then b rows `[2 v_heads, hidden]`, grouped.
    pub ab: Mat<B>,
    /// `[width, channels]`, value channels grouped.
    pub conv: B::Buffer,
    pub a_log: B::Buffer,
    pub dt_bias: B::Buffer,
    pub norm: B::Buffer,
    /// Tiled value-head columns.
    pub out: Mat<B>,
}

pub struct QsaWeights<B: Backend> {
    /// Query rows then gate rows `[2 q_dim, hidden]`.
    pub q: Mat<B>,
    pub k: Mat<B>,
    pub v: Mat<B>,
    pub o: Mat<B>,
    pub q_norm: B::Buffer,
    pub k_norm: B::Buffer,
    pub index_q: Mat<B>,
    pub index_k: Mat<B>,
    pub index_q_norm: B::Buffer,
    pub index_k_norm: B::Buffer,
}

pub enum Mixer<B: Backend> {
    Gdn(GdnWeights<B>),
    Qsa(QsaWeights<B>),
}

pub struct MoeWeights<B: Backend> {
    /// Router rows then the shared expert's gate row `[experts + 1, hidden]`.
    pub router: Mat<B>,
    pub experts: ExpertRecords<B>,
    pub shared_gate: Mat<B>,
    pub shared_up: Mat<B>,
    pub shared_down: Mat<B>,
}

pub struct PleWeights<B: Backend> {
    pub key: Mat<B>,
    pub value: Mat<B>,
    pub norm_key: B::Buffer,
    pub norm_query: B::Buffer,
    pub norm_conv: B::Buffer,
    /// `[width, streams * hidden]`.
    pub conv: B::Buffer,
}

pub struct Layer<B: Backend> {
    pub hc_attn: HcWeights<B>,
    pub mixer: Mixer<B>,
    pub hc_ffn: HcWeights<B>,
    pub moe: MoeWeights<B>,
    pub ple: Option<PleWeights<B>>,
}

pub struct Qwen4Weights<B: Backend> {
    /// In system RAM: one row per token.
    pub embedding: Mat<B>,
    pub output: Mat<B>,
    pub hc_head: HcWeights<B>,
    pub layers: Vec<Layer<B>>,
    pub ple_table: Option<PleTable>,
    /// Bytes on the device, and in system RAM (expert records, the embedding).
    pub device_bytes: usize,
    pub host_bytes: usize,
}

impl<B: Backend> Qwen4Weights<B> {
    /// The largest expert record of any layer: the slot size of a shared pool.
    /// Every layer's routed experts, for [`crate::experts::ExpertCache::update`].
    pub fn expert_records(&self) -> Vec<&ExpertRecords<B>> {
        self.layers.iter().map(|l| &l.moe.experts).collect()
    }

    pub fn max_record_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.moe.experts.record_bytes)
            .max()
            .unwrap_or(16)
    }
}

/// The GGUF shards of one model, tensors looked up across all of them.
pub struct Shards {
    pub files: Vec<Gguf>,
}

impl Shards {
    /// Opens `first` and, for a split GGUF (`split.count`), its siblings named
    /// `...-0000i-of-0000n.gguf` next to it.
    pub fn open(first: &Path) -> Result<Self> {
        let g = Gguf::open(first).map_err(werr)?;
        let count = g.meta_u64("split.count").unwrap_or(1) as usize;
        let mut files = vec![g];
        if count > 1 {
            let name = first
                .file_name()
                .and_then(|n| n.to_str())
                .ok_or_else(|| Error::Config(format!("{}: bad file name", first.display())))?;
            let tag = format!("-00001-of-{count:05}.gguf");
            let stem = name.strip_suffix(&tag).ok_or_else(|| {
                Error::Config(format!(
                    "{name}: split GGUF of {count} shards should end in {tag}"
                ))
            })?;
            for i in 2..=count {
                let p: PathBuf = first.with_file_name(format!("{stem}-{i:05}-of-{count:05}.gguf"));
                files.push(Gguf::open(&p).map_err(werr)?);
            }
        }
        Ok(Shards { files })
    }

    /// Adds the tensors of another GGUF (a sidecar such as an MTP head).
    pub fn add(&mut self, path: &Path) -> Result<()> {
        self.files.push(Gguf::open(path).map_err(werr)?);
        Ok(())
    }

    pub fn first(&self) -> &Gguf {
        &self.files[0]
    }

    pub fn tensor(&self, name: &str) -> Result<(&Gguf, &GgufTensor)> {
        self.files
            .iter()
            .find_map(|g| g.tensor(name).map(|t| (g, t)))
            .ok_or_else(|| Error::Config(format!("GGUF has no tensor {name}")))
    }
}

/// Format and bytes per row of `k` values of a GGUF tensor type.
fn format_of(t: &GgufTensor, k: usize) -> Result<(WeightFormat, usize)> {
    match t.ggml_type {
        0 => Ok((WeightFormat::F32, 4 * k)),
        30 => Ok((WeightFormat::Bf16, 2 * k)),
        id => {
            let g = GgmlType::from_gguf_id(id).ok_or_else(|| {
                Error::Unsupported(format!("{}: ggml type {id} is not supported", t.name))
            })?;
            let f = WeightFormat::Ggml(g);
            f.check_shape(1, k)?;
            Ok((f, g.row_bytes(k)))
        }
    }
}

/// Grouped position of each value head: `grouped[g * per + j] = j * k_heads + g` (its tiled index).
fn tiled_of_grouped(k_heads: usize, per: usize) -> Vec<usize> {
    (0..k_heads * per)
        .map(|h| (h % per) * k_heads + h / per)
        .collect()
}

struct Loader<'a, B: Backend> {
    b: &'a B,
    shards: &'a Shards,
    device_bytes: usize,
    host_bytes: usize,
}

impl<B: Backend> Loader<'_, B> {
    fn read(&self, name: &str) -> Result<(&GgufTensor, Vec<u8>, usize, usize)> {
        let (g, t) = self.shards.tensor(name)?;
        let (k, n) = match t.dims[..] {
            [k] => (k as usize, 1),
            [k, n] => (k as usize, n as usize),
            _ => {
                return Err(Error::Config(format!(
                    "{name}: {:?} is not a matrix",
                    t.dims
                )))
            }
        };
        let row = match t.ggml_type {
            1 => 2 * k,
            _ => format_of(t, k)?.1,
        };
        let mut bytes = vec![0u8; n * row];
        g.read_at(t.offset, &mut bytes).map_err(werr)?;
        Ok((t, bytes, n, k))
    }

    fn upload(&mut self, bytes: &[u8]) -> Result<B::Buffer> {
        self.device_bytes += bytes.len();
        self.b.upload_bytes(bytes)
    }

    /// A GGUF matrix `[n, k]` with rows permuted by `perm` (identity when `None`).
    fn matrix(&mut self, name: &str, perm: Option<&[usize]>) -> Result<Mat<B>> {
        let (t, bytes, n, k) = self.read(name)?;
        let (format, row) = format_of(t, k)?;
        let bytes = match perm {
            None => bytes,
            Some(p) => {
                if p.len() != n {
                    return Err(Error::Config(format!("{name}: permutation of {}", p.len())));
                }
                p.iter()
                    .flat_map(|&r| bytes[r * row..(r + 1) * row].iter().copied())
                    .collect()
            }
        };
        let act = match format {
            WeightFormat::Ggml(_) => ActPolicy::AllowA8,
            _ => ActPolicy::A16Only,
        };
        Ok(Mat {
            buf: self.upload(&bytes)?,
            format,
            rows: n,
            k,
            act,
        })
    }

    /// A bf16 matrix with every value times `scale` (a power of two: exact).
    fn scaled_bf16(&mut self, name: &str, scale: f32) -> Result<Mat<B>> {
        let (t, bytes, n, k) = self.read(name)?;
        if t.ggml_type != 30 {
            return Err(Error::Unsupported(format!("{name}: expected bf16")));
        }
        let bytes: Vec<u8> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|c| f32_to_bf16(bf16_to_f32(u16::from_le_bytes(*c)) * scale).to_le_bytes())
            .collect();
        Ok(Mat {
            buf: self.upload(&bytes)?,
            format: WeightFormat::Bf16,
            rows: n,
            k,
            act: ActPolicy::A16Only,
        })
    }

    /// A small tensor as f32 values (F32, F16 or BF16), read flat.
    fn values(&self, name: &str) -> Result<Vec<f32>> {
        let (t, bytes, ..) = self.read(name)?;
        Ok(match t.ggml_type {
            0 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
            1 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f16_to_f32(u16::from_le_bytes(*c)))
                .collect(),
            30 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| bf16_to_f32(u16::from_le_bytes(*c)))
                .collect(),
            other => {
                return Err(Error::Unsupported(format!(
                    "{name}: small tensor of ggml type {other}"
                )))
            }
        })
    }

    fn vector(&mut self, name: &str, len: usize) -> Result<B::Buffer> {
        let v = self.values(name)?;
        expect_len(name, v.len(), len)?;
        self.device_bytes += 4 * len;
        self.b.upload(&v)
    }

    /// `[channels][width]` as `[width][channels]`, channel `o` taken from source channel
    /// `src(o)`.
    fn conv(
        &mut self,
        name: &str,
        channels: usize,
        width: usize,
        src: impl Fn(usize) -> usize,
    ) -> Result<B::Buffer> {
        let v = self.values(name)?;
        expect_len(name, v.len(), channels * width)?;
        let t: Vec<f32> = (0..width * channels)
            .map(|i| v[src(i % channels) * width + i / channels])
            .collect();
        self.device_bytes += 4 * t.len();
        self.b.upload(&t)
    }

    fn hc(&mut self, prefix: &str, c: &Qwen4Config, inject: bool) -> Result<HcWeights<B>> {
        let s = 1.0 / c.streams as f32;
        Ok(HcWeights {
            norm: self.vector(&format!("{prefix}_norm.weight"), c.wide())?,
            down: self.scaled_bf16(&format!("{prefix}_down.weight"), s)?,
            up: self.matrix(&format!("{prefix}_up.weight"), None)?,
            inject: if inject {
                Some(self.scaled_bf16(&format!("{prefix}_inject.weight"), s)?)
            } else {
                None
            },
        })
    }

    fn gdn(&mut self, i: usize, c: &Qwen4Config) -> Result<GdnWeights<B>> {
        let t = |s: &str| format!("blk.{i}.{s}");
        let (kh, per, hd) = (c.gdn_k_heads, c.gdn_v_heads / c.gdn_k_heads, c.gdn_v_dim);
        let grouped = tiled_of_grouped(kh, per);
        let qk = c.gdn_qk();
        // q | k rows unchanged, value rows back to grouped head order.
        let mut perm: Vec<usize> = (0..qk).collect();
        for &h in &grouped {
            perm.extend((0..hd).map(|e| qk + h * hd + e));
        }
        let qkv = self.matrix(&t("attn_qkv.weight"), Some(&perm))?;
        let z = self.matrix(&t("attn_gate.weight"), None)?;
        // a and b in one [2 v_heads, hidden] parent, both grouped.
        let ab = {
            let (ta, a, na, ka) = self.read(&t("ssm_alpha.weight"))?;
            let (tb, bb, nb, kb) = self.read(&t("ssm_beta.weight"))?;
            if (ta.ggml_type, na, ka) != (tb.ggml_type, nb, kb) || na != c.gdn_v_heads {
                return Err(Error::Config(format!(
                    "layer {i}: ssm_alpha / ssm_beta differ"
                )));
            }
            let (format, row) = format_of(ta, ka)?;
            let mut bytes = Vec::with_capacity(a.len() * 2);
            for src in [&a, &bb] {
                for &h in &grouped {
                    bytes.extend_from_slice(&src[h * row..(h + 1) * row]);
                }
            }
            Mat {
                buf: self.upload(&bytes)?,
                format,
                rows: 2 * na,
                k: ka,
                act: ActPolicy::A16Only,
            }
        };
        let a = self.values(&t("ssm_a"))?;
        let dt = self.values(&t("ssm_dt.bias"))?;
        expect_len(&t("ssm_a"), a.len(), c.gdn_v_heads)?;
        expect_len(&t("ssm_dt.bias"), dt.len(), c.gdn_v_heads)?;
        if let Some(x) = a.iter().find(|x| x.is_nan() || **x >= 0.0) {
            return Err(Error::Config(format!(
                "{}: A = {x} is not negative",
                t("ssm_a")
            )));
        }
        // ssm_a = -exp(A_log).
        let a_log: Vec<f32> = grouped
            .iter()
            .map(|&h| f64::from(-a[h]).ln() as f32)
            .collect();
        let dt: Vec<f32> = grouped.iter().map(|&h| dt[h]).collect();
        self.device_bytes += 8 * c.gdn_v_heads;
        let src_ch = |o: usize| {
            if o < qk {
                o
            } else {
                let (h, e) = ((o - qk) / hd, (o - qk) % hd);
                qk + grouped[h] * hd + e
            }
        };
        Ok(GdnWeights {
            qkv,
            z,
            ab,
            conv: self.conv(&t("ssm_conv1d.weight"), c.gdn_qkv(), c.conv_width, src_ch)?,
            a_log: self.b.upload(&a_log)?,
            dt_bias: self.b.upload(&dt)?,
            norm: self.vector(&t("ssm_norm.weight"), hd)?,
            out: self.matrix(&t("ssm_out.weight"), None)?,
        })
    }

    fn qsa(&mut self, i: usize, c: &Qwen4Config) -> Result<QsaWeights<B>> {
        let t = |s: &str| format!("blk.{i}.{s}");
        // attn_q rows: head h's query at [2h d, (2h + 1) d), its gate right after.
        let (heads, d) = (c.heads, c.head_dim);
        let mut perm = Vec::with_capacity(2 * heads * d);
        for gate in 0..2 {
            for h in 0..heads {
                perm.extend((0..d).map(|e| (2 * h + gate) * d + e));
            }
        }
        Ok(QsaWeights {
            q: self.matrix(&t("attn_q.weight"), Some(&perm))?,
            k: self.matrix(&t("attn_k.weight"), None)?,
            v: self.matrix(&t("attn_v.weight"), None)?,
            o: self.matrix(&t("attn_output.weight"), None)?,
            q_norm: self.vector(&t("attn_q_norm.weight"), d)?,
            k_norm: self.vector(&t("attn_k_norm.weight"), d)?,
            index_q: self.matrix(&t("indexer.q_proj.weight"), None)?,
            index_k: self.matrix(&t("indexer.k_proj.weight"), None)?,
            index_q_norm: self.vector(&t("indexer.q_norm.weight"), c.qsa_dim)?,
            index_k_norm: self.vector(&t("indexer.k_norm.weight"), c.qsa_dim)?,
        })
    }

    /// A layer's routed experts as `[gate | up | down]` records in system RAM, read straight
    /// from the file into place.
    fn experts(&mut self, i: usize, c: &Qwen4Config) -> Result<ExpertRecords<B>> {
        let (e, w, h) = (c.experts, c.expert_width, c.hidden);
        let part =
            |name: &str, rows: usize, k: usize| -> Result<(&Gguf, u64, WeightFormat, usize)> {
                let (g, t) = self.shards.tensor(&format!("blk.{i}.{name}.weight"))?;
                if t.dims[..] != [k as u64, rows as u64, e as u64] {
                    return Err(Error::Config(format!(
                        "blk.{i}.{name}: dims {:?}, expected [{k}, {rows}, {e}]",
                        t.dims
                    )));
                }
                let (f, row) = format_of(t, k)?;
                Ok((g, t.offset, f, rows * row))
            };
        let gate = part("ffn_gate_exps", w, h)?;
        let up = part("ffn_up_exps", w, h)?;
        let down = part("ffn_down_exps", h, w)?;
        if gate.2 != up.2 {
            return Err(Error::Unsupported(format!(
                "layer {i}: gate and up experts in different formats"
            )));
        }
        let record_bytes = (gate.3 + up.3 + down.3).div_ceil(16) * 16;
        let offsets = [0, gate.3, gate.3 + up.3];
        let fill = |bytes: &mut [u8]| -> Result<()> {
            for x in 0..e {
                let rec = &mut bytes[x * record_bytes..(x + 1) * record_bytes];
                for (p, &off) in [&gate, &up, &down].into_iter().zip(&offsets) {
                    p.0.read_at(p.1 + (x * p.3) as u64, &mut rec[off..off + p.3])
                        .map_err(werr)?;
                }
            }
            Ok(())
        };
        let mut records = self.b.alloc_host(e * record_bytes / 4)?;
        match self.b.host_bytes_mut(&mut records) {
            // Read straight into the pinned records.
            Some(bytes) => fill(bytes)?,
            // A backend without such memory (the CPU oracle): assemble, then store.
            None => {
                let mut v = vec![0u8; e * record_bytes];
                fill(&mut v)?;
                records = self.b.upload_bytes_host(&v)?;
            }
        }
        self.host_bytes += e * record_bytes;
        let m = |format, offset, rows, k| ExpertMatrix {
            format,
            offset,
            rows,
            k,
        };
        Ok(ExpertRecords {
            records,
            record_bytes,
            gate: m(gate.2, 0, w, h),
            up: m(up.2, offsets[1], w, h),
            down: m(down.2, offsets[2], h, w),
        })
    }

    fn moe(&mut self, i: usize, c: &Qwen4Config) -> Result<MoeWeights<B>> {
        let t = |s: &str| format!("blk.{i}.{s}");
        // Router rows, then the shared expert's gate as one more row.
        let router = {
            let (tr, mut r, n, k) = self.read(&t("ffn_gate_inp.weight"))?;
            let (ts, s, _, ks) = self.read(&t("ffn_gate_inp_shexp.weight"))?;
            if tr.ggml_type != ts.ggml_type || k != ks || n != c.experts {
                return Err(Error::Config(format!(
                    "layer {i}: router [{n}, {k}] and shared gate [{ks}] differ"
                )));
            }
            let (format, _) = format_of(tr, k)?;
            r.extend_from_slice(&s);
            Mat {
                buf: self.upload(&r)?,
                format,
                rows: n + 1,
                k,
                act: ActPolicy::A16Only,
            }
        };
        Ok(MoeWeights {
            router,
            experts: self.experts(i, c)?,
            shared_gate: self.matrix(&t("ffn_gate_shexp.weight"), None)?,
            shared_up: self.matrix(&t("ffn_up_shexp.weight"), None)?,
            shared_down: self.matrix(&t("ffn_down_shexp.weight"), None)?,
        })
    }

    fn ple(&mut self, i: usize, c: &Qwen4Config) -> Result<PleWeights<B>> {
        let t = |s: &str| format!("blk.{i}.{s}");
        let wide = c.wide();
        let width = c.ple.as_ref().map_or(4, |p| p.conv_width);
        Ok(PleWeights {
            key: self.matrix(&t("ple_key.weight"), None)?,
            value: self.matrix(&t("ple_value.weight"), None)?,
            norm_key: self.vector(&t("ple_norm_key.weight"), wide)?,
            norm_query: self.vector(&t("ple_norm_query.weight"), wide)?,
            norm_conv: self.vector(&t("ple_norm_conv.weight"), wide)?,
            conv: self.conv(&t("ple_conv1d.weight"), wide, width, |o| o)?,
        })
    }
}

fn expect_len(what: &str, got: usize, want: usize) -> Result<()> {
    if got != want {
        return Err(Error::Config(format!(
            "{what}: {got} values, config implies {want}"
        )));
    }
    Ok(())
}

/// The MTP head (`blk.{layers}`, `blk.{layers}.nextn.*`): one QSA layer fed by
/// `fc_e(norm(embed(x))) + fc_h(norm(h_s))` per residual stream, its own head mixer.
pub struct MtpWeights<B: Backend> {
    pub fc_e: Mat<B>,
    pub fc_h: Mat<B>,
    pub enorm: B::Buffer,
    /// Per stream `[streams * hidden]`.
    pub hnorm: B::Buffer,
    pub hc_attn: HcWeights<B>,
    pub attn: QsaWeights<B>,
    pub hc_ffn: HcWeights<B>,
    pub moe: MoeWeights<B>,
    pub hc_head: HcWeights<B>,
}

/// Loads the MTP head from `shards` (the model's, or a sidecar added to them), when present.
pub fn load_mtp<B: Backend>(
    b: &B,
    shards: &Shards,
    c: &Qwen4Config,
) -> Result<Option<MtpWeights<B>>> {
    let i = c.layers.len();
    let t = |s: &str| format!("blk.{i}.{s}");
    if shards.tensor(&t("nextn.fc_e.weight")).is_err() {
        return Ok(None);
    }
    let mut l = Loader {
        b,
        shards,
        device_bytes: 0,
        host_bytes: 0,
    };
    Ok(Some(MtpWeights {
        fc_e: l.matrix(&t("nextn.fc_e.weight"), None)?,
        fc_h: l.matrix(&t("nextn.fc_h.weight"), None)?,
        enorm: l.vector(&t("nextn.enorm.weight"), c.hidden)?,
        hnorm: l.vector(&t("nextn.hnorm.weight"), c.wide())?,
        hc_attn: l.hc(&t("hc_attn"), c, true)?,
        attn: l.qsa(i, c)?,
        hc_ffn: l.hc(&t("hc_ffn"), c, true)?,
        moe: l.moe(i, c)?,
        hc_head: l.hc(&t("nextn.hc_head"), c, false)?,
    }))
}

/// Loads every weight of the model in `shards`; `progress` gets a line per layer.
pub fn load<B: Backend>(
    b: &B,
    shards: &Shards,
    c: &Qwen4Config,
    mut progress: impl FnMut(&str),
) -> Result<Qwen4Weights<B>> {
    let mut l = Loader {
        b,
        shards,
        device_bytes: 0,
        host_bytes: 0,
    };
    // The embedding is read one row per token: system RAM.
    let embedding = {
        let (t, bytes, n, k) = l.read("token_embd.weight")?;
        let (format, _) = format_of(t, k)?;
        l.host_bytes += bytes.len();
        Mat {
            buf: b.upload_bytes_host(&bytes)?,
            format,
            rows: n,
            k,
            act: ActPolicy::A16Only,
        }
    };
    let output = l.matrix("output.weight", None)?;
    let hc_head = l.hc("output_hc", c, false)?;
    let mut layers = Vec::with_capacity(c.layers.len());
    for (i, kind) in c.layers.iter().enumerate() {
        progress(&format!("layer {}/{}", i + 1, c.layers.len()));
        let hc_attn = l.hc(&format!("blk.{i}.hc_attn"), c, true)?;
        let mixer = match kind {
            LayerKind::Gdn => Mixer::Gdn(l.gdn(i, c)?),
            LayerKind::Qsa => Mixer::Qsa(l.qsa(i, c)?),
        };
        let hc_ffn = l.hc(&format!("blk.{i}.hc_ffn"), c, true)?;
        let moe = l.moe(i, c)?;
        let ple = match &c.ple {
            Some(p) if p.layer == i => Some(l.ple(i, c)?),
            _ => None,
        };
        layers.push(Layer {
            hc_attn,
            mixer,
            hc_ffn,
            moe,
            ple,
        });
    }
    let ple_table = match &c.ple {
        Some(p) => Some(PleTable::open(shards, p)?),
        None => None,
    };
    Ok(Qwen4Weights {
        embedding,
        output,
        hc_head,
        layers,
        ple_table,
        device_bytes: l.device_bytes,
        host_bytes: l.host_bytes,
    })
}
