//! Qwen3-ASR weights straight from the llama.cpp GGUF pair (text model + `mmproj` audio tower).
//! Matrices keep their GGML blocks; F16 tensors (the convolutions) are widened to F32.

use std::path::Path;

use velox_core::format::{bf16_to_f32, f16_to_f32};
use velox_core::ggml::GgmlType;
use velox_core::{ActPolicy, Backend, Error, Result, WeightFormat, WeightView};
use velox_weights::gguf::{Gguf, GgufTensor};

use super::config::AsrConfig;

fn werr(e: velox_weights::Error) -> Error {
    Error::Config(e.to_string())
}

/// A stored matrix `[rows, k]`.
pub struct Mat<B: Backend> {
    pub buf: B::Buffer,
    pub format: WeightFormat,
    pub rows: usize,
    pub k: usize,
}

impl<B: Backend> Mat<B> {
    pub fn view(&self) -> WeightView<'_, B::Buffer> {
        // A16Only keeps activations exact; reduced-precision activations are a later, measured
        // opt-in per backend.
        WeightView::full(&self.buf, self.format, self.rows, self.k).with_act(ActPolicy::A16Only)
    }
}

/// A matrix with its bias.
pub struct Linear<B: Backend> {
    pub w: Mat<B>,
    pub bias: Option<B::Buffer>,
}

pub struct EncLayer<B: Backend> {
    pub ln1: (B::Buffer, B::Buffer),
    pub ln2: (B::Buffer, B::Buffer),
    pub q: Linear<B>,
    pub k: Linear<B>,
    pub v: Linear<B>,
    pub out: Linear<B>,
    pub fc1: Linear<B>,
    pub fc2: Linear<B>,
}

pub struct EncWeights<B: Backend> {
    /// Conv kernels `[480, cin * 9]` (F32), row index `cin * 9 + ky * 3 + kx`, and biases (host).
    pub conv: [(Mat<B>, Vec<f32>); 3],
    pub conv_out: Mat<B>,
    pub layers: Vec<EncLayer<B>>,
    pub ln_post: (B::Buffer, B::Buffer),
    pub proj1: Linear<B>,
    pub proj2: Linear<B>,
}

pub struct DecLayer<B: Backend> {
    pub attn_norm: B::Buffer,
    pub ffn_norm: B::Buffer,
    pub q: Mat<B>,
    pub k: Mat<B>,
    pub v: Mat<B>,
    pub o: Mat<B>,
    /// Per-head norm weights tiled over the heads (`[heads * head_dim]`, as `rms_norm_groups`
    /// takes them).
    pub q_norm: B::Buffer,
    pub k_norm: B::Buffer,
    pub gate: Mat<B>,
    pub up: Mat<B>,
    pub down: Mat<B>,
}

pub struct DecWeights<B: Backend> {
    pub embed: Mat<B>,
    pub layers: Vec<DecLayer<B>>,
    pub norm: B::Buffer,
    pub lm_head: Mat<B>,
}

pub struct Files {
    pub text: Gguf,
    pub mmproj: Gguf,
}

impl Files {
    pub fn open(text: &Path, mmproj: &Path) -> Result<Self> {
        Ok(Files {
            text: Gguf::open(text).map_err(werr)?,
            mmproj: Gguf::open(mmproj).map_err(werr)?,
        })
    }
}

/// `(k, n)`: the contiguous extent (all dims but the last) and the row count.
fn extents(t: &GgufTensor) -> (usize, usize) {
    match t.dims.split_last() {
        Some((&n, rest)) if !rest.is_empty() => (rest.iter().product::<u64>() as usize, n as usize),
        Some((&k, _)) => (k as usize, 1),
        None => (0, 0),
    }
}

struct Loader<'a, B: Backend> {
    b: &'a B,
    g: &'a Gguf,
    bytes: u64,
}

impl<B: Backend> Loader<'_, B> {
    fn tensor(&self, name: &str) -> Result<&GgufTensor> {
        self.g
            .tensor(name)
            .ok_or_else(|| Error::Config(format!("{}: no tensor {name}", self.g.path.display())))
    }

    fn raw(&self, t: &GgufTensor, len: usize) -> Result<Vec<u8>> {
        let mut bytes = vec![0u8; len];
        self.g.read_at(t.offset, &mut bytes).map_err(werr)?;
        Ok(bytes)
    }

    /// Any float tensor as f32 values, flat.
    fn values(&self, name: &str) -> Result<Vec<f32>> {
        let t = self.tensor(name)?;
        let (k, n) = extents(t);
        let count = k * n;
        Ok(match t.ggml_type {
            0 => self
                .raw(t, 4 * count)?
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
            1 => self
                .raw(t, 2 * count)?
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f16_to_f32(u16::from_le_bytes(*c)))
                .collect(),
            30 => self
                .raw(t, 2 * count)?
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| bf16_to_f32(u16::from_le_bytes(*c)))
                .collect(),
            other => {
                return Err(Error::Unsupported(format!(
                    "{name}: ggml type {other} as values"
                )))
            }
        })
    }

    fn vector(&mut self, name: &str, len: usize) -> Result<B::Buffer> {
        let v = self.values(name)?;
        if v.len() != len {
            return Err(Error::Config(format!(
                "{name}: {} values, expected {len}",
                v.len()
            )));
        }
        self.bytes += 4 * len as u64;
        self.b.upload(&v)
    }

    /// A matrix `[rows, k]`; GGML-quantized tensors stay in their blocks, F16 / BF16 widen to F32.
    fn matrix(&mut self, name: &str, rows: usize, k: usize) -> Result<Mat<B>> {
        let t = self.tensor(name)?;
        let (tk, tn) = extents(t);
        if (tk, tn) != (k, rows) {
            return Err(Error::Config(format!(
                "{name}: [{tn}, {tk}], expected [{rows}, {k}]"
            )));
        }
        let (buf, format) = match t.ggml_type {
            0 | 1 | 30 => {
                let v = self.values(name)?;
                self.bytes += 4 * v.len() as u64;
                (self.b.upload(&v)?, WeightFormat::F32)
            }
            id => {
                let g = GgmlType::from_gguf_id(id).ok_or_else(|| {
                    Error::Unsupported(format!("{name}: ggml type {id} is not supported"))
                })?;
                let f = WeightFormat::Ggml(g);
                f.check_shape(rows, k)?;
                let bytes = self.raw(t, rows * g.row_bytes(k))?;
                self.bytes += bytes.len() as u64;
                (self.b.upload_bytes(&bytes)?, f)
            }
        };
        Ok(Mat {
            buf,
            format,
            rows,
            k,
        })
    }

    fn linear(&mut self, w: &str, bias: Option<&str>, rows: usize, k: usize) -> Result<Linear<B>> {
        Ok(Linear {
            w: self.matrix(w, rows, k)?,
            bias: match bias {
                Some(b) => Some(self.vector(b, rows)?),
                None => None,
            },
        })
    }

    /// A `[dim]` norm weight repeated `heads` times.
    fn tiled(&mut self, name: &str, dim: usize, heads: usize) -> Result<B::Buffer> {
        let v = self.values(name)?;
        if v.len() != dim {
            return Err(Error::Config(format!(
                "{name}: {} values, expected {dim}",
                v.len()
            )));
        }
        let t: Vec<f32> = (0..heads).flat_map(|_| v.iter().copied()).collect();
        self.bytes += 4 * t.len() as u64;
        self.b.upload(&t)
    }
}

/// Loads the audio tower; returns the weights and their device bytes.
pub fn load_encoder<B: Backend>(b: &B, f: &Files, c: &AsrConfig) -> Result<(EncWeights<B>, u64)> {
    let mut l = Loader {
        b,
        g: &f.mmproj,
        bytes: 0,
    };
    let e = &c.enc;
    let conv = |l: &mut Loader<'_, B>, i: usize, cin: usize| -> Result<(Mat<B>, Vec<f32>)> {
        let w = l.matrix(&format!("a.conv2d.{i}.weight"), e.conv_ch, cin * 9)?;
        let bias = l.values(&format!("a.conv2d.{i}.bias"))?;
        if bias.len() != e.conv_ch {
            return Err(Error::Config(format!(
                "a.conv2d.{i}.bias: {} values",
                bias.len()
            )));
        }
        Ok((w, bias))
    };
    let conv = [
        conv(&mut l, 1, 1)?,
        conv(&mut l, 2, e.conv_ch)?,
        conv(&mut l, 3, e.conv_ch)?,
    ];
    let conv_out = l.matrix("a.conv_out.weight", e.d, e.conv_ch * e.conv_freq())?;
    let (d, ff) = (e.d, e.ffn);
    let mut layers = Vec::with_capacity(e.layers);
    for n in 0..e.layers {
        let p = |s: &str| format!("a.blk.{n}.{s}");
        let lin = |l: &mut Loader<'_, B>, s: &str, rows: usize, k: usize| {
            l.linear(
                &p(&format!("{s}.weight")),
                Some(&p(&format!("{s}.bias"))),
                rows,
                k,
            )
        };
        layers.push(EncLayer {
            ln1: (l.vector(&p("ln1.weight"), d)?, l.vector(&p("ln1.bias"), d)?),
            ln2: (l.vector(&p("ln2.weight"), d)?, l.vector(&p("ln2.bias"), d)?),
            q: lin(&mut l, "attn_q", d, d)?,
            k: lin(&mut l, "attn_k", d, d)?,
            v: lin(&mut l, "attn_v", d, d)?,
            out: lin(&mut l, "attn_out", d, d)?,
            fc1: lin(&mut l, "ffn_up", ff, d)?,
            fc2: lin(&mut l, "ffn_down", d, ff)?,
        });
    }
    let ln_post = (
        l.vector("a.post_ln.weight", d)?,
        l.vector("a.post_ln.bias", d)?,
    );
    let proj1 = l.linear("mm.a.mlp.1.weight", Some("mm.a.mlp.1.bias"), d, d)?;
    let proj2 = l.linear(
        "mm.a.mlp.2.weight",
        Some("mm.a.mlp.2.bias"),
        c.dec.hidden,
        d,
    )?;
    let bytes = l.bytes;
    Ok((
        EncWeights {
            conv,
            conv_out,
            layers,
            ln_post,
            proj1,
            proj2,
        },
        bytes,
    ))
}

/// Loads the text decoder; returns the weights and their device bytes.
pub fn load_decoder<B: Backend>(b: &B, f: &Files, c: &AsrConfig) -> Result<(DecWeights<B>, u64)> {
    let mut l = Loader {
        b,
        g: &f.text,
        bytes: 0,
    };
    let t = &c.dec;
    let (h, hd) = (t.hidden, t.head_dim);
    let (qd, kvd) = (t.heads * hd, t.kv_heads * hd);
    let embed = l.matrix("token_embd.weight", t.vocab, h)?;
    let mut layers = Vec::with_capacity(t.layers);
    for n in 0..t.layers {
        let p = |s: &str| format!("blk.{n}.{s}.weight");
        layers.push(DecLayer {
            attn_norm: l.vector(&p("attn_norm"), h)?,
            ffn_norm: l.vector(&p("ffn_norm"), h)?,
            q: l.matrix(&p("attn_q"), qd, h)?,
            k: l.matrix(&p("attn_k"), kvd, h)?,
            v: l.matrix(&p("attn_v"), kvd, h)?,
            o: l.matrix(&p("attn_output"), h, qd)?,
            q_norm: l.tiled(&p("attn_q_norm"), hd, t.heads)?,
            k_norm: l.tiled(&p("attn_k_norm"), hd, t.kv_heads)?,
            gate: l.matrix(&p("ffn_gate"), t.ffn, h)?,
            up: l.matrix(&p("ffn_up"), t.ffn, h)?,
            down: l.matrix(&p("ffn_down"), h, t.ffn)?,
        });
    }
    let norm = l.vector("output_norm.weight", h)?;
    let lm_head = match f.text.tensor("output.weight") {
        Some(_) => l.matrix("output.weight", t.vocab, h)?,
        None => l.matrix("token_embd.weight", t.vocab, h)?,
    };
    let bytes = l.bytes;
    Ok((
        DecWeights {
            embed,
            layers,
            norm,
            lm_head,
        },
        bytes,
    ))
}
