//! Binds `text/...` logical parameters from a `.vi` / `.ninfer` artifact onto a backend.
//!
//! Large matrices stay in their stored encoding: each parent object is uploaded once and logical
//! parameters become row views into it. Small parameters (norms, GDN controls, convolution) are
//! decoded to f32 on the host.

use std::collections::HashMap;

use serde_json::Value;
use velox_core::backend::{ActPolicy, Backend, WeightView};
use velox_core::format::{decode_rows, WeightFormat};
use velox_core::{Error, Result};
use velox_weights::{Artifact, ObjectKind};

use super::config::{LayerKind, TextConfig};

/// A row view into an uploaded parent object.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WDesc {
    pub parent: usize,
    pub format: WeightFormat,
    pub n: usize,
    pub k: usize,
    pub row_begin: usize,
    pub rows: usize,
    /// Activation rounding the artifact permits for this projection.
    pub act: ActPolicy,
}

pub enum Mixer {
    Gdn {
        /// q | k | v projection rows, contiguous in one parent.
        qkv: WDesc,
        z: WDesc,
        /// a | b projection rows, contiguous in one parent.
        ab: WDesc,
        out: WDesc,
        conv: usize,
        a_log: usize,
        dt_bias: usize,
        norm: usize,
    },
    Attention {
        query: WDesc,
        gate: WDesc,
        key: WDesc,
        value: WDesc,
        output: WDesc,
        q_norm: usize,
        k_norm: usize,
    },
}

pub struct Layer {
    pub input_norm: usize,
    pub post_norm: usize,
    pub mixer: Mixer,
    pub mlp_gate: WDesc,
    pub mlp_up: WDesc,
    pub mlp_down: WDesc,
}

pub struct TextWeights<B: Backend> {
    parents: Vec<B::Buffer>,
    params: Vec<B::Buffer>,
    pub embed: WDesc,
    pub head: WDesc,
    pub final_norm: usize,
    pub layers: Vec<Layer>,
    /// Bytes uploaded for weights (parents + decoded params).
    pub bytes: u64,
}

impl<B: Backend> TextWeights<B> {
    pub fn view(&self, d: &WDesc) -> WeightView<'_, B::Buffer> {
        WeightView {
            buf: &self.parents[d.parent],
            format: d.format,
            parent_rows: d.n,
            k: d.k,
            row_begin: d.row_begin,
            rows: d.rows,
            act: d.act,
        }
    }

    pub fn param(&self, i: usize) -> &B::Buffer {
        &self.params[i]
    }

    /// Uploads every `text/` parameter the model uses. `progress` receives a short line per layer.
    pub fn load(
        b: &B,
        art: &Artifact,
        cfg: &TextConfig,
        mut progress: impl FnMut(&str),
    ) -> Result<Self> {
        let mut l = Loader::new(b, art);
        let embed = l.matrix("text/token_embedding")?;
        let head = l.matrix("text/output_head")?;
        let final_norm = l.param("text/final_norm", cfg.hidden)?;
        if embed.k != cfg.hidden || head.k != cfg.hidden || head.rows != cfg.vocab {
            return Err(Error::Config(
                "embedding/head shapes do not match the config".into(),
            ));
        }
        let mut layers = Vec::with_capacity(cfg.layers.len());
        for (i, kind) in cfg.layers.iter().enumerate() {
            let p = |s: &str| format!("text/layers/{i}/{s}");
            let input_norm = l.param(&p("input_norm"), cfg.hidden)?;
            let mixer = match kind {
                LayerKind::Gdn => {
                    let qkv = l.rows_of(&[&p("gdn/query"), &p("gdn/key"), &p("gdn/value")])?;
                    let ab = l.rows_of(&[&p("gdn/a_projection"), &p("gdn/b_projection")])?;
                    let z = l.matrix(&p("gdn/z"))?;
                    let out = l.matrix(&p("gdn/output"))?;
                    expect(&p("gdn qkv"), qkv.rows, cfg.gdn_qkv())?;
                    expect(&p("gdn ab"), ab.rows, 2 * cfg.gdn_v_heads)?;
                    expect(&p("gdn z"), z.rows, cfg.gdn_v())?;
                    Mixer::Gdn {
                        qkv,
                        z,
                        ab,
                        out,
                        conv: l.param(&p("gdn/convolution"), cfg.conv_width * cfg.gdn_qkv())?,
                        a_log: l.param(&p("gdn/a_log"), cfg.gdn_v_heads)?,
                        dt_bias: l.param(&p("gdn/dt_bias"), cfg.gdn_v_heads)?,
                        norm: l.param(&p("gdn/norm"), cfg.gdn_v_dim)?,
                    }
                }
                LayerKind::FullAttention => {
                    let (query, gate) = (
                        l.matrix(&p("attention/query"))?,
                        l.matrix(&p("attention/gate"))?,
                    );
                    let (key, value) = (
                        l.matrix(&p("attention/key"))?,
                        l.matrix(&p("attention/value"))?,
                    );
                    expect(&p("attention query"), query.rows, cfg.q_dim())?;
                    expect(&p("attention gate"), gate.rows, cfg.q_dim())?;
                    expect(&p("attention key"), key.rows, cfg.kv_dim())?;
                    expect(&p("attention value"), value.rows, cfg.kv_dim())?;
                    Mixer::Attention {
                        query,
                        gate,
                        key,
                        value,
                        output: l.matrix(&p("attention/output"))?,
                        q_norm: l.param(&p("attention/query_norm"), cfg.head_dim)?,
                        k_norm: l.param(&p("attention/key_norm"), cfg.head_dim)?,
                    }
                }
            };
            let layer = Layer {
                input_norm,
                post_norm: l.param(&p("post_attention_norm"), cfg.hidden)?,
                mixer,
                mlp_gate: l.matrix(&p("mlp/gate"))?,
                mlp_up: l.matrix(&p("mlp/up"))?,
                mlp_down: l.matrix(&p("mlp/down"))?,
            };
            expect(&p("mlp gate"), layer.mlp_gate.rows, cfg.intermediate)?;
            layers.push(layer);
            progress(&format!(
                "layer {}/{} ({:.1} GB)",
                i + 1,
                cfg.layers.len(),
                l.bytes as f64 / 1e9
            ));
        }
        Ok(TextWeights {
            parents: l.parents,
            params: l.params,
            embed,
            head,
            final_norm,
            layers,
            bytes: l.bytes,
        })
    }
}

fn expect(what: &str, got: usize, want: usize) -> Result<()> {
    if got != want {
        return Err(Error::Config(format!(
            "{what}: {got} rows, config implies {want}"
        )));
    }
    Ok(())
}

pub(crate) struct Loader<'a, B: Backend> {
    b: &'a B,
    art: &'a Artifact,
    parents: Vec<B::Buffer>,
    index: HashMap<String, (usize, WeightFormat, usize, usize)>,
    /// Activation policy per logical parameter, from the artifact's `uses`.
    policies: HashMap<String, ActPolicy>,
    params: Vec<B::Buffer>,
    bytes: u64,
}

fn werr(e: velox_weights::Error) -> Error {
    Error::Config(e.to_string())
}

/// Activation policy per parameter from the artifact's `uses` (`activation_policy` plus the
/// `activation_input_divisor` auxiliary). A parameter with several uses gets the strictest; anything
/// unreadable falls back to `A16Only`, which is always correct.
fn read_policies(art: &Artifact) -> HashMap<String, ActPolicy> {
    let mut out: HashMap<String, ActPolicy> = HashMap::new();
    let uses = art.directory["uses"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    for u in uses {
        let Some(param) = u["parameter"].as_str() else {
            continue;
        };
        let divisor = u["auxiliaries"]["activation_input_divisor"]["object"]
            .as_str()
            .and_then(|id| art.read_object(id).ok())
            .and_then(|b| Some(f32::from_le_bytes(b.get(..4)?.try_into().ok()?)))
            .filter(|d| d.is_finite() && *d > 0.0);
        let p = match (u["activation_policy"].as_str(), divisor) {
            (Some("AllowA4"), Some(divisor)) => ActPolicy::AllowA4 { divisor },
            (Some("AllowA4" | "AllowA8"), _) => ActPolicy::AllowA8,
            _ => ActPolicy::A16Only,
        };
        out.entry(param.to_string())
            .and_modify(|q| *q = q.meet(p))
            .or_insert(p);
    }
    out
}

impl<'a, B: Backend> Loader<'a, B> {
    /// (object id, element range) of a binding with one part or a whole object.
    fn binding(&self, name: &str) -> Result<(String, Option<(u64, u64)>)> {
        let b = self.art.directory["bindings"]
            .get(name)
            .ok_or_else(|| Error::Config(format!("artifact has no binding '{name}'")))?;
        if let Some(id) = b.get("object").and_then(Value::as_str) {
            return Ok((id.to_string(), None));
        }
        let parts = b["parts"]
            .as_array()
            .ok_or_else(|| Error::Config(format!("binding '{name}' has no parts")))?;
        // Consecutive parts of one parent merge into one range.
        let mut id = String::new();
        let mut range: Option<(u64, u64)> = None;
        for p in parts {
            let pid = p["object"].as_str().unwrap_or_default();
            let (a, e) = (
                p["range"][0].as_u64().unwrap_or(0),
                p["range"][1].as_u64().unwrap_or(0),
            );
            match range {
                None => {
                    id = pid.to_string();
                    range = Some((a, e));
                }
                Some((ra, re)) if pid == id && a == re => range = Some((ra, e)),
                _ => {
                    return Err(Error::Unsupported(format!(
                        "binding '{name}' gathers non-contiguous parts"
                    )))
                }
            }
        }
        Ok((id, range))
    }

    /// Uploads parent `id` once; returns (slot, format, n, k).
    fn parent(&mut self, id: &str) -> Result<(usize, WeightFormat, usize, usize)> {
        if let Some(&p) = self.index.get(id) {
            return Ok(p);
        }
        let obj = self
            .art
            .object(id)
            .ok_or_else(|| Error::Config(format!("missing object {id}")))?;
        let (fmt, n, k) = match &obj.kind {
            ObjectKind::Tensor {
                shape,
                format,
                layout,
            } if shape.len() == 2 => (
                WeightFormat::from_names(format, layout)?,
                shape[0] as usize,
                shape[1] as usize,
            ),
            other => {
                return Err(Error::Config(format!(
                    "object {id} is not a matrix: {other:?}"
                )))
            }
        };
        let bytes = self.art.read_object(id).map_err(werr)?;
        if bytes.len() != fmt.payload_bytes(n, k) {
            return Err(Error::Config(format!(
                "object {id}: {} bytes, layout implies {}",
                bytes.len(),
                fmt.payload_bytes(n, k)
            )));
        }
        self.bytes += bytes.len() as u64;
        self.parents.push(self.b.upload_bytes(&bytes)?);
        let entry = (self.parents.len() - 1, fmt, n, k);
        self.index.insert(id.to_string(), entry);
        Ok(entry)
    }

    pub(crate) fn matrix(&mut self, name: &str) -> Result<WDesc> {
        let (id, range) = self.binding(name)?;
        let (parent, format, n, k) = self.parent(&id)?;
        let (row_begin, rows) = match range {
            None => (0, n),
            Some((a, e)) => {
                let k64 = k as u64;
                if a % k64 != 0 || e % k64 != 0 {
                    return Err(Error::Unsupported(format!(
                        "binding '{name}' range [{a},{e}) is not whole rows of {id}"
                    )));
                }
                ((a / k64) as usize, ((e - a) / k64) as usize)
            }
        };
        Ok(WDesc {
            parent,
            format,
            n,
            k,
            row_begin,
            rows,
            act: self.policy(name),
        })
    }

    /// Concatenation of row views that are adjacent in one parent, as one view.
    pub(crate) fn rows_of(&mut self, names: &[&str]) -> Result<WDesc> {
        let mut acc: Option<WDesc> = None;
        for name in names {
            let d = self.matrix(name)?;
            acc = Some(match acc {
                None => d,
                Some(a) if a.parent == d.parent && a.row_begin + a.rows == d.row_begin => WDesc {
                    rows: a.rows + d.rows,
                    act: a.act.meet(d.act),
                    ..a
                },
                Some(_) => {
                    return Err(Error::Unsupported(format!(
                        "{names:?} are not adjacent rows of one parent"
                    )))
                }
            });
        }
        acc.ok_or_else(|| Error::Config("empty row list".into()))
    }

    /// Decodes a small direct-format parameter to f32 and uploads it.
    /// Decodes a small direct-format parameter to f32 on the host.
    pub(crate) fn host_param(&mut self, name: &str, elements: usize) -> Result<Vec<f32>> {
        let (id, range) = self.binding(name)?;
        let obj = self
            .art
            .object(&id)
            .ok_or_else(|| Error::Config(format!("missing object {id}")))?;
        let (fmt, total) = match &obj.kind {
            ObjectKind::Tensor {
                shape,
                format,
                layout,
            } => (
                WeightFormat::from_names(format, layout)?,
                shape.iter().product::<u64>() as usize,
            ),
            _ => return Err(Error::Config(format!("{name} is not a tensor"))),
        };
        if !matches!(fmt, WeightFormat::F32 | WeightFormat::Bf16) {
            return Err(Error::Unsupported(format!(
                "{name}: small parameter in {}",
                fmt.name()
            )));
        }
        let bytes = self.art.read_object(&id).map_err(werr)?;
        let all = decode_rows(fmt, &bytes, 1, total, 0, 1)?;
        let values = match range {
            None => all,
            Some((a, e)) => all[a as usize..e as usize].to_vec(),
        };
        if values.len() != elements {
            return Err(Error::Config(format!(
                "{name}: {} elements, config implies {elements}",
                values.len()
            )));
        }
        Ok(values)
    }

    /// Decodes a small parameter and uploads it as f32; returns its slot.
    pub(crate) fn param(&mut self, name: &str, elements: usize) -> Result<usize> {
        let values = self.host_param(name, elements)?;
        self.bytes += 4 * values.len() as u64;
        self.params.push(self.b.upload(&values)?);
        Ok(self.params.len() - 1)
    }

    pub(crate) fn new(b: &'a B, art: &'a Artifact) -> Self {
        Loader {
            b,
            art,
            parents: Vec::new(),
            index: HashMap::new(),
            policies: read_policies(art),
            params: Vec::new(),
            bytes: 0,
        }
    }

    /// The activation policy of `name`: the meet over its uses; `A16Only` when it has none.
    fn policy(&self, name: &str) -> ActPolicy {
        self.policies.get(name).copied().unwrap_or_default()
    }

    /// (parent buffers, parameter buffers, uploaded bytes).
    pub(crate) fn finish(self) -> (Vec<B::Buffer>, Vec<B::Buffer>, u64) {
        (self.parents, self.params, self.bytes)
    }
}
