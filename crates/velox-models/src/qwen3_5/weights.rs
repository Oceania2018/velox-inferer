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
        /// qkv and z as one view when they are adjacent in one parent (one GEMM, then a split).
        qkvz: Option<WDesc>,
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
        /// query, gate, key and value as one view when they tile one parent row range.
        fused: Option<WDesc>,
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
    /// gate and up as one view when adjacent in one parent.
    pub mlp_gate_up: Option<WDesc>,
}

pub struct TextWeights<B: Backend> {
    parents: Vec<B::Buffer>,
    params: Vec<B::Buffer>,
    pub embed: WDesc,
    pub head: WDesc,
    pub final_norm: usize,
    pub layers: Vec<Layer>,
    /// The MTP predictor (`mtp/`), when the artifact has one.
    pub mtp: Option<MtpWeights>,
    /// The draft vocabulary head (`proposal/`): a row subset of the output head for drafting.
    pub proposal: Option<Proposal>,
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

    /// Uploads every `text/` parameter the model uses, and the MTP predictor when the artifact has
    /// one. `progress` receives a short line per layer.
    pub fn load(
        b: &B,
        art: &Artifact,
        cfg: &TextConfig,
        progress: impl FnMut(&str),
    ) -> Result<Self> {
        Self::load_with(b, art, cfg, true, progress)
    }

    /// [`Self::load`], loading the MTP predictor only when `mtp` is set.
    pub fn load_with(
        b: &B,
        art: &Artifact,
        cfg: &TextConfig,
        mtp: bool,
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
                    let qkvz = fuse(&[&qkv, &z]);
                    Mixer::Gdn {
                        qkv,
                        z,
                        qkvz,
                        ab,
                        out,
                        conv: l.param(&p("gdn/convolution"), cfg.conv_width * cfg.gdn_qkv())?,
                        a_log: l.param(&p("gdn/a_log"), cfg.gdn_v_heads)?,
                        dt_bias: l.param(&p("gdn/dt_bias"), cfg.gdn_v_heads)?,
                        norm: l.param(&p("gdn/norm"), cfg.gdn_v_dim)?,
                    }
                }
                LayerKind::FullAttention => load_attention(&mut l, cfg, &p)?,
            };
            let layer = finish_layer(&mut l, cfg, &p, input_norm, mixer)?;
            layers.push(layer);
            progress(&format!(
                "layer {}/{} ({:.1} GB)",
                i + 1,
                cfg.layers.len(),
                l.bytes as f64 / 1e9
            ));
        }
        let mtp = if mtp { load_mtp(&mut l, cfg)? } else { None };
        let proposal = load_proposal(&mut l, art, cfg)?;
        Ok(TextWeights {
            parents: l.parents,
            params: l.params,
            embed,
            head,
            final_norm,
            layers,
            mtp,
            proposal,
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
    pub(crate) art: &'a Artifact,
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
    pub(crate) fn binding(&self, name: &str) -> Result<(String, Option<(u64, u64)>)> {
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

/// MTP predictor weights (`Qwen3_5MTP`): stem norms and input projection, one attention block
/// with the target's geometry, and a final norm. Embedding and output head are the target's.
pub struct MtpWeights {
    pub input_projection: WDesc,
    pub embedding_norm: usize,
    pub hidden_norm: usize,
    pub final_norm: usize,
    pub layer: Layer,
}

/// Draft vocabulary head: `head` rows are the tokens `token_ids` (a subset of the vocabulary).
pub struct Proposal {
    pub head: WDesc,
    pub token_ids: Vec<u32>,
}

fn load_attention<B: Backend>(
    l: &mut Loader<'_, B>,
    cfg: &TextConfig,
    p: &dyn Fn(&str) -> String,
) -> Result<Mixer> {
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
    let fused = fuse(&[&query, &gate, &key, &value]);
    Ok(Mixer::Attention {
        query,
        gate,
        key,
        value,
        fused,
        output: l.matrix(&p("attention/output"))?,
        q_norm: l.param(&p("attention/query_norm"), cfg.head_dim)?,
        k_norm: l.param(&p("attention/key_norm"), cfg.head_dim)?,
    })
}

fn finish_layer<B: Backend>(
    l: &mut Loader<'_, B>,
    cfg: &TextConfig,
    p: &dyn Fn(&str) -> String,
    input_norm: usize,
    mixer: Mixer,
) -> Result<Layer> {
    let layer = Layer {
        input_norm,
        post_norm: l.param(&p("post_attention_norm"), cfg.hidden)?,
        mixer,
        mlp_gate: l.matrix(&p("mlp/gate"))?,
        mlp_up: l.matrix(&p("mlp/up"))?,
        mlp_down: l.matrix(&p("mlp/down"))?,
        mlp_gate_up: None,
    };
    let layer = Layer {
        mlp_gate_up: fuse(&[&layer.mlp_gate, &layer.mlp_up]),
        ..layer
    };
    expect(&p("mlp gate"), layer.mlp_gate.rows, cfg.intermediate)?;
    Ok(layer)
}

fn has_binding(art: &Artifact, name: &str) -> bool {
    art.directory["bindings"].get(name).is_some()
}

fn load_mtp<B: Backend>(l: &mut Loader<'_, B>, cfg: &TextConfig) -> Result<Option<MtpWeights>> {
    if !has_binding(l.art, "mtp/input_projection") {
        return Ok(None);
    }
    let p = |s: &str| format!("mtp/layers/0/{s}");
    let input_projection = l.matrix("mtp/input_projection")?;
    if input_projection.rows != cfg.hidden || input_projection.k != 2 * cfg.hidden {
        return Err(Error::Config(
            "mtp/input_projection is not [hidden, 2 * hidden]".into(),
        ));
    }
    let embedding_norm = l.param("mtp/embedding_norm", cfg.hidden)?;
    let hidden_norm = l.param("mtp/hidden_norm", cfg.hidden)?;
    let final_norm = l.param("mtp/final_norm", cfg.hidden)?;
    let input_norm = l.param(&p("input_norm"), cfg.hidden)?;
    let mixer = load_attention(l, cfg, &p)?;
    let layer = finish_layer(l, cfg, &p, input_norm, mixer)?;
    Ok(Some(MtpWeights {
        input_projection,
        embedding_norm,
        hidden_norm,
        final_norm,
        layer,
    }))
}

fn load_proposal<B: Backend>(
    l: &mut Loader<'_, B>,
    art: &Artifact,
    cfg: &TextConfig,
) -> Result<Option<Proposal>> {
    if !has_binding(art, "proposal/head") {
        return Ok(None);
    }
    let head = l.matrix("proposal/head")?;
    let (id, _) = l.binding("proposal/token_ids")?;
    let raw = art.read_object(&id).map_err(werr)?;
    let token_ids: Vec<u32> = raw
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u32)
        .collect();
    if head.k != cfg.hidden
        || token_ids.len() != head.rows
        || token_ids.iter().any(|&t| t as usize >= cfg.vocab)
    {
        return Err(Error::Config("proposal head and token map disagree".into()));
    }
    Ok(Some(Proposal { head, token_ids }))
}

/// One view covering `parts` when they are row ranges of one parent that tile a contiguous range
/// (any order). Each part's columns in the fused output start at `part.row_begin - fused.row_begin`.
pub fn fuse(parts: &[&WDesc]) -> Option<WDesc> {
    let first = parts.first()?;
    let mut sorted: Vec<&WDesc> = parts.to_vec();
    sorted.sort_by_key(|d| d.row_begin);
    let mut end = sorted[0].row_begin;
    let mut act = first.act;
    for d in &sorted {
        if d.parent != first.parent || d.row_begin != end {
            return None;
        }
        end += d.rows;
        act = act.meet(d.act);
    }
    Some(WDesc {
        row_begin: sorted[0].row_begin,
        rows: end - sorted[0].row_begin,
        act,
        ..**first
    })
}
