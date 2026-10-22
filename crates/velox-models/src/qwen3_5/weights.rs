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

use super::config::{FfnConfig, LayerKind, MoeConfig, TextConfig};

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

/// A projection's output rows as runs of adjacent rows of one parent each, in output order.
/// Artifacts usually store a projection as one run; mixed-precision ones need several (the IQ3_XXS
/// Qwen3.8-27B artifact keeps GDN q|k in a Q4 parent and v|z in a Q5 parent).
#[derive(Clone, Debug, PartialEq)]
pub struct Runs(pub Vec<WDesc>);

impl Runs {
    pub fn rows(&self) -> usize {
        self.0.iter().map(|d| d.rows).sum()
    }

    /// The projection as one view, when it is a single run.
    pub fn single(&self) -> Option<&WDesc> {
        match self.0.as_slice() {
            [d] => Some(d),
            _ => None,
        }
    }
}

pub enum Mixer {
    Gdn {
        /// q | k | v projection rows.
        qkv: Runs,
        z: WDesc,
        /// qkv and z as one view when they are adjacent in one parent (one GEMM, then a split).
        qkvz: Option<WDesc>,
        /// a | b projection rows.
        ab: Runs,
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

/// A SwiGLU FFN `down(silu(gate(h)) * up(h))`: the Dense block FFN, or a MoE block's shared expert.
pub struct DenseFfn {
    pub gate: WDesc,
    pub up: WDesc,
    pub down: WDesc,
    /// gate and up as one view when adjacent in one parent.
    pub gate_up: Option<WDesc>,
}

/// `experts` equally shaped matrices in one parent: expert `e` is `first` moved down by `e * stride`
/// rows. Banks keep the stored encoding, like every other view.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExpertBank {
    pub first: WDesc,
    pub experts: usize,
    pub stride: usize,
}

impl ExpertBank {
    /// The bank formed by per-expert views (expert order); they must share a parent, format and
    /// shape and sit at a constant row stride.
    pub fn from_experts(what: &str, views: &[WDesc]) -> Result<Self> {
        let (first, rest) = views
            .split_first()
            .ok_or_else(|| Error::Config(format!("{what}: no experts")))?;
        let stride = match rest.first() {
            None => first.rows,
            Some(d) => d.row_begin.saturating_sub(first.row_begin),
        };
        for (e, d) in views.iter().enumerate() {
            let at = e
                .checked_mul(stride)
                .and_then(|o| o.checked_add(first.row_begin));
            let regular = d.parent == first.parent
                && d.format == first.format
                && (d.n, d.k, d.rows) == (first.n, first.k, first.rows)
                && at == Some(d.row_begin);
            if !regular {
                return Err(Error::Unsupported(format!(
                    "{what}: expert {e} is not at a constant row stride of one parent"
                )));
            }
        }
        if stride < first.rows {
            return Err(Error::Config(format!("{what}: experts overlap")));
        }
        Ok(ExpertBank {
            first: WDesc {
                act: views.iter().fold(first.act, |a, d| a.meet(d.act)),
                ..*first
            },
            experts: views.len(),
            stride,
        })
    }

    pub fn expert(&self, e: usize) -> WDesc {
        WDesc {
            row_begin: self.first.row_begin + e * self.stride,
            ..self.first
        }
    }
}

/// A sparse MoE FFN (`Qwen3_5MoeForCausalLM`):
///
/// ```text
/// p = softmax(router(h)); (w, ids) = top_k(p); w /= sum(w)
/// y = sum_j w_j * down[ids_j](silu(gate[ids_j](h)) * up[ids_j](h)) + sigmoid(shared_score(h)) * shared(h)
/// ```
pub struct MoeFfn {
    /// `[experts, hidden]` router logits.
    pub router: WDesc,
    /// `[1, hidden]` shared-expert gate logit.
    pub shared_score: WDesc,
    /// router and shared score as one view when adjacent in one parent (one GEMM).
    pub router_score: Option<WDesc>,
    /// Routed experts: gate and up `[intermediate, hidden]`, down `[hidden, intermediate]`. Gate
    /// and up may interleave in one parent (ninfer stores expert e's gate then up at rows
    /// `2e * I` and `(2e + 1) * I`).
    pub gate: ExpertBank,
    pub up: ExpertBank,
    pub down: ExpertBank,
    pub shared: DenseFfn,
}

// One per layer, built once at load: the variant size difference costs nothing.
#[allow(clippy::large_enum_variant)]
pub enum Ffn {
    Dense(DenseFfn),
    Moe(MoeFfn),
}

pub struct Layer {
    pub input_norm: usize,
    pub post_norm: usize,
    pub mixer: Mixer,
    pub ffn: Ffn,
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
    /// Bytes placed in system RAM (see [`TextWeights::load_opts`]).
    pub host_bytes: u64,
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
        progress: impl FnMut(&str),
    ) -> Result<Self> {
        Self::load_opts(b, art, cfg, mtp, &[], progress)
    }

    /// [`Self::load_with`], keeping the matrices named in `host` (e.g. `text/token_embedding`,
    /// which is only gathered a few rows at a time) in system RAM the device reads directly
    /// ([`Backend::alloc_host`]) instead of device memory.
    pub fn load_opts(
        b: &B,
        art: &Artifact,
        cfg: &TextConfig,
        mtp: bool,
        host: &[String],
        mut progress: impl FnMut(&str),
    ) -> Result<Self> {
        let mut l = Loader::new(b, art);
        l.host = host.to_vec();
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
                    let qkv = l.runs_of(&[&p("gdn/query"), &p("gdn/key"), &p("gdn/value")])?;
                    let ab = l.runs_of(&[&p("gdn/a_projection"), &p("gdn/b_projection")])?;
                    let z = l.matrix(&p("gdn/z"))?;
                    let out = l.matrix(&p("gdn/output"))?;
                    expect(&p("gdn qkv"), qkv.rows(), cfg.gdn_qkv())?;
                    expect(&p("gdn ab"), ab.rows(), 2 * cfg.gdn_v_heads)?;
                    expect(&p("gdn z"), z.rows, cfg.gdn_v())?;
                    let qkvz = qkv.single().and_then(|q| fuse(&[q, &z]));
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
            host_bytes: l.host_bytes,
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
    /// Matrices placed in system RAM (by binding name); `bytes` counts device bytes only.
    host: Vec<String>,
    pub(crate) host_bytes: u64,
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

    /// Uploads parent `id` once (to system RAM when `host`); returns (slot, format, n, k).
    fn parent(&mut self, id: &str, host: bool) -> Result<(usize, WeightFormat, usize, usize)> {
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
        if host {
            self.host_bytes += bytes.len() as u64;
            self.parents.push(self.b.upload_bytes_host(&bytes)?);
        } else {
            self.bytes += bytes.len() as u64;
            self.parents.push(self.b.upload_bytes(&bytes)?);
        }
        let entry = (self.parents.len() - 1, fmt, n, k);
        self.index.insert(id.to_string(), entry);
        Ok(entry)
    }

    pub(crate) fn matrix(&mut self, name: &str) -> Result<WDesc> {
        let (id, range) = self.binding(name)?;
        let host = self.host.iter().any(|h| h == name);
        let (parent, format, n, k) = self.parent(&id, host)?;
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

    /// The named row views in order, merging each into the previous run when adjacent in one
    /// parent.
    pub(crate) fn runs_of(&mut self, names: &[&str]) -> Result<Runs> {
        let mut runs: Vec<WDesc> = Vec::new();
        for name in names {
            let d = self.matrix(name)?;
            match runs.last_mut() {
                Some(a) if a.parent == d.parent && a.row_begin + a.rows == d.row_begin => {
                    a.rows += d.rows;
                    a.act = a.act.meet(d.act);
                }
                _ => runs.push(d),
            }
        }
        if runs.is_empty() {
            return Err(Error::Config("empty row list".into()));
        }
        Ok(Runs(runs))
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
            host: Vec::new(),
            host_bytes: 0,
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

/// MTP predictor weights (`Qwen3_5MTP`, or `Qwen3_5MoeMTP` whose block FFN is MoE like the
/// target's): stem norms and input projection, one attention block with the target's geometry,
/// and a final norm. Embedding and output head are the target's.
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
    let post_norm = l.param(&p("post_attention_norm"), cfg.hidden)?;
    let ffn = match &cfg.ffn {
        FfnConfig::Dense { intermediate } => Ffn::Dense(load_swiglu(
            l,
            cfg,
            &|s| p(&format!("mlp/{s}")),
            *intermediate,
        )?),
        FfnConfig::Moe(m) => Ffn::Moe(load_moe(l, cfg, p, m)?),
    };
    Ok(Layer {
        input_norm,
        post_norm,
        mixer,
        ffn,
    })
}

/// `{gate, up, down}` of a SwiGLU with the given intermediate width.
fn load_swiglu<B: Backend>(
    l: &mut Loader<'_, B>,
    cfg: &TextConfig,
    p: &dyn Fn(&str) -> String,
    intermediate: usize,
) -> Result<DenseFfn> {
    let (gate, up, down) = (
        l.matrix(&p("gate"))?,
        l.matrix(&p("up"))?,
        l.matrix(&p("down"))?,
    );
    expect(&p("gate"), gate.rows, intermediate)?;
    expect(&p("up"), up.rows, intermediate)?;
    expect(&p("down"), down.rows, cfg.hidden)?;
    Ok(DenseFfn {
        gate_up: fuse(&[&gate, &up]),
        gate,
        up,
        down,
    })
}

fn load_moe<B: Backend>(
    l: &mut Loader<'_, B>,
    cfg: &TextConfig,
    p: &dyn Fn(&str) -> String,
    m: &MoeConfig,
) -> Result<MoeFfn> {
    let router = l.matrix(&p("moe/router"))?;
    let shared_score = l.matrix(&p("moe/shared_score"))?;
    expect(&p("moe/router"), router.rows, m.experts)?;
    expect(&p("moe/shared_score"), shared_score.rows, 1)?;
    let mut bank = |role: &str, rows: usize| -> Result<ExpertBank> {
        let views = (0..m.experts)
            .map(|e| l.matrix(&p(&format!("moe/experts/{e}/{role}"))))
            .collect::<Result<Vec<_>>>()?;
        let bank = ExpertBank::from_experts(&p(&format!("moe/experts/*/{role}")), &views)?;
        expect(&p(&format!("moe/experts/0/{role}")), bank.first.rows, rows)?;
        Ok(bank)
    };
    let gate = bank("gate", m.intermediate)?;
    let up = bank("up", m.intermediate)?;
    let down = bank("down", cfg.hidden)?;
    Ok(MoeFfn {
        router_score: fuse(&[&router, &shared_score]),
        router,
        shared_score,
        gate,
        up,
        down,
        shared: load_swiglu(
            l,
            cfg,
            &|s| p(&format!("moe/shared/{s}")),
            m.shared_intermediate,
        )?,
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    fn view(parent: usize, row_begin: usize, rows: usize) -> WDesc {
        WDesc {
            parent,
            format: WeightFormat::Q4G64,
            n: 64,
            k: 128,
            row_begin,
            rows,
            act: ActPolicy::A16Only,
        }
    }

    #[test]
    fn interleaved_gate_up_form_two_banks() {
        // ninfer's layout: expert e's gate at rows 2e*I, its up at (2e + 1)*I.
        let i = 8;
        let gates: Vec<_> = (0..4).map(|e| view(0, 2 * e * i, i)).collect();
        let ups: Vec<_> = (0..4).map(|e| view(0, (2 * e + 1) * i, i)).collect();
        let (g, u) = (
            ExpertBank::from_experts("gate", &gates).unwrap(),
            ExpertBank::from_experts("up", &ups).unwrap(),
        );
        assert_eq!((g.experts, g.stride, u.stride), (4, 2 * i, 2 * i));
        assert_eq!(u.expert(3), ups[3]);
    }

    #[test]
    fn irregular_experts_are_refused() {
        let i = 8;
        let mut v: Vec<_> = (0..3).map(|e| view(0, e * i, i)).collect();
        assert_eq!(ExpertBank::from_experts("down", &v).unwrap().stride, i);
        v[2].row_begin += 1;
        assert!(ExpertBank::from_experts("down", &v).is_err());
        v[2] = view(1, 2 * i, i);
        assert!(ExpertBank::from_experts("down", &v).is_err());
        let reversed: Vec<_> = (0..3).rev().map(|e| view(0, e * i, i)).collect();
        assert!(ExpertBank::from_experts("down", &reversed).is_err());
        let overlapping: Vec<_> = (0..3).map(|e| view(0, e * 4, i)).collect();
        assert!(ExpertBank::from_experts("down", &overlapping).is_err());
    }

    #[test]
    fn one_expert_is_a_bank() {
        let b = ExpertBank::from_experts("gate", &[view(0, 16, 8)]).unwrap();
        assert_eq!((b.experts, b.stride, b.expert(0).row_begin), (1, 8, 16));
    }
}
