//! Imports a llama.cpp GGUF of Qwen3.5 / 3.8 (`general.architecture = "qwen35"`) as a `.vi`.
//!
//! The GGUF supplies the text stack. A donor artifact (`.ninfer` / `.vi` of the same model)
//! supplies everything a GGUF does not carry in velox's terms: the text config, tokenizer and
//! chat template, the vision tower, the DFlash2 drafter and its draft vocabulary head. The donor's
//! `text/` weights and its MTP head are dropped.
//!
//! Quantized matrices keep their GGML super-blocks byte for byte (`ggml_<type>` /
//! `ggml_block_v1`); only whole rows move, which is exact for every type. llama.cpp's converter
//! rewrites some tensors before quantizing, and the import undoes what it can:
//!
//! - `*norm.weight` except the GDN norm hold `1 + w`: stored as `w` again (velox applies `1 + w`).
//! - `ssm_a` holds `-exp(A_log)`: stored as `A_log`.
//! - `ssm_conv1d` is `[channels][width]`: transposed to velox's `[width][channels]`.
//! - GDN value heads are in tiled order (`j * k_heads + g` for value head `j` of key head `g`). The
//!   value rows of `attn_qkv`, the conv channels, `ssm_alpha` / `ssm_beta`, `A_log` and `dt_bias`
//!   go back to grouped order (`g * (v_heads / k_heads) + j`). `attn_gate` rows and `ssm_out`
//!   columns stay tiled (columns cannot move inside a block), so the text config gets
//!   `"velox_gdn_value_heads": "tiled"` and the model transposes the recurrence output instead.
//! - Attention `attn_q` interleaves each head's query and gate rows: split into query rows then
//!   gate rows.

use std::path::Path;

use serde_json::{json, Map, Value};
use velox_core::ggml::GgmlType;
use velox_core::{Error, Result};
use velox_weights::gguf::{Gguf, GgufTensor};
use velox_weights::{Artifact, Container};

use super::config::{FfnConfig, LayerKind, TextConfig};

fn werr(e: velox_weights::Error) -> Error {
    Error::Config(e.to_string())
}

/// Where an output object's bytes come from.
enum Src {
    /// The donor's logical payload from this offset.
    Donor(u64),
    /// GGUF bytes from this file offset, unchanged.
    Raw(u64),
    /// Rows of `row_bytes` from a GGUF tensor at this file offset: output row `i` is input row
    /// `perm[i]`.
    Rows {
        at: u64,
        row_bytes: u64,
        perm: Vec<u32>,
    },
    /// Bytes computed here (small vectors).
    Data(Vec<u8>),
}

struct Out {
    /// The directory entry without `offset` / `bytes`.
    entry: Map<String, Value>,
    bytes: u64,
    src: Src,
    offset: u64,
}

/// Summary of an import.
pub struct ImportReport {
    /// (format, objects, bytes) of the text weights taken from the GGUF.
    pub text_formats: Vec<(String, usize, u64)>,
    pub text_bytes: u64,
    pub donor_bytes: u64,
    pub payload_bytes: u64,
}

struct Builder<'a> {
    gguf: &'a Gguf,
    outs: Vec<Out>,
    bindings: Map<String, Value>,
    /// Container format of each new binding (for its activation policy).
    formats: Vec<(String, String)>,
}

/// (container format, layout, bytes per row of `k`) of a GGUF tensor type.
fn tensor_format(t: &GgufTensor, k: u64) -> Result<(String, &'static str, u64)> {
    match t.ggml_type {
        0 => Ok(("fp32".into(), "contiguous_le_v1", 4 * k)),
        30 => Ok(("bf16".into(), "contiguous_le_v1", 2 * k)),
        id => {
            let g = GgmlType::from_gguf_id(id).ok_or_else(|| {
                Error::Unsupported(format!("{}: ggml type {id} is not supported", t.name))
            })?;
            if !k.is_multiple_of(g.block_values() as u64) {
                return Err(Error::Config(format!(
                    "{}: K = {k} is not a multiple of {}",
                    t.name,
                    g.block_values()
                )));
            }
            Ok((
                g.name().into(),
                "ggml_block_v1",
                g.row_bytes(k as usize) as u64,
            ))
        }
    }
}

impl<'a> Builder<'a> {
    fn get(&self, name: &str) -> Result<&'a GgufTensor> {
        self.gguf
            .tensor(name)
            .ok_or_else(|| Error::Config(format!("GGUF has no tensor {name}")))
    }

    fn push(&mut self, id: String, entry: Map<String, Value>, bytes: u64, src: Src) -> String {
        let mut e = entry;
        e.insert("id".into(), id.clone().into());
        self.outs.push(Out {
            entry: e,
            bytes,
            src,
            offset: 0,
        });
        id
    }

    /// A GGUF matrix `[n, k]` (ggml dims `[k, n]`) as one object with rows permuted by `perm`
    /// (identity when `None`), bound as `parts`: (binding, first row, rows).
    fn matrix(
        &mut self,
        name: &str,
        perm: Option<Vec<u32>>,
        parts: &[(&str, u64, u64)],
    ) -> Result<()> {
        let t = self.get(name)?;
        let [k, n] = t.dims[..] else {
            return Err(Error::Config(format!(
                "{name}: {:?} is not a matrix",
                t.dims
            )));
        };
        let (format, layout, row_bytes) = tensor_format(t, k)?;
        let mut entry = Map::new();
        entry.insert("kind".into(), "tensor".into());
        entry.insert("shape".into(), json!([n, k]));
        entry.insert("format".into(), format.clone().into());
        entry.insert("layout".into(), layout.into());
        let src = match perm {
            Some(p) => {
                if p.len() as u64 != n {
                    return Err(Error::Config(format!(
                        "{name}: row permutation of {}",
                        p.len()
                    )));
                }
                Src::Rows {
                    at: t.offset,
                    row_bytes,
                    perm: p,
                }
            }
            None => Src::Raw(t.offset),
        };
        let id = self.push(format!("gguf/{name}"), entry, n * row_bytes, src);
        for &(binding, r0, rows) in parts {
            if r0 + rows > n {
                return Err(Error::Config(format!(
                    "{binding}: rows {r0}+{rows} past {name} [{n}]"
                )));
            }
            let b = if r0 == 0 && rows == n {
                json!({ "object": id })
            } else {
                json!({ "parts": [{ "object": id, "range": [r0 * k, (r0 + rows) * k] }] })
            };
            self.bindings.insert(binding.into(), b);
            self.formats.push((binding.into(), format.clone()));
        }
        Ok(())
    }

    /// A GGUF vector (or matrix, read flat) as f32 values.
    fn values(&self, name: &str) -> Result<Vec<f32>> {
        let t = self.get(name)?;
        let n: u64 = t.dims.iter().product();
        let mut raw = vec![0u8; (n * 4) as usize];
        match t.ggml_type {
            0 => self.gguf.read_at(t.offset, &mut raw).map_err(werr)?,
            other => {
                return Err(Error::Unsupported(format!(
                    "{name}: small tensor of ggml type {other}, expected F32"
                )))
            }
        }
        Ok(raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }

    /// `values` stored as an fp32 tensor of `shape` and bound as `binding`.
    fn vector(&mut self, binding: &str, id: &str, shape: Value, values: &[f32]) {
        let mut entry = Map::new();
        entry.insert("kind".into(), "tensor".into());
        entry.insert("shape".into(), shape);
        entry.insert("format".into(), "fp32".into());
        entry.insert("layout".into(), "contiguous_le_v1".into());
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let id = self.push(
            format!("gguf/{id}"),
            entry,
            bytes.len() as u64,
            Src::Data(bytes),
        );
        self.bindings
            .insert(binding.into(), json!({ "object": id }));
        self.formats.push((binding.into(), "fp32".into()));
    }

    /// A norm stored as `1 + w` by llama.cpp, bound as `w`.
    fn offset_norm(&mut self, binding: &str, name: &str, len: usize) -> Result<()> {
        let v: Vec<f32> = self.values(name)?.iter().map(|x| x - 1.0).collect();
        expect_len(name, v.len(), len)?;
        self.vector(binding, name, json!([len]), &v);
        Ok(())
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

/// Grouped position of each value head: `grouped[g * per + j] = j * k_heads + g` (its tiled index).
fn tiled_of_grouped(k_heads: usize, per: usize) -> Vec<usize> {
    (0..k_heads * per)
        .map(|h| (h % per) * k_heads + h / per)
        .collect()
}

/// Checks the GGUF's hyperparameters against the donor's text config.
fn check_geometry(g: &Gguf, c: &TextConfig) -> Result<()> {
    let arch = g
        .metadata
        .get("general.architecture")
        .and_then(Value::as_str);
    if arch != Some("qwen35") {
        return Err(Error::Unsupported(format!(
            "GGUF architecture {arch:?}, expected \"qwen35\" (Qwen3.5 / 3.8 dense)"
        )));
    }
    let FfnConfig::Dense { intermediate } = c.ffn else {
        return Err(Error::Unsupported("GGUF import of MoE models".into()));
    };
    let want = [
        ("qwen35.block_count", c.layers.len()),
        ("qwen35.embedding_length", c.hidden),
        ("qwen35.feed_forward_length", intermediate),
        ("qwen35.attention.head_count", c.heads),
        ("qwen35.attention.head_count_kv", c.kv_heads),
        ("qwen35.attention.key_length", c.head_dim),
        ("qwen35.ssm.group_count", c.gdn_k_heads),
        ("qwen35.ssm.time_step_rank", c.gdn_v_heads),
        ("qwen35.ssm.state_size", c.gdn_k_dim),
        ("qwen35.ssm.inner_size", c.gdn_v()),
        ("qwen35.ssm.conv_kernel", c.conv_width),
    ];
    for (key, v) in want {
        if g.meta_u64(key) != Some(v as u64) {
            return Err(Error::Config(format!(
                "GGUF {key} = {:?}, donor config implies {v}",
                g.metadata.get(key)
            )));
        }
    }
    if c.gdn_k_dim != c.gdn_v_dim {
        return Err(Error::Unsupported(
            "GDN key and value head sizes differ".into(),
        ));
    }
    Ok(())
}

/// The text stack of `g` as objects and `text/` bindings.
fn text_objects<'a>(g: &'a Gguf, c: &TextConfig) -> Result<Builder<'a>> {
    let mut b = Builder {
        gguf: g,
        outs: Vec::new(),
        bindings: Map::new(),
        formats: Vec::new(),
    };
    let vocab = c.vocab as u64;
    b.matrix(
        "token_embd.weight",
        None,
        &[("text/token_embedding", 0, vocab)],
    )?;
    b.matrix("output.weight", None, &[("text/output_head", 0, vocab)])?;
    b.offset_norm("text/final_norm", "output_norm.weight", c.hidden)?;
    let (kh, per, hd) = (c.gdn_k_heads, c.gdn_v_heads / c.gdn_k_heads, c.gdn_v_dim);
    let grouped = tiled_of_grouped(kh, per);
    let qk = 2 * kh * c.gdn_k_dim;
    for (i, kind) in c.layers.iter().enumerate() {
        let t = |s: &str| format!("blk.{i}.{s}");
        let p = |s: &str| format!("text/layers/{i}/{s}");
        b.offset_norm(&p("input_norm"), &t("attn_norm.weight"), c.hidden)?;
        b.offset_norm(
            &p("post_attention_norm"),
            &t("post_attention_norm.weight"),
            c.hidden,
        )?;
        match kind {
            LayerKind::Gdn => {
                if b.gguf.tensor(&t("attn_qkv.weight")).is_none() {
                    return Err(Error::Config(format!(
                        "layer {i} is linear attention but has no attn_qkv"
                    )));
                }
                // q | k rows unchanged, value rows back to grouped head order.
                let mut perm: Vec<u32> = (0..qk as u32).collect();
                for &h in &grouped {
                    perm.extend((0..hd).map(|e| (qk + h * hd + e) as u32));
                }
                let kd = (kh * c.gdn_k_dim) as u64;
                let (qk, v) = (qk as u64, c.gdn_v() as u64);
                b.matrix(
                    &t("attn_qkv.weight"),
                    Some(perm),
                    &[
                        (&p("gdn/query"), 0, kd),
                        (&p("gdn/key"), kd, kd),
                        (&p("gdn/value"), qk, v),
                    ],
                )?;
                b.matrix(&t("attn_gate.weight"), None, &[(&p("gdn/z"), 0, v)])?;
                let heads: Vec<u32> = grouped.iter().map(|&h| h as u32).collect();
                let nv = c.gdn_v_heads as u64;
                b.matrix(
                    &t("ssm_alpha.weight"),
                    Some(heads.clone()),
                    &[(&p("gdn/a_projection"), 0, nv)],
                )?;
                b.matrix(
                    &t("ssm_beta.weight"),
                    Some(heads),
                    &[(&p("gdn/b_projection"), 0, nv)],
                )?;
                b.matrix(
                    &t("ssm_out.weight"),
                    None,
                    &[(&p("gdn/output"), 0, c.hidden as u64)],
                )?;
                // A = -exp(A_log) and dt_bias, per value head.
                let a = b.values(&t("ssm_a"))?;
                let dt = b.values(&t("ssm_dt.bias"))?;
                expect_len(&t("ssm_a"), a.len(), c.gdn_v_heads)?;
                expect_len(&t("ssm_dt.bias"), dt.len(), c.gdn_v_heads)?;
                if let Some(x) = a.iter().find(|x| x.is_nan() || **x >= 0.0) {
                    return Err(Error::Config(format!(
                        "{}: A = {x} is not negative",
                        t("ssm_a")
                    )));
                }
                let a_log: Vec<f32> = grouped
                    .iter()
                    .map(|&h| f64::from(-a[h]).ln() as f32)
                    .collect();
                let dt: Vec<f32> = grouped.iter().map(|&h| dt[h]).collect();
                b.vector(&p("gdn/a_log"), &t("a_log"), json!([c.gdn_v_heads]), &a_log);
                b.vector(
                    &p("gdn/dt_bias"),
                    &t("dt_bias"),
                    json!([c.gdn_v_heads]),
                    &dt,
                );
                // [channels][width] -> [width][channels], value channels grouped.
                let ch = c.gdn_qkv();
                let w = c.conv_width;
                let conv = b.values(&t("ssm_conv1d.weight"))?;
                expect_len(&t("ssm_conv1d.weight"), conv.len(), ch * w)?;
                let src_ch = |o: usize| {
                    if o < qk as usize {
                        o
                    } else {
                        let (h, e) = ((o - qk as usize) / hd, (o - qk as usize) % hd);
                        qk as usize + grouped[h] * hd + e
                    }
                };
                let conv: Vec<f32> = (0..w * ch)
                    .map(|i| conv[src_ch(i % ch) * w + i / ch])
                    .collect();
                b.vector(&p("gdn/convolution"), &t("conv"), json!([w, ch]), &conv);
                let norm = b.values(&t("ssm_norm.weight"))?;
                expect_len(&t("ssm_norm.weight"), norm.len(), hd)?;
                b.vector(&p("gdn/norm"), &t("ssm_norm.weight"), json!([hd]), &norm);
            }
            LayerKind::FullAttention => {
                // attn_q rows: head h's query at [2h d, (2h + 1) d), its gate right after.
                let (heads, d) = (c.heads, c.head_dim);
                let mut perm = Vec::with_capacity(2 * heads * d);
                for gate in 0..2 {
                    for h in 0..heads {
                        perm.extend((0..d).map(|e| ((2 * h + gate) * d + e) as u32));
                    }
                }
                let (q, kv) = (c.q_dim() as u64, c.kv_dim() as u64);
                b.matrix(
                    &t("attn_q.weight"),
                    Some(perm),
                    &[(&p("attention/query"), 0, q), (&p("attention/gate"), q, q)],
                )?;
                b.matrix(&t("attn_k.weight"), None, &[(&p("attention/key"), 0, kv)])?;
                b.matrix(&t("attn_v.weight"), None, &[(&p("attention/value"), 0, kv)])?;
                b.matrix(
                    &t("attn_output.weight"),
                    None,
                    &[(&p("attention/output"), 0, c.hidden as u64)],
                )?;
                b.offset_norm(&p("attention/query_norm"), &t("attn_q_norm.weight"), d)?;
                b.offset_norm(&p("attention/key_norm"), &t("attn_k_norm.weight"), d)?;
            }
        }
        let width = c.mlp_width() as u64;
        b.matrix(&t("ffn_gate.weight"), None, &[(&p("mlp/gate"), 0, width)])?;
        b.matrix(&t("ffn_up.weight"), None, &[(&p("mlp/up"), 0, width)])?;
        b.matrix(
            &t("ffn_down.weight"),
            None,
            &[(&p("mlp/down"), 0, c.hidden as u64)],
        )?;
    }
    Ok(b)
}

/// Object ids a binding JSON references.
fn binding_objects(b: &Value) -> Vec<String> {
    if let Some(id) = b.get("object").and_then(Value::as_str) {
        return vec![id.to_string()];
    }
    b.get("parts")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|p| p.get("object").and_then(Value::as_str).map(str::to_string))
        .collect()
}

/// Writes `out`: the GGUF's text stack plus the donor's config, resources, vision tower and
/// DFlash2 drafter. `progress` gets a line per written gigabyte.
pub fn import(
    gguf: &Path,
    donor: &Path,
    out: &Path,
    progress: &mut dyn FnMut(&str),
) -> Result<ImportReport> {
    let g = Gguf::open(gguf).map_err(werr)?;
    let d = Artifact::open(donor).map_err(werr)?;
    let mut components = d.components().clone();
    let text_cfg = components
        .get("text")
        .and_then(|t| t.get("config"))
        .cloned()
        .ok_or_else(|| Error::Config("donor has no text config".into()))?;
    let c = TextConfig::from_json(&text_cfg)?;
    check_geometry(&g, &c)?;
    let b = text_objects(&g, &c)?;
    let Builder {
        mut outs,
        mut bindings,
        formats,
        ..
    } = b;

    // Donor parts kept: everything but the text weights and the MTP head.
    let keep = |name: &str| !name.starts_with("text/") && !name.starts_with("mtp/");
    let donor_bindings = d.directory["bindings"]
        .as_object()
        .cloned()
        .unwrap_or_default();
    let mut used: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (name, v) in &donor_bindings {
        if keep(name) {
            used.extend(binding_objects(v));
            bindings.insert(name.clone(), v.clone());
        }
    }
    components.remove("mtp");
    for comp in components.values() {
        for r in comp
            .get("resources")
            .and_then(Value::as_object)
            .into_iter()
            .flatten()
        {
            used.extend(r.1.as_str().map(str::to_string));
        }
    }
    // Uses: the donor's for kept parameters; for the new text matrices, the donor's
    // (parameter, input) pairs with a policy for the new encoding: A8 for GGML blocks (llama.cpp
    // runs them on 8-bit activations), A16 otherwise.
    let mut uses = Vec::new();
    for u in d.directory["uses"].as_array().into_iter().flatten() {
        let Some(param) = u["parameter"].as_str() else {
            continue;
        };
        if keep(param) {
            for aux in u
                .get("auxiliaries")
                .and_then(Value::as_object)
                .into_iter()
                .flatten()
            {
                used.extend(binding_objects(aux.1));
            }
            uses.push(u.clone());
        } else if let Some((_, f)) = formats.iter().find(|(b, _)| b == param) {
            let policy = if f.starts_with("ggml_") {
                "AllowA8"
            } else {
                "A16Only"
            };
            uses.push(
                json!({ "parameter": param, "input": u["input"], "activation_policy": policy }),
            );
        }
    }
    let mut donor_bytes = 0;
    for o in &d.objects {
        if !used.contains(&o.id) {
            continue;
        }
        let entry = d.directory["objects"]
            .as_array()
            .and_then(|a| a.iter().find(|e| e["id"] == o.id.as_str()))
            .and_then(Value::as_object)
            .cloned()
            .ok_or_else(|| Error::Config(format!("donor object {} has no entry", o.id)))?;
        let mut entry = entry;
        entry.remove("offset");
        entry.remove("bytes");
        donor_bytes += o.bytes;
        outs.push(Out {
            entry,
            bytes: o.bytes,
            src: Src::Donor(o.offset),
            offset: 0,
        });
    }

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

    let text = components
        .get_mut("text")
        .and_then(Value::as_object_mut)
        .unwrap();
    let mut cfg = text_cfg.as_object().cloned().unwrap_or_default();
    cfg.insert("velox_gdn_value_heads".into(), "tiled".into());
    text.insert("config".into(), Value::Object(cfg));

    let file_name = |p: &Path| {
        p.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string()
    };
    let mut dir = Map::new();
    dir.insert("components".into(), Value::Object(components));
    dir.insert("objects".into(), Value::Array(objects));
    dir.insert("bindings".into(), Value::Object(bindings));
    dir.insert("uses".into(), Value::Array(uses));
    dir.insert(
        "metadata".into(),
        d.directory
            .get("metadata")
            .cloned()
            .unwrap_or_else(|| json!({})),
    );
    dir.insert(
        "provenance".into(),
        json!({
            "converter": "velox import-gguf",
            "gguf": {
                "file": file_name(gguf),
                "name": g.metadata.get("general.name"),
                "file_type": g.metadata.get("general.file_type"),
            },
            "donor": {
                "file": file_name(donor),
                "artifact_id": velox_weights::hex(&d.artifact_id),
                "provenance": d.directory.get("provenance"),
            },
        }),
    );
    dir.insert(
        "velox".into(),
        json!({
            "format_version": 1,
            "source_container": "gguf",
            "source_artifact_id": velox_weights::hex(&d.artifact_id),
        }),
    );

    let mut text_formats: Vec<(String, usize, u64)> = Vec::new();
    let mut text_bytes = 0;
    for o in outs.iter().filter(|o| !matches!(o.src, Src::Donor(_))) {
        let f = o.entry["format"].as_str().unwrap_or("?").to_string();
        text_bytes += o.bytes;
        match text_formats.iter_mut().find(|x| x.0 == f) {
            Some(x) => {
                x.1 += 1;
                x.2 += o.bytes;
            }
            None => text_formats.push((f, 1, o.bytes)),
        }
    }
    text_formats.sort_by_key(|x| std::cmp::Reverse(x.2));

    let mut next_report = 1u64 << 30;
    let mut fill = |at: u64, buf: &mut [u8]| -> velox_weights::Result<()> {
        buf.fill(0);
        let end = at + buf.len() as u64;
        let first = outs.partition_point(|o| o.offset + o.bytes <= at);
        for o in &outs[first..] {
            if o.offset >= end {
                break;
            }
            let (lo, hi) = (at.max(o.offset), end.min(o.offset + o.bytes));
            let dst = &mut buf[(lo - at) as usize..(hi - at) as usize];
            let rel = lo - o.offset;
            match &o.src {
                Src::Donor(off) => d.read_range(off + rel, dst)?,
                Src::Raw(off) => g.read_at(off + rel, dst)?,
                Src::Data(v) => dst.copy_from_slice(&v[rel as usize..rel as usize + dst.len()]),
                Src::Rows {
                    at: base,
                    row_bytes,
                    perm,
                } => {
                    let mut done = 0;
                    while done < dst.len() {
                        let p = rel + done as u64;
                        let (r, col) = (p / row_bytes, p % row_bytes);
                        let n = ((row_bytes - col) as usize).min(dst.len() - done);
                        let src = base + u64::from(perm[r as usize]) * row_bytes + col;
                        g.read_at(src, &mut dst[done..done + n])?;
                        done += n;
                    }
                }
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
        dir,
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
        text_formats,
        text_bytes,
        donor_bytes,
        payload_bytes: payload,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiled_heads_map_back_to_grouped_order() {
        // 2 key heads x 3 value heads each: llama.cpp's tiled order is [G0v0, G1v0, G0v1, ...].
        let tiled_of = tiled_of_grouped(2, 3);
        // Grouped position g * 3 + j holds tiled head j * 2 + g.
        assert_eq!(tiled_of, vec![0, 2, 4, 1, 3, 5]);
        // llama.cpp's _reorder_v_heads applied to grouped ids [0..6) gives the tiled list; reading
        // it at tiled_of restores the grouped ids.
        let tiled: Vec<usize> = (0..6).map(|t| (t % 2) * 3 + t / 2).collect();
        let back: Vec<usize> = tiled_of.iter().map(|&t| tiled[t]).collect();
        assert_eq!(back, (0..6).collect::<Vec<_>>());
    }
}
