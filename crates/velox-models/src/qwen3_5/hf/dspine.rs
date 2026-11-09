//! A DSpine drafter checkpoint (HF `DFlashDraftModel` with a code bridge, e.g.
//! `tencent/Qwen3.8-27B-DSpine-b16`) as a standalone `.vi` holding only the `dspine` component,
//! served next to its target's artifact (`velox serve --drafter dspine --draft-artifact`).
//!
//! Bindings (BF16 as stored unless `quant` asks for NVFP4):
//!
//! - `dspine/feature_projection` (`fc`), `dspine/context_norm`, `dspine/final_norm`;
//! - per layer `dspine/layers/{i}/...` like `dflash2/` without the convolutions: attention
//!   `query | key | value` stacked (the context key / value share the key / value rows), output,
//!   MLP gate | up and down, the four norms;
//! - the bridge: `dspine/bridge/read/{i}` (`early_read_weights[i]`, then `read_weight` for the last
//!   layer), `dspine/bridge/write/{i}` (`feature_write_weights[i]`), `gate | predecessor_gate |
//!   prev` stacked, `gate_bias` and the `codebook`. The bridge always stays BF16: the refinement
//!   depends on its rounding.

use std::path::Path;

use serde_json::{json, Map, Value};
use velox_core::{Error, Result};

use super::super::dflash::{DFlashConfig, DraftKind};
use super::{read_json, write_artifact, Builder, ImportReport, RowSrc, SafeTensors};

/// Which DSpine matrices to encode as NVFP4.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DSpineQuant {
    /// Everything BF16 as stored.
    None,
    /// The MLPs (most of the weights; what DFlash2 tolerates).
    Mlp,
    /// The MLPs, the attention projections and the feature projection.
    All,
}

impl DSpineQuant {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "none" => Ok(DSpineQuant::None),
            "mlp" => Ok(DSpineQuant::Mlp),
            "all" => Ok(DSpineQuant::All),
            other => Err(Error::Config(format!(
                "--nvfp4 must be none, mlp or all, got '{other}'"
            ))),
        }
    }
}

fn int(v: &Value, key: &str) -> Result<u64> {
    v[key]
        .as_u64()
        .ok_or_else(|| Error::Config(format!("dspine config: '{key}' must be an integer")))
}

pub fn import_dspine(
    dir: &Path,
    out: &Path,
    quant: DSpineQuant,
    progress: &mut dyn FnMut(&str),
) -> Result<ImportReport> {
    let v = read_json(&dir.join("config.json"))?;
    let dc = DFlashConfig::from_json(&v)?;
    if dc.kind != DraftKind::DSpine {
        return Err(Error::Config(format!(
            "{}: not a DSpine drafter",
            dir.display()
        )));
    }
    let st = SafeTensors::open_dir(dir)?;
    let pre = if st.tensor("fc.weight").is_some() {
        ""
    } else {
        "model."
    };
    let t = |s: &str| format!("{pre}{s}");
    let h = int(&v, "hidden_size")?;
    let vocab = int(&v, "vocab_size")?;
    if v["dflash_config"]["draft_vocab_size"]
        .as_u64()
        .is_some_and(|d| d != vocab)
    {
        return Err(Error::Unsupported(
            "dspine: a draft vocabulary smaller than the target's".into(),
        ));
    }
    if let Some(n) = v["dflash_config"]["num_target_layers"].as_u64() {
        if dc.target_layers.iter().any(|&l| l as u64 >= n) {
            return Err(Error::Config(format!(
                "dspine target layers {:?} past the target's {n} layers",
                dc.target_layers
            )));
        }
    }
    // Checkpoint flags the reference refuses to serve without.
    for name in [
        "draft_block_causal_version",
        "last_write_refine_version",
        "selected_prefix_bridge.independent_write_version",
        "selected_prefix_bridge.terminal_disabled_version",
    ] {
        let tensor = st
            .tensor(&t(name))
            .ok_or_else(|| Error::Config(format!("dspine checkpoint has no {name}")))?;
        let raw = st.read_all(tensor)?;
        if raw.len() != 8 || i64::from_le_bytes(raw[..8].try_into().expect("8 bytes")) != 1 {
            return Err(Error::Unsupported(format!("dspine: {name} must be 1")));
        }
    }
    let out_w = st
        .tensor(&t("selected_prefix_bridge.out.weight"))
        .ok_or_else(|| Error::Config("dspine checkpoint has no bridge out.weight".into()))?;
    if st
        .read_all(out_w)?
        .chunks_exact(2)
        .any(|b| (u16::from_le_bytes([b[0], b[1]]) & 0x7fff) != 0)
    {
        return Err(Error::Unsupported(
            "dspine: a nonzero terminal projection (terminal-corrected bridge)".into(),
        ));
    }

    let mut b = Builder::new(&st);
    // Whole tensors `srcs` stacked into one matrix `id`, bound as `parts`; NVFP4 when `q` and
    // the shape allows, else as stored.
    let mat = |b: &mut Builder<'_>,
               id: &str,
               srcs: &[String],
               parts: &[(&str, u64, u64)],
               q: bool|
     -> Result<()> {
        let shapes: Vec<(u64, u64)> = srcs
            .iter()
            .map(|n| b.get(n).map(|t| t.matrix()))
            .collect::<Result<_>>()?;
        let n: u64 = shapes.iter().map(|s| s.0).sum();
        let k = shapes.first().map_or(0, |s| s.1);
        if q && n.is_multiple_of(128) && k.is_multiple_of(64) {
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
    let (mlp_q, attn_q) = match quant {
        DSpineQuant::None => (false, false),
        DSpineQuant::Mlp => (true, false),
        DSpineQuant::All => (true, true),
    };
    let fc = t("fc.weight");
    mat(
        &mut b,
        &fc,
        std::slice::from_ref(&fc),
        &[("dspine/feature_projection", 0, h)],
        attn_q,
    )?;
    b.vector("dspine/context_norm", &t("hidden_norm.weight"), h as usize)?;
    b.vector("dspine/final_norm", &t("norm.weight"), h as usize)?;
    let (qd, kvd) = (
        (dc.heads * dc.head_dim) as u64,
        (dc.kv_heads * dc.head_dim) as u64,
    );
    let inter = dc.intermediate as u64;
    for i in 0..dc.layers {
        let l = |s: &str| t(&format!("layers.{i}.{s}"));
        let q = |s: &str| format!("dspine/layers/{i}/{s}");
        b.vector(&q("input_norm"), &l("input_layernorm.weight"), h as usize)?;
        b.vector(
            &q("post_attention_norm"),
            &l("post_attention_layernorm.weight"),
            h as usize,
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
            &mut b,
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
            attn_q,
        )?;
        let o = l("self_attn.o_proj.weight");
        mat(
            &mut b,
            &o,
            std::slice::from_ref(&o),
            &[(&q("attention/output"), 0, h)],
            attn_q,
        )?;
        mat(
            &mut b,
            &l("mlp.gate_up"),
            &[l("mlp.gate_proj.weight"), l("mlp.up_proj.weight")],
            &[(&q("mlp/gate"), 0, inter), (&q("mlp/up"), inter, inter)],
            mlp_q,
        )?;
        let down = l("mlp.down_proj.weight");
        mat(
            &mut b,
            &down,
            std::slice::from_ref(&down),
            &[(&q("mlp/down"), 0, h)],
            mlp_q,
        )?;
        progress(&format!("layer {}/{} planned", i + 1, dc.layers));
    }
    // Bridge (BF16).
    let br = |s: &str| t(&format!("selected_prefix_bridge.{s}"));
    let (code, rank, layers) = (dc.code_size as u64, dc.rank as u64, dc.layers as u64);
    let read_names: Vec<String> = (0..dc.layers)
        .map(|i| format!("dspine/bridge/read/{i}"))
        .collect();
    let read_parts: Vec<(&str, u64, u64)> = read_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i as u64 * code, code))
        .collect();
    b.stacked(
        "dspine.bridge.read",
        &[
            RowSrc::all(br("early_read_weights")),
            RowSrc::all(br("read_weight")),
        ],
        &read_parts,
    )?;
    let write_names: Vec<String> = (0..dc.layers)
        .map(|i| format!("dspine/bridge/write/{i}"))
        .collect();
    let write_parts: Vec<(&str, u64, u64)> = write_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i as u64 * h, h))
        .collect();
    b.stacked(
        "dspine.bridge.write",
        &[RowSrc::all(br("feature_write_weights"))],
        &write_parts,
    )?;
    b.stacked(
        "dspine.bridge.packed",
        &[
            RowSrc::all(br("gate.weight")),
            RowSrc::all(br("predecessor_gate.weight")),
            RowSrc::all(br("prev.weight")),
        ],
        &[
            ("dspine/bridge/gate", 0, rank),
            ("dspine/bridge/predecessor_gate", rank, rank),
            ("dspine/bridge/prev", 2 * rank, rank),
        ],
    )?;
    b.vector("dspine/bridge/gate_bias", &br("gate.bias"), rank as usize)?;
    b.stacked(
        "dspine.bridge.codebook",
        &[RowSrc::all(br("codebook"))],
        &[("dspine/bridge/codebook", 0, vocab)],
    )?;
    let early = b.get(&br("early_read_weights"))?.matrix();
    let write = b.get(&br("feature_write_weights"))?.matrix();
    if early != ((layers - 1) * code, h) || write != (layers * h, rank) {
        return Err(Error::Config("dspine bridge tensor shapes".into()));
    }

    let name = dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_string();
    let mut components = Map::new();
    components.insert("dspine".into(), json!({ "config": v }));
    let provenance = json!({
        "converter": "velox import-dspine",
        "hf": { "dir": name, "architectures": v.get("architectures") },
        "nvfp4": format!("{quant:?}").to_lowercase(),
    });
    let w = write_artifact(
        &st,
        b,
        components,
        json!({ "name": name }),
        provenance,
        out,
        progress,
    )?;
    Ok(ImportReport {
        formats: w.formats,
        objects: w.objects,
        bindings: w.bindings,
        uses: w.uses,
        payload_bytes: w.payload,
        mtp: false,
    })
}
