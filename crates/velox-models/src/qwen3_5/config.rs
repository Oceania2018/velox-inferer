//! Text configuration of `Qwen3_5ForCausalLM` (Dense) and `Qwen3_5MoeForCausalLM` (sparse MoE) as
//! stored in the artifact's `components.text.config`. Both share the hybrid GDN/attention stack;
//! they differ only in the FFN of every block.

use serde_json::Value;
use velox_core::{Error, Result};

pub const DENSE_ARCHITECTURE: &str = "Qwen3_5ForCausalLM";
pub const MOE_ARCHITECTURE: &str = "Qwen3_5MoeForCausalLM";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    FullAttention,
    /// Gated DeltaNet (`linear_attention`).
    Gdn,
}

/// Sparse MoE FFN geometry. Every block is MoE: `experts` routed SwiGLU experts of which `top_k`
/// run per token (softmax over the router logits, top-k, weights renormalised over the selection),
/// plus one always-on shared SwiGLU expert scaled by `sigmoid(shared_score . h)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeConfig {
    pub experts: usize,
    pub top_k: usize,
    /// Intermediate width of each routed expert.
    pub intermediate: usize,
    /// Intermediate width of the shared expert.
    pub shared_intermediate: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfnConfig {
    Dense { intermediate: usize },
    Moe(MoeConfig),
}

#[derive(Clone, Debug, PartialEq)]
pub struct TextConfig {
    pub hidden: usize,
    pub vocab: usize,
    pub layers: Vec<LayerKind>,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub rotary_dim: usize,
    pub gdn_k_heads: usize,
    pub gdn_k_dim: usize,
    pub gdn_v_heads: usize,
    pub gdn_v_dim: usize,
    pub conv_width: usize,
    pub ffn: FfnConfig,
    pub eps: f32,
    pub max_positions: usize,
}

fn usize_field(v: &Value, key: &str) -> Result<usize> {
    v.get(key)
        .and_then(Value::as_u64)
        .filter(|&x| x > 0)
        .map(|x| x as usize)
        .ok_or_else(|| Error::Config(format!("text config: '{key}' must be a positive integer")))
}

fn f64_field(v: &Value, key: &str) -> Result<f64> {
    v.get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| Error::Config(format!("text config: '{key}' must be a number")))
}

impl TextConfig {
    pub fn from_json(v: &Value) -> Result<Self> {
        let arch = v
            .get("architectures")
            .and_then(|a| a.get(0))
            .and_then(Value::as_str)
            .unwrap_or("");
        if arch != DENSE_ARCHITECTURE && arch != MOE_ARCHITECTURE {
            return Err(Error::Unsupported(format!(
                "text architecture '{arch}' (supported: {DENSE_ARCHITECTURE}, {MOE_ARCHITECTURE})"
            )));
        }
        let ffn = ffn_config(v, arch == MOE_ARCHITECTURE)?;
        let layers = v
            .get("layer_types")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::Config("text config: layer_types missing".into()))?
            .iter()
            .map(|t| match t.as_str() {
                Some("full_attention") => Ok(LayerKind::FullAttention),
                Some("linear_attention") => Ok(LayerKind::Gdn),
                other => Err(Error::Config(format!("unknown layer type {other:?}"))),
            })
            .collect::<Result<Vec<_>>>()?;
        if layers.len() != usize_field(v, "num_hidden_layers")? {
            return Err(Error::Config(
                "layer_types length differs from num_hidden_layers".into(),
            ));
        }
        let rope = v
            .get("rope_parameters")
            .ok_or_else(|| Error::Config("rope_parameters missing".into()))?;
        let head_dim = usize_field(v, "head_dim")?;
        let partial = f64_field(rope, "partial_rotary_factor").unwrap_or(1.0);
        let rotary_dim = (head_dim as f64 * partial).round() as usize;
        // Text tokens use equal positions on all three MRoPE axes, which is plain partial RoPE.
        // Image tokens need the 3-axis positions; that arrives with the vision component.
        let cfg = TextConfig {
            hidden: usize_field(v, "hidden_size")?,
            vocab: usize_field(v, "vocab_size")?,
            layers,
            heads: usize_field(v, "num_attention_heads")?,
            kv_heads: usize_field(v, "num_key_value_heads")?,
            head_dim,
            rope_theta: f64_field(rope, "rope_theta")? as f32,
            rotary_dim,
            gdn_k_heads: usize_field(v, "linear_num_key_heads")?,
            gdn_k_dim: usize_field(v, "linear_key_head_dim")?,
            gdn_v_heads: usize_field(v, "linear_num_value_heads")?,
            gdn_v_dim: usize_field(v, "linear_value_head_dim")?,
            conv_width: usize_field(v, "linear_conv_kernel_dim")?,
            ffn,
            eps: f64_field(v, "rms_norm_eps")? as f32,
            max_positions: usize_field(v, "max_position_embeddings")?,
        };
        if v.get("tie_word_embeddings").and_then(Value::as_bool) == Some(true) {
            return Err(Error::Unsupported(
                "tied embeddings are not wired yet".into(),
            ));
        }
        if !cfg.heads.is_multiple_of(cfg.kv_heads)
            || !cfg.gdn_v_heads.is_multiple_of(cfg.gdn_k_heads)
            || !rotary_dim.is_multiple_of(2)
        {
            return Err(Error::Config(format!(
                "inconsistent head geometry: {cfg:?}"
            )));
        }
        Ok(cfg)
    }

    pub fn q_dim(&self) -> usize {
        self.heads * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    pub fn gdn_qkv(&self) -> usize {
        2 * self.gdn_k_heads * self.gdn_k_dim + self.gdn_v_heads * self.gdn_v_dim
    }

    pub fn gdn_v(&self) -> usize {
        self.gdn_v_heads * self.gdn_v_dim
    }

    pub fn attention_layers(&self) -> usize {
        self.layers
            .iter()
            .filter(|&&l| l == LayerKind::FullAttention)
            .count()
    }

    pub fn moe(&self) -> Option<&MoeConfig> {
        match &self.ffn {
            FfnConfig::Moe(m) => Some(m),
            FfnConfig::Dense { .. } => None,
        }
    }

    /// Intermediate width of the dense SwiGLU every block runs: the FFN itself (Dense) or the
    /// shared expert (MoE). Routed experts have their own width ([`MoeConfig::intermediate`]).
    pub fn mlp_width(&self) -> usize {
        match &self.ffn {
            FfnConfig::Dense { intermediate } => *intermediate,
            FfnConfig::Moe(m) => m.shared_intermediate,
        }
    }

    pub fn architecture(&self) -> &'static str {
        match self.ffn {
            FfnConfig::Dense { .. } => DENSE_ARCHITECTURE,
            FfnConfig::Moe(_) => MOE_ARCHITECTURE,
        }
    }
}

const MOE_FIELDS: [&str; 4] = [
    "num_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "shared_expert_intermediate_size",
];

/// The FFN geometry; Dense and MoE fields are mutually exclusive (as ninfer's converter writes them).
fn ffn_config(v: &Value, moe: bool) -> Result<FfnConfig> {
    if !moe {
        if let Some(key) = MOE_FIELDS.iter().find(|k| v.get(**k).is_some()) {
            return Err(Error::Config(format!(
                "Dense text config carries MoE field '{key}'"
            )));
        }
        return Ok(FfnConfig::Dense {
            intermediate: usize_field(v, "intermediate_size")?,
        });
    }
    if v.get("intermediate_size").is_some() {
        return Err(Error::Config(
            "MoE text config carries the Dense 'intermediate_size'".into(),
        ));
    }
    let m = MoeConfig {
        experts: usize_field(v, MOE_FIELDS[0])?,
        top_k: usize_field(v, MOE_FIELDS[1])?,
        intermediate: usize_field(v, MOE_FIELDS[2])?,
        shared_intermediate: usize_field(v, MOE_FIELDS[3])?,
    };
    if m.top_k > m.experts {
        return Err(Error::Config(format!(
            "text config selects {} of {} experts",
            m.top_k, m.experts
        )));
    }
    Ok(FfnConfig::Moe(m))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn base(arch: &str) -> Value {
        json!({
            "architectures": [arch],
            "hidden_size": 64, "vocab_size": 100, "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rope_parameters": {"rope_theta": 1e7, "partial_rotary_factor": 0.25},
            "linear_num_key_heads": 2, "linear_key_head_dim": 16,
            "linear_num_value_heads": 4, "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4, "rms_norm_eps": 1e-6, "max_position_embeddings": 4096
        })
    }

    fn with(mut v: Value, fields: Value) -> Value {
        for (k, x) in fields.as_object().unwrap() {
            v[k] = x.clone();
        }
        v
    }

    fn moe_fields() -> Value {
        json!({"num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
               "shared_expert_intermediate_size": 48})
    }

    #[test]
    fn dense_config_keeps_its_width() {
        let c = TextConfig::from_json(&with(
            base(DENSE_ARCHITECTURE),
            json!({"intermediate_size": 96}),
        ))
        .unwrap();
        assert_eq!(c.ffn, FfnConfig::Dense { intermediate: 96 });
        assert_eq!(
            (c.mlp_width(), c.moe(), c.architecture()),
            (96, None, DENSE_ARCHITECTURE)
        );
    }

    #[test]
    fn moe_config_reads_expert_geometry() {
        let c = TextConfig::from_json(&with(base(MOE_ARCHITECTURE), moe_fields())).unwrap();
        let want = MoeConfig {
            experts: 8,
            top_k: 2,
            intermediate: 32,
            shared_intermediate: 48,
        };
        assert_eq!(c.moe(), Some(&want));
        assert_eq!((c.mlp_width(), c.architecture()), (48, MOE_ARCHITECTURE));
    }

    #[test]
    fn dense_and_moe_fields_are_exclusive() {
        let dense_with_moe = with(
            with(base(DENSE_ARCHITECTURE), json!({"intermediate_size": 96})),
            json!({"num_experts": 8}),
        );
        assert!(TextConfig::from_json(&dense_with_moe).is_err());
        let moe_with_dense = with(
            with(base(MOE_ARCHITECTURE), moe_fields()),
            json!({"intermediate_size": 96}),
        );
        assert!(TextConfig::from_json(&moe_with_dense).is_err());
    }

    #[test]
    fn moe_cannot_select_more_experts_than_it_has() {
        let v = with(
            with(base(MOE_ARCHITECTURE), moe_fields()),
            json!({"num_experts_per_tok": 9}),
        );
        assert!(TextConfig::from_json(&v).is_err());
    }
}
