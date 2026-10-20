//! Text configuration of `Qwen3_5ForCausalLM` as stored in the artifact's `components.text.config`.

use serde_json::Value;
use velox_core::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    FullAttention,
    /// Gated DeltaNet (`linear_attention`).
    Gdn,
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
    pub intermediate: usize,
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
        if arch != "Qwen3_5ForCausalLM" {
            return Err(Error::Unsupported(format!(
                "text architecture '{arch}' (supported: Qwen3_5ForCausalLM)"
            )));
        }
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
            intermediate: usize_field(v, "intermediate_size")?,
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
}
