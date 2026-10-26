//! Qwen4-Exp text configuration, read from a llama.cpp GGUF (`general.architecture = "qwen4exp"`).

use serde_json::Value;
use velox_core::{Error, Result};
use velox_weights::gguf::Gguf;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    /// Gated DeltaNet (`linear_attention`).
    Gdn,
    /// Gated attention through the QSA indexer (`indexed_attention`).
    Qsa,
}

/// The per-layer n-gram embedding (docs/qwen4_exp.md#ple-per-layer-n-gram-embedding-layer-1).
#[derive(Clone, Debug, PartialEq)]
pub struct PleConfig {
    pub layer: usize,
    pub ngram: usize,
    pub heads_per_ngram: usize,
    pub conv_width: usize,
    /// Dilation of the convolution: the n-gram size.
    pub dilation: usize,
    pub eos: u32,
    /// What image positions hash as (the image token, else EOS).
    pub image_token: u32,
    /// Values per table row (one head's slice).
    pub row_dim: usize,
    pub multipliers: Vec<u64>,
    pub offsets: Vec<u64>,
    pub vocab_sizes: Vec<u64>,
}

impl PleConfig {
    /// Hashed heads: `heads_per_ngram` for every n-gram order from 2 to `ngram`.
    pub fn heads(&self) -> usize {
        (self.ngram - 1) * self.heads_per_ngram
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen4Config {
    pub hidden: usize,
    pub vocab: usize,
    pub layers: Vec<LayerKind>,
    pub eps: f32,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub rope_theta: f32,
    pub gdn_k_heads: usize,
    pub gdn_v_heads: usize,
    pub gdn_k_dim: usize,
    pub gdn_v_dim: usize,
    pub conv_width: usize,
    pub experts: usize,
    pub top_k: usize,
    pub expert_width: usize,
    pub shared_width: usize,
    /// Residual streams (`hc_count`).
    pub streams: usize,
    pub hc_rank: usize,
    pub qsa_heads: usize,
    pub qsa_dim: usize,
    pub qsa_budget: usize,
    pub qsa_ratio: usize,
    pub ple: Option<PleConfig>,
}

impl Qwen4Config {
    pub fn gdn_qk(&self) -> usize {
        2 * self.gdn_k_heads * self.gdn_k_dim
    }

    pub fn gdn_v(&self) -> usize {
        self.gdn_v_heads * self.gdn_v_dim
    }

    /// Conv channels and qkv projection width.
    pub fn gdn_qkv(&self) -> usize {
        self.gdn_qk() + self.gdn_v()
    }

    pub fn q_dim(&self) -> usize {
        self.heads * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// Width of the residual: `streams * hidden`.
    pub fn wide(&self) -> usize {
        self.streams * self.hidden
    }

    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arch = g
            .metadata
            .get("general.architecture")
            .and_then(Value::as_str);
        if arch != Some("qwen4exp") {
            return Err(Error::Unsupported(format!(
                "GGUF architecture {arch:?}, expected \"qwen4exp\" (Qwen3.8-Flash-Next)"
            )));
        }
        let key = |k: &str| format!("qwen4exp.{k}");
        let u = |k: &str| -> Result<usize> {
            g.meta_u64(&key(k))
                .map(|v| v as usize)
                .ok_or_else(|| Error::Config(format!("GGUF has no {}", key(k))))
        };
        let f = |k: &str| -> Result<f32> {
            g.metadata
                .get(&key(k))
                .and_then(Value::as_f64)
                .map(|v| v as f32)
                .ok_or_else(|| Error::Config(format!("GGUF has no {}", key(k))))
        };
        let arr = |k: &str| -> Result<Vec<u64>> {
            g.metadata
                .get(&key(k))
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_u64).collect())
                .ok_or_else(|| Error::Config(format!("GGUF has no array {}", key(k))))
        };
        let n = u("block_count")?;
        let interval = u("full_attention_interval").unwrap_or(4);
        let layers: Vec<LayerKind> = (0..n)
            .map(|i| {
                if (i + 1) % interval == 0 {
                    LayerKind::Qsa
                } else {
                    LayerKind::Gdn
                }
            })
            .collect();
        let ratios = arr("attention.compress_ratios")?;
        let ratio = layers
            .iter()
            .zip(&ratios)
            .find(|(k, _)| **k == LayerKind::Qsa)
            .map(|(_, &r)| r as usize)
            .ok_or_else(|| Error::Config("no QSA compress ratio".into()))?;
        for (i, (k, &r)) in layers.iter().zip(&ratios).enumerate() {
            let want = if *k == LayerKind::Qsa {
                ratio as u64
            } else {
                0
            };
            if r != want {
                return Err(Error::Unsupported(format!(
                    "layer {i}: compress ratio {r}, expected {want} (one ratio on attention layers)"
                )));
            }
        }
        let vocab = g
            .tensor("token_embd.weight")
            .map(|t| t.dims[1] as usize)
            .ok_or_else(|| Error::Config("GGUF has no token_embd.weight".into()))?;
        let ple = match g.metadata.get(&key("ple.layers")).and_then(Value::as_array) {
            None => None,
            Some(l) if l.is_empty() => None,
            Some(l) => {
                if l.len() != 1 {
                    return Err(Error::Unsupported(format!(
                        "{} PLE layers (one is supported)",
                        l.len()
                    )));
                }
                let layer = l[0].as_u64().unwrap_or(u64::MAX) as usize;
                let ngram = u("ple.ngram_size")?;
                let eos = u("ple.eos_token_id")? as u32;
                let p = PleConfig {
                    layer,
                    ngram,
                    heads_per_ngram: u("ple.heads_per_ngram")?,
                    conv_width: u("ple.conv_kernel")?,
                    dilation: ngram,
                    eos,
                    image_token: g
                        .meta_u64(&key("ple.image_token_id"))
                        .map_or(eos, |v| v as u32),
                    row_dim: u("embedding_length_per_layer_input")?,
                    multipliers: arr("ple.layer_multipliers")?,
                    offsets: arr("ple.head_offsets")?,
                    vocab_sizes: arr("ple.head_vocab_sizes")?,
                };
                if layer >= n || layers[layer] != LayerKind::Gdn || ngram < 2 {
                    return Err(Error::Config(format!(
                        "PLE layer {layer} (n-gram {ngram}) must be a linear attention layer"
                    )));
                }
                if p.multipliers.len() < ngram
                    || p.offsets.len() < p.heads()
                    || p.vocab_sizes.len() < p.heads()
                {
                    return Err(Error::Config("PLE hash constants are incomplete".into()));
                }
                Some(p)
            }
        };
        let c = Qwen4Config {
            hidden: u("embedding_length")?,
            vocab,
            layers,
            eps: f("attention.layer_norm_rms_epsilon")?,
            heads: u("attention.head_count")?,
            kv_heads: u("attention.head_count_kv")?,
            head_dim: u("attention.key_length")?,
            rotary_dim: u("rope.dimension_count")?,
            rope_theta: f("rope.freq_base")?,
            gdn_k_heads: u("ssm.group_count")?,
            gdn_v_heads: u("ssm.time_step_rank")?,
            gdn_k_dim: u("ssm.state_size")?,
            gdn_v_dim: u("ssm.state_size")?,
            conv_width: u("ssm.conv_kernel")?,
            experts: u("expert_count")?,
            top_k: u("expert_used_count")?,
            expert_width: u("expert_feed_forward_length")?,
            shared_width: u("expert_shared_feed_forward_length")?,
            streams: u("hyper_connection.count")?,
            hc_rank: u("hyper_connection.low_rank")?,
            qsa_heads: u("attention.indexer.head_count")?,
            qsa_dim: u("attention.indexer.key_length")?,
            qsa_budget: u("attention.indexer.top_k")?,
            qsa_ratio: ratio,
            ple,
        };
        if u("attention.value_length")? != c.head_dim || u("ssm.inner_size")? != c.gdn_v() {
            return Err(Error::Unsupported(
                "attention value length or GDN inner size differ from the key geometry".into(),
            ));
        }
        if let Some(p) = &c.ple {
            if p.heads() * p.row_dim != c.hidden {
                return Err(Error::Config(format!(
                    "PLE rows: {} heads x {} != hidden {}",
                    p.heads(),
                    p.row_dim,
                    c.hidden
                )));
            }
        }
        if !c.qsa_budget.is_multiple_of(c.qsa_ratio) || c.rotary_dim > c.qsa_dim {
            return Err(Error::Config(format!(
                "QSA budget {} / ratio {} / rotary {} > index dim {}",
                c.qsa_budget, c.qsa_ratio, c.rotary_dim, c.qsa_dim
            )));
        }
        Ok(c)
    }
}
