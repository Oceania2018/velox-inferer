//! VRAM planning: how much of a card is left for the KV cache once weights, workspace and the
//! driver/display reserve are placed. This is where a 32 GB desktop and a 24 GB laptop RTX 5090
//! diverge even though they run the same kernels.

use crate::error::{Error, Result};
use crate::profile::DeviceProfile;

const MIB: u64 = 1 << 20;

/// Shape of a decoder-only transformer, for sizing and for the synthetic benchmark model.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelSpec {
    pub name: &'static str,
    pub layers: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    pub max_context: usize,
    pub tied_embeddings: bool,
    pub rope_theta: f32,
    pub norm_eps: f32,
}

impl ModelSpec {
    /// Shapes from the models' published configs. Sizing only: extra per-model layers such as
    /// Qwen3's q/k norms are omitted because they do not change memory materially.
    pub const PRESETS: &'static [ModelSpec] = &[
        ModelSpec {
            name: "tiny",
            layers: 2,
            hidden: 256,
            intermediate: 512,
            heads: 4,
            kv_heads: 2,
            head_dim: 64,
            vocab: 1024,
            max_context: 2048,
            tied_embeddings: true,
            rope_theta: 10_000.0,
            norm_eps: 1e-6,
        },
        ModelSpec {
            name: "llama-3.1-8b",
            layers: 32,
            hidden: 4096,
            intermediate: 14336,
            heads: 32,
            kv_heads: 8,
            head_dim: 128,
            vocab: 128_256,
            max_context: 131_072,
            tied_embeddings: false,
            rope_theta: 500_000.0,
            norm_eps: 1e-5,
        },
        ModelSpec {
            name: "qwen3-32b",
            layers: 64,
            hidden: 5120,
            intermediate: 25600,
            heads: 64,
            kv_heads: 8,
            head_dim: 128,
            vocab: 151_936,
            max_context: 32_768,
            tied_embeddings: false,
            rope_theta: 1_000_000.0,
            norm_eps: 1e-6,
        },
    ];

    pub fn preset(name: &str) -> Result<ModelSpec> {
        Self::PRESETS
            .iter()
            .find(|m| m.name.eq_ignore_ascii_case(name))
            .cloned()
            .ok_or_else(|| {
                let names: Vec<_> = Self::PRESETS.iter().map(|m| m.name).collect();
                Error::Config(format!(
                    "unknown model '{name}'; presets: {}",
                    names.join(", ")
                ))
            })
    }

    pub fn q_dim(&self) -> usize {
        self.heads * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    pub fn params_per_layer(&self) -> u64 {
        let h = self.hidden as u64;
        let attn = h * self.q_dim() as u64 * 2 + h * self.kv_dim() as u64 * 2;
        let mlp = 3 * h * self.intermediate as u64;
        attn + mlp + 2 * h
    }

    pub fn params(&self) -> u64 {
        let embed = (self.vocab * self.hidden) as u64;
        let head = if self.tied_embeddings { 0 } else { embed };
        embed + head + self.layers as u64 * self.params_per_layer() + self.hidden as u64
    }

    pub fn kv_bytes_per_token(&self, kv_bits: u32) -> u64 {
        2 * self.layers as u64 * self.kv_dim() as u64 * u64::from(kv_bits) / 8
    }
}

/// Per-device policy chosen by the hardware variant. Defaults come from the profile; a branch can
/// override them, e.g. to keep a larger reserve on laptops that also drive the display.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanPolicy {
    pub reserved_mib: u64,
    /// Largest prefill chunk; sizes the activation workspace.
    pub workspace_tokens: usize,
}

impl PlanPolicy {
    pub fn for_profile(p: &DeviceProfile) -> Self {
        PlanPolicy {
            reserved_mib: u64::from(p.reserved_mib),
            workspace_tokens: 2048,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanRequest {
    /// Average stored bits per weight, e.g. 16 for BF16 or ~4.5 for 4-bit groupwise.
    pub weight_bits: f32,
    pub kv_bits: u32,
    /// Concurrent sequences sharing the KV budget.
    pub sequences: usize,
}

impl Default for PlanRequest {
    fn default() -> Self {
        PlanRequest {
            weight_bits: 16.0,
            kv_bits: 16,
            sequences: 1,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MemoryPlan {
    pub vram: u64,
    pub reserved: u64,
    pub weights: u64,
    pub workspace: u64,
    /// May be negative when the model does not fit.
    pub kv_budget: i64,
    pub kv_bytes_per_token: u64,
    pub max_kv_tokens: u64,
    pub context_per_sequence: u64,
    pub fits: bool,
}

/// Smallest per-sequence context worth serving; below it the plan is reported as not fitting.
pub const MIN_CONTEXT: u64 = 2048;

pub fn plan(
    profile: &DeviceProfile,
    policy: PlanPolicy,
    model: &ModelSpec,
    req: PlanRequest,
) -> Result<MemoryPlan> {
    if req.weight_bits.is_nan() || req.weight_bits <= 0.0 || req.kv_bits == 0 || req.sequences == 0
    {
        return Err(Error::Config(
            "weight bits, kv bits and sequences must be positive".into(),
        ));
    }
    let vram = profile.vram_bytes();
    let reserved = policy.reserved_mib * MIB;
    let weights = (model.params() as f64 * f64::from(req.weight_bits) / 8.0).ceil() as u64;
    // f32 activations of one prefill chunk: residual, norm, q/k/v, attention, MLP and logits.
    let per_token = 4 * model.hidden
        + model.q_dim()
        + 2 * model.kv_dim()
        + 3 * model.intermediate
        + model.vocab;
    let workspace = (policy.workspace_tokens * per_token * 4) as u64;
    let kv_budget = vram as i64 - (reserved + weights + workspace) as i64;
    let kv_bytes_per_token = model.kv_bytes_per_token(req.kv_bits);
    let max_kv_tokens = if kv_budget > 0 {
        kv_budget as u64 / kv_bytes_per_token
    } else {
        0
    };
    let context_per_sequence = (max_kv_tokens / req.sequences as u64).min(model.max_context as u64);
    let fits = context_per_sequence >= MIN_CONTEXT.min(model.max_context as u64);
    Ok(MemoryPlan {
        vram,
        reserved,
        weights,
        workspace,
        kv_budget,
        kv_bytes_per_token,
        max_kv_tokens,
        context_per_sequence,
        fits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::by_id;

    #[test]
    fn preset_param_counts_match_published_sizes() {
        let llama = ModelSpec::preset("llama-3.1-8b").unwrap().params() as f64 / 1e9;
        assert!((llama - 8.03).abs() < 0.02, "{llama}");
        let qwen = ModelSpec::preset("qwen3-32b").unwrap().params() as f64 / 1e9;
        assert!((qwen - 32.76).abs() < 0.05, "{qwen}");
    }

    #[test]
    fn laptop_gets_less_context_than_desktop() {
        let model = ModelSpec::preset("qwen3-32b").unwrap();
        let req = PlanRequest {
            weight_bits: 4.5,
            kv_bits: 8,
            sequences: 1,
        };
        let run = |id| {
            let p = by_id(id).unwrap();
            plan(p, PlanPolicy::for_profile(p), &model, req).unwrap()
        };
        let desktop = run("rtx-5090");
        let laptop = run("rtx-5090-laptop");
        assert!(desktop.fits && laptop.fits);
        assert!(desktop.max_kv_tokens > 2 * laptop.max_kv_tokens);
    }

    #[test]
    fn bf16_32b_does_not_fit_in_32gb() {
        let p = by_id("rtx-5090").unwrap();
        let model = ModelSpec::preset("qwen3-32b").unwrap();
        let plan = plan(
            p,
            PlanPolicy::for_profile(p),
            &model,
            PlanRequest::default(),
        )
        .unwrap();
        assert!(!plan.fits && plan.kv_budget < 0 && plan.max_kv_tokens == 0);
    }
}
