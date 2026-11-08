//! Qwen3-ASR geometry, read from the GGUF pair (docs/qwen3_asr.md#geometry-17b).

use serde_json::Value;
use velox_core::{Error, Result};
use velox_weights::gguf::Gguf;

use super::mel::N_MELS;

#[derive(Clone, Debug, PartialEq)]
pub struct EncConfig {
    pub d: usize,
    pub layers: usize,
    pub heads: usize,
    pub ffn: usize,
    pub conv_ch: usize,
    pub mels: usize,
    pub ln_eps: f32,
    /// Mel frames per conv chunk (`2 * n_window`).
    pub chunk_frames: usize,
    /// Chunks per attention window (`n_window_infer / chunk_frames`).
    pub window_chunks: usize,
}

impl EncConfig {
    pub fn head_dim(&self) -> usize {
        self.d / self.heads
    }

    /// Frequency rows after the three stride-2 convolutions.
    pub fn conv_freq(&self) -> usize {
        half(half(half(self.mels)))
    }

    /// Encoder rows of one conv chunk of `frames` mel frames.
    pub fn chunk_tokens(frames: usize) -> usize {
        half(half(half(frames)))
    }

    /// Encoder rows of a clip of `frames` mel frames.
    pub fn tokens(&self, frames: usize) -> usize {
        let per = Self::chunk_tokens(self.chunk_frames);
        (frames / self.chunk_frames) * per + Self::chunk_tokens(frames % self.chunk_frames)
    }

    /// Rows per attention window for a clip of `frames` frames (the reference scales the longest
    /// chunk's rows, which is a full chunk unless the clip is shorter than one).
    pub fn window_tokens(&self, frames: usize) -> usize {
        Self::chunk_tokens(frames.min(self.chunk_frames)) * self.window_chunks
    }
}

/// `ceil(n / 2)`: the output length of a 3x3, stride 2, pad 1 convolution.
pub fn half(n: usize) -> usize {
    n.div_ceil(2)
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecConfig {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub vocab: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AsrConfig {
    pub enc: EncConfig,
    pub dec: DecConfig,
}

pub const IM_START: u32 = 151_644;
pub const IM_END: u32 = 151_645;
pub const AUDIO_START: u32 = 151_669;
pub const AUDIO_END: u32 = 151_670;
pub const AUDIO_PAD: u32 = 151_676;
pub const ASR_TEXT: u32 = 151_704;

fn num(g: &Gguf, key: &str) -> Result<f64> {
    g.metadata
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| Error::Config(format!("{}: no numeric {key}", g.path.display())))
}

fn int(g: &Gguf, key: &str) -> Result<usize> {
    g.meta_u64(key)
        .filter(|&v| v > 0)
        .map(|v| v as usize)
        .ok_or_else(|| Error::Config(format!("{}: no positive {key}", g.path.display())))
}

impl AsrConfig {
    pub fn from_gguf(text: &Gguf, mmproj: &Gguf) -> Result<Self> {
        let arch = text
            .metadata
            .get("general.architecture")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let proj = mmproj
            .metadata
            .get("clip.audio.projector_type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if arch != "qwen3vl" || proj != "qwen3a" {
            return Err(Error::Unsupported(format!(
                "not a Qwen3-ASR GGUF pair (architecture '{arch}', audio projector '{proj}')"
            )));
        }
        let enc = EncConfig {
            d: int(mmproj, "clip.audio.embedding_length")?,
            layers: int(mmproj, "clip.audio.block_count")?,
            heads: int(mmproj, "clip.audio.attention.head_count")?,
            ffn: int(mmproj, "clip.audio.feed_forward_length")?,
            conv_ch: mmproj
                .tensor("a.conv2d.1.weight")
                .and_then(|t| t.dims.last().copied())
                .ok_or_else(|| Error::Config("mmproj: no a.conv2d.1.weight".into()))?
                as usize,
            mels: int(mmproj, "clip.audio.num_mel_bins")?,
            ln_eps: num(mmproj, "clip.audio.attention.layer_norm_epsilon")? as f32,
            // Not in the GGUF: n_window 50 and n_window_infer 800 in the checkpoint's config.json.
            chunk_frames: 100,
            window_chunks: 8,
        };
        if enc.mels != N_MELS || !enc.d.is_multiple_of(enc.heads) {
            return Err(Error::Unsupported(format!("audio geometry {enc:?}")));
        }
        let p = "qwen3vl";
        let hidden = int(text, &format!("{p}.embedding_length"))?;
        let heads = int(text, &format!("{p}.attention.head_count"))?;
        let dec = DecConfig {
            hidden,
            layers: int(text, &format!("{p}.block_count"))?,
            heads,
            kv_heads: int(text, &format!("{p}.attention.head_count_kv"))?,
            head_dim: text
                .meta_u64(&format!("{p}.attention.key_length"))
                .map_or(hidden / heads, |v| v as usize),
            ffn: int(text, &format!("{p}.feed_forward_length"))?,
            vocab: text
                .tensor("token_embd.weight")
                .and_then(|t| t.dims.get(1).copied())
                .ok_or_else(|| Error::Config("text: no token_embd.weight".into()))?
                as usize,
            rope_theta: num(text, &format!("{p}.rope.freq_base"))? as f32,
            rms_eps: num(text, &format!("{p}.attention.layer_norm_rms_epsilon"))? as f32,
        };
        Ok(AsrConfig { enc, dec })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_counts_match_the_reference_formula() {
        // qwen-asr `_get_feat_extract_output_lengths`.
        let reference = |n: usize| {
            let leave = n % 100;
            let feat = if leave == 0 { 0 } else { (leave - 1) / 2 + 1 };
            let tail = if leave == 0 {
                0
            } else {
                ((feat - 1) / 2 + 1 - 1) / 2 + 1
            };
            tail + (n / 100) * 13
        };
        let c = EncConfig {
            d: 1024,
            layers: 1,
            heads: 16,
            ffn: 4096,
            conv_ch: 480,
            mels: 128,
            ln_eps: 1e-5,
            chunk_frames: 100,
            window_chunks: 8,
        };
        for n in [1, 37, 99, 100, 101, 650, 937, 3000] {
            assert_eq!(c.tokens(n), reference(n), "{n} frames");
        }
        assert_eq!(c.window_tokens(937), 104);
        assert_eq!(c.conv_freq(), 16);
    }
}
