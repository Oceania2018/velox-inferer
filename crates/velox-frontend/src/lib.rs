//! Text frontend: HF tokenizer, Jinja chat template (rendered like HF `apply_chat_template`),
//! stop tokens and Qwen XML tool-call parsing. Resources come from the artifact.

pub mod image;
pub mod template;
pub mod tool_calls;

use serde_json::Value;
use tokenizers::Tokenizer;
use velox_core::{Error, Result};
use velox_weights::Artifact;

pub use template::ChatTemplate;
pub use tool_calls::{parse_output, ParsedOutput};

pub struct Frontend {
    tokenizer: Tokenizer,
    pub template: ChatTemplate,
    /// Stop ids from `generation_config.json` `eos_token_id`.
    pub eos: Vec<u32>,
    /// Token ids the tokenizer can produce; logits past it are never sampled.
    pub public_vocab: usize,
    /// Vision `(min_pixels, max_pixels)` from `preprocessor_config.json`, when the artifact has a
    /// vision component.
    pub vision_pixels: Option<(u64, u64)>,
}

fn resource(art: &Artifact, role: &str) -> Result<Vec<u8>> {
    let id = art.components()["text"]["resources"]
        .get(role)
        .and_then(Value::as_str)
        .ok_or_else(|| Error::Config(format!("artifact has no text resource '{role}'")))?;
    art.read_object(id)
        .map_err(|e| Error::Config(e.to_string()))
}

impl Frontend {
    /// Loads tokenizer, generation config and chat template from the artifact. `template`
    /// overrides the stored chat template (ninfer's `--chat-template`).
    pub fn from_artifact(art: &Artifact, template: Option<String>) -> Result<Self> {
        let tokenizer = Tokenizer::from_bytes(resource(art, "tokenizer.json")?)
            .map_err(|e| Error::Config(format!("tokenizer.json: {e}")))?;
        let gen: Value = serde_json::from_slice(&resource(art, "generation_config.json")?)
            .map_err(|e| Error::Config(format!("generation_config.json: {e}")))?;
        let eos = match &gen["eos_token_id"] {
            Value::Number(n) => vec![n.as_u64().unwrap_or(0) as u32],
            Value::Array(a) => a
                .iter()
                .filter_map(Value::as_u64)
                .map(|n| n as u32)
                .collect(),
            _ => {
                return Err(Error::Config(
                    "generation_config.json has no eos_token_id".into(),
                ))
            }
        };
        let source = match template {
            Some(t) => t,
            None => String::from_utf8(resource(art, "chat_template.jinja")?)
                .map_err(|_| Error::Config("chat template is not UTF-8".into()))?,
        };
        let public_vocab = tokenizer.get_vocab_size(true);
        let vision_pixels = vision_pixels(art)?;
        Ok(Frontend {
            tokenizer,
            template: ChatTemplate::new(source)?,
            eos,
            public_vocab,
            vision_pixels,
        })
    }

    /// Token ids of rendered prompt text; special tokens in the text are recognised.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| Error::Config(format!("tokenize: {e}")))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Text of generated ids, special tokens included (tool-call tags are ordinary text).
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, false)
            .map_err(|e| Error::Config(format!("detokenize: {e}")))
    }

    pub fn is_eos(&self, id: u32) -> bool {
        self.eos.contains(&id)
    }
}

/// `size.shortest_edge` / `size.longest_edge` (pixel areas) of the vision preprocessor config.
fn vision_pixels(art: &Artifact) -> Result<Option<(u64, u64)>> {
    let Some(id) = art
        .components()
        .get("vision")
        .and_then(|v| v["resources"].get("preprocessor_config.json"))
        .and_then(Value::as_str)
    else {
        return Ok(None);
    };
    let bytes = art
        .read_object(id)
        .map_err(|e| Error::Config(e.to_string()))?;
    let cfg: Value = serde_json::from_slice(&bytes)
        .map_err(|e| Error::Config(format!("preprocessor_config.json: {e}")))?;
    let get = |k: &str| {
        cfg["size"][k]
            .as_u64()
            .ok_or_else(|| Error::Config(format!("preprocessor_config.json: size.{k} missing")))
    };
    Ok(Some((get("shortest_edge")?, get("longest_edge")?)))
}
