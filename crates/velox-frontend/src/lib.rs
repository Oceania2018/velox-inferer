//! Text frontend: HF tokenizer, Jinja chat template (rendered like HF `apply_chat_template`),
//! stop tokens and Qwen XML tool-call parsing. Resources come from the artifact.

pub mod image;
pub mod template;
pub mod tool_calls;

use std::collections::HashMap;
use std::sync::Mutex;

use serde_json::Value;
use tokenizers::Tokenizer;
use velox_core::{Error, Result};
use velox_weights::Artifact;

pub use template::ChatTemplate;
pub use tool_calls::{parse_output, ParsedOutput};

pub struct Frontend {
    tokenizer: Tokenizer,
    /// Encodings of prompt segments (see [`Frontend::encode`]), keyed by their text.
    segments: Mutex<HashMap<String, Vec<u32>>>,
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
        let source = match template {
            Some(t) => t,
            None => String::from_utf8(resource(art, "chat_template.jinja")?)
                .map_err(|_| Error::Config("chat template is not UTF-8".into()))?,
        };
        Self::from_parts(
            &resource(art, "tokenizer.json")?,
            &resource(art, "generation_config.json")?,
            source,
            vision_pixels(art)?,
        )
    }

    /// Loads `tokenizer.json`, `generation_config.json` and `chat_template.jinja` (unless
    /// `template` overrides it) from a Hugging Face model directory, for weights that come
    /// without them (GGUF).
    pub fn from_dir(dir: &std::path::Path, template: Option<String>) -> Result<Self> {
        let read = |name: &str| {
            std::fs::read(dir.join(name))
                .map_err(|e| Error::Config(format!("{}: {e}", dir.join(name).display())))
        };
        let source = match template {
            Some(t) => t,
            None => String::from_utf8(read("chat_template.jinja")?)
                .map_err(|_| Error::Config("chat template is not UTF-8".into()))?,
        };
        Self::from_parts(
            &read("tokenizer.json")?,
            &read("generation_config.json")?,
            source,
            None,
        )
    }

    fn from_parts(
        tokenizer: &[u8],
        generation_config: &[u8],
        source: String,
        vision_pixels: Option<(u64, u64)>,
    ) -> Result<Self> {
        let tokenizer = Tokenizer::from_bytes(tokenizer)
            .map_err(|e| Error::Config(format!("tokenizer.json: {e}")))?;
        let gen: Value = serde_json::from_slice(generation_config)
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
        let public_vocab = tokenizer.get_vocab_size(true);
        Ok(Frontend {
            tokenizer,
            segments: Mutex::new(HashMap::new()),
            template: ChatTemplate::new(source)?,
            eos,
            public_vocab,
            vision_pixels,
        })
    }

    /// Token ids of rendered prompt text; special tokens in the text are recognised.
    ///
    /// The tokenizer splits on special tokens before anything else, so text cut just before a
    /// special token encodes piecewise to the same ids. Chat prompts are cut before each
    /// `<|im_start|>` and every message's ids are cached: an agent's next request repeats all but
    /// its last turns, and re-tokenizing a 16k-token history costs ~20 ms.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        const CUT: &str = "<|im_start|>";
        const MAX_SEGMENTS: usize = 4096;
        let mut ids = Vec::new();
        let mut cache = self
            .segments
            .lock()
            .map_err(|_| Error::Backend("tokenizer cache poisoned".into()))?;
        if cache.len() > MAX_SEGMENTS {
            cache.clear();
        }
        let mut rest = text;
        while !rest.is_empty() {
            // The segment runs to the next cut after its own leading one.
            let skip = if rest.starts_with(CUT) { CUT.len() } else { 0 };
            let end = rest[skip..].find(CUT).map_or(rest.len(), |i| i + skip);
            let seg = &rest[..end];
            match cache.get(seg) {
                Some(v) => ids.extend_from_slice(v),
                None => {
                    let v = self.encode_whole(seg)?;
                    ids.extend_from_slice(&v);
                    cache.insert(seg.to_string(), v);
                }
            }
            rest = &rest[end..];
        }
        drop(cache);
        if std::env::var_os("VELOX_CHECK_TOKENIZE").is_some() && ids != self.encode_whole(text)? {
            return Err(Error::Backend(
                "segmented tokenization differs from whole-text".into(),
            ));
        }
        Ok(ids)
    }

    fn encode_whole(&self, text: &str) -> Result<Vec<u32>> {
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
