//! Text frontend: HF tokenizer, Jinja chat template (rendered like HF `apply_chat_template`),
//! stop tokens and Qwen XML tool-call parsing. Resources come from the artifact.

pub mod image;
pub mod plain;
pub mod template;
pub mod tool_calls;

use std::collections::HashMap;
use std::sync::Mutex;

use serde_json::Value;
use tokenizers::Tokenizer;
use velox_core::{Error, Result};
use velox_weights::Artifact;

pub use plain::PlainTokenizer;
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

    /// Token ids of several rendered prompts that share a long start (System One questions over
    /// one state): the shared start is tokenized once, then each text's rest. Same ids as
    /// [`Self::encode`] of each text; the first text is also encoded whole to check, and on any
    /// difference (or no usable cut) every text is encoded whole.
    pub fn encode_shared(&self, texts: &[String]) -> Result<Vec<Vec<u32>>> {
        let whole = |texts: &[String]| texts.iter().map(|t| self.encode(t)).collect();
        let Some(cut) = shared_cut(texts) else {
            return whole(texts);
        };
        let head = self.encode(&texts[0][..cut])?;
        let mut out = Vec::with_capacity(texts.len());
        for t in texts {
            let mut ids = head.clone();
            ids.extend(self.encode(&t[cut..])?);
            out.push(ids);
        }
        if out[0] != self.encode(&texts[0])? {
            return whole(texts);
        }
        Ok(out)
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

/// Shared starts shorter than this are tokenized with each text (not worth the check).
const MIN_SHARED_CUT: usize = 2048;

/// Where texts that share a start can be tokenized apart: before the last newline inside the
/// shared start that follows a letter or digit. The byte-level pre-tokenizer ends a letter or
/// digit run before any newline, whatever comes after it, so the ids before and after the cut
/// are those of the whole text. `None` for fewer than two texts or a short shared start.
fn shared_cut(texts: &[String]) -> Option<usize> {
    let (first, rest) = texts.split_first()?;
    if rest.is_empty() {
        return None;
    }
    let shared = rest.iter().fold(first.len(), |n, t| {
        first
            .bytes()
            .zip(t.bytes())
            .take(n)
            .take_while(|(a, b)| a == b)
            .count()
    });
    // The newline itself is shared, so the cut is inside every text, on a char boundary.
    let head = &first.as_bytes()[..shared];
    let cut = (1..shared)
        .rev()
        .find(|&i| head[i] == b'\n' && is_alnum_before(first, i))?;
    (cut >= MIN_SHARED_CUT).then_some(cut)
}

/// Whether the char that ends at byte `i` of `s` is a letter or digit.
fn is_alnum_before(s: &str, i: usize) -> bool {
    s[..i]
        .chars()
        .next_back()
        .is_some_and(char::is_alphanumeric)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn shared_cut_is_before_a_newline_after_a_word() {
        let state = "word ".repeat(500); // 2500 bytes
        let a = format!("<|im_start|>user\n{state}tail\n\nQuestion: a?");
        let b = format!("<|im_start|>user\n{state}tail\n\nQuestion: b?");
        let cut = shared_cut(&texts(&[&a, &b])).unwrap();
        assert_eq!(&a[cut - 4..cut + 2], "tail\n\n");
        assert_eq!(a[..cut], b[..cut]);
    }

    #[test]
    fn shared_cut_skips_punctuation_and_short_starts() {
        let state = "word ".repeat(500);
        // Only newlines after punctuation in the shared start past the prompt head: no cut there.
        let a = format!("x\n{state}end.\n\nQ: a");
        let b = format!("x\n{state}end.\n\nQ: b");
        assert_eq!(shared_cut(&texts(&[&a, &b])), None);
        assert_eq!(shared_cut(&texts(&["short\nA", "short\nB"])), None);
        assert_eq!(shared_cut(&texts(&[&a])), None);
        // Multi-byte letters before the newline.
        let c = format!("{state}é\nQ: a");
        let d = format!("{state}é\nQ: b");
        let cut = shared_cut(&texts(&[&c, &d])).unwrap();
        assert!(c.is_char_boundary(cut) && c[cut..].starts_with('\n'));
    }
}
