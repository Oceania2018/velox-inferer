//! A bare `tokenizer.json` tokenizer for models without a chat frontend (Qwen3-ASR renders its
//! own prompt): text to ids with added tokens recognised, ids back to text.

use tokenizers::Tokenizer;
use velox_core::{Error, Result};

pub struct PlainTokenizer {
    inner: Tokenizer,
}

impl PlainTokenizer {
    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        let inner = Tokenizer::from_file(path)
            .map_err(|e| Error::Config(format!("{}: {e}", path.display())))?;
        Ok(PlainTokenizer { inner })
    }

    /// Ids of `text`; added tokens written in it (`<|im_start|>`, `<asr_text>`) map to their ids.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.inner
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| Error::Config(format!("tokenize: {e}")))
    }

    /// Text of `ids`, special tokens included.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, false)
            .map_err(|e| Error::Config(format!("detokenize: {e}")))
    }
}
