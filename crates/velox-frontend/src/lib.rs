//! Text frontend: HF tokenizer, Jinja chat template (rendered like HF `apply_chat_template`),
//! stop tokens and Qwen XML tool-call parsing. Resources come from the artifact.

pub mod image;
pub mod plain;
pub mod template;
pub mod tool_calls;

use std::collections::HashMap;
use std::sync::Mutex;

use serde_json::Value;
use tokenizers::{
    Normalizer, OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer, Tokenizer,
};
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
        let Some(cut) = shared_len(texts).and_then(|shared| self.shared_cut(&texts[0], shared))
        else {
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

    /// Where texts whose first `shared` bytes are equal can be tokenized apart: a pre-token
    /// boundary of `first` (the tokenizer's own normalizer and pre-tokenizer) followed by one
    /// whole pre-token and a byte of lookahead inside the shared start. BPE never merges across
    /// pre-tokens and the pre-tokenizer has no lookbehind, so whatever follows the shared start,
    /// every text splits there and its ids before and after the cut are those of the whole text.
    /// (The old rule, before a newline that follows a letter or digit, found no cut in a state
    /// whose lines end in punctuation: `.` takes the newlines after it into its pre-token.)
    /// `None` when the cut would be short of [`MIN_SHARED_CUT`].
    fn shared_cut(&self, first: &str, shared: usize) -> Option<usize> {
        if shared < MIN_SHARED_CUT {
            return None;
        }
        let mut pre = PreTokenizedString::from(&first[..shared]);
        if let Some(n) = self.tokenizer.get_normalizer() {
            pre.normalize(|s| n.normalize(s)).ok()?;
        }
        if let Some(p) = self.tokenizer.get_pre_tokenizer() {
            p.pre_tokenize(&mut pre).ok()?;
        }
        let splits = pre.get_splits(OffsetReferential::Original, OffsetType::Byte);
        // The last pre-token that ends before the shared start's last byte: its start is the cut.
        let (_, (cut, _), _) = splits.iter().rev().find(|(_, (_, end), _)| *end < shared)?;
        (*cut >= MIN_SHARED_CUT && first.is_char_boundary(*cut)).then_some(*cut)
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

/// Bytes at the start of every text that are equal (on a char boundary of the first). `None` for
/// fewer than two texts.
fn shared_len(texts: &[String]) -> Option<usize> {
    let (first, rest) = texts.split_first()?;
    if rest.is_empty() {
        return None;
    }
    let mut shared = rest.iter().fold(first.len(), |n, t| {
        first
            .bytes()
            .zip(t.bytes())
            .take(n)
            .take_while(|(a, b)| a == b)
            .count()
    });
    while !first.is_char_boundary(shared) {
        shared -= 1;
    }
    Some(shared)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    /// Qwen3.8's pre-tokenizer pattern (`tokenizer.json` `pre_tokenizer.pretokenizers[0]`).
    const QWEN_PRE_TOKENIZER: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    /// Qwen3.8's normalizer and pre-tokenizer verbatim, a byte-level BPE whose merges reach
    /// across the places a cut must not go (punctuation into the newlines after it, a space into
    /// the word after it, a newline run) and `<|im_start|>` as a special token.
    fn qwen_like() -> Frontend {
        let mut bytes: Vec<u32> = (b'!'..=b'~')
            .chain(0xa1..=0xac)
            .chain(0xae..=0xff)
            .map(u32::from)
            .collect();
        let mut chars = bytes.clone();
        let mut extra = 0;
        for b in 0..256u32 {
            if !bytes.contains(&b) {
                bytes.push(b);
                chars.push(256 + extra);
                extra += 1;
            }
        }
        let byte = |b: u8| {
            char::from_u32(chars[bytes.iter().position(|&x| x == u32::from(b)).unwrap()])
                .unwrap()
                .to_string()
        };
        let mut vocab = serde_json::Map::new();
        for b in 0..=255u8 {
            let n = vocab.len();
            vocab.insert(byte(b), n.into());
        }
        let mut merges = Vec::new();
        for (l, r) in [
            (".", "\n"),
            (".\n", "\n"),
            ("\n", "\n"),
            (" ", "t"),
            (" t", "i"),
            ("w", "o"),
            ("wo", "r"),
            ("1", "2"),
        ] {
            let enc = |s: &str| s.bytes().map(byte).collect::<String>();
            let (l, r) = (enc(l), enc(r));
            let n = vocab.len();
            vocab.entry(format!("{l}{r}")).or_insert(n.into());
            merges.push(format!("{l} {r}"));
        }
        let im_start = vocab.len();
        let json = serde_json::json!({
            "version": "1.0",
            "added_tokens": [{"id": im_start, "content": "<|im_start|>", "single_word": false,
                "lstrip": false, "rstrip": false, "normalized": false, "special": true}],
            "normalizer": {"type": "NFC"},
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": QWEN_PRE_TOKENIZER},
                 "behavior": "Isolated", "invert": false},
                {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}]},
            "post_processor": null,
            "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
            "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": "",
                "end_of_word_suffix": "", "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                "vocab": vocab, "merges": merges},
        });
        frontend(&serde_json::to_vec(&json).unwrap())
    }

    fn frontend(tokenizer: &[u8]) -> Frontend {
        Frontend::from_parts(tokenizer, br#"{"eos_token_id": 0}"#, String::new(), None).unwrap()
    }

    /// Texts over one long state that differ after it, the way System One renders questions:
    /// random lines of letters, digits, punctuation, spaces, accents and combining marks.
    fn questions(seed: u64, state_end: &str) -> Vec<String> {
        const PIECES: &[&str] = &[
            "word", "two", " ", "  ", "\n", "\n\n", ".", ".\n\n", "#12", "é", "e\u{301}", "ti",
            "!?", " \n", "'s", "\t", "12", "}",
        ];
        let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut next = |n: usize| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 33) as usize) % n
        };
        let mut state = String::from("<|im_start|>user\n");
        while state.len() < 3000 {
            state.push_str(PIECES[next(PIECES.len())]);
        }
        state.push_str(state_end);
        (0..6)
            .map(|_| {
                let mut q = state.clone();
                for _ in 0..1 + next(12) {
                    q.push_str(PIECES[next(PIECES.len())]);
                }
                q.push_str("\n<|im_start|>assistant\nAnswer:");
                q
            })
            .collect()
    }

    fn check_shared(fe: &Frontend, cases: u64) -> usize {
        let mut cut_found = 0;
        for seed in 0..cases {
            for end in [
                "",
                ".\n\n",
                "end.",
                "}\n",
                "  ",
                "Question: ticket #10",
                "é",
            ] {
                let texts = questions(seed, end);
                let shared = shared_len(&texts).unwrap();
                if fe.shared_cut(&texts[0], shared).is_some() {
                    cut_found += 1;
                }
                let got = fe.encode_shared(&texts).unwrap();
                for (t, ids) in texts.iter().zip(&got) {
                    assert_eq!(
                        ids,
                        &fe.encode_whole(t).unwrap(),
                        "seed {seed}, end {end:?}: {t:?}"
                    );
                }
            }
        }
        cut_found
    }

    #[test]
    fn encode_shared_matches_whole_texts() {
        let fe = qwen_like();
        let cases = 40;
        // Shared starts end anywhere (in a word, after punctuation, in a space run, before an
        // accent), yet a cut is found for nearly every set and the ids never change.
        assert!(check_shared(&fe, cases) as u64 >= cases * 7 * 9 / 10);
    }

    #[test]
    fn shared_cut_after_punctuation_is_near_the_divergence() {
        let fe = qwen_like();
        let state = "A ticket line that ends in a full stop.\n\n".repeat(80);
        let texts: Vec<String> = ["1000", "1001"]
            .iter()
            .map(|n| format!("<|im_start|>user\n{state}Question: Which trade handles ticket #{n}?"))
            .collect();
        let shared = shared_len(&texts).unwrap();
        let cut = fe.shared_cut(&texts[0], shared).unwrap();
        // The old rule found no cut here (every newline follows a '.'); this one is in the
        // question, a few bytes before the texts differ.
        assert!(shared - cut <= 8, "cut {cut}, shared {shared}");
        // Digits are pre-tokens of their own, so the cut can sit inside the number.
        assert!(
            texts[0][..cut].ends_with("ticket #1"),
            "{:?}",
            &texts[0][cut - 20..cut]
        );
        assert_eq!(
            fe.encode_shared(&texts).unwrap()[1],
            fe.encode_whole(&texts[1]).unwrap()
        );
    }

    #[test]
    fn shared_cut_needs_two_texts_and_a_long_start() {
        let fe = qwen_like();
        assert_eq!(shared_len(&texts(&["only one"])), None);
        let short = texts(&["short\nA", "short\nB"]);
        assert_eq!(fe.shared_cut(&short[0], shared_len(&short).unwrap()), None);
        // A shared start that ends inside a multi-byte char is cut back to its boundary.
        let a = format!("{}é", "word ".repeat(500));
        let b = format!("{}è", "word ".repeat(500));
        let n = shared_len(&texts(&[&a, &b])).unwrap();
        assert!(a.is_char_boundary(n) && n == "word ".repeat(500).len());
    }

    /// The same property on the model's own tokenizer: `VELOX_TOKENIZER_JSON=<tokenizer.json>`.
    #[test]
    fn encode_shared_matches_whole_texts_on_a_real_tokenizer() {
        let Some(path) = std::env::var_os("VELOX_TOKENIZER_JSON") else {
            eprintln!("VELOX_TOKENIZER_JSON not set; skipped");
            return;
        };
        let fe = frontend(&std::fs::read(path).unwrap());
        let cases = 200;
        assert!(check_shared(&fe, cases) as u64 >= cases * 7 * 9 / 10);
    }
}
