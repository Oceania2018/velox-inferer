//! Chat engine for Qwen3.5 text + vision models behind the OpenAI-compatible server.
//!
//! One generation lane. Agent loops resend almost the same prompt every step, so the lane keeps
//! the keys of the tokens its state holds plus recurrent-state checkpoints taken where the next
//! request is likely to diverge (before the last image and before the generation prompt). A new
//! request restores the latest checkpoint inside the common prefix and prefills only the rest;
//! the vision tower runs only for images in that remainder.

mod prompt;
mod sampling;

use std::sync::Mutex;
use std::time::Instant;

use serde_json::Value;
use velox_core::{Backend, Error as CoreError};
use velox_frontend::image::{decode_data_url, preprocess, ImagePatches};
use velox_frontend::{parse_output, Frontend};
use velox_models::qwen3_5::{
    Checkpoint, EmbedInsert, Qwen35Text, Qwen35Vision, Session, Workspace,
};
use velox_serve::{ApiError, ChatEngine, Delta, Outcome};

pub use prompt::{build as build_prompt, ImageRun, Prompt};
pub use sampling::Sampler;

#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// Maximum tokens (prompt + completion) per request.
    pub capacity: usize,
    /// Prefill chunk size.
    pub chunk: usize,
    pub default_max_tokens: usize,
    /// Default for `chat_template_kwargs.enable_thinking` (ninfer `--no-thinking` = false).
    pub enable_thinking: bool,
    /// Recurrent-state checkpoints kept for prefix reuse.
    pub max_checkpoints: usize,
}

struct Lane<B: Backend> {
    session: Session<B>,
    ws: Workspace<B>,
    /// Keys of the tokens the session state has consumed, in order (see `Prompt::keys`).
    keys: Vec<u32>,
    checkpoints: Vec<Checkpoint<B>>,
    zero: Checkpoint<B>,
}

pub struct QwenEngine<B: Backend> {
    b: B,
    model: Qwen35Text<B>,
    vision: Option<Qwen35Vision<B>>,
    fe: Frontend,
    cfg: EngineConfig,
    lane: Mutex<Lane<B>>,
    special: SpecialTokens,
}

struct SpecialTokens {
    im_start: Option<u32>,
    vision_start: Option<u32>,
    image_pad: Option<u32>,
}

fn core(e: CoreError) -> ApiError {
    ApiError {
        status: 500,
        kind: "server_error",
        code: "engine_error",
        message: e.to_string(),
    }
}

fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

fn is_image_part(p: &Value) -> bool {
    p.get("image_url").is_some() || p.get("type").and_then(Value::as_str) == Some("image_url")
}

/// OpenAI messages -> template input: tool-call `arguments` strings become objects (the template
/// iterates them). Returns the image URLs in prompt order.
fn normalize_messages(messages: &Value) -> Result<(Value, Vec<String>), ApiError> {
    let mut out = messages.clone();
    let mut images = Vec::new();
    for m in out.as_array_mut().into_iter().flatten() {
        for p in m
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if is_image_part(p) {
                let url = p
                    .pointer("/image_url/url")
                    .or_else(|| p.get("image_url"))
                    .and_then(Value::as_str);
                images.push(
                    url.ok_or_else(|| ApiError::invalid("image_url part without a url"))?
                        .to_string(),
                );
            }
        }
        for call in m
            .get_mut("tool_calls")
            .and_then(Value::as_array_mut)
            .into_iter()
            .flatten()
        {
            if let Some(args) = call.pointer_mut("/function/arguments") {
                if let Some(s) = args.as_str() {
                    *args =
                        serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.to_string()));
                }
            }
        }
    }
    Ok((out, images))
}

fn single_token(fe: &Frontend, text: &str) -> Option<u32> {
    fe.encode(text).ok().filter(|v| v.len() == 1).map(|v| v[0])
}

impl<B: Backend> QwenEngine<B> {
    pub fn new(
        b: B,
        model: Qwen35Text<B>,
        vision: Option<Qwen35Vision<B>>,
        fe: Frontend,
        cfg: EngineConfig,
    ) -> Result<Self, CoreError> {
        let session = model.session(&b, cfg.capacity)?;
        let ws = model.workspace(&b, cfg.chunk)?;
        let zero = session.checkpoint(&b)?;
        let special = SpecialTokens {
            im_start: single_token(&fe, "<|im_start|>"),
            vision_start: single_token(&fe, "<|vision_start|>"),
            image_pad: single_token(&fe, "<|image_pad|>"),
        };
        let lane = Mutex::new(Lane {
            session,
            ws,
            keys: Vec::new(),
            checkpoints: Vec::new(),
            zero,
        });
        Ok(QwenEngine {
            b,
            model,
            vision,
            fe,
            cfg,
            lane,
            special,
        })
    }

    /// Positions worth checkpointing in `ids`: before the last image and before the generation
    /// prompt (the last `<|im_start|>`).
    fn checkpoint_positions(&self, ids: &[u32], after: usize) -> Vec<usize> {
        let last = |tok: Option<u32>| tok.and_then(|t| ids.iter().rposition(|&x| x == t));
        let mut v: Vec<usize> = [last(self.special.vision_start), last(self.special.im_start)]
            .into_iter()
            .flatten()
            .filter(|&p| p > after && p < ids.len())
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Restores the best reusable state for `keys` and returns how many tokens are already in it.
    fn reuse(&self, lane: &mut Lane<B>, keys: &[u32]) -> Result<usize, CoreError> {
        // At least the last prompt token must be fed to produce logits.
        let target = common_prefix(&lane.keys, keys).min(keys.len() - 1);
        if lane.session.pos == lane.keys.len() && lane.keys.len() <= target {
            return Ok(lane.keys.len());
        }
        let best = lane
            .checkpoints
            .iter()
            .filter(|c| c.pos <= target)
            .max_by_key(|c| c.pos);
        match best {
            Some(cp) => lane.session.restore(&self.b, cp)?,
            None => lane.session.restore(&self.b, &lane.zero)?,
        }
        Ok(lane.session.pos)
    }

    fn load_images(&self, urls: &[String]) -> Result<Vec<ImagePatches>, ApiError> {
        if urls.is_empty() {
            return Ok(Vec::new());
        }
        let (Some(_), Some((min_px, max_px))) = (&self.vision, self.fe.vision_pixels) else {
            return Err(ApiError::invalid(
                "this model was loaded without vision; images are not accepted",
            ));
        };
        urls.iter()
            .map(|u| {
                let (rgb, w, h) =
                    decode_data_url(u).map_err(|e| ApiError::invalid(e.to_string()))?;
                preprocess(&rgb, w, h, min_px, max_px).map_err(|e| ApiError::invalid(e.to_string()))
            })
            .collect()
    }

    /// Prefills `p.ids[start..]` (checkpointing on the way) and returns the last logits.
    fn prefill(
        &self,
        lane: &mut Lane<B>,
        p: &Prompt,
        images: &[ImagePatches],
        start: usize,
    ) -> Result<Vec<f32>, CoreError> {
        // Vision embeddings for images not already in the reused prefix.
        let mut embeds: Vec<(usize, B::Buffer)> = Vec::new();
        for (r, run) in p.runs.iter().enumerate() {
            if run.start + run.len > start {
                let vision = self
                    .vision
                    .as_ref()
                    .ok_or_else(|| CoreError::Unsupported("no vision tower".into()))?;
                let img = &images[run.image];
                embeds.push((
                    r,
                    vision.encode(&self.b, &img.data, img.grid_h, img.grid_w)?,
                ));
            }
        }
        lane.session.rope_delta = p.rope_delta;
        let mut pos = start;
        let mut logits = Vec::new();
        for cp in self
            .checkpoint_positions(&p.ids, start)
            .into_iter()
            .chain(std::iter::once(p.ids.len()))
        {
            if cp > pos {
                let inserts: Vec<EmbedInsert<'_, B::Buffer>> = embeds
                    .iter()
                    .filter_map(|(r, buf)| {
                        let run = &p.runs[*r];
                        let (lo, hi) = (run.start.max(pos), (run.start + run.len).min(cp));
                        (lo < hi).then(|| EmbedInsert {
                            token: lo - pos,
                            rows: hi - lo,
                            buf,
                            row_offset: lo - run.start,
                        })
                    })
                    .collect();
                let positions = p.positions.as_ref().map(|v| &v[pos..cp]);
                logits = self.model.forward_with(
                    &self.b,
                    &mut lane.session,
                    &mut lane.ws,
                    &p.ids[pos..cp],
                    positions,
                    &inserts,
                )?;
                pos = cp;
            }
            if cp < p.ids.len() {
                lane.checkpoints.push(lane.session.checkpoint(&self.b)?);
            }
        }
        Ok(logits)
    }
}

impl<B> ChatEngine for QwenEngine<B>
where
    B: Backend + Send + Sync,
    B::Buffer: Send + Sync,
{
    fn architecture(&self) -> &str {
        "Qwen3_5ForCausalLM"
    }

    fn generate(&self, req: &Value, emit: &mut dyn FnMut(Delta)) -> Result<Outcome, ApiError> {
        self.b.bind_thread().map_err(core)?;
        let (messages, urls) = normalize_messages(&req["messages"])?;
        let tools = req
            .get("tools")
            .filter(|t| t.as_array().is_some_and(|a| !a.is_empty()));
        let thinking = req
            .pointer("/chat_template_kwargs/enable_thinking")
            .and_then(Value::as_bool)
            .unwrap_or(self.cfg.enable_thinking);
        let text = self
            .fe
            .template
            .render(&messages, tools, thinking, true)
            .map_err(|e| ApiError::invalid(e.to_string()))?;
        let ids = self.fe.encode(&text).map_err(core)?;
        let images = self.load_images(&urls)?;
        let p = match self.special.image_pad {
            Some(pad) => build_prompt(&ids, pad, &images).map_err(ApiError::invalid)?,
            None if images.is_empty() => {
                build_prompt(&ids, u32::MAX, &[]).map_err(ApiError::invalid)?
            }
            None => return Err(ApiError::invalid("tokenizer has no <|image_pad|> token")),
        };
        let max_tokens = req
            .get("max_completion_tokens")
            .or_else(|| req.get("max_tokens"))
            .and_then(Value::as_u64)
            .map_or(self.cfg.default_max_tokens, |n| n as usize)
            .max(1);
        if p.ids.len() + max_tokens > self.cfg.capacity {
            return Err(ApiError {
                status: 400,
                kind: "invalid_request_error",
                code: "context_length_exceeded",
                message: format!(
                    "{} prompt + {} completion tokens exceed the context of {}",
                    p.ids.len(),
                    max_tokens,
                    self.cfg.capacity
                ),
            });
        }
        let mut sampler = Sampler::from_request(req);

        let mut lane = self
            .lane
            .lock()
            .map_err(|_| core(CoreError::Backend("engine lane poisoned".into())))?;
        let lane = &mut *lane;
        let t0 = Instant::now();
        let start = self.reuse(lane, &p.keys).map_err(core)?;
        // Checkpoints still valid for this prompt are those inside the shared prefix.
        lane.checkpoints.retain(|c| c.pos <= start);
        lane.keys.truncate(start);
        let prefilled = self.prefill(lane, &p, &images, start);
        let mut logits = match prefilled {
            Ok(l) => l,
            Err(e) => {
                // The state is partially advanced; forget it so the next request starts clean.
                lane.keys.clear();
                lane.checkpoints.clear();
                return Err(core(e));
            }
        };
        lane.keys.extend_from_slice(&p.keys[start..]);
        while lane.checkpoints.len() > self.cfg.max_checkpoints {
            lane.checkpoints.remove(0);
        }
        let prompt_ms = t0.elapsed().as_secs_f64() * 1e3;

        let t1 = Instant::now();
        let mut out = Vec::new();
        let mut finish = "length";
        loop {
            let next = sampler.sample(&logits, self.fe.public_vocab);
            if self.fe.is_eos(next) {
                finish = "stop";
                break;
            }
            out.push(next);
            if out.len() >= max_tokens {
                break;
            }
            logits = self
                .model
                .forward(&self.b, &mut lane.session, &mut lane.ws, &[next])
                .map_err(core)?;
            lane.keys.push(next);
        }
        let predicted_ms = t1.elapsed().as_secs_f64() * 1e3;

        let text = self.fe.decode(&out).map_err(core)?;
        let parsed = parse_output(&text, tools);
        if !parsed.content.is_empty() {
            emit(Delta::Content(parsed.content));
        }
        let has_calls = !parsed.tool_calls.is_empty();
        for call in parsed.tool_calls {
            emit(Delta::ToolCall(call));
        }
        Ok(Outcome {
            finish_reason: if has_calls {
                "tool_calls".into()
            } else {
                finish.into()
            },
            prompt_tokens: p.ids.len() as u64,
            cached_tokens: start as u64,
            completion_tokens: out.len() as u64,
            prompt_ms,
            predicted_ms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tool_call_arguments_become_objects() {
        let m = json!([{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f", "arguments": "{\"x\": 1}"}}]}]);
        let (n, imgs) = normalize_messages(&m).unwrap();
        assert_eq!(
            n[0]["tool_calls"][0]["function"]["arguments"],
            json!({"x": 1})
        );
        assert!(imgs.is_empty());
    }

    #[test]
    fn images_are_collected_in_order() {
        let m = json!([
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:a"}}, {"type": "text", "text": "x"}]},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:b"}}]}
        ]);
        assert_eq!(normalize_messages(&m).unwrap().1, vec!["data:a", "data:b"]);
    }

    #[test]
    fn prefix_length() {
        assert_eq!(common_prefix(&[1, 2, 3], &[1, 2, 4, 5]), 2);
        assert_eq!(common_prefix(&[], &[1]), 0);
    }
}
