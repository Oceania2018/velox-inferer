//! Chat engine for Qwen3.5 text models behind the OpenAI-compatible server.
//!
//! One generation lane. Agent loops resend almost the same prompt every step, so the lane keeps
//! the token sequence its state holds plus recurrent-state checkpoints taken where the next
//! request is likely to diverge (before the last image and before the generation prompt). A new
//! request restores the latest checkpoint inside the common prefix and prefills only the rest.

mod sampling;

use std::sync::Mutex;
use std::time::Instant;

use serde_json::Value;
use velox_core::{Backend, Error as CoreError};
use velox_frontend::{parse_output, Frontend};
use velox_models::qwen3_5::{Checkpoint, Qwen35Text, Session, Workspace};
use velox_serve::{ApiError, ChatEngine, Delta, Outcome};

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
    /// Tokens the session state has consumed, in order.
    keys: Vec<u32>,
    checkpoints: Vec<Checkpoint<B>>,
    zero: Checkpoint<B>,
}

pub struct QwenEngine<B: Backend> {
    b: B,
    model: Qwen35Text<B>,
    fe: Frontend,
    cfg: EngineConfig,
    lane: Mutex<Lane<B>>,
    special: SpecialTokens,
}

struct SpecialTokens {
    im_start: Option<u32>,
    vision_start: Option<u32>,
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

/// OpenAI messages -> template input: tool-call `arguments` strings become objects (the template
/// iterates them), and image parts are rejected until the vision path lands.
fn normalize_messages(messages: &Value) -> Result<Value, ApiError> {
    let mut out = messages.clone();
    for m in out.as_array_mut().into_iter().flatten() {
        if let Some(parts) = m.get("content").and_then(Value::as_array) {
            if parts.iter().any(|p| {
                p.get("image_url").is_some()
                    || p.get("type").and_then(Value::as_str) == Some("image_url")
            }) {
                return Err(ApiError::not_implemented(
                    "image inputs are not supported by velox-inferer yet (vision is in progress)",
                ));
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
    Ok(out)
}

impl<B: Backend> QwenEngine<B> {
    pub fn new(
        b: B,
        model: Qwen35Text<B>,
        fe: Frontend,
        cfg: EngineConfig,
    ) -> Result<Self, CoreError> {
        let session = model.session(&b, cfg.capacity)?;
        let ws = model.workspace(&b, cfg.chunk)?;
        let zero = session.checkpoint(&b)?;
        let special = SpecialTokens {
            im_start: fe
                .encode("<|im_start|>")
                .ok()
                .filter(|v| v.len() == 1)
                .map(|v| v[0]),
            vision_start: fe
                .encode("<|vision_start|>")
                .ok()
                .filter(|v| v.len() == 1)
                .map(|v| v[0]),
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

    /// Restores the best reusable state for `ids` and returns how many tokens are already in it.
    fn reuse(&self, lane: &mut Lane<B>, ids: &[u32]) -> Result<usize, CoreError> {
        // At least the last prompt token must be fed to produce logits.
        let target = common_prefix(&lane.keys, ids).min(ids.len() - 1);
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
        let messages = normalize_messages(&req["messages"])?;
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
        let max_tokens = req
            .get("max_completion_tokens")
            .or_else(|| req.get("max_tokens"))
            .and_then(Value::as_u64)
            .map_or(self.cfg.default_max_tokens, |n| n as usize)
            .max(1);
        if ids.len() + max_tokens > self.cfg.capacity {
            return Err(ApiError {
                status: 400,
                kind: "invalid_request_error",
                code: "context_length_exceeded",
                message: format!(
                    "{} prompt + {} completion tokens exceed the context of {}",
                    ids.len(),
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
        let start = self.reuse(lane, &ids).map_err(core)?;
        // Checkpoints still valid for this prompt are those inside the shared prefix.
        lane.checkpoints.retain(|c| c.pos <= start);
        lane.keys.truncate(start);
        let mut pos = start;
        let mut logits = Vec::new();
        for cp in self
            .checkpoint_positions(&ids, start)
            .into_iter()
            .chain(std::iter::once(ids.len()))
        {
            if cp > pos {
                logits = self
                    .model
                    .forward(&self.b, &mut lane.session, &mut lane.ws, &ids[pos..cp])
                    .map_err(core)?;
                pos = cp;
            }
            if cp < ids.len() {
                lane.checkpoints
                    .push(lane.session.checkpoint(&self.b).map_err(core)?);
            }
        }
        lane.keys.extend_from_slice(&ids[start..]);
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
            prompt_tokens: ids.len() as u64,
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
        let n = normalize_messages(&m).unwrap();
        assert_eq!(
            n[0]["tool_calls"][0]["function"]["arguments"],
            json!({"x": 1})
        );
    }

    #[test]
    fn images_are_rejected_until_vision() {
        let m = json!([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:..."}}]}]);
        assert_eq!(normalize_messages(&m).unwrap_err().status, 501);
    }

    #[test]
    fn prefix_length() {
        assert_eq!(common_prefix(&[1, 2, 3], &[1, 2, 4, 5]), 2);
        assert_eq!(common_prefix(&[], &[1]), 0);
    }
}
