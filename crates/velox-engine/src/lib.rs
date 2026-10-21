//! Chat engine for Qwen3.5 text + vision models behind the OpenAI-compatible server.
//!
//! One generation lane. Agent loops resend almost the same prompt every step, so the lane keeps
//! the keys of the tokens its state holds plus recurrent-state checkpoints taken where the next
//! request is likely to diverge (before the last image and before the generation prompt). A new
//! request restores the latest checkpoint inside the common prefix and prefills only the rest;
//! the vision tower runs only for images in that remainder.

mod prompt;
mod sampling;
pub mod spec;

use spec::{DFlashDrafter, Drafter, MtpDrafter};

use std::sync::Mutex;
use std::time::Instant;

use serde_json::{json, Value};
use velox_core::{Backend, Error as CoreError, KvFormat};
use velox_frontend::image::{decode_data_url, preprocess, ImagePatches};
use velox_frontend::{parse_output, Frontend};
use velox_models::qwen3_5::{
    Checkpoint, DFlashModel, EmbedInsert, Qwen35Text, Qwen35Vision, Session, Workspace,
};
use velox_serve::{ApiError, ChatEngine, Delta, Outcome};

pub use prompt::{build as build_prompt, ImageRun, Prompt};
pub use sampling::{Sampler, CANDIDATES};

#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// Maximum tokens (prompt + completion) per request.
    pub capacity: usize,
    /// Prefill chunk size.
    pub chunk: usize,
    pub default_max_tokens: usize,
    /// Default for `chat_template_kwargs.enable_thinking` (ninfer `--no-thinking` = false).
    pub enable_thinking: bool,
    /// Recurrent-state checkpoints kept per lane for prefix reuse.
    pub max_checkpoints: usize,
    /// Keep checkpoints in system RAM (pinned, device-mapped) rather than device memory.
    pub host_checkpoints: bool,
    /// Take no checkpoints after the prompt's last image: agent screenshots are replaced in the
    /// next request's history, so state past them is never shared (ninfer's media-tail skip).
    pub skip_media_tail: bool,
    /// Capacity of a second, small lane for requests that share nothing with the main lane
    /// (probes, side questions), so they do not evict the agent's context. 0 disables it.
    pub scratch_capacity: usize,
    /// MTP drafts per speculative round on the main lane (0: plain decoding).
    pub mtp_drafts: usize,
    /// DFlash2 drafts per block on the main lane when the engine has a DFlash2 model (0: use MTP
    /// instead).
    pub dflash_drafts: usize,
    /// Storage format of the target model's attention KV caches.
    pub kv_format: KvFormat,
}

/// A sequence slot: its own KV cache and recurrent state, the keys of the tokens it holds, and
/// checkpoints of its recurrent state.
struct Lane<B: Backend> {
    session: Session<B>,
    /// Keys of the tokens the session state has consumed, in order (see `Prompt::keys`).
    keys: Vec<u32>,
    checkpoints: Vec<Checkpoint<B>>,
    /// Storage of dropped checkpoints, reused for new ones (allocating and freeing them would
    /// synchronize the device): device storage here, system-RAM storage in `host_spare`.
    spare: Vec<Checkpoint<B>>,
    host_spare: Vec<Checkpoint<B>>,
    zero: Checkpoint<B>,
    /// Draft model for speculative decoding (main lane only).
    drafter: Option<Drafter<B>>,
    /// Speculative rounds of the current request: (rounds, drafted, accepted).
    spec_stats: (usize, usize, usize),
}

/// All lanes plus the activation workspace they share (requests run one at a time).
struct Lanes<B: Backend> {
    ws: Workspace<B>,
    lanes: Vec<Lane<B>>,
}

/// Checkpoint storage kept for reuse per lane beyond the live checkpoints.
const MAX_SPARE: usize = 4;

/// Below this much shared prefix a request counts as unrelated to a lane.
const MIN_REUSE: usize = 64;

pub struct QwenEngine<B: Backend> {
    b: B,
    model: Qwen35Text<B>,
    vision: Option<Qwen35Vision<B>>,
    /// DFlash2 block drafter weights, when loaded.
    dflash: Option<DFlashModel<B>>,
    fe: Frontend,
    cfg: EngineConfig,
    state: Mutex<Lanes<B>>,
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

/// A request turned into chat-template input exactly as ninfer builds it, so both engines render
/// identical prompts for the same request:
/// - messages become `{role, content}`: text-only content is one string (parts joined without a
///   separator); content with images stays a list of `{type: text, text}` / `{type: image}` in
///   order; non-empty `reasoning_content`, `tool_calls` (`{id, type, function: {name, arguments}}`
///   with arguments parsed, `{}` when empty) and `tool_call_id` are kept; `name` is dropped;
/// - each tool is rebuilt as `{type: function, function: {name, parameters, strict,
///   description}}` with `parameters` defaulting to an empty object schema and `strict` to false.
pub struct Normalized {
    pub messages: Value,
    pub tools: Option<Value>,
    /// Image URLs in prompt order.
    pub images: Vec<String>,
}

pub fn normalize_request(req: &Value) -> Result<Normalized, ApiError> {
    let src = req
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| ApiError::invalid("messages must be an array"))?;
    let mut messages = Vec::with_capacity(src.len());
    let mut images = Vec::new();
    for m in src {
        let role = m
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| ApiError::invalid("message without a role"))?;
        let mut out = serde_json::Map::new();
        out.insert("role".into(), role.into());
        let content = match m.get("content") {
            None | Some(Value::Null) => Value::String(String::new()),
            Some(Value::String(s)) => Value::String(s.clone()),
            Some(Value::Array(parts)) if parts.iter().any(is_image_part) => {
                let mut items = Vec::new();
                for p in parts {
                    if is_image_part(p) {
                        let url = p
                            .pointer("/image_url/url")
                            .or_else(|| p.get("image_url"))
                            .and_then(Value::as_str);
                        images.push(
                            url.ok_or_else(|| ApiError::invalid("image_url part without a url"))?
                                .to_string(),
                        );
                        items.push(json!({"type": "image"}));
                    } else if let Some(t) = p.get("text").and_then(Value::as_str) {
                        items.push(json!({"type": "text", "text": t}));
                    }
                }
                Value::Array(items)
            }
            Some(Value::Array(parts)) => Value::String(
                parts
                    .iter()
                    .filter_map(|p| p.get("text").and_then(Value::as_str))
                    .collect(),
            ),
            Some(other) => {
                return Err(ApiError::invalid(format!(
                    "unsupported message content {other}"
                )))
            }
        };
        out.insert("content".into(), content);
        let reasoning = m
            .get("reasoning_content")
            .or_else(|| m.get("reasoning"))
            .and_then(Value::as_str)
            .unwrap_or("");
        if !reasoning.is_empty() {
            out.insert("reasoning_content".into(), reasoning.into());
        }
        if let Some(calls) = m
            .get("tool_calls")
            .and_then(Value::as_array)
            .filter(|c| !c.is_empty())
        {
            let calls: Vec<Value> = calls
                .iter()
                .map(|c| {
                    let f = c.get("function").unwrap_or(c);
                    let args = match f.get("arguments") {
                        Some(Value::String(s)) if s.trim().is_empty() => json!({}),
                        Some(Value::String(s)) => serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.clone())),
                        Some(Value::Null) | None => json!({}),
                        Some(v) => v.clone(),
                    };
                    json!({"id": c.get("id").cloned().unwrap_or_else(|| "".into()), "type": "function",
                           "function": {"name": f.get("name").cloned().unwrap_or_else(|| "".into()), "arguments": args}})
                })
                .collect();
            out.insert("tool_calls".into(), Value::Array(calls));
        }
        if let Some(id) = m
            .get("tool_call_id")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            out.insert("tool_call_id".into(), id.into());
        }
        messages.push(Value::Object(out));
    }
    let tools = match req.get("tools").and_then(Value::as_array) {
        Some(list) if !list.is_empty() => Some(Value::Array(
            list.iter()
                .map(|t| {
                    let f = t.get("function").unwrap_or(t);
                    let mut function = serde_json::Map::new();
                    function.insert(
                        "name".into(),
                        f.get("name").cloned().unwrap_or_else(|| "".into()),
                    );
                    let params = f
                        .get("parameters")
                        .filter(|p| !p.is_null())
                        .cloned()
                        .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
                    function.insert("parameters".into(), params);
                    function.insert(
                        "strict".into(),
                        f.get("strict")
                            .and_then(Value::as_bool)
                            .unwrap_or(false)
                            .into(),
                    );
                    if let Some(d) = f
                        .get("description")
                        .and_then(Value::as_str)
                        .filter(|d| !d.is_empty())
                    {
                        function.insert("description".into(), d.into());
                    }
                    json!({"type": "function", "function": function})
                })
                .collect(),
        )),
        _ => None,
    };
    Ok(Normalized {
        messages: Value::Array(messages),
        tools,
        images,
    })
}

fn single_token(fe: &Frontend, text: &str) -> Option<u32> {
    fe.encode(text).ok().filter(|v| v.len() == 1).map(|v| v[0])
}

impl<B: Backend> QwenEngine<B> {
    pub fn new(
        b: B,
        model: Qwen35Text<B>,
        vision: Option<Qwen35Vision<B>>,
        dflash: Option<DFlashModel<B>>,
        fe: Frontend,
        cfg: EngineConfig,
    ) -> Result<Self, CoreError> {
        let dflash = dflash.filter(|_| cfg.dflash_drafts > 0);
        let mut ws = model.workspace(&b, cfg.chunk)?;
        if let Some(dm) = &dflash {
            model.enable_capture(&b, &mut ws, &dm.cfg.target_layers)?;
        }
        let mut lanes = Vec::new();
        for (i, capacity) in [cfg.capacity, cfg.scratch_capacity.min(cfg.capacity)]
            .into_iter()
            .enumerate()
        {
            if capacity == 0 {
                continue;
            }
            let session = model.session_with(&b, capacity, cfg.kv_format)?;
            let zero = session.checkpoint_in(&b, cfg.host_checkpoints)?;
            let drafter = match &dflash {
                // Every lane drafts: short unrelated requests (System One, probes) land on the scratch lane.
                Some(dm) => Some(Drafter::DFlash(DFlashDrafter::new(
                    &b,
                    dm,
                    &model,
                    cfg.chunk,
                    cfg.dflash_drafts,
                )?)),
                None if i == 0 && cfg.mtp_drafts > 0 && model.has_mtp() => Some(Drafter::Mtp(
                    MtpDrafter::new(&b, &model, capacity, cfg.chunk, cfg.mtp_drafts)?,
                )),
                _ => None,
            };
            lanes.push(Lane {
                session,
                keys: Vec::new(),
                checkpoints: Vec::new(),
                spare: Vec::new(),
                host_spare: Vec::new(),
                spec_stats: (0, 0, 0),
                zero,
                drafter,
            });
        }
        let special = SpecialTokens {
            im_start: single_token(&fe, "<|im_start|>"),
            vision_start: single_token(&fe, "<|vision_start|>"),
            image_pad: single_token(&fe, "<|image_pad|>"),
        };
        let state = Mutex::new(Lanes { ws, lanes });
        Ok(QwenEngine {
            b,
            model,
            vision,
            dflash,
            fe,
            cfg,
            state,
            special,
        })
    }

    /// The lane for a prompt of `keys` needing `needed` tokens of capacity: the one that can reuse
    /// the most, or, when no lane shares a meaningful prefix, the smallest lane that fits, so
    /// unrelated short requests do not evict a long context.
    fn pick_lane(lanes: &[Lane<B>], keys: &[u32], needed: usize) -> Option<usize> {
        let fits: Vec<usize> = (0..lanes.len())
            .filter(|&i| lanes[i].session.capacity >= needed)
            .collect();
        let (best, shared) = fits
            .iter()
            .map(|&i| (i, common_prefix(&lanes[i].keys, keys)))
            .max_by_key(|&(_, s)| s)?;
        if shared >= MIN_REUSE {
            return Some(best);
        }
        fits.into_iter().min_by_key(|&i| lanes[i].session.capacity)
    }

    /// Positions worth checkpointing in `ids` after `after`: every turn boundary (`<|im_start|>`,
    /// where agent loops rewrite history) and the last image.
    fn checkpoint_positions(&self, ids: &[u32], after: usize) -> Vec<usize> {
        let image = self
            .special
            .vision_start
            .and_then(|t| ids.iter().rposition(|&x| x == t));
        let turns = ids
            .iter()
            .enumerate()
            .filter(|&(_, &t)| Some(t) == self.special.im_start)
            .map(|(i, _)| i);
        let tail = match image {
            Some(p) if self.cfg.skip_media_tail => p,
            _ => usize::MAX,
        };
        let mut v: Vec<usize> = turns
            .chain(image)
            .filter(|&p| p > after && p < ids.len() && p <= tail)
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Keeps at most `max` checkpoints, dropping the one closest to its predecessor so the rest
    /// stay spread over the prefix (the oldest and newest are always kept). Checkpoints at or
    /// after `fresh` (taken by the latest prefill, the likeliest to be reused next) go last.
    /// Dropped storage goes to `spare`.
    fn prune_checkpoints(
        checkpoints: &mut Vec<Checkpoint<B>>,
        spare: &mut Vec<Checkpoint<B>>,
        max: usize,
        fresh: usize,
    ) {
        checkpoints.sort_by_key(|c| c.pos);
        while checkpoints.len() > max.max(1) {
            let gap = |i: &usize| checkpoints[*i].pos - checkpoints[i - 1].pos;
            let inner = 1..checkpoints.len() - 1;
            let drop = inner
                .clone()
                .filter(|&i| checkpoints[i].pos < fresh)
                .min_by_key(gap)
                .or_else(|| inner.min_by_key(gap))
                .unwrap_or(0);
            spare.push(checkpoints.remove(drop));
        }
        spare.truncate(MAX_SPARE);
    }

    /// Host checkpoint mode: the latest prefill's checkpoints (the next request's likeliest
    /// restore points) stay on the device; of the older ones only the two earliest (system prompt
    /// and first turn: what a new conversation shares) are kept, moved to system RAM once; the
    /// rest are dropped. Device memory then holds a few checkpoints however long the context.
    fn place_checkpoints(&self, lane: &mut Lane<B>, fresh: usize) -> Result<(), CoreError> {
        const HOST_ANCHORS: usize = 2;
        // Pruning returns storage of either kind to `spare`; snapshots must get device storage.
        let (host, device): (Vec<_>, Vec<_>) = lane.spare.drain(..).partition(|c| c.on_host());
        lane.spare = device;
        lane.host_spare.extend(host);
        lane.checkpoints.sort_by_key(|c| c.pos);
        let mut kept = Vec::with_capacity(lane.checkpoints.len());
        let mut anchors = 0;
        for cp in lane.checkpoints.drain(..) {
            if cp.pos >= fresh {
                kept.push(cp);
            } else if anchors < HOST_ANCHORS {
                anchors += 1;
                if cp.on_host() {
                    kept.push(cp);
                } else {
                    let mut h = match lane.host_spare.pop() {
                        Some(h) => h,
                        None => lane.session.blank_checkpoint_in(&self.b, cp.pos, true)?,
                    };
                    cp.copy_into(&self.b, &mut h)?;
                    kept.push(h);
                    lane.spare.push(cp);
                }
            } else if cp.on_host() {
                lane.host_spare.push(cp);
            } else {
                lane.spare.push(cp);
            }
        }
        lane.checkpoints = kept;
        lane.spare.truncate(MAX_SPARE);
        lane.host_spare.truncate(MAX_SPARE);
        Ok(())
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
    /// Speculative decoding on a lane with an MTP drafter, from the candidates after the prompt.
    /// Each round verifies `[pending, drafts..]`, accepts drafts by speculative sampling (exact
    /// match when greedy), commits the fed prefix and drafts again. Returns the finish reason.
    fn decode_speculative(
        &self,
        lane: &mut Lane<B>,
        ws: &mut Workspace<B>,
        sampler: &mut Sampler,
        cands: Vec<(u32, f32)>,
        max_tokens: usize,
        out: &mut Vec<u32>,
    ) -> Result<&'static str, ApiError> {
        let limit = self.fe.public_vocab;
        let Lane {
            session,
            keys,
            drafter,
            spec_stats,
            ..
        } = lane;
        let dr = drafter.as_mut().expect("caller checked");
        let (b, model) = (&self.b, &self.model);
        let mut c = sampler.sample(&cands);
        // Rows of the previous verify still to be paired in MTP (none: the prefill carry).
        let (mut row0, mut kept_next) = (0usize, Vec::new());
        let mut first = true;
        loop {
            if self.fe.is_eos(c) {
                return Ok("stop");
            }
            out.push(c);
            sampler.observe(c);
            if out.len() >= max_tokens {
                return Ok("length");
            }
            let (drafts, qs) = match dr {
                Drafter::Mtp(m) => (
                    m.draft(b, model, session, ws, row0, &kept_next, c, limit)
                        .map_err(core)?,
                    None,
                ),
                Drafter::DFlash(d) => {
                    let dm = self
                        .dflash
                        .as_ref()
                        .expect("a DFlash drafter has its model");
                    let kept = if first { 0 } else { kept_next.len() + 1 };
                    let temperature = sampler.temperature();
                    let (drafts, qs) = d
                        .draft(
                            b,
                            dm,
                            model,
                            ws,
                            row0,
                            kept,
                            session.pos,
                            c,
                            limit,
                            temperature,
                        )
                        .map_err(core)?;
                    (drafts, Some(qs))
                }
            };
            first = false;
            let mut window = vec![c];
            window.extend(drafts.into_iter().take(max_tokens - out.len()));
            let p = session.pos;
            let rows = model
                .verify(b, session, ws, &window, CANDIDATES, limit)
                .map_err(core)?;
            let mut keep = 1;
            let mut finish = None;
            let mut next = None;
            for j in 0..window.len() {
                if j + 1 == window.len() {
                    next = Some(sampler.sample(&rows[j]));
                    break;
                }
                let verdict = match &qs {
                    Some(q) => sampler.verify_draft_q(&rows[j], window[j + 1], &q[j]),
                    None => sampler.verify_draft(&rows[j], window[j + 1]),
                };
                match verdict {
                    Ok(()) => {
                        keep += 1;
                        let d = window[j + 1];
                        if self.fe.is_eos(d) {
                            finish = Some("stop");
                            break;
                        }
                        out.push(d);
                        sampler.observe(d);
                        if out.len() >= max_tokens {
                            finish = Some("length");
                            break;
                        }
                    }
                    Err(t) => {
                        next = Some(t);
                        break;
                    }
                }
            }
            spec_stats.0 += 1;
            spec_stats.1 += window.len() - 1;
            spec_stats.2 += keep - 1;
            model.commit(b, session, ws, keep).map_err(core)?;
            match dr {
                Drafter::Mtp(m) => m.record(keep - 1, window.len() - 1),
                Drafter::DFlash(d) => d.observe(keep - 1),
            }
            keys.extend_from_slice(&window[..keep]);
            if let Some(f) = finish {
                return Ok(f);
            }
            row0 = p;
            kept_next = window[1..keep].to_vec();
            c = next.expect("a round ends with a sampled token");
        }
    }

    fn prefill(
        &self,
        lane: &mut Lane<B>,
        ws: &mut Workspace<B>,
        p: &Prompt,
        images: &[ImagePatches],
        start: usize,
    ) -> Result<(), CoreError> {
        let debug = std::env::var_os("VELOX_DEBUG_PREFIX").is_some();
        let t0 = Instant::now();
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
        if debug {
            self.b.synchronize()?;
        }
        let vision_ms = t0.elapsed().as_secs_f64() * 1e3;
        let t1 = Instant::now();
        lane.session.rope_delta = p.rope_delta;
        // Checkpoints are filled inside the forward pass, so the prompt is not cut into short
        // segments there (each segment would stream every weight once more).
        let mut snaps = Vec::new();
        for pos in self.checkpoint_positions(&p.ids, start) {
            let mut cp = match lane.spare.pop() {
                Some(cp) => cp,
                // Snapshots are taken on the device (fast); see `Self::place_checkpoints`.
                None => lane.session.blank_checkpoint(&self.b, pos)?,
            };
            cp.pos = pos;
            cp.rope_delta = p.rope_delta;
            snaps.push(cp);
        }
        let inserts: Vec<EmbedInsert<'_, B::Buffer>> = embeds
            .iter()
            .filter_map(|(r, buf)| {
                let run = &p.runs[*r];
                let lo = run.start.max(start);
                (lo < run.start + run.len).then(|| EmbedInsert {
                    token: lo - start,
                    rows: run.start + run.len - lo,
                    buf,
                    row_offset: lo - run.start,
                })
            })
            .collect();
        let ids = &p.ids[start..];
        let positions = p.positions.as_ref().map(|v| &v[start..]);
        let Lane {
            session, drafter, ..
        } = &mut *lane;
        match drafter.as_mut() {
            Some(dr) => {
                let (row0, delta) = (session.pos, session.rope_delta);
                let model = &self.model;
                let dflash = self.dflash.as_ref();
                let b = &self.b;
                model.advance_rows(
                    b,
                    session,
                    ws,
                    ids,
                    positions,
                    &inserts,
                    &mut snaps,
                    &mut |ws, base, t| {
                        let rows = positions.map(|v| &v[base..base + t]);
                        match &mut *dr {
                            Drafter::Mtp(m) => {
                                m.prefill_chunk(b, model, ws, row0 + base, t, delta, rows)
                            }
                            Drafter::DFlash(d) => d.prefill_chunk(
                                b,
                                dflash.expect("a DFlash drafter has its model"),
                                ws,
                                row0 + base,
                                t,
                            ),
                        }
                    },
                )?;
            }
            None => self
                .model
                .advance_snap(&self.b, session, ws, ids, positions, &inserts, &mut snaps)?,
        }
        lane.checkpoints.extend(snaps);
        if debug {
            self.b.synchronize()?;
            eprintln!(
                "[prefill] {} images {vision_ms:.0} ms, {} tokens from {start} in {:.0} ms",
                embeds.len(),
                p.ids.len() - start,
                t1.elapsed().as_secs_f64() * 1e3
            );
        }
        Ok(())
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
        let t_req = Instant::now();
        self.b.bind_thread().map_err(core)?;
        let Normalized {
            messages,
            tools,
            images: urls,
        } = normalize_request(req)?;
        let tools = tools.as_ref();
        let thinking = req
            .pointer("/chat_template_kwargs/enable_thinking")
            .and_then(Value::as_bool)
            .unwrap_or(self.cfg.enable_thinking);
        let text = self
            .fe
            .template
            .render(&messages, tools, thinking, true)
            .map_err(|e| ApiError::invalid(e.to_string()))?;
        let t_tpl = t_req.elapsed();
        let ids = self.fe.encode(&text).map_err(core)?;
        let t_tok = t_req.elapsed();
        let images = self.load_images(&urls)?;
        if std::env::var_os("VELOX_DEBUG_PREFIX").is_some() {
            eprintln!(
                "[request] template {:.1} ms, tokenize {:.1} ms ({} ids), images {:.1} ms ({})",
                t_tpl.as_secs_f64() * 1e3,
                (t_tok - t_tpl).as_secs_f64() * 1e3,
                ids.len(),
                (t_req.elapsed() - t_tok).as_secs_f64() * 1e3,
                images.len()
            );
        }
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
        let mut sampler = Sampler::from_request(req, thinking);

        let mut state = self
            .state
            .lock()
            .map_err(|_| core(CoreError::Backend("engine lanes poisoned".into())))?;
        let Lanes { ws, lanes } = &mut *state;
        let li = Self::pick_lane(lanes, &p.keys, p.ids.len() + max_tokens)
            .ok_or_else(|| ApiError::invalid("no lane can hold this request"))?;
        let lane = &mut lanes[li];
        let t0 = Instant::now();
        if std::env::var_os("VELOX_DEBUG_PREFIX").is_some() {
            let shared = common_prefix(&lane.keys, &p.keys);
            let cps: Vec<usize> = lane.checkpoints.iter().map(|c| c.pos).collect();
            eprintln!(
                "[prefix] lane {li} prompt {} held {} pos {} shared {} checkpoints {:?} next {:?}",
                p.keys.len(),
                lane.keys.len(),
                lane.session.pos,
                shared,
                cps,
                self.checkpoint_positions(&p.ids, 0)
            );
        }
        match lane.drafter.as_mut() {
            Some(Drafter::Mtp(m)) => m.reset(),
            Some(Drafter::DFlash(d)) => d.reset(),
            None => {}
        }
        let start = self.reuse(lane, &p.keys).map_err(core)?;
        let t_reuse = t0.elapsed();
        // Checkpoints still valid for this prompt are those inside the shared prefix.
        let (keep, stale): (Vec<_>, Vec<_>) =
            lane.checkpoints.drain(..).partition(|c| c.pos <= start);
        lane.checkpoints = keep;
        for cp in stale {
            if cp.on_host() {
                lane.host_spare.push(cp);
            } else {
                lane.spare.push(cp);
            }
        }
        lane.keys.truncate(start);
        let mut t_fill = t_reuse;
        let prefilled = self.prefill(lane, ws, &p, &images, start).and_then(|()| {
            t_fill = t0.elapsed();
            self.model
                .candidates(&self.b, ws, CANDIDATES, self.fe.public_vocab)
        });
        let mut cands = match prefilled {
            Ok(c) => c,
            Err(e) => {
                // The state is partially advanced; forget it so the next request starts clean.
                lane.keys.clear();
                lane.checkpoints.clear();
                return Err(core(e));
            }
        };
        lane.keys.extend_from_slice(&p.keys[start..]);
        Self::prune_checkpoints(
            &mut lane.checkpoints,
            &mut lane.spare,
            self.cfg.max_checkpoints,
            start + 1,
        );
        if self.cfg.host_checkpoints {
            self.place_checkpoints(lane, start + 1).map_err(core)?;
        }
        let prompt_ms = t0.elapsed().as_secs_f64() * 1e3;
        if std::env::var_os("VELOX_DEBUG_PREFIX").is_some() {
            eprintln!(
                "[prompt] reuse {:.1} ms, prefill {:.1} ms, candidates+prune {:.1} ms, total {prompt_ms:.1} ms",
                t_reuse.as_secs_f64() * 1e3,
                (t_fill - t_reuse).as_secs_f64() * 1e3,
                (t0.elapsed() - t_fill).as_secs_f64() * 1e3
            );
        }

        let t1 = Instant::now();
        let mut out = Vec::new();
        let mut finish = "length";
        if lane.drafter.is_some() {
            lane.spec_stats = (0, 0, 0);
            finish =
                self.decode_speculative(lane, ws, &mut sampler, cands, max_tokens, &mut out)?;
        } else {
            loop {
                let next = sampler.sample(&cands);
                if self.fe.is_eos(next) {
                    finish = "stop";
                    break;
                }
                out.push(next);
                sampler.observe(next);
                if out.len() >= max_tokens {
                    break;
                }
                self.model
                    .advance(&self.b, &mut lane.session, ws, &[next], None, &[])
                    .and_then(|()| {
                        self.model
                            .candidates(&self.b, ws, CANDIDATES, self.fe.public_vocab)
                    })
                    .map(|c| cands = c)
                    .map_err(core)?;
                lane.keys.push(next);
            }
        }
        let predicted_ms = t1.elapsed().as_secs_f64() * 1e3;
        if std::env::var_os("VELOX_DEBUG_PREFIX").is_some() && lane.drafter.is_some() {
            let (rounds, drafted, accepted) = lane.spec_stats;
            eprintln!(
                "[decode] {} tokens in {predicted_ms:.0} ms, {rounds} rounds, accepted {accepted}/{drafted} drafts, {:.2} tokens/round",
                out.len(),
                out.len() as f64 / rounds.max(1) as f64
            );
        }

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
    fn messages_follow_ninfer_shape() {
        let r = json!({"messages": [
            {"role": "system", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}], "name": "x"},
            {"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{\"x\": 1}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
            {"role": "user", "content": [{"type": "text", "text": "see"}, {"type": "image_url", "image_url": {"url": "data:a"}}]}
        ]});
        let n = normalize_request(&r).unwrap();
        assert_eq!(n.messages[0], json!({"role": "system", "content": "ab"}));
        assert_eq!(
            n.messages[1]["tool_calls"][0],
            json!({"id": "c1", "type": "function", "function": {"name": "f", "arguments": {"x": 1}}})
        );
        assert_eq!(
            n.messages[2],
            json!({"role": "tool", "content": "ok", "tool_call_id": "c1"})
        );
        assert_eq!(
            n.messages[3]["content"],
            json!([{"type": "text", "text": "see"}, {"type": "image"}])
        );
        assert_eq!(n.images, vec!["data:a"]);
        assert!(n.tools.is_none());
    }

    #[test]
    fn tools_are_rebuilt_with_strict() {
        let r = json!({"messages": [], "tools": [
            {"type": "function", "function": {"name": "click", "description": "Click it.", "parameters": {"type": "object"}, "extra": 1}},
            {"type": "function", "function": {"name": "noop", "description": ""}}
        ]});
        let tools = normalize_request(&r).unwrap().tools.unwrap();
        assert_eq!(
            velox_frontend::template::py_json(&tools[0]),
            r#"{"type": "function", "function": {"name": "click", "parameters": {"type": "object"}, "strict": false, "description": "Click it."}}"#
        );
        assert_eq!(
            tools[1]["function"],
            json!({"name": "noop", "parameters": {"type": "object", "properties": {}}, "strict": false})
        );
    }

    #[test]
    fn prefix_length() {
        assert_eq!(common_prefix(&[1, 2, 3], &[1, 2, 4, 5]), 2);
        assert_eq!(common_prefix(&[], &[1]), 0);
    }
}
