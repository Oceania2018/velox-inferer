//! Chat engine for Qwen3.5 text + vision models behind the OpenAI-compatible server.
//!
//! Several lanes (`--max-concurrency`), each a sequence slot with its own KV cache and recurrent
//! state, served by one worker thread: queued requests are admitted to free lanes and prefilled,
//! and every decoding request advances by one speculative round per step, all DFlash windows
//! verified in one batched forward pass (weights are read once for every sequence).
//!
//! Agent loops resend almost the same prompt every step, so a lane keeps the keys of the tokens
//! its state holds plus a checkpoint where the next request is likely to diverge (the last history
//! turn before the screenshot). A new request restores the latest checkpoint inside the common
//! prefix, or a context-store entry (system RAM, see `store`) when that covers more, and
//! prefills only the rest; the vision tower runs only for images in that remainder.

mod disk;
mod prompt;
mod sampling;
pub mod spec;
mod store;

use disk::DiskTier;
use store::{ContextStore, Pinner};

use spec::{DFlashDrafter, Drafter, MtpDrafter};

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use serde_json::{json, Value};
use velox_core::{Backend, Error as CoreError, KvFormat};
use velox_frontend::image::{
    decode_data_url, preprocess, smart_resize, ImagePatches, MERGE, PATCH, TEMPORAL,
};
use velox_frontend::{parse_output, Frontend};
use velox_models::qwen3_5::{
    shift_inserts, BatchWorkspace, Checkpoint, DFlashModel, DFlashState, DFlashWorkspace,
    EmbedInsert, PrefillChunk, Qwen35Text, Qwen35Vision, Session, Workspace, DRAFT_PAGE_ROWS,
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
    /// Recurrent-state checkpoints kept per lane for prefix reuse (device checkpoints).
    pub max_checkpoints: usize,
    /// Superseded history points the context store keeps per request chain for retries and
    /// rewinds (ninfer `--ctx-checkpoints`; `usize::MAX`: unbounded).
    pub chain_superseded: usize,
    /// Keep checkpoints in system RAM (pinned, device-mapped) rather than device memory.
    pub host_checkpoints: bool,
    /// Take no checkpoints at or after the prompt's last image: agent screenshots are replaced in the
    /// next request's history, so state past them is never shared (ninfer's media-tail skip).
    pub skip_media_tail: bool,
    /// System RAM for the context store (prefixes any lane can resume after other conversations
    /// overwrote it; see `store`). 0 disables it.
    pub host_context_bytes: usize,
    /// LRFU decay per request of the context store (0: least recently used).
    pub lrfu_lambda: f64,
    /// NVMe tier below the context store (see `disk`); 0 disables it.
    pub disk_context_bytes: u64,
    /// Directory of the NVMe tier's files (a private subdirectory is created and removed).
    pub disk_context_dir: std::path::PathBuf,
    /// Lanes: requests decoding at once (ninfer `--max-concurrency`), each with `capacity`
    /// tokens of KV cache.
    pub lanes: usize,
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
    /// Device checkpoints of the lane's own state (valid while its KV holds `keys`).
    checkpoints: Vec<Checkpoint<B>>,
    /// Storage of dropped checkpoints, reused for new ones (allocating and freeing them would
    /// synchronize the device).
    spare: Vec<Checkpoint<B>>,
    zero: Checkpoint<B>,
    /// The deepest context-store entry on the prefix the current request resumed from.
    chain: Option<u64>,
    /// Draft model for speculative decoding (main lane only).
    drafter: Option<Drafter<B>>,
    /// Speculative rounds of the current request: (rounds, drafted, accepted).
    spec_stats: (usize, usize, usize),
    /// Admission clock value of the lane's latest request (least recently used lanes go first).
    last_used: u64,
}

/// What the worker owns: the lanes and the activation workspaces they share.
struct Lanes<B: Backend> {
    ws: Workspace<B>,
    lanes: Vec<Lane<B>>,
    store: ContextStore<B>,
    /// DFlash drafting activations (shared by every lane's drafter).
    dw: Option<DFlashWorkspace<B>>,
    /// Batched-verify buffers, with more than one lane.
    bw: Option<BatchWorkspace<B>>,
    clock: u64,
    /// Device checkpoint storage not in use, shared by the lanes (lent to the lane being admitted):
    /// prefills snapshot into it instead of allocating, which would synchronize the device.
    spare: Vec<Checkpoint<B>>,
}

/// Device checkpoint storage kept for reuse across lanes.
const DEVICE_SPARE: usize = 2;

/// Lanes whose chain head stays in device memory (the most recently admitted); other heads move
/// to the context store, from where their next request restores them (~5 ms over PCIe).
const DEVICE_HEADS: usize = 2;

/// Checkpoint storage kept for reuse per lane beyond the live checkpoints.
const MAX_SPARE: usize = 4;

/// Prompt tokens per mixed step (see `batch_round`): `VELOX_MIXED_CHUNK`, default a whole
/// workspace chunk (mixed steps run eagerly: short ones pay launch overhead, measured worse at
/// 256 and 512).
fn mixed_chunk() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("VELOX_MIXED_CHUNK")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n: &usize| n > 0)
            .unwrap_or(usize::MAX)
    })
}

/// `VELOX_SOLO_DRAFT=1`: batch rounds draft lane by lane (A/B against batched proposals).
fn solo_draft() -> bool {
    static B: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *B.get_or_init(|| std::env::var("VELOX_SOLO_DRAFT").is_ok_and(|v| v == "1"))
}

/// Below this much shared prefix a request counts as unrelated to a lane.
const MIN_REUSE: usize = 64;

/// A lane's chain head shorter than this is not worth storing when another request takes the
/// lane (prefilling it again costs about as much as the copies).
const MIN_CAPTURE: usize = 1024;

/// Tokens a context-store entry must cover beyond the lane's own best restore point to be used
/// instead (its state and KV come over PCIe).
const STORE_GAIN: usize = 128;

/// Tokens a disk record must cover beyond every resident source to be read back.
const DISK_GAIN: usize = 1024;

pub struct QwenEngine<B: Backend> {
    core: Arc<Core<B>>,
    queue: Arc<JobQueue>,
    worker: Option<JoinHandle<()>>,
}

/// The model and request handling shared by the HTTP threads and the worker.
struct Core<B: Backend> {
    b: B,
    model: Qwen35Text<B>,
    vision: Option<Qwen35Vision<B>>,
    /// DFlash2 block drafter weights, when loaded.
    dflash: Option<DFlashModel<B>>,
    fe: Frontend,
    cfg: EngineConfig,
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

/// A prepared request waiting for (or holding) a lane.
struct Job {
    p: Prompt,
    images: Vec<ImagePatches>,
    sampler: Sampler,
    max_tokens: usize,
    reply: mpsc::Sender<Result<Done, ApiError>>,
}

/// What the worker hands back for a finished request.
struct Done {
    out: Vec<u32>,
    finish: &'static str,
    start: usize,
    prompt_ms: f64,
    predicted_ms: f64,
}

/// A request's prompt still being fed to its lane.
struct Prefill<B: Backend> {
    /// The first prompt token past the restored state.
    start: usize,
    /// Prompt tokens fed since `start`.
    done: usize,
    chunks: usize,
    /// Vision embeddings of the images past `start`, by image run.
    embeds: Vec<(usize, B::Buffer)>,
    /// Checkpoint storage the forward passes fill at their positions.
    snaps: Vec<Checkpoint<B>>,
    anchors: Vec<usize>,
    t0: Instant,
}

/// The image rows of `embeds` (vision outputs by image run) to insert into `p` from `start` on,
/// relative to `start`.
fn prefill_inserts<'a, Buf>(
    embeds: &'a [(usize, Buf)],
    start: usize,
    p: &Prompt,
) -> Vec<EmbedInsert<'a, Buf>> {
    embeds
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
        .collect()
}

/// A request on a lane: prefilling (`pre`) or decoding.
struct Active<B: Backend> {
    job: Job,
    lane: usize,
    out: Vec<u32>,
    /// The sampled token to feed next.
    c: u32,
    finish: Option<&'static str>,
    start: usize,
    prompt_ms: f64,
    t1: Instant,
    /// MTP: cache row of the previous verify and its kept drafts (see `MtpDrafter::draft`).
    mtp_row0: usize,
    mtp_kept: Vec<u32>,
    /// Plain decoding: the candidates after the last fed token.
    cands: Vec<(u32, f32)>,
    /// While the prompt is still being fed.
    pre: Option<Prefill<B>>,
}

/// Requests handed from the HTTP threads to the worker.
struct JobQueue {
    jobs: Mutex<VecDeque<Job>>,
    cv: Condvar,
    stop: AtomicBool,
}

impl<B> QwenEngine<B>
where
    B: Backend + Send + Sync + 'static,
    B::Buffer: Send + Sync + 'static,
{
    pub fn new(
        b: B,
        model: Qwen35Text<B>,
        vision: Option<Qwen35Vision<B>>,
        dflash: Option<DFlashModel<B>>,
        fe: Frontend,
        cfg: EngineConfig,
    ) -> Result<Self, CoreError> {
        let dflash = dflash.filter(|_| cfg.dflash_drafts > 0);
        // Chunks keep their rows only for MTP drafting.
        let mut ws = model.workspace_with(&b, cfg.chunk, dflash.is_none())?;
        if let Some(dm) = &dflash {
            model.enable_capture(&b, &mut ws, &dm.cfg.target_layers)?;
        }
        let n_lanes = cfg.lanes.max(1);
        let mut lanes = Vec::with_capacity(n_lanes);
        for i in 0..n_lanes {
            let mut session = model.session_with(&b, cfg.capacity, cfg.kv_format)?;
            if dflash.is_some() || cfg.mtp_drafts > 0 {
                model.reserve_verify(&b, &mut session, &mut ws)?;
            }
            let zero = session.checkpoint_in(&b, cfg.host_checkpoints)?;
            let drafter = match &dflash {
                Some(dm) => Some(Drafter::DFlash(DFlashDrafter::new(
                    &b,
                    dm,
                    cfg.dflash_drafts,
                )?)),
                // MTP and plain decoding run one request at a time (see `Core::batchable`).
                None if i == 0 && cfg.mtp_drafts > 0 && model.has_mtp() => Some(Drafter::Mtp(
                    MtpDrafter::new(&b, &model, cfg.capacity, cfg.chunk, cfg.mtp_drafts)?,
                )),
                _ => None,
            };
            lanes.push(Lane {
                session,
                keys: Vec::new(),
                checkpoints: Vec::new(),
                spare: Vec::new(),
                chain: None,
                spec_stats: (0, 0, 0),
                zero,
                drafter,
                last_used: 0,
            });
        }
        let dw = match &dflash {
            Some(dm) => Some(DFlashDrafter::workspace(
                &b,
                dm,
                &model,
                cfg.chunk,
                cfg.dflash_drafts,
                n_lanes,
            )?),
            None => None,
        };
        let bw = match &dflash {
            Some(_) if n_lanes > 1 => Some(model.batch_workspace(&b, n_lanes)?),
            _ => None,
        };
        let special = SpecialTokens {
            im_start: single_token(&fe, "<|im_start|>"),
            vision_start: single_token(&fe, "<|vision_start|>"),
            image_pad: single_token(&fe, "<|image_pad|>"),
        };
        let disk = if cfg.disk_context_bytes > 0 && cfg.host_context_bytes > 0 {
            Some(
                DiskTier::new(cfg.disk_context_dir.clone(), cfg.disk_context_bytes)
                    .map_err(|e| CoreError::Config(format!("disk context tier: {e}")))?,
            )
        } else {
            None
        };
        let mut store = ContextStore::new(cfg.host_context_bytes, cfg.lrfu_lambda, disk);
        if store.enabled() && std::env::var_os("VELOX_NO_PINNER").is_none() {
            if let Some(alloc) = b.host_alloc() {
                store.set_pinner(
                    Pinner::spawn(alloc)
                        .map_err(|e| CoreError::Backend(format!("pinning thread: {e}")))?,
                );
            }
        }
        let state = Lanes {
            ws,
            lanes,
            store,
            dw,
            bw,
            clock: 0,
            spare: Vec::new(),
        };
        let core = Arc::new(Core {
            b,
            model,
            vision,
            dflash,
            fe,
            cfg,
            special,
        });
        let mut state = state;
        core.warm_up(&mut state)?;
        let queue = Arc::new(JobQueue {
            jobs: Mutex::new(VecDeque::new()),
            cv: Condvar::new(),
            stop: AtomicBool::new(false),
        });
        let worker = {
            let (core, queue) = (Arc::clone(&core), Arc::clone(&queue));
            std::thread::Builder::new()
                .name("velox-engine".into())
                .spawn(move || core.run(state, &queue))
                .map_err(|e| CoreError::Backend(format!("engine worker: {e}")))?
        };
        Ok(QwenEngine {
            core,
            queue,
            worker: Some(worker),
        })
    }
}

impl<B: Backend> Drop for QwenEngine<B> {
    fn drop(&mut self) {
        self.queue.stop.store(true, Ordering::SeqCst);
        self.queue.cv.notify_all();
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

impl<B: Backend> Core<B> {
    /// Runs the largest shapes once (a prefill chunk at the end of the window, the longest verify
    /// window and draft block, a verify batch over every lane, the largest image) so the kernels'
    /// grow-only device workspaces reach their final size before serving. Memory then stays flat
    /// under load: growing scratch keeps every smaller buffer (recorded graphs may hold them), and
    /// past the card's memory WDDM spills to system RAM and everything crawls. The lanes are reset
    /// afterwards.
    fn warm_up(&self, st: &mut Lanes<B>) -> Result<(), CoreError> {
        use velox_models::qwen3_5::VERIFY_MAX;
        let (b, model) = (&self.b, &self.model);
        let limit = self.fe.public_vocab;
        let chunk = self.cfg.chunk;
        let Lanes {
            ws, lanes, dw, bw, ..
        } = st;
        {
            let lane = &mut lanes[0];
            let pos = lane.session.capacity.saturating_sub(chunk + VERIFY_MAX + 1);
            lane.session.pos = pos;
            let ids = vec![0u32; chunk.min(lane.session.capacity - pos)];
            match (lane.drafter.as_mut(), self.dflash.as_ref(), dw.as_mut()) {
                (Some(Drafter::DFlash(d)), Some(dm), Some(dw)) => {
                    model.advance_rows(
                        b,
                        &mut lane.session,
                        ws,
                        &ids,
                        None,
                        &[],
                        &mut [],
                        &mut |ws, base, t| d.prefill_chunk(b, dm, dw, ws, pos + base, t),
                    )?;
                }
                _ => model.advance(b, &mut lane.session, ws, &ids, None, &[])?,
            }
            model.candidates(b, ws, CANDIDATES, limit)?;
            if lane.drafter.is_some() {
                let window = vec![0u32; VERIFY_MAX.min(lane.session.capacity - lane.session.pos)];
                model.verify(b, &mut lane.session, ws, &window, CANDIDATES, limit)?;
                model.commit(b, &mut lane.session, ws, 1)?;
            }
            if let (Some(Drafter::DFlash(d)), Some(dm), Some(dw)) =
                (lane.drafter.as_mut(), self.dflash.as_ref(), dw.as_mut())
            {
                d.drafts = d.long_block();
                d.draft(b, dm, dw, model, ws, 0, 0, lane.session.pos, 0, limit, 0.0)?;
            }
        }
        if let Some(bw) = bw.as_mut() {
            let m = 8usize;
            let mut sessions: Vec<&mut Session<B>> =
                lanes.iter_mut().map(|l| &mut l.session).collect();
            for s in sessions.iter_mut() {
                s.pos = s.capacity.saturating_sub(m + 1);
            }
            let window = vec![0u32; m];
            let windows: Vec<&[u32]> = vec![&window[..]; sessions.len().min(bw.max_seqs())];
            sessions.truncate(windows.len());
            model.verify_batch(b, &mut sessions, ws, bw, &windows, CANDIDATES, limit)?;
            for s in sessions {
                model.commit_batched(b, s, bw, 1)?;
            }
        }
        if let (Some(v), Some((min_px, max_px))) = (&self.vision, self.fe.vision_pixels) {
            // A square image at the pixel budget: the most patches the tower sees.
            let side = ((max_px.min(1 << 26)) as f64).sqrt() as usize;
            if let Ok((rh, rw)) = smart_resize(side, side, min_px, max_px) {
                let (gh, gw) = (rh / PATCH, rw / PATCH);
                if gh % MERGE == 0 && gw % MERGE == 0 {
                    let patches = vec![0.0f32; gh * gw * 3 * TEMPORAL * PATCH * PATCH];
                    v.encode(b, &patches, gh, gw)?;
                }
            }
        }
        b.synchronize()?;
        for lane in lanes.iter_mut() {
            lane.session.restore(b, &lane.zero)?;
            lane.keys.clear();
            lane.checkpoints.clear();
            if let (Some(Drafter::DFlash(d)), Some(dm)) =
                (lane.drafter.as_mut(), self.dflash.as_ref())
            {
                d.forget(dm, 0);
                d.reset();
            }
        }
        b.synchronize()
    }

    /// MTP and plain decoding keep per-step rows in the shared workspace between rounds, so only
    /// DFlash requests decode side by side.
    fn batchable(&self) -> bool {
        self.dflash.is_some()
    }

    /// The worker: admits queued requests to free lanes (prefilling them) and runs speculative
    /// rounds over every decoding request, batched into one verify pass.
    fn run(&self, mut st: Lanes<B>, queue: &JobQueue) {
        if let Err(e) = self.b.bind_thread() {
            eprintln!("engine worker: {e}");
            return;
        }
        let mut waiting: VecDeque<Job> = VecDeque::new();
        let mut active: Vec<Active<B>> = Vec::new();
        let draft = self.dflash.as_ref().map(|dm| {
            (
                dm.context_window().div_ceil(DRAFT_PAGE_ROWS),
                dm.context_page_len(),
            )
        });
        loop {
            st.store.feed(&self.b, &st.lanes[0].session, draft);
            // New requests; while idle, pin context-store storage between them.
            loop {
                let mut q = queue.jobs.lock().expect("queue lock");
                let stop = queue.stop.load(Ordering::SeqCst);
                if !q.is_empty() || !active.is_empty() || !waiting.is_empty() || stop {
                    waiting.extend(q.drain(..));
                    break;
                }
                drop(q);
                // One drafter window of context pages: a capture that saves the drafter context
                // does not allocate pinned memory on the request path.
                let more = match st.store.top_up(&self.b, &st.lanes[0].session, draft) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("context store: reserve allocation failed: {e}");
                        false
                    }
                };
                if !more {
                    let q = queue.jobs.lock().expect("queue lock");
                    let _q = queue
                        .cv
                        .wait_while(q, |q| q.is_empty() && !queue.stop.load(Ordering::SeqCst))
                        .expect("queue lock");
                }
            }
            if queue.stop.load(Ordering::SeqCst) {
                let gone = || ApiError {
                    status: 503,
                    kind: "server_error",
                    code: "shutting_down",
                    message: "engine stopped".into(),
                };
                for j in waiting.drain(..).chain(active.drain(..).map(|a| a.job)) {
                    let _ = j.reply.send(Err(gone()));
                }
                return;
            }
            // Admission: a free lane for each waiting request, in arrival order.
            while !waiting.is_empty() {
                if !self.batchable() && !active.is_empty() {
                    break;
                }
                let busy: Vec<usize> = active.iter().map(|a| a.lane).collect();
                let Some(li) = Self::pick_lane(&st.lanes, &busy, &waiting[0].p.keys) else {
                    break;
                };
                let job = waiting.pop_front().expect("checked");
                match self.begin(&mut st, li, job) {
                    Ok(a) => {
                        active.push(a);
                        if let Err(e) = self.evict_heads(&mut st, li) {
                            eprintln!("context store: moving a chain head failed: {e}");
                        }
                    }
                    Err((job, e)) => {
                        let _ = job.reply.send(Err(e));
                    }
                }
            }
            if active.is_empty() {
                continue;
            }
            if let Err(e) = self.round(&mut st, &mut active) {
                // A device error leaves every decoding lane in an unknown state.
                for a in active.drain(..) {
                    let lane = &mut st.lanes[a.lane];
                    lane.keys.clear();
                    lane.checkpoints.clear();
                    let _ = a.job.reply.send(Err(e.clone()));
                }
                continue;
            }
            let (done, rest): (Vec<Active<B>>, Vec<Active<B>>) =
                active.drain(..).partition(|a| a.finish.is_some());
            active = rest;
            for a in done {
                self.complete(&st, a);
            }
        }
    }

    /// Keeps device checkpoints on the [`DEVICE_HEADS`] most recently admitted lanes: older lanes'
    /// chain heads go to the context store (their KV rows are still in the lane) and their storage
    /// to the shared spares.
    fn evict_heads(&self, st: &mut Lanes<B>, current: usize) -> Result<(), CoreError> {
        let Lanes {
            lanes,
            store,
            spare,
            ..
        } = st;
        loop {
            let holders: Vec<usize> = (0..lanes.len())
                .filter(|&i| !lanes[i].checkpoints.is_empty())
                .collect();
            if holders.len() <= DEVICE_HEADS {
                break;
            }
            let Some(j) = holders
                .into_iter()
                .filter(|&i| i != current)
                .min_by_key(|&i| lanes[i].last_used)
            else {
                break;
            };
            let lane = &mut lanes[j];
            for cp in std::mem::take(&mut lane.checkpoints) {
                if store.enabled()
                    && cp.pos >= MIN_CAPTURE
                    && cp.pos <= lane.keys.len()
                    && !store.holds(&lane.keys[..cp.pos])
                {
                    let drafter = self.drafter_context(lane);
                    store.insert(
                        &self.b,
                        &lane.session,
                        &lane.keys,
                        &cp,
                        false,
                        lane.chain,
                        drafter,
                    )?;
                }
                spare.push(cp);
            }
        }
        spare.truncate(DEVICE_SPARE);
        Ok(())
    }

    /// A free lane for a prompt of `keys`: the one that can reuse the most, else the least
    /// recently used (its conversation is the least likely to come back soon; its chain head goes
    /// to the context store when the new request takes it).
    fn pick_lane(lanes: &[Lane<B>], busy: &[usize], keys: &[u32]) -> Option<usize> {
        let free: Vec<usize> = (0..lanes.len()).filter(|i| !busy.contains(i)).collect();
        let (best, shared) = free
            .iter()
            .map(|&i| (i, common_prefix(&lanes[i].keys, keys)))
            .max_by_key(|&(i, s)| (s, std::cmp::Reverse(i)))?;
        if shared >= MIN_REUSE {
            return Some(best);
        }
        free.into_iter().min_by_key(|&i| lanes[i].last_used)
    }

    /// Takes `job` onto lane `li`: restores the best reusable state and prepares the prefill of the
    /// rest (vision embeddings, checkpoint storage). The prompt is then fed by [`Self::prefill_rest`]
    /// or chunk by chunk alongside other requests' decoding (see `batch_round`).
    fn begin(&self, st: &mut Lanes<B>, li: usize, job: Job) -> Result<Active<B>, (Job, ApiError)> {
        let debug = std::env::var_os("VELOX_DEBUG_PREFIX").is_some();
        let Lanes {
            lanes,
            store,
            clock,
            spare: pool,
            ..
        } = st;
        *clock += 1;
        let lane = &mut lanes[li];
        lane.last_used = *clock;
        // The lane works with the shared spare storage while it is set up.
        lane.spare.append(pool);
        let p = &job.p;
        let t0 = Instant::now();
        if debug {
            let shared = common_prefix(&lane.keys, &p.keys);
            let cps: Vec<usize> = lane.checkpoints.iter().map(|c| c.pos).collect();
            eprintln!(
                "[prefix] lane {li} prompt {} held {} pos {} shared {} checkpoints {:?} next {:?} store [{}]",
                p.keys.len(),
                lane.keys.len(),
                lane.session.pos,
                shared,
                cps,
                self.checkpoint_positions(&p.ids, 0),
                store.describe()
            );
        }
        match lane.drafter.as_mut() {
            Some(Drafter::Mtp(m)) => m.reset(),
            Some(Drafter::DFlash(d)) => d.reset(),
            None => {}
        }
        store.tick();
        let shared = common_prefix(&lane.keys, &p.keys);
        let r = (|| -> Result<Prefill<B>, CoreError> {
            if store.enabled() {
                self.capture_head(lane, store, shared.min(p.keys.len() - 1))?;
            }
            let (start, adopted) = self.reuse(lane, store, &p.keys)?;
            lane.chain = store.resumed(&p.keys, start, adopted);
            if adopted.is_some() && debug {
                eprintln!(
                    "[store] lane {li} resumed at {start} ({} entries, {:.2} GB; {} on disk)",
                    store.len(),
                    store.bytes(&self.b) as f64 / 1e9,
                    store.disk_len()
                );
            }
            // Checkpoints still valid for this prompt are the lane's own inside the shared prefix.
            let (keep, stale): (Vec<_>, Vec<_>) = lane
                .checkpoints
                .drain(..)
                .partition(|c| c.pos <= start && c.pos <= shared);
            lane.checkpoints = keep;
            lane.spare.extend(stale);
            lane.keys = p.keys[..start].to_vec();
            let anchors: Vec<usize> = self
                .checkpoint_positions(&p.ids, 0)
                .into_iter()
                .take(2)
                .collect();
            self.begin_prefill(lane, store, p, &job.images, start, anchors, t0)
        })();
        lane.spare.truncate(DEVICE_SPARE);
        pool.append(&mut lane.spare);
        match r {
            Ok(pre) => Ok(Active {
                job,
                lane: li,
                out: Vec::new(),
                c: 0,
                finish: None,
                start: pre.start,
                prompt_ms: 0.0,
                t1: Instant::now(),
                mtp_row0: 0,
                mtp_kept: Vec::new(),
                cands: Vec::new(),
                pre: Some(pre),
            }),
            Err(e) => {
                lane.keys.clear();
                lane.checkpoints.clear();
                Err((job, core(e)))
            }
        }
    }

    /// The prompt is in the lane's state: checkpoints placed, the first token sampled from
    /// `cands`; the request decodes from the next round on.
    fn finish_prefill(
        &self,
        st: &mut Lanes<B>,
        a: &mut Active<B>,
        cands: Vec<(u32, f32)>,
    ) -> Result<(), CoreError> {
        let debug = std::env::var_os("VELOX_DEBUG_PREFIX").is_some();
        let pre = a.pre.take().expect("a prefilling request");
        let Lanes {
            lanes,
            store,
            spare: pool,
            ..
        } = st;
        let lane = &mut lanes[a.lane];
        let p = &a.job.p;
        lane.spare.append(pool);
        lane.keys.extend_from_slice(&p.keys[pre.start..]);
        lane.checkpoints.extend(pre.snaps);
        let placed = if self.cfg.host_checkpoints {
            self.place_checkpoints(lane, store, &pre.anchors, p.keys.len() - 1, pre.start)
        } else {
            Self::prune_checkpoints(
                &mut lane.checkpoints,
                &mut lane.spare,
                self.cfg.max_checkpoints,
                pre.start + 1,
            );
            Ok(())
        };
        lane.spare.truncate(DEVICE_SPARE);
        pool.append(&mut lane.spare);
        placed?;
        a.prompt_ms = pre.t0.elapsed().as_secs_f64() * 1e3;
        if debug {
            eprintln!(
                "[prompt] lane {} {} tokens from {} in {:.1} ms ({} chunks), top {:?}",
                a.lane,
                p.ids.len() - pre.start,
                pre.start,
                a.prompt_ms,
                pre.chunks,
                &cands[..cands.len().min(3)]
            );
        }
        lane.spec_stats = (0, 0, 0);
        a.c = a.job.sampler.sample(&cands);
        a.cands = cands;
        a.t1 = Instant::now();
        Ok(())
    }

    /// Feeds what is left of a prefilling request's prompt in one go (no other request decodes).
    fn prefill_rest(&self, st: &mut Lanes<B>, a: &mut Active<B>) -> Result<(), CoreError> {
        let cands = {
            let Lanes { ws, lanes, dw, .. } = &mut *st;
            let lane = &mut lanes[a.lane];
            let pre = a.pre.as_mut().expect("a prefilling request");
            let p = &a.job.p;
            let from = pre.start + pre.done;
            let inserts = shift_inserts(&prefill_inserts(&pre.embeds, pre.start, p), pre.done);
            let ids = &p.ids[from..];
            let positions = p.positions.as_ref().map(|v| &v[from..]);
            let Lane {
                session, drafter, ..
            } = &mut *lane;
            match drafter.as_mut() {
                Some(dr) => {
                    let (row0, delta) = (session.pos, session.rope_delta);
                    let model = &self.model;
                    let dflash = self.dflash.as_ref();
                    let b = &self.b;
                    let mut dw = dw.as_mut();
                    model.advance_rows(
                        b,
                        session,
                        ws,
                        ids,
                        positions,
                        &inserts,
                        &mut pre.snaps,
                        &mut |ws, base, t| {
                            let rows = positions.map(|v| &v[base..base + t]);
                            match &mut *dr {
                                Drafter::Mtp(m) => {
                                    m.prefill_chunk(b, model, ws, row0 + base, t, delta, rows)
                                }
                                Drafter::DFlash(d) => d.prefill_chunk(
                                    b,
                                    dflash.expect("a DFlash drafter has its model"),
                                    dw.as_deref_mut().expect("DFlash lanes have a workspace"),
                                    ws,
                                    row0 + base,
                                    t,
                                ),
                            }
                        },
                    )?;
                }
                None => self.model.advance_snap(
                    &self.b,
                    session,
                    ws,
                    ids,
                    positions,
                    &inserts,
                    &mut pre.snaps,
                )?,
            }
            pre.done = p.ids.len() - pre.start;
            pre.chunks += 1;
            self.model
                .candidates(&self.b, ws, CANDIDATES, self.fe.public_vocab)?
        };
        self.finish_prefill(st, a, cands)
    }

    /// Hands a finished request back to its HTTP thread.
    fn complete(&self, st: &Lanes<B>, a: Active<B>) {
        let predicted_ms = a.t1.elapsed().as_secs_f64() * 1e3;
        let lane = &st.lanes[a.lane];
        if std::env::var_os("VELOX_DEBUG_PREFIX").is_some() && lane.drafter.is_some() {
            let (rounds, drafted, accepted) = lane.spec_stats;
            eprintln!(
                "[decode] lane {} {} tokens in {predicted_ms:.0} ms, {rounds} rounds, accepted {accepted}/{drafted} drafts, {:.2} tokens/round",
                a.lane,
                a.out.len(),
                a.out.len() as f64 / rounds.max(1) as f64
            );
        }
        let _ = a.job.reply.send(Ok(Done {
            out: a.out,
            finish: a.finish.unwrap_or("length"),
            start: a.start,
            prompt_ms: a.prompt_ms,
            predicted_ms,
        }));
    }

    /// One step for every active request: decoding requests take their pending token, then draft,
    /// verify (batched when several DFlash requests decode) and commit; a prefilling request feeds
    /// its next prompt chunk in the same forward pass, or its whole prompt when nothing decodes.
    fn round(&self, st: &mut Lanes<B>, active: &mut [Active<B>]) -> Result<(), ApiError> {
        for a in active.iter_mut().filter(|a| a.pre.is_none()) {
            let c = a.c;
            if self.fe.is_eos(c) {
                a.finish = Some("stop");
                continue;
            }
            a.out.push(c);
            a.job.sampler.observe(c);
            if a.out.len() >= a.job.max_tokens {
                a.finish = Some("length");
            }
        }
        let live: Vec<usize> = (0..active.len())
            .filter(|&i| active[i].pre.is_none() && active[i].finish.is_none())
            .collect();
        let dflash: Vec<usize> = live
            .iter()
            .copied()
            .filter(|&i| matches!(st.lanes[active[i].lane].drafter, Some(Drafter::DFlash(_))))
            .collect();
        if let Some(pi) = (0..active.len()).find(|&i| active[i].pre.is_some()) {
            if !dflash.is_empty() && st.bw.is_some() {
                // The next prompt chunk rides with every decoding request's window.
                return self.batch_round(st, active, &dflash, Some(pi));
            }
            self.prefill_rest(st, &mut active[pi]).map_err(core)?;
        }
        if dflash.len() > 1 {
            self.batch_round(st, active, &dflash, None)?;
        } else if let Some(&i) = dflash.first() {
            self.dflash_round(st, &mut active[i])?;
        }
        for &i in live.iter().filter(|i| !dflash.contains(i)) {
            match st.lanes[active[i].lane].drafter {
                Some(Drafter::Mtp(_)) => self.mtp_round(st, &mut active[i])?,
                _ => self.plain_step(st, &mut active[i])?,
            }
        }
        Ok(())
    }

    /// Accepts the drafts of a verified `window` against the target's candidate `rows` (exact
    /// match when greedy, speculative sampling against the proposal `qs` otherwise), appending
    /// them to the output. Returns the kept prefix length and the next token.
    fn accept(
        &self,
        a: &mut Active<B>,
        window: &[u32],
        rows: &[Vec<(u32, f32)>],
        qs: Option<&Vec<Vec<(u32, f32)>>>,
    ) -> (usize, Option<u32>) {
        let mut keep = 1;
        let mut next = None;
        for j in 0..window.len() {
            if j + 1 == window.len() {
                next = Some(a.job.sampler.sample(&rows[j]));
                break;
            }
            let verdict = match qs {
                Some(q) => a.job.sampler.verify_draft_q(&rows[j], window[j + 1], &q[j]),
                None => a.job.sampler.verify_draft(&rows[j], window[j + 1]),
            };
            match verdict {
                Ok(()) => {
                    keep += 1;
                    let d = window[j + 1];
                    if self.fe.is_eos(d) {
                        a.finish = Some("stop");
                        break;
                    }
                    a.out.push(d);
                    a.job.sampler.observe(d);
                    if a.out.len() >= a.job.max_tokens {
                        a.finish = Some("length");
                        break;
                    }
                }
                Err(t) => {
                    next = Some(t);
                    break;
                }
            }
        }
        (keep, next)
    }

    /// The draft window of a DFlash request: the pending token and up to the remaining budget of
    /// drafts, with the proposal distribution of each draft.
    #[allow(clippy::type_complexity)]
    fn dflash_window(
        &self,
        lane: &mut Lane<B>,
        dw: &mut DFlashWorkspace<B>,
        ws: &Workspace<B>,
        a: &Active<B>,
        whole: bool,
    ) -> Result<(Vec<u32>, Vec<Vec<(u32, f32)>>), ApiError> {
        let dm = self
            .dflash
            .as_ref()
            .expect("a DFlash drafter has its model");
        let Some(Drafter::DFlash(d)) = lane.drafter.as_mut() else {
            unreachable!("caller checked");
        };
        let (drafts, qs) = d
            .draft(
                &self.b,
                dm,
                dw,
                &self.model,
                ws,
                0,
                0,
                lane.session.pos,
                a.c,
                self.fe.public_vocab,
                a.job.sampler.temperature(),
            )
            .map_err(core)?;
        // `whole`: keep drafts past the request's budget (acceptance stops at the budget anyway),
        // so every window of a batch has the same length.
        let budget = if whole {
            drafts.len()
        } else {
            a.job.max_tokens - a.out.len()
        };
        let mut window = vec![a.c];
        window.extend(drafts.into_iter().take(budget));
        Ok((window, qs))
    }

    /// Bookkeeping after a request's window was verified and `keep` tokens committed: drafter
    /// context for the kept rows (features at `feat_row` of the verify), its adaptation, the
    /// lane's keys and the next pending token.
    #[allow(clippy::too_many_arguments)]
    fn after_commit(
        &self,
        lane: &mut Lane<B>,
        dw: &mut DFlashWorkspace<B>,
        ws: &Workspace<B>,
        a: &mut Active<B>,
        window: &[u32],
        p: usize,
        keep: usize,
        feat_row: usize,
        next: Option<u32>,
    ) -> Result<(), ApiError> {
        let dm = self
            .dflash
            .as_ref()
            .expect("a DFlash drafter has its model");
        let Some(Drafter::DFlash(d)) = lane.drafter.as_mut() else {
            unreachable!("caller checked");
        };
        lane.spec_stats.0 += 1;
        lane.spec_stats.1 += window.len() - 1;
        lane.spec_stats.2 += keep - 1;
        d.commit_rows(&self.b, dm, dw, ws, feat_row, p, keep)
            .map_err(core)?;
        d.observe(keep - 1);
        lane.keys.extend_from_slice(&window[..keep]);
        if a.finish.is_none() {
            a.c = next.expect("a window ends with a sampled token");
        }
        Ok(())
    }

    /// A single DFlash request's round: the recorded-graph verify path.
    fn dflash_round(&self, st: &mut Lanes<B>, a: &mut Active<B>) -> Result<(), ApiError> {
        let Lanes { ws, lanes, dw, .. } = st;
        let dw = dw.as_mut().expect("DFlash lanes have a workspace");
        let lane = &mut lanes[a.lane];
        let (window, qs) = self.dflash_window(lane, dw, ws, a, false)?;
        let p = lane.session.pos;
        let rows = self
            .model
            .verify(
                &self.b,
                &mut lane.session,
                ws,
                &window,
                CANDIDATES,
                self.fe.public_vocab,
            )
            .map_err(core)?;
        let (keep, next) = self.accept(a, &window, &rows, Some(&qs));
        self.model
            .commit(&self.b, &mut lane.session, ws, keep)
            .map_err(core)?;
        self.after_commit(lane, dw, ws, a, &window, p, keep, 0, next)
    }

    /// Several DFlash requests' round: every window verified in one forward pass, with the next
    /// chunk of the prefilling request `pre` (if any) in the same pass.
    fn batch_round(
        &self,
        st: &mut Lanes<B>,
        active: &mut [Active<B>],
        idx: &[usize],
        pre: Option<usize>,
    ) -> Result<(), ApiError> {
        let timing = std::env::var_os("VELOX_DEBUG_BATCH").is_some();
        let t0 = Instant::now();
        let mut done_prefill = None;
        let (n_rows, tp) = {
            let Lanes {
                ws, lanes, dw, bw, ..
            } = &mut *st;
            let dw = dw.as_mut().expect("DFlash lanes have a workspace");
            let bw = bw.as_mut().expect("several lanes have a batch workspace");
            // Every sequence drafts the short block (ninfer's fixed 7 drafts): one shape, so one
            // recorded step, and two sequences fit the 16-row tiles a single long window would
            // fill. Long blocks pay off only for a request decoding alone.
            let drafts = idx
                .iter()
                .filter_map(|&i| match &lanes[active[i].lane].drafter {
                    Some(Drafter::DFlash(d)) => Some(d.short_block()),
                    _ => None,
                })
                .min()
                .unwrap_or(1);
            let room = idx.iter().all(|&i| {
                lanes[active[i].lane].session.pos + drafts < lanes[active[i].lane].session.capacity
            });
            let mut windows = Vec::with_capacity(idx.len());
            if idx.len() == 1 || solo_draft() {
                // One drafter at a time: the recorded proposal step.
                for &i in idx {
                    let a = &active[i];
                    let lane = &mut lanes[a.lane];
                    if let Some(Drafter::DFlash(d)) = lane.drafter.as_mut() {
                        d.drafts = drafts;
                    }
                    windows.push(self.dflash_window(lane, dw, ws, a, room)?);
                }
            } else {
                // Every drafter in one block forward (one host sync), then each picks its drafts.
                let dm = self
                    .dflash
                    .as_ref()
                    .expect("a DFlash drafter has its model");
                for &i in idx {
                    let lane = &mut lanes[active[i].lane];
                    if let Some(Drafter::DFlash(d)) = lane.drafter.as_mut() {
                        d.drafts = drafts;
                        d.align(dm, lane.session.pos);
                    }
                }
                let proposals = {
                    let states: Vec<&DFlashState<B>> = idx
                        .iter()
                        .map(|&i| match &lanes[active[i].lane].drafter {
                            Some(Drafter::DFlash(d)) => d.context(),
                            _ => unreachable!("caller checked"),
                        })
                        .collect();
                    let anchors: Vec<u32> = idx.iter().map(|&i| active[i].c).collect();
                    dm.propose_batch(
                        &self.b,
                        &self.model,
                        &states,
                        dw,
                        &anchors,
                        drafts,
                        self.fe.public_vocab,
                    )
                    .map_err(core)?
                };
                for (j, &i) in idx.iter().enumerate() {
                    let a = &active[i];
                    let Some(Drafter::DFlash(d)) = lanes[a.lane].drafter.as_mut() else {
                        unreachable!("caller checked");
                    };
                    let (picked, qs) = d.pick(dm, &proposals[j], a.c, a.job.sampler.temperature());
                    let budget = if room {
                        picked.len()
                    } else {
                        a.job.max_tokens - a.out.len()
                    };
                    let mut window = vec![a.c];
                    window.extend(picked.into_iter().take(budget));
                    windows.push((window, qs));
                }
            }
            let v: usize = windows.iter().map(|(w, _)| w.len()).sum();
            // Disjoint lanes of this batch, in `idx` order, and the prefilling lane.
            let mut slots: Vec<Option<&mut Lane<B>>> = lanes.iter_mut().map(Some).collect();
            let mut batch: Vec<&mut Lane<B>> = idx
                .iter()
                .map(|&i| slots[active[i].lane].take().expect("one request per lane"))
                .collect();
            let mut plane =
                pre.map(|pi| slots[active[pi].lane].take().expect("one request per lane"));
            let starts: Vec<usize> = batch.iter().map(|l| l.session.pos).collect();
            let t_draft = t0.elapsed();
            let ids: Vec<&[u32]> = windows.iter().map(|(w, _)| &w[..]).collect();
            let mut tp = 0;
            let rows = match (pre, plane.as_deref_mut()) {
                (Some(pi), Some(pl)) => {
                    let mut pstate = active[pi].pre.take().expect("a prefilling request");
                    let r = (|| -> Result<_, CoreError> {
                        let p = &active[pi].job.p;
                        let from = pstate.start + pstate.done;
                        // Chunks short enough that the decoding requests advance several rounds
                        // during one prompt, long enough to stay compute-bound.
                        tp = (p.ids.len() - from)
                            .min(ws.max_tokens() - v)
                            .min(mixed_chunk());
                        let last = from + tp == p.ids.len();
                        let all = prefill_inserts(&pstate.embeds, pstate.start, p);
                        let inserts = shift_inserts(&all, pstate.done);
                        let row0 = pl.session.pos;
                        let mut sessions: Vec<&mut Session<B>> =
                            batch.iter_mut().map(|l| &mut l.session).collect();
                        let (cands, rows) = self.model.mixed_step(
                            &self.b,
                            PrefillChunk {
                                session: &mut pl.session,
                                ids: &p.ids[from..from + tp],
                                positions: p.positions.as_ref().map(|q| &q[from..from + tp]),
                                inserts: &inserts,
                                snaps: &mut pstate.snaps,
                                candidates: last,
                            },
                            &mut sessions,
                            ws,
                            bw,
                            &ids,
                            CANDIDATES,
                            self.fe.public_vocab,
                        )?;
                        // The prefilling drafter's context for the chunk (features rows 0 .. tp).
                        if let (Some(Drafter::DFlash(d)), Some(dm)) =
                            (pl.drafter.as_mut(), self.dflash.as_ref())
                        {
                            d.prefill_chunk(&self.b, dm, dw, ws, row0, tp)?;
                        }
                        Ok((cands, rows))
                    })();
                    pstate.done += tp;
                    pstate.chunks += 1;
                    active[pi].pre = Some(pstate);
                    let (cands, rows) = r.map_err(core)?;
                    done_prefill = cands.map(|c| (pi, c));
                    rows
                }
                _ => {
                    let mut sessions: Vec<&mut Session<B>> =
                        batch.iter_mut().map(|l| &mut l.session).collect();
                    self.model
                        .verify_batch(
                            &self.b,
                            &mut sessions,
                            ws,
                            bw,
                            &ids,
                            CANDIDATES,
                            self.fe.public_vocab,
                        )
                        .map_err(core)?
                }
            };
            let t_verify = t0.elapsed();
            // Window features follow the prefill rows.
            let mut feat_row = tp;
            for (j, &i) in idx.iter().enumerate() {
                let a = &mut active[i];
                let (window, qs) = &windows[j];
                let (keep, next) = self.accept(a, window, &rows[j], Some(qs));
                let lane = &mut *batch[j];
                self.model
                    .commit_batched(&self.b, &mut lane.session, bw, keep)
                    .map_err(core)?;
                self.after_commit(lane, dw, ws, a, window, starts[j], keep, feat_row, next)?;
                feat_row += window.len();
            }
            if timing {
                self.b.synchronize().map_err(core)?;
                let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
                eprintln!(
                    "[batch] {} seqs, {} rows + {tp} prefill: draft {:.1} ms, verify {:.1} ms, commit {:.1} ms",
                    idx.len(),
                    feat_row - tp,
                    ms(t_draft),
                    ms(t_verify - t_draft),
                    ms(t0.elapsed() - t_verify)
                );
            }
            (feat_row - tp, tp)
        };
        let _ = (n_rows, tp);
        if let Some((pi, cands)) = done_prefill {
            self.finish_prefill(st, &mut active[pi], cands)
                .map_err(core)?;
        }
        Ok(())
    }

    /// An MTP request's round (one request at a time; see `batchable`).
    fn mtp_round(&self, st: &mut Lanes<B>, a: &mut Active<B>) -> Result<(), ApiError> {
        let Lanes { ws, lanes, .. } = st;
        let lane = &mut lanes[a.lane];
        let limit = self.fe.public_vocab;
        let (b, model) = (&self.b, &self.model);
        let Lane {
            session,
            keys,
            drafter,
            spec_stats,
            ..
        } = lane;
        let Some(Drafter::Mtp(m)) = drafter.as_mut() else {
            unreachable!("caller checked");
        };
        let drafts = m
            .draft(b, model, session, ws, a.mtp_row0, &a.mtp_kept, a.c, limit)
            .map_err(core)?;
        let mut window = vec![a.c];
        window.extend(drafts.into_iter().take(a.job.max_tokens - a.out.len()));
        let p = session.pos;
        let rows = model
            .verify(b, session, ws, &window, CANDIDATES, limit)
            .map_err(core)?;
        let (keep, next) = self.accept(a, &window, &rows, None);
        spec_stats.0 += 1;
        spec_stats.1 += window.len() - 1;
        spec_stats.2 += keep - 1;
        model.commit(b, session, ws, keep).map_err(core)?;
        m.record(keep - 1, window.len() - 1);
        keys.extend_from_slice(&window[..keep]);
        if a.finish.is_none() {
            a.c = next.expect("a window ends with a sampled token");
            a.mtp_row0 = p;
            a.mtp_kept = window[1..keep].to_vec();
        }
        Ok(())
    }

    /// Plain decoding: feed the pending token, sample the next.
    fn plain_step(&self, st: &mut Lanes<B>, a: &mut Active<B>) -> Result<(), ApiError> {
        let Lanes { ws, lanes, .. } = st;
        let lane = &mut lanes[a.lane];
        self.model
            .advance(&self.b, &mut lane.session, ws, &[a.c], None, &[])
            .map_err(core)?;
        lane.keys.push(a.c);
        a.cands = self
            .model
            .candidates(&self.b, ws, CANDIDATES, self.fe.public_vocab)
            .map_err(core)?;
        a.c = a.job.sampler.sample(&a.cands);
        Ok(())
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
            .filter(|&p| p > after && p < ids.len() && p < tail)
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

    /// Host checkpoint mode, after a prefill: checkpoints at `anchors` (system prompt and task:
    /// what other conversations share) go to the context store as roots, the one at `tail` (the
    /// prompt's end, for exact retries) goes there as an ordinary entry, and the lane keeps only
    /// its deepest other checkpoint (its chain head, the next request's likeliest restore point;
    /// with several lanes, device memory holds about one checkpoint per lane).
    fn place_checkpoints(
        &self,
        lane: &mut Lane<B>,
        store: &mut ContextStore<B>,
        anchors: &[usize],
        tail: usize,
        resumed: usize,
    ) -> Result<(), CoreError> {
        lane.checkpoints.sort_by_key(|c| c.pos);
        let head = lane
            .checkpoints
            .iter()
            .map(|c| c.pos)
            .filter(|&p| p != tail)
            .max();
        let mut behind = None;
        for cp in std::mem::take(&mut lane.checkpoints) {
            if cp.pos == tail && store.enabled() {
                // The prompt's end goes straight to system RAM: only an exact retry resumes there.
                let drafter = self.drafter_context(lane);
                let id = store.insert(
                    &self.b,
                    &lane.session,
                    &lane.keys,
                    &cp,
                    false,
                    None,
                    drafter,
                )?;
                store.mark_tail(id, head.unwrap_or(0));
                lane.spare.push(cp);
                continue;
            }
            if store.enabled()
                && cp.pos >= MIN_REUSE
                && anchors.contains(&cp.pos)
                && !store.holds(&lane.keys[..cp.pos])
            {
                let drafter = self.drafter_context(lane);
                store.insert(&self.b, &lane.session, &lane.keys, &cp, true, None, drafter)?;
            }
            if Some(cp.pos) == head {
                lane.checkpoints.push(cp);
                continue;
            }
            // The point this request resumed from (the previous step's history point) is behind
            // the new head now: the store keeps it superseded, for a retry or rewind of this step.
            if self.cfg.chain_superseded > 0
                && cp.pos == resumed
                && head.is_some_and(|h| h > cp.pos)
                && cp.pos >= MIN_CAPTURE
                && !anchors.contains(&cp.pos)
            {
                behind = Some(cp);
                continue;
            }
            lane.spare.push(cp);
        }
        if let Some(cp) = behind {
            // Last, and only when a state is ready for it: a fallback is not worth pinning memory
            // mid-request, nor taking the state another capture of this request needs.
            if store.has_spare_state() && !store.holds(&lane.keys[..cp.pos]) {
                let drafter = self.drafter_context(lane);
                let id = store.insert(
                    &self.b,
                    &lane.session,
                    &lane.keys,
                    &cp,
                    false,
                    None,
                    drafter,
                )?;
                store.supersede(id, self.cfg.chain_superseded);
            }
            lane.spare.push(cp);
        }
        // Device memory: one checkpoint per lane plus the shared spares (see `DEVICE_SPARE`).
        Ok(())
    }

    /// The lane's DFlash drafter context, to be saved with a context-store entry.
    #[allow(clippy::type_complexity)]
    fn drafter_context<'a>(
        &'a self,
        lane: &'a Lane<B>,
    ) -> Option<(&'a DFlashModel<B>, &'a DFlashState<B>)> {
        match (lane.drafter.as_ref(), self.dflash.as_ref()) {
            (Some(Drafter::DFlash(d)), Some(dm)) => Some((dm, d.context())),
            _ => None,
        }
    }

    /// Before a request overwrites a lane that holds another conversation, stores the lane's
    /// deepest checkpoint beyond the `shared` prefix (that conversation's chain head) so it can
    /// resume there later instead of prefilling its history again.
    fn capture_head(
        &self,
        lane: &Lane<B>,
        store: &mut ContextStore<B>,
        shared: usize,
    ) -> Result<(), CoreError> {
        let head = lane
            .checkpoints
            .iter()
            .filter(|c| c.pos > shared && c.pos <= lane.keys.len() && c.pos >= MIN_CAPTURE)
            .max_by_key(|c| c.pos);
        if let Some(cp) = head {
            let drafter = self.drafter_context(lane);
            store.insert(
                &self.b,
                &lane.session,
                &lane.keys,
                cp,
                false,
                lane.chain,
                drafter,
            )?;
        }
        Ok(())
    }

    /// Restores the best reusable state for `keys`: the lane's own state or checkpoints, or a
    /// context-store entry when that covers more. Returns how many tokens are already in the
    /// state and the store entry restored, if any.
    fn reuse(
        &self,
        lane: &mut Lane<B>,
        store: &mut ContextStore<B>,
        keys: &[u32],
    ) -> Result<(usize, Option<u64>), CoreError> {
        // At least the last prompt token must be fed to produce logits.
        let limit = keys.len() - 1;
        let shared = common_prefix(&lane.keys, keys);
        let target = shared.min(limit);
        let live = lane.session.pos == lane.keys.len() && lane.keys.len() <= target;
        let best = lane
            .checkpoints
            .iter()
            .filter(|c| c.pos <= target)
            .max_by_key(|c| c.pos);
        let own = if live {
            lane.keys.len()
        } else {
            best.map_or(0, |c| c.pos)
        };
        let reach = limit.min(lane.session.capacity);
        let resident = own.max(store.best_pos(keys, reach));
        if store.promote(
            &self.b,
            &lane.session,
            keys,
            reach,
            resident,
            DISK_GAIN,
            self.dflash.as_ref(),
        )? && std::env::var_os("VELOX_DEBUG_PREFIX").is_some()
        {
            eprintln!("[disk] read back a prefix beyond {resident}");
        }
        if let Some(e) = store.best(keys, reach) {
            if e.pos() >= own + STORE_GAIN {
                lane.session.restore(&self.b, &e.state)?;
                // KV rows below the shared prefix are already the same.
                for seg in &e.kv {
                    lane.session.load_kv(&self.b, seg, shared)?;
                }
                // The drafter continues from the context saved with the entry; without one, its
                // context past the shared prefix belongs to the other conversation and is dropped
                // (MTP KV rows there only cost acceptance until overwritten).
                if let (Some(Drafter::DFlash(d)), Some(dm)) =
                    (lane.drafter.as_mut(), self.dflash.as_ref())
                {
                    match &e.draft {
                        Some(ctx) => d.restore_context(&self.b, dm, ctx)?,
                        None => d.forget(dm, shared),
                    }
                }
                return Ok((e.pos(), Some(e.id)));
            }
        }
        if live {
            return Ok((own, None));
        }
        match best {
            Some(cp) => lane.session.restore(&self.b, cp)?,
            None => lane.session.restore(&self.b, &lane.zero)?,
        }
        Ok((lane.session.pos, None))
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

    /// Prepares feeding `p.ids[start..]` to the lane: vision embeddings of the images past
    /// `start`, and storage for the snapshots to take on the way (the deepest checkpoint position,
    /// `anchors` the context store does not hold yet, and the prompt's end).
    #[allow(clippy::too_many_arguments)]
    fn begin_prefill(
        &self,
        lane: &mut Lane<B>,
        store: &ContextStore<B>,
        p: &Prompt,
        images: &[ImagePatches],
        start: usize,
        anchors: Vec<usize>,
        t0: Instant,
    ) -> Result<Prefill<B>, CoreError> {
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
        // Checkpoints are filled inside the forward pass, so the prompt is not cut into short
        // segments there (each segment would stream every weight once more).
        let mut positions = self.checkpoint_positions(&p.ids, start);
        let deepest = positions.last().copied();
        // The end of the prompt (less the token fed for logits): an exact retry or replay of this
        // request resumes there (see `place_checkpoints`).
        let tail = p.ids.len() - 1;
        if self.cfg.host_checkpoints
            && store.enabled()
            && tail > start
            && Some(tail) > deepest
            && !store.holds(&p.keys[..tail])
        {
            positions.push(tail);
        }
        let mut snaps = Vec::new();
        for pos in positions {
            let anchor = anchors.contains(&pos) && store.enabled() && !store.holds(&p.keys[..pos]);
            if Some(pos) != deepest && pos != tail && !(anchor || !self.cfg.host_checkpoints) {
                continue;
            }
            // Storage: spare, else the lane's oldest checkpoint (its restore, if any, is queued
            // before the snapshot), else new.
            let mut cp = match lane.spare.pop() {
                Some(cp) => cp,
                None if !lane.checkpoints.is_empty() && self.cfg.host_checkpoints => {
                    lane.checkpoints.sort_by_key(|c| c.pos);
                    lane.checkpoints.remove(0)
                }
                None => lane.session.blank_checkpoint(&self.b, pos)?,
            };
            cp.pos = pos;
            cp.rope_delta = p.rope_delta;
            snaps.push(cp);
        }
        Ok(Prefill {
            start,
            done: 0,
            chunks: 0,
            embeds,
            snaps,
            anchors,
            t0,
        })
    }

    /// Request-side preparation: the rendered, tokenized prompt with its images, the sampler and
    /// the completion budget.
    fn prepare(
        &self,
        req: &Value,
    ) -> Result<(Prompt, Vec<ImagePatches>, Sampler, usize, Option<Value>), ApiError> {
        let t_req = Instant::now();
        let Normalized {
            messages,
            tools,
            images: urls,
        } = normalize_request(req)?;
        let thinking = req
            .pointer("/chat_template_kwargs/enable_thinking")
            .and_then(Value::as_bool)
            .unwrap_or(self.cfg.enable_thinking);
        let text = self
            .fe
            .template
            .render(&messages, tools.as_ref(), thinking, true)
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
        let sampler = Sampler::from_request(req, thinking);
        Ok((p, images, sampler, max_tokens, tools))
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
        let core_ = &self.core;
        let (p, images, sampler, max_tokens, tools) = core_.prepare(req)?;
        let prompt_tokens = p.ids.len() as u64;
        let (tx, rx) = mpsc::channel();
        {
            let mut q = self
                .queue
                .jobs
                .lock()
                .map_err(|_| core(CoreError::Backend("engine queue poisoned".into())))?;
            q.push_back(Job {
                p,
                images,
                sampler,
                max_tokens,
                reply: tx,
            });
        }
        self.queue.cv.notify_one();
        let done = rx
            .recv()
            .map_err(|_| core(CoreError::Backend("engine worker stopped".into())))??;
        let text = core_.fe.decode(&done.out).map_err(core)?;
        let parsed = parse_output(&text, tools.as_ref());
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
                done.finish.into()
            },
            prompt_tokens,
            cached_tokens: done.start as u64,
            completion_tokens: done.out.len() as u64,
            prompt_ms: done.prompt_ms,
            predicted_ms: done.predicted_ms,
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
