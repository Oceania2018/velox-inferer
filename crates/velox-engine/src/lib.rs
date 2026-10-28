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
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use serde_json::{json, Value};
use velox_core::{Backend, Error as CoreError, KvFormat, StateFormat};
use velox_frontend::image::{
    decode_data_url, preprocess, smart_resize, ImagePatches, MERGE, PATCH, TEMPORAL,
};
use velox_frontend::{parse_output, Frontend};
use velox_models::qwen3_5::{
    shift_inserts, BatchWorkspace, Checkpoint, DFlashModel, DFlashState, DFlashWorkspace,
    EmbedInsert, KvPool, KvShare, PrefillChunk, Qwen35Text, Qwen35Vision, Session, Workspace,
    DRAFT_PAGE_ROWS, KV_PAGE_ROWS,
};
use velox_serve::{ApiError, ChatEngine, Delta, Outcome, ScoreOutcome, ScoreQuery};

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
    /// A request that offers tools answers with a tool call only: the prompt ends in the opening
    /// `<tool_call>` tag, so the model skips the sentence it would otherwise write first (ninfer's
    /// tool-call grammar constraint has the same effect; the agent reads the call's own `why`).
    pub force_tool_call: bool,
    /// Streaming requests receive their text as it is generated (ninfer does the same), up to the
    /// first `<tool_call>`; off: the whole answer at the end. Tool calls and reasoning always
    /// arrive whole.
    pub stream_content: bool,
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
    /// Storage format of the recurrent states the context store keeps (ninfer
    /// `--host-state-dtype`): bf16 holds twice as many prefixes as f32 in the same RAM, nvfp4
    /// about 5.5 times; lanes always run on f32 state.
    pub host_state_format: StateFormat,
    /// NVMe tier below the context store (see `disk`): roots and branch points several requests
    /// resumed from; 0 disables it.
    pub disk_context_bytes: u64,
    /// Directory of the NVMe tier's files.
    pub disk_context_dir: std::path::PathBuf,
    /// Identity of the loaded model (weights and kernels): records are kept across restarts in a
    /// subdirectory named after it and the storage layout. `None`: a private subdirectory removed
    /// on shutdown.
    pub disk_context_tag: Option<String>,
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
    /// Page the lanes' attention caches in virtual memory when the backend can: backed as each
    /// sequence grows instead of at full capacity, with identical prefix chunks (system prompt,
    /// task) mapped once for every lane that holds them.
    pub kv_paging: bool,
    /// Paged caches: tokens of attention cache in the startup pool every lane draws from (`None`:
    /// what device memory allows after everything else, minus `vram_headroom`).
    pub kv_capacity: Option<usize>,
    /// Device memory left unallocated at startup (paged caches; WDDM spills to system memory
    /// when the card fills, slowing every step several times).
    pub vram_headroom: usize,
    /// Device memory recorded graphs may hold; past it the least recently run are released.
    pub graph_budget: usize,
    /// Device memory in use once started (every process, graph budget included): the attention
    /// pool gets what is left. Takes precedence over `vram_headroom`.
    pub vram_budget: Option<usize>,
    /// System One rank branches scored per forward pass (ninfer `--score-branch-rows`; 0
    /// rejects branch scoring). Each takes about 0.5 MB of scoring buffers on Qwen3.8-27B.
    pub score_rows: usize,
}

/// Tokens of one scoring window, at most: the prompt's last token, then a rank branch
/// (`"<id>] "`, at most five tokens).
const SCORE_WINDOW: usize = 1 + velox_serve::systemone::MAX_BRANCH_TOKENS;

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
    /// The lane's latest request was a scoring request: its state is not worth keeping for
    /// later (see `pick_lane`, `evict_heads`).
    scored: bool,
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
    /// Scoring buffers (System One): `score_rows` windows per pass, at least one.
    sw: BatchWorkspace<B>,
    clock: u64,
    /// Checkpoint storage not in use, shared by the lanes (lent to the lane being admitted):
    /// prefills snapshot into it instead of allocating, which would synchronize the device. Holds
    /// the device checkpoints (see [`DEVICE_SPARE`]) not in a lane, and host ones up to its trim.
    spare: Vec<Checkpoint<B>>,
    /// Published prefix chunks of the lanes' paged attention caches.
    share: KvShare<B>,
    /// The chunk sets those caches draw from, created at startup (paged caches only).
    pool: Option<Arc<KvPool<B>>>,
}

/// Device checkpoint storage kept for reuse across lanes.
const DEVICE_SPARE: usize = 2;

/// Device checkpoints (one recurrent state each, 157 MB for Qwen3.8-27B), created at startup
/// inside the memory budget and never more: the lanes' kept heads ([`DEVICE_HEADS`]) plus
/// [`DEVICE_SPARE`], each at most one per lane (`budget_memory`). A prefill snapshot finding none
/// free takes pinned host storage instead; allocating them per prefill took ~1.4 GB for seconds
/// when six fresh prompts started together.

/// Opening tag of a Qwen tool call, as the chat template writes it (see
/// `EngineConfig::force_tool_call`).
const TOOL_CALL_OPEN: &str = "<tool_call>\n";

/// The part of a partial answer that is content whatever follows: what `parse_output` returns
/// as content for every continuation. Held back: text from the first `<tool_call>` on, a trailing
/// piece of a tag, surrounding whitespace (the parser trims it) and an incomplete UTF-8
/// character. `None` once the answer has reasoning tags (reasoning is not content).
fn streamable_content(text: &str) -> Option<&str> {
    const TAGS: [&str; 3] = ["<tool_call>", "<think>", "</think>"];
    if text.contains("<think>") || text.contains("</think>") {
        return None;
    }
    let mut end = text.find(TAGS[0]).unwrap_or(text.len());
    let head = &text[..end];
    for tag in TAGS {
        // Tags are ASCII, so `end` stays on a character boundary.
        if let Some(k) = (1..tag.len()).rev().find(|&k| head.ends_with(&tag[..k])) {
            end = end.min(head.len() - k);
        }
    }
    Some(text[..end].trim_end_matches('\u{FFFD}').trim())
}

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

/// Paged caches: rows past a new prompt a lane keeps backed (room for its output) before the rest
/// is returned. Without the trim every lane keeps its longest prompt ever backed, and on a 32 GB
/// card six lanes near capacity push VRAM into WDDM's system-memory spill (decode 3x slower).
const KV_TRIM_SLACK: usize = 4096;

/// Device memory a budget keeps back for kernel scratch that grows after warm-up (grow-only
/// buffers sized by the largest call so far; a 6-way stress run went ~0.2 GB past warm-up).
const SCRATCH_RESERVE: usize = 256 << 20;

/// Prefill snapshots by storage: a device checkpoint, a pinned state the store had ready, or
/// one pinned on the request path (logged with device memory).
static SNAPS_DEVICE: AtomicUsize = AtomicUsize::new(0);
static SNAPS_HOST: AtomicUsize = AtomicUsize::new(0);
static SNAPS_PINNED: AtomicUsize = AtomicUsize::new(0);

/// How often the engine logs device memory (free, attention pool, graphs).
const MEM_LOG_EVERY: std::time::Duration = std::time::Duration::from_secs(300);

pub struct QwenEngine<B: Backend> {
    core: Arc<Core<B>>,
    queue: Arc<JobQueue>,
    worker: Option<JoinHandle<()>>,
    /// Recent scoring answers (see `ScoreCache`).
    results: ScoreCache,
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

/// Keeps at most `max` host checkpoints in `spare` and every device one: device checkpoints
/// are a fixed set (see [`DEVICE_SPARE`]), so dropping one would only shrink it.
fn trim_spare<B: Backend>(spare: &mut Vec<Checkpoint<B>>, max: usize) {
    let mut host = 0;
    spare.retain(|c| {
        if !c.on_host() {
            return true;
        }
        host += 1;
        host <= max
    });
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
            // Named by type only: the server logs error messages, and the value is request data.
            Some(other) => {
                let kind = match other {
                    Value::Bool(_) => "a boolean",
                    Value::Number(_) => "a number",
                    _ => "an object",
                };
                return Err(ApiError::invalid(format!(
                    "message content must be a string or an array of parts, not {kind}"
                )));
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
    /// Send the output tokens as each round commits them (`Reply::Tokens`), not only at the end.
    stream: bool,
    reply: mpsc::Sender<Reply>,
    queued: Instant,
    /// A scoring request: nothing is generated (see [`Scoring`]).
    score: Option<Scoring>,
}

/// What a scoring request scores once its prompt is in the lane (see `Core::finish_prefill`).
struct Scoring {
    /// Ids whose log-probabilities are read after the prompt's last token, or after each window.
    candidates: Vec<u32>,
    /// Token windows scored from the prompt's state, each as if it alone followed the prompt
    /// (none: the candidates are read off the prompt's own last row).
    windows: Vec<Vec<u32>>,
    /// Where the prompts of the request's later questions part from this one: the lane keeps a
    /// checkpoint there, so they resume after the shared state and images.
    fork: Option<usize>,
}

/// What the worker sends a request's HTTP thread: committed tokens while it runs (streaming
/// jobs), then the result.
enum Reply {
    Tokens(Vec<u32>),
    Done(Result<Done, ApiError>),
}

/// What the worker hands back for a finished request.
struct Done {
    out: Vec<u32>,
    /// A scoring request's candidate log-probabilities, window-major.
    scores: Vec<f32>,
    finish: &'static str,
    start: usize,
    queue_ms: f64,
    prompt_ms: f64,
    predicted_ms: f64,
    first_token: Instant,
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
    /// Output tokens already sent to a streaming job.
    sent: usize,
    /// The sampled token to feed next.
    c: u32,
    finish: Option<&'static str>,
    start: usize,
    /// From hand-off to admission onto the lane.
    queue_ms: f64,
    prompt_ms: f64,
    /// When the first token was sampled.
    t1: Instant,
    /// MTP: cache row of the previous verify and its kept drafts (see `MtpDrafter::draft`).
    mtp_row0: usize,
    mtp_kept: Vec<u32>,
    /// Plain decoding: the candidates after the last fed token.
    cands: Vec<(u32, f32)>,
    /// While the prompt is still being fed.
    pre: Option<Prefill<B>>,
    /// A scoring request's result (see `Done::scores`).
    scores: Vec<f32>,
}

impl<B: Backend> Active<B> {
    /// Sends a streaming job the output it has not seen yet (`complete` sends the rest).
    fn send_new(&mut self) {
        if self.job.stream && self.out.len() > self.sent {
            let _ = self
                .job
                .reply
                .send(Reply::Tokens(self.out[self.sent..].to_vec()));
            self.sent = self.out.len();
        }
    }
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
    B::Chunk: Send + Sync + 'static,
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
            let mut session = if cfg.kv_paging {
                model.session_paged(&b, cfg.capacity, cfg.kv_format)?
            } else {
                model.session_with(&b, cfg.capacity, cfg.kv_format)?
            };
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
                scored: false,
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
            // Windows are the pending token plus the drafts (see `DFlashDrafter::workspace`).
            Some(_) if n_lanes > 1 => Some(model.batch_workspace(
                &b,
                n_lanes,
                DFlashDrafter::<B>::max_drafts(cfg.dflash_drafts) + 1,
            )?),
            _ => None,
        };
        // Before warm-up and the memory budget, so the attention pool is sized without them.
        let sw =
            model.score_workspace(&b, cfg.score_rows.clamp(1, ws.max_tokens()), SCORE_WINDOW)?;
        let special = SpecialTokens {
            im_start: single_token(&fe, "<|im_start|>"),
            vision_start: single_token(&fe, "<|vision_start|>"),
            image_pad: single_token(&fe, "<|image_pad|>"),
        };
        let disk = if cfg.disk_context_bytes > 0 && cfg.host_context_bytes > 0 {
            // Records hold raw storage: they only fit sessions of the same shapes and formats.
            let session = &lanes[0].session;
            let layout = format!(
                "{:?} {} {} {} {} {:?}",
                session.state_lens(&b, cfg.host_state_format),
                session.kv_page_len(),
                KV_PAGE_ROWS,
                cfg.kv_format,
                cfg.host_state_format,
                dflash.as_ref().map(|dm| dm.context_page_len()),
            );
            Some(
                DiskTier::new(
                    cfg.disk_context_dir.clone(),
                    cfg.disk_context_bytes,
                    cfg.disk_context_tag.as_deref(),
                    disk::fnv64(disk::FNV_OFFSET, layout.as_bytes()),
                )
                .map_err(|e| CoreError::Config(format!("disk context tier: {e}")))?,
            )
        } else {
            None
        };
        let mut store = ContextStore::new(
            cfg.host_context_bytes,
            cfg.lrfu_lambda,
            disk,
            DISK_GAIN,
            cfg.host_state_format,
        );
        // Pin the store's memory before serving: until the store fills, every capture would
        // otherwise pin fresh memory on the request path (+90 ms on a 3K-token incremental
        // prefill, +140-600 ms on a 20K one, RTX 5090 under WSL). A sixteenth more covers the
        // free storage the store keeps beside its entries.
        if store.enabled() {
            let t = Instant::now();
            let want = cfg.host_context_bytes + cfg.host_context_bytes / 16;
            let got = b.reserve_host(want);
            if got > 0 {
                eprintln!(
                    "context store: pinned {} MiB in {:.1} s",
                    got >> 20,
                    t.elapsed().as_secs_f64()
                );
            }
        }
        // Opt-in (VELOX_PINNER=1): under WSL/WDDM, pinning on another thread stalled the device
        // (6-agent replays 74-79 s against 37-44 s) more than pinning on the engine thread costs.
        if store.enabled() && std::env::var("VELOX_PINNER").is_ok_and(|v| v == "1") {
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
            sw,
            clock: 0,
            spare: Vec::new(),
            share: KvShare::default(),
            pool: None,
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
        if let Some(used) = core.b.device_in_use() {
            eprintln!(
                "memory: {:.2} GiB in use with {} lanes' states and the workspaces",
                used as f64 / (1u64 << 30) as f64,
                state.lanes.len()
            );
        }
        core.warm_up(&mut state)?;
        // Warm-up wrote near the end of the caches; paged lanes start empty.
        for lane in &mut state.lanes {
            lane.session.trim_kv(&core.b, 0)?;
        }
        core.budget_memory(&mut state)?;
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
            results: ScoreCache::default(),
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
    /// Fixes device memory at startup, as ninfer does: recorded graphs get `graph_budget`, and the
    /// paged caches' chunks are created now, as many as `kv_capacity` asks or the device has
    /// left after everything else and `vram_headroom`. Requests then allocate no device memory
    /// for caches, so VRAM stays flat instead of creeping into WDDM's system-memory spill.
    fn budget_memory(&self, st: &mut Lanes<B>) -> Result<(), CoreError> {
        let cfg = &self.cfg;
        // The device checkpoints there will ever be, before the measurement below counts them.
        // One lane needs its head and one more for the next prefill's deepest snapshot.
        let lanes = st.lanes.len();
        let states = DEVICE_HEADS.min(lanes) + DEVICE_SPARE.min(lanes);
        let device = st.spare.iter().filter(|c| !c.on_host()).count();
        for _ in device..states {
            st.spare
                .push(st.lanes[0].session.blank_checkpoint(&self.b, 0)?);
        }
        self.b.set_graph_budget(cfg.graph_budget);
        let Some((per_set, rows_per_set)) = st.lanes[0].session.kv_set_shape() else {
            eprintln!("attention caches: {} lanes, whole", st.lanes.len());
            return Ok(());
        };
        let set_bytes = per_set * self.b.vm_chunk_len() * 4;
        let full = st.lanes.len() * st.lanes[0].session.kv_sets_for(cfg.capacity);
        let (graphs, graph_bytes, _) = self.b.graph_stats();
        let gib = |x: usize| x as f64 / (1u64 << 30) as f64;
        let used = self.b.device_in_use();
        if let Some(used) = used {
            eprintln!(
                "memory: {:.2} GiB in use after warm-up, before the attention pool",
                gib(used)
            );
        }
        let graphs_to_come = cfg.graph_budget.saturating_sub(graph_bytes);
        // The budget is in nvidia-smi's terms (device_in_use); the headroom is kept from what
        // CUDA says it can still allocate (mem_info), the stricter figure under WDDM.
        let sets = match (cfg.kv_capacity, cfg.vram_budget, used, self.b.mem_info()) {
            (Some(tokens), ..) => tokens.div_ceil(rows_per_set.max(1)),
            (None, Some(budget), Some(used), _) => {
                let room = budget.saturating_sub(used + graphs_to_come + SCRATCH_RESERVE)
                    / set_bytes.max(1);
                // One full-length request must always fit.
                let floor = st.lanes[0].session.kv_sets_for(cfg.capacity);
                if room < floor {
                    eprintln!(
                        "memory budget: {:.2} GiB in use and {:.2} GiB of graph budget leave \
                         {room} attention sets under --vram-budget-mib; taking {floor} (one full \
                         context) over it",
                        gib(used),
                        gib(graphs_to_come)
                    );
                }
                room.max(floor)
            }
            (None, _, _, Some((free, _))) => {
                free.saturating_sub(cfg.vram_headroom + graphs_to_come) / set_bytes.max(1)
            }
            // No way to measure: every lane's full capacity.
            _ => full,
        }
        .min(full);
        let pool = KvPool::new(&self.b, sets, per_set)?;
        for lane in &mut st.lanes {
            lane.session.set_kv_pool(Arc::clone(&pool));
        }
        st.pool = Some(pool);
        eprintln!(
            "memory budget: attention pool {sets} sets of {rows_per_set} rows ({} tokens, {:.2} GiB; \
             all lanes full: {full}), graphs up to {:.2} GiB ({graphs} recorded, {:.2} GiB); \
             in use now {}",
            sets * rows_per_set,
            gib(sets * set_bytes),
            gib(cfg.graph_budget),
            gib(graph_bytes),
            self.b
                .device_in_use()
                .map_or("unknown".into(), |u| format!("{:.2} GiB", gib(u)))
        );
        Ok(())
    }

    /// Room in the attention pool for a prompt of `len` tokens on lane `li` (with output room):
    /// idle lanes, least recently used first, give their chunks back (their conversations resume
    /// from the context store). False when that is not enough while other requests run: the
    /// request waits for them. Alone, it runs anyway (the pool lends past its size).
    fn kv_room(&self, st: &mut Lanes<B>, li: usize, len: usize, busy: &[usize]) -> bool {
        let Some(pool) = st.pool.clone() else {
            return true;
        };
        let need = st.lanes[li].session.kv_need(len + KV_TRIM_SLACK);
        while pool.free_sets() < need {
            let Some(j) = (0..st.lanes.len())
                .filter(|&j| j != li && !busy.contains(&j))
                .filter(|&j| st.lanes[j].session.kv_private_sets() > 1)
                .min_by_key(|&j| st.lanes[j].last_used)
            else {
                return busy.is_empty();
            };
            if let Err(e) = self.release_lane(st, j) {
                eprintln!("attention pool: releasing lane {j} failed: {e}");
                return busy.is_empty();
            }
        }
        true
    }

    fn log_memory(&self, st: &Lanes<B>) {
        let gb = |x: usize| x as f64 / 1e9;
        let (graphs, graph_bytes, released) = self.b.graph_stats();
        let pool = st.pool.as_ref().map_or(String::new(), |p| {
            format!(
                ", attention pool {}/{} sets free ({} lent past it)",
                p.free_sets(),
                p.sets(),
                p.overflow()
            )
        });
        let snaps = format!(
            ", snapshots {} device / {} host ({} pinned on demand)",
            SNAPS_DEVICE.load(Ordering::Relaxed),
            SNAPS_HOST.load(Ordering::Relaxed),
            SNAPS_PINNED.load(Ordering::Relaxed)
        );
        let allocs = self
            .b
            .device_allocs()
            .map_or(String::new(), |(live, peak, chunks)| {
                format!(
                    ", buffers {:.2} GiB (peak {:.2}), chunks {:.2} GiB",
                    live as f64 / (1u64 << 30) as f64,
                    peak as f64 / (1u64 << 30) as f64,
                    chunks as f64 / (1u64 << 30) as f64
                )
            });
        eprintln!(
            "[mem] device in use {}{pool}{allocs}{snaps}, graphs {graphs} ({:.2} GB, {released} \
             released)",
            self.b.device_in_use().map_or("unknown".into(), |u| format!(
                "{:.2} GiB",
                u as f64 / (1u64 << 30) as f64
            )),
            gb(graph_bytes)
        );
    }

    /// Returns idle lane `j`'s attention chunks to the pool; its chain head goes to the context
    /// store first, so its conversation resumes from there.
    fn release_lane(&self, st: &mut Lanes<B>, j: usize) -> Result<(), CoreError> {
        let Lanes {
            lanes,
            store,
            spare,
            ..
        } = st;
        let lane = &mut lanes[j];
        if store.enabled() {
            self.capture_head(lane, store, 0)?;
        }
        let n = lane.session.trim_kv(&self.b, 0)?;
        lane.keys.clear();
        lane.chain = None;
        spare.append(&mut lane.checkpoints);
        trim_spare(spare, DEVICE_SPARE);
        if std::env::var_os("VELOX_DEBUG_PREFIX").is_some() {
            eprintln!("[kv] lane {j} released {n} chunks to the pool");
        }
        Ok(())
    }

    fn warm_up(&self, st: &mut Lanes<B>) -> Result<(), CoreError> {
        use velox_models::qwen3_5::VERIFY_MAX;
        let (b, model) = (&self.b, &self.model);
        let limit = self.fe.public_vocab;
        let chunk = self.cfg.chunk;
        let Lanes {
            ws,
            lanes,
            dw,
            bw,
            sw,
            ..
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
            // The widest scoring pass (System One rank questions).
            let window = [0u32; SCORE_WINDOW];
            let n = sw.max_seqs().min(ws.max_tokens() / SCORE_WINDOW).max(1);
            model.score_windows(
                b,
                &mut lane.session,
                ws,
                sw,
                &vec![&window[..]; n],
                &[0],
                limit,
            )?;
            if lane.drafter.is_some() {
                let window = vec![0u32; VERIFY_MAX.min(lane.session.capacity - lane.session.pos)];
                model.verify(b, &mut lane.session, ws, &window, CANDIDATES, limit)?;
                model.commit(b, &mut lane.session, ws, 1)?;
            }
            if let (Some(Drafter::DFlash(d)), Some(dm), Some(dw)) =
                (lane.drafter.as_mut(), self.dflash.as_ref(), dw.as_mut())
            {
                d.drafts = d.long_block();
                d.draft(
                    b,
                    dm,
                    dw,
                    model,
                    ws,
                    0,
                    0,
                    lane.session.pos,
                    0,
                    limit,
                    &Sampler::greedy(),
                )?;
            }
        }
        if let Some(bw) = bw.as_mut() {
            let m = bw.window();
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
        let mut mem_logged = Instant::now();
        // Drafter context pages kept ready: a new conversation's captures save two windows, so
        // a burst of them (see the store's RESERVE_STATES) needs several.
        const DRAFT_RESERVE_WINDOWS: usize = 6;
        let draft = self.dflash.as_ref().map(|dm| {
            (
                DRAFT_RESERVE_WINDOWS * dm.context_window().div_ceil(DRAFT_PAGE_ROWS),
                dm.context_page_len(),
            )
        });
        loop {
            if mem_logged.elapsed() >= MEM_LOG_EVERY {
                mem_logged = Instant::now();
                self.log_memory(&st);
            }
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
                if let Err(e) = st.store.persist(&self.b) {
                    eprintln!("disk context: {e}");
                }
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
                    let _ = j.reply.send(Reply::Done(Err(gone())));
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
                if !self.kv_room(&mut st, li, waiting[0].p.keys.len(), &busy) {
                    break;
                }
                let job = waiting.pop_front().expect("checked");
                match self.begin(&mut st, li, job) {
                    Ok(a) => {
                        active.push(a);
                        if let Err(e) = self.evict_heads(&mut st, li) {
                            eprintln!("context store: moving a chain head failed: {e}");
                        }
                    }
                    Err((job, e)) => {
                        let _ = job.reply.send(Reply::Done(Err(e)));
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
                    let _ = a.job.reply.send(Reply::Done(Err(e.clone())));
                }
                continue;
            }
            // Streaming jobs get the tokens this round committed.
            for a in active.iter_mut() {
                a.send_new();
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
                    && !lane.scored
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
        trim_spare(spare, DEVICE_SPARE);
        Ok(())
    }

    /// A free lane for a prompt of `keys`: the one that can reuse the most, else one whose latest
    /// request was a scoring one (nothing there is kept for later), else the least recently used
    /// (its conversation is the least likely to come back soon; its chain head goes to the
    /// context store when the new request takes it).
    fn pick_lane(lanes: &[Lane<B>], busy: &[usize], keys: &[u32]) -> Option<usize> {
        let free: Vec<usize> = (0..lanes.len()).filter(|i| !busy.contains(i)).collect();
        let (best, shared) = free
            .iter()
            .map(|&i| (i, common_prefix(&lanes[i].keys, keys)))
            .max_by_key(|&(i, s)| (s, std::cmp::Reverse(i)))?;
        if shared >= MIN_REUSE {
            return Some(best);
        }
        free.into_iter()
            .min_by_key(|&i| (!lanes[i].scored, lanes[i].last_used))
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
            share,
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
        // Hot entries the previous requests stored go to disk in the background.
        if let Err(e) = store.persist(&self.b) {
            eprintln!("disk context: {e}");
        }
        let shared = common_prefix(&lane.keys, &p.keys);
        let r = (|| -> Result<Prefill<B>, CoreError> {
            if store.enabled() {
                self.capture_head(lane, store, shared.min(p.keys.len() - 1))?;
            }
            let scoring = job.score.is_some();
            lane.scored = scoring;
            // Windows follow a prompt that is complete as a state: it may come whole from one.
            let whole = job.score.as_ref().is_some_and(|s| !s.windows.is_empty());
            let (start, adopted) = self.reuse(lane, store, share, &p.keys, whole)?;
            lane.chain = store.resumed(&p.keys, start, adopted);
            if scoring {
                // A scoring prompt is fed without the drafter (nothing is drafted after it), so
                // the drafter's context ends where the prompt resumes.
                if let (Some(Drafter::DFlash(d)), Some(dm)) =
                    (lane.drafter.as_mut(), self.dflash.as_ref())
                {
                    d.forget(dm, start);
                }
            }
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
            // No checkpoint left reaches past `start`, so rows past this prompt hold nothing.
            let trimmed = lane
                .session
                .trim_kv(&self.b, p.keys.len() + KV_TRIM_SLACK)?;
            // The prompt's rows are backed now rather than as its chunks run, so the next
            // admission's room check (`kv_room`) sees this request's share of the pool taken.
            lane.session.reserve_kv(&self.b, p.keys.len())?;
            if trimmed > 0 && debug {
                eprintln!(
                    "[kv] lane {li} returned {trimmed} chunks, {:.2} GB backed",
                    lane.session.kv_bytes(&self.b) as f64 / 1e9
                );
            }
            // Scoring prompts place no roots in the context store: each would take reserved
            // storage, pinned again on the engine thread before the next request is admitted.
            let anchors: Vec<usize> = self
                .checkpoint_positions(&p.ids, 0)
                .into_iter()
                .take(if scoring { 0 } else { 2 })
                .collect();
            let fork = job.score.as_ref().and_then(|s| s.fork);
            self.begin_prefill(
                lane,
                store,
                p,
                &job.images,
                start,
                anchors,
                fork,
                scoring,
                t0,
            )
        })();
        trim_spare(&mut lane.spare, DEVICE_SPARE);
        pool.append(&mut lane.spare);
        if debug {
            // Admission runs on the engine thread between rounds; with VELOX_DEBUG_BATCH its GPU
            // work (state restore, vision encode) is waited for so the time is the whole cost.
            if std::env::var_os("VELOX_DEBUG_BATCH").is_some() {
                let _ = self.b.synchronize();
            }
            eprintln!(
                "[admit] lane {li} from {} in {:.1} ms, queued {:.1} ms",
                r.as_ref().map_or(0, |pre| pre.start),
                t0.elapsed().as_secs_f64() * 1e3,
                t0.duration_since(job.queued).as_secs_f64() * 1e3
            );
        }
        let queue_ms = t0.duration_since(job.queued).as_secs_f64() * 1e3;
        match r {
            Ok(pre) => Ok(Active {
                job,
                lane: li,
                out: Vec::new(),
                sent: 0,
                c: 0,
                finish: None,
                start: pre.start,
                queue_ms,
                prompt_ms: 0.0,
                t1: Instant::now(),
                mtp_row0: 0,
                mtp_kept: Vec::new(),
                cands: Vec::new(),
                pre: Some(pre),
                scores: Vec::new(),
            }),
            Err(e) => {
                lane.keys.clear();
                lane.checkpoints.clear();
                Err((job, core(e)))
            }
        }
    }

    /// The prompt is in the lane's state: checkpoints placed, the first token sampled from
    /// `cands`; the request decodes from the next round on. A scoring request is scored here
    /// instead and finishes.
    fn finish_prefill(
        &self,
        st: &mut Lanes<B>,
        a: &mut Active<B>,
        cands: Vec<(u32, f32)>,
    ) -> Result<(), CoreError> {
        let debug = std::env::var_os("VELOX_DEBUG_PREFIX").is_some();
        let pre = a.pre.take().expect("a prefilling request");
        let Lanes {
            ws,
            lanes,
            store,
            spare: pool,
            share,
            sw,
            ..
        } = st;
        let lane = &mut lanes[a.lane];
        let p = &a.job.p;
        lane.spare.append(pool);
        lane.keys.extend_from_slice(&p.keys[pre.start..]);
        // The prompt's full chunks become mappable by other lanes on the same prefix.
        lane.session.publish_kv(share, &p.keys);
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
        trim_spare(&mut lane.spare, DEVICE_SPARE);
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
        if let Some(s) = &a.job.score {
            // Nothing is sampled. Without windows the scores were read off the prompt's last row
            // (`prefill_rest`, `batch_round`); otherwise every window is scored from the prompt's
            // state, which the lane keeps for the next request.
            if !s.windows.is_empty() {
                let t0 = Instant::now();
                let refs: Vec<&[u32]> = s.windows.iter().map(|w| &w[..]).collect();
                a.scores = self.model.score_windows(
                    &self.b,
                    &mut lane.session,
                    ws,
                    sw,
                    &refs,
                    &s.candidates,
                    self.fe.public_vocab,
                )?;
                if debug {
                    eprintln!(
                        "[score] lane {} {} windows (up to {} tokens) x {} candidates in {:.1} ms",
                        a.lane,
                        refs.len(),
                        refs.iter().map(|w| w.len()).max().unwrap_or(0),
                        s.candidates.len(),
                        t0.elapsed().as_secs_f64() * 1e3
                    );
                }
            }
            a.finish = Some("score");
            a.t1 = Instant::now();
            return Ok(());
        }
        lane.spec_stats = (0, 0, 0);
        a.c = a.job.sampler.sample(&cands);
        a.cands = cands;
        a.t1 = Instant::now();
        Ok(())
    }

    /// Feeds the next workspace chunk of a prefilling request's prompt (no other request decodes).
    /// One chunk per round: a scoring request that arrives meanwhile is admitted and fed before
    /// the rest of a long prompt (see `round`).
    fn prefill_rest(&self, st: &mut Lanes<B>, a: &mut Active<B>) -> Result<(), CoreError> {
        let cands = {
            let Lanes { ws, lanes, dw, .. } = &mut *st;
            let lane = &mut lanes[a.lane];
            let pre = a.pre.as_mut().expect("a prefilling request");
            let p = &a.job.p;
            let from = pre.start + pre.done;
            let n = (p.ids.len() - from).min(ws.max_tokens());
            let inserts = shift_inserts(&prefill_inserts(&pre.embeds, pre.start, p), pre.done);
            let ids = &p.ids[from..from + n];
            let positions = p.positions.as_ref().map(|v| &v[from..from + n]);
            let Lane {
                session, drafter, ..
            } = &mut *lane;
            // Nothing is drafted after a scoring prompt (see `begin`).
            match drafter.as_mut().filter(|_| a.job.score.is_none()) {
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
            pre.done += n;
            pre.chunks += 1;
            if from + n < p.ids.len() {
                return Ok(());
            }
            match a.job.score.as_ref().filter(|s| s.windows.is_empty()) {
                Some(s) => {
                    a.scores = self.model.last_logprobs(
                        &self.b,
                        ws,
                        &s.candidates,
                        self.fe.public_vocab,
                    )?;
                    Vec::new()
                }
                None => self
                    .model
                    .candidates(&self.b, ws, CANDIDATES, self.fe.public_vocab)?,
            }
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
            // Cumulative since load: diff consecutive lines for a request's own hit rate.
            if let Some(r) = self.model.expert_report() {
                eprintln!(
                    "[experts] hits {} misses {} moves {}",
                    r.hits, r.misses, r.moves
                );
            }
        }
        let _ = a.job.reply.send(Reply::Done(Ok(Done {
            out: a.out,
            scores: a.scores,
            finish: a.finish.unwrap_or("length"),
            start: a.start,
            queue_ms: a.queue_ms,
            prompt_ms: a.prompt_ms,
            predicted_ms,
            first_token: a.t1,
        })));
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
            // The token sampled last round (the first one: at the end of the prefill) goes out
            // now, not after this round's forward pass.
            a.send_new();
        }
        // A scoring prompt already in the lane's state (an exact repeat) is scored right away.
        for a in active.iter_mut() {
            if a.job.score.is_some()
                && a.pre
                    .as_ref()
                    .is_some_and(|p| p.start + p.done == a.job.p.ids.len())
            {
                self.finish_prefill(st, a, Vec::new()).map_err(core)?;
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
        // Scoring prompts go first: their answer follows right after them, while a chat prompt
        // is followed by its whole decode.
        let prefilling = |i: &usize| active[*i].pre.is_some();
        let next = (0..active.len())
            .filter(prefilling)
            .find(|&i| active[i].job.score.is_some())
            .or_else(|| (0..active.len()).find(prefilling));
        if let Some(pi) = next {
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
                &a.job.sampler,
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
                    let (picked, qs) = d.pick(dm, &proposals[j], a.c, &a.job.sampler);
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
                        // The prefilling drafter's context for the chunk (features rows 0 .. tp);
                        // nothing is drafted after a scoring prompt.
                        if let (Some(Drafter::DFlash(d)), Some(dm), None) = (
                            pl.drafter.as_mut(),
                            self.dflash.as_ref(),
                            &active[pi].job.score,
                        ) {
                            d.prefill_chunk(&self.b, dm, dw, ws, row0, tp)?;
                        }
                        // A scoring prompt without windows is answered off its last row.
                        let scores = match (&cands, &active[pi].job.score) {
                            (Some(_), Some(s)) if s.windows.is_empty() => {
                                Some(self.model.last_logprobs(
                                    &self.b,
                                    ws,
                                    &s.candidates,
                                    self.fe.public_vocab,
                                )?)
                            }
                            _ => None,
                        };
                        Ok((cands, rows, scores))
                    })();
                    pstate.done += tp;
                    pstate.chunks += 1;
                    active[pi].pre = Some(pstate);
                    let (cands, rows, scores) = r.map_err(core)?;
                    if let Some(scores) = scores {
                        active[pi].scores = scores;
                    }
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
        trim_spare(spare, MAX_SPARE);
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
                let (id, back) = store.insert_owned(
                    &self.b,
                    &lane.session,
                    &lane.keys,
                    cp,
                    false,
                    None,
                    drafter,
                )?;
                store.mark_tail(id, head.unwrap_or(0));
                lane.spare.extend(back);
                continue;
            }
            if store.enabled()
                && cp.pos >= MIN_REUSE
                && anchors.contains(&cp.pos)
                && !store.holds(&lane.keys[..cp.pos])
            {
                let drafter = self.drafter_context(lane);
                if Some(cp.pos) == head {
                    store.insert(&self.b, &lane.session, &lane.keys, &cp, true, None, drafter)?;
                } else {
                    // Not the head and an anchor (never kept behind): the store owns it now.
                    let (_, back) = store.insert_owned(
                        &self.b,
                        &lane.session,
                        &lane.keys,
                        cp,
                        true,
                        None,
                        drafter,
                    )?;
                    lane.spare.extend(back);
                    continue;
                }
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
        // A scoring prompt's state is not kept for later (an exact repeat is answered from the
        // engine's results).
        if lane.scored {
            return Ok(());
        }
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
        share: &KvShare<B>,
        keys: &[u32],
        whole: bool,
    ) -> Result<(usize, Option<u64>), CoreError> {
        // At least the last prompt token must be fed to produce logits, unless `whole`: scoring
        // windows follow the prompt (the first starts with its held-back last token), so all of
        // it can come from a state.
        let limit = keys.len() - usize::from(!whole);
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
                // Paged caches: chunks of this prefix other lanes hold are mapped instead of
                // copied (load_kv skips them).
                lane.session.retarget_kv(&self.b, keys, e.pos())?;
                let adopted = lane.session.adopt_kv(&self.b, share, keys, e.pos())?;
                if adopted > 0 && std::env::var_os("VELOX_DEBUG_PREFIX").is_some() {
                    eprintln!("[kv] mapped {adopted} shared prefix chunks");
                }
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
        store: &mut ContextStore<B>,
        p: &Prompt,
        images: &[ImagePatches],
        start: usize,
        anchors: Vec<usize>,
        fork: Option<usize>,
        scoring: bool,
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
        // The point the request's later questions resume from is the lane's head (kept on the
        // device), even past the media tail.
        if let Some(f) = fork.filter(|&f| f > start && f < p.ids.len()) {
            positions.push(f);
            positions.sort_unstable();
            positions.dedup();
        }
        let deepest = positions.last().copied();
        // The end of the prompt (less the token fed for logits): an exact retry or replay of this
        // request resumes there (see `place_checkpoints`). Not for scoring: exact repeats are
        // answered from the engine's results, and a prompt end per question would crowd chat
        // out of the context store.
        let tail = p.ids.len() - 1;
        if self.cfg.host_checkpoints
            && !scoring
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
            // Storage. Only the deepest snapshot can become the lane's head (kept on the device);
            // the others go to the context store, so they are written straight to pinned storage
            // the store takes over (one copy over PCIe instead of a device copy then a download).
            // Then: spare, else the lane's oldest checkpoint (its restore, if any, is queued before
            // the snapshot), else pinned storage the store has ready or new.
            let to_store = Some(pos) != deepest && store.enabled() && self.cfg.host_checkpoints;
            let ready = if to_store {
                store.take_spare_state()
            } else {
                None
            };
            let device = lane.spare.iter().position(|c| !c.on_host());
            let mut cp = match (ready, device) {
                (Some(cp), _) => cp,
                (None, Some(i)) => lane.spare.swap_remove(i),
                (None, None) => match lane.spare.pop() {
                    Some(cp) => cp,
                    None if !lane.checkpoints.is_empty() && self.cfg.host_checkpoints => {
                        lane.checkpoints.sort_by_key(|c| c.pos);
                        lane.checkpoints.remove(0)
                    }
                    None => match store.take_spare_state() {
                        Some(cp) => cp,
                        None => {
                            SNAPS_PINNED.fetch_add(1, Ordering::Relaxed);
                            lane.session.blank_checkpoint_in(&self.b, pos, true)?
                        }
                    },
                },
            };
            if cp.on_host() {
                SNAPS_HOST.fetch_add(1, Ordering::Relaxed);
            } else {
                SNAPS_DEVICE.fetch_add(1, Ordering::Relaxed);
            }
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
    /// Whether the request's answer opens with reasoning (`chat_template_kwargs.enable_thinking`).
    fn thinking(&self, req: &Value) -> bool {
        req.pointer("/chat_template_kwargs/enable_thinking")
            .and_then(Value::as_bool)
            .unwrap_or(self.cfg.enable_thinking)
    }

    fn prepare(
        &self,
        req: &Value,
    ) -> Result<
        (
            Prompt,
            Vec<ImagePatches>,
            Sampler,
            usize,
            Option<Value>,
            bool,
        ),
        ApiError,
    > {
        let t_req = Instant::now();
        let Normalized {
            messages,
            tools,
            images: urls,
        } = normalize_request(req)?;
        let thinking = self.thinking(req);
        let mut text = self
            .fe
            .template
            .render(&messages, tools.as_ref(), thinking, true)
            .map_err(|e| ApiError::invalid(e.to_string()))?;
        // Forced tool call: the prompt ends in the opening tag, so the model continues inside
        // the call (`generate` puts the tag back before parsing the output).
        let forced = self.cfg.force_tool_call
            && tools
                .as_ref()
                .and_then(Value::as_array)
                .is_some_and(|t| !t.is_empty());
        if forced {
            text.push_str(TOOL_CALL_OPEN);
        }
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
        let p = self.expand(&ids, &images)?;
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
        Ok((p, images, sampler, max_tokens, tools, forced))
    }

    /// Rendered ids plus images as model input (image placeholders expanded, MRoPE positions).
    fn expand(&self, ids: &[u32], images: &[ImagePatches]) -> Result<Prompt, ApiError> {
        match self.special.image_pad {
            Some(pad) => build_prompt(ids, pad, images).map_err(ApiError::invalid),
            None if images.is_empty() => {
                build_prompt(ids, u32::MAX, &[]).map_err(ApiError::invalid)
            }
            None => Err(ApiError::invalid("tokenizer has no <|image_pad|> token")),
        }
    }

    /// A scoring query (see `ChatEngine::score`), rendered and tokenized: the prompt renders as a
    /// continuation of its final assistant message with thinking off, as ninfer's System One
    /// requests do. Candidates must be one token each and branches 1 to 5 (ninfer's limits and
    /// error codes). Its images are decoded later, by [`Self::expand_score`].
    fn prepare_score(
        &self,
        req: &Value,
        candidates: &[String],
        branches: &[String],
    ) -> Result<Draft, ApiError> {
        use velox_serve::systemone::{MAX_BRANCH_TOKENS, MAX_RANK_CANDIDATES};
        let invalid = |code: &'static str, message: String| ApiError {
            code,
            ..ApiError::invalid(message)
        };
        if candidates.is_empty() || candidates.len() > velox_core::check::GATHER_MAX {
            return Err(invalid(
                "invalid_request",
                format!("1 to {} score candidates", velox_core::check::GATHER_MAX),
            ));
        }
        if branches.len() > MAX_RANK_CANDIDATES {
            return Err(invalid(
                "too_many_candidates",
                format!("at most {MAX_RANK_CANDIDATES} score branches"),
            ));
        }
        if !branches.is_empty() && self.cfg.score_rows == 0 {
            return Err(invalid(
                "branch_scoring_disabled",
                "branch scoring is disabled on this server (--score-branch-rows 0)".into(),
            ));
        }
        let Normalized {
            messages,
            images: urls,
            ..
        } = normalize_request(req)?;
        let text = self
            .fe
            .template
            .render_continuation(&messages, false)
            .map_err(|e| ApiError::invalid(e.to_string()))?;
        let ids = self.fe.encode(&text).map_err(core)?;
        let candidates = candidates
            .iter()
            .map(|c| {
                let ids = self.fe.encode(c).map_err(core)?;
                match ids[..] {
                    [id] => Ok(id),
                    _ => Err(invalid(
                        "candidate_not_single_token",
                        format!(
                            "score candidate {c:?} encodes to {} tokens; exactly one is required",
                            ids.len()
                        ),
                    )),
                }
            })
            .collect::<Result<Vec<u32>, ApiError>>()?;
        let branches = branches
            .iter()
            .map(|br| {
                let ids = self.fe.encode(br).map_err(core)?;
                if ids.is_empty() || ids.len() > MAX_BRANCH_TOKENS {
                    return Err(invalid(
                        "branch_too_long",
                        format!(
                            "score branch {br:?} encodes to {} tokens; 1 to {MAX_BRANCH_TOKENS} are \
                             allowed",
                            ids.len()
                        ),
                    ));
                }
                Ok(ids)
            })
            .collect::<Result<Vec<Vec<u32>>, ApiError>>()?;
        Ok(Draft {
            ids,
            urls,
            candidates,
            branches,
        })
    }

    /// A drafted scoring query with its images decoded (`decoded` holds those of the request's
    /// other questions, by URL) and expanded into the prompt.
    fn expand_score(
        &self,
        d: Draft,
        decoded: &mut Vec<(String, ImagePatches)>,
    ) -> Result<Question, ApiError> {
        let mut images = Vec::with_capacity(d.urls.len());
        for u in d.urls {
            match decoded.iter().find(|(x, _)| *x == u) {
                Some((_, img)) => images.push(img.clone()),
                None => {
                    let img = self
                        .load_images(std::slice::from_ref(&u))?
                        .pop()
                        .expect("one image per URL");
                    images.push(img.clone());
                    decoded.push((u, img));
                }
            }
        }
        let p = self.expand(&d.ids, &images)?;
        let (candidates, branches) = (d.candidates, d.branches);
        let window = 1 + branches.iter().map(Vec::len).max().unwrap_or(0);
        if p.ids.len() < 2 || p.ids.len() - 1 + window > self.cfg.capacity {
            return Err(ApiError {
                status: 400,
                kind: "invalid_request_error",
                code: "context_length_exceeded",
                message: format!(
                    "{} prompt + {} scored tokens exceed the context of {}",
                    p.ids.len(),
                    window - 1,
                    self.cfg.capacity
                ),
            });
        }
        Ok(Question {
            p,
            images,
            candidates,
            branches,
        })
    }

    /// The jobs that answer `qs` (one request's questions), in the order they run, each with the
    /// question it answers per window (or the one question it answers off its prompt) and the
    /// columns of that question's candidates in the job's. Questions without branches that share
    /// the start of their prompts are one job: the shared prefix, then each question's rest as a
    /// window (when the rests are text and fit a scoring pass). Branch questions run after them:
    /// the prompt but its last token, which starts every branch window. When several jobs follow
    /// one another, the first keeps a checkpoint where their prompts part.
    fn score_plan(&self, qs: &[Question]) -> Vec<(Job, Vec<Answer>)> {
        let mut plan = Vec::new();
        let plain: Vec<usize> = (0..qs.len())
            .filter(|&i| qs[i].branches.is_empty())
            .collect();
        let mut batched = false;
        if plain.len() > 1 {
            let first = &qs[plain[0]].p;
            let fork = plain
                .iter()
                .map(|&i| common_prefix(&first.keys, &qs[i].p.keys).min(qs[i].p.ids.len() - 1))
                .min()
                .unwrap_or(0);
            let longest = plain
                .iter()
                .map(|&i| qs[i].p.ids.len() - fork)
                .max()
                .unwrap_or(0);
            // The rests are text (no image after the fork), so the windows take the rotary
            // positions after the prefix.
            let text = plain
                .iter()
                .all(|&i| qs[i].p.runs.iter().all(|r| r.start + r.len <= fork));
            if fork > 0
                && text
                && plain.len() * longest <= self.cfg.score_rows.max(1) * SCORE_WINDOW
            {
                let mut candidates: Vec<u32> = Vec::new();
                let mut answers = Vec::new();
                for &i in &plain {
                    let cols = qs[i]
                        .candidates
                        .iter()
                        .map(|c| match candidates.iter().position(|x| x == c) {
                            Some(j) => j,
                            None => {
                                candidates.push(*c);
                                candidates.len() - 1
                            }
                        })
                        .collect();
                    answers.push((i, cols));
                }
                let windows = plain
                    .iter()
                    .map(|&i| qs[i].p.ids[fork..].to_vec())
                    .collect();
                let q = &qs[plain[0]];
                let p = Prompt {
                    ids: q.p.ids[..fork].to_vec(),
                    keys: q.p.keys[..fork].to_vec(),
                    positions: q.p.positions.as_ref().map(|v| v[..fork].to_vec()),
                    rope_delta: q.p.rope_delta,
                    runs: q.p.runs.clone(),
                };
                plan.push((score_job(p, q.images.clone(), candidates, windows), answers));
                batched = true;
            }
        }
        for (i, q) in qs.iter().enumerate() {
            let all = (0..q.candidates.len()).collect::<Vec<_>>();
            if q.branches.is_empty() {
                if !batched {
                    // The candidates are read off the prompt's last row.
                    let job =
                        score_job(q.p.clone(), q.images.clone(), q.candidates.clone(), vec![]);
                    plan.push((job, vec![(i, all)]));
                }
                continue;
            }
            // The last token is text (the assistant prefill): fed at cache row n - 1 it takes
            // rotary position n - 1 + rope_delta, as any token after the prompt does, so the
            // windows are positioned exactly as the full prompt would have them.
            let mut p = q.p.clone();
            let tail = p.ids.pop().expect("checked in prepare_score");
            p.keys.pop();
            if let Some(positions) = p.positions.as_mut() {
                positions.pop();
            }
            let windows = q
                .branches
                .iter()
                .map(|br| std::iter::once(tail).chain(br.iter().copied()).collect())
                .collect();
            let job = score_job(p, q.images.clone(), q.candidates.clone(), windows);
            plan.push((job, vec![(i, all)]));
        }
        if let [(first, _), rest @ ..] = &plan[..] {
            let fork = rest
                .iter()
                .map(|(j, _)| common_prefix(&first.p.keys, &j.p.keys))
                .min()
                .filter(|&f| f >= MIN_REUSE);
            if let Some(s) = plan[0].0.score.as_mut() {
                s.fork = fork;
            }
        }
        plan
    }
}

/// The answers of recent scoring queries, so an exact repeat (the same prompt, images, candidates
/// and branches) is answered without the GPU or decoding its images: the scores are
/// deterministic, and ninfer answers repeats from the prompt ends it keeps. Least recently used
/// out first.
#[derive(Default)]
struct ScoreCache {
    entries: Mutex<VecDeque<Cached>>,
}

/// A cached answer: its key (see [`ScoreCache::key`]), the scores and the prompt's length.
struct Cached {
    hash: u64,
    key: Vec<u32>,
    scores: Vec<f32>,
    prompt_tokens: u64,
}

impl ScoreCache {
    const CAPACITY: usize = 256;

    /// Everything a query's scores depend on, as one key (with its hash, compared first): its
    /// rendered prompt, a 64-bit hash and the length of each image URL (the encoded bytes), its
    /// candidates and branches.
    fn key(d: &Draft) -> (u64, Vec<u32>) {
        use std::hash::{Hash, Hasher};
        let mut k = Vec::with_capacity(d.ids.len() + d.candidates.len() + 8);
        k.push(d.ids.len() as u32);
        k.extend_from_slice(&d.ids);
        for u in &d.urls {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            u.hash(&mut h);
            let h = h.finish();
            k.extend([h as u32, (h >> 32) as u32, u.len() as u32]);
        }
        k.push(d.candidates.len() as u32);
        k.extend_from_slice(&d.candidates);
        for br in &d.branches {
            k.push(br.len() as u32);
            k.extend_from_slice(br);
        }
        let mut h = std::collections::hash_map::DefaultHasher::new();
        k.hash(&mut h);
        (h.finish(), k)
    }

    /// The scores and prompt length of a cached answer.
    fn get(&self, key: &(u64, Vec<u32>)) -> Option<(Vec<f32>, u64)> {
        let mut e = self.entries.lock().ok()?;
        let i = e.iter().position(|c| c.hash == key.0 && c.key == key.1)?;
        let hit = e.remove(i)?;
        let out = (hit.scores.clone(), hit.prompt_tokens);
        e.push_back(hit);
        Some(out)
    }

    fn put(&self, key: (u64, Vec<u32>), scores: Vec<f32>, prompt_tokens: u64) {
        let Ok(mut e) = self.entries.lock() else {
            return;
        };
        e.retain(|c| !(c.hash == key.0 && c.key == key.1));
        if e.len() >= Self::CAPACITY {
            e.pop_front();
        }
        e.push_back(Cached {
            hash: key.0,
            key: key.1,
            scores,
            prompt_tokens,
        });
    }
}

/// A question a scoring job answers, and the columns of its candidates among the job's.
type Answer = (usize, Vec<usize>);

/// A scoring query, rendered and tokenized but its images not decoded yet.
struct Draft {
    ids: Vec<u32>,
    urls: Vec<String>,
    candidates: Vec<u32>,
    branches: Vec<Vec<u32>>,
}

/// A prepared scoring query: its whole prompt (the candidates are read after its last token, or
/// after each branch appended to it).
struct Question {
    p: Prompt,
    images: Vec<ImagePatches>,
    candidates: Vec<u32>,
    branches: Vec<Vec<u32>>,
}

/// A scoring job for the queue (its reply channel is set when it is sent).
fn score_job(
    p: Prompt,
    images: Vec<ImagePatches>,
    candidates: Vec<u32>,
    windows: Vec<Vec<u32>>,
) -> Job {
    Job {
        p,
        images,
        sampler: Sampler::greedy(),
        max_tokens: 1,
        stream: false,
        reply: mpsc::channel().0,
        queued: Instant::now(),
        score: Some(Scoring {
            candidates,
            windows,
            fork: None,
        }),
    }
}

impl<B> ChatEngine for QwenEngine<B>
where
    B: Backend + Send + Sync,
    B::Buffer: Send + Sync,
    B::Chunk: Send + Sync,
{
    fn architecture(&self) -> &str {
        "Qwen3_5ForCausalLM"
    }

    fn generate(&self, req: &Value, emit: &mut dyn FnMut(Delta)) -> Result<Outcome, ApiError> {
        let core_ = &self.core;
        let (p, images, sampler, max_tokens, tools, forced) = core_.prepare(req)?;
        let prompt_tokens = p.ids.len() as u64;
        // A forced call has no text before it, reasoning arrives whole, and non-streaming answers
        // are assembled whole anyway.
        let stream = core_.cfg.stream_content
            && !forced
            && !core_.thinking(req)
            && req.get("stream").and_then(Value::as_bool) == Some(true);
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
                stream,
                reply: tx,
                queued: Instant::now(),
                score: None,
            });
        }
        self.queue.cv.notify_one();
        let stopped = || core(CoreError::Backend("engine worker stopped".into()));
        let mut tokens = Vec::new();
        // Content already emitted: always a prefix of what `parse_output` returns at the end.
        let mut streamed = String::new();
        let done = loop {
            match rx.recv().map_err(|_| stopped())? {
                Reply::Tokens(t) => {
                    tokens.extend(t);
                    let text = core_.fe.decode(&tokens).map_err(core)?;
                    if let Some(s) = streamable_content(&text) {
                        if s.len() > streamed.len() && s.starts_with(streamed.as_str()) {
                            emit(Delta::Content(s[streamed.len()..].to_string()));
                            streamed = s.to_string();
                        }
                    }
                }
                Reply::Done(r) => break r?,
            }
        };
        let mut text = core_.fe.decode(&done.out).map_err(core)?;
        if forced {
            text.insert_str(0, TOOL_CALL_OPEN);
        }
        let parsed = parse_output(&text, tools.as_ref());
        match parsed.content.strip_prefix(streamed.as_str()) {
            Some(rest) if !rest.is_empty() => emit(Delta::Content(rest.to_string())),
            Some(_) => {}
            // Cannot happen while `streamable_content` follows `parse_output`; the client then
            // already holds a prefix of the answer, and repeating it would be worse.
            None => eprintln!("stream: emitted content is not a prefix of the parsed answer"),
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
            queue_ms: done.queue_ms,
            prompt_ms: done.prompt_ms,
            predicted_ms: done.predicted_ms,
            first_token: Some(done.first_token),
        })
    }

    fn scores(&self) -> bool {
        true
    }

    fn score(&self, queries: &[ScoreQuery<'_>]) -> Result<Vec<ScoreOutcome>, ApiError> {
        let t0 = Instant::now();
        let drafts = queries
            .iter()
            .map(|q| self.core.prepare_score(q.request, q.candidates, q.branches))
            .collect::<Result<Vec<_>, _>>()?;
        let keys: Vec<_> = drafts.iter().map(ScoreCache::key).collect();
        let mut out: Vec<Option<ScoreOutcome>> = keys
            .iter()
            .map(|k| {
                self.results
                    .get(k)
                    .map(|(logprobs, prompt_tokens)| ScoreOutcome {
                        logprobs,
                        prompt_tokens,
                        cached_tokens: prompt_tokens,
                        queue_ms: 0.0,
                    })
            })
            .collect();
        if out.iter().any(Option::is_none) {
            // The questions of one request share their prefill, so all of them run again.
            let mut decoded = Vec::new();
            let qs = drafts
                .into_iter()
                .map(|d| self.core.expand_score(d, &mut decoded))
                .collect::<Result<Vec<_>, _>>()?;
            if std::env::var_os("VELOX_DEBUG_PREFIX").is_some() {
                eprintln!(
                    "[systemone] {} questions prepared in {:.1} ms ({} images)",
                    qs.len(),
                    t0.elapsed().as_secs_f64() * 1e3,
                    decoded.len()
                );
            }
            for (mut job, answers) in self.core.score_plan(&qs) {
                let width = job.score.as_ref().map_or(0, |s| s.candidates.len());
                let (tx, rx) = mpsc::channel();
                job.reply = tx;
                job.queued = Instant::now();
                self.queue
                    .jobs
                    .lock()
                    .map_err(|_| core(CoreError::Backend("engine queue poisoned".into())))?
                    .push_back(job);
                self.queue.cv.notify_one();
                let done = loop {
                    match rx
                        .recv()
                        .map_err(|_| core(CoreError::Backend("engine worker stopped".into())))?
                    {
                        Reply::Tokens(_) => {}
                        Reply::Done(r) => break r?,
                    }
                };
                let single = answers.len() == 1;
                for (k, (qi, cols)) in answers.into_iter().enumerate() {
                    // One question: all of the job's scores (per branch, if any); several: window
                    // k's row, in the question's candidate order.
                    let logprobs = if single {
                        done.scores.clone()
                    } else {
                        let row = &done.scores[k * width..(k + 1) * width];
                        cols.iter().map(|&c| row[c]).collect()
                    };
                    let prompt_tokens = qs[qi].p.ids.len() as u64;
                    self.results
                        .put(keys[qi].clone(), logprobs.clone(), prompt_tokens);
                    out[qi] = Some(ScoreOutcome {
                        logprobs,
                        prompt_tokens,
                        cached_tokens: (done.start as u64).min(prompt_tokens),
                        queue_ms: done.queue_ms,
                    });
                }
            }
        }
        if std::env::var_os("VELOX_DEBUG_PREFIX").is_some() {
            eprintln!(
                "[systemone] answered in {:.1} ms",
                t0.elapsed().as_secs_f64() * 1e3
            );
        }
        out.into_iter()
            .map(|o| o.ok_or_else(|| core(CoreError::Backend("question left unanswered".into()))))
            .collect()
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

    #[test]
    fn score_cache_answers_exact_repeats_only() {
        let draft = |ids: Vec<u32>, url: &str, branches: Vec<Vec<u32>>| Draft {
            ids,
            urls: vec![url.to_string()],
            candidates: vec![7, 8],
            branches,
        };
        let c = ScoreCache::default();
        let a = draft(vec![1, 2, 3], "data:a", vec![]);
        c.put(ScoreCache::key(&a), vec![-0.5, -1.0], 40);
        assert_eq!(c.get(&ScoreCache::key(&a)), Some((vec![-0.5, -1.0], 40)));
        // Another image, prompt or branch set is another query.
        for other in [
            draft(vec![1, 2, 3], "data:b", vec![]),
            draft(vec![1, 2, 4], "data:a", vec![]),
            draft(vec![1, 2, 3], "data:a", vec![vec![5]]),
        ] {
            assert_eq!(c.get(&ScoreCache::key(&other)), None);
        }
        // Least recently used out first: `a`, just read, outlives the next oldest.
        for i in 0..ScoreCache::CAPACITY as u32 - 1 {
            c.put(
                ScoreCache::key(&draft(vec![i + 100], "", vec![])),
                vec![0.0],
                1,
            );
        }
        c.get(&ScoreCache::key(&a));
        c.put(
            ScoreCache::key(&draft(vec![9999], "", vec![])),
            vec![0.0],
            1,
        );
        assert!(c.get(&ScoreCache::key(&a)).is_some());
        assert_eq!(c.get(&ScoreCache::key(&draft(vec![100], "", vec![]))), None);
    }

    #[test]
    fn streamed_content_is_a_prefix_of_the_parsed_answer() {
        let answer = "\n I'll open the menü < settings.\n\n<tool_call>\n<function=click>\n\
                      <parameter=x>\n4\n</parameter>\n</function>\n</tool_call>";
        let content = parse_output(answer, None).content;
        let mut last = "";
        for end in (0..=answer.len()).filter(|&i| answer.is_char_boundary(i)) {
            let s = streamable_content(&answer[..end]).unwrap();
            assert!(content.starts_with(s), "{s:?} at {end}");
            assert!(s.starts_with(last), "streamed text shrank at {end}");
            last = s;
        }
        assert_eq!(last, content);
        // An incomplete character and a partial tag are held back.
        assert_eq!(streamable_content("ok \u{FFFD}"), Some("ok"));
        assert_eq!(streamable_content("ok <tool_"), Some("ok"));
        assert_eq!(streamable_content("<think>\nplan"), None);
    }
}
