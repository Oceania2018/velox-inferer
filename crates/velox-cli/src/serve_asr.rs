//! `velox serve-asr`: Qwen3-ASR behind llama-server's `/v1/chat/completions` (what the qwen3-asr
//! sidecar sends: a WAV in an `input_audio` part, an optional system context and an optional
//! assistant prefix `language X<asr_text>`). The response content is the prefix plus the
//! transcription, as llama-server returns it.
//!
//! Request threads parse JSON, decode the WAV and compute the log-mel on the host; one engine
//! thread owns the device: it encodes and prefills new requests between steps and advances every
//! active request with one batched decode step (docs/qwen3_asr.md).

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};
use velox_core::{Backend, Error, Result};
use velox_frontend::PlainTokenizer;
use velox_models::qwen3_asr::mel::{MelFrontend, SAMPLE_RATE};
use velox_models::qwen3_asr::{
    BatchWorkspace, KvPool, PoolSeq, Qwen3Asr, AUDIO_PAD, IM_END, MAX_BATCH, PREFILL_ROWS,
};

use crate::args::Args;
use crate::select::Selected;

const MODEL_NAME: &str = "qwen3-asr-1.7b";
const MAX_BODY: usize = 64 << 20;

/// A transcription request, ready for the device.
struct Job {
    prompt: Vec<u32>,
    mel: Vec<f32>,
    frames: usize,
    max_new: usize,
    /// Scheduling class from the request's `priority` (0 = most urgent, the default). The qwen3-asr
    /// sidecar sends 0 for a realtime final, 1 for batch files and 2 for partials (provisional text
    /// the next partial or the final replaces). See `admissible`.
    priority: u8,
    received: Instant,
    reply: Sender<std::result::Result<Done, String>>,
}

struct Done {
    ids: Vec<u32>,
    prompt_tokens: usize,
    truncated: bool,
    /// From the request reaching the engine queue to its last token.
    queued: std::time::Duration,
}

pub fn run(a: &Args) -> Result<ExitCode> {
    let need = |k: &str| {
        a.get(k)
            .map(PathBuf::from)
            .ok_or_else(|| Error::Config(format!("--{k} is required")))
    };
    let (text, mmproj, tok_path) = (need("gguf")?, need("mmproj")?, need("tokenizer")?);
    let host = a.get("host").unwrap_or("0.0.0.0").to_string();
    let port: u16 = a.parse("port", 8080)?;
    let max_batch: usize = a.parse("max-batch", MAX_BATCH)?;
    let default_max_new: usize = a.parse("max-tokens", 512)?;
    // F32 KV costs layers * 2 * kv_heads * head_dim * 4 bytes per row (229 KB for 1.7B).
    let kv_rows: usize = a.parse("kv-rows", 8192)?;
    let tok = Arc::new(PlainTokenizer::from_file(&tok_path)?);
    let warm_prompt = tok.encode(
        "<|im_start|>system
<|im_end|>
<|im_start|>user
<|audio_start|><|audio_pad|><|audio_end|><|im_end|>
<|im_start|>assistant
language English<asr_text>",
    )?;
    let mel = Arc::new(MelFrontend::new());
    let (tx, rx) = mpsc::channel::<Job>();
    let (ready_tx, ready_rx) = mpsc::channel::<Result<()>>();
    std::thread::scope(|sc| -> Result<ExitCode> {
        let engine = sc.spawn(move || {
            let sel = match crate::select::backend(a) {
                Ok(s) => s,
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                    return;
                }
            };
            let r = match sel {
                Selected::Cpu(b) => engine(
                    &b,
                    &text,
                    &mmproj,
                    rx,
                    max_batch,
                    kv_rows,
                    warm_prompt.clone(),
                    &ready_tx,
                ),
                #[cfg(feature = "cuda")]
                Selected::Cuda(b) => engine(
                    &b,
                    &text,
                    &mmproj,
                    rx,
                    max_batch,
                    kv_rows,
                    warm_prompt.clone(),
                    &ready_tx,
                ),
            };
            if let Err(e) = r {
                eprintln!("engine stopped: {e}");
                let _ = ready_tx.send(Err(e));
            }
        });
        ready_rx
            .recv()
            .map_err(|_| Error::Backend("engine exited during load".into()))??;
        let listener = TcpListener::bind((host.as_str(), port))
            .map_err(|e| Error::Config(format!("bind {host}:{port}: {e}")))?;
        eprintln!("serve-asr listening on {host}:{port} (max batch {max_batch})");
        for stream in listener.incoming() {
            if engine.is_finished() {
                break;
            }
            let Ok(stream) = stream else { continue };
            let (tx, tok, mel) = (tx.clone(), tok.clone(), mel.clone());
            std::thread::spawn(move || {
                let _ = connection(stream, &tx, &tok, &mel, default_max_new);
            });
        }
        Ok(ExitCode::FAILURE)
    })
}

/// KV rows reserved for a request's output: ASR output is bounded by the audio (about 3 tokens
/// per second of speech; twice that plus a margin), capped by the request's max_tokens.
fn output_reserve(frames: usize, max_new: usize) -> usize {
    let seconds = frames.div_ceil(100);
    (seconds * 6 + 32).min(max_new)
}

struct Active {
    seq: PoolSeq,
    job: Job,
    ids: Vec<u32>,
    last: u32,
    /// Output tokens this sequence may still produce in its KV range.
    room: usize,
}

#[allow(clippy::too_many_arguments)]
fn engine<B: Backend>(
    b: &B,
    text: &std::path::Path,
    mmproj: &std::path::Path,
    rx: Receiver<Job>,
    max_batch: usize,
    kv_rows: usize,
    warm_prompt: Vec<u32>,
    ready: &Sender<Result<()>>,
) -> Result<()> {
    b.bind_thread()?;
    let t0 = Instant::now();
    let m = Qwen3Asr::load(b, text, mmproj)?;
    let mut bw = BatchWorkspace::new(b, &m)?;
    let mut pool = KvPool::new(b, &m.cfg, kv_rows)?;
    eprintln!(
        "loaded {:.2} GB in {:.1?} on {}; KV pool {kv_rows} rows",
        m.bytes as f64 / 1e9,
        t0.elapsed(),
        b.describe()
    );
    let mut waiting: VecDeque<Job> = VecDeque::new();
    let mut active: Vec<Active> = Vec::new();
    let mut steps = 0u64;
    // Warm-up before serving: kernels load lazily and each (batch size, KV bucket) records its
    // graph on first use, which cost the first requests ~25%. Synthetic 10 s clips with staggered
    // limits end one by one, so decode steps run at every batch size from WARM down to 1.
    let warm_n = max_batch.min(16);
    let (warm_tx, warm_rx) = mpsc::channel();
    {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let pcm: Vec<i16> = (0..10 * SAMPLE_RATE)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                ((seed >> 40) as i32 % 2000) as i16
            })
            .collect();
        let frames = MelFrontend::frames(pcm.len());
        let mel = m.mel().compute(&pcm);
        for i in 0..warm_n {
            waiting.push_back(Job {
                prompt: warm_prompt.clone(),
                mel: mel.clone(),
                frames,
                max_new: i + 2,
                priority: 0,
                received: Instant::now(),
                reply: warm_tx.clone(),
            });
        }
    }
    drop(warm_tx);
    let (mut warm_left, t_warm) = (warm_n, Instant::now());
    let mut serving = false;
    // `VELOX_ASR_PHASES`: wall time spent admitting (encode + prefill) vs decoding, by decode batch
    // size, printed whenever the engine goes idle. Both phases end in a host readback, so the split
    // is exact with graphs on.
    let phases = std::env::var_os("VELOX_ASR_PHASES").is_some();
    let mut ph = Phases::default();
    loop {
        // Take what arrived: block while idle.
        if active.is_empty() && waiting.is_empty() {
            if phases && serving {
                ph.report();
            }
            match rx.recv() {
                Ok(j) => waiting.push_back(j),
                Err(_) => return Ok(()),
            }
        }
        loop {
            match rx.try_recv() {
                Ok(j) => waiting.push_back(j),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) if active.is_empty() && waiting.is_empty() => {
                    return Ok(())
                }
                Err(TryRecvError::Disconnected) => break,
            }
        }
        // Admit by priority, then arrival, while the batch and the pool have room: one encoder pass
        // and one prefill pass for a group (prompts within PREFILL_ROWS, every clip at least a
        // second long, one priority); a prompt longer than that, or a shorter clip, goes alone on
        // the chunked path.
        waiting.make_contiguous().sort_by_key(|j| j.priority);
        let urgent = active.iter().map(|a| a.job.priority).min();
        let mut group: Vec<(PoolSeq, Job)> = Vec::new();
        let mut rows = 0;
        while active.len() + group.len() < max_batch && group.len() < ADMIT_MAX {
            let Some(job) = waiting.front() else { break };
            if !admissible(job.priority, urgent, group.first().map(|g| g.1.priority)) {
                break;
            }
            let prompt_rows = job.prompt.len() - 1 + m.cfg.enc.tokens(job.frames);
            let need = prompt_rows + output_reserve(job.frames, job.max_new) + 2;
            if need > pool.rows() {
                let job = waiting.pop_front().expect("front");
                let _ = job.reply.send(Err(format!(
                    "request needs {need} KV rows, the pool has {}",
                    pool.rows()
                )));
                continue;
            }
            let alone = prompt_rows > PREFILL_ROWS || job.frames < m.cfg.enc.chunk_frames;
            if !group.is_empty() && (alone || rows + prompt_rows > PREFILL_ROWS) {
                break;
            }
            let Some(seq) = pool.alloc(need) else { break };
            let job = waiting.pop_front().expect("front");
            rows += prompt_rows;
            group.push((seq, job));
            if alone {
                break;
            }
        }
        if !group.is_empty() {
            let replies: Vec<_> = group.iter().map(|g| g.1.reply.clone()).collect();
            let (t, n, rows0) = (Instant::now(), group.len(), rows);
            let mut enc_s = 0.0;
            let admitted = admit_group(
                b,
                &m,
                &mut pool,
                &mut bw,
                group,
                phases.then_some(&mut enc_s),
            );
            if serving {
                ph.admit(t.elapsed().as_secs_f64(), n, rows0);
                ph.enc_s += enc_s;
            }
            match admitted {
                Ok(done) => active.extend(done),
                Err(e) => {
                    eprintln!("request failed: {e}");
                    for r in replies {
                        let _ = r.send(Err(e.to_string()));
                    }
                }
            }
        }
        if active.is_empty() {
            while warm_rx.try_recv().is_ok() {
                warm_left -= 1;
            }
            if !serving && warm_left == 0 && waiting.is_empty() {
                serving = true;
                eprintln!("warmed up in {:.1?}", t_warm.elapsed());
                let _ = ready.send(Ok(()));
            }
            continue;
        }
        let ids: Vec<u32> = active.iter().map(|a| a.last).collect();
        let t_step = Instant::now();
        let next = {
            let mut ws = m.decoder_workspace()?;
            let mut seqs: Vec<&mut PoolSeq> = active.iter_mut().map(|a| &mut a.seq).collect();
            m.step_batch(b, &mut pool, &mut seqs, &ids, &mut ws, &mut bw)?
        };
        if serving {
            ph.step(t_step.elapsed().as_secs_f64(), active.len());
        }
        steps += 1;
        let mut i = 0;
        let mut next = next.into_iter();
        while i < active.len() {
            let id = next.next().unwrap_or(IM_END);
            let a = &mut active[i];
            if id == IM_END {
                let a = active.remove(i);
                finish(&mut pool, a, false);
                continue;
            }
            a.ids.push(id);
            a.last = id;
            a.room -= 1;
            if a.ids.len() >= a.job.max_new || a.room == 0 {
                let a = active.remove(i);
                finish(&mut pool, a, true);
                continue;
            }
            i += 1;
        }
        while warm_rx.try_recv().is_ok() {
            warm_left -= 1;
        }
        if !serving && warm_left == 0 {
            serving = true;
            eprintln!("warmed up in {:.1?}", t_warm.elapsed());
            let _ = ready.send(Ok(()));
        }
        if steps % 5000 == 0 {
            eprintln!(
                "{steps} decode steps, {} active, {} waiting, {} KV rows free",
                active.len(),
                waiting.len(),
                pool.free_rows()
            );
        }
    }
}

#[derive(Default)]
struct Phases {
    admit_s: f64,
    /// Of `admit_s`, the encoder (synchronized).
    enc_s: f64,
    admits: usize,
    admitted: usize,
    admit_rows: usize,
    /// Decode time and steps by batch size.
    steps: std::collections::BTreeMap<usize, (f64, usize)>,
}

impl Phases {
    fn admit(&mut self, s: f64, n: usize, rows: usize) {
        self.admit_s += s;
        self.admits += 1;
        self.admitted += n;
        self.admit_rows += rows;
    }
    fn step(&mut self, s: f64, batch: usize) {
        let e = self.steps.entry(batch).or_default();
        e.0 += s;
        e.1 += 1;
    }
    fn report(&mut self) {
        let dec: f64 = self.steps.values().map(|v| v.0).sum();
        let n: usize = self.steps.values().map(|v| v.1).sum();
        if self.admits == 0 && n == 0 {
            return;
        }
        eprintln!(
            "[phases] admit {:.3}s enc {:.3}s ({} groups, {} requests, {} rows, {:.1} us/row)  decode {:.3}s ({n} steps)",
            self.admit_s,
            self.enc_s,
            self.admits,
            self.admitted,
            self.admit_rows,
            1e6 * self.admit_s / self.admit_rows.max(1) as f64,
            dec
        );
        let line: Vec<String> = self
            .steps
            .iter()
            .map(|(b, (s, n))| format!("{b}:{n}x{:.2}ms", 1e3 * s / *n as f64))
            .collect();
        eprintln!("[phases] decode by batch {}", line.join(" "));
        *self = Phases::default();
    }
}

/// Requests admitted per round at most.
const ADMIT_MAX: usize = 8;

/// Whether a waiting request of `priority` may be admitted now. Not while a more urgent request
/// is decoding: admission (encode + prefill) runs between decode steps, so admitting partials
/// stretched every step of a final that was decoding (10 realtime streams on an RTX 5060 Ti:
/// final p99 3-7 s, most of it partial admissions, against 1.8 s for one-at-a-time llama.cpp).
/// A final decodes in ~0.2 s, so the partials wait about that long. A group holds one priority,
/// so a final does not pay for encoding partials alongside it.
fn admissible(priority: u8, decoding: Option<u8>, group: Option<u8>) -> bool {
    decoding.is_none_or(|d| priority <= d) && group.is_none_or(|g| priority == g)
}

/// Encodes and prefills a group of requests (one pass each when several; the chunked path for a
/// single one) and returns those still decoding. On error every sequence goes back to the pool.
fn admit_group<B: Backend>(
    b: &B,
    m: &Qwen3Asr<B>,
    pool: &mut KvPool<B>,
    bw: &mut BatchWorkspace<B>,
    mut group: Vec<(PoolSeq, Job)>,
    enc_s: Option<&mut f64>,
) -> Result<Vec<Active>> {
    let t_enc = Instant::now();
    let r = (|| -> Result<Vec<u32>> {
        let clips: Vec<(&[f32], usize)> =
            group.iter().map(|g| (&g.1.mel[..], g.1.frames)).collect();
        let (audio, counts) = m.encode_many(b, &clips)?;
        if let Some(e) = enc_s {
            b.synchronize()?;
            *e = t_enc.elapsed().as_secs_f64();
        }
        for (g, &n) in group.iter_mut().zip(&counts) {
            g.1.prompt = expand(&g.1.prompt, n);
            g.1.mel = Vec::new();
        }
        let mut ws = m.decoder_workspace()?;
        if let [(seq, job)] = &mut group[..] {
            if job.prompt.len() > PREFILL_ROWS {
                return Ok(vec![m.start_pooled(
                    b,
                    pool,
                    seq,
                    &mut ws,
                    bw,
                    &job.prompt,
                    &audio,
                )?]);
            }
        }
        let mut at = 0;
        let mut reqs: Vec<(&mut PoolSeq, &[u32], usize)> = Vec::with_capacity(group.len());
        for ((seq, job), &n) in group.iter_mut().zip(&counts) {
            reqs.push((seq, &job.prompt[..], at));
            at += n;
        }
        m.prefill_many(b, pool, &mut reqs, &audio, &mut ws, bw)
    })();
    let firsts = match r {
        Ok(f) => f,
        Err(e) => {
            for (seq, _) in group {
                pool.release(seq);
            }
            return Err(e);
        }
    };
    let mut out = Vec::new();
    for ((seq, job), first) in group.into_iter().zip(firsts) {
        let room = seq.capacity() - seq.pos - 1;
        let mut a = Active {
            seq,
            job,
            ids: Vec::new(),
            last: first,
            room,
        };
        if first == IM_END {
            finish(pool, a, false);
            continue;
        }
        a.ids.push(first);
        if a.ids.len() >= a.job.max_new || a.room == 0 {
            finish(pool, a, true);
            continue;
        }
        out.push(a);
    }
    Ok(out)
}

fn finish<B: Backend>(pool: &mut KvPool<B>, a: Active, truncated: bool) {
    let done = Done {
        prompt_tokens: a.job.prompt.len(),
        ids: a.ids,
        truncated,
        queued: a.job.received.elapsed(),
    };
    let _ = a.job.reply.send(Ok(done));
    pool.release(a.seq);
}

/// The prompt with its single `<|audio_pad|>` expanded to `n` (the processor's
/// replace_multimodal_special_tokens).
fn expand(prompt: &[u32], n: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(prompt.len() + n);
    for &id in prompt {
        if id == AUDIO_PAD {
            out.extend(std::iter::repeat_n(AUDIO_PAD, n));
        } else {
            out.push(id);
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// HTTP

struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
    keep_alive: bool,
}

fn read_request(r: &mut BufReader<TcpStream>) -> std::io::Result<Option<Request>> {
    let mut line = String::new();
    if r.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts
        .next()
        .unwrap_or("")
        .split('?')
        .next()
        .unwrap_or("")
        .to_string();
    let version = parts.next().unwrap_or("HTTP/1.1").to_string();
    let (mut length, mut keep_alive) = (0usize, version != "HTTP/1.0");
    loop {
        let mut h = String::new();
        if r.read_line(&mut h)? == 0 {
            return Ok(None);
        }
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            match k.trim().to_ascii_lowercase().as_str() {
                "content-length" => length = v.trim().parse().unwrap_or(0),
                "connection" => keep_alive = !v.trim().eq_ignore_ascii_case("close"),
                _ => {}
            }
        }
    }
    if length > MAX_BODY {
        return Err(std::io::Error::other("body too large"));
    }
    let mut body = vec![0u8; length];
    r.read_exact(&mut body)?;
    Ok(Some(Request {
        method,
        path,
        body,
        keep_alive,
    }))
}

fn respond(s: &mut TcpStream, code: u16, body: &Value, keep_alive: bool) -> std::io::Result<()> {
    let text = body.to_string();
    let reason = match code {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "Internal Server Error",
    };
    write!(
        s,
        "HTTP/1.1 {code} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: {}\r\n\r\n{text}",
        text.len(),
        if keep_alive { "keep-alive" } else { "close" }
    )?;
    s.flush()
}

fn connection(
    stream: TcpStream,
    tx: &Sender<Job>,
    tok: &PlainTokenizer,
    mel: &MelFrontend,
    default_max_new: usize,
) -> std::io::Result<()> {
    let _ = stream.set_nodelay(true);
    let mut out = stream.try_clone()?;
    let mut r = BufReader::new(stream);
    while let Some(req) = read_request(&mut r)? {
        let (code, body) = match (req.method.as_str(), req.path.as_str()) {
            ("GET", "/health") => (200, json!({"status": "ok"})),
            ("GET", "/v1/models") => (
                200,
                json!({"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]}),
            ),
            ("POST", "/v1/chat/completions") => {
                match chat(&req.body, tx, tok, mel, default_max_new) {
                    Ok(v) => (200, v),
                    Err((code, msg)) => (code, json!({"error": {"message": msg, "code": code}})),
                }
            }
            _ => (404, json!({"error": {"message": "not found", "code": 404}})),
        };
        respond(&mut out, code, &body, req.keep_alive)?;
        if !req.keep_alive {
            break;
        }
    }
    Ok(())
}

fn text_of(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect(),
        _ => String::new(),
    }
}

fn chat(
    body: &[u8],
    tx: &Sender<Job>,
    tok: &PlainTokenizer,
    mel: &MelFrontend,
    default_max_new: usize,
) -> std::result::Result<Value, (u16, String)> {
    let bad = |m: String| (400u16, m);
    let req: Value = serde_json::from_slice(body).map_err(|e| bad(format!("invalid JSON: {e}")))?;
    let messages = req["messages"]
        .as_array()
        .ok_or_else(|| bad("messages must be an array".into()))?;
    let (mut system, mut prefix, mut wav) = (String::new(), String::new(), None);
    for (i, msg) in messages.iter().enumerate() {
        match msg["role"].as_str().unwrap_or("") {
            "system" => system.push_str(&text_of(&msg["content"])),
            "user" => {
                for part in msg["content"].as_array().into_iter().flatten() {
                    if let Some(d) = part["input_audio"]["data"].as_str() {
                        wav = Some(d.to_string());
                    }
                }
            }
            "assistant" if i + 1 == messages.len() => prefix = text_of(&msg["content"]),
            _ => {}
        }
    }
    let wav = wav.ok_or_else(|| bad("no input_audio part".into()))?;
    let bytes = base64_decode(&wav).ok_or_else(|| bad("input_audio data is not base64".into()))?;
    let pcm = wav_to_pcm16k(&bytes).map_err(bad)?;
    let frames = MelFrontend::frames(pcm.len());
    if frames == 0 {
        return Err(bad("audio shorter than 10 ms".into()));
    }
    let features = mel.compute(&pcm);
    // The checkpoint's processor template: the system turn is always there, empty included.
    // (llama-server renders the GGUF's template, which drops it when no system message is sent;
    // on 52 call segments that alone moved 24% of words away from the reference.)
    let rendered = format!(
        "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n{prefix}"
    );
    let prompt = tok.encode(&rendered).map_err(|e| bad(e.to_string()))?;
    let max_new = req["max_tokens"]
        .as_u64()
        .or_else(|| req["max_completion_tokens"].as_u64())
        .map_or(default_max_new, |v| v as usize)
        .clamp(1, 4096);
    let priority = req["priority"]
        .as_u64()
        .map_or(0, |p| p.min(u8::MAX as u64) as u8);
    let (reply, wait) = mpsc::channel();
    tx.send(Job {
        prompt,
        mel: features,
        frames,
        max_new,
        priority,
        received: Instant::now(),
        reply,
    })
    .map_err(|_| (500, "engine stopped".to_string()))?;
    let done = wait
        .recv()
        .map_err(|_| (500, "engine dropped the request".to_string()))?
        .map_err(|e| (500, e))?;
    let text = tok.decode(&done.ids).map_err(|e| (500, e.to_string()))?;
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    Ok(json!({
        "id": format!("chatcmpl-velox-asr-{created}"),
        "object": "chat.completion",
        "created": created,
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": format!("{prefix}{text}")},
            "finish_reason": if done.truncated { "length" } else { "stop" },
        }],
        "timings": {"engine_ms": done.queued.as_secs_f64() * 1e3},
        "usage": {
            "prompt_tokens": done.prompt_tokens,
            "completion_tokens": done.ids.len(),
            "total_tokens": done.prompt_tokens + done.ids.len(),
        },
    }))
}

fn base64_decode(s: &str) -> Option<Vec<u8>> {
    let val = |c: u8| -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            _ => return None,
        } as u32)
    };
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let (mut acc, mut bits) = (0u32, 0);
    for &c in s.as_bytes() {
        if c == b'=' {
            break;
        }
        if c.is_ascii_whitespace() {
            continue;
        }
        acc = (acc << 6) | val(c)?;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Some(out)
}

/// 16 kHz mono int16 from a PCM WAV (16-bit, any rate / channel count: mixed down and linearly
/// resampled like the sidecar's clients already do).
fn wav_to_pcm16k(b: &[u8]) -> std::result::Result<Vec<i16>, String> {
    if b.len() < 12 || &b[..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err("input_audio is not a RIFF/WAVE file".into());
    }
    let (mut at, mut fmt, mut data) = (12, None, None);
    while at + 8 <= b.len() {
        let id = &b[at..at + 4];
        let len = u32::from_le_bytes([b[at + 4], b[at + 5], b[at + 6], b[at + 7]]) as usize;
        let body = &b[at + 8..(at + 8 + len).min(b.len())];
        match id {
            b"fmt " if body.len() >= 16 => fmt = Some(body),
            b"data" => data = Some(body),
            _ => {}
        }
        at += 8 + len + (len & 1);
    }
    let fmt = fmt.ok_or("WAV has no fmt chunk")?;
    let data = data.ok_or("WAV has no data chunk")?;
    let tag = u16::from_le_bytes([fmt[0], fmt[1]]);
    let ch = u16::from_le_bytes([fmt[2], fmt[3]]) as usize;
    let rate = u32::from_le_bytes([fmt[4], fmt[5], fmt[6], fmt[7]]) as usize;
    let bits = u16::from_le_bytes([fmt[14], fmt[15]]);
    if !(tag == 1 || tag == 0xfffe) || bits != 16 || ch == 0 || rate == 0 {
        return Err(format!(
            "unsupported WAV: format {tag}, {bits}-bit, {ch} ch, {rate} Hz"
        ));
    }
    let frames = data.len() / (2 * ch);
    let mono: Vec<f32> = (0..frames)
        .map(|i| {
            (0..ch)
                .map(|c| {
                    f32::from(i16::from_le_bytes([
                        data[2 * (i * ch + c)],
                        data[2 * (i * ch + c) + 1],
                    ]))
                })
                .sum::<f32>()
                / ch as f32
        })
        .collect();
    if rate == SAMPLE_RATE {
        return Ok(mono.iter().map(|&v| v.round() as i16).collect());
    }
    let out_len = frames * SAMPLE_RATE / rate;
    Ok((0..out_len)
        .map(|i| {
            let t = i as f64 * rate as f64 / SAMPLE_RATE as f64;
            let (j, f) = (t as usize, (t - t.floor()) as f32);
            let a = mono[j.min(frames - 1)];
            let b2 = mono[(j + 1).min(frames - 1)];
            (a + (b2 - a) * f).round().clamp(-32768.0, 32767.0) as i16
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admission_by_priority() {
        // Idle engine, empty group: anything goes.
        assert!(admissible(2, None, None));
        // A final decoding holds back partials and batch, not other finals.
        assert!(admissible(0, Some(0), None));
        assert!(!admissible(2, Some(0), None));
        assert!(!admissible(1, Some(0), None));
        // Partials decoding never hold back a final.
        assert!(admissible(0, Some(2), None));
        // One priority per group.
        assert!(admissible(0, None, Some(0)));
        assert!(!admissible(2, None, Some(0)));
    }

    #[test]
    fn base64_round_trip() {
        assert_eq!(base64_decode("aGVsbG8gd29ybGQ=").unwrap(), b"hello world");
        assert_eq!(base64_decode("AAEC/w==").unwrap(), vec![0, 1, 2, 255]);
    }

    #[test]
    fn audio_pad_expands_in_place() {
        assert_eq!(
            expand(&[1, AUDIO_PAD, 2], 3),
            vec![1, AUDIO_PAD, AUDIO_PAD, AUDIO_PAD, 2]
        );
    }
}
