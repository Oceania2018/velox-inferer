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
    BatchWorkspace, KvPool, PoolSeq, Qwen3Asr, AUDIO_PAD, IM_END, MAX_BATCH,
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
                Selected::Cpu(b) => engine(&b, &text, &mmproj, rx, max_batch, kv_rows, &ready_tx),
                #[cfg(feature = "cuda")]
                Selected::Cuda(b) => engine(&b, &text, &mmproj, rx, max_batch, kv_rows, &ready_tx),
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
    let _ = ready.send(Ok(()));
    let mut waiting: VecDeque<Job> = VecDeque::new();
    let mut active: Vec<Active> = Vec::new();
    let mut steps = 0u64;
    loop {
        // Take what arrived: block while idle.
        if active.is_empty() && waiting.is_empty() {
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
        // Admit in arrival order while the batch and the pool have room (a few per step, so
        // running requests keep advancing).
        let mut admitted = 0;
        while active.len() < max_batch && admitted < 4 {
            let Some(job) = waiting.front() else { break };
            let need =
                job.prompt.len() + job.frames / 8 + output_reserve(job.frames, job.max_new) + 2;
            if need > pool.rows() {
                let job = waiting.pop_front().expect("front");
                let _ = job.reply.send(Err(format!(
                    "request needs {need} KV rows, the pool has {}",
                    pool.rows()
                )));
                continue;
            }
            let Some(seq) = pool.alloc(need) else { break };
            let job = waiting.pop_front().expect("front");
            admitted += 1;
            let reply = job.reply.clone();
            match admit(b, &m, &mut pool, &mut bw, seq, job) {
                Ok(Some(a)) => active.push(a),
                Ok(None) => {}
                Err(e) => {
                    eprintln!("request failed: {e}");
                    let _ = reply.send(Err(e.to_string()));
                }
            }
        }
        if active.is_empty() {
            continue;
        }
        let ids: Vec<u32> = active.iter().map(|a| a.last).collect();
        let next = {
            let mut ws = m.decoder_workspace()?;
            let mut seqs: Vec<&mut PoolSeq> = active.iter_mut().map(|a| &mut a.seq).collect();
            m.step_batch(b, &mut pool, &mut seqs, &ids, &mut ws, &mut bw)?
        };
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

/// Encodes and prefills a request into `seq`; `None` when it is already complete (stopped at its
/// first token or at its limit).
fn admit<B: Backend>(
    b: &B,
    m: &Qwen3Asr<B>,
    pool: &mut KvPool<B>,
    bw: &mut BatchWorkspace<B>,
    mut seq: PoolSeq,
    mut job: Job,
) -> Result<Option<Active>> {
    let r = (|| {
        let (audio, n) = m.encode_mel(b, std::mem::take(&mut job.mel), job.frames)?;
        job.prompt = expand(&job.prompt, n);
        let mut ws = m.decoder_workspace()?;
        m.start_pooled(b, pool, &mut seq, &mut ws, bw, &job.prompt, &audio)
    })();
    let first = match r {
        Ok(f) => f,
        Err(e) => {
            pool.release(seq);
            return Err(e);
        }
    };
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
        return Ok(None);
    }
    a.ids.push(first);
    if a.ids.len() >= a.job.max_new || a.room == 0 {
        finish(pool, a, true);
        return Ok(None);
    }
    Ok(Some(a))
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
    // The checkpoint's processor template always opens with a system turn; llama-server renders
    // the GGUF's template, which writes one only when a system message is sent.
    // VELOX_ASR_SYSTEM=auto reproduces llama-server.
    let omit = system.is_empty() && std::env::var("VELOX_ASR_SYSTEM").as_deref() == Ok("auto");
    let system_turn = if omit {
        String::new()
    } else {
        format!("<|im_start|>system\n{system}<|im_end|>\n")
    };
    let rendered = format!(
        "{system_turn}<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n{prefix}"
    );
    let prompt = tok.encode(&rendered).map_err(|e| bad(e.to_string()))?;
    let max_new = req["max_tokens"]
        .as_u64()
        .or_else(|| req["max_completion_tokens"].as_u64())
        .map_or(default_max_new, |v| v as usize)
        .clamp(1, 4096);
    let (reply, wait) = mpsc::channel();
    tx.send(Job {
        prompt,
        mel: features,
        frames,
        max_new,
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
