//! OpenAI-compatible HTTP API.
//!
//! Endpoints: `GET /health`, `GET /v1/models`, `GET /metrics` (Prometheus text, including the
//! `velox_max_concurrency` gauge llm-inference-daemon probes), `POST /v1/chat/completions`
//! (JSON or SSE stream). Responses carry llama.cpp-style `timings` like ninfer's, so the replay
//! gate and the daemon read both engines the same way.
//!
//! The API layer is engine-agnostic: it validates and frames requests and hands them to a
//! [`ChatEngine`]. Model execution lives behind that trait.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

/// An OpenAI-style error: HTTP status, `type` and `code`.
#[derive(Clone, Debug, PartialEq)]
pub struct ApiError {
    pub status: u16,
    pub kind: &'static str,
    pub code: &'static str,
    pub message: String,
}

impl ApiError {
    pub fn invalid(message: impl Into<String>) -> Self {
        ApiError {
            status: 400,
            kind: "invalid_request_error",
            code: "invalid_request",
            message: message.into(),
        }
    }

    pub fn not_implemented(message: impl Into<String>) -> Self {
        ApiError {
            status: 501,
            kind: "server_error",
            code: "not_implemented",
            message: message.into(),
        }
    }

    fn body(&self) -> Value {
        json!({"error": {"message": self.message, "type": self.kind, "code": self.code, "param": Value::Null}})
    }
}

/// Incremental output of a generation.
#[derive(Clone, Debug, PartialEq)]
pub enum Delta {
    Content(String),
    /// One complete OpenAI `tool_calls[]` entry (`{"index","id","type","function":{...}}`).
    ToolCall(Value),
}

/// Final accounting for a generation.
#[derive(Clone, Debug, PartialEq)]
pub struct Outcome {
    pub finish_reason: String,
    pub prompt_tokens: u64,
    pub cached_tokens: u64,
    pub completion_tokens: u64,
    /// Engine-side wait from hand-off until the request's prefill began.
    pub queue_ms: f64,
    /// Server-side prefill time of the uncached prompt tokens.
    pub prompt_ms: f64,
    /// Server-side time from the first to the last generated token.
    pub predicted_ms: f64,
    /// When the first token was sampled. Engines that emit their output only once it is complete
    /// must set it; without it the first emitted delta stands in.
    pub first_token: Option<Instant>,
}

pub trait ChatEngine: Send + Sync {
    /// The architecture of the loaded model, e.g. "Qwen3_5ForCausalLM".
    fn architecture(&self) -> &str;
    /// Runs one chat request (already validated to have `messages`), emitting deltas as they
    /// are produced.
    fn generate(&self, request: &Value, emit: &mut dyn FnMut(Delta)) -> Result<Outcome, ApiError>;
    /// Housekeeping after a response has been delivered, while no request may be waiting (e.g.
    /// preparing memory the next one would otherwise allocate). Returns promptly once one is.
    fn idle(&self) {}
}

/// Engine for a model whose execution path velox does not implement yet. Loading, health,
/// routing and metrics work; chat requests answer 501 with the reason.
pub struct UnimplementedEngine {
    pub architecture: String,
}

impl ChatEngine for UnimplementedEngine {
    fn architecture(&self) -> &str {
        &self.architecture
    }

    fn generate(&self, _: &Value, _: &mut dyn FnMut(Delta)) -> Result<Outcome, ApiError> {
        Err(ApiError::not_implemented(format!(
            "velox-inferer has no execution path for {} yet (docs/architecture.md, next steps)",
            self.architecture
        )))
    }
}

pub struct Config {
    pub model_id: String,
    pub max_concurrency: u32,
    /// Extra `velox_info` labels, e.g. variant and profile.
    pub info: Vec<(String, String)>,
}

/// Counting semaphore bounding concurrent generations to the engine's lanes.
struct Lanes {
    free: Mutex<u32>,
    cv: Condvar,
}

impl Lanes {
    fn acquire(&self) -> LaneGuard<'_> {
        let mut free = self.free.lock().unwrap();
        while *free == 0 {
            free = self.cv.wait(free).unwrap();
        }
        *free -= 1;
        LaneGuard(self)
    }
}

/// Histogram bucket bounds (seconds), ninfer's, so one query reads both engines.
const BUCKETS: [f64; 20] = [
    0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0,
    160.0, 320.0, 640.0, 1280.0,
];

#[derive(Default)]
struct Histogram {
    /// Per-bucket counts (inclusive upper bounds), the last for `+Inf`.
    bins: [u64; BUCKETS.len() + 1],
    sum: f64,
}

impl Histogram {
    fn observe(&mut self, seconds: f64) {
        if !seconds.is_finite() || seconds < 0.0 {
            return;
        }
        self.bins[BUCKETS.partition_point(|&b| b < seconds)] += 1;
        self.sum += seconds;
    }

    fn render(&self, out: &mut String, name: &str, help: &str) {
        use std::fmt::Write as _;
        let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} histogram");
        let mut count = 0;
        for (i, n) in self.bins.iter().enumerate() {
            count += n;
            let le = BUCKETS.get(i).map_or("+Inf".to_string(), f64::to_string);
            let _ = writeln!(out, "{name}_bucket{{le=\"{le}\"}} {count}");
        }
        let _ = writeln!(out, "{name}_sum {}\n{name}_count {count}", self.sum);
    }
}

/// Latency of settled generations (completed, or abandoned by the client); engine failures are
/// counted in `velox_request_errors_total` only.
#[derive(Default)]
struct Latency {
    ttft: Histogram,
    queue: Histogram,
    duration: Histogram,
}

struct LaneGuard<'a>(&'a Lanes);

impl Drop for LaneGuard<'_> {
    fn drop(&mut self) {
        *self.0.free.lock().unwrap() += 1;
        self.0.cv.notify_one();
    }
}

pub struct Server {
    cfg: Config,
    engine: Box<dyn ChatEngine>,
    lanes: Lanes,
    requests: AtomicU64,
    errors: AtomicU64,
    prompt_tokens: AtomicU64,
    cached_tokens: AtomicU64,
    latency: Mutex<Latency>,
    started: u64,
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
}

const MAX_BODY: usize = 256 << 20;

fn read_request(r: &mut BufReader<TcpStream>) -> std::io::Result<Option<Request>> {
    let mut line = String::new();
    if r.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    let mut parts = line.split_whitespace();
    let (method, target) = (
        parts.next().unwrap_or("").to_string(),
        parts.next().unwrap_or("").to_string(),
    );
    let path = target.split('?').next().unwrap_or("").to_string();
    let (mut length, mut chunked) = (0usize, false);
    loop {
        let mut h = String::new();
        r.read_line(&mut h)?;
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            match k.trim().to_ascii_lowercase().as_str() {
                "content-length" => length = v.trim().parse().unwrap_or(0),
                "transfer-encoding" => chunked = v.to_ascii_lowercase().contains("chunked"),
                _ => {}
            }
        }
    }
    let mut body = Vec::new();
    if chunked {
        loop {
            let mut size = String::new();
            r.read_line(&mut size)?;
            let n = usize::from_str_radix(size.trim().split(';').next().unwrap_or("0"), 16)
                .unwrap_or(0);
            if n == 0 || body.len() + n > MAX_BODY {
                break;
            }
            let start = body.len();
            body.resize(start + n + 2, 0);
            r.read_exact(&mut body[start..])?;
            body.truncate(start + n);
        }
    } else if length > 0 {
        if length > MAX_BODY {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "request body too large",
            ));
        }
        body.resize(length, 0);
        r.read_exact(&mut body)?;
    }
    Ok(Some(Request { method, path, body }))
}

fn respond(w: &mut TcpStream, status: u16, content_type: &str, body: &[u8]) -> std::io::Result<()> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        501 => "Not Implemented",
        _ => "Error",
    };
    write!(w, "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len())?;
    w.write_all(body)?;
    w.flush()
}

fn respond_json(w: &mut TcpStream, status: u16, v: &Value) -> std::io::Result<()> {
    respond(w, status, "application/json", v.to_string().as_bytes())
}

/// Writes one HTTP/1.1 chunk holding an SSE `data:` event.
fn sse(w: &mut TcpStream, v: &Value) -> std::io::Result<()> {
    let payload = format!("data: {v}\n\n");
    write!(w, "{:x}\r\n{payload}\r\n", payload.len())?;
    w.flush()
}

impl Server {
    pub fn new(cfg: Config, engine: Box<dyn ChatEngine>) -> Arc<Self> {
        let lanes = Lanes {
            free: Mutex::new(cfg.max_concurrency.max(1)),
            cv: Condvar::new(),
        };
        Arc::new(Server {
            cfg,
            engine,
            lanes,
            requests: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            prompt_tokens: AtomicU64::new(0),
            cached_tokens: AtomicU64::new(0),
            latency: Mutex::default(),
            started: unix_now(),
        })
    }

    /// Serves until the process exits; one thread per connection.
    pub fn serve(self: &Arc<Self>, listener: TcpListener) -> std::io::Result<()> {
        for stream in listener.incoming() {
            let stream = stream?;
            let server = Arc::clone(self);
            std::thread::spawn(move || {
                let _ = server.handle(stream);
            });
        }
        Ok(())
    }

    fn handle(&self, stream: TcpStream) -> std::io::Result<()> {
        stream.set_read_timeout(Some(Duration::from_secs(300)))?;
        let mut w = stream.try_clone()?;
        let mut r = BufReader::new(stream);
        let Some(req) = read_request(&mut r)? else {
            return Ok(());
        };
        match (req.method.as_str(), req.path.as_str()) {
            ("GET", "/health") => respond_json(&mut w, 200, &json!({"status": "ok"})),
            ("GET", "/v1/models") => respond_json(
                &mut w,
                200,
                &json!({"object": "list", "data": [{"id": self.cfg.model_id, "object": "model", "created": self.started, "owned_by": "velox-inferer"}]}),
            ),
            ("GET", "/metrics") => respond(
                &mut w,
                200,
                "text/plain; version=0.0.4",
                self.metrics().as_bytes(),
            ),
            ("POST", "/v1/chat/completions") => {
                let r = self.chat(&mut w, &req.body);
                // The response is complete: let the client go before the engine's idle work.
                let _ = w.flush();
                let _ = w.shutdown(std::net::Shutdown::Write);
                self.engine.idle();
                r
            }
            (_, "/health" | "/v1/models" | "/metrics" | "/v1/chat/completions") => {
                respond_json(&mut w, 405, &ApiError::invalid("method not allowed").body())
            }
            _ => respond_json(
                &mut w,
                404,
                &json!({"error": {"message": format!("no route {}", req.path), "type": "invalid_request_error", "code": "not_found"}}),
            ),
        }
    }

    fn metrics(&self) -> String {
        let labels: Vec<String> = std::iter::once(("model".to_string(), self.cfg.model_id.clone()))
            .chain(std::iter::once((
                "architecture".to_string(),
                self.engine.architecture().to_string(),
            )))
            .chain(self.cfg.info.iter().cloned())
            .map(|(k, v)| format!("{k}=\"{}\"", v.replace('\\', "\\\\").replace('"', "\\\"")))
            .collect();
        let mut out = format!(
            "# HELP velox_max_concurrency Concurrent generation lanes fixed at startup\n\
             # TYPE velox_max_concurrency gauge\n\
             velox_max_concurrency {}\n\
             # TYPE velox_requests_total counter\n\
             velox_requests_total {}\n\
             # TYPE velox_request_errors_total counter\n\
             velox_request_errors_total {}\n\
             # HELP velox_prompt_tokens_total Prompt tokens of completed generations\n\
             # TYPE velox_prompt_tokens_total counter\n\
             velox_prompt_tokens_total {}\n\
             # HELP velox_prompt_tokens_cached_total Prompt tokens served from the prefix cache\n\
             # TYPE velox_prompt_tokens_cached_total counter\n\
             velox_prompt_tokens_cached_total {}\n\
             # TYPE velox_info gauge\n\
             velox_info{{{}}} 1\n",
            self.cfg.max_concurrency.max(1),
            self.requests.load(Ordering::Relaxed),
            self.errors.load(Ordering::Relaxed),
            self.prompt_tokens.load(Ordering::Relaxed),
            self.cached_tokens.load(Ordering::Relaxed),
            labels.join(",")
        );
        let l = self.latency.lock().unwrap();
        l.ttft.render(
            &mut out,
            "velox_time_to_first_token_seconds",
            "Request receipt through the first sampled token, including preparation and queueing",
        );
        l.queue.render(
            &mut out,
            "velox_request_queue_seconds",
            "Wait for a lane and for the engine to admit the request",
        );
        l.duration.render(
            &mut out,
            "velox_request_duration_seconds",
            "Request receipt through generation settlement, including client cancellation",
        );
        out
    }

    fn validate(&self, body: &[u8]) -> Result<Value, ApiError> {
        let req: Value = serde_json::from_slice(body)
            .map_err(|e| ApiError::invalid(format!("body is not JSON: {e}")))?;
        if !req
            .get("messages")
            .is_some_and(|m| m.as_array().is_some_and(|a| !a.is_empty()))
        {
            return Err(ApiError::invalid("messages must be a non-empty array"));
        }
        if let Some(model) = req.get("model").and_then(Value::as_str) {
            if model != self.cfg.model_id {
                return Err(ApiError {
                    status: 404,
                    kind: "invalid_request_error",
                    code: "model_not_found",
                    message: format!("model '{model}' is not served here"),
                });
            }
        }
        Ok(req)
    }

    /// Counts a rejected or failed generation and logs it: the reply goes only to the client.
    /// Messages carry counts and limits, never prompt text.
    fn count_error(&self, e: &ApiError, t_req: Instant) {
        self.errors.fetch_add(1, Ordering::Relaxed);
        eprintln!(
            "[error] {} {} after {:.1} ms: {}",
            e.status,
            e.code,
            t_req.elapsed().as_secs_f64() * 1e3,
            e.message
        );
    }

    fn chat(&self, w: &mut TcpStream, body: &[u8]) -> std::io::Result<()> {
        let t_req = Instant::now();
        self.requests.fetch_add(1, Ordering::Relaxed);
        let req = match self.validate(body) {
            Ok(r) => r,
            Err(e) => {
                self.count_error(&e, t_req);
                return respond_json(w, e.status, &e.body());
            }
        };
        let stream = req.get("stream").and_then(Value::as_bool).unwrap_or(false);
        let include_usage = req
            .pointer("/stream_options/include_usage")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let id = format!("chatcmpl-velox-{}", self.requests.load(Ordering::Relaxed));
        let created = unix_now();
        let model = self.cfg.model_id.clone();
        let chunk = |delta: Value, finish: Value| json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]});

        let t_lane = Instant::now();
        let _lane = self.lanes.acquire();
        let t0 = Instant::now();
        let mut content = String::new();
        let mut tool_calls = Vec::new();
        let mut headers_sent = false;
        let mut io_err = None;
        let mut first_delta = None;
        let result = self.engine.generate(&req, &mut |d| {
            first_delta.get_or_insert_with(Instant::now);
            if !stream {
                match d {
                    Delta::Content(t) => content.push_str(&t),
                    Delta::ToolCall(c) => tool_calls.push(c),
                }
                return;
            }
            if io_err.is_some() {
                return;
            }
            let res = (|| {
                if !headers_sent {
                    w.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n")?;
                    sse(w, &chunk(json!({"role": "assistant", "content": ""}), Value::Null))?;
                    headers_sent = true;
                }
                match d {
                    Delta::Content(t) => sse(w, &chunk(json!({"content": t}), Value::Null)),
                    Delta::ToolCall(c) => sse(w, &chunk(json!({"tool_calls": [c]}), Value::Null)),
                }
            })();
            if let Err(e) = res {
                io_err = Some(e);
            }
        });
        // Counted before the stream-error check: a client that hung up still cost the prefill.
        if let Ok(o) = &result {
            self.prompt_tokens
                .fetch_add(o.prompt_tokens, Ordering::Relaxed);
            self.cached_tokens
                .fetch_add(o.cached_tokens, Ordering::Relaxed);
            let mut l = self.latency.lock().unwrap();
            if let Some(t) = o.first_token.or(first_delta) {
                l.ttft.observe(t.duration_since(t_req).as_secs_f64());
            }
            l.queue
                .observe((t0 - t_lane).as_secs_f64() + o.queue_ms / 1e3);
            l.duration.observe(t_req.elapsed().as_secs_f64());
        }
        if let Some(e) = io_err {
            return Err(e);
        }
        let outcome = match result {
            Ok(o) => o,
            Err(e) => {
                self.count_error(&e, t_req);
                if headers_sent {
                    sse(w, &e.body())?;
                    return w.write_all(b"0\r\n\r\n");
                }
                return respond_json(w, e.status, &e.body());
            }
        };
        let usage = json!({
            "prompt_tokens": outcome.prompt_tokens,
            "completion_tokens": outcome.completion_tokens,
            "total_tokens": outcome.prompt_tokens + outcome.completion_tokens,
            "prompt_tokens_details": {"cached_tokens": outcome.cached_tokens},
        });
        let prompt_n = outcome.prompt_tokens.saturating_sub(outcome.cached_tokens);
        let per_s = |n: u64, ms: f64| if ms > 0.0 { n as f64 * 1e3 / ms } else { 0.0 };
        let timings = json!({
            "cache_n": outcome.cached_tokens,
            "prompt_n": prompt_n,
            "prompt_ms": outcome.prompt_ms,
            "prompt_per_second": per_s(prompt_n, outcome.prompt_ms),
            "predicted_n": outcome.completion_tokens,
            "predicted_ms": outcome.predicted_ms,
            "predicted_per_second": per_s(outcome.completion_tokens.saturating_sub(1), outcome.predicted_ms),
            "total_ms": t0.elapsed().as_secs_f64() * 1e3,
        });
        if stream {
            if !headers_sent {
                w.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n")?;
            }
            let mut last = chunk(json!({}), outcome.finish_reason.clone().into());
            last["timings"] = timings;
            sse(w, &last)?;
            if include_usage {
                let mut u = json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": []});
                u["usage"] = usage;
                sse(w, &u)?;
            }
            let done = "data: [DONE]\n\n";
            write!(w, "{:x}\r\n{done}\r\n0\r\n\r\n", done.len())?;
            return w.flush();
        }
        let mut message = json!({"role": "assistant", "content": if content.is_empty() && !tool_calls.is_empty() { Value::Null } else { content.into() }});
        if !tool_calls.is_empty() {
            message["tool_calls"] = Value::Array(tool_calls);
        }
        respond_json(
            w,
            200,
            &json!({
                "id": id, "object": "chat.completion", "created": created, "model": model,
                "choices": [{"index": 0, "message": message, "finish_reason": outcome.finish_reason}],
                "usage": usage, "timings": timings,
            }),
        )
    }
}

#[cfg(test)]
mod tests;
