//! Replay gate: re-issues a recorded agent session's requests, in order, to an OpenAI-compatible
//! server and measures what users feel (time to first token, end-to-end latency, decode speed).
//! The same session replayed against the ninfer baseline and against velox gives two reports;
//! `replay-compare` fails when velox regresses.
//!
//! Sessions hold confidential business data. Replay only talks to the endpoint given (default
//! localhost), and reports keep metrics and an output hash, never prompt or output text.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use velox_core::{Error, Result};

use crate::args::Args;

const IMAGE_PLACEHOLDER: &str = "[image data url";

fn cfg(msg: impl Into<String>) -> Error {
    Error::Config(msg.into())
}

struct Step {
    index: usize,
    request: Value,
    screenshot: Option<PathBuf>,
    recorded: Value,
}

/// The full request of a step record: an inline `requestPrompt` string, or the session's
/// `requestEnvelope` (constant fields plus the system message) with the step's `requestMessages`.
fn step_request(m: &serde_json::Map<String, Value>, env: Option<&Value>) -> Result<Option<Value>> {
    if let Some(Value::String(prompt)) = m.get("requestPrompt") {
        return serde_json::from_str(prompt)
            .map(Some)
            .map_err(|e| cfg(format!("step requestPrompt is not JSON: {e}")));
    }
    let (Some(Value::Array(dynamic)), Some(Value::Object(env))) = (m.get("requestMessages"), env)
    else {
        return Ok(None);
    };
    let mut req = env.clone();
    let mut messages: Vec<Value> = req.remove("system").into_iter().collect();
    messages.extend(dynamic.iter().cloned());
    req.insert("messages".into(), Value::Array(messages));
    Ok(Some(Value::Object(req)))
}

/// Finds every step record (an object with a request, see `step_request`) in document order.
fn collect_steps(v: &Value, dir: &Path, env: Option<&Value>, out: &mut Vec<Step>) -> Result<()> {
    match v {
        Value::Object(m) => {
            let env = m.get("requestEnvelope").or(env);
            if let Some(request) = step_request(m, env)? {
                let file = match m.get("screenshot") {
                    Some(Value::String(s)) => Some(s.clone()),
                    Some(Value::Object(o)) => {
                        o.get("file").and_then(Value::as_str).map(String::from)
                    }
                    _ => None,
                };
                let screenshot = file.map(|f| {
                    let p = Path::new(&f);
                    if p.is_absolute() {
                        p.to_path_buf()
                    } else if dir.join(p).exists() {
                        dir.join(p)
                    } else {
                        dir.join("screenshots")
                            .join(p.file_name().unwrap_or_default())
                    }
                });
                let mut recorded = json!({});
                for k in [
                    "promptTokens",
                    "cachedTokens",
                    "prefilledTokens",
                    "prefillMs",
                    "latencyMs",
                    "completionTokens",
                    "decodeTokensPerSec",
                ] {
                    if let Some(x) = m.get(k).or_else(|| m.get("usage").and_then(|u| u.get(k))) {
                        recorded[k] = x.clone();
                    }
                }
                out.push(Step {
                    index: out.len() + 1,
                    request,
                    screenshot,
                    recorded,
                });
                return Ok(());
            }
            for x in m.values() {
                collect_steps(x, dir, env, out)?;
            }
        }
        Value::Array(a) => {
            for x in a {
                collect_steps(x, dir, env, out)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(data.len().div_ceil(3) * 4);
    for c in data.chunks(3) {
        let n = (u32::from(c[0]) << 16)
            | (u32::from(*c.get(1).unwrap_or(&0)) << 8)
            | u32::from(*c.get(2).unwrap_or(&0));
        for (i, shift) in [18, 12, 6, 0].into_iter().enumerate() {
            s.push(if i <= c.len() {
                T[(n >> shift) as usize & 63] as char
            } else {
                '='
            });
        }
    }
    s
}

/// Replaces recorded image placeholders with the step's screenshot. Returns how many it filled.
fn fill_images(v: &mut Value, data_url: &Option<String>) -> Result<usize> {
    match v {
        Value::Object(m) => {
            if let Some(Value::String(u)) = m.get("url") {
                if u.starts_with(IMAGE_PLACEHOLDER) {
                    let url = data_url.clone().ok_or_else(|| {
                        cfg("request has an image placeholder but the step has no screenshot")
                    })?;
                    m.insert("url".into(), Value::String(url));
                    return Ok(1);
                }
            }
            m.values_mut().map(|x| fill_images(x, data_url)).sum()
        }
        Value::Array(a) => a.iter_mut().map(|x| fill_images(x, data_url)).sum(),
        _ => Ok(0),
    }
}

/// FNV-1a; identifies outputs for determinism checks without storing them.
fn fnv(data: &[u8]) -> String {
    let h = data.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
    });
    format!("{h:016x}")
}

struct Endpoint {
    host: String,
    port: u16,
}

impl Endpoint {
    fn parse(url: &str) -> Result<Self> {
        let rest = url
            .strip_prefix("http://")
            .ok_or_else(|| cfg("endpoint must be http://host:port"))?;
        let hostport = rest.split('/').next().unwrap_or("");
        let (host, port) = hostport
            .rsplit_once(':')
            .ok_or_else(|| cfg("endpoint needs a port"))?;
        Ok(Endpoint {
            host: host.into(),
            port: port.parse().map_err(|_| cfg("bad endpoint port"))?,
        })
    }
}

struct Measured {
    ttft: Option<f64>,
    total: f64,
    usage: Value,
    /// llama.cpp-style `timings` (prompt_ms, predicted_per_second, ...) when the server sends them.
    timings: Value,
    output_hash: String,
    finish: Option<String>,
}

/// Streams one chat completion and timestamps the first generated delta.
fn stream_chat(ep: &Endpoint, body: &[u8]) -> Result<Measured> {
    let io = |e: std::io::Error| Error::Backend(format!("http: {e}"));
    let mut s = TcpStream::connect((ep.host.as_str(), ep.port)).map_err(io)?;
    s.set_read_timeout(Some(Duration::from_secs(600)))
        .map_err(io)?;
    let head = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: {}:{}\r\nContent-Type: application/json\r\nAccept: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        ep.host,
        ep.port,
        body.len()
    );
    let t0 = Instant::now();
    s.write_all(head.as_bytes()).map_err(io)?;
    s.write_all(body).map_err(io)?;
    let mut r = BufReader::new(s);

    let mut status = String::new();
    r.read_line(&mut status).map_err(io)?;
    let mut chunked = false;
    loop {
        let mut h = String::new();
        r.read_line(&mut h).map_err(io)?;
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        if h.to_ascii_lowercase().starts_with("transfer-encoding:")
            && h.to_ascii_lowercase().contains("chunked")
        {
            chunked = true;
        }
    }
    if !status.contains(" 200") {
        let mut rest = String::new();
        let _ = r.read_to_string(&mut rest);
        return Err(Error::Backend(format!(
            "server answered {}: {}",
            status.trim(),
            rest.chars().take(300).collect::<String>()
        )));
    }

    let mut ttft = None;
    let mut usage = Value::Null;
    let mut timings = Value::Null;
    let mut finish = None;
    let mut output = Vec::new();
    let mut pending = String::new();
    let mut on_data = |text: &str, now: f64| {
        pending.push_str(text);
        while let Some(pos) = pending.find('\n') {
            let line: String = pending.drain(..=pos).collect();
            let Some(data) = line.trim().strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            if data == "[DONE]" {
                continue;
            }
            let Ok(ev) = serde_json::from_str::<Value>(data) else {
                continue;
            };
            if let Some(u) = ev.get("usage").filter(|u| !u.is_null()) {
                usage = u.clone();
            }
            if let Some(t) = ev.get("timings").filter(|t| t.is_object()) {
                timings = t.clone();
            }
            for c in ev
                .get("choices")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let d = &c["delta"];
                let mut produced = false;
                for k in ["content", "reasoning_content", "reasoning"] {
                    if let Some(t) = d.get(k).and_then(Value::as_str).filter(|t| !t.is_empty()) {
                        output.extend_from_slice(t.as_bytes());
                        produced = true;
                    }
                }
                if let Some(tc) = d.get("tool_calls").filter(|t| !t.is_null()) {
                    output.extend_from_slice(tc.to_string().as_bytes());
                    produced = true;
                }
                if produced && ttft.is_none() {
                    ttft = Some(now);
                }
                if let Some(f) = c.get("finish_reason").and_then(Value::as_str) {
                    finish = Some(f.to_string());
                }
            }
        }
    };

    let ms = |t: Instant| t.elapsed().as_secs_f64() * 1e3;
    if chunked {
        loop {
            let mut size = String::new();
            if r.read_line(&mut size).map_err(io)? == 0 {
                break;
            }
            let n = usize::from_str_radix(size.trim().split(';').next().unwrap_or(""), 16)
                .map_err(|_| Error::Backend("bad chunk size".into()))?;
            if n == 0 {
                break;
            }
            let mut buf = vec![0u8; n + 2];
            r.read_exact(&mut buf).map_err(io)?;
            on_data(&String::from_utf8_lossy(&buf[..n]), ms(t0));
        }
    } else {
        let mut line = String::new();
        while r.read_line(&mut line).map_err(io)? > 0 {
            on_data(&line, ms(t0));
            line.clear();
        }
    }
    let total = ms(t0);
    // Local inspection of what the model produced (confidential: never leaves the machine).
    if std::env::var_os("VELOX_REPLAY_DUMP").is_some() {
        eprintln!("[output] {}", String::from_utf8_lossy(&output));
    }
    Ok(Measured {
        ttft,
        total,
        usage,
        timings,
        output_hash: fnv(&output),
        finish,
    })
}

fn stats(v: &mut [f64]) -> Value {
    if v.is_empty() {
        return Value::Null;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let pct = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
    json!({"n": v.len(), "mean": v.iter().sum::<f64>() / v.len() as f64, "p50": pct(0.5), "p90": pct(0.9), "min": v[0], "max": v[v.len() - 1]})
}

/// The session's requests with screenshots restored, plus each step's recorded metrics
/// (`--session`, optional `--steps`). Used by `prompt-debug`.
pub fn load_requests(a: &Args) -> Result<Vec<(Value, Value)>> {
    let dir = PathBuf::from(
        a.get("session")
            .ok_or_else(|| cfg("--session <dir> is required"))?,
    );
    let text = std::fs::read_to_string(dir.join("session.json"))
        .map_err(|e| cfg(format!("session.json: {e}")))?;
    let session: Value =
        serde_json::from_str(&text).map_err(|e| cfg(format!("session.json: {e}")))?;
    let mut steps = Vec::new();
    collect_steps(&session, &dir, None, &mut steps)?;
    steps.truncate(a.parse("steps", usize::MAX)?);
    steps
        .into_iter()
        .map(|st| {
            let mut req = st.request;
            let data_url = match &st.screenshot {
                Some(p) if p.exists() => Some(format!(
                    "data:image/jpeg;base64,{}",
                    base64(&std::fs::read(p).map_err(|e| cfg(format!("{}: {e}", p.display())))?)
                )),
                _ => None,
            };
            fill_images(&mut req, &data_url)?;
            Ok((req, st.recorded))
        })
        .collect()
}

pub fn run(a: &Args) -> Result<ExitCode> {
    let dir = PathBuf::from(
        a.get("session")
            .ok_or_else(|| cfg("--session <dir> is required"))?,
    );
    let endpoint_url = a.get("endpoint").unwrap_or("http://127.0.0.1:8010");
    let ep = Endpoint::parse(endpoint_url)?;
    let label = a
        .get("label")
        .map(String::from)
        .unwrap_or_else(|| format!("{}-{}", ep.host, ep.port));
    let out = a
        .get("out")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(format!("target/replay/{label}.json")));
    let max_steps: usize = a.parse("steps", usize::MAX)?;

    let text = std::fs::read_to_string(dir.join("session.json"))
        .map_err(|e| cfg(format!("{}: {e}", dir.join("session.json").display())))?;
    let session: Value =
        serde_json::from_str(&text).map_err(|e| cfg(format!("session.json: {e}")))?;
    let mut steps = Vec::new();
    collect_steps(&session, &dir, None, &mut steps)?;
    steps.truncate(max_steps);
    // --from N: start at the N-th step (1-based), e.g. `--from 2 --steps 2` replays step 2 only.
    let from: usize = a.parse("from", 1)?;
    steps.drain(..from.saturating_sub(1).min(steps.len()));
    if steps.is_empty() {
        return Err(cfg("no steps with a recorded request found"));
    }
    eprintln!(
        "replaying {} steps against {endpoint_url} as '{label}'",
        steps.len()
    );

    let mut rows = Vec::new();
    let (mut ttfts, mut totals, mut rec_lat) = (vec![], vec![], vec![]);
    let (mut srv_prefill, mut srv_decode) = (vec![], vec![]);
    let wall = Instant::now();
    for st in steps {
        let mut req = st.request;
        let data_url = match &st.screenshot {
            Some(p) if p.exists() => Some(format!(
                "data:image/jpeg;base64,{}",
                base64(&std::fs::read(p).map_err(|e| cfg(format!("{}: {e}", p.display())))?)
            )),
            _ => None,
        };
        let images = fill_images(&mut req, &data_url)?;
        if let Some(m) = a.get("model") {
            req["model"] = m.into();
        }
        req["stream"] = true.into();
        req["stream_options"] = json!({"include_usage": true});
        let body = serde_json::to_vec(&req).expect("serialisable");
        let m = stream_chat(&ep, &body)?;

        // Decode speed comes only from the server's timings: tool-call answers arrive in one
        // burst at the end, so a client-side TTFT-to-end rate is meaningless.
        let completion = m
            .usage
            .get("completion_tokens")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        if let Some(t) = m.ttft {
            ttfts.push(t);
        }
        totals.push(m.total);
        if let Some(p) = m.timings.get("prompt_ms").and_then(Value::as_f64) {
            srv_prefill.push(p);
        }
        if let Some(d) = m
            .timings
            .get("predicted_per_second")
            .and_then(Value::as_f64)
            .filter(|d| *d > 0.0)
        {
            srv_decode.push(d);
        }
        if let Some(l) = st.recorded.get("latencyMs").and_then(Value::as_f64) {
            rec_lat.push(l);
        }
        eprintln!(
            "step {:>3}: ttft {:>7.1} ms  total {:>7.1} ms  {:>4} tok  prefill {:>6.1} ms  decode {:>6.1} tok/s  (recorded {} ms)",
            st.index,
            m.ttft.unwrap_or(f64::NAN),
            m.total,
            completion,
            m.timings.get("prompt_ms").and_then(Value::as_f64).unwrap_or(f64::NAN),
            m.timings.get("predicted_per_second").and_then(Value::as_f64).unwrap_or(f64::NAN),
            st.recorded.get("latencyMs").map_or("-".into(), |v| v.to_string())
        );
        rows.push(json!({
            "step": st.index,
            "images": images,
            "ttft_ms": m.ttft,
            "total_ms": m.total,
            "usage": m.usage,
            "timings": m.timings,
            "finish_reason": m.finish,
            "output_hash": m.output_hash,
            "recorded": st.recorded,
        }));
    }
    let report = json!({
        "label": label,
        "endpoint": endpoint_url,
        "session": dir.file_name().map(|n| n.to_string_lossy().to_string()),
        "wall_s": wall.elapsed().as_secs_f64(),
        "summary": {"ttft_ms": stats(&mut ttfts), "total_ms": stats(&mut totals),"server_prefill_ms": stats(&mut srv_prefill), "server_decode_tps": stats(&mut srv_decode), "recorded_latency_ms": stats(&mut rec_lat)},
        "steps": rows,
    });
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent).map_err(|e| cfg(format!("{}: {e}", parent.display())))?;
    }
    std::fs::write(&out, serde_json::to_vec_pretty(&report).unwrap())
        .map_err(|e| cfg(format!("{}: {e}", out.display())))?;
    println!(
        "{}",
        serde_json::to_string_pretty(&report["summary"]).unwrap()
    );
    println!("report: {}", out.display());
    Ok(ExitCode::SUCCESS)
}

/// Fails when the candidate's median TTFT or latency rises, or median decode speed falls, by more
/// than the tolerance; also reports how many outputs are identical (temperature-0 sessions).
pub fn compare(a: &Args) -> Result<ExitCode> {
    let load = |k: &str| -> Result<Value> {
        let p = a
            .get(k)
            .ok_or_else(|| cfg(format!("--{k} <report.json> is required")))?;
        serde_json::from_str(&std::fs::read_to_string(p).map_err(|e| cfg(format!("{p}: {e}")))?)
            .map_err(|e| cfg(format!("{p}: {e}")))
    };
    let (base, cand) = (load("baseline")?, load("candidate")?);
    let tol: f64 = a.parse("tolerance", 3.0)? / 100.0;
    let p50 = |r: &Value, k: &str| r["summary"][k]["p50"].as_f64();
    let mut ok = true;
    println!(
        "{:<12} {:>12} {:>12} {:>9}  verdict",
        "metric", "baseline", "candidate", "change"
    );
    for (k, higher_is_better) in [
        ("ttft_ms", false),
        ("total_ms", false),
        ("server_prefill_ms", false),
        ("server_decode_tps", true),
    ] {
        let (Some(b), Some(c)) = (p50(&base, k), p50(&cand, k)) else {
            println!("{k:<12} missing in a report  FAIL");
            ok = false;
            continue;
        };
        let change = (c - b) / b;
        let regressed = if higher_is_better {
            change < -tol
        } else {
            change > tol
        };
        ok &= !regressed;
        println!(
            "{k:<12} {b:>12.1} {c:>12.1} {:>+8.1}%  {}",
            change * 100.0,
            if regressed { "REGRESSION" } else { "ok" }
        );
    }
    let hashes = |r: &Value| -> Vec<String> {
        r["steps"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|s| s["output_hash"].as_str().unwrap_or("").to_string())
            .collect()
    };
    let (hb, hc) = (hashes(&base), hashes(&cand));
    let same = hb.iter().zip(&hc).filter(|(x, y)| x == y).count();
    println!(
        "identical outputs: {same}/{} steps (temperature 0)",
        hb.len().min(hc.len())
    );
    println!(
        "tolerance {:.1}% on medians: {}",
        tol * 100.0,
        if ok { "PASS" } else { "FAIL" }
    );
    Ok(if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_rfc4648_vectors() {
        for (i, o) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(base64(i.as_bytes()), o);
        }
    }

    #[test]
    fn rebuilds_requests_from_the_envelope() {
        let session = json!({
            "requestEnvelope": {"model": "m", "temperature": 0, "system": {"role": "system", "content": "s"}},
            "steps": [
                {"requestMessages": [{"role": "user", "content": "a"}]},
                {"requestPrompt": "{\"model\":\"inline\",\"messages\":[]}"}
            ]
        });
        let mut steps = Vec::new();
        collect_steps(&session, Path::new("."), None, &mut steps).unwrap();
        assert_eq!(steps.len(), 2);
        assert_eq!(
            steps[0].request,
            json!({"model": "m", "temperature": 0, "messages": [
                {"role": "system", "content": "s"}, {"role": "user", "content": "a"}]})
        );
        assert_eq!(steps[1].request["model"], "inline");
    }

    #[test]
    fn fills_only_placeholders() {
        let mut v = json!({"messages": [{"content": [{"type": "image_url", "image_url": {"url": "[image data url omitted]"}}, {"type": "image_url", "image_url": {"url": "https://x/y.png"}}]}]});
        assert_eq!(
            fill_images(&mut v, &Some("data:image/jpeg;base64,AA==".into())).unwrap(),
            1
        );
        assert_eq!(
            v["messages"][0]["content"][0]["image_url"]["url"],
            "data:image/jpeg;base64,AA=="
        );
        assert_eq!(
            v["messages"][0]["content"][1]["image_url"]["url"],
            "https://x/y.png"
        );
        assert!(fill_images(&mut json!({"url": "[image data url]"}), &None).is_err());
    }

    #[test]
    fn endpoint_parsing() {
        let e = Endpoint::parse("http://127.0.0.1:8010/v1").unwrap();
        assert_eq!((e.host.as_str(), e.port), ("127.0.0.1", 8010));
        assert!(Endpoint::parse("https://x:1").is_err());
    }
}
