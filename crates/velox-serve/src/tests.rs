use super::*;

/// Emits "hi" then one tool call, with fixed accounting.
struct Scripted;

impl ChatEngine for Scripted {
    fn architecture(&self) -> &str {
        "Scripted"
    }

    fn generate(&self, _: &Value, emit: &mut dyn FnMut(Delta)) -> Result<Outcome, ApiError> {
        emit(Delta::Content("hi".into()));
        emit(Delta::ToolCall(
            json!({"index": 0, "id": "c1", "type": "function", "function": {"name": "click", "arguments": "{}"}}),
        ));
        Ok(Outcome {
            finish_reason: "tool_calls".into(),
            prompt_tokens: 100,
            cached_tokens: 60,
            completion_tokens: 11,
            queue_ms: 300.0,
            prompt_ms: 20.0,
            predicted_ms: 50.0,
            first_token: None,
            slot: None,
        })
    }
}

/// Emits a thinking answer: its reasoning, then the content and a call.
struct Thinking;

impl ChatEngine for Thinking {
    fn architecture(&self) -> &str {
        "Thinking"
    }

    fn generate(&self, _: &Value, emit: &mut dyn FnMut(Delta)) -> Result<Outcome, ApiError> {
        emit(Delta::Reasoning("The menu is open.".into()));
        emit(Delta::Content("hi".into()));
        emit(Delta::ToolCall(
            json!({"index": 0, "id": "c1", "type": "function", "function": {"name": "click", "arguments": "{}"}}),
        ));
        Ok(Outcome {
            finish_reason: "tool_calls".into(),
            prompt_tokens: 100,
            cached_tokens: 0,
            completion_tokens: 20,
            queue_ms: 0.0,
            prompt_ms: 1.0,
            predicted_ms: 1.0,
            first_token: None,
            slot: None,
        })
    }
}

#[test]
fn reasoning_comes_back_as_reasoning_content() {
    let port = start(Box::new(Thinking));
    let chat = r#"{"model":"m","messages":[{"role":"user","content":"x"}]}"#;
    let (status, body) = http(port, "POST", "/v1/chat/completions", chat);
    assert_eq!(status, 200);
    let v: Value = serde_json::from_str(&body).unwrap();
    let m = &v["choices"][0]["message"];
    assert_eq!(m["reasoning_content"], "The menu is open.");
    assert_eq!(m["content"], "hi");
    assert_eq!(m["tool_calls"][0]["function"]["name"], "click");

    let streamed = r#"{"model":"m","messages":[{"role":"user","content":"x"}],"stream":true}"#;
    let (_, body) = http(port, "POST", "/v1/chat/completions", streamed);
    let deltas: Vec<Value> = body
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str::<Value>(d).unwrap()["choices"][0]["delta"].clone())
        .collect();
    let first = |key: &str| deltas.iter().position(|d| d.get(key).is_some());
    assert_eq!(
        deltas[first("reasoning_content").unwrap()]["reasoning_content"],
        "The menu is open."
    );
    assert!(first("reasoning_content") < first("tool_calls"));
}

/// A reply with no reasoning carries no `reasoning_content` field at all.
#[test]
fn an_answer_without_reasoning_has_no_reasoning_field() {
    let port = start(Box::new(Scripted));
    let chat = r#"{"model":"m","messages":[{"role":"user","content":"x"}]}"#;
    let (_, body) = http(port, "POST", "/v1/chat/completions", chat);
    let v: Value = serde_json::from_str(&body).unwrap();
    assert!(v["choices"][0]["message"]
        .get("reasoning_content")
        .is_none());
}

fn start(engine: Box<dyn ChatEngine>) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = Server::new(
        Config {
            model_id: "m".into(),
            max_concurrency: 3,
            info: vec![("variant".into(), "generic".into())],
        },
        engine,
    );
    std::thread::spawn(move || server.serve(listener));
    port
}

fn http(port: u16, method: &str, path: &str, body: &str) -> (u16, String) {
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    write!(
        s,
        "{method} {path} HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    let status = out[9..12].parse().unwrap();
    (status, out.split_once("\r\n\r\n").unwrap().1.to_string())
}

/// An engine whose device context has failed: every generation errors like a sticky CUDA error.
struct Lost;

impl ChatEngine for Lost {
    fn architecture(&self) -> &str {
        "Lost"
    }

    fn generate(&self, _: &Value, _: &mut dyn FnMut(Delta)) -> Result<Outcome, ApiError> {
        Err(ApiError {
            status: 500,
            kind: "server_error",
            code: "engine_error",
            message: "backend error: x: CUDA error 999: unknown error".into(),
        })
    }

    fn lost(&self) -> Option<String> {
        Some("x: CUDA error 999: unknown error".into())
    }
}

#[test]
fn health_reports_a_lost_device() {
    let port = start(Box::new(Lost));
    let (s, body) = http(port, "GET", "/health", "");
    assert_eq!(s, 503);
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["status"], "device_lost");
    assert!(v["reason"].as_str().unwrap().contains("999"), "{body}");
    // The failing request still gets its 500 before the process would exit.
    let chat = r#"{"model":"m","messages":[{"role":"user","content":"x"}]}"#;
    assert_eq!(http(port, "POST", "/v1/chat/completions", chat).0, 500);
}

#[test]
fn models_health_metrics() {
    let port = start(Box::new(Scripted));
    assert_eq!(http(port, "GET", "/health", "").0, 200);
    let (s, body) = http(port, "GET", "/v1/models", "");
    assert_eq!(s, 200);
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["data"][0]["id"],
        "m"
    );
    let (_, metrics) = http(port, "GET", "/metrics", "");
    assert!(
        metrics.lines().any(|l| l == "velox_max_concurrency 3"),
        "{metrics}"
    );
    assert!(metrics.contains("variant=\"generic\""));
    assert!(metrics.lines().any(|l| l == "velox_prompt_tokens_total 0"));
    let chat = r#"{"model":"m","messages":[{"role":"user","content":"x"}]}"#;
    for _ in 0..2 {
        assert_eq!(http(port, "POST", "/v1/chat/completions", chat).0, 200);
    }
    let (_, metrics) = http(port, "GET", "/metrics", "");
    assert!(
        metrics
            .lines()
            .any(|l| l == "velox_prompt_tokens_total 200"),
        "{metrics}"
    );
    assert!(
        metrics
            .lines()
            .any(|l| l == "velox_prompt_tokens_cached_total 120"),
        "{metrics}"
    );
    // The engine's 300 ms queue wait lands between the 0.25 and 0.5 bounds.
    for line in [
        "velox_request_queue_seconds_bucket{le=\"0.25\"} 0",
        "velox_request_queue_seconds_bucket{le=\"0.5\"} 2",
        "velox_request_queue_seconds_count 2",
        "velox_time_to_first_token_seconds_bucket{le=\"+Inf\"} 2",
        "velox_request_duration_seconds_count 2",
    ] {
        assert!(metrics.lines().any(|l| l == line), "{line}\n{metrics}");
    }
    // Without scoring, the daemon's System One probe must read "unsupported".
    assert_eq!(http(port, "POST", "/v1/systemone", "{}").0, 404);
}

/// Scores the first candidate at -0.1 and the others at -3, after the prompt or after each
/// branch.
struct Scoring;

impl ChatEngine for Scoring {
    fn architecture(&self) -> &str {
        "Scoring"
    }

    fn generate(&self, _: &Value, _: &mut dyn FnMut(Delta)) -> Result<Outcome, ApiError> {
        Err(ApiError::not_implemented("scoring only"))
    }

    fn scores(&self) -> bool {
        true
    }

    fn score(&self, queries: &[ScoreQuery<'_>]) -> Result<Vec<ScoreOutcome>, ApiError> {
        Ok(queries
            .iter()
            .map(|q| {
                let row = (0..q.candidates.len()).map(|i| if i == 0 { -0.1 } else { -3.0 });
                ScoreOutcome {
                    logprobs: (0..q.branches.len().max(1))
                        .flat_map(|_| row.clone())
                        .collect(),
                    prompt_tokens: 50,
                    cached_tokens: 40,
                    queue_ms: 0.0,
                }
            })
            .collect())
    }
}

#[test]
fn systemone_answers_every_question() {
    let port = start(Box::new(Scoring));
    // The daemon's probe: an empty body is a 400 from a server that has the endpoint.
    let (s, body) = http(port, "POST", "/v1/systemone", "{}");
    assert_eq!(s, 400);
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["error"],
        "missing required field: state"
    );
    let req = json!({
        "model": "m",
        "state": "s",
        "questions": {
            "pick": {"type": "choice", "instructions": "?", "criteria": {"a": null, "b": null}},
            "similar": {"type": "rank", "instructions": "?", "criteria": {"x": "1", "y": "2"}},
        },
    });
    let (s, body) = http(port, "POST", "/v1/systemone", &req.to_string());
    assert_eq!(s, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["model"], "m");
    assert_eq!(v["answers"]["pick"]["choice"], "a");
    assert_eq!(v["answers"]["similar"]["type"], "rank");
    assert_eq!(
        v["usage"],
        json!({"input_tokens": 100, "cached_input_tokens": 80, "output_tokens": 3})
    );
    // llama.cpp's form leaves the model out: the loaded one answers.
    let mut no_model = req.clone();
    no_model.as_object_mut().unwrap().remove("model");
    let (s, body) = http(port, "POST", "/v1/systemone", &no_model.to_string());
    assert_eq!(s, 200, "{body}");
    assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["model"], "m");
    let mut wrong = req.clone();
    wrong["model"] = "other".into();
    assert_eq!(
        http(port, "POST", "/v1/systemone", &wrong.to_string()).0,
        404
    );
    assert_eq!(http(port, "GET", "/v1/systemone", "").0, 405);
    let (_, metrics) = http(port, "GET", "/metrics", "");
    assert!(
        metrics
            .lines()
            .any(|l| l == "velox_prompt_tokens_total 200"),
        "{metrics}"
    );
}

#[test]
fn histogram_buckets_are_inclusive() {
    let mut h = Histogram::default();
    for s in [0.25, 0.2501, 2000.0, f64::NAN, -1.0] {
        h.observe(s);
    }
    let mut out = String::new();
    h.render(&mut out, "x", "help");
    for line in [
        "x_bucket{le=\"0.1\"} 0",
        "x_bucket{le=\"0.25\"} 1",
        "x_bucket{le=\"0.5\"} 2",
        "x_bucket{le=\"1\"} 2",
        "x_bucket{le=\"1280\"} 2",
        "x_bucket{le=\"+Inf\"} 3",
        "x_count 3",
    ] {
        assert!(out.lines().any(|l| l == line), "{line}\n{out}");
    }
}

#[test]
fn non_streaming_completion() {
    let port = start(Box::new(Scripted));
    let (s, body) = http(
        port,
        "POST",
        "/v1/chat/completions",
        r#"{"model":"m","messages":[{"role":"user","content":"x"}]}"#,
    );
    assert_eq!(s, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "hi");
    assert_eq!(
        v["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "click"
    );
    assert_eq!(v["usage"]["prompt_tokens_details"]["cached_tokens"], 60);
    assert_eq!(v["timings"]["prompt_n"], 40);
    assert_eq!(v["timings"]["predicted_per_second"], 200.0);
}

#[test]
fn streaming_completion_is_chunked_sse() {
    let port = start(Box::new(Scripted));
    let (s, body) = http(
        port,
        "POST",
        "/v1/chat/completions",
        r#"{"messages":[{"role":"user","content":"x"}],"stream":true,"stream_options":{"include_usage":true}}"#,
    );
    assert_eq!(s, 200);
    let events: Vec<Value> = body
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).unwrap())
        .collect();
    assert_eq!(events[1]["choices"][0]["delta"]["content"], "hi");
    assert_eq!(
        events[2]["choices"][0]["delta"]["tool_calls"][0]["id"],
        "c1"
    );
    assert_eq!(events[3]["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(events[3]["timings"]["prompt_ms"], 20.0);
    assert_eq!(events[4]["usage"]["completion_tokens"], 11);
    assert!(body.contains("data: [DONE]"));
}

#[test]
fn errors_follow_openai_shape() {
    let port = start(Box::new(UnimplementedEngine {
        architecture: "Qwen3_5ForCausalLM".into(),
    }));
    let (s, body) = http(
        port,
        "POST",
        "/v1/chat/completions",
        r#"{"messages":[{"role":"user","content":"x"}]}"#,
    );
    assert_eq!(s, 501);
    assert!(
        serde_json::from_str::<Value>(&body).unwrap()["error"]["message"]
            .as_str()
            .unwrap()
            .contains("Qwen3_5ForCausalLM")
    );
    assert_eq!(
        http(port, "POST", "/v1/chat/completions", "not json").0,
        400
    );
    assert_eq!(
        http(port, "POST", "/v1/chat/completions", r#"{"messages":[]}"#).0,
        400
    );
    assert_eq!(
        http(
            port,
            "POST",
            "/v1/chat/completions",
            r#"{"model":"other","messages":[{"role":"user","content":"x"}]}"#
        )
        .0,
        404
    );
    let (_, metrics) = http(port, "GET", "/metrics", "");
    assert!(
        metrics.contains("velox_request_errors_total 4"),
        "{metrics}"
    );
}

/// An engine with reserved slots: lane 2 for token "a" (renewed when it asks again), 409 for
/// "full", and "a" is the only active token; chat requests report the lane they held.
struct Slotted;

impl ChatEngine for Slotted {
    fn architecture(&self) -> &str {
        "Slotted"
    }

    fn generate(&self, req: &Value, _: &mut dyn FnMut(Delta)) -> Result<Outcome, ApiError> {
        let held = req.get("slot_token").and_then(Value::as_str) == Some("a");
        Ok(Outcome {
            finish_reason: "stop".into(),
            prompt_tokens: 1,
            cached_tokens: 0,
            completion_tokens: 1,
            queue_ms: 0.0,
            prompt_ms: 1.0,
            predicted_ms: 1.0,
            first_token: None,
            slot: held.then_some(2),
        })
    }

    fn slot_reserve(&self, token: &str, d: Duration, w: Duration) -> Result<SlotGrant, SlotError> {
        if w.is_zero() {
            // A request whose token holds nothing reserves again without waiting (here: full).
            return Err(SlotError::NoSlot("held".into()));
        }
        assert_eq!(w, Duration::from_secs(30), "default wait");
        match token {
            "full" => Err(SlotError::NoSlot("held".into())),
            _ => Ok(SlotGrant {
                token: token.into(),
                lane: 2,
                expires_at_ms: 1_000 + d.as_millis() as u64,
                idle_timeout_s: d.as_secs_f64(),
                renewed: token == "a",
                waited_ms: 0,
            }),
        }
    }

    fn slot_release(&self, token: &str) -> Result<bool, SlotError> {
        Ok(token == "a")
    }

    fn slot_status(&self) -> Option<SlotStatus> {
        Some(SlotStatus {
            lanes: 4,
            max_holds: 2,
            holds: vec![SlotHold {
                token: "a".into(),
                lane: 2,
                expires_at_ms: 5_000,
                busy: false,
            }],
        })
    }

    fn slot_active(&self, token: &str) -> bool {
        token == "a"
    }

    fn load(&self) -> Option<EngineLoad> {
        Some(EngineLoad {
            slot_holds: 1,
            queue_depth: 3,
            lanes_busy: 2,
        })
    }
}

#[test]
fn slot_endpoints_follow_the_contract() {
    let port = start(Box::new(Slotted));
    let (status, body) = http(
        port,
        "POST",
        "/v1/slots/reserve",
        r#"{"token":"a"}"#,
    );
    assert_eq!(status, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        v,
        json!({"token": "a", "lane": 2, "expires_at_ms": 61_000, "idle_timeout_s": 60.0, "renewed": true, "waited_ms": 0})
    );
    // An older client's duration_s is ignored: the idle timeout is the server's.
    let (status, body) = http(port, "POST", "/v1/slots/reserve", r#"{"token":"a","duration_s":5}"#);
    assert_eq!(status, 200, "{body}");
    assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["idle_timeout_s"], 60.0);
    // A generated token when none is given.
    let (status, body) = http(port, "POST", "/v1/slots/reserve", "{}");
    assert_eq!(status, 200);
    let v: Value = serde_json::from_str(&body).unwrap();
    assert!(v["token"].as_str().unwrap().starts_with("slot_"));
    let (status, body) = http(
        port,
        "POST",
        "/v1/slots/reserve",
        r#"{"token":"full"}"#,
    );
    assert_eq!(status, 409);
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["error"]["code"],
        "no_slot"
    );
    for bad in [
        r#"{"token":"a b"}"#,
        r#"{"wait_timeout_s":121}"#,
        "nope",
    ] {
        assert_eq!(http(port, "POST", "/v1/slots/reserve", bad).0, 400, "{bad}");
    }
    assert_eq!(
        http(port, "POST", "/v1/slots/release", r#"{"token":"a"}"#).0,
        200
    );
    let (status, body) = http(port, "POST", "/v1/slots/release", r#"{"token":"b"}"#);
    assert_eq!(status, 404);
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["error"]["code"],
        "unknown_token"
    );
    let (status, body) = http(port, "GET", "/v1/slots", "");
    assert_eq!(status, 200);
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap(),
        json!({"lanes": 4, "max_holds": 2, "holds": [{"token": "a", "lane": 2, "expires_at_ms": 5000, "busy": false}]})
    );
    let (_, metrics) = http(port, "GET", "/metrics", "");
    assert!(metrics.contains("velox_slot_holds 1"));
    assert!(metrics.contains("velox_queue_depth 3"));
    assert!(metrics.contains("velox_lanes_busy 2"));
}

#[test]
fn chat_timings_report_the_slot() {
    let port = start(Box::new(Slotted));
    let held = r#"{"model":"m","messages":[{"role":"user","content":"x"}],"slot_token":"a"}"#;
    let (_, body) = http(port, "POST", "/v1/chat/completions", held);
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["timings"]["slot"], json!({"held": true, "lane": 2}));
    // An unknown token is not an error: not reserved again (all held), it runs unreserved.
    let other = r#"{"model":"m","messages":[{"role":"user","content":"x"}],"slot_token":"zz","stream":true}"#;
    let (status, body) = http(port, "POST", "/v1/chat/completions", other);
    assert_eq!(status, 200);
    let last = body
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str::<Value>(d).unwrap())
        .find(|c| c.get("timings").is_some())
        .unwrap();
    assert_eq!(
        last["timings"]["slot"],
        json!({"held": false, "lane": null})
    );
}

#[test]
fn engines_without_slots_answer_501() {
    let port = start(Box::new(Scripted));
    assert_eq!(http(port, "GET", "/v1/slots", "").0, 501);
    let (status, body) = http(port, "POST", "/v1/slots/reserve", "{}");
    assert_eq!(status, 501);
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["error"]["code"],
        "slots_unsupported"
    );
}
