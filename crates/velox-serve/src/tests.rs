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
        })
    }
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
        "missing required field: model"
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
            .any(|l| l == "velox_prompt_tokens_total 100"),
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
