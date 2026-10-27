//! Ollama-compatible System One decision endpoint (`POST /v1/systemone`), as ninfer serves it
//! (`src/serve/systemone.cpp`): every question becomes one scoring request whose answer comes
//! from candidate-token log-probabilities at a fixed position; nothing is decoded. Besides the
//! standard `choice`, `noul` and `score` questions, ninfer's `rank` extension grades up to 256
//! keyed candidates against the shared state in one request (one prefill plus one short branch per
//! candidate) on an ordered similarity scale. As in the spec (docs.ollama.com/api/systemone),
//! `images` are base64 strings shared by every question, answers carry their `type`, and errors
//! are `{"error": "<message>"}`. Extensions: `rank`, `system`, and `usage.cached_input_tokens`.
//!
//! Prompts, labels and answer formulas follow ninfer exactly, so both engines score the same
//! tokens for the same request.

use serde_json::{json, Map, Value};

use crate::ApiError;

/// Questions per request.
const MAX_QUESTIONS: usize = 64;
/// Choice options: single-token labels A..Z.
const MAX_CHOICE_OPTIONS: usize = 26;
/// Score levels and rank scale levels: single-token labels 0..9.
const MAX_SCORE_LEVELS: usize = 10;
/// Rank candidates. A rank branch is `"<id>] "`: ids below 1000 take at most five tokens.
pub const MAX_RANK_CANDIDATES: usize = 256;
/// Tokens of one rank branch, at most.
pub const MAX_BRANCH_TOKENS: usize = 5;

/// Default rank scale, lowest first; callers replace it with `scale`. Level 3 separates problems
/// on the same item that may share the cause or the remedy (including vague complaints) from
/// clearly different ones.
const DEFAULT_RANK_SCALE: [&str; 5] = [
    "unrelated item and problem",
    "different item with a similar symptom",
    "same item, clearly different problem",
    "same item, a problem that could share the cause or the remedy (including vague complaints \
     such as 'not working')",
    // A symptom-only request matches a past order that diagnosed a cause of that symptom.
    "same underlying problem: the same item and the same failure, including a past order whose \
     diagnosed cause explains the symptom in the new request",
];

const DEFAULT_SYSTEM: &str =
    "You answer decision questions about the given state. Follow the answer format exactly.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Choice,
    Noul,
    Score,
    Rank,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Choice => "choice",
            Kind::Noul => "noul",
            Kind::Score => "score",
            Kind::Rank => "rank",
        }
    }
}

/// One question as a scoring request.
#[derive(Clone, Debug)]
pub struct Question {
    pub name: String,
    pub kind: Kind,
    /// choice: option keys in request order. rank: candidate keys in scoring order (sorted, so
    /// the result does not depend on the order the caller listed them).
    pub keys: Vec<String>,
    /// score, rank: level descriptions, lowest first.
    pub legend: Vec<String>,
    /// rank: minimum similarity listed in the ranking.
    pub threshold: f64,
    /// rank: maximum ranking length.
    pub limit: Option<usize>,
    /// The chat request to score: system, user (images, then the text), and the assistant prefill
    /// the scored token follows; thinking off.
    pub request: Value,
    /// Texts whose (single) token is scored after the prompt, or after each branch.
    pub candidates: Vec<String>,
    /// rank: one continuation per candidate, appended to the prompt on its own.
    pub branches: Vec<String>,
}

#[derive(Debug)]
pub struct Request {
    pub model: String,
    pub questions: Vec<Question>,
}

fn bad_request(message: impl Into<String>) -> ApiError {
    ApiError::invalid(message)
}

fn with_code(message: impl Into<String>, code: &'static str) -> ApiError {
    ApiError {
        code,
        ..ApiError::invalid(message)
    }
}

fn require_text(value: &Value, param: &str) -> Result<String, ApiError> {
    match value.as_str() {
        Some(s) if !s.is_empty() => Ok(s.to_string()),
        _ => Err(bad_request(format!("{param} must be a nonempty string"))),
    }
}

fn optional_text(object: &Value, key: &str, param: &str) -> Result<Option<String>, ApiError> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => require_text(v, param).map(Some),
    }
}

fn is_empty_container(v: &Value) -> bool {
    match v {
        Value::Object(o) => o.is_empty(),
        Value::Array(a) => a.is_empty(),
        _ => false,
    }
}

fn pretty(v: &Value) -> String {
    serde_json::to_string_pretty(v).expect("a JSON value serializes")
}

/// Spec: "a nonempty string or JSON value"; a JSON value is shown to the model as indented JSON.
fn render_instructions(value: &Value, param: &str) -> Result<String, ApiError> {
    if value.is_string() {
        return require_text(value, param);
    }
    if value.is_null() || is_empty_container(value) {
        return Err(bad_request(format!(
            "{param} must be a nonempty string or JSON value"
        )));
    }
    Ok(pretty(value))
}

/// Media type from the leading base64 characters (the image is decoded with the rest of the
/// request's media).
fn sniff_image_type(base64: &str) -> &'static str {
    const KNOWN: [(&str, &str); 5] = [
        ("iVBORw0KGgo", "image/png"),
        ("/9j/", "image/jpeg"),
        ("UklGR", "image/webp"),
        ("R0lGOD", "image/gif"),
        ("Qk", "image/bmp"),
    ];
    KNOWN
        .iter()
        .find(|(prefix, _)| base64.starts_with(prefix))
        .map_or("application/octet-stream", |(_, t)| t)
}

/// Spec: base64-encoded images shared by every question, in request order; URLs and data URLs
/// are not accepted. Returned as OpenAI `image_url` parts with data URLs.
fn parse_images(body: &Value) -> Result<Vec<Value>, ApiError> {
    let images = match body.get("images") {
        None | Some(Value::Null) => return Ok(Vec::new()),
        Some(Value::Array(a)) => a,
        Some(_) => {
            return Err(bad_request(
                "images must be an array of base64-encoded images",
            ))
        }
    };
    images
        .iter()
        .enumerate()
        .map(|(i, image)| {
            let param = format!("images[{i}]");
            let data = require_text(image, &param)?;
            if data.starts_with("data:")
                || data.starts_with("http://")
                || data.starts_with("https://")
            {
                return Err(bad_request(format!(
                    "{param} must be base64-encoded image data; URLs and data URLs are not supported"
                )));
            }
            let url = format!("data:{};base64,{data}", sniff_image_type(&data));
            Ok(json!({"type": "image_url", "image_url": {"url": url}}))
        })
        .collect()
}

fn render_state(body: &Value) -> Result<String, ApiError> {
    let state = body
        .get("state")
        .ok_or_else(|| bad_request("missing required field: state"))?;
    if state.is_string() {
        return require_text(state, "state");
    }
    if (state.is_object() || state.is_array()) && !is_empty_container(state) {
        return Ok(pretty(state));
    }
    Err(bad_request(
        "state must be a nonempty string, object or array",
    ))
}

/// Scoring request: thinking off, and the final assistant prefill ends exactly where the scored
/// token belongs.
fn scoring_request(system: &str, user: String, assistant_prefill: &str) -> Value {
    json!({
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant_prefill},
        ],
        "chat_template_kwargs": {"enable_thinking": false},
    })
}

fn option_line(label: &str, key: &str, description: &Value) -> String {
    let mut line = format!("{label}. {key}");
    if let Some(d) = description.as_str().filter(|d| !d.is_empty()) {
        line.push_str(": ");
        line.push_str(d);
    }
    line
}

/// Builds a question's prompt and candidates: (item, param, state, instructions, system, out).
type BuildFn = fn(&Value, &str, &str, &str, &str, &mut Question) -> Result<(), ApiError>;

fn criteria(item: &Value) -> Option<&Value> {
    item.get("criteria").filter(|c| !c.is_null())
}

fn parse_choice(
    item: &Value,
    param: &str,
    state: &str,
    instructions: &str,
    system: &str,
    q: &mut Question,
) -> Result<(), ApiError> {
    let Some(Value::Object(options)) = item.get("criteria") else {
        return Err(bad_request(format!(
            "{param}.criteria must be an object of options"
        )));
    };
    if options.len() < 2 || options.len() > MAX_CHOICE_OPTIONS {
        return Err(with_code(
            format!("{param}.criteria must have 2 to {MAX_CHOICE_OPTIONS} options for this model"),
            "too_many_options",
        ));
    }
    let mut user = format!("{state}\n\nQuestion: {instructions}\n\nOptions:");
    for (i, (key, description)) in options.iter().enumerate() {
        if !description.is_null() && !description.is_string() {
            return Err(bad_request(format!(
                "{param}.criteria values must be strings or null"
            )));
        }
        let label = char::from(b'A' + i as u8).to_string();
        user.push('\n');
        user.push_str(&option_line(&label, key, description));
        q.candidates.push(format!(" {label}"));
        q.keys.push(key.clone());
    }
    user.push_str("\n\nAnswer with the letter of the single best option.");
    q.request = scoring_request(system, user, "Answer:");
    Ok(())
}

fn parse_noul(
    item: &Value,
    param: &str,
    state: &str,
    instructions: &str,
    system: &str,
    q: &mut Question,
) -> Result<(), ApiError> {
    let mut user = format!("{state}\n\nQuestion: {instructions}");
    if let Some(c) = criteria(item) {
        let Value::Object(map) = c else {
            return Err(bad_request(format!("{param}.criteria must be an object")));
        };
        if map.keys().any(|k| k != "true" && k != "false") {
            return Err(bad_request(format!(
                "{param}.criteria accepts only true and false"
            )));
        }
        if let Some(text) = optional_text(c, "true", &format!("{param}.criteria.true"))? {
            user.push_str(&format!("\ntrue means: {text}"));
        }
        if let Some(text) = optional_text(c, "false", &format!("{param}.criteria.false"))? {
            user.push_str(&format!("\nfalse means: {text}"));
        }
    }
    user.push_str("\n\nAnswer true or false.");
    q.request = scoring_request(system, user, "Answer:");
    q.candidates = [" true", " True", " false", " False"]
        .map(String::from)
        .to_vec();
    Ok(())
}

fn parse_score(
    item: &Value,
    param: &str,
    state: &str,
    instructions: &str,
    system: &str,
    q: &mut Question,
) -> Result<(), ApiError> {
    let Some(Value::Array(levels)) = item.get("criteria") else {
        return Err(bad_request(format!(
            "{param}.criteria must be an array of level descriptions"
        )));
    };
    if levels.len() < 2 || levels.len() > MAX_SCORE_LEVELS {
        return Err(bad_request(format!(
            "{param}.criteria must have 2 to 10 levels"
        )));
    }
    let mut user = format!("{state}\n\nQuestion: {instructions}\n\nScale (lowest first):");
    for (level, d) in levels.iter().enumerate() {
        let description = require_text(d, &format!("{param}.criteria[{level}]"))?;
        user.push_str(&format!("\n{level}: {description}"));
        q.legend.push(description);
        q.candidates.push(level.to_string());
    }
    // Qwen encodes " 7" as two tokens, so the prefill ends at "(" and the bare digit is scored.
    user.push_str("\n\nAnswer with the number of the level that fits best, in parentheses.");
    q.request = scoring_request(system, user, "Answer: (");
    Ok(())
}

/// The candidate list, the rating scale and the reply format form one shared prefix that ends
/// inside the assistant answer at "["; each candidate is then the branch "<id>] " and the rating
/// digit that follows is scored.
fn parse_rank(
    item: &Value,
    param: &str,
    state: &str,
    instructions: &str,
    system: &str,
    q: &mut Question,
) -> Result<(), ApiError> {
    let Some(Value::Object(map)) = item.get("criteria") else {
        return Err(bad_request(format!(
            "{param}.criteria must be an object of keyed candidate texts"
        )));
    };
    if map.is_empty() || map.len() > MAX_RANK_CANDIDATES {
        return Err(with_code(
            format!("{param}.criteria must have 1 to {MAX_RANK_CANDIDATES} candidates"),
            "too_many_candidates",
        ));
    }
    let mut candidates = Vec::with_capacity(map.len());
    for (key, text) in map {
        candidates.push((
            key.clone(),
            require_text(text, &format!("{param}.criteria.{key}"))?,
        ));
    }
    candidates.sort();
    if let Some(t) = item.get("threshold").filter(|t| !t.is_null()) {
        match t.as_f64() {
            Some(v) if (0.0..=1.0).contains(&v) => q.threshold = v,
            _ => {
                return Err(bad_request(format!(
                    "{param}.threshold must be a number in [0, 1]"
                )))
            }
        }
    }
    if let Some(l) = item.get("limit").filter(|l| !l.is_null()) {
        // Integers only, as ninfer's is_number_integer (2.5 and 10.0 are rejected).
        match l.as_i64() {
            Some(n) if n >= 1 => q.limit = Some(n as usize),
            _ => {
                return Err(bad_request(format!(
                    "{param}.limit must be a positive integer"
                )))
            }
        }
    }
    match item.get("scale").filter(|s| !s.is_null()) {
        Some(scale) => {
            let levels = scale
                .as_array()
                .filter(|a| (2..=MAX_SCORE_LEVELS).contains(&a.len()))
                .ok_or_else(|| {
                    bad_request(format!(
                        "{param}.scale must be an array of 2 to 10 level descriptions"
                    ))
                })?;
            for (level, d) in levels.iter().enumerate() {
                q.legend
                    .push(require_text(d, &format!("{param}.scale[{level}]"))?);
            }
        }
        None => q.legend = DEFAULT_RANK_SCALE.map(String::from).to_vec(),
    }

    let mut user = format!("{state}\n\nCandidates:");
    for (id, (key, text)) in candidates.into_iter().enumerate() {
        // Newlines inside a candidate would read as the start of the next candidate.
        user.push_str(&format!("\n[{id}] {}", text.replace('\n', " ")));
        q.keys.push(key);
        q.branches.push(format!("{id}] "));
    }
    user.push_str(&format!("\n\n{instructions}\n"));
    for (level, description) in q.legend.iter().enumerate().rev() {
        user.push_str(&format!("\n{level}: {description}"));
    }
    user.push_str(
        "\n\nI will name one candidate by its id. Reply with the id in brackets followed by its \
         rating.",
    );
    q.request = scoring_request(system, user, "[");
    q.candidates = (0..q.legend.len()).map(|l| l.to_string()).collect();
    Ok(())
}

/// Parses a `/v1/systemone` body into one scoring request per question, in request order.
pub fn parse(body: &Value) -> Result<Request, ApiError> {
    if !body.is_object() {
        return Err(bad_request("request body must be a JSON object"));
    }
    let model = require_text(
        body.get("model")
            .ok_or_else(|| bad_request("missing required field: model"))?,
        "model",
    )?;
    let images = parse_images(body)?;
    // Accepted for compatibility; the model stays loaded for the life of the server.
    if let Some(k) = body.get("keep_alive") {
        if !k.is_null() && !k.is_string() && !k.is_number() {
            return Err(bad_request(
                "keep_alive must be a duration string or a number of seconds",
            ));
        }
    }
    let state = render_state(body)?;
    let system = optional_text(body, "system", "system")?.unwrap_or_else(|| DEFAULT_SYSTEM.into());
    let questions = match body.get("questions") {
        Some(Value::Object(q)) if !q.is_empty() && q.len() <= MAX_QUESTIONS => q,
        _ => {
            return Err(bad_request(
                "questions must be an object of 1 to 64 named questions",
            ))
        }
    };
    let mut out = Vec::with_capacity(questions.len());
    for (name, item) in questions {
        let param = format!("questions.{name}");
        if !item.is_object() {
            return Err(bad_request(format!("{param} must be an object")));
        }
        let kind = require_text(
            item.get("type")
                .ok_or_else(|| bad_request(format!("{param}.type is required")))?,
            &format!("{param}.type"),
        )?;
        let instructions = render_instructions(
            item.get("instructions")
                .ok_or_else(|| bad_request(format!("{param}.instructions is required")))?,
            &format!("{param}.instructions"),
        )?;
        let mut q = Question {
            name: name.clone(),
            kind: Kind::Choice,
            keys: Vec::new(),
            legend: Vec::new(),
            threshold: 0.5,
            limit: None,
            request: Value::Null,
            candidates: Vec::new(),
            branches: Vec::new(),
        };
        let (kind, build): (Kind, BuildFn) = match kind.as_str() {
            "choice" => (Kind::Choice, parse_choice),
            "noul" => (Kind::Noul, parse_noul),
            "score" => (Kind::Score, parse_score),
            "rank" => (Kind::Rank, parse_rank),
            _ => {
                return Err(bad_request(format!(
                    "{param}.type must be choice, noul, score or rank"
                )))
            }
        };
        q.kind = kind;
        build(item, &param, &state, &instructions, &system, &mut q)?;
        // Images come before the text in the user turn, as in an ordinary vision prompt.
        if !images.is_empty() {
            let user = &mut q.request["messages"][1];
            let text = user["content"].take();
            let mut parts = images.clone();
            parts.push(json!({"type": "text", "text": text}));
            user["content"] = Value::Array(parts);
        }
        out.push(q);
    }
    Ok(Request {
        model,
        questions: out,
    })
}

/// Probabilities of `logprobs` renormalized among themselves (all zero when none is finite).
fn normalized(logprobs: &[f32]) -> Vec<f64> {
    let peak = logprobs
        .iter()
        .map(|&v| f64::from(v))
        .fold(f64::NEG_INFINITY, f64::max);
    if !peak.is_finite() {
        return vec![0.0; logprobs.len()];
    }
    let e: Vec<f64> = logprobs
        .iter()
        .map(|&v| (f64::from(v) - peak).exp())
        .collect();
    let sum: f64 = e.iter().sum();
    e.into_iter().map(|v| v / sum).collect()
}

/// Spec: 1 - H(p)/ln(N); 0 for uniform probabilities, near 1 when one candidate dominates.
fn confidence(p: &[f64]) -> f64 {
    let entropy: f64 = p.iter().filter(|&&v| v > 0.0).map(|&v| -v * v.ln()).sum();
    1.0 - entropy / (p.len() as f64).ln()
}

/// The answer object for `q` from its candidates' log-probabilities (branch-major for rank).
pub fn answer(q: &Question, logprobs: &[f32]) -> Result<Value, ApiError> {
    let mut a = Map::new();
    a.insert("type".into(), q.kind.name().into());
    let want = match q.kind {
        Kind::Rank => q.keys.len() * q.legend.len(),
        _ => q.candidates.len(),
    };
    if logprobs.len() != want {
        return Err(ApiError {
            status: 500,
            kind: "server_error",
            code: "internal_error",
            message: format!(
                "scoring returned {} values for {want} candidates",
                logprobs.len()
            ),
        });
    }
    let legend = |q: &Question| -> Map<String, Value> {
        q.legend
            .iter()
            .enumerate()
            .map(|(l, d)| (l.to_string(), d.clone().into()))
            .collect()
    };
    match q.kind {
        Kind::Choice => {
            let p = normalized(logprobs);
            let mut best = 0;
            let mut probabilities = Map::new();
            for (i, &v) in p.iter().enumerate() {
                probabilities.insert(q.keys[i].clone(), v.into());
                // Ties keep the earlier option.
                if v > p[best] {
                    best = i;
                }
            }
            a.insert("choice".into(), q.keys[best].clone().into());
            a.insert("probabilities".into(), probabilities.into());
            a.insert("confidence".into(), confidence(&p).into());
        }
        Kind::Noul => {
            // The affirmative variants (" true", " True") against the negative ones.
            let p = normalized(logprobs);
            let half = p.len() / 2;
            a.insert("noul".into(), p[..half].iter().sum::<f64>().into());
        }
        Kind::Score => {
            let p = normalized(logprobs);
            let mut probabilities = Map::new();
            let mut score = 0.0;
            for (level, &v) in p.iter().enumerate() {
                score += level as f64 * v;
                probabilities.insert(level.to_string(), v.into());
            }
            a.insert("score".into(), score.into());
            a.insert("legend".into(), legend(q).into());
            a.insert("probabilities".into(), probabilities.into());
            a.insert("confidence".into(), confidence(&p).into());
        }
        Kind::Rank => {
            let levels = q.legend.len();
            // Similarity is the expected level over the rating distribution, scaled to [0, 1].
            let mut scored: Vec<(f64, usize)> = (0..q.keys.len())
                .map(|id| {
                    let p = normalized(&logprobs[id * levels..(id + 1) * levels]);
                    let expected: f64 = p.iter().enumerate().map(|(l, v)| l as f64 * v).sum();
                    (expected / (levels - 1) as f64, id)
                })
                .collect();
            // Most similar first; equal similarities keep the sorted key order.
            scored.sort_by(|x, y| y.0.total_cmp(&x.0));
            let mut ranking = Vec::new();
            let mut similarities = Map::new();
            for &(similarity, id) in &scored {
                similarities.insert(q.keys[id].clone(), similarity.into());
                if similarity >= q.threshold && q.limit.is_none_or(|l| ranking.len() < l) {
                    ranking.push(json!({"key": q.keys[id], "similarity": similarity}));
                }
            }
            a.insert("ranking".into(), ranking.into());
            a.insert("similarities".into(), similarities.into());
            a.insert("legend".into(), legend(q).into());
            a.insert("threshold".into(), q.threshold.into());
        }
    }
    Ok(Value::Object(a))
}

/// Internal scoring positions a question consumes (`usage.output_tokens`).
pub fn output_tokens(q: &Question) -> u64 {
    match q.kind {
        Kind::Rank => q.keys.len() as u64,
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rank_body(criteria: Value) -> Value {
        json!({
            "model": "qwen3.8-27b",
            "state": "New request: kitchen faucet leaking",
            "questions": {"similar": {
                "type": "rank",
                "instructions": "Rate how similar each past work order is to the new request:",
                "criteria": criteria,
            }},
        })
    }

    fn numbered(count: usize) -> Value {
        Value::Object(
            (0..count)
                .map(|i| (format!("WO-{i}"), format!("candidate {i}").into()))
                .collect(),
        )
    }

    /// The error code a parse raises, or "" when it succeeds.
    fn rejection(body: &Value) -> String {
        match parse(body) {
            Ok(_) => String::new(),
            Err(e) => e.code.to_string(),
        }
    }

    fn user_text(q: &Question) -> String {
        match &q.request["messages"][1]["content"] {
            Value::String(s) => s.clone(),
            Value::Array(parts) => parts.last().unwrap()["text"].as_str().unwrap().into(),
            _ => panic!("user content"),
        }
    }

    #[test]
    fn rank_scores_candidates_in_key_order() {
        let r = parse(&rank_body(
            json!({"b": "second", "a": "first", "c": "third"}),
        ))
        .unwrap();
        assert_eq!(r.questions.len(), 1);
        let q = &r.questions[0];
        assert_eq!(q.kind, Kind::Rank);
        assert_eq!(q.keys, ["a", "b", "c"]);
        assert_eq!(q.branches, ["0] ", "1] ", "2] "]);
        assert_eq!(q.candidates, ["0", "1", "2", "3", "4"]);
        assert_eq!((q.threshold, q.limit), (0.5, None));
        let prompt = user_text(q);
        assert!(prompt.contains("[0] first") && prompt.contains("[2] third"));
        assert_eq!(q.request["messages"][2]["content"], "[");
        assert_eq!(q.request["chat_template_kwargs"]["enable_thinking"], false);
        // The same candidates listed differently build the same prompt.
        let reordered = parse(&rank_body(
            json!({"c": "third", "a": "first", "b": "second"}),
        ))
        .unwrap();
        assert_eq!(user_text(&reordered.questions[0]), prompt);
    }

    #[test]
    fn rank_prompt_is_ninfers() {
        let mut body = rank_body(json!({"x": "line one\nline two"}));
        body["questions"]["similar"]["scale"] = json!(["no", "yes"]);
        let q = &parse(&body).unwrap().questions[0];
        assert_eq!(
            user_text(q),
            "New request: kitchen faucet leaking\n\nCandidates:\n[0] line one line two\n\n\
             Rate how similar each past work order is to the new request:\n\n1: yes\n0: no\n\n\
             I will name one candidate by its id. Reply with the id in brackets followed by its \
             rating."
        );
        assert_eq!(q.request["messages"][0]["content"], DEFAULT_SYSTEM);
    }

    #[test]
    fn rank_options_and_limits() {
        let mut custom = rank_body(json!({"x": "only"}));
        custom["questions"]["similar"]["scale"] = json!(["no", "partly", "yes"]);
        custom["questions"]["similar"]["threshold"] = json!(0.7);
        custom["questions"]["similar"]["limit"] = json!(10);
        let q = &parse(&custom).unwrap().questions[0];
        assert_eq!(q.candidates.len(), 3);
        assert_eq!((q.threshold, q.limit), (0.7, Some(10)));

        assert_eq!(rejection(&rank_body(numbered(MAX_RANK_CANDIDATES))), "");
        assert_eq!(
            rejection(&rank_body(numbered(MAX_RANK_CANDIDATES + 1))),
            "too_many_candidates"
        );
        assert_eq!(rejection(&rank_body(json!({}))), "too_many_candidates");
        for (field, value) in [
            ("threshold", json!(1.5)),
            ("scale", json!(["only one level"])),
            ("limit", json!(0)),
            ("limit", json!(2.5)),
        ] {
            let mut bad = rank_body(json!({"x": "only"}));
            bad["questions"]["similar"][field] = value;
            assert!(!rejection(&bad).is_empty(), "{field}");
        }
    }

    #[test]
    fn images_are_base64_and_precede_the_text() {
        let mut body = rank_body(json!({"x": "only"}));
        body["images"] = json!(["iVBORw0KGgoAAAA=", "/9j/4AAQ"]);
        body["questions"]["done"] = json!({"type": "noul", "instructions": "Is the task done?"});
        for q in parse(&body).unwrap().questions {
            let parts = q.request["messages"][1]["content"]
                .as_array()
                .unwrap()
                .clone();
            assert_eq!(parts.len(), 3);
            assert_eq!(
                parts[0]["image_url"]["url"],
                "data:image/png;base64,iVBORw0KGgoAAAA="
            );
            assert_eq!(
                parts[1]["image_url"]["url"],
                "data:image/jpeg;base64,/9j/4AAQ"
            );
            assert_eq!(parts[2]["type"], "text");
        }
        for source in ["data:image/png;base64,iVBORw0KGgo=", "https://x/a.png", ""] {
            let mut bad = rank_body(json!({"x": "only"}));
            bad["images"] = json!([source]);
            assert!(!rejection(&bad).is_empty(), "{source}");
        }
        let mut object_images = rank_body(json!({"x": "only"}));
        object_images["images"] = json!({"data": "aGVsbG8="});
        assert!(!rejection(&object_images).is_empty());
        let mut empty = rank_body(json!({"x": "only"}));
        empty["images"] = json!([]);
        assert_eq!(rejection(&empty), "");
    }

    #[test]
    fn instructions_and_keep_alive() {
        let mut json_instructions = rank_body(json!({"x": "only"}));
        json_instructions["questions"]["similar"]["instructions"] =
            json!({"goal": "rate similarity"});
        json_instructions["keep_alive"] = json!("5m");
        let q = &parse(&json_instructions).unwrap().questions[0];
        assert!(user_text(q).contains("{\n  \"goal\": \"rate similarity\"\n}"));
        let mut empty = rank_body(json!({"x": "only"}));
        empty["questions"]["similar"]["instructions"] = json!({});
        assert!(!rejection(&empty).is_empty());
        let mut bad_keep_alive = rank_body(json!({"x": "only"}));
        bad_keep_alive["keep_alive"] = json!([]);
        assert!(!rejection(&bad_keep_alive).is_empty());
        assert!(!rejection(&json!({"model": "m", "state": "s"})).is_empty());
        let mut unknown = rank_body(json!({"x": "only"}));
        unknown["questions"]["similar"]["type"] = json!("essay");
        assert!(!rejection(&unknown).is_empty());
        // The daemon's probe: an empty body is a 400.
        assert_eq!(parse(&json!({})).unwrap_err().status, 400);
    }

    #[test]
    fn choice_noul_and_score_prompts() {
        let body = json!({
            "model": "m",
            "state": {"screen": "login"},
            "system": "Decide.",
            "questions": {
                "pick": {"type": "choice", "instructions": "Which one?",
                         "criteria": {"a": null, "b": "second"}},
                "done": {"type": "noul", "instructions": "Done?",
                         "criteria": {"false": "not yet", "true": "finished"}},
                "level": {"type": "score", "instructions": "How bad?", "criteria": ["low", "high"]},
            },
        });
        let r = parse(&body).unwrap();
        let names: Vec<&str> = r.questions.iter().map(|q| q.name.as_str()).collect();
        assert_eq!(names, ["pick", "done", "level"]);
        let state = "{\n  \"screen\": \"login\"\n}";
        let (pick, done, level) = (&r.questions[0], &r.questions[1], &r.questions[2]);
        assert_eq!(
            user_text(pick),
            format!(
                "{state}\n\nQuestion: Which one?\n\nOptions:\nA. a\nB. b: second\n\n\
                 Answer with the letter of the single best option."
            )
        );
        assert_eq!(pick.candidates, [" A", " B"]);
        assert_eq!(pick.request["messages"][2]["content"], "Answer:");
        assert_eq!(pick.request["messages"][0]["content"], "Decide.");
        // true is described first whatever order the caller used.
        assert_eq!(
            user_text(done),
            format!(
                "{state}\n\nQuestion: Done?\ntrue means: finished\nfalse means: not yet\n\n\
                 Answer true or false."
            )
        );
        assert_eq!(done.candidates, [" true", " True", " false", " False"]);
        assert_eq!(
            user_text(level),
            format!(
                "{state}\n\nQuestion: How bad?\n\nScale (lowest first):\n0: low\n1: high\n\n\
                 Answer with the number of the level that fits best, in parentheses."
            )
        );
        assert_eq!(level.request["messages"][2]["content"], "Answer: (");
        assert_eq!(level.candidates, ["0", "1"]);
    }

    #[test]
    fn answers_follow_ninfers_formulas() {
        let body = json!({
            "model": "m", "state": "s",
            "questions": {
                "pick": {"type": "choice", "instructions": "?", "criteria": {"a": null, "b": null}},
                "done": {"type": "noul", "instructions": "?"},
                "level": {"type": "score", "instructions": "?", "criteria": ["l", "m", "h"]},
                "similar": {"type": "rank", "instructions": "?", "scale": ["no", "yes"],
                            "criteria": {"x": "1", "y": "2", "z": "3"}, "threshold": 0.4,
                            "limit": 1},
            },
        });
        let r = parse(&body).unwrap();
        let ln = |p: f64| p.ln() as f32;
        let pick = answer(&r.questions[0], &[ln(0.25), ln(0.75)]).unwrap();
        assert_eq!(pick["type"], "choice");
        assert_eq!(pick["choice"], "b");
        let c = pick["confidence"].as_f64().unwrap();
        let h = -(0.25f64 * 0.25f64.ln() + 0.75 * 0.75f64.ln());
        assert!((c - (1.0 - h / 2f64.ln())).abs() < 1e-6);
        // Candidates renormalize among themselves.
        let done = answer(
            &r.questions[1],
            &[ln(0.1), ln(0.1), ln(0.2), f32::NEG_INFINITY],
        )
        .unwrap();
        assert!((done["noul"].as_f64().unwrap() - 0.5).abs() < 1e-6);
        let level = answer(&r.questions[2], &[ln(0.2), ln(0.3), ln(0.5)]).unwrap();
        assert!((level["score"].as_f64().unwrap() - 1.3).abs() < 1e-6);
        assert_eq!(level["legend"]["2"], "h");
        // Rank: x rates yes with 0.2, y with 0.9, z with 0.5; threshold 0.4, limit 1.
        let rank = answer(
            &r.questions[3],
            &[ln(0.8), ln(0.2), ln(0.1), ln(0.9), ln(0.5), ln(0.5)],
        )
        .unwrap();
        assert_eq!(rank["type"], "rank");
        let order: Vec<&str> = rank["similarities"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(order, ["y", "z", "x"]);
        assert_eq!(rank["ranking"].as_array().unwrap().len(), 1);
        assert_eq!(rank["ranking"][0]["key"], "y");
        assert!((rank["similarities"]["x"].as_f64().unwrap() - 0.2).abs() < 1e-6);
        assert_eq!(rank["threshold"], 0.4);
        assert!(answer(&r.questions[3], &[0.0]).is_err());
        assert_eq!(output_tokens(&r.questions[3]), 3);
        assert_eq!(output_tokens(&r.questions[0]), 1);
    }
}
