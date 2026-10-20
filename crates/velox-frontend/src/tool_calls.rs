//! Parses Qwen XML tool calls out of generated text into OpenAI `tool_calls`:
//!
//! ```text
//! <tool_call>
//! <function=NAME>
//! <parameter=P>
//! VALUE
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! Parameter values are typed with the request's tool schema: `string` parameters keep the raw
//! text; others are parsed as JSON and fall back to the raw text.

use serde_json::{json, Map, Value};

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedOutput {
    pub content: String,
    pub reasoning: Option<String>,
    /// OpenAI `tool_calls` entries (`arguments` is a JSON string).
    pub tool_calls: Vec<Value>,
}

fn param_type<'a>(tools: Option<&'a Value>, function: &str, param: &str) -> Option<&'a str> {
    tools?.as_array()?.iter().find_map(|t| {
        let f = t.get("function").unwrap_or(t);
        (f.get("name")?.as_str()? == function)
            .then(|| {
                f.pointer(&format!("/parameters/properties/{param}/type"))?
                    .as_str()
            })
            .flatten()
    })
}

fn typed_value(raw: &str, ty: Option<&str>) -> Value {
    if ty == Some("string") {
        return Value::String(raw.to_string());
    }
    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
}

/// Strips one leading and one trailing newline, as the template writes `\nVALUE\n`.
fn strip_newlines(s: &str) -> &str {
    let s = s.strip_prefix('\n').unwrap_or(s);
    s.strip_suffix('\n').unwrap_or(s)
}

fn parse_call(body: &str, tools: Option<&Value>, index: usize) -> Option<Value> {
    let start = body.find("<function=")? + "<function=".len();
    let name_end = start + body[start..].find('>')?;
    let name = body[start..name_end].trim().to_string();
    let mut args = Map::new();
    let mut rest = &body[name_end + 1..];
    while let Some(p) = rest.find("<parameter=") {
        let ps = p + "<parameter=".len();
        let pe = ps + rest[ps..].find('>')?;
        let pname = rest[ps..pe].trim().to_string();
        let ve = pe + 1 + rest[pe + 1..].find("</parameter>")?;
        let raw = strip_newlines(&rest[pe + 1..ve]);
        args.insert(
            pname.clone(),
            typed_value(raw, param_type(tools, &name, &pname)),
        );
        rest = &rest[ve + "</parameter>".len()..];
    }
    Some(json!({
        "index": index,
        "id": format!("call_{index}"),
        "type": "function",
        "function": {"name": name, "arguments": Value::Object(args).to_string()},
    }))
}

/// Splits generated text into reasoning, visible content and tool calls.
pub fn parse_output(text: &str, tools: Option<&Value>) -> ParsedOutput {
    let (reasoning, after) = match text.find("</think>") {
        Some(i) => {
            let r = text[..i].trim_start_matches("<think>").trim();
            (
                Some(r.to_string()).filter(|r| !r.is_empty()),
                &text[i + "</think>".len()..],
            )
        }
        None => (None, text),
    };
    let mut tool_calls = Vec::new();
    let content_end = after.find("<tool_call>").unwrap_or(after.len());
    let mut rest = &after[content_end..];
    while let Some(s) = rest.find("<tool_call>") {
        let body_start = s + "<tool_call>".len();
        let end = rest[body_start..]
            .find("</tool_call>")
            .map(|e| body_start + e);
        let body = &rest[body_start..end.unwrap_or(rest.len())];
        if let Some(call) = parse_call(body, tools, tool_calls.len()) {
            tool_calls.push(call);
        }
        match end {
            Some(e) => rest = &rest[e + "</tool_call>".len()..],
            None => break,
        }
    }
    ParsedOutput {
        content: after[..content_end].trim().to_string(),
        reasoning,
        tool_calls,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_typed_parameters() {
        let tools = json!([{"type": "function", "function": {"name": "click", "parameters": {"properties": {
            "x": {"type": "integer"}, "label": {"type": "string"}, "opts": {"type": "object"}}}}}]);
        let text = "I will click.\n\n<tool_call>\n<function=click>\n<parameter=x>\n42\n</parameter>\n\
                    <parameter=label>\n007\n</parameter>\n<parameter=opts>\n{\"double\": true}\n</parameter>\n</function>\n</tool_call>";
        let p = parse_output(text, Some(&tools));
        assert_eq!(p.content, "I will click.");
        assert_eq!(p.tool_calls.len(), 1);
        let args: Value =
            serde_json::from_str(p.tool_calls[0]["function"]["arguments"].as_str().unwrap())
                .unwrap();
        assert_eq!(
            args,
            json!({"x": 42, "label": "007", "opts": {"double": true}})
        );
        assert_eq!(p.tool_calls[0]["function"]["name"], "click");
    }

    #[test]
    fn multiple_calls_reasoning_and_multiline_values() {
        let text = "<think>\nplan\n</think>\n\n<tool_call>\n<function=a>\n<parameter=t>\nline1\nline2\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=b>\n</function>\n</tool_call>";
        let p = parse_output(text, None);
        assert_eq!(p.reasoning.as_deref(), Some("plan"));
        assert_eq!(p.content, "");
        assert_eq!(p.tool_calls.len(), 2);
        assert_eq!(
            p.tool_calls[0]["function"]["arguments"],
            "{\"t\":\"line1\\nline2\"}"
        );
        assert_eq!(p.tool_calls[1]["id"], "call_1");
    }

    #[test]
    fn plain_text_has_no_calls() {
        let p = parse_output("Paris.", None);
        assert_eq!((p.content.as_str(), p.tool_calls.len()), ("Paris.", 0));
    }
}
