//! Chat template rendering with HF `apply_chat_template` semantics (trim_blocks, lstrip_blocks)
//! and a `tojson` matching Python's `json.dumps` defaults (", " / ": ", no ASCII or HTML escaping,
//! key order kept), which is what ninfer's template engine produces.

use minijinja::{Environment, Error as JinjaError, ErrorKind, Value as JValue};
use serde_json::Value;
use velox_core::{Error, Result};

pub struct ChatTemplate {
    env: Environment<'static>,
}

/// Python `json.dumps(v, ensure_ascii=False)` with default separators.
pub fn py_json(v: &Value) -> String {
    let mut s = String::new();
    write_py_json(v, &mut s);
    s
}

fn write_py_json(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::String(s) => {
            out.push('"');
            for c in s.chars() {
                match c {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    '\u{08}' => out.push_str("\\b"),
                    '\u{0c}' => out.push_str("\\f"),
                    c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                    c => out.push(c),
                }
            }
            out.push('"');
        }
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_py_json(x, out);
            }
            out.push(']');
        }
        Value::Object(m) => {
            out.push('{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_py_json(&Value::String(k.clone()), out);
                out.push_str(": ");
                write_py_json(x, out);
            }
            out.push('}');
        }
    }
}

fn tojson(v: JValue) -> std::result::Result<JValue, JinjaError> {
    let json: Value = serde_json::to_value(&v)
        .map_err(|e| JinjaError::new(ErrorKind::InvalidOperation, e.to_string()))?;
    Ok(JValue::from_safe_string(py_json(&json)))
}

fn raise_exception(msg: String) -> std::result::Result<JValue, JinjaError> {
    Err(JinjaError::new(ErrorKind::InvalidOperation, msg))
}

impl ChatTemplate {
    pub fn new(source: String) -> Result<Self> {
        let mut env = Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_filter("tojson", tojson);
        env.add_function("raise_exception", raise_exception);
        env.add_template_owned("chat", source)
            .map_err(|e| Error::Config(format!("chat template: {e}")))?;
        Ok(ChatTemplate { env })
    }

    /// Renders `messages` (OpenAI shape; tool-call arguments already parsed to objects) with the
    /// given tools and flags.
    pub fn render(
        &self,
        messages: &Value,
        tools: Option<&Value>,
        enable_thinking: bool,
        add_generation_prompt: bool,
    ) -> Result<String> {
        let tmpl = self
            .env
            .get_template("chat")
            .map_err(|e| Error::Config(e.to_string()))?;
        let ctx = minijinja::context! {
            messages => JValue::from_serialize(messages),
            tools => tools.map(JValue::from_serialize),
            enable_thinking => enable_thinking,
            add_generation_prompt => add_generation_prompt,
        };
        tmpl.render(ctx)
            .map_err(|e| Error::Config(format!("chat template: {e:#}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn py_json_matches_python_dumps() {
        let v = json!({"type": "function", "f": {"name": "click", "d": "a<b> & \"q\" é"}, "n": [1, 2.5, true, null]});
        assert_eq!(
            py_json(&v),
            r#"{"type": "function", "f": {"name": "click", "d": "a<b> & \"q\" é"}, "n": [1, 2.5, true, null]}"#
        );
    }

    #[test]
    fn renders_qwen_style_template() {
        let src = "{%- for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}\
                   {%- if tools %}{{ tools[0] | tojson }}{% endif %}\
                   {%- if add_generation_prompt %}<|im_start|>assistant\n{% if enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}{% endif %}";
        let t = ChatTemplate::new(src.into()).unwrap();
        let out = t
            .render(
                &json!([{"role": "user", "content": "hi"}]),
                Some(&json!([{"name": "x"}])),
                false,
                true,
            )
            .unwrap();
        assert_eq!(out, "<|im_start|>user\nhi<|im_end|>\n{\"name\": \"x\"}<|im_start|>assistant\n<think>\n\n</think>\n\n");
    }

    #[test]
    fn python_string_methods_work() {
        let t =
            ChatTemplate::new("{{ 'abc'.startswith('a') }}{{ messages[::-1][0] }}".into()).unwrap();
        // Booleans print Python-style, like jinja2.
        assert_eq!(
            t.render(&json!([1, 2]), None, false, false).unwrap(),
            "True2"
        );
    }
}
