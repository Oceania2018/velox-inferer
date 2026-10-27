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
        self.render_with(
            messages,
            tools,
            enable_thinking,
            add_generation_prompt,
            false,
        )
    }

    /// Renders `messages` so the prompt ends inside their final assistant message, which the
    /// model then continues (ninfer's assistant prefill: `continue_final_message`, no generation
    /// prompt). The template decides how the open turn starts; with thinking off, ninfer's
    /// `qwen3_8.jinja` opens it with an empty think block as the generation prompt does.
    pub fn render_continuation(&self, messages: &Value, enable_thinking: bool) -> Result<String> {
        let last = messages.as_array().and_then(|m| m.last());
        if last.and_then(|m| m.get("role")).and_then(Value::as_str) != Some("assistant") {
            return Err(Error::Config(
                "assistant continuation requires a final assistant message".into(),
            ));
        }
        self.render_with(messages, None, enable_thinking, false, true)
    }

    fn render_with(
        &self,
        messages: &Value,
        tools: Option<&Value>,
        enable_thinking: bool,
        add_generation_prompt: bool,
        continue_final_message: bool,
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
            continue_final_message => continue_final_message,
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
    fn continuation_leaves_the_final_assistant_turn_open() {
        // The continuation branch of ninfer's qwen3_8.jinja, reduced.
        let src = "{%- for m in messages %}{%- set continuing = continue_final_message|default(false) and loop.last %}\
                   {%- if m.role == 'assistant' and continuing and enable_thinking is false %}<|im_start|>assistant\n<think>\n\n</think>\n\n{{ m.content|trim }}\
                   {%- else %}<|im_start|>{{ m.role }}\n{{ m.content|trim }}{% if not continuing %}<|im_end|>\n{% endif %}{% endif %}{% endfor %}\
                   {%- if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
        let t = ChatTemplate::new(src.into()).unwrap();
        let messages = json!([
            {"role": "system", "content": "s"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "Answer: ("}
        ]);
        assert_eq!(
            t.render_continuation(&messages, false).unwrap(),
            "<|im_start|>system\ns<|im_end|>\n<|im_start|>user\nq<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\nAnswer: ("
        );
        assert!(t
            .render_continuation(&json!([{"role": "user", "content": "q"}]), false)
            .is_err());
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
