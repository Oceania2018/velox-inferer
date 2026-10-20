//! `velox generate`: greedy generation with a real artifact, from token ids or chat prompts.
//! `--prompts-file` runs one prompt per line with a single model load and prints JSON lines, which
//! is how outputs are compared with another engine.

use std::process::ExitCode;
use std::time::Instant;

use serde_json::json;
use velox_core::{Backend, Error, Result};
use velox_frontend::Frontend;
use velox_models::qwen3_5::{argmax, Qwen35Text, TextConfig, TextWeights};
use velox_weights::Artifact;

use crate::args::Args;
use crate::select::Selected;

pub fn run(a: &Args) -> Result<ExitCode> {
    let sel = crate::select::backend(a)?;
    match &sel {
        Selected::Cpu(b) => generate(b, a),
        #[cfg(feature = "cuda")]
        Selected::Cuda(b) => generate(b, a),
    }
}

enum Input {
    Prompt(String),
    Ids(Vec<u32>),
}

fn generate<B: Backend>(b: &B, a: &Args) -> Result<ExitCode> {
    let path = a
        .get("artifact")
        .ok_or_else(|| Error::Config("--artifact is required".into()))?;
    let max_tokens: usize = a.parse("max-tokens", 64)?;
    let chunk: usize = a.parse("chunk", 256)?;
    let capacity: usize = a.parse("capacity", 4096)?;
    let art = Artifact::open(path).map_err(|e| Error::Config(e.to_string()))?;
    let template = match a.get("chat-template") {
        Some(p) => {
            Some(std::fs::read_to_string(p).map_err(|e| Error::Config(format!("{p}: {e}")))?)
        }
        None => None,
    };
    let fe = Frontend::from_artifact(&art, template)?;
    let (inputs, jsonl) = match (a.get("prompt"), a.get("prompts-file"), a.get("ids")) {
        (Some(p), _, _) => (vec![Input::Prompt(p.to_string())], false),
        (None, Some(f), _) => {
            let text =
                std::fs::read_to_string(f).map_err(|e| Error::Config(format!("{f}: {e}")))?;
            (
                text.lines()
                    .filter(|l| !l.trim().is_empty())
                    .map(|l| Input::Prompt(l.to_string()))
                    .collect(),
                true,
            )
        }
        (None, None, Some(list)) => {
            let ids = list
                .split(',')
                .map(|s| {
                    s.trim()
                        .parse()
                        .map_err(|_| Error::Config(format!("bad token id '{s}'")))
                })
                .collect::<Result<_>>()?;
            (vec![Input::Ids(ids)], false)
        }
        _ => {
            return Err(Error::Config(
                "--prompt, --prompts-file or --ids is required".into(),
            ))
        }
    };

    let cfg = TextConfig::from_json(&art.components()["text"]["config"])?;
    eprintln!("backend {}", b.describe());
    let t0 = Instant::now();
    let w = TextWeights::load(b, &art, &cfg, |line| eprint!("\rloading {line}   "))?;
    eprintln!(
        "\nloaded {:.2} GB in {:.1}s",
        w.bytes as f64 / 1e9,
        t0.elapsed().as_secs_f64()
    );
    let model = Qwen35Text::new(cfg, w);
    let mut ws = model.workspace(b, chunk)?;

    for input in inputs {
        let (prompt, ids) = match input {
            Input::Prompt(p) => {
                let text = fe.template.render(
                    &json!([{"role": "user", "content": p}]),
                    None,
                    false,
                    true,
                )?;
                let ids = fe.encode(&text)?;
                (Some(p), ids)
            }
            Input::Ids(ids) => (None, ids),
        };
        let mut session = model.session(b, capacity)?;
        let t1 = Instant::now();
        let mut logits = model.forward(b, &mut session, &mut ws, &ids)?;
        let prefill = t1.elapsed().as_secs_f64();
        let mut out = Vec::new();
        let t2 = Instant::now();
        loop {
            let next = argmax(&logits, fe.public_vocab);
            if fe.is_eos(next) {
                break;
            }
            out.push(next);
            if out.len() >= max_tokens {
                break;
            }
            logits = model.forward(b, &mut session, &mut ws, &[next])?;
        }
        let decode = t2.elapsed().as_secs_f64();
        let text = fe.decode(&out)?;
        if jsonl {
            let line = json!({"prompt": prompt, "prompt_tokens": ids.len(), "ids": out, "text": text,
                              "prefill_s": prefill, "decode_tps": out.len() as f64 / decode.max(1e-9)});
            println!("{line}");
        } else {
            println!(
                "prompt    {} tokens, prefill {:.2}s ({:.1} tok/s)",
                ids.len(),
                prefill,
                ids.len() as f64 / prefill
            );
            println!(
                "decode    {} tokens in {:.2}s ({:.2} tok/s)",
                out.len(),
                decode,
                out.len() as f64 / decode.max(1e-9)
            );
            println!("ids       {out:?}");
            println!("text      {text:?}");
        }
    }
    Ok(ExitCode::SUCCESS)
}
