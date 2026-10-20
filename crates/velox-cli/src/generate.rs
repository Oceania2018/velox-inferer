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
    let profile = a.get("profile-ops").is_some();
    match sel {
        Selected::Cpu(b) => generate(&b, a, &mut |_| {}),
        #[cfg(feature = "cuda")]
        Selected::Cuda(b) if profile => {
            let p = crate::profile::Profiled::new(b);
            generate(&p, a, &mut |phase| {
                eprintln!("[{phase}] {}", p.report());
                p.reset();
            })
        }
        #[cfg(feature = "cuda")]
        Selected::Cuda(b) => generate(&b, a, &mut |_| {}),
    }
}

enum Input {
    Prompt(String),
    Ids(Vec<u32>),
}

fn generate<B: Backend>(b: &B, a: &Args, on_phase: &mut dyn FnMut(&str)) -> Result<ExitCode> {
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
        (Some(p), _, _) => {
            // --filler-tokens N prepends ~N tokens of filler text to measure long contexts.
            let filler: usize = a.parse("filler-tokens", 0)?;
            let pad = "The quick brown fox jumps over the lazy dog. ".repeat(filler / 10);
            (vec![Input::Prompt(format!("{pad}{p}"))], false)
        }
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
        if a.get("spec-check").is_some() {
            spec_check(
                b,
                &model,
                &mut session,
                &mut ws,
                &ids,
                max_tokens,
                fe.public_vocab,
            )?;
            continue;
        }
        let t1 = Instant::now();
        let mut logits = model.forward(b, &mut session, &mut ws, &ids)?;
        let prefill = t1.elapsed().as_secs_f64();
        on_phase("prefill");
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
        on_phase("decode");
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

/// `--spec-check`: speculative verify/commit must reproduce sequential decoding. Decodes greedily
/// token by token, rewinds to the prefill checkpoint, then replays the same tokens through
/// `verify` windows of 4 with a deliberately wrong draft every third position (forcing
/// rollbacks), comparing each verified row's best candidate with the sequential step.
fn spec_check<B: Backend>(
    b: &B,
    model: &Qwen35Text<B>,
    s: &mut velox_models::qwen3_5::Session<B>,
    ws: &mut velox_models::qwen3_5::Workspace<B>,
    ids: &[u32],
    steps: usize,
    limit: usize,
) -> Result<()> {
    model.advance(b, s, ws, ids, None, &[])?;
    let cp = s.checkpoint(b)?;
    // Sequential reference: (token, logit) after each step.
    let mut want = Vec::new();
    let mut c = model.candidates(b, ws, 1, limit)?[0];
    for _ in 0..steps {
        want.push(c);
        model.advance(b, s, ws, &[c.0], None, &[])?;
        c = model.candidates(b, ws, 1, limit)?[0];
    }
    want.push(c);
    s.restore(b, &cp)?;
    // want[i] is the token generated at step i; the pending anchor starts as want[0].
    let (mut done, mut rounds, mut accepted_total, mut max_err, mut mismatches) =
        (0usize, 0, 0, 0f32, 0);
    while done < steps {
        let mut window = vec![want[done].0];
        for j in 1..4.min(steps - done) {
            let good = want[done + j].0;
            window.push(if (done + j) % 3 == 0 { good + 1 } else { good });
        }
        let rows = model.verify(b, s, ws, &window, 1, limit)?;
        // Row j predicts the token after window[j]; drafts are accepted while they match.
        let mut keep = 1;
        for j in 0..window.len() {
            let got = rows[j][0];
            let reference = want[done + j + 1];
            if got.0 != reference.0 {
                mismatches += 1;
            }
            max_err = max_err.max((got.1 - reference.1).abs() / (1.0 + reference.1.abs()));
            if j + 1 < window.len() && window[j + 1] == got.0 {
                keep += 1;
            } else {
                break;
            }
        }
        model.commit(b, s, ws, keep)?;
        done += keep;
        accepted_total += keep - 1;
        rounds += 1;
    }
    println!(
        "spec-check: {steps} steps in {rounds} rounds ({accepted_total} drafts accepted), \
         top-1 mismatches {mismatches}, max logit rel. err {max_err:.2e}"
    );
    Ok(())
}
