//! `velox generate`: greedy generation with a real artifact, from token ids or chat prompts.
//! `--prompts-file` runs one prompt per line with a single model load and prints JSON lines, which
//! is how outputs are compared with another engine.

use std::process::ExitCode;
use std::time::Instant;

use serde_json::json;
use velox_core::{Backend, Error, KvFormat, Result};
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
    let kv_format = KvFormat::parse(a.get("kv-format").unwrap_or("f32"))?;
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
    // As `velox serve`: matrices in --host-weights stay in system RAM (default the token
    // embedding; `none` keeps everything on the device), and DFlash2 runs skip the MTP weights.
    let host: Vec<String> = match a.get("host-weights").unwrap_or("text/token_embedding") {
        "none" | "" => Vec::new(),
        list => list.split(',').map(|s| s.trim().to_string()).collect(),
    };
    let mtp = a.get("dflash").is_none();
    let w = TextWeights::load_opts(b, &art, &cfg, mtp, &host, |line| {
        eprint!("\rloading {line}   ")
    })?;
    eprintln!(
        "\nloaded {:.2} GB to the device and {:.2} GB to system RAM in {:.1}s",
        w.bytes as f64 / 1e9,
        w.host_bytes as f64 / 1e9,
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
        let mut session = model.session_with(b, capacity, kv_format)?;
        if let Some(n) = a.get("mtp") {
            let draft_len: usize = n
                .parse()
                .map_err(|_| Error::Config("--mtp takes a draft count".into()))?;
            mtp_generate(
                b,
                &model,
                &mut session,
                &mut ws,
                &ids,
                max_tokens,
                draft_len,
                chunk,
                &fe,
                on_phase,
            )?;
            continue;
        }
        if let Some(n) = a.get("dflash") {
            let drafts: usize = n
                .parse()
                .map_err(|_| Error::Config("--dflash takes a draft count".into()))?;
            dflash_generate(
                b,
                &art,
                &model,
                &mut session,
                &mut ws,
                &ids,
                max_tokens,
                drafts,
                chunk,
                &fe,
                on_phase,
            )?;
            continue;
        }
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

/// `--mtp N`: greedy speculative decoding with N MTP drafts per round. Prints the text, the
/// acceptance statistics and the decode rate (the output should match plain greedy decoding).
#[allow(clippy::too_many_arguments)]
fn mtp_generate<B: Backend>(
    b: &B,
    model: &Qwen35Text<B>,
    s: &mut velox_models::qwen3_5::Session<B>,
    ws: &mut velox_models::qwen3_5::Workspace<B>,
    ids: &[u32],
    max_tokens: usize,
    draft_len: usize,
    chunk: usize,
    fe: &Frontend,
    on_phase: &mut dyn FnMut(&str),
) -> Result<()> {
    use velox_engine::spec::MtpDrafter;
    let limit = fe.public_vocab;
    let mut dr = MtpDrafter::new(b, model, s.capacity, chunk, draft_len)?;
    let row0 = s.pos;
    let t0 = Instant::now();
    model.advance_rows(b, s, ws, ids, None, &[], &mut [], &mut |ws, base, t| {
        dr.prefill_chunk(b, model, ws, row0 + base, t, 0, None)
    })?;
    let mut c = model.candidates(b, ws, 1, limit)?[0].0;
    let prefill = t0.elapsed().as_secs_f64();
    on_phase("prefill");
    let t1 = Instant::now();
    let mut out = vec![c];
    let mut drafts = dr.draft(b, model, s, ws, 0, &[], c, limit)?;
    let (mut rounds, mut accepted, mut proposed) = (0usize, 0usize, 0usize);
    while out.len() < max_tokens && !fe.is_eos(c) {
        let mut window = vec![c];
        window.extend(drafts.iter().copied().take(max_tokens - out.len()));
        let p = s.pos;
        let rows = model.verify(b, s, ws, &window, 1, limit)?;
        let mut keep = 1;
        while keep < window.len() && rows[keep - 1][0].0 == window[keep] {
            keep += 1;
        }
        let next = rows[keep - 1][0].0;
        model.commit(b, s, ws, keep)?;
        dr.record(keep - 1, window.len() - 1);
        rounds += 1;
        proposed += window.len() - 1;
        accepted += keep - 1;
        let mut stop = false;
        for &tok in window[1..keep].iter().chain(std::iter::once(&next)) {
            out.push(tok);
            if fe.is_eos(tok) || out.len() >= max_tokens {
                stop = true;
                break;
            }
        }
        if stop {
            break;
        }
        c = next;
        drafts = dr.draft(b, model, s, ws, p, &window[1..keep], c, limit)?;
    }
    let decode = t1.elapsed().as_secs_f64();
    on_phase("decode");
    if out.last().is_some_and(|&t| fe.is_eos(t)) {
        out.pop();
    }
    println!("prompt    {} tokens, prefill {prefill:.2}s", ids.len());
    println!(
        "decode    {} tokens in {decode:.2}s ({:.2} tok/s), {rounds} rounds, accepted {accepted}/{proposed} drafts ({:.0}%), {:.2} tokens/round",
        out.len(),
        out.len() as f64 / decode.max(1e-9),
        100.0 * accepted as f64 / proposed.max(1) as f64,
        out.len() as f64 / rounds.max(1) as f64
    );
    println!("ids       {out:?}");
    println!("text      {:?}", fe.decode(&out)?);
    Ok(())
}

/// `--dflash N`: greedy decoding with DFlash2 block drafts (N per block), verified on the
/// target; the output must equal plain greedy decoding.
#[allow(clippy::too_many_arguments)]
fn dflash_generate<B: Backend>(
    b: &B,
    art: &Artifact,
    model: &Qwen35Text<B>,
    s: &mut velox_models::qwen3_5::Session<B>,
    ws: &mut velox_models::qwen3_5::Workspace<B>,
    ids: &[u32],
    max_tokens: usize,
    drafts_per_block: usize,
    chunk: usize,
    fe: &Frontend,
    on_phase: &mut dyn FnMut(&str),
) -> Result<()> {
    use velox_engine::spec::DFlashDrafter;
    use velox_models::qwen3_5::DFlashModel;
    let limit = fe.public_vocab;
    let dm = DFlashModel::load(b, art, model.cfg.hidden)?
        .ok_or_else(|| Error::Config("artifact has no dflash2 component".into()))?;
    model.enable_capture(b, ws, &dm.cfg.target_layers)?;
    let mut dr = DFlashDrafter::new(b, &dm, drafts_per_block)?;
    let mut dw = DFlashDrafter::workspace(b, &dm, model, chunk, drafts_per_block, 1)?;
    let row0 = s.pos;
    let t0 = Instant::now();
    model.advance_rows(b, s, ws, ids, None, &[], &mut [], &mut |ws, base, t| {
        dr.prefill_chunk(b, &dm, &mut dw, ws, row0 + base, t)
    })?;
    let mut c = model.candidates(b, ws, 1, limit)?[0].0;
    let prefill = t0.elapsed().as_secs_f64();
    on_phase("prefill");
    let t1 = Instant::now();
    let mut out = vec![c];
    let (mut rounds, mut accepted, mut proposed) = (0usize, 0usize, 0usize);
    let (mut row, mut kept) = (0usize, 0usize);
    let (mut draft_s, mut verify_s, mut commit_s) = (0.0, 0.0, 0.0);
    // VELOX_PROFILE_ROUNDS=a..b brackets those rounds for `ncu --profile-from-start off`.
    let profile_rounds = std::env::var("VELOX_PROFILE_ROUNDS").ok().and_then(|r| {
        let (a, b) = r.split_once("..")?;
        Some(a.parse::<usize>().ok()?..b.parse::<usize>().ok()?)
    });
    while out.len() < max_tokens && !fe.is_eos(c) {
        if let Some(r) = &profile_rounds {
            if rounds == r.start || rounds == r.end {
                b.synchronize()?;
                #[cfg(feature = "cuda")]
                velox_cuda::profiler(rounds == r.start)?;
            }
        }
        let td = Instant::now();
        let (drafts, _) = dr.draft(b, &dm, &mut dw, model, ws, row, kept, s.pos, c, limit, 0.0)?;
        draft_s += td.elapsed().as_secs_f64();
        let mut window = vec![c];
        window.extend(drafts.iter().copied().take(max_tokens - out.len()));
        let p = s.pos;
        let tv = Instant::now();
        let rows = model.verify(b, s, ws, &window, 1, limit)?;
        verify_s += tv.elapsed().as_secs_f64();
        let mut keep = 1;
        while keep < window.len() && rows[keep - 1][0].0 == window[keep] {
            keep += 1;
        }
        let next = rows[keep - 1][0].0;
        // Commit is timed as submitted, not waited for (like the server): the next draft's
        // submission overlaps it, and its GPU time lands in the next round's drafting.
        let tc = Instant::now();
        model.commit(b, s, ws, keep)?;
        commit_s += tc.elapsed().as_secs_f64();
        rounds += 1;
        proposed += window.len() - 1;
        accepted += keep - 1;
        dr.observe(keep - 1);
        let mut stop = false;
        for &tok in window[1..keep].iter().chain(std::iter::once(&next)) {
            out.push(tok);
            if fe.is_eos(tok) || out.len() >= max_tokens {
                stop = true;
                break;
            }
        }
        if stop {
            break;
        }
        c = next;
        (row, kept) = (p, keep);
    }
    let decode = t1.elapsed().as_secs_f64();
    on_phase("decode");
    if out.last().is_some_and(|&t| fe.is_eos(t)) {
        out.pop();
    }
    println!("prompt    {} tokens, prefill {prefill:.2}s", ids.len());
    println!(
        "decode    {} tokens in {decode:.2}s ({:.2} tok/s), {rounds} rounds, accepted {accepted}/{proposed} drafts ({:.0}%), {:.2} tokens/round, drafting {:.2} ms, verify {:.2} ms, commit {:.2} ms per round",
        out.len(),
        out.len() as f64 / decode.max(1e-9),
        100.0 * accepted as f64 / proposed.max(1) as f64,
        out.len() as f64 / rounds.max(1) as f64,
        draft_s * 1e3 / rounds.max(1) as f64,
        verify_s * 1e3 / rounds.max(1) as f64,
        commit_s * 1e3 / rounds.max(1) as f64
    );
    println!("ids       {out:?}");
    println!("text      {:?}", fe.decode(&out)?);
    Ok(())
}
