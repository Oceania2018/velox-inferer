//! `velox generate --gguf <first shard> --hf-dir <dir>`: greedy generation with a Qwen4-Exp
//! (Qwen3.8-Flash-Next) GGUF, routed experts offloaded to system RAM behind a device cache.

use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use serde_json::json;
use velox_core::{Backend, Error, KvFormat, Result};
use velox_frontend::Frontend;
use velox_models::qwen4_exp::{
    load, load_mtp, ExpertCache, Mtp, MtpSession, Qwen4Config, Qwen4Exp, Shards, Workspace,
    VERIFY_MAX,
};

use crate::args::Args;

pub fn generate<B: Backend>(b: &B, a: &Args, on_phase: &mut dyn FnMut(&str)) -> Result<ExitCode> {
    let gguf = a
        .get("gguf")
        .ok_or_else(|| Error::Config("--gguf is required".into()))?;
    let hf = a
        .get("hf-dir")
        .ok_or_else(|| Error::Config("--hf-dir (tokenizer, chat template) is required".into()))?;
    let max_tokens: usize = a.parse("max-tokens", 64)?;
    let chunk: usize = a.parse("chunk", 256)?;
    let capacity: usize = a.parse("capacity", 4096)?;
    let kv_format = KvFormat::parse(a.get("kv-format").unwrap_or("f32"))?;
    // Device memory left free after the expert slots (WDDM pages weights out near the limit).
    let margin_mib: usize = a.parse("vram-margin-mib", 768)?;
    let max_slots: usize = a.parse("expert-slots", usize::MAX)?;
    // Decode steps between expert cache updates, and experts moved per update at most.
    let interval: usize = a.parse("cache-interval", 8)?;
    let moves: usize = a.parse("cache-moves", 512)?;
    // Drafts per verify (the MTP layer chained on its own residual past the first).
    let n_drafts: usize = a.parse("mtp-drafts", 2)?;
    if n_drafts == 0 || n_drafts >= VERIFY_MAX {
        return Err(Error::Config(format!(
            "--mtp-drafts must be 1..{}",
            VERIFY_MAX - 1
        )));
    }
    let template = match a.get("chat-template") {
        Some(p) => {
            Some(std::fs::read_to_string(p).map_err(|e| Error::Config(format!("{p}: {e}")))?)
        }
        None => None,
    };
    let fe = Frontend::from_dir(Path::new(hf), template)?;
    let prompts: Vec<String> = match (a.get("prompt"), a.get("prompts-file")) {
        (Some(p), _) => vec![p.to_string()],
        (None, Some(f)) => std::fs::read_to_string(f)
            .map_err(|e| Error::Config(format!("{f}: {e}")))?
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(str::to_string)
            .collect(),
        _ => {
            return Err(Error::Config(
                "--prompt or --prompts-file is required".into(),
            ))
        }
    };
    let jsonl = a.get("prompts-file").is_some();
    // Log-probabilities of the best candidates for the first generated position (comparing with
    // another engine's logprobs at a divergence).
    let show_top: usize = a.parse("show-top", 0)?;
    // Text appended after the rendered prompt (e.g. the shared start of two engines' answers).
    let prefix = a.get("assistant-prefix").unwrap_or("").to_string();

    eprintln!("backend {}", b.describe());
    let t0 = Instant::now();
    let mut shards = Shards::open(Path::new(gguf))?;
    // An MTP head drafts one token per step for a two-token verify (speculative greedy).
    if let Some(m) = a.get("mtp") {
        shards.add(Path::new(m))?;
    }
    let cfg = Qwen4Config::from_gguf(shards.first())?;
    let w = load(b, &shards, &cfg, |line| eprint!("\rloading {line}   "))?;
    eprintln!(
        "\nloaded {:.2} GB to the device and {:.2} GB to system RAM in {:.1}s",
        w.device_bytes as f64 / 1e9,
        w.host_bytes as f64 / 1e9,
        t0.elapsed().as_secs_f64()
    );
    let (layers, experts, slot_bytes) = (cfg.layers.len(), cfg.experts, w.max_record_bytes());
    let mtp_w = match a.get("mtp") {
        Some(_) => Some(
            load_mtp(b, &shards, &cfg)?
                .ok_or_else(|| Error::Config("--mtp: no MTP head (blk.N.nextn.*) found".into()))?,
        ),
        None => None,
    };
    let mut model = Qwen4Exp::new(cfg, w);
    model.ple_threads = a.parse("ple-threads", model.ple_threads)?;
    let mut session = model.session(b, capacity, kv_format)?;
    let mut ws = {
        let probe = ExpertCache::new(b, layers, experts, slot_bytes, 0)?;
        model.workspace(b, &probe, chunk, session.capacity)?
    };
    let mut mtp = match mtp_w {
        Some(w) => {
            let m = model.mtp(b, w)?;
            let ms = model.mtp_session(b, &m, session.capacity, kv_format, ws.max_tokens)?;
            Some((m, ms))
        }
        None => None,
    };
    b.synchronize()?;
    let free = b.mem_info().map_or(0, |(free, _)| free);
    let slots = (free.saturating_sub(margin_mib << 20) / slot_bytes).min(max_slots);
    let mut cache = ExpertCache::new(b, layers, experts, slot_bytes, slots)?;
    cache.decay = a.parse("cache-decay", cache.decay)?;
    // Per prompt, the trace index of its prefill update and its end.
    let trace_path = a.get("expert-trace");
    if trace_path.is_some() {
        cache.trace = Some(Vec::new());
    }
    let mut trace_spans = Vec::new();
    eprintln!(
        "expert cache: {slots} slots of {:.2} MiB ({:.2} GB, {:.1}% of {} experts)",
        slot_bytes as f64 / (1 << 20) as f64,
        (slots * slot_bytes) as f64 / 1e9,
        100.0 * slots as f64 / (layers * experts) as f64,
        layers * experts
    );

    let limit = fe.public_vocab;
    for (pi, p) in prompts.iter().enumerate() {
        if pi > 0 {
            model.reset(b, &mut session)?;
        }
        let text =
            fe.template
                .render(&json!([{"role": "user", "content": p}]), None, false, true)?;
        let ids = fe.encode(&(text + &prefix))?;
        let t1 = Instant::now();
        let chunks: Vec<_> = ids.chunks(ws.max_tokens).collect();
        let mut pending: Option<(usize, usize)> = None;
        for (ci, part) in chunks.iter().enumerate() {
            let pos0 = session.pos;
            model.forward(b, &mut session, &mut ws, &mut cache, part)?;
            // MTP entry p pairs h_p with token p + 1; the last chunk's waits for the first
            // generated token.
            if let Some((m, ms)) = mtp.as_mut() {
                if ci + 1 < chunks.len() {
                    let toks: Vec<u32> = ids[pos0 + 1..=pos0 + part.len()].to_vec();
                    model.mtp_forward(b, m, ms, &mut ws, pos0, &toks)?;
                } else {
                    pending = Some((pos0, part.len()));
                }
            }
        }
        if show_top > 0 {
            let logits = model.logits(b, &ws)?;
            let max = logits[..limit]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let lse = max as f64
                + logits[..limit]
                    .iter()
                    .map(|&l| (f64::from(l - max)).exp())
                    .sum::<f64>()
                    .ln();
            for (id, l) in velox_core::backend::top_k_row(&logits[..limit], show_top) {
                eprintln!(
                    "top {:>8} {:>8.3} {:?}",
                    id,
                    f64::from(l) - lse,
                    fe.decode(&[id])?
                );
            }
        }
        let mut next = model.argmax(b, &mut ws, limit)?;
        // The drafts for the tokens after `next`.
        let mut drafts = Vec::new();
        if let (Some((m, ms)), Some((pos0, n))) = (mtp.as_mut(), pending) {
            let mut toks: Vec<u32> = ids[pos0 + 1..pos0 + n].to_vec();
            toks.push(next);
            model.mtp_forward(b, m, ms, &mut ws, pos0, &toks)?;
            drafts = chain_drafts(b, &model, m, ms, &mut ws, session.pos, n_drafts, limit)?;
        }
        let prefill = t1.elapsed().as_secs_f64();
        // The prompt's routing decides the first placement.
        let filled = cache.update(b, &model.w, usize::MAX)?;
        let trace_start = cache.trace.as_ref().map_or(0, |t| t.len() - 1);
        let fill = t1.elapsed().as_secs_f64() - prefill;
        on_phase("prefill");
        let mut out = Vec::new();
        let (mut hits, mut misses, mut moved) = (0u64, 0u64, 0usize);
        let (mut rounds, mut accepted) = (0usize, 0usize);
        let mut since_update = 0;
        let t2 = Instant::now();
        while let Some((m, ms)) = mtp.as_mut() {
            if fe.is_eos(next) || out.len() >= max_tokens {
                break;
            }
            out.push(next);
            if out.len() >= max_tokens {
                break;
            }
            // Verify [next, drafts]: row j predicts the token after the window's j-th.
            let pos0 = session.pos;
            let mut window = vec![next];
            window.extend_from_slice(&drafts);
            model.verify(b, &mut session, &mut ws, &mut cache, &window)?;
            let best = model.argmax_rows(b, &ws, window.len(), limit)?;
            let n = drafts.iter().zip(&best).take_while(|(d, g)| d == g).count();
            rounds += 1;
            accepted += n;
            model.commit(b, &mut session, &window, n + 1)?;
            next = best[n];
            let mut stop = false;
            for &d in &drafts[..n] {
                if fe.is_eos(d) || out.len() >= max_tokens {
                    stop = true;
                    break;
                }
                out.push(d);
            }
            if stop || out.len() >= max_tokens {
                break;
            }
            // MTP entries for the kept positions: h of row j with the token after it.
            let mut toks = drafts[..n].to_vec();
            toks.push(next);
            model.mtp_forward(b, m, ms, &mut ws, pos0, &toks)?;
            drafts = chain_drafts(b, &model, m, ms, &mut ws, session.pos, n_drafts, limit)?;
            since_update += toks.len();
            if since_update >= interval {
                since_update = 0;
                let r = cache.update(b, &model.w, moves)?;
                (hits, misses, moved) = (hits + r.hits, misses + r.misses, moved + r.moves);
            }
        }
        while mtp.is_none() && !fe.is_eos(next) && out.len() < max_tokens {
            out.push(next);
            if out.len() >= max_tokens {
                break;
            }
            model.forward(b, &mut session, &mut ws, &mut cache, &[next])?;
            next = model.argmax(b, &mut ws, limit)?;
            if out.len() % interval == 0 {
                let r = cache.update(b, &model.w, moves)?;
                (hits, misses, moved) = (hits + r.hits, misses + r.misses, moved + r.moves);
            }
        }
        let decode = t2.elapsed().as_secs_f64();
        if cache.trace.is_some() {
            // Fold the tail in so the trace covers every generated token.
            cache.update(b, &model.w, 0)?;
            trace_spans.push((
                trace_start,
                cache.trace.as_ref().map_or(0, Vec::len),
                out.len(),
            ));
        }
        on_phase("decode");
        let hit_rate = hits as f64 / (hits + misses).max(1) as f64;
        let text = fe.decode(&out)?;
        if jsonl {
            let line = json!({"prompt": p, "prompt_tokens": ids.len(), "ids": out, "text": text,
                              "prefill_s": prefill, "decode_tps": out.len() as f64 / decode.max(1e-9),
                              "expert_hit_rate": hit_rate,
                              "mtp_accept": accepted as f64 / rounds.max(1) as f64 / n_drafts as f64,
                              "mtp_per_verify": accepted as f64 / rounds.max(1) as f64});
            println!("{line}");
        } else {
            println!(
                "prompt    {} tokens, prefill {:.2}s ({:.1} tok/s); first expert placement {} moves in {:.2}s (prompt hit rate {:.1}%)",
                ids.len(),
                prefill,
                ids.len() as f64 / prefill,
                filled.moves,
                fill,
                100.0 * filled.hit_rate()
            );
            println!(
                "decode    {} tokens in {:.2}s ({:.2} tok/s); expert hit rate {:.1}%, {moved} moves",
                out.len(),
                decode,
                out.len() as f64 / decode.max(1e-9),
                100.0 * hit_rate
            );
            if rounds > 0 {
                println!(
                    "mtp       {rounds} verifies, {accepted} drafts accepted ({:.2} per verify)",
                    accepted as f64 / rounds as f64
                );
            }
            println!("ids       {out:?}");
            println!("text      {text:?}");
        }
    }
    if let (Some(path), Some(t)) = (trace_path, &cache.trace) {
        // {"slots", "experts", "prompts": [[prefill update, end, tokens]], "updates": [[[l, e, n]]]}
        let v = json!({"slots": cache.capacity, "layers": layers, "experts": experts,
                       "interval": interval, "prompts": trace_spans, "updates": t});
        std::fs::write(path, v.to_string()).map_err(|e| Error::Config(format!("{path}: {e}")))?;
    }
    if let Some(path) = a.get("expert-stats") {
        // Fold the counts since the last update in (no moves).
        cache.update(b, &model.w, 0)?;
        // Selections per [layer][expert], for studying routing skew and cache policies.
        let v = serde_json::to_string(cache.totals()).map_err(|e| Error::Config(e.to_string()))?;
        std::fs::write(path, v).map_err(|e| Error::Config(format!("{path}: {e}")))?;
    }
    Ok(ExitCode::SUCCESS)
}

/// `n` greedy drafts after the last MTP entry (whose logits are in `ws`), the first at session
/// position `pos + 1`: each further one chains the MTP layer on its own residual.
#[allow(clippy::too_many_arguments)]
fn chain_drafts<B: Backend>(
    b: &B,
    model: &Qwen4Exp<B>,
    m: &mut Mtp<B>,
    ms: &mut MtpSession<B>,
    ws: &mut Workspace<B>,
    pos: usize,
    n: usize,
    limit: usize,
) -> Result<Vec<u32>> {
    let mut drafts = vec![model.argmax(b, ws, limit)?];
    for i in 1..n {
        model.mtp_chain(b, m, ms, ws, pos + i - 1, drafts[i - 1])?;
        drafts.push(model.argmax(b, ws, limit)?);
    }
    Ok(drafts)
}
