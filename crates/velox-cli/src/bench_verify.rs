//! `velox bench-verify`: wall time of speculative verify rounds for a batch of sequences
//! (`Qwen35Text::verify_batch` + `commit_batched`) against the same windows verified one sequence
//! at a time (`verify` + `commit`), on synthetic sessions at a fixed context length (timing does
//! not depend on the cache contents).

use std::process::ExitCode;
use std::time::Instant;

use velox_core::{Backend, Error, KvFormat, Result};
use velox_models::qwen3_5::{DFlashModel, Qwen35Text, TextConfig, TextWeights, VERIFY_MAX};
use velox_weights::Artifact;

use crate::args::Args;
use crate::select::Selected;

pub fn run(a: &Args) -> Result<ExitCode> {
    match crate::select::backend(a)? {
        Selected::Cpu(b) => bench(&b, a),
        #[cfg(feature = "cuda")]
        Selected::Cuda(b) => bench(&b, a),
    }
}

fn bench<B: Backend>(b: &B, a: &Args) -> Result<ExitCode> {
    let path = a
        .get("artifact")
        .ok_or_else(|| Error::Config("--artifact is required".into()))?;
    let n: usize = a.parse("batch", 3)?;
    let m: usize = a.parse("window", 12)?;
    let context: usize = a.parse("context", 13000)?;
    let rounds: usize = a.parse("rounds", 40)?;
    let keep: usize = a.parse("keep", m / 2)?;
    let kv_format = KvFormat::parse(a.get("kv-format").unwrap_or("nvfp4"))?;
    if m == 0 || m > VERIFY_MAX || keep == 0 || keep > m || n == 0 {
        return Err(Error::Config(format!(
            "--window 1..={VERIFY_MAX}, --keep 1..=window, --batch >= 1"
        )));
    }
    let art = Artifact::open(path).map_err(|e| Error::Config(e.to_string()))?;
    let cfg = TextConfig::from_json(&art.components()["text"]["config"])?;
    eprintln!("backend {}", b.describe());
    let host = vec!["text/token_embedding".to_string()];
    let w = TextWeights::load_opts(b, &art, &cfg, false, &host, |l| eprint!("\rloading {l}   "))?;
    eprintln!("\nloaded {:.2} GB", w.bytes as f64 / 1e9);
    let model = Qwen35Text::new(cfg, w);
    let mut ws = model.workspace(b, 2048)?;
    // Verify windows capture DFlash features when the artifact drafts with DFlash2.
    if let Some(dm) = DFlashModel::load(b, &art, model.cfg.hidden)? {
        model.enable_capture(b, &mut ws, &dm.cfg.target_layers)?;
    }
    let capacity = context + 2 * rounds * m + 64;
    let mut sessions = Vec::with_capacity(n);
    for _ in 0..n {
        let mut s = model.session_with(b, capacity, kv_format)?;
        s.pos = context;
        sessions.push(s);
    }
    let mut bw = model.batch_workspace(b, n.max(2))?;
    let window: Vec<u32> = (1..=m as u32).collect();
    let limit = model.cfg.vocab;

    let batch_round = |sessions: &mut Vec<_>, ws: &mut _, bw: &mut _| -> Result<()> {
        let mut refs: Vec<&mut _> = sessions.iter_mut().collect();
        let windows: Vec<&[u32]> = vec![&window[..]; n];
        model.verify_batch(b, &mut refs, ws, bw, &windows, 8, limit)?;
        for s in refs {
            model.commit_batched(b, s, bw, keep)?;
        }
        Ok(())
    };
    let solo_round = |sessions: &mut Vec<_>, ws: &mut _| -> Result<()> {
        for s in sessions.iter_mut() {
            model.verify(b, s, ws, &window, 8, limit)?;
            model.commit(b, s, ws, keep)?;
        }
        Ok(())
    };
    let time = |f: &mut dyn FnMut() -> Result<()>| -> Result<f64> {
        for _ in 0..3 {
            f()?;
        }
        b.synchronize()?;
        let t = Instant::now();
        for _ in 0..rounds {
            f()?;
        }
        b.synchronize()?;
        Ok(t.elapsed().as_secs_f64() * 1e3 / rounds as f64)
    };
    // --bracket batched|solo: brackets two warm rounds of that mode for `ncu --profile-from-start off`.
    if let Some(mode) = a.get("bracket") {
        let mut f: Box<dyn FnMut() -> Result<()>> = match mode {
            "solo" => Box::new(|| solo_round(&mut sessions, &mut ws)),
            _ => Box::new(|| batch_round(&mut sessions, &mut ws, &mut bw)),
        };
        for _ in 0..3 {
            f()?;
        }
        b.synchronize()?;
        #[cfg(feature = "cuda")]
        velox_cuda::profiler(true)?;
        for _ in 0..2 {
            f()?;
        }
        b.synchronize()?;
        #[cfg(feature = "cuda")]
        velox_cuda::profiler(false)?;
        return Ok(ExitCode::SUCCESS);
    }
    let solo = time(&mut || solo_round(&mut sessions, &mut ws))?;
    let batched = if n > 1 {
        time(&mut || batch_round(&mut sessions, &mut ws, &mut bw))?
    } else {
        solo
    };
    println!(
        "{n} sequences x {m}-token windows at {context} keys (keep {keep}): batched {batched:.2} ms/round, \
         one at a time {solo:.2} ms/round ({:.2} ms per sequence), speedup {:.2}x",
        solo / n as f64,
        solo / batched
    );
    Ok(ExitCode::SUCCESS)
}
