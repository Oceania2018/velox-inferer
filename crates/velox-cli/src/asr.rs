//! `velox asr --gguf <text.gguf> --mmproj <mmproj.gguf> --golden <dir>`: runs Qwen3-ASR on the
//! clips of a golden directory (scripts/qwen3_asr_golden.py) on the selected backend, checks the
//! encoder output, first-step logits and greedy tokens against the reference, and reports stage
//! timings. `--repeat N` reruns each clip (the first run includes one-off warm-up).

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use velox_core::{Backend, Error, Result};
use velox_models::qwen3_asr::{argmax, Qwen3Asr};

use crate::args::Args;
use crate::select::Selected;

pub fn run(a: &Args) -> Result<ExitCode> {
    let sel = crate::select::backend(a)?;
    let profile = a.get("profile-ops").is_some();
    match sel {
        Selected::Cpu(b) => asr(&b, a, &mut |_| {}),
        #[cfg(feature = "cuda")]
        Selected::Cuda(b) if profile => {
            let p = crate::profile::Profiled::new(b);
            asr(&p, a, &mut |phase| {
                eprintln!("[{phase}] {}", p.report());
                p.reset();
            })
        }
        #[cfg(feature = "cuda")]
        Selected::Cuda(b) => asr(&b, a, &mut |_| {}),
    }
}

fn read(p: &Path) -> Result<Vec<u8>> {
    std::fs::read(p).map_err(|e| Error::Config(format!("{}: {e}", p.display())))
}

fn f32s(p: &Path) -> Result<Vec<f32>> {
    Ok(read(p)?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

fn u32s(p: &Path) -> Result<Vec<u32>> {
    Ok(read(p)?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect())
}

/// Max |a - b| / max |b|.
fn rel_err(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    let scale = b.iter().map(|v| f64::from(v.abs())).fold(0.0, f64::max);
    let worst = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from((x - y).abs()))
        .fold(0.0, f64::max);
    worst / scale.max(1e-30)
}

fn asr<B: Backend>(b: &B, a: &Args, on_phase: &mut dyn FnMut(&str)) -> Result<ExitCode> {
    let need = |k: &str| {
        a.get(k)
            .map(PathBuf::from)
            .ok_or_else(|| Error::Config(format!("--{k} is required")))
    };
    let (text, mmproj, golden) = (need("gguf")?, need("mmproj")?, need("golden")?);
    let repeat: usize = a.parse("repeat", 2)?;
    // Per-stage tolerances: the CPU oracle meets 1e-4 against the f64 reference; device kernels
    // that round activations are allowed more, but greedy tokens must still match.
    let enc_tol: f64 = a.parse("enc-tol", 2e-2)?;
    let logit_tol: f64 = a.parse("logit-tol", 2e-2)?;
    println!("backend {}", b.describe());
    let t0 = Instant::now();
    let m = Qwen3Asr::load(b, &text, &mmproj)?;
    b.synchronize()?;
    println!(
        "loaded {:.2} GB in {:.1?}",
        m.bytes as f64 / 1e9,
        t0.elapsed()
    );
    on_phase("load");
    let mut clips: Vec<PathBuf> = std::fs::read_dir(&golden)
        .map_err(|e| Error::Config(format!("{}: {e}", golden.display())))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join("meta.json").exists())
        .collect();
    clips.sort();
    let mut ok = true;
    for dir in clips {
        let name = dir
            .file_name()
            .map_or_else(String::new, |n| n.to_string_lossy().into());
        let pcm: Vec<i16> = read(&dir.join("pcm.i16"))?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| i16::from_le_bytes(*c))
            .collect();
        let prompt = u32s(&dir.join("prompt.u32"))?;
        let want_tokens = u32s(&dir.join("tokens.u32"))?;
        let want_enc = f32s(&dir.join("enc.f32"))?;
        let want_logits = f32s(&dir.join("logits0.f32"))?;
        let audio_s = pcm.len() as f64 / 16_000.0;
        let fe = velox_models::qwen3_asr::mel::MelFrontend::new();
        let t = Instant::now();
        let _ = fe.compute(&pcm);
        println!("{name}: host mel {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
        for run in 0..repeat {
            let t = Instant::now();
            let (enc, n) = m.encode(b, &pcm)?;
            b.synchronize()?;
            let t_enc = t.elapsed();
            on_phase("encode");
            let t = Instant::now();
            let (tokens, logits0, t_prefill) =
                m.generate_timed(b, &prompt, &enc, want_tokens.len() + 8)?;
            let t_gen = t.elapsed();
            on_phase("generate");
            let mut enc_host = vec![0f32; n * m.cfg.dec.hidden];
            b.download(&enc, &mut enc_host)?;
            let (e_err, l_err) = (
                rel_err(&enc_host, &want_enc),
                rel_err(&logits0, &want_logits),
            );
            let same = tokens == want_tokens;
            let pass = same && e_err < enc_tol && l_err < logit_tol;
            ok &= pass;
            let decode = t_gen.saturating_sub(t_prefill);
            println!(
                "{name} run {run}: {audio_s:.2} s audio | encode {:.1} ms | prefill {} tok {:.1} ms | decode {} tok {:.1} ms ({:.2} ms/tok) | total {:.1} ms = {:.0}x realtime | enc rel {e_err:.1e} logits rel {l_err:.1e} argmax {}/{} tokens {} {}",
                t_enc.as_secs_f64() * 1e3,
                prompt.len(),
                t_prefill.as_secs_f64() * 1e3,
                tokens.len(),
                decode.as_secs_f64() * 1e3,
                decode.as_secs_f64() * 1e3 / tokens.len().max(1) as f64,
                (t_enc + t_gen).as_secs_f64() * 1e3,
                audio_s / (t_enc + t_gen).as_secs_f64(),
                argmax(&logits0),
                argmax(&want_logits),
                if same { "match" } else { "DIFFER" },
                if pass { "PASS" } else { "FAIL" },
            );
        }
    }
    Ok(if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}
