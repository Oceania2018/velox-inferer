//! Qwen3-ASR against the reference (docs/qwen3_asr.md#verification).
//!
//! Needs dumps from `scripts/qwen3_asr_golden.py` and the GGUF pair; skipped without them:
//!   VELOX_ASR_GOLDEN=<dir with one sub-directory per clip>
//!   VELOX_ASR_GGUF=<Qwen3-ASR-1.7B-Q8_0.gguf> VELOX_ASR_MMPROJ=<mmproj-Qwen3-ASR-1.7B-Q8_0.gguf>
//! The mel test needs only the dumps. The model tests run the CPU oracle (minutes in release):
//!   cargo test --release -p velox-models --test qwen3_asr_golden -- --ignored --nocapture

use std::path::{Path, PathBuf};

use velox_core::Backend;
use velox_cpu::CpuBackend;
use velox_models::qwen3_asr::mel::MelFrontend;
use velox_models::qwen3_asr::{argmax, Qwen3Asr};

fn clips() -> Option<Vec<PathBuf>> {
    let dir = std::env::var_os("VELOX_ASR_GOLDEN")?;
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join("meta.json").exists())
        .collect();
    v.sort();
    Some(v)
}

fn model_paths() -> Option<(PathBuf, PathBuf)> {
    Some((
        std::env::var_os("VELOX_ASR_GGUF")?.into(),
        std::env::var_os("VELOX_ASR_MMPROJ")?.into(),
    ))
}

fn f32s(p: &Path) -> Vec<f32> {
    std::fs::read(p)
        .unwrap_or_else(|e| panic!("{}: {e}", p.display()))
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn u32s(p: &Path) -> Vec<u32> {
    std::fs::read(p)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

fn pcm(dir: &Path) -> Vec<i16> {
    std::fs::read(dir.join("pcm.i16"))
        .unwrap()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| i16::from_le_bytes(*c))
        .collect()
}

/// Max |a - b| / max |b| (relative to the reference's scale).
fn rel_err(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "length");
    let scale = b.iter().map(|v| f64::from(v.abs())).fold(0.0, f64::max);
    let worst = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from((x - y).abs()))
        .fold(0.0, f64::max);
    worst / scale.max(1e-30)
}

fn meta(dir: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(dir.join("meta.json")).unwrap()).unwrap()
}

#[test]
fn mel_matches_reference() {
    let Some(clips) = clips() else {
        eprintln!("skipped: VELOX_ASR_GOLDEN not set");
        return;
    };
    let fe = MelFrontend::new();
    for dir in clips {
        let want = f32s(&dir.join("mel.f32"));
        let got = fe.compute(&pcm(&dir));
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("{}: mel max abs err {worst:.2e}", dir.display());
        // Log-mel units: the reference runs a float32 FFT.
        assert!(worst < 2e-4, "{}: mel max abs err {worst}", dir.display());
    }
}

#[test]
#[ignore = "runs the full model on the CPU oracle"]
fn model_matches_reference_on_cpu() {
    let (Some(clips), Some((text, mmproj))) = (clips(), model_paths()) else {
        eprintln!("skipped: VELOX_ASR_GOLDEN / VELOX_ASR_GGUF / VELOX_ASR_MMPROJ not set");
        return;
    };
    let b = CpuBackend;
    let t0 = std::time::Instant::now();
    let m = Qwen3Asr::load(&b, &text, &mmproj).expect("load");
    eprintln!(
        "loaded {:.2} GB in {:.1?}",
        m.bytes as f64 / 1e9,
        t0.elapsed()
    );
    for dir in clips {
        let name = dir.display().to_string();
        let meta = meta(&dir);
        let t0 = std::time::Instant::now();
        let (enc, n, trace) = m.encode_traced(&b, &pcm(&dir), true).expect("encode");
        let trace = trace.expect("trace");
        assert_eq!(
            n as u64,
            meta["audio_tokens"].as_u64().unwrap(),
            "{name}: audio tokens"
        );
        let conv = rel_err(&trace.conv, &f32s(&dir.join("conv.f32")));
        let mut enc_host = vec![0f32; n * m.cfg.dec.hidden];
        b.download(&enc, &mut enc_host).unwrap();
        let enc_want = f32s(&dir.join("enc.f32"));
        let enc_err = rel_err(&enc_host, &enc_want);
        eprintln!(
            "{name}: conv rel {conv:.2e}, encoder rel {enc_err:.2e} ({:.1?})",
            t0.elapsed()
        );
        // Where the error sits: per 104-row window, and the cosine of the worst row.
        let h = m.cfg.dec.hidden;
        for (w, rows) in (0..n).collect::<Vec<_>>().chunks(104).enumerate() {
            let (a, b) = (rows[0] * h, (rows[rows.len() - 1] + 1) * h);
            eprintln!(
                "{name}:   window {w} rows {}..{}: rel {:.2e}",
                rows[0],
                rows[rows.len() - 1] + 1,
                rel_err(&enc_host[a..b], &enc_want[a..b])
            );
        }
        let cos = |r: usize| {
            let (x, y) = (&enc_host[r * h..][..h], &enc_want[r * h..][..h]);
            let dot: f64 = x
                .iter()
                .zip(y)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum();
            let nx: f64 = x.iter().map(|a| f64::from(*a).powi(2)).sum::<f64>().sqrt();
            let ny: f64 = y.iter().map(|a| f64::from(*a).powi(2)).sum::<f64>().sqrt();
            dot / (nx * ny)
        };
        let worst_cos = (0..n).map(cos).fold(1.0, f64::min);
        eprintln!("{name}:   min row cosine {worst_cos:.8}");
        let lp = dir.join("layers.f32");
        if lp.exists() {
            let want = f32s(&lp);
            let d = m.cfg.enc.d;
            for (i, got) in trace.layers.iter().enumerate() {
                let w = &want[i * n * d..][..n * d];
                eprintln!("{name}:   block {i:2}: rel {:.2e}", rel_err(got, w));
            }
        }
        if std::env::var_os("VELOX_ASR_ENCODER_ONLY").is_some() {
            continue;
        }
        assert!(conv < 1e-5, "{name}: conv rel err {conv}");
        assert!(enc_err < 1e-4, "{name}: encoder rel err {enc_err}");

        let prompt = u32s(&dir.join("prompt.u32"));
        let want_tokens = u32s(&dir.join("tokens.u32"));
        let t0 = std::time::Instant::now();
        let (tokens, logits0) = m
            .generate(&b, &prompt, &enc, want_tokens.len() + 8)
            .expect("generate");
        let want_logits = f32s(&dir.join("logits0.f32"));
        let l_err = rel_err(&logits0, &want_logits);
        eprintln!(
            "{name}: logits0 rel {l_err:.2e}, argmax {} vs {}, {} tokens ({:.1?})",
            argmax(&logits0),
            argmax(&want_logits),
            tokens.len(),
            t0.elapsed()
        );
        assert!(l_err < 1e-4, "{name}: logits0 rel err {l_err}");
        assert_eq!(tokens, want_tokens, "{name}: greedy tokens");
    }
}
