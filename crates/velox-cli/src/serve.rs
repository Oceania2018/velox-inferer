//! `velox serve`: OpenAI-compatible API over a loaded weight file.

use std::net::TcpListener;
use std::process::ExitCode;
use std::time::Instant;

use velox_core::{Backend, Error, Result};
use velox_engine::{EngineConfig, QwenEngine};
use velox_frontend::Frontend;
use velox_models::qwen3_5::{Qwen35Text, Qwen35Vision, TextConfig, TextWeights};
use velox_serve::{ChatEngine, Config, Server, UnimplementedEngine};
use velox_weights::Artifact;

use crate::args::Args;
use crate::select::Selected;

fn build<B>(b: B, art: &Artifact, a: &Args) -> Result<Box<dyn ChatEngine>>
where
    B: Backend + Send + Sync + 'static,
    B::Buffer: Send + Sync + 'static,
{
    let template = match a.get("chat-template") {
        Some(p) => {
            Some(std::fs::read_to_string(p).map_err(|e| Error::Config(format!("{p}: {e}")))?)
        }
        None => None,
    };
    let fe = Frontend::from_artifact(art, template)?;
    let cfg = TextConfig::from_json(&art.components()["text"]["config"])?;
    let t0 = Instant::now();
    let w = TextWeights::load(&b, art, &cfg, |line| eprint!("\rloading {line}   "))?;
    eprintln!(
        "\nloaded {:.2} GB in {:.1}s on {}",
        w.bytes as f64 / 1e9,
        t0.elapsed().as_secs_f64(),
        b.describe()
    );
    let engine_cfg = EngineConfig {
        capacity: a.parse("max-context", 32768)?,
        chunk: a.parse("prefill-chunk", 512)?,
        default_max_tokens: a.parse("default-max-tokens", 4096)?,
        enable_thinking: a.get("enable-thinking").is_some_and(|v| v == "true"),
        max_checkpoints: a.parse("ctx-checkpoints", 8)?,
        scratch_capacity: a.parse("scratch-context", 8192)?,
        mtp_drafts: a.parse("mtp-drafts", 7)?,
    };
    let vision = if art.components().contains_key("vision")
        && a.get("vision").is_none_or(|v| v != "false")
    {
        let t1 = Instant::now();
        let v = Qwen35Vision::load(&b, art, cfg.hidden)?;
        eprintln!(
            "vision tower loaded ({:.2} GB) in {:.1}s",
            v.bytes as f64 / 1e9,
            t1.elapsed().as_secs_f64()
        );
        Some(v)
    } else {
        None
    };
    let model = Qwen35Text::new(cfg, w);
    Ok(Box::new(QwenEngine::new(b, model, vision, fe, engine_cfg)?))
}

pub fn run(a: &Args) -> Result<ExitCode> {
    let path = a
        .get("artifact")
        .ok_or_else(|| Error::Config("--artifact <file.vi|file.ninfer> is required".into()))?;
    let art = Artifact::open(path).map_err(|e| Error::Config(e.to_string()))?;
    let architecture = art.components()["text"]["config"]
        .get("architectures")
        .and_then(|v| v.get(0))
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();
    let model_id = a
        .get("model-id")
        .map(String::from)
        .or_else(|| art.name().map(String::from))
        .unwrap_or_else(|| "velox".into());
    let host = a.get("host").unwrap_or("0.0.0.0");
    let port: u16 = a.parse("port", 8010)?;
    let profile = crate::select::profile(a)?.map_or("unknown", |p| p.id);

    let engine: Box<dyn ChatEngine> = if architecture == "Qwen3_5ForCausalLM" {
        match crate::select::backend(a)? {
            Selected::Cpu(b) => build(b, &art, a)?,
            #[cfg(feature = "cuda")]
            Selected::Cuda(b) => build(b, &art, a)?,
        }
    } else {
        eprintln!(
            "note: no velox execution path for {architecture}; chat requests will answer 501"
        );
        Box::new(UnimplementedEngine {
            architecture: architecture.clone(),
        })
    };
    // One generation lane today; concurrent requests queue on it.
    let cfg = Config {
        model_id: model_id.clone(),
        max_concurrency: 1,
        info: vec![
            ("variant".into(), velox_variant::INFO.id.into()),
            ("branch".into(), velox_variant::INFO.branch.into()),
            ("profile".into(), profile.into()),
            ("container".into(), art.container.to_string()),
        ],
    };
    let listener = TcpListener::bind((host, port))
        .map_err(|e| Error::Config(format!("bind {host}:{port}: {e}")))?;
    eprintln!(
        "listening on http://{host}:{port} | model {model_id} ({architecture}, {}) | variant {} | profile {profile}",
        art.container,
        velox_variant::INFO.id
    );
    Server::new(cfg, engine)
        .serve(listener)
        .map_err(|e| Error::Backend(e.to_string()))?;
    Ok(ExitCode::SUCCESS)
}
