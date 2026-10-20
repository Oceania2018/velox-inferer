//! `velox serve`: OpenAI-compatible API over a loaded weight file.

use std::net::TcpListener;
use std::process::ExitCode;

use velox_core::{Error, Result};
use velox_serve::{Config, Server, UnimplementedEngine};
use velox_weights::Artifact;

use crate::args::Args;

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
    let max_concurrency: u32 = a.parse("max-concurrency", 1)?;
    let profile = crate::select::profile(a)?.map_or("unknown", |p| p.id);
    let cfg = Config {
        model_id: model_id.clone(),
        max_concurrency,
        info: vec![
            ("variant".into(), velox_variant::INFO.id.into()),
            ("branch".into(), velox_variant::INFO.branch.into()),
            ("profile".into(), profile.into()),
            ("container".into(), art.container.to_string()),
        ],
    };
    // No architecture has a velox execution path yet; the API, routing and metrics are live and
    // chat requests answer 501 with the reason instead of pretending.
    eprintln!(
        "note: no velox execution path for {architecture} yet; chat requests will answer 501"
    );
    let engine = Box::new(UnimplementedEngine {
        architecture: architecture.clone(),
    });
    let listener = TcpListener::bind((host, port))
        .map_err(|e| Error::Config(format!("bind {host}:{port}: {e}")))?;
    eprintln!(
        "listening on http://{host}:{port} | model {model_id} ({architecture}, {}) | variant {} | profile {profile} | lanes {max_concurrency}",
        art.container,
        velox_variant::INFO.id
    );
    Server::new(cfg, engine)
        .serve(listener)
        .map_err(|e| Error::Backend(e.to_string()))?;
    Ok(ExitCode::SUCCESS)
}
