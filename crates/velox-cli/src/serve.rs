//! `velox serve`: OpenAI-compatible API over a loaded weight file.

use std::net::TcpListener;
use std::process::ExitCode;
use std::time::Instant;

use velox_core::{Backend, Error, KvFormat, Result};
use velox_engine::{EngineConfig, QwenEngine};
use velox_frontend::Frontend;
use velox_models::qwen3_5::{DFlashModel, Qwen35Text, Qwen35Vision, TextConfig, TextWeights};
use velox_serve::{ChatEngine, Config, Server, UnimplementedEngine};
use velox_weights::Artifact;

use crate::args::Args;
use crate::select::Selected;

/// "; device in use X GiB" (every process, as nvidia-smi counts), when the backend can tell.
fn in_use<B: Backend>(b: &B) -> String {
    b.device_in_use().map_or(String::new(), |used| {
        format!(
            "; device in use {:.2} GiB",
            used as f64 / (1u64 << 30) as f64
        )
    })
}

fn build<B>(b: B, art: &Artifact, a: &Args) -> Result<Box<dyn ChatEngine>>
where
    B: Backend + Send + Sync + 'static,
    B::Buffer: Send + Sync + 'static,
    B::Chunk: Send + Sync + 'static,
{
    let template = match a.get("chat-template") {
        Some(p) => {
            Some(std::fs::read_to_string(p).map_err(|e| Error::Config(format!("{p}: {e}")))?)
        }
        None => None,
    };
    let mut fe = Frontend::from_artifact(art, template)?;
    // --max-vision-tokens N (default 1024, ninfer NINFER_MAX_VISION_ITEM_TOKENS as deployed): images
    // with more merged tokens are downscaled to fit (one token covers 32 x 32 pixels), which bounds
    // the vision tower's device workspace.
    let max_vision: u64 = a.parse("max-vision-tokens", 1024)?;
    if let Some((lo, hi)) = fe.vision_pixels {
        fe.vision_pixels = Some((lo.min(max_vision * 1024), hi.min(max_vision * 1024)));
    }
    let cfg = TextConfig::from_json(&art.components()["text"]["config"])?;
    // --dflash-drafts N (default 7, ninfer --draft-tokens 7 as deployed): drafts per block. Above 7
    // the drafter switches to a longer block while recent rounds accept nearly everything; in
    // production that proposed more drafts without accepting more (4.03 vs ninfer 4.06 per round).
    let dflash_drafts: usize = a.parse("dflash-drafts", 7)?;
    let use_dflash = dflash_drafts > 0 && art.components().contains_key("dflash2");
    let t0 = Instant::now();
    // With DFlash2 the MTP predictor is never used: skip its weights.
    // Matrices kept in system RAM (ninfer NINFER_HOST_MAPPED_WEIGHTS): the token embedding is
    // only gathered a few rows per step. `--host-weights none` keeps everything on the device.
    let host: Vec<String> = match a.get("host-weights").unwrap_or("text/token_embedding") {
        "none" | "" => Vec::new(),
        list => list.split(',').map(|s| s.trim().to_string()).collect(),
    };
    let w = TextWeights::load_opts(&b, art, &cfg, !use_dflash, &host, |line| {
        eprint!("\rloading {line}   ")
    })?;
    eprintln!(
        "\nloaded {:.2} GB to the device and {:.2} GB to system RAM in {:.1}s on {}{}",
        w.bytes as f64 / 1e9,
        w.host_bytes as f64 / 1e9,
        t0.elapsed().as_secs_f64(),
        b.describe(),
        in_use(&b)
    );
    let engine_cfg = EngineConfig {
        capacity: a.parse("max-context", 32768)?,
        chunk: a.parse("prefill-chunk", 512)?,
        default_max_tokens: a.parse("default-max-tokens", 4096)?,
        enable_thinking: a.get("enable-thinking").is_some_and(|v| v == "true"),
        // --ctx-checkpoints N (ninfer's name): with the host store, superseded history points kept
        // per request chain (default 1, as ninfer is deployed; 0: unbounded); with
        // --ctx-checkpoint-store device, checkpoints per lane (default 8).
        max_checkpoints: a.parse("ctx-checkpoints", 8)?,
        chain_superseded: match a.parse("ctx-checkpoints", 1)? {
            0 => usize::MAX,
            n => n,
        },
        // --ctx-checkpoint-store host|device (default host: system RAM).
        host_checkpoints: a.get("ctx-checkpoint-store").is_none_or(|v| v != "device"),
        // --ctx-checkpoint-skip-media-tail on|off (default on).
        skip_media_tail: a
            .get("ctx-checkpoint-skip-media-tail")
            .is_none_or(|v| v != "off"),
        // --host-context-mib N (ninfer's name): system RAM for prefixes any request can resume.
        host_context_bytes: a.parse::<usize>("host-context-mib", 16384)? << 20,
        // --kv-paging on|off (default on): lanes' attention caches in virtual memory, backed as
        // they grow, with identical prefix chunks shared between lanes.
        kv_paging: a.get("kv-paging").is_none_or(|v| v != "off"),
        // --kv-capacity N|auto (ninfer's name; default auto): tokens of attention cache in the pool
        // the lanes share, created at startup; auto takes what device memory leaves after the
        // weights, workspaces, graph budget and --vram-headroom-mib.
        kv_capacity: match a.get("kv-capacity") {
            None | Some("auto") => None,
            Some(_) => Some(a.parse("kv-capacity", 0)?),
        },
        // --vram-headroom-mib N (default 1024, as ninfer): device memory left free at startup.
        vram_headroom: a.parse::<usize>("vram-headroom-mib", 1024)? << 20,
        // --vram-budget-mib N: device memory in use (every process, as nvidia-smi counts) once
        // started, graph budget included; the attention pool gets the rest. Overrides the headroom.
        vram_budget: match a.get("vram-budget-mib") {
            None => None,
            Some(_) => Some(a.parse::<usize>("vram-budget-mib", 0)? << 20),
        },
        // --graph-mib N (default 1024): device memory recorded CUDA graphs may hold.
        graph_budget: a.parse::<usize>("graph-mib", 1024)? << 20,
        // --ctx-lrfu-lambda F: LRFU decay per request of that store (0: LRU).
        lrfu_lambda: a.parse("ctx-lrfu-lambda", 0.1)?,
        // --disk-context-mib N (default 0: off) and --disk-context-dir DIR: NVMe tier below RAM.
        disk_context_bytes: a.parse::<u64>("disk-context-mib", 0)? << 20,
        disk_context_dir: match a.get("disk-context-dir") {
            Some(d) => d.into(),
            None => std::env::var_os("XDG_CACHE_HOME")
                .map(std::path::PathBuf::from)
                .or_else(|| {
                    std::env::var_os("HOME").map(|h| std::path::Path::new(&h).join(".cache"))
                })
                .unwrap_or_else(std::env::temp_dir)
                .join("velox/disk-context"),
        },
        // --max-concurrency N: lanes decoding at once (default 6, as ninfer on this machine).
        lanes: a.parse("max-concurrency", 6)?,
        mtp_drafts: a.parse("mtp-drafts", 7)?,
        dflash_drafts,
        kv_format: KvFormat::parse(a.get("kv-format").unwrap_or("f32"))?,
    };
    let vision = if art.components().contains_key("vision")
        && a.get("vision").is_none_or(|v| v != "false")
    {
        let t1 = Instant::now();
        let v = Qwen35Vision::load(&b, art, cfg.hidden)?;
        eprintln!(
            "vision tower loaded ({:.2} GB) in {:.1}s{}",
            v.bytes as f64 / 1e9,
            t1.elapsed().as_secs_f64(),
            in_use(&b)
        );
        Some(v)
    } else {
        None
    };
    let dflash = if use_dflash {
        let t2 = Instant::now();
        let d = DFlashModel::load(&b, art, cfg.hidden)?;
        if let Some(d) = &d {
            eprintln!(
                "dflash2 drafter loaded ({:.2} GB, {} drafts per block) in {:.1}s{}",
                d.bytes as f64 / 1e9,
                dflash_drafts,
                t2.elapsed().as_secs_f64(),
                in_use(&b)
            );
        }
        d
    } else {
        None
    };
    let model = Qwen35Text::new(cfg, w);
    Ok(Box::new(QwenEngine::new(
        b, model, vision, dflash, fe, engine_cfg,
    )?))
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
    // As many concurrent generations as engine lanes (more wait for one here).
    let cfg = Config {
        model_id: model_id.clone(),
        max_concurrency: a.parse("max-concurrency", 6u32)?,
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
