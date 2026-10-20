//! `velox` command line.

mod args;
mod bench;
mod generate;
mod profile;
mod prompt_debug;
mod replay;
mod select;
mod serve;
mod weights;

use std::process::ExitCode;

use velox_core::plan::{self, ModelSpec, PlanRequest};
use velox_core::profile::CATALOG;
use velox_core::{Op, Result};

use args::Args;
use select::Selected;

const USAGE: &str = "\
velox - velox-inferer command line

USAGE: velox <command> [options]

COMMANDS:
  devices                 Detected GPUs, their profiles and the branch tuned for each
  profiles                Hardware catalog
  variant                 Variant compiled into this build and its native ops
  plan                    VRAM plan for a model on a profile
      --model <name>        tiny | llama-3.1-8b | qwen3-32b   (default llama-3.1-8b)
      --profile <id>        default: detected device or VELOX_PROFILE
      --weight-bits <f>     average stored bits per weight   (default 16)
      --kv-bits <n>         KV cache bits per element        (default 16)
      --sequences <n>       concurrent sequences             (default 1)
  selftest                Conformance suite against the CPU reference
  bench                   Synthetic decoder-stack benchmark
      --model <name>        shape preset (default llama-3.1-8b)
      --layers <n>          layers to instantiate (default 2)
      --context <n>         cached tokens before measuring (default 1024)
      --prefill <n>         prefill chunk to time (default 128)
      --iters <n>           timed iterations (default 20)
  inspect                 Validate and summarise a weight file (.vi or .ninfer)
      --artifact <path>
  convert                 Convert between .ninfer v3 and .vi v1 (lossless)
      --input <path> --output <path> [--to vi|ninfer] [--part-limit <bytes>]
  generate                Greedy generation from token ids with a real artifact
      --artifact <path> (--prompt <text> | --ids <1,2,3>) [--chat-template <file>]
      [--max-tokens 64] [--chunk 256] [--capacity 4096]
  serve                   OpenAI-compatible API (/v1/chat/completions, /v1/models, /metrics, /health)
      --artifact <path>     .vi or .ninfer weights
      --port <n>            default 8010   --host <addr> default 0.0.0.0
      --model-id <id>       default: artifact metadata name
      --chat-template <file> --max-context <n> (32768) --prefill-chunk <n> (512)
      --ctx-checkpoints <n> (8) --enable-thinking true|false (false)
  replay                  Replay a recorded agent session against an OpenAI-compatible server
      --session <dir>       directory with session.json and screenshots/
      --endpoint <url>      e.g. http://127.0.0.1:8010 (default)
      --out <file>          JSON report (default target/replay/<label>.json)
      --label <name>        report label (default: endpoint host:port)
      --steps <n>           replay only the first n steps
  replay-compare          Gate a candidate report against a baseline
      --baseline <file> --candidate <file> [--tolerance <pct>] (default 3)

COMMON OPTIONS:
  --backend <auto|cpu|cuda>  default auto (cuda when built with it and a GPU is visible)
  --device <index>           CUDA device (default 0)
  --profile <id>             override profile detection
";

fn main() -> ExitCode {
    let args = match Args::from_args(std::env::args().skip(1)) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    match run(&args) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(a: &Args) -> Result<ExitCode> {
    match a.command.as_deref() {
        Some("devices") => devices(),
        Some("profiles") => profiles(),
        Some("variant") => variant(a),
        Some("plan") => plan_cmd(a),
        Some("selftest") => selftest(a),
        Some("bench") => bench_cmd(a),
        Some("inspect") => weights::inspect(a),
        Some("convert") => weights::convert(a),
        Some("generate") => generate::run(a),
        Some("serve") => serve::run(a),
        Some("prompt-debug") => prompt_debug::run(a),
        Some("replay") => replay::run(a),
        Some("replay-compare") => replay::compare(a),
        None | Some("help") | Some("--help") | Some("-h") => {
            print!("{USAGE}");
            Ok(ExitCode::SUCCESS)
        }
        Some(other) => Err(velox_core::Error::Config(format!(
            "unknown command '{other}'"
        ))),
    }
}

fn devices() -> Result<ExitCode> {
    let found = select::cuda_devices();
    if found.is_empty() {
        println!(
            "no CUDA devices visible{}",
            if cfg!(feature = "cuda") {
                ""
            } else {
                " (built without --features cuda)"
            }
        );
    }
    let v = &velox_variant::INFO;
    for (idx, name, arch, sms, mem, prof) in found {
        println!(
            "cuda:{idx}  {name}  {arch}  {sms} SMs  {:.1} GiB",
            mem as f64 / (1u64 << 30) as f64
        );
        match prof {
            Some(p) => {
                println!("        profile {}  tuned branch: {}", p.id, p.tuned_branch.unwrap_or("-"));
                if let Some(advice) = v.advice(p) {
                    println!("        note: {advice}");
                }
            }
            None => println!("        profile: unknown card, portable kernels only (pass --profile to plan for it)"),
        }
    }
    Ok(ExitCode::SUCCESS)
}

fn profiles() -> Result<ExitCode> {
    println!(
        "{:<18} {:<8} {:<8} {:>5} {:>5} {:>7}  {:<12} name",
        "id", "arch", "form", "VRAM", "SMs", "GB/s", "branch"
    );
    for p in CATALOG {
        println!(
            "{:<18} {:<8} {:<8} {:>4}G {:>5} {:>7}  {:<12} {}",
            p.id,
            p.arch.to_string(),
            format!("{:?}", p.form_factor).to_lowercase(),
            p.vram_gib,
            p.sm_count,
            p.mem_bandwidth_gbs,
            p.tuned_branch.unwrap_or("(none)"),
            p.name
        );
    }
    Ok(ExitCode::SUCCESS)
}

fn variant(a: &Args) -> Result<ExitCode> {
    let v = &velox_variant::INFO;
    println!("variant  {} (branch {})", v.id, v.branch);
    println!("         {}", v.description);
    let profiles = if v.is_generic() {
        "any".to_string()
    } else {
        v.profiles.join(", ")
    };
    println!("tunes    {profiles}");
    let archs: Vec<_> = v.archs.iter().map(|a| a.to_string()).collect();
    println!(
        "archs    {}",
        if archs.is_empty() {
            "any (portable)".into()
        } else {
            archs.join(", ")
        }
    );
    for id in v.unknown_profiles() {
        println!("warning: variant names profile '{id}' which is not in the catalog");
    }
    let sel = select::backend(a)?;
    let native = sel.with(|b| b.native_ops());
    println!("backend  {}", sel.with(|b| b.describe()));
    for op in Op::ALL {
        let state = if native.contains(op) {
            "native"
        } else {
            "portable fallback"
        };
        println!("  {:<10} {state}", op.name());
    }
    Ok(ExitCode::SUCCESS)
}

fn plan_cmd(a: &Args) -> Result<ExitCode> {
    let model = ModelSpec::preset(a.get("model").unwrap_or("llama-3.1-8b"))?;
    let profile = select::profile(a)?.ok_or_else(|| {
        velox_core::Error::Config(
            "no device detected; pass --profile <id> (see `velox profiles`)".into(),
        )
    })?;
    let req = PlanRequest {
        weight_bits: a.parse("weight-bits", 16.0)?,
        kv_bits: a.parse("kv-bits", 16)?,
        sequences: a.parse("sequences", 1)?,
    };
    let policy = velox_variant::plan_policy(profile);
    let p = plan::plan(profile, policy, &model, req)?;
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    println!(
        "model     {} ({:.2}B params, {} bits/weight, KV {} bits)",
        model.name,
        model.params() as f64 / 1e9,
        req.weight_bits,
        req.kv_bits
    );
    println!(
        "profile   {} ({} GiB, variant {})",
        profile.id,
        profile.vram_gib,
        velox_variant::INFO.id
    );
    println!("reserved  {:>7.2} GiB", gib(p.reserved));
    println!("weights   {:>7.2} GiB", gib(p.weights));
    println!(
        "workspace {:>7.2} GiB  ({} tokens)",
        gib(p.workspace),
        policy.workspace_tokens
    );
    println!(
        "kv budget {:>7.2} GiB  ({} KiB/token)",
        p.kv_budget as f64 / (1u64 << 30) as f64,
        p.kv_bytes_per_token / 1024
    );
    println!(
        "kv tokens {:>7}       context/sequence {} (model max {})",
        p.max_kv_tokens, p.context_per_sequence, model.max_context
    );
    if let Some(advice) = velox_variant::INFO.advice(profile) {
        println!("note      {advice}");
    }
    if p.fits {
        println!("fits      yes");
        Ok(ExitCode::SUCCESS)
    } else {
        println!(
            "fits      NO: less than {} tokens of context per sequence",
            plan::MIN_CONTEXT
        );
        Ok(ExitCode::FAILURE)
    }
}

fn selftest(a: &Args) -> Result<ExitCode> {
    let sel = select::backend(a)?;
    println!("backend {}", sel.with(|b| b.describe()));
    let results = match &sel {
        Selected::Cpu(b) => velox_conformance::run(b)?,
        #[cfg(feature = "cuda")]
        Selected::Cuda(b) => velox_conformance::run(b)?,
    };
    let mut failed = 0;
    for r in &results {
        let mark = if r.passed() { "ok  " } else { "FAIL" };
        failed += usize::from(!r.passed());
        println!(
            "{mark} {:<48} err {:.2e} (tol {:.0e})",
            r.name, r.max_err, r.tol
        );
    }
    println!("{} passed, {failed} failed", results.len() - failed);
    Ok(if failed == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

fn bench_cmd(a: &Args) -> Result<ExitCode> {
    let cfg = bench::Config {
        model: ModelSpec::preset(a.get("model").unwrap_or("llama-3.1-8b"))?,
        layers: a.parse("layers", 2)?,
        context: a.parse("context", 1024)?,
        prefill: a.parse("prefill", 128)?,
        iters: a.parse("iters", 20)?,
    };
    let sel = select::backend(a)?;
    match &sel {
        Selected::Cpu(b) => bench::run(b, &cfg)?,
        #[cfg(feature = "cuda")]
        Selected::Cuda(b) => bench::run(b, &cfg)?,
    }
    Ok(ExitCode::SUCCESS)
}
