//! `velox` command line.

mod args;
mod bench;
mod bench_verify;
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
  bench-linear            Time one linear at a model shape with random stored weights
      --format <name>       nvfp4 | fp8 | bf16 | q4 | q8 | iq3 | ggml_<type> ... (default nvfp4)
      --n <rows> --k <cols> default 17408 x 5120 (Qwen3.8 MLP gate)
      --m <list>            token counts, comma separated (default 1,4,16,128,512)
      --act <a16|a8|a4>     activation policy (default a16); --divisor <f> for a4 (default 500)
      --iters <n>           timed iterations (default 20)
  bench-gdn               Time one gated_delta_net at a model head geometry with random inputs
      --k-heads <n> --v-heads <n> --k-dim <n> --v-dim <n>  default 16/48 heads, 128/128 (Qwen3.8)
      --tokens <list>       token counts, comma separated (default 1,8,64,512)
      --iters <n>           timed iterations (default 50)
  bench-attn              Time causal attention over a KV cache in each storage format
      --heads <n> --kv-heads <n> --head-dim <n>  default 24/4, 256 (Qwen3.8)
      --pos0 <n> (16384) --tokens <list> (1,8,16) --kv-format <list> (f32,bf16,fp8,nvfp4) --iters <n> (50)
  bench-verify            Time batched speculative verify rounds against one sequence at a time
      --artifact <path> --batch <n> (3) --window <n> (12) --context <n> (13000) --rounds <n> (40)
      [--keep <n>] [--kv-format nvfp4] [--prefill <n>] (mixed rounds) [--only batched]
  inspect                 Validate and summarise a weight file (.vi or .ninfer)
      --artifact <path> [--component <id>] (prints its config) [--bindings <prefix>] [--uses true]
  convert                 Convert between .ninfer v3 and .vi v1 (lossless)
  import-gguf             Qwen3.5/3.8 GGUF text weights + a donor artifact's config, tokenizer,
                          vision tower and DFlash2 drafter -> .vi (GGML blocks kept as stored)
      --gguf <path> --donor <.ninfer|.vi> --output <.vi>
      --input <path> --output <path> [--to vi|ninfer] [--part-limit <bytes>]
  generate                Greedy generation from token ids with a real artifact
      --artifact <path> (--prompt <text> | --ids <1,2,3>) [--chat-template <file>]
      [--max-tokens 64] [--chunk 256] [--capacity 4096] [--mtp N | --dflash N] (speculative)
      [--kv-format f32|bf16|fp8|nvfp4] (attention KV cache storage, default f32)
      [--host-weights <list>|none] (matrices kept in system RAM, default text/token_embedding)
      [--state-quant bf16,nvfp4 [--split-tail 256]] (resume from a recurrent state stored in
      each format vs exact: storage error, greedy divergence, logit deltas)
  serve                  OpenAI-compatible API (/v1/chat/completions, /v1/models, /metrics, /health)
      --artifact <path>     .vi or .ninfer weights
      --port <n>            default 8010   --host <addr> default 0.0.0.0
      --model-id <id>       default: artifact metadata name
      --chat-template <file> --max-context <n> (32768) --prefill-chunk <n> (512)
      --ctx-checkpoints <n> (8) --enable-thinking true|false (false)
      --force-tool-call on|off (off)  requests offering tools answer with a tool call only
      --stream-content on|off (on)    streaming requests get their text as it is generated
      --kv-format <fmt>     attention KV cache storage: f32 | bf16 | fp8 | nvfp4 (f32)
      --host-state-dtype <fmt>  context-store recurrent states: f32 | bf16 | nvfp4 (bf16)
      --mtp-drafts <n>      most MTP drafts per round, adapted to acceptance (7; 0 disables)
      --dflash-drafts <n>   DFlash2 drafts per block when the artifact has dflash2 (7; 0: use MTP)
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
        Some("bench-linear") => bench_linear(a),
        Some("bench-gdn") => bench_gdn(a),
        Some("bench-attn") => bench_attn(a),
        Some("bench-verify") => bench_verify::run(a),
        Some("inspect") => weights::inspect(a),
        Some("convert") => weights::convert(a),
        Some("import-gguf") => weights::import_gguf(a),
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

fn bench_linear(a: &Args) -> Result<ExitCode> {
    use velox_core::{ActPolicy, Error, WeightFormat};
    let format = match a.get("format").unwrap_or("nvfp4") {
        "nvfp4" => WeightFormat::Nvfp4,
        "fp8" => WeightFormat::Fp8RowBf16,
        "bf16" => WeightFormat::Bf16,
        "f32" => WeightFormat::F32,
        "q4" => WeightFormat::Q4G64,
        "q5" => WeightFormat::Q5G64,
        "q6" => WeightFormat::Q6G64,
        "q8" => WeightFormat::Q8G32,
        "iq3" => WeightFormat::Iq3Xxs,
        other => match velox_core::ggml::GgmlType::from_name(other) {
            Some(t) => WeightFormat::Ggml(t),
            None => return Err(Error::Config(format!("unknown --format {other}"))),
        },
    };
    let act = match a.get("act").unwrap_or("a16") {
        "a16" => ActPolicy::A16Only,
        "a8" => ActPolicy::AllowA8,
        "a4" => ActPolicy::AllowA4 {
            divisor: a.parse("divisor", 500.0)?,
        },
        other => return Err(Error::Config(format!("unknown --act {other}"))),
    };
    let ms = a
        .get("m")
        .unwrap_or("1,4,16,128,512")
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|_| Error::Config("--m takes comma-separated integers".into()))?;
    let cfg = bench::LinearConfig {
        format,
        n: a.parse("n", 17408)?,
        k: a.parse("k", 5120)?,
        ms,
        act,
        iters: a.parse("iters", 20)?,
    };
    let sel = select::backend(a)?;
    match &sel {
        Selected::Cpu(b) => bench::run_linear(b, &cfg)?,
        #[cfg(feature = "cuda")]
        Selected::Cuda(b) => bench::run_linear(b, &cfg)?,
    }
    Ok(ExitCode::SUCCESS)
}

fn bench_gdn(a: &Args) -> Result<ExitCode> {
    let tokens = a
        .get("tokens")
        .unwrap_or("1,8,64,512")
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|_| velox_core::Error::Config("--tokens takes comma-separated integers".into()))?;
    let cfg = bench::GdnConfig {
        k_heads: a.parse("k-heads", 16)?,
        v_heads: a.parse("v-heads", 48)?,
        k_dim: a.parse("k-dim", 128)?,
        v_dim: a.parse("v-dim", 128)?,
        tokens,
        iters: a.parse("iters", 50)?,
    };
    let sel = select::backend(a)?;
    match &sel {
        Selected::Cpu(b) => bench::run_gdn(b, &cfg)?,
        #[cfg(feature = "cuda")]
        Selected::Cuda(b) => bench::run_gdn(b, &cfg)?,
    }
    Ok(ExitCode::SUCCESS)
}

fn bench_attn(a: &Args) -> Result<ExitCode> {
    let list = |key: &str, default: &str| -> Vec<String> {
        a.get(key)
            .unwrap_or(default)
            .split(',')
            .map(|s| s.trim().to_string())
            .collect()
    };
    let tokens = list("tokens", "1,8,16")
        .iter()
        .map(|s| s.parse::<usize>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|_| velox_core::Error::Config("--tokens takes comma-separated integers".into()))?;
    let formats = list("kv-format", "f32,bf16,fp8,nvfp4")
        .iter()
        .map(|s| velox_core::KvFormat::parse(s))
        .collect::<Result<Vec<_>>>()?;
    let cfg = bench::AttnConfig {
        heads: a.parse("heads", 24)?,
        kv_heads: a.parse("kv-heads", 4)?,
        head_dim: a.parse("head-dim", 256)?,
        pos0: a.parse("pos0", 16384)?,
        tokens,
        formats,
        iters: a.parse("iters", 50)?,
    };
    let sel = select::backend(a)?;
    match &sel {
        Selected::Cpu(b) => bench::run_attn(b, &cfg)?,
        #[cfg(feature = "cuda")]
        Selected::Cuda(b) => bench::run_attn(b, &cfg)?,
    }
    Ok(ExitCode::SUCCESS)
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
