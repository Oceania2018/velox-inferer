//! End-to-end Qwen4-Exp case: a small random model written as a llama.cpp GGUF (every tensor
//! type and layout the Flash-Next GGUFs use, PLE, a QSA layer whose budget the prompt outgrows),
//! loaded through the real loader on the oracle and on the backend under test. The backend runs
//! with an expert cache that changes placement between steps, which must not change any logits.

use velox_core::format::WeightFormat;
use velox_core::ggml::GgmlType;
use velox_core::rng::Rng;
use velox_core::{Backend, KvFormat, Result};
use velox_cpu::CpuBackend;
use velox_models::qwen4_exp::{load, ExpertCache, Qwen4Config, Qwen4Exp, Shards};

use crate::{max_err, random_weight, CaseResult};

/// Metadata values of the synthetic GGUF.
enum Val {
    U32(u32),
    F32(f32),
    Str(&'static str),
    ArrU32(Vec<u32>),
    ArrU64(Vec<u64>),
}

#[derive(Default)]
struct Writer {
    kv: Vec<(String, Val)>,
    /// (name, ggml dims, ggml type, data)
    tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)>,
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

impl Writer {
    fn meta(&mut self, key: &str, v: Val) {
        self.kv.push((format!("qwen4exp.{key}"), v));
    }

    fn tensor(&mut self, name: &str, dims: &[u64], ty: u32, data: Vec<u8>) {
        self.tensors.push((name.into(), dims.to_vec(), ty, data));
    }

    fn bytes(&self) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        b.extend_from_slice(&(self.kv.len() as u64).to_le_bytes());
        for (k, v) in &self.kv {
            put_str(&mut b, k);
            match v {
                Val::U32(x) => {
                    b.extend_from_slice(&4u32.to_le_bytes());
                    b.extend_from_slice(&x.to_le_bytes());
                }
                Val::F32(x) => {
                    b.extend_from_slice(&6u32.to_le_bytes());
                    b.extend_from_slice(&x.to_le_bytes());
                }
                Val::Str(s) => {
                    b.extend_from_slice(&8u32.to_le_bytes());
                    put_str(&mut b, s);
                }
                Val::ArrU32(a) => {
                    b.extend_from_slice(&9u32.to_le_bytes());
                    b.extend_from_slice(&4u32.to_le_bytes());
                    b.extend_from_slice(&(a.len() as u64).to_le_bytes());
                    a.iter().for_each(|x| b.extend_from_slice(&x.to_le_bytes()));
                }
                Val::ArrU64(a) => {
                    b.extend_from_slice(&9u32.to_le_bytes());
                    b.extend_from_slice(&10u32.to_le_bytes());
                    b.extend_from_slice(&(a.len() as u64).to_le_bytes());
                    a.iter().for_each(|x| b.extend_from_slice(&x.to_le_bytes()));
                }
            }
        }
        let mut offset = 0u64;
        for (name, dims, ty, data) in &self.tensors {
            put_str(&mut b, name);
            b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            dims.iter()
                .for_each(|d| b.extend_from_slice(&d.to_le_bytes()));
            b.extend_from_slice(&ty.to_le_bytes());
            b.extend_from_slice(&offset.to_le_bytes());
            offset += (data.len() as u64).div_ceil(32) * 32;
        }
        b.resize(b.len().div_ceil(32) * 32, 0);
        for (.., data) in &self.tensors {
            b.extend_from_slice(data);
            b.resize(b.len().div_ceil(32) * 32, 0);
        }
        b
    }
}

const H: usize = 64;
const VOCAB: usize = 50;
const EOS: u32 = 3;

/// The synthetic model: 4 layers (GDN x3, QSA), 8 experts of width 64, PLE on layer 1.
fn model_gguf(rng: &mut Rng) -> Vec<u8> {
    let mut w = Writer::default();
    w.kv.push(("general.architecture".into(), Val::Str("qwen4exp")));
    for (k, v) in [
        ("block_count", 4),
        ("embedding_length", H as u32),
        ("attention.head_count", 4),
        ("attention.head_count_kv", 2),
        ("attention.key_length", 32),
        ("attention.value_length", 32),
        ("rope.dimension_count", 8),
        ("expert_count", 8),
        ("expert_used_count", 2),
        ("expert_feed_forward_length", 64),
        ("expert_shared_feed_forward_length", 32),
        ("ssm.conv_kernel", 4),
        ("ssm.state_size", 16),
        ("ssm.group_count", 2),
        ("ssm.time_step_rank", 4),
        ("ssm.inner_size", 64),
        ("full_attention_interval", 4),
        ("hyper_connection.count", 4),
        ("hyper_connection.low_rank", 16),
        ("attention.indexer.head_count", 2),
        ("attention.indexer.key_length", 32),
        ("attention.indexer.top_k", 8),
        ("ple.ngram_size", 3),
        ("ple.heads_per_ngram", 1),
        ("ple.conv_kernel", 4),
        ("ple.eos_token_id", EOS),
        ("embedding_length_per_layer_input", 32),
    ] {
        w.meta(k, Val::U32(v));
    }
    w.meta("rope.freq_base", Val::F32(10000.0));
    w.meta("attention.layer_norm_rms_epsilon", Val::F32(1e-6));
    w.meta("attention.compress_ratios", Val::ArrU32(vec![0, 0, 0, 4]));
    w.meta("ple.layers", Val::ArrU32(vec![1]));
    w.meta(
        "ple.layer_multipliers",
        Val::ArrU64(vec![
            23_703_573_157_769,
            20_109_073_645_365,
            8_052_911_324_071,
        ]),
    );
    w.meta("ple.head_offsets", Val::ArrU64(vec![0, 53]));
    w.meta("ple.head_vocab_sizes", Val::ArrU64(vec![53, 59]));

    let q8 = WeightFormat::Ggml(GgmlType::Q8_0);
    let nl = WeightFormat::Ggml(GgmlType::Iq4Nl);
    let q2 = WeightFormat::Ggml(GgmlType::Q2_0);
    let id = |f: WeightFormat| match f {
        WeightFormat::Ggml(t) => t.gguf_id(),
        WeightFormat::Bf16 => 30,
        _ => 0,
    };
    // A [n, k] matrix (ggml dims [k, n]).
    let mat = |w: &mut Writer, rng: &mut Rng, name: &str, f: WeightFormat, n: usize, k: usize| {
        let data = random_weight(rng, f, n, k);
        w.tensor(name, &[k as u64, n as u64], id(f), data);
    };
    let f32s = |v: Vec<f32>| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let near_one = |rng: &mut Rng, n: usize| -> Vec<u8> {
        f32s(rng.vec(n, 0.1).into_iter().map(|x| 1.0 + x).collect())
    };
    let bf16 = WeightFormat::Bf16;
    mat(&mut w, rng, "token_embd.weight", q8, VOCAB, H);
    mat(&mut w, rng, "output.weight", q8, VOCAB, H);
    let wide = 4 * H;
    let hc = |w: &mut Writer, rng: &mut Rng, p: &str, inject: bool| {
        let norm = near_one(rng, wide);
        w.tensor(&format!("{p}_norm.weight"), &[H as u64, 4], 0, norm);
        for (s, n, k) in [("down", 16, wide), ("up", wide, 16)] {
            let data = random_weight(rng, bf16, n, k);
            w.tensor(&format!("{p}_{s}.weight"), &[k as u64, n as u64], 30, data);
        }
        if inject {
            let data = random_weight(rng, bf16, 4, wide);
            w.tensor(&format!("{p}_inject.weight"), &[wide as u64, 4], 30, data);
        }
    };
    hc(&mut w, rng, "output_hc", false);
    for i in 0..4 {
        let t = |s: &str| format!("blk.{i}.{s}");
        hc(&mut w, rng, &t("hc_attn"), true);
        hc(&mut w, rng, &t("hc_ffn"), true);
        if i < 3 {
            mat(&mut w, rng, &t("attn_qkv.weight"), q8, 128, H);
            mat(&mut w, rng, &t("attn_gate.weight"), q8, 64, H);
            mat(&mut w, rng, &t("ssm_alpha.weight"), bf16, 4, H);
            mat(&mut w, rng, &t("ssm_beta.weight"), bf16, 4, H);
            let a: Vec<f32> = rng.vec(4, 0.5).into_iter().map(|x| -1.0 - x).collect();
            w.tensor(&t("ssm_a"), &[4], 0, f32s(a));
            w.tensor(&t("ssm_dt.bias"), &[4], 0, f32s(rng.vec(4, 0.5)));
            w.tensor(
                &t("ssm_conv1d.weight"),
                &[4, 128],
                0,
                f32s(rng.vec(512, 0.5)),
            );
            w.tensor(&t("ssm_norm.weight"), &[16], 0, near_one(rng, 16));
            mat(&mut w, rng, &t("ssm_out.weight"), q8, H, 64);
        } else {
            mat(&mut w, rng, &t("attn_q.weight"), q8, 256, H);
            mat(&mut w, rng, &t("attn_k.weight"), q8, 64, H);
            mat(&mut w, rng, &t("attn_v.weight"), q8, 64, H);
            mat(&mut w, rng, &t("attn_output.weight"), q8, H, 128);
            w.tensor(&t("attn_q_norm.weight"), &[32], 0, near_one(rng, 32));
            w.tensor(&t("attn_k_norm.weight"), &[32], 0, near_one(rng, 32));
            mat(&mut w, rng, &t("indexer.q_proj.weight"), bf16, 64, H);
            mat(&mut w, rng, &t("indexer.k_proj.weight"), bf16, 32, H);
            w.tensor(&t("indexer.q_norm.weight"), &[32], 0, near_one(rng, 32));
            w.tensor(&t("indexer.k_norm.weight"), &[32], 0, near_one(rng, 32));
        }
        mat(&mut w, rng, &t("ffn_gate_inp.weight"), bf16, 8, H);
        let shexp = random_weight(rng, bf16, 1, H);
        w.tensor(&t("ffn_gate_inp_shexp.weight"), &[H as u64], 30, shexp);
        for (name, f, rows, k) in [
            ("ffn_gate_exps", q8, 64, H),
            ("ffn_up_exps", q8, 64, H),
            ("ffn_down_exps", q2, H, 64),
        ] {
            let data: Vec<u8> = (0..8)
                .flat_map(|_| random_weight(rng, f, rows, k))
                .collect();
            w.tensor(
                &t(&format!("{name}.weight")),
                &[k as u64, rows as u64, 8],
                id(f),
                data,
            );
        }
        mat(&mut w, rng, &t("ffn_gate_shexp.weight"), nl, 32, H);
        mat(&mut w, rng, &t("ffn_up_shexp.weight"), q8, 32, H);
        mat(&mut w, rng, &t("ffn_down_shexp.weight"), nl, H, 32);
        if i == 1 {
            mat(&mut w, rng, &t("ple_key.weight"), bf16, wide, H);
            mat(&mut w, rng, &t("ple_value.weight"), bf16, H, H);
            for n in ["ple_norm_key", "ple_norm_query", "ple_norm_conv"] {
                let v = near_one(rng, wide);
                w.tensor(&t(&format!("{n}.weight")), &[H as u64, 4], 0, v);
            }
            // F16 [channels][width], as the converter writes it.
            // Binary16 magnitudes 2^-3 .. 2^-1 with random signs and mantissas.
            let conv: Vec<u8> = (0..wide * 4)
                .flat_map(|_| {
                    let r = rng.next_u64();
                    let bits = ((r & 1) << 15) | ((12 + (r >> 1) % 2) << 10) | ((r >> 8) & 1023);
                    (bits as u16).to_le_bytes()
                })
                .collect();
            w.tensor(&t("ple_conv1d.weight"), &[4, wide as u64], 1, conv);
        }
    }
    mat(&mut w, rng, "per_layer_token_embd.weight", nl, 128, 32);
    w.bytes()
}

fn logits<B: Backend>(
    b: &B,
    m: &Qwen4Exp<B>,
    ws: &velox_models::qwen4_exp::Workspace<B>,
) -> Result<Vec<f32>> {
    m.logits(b, ws)
}

/// Runs the prompt in `chunks` on the oracle and on `b`; returns one result per step.
pub fn qwen4_stack<B: Backend>(b: &B, rng: &mut Rng) -> Result<Vec<CaseResult>> {
    let dir = std::env::temp_dir().join(format!("velox-qwen4-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| velox_core::Error::Config(e.to_string()))?;
    let path = dir.join("synthetic.gguf");
    std::fs::write(&path, model_gguf(rng)).map_err(|e| velox_core::Error::Config(e.to_string()))?;
    let shards = Shards::open(&path)?;
    let cfg = Qwen4Config::from_gguf(shards.first())?;
    let cpu = CpuBackend;
    let mc = Qwen4Exp::new(cfg.clone(), load(&cpu, &shards, &cfg, |_| {})?);
    let md = Qwen4Exp::new(cfg.clone(), load(b, &shards, &cfg, |_| {})?);
    let slot_bytes = md.w.max_record_bytes();
    let mut cc = ExpertCache::new(&cpu, 4, 8, slot_bytes, 0)?;
    // Fewer slots than experts: placements must keep changing.
    let mut cd = ExpertCache::new(b, 4, 8, slot_bytes, 11)?;
    let (cap, chunk) = (64, 8);
    let mut wc = mc.workspace(&cpu, &cc, chunk, cap)?;
    let mut wd = md.workspace(b, &cd, chunk, cap)?;
    let mut sc = mc.session(&cpu, cap, KvFormat::F32)?;
    let mut sd = md.session(b, cap, KvFormat::F32)?;
    // One EOS mid-prompt (PLE cuts its n-grams there); past position 11 QSA turns sparse.
    let mut ids: Vec<u32> = (0..30)
        .map(|_| (rng.next_u64() % VOCAB as u64) as u32)
        .collect();
    ids[9] = EOS;
    let mut out = Vec::new();
    let mut pos = 0;
    for (step, n) in [7usize, 8, 1, 1, 8, 5].into_iter().enumerate() {
        let part = &ids[pos..pos + n];
        mc.forward(&cpu, &mut sc, &mut wc, &mut cc, part)?;
        md.forward(b, &mut sd, &mut wd, &mut cd, part)?;
        let want = logits(&cpu, &mc, &wc)?;
        let got = logits(b, &md, &wd)?;
        // Relative L2 error of the logits: kernels that round activations as their ops report
        // (bf16, int8) drift by ~1e-2 through a random model, which the per-op cases check
        // exactly; a wrong layer or state is O(1).
        let (num, den) = got.iter().zip(&want).fold((0f64, 0f64), |(n, d), (&g, &w)| {
            let e = f64::from(g) - f64::from(w);
            (n + e * e, d + f64::from(w) * f64::from(w))
        });
        let rel = if num.is_nan() { f64::INFINITY } else { (num / den.max(1e-30)).sqrt() };
        out.push(CaseResult {
            name: format!("qwen4 synthetic step {step} ({n} tokens at {pos}), relative L2"),
            op: None,
            max_err: rel,
            tol: 3e-2,
        });
        pos += n;
        cd.update(b, &md.w, if step % 2 == 0 { usize::MAX } else { 3 })?;
    }
    // Token by token on the oracle reaches the same last logits as the chunked run.
    let mut s1 = mc.session(&cpu, cap, KvFormat::F32)?;
    for &t in &ids {
        mc.forward(&cpu, &mut s1, &mut wc, &mut cc, &[t])?;
    }
    let once = logits(&cpu, &mc, &wc)?;
    let chunked = {
        let mut s2 = mc.session(&cpu, cap, KvFormat::F32)?;
        for part in [
            &ids[..7],
            &ids[7..15],
            &ids[15..16],
            &ids[16..17],
            &ids[17..25],
            &ids[25..30],
        ] {
            mc.forward(&cpu, &mut s2, &mut wc, &mut cc, part)?;
        }
        logits(&cpu, &mc, &wc)?
    };
    out.push(CaseResult {
        name: "qwen4 synthetic oracle: token by token == chunked".into(),
        op: None,
        max_err: max_err(&once, &chunked),
        tol: 1e-4,
    });
    let _ = std::fs::remove_dir_all(&dir);
    Ok(out)
}
