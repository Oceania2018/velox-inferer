//! A small synthetic compressed-tensors checkpoint (one GDN and one attention layer, 4 experts
//! top-2, an MTP head with fused BF16 experts) converted and checked value by value.

use super::*;
use velox_core::format::{bf16_to_f32, decode_rows, e2m1_to_f32, e4m3fn_to_f32, WeightFormat};
use velox_cpu::CpuBackend;

use crate::qwen3_5::{Ffn, Mixer, TextWeights};

const H: u64 = 128;
const V: u64 = 256;
const E: u64 = 4;
const I: u64 = 128;
const D: u64 = 16; // head_dim, GDN key/value head size
const HEADS: u64 = 2;
const KV: u64 = 1;
const GK: u64 = 2; // GDN key heads
const GV: u64 = 4; // GDN value heads
const W: u64 = 4; // conv width
const QKV: u64 = 2 * GK * D + GV * D;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn byte(&mut self) -> u8 {
        (self.next() >> 24) as u8
    }
}

#[derive(Default)]
struct Ck {
    tensors: Vec<(String, &'static str, Vec<u64>, Vec<u8>)>,
}

impl Ck {
    fn add(&mut self, name: &str, dtype: &'static str, shape: &[u64], data: Vec<u8>) {
        self.tensors
            .push((name.to_string(), dtype, shape.to_vec(), data));
    }

    fn data(&self, name: &str) -> &[u8] {
        &self
            .tensors
            .iter()
            .find(|t| t.0 == name)
            .unwrap_or_else(|| panic!("no {name}"))
            .3
    }

    fn bf16(&mut self, r: &mut Rng, name: &str, shape: &[u64]) {
        let n: u64 = shape.iter().product();
        let data = (0..n)
            .flat_map(|_| {
                let f = (r.next() % 2001) as f32 / 1000.0 - 1.0;
                ((f.to_bits() >> 16) as u16).to_le_bytes()
            })
            .collect();
        self.add(name, "BF16", shape, data);
    }

    fn f32s(&mut self, name: &str, shape: &[u64], v: &[f32]) {
        let data = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.add(name, "F32", shape, data);
    }

    /// Channel-wise FP8: E4M3 codes (no NaN) and BF16 row scales.
    fn fp8(&mut self, r: &mut Rng, base: &str, n: u64, k: u64) {
        let codes = (0..n * k)
            .map(|_| {
                let b = r.byte();
                if b & 0x7f == 0x7f {
                    b ^ 1
                } else {
                    b
                }
            })
            .collect();
        self.add(&format!("{base}.weight"), "F8_E4M3", &[n, k], codes);
        self.bf16(r, &format!("{base}.weight_scale"), &[n, 1]);
    }

    fn nvfp4(&mut self, r: &mut Rng, base: &str, n: u64, k: u64, gs: f32, igs: f32) {
        let packed = (0..n * k / 2).map(|_| r.byte()).collect();
        let scales = (0..n * k / 16).map(|_| 0x30 + r.byte() % 0x18).collect();
        self.add(&format!("{base}.weight_packed"), "U8", &[n, k / 2], packed);
        self.add(
            &format!("{base}.weight_scale"),
            "F8_E4M3",
            &[n, k / 16],
            scales,
        );
        self.f32s(&format!("{base}.weight_global_scale"), &[1], &[gs]);
        self.f32s(&format!("{base}.input_global_scale"), &[1], &[igs]);
    }

    /// Writes the tensors over two safetensors shards plus an index.
    fn write(&self, dir: &Path) {
        let shards = [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ];
        let mut map = Map::new();
        for (s, file) in shards.iter().enumerate() {
            let mine: Vec<_> = self
                .tensors
                .iter()
                .enumerate()
                .filter(|(i, _)| i % 2 == s)
                .collect();
            let mut header = Map::new();
            header.insert("__metadata__".into(), json!({ "format": "pt" }));
            let mut data = Vec::new();
            for (_, (name, dtype, shape, bytes)) in &mine {
                let a = data.len();
                data.extend_from_slice(bytes);
                header.insert(
                    name.clone(),
                    json!({ "dtype": dtype, "shape": shape, "data_offsets": [a, data.len()] }),
                );
                map.insert(name.clone(), (*file).into());
            }
            let mut h = serde_json::to_vec(&Value::Object(header)).unwrap();
            while !h.len().is_multiple_of(8) {
                h.push(b' ');
            }
            let mut out = (h.len() as u64).to_le_bytes().to_vec();
            out.extend_from_slice(&h);
            out.extend_from_slice(&data);
            std::fs::write(dir.join(file), out).unwrap();
        }
        let index = json!({ "metadata": {}, "weight_map": map });
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_vec(&index).unwrap(),
        )
        .unwrap();
    }
}

fn expert_scale(layer: u64, e: u64, role: u64) -> f32 {
    100.0 + (layer * 100 + e * 10 + role) as f32
}

/// The checkpoint; layer 0's shared gate/up share a global scale (fused), layer 1's do not.
fn checkpoint(dir: &Path) -> Ck {
    let mut r = Rng(0x9e37_79b9_7f4a_7c15);
    let mut c = Ck::default();
    let m = "model.language_model";
    c.bf16(&mut r, &format!("{m}.embed_tokens.weight"), &[V, H]);
    c.bf16(&mut r, &format!("{m}.norm.weight"), &[H]);
    c.fp8(&mut r, "lm_head", V, H);
    c.bf16(&mut r, "model.visual.blocks.0.attn.qkv.weight", &[8, 8]);
    for l in 0..2u64 {
        let p = format!("{m}.layers.{l}");
        c.bf16(&mut r, &format!("{p}.input_layernorm.weight"), &[H]);
        c.bf16(
            &mut r,
            &format!("{p}.post_attention_layernorm.weight"),
            &[H],
        );
        if l == 0 {
            let a = format!("{p}.linear_attn");
            c.fp8(&mut r, &format!("{a}.in_proj_qkv"), QKV, H);
            c.fp8(&mut r, &format!("{a}.in_proj_z"), GV * D, H);
            c.fp8(&mut r, &format!("{a}.out_proj"), H, GV * D);
            c.bf16(&mut r, &format!("{a}.in_proj_a.weight"), &[GV, H]);
            c.bf16(&mut r, &format!("{a}.in_proj_b.weight"), &[GV, H]);
            c.bf16(&mut r, &format!("{a}.conv1d.weight"), &[QKV, 1, W]);
            c.f32s(&format!("{a}.A_log"), &[GV], &[0.1, 0.2, 0.3, 0.4]);
            c.bf16(&mut r, &format!("{a}.dt_bias"), &[GV]);
            c.bf16(&mut r, &format!("{a}.norm.weight"), &[D]);
        } else {
            let a = format!("{p}.self_attn");
            c.fp8(&mut r, &format!("{a}.q_proj"), 2 * HEADS * D, H);
            c.fp8(&mut r, &format!("{a}.k_proj"), KV * D, H);
            c.fp8(&mut r, &format!("{a}.v_proj"), KV * D, H);
            c.fp8(&mut r, &format!("{a}.o_proj"), H, HEADS * D);
            c.bf16(&mut r, &format!("{a}.q_norm.weight"), &[D]);
            c.bf16(&mut r, &format!("{a}.k_norm.weight"), &[D]);
            c.f32s(&format!("{a}.k_scale"), &[], &[1.0]);
            c.f32s(&format!("{a}.v_scale"), &[], &[1.0]);
        }
        c.bf16(&mut r, &format!("{p}.mlp.gate.weight"), &[E, H]);
        c.bf16(
            &mut r,
            &format!("{p}.mlp.shared_expert_gate.weight"),
            &[1, H],
        );
        for e in 0..E {
            for (j, (role, n, k)) in [("gate", I, H), ("up", I, H), ("down", H, I)]
                .iter()
                .enumerate()
            {
                let base = format!("{p}.mlp.experts.{e}.{role}_proj");
                c.nvfp4(
                    &mut r,
                    &base,
                    *n,
                    *k,
                    expert_scale(l, e, j as u64),
                    50.0 + e as f32,
                );
            }
        }
        let s = format!("{p}.mlp.shared_expert");
        let up_gs = if l == 0 { 7.0 } else { 9.0 };
        c.nvfp4(&mut r, &format!("{s}.gate_proj"), I, H, 7.0, 3.0);
        c.nvfp4(&mut r, &format!("{s}.up_proj"), I, H, up_gs, 3.0);
        c.nvfp4(&mut r, &format!("{s}.down_proj"), H, I, 11.0, 4.0);
    }
    // MTP: BF16 throughout, fused experts.
    c.bf16(&mut r, "mtp.fc.weight", &[H, 2 * H]);
    for n in ["pre_fc_norm_embedding", "pre_fc_norm_hidden", "norm"] {
        c.bf16(&mut r, &format!("mtp.{n}.weight"), &[H]);
    }
    let p = "mtp.layers.0";
    c.bf16(&mut r, &format!("{p}.input_layernorm.weight"), &[H]);
    c.bf16(
        &mut r,
        &format!("{p}.post_attention_layernorm.weight"),
        &[H],
    );
    c.bf16(
        &mut r,
        &format!("{p}.self_attn.q_proj.weight"),
        &[2 * HEADS * D, H],
    );
    c.bf16(
        &mut r,
        &format!("{p}.self_attn.k_proj.weight"),
        &[KV * D, H],
    );
    c.bf16(
        &mut r,
        &format!("{p}.self_attn.v_proj.weight"),
        &[KV * D, H],
    );
    c.bf16(
        &mut r,
        &format!("{p}.self_attn.o_proj.weight"),
        &[H, HEADS * D],
    );
    c.bf16(&mut r, &format!("{p}.self_attn.q_norm.weight"), &[D]);
    c.bf16(&mut r, &format!("{p}.self_attn.k_norm.weight"), &[D]);
    c.bf16(&mut r, &format!("{p}.mlp.gate.weight"), &[E, H]);
    c.bf16(
        &mut r,
        &format!("{p}.mlp.shared_expert_gate.weight"),
        &[1, H],
    );
    c.bf16(
        &mut r,
        &format!("{p}.mlp.shared_expert.gate_proj.weight"),
        &[I, H],
    );
    c.bf16(
        &mut r,
        &format!("{p}.mlp.shared_expert.up_proj.weight"),
        &[I, H],
    );
    c.bf16(
        &mut r,
        &format!("{p}.mlp.shared_expert.down_proj.weight"),
        &[H, I],
    );
    c.bf16(
        &mut r,
        &format!("{p}.mlp.experts.gate_up_proj"),
        &[E, 2 * I, H],
    );
    c.bf16(&mut r, &format!("{p}.mlp.experts.down_proj"), &[E, H, I]);
    c.write(dir);

    let config = json!({
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": H, "vocab_size": V, "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "num_attention_heads": HEADS, "num_key_value_heads": KV, "head_dim": D,
            "rope_parameters": {"rope_theta": 1e7, "partial_rotary_factor": 0.25,
                                "mrope_interleaved": true, "mrope_section": [1, 1, 0]},
            "linear_num_key_heads": GK, "linear_key_head_dim": D,
            "linear_num_value_heads": GV, "linear_value_head_dim": D,
            "linear_conv_kernel_dim": W, "rms_norm_eps": 1e-6, "max_position_embeddings": 4096,
            "num_experts": E, "num_experts_per_tok": 2, "moe_intermediate_size": I,
            "shared_expert_intermediate_size": I, "tie_word_embeddings": false,
            "mtp_num_hidden_layers": 1
        },
        "vision_config": {"depth": 1},
        "quantization_config": {"quant_method": "compressed-tensors", "format": "mixed-precision"}
    });
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    std::fs::write(dir.join("tokenizer.json"), b"{\"tok\": 1}").unwrap();
    std::fs::write(
        dir.join("generation_config.json"),
        b"{\"eos_token_id\": [1]}",
    )
    .unwrap();
    std::fs::write(
        dir.join("tokenizer_config.json"),
        serde_json::to_vec(&json!({ "chat_template": "{{ messages }}" })).unwrap(),
    )
    .unwrap();
    c
}

/// (format, parent rows, k, decoded rows) of `binding`.
fn decoded(art: &Artifact, binding: &str) -> (WeightFormat, usize, usize, Vec<f32>) {
    let b = &art.directory["bindings"][binding];
    let (id, range) = match b["object"].as_str() {
        Some(id) => (id.to_string(), None),
        None => (
            b["parts"][0]["object"].as_str().unwrap().to_string(),
            Some((
                b["parts"][0]["range"][0].as_u64().unwrap(),
                b["parts"][0]["range"][1].as_u64().unwrap(),
            )),
        ),
    };
    let Some(velox_weights::ObjectKind::Tensor {
        shape,
        format,
        layout,
    }) = art.object(&id).map(|o| o.kind.clone())
    else {
        panic!("{binding}: not a tensor");
    };
    let fmt = WeightFormat::from_names(&format, &layout).unwrap();
    let (n, k) = if shape.len() == 2 {
        (shape[0] as usize, shape[1] as usize)
    } else {
        (1, shape.iter().product::<u64>() as usize)
    };
    let (r0, rows) = match range {
        None => (0, n),
        Some((a, e)) => ((a as usize) / k, ((e - a) as usize) / k),
    };
    let bytes = art.read_object(&id).unwrap();
    assert_eq!(bytes.len(), fmt.payload_bytes(n, k), "{binding}");
    (fmt, n, k, decode_rows(fmt, &bytes, n, k, r0, rows).unwrap())
}

fn object_of(art: &Artifact, binding: &str) -> String {
    let b = &art.directory["bindings"][binding];
    b["object"]
        .as_str()
        .or_else(|| b["parts"][0]["object"].as_str())
        .unwrap()
        .to_string()
}

fn bf16_values(raw: &[u8]) -> Vec<f32> {
    raw.as_chunks::<2>()
        .0
        .iter()
        .map(|c| bf16_to_f32(u16::from_le_bytes(*c)))
        .collect()
}

/// Dequantised rows `rows` of a compressed-tensors FP8 linear.
fn fp8_ref(c: &Ck, base: &str, k: usize, rows: impl Iterator<Item = usize>) -> Vec<f32> {
    let (w, s) = (
        c.data(&format!("{base}.weight")),
        bf16_values(c.data(&format!("{base}.weight_scale"))),
    );
    rows.flat_map(|r| {
        (0..k)
            .map(|j| e4m3fn_to_f32(w[r * k + j]) * s[r])
            .collect::<Vec<_>>()
    })
    .collect()
}

/// Dequantised compressed-tensors NVFP4 linear `[n, k]`.
fn nvfp4_ref(c: &Ck, base: &str, n: usize, k: usize) -> Vec<f32> {
    let p = c.data(&format!("{base}.weight_packed"));
    let s = c.data(&format!("{base}.weight_scale"));
    let gs = f32::from_le_bytes(
        c.data(&format!("{base}.weight_global_scale"))
            .try_into()
            .unwrap(),
    );
    let mut out = Vec::with_capacity(n * k);
    for r in 0..n {
        for j in 0..k {
            let b = p[(r * k + j) / 2];
            let code = if j % 2 == 0 { b & 0xf } else { b >> 4 };
            out.push(e2m1_to_f32(code) * e4m3fn_to_f32(s[r * k / 16 + j / 16]) / gs);
        }
    }
    out
}

fn policy(art: &Artifact, binding: &str) -> (String, Option<f32>) {
    let u = art.directory["uses"]
        .as_array()
        .unwrap()
        .iter()
        .find(|u| u["parameter"] == binding)
        .unwrap_or_else(|| panic!("no use of {binding}"));
    let d = u["auxiliaries"]["activation_input_divisor"]["object"]
        .as_str()
        .map(|id| f32::from_le_bytes(art.read_object(id).unwrap()[..4].try_into().unwrap()));
    (u["activation_policy"].as_str().unwrap().to_string(), d)
}

#[test]
fn compressed_tensors_checkpoint_imports_exactly() {
    let dir = std::env::temp_dir().join(format!(
        "velox-hf-{}",
        velox_weights::hex(&velox_weights::new_artifact_id())
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let c = checkpoint(&dir);
    let out = dir.join("model.vi");
    let report = import(&dir, &out, &mut |_| {}).unwrap();
    assert!(report.mtp);
    let art = Artifact::open(&out).unwrap();
    let m = "model.language_model";

    // Config and resources.
    let cfg = TextConfig::from_json(&art.components()["text"]["config"]).unwrap();
    assert_eq!(cfg.architecture(), MOE_ARCHITECTURE);
    assert!(!cfg.gdn_v_tiled);
    assert_eq!((cfg.rotary_dim, cfg.moe().unwrap().experts), (4, 4));
    let res = &art.components()["text"]["resources"];
    let read_res = |role: &str| art.read_object(res[role].as_str().unwrap()).unwrap();
    assert_eq!(read_res("chat_template.jinja"), b"{{ messages }}");
    assert_eq!(read_res("tokenizer.json"), b"{\"tok\": 1}");
    assert!(!art.directory["bindings"]
        .as_object()
        .unwrap()
        .keys()
        .any(|k| k.contains("visual")));

    // Routed NVFP4 expert: one object each, value = e2m1 * scale / global scale.
    for (e, role, n, k) in [(2, "down", H, I), (0, "gate", I, H), (3, "up", I, H)] {
        let binding = format!("text/layers/1/moe/experts/{e}/{role}");
        let (fmt, pn, _, got) = decoded(&art, &binding);
        assert_eq!((fmt, pn), (WeightFormat::Nvfp4, n as usize));
        let want = nvfp4_ref(
            &c,
            &format!("{m}.layers.1.mlp.experts.{e}.{role}_proj"),
            n as usize,
            k as usize,
        );
        assert_eq!(got, want, "{binding}");
        assert_eq!(
            policy(&art, &binding),
            ("AllowA4".into(), Some(50.0 + e as f32))
        );
    }
    assert_ne!(
        object_of(&art, "text/layers/0/moe/experts/0/gate"),
        object_of(&art, "text/layers/0/moe/experts/1/gate")
    );

    // Shared expert: gate | up fused when the global scales agree (layer 0), separate otherwise.
    for l in 0..2 {
        let s = format!("{m}.layers.{l}.mlp.shared_expert");
        let (g, u) = (
            format!("text/layers/{l}/moe/shared/gate"),
            format!("text/layers/{l}/moe/shared/up"),
        );
        assert_eq!(object_of(&art, &g) == object_of(&art, &u), l == 0);
        assert_eq!(
            decoded(&art, &g).3,
            nvfp4_ref(&c, &format!("{s}.gate_proj"), I as usize, H as usize)
        );
        assert_eq!(
            decoded(&art, &u).3,
            nvfp4_ref(&c, &format!("{s}.up_proj"), I as usize, H as usize)
        );
        assert_eq!(policy(&art, &u), ("AllowA4".into(), Some(3.0)));
    }

    // Router | shared gate: one BF16 object.
    let router = "text/layers/0/moe/router".to_string();
    assert_eq!(
        object_of(&art, &router),
        object_of(&art, "text/layers/0/moe/shared_score")
    );
    assert_eq!(
        decoded(&art, &router).3,
        bf16_values(c.data(&format!("{m}.layers.0.mlp.gate.weight")))
    );
    assert_eq!(
        decoded(&art, "text/layers/0/moe/shared_score").3,
        bf16_values(c.data(&format!("{m}.layers.0.mlp.shared_expert_gate.weight")))
    );

    // Attention: q_proj's per-head query / gate rows split, then k, v; FP8 = e4m3 * row scale.
    let (d, hk) = (D as usize, H as usize);
    let q = format!("{m}.layers.1.self_attn.q_proj");
    let rows_of = |gate: usize| {
        (0..HEADS as usize).flat_map(move |h| ((2 * h + gate) * d)..((2 * h + gate + 1) * d))
    };
    let (fmt, _, _, got) = decoded(&art, "text/layers/1/attention/query");
    assert_eq!(fmt, WeightFormat::Fp8RowBf16);
    assert_eq!(got, fp8_ref(&c, &q, hk, rows_of(0)));
    assert_eq!(
        decoded(&art, "text/layers/1/attention/gate").3,
        fp8_ref(&c, &q, hk, rows_of(1))
    );
    assert_eq!(
        decoded(&art, "text/layers/1/attention/value").3,
        fp8_ref(&c, &format!("{m}.layers.1.self_attn.v_proj"), hk, 0..d)
    );
    let parent = object_of(&art, "text/layers/1/attention/query");
    for r in ["gate", "key", "value"] {
        assert_eq!(
            object_of(&art, &format!("text/layers/1/attention/{r}")),
            parent
        );
    }
    assert_eq!(policy(&art, "text/layers/1/attention/key").0, "AllowA8");
    assert_eq!(
        decoded(&art, "text/output_head").3,
        fp8_ref(&c, "lm_head", hk, 0..V as usize)
    );

    // GDN: value rows and z unchanged (grouped heads), conv transposed to [width, channels].
    let a = format!("{m}.layers.0.linear_attn");
    let (qk, v) = ((2 * GK * D) as usize, (GV * D) as usize);
    assert_eq!(
        decoded(&art, "text/layers/0/gdn/value").3,
        fp8_ref(&c, &format!("{a}.in_proj_qkv"), hk, qk..qk + v)
    );
    assert_eq!(
        decoded(&art, "text/layers/0/gdn/z").3,
        fp8_ref(&c, &format!("{a}.in_proj_z"), hk, 0..v)
    );
    let conv_src = bf16_values(c.data(&format!("{a}.conv1d.weight")));
    let conv = decoded(&art, "text/layers/0/gdn/convolution").3;
    let ch = QKV as usize;
    for j in 0..W as usize {
        for x in 0..ch {
            assert_eq!(conv[j * ch + x], conv_src[x * W as usize + j]);
        }
    }
    assert_eq!(
        decoded(&art, "text/layers/0/gdn/a_log").3,
        vec![0.1, 0.2, 0.3, 0.4]
    );
    // Norms are stored as HF keeps them (zero-centred; velox applies 1 + w).
    assert_eq!(
        decoded(&art, "text/layers/0/input_norm").3,
        bf16_values(c.data(&format!("{m}.layers.0.input_layernorm.weight")))
    );

    // MTP experts: gate rows [0, I), up rows [I, 2I) of gate_up_proj[e].
    let gu = bf16_values(c.data("mtp.layers.0.mlp.experts.gate_up_proj"));
    let (iu, e) = (I as usize, 1usize);
    let at = |r: usize| e * 2 * iu * hk + r * hk;
    assert_eq!(
        decoded(&art, "mtp/layers/0/moe/experts/1/gate").3,
        gu[at(0)..at(iu)]
    );
    assert_eq!(
        decoded(&art, "mtp/layers/0/moe/experts/1/up").3,
        gu[at(iu)..at(2 * iu)]
    );
    let dn = bf16_values(c.data("mtp.layers.0.mlp.experts.down_proj"));
    assert_eq!(
        decoded(&art, "mtp/layers/0/moe/experts/1/down").3,
        dn[hk * iu..2 * hk * iu]
    );
    assert_eq!(policy(&art, "mtp/layers/0/moe/experts/1/down").0, "A16Only");

    // The loader takes the whole artifact, with the stacked projections as single views.
    let w = TextWeights::<CpuBackend>::load(&CpuBackend, &art, &cfg, |_| {}).unwrap();
    assert!(w.mtp.is_some());
    let Mixer::Gdn { qkvz, .. } = &w.layers[0].mixer else {
        panic!("layer 0 is GDN")
    };
    assert!(qkvz.is_some());
    let Mixer::Attention { fused, .. } = &w.layers[1].mixer else {
        panic!("layer 1 is attention")
    };
    assert!(fused.is_some());
    for (l, shared_fused) in [(0, true), (1, false)] {
        let Ffn::Moe(m) = &w.layers[l].ffn else {
            panic!("MoE layer")
        };
        assert!(m.router_score.is_some());
        assert_eq!(m.shared.gate_up.is_some(), shared_fused);
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn text_config_takes_the_moe_architecture() {
    let v = text_config(&json!({
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": {
            "hidden_size": 64, "vocab_size": 100, "num_hidden_layers": 1,
            "layer_types": ["full_attention"], "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 16, "rope_parameters": {"rope_theta": 1e7, "partial_rotary_factor": 0.25},
            "linear_num_key_heads": 2, "linear_key_head_dim": 16, "linear_num_value_heads": 4,
            "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4, "rms_norm_eps": 1e-6,
            "max_position_embeddings": 4096, "num_experts": 8, "num_experts_per_tok": 2,
            "moe_intermediate_size": 32, "shared_expert_intermediate_size": 32,
            "intermediate_size": 1024
        }
    }))
    .unwrap();
    assert_eq!(v["architectures"][0], MOE_ARCHITECTURE);
    assert!(v.get("intermediate_size").is_none());
}
