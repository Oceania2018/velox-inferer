use super::*;
use serde_json::json;

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let p =
            std::env::temp_dir().join(format!("velox-weights-{tag}-{}", hex(&new_artifact_id())));
        std::fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A small single-file ninfer v3 artifact: two tensors and one resource, payload bytes i % 251.
fn write_ninfer(dir: &Path, mutate: impl FnOnce(&mut Value)) -> PathBuf {
    let payload_len = 768 + 9000;
    let mut d = json!({
        "components": {
            "text": {"config": {"architectures": ["Qwen3_5ForCausalLM"]}, "resources": {"tokenizer.json": "r.tok"}},
            "mtp": {"config": {"architectures": ["Qwen3_5MTP"]}, "target": "text"}
        },
        "objects": [
            {"id": "w.qk", "kind": "tensor", "shape": [8, 16], "format": "bf16", "layout": "contiguous_le_v1", "offset": 0, "bytes": 256},
            {"id": "w.norm", "kind": "tensor", "shape": [16], "format": "fp32", "layout": "contiguous_le_v1", "offset": 512, "bytes": 64},
            {"id": "r.tok", "kind": "resource", "encoding": "raw_bytes_v1", "offset": 768, "bytes": 9000}
        ],
        "bindings": {
            "text/q": {"parts": [{"object": "w.qk", "range": [0, 64]}]},
            "text/k": {"parts": [{"object": "w.qk", "range": [64, 128]}]},
            "text/final_norm": {"object": "w.norm"}
        },
        "uses": [{"parameter": "text/q", "input": "text/x", "activation_policy": "AllowA8"}],
        "files": [{"path": null, "payload_bytes": payload_len}],
        "metadata": {"name": "synthetic"}
    });
    mutate(&mut d);
    let json = serde_json::to_vec(&d).unwrap();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&NINFER_ENTRY_MAGIC);
    bytes.extend_from_slice(&(json.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&[7u8; 16]);
    bytes.extend_from_slice(&json);
    bytes.resize(align_up(bytes.len() as u64, PAGE) as usize, 0);
    bytes.extend((0..payload_len).map(|i| (i % 251) as u8));
    let p = dir.join("model.ninfer");
    std::fs::write(&p, bytes).unwrap();
    p
}

#[test]
fn opens_valid_ninfer() {
    let t = TempDir::new("open");
    let a = Artifact::open(write_ninfer(&t.0, |_| {})).unwrap();
    assert_eq!(a.container, Container::NinferV3);
    assert_eq!(a.objects.len(), 3);
    assert_eq!(a.bindings_len(), 3);
    assert_eq!(a.name(), Some("synthetic"));
    let tok = a.read_object("r.tok").unwrap();
    assert_eq!(tok[0], (768 % 251) as u8);
}

#[test]
fn rejects_structural_errors() {
    let cases: Vec<(&str, Box<dyn FnOnce(&mut Value)>)> = vec![
        (
            "overlap",
            Box::new(|d| d["objects"][1]["offset"] = json!(0)),
        ),
        (
            "unaligned tensor",
            Box::new(|d| d["objects"][1]["offset"] = json!(300)),
        ),
        (
            "part range",
            Box::new(|d| d["bindings"]["text/k"]["parts"][0]["range"] = json!([64, 129])),
        ),
        (
            "missing object",
            Box::new(|d| d["bindings"]["text/final_norm"]["object"] = json!("nope")),
        ),
        (
            "resource bound as tensor",
            Box::new(|d| d["bindings"]["text/final_norm"]["object"] = json!("r.tok")),
        ),
        ("unknown root", Box::new(|d| d["extra"] = json!(1))),
        (
            "velox field in ninfer",
            Box::new(|d| d["velox"] = json!({})),
        ),
        (
            "bad policy",
            Box::new(|d| d["uses"][0]["activation_policy"] = json!("A4Only")),
        ),
        (
            "dangling target",
            Box::new(|d| d["components"]["mtp"]["target"] = json!("vision")),
        ),
        (
            "length mismatch",
            Box::new(|d| d["files"][0]["payload_bytes"] = json!(9769)),
        ),
    ];
    for (name, mutate) in cases {
        let t = TempDir::new("bad");
        assert!(
            Artifact::open(write_ninfer(&t.0, mutate)).is_err(),
            "{name} was accepted"
        );
    }
}

#[test]
fn converts_to_multi_part_vi_and_back() {
    let t = TempDir::new("convert");
    let src = Artifact::open(write_ninfer(&t.0, |_| {})).unwrap();
    let vi = t.0.join("model.vi");
    // A tiny limit forces the payload across several files, splitting objects.
    convert(&src, &vi, Container::VeloxV1, 3 * PAGE, new_artifact_id()).unwrap();
    let a = Artifact::open(&vi).unwrap();
    assert_eq!(a.container, Container::VeloxV1);
    assert!(a.files.len() > 1, "expected parts, got {}", a.files.len());
    assert_eq!(a.directory["velox"]["source_container"], "ninfer v3");
    for o in &src.objects {
        assert_eq!(
            a.read_object(&o.id).unwrap(),
            src.read_object(&o.id).unwrap(),
            "{}",
            o.id
        );
    }
    // Round trip back to ninfer drops the velox record and stays readable.
    let back = t.0.join("again.ninfer");
    convert(
        &a,
        &back,
        Container::NinferV3,
        DEFAULT_PART_LIMIT,
        new_artifact_id(),
    )
    .unwrap();
    let b = Artifact::open(&back).unwrap();
    assert!(b.directory.get("velox").is_none());
    assert_eq!(
        b.read_object("w.qk").unwrap(),
        src.read_object("w.qk").unwrap()
    );
    assert!(
        convert(
            &a,
            &back,
            Container::NinferV3,
            DEFAULT_PART_LIMIT,
            new_artifact_id()
        )
        .is_err(),
        "must not overwrite"
    );
}

fn spec<'a>(
    pick: &'a dyn Fn(&str) -> bool,
    replace: &'a dyn Fn(&str) -> bool,
    suffix: &'a str,
) -> ComposeSpec<'a> {
    ComposeSpec {
        pick,
        replace,
        drop: &|_| false,
        drop_components: &[],
        suffix,
    }
}

#[test]
fn composes_alternate_encodings() {
    let t = TempDir::new("compose");
    let base = Artifact::open(write_ninfer(&t.0, |_| {})).unwrap();
    let alt_dir = t.0.join("alt");
    std::fs::create_dir_all(&alt_dir).unwrap();
    // The alternate: the same bindings, with q's use carrying an auxiliary object.
    let alt = Artifact::open(write_ninfer(&alt_dir, |d| {
        d["objects"].as_array_mut().unwrap().insert(
            1,
            json!({"id": "aux.div", "kind": "tensor", "shape": [1], "format": "fp32", "layout": "contiguous_le_v1", "offset": 256, "bytes": 4}),
        );
        d["uses"][0]["auxiliaries"] = json!({"activation_input_divisor": {"object": "aux.div"}});
    }))
    .unwrap();
    let out = t.0.join("hybrid.vi");
    let r = compose_alternates(
        &base,
        &alt,
        &spec(&|n| n == "text/q" || n == "text/k", &|_| false, "prefill"),
        &out,
        DEFAULT_PART_LIMIT,
    )
    .unwrap();
    assert_eq!((r.bindings, r.objects), (2, 2));
    let c = Artifact::open(&out).unwrap();
    assert_eq!(c.container, Container::VeloxV1);
    // The base's payload and bindings are unchanged.
    let mut a = vec![0u8; 256];
    let mut b = vec![0u8; 256];
    base.read_range(0, &mut a).unwrap();
    c.read_range(0, &mut b).unwrap();
    assert_eq!(a, b);
    assert_eq!(
        c.directory["bindings"]["text/q"],
        base.directory["bindings"]["text/q"]
    );
    // The alternates name renamed copies holding the alternate's bytes.
    assert_eq!(
        c.directory["bindings"]["text/k@prefill"]["parts"][0]["object"],
        "prefill/w.qk"
    );
    assert_eq!(
        c.read_object("prefill/w.qk").unwrap(),
        alt.read_object("w.qk").unwrap()
    );
    assert_eq!(
        c.read_object("prefill/aux.div").unwrap(),
        alt.read_object("aux.div").unwrap()
    );
    let uses = c.directory["uses"].as_array().unwrap();
    let u = uses
        .iter()
        .find(|u| u["parameter"] == "text/q@prefill")
        .unwrap();
    assert_eq!(
        u["auxiliaries"]["activation_input_divisor"]["object"],
        "prefill/aux.div"
    );
    // Composing twice into the same file, or with a separator in the suffix, is refused.
    assert!(compose_alternates(
        &base,
        &alt,
        &spec(&|_| true, &|_| false, "prefill"),
        &out,
        DEFAULT_PART_LIMIT
    )
    .is_err());
    assert!(compose_alternates(
        &base,
        &alt,
        &spec(&|_| true, &|_| false, "a/b"),
        &t.0.join("x.vi"),
        DEFAULT_PART_LIMIT
    )
    .is_err());
}

#[test]
fn composes_replacements() {
    let t = TempDir::new("replace");
    let base = Artifact::open(write_ninfer(&t.0, |_| {})).unwrap();
    let alt_dir = t.0.join("alt");
    std::fs::create_dir_all(&alt_dir).unwrap();
    let alt = Artifact::open(write_ninfer(&alt_dir, |d| {
        d["uses"][0]["activation_policy"] = json!("AllowA4");
    }))
    .unwrap();
    let out = t.0.join("replaced.vi");
    // q switches encoding; k gets an alternate.
    compose_alternates(
        &base,
        &alt,
        &spec(
            &|n| n == "text/q" || n == "text/k",
            &|n| n == "text/q",
            "prefill",
        ),
        &out,
        DEFAULT_PART_LIMIT,
    )
    .unwrap();
    let c = Artifact::open(&out).unwrap();
    assert_eq!(
        c.directory["bindings"]["text/q"]["parts"][0]["object"],
        "prefill/w.qk"
    );
    assert!(c.directory["bindings"].get("text/q@prefill").is_none());
    assert!(c.directory["bindings"].get("text/k@prefill").is_some());
    let uses: Vec<&Value> = c.directory["uses"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|u| u["parameter"] == "text/q")
        .collect();
    assert_eq!(uses.len(), 1);
    assert_eq!(uses[0]["activation_policy"], "AllowA4");
}

#[test]
fn composes_without_dropped_or_replaced_objects() {
    let t = TempDir::new("prune");
    let base = Artifact::open(write_ninfer(&t.0, |_| {})).unwrap();
    let alt_dir = t.0.join("alt");
    std::fs::create_dir_all(&alt_dir).unwrap();
    let alt = Artifact::open(write_ninfer(&alt_dir, |_| {})).unwrap();
    let out = t.0.join("pruned.vi");
    // q and k switch to the alternate's copy of w.qk; final_norm and the MTP component go.
    let s = ComposeSpec {
        pick: &|n| n == "text/q" || n == "text/k",
        replace: &|_| true,
        drop: &|n| n == "text/final_norm",
        drop_components: &["mtp"],
        suffix: "alt",
    };
    let r = compose_alternates(&base, &alt, &s, &out, DEFAULT_PART_LIMIT).unwrap();
    assert_eq!((r.bindings, r.dropped, r.kept_base), (2, 1, 1)); // the tokenizer resource stays
    let c = Artifact::open(&out).unwrap();
    assert!(c.object("w.qk").is_none() && c.object("w.norm").is_none());
    assert!(c.components().get("mtp").is_none());
    assert_eq!(
        c.read_object("r.tok").unwrap(),
        base.read_object("r.tok").unwrap()
    );
    assert_eq!(
        c.read_object("alt/w.qk").unwrap(),
        alt.read_object("w.qk").unwrap()
    );
    assert!(c.payload_bytes() < base.payload_bytes());
}
