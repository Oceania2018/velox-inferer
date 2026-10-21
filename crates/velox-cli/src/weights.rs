//! `velox inspect` and `velox convert`.

use std::path::Path;
use std::process::ExitCode;

use velox_core::{Error, Result};
use velox_weights::{Artifact, Container, ObjectKind};

use crate::args::Args;

fn open(path: &str) -> Result<Artifact> {
    Artifact::open(path).map_err(|e| Error::Config(e.to_string()))
}

fn required<'a>(a: &'a Args, key: &str) -> Result<&'a str> {
    a.get(key)
        .ok_or_else(|| Error::Config(format!("--{key} is required")))
}

fn gb(b: u64) -> f64 {
    b as f64 / 1e9
}

pub fn inspect(a: &Args) -> Result<ExitCode> {
    let art = open(required(a, "artifact")?)?;
    println!(
        "container  {}  id {}",
        art.container,
        velox_weights::hex(&art.artifact_id)
    );
    println!("name       {}", art.name().unwrap_or("-"));
    println!(
        "payload    {:.2} GB in {} file(s)",
        gb(art.payload_bytes()),
        art.files.len()
    );
    for (id, c) in art.components() {
        let arch = c["config"]
            .get("architectures")
            .and_then(|v| v.get(0))
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        let target = c
            .get("target")
            .and_then(|t| t.as_str())
            .map(|t| format!(" -> {t}"))
            .unwrap_or_default();
        let res = c
            .get("resources")
            .and_then(|r| r.as_object())
            .map_or(0, |r| r.len());
        println!("component  {id:<8} {arch}{target}  ({res} resources)");
    }
    let tensors = art
        .objects
        .iter()
        .filter(|o| matches!(o.kind, ObjectKind::Tensor { .. }))
        .count();
    println!(
        "objects    {} ({tensors} tensors, {} resources), {} bindings, {} uses",
        art.objects.len(),
        art.objects.len() - tensors,
        art.bindings_len(),
        art.uses_len()
    );
    for (key, n, bytes) in art.format_summary() {
        println!("  {key:<48} {n:>5} objects {:>9.3} GB", gb(bytes));
    }
    println!("structure  ok");
    if let Some(id) = a.get("component") {
        let c = art
            .components()
            .get(id)
            .ok_or_else(|| Error::Config(format!("no component '{id}'")))?;
        println!(
            "{}",
            serde_json::to_string_pretty(c).map_err(|e| Error::Config(e.to_string()))?
        );
    }
    if a.parse("uses", false)? {
        print_uses(&art)?;
    }
    if let Some(prefix) = a.get("bindings") {
        // Where each logical parameter lives: object, or parts (object + element range).
        if let Some(b) = art.directory["bindings"].as_object() {
            for (name, v) in b.iter().filter(|(n, _)| n.starts_with(prefix)) {
                let id = v["object"]
                    .as_str()
                    .or_else(|| v["parts"][0]["object"].as_str());
                let kind = id
                    .and_then(|id| art.object(id))
                    .map(|o| match &o.kind {
                        ObjectKind::Tensor {
                            shape,
                            format,
                            layout,
                        } => format!("{format}/{layout} {shape:?}"),
                        ObjectKind::Resource { encoding } => format!("resource {encoding}"),
                    })
                    .unwrap_or_default();
                println!("binding  {name:<48} {v}  {kind}");
            }
        }
    }
    Ok(ExitCode::SUCCESS)
}

/// One line per use: parameter, its activation policy and the activation input divisor.
fn print_uses(art: &Artifact) -> Result<()> {
    let uses = art.directory["uses"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    for u in &uses {
        let param = u["parameter"].as_str().unwrap_or("?");
        let policy = u["activation_policy"].as_str().unwrap_or("-");
        let divisor = match u["auxiliaries"]["activation_input_divisor"]["object"].as_str() {
            Some(id) => {
                let b = art
                    .read_object(id)
                    .map_err(|e| Error::Config(e.to_string()))?;
                format!("{}", f32::from_le_bytes(b[..4].try_into().unwrap()))
            }
            None => "-".into(),
        };
        println!("use  {param:<56} {policy:<10} {divisor}");
    }
    Ok(())
}

pub fn convert(a: &Args) -> Result<ExitCode> {
    let src = open(required(a, "input")?)?;
    let output = Path::new(required(a, "output")?);
    let target = match a.get("to").unwrap_or("vi") {
        "vi" => Container::VeloxV1,
        "ninfer" => Container::NinferV3,
        other => {
            return Err(Error::Config(format!(
                "--to must be vi or ninfer, got '{other}'"
            )))
        }
    };
    let limit = a.parse("part-limit", velox_weights::DEFAULT_PART_LIMIT)?;
    let t0 = std::time::Instant::now();
    velox_weights::convert(
        &src,
        output,
        target,
        limit,
        velox_weights::new_artifact_id(),
    )
    .map_err(|e| Error::Backend(e.to_string()))?;
    let out = open(&output.to_string_lossy())?;
    println!(
        "wrote {} ({}, {:.2} GB, {} file(s)) in {:.1}s; structure ok",
        output.display(),
        out.container,
        gb(out.payload_bytes()),
        out.files.len(),
        t0.elapsed().as_secs_f64()
    );
    Ok(ExitCode::SUCCESS)
}
