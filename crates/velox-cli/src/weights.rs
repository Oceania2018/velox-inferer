//! `velox inspect`, `velox convert`, `velox import-gguf` and `velox import-hf`.

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

pub fn import_gguf(a: &Args) -> Result<ExitCode> {
    let (gguf, donor) = (
        Path::new(required(a, "gguf")?),
        Path::new(required(a, "donor")?),
    );
    let output = Path::new(required(a, "output")?);
    let t0 = std::time::Instant::now();
    let r = velox_models::qwen3_5::gguf::import(gguf, donor, output, &mut |line| {
        eprintln!("[import] {line}")
    })?;
    println!("text weights from the GGUF ({:.2} GB):", gb(r.text_bytes));
    for (format, objects, bytes) in &r.text_formats {
        println!("  {format:<14} {objects:>4} objects {:>7.2} GB", gb(*bytes));
    }
    println!(
        "wrote {} ({:.2} GB, {:.2} GB from the donor) in {:.1}s; structure ok",
        output.display(),
        gb(r.payload_bytes),
        gb(r.donor_bytes),
        t0.elapsed().as_secs_f64()
    );
    Ok(ExitCode::SUCCESS)
}

pub fn import_hf(a: &Args) -> Result<ExitCode> {
    let dir = Path::new(required(a, "hf-dir")?);
    let output = Path::new(required(a, "output")?);
    let t0 = std::time::Instant::now();
    // `--dflash <dir>`: a DFlash2 drafter checkpoint for this target, as the dflash2 component.
    let dflash = a.get("dflash").map(Path::new);
    // `--dflash-nvfp4`: the drafter's matrices as NVFP4 (smaller and faster to draft with).
    let dflash_nvfp4 = a.get("dflash-nvfp4").is_some();
    // `--proposal-ids <file>`: little-endian i32 token ids for an MTP draft vocabulary head.
    let proposal = match a.get("proposal-ids") {
        Some(p) => {
            let raw =
                std::fs::read(p).map_err(|e| velox_core::Error::Config(format!("{p}: {e}")))?;
            if raw.is_empty() || raw.len() % 4 != 0 {
                return Err(velox_core::Error::Config(format!(
                    "{p}: expected little-endian i32 token ids"
                )));
            }
            Some(
                raw.chunks_exact(4)
                    .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u32)
                    .collect::<Vec<u32>>(),
            )
        }
        None => None,
    };
    let r = velox_models::qwen3_5::hf::import_with(
        dir,
        output,
        dflash,
        dflash_nvfp4,
        proposal.as_deref(),
        &mut |line| eprintln!("[import] {line}"),
    )?;
    for (format, objects, bytes) in &r.formats {
        println!("  {format:<22} {objects:>6} objects {:>7.2} GB", gb(*bytes));
    }
    println!(
        "wrote {} ({:.2} GB; {} objects, {} bindings, {} uses; MTP {}) in {:.1}s; structure ok",
        output.display(),
        gb(r.payload_bytes),
        r.objects,
        r.bindings,
        r.uses,
        if r.mtp { "yes" } else { "no" },
        t0.elapsed().as_secs_f64()
    );
    Ok(ExitCode::SUCCESS)
}

pub fn import_dspine(a: &Args) -> Result<ExitCode> {
    let dir = Path::new(required(a, "dir")?);
    let output = Path::new(required(a, "output")?);
    let quant = velox_models::qwen3_5::hf::DSpineQuant::parse(a.get("nvfp4").unwrap_or("mlp"))?;
    let t0 = std::time::Instant::now();
    let r = velox_models::qwen3_5::hf::import_dspine(dir, output, quant, &mut |line| {
        eprintln!("[import] {line}")
    })?;
    for (format, objects, bytes) in &r.formats {
        println!("  {format:<22} {objects:>6} objects {:>7.2} GB", gb(*bytes));
    }
    println!(
        "wrote {} ({:.2} GB; {} objects, {} bindings) in {:.1}s; structure ok",
        output.display(),
        gb(r.payload_bytes),
        r.objects,
        r.bindings,
        t0.elapsed().as_secs_f64()
    );
    Ok(ExitCode::SUCCESS)
}

/// Whether `name` matches `pattern` as a whole, `*` standing for any run of characters.
fn glob(pattern: &str, name: &str) -> bool {
    let mut parts = pattern.split('*');
    let first = parts.next().unwrap_or_default();
    let Some(mut rest) = name.strip_prefix(first) else {
        return false;
    };
    let parts: Vec<&str> = parts.collect();
    for (i, p) in parts.iter().enumerate() {
        if i + 1 == parts.len() {
            return rest.ends_with(p);
        }
        match rest.find(p) {
            Some(at) => rest = &rest[at + p.len()..],
            None => return false,
        }
    }
    rest.is_empty()
}

/// `compose`: `--base` plus `--alt`'s bindings whose names match one of the comma-separated
/// `--match` globs, as `<name>@<suffix>` alternate encodings (see
/// `velox_weights::compose_alternates`).
pub fn compose(a: &Args) -> Result<ExitCode> {
    let base = open(required(a, "base")?)?;
    let alt = open(required(a, "alt")?)?;
    let output = Path::new(required(a, "output")?);
    let suffix = a.get("suffix").unwrap_or("prefill");
    let patterns: Vec<&str> = required(a, "match")?
        .split(',')
        .filter(|s| !s.is_empty())
        .collect();
    // Bindings among the matched ones that replace the base's instead of adding an alternate.
    let replace: Vec<&str> = a
        .get("replace")
        .unwrap_or_default()
        .split(',')
        .filter(|s| !s.is_empty())
        .collect();
    // Bindings (globs) and components of the base to leave out (e.g. `mtp/*` and `mtp`).
    let list = |key: &str| -> Vec<String> {
        a.get(key)
            .unwrap_or_default()
            .split(',')
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect()
    };
    let (drop, drop_components) = (list("drop"), list("drop-components"));
    let drop_components: Vec<&str> = drop_components.iter().map(String::as_str).collect();
    let limit = a.parse("part-limit", velox_weights::DEFAULT_PART_LIMIT)?;
    let t0 = std::time::Instant::now();
    let spec = velox_weights::ComposeSpec {
        pick: &|name| patterns.iter().any(|p| glob(p, name)),
        replace: &|name| replace.iter().any(|p| glob(p, name)),
        drop: &|name| drop.iter().any(|p| glob(p, name)),
        drop_components: &drop_components,
        suffix,
    };
    let r = velox_weights::compose_alternates(&base, &alt, &spec, output, limit)
        .map_err(|e| Error::Backend(e.to_string()))?;
    let out = open(&output.to_string_lossy())?;
    println!(
        "wrote {} ({:.2} GB): {} bindings taken, {} dropped, {} base objects kept, {} objects ({:.2} GB) from {} in {:.1}s; structure ok",
        output.display(),
        gb(r.payload_bytes),
        r.bindings,
        r.dropped,
        r.kept_base,
        r.objects,
        gb(r.bytes),
        alt.path.display(),
        t0.elapsed().as_secs_f64()
    );
    debug_assert_eq!(out.payload_bytes(), r.payload_bytes);
    Ok(ExitCode::SUCCESS)
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

#[cfg(test)]
mod tests {
    use super::glob;

    #[test]
    fn globs_match_whole_names() {
        assert!(glob("text/layers/*/mlp/gate", "text/layers/12/mlp/gate"));
        assert!(!glob("text/layers/*/mlp/gate", "dflash2/layers/0/mlp/gate"));
        assert!(!glob(
            "text/layers/*/mlp/gate",
            "text/layers/1/mlp/gate@prefill"
        ));
        assert!(glob("a*b*c", "aXbYc") && !glob("a*b*c", "aXbYcd") && glob("abc", "abc"));
    }
}
