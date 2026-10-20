//! `velox prompt-debug`: renders a recorded session's requests exactly as `velox serve` would
//! (template, tokenizer, image expansion) without a GPU, and reports token counts, image tokens,
//! and the prefix each step shares with the previous one. Diagnoses prompt mismatches against
//! another engine and prefix-reuse misses.

use std::process::ExitCode;

use serde_json::Value;
use velox_core::{Error, Result};
use velox_engine::{build_prompt, normalize_request};
use velox_frontend::image::{decode_data_url, preprocess};
use velox_frontend::Frontend;
use velox_weights::Artifact;

use crate::args::Args;

pub fn run(a: &Args) -> Result<ExitCode> {
    let art = Artifact::open(
        a.get("artifact")
            .ok_or_else(|| Error::Config("--artifact is required".into()))?,
    )
    .map_err(|e| Error::Config(e.to_string()))?;
    let template = match a.get("chat-template") {
        Some(p) => {
            Some(std::fs::read_to_string(p).map_err(|e| Error::Config(format!("{p}: {e}")))?)
        }
        None => None,
    };
    let fe = Frontend::from_artifact(&art, template)?;
    let (min_px, max_px) = fe.vision_pixels.unwrap_or((0, u64::MAX));
    // Images arrive already downscaled by llm-inference-daemon (1 Mpx); apply the same cap.
    let image_cap: u64 = a.parse("image-max-pixels", 1_048_576)?;
    let pad = fe.encode("<|image_pad|>")?[0];
    let im_start = fe.encode("<|im_start|>")?[0];
    let vision_start = fe.encode("<|vision_start|>")?[0];
    let steps = crate::replay::load_requests(a)?;
    let mut prev: Vec<u32> = Vec::new();
    for (i, (mut req, recorded)) in steps.into_iter().enumerate() {
        if a.get("strip-tools").is_some() {
            req.as_object_mut().map(|o| o.remove("tools"));
        }
        if a.get("strip-images").is_some() {
            for m in req["messages"].as_array_mut().into_iter().flatten() {
                if let Some(parts) = m.get_mut("content").and_then(Value::as_array_mut) {
                    parts.retain(|p| p.get("image_url").is_none());
                }
            }
        }
        // The same normalization `velox serve` applies.
        let n = normalize_request(&req).map_err(|e| Error::Config(e.message))?;
        let urls = n.images;
        let text = fe
            .template
            .render(&n.messages, n.tools.as_ref(), false, true)?;
        let ids = fe.encode(&text)?;
        let mut images = Vec::new();
        for u in &urls {
            let (rgb, w, h) = decode_data_url(u)?;
            images.push(preprocess(&rgb, w, h, min_px, max_px.min(image_cap))?);
        }
        let p = build_prompt(&ids, pad, &images).map_err(Error::Config)?;
        let shared = prev.iter().zip(&p.keys).take_while(|(x, y)| x == y).count();
        let last = |t: u32| p.ids.iter().rposition(|&x| x == t).map_or(-1, |v| v as i64);
        let img_tokens: usize = p.runs.iter().map(|r| r.len).sum();
        let grids: Vec<String> = images
            .iter()
            .map(|im| format!("{}x{}", im.grid_w * 16, im.grid_h * 16))
            .collect();
        println!(
            "step {:>3}: tokens {:>6} (recorded {:>6})  text {:>6}  image {:>5} {:?}  shared-with-prev {:>6}  last <|vision_start|> {:>6}  last <|im_start|> {:>6}  text-bytes {}",
            i + 1,
            p.ids.len(),
            recorded.get("promptTokens").map_or("-".into(), |v| v.to_string()),
            p.ids.len() - img_tokens,
            img_tokens,
            grids,
            shared,
            last(vision_start),
            last(im_start),
            text.len()
        );
        if let Some(path) = a.get("dump-dir") {
            std::fs::write(format!("{path}/step-{}.txt", i + 1), &text)
                .map_err(|e| Error::Config(e.to_string()))?;
        }
        prev = p.keys;
    }
    Ok(ExitCode::SUCCESS)
}
