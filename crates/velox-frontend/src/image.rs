//! Qwen3.5 still-image preprocessing (docs/qwen3_5-vision.md): decode, smart resize, Pillow-style
//! antialiased bicubic resampling with uint8 rounding after each pass, BF16-rounded normalisation,
//! and patchify in merge-group order. Matching ninfer bit for bit matters: the vision tower sees
//! exactly these values.

use base64::Engine as _;
use velox_core::format::{bf16_to_f32, f32_to_bf16};
use velox_core::{Error, Result};

pub const PATCH: usize = 16;
pub const TEMPORAL: usize = 2;
pub const MERGE: usize = 2;
const FACTOR: usize = PATCH * MERGE;

/// An image ready for the vision tower.
#[derive(Clone, Debug, PartialEq)]
pub struct ImagePatches {
    /// Patch grid (rows, columns), both even.
    pub grid_h: usize,
    pub grid_w: usize,
    /// `[grid_h * grid_w, 1536]` in merge-group order.
    pub data: Vec<f32>,
    /// Content hash (of the decoded pixels and size) for prefix-cache keys.
    pub hash: u64,
}

impl ImagePatches {
    pub fn merged_tokens(&self) -> usize {
        self.grid_h * self.grid_w / (MERGE * MERGE)
    }
}

/// `smart_resize`: dimensions that are multiples of 32 within `[min_pixels, max_pixels]`.
pub fn smart_resize(
    height: usize,
    width: usize,
    min_pixels: u64,
    max_pixels: u64,
) -> Result<(usize, usize)> {
    if height == 0 || width == 0 {
        return Err(Error::Config("image has a zero dimension".into()));
    }
    let (h, w) = (height as f64, width as f64);
    if h.max(w) / h.min(w) > 200.0 {
        return Err(Error::Config(format!(
            "image aspect ratio {width}x{height} exceeds 200"
        )));
    }
    let f = FACTOR as f64;
    let mut rh = (h / f).round_ties_even() * f;
    let mut rw = (w / f).round_ties_even() * f;
    if rh * rw > max_pixels as f64 {
        let beta = (h * w / max_pixels as f64).sqrt();
        rh = f.max((h / beta / f).floor() * f);
        rw = f.max((w / beta / f).floor() * f);
    } else if rh * rw < min_pixels as f64 {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        rh = (h * beta / f).ceil() * f;
        rw = (w * beta / f).ceil() * f;
    }
    Ok((rh as usize, rw as usize))
}

fn cubic(x: f32) -> f32 {
    const A: f32 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// Per output index: first input index and normalised tap weights.
fn coefficients(input: usize, output: usize) -> Vec<(usize, Vec<f32>)> {
    let scale = input as f32 / output as f32;
    let filter_scale = scale.max(1.0);
    let support = 2.0 * filter_scale;
    let ss = 1.0 / filter_scale;
    (0..output)
        .map(|dst| {
            let center = scale * (dst as f32 + 0.5);
            let begin = ((center - support + 0.5) as i64).max(0) as usize;
            let end = ((center + support + 0.5) as i64).min(input as i64) as usize;
            let mut w: Vec<f32> = (begin..end)
                .map(|j| cubic((j as f32 - center + 0.5) * ss))
                .collect();
            let total: f32 = w.iter().sum();
            if total != 0.0 {
                w.iter_mut().for_each(|v| *v /= total);
            }
            (begin, w)
        })
        .collect()
}

fn to_u8(v: f32) -> u8 {
    v.round_ties_even().clamp(0.0, 255.0) as u8
}

/// Separable bicubic resize of an RGB8 image (horizontal pass first), rounding to uint8 after
/// each pass.
pub fn resize_rgb(src: &[u8], w: usize, h: usize, out_w: usize, out_h: usize) -> Vec<u8> {
    let horizontal = if out_w == w {
        src.to_vec()
    } else {
        let cx = coefficients(w, out_w);
        let mut tmp = vec![0u8; h * out_w * 3];
        for y in 0..h {
            for (x, (begin, wts)) in cx.iter().enumerate() {
                for c in 0..3 {
                    let acc: f32 = wts
                        .iter()
                        .enumerate()
                        .map(|(j, wt)| wt * f32::from(src[(y * w + begin + j) * 3 + c]))
                        .sum();
                    tmp[(y * out_w + x) * 3 + c] = to_u8(acc);
                }
            }
        }
        tmp
    };
    if out_h == h {
        return horizontal;
    }
    let cy = coefficients(h, out_h);
    let mut out = vec![0u8; out_h * out_w * 3];
    for (y, (begin, wts)) in cy.iter().enumerate() {
        for x in 0..out_w {
            for c in 0..3 {
                let acc: f32 = wts
                    .iter()
                    .enumerate()
                    .map(|(j, wt)| wt * f32::from(horizontal[((begin + j) * out_w + x) * 3 + c]))
                    .sum();
                out[(y * out_w + x) * 3 + c] = to_u8(acc);
            }
        }
    }
    out
}

/// Decodes a `data:image/...;base64,` URL or bare base64 into RGB8.
pub fn decode_data_url(url: &str) -> Result<(Vec<u8>, usize, usize)> {
    let b64 = match url.strip_prefix("data:") {
        Some(rest) => rest
            .split_once(";base64,")
            .map(|(_, d)| d)
            .ok_or_else(|| Error::Config("image data URL is not base64".into()))?,
        None if url.starts_with("http://") || url.starts_with("https://") => {
            return Err(Error::Unsupported(
                "remote image URLs are not fetched; send a data URL".into(),
            ))
        }
        None => url,
    };
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64.trim())
        .map_err(|e| Error::Config(format!("image base64: {e}")))?;
    let img = image::load_from_memory(&bytes)
        .map_err(|e| Error::Config(format!("image decode: {e}")))?
        .to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    Ok((img.into_raw(), w, h))
}

fn fnv(data: &[u8], seed: u64) -> u64 {
    data.iter().fold(seed, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
    })
}

/// Full preprocessing of an RGB8 image.
pub fn preprocess(
    rgb: &[u8],
    w: usize,
    h: usize,
    min_pixels: u64,
    max_pixels: u64,
) -> Result<ImagePatches> {
    let (rh, rw) = smart_resize(h, w, min_pixels, max_pixels)?;
    let img = resize_rgb(rgb, w, h, rw, rh);
    // Lookup table: bf16_rne(v / 127.5 - 1).
    let lut: Vec<f32> = (0..=255u32)
        .map(|v| bf16_to_f32(f32_to_bf16(v as f32 / 127.5 - 1.0)))
        .collect();
    let (gh, gw) = (rh / PATCH, rw / PATCH);
    let feat = 3 * TEMPORAL * PATCH * PATCH;
    let mut data = Vec::with_capacity(gh * gw * feat);
    for by in 0..gh / MERGE {
        for bx in 0..gw / MERGE {
            for my in 0..MERGE {
                for mx in 0..MERGE {
                    let (py, px) = (by * MERGE + my, bx * MERGE + mx);
                    // idx = c*512 + t*256 + y*16 + x; the still frame fills both temporal slots.
                    for c in 0..3 {
                        for _t in 0..TEMPORAL {
                            for y in 0..PATCH {
                                for x in 0..PATCH {
                                    let (iy, ix) = (py * PATCH + y, px * PATCH + x);
                                    data.push(lut[usize::from(img[(iy * rw + ix) * 3 + c])]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let hash = fnv(
        &img,
        fnv(
            &(rw as u64 * 100_003 + rh as u64).to_le_bytes(),
            0xcbf2_9ce4_8422_2325,
        ),
    );
    Ok(ImagePatches {
        grid_h: gh,
        grid_w: gw,
        data,
        hash,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_rules() {
        // 1344x768 is already aligned and inside the bounds.
        assert_eq!(
            smart_resize(768, 1344, 65_536, 16_777_216).unwrap(),
            (768, 1344)
        );
        // Rounds to the nearest multiple of 32, ties to even.
        assert_eq!(
            smart_resize(1080, 1920, 65_536, 16_777_216).unwrap(),
            (1088, 1920)
        );
        // Too small: scaled up to reach min_pixels.
        let (h, w) = smart_resize(100, 100, 65_536, 16_777_216).unwrap();
        assert!(h * w >= 65_536 && h % 32 == 0 && w % 32 == 0);
        // Too large: scaled down under max_pixels.
        let (h, w) = smart_resize(1080, 1920, 65_536, 1_048_576).unwrap();
        // 1920x1080 under a 1 Mpx cap becomes 1344x768 (the daemon's documented result).
        assert!(h * w <= 1_048_576 && (h, w) == (768, 1344), "{h}x{w}");
        assert!(smart_resize(1, 300, 0, u64::MAX).is_err());
    }

    #[test]
    fn identity_resize_and_constant_images() {
        let src: Vec<u8> = (0..4 * 3 * 3).map(|i| (i * 7 % 256) as u8).collect();
        assert_eq!(resize_rgb(&src, 4, 3, 4, 3), src);
        // A constant image stays constant under any resize (weights are normalised).
        let flat = vec![200u8; 64 * 48 * 3];
        assert!(resize_rgb(&flat, 64, 48, 32, 96).iter().all(|&v| v == 200));
    }

    #[test]
    fn patch_layout_and_normalisation() {
        // 64x64 white image -> 4x4 patches, 4 merged tokens; every value is bf16(255/127.5 - 1) = 1.
        let rgb = vec![255u8; 64 * 64 * 3];
        let p = preprocess(&rgb, 64, 64, 0, u64::MAX).unwrap();
        assert_eq!((p.grid_h, p.grid_w, p.merged_tokens()), (4, 4, 4));
        assert_eq!(p.data.len(), 16 * 1536);
        assert!(p.data.iter().all(|&v| v == 1.0));
    }
}
