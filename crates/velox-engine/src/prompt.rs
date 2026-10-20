//! Turns rendered prompt ids plus preprocessed images into model input: image placeholders
//! expanded to one token per merged patch, MRoPE (t, h, w) positions, and prefix-cache keys that
//! distinguish different images behind identical placeholder ids.

use velox_frontend::image::ImagePatches;

/// One image's placeholder run in the expanded prompt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageRun {
    pub start: usize,
    pub len: usize,
    pub image: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Prompt {
    pub ids: Vec<u32>,
    /// Equal to `ids` except image tokens, which carry an image-content hash (bit 31 set, which
    /// no vocabulary id has).
    pub keys: Vec<u32>,
    /// MRoPE positions per token; None for text-only prompts (positions equal indices).
    pub positions: Option<Vec<[u32; 3]>>,
    /// Rotary position minus index for tokens after the prompt.
    pub rope_delta: i64,
    pub runs: Vec<ImageRun>,
}

/// Expands each `image_pad` id into the matching image's merged-token count (images in order).
pub fn build(ids: &[u32], image_pad: u32, images: &[ImagePatches]) -> Result<Prompt, String> {
    let pads = ids.iter().filter(|&&t| t == image_pad).count();
    if pads != images.len() {
        return Err(format!(
            "prompt has {pads} image placeholders for {} images",
            images.len()
        ));
    }
    if images.is_empty() {
        return Ok(Prompt {
            ids: ids.to_vec(),
            keys: ids.to_vec(),
            positions: None,
            rope_delta: 0,
            runs: Vec::new(),
        });
    }
    let mut out = Prompt {
        ids: Vec::new(),
        keys: Vec::new(),
        positions: Some(Vec::new()),
        rope_delta: 0,
        runs: Vec::new(),
    };
    let positions = out.positions.as_mut().unwrap();
    let (mut current, mut max_pos, mut next_image) = (0u32, 0u32, 0usize);
    for &t in ids {
        if t != image_pad {
            out.ids.push(t);
            out.keys.push(t);
            positions.push([current; 3]);
            max_pos = max_pos.max(current);
            current += 1;
            continue;
        }
        let img = &images[next_image];
        let (gh, gw) = (img.grid_h / 2, img.grid_w / 2);
        out.runs.push(ImageRun {
            start: out.ids.len(),
            len: gh * gw,
            image: next_image,
        });
        for y in 0..gh {
            for x in 0..gw {
                out.ids.push(image_pad);
                out.keys.push(
                    0x8000_0000
                        | ((img.hash as u32).wrapping_add((y * gw + x) as u32) & 0x7fff_ffff),
                );
                let p = [current, current + y as u32, current + x as u32];
                max_pos = max_pos.max(p[1]).max(p[2]);
                positions.push(p);
            }
        }
        current += gh.max(gw) as u32;
        next_image += 1;
    }
    out.rope_delta = i64::from(max_pos) + 1 - out.ids.len() as i64;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(gh: usize, gw: usize, hash: u64) -> ImagePatches {
        ImagePatches {
            grid_h: gh,
            grid_w: gw,
            data: Vec::new(),
            hash,
        }
    }

    #[test]
    fn text_only_is_identity() {
        let p = build(&[1, 2, 3], 99, &[]).unwrap();
        assert_eq!((p.ids, p.positions, p.rope_delta), (vec![1, 2, 3], None, 0));
    }

    #[test]
    fn image_expansion_and_mrope_positions() {
        // text, <vision_start>, pad (grid 4x6 -> merged 2x3), <vision_end>, text
        let p = build(&[10, 11, 99, 12, 13], 99, &[image(4, 6, 7)]).unwrap();
        assert_eq!(p.ids.len(), 2 + 6 + 2);
        assert_eq!(
            p.runs,
            vec![ImageRun {
                start: 2,
                len: 6,
                image: 0
            }]
        );
        let pos = p.positions.unwrap();
        assert_eq!(pos[1], [1, 1, 1]);
        // Image tokens: (c, c+y, c+x) with c = 2, y in 0..2, x in 0..3, row-major.
        assert_eq!(
            &pos[2..8],
            &[
                [2, 2, 2],
                [2, 2, 3],
                [2, 2, 4],
                [2, 3, 2],
                [2, 3, 3],
                [2, 3, 4]
            ]
        );
        // After the image, c advances by max(2, 3) = 3.
        assert_eq!(pos[8], [5, 5, 5]);
        assert_eq!(pos[9], [6, 6, 6]);
        // max position 6, 10 tokens: delta = 6 + 1 - 10.
        assert_eq!(p.rope_delta, -3);
        assert!(p.keys[2..8].iter().all(|&k| k & 0x8000_0000 != 0));
        assert_eq!(&p.keys[..2], &[10, 11]);
    }

    #[test]
    fn different_images_get_different_keys() {
        let a = build(&[99], 99, &[image(2, 2, 1)]).unwrap();
        let b = build(&[99], 99, &[image(2, 2, 2)]).unwrap();
        assert_ne!(a.keys, b.keys);
        assert!(build(&[99, 99], 99, &[image(2, 2, 1)]).is_err());
    }
}
