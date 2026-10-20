//! Persistent weight formats: exact decoding and physical geometry.
//!
//! Semantics follow the `.ninfer` v3 / `.vi` v1 numeric-format and storage-layout contracts
//! (docs/weights.md). Weights stay in their stored encoding on the device; kernels decode on the
//! fly. The functions here are the reference decoders the CPU oracle and conformance use.

use crate::error::{Error, Result};

/// Stored weight encodings velox executes. Each implies its storage layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WeightFormat {
    /// `fp32` / `contiguous_le_v1`.
    F32,
    /// `bf16` / `contiguous_le_v1`.
    Bf16,
    /// `fp8_e4m3fn_row_bf16` / `row_scale_v1`: E4M3FN codes, one BF16 multiplier per row.
    Fp8RowBf16,
    /// `nvfp4` / `block_scale_k16_m128x4_v1`: E2M1 codes, E4M3FN scale per 16, FP32 divisor.
    Nvfp4,
    /// `q4_g64_fp16` / `row_split_k128_v1`: signed 4-bit codes, FP16 multiplier per 64.
    Q4G64,
    /// `q5_g64_fp16` / `row_split_k128_v1`.
    Q5G64,
    /// `q6_g64_fp16` / `row_split_k128_v1`.
    Q6G64,
    /// `q8_g32_fp16` / `row_split_k128_v1`: codes in [-127, 127], FP16 multiplier per 32.
    Q8G32,
}

/// Geometry of a `row_split_k128_v1` format: code bits, group size, base and high-bit bytes per
/// group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowSplit {
    pub bits: u32,
    pub group: usize,
    pub base: usize,
    pub high: usize,
}

impl RowSplit {
    pub fn groups_per_row(&self, k: usize) -> usize {
        k.div_ceil(128) * 128 / self.group
    }

    /// (high-plane offset, scale-plane offset, payload bytes) for a parent `[n, k]`.
    pub fn planes(&self, n: usize, k: usize) -> (usize, usize, usize) {
        let groups = n * self.groups_per_row(k);
        let high_off = align256(groups * self.base);
        let scale_off = high_off + align256(groups * self.high);
        (high_off, scale_off, scale_off + 2 * groups)
    }
}

impl WeightFormat {
    /// Maps container (format, layout) names to a format velox executes.
    pub fn from_names(format: &str, layout: &str) -> Result<Self> {
        match (format, layout) {
            ("fp32", "contiguous_le_v1") => Ok(WeightFormat::F32),
            ("bf16", "contiguous_le_v1") => Ok(WeightFormat::Bf16),
            ("fp8_e4m3fn_row_bf16", "row_scale_v1") => Ok(WeightFormat::Fp8RowBf16),
            ("nvfp4", "block_scale_k16_m128x4_v1") => Ok(WeightFormat::Nvfp4),
            ("q4_g64_fp16", "row_split_k128_v1") => Ok(WeightFormat::Q4G64),
            ("q5_g64_fp16", "row_split_k128_v1") => Ok(WeightFormat::Q5G64),
            ("q6_g64_fp16", "row_split_k128_v1") => Ok(WeightFormat::Q6G64),
            ("q8_g32_fp16", "row_split_k128_v1") => Ok(WeightFormat::Q8G32),
            _ => Err(Error::Unsupported(format!(
                "weight format {format} / {layout}"
            ))),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            WeightFormat::F32 => "fp32",
            WeightFormat::Bf16 => "bf16",
            WeightFormat::Fp8RowBf16 => "fp8_e4m3fn_row_bf16",
            WeightFormat::Nvfp4 => "nvfp4",
            WeightFormat::Q4G64 => "q4_g64_fp16",
            WeightFormat::Q5G64 => "q5_g64_fp16",
            WeightFormat::Q6G64 => "q6_g64_fp16",
            WeightFormat::Q8G32 => "q8_g32_fp16",
        }
    }

    pub fn row_split(&self) -> Option<RowSplit> {
        let rs = |bits, group, base, high| {
            Some(RowSplit {
                bits,
                group,
                base,
                high,
            })
        };
        match self {
            WeightFormat::Q4G64 => rs(4, 64, 32, 0),
            WeightFormat::Q5G64 => rs(5, 64, 32, 8),
            WeightFormat::Q6G64 => rs(6, 64, 32, 16),
            WeightFormat::Q8G32 => rs(8, 32, 32, 0),
            _ => None,
        }
    }

    /// Shape restrictions of the layout for a parent matrix `[n, k]`.
    pub fn check_shape(&self, n: usize, k: usize) -> Result<()> {
        if n == 0 || k == 0 {
            return Err(Error::Shape(format!(
                "{} matrix [{n},{k}] is empty",
                self.name()
            )));
        }
        if *self == WeightFormat::Nvfp4 && (n % 128 != 0 || k % 64 != 0) {
            return Err(Error::Shape(format!(
                "nvfp4 needs N % 128 == 0 and K % 64 == 0, got [{n},{k}]"
            )));
        }
        Ok(())
    }

    /// Encoded payload bytes of a parent matrix `[n, k]`.
    pub fn payload_bytes(&self, n: usize, k: usize) -> usize {
        match self {
            WeightFormat::F32 => n * k * 4,
            WeightFormat::Bf16 => n * k * 2,
            WeightFormat::Fp8RowBf16 => align256(n * k) + 2 * n,
            WeightFormat::Nvfp4 => align256(n * k / 2) + n * k / 16 + 4,
            q => q.row_split().map_or(0, |r| r.planes(n, k).2),
        }
    }
}

pub fn align256(x: usize) -> usize {
    x.div_ceil(256) * 256
}

/// IEEE binary16 to f32 (exact).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let e = i32::from((h >> 10) & 0x1f);
    let m = f32::from(h & 0x3ff);
    match e {
        0 => sign * m * 2f32.powi(-24),
        31 => {
            if m == 0.0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => sign * (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    }
}

/// Decoded weight `[r, c]` of a `row_split_k128_v1` parent `[n, k]`.
pub fn row_split_at(rs: RowSplit, payload: &[u8], n: usize, k: usize, r: usize, c: usize) -> f32 {
    let (high_off, scale_off, _) = rs.planes(n, k);
    let gi = r * rs.groups_per_row(k) + c / rs.group;
    let lane = c % rs.group;
    let mut u = if rs.bits == 8 {
        u32::from(payload[gi * rs.base + lane])
    } else {
        let byte = payload[gi * rs.base + lane / 2];
        u32::from(if lane % 2 == 0 { byte & 0xf } else { byte >> 4 })
    };
    // High bits per lane: Q5 1, Q6 2; Q4 and Q8 have no high-bit plane.
    let hb = if rs.high > 0 { rs.bits as usize - 4 } else { 0 };
    for j in 0..hb {
        let t = lane * hb + j;
        let bit = (payload[high_off + gi * rs.high + t / 8] >> (t % 8)) & 1;
        u |= u32::from(bit) << (4 + j);
    }
    let q = if u >= 1 << (rs.bits - 1) {
        u as i32 - (1 << rs.bits)
    } else {
        u as i32
    };
    let s = f16_to_f32(u16::from_le_bytes([
        payload[scale_off + 2 * gi],
        payload[scale_off + 2 * gi + 1],
    ]));
    s * q as f32
}

pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Round-to-nearest-even f32 -> bf16 (NaN kept quiet).
pub fn f32_to_bf16(v: f32) -> u16 {
    let b = v.to_bits();
    if v.is_nan() {
        return ((b >> 16) | 0x40) as u16;
    }
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// E4M3FN: bias 7, no infinities, 0x7f / 0xff are NaN.
pub fn e4m3fn_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = i32::from((b >> 3) & 0xf);
    let m = f32::from(b & 7);
    if e == 15 && b & 7 == 7 {
        return f32::NAN;
    }
    if e == 0 {
        return sign * m * 2f32.powi(-9);
    }
    sign * (1.0 + m / 8.0) * 2f32.powi(e - 7)
}

/// E2M1 code (low 4 bits).
pub fn e2m1_to_f32(c: u8) -> f32 {
    const V: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let v = V[usize::from(c & 7)];
    if c & 8 != 0 {
        -v
    } else {
        v
    }
}

/// Byte offset of NVFP4 scale word (row `n`, 16-wide group `g`) inside the scale plane.
pub fn nvfp4_scale_offset(n: usize, g: usize, k: usize) -> usize {
    let k_tiles = k / 64;
    let (row_tile, row_inner) = (n / 128, n % 128);
    let (scale_tile, scale_lane) = (g / 4, g % 4);
    (row_tile * k_tiles + scale_tile) * 512
        + (row_inner % 32) * 16
        + (row_inner / 32) * 4
        + scale_lane
}

/// Decodes rows `[row_begin, row_begin + rows)` of a parent `[n, k]` matrix to f32 (`[rows, k]`).
pub fn decode_rows(
    fmt: WeightFormat,
    payload: &[u8],
    n: usize,
    k: usize,
    row_begin: usize,
    rows: usize,
) -> Result<Vec<f32>> {
    fmt.check_shape(n, k)?;
    if payload.len() < fmt.payload_bytes(n, k) || row_begin + rows > n {
        return Err(Error::Shape(format!(
            "{} [{n},{k}] rows {row_begin}+{rows}: payload {} bytes",
            fmt.name(),
            payload.len()
        )));
    }
    let mut out = Vec::with_capacity(rows * k);
    for r in row_begin..row_begin + rows {
        match fmt {
            WeightFormat::F32 => out.extend(
                payload[r * k * 4..(r + 1) * k * 4]
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap())),
            ),
            WeightFormat::Bf16 => out.extend(
                payload[r * k * 2..(r + 1) * k * 2]
                    .chunks_exact(2)
                    .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))),
            ),
            WeightFormat::Fp8RowBf16 => {
                let so = align256(n * k) + 2 * r;
                let scale = bf16_to_f32(u16::from_le_bytes([payload[so], payload[so + 1]]));
                out.extend(
                    payload[r * k..(r + 1) * k]
                        .iter()
                        .map(|&c| e4m3fn_to_f32(c) * scale),
                );
            }
            WeightFormat::Nvfp4 => {
                let scale_plane = align256(n * k / 2);
                let div_off = scale_plane + n * k / 16;
                let divisor = f32::from_le_bytes(payload[div_off..div_off + 4].try_into().unwrap());
                for c in 0..k {
                    let byte = payload[(r * k + c) / 2];
                    let code = if c % 2 == 0 { byte & 0xf } else { byte >> 4 };
                    let scale =
                        e4m3fn_to_f32(payload[scale_plane + nvfp4_scale_offset(r, c / 16, k)]);
                    out.push(e2m1_to_f32(code) * scale / divisor);
                }
            }
            q => {
                let rs = q.row_split().expect("row-split format");
                out.extend((0..k).map(|c| row_split_at(rs, payload, n, k, r, c)));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3fn_reference_values() {
        assert_eq!(e4m3fn_to_f32(0x00), 0.0);
        assert_eq!(e4m3fn_to_f32(0x38), 1.0); // e=7, m=0
        assert_eq!(e4m3fn_to_f32(0x7e), 448.0); // max finite
        assert_eq!(e4m3fn_to_f32(0x01), 2f32.powi(-9)); // min subnormal
        assert_eq!(e4m3fn_to_f32(0xb8), -1.0);
        assert!(e4m3fn_to_f32(0x7f).is_nan());
    }

    #[test]
    fn e2m1_and_bf16() {
        assert_eq!(e2m1_to_f32(0x7), 6.0);
        assert_eq!(e2m1_to_f32(0xb), -1.5);
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
        assert_eq!(f32_to_bf16(1.0), 0x3f80);
        assert_eq!(bf16_to_f32(f32_to_bf16(3.140625)), 3.140625);
    }

    #[test]
    fn payload_sizes_match_layout_contract() {
        // Values from the real Qwen3.8-27B artifact.
        assert_eq!(
            WeightFormat::Fp8RowBf16.payload_bytes(16384, 5120),
            83_918_848
        );
        assert_eq!(WeightFormat::Nvfp4.payload_bytes(34816, 5120), 100_270_084);
        assert_eq!(WeightFormat::Nvfp4.payload_bytes(5120, 17408), 50_135_044);
        assert_eq!(WeightFormat::Bf16.payload_bytes(96, 5120), 983_040);
        // Worked example from the row_split_k128_v1 contract: Q5 [2,130].
        assert_eq!(
            WeightFormat::Q5G64.row_split().unwrap().planes(2, 130),
            (256, 512, 528)
        );
    }

    #[test]
    fn nvfp4_scale_swizzle_is_a_permutation() {
        let (n, k) = (256, 128);
        let mut seen = vec![false; n * k / 16];
        for r in 0..n {
            for g in 0..k / 16 {
                let o = nvfp4_scale_offset(r, g, k);
                assert!(!seen[o], "collision at {r},{g}");
                seen[o] = true;
            }
        }
    }
}
