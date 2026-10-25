//! Persistent weight formats: exact decoding and physical geometry.
//!
//! Semantics follow the `.ninfer` v3 / `.vi` v1 numeric-format and storage-layout contracts
//! (docs/weights.md). Weights stay in their stored encoding on the device; kernels decode on the
//! fly. The functions here are the reference decoders the CPU oracle and conformance use.

use crate::error::{Error, Result};
use crate::ggml::GgmlType;

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
    /// `iq3_xxs` / `iq3_xxs_planar_v1`: llama.cpp IQ3_XXS (256-value blocks of grid codes, one
    /// binary16 scale per block), re-ordered by ninfer's converter. See [`iq3_xxs_at`].
    Iq3Xxs,
    /// `ggml_<type>` / `ggml_block_v1`: llama.cpp super-blocks as GGUF stores them (see
    /// [`crate::ggml`]).
    Ggml(GgmlType),
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
            ("iq3_xxs", "iq3_xxs_planar_v1") => Ok(WeightFormat::Iq3Xxs),
            (f, "ggml_block_v1") if GgmlType::from_name(f).is_some() => {
                Ok(WeightFormat::Ggml(GgmlType::from_name(f).unwrap()))
            }
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
            WeightFormat::Iq3Xxs => "iq3_xxs",
            WeightFormat::Ggml(t) => t.name(),
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
        if *self == WeightFormat::Iq3Xxs && !k.is_multiple_of(IQ3_XXS_BLOCK) {
            return Err(Error::Shape(format!(
                "iq3_xxs needs K % 256 == 0, got [{n},{k}]"
            )));
        }
        if let WeightFormat::Ggml(t) = self {
            if !k.is_multiple_of(t.block_values()) {
                return Err(Error::Shape(format!(
                    "{} needs K % {} == 0, got [{n},{k}]",
                    t.name(),
                    t.block_values()
                )));
            }
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
            WeightFormat::Iq3Xxs => {
                let blocks = n * (k / IQ3_XXS_BLOCK);
                align256(blocks * IQ3_XXS_CODE_BYTES) + 2 * blocks
            }
            WeightFormat::Ggml(t) => n * t.row_bytes(k),
            q => q.row_split().map_or(0, |r| r.planes(n, k).2),
        }
    }

    /// Container layout name.
    pub fn layout(&self) -> &'static str {
        match self {
            WeightFormat::F32 | WeightFormat::Bf16 => "contiguous_le_v1",
            WeightFormat::Fp8RowBf16 => "row_scale_v1",
            WeightFormat::Nvfp4 => "block_scale_k16_m128x4_v1",
            WeightFormat::Iq3Xxs => "iq3_xxs_planar_v1",
            WeightFormat::Ggml(_) => "ggml_block_v1",
            _ => "row_split_k128_v1",
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

/// Values per IQ3_XXS block (one binary16 scale) and code bytes per block.
pub const IQ3_XXS_BLOCK: usize = 256;
pub const IQ3_XXS_CODE_BYTES: usize = 96;

/// llama.cpp `iq3xxs_grid` (ggml-common.h, MIT): four unsigned magnitudes per entry, byte `j`
/// little-endian.
pub const IQ3_XXS_GRID: [u32; 256] = [
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
];

/// Decoded weight `[r, c]` of an `iq3_xxs_planar_v1` parent `[n, k]` (ninfer
/// `tools/artifact/codecs/iq3_xxs.py`, `dequantize_iq3_xxs`).
///
/// Payload: a code plane `[n, k/256, 96]`, then (256-byte aligned) the binary16 block scales `d`
/// `[n, k/256]`. Each block holds 8 sub-blocks `s` of 32 values; value `j` of sub-block `s` uses
/// grid entry `e = j / 4` and sign group `g = j / 8`. Inside a block's 96 bytes:
///
/// - byte `16 (e / 2) + 2 s + e % 2` is the grid index of entry `e` of sub-block `s`;
/// - bytes `64 + 8 q ..` are a little-endian u64 lane word `q`: bits `[7 s, 7 s + 7)` are the
///   7-bit sign index of group `q` of sub-block `s`, bits `[56, 60)` and `[60, 64)` the 4-bit scales
///   `ls` of sub-blocks `2 q` and `2 q + 1`.
///
/// The eighth sign bit is the parity of the other seven; value = magnitude (byte `j % 4` of the
/// grid entry) x sign (bit `j % 8`) x `d (0.25 + 0.5 ls)`.
pub fn iq3_xxs_at(payload: &[u8], n: usize, k: usize, r: usize, c: usize) -> f32 {
    let blocks = k / IQ3_XXS_BLOCK;
    let bi = r * blocks + c / IQ3_XXS_BLOCK;
    let code = &payload[bi * IQ3_XXS_CODE_BYTES..(bi + 1) * IQ3_XXS_CODE_BYTES];
    let scale_off = align256(n * blocks * IQ3_XXS_CODE_BYTES) + 2 * bi;
    let d = f16_to_f32(u16::from_le_bytes([
        payload[scale_off],
        payload[scale_off + 1],
    ]));
    let (s, j) = ((c % IQ3_XXS_BLOCK) / 32, c % 32);
    let e = j / 4;
    let index = code[16 * (e / 2) + 2 * s + e % 2];
    let magnitude = (IQ3_XXS_GRID[usize::from(index)] >> (8 * (j % 4))) & 0xff;
    let lane = |q: usize| u64::from_le_bytes(code[64 + 8 * q..72 + 8 * q].try_into().unwrap());
    let sign7 = (lane(j / 8) >> (7 * s)) & 127;
    let sign8 = sign7 | (u64::from(sign7.count_ones() & 1) << 7);
    let ls = (lane(s / 2) >> (56 + 4 * (s % 2))) & 15;
    let v = magnitude as f32 * d * (0.25 + 0.5 * ls as f32);
    if sign8 >> (j % 8) & 1 != 0 {
        -v
    } else {
        v
    }
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

/// Nearest non-negative E4M3FN code to `x` (saturating at 448; ties to the lower code).
pub fn f32_to_e4m3fn(x: f32) -> u8 {
    if !(x > 0.0) {
        return 0;
    }
    // Positive finite codes 0..=0x7e are increasing in value: binary search.
    let (mut lo, mut hi) = (0u8, 0x7eu8);
    if x >= e4m3fn_to_f32(hi) {
        return hi;
    }
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if e4m3fn_to_f32(mid) <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    if x - e4m3fn_to_f32(lo) <= e4m3fn_to_f32(hi) - x {
        lo
    } else {
        hi
    }
}

/// Encodes `[n, k]` row-major values as an `nvfp4` payload (`block_scale_k16_m128x4_v1`), as
/// compressed-tensors' NVFP4 does: divisor `448 * 6 / amax`, per 16 values an E4M3 scale of
/// `amax16 / 6 * divisor`, each value the nearest E2M1 code of `x * divisor / scale`. Needs
/// `n % 128 == 0` and `k % 64 == 0`.
pub fn encode_nvfp4(x: &[f32], n: usize, k: usize) -> Vec<u8> {
    assert!(
        x.len() == n * k && n % 128 == 0 && k % 64 == 0,
        "nvfp4 shape [{n}, {k}]"
    );
    const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let divisor = if amax > 0.0 { 448.0 * 6.0 / amax } else { 1.0 };
    let scale_plane = align256(n * k / 2);
    let mut out = vec![0u8; scale_plane + n * k / 16 + 4];
    for r in 0..n {
        for g in 0..k / 16 {
            let blk = &x[r * k + g * 16..r * k + g * 16 + 16];
            let bmax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
            let sc = f32_to_e4m3fn(bmax / 6.0 * divisor);
            out[scale_plane + nvfp4_scale_offset(r, g, k)] = sc;
            let step = e4m3fn_to_f32(sc) / divisor;
            if step <= 0.0 {
                continue;
            }
            for (j, &v) in blk.iter().enumerate() {
                let q = (v.abs() / step).min(6.0);
                // Nearest magnitude (ties to the lower one).
                let mut code = 0u8;
                for c in 1..8 {
                    if q - MAG[c - 1] > MAG[c] - q {
                        code = c as u8;
                    }
                }
                let code = code | if v < 0.0 && code != 0 { 8 } else { 0 };
                let col = g * 16 + j;
                let byte = &mut out[(r * k + col) / 2];
                *byte |= if col % 2 == 0 { code } else { code << 4 };
            }
        }
    }
    out[scale_plane + n * k / 16..].copy_from_slice(&divisor.to_le_bytes());
    out
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
    if let WeightFormat::Ggml(t) = fmt {
        let mut out = Vec::with_capacity(rows * k);
        crate::ggml::decode_rows_into(t, payload, k, row_begin, rows, &mut out);
        return Ok(out);
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
            WeightFormat::Iq3Xxs => out.extend((0..k).map(|c| iq3_xxs_at(payload, n, k, r, c))),
            WeightFormat::Ggml(_) => unreachable!("decoded above"),
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
    #[test]
    fn nvfp4_encoding_round_trips_its_grid() {
        // Values on a block's representable grid decode back exactly; others within half a step.
        let (n, k) = (128, 64);
        let mut x: Vec<f32> = (0..n * k)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.013)
            .collect();
        x[5] = 0.0;
        let p = encode_nvfp4(&x, n, k);
        assert_eq!(p.len(), WeightFormat::Nvfp4.payload_bytes(n, k));
        let y = decode_rows(WeightFormat::Nvfp4, &p, n, k, 0, n).unwrap();
        let again = encode_nvfp4(&y, n, k);
        assert_eq!(
            decode_rows(WeightFormat::Nvfp4, &again, n, k, 0, n).unwrap(),
            y
        );
        for (blk, (xs, ys)) in x.chunks(16).zip(y.chunks(16)).enumerate() {
            let amax = xs.iter().fold(0f32, |m, v| m.max(v.abs()));
            for (a, b) in xs.iter().zip(ys) {
                // Half the widest E2M1 gap (2 at the top of the range) plus E4M3 scale rounding.
                assert!(
                    (a - b).abs() <= amax / 6.0 * 1.07 + 1e-6,
                    "block {blk}: {a} vs {b}"
                );
            }
        }
        assert_eq!(f32_to_e4m3fn(448.0), 0x7e);
        assert_eq!(f32_to_e4m3fn(1e9), 0x7e);
        assert_eq!(e4m3fn_to_f32(f32_to_e4m3fn(1.0)), 1.0);
    }

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
        // Dense MLP down projection of the IQ3_XXS Qwen3.8-27B artifact: [5120, 17408].
        assert_eq!(WeightFormat::Iq3Xxs.payload_bytes(5120, 17408), 34_119_680);
        assert_eq!(WeightFormat::Iq3Xxs.payload_bytes(34816, 5120), 68_239_360);
        // Worked example from the row_split_k128_v1 contract: Q5 [2,130].
        assert_eq!(
            WeightFormat::Q5G64.row_split().unwrap().planes(2, 130),
            (256, 512, 528)
        );
    }

    #[test]
    fn iq3_xxs_decodes_a_real_block_like_ninfer() {
        // Block 37 of row 1234 of text/layers/0/mlp/down in qwen3_8_27b_iq3xxs_dflash2.ninfer;
        // expected values from ninfer's `dequantize_iq3_xxs`, stored here as a one-block [1, 256]
        // matrix.
        let codes = concat!(
            "f6a8536319ba557e9c2a007b29a33d5482297016c42838e811792f3a45510323",
            "cfa110cdd301342f59086310916a4992c71a42a123cdac6604540f969ad07993",
            "28035a5337d63a85666609b2416b8c476adebc747afbecf8efabb8d93cd8b365"
        );
        let mut p: Vec<u8> = (0..96)
            .map(|i| u8::from_str_radix(&codes[2 * i..2 * i + 2], 16).unwrap())
            .collect();
        p.resize(256, 0);
        p.extend_from_slice(&[0x44, 0x07]);
        assert_eq!(p.len(), WeightFormat::Iq3Xxs.payload_bytes(1, 256));
        let v = decode_rows(WeightFormat::Iq3Xxs, &p, 1, 256, 0, 1).unwrap();
        let close = |got: f64, want: f64| (got - want).abs() <= 1e-6 * (1.0 + want.abs());
        let sum: f64 = v.iter().map(|&x| f64::from(x)).sum();
        let abs: f64 = v.iter().map(|&x| f64::from(x).abs()).sum();
        let weighted: f64 = v
            .iter()
            .enumerate()
            .map(|(i, &x)| f64::from(x) * (i + 1) as f64)
            .sum();
        assert!(close(sum, -0.24983346462249756), "sum {sum}");
        assert!(close(abs, 2.123667597770691), "abs {abs}");
        assert!(close(weighted, -40.206724405288696), "weighted {weighted}");
        assert!(
            close(f64::from(v[0]), 0.0012195110321044922),
            "v[0] = {}",
            v[0]
        );
        assert!(
            close(f64::from(v[3]), -0.01890242099761963),
            "v[3] = {}",
            v[3]
        );
        assert!(
            close(f64::from(v[31]), 0.0012195110321044922),
            "v[31] = {}",
            v[31]
        );
        assert!(
            close(f64::from(v[32]), 0.0018846988677978516),
            "v[32] = {}",
            v[32]
        );
        assert!(
            close(f64::from(v[77]), -0.025776028633117676),
            "v[77] = {}",
            v[77]
        );
        assert!(
            close(f64::from(v[130]), -0.009423494338989258),
            "v[130] = {}",
            v[130]
        );
        assert!(
            close(f64::from(v[200]), 0.0012195110321044922),
            "v[200] = {}",
            v[200]
        );
        assert!(
            close(f64::from(v[255]), 0.010088682174682617),
            "v[255] = {}",
            v[255]
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
