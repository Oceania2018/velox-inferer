//! KV-cache storage formats.
//!
//! A cache is `[positions, kv_heads]` rows, each row one head vector of `head_dim` values stored in
//! [`KvFormat::row_bytes`] contiguous bytes (codes, then scales, zero-padded to 16 bytes), inside an
//! f32-slot buffer like encoded weights. The semantics follow ninfer's `--kv-dtype`:
//!
//! - `F32`: the values themselves.
//! - `Bf16`: each value rounded to bf16 (RNE).
//! - `Fp8`: K rows are rotated by the normalized Hadamard transform `H` (V rows are not), then one
//!   scale per row: `s = f16(clamp(amax / 448, 2^-24, 65504))`, codes `e4m3(x * (1 / s))` (RNE,
//!   saturating). Row layout: `head_dim` code bytes, then `s` as an f32 (exactly an f16 value).
//! - `Nvfp4`: K and V rows are rotated by `H`, then per contiguous group of 16:
//!   `s = e4m3(clamp(amax / 6, 2^-9, 448))`, codes `e2m1(x / s)` (RNE, saturating); no global
//!   divisor. Row layout: `head_dim / 2` code bytes (element `2i` in the low nibble), then
//!   `head_dim / 16` E4M3 scale bytes.
//!
//! `H` is the Sylvester-Hadamard matrix scaled by `1 / sqrt(head_dim)`: symmetric and orthonormal,
//! so `H⁻¹ = H`. A stored row therefore *represents* `H · dequant(codes)` when rotated and
//! `dequant(codes)` otherwise; [`decode_row`] returns that row and attention is exact softmax
//! attention over the represented rows. Kernels may instead rotate the query and the output
//! (`q · H d = (H q) · d`), which is the same quantity.
//!
//! Encoding is specified operation by operation (f32, round to nearest even) so device writers and
//! [`encode_row`] produce identical bytes: the Hadamard butterflies run in ascending stride order,
//! `a + b` / `a - b` per pair, followed by one multiply by [`hadamard_norm`].

use std::fmt;

use crate::error::{Error, Result};
use crate::format::{bf16_to_f32, e2m1_to_f32, e4m3fn_to_f32, f32_to_bf16};
use crate::quant::{f32_to_e2m1, f32_to_e4m3fn, round_f16};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum KvFormat {
    #[default]
    F32,
    Bf16,
    Fp8,
    Nvfp4,
}

impl fmt::Display for KvFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl KvFormat {
    pub const ALL: [KvFormat; 4] = [
        KvFormat::F32,
        KvFormat::Bf16,
        KvFormat::Fp8,
        KvFormat::Nvfp4,
    ];

    pub fn name(self) -> &'static str {
        match self {
            KvFormat::F32 => "f32",
            KvFormat::Bf16 => "bf16",
            KvFormat::Fp8 => "fp8",
            KvFormat::Nvfp4 => "nvfp4",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        KvFormat::ALL
            .into_iter()
            .find(|f| f.name() == s)
            .ok_or_else(|| Error::Config(format!("unknown KV format '{s}' (f32|bf16|fp8|nvfp4)")))
    }

    /// Whether stored K rows are Hadamard-rotated.
    pub fn rotates_k(self) -> bool {
        matches!(self, KvFormat::Fp8 | KvFormat::Nvfp4)
    }

    /// Whether stored V rows are Hadamard-rotated.
    pub fn rotates_v(self) -> bool {
        self == KvFormat::Nvfp4
    }

    /// The head dims this format can store. Rotated formats need a power of two in 32..=256 (one
    /// warp-wide Hadamard transform).
    pub fn check_head_dim(self, head_dim: usize) -> Result<()> {
        let ok = match self {
            KvFormat::F32 | KvFormat::Bf16 => head_dim > 0,
            KvFormat::Fp8 | KvFormat::Nvfp4 => {
                head_dim.is_power_of_two() && (32..=256).contains(&head_dim)
            }
        };
        if ok {
            Ok(())
        } else {
            Err(Error::Shape(format!(
                "KV format {self} does not support head_dim {head_dim}"
            )))
        }
    }

    /// Bytes of one stored `(position, kv_head)` row: `4 head_dim` for `F32`, otherwise padded to
    /// a multiple of 16 so kernels can read rows with 16-byte loads.
    pub fn row_bytes(self, head_dim: usize) -> usize {
        match self {
            KvFormat::F32 => 4 * head_dim,
            KvFormat::Bf16 => (2 * head_dim).next_multiple_of(16),
            KvFormat::Fp8 => (head_dim + 4).next_multiple_of(16),
            KvFormat::Nvfp4 => (head_dim / 2 + head_dim / 16).next_multiple_of(16),
        }
    }

    /// f32 slots of one stored row.
    pub fn row_slots(self, head_dim: usize) -> usize {
        self.row_bytes(head_dim) / 4
    }

    /// f32 slots of a cache holding `positions` positions of `kv_heads` rows, or `None` on
    /// overflow.
    pub fn cache_slots(self, positions: usize, kv_heads: usize, head_dim: usize) -> Option<usize> {
        positions
            .checked_mul(kv_heads)?
            .checked_mul(self.row_slots(head_dim))
    }
}

/// `1 / sqrt(head_dim)` in f32, the normalization of the stored Hadamard transform.
pub fn hadamard_norm(head_dim: usize) -> f32 {
    (1.0 / (head_dim as f64).sqrt()) as f32
}

/// Normalized Sylvester-Hadamard transform in f32, in the device operation order (ascending
/// butterfly strides, then one multiply by [`hadamard_norm`]). `x.len()` must be a power of two.
pub fn hadamard_f32(x: &mut [f32]) {
    let n = x.len();
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(2 * h) {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
    let norm = hadamard_norm(n);
    for v in x {
        *v *= norm;
    }
}

/// The normalized Hadamard transform in f64 (reference decode).
fn hadamard_f64(x: &mut [f64]) {
    let n = x.len();
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(2 * h) {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
    let norm = 1.0 / (n as f64).sqrt();
    for v in x {
        *v *= norm;
    }
}

/// Encodes one row (`x.len() = head_dim` values of a K row, or a V row with `is_v`) into
/// `out[..fmt.row_bytes(head_dim)]`. The head dim must pass [`KvFormat::check_head_dim`].
pub fn encode_row(fmt: KvFormat, x: &[f32], is_v: bool, out: &mut [u8]) {
    let d = x.len();
    let out = &mut out[..fmt.row_bytes(d)];
    out.fill(0);
    let rotate = if is_v {
        fmt.rotates_v()
    } else {
        fmt.rotates_k()
    };
    let mut y = x.to_vec();
    if rotate {
        hadamard_f32(&mut y);
    }
    match fmt {
        KvFormat::F32 => {
            for (o, v) in out.as_chunks_mut::<4>().0.iter_mut().zip(&y) {
                *o = v.to_le_bytes();
            }
        }
        KvFormat::Bf16 => {
            for (o, &v) in out.as_chunks_mut::<2>().0.iter_mut().zip(&y) {
                *o = f32_to_bf16(v).to_le_bytes();
            }
        }
        KvFormat::Fp8 => {
            let amax = y.iter().fold(0f32, |m, v| m.max(v.abs()));
            let (scale, inv) = if amax == 0.0 {
                (0.0, 0.0)
            } else {
                let s = round_f16((amax / 448.0).clamp(2f32.powi(-24), 65504.0));
                (s, 1.0 / s)
            };
            for (o, &v) in out.iter_mut().zip(&y) {
                *o = if inv == 0.0 {
                    0
                } else {
                    f32_to_e4m3fn(v * inv)
                };
            }
            out[d..d + 4].copy_from_slice(&scale.to_le_bytes());
        }
        KvFormat::Nvfp4 => {
            for (g, group) in y.chunks(16).enumerate() {
                let amax = group.iter().fold(0f32, |m, v| m.max(v.abs()));
                if amax == 0.0 {
                    continue; // zero scale and codes
                }
                let sc = f32_to_e4m3fn((amax / 6.0).clamp(2f32.powi(-9), 448.0));
                let rep = e4m3fn_to_f32(sc);
                for (i, &v) in group.iter().enumerate() {
                    let e = g * 16 + i;
                    out[e / 2] |= f32_to_e2m1(v / rep) << (4 * (e % 2));
                }
                out[d / 2 + g] = sc;
            }
        }
    }
}

/// The row a stored row represents (see the module docs), `out.len() = head_dim`.
pub fn decode_row(fmt: KvFormat, bytes: &[u8], is_v: bool, out: &mut [f32]) {
    let d = out.len();
    let rotate = if is_v {
        fmt.rotates_v()
    } else {
        fmt.rotates_k()
    };
    let mut y: Vec<f64> = match fmt {
        KvFormat::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .take(d)
            .map(|&b| f64::from(f32::from_le_bytes(b)))
            .collect(),
        KvFormat::Bf16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .take(d)
            .map(|&b| f64::from(bf16_to_f32(u16::from_le_bytes(b))))
            .collect(),
        KvFormat::Fp8 => {
            let s = f32::from_le_bytes([bytes[d], bytes[d + 1], bytes[d + 2], bytes[d + 3]]);
            bytes[..d]
                .iter()
                .map(|&c| f64::from(e4m3fn_to_f32(c)) * f64::from(s))
                .collect()
        }
        KvFormat::Nvfp4 => (0..d)
            .map(|e| {
                let code = bytes[e / 2] >> (4 * (e % 2));
                f64::from(e2m1_to_f32(code)) * f64::from(e4m3fn_to_f32(bytes[d / 2 + e / 16]))
            })
            .collect(),
    };
    if rotate {
        hadamard_f64(&mut y);
    }
    for (o, v) in out.iter_mut().zip(y) {
        *o = v as f32;
    }
}

/// Encodes `[rows, head_dim]` f32 rows into a cache payload.
pub fn encode_cache(fmt: KvFormat, x: &[f32], head_dim: usize, is_v: bool) -> Vec<u8> {
    let rb = fmt.row_bytes(head_dim);
    let rows = x.len() / head_dim;
    let mut out = vec![0u8; rows * rb];
    for (r, o) in x.chunks_exact(head_dim).zip(out.chunks_exact_mut(rb)) {
        encode_row(fmt, r, is_v, o);
    }
    out
}

/// Decodes the first `rows` rows of a cache payload to `[rows, head_dim]` represented values.
pub fn decode_cache(
    fmt: KvFormat,
    bytes: &[u8],
    rows: usize,
    head_dim: usize,
    is_v: bool,
) -> Vec<f32> {
    let rb = fmt.row_bytes(head_dim);
    let mut out = vec![0f32; rows * head_dim];
    for (o, b) in out.chunks_exact_mut(head_dim).zip(bytes.chunks_exact(rb)) {
        decode_row(fmt, b, is_v, o);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn row_sizes() {
        assert_eq!(KvFormat::F32.row_bytes(256), 1024);
        assert_eq!(KvFormat::Bf16.row_bytes(256), 512);
        assert_eq!(KvFormat::Bf16.row_bytes(3), 16);
        assert_eq!(KvFormat::Fp8.row_bytes(256), 272);
        assert_eq!(KvFormat::Nvfp4.row_bytes(256), 144);
        assert_eq!(KvFormat::Nvfp4.row_bytes(32), 32);
        assert_eq!(KvFormat::F32.cache_slots(10, 4, 256), Some(10 * 4 * 256));
        assert!(KvFormat::Nvfp4.check_head_dim(96).is_err());
        assert!(KvFormat::Fp8.check_head_dim(128).is_ok());
        assert!(KvFormat::Bf16.check_head_dim(96).is_ok());
        for f in KvFormat::ALL {
            assert_eq!(KvFormat::parse(f.name()).unwrap(), f);
        }
    }

    #[test]
    fn hadamard_is_an_involution() {
        let mut rng = Rng::new(3);
        let x = rng.vec(256, 2.0);
        let mut y: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
        hadamard_f64(&mut y);
        hadamard_f64(&mut y);
        for (a, b) in x.iter().zip(&y) {
            assert!((f64::from(*a) - b).abs() < 1e-12);
        }
        let mut z = x.clone();
        hadamard_f32(&mut z);
        let mut w: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
        hadamard_f64(&mut w);
        for (a, b) in z.iter().zip(&w) {
            assert!((f64::from(*a) - b).abs() < 1e-5);
        }
    }

    #[test]
    fn round_trips_within_format_error() {
        let mut rng = Rng::new(9);
        for (fmt, d, tol) in [
            (KvFormat::F32, 40, 0.0),
            (KvFormat::Bf16, 40, 1.0 / 256.0),
            (KvFormat::Fp8, 64, 1.0 / 16.0),
            (KvFormat::Fp8, 256, 1.0 / 16.0),
            (KvFormat::Nvfp4, 256, 0.25),
            (KvFormat::Nvfp4, 32, 0.25),
        ] {
            for is_v in [false, true] {
                let x = rng.vec(d, 2.0);
                let mut bytes = vec![0u8; fmt.row_bytes(d)];
                encode_row(fmt, &x, is_v, &mut bytes);
                let mut y = vec![0f32; d];
                decode_row(fmt, &bytes, is_v, &mut y);
                let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
                let err = x
                    .iter()
                    .zip(&y)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(err <= tol * amax + 1e-6, "{fmt} d={d} v={is_v}: {err}");
            }
        }
    }

    #[test]
    fn zero_rows_encode_to_zero() {
        for fmt in [KvFormat::Fp8, KvFormat::Nvfp4] {
            let mut bytes = vec![0xffu8; fmt.row_bytes(64)];
            encode_row(fmt, &[0.0; 64], false, &mut bytes);
            assert!(bytes.iter().all(|&b| b == 0), "{fmt}");
        }
    }

    #[test]
    fn huge_values_saturate() {
        let mut x = vec![0f32; 32];
        x[0] = 1e9;
        x[5] = -3e4;
        for fmt in [KvFormat::Fp8, KvFormat::Nvfp4] {
            let mut bytes = vec![0u8; fmt.row_bytes(32)];
            encode_row(fmt, &x, true, &mut bytes);
            let mut y = vec![0f32; 32];
            decode_row(fmt, &bytes, true, &mut y);
            assert!(y.iter().all(|v| v.is_finite()), "{fmt}");
        }
    }
}
