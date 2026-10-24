//! Storage formats for recurrent-state checkpoints (the GDN `S` matrices) kept in the context
//! store (ninfer's `--host-state-dtype`). Device state is always f32; a packed checkpoint is
//! written by [`crate::Backend::state_pack`] and restored by [`crate::Backend::state_unpack`].
//!
//! A state of `n` values (a multiple of [`STATE_ROW`]) is cut into rows of [`STATE_ROW`]
//! contiguous values (one row of an `S[h]` matrix). Packed layouts, in an f32-slot buffer of
//! [`StateFormat::slots`] slots:
//!
//! - `F32`: the values themselves.
//! - `Bf16`: `n` bf16 words (RNE; element `i` at bytes `2i`).
//! - `Nvfp4`: `n / 2` E2M1 code bytes (element `2j` in the low nibble), then `n / 16` E4M3 group
//!   scale bytes (16 values each), then `n / 128` f32 row divisors. Per row, with f32 operations
//!   rounded to nearest even:
//!   `d = 2688 / amax` (0 when `amax` is 0 or `d` is not finite: the row is zero), so the largest
//!   group scale lands on E4M3's 448; per group `s = e4m3((d · amax_g) / 6)` (RNE, saturating);
//!   codes `e2m1((x · d) / s)` (RNE, saturating; 0 when `s` is 0). The value represented is
//!   `(e2m1 · s) / d` (0 when `d` or `s` is 0); the product is exact.
//!
//! Measured on Qwen3.8-27B (2026-10-24): a 1-ulp perturbation of the state changes greedy output
//! as often as either lossy format, i.e. the formats cost nothing measurable beyond the run-to-run
//! sensitivity of the model; ninfer's production A/B gave the same tool on 88/90 (bf16) and 87/90
//! (nvfp4) steps.

use std::fmt;

use crate::error::{Error, Result};
use crate::format::{bf16_to_f32, e2m1_to_f32, e4m3fn_to_f32, f32_to_bf16};
use crate::quant::{f32_to_e2m1, f32_to_e4m3fn};

/// Values per scaled row.
pub const STATE_ROW: usize = 128;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum StateFormat {
    #[default]
    F32,
    Bf16,
    Nvfp4,
}

impl fmt::Display for StateFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl StateFormat {
    pub const ALL: [StateFormat; 3] = [StateFormat::F32, StateFormat::Bf16, StateFormat::Nvfp4];

    pub fn name(self) -> &'static str {
        match self {
            StateFormat::F32 => "f32",
            StateFormat::Bf16 => "bf16",
            StateFormat::Nvfp4 => "nvfp4",
        }
    }

    /// Accepts ninfer's spelling `fp32` too.
    pub fn parse(s: &str) -> Result<Self> {
        let s = if s == "fp32" { "f32" } else { s };
        StateFormat::ALL
            .into_iter()
            .find(|f| f.name() == s)
            .ok_or_else(|| Error::Config(format!("unknown state format '{s}' (f32|bf16|nvfp4)")))
    }

    /// Whether `n` values can be stored (whole rows).
    pub fn check_len(self, n: usize) -> Result<()> {
        if n.is_multiple_of(STATE_ROW) {
            Ok(())
        } else {
            Err(Error::Shape(format!(
                "state of {n} values is not whole rows of {STATE_ROW}"
            )))
        }
    }

    /// f32 slots of a packed state of `n` values (a multiple of [`STATE_ROW`]).
    pub fn slots(self, n: usize) -> usize {
        match self {
            StateFormat::F32 => n,
            StateFormat::Bf16 => n / 2,
            // n/2 code bytes + n/16 scale bytes + 4 n/128 divisor bytes = 19 n / 32 bytes.
            StateFormat::Nvfp4 => 19 * n / 128,
        }
    }
}

/// Encodes `x` (whole rows) into `out` (at least `4 · fmt.slots(x.len())` bytes).
pub fn pack(fmt: StateFormat, x: &[f32], out: &mut [u8]) -> Result<()> {
    let n = x.len();
    fmt.check_len(n)?;
    let out = out
        .get_mut(..4 * fmt.slots(n))
        .ok_or_else(|| Error::Shape("state pack: output too small".into()))?;
    match fmt {
        StateFormat::F32 => {
            for (o, v) in out.as_chunks_mut::<4>().0.iter_mut().zip(x) {
                *o = v.to_le_bytes();
            }
        }
        StateFormat::Bf16 => {
            for (o, &v) in out.as_chunks_mut::<2>().0.iter_mut().zip(x) {
                *o = f32_to_bf16(v).to_le_bytes();
            }
        }
        StateFormat::Nvfp4 => {
            let (codes, rest) = out.split_at_mut(n / 2);
            let (scales, divs) = rest.split_at_mut(n / 16);
            codes.fill(0);
            for (r, row) in x.as_chunks::<STATE_ROW>().0.iter().enumerate() {
                let d = row_divisor(row);
                divs[4 * r..4 * r + 4].copy_from_slice(&d.to_le_bytes());
                for (g, group) in row.as_chunks::<16>().0.iter().enumerate() {
                    let ga = group.iter().fold(0f32, |m, v| m.max(v.abs()));
                    let sc = f32_to_e4m3fn(d * ga / 6.0);
                    let s = e4m3fn_to_f32(sc);
                    let gi = r * (STATE_ROW / 16) + g;
                    scales[gi] = sc;
                    for (j, &v) in group.iter().enumerate() {
                        let c = if s == 0.0 { 0 } else { f32_to_e2m1(v * d / s) };
                        let i = gi * 16 + j;
                        codes[i / 2] |= if i.is_multiple_of(2) { c } else { c << 4 };
                    }
                }
            }
        }
    }
    Ok(())
}

/// Decodes a packed state of `out.len()` values (whole rows) from `bytes`.
pub fn unpack(fmt: StateFormat, bytes: &[u8], out: &mut [f32]) -> Result<()> {
    let n = out.len();
    fmt.check_len(n)?;
    let bytes = bytes
        .get(..4 * fmt.slots(n))
        .ok_or_else(|| Error::Shape("state unpack: input too small".into()))?;
    match fmt {
        StateFormat::F32 => {
            for (o, b) in out.iter_mut().zip(bytes.as_chunks::<4>().0) {
                *o = f32::from_le_bytes(*b);
            }
        }
        StateFormat::Bf16 => {
            for (o, b) in out.iter_mut().zip(bytes.as_chunks::<2>().0) {
                *o = bf16_to_f32(u16::from_le_bytes(*b));
            }
        }
        StateFormat::Nvfp4 => {
            let (codes, rest) = bytes.split_at(n / 2);
            let (scales, divs) = rest.split_at(n / 16);
            for (i, o) in out.iter_mut().enumerate() {
                let r = i / STATE_ROW;
                let d = f32::from_le_bytes(divs[4 * r..4 * r + 4].try_into().expect("4 bytes"));
                let s = e4m3fn_to_f32(scales[i / 16]);
                let c = if i % 2 == 0 {
                    codes[i / 2] & 15
                } else {
                    codes[i / 2] >> 4
                };
                *o = if d == 0.0 || s == 0.0 {
                    0.0
                } else {
                    e2m1_to_f32(c) * s / d
                };
            }
        }
    }
    Ok(())
}

/// `2688 / amax` of a row, or 0 for a zero row or one too small for a finite divisor.
fn row_divisor(row: &[f32]) -> f32 {
    let amax = row.iter().fold(0f32, |m, v| m.max(v.abs()));
    let d = if amax > 0.0 { 2688.0 / amax } else { 0.0 };
    if d.is_finite() {
        d
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rows with very different magnitudes (as GDN heads decay at different rates), an outlier,
    /// a zero row and a row too small for a finite divisor.
    fn sample() -> Vec<f32> {
        let mut x: Vec<f32> = (0..6 * STATE_ROW)
            .map(|i| {
                let mag = [1e-3f32, 0.1, 1.0, 30.0, 0.0, 1e-38][i / STATE_ROW];
                ((i * 37 % 101) as f32 - 50.0) / 50.0 * mag
            })
            .collect();
        x[2 * STATE_ROW + 5] = 40.0;
        x
    }

    fn round(fmt: StateFormat, x: &[f32]) -> Vec<f32> {
        let mut bytes = vec![0u8; 4 * fmt.slots(x.len())];
        pack(fmt, x, &mut bytes).unwrap();
        let mut y = vec![0f32; x.len()];
        unpack(fmt, &bytes, &mut y).unwrap();
        y
    }

    fn rel_err(a: &[f32], b: &[f32]) -> f64 {
        let (mut e, mut n) = (0f64, 0f64);
        for (x, y) in a.iter().zip(b) {
            e += f64::from(x - y).powi(2);
            n += f64::from(*y).powi(2);
        }
        if n == 0.0 {
            e.sqrt()
        } else {
            (e / n).sqrt()
        }
    }

    #[test]
    fn f32_round_trips_and_rows_must_be_whole() {
        let x = sample();
        assert_eq!(round(StateFormat::F32, &x), x);
        let mut out = vec![0u8; 400];
        assert!(pack(StateFormat::Bf16, &x[..100], &mut out).is_err());
        assert!(pack(StateFormat::Nvfp4, &x[..128], &mut out[..10]).is_err());
    }

    #[test]
    fn errors_are_per_row_and_ordered() {
        let x = sample();
        let (b, f) = (round(StateFormat::Bf16, &x), round(StateFormat::Nvfp4, &x));
        for r in 0..4 {
            let rows = r * STATE_ROW..(r + 1) * STATE_ROW;
            // Per-row divisors: the 1e-3 row is as accurate, relatively, as the 30 one.
            assert!(rel_err(&b[rows.clone()], &x[rows.clone()]) < 4e-3);
            assert!(rel_err(&f[rows.clone()], &x[rows.clone()]) < 0.2);
        }
        assert!(rel_err(&b, &x) < rel_err(&f, &x));
        // Zero and sub-divisor rows decode to zero.
        assert!(f[4 * STATE_ROW..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn nvfp4_values_are_codes_times_scales_over_the_divisor() {
        let x = sample();
        let n = x.len();
        let mut bytes = vec![0u8; 4 * StateFormat::Nvfp4.slots(n)];
        pack(StateFormat::Nvfp4, &x, &mut bytes).unwrap();
        let y = round(StateFormat::Nvfp4, &x);
        // Row 3 (magnitude 30): its largest group's scale is E4M3's 448.
        let (r, g) = (3, 3 * STATE_ROW / 16);
        let scales = &bytes[n / 2..n / 2 + n / 16];
        assert!(scales[g..g + STATE_ROW / 16].contains(&0x7e));
        let d = f32::from_le_bytes(bytes[n / 2 + n / 16 + 4 * r..][..4].try_into().unwrap());
        let amax = x[r * STATE_ROW..(r + 1) * STATE_ROW]
            .iter()
            .fold(0f32, |m, v| m.max(v.abs()));
        assert_eq!(d, 2688.0 / amax);
        for (i, v) in y[r * STATE_ROW..(r + 1) * STATE_ROW].iter().enumerate() {
            let s = e4m3fn_to_f32(scales[g + i / 16]);
            let code = v * d / s;
            assert!((code - e2m1_to_f32(f32_to_e2m1(code))).abs() < 1e-5, "{v}");
        }
    }

    #[test]
    fn packed_sizes() {
        let n = 48 * 128 * 128;
        assert_eq!(4 * StateFormat::F32.slots(n), 3_145_728);
        assert_eq!(4 * StateFormat::Bf16.slots(n), 1_572_864);
        assert_eq!(4 * StateFormat::Nvfp4.slots(n), 393_216 + 49_152 + 24_576);
        assert_eq!(StateFormat::parse("fp32").unwrap(), StateFormat::F32);
    }
}
