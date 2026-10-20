//! Reference activation quantizers. Tuned `linear` kernels may round their inputs to FP8 or NVFP4
//! (see [`ActQuant`]); the conformance suite applies the same rounding on the host, bit for bit,
//! before running the exact CPU product. Every step is the single-precision operation the device
//! performs, in the same order, so the encoded codes and scales agree exactly.

use crate::backend::ActQuant;
use crate::format::{e2m1_to_f32, e4m3fn_to_f32};

/// `f32 -> E4M3FN`, round to nearest even, saturating to ±448 (`cvt.rn.satfinite.e4m3x2.f32`).
/// NaN encodes as 0x7f.
pub fn f32_to_e4m3fn(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7f;
    }
    let sign = if x.is_sign_negative() { 0x80 } else { 0 };
    let a = x.abs();
    let code = if a < 2f32.powi(-6) {
        // Subnormal step 2^-9; rounding up to 8 lands on the smallest normal, code 0x08.
        (a * 512.0).round_ties_even() as u8
    } else {
        let e = a.log2().floor() as i32;
        // log2 can be off by one ulp near powers of two; fix the exponent so mantissa is in [1,2).
        let e = if a < 2f32.powi(e) {
            e - 1
        } else if a >= 2f32.powi(e + 1) {
            e + 1
        } else {
            e
        };
        let q = ((a / 2f32.powi(e) - 1.0) * 8.0).round_ties_even() as i32;
        let (e, q) = if q == 8 { (e + 1, 0) } else { (e, q) };
        if e > 8 {
            0x7e
        } else {
            (((e + 7) << 3) | q).min(0x7e) as u8
        }
    };
    sign | code
}

/// `f32 -> E2M1` (low nibble), round to nearest even, saturating to ±6
/// (`cvt.rn.satfinite.e2m1x2.f32`).
pub fn f32_to_e2m1(x: f32) -> u8 {
    let sign = if x.is_sign_negative() { 8 } else { 0 };
    let a = x.abs();
    let code = if a.is_nan() {
        7
    } else if a < 1.0 {
        (a * 2.0).round_ties_even() as u8 // 0, 0.5, 1.0 -> codes 0..2
    } else if a < 2.0 {
        2 + ((a - 1.0) * 2.0).round_ties_even() as u8
    } else if a < 4.0 {
        4 + (a - 2.0).round_ties_even() as u8
    } else {
        (6 + ((a - 4.0) * 0.5).round_ties_even() as u8).min(7)
    };
    sign | code
}

/// Replaces each row of `x` (`[rows, k]`) by the values a kernel with rounding `q` multiplies:
/// its codes times their scales.
pub fn quantize_activations(x: &mut [f32], k: usize, q: ActQuant) {
    match q {
        ActQuant::Exact => {}
        ActQuant::Fp8Row => {
            for row in x.chunks_mut(k) {
                let amax = row.iter().fold(0f32, |m, v| m.max(v.abs()));
                let scale = amax / 448.0;
                let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                for v in row {
                    *v = e4m3fn_to_f32(f32_to_e4m3fn(*v * inv)) * scale;
                }
            }
        }
        ActQuant::Nvfp4 { divisor } => {
            for group in x.chunks_mut(16) {
                let amax = group.iter().fold(0f32, |m, v| m.max(v.abs()));
                let s = f32_to_e4m3fn(divisor * amax / 6.0);
                let ds = e4m3fn_to_f32(s);
                for v in group {
                    *v = if s == 0 {
                        0.0
                    } else {
                        let code = f32_to_e2m1(*v * divisor / ds);
                        // Exact in f64; the kernel folds 1 / divisor into its epilogue.
                        (f64::from(e2m1_to_f32(code)) * f64::from(ds) / f64::from(divisor)) as f32
                    };
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_round_trips_every_finite_code() {
        for b in 0u8..=255 {
            if b & 0x7f == 0x7f {
                continue;
            }
            let v = e4m3fn_to_f32(b);
            let back = f32_to_e4m3fn(v);
            // ±0 both decode to zero.
            if v == 0.0 {
                assert_eq!(back & 0x7f, 0);
            } else {
                assert_eq!(back, b, "code {b:#04x} value {v}");
            }
        }
    }

    #[test]
    fn e4m3_rounds_to_nearest_even_and_saturates() {
        // Between 1.0 (0x38) and 1.125 (0x39): the midpoint goes to the even mantissa.
        assert_eq!(f32_to_e4m3fn(1.0625), 0x38);
        assert_eq!(f32_to_e4m3fn(1.0626), 0x39);
        // Between 1.125 and 1.25 the midpoint goes up to 0x3a.
        assert_eq!(f32_to_e4m3fn(1.1875), 0x3a);
        assert_eq!(f32_to_e4m3fn(1e6), 0x7e);
        assert_eq!(f32_to_e4m3fn(-1e6), 0xfe);
        // Largest subnormal rounds into the smallest normal.
        assert_eq!(f32_to_e4m3fn(2f32.powi(-6) - 2f32.powi(-11)), 0x08);
        assert_eq!(f32_to_e4m3fn(2f32.powi(-10)), 0x00); // half of the subnormal step: even
    }

    #[test]
    fn e2m1_round_trips_and_ties_to_even() {
        for c in 0u8..16 {
            let v = e2m1_to_f32(c);
            if v != 0.0 {
                assert_eq!(f32_to_e2m1(v), c);
            }
        }
        for (x, want) in [
            (0.25, 0.0),
            (0.75, 1.0),
            (1.25, 1.0),
            (1.75, 2.0),
            (2.5, 2.0),
            (3.5, 4.0),
            (5.0, 4.0),
            (5.01, 6.0),
            (100.0, 6.0),
            (-2.6, -3.0),
        ] {
            assert_eq!(e2m1_to_f32(f32_to_e2m1(x)), want, "x = {x}");
        }
    }

    #[test]
    fn nvfp4_activation_error_is_bounded_by_the_block_scale() {
        let mut x: Vec<f32> = (0..64)
            .map(|i| ((i * 37 % 29) as f32 - 14.0) * 0.37)
            .collect();
        let orig = x.clone();
        quantize_activations(&mut x, 64, ActQuant::Nvfp4 { divisor: 500.0 });
        for (g, o) in x.chunks(16).zip(orig.chunks(16)) {
            let amax = o.iter().fold(0f32, |m, v| m.max(v.abs()));
            for (a, b) in g.iter().zip(o) {
                // Half an E2M1 step at the top of the range, plus the E4M3 scale rounding.
                assert!((a - b).abs() <= amax / 6.0 * 1.07 + 1e-6, "{a} vs {b}");
            }
        }
    }
}
