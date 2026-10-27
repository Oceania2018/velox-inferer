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
        6u8.saturating_add(((a - 4.0) * 0.5).round_ties_even() as u8)
            .min(7)
    };
    sign | code
}

/// Replaces each row of `x` (`[rows, k]`) by the values a kernel with rounding `q` multiplies:
/// its codes times their scales.
pub fn quantize_activations(x: &mut [f32], k: usize, q: ActQuant) {
    match q {
        ActQuant::Exact => {}
        ActQuant::Bf16 => {
            for v in x.iter_mut() {
                *v = crate::format::bf16_to_f32(crate::format::f32_to_bf16(*v));
            }
        }
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
        ActQuant::Int8G32 => {
            for row in x.chunks_mut(k) {
                for group in row.chunks_mut(32) {
                    let amax = group.iter().fold(0f32, |m, v| m.max(v.abs()));
                    let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
                    let scale = amax / 127.0;
                    for v in group {
                        *v = (*v * inv).round_ties_even() * scale;
                    }
                }
            }
        }
        ActQuant::Int8G256 => {
            for row in x.chunks_mut(k) {
                for group in row.chunks_mut(256) {
                    for v in group.iter_mut() {
                        *v = crate::format::bf16_to_f32(crate::format::f32_to_bf16(*v));
                    }
                    let amax = group.iter().fold(0f32, |m, v| m.max(v.abs()));
                    let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
                    let scale = amax / 127.0;
                    for v in group {
                        *v = (*v * inv).round_ties_even() * scale;
                    }
                }
            }
        }
    }
}

/// `f32 -> f16 -> f32`, round to nearest even (`__float2half_rn`), overflowing to infinity.
pub fn round_f16(x: f32) -> f32 {
    if !x.is_finite() || x == 0.0 {
        return x;
    }
    let a = x.abs();
    // Spacing of f16 values around `a`: 2^(e - 10) for normals, 2^-24 for subnormals.
    let e = a.log2().floor() as i32;
    let e = if a < 2f32.powi(e) {
        e - 1
    } else if a >= 2f32.powi(e + 1) {
        e + 1
    } else {
        e
    };
    let step = 2f32.powi(e.max(-14) - 10);
    let r = (a / step).round_ties_even() * step;
    let r = if r > 65504.0 { f32::INFINITY } else { r };
    r.copysign(x)
}

/// Applies the input rounding of `p` to an `attention_full` call's operands in place.
pub fn round_full_attention_inputs(
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
    p: crate::backend::AttnPrecision,
) {
    match p {
        crate::backend::AttnPrecision::Exact => {}
        crate::backend::AttnPrecision::F16 => {
            for x in q.iter_mut().chain(k.iter_mut()).chain(v.iter_mut()) {
                *x = round_f16(*x);
            }
        }
        // No FP8 contract for attention_full; the conformance suite rejects a backend reporting it.
        crate::backend::AttnPrecision::Fp8QkF16Pv => {}
    }
}

/// Applies the input rounding of `p` to a causal attention call's operands in place:
/// `q` is `[tokens, heads, head_dim]`, `k`/`v` hold at least `kv_len` rows of `[kv_heads, head_dim]`.
pub fn round_attention_inputs(
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
    s: crate::backend::AttnShape,
    p: crate::backend::AttnPrecision,
) {
    if p == crate::backend::AttnPrecision::Exact {
        return;
    }
    let kv = s.kv_len() * s.kv_heads * s.head_dim;
    if p == crate::backend::AttnPrecision::F16 {
        for x in q
            .iter_mut()
            .chain(k[..kv].iter_mut())
            .chain(v[..kv].iter_mut())
        {
            *x = round_f16(*x);
        }
        return;
    }
    quantize_activations(q, s.head_dim, ActQuant::Fp8Row);
    quantize_activations(&mut k[..kv], s.head_dim, ActQuant::Fp8Row);
    for x in &mut v[..kv] {
        *x = round_f16(*x);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_rounding_matches_the_format() {
        use crate::format::f16_to_f32;
        // Every finite f16 value is a fixed point; midpoints go to the even neighbour.
        for h in (0u16..0x7c00).step_by(7) {
            let v = f16_to_f32(h);
            assert_eq!(round_f16(v), v);
            let next = f16_to_f32(h + 1);
            let mid = (v + next) / 2.0;
            let want = if h % 2 == 0 { v } else { next };
            assert_eq!(round_f16(mid), want, "h = {h:#06x}");
        }
        assert_eq!(round_f16(1e6), f32::INFINITY);
    }

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

    #[test]
    fn int8_activations_are_codes_times_the_group_scale() {
        // Two rows of two 256-value groups, one of them all zero.
        let mut x: Vec<f32> = (0..1024)
            .map(|i| {
                if (256..512).contains(&i) {
                    0.0
                } else {
                    ((i * 53 % 101) as f32 - 50.0) * 0.013
                }
            })
            .collect();
        let orig = x.clone();
        quantize_activations(&mut x, 512, ActQuant::Int8G256);
        for (g, o) in x.chunks(256).zip(orig.chunks(256)) {
            // The scale comes from the bf16-rounded group.
            let amax = o.iter().fold(0f32, |m, v| {
                m.max(crate::format::bf16_to_f32(crate::format::f32_to_bf16(*v)).abs())
            });
            let scale = amax / 127.0;
            for (a, b) in g.iter().zip(o) {
                let code = if scale > 0.0 { a / scale } else { 0.0 };
                assert!(
                    (code - code.round()).abs() < 1e-3 && code.abs() <= 127.0,
                    "{a}"
                );
                // Half a step, plus the bf16 rounding of the input.
                assert!(
                    (a - b).abs() <= scale * 0.5 + b.abs() / 256.0 + 1e-7,
                    "{a} vs {b}"
                );
            }
        }
        assert!(x[256..512].iter().all(|&v| v == 0.0));
    }
}
