//! CPU reference backend.
//!
//! Straightforward loops with f64 accumulation. It is the oracle the conformance suite compares
//! every device backend against, so clarity wins over speed here: never optimise it in ways that
//! change its numerics.

use velox_core::backend::{AttnShape, Backend, BackendKind, RopeShape};
use velox_core::{check, DeviceProfile, Op, Result};

#[derive(Default, Clone, Copy)]
pub struct CpuBackend;

impl Backend for CpuBackend {
    type Buffer = Vec<f32>;

    fn kind(&self) -> BackendKind {
        BackendKind::Cpu
    }

    fn describe(&self) -> String {
        "cpu (reference)".into()
    }

    fn profile(&self) -> Option<&'static DeviceProfile> {
        None
    }

    fn native_ops(&self) -> Vec<Op> {
        Op::ALL.to_vec()
    }

    fn alloc(&self, len: usize) -> Result<Vec<f32>> {
        Ok(vec![0.0; len])
    }

    fn upload(&self, data: &[f32]) -> Result<Vec<f32>> {
        Ok(data.to_vec())
    }

    fn download(&self, buf: &Vec<f32>, out: &mut [f32]) -> Result<()> {
        check::elementwise("download", &[buf.len()], out.len())?;
        out.copy_from_slice(&buf[..out.len()]);
        Ok(())
    }

    fn len(&self, buf: &Vec<f32>) -> usize {
        buf.len()
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }

    fn matmul(
        &self,
        x: &Vec<f32>,
        w: &Vec<f32>,
        y: &mut Vec<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        check::matmul(x.len(), w.len(), y.len(), m, n, k)?;
        for i in 0..m {
            let xr = &x[i * k..(i + 1) * k];
            for j in 0..n {
                let wr = &w[j * k..(j + 1) * k];
                y[i * n + j] = xr
                    .iter()
                    .zip(wr)
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum::<f64>() as f32;
            }
        }
        Ok(())
    }

    fn rms_norm(
        &self,
        x: &Vec<f32>,
        weight: &Vec<f32>,
        y: &mut Vec<f32>,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<()> {
        check::rms_norm(x.len(), weight.len(), y.len(), rows, dim)?;
        for r in 0..rows {
            let xr = &x[r * dim..(r + 1) * dim];
            let ms = xr.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / dim as f64;
            let inv = 1.0 / (ms + f64::from(eps)).sqrt();
            for c in 0..dim {
                y[r * dim + c] = (f64::from(xr[c]) * inv * f64::from(weight[c])) as f32;
            }
        }
        Ok(())
    }

    fn rope(&self, x: &mut Vec<f32>, s: RopeShape) -> Result<()> {
        check::rope(x.len(), &s)?;
        let half = s.head_dim / 2;
        for t in 0..s.tokens {
            let pos = (s.pos0 + t) as f64;
            for h in 0..s.heads {
                let base = (t * s.heads + h) * s.head_dim;
                for i in 0..half {
                    let freq = f64::from(s.theta).powf(-2.0 * i as f64 / s.head_dim as f64);
                    let (sin, cos) = (pos * freq).sin_cos();
                    let a = f64::from(x[base + i]);
                    let b = f64::from(x[base + i + half]);
                    x[base + i] = (a * cos - b * sin) as f32;
                    x[base + i + half] = (a * sin + b * cos) as f32;
                }
            }
        }
        Ok(())
    }

    fn silu_mul(&self, gate: &mut Vec<f32>, up: &Vec<f32>, len: usize) -> Result<()> {
        check::elementwise("silu_mul", &[gate.len(), up.len()], len)?;
        for (g, &u) in gate[..len].iter_mut().zip(&up[..len]) {
            let v = f64::from(*g);
            *g = (v / (1.0 + (-v).exp()) * f64::from(u)) as f32;
        }
        Ok(())
    }

    fn add(&self, acc: &mut Vec<f32>, x: &Vec<f32>, len: usize) -> Result<()> {
        check::elementwise("add", &[acc.len(), x.len()], len)?;
        for (a, &b) in acc[..len].iter_mut().zip(&x[..len]) {
            *a += b;
        }
        Ok(())
    }

    fn copy(
        &self,
        src: &Vec<f32>,
        dst: &mut Vec<f32>,
        dst_offset: usize,
        len: usize,
    ) -> Result<()> {
        check::copy(src.len(), dst.len(), dst_offset, len)?;
        dst[dst_offset..dst_offset + len].copy_from_slice(&src[..len]);
        Ok(())
    }

    fn attention(
        &self,
        q: &Vec<f32>,
        k: &Vec<f32>,
        v: &Vec<f32>,
        out: &mut Vec<f32>,
        s: AttnShape,
    ) -> Result<()> {
        check::attention(q.len(), k.len(), v.len(), out.len(), &s)?;
        let d = s.head_dim;
        let group = s.heads / s.kv_heads;
        let scale = f64::from(s.scale());
        let mut scores = vec![0f64; s.kv_len()];
        for t in 0..s.tokens {
            let visible = s.pos0 + t + 1;
            for h in 0..s.heads {
                let kvh = h / group;
                let qv = &q[(t * s.heads + h) * d..][..d];
                for (j, sc) in scores[..visible].iter_mut().enumerate() {
                    let kv = &k[(j * s.kv_heads + kvh) * d..][..d];
                    *sc = qv
                        .iter()
                        .zip(kv)
                        .map(|(&a, &b)| f64::from(a) * f64::from(b))
                        .sum::<f64>()
                        * scale;
                }
                let max = scores[..visible]
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);
                let mut denom = 0.0;
                for sc in &mut scores[..visible] {
                    *sc = (*sc - max).exp();
                    denom += *sc;
                }
                let o = &mut out[(t * s.heads + h) * d..][..d];
                for (c, oc) in o.iter_mut().enumerate() {
                    let acc: f64 = (0..visible)
                        .map(|j| scores[j] * f64::from(v[(j * s.kv_heads + kvh) * d + c]))
                        .sum();
                    *oc = (acc / denom) as f32;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_small() {
        let b = CpuBackend;
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2,3]
        let w = vec![1.0, 0.0, 0.0, 0.0, 1.0, 1.0]; // [2,3]
        let mut y = vec![0.0; 4];
        b.matmul(&x, &w, &mut y, 2, 2, 3).unwrap();
        assert_eq!(y, vec![1.0, 5.0, 4.0, 11.0]);
    }

    #[test]
    fn single_key_attention_returns_value() {
        let b = CpuBackend;
        let s = AttnShape {
            tokens: 1,
            heads: 2,
            kv_heads: 1,
            head_dim: 2,
            pos0: 0,
        };
        let q = vec![1.0, 2.0, -3.0, 0.5];
        let k = vec![0.3, 0.7];
        let v = vec![4.0, -1.0];
        let mut out = vec![0.0; 4];
        b.attention(&q, &k, &v, &mut out, s).unwrap();
        assert_eq!(out, vec![4.0, -1.0, 4.0, -1.0]);
    }

    #[test]
    fn rope_at_position_zero_is_identity() {
        let b = CpuBackend;
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        b.rope(
            &mut x,
            RopeShape {
                tokens: 1,
                heads: 1,
                head_dim: 4,
                pos0: 0,
                theta: 10_000.0,
            },
        )
        .unwrap();
        assert_eq!(x, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
