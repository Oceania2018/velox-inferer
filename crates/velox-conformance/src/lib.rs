//! Conformance suite: runs every op and an end-to-end decoder stack on a backend and compares
//! against the CPU reference. A hardware branch is releasable only when this passes on its cards.
//!
//! Shapes deliberately include awkward sizes (odd K, non-power-of-two head_dim, long positions)
//! because tuned kernels usually specialise on aligned shapes and must fall back correctly.

use velox_core::backend::{AttnShape, Backend, RopeShape};
use velox_core::model::{HostLayer, Stack};
use velox_core::rng::Rng;
use velox_core::{ModelSpec, Op, Result};
use velox_cpu::CpuBackend;

#[derive(Clone, Debug)]
pub struct CaseResult {
    pub name: String,
    pub op: Option<Op>,
    /// Max of |got - want| / (1 + |want|); infinite for NaN or length mismatch.
    pub max_err: f64,
    pub tol: f64,
}

impl CaseResult {
    pub fn passed(&self) -> bool {
        self.max_err <= self.tol
    }
}

pub fn max_err(got: &[f32], want: &[f32]) -> f64 {
    if got.len() != want.len() {
        return f64::INFINITY;
    }
    got.iter()
        .zip(want)
        .map(|(&g, &w)| {
            let e = (f64::from(g) - f64::from(w)).abs() / (1.0 + f64::from(w).abs());
            if e.is_nan() {
                f64::INFINITY
            } else {
                e
            }
        })
        .fold(0.0, f64::max)
}

fn fetch<B: Backend>(b: &B, buf: &B::Buffer, len: usize) -> Result<Vec<f32>> {
    let mut out = vec![0.0; len];
    b.synchronize()?;
    b.download(buf, &mut out)?;
    Ok(out)
}

struct Suite<'a, B: Backend> {
    b: &'a B,
    rng: Rng,
    results: Vec<CaseResult>,
}

impl<'a, B: Backend> Suite<'a, B> {
    fn record(&mut self, name: String, op: Option<Op>, got: &[f32], want: &[f32], tol: f64) {
        self.results.push(CaseResult {
            name,
            op,
            max_err: max_err(got, want),
            tol,
        });
    }

    fn matmul(&mut self, m: usize, n: usize, k: usize) -> Result<()> {
        let x = self.rng.vec(m * k, 1.0);
        let w = self.rng.vec(n * k, 1.0);
        let mut want = vec![0.0; m * n];
        CpuBackend.matmul(&x, &w, &mut want, m, n, k)?;
        let (xd, wd) = (self.b.upload(&x)?, self.b.upload(&w)?);
        let mut yd = self.b.alloc(m * n)?;
        self.b.matmul(&xd, &wd, &mut yd, m, n, k)?;
        let got = fetch(self.b, &yd, m * n)?;
        self.record(
            format!("matmul m={m} n={n} k={k}"),
            Some(Op::MatMul),
            &got,
            &want,
            2e-4,
        );
        Ok(())
    }

    fn rms_norm(&mut self, rows: usize, dim: usize) -> Result<()> {
        let x = self.rng.vec(rows * dim, 3.0);
        let w = self.rng.vec(dim, 1.0);
        let mut want = vec![0.0; rows * dim];
        CpuBackend.rms_norm(&x, &w, &mut want, rows, dim, 1e-6)?;
        let (xd, wd) = (self.b.upload(&x)?, self.b.upload(&w)?);
        let mut yd = self.b.alloc(rows * dim)?;
        self.b.rms_norm(&xd, &wd, &mut yd, rows, dim, 1e-6)?;
        let got = fetch(self.b, &yd, rows * dim)?;
        self.record(
            format!("rms_norm rows={rows} dim={dim}"),
            Some(Op::RmsNorm),
            &got,
            &want,
            1e-5,
        );
        Ok(())
    }

    fn rope(&mut self, s: RopeShape) -> Result<()> {
        let len = s.tokens * s.heads * s.head_dim;
        let mut want = self.rng.vec(len, 1.0);
        let mut xd = self.b.upload(&want)?;
        CpuBackend.rope(&mut want, s)?;
        self.b.rope(&mut xd, s)?;
        let got = fetch(self.b, &xd, len)?;
        let name = format!(
            "rope t={} h={} d={} pos0={} theta={}",
            s.tokens, s.heads, s.head_dim, s.pos0, s.theta
        );
        self.record(name, Some(Op::Rope), &got, &want, 1e-5);
        Ok(())
    }

    fn elementwise(&mut self, len: usize) -> Result<()> {
        let (g, u) = (self.rng.vec(len, 4.0), self.rng.vec(len, 1.0));
        let mut want = g.clone();
        CpuBackend.silu_mul(&mut want, &u, len)?;
        let (mut gd, ud) = (self.b.upload(&g)?, self.b.upload(&u)?);
        self.b.silu_mul(&mut gd, &ud, len)?;
        let got = fetch(self.b, &gd, len)?;
        self.record(
            format!("silu_mul len={len}"),
            Some(Op::SiluMul),
            &got,
            &want,
            1e-5,
        );

        let mut want = g.clone();
        CpuBackend.add(&mut want, &u, len)?;
        let mut ad = self.b.upload(&g)?;
        self.b.add(&mut ad, &ud, len)?;
        let got = fetch(self.b, &ad, len)?;
        self.record(format!("add len={len}"), Some(Op::Add), &got, &want, 1e-6);
        Ok(())
    }

    fn copy(&mut self, len: usize, dst_len: usize, offset: usize) -> Result<()> {
        let src = self.rng.vec(len, 1.0);
        let dst = self.rng.vec(dst_len, 1.0);
        let mut want = dst.clone();
        CpuBackend.copy(&src, &mut want, offset, len)?;
        let sd = self.b.upload(&src)?;
        let mut dd = self.b.upload(&dst)?;
        self.b.copy(&sd, &mut dd, offset, len)?;
        let got = fetch(self.b, &dd, dst_len)?;
        self.record(
            format!("copy len={len} offset={offset}"),
            Some(Op::Copy),
            &got,
            &want,
            0.0,
        );
        Ok(())
    }

    fn attention(&mut self, s: AttnShape) -> Result<()> {
        let q_len = s.tokens * s.heads * s.head_dim;
        // Caches carry spare capacity past kv_len, like the model's.
        let kv_len = (s.kv_len() + 3) * s.kv_heads * s.head_dim;
        let q = self.rng.vec(q_len, 2.0);
        let k = self.rng.vec(kv_len, 2.0);
        let v = self.rng.vec(kv_len, 1.0);
        let mut want = vec![0.0; q_len];
        CpuBackend.attention(&q, &k, &v, &mut want, s)?;
        let (qd, kd, vd) = (self.b.upload(&q)?, self.b.upload(&k)?, self.b.upload(&v)?);
        let mut od = self.b.alloc(q_len)?;
        self.b.attention(&qd, &kd, &vd, &mut od, s)?;
        let got = fetch(self.b, &od, q_len)?;
        let name = format!(
            "attention t={} h={}/{} d={} pos0={}",
            s.tokens, s.heads, s.kv_heads, s.head_dim, s.pos0
        );
        self.record(name, Some(Op::Attention), &got, &want, 2e-4);
        Ok(())
    }

    /// Prefill then two decode steps through a two-layer stack, compared step by step.
    fn stack(&mut self, spec: &ModelSpec) -> Result<()> {
        let layers: Vec<HostLayer> = (0..spec.layers)
            .map(|i| HostLayer::random(spec, 100 + i as u64))
            .collect();
        let (max_seq, max_tokens) = (64, 16);
        let mut cpu = Stack::new(&CpuBackend, spec, &layers, max_seq, max_tokens)?;
        let mut dev = Stack::new(self.b, spec, &layers, max_seq, max_tokens)?;
        let mut pos = 0;
        for (step, tokens) in [7usize, 1, 1].into_iter().enumerate() {
            let x = self.rng.vec(tokens * spec.hidden, 1.0);
            let mut want = x.clone();
            cpu.forward(&CpuBackend, &mut want, tokens, pos)?;
            let mut xd = self.b.upload(&x)?;
            dev.forward(self.b, &mut xd, tokens, pos)?;
            let got = fetch(self.b, &xd, tokens * spec.hidden)?;
            self.record(
                format!("stack {} step {step} ({tokens} tokens at {pos})", spec.name),
                None,
                &got,
                &want,
                2e-3,
            );
            pos += tokens;
        }
        Ok(())
    }
}

pub fn run<B: Backend>(b: &B) -> Result<Vec<CaseResult>> {
    let mut s = Suite {
        b,
        rng: Rng::new(7),
        results: Vec::new(),
    };
    for (m, n, k) in [
        (1, 77, 130),
        (1, 512, 1024),
        (2, 300, 4096),
        (4, 96, 64),
        (7, 256, 512),
        (33, 100, 257),
        (128, 64, 96),
    ] {
        s.matmul(m, n, k)?;
    }
    s.rms_norm(1, 4096)?;
    s.rms_norm(5, 130)?;
    s.rope(RopeShape {
        tokens: 3,
        heads: 4,
        head_dim: 64,
        pos0: 0,
        theta: 10_000.0,
    })?;
    s.rope(RopeShape {
        tokens: 2,
        heads: 8,
        head_dim: 128,
        pos0: 100_000,
        theta: 1_000_000.0,
    })?;
    s.elementwise(1000)?;
    s.elementwise(4097)?;
    s.copy(300, 1000, 123)?;
    for a in [
        AttnShape {
            tokens: 1,
            heads: 8,
            kv_heads: 2,
            head_dim: 64,
            pos0: 37,
        },
        AttnShape {
            tokens: 5,
            heads: 4,
            kv_heads: 4,
            head_dim: 128,
            pos0: 0,
        },
        AttnShape {
            tokens: 3,
            heads: 6,
            kv_heads: 3,
            head_dim: 96,
            pos0: 10,
        },
        AttnShape {
            tokens: 1,
            heads: 2,
            kv_heads: 1,
            head_dim: 256,
            pos0: 1500,
        },
        AttnShape {
            tokens: 1,
            heads: 32,
            kv_heads: 8,
            head_dim: 128,
            pos0: 4095,
        },
    ] {
        s.attention(a)?;
    }
    s.stack(&ModelSpec::preset("tiny")?)?;
    Ok(s.results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_metric_flags_nan_and_length() {
        assert_eq!(max_err(&[1.0], &[1.0]), 0.0);
        assert!(max_err(&[f32::NAN], &[1.0]).is_infinite());
        assert!(max_err(&[1.0], &[1.0, 2.0]).is_infinite());
    }
}
