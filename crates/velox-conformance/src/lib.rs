//! Conformance suite: runs every op and an end-to-end decoder stack on a backend and compares
//! against the CPU reference. A hardware branch is releasable only when this passes on its cards.
//!
//! Shapes deliberately include awkward sizes (odd K, non-power-of-two head_dim, long positions,
//! row views into a parent weight) because tuned kernels usually specialise on aligned shapes and
//! must fall back correctly.

use velox_core::backend::{
    AttnShape, Backend, ConvShape, GdnShape, NormKind, RopeShape, WeightView,
};
use velox_core::format::{align256, f32_to_bf16, WeightFormat};
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

/// A random valid encoding of a `[n, k]` matrix in `fmt` (finite codes, positive scales).
pub fn random_weight(rng: &mut Rng, fmt: WeightFormat, n: usize, k: usize) -> Vec<u8> {
    let byte = |rng: &mut Rng| (rng.next_u64() >> 56) as u8;
    match fmt {
        WeightFormat::F32 => rng
            .vec(n * k, 0.5)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        WeightFormat::Bf16 => rng
            .vec(n * k, 0.5)
            .iter()
            .flat_map(|&v| f32_to_bf16(v).to_le_bytes())
            .collect(),
        WeightFormat::Fp8RowBf16 => {
            let mut p: Vec<u8> = (0..n * k)
                .map(|_| {
                    let b = byte(rng);
                    if b & 0x7f == 0x7f {
                        b & 0xfe // avoid the NaN words
                    } else {
                        b
                    }
                })
                .collect();
            p.resize(align256(n * k), 0);
            for _ in 0..n {
                let scale = 0.001 + 0.01 * (rng.uniform() + 1.0);
                p.extend_from_slice(&f32_to_bf16(scale).to_le_bytes());
            }
            p
        }
        WeightFormat::Nvfp4 => {
            let mut p: Vec<u8> = (0..n * k / 2).map(|_| byte(rng)).collect();
            p.resize(align256(n * k / 2), 0);
            // Positive E4M3FN scales with exponent 4..11 (0.125 .. 30).
            p.extend((0..n * k / 16).map(|_| (((byte(rng) % 8) + 4) << 3) | (byte(rng) & 7)));
            p.extend_from_slice(&96.0f32.to_le_bytes());
            p
        }
    }
}

struct Suite<'a, B: Backend> {
    b: &'a B,
    rng: Rng,
    results: Vec<CaseResult>,
}

impl<B: Backend> Suite<'_, B> {
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

    /// `linear` on rows `[begin, begin + rows)` of a random `[n, k]` weight in `fmt`.
    fn linear(
        &mut self,
        fmt: WeightFormat,
        n: usize,
        k: usize,
        begin: usize,
        rows: usize,
        m: usize,
    ) -> Result<()> {
        let bytes = random_weight(&mut self.rng, fmt, n, k);
        let x = self.rng.vec(m * k, 1.0);
        let wc = CpuBackend.upload_bytes(&bytes)?;
        let mut want = vec![0.0; m * rows];
        CpuBackend.linear(
            &x,
            &WeightView::full(&wc, fmt, n, k).rows(begin, rows),
            &mut want,
            m,
        )?;
        let wd = self.b.upload_bytes(&bytes)?;
        let xd = self.b.upload(&x)?;
        let mut yd = self.b.alloc(m * rows)?;
        self.b.linear(
            &xd,
            &WeightView::full(&wd, fmt, n, k).rows(begin, rows),
            &mut yd,
            m,
        )?;
        let got = fetch(self.b, &yd, m * rows)?;
        let name = format!("linear {} [{n},{k}] rows {begin}+{rows} m={m}", fmt.name());
        self.record(name, Some(Op::Linear), &got, &want, 2e-4);
        Ok(())
    }

    fn embed(&mut self, fmt: WeightFormat, n: usize, k: usize, ids: &[u32]) -> Result<()> {
        let bytes = random_weight(&mut self.rng, fmt, n, k);
        let wc = CpuBackend.upload_bytes(&bytes)?;
        let mut want = vec![0.0; ids.len() * k];
        CpuBackend.embed(ids, &WeightView::full(&wc, fmt, n, k), &mut want)?;
        let wd = self.b.upload_bytes(&bytes)?;
        let mut od = self.b.alloc(ids.len() * k)?;
        self.b
            .embed(ids, &WeightView::full(&wd, fmt, n, k), &mut od)?;
        let got = fetch(self.b, &od, ids.len() * k)?;
        self.record(
            format!("embed {} [{n},{k}] ids {ids:?}", fmt.name()),
            Some(Op::Embed),
            &got,
            &want,
            1e-6,
        );
        Ok(())
    }

    fn rms_norm(&mut self, rows: usize, dim: usize, kind: NormKind) -> Result<()> {
        let x = self.rng.vec(rows * dim, 3.0);
        let w = self.rng.vec(dim, 1.0);
        let mut want = vec![0.0; rows * dim];
        CpuBackend.rms_norm(&x, &w, &mut want, rows, dim, 1e-6, kind)?;
        let (xd, wd) = (self.b.upload(&x)?, self.b.upload(&w)?);
        let mut yd = self.b.alloc(rows * dim)?;
        self.b.rms_norm(&xd, &wd, &mut yd, rows, dim, 1e-6, kind)?;
        let got = fetch(self.b, &yd, rows * dim)?;
        self.record(
            format!("rms_norm {kind:?} rows={rows} dim={dim}"),
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
            "rope t={} h={} d={} r={} pos0={} theta={}",
            s.tokens, s.heads, s.head_dim, s.rotary_dim, s.pos0, s.theta
        );
        self.record(name, Some(Op::Rope), &got, &want, 1e-5);
        Ok(())
    }

    fn elementwise(&mut self, len: usize) -> Result<()> {
        let (g, u) = (self.rng.vec(len, 4.0), self.rng.vec(len, 1.0));
        let ud = self.b.upload(&u)?;
        for (op, name) in [
            (Op::SiluMul, "silu_mul"),
            (Op::SigmoidMul, "sigmoid_mul"),
            (Op::Add, "add"),
        ] {
            let mut want = g.clone();
            let mut d = self.b.upload(&g)?;
            match op {
                Op::SiluMul => {
                    CpuBackend.silu_mul(&mut want, &u, len)?;
                    self.b.silu_mul(&mut d, &ud, len)?;
                }
                Op::SigmoidMul => {
                    CpuBackend.sigmoid_mul(&mut want, &u, len)?;
                    self.b.sigmoid_mul(&mut d, &ud, len)?;
                }
                _ => {
                    CpuBackend.add(&mut want, &u, len)?;
                    self.b.add(&mut d, &ud, len)?;
                }
            }
            let got = fetch(self.b, &d, len)?;
            let tol = if op == Op::Add { 1e-6 } else { 1e-5 };
            self.record(format!("{name} len={len}"), Some(op), &got, &want, tol);
        }
        Ok(())
    }

    fn copy(&mut self, len: usize, dst_len: usize, offset: usize) -> Result<()> {
        // Source carries a 7-element prefix so the source offset is exercised too.
        let src = self.rng.vec(len + 7, 1.0);
        let dst = self.rng.vec(dst_len, 1.0);
        let mut want = dst.clone();
        CpuBackend.copy(&src, 7, &mut want, offset, len)?;
        let sd = self.b.upload(&src)?;
        let mut dd = self.b.upload(&dst)?;
        self.b.copy(&sd, 7, &mut dd, offset, len)?;
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

    /// Two calls (prefill chunk, then one token) so the rolling state is exercised.
    fn conv1d(&mut self, channels: usize, width: usize, chunks: [usize; 2]) -> Result<()> {
        let w = self.rng.vec(width * channels, 0.5);
        let state0 = self.rng.vec((width - 1) * channels, 1.0);
        let (wd, mut sd) = (self.b.upload(&w)?, self.b.upload(&state0)?);
        let mut sc = state0;
        for (i, tokens) in chunks.into_iter().enumerate() {
            let s = ConvShape {
                tokens,
                channels,
                width,
            };
            let x = self.rng.vec(tokens * channels, 1.0);
            let mut want = vec![0.0; tokens * channels];
            CpuBackend.conv1d_silu(&x, &w, &mut sc, &mut want, s)?;
            let xd = self.b.upload(&x)?;
            let mut od = self.b.alloc(tokens * channels)?;
            self.b.conv1d_silu(&xd, &wd, &mut sd, &mut od, s)?;
            let mut got = fetch(self.b, &od, tokens * channels)?;
            got.extend(fetch(self.b, &sd, (width - 1) * channels)?);
            want.extend_from_slice(&sc);
            let name =
                format!("conv1d_silu c={channels} w={width} call {i} ({tokens} tokens) + state");
            self.record(name, Some(Op::Conv1dSilu), &got, &want, 1e-5);
        }
        Ok(())
    }

    /// Two calls so the recurrent state carries over; output and final state are compared.
    fn gdn(
        &mut self,
        k_heads: usize,
        v_heads: usize,
        k_dim: usize,
        v_dim: usize,
        chunks: [usize; 2],
    ) -> Result<()> {
        let a_log = self.rng.vec(v_heads, 1.0);
        let dt_bias = self.rng.vec(v_heads, 1.0);
        let state_len = v_heads * v_dim * k_dim;
        let (ad, dd) = (self.b.upload(&a_log)?, self.b.upload(&dt_bias)?);
        let mut sd = self.b.alloc(state_len)?;
        let mut sc = vec![0.0; state_len];
        for (i, tokens) in chunks.into_iter().enumerate() {
            let s = GdnShape {
                tokens,
                k_heads,
                v_heads,
                k_dim,
                v_dim,
            };
            let qkv = self.rng.vec(tokens * s.qkv_width(), 1.0);
            let ab = self.rng.vec(tokens * 2 * v_heads, 2.0);
            let out_len = tokens * v_heads * v_dim;
            let mut want = vec![0.0; out_len];
            CpuBackend.gated_delta_net(&qkv, &ab, &a_log, &dt_bias, &mut sc, &mut want, s)?;
            let (qd, abd) = (self.b.upload(&qkv)?, self.b.upload(&ab)?);
            let mut od = self.b.alloc(out_len)?;
            self.b
                .gated_delta_net(&qd, &abd, &ad, &dd, &mut sd, &mut od, s)?;
            let mut got = fetch(self.b, &od, out_len)?;
            got.extend(fetch(self.b, &sd, state_len)?);
            want.extend_from_slice(&sc);
            let name = format!("gated_delta_net heads {k_heads}/{v_heads} d {k_dim}/{v_dim} call {i} ({tokens} tokens) + state");
            self.record(name, Some(Op::GatedDeltaNet), &got, &want, 1e-4);
        }
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
    s.linear(WeightFormat::F32, 8, 40, 0, 8, 3)?;
    s.linear(WeightFormat::Bf16, 33, 130, 5, 20, 2)?;
    s.linear(WeightFormat::Fp8RowBf16, 96, 300, 10, 40, 1)?;
    s.linear(WeightFormat::Fp8RowBf16, 64, 512, 0, 64, 9)?;
    // m >= 16 takes the tiled path, with ragged edges in every dimension.
    s.linear(WeightFormat::Fp8RowBf16, 200, 300, 17, 150, 70)?;
    s.linear(WeightFormat::Nvfp4, 256, 320, 0, 256, 33)?;
    s.linear(WeightFormat::Bf16, 90, 70, 0, 90, 16)?;
    s.linear(WeightFormat::Nvfp4, 256, 128, 0, 256, 1)?;
    s.linear(WeightFormat::Nvfp4, 384, 192, 128, 200, 5)?;
    s.embed(WeightFormat::Fp8RowBf16, 50, 64, &[3, 0, 49, 3])?;
    s.embed(WeightFormat::Bf16, 10, 130, &[9, 1])?;
    s.embed(WeightFormat::Nvfp4, 128, 64, &[127, 0, 64])?;
    s.rms_norm(1, 4096, NormKind::Plain)?;
    s.rms_norm(5, 130, NormKind::Plain)?;
    s.rms_norm(6, 256, NormKind::Offset)?;
    s.rope(RopeShape {
        tokens: 3,
        heads: 4,
        head_dim: 64,
        rotary_dim: 64,
        pos0: 0,
        theta: 10_000.0,
    })?;
    s.rope(RopeShape {
        tokens: 2,
        heads: 8,
        head_dim: 128,
        rotary_dim: 128,
        pos0: 100_000,
        theta: 1_000_000.0,
    })?;
    // Qwen3.5 attention: partial rotary over 64 of 256 dims, theta 1e7.
    s.rope(RopeShape {
        tokens: 3,
        heads: 6,
        head_dim: 256,
        rotary_dim: 64,
        pos0: 50_000,
        theta: 10_000_000.0,
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
        AttnShape {
            tokens: 4,
            heads: 24,
            kv_heads: 4,
            head_dim: 256,
            pos0: 20,
        },
    ] {
        s.attention(a)?;
    }
    s.conv1d(70, 4, [5, 1])?;
    s.conv1d(1024, 4, [1, 1])?;
    s.gdn(1, 3, 32, 32, [4, 1])?;
    // Qwen3.5 GDN head geometry (16/48 heads scaled down to 2/6), Dk = Dv = 128.
    s.gdn(2, 6, 128, 128, [6, 1])?;
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

    #[test]
    fn random_weights_have_layout_sizes() {
        let mut rng = Rng::new(1);
        for (f, n, k) in [
            (WeightFormat::Fp8RowBf16, 96, 300),
            (WeightFormat::Nvfp4, 256, 128),
            (WeightFormat::Bf16, 3, 5),
        ] {
            assert_eq!(
                random_weight(&mut rng, f, n, k).len(),
                f.payload_bytes(n, k),
                "{f:?}"
            );
        }
    }
}
