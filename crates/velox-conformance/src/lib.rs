//! Conformance suite: runs every op and an end-to-end decoder stack on a backend and compares
//! against the CPU reference. A hardware branch is releasable only when this passes on its cards.
//!
//! Shapes deliberately include awkward sizes (odd K, non-power-of-two head_dim, long positions,
//! row views into a parent weight) because tuned kernels usually specialise on aligned shapes and
//! must fall back correctly.

use velox_core::backend::{
    ActPolicy, ActQuant, AttnPrecision, AttnShape, Backend, BlockAttnShape, ConvShape, Copy2dShape,
    CopySpec, DynConvShape, FullAttnShape, GdnReplayLayer, GdnShape, GeluKind, KvStoreShape,
    NormKind, RopeLayout, RopePosShape, RopeShape, WeightView,
};
use velox_core::format::{align256, f32_to_bf16, RowSplit, WeightFormat};
use velox_core::kv::{decode_cache, encode_cache, KvFormat};
use velox_core::model::{HostLayer, Stack};
use velox_core::quant::{
    quantize_activations, round_attention_inputs, round_full_attention_inputs,
};
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
        WeightFormat::Iq3Xxs => {
            // Every code byte is legal (grid index, sign index, 4-bit scale). Block scales are
            // exact binary16 2^-9 .. 2^-6 with random mantissas: weights of order 1e-2 .. 1.
            let blocks = n * k / 256;
            let mut p: Vec<u8> = (0..blocks * 96).map(|_| byte(rng)).collect();
            p.resize(align256(blocks * 96), 0);
            for _ in 0..blocks {
                let d = (6 + rng.next_u64() % 4) << 10 | (rng.next_u64() % 1024);
                p.extend_from_slice(&(d as u16).to_le_bytes());
            }
            p
        }
        q => {
            let rs = q.row_split().expect("row-split format");
            let codes = random_codes(rng, rs, n, k);
            encode_row_split(rs, n, k, &codes.0, &codes.1)
        }
    }
}

/// Random legal signed codes `[n, k]` and positive binary16 scales (one per logical group).
pub fn random_codes(rng: &mut Rng, rs: RowSplit, n: usize, k: usize) -> (Vec<i32>, Vec<u16>) {
    let (lo, span) = if rs.bits == 8 {
        (-127, 255)
    } else {
        (-(1 << (rs.bits - 1)), 1 << rs.bits)
    };
    let codes = (0..n * k)
        .map(|_| lo + (rng.next_u64() % span as u64) as i32)
        .collect();
    // Exact binary16 scales 2^-7 .. 2^-3 with random mantissas.
    let scales = (0..n * k.div_ceil(rs.group))
        .map(|_| ((8 + rng.next_u64() % 5) << 10 | (rng.next_u64() % 1024)) as u16)
        .collect();
    (codes, scales)
}

/// Encodes codes/scales exactly as `row_split_k128_v1` specifies (planes, nibble order, lane-major
/// high bits, zero physical padding), independently of the decoders under test.
pub fn encode_row_split(
    rs: RowSplit,
    n: usize,
    k: usize,
    codes: &[i32],
    scales: &[u16],
) -> Vec<u8> {
    let gpr = rs.groups_per_row(k);
    let logical_groups = k.div_ceil(rs.group);
    let (high_off, scale_off, total) = rs.planes(n, k);
    let mut p = vec![0u8; total];
    for r in 0..n {
        for g in 0..gpr {
            let gi = r * gpr + g;
            for lane in 0..rs.group {
                let c = g * rs.group + lane;
                let q = if c < k { codes[r * k + c] } else { 0 };
                let u = (q & ((1 << rs.bits) - 1)) as u8;
                if rs.bits == 8 {
                    p[gi * 32 + lane] = u;
                } else {
                    p[gi * 32 + lane / 2] |= (u & 0xf) << (4 * (lane % 2));
                    let hb = rs.bits as usize - 4;
                    for j in 0..hb {
                        let t = lane * hb + j;
                        p[high_off + gi * rs.high + t / 8] |= ((u >> (4 + j)) & 1) << (t % 8);
                    }
                }
            }
            let s = if g < logical_groups {
                scales[r * logical_groups + g]
            } else {
                0
            };
            p[scale_off + 2 * gi..scale_off + 2 * gi + 2].copy_from_slice(&s.to_le_bytes());
        }
    }
    p
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
        self.linear_act(fmt, n, k, begin, rows, m, ActPolicy::A16Only)
    }

    /// `linear` under an activation policy. The reference rounds the input exactly as the backend
    /// reports it does (`linear_act`), then multiplies exactly.
    #[allow(clippy::too_many_arguments)]
    fn linear_act(
        &mut self,
        fmt: WeightFormat,
        n: usize,
        k: usize,
        begin: usize,
        rows: usize,
        m: usize,
        policy: ActPolicy,
    ) -> Result<()> {
        let bytes = random_weight(&mut self.rng, fmt, n, k);
        let x = self.rng.vec(m * k, 1.0);
        let wd = self.b.upload_bytes(&bytes)?;
        let view = WeightView::full(&wd, fmt, n, k)
            .rows(begin, rows)
            .with_act(policy);
        let act = self.b.linear_act(&view, m);
        let permitted = match (act, policy) {
            (ActQuant::Exact | ActQuant::Bf16, _) => true,
            (ActQuant::Fp8Row, ActPolicy::AllowA8 | ActPolicy::AllowA4 { .. }) => true,
            (ActQuant::Nvfp4 { divisor: a }, ActPolicy::AllowA4 { divisor: b }) => a == b,
            _ => false,
        };
        let mut xq = x.clone();
        quantize_activations(&mut xq, k, act);
        let wc = CpuBackend.upload_bytes(&bytes)?;
        let mut want = vec![0.0; m * rows];
        CpuBackend.linear(
            &xq,
            &WeightView::full(&wc, fmt, n, k).rows(begin, rows),
            &mut want,
            m,
        )?;
        let xd = self.b.upload(&x)?;
        let mut yd = self.b.alloc(m * rows)?;
        self.b.linear(&xd, &view, &mut yd, m)?;
        let got = fetch(self.b, &yd, m * rows)?;
        let tag = match act {
            ActQuant::Exact => "",
            ActQuant::Fp8Row => " a8",
            ActQuant::Nvfp4 { .. } => " a4",
            ActQuant::Bf16 => " bf16",
        };
        let name = format!(
            "linear {} [{n},{k}] rows {begin}+{rows} m={m}{tag}",
            fmt.name()
        );
        if permitted {
            self.record(name, Some(Op::Linear), &got, &want, 2e-4);
        } else {
            // Rounding the policy forbids fails whatever the numbers say.
            let name = name + " (rounding not permitted)";
            self.record(name, Some(Op::Linear), &[f32::NAN], &[0.0], 0.0);
        }
        Ok(())
    }

    /// `linear_swiglu` against the oracle: gate rows then up rows of one `[2 * width + extra, k]`
    /// parent from row `begin` (`swap`: up first), or (`split`) separate parents.
    #[allow(clippy::too_many_arguments)]
    fn linear_swiglu(
        &mut self,
        fmt: WeightFormat,
        k: usize,
        width: usize,
        begin: usize,
        m: usize,
        swap: bool,
        split: bool,
    ) -> Result<()> {
        let n = (begin + 2 * width + 16).div_ceil(128) * 128; // nvfp4 needs whole 128-row tiles
        let bytes = random_weight(&mut self.rng, fmt, n, k);
        let bytes2 = random_weight(&mut self.rng, fmt, n, k);
        let x = self.rng.vec(m * k, 1.0);
        let (gb, ub) = if swap {
            (begin + width, begin)
        } else {
            (begin, begin + width)
        };
        let (wd, wd2) = (self.b.upload_bytes(&bytes)?, self.b.upload_bytes(&bytes2)?);
        let gate = WeightView::full(&wd, fmt, n, k).rows(gb, width);
        let up = WeightView::full(if split { &wd2 } else { &wd }, fmt, n, k).rows(ub, width);
        let act = self.b.linear_swiglu_act(&gate, &up, m);
        let mut xq = x.clone();
        quantize_activations(&mut xq, k, act);
        let (wc, wc2) = (
            CpuBackend.upload_bytes(&bytes)?,
            CpuBackend.upload_bytes(&bytes2)?,
        );
        let (mut g, mut u) = (vec![0.0; m * width], vec![0.0; m * width]);
        CpuBackend.linear(
            &xq,
            &WeightView::full(&wc, fmt, n, k).rows(gb, width),
            &mut g,
            m,
        )?;
        let up_parent = if split { &wc2 } else { &wc };
        CpuBackend.linear(
            &xq,
            &WeightView::full(up_parent, fmt, n, k).rows(ub, width),
            &mut u,
            m,
        )?;
        CpuBackend.silu_mul(&mut g, &u, m * width)?;
        let xd = self.b.upload(&x)?;
        let (mut od, mut sd) = (self.b.alloc(m * width)?, self.b.alloc(2 * m * width)?);
        self.b.linear_swiglu(&xd, &gate, &up, &mut od, m, &mut sd)?;
        let got = fetch(self.b, &od, m * width)?;
        let tag = match act {
            ActQuant::Exact => "",
            ActQuant::Fp8Row => " a8",
            ActQuant::Nvfp4 { .. } => " a4",
            ActQuant::Bf16 => " bf16",
        };
        let layout = if split {
            " split"
        } else if swap {
            " up|gate"
        } else {
            ""
        };
        self.record(
            format!(
                "linear_swiglu {} [{width}x2,{k}] from {begin} m={m}{layout}{tag}",
                fmt.name()
            ),
            Some(Op::LinearSwiglu),
            &got,
            &g,
            2e-4,
        );
        Ok(())
    }

    /// Splits `[rows, sum(widths)]` into consecutive column blocks (5 parts: two launches).
    fn split_cols(&mut self, rows: usize, widths: &[usize]) -> Result<()> {
        let total: usize = widths.iter().sum();
        let src = self.rng.vec(rows * total, 1.0);
        let mut want: Vec<Vec<f32>> = widths.iter().map(|w| vec![0.0; rows * w]).collect();
        let offs: Vec<usize> = widths
            .iter()
            .scan(0, |o, w| {
                let s = *o;
                *o += w;
                Some(s)
            })
            .collect();
        {
            let mut outs: Vec<(&mut Vec<f32>, usize, usize)> = want
                .iter_mut()
                .zip(&offs)
                .zip(widths)
                .map(|((b, &o), &w)| (b, o, w))
                .collect();
            CpuBackend.split_cols(&src, rows, total, &mut outs)?;
        }
        let sd = self.b.upload(&src)?;
        let mut dev: Vec<B::Buffer> = widths
            .iter()
            .map(|w| self.b.alloc(rows * w))
            .collect::<Result<_>>()?;
        {
            let mut outs: Vec<(&mut B::Buffer, usize, usize)> = dev
                .iter_mut()
                .zip(&offs)
                .zip(widths)
                .map(|((b, &o), &w)| (b, o, w))
                .collect();
            self.b.split_cols(&sd, rows, total, &mut outs)?;
        }
        let mut got = Vec::new();
        let mut all_want = Vec::new();
        for (d, (w, wv)) in dev.iter().zip(widths.iter().zip(&want)) {
            got.extend(fetch(self.b, d, rows * w)?);
            all_want.extend_from_slice(wv);
        }
        self.record(
            format!("split_cols rows={rows} {widths:?}"),
            Some(Op::SplitCols),
            &got,
            &all_want,
            0.0,
        );
        Ok(())
    }

    fn swiglu(&mut self, rows: usize, width: usize) -> Result<()> {
        let cols = 2 * width + 3;
        let src = self.rng.vec(rows * cols, 3.0);
        let (gate, up) = (width + 3, 0);
        let mut want = vec![0.0; rows * width];
        CpuBackend.swiglu(&src, rows, cols, gate, up, width, &mut want)?;
        let sd = self.b.upload(&src)?;
        let mut od = self.b.alloc(rows * width)?;
        self.b.swiglu(&sd, rows, cols, gate, up, width, &mut od)?;
        let got = fetch(self.b, &od, rows * width)?;
        self.record(
            format!("swiglu rows={rows} width={width}"),
            Some(Op::Swiglu),
            &got,
            &want,
            1e-5,
        );
        Ok(())
    }

    fn concat_cols(&mut self, rows: usize, a_cols: usize, b_cols: usize) -> Result<()> {
        let (a, bb) = (
            self.rng.vec(rows * a_cols, 1.0),
            self.rng.vec(rows * b_cols, 1.0),
        );
        let n = rows * (a_cols + b_cols);
        let mut want = vec![0.0; n];
        CpuBackend.concat_cols(&a, &bb, &mut want, rows, a_cols, b_cols)?;
        let (ad, bd) = (self.b.upload(&a)?, self.b.upload(&bb)?);
        let mut od = self.b.alloc(n)?;
        self.b
            .concat_cols(&ad, &bd, &mut od, rows, a_cols, b_cols)?;
        let got = fetch(self.b, &od, n)?;
        let name = format!("concat_cols rows={rows} {a_cols}+{b_cols}");
        self.record(name, Some(Op::ConcatCols), &got, &want, 0.0);
        Ok(())
    }

    /// `top_k` must agree exactly (indices and values) with the host selection, including ties
    /// (lower index first) and NaN (ranked last).
    fn top_k(&mut self, rows: usize, row_len: usize, limit: usize, k: usize) -> Result<()> {
        let mut x = self.rng.vec(rows * row_len, 4.0);
        // Ties and NaN: copy a value forward and poison a few entries.
        for r in 0..rows {
            let base = r * row_len;
            x[base + limit / 2] = x[base + limit / 3];
            x[base + 1] = f32::NAN;
        }
        let want = CpuBackend.top_k(&x, rows, row_len, limit, k)?;
        let xd = self.b.upload(&x)?;
        let got = self.b.top_k(&xd, rows, row_len, limit, k)?;
        let same = got.len() == want.len()
            && got.iter().zip(&want).all(|(g, w)| {
                g.len() == w.len()
                    && g.iter()
                        .zip(w)
                        .all(|(a, b)| a.0 == b.0 && (a.1 == b.1 || (a.1.is_nan() && b.1.is_nan())))
            });
        let name = format!("top_k rows={rows} len={row_len} limit={limit} k={k}");
        self.record(
            name,
            Some(Op::TopK),
            &[if same { 0.0 } else { f32::NAN }],
            &[0.0],
            0.0,
        );
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

    /// `kv_store` into caches holding valid prior rows; every cache row (written or not) is
    /// decoded on the host and must match the oracle exactly (the encoding is bit-exact).
    fn kv_store(
        &mut self,
        format: KvFormat,
        kv_heads: usize,
        head_dim: usize,
        pos0: usize,
        tokens: usize,
    ) -> Result<()> {
        let s = KvStoreShape {
            tokens,
            kv_heads,
            head_dim,
            pos0,
            format,
        };
        let n = tokens * kv_heads * head_dim;
        let (mut k, v) = (self.rng.vec(n, 3.0), self.rng.vec(n, 1.0));
        // A zero row (zero scales) and a saturating outlier.
        k[..head_dim].fill(0.0);
        k[n - 1] = 3e5;
        let rows = (pos0 + tokens + 2) * kv_heads;
        let kc0 = encode_cache(format, &self.rng.vec(rows * head_dim, 1.0), head_dim, false);
        let vc0 = encode_cache(format, &self.rng.vec(rows * head_dim, 1.0), head_dim, true);
        let (mut kc, mut vc) = (
            CpuBackend.upload_bytes(&kc0)?,
            CpuBackend.upload_bytes(&vc0)?,
        );
        CpuBackend.kv_store(&k, &v, &mut kc, &mut vc, s)?;
        let (kd, vd) = (self.b.upload(&k)?, self.b.upload(&v)?);
        let (mut kcd, mut vcd) = (self.b.upload_bytes(&kc0)?, self.b.upload_bytes(&vc0)?);
        self.b.kv_store(&kd, &vd, &mut kcd, &mut vcd, s)?;
        let decode = |slots: &[f32], is_v| {
            let bytes: Vec<u8> = slots.iter().flat_map(|x| x.to_le_bytes()).collect();
            decode_cache(format, &bytes, rows, head_dim, is_v)
        };
        let mut want = decode(&kc, false);
        want.extend(decode(&vc, true));
        let mut got = decode(&fetch(self.b, &kcd, kc.len())?, false);
        got.extend(decode(&fetch(self.b, &vcd, vc.len())?, true));
        let name =
            format!("kv_store {format} t={tokens} kv_heads={kv_heads} d={head_dim} pos0={pos0}");
        self.record(name, Some(Op::KvStore), &got, &want, 0.0);
        Ok(())
    }

    /// `kv_load` of rows `pos0 .. pos0 + tokens` from caches of random rows, against the oracle.
    fn kv_load(
        &mut self,
        format: KvFormat,
        kv_heads: usize,
        head_dim: usize,
        pos0: usize,
        tokens: usize,
    ) -> Result<()> {
        let s = KvStoreShape {
            tokens,
            kv_heads,
            head_dim,
            pos0,
            format,
        };
        let rows = (pos0 + tokens + 1) * kv_heads;
        let kc0 = encode_cache(format, &self.rng.vec(rows * head_dim, 3.0), head_dim, false);
        let vc0 = encode_cache(format, &self.rng.vec(rows * head_dim, 1.0), head_dim, true);
        let n = tokens * kv_heads * head_dim;
        let (kc, vc) = (
            CpuBackend.upload_bytes(&kc0)?,
            CpuBackend.upload_bytes(&vc0)?,
        );
        let (mut k, mut v) = (vec![0.0; n], vec![0.0; n]);
        CpuBackend.kv_load(&kc, &vc, &mut k, &mut v, s)?;
        let (kcd, vcd) = (self.b.upload_bytes(&kc0)?, self.b.upload_bytes(&vc0)?);
        let (mut kd, mut vd) = (self.b.alloc(n)?, self.b.alloc(n)?);
        self.b.kv_load(&kcd, &vcd, &mut kd, &mut vd, s)?;
        let mut got = fetch(self.b, &kd, n)?;
        got.extend(fetch(self.b, &vd, n)?);
        k.extend(v);
        let name =
            format!("kv_load {format} t={tokens} kv_heads={kv_heads} d={head_dim} pos0={pos0}");
        self.record(name, Some(Op::KvLoad), &got, &k, 1e-6);
        Ok(())
    }

    /// Attention over a cache in `s.kv`: the oracle decodes the cache to the rows it represents
    /// and runs the f32 reference over them.
    fn attention_kv(&mut self, s: AttnShape) -> Result<()> {
        let d = s.head_dim;
        let q_len = s.tokens * s.heads * d;
        let rows = (s.kv_len() + 3) * s.kv_heads;
        let q = self.rng.vec(q_len, 2.0);
        let kb = encode_cache(s.kv, &self.rng.vec(rows * d, 2.0), d, false);
        let vb = encode_cache(s.kv, &self.rng.vec(rows * d, 1.0), d, true);
        let (mut kr, mut vr) = (
            decode_cache(s.kv, &kb, rows, d, false),
            decode_cache(s.kv, &vb, rows, d, true),
        );
        let f32_shape = AttnShape {
            kv: KvFormat::F32,
            ..s
        };
        let precision = self.b.attention_precision(s);
        let mut qr = q.clone();
        round_attention_inputs(&mut qr, &mut kr, &mut vr, f32_shape, precision);
        let mut want = vec![0.0; q_len];
        CpuBackend.attention(&qr, &kr, &vr, &mut want, f32_shape)?;
        let (qd, kd, vd) = (
            self.b.upload(&q)?,
            self.b.upload_bytes(&kb)?,
            self.b.upload_bytes(&vb)?,
        );
        let mut od = self.b.alloc(q_len)?;
        self.b.attention(&qd, &kd, &vd, &mut od, s)?;
        let got = fetch(self.b, &od, q_len)?;
        let (tag, tol) = match precision {
            AttnPrecision::Exact => ("", 2e-4),
            AttnPrecision::Fp8QkF16Pv => (" fp8qk", 2e-3),
            AttnPrecision::F16 => (" f16", 2e-3),
        };
        let name = format!(
            "attention kv {} t={} h={}/{} d={} pos0={}{tag}",
            s.kv, s.tokens, s.heads, s.kv_heads, s.head_dim, s.pos0
        );
        self.record(name, Some(Op::Attention), &got, &want, tol);
        Ok(())
    }

    fn attention(&mut self, s: AttnShape) -> Result<()> {
        if s.kv != KvFormat::F32 {
            return self.attention_kv(s);
        }
        let q_len = s.tokens * s.heads * s.head_dim;
        // Caches carry spare capacity past kv_len, like the model's.
        let kv_len = (s.kv_len() + 3) * s.kv_heads * s.head_dim;
        let q = self.rng.vec(q_len, 2.0);
        let k = self.rng.vec(kv_len, 2.0);
        let v = self.rng.vec(kv_len, 1.0);
        // The reference rounds its inputs as the backend says its kernel does. The f16 softmax
        // weights in the value product are not emulated; they set the looser tolerance.
        let precision = self.b.attention_precision(s);
        let (mut qr, mut kr, mut vr) = (q.clone(), k.clone(), v.clone());
        round_attention_inputs(&mut qr, &mut kr, &mut vr, s, precision);
        let mut want = vec![0.0; q_len];
        CpuBackend.attention(&qr, &kr, &vr, &mut want, s)?;
        let (qd, kd, vd) = (self.b.upload(&q)?, self.b.upload(&k)?, self.b.upload(&v)?);
        let mut od = self.b.alloc(q_len)?;
        self.b.attention(&qd, &kd, &vd, &mut od, s)?;
        let got = fetch(self.b, &od, q_len)?;
        let (tag, tol) = match precision {
            AttnPrecision::Exact => ("", 2e-4),
            AttnPrecision::Fp8QkF16Pv => (" fp8qk", 2e-3),
            AttnPrecision::F16 => (" f16", 2e-3),
        };
        let name = format!(
            "attention t={} h={}/{} d={} pos0={}{tag}",
            s.tokens, s.heads, s.kv_heads, s.head_dim, s.pos0
        );
        self.record(name, Some(Op::Attention), &got, &want, tol);
        Ok(())
    }

    fn attention_block(&mut self, s: BlockAttnShape) -> Result<()> {
        let q_len = s.tokens * s.heads * s.head_dim;
        let ctx_len = s.ring * s.kv_heads * s.head_dim;
        let blk_len = s.tokens * s.kv_heads * s.head_dim;
        let q = self.rng.vec(q_len, 2.0);
        let (kc, vc) = (self.rng.vec(ctx_len, 2.0), self.rng.vec(ctx_len, 1.0));
        let (kb, vb) = (self.rng.vec(blk_len, 2.0), self.rng.vec(blk_len, 1.0));
        let mut want = vec![0.0; q_len];
        CpuBackend.attention_block(&q, &kc, &vc, &kb, &vb, &mut want, s)?;
        let up = |x: &Vec<f32>| self.b.upload(x);
        let (qd, kcd, vcd, kbd, vbd) = (up(&q)?, up(&kc)?, up(&vc)?, up(&kb)?, up(&vb)?);
        let mut od = self.b.alloc(q_len)?;
        self.b
            .attention_block(&qd, &kcd, &vcd, &kbd, &vbd, &mut od, s)?;
        let got = fetch(self.b, &od, q_len)?;
        let name = format!(
            "attention_block t={} h={}/{} d={} pos0={} lo={} window={} ring={}",
            s.tokens, s.heads, s.kv_heads, s.head_dim, s.pos0, s.ctx_lo, s.window, s.ring
        );
        self.record(name, Some(Op::AttentionBlock), &got, &want, 2e-4);
        Ok(())
    }

    fn dyn_conv(
        &mut self,
        rows: usize,
        channels: usize,
        side: usize,
        accumulate: bool,
        block: usize,
    ) -> Result<()> {
        let s = DynConvShape {
            rows,
            channels,
            group: 16,
            side,
            accumulate,
            block,
        };
        let x = self.rng.vec(rows * channels, 2.0);
        let base = self.rng.vec(4 * channels, 1.0);
        let delta = self.rng.vec(rows * 4 * channels / 16, 0.5);
        let out0 = self.rng.vec(rows * channels, 1.0);
        let mut want = out0.clone();
        CpuBackend.dyn_conv(&x, &base, &delta, &mut want, s)?;
        let (xd, bd, dd) = (
            self.b.upload(&x)?,
            self.b.upload(&base)?,
            self.b.upload(&delta)?,
        );
        let mut od = self.b.upload(&out0)?;
        self.b.dyn_conv(&xd, &bd, &dd, &mut od, s)?;
        let got = fetch(self.b, &od, rows * channels)?;
        self.record(
            format!("dyn_conv rows={rows} channels={channels} side={side} acc={accumulate} block={block}"),
            Some(Op::DynConv),
            &got,
            &want,
            1e-5,
        );
        Ok(())
    }

    /// `copy_batch`: several copies from two sources into three destinations (one in system RAM
    /// when the backend has it), some into the same destination.
    fn copy_batch(&mut self) -> Result<()> {
        let (a, b) = (self.rng.vec(500, 1.0), self.rng.vec(300, 1.0));
        let lens = [200usize, 150, 400];
        let init: Vec<Vec<f32>> = lens.iter().map(|&n| self.rng.vec(n, 1.0)).collect();
        // (source, source offset, destination, destination offset, length)
        let plan = [
            (0, 7, 0, 3, 50),
            (1, 0, 0, 100, 60),
            (0, 100, 1, 0, 150),
            (1, 40, 2, 10, 200),
            (0, 300, 2, 250, 150),
        ];
        let mut want = init.clone();
        for &(s, so, d, doff, n) in &plan {
            let src = if s == 0 { &a } else { &b };
            want[d][doff..doff + n].copy_from_slice(&src[so..so + n]);
        }
        let (da, db) = (self.b.upload(&a)?, self.b.upload(&b)?);
        let mut dsts = Vec::new();
        for (i, v) in init.iter().enumerate() {
            let up = self.b.upload(v)?;
            let mut d = if i == 1 {
                self.b.alloc_host(v.len())?
            } else {
                self.b.alloc(v.len())?
            };
            self.b.copy(&up, 0, &mut d, 0, v.len())?;
            dsts.push(d);
        }
        let ops: Vec<CopySpec<'_, B::Buffer>> = plan
            .iter()
            .map(|&(s, so, d, doff, n)| CopySpec {
                src: if s == 0 { &da } else { &db },
                src_offset: so,
                dst: d,
                dst_offset: doff,
                len: n,
            })
            .collect();
        let mut refs: Vec<&mut B::Buffer> = dsts.iter_mut().collect();
        self.b.copy_batch(&mut refs, &ops)?;
        let (mut got, mut all) = (Vec::new(), Vec::new());
        for (d, w) in dsts.iter().zip(&want) {
            got.extend(fetch(self.b, d, w.len())?);
            all.extend_from_slice(w);
        }
        self.record(
            "copy_batch 5 copies, 3 destinations".into(),
            Some(Op::CopyBatch),
            &got,
            &all,
            0.0,
        );
        Ok(())
    }

    /// Virtual memory (backends that have it): one physical chunk mapped in two buffers shares
    /// contents both ways, and a remapped chunk shows its new backing.
    fn virtual_memory(&mut self) -> Result<()> {
        let c = self.b.vm_chunk_len();
        if c == 0 {
            return Ok(());
        }
        let (mut x, mut y) = (self.b.alloc_virtual(3 * c)?, self.b.alloc_virtual(2 * c)?);
        let chunks = [
            self.b.alloc_chunk()?,
            self.b.alloc_chunk()?,
            self.b.alloc_chunk()?,
        ];
        for (i, ch) in chunks.iter().enumerate() {
            self.b.map_chunk(&mut x, i, ch)?;
        }
        // y chunk 0 aliases x chunk 2, y chunk 1 aliases x chunk 0.
        self.b.map_chunk(&mut y, 0, &chunks[2])?;
        self.b.map_chunk(&mut y, 1, &chunks[0])?;
        let d = self.rng.vec(3 * c, 1.0);
        let up = self.b.upload(&d)?;
        self.b.copy(&up, 0, &mut x, 0, 3 * c)?;
        let mut got = fetch(self.b, &y, 2 * c)?;
        let mut want = [&d[2 * c..], &d[..c]].concat();
        // Writes through y show in x.
        let e = self.rng.vec(c, 1.0);
        let ue = self.b.upload(&e)?;
        self.b.copy(&ue, 0, &mut y, 0, c)?;
        got.extend(fetch(self.b, &x, 3 * c)?[2 * c..].to_vec());
        want.extend_from_slice(&e);
        // Remapping y chunk 1 to x chunk 1 backing.
        self.b.synchronize()?;
        self.b.unmap_chunk(&mut y, 1)?;
        self.b.map_chunk(&mut y, 1, &chunks[1])?;
        got.extend(fetch(self.b, &y, 2 * c)?[c..].to_vec());
        want.extend_from_slice(&d[c..2 * c]);
        self.b.synchronize()?;
        self.record(
            "virtual memory: aliased and remapped chunks".into(),
            None,
            &got,
            &want,
            0.0,
        );
        Ok(())
    }

    fn copy2d(&mut self, s: Copy2dShape) -> Result<()> {
        let src_len = s.src_offset + s.rows * s.src_stride;
        let dst_len = s.dst_offset + s.rows * s.dst_stride;
        let src = self.rng.vec(src_len, 1.0);
        let dst0 = self.rng.vec(dst_len, 1.0);
        let mut want = dst0.clone();
        CpuBackend.copy2d(&src, &mut want, s)?;
        let sd = self.b.upload(&src)?;
        let mut dd = self.b.upload(&dst0)?;
        self.b.copy2d(&sd, &mut dd, s)?;
        let got = fetch(self.b, &dd, dst_len)?;
        self.record(format!("copy2d {s:?}"), Some(Op::Copy2d), &got, &want, 0.0);
        Ok(())
    }

    /// Two calls (prefill chunk, then one token) so the rolling state is exercised.
    fn conv1d(&mut self, channels: usize, width: usize, chunks: [usize; 2]) -> Result<()> {
        let w = self.rng.vec(width * channels, 0.5);
        let state0 = self.rng.vec((width - 1) * channels, 1.0);
        let (wd, mut sd) = (self.b.upload(&w)?, self.b.upload(&state0)?);
        let mut sc = state0;
        for (i, tokens) in chunks.into_iter().enumerate() {
            // The second call starts at a row offset (a segment of a chunk).
            let row0 = 3 * i;
            let s = ConvShape {
                tokens,
                channels,
                width,
                row0,
            };
            let rows = (row0 + tokens) * channels;
            let x = self.rng.vec(rows, 1.0);
            let mut want = vec![0.0; rows];
            CpuBackend.conv1d_silu(&x, &w, &mut sc, &mut want, s)?;
            let xd = self.b.upload(&x)?;
            let mut od = self.b.alloc(rows)?;
            self.b.conv1d_silu(&xd, &wd, &mut sd, &mut od, s)?;
            want.drain(..row0 * channels);
            let mut got = fetch(self.b, &od, rows)?;
            got.drain(..row0 * channels);
            got.extend(fetch(self.b, &sd, (width - 1) * channels)?);
            want.extend_from_slice(&sc);
            let name =
                format!("conv1d_silu c={channels} w={width} call {i} ({tokens} tokens at row {row0}) + state");
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
            let row0 = 3 * i;
            let s = GdnShape {
                tokens,
                k_heads,
                v_heads,
                k_dim,
                v_dim,
                row0,
            };
            let rows = row0 + tokens;
            let qkv = self.rng.vec(rows * s.qkv_width(), 1.0);
            let ab = self.rng.vec(rows * 2 * v_heads, 2.0);
            let out_len = rows * v_heads * v_dim;
            let mut want = vec![0.0; out_len];
            CpuBackend.gated_delta_net(&qkv, &ab, &a_log, &dt_bias, &mut sc, &mut want, s)?;
            let (qd, abd) = (self.b.upload(&qkv)?, self.b.upload(&ab)?);
            let mut od = self.b.alloc(out_len)?;
            self.b
                .gated_delta_net(&qd, &abd, &ad, &dd, &mut sd, &mut od, s)?;
            let skip = row0 * v_heads * v_dim;
            want.drain(..skip);
            let mut got = fetch(self.b, &od, out_len)?;
            got.drain(..skip);
            got.extend(fetch(self.b, &sd, state_len)?);
            want.extend_from_slice(&sc);
            let name = format!("gated_delta_net heads {k_heads}/{v_heads} d {k_dim}/{v_dim} call {i} ({tokens} tokens at row {row0}) + state");
            self.record(name, Some(Op::GatedDeltaNet), &got, &want, 1e-4);
        }
        Ok(())
    }

    /// `gated_delta_net_from` / `conv1d_silu_from`: the result state lands in a second buffer and
    /// the input state stays as it was (both compared, with the outputs).
    fn state_from(
        &mut self,
        k_heads: usize,
        v_heads: usize,
        dk: usize,
        tokens: usize,
    ) -> Result<()> {
        let gs = GdnShape {
            tokens,
            k_heads,
            v_heads,
            k_dim: dk,
            v_dim: dk,
            row0: 0,
        };
        let (state_len, out_len) = (v_heads * dk * dk, tokens * v_heads * dk);
        let qkv = self.rng.vec(tokens * gs.qkv_width(), 1.0);
        let ab = self.rng.vec(tokens * 2 * v_heads, 2.0);
        let (a_log, dt_bias) = (self.rng.vec(v_heads, 1.0), self.rng.vec(v_heads, 1.0));
        let s0 = self.rng.vec(state_len, 0.1);
        let (mut want_s, mut want_o) = (vec![0.0; state_len], vec![0.0; out_len]);
        CpuBackend.gated_delta_net_from(
            &qkv,
            &ab,
            &a_log,
            &dt_bias,
            &s0,
            &mut want_s,
            &mut want_o,
            gs,
        )?;
        let (qd, abd) = (self.b.upload(&qkv)?, self.b.upload(&ab)?);
        let (ad, dd, sd) = (
            self.b.upload(&a_log)?,
            self.b.upload(&dt_bias)?,
            self.b.upload(&s0)?,
        );
        let (mut so, mut od) = (self.b.alloc(state_len)?, self.b.alloc(out_len)?);
        self.b
            .gated_delta_net_from(&qd, &abd, &ad, &dd, &sd, &mut so, &mut od, gs)?;
        let mut got = fetch(self.b, &od, out_len)?;
        got.extend(fetch(self.b, &so, state_len)?);
        got.extend(fetch(self.b, &sd, state_len)?);
        want_o.extend(want_s);
        want_o.extend_from_slice(&s0);
        let name = format!(
            "gated_delta_net_from heads {k_heads}/{v_heads} d {dk} ({tokens} tokens) + states"
        );
        self.record(name, Some(Op::GatedDeltaNet), &got, &want_o, 1e-4);

        let (channels, width) = (gs.qkv_width(), 4);
        let cs = ConvShape {
            tokens,
            channels,
            width,
            row0: 0,
        };
        let w = self.rng.vec(width * channels, 0.5);
        let c0 = self.rng.vec((width - 1) * channels, 1.0);
        let (mut want_c, mut want_y) = (vec![0.0; c0.len()], vec![0.0; tokens * channels]);
        CpuBackend.conv1d_silu_from(&qkv, &w, &c0, &mut want_c, &mut want_y, cs)?;
        let (wd, cd) = (self.b.upload(&w)?, self.b.upload(&c0)?);
        let (mut co, mut yd) = (self.b.alloc(c0.len())?, self.b.alloc(tokens * channels)?);
        self.b
            .conv1d_silu_from(&qd, &wd, &cd, &mut co, &mut yd, cs)?;
        let mut got = fetch(self.b, &yd, tokens * channels)?;
        got.extend(fetch(self.b, &co, c0.len())?);
        got.extend(fetch(self.b, &cd, c0.len())?);
        want_y.extend(want_c);
        want_y.extend_from_slice(&c0);
        let name = format!("conv1d_silu_from c={channels} w={width} ({tokens} tokens) + states");
        self.record(name, Some(Op::Conv1dSilu), &got, &want_y, 1e-5);
        Ok(())
    }

    /// `gdn_replay`: `layers` GDN layers advanced over the first `keep` rows of a `window`-row
    /// window from its recorded inputs and conv outputs; the conv and recurrent states must equal
    /// running `conv1d_silu` and `gated_delta_net` over those rows. The window starts at row
    /// `row0` of buffers whose earlier rows belong to other windows.
    fn gdn_replay(&mut self, layers: usize, window: usize, keep: usize, row0: usize) -> Result<()> {
        let (k_heads, v_heads, dk, width) = (2, 6, 64, 4);
        let gs = GdnShape {
            tokens: keep,
            k_heads,
            v_heads,
            k_dim: dk,
            v_dim: dk,
            row0: 0,
        };
        let ch = gs.qkv_width();
        let cs = ConvShape {
            tokens: keep,
            channels: ch,
            width,
            row0: 0,
        };
        let (state_len, conv_len) = (v_heads * dk * dk, (width - 1) * ch);
        struct L {
            qkv: Vec<f32>,
            conv_out: Vec<f32>,
            ab: Vec<f32>,
            a_log: Vec<f32>,
            dt: Vec<f32>,
            c0: Vec<f32>,
            s0: Vec<f32>,
            want_c: Vec<f32>,
            want_s: Vec<f32>,
        }
        let mut ls = Vec::new();
        for _ in 0..layers {
            let qkv = self.rng.vec(window * ch, 1.0);
            let w = self.rng.vec(width * ch, 0.5);
            let c0 = self.rng.vec(conv_len, 1.0);
            let ab = self.rng.vec(window * 2 * v_heads, 2.0);
            let (junk_qkv, junk_ab) = (
                self.rng.vec(row0 * ch, 1.0),
                self.rng.vec(row0 * 2 * v_heads, 2.0),
            );
            let (a_log, dt) = (self.rng.vec(v_heads, 1.0), self.rng.vec(v_heads, 1.0));
            let s0 = self.rng.vec(state_len, 0.1);
            // The window's conv outputs, then the reference states after `keep` rows.
            let (mut c, mut conv_out) = (c0.clone(), vec![0.0; window * ch]);
            let full = ConvShape {
                tokens: window,
                ..cs
            };
            CpuBackend.conv1d_silu(&qkv, &w, &mut c, &mut conv_out, full)?;
            let (mut want_c, mut y) = (c0.clone(), vec![0.0; keep * ch]);
            CpuBackend.conv1d_silu(&qkv, &w, &mut want_c, &mut y, cs)?;
            let (mut want_s, mut o) = (s0.clone(), vec![0.0; keep * v_heads * dk]);
            CpuBackend.gated_delta_net(&conv_out, &ab, &a_log, &dt, &mut want_s, &mut o, gs)?;
            // Other windows' rows ahead of this one.
            let at = |junk: &[f32], rows: Vec<f32>| [junk, &rows[..]].concat();
            let junk_out = self.rng.vec(row0 * ch, 1.0);
            ls.push(L {
                qkv: at(&junk_qkv, qkv),
                conv_out: at(&junk_out, conv_out),
                ab: at(&junk_ab, ab),
                a_log,
                dt,
                c0,
                s0,
                want_c,
                want_s,
            });
        }
        let up = |v: &[f32]| self.b.upload(v);
        let mut dev = Vec::new();
        for l in &ls {
            dev.push((
                up(&l.qkv)?,
                up(&l.conv_out)?,
                up(&l.ab)?,
                up(&l.a_log)?,
                up(&l.dt)?,
                up(&l.c0)?,
                up(&l.s0)?,
            ));
        }
        let mut items: Vec<GdnReplayLayer<'_, B::Buffer>> = dev
            .iter_mut()
            .map(|(q, co, ab, al, dt, c, s)| GdnReplayLayer {
                qkv: q,
                conv_out: co,
                ab,
                a_log: al,
                dt_bias: dt,
                conv_state: c,
                ssm: s,
            })
            .collect();
        self.b.gdn_replay(
            &mut items,
            ConvShape { row0, ..cs },
            GdnShape { row0, ..gs },
        )?;
        drop(items);
        let (mut got, mut want) = (Vec::new(), Vec::new());
        for (l, d) in ls.iter().zip(&dev) {
            got.extend(fetch(self.b, &d.5, conv_len)?);
            got.extend(fetch(self.b, &d.6, state_len)?);
            want.extend_from_slice(&l.want_c);
            want.extend_from_slice(&l.want_s);
        }
        let name = format!("gdn_replay {layers} layers, {keep} of {window} rows at {row0}");
        self.record(name, Some(Op::GdnReplay), &got, &want, 1e-4);
        Ok(())
    }

    /// `rms_norm_gated`: the gated rows.
    fn rms_norm_gated(&mut self, rows: usize, dim: usize, kind: NormKind) -> Result<()> {
        let n = rows * dim;
        let (x, w, g0) = (
            self.rng.vec(n, 2.0),
            self.rng.vec(dim, 0.5),
            self.rng.vec(n, 3.0),
        );
        let mut want = g0.clone();
        CpuBackend.rms_norm_gated(&x, &w, &mut want, rows, dim, 1e-6, kind)?;
        let (xd, wd) = (self.b.upload(&x)?, self.b.upload(&w)?);
        let mut gd = self.b.upload(&g0)?;
        self.b
            .rms_norm_gated(&xd, &wd, &mut gd, rows, dim, 1e-6, kind)?;
        let got = fetch(self.b, &gd, n)?;
        self.record(
            format!("rms_norm_gated {rows}x{dim} {kind:?}"),
            Some(Op::RmsNormGated),
            &got,
            &want,
            1e-5,
        );
        Ok(())
    }

    /// `add_rms_norm`: the updated residual and the normed rows.
    fn add_rms_norm(&mut self, rows: usize, dim: usize, kind: NormKind) -> Result<()> {
        let n = rows * dim;
        let (acc0, x, w) = (
            self.rng.vec(n, 2.0),
            self.rng.vec(n, 1.0),
            self.rng.vec(dim, 0.5),
        );
        let (mut acc, mut y) = (acc0.clone(), vec![0.0; n]);
        CpuBackend.add_rms_norm(&mut acc, &x, &w, &mut y, rows, dim, 1e-6, kind)?;
        let (mut ad, xd, wd) = (
            self.b.upload(&acc0)?,
            self.b.upload(&x)?,
            self.b.upload(&w)?,
        );
        let mut yd = self.b.alloc(n)?;
        self.b
            .add_rms_norm(&mut ad, &xd, &wd, &mut yd, rows, dim, 1e-6, kind)?;
        let mut got = fetch(self.b, &ad, n)?;
        got.extend(fetch(self.b, &yd, n)?);
        acc.extend(y);
        self.record(
            format!("add_rms_norm {rows}x{dim} {kind:?}"),
            Some(Op::RmsNorm),
            &got,
            &acc,
            1e-5,
        );
        Ok(())
    }

    fn attention_full(&mut self, s: FullAttnShape) -> Result<()> {
        let len = s.tokens * s.heads * s.head_dim;
        let (q, k, v) = (
            self.rng.vec(len, 2.0),
            self.rng.vec(len, 2.0),
            self.rng.vec(len, 1.0),
        );
        let precision = self.b.attention_full_precision(s);
        let (mut qr, mut kr, mut vr) = (q.clone(), k.clone(), v.clone());
        round_full_attention_inputs(&mut qr, &mut kr, &mut vr, precision);
        let mut want = vec![0.0; len];
        CpuBackend.attention_full(&qr, &kr, &vr, &mut want, s)?;
        let (qd, kd, vd) = (self.b.upload(&q)?, self.b.upload(&k)?, self.b.upload(&v)?);
        let mut od = self.b.alloc(len)?;
        self.b.attention_full(&qd, &kd, &vd, &mut od, s)?;
        let got = fetch(self.b, &od, len)?;
        self.record(
            format!(
                "attention_full t={} h={} d={}",
                s.tokens, s.heads, s.head_dim
            ),
            Some(Op::AttentionFull),
            &got,
            &want,
            match precision {
                AttnPrecision::Exact => 2e-4,
                // f16 softmax weights are not emulated.
                AttnPrecision::F16 => 2e-3,
                // No FP8 contract exists for attention_full.
                AttnPrecision::Fp8QkF16Pv => -1.0,
            },
        );
        Ok(())
    }

    fn vision_elementwise(&mut self, rows: usize, dim: usize) -> Result<()> {
        let len = rows * dim;
        let (x, w, b) = (
            self.rng.vec(len, 3.0),
            self.rng.vec(dim, 1.0),
            self.rng.vec(dim, 0.5),
        );
        let (xd, wd, bd) = (self.b.upload(&x)?, self.b.upload(&w)?, self.b.upload(&b)?);
        let mut want = vec![0.0; len];
        CpuBackend.layer_norm(&x, &w, &b, &mut want, rows, dim, 1e-6)?;
        let mut yd = self.b.alloc(len)?;
        self.b.layer_norm(&xd, &wd, &bd, &mut yd, rows, dim, 1e-6)?;
        let got = fetch(self.b, &yd, len)?;
        self.record(
            format!("layer_norm rows={rows} dim={dim}"),
            Some(Op::LayerNorm),
            &got,
            &want,
            1e-5,
        );

        let mut want = x.clone();
        CpuBackend.add_bias(&mut want, &b, rows, dim)?;
        let mut d = self.b.upload(&x)?;
        self.b.add_bias(&mut d, &bd, rows, dim)?;
        let got = fetch(self.b, &d, len)?;
        self.record(
            format!("add_bias rows={rows} dim={dim}"),
            Some(Op::AddBias),
            &got,
            &want,
            1e-6,
        );

        for kind in [GeluKind::Tanh, GeluKind::Erf] {
            let mut want = x.clone();
            CpuBackend.gelu(&mut want, len, kind)?;
            let mut d = self.b.upload(&x)?;
            self.b.gelu(&mut d, len, kind)?;
            let got = fetch(self.b, &d, len)?;
            self.record(
                format!("gelu {kind:?} len={len}"),
                Some(Op::Gelu),
                &got,
                &want,
                1e-5,
            );
        }
        Ok(())
    }

    fn rope_positions(&mut self, s: RopePosShape) -> Result<()> {
        let len = s.tokens * s.heads * s.head_dim;
        let positions: Vec<u32> = (0..s.axes * s.tokens)
            .map(|_| (self.rng.next_u64() % 5000) as u32)
            .collect();
        let mut want = self.rng.vec(len, 1.0);
        let mut xd = self.b.upload(&want)?;
        CpuBackend.rope_positions(&mut want, &positions, s)?;
        self.b.rope_positions(&mut xd, &positions, s)?;
        let got = fetch(self.b, &xd, len)?;
        let name = format!(
            "rope_positions {:?} axes={} t={} d={} r={}",
            s.layout, s.axes, s.tokens, s.head_dim, s.rotary_dim
        );
        self.record(name, Some(Op::RopePositions), &got, &want, 1e-5);
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
    // Batched speculative verify: several sequences' windows in one pass (17..96 rows).
    s.linear(WeightFormat::Fp8RowBf16, 160, 1024, 16, 128, 24)?;
    s.linear(WeightFormat::Fp8RowBf16, 96, 512, 0, 96, 40)?;
    s.linear(WeightFormat::Fp8RowBf16, 64, 768, 0, 64, 64)?;
    s.linear(WeightFormat::Fp8RowBf16, 64, 512, 0, 64, 90)?;
    // m >= 16 takes the tiled path, with ragged edges in every dimension.
    s.linear(WeightFormat::Fp8RowBf16, 200, 300, 17, 150, 70)?;
    s.linear(WeightFormat::Nvfp4, 256, 320, 0, 256, 33)?;
    s.linear(WeightFormat::Bf16, 90, 70, 0, 90, 16)?;
    // Verify and draft sizes with K a multiple of 128 (GDN gate projection, a row view).
    s.linear(WeightFormat::Bf16, 96, 5120, 0, 96, 8)?;
    s.linear(WeightFormat::Bf16, 130, 1024, 3, 98, 3)?;
    s.linear(WeightFormat::Bf16, 40, 256, 0, 40, 2)?;
    s.linear(WeightFormat::Bf16, 96, 5120, 0, 96, 12)?;
    s.linear(WeightFormat::Bf16, 1280, 1024, 0, 1280, 16)?;
    s.linear(WeightFormat::Bf16, 300, 1024, 40, 200, 3)?;
    s.linear(WeightFormat::Nvfp4, 256, 128, 0, 256, 1)?;
    s.linear(WeightFormat::Nvfp4, 384, 192, 128, 200, 5)?;
    // Activation policies: tuned kernels may round inputs to FP8 / NVFP4 where the artifact allows.
    // Shapes cover MMA tile edges in tokens, rows and K, and row views into a fused parent.
    let a4 = ActPolicy::AllowA4 { divisor: 600.0 };
    for (n, k, begin, rows, m) in [
        (256, 512, 0, 256, 16),
        (384, 640, 128, 256, 37),
        (512, 1024, 0, 512, 130),
        (256, 256, 128, 128, 300),
        (128, 64, 0, 128, 1),
        // Skinny verify windows: wide K stages, a K whose code bytes need the narrow fallback, and
        // row views that start inside a 128-row scale tile.
        (512, 2048, 128, 384, 8),
        (384, 640, 128, 256, 8),
        (256, 1024, 64, 192, 3),
    ] {
        s.linear_act(WeightFormat::Nvfp4, n, k, begin, rows, m, a4)?;
    }
    for (n, k, begin, rows, m) in [
        (256, 512, 0, 256, 16),
        (200, 320, 17, 150, 70),
        (512, 1024, 64, 384, 130),
        (96, 300, 0, 96, 40),
        (64, 128, 0, 64, 2),
        (512, 2048, 64, 384, 8),
        (300, 192, 0, 300, 5),
    ] {
        s.linear_act(
            WeightFormat::Fp8RowBf16,
            n,
            k,
            begin,
            rows,
            m,
            ActPolicy::AllowA8,
        )?;
    }
    // A16-only skinny windows (draft models): tuned kernels may round activations to bf16 only.
    for (n, k, begin, rows, m) in [
        (512, 2048, 64, 384, 8),
        (256, 5120, 0, 256, 16),
        (384, 1024, 128, 192, 2),
    ] {
        s.linear_act(
            WeightFormat::Nvfp4,
            n,
            k,
            begin,
            rows,
            m,
            ActPolicy::A16Only,
        )?;
    }
    // A16-only views stay exact even where a kernel could round.
    s.linear_act(
        WeightFormat::Nvfp4,
        256,
        512,
        0,
        256,
        64,
        ActPolicy::A16Only,
    )?;
    s.linear_act(
        WeightFormat::Fp8RowBf16,
        256,
        512,
        0,
        256,
        64,
        ActPolicy::A16Only,
    )?;
    // Batched drafting with A16-only NVFP4 weights (draft model, proposal head): 17..96 rows.
    for m in [21, 42, 48, 96] {
        s.linear_act(
            WeightFormat::Nvfp4,
            384,
            1024,
            64,
            256,
            m,
            ActPolicy::A16Only,
        )?;
    }
    // Groupwise integer formats; K not a multiple of 128 exercises physical padding.
    for f in [
        WeightFormat::Q4G64,
        WeightFormat::Q5G64,
        WeightFormat::Q6G64,
        WeightFormat::Q8G32,
    ] {
        s.linear(f, 40, 130, 3, 30, 2)?;
        // Draft-sized windows: many K stages (split-K) and a row view.
        s.linear(f, 512, 4096, 0, 512, 8)?;
        s.linear(f, 384, 2560, 64, 256, 16)?;
        s.linear(f, 72, 1152, 0, 72, 20)?;
        // Batched drafting: several sequences' blocks / proposal rows (17..63 rows).
        s.linear(f, 384, 2560, 64, 256, 42)?;
        s.linear(f, 512, 1024, 0, 512, 48)?;
        // Vision-tower sizes, where tuned kernels switch to tensor cores (bf16 activations):
        // ragged token counts, a row view, and K = 4304 padded to 4352 in storage.
        // Small m (decode, drafting): the vectorized GEMV for Q4 / Q8, row views and K-split rows.
        s.linear(f, 3000, 1152, 37, 2500, 1)?;
        s.linear(f, 300, 1152, 0, 300, 5)?;
        s.linear(f, 160, 1152, 32, 96, 130)?;
        s.linear(f, 96, 4304, 0, 96, 70)?;
    }
    // Sampling candidates: a Qwen-sized vocabulary row (public limit below the padded width), a
    // small one, and k larger than the row.
    s.concat_cols(3, 5120, 5120)?;
    s.split_cols(3, &[6144, 1024, 6144, 1024])?;
    s.split_cols(2, &[7, 5, 3, 9, 4])?;
    s.swiglu(3, 17408)?;
    s.swiglu(5, 77)?;
    // DFlash2 drafting: feature placement, dynamic convolution, windowed block attention.
    s.copy2d(Copy2dShape {
        rows: 7,
        cols: 64,
        src_offset: 3,
        src_stride: 64,
        dst_offset: 128,
        dst_stride: 320,
    })?;
    s.dyn_conv(8, 5120, 0, false, 0)?;
    s.dyn_conv(8, 5120, 1, true, 0)?;
    s.dyn_conv(1, 64, 0, false, 0)?;
    s.dyn_conv(24, 64, 0, false, 8)?;
    s.dyn_conv(24, 64, 1, true, 8)?;
    for (pos0, ctx_lo, window, ring) in [
        (3000, 1000, 2048, 2048),
        (40, 0, 2048, 2048),
        (100, 50, 16, 64),
        (0, 0, 8, 8),
    ] {
        s.attention_block(BlockAttnShape {
            tokens: 8,
            heads: 32,
            kv_heads: 8,
            head_dim: 128,
            pos0,
            ctx_lo,
            window,
            ring,
        })?;
    }
    // Other GQA geometries of the shared-tile kernel: d64 with a partial row group, d256 at 16 tokens.
    for (tokens, heads, kv_heads, head_dim, pos0, ctx_lo, window, ring) in [
        (5, 12, 4, 64, 300, 100, 128, 256),
        (16, 16, 2, 256, 5000, 4000, 2048, 2048),
    ] {
        s.attention_block(BlockAttnShape {
            tokens,
            heads,
            kv_heads,
            head_dim,
            pos0,
            ctx_lo,
            window,
            ring,
        })?;
    }
    s.concat_cols(7, 13, 5)?;
    s.top_k(2, 248_320, 248_077, 64)?;
    s.top_k(3, 5000, 4097, 20)?;
    s.top_k(1, 50, 50, 64)?;
    s.top_k(4, 1000, 1000, 1)?;
    s.embed(WeightFormat::Fp8RowBf16, 50, 64, &[3, 0, 49, 3])?;
    s.embed(WeightFormat::Bf16, 10, 130, &[9, 1])?;
    s.embed(WeightFormat::Nvfp4, 128, 64, &[127, 0, 64])?;
    s.embed(WeightFormat::Q5G64, 20, 200, &[19, 0])?;
    s.add_rms_norm(8, 5120, NormKind::Offset)?;
    s.rms_norm_gated(8 * 48, 128, NormKind::Plain)?;
    s.rms_norm_gated(5, 96, NormKind::Offset)?;
    s.add_rms_norm(3, 130, NormKind::Plain)?;
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
    s.copy_batch()?;
    s.virtual_memory()?;
    for a in [
        // Prefill at the Qwen3.5 geometry (head_dim 256, 6 query heads per KV head); tensor-core
        // kernels tile 64 queries x 32 keys, so these are ragged in both.
        AttnShape {
            tokens: 70,
            heads: 6,
            kv_heads: 1,
            head_dim: 256,
            pos0: 0,
            kv: KvFormat::F32,
        },
        AttnShape {
            tokens: 130,
            heads: 12,
            kv_heads: 2,
            head_dim: 256,
            pos0: 333,
            kv: KvFormat::F32,
        },
        AttnShape {
            tokens: 17,
            heads: 6,
            kv_heads: 1,
            head_dim: 256,
            pos0: 2_000,
            kv: KvFormat::F32,
        },
        AttnShape {
            tokens: 1,
            heads: 8,
            kv_heads: 2,
            head_dim: 64,
            pos0: 37,
            kv: KvFormat::F32,
        },
        AttnShape {
            tokens: 5,
            heads: 4,
            kv_heads: 4,
            head_dim: 128,
            pos0: 0,
            kv: KvFormat::F32,
        },
        AttnShape {
            tokens: 3,
            heads: 6,
            kv_heads: 3,
            head_dim: 96,
            pos0: 10,
            kv: KvFormat::F32,
        },
        AttnShape {
            tokens: 1,
            heads: 2,
            kv_heads: 1,
            head_dim: 256,
            pos0: 1500,
            kv: KvFormat::F32,
        },
        AttnShape {
            tokens: 1,
            heads: 32,
            kv_heads: 8,
            head_dim: 128,
            pos0: 4095,
            kv: KvFormat::F32,
        },
        AttnShape {
            tokens: 4,
            heads: 24,
            kv_heads: 4,
            head_dim: 256,
            pos0: 20,
            kv: KvFormat::F32,
        },
        // Split-KV path: few queries over long caches, ragged last split.
        AttnShape {
            tokens: 1,
            heads: 24,
            kv_heads: 4,
            head_dim: 256,
            pos0: 16_999,
            kv: KvFormat::F32,
        },
        AttnShape {
            tokens: 3,
            heads: 6,
            kv_heads: 2,
            head_dim: 96,
            pos0: 1_000,
            kv: KvFormat::F32,
        },
        // Speculative verify windows over long caches (GQA-packed split path).
        AttnShape {
            tokens: 8,
            heads: 24,
            kv_heads: 4,
            head_dim: 256,
            pos0: 17_000,
            kv: KvFormat::F32,
        },
        AttnShape {
            tokens: 16,
            heads: 24,
            kv_heads: 4,
            head_dim: 256,
            pos0: 37,
            kv: KvFormat::F32,
        },
        AttnShape {
            tokens: 5,
            heads: 32,
            kv_heads: 8,
            head_dim: 128,
            pos0: 2_049,
            kv: KvFormat::F32,
        },
    ] {
        s.attention(a)?;
    }
    // Quantized KV caches: store, then decode / verify / prefill over packed rows.
    for f in KvFormat::ALL {
        s.kv_store(f, 4, 256, 16_999, 1)?;
        s.kv_store(f, 4, 256, 100, 16)?;
        s.kv_store(f, 4, 256, 0, 70)?;
    }
    s.kv_store(KvFormat::Bf16, 2, 72, 5, 3)?;
    s.kv_store(KvFormat::Fp8, 2, 64, 5, 3)?;
    s.kv_store(KvFormat::Fp8, 8, 128, 33, 9)?;
    s.kv_store(KvFormat::Nvfp4, 3, 32, 7, 2)?;
    s.kv_store(KvFormat::Nvfp4, 2, 128, 7, 5)?;
    for f in [
        KvFormat::F32,
        KvFormat::Bf16,
        KvFormat::Fp8,
        KvFormat::Nvfp4,
    ] {
        s.kv_load(f, 4, 256, 0, 300)?;
        s.kv_load(f, 2, 64, 17, 5)?;
    }
    let attn = |tokens, heads, kv_heads, head_dim, pos0, kv| AttnShape {
        tokens,
        heads,
        kv_heads,
        head_dim,
        pos0,
        kv,
    };
    for f in [KvFormat::Bf16, KvFormat::Fp8, KvFormat::Nvfp4] {
        // Qwen3.8 geometry: decode, verify windows (1..3 row groups), prefill chunks.
        s.attention(attn(1, 24, 4, 256, 16_999, f))?;
        s.attention(attn(2, 24, 4, 256, 300, f))?;
        s.attention(attn(8, 24, 4, 256, 17_000, f))?;
        s.attention(attn(16, 24, 4, 256, 37, f))?;
        s.attention(attn(5, 24, 4, 256, 2_049, f))?;
        s.attention(attn(20, 6, 1, 256, 16_500, f))?;
        s.attention(attn(100, 24, 4, 256, 200, f))?;
        s.attention(attn(70, 6, 1, 256, 0, f))?;
    }
    // Other head dims: non-power-of-two (bf16 only), narrower rotations, no GQA.
    s.attention(attn(3, 6, 2, 96, 1_000, KvFormat::Bf16))?;
    s.attention(attn(1, 8, 2, 72, 37, KvFormat::Bf16))?;
    s.attention(attn(40, 8, 2, 72, 10, KvFormat::Bf16))?;
    s.attention(attn(5, 32, 8, 128, 2_049, KvFormat::Fp8))?;
    s.attention(attn(1, 8, 2, 64, 37, KvFormat::Fp8))?;
    s.attention(attn(3, 4, 4, 128, 500, KvFormat::Fp8))?;
    s.attention(attn(4, 8, 2, 128, 500, KvFormat::Nvfp4))?;
    s.attention(attn(3, 4, 4, 32, 64, KvFormat::Nvfp4))?;
    s.attention(attn(1, 16, 2, 64, 3_000, KvFormat::Nvfp4))?;
    s.conv1d(70, 4, [5, 1])?;
    s.conv1d(1024, 4, [1, 1])?;
    s.state_from(16, 48, 128, 8)?;
    s.state_from(2, 6, 64, 3)?;
    s.state_from(2, 4, 128, 20)?;
    s.gdn_replay(3, 8, 5, 0)?;
    s.gdn_replay(2, 8, 1, 0)?;
    s.gdn_replay(2, 12, 12, 0)?;
    s.gdn_replay(3, 8, 5, 11)?;
    s.gdn(1, 3, 32, 32, [4, 1])?;
    // Qwen3.5 GDN head geometry (16/48 heads scaled down to 2/6), Dk = Dv = 128.
    s.gdn(2, 6, 128, 128, [6, 1])?;
    // Prefill chunks (chunkwise-parallel kernels): full model heads with a ragged second chunk,
    // a decode step before and after a chunk, and small or uneven head dims.
    s.gdn(16, 48, 128, 128, [512, 200])?;
    s.gdn(2, 6, 128, 128, [64, 1])?;
    s.gdn(2, 6, 128, 128, [16, 31])?;
    s.gdn(2, 6, 128, 128, [1, 77])?;
    s.gdn(1, 3, 32, 32, [100, 3])?;
    s.gdn(1, 2, 64, 48, [70, 65])?;
    // Vision tower geometry: 16 heads x 72, hidden 1152.
    s.attention_full(FullAttnShape {
        tokens: 64,
        heads: 4,
        head_dim: 72,
    })?;
    // A screenshot-sized tower slice for tensor-core kernels: ragged in 64-query tiles.
    s.attention_full(FullAttnShape {
        tokens: 300,
        heads: 3,
        head_dim: 72,
    })?;
    s.attention_full(FullAttnShape {
        tokens: 9,
        heads: 2,
        head_dim: 64,
    })?;
    s.vision_elementwise(5, 1152)?;
    s.vision_elementwise(3, 4304)?;
    // Text MRoPE (3 interleaved axes, 64 of 256 dims, theta 1e7) and vision 2D RoPE (2 blocked
    // axes over all 72 dims, theta 1e4).
    s.rope_positions(RopePosShape {
        tokens: 5,
        heads: 3,
        head_dim: 256,
        rotary_dim: 64,
        theta: 10_000_000.0,
        axes: 3,
        layout: RopeLayout::Interleaved,
    })?;
    s.rope_positions(RopePosShape {
        tokens: 8,
        heads: 2,
        head_dim: 72,
        rotary_dim: 72,
        theta: 10_000.0,
        axes: 2,
        layout: RopeLayout::Blocked,
    })?;
    s.stack(&ModelSpec::preset("tiny")?)?;
    // Cases added later go last: the suite draws from one random stream, so inserting earlier
    // would change every following case's data.
    // IQ3_XXS dense MLP: decode, a verify window over an odd row view, and the tiled path.
    s.linear(WeightFormat::Iq3Xxs, 96, 512, 0, 96, 1)?;
    s.linear(WeightFormat::Iq3Xxs, 130, 768, 3, 100, 8)?;
    s.linear(WeightFormat::Iq3Xxs, 200, 256, 17, 150, 33)?;
    s.embed(WeightFormat::Iq3Xxs, 12, 512, &[11, 0, 5])?;
    // Greedy speculative verify: the argmax of each row of a verify window.
    s.top_k(8, 248_320, 248_077, 1)?;
    // Fused SwiGLU projections at decode and verify widths: the dense MLP's formats, ragged
    // widths, both row orders and separate parents.
    for m in [1, 2, 8, 16] {
        s.linear_swiglu(WeightFormat::Iq3Xxs, 512, 40, 0, m, false, false)?;
    }
    s.linear_swiglu(WeightFormat::Iq3Xxs, 768, 24, 8, 3, false, false)?;
    s.linear_swiglu(WeightFormat::Iq3Xxs, 256, 16, 0, 8, true, false)?;
    s.linear_swiglu(WeightFormat::Iq3Xxs, 512, 16, 0, 8, false, true)?;
    s.linear_swiglu(WeightFormat::Q4G64, 320, 40, 3, 8, false, false)?;
    s.linear_swiglu(WeightFormat::Nvfp4, 512, 128, 0, 8, false, false)?;
    s.linear_swiglu(WeightFormat::Bf16, 96, 33, 1, 5, false, false)?;
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
    fn row_split_decode_matches_encoded_codes() {
        use velox_core::format::{decode_rows, f16_to_f32};
        let mut rng = Rng::new(5);
        for f in [
            WeightFormat::Q4G64,
            WeightFormat::Q5G64,
            WeightFormat::Q6G64,
            WeightFormat::Q8G32,
        ] {
            let rs = f.row_split().unwrap();
            let (n, k) = (3, 200);
            let (codes, scales) = random_codes(&mut rng, rs, n, k);
            let bytes = encode_row_split(rs, n, k, &codes, &scales);
            assert_eq!(bytes.len(), f.payload_bytes(n, k));
            let got = decode_rows(f, &bytes, n, k, 0, n).unwrap();
            let groups = k.div_ceil(rs.group);
            for r in 0..n {
                for c in 0..k {
                    let want =
                        f16_to_f32(scales[r * groups + c / rs.group]) * codes[r * k + c] as f32;
                    assert_eq!(got[r * k + c], want, "{f:?} [{r},{c}]");
                }
            }
        }
    }

    #[test]
    fn random_weights_have_layout_sizes() {
        let mut rng = Rng::new(1);
        for (f, n, k) in [
            (WeightFormat::Fp8RowBf16, 96, 300),
            (WeightFormat::Nvfp4, 256, 128),
            (WeightFormat::Bf16, 3, 5),
            (WeightFormat::Iq3Xxs, 5, 768),
        ] {
            assert_eq!(
                random_weight(&mut rng, f, n, k).len(),
                f.payload_bytes(n, k),
                "{f:?}"
            );
        }
    }
}
