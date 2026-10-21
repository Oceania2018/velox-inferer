//! CPU reference backend.
//!
//! Straightforward loops with f64 accumulation. It is the oracle the conformance suite compares
//! every device backend against, so clarity wins over speed here: never optimise it in ways that
//! change its numerics.

use velox_core::backend::{
    top_k_row, AttnShape, Backend, BackendKind, BlockAttnShape, ConvShape, Copy2dShape,
    DynConvShape, FullAttnShape, GdnShape, GeluKind, NormKind, RopePosShape, RopeShape, WeightView,
};
use velox_core::format::decode_rows;
use velox_core::{check, DeviceProfile, Op, Result};

#[derive(Default, Clone, Copy)]
pub struct CpuBackend;

/// Raw bytes of an f32 buffer (encoded weights are stored padded into f32 slots).
fn as_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding or invalid bit patterns; u8 has alignment 1.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), v.len() * 4) }
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum()
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f64) -> f64 {
    x * sigmoid(x)
}

fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

/// Decoded rows of a weight view, `[rows, k]`.
fn decode_view(w: &WeightView<'_, Vec<f32>>) -> Result<Vec<f32>> {
    decode_rows(
        w.format,
        as_bytes(w.buf),
        w.parent_rows,
        w.k,
        w.row_begin,
        w.rows,
    )
}

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

    fn upload_bytes(&self, data: &[u8]) -> Result<Vec<f32>> {
        let mut v = vec![0f32; data.len().div_ceil(4)];
        // SAFETY: v holds at least data.len() bytes; any bit pattern is a valid f32.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), v.as_mut_ptr().cast::<u8>(), data.len())
        };
        Ok(v)
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
                y[i * n + j] = dot(xr, &w[j * k..(j + 1) * k]) as f32;
            }
        }
        Ok(())
    }

    fn linear(
        &self,
        x: &Vec<f32>,
        w: &WeightView<'_, Vec<f32>>,
        y: &mut Vec<f32>,
        m: usize,
    ) -> Result<()> {
        check::linear(x.len(), w, w.buf.len(), y.len(), m)?;
        let wd = decode_view(w)?;
        let (k, n) = (w.k, w.rows);
        for i in 0..m {
            let xr = &x[i * k..(i + 1) * k];
            for j in 0..n {
                y[i * n + j] = dot(xr, &wd[j * k..(j + 1) * k]) as f32;
            }
        }
        Ok(())
    }

    fn embed(
        &self,
        ids: &[u32],
        table: &WeightView<'_, Vec<f32>>,
        out: &mut Vec<f32>,
    ) -> Result<()> {
        check::embed(ids, table, table.buf.len(), out.len())?;
        let k = table.k;
        for (t, &id) in ids.iter().enumerate() {
            let row = decode_view(&table.rows(id as usize, 1))?;
            out[t * k..(t + 1) * k].copy_from_slice(&row);
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
        kind: NormKind,
    ) -> Result<()> {
        check::rms_norm(x.len(), weight.len(), y.len(), rows, dim)?;
        let offset = if kind == NormKind::Offset { 1.0 } else { 0.0 };
        for r in 0..rows {
            let xr = &x[r * dim..(r + 1) * dim];
            let ms = xr.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / dim as f64;
            let inv = 1.0 / (ms + f64::from(eps)).sqrt();
            for c in 0..dim {
                y[r * dim + c] = (f64::from(xr[c]) * inv * (offset + f64::from(weight[c]))) as f32;
            }
        }
        Ok(())
    }

    fn rope(&self, x: &mut Vec<f32>, s: RopeShape) -> Result<()> {
        check::rope(x.len(), &s)?;
        let half = s.rotary_dim / 2;
        for t in 0..s.tokens {
            let pos = (s.pos0 + t) as f64;
            for h in 0..s.heads {
                let base = (t * s.heads + h) * s.head_dim;
                for i in 0..half {
                    let freq = f64::from(s.theta).powf(-2.0 * i as f64 / s.rotary_dim as f64);
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
            *g = (silu(f64::from(*g)) * f64::from(u)) as f32;
        }
        Ok(())
    }

    fn sigmoid_mul(&self, x: &mut Vec<f32>, gate: &Vec<f32>, len: usize) -> Result<()> {
        check::elementwise("sigmoid_mul", &[x.len(), gate.len()], len)?;
        for (v, &g) in x[..len].iter_mut().zip(&gate[..len]) {
            *v = (f64::from(*v) * sigmoid(f64::from(g))) as f32;
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
        src_offset: usize,
        dst: &mut Vec<f32>,
        dst_offset: usize,
        len: usize,
    ) -> Result<()> {
        check::copy(src.len(), src_offset, dst.len(), dst_offset, len)?;
        dst[dst_offset..dst_offset + len].copy_from_slice(&src[src_offset..src_offset + len]);
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
                    *sc = dot(qv, &k[(j * s.kv_heads + kvh) * d..][..d]) * scale;
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

    fn copy2d(&self, src: &Vec<f32>, dst: &mut Vec<f32>, s: Copy2dShape) -> Result<()> {
        check::copy2d(src.len(), dst.len(), &s)?;
        for r in 0..s.rows {
            let (a, b) = (
                s.src_offset + r * s.src_stride,
                s.dst_offset + r * s.dst_stride,
            );
            dst[b..b + s.cols].copy_from_slice(&src[a..a + s.cols]);
        }
        Ok(())
    }

    fn dyn_conv(
        &self,
        x: &Vec<f32>,
        base: &Vec<f32>,
        delta: &Vec<f32>,
        out: &mut Vec<f32>,
        s: DynConvShape,
    ) -> Result<()> {
        check::dyn_conv(x.len(), base.len(), delta.len(), out.len(), &s)?;
        let (c, g_n) = (s.channels, s.channels / s.group);
        let (t0, t1) = (2 * s.side, 2 * s.side + 1);
        for i in 0..s.rows {
            let d = &delta[i * 4 * g_n..][..4 * g_n];
            for h in 0..c {
                let g = h / s.group;
                let mut y = (f64::from(base[t0 * c + h]) + f64::from(d[t0 * g_n + g]))
                    * f64::from(x[i * c + h]);
                if i > 0 {
                    y += (f64::from(base[t1 * c + h]) + f64::from(d[t1 * g_n + g]))
                        * f64::from(x[(i - 1) * c + h]);
                }
                let o = &mut out[i * c + h];
                *o = if s.accumulate {
                    (f64::from(*o) + y) as f32
                } else {
                    y as f32
                };
            }
        }
        Ok(())
    }

    fn attention_block(
        &self,
        q: &Vec<f32>,
        k_ctx: &Vec<f32>,
        v_ctx: &Vec<f32>,
        k_blk: &Vec<f32>,
        v_blk: &Vec<f32>,
        out: &mut Vec<f32>,
        s: BlockAttnShape,
    ) -> Result<()> {
        check::attention_block(
            q.len(),
            k_ctx.len(),
            v_ctx.len(),
            k_blk.len(),
            v_blk.len(),
            out.len(),
            &s,
        )?;
        let d = s.head_dim;
        let group = s.heads / s.kv_heads;
        let scale = 1.0 / (d as f64).sqrt();
        for t in 0..s.tokens {
            // Key rows: (buffer is context, row) for the visible context, then the block.
            let keys: Vec<(bool, usize)> = (s.ctx_start(t)..s.pos0)
                .map(|p| (true, p % s.ring))
                .chain((0..s.tokens).map(|j| (false, j)))
                .collect();
            for h in 0..s.heads {
                let kvh = h / group;
                let qv = &q[(t * s.heads + h) * d..][..d];
                let row = |ctx: bool, r: usize, which: usize| -> &[f32] {
                    let buf = match (ctx, which) {
                        (true, 0) => k_ctx,
                        (true, _) => v_ctx,
                        (false, 0) => k_blk,
                        (false, _) => v_blk,
                    };
                    &buf[(r * s.kv_heads + kvh) * d..][..d]
                };
                let scores: Vec<f64> = keys
                    .iter()
                    .map(|&(c, r)| dot(qv, row(c, r, 0)) * scale)
                    .collect();
                let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = scores.iter().map(|sc| (sc - max).exp()).collect();
                let denom: f64 = e.iter().sum();
                let o = &mut out[(t * s.heads + h) * d..][..d];
                for (c, oc) in o.iter_mut().enumerate() {
                    let acc: f64 = keys
                        .iter()
                        .zip(&e)
                        .map(|(&(ctx, r), w)| w * f64::from(row(ctx, r, 1)[c]))
                        .sum();
                    *oc = (acc / denom) as f32;
                }
            }
        }
        Ok(())
    }

    fn conv1d_silu(
        &self,
        x: &Vec<f32>,
        weight: &Vec<f32>,
        state: &mut Vec<f32>,
        out: &mut Vec<f32>,
        s: ConvShape,
    ) -> Result<()> {
        check::conv1d(x.len(), weight.len(), state.len(), out.len(), &s)?;
        let (c, w) = (s.channels, s.width);
        // u = [state; x], rows of `c` channels.
        let mut u = state[..(w - 1) * c].to_vec();
        u.extend_from_slice(&x[..s.tokens * c]);
        for t in 0..s.tokens {
            for ch in 0..c {
                let acc: f64 = (0..w)
                    .map(|j| f64::from(weight[j * c + ch]) * f64::from(u[(t + j) * c + ch]))
                    .sum();
                out[t * c + ch] = silu(acc) as f32;
            }
        }
        state[..(w - 1) * c].copy_from_slice(&u[s.tokens * c..]);
        Ok(())
    }

    fn gated_delta_net(
        &self,
        qkv: &Vec<f32>,
        ab: &Vec<f32>,
        a_log: &Vec<f32>,
        dt_bias: &Vec<f32>,
        state: &mut Vec<f32>,
        out: &mut Vec<f32>,
        s: GdnShape,
    ) -> Result<()> {
        check::gdn(
            qkv.len(),
            ab.len(),
            a_log.len(),
            dt_bias.len(),
            state.len(),
            out.len(),
            &s,
        )?;
        let (dk, dv, w) = (s.k_dim, s.v_dim, s.qkv_width());
        let group = s.v_heads / s.k_heads;
        let l2 = |x: &[f32]| -> Vec<f64> {
            let inv =
                1.0 / (x.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() + 1e-6).sqrt();
            x.iter().map(|&v| f64::from(v) * inv).collect()
        };
        for t in 0..s.tokens {
            let row = &qkv[t * w..(t + 1) * w];
            for h in 0..s.v_heads {
                let kh = h / group;
                let q = l2(&row[kh * dk..(kh + 1) * dk]);
                let k = l2(&row[(s.k_heads + kh) * dk..(s.k_heads + kh + 1) * dk]);
                let v = &row[2 * s.k_heads * dk + h * dv..][..dv];
                let a = f64::from(ab[t * 2 * s.v_heads + h]);
                let b = f64::from(ab[t * 2 * s.v_heads + s.v_heads + h]);
                let g = -f64::from(a_log[h]).exp() * softplus(a + f64::from(dt_bias[h]));
                let (alpha, beta) = (g.exp(), sigmoid(b));
                let st = &mut state[h * dv * dk..(h + 1) * dv * dk];
                for i in 0..dv {
                    let srow = &mut st[i * dk..(i + 1) * dk];
                    let mut sk = 0.0;
                    for (sv, &kv) in srow.iter_mut().zip(&k) {
                        *sv = (f64::from(*sv) * alpha) as f32;
                        sk += f64::from(*sv) * kv;
                    }
                    let delta = beta * (f64::from(v[i]) - sk);
                    let mut o = 0.0;
                    for ((sv, &kv), &qv) in srow.iter_mut().zip(&k).zip(&q) {
                        *sv = (f64::from(*sv) + delta * kv) as f32;
                        o += f64::from(*sv) * qv;
                    }
                    out[(t * s.v_heads + h) * dv + i] = (o / (dk as f64).sqrt()) as f32;
                }
            }
        }
        Ok(())
    }

    fn attention_full(
        &self,
        q: &Vec<f32>,
        k: &Vec<f32>,
        v: &Vec<f32>,
        out: &mut Vec<f32>,
        s: FullAttnShape,
    ) -> Result<()> {
        check::attention_full(q.len(), k.len(), v.len(), out.len(), &s)?;
        let (n, h, d) = (s.tokens, s.heads, s.head_dim);
        let scale = 1.0 / (d as f64).sqrt();
        let mut scores = vec![0f64; n];
        for t in 0..n {
            for hh in 0..h {
                let qv = &q[(t * h + hh) * d..][..d];
                for (j, sc) in scores.iter_mut().enumerate() {
                    *sc = dot(qv, &k[(j * h + hh) * d..][..d]) * scale;
                }
                let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let denom: f64 = scores
                    .iter_mut()
                    .map(|sc| {
                        *sc = (*sc - max).exp();
                        *sc
                    })
                    .sum();
                for c in 0..d {
                    let acc: f64 = (0..n)
                        .map(|j| scores[j] * f64::from(v[(j * h + hh) * d + c]))
                        .sum();
                    out[(t * h + hh) * d + c] = (acc / denom) as f32;
                }
            }
        }
        Ok(())
    }

    fn layer_norm(
        &self,
        x: &Vec<f32>,
        weight: &Vec<f32>,
        bias: &Vec<f32>,
        y: &mut Vec<f32>,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<()> {
        check::layer_norm(x.len(), weight.len(), bias.len(), y.len(), rows, dim)?;
        for r in 0..rows {
            let xr = &x[r * dim..(r + 1) * dim];
            let mean = xr.iter().map(|&v| f64::from(v)).sum::<f64>() / dim as f64;
            let var = xr
                .iter()
                .map(|&v| (f64::from(v) - mean).powi(2))
                .sum::<f64>()
                / dim as f64;
            let inv = 1.0 / (var + f64::from(eps)).sqrt();
            for c in 0..dim {
                y[r * dim + c] = ((f64::from(xr[c]) - mean) * inv * f64::from(weight[c])
                    + f64::from(bias[c])) as f32;
            }
        }
        Ok(())
    }

    fn add_bias(&self, x: &mut Vec<f32>, bias: &Vec<f32>, rows: usize, dim: usize) -> Result<()> {
        check::add_bias(x.len(), bias.len(), rows, dim)?;
        for r in 0..rows {
            for c in 0..dim {
                x[r * dim + c] += bias[c];
            }
        }
        Ok(())
    }

    fn gelu(&self, x: &mut Vec<f32>, len: usize, kind: GeluKind) -> Result<()> {
        check::elementwise("gelu", &[x.len()], len)?;
        for v in &mut x[..len] {
            let a = f64::from(*v);
            let g = match kind {
                GeluKind::Tanh => {
                    0.5 * a
                        * (1.0
                            + ((2.0 / std::f64::consts::PI).sqrt() * (a + 0.044715 * a * a * a))
                                .tanh())
                }
                GeluKind::Erf => 0.5 * a * (1.0 + erf(a / std::f64::consts::SQRT_2)),
            };
            *v = g as f32;
        }
        Ok(())
    }

    fn split_cols(
        &self,
        src: &Vec<f32>,
        rows: usize,
        src_cols: usize,
        outs: &mut [(&mut Vec<f32>, usize, usize)],
    ) -> Result<()> {
        let dims: Vec<(usize, usize, usize)> = outs.iter().map(|o| (o.0.len(), o.1, o.2)).collect();
        check::split_cols(src.len(), rows, src_cols, &dims)?;
        for (dst, off, cols) in outs.iter_mut() {
            for r in 0..rows {
                dst[r * *cols..(r + 1) * *cols]
                    .copy_from_slice(&src[r * src_cols + *off..r * src_cols + *off + *cols]);
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn swiglu(
        &self,
        src: &Vec<f32>,
        rows: usize,
        src_cols: usize,
        gate: usize,
        up: usize,
        width: usize,
        out: &mut Vec<f32>,
    ) -> Result<()> {
        check::swiglu(src.len(), rows, src_cols, gate, up, width, out.len())?;
        for r in 0..rows {
            for i in 0..width {
                let g = f64::from(src[r * src_cols + gate + i]);
                let u = f64::from(src[r * src_cols + up + i]);
                out[r * width + i] = (g / (1.0 + (-g).exp()) * u) as f32;
            }
        }
        Ok(())
    }

    fn concat_cols(
        &self,
        a: &Vec<f32>,
        b: &Vec<f32>,
        out: &mut Vec<f32>,
        rows: usize,
        a_cols: usize,
        b_cols: usize,
    ) -> Result<()> {
        check::concat_cols(a.len(), b.len(), out.len(), rows, a_cols, b_cols)?;
        let w = a_cols + b_cols;
        for r in 0..rows {
            out[r * w..r * w + a_cols].copy_from_slice(&a[r * a_cols..(r + 1) * a_cols]);
            out[r * w + a_cols..(r + 1) * w].copy_from_slice(&b[r * b_cols..(r + 1) * b_cols]);
        }
        Ok(())
    }

    fn top_k(
        &self,
        x: &Vec<f32>,
        rows: usize,
        row_len: usize,
        limit: usize,
        k: usize,
    ) -> Result<Vec<Vec<(u32, f32)>>> {
        check::top_k(x.len(), rows, row_len, limit, k)?;
        Ok((0..rows)
            .map(|r| top_k_row(&x[r * row_len..r * row_len + limit], k))
            .collect())
    }

    fn rope_positions(&self, x: &mut Vec<f32>, positions: &[u32], s: RopePosShape) -> Result<()> {
        check::rope_positions(x.len(), positions, &s)?;
        let half = s.rotary_dim / 2;
        for t in 0..s.tokens {
            for h in 0..s.heads {
                let base = (t * s.heads + h) * s.head_dim;
                for i in 0..half {
                    let (axis, freq) = s.pair(i);
                    let (sin, cos) = (f64::from(positions[axis * s.tokens + t]) * freq).sin_cos();
                    let a = f64::from(x[base + i]);
                    let b = f64::from(x[base + i + half]);
                    x[base + i] = (a * cos - b * sin) as f32;
                    x[base + i + half] = (a * sin + b * cos) as f32;
                }
            }
        }
        Ok(())
    }
}

/// f64 error function: Maclaurin series below 2.5, erfc continued fraction above (both ~1e-15).
fn erf(x: f64) -> f64 {
    // Series for |x| < 2.5, continued fraction for the tail; both accurate to ~1e-15.
    let ax = x.abs();
    let r = if ax < 2.5 {
        let (mut term, mut sum, x2) = (ax, ax, ax * ax);
        let mut n = 0.0;
        while term.abs() > 1e-17 * sum.abs() {
            n += 1.0;
            term *= -x2 / n;
            sum += term / (2.0 * n + 1.0);
        }
        2.0 / std::f64::consts::PI.sqrt() * sum
    } else {
        // erfc continued fraction (Lentz), then erf = 1 - erfc.
        let mut f = ax;
        for k in (1..60).rev() {
            f = ax + (k as f64 / 2.0) / f;
        }
        1.0 - (-ax * ax).exp() / (std::f64::consts::PI.sqrt() * f)
    };
    if x < 0.0 {
        -r
    } else {
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use velox_core::WeightFormat;

    #[test]
    fn erf_reference_values() {
        for (x, want) in [
            (0.0, 0.0),
            (0.5, 0.520_499_877_813_046_5),
            (1.0, 0.842_700_792_949_714_9),
            (-2.0, -0.995_322_265_018_952_7),
            (3.0, 0.999_977_909_503_001_4),
        ] {
            assert!((erf(x) - want).abs() < 1e-13, "erf({x}) = {}", erf(x));
        }
    }

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
    fn linear_bf16_matches_matmul() {
        let b = CpuBackend;
        // W = [[1, 2], [-0.5, 3]] as bf16 words.
        let words: [u16; 4] = [0x3f80, 0x4000, 0xbf00, 0x4040];
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let wb = b.upload_bytes(&bytes).unwrap();
        let view = WeightView::full(&wb, WeightFormat::Bf16, 2, 2);
        let mut y = vec![0.0; 2];
        b.linear(&vec![1.0, 1.0], &view, &mut y, 1).unwrap();
        assert_eq!(y, vec![3.0, 2.5]);
        let mut e = vec![0.0; 2];
        b.embed(&[1], &view, &mut e).unwrap();
        assert_eq!(e, vec![-0.5, 3.0]);
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
        let mut out = vec![0.0; 4];
        b.attention(
            &vec![1.0, 2.0, -3.0, 0.5],
            &vec![0.3, 0.7],
            &vec![4.0, -1.0],
            &mut out,
            s,
        )
        .unwrap();
        assert_eq!(out, vec![4.0, -1.0, 4.0, -1.0]);
    }

    #[test]
    fn partial_rope_leaves_tail_untouched() {
        let b = CpuBackend;
        let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        b.rope(
            &mut x,
            RopeShape {
                tokens: 1,
                heads: 1,
                head_dim: 6,
                rotary_dim: 4,
                pos0: 3,
                theta: 10_000.0,
            },
        )
        .unwrap();
        assert_eq!(&x[4..], &[5.0, 6.0]);
        assert_ne!(x[0], 1.0);
    }

    #[test]
    fn conv_state_carries_across_calls() {
        let b = CpuBackend;
        let s = ConvShape {
            tokens: 1,
            channels: 1,
            width: 3,
        };
        let w = vec![1.0, 1.0, 1.0];
        let mut state = vec![0.0; 2];
        let mut out = vec![0.0; 1];
        for x in [1.0f32, 2.0, 3.0] {
            b.conv1d_silu(&vec![x], &w, &mut state, &mut out, s)
                .unwrap();
        }
        // Last output sums 1 + 2 + 3 before SiLU.
        assert!((f64::from(out[0]) - silu(6.0)).abs() < 1e-6);
        assert_eq!(state, vec![2.0, 3.0]);
    }

    #[test]
    fn gdn_first_step_from_zero_state() {
        // With S = 0: S' = β v ⊗ k̂ and o = S' q̂ / sqrt(dk) = β v (k̂·q̂) / sqrt(dk).
        let b = CpuBackend;
        let s = GdnShape {
            tokens: 1,
            k_heads: 1,
            v_heads: 1,
            k_dim: 2,
            v_dim: 2,
        };
        let qkv = vec![1.0, 0.0, 1.0, 0.0, 2.0, -4.0];
        let ab = vec![0.0, 0.0]; // β = 0.5
        let mut state = vec![0.0; 4];
        let mut out = vec![0.0; 2];
        b.gated_delta_net(&qkv, &ab, &vec![0.0], &vec![0.0], &mut state, &mut out, s)
            .unwrap();
        let kq = 1.0 / (1.0 + 1e-6); // both normalised vectors are (1/sqrt(1+1e-6), 0)
        let want = [0.5 * 2.0 * kq / 2f64.sqrt(), 0.5 * -4.0 * kq / 2f64.sqrt()];
        assert!(
            (f64::from(out[0]) - want[0]).abs() < 1e-6
                && (f64::from(out[1]) - want[1]).abs() < 1e-6,
            "{out:?}"
        );
    }
}
