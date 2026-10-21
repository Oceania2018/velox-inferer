//! A `Backend` wrapper that synchronizes after every op and accumulates wall time per op, for
//! `velox generate --profile-ops`. Synchronizing serializes the GPU, so totals are slightly
//! higher than unprofiled runs; the breakdown is what matters.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::time::Instant;

use velox_core::backend::{
    ActQuant, AttnShape, Backend, BackendKind, BlockAttnShape, ConvShape, Copy2dShape,
    DynConvShape, FullAttnShape, GdnShape, GeluKind, KvStoreShape, NormKind, RopePosShape,
    RopeShape, WeightView,
};
use velox_core::{DeviceProfile, Op, Result};

pub struct Profiled<B: Backend> {
    pub inner: B,
    times: RefCell<BTreeMap<String, (f64, u64)>>,
}

impl<B: Backend> Profiled<B> {
    pub fn new(inner: B) -> Self {
        Profiled {
            inner,
            times: RefCell::new(BTreeMap::new()),
        }
    }

    fn time<T>(&self, name: &str, f: impl FnOnce() -> Result<T>) -> Result<T> {
        let t0 = Instant::now();
        let r = f()?;
        self.inner.synchronize()?;
        let mut m = self.times.borrow_mut();
        let e = m.entry(name.to_string()).or_insert((0.0, 0));
        e.0 += t0.elapsed().as_secs_f64();
        e.1 += 1;
        Ok(r)
    }

    pub fn report(&self) -> String {
        let m = self.times.borrow();
        let total: f64 = m.values().map(|v| v.0).sum();
        let mut rows: Vec<_> = m.iter().collect();
        rows.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0));
        let mut s = format!("op breakdown (synchronized), total {total:.2}s\n");
        for (name, (t, n)) in rows {
            s.push_str(&format!(
                "  {name:<28} {t:>8.3}s {:>5.1}%  {n:>7} calls\n",
                100.0 * t / total.max(1e-12)
            ));
        }
        s
    }

    pub fn reset(&self) {
        self.times.borrow_mut().clear();
    }
}

impl<B: Backend> Backend for Profiled<B> {
    type Buffer = B::Buffer;

    fn kind(&self) -> BackendKind {
        self.inner.kind()
    }
    fn describe(&self) -> String {
        format!("{} [profiled]", self.inner.describe())
    }
    fn profile(&self) -> Option<&'static DeviceProfile> {
        self.inner.profile()
    }
    fn native_ops(&self) -> Vec<Op> {
        self.inner.native_ops()
    }
    fn alloc(&self, len: usize) -> Result<B::Buffer> {
        self.inner.alloc(len)
    }
    fn upload(&self, data: &[f32]) -> Result<B::Buffer> {
        self.inner.upload(data)
    }
    fn download(&self, buf: &B::Buffer, out: &mut [f32]) -> Result<()> {
        self.inner.download(buf, out)
    }
    fn len(&self, buf: &B::Buffer) -> usize {
        self.inner.len(buf)
    }
    fn upload_bytes(&self, data: &[u8]) -> Result<B::Buffer> {
        self.inner.upload_bytes(data)
    }
    fn alloc_host(&self, len: usize) -> Result<B::Buffer> {
        self.inner.alloc_host(len)
    }
    fn upload_bytes_host(&self, data: &[u8]) -> Result<B::Buffer> {
        self.inner.upload_bytes_host(data)
    }
    fn copy_batch(
        &self,
        dsts: &mut [&mut B::Buffer],
        ops: &[velox_core::CopySpec<'_, B::Buffer>],
    ) -> Result<()> {
        self.time("copy_batch", || self.inner.copy_batch(dsts, ops))
    }
    fn host_bytes<'a>(&self, buf: &'a B::Buffer) -> Option<&'a [u8]> {
        self.inner.host_bytes(buf)
    }
    fn host_bytes_mut<'a>(&self, buf: &'a mut B::Buffer) -> Option<&'a mut [u8]> {
        self.inner.host_bytes_mut(buf)
    }
    fn synchronize(&self) -> Result<()> {
        self.inner.synchronize()
    }
    fn bind_thread(&self) -> Result<()> {
        self.inner.bind_thread()
    }
    fn matmul(
        &self,
        x: &B::Buffer,
        w: &B::Buffer,
        y: &mut B::Buffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        self.time("matmul", || self.inner.matmul(x, w, y, m, n, k))
    }
    fn linear(
        &self,
        x: &B::Buffer,
        w: &WeightView<'_, B::Buffer>,
        y: &mut B::Buffer,
        m: usize,
    ) -> Result<()> {
        let name = format!(
            "linear {} {}",
            w.format.name(),
            if m == 1 { "m=1" } else { "m>1" }
        );
        self.time(&name, || self.inner.linear(x, w, y, m))
    }
    fn linear_act(&self, w: &WeightView<'_, B::Buffer>, m: usize) -> ActQuant {
        self.inner.linear_act(w, m)
    }
    fn embed(
        &self,
        ids: &[u32],
        table: &WeightView<'_, B::Buffer>,
        out: &mut B::Buffer,
    ) -> Result<()> {
        self.time("embed", || self.inner.embed(ids, table, out))
    }
    fn rms_norm(
        &self,
        x: &B::Buffer,
        w: &B::Buffer,
        y: &mut B::Buffer,
        rows: usize,
        dim: usize,
        eps: f32,
        kind: NormKind,
    ) -> Result<()> {
        self.time("rms_norm", || {
            self.inner.rms_norm(x, w, y, rows, dim, eps, kind)
        })
    }
    fn rms_norm_gated(
        &self,
        x: &B::Buffer,
        w: &B::Buffer,
        gate: &mut B::Buffer,
        rows: usize,
        dim: usize,
        eps: f32,
        kind: NormKind,
    ) -> Result<()> {
        self.time("rms_norm_gated", || {
            self.inner.rms_norm_gated(x, w, gate, rows, dim, eps, kind)
        })
    }
    fn rope(&self, x: &mut B::Buffer, s: RopeShape) -> Result<()> {
        self.time("rope", || self.inner.rope(x, s))
    }
    fn silu_mul(&self, g: &mut B::Buffer, u: &B::Buffer, len: usize) -> Result<()> {
        self.time("silu_mul", || self.inner.silu_mul(g, u, len))
    }
    fn sigmoid_mul(&self, x: &mut B::Buffer, g: &B::Buffer, len: usize) -> Result<()> {
        self.time("sigmoid_mul", || self.inner.sigmoid_mul(x, g, len))
    }
    fn add(&self, acc: &mut B::Buffer, x: &B::Buffer, len: usize) -> Result<()> {
        self.time("add", || self.inner.add(acc, x, len))
    }
    fn copy(
        &self,
        src: &B::Buffer,
        so: usize,
        dst: &mut B::Buffer,
        d: usize,
        len: usize,
    ) -> Result<()> {
        self.time("copy", || self.inner.copy(src, so, dst, d, len))
    }
    fn attention(
        &self,
        q: &B::Buffer,
        k: &B::Buffer,
        v: &B::Buffer,
        out: &mut B::Buffer,
        s: AttnShape,
    ) -> Result<()> {
        let name = if s.tokens == 1 {
            "attention decode"
        } else {
            "attention prefill"
        };
        self.time(name, || self.inner.attention(q, k, v, out, s))
    }
    fn kv_store(
        &self,
        k: &B::Buffer,
        v: &B::Buffer,
        kc: &mut B::Buffer,
        vc: &mut B::Buffer,
        s: KvStoreShape,
    ) -> Result<()> {
        self.time("kv_store", || self.inner.kv_store(k, v, kc, vc, s))
    }
    fn kv_load(
        &self,
        kc: &B::Buffer,
        vc: &B::Buffer,
        k: &mut B::Buffer,
        v: &mut B::Buffer,
        s: KvStoreShape,
    ) -> Result<()> {
        self.time("kv_load", || self.inner.kv_load(kc, vc, k, v, s))
    }
    fn attention_precision(&self, s: AttnShape) -> velox_core::AttnPrecision {
        self.inner.attention_precision(s)
    }
    fn attention_full_precision(&self, s: FullAttnShape) -> velox_core::AttnPrecision {
        self.inner.attention_full_precision(s)
    }
    fn conv1d_silu(
        &self,
        x: &B::Buffer,
        w: &B::Buffer,
        st: &mut B::Buffer,
        out: &mut B::Buffer,
        s: ConvShape,
    ) -> Result<()> {
        self.time("conv1d_silu", || self.inner.conv1d_silu(x, w, st, out, s))
    }
    fn gated_delta_net(
        &self,
        qkv: &B::Buffer,
        ab: &B::Buffer,
        a_log: &B::Buffer,
        dt: &B::Buffer,
        st: &mut B::Buffer,
        out: &mut B::Buffer,
        s: GdnShape,
    ) -> Result<()> {
        let name = if s.tokens == 1 {
            "gated_delta_net decode"
        } else {
            "gated_delta_net prefill"
        };
        self.time(name, || {
            self.inner.gated_delta_net(qkv, ab, a_log, dt, st, out, s)
        })
    }
    fn gdn_replay(
        &self,
        layers: &mut [velox_core::GdnReplayLayer<'_, B::Buffer>],
        conv: ConvShape,
        gdn: GdnShape,
    ) -> Result<()> {
        self.time("gdn_replay", || self.inner.gdn_replay(layers, conv, gdn))
    }
    fn attention_full(
        &self,
        q: &B::Buffer,
        k: &B::Buffer,
        v: &B::Buffer,
        out: &mut B::Buffer,
        s: FullAttnShape,
    ) -> Result<()> {
        self.time("attention_full (vision)", || {
            self.inner.attention_full(q, k, v, out, s)
        })
    }
    fn layer_norm(
        &self,
        x: &B::Buffer,
        w: &B::Buffer,
        b: &B::Buffer,
        y: &mut B::Buffer,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<()> {
        self.time("layer_norm", || {
            self.inner.layer_norm(x, w, b, y, rows, dim, eps)
        })
    }
    fn add_bias(&self, x: &mut B::Buffer, bias: &B::Buffer, rows: usize, dim: usize) -> Result<()> {
        self.time("add_bias", || self.inner.add_bias(x, bias, rows, dim))
    }
    fn gelu(&self, x: &mut B::Buffer, len: usize, kind: GeluKind) -> Result<()> {
        self.time("gelu", || self.inner.gelu(x, len, kind))
    }
    fn split_cols(
        &self,
        src: &B::Buffer,
        rows: usize,
        src_cols: usize,
        outs: &mut [(&mut B::Buffer, usize, usize)],
    ) -> Result<()> {
        self.time("split_cols", || {
            self.inner.split_cols(src, rows, src_cols, outs)
        })
    }
    #[allow(clippy::too_many_arguments)]
    fn swiglu(
        &self,
        src: &B::Buffer,
        rows: usize,
        src_cols: usize,
        gate: usize,
        up: usize,
        width: usize,
        out: &mut B::Buffer,
    ) -> Result<()> {
        self.time("swiglu", || {
            self.inner.swiglu(src, rows, src_cols, gate, up, width, out)
        })
    }
    fn concat_cols(
        &self,
        a: &B::Buffer,
        b: &B::Buffer,
        out: &mut B::Buffer,
        rows: usize,
        a_cols: usize,
        b_cols: usize,
    ) -> Result<()> {
        self.time("concat_cols", || {
            self.inner.concat_cols(a, b, out, rows, a_cols, b_cols)
        })
    }
    fn top_k(
        &self,
        x: &B::Buffer,
        rows: usize,
        row_len: usize,
        limit: usize,
        k: usize,
    ) -> Result<Vec<Vec<(u32, f32)>>> {
        self.time("top_k", || self.inner.top_k(x, rows, row_len, limit, k))
    }
    fn rope_positions(&self, x: &mut B::Buffer, p: &[u32], s: RopePosShape) -> Result<()> {
        self.time("rope_positions", || self.inner.rope_positions(x, p, s))
    }
    fn copy2d(&self, src: &B::Buffer, dst: &mut B::Buffer, s: Copy2dShape) -> Result<()> {
        self.time("copy2d", || self.inner.copy2d(src, dst, s))
    }
    fn dyn_conv(
        &self,
        x: &B::Buffer,
        base: &B::Buffer,
        delta: &B::Buffer,
        out: &mut B::Buffer,
        s: DynConvShape,
    ) -> Result<()> {
        self.time("dyn_conv", || self.inner.dyn_conv(x, base, delta, out, s))
    }
    #[allow(clippy::too_many_arguments)]
    fn attention_block(
        &self,
        q: &B::Buffer,
        k_ctx: &B::Buffer,
        v_ctx: &B::Buffer,
        k_blk: &B::Buffer,
        v_blk: &B::Buffer,
        out: &mut B::Buffer,
        s: BlockAttnShape,
    ) -> Result<()> {
        self.time("attention_block", || {
            self.inner
                .attention_block(q, k_ctx, v_ctx, k_blk, v_blk, out, s)
        })
    }
}
