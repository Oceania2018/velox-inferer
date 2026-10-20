//! RTX 5090 kernel set. Each override tries the tuned kernel and falls back to the portable one
//! when the kernel reports the shape as unsupported.

use velox_core::backend::AttnShape;
use velox_core::check::to_i32;
use velox_core::{Error, Op, Result};
use velox_cuda::{portable, status, CudaBuffer, CudaKernels};

extern "C" {
    fn velox_5090_matmul_small_m(x: *const f32, w: *const f32, y: *mut f32, m: i32, n: i32, k: i32) -> i32;
    fn velox_5090_matmul_tiled(x: *const f32, w: *const f32, y: *mut f32, m: i32, n: i32, k: i32) -> i32;
    fn velox_5090_attention(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        out: *mut f32,
        tokens: i32,
        heads: i32,
        kv_heads: i32,
        head_dim: i32,
        pos0: i32,
        scale: f32,
    ) -> i32;
}

/// Above this many rows the tiled SGEMM beats re-streaming weights per 8-row tile.
const SMALL_M_MAX: usize = 8;

pub struct Rtx5090Kernels;

fn or_fallback(r: Result<()>, fallback: impl FnOnce() -> Result<()>) -> Result<()> {
    match r {
        Err(Error::Unsupported(_)) => fallback(),
        other => other,
    }
}

impl CudaKernels for Rtx5090Kernels {
    fn native_ops(&self) -> &'static [Op] {
        &[Op::MatMul, Op::Attention]
    }

    // Shapes were validated by CudaBackend before these are called.
    fn matmul(&self, x: &CudaBuffer, w: &CudaBuffer, y: &mut CudaBuffer, m: usize, n: usize, k: usize) -> Result<()> {
        let (mi, ni, ki) = (to_i32("m", m)?, to_i32("n", n)?, to_i32("k", k)?);
        let launch = if m <= SMALL_M_MAX { velox_5090_matmul_small_m } else { velox_5090_matmul_tiled };
        // SAFETY: extents validated by CudaBackend; kernels guard their own edges.
        let r = status(unsafe { launch(x.as_ptr(), w.as_ptr(), y.as_mut_ptr(), mi, ni, ki) }, "rtx5090 matmul");
        or_fallback(r, || portable::matmul(x, w, y, m, n, k))
    }

    fn attention(&self, q: &CudaBuffer, k: &CudaBuffer, v: &CudaBuffer, out: &mut CudaBuffer, s: AttnShape) -> Result<()> {
        to_i32("kv_len", s.kv_len())?;
        // SAFETY: extents validated by CudaBackend.
        let r = status(
            unsafe {
                velox_5090_attention(
                    q.as_ptr(),
                    k.as_ptr(),
                    v.as_ptr(),
                    out.as_mut_ptr(),
                    to_i32("tokens", s.tokens)?,
                    to_i32("heads", s.heads)?,
                    to_i32("kv_heads", s.kv_heads)?,
                    to_i32("head_dim", s.head_dim)?,
                    s.pos0 as i32,
                    s.scale(),
                )
            },
            "rtx5090 attention",
        );
        or_fallback(r, || portable::attention(q, k, v, out, s))
    }
}
