//! RTX 5090 kernel set. Each override tries the tuned kernel and falls back to the portable one
//! when the kernel reports the shape as unsupported.

use velox_core::backend::{ActPolicy, ActQuant, WeightView};
use velox_core::check::to_i32;
use velox_core::{Error, Op, Result, WeightFormat};
use velox_cuda::{portable, status, CudaBuffer, CudaKernels};

extern "C" {
    fn velox_5090_matmul_small_m(x: *const f32, w: *const f32, y: *mut f32, m: i32, n: i32, k: i32) -> i32;
    fn velox_5090_matmul_tiled(x: *const f32, w: *const f32, y: *mut f32, m: i32, n: i32, k: i32) -> i32;
    fn velox_5090_linear_a4(
        x: *const f32,
        m: i32,
        w: *const u8,
        parent_rows: i64,
        k: i64,
        row_begin: i64,
        rows: i32,
        input_divisor: f32,
        y: *mut f32,
    ) -> i32;
    fn velox_5090_linear_a8(
        x: *const f32,
        m: i32,
        w: *const u8,
        parent_rows: i64,
        k: i64,
        row_begin: i64,
        rows: i32,
        y: *mut f32,
    ) -> i32;
}

/// Above this many rows the tiled SGEMM beats re-streaming weights per 8-row tile.
const SMALL_M_MAX: usize = 8;

/// Up to this many tokens `linear` stays on main's exact, bandwidth-bound GEMV; above it the
/// tensor-core GEMM with rounded activations takes over where the artifact allows.
const GEMV_MAX_M: usize = 8;

pub struct Rtx5090Kernels;

fn or_fallback(r: Result<()>, fallback: impl FnOnce() -> Result<()>) -> Result<()> {
    match r {
        Err(Error::Unsupported(_)) => fallback(),
        other => other,
    }
}

impl CudaKernels for Rtx5090Kernels {
    fn native_ops(&self) -> &'static [Op] {
        &[Op::MatMul, Op::Linear]
    }

    // Shapes were validated by CudaBackend before these are called.
    fn matmul(&self, x: &CudaBuffer, w: &CudaBuffer, y: &mut CudaBuffer, m: usize, n: usize, k: usize) -> Result<()> {
        let (mi, ni, ki) = (to_i32("m", m)?, to_i32("n", n)?, to_i32("k", k)?);
        let launch = if m <= SMALL_M_MAX { velox_5090_matmul_small_m } else { velox_5090_matmul_tiled };
        // SAFETY: extents validated by CudaBackend; kernels guard their own edges.
        let r = status(unsafe { launch(x.as_ptr(), w.as_ptr(), y.as_mut_ptr(), mi, ni, ki) }, "rtx5090 matmul");
        or_fallback(r, || portable::matmul(x, w, y, m, n, k))
    }

    fn linear_act(&self, w: &WeightView<'_, CudaBuffer>, m: usize) -> ActQuant {
        if m <= GEMV_MAX_M {
            return ActQuant::Exact;
        }
        match (w.format, w.act) {
            (WeightFormat::Nvfp4, ActPolicy::AllowA4 { divisor }) if w.k % 128 == 0 && w.row_begin % 128 == 0 => {
                ActQuant::Nvfp4 { divisor }
            }
            (WeightFormat::Fp8RowBf16, ActPolicy::AllowA8 | ActPolicy::AllowA4 { .. }) if w.k % 64 == 0 => {
                ActQuant::Fp8Row
            }
            _ => ActQuant::Exact,
        }
    }

    fn linear(&self, x: &CudaBuffer, w: &WeightView<'_, CudaBuffer>, y: &mut CudaBuffer, m: usize) -> Result<()> {
        let (mi, rows) = (to_i32("m", m)?, to_i32("rows", w.rows)?);
        let (pr, k, rb) = (w.parent_rows as i64, w.k as i64, w.row_begin as i64);
        let wp = w.buf.as_ptr().cast::<u8>();
        // linear_act screened the shapes. A launcher refusing one anyway is an error, not a
        // fallback: falling back would change the rounding linear_act reported.
        // SAFETY: extents validated by CudaBackend (check::linear).
        match self.linear_act(w, m) {
            ActQuant::Exact => portable::linear(x, w, y, m),
            ActQuant::Nvfp4 { divisor } => status(
                unsafe { velox_5090_linear_a4(x.as_ptr(), mi, wp, pr, k, rb, rows, divisor, y.as_mut_ptr()) },
                "rtx5090 linear a4",
            ),
            ActQuant::Fp8Row => status(
                unsafe { velox_5090_linear_a8(x.as_ptr(), mi, wp, pr, k, rb, rows, y.as_mut_ptr()) },
                "rtx5090 linear a8",
            ),
        }
    }
}
