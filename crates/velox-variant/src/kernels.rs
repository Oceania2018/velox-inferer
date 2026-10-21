//! RTX 5090 kernel set. Each override tries the tuned kernel and falls back to the portable one
//! when the kernel reports the shape as unsupported.

use velox_core::backend::{ActPolicy, ActQuant, AttnPrecision, AttnShape, FullAttnShape, WeightView};
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
    fn velox_5090_attention_full(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        out: *mut f32,
        tokens: i32,
        heads: i32,
        head_dim: i32,
    ) -> i32;
    fn velox_5090_attention_prefill(
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
    fn velox_5090_linear_skinny16(
        x: *const f32,
        m: i32,
        w: *const u8,
        parent_rows: i64,
        k: i64,
        row_begin: i64,
        rows: i32,
        format: i32,
        y: *mut f32,
    ) -> i32;
    fn velox_5090_linear_rowsplit(
        x: *const f32,
        m: i32,
        w: *const u8,
        parent_rows: i64,
        k: i64,
        row_begin: i64,
        rows: i32,
        bits: i32,
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

/// Single-token `linear` stays on main's exact, bandwidth-bound GEMV. From two tokens
/// (speculative verify, short chunks) the tensor-core GEMM with rounded activations takes over
/// where the artifact allows: its 16-token tile with split-K streams the weights once.
const GEMV_MAX_M: usize = 1;

/// From this many query tokens attention uses the tensor-core prefill kernel; below it the exact
/// split-KV kernel, which is bound by reading the cache anyway.
const FLASH_MIN_TOKENS: usize = 16;

/// From this many tokens groupwise-integer weights (vision tower, draft models) use the bf16
/// tensor-core kernel; below it the exact portable path.
const ROWSPLIT_MIN_M: usize = 64;

/// Up to this many tokens, A16Only NVFP4 and q4/q8 weights (draft model, proposal head) use the
/// weight-streaming bf16 tensor-core kernel.
const SKINNY16_MAX_M: usize = 16;

/// `velox_5090_linear_skinny16`'s format code for a weight it serves at `m` tokens.
fn skinny16_format(w: &WeightView<'_, CudaBuffer>, m: usize) -> Option<i32> {
    if !(2..=SKINNY16_MAX_M).contains(&m) {
        return None;
    }
    match w.format {
        WeightFormat::Nvfp4 if w.k % 128 == 0 && w.row_begin % 64 == 0 => Some(0),
        WeightFormat::Q4G64 => Some(4),
        WeightFormat::Q8G32 => Some(8),
        _ => None,
    }
}

pub struct Rtx5090Kernels;

fn or_fallback(r: Result<()>, fallback: impl FnOnce() -> Result<()>) -> Result<()> {
    match r {
        Err(Error::Unsupported(_)) => fallback(),
        other => other,
    }
}

impl CudaKernels for Rtx5090Kernels {
    fn native_ops(&self) -> &'static [Op] {
        &[Op::MatMul, Op::Linear, Op::Attention, Op::AttentionFull]
    }

    // Shapes were validated by CudaBackend before these are called.
    fn matmul(&self, x: &CudaBuffer, w: &CudaBuffer, y: &mut CudaBuffer, m: usize, n: usize, k: usize) -> Result<()> {
        let (mi, ni, ki) = (to_i32("m", m)?, to_i32("n", n)?, to_i32("k", k)?);
        let launch = if m <= SMALL_M_MAX { velox_5090_matmul_small_m } else { velox_5090_matmul_tiled };
        // SAFETY: extents validated by CudaBackend; kernels guard their own edges.
        let r = status(unsafe { launch(x.as_ptr(), w.as_ptr(), y.as_mut_ptr(), mi, ni, ki) }, "rtx5090 matmul");
        or_fallback(r, || portable::matmul(x, w, y, m, n, k))
    }

    fn attention_precision(&self, s: AttnShape) -> AttnPrecision {
        if s.tokens >= FLASH_MIN_TOKENS && s.head_dim == 256 && s.heads % s.kv_heads == 0 && s.kv_len() <= i32::MAX as usize {
            AttnPrecision::Fp8QkF16Pv
        } else {
            AttnPrecision::Exact
        }
    }

    fn attention(&self, q: &CudaBuffer, k: &CudaBuffer, v: &CudaBuffer, out: &mut CudaBuffer, s: AttnShape) -> Result<()> {
        if self.attention_precision(s) == AttnPrecision::Exact {
            return portable::attention(q, k, v, out, s);
        }
        // SAFETY: extents validated by CudaBackend (check::attention).
        status(
            unsafe {
                velox_5090_attention_prefill(
                    q.as_ptr(),
                    k.as_ptr(),
                    v.as_ptr(),
                    out.as_mut_ptr(),
                    to_i32("tokens", s.tokens)?,
                    to_i32("heads", s.heads)?,
                    to_i32("kv_heads", s.kv_heads)?,
                    to_i32("head_dim", s.head_dim)?,
                    to_i32("pos0", s.pos0)?,
                    s.scale(),
                )
            },
            "rtx5090 attention prefill",
        )
    }

    fn attention_full_precision(&self, s: FullAttnShape) -> AttnPrecision {
        if s.tokens >= FLASH_MIN_TOKENS && s.head_dim <= 128 {
            AttnPrecision::F16
        } else {
            AttnPrecision::Exact
        }
    }

    fn attention_full(&self, q: &CudaBuffer, k: &CudaBuffer, v: &CudaBuffer, out: &mut CudaBuffer, s: FullAttnShape) -> Result<()> {
        if self.attention_full_precision(s) == AttnPrecision::Exact {
            return portable::attention_full(q, k, v, out, s);
        }
        // SAFETY: extents validated by CudaBackend (check::attention_full).
        status(
            unsafe {
                velox_5090_attention_full(
                    q.as_ptr(),
                    k.as_ptr(),
                    v.as_ptr(),
                    out.as_mut_ptr(),
                    to_i32("tokens", s.tokens)?,
                    to_i32("heads", s.heads)?,
                    to_i32("head_dim", s.head_dim)?,
                )
            },
            "rtx5090 attention_full",
        )
    }

    fn linear_act(&self, w: &WeightView<'_, CudaBuffer>, m: usize) -> ActQuant {
        if m >= ROWSPLIT_MIN_M && w.format.row_split().is_some() {
            return ActQuant::Bf16;
        }
        if m <= GEMV_MAX_M {
            return ActQuant::Exact;
        }
        if let Some(f) = skinny16_format(w, m) {
            // NVFP4 views the FP4 tensor-core path serves (A4 allowed, 128-row aligned) keep it.
            let a4 = matches!(w.act, ActPolicy::AllowA4 { .. }) && w.row_begin % 128 == 0;
            if f != 0 || !a4 {
                return ActQuant::Bf16;
            }
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
            ActQuant::Bf16 => {
                if let Some(f) = skinny16_format(w, m) {
                    return status(
                        unsafe { velox_5090_linear_skinny16(x.as_ptr(), mi, wp, pr, k, rb, rows, f, y.as_mut_ptr()) },
                        "rtx5090 linear skinny16",
                    );
                }
                let bits = w.format.row_split().map_or(0, |r| r.bits as i32);
                status(
                    unsafe { velox_5090_linear_rowsplit(x.as_ptr(), mi, wp, pr, k, rb, rows, bits, y.as_mut_ptr()) },
                    "rtx5090 linear rowsplit",
                )
            }
        }
    }
}
