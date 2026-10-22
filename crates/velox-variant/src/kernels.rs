//! RTX 5060 Ti kernel set (ported from rtx-5090-32g). Each override tries the tuned kernel and falls back to the portable one
//! when the kernel reports the shape as unsupported.

use velox_core::backend::{
    ActPolicy, ActQuant, AttnPrecision, AttnShape, FullAttnShape, WeightView,
};
use velox_core::check::to_i32;
use velox_core::{Error, KvFormat, Op, Result, WeightFormat};
use velox_cuda::{portable, status, CudaBuffer, CudaKernels};

extern "C" {
    fn velox_5060_matmul_small_m(
        x: *const f32,
        w: *const f32,
        y: *mut f32,
        m: i32,
        n: i32,
        k: i32,
    ) -> i32;
    fn velox_5060_matmul_tiled(
        x: *const f32,
        w: *const f32,
        y: *mut f32,
        m: i32,
        n: i32,
        k: i32,
    ) -> i32;
    fn velox_5060_linear_a4(
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
    fn velox_5060_attention_full(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        out: *mut f32,
        tokens: i32,
        heads: i32,
        head_dim: i32,
    ) -> i32;
    fn velox_5060_attention_prefill(
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
    fn velox_5060_attention_verify_nvfp4(
        q: *const f32,
        k: *const u8,
        v: *const u8,
        out: *mut f32,
        tokens: i32,
        heads: i32,
        kv_heads: i32,
        head_dim: i32,
        pos0: i32,
        scale: f32,
        norm: f32,
        params: *const i64,
        kv_grid: i32,
    ) -> i32;
    fn velox_5060_linear_a16_fp8(
        x: *const f32,
        m: i32,
        w: *const u8,
        parent_rows: i64,
        k: i64,
        row_begin: i64,
        rows: i32,
        y: *mut f32,
    ) -> i32;
    fn velox_5060_linear_skinny16(
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
    fn velox_5060_linear_swiglu_iq3_direct(
        x: *const f32,
        m: i32,
        w: *const u8,
        parent_rows: i64,
        gate_begin: i64,
        up_w: *const u8,
        up_parent_rows: i64,
        up_begin: i64,
        k: i64,
        rows: i32,
        out: *mut f32,
    ) -> i32;
    fn velox_5060_linear_swiglu_skinny_iq3(
        x: *const f32,
        m: i32,
        w: *const u8,
        parent_rows: i64,
        gate_begin: i64,
        up_w: *const u8,
        up_parent_rows: i64,
        up_begin: i64,
        k: i64,
        rows: i32,
        out: *mut f32,
    ) -> i32;
    fn velox_5060_linear_nvfp4_sliced(
        x: *const f32,
        m: i32,
        w: *const u8,
        parent_rows: i64,
        k: i64,
        row_begin: i64,
        rows: i32,
        y: *mut f32,
    ) -> i32;
    fn velox_5060_linear_rs_sliced(
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
    fn velox_5060_linear_iq3_direct(
        x: *const f32,
        m: i32,
        w: *const u8,
        parent_rows: i64,
        k: i64,
        row_begin: i64,
        rows: i32,
        y: *mut f32,
    ) -> i32;
    fn velox_5060_gemv(
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
    fn velox_5060_linear_skinny_rs(
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
    fn velox_5060_linear_skinny_iq3(
        x: *const f32,
        m: i32,
        w: *const u8,
        parent_rows: i64,
        k: i64,
        row_begin: i64,
        rows: i32,
        y: *mut f32,
    ) -> i32;
    fn velox_5060_linear_rowsplit(
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
    fn velox_5060_linear_nvfp4_mma(
        x: *const f32,
        m: i32,
        w: *const u8,
        parent_rows: i64,
        k: i64,
        row_begin: i64,
        rows: i32,
        y: *mut f32,
    ) -> i32;
    fn velox_5060_linear_iq3_q8(
        x: *const f32,
        m: i32,
        w: *const u8,
        parent_rows: i64,
        k: i64,
        row_begin: i64,
        rows: i32,
        y: *mut f32,
    ) -> i32;
    fn velox_5060_linear_swiglu_iq3_q8(
        x: *const f32,
        m: i32,
        w: *const u8,
        parent_rows: i64,
        gate_begin: i64,
        up_w: *const u8,
        up_parent_rows: i64,
        up_begin: i64,
        k: i64,
        rows: i32,
        out: *mut f32,
    ) -> i32;
    fn velox_5060_linear_rs_q8(
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
    fn velox_5060_linear_a8(
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

/// Decode, verify and prefill over an NVFP4 cache: f16 tensor-core flash attention that decodes
/// the cache tiles in shared memory (the f32-cache prefill kernel does not read packed caches).
fn verify_nvfp4(s: &AttnShape) -> bool {
    s.kv == KvFormat::Nvfp4
        && s.head_dim == 256
        && s.heads % s.kv_heads == 0
        && s.kv_len() <= i32::MAX as usize
}

/// From this many tokens groupwise-integer weights (vision tower, draft models) use the bf16
/// tensor-core kernel; below it the exact portable path.
const ROWSPLIT_MIN_M: usize = 64;

/// Up to this many tokens, A16Only NVFP4 and q4/q8 weights (draft model, proposal head) use the
/// weight-streaming bf16 tensor-core kernel.
const SKINNY16_MAX_M: usize = 16;

/// Batched drafting (up to 6 sequences x 8-row blocks, or their 7 proposal rows each) runs the
/// skinny kernel in blocks of 16 rows up to this many rows, rather than the exact portable path.
/// Groupwise-integer weights hand over to the row-split kernel at `ROWSPLIT_MIN_M`.
const SKINNY_MAX_M: usize = 96;

/// Skinny FP8 row-scaled linears (speculative verify, drafts) take the one-kernel bf16-activation
/// path: no activation quantizer or split-K pass, and bf16 activations (finer than E4M3).
fn a16_fp8(w: &WeightView<'_, CudaBuffer>, m: usize) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VELOX_NO_A16FP8").is_none())
        && w.format == WeightFormat::Fp8RowBf16
        && (2..=A16_FP8_MAX_M).contains(&m)
        && w.k % 256 == 0
}

/// Batched speculative verify (up to 6 sequences x 16 rows) keeps the bf16-activation path: one
/// pass over the weights per 64 rows.
const A16_FP8_MAX_M: usize = 96;
const A16_FP8_BLOCK_M: usize = 64;

/// Format code of the skinny kernel serving a weight at `m` tokens: NVFP4 (0) runs
/// `velox_5060_linear_skinny16`, groupwise q4/q5/q8 (4/5/8) `velox_5060_linear_skinny_rs`. Single
/// tokens stay on main's GEMV except for q5, which has no fast GEMV there.
fn skinny16_format(w: &WeightView<'_, CudaBuffer>, m: usize) -> Option<i32> {
    let max = if w.format.row_split().is_some() {
        ROWSPLIT_MIN_M - 1
    } else {
        SKINNY_MAX_M
    };
    let min = if w.format == WeightFormat::Q5G64 {
        1
    } else {
        2
    };
    if !(min..=max).contains(&m) {
        return None;
    }
    match w.format {
        WeightFormat::Nvfp4 if w.k % 128 == 0 && w.row_begin % 64 == 0 => Some(0),
        WeightFormat::Q4G64 => Some(4),
        WeightFormat::Q5G64 => Some(5),
        WeightFormat::Q8G32 => Some(8),
        _ => None,
    }
}

/// Single-token q5_g64 decode: the register-streaming f32 GEMV (exact activations). From two tokens
/// the tensor-core kernels win (measured: q5 [12288,5120] m=1 112 us vs 179 us skinny; m=2 184 vs
/// 183). IQ3_XXS decodes faster on its direct tensor-core kernel (m=1 205 vs 217 us GEMV).
fn gemv_format(w: &WeightView<'_, CudaBuffer>, m: usize) -> Option<i32> {
    if m != 1 {
        return None;
    }
    match w.format {
        WeightFormat::Q5G64 if w.k % 32 == 0 => Some(5),
        _ => None,
    }
}

/// Which IQ3_XXS kernel serves a 16-token block of `m` tokens: the register-direct one (codes
/// straight to registers) or the shared-memory staged one. Measured (us, gate/up [34816,5120] /
/// down [5120,17408]): m=1 207 / 99 direct vs 215 / 118 staged; m=8 235 / 110 vs 213 / 120;
/// m=16 238 / 151 vs 239 / 137.
fn iq3_direct(w: &WeightView<'_, CudaBuffer>, m: usize) -> bool {
    let blocks = w.k / 256;
    m.min(SKINNY16_MAX_M) <= 2 || (blocks > 32 && m.min(SKINNY16_MAX_M) <= 8)
}

/// IQ3_XXS at any token count: the weight-streaming bf16 kernels (prefill in 64-token tiles).
fn skinny_iq3(w: &WeightView<'_, CudaBuffer>, m: usize) -> bool {
    w.format == WeightFormat::Iq3Xxs && m >= 1
}

/// Prefill on int8 tensor cores (ActQuant::Int8G256, ninfer's int8 path and thresholds on this
/// card): IQ3_XXS from this many tokens, q4/q5 from `RS_Q8_MIN_M`.
const IQ3_Q8_MIN_M: usize = 33;
const RS_Q8_MIN_M: usize = 64;

/// Format code of the int8 prefill kernel serving `w` at `m` tokens (3: IQ3_XXS, 4/5: q4/q5), when
/// its activation policy permits 8-bit activations. `VELOX_NO_Q8` turns the path off.
fn q8_format(w: &WeightView<'_, CudaBuffer>, m: usize) -> Option<i32> {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VELOX_NO_Q8").is_none())
        || !matches!(w.act, ActPolicy::AllowA8 | ActPolicy::AllowA4 { .. })
        || w.k % 256 != 0
    {
        return None;
    }
    match w.format {
        WeightFormat::Iq3Xxs if m >= IQ3_Q8_MIN_M => Some(3),
        WeightFormat::Q4G64 if m >= RS_Q8_MIN_M => Some(4),
        WeightFormat::Q5G64 if m >= RS_Q8_MIN_M => Some(5),
        _ => None,
    }
}

/// NVFP4 views the FP4-activation path does not serve (A16-only, or not 128-row aligned) from
/// this many tokens: the tiled bf16 tensor-core kernel (ninfer's prefill kernel and schedule).
/// Below, the sliced-K kernel in 16-token blocks.
const NVFP4_MMA_MIN_M: usize = 33;

fn nvfp4_mma(w: &WeightView<'_, CudaBuffer>, m: usize) -> bool {
    let a4 = matches!(w.act, ActPolicy::AllowA4 { .. }) && w.row_begin % 128 == 0 && w.k % 128 == 0;
    w.format == WeightFormat::Nvfp4 && w.k % 64 == 0 && m >= NVFP4_MMA_MIN_M && !a4
}

pub struct Rtx5060Kernels;

fn or_fallback(r: Result<()>, fallback: impl FnOnce() -> Result<()>) -> Result<()> {
    match r {
        Err(Error::Unsupported(_)) => fallback(),
        other => other,
    }
}

impl CudaKernels for Rtx5060Kernels {
    fn native_ops(&self) -> &'static [Op] {
        &[
            Op::MatMul,
            Op::Linear,
            Op::Attention,
            Op::AttentionFull,
            Op::LinearSwiglu,
        ]
    }

    // Shapes were validated by CudaBackend before these are called.
    fn matmul(
        &self,
        x: &CudaBuffer,
        w: &CudaBuffer,
        y: &mut CudaBuffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let (mi, ni, ki) = (to_i32("m", m)?, to_i32("n", n)?, to_i32("k", k)?);
        let launch = if m <= SMALL_M_MAX {
            velox_5060_matmul_small_m
        } else {
            velox_5060_matmul_tiled
        };
        // SAFETY: extents validated by CudaBackend; kernels guard their own edges.
        let r = status(
            unsafe { launch(x.as_ptr(), w.as_ptr(), y.as_mut_ptr(), mi, ni, ki) },
            "rtx5060 matmul",
        );
        or_fallback(r, || portable::matmul(x, w, y, m, n, k))
    }

    fn attention_precision(&self, s: AttnShape) -> AttnPrecision {
        if verify_nvfp4(&s) {
            return AttnPrecision::F16;
        }
        // The flash kernel reads f32 caches; other packed caches take the portable KV path.
        if s.kv == KvFormat::F32
            && s.tokens >= FLASH_MIN_TOKENS
            && s.head_dim == 256
            && s.heads % s.kv_heads == 0
            && s.kv_len() <= i32::MAX as usize
        {
            AttnPrecision::Fp8QkF16Pv
        } else {
            AttnPrecision::Exact
        }
    }

    fn attention(
        &self,
        q: &CudaBuffer,
        k: &CudaBuffer,
        v: &CudaBuffer,
        out: &mut CudaBuffer,
        s: AttnShape,
    ) -> Result<()> {
        if verify_nvfp4(&s) {
            velox_core::check::attention(q.len(), k.len(), v.len(), out.len(), &s)?;
            // Graph steps stage the caches too: a recording serves any sequence.
            let Some(params) =
                portable::graph_params(&[s.pos0 as i64, k.as_ptr() as i64, v.as_ptr() as i64])?
            else {
                return Ok(()); // graph replay: the recorded launch reads the staged values
            };
            // SAFETY: extents validated above (caches sized for the format).
            return status(
                unsafe {
                    velox_5060_attention_verify_nvfp4(
                        q.as_ptr(),
                        k.as_ptr().cast(),
                        v.as_ptr().cast(),
                        out.as_mut_ptr(),
                        to_i32("tokens", s.tokens)?,
                        to_i32("heads", s.heads)?,
                        to_i32("kv_heads", s.kv_heads)?,
                        to_i32("head_dim", s.head_dim)?,
                        s.pos0 as i32,
                        s.scale(),
                        velox_core::kv::hadamard_norm(s.head_dim),
                        params,
                        to_i32("kv grid", portable::kv_grid(s.kv_len()))?,
                    )
                },
                "rtx5060 attention verify (nvfp4)",
            );
        }
        if self.attention_precision(s) == AttnPrecision::Exact {
            return portable::attention(q, k, v, out, s);
        }
        // SAFETY: extents validated by CudaBackend (check::attention).
        status(
            unsafe {
                velox_5060_attention_prefill(
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
            "rtx5060 attention prefill",
        )
    }

    fn attention_full_precision(&self, s: FullAttnShape) -> AttnPrecision {
        if s.tokens >= FLASH_MIN_TOKENS && s.head_dim <= 128 {
            AttnPrecision::F16
        } else {
            AttnPrecision::Exact
        }
    }

    fn attention_full(
        &self,
        q: &CudaBuffer,
        k: &CudaBuffer,
        v: &CudaBuffer,
        out: &mut CudaBuffer,
        s: FullAttnShape,
    ) -> Result<()> {
        if self.attention_full_precision(s) == AttnPrecision::Exact {
            return portable::attention_full(q, k, v, out, s);
        }
        // SAFETY: extents validated by CudaBackend (check::attention_full).
        status(
            unsafe {
                velox_5060_attention_full(
                    q.as_ptr(),
                    k.as_ptr(),
                    v.as_ptr(),
                    out.as_mut_ptr(),
                    to_i32("tokens", s.tokens)?,
                    to_i32("heads", s.heads)?,
                    to_i32("head_dim", s.head_dim)?,
                )
            },
            "rtx5060 attention_full",
        )
    }

    fn linear_act(&self, w: &WeightView<'_, CudaBuffer>, m: usize) -> ActQuant {
        if q8_format(w, m).is_some() {
            return ActQuant::Int8G256;
        }
        if m >= ROWSPLIT_MIN_M && w.format.row_split().is_some() {
            return ActQuant::Bf16;
        }
        if gemv_format(w, m).is_some() {
            return ActQuant::Exact;
        }
        if nvfp4_mma(w, m) {
            return ActQuant::Bf16;
        }
        if skinny_iq3(w, m) || (m == 1 && skinny16_format(w, m).is_some()) {
            return ActQuant::Bf16;
        }
        if m <= GEMV_MAX_M {
            return ActQuant::Exact;
        }
        if a16_fp8(w, m) {
            return ActQuant::Bf16;
        }
        if let Some(f) = skinny16_format(w, m) {
            // NVFP4 views the FP4 tensor-core path serves (A4 allowed, 128-row aligned) keep it.
            let a4 = matches!(w.act, ActPolicy::AllowA4 { .. }) && w.row_begin % 128 == 0;
            if f != 0 || !a4 {
                return ActQuant::Bf16;
            }
        }
        match (w.format, w.act) {
            (WeightFormat::Nvfp4, ActPolicy::AllowA4 { divisor })
                if w.k % 128 == 0 && w.row_begin % 128 == 0 =>
            {
                ActQuant::Nvfp4 { divisor }
            }
            (WeightFormat::Fp8RowBf16, ActPolicy::AllowA8 | ActPolicy::AllowA4 { .. })
                if w.k % 64 == 0 =>
            {
                ActQuant::Fp8Row
            }
            _ => ActQuant::Exact,
        }
    }

    fn linear_swiglu_act(
        &self,
        gate: &WeightView<'_, CudaBuffer>,
        up: &WeightView<'_, CudaBuffer>,
        m: usize,
    ) -> Option<ActQuant> {
        // IQ3_XXS gate / up (the dense MLP) at decode and verify widths: the IQ3 kernels with
        // gate and up rows paired in each MMA tile and SwiGLU in the epilogue.
        let pair = gate.format == WeightFormat::Iq3Xxs
            && up.format == WeightFormat::Iq3Xxs
            && gate.k == up.k
            && gate.rows == up.rows;
        if pair && q8_format(gate, m).is_some() && q8_format(up, m).is_some() {
            // Prefill: the int8 kernel, gate and up paired the same way.
            return Some(ActQuant::Int8G256);
        }
        (pair && (1..=SKINNY16_MAX_M).contains(&m)).then_some(ActQuant::Bf16)
    }

    fn linear_swiglu(
        &self,
        x: &CudaBuffer,
        gate: &WeightView<'_, CudaBuffer>,
        up: &WeightView<'_, CudaBuffer>,
        out: &mut CudaBuffer,
        m: usize,
    ) -> Result<()> {
        let launch = match self.linear_swiglu_act(gate, up, m) {
            None => return Err(Error::Unsupported("linear_swiglu shape".into())),
            Some(ActQuant::Int8G256) => velox_5060_linear_swiglu_iq3_q8,
            Some(_) if iq3_direct(gate, m) => velox_5060_linear_swiglu_iq3_direct,
            Some(_) => velox_5060_linear_swiglu_skinny_iq3,
        };
        // SAFETY: both views and the output were validated by CudaBackend (check::linear).
        status(
            unsafe {
                launch(
                    x.as_ptr(),
                    to_i32("m", m)?,
                    gate.buf.as_ptr().cast(),
                    gate.parent_rows as i64,
                    gate.row_begin as i64,
                    up.buf.as_ptr().cast(),
                    up.parent_rows as i64,
                    up.row_begin as i64,
                    gate.k as i64,
                    to_i32("rows", gate.rows)?,
                    out.as_mut_ptr(),
                )
            },
            "rtx5060 linear_swiglu iq3",
        )
    }

    fn linear(
        &self,
        x: &CudaBuffer,
        w: &WeightView<'_, CudaBuffer>,
        y: &mut CudaBuffer,
        m: usize,
    ) -> Result<()> {
        let (mi, rows) = (to_i32("m", m)?, to_i32("rows", w.rows)?);
        let (pr, k, rb) = (w.parent_rows as i64, w.k as i64, w.row_begin as i64);
        let wp = w.buf.as_ptr().cast::<u8>();
        // linear_act screened the shapes. A launcher refusing one anyway is an error, not a
        // fallback: falling back would change the rounding linear_act reported.
        // SAFETY: extents validated by CudaBackend (check::linear).
        match self.linear_act(w, m) {
            ActQuant::Exact => match gemv_format(w, m) {
                Some(f) => status(
                    unsafe {
                        velox_5060_gemv(x.as_ptr(), mi, wp, pr, k, rb, rows, f, y.as_mut_ptr())
                    },
                    "rtx5060 gemv",
                ),
                None => portable::linear(x, w, y, m),
            },
            ActQuant::Nvfp4 { divisor } => status(
                unsafe {
                    velox_5060_linear_a4(
                        x.as_ptr(),
                        mi,
                        wp,
                        pr,
                        k,
                        rb,
                        rows,
                        divisor,
                        y.as_mut_ptr(),
                    )
                },
                "rtx5060 linear a4",
            ),
            ActQuant::Fp8Row => status(
                unsafe {
                    velox_5060_linear_a8(x.as_ptr(), mi, wp, pr, k, rb, rows, y.as_mut_ptr())
                },
                "rtx5060 linear a8",
            ),
            ActQuant::Int8G256 => match q8_format(w, m) {
                Some(3) => status(
                    unsafe {
                        velox_5060_linear_iq3_q8(
                            x.as_ptr(),
                            mi,
                            wp,
                            pr,
                            k,
                            rb,
                            rows,
                            y.as_mut_ptr(),
                        )
                    },
                    "rtx5060 linear iq3 q8",
                ),
                Some(f) => status(
                    unsafe {
                        velox_5060_linear_rs_q8(
                            x.as_ptr(),
                            mi,
                            wp,
                            pr,
                            k,
                            rb,
                            rows,
                            f,
                            y.as_mut_ptr(),
                        )
                    },
                    "rtx5060 linear rs q8",
                ),
                None => Err(Error::Unsupported("int8 linear shape".into())),
            },
            ActQuant::Bf16 => {
                if nvfp4_mma(w, m) {
                    return status(
                        unsafe {
                            velox_5060_linear_nvfp4_mma(
                                x.as_ptr(),
                                mi,
                                wp,
                                pr,
                                k,
                                rb,
                                rows,
                                y.as_mut_ptr(),
                            )
                        },
                        "rtx5060 linear nvfp4 mma",
                    );
                }
                if skinny_iq3(w, m) && m > SKINNY16_MAX_M {
                    // Prefill: the staged kernel tiles the tokens itself (64 per weight pass).
                    return status(
                        unsafe {
                            velox_5060_linear_skinny_iq3(
                                x.as_ptr(),
                                mi,
                                wp,
                                pr,
                                k,
                                rb,
                                rows,
                                y.as_mut_ptr(),
                            )
                        },
                        "rtx5060 linear iq3 prefill",
                    );
                }
                if skinny_iq3(w, m) {
                    let direct = iq3_direct(w, m);
                    // The kernel takes up to 16 rows; larger m runs in blocks.
                    for r0 in (0..m).step_by(SKINNY16_MAX_M) {
                        let mb = to_i32("m", (m - r0).min(SKINNY16_MAX_M))?;
                        // SAFETY: rows r0.. of x and y lie inside the buffers check::linear validated.
                        let (xp, yp) =
                            unsafe { (x.as_ptr().add(r0 * w.k), y.as_mut_ptr().add(r0 * w.rows)) };
                        status(
                            unsafe {
                                if direct {
                                    velox_5060_linear_iq3_direct(xp, mb, wp, pr, k, rb, rows, yp)
                                } else {
                                    velox_5060_linear_skinny_iq3(xp, mb, wp, pr, k, rb, rows, yp)
                                }
                            },
                            "rtx5060 linear iq3",
                        )?;
                    }
                    return Ok(());
                }
                if a16_fp8(w, m) {
                    // The kernel takes up to 64 rows; larger m runs in blocks (activations [m, k],
                    // outputs [m, rows]).
                    for r0 in (0..m).step_by(A16_FP8_BLOCK_M) {
                        let mb = to_i32("m", (m - r0).min(A16_FP8_BLOCK_M))?;
                        // SAFETY: rows r0.. of x and y lie inside the buffers check::linear validated.
                        let (xp, yp) =
                            unsafe { (x.as_ptr().add(r0 * w.k), y.as_mut_ptr().add(r0 * w.rows)) };
                        status(
                            unsafe { velox_5060_linear_a16_fp8(xp, mb, wp, pr, k, rb, rows, yp) },
                            "rtx5060 linear a16 fp8",
                        )?;
                    }
                    return Ok(());
                }
                if let Some(f) = skinny16_format(w, m) {
                    // The kernel takes up to 16 rows; larger m runs in blocks (activations [m, k],
                    // outputs [m, rows]).
                    for r0 in (0..m).step_by(SKINNY16_MAX_M) {
                        let mb = to_i32("m", (m - r0).min(SKINNY16_MAX_M))?;
                        // SAFETY: rows r0.. of x and y lie inside the buffers check::linear validated.
                        let (xp, yp) =
                            unsafe { (x.as_ptr().add(r0 * w.k), y.as_mut_ptr().add(r0 * w.rows)) };
                        if f == 0 && w.k % 512 == 0 {
                            // NVFP4 with K in whole 512-value tiles: the sliced-K kernel. Measured
                            // at m=8: [5120,6144] 58 vs 90 us, [34816,5120] 293 vs 350 us,
                            // [248320,5120] 2019 vs 2208 us (skinny16).
                            status(
                                unsafe {
                                    velox_5060_linear_nvfp4_sliced(xp, mb, wp, pr, k, rb, rows, yp)
                                },
                                "rtx5060 linear nvfp4 sliced",
                            )?;
                        } else if f == 0 {
                            status(
                                unsafe {
                                    velox_5060_linear_skinny16(xp, mb, wp, pr, k, rb, rows, f, yp)
                                },
                                "rtx5060 linear skinny16",
                            )?;
                        } else {
                            // q4 / q5: the sliced-K kernel (16 rows per block, 4 warps over each
                            // 256-value K tile); q8: the staged 128-row kernel. Measured at m=8: q5
                            // [12288,5120] 123 vs 186 us, q4 [8192,5120] 84 vs 90 us.
                            status(
                                unsafe {
                                    if f == 8 {
                                        velox_5060_linear_skinny_rs(
                                            xp, mb, wp, pr, k, rb, rows, f, yp,
                                        )
                                    } else {
                                        velox_5060_linear_rs_sliced(
                                            xp, mb, wp, pr, k, rb, rows, f, yp,
                                        )
                                    }
                                },
                                "rtx5060 linear rs",
                            )?;
                        }
                    }
                    return Ok(());
                }
                let bits = w.format.row_split().map_or(0, |r| r.bits as i32);
                status(
                    unsafe {
                        velox_5060_linear_rowsplit(
                            x.as_ptr(),
                            mi,
                            wp,
                            pr,
                            k,
                            rb,
                            rows,
                            bits,
                            y.as_mut_ptr(),
                        )
                    },
                    "rtx5060 linear rowsplit",
                )
            }
        }
    }
}
