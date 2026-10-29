//! RTX 5060 Ti kernel set (ported from rtx-5090-32g). Each override tries the tuned kernel and falls back to the portable one
//! when the kernel reports the shape as unsupported.

use velox_core::backend::{
    linear_part_whole, ActPolicy, ActQuant, AttnPrecision, AttnShape, ExpertBank, FullAttnShape,
    LinearDst, NormKind, WeightView,
};
use velox_core::check::to_i32;
use velox_core::{Error, KvFormat, Op, Result, WeightFormat};
use velox_cuda::{graph_mode, portable, status, CudaBuffer, CudaKernels, GraphMode};

extern "C" {
    fn velox_4090_matmul_small_m(
        x: *const f32,
        w: *const f32,
        y: *mut f32,
        m: i32,
        n: i32,
        k: i32,
    ) -> i32;
    fn velox_4090_matmul_tiled(
        x: *const f32,
        w: *const f32,
        y: *mut f32,
        m: i32,
        n: i32,
        k: i32,
    ) -> i32;
    fn velox_4090_attention_full(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        out: *mut f32,
        tokens: i32,
        heads: i32,
        head_dim: i32,
    ) -> i32;
    fn velox_4090_attention_prefill(
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
    fn velox_4090_attention_windows(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        out: *mut f32,
        tokens: i32,
        window: i32,
        heads: i32,
        kv_heads: i32,
        head_dim: i32,
        pos0: i32,
        scale: f32,
    ) -> i32;
    fn velox_4090_attention_verify_nvfp4(
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
    fn velox_4090_linear_a16_fp8(
        x: *const f32,
        m: i32,
        w: *const u8,
        parent_rows: i64,
        k: i64,
        row_begin: i64,
        rows: i32,
        y: *mut f32,
    ) -> i32;
    fn velox_4090_linear_skinny16(
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
    fn velox_4090_linear_swiglu_iq3_direct(
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
    fn velox_4090_linear_swiglu_skinny_iq3(
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
    fn velox_4090_linear_nvfp4_sliced(
        x: *const f32,
        m: i32,
        w: *const u8,
        parent_rows: i64,
        k: i64,
        row_begin: i64,
        rows: i32,
        y: *mut f32,
    ) -> i32;
    fn velox_4090_linear_rs_sliced(
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
    fn velox_4090_linear_bf16_skinny(
        x: *const f32,
        m: i32,
        w: *const u8,
        k: i64,
        row_begin: i64,
        rows: i32,
        y: *mut f32,
        ldy: i64,
    ) -> i32;
    fn velox_4090_q8_quantize(x: *const f32, m: i32, k: i64, layout: i32) -> i32;
    fn velox_4090_q8_act_bytes(m: i32, k: i64) -> usize;
    fn velox_4090_add_rms_norm_q8(
        acc: *mut f32,
        x: *const f32,
        weight: *const f32,
        y: *mut f32,
        blob: *mut u8,
        layout: i32,
        rows: i32,
        dim: i32,
        eps: f32,
        offset: i32,
    ) -> i32;
    fn velox_4090_linear_iq3_q8_from(
        blob: *const u8,
        m: i32,
        w: *const u8,
        parent_rows: i64,
        k: i64,
        row_begin: i64,
        rows: i32,
        y: *mut f32,
        ldy: i64,
    ) -> i32;
    fn velox_4090_linear_rs_q8_from(
        blob: *const u8,
        m: i32,
        w: *const u8,
        parent_rows: i64,
        k: i64,
        row_begin: i64,
        rows: i32,
        format: i32,
        y: *mut f32,
        ldy: i64,
    ) -> i32;
    fn velox_4090_linear_swiglu_iq3_q8_from(
        blob: *const u8,
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
    fn velox_4090_bf16_act(x: *const f32, m: i32, k: i64, xb: *mut *const u8) -> i32;
    fn velox_4090_gdn_chunk(
        qkv: *const f32,
        ab: *const f32,
        a_log: *const f32,
        dt_bias: *const f32,
        state_in: *const f32,
        state: *mut f32,
        out: *mut f32,
        tokens: i32,
        k_heads: i32,
        v_heads: i32,
        k_dim: i32,
        v_dim: i32,
    ) -> i32;
    fn velox_4090_ggml_act(
        x: *const f32,
        m: i32,
        k: i64,
        q: *mut *const i8,
        d: *mut *const f32,
    ) -> i32;
    fn velox_4090_linear_ggml_from(
        xq: *const i8,
        xd: *const f32,
        m: i32,
        w: *const u8,
        ty: i32,
        k: i64,
        row_begin: i64,
        rows: i32,
        y: *mut f32,
        ldy: i64,
        mode: i32,
    ) -> i32;
    fn velox_4090_linear_ggml_q8(
        blob: *const u8,
        m: i32,
        w: *const u8,
        ty: i32,
        k: i64,
        row_begin: i64,
        rows: i32,
        y: *mut f32,
        ldy: i64,
        mode: i32,
    ) -> i32;
    fn velox_4090_gdn_step(
        qkv: *const f32,
        ab: *const f32,
        a_log: *const f32,
        dt_bias: *const f32,
        state_in: *const f32,
        state: *mut f32,
        out: *mut f32,
        tokens: i32,
        k_heads: i32,
        v_heads: i32,
        k_dim: i32,
        v_dim: i32,
        pp: *const i64,
    ) -> i32;
    fn velox_4090_linear_iq3_direct_bf16(
        xb: *const u8,
        m: i32,
        w: *const u8,
        parent_rows: i64,
        k: i64,
        row_begin: i64,
        rows: i32,
        y: *mut f32,
    ) -> i32;
    fn velox_4090_linear_swiglu_iq3_direct_twin(
        xb: *const u8,
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
        outb: *mut u16,
    ) -> i32;
    fn velox_4090_linear_nvfp4_mma_bf16(
        xb: *const u8,
        m: i32,
        w: *const u8,
        parent_rows: i64,
        k: i64,
        row_begin: i64,
        rows: i32,
        y: *mut f32,
    ) -> i32;
    fn velox_4090_linear_nvfp4_sliced_bf16(
        xb: *const u8,
        m: i32,
        w: *const u8,
        parent_rows: i64,
        k: i64,
        row_begin: i64,
        rows: i32,
        y: *mut f32,
    ) -> i32;
    fn velox_4090_linear_rs_sliced_bf16(
        xb: *const u8,
        m: i32,
        w: *const u8,
        parent_rows: i64,
        k: i64,
        row_begin: i64,
        rows: i32,
        format: i32,
        y: *mut f32,
        ldy: i64,
    ) -> i32;
    fn velox_4090_linear_iq3_q8_pre(
        m: i32,
        w: *const u8,
        parent_rows: i64,
        k: i64,
        row_begin: i64,
        rows: i32,
        y: *mut f32,
        ldy: i64,
    ) -> i32;
    fn velox_4090_linear_rs_q8_pre(
        m: i32,
        w: *const u8,
        parent_rows: i64,
        k: i64,
        row_begin: i64,
        rows: i32,
        format: i32,
        y: *mut f32,
        ldy: i64,
    ) -> i32;
    fn velox_4090_linear_iq3_direct(
        x: *const f32,
        m: i32,
        w: *const u8,
        parent_rows: i64,
        k: i64,
        row_begin: i64,
        rows: i32,
        y: *mut f32,
    ) -> i32;
    fn velox_4090_gemv(
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
    fn velox_4090_linear_skinny_rs(
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
    fn velox_4090_linear_skinny_iq3(
        x: *const f32,
        m: i32,
        w: *const u8,
        parent_rows: i64,
        k: i64,
        row_begin: i64,
        rows: i32,
        y: *mut f32,
    ) -> i32;
    fn velox_4090_linear_rowsplit(
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
    fn velox_4090_linear_nvfp4_mma(
        x: *const f32,
        m: i32,
        w: *const u8,
        parent_rows: i64,
        k: i64,
        row_begin: i64,
        rows: i32,
        y: *mut f32,
    ) -> i32;
    fn velox_4090_linear_iq3_q8(
        x: *const f32,
        m: i32,
        w: *const u8,
        parent_rows: i64,
        k: i64,
        row_begin: i64,
        rows: i32,
        y: *mut f32,
    ) -> i32;
    fn velox_4090_linear_swiglu_iq3_q8(
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
    fn velox_4090_linear_rs_q8(
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
/// `velox_4090_linear_skinny16`, groupwise q4/q5/q8 (4/5/8) `velox_4090_linear_skinny_rs`. Single
/// tokens: NVFP4 projections on the sliced-K kernel (measured [5120,6144] m=1 54 us vs 60 us
/// GEMV; the 248k-row head stays on the GEMV, 1783 vs 1869 us), q5 on `gemv_format`'s GEMV,
/// q4/q8 on main's (the sliced-K kernels lose at m=1: q4 [8192,5120] 82 vs 67 us, q5
/// [12288,5120] 123 vs 114 us).
fn skinny16_format(w: &WeightView<'_, CudaBuffer>, m: usize) -> Option<i32> {
    let max = if w.format.row_split().is_some() {
        ROWSPLIT_MIN_M - 1
    } else {
        SKINNY_MAX_M
    };
    let min = match w.format {
        WeightFormat::Nvfp4 if w.rows <= 65536 => 1,
        _ => 2,
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
/// straight to registers) or the shared-memory staged one. Up to 8 tokens the direct one is faster
/// in the model (verify m=8, nsys: gate/up pair 200 us staged; down 99 vs 120 us); at 16 the staged
/// one (gate/up 239 vs 227, down 151 vs 137 us, bench-linear).
fn iq3_direct(m: usize) -> bool {
    m.min(SKINNY16_MAX_M) <= 8
}

/// bf16 weights with few rows (the GDN gate projections) past the skinny widths, and any bf16
/// weight up to one 16-token block (the DFlash conv kernel projections, [1280, 5120]): the bf16
/// tensor-core kernel. Qwen3.8 a/b ([96, 5120]) at 1024 tokens: 228 us exact f32 (portable) -> 64
/// us; [1280, 5120] at 8 tokens: 57 -> 36 us.
fn bf16_skinny(w: &WeightView<'_, CudaBuffer>, m: usize) -> bool {
    w.format == WeightFormat::Bf16
        && m >= 2
        && (w.rows <= 256 || m <= SKINNY16_MAX_M)
        && w.k % 32 == 0
}

/// bf16 weights of many rows at prefill widths: bf16 tensor cores (rtx4090_bf16.cu; K a multiple
/// of 16, as Qwen3-VL's ViT fc2 with K = 4304).
fn bf16_mma(w: &WeightView<'_, CudaBuffer>, m: usize) -> bool {
    w.format == WeightFormat::Bf16 && m >= BF16_MMA_MIN_M && w.k % 16 == 0 && !bf16_skinny(w, m)
}

const BF16_MMA_MIN_M: usize = 64;

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
    if let WeightFormat::Ggml(t) = w.format {
        // The prefill kernel copies each block from the 4-byte aligned address at or below it,
        // which stays inside the row when rows are 4-byte multiples.
        let aligned = w.k % 1024 == 0 || t.row_bytes(w.k) % 4 == 0;
        return (ggml_served(t) && m >= GGML_Q8_MIN_M && aligned).then(|| 16 + ggml_index(t));
    }
    match w.format {
        WeightFormat::Iq3Xxs if m >= IQ3_Q8_MIN_M => Some(3),
        WeightFormat::Q4G64 if m >= RS_Q8_MIN_M => Some(4),
        WeightFormat::Q5G64 if m >= RS_Q8_MIN_M => Some(5),
        _ => None,
    }
}

/// Gate / up for `ggml_swiglu_q8`: both on the int8 prefill kernels, GGML (formats from 16) and
/// at most one IQ3_XXS (3).
fn ggml_q8_pair(
    gate: &WeightView<'_, CudaBuffer>,
    up: &WeightView<'_, CudaBuffer>,
    m: usize,
) -> bool {
    match (q8_format(gate, m), q8_format(up, m)) {
        (Some(g), Some(u)) => (g >= 16 || u >= 16) && (g >= 16 || g == 3) && (u >= 16 || u == 3),
        _ => false,
    }
}

/// The int8 kernels' activation layout for a `q8_format` code (0: IQ3_XXS, 1: q4/q5).
fn q8_layout(format: i32) -> i32 {
    if format == 3 || format >= 16 {
        0
    } else {
        1
    }
}

/// Format code (4/5) of a q4/q5 part `linears` runs on the sliced-K kernel from a shared bf16
/// conversion: the widths `linear` serves with it, up to one 16-token block, K without padding.
fn rs_shared(w: &WeightView<'_, CudaBuffer>, m: usize) -> Option<i32> {
    match skinny16_format(w, m) {
        Some(f @ (4 | 5)) if m <= SKINNY16_MAX_M && w.k % 128 == 0 => Some(f),
        _ => None,
    }
}

/// The kernel `linear_bf16` runs for a view at `m` tokens, reading a bf16 twin (natural order)
/// instead of converting: the direct IQ3_XXS kernel, the q4/q5 sliced-K one and the NVFP4 sliced-K
/// one at the widths `linear` serves with them, and the NVFP4 mma GEMM at prefill widths.
#[derive(Clone, Copy)]
enum Bf16In {
    Iq3,
    Rs(i32),
    Nvfp4,
    Nvfp4Mma,
}

fn bf16_in(w: &WeightView<'_, CudaBuffer>, m: usize) -> Option<Bf16In> {
    if nvfp4_mma(w, m) {
        return Some(Bf16In::Nvfp4Mma);
    }
    if !(1..=SKINNY16_MAX_M).contains(&m) {
        return None;
    }
    if skinny_iq3(w, m) && iq3_direct(m) {
        return Some(Bf16In::Iq3);
    }
    if let Some(f) = rs_shared(w, m) {
        return Some(Bf16In::Rs(f));
    }
    (skinny16_format(w, m) == Some(0) && w.k % 512 == 0).then_some(Bf16In::Nvfp4)
}

/// Whether `linears` writes this part in place at `m` tokens: int8 parts (which share one
/// quantization per layout), q4/q5 sliced-K ones (which share one bf16 conversion) and bf16
/// few-row ones.
fn linears_in_place(w: &WeightView<'_, CudaBuffer>, m: usize) -> bool {
    q8_format(w, m).is_some()
        || rs_shared(w, m).is_some()
        || bf16_skinny(w, m)
        || ggml_mmvq(w, m).is_some()
}

/// NVFP4 views from this many tokens: the tiled bf16 tensor-core kernel (ninfer's prefill kernel
/// and schedule; Ada has no FP4 tensor cores, so AllowA4 views take it too). Below, the sliced-K
/// kernel in 16-token blocks.
const NVFP4_MMA_MIN_M: usize = 33;

fn nvfp4_mma(w: &WeightView<'_, CudaBuffer>, m: usize) -> bool {
    w.format == WeightFormat::Nvfp4 && w.k % 64 == 0 && m >= NVFP4_MMA_MIN_M
}

/// Decode windows up to this many tokens run the GDN step kernel (ninfer's recurrent layout:
/// vector loads of the state, 4 rows per warp); `VELOX_GDN_STEP_MAX_T` overrides it for tuning.
fn gdn_step_max_t() -> usize {
    static MAX: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MAX.get_or_init(|| {
        std::env::var("VELOX_GDN_STEP_MAX_T")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16)
    })
}

/// Whether the step kernel serves `s` with `qkv` (its window row 0 must be 16-byte aligned): the
/// kernel's own shape contract, checked here so a graph step stages its pointers once.
fn gdn_step(qkv: &CudaBuffer, s: &velox_core::backend::GdnShape) -> bool {
    (1..=gdn_step_max_t()).contains(&s.tokens)
        && matches!(s.k_dim, 32 | 64 | 128)
        && s.v_dim % 16 == 0
        && s.k_heads > 0
        && s.v_heads % s.k_heads == 0
        && s.qkv_width() % 4 == 0
        && (qkv.as_ptr() as usize + s.row0 * s.qkv_width() * 4) % 16 == 0
}

/// From this many tokens GGML weights run the int8 prefill kernel (Int8G256 activations, the q8
/// path's format 16 + type); below, the per-32 kernels of `ggml_mmvq`.
const GGML_Q8_MIN_M: usize = 17;

/// From this many tokens (rtx4090_ggml.cu mma_min_m, VELOX_GGML_MMA_MIN unset) the decode-width
/// GGML linears run on int8 tensor cores instead of dp4a.
const GGML_MMA_MIN_M: usize = 5;

/// Types rtx4090_ggml.cu decodes at every width: the first nine of `GgmlType::ALL` (q2_k ..
/// iq4_xs) and IQ1_S (dp4a up to 16 tokens, the int8 prefill kernel above). IQ4_NL runs its dp4a kernel up to 16 tokens (`ggml_mmvq`) and the MoE kernels; Q2_0
/// only the MoE kernels; the rest (q6_k, q8_0) the portable kernels.
fn ggml_served(t: velox_core::ggml::GgmlType) -> bool {
    ggml_index(t) < 9 || t == velox_core::ggml::GgmlType::Iq1S
}

fn ggml_index(t: velox_core::ggml::GgmlType) -> i32 {
    velox_core::ggml::GgmlType::ALL
        .iter()
        .position(|&x| x == t)
        .map_or(0, |i| i as i32)
}

/// The GGML type code (`GgmlType::ALL` order) of a weight rtx4090_ggml.cu serves at `m` tokens
/// (int8 activations per 32 values: dp4a up to 4 tokens, int8 tensor cores from 5, prefill
/// included): AllowA8 (the importer's policy for GGML matrices), K a multiple of 1024 so two
/// blocks of a row start 4-byte aligned for every block size. The dp4a kernel (up to 4 tokens)
/// also takes K a multiple of 256 whose rows are 16-byte multiples (Qwen3.8-Flash-Next's 2560).
/// IQ4_NL (32-value blocks) runs the dp4a kernel at every width up to 16 tokens, K a multiple of
/// 256: model widths, not the narrow layers (K 32 / 64) of small test stacks, whose exact
/// reference the int8 rounding would push past the end-to-end tolerance.
fn ggml_mmvq(w: &WeightView<'_, CudaBuffer>, m: usize) -> Option<i32> {
    let WeightFormat::Ggml(t) = w.format else {
        return None;
    };
    let a8 = matches!(w.act, ActPolicy::AllowA8 | ActPolicy::AllowA4 { .. });
    if t == velox_core::ggml::GgmlType::Iq4Nl {
        return (a8 && (1..GGML_Q8_MIN_M).contains(&m) && w.k % 256 == 0).then(|| ggml_index(t));
    }
    let dp4a_rows = m < GGML_MMA_MIN_M && w.k % 256 == 0 && t.row_bytes(w.k) % 16 == 0;
    if !ggml_served(t) || !(1..GGML_Q8_MIN_M).contains(&m) || (w.k % 1024 != 0 && !dp4a_rows) || !a8
    {
        return None;
    }
    velox_core::ggml::GgmlType::ALL
        .iter()
        .position(|&x| x == t)
        .map(|i| i as i32)
}

/// One `Int8G32` quantization of a GGML linear's input (velox_4090_ggml_act), shared by every GGML
/// part reading the same `x` at the same K.
struct GgmlAct {
    q: *const i8,
    d: *const f32,
    k: usize,
}

impl GgmlAct {
    fn new(x: &CudaBuffer, m: usize, k: usize) -> Result<Self> {
        let (mut q, mut d) = (std::ptr::null(), std::ptr::null());
        // SAFETY: x holds m * k values (the caller's linear checks); the shim writes its scratch.
        status(
            unsafe { velox_4090_ggml_act(x.as_ptr(), to_i32("m", m)?, k as i64, &mut q, &mut d) },
            "rtx4090 ggml act",
        )?;
        Ok(GgmlAct { q, d, k })
    }

    /// `y[t * ldy + r]` for the view's rows (mode 1: `silu(y) * ..`, a SwiGLU's up projection).
    fn run(
        &self,
        w: &WeightView<'_, CudaBuffer>,
        ty: i32,
        m: usize,
        y: *mut f32,
        ldy: usize,
        mode: i32,
    ) -> Result<()> {
        // SAFETY: the caller checked y's extent for m rows of stride ldy; the blob matches m, k.
        status(
            unsafe {
                velox_4090_linear_ggml_from(
                    self.q,
                    self.d,
                    to_i32("m", m)?,
                    w.buf.as_ptr().cast::<u8>(),
                    ty,
                    w.k as i64,
                    w.row_begin as i64,
                    to_i32("rows", w.rows)?,
                    y,
                    ldy as i64,
                    mode,
                )
            },
            "rtx4090 linear ggml",
        )
    }
}

pub struct Rtx4090Kernels;

impl Rtx4090Kernels {
    /// SwiGLU over separate gate / up parents at prefill widths, GGML and at most one of them
    /// IQ3_XXS (GGUF imports keep GGML blocks but turn IQ3_XXS planar): one GEMM into `out`, the
    /// GGML one second with SwiGLU in its epilogue (gate first: silu(out) * up; up first: silu(gate)
    /// * out), both from one int8 quantization of `x` (or from `x` itself when `twin` is set: an
    /// int8 twin in layout 0).
    fn ggml_swiglu_q8(
        &self,
        x: &CudaBuffer,
        twin: Option<()>,
        gate: &WeightView<'_, CudaBuffer>,
        up: &WeightView<'_, CudaBuffer>,
        out: &mut CudaBuffer,
        m: usize,
    ) -> Result<()> {
        let (Some(fg), Some(fu)) = (q8_format(gate, m), q8_format(up, m)) else {
            return Err(Error::Unsupported("ggml swiglu q8 views".into()));
        };
        let (mi, rows, k) = (to_i32("m", m)?, to_i32("rows", gate.rows)?, gate.k as i64);
        let blob = match twin {
            Some(()) => x.as_ptr().cast::<u8>(),
            None => {
                // SAFETY: x holds m * k values (CudaBackend::linear_swiglu checked the views).
                status(
                    unsafe { velox_4090_q8_quantize(x.as_ptr(), mi, k, 0) },
                    "rtx4090 q8 quantize",
                )?;
                std::ptr::null()
            }
        };
        let order = if fu == 3 {
            [(up, fu, 0), (gate, fg, 2)]
        } else {
            [(gate, fg, 0), (up, fu, 1)]
        };
        for (w, f, mode) in order {
            let (wp, rb) = (w.buf.as_ptr().cast::<u8>(), w.row_begin as i64);
            // SAFETY: out holds m * rows values (checked by the caller); the activations match m, k.
            let code = unsafe {
                match (f, twin) {
                    (3, Some(())) => velox_4090_linear_iq3_q8_from(
                        blob,
                        mi,
                        wp,
                        w.parent_rows as i64,
                        k,
                        rb,
                        rows,
                        out.as_mut_ptr(),
                        rows as i64,
                    ),
                    (3, None) => velox_4090_linear_iq3_q8_pre(
                        mi,
                        wp,
                        w.parent_rows as i64,
                        k,
                        rb,
                        rows,
                        out.as_mut_ptr(),
                        rows as i64,
                    ),
                    _ => velox_4090_linear_ggml_q8(
                        blob,
                        mi,
                        wp,
                        f - 16,
                        k,
                        rb,
                        rows,
                        out.as_mut_ptr(),
                        rows as i64,
                        mode,
                    ),
                }
            };
            status(code, "rtx4090 linear_swiglu ggml q8")?;
        }
        Ok(())
    }
}

extern "C" {
    fn velox_4090_moe_scratch(rows: i32, k: i32, width: i32, hidden: i32) -> i64;
    fn velox_4090_linear_bf16_mma(
        x: *const f32,
        m: i32,
        w: *const u16,
        k: i64,
        row_begin: i64,
        rows: i32,
        y: *mut f32,
        ldy: i64,
    ) -> i32;
    fn velox_4090_moe_grouped_scratch(
        rows: i32,
        k: i32,
        width: i32,
        hidden: i32,
        experts: i32,
    ) -> i64;
    #[allow(clippy::too_many_arguments)]
    fn velox_4090_moe_grouped(
        x: *const f32,
        ids: *const f32,
        wts: *const f32,
        records: *const u8,
        slots: *const u8,
        map: *const f32,
        record_bytes: i64,
        slot_bytes: i64,
        up_type: i32,
        gate_off: i64,
        up_off: i64,
        down_type: i32,
        down_off: i64,
        width: i32,
        hidden: i32,
        rows: i32,
        k: i32,
        experts: i32,
        y: *mut f32,
        scratch: *mut f32,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn velox_4090_moe_experts(
        x: *const f32,
        ids: *const f32,
        wts: *const f32,
        records: *const u8,
        slots: *const u8,
        map: *const f32,
        record_bytes: i64,
        slot_bytes: i64,
        up_type: i32,
        gate_off: i64,
        up_off: i64,
        down_type: i32,
        down_off: i64,
        width: i32,
        hidden: i32,
        rows: i32,
        k: i32,
        y: *mut f32,
        scratch: *mut f32,
    ) -> i32;
}

/// Most rows the decode MoE kernels take (decode and verify windows); larger chunks run the
/// kernels grouped by expert.
const MOE_DECODE_MAX: usize = 16;

extern "C" {
    fn velox_4090_moe_nvfp4_scratch(
        rows: i32,
        k: i32,
        width: i32,
        hidden: i32,
        experts: i32,
        mode: i32,
    ) -> i64;
    #[allow(clippy::too_many_arguments)]
    fn velox_4090_moe_nvfp4(
        x: *const f32,
        ids: *const f32,
        wts: *const f32,
        records: *const u8,
        slots: *const u8,
        map: *const f32,
        record_bytes: i64,
        slot_bytes: i64,
        gate_off: i64,
        up_off: i64,
        down_off: i64,
        width: i32,
        hidden: i32,
        rows: i32,
        k: i32,
        experts: i32,
        mode: i32,
        y: *mut f32,
        scratch: *mut f32,
    ) -> i32;
}

/// Most rows rtx4090_moe_nvfp4.cu's GEMV kernels (exact activations) take; larger chunks run
/// grouped by expert on tensor cores (bf16 activations).
const MOE_NVFP4_DECODE_MAX: usize = 16;

/// Most rows rtx4090_moe_nvfp4.cu's exact per-expert GEMVs take; wider verify windows stream on
/// tensor cores.
const MOE_NVFP4_SMALL_MAX: usize = 4;

/// rtx4090_moe_nvfp4.cu's mode for `rows`: 0 a GEMV per routed pair (decode), 1 the GEMVs per
/// routed expert over its pairs (small verify windows), 3 persistent tensor-core streams over the
/// routed experts (verify windows), 2 grouped on tensor cores (prefill). Modes 0 and 1 keep
/// activations exact, 2 and 3 round them to bf16.
fn moe_nvfp4_mode(rows: usize) -> i32 {
    match rows {
        0 | 1 => 0,
        r if r <= MOE_NVFP4_SMALL_MAX => 1,
        r if r <= MOE_NVFP4_DECODE_MAX => 3,
        _ => 2,
    }
}

/// Whether rtx4090_moe_nvfp4.cu covers `bank`: NVFP4 gate, up and down (any activation policy:
/// the kernels keep activations exact or round them to bf16), within the launcher's limits so
/// moe_experts_act never reports a rounding the portable fallback would not apply.
fn moe_nvfp4(bank: &ExpertBank<'_, CudaBuffer>, rows: usize) -> bool {
    let nv = |m: &velox_core::ExpertMatrix| m.format == WeightFormat::Nvfp4 && m.offset % 4 == 0;
    rows > 0
        && nv(&bank.gate)
        && nv(&bank.up)
        && nv(&bank.down)
        && bank.record_bytes % 16 == 0
        && bank.slot_bytes % 16 == 0
        && bank.experts <= 6144
}

/// (gate/up type, down type) of rtx4090_ggml.cu's MoE kernels for `bank` at `rows` rows: GGML
/// experts of the served types, Q2_0 or IQ4_NL, 8-bit activations allowed.
fn moe_types(bank: &ExpertBank<'_, CudaBuffer>, rows: usize) -> Option<(i32, i32)> {
    let ty = |f: WeightFormat| match f {
        WeightFormat::Ggml(t) if ggml_index(t) <= 10 => Some(ggml_index(t)),
        _ => None,
    };
    if rows == 0
        || !matches!(bank.act, ActPolicy::AllowA8 | ActPolicy::AllowA4 { .. })
        || bank.gate.format != bank.up.format
    {
        return None;
    }
    // The kernels stage rows with 4-byte loads.
    let aligned =
        |m: &velox_core::ExpertMatrix| m.format.payload_bytes(1, m.k) % 4 == 0 && m.offset % 4 == 0;
    if !(aligned(&bank.gate) && aligned(&bank.up) && aligned(&bank.down)) {
        return None;
    }
    Some((ty(bank.gate.format)?, ty(bank.down.format)?))
}

fn or_fallback(r: Result<()>, fallback: impl FnOnce() -> Result<()>) -> Result<()> {
    match r {
        Err(Error::Unsupported(_)) => fallback(),
        other => other,
    }
}

impl CudaKernels for Rtx4090Kernels {
    fn native_ops(&self) -> &'static [Op] {
        &[
            Op::MatMul,
            Op::Linear,
            Op::Attention,
            Op::AttentionFull,
            Op::LinearSwiglu,
            Op::MoeExperts,
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
            velox_4090_matmul_small_m
        } else {
            velox_4090_matmul_tiled
        };
        // SAFETY: extents validated by CudaBackend; kernels guard their own edges.
        let r = status(
            unsafe { launch(x.as_ptr(), w.as_ptr(), y.as_mut_ptr(), mi, ni, ki) },
            "rtx4090 matmul",
        );
        or_fallback(r, || portable::matmul(x, w, y, m, n, k))
    }

    fn moe_scratch_len(&self, bank: &ExpertBank<'_, CudaBuffer>, rows: usize, k: usize) -> usize {
        let portable = portable::moe_scratch_len(bank, rows, k);
        if moe_nvfp4(bank, rows) {
            // SAFETY: a pure size computation.
            let bytes = unsafe {
                velox_4090_moe_nvfp4_scratch(
                    rows as i32,
                    k as i32,
                    bank.width() as i32,
                    bank.hidden() as i32,
                    bank.experts as i32,
                    moe_nvfp4_mode(rows),
                )
            };
            return portable.max((bytes as usize).div_ceil(4));
        }
        if moe_types(bank, rows).is_none() {
            return portable;
        }
        let (r, kk, w, h) = (
            rows as i32,
            k as i32,
            bank.width() as i32,
            bank.hidden() as i32,
        );
        // SAFETY: pure size computations.
        let bytes = unsafe {
            if rows <= MOE_DECODE_MAX {
                velox_4090_moe_scratch(r, kk, w, h)
            } else {
                velox_4090_moe_grouped_scratch(r, kk, w, h, bank.experts as i32)
            }
        };
        portable.max((bytes as usize).div_ceil(4))
    }

    fn moe_experts_act(&self, bank: &ExpertBank<'_, CudaBuffer>, rows: usize) -> ActQuant {
        if moe_types(bank, rows).is_some() {
            ActQuant::Int8G32
        } else if moe_nvfp4(bank, rows) && moe_nvfp4_mode(rows) >= 2 {
            ActQuant::Bf16
        } else {
            ActQuant::Exact
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn moe_experts(
        &self,
        x: &CudaBuffer,
        ids: &CudaBuffer,
        weights: &CudaBuffer,
        bank: &ExpertBank<'_, CudaBuffer>,
        y: &mut CudaBuffer,
        scratch: &mut CudaBuffer,
        rows: usize,
        k: usize,
    ) -> Result<()> {
        if moe_nvfp4(bank, rows) {
            // SAFETY: CudaBackend validated the bank and extents (check::moe_experts); the scratch
            // holds moe_scratch_len slots for this row count; expert ids past the bank are skipped.
            let r = status(
                unsafe {
                    velox_4090_moe_nvfp4(
                        x.as_ptr(),
                        ids.as_ptr(),
                        weights.as_ptr(),
                        bank.records.as_ptr().cast(),
                        bank.slots.as_ptr().cast(),
                        bank.map.as_ptr(),
                        bank.record_bytes as i64,
                        bank.slot_bytes as i64,
                        bank.gate.offset as i64,
                        bank.up.offset as i64,
                        bank.down.offset as i64,
                        to_i32("width", bank.width())?,
                        to_i32("hidden", bank.hidden())?,
                        to_i32("rows", rows)?,
                        to_i32("k", k)?,
                        to_i32("experts", bank.experts)?,
                        moe_nvfp4_mode(rows),
                        y.as_mut_ptr(),
                        scratch.as_mut_ptr(),
                    )
                },
                "rtx4090 moe_experts (nvfp4)",
            );
            return or_fallback(r, || {
                portable::moe_experts(x, ids, weights, bank, y, scratch, rows, k)
            });
        }
        let Some((tu, td)) = moe_types(bank, rows) else {
            return portable::moe_experts(x, ids, weights, bank, y, scratch, rows, k);
        };
        // SAFETY: CudaBackend validated the bank and extents (check::moe_experts); the scratch
        // holds moe_scratch_len slots; expert ids come from moe_route.
        if rows > MOE_DECODE_MAX {
            // SAFETY: as below; the scratch holds moe_scratch_len slots for this row count.
            let r = status(
                unsafe {
                    velox_4090_moe_grouped(
                        x.as_ptr(),
                        ids.as_ptr(),
                        weights.as_ptr(),
                        bank.records.as_ptr().cast(),
                        bank.slots.as_ptr().cast(),
                        bank.map.as_ptr(),
                        bank.record_bytes as i64,
                        bank.slot_bytes as i64,
                        tu,
                        bank.gate.offset as i64,
                        bank.up.offset as i64,
                        td,
                        bank.down.offset as i64,
                        to_i32("width", bank.width())?,
                        to_i32("hidden", bank.hidden())?,
                        to_i32("rows", rows)?,
                        k as i32,
                        to_i32("experts", bank.experts)?,
                        y.as_mut_ptr(),
                        scratch.as_mut_ptr(),
                    )
                },
                "rtx4090 moe_experts (grouped)",
            );
            return or_fallback(r, || {
                portable::moe_experts(x, ids, weights, bank, y, scratch, rows, k)
            });
        }
        let r = status(
            unsafe {
                velox_4090_moe_experts(
                    x.as_ptr(),
                    ids.as_ptr(),
                    weights.as_ptr(),
                    bank.records.as_ptr().cast(),
                    bank.slots.as_ptr().cast(),
                    bank.map.as_ptr(),
                    bank.record_bytes as i64,
                    bank.slot_bytes as i64,
                    tu,
                    bank.gate.offset as i64,
                    bank.up.offset as i64,
                    td,
                    bank.down.offset as i64,
                    to_i32("width", bank.width())?,
                    to_i32("hidden", bank.hidden())?,
                    rows as i32,
                    k as i32,
                    y.as_mut_ptr(),
                    scratch.as_mut_ptr(),
                )
            },
            "rtx4090 moe_experts",
        );
        or_fallback(r, || {
            portable::moe_experts(x, ids, weights, bank, y, scratch, rows, k)
        })
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
                    velox_4090_attention_verify_nvfp4(
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
                "rtx4090 attention verify (nvfp4)",
            );
        }
        if self.attention_precision(s) == AttnPrecision::Exact {
            return portable::attention(q, k, v, out, s);
        }
        // SAFETY: extents validated by CudaBackend (check::attention).
        status(
            unsafe {
                velox_4090_attention_prefill(
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
            "rtx4090 attention prefill",
        )
    }

    /// Scoring windows take the flash schedule where `attention` does, over f16 operands: E4M3
    /// scores move the candidate distributions a window is read for.
    fn attention_windows_precision(&self, s: AttnShape) -> AttnPrecision {
        if s.kv == KvFormat::F32
            && s.tokens >= FLASH_MIN_TOKENS
            && s.head_dim == 256
            && s.heads % s.kv_heads == 0
            && s.kv_len() <= i32::MAX as usize
        {
            AttnPrecision::F16
        } else {
            AttnPrecision::Exact
        }
    }

    fn attention_windows(
        &self,
        q: &CudaBuffer,
        k: &CudaBuffer,
        v: &CudaBuffer,
        out: &mut CudaBuffer,
        s: AttnShape,
        window: usize,
    ) -> Result<()> {
        if self.attention_windows_precision(s) == AttnPrecision::Exact {
            return portable::attention_windows(q, k, v, out, s, window);
        }
        velox_core::check::attention_windows(q.len(), k.len(), v.len(), out.len(), &s, window)?;
        // SAFETY: extents validated above.
        status(
            unsafe {
                velox_4090_attention_windows(
                    q.as_ptr(),
                    k.as_ptr(),
                    v.as_ptr(),
                    out.as_mut_ptr(),
                    to_i32("tokens", s.tokens)?,
                    to_i32("window", window)?,
                    to_i32("heads", s.heads)?,
                    to_i32("kv_heads", s.kv_heads)?,
                    to_i32("head_dim", s.head_dim)?,
                    to_i32("pos0", s.pos0)?,
                    s.scale(),
                )
            },
            "rtx4090 attention windows",
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
                velox_4090_attention_full(
                    q.as_ptr(),
                    k.as_ptr(),
                    v.as_ptr(),
                    out.as_mut_ptr(),
                    to_i32("tokens", s.tokens)?,
                    to_i32("heads", s.heads)?,
                    to_i32("head_dim", s.head_dim)?,
                )
            },
            "rtx4090 attention_full",
        )
    }

    fn linear_act(&self, w: &WeightView<'_, CudaBuffer>, m: usize) -> ActQuant {
        if ggml_mmvq(w, m).is_some() {
            return ActQuant::Int8G32;
        }
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
        if bf16_skinny(w, m) || bf16_mma(w, m) {
            return ActQuant::Bf16;
        }
        if m <= GEMV_MAX_M {
            return ActQuant::Exact;
        }
        if a16_fp8(w, m) {
            return ActQuant::Bf16;
        }
        if skinny16_format(w, m).is_some() {
            return ActQuant::Bf16;
        }
        // No E4M3 activations: Ada's E4M3 MMA truncates inside each k32 step (measured 6e-5
        // relative on one step, 2e-3 to 7e-2 over model K), past what ActQuant::Fp8Row promises.
        // FP8 weights outside a16_fp8's widths take the exact portable path.
        ActQuant::Exact
    }

    fn serves_linears(&self, dsts: &[LinearDst<'_, '_, '_, CudaBuffer>], m: usize) -> bool {
        // Every part in place or a whole destination (`linear`), and something to gain: an int8
        // quantization shared, or a column range written without a copy.
        let parts = || {
            dsts.iter()
                .flat_map(|d| d.parts.iter().map(move |p| (p, d.ld)))
        };
        let shared = parts()
            .filter(|(p, _)| q8_format(&p.w, m).is_some())
            .count()
            > 1
            || parts()
                .filter(|(p, _)| rs_shared(&p.w, m).is_some())
                .count()
                > 1
            || parts()
                .filter(|(p, _)| ggml_mmvq(&p.w, m).is_some())
                .count()
                > 1;
        let ranges = parts().any(|(p, ld)| !linear_part_whole(p, ld));
        (shared || ranges)
            && parts().all(|(p, ld)| linears_in_place(&p.w, m) || linear_part_whole(p, ld))
    }

    fn linears(
        &self,
        x: &CudaBuffer,
        twin: Option<&CudaBuffer>,
        xq: Option<(&CudaBuffer, u32)>,
        dsts: &mut [LinearDst<'_, '_, '_, CudaBuffer>],
        m: usize,
    ) -> Result<()> {
        let mi = to_i32("m", m)?;
        // q4/q5 sliced-K parts: the input's bf16 twin, or one conversion, then each GEMM into its
        // columns.
        let mut xb: *const u8 = twin.map_or(std::ptr::null(), |t| t.as_ptr().cast::<u8>());
        for d in dsts.iter_mut() {
            let ld = d.ld;
            for p in d.parts {
                let Some(f) = rs_shared(&p.w, m) else {
                    continue;
                };
                let w = &p.w;
                if xb.is_null() {
                    // SAFETY: x holds m * k values (CudaBackend::linears checked every part).
                    status(
                        unsafe { velox_4090_bf16_act(x.as_ptr(), mi, w.k as i64, &mut xb) },
                        "rtx4090 bf16 activations",
                    )?;
                }
                // SAFETY: the destination holds m * ld values and col + rows <= ld (checked).
                let y = unsafe { d.y.as_mut_ptr().add(p.col) };
                status(
                    unsafe {
                        velox_4090_linear_rs_sliced_bf16(
                            xb,
                            mi,
                            w.buf.as_ptr().cast::<u8>(),
                            w.parent_rows as i64,
                            w.k as i64,
                            w.row_begin as i64,
                            to_i32("rows", w.rows)?,
                            f,
                            y,
                            ld as i64,
                        )
                    },
                    "rtx4090 linears rs sliced",
                )?;
            }
        }
        // int8 parts: the input's int8 twin when it is in the layout, else one quantization per
        // layout, then each GEMM into its columns.
        for layout in [0, 1] {
            let blob = xq
                .filter(|&(_, l)| l == layout as u32)
                .map(|(b, _)| b.as_ptr().cast::<u8>());
            let mut quantized = false;
            for d in dsts.iter_mut() {
                let ld = d.ld;
                for p in d.parts {
                    let Some(f) = q8_format(&p.w, m).filter(|&f| q8_layout(f) == layout) else {
                        continue;
                    };
                    let w = &p.w;
                    if blob.is_none() && !quantized {
                        // SAFETY: x holds m * k values (CudaBackend::linears checked every part).
                        status(
                            unsafe { velox_4090_q8_quantize(x.as_ptr(), mi, w.k as i64, layout) },
                            "rtx4090 q8 quantize",
                        )?;
                        quantized = true;
                    }
                    let rows = to_i32("rows", w.rows)?;
                    let (wp, pr, k, rb) = (
                        w.buf.as_ptr().cast::<u8>(),
                        w.parent_rows as i64,
                        w.k as i64,
                        w.row_begin as i64,
                    );
                    // SAFETY: the destination holds m * ld values and col + rows <= ld (checked);
                    // the twin holds the blob of [m, k] (checked).
                    let y = unsafe { d.y.as_mut_ptr().add(p.col) };
                    status(
                        unsafe {
                            match (blob, f) {
                                (b, f) if f >= 16 => velox_4090_linear_ggml_q8(
                                    b.unwrap_or(std::ptr::null()),
                                    mi,
                                    wp,
                                    f - 16,
                                    k,
                                    rb,
                                    rows,
                                    y,
                                    ld as i64,
                                    0,
                                ),
                                (Some(q), 3) => velox_4090_linear_iq3_q8_from(
                                    q, mi, wp, pr, k, rb, rows, y, ld as i64,
                                ),
                                (Some(q), _) => velox_4090_linear_rs_q8_from(
                                    q, mi, wp, pr, k, rb, rows, f, y, ld as i64,
                                ),
                                (None, 3) => velox_4090_linear_iq3_q8_pre(
                                    mi, wp, pr, k, rb, rows, y, ld as i64,
                                ),
                                (None, _) => velox_4090_linear_rs_q8_pre(
                                    mi, wp, pr, k, rb, rows, f, y, ld as i64,
                                ),
                            }
                        },
                        "rtx4090 linears q8",
                    )?;
                }
            }
        }
        // GGML parts: one Int8G32 quantization per K, each GEMM into its columns.
        let mut ggml: Option<GgmlAct> = None;
        for d in dsts.iter_mut() {
            let ld = d.ld;
            for p in d.parts {
                let Some(ty) = ggml_mmvq(&p.w, m) else {
                    continue;
                };
                if ggml.as_ref().is_none_or(|a| a.k != p.w.k) {
                    ggml = Some(GgmlAct::new(x, m, p.w.k)?);
                }
                // SAFETY: the destination holds m * ld values and col + rows <= ld (checked).
                let y = unsafe { d.y.as_mut_ptr().add(p.col) };
                ggml.as_ref()
                    .expect("set above")
                    .run(&p.w, ty, m, y, ld, 0)?;
            }
        }
        for d in dsts.iter_mut() {
            let ld = d.ld;
            for p in d.parts {
                let w = &p.w;
                if q8_format(w, m).is_some()
                    || rs_shared(w, m).is_some()
                    || ggml_mmvq(w, m).is_some()
                {
                    continue;
                }
                if bf16_skinny(w, m) {
                    // SAFETY: as above.
                    let y = unsafe { d.y.as_mut_ptr().add(p.col) };
                    status(
                        unsafe {
                            velox_4090_linear_bf16_skinny(
                                x.as_ptr(),
                                mi,
                                w.buf.as_ptr().cast::<u8>(),
                                w.k as i64,
                                w.row_begin as i64,
                                to_i32("rows", w.rows)?,
                                y,
                                ld as i64,
                            )
                        },
                        "rtx4090 linears bf16 skinny",
                    )?;
                } else {
                    // serves_linears admitted it as a whole destination.
                    self.linear(x, w, d.y, m)?;
                }
            }
        }
        Ok(())
    }

    /// Int8 twins: the layout the int8 prefill GEMM for `w` reads (q8_layout), for hidden widths
    /// the fused norm handles (a multiple of 256, at most 12288).
    fn int8_twin_layout(&self, w: &WeightView<'_, CudaBuffer>, m: usize) -> Option<u32> {
        (w.k % 256 == 0 && w.k <= 12288)
            .then(|| q8_format(w, m).map(|f| q8_layout(f) as u32))
            .flatten()
    }

    fn int8_twin_slots(&self, m: usize, k: usize, _layout: u32) -> usize {
        let (Ok(mi), Ok(ki)) = (i32::try_from(m), i64::try_from(k)) else {
            return 0;
        };
        // SAFETY: plain value arguments.
        unsafe { velox_4090_q8_act_bytes(mi, ki) }.div_ceil(4)
    }

    #[allow(clippy::too_many_arguments)]
    fn add_rms_norm_q8(
        &self,
        acc: &mut CudaBuffer,
        x: &CudaBuffer,
        weight: &CudaBuffer,
        y: &mut CudaBuffer,
        yq: &mut CudaBuffer,
        layout: u32,
        rows: usize,
        dim: usize,
        eps: f32,
        kind: NormKind,
    ) -> Result<()> {
        // SAFETY: CudaBackend::add_rms_norm_q8 checked the extents and the twin's slots.
        status(
            unsafe {
                velox_4090_add_rms_norm_q8(
                    acc.as_mut_ptr(),
                    x.as_ptr(),
                    weight.as_ptr(),
                    y.as_mut_ptr(),
                    yq.as_mut_ptr().cast::<u8>(),
                    layout as i32,
                    to_i32("rows", rows)?,
                    to_i32("dim", dim)?,
                    eps,
                    i32::from(kind == NormKind::Offset),
                )
            },
            "rtx4090 add_rms_norm q8",
        )
    }

    fn linear_q8(
        &self,
        xq: &CudaBuffer,
        layout: u32,
        w: &WeightView<'_, CudaBuffer>,
        y: &mut CudaBuffer,
        m: usize,
    ) -> Result<()> {
        let Some(f) = q8_format(w, m).filter(|&f| q8_layout(f) as u32 == layout) else {
            return Err(Error::Unsupported("linear_q8 view".into()));
        };
        let (mi, rows) = (to_i32("m", m)?, to_i32("rows", w.rows)?);
        let (q, wp) = (xq.as_ptr().cast::<u8>(), w.buf.as_ptr().cast::<u8>());
        let (pr, k, rb) = (w.parent_rows as i64, w.k as i64, w.row_begin as i64);
        // SAFETY: CudaBackend::linear_q8 checked the twin, view and output extents.
        let code = unsafe {
            if f >= 16 {
                velox_4090_linear_ggml_q8(
                    q,
                    mi,
                    wp,
                    f - 16,
                    k,
                    rb,
                    rows,
                    y.as_mut_ptr(),
                    rows as i64,
                    0,
                )
            } else if f == 3 {
                velox_4090_linear_iq3_q8_from(
                    q,
                    mi,
                    wp,
                    pr,
                    k,
                    rb,
                    rows,
                    y.as_mut_ptr(),
                    rows as i64,
                )
            } else {
                velox_4090_linear_rs_q8_from(
                    q,
                    mi,
                    wp,
                    pr,
                    k,
                    rb,
                    rows,
                    f,
                    y.as_mut_ptr(),
                    rows as i64,
                )
            }
        };
        status(code, "rtx4090 linear q8 twin")
    }

    fn linear_swiglu_q8(
        &self,
        xq: &CudaBuffer,
        layout: u32,
        gate: &WeightView<'_, CudaBuffer>,
        up: &WeightView<'_, CudaBuffer>,
        out: &mut CudaBuffer,
        m: usize,
    ) -> Result<()> {
        if self.linear_swiglu_act(gate, up, m) != Some(ActQuant::Int8G256) || layout != 0 {
            return Err(Error::Unsupported("linear_swiglu_q8 views".into()));
        }
        if ggml_q8_pair(gate, up, m) {
            return self.ggml_swiglu_q8(xq, Some(()), gate, up, out, m);
        }
        // SAFETY: CudaBackend::linear_swiglu_q8 checked the twin, views and output extents.
        status(
            unsafe {
                velox_4090_linear_swiglu_iq3_q8_from(
                    xq.as_ptr().cast::<u8>(),
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
            "rtx4090 linear_swiglu q8 twin",
        )
    }

    fn takes_bf16(&self, w: &WeightView<'_, CudaBuffer>, m: usize) -> bool {
        self.linear_act(w, m) == ActQuant::Bf16 && bf16_in(w, m).is_some()
    }

    fn linear_bf16(
        &self,
        xb: &CudaBuffer,
        w: &WeightView<'_, CudaBuffer>,
        y: &mut CudaBuffer,
        m: usize,
    ) -> Result<()> {
        let (mi, rows) = (to_i32("m", m)?, to_i32("rows", w.rows)?);
        let (xp, wp) = (xb.as_ptr().cast::<u8>(), w.buf.as_ptr().cast::<u8>());
        let (pr, k, rb) = (w.parent_rows as i64, w.k as i64, w.row_begin as i64);
        let yp = y.as_mut_ptr();
        // SAFETY: CudaBackend::linear_bf16 checked the twin, view and output extents.
        let code = match bf16_in(w, m) {
            Some(Bf16In::Iq3) => unsafe {
                velox_4090_linear_iq3_direct_bf16(xp, mi, wp, pr, k, rb, rows, yp)
            },
            Some(Bf16In::Rs(f)) => unsafe {
                velox_4090_linear_rs_sliced_bf16(xp, mi, wp, pr, k, rb, rows, f, yp, rows as i64)
            },
            Some(Bf16In::Nvfp4) => unsafe {
                velox_4090_linear_nvfp4_sliced_bf16(xp, mi, wp, pr, k, rb, rows, yp)
            },
            Some(Bf16In::Nvfp4Mma) => unsafe {
                velox_4090_linear_nvfp4_mma_bf16(xp, mi, wp, pr, k, rb, rows, yp)
            },
            None => return Err(Error::Unsupported("linear_bf16 view".into())),
        };
        status(code, "rtx4090 linear bf16 twin")
    }

    fn swiglu_twin(
        &self,
        gate: &WeightView<'_, CudaBuffer>,
        up: &WeightView<'_, CudaBuffer>,
        m: usize,
    ) -> bool {
        self.linear_swiglu_act(gate, up, m) == Some(ActQuant::Bf16)
            && gate.format == WeightFormat::Iq3Xxs
            && iq3_direct(m)
    }

    fn linear_swiglu_twin(
        &self,
        xb: &CudaBuffer,
        gate: &WeightView<'_, CudaBuffer>,
        up: &WeightView<'_, CudaBuffer>,
        out: &mut CudaBuffer,
        outb: &mut CudaBuffer,
        m: usize,
    ) -> Result<()> {
        // SAFETY: CudaBackend::linear_swiglu_twin checked the twins, views and output extents.
        status(
            unsafe {
                velox_4090_linear_swiglu_iq3_direct_twin(
                    xb.as_ptr().cast::<u8>(),
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
                    outb.as_mut_ptr().cast(),
                )
            },
            "rtx4090 linear_swiglu twin",
        )
    }

    /// Prefill widths outside recorded steps: the chunked recurrence on bf16 tensor cores (f32
    /// state, bf16 products; ninfer's precision). Qwen3.8 (48 heads, 128 x 128) at 1024 tokens:
    /// 657 us exact f32 -> see the commit. Decode widths and recorded steps stay on main's exact
    /// kernels.
    #[allow(clippy::too_many_arguments)]
    fn gated_delta_net(
        &self,
        qkv: &CudaBuffer,
        ab: &CudaBuffer,
        a_log: &CudaBuffer,
        dt_bias: &CudaBuffer,
        state_in: Option<&CudaBuffer>,
        state: &mut CudaBuffer,
        out: &mut CudaBuffer,
        s: velox_core::backend::GdnShape,
    ) -> Result<()> {
        let chunked = s.tokens >= 16
            && s.v_dim % 4 == 0
            && matches!(s.k_dim, 32 | 64 | 128)
            && s.v_heads % s.k_heads == 0
            && graph_mode() == GraphMode::None;
        if chunked {
            let si = state_in.map_or(state.as_ptr(), CudaBuffer::as_ptr);
            // SAFETY: CudaBackend validated the buffers against `s` (check::gdn) before this hook.
            let code = unsafe {
                velox_4090_gdn_chunk(
                    qkv.as_ptr().add(s.row0 * s.qkv_width()),
                    ab.as_ptr().add(s.row0 * 2 * s.v_heads),
                    a_log.as_ptr(),
                    dt_bias.as_ptr(),
                    si,
                    state.as_mut_ptr(),
                    out.as_mut_ptr().add(s.row0 * s.v_heads * s.v_dim),
                    to_i32("tokens", s.tokens)?,
                    to_i32("k_heads", s.k_heads)?,
                    to_i32("v_heads", s.v_heads)?,
                    to_i32("k_dim", s.k_dim)?,
                    to_i32("v_dim", s.v_dim)?,
                )
            };
            return or_fallback(status(code, "rtx4090 gdn chunk"), || {
                portable::gated_delta_net(qkv, ab, a_log, dt_bias, state_in, state, out, s)
            });
        }
        if !gdn_step(qkv, &s) {
            return portable::gated_delta_net(qkv, ab, a_log, dt_bias, state_in, state, out, s);
        }
        let si = state_in.map_or(state.as_ptr(), CudaBuffer::as_ptr);
        let Some(pp) = portable::graph_params(&[si as i64, state.as_mut_ptr() as i64])? else {
            return Ok(());
        };
        // SAFETY: CudaBackend validated the buffers against `s` (check::gdn); `gdn_step` checked
        // the kernel's shape contract, so it launches (no fallback after staging).
        status(
            unsafe {
                velox_4090_gdn_step(
                    qkv.as_ptr().add(s.row0 * s.qkv_width()),
                    ab.as_ptr().add(s.row0 * 2 * s.v_heads),
                    a_log.as_ptr(),
                    dt_bias.as_ptr(),
                    si,
                    state.as_mut_ptr(),
                    out.as_mut_ptr().add(s.row0 * s.v_heads * s.v_dim),
                    to_i32("tokens", s.tokens)?,
                    to_i32("k_heads", s.k_heads)?,
                    to_i32("v_heads", s.v_heads)?,
                    to_i32("k_dim", s.k_dim)?,
                    to_i32("v_dim", s.v_dim)?,
                    pp,
                )
            },
            "rtx4090 gdn step",
        )
    }

    fn gated_delta_net_out(
        &self,
        qkv: &CudaBuffer,
        ab: &CudaBuffer,
        a_log: &CudaBuffer,
        dt_bias: &CudaBuffer,
        state_in: &CudaBuffer,
        out: &mut CudaBuffer,
        s: velox_core::backend::GdnShape,
    ) -> Result<()> {
        if !gdn_step(qkv, &s) {
            return portable::gated_delta_net_out(qkv, ab, a_log, dt_bias, state_in, out, s);
        }
        let Some(pp) = portable::graph_params(&[state_in.as_ptr() as i64, 0])? else {
            return Ok(());
        };
        // SAFETY: as in `gated_delta_net`; a null state is never written.
        status(
            unsafe {
                velox_4090_gdn_step(
                    qkv.as_ptr().add(s.row0 * s.qkv_width()),
                    ab.as_ptr().add(s.row0 * 2 * s.v_heads),
                    a_log.as_ptr(),
                    dt_bias.as_ptr(),
                    state_in.as_ptr(),
                    std::ptr::null_mut(),
                    out.as_mut_ptr().add(s.row0 * s.v_heads * s.v_dim),
                    to_i32("tokens", s.tokens)?,
                    to_i32("k_heads", s.k_heads)?,
                    to_i32("v_heads", s.v_heads)?,
                    to_i32("k_dim", s.k_dim)?,
                    to_i32("v_dim", s.v_dim)?,
                    pp,
                )
            },
            "rtx4090 gdn step",
        )
    }

    fn linear_swiglu_act(
        &self,
        gate: &WeightView<'_, CudaBuffer>,
        up: &WeightView<'_, CudaBuffer>,
        m: usize,
    ) -> Option<ActQuant> {
        // GGML gate / up at prefill widths (one of them may be IQ3_XXS): the int8 prefill kernels,
        // one quantization, the GGML GEMM's epilogue forming silu(gate) * up in place.
        if gate.k == up.k && gate.rows == up.rows && ggml_q8_pair(gate, up, m) {
            return Some(ActQuant::Int8G256);
        }
        // GGML gate / up (GGUF imports): one Int8G32 quantization, the gate GEMM into `out`, the up
        // GEMM's epilogue forming silu(gate) * up in place. (VELOX_GGML_NO_SWIGLU: composed, A/B.)
        static NO_FUSE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let fuse = !*NO_FUSE.get_or_init(|| std::env::var_os("VELOX_GGML_NO_SWIGLU").is_some());
        if fuse
            && gate.k == up.k
            && gate.rows == up.rows
            && ggml_mmvq(gate, m).is_some()
            && ggml_mmvq(up, m).is_some()
        {
            return Some(ActQuant::Int8G32);
        }
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
            Some(ActQuant::Int8G32) => {
                let (tg, tu) = (ggml_mmvq(gate, m), ggml_mmvq(up, m));
                let (Some(tg), Some(tu)) = (tg, tu) else {
                    return Err(Error::Unsupported("ggml swiglu shape".into()));
                };
                // Gate into out, then up with silu(out) * up in its epilogue (same stream order).
                let act = GgmlAct::new(x, m, gate.k)?;
                act.run(gate, tg, m, out.as_mut_ptr(), gate.rows, 0)?;
                return act.run(up, tu, m, out.as_mut_ptr(), gate.rows, 1);
            }
            Some(ActQuant::Int8G256) if ggml_q8_pair(gate, up, m) => {
                return self.ggml_swiglu_q8(x, None, gate, up, out, m);
            }
            Some(ActQuant::Int8G256) => velox_4090_linear_swiglu_iq3_q8,
            Some(_) if iq3_direct(m) => velox_4090_linear_swiglu_iq3_direct,
            Some(_) => velox_4090_linear_swiglu_skinny_iq3,
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
            "rtx4090 linear_swiglu iq3",
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
                        velox_4090_gemv(x.as_ptr(), mi, wp, pr, k, rb, rows, f, y.as_mut_ptr())
                    },
                    "rtx4090 gemv",
                ),
                None => portable::linear(x, w, y, m),
            },
            // Ada has no FP4 tensor cores and no full-precision E4M3 MMA: linear_act never reports
            // 4- or 8-bit float activations.
            ActQuant::Nvfp4 { .. } | ActQuant::Fp8Row => Err(Error::Unsupported(
                "rtx4090 float8 / float4 activations".into(),
            )),
            ActQuant::Int8G32 => {
                let ty = ggml_mmvq(w, m).ok_or_else(|| Error::Unsupported("ggml shape".into()))?;
                GgmlAct::new(x, m, w.k)?.run(w, ty, m, y.as_mut_ptr(), w.rows, 0)
            }
            ActQuant::Int8G256 => match q8_format(w, m) {
                Some(f) if f >= 16 => {
                    status(
                        unsafe { velox_4090_q8_quantize(x.as_ptr(), mi, k, 0) },
                        "rtx4090 q8 quantize",
                    )?;
                    status(
                        unsafe {
                            velox_4090_linear_ggml_q8(
                                std::ptr::null(),
                                mi,
                                wp,
                                f - 16,
                                k,
                                rb,
                                rows,
                                y.as_mut_ptr(),
                                rows as i64,
                                0,
                            )
                        },
                        "rtx4090 linear ggml q8",
                    )
                }
                Some(3) => status(
                    unsafe {
                        velox_4090_linear_iq3_q8(
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
                    "rtx4090 linear iq3 q8",
                ),
                Some(f) => status(
                    unsafe {
                        velox_4090_linear_rs_q8(
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
                    "rtx4090 linear rs q8",
                ),
                None => Err(Error::Unsupported("int8 linear shape".into())),
            },
            ActQuant::Bf16 => {
                if bf16_skinny(w, m) {
                    return status(
                        unsafe {
                            velox_4090_linear_bf16_skinny(
                                x.as_ptr(),
                                mi,
                                wp,
                                k,
                                rb,
                                rows,
                                y.as_mut_ptr(),
                                rows as i64,
                            )
                        },
                        "rtx4090 linear bf16 skinny",
                    );
                }
                if bf16_mma(w, m) {
                    return status(
                        unsafe {
                            velox_4090_linear_bf16_mma(
                                x.as_ptr(),
                                mi,
                                w.buf.as_ptr().cast(),
                                w.k as i64,
                                w.row_begin as i64,
                                rows,
                                y.as_mut_ptr(),
                                w.rows as i64,
                            )
                        },
                        "rtx4090 linear bf16 mma",
                    );
                }
                if nvfp4_mma(w, m) {
                    return status(
                        unsafe {
                            velox_4090_linear_nvfp4_mma(
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
                        "rtx4090 linear nvfp4 mma",
                    );
                }
                if skinny_iq3(w, m) && m > SKINNY16_MAX_M {
                    // Prefill: the staged kernel tiles the tokens itself (64 per weight pass).
                    return status(
                        unsafe {
                            velox_4090_linear_skinny_iq3(
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
                        "rtx4090 linear iq3 prefill",
                    );
                }
                if skinny_iq3(w, m) {
                    let direct = iq3_direct(m);
                    // The kernel takes up to 16 rows; larger m runs in blocks.
                    for r0 in (0..m).step_by(SKINNY16_MAX_M) {
                        let mb = to_i32("m", (m - r0).min(SKINNY16_MAX_M))?;
                        // SAFETY: rows r0.. of x and y lie inside the buffers check::linear validated.
                        let (xp, yp) =
                            unsafe { (x.as_ptr().add(r0 * w.k), y.as_mut_ptr().add(r0 * w.rows)) };
                        status(
                            unsafe {
                                if direct {
                                    velox_4090_linear_iq3_direct(xp, mb, wp, pr, k, rb, rows, yp)
                                } else {
                                    velox_4090_linear_skinny_iq3(xp, mb, wp, pr, k, rb, rows, yp)
                                }
                            },
                            "rtx4090 linear iq3",
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
                            unsafe { velox_4090_linear_a16_fp8(xp, mb, wp, pr, k, rb, rows, yp) },
                            "rtx4090 linear a16 fp8",
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
                                    velox_4090_linear_nvfp4_sliced(xp, mb, wp, pr, k, rb, rows, yp)
                                },
                                "rtx4090 linear nvfp4 sliced",
                            )?;
                        } else if f == 0 {
                            status(
                                unsafe {
                                    velox_4090_linear_skinny16(xp, mb, wp, pr, k, rb, rows, f, yp)
                                },
                                "rtx4090 linear skinny16",
                            )?;
                        } else {
                            // q4 / q5: the sliced-K kernel (16 rows per block, 4 warps over each
                            // 256-value K tile); q8: the staged 128-row kernel. Measured at m=8: q5
                            // [12288,5120] 123 vs 186 us, q4 [8192,5120] 84 vs 90 us.
                            status(
                                unsafe {
                                    if f == 8 {
                                        velox_4090_linear_skinny_rs(
                                            xp, mb, wp, pr, k, rb, rows, f, yp,
                                        )
                                    } else {
                                        velox_4090_linear_rs_sliced(
                                            xp, mb, wp, pr, k, rb, rows, f, yp,
                                        )
                                    }
                                },
                                "rtx4090 linear rs",
                            )?;
                        }
                    }
                    return Ok(());
                }
                let bits = w.format.row_split().map_or(0, |r| r.bits as i32);
                status(
                    unsafe {
                        velox_4090_linear_rowsplit(
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
                    "rtx4090 linear rowsplit",
                )
            }
        }
    }
}
