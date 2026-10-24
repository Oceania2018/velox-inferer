//! Safe wrappers over the portable kernels. Variants call these as the fallback for shapes their
//! tuned kernels do not cover. Each wrapper re-validates, so it is safe to call directly.

use velox_core::backend::{
    AttnShape, BlockAttnShape, ConvShape, Copy2dShape, DilatedConvShape, DynConvShape, ExpertBank,
    ExpertMatrix, FullAttnShape, GdnReplayLayer, GdnShape, GeluKind, HcCombineShape, KvStoreShape,
    MoeRouteShape, NormKind, QsaPoolShape, QsaSelectShape, RopeLayout, RopePosShape, RopeShape,
    RowGateShape, WeightView,
};
use velox_core::check::{self, to_i32};
use velox_core::kv::{hadamard_norm, KvFormat};
use velox_core::{Error, Result, WeightFormat, GRAPH_KV_BUCKET};

use crate::{dtod, graph_mode, stage, status, CudaBuffer, GraphMode};

extern "C" {
    fn velox_portable_matmul(
        x: *const f32,
        w: *const f32,
        y: *mut f32,
        m: i32,
        n: i32,
        k: i32,
    ) -> i32;
    fn velox_portable_add_rms_norm(
        acc: *mut f32,
        x: *const f32,
        weight: *const f32,
        y: *mut f32,
        rows: i32,
        dim: i32,
        eps: f32,
        offset: i32,
    ) -> i32;
    fn velox_portable_rms_norm(
        x: *const f32,
        weight: *const f32,
        y: *mut f32,
        rows: i32,
        dim: i32,
        eps: f32,
        offset: i32,
    ) -> i32;
    fn velox_portable_rms_norm_gated(
        x: *const f32,
        weight: *const f32,
        gate: *mut f32,
        rows: i32,
        dim: i32,
        eps: f32,
        offset: i32,
    ) -> i32;
    fn velox_portable_rope(
        x: *mut f32,
        tokens: i32,
        heads: i32,
        head_dim: i32,
        rotary_dim: i32,
        pos0: i32,
        theta: f32,
        params: *const i64,
    ) -> i32;
    fn velox_portable_silu_mul(gate: *mut f32, up: *const f32, len: i64) -> i32;
    fn velox_portable_add(acc: *mut f32, x: *const f32, len: i64) -> i32;
    fn velox_portable_attention(
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
        params: *const i64,
        kv_grid: i32,
    ) -> i32;
    fn velox_portable_linear(
        x: *const f32,
        w: *const u8,
        format: i32,
        n: i64,
        k: i64,
        row_begin: i64,
        rows: i32,
        m: i32,
        y: *mut f32,
    ) -> i32;
    fn velox_portable_embed(
        ids: *const u32,
        count: i32,
        w: *const u8,
        format: i32,
        n: i64,
        k: i64,
        row_begin: i64,
        out: *mut f32,
    ) -> i32;
    fn velox_portable_sigmoid_mul(x: *mut f32, gate: *const f32, len: i64) -> i32;
    fn velox_portable_sigmoid_mul_twin(
        x: *mut f32,
        gate: *const f32,
        xb: *mut u16,
        len: i64,
    ) -> i32;
    fn velox_portable_add_rms_norm_twin(
        acc: *mut f32,
        x: *const f32,
        weight: *const f32,
        y: *mut f32,
        yb: *mut u16,
        rows: i32,
        dim: i32,
        eps: f32,
        offset: i32,
    ) -> i32;
    fn velox_portable_rms_norm_gated_twin(
        x: *const f32,
        weight: *const f32,
        gate: *mut f32,
        gb: *mut u16,
        rows: i32,
        dim: i32,
        eps: f32,
        offset: i32,
    ) -> i32;
    fn velox_portable_conv1d_silu(
        x: *const f32,
        w: *const f32,
        state_in: *const f32,
        state: *mut f32,
        out: *mut f32,
        tokens: i32,
        channels: i32,
        width: i32,
        pp: *const i64,
    ) -> i32;
    fn velox_portable_gated_delta_net(
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
}

/// Format codes shared with the kernels (`VELOX_FMT_*` in portable_model.cu).
pub fn format_code(f: WeightFormat) -> i32 {
    match f {
        WeightFormat::F32 => 0,
        WeightFormat::Bf16 => 1,
        WeightFormat::Fp8RowBf16 => 2,
        WeightFormat::Nvfp4 => 3,
        WeightFormat::Q4G64 => 4,
        WeightFormat::Q5G64 => 5,
        WeightFormat::Q6G64 => 6,
        WeightFormat::Q8G32 => 7,
        WeightFormat::Iq3Xxs => 8,
        // VELOX_FMT_GGML_BASE + the type's position in GgmlType::ALL.
        WeightFormat::Ggml(t) => {
            16 + velox_core::ggml::GgmlType::ALL
                .iter()
                .position(|&u| u == t)
                .expect("listed type") as i32
        }
    }
}

pub fn matmul(
    x: &CudaBuffer,
    w: &CudaBuffer,
    y: &mut CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    check::matmul(x.len(), w.len(), y.len(), m, n, k)?;
    let (m, n, k) = (to_i32("m", m)?, to_i32("n", n)?, to_i32("k", k)?);
    // SAFETY: buffer extents validated above.
    status(
        unsafe { velox_portable_matmul(x.as_ptr(), w.as_ptr(), y.as_mut_ptr(), m, n, k) },
        "portable matmul",
    )
}

pub fn linear(
    x: &CudaBuffer,
    w: &WeightView<'_, CudaBuffer>,
    y: &mut CudaBuffer,
    m: usize,
) -> Result<()> {
    check::linear(x.len(), w, w.buf.len(), y.len(), m)?;
    // SAFETY: input, weight payload and output extents validated above.
    status(
        unsafe {
            velox_portable_linear(
                x.as_ptr(),
                w.buf.as_ptr().cast(),
                format_code(w.format),
                w.parent_rows as i64,
                w.k as i64,
                w.row_begin as i64,
                to_i32("rows", w.rows)?,
                to_i32("m", m)?,
                y.as_mut_ptr(),
            )
        },
        "portable linear",
    )
}

pub fn embed(ids: &[u32], table: &WeightView<'_, CudaBuffer>, out: &mut CudaBuffer) -> Result<()> {
    check::embed(ids, table, table.buf.len(), out.len())?;
    if ids.is_empty() {
        return Ok(());
    }
    let id_bytes: Vec<u8> = ids.iter().flat_map(|i| i.to_le_bytes()).collect();
    let dev_ids = stage(&id_bytes)?;
    if graph_mode() == GraphMode::Replay {
        return Ok(());
    }
    // SAFETY: ids are in range (checked) and out holds ids.len() rows.
    status(
        unsafe {
            velox_portable_embed(
                dev_ids.cast(),
                to_i32("ids", ids.len())?,
                table.buf.as_ptr().cast(),
                format_code(table.format),
                table.parent_rows as i64,
                table.k as i64,
                table.row_begin as i64,
                out.as_mut_ptr(),
            )
        },
        "portable embed",
    )
}

/// [`embed`] reading `count` ids at `ids[offset..]` on the device (no staging: the address is
/// part of a recording).
pub fn embed_from(
    ids: &CudaBuffer,
    offset: usize,
    count: usize,
    table: &WeightView<'_, CudaBuffer>,
    out: &mut CudaBuffer,
) -> Result<()> {
    check::embed_from(ids.len(), offset, count, table, table.buf.len(), out.len())?;
    if count == 0 {
        return Ok(());
    }
    // SAFETY: ids holds offset + count slots and out holds count rows (checked); the kernels
    // write zeros for an id past the table.
    status(
        unsafe {
            velox_portable_embed(
                ids.as_ptr().add(offset).cast(),
                to_i32("ids", count)?,
                table.buf.as_ptr().cast(),
                format_code(table.format),
                table.parent_rows as i64,
                table.k as i64,
                table.row_begin as i64,
                out.as_mut_ptr(),
            )
        },
        "portable embed_from",
    )
}

/// See [`velox_core::backend::Backend::argmax_to`].
pub fn argmax_to(
    x: &CudaBuffer,
    limit: usize,
    dst: &mut CudaBuffer,
    dst_offset: usize,
) -> Result<()> {
    if limit == 0 || limit > x.len() {
        return Err(velox_core::Error::Shape(format!(
            "argmax_to: limit {limit} outside the {} values",
            x.len()
        )));
    }
    if dst_offset >= dst.len() {
        return Err(velox_core::Error::Shape(format!(
            "argmax_to: slot {dst_offset} past the {} slots",
            dst.len()
        )));
    }
    // SAFETY: extents checked above.
    status(
        unsafe {
            velox_portable_argmax_to(
                x.as_ptr(),
                x.len() as i64,
                to_i32("limit", limit)?,
                dst.as_mut_ptr().add(dst_offset).cast(),
            )
        },
        "portable argmax_to",
    )
}

/// See [`velox_core::backend::Backend::fetch_u32_begin`].
pub fn fetch_u32_begin(src: &CudaBuffer, offset: usize, slot: usize) -> Result<()> {
    if offset >= src.len() {
        return Err(velox_core::Error::Shape(format!(
            "fetch_u32: slot {offset} past the {} slots",
            src.len()
        )));
    }
    // SAFETY: offset is in bounds (checked); the shim copies 4 bytes into its pinned ring.
    status(
        unsafe {
            velox_portable_fetch_u32_begin(src.as_ptr().add(offset).cast(), to_i32("slot", slot)?)
        },
        "portable fetch_u32",
    )
}

/// See [`velox_core::backend::Backend::fetch_u32`].
pub fn fetch_u32(slot: usize) -> Result<u32> {
    let mut v = 0u32;
    // SAFETY: out-pointer to a live local.
    status(
        unsafe { velox_portable_fetch_u32_wait(to_i32("slot", slot)?, &mut v) },
        "portable fetch_u32",
    )?;
    Ok(v)
}

#[allow(clippy::too_many_arguments)]
pub fn add_rms_norm(
    acc: &mut CudaBuffer,
    x: &CudaBuffer,
    weight: &CudaBuffer,
    y: &mut CudaBuffer,
    rows: usize,
    dim: usize,
    eps: f32,
    kind: NormKind,
) -> Result<()> {
    check::rms_norm(acc.len(), weight.len(), y.len(), rows, dim)?;
    check::elementwise("add_rms_norm", &[x.len()], rows * dim)?;
    let (rows, dim) = (to_i32("rows", rows)?, to_i32("dim", dim)?);
    // SAFETY: buffer extents validated above.
    status(
        unsafe {
            velox_portable_add_rms_norm(
                acc.as_mut_ptr(),
                x.as_ptr(),
                weight.as_ptr(),
                y.as_mut_ptr(),
                rows,
                dim,
                eps,
                i32::from(kind == NormKind::Offset),
            )
        },
        "portable add_rms_norm",
    )
}

pub fn rms_norm(
    x: &CudaBuffer,
    weight: &CudaBuffer,
    y: &mut CudaBuffer,
    rows: usize,
    dim: usize,
    eps: f32,
    kind: NormKind,
) -> Result<()> {
    check::rms_norm(x.len(), weight.len(), y.len(), rows, dim)?;
    let (rows, dim) = (to_i32("rows", rows)?, to_i32("dim", dim)?);
    let offset = i32::from(kind == NormKind::Offset);
    // SAFETY: buffer extents validated above.
    status(
        unsafe {
            velox_portable_rms_norm(
                x.as_ptr(),
                weight.as_ptr(),
                y.as_mut_ptr(),
                rows,
                dim,
                eps,
                offset,
            )
        },
        "portable rms_norm",
    )
}

#[allow(clippy::too_many_arguments)]
pub fn rms_norm_gated(
    x: &CudaBuffer,
    weight: &CudaBuffer,
    gate: &mut CudaBuffer,
    rows: usize,
    dim: usize,
    eps: f32,
    kind: NormKind,
) -> Result<()> {
    check::rms_norm(x.len(), weight.len(), gate.len(), rows, dim)?;
    let (rows, dim) = (to_i32("rows", rows)?, to_i32("dim", dim)?);
    let offset = i32::from(kind == NormKind::Offset);
    // SAFETY: buffer extents validated above.
    status(
        unsafe {
            velox_portable_rms_norm_gated(
                x.as_ptr(),
                weight.as_ptr(),
                gate.as_mut_ptr(),
                rows,
                dim,
                eps,
                offset,
            )
        },
        "portable rms_norm_gated",
    )
}

pub fn rope(x: &mut CudaBuffer, s: RopeShape) -> Result<()> {
    check::rope(x.len(), &s)?;
    let (t, h, d, r) = (
        to_i32("tokens", s.tokens)?,
        to_i32("heads", s.heads)?,
        to_i32("head_dim", s.head_dim)?,
        to_i32("rotary_dim", s.rotary_dim)?,
    );
    // The kernel forms positions as pos0 + t in i32.
    to_i32("pos0 + tokens", s.pos0 + s.tokens)?;
    let Some(params) = graph_params(&[s.pos0 as i64])? else {
        return Ok(());
    };
    // SAFETY: buffer extent validated above.
    status(
        unsafe { velox_portable_rope(x.as_mut_ptr(), t, h, d, r, s.pos0 as i32, s.theta, params) },
        "portable rope",
    )
}

pub fn silu_mul(gate: &mut CudaBuffer, up: &CudaBuffer, len: usize) -> Result<()> {
    check::elementwise("silu_mul", &[gate.len(), up.len()], len)?;
    // SAFETY: buffer extents validated above.
    status(
        unsafe { velox_portable_silu_mul(gate.as_mut_ptr(), up.as_ptr(), len as i64) },
        "portable silu_mul",
    )
}

pub fn sigmoid_mul(x: &mut CudaBuffer, gate: &CudaBuffer, len: usize) -> Result<()> {
    check::elementwise("sigmoid_mul", &[x.len(), gate.len()], len)?;
    // SAFETY: buffer extents validated above.
    status(
        unsafe { velox_portable_sigmoid_mul(x.as_mut_ptr(), gate.as_ptr(), len as i64) },
        "portable sigmoid_mul",
    )
}

/// A bf16 twin of `n` values packs two per f32 slot.
fn check_twin(what: &str, twin: &CudaBuffer, n: usize) -> Result<()> {
    check::elementwise(what, &[twin.len() * 2], n)
}

/// [`add_rms_norm`] that also writes `y` rounded to bf16 into `yb` (see `Backend::add_rms_norm_twin`).
#[allow(clippy::too_many_arguments)]
pub fn add_rms_norm_twin(
    acc: &mut CudaBuffer,
    x: &CudaBuffer,
    weight: &CudaBuffer,
    y: &mut CudaBuffer,
    yb: &mut CudaBuffer,
    rows: usize,
    dim: usize,
    eps: f32,
    kind: NormKind,
) -> Result<()> {
    check::rms_norm(acc.len(), weight.len(), y.len(), rows, dim)?;
    check::elementwise("add_rms_norm", &[x.len()], rows * dim)?;
    check_twin("add_rms_norm twin", yb, rows * dim)?;
    let (rows, dim) = (to_i32("rows", rows)?, to_i32("dim", dim)?);
    // SAFETY: buffer extents validated above.
    status(
        unsafe {
            velox_portable_add_rms_norm_twin(
                acc.as_mut_ptr(),
                x.as_ptr(),
                weight.as_ptr(),
                y.as_mut_ptr(),
                yb.as_mut_ptr().cast(),
                rows,
                dim,
                eps,
                i32::from(kind == NormKind::Offset),
            )
        },
        "portable add_rms_norm twin",
    )
}

/// [`rms_norm_gated`] that also writes the result rounded to bf16 into `gb`.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_gated_twin(
    x: &CudaBuffer,
    weight: &CudaBuffer,
    gate: &mut CudaBuffer,
    gb: &mut CudaBuffer,
    rows: usize,
    dim: usize,
    eps: f32,
    kind: NormKind,
) -> Result<()> {
    check::rms_norm(x.len(), weight.len(), gate.len(), rows, dim)?;
    check_twin("rms_norm_gated twin", gb, rows * dim)?;
    let (rows, dim) = (to_i32("rows", rows)?, to_i32("dim", dim)?);
    // SAFETY: buffer extents validated above.
    status(
        unsafe {
            velox_portable_rms_norm_gated_twin(
                x.as_ptr(),
                weight.as_ptr(),
                gate.as_mut_ptr(),
                gb.as_mut_ptr().cast(),
                rows,
                dim,
                eps,
                i32::from(kind == NormKind::Offset),
            )
        },
        "portable rms_norm_gated twin",
    )
}

/// [`sigmoid_mul`] that also writes the result rounded to bf16 into `xb`.
pub fn sigmoid_mul_twin(
    x: &mut CudaBuffer,
    gate: &CudaBuffer,
    xb: &mut CudaBuffer,
    len: usize,
) -> Result<()> {
    check::elementwise("sigmoid_mul", &[x.len(), gate.len()], len)?;
    check_twin("sigmoid_mul twin", xb, len)?;
    // SAFETY: buffer extents validated above.
    status(
        unsafe {
            velox_portable_sigmoid_mul_twin(
                x.as_mut_ptr(),
                gate.as_ptr(),
                xb.as_mut_ptr().cast(),
                len as i64,
            )
        },
        "portable sigmoid_mul twin",
    )
}

pub fn add(acc: &mut CudaBuffer, x: &CudaBuffer, len: usize) -> Result<()> {
    check::elementwise("add", &[acc.len(), x.len()], len)?;
    // SAFETY: buffer extents validated above.
    status(
        unsafe { velox_portable_add(acc.as_mut_ptr(), x.as_ptr(), len as i64) },
        "portable add",
    )
}

pub fn copy(
    src: &CudaBuffer,
    src_offset: usize,
    dst: &mut CudaBuffer,
    dst_offset: usize,
    len: usize,
) -> Result<()> {
    check::copy(src.len(), src_offset, dst.len(), dst_offset, len)?;
    if len == 0 {
        return Ok(());
    }
    if graph_mode() != GraphMode::None {
        // A graph step copies at offsets it stages, so its recording does not depend on them.
        let Some(params) = graph_params(&[src_offset as i64, dst_offset as i64])? else {
            return Ok(());
        };
        // SAFETY: both ranges validated above; the kernel adds the staged offsets to the bases.
        return status(
            unsafe {
                velox_portable_copy_param(src.as_ptr(), dst.as_mut_ptr(), len as i64, params)
            },
            "portable copy",
        );
    }
    // SAFETY: both ranges validated above, so the offset pointers are in bounds.
    dtod(
        unsafe { dst.as_mut_ptr().add(dst_offset) },
        unsafe { src.as_ptr().add(src_offset) },
        len,
    )
}

pub fn attention(
    q: &CudaBuffer,
    k: &CudaBuffer,
    v: &CudaBuffer,
    out: &mut CudaBuffer,
    s: AttnShape,
) -> Result<()> {
    check::attention(q.len(), k.len(), v.len(), out.len(), &s)?;
    to_i32("kv_len", s.kv_len())?;
    // Graph steps stage the caches too: a recording serves any session.
    let Some(params) = graph_params(&[s.pos0 as i64, k.as_ptr() as i64, v.as_ptr() as i64])? else {
        return Ok(());
    };
    let kv_grid = to_i32("kv grid", kv_grid(s.kv_len()))?;
    // Decode and verify windows over f32 caches also take the GQA verify kernel of the KV path,
    // which reads each K/V tile once per block for every query row of the group.
    let verify =
        s.tokens <= 16 && matches!(s.head_dim, 32 | 64 | 128 | 256) && s.heads / s.kv_heads > 1;
    if s.kv != KvFormat::F32 || verify {
        // SAFETY: buffer extents validated above (caches sized for the format).
        return status(
            unsafe {
                velox_portable_attention_kv(
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
                    kv_format_code(s.kv),
                    hadamard_norm(s.head_dim),
                    params,
                    kv_grid,
                )
            },
            "portable attention (KV path)",
        );
    }
    // SAFETY: buffer extents validated above.
    status(
        unsafe {
            velox_portable_attention(
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
                params,
                kv_grid,
            )
        },
        "portable attention",
    )
}

extern "C" {
    fn velox_portable_attention_kv(
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
        format: i32,
        norm: f32,
        params: *const i64,
        kv_grid: i32,
    ) -> i32;
    fn velox_portable_kv_load(
        k_cache: *const u8,
        v_cache: *const u8,
        k: *mut f32,
        v: *mut f32,
        tokens: i32,
        kv_heads: i32,
        head_dim: i32,
        pos0: i64,
        format: i32,
        norm: f32,
    ) -> i32;
    fn velox_portable_kv_store(
        k: *const f32,
        v: *const f32,
        k_cache: *mut u8,
        v_cache: *mut u8,
        tokens: i32,
        kv_heads: i32,
        head_dim: i32,
        pos0: i64,
        format: i32,
        norm: f32,
        params: *const i64,
    ) -> i32;
}

/// KV format codes shared with the kernels (`VELOX_KV_*` in portable_kv.cu).
pub fn kv_format_code(f: KvFormat) -> i32 {
    match f {
        KvFormat::F32 => 0,
        KvFormat::Bf16 => 1,
        KvFormat::Fp8 => 2,
        KvFormat::Nvfp4 => 3,
    }
}

pub fn kv_store(
    k: &CudaBuffer,
    v: &CudaBuffer,
    k_cache: &mut CudaBuffer,
    v_cache: &mut CudaBuffer,
    s: KvStoreShape,
) -> Result<()> {
    check::kv_store(k.len(), v.len(), k_cache.len(), v_cache.len(), &s)?;
    let Some(params) = graph_params(&[
        s.pos0 as i64,
        k_cache.as_ptr() as i64,
        v_cache.as_ptr() as i64,
    ])?
    else {
        return Ok(());
    };
    // SAFETY: buffer extents validated above.
    status(
        unsafe {
            velox_portable_kv_store(
                k.as_ptr(),
                v.as_ptr(),
                k_cache.as_mut_ptr().cast(),
                v_cache.as_mut_ptr().cast(),
                to_i32("tokens", s.tokens)?,
                to_i32("kv_heads", s.kv_heads)?,
                to_i32("head_dim", s.head_dim)?,
                s.pos0 as i64,
                kv_format_code(s.format),
                hadamard_norm(s.head_dim),
                params,
            )
        },
        "portable kv_store",
    )
}

/// `state_in`: the state to advance from when it is not `state` itself (which receives the result).
pub fn conv1d_silu(
    x: &CudaBuffer,
    weight: &CudaBuffer,
    state_in: Option<&CudaBuffer>,
    state: &mut CudaBuffer,
    out: &mut CudaBuffer,
    s: ConvShape,
) -> Result<()> {
    check::conv1d(x.len(), weight.len(), state.len(), out.len(), &s)?;
    if let Some(si) = state_in {
        check::conv1d(x.len(), weight.len(), si.len(), out.len(), &s)?;
    }
    let state_in = state_in.map_or(state.as_ptr(), CudaBuffer::as_ptr);
    // Graph steps stage the states: a recording serves any session.
    let Some(pp) = graph_params(&[state_in as i64, state.as_mut_ptr() as i64])? else {
        return Ok(());
    };
    // SAFETY: buffer extents validated above.
    status(
        unsafe {
            velox_portable_conv1d_silu(
                x.as_ptr().add(s.row0 * s.channels),
                weight.as_ptr(),
                state_in,
                state.as_mut_ptr(),
                out.as_mut_ptr().add(s.row0 * s.channels),
                to_i32("tokens", s.tokens)?,
                to_i32("channels", s.channels)?,
                to_i32("width", s.width)?,
                pp,
            )
        },
        "portable conv1d_silu",
    )
}

#[allow(clippy::too_many_arguments)]
/// `state_in`: the state to update from when it is not `state` itself (which receives the result).
pub fn gated_delta_net(
    qkv: &CudaBuffer,
    ab: &CudaBuffer,
    a_log: &CudaBuffer,
    dt_bias: &CudaBuffer,
    state_in: Option<&CudaBuffer>,
    state: &mut CudaBuffer,
    out: &mut CudaBuffer,
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
    if let Some(si) = state_in {
        check::gdn(
            qkv.len(),
            ab.len(),
            a_log.len(),
            dt_bias.len(),
            si.len(),
            out.len(),
            &s,
        )?;
    }
    let state_in = state_in.map_or(state.as_ptr(), CudaBuffer::as_ptr);
    let Some(pp) = graph_params(&[state_in as i64, state.as_mut_ptr() as i64])? else {
        return Ok(());
    };
    // SAFETY: buffer extents validated above.
    status(
        unsafe {
            velox_portable_gated_delta_net(
                qkv.as_ptr().add(s.row0 * s.qkv_width()),
                ab.as_ptr().add(s.row0 * 2 * s.v_heads),
                a_log.as_ptr(),
                dt_bias.as_ptr(),
                state_in,
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
        "portable gated_delta_net",
    )
}

/// [`gated_delta_net`] from `state_in` writing only `out` (no state): the per-row kernel with a null
/// state target, also through graph staging.
pub fn gated_delta_net_out(
    qkv: &CudaBuffer,
    ab: &CudaBuffer,
    a_log: &CudaBuffer,
    dt_bias: &CudaBuffer,
    state_in: &CudaBuffer,
    out: &mut CudaBuffer,
    s: GdnShape,
) -> Result<()> {
    check::gdn(
        qkv.len(),
        ab.len(),
        a_log.len(),
        dt_bias.len(),
        state_in.len(),
        out.len(),
        &s,
    )?;
    let Some(pp) = graph_params(&[state_in.as_ptr() as i64, 0])? else {
        return Ok(());
    };
    // SAFETY: buffer extents validated above; a null state is never written.
    status(
        unsafe {
            velox_portable_gated_delta_net(
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
        "portable gated_delta_net_out",
    )
}

/// Shapes validated by the caller (`check::gdn_replay_layer` per layer).
pub fn gdn_replay(
    layers: &mut [GdnReplayLayer<'_, CudaBuffer>],
    conv: ConvShape,
    gdn: GdnShape,
) -> Result<()> {
    // The window starts at row0 of the per-row inputs; the kernels read from row 0 of these.
    let (r, ch, abw) = (conv.row0, conv.channels, 2 * gdn.v_heads);
    // SAFETY: offsets inside the buffers (check::gdn_replay_layer covered row0 + tokens rows).
    let qkv: Vec<*const f32> = layers
        .iter()
        .map(|l| unsafe { l.qkv.as_ptr().add(r * ch) })
        .collect();
    let conv_out: Vec<*const f32> = layers
        .iter()
        .map(|l| unsafe { l.conv_out.as_ptr().add(r * ch) })
        .collect();
    let ab: Vec<*const f32> = layers
        .iter()
        .map(|l| unsafe { l.ab.as_ptr().add(r * abw) })
        .collect();
    let a_log: Vec<*const f32> = layers.iter().map(|l| l.a_log.as_ptr()).collect();
    let dt_bias: Vec<*const f32> = layers.iter().map(|l| l.dt_bias.as_ptr()).collect();
    let conv_state: Vec<*mut f32> = layers
        .iter_mut()
        .map(|l| l.conv_state.as_mut_ptr())
        .collect();
    let ssm: Vec<*mut f32> = layers.iter_mut().map(|l| l.ssm.as_mut_ptr()).collect();
    // SAFETY: one pointer per layer in every array; extents validated by the caller.
    status(
        unsafe {
            velox_portable_gdn_replay(
                qkv.as_ptr(),
                conv_out.as_ptr(),
                ab.as_ptr(),
                a_log.as_ptr(),
                dt_bias.as_ptr(),
                conv_state.as_ptr(),
                ssm.as_ptr(),
                to_i32("layers", layers.len())?,
                to_i32("tokens", gdn.tokens)?,
                to_i32("channels", conv.channels)?,
                to_i32("width", conv.width)?,
                to_i32("k_heads", gdn.k_heads)?,
                to_i32("v_heads", gdn.v_heads)?,
                to_i32("k_dim", gdn.k_dim)?,
                to_i32("v_dim", gdn.v_dim)?,
            )
        },
        "portable gdn_replay",
    )
}

extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn velox_portable_gdn_replay(
        qkv: *const *const f32,
        conv_out: *const *const f32,
        ab: *const *const f32,
        a_log: *const *const f32,
        dt_bias: *const *const f32,
        conv_state: *const *mut f32,
        ssm: *const *mut f32,
        layers: i32,
        tokens: i32,
        channels: i32,
        width: i32,
        k_heads: i32,
        v_heads: i32,
        k_dim: i32,
        v_dim: i32,
    ) -> i32;
    fn velox_portable_attention_full(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        out: *mut f32,
        tokens: i32,
        heads: i32,
        head_dim: i32,
    ) -> i32;
    fn velox_portable_layer_norm(
        x: *const f32,
        w: *const f32,
        b: *const f32,
        y: *mut f32,
        rows: i32,
        dim: i32,
        eps: f32,
    ) -> i32;
    fn velox_portable_add_bias(x: *mut f32, bias: *const f32, len: i64, dim: i32) -> i32;
    fn velox_portable_gelu(x: *mut f32, len: i64, erf_kind: i32) -> i32;
    fn velox_portable_rope_positions(
        x: *mut f32,
        pos: *const u32,
        tokens: i32,
        heads: i32,
        head_dim: i32,
        rotary_dim: i32,
        theta: f32,
        axes: i32,
        layout: i32,
    ) -> i32;
}

pub fn attention_full(
    q: &CudaBuffer,
    k: &CudaBuffer,
    v: &CudaBuffer,
    out: &mut CudaBuffer,
    s: FullAttnShape,
) -> Result<()> {
    check::attention_full(q.len(), k.len(), v.len(), out.len(), &s)?;
    // SAFETY: buffer extents validated above.
    status(
        unsafe {
            velox_portable_attention_full(
                q.as_ptr(),
                k.as_ptr(),
                v.as_ptr(),
                out.as_mut_ptr(),
                to_i32("tokens", s.tokens)?,
                to_i32("heads", s.heads)?,
                to_i32("head_dim", s.head_dim)?,
            )
        },
        "portable attention_full",
    )
}

#[allow(clippy::too_many_arguments)]
pub fn layer_norm(
    x: &CudaBuffer,
    weight: &CudaBuffer,
    bias: &CudaBuffer,
    y: &mut CudaBuffer,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<()> {
    check::layer_norm(x.len(), weight.len(), bias.len(), y.len(), rows, dim)?;
    // SAFETY: buffer extents validated above.
    status(
        unsafe {
            velox_portable_layer_norm(
                x.as_ptr(),
                weight.as_ptr(),
                bias.as_ptr(),
                y.as_mut_ptr(),
                to_i32("rows", rows)?,
                to_i32("dim", dim)?,
                eps,
            )
        },
        "portable layer_norm",
    )
}

pub fn add_bias(x: &mut CudaBuffer, bias: &CudaBuffer, rows: usize, dim: usize) -> Result<()> {
    check::add_bias(x.len(), bias.len(), rows, dim)?;
    // SAFETY: buffer extents validated above.
    status(
        unsafe {
            velox_portable_add_bias(
                x.as_mut_ptr(),
                bias.as_ptr(),
                (rows * dim) as i64,
                to_i32("dim", dim)?,
            )
        },
        "portable add_bias",
    )
}

pub fn gelu(x: &mut CudaBuffer, len: usize, kind: GeluKind) -> Result<()> {
    check::elementwise("gelu", &[x.len()], len)?;
    // SAFETY: buffer extent validated above.
    status(
        unsafe {
            velox_portable_gelu(x.as_mut_ptr(), len as i64, i32::from(kind == GeluKind::Erf))
        },
        "portable gelu",
    )
}

pub fn rope_positions(x: &mut CudaBuffer, positions: &[u32], s: RopePosShape) -> Result<()> {
    check::rope_positions(x.len(), positions, &s)?;
    let bytes: Vec<u8> = positions.iter().flat_map(|p| p.to_le_bytes()).collect();
    let pos = stage(&bytes)?;
    if graph_mode() == GraphMode::Replay {
        return Ok(());
    }
    // SAFETY: buffer extents validated above; pos holds axes * tokens u32 values.
    status(
        unsafe {
            velox_portable_rope_positions(
                x.as_mut_ptr(),
                pos.cast(),
                to_i32("tokens", s.tokens)?,
                to_i32("heads", s.heads)?,
                to_i32("head_dim", s.head_dim)?,
                to_i32("rotary_dim", s.rotary_dim)?,
                s.theta,
                to_i32("axes", s.axes)?,
                i32::from(s.layout == RopeLayout::Blocked),
            )
        },
        "portable rope_positions",
    )
}

extern "C" {
    fn velox_portable_top_k(
        x: *const f32,
        rows: i32,
        row_len: i64,
        limit: i32,
        k: i32,
        host_idx: *mut i32,
        host_val: *mut f32,
    ) -> i32;
}

pub fn top_k(
    x: &CudaBuffer,
    rows: usize,
    row_len: usize,
    limit: usize,
    k: usize,
) -> Result<Vec<Vec<(u32, f32)>>> {
    check::top_k(x.len(), rows, row_len, limit, k)?;
    let k = k.min(limit);
    if rows == 0 || k == 0 {
        return Ok(vec![Vec::new(); rows]);
    }
    let mut idx = vec![0i32; rows * k];
    let mut val = vec![0f32; rows * k];
    // SAFETY: x holds rows * row_len values (checked); the host arrays hold rows * k entries.
    let code = unsafe {
        velox_portable_top_k(
            x.as_ptr(),
            to_i32("rows", rows)?,
            row_len as i64,
            to_i32("limit", limit)?,
            to_i32("k", k)?,
            idx.as_mut_ptr(),
            val.as_mut_ptr(),
        )
    };
    if let Err(velox_core::Error::Unsupported(_)) = status(code, "portable top_k") {
        // Too many candidates for one merge block: select on the host.
        let mut host = vec![0f32; rows * row_len];
        // SAFETY: x holds rows * row_len values (checked).
        status(
            unsafe {
                crate::ffi::velox_cuda_dtoh(
                    host.as_mut_ptr().cast(),
                    x.as_ptr().cast(),
                    host.len() * 4,
                )
            },
            "download",
        )?;
        return Ok((0..rows)
            .map(|r| velox_core::backend::top_k_row(&host[r * row_len..r * row_len + limit], k))
            .collect());
    }
    status(code, "portable top_k")?;
    Ok((0..rows)
        .map(|r| {
            (0..k)
                .filter(|&j| idx[r * k + j] >= 0)
                .map(|j| (idx[r * k + j] as u32, val[r * k + j]))
                .collect()
        })
        .collect())
}

extern "C" {
    fn velox_portable_concat_cols(
        a: *const f32,
        b: *const f32,
        out: *mut f32,
        rows: i64,
        a_cols: i32,
        b_cols: i32,
    ) -> i32;
}

pub fn concat_cols(
    a: &CudaBuffer,
    b: &CudaBuffer,
    out: &mut CudaBuffer,
    rows: usize,
    a_cols: usize,
    b_cols: usize,
) -> Result<()> {
    check::concat_cols(a.len(), b.len(), out.len(), rows, a_cols, b_cols)?;
    // SAFETY: extents validated above.
    status(
        unsafe {
            velox_portable_concat_cols(
                a.as_ptr(),
                b.as_ptr(),
                out.as_mut_ptr(),
                rows as i64,
                to_i32("a_cols", a_cols)?,
                to_i32("b_cols", b_cols)?,
            )
        },
        "portable concat_cols",
    )
}

extern "C" {
    fn velox_portable_split_cols(
        src: *const f32,
        rows: i64,
        src_cols: i32,
        ptrs: *const *mut f32,
        offs: *const i32,
        cols: *const i32,
        n: i32,
    ) -> i32;
    fn velox_portable_swiglu(
        src: *const f32,
        rows: i64,
        src_cols: i32,
        gate: i32,
        up: i32,
        width: i32,
        out: *mut f32,
    ) -> i32;
}

pub fn split_cols(
    src: &CudaBuffer,
    rows: usize,
    src_cols: usize,
    outs: &mut [(&mut CudaBuffer, usize, usize)],
) -> Result<()> {
    let dims: Vec<(usize, usize, usize)> = outs.iter().map(|o| (o.0.len(), o.1, o.2)).collect();
    check::split_cols(src.len(), rows, src_cols, &dims)?;
    for group in outs.chunks_mut(4) {
        let ptrs: Vec<*mut f32> = group.iter_mut().map(|o| o.0.as_mut_ptr()).collect();
        let offs: Vec<i32> = group.iter().map(|o| o.1 as i32).collect();
        let cols: Vec<i32> = group.iter().map(|o| o.2 as i32).collect();
        // SAFETY: extents validated above; the kernel writes only inside each output.
        status(
            unsafe {
                velox_portable_split_cols(
                    src.as_ptr(),
                    rows as i64,
                    to_i32("src_cols", src_cols)?,
                    ptrs.as_ptr(),
                    offs.as_ptr(),
                    cols.as_ptr(),
                    group.len() as i32,
                )
            },
            "portable split_cols",
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn swiglu(
    src: &CudaBuffer,
    rows: usize,
    src_cols: usize,
    gate: usize,
    up: usize,
    width: usize,
    out: &mut CudaBuffer,
) -> Result<()> {
    check::swiglu(src.len(), rows, src_cols, gate, up, width, out.len())?;
    // SAFETY: extents validated above.
    status(
        unsafe {
            velox_portable_swiglu(
                src.as_ptr(),
                rows as i64,
                to_i32("src_cols", src_cols)?,
                gate as i32,
                up as i32,
                to_i32("width", width)?,
                out.as_mut_ptr(),
            )
        },
        "portable swiglu",
    )
}

extern "C" {
    fn velox_portable_attention_block(
        q: *const f32,
        k_ctx: *const f32,
        v_ctx: *const f32,
        k_blk: *const f32,
        v_blk: *const f32,
        out: *mut f32,
        tokens: i32,
        heads: i32,
        kv_heads: i32,
        head_dim: i32,
        pos0: i64,
        ctx_lo: i64,
        window: i32,
        ring: i32,
        params: *const i64,
    ) -> i32;
    fn velox_portable_dyn_conv(
        x: *const f32,
        base: *const f32,
        delta: *const f32,
        out: *mut f32,
        rows: i32,
        channels: i32,
        group: i32,
        side: i32,
        accumulate: i32,
        block: i32,
    ) -> i32;
    fn velox_portable_transpose_heads(
        x: *const f32,
        y: *mut f32,
        rows: i64,
        a: i32,
        b: i32,
        dim: i32,
    ) -> i32;
    fn velox_portable_copy2d(
        src: *const f32,
        src_stride: i64,
        dst: *mut f32,
        dst_stride: i64,
        rows: i64,
        cols: i64,
    ) -> i32;
    fn velox_portable_copy2d_param(
        src: *const f32,
        src_stride: i64,
        dst: *mut f32,
        dst_stride: i64,
        rows: i64,
        cols: i64,
        params: *const i64,
    ) -> i32;
    fn velox_portable_argmax_to(x: *const f32, row_len: i64, limit: i32, dst: *mut u32) -> i32;
    fn velox_portable_fetch_u32_begin(src: *const u32, slot: i32) -> i32;
    fn velox_portable_fetch_u32_wait(slot: i32, out: *mut u32) -> i32;
    fn velox_portable_copy_param(
        src: *const f32,
        dst: *mut f32,
        len: i64,
        params: *const i64,
    ) -> i32;
}

#[allow(clippy::too_many_arguments)]
pub fn attention_block(
    q: &CudaBuffer,
    k_ctx: &CudaBuffer,
    v_ctx: &CudaBuffer,
    k_blk: &CudaBuffer,
    v_blk: &CudaBuffer,
    out: &mut CudaBuffer,
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
    // Graph steps stage the positions and the context rings: a recording serves any sequence.
    let Some(params) = graph_params(&[
        s.pos0 as i64,
        s.ctx_lo as i64,
        k_ctx.as_ptr() as i64,
        v_ctx.as_ptr() as i64,
    ])?
    else {
        return Ok(());
    };
    // SAFETY: extents validated above.
    status(
        unsafe {
            velox_portable_attention_block(
                q.as_ptr(),
                k_ctx.as_ptr(),
                v_ctx.as_ptr(),
                k_blk.as_ptr(),
                v_blk.as_ptr(),
                out.as_mut_ptr(),
                to_i32("tokens", s.tokens)?,
                to_i32("heads", s.heads)?,
                to_i32("kv_heads", s.kv_heads)?,
                to_i32("head_dim", s.head_dim)?,
                s.pos0 as i64,
                s.ctx_lo as i64,
                to_i32("window", s.window)?,
                to_i32("ring", s.ring)?,
                params,
            )
        },
        "portable attention_block",
    )
}

pub fn dyn_conv(
    x: &CudaBuffer,
    base: &CudaBuffer,
    delta: &CudaBuffer,
    out: &mut CudaBuffer,
    s: DynConvShape,
) -> Result<()> {
    check::dyn_conv(x.len(), base.len(), delta.len(), out.len(), &s)?;
    // SAFETY: extents validated above.
    status(
        unsafe {
            velox_portable_dyn_conv(
                x.as_ptr(),
                base.as_ptr(),
                delta.as_ptr(),
                out.as_mut_ptr(),
                to_i32("rows", s.rows)?,
                to_i32("channels", s.channels)?,
                to_i32("group", s.group)?,
                s.side as i32,
                i32::from(s.accumulate),
                to_i32("block", s.block_rows())?,
            )
        },
        "portable dyn_conv",
    )
}

pub fn transpose_heads(
    x: &CudaBuffer,
    y: &mut CudaBuffer,
    rows: usize,
    a: usize,
    b: usize,
    dim: usize,
) -> Result<()> {
    check::transpose_heads(x.len(), y.len(), rows, a, b, dim)?;
    // SAFETY: both buffers hold rows * a * b * dim values (checked).
    status(
        unsafe {
            velox_portable_transpose_heads(
                x.as_ptr(),
                y.as_mut_ptr(),
                rows as i64,
                to_i32("a", a)?,
                to_i32("b", b)?,
                to_i32("dim", dim)?,
            )
        },
        "portable transpose_heads",
    )
}

pub fn copy2d(src: &CudaBuffer, dst: &mut CudaBuffer, s: Copy2dShape) -> Result<()> {
    check::copy2d(src.len(), dst.len(), &s)?;
    if s.rows == 0 || s.cols == 0 {
        return Ok(());
    }
    if graph_mode() != GraphMode::None {
        let Some(params) = graph_params(&[s.src_offset as i64, s.dst_offset as i64])? else {
            return Ok(());
        };
        // SAFETY: both strided ranges validated above; the kernel adds the staged offsets.
        return status(
            unsafe {
                velox_portable_copy2d_param(
                    src.as_ptr(),
                    s.src_stride as i64,
                    dst.as_mut_ptr(),
                    s.dst_stride as i64,
                    s.rows as i64,
                    s.cols as i64,
                    params,
                )
            },
            "portable copy2d",
        );
    }
    // SAFETY: both strided ranges validated above, so the offset pointers are in bounds.
    status(
        unsafe {
            velox_portable_copy2d(
                src.as_ptr().add(s.src_offset),
                s.src_stride as i64,
                dst.as_mut_ptr().add(s.dst_offset),
                s.dst_stride as i64,
                s.rows as i64,
                s.cols as i64,
            )
        },
        "portable copy2d",
    )
}

/// Per-call values of an op inside a graph step: staged into the step's region, so a replay
/// refreshes them without re-recording. `Ok(None)` means the step is a replay and the caller
/// must return without launching; outside graphs the result is a null pointer (values are
/// passed by value).
pub fn graph_params(values: &[i64]) -> Result<Option<*const i64>> {
    match graph_mode() {
        GraphMode::None => Ok(Some(std::ptr::null())),
        mode => {
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            let p = stage(&bytes)?.cast::<i64>();
            Ok((mode == GraphMode::Capture).then_some(p))
        }
    }
}

/// Context length a graph step sizes attention grids for: one bucket of 1024 keys, so steps
/// within a bucket share a recording (the model's graph keys carry the same bucket).
pub fn kv_bucket(kv_len: usize) -> usize {
    kv_len.div_ceil(GRAPH_KV_BUCKET) * GRAPH_KV_BUCKET
}

/// The keys an attention grid covers: inside a static graph step, the KV bucket its key carries
/// (bits 24..40, see `Backend::graph`; several sequences of one step share the largest),
/// otherwise the bucket of `kv_len`.
pub fn kv_grid(kv_len: usize) -> usize {
    kv_bucket(kv_len).max(crate::static_kv_bucket().unwrap_or(0))
}

/// Decodes cache rows to the f32 rows they represent; see `Backend::kv_load`.
pub fn kv_load(
    k_cache: &CudaBuffer,
    v_cache: &CudaBuffer,
    k: &mut CudaBuffer,
    v: &mut CudaBuffer,
    s: KvStoreShape,
) -> Result<()> {
    check::kv_load(k_cache.len(), v_cache.len(), k.len(), v.len(), &s)?;
    // SAFETY: buffer extents validated above.
    status(
        unsafe {
            velox_portable_kv_load(
                k_cache.as_ptr().cast(),
                v_cache.as_ptr().cast(),
                k.as_mut_ptr(),
                v.as_mut_ptr(),
                to_i32("tokens", s.tokens)?,
                to_i32("kv_heads", s.kv_heads)?,
                to_i32("head_dim", s.head_dim)?,
                s.pos0 as i64,
                kv_format_code(s.format),
                hadamard_norm(s.head_dim),
            )
        },
        "portable kv_load",
    )
}

extern "C" {
    fn velox_portable_rms_norm_groups(
        x: *const f32,
        w: *const f32,
        y: *mut f32,
        rows: i32,
        groups: i32,
        dim: i32,
        eps: f32,
    ) -> i32;
    fn velox_portable_silu(x: *mut f32, len: i64) -> i32;
    fn velox_portable_hc_mix(
        xn: *const f32,
        gate: *const f32,
        out: *mut f32,
        rows: i64,
        streams: i32,
        dim: i32,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn velox_portable_hc_combine(
        res: *mut f32,
        y: *const f32,
        inj: *const f32,
        rows: i64,
        streams: i32,
        dim: i32,
        ld: i32,
        col: i32,
    ) -> i32;
    fn velox_portable_ple_gate(
        key: *const f32,
        q: *const f32,
        value: *const f32,
        out: *mut f32,
        rows: i32,
        streams: i32,
        dim: i32,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn velox_portable_conv1d_dilated(
        x: *const f32,
        w: *const f32,
        state: *mut f32,
        out: *mut f32,
        tokens: i32,
        channels: i32,
        width: i32,
        dil: i32,
    ) -> i32;
    fn velox_portable_sigmoid_scale_rows(
        x: *mut f32,
        gate: *const f32,
        rows: i64,
        dim: i32,
        goff: i64,
        gstride: i64,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn velox_portable_moe_route(
        logits: *const f32,
        ids: *mut f32,
        wts: *mut f32,
        counts: *mut f32,
        rows: i32,
        row_len: i32,
        experts: i32,
        k: i32,
    ) -> i32;
    fn velox_portable_moe_scratch(experts: i32, pairs: i32, width: i32, hidden: i32) -> i64;
    #[allow(clippy::too_many_arguments)]
    fn velox_portable_moe_experts(
        x: *const f32,
        ids: *const f32,
        w: *const f32,
        records: *const u8,
        slots: *const u8,
        map: *const f32,
        record_bytes: i64,
        slot_bytes: i64,
        experts: i32,
        gate_fmt: i32,
        gate_off: i64,
        up_fmt: i32,
        up_off: i64,
        down_fmt: i32,
        down_off: i64,
        width: i32,
        hidden: i32,
        rows: i32,
        k: i32,
        y: *mut f32,
        scratch: *mut f32,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn velox_portable_qsa_pool(
        raw: *const f32,
        w: *const f32,
        pooled: *mut f32,
        block0: i32,
        blocks: i32,
        ratio: i32,
        dim: i32,
        rotary: i32,
        theta: f32,
        eps: f32,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn velox_portable_qsa_select(
        q: *const f32,
        pooled: *const f32,
        sel: *mut f32,
        scratch: *mut f32,
        tokens: i32,
        heads: i32,
        dim: i32,
        pos0: i32,
        ratio: i32,
        budget: i32,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn velox_portable_attention_sparse(
        q: *const f32,
        k: *const u8,
        v: *const u8,
        sel: *const f32,
        sel_len: i32,
        out: *mut f32,
        tokens: i32,
        heads: i32,
        kv_heads: i32,
        head_dim: i32,
        scale: f32,
        format: i32,
        norm: f32,
    ) -> i32;
}

/// Ops whose per-call host values (positions, block ranges) are not staged for graph replay.
fn outside_graphs(what: &str) -> Result<()> {
    if graph_mode() != GraphMode::None {
        return Err(Error::Unsupported(format!(
            "{what} inside a recorded graph step"
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn rms_norm_groups(
    x: &CudaBuffer,
    weight: &CudaBuffer,
    y: &mut CudaBuffer,
    rows: usize,
    groups: usize,
    dim: usize,
    eps: f32,
) -> Result<()> {
    check::rms_norm_groups(x.len(), weight.len(), y.len(), rows, groups, dim)?;
    to_i32("rows * groups", rows * groups)?;
    // SAFETY: buffer extents validated above.
    status(
        unsafe {
            velox_portable_rms_norm_groups(
                x.as_ptr(),
                weight.as_ptr(),
                y.as_mut_ptr(),
                to_i32("rows", rows)?,
                to_i32("groups", groups)?,
                to_i32("dim", dim)?,
                eps,
            )
        },
        "portable rms_norm_groups",
    )
}

pub fn silu(x: &mut CudaBuffer, len: usize) -> Result<()> {
    check::elementwise("silu", &[x.len()], len)?;
    // SAFETY: buffer extent validated above.
    status(
        unsafe { velox_portable_silu(x.as_mut_ptr(), len as i64) },
        "portable silu",
    )
}

pub fn hc_mix(
    xn: &CudaBuffer,
    gate: &CudaBuffer,
    out: &mut CudaBuffer,
    rows: usize,
    streams: usize,
    dim: usize,
) -> Result<()> {
    check::hc_mix(xn.len(), gate.len(), out.len(), rows, streams, dim)?;
    // SAFETY: buffer extents validated above.
    status(
        unsafe {
            velox_portable_hc_mix(
                xn.as_ptr(),
                gate.as_ptr(),
                out.as_mut_ptr(),
                rows as i64,
                to_i32("streams", streams)?,
                to_i32("dim", dim)?,
            )
        },
        "portable hc_mix",
    )
}

pub fn hc_combine(
    res: &mut CudaBuffer,
    y: &CudaBuffer,
    inject: &CudaBuffer,
    s: HcCombineShape,
) -> Result<()> {
    check::hc_combine(res.len(), y.len(), inject.len(), &s)?;
    // SAFETY: buffer extents validated above.
    status(
        unsafe {
            velox_portable_hc_combine(
                res.as_mut_ptr(),
                y.as_ptr(),
                inject.as_ptr(),
                s.rows as i64,
                to_i32("streams", s.streams)?,
                to_i32("dim", s.dim)?,
                to_i32("inject_ld", s.inject_ld)?,
                to_i32("inject_col", s.inject_col)?,
            )
        },
        "portable hc_combine",
    )
}

#[allow(clippy::too_many_arguments)]
pub fn ple_gate(
    key_n: &CudaBuffer,
    q_n: &CudaBuffer,
    value: &CudaBuffer,
    out: &mut CudaBuffer,
    rows: usize,
    streams: usize,
    dim: usize,
) -> Result<()> {
    check::ple_gate(
        key_n.len(),
        q_n.len(),
        value.len(),
        out.len(),
        rows,
        streams,
        dim,
    )?;
    to_i32("rows * streams", rows * streams)?;
    // SAFETY: buffer extents validated above.
    status(
        unsafe {
            velox_portable_ple_gate(
                key_n.as_ptr(),
                q_n.as_ptr(),
                value.as_ptr(),
                out.as_mut_ptr(),
                to_i32("rows", rows)?,
                to_i32("streams", streams)?,
                to_i32("dim", dim)?,
            )
        },
        "portable ple_gate",
    )
}

pub fn conv1d_dilated_silu(
    x: &CudaBuffer,
    weight: &CudaBuffer,
    state: &mut CudaBuffer,
    out: &mut CudaBuffer,
    s: DilatedConvShape,
) -> Result<()> {
    check::conv1d_dilated(x.len(), weight.len(), state.len(), out.len(), &s)?;
    // SAFETY: buffer extents validated above.
    status(
        unsafe {
            velox_portable_conv1d_dilated(
                x.as_ptr(),
                weight.as_ptr(),
                state.as_mut_ptr(),
                out.as_mut_ptr(),
                to_i32("tokens", s.tokens)?,
                to_i32("channels", s.channels)?,
                to_i32("width", s.width)?,
                to_i32("dilation", s.dilation)?,
            )
        },
        "portable conv1d_dilated_silu",
    )
}

pub fn sigmoid_scale_rows(x: &mut CudaBuffer, gate: &CudaBuffer, s: RowGateShape) -> Result<()> {
    check::row_gate(x.len(), gate.len(), &s)?;
    // SAFETY: buffer extents validated above.
    status(
        unsafe {
            velox_portable_sigmoid_scale_rows(
                x.as_mut_ptr(),
                gate.as_ptr(),
                s.rows as i64,
                to_i32("dim", s.dim)?,
                s.gate_offset as i64,
                s.gate_stride as i64,
            )
        },
        "portable sigmoid_scale_rows",
    )
}

pub fn moe_route(
    logits: &CudaBuffer,
    ids: &mut CudaBuffer,
    weights: &mut CudaBuffer,
    counts: Option<&mut CudaBuffer>,
    s: MoeRouteShape,
) -> Result<()> {
    check::moe_route(
        logits.len(),
        ids.len(),
        weights.len(),
        counts.as_ref().map(|c| c.len()),
        &s,
    )?;
    let counts = counts.map_or(std::ptr::null_mut(), |c| c.as_mut_ptr());
    // SAFETY: buffer extents validated above; counts is null or holds `experts` slots.
    status(
        unsafe {
            velox_portable_moe_route(
                logits.as_ptr(),
                ids.as_mut_ptr(),
                weights.as_mut_ptr(),
                counts,
                to_i32("rows", s.rows)?,
                to_i32("row_len", s.row_len)?,
                to_i32("experts", s.experts)?,
                to_i32("k", s.k)?,
            )
        },
        "portable moe_route",
    )
}

/// f32 slots of scratch [`moe_experts`] needs.
pub fn moe_scratch_len(bank: &ExpertBank<'_, CudaBuffer>, rows: usize, k: usize) -> usize {
    // SAFETY: a pure size computation.
    let n = unsafe {
        velox_portable_moe_scratch(
            bank.experts as i32,
            (rows * k) as i32,
            bank.width() as i32,
            bank.hidden() as i32,
        )
    };
    n.max(1) as usize
}

#[allow(clippy::too_many_arguments)]
pub fn moe_experts(
    x: &CudaBuffer,
    ids: &CudaBuffer,
    weights: &CudaBuffer,
    bank: &ExpertBank<'_, CudaBuffer>,
    y: &mut CudaBuffer,
    scratch: &mut CudaBuffer,
    rows: usize,
    k: usize,
) -> Result<()> {
    let bufs = (bank.records.len(), bank.slots.len(), bank.map.len());
    check::moe_experts(
        x.len(),
        ids.len(),
        weights.len(),
        bank,
        bufs,
        y.len(),
        rows,
        k,
    )?;
    to_i32("rows * k", rows * k)?;
    let need = moe_scratch_len(bank, rows, k);
    if scratch.len() < need {
        return Err(Error::Shape(format!(
            "moe_experts scratch: buffer has {} elements, needs {need}",
            scratch.len()
        )));
    }
    let off = |m: &ExpertMatrix| m.offset as i64;
    // SAFETY: extents validated above; expert ids past the bank are skipped by the kernels.
    status(
        unsafe {
            velox_portable_moe_experts(
                x.as_ptr(),
                ids.as_ptr(),
                weights.as_ptr(),
                bank.records.as_ptr().cast(),
                bank.slots.as_ptr().cast(),
                bank.map.as_ptr(),
                bank.record_bytes as i64,
                bank.slot_bytes as i64,
                to_i32("experts", bank.experts)?,
                format_code(bank.gate.format),
                off(&bank.gate),
                format_code(bank.up.format),
                off(&bank.up),
                format_code(bank.down.format),
                off(&bank.down),
                to_i32("width", bank.width())?,
                to_i32("hidden", bank.hidden())?,
                rows as i32,
                k as i32,
                y.as_mut_ptr(),
                scratch.as_mut_ptr(),
            )
        },
        "portable moe_experts",
    )
}

pub fn qsa_pool(
    raw: &CudaBuffer,
    weight: &CudaBuffer,
    pooled: &mut CudaBuffer,
    s: QsaPoolShape,
) -> Result<()> {
    check::qsa_pool(raw.len(), weight.len(), pooled.len(), &s)?;
    outside_graphs("qsa_pool")?;
    to_i32("pooled positions", (s.block0 + s.blocks) * s.ratio)?;
    // SAFETY: buffer extents validated above.
    status(
        unsafe {
            velox_portable_qsa_pool(
                raw.as_ptr(),
                weight.as_ptr(),
                pooled.as_mut_ptr(),
                s.block0 as i32,
                to_i32("blocks", s.blocks)?,
                to_i32("ratio", s.ratio)?,
                to_i32("dim", s.dim)?,
                to_i32("rotary_dim", s.rotary_dim)?,
                s.theta,
                s.eps,
            )
        },
        "portable qsa_pool",
    )
}

pub fn qsa_select(
    q: &CudaBuffer,
    pooled: &CudaBuffer,
    sel: &mut CudaBuffer,
    scratch: &mut CudaBuffer,
    s: QsaSelectShape,
) -> Result<()> {
    check::qsa_select(q.len(), pooled.len(), sel.len(), scratch.len(), &s)?;
    outside_graphs("qsa_select")?;
    to_i32("positions", s.pos0 + s.tokens)?;
    // SAFETY: buffer extents validated above.
    status(
        unsafe {
            velox_portable_qsa_select(
                q.as_ptr(),
                pooled.as_ptr(),
                sel.as_mut_ptr(),
                scratch.as_mut_ptr(),
                to_i32("tokens", s.tokens)?,
                to_i32("heads", s.heads)?,
                to_i32("dim", s.dim)?,
                s.pos0 as i32,
                to_i32("ratio", s.ratio)?,
                to_i32("budget", s.budget)?,
            )
        },
        "portable qsa_select",
    )
}

#[allow(clippy::too_many_arguments)]
pub fn attention_sparse(
    q: &CudaBuffer,
    k: &CudaBuffer,
    v: &CudaBuffer,
    sel: &CudaBuffer,
    sel_len: usize,
    out: &mut CudaBuffer,
    s: AttnShape,
) -> Result<()> {
    check::attention_sparse(q.len(), k.len(), v.len(), sel.len(), sel_len, out.len(), &s)?;
    outside_graphs("attention_sparse")?;
    to_i32("kv_len", s.kv_len())?;
    // SAFETY: buffer extents validated above; selected positions index rows below kv_len.
    status(
        unsafe {
            velox_portable_attention_sparse(
                q.as_ptr(),
                k.as_ptr().cast(),
                v.as_ptr().cast(),
                sel.as_ptr(),
                to_i32("sel_len", sel_len)?,
                out.as_mut_ptr(),
                to_i32("tokens", s.tokens)?,
                to_i32("heads", s.heads)?,
                to_i32("kv_heads", s.kv_heads)?,
                to_i32("head_dim", s.head_dim)?,
                s.scale(),
                kv_format_code(s.kv),
                hadamard_norm(s.head_dim),
            )
        },
        "portable attention_sparse",
    )
}
