//! Safe wrappers over the portable kernels. Variants call these as the fallback for shapes their
//! tuned kernels do not cover. Each wrapper re-validates, so it is safe to call directly.

use velox_core::backend::{
    AttnShape, ConvShape, FullAttnShape, GdnShape, GeluKind, NormKind, RopeLayout, RopePosShape,
    RopeShape, WeightView,
};
use velox_core::check::{self, to_i32};
use velox_core::{Result, WeightFormat};

use crate::{dtod, status, upload_raw, CudaBuffer};

extern "C" {
    fn velox_portable_matmul(
        x: *const f32,
        w: *const f32,
        y: *mut f32,
        m: i32,
        n: i32,
        k: i32,
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
    fn velox_portable_rope(
        x: *mut f32,
        tokens: i32,
        heads: i32,
        head_dim: i32,
        rotary_dim: i32,
        pos0: i32,
        theta: f32,
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
    fn velox_portable_conv1d_silu(
        x: *const f32,
        w: *const f32,
        state: *mut f32,
        out: *mut f32,
        tokens: i32,
        channels: i32,
        width: i32,
    ) -> i32;
    fn velox_portable_gated_delta_net(
        qkv: *const f32,
        ab: *const f32,
        a_log: *const f32,
        dt_bias: *const f32,
        state: *mut f32,
        out: *mut f32,
        tokens: i32,
        k_heads: i32,
        v_heads: i32,
        k_dim: i32,
        v_dim: i32,
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
    let dev_ids = upload_raw(&id_bytes)?;
    // SAFETY: ids are in range (checked) and out holds ids.len() rows.
    status(
        unsafe {
            velox_portable_embed(
                dev_ids.as_ptr().cast(),
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
    // SAFETY: buffer extent validated above.
    status(
        unsafe { velox_portable_rope(x.as_mut_ptr(), t, h, d, r, s.pos0 as i32, s.theta) },
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
            )
        },
        "portable attention",
    )
}

pub fn conv1d_silu(
    x: &CudaBuffer,
    weight: &CudaBuffer,
    state: &mut CudaBuffer,
    out: &mut CudaBuffer,
    s: ConvShape,
) -> Result<()> {
    check::conv1d(x.len(), weight.len(), state.len(), out.len(), &s)?;
    // SAFETY: buffer extents validated above.
    status(
        unsafe {
            velox_portable_conv1d_silu(
                x.as_ptr(),
                weight.as_ptr(),
                state.as_mut_ptr(),
                out.as_mut_ptr(),
                to_i32("tokens", s.tokens)?,
                to_i32("channels", s.channels)?,
                to_i32("width", s.width)?,
            )
        },
        "portable conv1d_silu",
    )
}

#[allow(clippy::too_many_arguments)]
pub fn gated_delta_net(
    qkv: &CudaBuffer,
    ab: &CudaBuffer,
    a_log: &CudaBuffer,
    dt_bias: &CudaBuffer,
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
    // SAFETY: buffer extents validated above.
    status(
        unsafe {
            velox_portable_gated_delta_net(
                qkv.as_ptr(),
                ab.as_ptr(),
                a_log.as_ptr(),
                dt_bias.as_ptr(),
                state.as_mut_ptr(),
                out.as_mut_ptr(),
                to_i32("tokens", s.tokens)?,
                to_i32("k_heads", s.k_heads)?,
                to_i32("v_heads", s.v_heads)?,
                to_i32("k_dim", s.k_dim)?,
                to_i32("v_dim", s.v_dim)?,
            )
        },
        "portable gated_delta_net",
    )
}

extern "C" {
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
    let pos = upload_raw(&bytes)?;
    // SAFETY: buffer extents validated above; pos holds axes * tokens u32 values.
    status(
        unsafe {
            velox_portable_rope_positions(
                x.as_mut_ptr(),
                pos.as_ptr().cast(),
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
