//! Shared shape validation. Every backend calls these before launching work, so a malformed call
//! fails identically on CPU, CUDA and future backends instead of reading out of bounds.

use crate::backend::{
    AttnShape, BlockAttnShape, ConvShape, Copy2dShape, DilatedConvShape, DynConvShape, ExpertBank,
    FullAttnShape, GdnShape, HcCombineShape, KvStoreShape, MoeRouteShape, QsaPoolShape,
    QsaSelectShape, RopePosShape, RopeShape, RowGateShape, WeightView,
};
use crate::error::{Error, Result};
use crate::kv::KvFormat;

pub fn attention_full(q: usize, k: usize, v: usize, out: usize, s: &FullAttnShape) -> Result<()> {
    if s.tokens == 0 || s.heads == 0 || s.head_dim == 0 {
        return Err(Error::Shape(format!(
            "attention_full dims must be non-zero: {s:?}"
        )));
    }
    let len = product(&[s.tokens, s.heads, s.head_dim])?;
    for (what, have) in [("q", q), ("k", k), ("v", v), ("out", out)] {
        need(&format!("attention_full {what}"), have, len)?;
    }
    Ok(())
}

pub fn layer_norm(
    x: usize,
    weight: usize,
    bias: usize,
    y: usize,
    rows: usize,
    dim: usize,
) -> Result<()> {
    rms_norm(x, weight, y, rows, dim)?;
    need("layer_norm bias", bias, dim)
}

pub fn add_bias(x: usize, bias: usize, rows: usize, dim: usize) -> Result<()> {
    if rows == 0 || dim == 0 {
        return Err(Error::Shape(
            "add_bias rows and dim must be non-zero".into(),
        ));
    }
    need("add_bias x", x, product(&[rows, dim])?)?;
    need("add_bias bias", bias, dim)
}

pub fn rope_positions(x: usize, positions: &[u32], s: &RopePosShape) -> Result<()> {
    let pairs = s.rotary_dim / 2;
    if s.rotary_dim == 0
        || !s.rotary_dim.is_multiple_of(2)
        || s.rotary_dim > s.head_dim
        || s.axes == 0
    {
        return Err(Error::Shape(format!(
            "rope_positions: bad rotary geometry {s:?}"
        )));
    }
    if s.layout == crate::backend::RopeLayout::Blocked && !pairs.is_multiple_of(s.axes) {
        return Err(Error::Shape(format!(
            "rope_positions: {pairs} pairs do not split into {} axes",
            s.axes
        )));
    }
    if positions.len() != s.axes * s.tokens {
        return Err(Error::Shape(format!(
            "rope_positions: {} positions for {} axes x {} tokens",
            positions.len(),
            s.axes,
            s.tokens
        )));
    }
    need(
        "rope_positions x",
        x,
        product(&[s.tokens, s.heads, s.head_dim])?,
    )
}

/// A weight view against its buffer (`buf_slots` f32 slots): layout shape rules, row range, and
/// that the buffer holds the parent's full encoded payload.
pub fn weight<B>(w: &WeightView<'_, B>, buf_slots: usize) -> Result<()> {
    w.format.check_shape(w.parent_rows, w.k)?;
    if w.rows == 0 || w.row_begin + w.rows > w.parent_rows {
        return Err(Error::Shape(format!(
            "weight rows {}+{} outside parent of {} rows",
            w.row_begin, w.rows, w.parent_rows
        )));
    }
    let bytes = w.format.payload_bytes(w.parent_rows, w.k);
    need(
        &format!("{} weight", w.format.name()),
        buf_slots,
        bytes.div_ceil(4),
    )
}

pub fn linear<B>(
    x: usize,
    w: &WeightView<'_, B>,
    w_slots: usize,
    y: usize,
    m: usize,
) -> Result<()> {
    if m == 0 {
        return Err(Error::Shape("linear m must be non-zero".into()));
    }
    weight(w, w_slots)?;
    need("linear x", x, product(&[m, w.k])?)?;
    need("linear y", y, product(&[m, w.rows])?)
}

pub fn embed<B>(
    ids: &[u32],
    table: &WeightView<'_, B>,
    table_slots: usize,
    out: usize,
) -> Result<()> {
    weight(table, table_slots)?;
    if let Some(&bad) = ids.iter().find(|&&i| i as usize >= table.rows) {
        return Err(Error::Shape(format!(
            "token id {bad} outside vocabulary of {}",
            table.rows
        )));
    }
    need("embed out", out, product(&[ids.len(), table.k])?)
}

/// `embed_from`: `count` ids at `ids[offset..]` (a buffer of `ids_slots`), the table and the
/// output extent. The ids themselves are on the device: a backend treats one past the table as
/// a row of zeros.
pub fn embed_from<B>(
    ids_slots: usize,
    offset: usize,
    count: usize,
    table: &WeightView<'_, B>,
    table_slots: usize,
    out: usize,
) -> Result<()> {
    weight(table, table_slots)?;
    need(
        "embed_from ids",
        ids_slots,
        offset.checked_add(count).unwrap_or(usize::MAX),
    )?;
    need("embed_from out", out, product(&[count, table.k])?)
}

fn need(what: &str, have: usize, required: usize) -> Result<()> {
    if have < required {
        return Err(Error::Shape(format!(
            "{what}: buffer has {have} elements, needs {required}"
        )));
    }
    Ok(())
}

fn product(dims: &[usize]) -> Result<usize> {
    dims.iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| Error::Shape(format!("dimension product overflows: {dims:?}")))
}

pub fn matmul(x: usize, w: usize, y: usize, m: usize, n: usize, k: usize) -> Result<()> {
    if m == 0 || n == 0 || k == 0 {
        return Err(Error::Shape(format!(
            "matmul dims must be non-zero: m={m} n={n} k={k}"
        )));
    }
    need("matmul x", x, product(&[m, k])?)?;
    need("matmul w", w, product(&[n, k])?)?;
    need("matmul y", y, product(&[m, n])?)
}

pub fn rms_norm(x: usize, weight: usize, y: usize, rows: usize, dim: usize) -> Result<()> {
    if rows == 0 || dim == 0 {
        return Err(Error::Shape(
            "rms_norm rows and dim must be non-zero".into(),
        ));
    }
    let total = product(&[rows, dim])?;
    need("rms_norm x", x, total)?;
    need("rms_norm weight", weight, dim)?;
    need("rms_norm y", y, total)
}

pub fn rope(x: usize, s: &RopeShape) -> Result<()> {
    if s.rotary_dim == 0 || !s.rotary_dim.is_multiple_of(2) || s.rotary_dim > s.head_dim {
        return Err(Error::Shape(format!(
            "rope rotary_dim must be even, non-zero and <= head_dim {}, got {}",
            s.head_dim, s.rotary_dim
        )));
    }
    need("rope x", x, product(&[s.tokens, s.heads, s.head_dim])?)
}

pub fn conv1d(x: usize, weight: usize, state: usize, out: usize, s: &ConvShape) -> Result<()> {
    if s.tokens == 0 || s.channels == 0 || s.width < 2 {
        return Err(Error::Shape(format!(
            "conv1d needs tokens, channels > 0 and width >= 2: {s:?}"
        )));
    }
    let rows = product(&[s.row0 + s.tokens, s.channels])?;
    need("conv1d x", x, rows)?;
    need("conv1d weight", weight, product(&[s.width, s.channels])?)?;
    need("conv1d state", state, product(&[s.width - 1, s.channels])?)?;
    need("conv1d out", out, rows)
}

pub fn gdn(
    qkv: usize,
    ab: usize,
    a_log: usize,
    dt_bias: usize,
    state: usize,
    out: usize,
    s: &GdnShape,
) -> Result<()> {
    if s.tokens == 0 || s.k_heads == 0 || s.v_heads == 0 || s.k_dim == 0 || s.v_dim == 0 {
        return Err(Error::Shape(format!(
            "gated_delta_net dims must be non-zero: {s:?}"
        )));
    }
    if !s.v_heads.is_multiple_of(s.k_heads) {
        return Err(Error::Shape(format!(
            "v_heads {} not divisible by k_heads {}",
            s.v_heads, s.k_heads
        )));
    }
    need(
        "gdn qkv",
        qkv,
        product(&[s.row0 + s.tokens, s.qkv_width()])?,
    )?;
    need("gdn ab", ab, product(&[s.row0 + s.tokens, 2 * s.v_heads])?)?;
    need("gdn a_log", a_log, s.v_heads)?;
    need("gdn dt_bias", dt_bias, s.v_heads)?;
    need("gdn state", state, product(&[s.v_heads, s.v_dim, s.k_dim])?)?;
    need(
        "gdn out",
        out,
        product(&[s.row0 + s.tokens, s.v_heads, s.v_dim])?,
    )
}

/// One layer of `Backend::gdn_replay`.
#[allow(clippy::too_many_arguments)]
pub fn gdn_replay_layer(
    qkv: usize,
    conv_out: usize,
    ab: usize,
    a_log: usize,
    dt_bias: usize,
    conv_state: usize,
    ssm: usize,
    cs: &ConvShape,
    gs: &GdnShape,
) -> Result<()> {
    if cs.tokens != gs.tokens || cs.row0 != gs.row0 || cs.channels != gs.qkv_width() {
        return Err(Error::Shape(format!(
            "gdn_replay shapes disagree: {cs:?} {gs:?}"
        )));
    }
    conv1d(qkv, cs.width * cs.channels, conv_state, conv_out, cs)?;
    gdn(
        conv_out,
        ab,
        a_log,
        dt_bias,
        ssm,
        (gs.row0 + gs.tokens) * gs.v_heads * gs.v_dim,
        gs,
    )
}

pub fn elementwise(what: &str, lens: &[usize], len: usize) -> Result<()> {
    for &have in lens {
        need(what, have, len)?;
    }
    Ok(())
}

pub fn copy(
    src: usize,
    src_offset: usize,
    dst: usize,
    dst_offset: usize,
    len: usize,
) -> Result<()> {
    let src_end = src_offset
        .checked_add(len)
        .ok_or_else(|| Error::Shape("copy range overflows".into()))?;
    need("copy src", src, src_end)?;
    let end = dst_offset
        .checked_add(len)
        .ok_or_else(|| Error::Shape("copy range overflows".into()))?;
    need("copy dst", dst, end)
}

pub fn attention(q: usize, k: usize, v: usize, out: usize, s: &AttnShape) -> Result<()> {
    if s.tokens == 0 || s.heads == 0 || s.kv_heads == 0 || s.head_dim == 0 {
        return Err(Error::Shape(format!(
            "attention dims must be non-zero: {s:?}"
        )));
    }
    if !s.heads.is_multiple_of(s.kv_heads) {
        return Err(Error::Shape(format!(
            "heads {} not divisible by kv_heads {}",
            s.heads, s.kv_heads
        )));
    }
    s.kv.check_head_dim(s.head_dim)?;
    let q_len = product(&[s.tokens, s.heads, s.head_dim])?;
    let kv = cache_slots(s.kv, s.kv_len(), s.kv_heads, s.head_dim)?;
    need("attention q", q, q_len)?;
    need("attention k", k, kv)?;
    need("attention v", v, kv)?;
    need("attention out", out, q_len)
}

fn cache_slots(f: KvFormat, positions: usize, kv_heads: usize, head_dim: usize) -> Result<usize> {
    f.cache_slots(positions, kv_heads, head_dim)
        .ok_or_else(|| Error::Shape(format!("KV cache size overflows: {positions} positions")))
}

pub fn kv_store(
    k: usize,
    v: usize,
    k_cache: usize,
    v_cache: usize,
    s: &KvStoreShape,
) -> Result<()> {
    if s.tokens == 0 || s.kv_heads == 0 || s.head_dim == 0 {
        return Err(Error::Shape(format!(
            "kv_store dims must be non-zero: {s:?}"
        )));
    }
    s.format.check_head_dim(s.head_dim)?;
    let rows = product(&[s.tokens, s.kv_heads, s.head_dim])?;
    need("kv_store k", k, rows)?;
    need("kv_store v", v, rows)?;
    let end = s
        .pos0
        .checked_add(s.tokens)
        .ok_or_else(|| Error::Shape("kv_store range overflows".into()))?;
    let cache = cache_slots(s.format, end, s.kv_heads, s.head_dim)?;
    need("kv_store k_cache", k_cache, cache)?;
    need("kv_store v_cache", v_cache, cache)
}

/// `state_pack`: `n` f32 values (whole rows) into a packed state of `fmt`.
pub fn state_pack(src: usize, dst: usize, n: usize, fmt: crate::StateFormat) -> Result<()> {
    fmt.check_len(n)?;
    need("state_pack src", src, n)?;
    need("state_pack dst", dst, fmt.slots(n))
}

/// `state_unpack` reads what `state_pack` writes.
pub fn state_unpack(src: usize, dst: usize, n: usize, fmt: crate::StateFormat) -> Result<()> {
    state_pack(dst, src, n, fmt)
}

/// `kv_load` reads what `kv_store` writes: the same extents.
pub fn kv_load(k_cache: usize, v_cache: usize, k: usize, v: usize, s: &KvStoreShape) -> Result<()> {
    kv_store(k, v, k_cache, v_cache, s)
}

pub fn copy2d(src: usize, dst: usize, s: &Copy2dShape) -> Result<()> {
    if s.rows == 0 || s.cols == 0 {
        return Ok(());
    }
    if s.cols > s.src_stride || s.cols > s.dst_stride {
        return Err(Error::Shape(format!("copy2d: rows overlap: {s:?}")));
    }
    let end = |off: usize, stride: usize| -> Result<usize> {
        product(&[s.rows - 1, stride])?
            .checked_add(off + s.cols)
            .ok_or_else(|| Error::Shape("copy2d range overflows".into()))
    };
    need("copy2d src", src, end(s.src_offset, s.src_stride)?)?;
    need("copy2d dst", dst, end(s.dst_offset, s.dst_stride)?)
}

/// `transpose_heads`: `rows` rows of `a * b` heads of `dim` values in both buffers.
pub fn transpose_heads(
    x: usize,
    y: usize,
    rows: usize,
    a: usize,
    b: usize,
    dim: usize,
) -> Result<()> {
    let n = product(&[rows, a, b, dim])?;
    need("transpose_heads x", x, n)?;
    need("transpose_heads y", y, n)
}

pub fn dyn_conv(x: usize, base: usize, delta: usize, out: usize, s: &DynConvShape) -> Result<()> {
    if s.block != 0 && s.rows % s.block != 0 {
        return Err(Error::Shape(format!(
            "dyn_conv: {} rows in blocks of {}",
            s.rows, s.block
        )));
    }
    if s.rows == 0 || s.group == 0 || !s.channels.is_multiple_of(s.group) || s.side > 1 {
        return Err(Error::Shape(format!("dyn_conv: bad shape {s:?}")));
    }
    let n = product(&[s.rows, s.channels])?;
    need("dyn_conv x", x, n)?;
    need("dyn_conv out", out, n)?;
    need("dyn_conv base", base, 4 * s.channels)?;
    need(
        "dyn_conv delta",
        delta,
        product(&[s.rows, 4, s.channels / s.group])?,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn attention_block(
    q: usize,
    k_ctx: usize,
    v_ctx: usize,
    k_blk: usize,
    v_blk: usize,
    out: usize,
    s: &BlockAttnShape,
) -> Result<()> {
    if s.tokens == 0 || s.heads == 0 || s.kv_heads == 0 || s.head_dim == 0 || s.ring == 0 {
        return Err(Error::Shape(format!(
            "attention_block dims must be non-zero: {s:?}"
        )));
    }
    if !s.heads.is_multiple_of(s.kv_heads)
        || s.tokens > s.window
        || s.ctx_lo > s.pos0
        || s.pos0 - s.ctx_lo > s.ring
    {
        return Err(Error::Shape(format!("attention_block: bad shape {s:?}")));
    }
    let q_len = product(&[s.tokens, s.heads, s.head_dim])?;
    let ctx = product(&[s.ring, s.kv_heads, s.head_dim])?;
    let blk = product(&[s.tokens, s.kv_heads, s.head_dim])?;
    need("attention_block q", q, q_len)?;
    need("attention_block k_ctx", k_ctx, ctx)?;
    need("attention_block v_ctx", v_ctx, ctx)?;
    need("attention_block k_blk", k_blk, blk)?;
    need("attention_block v_blk", v_blk, blk)?;
    need("attention_block out", out, q_len)
}

/// Converts a dimension for a C ABI that takes `int`.
pub fn to_i32(what: &str, v: usize) -> Result<i32> {
    i32::try_from(v).map_err(|_| Error::Shape(format!("{what}={v} exceeds i32 range")))
}

pub fn concat_cols(
    a: usize,
    b: usize,
    out: usize,
    rows: usize,
    a_cols: usize,
    b_cols: usize,
) -> Result<()> {
    need("concat_cols a", a, rows * a_cols)?;
    need("concat_cols b", b, rows * b_cols)?;
    need("concat_cols out", out, rows * (a_cols + b_cols))
}

pub fn split_cols(
    src: usize,
    rows: usize,
    src_cols: usize,
    outs: &[(usize, usize, usize)],
) -> Result<()> {
    need("split_cols src", src, rows * src_cols)?;
    for &(len, off, cols) in outs {
        if off + cols > src_cols {
            return Err(Error::Shape(format!(
                "split_cols: columns {off}+{cols} past {src_cols}"
            )));
        }
        need("split_cols out", len, rows * cols)?;
    }
    Ok(())
}

pub fn swiglu(
    src: usize,
    rows: usize,
    src_cols: usize,
    gate: usize,
    up: usize,
    width: usize,
    out: usize,
) -> Result<()> {
    if gate + width > src_cols || up + width > src_cols {
        return Err(Error::Shape(format!(
            "swiglu: gate {gate} / up {up} + {width} past {src_cols}"
        )));
    }
    need("swiglu src", src, rows * src_cols)?;
    need("swiglu out", out, rows * width)
}

/// Largest k supported by `top_k` (sampling needs a few dozen candidates at most).
pub const TOP_K_MAX: usize = 256;

pub fn rms_norm_groups(
    x: usize,
    weight: usize,
    y: usize,
    rows: usize,
    groups: usize,
    dim: usize,
) -> Result<()> {
    if rows == 0 || groups == 0 || dim == 0 {
        return Err(Error::Shape(format!(
            "rms_norm_groups dims must be non-zero: rows={rows} groups={groups} dim={dim}"
        )));
    }
    let total = product(&[rows, groups, dim])?;
    need("rms_norm_groups x", x, total)?;
    need("rms_norm_groups weight", weight, product(&[groups, dim])?)?;
    need("rms_norm_groups y", y, total)
}

/// Rows of `streams` streams of `dim`: the shared extent rules of the hyper-connection ops.
fn streams(what: &str, rows: usize, streams: usize, dim: usize) -> Result<usize> {
    if rows == 0 || streams == 0 || dim == 0 {
        return Err(Error::Shape(format!(
            "{what} dims must be non-zero: rows={rows} streams={streams} dim={dim}"
        )));
    }
    product(&[rows, streams, dim])
}

pub fn hc_mix(xn: usize, gate: usize, out: usize, rows: usize, s: usize, dim: usize) -> Result<()> {
    let wide = streams("hc_mix", rows, s, dim)?;
    need("hc_mix xn", xn, wide)?;
    need("hc_mix gate", gate, wide)?;
    need("hc_mix out", out, product(&[rows, dim])?)
}

pub fn hc_combine(res: usize, y: usize, inject: usize, s: &HcCombineShape) -> Result<()> {
    let wide = streams("hc_combine", s.rows, s.streams, s.dim)?;
    if s.inject_col + s.streams > s.inject_ld {
        return Err(Error::Shape(format!(
            "hc_combine: inject columns {}+{} past its row of {}",
            s.inject_col, s.streams, s.inject_ld
        )));
    }
    need("hc_combine res", res, wide)?;
    need("hc_combine y", y, product(&[s.rows, s.dim])?)?;
    need(
        "hc_combine inject",
        inject,
        product(&[s.rows - 1, s.inject_ld])? + s.inject_col + s.streams,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn ple_gate(
    key_n: usize,
    q_n: usize,
    value: usize,
    out: usize,
    rows: usize,
    s: usize,
    dim: usize,
) -> Result<()> {
    let wide = streams("ple_gate", rows, s, dim)?;
    need("ple_gate key_n", key_n, wide)?;
    need("ple_gate q_n", q_n, wide)?;
    need("ple_gate value", value, product(&[rows, dim])?)?;
    need("ple_gate out", out, wide)
}

pub fn conv1d_dilated(
    x: usize,
    weight: usize,
    state: usize,
    out: usize,
    s: &DilatedConvShape,
) -> Result<()> {
    if s.tokens == 0 || s.channels == 0 || s.width < 2 || s.dilation == 0 {
        return Err(Error::Shape(format!(
            "conv1d_dilated needs tokens, channels, dilation > 0 and width >= 2: {s:?}"
        )));
    }
    let rows = product(&[s.tokens, s.channels])?;
    need("conv1d_dilated x", x, rows)?;
    need(
        "conv1d_dilated weight",
        weight,
        product(&[s.width, s.channels])?,
    )?;
    need(
        "conv1d_dilated state",
        state,
        product(&[s.state_rows(), s.channels])?,
    )?;
    need("conv1d_dilated out", out, rows)
}

pub fn row_gate(x: usize, gate: usize, s: &RowGateShape) -> Result<()> {
    if s.rows == 0 || s.dim == 0 {
        return Err(Error::Shape(format!(
            "sigmoid_scale_rows dims must be non-zero: {s:?}"
        )));
    }
    need("sigmoid_scale_rows x", x, product(&[s.rows, s.dim])?)?;
    need(
        "sigmoid_scale_rows gate",
        gate,
        product(&[s.rows - 1, s.gate_stride])? + s.gate_offset + 1,
    )
}

/// Largest top-k `moe_route` serves (kernels keep the selection in registers).
pub const MOE_K_MAX: usize = 32;

pub fn moe_route(
    logits: usize,
    ids: usize,
    weights: usize,
    counts: Option<usize>,
    s: &MoeRouteShape,
) -> Result<()> {
    if s.rows == 0 || s.k == 0 || s.k > MOE_K_MAX || s.k > s.experts || s.experts > s.row_len {
        return Err(Error::Shape(format!(
            "moe_route needs 0 < k <= min(experts, {MOE_K_MAX}) and experts <= row_len: {s:?}"
        )));
    }
    need("moe_route logits", logits, product(&[s.rows, s.row_len])?)?;
    need("moe_route ids", ids, product(&[s.rows, s.k])?)?;
    need("moe_route weights", weights, product(&[s.rows, s.k])?)?;
    if let Some(c) = counts {
        need("moe_route counts", c, s.experts)?;
    }
    Ok(())
}

/// An expert bank's geometry against its buffers (`records`, `slots`, `map` in f32 slots).
pub fn expert_bank<B>(
    bank: &ExpertBank<'_, B>,
    records: usize,
    slots: usize,
    map: usize,
) -> Result<()> {
    let (w, h) = (bank.width(), bank.hidden());
    let parts = [
        ("gate", &bank.gate, w, h),
        ("up", &bank.up, w, h),
        ("down", &bank.down, h, w),
    ];
    for (name, m, rows, k) in parts {
        if (m.rows, m.k) != (rows, k) {
            return Err(Error::Shape(format!(
                "expert {name} is [{},{}], expected [{rows},{k}]",
                m.rows, m.k
            )));
        }
        m.format.check_shape(m.rows, m.k)?;
        let end = m.offset + m.format.payload_bytes(m.rows, m.k);
        if end > bank.record_bytes {
            return Err(Error::Shape(format!(
                "expert {name} ends at byte {end}, past the {}-byte record",
                bank.record_bytes
            )));
        }
    }
    if bank.experts == 0
        || !bank.record_bytes.is_multiple_of(16)
        || !bank.slot_bytes.is_multiple_of(16)
        || bank.slot_bytes < bank.record_bytes
    {
        return Err(Error::Shape(format!(
            "expert bank needs experts > 0, 16-byte records and slots at least a record: \
             {} x {} bytes, slots of {}",
            bank.experts, bank.record_bytes, bank.slot_bytes
        )));
    }
    need(
        "expert records",
        records,
        product(&[bank.experts, bank.record_bytes])? / 4,
    )?;
    need(
        "expert slots",
        slots,
        product(&[bank.slot_count, bank.slot_bytes])? / 4,
    )?;
    need("expert map", map, bank.experts)
}

#[allow(clippy::too_many_arguments)]
pub fn moe_experts<B>(
    x: usize,
    ids: usize,
    weights: usize,
    bank: &ExpertBank<'_, B>,
    bank_bufs: (usize, usize, usize),
    y: usize,
    rows: usize,
    k: usize,
) -> Result<()> {
    if rows == 0 || k == 0 || k > MOE_K_MAX {
        return Err(Error::Shape(format!(
            "moe_experts needs rows > 0 and 0 < k <= {MOE_K_MAX}: rows={rows} k={k}"
        )));
    }
    expert_bank(bank, bank_bufs.0, bank_bufs.1, bank_bufs.2)?;
    let wide = product(&[rows, bank.hidden()])?;
    need("moe_experts x", x, wide)?;
    need("moe_experts ids", ids, product(&[rows, k])?)?;
    need("moe_experts weights", weights, product(&[rows, k])?)?;
    need("moe_experts y", y, wide)
}

pub fn qsa_pool(raw: usize, weight: usize, pooled: usize, s: &QsaPoolShape) -> Result<()> {
    if s.blocks == 0 || s.ratio == 0 || s.dim == 0 {
        return Err(Error::Shape(format!(
            "qsa_pool dims must be non-zero: {s:?}"
        )));
    }
    if s.rotary_dim > s.dim || !s.rotary_dim.is_multiple_of(2) {
        return Err(Error::Shape(format!(
            "qsa_pool rotary_dim must be even and <= dim: {s:?}"
        )));
    }
    let end = s.block0 + s.blocks;
    need("qsa_pool raw", raw, product(&[end, s.ratio, s.dim])?)?;
    need("qsa_pool weight", weight, s.dim)?;
    need("qsa_pool pooled", pooled, product(&[end, s.dim])?)
}

pub fn qsa_select(
    q: usize,
    pooled: usize,
    sel: usize,
    scratch: usize,
    s: &QsaSelectShape,
) -> Result<()> {
    if s.tokens == 0 || s.heads == 0 || s.dim == 0 || s.ratio == 0 {
        return Err(Error::Shape(format!(
            "qsa_select dims must be non-zero: {s:?}"
        )));
    }
    if s.budget == 0 || !s.budget.is_multiple_of(s.ratio) {
        return Err(Error::Shape(format!(
            "qsa_select budget must be a non-zero multiple of the ratio: {s:?}"
        )));
    }
    let blocks = s.complete_blocks(s.tokens - 1);
    need("qsa_select q", q, product(&[s.tokens, s.heads, s.dim])?)?;
    need("qsa_select pooled", pooled, product(&[blocks, s.dim])?)?;
    need("qsa_select sel", sel, product(&[s.tokens, s.sel_len()])?)?;
    need("qsa_select scratch", scratch, product(&[s.tokens, blocks])?)
}

#[allow(clippy::too_many_arguments)]
pub fn attention_sparse(
    q: usize,
    k: usize,
    v: usize,
    sel: usize,
    sel_len: usize,
    out: usize,
    s: &AttnShape,
) -> Result<()> {
    attention(q, k, v, out, s)?;
    if sel_len == 0 {
        return Err(Error::Shape(
            "attention_sparse sel_len must be non-zero".into(),
        ));
    }
    need("attention_sparse sel", sel, product(&[s.tokens, sel_len])?)
}

pub fn top_k(x: usize, rows: usize, row_len: usize, limit: usize, k: usize) -> Result<()> {
    if limit > row_len || k == 0 || k > TOP_K_MAX {
        return Err(Error::Shape(format!(
            "top_k: k {k} (max {TOP_K_MAX}), limit {limit}, row_len {row_len}"
        )));
    }
    need("top_k input", x, rows * row_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_rejects_short_buffers() {
        assert!(matmul(6, 12, 8, 2, 4, 3).is_ok());
        assert!(matmul(5, 12, 8, 2, 4, 3).is_err());
        assert!(matmul(6, 12, 8, 0, 4, 3).is_err());
    }

    #[test]
    fn attention_requires_grouping() {
        let s = AttnShape {
            tokens: 1,
            heads: 6,
            kv_heads: 4,
            head_dim: 8,
            pos0: 0,
            kv: KvFormat::F32,
        };
        assert!(attention(48, 32, 32, 48, &s).is_err());
    }

    #[test]
    fn quantized_caches_are_sized_by_format() {
        let s = AttnShape {
            tokens: 2,
            heads: 4,
            kv_heads: 2,
            head_dim: 256,
            pos0: 3,
            kv: KvFormat::Nvfp4,
        };
        // 5 positions x 2 heads x 36 slots.
        assert!(attention(2048, 360, 360, 2048, &s).is_ok());
        assert!(attention(2048, 359, 360, 2048, &s).is_err());
        let odd = AttnShape { head_dim: 96, ..s };
        assert!(attention(1 << 20, 1 << 20, 1 << 20, 1 << 20, &odd).is_err());
        let st = KvStoreShape {
            tokens: 2,
            kv_heads: 2,
            head_dim: 64,
            pos0: 3,
            format: KvFormat::Fp8,
        };
        // 5 positions x 2 heads x 20 slots (68 bytes padded to 80).
        assert!(kv_store(256, 256, 200, 200, &st).is_ok());
        assert!(kv_store(256, 256, 199, 200, &st).is_err());
    }

    #[test]
    fn copy_checks_destination_end() {
        assert!(copy(4, 0, 10, 6, 4).is_ok());
        assert!(copy(4, 0, 10, 7, 4).is_err());
        assert!(copy(4, 0, 10, usize::MAX, 4).is_err());
        assert!(copy(6, 2, 10, 0, 4).is_ok());
        assert!(copy(6, 3, 10, 0, 4).is_err());
    }
}
