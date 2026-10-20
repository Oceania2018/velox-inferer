//! Shared shape validation. Every backend calls these before launching work, so a malformed call
//! fails identically on CPU, CUDA and future backends instead of reading out of bounds.

use crate::backend::{
    AttnShape, ConvShape, FullAttnShape, GdnShape, RopePosShape, RopeShape, WeightView,
};
use crate::error::{Error, Result};

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
    let rows = product(&[s.tokens, s.channels])?;
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
    need("gdn qkv", qkv, product(&[s.tokens, s.qkv_width()])?)?;
    need("gdn ab", ab, product(&[s.tokens, 2 * s.v_heads])?)?;
    need("gdn a_log", a_log, s.v_heads)?;
    need("gdn dt_bias", dt_bias, s.v_heads)?;
    need("gdn state", state, product(&[s.v_heads, s.v_dim, s.k_dim])?)?;
    need("gdn out", out, product(&[s.tokens, s.v_heads, s.v_dim])?)
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
    let q_len = product(&[s.tokens, s.heads, s.head_dim])?;
    let kv = product(&[s.kv_len(), s.kv_heads, s.head_dim])?;
    need("attention q", q, q_len)?;
    need("attention k", k, kv)?;
    need("attention v", v, kv)?;
    need("attention out", out, q_len)
}

/// Converts a dimension for a C ABI that takes `int`.
pub fn to_i32(what: &str, v: usize) -> Result<i32> {
    i32::try_from(v).map_err(|_| Error::Shape(format!("{what}={v} exceeds i32 range")))
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
        };
        assert!(attention(48, 32, 32, 48, &s).is_err());
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
