//! Shared shape validation. Every backend calls these before launching work, so a malformed call
//! fails identically on CPU, CUDA and future backends instead of reading out of bounds.

use crate::backend::{AttnShape, RopeShape};
use crate::error::{Error, Result};

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
    if s.head_dim == 0 || !s.head_dim.is_multiple_of(2) {
        return Err(Error::Shape(format!(
            "rope head_dim must be even and non-zero, got {}",
            s.head_dim
        )));
    }
    need("rope x", x, product(&[s.tokens, s.heads, s.head_dim])?)
}

pub fn elementwise(what: &str, lens: &[usize], len: usize) -> Result<()> {
    for &have in lens {
        need(what, have, len)?;
    }
    Ok(())
}

pub fn copy(src: usize, dst: usize, dst_offset: usize, len: usize) -> Result<()> {
    need("copy src", src, len)?;
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
        assert!(copy(4, 10, 6, 4).is_ok());
        assert!(copy(4, 10, 7, 4).is_err());
        assert!(copy(4, 10, usize::MAX, 4).is_err());
    }
}
