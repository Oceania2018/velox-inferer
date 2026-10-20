//! Safe wrappers over the portable kernels. Variants call these as the fallback for shapes their
//! tuned kernels do not cover. Each wrapper re-validates, so it is safe to call directly.

use velox_core::backend::{AttnShape, RopeShape};
use velox_core::check::{self, to_i32};
use velox_core::Result;

use crate::{dtod, status, CudaBuffer};

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
    ) -> i32;
    fn velox_portable_rope(
        x: *mut f32,
        tokens: i32,
        heads: i32,
        head_dim: i32,
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

pub fn rms_norm(
    x: &CudaBuffer,
    weight: &CudaBuffer,
    y: &mut CudaBuffer,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<()> {
    check::rms_norm(x.len(), weight.len(), y.len(), rows, dim)?;
    let (rows, dim) = (to_i32("rows", rows)?, to_i32("dim", dim)?);
    // SAFETY: buffer extents validated above.
    status(
        unsafe {
            velox_portable_rms_norm(x.as_ptr(), weight.as_ptr(), y.as_mut_ptr(), rows, dim, eps)
        },
        "portable rms_norm",
    )
}

pub fn rope(x: &mut CudaBuffer, s: RopeShape) -> Result<()> {
    check::rope(x.len(), &s)?;
    let (t, h, d) = (
        to_i32("tokens", s.tokens)?,
        to_i32("heads", s.heads)?,
        to_i32("head_dim", s.head_dim)?,
    );
    // The kernel forms positions as pos0 + t in i32.
    to_i32("pos0 + tokens", s.pos0 + s.tokens)?;
    // SAFETY: buffer extent validated above.
    status(
        unsafe { velox_portable_rope(x.as_mut_ptr(), t, h, d, s.pos0 as i32, s.theta) },
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

pub fn add(acc: &mut CudaBuffer, x: &CudaBuffer, len: usize) -> Result<()> {
    check::elementwise("add", &[acc.len(), x.len()], len)?;
    // SAFETY: buffer extents validated above.
    status(
        unsafe { velox_portable_add(acc.as_mut_ptr(), x.as_ptr(), len as i64) },
        "portable add",
    )
}

pub fn copy(src: &CudaBuffer, dst: &mut CudaBuffer, dst_offset: usize, len: usize) -> Result<()> {
    check::copy(src.len(), dst.len(), dst_offset, len)?;
    if len == 0 {
        return Ok(());
    }
    // SAFETY: dst_offset + len <= dst.len() validated above, so the offset pointer is in bounds.
    dtod(
        unsafe { dst.as_mut_ptr().add(dst_offset) },
        src.as_ptr(),
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
