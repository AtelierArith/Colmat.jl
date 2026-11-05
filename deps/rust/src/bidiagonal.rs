//! Native rectangular bidiagonalization and square bidiagonal SVD.
use crate::guarded;
use rustcall_julia_macros::julia;
mod construction;
mod workspace;
#[julia]
pub fn bidiagonalize_data(
    input: *const u8,
    m: usize,
    n: usize,
    u: *mut u8,
    v: *mut u8,
    d: *mut u8,
    e: *mut u8,
    tag: i32,
    full: bool,
    bs: usize,
) -> Result<(), String> {
    guarded(|| construction::allocate(input, m, n, u, v, d, e, tag, full, bs))
}
#[julia]
pub fn bidiagonal_svd_data(
    d: *const u8,
    e: *const u8,
    n: usize,
    s: *mut u8,
    u: *mut u8,
    v: *mut u8,
    tag: i32,
    upper: bool,
    threshold: usize,
) -> Result<(), String> {
    guarded(|| workspace::allocate(d, e, n, s, u, v, tag, upper, threshold))
}

mod compact;
