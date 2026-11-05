//! Self-adjoint tridiagonalization and direct band eigendecomposition.
use crate::guarded;
use rustcall_julia_macros::julia;
mod construction;
#[julia]
pub fn tridiagonalize_data(
    input: *const u8,
    n: usize,
    q: *mut u8,
    d: *mut u8,
    e: *mut u8,
    tag: i32,
    bs: usize,
) -> Result<(), String> {
    guarded(|| construction::allocate(input, n, q, d, e, tag, bs))
}
mod workspace;
#[julia]
pub fn tridiagonal_eigen_data(
    d: *const u8,
    e: *const u8,
    n: usize,
    values: *mut u8,
    vectors: *mut u8,
    tag: i32,
    compute: bool,
    threshold: usize,
) -> Result<(), String> {
    guarded(|| workspace::allocate(d, e, n, values, vectors, tag, compute, threshold))
}

mod compact;
