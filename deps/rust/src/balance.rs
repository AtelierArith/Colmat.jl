//! Balancing and right eigenvector back-transforms on disjoint Julia buffers.
use crate::guarded;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd;
use faer::traits::ComplexField;
use faer::{MatMut, Par};
use rustcall_julia_macros::julia;
unsafe fn run<T: ComplexField>(
    data: *mut u8,
    n: usize,
    perm: *mut usize,
    scale: *mut u8,
    active: *mut usize,
) -> Result<(), String> {
    let a = MatMut::<T>::from_raw_parts_mut(data.cast(), n, n, 1, n as isize);
    let p = std::slice::from_raw_parts_mut(perm, n);
    let s = std::slice::from_raw_parts_mut(scale.cast::<T::Real>(), n);
    let mut mem = MemBuffer::try_new(evd::balance_scratch()).map_err(|e| format!("{e:?}"))?;
    let range = evd::balance_in_place(a, p, s, Par::Seq, MemStack::new(&mut mem))
        .map_err(|e| format!("{e:?}"))?;
    active.write(range.start);
    active.add(1).write(range.end);
    Ok(())
}
#[julia]
pub fn balance_data(
    data: *mut u8,
    n: usize,
    perm: *mut usize,
    scale: *mut u8,
    active: *mut usize,
    tag: i32,
) -> Result<(), String> {
    guarded(|| unsafe {
        match tag {
            0 => run::<f32>(data, n, perm, scale, active),
            1 => run::<f64>(data, n, perm, scale, active),
            2 => run::<faer::c32>(data, n, perm, scale, active),
            3 => run::<faer::c64>(data, n, perm, scale, active),
            _ => Err("unsupported balance scalar".into()),
        }
    })
}
unsafe fn vectors<T: ComplexField>(
    data: *mut u8,
    n: usize,
    k: usize,
    perm: *const usize,
    scale: *const u8,
) -> Result<(), String> {
    let a = MatMut::<T>::from_raw_parts_mut(data.cast(), n, k, 1, n as isize);
    let p = std::slice::from_raw_parts(perm, n);
    let s = std::slice::from_raw_parts(scale.cast::<T::Real>(), n);
    let mut mem = MemBuffer::try_new(evd::balance_eigenvectors_scratch::<T>(n))
        .map_err(|e| format!("{e:?}"))?;
    evd::balance_eigenvectors_in_place(a, p, s, Par::Seq, MemStack::new(&mut mem));
    Ok(())
}
#[julia]
pub fn balance_vectors_data(
    data: *mut u8,
    n: usize,
    k: usize,
    perm: *const usize,
    scale: *const u8,
    tag: i32,
) -> Result<(), String> {
    guarded(|| unsafe {
        match tag {
            0 => vectors::<f32>(data, n, k, perm, scale),
            1 => vectors::<f64>(data, n, k, perm, scale),
            2 => vectors::<faer::c32>(data, n, k, perm, scale),
            3 => vectors::<faer::c64>(data, n, k, perm, scale),
            _ => Err("unsupported balance scalar".into()),
        }
    })
}
