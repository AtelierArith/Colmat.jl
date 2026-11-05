//! Self-adjoint tridiagonalization and direct band eigendecomposition.
use crate::{guarded, typed::finite};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd;
use faer::traits::{math_utils::*, ComplexField};
use faer::{Mat, MatRef, Par};
use rustcall_julia_macros::julia;
unsafe fn reduction<T: ComplexField>(
    input: *const u8,
    n: usize,
    qout: *mut u8,
    dout: *mut u8,
    eout: *mut u8,
    blocksize: usize,
) -> Result<(), String> {
    let a = MatRef::<T>::from_raw_parts(input.cast(), n, n, 1, n as isize);
    finite(a)?;
    let mut work = a.to_owned();
    let mut q = Mat::<T>::identity(n, n);
    if n > 1 {
        let scaling = crate::schur::scale(&mut work);
        let bs = if blocksize == 0 {
            faer::linalg::qr::no_pivoting::factor::recommended_block_size::<T>(n - 1, n - 1)
                .min(n - 1)
                .max(1)
        } else {
            blocksize
        };
        let mut factor = Mat::<T>::zeros(bs, n - 1);
        let req =
            evd::tridiag::tridiag_in_place_scratch::<T>(n, Par::Seq.shape(), Default::default());
        let mut mem = MemBuffer::try_new(req).map_err(|e| format!("{e:?}"))?;
        evd::tridiag::tridiag_in_place(
            work.as_mut(),
            factor.as_mut(),
            Par::Seq,
            MemStack::new(&mut mem),
            Default::default(),
        );
        crate::householder_sequence::apply_compact(
            work.as_ref().submatrix(1, 0, n - 1, n - 1).to_owned(),
            factor,
            q.as_mut().subrows_mut(1, n - 1),
        )?;
        // A complex tridiagonal's phases are absorbed into Q to yield real H.
        if !T::IS_REAL {
            let mut phase = one::<T>();
            for i in 1..n {
                let e = &work[(i, i - 1)];
                let norm = abs(e);
                if norm != zero::<T::Real>() {
                    phase = &phase * (e / from_real::<T>(&norm));
                    let pnorm = abs(&phase);
                    phase = &phase / from_real::<T>(&pnorm);
                }
                for row in 0..n {
                    q[(row, i)] = &q[(row, i)] * &phase;
                }
            }
        }
        if let Some(scale) = scaling {
            for i in 0..n {
                work[(i, i)] = &work[(i, i)] / from_real::<T>(&scale);
                if i + 1 < n {
                    work[(i + 1, i)] = &work[(i + 1, i)] / from_real::<T>(&scale);
                }
            }
        }
    }
    for i in 0..n {
        dout.cast::<T::Real>().add(i).write(real(&work[(i, i)]));
        if i + 1 < n {
            eout.cast::<T::Real>().add(i).write(if T::IS_REAL {
                real(&work[(i + 1, i)])
            } else {
                abs(&work[(i + 1, i)])
            });
        }
    }
    for j in 0..n {
        for i in 0..n {
            qout.cast::<T>().add(i + j * n).write(copy(&q[(i, j)]));
        }
    }
    Ok(())
}
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
    guarded(|| unsafe {
        match tag {
            0 => reduction::<f32>(input, n, q, d, e, bs),
            1 => reduction::<f64>(input, n, q, d, e, bs),
            2 => reduction::<faer::c32>(input, n, q, d, e, bs),
            3 => reduction::<faer::c64>(input, n, q, d, e, bs),
            _ => Err("unsupported tridiagonal scalar".into()),
        }
    })
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
