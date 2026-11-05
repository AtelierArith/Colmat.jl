//! Native rectangular bidiagonalization and square bidiagonal SVD.
use crate::{guarded, typed::finite};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::svd;
use faer::traits::{math_utils::*, ComplexField};
use faer::{Mat, MatRef, Par};
mod workspace;
use rustcall_julia_macros::julia;
unsafe fn reduce<T: ComplexField>(
    input: *const u8,
    m: usize,
    n: usize,
    uout: *mut u8,
    vout: *mut u8,
    dout: *mut u8,
    eout: *mut u8,
    full: bool,
    blocksize: usize,
) -> Result<(), String> {
    let a = MatRef::<T>::from_raw_parts(input.cast(), m, n, 1, m as isize);
    finite(a)?;
    let wide = m < n;
    let mut work = if wide {
        a.adjoint().to_owned()
    } else {
        a.to_owned()
    };
    let rows = work.nrows();
    let k = work.ncols();
    let mut left = Mat::<T>::from_fn(rows, if full { rows } else { k }, |i, j| {
        if i == j {
            one()
        } else {
            zero()
        }
    });
    let mut right = Mat::<T>::identity(k, k);
    let scaling = crate::schur::scale(&mut work);
    if k > 0 {
        let bs = if blocksize == 0 {
            faer::linalg::qr::no_pivoting::factor::recommended_block_size::<T>(rows, k)
                .min(k)
                .max(1)
        } else {
            blocksize
        };
        let mut hl = Mat::<T>::zeros(bs, k);
        let mut hr = Mat::<T>::zeros(bs, k - 1);
        let mut mem = MemBuffer::try_new(svd::bidiag::bidiag_in_place_scratch::<T>(
            rows,
            k,
            Par::Seq.shape(),
            Default::default(),
        ))
        .map_err(|e| format!("{e:?}"))?;
        svd::bidiag::bidiag_in_place(
            work.as_mut(),
            hl.as_mut(),
            hr.as_mut(),
            Par::Seq,
            MemStack::new(&mut mem),
            Default::default(),
        );
        finite(work.as_ref()).map_err(|_| {
            format!("nonfinite native bidiagonal reduction at {rows}x{k}, blocksize {bs}")
        })?;
        crate::householder_sequence::apply_compact(work.clone(), hl, left.as_mut())?;
        if k > 1 {
            let basis = work
                .as_ref()
                .submatrix(0, 1, k - 1, k - 1)
                .transpose()
                .to_owned();
            crate::householder_sequence::apply_compact_conj(
                basis,
                hr,
                right.as_mut().subrows_mut(1, k - 1),
                true,
            )?;
        }
    }
    finite(left.as_ref()).map_err(|_| format!("nonfinite bidiagonal left basis at {rows}x{k}"))?;
    finite(right.as_ref())
        .map_err(|_| format!("nonfinite bidiagonal right basis at {rows}x{k}"))?;
    for i in 0..k {
        let mut d = copy(&work[(i, i)]);
        let mut e = if i + 1 < k {
            copy(&work[(i, i + 1)])
        } else {
            zero()
        };
        if let Some(ref f) = scaling {
            let inv_scale = recip(f);
            d = mul_real(&d, &inv_scale);
            e = mul_real(&e, &inv_scale);
        }
        dout.cast::<T>()
            .add(i)
            .write(if wide { conj(&d) } else { d });
        if i + 1 < k {
            eout.cast::<T>()
                .add(i)
                .write(if wide { conj(&e) } else { e });
        }
    }
    let (u, v) = if wide {
        (&right, &left)
    } else {
        (&left, &right)
    };
    for j in 0..u.ncols() {
        for i in 0..m {
            uout.cast::<T>().add(i + j * m).write(copy(&u[(i, j)]));
        }
    }
    for j in 0..v.ncols() {
        for i in 0..n {
            vout.cast::<T>().add(i + j * n).write(copy(&v[(i, j)]));
        }
    }
    Ok(())
}
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
    guarded(|| unsafe {
        match tag {
            0 => reduce::<f32>(input, m, n, u, v, d, e, full, bs),
            1 => reduce::<f64>(input, m, n, u, v, d, e, full, bs),
            2 => reduce::<faer::c32>(input, m, n, u, v, d, e, full, bs),
            3 => reduce::<faer::c64>(input, m, n, u, v, d, e, full, bs),
            _ => Err("unsupported bidiagonal scalar".into()),
        }
    })
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
