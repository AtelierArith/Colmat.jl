//! Dense composition of split-factor solves and bandwidth-preserving reflectors.
//! This owns matrices and scratch; it does not reproduce LAPACK's band bulge chase.
use crate::generalized_selfadjoint::GeneralizedScalar;
use crate::householder::ReflectorScalar;
use crate::schur::multiply;
use colmat::Mat;

unsafe fn band<T: GeneralizedScalar>(
    n: usize,
    width: usize,
    lower: bool,
    p: *const T,
    ld: usize,
) -> Mat<T> {
    let mut a = Mat::<T>::zeros(n, n);
    for j in 0..n {
        let rows = if lower {
            j..n.min(j.saturating_add(width).saturating_add(1))
        } else {
            j.saturating_sub(width)..j + 1
        };
        for i in rows {
            let row = if lower { i - j } else { width - (j - i) };
            let entry = p.add(row + j * ld);
            let value = if i == j { T::diagonal(entry) } else { *entry };
            a[(i, j)] = value;
            a[(j, i)] = value.cj();
        }
    }
    a
}
unsafe fn split<T: GeneralizedScalar>(
    n: usize,
    width: usize,
    lower: bool,
    p: *const T,
    ld: usize,
) -> Mat<T> {
    let middle = (n / 2 + width / 2 + (n % 2 + width % 2) / 2).min(n);
    let mut s = Mat::<T>::zeros(n, n);
    for j in 0..n {
        let rows = if lower {
            j..n.min(j.saturating_add(width).saturating_add(1))
        } else {
            j.saturating_sub(width)..j + 1
        };
        for i in rows {
            let row = if lower { i - j } else { width - (j - i) };
            let entry = p.add(row + j * ld);
            let value = if i == j { T::diagonal(entry) } else { *entry };
            if lower {
                if i >= middle {
                    s[(i, j)] = value;
                } else {
                    s[(j, i)] = value.cj();
                }
            } else if j >= middle {
                s[(j, i)] = value.cj();
            } else {
                s[(i, j)] = value;
            }
        }
    }
    s
}
fn lower_storage<T: GeneralizedScalar>(matrix: &Mat<T>, width: usize) -> (Vec<T>, usize) {
    let n = matrix.nrows();
    let actual = width.min(n.saturating_sub(1));
    let ld = actual + 1;
    let mut data = vec![T::z(); ld * n];
    for j in 0..n {
        for i in j..n.min(j + actual + 1) {
            data[i - j + j * ld] = matrix[(i, j)];
        }
    }
    (data, ld)
}
unsafe fn commit_split<T: GeneralizedScalar>(
    n: usize,
    width: usize,
    lower: bool,
    b: *mut T,
    ld: usize,
    factor: &[T],
    ldf: usize,
) {
    let actual = width.min(n.saturating_sub(1));
    for j in 0..n {
        for i in j..n.min(j + actual + 1) {
            let value = factor[i - j + j * ldf];
            if lower {
                *b.add(i - j + j * ld) = value;
            } else {
                *b.add(width - (i - j) + i * ld) = value.cj();
            }
        }
    }
}
pub(crate) unsafe fn eigen<T: GeneralizedScalar + ReflectorScalar>(
    job: u8,
    n: usize,
    ka: usize,
    kb: usize,
    lower: bool,
    ab: *const T,
    lda: usize,
    bb: *mut T,
    ldb: usize,
) -> Result<
    crate::generalized_selfadjoint_drivers::NativeResult<T>,
    crate::generalized_selfadjoint_drivers::Failure,
> {
    use crate::generalized_selfadjoint_drivers::{Failure, NativeResult};
    if n == 0 {
        return Ok(NativeResult {
            values: Vec::new(),
            vectors: Mat::zeros(0, 0),
            reduced: Mat::zeros(0, 0),
        });
    }
    let a = band(n, ka, lower, ab, lda);
    let b = band(n, kb, lower, bb, ldb);
    let (aa, la) = lower_storage(&a, ka);
    let (mut factor, lf) = lower_storage(&b, kb);
    // The declared KB determines PBSTF's split even when KB exceeds N.
    // A compact lower stage suffices: no actual column has more than N rows.
    let status = crate::band_cholesky::factor(true, n, kb, factor.as_mut_ptr(), lf, true);
    if status != 0 {
        commit_split(n, kb, lower, bb, ldb, &factor, lf);
        return Err(Failure::NonPositive(status));
    }
    let (c, x) = standardize(n, ka, kb, true, aa.as_ptr(), la, factor.as_ptr(), lf)
        .ok_or(Failure::Native)?;
    let (values, vectors) =
        crate::lapack::selfadj(c.as_ref(), job == b'V').map_err(|_| Failure::Native)?;
    let vectors = if job == b'V' {
        multiply(x.as_ref(), b'N', vectors.as_ref(), b'N')
    } else {
        vectors
    };
    if !values.iter().all(|v| v.is_finite()) || !vectors.as_ref().is_all_finite() {
        return Err(Failure::Native);
    }
    commit_split(n, kb, lower, bb, ldb, &factor, lf);
    Ok(NativeResult {
        values,
        vectors,
        reduced: c,
    })
}

/// Return C=X^H A X and X^H B X=I, retaining the requested A bandwidth.
/// Input BB is the saved PBSTF factor S, with B=S^H S.
pub(crate) unsafe fn standardize<T: GeneralizedScalar + ReflectorScalar>(
    n: usize,
    ka: usize,
    kb: usize,
    lower: bool,
    ab: *const T,
    lda: usize,
    bb: *const T,
    ldb: usize,
) -> Option<(Mat<T>, Mat<T>)> {
    let a = band(n, ka, lower, ab, lda);
    let s = split(n, kb, lower, bb, ldb);
    let middle = (n / 2 + kb / 2 + (n % 2 + kb % 2) / 2).min(n);
    let mut x = Mat::<T>::from_fn(n, n, |i, j| if i == j { T::o() } else { T::z() });
    crate::threading::with_par(|mut par| {
        if middle > 0 {
            colmat::linalg::triangular_solve::solve_upper_triangular_in_place(
                s.as_ref().get(..middle, ..middle),
                x.as_mut().get_mut(..middle, ..),
                par.reborrow(),
            );
        }
        if middle < n {
            let update = multiply(
                s.as_ref().get(middle.., ..middle),
                b'N',
                x.as_ref().get(..middle, ..),
                b'N',
            );
            for j in 0..n {
                for i in middle..n {
                    x[(i, j)] = x[(i, j)] - update[(i - middle, j)];
                }
            }
            colmat::linalg::triangular_solve::solve_lower_triangular_in_place(
                s.as_ref().get(middle.., middle..),
                x.as_mut().get_mut(middle.., ..),
                par,
            );
        }
    });
    if !x.as_ref().is_all_finite() {
        return None;
    }
    let mut c = multiply(
        x.as_ref(),
        b'C',
        multiply(a.as_ref(), b'N', x.as_ref(), b'N').as_ref(),
        b'N',
    );
    if !c.as_ref().is_all_finite() {
        return None;
    }
    // The two native products can leave roundoff in the imaginary diagonal
    // even when no bandwidth-reduction step is necessary.
    for j in 0..n {
        c[(j, j)] = T::real(c[(j, j)].re());
        for i in j + 1..n {
            c[(j, i)] = c[(i, j)].cj();
        }
    }
    let width = ka.min(n.saturating_sub(1));
    if width > 0 {
        let mut work = vec![T::z(); n];
        for j in 0..n.saturating_sub(width + 1) {
            let first = j + width;
            let size = n - first;
            let mut v: Vec<T> = (first..n).map(|i| c[(i, j)]).collect();
            let mut alpha = v[0];
            let mut tau = T::z();
            crate::householder::generate(
                size as isize,
                &mut alpha,
                v.as_mut_ptr().add(1),
                1,
                &mut tau,
                false,
            );
            v[0] = T::o();
            let ldc = c.as_ref().col_stride();
            let ldx = x.as_ref().col_stride();
            crate::householder::apply(
                b'L',
                size as isize,
                n as isize,
                v.as_ptr(),
                1,
                tau.cj(),
                c.as_mut().as_ptr_mut().add(first),
                ldc,
                work.as_mut_ptr(),
            );
            crate::householder::apply(
                b'R',
                n as isize,
                size as isize,
                v.as_ptr(),
                1,
                tau,
                c.as_mut().as_ptr_mut().offset(first as isize * ldc),
                ldc,
                work.as_mut_ptr(),
            );
            crate::householder::apply(
                b'R',
                n as isize,
                size as isize,
                v.as_ptr(),
                1,
                tau,
                x.as_mut().as_ptr_mut().offset(first as isize * ldx),
                ldx,
                work.as_mut_ptr(),
            );
            c[(first, j)] = alpha;
            c[(j, first)] = alpha.cj();
            for i in first + 1..n {
                c[(i, j)] = T::z();
                c[(j, i)] = T::z();
            }
            for col in 0..n {
                c[(col, col)] = T::real(c[(col, col)].re());
                for row in col + 1..n {
                    c[(col, row)] = c[(row, col)].cj();
                }
            }
        }
    }
    if c.as_ref().is_all_finite() && x.as_ref().is_all_finite() {
        Some((c, x))
    } else {
        None
    }
}

unsafe fn driver<T: GeneralizedScalar + ReflectorScalar, I: crate::lapack::LapackInt>(
    vect: u8,
    u: u8,
    n: isize,
    ka: isize,
    kb: isize,
    ab: *mut T,
    lda: isize,
    bb: *const T,
    ldb: isize,
    x: *mut T,
    ldx: isize,
    info: *mut I,
) -> bool {
    *info = I::from_i64(0);
    for (arg, ok) in [
        (1, matches!(vect, b'N' | b'V')),
        (2, matches!(u, b'U' | b'L')),
        (3, n >= 0),
        (4, ka >= 0),
        (5, kb >= 0 && kb <= ka),
        (7, ka.checked_add(1).is_some_and(|k| lda >= k)),
        (9, kb.checked_add(1).is_some_and(|k| ldb >= k)),
        (11, ldx >= 1 && (vect == b'N' || ldx >= n)),
    ] {
        if !ok {
            *info = I::from_i64(-arg);
            crate::invalid(arg as i32);
            return true;
        }
    }
    if n == 0 {
        return true;
    }
    let rowsa = if u == b'L' { (ka + 1).min(n) } else { ka + 1 };
    let rowsb = if u == b'L' { (kb + 1).min(n) } else { kb + 1 };
    for (arg, ok) in [
        (3, crate::kernels::matrix_addressable::<T>(n, n, n)),
        (7, crate::kernels::matrix_addressable::<T>(rowsa, n, lda)),
        (9, crate::kernels::matrix_addressable::<T>(rowsb, n, ldb)),
        (
            11,
            vect == b'N' || crate::kernels::matrix_addressable::<T>(n, n, ldx),
        ),
    ] {
        if !ok {
            *info = I::from_i64(-arg);
            crate::invalid(arg as i32);
            return true;
        }
    }
    let Some((c, transform)) = standardize(
        n as usize,
        ka as usize,
        kb as usize,
        u == b'L',
        ab,
        lda as usize,
        bb,
        ldb as usize,
    ) else {
        return false;
    };
    for j in 0..n as usize {
        let rows = if u == b'L' {
            j..(n as usize).min(j.saturating_add(ka as usize).saturating_add(1))
        } else {
            j.saturating_sub(ka as usize)..j + 1
        };
        for i in rows {
            let row = if u == b'L' {
                i - j
            } else {
                ka as usize - (j - i)
            };
            *ab.offset(row as isize + j as isize * lda) = c[(i, j)];
        }
    }
    if vect == b'V' {
        crate::lapack::output(&transform, x, ldx);
    }
    true
}
macro_rules! real_exports {($p:ident,$t:ty,$i:ty,$s:tt,$id:expr)=>{paste::paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p sbgst $s>](vect:*const u8,u:*const u8,n:*const $i,ka:*const $i,kb:*const $i,ab:*mut $t,lda:*const $i,bb:*const $t,ldb:*const $i,x:*mut $t,ldx:*const $i,work:*mut $t,info:*mut $i) {
        use crate::lapack::LapackInt;
        if driver(crate::kernels::flag(vect),crate::kernels::flag(u),(*n).val(),(*ka).val(),(*kb).val(),ab,(*lda).val(),bb,(*ldb).val(),x,(*ldx).val(),info) {return;}
        let f:unsafe extern "C" fn(*const u8,*const u8,*const i64,*const i64,*const i64,*mut $t,*const i64,*const $t,*const i64,*mut $t,*const i64,*mut $t,*mut i64,usize,usize)=std::mem::transmute(crate::structured_generalized::fallback(12+$id));
        let mut status=0i64;f(vect,u,&((*n).val() as i64),&((*ka).val() as i64),&((*kb).val() as i64),ab,&((*lda).val() as i64),bb,&((*ldb).val() as i64),x,&((*ldx).val() as i64),work,&mut status,1,1);*info=status as $i;
    }
}};}
macro_rules! complex_exports {($p:ident,$t:ty,$r:ty,$i:ty,$s:tt,$id:expr)=>{paste::paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p hbgst $s>](vect:*const u8,u:*const u8,n:*const $i,ka:*const $i,kb:*const $i,ab:*mut $t,lda:*const $i,bb:*const $t,ldb:*const $i,x:*mut $t,ldx:*const $i,work:*mut $t,rwork:*mut $r,info:*mut $i) {
        use crate::lapack::LapackInt;
        if driver(crate::kernels::flag(vect),crate::kernels::flag(u),(*n).val(),(*ka).val(),(*kb).val(),ab,(*lda).val(),bb,(*ldb).val(),x,(*ldx).val(),info) {return;}
        let f:unsafe extern "C" fn(*const u8,*const u8,*const i64,*const i64,*const i64,*mut $t,*const i64,*const $t,*const i64,*mut $t,*const i64,*mut $t,*mut $r,*mut i64,usize,usize)=std::mem::transmute(crate::structured_generalized::fallback(12+$id));
        let mut status=0i64;f(vect,u,&((*n).val() as i64),&((*ka).val() as i64),&((*kb).val() as i64),ab,&((*lda).val() as i64),bb,&((*ldb).val() as i64),x,&((*ldx).val() as i64),work,rwork,&mut status,1,1);*info=status as $i;
    }
}};}
real_exports!(s, f32, i32, _, 0);
real_exports!(d, f64, i32, _, 1);
real_exports!(s, f32, i64, _64_, 0);
real_exports!(d, f64, i64, _64_, 1);
complex_exports!(c, colmat::c32, f32, i32, _, 2);
complex_exports!(z, colmat::c64, f64, i32, _, 3);
complex_exports!(c, colmat::c32, f32, i64, _64_, 2);
complex_exports!(z, colmat::c64, f64, i64, _64_, 3);

#[cfg(test)]
mod tests {
    use super::*;
    fn abi<T: GeneralizedScalar + ReflectorScalar, I: crate::lapack::LapackInt>(tol: f64) {
        for u in [b'U', b'L'] {
            for vect in [b'N', b'V'] {
                let mut a = [T::real(777.0), T::real(3.0), T::real(777.0)];
                let b = [T::real(777.0), T::real(2.0), T::real(777.0)];
                let mut x = [T::real(777.0); 3];
                let mut info = [I::from_i64(-777); 3];
                assert!(unsafe {
                    driver(
                        vect,
                        u,
                        1,
                        0,
                        0,
                        a.as_mut_ptr().add(1),
                        isize::MAX,
                        b.as_ptr().add(1),
                        isize::MAX,
                        if vect == b'V' {
                            x.as_mut_ptr().add(1)
                        } else {
                            std::ptr::null_mut()
                        },
                        isize::MAX,
                        &mut info[1],
                    )
                });
                assert_eq!(info[1].val(), 0);
                assert_eq!(info[0].val(), -777);
                assert_eq!(info[2].val(), -777);
                assert_eq!(crate::colmat_blas_last_error(), 0);
                assert!((a[1] - T::real(0.75)).magnitude() < tol);
                assert_eq!(a[0].re(), 777.0);
                assert_eq!(a[2].re(), 777.0);
                assert!((x[1] - T::real(if vect == b'V' { 0.5 } else { 777.0 })).magnitude() < tol);
                assert_eq!(x[0].re(), 777.0);
                assert_eq!(x[2].re(), 777.0);
            }
        }
    }
    #[test]
    fn band_driver_both_integer_widths() {
        abi::<f32, i32>(1e-5);
        abi::<f32, i64>(1e-5);
        abi::<f64, i32>(1e-12);
        abi::<f64, i64>(1e-12);
        abi::<colmat::c32, i32>(1e-5);
        abi::<colmat::c32, i64>(1e-5);
        abi::<colmat::c64, i32>(1e-12);
        abi::<colmat::c64, i64>(1e-12);
    }
    #[test]
    fn band_driver_headers_precede_unused_arrays() {
        for (vect, u, n, ka, kb, lda, ldb, ldx, expected) in [
            (b'?', b'L', 0, 0, 0, 1, 1, 1, 1),
            (b'N', b'?', 0, 0, 0, 1, 1, 1, 2),
            (b'N', b'L', -1, 0, 0, 1, 1, 1, 3),
            (b'N', b'L', 0, -1, 0, 1, 1, 1, 4),
            (b'N', b'L', 0, 0, 1, 1, 1, 1, 5),
            (b'N', b'L', 0, 1, 0, 1, 1, 1, 7),
            (b'N', b'L', 0, 1, 1, 2, 1, 1, 9),
            (b'N', b'L', 0, 0, 0, 1, 1, 0, 11),
            (b'N', b'L', 0, 0, 0, 1, 1, 1, 0),
        ] {
            let mut info = -777i64;
            assert!(unsafe {
                driver::<f64, i64>(
                    vect,
                    u,
                    n,
                    ka,
                    kb,
                    std::ptr::null_mut(),
                    lda,
                    std::ptr::null(),
                    ldb,
                    std::ptr::null_mut(),
                    ldx,
                    &mut info,
                )
            });
            assert_eq!(info, -expected);
            assert_eq!(crate::colmat_blas_last_error(), expected as i32);
        }
    }
    fn eigensystem<T: GeneralizedScalar + ReflectorScalar>(phase: T, tol: f64) {
        for n in [0usize, 1, 5, 9] {
            for ka in [0usize, 1, 3, n + 1] {
                for kb in 0..=ka.min(2) {
                    for lower in [false, true] {
                        let a = Mat::<T>::from_fn(n, n, |i, j| {
                            if i == j {
                                T::real(1.0 + i as f64 * 0.1)
                            } else if i > j && i - j <= ka {
                                phase
                            } else if j > i && j - i <= ka {
                                phase.cj()
                            } else {
                                T::z()
                            }
                        });
                        let b = Mat::<T>::from_fn(n, n, |i, j| {
                            if i == j {
                                T::real(3.0)
                            } else if i > j && i - j <= kb {
                                phase
                            } else if j > i && j - i <= kb {
                                phase.cj()
                            } else {
                                T::z()
                            }
                        });
                        let (lda, ldb) = (ka + 3, kb + 3);
                        let mut ab = vec![T::real(f64::NAN); lda * n];
                        let mut bb = vec![T::real(f64::NAN); ldb * n];
                        for j in 0..n {
                            for i in 0..n {
                                if (lower && i >= j && i - j <= ka)
                                    || (!lower && i <= j && j - i <= ka)
                                {
                                    let row = if lower { i - j } else { ka - (j - i) };
                                    ab[row + j * lda] = a[(i, j)];
                                }
                                if (lower && i >= j && i - j <= kb)
                                    || (!lower && i <= j && j - i <= kb)
                                {
                                    let row = if lower { i - j } else { kb - (j - i) };
                                    bb[row + j * ldb] = b[(i, j)];
                                }
                            }
                        }
                        let result = match unsafe {
                            eigen(
                                b'V',
                                n,
                                ka,
                                kb,
                                lower,
                                ab.as_ptr(),
                                lda,
                                bb.as_mut_ptr(),
                                ldb,
                            )
                        } {
                            Ok(x) => x,
                            Err(_) => panic!(
                                "positive definite band eigenproblem failed: n={n},ka={ka},kb={kb}"
                            ),
                        };
                        assert!(result.values.windows(2).all(|v| v[0] <= v[1]));
                        let az = multiply(a.as_ref(), b'N', result.vectors.as_ref(), b'N');
                        let bz = multiply(b.as_ref(), b'N', result.vectors.as_ref(), b'N');
                        let gram = multiply(result.vectors.as_ref(), b'C', bz.as_ref(), b'N');
                        let s = unsafe { split(n, kb, lower, bb.as_ptr(), ldb) };
                        let rebuilt = multiply(s.as_ref(), b'C', s.as_ref(), b'N');
                        for j in 0..n {
                            for i in 0..n {
                                assert!(
                                    (az[(i, j)] - bz[(i, j)] * T::real(result.values[j]))
                                        .magnitude()
                                        < tol
                                );
                                assert!(
                                    (gram[(i, j)] - if i == j { T::o() } else { T::z() })
                                        .magnitude()
                                        < tol
                                );
                                assert!((rebuilt[(i, j)] - b[(i, j)]).magnitude() < tol);
                            }
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn band_eigenpairs_and_split_outputs_all_types() {
        eigensystem::<f32>(0.1, 3e-4);
        eigensystem::<f64>(0.1, 1e-11);
        eigensystem::<colmat::c32>(colmat::c32::new(0.1, 0.15), 5e-4);
        eigensystem::<colmat::c64>(colmat::c64::new(0.1, 0.15), 1e-11);
    }
    fn check<T: GeneralizedScalar + ReflectorScalar>(phase: T, tol: f64) {
        for n in [0usize, 1, 4, 7] {
            for ka in [0usize, 1, 2, n + 1] {
                for kb in 0..=ka.min(2) {
                    for lower in [false, true] {
                        let middle = (n / 2 + kb / 2 + (n % 2 + kb % 2) / 2).min(n);
                        let s = Mat::<T>::from_fn(n, n, |i, j| {
                            if i == j {
                                T::real(2.0 + i as f64 * 0.1)
                            } else if (i < middle && j < middle && i < j && j - i <= kb)
                                || (i >= middle && j < i && i - j <= kb)
                            {
                                phase
                            } else {
                                T::z()
                            }
                        });
                        let a = Mat::<T>::from_fn(n, n, |i, j| {
                            if i == j {
                                T::real(4.0 + i as f64 * 0.1)
                            } else if i > j && i - j <= ka {
                                phase
                            } else if j > i && j - i <= ka {
                                phase.cj()
                            } else {
                                T::z()
                            }
                        });
                        let b = multiply(s.as_ref(), b'C', s.as_ref(), b'N');
                        let (lda, ldb) = (ka + 3, kb + 3);
                        let mut ab = vec![T::real(f64::NAN); lda * n];
                        let mut bb = vec![T::real(f64::NAN); ldb * n];
                        for j in 0..n {
                            for i in 0..n {
                                if (lower && i >= j && i - j <= ka)
                                    || (!lower && i <= j && j - i <= ka)
                                {
                                    let row = if lower { i - j } else { ka - (j - i) };
                                    ab[row + j * lda] = a[(i, j)];
                                }
                                if (lower && i >= j && i - j <= kb)
                                    || (!lower && i <= j && j - i <= kb)
                                {
                                    let row = if lower { i - j } else { kb - (j - i) };
                                    bb[row + j * ldb] = if lower {
                                        if i >= middle {
                                            s[(i, j)]
                                        } else {
                                            s[(j, i)].cj()
                                        }
                                    } else if j >= middle {
                                        s[(j, i)].cj()
                                    } else {
                                        s[(i, j)]
                                    };
                                }
                            }
                        }
                        let (c, x) = unsafe {
                            standardize(n, ka, kb, lower, ab.as_ptr(), lda, bb.as_ptr(), ldb)
                        }
                        .expect("finite split congruence");
                        let ax = multiply(
                            x.as_ref(),
                            b'C',
                            multiply(a.as_ref(), b'N', x.as_ref(), b'N').as_ref(),
                            b'N',
                        );
                        let bx = multiply(
                            x.as_ref(),
                            b'C',
                            multiply(b.as_ref(), b'N', x.as_ref(), b'N').as_ref(),
                            b'N',
                        );
                        for j in 0..n {
                            for i in 0..n {
                                assert!(
                                    (ax[(i, j)] - c[(i, j)]).magnitude() < tol,
                                    "n={n} ka={ka} kb={kb} lower={lower}"
                                );
                                assert!(
                                    (bx[(i, j)] - if i == j { T::o() } else { T::z() }).magnitude()
                                        < tol
                                );
                                if i == j {
                                    assert_eq!(c[(i, j)].im(), 0.0);
                                }
                                if i.abs_diff(j) > ka {
                                    assert_eq!(c[(i, j)], T::z());
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn real32_split_congruence() {
        check::<f32>(0.1, 2e-4);
    }
    #[test]
    fn real64_split_congruence() {
        check::<f64>(0.1, 1e-11);
    }
    #[test]
    fn complex32_split_congruence() {
        check::<colmat::c32>(colmat::c32::new(0.1, 0.15), 3e-4);
    }
    #[test]
    fn complex64_split_congruence() {
        check::<colmat::c64>(colmat::c64::new(0.1, 0.15), 1e-11);
    }
}
