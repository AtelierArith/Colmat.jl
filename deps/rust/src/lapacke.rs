//! LAPACKE layout and integer adapters over the existing native LAPACK entries.
//! Reference contract: Reference-LAPACK v3.12.1, lapacke_*getrf_work.c.
//! Layout is always C int; dimensions, pivots and returned INFO follow lapack_int.
use crate::kernels::Scalar;
mod balance;
mod band_cholesky;
mod band_equilibration;
mod band_lu;
mod band_norms;
mod band_tridiagonal;
mod block_reflector;
mod block_reflector_apply;
mod cholesky;
mod compact_qr;
mod complex_auxiliary;
mod general_eigen;
mod general_expert;
mod generalized_eigen_driver;
mod generalized_expert_driver;
mod generalized_hessenberg;
mod generalized_qr;
mod generalized_schur_vectors;
mod givens;
mod householder;
mod lauum;
mod least_squares;
mod lq;
mod matrix_aux;
mod matrix_norms;
mod matrix_permutation;
mod packed_cholesky;
mod packed_lu;
mod packed_vectors;
mod pentagonal_qr;
mod pivoted_qr;
mod positive_qr;
mod positive_tridiagonal;
mod q_application;
mod q_generators;
mod qr;
mod qz;
mod reduction_vectors;
mod reordering;
mod reverse_qr;
mod rz;
mod scaling;
mod schur_driver;
mod schur_expert_driver;
mod schur_selection;
mod schur_vectors;
mod self_adjoint;
mod self_adjoint_range;
mod selfadjoint_vectors;
mod svd;
mod symmetric_lu;
mod symmetric_rank;
mod triangular;
mod triangular_aux;
mod tridiagonal_evd;
mod tridiagonal_lu;

const ROW_MAJOR: i32 = 101;
const COL_MAJOR: i32 = 102;
const TRANSPOSE_MEMORY_ERROR: i64 = -1011;

use std::sync::atomic::{AtomicI32, Ordering};
static NANCHECK: AtomicI32 = AtomicI32::new(-1);

#[no_mangle]
pub extern "C" fn LAPACKE_set_nancheck(flag: i32) {
    NANCHECK.store(i32::from(flag != 0), Ordering::Relaxed);
}

// Match atoi's zero/nonzero result without overflowing an intermediate integer.
fn environment_nancheck() -> i32 {
    let Some(value) = std::env::var_os("LAPACKE_NANCHECK") else {
        return 1;
    };
    let value = value.to_string_lossy();
    let value = value.trim_start_matches(|c: char| c.is_ascii_whitespace());
    let value = value.strip_prefix(['+', '-']).unwrap_or(value);
    i32::from(
        value
            .bytes()
            .take_while(u8::is_ascii_digit)
            .any(|c| c != b'0'),
    )
}

#[no_mangle]
pub extern "C" fn LAPACKE_get_nancheck() -> i32 {
    let flag = NANCHECK.load(Ordering::Relaxed);
    if flag != -1 {
        return flag;
    }
    let initial = environment_nancheck();
    match NANCHECK.compare_exchange(-1, initial, Ordering::Relaxed, Ordering::Relaxed) {
        Ok(_) => initial,
        Err(actual) => actual,
    }
}

// LBT also exposes suffixed helper trampolines. Their flags and return values
// remain C int (not lapack_int), and both aliases share the same policy state.
#[no_mangle]
pub extern "C" fn LAPACKE_set_nancheck64_(flag: i32) {
    LAPACKE_set_nancheck(flag);
}
#[no_mangle]
pub extern "C" fn LAPACKE_get_nancheck64_() -> i32 {
    LAPACKE_get_nancheck()
}

/// The reference NaN checker scans only min(logical extent, leading dimension)
/// in each stored row/column, even before the numerical argument checks.
unsafe fn ge_nancheck<T: Scalar>(
    layout: i32,
    m: i64,
    n: i64,
    a: *const T,
    lda: i64,
) -> Result<bool, i64> {
    if a.is_null() {
        return Ok(false);
    }
    let (outer, inner) = if layout == ROW_MAJOR {
        (m.max(0), n.min(lda).max(0))
    } else {
        (n.max(0), m.min(lda).max(0))
    };
    if outer == 0 || inner == 0 {
        return Ok(false);
    }
    let span = (outer as usize - 1)
        .checked_mul(lda as usize)
        .and_then(|v| v.checked_add(inner as usize))
        .and_then(|v| v.checked_mul(core::mem::size_of::<T>()));
    if span.is_none_or(|v| v > isize::MAX as usize) {
        return Err(TRANSPOSE_MEMORY_ERROR);
    }
    for i in 0..outer as usize {
        for j in 0..inner as usize {
            let x = *a.add(i * lda as usize + j);
            if x != x {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Copy only logical entries; padding belongs to the caller. Allocation failure
/// and address-span overflow return the LAPACKE transpose-memory diagnostic.
unsafe fn row_to_column<T: Scalar>(
    m: usize,
    n: usize,
    a: *const T,
    lda: usize,
) -> Result<Vec<T>, i64> {
    if m == 0 || n == 0 {
        return Ok(Vec::new());
    }
    let count = m.checked_mul(n).ok_or(TRANSPOSE_MEMORY_ERROR)?;
    let span = (m - 1)
        .checked_mul(lda)
        .and_then(|x| x.checked_add(n))
        .and_then(|x| x.checked_mul(core::mem::size_of::<T>()))
        .ok_or(TRANSPOSE_MEMORY_ERROR)?;
    if span > isize::MAX as usize {
        return Err(TRANSPOSE_MEMORY_ERROR);
    }
    let mut out = Vec::new();
    out.try_reserve_exact(count)
        .map_err(|_| TRANSPOSE_MEMORY_ERROR)?;
    out.resize(count, T::z());
    for j in 0..n {
        for i in 0..m {
            out[i + j * m] = *a.add(i * lda + j);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod empty_layout_tests {
    #[test]
    fn empty_row_conversion_never_reads_or_computes_a_nonempty_span() {
        for (m, n) in [(0, 0), (0, 3), (3, 0), (usize::MAX, 0)] {
            let values =
                unsafe { super::row_to_column::<f64>(m, n, core::ptr::null(), usize::MAX) };
            assert!(values.unwrap().is_empty());
        }
    }
}

macro_rules! getrf_work {
    ($p:ident, $t:ty, $i:ty, [$($suffix:tt)*], $fortran:tt) => { paste::paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p getrf $($suffix)*>](
            matrix_layout: i32, m: $i, n: $i, a: *mut $t, lda: $i, ipiv: *mut $i,
        ) -> $i {
            if matrix_layout != ROW_MAJOR && matrix_layout != COL_MAJOR { return -1; }
            if LAPACKE_get_nancheck() != 0 {
                match ge_nancheck(matrix_layout, m as i64, n as i64, a, lda as i64) {
                    Ok(true) => return -4, Err(info) => return info as $i, Ok(false) => {},
                }
            }
            [<LAPACKE_ $p getrf_work $($suffix)*>](matrix_layout, m, n, a, lda, ipiv)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p getrf_work $($suffix)*>](
            matrix_layout: i32, m: $i, n: $i, a: *mut $t, lda: $i, ipiv: *mut $i,
        ) -> $i {
            if matrix_layout != ROW_MAJOR && matrix_layout != COL_MAJOR { return -1; }
            // The reference row-major wrapper checks LDA before the Fortran call.
            if matrix_layout == ROW_MAJOR && lda < n { return -5; }
            if m < 0 { return -2; }
            if n < 0 { return -3; }
            if matrix_layout == COL_MAJOR && lda < 1.max(m) { return -5; }
            if m == 0 || n == 0 { return 0; }
            let mut info: $i = 0;
            if matrix_layout == COL_MAJOR {
                crate::lapack::[<$p getrf $fortran>](&m, &n, a, &lda, ipiv, &mut info);
            } else {
                let mut tmp = match row_to_column(m as usize, n as usize, a, lda as usize) {
                    Ok(tmp) => tmp, Err(info) => return info as $i,
                };
                crate::lapack::[<$p getrf $fortran>](&m, &n, tmp.as_mut_ptr(), &m, ipiv, &mut info);
                for j in 0..n as usize { for i in 0..m as usize {
                    *a.add(i*lda as usize+j) = tmp[i+j*m as usize];
                } }
            }
            if info < 0 { info - 1 } else { info }
        }
    }};
}
getrf_work!(s, f32, i32, [], _);
getrf_work!(d, f64, i32, [], _);
getrf_work!(c, colmat::c32, i32, [], _);
getrf_work!(z, colmat::c64, i32, [], _);
getrf_work!(s, f32, i64, [64_], _64_);
getrf_work!(d, f64, i64, [64_], _64_);
getrf_work!(c, colmat::c32, i64, [64_], _64_);
getrf_work!(z, colmat::c64, i64, [64_], _64_);

#[cfg(test)]
mod tests {
    use super::*;
    use core::ptr::null_mut;
    #[test]
    fn errors_and_empty_calls_do_not_access_buffers() {
        unsafe {
            assert_eq!(LAPACKE_dgetrf_work(0, 2, 2, null_mut(), 2, null_mut()), -1);
            assert_eq!(
                LAPACKE_dgetrf_work(102, -1, 2, null_mut(), 2, null_mut()),
                -2
            );
            assert_eq!(
                LAPACKE_dgetrf_work(102, 2, -1, null_mut(), 2, null_mut()),
                -3
            );
            assert_eq!(
                LAPACKE_dgetrf_work(101, 2, 3, null_mut(), 2, null_mut()),
                -5
            );
            assert_eq!(LAPACKE_dgetrf_work(101, 0, 0, null_mut(), 0, null_mut()), 0);
            assert_eq!(
                LAPACKE_dgetrf_work64_(102, -4294967293, 2, null_mut(), 2, null_mut()),
                -2
            );
            assert_eq!(
                LAPACKE_dgetrf_work64_(101, i64::MAX, 2, null_mut(), 2, null_mut()),
                -1011
            );
        }
    }
    macro_rules! check {
        ($name:ident,$call:ident,$t:ty,$i:ty,$phase:expr) => {
            #[test]
            fn $name() {
                unsafe {
                    // Rectangular factorization, mandatory pivot, padded rows/columns.
                    let original =
                        [[2., 1.], [4., 3.], [1., 5.]].map(|r| r.map(|x| <$t>::real(x) * $phase));
                    let guard = <$t>::real(777.);
                    let mut row = vec![guard; 12];
                    let mut col = vec![guard; 10];
                    for i in 0..3 {
                        for j in 0..2 {
                            row[i * 4 + j] = original[i][j];
                            col[i + j * 5] = original[i][j];
                        }
                    }
                    let mut pr = [0 as $i; 2];
                    let mut pc = pr;
                    assert_eq!($call(101, 3, 2, row.as_mut_ptr(), 4, pr.as_mut_ptr()), 0);
                    assert_eq!($call(102, 3, 2, col.as_mut_ptr(), 5, pc.as_mut_ptr()), 0);
                    assert_eq!(pr, pc);
                    assert_eq!(pr[0], 2);
                    let mut perm = original;
                    for k in 0..2 {
                        perm.swap(k, pr[k] as usize - 1);
                    }
                    for i in 0..3 {
                        for j in 0..2 {
                            assert_eq!(row[i * 4 + j], col[i + j * 5]);
                            let mut sum = <$t>::z();
                            for k in 0..2 {
                                let l = if i == k {
                                    <$t>::o()
                                } else if i > k {
                                    row[i * 4 + k]
                                } else {
                                    <$t>::z()
                                };
                                let u = if k <= j { row[k * 4 + j] } else { <$t>::z() };
                                sum = sum + l * u;
                            }
                            assert!((sum - perm[i][j]).magnitude() < 1e-5);
                        }
                    }
                    for i in 0..3 {
                        assert_eq!(&row[i * 4 + 2..i * 4 + 4], &[guard, guard]);
                    }
                    for j in 0..2 {
                        assert_eq!(&col[j * 5 + 3..j * 5 + 5], &[guard, guard]);
                    }
                    let mut singular = [
                        <$t>::real(1.),
                        <$t>::real(2.),
                        <$t>::real(2.),
                        <$t>::real(4.),
                    ];
                    assert_eq!(
                        $call(101, 2, 2, singular.as_mut_ptr(), 2, pr.as_mut_ptr()),
                        2
                    );
                }
            }
        };
    }
    check!(real32_lp64, LAPACKE_sgetrf_work, f32, i32, 1.0f32);
    check!(real64_lp64, LAPACKE_dgetrf_work, f64, i32, 1.0f64);
    check!(
        complex32_lp64,
        LAPACKE_cgetrf_work,
        colmat::c32,
        i32,
        colmat::c32::new(0.8, -0.6)
    );
    check!(
        complex64_lp64,
        LAPACKE_zgetrf_work,
        colmat::c64,
        i32,
        colmat::c64::new(0.8, -0.6)
    );
    check!(real32_ilp64, LAPACKE_sgetrf_work64_, f32, i64, 1.0f32);
    check!(real64_ilp64, LAPACKE_dgetrf_work64_, f64, i64, 1.0f64);
    check!(
        complex32_ilp64,
        LAPACKE_cgetrf_work64_,
        colmat::c32,
        i64,
        colmat::c32::new(0.8, -0.6)
    );
    check!(
        complex64_ilp64,
        LAPACKE_zgetrf_work64_,
        colmat::c64,
        i64,
        colmat::c64::new(0.8, -0.6)
    );
}

macro_rules! getrs {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt) => {paste::paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p getrs $($suffix)*>](
            layout: i32, trans: u8, n: $i, nrhs: $i, a: *const $t, lda: $i,
            ipiv: *const $i, b: *mut $t, ldb: $i,
        ) -> $i {
            if layout != ROW_MAJOR && layout != COL_MAJOR { return -1; }
            if LAPACKE_get_nancheck() != 0 {
                match ge_nancheck(layout,n as i64,n as i64,a,lda as i64) {
                    Ok(true) => return -5, Err(info) => return info as $i, Ok(false) => {},
                }
                match ge_nancheck(layout,n as i64,nrhs as i64,b,ldb as i64) {
                    Ok(true) => return -8, Err(info) => return info as $i, Ok(false) => {},
                }
            }
            [<LAPACKE_ $p getrs_work $($suffix)*>](layout,trans,n,nrhs,a,lda,ipiv,b,ldb)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p getrs_work $($suffix)*>](
            layout: i32, trans: u8, n: $i, nrhs: $i, a: *const $t, lda: $i,
            ipiv: *const $i, b: *mut $t, ldb: $i,
        ) -> $i {
            if layout != ROW_MAJOR && layout != COL_MAJOR { return -1; }
            if layout == ROW_MAJOR {
                if lda < n { return -6; }
                if ldb < nrhs { return -9; }
            }
            if !crate::kernels::trans(trans.to_ascii_uppercase()) { return -2; }
            if n < 0 { return -3; }
            if nrhs < 0 { return -4; }
            if layout == COL_MAJOR {
                if lda < 1.max(n) { return -6; }
                if ldb < 1.max(n) { return -9; }
            }
            if n == 0 || nrhs == 0 { return 0; }
            let mut info: $i = 0;
            if layout == COL_MAJOR {
                crate::lapack::[<$p getrs $fortran>](&trans,&n,&nrhs,a,&lda,ipiv,b,&ldb,&mut info);
            } else {
                let aa = match row_to_column(n as usize,n as usize,a,lda as usize) {
                    Ok(v) => v, Err(info) => return info as $i,
                };
                let mut bb = match row_to_column(n as usize,nrhs as usize,b,ldb as usize) {
                    Ok(v) => v, Err(info) => return info as $i,
                };
                crate::lapack::[<$p getrs $fortran>](&trans,&n,&nrhs,aa.as_ptr(),&n,ipiv,bb.as_mut_ptr(),&n,&mut info);
                for j in 0..nrhs as usize { for i in 0..n as usize {
                    *b.add(i*ldb as usize+j) = bb[i+j*n as usize];
                } }
            }
            if info < 0 { info-1 } else { info }
        }
    }};
}
getrs!(s, f32, i32, [], _);
getrs!(d, f64, i32, [], _);
getrs!(c, colmat::c32, i32, [], _);
getrs!(z, colmat::c64, i32, [], _);
getrs!(s, f32, i64, [64_], _64_);
getrs!(d, f64, i64, [64_], _64_);
getrs!(c, colmat::c32, i64, [64_], _64_);
getrs!(z, colmat::c64, i64, [64_], _64_);

macro_rules! gesv {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt) => {paste::paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p gesv $($suffix)*>](
            layout:i32,n:$i,nrhs:$i,a:*mut $t,lda:$i,ipiv:*mut $i,b:*mut $t,ldb:$i,
        ) -> $i {
            if layout != ROW_MAJOR && layout != COL_MAJOR { return -1; }
            if LAPACKE_get_nancheck() != 0 {
                match ge_nancheck(layout,n as i64,n as i64,a,lda as i64) {
                    Ok(true) => return -4, Err(info) => return info as $i, Ok(false) => {},
                }
                match ge_nancheck(layout,n as i64,nrhs as i64,b,ldb as i64) {
                    Ok(true) => return -7, Err(info) => return info as $i, Ok(false) => {},
                }
            }
            [<LAPACKE_ $p gesv_work $($suffix)*>](layout,n,nrhs,a,lda,ipiv,b,ldb)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p gesv_work $($suffix)*>](
            layout:i32,n:$i,nrhs:$i,a:*mut $t,lda:$i,ipiv:*mut $i,b:*mut $t,ldb:$i,
        ) -> $i {
            if layout != ROW_MAJOR && layout != COL_MAJOR { return -1; }
            if layout == ROW_MAJOR {
                if lda < n { return -5; }
                if ldb < nrhs { return -8; }
            }
            if n < 0 { return -2; }
            if nrhs < 0 { return -3; }
            if layout == COL_MAJOR {
                if lda < 1.max(n) { return -5; }
                if ldb < 1.max(n) { return -8; }
            }
            if n == 0 { return 0; }
            // NRHS=0 still factors A and writes IPIV, including singular INFO.
            let mut info:$i=0;
            if layout == COL_MAJOR {
                crate::lapack::[<$p gesv $fortran>](&n,&nrhs,a,&lda,ipiv,b,&ldb,&mut info);
            } else {
                let mut aa=match row_to_column(n as usize,n as usize,a,lda as usize) {
                    Ok(v)=>v,Err(info)=>return info as $i,
                };
                let mut bb=match row_to_column(n as usize,nrhs as usize,b,ldb as usize) {
                    Ok(v)=>v,Err(info)=>return info as $i,
                };
                crate::lapack::[<$p gesv $fortran>](&n,&nrhs,aa.as_mut_ptr(),&n,ipiv,bb.as_mut_ptr(),&n,&mut info);
                for j in 0..n as usize { for i in 0..n as usize {
                    *a.add(i*lda as usize+j)=aa[i+j*n as usize];
                } }
                for j in 0..nrhs as usize { for i in 0..n as usize {
                    *b.add(i*ldb as usize+j)=bb[i+j*n as usize];
                } }
            }
            if info<0 {info-1} else {info}
        }
    }};
}
gesv!(s, f32, i32, [], _);
gesv!(d, f64, i32, [], _);
gesv!(c, colmat::c32, i32, [], _);
gesv!(z, colmat::c64, i32, [], _);
gesv!(s, f32, i64, [64_], _64_);
gesv!(d, f64, i64, [64_], _64_);
gesv!(c, colmat::c32, i64, [64_], _64_);
gesv!(z, colmat::c64, i64, [64_], _64_);

macro_rules! getri {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt) => {paste::paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p getri $($suffix)*>](
            layout:i32,n:$i,a:*mut $t,lda:$i,ipiv:*const $i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                match ge_nancheck(layout,n as i64,n as i64,a,lda as i64) {
                    Ok(true)=>return -3,Err(info)=>return info as $i,Ok(false)=>{},
                }
            }
            let mut query=<$t>::z();
            let info=[<LAPACKE_ $p getri_work $($suffix)*>](layout,n,a,lda,ipiv,&mut query,-1);
            if info!=0 {return info;}
            let recommended=query.re();
            if !recommended.is_finite() || recommended<1.0 || recommended> <$i>::MAX as f64 {
                return -1010;
            }
            let lwork=recommended as $i;
            let mut work=Vec::<$t>::new();
            if work.try_reserve_exact(lwork as usize).is_err() {return -1010;}
            work.resize(lwork as usize,<$t>::z());
            [<LAPACKE_ $p getri_work $($suffix)*>](layout,n,a,lda,ipiv,work.as_mut_ptr(),lwork)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p getri_work $($suffix)*>](
            layout:i32,n:$i,a:*mut $t,lda:$i,ipiv:*const $i,work:*mut $t,lwork:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -4;}
            if n<0 {return -2;}
            if layout==COL_MAJOR && lda<1.max(n) {return -4;}
            if lwork!= -1 && lwork<1.max(n) {return -7;}
            let mut info:$i=0;
            if layout==COL_MAJOR {
                crate::lapack::[<$p getri $fortran>](&n,a,&lda,ipiv,work,&lwork,&mut info);
            } else if lwork== -1 || n==0 {
                // Query and empty calls do not touch A or IPIV; no transpose needed.
                let ld=1.max(n);
                crate::lapack::[<$p getri $fortran>](&n,a,&ld,ipiv,work,&lwork,&mut info);
            } else {
                let mut aa=match row_to_column(n as usize,n as usize,a,lda as usize) {
                    Ok(v)=>v,Err(info)=>return info as $i,
                };
                crate::lapack::[<$p getri $fortran>](&n,aa.as_mut_ptr(),&n,ipiv,work,&lwork,&mut info);
                for j in 0..n as usize {for i in 0..n as usize {
                    *a.add(i*lda as usize+j)=aa[i+j*n as usize];
                }}
            }
            if info<0 {info-1} else {info}
        }
    }};
}
getri!(s, f32, i32, [], _);
getri!(d, f64, i32, [], _);
getri!(c, colmat::c32, i32, [], _);
getri!(z, colmat::c64, i32, [], _);
getri!(s, f32, i64, [64_], _64_);
getri!(d, f64, i64, [64_], _64_);
getri!(c, colmat::c32, i64, [64_], _64_);
getri!(z, colmat::c64, i64, [64_], _64_);

// The existing Fortran GETF2/GETRF2 entries use the same public native LU
// factorizer. Preserve that documented numerical contract without claiming
// unblocked/recursive LAPACK algorithm or performance equivalence.
macro_rules! lu_alias {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$routine:ident)=>{paste::paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $routine $($suffix)*>](
            layout:i32,m:$i,n:$i,a:*mut $t,lda:$i,ipiv:*mut $i,
        )->$i {
            [<LAPACKE_ $p getrf $($suffix)*>](layout,m,n,a,lda,ipiv)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $routine _work $($suffix)*>](
            layout:i32,m:$i,n:$i,a:*mut $t,lda:$i,ipiv:*mut $i,
        )->$i {
            [<LAPACKE_ $p getrf_work $($suffix)*>](layout,m,n,a,lda,ipiv)
        }
    }};
}
macro_rules! lu_aliases {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*])=>{
        lu_alias!($p,$t,$i,[$($suffix)*],getf2);
        lu_alias!($p,$t,$i,[$($suffix)*],getrf2);
    };
}
lu_aliases!(s, f32, i32, []);
lu_aliases!(d, f64, i32, []);
lu_aliases!(c, colmat::c32, i32, []);
lu_aliases!(z, colmat::c64, i32, []);
lu_aliases!(s, f32, i64, [64_]);
lu_aliases!(d, f64, i64, [64_]);
lu_aliases!(c, colmat::c32, i64, [64_]);
lu_aliases!(z, colmat::c64, i64, [64_]);
