//! LAPACKE layout and integer adapters over the existing native LAPACK entries.
//! Reference contract: Reference-LAPACK v3.12.1, lapacke_*getrf_work.c.
//! Layout is always C int; dimensions, pivots and returned INFO follow lapack_int.
use crate::kernels::Scalar;

const ROW_MAJOR: i32 = 101;
const COL_MAJOR: i32 = 102;
const TRANSPOSE_MEMORY_ERROR: i64 = -1011;

/// Copy only logical entries; padding belongs to the caller. Allocation failure
/// and address-span overflow return the LAPACKE transpose-memory diagnostic.
unsafe fn row_to_column<T: Scalar>(
    m: usize,
    n: usize,
    a: *const T,
    lda: usize,
) -> Result<Vec<T>, i64> {
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

macro_rules! getrf_work {
    ($p:ident, $t:ty, $i:ty, [$($suffix:tt)*], $fortran:tt) => { paste::paste! {
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
