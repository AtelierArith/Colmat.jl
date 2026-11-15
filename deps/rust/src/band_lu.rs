//! Band LU keeps LAPACK's interleaved pivots rather than packing dense LU.
//! Each stored column is factored by native partial-pivoting LU, then a
//! native rank-one product updates the active skew-strided band block.
use crate::kernels::Scalar;
use colmat::{Accum, Conj, MatMut, MatRef};

pub(crate) unsafe fn factor<T: Scalar>(
    m: usize,
    n: usize,
    kl: usize,
    ku: usize,
    ab: *mut T,
    ld: usize,
) -> (Vec<usize>, usize) {
    if m == 0 || n == 0 {
        return (Vec::new(), 0);
    }
    let kv = kl + ku;
    // These are the only initially uninitialized fill-in slots touched by
    // the first panels. Padding outside the documented factor is preserved.
    for j in (ku + 1)..kv.min(n) {
        for i in (kv - j)..kl {
            *ab.add(i + j * ld) = T::z();
        }
    }
    let mut last = 0;
    let mut pivots = Vec::with_capacity(m.min(n));
    let mut info = 0;
    for j in 0..m.min(n) {
        if j + kv < n {
            for i in 0..kl {
                *ab.add(i + (j + kv) * ld) = T::z();
            }
        }
        let km = kl.min(m - j - 1);
        let column = ab.add(kv + j * ld);
        let (piv, singular) = crate::lapack::lu(MatMut::from_raw_parts_mut(
            column,
            km + 1,
            1,
            1,
            ld as isize,
        ));
        let pivot = piv[0];
        pivots.push(j + pivot);
        if singular != 0 {
            if info == 0 {
                info = j + 1;
            }
            continue;
        }
        last = last.max((j + ku + pivot).min(n - 1));
        if pivot != 0 {
            // The native column factorization already swapped column j.
            // LAPACK does not apply these swaps to the previous L columns.
            for c in (j + 1)..=last {
                std::ptr::swap(
                    ab.add(kv + j - c + c * ld),
                    ab.add(kv + j + pivot - c + c * ld),
                );
            }
        }
        if km > 0 && last > j {
            let left = MatRef::from_raw_parts(column.add(1), km, 1, 1, 0);
            let right = MatRef::from_raw_parts(
                ab.add(kv - 1 + (j + 1) * ld),
                1,
                last - j,
                0,
                (ld - 1) as isize,
            );
            let target = MatMut::from_raw_parts_mut(
                ab.add(kv + (j + 1) * ld),
                km,
                last - j,
                1,
                (ld - 1) as isize,
            );
            crate::threading::with_par(|par| {
                colmat::linalg::matmul::matmul_with_conj(
                    target,
                    Accum::Add,
                    left,
                    Conj::No,
                    right,
                    Conj::No,
                    -T::o(),
                    par,
                );
            });
        }
    }
    (pivots, info)
}

unsafe fn solve<T: Scalar>(
    n: usize,
    kl: usize,
    ku: usize,
    ab: *const T,
    ld: usize,
    pivots: &[usize],
    b: *mut T,
    ldb: usize,
    nrhs: usize,
    tr: u8,
) {
    if n == 0 || nrhs == 0 {
        return;
    }
    let kv = kl + ku;
    let swap = |j: usize| {
        for c in 0..nrhs {
            std::ptr::swap(b.add(j + c * ldb), b.add(pivots[j] + c * ldb));
        }
    };
    if tr == b'N' {
        for j in 0..n {
            swap(j);
            let km = kl.min(n - j - 1);
            if km > 0 {
                let left = MatRef::from_raw_parts(ab.add(kv + 1 + j * ld), km, 1, 1, 0);
                let right = MatRef::from_raw_parts(b.add(j), 1, nrhs, 0, ldb as isize);
                let target = MatMut::from_raw_parts_mut(b.add(j + 1), km, nrhs, 1, ldb as isize);
                crate::threading::with_par(|par| {
                    colmat::linalg::matmul::matmul_with_conj(
                        target,
                        Accum::Add,
                        left,
                        Conj::No,
                        right,
                        Conj::No,
                        -T::o(),
                        par,
                    );
                });
            }
        }
    }
    for c in 0..nrhs {
        crate::structured_triangular::solve_upper_band(n, kv, ab, ld, b.add(c * ldb), tr);
    }
    if tr != b'N' {
        for j in (0..n).rev() {
            let km = kl.min(n - j - 1);
            if km > 0 {
                let left = MatRef::from_raw_parts(ab.add(kv + 1 + j * ld), 1, km, 0, 1);
                let right = MatRef::from_raw_parts(b.add(j + 1), km, nrhs, 1, ldb as isize);
                let target = MatMut::from_raw_parts_mut(b.add(j), 1, nrhs, 0, ldb as isize);
                crate::threading::with_par(|par| {
                    colmat::linalg::matmul::matmul_with_conj(
                        target,
                        Accum::Add,
                        left,
                        if tr == b'C' { Conj::Yes } else { Conj::No },
                        right,
                        Conj::No,
                        -T::o(),
                        par,
                    );
                });
            }
            swap(j);
        }
    }
}

macro_rules! exports {
    ($p:ident,$t:ty,$i:ty,$suffix:tt) => {
        paste::paste! {
            #[no_mangle]
            pub unsafe extern "C" fn [<$p gbtrf $suffix>](
                m: *const $i, n: *const $i, kl: *const $i, ku: *const $i,
                ab: *mut $t, ld: *const $i, piv: *mut $i, info: *mut $i,
            ) {
                *info = 0;
                let (m,n,kl,ku,ld) = (*m as isize,*n as isize,*kl as isize,*ku as isize,*ld as isize);
                for (arg, valid) in [(1,m>=0),(2,n>=0),(3,kl>=0),(4,ku>=0)] {
                    if !valid {
                        crate::lapack::bad(&mut *info,arg);
                        return;
                    }
                }
                let required = kl.checked_mul(2).and_then(|v|v.checked_add(ku)).and_then(|v|v.checked_add(1));
                if !required.is_some_and(|required|ld>=required) {
                    crate::lapack::bad(&mut *info,6);
                    return;
                }
                if m==0 || n==0 { return; }
                let (p,err)=factor(m as usize,n as usize,kl as usize,ku as usize,ab,ld as usize);
                for (j, p) in p.iter().enumerate() {
                    *piv.add(j) = (*p+1) as $i;
                }
                *info=err as $i;
            }
            #[no_mangle]
            pub unsafe extern "C" fn [<$p gbtf2 $suffix>](
                m: *const $i, n: *const $i, kl: *const $i, ku: *const $i,
                ab: *mut $t, ld: *const $i, piv: *mut $i, info: *mut $i,
            ) {
                [<$p gbtrf $suffix>](m,n,kl,ku,ab,ld,piv,info);
            }
            #[no_mangle]
            pub unsafe extern "C" fn [<$p gbtrs $suffix>](
                tr: *const u8, n: *const $i, kl: *const $i, ku: *const $i,
                nrhs: *const $i, ab: *const $t, ld: *const $i, piv: *const $i,
                b: *mut $t, ldb: *const $i, info: *mut $i,
            ) {
                *info=0;
                let (tr,n,kl,ku,nrhs,ld,ldb)=(crate::kernels::flag(tr),*n as isize,*kl as isize,*ku as isize,*nrhs as isize,*ld as isize,*ldb as isize);
                let required=kl.checked_mul(2).and_then(|v|v.checked_add(ku)).and_then(|v|v.checked_add(1));
                for (arg,valid) in [
                    (1,crate::kernels::trans(tr)),(2,n>=0),(3,kl>=0),(4,ku>=0),
                    (5,nrhs>=0),(7,required.is_some_and(|required|ld>=required)),
                    (10,ldb>=n.max(1)),
                ] {
                    if !valid { crate::lapack::bad(&mut *info,arg);return; }
                }
                if n==0 || nrhs==0 { return; }
                let pivots: Vec<usize>=(0..n as usize).map(|j|(*piv.add(j) as usize)-1).collect();
                solve(n as usize,kl as usize,ku as usize,ab,ld as usize,&pivots,b,ldb as usize,nrhs as usize,tr);
            }
            #[no_mangle]
            pub unsafe extern "C" fn [<$p gbsv $suffix>](
                n: *const $i, kl: *const $i, ku: *const $i, nrhs: *const $i,
                ab: *mut $t, ld: *const $i, piv: *mut $i, b: *mut $t,
                ldb: *const $i, info: *mut $i,
            ) {
                *info=0;
                let required=(*kl as isize).checked_mul(2).and_then(|v|v.checked_add(*ku as isize)).and_then(|v|v.checked_add(1));
                for (arg,valid) in [
                    (1,*n>=0),(2,*kl>=0),(3,*ku>=0),(4,*nrhs>=0),
                    (6,required.is_some_and(|required|*ld as isize>=required)),
                    (9,*ldb>=(*n).max(1)),
                ] {
                    if !valid { crate::lapack::bad(&mut *info,arg);return; }
                }
                [<$p gbtrf $suffix>](n,n,kl,ku,ab,ld,piv,info);
                if *info==0 {
                    [<$p gbtrs $suffix>](b"N".as_ptr(),n,kl,ku,nrhs,ab,ld,piv,b,ldb,info);
                }
            }
        }
    };
}
exports!(s, f32, i32, _);
exports!(d, f64, i32, _);
exports!(c, colmat::c32, i32, _);
exports!(z, colmat::c64, i32, _);
exports!(s, f32, i64, _64_);
exports!(d, f64, i64, _64_);
exports!(c, colmat::c32, i64, _64_);
exports!(z, colmat::c64, i64, _64_);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_band_does_not_access_storage() {
        let (p, info) = unsafe { factor::<f64>(0, 7, 2, 3, std::ptr::null_mut(), 8) };
        assert!(p.is_empty());
        assert_eq!(info, 0);
    }

    #[test]
    fn complex_pivot_uses_component_one_norm_and_first_ties() {
        use colmat::c64;
        // |1+i|_1=2 exceeds |1.5|_1, even though its modulus is smaller.
        let mut ab = [c64::new(99.0, 0.0); 3];
        ab[1] = c64::new(1.5, 0.0);
        ab[2] = c64::new(1.0, 1.0);
        let (p, info) = unsafe { factor(2, 1, 1, 0, ab.as_mut_ptr(), 3) };
        assert_eq!(p, [1]);
        assert_eq!(info, 0);
        assert_eq!(ab[1], c64::new(1.0, 1.0));
        assert_eq!(ab[2], c64::new(0.75, -0.75));
        ab[1] = c64::new(2.0, 0.0);
        ab[2] = c64::new(1.0, 1.0);
        let (p, _) = unsafe { factor(2, 1, 1, 0, ab.as_mut_ptr(), 3) };
        assert_eq!(p, [0]);
    }

    #[test]
    fn fortran_entries_preserve_integer_width_and_argument_precedence() {
        unsafe {
            let mut info = 17i64;
            let mut pivot = 123i64;
            dgbtrf_64_(
                &((1i64 << 32) + 1),
                &0,
                &0,
                &0,
                std::ptr::null_mut(),
                &1,
                &mut pivot,
                &mut info,
            );
            assert_eq!(info, 0);
            assert_eq!(pivot, 123);
            dgbtf2_64_(
                &(-(1i64 << 32) + 1),
                &-1,
                &-1,
                &-1,
                std::ptr::null_mut(),
                &0,
                &mut pivot,
                &mut info,
            );
            assert_eq!(info, -1);
            assert_eq!(crate::colmat_blas_last_error(), 1);
            dgbtrf_64_(
                &0,
                &0,
                &i64::MAX,
                &0,
                std::ptr::null_mut(),
                &i64::MAX,
                &mut pivot,
                &mut info,
            );
            assert_eq!(info, -6);
            assert_eq!(crate::colmat_blas_last_error(), 6);
        }
    }

    #[test]
    fn tridiagonal_boundary_pivot_keeps_fill_and_padding() {
        // A = [0 2 0; 3 4 5; 0 6 7], LDAB=6 with two padding rows.
        let mut ab = vec![99.0f64; 18];
        ab[2] = 0.0;
        ab[3] = 3.0;
        ab[7] = 2.0;
        ab[8] = 4.0;
        ab[9] = 6.0;
        ab[13] = 5.0;
        ab[14] = 7.0;
        let (p, info) = unsafe { factor(3, 3, 1, 1, ab.as_mut_ptr(), 6) };
        assert_eq!(info, 0);
        assert_eq!(p, vec![1, 2, 2]);
        assert_eq!(ab[2], 3.0);
        assert_eq!(ab[3], 0.0);
        assert_eq!(ab[7], 4.0);
        assert_eq!(ab[8], 6.0);
        assert!((ab[9] - 1.0 / 3.0).abs() < 1e-15);
        assert_eq!(ab[12], 5.0);
        assert_eq!(ab[13], 7.0);
        assert!((ab[14] + 7.0 / 3.0).abs() < 1e-15);
        for j in 0..3 {
            assert_eq!(&ab[j * 6 + 4..j * 6 + 6], &[99.0, 99.0]);
        }
    }

    #[test]
    fn rectangular_factors_reconstruct_with_interleaved_swaps() {
        for m in 1usize..8 {
            for n in 1usize..8 {
                for kl in 0usize..5 {
                    for ku in 0usize..5 {
                        let kv = kl + ku;
                        let ld = kl + kv + 3;
                        let mut original = vec![0.0f64; m * n];
                        let mut ab = vec![f64::NAN; ld * n];
                        for j in 0..n {
                            for i in j.saturating_sub(ku)..m.min(j + kl + 1) {
                                let value = ((i * 7 + j * 3 + 1) % 13) as f64 - 6.0;
                                original[i + j * m] = value;
                                ab[kv + i - j + j * ld] = value;
                            }
                        }
                        let (p, _) = unsafe { factor(m, n, kl, ku, ab.as_mut_ptr(), ld) };
                        let mut rebuilt = vec![0.0; m * n];
                        for j in 0..n {
                            for i in j.saturating_sub(kv)..m.min(j + 1) {
                                rebuilt[i + j * m] = ab[kv + i - j + j * ld];
                            }
                        }
                        // Undo each elimination and its row swap in reverse.
                        // Applying a single dense permutation to packed L is
                        // insufficient: the saved L columns were not swapped.
                        for j in (0..m.min(n)).rev() {
                            for i in (j + 1)..m.min(j + kl + 1) {
                                let multiplier = ab[kv + i - j + j * ld];
                                for c in j..n {
                                    rebuilt[i + c * m] += multiplier * rebuilt[j + c * m];
                                }
                            }
                            for c in j..n {
                                rebuilt.swap(j + c * m, p[j] + c * m);
                            }
                        }
                        for (actual, expected) in rebuilt.iter().zip(&original) {
                            assert!(
                                (actual - expected).abs() < 1e-10,
                                "m={m}, n={n}, kl={kl}, ku={ku}: {actual} != {expected}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn band_solve_recovers_known_rhs_for_all_transposes() {
        for n in 1usize..8 {
            for kl in 0usize..4 {
                for ku in 0usize..4 {
                    let kv = kl + ku;
                    let ld = kl + kv + 2;
                    let ldb = n + 2;
                    let mut original = vec![0.0f64; n * n];
                    let mut ab = vec![f64::NAN; ld * n];
                    for j in 0..n {
                        for i in j.saturating_sub(ku)..n.min(j + kl + 1) {
                            let value = if i == j {
                                12.0
                            } else {
                                (i as f64 - j as f64) / 4.0
                            };
                            original[i + j * n] = value;
                            ab[kv + i - j + j * ld] = value;
                        }
                    }
                    let (p, info) = unsafe { factor(n, n, kl, ku, ab.as_mut_ptr(), ld) };
                    assert_eq!(info, 0);
                    for tr in [b'N', b'T', b'C'] {
                        let mut b = vec![999.0; ldb * 2];
                        for c in 0..2 {
                            for i in 0..n {
                                b[i + c * ldb] = (0..n)
                                    .map(|j| {
                                        let a = if tr == b'N' {
                                            original[i + j * n]
                                        } else {
                                            original[j + i * n]
                                        };
                                        a * (1.0 + j as f64 / 8.0 + c as f64)
                                    })
                                    .sum();
                            }
                        }
                        unsafe {
                            solve(n, kl, ku, ab.as_ptr(), ld, &p, b.as_mut_ptr(), ldb, 2, tr);
                        }
                        for c in 0..2 {
                            for i in 0..n {
                                let expected = 1.0 + i as f64 / 8.0 + c as f64;
                                assert!((b[i + c * ldb] - expected).abs() < 1e-12);
                            }
                            assert_eq!(&b[n + c * ldb..(c + 1) * ldb], &[999.0, 999.0]);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn complex_band_solve_reverses_pivots_and_conjugates_both_factors() {
        use colmat::c64;
        let z = c64::new(0.0, 0.0);
        let original = [
            z,
            c64::new(3.0, 2.0),
            z,
            c64::new(2.0, -1.0),
            c64::new(4.0, 1.0),
            c64::new(6.0, -2.0),
            z,
            c64::new(5.0, 1.0),
            c64::new(7.0, 3.0),
        ];
        let mut ab = [c64::new(f64::NAN, f64::NAN); 12];
        for j in 0usize..3 {
            for i in j.saturating_sub(1)..3.min(j + 2) {
                ab[2 + i - j + j * 4] = original[i + j * 3];
            }
        }
        let (p, info) = unsafe { factor(3, 3, 1, 1, ab.as_mut_ptr(), 4) };
        assert_eq!(info, 0);
        assert_eq!(p[0], 1);
        let x = [c64::new(1.0, 2.0), c64::new(3.0, -1.0), c64::new(-2.0, 1.0)];
        for tr in [b'N', b'T', b'C'] {
            let mut b = [z; 3];
            for i in 0..3 {
                for j in 0..3 {
                    let a = if tr == b'N' {
                        original[i + j * 3]
                    } else {
                        original[j + i * 3]
                    };
                    b[i] += if tr == b'C' { a.cj() } else { a } * x[j];
                }
            }
            unsafe {
                solve(3, 1, 1, ab.as_ptr(), 4, &p, b.as_mut_ptr(), 3, 1, tr);
            }
            for i in 0..3 {
                assert!((b[i] - x[i]).magnitude() < 1e-12);
            }
        }
    }
}
