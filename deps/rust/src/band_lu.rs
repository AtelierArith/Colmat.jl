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
}
