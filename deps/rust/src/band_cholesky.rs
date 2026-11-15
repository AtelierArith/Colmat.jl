//! Band and split Cholesky from native diagonal LLT, triangular solves,
//! and selected-triangle rank-one products over the stored band.
use crate::generalized_selfadjoint::GeneralizedScalar;
use colmat::{Accum, Conj, Mat, MatMut, MatRef};

unsafe fn step<T: GeneralizedScalar>(
    lower: bool,
    width: usize,
    j: usize,
    first: usize,
    count: usize,
    backwards: bool,
    ab: *mut T,
    ld: usize,
) -> bool {
    let row = if lower { 0 } else { width };
    let diagonal = ab.add(row + j * ld);
    // Hermitian diagonal imaginary components are not input data.
    *diagonal = T::diagonal(diagonal);
    let nan = (*diagonal).re().is_nan();
    let status = crate::lapack::cholesky(MatMut::from_raw_parts_mut(diagonal, 1, 1, 1, 1));
    // Reference PBTF2/PBSTF do not classify NaNs as non-positive pivots.
    if status != 0 && !nan {
        return false;
    }
    if nan {
        *diagonal = T::real(f64::NAN);
    }
    if count == 0 {
        return true;
    }
    let (vector, increment) = match (lower, backwards) {
        (true, false) => (ab.add(1 + j * ld), 1),
        (false, false) => (ab.add(width - 1 + (j + 1) * ld), ld - 1),
        (false, true) => (ab.add(width - count + j * ld), 1),
        (true, true) => (ab.add(count + first * ld), ld - 1),
    };
    crate::threading::with_par(|par| {
        colmat::linalg::triangular_solve::solve_lower_triangular_in_place(
            MatRef::from_raw_parts(diagonal, 1, 1, 1, 1),
            MatMut::from_raw_parts_mut(vector, 1, count, 0, increment as isize),
            par,
        );
    });
    // The unused half of a skew-strided target view can overlap the stored
    // vector. Snapshot it before constructing the mutable target, keeping
    // the native product's input/output views disjoint even there.
    let vector = Mat::from_fn(count, 1, |i, _| *vector.add(i * increment));
    let target =
        MatMut::from_raw_parts_mut(ab.add(row + first * ld), count, count, 1, (ld - 1) as isize);
    let conjugate_left = lower == backwards;
    use colmat::linalg::matmul::triangular::{matmul_with_conj, BlockStructure::*};
    crate::threading::with_par(|par| {
        matmul_with_conj(
            target,
            if lower {
                TriangularLower
            } else {
                TriangularUpper
            },
            Accum::Add,
            vector.as_ref(),
            Rectangular,
            if conjugate_left { Conj::Yes } else { Conj::No },
            vector.as_ref().transpose(),
            Rectangular,
            if conjugate_left { Conj::No } else { Conj::Yes },
            -T::o(),
            par,
        );
    });
    // HER makes every updated diagonal real even if the rank-one vector is
    // zero. Preserve that partial-output contract before a later pivot fails.
    if T::COMPLEX {
        for i in first..first + count {
            let diagonal = ab.add(row + i * ld);
            *diagonal = T::diagonal(diagonal);
        }
    }
    true
}
pub(crate) unsafe fn factor<T: GeneralizedScalar>(
    lower: bool,
    n: usize,
    width: usize,
    ab: *mut T,
    ld: usize,
    split: bool,
) -> usize {
    if n == 0 {
        return 0;
    }
    // With a redundant declared width the mathematical split cannot exceed
    // n. In particular, never index a nonexistent column when width>n.
    let middle = if split {
        (n / 2 + width / 2 + (n % 2 + width % 2) / 2).min(n)
    } else {
        n
    };
    if split {
        for j in (middle..n).rev() {
            let count = width.min(j);
            if !step(lower, width, j, j - count, count, true, ab, ld) {
                return j + 1;
            }
        }
    }
    for j in 0..middle {
        let count = width.min(middle - j - 1);
        if !step(lower, width, j, j + 1, count, false, ab, ld) {
            return j + 1;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::householder::ReflectorScalar;

    fn reconstruction<T: GeneralizedScalar + ReflectorScalar>(tolerance: f64) {
        for n in 0usize..9 {
            for width in 0usize..10 {
                for split in [false, true] {
                    let middle = if split { ((n + width) / 2).min(n) } else { n };
                    let mut source = vec![T::z(); n * n];
                    for i in 0..n {
                        let (first, end) = if split && i >= middle {
                            (i.saturating_sub(width), i + 1)
                        } else {
                            (i, middle.min(i + width + 1))
                        };
                        for j in first..end {
                            source[i + j * n] = if i == j {
                                T::real(2.0 + i as f64 / 8.0)
                            } else {
                                T::parts(
                                    (i + j + 1) as f64 / 16.0,
                                    if T::COMPLEX {
                                        (i as f64 - j as f64) / 32.0
                                    } else {
                                        0.0
                                    },
                                )
                            };
                        }
                    }
                    let mut original = vec![T::z(); n * n];
                    for j in 0..n {
                        for i in 0..n {
                            for k in 0..n {
                                original[i + j * n] = original[i + j * n]
                                    + source[k + i * n].cj() * source[k + j * n];
                            }
                        }
                    }
                    for lower in [false, true] {
                        let ld = width + 3;
                        let mut ab = vec![T::parts(f64::NAN, f64::NAN); ld * n];
                        let mut selected = vec![false; ld * n];
                        for j in 0..n {
                            ab[ld - 2 + j * ld] = T::real(777.0);
                            ab[ld - 1 + j * ld] = T::real(777.0);
                            let (first, end) = if lower {
                                (j, n.min(j + width + 1))
                            } else {
                                (j.saturating_sub(width), j + 1)
                            };
                            for i in first..end {
                                let row = if lower { i - j } else { width + i - j };
                                ab[row + j * ld] = if T::COMPLEX && i == j {
                                    T::parts(original[i + j * n].re(), f64::NAN)
                                } else {
                                    original[i + j * n]
                                };
                                selected[row + j * ld] = true;
                            }
                        }
                        assert_eq!(
                            unsafe { factor(lower, n, width, ab.as_mut_ptr(), ld, split) },
                            0
                        );
                        let mut actual = vec![T::z(); n * n];
                        for j in 0..n {
                            for row in 0..width + 1 {
                                if !selected[row + j * ld] {
                                    assert!(ab[row + j * ld].re().is_nan());
                                    continue;
                                }
                                let i = if lower { j + row } else { j + row - width };
                                let value = ab[row + j * ld];
                                if split {
                                    if lower {
                                        if i >= middle {
                                            actual[i + j * n] = value;
                                        } else {
                                            actual[j + i * n] = value.cj();
                                        }
                                    } else if j >= middle {
                                        actual[j + i * n] = value.cj();
                                    } else {
                                        actual[i + j * n] = value;
                                    }
                                } else {
                                    actual[i + j * n] = value;
                                }
                            }
                            assert_eq!(ab[ld - 2 + j * ld], T::real(777.0));
                            assert_eq!(ab[ld - 1 + j * ld], T::real(777.0));
                        }
                        for j in 0..n {
                            for i in 0..n {
                                let mut value = T::z();
                                for k in 0..n {
                                    value = value
                                        + if split || !lower {
                                            actual[k + i * n].cj() * actual[k + j * n]
                                        } else {
                                            actual[i + k * n] * actual[j + k * n].cj()
                                        };
                                }
                                assert!(
                                    (value - original[i + j * n]).magnitude() < tolerance,
                                    "n={n} width={width} lower={lower} split={split}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn real32_band_and_split_reconstruct() {
        reconstruction::<f32>(5e-4);
    }
    #[test]
    fn real64_band_and_split_reconstruct() {
        reconstruction::<f64>(1e-10);
    }
    #[test]
    fn complex32_band_and_split_reconstruct() {
        reconstruction::<colmat::c32>(5e-4);
    }
    #[test]
    fn complex64_band_and_split_reconstruct() {
        reconstruction::<colmat::c64>(1e-10);
    }
    #[test]
    fn diagonal_only_and_empty_storage() {
        unsafe {
            assert_eq!(factor::<f64>(true, 0, 7, std::ptr::null_mut(), 8, true), 0);
            let mut ab = [4.0, 9.0, 16.0];
            assert_eq!(factor(true, 3, 0, ab.as_mut_ptr(), 1, false), 0);
            assert_eq!(ab, [2.0, 3.0, 4.0]);
            let mut ab = [4.0, 9.0, 16.0];
            assert_eq!(factor(true, 3, 0, ab.as_mut_ptr(), 1, true), 0);
            assert_eq!(ab, [2.0, 3.0, 4.0]);
        }
    }
    #[test]
    fn split_failure_uses_reverse_pivot_order() {
        unsafe {
            let mut ab = [4.0, -9.0, 16.0];
            assert_eq!(factor(true, 3, 0, ab.as_mut_ptr(), 1, true), 2);
            assert_eq!(ab, [4.0, -9.0, 4.0]);
            let mut ab = [4.0, -9.0, 16.0];
            assert_eq!(factor(true, 3, 0, ab.as_mut_ptr(), 1, false), 2);
            assert_eq!(ab, [2.0, -9.0, 16.0]);
        }
    }
    #[test]
    fn hermitian_diagonal_ignores_nan_imaginary_component() {
        unsafe {
            let mut ab = [colmat::c64::new(4.0, f64::NAN)];
            assert_eq!(factor(true, 1, 0, ab.as_mut_ptr(), 1, false), 0);
            assert_eq!(ab, [colmat::c64::new(2.0, 0.0)]);
        }
    }
}

unsafe fn solve<T: GeneralizedScalar>(
    lower: bool,
    n: usize,
    width: usize,
    ab: *const T,
    ld: usize,
    b: *mut T,
    ldb: usize,
    nrhs: usize,
) {
    for c in 0..nrhs {
        let x = b.add(c * ldb);
        if lower {
            crate::structured_triangular::solve_lower_band(n, width, ab, ld, x, b'N');
            crate::structured_triangular::solve_lower_band(n, width, ab, ld, x, b'C');
        } else {
            crate::structured_triangular::solve_upper_band(n, width, ab, ld, x, b'C');
            crate::structured_triangular::solve_upper_band(n, width, ab, ld, x, b'N');
        }
    }
}
macro_rules! valid {($info:expr,$($arg:expr=>$test:expr),+)=>{$(if !$test{crate::lapack::bad(&mut *$info,$arg);return;})+};}
macro_rules! factor_export {($p:ident,$name:ident,$t:ty,$i:ty,$s:tt,$split:expr)=>{paste::paste!{
    #[no_mangle]
    pub unsafe extern "C" fn [<$p $name $s>](u:*const u8,n:*const $i,k:*const $i,ab:*mut $t,ld:*const $i,info:*mut $i) {
        *info=0;let u=crate::kernels::flag(u);
        valid!(info,1=>crate::kernels::uplo(u),2=>*n>=0,3=>*k>=0,
            5=>(*k as isize).checked_add(1).is_some_and(|required|*ld as isize>=required));
        if *n==0{return;}
        *info=factor(u==b'L',*n as usize,*k as usize,ab,*ld as usize,$split)as $i;
    }
}};}
macro_rules! exports {($p:ident,$t:ty,$i:ty,$s:tt)=>{
    factor_export!($p,pbtrf,$t,$i,$s,false);
    factor_export!($p,pbtf2,$t,$i,$s,false);
    factor_export!($p,pbstf,$t,$i,$s,true);
    paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p pbtrs $s>](u:*const u8,n:*const $i,k:*const $i,nrhs:*const $i,ab:*const $t,ld:*const $i,b:*mut $t,ldb:*const $i,info:*mut $i) {
            *info=0;let u=crate::kernels::flag(u);
            valid!(info,1=>crate::kernels::uplo(u),2=>*n>=0,3=>*k>=0,4=>*nrhs>=0,
                6=>(*k as isize).checked_add(1).is_some_and(|required|*ld as isize>=required),8=>*ldb>=(*n).max(1));
            if *n==0||*nrhs==0{return;}
            solve(u==b'L',*n as usize,*k as usize,ab,*ld as usize,b,*ldb as usize,*nrhs as usize);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p pbsv $s>](u:*const u8,n:*const $i,k:*const $i,nrhs:*const $i,ab:*mut $t,ld:*const $i,b:*mut $t,ldb:*const $i,info:*mut $i) {
            *info=0;let flag=crate::kernels::flag(u);
            valid!(info,1=>crate::kernels::uplo(flag),2=>*n>=0,3=>*k>=0,4=>*nrhs>=0,
                6=>(*k as isize).checked_add(1).is_some_and(|required|*ld as isize>=required),8=>*ldb>=(*n).max(1));
            [<$p pbtrf $s>](u,n,k,ab,ld,info);
            if *info==0{[<$p pbtrs $s>](u,n,k,nrhs,ab,ld,b,ldb,info);}
        }
    }
};}
exports!(s, f32, i32, _);
exports!(d, f64, i32, _);
exports!(c, colmat::c32, i32, _);
exports!(z, colmat::c64, i32, _);
exports!(s, f32, i64, _64_);
exports!(d, f64, i64, _64_);
exports!(c, colmat::c32, i64, _64_);
exports!(z, colmat::c64, i64, _64_);

#[cfg(test)]
mod solve_tests {
    use super::*;
    use crate::kernels::Scalar;
    #[test]
    fn zero_rank_update_cleans_unvisited_hermitian_diagonals() {
        use colmat::c64;
        let mut a = [c64::new(f64::NAN, f64::NAN); 9];
        a[0] = c64::new(1.0, f64::NAN);
        a[1] = c64::new(0.0, 0.0);
        a[2] = c64::new(0.0, 0.0);
        a[3] = c64::new(-1.0, f64::NAN);
        a[4] = c64::new(0.0, 0.0);
        a[6] = c64::new(1.0, f64::NAN);
        assert_eq!(unsafe { factor(true, 3, 2, a.as_mut_ptr(), 3, false) }, 2);
        assert_eq!(a[6], c64::new(1.0, 0.0));
    }
    #[test]
    fn complex_band_solve_uses_both_conjugated_and_unconjugated_factors() {
        use colmat::c64;
        let a = [
            c64::new(4.0, 0.0),
            c64::new(1.0, -2.0),
            c64::new(0.0, 0.0),
            c64::new(1.0, 2.0),
            c64::new(8.0, 0.0),
            c64::new(2.0, -1.0),
            c64::new(0.0, 0.0),
            c64::new(2.0, 1.0),
            c64::new(9.0, 0.0),
        ];
        let x = [c64::new(1.0, 1.0), c64::new(2.0, -1.0), c64::new(-1.0, 2.0)];
        for lower in [false, true] {
            let mut ab = [c64::new(f64::NAN, f64::NAN); 9];
            let mut b = [c64::new(0.0, 0.0); 3];
            for j in 0usize..3 {
                ab[2 + j * 3] = c64::new(777.0, 0.0);
                for i in 0..3 {
                    b[i] += a[i + j * 3] * x[j];
                }
                let (first, end) = if lower {
                    (j, 3.min(j + 2))
                } else {
                    (j.saturating_sub(1), j + 1)
                };
                for i in first..end {
                    ab[(if lower { i - j } else { 1 + i - j }) + j * 3] = a[i + j * 3];
                }
            }
            unsafe {
                assert_eq!(factor(lower, 3, 1, ab.as_mut_ptr(), 3, false), 0);
                solve(lower, 3, 1, ab.as_ptr(), 3, b.as_mut_ptr(), 3, 1);
            }
            for i in 0..3 {
                assert!((b[i] - x[i]).magnitude() < 1e-12);
            }
            for j in 0..3 {
                assert_eq!(ab[2 + j * 3], c64::new(777.0, 0.0));
            }
        }
    }
    #[test]
    fn nan_pivots_propagate_in_ordinary_and_split_contracts() {
        unsafe {
            let mut a = [f64::NAN];
            assert_eq!(factor(true, 1, 0, a.as_mut_ptr(), 1, false), 0);
            assert_eq!(factor(true, 1, 0, a.as_mut_ptr(), 1, true), 0);
            assert!(a[0].is_nan());
        }
    }
}
