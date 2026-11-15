//! Positive tridiagonal LDL adaptation. Normalize each native LDL panel by
//! a power of two, so a representable tiny pivot does not need an overflowing
//! reciprocal. Update the real D through native real products, including
//! complex E's two real components.
use crate::generalized_selfadjoint::GeneralizedScalar;
use crate::householder::ReflectorScalar;
use crate::kernels::Scalar;
use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::{Accum, Conj, Mat, MatMut, MatRef};

pub(crate) trait PtReal: Scalar {
    const EXPONENT_CHUNK: i32;
}
impl PtReal for f32 {
    const EXPONENT_CHUNK: i32 = 126;
}
impl PtReal for f64 {
    const EXPONENT_CHUNK: i32 = 1022;
}
fn scale_pow2<R: PtReal>(mut value: R, mut shift: i32) -> R {
    while shift != 0 {
        let step = shift.clamp(-R::EXPONENT_CHUNK, R::EXPONENT_CHUNK);
        value = value * R::real(2.0f64.powi(step));
        shift -= step;
    }
    value
}
fn exponent(value: f64) -> i32 {
    let bits = value.to_bits();
    let field = ((bits >> 52) & 0x7ff) as i32;
    if field == 0 {
        63 - (bits & ((1u64 << 52) - 1)).leading_zeros() as i32 - 1074
    } else {
        field - 1023
    }
}
pub(crate) unsafe fn factor<T: GeneralizedScalar + ReflectorScalar, R: PtReal>(
    n: usize,
    d: *mut R,
    e: *mut T,
) -> usize {
    if n == 0 {
        return 0;
    }
    for j in 0..n - 1 {
        let pivot = *d.add(j);
        if pivot.re() <= 0.0 {
            return j + 1;
        }
        let original = *e.add(j);
        if original == T::z() && !pivot.re().is_nan() {
            continue;
        }
        let (normalized_d, normalized_e) = if pivot.re().is_finite() {
            // A normalized pivot in [1/2,1) also prevents a finite multiplier
            // near MAX from overflowing during normalization itself.
            let shift = -(exponent(pivot.re()) + 1);
            let diagonal = scale_pow2(pivot, shift);
            let re = scale_pow2(R::real(original.re()), shift);
            let im = scale_pow2(R::real(original.im()), shift);
            (T::real(diagonal.re()), T::parts(re.re(), im.re()))
        } else {
            // The native LDL rejects nonfinite diagonals. Preserve PTTRF's
            // IEEE component division at this boundary, then still run the
            // native panel and real Schur update below.
            let re = R::real(original.re()) / pivot;
            let im = R::real(original.im()) / pivot;
            (T::o(), T::parts(re.re(), im.re()))
        };
        let mut panel = Mat::<T>::from_fn(2, 2, |i, k| match (i, k) {
            (0, 0) => normalized_d,
            (1, 0) => normalized_e,
            _ => T::z(),
        });
        use colmat::linalg::cholesky::ldlt::factor::{
            cholesky_in_place, cholesky_in_place_scratch, LdltError,
        };
        let status = crate::threading::with_par(|par| {
            let mut mem = MemBuffer::new(cholesky_in_place_scratch::<T>(
                2,
                par.shape(),
                Default::default(),
            ));
            cholesky_in_place(
                panel.as_mut(),
                Default::default(),
                par,
                MemStack::new(&mut mem),
                Default::default(),
            )
        });
        if matches!(status, Err(LdltError::ZeroPivot { index: 0 })) {
            return j + 1;
        }
        // Only the completed first panel column is used. Its dummy trailing
        // pivot can be zero/nonfinite; that diagnostic is not a PT pivot.
        let multiplier = panel[(1, 0)];
        *e.add(j) = multiplier;
        let left = [R::real(multiplier.re()), R::real(multiplier.im())];
        let right = [R::real(original.re()), R::real(original.im())];
        crate::threading::with_par(|par| {
            colmat::linalg::matmul::matmul_with_conj(
                MatMut::from_raw_parts_mut(d.add(j + 1), 1, 1, 1, 1),
                Accum::Add,
                MatRef::from_raw_parts(left.as_ptr(), 1, 2, 0, 1),
                Conj::No,
                MatRef::from_raw_parts(right.as_ptr(), 2, 1, 1, 0),
                Conj::No,
                -R::o(),
                par,
            );
        });
    }
    if (*d.add(n - 1)).re() <= 0.0 {
        n
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn real_ldl_and_nonpositive_prefix() {
        unsafe {
            let mut d = [4.0f64, 3.0, 2.0];
            let mut e = [2.0, 1.0];
            assert_eq!(factor(3, d.as_mut_ptr(), e.as_mut_ptr()), 0);
            assert_eq!(d, [4.0, 2.0, 1.5]);
            assert_eq!(e, [0.5, 0.5]);
            let mut d = [2.0f64, 1.0, 9.0];
            let mut e = [2.0, 7.0];
            assert_eq!(factor(3, d.as_mut_ptr(), e.as_mut_ptr()), 2);
            assert_eq!(d, [2.0, -1.0, 9.0]);
            assert_eq!(e, [1.0, 7.0]);
        }
    }
    #[test]
    fn complex_e_keeps_d_real() {
        unsafe {
            let mut d = [4.0f64, 3.0];
            let mut e = [colmat::c64::new(1.0, 2.0)];
            assert_eq!(factor(2, d.as_mut_ptr(), e.as_mut_ptr()), 0);
            assert_eq!(e, [colmat::c64::new(0.25, 0.5)]);
            assert_eq!(d, [4.0, 1.75]);
        }
    }
    #[test]
    fn minimum_subnormal_pivot_has_finite_multiplier() {
        unsafe {
            let tiny = f64::from_bits(1);
            let mut d = [tiny, 2.0 * tiny];
            let mut e = [tiny];
            assert_eq!(factor(2, d.as_mut_ptr(), e.as_mut_ptr()), 0);
            assert_eq!(e, [1.0]);
            assert_eq!(d, [tiny, tiny]);
            let tiny = f32::from_bits(1);
            let mut d = [tiny, 2.0 * tiny];
            let mut e = [tiny];
            assert_eq!(factor(2, d.as_mut_ptr(), e.as_mut_ptr()), 0);
            assert_eq!(e, [1.0]);
            assert_eq!(d, [tiny, tiny]);
        }
    }
    #[test]
    fn huge_multiplier_does_not_overflow_during_normalization() {
        unsafe {
            let tiny = 3.0 * f64::from_bits(1);
            let mut d = [tiny, 1.0e308];
            let mut e = [1.0e-15f64];
            assert_eq!(factor(2, d.as_mut_ptr(), e.as_mut_ptr()), 0);
            assert!(e[0].is_finite());
            assert!((e[0] / (1.0e-15 / tiny) - 1.0).abs() < 1e-14);
            assert!(d[1] > 0.0 && d[1].is_finite());
        }
    }
    #[test]
    fn empty_or_first_nonpositive_pivot_reads_no_e() {
        unsafe {
            assert_eq!(
                factor::<f64, f64>(0, std::ptr::null_mut(), std::ptr::null_mut()),
                0
            );
            let mut d = [-1.0f64, 1.0];
            assert_eq!(
                factor::<f64, f64>(2, d.as_mut_ptr(), std::ptr::null_mut()),
                1
            );
        }
    }
}

fn band_fits<T: Scalar>(n: usize) -> bool {
    n.checked_mul(2)
        .and_then(|count| count.checked_mul(std::mem::size_of::<T>()))
        .is_some_and(|bytes| bytes <= isize::MAX as usize)
}
fn stride_fits<T: Scalar>(nrhs: usize, ldb: isize) -> bool {
    nrhs <= 1 || ldb.checked_mul(if T::COMPLEX { 2 } else { 1 }).is_some()
}
// Canonical T/R pairs are (f32,f32),(f64,f64),(c32,f32),(c64,f64).
unsafe fn solve<T: GeneralizedScalar + ReflectorScalar, R: PtReal>(
    n: usize,
    nrhs: usize,
    upper: bool,
    d: *const R,
    e: *const T,
    b: *mut T,
    ldb: usize,
) {
    if n == 0 || nrhs == 0 {
        return;
    }
    let mut band = vec![T::z(); 2 * n];
    for j in 0..n - 1 {
        band[2 * j + 1] = if upper { (*e.add(j)).cj() } else { *e.add(j) };
    }
    for c in 0..nrhs {
        crate::structured_triangular::solve_unit_lower_band(
            n,
            1,
            band.as_ptr(),
            2,
            b.add(c * ldb),
            b'N',
        );
    }
    let components = if T::COMPLEX { 2 } else { 1 };
    let raw = b.cast::<R>();
    let column_stride = if nrhs == 1 { 0 } else { ldb * components };
    for j in 0..n {
        let pivot = *d.add(j);
        let normalized = if pivot.re().is_finite() && pivot.re() != 0.0 {
            let shift = -(exponent(pivot.re().abs()) + 1);
            for c in 0..nrhs {
                for part in 0..components {
                    let entry = raw.add((j + c * ldb) * components + part);
                    *entry = scale_pow2(*entry, shift);
                }
            }
            scale_pow2(pivot, shift)
        } else {
            for c in 0..nrhs {
                for part in 0..components {
                    let entry = raw.add((j + c * ldb) * components + part);
                    *entry = *entry / pivot;
                }
            }
            R::o()
        };
        for part in 0..components {
            crate::threading::with_par(|par| {
                colmat::linalg::triangular_solve::solve_lower_triangular_in_place(
                    MatRef::from_raw_parts(&normalized, 1, 1, 1, 1),
                    MatMut::from_raw_parts_mut(
                        raw.add(j * components + part),
                        1,
                        nrhs,
                        0,
                        column_stride as isize,
                    ),
                    par,
                );
            });
        }
    }
    for c in 0..nrhs {
        crate::structured_triangular::solve_unit_lower_band(
            n,
            1,
            band.as_ptr(),
            2,
            b.add(c * ldb),
            b'C',
        );
    }
}
macro_rules! valid {($info:expr,$($arg:expr=>$test:expr),+)=>{$(if !$test{crate::lapack::bad(&mut *$info,$arg);return;})+};}
macro_rules! common_exports {($p:ident,$t:ty,$r:ty,$i:ty,$s:tt)=>{paste::paste!{
    #[no_mangle]
    pub unsafe extern "C" fn [<$p pttrf $s>](n:*const $i,d:*mut $r,e:*mut $t,info:*mut $i) {
        *info=0;valid!(info,1=>*n>=0);if *n==0{return;}
        *info=factor(*n as usize,d,e)as $i;
    }
    #[no_mangle]
    pub unsafe extern "C" fn [<$p ptsv $s>](n:*const $i,nrhs:*const $i,d:*mut $r,e:*mut $t,b:*mut $t,ldb:*const $i,info:*mut $i) {
        *info=0;valid!(info,1=>*n>=0,2=>*nrhs>=0,6=>*ldb>=(*n).max(1));
        let(n,k)=(*n as usize,*nrhs as usize);if n==0{return;}
        if k>0 {valid!(info,1=>band_fits::<$t>(n),6=>stride_fits::<$t>(k,*ldb as isize));}
        *info=factor(n,d,e)as $i;
        if *info==0{solve(n,k,false,d,e,b,*ldb as usize);}
    }
}};}
macro_rules! real_exports {($p:ident,$t:ty,$i:ty,$s:tt)=>{
    common_exports!($p,$t,$t,$i,$s);
    paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p pttrs $s>](n:*const $i,nrhs:*const $i,d:*const $t,e:*const $t,b:*mut $t,ldb:*const $i,info:*mut $i) {
            *info=0;valid!(info,1=>*n>=0,2=>*nrhs>=0,6=>*ldb>=(*n).max(1));
            let(n,k)=(*n as usize,*nrhs as usize);if n==0||k==0{return;}
            valid!(info,1=>band_fits::<$t>(n),6=>stride_fits::<$t>(k,*ldb as isize));
            solve(n,k,false,d,e,b,*ldb as usize);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p ptts2 $s>](n:*const $i,nrhs:*const $i,d:*const $t,e:*const $t,b:*mut $t,ldb:*const $i) {
            for(arg,test)in[(1,*n>=0),(2,*nrhs>=0),(6,*ldb>=(*n).max(1))]{if !test{crate::invalid(arg);return;}}
            let(n,k)=(*n as usize,*nrhs as usize);if n==0||k==0{return;}
            for(arg,test)in[(1,band_fits::<$t>(n)),(6,stride_fits::<$t>(k,*ldb as isize))]{if !test{crate::invalid(arg);return;}}
            solve(n,k,false,d,e,b,*ldb as usize);
        }
    }
};}
macro_rules! complex_exports {($p:ident,$t:ty,$r:ty,$i:ty,$s:tt)=>{
    common_exports!($p,$t,$r,$i,$s);
    paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p pttrs $s>](u:*const u8,n:*const $i,nrhs:*const $i,d:*const $r,e:*const $t,b:*mut $t,ldb:*const $i,info:*mut $i) {
            *info=0;let u=crate::kernels::flag(u);valid!(info,1=>crate::kernels::uplo(u),2=>*n>=0,3=>*nrhs>=0,7=>*ldb>=(*n).max(1));
            let(n,k)=(*n as usize,*nrhs as usize);if n==0||k==0{return;}
            valid!(info,2=>band_fits::<$t>(n),7=>stride_fits::<$t>(k,*ldb as isize));
            solve(n,k,u==b'U',d,e,b,*ldb as usize);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p ptts2 $s>](iu:*const $i,n:*const $i,nrhs:*const $i,d:*const $r,e:*const $t,b:*mut $t,ldb:*const $i) {
            for(arg,test)in[(1,matches!(*iu,0|1)),(2,*n>=0),(3,*nrhs>=0),(7,*ldb>=(*n).max(1))]{if !test{crate::invalid(arg);return;}}
            let(n,k)=(*n as usize,*nrhs as usize);if n==0||k==0{return;}
            for(arg,test)in[(2,band_fits::<$t>(n)),(7,stride_fits::<$t>(k,*ldb as isize))]{if !test{crate::invalid(arg);return;}}
            solve(n,k,*iu==1,d,e,b,*ldb as usize);
        }
    }
};}
real_exports!(s, f32, i32, _);
real_exports!(d, f64, i32, _);
complex_exports!(c, colmat::c32, f32, i32, _);
complex_exports!(z, colmat::c64, f64, i32, _);
real_exports!(s, f32, i64, _64_);
real_exports!(d, f64, i64, _64_);
complex_exports!(c, colmat::c32, f32, i64, _64_);
complex_exports!(z, colmat::c64, f64, i64, _64_);

#[cfg(test)]
mod solve_tests {
    use super::*;
    #[test]
    fn nan_pivot_propagates_even_with_zero_off_diagonal() {
        unsafe {
            let mut d = [f64::NAN, 2.0];
            let mut e = [0.0f64];
            assert_eq!(factor(2, d.as_mut_ptr(), e.as_mut_ptr()), 0);
            assert!(d[1].is_nan());
            assert!(e[0].is_nan());
        }
    }
    #[test]
    fn tiny_and_max_diagonal_scaling_remains_representable() {
        unsafe {
            for value in [f64::from_bits(1), f64::MAX] {
                let d = [value];
                let mut b = [value];
                solve::<f64, f64>(
                    1,
                    1,
                    false,
                    d.as_ptr(),
                    std::ptr::null(),
                    b.as_mut_ptr(),
                    usize::MAX,
                );
                assert_eq!(b, [1.0]);
            }
            let tiny = f32::from_bits(1);
            let d = [tiny];
            let mut b = [colmat::c32::new(tiny, -tiny)];
            solve::<colmat::c32, f32>(1, 1, false, d.as_ptr(), std::ptr::null(), b.as_mut_ptr(), 1);
            assert_eq!(b, [colmat::c32::new(1.0, -1.0)]);
        }
    }
    #[test]
    fn complex_upper_and_lower_use_distinct_conjugation() {
        use colmat::c64;
        let d = [2.0f64, 3.0];
        let e = [c64::new(0.25, 0.5)];
        let x = [c64::new(1.0, 2.0), c64::new(-1.0, 1.0)];
        for upper in [false, true] {
            let lower = if upper { e[0].cj() } else { e[0] };
            let a00 = c64::new(d[0], 0.0);
            let a10 = lower * d[0];
            let a11 = c64::new(d[1] + lower.magnitude().powi(2) * d[0], 0.0);
            let mut b = [a00 * x[0] + a10.cj() * x[1], a10 * x[0] + a11 * x[1]];
            unsafe {
                solve(2, 1, upper, d.as_ptr(), e.as_ptr(), b.as_mut_ptr(), 2);
            }
            for i in 0..2 {
                assert!((b[i] - x[i]).magnitude() < 1e-14);
            }
        }
    }
    #[test]
    fn zero_rhs_and_ilp64_argument_checks_need_no_arrays() {
        unsafe {
            let mut info = 777i64;
            let large = (1i64 << 32) + 1;
            zpttrs_64_(
                b"U".as_ptr(),
                &large,
                &0,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null_mut(),
                &large,
                &mut info,
            );
            assert_eq!(info, 0);
            zpttrf_64_(
                &(-large),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut info,
            );
            assert_eq!(info, -1);
            assert_eq!(crate::colmat_blas_last_error(), 1);
            zptts2_64_(
                &2,
                &0,
                &0,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null_mut(),
                &1,
            );
            assert_eq!(crate::colmat_blas_last_error(), 1);
        }
    }
}
