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
        if original == T::z() {
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
