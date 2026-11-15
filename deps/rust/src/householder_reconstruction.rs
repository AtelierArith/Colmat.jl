//! Recover LAPACK compact-WY reflectors from orthonormal thin columns.
use crate::kernels::{matrix_addressable, Scalar};
use crate::lapack::LapackInt;
use colmat::{Conj, MatMut, MatRef};
use paste::paste;

pub(crate) unsafe fn reconstruct<T: Scalar>(
    m: usize,
    n: usize,
    nb: usize,
    a: *mut T,
    lda: isize,
    t: *mut T,
    ldt: isize,
    d: *mut T,
) {
    // Modified LU of Q - [D;0]. Choosing D opposite to the real diagonal
    // sign keeps each pivot away from zero for orthonormal input columns.
    for j in 0..n {
        let diagonal = a.offset(j as isize * lda + j as isize);
        let sign = T::real(if (*diagonal).re().is_sign_negative() {
            1.0
        } else {
            -1.0
        });
        *d.add(j) = sign;
        *diagonal = *diagonal - sign;
        let pivot = *diagonal;
        for i in j + 1..m {
            let p = a.offset(i as isize + j as isize * lda);
            *p = *p / pivot;
        }
        if j + 1 < m && j + 1 < n {
            crate::kernels::gemm(
                b'N',
                b'N',
                (m - j - 1) as isize,
                (n - j - 1) as isize,
                1,
                -T::o(),
                a.offset(j as isize * lda + j as isize + 1),
                lda,
                a.offset((j + 1) as isize * lda + j as isize),
                lda,
                T::o(),
                a.offset((j + 1) as isize * lda + j as isize + 1),
                lda,
            );
        }
    }
    for j in (0..n).step_by(nb) {
        let width = nb.min(n - j);
        let mut coefficients = vec![T::z(); width * width];
        for col in 0..width {
            for i in 0..=col {
                coefficients[i + col * width] =
                    -(*a.offset((j + i) as isize + (j + col) as isize * lda)) * (*d.add(j + col));
            }
        }
        // T * V1^H = -U * D. Transpose the equation so the public native
        // unit-lower solve can operate on the right-sided system.
        let v = MatRef::from_raw_parts(
            a.offset(j as isize * lda + j as isize),
            width,
            width,
            1,
            lda,
        );
        let rhs =
            MatMut::from_raw_parts_mut(coefficients.as_mut_ptr(), width, width, 1, width as isize)
                .transpose_mut();
        crate::threading::with_par(|par| {
            colmat::linalg::triangular_solve::solve_unit_lower_triangular_in_place_with_conj(
                v,
                if T::COMPLEX { Conj::Yes } else { Conj::No },
                rhs,
                par,
            )
        });
        for col in 0..width {
            for i in 0..width {
                *t.offset(i as isize + (j + col) as isize * ldt) = coefficients[i + col * width];
            }
        }
    }
}
unsafe fn driver<T: Scalar, I: LapackInt>(
    m: isize,
    n: isize,
    nb: isize,
    a: *mut T,
    lda: isize,
    t: *mut T,
    ldt: isize,
    d: *mut T,
    info: *mut I,
) {
    *info = I::from_i64(0);
    for (arg, ok) in [
        (1, m >= 0),
        (2, n >= 0 && n <= m),
        (3, nb >= 1),
        (5, lda >= m.max(1)),
        (7, ldt >= nb.min(n).max(1)),
    ] {
        if !ok {
            *info = I::from_i64(-arg);
            crate::invalid(arg as i32);
            return;
        }
    }
    if n == 0 {
        return;
    }
    for (arg, ok) in [
        (5, matrix_addressable::<T>(m, n, lda)),
        (7, matrix_addressable::<T>(nb.min(n), n, ldt)),
    ] {
        if !ok {
            *info = I::from_i64(-arg);
            crate::invalid(arg as i32);
            return;
        }
    }
    reconstruct(
        m as usize,
        n as usize,
        nb.min(n) as usize,
        a,
        lda,
        t,
        ldt,
        d,
    );
}
macro_rules! exports {($p:ident,$name:ident,$t:ty,$i:ty,$s:tt)=>{paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p $name $s>](m:*const $i,n:*const $i,nb:*const $i,a:*mut $t,lda:*const $i,t:*mut $t,ldt:*const $i,d:*mut $t,info:*mut $i) {driver(*m as isize,*n as isize,*nb as isize,a,*lda as isize,t,*ldt as isize,d,info);}
}};}
exports!(s, orhr_col, f32, i32, _);
exports!(d, orhr_col, f64, i32, _);
exports!(c, unhr_col, colmat::c32, i32, _);
exports!(z, unhr_col, colmat::c64, i32, _);
exports!(s, orhr_col, f32, i64, _64_);
exports!(d, orhr_col, f64, i64, _64_);
exports!(c, unhr_col, colmat::c32, i64, _64_);
exports!(z, unhr_col, colmat::c64, i64, _64_);
#[cfg(test)]
mod tests {
    use super::*;
    use crate::householder::ReflectorScalar;
    fn verify<T: ReflectorScalar>(imag: T, tolerance: f64) {
        unsafe {
            for (m, n) in [(0usize, 0usize), (1, 1), (7, 3), (11, 4)] {
                for nb in [1, 2, 5] {
                    for identity in [false, true] {
                        let mut a = vec![T::z(); m * n];
                        for j in 0..n {
                            for i in 0..m {
                                a[i + j * m] = if identity {
                                    if i == j {
                                        T::o()
                                    } else {
                                        T::z()
                                    }
                                } else {
                                    T::real(((i * 11 + j * 7) % 19) as f64 / 13.0 - 0.4)
                                        + imag * T::real(((i + j * 3) % 11) as f64 / 17.0)
                                };
                            }
                        }
                        let mut t = vec![T::z(); nb * n];
                        if !identity {
                            crate::tsqr::factor(
                                m,
                                n,
                                m + 1,
                                nb.min(n).max(1),
                                a.as_mut_ptr(),
                                m as isize,
                                t.as_mut_ptr(),
                                nb as isize,
                            );
                            a = crate::tsqr::thin_q(
                                m,
                                n,
                                m + 1,
                                nb.min(n).max(1),
                                a.as_ptr(),
                                m as isize,
                                t.as_ptr(),
                                nb as isize,
                            );
                        }
                        let original = a.clone();
                        let mut d = vec![T::z(); n];
                        reconstruct(
                            m,
                            n,
                            nb,
                            a.as_mut_ptr(),
                            m as isize,
                            t.as_mut_ptr(),
                            nb as isize,
                            d.as_mut_ptr(),
                        );
                        let q = crate::tsqr::thin_q(
                            m,
                            n,
                            m + 1,
                            nb,
                            a.as_ptr(),
                            m as isize,
                            t.as_ptr(),
                            nb as isize,
                        );
                        for j in 0..n {
                            assert!(d[j] == T::o() || d[j] == -T::o());
                            for i in 0..m {
                                assert!(
                                    (q[i + j * m] * d[j] - original[i + j * m]).magnitude()
                                        < tolerance
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn real32() {
        verify(0.0f32, 4e-5);
    }
    #[test]
    fn real64() {
        verify(0.0f64, 4e-13);
    }
    #[test]
    fn complex32() {
        verify(colmat::c32::new(0.0, 1.0), 4e-5);
    }
    #[test]
    fn complex64() {
        verify(colmat::c64::new(0.0, 1.0), 4e-13);
    }
}
