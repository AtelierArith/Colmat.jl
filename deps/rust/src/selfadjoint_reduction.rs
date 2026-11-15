//! Adapt native self-adjoint tridiagonalization to LAPACK triangle storage.
use crate::generalized_selfadjoint::{hermitian, GeneralizedScalar};
use crate::householder::ReflectorScalar;
use crate::kernels::Scalar;
use crate::kernels::{flag, matrix_addressable};
use crate::lapack::LapackInt;
use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::Mat;
use paste::paste;

// Native real reflectors have LAPACK's real-beta gauge. Complex native
// reflectors need a separate real-E gauge adaptation before being exposed.
pub(crate) unsafe fn native_real<T: ReflectorScalar + GeneralizedScalar, R: Scalar>(
    lower: bool,
    n: usize,
    a: *mut T,
    lda: isize,
    d: *mut R,
    e: *mut R,
    tau: *mut T,
) {
    assert!(!T::COMPLEX);
    if n == 0 {
        return;
    }
    let mut matrix = hermitian(if lower { b'L' } else { b'U' }, n, a, lda);
    if !lower {
        // Reversal maps LAPACK's descending upper reflectors to the native
        // ascending lower sequence without changing their real gauge.
        for j in 0..n {
            for i in 0..n {
                let (r, c) = (n - 1 - i, n - 1 - j);
                if i + j * n < r + c * n {
                    let old = matrix[(i, j)];
                    matrix[(i, j)] = matrix[(r, c)];
                    matrix[(r, c)] = old;
                }
            }
        }
    }
    let mut coefficients = Mat::<T>::zeros(1, n.saturating_sub(1));
    use colmat::linalg::evd::tridiag::{tridiag_in_place, tridiag_in_place_scratch};
    crate::threading::with_par(|par| {
        let mut memory = MemBuffer::new(tridiag_in_place_scratch::<T>(
            n,
            par.shape(),
            Default::default(),
        ));
        tridiag_in_place(
            matrix.as_mut(),
            coefficients.as_mut(),
            par,
            MemStack::new(&mut memory),
            Default::default(),
        );
    });
    for j in 0..n {
        let column = if lower { j } else { n - 1 - j };
        *d.add(j) = R::real(matrix[(column, column)].re());
        if j + 1 < n {
            let stage = if lower { j } else { n - 2 - j };
            *e.add(j) = R::real(matrix[(stage + 1, stage)].re());
            *tau.add(j) = T::real(1.0 / coefficients[(0, stage)].re());
        }
        let rows = if lower { j..n } else { 0..j + 1 };
        for i in rows {
            *a.offset(i as isize + j as isize * lda) = if lower {
                matrix[(i, j)]
            } else {
                matrix[(n - 1 - i, n - 1 - j)]
            };
        }
    }
}

// Native reflector panels plus Hermitian rank-two updates retain the complex
// LAPACK real-E gauge. The same step supplies LATRD's deferred W columns.
pub(crate) unsafe fn composition<T: ReflectorScalar + GeneralizedScalar, R: Scalar>(
    lower: bool,
    n: usize,
    nb: usize,
    a: *mut T,
    lda: isize,
    d: Option<*mut R>,
    e: *mut R,
    tau: *mut T,
    w: Option<(*mut T, isize)>,
) {
    if n == 0 || nb == 0 {
        return;
    }
    let mut matrix = hermitian(if lower { b'L' } else { b'U' }, n, a, lda);
    if !lower {
        for j in 0..n {
            for i in 0..n {
                let (r, c) = (n - 1 - i, n - 1 - j);
                if i + j * n < r + c * n {
                    let value = matrix[(i, j)];
                    matrix[(i, j)] = matrix[(r, c)];
                    matrix[(r, c)] = value;
                }
            }
        }
    }
    let ld = matrix.col_stride();
    let base = matrix.as_mut().as_ptr_mut();
    for stage in 0..nb.min(n - 1) {
        let size = n - stage - 1;
        let column = base.offset((stage + 1) as isize + stage as isize * ld);
        let coefficient = crate::bidiagonal::reflector(size, column, 1);
        let beta = (*column).re();
        *column = T::o();
        let mut update = vec![T::z(); size];
        if coefficient != T::z() {
            let trailing = column.offset(ld);
            crate::kernels::gemv(
                b'N',
                size as isize,
                size as isize,
                coefficient,
                trailing,
                ld,
                column,
                1,
                T::z(),
                update.as_mut_ptr(),
                1,
            );
            let dot = (0..size).fold(T::z(), |sum, i| sum + update[i].cj() * *column.add(i));
            let correction = -T::real(0.5) * coefficient * dot;
            for i in 0..size {
                update[i] = update[i] + correction * *column.add(i);
            }
            crate::dense_extensions::rank2(
                b'L',
                size as isize,
                -T::o(),
                column,
                1,
                update.as_ptr(),
                1,
                trailing,
                ld,
                true,
            );
            for j in stage + 1..n {
                for i in j + 1..n {
                    *base.offset(j as isize + i as isize * ld) =
                        (*base.offset(i as isize + j as isize * ld)).cj();
                }
            }
        }
        let index = if lower { stage } else { n - 2 - stage };
        *e.add(index) = R::real(beta);
        *tau.add(index) = coefficient;
        if let Some((output, stride)) = w {
            let outcol = if lower { stage } else { nb - 1 - stage };
            for i in 0..size {
                let row = if lower {
                    stage + 1 + i
                } else {
                    n - 2 - stage - i
                };
                *output.offset(row as isize + outcol as isize * stride) = update[i];
            }
        } else {
            *column = T::real(beta);
        }
    }
    for column in 0..if w.is_some() { nb } else { n } {
        let j = if lower { column } else { n - 1 - column };
        if let Some(diagonal) = d {
            *diagonal.add(j) = R::real(matrix[(column, column)].re());
        }
        let rows = if lower { j..n } else { 0..j + 1 };
        for i in rows {
            *a.offset(i as isize + j as isize * lda) = if lower {
                matrix[(i, j)]
            } else {
                matrix[(n - 1 - i, n - 1 - j)]
            };
        }
    }
}

pub(crate) unsafe fn ordinary<T: ReflectorScalar + GeneralizedScalar, R: Scalar>(
    lower: bool,
    n: usize,
    a: *mut T,
    lda: isize,
    d: *mut R,
    e: *mut R,
    tau: *mut T,
) {
    if n == 0 {
        return;
    }
    if T::COMPLEX {
        composition(lower, n, n, a, lda, Some(d), e, tau, None);
        return;
    }
    let mut largest = 0.0f64;
    let mut smallest = f64::INFINITY;
    let mut finite = true;
    for j in 0..n {
        for i in if lower { j..n } else { 0..j + 1 } {
            let value = if i == j {
                T::diagonal(a.offset(i as isize + j as isize * lda))
            } else {
                *a.offset(i as isize + j as isize * lda)
            };
            let magnitude = value.magnitude();
            finite &= magnitude.is_finite();
            largest = largest.max(magnitude);
            if magnitude > 0.0 {
                smallest = smallest.min(magnitude);
            }
        }
    }
    let (minimum, maximum) = if T::EPS > 1e-10 {
        (f32::MIN_POSITIVE as f64, f32::MAX as f64)
    } else {
        (f64::MIN_POSITIVE, f64::MAX)
    };
    // Bound intermediate norms by the input's Frobenius norm. Outside this
    // range, normalize individual reflector panels without scaling away tiny
    // entries elsewhere in the matrix.
    let low = minimum.sqrt() / T::EPS;
    let high = maximum.sqrt() / (8.0 * n as f64);
    if finite && (largest == 0.0 || (smallest >= low && largest <= high)) {
        native_real(lower, n, a, lda, d, e, tau);
    } else {
        composition(lower, n, n, a, lda, Some(d), e, tau, None);
    }
}

unsafe fn factor_driver<T: ReflectorScalar + GeneralizedScalar, R: Scalar, I: LapackInt>(
    u: u8,
    n: isize,
    a: *mut T,
    lda: isize,
    d: *mut R,
    e: *mut R,
    tau: *mut T,
    work: *mut T,
    lwork: Option<isize>,
    info: *mut I,
) {
    *info = I::from_i64(0);
    for (arg, valid) in [
        (1, matches!(u, b'U' | b'L')),
        (2, n >= 0),
        (4, lda >= n.max(1)),
    ] {
        if !valid {
            *info = I::from_i64(-arg);
            crate::invalid(arg as i32);
            return;
        }
    }
    if let Some(lw) = lwork {
        *work = T::o();
        if lw != -1 && lw < 1 {
            *info = I::from_i64(-9);
            crate::invalid(9);
            return;
        }
        if lw == -1 {
            return;
        }
    }
    if n == 0 {
        return;
    }
    if !matrix_addressable::<T>(n, n, lda) {
        *info = I::from_i64(-4);
        crate::invalid(4);
        return;
    }
    ordinary(u == b'L', n as usize, a, lda, d, e, tau);
}
unsafe fn panel_driver<T: ReflectorScalar + GeneralizedScalar, R: Scalar>(
    u: u8,
    n: isize,
    nb: isize,
    a: *mut T,
    lda: isize,
    e: *mut R,
    tau: *mut T,
    w: *mut T,
    ldw: isize,
) {
    for (arg, valid) in [
        (1, matches!(u, b'U' | b'L')),
        (2, n >= 0),
        (3, nb >= 0 && nb <= n),
        (5, lda >= n.max(1)),
        (9, ldw >= n.max(1)),
    ] {
        if !valid {
            crate::invalid(arg);
            return;
        }
    }
    if n == 0 || nb == 0 {
        return;
    }
    for (arg, valid) in [
        (5, matrix_addressable::<T>(n, n, lda)),
        (9, matrix_addressable::<T>(n, nb, ldw)),
    ] {
        if !valid {
            crate::invalid(arg);
            return;
        }
    }
    composition(
        u == b'L',
        n as usize,
        nb as usize,
        a,
        lda,
        None,
        e,
        tau,
        Some((w, ldw)),
    );
}
macro_rules! exports {
    ($p:ident,$trd:ident,$td2:ident,$t:ty,$r:ty,$i:ty,$suffix:tt)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $trd $suffix>](u:*const u8,n:*const $i,a:*mut $t,lda:*const $i,
            d:*mut $r,e:*mut $r,tau:*mut $t,work:*mut $t,lwork:*const $i,info:*mut $i) {
            factor_driver(flag(u),*n as isize,a,*lda as isize,d,e,tau,work,Some(*lwork as isize),info);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $td2 $suffix>](u:*const u8,n:*const $i,a:*mut $t,lda:*const $i,
            d:*mut $r,e:*mut $r,tau:*mut $t,info:*mut $i) {
            factor_driver(flag(u),*n as isize,a,*lda as isize,d,e,tau,std::ptr::null_mut(),None,info);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p latrd $suffix>](u:*const u8,n:*const $i,nb:*const $i,a:*mut $t,
            lda:*const $i,e:*mut $r,tau:*mut $t,w:*mut $t,ldw:*const $i) {
            panel_driver(flag(u),*n as isize,*nb as isize,a,*lda as isize,e,tau,w,*ldw as isize);
        }
    }};
}
macro_rules! abi {
    ($i:ty,$suffix:tt) => {
        exports!(s, sytrd, sytd2, f32, f32, $i, $suffix);
        exports!(d, sytrd, sytd2, f64, f64, $i, $suffix);
        exports!(c, hetrd, hetd2, colmat::c32, f32, $i, $suffix);
        exports!(z, hetrd, hetd2, colmat::c64, f64, $i, $suffix);
    };
}
abi!(i32, "_");
abi!(i64, "_64_");

#[cfg(test)]
mod tests {
    use super::*;
    fn verify<T: ReflectorScalar + GeneralizedScalar>(imaginary: T, strategy: u8, tolerance: f64) {
        unsafe {
            for lower in [false, true] {
                for n in [0, 1, 2, 5, 8] {
                    for mode in 0..3 {
                        let scales: &[f64] = if strategy == 2 {
                            if T::EPS > 1e-10 {
                                &[1e-38, 1.0, 1e30]
                            } else {
                                &[1e-310, 1.0, 1e300]
                            }
                        } else {
                            &[1.0]
                        };
                        for &scale in scales {
                            let lda = n + 3;
                            let mut a = vec![T::real(777.0); lda * n + 2];
                            let mut original = vec![T::z(); n * n];
                            for j in 0..n {
                                for i in 0..n {
                                    let value = match mode {
                                        0 => {
                                            T::real(
                                                ((i.min(j) * 17 + i.max(j) * 31 + i * j) % 29)
                                                    as f64
                                                    / 13.0
                                                    - 0.8,
                                            ) + imaginary
                                                * T::real(if i == j {
                                                    0.0
                                                } else {
                                                    (((i.min(j) * 7 + i.max(j) * 11) % 17) as f64
                                                        / 9.0
                                                        + 0.1)
                                                        * if i > j { 1.0 } else { -1.0 }
                                                })
                                        }
                                        1 => {
                                            if i == j {
                                                T::real((i + 1) as f64)
                                            } else {
                                                T::z()
                                            }
                                        }
                                        _ => T::z(),
                                    };
                                    let value = value * T::real(scale);
                                    original[i + j * n] = value;
                                    a[1 + i + j * lda] = if (lower && i >= j) || (!lower && i <= j)
                                    {
                                        if i == j && T::COMPLEX {
                                            T::parts(value.re(), f64::NAN)
                                        } else {
                                            value
                                        }
                                    } else {
                                        T::real(f64::NAN)
                                    };
                                }
                            }
                            let before = a.clone();
                            let mut d = vec![777.0f64; n + 2];
                            let mut e = d.clone();
                            let mut tau = vec![T::real(777.0); n + 2];
                            if strategy == 2 {
                                ordinary(
                                    lower,
                                    n,
                                    a.as_mut_ptr().add(1),
                                    lda as isize,
                                    d.as_mut_ptr().add(1),
                                    e.as_mut_ptr().add(1),
                                    tau.as_mut_ptr().add(1),
                                );
                            } else if strategy == 1 {
                                composition(
                                    lower,
                                    n,
                                    n,
                                    a.as_mut_ptr().add(1),
                                    lda as isize,
                                    Some(d.as_mut_ptr().add(1)),
                                    e.as_mut_ptr().add(1),
                                    tau.as_mut_ptr().add(1),
                                    None,
                                );
                            } else {
                                native_real(
                                    lower,
                                    n,
                                    a.as_mut_ptr().add(1),
                                    lda as isize,
                                    d.as_mut_ptr().add(1),
                                    e.as_mut_ptr().add(1),
                                    tau.as_mut_ptr().add(1),
                                );
                            }
                            let mut q = vec![T::z(); n * n];
                            for i in 0..n {
                                q[i + i * n] = T::o();
                            }
                            for step in 0..n.saturating_sub(1) {
                                let stage = if lower { step } else { n - 2 - step };
                                let mut v = vec![T::z(); n];
                                if lower {
                                    v[stage + 1] = T::o();
                                    for i in stage + 2..n {
                                        v[i] = a[1 + i + stage * lda];
                                    }
                                } else {
                                    v[stage] = T::o();
                                    for i in 0..stage {
                                        v[i] = a[1 + i + (stage + 1) * lda];
                                    }
                                }
                                for i in 0..n {
                                    let dot =
                                        (0..n).fold(T::z(), |sum, j| sum + q[i + j * n] * v[j]);
                                    for j in 0..n {
                                        q[i + j * n] =
                                            q[i + j * n] - tau[stage + 1] * dot * v[j].cj();
                                    }
                                }
                            }
                            for j in 0..n {
                                for i in 0..n {
                                    let mut value = T::z();
                                    let mut dot = T::z();
                                    for s in 0..n {
                                        value = value
                                            + q[i + s * n]
                                                * T::real(d[s + 1] / scale)
                                                * q[j + s * n].cj();
                                        dot = dot + q[s + i * n].cj() * q[s + j * n];
                                    }
                                    for s in 0..n.saturating_sub(1) {
                                        value = value
                                            + T::real(e[s + 1] / scale)
                                                * (q[i + s * n] * q[j + (s + 1) * n].cj()
                                                    + q[i + (s + 1) * n] * q[j + s * n].cj());
                                    }
                                    assert!(
                                        (value
                                            - T::parts(
                                                original[i + j * n].re() / scale,
                                                original[i + j * n].im() / scale
                                            ))
                                        .magnitude()
                                            < tolerance,
                                        "n={n} lower={lower} mode={mode} i={i} j={j}"
                                    );
                                    assert!(
                                        (dot - if i == j { T::o() } else { T::z() }).magnitude()
                                            < tolerance
                                    );
                                    if (lower && i < j) || (!lower && i > j) {
                                        assert!(a[1 + i + j * lda].re().is_nan());
                                    }
                                }
                                for i in n..lda {
                                    assert_eq!(a[1 + i + j * lda], before[1 + i + j * lda]);
                                }
                            }
                            assert_eq!(a[0], T::real(777.0));
                            assert_eq!(*a.last().unwrap(), T::real(777.0));
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn native_real32_triangles() {
        verify(0.0f32, 0, 5e-4);
    }
    #[test]
    fn native_real64_triangles() {
        verify(0.0f64, 0, 1e-11);
    }
    #[test]
    fn composed_real32_triangles() {
        verify(0.0f32, 1, 5e-4);
    }
    #[test]
    fn composed_real64_triangles() {
        verify(0.0f64, 1, 1e-11);
    }
    #[test]
    fn complex32_real_e_gauge() {
        verify(colmat::c32::new(0.0, 1.0), 1, 8e-4);
    }
    #[test]
    fn complex64_real_e_gauge() {
        verify(colmat::c64::new(0.0, 1.0), 1, 1e-11);
    }
    #[test]
    fn scaled_real32_triangles() {
        verify(0.0f32, 2, 8e-4);
    }
    #[test]
    fn scaled_real64_triangles() {
        verify(0.0f64, 2, 1e-11);
    }
    #[test]
    fn scaled_complex32_triangles() {
        verify(colmat::c32::new(0.0, 1.0), 2, 8e-4);
    }
    #[test]
    fn scaled_complex64_triangles() {
        verify(colmat::c64::new(0.0, 1.0), 2, 1e-11);
    }
    #[test]
    fn mixed_real_scales_keep_small_offdiagonals() {
        unsafe {
            let mut a = [1e300f64, 1e-310, 2e-310, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
            let mut d = [0.0; 3];
            let mut e = [0.0; 2];
            let mut tau = [0.0; 2];
            ordinary(
                true,
                3,
                a.as_mut_ptr(),
                3,
                d.as_mut_ptr(),
                e.as_mut_ptr(),
                tau.as_mut_ptr(),
            );
            assert_eq!(d[0], 1e300);
            assert!(e[0] != 0.0);
            assert!((e[0] / 1e-310 + 5.0f64.sqrt()).abs() < 1e-11);
        }
    }
}
