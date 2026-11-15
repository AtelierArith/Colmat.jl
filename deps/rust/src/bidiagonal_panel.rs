//! LABRD panel contract: retain caller A22 and expose the two deferred updates.
//! A private working matrix is updated eagerly using native products, while
//! only the caller's leading NB rows/columns are committed on return.
use crate::householder::ReflectorScalar;
use crate::kernels::{gemv, Scalar};
use colmat::MatRef;
use paste::paste;

pub(crate) unsafe fn panel<T: ReflectorScalar, R: Scalar>(
    m: usize,
    n: usize,
    nb: usize,
    a: *mut T,
    lda: isize,
    d: *mut R,
    e: *mut R,
    tq: *mut T,
    tp: *mut T,
    x: *mut T,
    ldx: isize,
    y: *mut T,
    ldy: isize,
) {
    if m == 0 || n == 0 || nb == 0 {
        return;
    }
    let mut matrix = MatRef::from_raw_parts(a, m, n, 1, if n == 1 { 0 } else { lda }).to_owned();
    let ld = matrix.col_stride();
    let base = matrix.as_mut().as_ptr_mut();
    let mut work = vec![T::z(); m.max(n)];
    for j in 0..nb {
        let diagonal = base.offset(j as isize * (ld + 1));
        let old = *diagonal;
        if m >= n {
            let tau = crate::bidiagonal::reflector(m - j, diagonal, 1);
            *tq.add(j) = tau;
            *d.add(j) = R::real((*diagonal).re());
            if j + 1 < n {
                *diagonal = T::o();
                gemv(
                    b'C',
                    (m - j) as isize,
                    (n - j - 1) as isize,
                    tau,
                    diagonal.offset(ld),
                    ld,
                    diagonal,
                    1,
                    T::z(),
                    y.offset((j + 1) as isize + j as isize * ldy),
                    1,
                );
                crate::householder::apply(
                    b'L',
                    (m - j) as isize,
                    (n - j - 1) as isize,
                    diagonal,
                    1,
                    tau.cj(),
                    diagonal.offset(ld),
                    ld,
                    work.as_mut_ptr(),
                );
                let row = diagonal.offset(ld);
                for i in 0..n - j - 1 {
                    *row.offset(i as isize * ld) = (*row.offset(i as isize * ld)).cj();
                }
                let tau = crate::bidiagonal::reflector(n - j - 1, row, ld);
                *tp.add(j) = tau;
                *e.add(j) = R::real((*row).re());
                *row = T::o();
                gemv(
                    b'N',
                    (m - j - 1) as isize,
                    (n - j - 1) as isize,
                    tau,
                    row.add(1),
                    ld,
                    row,
                    ld,
                    T::z(),
                    x.offset((j + 1) as isize + j as isize * ldx),
                    1,
                );
                crate::householder::apply(
                    b'R',
                    (m - j - 1) as isize,
                    (n - j - 1) as isize,
                    row,
                    ld,
                    tau,
                    row.add(1),
                    ld,
                    work.as_mut_ptr(),
                );
                for i in 0..n - j - 1 {
                    *row.offset(i as isize * ld) = (*row.offset(i as isize * ld)).cj();
                }
            } else if T::COMPLEX {
                *diagonal = old;
            }
        } else {
            for i in 0..n - j {
                *diagonal.offset(i as isize * ld) = (*diagonal.offset(i as isize * ld)).cj();
            }
            let tau = crate::bidiagonal::reflector(n - j, diagonal, ld);
            *tp.add(j) = tau;
            *d.add(j) = R::real((*diagonal).re());
            if j + 1 < m {
                *diagonal = T::o();
                gemv(
                    b'N',
                    (m - j - 1) as isize,
                    (n - j) as isize,
                    tau,
                    diagonal.add(1),
                    ld,
                    diagonal,
                    ld,
                    T::z(),
                    x.offset((j + 1) as isize + j as isize * ldx),
                    1,
                );
                crate::householder::apply(
                    b'R',
                    (m - j - 1) as isize,
                    (n - j) as isize,
                    diagonal,
                    ld,
                    tau,
                    diagonal.add(1),
                    ld,
                    work.as_mut_ptr(),
                );
                for i in 0..n - j {
                    *diagonal.offset(i as isize * ld) = (*diagonal.offset(i as isize * ld)).cj();
                }
                let column = diagonal.add(1);
                let tau = crate::bidiagonal::reflector(m - j - 1, column, 1);
                *tq.add(j) = tau;
                *e.add(j) = R::real((*column).re());
                *column = T::o();
                gemv(
                    b'C',
                    (m - j - 1) as isize,
                    (n - j - 1) as isize,
                    tau,
                    column.offset(ld),
                    ld,
                    column,
                    1,
                    T::z(),
                    y.offset((j + 1) as isize + j as isize * ldy),
                    1,
                );
                crate::householder::apply(
                    b'L',
                    (m - j - 1) as isize,
                    (n - j - 1) as isize,
                    column,
                    1,
                    tau.cj(),
                    column.offset(ld),
                    ld,
                    work.as_mut_ptr(),
                );
            } else {
                for i in 0..n - j {
                    *diagonal.offset(i as isize * ld) = (*diagonal.offset(i as isize * ld)).cj();
                }
                if T::COMPLEX {
                    *diagonal = old;
                }
            }
        }
    }
    for j in 0..n {
        for i in 0..m {
            if i < nb || j < nb {
                *a.offset(i as isize + j as isize * lda) = matrix[(i, j)];
            }
        }
    }
}

unsafe fn driver<T: ReflectorScalar, R: Scalar>(
    m: isize,
    n: isize,
    nb: isize,
    a: *mut T,
    lda: isize,
    d: *mut R,
    e: *mut R,
    tq: *mut T,
    tp: *mut T,
    x: *mut T,
    ldx: isize,
    y: *mut T,
    ldy: isize,
) {
    for (arg, valid) in [
        (1, m >= 0),
        (2, n >= 0),
        (3, nb >= 0 && nb <= m.min(n)),
        (5, lda >= m.max(1)),
        (11, ldx >= m.max(1)),
        (13, ldy >= n.max(1)),
    ] {
        if !valid {
            crate::invalid(arg);
            return;
        }
    }
    if m == 0 || n == 0 || nb == 0 {
        return;
    }
    for (arg, rows, cols, ld) in [(5, m, n, lda), (11, m, nb, ldx), (13, n, nb, ldy)] {
        if !crate::kernels::matrix_addressable::<T>(rows, cols, ld) {
            crate::invalid(arg);
            return;
        }
    }
    for (arg, elements) in [(3, m.checked_mul(n))] {
        if elements
            .and_then(|x| x.checked_mul(std::mem::size_of::<T>() as isize))
            .is_none()
        {
            crate::invalid(arg);
            return;
        }
    }
    panel(
        m as usize,
        n as usize,
        nb as usize,
        a,
        lda,
        d,
        e,
        tq,
        tp,
        x,
        ldx,
        y,
        ldy,
    );
}
macro_rules! exports {
    ($p:ident,$t:ty,$r:ty,$i:ty,$suffix:tt)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p labrd $suffix>](m:*const $i,n:*const $i,nb:*const $i,
            a:*mut $t,lda:*const $i,d:*mut $r,e:*mut $r,tq:*mut $t,tp:*mut $t,
            x:*mut $t,ldx:*const $i,y:*mut $t,ldy:*const $i) {
            driver(*m as isize,*n as isize,*nb as isize,a,*lda as isize,d,e,tq,tp,x,*ldx as isize,y,*ldy as isize);
        }
    }};
}
macro_rules! abi {
    ($i:ty,$suffix:tt) => {
        exports!(s, f32, f32, $i, $suffix);
        exports!(d, f64, f64, $i, $suffix);
        exports!(c, colmat::c32, f32, $i, $suffix);
        exports!(z, colmat::c64, f64, $i, $suffix);
    };
}
abi!(i32, "_");
abi!(i64, "_64_");

#[cfg(test)]
mod tests {
    use super::*;
    fn multiply<T: Scalar>(matrix: &mut [T], size: usize, v: &[T], tau: T) {
        for i in 0..size {
            let dot = (0..size).fold(T::z(), |s, j| s + matrix[i + j * size] * v[j]);
            for j in 0..size {
                matrix[i + j * size] = matrix[i + j * size] - tau * dot * v[j].cj();
            }
        }
    }
    fn deferred_updates<T: ReflectorScalar>(imaginary: T, tolerance: f64) {
        unsafe {
            for (m, n) in [
                (0, 4),
                (4, 0),
                (1, 1),
                (1, 4),
                (4, 1),
                (4, 6),
                (6, 4),
                (5, 5),
            ] {
                let upper = m >= n;
                let lda = m + 2;
                let ldx = m + 2;
                let ldy = n + 3;
                for nb in 0..=m.min(n) {
                    let mut a = vec![T::real(777.0); lda * n + 2];
                    let mut original = vec![T::z(); m * n];
                    for j in 0..n {
                        for i in 0..m {
                            let value = T::real(((i + 3 * j + i * j) % 23) as f64 / 9.0 - 0.7)
                                + imaginary
                                    * T::real(((5 * i + 2 * j + 1) % 17) as f64 / 11.0 - 0.5);
                            a[1 + i + j * lda] = value;
                            original[i + j * m] = value;
                        }
                    }
                    let before = a.clone();
                    let mut d = vec![777.0f64; nb + 2];
                    let mut e = d.clone();
                    let mut tq = vec![T::real(777.0); nb + 2];
                    let mut tp = tq.clone();
                    let mut x = vec![T::real(777.0); ldx * nb + 2];
                    let mut y = vec![T::real(777.0); ldy * nb + 2];
                    panel(
                        m,
                        n,
                        nb,
                        a.as_mut_ptr().add(1),
                        lda as isize,
                        d.as_mut_ptr().add(1),
                        e.as_mut_ptr().add(1),
                        tq.as_mut_ptr().add(1),
                        tp.as_mut_ptr().add(1),
                        x.as_mut_ptr().add(1),
                        ldx as isize,
                        y.as_mut_ptr().add(1),
                        ldy as isize,
                    );
                    let mut q = vec![T::z(); m * m];
                    let mut p = vec![T::z(); n * n];
                    for i in 0..m {
                        q[i + i * m] = T::o();
                    }
                    for i in 0..n {
                        p[i + i * n] = T::o();
                    }
                    for j in 0..nb {
                        let start = if upper { j } else { j + 1 };
                        if start < m {
                            let mut v = vec![T::z(); m];
                            v[start] = T::o();
                            for i in start + 1..m {
                                v[i] = a[1 + i + j * lda];
                            }
                            multiply(&mut q, m, &v, tq[j + 1]);
                        }
                        let start = if upper { j + 1 } else { j };
                        if start < n {
                            let mut u = vec![T::z(); n];
                            u[start] = T::o();
                            for i in start + 1..n {
                                u[i] = a[1 + j + i * lda].cj();
                            }
                            multiply(&mut p, n, &u, tp[j + 1]);
                        }
                    }
                    for j in 0..n {
                        for i in 0..m {
                            let mut transformed = T::z();
                            for r in 0..m {
                                for c in 0..n {
                                    transformed = transformed
                                        + q[r + i * m].cj() * original[r + c * m] * p[c + j * n];
                                }
                            }
                            let expected = if i < nb || j < nb {
                                if i == j {
                                    T::real(d[i + 1])
                                } else if upper && j == i + 1 {
                                    T::real(e[i + 1])
                                } else if !upper && i == j + 1 {
                                    T::real(e[j + 1])
                                } else {
                                    T::z()
                                }
                            } else {
                                // Only caller A22 is untouched; applying its two
                                // deferred products must match the independent Q^H*A*P.
                                assert_eq!(a[1 + i + j * lda], before[1 + i + j * lda]);
                                let mut value = original[i + j * m];
                                for s in 0..nb {
                                    let vstart = if upper { s } else { s + 1 };
                                    let ustart = if upper { s + 1 } else { s };
                                    let v = if i == vstart {
                                        T::o()
                                    } else {
                                        a[1 + i + s * lda]
                                    };
                                    let uh = if j == ustart {
                                        T::o()
                                    } else {
                                        a[1 + s + j * lda]
                                    };
                                    value = value
                                        - v * y[1 + j + s * ldy].cj()
                                        - x[1 + i + s * ldx] * uh;
                                }
                                value
                            };
                            assert!(
                                (transformed - expected).magnitude() < tolerance,
                                "m={m} n={n} nb={nb} row={i} col={j}"
                            );
                        }
                    }
                    for j in 0..n {
                        for i in m..lda {
                            assert_eq!(a[1 + i + j * lda], T::real(777.0));
                        }
                    }
                    for j in 0..nb {
                        for i in m..ldx {
                            assert_eq!(x[1 + i + j * ldx], T::real(777.0));
                        }
                        for i in n..ldy {
                            assert_eq!(y[1 + i + j * ldy], T::real(777.0));
                        }
                    }
                    for v in [&a, &x, &y, &tq, &tp] {
                        assert_eq!(v[0], T::real(777.0));
                        assert_eq!(*v.last().unwrap(), T::real(777.0));
                    }
                    for v in [&d, &e] {
                        assert_eq!(v[0], 777.0);
                        assert_eq!(*v.last().unwrap(), 777.0);
                    }
                }
            }
        }
    }
    #[test]
    fn real32_deferred_updates() {
        deferred_updates(0.0f32, 3e-4);
    }
    #[test]
    fn real64_deferred_updates() {
        deferred_updates(0.0f64, 1e-11);
    }
    #[test]
    fn complex32_deferred_updates() {
        deferred_updates(colmat::c32::new(0.0, 1.0), 5e-4);
    }
    #[test]
    fn complex64_deferred_updates() {
        deferred_updates(colmat::c64::new(0.0, 1.0), 1e-11);
    }
}
