//! ORM22/UNM22: apply a 2-by-2 block matrix with triangular off-diagonal
//! blocks through native structured products. No full Q expansion is needed.
use crate::kernels::{flag, Scalar};
use crate::lapack::LapackInt;
use colmat::linalg::matmul::triangular::{matmul_with_conj, BlockStructure::*};
use colmat::{Accum, Conj, Mat, MatRef};
use paste::paste;

unsafe fn apply<T: Scalar>(
    left: bool,
    tr: u8,
    m: usize,
    n: usize,
    n1: usize,
    n2: usize,
    q: *const T,
    ldq: isize,
    c: *mut T,
    ldc: isize,
) {
    let nq = n1 + n2;
    let matrix = MatRef::from_raw_parts(q, nq, nq, 1, ldq);
    let source = MatRef::from_raw_parts(c, m, n, 1, ldc);
    let mut result = Mat::<T>::zeros(m, n);
    let conjugate = if tr == b'C' { Conj::Yes } else { Conj::No };
    // Row partition is (N1,N2), but column partition is (N2,N1).
    // Treating both partitions as (N1,N2) fails whenever N1 != N2.
    for (row, col, rows, cols, structure) in [
        (0, 0, n1, n2, Rectangular),
        (0, n2, n1, n1, TriangularLower),
        (n1, 0, n2, n2, TriangularUpper),
        (n1, n2, n2, n1, Rectangular),
    ] {
        if rows == 0 || cols == 0 {
            continue;
        }
        let block = matrix.submatrix(row, col, rows, cols);
        let (block, row, col, rows, cols, structure) = if tr == b'N' {
            (block, row, col, rows, cols, structure)
        } else {
            (
                block.transpose(),
                col,
                row,
                cols,
                rows,
                match structure {
                    TriangularLower => TriangularUpper,
                    TriangularUpper => TriangularLower,
                    _ => Rectangular,
                },
            )
        };
        crate::threading::with_par(|par| {
            if left {
                matmul_with_conj(
                    result.as_mut().submatrix_mut(row, 0, rows, n),
                    Rectangular,
                    Accum::Add,
                    block,
                    structure,
                    conjugate,
                    source.submatrix(col, 0, cols, n),
                    Rectangular,
                    Conj::No,
                    T::o(),
                    par,
                );
            } else {
                matmul_with_conj(
                    result.as_mut().submatrix_mut(0, col, m, cols),
                    Rectangular,
                    Accum::Add,
                    source.submatrix(0, row, m, rows),
                    Rectangular,
                    Conj::No,
                    block,
                    structure,
                    conjugate,
                    T::o(),
                    par,
                );
            }
        });
    }
    for j in 0..n {
        for i in 0..m {
            *c.offset(i as isize + j as isize * ldc) = result[(i, j)];
        }
    }
}

unsafe fn driver<T: Scalar, I: LapackInt>(
    side: u8,
    tr: u8,
    m: isize,
    n: isize,
    n1: isize,
    n2: isize,
    q: *const T,
    ldq: isize,
    c: *mut T,
    ldc: isize,
    work: *mut T,
    lwork: isize,
    info: *mut I,
) {
    *info = I::from_i64(0);
    let left = side == b'L';
    let nq = if left { m } else { n };
    // Reference-LAPACK's executable contract uses NQ, despite the prose
    // documentation interchanging M/N for the nondegenerate minimum.
    let minimum = if n1 == 0 || n2 == 0 { 1 } else { nq };
    for (arg, valid) in [
        (1, matches!(side, b'L' | b'R')),
        (2, tr == b'N' || tr == if T::COMPLEX { b'C' } else { b'T' }),
        (3, m >= 0),
        (4, n >= 0),
        (5, n1 >= 0 && n1.checked_add(n2) == Some(nq)),
        (6, n2 >= 0),
        (8, ldq >= nq.max(1)),
        (10, ldc >= m.max(1)),
        (12, lwork == -1 || lwork >= minimum),
    ] {
        if !valid {
            *info = I::from_i64(-arg);
            crate::invalid(arg as i32);
            return;
        }
    }
    let Some(optimal) = m.checked_mul(n) else {
        *info = I::from_i64(-12);
        crate::invalid(12);
        return;
    };
    *work = crate::lapack::workspace::<T>(optimal);
    if lwork == -1 {
        return;
    }
    if m == 0 || n == 0 {
        *work = T::o();
        return;
    }
    let bytes = std::mem::size_of::<T>() as isize;
    for (arg, count) in [
        (8, nq.checked_mul(ldq)),
        (10, n.checked_mul(ldc)),
        (12, Some(optimal)),
    ] {
        if count.and_then(|x| x.checked_mul(bytes)).is_none() {
            *info = I::from_i64(-arg);
            crate::invalid(arg as i32);
            return;
        }
    }
    apply(
        left,
        tr,
        m as usize,
        n as usize,
        n1 as usize,
        n2 as usize,
        q,
        ldq,
        c,
        ldc,
    );
    if n1 == 0 || n2 == 0 {
        *work = T::o();
    }
}
macro_rules! exports {
    ($p:ident,$family:ident,$t:ty,$i:ty,$suffix:tt)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $family $suffix>](
            side:*const u8,tr:*const u8,m:*const $i,n:*const $i,n1:*const $i,n2:*const $i,
            q:*const $t,ldq:*const $i,c:*mut $t,ldc:*const $i,work:*mut $t,lwork:*const $i,info:*mut $i) {
            driver(flag(side),flag(tr),*m as isize,*n as isize,*n1 as isize,*n2 as isize,
                q,*ldq as isize,c,*ldc as isize,work,*lwork as isize,info);
        }
    }};
}
macro_rules! abi {
    ($i:ty,$suffix:tt) => {
        exports!(s, orm22, f32, $i, $suffix);
        exports!(d, orm22, f64, $i, $suffix);
        exports!(c, unm22, colmat::c32, $i, $suffix);
        exports!(z, unm22, colmat::c64, $i, $suffix);
    };
}
abi!(i32, "_");
abi!(i64, "_64_");

#[cfg(test)]
mod tests {
    use super::*;
    fn products<T: Scalar>(imaginary: T, tolerance: f64) {
        unsafe {
            for n1 in 0..5 {
                for n2 in 0..5 {
                    let nq = n1 + n2;
                    if nq == 0 {
                        continue;
                    }
                    let ldq = nq + 2;
                    let mut q = vec![T::real(777.0); ldq * nq];
                    let mut dense = vec![T::z(); nq * nq];
                    for j in 0..nq {
                        for i in 0..nq {
                            let unused = (i < n1 && j >= n2 && i < j - n2)
                                || (i >= n1 && j < n2 && i - n1 > j);
                            if unused {
                                q[i + j * ldq] = T::real(f64::NAN);
                            } else {
                                let value = T::real((i + 2 * j + 1) as f64 / 7.0)
                                    + imaginary * T::real((2 * i + j + 1) as f64 / 11.0);
                                q[i + j * ldq] = value;
                                dense[i + j * nq] = value;
                            }
                        }
                    }
                    let before = q.clone();
                    for left in [false, true] {
                        for tr in [b'N', if T::COMPLEX { b'C' } else { b'T' }] {
                            let (m, n) = if left { (nq, 3) } else { (3, nq) };
                            let ldc = m + 2;
                            let mut c = vec![T::real(777.0); ldc * n];
                            for j in 0..n {
                                for i in 0..m {
                                    c[i + j * ldc] = T::real((i + j + 1) as f64 / 5.0)
                                        + imaginary * T::real((2 * i + j + 3) as f64 / 13.0);
                                }
                            }
                            let original = c.clone();
                            let op = |i: usize, j: usize| {
                                if tr == b'N' {
                                    dense[i + j * nq]
                                } else {
                                    let x = dense[j + i * nq];
                                    if tr == b'C' {
                                        x.cj()
                                    } else {
                                        x
                                    }
                                }
                            };
                            apply(
                                left,
                                tr,
                                m,
                                n,
                                n1,
                                n2,
                                q.as_ptr(),
                                ldq as isize,
                                c.as_mut_ptr(),
                                ldc as isize,
                            );
                            for j in 0..n {
                                for i in 0..m {
                                    let mut expected = T::z();
                                    for k in 0..nq {
                                        expected = expected
                                            + if left {
                                                op(i, k) * original[k + j * ldc]
                                            } else {
                                                original[i + k * ldc] * op(k, j)
                                            };
                                    }
                                    assert!(
                                        (c[i + j * ldc] - expected).magnitude() < tolerance,
                                        "n1={n1} n2={n2} left={left} tr={tr} row={i} col={j}"
                                    );
                                }
                            }
                            for j in 0..n {
                                for i in m..ldc {
                                    assert_eq!(c[i + j * ldc], T::real(777.0));
                                }
                            }
                        }
                    }
                    for (x, y) in q.iter().zip(before) {
                        assert!(*x == y || (x.re().is_nan() && y.re().is_nan()));
                    }
                }
            }
        }
    }
    #[test]
    fn real32_blocks() {
        products(0.0f32, 2e-4);
    }
    #[test]
    fn real64_blocks() {
        products(0.0f64, 1e-11);
    }
    #[test]
    fn complex32_blocks() {
        products(colmat::c32::new(0.0, 1.0), 5e-4);
    }
    #[test]
    fn complex64_blocks() {
        products(colmat::c64::new(0.0, 1.0), 1e-11);
    }
    #[test]
    fn query_validation_and_empty_cases_do_not_read_arrays() {
        unsafe {
            let mut work = 777.0f64;
            let mut info = -99i64;
            dorm22_64_(
                b"L".as_ptr(),
                b"N".as_ptr(),
                &5,
                &3,
                &2,
                &3,
                std::ptr::null(),
                &5,
                std::ptr::null_mut(),
                &5,
                &mut work,
                &-1,
                &mut info,
            );
            assert_eq!(info, 0);
            assert_eq!(work, 15.0);
            work = 777.0;
            dorm22_64_(
                b"L".as_ptr(),
                b"N".as_ptr(),
                &5,
                &3,
                &2,
                &3,
                std::ptr::null(),
                &5,
                std::ptr::null_mut(),
                &5,
                &mut work,
                &4,
                &mut info,
            );
            assert_eq!(info, -12);
            assert_eq!(work, 777.0);
            dorm22_64_(
                b"L".as_ptr(),
                b"N".as_ptr(),
                &5,
                &0,
                &2,
                &3,
                std::ptr::null(),
                &5,
                std::ptr::null_mut(),
                &5,
                &mut work,
                &-1,
                &mut info,
            );
            assert_eq!(info, 0);
            assert_eq!(work, 0.0);
            dorm22_64_(
                b"L".as_ptr(),
                b"N".as_ptr(),
                &5,
                &0,
                &2,
                &3,
                std::ptr::null(),
                &5,
                std::ptr::null_mut(),
                &5,
                &mut work,
                &5,
                &mut info,
            );
            assert_eq!(info, 0);
            assert_eq!(work, 1.0);
            dorm22_64_(
                b"L".as_ptr(),
                b"N".as_ptr(),
                &5,
                &3,
                &6,
                &-1,
                std::ptr::null(),
                &5,
                std::ptr::null_mut(),
                &5,
                &mut work,
                &-1,
                &mut info,
            );
            assert_eq!(info, -6);
            dorm22_64_(
                b"L".as_ptr(),
                b"N".as_ptr(),
                &5,
                &3,
                &4,
                &-1,
                std::ptr::null(),
                &5,
                std::ptr::null_mut(),
                &5,
                &mut work,
                &-1,
                &mut info,
            );
            assert_eq!(info, -5);
        }
    }
}
