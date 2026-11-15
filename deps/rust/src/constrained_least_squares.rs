//! Constrained least-squares compositions over generalized QR/RQ.
//! Derived from Reference LAPACK v3.12.1 SRC/dgglse.f and SRC/dggglm.f:
//! https://github.com/Reference-LAPACK/lapack/tree/v3.12.1/SRC
//! The factor/solve/update order and destructive output contracts follow those
//! drivers; complex adjoints and adapter-owned factor coefficients are used.
//! Copyright (c) 1992-2023 The University of Tennessee and The University
//! of Tennessee Research Foundation. All rights reserved.
//! Copyright (c) 2000-2023 The University of California Berkeley. All rights reserved.
//! Copyright (c) 2006-2023 The University of Colorado Denver. All rights reserved.
//! SPDX-License-Identifier: BSD-3-Clause
//! Full terms: ../licenses/LAPACK-BSD-3-Clause.txt.
use crate::householder::ReflectorScalar;
use crate::kernels::matrix_addressable;
use crate::lapack::LapackInt;
use colmat::{MatMut, MatRef};
use paste::paste;

unsafe fn error<I: LapackInt>(info: *mut I, arg: i64) {
    *info = I::from_i64(-arg);
    crate::invalid(arg as i32);
}
unsafe fn solve<T: ReflectorScalar>(n: usize, a: *const T, lda: isize, b: *mut T) -> bool {
    // TRTRS detects all zero diagonals before overwriting its right-hand side.
    for i in 0..n {
        if *a.offset(i as isize * (lda + 1)) == T::z() {
            return false;
        }
    }
    if n > 0 {
        // The public native solve uses scaled complex reciprocals; operator `/`
        // on the FFI complex scalar squares its denominator at extreme scales.
        crate::threading::with_par(|par| {
            colmat::linalg::triangular_solve::solve_upper_triangular_in_place(
                MatRef::from_raw_parts(a, n, n, 1, lda),
                MatMut::from_raw_parts_mut(b, n, 1, 1, n as isize),
                par,
            );
        });
    }
    true
}
unsafe fn lse<T: ReflectorScalar>(
    m: usize,
    n: usize,
    p: usize,
    a: *mut T,
    la: isize,
    b: *mut T,
    lb: isize,
    c: *mut T,
    d: *mut T,
    x: *mut T,
) -> i64 {
    let mn = m.min(n);
    let free = n - p;
    let mut tb = vec![T::z(); p];
    let mut ta = vec![T::z(); mn];
    crate::generalized_qr::pair(
        true,
        p,
        m,
        n,
        b,
        lb,
        tb.as_mut_ptr(),
        a,
        la,
        ta.as_mut_ptr(),
    );
    let adj = if T::COMPLEX { b'C' } else { b'T' };
    crate::qr::apply_q(
        b'L',
        adj,
        m,
        1,
        mn,
        a,
        la,
        ta.as_ptr(),
        c,
        m.max(1) as isize,
    );
    if p > 0 {
        if !solve(p, b.offset(free as isize * lb), lb, d) {
            return 1;
        }
        for j in 0..p {
            *x.add(free + j) = *d.add(j);
        }
        for i in 0..free {
            let mut z = *c.add(i);
            for j in 0..p {
                z = z - *a.offset(i as isize + (free + j) as isize * la) * *d.add(j);
            }
            *c.add(i) = z;
        }
    }
    if free > 0 {
        if !solve(free, a, la, c) {
            return 2;
        }
        for i in 0..free {
            *x.add(i) = *c.add(i);
        }
    }
    // Preserve the LAPACK C/D residual payload, including rectangular R22.
    let nr = p.min(m - free);
    let original_d: Vec<T> = (0..p).map(|j| *d.add(j)).collect();
    for i in 0..nr {
        let mut lead = T::z();
        let mut tail = T::z();
        for j in i..nr {
            lead = lead + *a.offset((free + i) as isize + (free + j) as isize * la) * original_d[j];
        }
        for j in nr..p {
            tail = tail + *a.offset((free + i) as isize + (free + j) as isize * la) * original_d[j];
        }
        *d.add(i) = lead;
        *c.add(free + i) = *c.add(free + i) - lead - tail;
    }
    crate::reverse_qr::apply_rq(b'L', adj, n, 1, p, b, lb, tb.as_ptr(), x, n as isize);
    0
}
unsafe fn glm<T: ReflectorScalar>(
    n: usize,
    m: usize,
    p: usize,
    a: *mut T,
    la: isize,
    b: *mut T,
    lb: isize,
    d: *mut T,
    x: *mut T,
    y: *mut T,
) -> i64 {
    let np = n.min(p);
    let constrained = n - m;
    let free = p - constrained;
    let mut ta = vec![T::z(); m];
    let mut tb = vec![T::z(); np];
    crate::generalized_qr::pair(
        false,
        n,
        m,
        p,
        a,
        la,
        ta.as_mut_ptr(),
        b,
        lb,
        tb.as_mut_ptr(),
    );
    let adj = if T::COMPLEX { b'C' } else { b'T' };
    crate::qr::apply_q(b'L', adj, n, 1, m, a, la, ta.as_ptr(), d, n as isize);
    if constrained > 0 {
        if !solve(
            constrained,
            b.offset(m as isize + free as isize * lb),
            lb,
            d.add(m),
        ) {
            return 1;
        }
        for i in 0..constrained {
            *y.add(free + i) = *d.add(m + i);
        }
    }
    for i in 0..free {
        *y.add(i) = T::z();
    }
    for i in 0..m {
        let mut z = *d.add(i);
        for j in 0..constrained {
            z = z - *b.offset(i as isize + (free + j) as isize * lb) * *y.add(free + j);
        }
        *d.add(i) = z;
    }
    if m > 0 {
        if !solve(m, a, la, d) {
            return 2;
        }
        for i in 0..m {
            *x.add(i) = *d.add(i);
        }
    }
    if np > 0 {
        crate::reverse_qr::apply_rq(
            b'L',
            adj,
            p,
            1,
            np,
            b.add(n - np),
            lb,
            tb.as_ptr(),
            y,
            p.max(1) as isize,
        );
    }
    0
}
unsafe fn driver<T: ReflectorScalar, I: LapackInt>(
    is_lse: bool,
    first: isize,
    second: isize,
    p: isize,
    a: *mut T,
    la: isize,
    b: *mut T,
    lb: isize,
    c: *mut T,
    d_or_x: *mut T,
    x_or_y: *mut T,
    work: *mut T,
    lwork: isize,
    info: *mut I,
) {
    *info = I::from_i64(0);
    let (rows, cols, br) = if is_lse {
        (first, second, p)
    } else {
        (first, second, first)
    };
    for (arg, ok) in [
        (1, first >= 0),
        (2, second >= 0 && (is_lse || second <= first)),
        (
            3,
            p >= 0
                && if is_lse {
                    p <= second && p >= second - first
                } else {
                    p >= first - second
                },
        ),
        (5, la >= rows.max(1)),
        (7, lb >= br.max(1)),
    ] {
        if !ok {
            error(info, arg);
            return;
        }
    }
    let n = if is_lse { second } else { first };
    let minimum = if n == 0 {
        1
    } else {
        let Some(sum) = first.checked_add(second).and_then(|s| s.checked_add(p)) else {
            error(info, 12);
            return;
        };
        sum
    };
    *work = crate::lapack::workspace::<T>(minimum);
    if lwork != -1 && lwork < minimum {
        error(info, 12);
        return;
    }
    if lwork == -1 {
        return;
    }
    if n == 0 {
        if !is_lse {
            for i in 0..second as usize {
                *d_or_x.add(i) = T::z();
            }
            for i in 0..p as usize {
                *x_or_y.add(i) = T::z();
            }
        }
        return;
    }
    let bc = if is_lse { second } else { p };
    if !matrix_addressable::<T>(rows, cols, la) {
        error(info, 5);
        return;
    }
    if !matrix_addressable::<T>(br, bc, lb) {
        error(info, 7);
        return;
    }
    let result = if is_lse {
        lse(
            first as usize,
            second as usize,
            p as usize,
            a,
            la,
            b,
            lb,
            c,
            d_or_x,
            x_or_y,
        )
    } else {
        glm(
            first as usize,
            second as usize,
            p as usize,
            a,
            la,
            b,
            lb,
            c,
            d_or_x,
            x_or_y,
        )
    };
    *info = I::from_i64(result);
}
macro_rules! exports {($p:ident,$t:ty,$i:ty,$s:tt)=>{paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p gglse $s>](m:*const $i,n:*const $i,p:*const $i,a:*mut $t,la:*const $i,b:*mut $t,lb:*const $i,c:*mut $t,d:*mut $t,x:*mut $t,work:*mut $t,lwork:*const $i,info:*mut $i){driver(true,*m as isize,*n as isize,*p as isize,a,*la as isize,b,*lb as isize,c,d,x,work,*lwork as isize,info);}
    #[no_mangle] pub unsafe extern "C" fn [<$p ggglm $s>](n:*const $i,m:*const $i,p:*const $i,a:*mut $t,la:*const $i,b:*mut $t,lb:*const $i,d:*mut $t,x:*mut $t,y:*mut $t,work:*mut $t,lwork:*const $i,info:*mut $i){driver(false,*n as isize,*m as isize,*p as isize,a,*la as isize,b,*lb as isize,d,x,y,work,*lwork as isize,info);}
}};}
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
    use crate::kernels::Scalar;
    fn near<T: Scalar>(x: T, y: T, tol: f64) {
        assert!(
            (x - y).abs1() < tol * (1.0 + y.abs1()),
            "residual {}",
            (x - y).abs1()
        );
    }
    unsafe fn check<T: ReflectorScalar>(tol: f64) {
        for m in 0..5 {
            for n in 1..5 {
                for p in 0..=n {
                    if n > m + p {
                        continue;
                    }
                    let free = n - p;
                    let la = m.max(1);
                    let lb = p.max(1);
                    let mut a = vec![T::z(); la * n];
                    let mut b = vec![T::z(); lb * n];
                    let expected: Vec<T> = (0..n)
                        .map(|j| T::parts((j + 1) as f64, 0.2 * j as f64))
                        .collect();
                    for i in 0..m {
                        for j in 0..n {
                            if i == j {
                                a[i + j * la] = T::parts(2.0, 0.3);
                            } else if j >= free {
                                a[i + j * la] = T::parts((i + j + 1) as f64 * 0.1, -0.07);
                            }
                        }
                    }
                    for i in 0..p {
                        b[i + (free + i) * lb] = T::parts(3.0, -0.2);
                    }
                    let mut c = vec![T::z(); m];
                    let mut d = vec![T::z(); p];
                    let mut x = vec![T::z(); n];
                    for i in 0..m {
                        for j in 0..n {
                            c[i] = c[i] + a[i + j * la] * expected[j];
                        }
                        if i >= free {
                            c[i] = c[i] + T::parts(0.4, -0.1);
                        }
                    }
                    for i in 0..p {
                        d[i] = b[i + (free + i) * lb] * expected[free + i];
                    }
                    assert_eq!(
                        lse(
                            m,
                            n,
                            p,
                            a.as_mut_ptr(),
                            la as isize,
                            b.as_mut_ptr(),
                            lb as isize,
                            c.as_mut_ptr(),
                            d.as_mut_ptr(),
                            x.as_mut_ptr()
                        ),
                        0
                    );
                    for j in 0..n {
                        near(x[j], expected[j], tol);
                    }
                }
            }
        }
        for n in 1..5 {
            for m in 0..=n {
                for p in (n - m)..5 {
                    let constrained = n - m;
                    let free = p - constrained;
                    let la = n;
                    let lb = n;
                    let mut a = vec![T::z(); la * m];
                    let mut b = vec![T::z(); lb * p];
                    let ex: Vec<T> = (0..m).map(|i| T::parts(i as f64 + 0.7, -0.2)).collect();
                    let ey: Vec<T> = (0..p)
                        .map(|j| {
                            if j < free {
                                T::z()
                            } else {
                                T::parts(j as f64 + 0.4, 0.3)
                            }
                        })
                        .collect();
                    for i in 0..m {
                        a[i + i * la] = T::parts(2.0, 0.2);
                    }
                    for j in 0..p {
                        for i in 0..m {
                            b[i + j * lb] = T::parts((i + j + 1) as f64 * 0.2, 0.1);
                        }
                    }
                    for i in 0..constrained {
                        b[m + i + (free + i) * lb] = T::parts(3.0, -0.2);
                    }
                    let mut d = vec![T::z(); n];
                    let mut x = vec![T::z(); m];
                    let mut y = vec![T::z(); p];
                    for i in 0..n {
                        for j in 0..m {
                            d[i] = d[i] + a[i + j * la] * ex[j];
                        }
                        for j in 0..p {
                            d[i] = d[i] + b[i + j * lb] * ey[j];
                        }
                    }
                    assert_eq!(
                        glm(
                            n,
                            m,
                            p,
                            a.as_mut_ptr(),
                            la as isize,
                            b.as_mut_ptr(),
                            lb as isize,
                            d.as_mut_ptr(),
                            x.as_mut_ptr(),
                            y.as_mut_ptr()
                        ),
                        0
                    );
                    for j in 0..m {
                        near(x[j], ex[j], tol);
                    }
                    for j in 0..p {
                        near(y[j], ey[j], tol);
                    }
                }
            }
        }
    }
    #[test]
    fn drivers_real32() {
        unsafe {
            check::<f32>(2e-4);
        }
    }
    #[test]
    fn drivers_real64() {
        unsafe {
            check::<f64>(1e-12);
        }
    }
    #[test]
    fn drivers_complex32() {
        unsafe {
            check::<colmat::c32>(2e-4);
        }
    }
    #[test]
    fn drivers_complex64() {
        unsafe {
            check::<colmat::c64>(1e-12);
        }
    }
}

#[cfg(test)]
mod scaled_solve_tests {
    use super::*;
    fn check<T: ReflectorScalar>(scales: &[f64]) {
        for &scale in scales {
            let a = [T::parts(2.0 * scale, 0.2 * scale)];
            let mut b = [T::parts(0.4 * scale, 0.8 * scale)];
            unsafe {
                assert!(solve(1, a.as_ptr(), 1, b.as_mut_ptr()));
            }
            let expected = T::parts(0.96 / 4.04, 1.52 / 4.04);
            assert!((b[0] - expected).abs1() < 100.0 * T::EPS);
        }
    }
    #[test]
    fn complex32_extreme_division() {
        check::<colmat::c32>(&[1.0, 1e-38, 1e30]);
    }
    #[test]
    fn complex64_extreme_division() {
        check::<colmat::c64>(&[1.0, 1e-308, 1e300]);
    }
}
