use super::*;

fn check<T: Scalar>(
    imag: T,
    tol: f64,
    call: impl Fn(
        bool,
        bool,
        i32,
        i32,
        i32,
        i32,
        i64,
        i64,
        T,
        *const T,
        i64,
        *const T,
        i64,
        T,
        *mut T,
        i64,
    ),
) {
    let value = |i: usize, j: usize| {
        T::real((1 + i + 3 * j) as f64 / 8.0) + imag * T::real((i as f64 - j as f64) / 16.0)
    };
    let alpha = T::real(0.75) + imag * T::real(0.25);
    let n = 3usize;
    for order in [101, 102] {
        for u in [121, 122] {
            for ta in 111..=114 {
                for tb in 111..=114 {
                    for k in [0usize, 4] {
                        for zero_beta in [false, true] {
                            let trans = |t| t == 112 || t == 113;
                            let (ar, ac) = if trans(ta) { (k, n) } else { (n, k) };
                            let (br, bc) = if trans(tb) { (n, k) } else { (k, n) };
                            let la = if order == 101 { ac + 2 } else { ar + 2 };
                            let lb = if order == 101 { bc + 3 } else { br + 3 };
                            let lc = n + 2;
                            let index =
                                |i, j, ld| if order == 101 { i * ld + j } else { i + j * ld };
                            let sentinel = T::real(97.0);
                            let mut a = vec![T::real(f64::NAN); la * ar.max(ac)];
                            let mut b = vec![T::real(f64::NAN); lb * br.max(bc)];
                            for i in 0..ar {
                                for j in 0..ac {
                                    a[index(i, j, la)] = value(i, j);
                                }
                            }
                            for i in 0..br {
                                for j in 0..bc {
                                    b[index(i, j, lb)] = value(i + 1, j + 2);
                                }
                            }
                            let op = |is_b: bool, i, j, t| {
                                let (i, j) = if trans(t) { (j, i) } else { (i, j) };
                                let v = if is_b {
                                    value(i + 1, j + 2)
                                } else {
                                    value(i, j)
                                };
                                if t == 113 || t == 114 {
                                    v.cj()
                                } else {
                                    v
                                }
                            };
                            let beta = if zero_beta {
                                T::z()
                            } else {
                                T::real(-0.5) + imag * T::real(0.125)
                            };
                            let mut initial = vec![sentinel; lc * n];
                            let mut expected = initial.clone();
                            for i in 0..n {
                                for j in 0..n {
                                    if (u == 121 && i <= j) || (u == 122 && i >= j) {
                                        initial[index(i, j, lc)] = if zero_beta {
                                            T::real(f64::NAN)
                                        } else {
                                            value(i, j + 1)
                                        };
                                        let mut sum = T::z();
                                        for l in 0..k {
                                            sum += op(false, i, l, ta) * op(true, l, j, tb);
                                        }
                                        expected[index(i, j, lc)] = alpha * sum
                                            + if zero_beta {
                                                T::z()
                                            } else {
                                                beta * value(i, j + 1)
                                            };
                                    }
                                }
                            }
                            for wide in [false, true] {
                                for alias in [false, true] {
                                    let mut c = initial.clone();
                                    call(
                                        wide,
                                        alias,
                                        order,
                                        u,
                                        ta,
                                        tb,
                                        n as i64,
                                        k as i64,
                                        alpha,
                                        a.as_ptr(),
                                        la as i64,
                                        b.as_ptr(),
                                        lb as i64,
                                        beta,
                                        c.as_mut_ptr(),
                                        lc as i64,
                                    );
                                    for (actual, expected) in c.iter().zip(&expected) {
                                        assert!((*actual-*expected).magnitude()<tol,"triangularGEMM wide={wide} alias={alias} order={order} u={u} ta={ta} tb={tb} k={k} beta0={zero_beta}");
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn s() {
    check::<f32>(
        0.0f32,
        1e-5,
        |wide, alias, order, u, ta, tb, n, k, alpha, a, la, b, lb, beta, c, lc| unsafe {
            match (wide, alias) {
                (false, false) => cblas_sgemmt(
                    order, u, ta, tb, n as _, k as _, alpha, a, la as _, b, lb as _, beta, c,
                    lc as _,
                ),
                (false, true) => cblas_sgemmtr(
                    order, u, ta, tb, n as _, k as _, alpha, a, la as _, b, lb as _, beta, c,
                    lc as _,
                ),
                (true, false) => cblas_sgemmt64_(
                    order, u, ta, tb, n as _, k as _, alpha, a, la as _, b, lb as _, beta, c,
                    lc as _,
                ),
                (true, true) => cblas_sgemmtr64_(
                    order, u, ta, tb, n as _, k as _, alpha, a, la as _, b, lb as _, beta, c,
                    lc as _,
                ),
            }
        },
    );
}

#[test]
fn d() {
    check::<f64>(
        0.0f64,
        1e-12,
        |wide, alias, order, u, ta, tb, n, k, alpha, a, la, b, lb, beta, c, lc| unsafe {
            match (wide, alias) {
                (false, false) => cblas_dgemmt(
                    order, u, ta, tb, n as _, k as _, alpha, a, la as _, b, lb as _, beta, c,
                    lc as _,
                ),
                (false, true) => cblas_dgemmtr(
                    order, u, ta, tb, n as _, k as _, alpha, a, la as _, b, lb as _, beta, c,
                    lc as _,
                ),
                (true, false) => cblas_dgemmt64_(
                    order, u, ta, tb, n as _, k as _, alpha, a, la as _, b, lb as _, beta, c,
                    lc as _,
                ),
                (true, true) => cblas_dgemmtr64_(
                    order, u, ta, tb, n as _, k as _, alpha, a, la as _, b, lb as _, beta, c,
                    lc as _,
                ),
            }
        },
    );
}

#[test]
fn c() {
    check::<c32>(
        c32::new(0.0, 1.0),
        1e-5,
        |wide, alias, order, u, ta, tb, n, k, alpha, a, la, b, lb, beta, c, lc| unsafe {
            match (wide, alias) {
                (false, false) => cblas_cgemmt(
                    order, u, ta, tb, n as _, k as _, &alpha, a, la as _, b, lb as _, &beta, c,
                    lc as _,
                ),
                (false, true) => cblas_cgemmtr(
                    order, u, ta, tb, n as _, k as _, &alpha, a, la as _, b, lb as _, &beta, c,
                    lc as _,
                ),
                (true, false) => cblas_cgemmt64_(
                    order, u, ta, tb, n as _, k as _, &alpha, a, la as _, b, lb as _, &beta, c,
                    lc as _,
                ),
                (true, true) => cblas_cgemmtr64_(
                    order, u, ta, tb, n as _, k as _, &alpha, a, la as _, b, lb as _, &beta, c,
                    lc as _,
                ),
            }
        },
    );
}

#[test]
fn z() {
    check::<c64>(
        c64::new(0.0, 1.0),
        1e-12,
        |wide, alias, order, u, ta, tb, n, k, alpha, a, la, b, lb, beta, c, lc| unsafe {
            match (wide, alias) {
                (false, false) => cblas_zgemmt(
                    order, u, ta, tb, n as _, k as _, &alpha, a, la as _, b, lb as _, &beta, c,
                    lc as _,
                ),
                (false, true) => cblas_zgemmtr(
                    order, u, ta, tb, n as _, k as _, &alpha, a, la as _, b, lb as _, &beta, c,
                    lc as _,
                ),
                (true, false) => cblas_zgemmt64_(
                    order, u, ta, tb, n as _, k as _, &alpha, a, la as _, b, lb as _, &beta, c,
                    lc as _,
                ),
                (true, true) => cblas_zgemmtr64_(
                    order, u, ta, tb, n as _, k as _, &alpha, a, la as _, b, lb as _, &beta, c,
                    lc as _,
                ),
            }
        },
    );
}

#[test]
fn headers_empty_and_zero_product_do_not_access_unused_storage() {
    unsafe {
        cblas_zgemmt64_(
            101,
            121,
            111,
            111,
            0,
            3,
            core::ptr::null(),
            core::ptr::null(),
            3,
            core::ptr::null(),
            1,
            core::ptr::null(),
            core::ptr::null_mut(),
            1,
        );
        assert_eq!(crate::colmat_blas_last_error(), 0);
        cblas_zgemmtr64_(
            101,
            121,
            111,
            111,
            2,
            3,
            core::ptr::null(),
            core::ptr::null(),
            2,
            core::ptr::null(),
            2,
            core::ptr::null(),
            core::ptr::null_mut(),
            2,
        );
        assert_eq!(crate::colmat_blas_last_error(), 9);
        cblas_zgemmt64_(
            102,
            121,
            111,
            111,
            2,
            3,
            core::ptr::null(),
            core::ptr::null(),
            2,
            core::ptr::null(),
            3,
            core::ptr::null(),
            core::ptr::null_mut(),
            1,
        );
        assert_eq!(crate::colmat_blas_last_error(), 14);
        let alpha = c64::new(0.75, 0.25);
        let zero = c64::new(0.0, 0.0);
        let sentinel = c64::new(97.0, 0.0);
        let nan = c64::new(f64::NAN, f64::NAN);
        let mut c = [nan, nan, sentinel, sentinel, nan, sentinel];
        cblas_zgemmtr64_(
            101,
            121,
            111,
            111,
            2,
            0,
            &alpha,
            core::ptr::null(),
            1,
            core::ptr::null(),
            2,
            &zero,
            c.as_mut_ptr(),
            3,
        );
        assert_eq!(c, [zero, zero, sentinel, sentinel, zero, sentinel]);
        assert_eq!(crate::colmat_blas_last_error(), 0);
    }
}
