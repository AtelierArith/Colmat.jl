use super::*;

fn check_copy<T: Scalar>(
    imag: T,
    tol: f64,
    call: impl Fn(bool, bool, i32, i32, i64, i64, T, *mut T, i64, *mut T, i64),
) {
    let value = |i: usize, j: usize| {
        T::real((1 + i + 3 * j) as f64 / 8.0) + imag * T::real((i as f64 - j as f64) / 16.0)
    };
    let sentinel = T::real(97.0);
    for (m, n) in [(2usize, 5usize), (5, 2), (1, 4)] {
        for order in [101, 102] {
            for t in 111..=114 {
                for alpha in [T::z(), T::o(), T::real(0.75) + imag * T::real(0.25)] {
                    let trans = t == 112 || t == 113;
                    let (rows, cols) = if trans { (n, m) } else { (m, n) };
                    let la = if order == 101 { n + 2 } else { m + 2 };
                    let lb = if order == 101 { cols + 3 } else { rows + 3 };
                    let index = |i, j, ld| if order == 101 { i * ld + j } else { i + j * ld };
                    let length = (la * m.max(n)).max(lb * rows.max(cols)) + 5;
                    let mut source = vec![sentinel; length];
                    for i in 0..m {
                        for j in 0..n {
                            source[index(i, j, la)] = value(i, j);
                        }
                    }
                    for inplace in [false, true] {
                        for wide in [false, true] {
                            let mut a = source.clone();
                            let mut b = vec![sentinel; length];
                            let mut expected = if inplace { source.clone() } else { b.clone() };
                            for i in 0..rows {
                                for j in 0..cols {
                                    let v = if trans { value(j, i) } else { value(i, j) };
                                    expected[index(i, j, lb)] =
                                        alpha * if t == 113 || t == 114 { v.cj() } else { v };
                                }
                            }
                            call(
                                wide,
                                inplace,
                                order,
                                t,
                                m as i64,
                                n as i64,
                                alpha,
                                a.as_mut_ptr(),
                                la as i64,
                                b.as_mut_ptr(),
                                lb as i64,
                            );
                            let actual = if inplace {
                                &a
                            } else {
                                assert!(a == source);
                                &b
                            };
                            for (actual, expected) in actual.iter().zip(&expected) {
                                assert!((*actual-*expected).magnitude()<tol,"matcopy wide={wide} inplace={inplace} order={order} t={t} m={m} n={n}");
                            }
                        }
                    }
                }
            }
        }
    }
}

fn check_add<T: Scalar>(
    imag: T,
    tol: f64,
    call: impl Fn(bool, i32, i64, i64, T, *const T, i64, T, *mut T, i64),
) {
    let value = |i: usize, j: usize| {
        T::real((1 + i + 3 * j) as f64 / 8.0) + imag * T::real((i as f64 - j as f64) / 16.0)
    };
    let nonzero_alpha = T::real(0.75) + imag * T::real(0.25);
    let nonzero_beta = T::real(-0.5) + imag * T::real(0.125);
    for (m, n) in [(2usize, 5usize), (5, 2), (1, 4)] {
        for order in [101, 102] {
            for (alpha, beta) in [
                (T::z(), T::z()),
                (T::z(), T::o()),
                (nonzero_alpha, T::z()),
                (nonzero_alpha, nonzero_beta),
            ] {
                let la = if order == 101 { n + 2 } else { m + 2 };
                let lb = la + 1;
                let index = |i, j, ld| if order == 101 { i * ld + j } else { i + j * ld };
                let sentinel = T::real(97.0);
                let mut a = vec![sentinel; la * m.max(n)];
                let mut initial = vec![sentinel; lb * m.max(n)];
                let mut expected = initial.clone();
                for i in 0..m {
                    for j in 0..n {
                        a[index(i, j, la)] = value(i, j);
                        initial[index(i, j, lb)] = if beta == T::z() {
                            T::real(f64::NAN)
                        } else {
                            value(i + 1, j + 2)
                        };
                        expected[index(i, j, lb)] =
                            alpha * value(i, j) + beta * value(i + 1, j + 2);
                    }
                }
                for wide in [false, true] {
                    let mut b = initial.clone();
                    call(
                        wide,
                        order,
                        m as i64,
                        n as i64,
                        alpha,
                        if alpha == T::z() {
                            core::ptr::null()
                        } else {
                            a.as_ptr()
                        },
                        la as i64,
                        beta,
                        b.as_mut_ptr(),
                        lb as i64,
                    );
                    for (actual, expected) in b.iter().zip(&expected) {
                        assert!(
                            (*actual - *expected).magnitude() < tol,
                            "geadd wide={wide} order={order} m={m} n={n}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn copy_s() {
    check_copy::<f32>(
        0.0f32,
        1e-5,
        |wide, inplace, order, t, m, n, alpha, a, la, b, lb| unsafe {
            match (wide, inplace) {
                (false, false) => {
                    cblas_somatcopy(order, t, m as _, n as _, alpha, a, la as _, b, lb as _)
                }
                (false, true) => {
                    cblas_simatcopy(order, t, m as _, n as _, alpha, a, la as _, lb as _)
                }
                (true, false) => {
                    cblas_somatcopy64_(order, t, m as _, n as _, alpha, a, la as _, b, lb as _)
                }
                (true, true) => {
                    cblas_simatcopy64_(order, t, m as _, n as _, alpha, a, la as _, lb as _)
                }
            }
        },
    );
}

#[test]
fn add_s() {
    check_add::<f32>(
        0.0f32,
        1e-5,
        |wide, order, m, n, alpha, a, la, beta, b, lb| unsafe {
            if wide {
                cblas_sgeadd64_(order, m as _, n as _, alpha, a, la as _, beta, b, lb as _);
            } else {
                cblas_sgeadd(order, m as _, n as _, alpha, a, la as _, beta, b, lb as _);
            }
        },
    );
}

#[test]
fn copy_d() {
    check_copy::<f64>(
        0.0f64,
        1e-12,
        |wide, inplace, order, t, m, n, alpha, a, la, b, lb| unsafe {
            match (wide, inplace) {
                (false, false) => {
                    cblas_domatcopy(order, t, m as _, n as _, alpha, a, la as _, b, lb as _)
                }
                (false, true) => {
                    cblas_dimatcopy(order, t, m as _, n as _, alpha, a, la as _, lb as _)
                }
                (true, false) => {
                    cblas_domatcopy64_(order, t, m as _, n as _, alpha, a, la as _, b, lb as _)
                }
                (true, true) => {
                    cblas_dimatcopy64_(order, t, m as _, n as _, alpha, a, la as _, lb as _)
                }
            }
        },
    );
}

#[test]
fn add_d() {
    check_add::<f64>(
        0.0f64,
        1e-12,
        |wide, order, m, n, alpha, a, la, beta, b, lb| unsafe {
            if wide {
                cblas_dgeadd64_(order, m as _, n as _, alpha, a, la as _, beta, b, lb as _);
            } else {
                cblas_dgeadd(order, m as _, n as _, alpha, a, la as _, beta, b, lb as _);
            }
        },
    );
}

#[test]
fn copy_c() {
    check_copy::<c32>(
        c32::new(0.0, 1.0),
        1e-5,
        |wide, inplace, order, t, m, n, alpha, a, la, b, lb| unsafe {
            match (wide, inplace) {
                (false, false) => {
                    cblas_comatcopy(order, t, m as _, n as _, &alpha, a, la as _, b, lb as _)
                }
                (false, true) => {
                    cblas_cimatcopy(order, t, m as _, n as _, &alpha, a, la as _, lb as _)
                }
                (true, false) => {
                    cblas_comatcopy64_(order, t, m as _, n as _, &alpha, a, la as _, b, lb as _)
                }
                (true, true) => {
                    cblas_cimatcopy64_(order, t, m as _, n as _, &alpha, a, la as _, lb as _)
                }
            }
        },
    );
}

#[test]
fn add_c() {
    check_add::<c32>(
        c32::new(0.0, 1.0),
        1e-5,
        |wide, order, m, n, alpha, a, la, beta, b, lb| unsafe {
            if wide {
                cblas_cgeadd64_(order, m as _, n as _, &alpha, a, la as _, &beta, b, lb as _);
            } else {
                cblas_cgeadd(order, m as _, n as _, &alpha, a, la as _, &beta, b, lb as _);
            }
        },
    );
}

#[test]
fn copy_z() {
    check_copy::<c64>(
        c64::new(0.0, 1.0),
        1e-12,
        |wide, inplace, order, t, m, n, alpha, a, la, b, lb| unsafe {
            match (wide, inplace) {
                (false, false) => {
                    cblas_zomatcopy(order, t, m as _, n as _, &alpha, a, la as _, b, lb as _)
                }
                (false, true) => {
                    cblas_zimatcopy(order, t, m as _, n as _, &alpha, a, la as _, lb as _)
                }
                (true, false) => {
                    cblas_zomatcopy64_(order, t, m as _, n as _, &alpha, a, la as _, b, lb as _)
                }
                (true, true) => {
                    cblas_zimatcopy64_(order, t, m as _, n as _, &alpha, a, la as _, lb as _)
                }
            }
        },
    );
}

#[test]
fn add_z() {
    check_add::<c64>(
        c64::new(0.0, 1.0),
        1e-12,
        |wide, order, m, n, alpha, a, la, beta, b, lb| unsafe {
            if wide {
                cblas_zgeadd64_(order, m as _, n as _, &alpha, a, la as _, &beta, b, lb as _);
            } else {
                cblas_zgeadd(order, m as _, n as _, &alpha, a, la as _, &beta, b, lb as _);
            }
        },
    );
}

#[test]
fn matrix_extension_headers_precede_pointer_access() {
    unsafe {
        cblas_zomatcopy64_(
            101,
            113,
            0,
            3,
            core::ptr::null(),
            core::ptr::null(),
            3,
            core::ptr::null_mut(),
            1,
        );
        assert_eq!(crate::colmat_blas_last_error(), 0);
        cblas_zimatcopy64_(
            101,
            112,
            2,
            3,
            core::ptr::null(),
            core::ptr::null_mut(),
            3,
            1,
        );
        assert_eq!(crate::colmat_blas_last_error(), 8);
        cblas_zomatcopy64_(
            101,
            112,
            2,
            3,
            core::ptr::null(),
            core::ptr::null(),
            3,
            core::ptr::null_mut(),
            1,
        );
        assert_eq!(crate::colmat_blas_last_error(), 9);
        cblas_zimatcopy64_(
            102,
            111,
            2,
            3,
            core::ptr::null(),
            core::ptr::null_mut(),
            i64::MAX,
            2,
        );
        assert_eq!(crate::colmat_blas_last_error(), 7);
        cblas_zgeadd64_(
            101,
            2,
            3,
            core::ptr::null(),
            core::ptr::null(),
            2,
            core::ptr::null(),
            core::ptr::null_mut(),
            3,
        );
        assert_eq!(crate::colmat_blas_last_error(), 6);
        cblas_zgeadd64_(
            101,
            0,
            3,
            core::ptr::null(),
            core::ptr::null(),
            3,
            core::ptr::null(),
            core::ptr::null_mut(),
            3,
        );
        assert_eq!(crate::colmat_blas_last_error(), 0);
    }
}
