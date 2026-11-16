use super::*;

fn check_rankk<T: Scalar>(
    h: bool,
    two: bool,
    imag: T,
    tolerance: f64,
    call: impl Fn(bool, i32, i32, i32, i64, i64, T, *const T, i64, *const T, i64, T, *mut T, i64),
) {
    let value = |i: usize, j: usize| {
        T::real((1 + i + 3 * j) as f64 / 8.0) + imag * T::real((i as f64 - j as f64) / 16.0)
    };
    let alpha = T::real(0.75)
        + if h && !two {
            T::z()
        } else {
            imag * T::real(0.25)
        };
    let modes: &[i32] = if !T::COMPLEX {
        &[111, 112, 113, 114]
    } else if h {
        &[111, 113]
    } else {
        &[111, 112]
    };
    for order in [101, 102] {
        for u in [121, 122] {
            for &t in modes {
                for k in [0usize, 4] {
                    for zero_beta in [false, true] {
                        let n = 3usize;
                        let trans = t == 112 || t == 113;
                        let (rows, cols) = if trans { (k, n) } else { (n, k) };
                        let lda = if order == 101 { cols + 2 } else { rows + 2 };
                        let ldb = lda + 1;
                        let ldc = n + 2;
                        let index = |i, j, ld| if order == 101 { i * ld + j } else { i + j * ld };
                        let sentinel = T::real(97.0);
                        let mut a = vec![sentinel; lda * rows.max(cols)];
                        let mut b = vec![sentinel; ldb * rows.max(cols)];
                        for i in 0..rows {
                            for j in 0..cols {
                                a[index(i, j, lda)] = value(i, j);
                                b[index(i, j, ldb)] = value(i + 1, j + 2);
                            }
                        }
                        let beta = if zero_beta {
                            T::z()
                        } else {
                            T::real(-0.5) + if h { T::z() } else { imag * T::real(0.125) }
                        };
                        let mut initial = vec![sentinel; ldc * n];
                        let mut expected = initial.clone();
                        let op = |is_b: bool, i, j| {
                            let (i, j) = if trans { (j, i) } else { (i, j) };
                            let v = if is_b {
                                value(i + 1, j + 2)
                            } else {
                                value(i, j)
                            };
                            if t == 113 {
                                v.cj()
                            } else {
                                v
                            }
                        };
                        for i in 0..n {
                            for j in 0..n {
                                if (u == 121 && i <= j) || (u == 122 && i >= j) {
                                    let old = if h && i == j {
                                        T::real(value(i, j).re())
                                    } else {
                                        value(i, j)
                                    };
                                    initial[index(i, j, ldc)] =
                                        if zero_beta { T::real(f64::NAN) } else { old };
                                    let mut sum = T::z();
                                    for l in 0..k {
                                        let rhs = op(two, j, l);
                                        sum += alpha
                                            * op(false, i, l)
                                            * if h { rhs.cj() } else { rhs };
                                        if two {
                                            let rhs = op(false, j, l);
                                            sum += if h { alpha.cj() } else { alpha }
                                                * op(true, i, l)
                                                * if h { rhs.cj() } else { rhs };
                                        }
                                    }
                                    let v = sum + if zero_beta { T::z() } else { beta * old };
                                    expected[index(i, j, ldc)] =
                                        if h && i == j { T::real(v.re()) } else { v };
                                }
                            }
                        }
                        for wide in [false, true] {
                            let mut c = initial.clone();
                            call(
                                wide,
                                order,
                                u,
                                t,
                                n as i64,
                                k as i64,
                                alpha,
                                a.as_ptr(),
                                lda as i64,
                                b.as_ptr(),
                                ldb as i64,
                                beta,
                                c.as_mut_ptr(),
                                ldc as i64,
                            );
                            for (actual, expected) in c.iter().zip(&expected) {
                                assert!((*actual-*expected).magnitude()<tolerance,"rankk h={h} two={two} wide={wide} order={order} u={u} t={t} k={k} beta0={zero_beta}");
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn ssyrk() {
    check_rankk::<f32>(
        false,
        false,
        0.0f32,
        1e-5,
        |wide, order, u, t, n, k, alpha, a, lda, b, ldb, beta, c, ldc| unsafe {
            let _ = (b, ldb);
            if wide {
                cblas_ssyrk64_(
                    order, u, t, n as _, k as _, alpha, a, lda as _, beta, c, ldc as _,
                );
            } else {
                cblas_ssyrk(
                    order, u, t, n as _, k as _, alpha, a, lda as _, beta, c, ldc as _,
                );
            }
        },
    );
}

#[test]
fn ssyr2k() {
    check_rankk::<f32>(
        false,
        true,
        0.0f32,
        1e-5,
        |wide, order, u, t, n, k, alpha, a, lda, b, ldb, beta, c, ldc| unsafe {
            let _ = (b, ldb);
            if wide {
                cblas_ssyr2k64_(
                    order, u, t, n as _, k as _, alpha, a, lda as _, b, ldb as _, beta, c, ldc as _,
                );
            } else {
                cblas_ssyr2k(
                    order, u, t, n as _, k as _, alpha, a, lda as _, b, ldb as _, beta, c, ldc as _,
                );
            }
        },
    );
}

#[test]
fn dsyrk() {
    check_rankk::<f64>(
        false,
        false,
        0.0f64,
        1e-12,
        |wide, order, u, t, n, k, alpha, a, lda, b, ldb, beta, c, ldc| unsafe {
            let _ = (b, ldb);
            if wide {
                cblas_dsyrk64_(
                    order, u, t, n as _, k as _, alpha, a, lda as _, beta, c, ldc as _,
                );
            } else {
                cblas_dsyrk(
                    order, u, t, n as _, k as _, alpha, a, lda as _, beta, c, ldc as _,
                );
            }
        },
    );
}

#[test]
fn dsyr2k() {
    check_rankk::<f64>(
        false,
        true,
        0.0f64,
        1e-12,
        |wide, order, u, t, n, k, alpha, a, lda, b, ldb, beta, c, ldc| unsafe {
            let _ = (b, ldb);
            if wide {
                cblas_dsyr2k64_(
                    order, u, t, n as _, k as _, alpha, a, lda as _, b, ldb as _, beta, c, ldc as _,
                );
            } else {
                cblas_dsyr2k(
                    order, u, t, n as _, k as _, alpha, a, lda as _, b, ldb as _, beta, c, ldc as _,
                );
            }
        },
    );
}

#[test]
fn csyrk() {
    check_rankk::<c32>(
        false,
        false,
        c32::new(0.0, 1.0),
        1e-5,
        |wide, order, u, t, n, k, alpha, a, lda, b, ldb, beta, c, ldc| unsafe {
            let _ = (b, ldb);
            if wide {
                cblas_csyrk64_(
                    order, u, t, n as _, k as _, &alpha, a, lda as _, &beta, c, ldc as _,
                );
            } else {
                cblas_csyrk(
                    order, u, t, n as _, k as _, &alpha, a, lda as _, &beta, c, ldc as _,
                );
            }
        },
    );
}

#[test]
fn csyr2k() {
    check_rankk::<c32>(
        false,
        true,
        c32::new(0.0, 1.0),
        1e-5,
        |wide, order, u, t, n, k, alpha, a, lda, b, ldb, beta, c, ldc| unsafe {
            let _ = (b, ldb);
            if wide {
                cblas_csyr2k64_(
                    order, u, t, n as _, k as _, &alpha, a, lda as _, b, ldb as _, &beta, c,
                    ldc as _,
                );
            } else {
                cblas_csyr2k(
                    order, u, t, n as _, k as _, &alpha, a, lda as _, b, ldb as _, &beta, c,
                    ldc as _,
                );
            }
        },
    );
}

#[test]
fn cherk() {
    check_rankk::<c32>(
        true,
        false,
        c32::new(0.0, 1.0),
        1e-5,
        |wide, order, u, t, n, k, alpha, a, lda, b, ldb, beta, c, ldc| unsafe {
            let _ = (b, ldb);
            if wide {
                cblas_cherk64_(
                    order, u, t, n as _, k as _, alpha.re, a, lda as _, beta.re, c, ldc as _,
                );
            } else {
                cblas_cherk(
                    order, u, t, n as _, k as _, alpha.re, a, lda as _, beta.re, c, ldc as _,
                );
            }
        },
    );
}

#[test]
fn cher2k() {
    check_rankk::<c32>(
        true,
        true,
        c32::new(0.0, 1.0),
        1e-5,
        |wide, order, u, t, n, k, alpha, a, lda, b, ldb, beta, c, ldc| unsafe {
            let _ = (b, ldb);
            if wide {
                cblas_cher2k64_(
                    order, u, t, n as _, k as _, &alpha, a, lda as _, b, ldb as _, beta.re, c,
                    ldc as _,
                );
            } else {
                cblas_cher2k(
                    order, u, t, n as _, k as _, &alpha, a, lda as _, b, ldb as _, beta.re, c,
                    ldc as _,
                );
            }
        },
    );
}

#[test]
fn zsyrk() {
    check_rankk::<c64>(
        false,
        false,
        c64::new(0.0, 1.0),
        1e-12,
        |wide, order, u, t, n, k, alpha, a, lda, b, ldb, beta, c, ldc| unsafe {
            let _ = (b, ldb);
            if wide {
                cblas_zsyrk64_(
                    order, u, t, n as _, k as _, &alpha, a, lda as _, &beta, c, ldc as _,
                );
            } else {
                cblas_zsyrk(
                    order, u, t, n as _, k as _, &alpha, a, lda as _, &beta, c, ldc as _,
                );
            }
        },
    );
}

#[test]
fn zsyr2k() {
    check_rankk::<c64>(
        false,
        true,
        c64::new(0.0, 1.0),
        1e-12,
        |wide, order, u, t, n, k, alpha, a, lda, b, ldb, beta, c, ldc| unsafe {
            let _ = (b, ldb);
            if wide {
                cblas_zsyr2k64_(
                    order, u, t, n as _, k as _, &alpha, a, lda as _, b, ldb as _, &beta, c,
                    ldc as _,
                );
            } else {
                cblas_zsyr2k(
                    order, u, t, n as _, k as _, &alpha, a, lda as _, b, ldb as _, &beta, c,
                    ldc as _,
                );
            }
        },
    );
}

#[test]
fn zherk() {
    check_rankk::<c64>(
        true,
        false,
        c64::new(0.0, 1.0),
        1e-12,
        |wide, order, u, t, n, k, alpha, a, lda, b, ldb, beta, c, ldc| unsafe {
            let _ = (b, ldb);
            if wide {
                cblas_zherk64_(
                    order, u, t, n as _, k as _, alpha.re, a, lda as _, beta.re, c, ldc as _,
                );
            } else {
                cblas_zherk(
                    order, u, t, n as _, k as _, alpha.re, a, lda as _, beta.re, c, ldc as _,
                );
            }
        },
    );
}

#[test]
fn zher2k() {
    check_rankk::<c64>(
        true,
        true,
        c64::new(0.0, 1.0),
        1e-12,
        |wide, order, u, t, n, k, alpha, a, lda, b, ldb, beta, c, ldc| unsafe {
            let _ = (b, ldb);
            if wide {
                cblas_zher2k64_(
                    order, u, t, n as _, k as _, &alpha, a, lda as _, b, ldb as _, beta.re, c,
                    ldc as _,
                );
            } else {
                cblas_zher2k(
                    order, u, t, n as _, k as _, &alpha, a, lda as _, b, ldb as _, beta.re, c,
                    ldc as _,
                );
            }
        },
    );
}

fn check_rank2<T: Scalar>(
    h: bool,
    imag: T,
    tolerance: f64,
    call: impl Fn(bool, i32, i32, i64, T, *const T, i64, *const T, i64, *mut T, i64),
) {
    let n = 3usize;
    let value = |i: usize, j: usize| {
        T::real((1 + i + 3 * j) as f64 / 8.0) + imag * T::real((i as f64 - j as f64) / 16.0)
    };
    let alpha = T::real(0.75) + imag * T::real(0.25);
    for order in [101, 102] {
        for u in [121, 122] {
            for ix in [-2i64, 1, 3] {
                for iy in [-3i64, 1, 2] {
                    let lda = n + 2;
                    let index = |i, j| {
                        if order == 101 {
                            i * lda + j
                        } else {
                            i + j * lda
                        }
                    };
                    let vi = |i: usize, inc: i64| {
                        if inc < 0 {
                            (n - 1 - i) * (-inc) as usize
                        } else {
                            i * inc as usize
                        }
                    };
                    let sentinel = T::real(97.0);
                    let mut x = vec![sentinel; 1 + (n - 1) * ix.unsigned_abs() as usize];
                    let mut y = vec![sentinel; 1 + (n - 1) * iy.unsigned_abs() as usize];
                    for i in 0..n {
                        x[vi(i, ix)] = value(i, 1);
                        y[vi(i, iy)] = value(i, 2);
                    }
                    let mut initial = vec![sentinel; lda * n];
                    let mut expected = initial.clone();
                    for i in 0..n {
                        for j in 0..n {
                            if (u == 121 && i <= j) || (u == 122 && i >= j) {
                                let old = if h && i == j {
                                    T::real(value(i, j).re())
                                } else {
                                    value(i, j)
                                };
                                initial[index(i, j)] = old;
                                let (xj, yj) = (value(j, 1), value(j, 2));
                                let v = old
                                    + alpha * value(i, 1) * if h { yj.cj() } else { yj }
                                    + if h { alpha.cj() } else { alpha }
                                        * value(i, 2)
                                        * if h { xj.cj() } else { xj };
                                expected[index(i, j)] =
                                    if h && i == j { T::real(v.re()) } else { v };
                            }
                        }
                    }
                    for wide in [false, true] {
                        let mut a = initial.clone();
                        call(
                            wide,
                            order,
                            u,
                            n as i64,
                            alpha,
                            x.as_ptr(),
                            ix,
                            y.as_ptr(),
                            iy,
                            a.as_mut_ptr(),
                            lda as i64,
                        );
                        for (actual, expected) in a.iter().zip(&expected) {
                            assert!(
                                (*actual - *expected).magnitude() < tolerance,
                                "rank2 h={h} wide={wide} order={order} u={u} ix={ix} iy={iy}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn ssyr2() {
    check_rank2::<f32>(
        false,
        0.0f32,
        1e-5,
        |wide, order, u, n, alpha, x, ix, y, iy, a, lda| unsafe {
            if wide {
                cblas_ssyr264_(order, u, n as _, alpha, x, ix as _, y, iy as _, a, lda as _);
            } else {
                cblas_ssyr2(order, u, n as _, alpha, x, ix as _, y, iy as _, a, lda as _);
            }
        },
    );
}

#[test]
fn dsyr2() {
    check_rank2::<f64>(
        false,
        0.0f64,
        1e-12,
        |wide, order, u, n, alpha, x, ix, y, iy, a, lda| unsafe {
            if wide {
                cblas_dsyr264_(order, u, n as _, alpha, x, ix as _, y, iy as _, a, lda as _);
            } else {
                cblas_dsyr2(order, u, n as _, alpha, x, ix as _, y, iy as _, a, lda as _);
            }
        },
    );
}

#[test]
fn cher2() {
    check_rank2::<c32>(
        true,
        c32::new(0.0, 1.0),
        1e-5,
        |wide, order, u, n, alpha, x, ix, y, iy, a, lda| unsafe {
            if wide {
                cblas_cher264_(
                    order, u, n as _, &alpha, x, ix as _, y, iy as _, a, lda as _,
                );
            } else {
                cblas_cher2(
                    order, u, n as _, &alpha, x, ix as _, y, iy as _, a, lda as _,
                );
            }
        },
    );
}

#[test]
fn zher2() {
    check_rank2::<c64>(
        true,
        c64::new(0.0, 1.0),
        1e-12,
        |wide, order, u, n, alpha, x, ix, y, iy, a, lda| unsafe {
            if wide {
                cblas_zher264_(
                    order, u, n as _, &alpha, x, ix as _, y, iy as _, a, lda as _,
                );
            } else {
                cblas_zher2(
                    order, u, n as _, &alpha, x, ix as _, y, iy as _, a, lda as _,
                );
            }
        },
    );
}

#[test]
fn rank_headers_and_zero_inner_dimension_ignore_payloads() {
    unsafe {
        cblas_zsyrk64_(
            101,
            121,
            113,
            2,
            3,
            core::ptr::null(),
            core::ptr::null(),
            3,
            core::ptr::null(),
            core::ptr::null_mut(),
            2,
        );
        assert_eq!(crate::colmat_blas_last_error(), 3);
        cblas_zherk64_(
            102,
            121,
            112,
            2,
            3,
            1.0,
            core::ptr::null(),
            3,
            0.0,
            core::ptr::null_mut(),
            2,
        );
        assert_eq!(crate::colmat_blas_last_error(), 3);
        cblas_zher2k64_(
            101,
            121,
            111,
            2,
            3,
            core::ptr::null(),
            core::ptr::null(),
            3,
            core::ptr::null(),
            2,
            0.0,
            core::ptr::null_mut(),
            2,
        );
        assert_eq!(crate::colmat_blas_last_error(), 10);
        cblas_zsyrk64_(
            101,
            121,
            111,
            0,
            3,
            core::ptr::null(),
            core::ptr::null(),
            3,
            core::ptr::null(),
            core::ptr::null_mut(),
            1,
        );
        assert_eq!(crate::colmat_blas_last_error(), 0);
        cblas_zher264_(
            101,
            121,
            0,
            core::ptr::null(),
            core::ptr::null(),
            1,
            core::ptr::null(),
            -1,
            core::ptr::null_mut(),
            1,
        );
        assert_eq!(crate::colmat_blas_last_error(), 0);
        let alpha = c64::new(0.75, 0.25);
        let sentinel = c64::new(97.0, 0.0);
        let nan = c64::new(f64::NAN, f64::NAN);
        let mut c = [nan, nan, sentinel, sentinel, nan, sentinel];
        cblas_zher2k64_(
            101,
            121,
            111,
            2,
            0,
            &alpha,
            core::ptr::null(),
            1,
            core::ptr::null(),
            2,
            0.0,
            c.as_mut_ptr(),
            3,
        );
        assert_eq!(
            c,
            [
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                sentinel,
                sentinel,
                c64::new(0.0, 0.0),
                sentinel
            ]
        );
        assert_eq!(crate::colmat_blas_last_error(), 0);
    }
}
