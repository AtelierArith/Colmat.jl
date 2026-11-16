use super::*;

fn check_general<T: Scalar>(
    imag: T,
    tol: f64,
    call: impl Fn(bool, i32, i32, i64, i64, i64, i64, T, *const T, i64, *const T, i64, T, *mut T, i64),
) {
    let value = |i: usize, j: usize| {
        T::real((1 + i + 3 * j) as f64 / 8.0) + imag * T::real((i as f64 - j as f64) / 16.0)
    };
    let alpha = T::real(0.75) + imag * T::real(0.25);
    for (m, n) in [(2usize, 5usize), (5, 2), (1, 4)] {
        for (kl, ku) in [(0usize, 0usize), (1, 2), (m + 1, n + 1)] {
            for order in [101, 102] {
                for t in 111..=114 {
                    for ix in [-2i64, 1, 3] {
                        for iy in [-3i64, 1, 2] {
                            for zero_beta in [false, true] {
                                let ld = kl + ku + 3;
                                let entry = |i: usize, j: usize| {
                                    if i <= j + kl && j <= i + ku {
                                        value(i, j)
                                    } else {
                                        T::z()
                                    }
                                };
                                let mut a =
                                    vec![T::real(f64::NAN); ld * if order == 101 { m } else { n }];
                                for i in 0..m {
                                    for j in 0..n {
                                        if i <= j + kl && j <= i + ku {
                                            let offset = if order == 101 {
                                                i * ld + kl + j - i
                                            } else {
                                                j * ld + ku + i - j
                                            };
                                            a[offset] = entry(i, j);
                                        }
                                    }
                                }
                                let transpose = t == 112 || t == 113;
                                let (nx, ny) = if transpose { (m, n) } else { (n, m) };
                                let vi = |i: usize, len: usize, inc: i64| {
                                    if inc < 0 {
                                        (len - 1 - i) * (-inc) as usize
                                    } else {
                                        i * inc as usize
                                    }
                                };
                                let sentinel = T::real(97.0);
                                let mut x =
                                    vec![sentinel; 1 + (nx - 1) * ix.unsigned_abs() as usize];
                                let mut initial =
                                    vec![sentinel; 1 + (ny - 1) * iy.unsigned_abs() as usize];
                                for i in 0..nx {
                                    x[vi(i, nx, ix)] = value(i, 1);
                                }
                                for i in 0..ny {
                                    initial[vi(i, ny, iy)] = if zero_beta {
                                        T::real(f64::NAN)
                                    } else {
                                        value(i, 2)
                                    };
                                }
                                let beta = if zero_beta {
                                    T::z()
                                } else {
                                    T::real(-0.5) + imag * T::real(0.125)
                                };
                                let mut expected = initial.clone();
                                for i in 0..ny {
                                    let mut sum = T::z();
                                    for j in 0..nx {
                                        let v = if transpose { entry(j, i) } else { entry(i, j) };
                                        sum += if t == 113 || t == 114 { v.cj() } else { v }
                                            * value(j, 1);
                                    }
                                    expected[vi(i, ny, iy)] = alpha * sum
                                        + if zero_beta {
                                            T::z()
                                        } else {
                                            beta * value(i, 2)
                                        };
                                }
                                for wide in [false, true] {
                                    let mut y = initial.clone();
                                    call(
                                        wide,
                                        order,
                                        t,
                                        m as i64,
                                        n as i64,
                                        kl as i64,
                                        ku as i64,
                                        alpha,
                                        a.as_ptr(),
                                        ld as i64,
                                        x.as_ptr(),
                                        ix,
                                        beta,
                                        y.as_mut_ptr(),
                                        iy,
                                    );
                                    for (actual, expected) in y.iter().zip(&expected) {
                                        assert!((*actual-*expected).magnitude()<tol,"GBMV wide={wide} order={order} t={t} m={m} n={n} kl={kl} ku={ku} ix={ix} iy={iy} beta0={zero_beta}");
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
fn general_s() {
    check_general::<f32>(
        0.0f32,
        1e-5,
        |wide, order, t, m, n, kl, ku, alpha, a, ld, x, ix, beta, y, iy| unsafe {
            if wide {
                cblas_sgbmv64_(
                    order, t, m as _, n as _, kl as _, ku as _, alpha, a, ld as _, x, ix as _,
                    beta, y, iy as _,
                );
            } else {
                cblas_sgbmv(
                    order, t, m as _, n as _, kl as _, ku as _, alpha, a, ld as _, x, ix as _,
                    beta, y, iy as _,
                );
            }
        },
    );
}

#[test]
fn general_d() {
    check_general::<f64>(
        0.0f64,
        1e-12,
        |wide, order, t, m, n, kl, ku, alpha, a, ld, x, ix, beta, y, iy| unsafe {
            if wide {
                cblas_dgbmv64_(
                    order, t, m as _, n as _, kl as _, ku as _, alpha, a, ld as _, x, ix as _,
                    beta, y, iy as _,
                );
            } else {
                cblas_dgbmv(
                    order, t, m as _, n as _, kl as _, ku as _, alpha, a, ld as _, x, ix as _,
                    beta, y, iy as _,
                );
            }
        },
    );
}

#[test]
fn general_c() {
    check_general::<c32>(
        c32::new(0.0, 1.0),
        1e-5,
        |wide, order, t, m, n, kl, ku, alpha, a, ld, x, ix, beta, y, iy| unsafe {
            if wide {
                cblas_cgbmv64_(
                    order, t, m as _, n as _, kl as _, ku as _, &alpha, a, ld as _, x, ix as _,
                    &beta, y, iy as _,
                );
            } else {
                cblas_cgbmv(
                    order, t, m as _, n as _, kl as _, ku as _, &alpha, a, ld as _, x, ix as _,
                    &beta, y, iy as _,
                );
            }
        },
    );
}

#[test]
fn general_z() {
    check_general::<c64>(
        c64::new(0.0, 1.0),
        1e-12,
        |wide, order, t, m, n, kl, ku, alpha, a, ld, x, ix, beta, y, iy| unsafe {
            if wide {
                cblas_zgbmv64_(
                    order, t, m as _, n as _, kl as _, ku as _, &alpha, a, ld as _, x, ix as _,
                    &beta, y, iy as _,
                );
            } else {
                cblas_zgbmv(
                    order, t, m as _, n as _, kl as _, ku as _, &alpha, a, ld as _, x, ix as _,
                    &beta, y, iy as _,
                );
            }
        },
    );
}

fn check_symmetric<T: Scalar>(
    h: bool,
    make: impl Fn(f64, f64) -> T,
    tol: f64,
    call: impl Fn(bool, bool, i32, i32, i64, i64, T, *const T, i64, *const T, i64, T, *mut T, i64),
) {
    let value =
        |i: usize, j: usize| make((1 + i + 3 * j) as f64 / 8.0, (i as f64 - j as f64) / 16.0);
    let alpha = make(0.75, 0.25);
    for n in [1usize, 4] {
        for band in [false, true] {
            for k in [0usize, 1, n + 1] {
                for order in [101, 102] {
                    for u in [121, 122] {
                        for ix in [-2i64, 1, 3] {
                            for iy in [-3i64, 1, 2] {
                                for zero_beta in [false, true] {
                                    if !band && k != n + 1 {
                                        continue;
                                    }
                                    let ld = k + 3;
                                    let inside = |i: usize, j: usize| {
                                        ((u == 121 && i <= j) || (u == 122 && i >= j))
                                            && (!band || i.abs_diff(j) <= k)
                                    };
                                    let entry = |i: usize, j: usize| {
                                        if band && i.abs_diff(j) > k {
                                            T::z()
                                        } else if i == j {
                                            T::real(2.0 + i as f64)
                                        } else if inside(i, j) {
                                            value(i, j)
                                        } else {
                                            let v = value(j, i);
                                            if h {
                                                v.cj()
                                            } else {
                                                v
                                            }
                                        }
                                    };
                                    let stored = |i, j| {
                                        if h && i == j {
                                            make(2.0 + i as f64, f64::NAN)
                                        } else {
                                            entry(i, j)
                                        }
                                    };
                                    let mut a = Vec::new();
                                    if band {
                                        a = vec![T::real(f64::NAN); ld * n];
                                        for i in 0..n {
                                            for j in 0..n {
                                                if inside(i, j) {
                                                    let index = if order == 101 {
                                                        i * ld
                                                            + if u == 121 {
                                                                j - i
                                                            } else {
                                                                k + j - i
                                                            }
                                                    } else {
                                                        j * ld
                                                            + if u == 121 {
                                                                k + i - j
                                                            } else {
                                                                i - j
                                                            }
                                                    };
                                                    a[index] = stored(i, j);
                                                }
                                            }
                                        }
                                    } else if order == 101 {
                                        for i in 0..n {
                                            for j in 0..n {
                                                if inside(i, j) {
                                                    a.push(stored(i, j));
                                                }
                                            }
                                        }
                                    } else {
                                        for j in 0..n {
                                            for i in 0..n {
                                                if inside(i, j) {
                                                    a.push(stored(i, j));
                                                }
                                            }
                                        }
                                    }
                                    let vi = |i: usize, inc: i64| {
                                        if inc < 0 {
                                            (n - 1 - i) * (-inc) as usize
                                        } else {
                                            i * inc as usize
                                        }
                                    };
                                    let sentinel = T::real(97.0);
                                    let mut x =
                                        vec![sentinel; 1 + (n - 1) * ix.unsigned_abs() as usize];
                                    let mut initial =
                                        vec![sentinel; 1 + (n - 1) * iy.unsigned_abs() as usize];
                                    for i in 0..n {
                                        x[vi(i, ix)] = value(i, 1);
                                        initial[vi(i, iy)] = if zero_beta {
                                            T::real(f64::NAN)
                                        } else {
                                            value(i, 2)
                                        };
                                    }
                                    let beta = if zero_beta { T::z() } else { make(-0.5, 0.125) };
                                    let mut expected = initial.clone();
                                    for i in 0..n {
                                        let mut sum = T::z();
                                        for j in 0..n {
                                            sum += entry(i, j) * value(j, 1);
                                        }
                                        expected[vi(i, iy)] = alpha * sum
                                            + if zero_beta {
                                                T::z()
                                            } else {
                                                beta * value(i, 2)
                                            };
                                    }
                                    for wide in [false, true] {
                                        let mut y = initial.clone();
                                        call(
                                            wide,
                                            band,
                                            order,
                                            u,
                                            n as i64,
                                            k as i64,
                                            alpha,
                                            a.as_ptr(),
                                            ld as i64,
                                            x.as_ptr(),
                                            ix,
                                            beta,
                                            y.as_mut_ptr(),
                                            iy,
                                        );
                                        for (actual, expected) in y.iter().zip(&expected) {
                                            assert!((*actual-*expected).magnitude()<tol,"symcompact wide={wide} band={band} order={order} u={u} n={n} k={k} ix={ix} iy={iy} beta0={zero_beta}");
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
}

#[test]
fn symmetric_s() {
    check_symmetric::<f32>(
        false,
        |r: f64, _i: f64| r as f32,
        1e-5,
        |wide, band, order, u, n, k, alpha, a, ld, x, ix, beta, y, iy| unsafe {
            match (wide, band) {
                (false, false) => {
                    cblas_sspmv(order, u, n as _, alpha, a, x, ix as _, beta, y, iy as _)
                }
                (false, true) => cblas_ssbmv(
                    order, u, n as _, k as _, alpha, a, ld as _, x, ix as _, beta, y, iy as _,
                ),
                (true, false) => {
                    cblas_sspmv64_(order, u, n as _, alpha, a, x, ix as _, beta, y, iy as _)
                }
                (true, true) => cblas_ssbmv64_(
                    order, u, n as _, k as _, alpha, a, ld as _, x, ix as _, beta, y, iy as _,
                ),
            }
        },
    );
}

#[test]
fn symmetric_d() {
    check_symmetric::<f64>(
        false,
        |r: f64, _i: f64| r,
        1e-12,
        |wide, band, order, u, n, k, alpha, a, ld, x, ix, beta, y, iy| unsafe {
            match (wide, band) {
                (false, false) => {
                    cblas_dspmv(order, u, n as _, alpha, a, x, ix as _, beta, y, iy as _)
                }
                (false, true) => cblas_dsbmv(
                    order, u, n as _, k as _, alpha, a, ld as _, x, ix as _, beta, y, iy as _,
                ),
                (true, false) => {
                    cblas_dspmv64_(order, u, n as _, alpha, a, x, ix as _, beta, y, iy as _)
                }
                (true, true) => cblas_dsbmv64_(
                    order, u, n as _, k as _, alpha, a, ld as _, x, ix as _, beta, y, iy as _,
                ),
            }
        },
    );
}

#[test]
fn symmetric_c() {
    check_symmetric::<c32>(
        true,
        |r: f64, i: f64| c32::new(r as f32, i as f32),
        1e-5,
        |wide, band, order, u, n, k, alpha, a, ld, x, ix, beta, y, iy| unsafe {
            match (wide, band) {
                (false, false) => {
                    cblas_chpmv(order, u, n as _, &alpha, a, x, ix as _, &beta, y, iy as _)
                }
                (false, true) => cblas_chbmv(
                    order, u, n as _, k as _, &alpha, a, ld as _, x, ix as _, &beta, y, iy as _,
                ),
                (true, false) => {
                    cblas_chpmv64_(order, u, n as _, &alpha, a, x, ix as _, &beta, y, iy as _)
                }
                (true, true) => cblas_chbmv64_(
                    order, u, n as _, k as _, &alpha, a, ld as _, x, ix as _, &beta, y, iy as _,
                ),
            }
        },
    );
}

#[test]
fn symmetric_z() {
    check_symmetric::<c64>(
        true,
        |r: f64, i: f64| c64::new(r, i),
        1e-12,
        |wide, band, order, u, n, k, alpha, a, ld, x, ix, beta, y, iy| unsafe {
            match (wide, band) {
                (false, false) => {
                    cblas_zhpmv(order, u, n as _, &alpha, a, x, ix as _, &beta, y, iy as _)
                }
                (false, true) => cblas_zhbmv(
                    order, u, n as _, k as _, &alpha, a, ld as _, x, ix as _, &beta, y, iy as _,
                ),
                (true, false) => {
                    cblas_zhpmv64_(order, u, n as _, &alpha, a, x, ix as _, &beta, y, iy as _)
                }
                (true, true) => cblas_zhbmv64_(
                    order, u, n as _, k as _, &alpha, a, ld as _, x, ix as _, &beta, y, iy as _,
                ),
            }
        },
    );
}

#[test]
fn headers_and_zero_alpha_ignore_unused_storage() {
    unsafe {
        cblas_zgbmv64_(
            101,
            113,
            0,
            3,
            0,
            1,
            core::ptr::null(),
            core::ptr::null(),
            2,
            core::ptr::null(),
            1,
            core::ptr::null(),
            core::ptr::null_mut(),
            1,
        );
        assert_eq!(crate::colmat_blas_last_error(), 0);
        cblas_zhpmv64_(
            101,
            121,
            0,
            core::ptr::null(),
            core::ptr::null(),
            core::ptr::null(),
            i64::MIN,
            core::ptr::null(),
            core::ptr::null_mut(),
            1,
        );
        assert_eq!(crate::colmat_blas_last_error(), 0);
        cblas_zgbmv64_(
            101,
            111,
            2,
            3,
            0,
            1,
            core::ptr::null(),
            core::ptr::null(),
            1,
            core::ptr::null(),
            1,
            core::ptr::null(),
            core::ptr::null_mut(),
            1,
        );
        assert_eq!(crate::colmat_blas_last_error(), 9);
        cblas_zgbmv64_(
            101,
            111,
            2,
            3,
            0,
            1,
            core::ptr::null(),
            core::ptr::null(),
            2,
            core::ptr::null(),
            1,
            core::ptr::null(),
            core::ptr::null_mut(),
            0,
        );
        assert_eq!(crate::colmat_blas_last_error(), 14);
        cblas_zhbmv64_(
            101,
            121,
            2,
            i64::MAX,
            core::ptr::null(),
            core::ptr::null(),
            i64::MAX,
            core::ptr::null(),
            1,
            core::ptr::null(),
            core::ptr::null_mut(),
            1,
        );
        assert_eq!(crate::colmat_blas_last_error(), 7);
        cblas_zhpmv64_(
            101,
            121,
            i64::MAX,
            core::ptr::null(),
            core::ptr::null(),
            core::ptr::null(),
            1,
            core::ptr::null(),
            core::ptr::null_mut(),
            1,
        );
        assert_eq!(crate::colmat_blas_last_error(), 3);
        let zero = c64::new(0.0, 0.0);
        let two = c64::new(2.0, 0.0);
        let sentinel = c64::new(97.0, 0.0);
        let mut y = [
            c64::new(1.0, 1.0),
            sentinel,
            c64::new(2.0, -1.0),
            sentinel,
            c64::new(3.0, 1.0),
        ];
        cblas_zgbmv64_(
            101,
            113,
            2,
            3,
            0,
            1,
            &zero,
            core::ptr::null(),
            2,
            core::ptr::null(),
            1,
            &two,
            y.as_mut_ptr(),
            -2,
        );
        assert_eq!(
            y,
            [
                c64::new(2.0, 2.0),
                sentinel,
                c64::new(4.0, -2.0),
                sentinel,
                c64::new(6.0, 2.0)
            ]
        );
        let mut y = [c64::new(1.0, 1.0), sentinel, c64::new(3.0, -1.0)];
        cblas_zhbmv64_(
            101,
            121,
            2,
            1,
            &zero,
            core::ptr::null(),
            2,
            core::ptr::null(),
            1,
            &two,
            y.as_mut_ptr(),
            -2,
        );
        assert_eq!(y, [c64::new(2.0, 2.0), sentinel, c64::new(6.0, -2.0)]);
    }
}
