use super::*;

fn check<T: Scalar>(
    imag: T,
    tolerance: f64,
    call: impl Fn(bool, bool, bool, i32, i32, i32, i32, i64, i64, *const T, i64, *mut T, i64),
) {
    let value = |i: usize, j: usize| {
        T::real((1 + i + 2 * j) as f64 / 16.0) + imag * T::real((i as f64 - j as f64) / 32.0)
    };
    for n in [1usize, 4] {
        for order in [101, 102] {
            for u in [121, 122] {
                for t in 111..=114 {
                    for d in [131, 132] {
                        for band in [false, true] {
                            for k in [0usize, 1, n + 1] {
                                if !band && k != n + 1 {
                                    continue;
                                }
                                let ld = k + 3;
                                let inside = |i: usize, j: usize| {
                                    ((u == 121 && i <= j) || (u == 122 && i >= j))
                                        && (!band || i.abs_diff(j) <= k)
                                };
                                let entry = |i, j| {
                                    if !inside(i, j) {
                                        T::z()
                                    } else if i == j {
                                        if d == 132 {
                                            T::o()
                                        } else {
                                            T::real(2.0 + i as f64 / 4.0) + imag * T::real(0.125)
                                        }
                                    } else {
                                        value(i, j)
                                    }
                                };
                                let stored = |i, j| {
                                    if i == j && d == 132 {
                                        T::real(f64::NAN)
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
                                                let offset = if order == 101 {
                                                    i * ld
                                                        + if u == 121 { j - i } else { k + j - i }
                                                } else {
                                                    j * ld
                                                        + if u == 121 { k + i - j } else { i - j }
                                                };
                                                a[offset] = stored(i, j);
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
                                let op = |i, j| {
                                    let v = if t == 112 || t == 113 {
                                        entry(j, i)
                                    } else {
                                        entry(i, j)
                                    };
                                    if t == 113 || t == 114 {
                                        v.cj()
                                    } else {
                                        v
                                    }
                                };
                                for inc in [-2i64, 1, 3] {
                                    let vi = |i: usize| {
                                        if inc < 0 {
                                            (n - 1 - i) * (-inc) as usize
                                        } else {
                                            i * inc as usize
                                        }
                                    };
                                    let sentinel = T::real(97.0);
                                    let mut original =
                                        vec![sentinel; 1 + (n - 1) * inc.unsigned_abs() as usize];
                                    let mut product = original.clone();
                                    for i in 0..n {
                                        original[vi(i)] = value(i, 2);
                                        let mut sum = T::z();
                                        for j in 0..n {
                                            sum += op(i, j) * value(j, 2);
                                        }
                                        product[vi(i)] = sum;
                                    }
                                    for solve in [false, true] {
                                        for wide in [false, true] {
                                            let (input, expected) = if solve {
                                                (&product, &original)
                                            } else {
                                                (&original, &product)
                                            };
                                            let mut x = input.clone();
                                            call(
                                                wide,
                                                band,
                                                solve,
                                                order,
                                                u,
                                                t,
                                                d,
                                                n as i64,
                                                k as i64,
                                                a.as_ptr(),
                                                ld as i64,
                                                x.as_mut_ptr(),
                                                inc,
                                            );
                                            for (actual, expected) in x.iter().zip(expected) {
                                                assert!((*actual-*expected).magnitude()<tolerance,"wide={wide} band={band} solve={solve} order={order} u={u} t={t} d={d} n={n} k={k} inc={inc}");
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
}

#[test]
fn s() {
    check::<f32>(
        0.0f32,
        1e-5,
        |wide, band, solve, order, u, t, d, n, k, a, ld, x, inc| unsafe {
            match (wide, band, solve) {
                (false, false, false) => cblas_stpmv(order, u, t, d, n as _, a, x, inc as _),
                (false, false, true) => cblas_stpsv(order, u, t, d, n as _, a, x, inc as _),
                (false, true, false) => {
                    cblas_stbmv(order, u, t, d, n as _, k as _, a, ld as _, x, inc as _)
                }
                (false, true, true) => {
                    cblas_stbsv(order, u, t, d, n as _, k as _, a, ld as _, x, inc as _)
                }
                (true, false, false) => cblas_stpmv64_(order, u, t, d, n as _, a, x, inc as _),
                (true, false, true) => cblas_stpsv64_(order, u, t, d, n as _, a, x, inc as _),
                (true, true, false) => {
                    cblas_stbmv64_(order, u, t, d, n as _, k as _, a, ld as _, x, inc as _)
                }
                (true, true, true) => {
                    cblas_stbsv64_(order, u, t, d, n as _, k as _, a, ld as _, x, inc as _)
                }
            }
        },
    );
}

#[test]
fn d() {
    check::<f64>(
        0.0f64,
        1e-12,
        |wide, band, solve, order, u, t, d, n, k, a, ld, x, inc| unsafe {
            match (wide, band, solve) {
                (false, false, false) => cblas_dtpmv(order, u, t, d, n as _, a, x, inc as _),
                (false, false, true) => cblas_dtpsv(order, u, t, d, n as _, a, x, inc as _),
                (false, true, false) => {
                    cblas_dtbmv(order, u, t, d, n as _, k as _, a, ld as _, x, inc as _)
                }
                (false, true, true) => {
                    cblas_dtbsv(order, u, t, d, n as _, k as _, a, ld as _, x, inc as _)
                }
                (true, false, false) => cblas_dtpmv64_(order, u, t, d, n as _, a, x, inc as _),
                (true, false, true) => cblas_dtpsv64_(order, u, t, d, n as _, a, x, inc as _),
                (true, true, false) => {
                    cblas_dtbmv64_(order, u, t, d, n as _, k as _, a, ld as _, x, inc as _)
                }
                (true, true, true) => {
                    cblas_dtbsv64_(order, u, t, d, n as _, k as _, a, ld as _, x, inc as _)
                }
            }
        },
    );
}

#[test]
fn c() {
    check::<c32>(
        c32::new(0.0, 1.0),
        1e-5,
        |wide, band, solve, order, u, t, d, n, k, a, ld, x, inc| unsafe {
            match (wide, band, solve) {
                (false, false, false) => cblas_ctpmv(order, u, t, d, n as _, a, x, inc as _),
                (false, false, true) => cblas_ctpsv(order, u, t, d, n as _, a, x, inc as _),
                (false, true, false) => {
                    cblas_ctbmv(order, u, t, d, n as _, k as _, a, ld as _, x, inc as _)
                }
                (false, true, true) => {
                    cblas_ctbsv(order, u, t, d, n as _, k as _, a, ld as _, x, inc as _)
                }
                (true, false, false) => cblas_ctpmv64_(order, u, t, d, n as _, a, x, inc as _),
                (true, false, true) => cblas_ctpsv64_(order, u, t, d, n as _, a, x, inc as _),
                (true, true, false) => {
                    cblas_ctbmv64_(order, u, t, d, n as _, k as _, a, ld as _, x, inc as _)
                }
                (true, true, true) => {
                    cblas_ctbsv64_(order, u, t, d, n as _, k as _, a, ld as _, x, inc as _)
                }
            }
        },
    );
}

#[test]
fn z() {
    check::<c64>(
        c64::new(0.0, 1.0),
        1e-12,
        |wide, band, solve, order, u, t, d, n, k, a, ld, x, inc| unsafe {
            match (wide, band, solve) {
                (false, false, false) => cblas_ztpmv(order, u, t, d, n as _, a, x, inc as _),
                (false, false, true) => cblas_ztpsv(order, u, t, d, n as _, a, x, inc as _),
                (false, true, false) => {
                    cblas_ztbmv(order, u, t, d, n as _, k as _, a, ld as _, x, inc as _)
                }
                (false, true, true) => {
                    cblas_ztbsv(order, u, t, d, n as _, k as _, a, ld as _, x, inc as _)
                }
                (true, false, false) => cblas_ztpmv64_(order, u, t, d, n as _, a, x, inc as _),
                (true, false, true) => cblas_ztpsv64_(order, u, t, d, n as _, a, x, inc as _),
                (true, true, false) => {
                    cblas_ztbmv64_(order, u, t, d, n as _, k as _, a, ld as _, x, inc as _)
                }
                (true, true, true) => {
                    cblas_ztbsv64_(order, u, t, d, n as _, k as _, a, ld as _, x, inc as _)
                }
            }
        },
    );
}

#[test]
fn compact_headers_and_unit_diagonal_do_not_access_storage() {
    unsafe {
        cblas_ztpmv64_(
            101,
            121,
            113,
            132,
            1,
            core::ptr::null(),
            core::ptr::null_mut(),
            i64::MIN,
        );
        assert_eq!(crate::colmat_blas_last_error(), 0);
        cblas_ztbsv64_(
            101,
            121,
            113,
            132,
            3,
            0,
            core::ptr::null(),
            1,
            core::ptr::null_mut(),
            -1,
        );
        assert_eq!(crate::colmat_blas_last_error(), 0);
        cblas_ztpsv64_(
            101,
            121,
            114,
            131,
            i64::MAX,
            core::ptr::null(),
            core::ptr::null_mut(),
            1,
        );
        assert_eq!(crate::colmat_blas_last_error(), 5);
        cblas_ztbmv64_(
            101,
            121,
            111,
            131,
            2,
            i64::MAX,
            core::ptr::null(),
            i64::MAX,
            core::ptr::null_mut(),
            1,
        );
        assert_eq!(crate::colmat_blas_last_error(), 8);
        cblas_ztpmv64_(
            101,
            121,
            111,
            131,
            2,
            core::ptr::null(),
            core::ptr::null_mut(),
            i64::MIN,
        );
        assert_eq!(crate::colmat_blas_last_error(), 8);
        cblas_ztbmv64_(
            101,
            121,
            111,
            131,
            2,
            1,
            core::ptr::null(),
            2,
            core::ptr::null_mut(),
            0,
        );
        assert_eq!(crate::colmat_blas_last_error(), 10);
    }
}
