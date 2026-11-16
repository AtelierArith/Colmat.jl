use super::*;

fn check<T: Scalar>(
    h: bool,
    make: impl Fn(f64, f64) -> T,
    tol: f64,
    call: impl Fn(bool, bool, i32, i32, i64, T, *const T, i64, *const T, i64, *mut T),
) {
    let value =
        |i: usize, j: usize| make((1 + i + 3 * j) as f64 / 8.0, (i as f64 - j as f64) / 16.0);
    for n in [1usize, 4] {
        for two in [false, true] {
            for order in [101, 102] {
                for u in [121, 122] {
                    for ix in [-2i64, 1, 3] {
                        for iy in [-3i64, 1, 2] {
                            for all_zero in [false, true] {
                                let alpha = if two { make(0.75, 0.25) } else { T::real(0.75) };
                                let vi = |i: usize, inc: i64| {
                                    if inc < 0 {
                                        (n - 1 - i) * (-inc) as usize
                                    } else {
                                        i * inc as usize
                                    }
                                };
                                let sentinel = T::real(97.0);
                                let xv = |i| {
                                    if all_zero || i % 2 == 1 {
                                        T::z()
                                    } else {
                                        value(i, 1)
                                    }
                                };
                                let yv = |i| {
                                    if all_zero || i % 2 == 1 {
                                        T::z()
                                    } else {
                                        value(i, 2)
                                    }
                                };
                                let mut x =
                                    vec![sentinel; 1 + (n - 1) * ix.unsigned_abs() as usize];
                                let mut y =
                                    vec![sentinel; 1 + (n - 1) * iy.unsigned_abs() as usize];
                                for i in 0..n {
                                    x[vi(i, ix)] = xv(i);
                                    y[vi(i, iy)] = yv(i);
                                }
                                // Enumerate logical row/column triangles independently of production
                                // packed indexing. Guard elements follow the complete payload.
                                let mut pairs = Vec::new();
                                if order == 101 {
                                    for i in 0..n {
                                        for j in 0..n {
                                            if (u == 121 && i <= j) || (u == 122 && i >= j) {
                                                pairs.push((i, j));
                                            }
                                        }
                                    }
                                } else {
                                    for j in 0..n {
                                        for i in 0..n {
                                            if (u == 121 && i <= j) || (u == 122 && i >= j) {
                                                pairs.push((i, j));
                                            }
                                        }
                                    }
                                }
                                let mut initial = vec![sentinel; pairs.len() + 3];
                                let mut expected = initial.clone();
                                for (position, &(i, j)) in pairs.iter().enumerate() {
                                    let old = if h && i == j {
                                        T::real(2.0 + i as f64)
                                    } else {
                                        value(i, j)
                                    };
                                    initial[position] = if h && i == j {
                                        make(2.0 + i as f64, f64::NAN)
                                    } else {
                                        old
                                    };
                                    let right = if two { yv(j) } else { xv(j) };
                                    let mut updated =
                                        old + alpha * xv(i) * if h { right.cj() } else { right };
                                    if two {
                                        let right = xv(j);
                                        updated += if h { alpha.cj() } else { alpha }
                                            * yv(i)
                                            * if h { right.cj() } else { right };
                                    }
                                    expected[position] = if h && i == j {
                                        T::real(updated.re())
                                    } else {
                                        updated
                                    };
                                }
                                for wide in [false, true] {
                                    let mut a = initial.clone();
                                    call(
                                        wide,
                                        two,
                                        order,
                                        u,
                                        n as i64,
                                        alpha,
                                        x.as_ptr(),
                                        ix,
                                        y.as_ptr(),
                                        iy,
                                        a.as_mut_ptr(),
                                    );
                                    for (actual, expected) in a.iter().zip(&expected) {
                                        assert!((*actual-*expected).magnitude()<tol,"packedrank wide={wide} two={two} order={order} u={u} n={n} ix={ix} iy={iy} zero={all_zero}");
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
        false,
        |r: f64, _i: f64| r as f32,
        1e-5,
        |wide, two, order, u, n, alpha, x, ix, y, iy, a| unsafe {
            match (wide, two) {
                (false, false) => cblas_sspr(order, u, n as _, alpha, x, ix as _, a),
                (false, true) => cblas_sspr2(order, u, n as _, alpha, x, ix as _, y, iy as _, a),
                (true, false) => cblas_sspr64_(order, u, n as _, alpha, x, ix as _, a),
                (true, true) => cblas_sspr264_(order, u, n as _, alpha, x, ix as _, y, iy as _, a),
            }
        },
    );
}

#[test]
fn d() {
    check::<f64>(
        false,
        |r: f64, _i: f64| r,
        1e-12,
        |wide, two, order, u, n, alpha, x, ix, y, iy, a| unsafe {
            match (wide, two) {
                (false, false) => cblas_dspr(order, u, n as _, alpha, x, ix as _, a),
                (false, true) => cblas_dspr2(order, u, n as _, alpha, x, ix as _, y, iy as _, a),
                (true, false) => cblas_dspr64_(order, u, n as _, alpha, x, ix as _, a),
                (true, true) => cblas_dspr264_(order, u, n as _, alpha, x, ix as _, y, iy as _, a),
            }
        },
    );
}

#[test]
fn c() {
    check::<c32>(
        true,
        |r: f64, i: f64| c32::new(r as f32, i as f32),
        1e-5,
        |wide, two, order, u, n, alpha, x, ix, y, iy, a| unsafe {
            match (wide, two) {
                (false, false) => cblas_chpr(order, u, n as _, alpha.re, x, ix as _, a),
                (false, true) => cblas_chpr2(order, u, n as _, &alpha, x, ix as _, y, iy as _, a),
                (true, false) => cblas_chpr64_(order, u, n as _, alpha.re, x, ix as _, a),
                (true, true) => cblas_chpr264_(order, u, n as _, &alpha, x, ix as _, y, iy as _, a),
            }
        },
    );
}

#[test]
fn z() {
    check::<c64>(
        true,
        |r: f64, i: f64| c64::new(r, i),
        1e-12,
        |wide, two, order, u, n, alpha, x, ix, y, iy, a| unsafe {
            match (wide, two) {
                (false, false) => cblas_zhpr(order, u, n as _, alpha.re, x, ix as _, a),
                (false, true) => cblas_zhpr2(order, u, n as _, &alpha, x, ix as _, y, iy as _, a),
                (true, false) => cblas_zhpr64_(order, u, n as _, alpha.re, x, ix as _, a),
                (true, true) => cblas_zhpr264_(order, u, n as _, &alpha, x, ix as _, y, iy as _, a),
            }
        },
    );
}

#[test]
fn invalid_headers_empty_and_zero_alpha_do_not_access_payloads() {
    unsafe {
        cblas_zhpr264_(
            101,
            121,
            0,
            core::ptr::null(),
            core::ptr::null(),
            i64::MIN,
            core::ptr::null(),
            1,
            core::ptr::null_mut(),
        );
        assert_eq!(crate::colmat_blas_last_error(), 0);
        cblas_zhpr64_(
            101,
            121,
            3,
            0.0,
            core::ptr::null(),
            1,
            core::ptr::null_mut(),
        );
        assert_eq!(crate::colmat_blas_last_error(), 0);
        let zero = c64::new(0.0, 0.0);
        cblas_zhpr264_(
            101,
            121,
            3,
            &zero,
            core::ptr::null(),
            1,
            core::ptr::null(),
            -1,
            core::ptr::null_mut(),
        );
        assert_eq!(crate::colmat_blas_last_error(), 0);
        cblas_zhpr264_(
            101,
            121,
            i64::MAX,
            core::ptr::null(),
            core::ptr::null(),
            1,
            core::ptr::null(),
            1,
            core::ptr::null_mut(),
        );
        assert_eq!(crate::colmat_blas_last_error(), 3);
        cblas_zhpr264_(
            101,
            121,
            2,
            core::ptr::null(),
            core::ptr::null(),
            i64::MIN,
            core::ptr::null(),
            1,
            core::ptr::null_mut(),
        );
        assert_eq!(crate::colmat_blas_last_error(), 6);
        cblas_zhpr264_(
            101,
            121,
            2,
            core::ptr::null(),
            core::ptr::null(),
            1,
            core::ptr::null(),
            0,
            core::ptr::null_mut(),
        );
        assert_eq!(crate::colmat_blas_last_error(), 8);
    }
}
