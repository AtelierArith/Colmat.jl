use super::*;
use crate::allocation_probe::count;
use crate::OwnedFactor;

fn sample(m: usize, n: usize) -> Mat<f64> {
    Mat::from_fn(m, n, |i, j| {
        let x = ((i * 7 + j * 3) % 11) as f64 - 5.0;
        if i == j {
            x + 12.0
        } else {
            x / 4.0
        }
    })
}
fn reserved(tag: i32, kind: i32, op: i32, m: usize, n: usize, k: usize) -> (OwnedScratch, usize) {
    let mut out = [0usize; 3];
    direct_factor_requirements(kind, op, m, n, k, tag, None, &mut out).unwrap();
    let mut s = OwnedScratch::new();
    s.scratch_reserve(out[0], out[1]).unwrap();
    (s, out[2])
}
#[test]
fn bare_matrix_product_allocation_on_this_target() {
    // Attribution for the residual counts of blocked factorizations: one faer
    // product with no Colmat code involved (t4a-faer upstream-reports/gemm-001).
    let a = sample(70, 35);
    let b = sample(35, 70);
    let mut c = Mat::<f64>::zeros(70, 70);
    let run = |c: &mut Mat<f64>| {
        faer::linalg::matmul::matmul(
            c.as_mut(),
            faer::Accum::Replace,
            a.as_ref(),
            b.as_ref(),
            1.0,
            Par::Seq,
        )
    };
    run(&mut c);
    let allocations = count(|| run(&mut c));
    eprintln!("bare 70x35x70 matmul: {allocations} native allocations");
}
fn spd(n: usize) -> Mat<f64> {
    let x = sample(n, n);
    &x * x.transpose() + Mat::<f64>::identity(n, n)
}
/// Symmetric indefinite with a tiny leading pivot, so Bunch–Kaufman takes
/// 2×2 blocks.
fn indefinite(n: usize) -> Mat<f64> {
    let x = sample(n, n);
    Mat::from_fn(n, n, |i, j| {
        let s = (x[(i, j)] + x[(j, i)]) / 8.0;
        if i == j {
            [1e-8, -3.0, 2.0, 0.0, -1.0, 4.0][i % 6]
        } else {
            s
        }
    })
}
fn perm_len(kind: i32, m: usize, n: usize) -> usize {
    match kind {
        LU => m,
        FULL_LU => m + n,
        LBLT | LBLT_ROOK => n,
        _ => 0,
    }
}

#[test]
fn direct_kernels_match_retained_factors_without_native_allocation() {
    for &(kind, m, n) in &[
        (LU, 6, 6),
        (LU, 70, 70),
        (QR, 6, 6),
        (QR, 9, 5),
        (QR, 4, 7),
        (QR, 80, 64),
        (LLT, 6, 6),
        (LLT, 70, 70),
        (FULL_LU, 6, 6),
        (FULL_LU, 9, 5),
        (FULL_LU, 4, 7),
        (FULL_LU, 70, 70),
        (LDLT, 6, 6),
        (LDLT, 70, 70),
        (LDLT, 140, 140),
        (LBLT, 6, 6),
        (LBLT_ROOK, 6, 6),
        (LBLT, 70, 70),
        (LBLT_ROOK, 70, 70),
        (LBLT, 140, 140),
    ] {
        let src = match kind {
            LLT | LDLT => spd(n),
            LBLT | LBLT_ROOK => indefinite(n),
            _ => sample(m, n),
        };
        let mut a = OwnedMatrix {
            matrix: src.clone(),
        };
        let mut perm = vec![0usize; perm_len(kind, m, n)];
        let mut perm_inv = perm.clone();
        let (mut work, bs) = reserved(1, kind, FACTOR, m, n, 0);
        let bs = if is_lblt(kind) { n } else { bs };
        let mut coeff = vec![0.0f64; bs.max(1) * m.min(n).max(1)];
        let p = coeff.as_mut_ptr().cast::<u8>();
        // Warm up faer's per-thread matmul packing storage on a throwaway copy,
        // so the count below measures the steady state of a reused workspace.
        let mut warm = OwnedMatrix {
            matrix: src.clone(),
        };
        let (mut wp, mut wi) = (perm.clone(), perm_inv.clone());
        let mut wc = coeff.clone();
        let first = count(|| {
            warm.direct_factor(
                kind,
                true,
                m,
                n,
                1,
                &mut wp,
                &mut wi,
                wc.as_mut_ptr().cast(),
                bs,
                &mut work,
                None,
            )
            .unwrap();
        });
        eprintln!("direct kind {kind} {m}x{n}: first call {first} native allocations");
        let mut info = usize::MAX;
        let allocations = count(|| {
            info = a
                .direct_factor(
                    kind,
                    true,
                    m,
                    n,
                    1,
                    &mut perm,
                    &mut perm_inv,
                    p,
                    bs,
                    &mut work,
                    None,
                )
                .unwrap();
        });
        assert_eq!(info, 0);
        // The same faer kernel on the same layout gives the retained packed factor.
        let mut old = OwnedFactor::new();
        let source = OwnedMatrix {
            matrix: src.clone(),
        };
        assert_eq!(old.initialize(&source, kind, true, m, n, 1).unwrap(), 0);
        let retained = count(|| {
            old.initialize(&source, kind, true, m, n, 1).unwrap();
        });
        eprintln!(
            "kind {kind} {m}x{n}: direct {allocations} (first {first}), retained {retained} native allocations"
        );
        // Small shapes run faer's unblocked kernels and must not allocate. Blocked
        // shapes reach faer's internal matrix products, which may allocate their
        // own buffers (see `bare_matrix_product_allocation_on_this_target`).
        if m * n <= 64 {
            assert_eq!(allocations, 0, "kind {kind} {m}x{n} factor allocated");
        }
        assert!(allocations < retained);
        let mut expected = OwnedMatrix::new();
        expected.factor_part(&old, 7, None).unwrap();
        assert_eq!(a.matrix, expected.matrix, "kind {kind} {m}x{n} packed");
        let mut rebuilt = OwnedMatrix::new();
        rebuilt
            .direct_part(&a, kind, 0, true, m, n, 1, &perm, &perm_inv, p, bs, None)
            .unwrap();
        assert!((&rebuilt.matrix - &src).norm_max() < 1e-9);
        if kind == LDLT || is_lblt(kind) {
            let mut d = OwnedMatrix::new();
            d.direct_part(&a, kind, 5, true, m, n, 1, &perm, &perm_inv, p, bs, None)
                .unwrap();
            let mut expected = OwnedMatrix::new();
            expected.factor_part(&old, 5, None).unwrap();
            assert_eq!(d.matrix, expected.matrix, "kind {kind} {m}x{n} D");
        }
        if (kind == QR && m < n) || (kind != QR && m != n) {
            continue;
        }
        for op in [SOLVE, TRANSPOSE, ADJOINT] {
            if op != SOLVE && m != n {
                continue;
            }
            let b = Mat::from_fn(m, 3, |i, j| (i + 2 * j) as f64 - 1.5);
            let mut x = OwnedMatrix { matrix: b.clone() };
            let (mut sw, _) = reserved(1, kind, op, m, n, 3);
            let allocations = count(|| {
                a.direct_solve(
                    &mut x, kind, op, true, m, n, 1, &perm, &perm_inv, p, bs, &mut sw, None,
                )
                .unwrap();
            });
            eprintln!("kind {kind} {m}x{n} op {op}: solve {allocations} native allocations");
            if m * n <= 64 {
                assert_eq!(allocations, 0, "kind {kind} op {op} solve allocated");
            }
            let sol = x.matrix.get(..n, ..).to_owned();
            if m == n {
                let lhs = if op == SOLVE {
                    src.clone()
                } else {
                    src.transpose().to_owned()
                };
                // Relative backward error: the indefinite samples are not
                // well conditioned.
                let residual = (&lhs * &sol - &b).norm_max();
                let scale = lhs.norm_max() * sol.norm_max() * n as f64;
                assert!(residual <= 1e-12 * scale, "kind {kind} op {op}: {residual}");
            } else {
                // Least squares: the residual is orthogonal to range(A).
                assert!((src.transpose() * (&src * &sol - &b)).norm_max() < 1e-8);
            }
        }
    }
}

#[test]
fn retained_construction_allocation_baseline() {
    // The "before" figure: the retained path allocates an independent factor.
    for &kind in &[LU, QR, LLT] {
        let n = 64;
        let a = OwnedMatrix { matrix: spd(n) };
        let mut f = OwnedFactor::new();
        let allocations = count(|| {
            f.initialize(&a, kind, true, n, n, 1).unwrap();
        });
        eprintln!("retained kind {kind} {n}x{n}: {allocations} native allocations");
        assert!(allocations > 0);
    }
}

#[test]
fn bare_triangular_solve_allocation_on_this_target() {
    // Attribution for blocked-size solve counts: faer's triangular solve alone
    // (its recursive blocks call the matrix product of gemm-001).
    let l = sample(140, 140);
    let mut x = Mat::<f64>::from_fn(140, 3, |i, j| (i + j) as f64);
    let mut run = || {
        faer::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
            l.as_ref(),
            x.as_mut(),
            Par::Seq,
        )
    };
    run();
    let allocations = count(run);
    eprintln!("bare 140x140 unit-lower solve, 3 RHS: {allocations} native allocations");
}

/// Complex Hermitian LDLᴴ and Bunch–Kaufman in both storage triangles: the
/// stored factor reconstructs A and solves A, Aᵀ and Aᴴ; D matches.
#[test]
fn complex_self_adjoint_kinds_in_both_triangles() {
    use faer::c64;
    let n = 7;
    let x = Mat::<c64>::from_fn(n, n, |i, j| {
        c64::new((i + 2 * j) as f64 % 5.0, i as f64 - j as f64)
    });
    let spd = &x * x.adjoint() + Mat::<c64>::identity(n, n);
    let indefinite = Mat::<c64>::from_fn(n, n, |i, j| {
        if i == j {
            c64::new([1e-9, -2.0, 3.0, 0.0, -1.0, 2.0, 0.5][i], 0.0)
        } else if i > j {
            c64::new(0.3 * (i + j) as f64, 0.2 * (i as f64 - j as f64))
        } else {
            c64::new(0.3 * (i + j) as f64, -0.2 * (j as f64 - i as f64))
        }
    });
    for (kind, h) in [(LDLT, &spd), (LBLT, &indefinite), (LBLT_ROOK, &indefinite)] {
        let mut reference = None;
        for lower in [true, false] {
            // Garbage in the unread triangle must be ignored.
            let mut a = Mat::from_fn(n, n, |i, j| {
                if (lower && i < j) || (!lower && i > j) {
                    c64::new(f64::NAN, 9.0)
                } else {
                    h[(i, j)]
                }
            });
            let (mut perm, mut inv) = (
                vec![0usize; perm_len(kind, n, n)],
                vec![0usize; perm_len(kind, n, n)],
            );
            let mut off = vec![c64::new(7.0, 7.0); n];
            let (mut work, _) = reserved(3, kind, FACTOR, n, n, 0);
            let info = factor(
                a.as_mut(),
                kind,
                lower,
                &mut perm,
                &mut inv,
                off.as_mut_ptr().cast(),
                if is_lblt(kind) { n } else { 0 },
                &mut work,
                Par::Seq,
            )
            .unwrap();
            assert_eq!(info, 0);
            let rows = if is_lblt(kind) { n } else { 0 };
            let get = |code| {
                part(
                    a.as_ref(),
                    kind,
                    code,
                    lower,
                    &perm,
                    &inv,
                    off.as_ptr().cast(),
                    rows,
                    Par::Seq,
                )
                .unwrap()
            };
            assert!((get(0) - h).norm_max() < 1e-10, "kind {kind} lower {lower}");
            let d = get(5);
            match &reference {
                None => reference = Some((get(7), d)),
                Some((packed, d0)) => {
                    assert!((&get(7) - packed).norm_max() < 1e-12);
                    assert!((&d - d0).norm_max() < 1e-12);
                }
            }
            let other = if lower {
                a.as_ref().get(0..1, 1..).norm_max()
            } else {
                a.as_ref().get(1.., 0..1).norm_max()
            };
            assert_eq!(other, 0.0);
            let b = Mat::<c64>::from_fn(n, 2, |i, j| c64::new(i as f64 - 1.0, j as f64 + 0.5));
            for op in [SOLVE, TRANSPOSE, ADJOINT] {
                let lhs = match op {
                    SOLVE | ADJOINT => h.clone(),
                    _ => h.transpose().to_owned(),
                };
                let mut rhs = b.clone();
                let (mut sw, _) = reserved(3, kind, op, n, n, 2);
                solve(
                    a.as_ref(),
                    rhs.as_mut(),
                    kind,
                    op,
                    lower,
                    &perm,
                    &inv,
                    off.as_ptr().cast(),
                    rows,
                    &mut sw,
                    Par::Seq,
                )
                .unwrap();
                assert!(
                    (&lhs * &rhs - &b).norm_max() < 1e-8,
                    "kind {kind} op {op} lower {lower}"
                );
            }
            // det through the stored factor matches faer's dense determinant.
            let ld = get(12);
            let det = h.as_ref().determinant(Par::Seq);
            assert!((ld[(0, 0)].re - det.norm().ln()).abs() < 1e-9);
            assert!((ld[(1, 0)] * c64::new(det.norm(), 0.0) - det).norm() < 1e-8 * det.norm());
        }
    }
}

#[test]
fn upper_complex_cholesky_mirrors_the_factor() {
    let n = 5;
    let x = Mat::<faer::c64>::from_fn(n, n, |i, j| {
        faer::c64::new((i + j) as f64, i as f64 - j as f64)
    });
    let h = &x * x.adjoint() + Mat::<faer::c64>::identity(n, n);
    for lower in [true, false] {
        let mut a = h.clone();
        let (mut work, _) = reserved(3, LLT, FACTOR, n, n, 0);
        let info = factor(
            a.as_mut(),
            LLT,
            lower,
            &mut [],
            &mut [],
            std::ptr::null_mut(),
            0,
            &mut work,
            Par::Seq,
        )
        .unwrap();
        assert_eq!(info, 0);
        let l = packed_lower(a.as_ref(), lower);
        assert!((&l * l.adjoint() - &h).norm_max() < 1e-10);
        let other = if lower {
            a.as_ref().get(0..1, 1..).norm_max()
        } else {
            a.as_ref().get(1.., 0..1).norm_max()
        };
        assert_eq!(other, 0.0);
    }
}

#[test]
fn validation_failures_do_not_write_the_matrix() {
    let mut a = OwnedMatrix {
        matrix: sample(40, 40),
    };
    let before = a.matrix.clone();
    let mut empty = OwnedScratch::new();
    empty.scratch_reserve(0, 1).unwrap();
    let mut perm = vec![0usize; 40];
    let mut inv = vec![0usize; 40];
    let mut coeff = vec![0.0f64; 40 * 40];
    for kind in [LU, FULL_LU, QR, LLT, LDLT, LBLT, LBLT_ROOK] {
        let rows = if is_lblt(kind) {
            40
        } else {
            block_size::<f64>(kind, 40, 40)
        };
        let needs = requirement::<f64>(kind, FACTOR, 40, 40, 0, Par::Seq.shape()).unwrap();
        if needs.size_bytes() == 0 {
            continue;
        }
        let len = perm_len(kind, 40, 40);
        let (mut perm, mut inv) = (vec![0usize; len], vec![0usize; len]);
        // Only the scratch capacity is insufficient here.
        assert!(check_aux::<f64>(kind, 40, 40, len, len, coeff.as_ptr().cast(), rows).is_ok());
        assert!(a
            .direct_factor(
                kind,
                true,
                40,
                40,
                1,
                &mut perm,
                &mut inv,
                coeff.as_mut_ptr().cast(),
                rows,
                &mut empty,
                None,
            )
            .is_err());
        assert_eq!(a.matrix, before);
    }
    // Non-finite input is rejected before writing the matrix.
    let mut bad = OwnedMatrix {
        matrix: sample(4, 4),
    };
    bad.matrix[(2, 1)] = f64::NAN;
    let snapshot = bad.matrix.clone();
    let (mut work, _) = reserved(1, LU, FACTOR, 4, 4, 0);
    let (mut p4, mut i4) = (vec![0usize; 4], vec![0usize; 4]);
    assert!(bad
        .direct_factor(
            LU,
            true,
            4,
            4,
            1,
            &mut p4,
            &mut i4,
            std::ptr::null_mut(),
            0,
            &mut work,
            None
        )
        .is_err());
    assert!(bad.matrix[(2, 1)].is_nan());
    assert_eq!(bad.matrix[(0, 0)], snapshot[(0, 0)]);
    // A corrupted permutation is rejected instead of indexing out of bounds.
    let mut good = OwnedMatrix {
        matrix: sample(4, 4),
    };
    good.direct_factor(
        LU,
        true,
        4,
        4,
        1,
        &mut p4,
        &mut i4,
        std::ptr::null_mut(),
        0,
        &mut work,
        None,
    )
    .unwrap();
    let mut rhs = OwnedMatrix {
        matrix: Mat::zeros(4, 1),
    };
    let mut broken = p4.clone();
    broken[0] = 9;
    let (mut sw, _) = reserved(1, LU, SOLVE, 4, 4, 1);
    assert!(good
        .direct_solve(
            &mut rhs,
            LU,
            SOLVE,
            true,
            4,
            4,
            1,
            &broken,
            &i4,
            std::ptr::null(),
            0,
            &mut sw,
            None,
        )
        .is_err());
    // Mismatched metadata is rejected.
    assert!(good
        .direct_factor(
            LU,
            true,
            4,
            3,
            1,
            &mut p4,
            &mut i4,
            std::ptr::null_mut(),
            0,
            &mut work,
            None,
        )
        .is_err());
}
