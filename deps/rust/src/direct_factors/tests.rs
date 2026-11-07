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
/// faer multiplies through private-gemm-x86 (no per-call allocation) on x86_64
/// with AVX2 and FMA, and through the allocating `gemm` crate elsewhere.
fn gemm_is_allocation_free() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}
fn spd(n: usize) -> Mat<f64> {
    let x = sample(n, n);
    &x * x.transpose() + Mat::<f64>::identity(n, n)
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
    ] {
        let src = if kind == LLT { spd(n) } else { sample(m, n) };
        let mut a = OwnedMatrix {
            matrix: src.clone(),
        };
        let mut perm = vec![0usize; if kind == LU { m } else { 0 }];
        let mut perm_inv = perm.clone();
        let (mut work, bs) = reserved(1, kind, FACTOR, m, n, 0);
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
        // shapes reach matrix products, which allocate inside the `gemm` crate on
        // targets without private-gemm-x86 (t4a-faer upstream-reports/gemm-001).
        if m * n <= 64 || gemm_is_allocation_free() {
            assert_eq!(allocations, 0, "kind {kind} {m}x{n} factor allocated");
        }
        assert!(allocations < retained);
        let mut expected = OwnedMatrix::new();
        expected.factor_part(&old, 7).unwrap();
        assert_eq!(a.matrix, expected.matrix, "kind {kind} {m}x{n} packed");
        let mut rebuilt = OwnedMatrix::new();
        rebuilt
            .direct_part(&a, kind, 0, true, m, n, 1, &perm, &perm_inv, p, bs)
            .unwrap();
        assert!((&rebuilt.matrix - &src).norm_max() < 1e-9);
        if kind == QR && m < n {
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
            if m * n <= 64 || gemm_is_allocation_free() {
                assert_eq!(allocations, 0, "kind {kind} op {op} solve allocated");
            }
            let sol = x.matrix.get(..n, ..).to_owned();
            if m == n {
                let lhs = if op == SOLVE {
                    src.clone()
                } else {
                    src.transpose().to_owned()
                };
                assert!((&lhs * &sol - &b).norm_max() < 1e-8);
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
        let l = cholesky_lower(a.as_ref(), lower);
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
    for kind in [LU, QR, LLT] {
        let rows = block_size::<f64>(kind, 40, 40);
        let needs = requirement::<f64>(kind, FACTOR, 40, 40, 0, Par::Seq.shape()).unwrap();
        if needs.size_bytes() == 0 {
            continue;
        }
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
