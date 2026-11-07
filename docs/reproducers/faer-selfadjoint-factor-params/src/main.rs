//! Self-adjoint factorization parameters in t4a-faer (see README.md).
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::lblt::factor::{self as lblt, LbltParams, PivotingStrategy};
use faer::linalg::cholesky::ldlt::factor::{self as ldlt, LdltRegularization};
use faer::linalg::cholesky::llt::factor::{self as llt, LltRegularization};
use faer::linalg::solvers::DenseSolveCore;
use faer::{c64, mat, Auto, Mat, Par, Side};

/// 1. LLᴴ dynamic regularization stores the wrong diagonal for a replaced pivot.
fn llt_regularized_diagonal(n: usize) -> f64 {
    // Diagonally dominant except for a negative first pivot, which the
    // regularization replaces by delta. The regularized factor must satisfy
    // L Lᴴ = A + E with E diagonal: here L[0,0] = sqrt(delta) = 1.
    let a = Mat::<f64>::from_fn(n, n, |i, j| {
        if i == j {
            if i == 0 { -3.0 } else { 4.0 }
        } else {
            0.5
        }
    });
    let mut l = a.clone();
    let mut mem = MemBuffer::new(llt::cholesky_in_place_scratch::<f64>(
        n,
        Par::Seq.shape(),
        Default::default(),
    ));
    // A + E = 3.5 I + 0.5 11ᵀ - 3 e0 e0ᵀ is positive definite, so the
    // regularized factorization exists.
    let info = match llt::cholesky_in_place(
        l.as_mut(),
        LltRegularization {
            dynamic_regularization_delta: 1.0,
            dynamic_regularization_epsilon: 1e-8,
        },
        Par::Seq,
        MemStack::new(&mut mem),
        Default::default(),
    ) {
        Ok(info) => info,
        Err(e) => {
            println!("llt n={n}: {e:?} (expected success)");
            return f64::INFINITY;
        }
    };
    for j in 0..n {
        for i in 0..j {
            l[(i, j)] = 0.0;
        }
    }
    let e = &l * l.adjoint() - &a;
    let off = Mat::from_fn(n, n, |i, j| if i == j { 0.0 } else { e[(i, j)] });
    println!(
        "llt n={n}: count={} L[0,0]={} expected sqrt(delta)=1, max|offdiag(L Lᴴ - A)|={:.3e}",
        info.dynamic_regularization_count,
        l[(0, 0)],
        off.norm_max()
    );
    off.norm_max() + (l[(0, 0)] - 1.0f64).abs()
}

/// 2. LDLᴴ dynamic regularization does not count pivots it replaces without
/// signs (or corrects towards a negative expected sign).
fn ldlt_count(signs: Option<&[i8]>) -> (usize, f64) {
    let a = mat![[0.0, 1.0], [1.0, 0.0f64]];
    let mut l = a.clone();
    let mut mem = MemBuffer::new(ldlt::cholesky_in_place_scratch::<f64>(
        2,
        Par::Seq.shape(),
        Default::default(),
    ));
    let info = ldlt::cholesky_in_place(
        l.as_mut(),
        LdltRegularization {
            dynamic_regularization_signs: signs,
            dynamic_regularization_delta: 1.0,
            dynamic_regularization_epsilon: 1e-8,
        },
        Par::Seq,
        MemStack::new(&mut mem),
        Default::default(),
    )
    .unwrap();
    (info.dynamic_regularization_count, l[(0, 0)])
}

/// 3. `solvers::Lblt::new` takes no `LbltParams`: only the low-level kernel
/// reaches the other pivoting strategies.
fn lblt_strategies() {
    let a = Mat::from_fn(6, 6, |i, j| {
        if i == j {
            [1e-8, -3.0, 2.0, 0.0, -1.0, 4.0][i]
        } else {
            1.0 / (1.0 + i as f64 + j as f64)
        }
    });
    let high = a.lblt(Side::Lower, Par::Seq);
    println!("lblt solvers::Lblt::new perm: {:?}", high.P().arrays().0);
    for strategy in [
        PivotingStrategy::Partial,
        PivotingStrategy::PartialDiag,
        PivotingStrategy::Rook,
        PivotingStrategy::RookDiag,
        PivotingStrategy::Full,
    ] {
        let mut params = <LbltParams as Auto<f64>>::auto();
        params.pivoting = strategy;
        let mut l = a.clone();
        let mut sub = faer::diag::Diag::<f64>::zeros(6);
        let (mut fwd, mut bwd) = (vec![0usize; 6], vec![0usize; 6]);
        let mut mem = MemBuffer::new(lblt::cholesky_in_place_scratch::<usize, f64>(
            6,
            Par::Seq.shape(),
            params.into(),
        ));
        lblt::cholesky_in_place(
            l.as_mut(),
            sub.as_mut(),
            &mut fwd,
            &mut bwd,
            Par::Seq,
            MemStack::new(&mut mem),
            params.into(),
        );
        println!("lblt low-level {strategy:?} perm: {fwd:?}");
    }
}

/// 4. LBLᴴ is self-adjoint only: a complex symmetric (non-Hermitian) input is
/// read as the Hermitian matrix of its lower triangle (LAPACK `zsytrf` has no
/// counterpart).
fn complex_symmetric() -> f64 {
    let a = mat![
        [c64::new(1.0, 1.0), c64::new(2.0, -1.0)],
        [c64::new(2.0, -1.0), c64::new(3.0, -2.0)],
    ];
    let f = a.lblt(Side::Lower, Par::Seq);
    let r = f.reconstruct(Par::Seq);
    let err = (&r - &a).norm_max();
    println!("complex symmetric LBL: max|reconstruct - A| = {err:.3e} (reconstruct = {r:?})");
    err
}

fn main() {
    let small = llt_regularized_diagonal(2);
    let unblocked = llt_regularized_diagonal(64);
    let blocked = llt_regularized_diagonal(200);
    let (none, d0) = ldlt_count(None);
    let (plus, _) = ldlt_count(Some(&[1, -1]));
    println!("ldlt no signs: count={none} D[0]={d0} (pivot 0 was replaced by delta)");
    println!("ldlt signs [+1,-1]: count={plus}");
    lblt_strategies();
    let sym = complex_symmetric();
    let defects = [small > 1e-12, unblocked > 1e-12, blocked > 1e-12, none == 0, sym > 1e-12];
    println!("defects reproduced: {defects:?}");
    std::process::exit(if defects.iter().all(|&d| d) { 1 } else { 0 });
}
