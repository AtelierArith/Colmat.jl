use faer::diag::Diag;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::solvers::{SelfAdjointEigen, Svd};
use faer::linalg::{evd, svd};
use faer::{Mat, MatRef, Par, Side};

fn check(a: MatRef<'_, f64>, u: MatRef<'_, f64>, s: &[f64], v: MatRef<'_, f64>) {
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            let actual: f64 = (0..s.len()).map(|k| u[(i, k)] * s[k] * v[(j, k)]).sum();
            assert!((actual - a[(i, j)]).abs() < 1e-10);
        }
    }
}
fn main() {
    let a = Mat::<f64>::from_fn(7, 5, |i, j| {
        if i == j {
            4.0 + i as f64
        } else {
            (i + j) as f64 / 10.0
        }
    });
    let owned = Svd::new_thin(a.as_ref(), Par::Seq).unwrap();
    let values: Vec<f64> = (0..5).map(|k| owned.S().column_vector()[k]).collect();
    check(a.as_ref(), owned.U(), &values, owned.V());
    let mut u = Mat::<f64>::zeros(7, 5);
    let mut v = Mat::<f64>::zeros(5, 5);
    let mut s = Diag::<f64>::zeros(5);
    let req = svd::svd_scratch::<f64>(
        7,
        5,
        svd::ComputeSvdVectors::Thin,
        svd::ComputeSvdVectors::Thin,
        Par::Seq.shape(),
        Default::default(),
    );
    let mut memory = MemBuffer::new(req);
    svd::svd(
        a.as_ref(),
        s.as_mut(),
        Some(u.as_mut()),
        Some(v.as_mut()),
        Par::Seq,
        MemStack::new(&mut memory),
        Default::default(),
    )
    .unwrap();
    let values: Vec<f64> = (0..5).map(|k| s.column_vector()[k]).collect();
    check(a.as_ref(), u.as_ref(), &values, v.as_ref());
    println!("owned SVD and caller-scratch low-level SVD reconstruct 7x5 input");
    let h = Mat::<f64>::from_fn(5, 5, |i, j| {
        if i == j {
            10.0 + i as f64
        } else {
            (i + j) as f64 / 10.0
        }
    });
    let eigen = SelfAdjointEigen::new(h.as_ref(), Side::Lower, Par::Seq).unwrap();
    let values: Vec<f64> = (0..5).map(|k| eigen.S().column_vector()[k]).collect();
    check(h.as_ref(), eigen.U(), &values, eigen.U());
    let mut eu = Mat::<f64>::zeros(5, 5);
    let mut es = Diag::<f64>::zeros(5);
    let ereq = evd::self_adjoint_evd_scratch::<f64>(
        5,
        evd::ComputeEigenvectors::Yes,
        Par::Seq.shape(),
        Default::default(),
    );
    let mut ememory = MemBuffer::new(ereq);
    evd::self_adjoint_evd(
        h.as_ref(),
        es.as_mut(),
        Some(eu.as_mut()),
        Par::Seq,
        MemStack::new(&mut ememory),
        Default::default(),
    )
    .unwrap();
    let values: Vec<f64> = (0..5).map(|k| es.column_vector()[k]).collect();
    check(h.as_ref(), eu.as_ref(), &values, eu.as_ref());
    println!("owned EVD and caller-scratch low-level EVD reconstruct 5x5 input");
    #[cfg(feature = "probe-owned-scratch")]
    {
        let _ = Svd::new_thin(a.as_ref(), Par::Seq, MemStack::new(&mut memory));
        let _ = SelfAdjointEigen::new(
            h.as_ref(),
            Side::Lower,
            Par::Seq,
            MemStack::new(&mut ememory),
        );
    }
}
