//! Lazy square orthogonal/unitary sparse-QR factor `Q` as a native operator.
//! Applies saved Householder reflectors on caller scratch; never forms dense Q.
use super::{Base, Data, Operator, OwnedLinearOperator};
use crate::guarded;
use crate::sparse_factors::{QrComponents, SparseFactor};
use crate::sparse_qr_owned::projection::Projection;
use faer::dyn_stack::{MemStack, StackReq};
use faer::matrix_free::{BiLinOp, LinOp};
use faer::traits::ComplexField;
use faer::{MatMut, MatRef, Par, ParShape};
use rustcall_julia_macros::julia;

#[derive(Clone)]
pub(super) struct SparseQrQ<T> {
    projection: Projection<T>,
}
impl<T> std::fmt::Debug for SparseQrQ<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SparseQrQ")
    }
}
impl<T: ComplexField> SparseQrQ<T> {
    pub(super) fn new(projection: Projection<T>) -> Self {
        Self { projection }
    }
    pub(super) fn map_values<U: ComplexField>(&self, cast: impl Fn(&T) -> U) -> SparseQrQ<U> {
        SparseQrQ {
            projection: self.projection.map_values(cast),
        }
    }
    fn scratch(&self, k: usize) -> StackReq {
        if self.projection.dimensions().0 == 0 || k == 0 {
            return StackReq::EMPTY;
        }
        self.projection.scratch(k)
    }
    // ADJ applies the reflectors in factorization order; C conjugates them.
    fn run<const ADJ: bool, const C: bool>(
        &self,
        out: MatMut<'_, T>,
        b: MatRef<'_, T>,
        par: Par<'_>,
        stack: &mut MemStack,
    ) {
        if self.projection.dimensions().0 == 0 || b.ncols() == 0 {
            return;
        }
        self.projection.apply::<ADJ, C, true>(out, b, par, stack);
    }
}
impl<T: ComplexField> LinOp<T> for SparseQrQ<T> {
    fn nrows(&self) -> usize {
        self.projection.dimensions().0
    }
    fn ncols(&self) -> usize {
        self.projection.dimensions().0
    }
    fn apply_scratch(&self, k: usize, _p: ParShape) -> StackReq {
        self.scratch(k)
    }
    fn apply(&self, o: MatMut<'_, T>, b: MatRef<'_, T>, p: Par<'_>, s: &mut MemStack) {
        self.run::<false, false>(o, b, p, s)
    }
    fn conj_apply(&self, o: MatMut<'_, T>, b: MatRef<'_, T>, p: Par<'_>, s: &mut MemStack) {
        self.run::<false, true>(o, b, p, s)
    }
}
impl<T: ComplexField> BiLinOp<T> for SparseQrQ<T> {
    fn transpose_apply_scratch(&self, k: usize, _p: ParShape) -> StackReq {
        self.scratch(k)
    }
    fn transpose_apply(&self, o: MatMut<'_, T>, b: MatRef<'_, T>, p: Par<'_>, s: &mut MemStack) {
        self.run::<true, true>(o, b, p, s)
    }
    fn adjoint_apply(&self, o: MatMut<'_, T>, b: MatRef<'_, T>, p: Par<'_>, s: &mut MemStack) {
        self.run::<true, false>(o, b, p, s)
    }
}

fn wrap<T: ComplexField>(projection: Projection<T>) -> Operator<T> {
    Operator::new(Base::SparseQrQ(Box::new(SparseQrQ::new(projection))))
}
pub(super) fn sparse_qr_q_operator(factor: &SparseFactor) -> Result<Data, String> {
    Ok(match factor.simplicial_qr_components()? {
        QrComponents::F32(f) => Data::F32(wrap(Projection::from_simplicial(f))),
        QrComponents::F64(f) => Data::F64(wrap(Projection::from_simplicial(f))),
        QrComponents::C32(f) => Data::C32(wrap(Projection::from_simplicial(f))),
        QrComponents::C64(f) => Data::C64(wrap(Projection::from_simplicial(f))),
        QrComponents::OwnedF32(f) => Data::F32(wrap(Projection::new(f))),
        QrComponents::OwnedF64(f) => Data::F64(wrap(Projection::new(f))),
        QrComponents::OwnedC32(f) => Data::C32(wrap(Projection::new(f))),
        QrComponents::OwnedC64(f) => Data::C64(wrap(Projection::new(f))),
        // Wide factors store a square QR of the leading pivot columns; its Q
        // is the full orthogonal/unitary factor of the wide input.
        QrComponents::WideF32(f) => Data::F32(wrap(Projection::new(f.head_ref().clone()))),
        QrComponents::WideF64(f) => Data::F64(wrap(Projection::new(f.head_ref().clone()))),
        QrComponents::WideC32(f) => Data::C32(wrap(Projection::new(f.head_ref().clone()))),
        QrComponents::WideC64(f) => Data::C64(wrap(Projection::new(f.head_ref().clone()))),
    })
}

#[julia]
impl OwnedLinearOperator {
    /// Lazy square `Q` of saved sparse QR factors (every algorithm and scalar).
    #[julia]
    pub fn operator_sparse_qr_q(&mut self, factor: &SparseFactor) -> Result<(), String> {
        guarded(|| {
            self.data = Some(sparse_qr_q_operator(factor)?);
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::dyn_stack::MemBuffer;
    use faer::sparse::linalg::{
        qr::{factorize_symbolic_qr, QrSymbolicParams},
        SupernodalThreshold,
    };
    use faer::sparse::{SparseColMat, Triplet};
    use faer::traits::math_utils::{conj, copy, from_f64, one, zero};
    use faer::Mat;
    use std::sync::Arc;

    fn apply_all<T: ComplexField>(q: &SparseQrQ<T>, b: MatRef<'_, T>) -> [Mat<T>; 4] {
        let m = LinOp::<T>::nrows(q);
        let mut mem = MemBuffer::new(q.scratch(b.ncols()));
        let mut out = [(); 4].map(|_| Mat::<T>::zeros(m, b.ncols()));
        let stack = MemStack::new(&mut mem);
        q.apply(out[0].as_mut(), b, Par::Seq, stack);
        q.conj_apply(out[1].as_mut(), b, Par::Seq, stack);
        q.transpose_apply(out[2].as_mut(), b, Par::Seq, stack);
        q.adjoint_apply(out[3].as_mut(), b, Par::Seq, stack);
        out
    }
    fn assert_matches<T: ComplexField>(q: &SparseQrQ<T>, full: &Mat<T>, tol: f64) {
        let m = full.nrows();
        let eye = Mat::from_fn(m, m, |i, j| if i == j { one::<T>() } else { zero::<T>() });
        let [a, c, t, h] = apply_all(q, eye.as_ref());
        let tol = from_f64::<T::Real>(tol);
        let conj_full = Mat::from_fn(m, m, |i, j| conj(&full[(i, j)]));
        assert!((&a - full).norm_max() <= tol);
        assert!((&c - &conj_full).norm_max() <= tol);
        assert!((&t - full.transpose()).norm_max() <= tol);
        assert!((&h - full.adjoint()).norm_max() <= tol);
    }
    fn check<T: ComplexField>(tol: f64, value: fn(f64, f64) -> T) {
        for (m, n) in [(0, 0), (5, 0), (5, 3), (5, 5), (9, 4), (40, 20)] {
            for empty in [false, true] {
                // `empty` leaves structurally empty columns (rank-deficient
                // reflectors) so the Q completion order is exercised.
                let entries: Vec<_> = (0..n)
                    .flat_map(|j| (0..m).map(move |i| (i, j)))
                    .filter(|&(i, j)| {
                        (!empty || j % 2 == 0) && (i == m - n + j || i == j || (i + j) % 4 == 1)
                    })
                    .map(|(i, j)| Triplet {
                        row: i,
                        col: j,
                        val: value(if i == m - n + j { 4.0 + j as f64 } else { 0.3 }, 0.2),
                    })
                    .collect();
                let a = SparseColMat::<usize, T>::try_new_from_triplets(m, n, &entries).unwrap();
                for threshold in [
                    SupernodalThreshold::FORCE_SIMPLICIAL,
                    SupernodalThreshold::FORCE_SUPERNODAL,
                ] {
                    let symbolic = Arc::new(
                        factorize_symbolic_qr(
                            a.symbolic(),
                            QrSymbolicParams {
                                supernodal_flop_ratio_threshold: threshold,
                                ..Default::default()
                            },
                        )
                        .unwrap(),
                    );
                    let f = crate::sparse_qr_owned::Factors::new(symbolic, a.as_ref()).unwrap();
                    let full = f.q_full().unwrap();
                    assert_matches(&SparseQrQ::new(Projection::new(f)), &full, tol);
                }
                let identity = faer::perm::Perm::new_checked(
                    (0..n).collect::<Vec<_>>().into_boxed_slice(),
                    (0..n).collect::<Vec<_>>().into_boxed_slice(),
                    n,
                );
                let plan = crate::sparse_qr_components::Plan::new(a.symbolic(), identity.as_ref())
                    .unwrap();
                let f = crate::sparse_qr_components::Factors::new(&plan, a.as_ref()).unwrap();
                let full = f.q_full();
                let q = SparseQrQ::new(Projection::from_simplicial(f));
                assert_matches(&q, &full, tol);
                // Value conversion keeps the native reflector layout and Q.
                assert_matches(&q.map_values(|x| copy(x)), &full, tol);
            }
        }
    }
    #[test]
    fn lazy_q_applies_without_allocation() {
        let (m, n) = (60, 30);
        let entries: Vec<_> = (0..n)
            .flat_map(|j| (0..m).map(move |i| (i, j)))
            .filter(|&(i, j)| i == j + 3 || (i * 5 + j) % 7 == 0)
            .map(|(i, j)| Triplet {
                row: i,
                col: j,
                val: faer::c64::new(1.0 + (i % 3) as f64, 0.5),
            })
            .collect();
        let a = SparseColMat::<usize, faer::c64>::try_new_from_triplets(m, n, &entries).unwrap();
        for threshold in [
            SupernodalThreshold::FORCE_SIMPLICIAL,
            SupernodalThreshold::FORCE_SUPERNODAL,
        ] {
            let symbolic = Arc::new(
                factorize_symbolic_qr(
                    a.symbolic(),
                    QrSymbolicParams {
                        supernodal_flop_ratio_threshold: threshold,
                        ..Default::default()
                    },
                )
                .unwrap(),
            );
            let f = crate::sparse_qr_owned::Factors::new(symbolic, a.as_ref()).unwrap();
            let q = SparseQrQ::new(Projection::new(f));
            let b = Mat::from_fn(m, 3, |i, j| faer::c64::new(i as f64, j as f64));
            let mut out = Mat::zeros(m, 3);
            let mut mem = MemBuffer::new(q.scratch(3));
            let count = crate::allocation_probe::count(|| {
                let stack = MemStack::new(&mut mem);
                q.apply(out.as_mut(), b.as_ref(), Par::Seq, stack);
                q.adjoint_apply(out.as_mut(), b.as_ref(), Par::Seq, stack);
                q.transpose_apply(out.as_mut(), b.as_ref(), Par::Seq, stack);
                q.conj_apply(out.as_mut(), b.as_ref(), Par::Seq, stack);
            });
            assert_eq!(count, 0);
        }
    }
    #[test]
    fn wide_q_operator_copies_the_head_once() {
        use crate::allocation_probe::measure;
        let (m, n) = (24, 60);
        let entries: Vec<_> = (0..n)
            .flat_map(|j| (0..m).map(move |i| (i, j)))
            .filter(|&(i, j)| i == j % m || (i * 3 + j) % 7 == 0)
            .map(|(i, j)| Triplet {
                row: i,
                col: j,
                val: 1.0 + ((i + 2 * j) % 5) as f64,
            })
            .collect();
        let a = SparseColMat::<usize, f64>::try_new_from_triplets(m, n, &entries).unwrap();
        let order: Vec<_> = (0..n).collect();
        let columns = faer::perm::Perm::new_checked(
            order.clone().into_boxed_slice(),
            order.into_boxed_slice(),
            n,
        );
        let plan =
            crate::sparse_qr_wide::Plan::new(a.symbolic(), columns, Default::default()).unwrap();
        let f = crate::sparse_qr_wide::Factors::new(&plan, a.as_ref()).unwrap();
        let operator = measure(|| drop(wrap(Projection::new(f.head_ref().clone()))));
        let head = measure(|| drop(f.head_ref().clone()));
        let copy = f.head_ref().clone();
        let layout = measure(|| drop(wrap(Projection::new(copy))));
        let whole = measure(|| drop(f.clone()));
        eprintln!("wide Qoperator={operator:?} head={head:?} layout={layout:?} factor={whole:?}");
        // One head snapshot plus the prepared reflector layout; neither the
        // whole wide factor (sparse R, permutation) nor a second head copy.
        assert_eq!(operator.0, head.0 + layout.0);
        // Measured (24×60 f64): operator 10 allocations / largest 2208 B =
        // head 2 / 2208 + layout 8 / 384; cloning the wide factor: 7 / 5216.
        assert!(operator.1 == head.1 && operator.1 < whole.1);
    }
    #[test]
    fn lazy_q_matches_full_q_f32() {
        check::<f32>(1e-4, |r, _| r as f32);
    }
    #[test]
    fn lazy_q_matches_full_q_f64() {
        check::<f64>(1e-12, |r, _| r);
    }
    #[test]
    fn lazy_q_matches_full_q_c32() {
        check::<faer::c32>(1e-4, |r, i| faer::c32::new(r as f32, i as f32));
    }
    #[test]
    fn lazy_q_matches_full_q_c64() {
        check::<faer::c64>(1e-12, faer::c64::new);
    }
}
