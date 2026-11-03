//! Owned native LinOp/BiLinOp/Precond adapters and conjugate gradient.
use crate::collections::{VectorData, VectorOwner, VectorStore};
use crate::sparse::{SparseData, SparseOwner, Storage};
use crate::typed::{MatrixData, TypedMatrix};
use crate::{guarded, OwnedMatrix};
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::mat::{AsMatMut, AsMatRef};
use faer::matrix_free::{
    conjugate_gradient::{self as cg, CgError},
    BiLinOp, BiPrecond, IdentityPrecond, InitialGuessStatus, LinOp, Precond,
};
use faer::traits::{math_utils::*, ComplexField};
use faer::{c32, c64, diag::Diag, Mat, MatMut, MatRef, Par, ParShape};
use rustcall_julia_macros::julia;
#[derive(Clone)]
enum Base<T> {
    Dense(Mat<T>),
    Sparse(Storage<T>),
    Diagonal(Diag<T>),
    Identity(IdentityPrecond),
    Product(Box<OperatorProduct<T>>),
    InverseLu(Box<InverseLu<T>>),
}
#[derive(Clone)]
struct Operator<T> {
    base: Base<T>,
    transpose: bool,
    conjugate: bool,
}
macro_rules! dispatch {
    ($op:expr,$a:ident=>$expr:expr) => {
        match &$op.base {
            Base::Dense($a) => $expr,
            Base::Sparse(Storage::Csc($a)) => $expr,
            Base::Sparse(Storage::Csr($a)) => $expr,
            Base::Diagonal($a) => $expr,
            Base::Identity($a) => $expr,
            Base::InverseLu($a) => {
                let $a = $a.as_ref();
                $expr
            }
            Base::Product($a) => {
                let $a = $a.as_ref();
                $expr
            }
        }
    };
}
impl<T: ComplexField> Operator<T> {
    fn new(base: Base<T>) -> Self {
        Self {
            base,
            transpose: false,
            conjugate: false,
        }
    }
    fn shape(&self) -> (usize, usize) {
        let shape = dispatch!(self,a=>(LinOp::<T>::nrows(a),LinOp::<T>::ncols(a)));
        if self.transpose {
            (shape.1, shape.0)
        } else {
            shape
        }
    }
    fn run(
        &self,
        out: MatMut<'_, T>,
        rhs: MatRef<'_, T>,
        par: Par,
        stack: &mut MemStack,
        transpose: bool,
        conjugate: bool,
    ) {
        let t = self.transpose ^ transpose;
        let c = self.conjugate ^ conjugate;
        dispatch!(self,a=>match (t,c) {
            (false,false)=>LinOp::<T>::apply(a,out,rhs,par,stack),(false,true)=>LinOp::<T>::conj_apply(a,out,rhs,par,stack),
            (true,false)=>BiLinOp::<T>::transpose_apply(a,out,rhs,par,stack),(true,true)=>BiLinOp::<T>::adjoint_apply(a,out,rhs,par,stack),
        });
    }
}
impl<T: ComplexField> std::fmt::Debug for Operator<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeOperator")
            .field("shape", &self.shape())
            .finish()
    }
}
impl<T: ComplexField> LinOp<T> for Operator<T> {
    fn nrows(&self) -> usize {
        self.shape().0
    }
    fn ncols(&self) -> usize {
        self.shape().1
    }
    fn apply_scratch(&self, k: usize, par: ParShape) -> StackReq {
        dispatch!(self,a=>if self.transpose {BiLinOp::<T>::transpose_apply_scratch(a,k,par)}else{LinOp::<T>::apply_scratch(a,k,par)})
    }
    fn apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, stack: &mut MemStack) {
        self.run(out, rhs, par, stack, false, false)
    }
    fn conj_apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, stack: &mut MemStack) {
        self.run(out, rhs, par, stack, false, true)
    }
}
impl<T: ComplexField> BiLinOp<T> for Operator<T> {
    fn transpose_apply_scratch(&self, k: usize, par: ParShape) -> StackReq {
        dispatch!(self,a=>if self.transpose {LinOp::<T>::apply_scratch(a,k,par)}else{BiLinOp::<T>::transpose_apply_scratch(a,k,par)})
    }
    fn transpose_apply(
        &self,
        out: MatMut<'_, T>,
        rhs: MatRef<'_, T>,
        par: Par,
        stack: &mut MemStack,
    ) {
        self.run(out, rhs, par, stack, true, false)
    }
    fn adjoint_apply(
        &self,
        out: MatMut<'_, T>,
        rhs: MatRef<'_, T>,
        par: Par,
        stack: &mut MemStack,
    ) {
        self.run(out, rhs, par, stack, true, true)
    }
}
impl<T: ComplexField> Precond<T> for Operator<T> {}
impl<T: ComplexField> BiPrecond<T> for Operator<T> {}
#[derive(Clone)]
enum Data {
    F32(Operator<f32>),
    F64(Operator<f64>),
    C32(Operator<c32>),
    C64(Operator<c64>),
}
#[julia]
#[derive(Default)]
pub struct OwnedLinearOperator {
    data: Option<Data>,
}
impl OwnedLinearOperator {
    fn data(&self) -> Result<&Data, String> {
        self.data
            .as_ref()
            .ok_or("linear operator is not initialized".into())
    }
}
#[julia]
impl OwnedLinearOperator {
    #[julia]
    pub fn new() -> Self {
        Self::default()
    }
    #[julia]
    pub fn operator_load(&mut self, source: *const u8, kind: i32) -> Result<(), String> {
        guarded(|| {
            self.data = Some(match kind {
                0 => Data::F64(Operator::new(Base::Dense(
                    unsafe { &*source.cast::<OwnedMatrix>() }.matrix.clone(),
                ))),
                1 => match unsafe { &*source.cast::<TypedMatrix>() }.data()? {
                    MatrixData::F32(a) => Data::F32(Operator::new(Base::Dense(a.clone()))),
                    MatrixData::F64(a) => Data::F64(Operator::new(Base::Dense(a.clone()))),
                    MatrixData::C32(a) => Data::C32(Operator::new(Base::Dense(a.clone()))),
                    MatrixData::C64(a) => Data::C64(Operator::new(Base::Dense(a.clone()))),
                },
                2 => match unsafe { &*source.cast::<SparseOwner>() }.data()? {
                    SparseData::F32(a) => Data::F32(Operator::new(Base::Sparse(a.clone()))),
                    SparseData::F64(a) => Data::F64(Operator::new(Base::Sparse(a.clone()))),
                    SparseData::C32(a) => Data::C32(Operator::new(Base::Sparse(a.clone()))),
                    SparseData::C64(a) => Data::C64(Operator::new(Base::Sparse(a.clone()))),
                },
                3 => match unsafe { &*source.cast::<VectorOwner>() }.data()? {
                    VectorData::F32(VectorStore::Diagonal(a)) => {
                        Data::F32(Operator::new(Base::Diagonal(a.clone())))
                    }
                    VectorData::F64(VectorStore::Diagonal(a)) => {
                        Data::F64(Operator::new(Base::Diagonal(a.clone())))
                    }
                    VectorData::C32(VectorStore::Diagonal(a)) => {
                        Data::C32(Operator::new(Base::Diagonal(a.clone())))
                    }
                    VectorData::C64(VectorStore::Diagonal(a)) => {
                        Data::C64(Operator::new(Base::Diagonal(a.clone())))
                    }
                    _ => return Err("operator source must be a diagonal owner".into()),
                },
                _ => return Err("invalid operator source kind".into()),
            });
            Ok(())
        })
    }
    #[julia]
    pub fn operator_identity(&mut self, n: usize, tag: i32) -> Result<(), String> {
        guarded(|| {
            let p = IdentityPrecond { dim: n };
            self.data = Some(match tag {
                0 => Data::F32(Operator::new(Base::Identity(p))),
                1 => Data::F64(Operator::new(Base::Identity(p))),
                2 => Data::C32(Operator::new(Base::Identity(p))),
                3 => Data::C64(Operator::new(Base::Identity(p))),
                _ => return Err("unsupported operator scalar".into()),
            });
            Ok(())
        })
    }
    #[julia]
    pub fn operator_clone(
        &mut self,
        source: &Self,
        transpose: bool,
        conjugate: bool,
    ) -> Result<(), String> {
        guarded(|| {
            fn clone<T: ComplexField>(a: &Operator<T>, t: bool, c: bool) -> Operator<T> {
                let mut a = a.clone();
                a.transpose ^= t;
                a.conjugate ^= c;
                a
            }
            self.data = Some(match source.data()? {
                Data::F32(a) => Data::F32(clone(a, transpose, conjugate)),
                Data::F64(a) => Data::F64(clone(a, transpose, conjugate)),
                Data::C32(a) => Data::C32(clone(a, transpose, conjugate)),
                Data::C64(a) => Data::C64(clone(a, transpose, conjugate)),
            });
            Ok(())
        })
    }
}
fn product<T: ComplexField>(a: &Operator<T>, b: MatRef<'_, T>) -> Result<Mat<T>, String> {
    if a.ncols() != b.nrows() {
        return Err("operator product dimensions must match".into());
    }
    let mut out = Mat::zeros(a.nrows(), b.ncols());
    let par = Par::Seq;
    let mut mem = MemBuffer::try_new(a.apply_scratch(b.ncols(), par.shape()))
        .map_err(|e| format!("{e:?}"))?;
    a.apply(out.as_mut(), b, par, MemStack::new(&mut mem));
    Ok(out)
}
fn precondition<T: ComplexField>(a: &Operator<T>, b: MatMut<'_, T>) -> Result<(), String> {
    if a.nrows() != a.ncols() || a.ncols() != b.nrows() {
        return Err("preconditioner dimensions must match".into());
    }
    let par = Par::Seq;
    let mut mem = MemBuffer::try_new(a.apply_in_place_scratch(b.ncols(), par.shape()))
        .map_err(|e| format!("{e:?}"))?;
    a.apply_in_place(b, par, MemStack::new(&mut mem));
    Ok(())
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn operator_product(
        &mut self,
        operator: &OwnedLinearOperator,
        rhs: &OwnedMatrix,
    ) -> Result<(), String> {
        guarded(|| {
            let b = rhs;
            match operator.data()? {
                Data::F64(a) => {
                    self.matrix = product(a, b.matrix.as_ref())?;
                    Ok(())
                }
                _ => Err("operator scalar types must match".into()),
            }
        })
    }
    #[julia]
    pub fn operator_dense_from(&mut self, operator: &OwnedLinearOperator) -> Result<(), String> {
        guarded(|| match operator.data()? {
            Data::F64(Operator {
                base: Base::Dense(a),
                ..
            }) => {
                self.matrix = a.clone();
                Ok(())
            }
            _ => Err("operator has no matching dense source".into()),
        })
    }
    #[julia]
    pub fn operator_precondition(&mut self, operator: &OwnedLinearOperator) -> Result<(), String> {
        guarded(|| match operator.data()? {
            Data::F64(a) => precondition(a, self.matrix.as_mut()),
            _ => Err("operator scalar types must match".into()),
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn operator_product(
        &mut self,
        operator: &OwnedLinearOperator,
        rhs: &TypedMatrix,
    ) -> Result<(), String> {
        guarded(|| {
            self.data = Some(match (operator.data()?, rhs.data()?) {
                (Data::F32(a), MatrixData::F32(b)) => MatrixData::F32(product(a, b.as_ref())?),
                (Data::F64(a), MatrixData::F64(b)) => MatrixData::F64(product(a, b.as_ref())?),
                (Data::C32(a), MatrixData::C32(b)) => MatrixData::C32(product(a, b.as_ref())?),
                (Data::C64(a), MatrixData::C64(b)) => MatrixData::C64(product(a, b.as_ref())?),
                _ => return Err("operator scalar types must match".into()),
            });
            Ok(())
        })
    }
    #[julia]
    pub fn operator_dense_from(&mut self, operator: &OwnedLinearOperator) -> Result<(), String> {
        guarded(|| {
            self.data = Some(match operator.data()? {
                Data::F32(Operator {
                    base: Base::Dense(a),
                    ..
                }) => MatrixData::F32(a.clone()),
                Data::F64(Operator {
                    base: Base::Dense(a),
                    ..
                }) => MatrixData::F64(a.clone()),
                Data::C32(Operator {
                    base: Base::Dense(a),
                    ..
                }) => MatrixData::C32(a.clone()),
                Data::C64(Operator {
                    base: Base::Dense(a),
                    ..
                }) => MatrixData::C64(a.clone()),
                _ => return Err("operator has no matching dense source".into()),
            });
            Ok(())
        })
    }
    #[julia]
    pub fn operator_precondition(&mut self, operator: &OwnedLinearOperator) -> Result<(), String> {
        guarded(|| {
            match (
                operator.data()?,
                self.data.as_mut().ok_or("matrix is not initialized")?,
            ) {
                (Data::F32(a), MatrixData::F32(b)) => precondition(a, b.as_mut()),
                (Data::F64(a), MatrixData::F64(b)) => precondition(a, b.as_mut()),
                (Data::C32(a), MatrixData::C32(b)) => precondition(a, b.as_mut()),
                (Data::C64(a), MatrixData::C64(b)) => precondition(a, b.as_mut()),
                _ => Err("operator scalar types must match".into()),
            }
        })
    }
}
#[julia]
impl SparseOwner {
    #[julia]
    pub fn operator_sparse_from(&mut self, operator: &OwnedLinearOperator) -> Result<(), String> {
        guarded(|| {
            self.data = Some(match operator.data()? {
                Data::F32(Operator {
                    base: Base::Sparse(a),
                    ..
                }) => SparseData::F32(a.clone()),
                Data::F64(Operator {
                    base: Base::Sparse(a),
                    ..
                }) => SparseData::F64(a.clone()),
                Data::C32(Operator {
                    base: Base::Sparse(a),
                    ..
                }) => SparseData::C32(a.clone()),
                Data::C64(Operator {
                    base: Base::Sparse(a),
                    ..
                }) => SparseData::C64(a.clone()),
                _ => return Err("operator has no matching sparse source".into()),
            });
            Ok(())
        })
    }
}
#[julia]
impl VectorOwner {
    #[julia]
    pub fn operator_diagonal_from(&mut self, operator: &OwnedLinearOperator) -> Result<(), String> {
        guarded(|| {
            self.data = Some(match operator.data()? {
                Data::F32(Operator {
                    base: Base::Diagonal(a),
                    ..
                }) => VectorData::F32(VectorStore::Diagonal(a.clone())),
                Data::F64(Operator {
                    base: Base::Diagonal(a),
                    ..
                }) => VectorData::F64(VectorStore::Diagonal(a.clone())),
                Data::C32(Operator {
                    base: Base::Diagonal(a),
                    ..
                }) => VectorData::C32(VectorStore::Diagonal(a.clone())),
                Data::C64(Operator {
                    base: Base::Diagonal(a),
                    ..
                }) => VectorData::C64(VectorStore::Diagonal(a.clone())),
                _ => return Err("operator has no matching diagonal source".into()),
            });
            Ok(())
        })
    }
    #[julia]
    pub fn operator_precondition(&mut self, operator: &OwnedLinearOperator) -> Result<(), String> {
        guarded(|| {
            match (
                operator.data()?,
                self.data.as_mut().ok_or("vector is not initialized")?,
            ) {
                (Data::F32(a), VectorData::F32(b)) => precondition(a, b.as_mat_mut()),
                (Data::F64(a), VectorData::F64(b)) => precondition(a, b.as_mat_mut()),
                (Data::C32(a), VectorData::C32(b)) => precondition(a, b.as_mat_mut()),
                (Data::C64(a), VectorData::C64(b)) => precondition(a, b.as_mat_mut()),
                _ => Err("operator scalar types must match".into()),
            }
        })
    }
}
fn solve<T: ComplexField>(
    a: &Operator<T>,
    m: Option<&Operator<T>>,
    b: MatRef<'_, T>,
    initial: MatRef<'_, T>,
    atol: f64,
    rtol: f64,
    maxiter: usize,
    stats: *mut u8,
    iterations: *mut usize,
) -> Result<Mat<T>, String> {
    if a.nrows() != a.ncols() || a.ncols() != b.nrows() || initial.shape() != b.shape() {
        return Err("CG dimensions must match".into());
    }
    let identity = Operator::<T>::new(Base::Identity(IdentityPrecond { dim: a.nrows() }));
    let m = m.unwrap_or(&identity);
    if m.shape() != a.shape() {
        return Err("CG preconditioner dimensions must match".into());
    }
    let mut out = initial.to_owned();
    let mut callbacks = 0usize;
    let par = Par::Seq;
    let mut mem = MemBuffer::try_new(cg::conjugate_gradient_scratch(m, a, b.ncols(), par.shape()))
        .map_err(|e| format!("{e:?}"))?;
    let result = cg::conjugate_gradient(
        out.as_mut(),
        m,
        a,
        b,
        cg::CgParams {
            initial_guess: InitialGuessStatus::MaybeNonZero,
            abs_tolerance: from_f64::<T::Real>(atol),
            rel_tolerance: from_f64::<T::Real>(rtol),
            max_iters: maxiter,
            ..Default::default()
        },
        |_| {
            callbacks += 1;
        },
        par,
        MemStack::new(&mut mem),
    );
    let (mut status, estimate_abs, estimate_rel, count) = match result {
        Ok(info) => (0, info.abs_residual, info.rel_residual, info.iter_count),
        Err(CgError::NoConvergence {
            abs_residual,
            rel_residual,
        }) => (1, abs_residual, rel_residual, callbacks),
        Err(CgError::NonPositiveDefiniteOperator) => {
            (2, nan::<T::Real>(), nan::<T::Real>(), callbacks)
        }
        Err(CgError::NonPositiveDefinitePreconditioner) => {
            (3, nan::<T::Real>(), nan::<T::Real>(), callbacks)
        }
    };
    let mut residual = product(a, out.as_ref())?;
    for j in 0..b.ncols() {
        for i in 0..b.nrows() {
            residual[(i, j)] = &b[(i, j)] - &residual[(i, j)];
        }
    }
    let actual = residual.norm_l2();
    let bnorm = b.norm_l2();
    let relative = if bnorm == zero::<T::Real>() {
        copy(&actual)
    } else {
        &actual / &bnorm
    };
    let abs_tol = from_f64::<T::Real>(atol);
    let rel_tol = from_f64::<T::Real>(rtol) * bnorm;
    let threshold = if abs_tol > rel_tol { abs_tol } else { rel_tol };
    if status == 0 && (!is_finite(&actual) || actual > threshold) {
        status = 4;
    }
    unsafe {
        let values = [
            from_f64::<T>(status as f64),
            from_real::<T>(&actual),
            from_real::<T>(&relative),
            from_real::<T>(&estimate_abs),
            from_real::<T>(&estimate_rel),
        ];
        for (i, v) in values.into_iter().enumerate() {
            stats.cast::<T>().add(i).write(v);
        }
        iterations.write(count);
    }
    Ok(out)
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn operator_cg(
        &mut self,
        operator: &OwnedLinearOperator,
        precond: *const u8,
        rhs: &OwnedMatrix,
        initial: &OwnedMatrix,
        atol: f64,
        rtol: f64,
        maxiter: usize,
        stats: *mut u8,
        iterations: *mut usize,
    ) -> Result<(), String> {
        guarded(|| {
            let a = operator.data()?;
            let m = if precond.is_null() {
                None
            } else {
                Some(unsafe { &*precond.cast::<OwnedLinearOperator>() }.data()?)
            };
            let b = rhs;
            let x = initial;
            match (a, m) {
                (Data::F64(a), None) => {
                    self.matrix = solve(
                        a,
                        None,
                        b.matrix.as_ref(),
                        x.matrix.as_ref(),
                        atol,
                        rtol,
                        maxiter,
                        stats,
                        iterations,
                    )?
                }
                (Data::F64(a), Some(Data::F64(m))) => {
                    self.matrix = solve(
                        a,
                        Some(m),
                        b.matrix.as_ref(),
                        x.matrix.as_ref(),
                        atol,
                        rtol,
                        maxiter,
                        stats,
                        iterations,
                    )?
                }
                _ => return Err("CG scalar types must match".into()),
            };
            Ok(())
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn operator_cg(
        &mut self,
        operator: &OwnedLinearOperator,
        precond: *const u8,
        rhs: &TypedMatrix,
        initial: &TypedMatrix,
        atol: f64,
        rtol: f64,
        maxiter: usize,
        stats: *mut u8,
        iterations: *mut usize,
    ) -> Result<(), String> {
        guarded(|| {
            let a = operator.data()?;
            let m = if precond.is_null() {
                None
            } else {
                Some(unsafe { &*precond.cast::<OwnedLinearOperator>() }.data()?)
            };
            let b = rhs.data()?;
            let x = initial.data()?;
            self.data = Some(match (a, m, b, x) {
                (Data::F32(a), None, MatrixData::F32(b), MatrixData::F32(x)) => {
                    MatrixData::F32(solve(
                        a,
                        None,
                        b.as_ref(),
                        x.as_ref(),
                        atol,
                        rtol,
                        maxiter,
                        stats,
                        iterations,
                    )?)
                }
                (Data::F32(a), Some(Data::F32(m)), MatrixData::F32(b), MatrixData::F32(x)) => {
                    MatrixData::F32(solve(
                        a,
                        Some(m),
                        b.as_ref(),
                        x.as_ref(),
                        atol,
                        rtol,
                        maxiter,
                        stats,
                        iterations,
                    )?)
                }
                (Data::C32(a), None, MatrixData::C32(b), MatrixData::C32(x)) => {
                    MatrixData::C32(solve(
                        a,
                        None,
                        b.as_ref(),
                        x.as_ref(),
                        atol,
                        rtol,
                        maxiter,
                        stats,
                        iterations,
                    )?)
                }
                (Data::C32(a), Some(Data::C32(m)), MatrixData::C32(b), MatrixData::C32(x)) => {
                    MatrixData::C32(solve(
                        a,
                        Some(m),
                        b.as_ref(),
                        x.as_ref(),
                        atol,
                        rtol,
                        maxiter,
                        stats,
                        iterations,
                    )?)
                }
                (Data::C64(a), None, MatrixData::C64(b), MatrixData::C64(x)) => {
                    MatrixData::C64(solve(
                        a,
                        None,
                        b.as_ref(),
                        x.as_ref(),
                        atol,
                        rtol,
                        maxiter,
                        stats,
                        iterations,
                    )?)
                }
                (Data::C64(a), Some(Data::C64(m)), MatrixData::C64(b), MatrixData::C64(x)) => {
                    MatrixData::C64(solve(
                        a,
                        Some(m),
                        b.as_ref(),
                        x.as_ref(),
                        atol,
                        rtol,
                        maxiter,
                        stats,
                        iterations,
                    )?)
                }
                _ => return Err("CG scalar types must match".into()),
            });
            Ok(())
        })
    }
}

fn solve_bicg<T: ComplexField>(
    a: &Operator<T>,
    m: Option<&Operator<T>>,
    right: Option<&Operator<T>>,
    b: MatRef<'_, T>,
    initial: MatRef<'_, T>,
    atol: f64,
    rtol: f64,
    maxiter: usize,
    stats: *mut u8,
    iterations: *mut usize,
) -> Result<Mat<T>, String> {
    if a.nrows() != a.ncols() || a.ncols() != b.nrows() || initial.shape() != b.shape() {
        return Err("BiCGSTAB dimensions must match".into());
    }
    let identity = Operator::<T>::new(Base::Identity(IdentityPrecond { dim: a.nrows() }));
    let m = m.unwrap_or(&identity);
    if m.shape() != a.shape() {
        return Err("BiCGSTAB preconditioner dimensions must match".into());
    }
    let right = right.unwrap_or(&identity);
    if right.shape() != a.shape() {
        return Err("BiCGSTAB right preconditioner dimensions must match".into());
    }
    let mut out = initial.to_owned();
    let mut callbacks = 0usize;
    let par = Par::Seq;
    let mut mem = MemBuffer::try_new(faer::matrix_free::bicgstab::bicgstab_scratch(
        m,
        right,
        a,
        b.ncols(),
        par.shape(),
    ))
    .map_err(|e| format!("{e:?}"))?;
    let result = faer::matrix_free::bicgstab::bicgstab(
        out.as_mut(),
        m,
        right,
        a,
        b,
        faer::matrix_free::bicgstab::BicgParams {
            initial_guess: InitialGuessStatus::MaybeNonZero,
            abs_tolerance: from_f64::<T::Real>(atol),
            rel_tolerance: from_f64::<T::Real>(rtol),
            max_iters: maxiter,
            ..Default::default()
        },
        |_| {
            callbacks += 1;
        },
        par,
        MemStack::new(&mut mem),
    );
    let (mut status, estimate_abs, estimate_rel, count) = match result {
        Ok(info) => (0, info.abs_residual, info.rel_residual, info.iter_count),
        Err(faer::matrix_free::bicgstab::BicgError::NoConvergence {
            abs_residual,
            rel_residual,
        }) => (1, abs_residual, rel_residual, callbacks),
    };
    let mut residual = product(a, out.as_ref())?;
    for j in 0..b.ncols() {
        for i in 0..b.nrows() {
            residual[(i, j)] = &b[(i, j)] - &residual[(i, j)];
        }
    }
    let actual = residual.norm_l2();
    let bnorm = b.norm_l2();
    let relative = if bnorm == zero::<T::Real>() {
        copy(&actual)
    } else {
        &actual / &bnorm
    };
    let abs_tol = from_f64::<T::Real>(atol);
    let rel_tol = from_f64::<T::Real>(rtol) * bnorm;
    let threshold = if abs_tol > rel_tol { abs_tol } else { rel_tol };
    if status == 0 && (!is_finite(&actual) || actual > threshold) {
        status = 4;
    }
    unsafe {
        let values = [
            from_f64::<T>(status as f64),
            from_real::<T>(&actual),
            from_real::<T>(&relative),
            from_real::<T>(&estimate_abs),
            from_real::<T>(&estimate_rel),
        ];
        for (i, v) in values.into_iter().enumerate() {
            stats.cast::<T>().add(i).write(v);
        }
        iterations.write(count);
    }
    Ok(out)
}

#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn operator_bicgstab(
        &mut self,
        operator: &OwnedLinearOperator,
        left: *const u8,
        right: *const u8,
        rhs: &OwnedMatrix,
        initial: &OwnedMatrix,
        atol: f64,
        rtol: f64,
        maxiter: usize,
        stats: *mut u8,
        iterations: *mut usize,
    ) -> Result<(), String> {
        guarded(|| {
            let a = operator.data()?;
            let left = if left.is_null() {
                None
            } else {
                Some(unsafe { &*left.cast::<OwnedLinearOperator>() }.data()?)
            };
            let right = if right.is_null() {
                None
            } else {
                Some(unsafe { &*right.cast::<OwnedLinearOperator>() }.data()?)
            };
            let b = rhs;
            let x = initial;
            let Data::F64(a) = a else {
                return Err("operator scalar types must match".into());
            };
            let left = match left {
                None => None,
                Some(Data::F64(m)) => Some(m),
                _ => return Err("preconditioner scalar types must match".into()),
            };
            let right = match right {
                None => None,
                Some(Data::F64(m)) => Some(m),
                _ => return Err("preconditioner scalar types must match".into()),
            };
            self.matrix = solve_bicg(
                a,
                left,
                right,
                b.matrix.as_ref(),
                x.matrix.as_ref(),
                atol,
                rtol,
                maxiter,
                stats,
                iterations,
            )?;
            Ok(())
        })
    }
}

#[julia]
impl TypedMatrix {
    #[julia]
    pub fn operator_bicgstab(
        &mut self,
        operator: &OwnedLinearOperator,
        left: *const u8,
        right: *const u8,
        rhs: &TypedMatrix,
        initial: &TypedMatrix,
        atol: f64,
        rtol: f64,
        maxiter: usize,
        stats: *mut u8,
        iterations: *mut usize,
    ) -> Result<(), String> {
        guarded(|| {
            let a = operator.data()?;
            let left = if left.is_null() {
                None
            } else {
                Some(unsafe { &*left.cast::<OwnedLinearOperator>() }.data()?)
            };
            let right = if right.is_null() {
                None
            } else {
                Some(unsafe { &*right.cast::<OwnedLinearOperator>() }.data()?)
            };
            let b = rhs.data()?;
            let x = initial.data()?;
            self.data = Some(match (a, b, x) {
                (Data::F32(a), MatrixData::F32(b), MatrixData::F32(x)) => {
                    let left = match left {
                        None => None,
                        Some(Data::F32(m)) => Some(m),
                        _ => return Err("preconditioner scalar types must match".into()),
                    };
                    let right = match right {
                        None => None,
                        Some(Data::F32(m)) => Some(m),
                        _ => return Err("preconditioner scalar types must match".into()),
                    };
                    MatrixData::F32(solve_bicg(
                        a,
                        left,
                        right,
                        b.as_ref(),
                        x.as_ref(),
                        atol,
                        rtol,
                        maxiter,
                        stats,
                        iterations,
                    )?)
                }
                (Data::C32(a), MatrixData::C32(b), MatrixData::C32(x)) => {
                    let left = match left {
                        None => None,
                        Some(Data::C32(m)) => Some(m),
                        _ => return Err("preconditioner scalar types must match".into()),
                    };
                    let right = match right {
                        None => None,
                        Some(Data::C32(m)) => Some(m),
                        _ => return Err("preconditioner scalar types must match".into()),
                    };
                    MatrixData::C32(solve_bicg(
                        a,
                        left,
                        right,
                        b.as_ref(),
                        x.as_ref(),
                        atol,
                        rtol,
                        maxiter,
                        stats,
                        iterations,
                    )?)
                }
                (Data::C64(a), MatrixData::C64(b), MatrixData::C64(x)) => {
                    let left = match left {
                        None => None,
                        Some(Data::C64(m)) => Some(m),
                        _ => return Err("preconditioner scalar types must match".into()),
                    };
                    let right = match right {
                        None => None,
                        Some(Data::C64(m)) => Some(m),
                        _ => return Err("preconditioner scalar types must match".into()),
                    };
                    MatrixData::C64(solve_bicg(
                        a,
                        left,
                        right,
                        b.as_ref(),
                        x.as_ref(),
                        atol,
                        rtol,
                        maxiter,
                        stats,
                        iterations,
                    )?)
                }
                _ => return Err("operator scalar types must match".into()),
            });
            Ok(())
        })
    }
}

fn solve_lsmr<T: ComplexField>(
    a: &Operator<T>,
    m: Option<&Operator<T>>,
    b: MatRef<'_, T>,
    initial: MatRef<'_, T>,
    atol: f64,
    rtol: f64,
    maxiter: usize,
    stats: *mut u8,
    iterations: *mut usize,
) -> Result<Mat<T>, String> {
    if a.nrows() != b.nrows() || initial.nrows() != a.ncols() || initial.ncols() != b.ncols() {
        return Err("LSMR dimensions must match".into());
    }
    let identity = Operator::<T>::new(Base::Identity(IdentityPrecond { dim: a.ncols() }));
    let m = m.unwrap_or(&identity);
    if m.shape() != (a.ncols(), a.ncols()) {
        return Err("LSMR preconditioner dimensions must match".into());
    }
    let mut adjoint = a.clone();
    adjoint.transpose = !adjoint.transpose;
    adjoint.conjugate = !adjoint.conjugate;
    let mut madjoint = m.clone();
    madjoint.transpose = !madjoint.transpose;
    madjoint.conjugate = !madjoint.conjugate;
    let normal_ref = product(&madjoint, product(&adjoint, b)?.as_ref())?.norm_l2();
    let abs_tol = from_f64::<T::Real>(atol);
    let requested_rel = from_f64::<T::Real>(rtol);
    let relative_abs = if normal_ref == zero::<T::Real>() {
        zero::<T::Real>()
    } else {
        &abs_tol / &normal_ref
    };
    let effective_rel = if relative_abs > requested_rel {
        relative_abs
    } else {
        copy(&requested_rel)
    };
    let mut out = initial.to_owned();
    let mut callbacks = 0usize;
    let par = Par::Seq;
    let (mut status, estimate_abs, estimate_rel, count) =
        if a.nrows() == 0 || a.ncols() == 0 || b.ncols() == 0 {
            out.fill(zero());
            (0, zero::<T::Real>(), zero::<T::Real>(), 0)
        } else {
            let mut mem = MemBuffer::try_new(faer::matrix_free::lsmr::lsmr_scratch(
                m,
                a,
                b.ncols(),
                par.shape(),
            ))
            .map_err(|e| format!("{e:?}"))?;
            let result = faer::matrix_free::lsmr::lsmr(
                out.as_mut(),
                m,
                a,
                b,
                faer::matrix_free::lsmr::LsmrParams {
                    initial_guess: InitialGuessStatus::MaybeNonZero,
                    abs_tolerance: copy(&abs_tol),
                    rel_tolerance: effective_rel,
                    max_iters: maxiter,
                    ..Default::default()
                },
                |_| {
                    callbacks += 1;
                },
                par,
                MemStack::new(&mut mem),
            );
            match result {
                Ok(info) => (0, info.abs_residual, info.rel_residual, info.iter_count),
                Err(faer::matrix_free::lsmr::LsmrError::NoConvergence {
                    abs_residual,
                    rel_residual,
                }) => (1, abs_residual, rel_residual, callbacks),
            }
        };
    let mut residual = product(a, out.as_ref())?;
    for j in 0..b.ncols() {
        for i in 0..b.nrows() {
            residual[(i, j)] = &b[(i, j)] - &residual[(i, j)];
        }
    }
    let actual = residual.norm_l2();
    let bnorm = b.norm_l2();
    let relative = if bnorm == zero::<T::Real>() {
        copy(&actual)
    } else {
        &actual / &bnorm
    };
    let normal = product(&madjoint, product(&adjoint, residual.as_ref())?.as_ref())?.norm_l2();
    let normal_relative = if normal_ref == zero::<T::Real>() {
        copy(&normal)
    } else {
        &normal / &normal_ref
    };
    let rel_tol = &requested_rel * &normal_ref;
    let threshold = if abs_tol > rel_tol { abs_tol } else { rel_tol };
    if status == 0 && (!is_finite(&normal) || normal > threshold) {
        status = 4;
    }
    unsafe {
        let values = [
            from_f64::<T>(status as f64),
            from_real::<T>(&actual),
            from_real::<T>(&relative),
            from_real::<T>(&normal),
            from_real::<T>(&normal_relative),
            from_real::<T>(&estimate_abs),
            from_real::<T>(&estimate_rel),
        ];
        for (i, v) in values.into_iter().enumerate() {
            stats.cast::<T>().add(i).write(v);
        }
        iterations.write(count);
    }
    Ok(out)
}

#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn operator_lsmr(
        &mut self,
        operator: &OwnedLinearOperator,
        precond: *const u8,
        rhs: &OwnedMatrix,
        initial: &OwnedMatrix,
        atol: f64,
        rtol: f64,
        maxiter: usize,
        stats: *mut u8,
        iterations: *mut usize,
    ) -> Result<(), String> {
        guarded(|| {
            let a = operator.data()?;
            let m = if precond.is_null() {
                None
            } else {
                Some(unsafe { &*precond.cast::<OwnedLinearOperator>() }.data()?)
            };
            let b = rhs;
            let x = initial;
            match (a, m) {
                (Data::F64(a), None) => {
                    self.matrix = solve_lsmr(
                        a,
                        None,
                        b.matrix.as_ref(),
                        x.matrix.as_ref(),
                        atol,
                        rtol,
                        maxiter,
                        stats,
                        iterations,
                    )?
                }
                (Data::F64(a), Some(Data::F64(m))) => {
                    self.matrix = solve_lsmr(
                        a,
                        Some(m),
                        b.matrix.as_ref(),
                        x.matrix.as_ref(),
                        atol,
                        rtol,
                        maxiter,
                        stats,
                        iterations,
                    )?
                }
                _ => return Err("LSMR scalar types must match".into()),
            };
            Ok(())
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn operator_lsmr(
        &mut self,
        operator: &OwnedLinearOperator,
        precond: *const u8,
        rhs: &TypedMatrix,
        initial: &TypedMatrix,
        atol: f64,
        rtol: f64,
        maxiter: usize,
        stats: *mut u8,
        iterations: *mut usize,
    ) -> Result<(), String> {
        guarded(|| {
            let a = operator.data()?;
            let m = if precond.is_null() {
                None
            } else {
                Some(unsafe { &*precond.cast::<OwnedLinearOperator>() }.data()?)
            };
            let b = rhs.data()?;
            let x = initial.data()?;
            self.data = Some(match (a, m, b, x) {
                (Data::F32(a), None, MatrixData::F32(b), MatrixData::F32(x)) => {
                    MatrixData::F32(solve_lsmr(
                        a,
                        None,
                        b.as_ref(),
                        x.as_ref(),
                        atol,
                        rtol,
                        maxiter,
                        stats,
                        iterations,
                    )?)
                }
                (Data::F32(a), Some(Data::F32(m)), MatrixData::F32(b), MatrixData::F32(x)) => {
                    MatrixData::F32(solve_lsmr(
                        a,
                        Some(m),
                        b.as_ref(),
                        x.as_ref(),
                        atol,
                        rtol,
                        maxiter,
                        stats,
                        iterations,
                    )?)
                }
                (Data::C32(a), None, MatrixData::C32(b), MatrixData::C32(x)) => {
                    MatrixData::C32(solve_lsmr(
                        a,
                        None,
                        b.as_ref(),
                        x.as_ref(),
                        atol,
                        rtol,
                        maxiter,
                        stats,
                        iterations,
                    )?)
                }
                (Data::C32(a), Some(Data::C32(m)), MatrixData::C32(b), MatrixData::C32(x)) => {
                    MatrixData::C32(solve_lsmr(
                        a,
                        Some(m),
                        b.as_ref(),
                        x.as_ref(),
                        atol,
                        rtol,
                        maxiter,
                        stats,
                        iterations,
                    )?)
                }
                (Data::C64(a), None, MatrixData::C64(b), MatrixData::C64(x)) => {
                    MatrixData::C64(solve_lsmr(
                        a,
                        None,
                        b.as_ref(),
                        x.as_ref(),
                        atol,
                        rtol,
                        maxiter,
                        stats,
                        iterations,
                    )?)
                }
                (Data::C64(a), Some(Data::C64(m)), MatrixData::C64(b), MatrixData::C64(x)) => {
                    MatrixData::C64(solve_lsmr(
                        a,
                        Some(m),
                        b.as_ref(),
                        x.as_ref(),
                        atol,
                        rtol,
                        maxiter,
                        stats,
                        iterations,
                    )?)
                }
                _ => return Err("LSMR scalar types must match".into()),
            });
            Ok(())
        })
    }
}

// Partial spectra export primitive buffers with dimensions rechecked against
// the current native owner before any write.
#[julia]
impl OwnedLinearOperator {
    #[julia]
    pub fn operator_partial(
        &self,
        kind: i32,
        k: usize,
        m: usize,
        n: usize,
        tag: i32,
        start: *const u8,
        tolerance: f64,
        min_dim: usize,
        max_dim: usize,
        max_restarts: usize,
        vectors: *mut u8,
        values: *mut u8,
        right: *mut u8,
    ) -> Result<usize, String> {
        guarded(|| {
            use faer::matrix_free::eigen as pe;
            let mut params = pe::PartialEigenParams::default();
            params.min_dim = min_dim;
            params.max_dim = max_dim;
            params.max_restarts = max_restarts;
            if !(0..=2).contains(&kind) || !tolerance.is_finite() || tolerance < 0.0 {
                return Err("invalid partial spectrum parameters".into());
            }
            macro_rules! run {
                ($a:expr, $t:ty, $c:ty, $tag:expr) => {{
                    let a = $a;
                    if tag != $tag || a.shape() != (m, n) || k > m.min(n) || (kind != 2 && m != n) {
                        return Err("partial spectrum owner/type/dimensions mismatch".into());
                    }
                    if k == 0 {
                        return Ok(0);
                    }
                    // Upstream self-adjoint code panics when its subspace spans n.
                    if kind != 0 && max_dim.max(64).max(2 * k) >= m.min(n) {
                        return Err(
                            "self-adjoint partial spectrum requires a smaller projection subspace"
                                .into(),
                        );
                    }
                    let v = faer::Col::<$t>::from_fn(n, |i| unsafe {
                        start.cast::<$t>().add(i).read()
                    });
                    if !v.norm_l2().is_finite() || v.norm_l2() == 0.0 {
                        return Err("partial spectrum start must have finite nonzero norm".into());
                    }
                    let req = match kind {
                        0 => pe::partial_eigen_scratch(a, k, Par::Seq.shape(), params),
                        1 => pe::partial_self_adjoint_eigen_scratch(a, k, Par::Seq.shape(), params),
                        _ => pe::partial_svd_scratch(a, k, Par::Seq.shape(), params),
                    };
                    let mut mem = MemBuffer::new(req);
                    let stack = MemStack::new(&mut mem);
                    if kind == 0 {
                        let mut u = Mat::<$c>::zeros(n, k);
                        let mut s = vec![<$c>::new(0.0, 0.0); k];
                        let info = pe::partial_eigen(
                            u.as_mut(),
                            &mut s,
                            a,
                            v.as_ref(),
                            tolerance as _,
                            Par::Seq,
                            stack,
                            params,
                        );
                        let count = info.n_converged_eigen.min(k);
                        for j in 0..count {
                            unsafe {
                                values.cast::<$c>().add(j).write(s[j]);
                                for i in 0..n {
                                    vectors.cast::<$c>().add(i + j * n).write(u[(i, j)]);
                                }
                            }
                        }
                        Ok(count)
                    } else {
                        let mut u = Mat::<$t>::zeros(m, k);
                        let mut w = Mat::<$t>::zeros(n, k);
                        let mut s = vec![from_f64::<$t>(0.0); k];
                        let info = if kind == 1 {
                            pe::partial_self_adjoint_eigen(
                                u.as_mut(),
                                &mut s,
                                a,
                                v.as_ref(),
                                tolerance as _,
                                Par::Seq,
                                stack,
                                params,
                            )
                        } else {
                            pe::partial_svd(
                                u.as_mut(),
                                w.as_mut(),
                                &mut s,
                                a,
                                v.as_ref(),
                                tolerance as _,
                                Par::Seq,
                                stack,
                                params,
                            )
                        };
                        let count = info.n_converged_eigen.min(k);
                        for j in 0..count {
                            unsafe {
                                values.cast::<$t>().add(j).write(s[j]);
                                for i in 0..m {
                                    vectors.cast::<$t>().add(i + j * m).write(u[(i, j)]);
                                }
                                if kind == 2 {
                                    for i in 0..n {
                                        right.cast::<$t>().add(i + j * n).write(w[(i, j)]);
                                    }
                                }
                            }
                        }
                        Ok(count)
                    }
                }};
            }
            match self.data()? {
                Data::F32(a) => run!(a, f32, c32, 0),
                Data::F64(a) => run!(a, f64, c64, 1),
                Data::C32(a) => run!(a, c32, c32, 2),
                Data::C64(a) => run!(a, c64, c64, 3),
            }
        })
    }
}

#[derive(Clone)]
struct OperatorProduct<T> {
    left: Operator<T>,
    right: Operator<T>,
}
impl<T> std::fmt::Debug for OperatorProduct<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("OperatorProduct")
    }
}
impl<T: ComplexField> OperatorProduct<T> {
    fn run(
        &self,
        out: MatMut<'_, T>,
        rhs: MatRef<'_, T>,
        mut par: Par,
        stack: &mut MemStack,
        t: bool,
        c: bool,
    ) {
        let (first, second) = if t {
            (&self.left, &self.right)
        } else {
            (&self.right, &self.left)
        };
        let (mut temp, stack) = faer::linalg::temp_mat_zeroed::<T, _, _>(
            if t { first.ncols() } else { first.nrows() },
            rhs.ncols(),
            stack,
        );
        first.run(temp.as_mat_mut(), rhs, par.reborrow(), stack, t, c);
        second.run(out, temp.as_mat_ref(), par, stack, t, c);
    }
}
impl<T: ComplexField> LinOp<T> for OperatorProduct<T> {
    fn nrows(&self) -> usize {
        self.left.nrows()
    }
    fn ncols(&self) -> usize {
        self.right.ncols()
    }
    fn apply_scratch(&self, k: usize, p: ParShape) -> StackReq {
        faer::linalg::temp_mat_scratch::<T>(self.left.ncols(), k).and(
            self.left
                .apply_scratch(k, p)
                .or(self.right.apply_scratch(k, p)),
        )
    }
    fn apply(&self, o: MatMut<'_, T>, b: MatRef<'_, T>, p: Par, s: &mut MemStack) {
        self.run(o, b, p, s, false, false)
    }
    fn conj_apply(&self, o: MatMut<'_, T>, b: MatRef<'_, T>, p: Par, s: &mut MemStack) {
        self.run(o, b, p, s, false, true)
    }
}
impl<T: ComplexField> BiLinOp<T> for OperatorProduct<T> {
    fn transpose_apply_scratch(&self, k: usize, p: ParShape) -> StackReq {
        faer::linalg::temp_mat_scratch::<T>(self.left.ncols(), k).and(
            self.left
                .transpose_apply_scratch(k, p)
                .or(self.right.transpose_apply_scratch(k, p)),
        )
    }
    fn transpose_apply(&self, o: MatMut<'_, T>, b: MatRef<'_, T>, p: Par, s: &mut MemStack) {
        self.run(o, b, p, s, true, false)
    }
    fn adjoint_apply(&self, o: MatMut<'_, T>, b: MatRef<'_, T>, p: Par, s: &mut MemStack) {
        self.run(o, b, p, s, true, true)
    }
}
#[julia]
impl OwnedLinearOperator {
    #[julia]
    pub fn operator_compose(&mut self, left: &Self, right: &Self) -> Result<(), String> {
        guarded(|| {
            fn compose<T: ComplexField>(
                a: &Operator<T>,
                b: &Operator<T>,
            ) -> Result<Operator<T>, String> {
                if a.ncols() != b.nrows() {
                    return Err("operator composition dimensions must match".into());
                }
                Ok(Operator::new(Base::Product(Box::new(OperatorProduct {
                    left: a.clone(),
                    right: b.clone(),
                }))))
            }
            self.data = Some(match (left.data()?, right.data()?) {
                (Data::F32(a), Data::F32(b)) => Data::F32(compose(a, b)?),
                (Data::F64(a), Data::F64(b)) => Data::F64(compose(a, b)?),
                (Data::C32(a), Data::C32(b)) => Data::C32(compose(a, b)?),
                (Data::C64(a), Data::C64(b)) => Data::C64(compose(a, b)?),
                _ => return Err("composition scalar types must match".into()),
            });
            Ok(())
        })
    }
}

trait NativeValue: ComplexField {
    fn parts(&self) -> (f64, f64);
    fn from_parts(re: f64, im: f64) -> Self;
}
impl NativeValue for f32 {
    fn parts(&self) -> (f64, f64) {
        (*self as f64, 0.0)
    }
    fn from_parts(r: f64, _: f64) -> Self {
        r as f32
    }
}
impl NativeValue for f64 {
    fn parts(&self) -> (f64, f64) {
        (*self, 0.0)
    }
    fn from_parts(r: f64, _: f64) -> Self {
        r
    }
}
impl NativeValue for c32 {
    fn parts(&self) -> (f64, f64) {
        (self.re as f64, self.im as f64)
    }
    fn from_parts(r: f64, i: f64) -> Self {
        Self::new(r as f32, i as f32)
    }
}
impl NativeValue for c64 {
    fn parts(&self) -> (f64, f64) {
        (self.re, self.im)
    }
    fn from_parts(r: f64, i: f64) -> Self {
        Self::new(r, i)
    }
}
fn convert_operator<S: NativeValue, T: NativeValue>(
    a: &Operator<S>,
) -> Result<Operator<T>, String> {
    let cast = |x: &S| {
        let (r, i) = x.parts();
        T::from_parts(r, i)
    };
    let base = match &a.base {
        Base::InverseLu(f) => Base::InverseLu(Box::new(InverseLu {
            l: Mat::from_fn(f.l.nrows(), f.l.ncols(), |i, j| cast(&f.l[(i, j)])),
            u: Mat::from_fn(f.u.nrows(), f.u.ncols(), |i, j| cast(&f.u[(i, j)])),
            permutation: f.permutation.clone(),
        })),
        Base::Dense(m) => Base::Dense(Mat::from_fn(m.nrows(), m.ncols(), |i, j| cast(&m[(i, j)]))),
        Base::Diagonal(d) => {
            let mut out = Diag::zeros(d.dim());
            for i in 0..d.dim() {
                out.column_vector_mut()[i] = cast(&d.column_vector()[i]);
            }
            Base::Diagonal(out)
        }
        Base::Identity(i) => Base::Identity(IdentityPrecond { dim: i.dim }),
        Base::Product(p) => Base::Product(Box::new(OperatorProduct {
            left: convert_operator(&p.left)?,
            right: convert_operator(&p.right)?,
        })),
        Base::Sparse(s) => {
            let csr = matches!(s, Storage::Csr(_));
            let m = s.csc()?;
            let entries: Vec<_> = m
                .as_ref()
                .triplet_iter()
                .map(|e| faer::sparse::Triplet {
                    row: e.row,
                    col: e.col,
                    val: cast(e.val),
                })
                .collect();
            let m =
                faer::sparse::SparseColMat::try_new_from_triplets(m.nrows(), m.ncols(), &entries)
                    .map_err(|e| format!("{e:?}"))?;
            Base::Sparse(Storage::Csc(m).converted(csr)?)
        }
    };
    Ok(Operator {
        base,
        transpose: a.transpose,
        conjugate: a.conjugate,
    })
}
#[julia]
impl OwnedLinearOperator {
    #[julia]
    pub fn operator_promote(&mut self, source: &Self, tag: i32) -> Result<(), String> {
        guarded(|| {
            self.data = Some(match (source.data()?, tag) {
                (Data::F32(a), 0) => Data::F32(a.clone()),
                (Data::F32(a), 1) => Data::F64(convert_operator(a)?),
                (Data::F32(a), 2) => Data::C32(convert_operator(a)?),
                (Data::F32(a), 3) => Data::C64(convert_operator(a)?),
                (Data::F64(a), 1) => Data::F64(a.clone()),
                (Data::F64(a), 3) => Data::C64(convert_operator(a)?),
                (Data::C32(a), 2) => Data::C32(a.clone()),
                (Data::C32(a), 3) => Data::C64(convert_operator(a)?),
                (Data::C64(a), 3) => Data::C64(a.clone()),
                _ => return Err("operator conversion requires widening scalar promotion".into()),
            });
            Ok(())
        })
    }
}

#[derive(Clone)]
struct InverseLu<T> {
    l: Mat<T>,
    u: Mat<T>,
    permutation: faer::perm::Perm<usize>,
}
impl<T> std::fmt::Debug for InverseLu<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("InverseLu")
    }
}
impl<T: ComplexField> InverseLu<T> {
    fn new(f: &faer::linalg::solvers::PartialPivLu<T>) -> Result<Self, String> {
        let n = f.L().nrows();
        if n != f.U().ncols() {
            return Err("LU preconditioner must be square".into());
        }
        if (0..n).any(|i| abs(&f.U()[(i, i)]) == zero::<T::Real>()) {
            return Err("LU preconditioner must be nonsingular".into());
        }
        let (p, inv) = f.P().arrays();
        Ok(Self {
            l: f.L().to_owned(),
            u: f.U().to_owned(),
            permutation: faer::perm::Perm::new_checked(
                p.to_vec().into_boxed_slice(),
                inv.to_vec().into_boxed_slice(),
                n,
            ),
        })
    }

    fn run(
        &self,
        mut out: MatMut<'_, T>,
        b: MatRef<'_, T>,
        p: Par,
        s: &mut MemStack,
        t: bool,
        c: bool,
    ) {
        out.copy_from(b);
        let conj = if c { faer::Conj::Yes } else { faer::Conj::No };
        use faer::linalg::lu::partial_pivoting::solve;
        if t {
            solve::solve_transpose_in_place_with_conj(
                self.l.as_ref(),
                self.u.as_ref(),
                self.permutation.as_ref(),
                conj,
                out,
                p,
                s,
            )
        } else {
            solve::solve_in_place_with_conj(
                self.l.as_ref(),
                self.u.as_ref(),
                self.permutation.as_ref(),
                conj,
                out,
                p,
                s,
            )
        }
    }
}
impl<T: ComplexField> LinOp<T> for InverseLu<T> {
    fn nrows(&self) -> usize {
        self.l.as_ref().nrows()
    }
    fn ncols(&self) -> usize {
        self.u.as_ref().ncols()
    }
    fn apply_scratch(&self, k: usize, p: ParShape) -> StackReq {
        faer::linalg::lu::partial_pivoting::solve::solve_in_place_scratch::<usize, T>(
            self.nrows(),
            k,
            p,
        )
    }
    fn apply(&self, o: MatMut<'_, T>, b: MatRef<'_, T>, p: Par, s: &mut MemStack) {
        self.run(o, b, p, s, false, false)
    }
    fn conj_apply(&self, o: MatMut<'_, T>, b: MatRef<'_, T>, p: Par, s: &mut MemStack) {
        self.run(o, b, p, s, false, true)
    }
}
impl<T: ComplexField> BiLinOp<T> for InverseLu<T> {
    fn transpose_apply_scratch(&self, k: usize, p: ParShape) -> StackReq {
        faer::linalg::lu::partial_pivoting::solve::solve_transpose_in_place_scratch::<usize, T>(
            self.nrows(),
            k,
            p,
        )
    }
    fn transpose_apply(&self, o: MatMut<'_, T>, b: MatRef<'_, T>, p: Par, s: &mut MemStack) {
        self.run(o, b, p, s, true, false)
    }
    fn adjoint_apply(&self, o: MatMut<'_, T>, b: MatRef<'_, T>, p: Par, s: &mut MemStack) {
        self.run(o, b, p, s, true, true)
    }
}
#[julia]
impl OwnedLinearOperator {
    #[julia]
    pub fn operator_lu_preconditioner(
        &mut self,
        factor: &crate::OwnedFactor,
    ) -> Result<(), String> {
        guarded(|| {
            let crate::DenseFactor::Lu(f) = factor.get()? else {
                return Err("preconditioner requires partial-pivot LU".into());
            };
            if f.L().nrows() != f.U().ncols() {
                return Err("preconditioner requires square LU".into());
            }
            self.data = Some(Data::F64(Operator::new(Base::InverseLu(Box::new(
                InverseLu::new(f)?,
            )))));
            Ok(())
        })
    }
}
#[julia]
impl OwnedLinearOperator {
    #[julia]
    pub fn operator_typed_lu_preconditioner(
        &mut self,
        factor: &crate::typed::TypedFactor,
    ) -> Result<(), String> {
        guarded(|| {
            use crate::typed::{Factor, FactorData};
            fn make<T: ComplexField>(
                f: &faer::linalg::solvers::PartialPivLu<T>,
            ) -> Result<Operator<T>, String> {
                if f.L().nrows() != f.U().ncols() {
                    return Err("preconditioner requires square LU".into());
                }
                Ok(Operator::new(Base::InverseLu(Box::new(InverseLu::new(f)?))))
            }
            self.data = Some(match factor.data()? {
                FactorData::F32(Factor::Lu(f)) => Data::F32(make(f)?),
                FactorData::F64(Factor::Lu(f)) => Data::F64(make(f)?),
                FactorData::C32(Factor::Lu(f)) => Data::C32(make(f)?),
                FactorData::C64(Factor::Lu(f)) => Data::C64(make(f)?),
                _ => return Err("preconditioner requires partial-pivot LU".into()),
            });
            Ok(())
        })
    }
}
