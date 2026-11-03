//! Owned native LinOp/BiLinOp/Precond adapters and conjugate gradient.
use crate::collections::{VectorData, VectorOwner, VectorStore};
use crate::sparse::{SparseData, SparseOwner, Storage};
use crate::typed::{MatrixData, TypedMatrix};
use crate::{guarded, OwnedMatrix};
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
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
        source: *const u8,
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
            self.data = Some(match unsafe { &*source.cast::<Self>() }.data()? {
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
    pub fn operator_product(&mut self, operator: *const u8, rhs: *const u8) -> Result<(), String> {
        guarded(|| {
            let b = unsafe { &*rhs.cast::<OwnedMatrix>() };
            match unsafe { &*operator.cast::<OwnedLinearOperator>() }.data()? {
                Data::F64(a) => {
                    self.matrix = product(a, b.matrix.as_ref())?;
                    Ok(())
                }
                _ => Err("operator scalar types must match".into()),
            }
        })
    }
    #[julia]
    pub fn operator_dense_from(&mut self, operator: *const u8) -> Result<(), String> {
        guarded(
            || match unsafe { &*operator.cast::<OwnedLinearOperator>() }.data()? {
                Data::F64(Operator {
                    base: Base::Dense(a),
                    ..
                }) => {
                    self.matrix = a.clone();
                    Ok(())
                }
                _ => Err("operator has no matching dense source".into()),
            },
        )
    }
    #[julia]
    pub fn operator_precondition(&mut self, operator: *const u8) -> Result<(), String> {
        guarded(
            || match unsafe { &*operator.cast::<OwnedLinearOperator>() }.data()? {
                Data::F64(a) => precondition(a, self.matrix.as_mut()),
                _ => Err("operator scalar types must match".into()),
            },
        )
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn operator_product(&mut self, operator: *const u8, rhs: *const u8) -> Result<(), String> {
        guarded(|| {
            self.data = Some(
                match (
                    unsafe { &*operator.cast::<OwnedLinearOperator>() }.data()?,
                    unsafe { &*rhs.cast::<TypedMatrix>() }.data()?,
                ) {
                    (Data::F32(a), MatrixData::F32(b)) => MatrixData::F32(product(a, b.as_ref())?),
                    (Data::F64(a), MatrixData::F64(b)) => MatrixData::F64(product(a, b.as_ref())?),
                    (Data::C32(a), MatrixData::C32(b)) => MatrixData::C32(product(a, b.as_ref())?),
                    (Data::C64(a), MatrixData::C64(b)) => MatrixData::C64(product(a, b.as_ref())?),
                    _ => return Err("operator scalar types must match".into()),
                },
            );
            Ok(())
        })
    }
    #[julia]
    pub fn operator_dense_from(&mut self, operator: *const u8) -> Result<(), String> {
        guarded(|| {
            self.data = Some(
                match unsafe { &*operator.cast::<OwnedLinearOperator>() }.data()? {
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
                },
            );
            Ok(())
        })
    }
    #[julia]
    pub fn operator_precondition(&mut self, operator: *const u8) -> Result<(), String> {
        guarded(|| {
            match (
                unsafe { &*operator.cast::<OwnedLinearOperator>() }.data()?,
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
    pub fn operator_sparse_from(&mut self, operator: *const u8) -> Result<(), String> {
        guarded(|| {
            self.data = Some(
                match unsafe { &*operator.cast::<OwnedLinearOperator>() }.data()? {
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
                },
            );
            Ok(())
        })
    }
}
#[julia]
impl VectorOwner {
    #[julia]
    pub fn operator_diagonal_from(&mut self, operator: *const u8) -> Result<(), String> {
        guarded(|| {
            self.data = Some(
                match unsafe { &*operator.cast::<OwnedLinearOperator>() }.data()? {
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
                },
            );
            Ok(())
        })
    }
    #[julia]
    pub fn operator_precondition(&mut self, operator: *const u8) -> Result<(), String> {
        guarded(|| {
            match (
                unsafe { &*operator.cast::<OwnedLinearOperator>() }.data()?,
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
        operator: *const u8,
        precond: *const u8,
        rhs: *const u8,
        initial: *const u8,
        atol: f64,
        rtol: f64,
        maxiter: usize,
        stats: *mut u8,
        iterations: *mut usize,
    ) -> Result<(), String> {
        guarded(|| {
            let a = unsafe { &*operator.cast::<OwnedLinearOperator>() }.data()?;
            let m = if precond.is_null() {
                None
            } else {
                Some(unsafe { &*precond.cast::<OwnedLinearOperator>() }.data()?)
            };
            let b = unsafe { &*rhs.cast::<OwnedMatrix>() };
            let x = unsafe { &*initial.cast::<OwnedMatrix>() };
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
        operator: *const u8,
        precond: *const u8,
        rhs: *const u8,
        initial: *const u8,
        atol: f64,
        rtol: f64,
        maxiter: usize,
        stats: *mut u8,
        iterations: *mut usize,
    ) -> Result<(), String> {
        guarded(|| {
            let a = unsafe { &*operator.cast::<OwnedLinearOperator>() }.data()?;
            let m = if precond.is_null() {
                None
            } else {
                Some(unsafe { &*precond.cast::<OwnedLinearOperator>() }.data()?)
            };
            let b = unsafe { &*rhs.cast::<TypedMatrix>() }.data()?;
            let x = unsafe { &*initial.cast::<TypedMatrix>() }.data()?;
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
