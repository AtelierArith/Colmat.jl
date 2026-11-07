//! Owned native LinOp/BiLinOp/Precond adapters and conjugate gradient.
use crate::teams::with_team;
use crate::collections::{VectorData, VectorOwner, VectorStore};
use crate::sparse::{SparseData, SparseOwner, Storage};
use crate::typed::{MatrixData, TypedMatrix};
use crate::sparse_factors::SparseFactor;
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
mod julia_callback;
mod qr_actions;
mod qr_pseudoinverse;
mod sparse_qr_q;
mod iterative_workspace;
mod partial_workspace;
use qr_actions::{OwnedQrInverse, WideQrInverse};
#[derive(Clone)]
enum Base<T> {
    QrPseudo(Box<qr_pseudoinverse::QrPseudoInverse<T>>),
    Callback(julia_callback::JuliaCallback),
    Dense(Mat<T>),
    Sparse(Storage<T>),
    Diagonal(Diag<T>),
    Identity(IdentityPrecond),
    Product(Box<OperatorProduct<T>>),
    InverseFactors(Box<InverseFactors<T>>),
    InverseQr(Box<InverseQr<T>>),
    InverseLdlt(Box<InverseLdlt<T>>),
    InverseSpectral(Box<InverseSpectral<T>>),
    InverseSparseFactors(Box<InverseSparseFactors<T>>),
    InverseSparseQr(Box<crate::sparse_qr_components::Factors<T>>),
    InverseOwnedSparseQr(Box<OwnedQrInverse<T>>),
    InverseWideSparseQr(Box<WideQrInverse<T>>),
    SparseQrQ(Box<sparse_qr_q::SparseQrQ<T>>),
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
            Base::QrPseudo($a) => {
                let $a = $a.as_ref();
                $expr
            }
            Base::Callback($a) => $expr,
            Base::Dense($a) => $expr,
            Base::Sparse(Storage::Csc($a)) => $expr,
            Base::Sparse(Storage::Csr($a)) => $expr,
            Base::Diagonal($a) => $expr,
            Base::Identity($a) => $expr,
            Base::InverseWideSparseQr($a) => {
                let $a = $a.as_ref();
                $expr
            }
            Base::InverseOwnedSparseQr($a) => {
                let $a = $a.as_ref();
                $expr
            }
            Base::SparseQrQ($a) => {
                let $a = $a.as_ref();
                $expr
            }
            Base::InverseSparseQr($a) => {
                let $a = $a.as_ref();
                $expr
            }
            Base::InverseSparseFactors($a) => {
                let $a = $a.as_ref();
                $expr
            }
            Base::InverseSpectral($a) => {
                let $a = $a.as_ref();
                $expr
            }
            Base::InverseLdlt($a) => {
                let $a = $a.as_ref();
                $expr
            }
            Base::InverseQr($a) => {
                let $a = $a.as_ref();
                $expr
            }
            Base::InverseFactors($a) => {
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
fn product_checked<T: ComplexField>(
    a: &Operator<T>,
    b: MatRef<'_, T>,
    m: usize,
    n: usize,
    columns: usize,
) -> Result<Mat<T>, String> {
    if (a.nrows(), a.ncols()) != (m, n) || b.shape() != (n, columns) {
        return Err("operator product native dimensions do not match requested metadata".into());
    }
    product(a, b)
}
// FFI leaf for an observer's independent Julia-owned column-major snapshot.
// Nonfinite failed iterates must reach the same operator path as native output,
// without going through the finite-only public matrix constructor.
fn observer_product<T: ComplexField>(
    a: &Operator<T>,
    buffer: *const u8,
    rows: usize,
    cols: usize,
) -> Result<Mat<T>, String> {
    if rows != a.ncols() {
        return Err("observer product dimensions must match".into());
    }
    let bytes = rows
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(core::mem::size_of::<T>()))
        .ok_or("observer buffer size overflow")?;
    if bytes > isize::MAX as usize || rows > isize::MAX as usize {
        return Err("observer buffer exceeds addressable range".into());
    }
    if bytes == 0 {
        return product(a, Mat::<T>::zeros(rows, cols).as_ref());
    }
    if buffer.is_null() || (buffer as usize) % core::mem::align_of::<T>() != 0 {
        return Err("observer buffer must be nonnull and aligned".into());
    }
    // SAFETY: _iteration_product passes a GC-preserved Matrix<T> with exactly
    // rows*cols live elements, scalar type checked against a, and column stride
    // rows. Dimensions, byte span and alignment are checked above. No pointer or
    // borrowed view is retained after the synchronous FFI call.
    let b = unsafe { MatRef::from_raw_parts(buffer.cast::<T>(), rows, cols, 1, rows as isize) };
    product(a, b)
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn operator_observer_product(
        &mut self,
        operator: &OwnedLinearOperator,
        buffer: *const u8,
        rows: usize,
        cols: usize,
    ) -> Result<(), String> {
        guarded(|| match operator.data()? {
            Data::F64(a) => {
                self.matrix = observer_product(a, buffer, rows, cols)?;
                Ok(())
            }
            _ => Err("observer scalar types must match".into()),
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn operator_observer_product(
        &mut self,
        operator: &OwnedLinearOperator,
        buffer: *const u8,
        rows: usize,
        cols: usize,
    ) -> Result<(), String> {
        guarded(|| {
            self.data = Some(match operator.data()? {
                Data::F32(a) => MatrixData::F32(observer_product(a, buffer, rows, cols)?),
                Data::F64(a) => MatrixData::F64(observer_product(a, buffer, rows, cols)?),
                Data::C32(a) => MatrixData::C32(observer_product(a, buffer, rows, cols)?),
                Data::C64(a) => MatrixData::C64(observer_product(a, buffer, rows, cols)?),
            });
            Ok(())
        })
    }
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
        m: usize,
        n: usize,
        columns: usize,
        tag: i32,
    ) -> Result<(), String> {
        guarded(|| {
            apply_workspace::validate(operator, m, n, tag)?;
            let b = rhs;
            match operator.data()? {
                Data::F64(a) => {
                    self.matrix = product_checked(a, b.matrix.as_ref(), m, n, columns)?;
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
        m: usize,
        n: usize,
        columns: usize,
        tag: i32,
    ) -> Result<(), String> {
        guarded(|| {
            apply_workspace::validate(operator, m, n, tag)?;
            self.data = Some(match (operator.data()?, rhs.data()?) {
                (Data::F32(a), MatrixData::F32(b)) => {
                    MatrixData::F32(product_checked(a, b.as_ref(), m, n, columns)?)
                }
                (Data::F64(a), MatrixData::F64(b)) => {
                    MatrixData::F64(product_checked(a, b.as_ref(), m, n, columns)?)
                }
                (Data::C32(a), MatrixData::C32(b)) => {
                    MatrixData::C32(product_checked(a, b.as_ref(), m, n, columns)?)
                }
                (Data::C64(a), MatrixData::C64(b)) => {
                    MatrixData::C64(product_checked(a, b.as_ref(), m, n, columns)?)
                }
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
// Synchronously borrowed on the calling Julia thread, outside worker kernels.
// The callback must copy values before returning; no pointer escapes.
fn observe_iteration<T: ComplexField>(
    x: MatRef<'_, T>,
    observing: bool,
    observer: extern "C" fn(*const u8, usize, usize, isize, isize) -> bool,
) -> core::ops::ControlFlow<()> {
    if observing
        && !observer(
            x.as_ptr().cast(),
            x.nrows(),
            x.ncols(),
            x.row_stride(),
            x.col_stride(),
        )
    {
        core::ops::ControlFlow::Break(())
    } else {
        core::ops::ControlFlow::Continue(())
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
    observing: bool,
    observer: extern "C" fn(*const u8, usize, usize, isize, isize) -> bool,
) -> Result<Mat<T>, String> {
    solve_resources(a, m, b, initial, atol, rtol, maxiter, stats, iterations, observing, observer, None, Par::Seq)
}
fn solve_resources<T: ComplexField>(
    a: &Operator<T>,
    m: Option<&Operator<T>>,
    b: MatRef<'_, T>,
    initial: MatRef<'_, T>,
    atol: f64,
    rtol: f64,
    maxiter: usize,
    stats: *mut u8,
    iterations: *mut usize,
    observing: bool,
    observer: extern "C" fn(*const u8, usize, usize, isize, isize) -> bool,
    scratch: Option<&mut crate::workspace::OwnedScratch>,
    mut par: Par<'_>,
) -> Result<Mat<T>, String> {
    if a.nrows() != a.ncols() || a.ncols() != b.nrows() || initial.shape() != b.shape() {
        return Err("CG dimensions must match".into());
    }
    let identity = Operator::<T>::new(Base::Identity(IdentityPrecond { dim: a.nrows() }));
    let m = m.unwrap_or(&identity);
    if m.shape() != a.shape() {
        return Err("CG preconditioner dimensions must match".into());
    }
    let req=iterative_workspace::requirement(a,m,m,0,b.ncols(),par.shape())?;
    let mut memory=if scratch.is_none(){Some(MemBuffer::try_new(req).map_err(|e|format!("{e:?}"))?)}else{None};
    let stack=if let Some(work)=scratch{work.stack(req)?}else{MemStack::new(memory.as_mut().unwrap())};
    let mut out = initial.to_owned();
    let mut callbacks = 0usize;
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
        |x| {
            callbacks += 1;
            observe_iteration(x, observing, observer)
        },
        par.reborrow(),
        stack,
    );
    let (mut status, estimate_abs, estimate_rel, count) = match result {
        Ok(info) => (
            if info.stopped { 5 } else { 0 },
            info.abs_residual,
            info.rel_residual,
            info.iter_count,
        ),
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
    let (actual,_normal)=iterative_workspace::residual_norms(a,m,out.as_ref(),b,0,par,stack);
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
        iterations: &mut [usize],
        observing: bool,
        observer: extern "C" fn(*const u8, usize, usize, isize, isize) -> bool,
    ) -> Result<(), String> {
        let iterations = crate::workspace::fixed_out(iterations, 1)?;
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
                        observing,
                        observer,
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
                        observing,
                        observer,
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
        iterations: &mut [usize],
        observing: bool,
        observer: extern "C" fn(*const u8, usize, usize, isize, isize) -> bool,
    ) -> Result<(), String> {
        let iterations = crate::workspace::fixed_out(iterations, 1)?;
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
                        observing,
                        observer,
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
                        observing,
                        observer,
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
                        observing,
                        observer,
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
                        observing,
                        observer,
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
                        observing,
                        observer,
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
                        observing,
                        observer,
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
    observing: bool,
    observer: extern "C" fn(*const u8, usize, usize, isize, isize) -> bool,
) -> Result<Mat<T>, String> {
    solve_bicg_resources(a, m, right, b, initial, atol, rtol, maxiter, stats, iterations, observing, observer, None, Par::Seq)
}
fn solve_bicg_resources<T: ComplexField>(
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
    observing: bool,
    observer: extern "C" fn(*const u8, usize, usize, isize, isize) -> bool,
    scratch: Option<&mut crate::workspace::OwnedScratch>,
    mut par: Par<'_>,
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
    let req=iterative_workspace::requirement(a,m,right,1,b.ncols(),par.shape())?;
    let mut memory=if scratch.is_none(){Some(MemBuffer::try_new(req).map_err(|e|format!("{e:?}"))?)}else{None};
    let stack=if let Some(work)=scratch{work.stack(req)?}else{MemStack::new(memory.as_mut().unwrap())};
    let mut out = initial.to_owned();
    let mut callbacks = 0usize;
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
        |x| {
            callbacks += 1;
            observe_iteration(x, observing, observer)
        },
        par.reborrow(),
        stack,
    );
    let (mut status, estimate_abs, estimate_rel, count) = match result {
        Ok(info) => (
            if info.stopped { 5 } else { 0 },
            info.abs_residual,
            info.rel_residual,
            info.iter_count,
        ),
        Err(faer::matrix_free::bicgstab::BicgError::NoConvergence {
            abs_residual,
            rel_residual,
        }) => (1, abs_residual, rel_residual, callbacks),
    };
    let (actual,_normal)=iterative_workspace::residual_norms(a,m,out.as_ref(),b,1,par,stack);
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
        iterations: &mut [usize],
        observing: bool,
        observer: extern "C" fn(*const u8, usize, usize, isize, isize) -> bool,
    ) -> Result<(), String> {
        let iterations = crate::workspace::fixed_out(iterations, 1)?;
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
                observing,
                observer,
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
        iterations: &mut [usize],
        observing: bool,
        observer: extern "C" fn(*const u8, usize, usize, isize, isize) -> bool,
    ) -> Result<(), String> {
        let iterations = crate::workspace::fixed_out(iterations, 1)?;
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
                        observing,
                        observer,
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
                        observing,
                        observer,
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
                        observing,
                        observer,
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
    observing: bool,
    observer: extern "C" fn(*const u8, usize, usize, isize, isize) -> bool,
) -> Result<Mat<T>, String> {
    solve_lsmr_resources(a, m, b, initial, atol, rtol, maxiter, stats, iterations, observing, observer, None, Par::Seq)
}
fn solve_lsmr_resources<T: ComplexField>(
    a: &Operator<T>,
    m: Option<&Operator<T>>,
    b: MatRef<'_, T>,
    initial: MatRef<'_, T>,
    atol: f64,
    rtol: f64,
    maxiter: usize,
    stats: *mut u8,
    iterations: *mut usize,
    observing: bool,
    observer: extern "C" fn(*const u8, usize, usize, isize, isize) -> bool,
    scratch: Option<&mut crate::workspace::OwnedScratch>,
    mut par: Par<'_>,
) -> Result<Mat<T>, String> {
    if a.nrows() != b.nrows() || initial.nrows() != a.ncols() || initial.ncols() != b.ncols() {
        return Err("LSMR dimensions must match".into());
    }
    let identity = Operator::<T>::new(Base::Identity(IdentityPrecond { dim: a.ncols() }));
    let m = m.unwrap_or(&identity);
    if m.shape() != (a.ncols(), a.ncols()) {
        return Err("LSMR preconditioner dimensions must match".into());
    }
    let req=iterative_workspace::requirement(a,m,m,2,b.ncols(),par.shape())?;
    let mut memory=if scratch.is_none(){Some(MemBuffer::try_new(req).map_err(|e|format!("{e:?}"))?)}else{None};
    let stack=if let Some(work)=scratch{work.stack(req)?}else{MemStack::new(memory.as_mut().unwrap())};
    let normal_ref=iterative_workspace::normal_norm(a,m,b,par.reborrow(),stack);
    let abs_tol = from_f64::<T::Real>(atol);
    let requested_rel = from_f64::<T::Real>(rtol);
    let mut out = initial.to_owned();
    let mut callbacks = 0usize;
    let (mut status, estimate_abs, estimate_rel, count) =
        if a.nrows() == 0 || a.ncols() == 0 || b.ncols() == 0 {
            out.fill(zero());
            (0, zero::<T::Real>(), zero::<T::Real>(), 0)
        } else {
            let result = faer::matrix_free::lsmr::lsmr(
                out.as_mut(),
                m,
                a,
                b,
                faer::matrix_free::lsmr::LsmrParams {
                    initial_guess: InitialGuessStatus::MaybeNonZero,
                    abs_tolerance: copy(&abs_tol),
                    rel_tolerance: copy(&requested_rel),
                    max_iters: maxiter,
                    ..Default::default()
                },
                |x| {
                    callbacks += 1;
                    observe_iteration(x, observing, observer)
                },
                par.reborrow(),
                stack,
            );
            match result {
                Ok(info) => (
                    if info.stopped { 5 } else { 0 },
                    info.abs_residual,
                    info.rel_residual,
                    info.iter_count,
                ),
                Err(faer::matrix_free::lsmr::LsmrError::NoConvergence {
                    abs_residual,
                    rel_residual,
                }) => (1, abs_residual, rel_residual, callbacks),
            }
        };
    let (actual,normal)=iterative_workspace::residual_norms(a,m,out.as_ref(),b,2,par,stack);
    let bnorm = b.norm_l2();
    let relative = if bnorm == zero::<T::Real>() {
        copy(&actual)
    } else {
        &actual / &bnorm
    };
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
        iterations: &mut [usize],
        observing: bool,
        observer: extern "C" fn(*const u8, usize, usize, isize, isize) -> bool,
    ) -> Result<(), String> {
        let iterations = crate::workspace::fixed_out(iterations, 1)?;
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
                        observing,
                        observer,
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
                        observing,
                        observer,
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
        iterations: &mut [usize],
        observing: bool,
        observer: extern "C" fn(*const u8, usize, usize, isize, isize) -> bool,
    ) -> Result<(), String> {
        let iterations = crate::workspace::fixed_out(iterations, 1)?;
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
                        observing,
                        observer,
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
                        observing,
                        observer,
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
                        observing,
                        observer,
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
                        observing,
                        observer,
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
                        observing,
                        observer,
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
                        observing,
                        observer,
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

pub(crate) trait NativeValue: ComplexField {
    fn tag() -> i32;
    fn parts(&self) -> (f64, f64);
    fn from_parts(re: f64, im: f64) -> Self;
}
impl NativeValue for f32 {
    fn tag() -> i32 {
        0
    }
    fn parts(&self) -> (f64, f64) {
        (*self as f64, 0.0)
    }
    fn from_parts(r: f64, _: f64) -> Self {
        r as f32
    }
}
impl NativeValue for f64 {
    fn tag() -> i32 {
        1
    }
    fn parts(&self) -> (f64, f64) {
        (*self, 0.0)
    }
    fn from_parts(r: f64, _: f64) -> Self {
        r
    }
}
impl NativeValue for c32 {
    fn tag() -> i32 {
        2
    }
    fn parts(&self) -> (f64, f64) {
        (self.re as f64, self.im as f64)
    }
    fn from_parts(r: f64, i: f64) -> Self {
        Self::new(r as f32, i as f32)
    }
}
impl NativeValue for c64 {
    fn tag() -> i32 {
        3
    }
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
        Base::QrPseudo(f) => Base::QrPseudo(Box::new(f.map_values(cast))),
        Base::Callback(f) => Base::Callback(julia_callback::JuliaCallback {
            tag: T::tag(),
            ..*f
        }),
        Base::InverseWideSparseQr(f) => Base::InverseWideSparseQr(Box::new(f.map_values(cast)?)),
        Base::InverseOwnedSparseQr(f) => Base::InverseOwnedSparseQr(Box::new(f.map_values(cast)?)),
        Base::SparseQrQ(f) => Base::SparseQrQ(Box::new(f.map_values(cast))),
        Base::InverseSparseQr(f) => {
            // Preserve native Householder pivot ordering; do not sort triplets.
            let convert=|a:&faer::sparse::SparseColMat<usize,S>|->Result<faer::sparse::SparseColMat<usize,T>,String> {
                let symbolic=a.symbolic().to_owned().map_err(|e|format!("{e:?}"))?;
                Ok(faer::sparse::SparseColMat::new(symbolic,a.val().iter().map(cast).collect()))
            };
            Base::InverseSparseQr(Box::new(crate::sparse_qr_components::Factors {
                plan: f.plan.clone(),
                r: convert(&f.r)?,
                h: convert(&f.h)?,
                tau: f.tau.iter().map(cast).collect(),
            }))
        }
        Base::InverseSparseFactors(f) => {
            let convert=|a:&faer::sparse::SparseColMat<usize,S>|->Result<faer::sparse::SparseColMat<usize,T>,String> {
                let entries:Vec<_>=a.as_ref().triplet_iter().map(|e|faer::sparse::Triplet{row:e.row,col:e.col,val:cast(e.val)}).collect();
                faer::sparse::SparseColMat::try_new_from_triplets(a.nrows(),a.ncols(),&entries).map_err(|e|format!("{e:?}"))
            };
            Base::InverseSparseFactors(Box::new(InverseSparseFactors {
                unit_lower: f.unit_lower,
                l: convert(&f.l)?,
                u: convert(&f.u)?,
                rows: f.rows.clone(),
                cols: f.cols.clone(),
            }))
        }
        Base::InverseSpectral(f) => Base::InverseSpectral(Box::new(InverseSpectral {
            u: Mat::from_fn(f.u.nrows(), f.u.ncols(), |i, j| cast(&f.u[(i, j)])),
            v: Mat::from_fn(f.v.nrows(), f.v.ncols(), |i, j| cast(&f.v[(i, j)])),
            weights: f.weights.iter().map(cast).collect(),
            zero: f.zero,
        })),
        Base::InverseLdlt(f) => {
            let mut d = Diag::zeros(f.diagonal.dim());
            for i in 0..d.dim() {
                d.column_vector_mut()[i] = cast(&f.diagonal.column_vector()[i]);
            }
            Base::InverseLdlt(Box::new(InverseLdlt {
                l: Mat::from_fn(f.l.nrows(), f.l.ncols(), |i, j| cast(&f.l[(i, j)])),
                diagonal: d,
                block: f.block.as_ref().map(|(d, p)| {
                    let mut out = Diag::zeros(d.dim());
                    for i in 0..d.dim() {
                        out.column_vector_mut()[i] = cast(&d.column_vector()[i]);
                    }
                    (out, p.clone())
                }),
            }))
        }
        Base::InverseQr(f) => Base::InverseQr(Box::new(InverseQr {
            basis: Mat::from_fn(f.basis.nrows(), f.basis.ncols(), |i, j| {
                cast(&f.basis[(i, j)])
            }),
            coeff: Mat::from_fn(f.coeff.nrows(), f.coeff.ncols(), |i, j| {
                cast(&f.coeff[(i, j)])
            }),
            r: Mat::from_fn(f.r.nrows(), f.r.ncols(), |i, j| cast(&f.r[(i, j)])),
            permutation: f.permutation.clone(),
        })),
        Base::InverseFactors(f) => Base::InverseFactors(Box::new(InverseFactors {
            l: Mat::from_fn(f.l.nrows(), f.l.ncols(), |i, j| cast(&f.l[(i, j)])),
            u: Mat::from_fn(f.u.nrows(), f.u.ncols(), |i, j| cast(&f.u[(i, j)])),
            permutation: f.permutation.clone(),
            unit_lower: f.unit_lower,
            column_permutation: f.column_permutation.clone(),
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
    pub fn operator_promote(
        &mut self,
        source: &Self,
        tag: i32,
        m: usize,
        n: usize,
        source_tag: i32,
    ) -> Result<(), String> {
        guarded(|| {
            apply_workspace::validate(source, m, n, source_tag)?;
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
struct InverseFactors<T> {
    l: Mat<T>,
    u: Mat<T>,
    permutation: faer::perm::Perm<usize>,
    unit_lower: bool,
    column_permutation: Option<faer::perm::Perm<usize>>,
}
impl<T> std::fmt::Debug for InverseFactors<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("InverseFactors")
    }
}
impl<T: ComplexField> InverseFactors<T> {
    fn full_lu(f: &faer::linalg::solvers::FullPivLu<T>) -> Result<Self, String> {
        let n = f.L().nrows();
        if n != f.U().ncols() {
            return Err("LU preconditioner must be square".into());
        }
        if (0..n).any(|i| abs(&f.U()[(i, i)]) == zero::<T::Real>()) {
            return Err("LU preconditioner must be nonsingular".into());
        }
        let own = |p: faer::perm::PermRef<'_, usize>| {
            let (a, b) = p.arrays();
            faer::perm::Perm::new_checked(
                a.to_vec().into_boxed_slice(),
                b.to_vec().into_boxed_slice(),
                n,
            )
        };
        Ok(Self {
            l: f.L().to_owned(),
            u: f.U().to_owned(),
            permutation: own(f.P()),
            column_permutation: Some(own(f.Q())),
            unit_lower: true,
        })
    }

    fn cholesky(f: &crate::cholesky_factors::Llt<T>) -> Result<Self, String> {
        let n = f.L().nrows();
        if (0..n).any(|i| abs(&f.L()[(i, i)]) == zero::<T::Real>()) {
            return Err("Cholesky preconditioner must be nonsingular".into());
        }
        let indices: Vec<usize> = (0..n).collect();
        Ok(Self {
            l: f.L().to_owned(),
            u: f.L().adjoint().to_owned(),
            column_permutation: None,
            unit_lower: false,
            permutation: faer::perm::Perm::new_checked(
                indices.clone().into_boxed_slice(),
                indices.into_boxed_slice(),
                n,
            ),
        })
    }

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
            column_permutation: None,
            unit_lower: true,
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
        mut p: Par,
        s: &mut MemStack,
        t: bool,
        c: bool,
    ) {
        out.copy_from(b);
        let conj = if c { faer::Conj::Yes } else { faer::Conj::No };
        if !self.unit_lower {
            use faer::linalg::triangular_solve::{
                solve_lower_triangular_in_place_with_conj as lower,
                solve_upper_triangular_in_place_with_conj as upper,
            };
            if t {
                lower(
                    self.u.as_ref().transpose(),
                    conj,
                    out.as_mut(),
                    p.reborrow(),
                );
                upper(self.l.as_ref().transpose(), conj, out, p);
            } else {
                lower(self.l.as_ref(), conj, out.as_mut(), p.reborrow());
                upper(self.u.as_ref(), conj, out, p);
            }
            return;
        }
        if let Some(q) = &self.column_permutation {
            use faer::linalg::lu::full_pivoting::solve;
            if t {
                solve::solve_transpose_in_place_with_conj(
                    self.l.as_ref(),
                    self.u.as_ref(),
                    self.permutation.as_ref(),
                    q.as_ref(),
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
                    q.as_ref(),
                    conj,
                    out,
                    p,
                    s,
                )
            }
            return;
        }
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
impl<T: ComplexField> LinOp<T> for InverseFactors<T> {
    fn nrows(&self) -> usize {
        self.l.as_ref().nrows()
    }
    fn ncols(&self) -> usize {
        self.u.as_ref().ncols()
    }
    fn apply_scratch(&self, k: usize, p: ParShape) -> StackReq {
        if !self.unit_lower {
            return StackReq::EMPTY;
        }
        if self.column_permutation.is_some() {
            return faer::linalg::lu::full_pivoting::solve::solve_in_place_scratch::<usize, T>(
                self.nrows(),
                k,
                p,
            );
        }
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
impl<T: ComplexField> BiLinOp<T> for InverseFactors<T> {
    fn transpose_apply_scratch(&self, k: usize, p: ParShape) -> StackReq {
        if !self.unit_lower {
            return StackReq::EMPTY;
        }
        if self.column_permutation.is_some() {
            return faer::linalg::lu::full_pivoting::solve::solve_transpose_in_place_scratch::<
                usize,
                T,
            >(self.nrows(), k, p);
        }
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
    pub fn operator_factor_preconditioner(
        &mut self,
        factor: &crate::OwnedFactor,
    ) -> Result<(), String> {
        guarded(|| {
            match factor.get()? {
                crate::DenseFactor::Lblt(f) => {
                    self.data = Some(Data::F64(Operator::new(Base::InverseLdlt(Box::new(
                        InverseLdlt::block(f)?,
                    )))));
                    return Ok(());
                }
                crate::DenseFactor::Ldlt(f) => {
                    self.data = Some(Data::F64(Operator::new(Base::InverseLdlt(Box::new(
                        InverseLdlt::new(f.L(), f.D())?,
                    )))));
                    return Ok(());
                }
                crate::DenseFactor::Qr(f) => {
                    self.data = Some(Data::F64(Operator::new(Base::InverseQr(Box::new(
                        InverseQr::new(f.Q_basis(), f.Q_coeff(), f.R(), None)?,
                    )))));
                    return Ok(());
                }
                crate::DenseFactor::PivotQr(f) => {
                    self.data = Some(Data::F64(Operator::new(Base::InverseQr(Box::new(
                        InverseQr::new(f.Q_basis(), f.Q_coeff(), f.R(), Some(f.P()))?,
                    )))));
                    return Ok(());
                }
                _ => (),
            }
            let saved = match factor.get()? {
                crate::DenseFactor::Lu(f) => InverseFactors::new(f)?,
                crate::DenseFactor::FullLu(f) => InverseFactors::full_lu(f)?,
                crate::DenseFactor::Llt(f) => InverseFactors::cholesky(f)?,
                _ => return Err("preconditioner requires LU or Cholesky".into()),
            };
            self.data = Some(Data::F64(Operator::new(Base::InverseFactors(Box::new(
                saved,
            )))));
            Ok(())
        })
    }
}
#[julia]
impl OwnedLinearOperator {
    #[julia]
    pub fn operator_typed_factor_preconditioner(
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
                Ok(Operator::new(Base::InverseFactors(Box::new(
                    InverseFactors::new(f)?,
                ))))
            }
            self.data = Some(match factor.data()? {
                FactorData::F32(Factor::Lu(f)) => Data::F32(make(f)?),
                FactorData::F32(Factor::Lblt(f)) => Data::F32(Operator::new(Base::InverseLdlt(
                    Box::new(InverseLdlt::block(f)?),
                ))),
                FactorData::F32(Factor::Ldlt(f)) => Data::F32(Operator::new(Base::InverseLdlt(
                    Box::new(InverseLdlt::new(f.L(), f.D())?),
                ))),
                FactorData::F32(Factor::Qr(f)) => Data::F32(Operator::new(Base::InverseQr(
                    Box::new(InverseQr::new(f.Q_basis(), f.Q_coeff(), f.R(), None)?),
                ))),
                FactorData::F32(Factor::PivotQr(f)) => {
                    Data::F32(Operator::new(Base::InverseQr(Box::new(InverseQr::new(
                        f.Q_basis(),
                        f.Q_coeff(),
                        f.R(),
                        Some(f.P()),
                    )?))))
                }
                FactorData::F32(Factor::FullLu(f)) => Data::F32(Operator::new(
                    Base::InverseFactors(Box::new(InverseFactors::full_lu(f)?)),
                )),
                FactorData::F32(Factor::Llt(f)) => Data::F32(Operator::new(Base::InverseFactors(
                    Box::new(InverseFactors::cholesky(f)?),
                ))),
                FactorData::F64(Factor::Lu(f)) => Data::F64(make(f)?),
                FactorData::F64(Factor::Lblt(f)) => Data::F64(Operator::new(Base::InverseLdlt(
                    Box::new(InverseLdlt::block(f)?),
                ))),
                FactorData::F64(Factor::Ldlt(f)) => Data::F64(Operator::new(Base::InverseLdlt(
                    Box::new(InverseLdlt::new(f.L(), f.D())?),
                ))),
                FactorData::F64(Factor::Qr(f)) => Data::F64(Operator::new(Base::InverseQr(
                    Box::new(InverseQr::new(f.Q_basis(), f.Q_coeff(), f.R(), None)?),
                ))),
                FactorData::F64(Factor::PivotQr(f)) => {
                    Data::F64(Operator::new(Base::InverseQr(Box::new(InverseQr::new(
                        f.Q_basis(),
                        f.Q_coeff(),
                        f.R(),
                        Some(f.P()),
                    )?))))
                }
                FactorData::F64(Factor::FullLu(f)) => Data::F64(Operator::new(
                    Base::InverseFactors(Box::new(InverseFactors::full_lu(f)?)),
                )),
                FactorData::F64(Factor::Llt(f)) => Data::F64(Operator::new(Base::InverseFactors(
                    Box::new(InverseFactors::cholesky(f)?),
                ))),
                FactorData::C32(Factor::Lu(f)) => Data::C32(make(f)?),
                FactorData::C32(Factor::Lblt(f)) => Data::C32(Operator::new(Base::InverseLdlt(
                    Box::new(InverseLdlt::block(f)?),
                ))),
                FactorData::C32(Factor::Ldlt(f)) => Data::C32(Operator::new(Base::InverseLdlt(
                    Box::new(InverseLdlt::new(f.L(), f.D())?),
                ))),
                FactorData::C32(Factor::Qr(f)) => Data::C32(Operator::new(Base::InverseQr(
                    Box::new(InverseQr::new(f.Q_basis(), f.Q_coeff(), f.R(), None)?),
                ))),
                FactorData::C32(Factor::PivotQr(f)) => {
                    Data::C32(Operator::new(Base::InverseQr(Box::new(InverseQr::new(
                        f.Q_basis(),
                        f.Q_coeff(),
                        f.R(),
                        Some(f.P()),
                    )?))))
                }
                FactorData::C32(Factor::FullLu(f)) => Data::C32(Operator::new(
                    Base::InverseFactors(Box::new(InverseFactors::full_lu(f)?)),
                )),
                FactorData::C32(Factor::Llt(f)) => Data::C32(Operator::new(Base::InverseFactors(
                    Box::new(InverseFactors::cholesky(f)?),
                ))),
                FactorData::C64(Factor::Lu(f)) => Data::C64(make(f)?),
                FactorData::C64(Factor::Lblt(f)) => Data::C64(Operator::new(Base::InverseLdlt(
                    Box::new(InverseLdlt::block(f)?),
                ))),
                FactorData::C64(Factor::Ldlt(f)) => Data::C64(Operator::new(Base::InverseLdlt(
                    Box::new(InverseLdlt::new(f.L(), f.D())?),
                ))),
                FactorData::C64(Factor::Qr(f)) => Data::C64(Operator::new(Base::InverseQr(
                    Box::new(InverseQr::new(f.Q_basis(), f.Q_coeff(), f.R(), None)?),
                ))),
                FactorData::C64(Factor::PivotQr(f)) => {
                    Data::C64(Operator::new(Base::InverseQr(Box::new(InverseQr::new(
                        f.Q_basis(),
                        f.Q_coeff(),
                        f.R(),
                        Some(f.P()),
                    )?))))
                }
                FactorData::C64(Factor::FullLu(f)) => Data::C64(Operator::new(
                    Base::InverseFactors(Box::new(InverseFactors::full_lu(f)?)),
                )),
                FactorData::C64(Factor::Llt(f)) => Data::C64(Operator::new(Base::InverseFactors(
                    Box::new(InverseFactors::cholesky(f)?),
                ))),
                _ => return Err("preconditioner requires partial-pivot LU".into()),
            });
            Ok(())
        })
    }
}

#[derive(Clone)]
struct InverseQr<T> {
    basis: Mat<T>,
    coeff: Mat<T>,
    r: Mat<T>,
    permutation: faer::perm::Perm<usize>,
}
impl<T> std::fmt::Debug for InverseQr<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("InverseQr")
    }
}
impl<T: ComplexField> InverseQr<T> {
    fn new(
        b: MatRef<'_, T>,
        c: MatRef<'_, T>,
        r: MatRef<'_, T>,
        p: Option<faer::perm::PermRef<'_, usize>>,
    ) -> Result<Self, String> {
        let n = b.nrows();
        if b.ncols() != n || r.nrows() != n || r.ncols() != n {
            return Err("QR preconditioner requires square factors".into());
        }
        if (0..n).any(|i| abs(&r[(i, i)]) == zero::<T::Real>()) {
            return Err("QR preconditioner requires full rank".into());
        }
        let owned = if let Some(p) = p {
            let (a, b) = p.arrays();
            faer::perm::Perm::new_checked(
                a.to_vec().into_boxed_slice(),
                b.to_vec().into_boxed_slice(),
                n,
            )
        } else {
            let a: Vec<_> = (0..n).collect();
            faer::perm::Perm::new_checked(a.clone().into_boxed_slice(), a.into_boxed_slice(), n)
        };
        Ok(Self {
            basis: b.to_owned(),
            coeff: c.to_owned(),
            r: r.to_owned(),
            permutation: owned,
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
        if self.nrows() == 0 {
            return;
        }
        let conj = if c { faer::Conj::Yes } else { faer::Conj::No };
        use faer::linalg::qr::col_pivoting::solve;
        if t {
            solve::solve_transpose_in_place_with_conj(
                self.basis.as_ref(),
                self.coeff.as_ref(),
                self.r.as_ref(),
                self.permutation.as_ref(),
                conj,
                out,
                p,
                s,
            )
        } else {
            solve::solve_in_place_with_conj(
                self.basis.as_ref(),
                self.coeff.as_ref(),
                self.r.as_ref(),
                self.permutation.as_ref(),
                conj,
                out,
                p,
                s,
            )
        }
    }
}
impl<T: ComplexField> LinOp<T> for InverseQr<T> {
    fn nrows(&self) -> usize {
        self.r.nrows()
    }
    fn ncols(&self) -> usize {
        self.r.ncols()
    }
    fn apply_scratch(&self, k: usize, p: ParShape) -> StackReq {
        if self.nrows() == 0 {
            return StackReq::EMPTY;
        }
        faer::linalg::qr::col_pivoting::solve::solve_in_place_scratch::<usize, T>(
            self.nrows(),
            self.coeff.nrows(),
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
impl<T: ComplexField> BiLinOp<T> for InverseQr<T> {
    fn transpose_apply_scratch(&self, k: usize, p: ParShape) -> StackReq {
        if self.nrows() == 0 {
            return StackReq::EMPTY;
        }
        faer::linalg::qr::col_pivoting::solve::solve_transpose_in_place_scratch::<usize, T>(
            self.nrows(),
            self.coeff.nrows(),
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

#[derive(Clone)]
struct InverseLdlt<T> {
    l: Mat<T>,
    diagonal: Diag<T>,
    block: Option<(Diag<T>, faer::perm::Perm<usize>)>,
}
impl<T> std::fmt::Debug for InverseLdlt<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("InverseLdlt")
    }
}
impl<T: ComplexField> InverseLdlt<T> {
    fn block(f: &crate::lblt::Lblt<T>) -> Result<Self, String> {
        let n = f.L().nrows();
        let mut diagonal = Diag::zeros(n);
        let mut sub = Diag::zeros(n);
        for i in 0..n {
            diagonal.column_vector_mut()[i] = copy(&f.B_diag()[i]);
            sub.column_vector_mut()[i] = copy(&f.B_subdiag()[i]);
        }
        let (p, inv) = f.P().arrays();
        Ok(Self {
            l: f.L().to_owned(),
            diagonal,
            block: Some((
                sub,
                faer::perm::Perm::new_checked(
                    p.to_vec().into_boxed_slice(),
                    inv.to_vec().into_boxed_slice(),
                    n,
                ),
            )),
        })
    }

    fn new(l: MatRef<'_, T>, d: faer::diag::DiagRef<'_, T>) -> Result<Self, String> {
        if l.nrows() != l.ncols() || d.dim() != l.nrows() {
            return Err("LDLT preconditioner requires square factors".into());
        }
        let mut diagonal = Diag::zeros(d.dim());
        for i in 0..d.dim() {
            if abs(&d[i]) == zero::<T::Real>() {
                return Err("LDLT preconditioner requires nonzero pivots".into());
            }
            diagonal.column_vector_mut()[i] = copy(&d[i]);
        }
        Ok(Self {
            l: l.to_owned(),
            diagonal,
            block: None,
        })
    }
    fn run(&self, mut out: MatMut<'_, T>, b: MatRef<'_, T>, p: Par, s: &mut MemStack, c: bool) {
        out.copy_from(b);
        if let Some((sub, perm)) = &self.block {
            faer::linalg::cholesky::lblt::solve::solve_in_place_with_conj(
                self.l.as_ref(),
                self.diagonal.as_ref(),
                sub.as_ref(),
                if c { faer::Conj::Yes } else { faer::Conj::No },
                perm.as_ref(),
                out,
                p,
                s,
            );
            return;
        }
        faer::linalg::cholesky::ldlt::solve::solve_in_place_with_conj(
            self.l.as_ref(),
            self.diagonal.as_ref(),
            if c { faer::Conj::Yes } else { faer::Conj::No },
            out,
            p,
            s,
        );
    }
}
impl<T: ComplexField> LinOp<T> for InverseLdlt<T> {
    fn nrows(&self) -> usize {
        self.l.nrows()
    }
    fn ncols(&self) -> usize {
        self.l.ncols()
    }
    fn apply_scratch(&self, k: usize, p: ParShape) -> StackReq {
        if self.block.is_some() {
            return faer::linalg::cholesky::lblt::solve::solve_in_place_scratch::<usize, T>(
                self.nrows(),
                k,
                p,
            );
        }
        faer::linalg::cholesky::ldlt::solve::solve_in_place_scratch::<T>(self.nrows(), k, p)
    }
    fn apply(&self, o: MatMut<'_, T>, b: MatRef<'_, T>, p: Par, s: &mut MemStack) {
        self.run(o, b, p, s, false)
    }
    fn conj_apply(&self, o: MatMut<'_, T>, b: MatRef<'_, T>, p: Par, s: &mut MemStack) {
        self.run(o, b, p, s, true)
    }
}
impl<T: ComplexField> BiLinOp<T> for InverseLdlt<T> {
    fn transpose_apply_scratch(&self, k: usize, p: ParShape) -> StackReq {
        self.apply_scratch(k, p)
    }
    fn transpose_apply(&self, o: MatMut<'_, T>, b: MatRef<'_, T>, p: Par, s: &mut MemStack) {
        self.run(o, b, p, s, true)
    }
    fn adjoint_apply(&self, o: MatMut<'_, T>, b: MatRef<'_, T>, p: Par, s: &mut MemStack) {
        self.run(o, b, p, s, false)
    }
}

#[derive(Clone)]
struct InverseSpectral<T> {
    u: Mat<T>,
    v: Mat<T>,
    weights: Vec<T>,
    zero: bool,
}
impl<T> std::fmt::Debug for InverseSpectral<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("InverseSpectral")
    }
}
impl<T: ComplexField> InverseSpectral<T> {
    fn new(
        f: &crate::spectral_factors::Decomposition<T>,
        atol: f64,
        rtol: f64,
        relative_f32: bool,
    ) -> Result<Self, String>
    where
        T::Real: crate::pseudoinverse::ToleranceReal,
    {
        let cutoff = crate::spectral_solve::threshold(f, atol, rtol, relative_f32)?;
        let k = f.s().dim();
        let weights: Vec<T> = (0..k)
            .map(|i| {
                let x = real(&f.s()[i]);
                if abs(&x) > cutoff {
                    from_real::<T>(&recip(&x))
                } else {
                    zero::<T>()
                }
            })
            .collect();
        Ok(Self {
            u: f.u().subcols(0, k).to_owned(),
            v: f.v().subcols(0, k).to_owned(),
            zero: weights.iter().all(|w| *w == zero::<T>()),
            weights,
        })
    }
    fn run(
        &self,
        mut out: MatMut<'_, T>,
        b: MatRef<'_, T>,
        mut p: Par,
        s: &mut MemStack,
        t: bool,
        c: bool,
    ) {
        let k = self.weights.len();
        if self.zero || k == 0 || b.ncols() == 0 {
            out.fill(zero::<T>());
            return;
        }
        let (input, output) = if t {
            (&self.v, &self.u)
        } else {
            (&self.u, &self.v)
        };
        let (mut projected, _) = faer::linalg::temp_mat_zeroed::<T, _, _>(k, b.ncols(), s);
        let transformed = t ^ c;
        use faer::linalg::matmul::matmul_with_conj;
        use faer::{Accum, Conj};
        matmul_with_conj(
            projected.as_mat_mut(),
            Accum::Replace,
            input.as_ref().transpose(),
            if transformed { Conj::No } else { Conj::Yes },
            b,
            Conj::No,
            one::<T>(),
            p.reborrow(),
        );
        for j in 0..b.ncols() {
            for i in 0..k {
                let x = copy(&projected.as_mat_ref()[(i, j)]);
                projected.as_mat_mut()[(i, j)] = if self.weights[i] == zero::<T>() {
                    zero::<T>()
                } else {
                    mul(&x, &self.weights[i])
                };
            }
        }
        matmul_with_conj(
            out,
            Accum::Replace,
            output.as_ref(),
            if transformed { Conj::Yes } else { Conj::No },
            projected.as_mat_ref(),
            Conj::No,
            one::<T>(),
            p,
        );
    }
}
impl<T: ComplexField> LinOp<T> for InverseSpectral<T> {
    fn nrows(&self) -> usize {
        self.v.nrows()
    }
    fn ncols(&self) -> usize {
        self.u.nrows()
    }
    fn apply_scratch(&self, k: usize, _: ParShape) -> StackReq {
        if self.zero {
            StackReq::EMPTY
        } else {
            faer::linalg::temp_mat_scratch::<T>(self.weights.len(), k)
        }
    }
    fn apply(&self, o: MatMut<'_, T>, b: MatRef<'_, T>, p: Par, s: &mut MemStack) {
        self.run(o, b, p, s, false, false)
    }
    fn conj_apply(&self, o: MatMut<'_, T>, b: MatRef<'_, T>, p: Par, s: &mut MemStack) {
        self.run(o, b, p, s, false, true)
    }
}
impl<T: ComplexField> BiLinOp<T> for InverseSpectral<T> {
    fn transpose_apply_scratch(&self, k: usize, p: ParShape) -> StackReq {
        self.apply_scratch(k, p)
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
    pub fn operator_spectral_inverse(
        &mut self,
        factor: &crate::spectral_factors::OwnedSpectral,
        atol: f64,
        rtol: f64,
        relative_f32: bool,
    ) -> Result<(), String> {
        guarded(|| {
            use crate::spectral_factors::SpectralData;
            self.data = Some(match factor.data()? {
                SpectralData::F32(f) => Data::F32(Operator::new(Base::InverseSpectral(Box::new(
                    InverseSpectral::new(f, atol, rtol, relative_f32)?,
                )))),
                SpectralData::F64(f) => Data::F64(Operator::new(Base::InverseSpectral(Box::new(
                    InverseSpectral::new(f, atol, rtol, relative_f32)?,
                )))),
                SpectralData::C32(f) => Data::C32(Operator::new(Base::InverseSpectral(Box::new(
                    InverseSpectral::new(f, atol, rtol, relative_f32)?,
                )))),
                SpectralData::C64(f) => Data::C64(Operator::new(Base::InverseSpectral(Box::new(
                    InverseSpectral::new(f, atol, rtol, relative_f32)?,
                )))),
            });
            Ok(())
        })
    }
}

#[derive(Clone)]
struct InverseSparseFactors<T> {
    unit_lower: bool,
    l: faer::sparse::SparseColMat<usize, T>,
    u: faer::sparse::SparseColMat<usize, T>,
    rows: faer::perm::Perm<usize>,
    cols: faer::perm::Perm<usize>,
}
impl<T> std::fmt::Debug for InverseSparseFactors<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("InverseSparseFactors")
    }
}
impl<T: ComplexField> InverseSparseFactors<T> {
    fn new(
        l: &Storage<T>,
        u: &Storage<T>,
        rows: &crate::permutations::OwnedPermutation,
        cols: &crate::permutations::OwnedPermutation,
        unit_lower: bool,
    ) -> Result<Self, String> {
        let (l, u) = (l.csc()?, u.csc()?);
        let n = l.nrows();
        if l.ncols() != n
            || u.nrows() != n
            || u.ncols() != n
            || rows.data()?.len() != n
            || cols.data()?.len() != n
        {
            return Err(
                "sparse LU components must have matching square shapes and permutations".into(),
            );
        }
        for j in 0..n {
            if l.as_ref().row_idx_of_col_raw(j).first() != Some(&j)
                || u.as_ref().row_idx_of_col_raw(j).last() != Some(&j)
            {
                return Err(
                    "sparse LU factors require stored leading L/trailing U diagonals".into(),
                );
            }
        }
        for j in 0..n {
            if l.as_ref().row_idx_of_col_raw(j).iter().any(|&i| i < j)
                || u.as_ref().row_idx_of_col_raw(j).iter().any(|&i| i > j)
            {
                return Err("sparse LU components must be lower/upper triangular".into());
            }
            if u.as_ref()
                .val_of_col(j)
                .last()
                .is_none_or(|x| abs(x) == zero::<T::Real>())
            {
                return Err("sparse LU preconditioner requires nonzero U pivots".into());
            }
        }
        Ok(Self {
            unit_lower,
            l,
            u,
            rows: rows.data()?.clone(),
            cols: cols.data()?.clone(),
        })
    }
    fn run(
        &self,
        mut out: MatMut<'_, T>,
        b: MatRef<'_, T>,
        mut p: Par,
        s: &mut MemStack,
        t: bool,
        c: bool,
    ) {
        let n = self.l.nrows();
        let (mut tmp, _) = faer::linalg::temp_mat_zeroed::<T, _, _>(n, b.ncols(), s);
        let (gather, scatter) = if t {
            (&self.cols, &self.rows)
        } else {
            (&self.rows, &self.cols)
        };
        let g = gather.as_ref().arrays().0;
        let q = scatter.as_ref().arrays().0;
        for j in 0..b.ncols() {
            for i in 0..n {
                tmp.as_mat_mut()[(i, j)] = copy(&b[(g[i], j)]);
            }
        }
        let conj = if c { faer::Conj::Yes } else { faer::Conj::No };
        use faer::sparse::linalg::triangular_solve as tri;
        if t {
            tri::solve_upper_triangular_transpose_in_place(
                self.u.as_ref(),
                conj,
                tmp.as_mat_mut(),
                p.reborrow(),
            );
            if self.unit_lower {
                tri::solve_unit_lower_triangular_transpose_in_place(
                    self.l.as_ref(),
                    conj,
                    tmp.as_mat_mut(),
                    p,
                );
            } else {
                tri::solve_lower_triangular_transpose_in_place(
                    self.l.as_ref(),
                    conj,
                    tmp.as_mat_mut(),
                    p,
                );
            }
        } else {
            if self.unit_lower {
                tri::solve_unit_lower_triangular_in_place(
                    self.l.as_ref(),
                    conj,
                    tmp.as_mat_mut(),
                    p.reborrow(),
                );
            } else {
                tri::solve_lower_triangular_in_place(
                    self.l.as_ref(),
                    conj,
                    tmp.as_mat_mut(),
                    p.reborrow(),
                );
            }
            tri::solve_upper_triangular_in_place(self.u.as_ref(), conj, tmp.as_mat_mut(), p);
        }
        for j in 0..b.ncols() {
            for i in 0..n {
                out[(q[i], j)] = copy(&tmp.as_mat_ref()[(i, j)]);
            }
        }
    }
}
impl<T: ComplexField> LinOp<T> for InverseSparseFactors<T> {
    fn nrows(&self) -> usize {
        self.l.nrows()
    }
    fn ncols(&self) -> usize {
        self.u.ncols()
    }
    fn apply_scratch(&self, k: usize, _: ParShape) -> StackReq {
        faer::linalg::temp_mat_scratch::<T>(self.nrows(), k)
    }
    fn apply(&self, o: MatMut<'_, T>, b: MatRef<'_, T>, p: Par, s: &mut MemStack) {
        self.run(o, b, p, s, false, false)
    }
    fn conj_apply(&self, o: MatMut<'_, T>, b: MatRef<'_, T>, p: Par, s: &mut MemStack) {
        self.run(o, b, p, s, false, true)
    }
}
impl<T: ComplexField> BiLinOp<T> for InverseSparseFactors<T> {
    fn transpose_apply_scratch(&self, k: usize, p: ParShape) -> StackReq {
        self.apply_scratch(k, p)
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
    pub fn operator_sparse_factor_inverse(
        &mut self,
        l: &SparseOwner,
        u: &SparseOwner,
        rows: &crate::permutations::OwnedPermutation,
        cols: &crate::permutations::OwnedPermutation,
        unit_lower: bool,
    ) -> Result<(), String> {
        guarded(|| {
            self.data = Some(match (l.data()?, u.data()?) {
                (SparseData::F32(l), SparseData::F32(u)) => {
                    Data::F32(Operator::new(Base::InverseSparseFactors(Box::new(
                        InverseSparseFactors::new(l, u, rows, cols, unit_lower)?,
                    ))))
                }
                (SparseData::F64(l), SparseData::F64(u)) => {
                    Data::F64(Operator::new(Base::InverseSparseFactors(Box::new(
                        InverseSparseFactors::new(l, u, rows, cols, unit_lower)?,
                    ))))
                }
                (SparseData::C32(l), SparseData::C32(u)) => {
                    Data::C32(Operator::new(Base::InverseSparseFactors(Box::new(
                        InverseSparseFactors::new(l, u, rows, cols, unit_lower)?,
                    ))))
                }
                (SparseData::C64(l), SparseData::C64(u)) => {
                    Data::C64(Operator::new(Base::InverseSparseFactors(Box::new(
                        InverseSparseFactors::new(l, u, rows, cols, unit_lower)?,
                    ))))
                }
                _ => return Err("sparse LU component scalar types must match".into()),
            });
            Ok(())
        })
    }
}

impl<T> std::fmt::Debug for crate::sparse_qr_components::Factors<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SimplicialQrInverse")
    }
}
fn sparse_qr_action<T: ComplexField>(
    f: &crate::sparse_qr_components::Factors<T>,
    mut out: MatMut<'_, T>,
    b: MatRef<'_, T>,
    p: Par,
    s: &mut MemStack,
    t: bool,
    c: bool,
) {
    let (m, n) = (f.plan.symbolic.nrows(), f.plan.symbolic.ncols());
    let (mut tmp, s) =
        faer::linalg::temp_mat_zeroed::<T, _, _>(if t { n } else { m }, b.ncols(), s);
    let conj = if c { faer::Conj::Yes } else { faer::Conj::No };
    let (perm, inv) = f.plan.cols.as_ref().arrays();
    if t {
        for j in 0..b.ncols() {
            for i in 0..n {
                tmp.as_mat_mut()[(i, j)] = copy(&b[(perm[i], j)]);
            }
        }
        faer::sparse::linalg::triangular_solve::solve_upper_triangular_transpose_in_place(
            f.r.as_ref(),
            conj,
            tmp.as_mat_mut(),
            p,
        );
        f.apply_q_into(tmp.as_mat_ref(), out, !c);
    } else {
        tmp.as_mat_mut().copy_from(b);
        f.as_ref()
            .solve_in_place_with_conj(conj, tmp.as_mat_mut(), p, s);
        for j in 0..b.ncols() {
            for i in 0..n {
                out[(i, j)] = copy(&tmp.as_mat_ref()[(inv[i], j)]);
            }
        }
    }
}
impl<T: ComplexField> LinOp<T> for crate::sparse_qr_components::Factors<T> {
    fn nrows(&self) -> usize {
        self.plan.symbolic.ncols()
    }
    fn ncols(&self) -> usize {
        self.plan.symbolic.nrows()
    }
    fn apply_scratch(&self, k: usize, p: ParShape) -> StackReq {
        faer::linalg::temp_mat_scratch::<T>(self.ncols(), k)
            .and(self.plan.symbolic.solve_in_place_scratch::<T>(k, p))
    }
    fn apply(&self, o: MatMut<'_, T>, b: MatRef<'_, T>, p: Par, s: &mut MemStack) {
        sparse_qr_action(self, o, b, p, s, false, false)
    }
    fn conj_apply(&self, o: MatMut<'_, T>, b: MatRef<'_, T>, p: Par, s: &mut MemStack) {
        sparse_qr_action(self, o, b, p, s, false, true)
    }
}
impl<T: ComplexField> BiLinOp<T> for crate::sparse_qr_components::Factors<T> {
    fn transpose_apply_scratch(&self, k: usize, p: ParShape) -> StackReq {
        self.apply_scratch(k, p)
    }
    fn transpose_apply(&self, o: MatMut<'_, T>, b: MatRef<'_, T>, p: Par, s: &mut MemStack) {
        sparse_qr_action(self, o, b, p, s, true, false)
    }
    fn adjoint_apply(&self, o: MatMut<'_, T>, b: MatRef<'_, T>, p: Par, s: &mut MemStack) {
        sparse_qr_action(self, o, b, p, s, true, true)
    }
}
/// Construction scratch of a sparse QR inverse action: nonempty only for wide
/// QR, whose dense `pinv(R)` is built from an SVD in caller scratch.
fn sparse_qr_inverse_requirement(
    factor: &crate::sparse_factors::SparseFactor,
    p: ParShape,
) -> Result<StackReq, String> {
    Ok(match factor.wide_qr_shape()? {
        None => StackReq::EMPTY,
        Some((m, n, 0)) => WideQrInverse::<f32>::construction_scratch(m, n, p),
        Some((m, n, 1)) => WideQrInverse::<f64>::construction_scratch(m, n, p),
        Some((m, n, 2)) => WideQrInverse::<c32>::construction_scratch(m, n, p),
        Some((m, n, _)) => WideQrInverse::<c64>::construction_scratch(m, n, p),
    })
}
fn sparse_qr_inverse(
    factor: &crate::sparse_factors::SparseFactor,
    scratch: &mut crate::workspace::OwnedScratch,
    par: Par<'_>,
) -> Result<Data, String> {
    use crate::sparse_factors::QrComponents;
    fn wrap<T: ComplexField>(
        f: crate::sparse_qr_components::Factors<T>,
    ) -> Result<Operator<T>, String> {
        for j in 0..f.r.ncols() {
            if f.r.as_ref().row_idx_of_col_raw(j).last() != Some(&j) {
                return Err("sparse QR inverse action requires stored R pivots".into());
            }
            if f.r
                .as_ref()
                .val_of_col(j)
                .last()
                .is_none_or(|x| abs(x) == zero::<T::Real>())
            {
                return Err("sparse QR inverse action requires full column rank".into());
            }
        }
        Ok(Operator::new(Base::InverseSparseQr(Box::new(f))))
    }
    fn wrap_owned<T: ComplexField>(
        f: crate::sparse_qr_owned::Factors<T>,
    ) -> Result<Operator<T>, String> {
        Ok(Operator::new(Base::InverseOwnedSparseQr(Box::new(
            OwnedQrInverse::new(f)?,
        ))))
    }
    fn wrap_wide<T: ComplexField>(
        f: &crate::sparse_qr_wide::Factors<T>,
        scratch: &mut crate::workspace::OwnedScratch,
        par: Par<'_>,
    ) -> Result<Operator<T>, String> {
        let (m, n) = f.dimensions();
        let stack = scratch.stack(WideQrInverse::<T>::construction_scratch(m, n, par.shape()))?;
        Ok(Operator::new(Base::InverseWideSparseQr(Box::new(
            WideQrInverse::new_in(f, par, stack)?,
        ))))
    }
    Ok(match factor.simplicial_qr_components()? {
        QrComponents::WideF32(f) => Data::F32(wrap_wide(f, scratch, par)?),
        QrComponents::WideF64(f) => Data::F64(wrap_wide(f, scratch, par)?),
        QrComponents::WideC32(f) => Data::C32(wrap_wide(f, scratch, par)?),
        QrComponents::WideC64(f) => Data::C64(wrap_wide(f, scratch, par)?),
        QrComponents::OwnedF32(f) => Data::F32(wrap_owned(f)?),
        QrComponents::OwnedF64(f) => Data::F64(wrap_owned(f)?),
        QrComponents::OwnedC32(f) => Data::C32(wrap_owned(f)?),
        QrComponents::OwnedC64(f) => Data::C64(wrap_owned(f)?),
        QrComponents::F32(f) => Data::F32(wrap(f)?),
        QrComponents::F64(f) => Data::F64(wrap(f)?),
        QrComponents::C32(f) => Data::C32(wrap(f)?),
        QrComponents::C64(f) => Data::C64(wrap(f)?),
    })
}
fn write_requirement(req: StackReq, out: *mut usize) {
    // SAFETY: Julia passes a live two-element UInt buffer.
    unsafe {
        out.write(req.size_bytes());
        out.add(1).write(req.align_bytes());
    }
}
#[julia]
impl SparseFactor {
    #[julia]
    pub fn operator_sparse_qr_inverse_requirements(
        &self,
        out: &mut [usize],
        team: Option<&crate::teams::OwnedTeam>,
    ) -> Result<(), String> {
        let out = crate::workspace::requirement_out(out)?;
        guarded(|| {
            let req = with_team(team, |p| sparse_qr_inverse_requirement(self, p.shape()))?;
            write_requirement(req, out);
            Ok(())
        })
    }

}
#[julia]
impl OwnedLinearOperator {
    #[julia]
    pub fn operator_sparse_qr_inverse(
        &mut self,
        factor: &crate::sparse_factors::SparseFactor,
        scratch: &mut crate::workspace::OwnedScratch,
        team: Option<&crate::teams::OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            let data = with_team(team, |p| sparse_qr_inverse(factor, scratch, p))?;
            self.data = Some(data);
            Ok(())
        })
    }

}

mod apply_workspace;
