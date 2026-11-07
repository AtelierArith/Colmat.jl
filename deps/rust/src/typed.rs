//! Typed native storage. Scalar buffers carry the exact Julia primitive layout.
use super::guarded;
use crate::cholesky_factors::{Ldlt, Llt};
use crate::lblt::Lblt;
use crate::views::Borrowed;
use faer::linalg::solvers::{
    ColPivQr, DenseSolveCore, FullPivLu, PartialPivLu, Qr, ShapeCore, Solve, SolveLstsq, Svd,
};
use faer::traits::{math_utils::*, ComplexField};
use faer::{c32, c64, Accum, Conj, Mat, MatRef, Par, Side};
use rustcall_julia_macros::julia;

#[derive(Clone)]
pub(crate) enum MatrixData {
    F32(Mat<f32>),
    F64(Mat<f64>),
    C32(Mat<c32>),
    C64(Mat<c64>),
}
macro_rules! dispatch {
    ($data:expr,$a:ident=>$body:expr) => {
        match $data {
            MatrixData::F32($a) => $body,
            MatrixData::F64($a) => $body,
            MatrixData::C32($a) => $body,
            MatrixData::C64($a) => $body,
        }
    };
}
macro_rules! mapped {
    ($data:expr,$a:ident=>$body:expr) => {
        match $data {
            MatrixData::F32($a) => MatrixData::F32($body),
            MatrixData::F64($a) => MatrixData::F64($body),
            MatrixData::C32($a) => MatrixData::C32($body),
            MatrixData::C64($a) => MatrixData::C64($body),
        }
    };
}
macro_rules! binary {
    ($left:expr,$right:expr,$a:ident,$b:ident=>$body:expr) => {
        match ($left, $right) {
            (MatrixData::F32($a), MatrixData::F32($b)) => MatrixData::F32($body),
            (MatrixData::F64($a), MatrixData::F64($b)) => MatrixData::F64($body),
            (MatrixData::C32($a), MatrixData::C32($b)) => MatrixData::C32($body),
            (MatrixData::C64($a), MatrixData::C64($b)) => MatrixData::C64($body),
            _ => return Err("native scalar types must match".into()),
        }
    };
}
pub(crate) fn finite<T: ComplexField>(a: MatRef<'_, T>) -> Result<(), String> {
    if !a.is_all_finite() {
        return Err("input must be finite".into());
    }
    Ok(())
}
unsafe fn read_matrix<T: ComplexField + Copy>(
    p: *const u8,
    m: usize,
    n: usize,
) -> Result<Mat<T>, String> {
    let len = m.checked_mul(n).ok_or("matrix size overflow")?;
    let data = if len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(p.cast::<T>(), len)
    };
    // Stored values keep Inf/NaN; numeric kernels check finiteness themselves.
    Ok(Mat::from_fn(m, n, |i, j| data[i + j * m]))
}
pub(crate) unsafe fn write_matrix<T: ComplexField>(a: MatRef<'_, T>, p: *mut u8) {
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            p.cast::<T>().add(i + j * a.nrows()).write(copy(&a[(i, j)]));
        }
    }
}
pub(crate) fn arithmetic<T: ComplexField>(
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    op: i32,
) -> Result<Mat<T>, String> {
    match op {
        0 | 1 => {
            if a.shape() != b.shape() {
                return Err("matrix sizes must match".into());
            }
            Ok(Mat::from_fn(a.nrows(), a.ncols(), |i, j| {
                if op == 0 {
                    add(&a[(i, j)], &b[(i, j)])
                } else {
                    sub(&a[(i, j)], &b[(i, j)])
                }
            }))
        }
        2 => {
            if a.ncols() != b.nrows() {
                return Err("inner dimensions must match".into());
            }
            let mut c = Mat::zeros(a.nrows(), b.ncols());
            faer::linalg::matmul::matmul(c.as_mut(), Accum::Replace, a, b, one::<T>(), Par::Seq);
            Ok(c)
        }
        3 => {
            a.nrows()
                .checked_mul(b.nrows())
                .ok_or("Kronecker size overflow")?;
            a.ncols()
                .checked_mul(b.ncols())
                .ok_or("Kronecker size overflow")?;
            Ok(a.kron(b))
        }
        _ => Err("unknown arithmetic operation".into()),
    }
}
unsafe fn scale<T: ComplexField>(a: MatRef<'_, T>, factor: *const u8) -> Mat<T> {
    let x = &*factor.cast::<T>();
    Mat::from_fn(a.nrows(), a.ncols(), |i, j| mul(x, &a[(i, j)]))
}
// Real factors scale complex entries componentwise, like Julia's `Real*Complex`,
// so Inf/NaN parts do not leak into the other component.
unsafe fn scale_real<T: ComplexField>(a: MatRef<'_, T>, factor: *const u8) -> Mat<T> {
    let x = &*factor.cast::<T::Real>();
    Mat::from_fn(a.nrows(), a.ncols(), |i, j| mul_real(&a[(i, j)], x))
}
unsafe fn scalar_output<T: ComplexField>(
    a: MatRef<'_, T>,
    out: *mut u8,
    code: i32,
) -> Result<(), String> {
    let value = match code {
        0 => a.sum(),
        1 => {
            if a.nrows() != a.ncols() {
                return Err("determinant requires square input".into());
            }
            a.determinant(Par::Seq)
        }
        2 => {
            if a.nrows() != a.ncols() {
                return Err("trace requires square input".into());
            }
            (0..a.nrows()).fold(zero::<T>(), |s, i| add(&s, &a[(i, i)]))
        }
        _ => return Err("unknown scalar reduction".into()),
    };
    out.cast::<T>().write(value);
    Ok(())
}
pub(crate) unsafe fn norm_output<T: ComplexField>(
    a: MatRef<'_, T>,
    out: *mut u8,
    code: i32,
) -> Result<(), String> {
    let value = match code {
        0 => a.norm_l2(),
        1 => {
            if T::IS_REAL {
                a.norm_l1()
            } else {
                (0..a.ncols()).fold(zero::<T::Real>(), |sum, j| {
                    (0..a.nrows()).fold(sum, |sum, i| add(&sum, &abs(&a[(i, j)])))
                })
            }
        }
        2 => {
            if T::IS_REAL {
                // faer's norm_max skips NaN (tensor4all/faer#316); Julia's normInf propagates it.
                if a.has_nan() {
                    nan::<T::Real>()
                } else {
                    a.norm_max()
                }
            } else {
                (0..a.ncols()).fold(zero::<T::Real>(), |value, j| {
                    (0..a.nrows()).fold(value, |value, i| {
                        let x = abs(&a[(i, j)]);
                        if is_nan(&value) {
                            value
                        } else if is_nan(&x) || x > value {
                            x
                        } else {
                            value
                        }
                    })
                })
            }
        }
        3 => a.squared_norm_l2(),
        _ => return Err("unknown norm".into()),
    };
    out.cast::<T::Real>().write(value);
    Ok(())
}
pub(crate) unsafe fn svd_output<T: ComplexField>(
    a: MatRef<'_, T>,
    u: *mut u8,
    s: *mut u8,
    v: *mut u8,
    full: bool,
    values_only: bool,
) -> Result<(), String> {
    finite(a)?;
    let (m, n) = a.shape();
    let k = m.min(n);
    if k == 0 {
        if full && !values_only {
            write_matrix(Mat::<T>::identity(m, m).as_ref(), u);
            write_matrix(Mat::<T>::identity(n, n).as_ref(), v);
        }
        return Ok(());
    }
    if values_only {
        let values = a.singular_values(Par::Seq).map_err(|e| format!("{e:?}"))?;
        for (i, x) in values.into_iter().enumerate() {
            s.cast::<T::Real>().add(i).write(x);
        }
        return Ok(());
    }
    let f = if full {
        a.svd(Par::Seq)
    } else {
        a.thin_svd(Par::Seq)
    }
    .map_err(|e| format!("{e:?}"))?;
    write_matrix(f.U(), u);
    write_matrix(f.V(), v);
    for i in 0..k {
        s.cast::<T::Real>().add(i).write(real(&f.S()[i]));
    }
    Ok(())
}

#[julia]
pub struct TypedMatrix {
    pub(crate) data: Option<MatrixData>,
}
impl TypedMatrix {
    pub(crate) fn data(&self) -> Result<&MatrixData, String> {
        self.data.as_ref().ok_or("matrix is not initialized".into())
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn new() -> Self {
        Self { data: None }
    }
    #[julia]
    pub fn duplicate(&self) -> Result<Self, String> {
        guarded(|| {
            Ok(Self {
                data: Some(self.data()?.clone()),
            })
        })
    }
    #[julia]
    pub fn load(&mut self, p: *const u8, m: usize, n: usize, tag: i32) -> Result<(), String> {
        guarded(|| {
            self.data = Some(unsafe {
                match tag {
                    0 => MatrixData::F32(read_matrix(p, m, n)?),
                    1 => MatrixData::F64(read_matrix(p, m, n)?),
                    2 => MatrixData::C32(read_matrix(p, m, n)?),
                    3 => MatrixData::C64(read_matrix(p, m, n)?),
                    _ => return Err("unsupported scalar type".into()),
                }
            });
            Ok(())
        })
    }
    #[julia]
    pub fn zeroes(&mut self, m: usize, n: usize, tag: i32) -> Result<(), String> {
        guarded(|| {
            m.checked_mul(n).ok_or("matrix size overflow")?;
            self.data = Some(match tag {
                0 => MatrixData::F32(Mat::zeros(m, n)),
                1 => MatrixData::F64(Mat::zeros(m, n)),
                2 => MatrixData::C32(Mat::zeros(m, n)),
                3 => MatrixData::C64(Mat::zeros(m, n)),
                _ => return Err("unsupported scalar type".into()),
            });
            Ok(())
        })
    }
    #[julia]
    pub fn to_buffer(&self, p: *mut u8, len: usize) -> Result<(), String> {
        guarded(|| {
            let a = self.data()?;
            let size = dispatch!(a,x=>x.nrows()*x.ncols());
            if len != size {
                return Err("buffer length mismatch".into());
            }
            unsafe {
                dispatch!(a,x=>write_matrix(x.as_ref(),p));
            }
            Ok(())
        })
    }
    #[julia]
    pub fn element(&self, i: usize, j: usize, p: *mut u8) -> Result<(), String> {
        guarded(|| {
            let a = self.data()?;
            dispatch!(a,x=>{if i>=x.nrows()||j>=x.ncols(){return Err("index out of bounds".into());}unsafe{write_element(x.as_ref(),i,j,p);}});
            Ok(())
        })
    }
    #[julia]
    pub fn set_element(&mut self, i: usize, j: usize, p: *const u8) -> Result<(), String> {
        guarded(|| {
            let a = self.data.as_mut().ok_or("matrix is not initialized")?;
            dispatch!(a,x=>{if i>=x.nrows()||j>=x.ncols(){return Err("index out of bounds".into());}unsafe{set_element(x,i,j,p);}});
            Ok(())
        })
    }
    #[julia]
    pub fn fill_value(&mut self, p: *const u8) -> Result<(), String> {
        guarded(|| {
            let a = self.data.as_mut().ok_or("matrix is not initialized")?;
            dispatch!(a,x=>unsafe{fill_matrix(x,p)});
            Ok(())
        })
    }
    #[julia]
    pub fn transform_from(&mut self, source: &TypedMatrix, op: i32) -> Result<(), String> {
        guarded(|| {
            let a = source.data()?;
            self.data = Some(
                mapped!(a,x=>match op{0=>x.clone(),1=>x.as_ref().transpose().to_owned(),2=>x.as_ref().adjoint().to_owned(),3=>x.as_ref().conjugate().to_owned(),_=>return Err("unknown transform".into())}),
            );
            Ok(())
        })
    }
    #[julia]
    pub fn arithmetic_from(
        &mut self,
        left: &TypedMatrix,
        right: &TypedMatrix,
        op: i32,
    ) -> Result<(), String> {
        guarded(|| {
            let a = left.data()?;
            let b = right.data()?;
            self.data = Some(binary!(a,b,x,y=>arithmetic(x.as_ref(),y.as_ref(),op)?));
            Ok(())
        })
    }
    #[julia]
    pub fn scale_from(&mut self, source: &TypedMatrix, factor: *const u8) -> Result<(), String> {
        guarded(|| {
            let a = source.data()?;
            self.data = Some(mapped!(a,x=>unsafe{scale(x.as_ref(),factor)}));
            Ok(())
        })
    }
    #[julia]
    pub fn scale_real_from(
        &mut self,
        source: &TypedMatrix,
        factor: *const u8,
    ) -> Result<(), String> {
        guarded(|| {
            let a = source.data()?;
            self.data = Some(mapped!(a,x=>unsafe{scale_real(x.as_ref(),factor)}));
            Ok(())
        })
    }
    #[julia]
    pub fn scalar_reduction(&self, out: *mut u8, code: i32) -> Result<(), String> {
        guarded(|| unsafe { dispatch!(self.data()?,x=>scalar_output(x.as_ref(),out,code)) })
    }
    #[julia]
    pub fn norm_reduction(&self, out: *mut u8, code: i32) -> Result<(), String> {
        guarded(|| unsafe { dispatch!(self.data()?,x=>norm_output(x.as_ref(),out,code)) })
    }
    #[julia]
    pub fn predicate(&self, code: i32) -> Result<bool, String> {
        guarded(|| {
            dispatch!(self.data()?,x=>Ok(match code{
                0=>x.is_all_finite(),1=>x.has_nan(),2=>x.nrows()==x.ncols()&&x.as_ref().is_identity(),
                3=>symmetric(x.as_ref(),false),4=>symmetric(x.as_ref(),true),
                5=>real_triangle(x.as_ref(),true),6=>real_triangle(x.as_ref(),false),_=>return Err("unknown predicate".into())
            }))
        })
    }
    #[julia]
    pub fn svd_buffers(
        &self,
        u: *mut u8,
        s: *mut u8,
        v: *mut u8,
        full: bool,
        values_only: bool,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            dispatch!(self.data()?,a=>svd_output(a.as_ref(),u,s,v,full,values_only))
        })
    }
}
unsafe fn write_element<T: ComplexField>(a: MatRef<'_, T>, i: usize, j: usize, p: *mut u8) {
    p.cast::<T>().write(copy(&a[(i, j)]));
}
unsafe fn set_element<T: ComplexField>(a: &mut Mat<T>, i: usize, j: usize, p: *const u8) {
    a[(i, j)] = copy(&*p.cast::<T>());
}
unsafe fn fill_matrix<T: ComplexField>(a: &mut Mat<T>, p: *const u8) {
    a.as_mut().fill(copy(&*p.cast::<T>()));
}
fn symmetric<T: ComplexField>(a: MatRef<'_, T>, hermitian: bool) -> bool {
    a.nrows() == a.ncols()
        && (0..a.ncols()).all(|j| {
            (0..=j).all(|i| {
                a[(i, j)]
                    == if hermitian {
                        conj(&a[(j, i)])
                    } else {
                        copy(&a[(j, i)])
                    }
            })
        })
}
/// Whether the stored lower (or upper) triangle, diagonal included, is real.
fn real_triangle<T: ComplexField>(a: MatRef<'_, T>, lower: bool) -> bool {
    (0..a.ncols()).all(|j| {
        let (start, end) = if lower {
            (j, a.nrows())
        } else {
            (0, (j + 1).min(a.nrows()))
        };
        (start..end).all(|i| a[(i, j)] == conj(&a[(i, j)]))
    })
}

#[derive(Clone)]
pub(crate) enum Factor<T: ComplexField> {
    Lu(PartialPivLu<T>),
    FullLu(FullPivLu<T>),
    Qr(Qr<T>),
    PivotQr(ColPivQr<T>),
    Llt(Llt<T>),
    Ldlt(Ldlt<T>),
    Lblt(Lblt<T>),
    Svd(Svd<T>),
}
macro_rules! factor_dispatch {
    ($f:expr,$x:ident=>$body:expr) => {
        match $f {
            Factor::Lu($x) => $body,
            Factor::FullLu($x) => $body,
            Factor::Qr($x) => $body,
            Factor::PivotQr($x) => $body,
            Factor::Llt($x) => $body,
            Factor::Ldlt($x) => $body,
            Factor::Lblt($x) => $body,
            Factor::Svd($x) => $body,
        }
    };
}
#[derive(Clone)]
pub(crate) enum FactorData {
    F32(Factor<f32>),
    F64(Factor<f64>),
    C32(Factor<c32>),
    C64(Factor<c64>),
}
macro_rules! with_factor {
    ($data:expr,$x:ident=>$body:expr) => {
        match $data {
            FactorData::F32($x) => $body,
            FactorData::F64($x) => $body,
            FactorData::C32($x) => $body,
            FactorData::C64($x) => $body,
        }
    };
}
macro_rules! factor_matrix {
    ($data:expr,$x:ident=>$body:expr) => {
        match $data {
            FactorData::F32($x) => MatrixData::F32($body),
            FactorData::F64($x) => MatrixData::F64($body),
            FactorData::C32($x) => MatrixData::C32($body),
            FactorData::C64($x) => MatrixData::C64($body),
        }
    };
}
macro_rules! factor_rhs {
    ($data:expr,$rhs:expr,$x:ident,$b:ident=>$body:expr) => {
        match ($data, $rhs) {
            (FactorData::F32($x), MatrixData::F32($b)) => MatrixData::F32($body),
            (FactorData::F64($x), MatrixData::F64($b)) => MatrixData::F64($body),
            (FactorData::C32($x), MatrixData::C32($b)) => MatrixData::C32($body),
            (FactorData::C64($x), MatrixData::C64($b)) => MatrixData::C64($body),
            _ => return Err("factor and RHS scalar types must match".into()),
        }
    };
}
fn factor_new<T: ComplexField>(
    a: MatRef<'_, T>,
    kind: i32,
    lower: bool,
    par: Par<'_>,
) -> Result<(Option<Factor<T>>, usize), String> {
    let h = if (4..=6).contains(&kind) || kind == 8 {
        Some(hermitian_copy(a, lower)?)
    } else {
        None
    };
    let a = if let Some(ref h) = h {
        h.as_ref()
    } else {
        finite(a)?;
        a
    };
    let side = if lower { Side::Lower } else { Side::Upper };
    let f = match kind {
        0 => Factor::Lu(a.partial_piv_lu(par)),
        1 => Factor::FullLu(a.full_piv_lu(par)),
        2 => Factor::Qr(a.qr(par)),
        3 => Factor::PivotQr(a.col_piv_qr(par)),
        4 => match Llt::new(a, side, par) {
            Ok(f) => Factor::Llt(f),
            Err(faer::linalg::solvers::LltError::NonPositivePivot { index }) => {
                return Ok((None, index + 1))
            }
        },
        5 => match Ldlt::new(a, side, par)? {
            Ok(f) => Factor::Ldlt(f),
            Err(faer::linalg::solvers::LdltError::ZeroPivot { index }) => {
                return Ok((None, index + 1))
            }
        },
        6 | 8 => Factor::Lblt(Lblt::new(a, side, par, kind == 8)),
        7 => Factor::Svd(a.thin_svd(par).map_err(|e| format!("{e:?}"))?),
        _ => return Err("unknown factorization".into()),
    };
    let u = match &f {
        Factor::Lu(x) => Some(x.U()),
        Factor::FullLu(x) => Some(x.U()),
        _ => None,
    };
    let mut info = if let Some(u) = u {
        (0..u.nrows().min(u.ncols()))
            .find(|&i| u[(i, i)] == zero::<T>())
            .map_or(0, |i| i + 1)
    } else {
        0
    };
    if let Factor::Lblt(x) = &f {
        let mut i = 0;
        while i < x.nrows() {
            let block = i + 1 < x.nrows() && x.B_subdiag()[i] != zero::<T>();
            let determinant = if block {
                sub(
                    &mul(&x.B_diag()[i], &x.B_diag()[i + 1]),
                    &mul(&x.B_subdiag()[i], &conj(&x.B_subdiag()[i])),
                )
            } else {
                copy(&x.B_diag()[i])
            };
            if determinant == zero::<T>() {
                info = i + 1;
                break;
            }
            i += if block { 2 } else { 1 };
        }
    }
    Ok((Some(f), info))
}
fn qr_rows<T: ComplexField>(f: &Factor<T>) -> Result<usize, String> {
    match f {
        Factor::Qr(x) => Ok(x.Q_coeff().nrows()),
        Factor::PivotQr(x) => Ok(x.Q_coeff().nrows()),
        _ => Err("not a QR factorization".into()),
    }
}
pub(crate) fn packed_diagonal<T: ComplexField>(
    lower: MatRef<'_, T>,
    diagonal: faer::diag::DiagRef<'_, T>,
) -> Mat<T> {
    Mat::from_fn(lower.nrows(), lower.ncols(), |i, j| {
        if i == j {
            copy(&diagonal[i])
        } else if i > j {
            copy(&lower[(i, j)])
        } else {
            zero::<T>()
        }
    })
}
pub(crate) fn packed<T: ComplexField>(lower: MatRef<'_, T>, upper: MatRef<'_, T>) -> Mat<T> {
    Mat::from_fn(lower.nrows(), upper.ncols(), |i, j| {
        if i > j {
            copy(&lower[(i, j)])
        } else {
            copy(&upper[(i, j)])
        }
    })
}
fn factor_part<T: ComplexField>(f: &Factor<T>, part: i32, par: Par<'_>) -> Result<Mat<T>, String> {
    Ok(match part {
        0 => factor_dispatch!(f,x=>x.reconstruct(par)),
        1 => match f {
            Factor::Lu(x) => x.L().to_owned(),
            Factor::FullLu(x) => x.L().to_owned(),
            Factor::Llt(x) => x.L().to_owned(),
            Factor::Ldlt(x) => x.L().to_owned(),
            Factor::Lblt(x) => x.L().to_owned(),
            _ => return Err("no L factor".into()),
        },
        2 => match f {
            Factor::Lu(x) => x.U().to_owned(),
            Factor::FullLu(x) => x.U().to_owned(),
            _ => return Err("no U factor".into()),
        },
        3 => match f {
            Factor::Qr(x) => x.compute_Q(par),
            Factor::PivotQr(x) => x.compute_Q(par),
            _ => return Err("no Q factor".into()),
        },
        4 => match f {
            Factor::Qr(x) => x.thin_R().to_owned(),
            Factor::PivotQr(x) => x.thin_R().to_owned(),
            _ => return Err("no R factor".into()),
        },
        5 => match f {
            Factor::Ldlt(x) => Mat::from_fn(x.nrows(), x.ncols(), |i, j| {
                if i == j {
                    copy(&x.D()[i])
                } else {
                    zero::<T>()
                }
            }),
            Factor::Lblt(x) => Mat::from_fn(x.nrows(), x.ncols(), |i, j| {
                if i == j {
                    copy(&x.B_diag()[i])
                } else if i == j + 1 {
                    copy(&x.B_subdiag()[j])
                } else if j == i + 1 {
                    conj(&x.B_subdiag()[i])
                } else {
                    zero::<T>()
                }
            }),
            _ => return Err("no D factor".into()),
        },
        6 => match f {
            Factor::Qr(x) => x.compute_thin_Q(par),
            Factor::PivotQr(x) => x.compute_thin_Q(par),
            _ => return Err("no Q factor".into()),
        },
        7 => match f {
            Factor::Lu(x) => packed(x.L(), x.U()),
            Factor::FullLu(x) => packed(x.L(), x.U()),
            Factor::Qr(x) => packed(x.Q_basis(), x.thin_R()),
            Factor::PivotQr(x) => packed(x.Q_basis(), x.thin_R()),
            Factor::Llt(x) => x.L().to_owned(),
            Factor::Ldlt(x) => packed_diagonal(x.L(), x.D()),
            Factor::Lblt(x) => packed_diagonal(x.L(), x.B_diag()),
            _ => return Err("no packed storage for factor".into()),
        },
        8 => match f {
            Factor::Qr(x) => x.Q_basis().to_owned(),
            Factor::PivotQr(x) => x.Q_basis().to_owned(),
            _ => return Err("no QR basis".into()),
        },
        9 => match f {
            Factor::Qr(x) => x.Q_coeff().to_owned(),
            Factor::PivotQr(x) => x.Q_coeff().to_owned(),
            _ => return Err("no QR coefficients".into()),
        },
        _ => return Err("unknown factor part".into()),
    })
}
pub(crate) fn rank_deficient<T: ComplexField>(r: MatRef<'_, T>) -> bool {
    let tolerance = mul(
        &mul(
            &eps::<T::Real>(),
            &from_f64::<T::Real>(r.nrows().max(r.ncols()) as f64),
        ),
        &r.norm_max(),
    );
    (0..r.nrows().min(r.ncols())).any(|i| abs(&r[(i, i)]) <= tolerance)
}
pub(crate) fn factor_solve<T: ComplexField>(
    f: &Factor<T>,
    b: MatRef<'_, T>,
    op: i32,
    mut par: Par<'_>,
) -> Result<Mat<T>, String> {
    finite(b)?;
    let m = factor_dispatch!(f,x=>x.nrows());
    let n = factor_dispatch!(f,x=>x.ncols());
    let transposed = op == 1 || op == 2;
    if b.nrows() != if transposed { n } else { m } {
        return Err("solve dimensions mismatch".into());
    }
    let deficient = match f {
        Factor::Qr(x) => rank_deficient(x.thin_R()),
        Factor::PivotQr(x) => rank_deficient(x.thin_R()),
        Factor::Svd(_) => true,
        _ => false,
    };
    if !deficient && m == n {
        return Ok(match op {
            0 => factor_dispatch!(f,x=>x.solve(b,par.reborrow())),
            1 => factor_dispatch!(f,x=>x.solve_transpose(b,par.reborrow())),
            2 => factor_dispatch!(f,x=>x.solve_adjoint(b,par.reborrow())),
            3 => factor_dispatch!(f,x=>x.solve_conjugate(b,par.reborrow())),
            _ => return Err("unknown solve transform".into()),
        });
    }
    if !deficient && !transposed && m >= n {
        return Ok(match f {
            Factor::Qr(x) => {
                if op == 3 {
                    x.solve_conjugate_lstsq(b, par.reborrow())
                } else {
                    x.solve_lstsq(b, par.reborrow())
                }
            }
            Factor::PivotQr(x) => {
                if op == 3 {
                    x.solve_conjugate_lstsq(b, par.reborrow())
                } else {
                    x.solve_lstsq(b, par.reborrow())
                }
            }
            _ => return Err("rectangular solve requires QR or SVD".into()),
        });
    }
    let a = factor_dispatch!(f,x=>x.reconstruct(par.reborrow()));
    let a = match op {
        0 => a,
        1 => a.as_ref().transpose().to_owned(),
        2 => a.as_ref().adjoint().to_owned(),
        3 => a.as_ref().conjugate().to_owned(),
        _ => return Err("unknown solve transform".into()),
    };
    let inverse = if m.min(n) == 0 {
        Mat::zeros(a.ncols(), a.nrows())
    } else {
        a.as_ref()
            .thin_svd(par.reborrow())
            .map_err(|e| format!("{e:?}"))?
            .pseudoinverse(par.reborrow())
    };
    let mut out = Mat::zeros(inverse.nrows(), b.ncols());
    faer::linalg::matmul::matmul(
        out.as_mut(),
        Accum::Replace,
        inverse.as_ref(),
        b,
        one::<T>(),
        par,
    );
    Ok(out)
}
#[julia]
pub struct TypedFactor {
    data: Option<FactorData>,
    info: usize,
}
impl TypedFactor {
    pub(crate) fn solve_ready(&self) -> Result<(), String> {
        self.data()?;
        if self.info != 0 {
            return Err("singular factorization".into());
        }
        Ok(())
    }
    pub(crate) fn data(&self) -> Result<&FactorData, String> {
        self.data
            .as_ref()
            .ok_or("factorization is not available".into())
    }
}
#[julia]
impl TypedFactor {
    #[julia]
    pub fn new() -> Self {
        Self {
            data: None,
            info: 0,
        }
    }
    #[julia]
    pub fn initialize(
        &mut self,
        source: &TypedMatrix,
        kind: i32,
        lower: bool,
        m: usize,
        n: usize,
        tag: i32,
        team: Option<&crate::teams::OwnedTeam>,
    ) -> Result<usize, String> {
        guarded(|| {
            crate::teams::with_team(team, |par| {
                self.initialize_with_par(source, kind, lower, par, m, n, tag)
            })
        })
    }

    #[julia]
    pub fn qr_blocksize(&self) -> Result<usize, String> {
        guarded(|| {
            let f = self.data()?;
            match f {
                FactorData::F32(x) => qr_rows(x),
                FactorData::F64(x) => qr_rows(x),
                FactorData::C32(x) => qr_rows(x),
                FactorData::C64(x) => qr_rows(x),
            }
        })
    }
    #[julia]
    pub fn clone_factor(&mut self, source: &TypedFactor) -> Result<(), String> {
        guarded(|| {
            let f = source;
            self.data = f.data.clone();
            self.info = f.info;
            Ok(())
        })
    }
    #[julia]
    pub fn permutation(&self, out: *mut usize, len: usize, column: bool) -> Result<(), String> {
        guarded(|| {
            let p = with_factor!(self.data()?,f=>match f{Factor::Lu(x)if !column=>x.P().arrays().0,Factor::FullLu(x)=>if column{x.Q().arrays().0}else{x.P().arrays().0},Factor::PivotQr(x)=>x.P().arrays().0,Factor::Lblt(x)=>x.P().arrays().0,_=>return Err("no permutation".into())});
            if len != p.len() {
                return Err("permutation size mismatch".into());
            }
            for (i, &x) in p.iter().enumerate() {
                unsafe {
                    out.add(i).write(x + 1);
                }
            }
            Ok(())
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn factor_part_from(
        &mut self,
        source: &TypedFactor,
        part: i32,
        team: Option<&crate::teams::OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            let f = source.data()?;
            let data = crate::teams::with_team(team, |par| {
                Ok(factor_matrix!(f,x=>factor_part(x,part,par)?))
            })?;
            self.data = Some(data);
            Ok(())
        })
    }
    #[julia]
    pub fn factor_solve_from(
        &mut self,
        source: &TypedFactor,
        rhs: &TypedMatrix,
        op: i32,
    ) -> Result<(), String> {
        guarded(|| {
            let f = source;
            if f.info != 0 {
                return Err("singular factorization".into());
            }
            let b = rhs.data()?;
            self.data = Some(factor_rhs!(f.data()?,b,x,y=>factor_solve(x,y.as_ref(),op,Par::Seq)?));
            Ok(())
        })
    }
    #[julia]
    pub fn inverse_from(
        &mut self,
        source: &TypedFactor,
        team: Option<&crate::teams::OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            let f = source;
            if f.info != 0 {
                return Err("singular factorization".into());
            }
            let data = f.data()?;
            self.data = Some(crate::teams::with_team(team, |par| {
                Ok(factor_matrix!(data,x=>{
                    if factor_dispatch!(x,f=>f.nrows()!=f.ncols()){return Err("inverse requires square input".into());}
                    let deficient=match x{Factor::Qr(f)=>rank_deficient(f.thin_R()),Factor::PivotQr(f)=>rank_deficient(f.thin_R()),_=>false};
                    if deficient{return Err("singular factorization".into());}
                    factor_dispatch!(x,f=>f.inverse(par))
                }))
            })?);
            Ok(())
        })
    }
}

macro_rules! paired {
    ($left:expr,$right:expr,$a:ident,$b:ident=>$body:expr) => {
        match ($left, $right) {
            (MatrixData::F32($a), MatrixData::F32($b)) => $body,
            (MatrixData::F64($a), MatrixData::F64($b)) => $body,
            (MatrixData::C32($a), MatrixData::C32($b)) => $body,
            (MatrixData::C64($a), MatrixData::C64($b)) => $body,
            _ => return Err("native scalar types must match".into()),
        }
    };
}
pub(crate) unsafe fn inner_output<T: ComplexField>(
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    out: *mut u8,
) -> Result<(), String> {
    if a.shape() != b.shape() {
        return Err("inner product dimensions must match".into());
    }
    let value = (0..a.ncols()).fold(zero::<T>(), |s, j| {
        add(
            &s,
            &faer::linalg::matmul::dot::inner_prod(
                a.col(j).transpose(),
                Conj::Yes,
                b.col(j),
                Conj::No,
            ),
        )
    });
    out.cast::<T>().write(value);
    Ok(())
}
pub(crate) fn hermitian_copy<T: ComplexField>(
    a: MatRef<'_, T>,
    lower: bool,
) -> Result<Mat<T>, String> {
    if a.nrows() != a.ncols() {
        return Err("self-adjoint input must be square".into());
    }
    let b = Mat::from_fn(a.nrows(), a.ncols(), |i, j| {
        if i == j {
            from_real::<T>(&real(&a[(i, i)]))
        } else if (lower && i >= j) || (!lower && i <= j) {
            copy(&a[(i, j)])
        } else {
            conj(&a[(j, i)])
        }
    });
    finite(b.as_ref())?;
    Ok(b)
}
pub(crate) unsafe fn eigen_output<T: ComplexField>(
    a: MatRef<'_, T>,
    values: *mut u8,
    vectors: *mut u8,
    compute_vectors: bool,
    hermitian: bool,
    lower: bool,
) -> Result<(), String> {
    if a.nrows() != a.ncols() {
        return Err("eigen input must be square".into());
    }
    let n = a.nrows();
    if n == 0 {
        return Ok(());
    }
    if hermitian {
        let a = hermitian_copy(a, lower)?;
        let vals = if compute_vectors {
            let f = a
                .as_ref()
                .self_adjoint_eigen(Side::Lower, Par::Seq)
                .map_err(|e| format!("{e:?}"))?;
            write_matrix(f.U(), vectors);
            (0..n).map(|i| real(&f.S()[i])).collect::<Vec<_>>()
        } else {
            a.as_ref()
                .self_adjoint_eigenvalues(Side::Lower, Par::Seq)
                .map_err(|e| format!("{e:?}"))?
        };
        for (i, x) in vals.into_iter().enumerate() {
            values.cast::<T::Real>().add(i).write(x);
        }
    } else {
        finite(a)?;
        let vals = if compute_vectors {
            let f = a.eigen(Par::Seq).map_err(|e| format!("{e:?}"))?;
            write_matrix(f.U(), vectors);
            (0..n).map(|i| copy(&f.S()[i])).collect::<Vec<_>>()
        } else {
            a.eigenvalues(Par::Seq).map_err(|e| format!("{e:?}"))?
        };
        for (i, x) in vals.into_iter().enumerate() {
            values
                .cast::<num_complex::Complex<T::Real>>()
                .add(i)
                .write(x);
        }
    }
    Ok(())
}
unsafe fn generalized_output<T: ComplexField>(
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    values: *mut u8,
    vectors: *mut u8,
    compute_vectors: bool,
    hermitian: bool,
    lower: bool,
    other_lower: bool,
) -> Result<(), String> {
    if a.nrows() != a.ncols() || a.shape() != b.shape() {
        return Err("generalized eigen input shapes must match and be square".into());
    }
    let n = a.nrows();
    if n == 0 {
        return Ok(());
    }
    if hermitian {
        let a = hermitian_copy(a, lower)?;
        let b = hermitian_copy(b, other_lower)?;
        let vals = if compute_vectors {
            let f = a
                .as_ref()
                .self_adjoint_generalized_eigen(b.as_ref(), Par::Seq)
                .map_err(|e| format!("{e:?}"))?;
            write_matrix(f.U(), vectors);
            f.values().to_vec()
        } else {
            a.as_ref()
                .self_adjoint_generalized_eigenvalues(b.as_ref(), Par::Seq)
                .map_err(|e| format!("{e:?}"))?
        };
        for (i, x) in vals.into_iter().enumerate() {
            values.cast::<T::Real>().add(i).write(x);
        }
    } else {
        finite(a)?;
        finite(b)?;
        let f = a
            .generalized_eigen(b, Par::Seq)
            .map_err(|e| format!("{e:?}"))?;
        if compute_vectors {
            write_matrix(f.U(), vectors);
        }
        for i in 0..n {
            let numerator = copy(&f.S_a()[i]);
            let denominator = copy(&f.S_b()[i]);
            let value = if denominator == zero::<num_complex::Complex<T::Real>>() {
                num_complex::Complex::<T::Real>::new(
                    if numerator == zero::<num_complex::Complex<T::Real>>() {
                        nan::<T::Real>()
                    } else {
                        infinity::<T::Real>()
                    },
                    zero::<T::Real>(),
                )
            } else {
                div(&numerator, &denominator)
            };
            values
                .cast::<num_complex::Complex<T::Real>>()
                .add(i)
                .write(value);
        }
    }
    Ok(())
}
fn triangular_check<T: ComplexField>(
    a: MatRef<'_, T>,
    lower: bool,
    unit: bool,
) -> Result<(), String> {
    if a.nrows() != a.ncols() {
        return Err("triangular matrix must be square".into());
    }
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            if (if lower { i >= j } else { i <= j }) && !(unit && i == j) {
                if !is_finite(&a[(i, j)]) {
                    return Err("input triangle must be finite".into());
                }
                if i == j && a[(i, j)] == zero::<T>() {
                    return Err("singular triangular matrix".into());
                }
            }
        }
    }
    Ok(())
}
fn triangular_inverse<T: ComplexField>(
    a: MatRef<'_, T>,
    lower: bool,
    unit: bool,
) -> Result<Mat<T>, String> {
    triangular_check(a, lower, unit)?;
    let mut x = Mat::zeros(a.nrows(), a.ncols());
    use faer::linalg::triangular_inverse::*;
    match (lower, unit) {
        (true, true) => invert_unit_lower_triangular(x.as_mut(), a, Par::Seq),
        (true, false) => invert_lower_triangular(x.as_mut(), a, Par::Seq),
        (false, true) => invert_unit_upper_triangular(x.as_mut(), a, Par::Seq),
        (false, false) => invert_upper_triangular(x.as_mut(), a, Par::Seq),
    }
    if unit {
        for i in 0..a.nrows() {
            x[(i, i)] = one::<T>();
        }
    }
    Ok(x)
}
fn triangular_solve<T: ComplexField>(
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    lower: bool,
    unit: bool,
) -> Result<Mat<T>, String> {
    triangular_check(a, lower, unit)?;
    finite(b)?;
    if a.nrows() != b.nrows() {
        return Err("triangular solve dimensions must match".into());
    }
    let mut x = b.to_owned();
    match (lower, unit) {
        (true, true) => a.solve_unit_lower_triangular_in_place(x.as_mut(), Par::Seq),
        (true, false) => a.solve_lower_triangular_in_place(x.as_mut(), Par::Seq),
        (false, true) => a.solve_unit_upper_triangular_in_place(x.as_mut(), Par::Seq),
        (false, false) => a.solve_upper_triangular_in_place(x.as_mut(), Par::Seq),
    }
    Ok(x)
}
unsafe fn pfaffian_output<T: ComplexField>(a: MatRef<'_, T>, out: *mut u8) -> Result<(), String> {
    if a.nrows() != a.ncols() {
        return Err("Pfaffian requires square input".into());
    }
    finite(a)?;
    for j in 0..a.ncols() {
        for i in j..a.nrows() {
            if a[(i, j)] != neg(&a[(j, i)]) {
                return Err("Pfaffian requires skew-symmetric input".into());
            }
        }
    }
    let mut memory =
        faer::dyn_stack::MemBuffer::new(faer::linalg::pfaffian::pfaffian_scratch::<T>(a.nrows()));
    let p =
        faer::linalg::pfaffian::pfaffian(a, Par::Seq, faer::dyn_stack::MemStack::new(&mut memory))
            .map_err(|e| format!("{e:?}"))?;
    out.cast::<T>().write(p);
    Ok(())
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn dot_buffer(&self, other: &TypedMatrix, out: *mut u8) -> Result<(), String> {
        guarded(|| {
            let b = other.data()?;
            unsafe { paired!(self.data()?,b,x,y=>inner_output(x.as_ref(),y.as_ref(),out)) }
        })
    }
    #[julia]
    pub fn eigen_typed_buffers(
        &self,
        values: *mut u8,
        vectors: *mut u8,
        compute_vectors: bool,
        hermitian: bool,
        lower: bool,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            dispatch!(self.data()?,x=>eigen_output(x.as_ref(),values,vectors,compute_vectors,hermitian,lower))
        })
    }
    #[julia]
    pub fn generalized_typed_buffers(
        &self,
        other: &TypedMatrix,
        values: *mut u8,
        vectors: *mut u8,
        compute_vectors: bool,
        hermitian: bool,
        lower: bool,
        other_lower: bool,
    ) -> Result<(), String> {
        guarded(|| {
            let b = other.data()?;
            unsafe {
                paired!(self.data()?,b,x,y=>generalized_output(x.as_ref(),y.as_ref(),values,vectors,compute_vectors,hermitian,lower,other_lower))
            }
        })
    }
    #[julia]
    pub fn triangular_inverse_typed(
        &mut self,
        source: &TypedMatrix,
        lower: bool,
        unit: bool,
    ) -> Result<(), String> {
        guarded(|| {
            let a = source.data()?;
            self.data = Some(mapped!(a,x=>triangular_inverse(x.as_ref(),lower,unit)?));
            Ok(())
        })
    }
    #[julia]
    pub fn triangular_solve_typed(
        &mut self,
        source: &TypedMatrix,
        rhs: &TypedMatrix,
        lower: bool,
        unit: bool,
    ) -> Result<(), String> {
        guarded(|| {
            let a = source.data()?;
            let b = rhs.data()?;
            self.data = Some(binary!(a,b,x,y=>triangular_solve(x.as_ref(),y.as_ref(),lower,unit)?));
            Ok(())
        })
    }
    #[julia]
    pub fn pfaffian_buffer(&self, out: *mut u8) -> Result<(), String> {
        guarded(|| unsafe { dispatch!(self.data()?,x=>pfaffian_output(x.as_ref(),out)) })
    }
}

trait NativeReal: faer::traits::RealField {
    fn logarithm(&self) -> Self;
}
impl NativeReal for f32 {
    fn logarithm(&self) -> Self {
        self.ln()
    }
}
impl NativeReal for f64 {
    fn logarithm(&self) -> Self {
        self.ln()
    }
}
unsafe fn determinant_parts<T: ComplexField>(
    f: &Factor<T>,
    log: *mut u8,
    phase: *mut u8,
) -> Result<(), String>
where
    T::Real: NativeReal,
{
    let n = factor_dispatch!(f,x=>x.nrows());
    if factor_dispatch!(f,x=>x.ncols()) != n {
        return Err("determinant requires square input".into());
    }
    let mut l = zero::<T::Real>();
    let mut p = one::<T>();
    let mut accumulate = |x: T| {
        let mag = abs(&x);
        l = add(&l, &mag.logarithm());
        p = if mag == zero::<T::Real>() {
            zero::<T>()
        } else {
            mul(&p, &div(&x, &from_real::<T>(&mag)))
        };
    };
    match f {
        Factor::Lu(x) => {
            for i in 0..n {
                accumulate(copy(&x.U()[(i, i)]));
            }
            p = mul(
                &p,
                &from_f64::<T>(super::permutation_sign(x.P().arrays().0)),
            );
        }
        Factor::FullLu(x) => {
            for i in 0..n {
                accumulate(copy(&x.U()[(i, i)]));
            }
            p = mul(
                &p,
                &from_f64::<T>(
                    super::permutation_sign(x.P().arrays().0)
                        * super::permutation_sign(x.Q().arrays().0),
                ),
            );
        }
        Factor::Llt(x) => {
            for i in 0..n {
                accumulate(copy(&x.L()[(i, i)]));
                accumulate(conj(&x.L()[(i, i)]));
            }
        }
        Factor::Ldlt(x) => {
            for i in 0..n {
                accumulate(copy(&x.D()[i]));
            }
        }
        Factor::Lblt(x) => {
            let mut i = 0;
            while i < n {
                if i + 1 < n && x.B_subdiag()[i] != zero::<T>() {
                    accumulate(sub(
                        &mul(&x.B_diag()[i], &x.B_diag()[i + 1]),
                        &mul(&x.B_subdiag()[i], &conj(&x.B_subdiag()[i])),
                    ));
                    i += 2;
                } else {
                    accumulate(copy(&x.B_diag()[i]));
                    i += 1;
                }
            }
        }
        _ => accumulate(
            factor_dispatch!(f,x=>x.reconstruct(Par::Seq))
                .as_ref()
                .determinant(Par::Seq),
        ),
    }
    log.cast::<T::Real>().write(l);
    phase.cast::<T>().write(p);
    Ok(())
}
#[julia]
impl TypedFactor {
    #[julia]
    pub fn determinant_parts(&self, log: *mut u8, phase: *mut u8) -> Result<(), String> {
        guarded(|| unsafe { with_factor!(self.data()?,f=>determinant_parts(f,log,phase)) })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn axis_sum_from(&mut self, source: &TypedMatrix, axis: i32) -> Result<(), String> {
        guarded(|| {
            let a = source.data()?;
            self.data = Some(mapped!(a,x=>match axis{
                1=>Mat::from_fn(1,x.ncols(),|_,j|x.as_ref().col(j).sum()),
                2=>Mat::from_fn(x.nrows(),1,|i,_|x.as_ref().row(i).sum()),
                _=>return Err("sum axis must be 1 or 2".into()),
            }));
            Ok(())
        })
    }
}

unsafe fn resize_matrix<T: ComplexField>(a: &mut Mat<T>, m: usize, n: usize, value: *const u8) {
    let x = &*value.cast::<T>();
    a.resize_with(m, n, |_, _| copy(x));
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn reserve_storage(&mut self, rows: usize, cols: usize) -> Result<(), String> {
        guarded(|| {
            rows.checked_mul(cols).ok_or("matrix capacity overflow")?;
            let a = self.data.as_mut().ok_or("matrix is not initialized")?;
            dispatch!(a,x=>x.try_reserve(rows,cols).map_err(|e|format!("{e:?}")))
        })
    }
    #[julia]
    pub fn resize_storage(
        &mut self,
        rows: usize,
        cols: usize,
        value: *const u8,
    ) -> Result<(), String> {
        guarded(|| {
            rows.checked_mul(cols).ok_or("matrix size overflow")?;
            let a = self.data.as_mut().ok_or("matrix is not initialized")?;
            unsafe {
                dispatch!(a,x=>resize_matrix(x,rows,cols,value));
            }
            Ok(())
        })
    }
    #[julia]
    pub fn truncate_storage(&mut self, rows: usize, cols: usize) -> Result<(), String> {
        guarded(|| {
            let a = self.data.as_mut().ok_or("matrix is not initialized")?;
            dispatch!(a,x=>{
                if rows>x.nrows()||cols>x.ncols(){return Err("truncate cannot grow a matrix".into());}x.truncate(rows,cols);
            });
            Ok(())
        })
    }
    #[julia]
    pub fn identity_storage(&mut self, rows: usize, cols: usize, tag: i32) -> Result<(), String> {
        guarded(|| {
            rows.checked_mul(cols).ok_or("matrix size overflow")?;
            self.data = Some(match tag {
                0 => MatrixData::F32(Mat::identity(rows, cols)),
                1 => MatrixData::F64(Mat::identity(rows, cols)),
                2 => MatrixData::C32(Mat::identity(rows, cols)),
                3 => MatrixData::C64(Mat::identity(rows, cols)),
                _ => return Err("unsupported scalar type".into()),
            });
            Ok(())
        })
    }
}

impl TypedFactor {
    fn initialize_with_par(
        &mut self,
        source: &TypedMatrix,
        kind: i32,
        lower: bool,
        par: Par<'_>,
        m: usize,
        n: usize,
        tag: i32,
    ) -> Result<usize, String> {
        let a = source.data()?;
        let (shape, actual_tag) = match a {
            MatrixData::F32(x) => (x.shape(), 0),
            MatrixData::F64(x) => (x.shape(), 1),
            MatrixData::C32(x) => (x.shape(), 2),
            MatrixData::C64(x) => (x.shape(), 3),
        };
        if actual_tag != tag || shape != (m, n) {
            return Err(
                "factor input native scalar type/dimensions do not match requested metadata".into(),
            );
        }

        let a = match a {
            MatrixData::F32(x) => Borrowed::F32(x.as_ref()),
            MatrixData::F64(x) => Borrowed::F64(x.as_ref()),
            MatrixData::C32(x) => Borrowed::C32(x.as_ref()),
            MatrixData::C64(x) => Borrowed::C64(x.as_ref()),
        };
        self.initialize_ref(a, kind, lower, par)
    }
    /// Factor an owned matrix or a borrowed view into owned storage.
    pub(crate) fn initialize_ref(
        &mut self,
        a: Borrowed<'_>,
        kind: i32,
        lower: bool,
        par: Par<'_>,
    ) -> Result<usize, String> {
        macro_rules! new {
            ($x:expr, $v:ident) => {{
                let (f, i) = factor_new($x, kind, lower, par)?;
                self.info = i;
                f.map(FactorData::$v)
            }};
        }
        self.data = match a {
            Borrowed::F32(x) => new!(x, F32),
            Borrowed::F64(x) => new!(x, F64),
            Borrowed::C32(x) => new!(x, C32),
            Borrowed::C64(x) => new!(x, C64),
        };
        Ok(self.info)
    }
}
