//! Native diagonal scaling of dense or triangular operands. The diagonal is
//! borrowed as its stored column (or a borrowed Julia buffer, owner kind 4) and never
//! expanded to an `n x n` matrix.
use crate::collections::{VectorData, VectorOwner, VectorStore};
use crate::typed::{MatrixData, TypedMatrix};
use crate::views::{borrowed, source, write_view, Borrowed, Spec};
use crate::{guarded, OwnedMatrix};
use faer::traits::{math_utils::*, ComplexField};
use faer::{Col, Mat, MatMut, MatRef};
use rustcall_julia_macros::julia;

trait Scalar: ComplexField + Sized {
    fn pick(b: Borrowed<'_>) -> Option<MatRef<'_, Self>>;
}
macro_rules! scalar {
    ($t:ty, $v:ident) => {
        impl Scalar for $t {
            fn pick(b: Borrowed<'_>) -> Option<MatRef<'_, Self>> {
                match b {
                    Borrowed::$v(x) => Some(x),
                    _ => None,
                }
            }
        }
    };
}
scalar!(f32, F32);
scalar!(f64, F64);
scalar!(faer::c32, C32);
scalar!(faer::c64, C64);
#[derive(Clone, Copy)]
struct Operands {
    dp: *const u8,
    ds: Spec,
    q: *const u8,
    qs: Spec,
    left: bool,
    structure: i32,
}
impl Operands {
    // SAFETY: Julia preserves both owners and descriptors for this call.
    unsafe fn read(
        dp: *const u8,
        dd: *const isize,
        q: *const u8,
        qd: *const isize,
        left: bool,
        structure: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            dp,
            ds: Spec::read(dd)?,
            q,
            qs: Spec::read(qd)?,
            left,
            structure,
        })
    }
    /// Borrow the stored diagonal as an `n x 1` column.
    unsafe fn diagonal<'a, T: Scalar>(&self) -> Result<MatRef<'a, T>, String> {
        let s = self.ds;
        if s.n != 1 || s.transpose || s.conjugate {
            return Err("diagonal descriptor must be one plain column".into());
        }
        T::pick(source(self.dp, s)?).ok_or_else(|| "diagonal scalar type must match".into())
    }
    /// A mutable destination must never be read through the raw operand
    /// pointers; Julia snapshots aliased operands before calling.
    fn reject_alias<D>(&self, dest: &D) -> Result<(), String> {
        let dest = (dest as *const D).cast::<u8>();
        if std::ptr::eq(self.q, dest) || std::ptr::eq(self.dp, dest) {
            return Err("diagonal product destination aliases an operand".into());
        }
        Ok(())
    }
    unsafe fn operand<'a, T: Scalar>(&self) -> Result<MatRef<'a, T>, String> {
        T::pick(source(self.q, self.qs)?)
            .ok_or_else(|| "diagonal product scalar types must match".into())
    }
    /// Scale rows (`left`) or columns of the operand by `alpha * diagonal`.
    /// Nonzero `structure` reads a square operand as faer's triangular block
    /// structures: the unused half is zero and a unit diagonal is one.
    unsafe fn scaled<T: Scalar>(&self, alpha: Option<&T>) -> Result<Mat<T>, String> {
        let (d, b) = (self.diagonal::<T>()?, self.operand::<T>()?);
        let (m, n) = b.shape();
        let structure = self.structure;
        if d.nrows() != if self.left { m } else { n } {
            return Err("diagonal product dimensions must match".into());
        }
        if !(0..=4).contains(&structure) || (structure != 0 && m != n) {
            return Err("invalid structured diagonal operand".into());
        }
        // Folding alpha into an O(n) diagonal keeps the product a single pass.
        let folded;
        let d = match alpha {
            Some(alpha) => {
                folded = Col::from_fn(d.nrows(), |i| mul(alpha, &d[(i, 0)]));
                folded.as_ref()
            }
            None => d.col(0),
        };
        let conjugate = self.qs.conjugate;
        if structure == 0 && !conjugate {
            let d = d.as_diagonal();
            return Ok(if self.left { d * b } else { b * d });
        }
        let lower = structure == 1 || structure == 3;
        let unit = structure >= 3;
        Ok(Mat::from_fn(m, n, |i, j| {
            let x = if unit && i == j {
                one::<T>()
            } else if structure != 0 && (if lower { i < j } else { i > j }) {
                zero::<T>()
            } else if conjugate {
                conj(&b[(i, j)])
            } else {
                copy(&b[(i, j)])
            };
            mul(&x, &d[if self.left { i } else { j }])
        }))
    }
    unsafe fn scaled_data(&self) -> Result<MatrixData, String> {
        Ok(match source(self.q, self.qs)? {
            Borrowed::F32(_) => MatrixData::F32(self.scaled(None)?),
            Borrowed::F64(_) => MatrixData::F64(self.scaled(None)?),
            Borrowed::C32(_) => MatrixData::C32(self.scaled(None)?),
            Borrowed::C64(_) => MatrixData::C64(self.scaled(None)?),
        })
    }
    /// `dest = alpha * product + beta * dest`. Every operand and destination
    /// read finishes before the destination is written. Julia snapshots a dense
    /// operand sharing the destination owner before this call.
    unsafe fn scale_into<T: Scalar>(
        &self,
        dest: MatMut<'_, T>,
        s: Spec,
        alpha: *const u8,
        beta: *const u8,
        alpha_zero: bool,
        beta_zero: bool,
    ) -> Result<(), String> {
        let shape = if s.transpose { (s.n, s.m) } else { (s.m, s.n) };
        let mut r = if alpha_zero {
            Mat::zeros(shape.0, shape.1)
        } else {
            let alpha = &*alpha.cast::<T>();
            self.scaled((alpha != &one::<T>()).then_some(alpha))?
        };
        if r.shape() != shape {
            return Err("diagonal product destination size must match".into());
        }
        if !beta_zero {
            let beta = &*beta.cast::<T>();
            let current = borrowed(dest.as_ref(), s)?;
            for j in 0..shape.1 {
                for i in 0..shape.0 {
                    r[(i, j)] = add(&r[(i, j)], &mul(beta, &current[(i, j)]));
                }
            }
        }
        write_view(dest, s, r.as_ref(), s.conjugate)
    }
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn diagonal_scaled(
        &mut self,
        dp: *const u8,
        dd: *const isize,
        q: *const u8,
        qd: *const isize,
        left: bool,
        structure: i32,
    ) -> Result<(), String> {
        guarded(|| {
            let o = unsafe { Operands::read(dp, dd, q, qd, left, structure)? };
            self.matrix = match unsafe { o.scaled_data()? } {
                MatrixData::F64(a) => a,
                _ => return Err("Float64 output requires Float64 operands".into()),
            };
            Ok(())
        })
    }
    #[julia]
    pub fn diagonal_scale_into(
        &mut self,
        desc: *const isize,
        dp: *const u8,
        dd: *const isize,
        q: *const u8,
        qd: *const isize,
        left: bool,
        structure: i32,
        alpha: *const u8,
        beta: *const u8,
        alpha_zero: bool,
        beta_zero: bool,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            let o = Operands::read(dp, dd, q, qd, left, structure)?;
            o.reject_alias(self)?;
            let s = Spec::read(desc)?;
            o.scale_into::<f64>(self.matrix.as_mut(), s, alpha, beta, alpha_zero, beta_zero)
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn diagonal_scaled(
        &mut self,
        dp: *const u8,
        dd: *const isize,
        q: *const u8,
        qd: *const isize,
        left: bool,
        structure: i32,
    ) -> Result<(), String> {
        guarded(|| {
            let o = unsafe { Operands::read(dp, dd, q, qd, left, structure)? };
            self.data = Some(unsafe { o.scaled_data()? });
            Ok(())
        })
    }
    #[julia]
    pub fn diagonal_scale_into(
        &mut self,
        desc: *const isize,
        dp: *const u8,
        dd: *const isize,
        q: *const u8,
        qd: *const isize,
        left: bool,
        structure: i32,
        alpha: *const u8,
        beta: *const u8,
        alpha_zero: bool,
        beta_zero: bool,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            let o = Operands::read(dp, dd, q, qd, left, structure)?;
            o.reject_alias(self)?;
            let s = Spec::read(desc)?;
            let (az, bz) = (alpha_zero, beta_zero);
            match self.data.as_mut().ok_or("matrix is not initialized")? {
                MatrixData::F32(a) => o.scale_into(a.as_mut(), s, alpha, beta, az, bz),
                MatrixData::F64(a) => o.scale_into(a.as_mut(), s, alpha, beta, az, bz),
                MatrixData::C32(a) => o.scale_into(a.as_mut(), s, alpha, beta, az, bz),
                MatrixData::C64(a) => o.scale_into(a.as_mut(), s, alpha, beta, az, bz),
            }
        })
    }
}
fn collected<T: ComplexField>(a: Mat<T>, kind: i32) -> Result<VectorStore<T>, String> {
    if a.nrows() != 1 && a.ncols() != 1 {
        return Err("diagonal vector product must have one row or column".into());
    }
    let m = a.nrows();
    VectorStore::new(kind, m * a.ncols(), |i| copy(&a[(i % m, i / m)]))
}
#[julia]
impl VectorOwner {
    /// `D * column` or `row * D`, returned as a native column or row.
    #[julia]
    pub fn diagonal_scaled_vector(
        &mut self,
        dp: *const u8,
        dd: *const isize,
        q: *const u8,
        qd: *const isize,
        left: bool,
        kind: i32,
    ) -> Result<(), String> {
        guarded(|| {
            let o = unsafe { Operands::read(dp, dd, q, qd, left, 0)? };
            self.data = Some(match unsafe { o.scaled_data()? } {
                MatrixData::F32(a) => VectorData::F32(collected(a, kind)?),
                MatrixData::F64(a) => VectorData::F64(collected(a, kind)?),
                MatrixData::C32(a) => VectorData::C32(collected(a, kind)?),
                MatrixData::C64(a) => VectorData::C64(collected(a, kind)?),
            });
            Ok(())
        })
    }
    #[julia]
    pub fn diagonal_scale_into(
        &mut self,
        desc: *const isize,
        dp: *const u8,
        dd: *const isize,
        q: *const u8,
        qd: *const isize,
        left: bool,
        structure: i32,
        alpha: *const u8,
        beta: *const u8,
        alpha_zero: bool,
        beta_zero: bool,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            let o = Operands::read(dp, dd, q, qd, left, structure)?;
            o.reject_alias(self)?;
            let s = Spec::read(desc)?;
            let (az, bz) = (alpha_zero, beta_zero);
            match self.data.as_mut().ok_or("collection is not initialized")? {
                VectorData::F32(a) => o.scale_into(a.as_mat_mut(), s, alpha, beta, az, bz),
                VectorData::F64(a) => o.scale_into(a.as_mat_mut(), s, alpha, beta, az, bz),
                VectorData::C32(a) => o.scale_into(a.as_mat_mut(), s, alpha, beta, az, bz),
                VectorData::C64(a) => o.scale_into(a.as_mat_mut(), s, alpha, beta, az, bz),
            }
        })
    }
}
