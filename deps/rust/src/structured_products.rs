//! Structured multiplication through `matmul::triangular`. Operands are borrowed
//! from native owners, ranges or preserved Julia buffers via view descriptors;
//! `*_into` writes the destination owner's range directly. Nothing is retained.
use crate::collections::{VectorData, VectorOwner, VectorStore};
use crate::typed::{MatrixData, TypedMatrix};
use crate::views::{borrowed_mut, source, Borrowed, Spec};
use crate::{guarded, OwnedMatrix};
use faer::linalg::matmul::triangular::{self, BlockStructure as B};
use faer::traits::{math_utils::*, ComplexField};
use faer::{Accum, Col, Conj, Mat, MatMut, MatRef, Par};
use rustcall_julia_macros::julia;

/// Operand tags: 0 dense, 1 lower, 2 upper, 3 unit lower, 4 unit upper and
/// 5 a diagonal stored as an n×1 column. Destination tags: 0 full, 1 lower
/// and 2 upper including the diagonal, 6 strict lower and 7 strict upper.
fn structure(tag: i32) -> Result<B, String> {
    Ok(match tag {
        0 => B::Rectangular,
        1 => B::TriangularLower,
        2 => B::TriangularUpper,
        3 => B::UnitTriangularLower,
        4 => B::UnitTriangularUpper,
        6 => B::StrictTriangularLower,
        7 => B::StrictTriangularUpper,
        _ => return Err("invalid structure".into()),
    })
}
fn in_block(tag: i32, i: usize, j: usize) -> bool {
    match tag {
        1 => i >= j,
        2 => i <= j,
        6 => i > j,
        7 => i < j,
        _ => true,
    }
}
fn conj_of(c: bool) -> Conj {
    if c {
        Conj::Yes
    } else {
        Conj::No
    }
}
/// Logical entry of a structured operand: ignored halves are zero and unit
/// diagonals are one, so unused stored values are never read.
fn entry<T: ComplexField>(x: MatRef<'_, T>, c: bool, tag: i32, i: usize, j: usize) -> T {
    let ignored = match tag {
        1 | 3 => i < j,
        2 | 4 => i > j,
        5 => i != j,
        _ => false,
    };
    if ignored {
        return zero::<T>();
    }
    if (tag == 3 || tag == 4) && i == j {
        return one::<T>();
    }
    let v = if tag == 5 { &x[(i, 0)] } else { &x[(i, j)] };
    if c {
        conj(v)
    } else {
        copy(v)
    }
}
fn shape<T>(x: MatRef<'_, T>, tag: i32) -> (usize, usize) {
    if tag == 5 {
        (x.nrows(), x.nrows())
    } else {
        x.shape()
    }
}
/// `dst = alpha * op(a) * op(b) + beta * dst` on the destination block, where
/// `beta = None` replaces without reading the destination.
#[allow(clippy::too_many_arguments)]
fn product_into<T: ComplexField>(
    mut dst: MatMut<'_, T>,
    ds: i32,
    (a, ca, sa): (MatRef<'_, T>, bool, i32),
    (b, cb, sb): (MatRef<'_, T>, bool, i32),
    alpha: T,
    beta: Option<T>,
) -> Result<(), String> {
    let dstructure = structure(ds)?;
    let ((m, k), (k2, n)) = (shape(a, sa), shape(b, sb));
    if sa == 5 && a.ncols() != 1 || sb == 5 && b.ncols() != 1 {
        return Err("diagonal operand must be stored as a column".into());
    }
    if k != k2 {
        return Err("structured product inner dimensions must match".into());
    }
    if (sa != 0 && m != k) || (sb != 0 && k != n) || (ds != 0 && m != n) {
        return Err("structured operand must be square".into());
    }
    if dst.shape() != (m, n) {
        return Err("structured product destination size must match".into());
    }
    // Pre-scale only the touched block; zero alpha never reads the operands.
    let accum = match &beta {
        None => Accum::Replace,
        Some(x) if *x == one::<T>() => Accum::Add,
        Some(x) => {
            for j in 0..n {
                for i in 0..m {
                    if in_block(ds, i, j) {
                        dst[(i, j)] = if *x == zero::<T>() {
                            zero::<T>()
                        } else {
                            mul(x, &dst[(i, j)])
                        };
                    }
                }
            }
            Accum::Add
        }
    };
    if alpha == zero::<T>() {
        if let Accum::Replace = accum {
            for j in 0..n {
                for i in 0..m {
                    if in_block(ds, i, j) {
                        dst[(i, j)] = zero::<T>();
                    }
                }
            }
        }
        return Ok(());
    }
    if sa == 5 || sb == 5 {
        for j in 0..n {
            for i in 0..m {
                if !in_block(ds, i, j) {
                    continue;
                }
                // Diagonal entries live in column zero; `entry` masks the rest.
                let v = if sa == 5 {
                    mul(&entry(a, ca, 5, i, i), &entry(b, cb, sb, i, j))
                } else {
                    mul(&entry(a, ca, sa, i, j), &entry(b, cb, 5, j, j))
                };
                let v = mul(&alpha, &v);
                dst[(i, j)] = match accum {
                    Accum::Replace => v,
                    Accum::Add => add(&dst[(i, j)], &v),
                };
            }
        }
        return Ok(());
    }
    triangular::matmul_with_conj(
        dst,
        dstructure,
        accum,
        a,
        structure(sa)?,
        conj_of(ca),
        b,
        structure(sb)?,
        conj_of(cb),
        alpha,
        Par::Seq,
    );
    Ok(())
}
unsafe fn scalar<T: ComplexField>(p: *const u8) -> T {
    copy(&*p.cast::<T>())
}
/// Operand descriptors and tags shared by the entry points.
#[derive(Clone, Copy)]
struct Operands {
    p: *const u8,
    pd: Spec,
    sa: i32,
    q: *const u8,
    qd: Spec,
    sb: i32,
}
impl Operands {
    unsafe fn read(
        p: *const u8,
        pd: *const isize,
        sa: i32,
        q: *const u8,
        qd: *const isize,
        sb: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            p,
            pd: Spec::read(pd)?,
            sa,
            q,
            qd: Spec::read(qd)?,
            sb,
        })
    }
}
macro_rules! with_operands {
    ($o:expr, $a:ident, $b:ident, $T:ident => $body:expr) => {
        match (source($o.p, $o.pd)?, source($o.q, $o.qd)?) {
            (Borrowed::F32($a), Borrowed::F32($b)) => {
                type $T = f32;
                $body
            }
            (Borrowed::F64($a), Borrowed::F64($b)) => {
                type $T = f64;
                $body
            }
            (Borrowed::C32($a), Borrowed::C32($b)) => {
                type $T = faer::c32;
                $body
            }
            (Borrowed::C64($a), Borrowed::C64($b)) => {
                type $T = faer::c64;
                $body
            }
            _ => return Err("structured operand scalar types must match".into()),
        }
    };
}
fn fresh<T: ComplexField>(
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    o: &Operands,
) -> Result<Mat<T>, String> {
    let (m, _) = shape(a, o.sa);
    let (_, n) = shape(b, o.sb);
    let mut out = Mat::<T>::zeros(m, n);
    product_into(
        out.as_mut(),
        0,
        (a, o.pd.conjugate, o.sa),
        (b, o.qd.conjugate, o.sb),
        one::<T>(),
        None,
    )?;
    Ok(out)
}
unsafe fn fresh_data(o: Operands) -> Result<MatrixData, String> {
    Ok(with_operands!(o, a, b, T => {
        let x: Mat<T> = fresh(a, b, &o)?;
        T::wrap(x)
    }))
}
trait Wrap: Sized {
    fn wrap(x: Mat<Self>) -> MatrixData;
    fn wrap_col(x: Col<Self>) -> VectorData;
}
macro_rules! wrap {
    ($T:ty, $V:ident) => {
        impl Wrap for $T {
            fn wrap(x: Mat<Self>) -> MatrixData {
                MatrixData::$V(x)
            }
            fn wrap_col(x: Col<Self>) -> VectorData {
                VectorData::$V(VectorStore::Column(x))
            }
        }
    };
}
wrap!(f32, F32);
wrap!(f64, F64);
wrap!(faer::c32, C32);
wrap!(faer::c64, C64);
#[allow(clippy::too_many_arguments)]
unsafe fn into<T: ComplexField>(
    dst: MatMut<'_, T>,
    dd: Spec,
    ds: i32,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    o: &Operands,
    alpha: *const u8,
    beta: *const u8,
) -> Result<(), String> {
    let dst = borrowed_mut(dst, dd)?;
    let beta = if beta.is_null() {
        None
    } else {
        Some(scalar::<T>(beta))
    };
    product_into(
        dst,
        ds,
        (a, o.pd.conjugate, o.sa),
        (b, o.qd.conjugate, o.sb),
        scalar::<T>(alpha),
        beta,
    )
}
macro_rules! into_owner {
    ($dst:expr, $dd:expr, $ds:expr, $o:expr, $alpha:expr, $beta:expr) => {
        with_operands!($o, a, b, T => {
            let d: MatMut<'_, T> = match $dst.pick_mut() {
                Some(d) => d,
                None => return Err("destination and operand scalar types must match".into()),
            };
            into(d, $dd, $ds, a, b, &$o, $alpha, $beta)
        })
    };
}
trait PickDst {
    fn pick_mut<T: 'static>(&mut self) -> Option<MatMut<'_, T>>;
}
fn cast_mut<S: 'static, T: 'static>(x: MatMut<'_, S>) -> Option<MatMut<'_, T>> {
    if std::any::TypeId::of::<S>() == std::any::TypeId::of::<T>() {
        // SAFETY: S and T are the same type.
        Some(unsafe { std::mem::transmute::<MatMut<'_, S>, MatMut<'_, T>>(x) })
    } else {
        None
    }
}
impl PickDst for MatrixData {
    fn pick_mut<T: 'static>(&mut self) -> Option<MatMut<'_, T>> {
        match self {
            MatrixData::F32(x) => cast_mut(x.as_mut()),
            MatrixData::F64(x) => cast_mut(x.as_mut()),
            MatrixData::C32(x) => cast_mut(x.as_mut()),
            MatrixData::C64(x) => cast_mut(x.as_mut()),
        }
    }
}
impl PickDst for Mat<f64> {
    fn pick_mut<T: 'static>(&mut self) -> Option<MatMut<'_, T>> {
        cast_mut(self.as_mut())
    }
}
impl PickDst for VectorData {
    fn pick_mut<T: 'static>(&mut self) -> Option<MatMut<'_, T>> {
        match self {
            VectorData::F32(x) => cast_mut(x.as_mat_mut()),
            VectorData::F64(x) => cast_mut(x.as_mat_mut()),
            VectorData::C32(x) => cast_mut(x.as_mat_mut()),
            VectorData::C64(x) => cast_mut(x.as_mat_mut()),
        }
    }
}

#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn structured_product(
        &mut self,
        p: *const u8,
        pd: *const isize,
        sa: i32,
        q: *const u8,
        qd: *const isize,
        sb: i32,
    ) -> Result<(), String> {
        guarded(|| {
            match unsafe { fresh_data(Operands::read(p, pd, sa, q, qd, sb)?)? } {
                MatrixData::F64(x) => self.matrix = x,
                _ => return Err("Float64 output requires Float64 operands".into()),
            }
            Ok(())
        })
    }
    #[julia]
    pub fn structured_product_into(
        &mut self,
        dd: *const isize,
        ds: i32,
        p: *const u8,
        pd: *const isize,
        sa: i32,
        q: *const u8,
        qd: *const isize,
        sb: i32,
        alpha: *const u8,
        beta: *const u8,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            let o = Operands::read(p, pd, sa, q, qd, sb)?;
            let dd = Spec::read(dd)?;
            into_owner!(self.matrix, dd, ds, o, alpha, beta)
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn structured_product(
        &mut self,
        p: *const u8,
        pd: *const isize,
        sa: i32,
        q: *const u8,
        qd: *const isize,
        sb: i32,
    ) -> Result<(), String> {
        guarded(|| {
            self.data = Some(unsafe { fresh_data(Operands::read(p, pd, sa, q, qd, sb)?)? });
            Ok(())
        })
    }
    #[julia]
    pub fn structured_product_into(
        &mut self,
        dd: *const isize,
        ds: i32,
        p: *const u8,
        pd: *const isize,
        sa: i32,
        q: *const u8,
        qd: *const isize,
        sb: i32,
        alpha: *const u8,
        beta: *const u8,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            let o = Operands::read(p, pd, sa, q, qd, sb)?;
            let dd = Spec::read(dd)?;
            let data = self.data.as_mut().ok_or("matrix is not initialized")?;
            into_owner!(data, dd, ds, o, alpha, beta)
        })
    }
}
#[julia]
impl VectorOwner {
    /// A structured matrix times a column, returned as a native column.
    #[julia]
    pub fn structured_vector(
        &mut self,
        p: *const u8,
        pd: *const isize,
        sa: i32,
        q: *const u8,
        qd: *const isize,
    ) -> Result<(), String> {
        guarded(|| {
            let o = unsafe { Operands::read(p, pd, sa, q, qd, 0)? };
            self.data = Some(unsafe {
                with_operands!(o, a, b, T => {
                    if b.ncols() != 1 {
                        return Err("vector operand must have one column".into());
                    }
                    let x: Mat<T> = fresh(a, b, &o)?;
                    T::wrap_col(x.col(0).to_owned())
                })
            });
            Ok(())
        })
    }
    #[julia]
    pub fn structured_product_into(
        &mut self,
        dd: *const isize,
        ds: i32,
        p: *const u8,
        pd: *const isize,
        sa: i32,
        q: *const u8,
        qd: *const isize,
        sb: i32,
        alpha: *const u8,
        beta: *const u8,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            let o = Operands::read(p, pd, sa, q, qd, sb)?;
            let dd = Spec::read(dd)?;
            let data = self.data.as_mut().ok_or("collection is not initialized")?;
            into_owner!(data, dd, ds, o, alpha, beta)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn raw(m: usize, n: usize, tag: isize) -> [isize; 9] {
        [4, tag, 1, m as isize, 0, m as isize, n as isize, 0, 0]
    }
    #[test]
    fn borrowed_buffers_ignore_unused_entries() {
        let a = [1.0f64, 2.0, f64::NAN, 4.0]; // lower: [1 0; 2 4]
        let b = [1.0f64, 1.0, 2.0, 3.0];
        let (ad, bd) = (raw(2, 2, 1), raw(2, 2, 1));
        let mut out = OwnedMatrix::new();
        out.structured_product(
            a.as_ptr().cast(),
            ad.as_ptr(),
            1,
            b.as_ptr().cast(),
            bd.as_ptr(),
            0,
        )
        .unwrap();
        assert_eq!(out.matrix[(0, 1)], 2.0);
        assert_eq!(out.matrix[(1, 1)], 16.0);
        let u = [f64::NAN, 2.0, 5.0, f64::NAN]; // unit upper: [1 5; 0 1]
        out.structured_product(
            u.as_ptr().cast(),
            ad.as_ptr(),
            4,
            b.as_ptr().cast(),
            bd.as_ptr(),
            0,
        )
        .unwrap();
        assert_eq!(out.matrix[(0, 0)], 6.0);
        assert_eq!(out.matrix[(1, 0)], 1.0);
        let d = [2.0f64, 3.0];
        let dd = raw(2, 1, 1);
        out.structured_product(
            d.as_ptr().cast(),
            dd.as_ptr(),
            5,
            a.as_ptr().cast(),
            ad.as_ptr(),
            1,
        )
        .unwrap();
        assert_eq!(
            (out.matrix[(0, 1)], out.matrix[(1, 0)], out.matrix[(1, 1)]),
            (0.0, 6.0, 12.0)
        );
        assert!(out
            .structured_product(
                a.as_ptr().cast(),
                ad.as_ptr(),
                1,
                d.as_ptr().cast(),
                dd.as_ptr(),
                0
            )
            .is_ok());
        let bad = raw(3, 1, 1);
        assert!(out
            .structured_product(
                a.as_ptr().cast(),
                ad.as_ptr(),
                0,
                d.as_ptr().cast(),
                bad.as_ptr(),
                0
            )
            .is_err());
    }
    #[test]
    fn accumulate_into_destination_blocks() {
        let a = [1.0f64, 2.0, 3.0, 4.0];
        let ad = raw(2, 2, 1);
        let mut c = OwnedMatrix::new();
        c.matrix = Mat::from_fn(2, 2, |_, _| f64::NAN);
        let whole = [0isize, 0, 1, 2, 0, 1, 2, 0, 0];
        let (alpha, beta) = (2.0f64, 0.0f64);
        // Zero beta overwrites NaN; strict lower touches only (1, 0).
        c.structured_product_into(
            whole.as_ptr(),
            6,
            a.as_ptr().cast(),
            ad.as_ptr(),
            0,
            a.as_ptr().cast(),
            ad.as_ptr(),
            0,
            (&alpha as *const f64).cast(),
            (&beta as *const f64).cast(),
        )
        .unwrap();
        assert_eq!(c.matrix[(1, 0)], 2.0 * (2.0 * 1.0 + 4.0 * 2.0));
        assert!(c.matrix[(0, 0)].is_nan() && c.matrix[(0, 1)].is_nan());
        let zero = 0.0f64;
        c.structured_product_into(
            whole.as_ptr(),
            0,
            a.as_ptr().cast(),
            ad.as_ptr(),
            1,
            a.as_ptr().cast(),
            ad.as_ptr(),
            0,
            (&zero as *const f64).cast(),
            std::ptr::null(),
        )
        .unwrap();
        assert_eq!(c.matrix, Mat::<f64>::zeros(2, 2));
        let mut wrong = TypedMatrix {
            data: Some(MatrixData::F32(Mat::zeros(2, 2))),
        };
        assert!(wrong
            .structured_product_into(
                whole.as_ptr(),
                0,
                a.as_ptr().cast(),
                ad.as_ptr(),
                0,
                a.as_ptr().cast(),
                ad.as_ptr(),
                0,
                (&alpha as *const f64).cast(),
                std::ptr::null(),
            )
            .is_err());
        let transposed = [0isize, 0, 1, 2, 0, 1, 2, 1, 0];
        assert!(c
            .structured_product_into(
                transposed.as_ptr(),
                0,
                a.as_ptr().cast(),
                ad.as_ptr(),
                0,
                a.as_ptr().cast(),
                ad.as_ptr(),
                0,
                (&alpha as *const f64).cast(),
                std::ptr::null(),
            )
            .is_err());
    }
}
