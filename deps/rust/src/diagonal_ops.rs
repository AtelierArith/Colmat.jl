//! Native diagonal products, solves, sums and in-place updates. The diagonal is
//! borrowed as a faer `DiagRef` from a native owner or a preserved Julia buffer;
//! dense operands are borrowed through view descriptors. Nothing is retained.
use crate::collections::{VectorData, VectorOwner, VectorStore};
use crate::typed::{MatrixData, TypedMatrix};
use crate::views::{source, Borrowed, Spec};
use crate::{guarded, OwnedMatrix};
use faer::traits::{math_utils::*, ComplexField};
use faer::{c32, c64, Col, ColRef, Mat, MatMut, MatRef, Row};
use rustcall_julia_macros::julia;

pub(crate) trait Pick: ComplexField + Sized {
    fn pick(v: &VectorData) -> Option<&VectorStore<Self>>;
}
macro_rules! pick {
    ($T:ty, $V:ident) => {
        impl Pick for $T {
            fn pick(v: &VectorData) -> Option<&VectorStore<Self>> {
                match v {
                    VectorData::$V(x) => Some(x),
                    _ => None,
                }
            }
        }
    };
}
pick!(f32, F32);
pick!(f64, F64);
pick!(c32, C32);
pick!(c64, C64);

/// A diagonal operand: a native `FaerDiagonal` owner or a Julia buffer whose
/// element type the Julia layer has already converted to the operand type.
#[derive(Clone, Copy)]
enum DiagSource<'a> {
    Owner(&'a VectorOwner),
    Values(*const u8, usize),
}
unsafe fn diagonal<'a, T: Pick>(d: DiagSource<'a>) -> Result<ColRef<'a, T>, String> {
    match d {
        DiagSource::Owner(o) => match T::pick(o.data()?) {
            Some(VectorStore::Diagonal(x)) => Ok(x.column_vector()),
            Some(_) => Err("diagonal operand required".into()),
            None => Err("diagonal and operand scalar types must match".into()),
        },
        DiagSource::Values(_, 0) => Ok(ColRef::from_slice(&[])),
        DiagSource::Values(p, n) => Ok(ColRef::from_slice(std::slice::from_raw_parts(
            p.cast::<T>(),
            n,
        ))),
    }
}
/// One-based index of the first exactly zero diagonal entry, or zero.
fn singular<T: ComplexField>(d: ColRef<'_, T>) -> usize {
    (0..d.nrows())
        .find(|&i| d[i] == zero::<T>())
        .map_or(0, |i| i + 1)
}
fn element<T: ComplexField>(a: MatRef<'_, T>, conjugate: bool, i: usize, j: usize) -> T {
    if conjugate {
        conj(&a[(i, j)])
    } else {
        copy(&a[(i, j)])
    }
}
/// Modes: 0 `D*A`, 1 `A*D`, 2 `D\A`, 3 `A/D`, 4 `A+D`, 5 `A-D`, 6 `D-A`.
fn matrix_op<T: ComplexField>(
    d: ColRef<'_, T>,
    a: MatRef<'_, T>,
    conjugate: bool,
    mode: i32,
) -> Result<Result<Mat<T>, usize>, String> {
    let n = d.nrows();
    let (m, k) = a.shape();
    let fits = match mode {
        0 | 2 => m == n,
        1 | 3 => k == n,
        4..=6 => m == n && k == n,
        _ => return Err("unknown diagonal operation".into()),
    };
    if !fits {
        return Err("diagonal and matrix dimensions must match".into());
    }
    if mode == 2 || mode == 3 {
        let i = singular(d);
        if i != 0 {
            return Ok(Err(i));
        }
    }
    let dg = d.as_diagonal();
    Ok(Ok(match (mode, conjugate) {
        (0, false) => dg * a,
        (0, true) => dg * a.conjugate(),
        (1, false) => a * dg,
        (1, true) => a.conjugate() * dg,
        (2, _) => Mat::from_fn(m, k, |i, j| div(&element(a, conjugate, i, j), &d[i])),
        (3, _) => Mat::from_fn(m, k, |i, j| div(&element(a, conjugate, i, j), &d[j])),
        _ => {
            let mut out = Mat::from_fn(m, k, |i, j| {
                let x = element(a, conjugate, i, j);
                if mode == 6 {
                    neg(&x)
                } else {
                    x
                }
            });
            let mut od = out.as_mut().diagonal_mut().column_vector_mut();
            for i in 0..n {
                od[i] = if mode == 5 {
                    sub(&od[i], &d[i])
                } else {
                    add(&od[i], &d[i])
                };
            }
            out
        }
    }))
}
unsafe fn matrix_data(
    d: DiagSource<'_>,
    p: *const u8,
    desc: *const isize,
    mode: i32,
) -> Result<Result<MatrixData, usize>, String> {
    let s = Spec::read(desc)?;
    macro_rules! run {
        ($a:expr, $V:ident) => {
            Ok(matrix_op(diagonal(d)?, $a, s.conjugate, mode)?.map(MatrixData::$V))
        };
    }
    match source(p, s)? {
        Borrowed::F32(a) => run!(a, F32),
        Borrowed::F64(a) => run!(a, F64),
        Borrowed::C32(a) => run!(a, C32),
        Borrowed::C64(a) => run!(a, C64),
    }
}
/// Modes: 0 `D*x` (column), 1 `x*D` (row), 2 `D\x` (column), 3 `x/D` (row).
fn vector_op<T: ComplexField>(
    d: ColRef<'_, T>,
    a: MatRef<'_, T>,
    conjugate: bool,
    mode: i32,
) -> Result<Result<VectorStore<T>, usize>, String> {
    let n = d.nrows();
    let column = match mode {
        0 | 2 => true,
        1 | 3 => false,
        _ => return Err("unknown diagonal vector operation".into()),
    };
    let fits = if column {
        a.shape() == (n, 1)
    } else {
        a.shape() == (1, n)
    };
    if !fits {
        return Err("diagonal and vector dimensions must match".into());
    }
    if mode >= 2 {
        let i = singular(d);
        if i != 0 {
            return Ok(Err(i));
        }
    }
    let dg = d.as_diagonal();
    Ok(Ok(match (mode, conjugate) {
        (0, false) => VectorStore::Column(dg * a.col(0)),
        (0, true) => VectorStore::Column(dg * a.col(0).conjugate()),
        (1, false) => VectorStore::Row(a.row(0) * dg),
        (1, true) => VectorStore::Row(a.row(0).conjugate() * dg),
        (2, _) => VectorStore::Column(Col::from_fn(n, |i| {
            div(&element(a, conjugate, i, 0), &d[i])
        })),
        _ => VectorStore::Row(Row::from_fn(n, |j| {
            div(&element(a, conjugate, 0, j), &d[j])
        })),
    }))
}
unsafe fn vector_data(
    d: DiagSource<'_>,
    p: *const u8,
    desc: *const isize,
    mode: i32,
) -> Result<Result<VectorData, usize>, String> {
    let s = Spec::read(desc)?;
    macro_rules! run {
        ($a:expr, $V:ident) => {
            Ok(vector_op(diagonal(d)?, $a, s.conjugate, mode)?.map(VectorData::$V))
        };
    }
    match source(p, s)? {
        Borrowed::F32(a) => run!(a, F32),
        Borrowed::F64(a) => run!(a, F64),
        Borrowed::C32(a) => run!(a, C32),
        Borrowed::C64(a) => run!(a, C64),
    }
}
/// In-place modes on a whole owner: 0 `lmul!(D,A)`, 1 `rmul!(A,D)`,
/// 2 `ldiv!(D,A)`, 3 `rdiv!(A,D)`. Singular divisors leave `A` unchanged.
fn scale_in_place<T: ComplexField>(
    mut a: MatMut<'_, T>,
    d: ColRef<'_, T>,
    mode: i32,
) -> Result<usize, String> {
    let n = d.nrows();
    let fits = match mode {
        0 | 2 => a.nrows() == n,
        1 | 3 => a.ncols() == n,
        _ => return Err("unknown diagonal update".into()),
    };
    if !fits {
        return Err("diagonal and destination dimensions must match".into());
    }
    if mode >= 2 {
        let i = singular(d);
        if i != 0 {
            return Ok(i);
        }
    }
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            let s = if mode == 0 || mode == 2 { &d[i] } else { &d[j] };
            a[(i, j)] = if mode < 2 {
                mul(s, &a[(i, j)])
            } else {
                div(&a[(i, j)], s)
            };
        }
    }
    Ok(0)
}
unsafe fn scale_matrix<T: Pick>(
    a: &mut Mat<T>,
    d: DiagSource<'_>,
    mode: i32,
) -> Result<usize, String> {
    scale_in_place(a.as_mut(), diagonal(d)?, mode)
}
unsafe fn scale_typed(a: &mut TypedMatrix, d: DiagSource<'_>, mode: i32) -> Result<usize, String> {
    match a.data.as_mut().ok_or("matrix is not initialized")? {
        MatrixData::F32(x) => scale_matrix(x, d, mode),
        MatrixData::F64(x) => scale_matrix(x, d, mode),
        MatrixData::C32(x) => scale_matrix(x, d, mode),
        MatrixData::C64(x) => scale_matrix(x, d, mode),
    }
}
/// Columns take left modes, rows right modes, diagonals either side.
unsafe fn scale_collection<T: Pick>(
    a: &mut VectorStore<T>,
    d: DiagSource<'_>,
    mode: i32,
) -> Result<usize, String> {
    let d = diagonal(d)?;
    let column_mode = mode == 0 || mode == 2;
    if !(0..=3).contains(&mode) {
        return Err("unknown diagonal update".into());
    }
    match a {
        VectorStore::Column(_) if !column_mode => {
            Err("a column destination needs a left diagonal".into())
        }
        VectorStore::Row(_) if column_mode => {
            Err("a row destination needs a right diagonal".into())
        }
        VectorStore::Row(r) => scale_in_place(r.as_mat_mut(), d, mode),
        VectorStore::Column(c) => scale_in_place(c.as_mat_mut(), d, mode),
        // A stored diagonal is a column internally; both sides scale entries.
        VectorStore::Diagonal(g) => {
            let left = if column_mode { mode } else { mode - 1 };
            scale_in_place(g.column_vector_mut().as_mat_mut(), d, left)
        }
    }
}
unsafe fn scale_owner(a: &mut VectorOwner, d: DiagSource<'_>, mode: i32) -> Result<usize, String> {
    match a.data.as_mut().ok_or("collection is not initialized")? {
        VectorData::F32(x) => scale_collection(x, d, mode),
        VectorData::F64(x) => scale_collection(x, d, mode),
        VectorData::C32(x) => scale_collection(x, d, mode),
        VectorData::C64(x) => scale_collection(x, d, mode),
    }
}
fn inverse<T: Pick>(d: &VectorStore<T>) -> Result<Result<VectorStore<T>, usize>, String> {
    let VectorStore::Diagonal(x) = d else {
        return Err("diagonal operand required".into());
    };
    let c = x.column_vector();
    let i = singular(c);
    if i != 0 {
        return Ok(Err(i));
    }
    Ok(Ok(VectorStore::new(2, c.nrows(), |i| recip(&c[i]))?))
}
fn extract<T: ComplexField>(
    a: MatRef<'_, T>,
    conjugate: bool,
    kind: i32,
) -> Result<VectorStore<T>, String> {
    let d = a.diagonal().column_vector();
    VectorStore::new(kind, d.nrows(), |i| {
        if conjugate {
            conj(&d[i])
        } else {
            copy(&d[i])
        }
    })
}
fn status<T>(r: Result<T, usize>, out: &mut Option<T>) -> usize {
    match r {
        Ok(x) => {
            *out = Some(x);
            0
        }
        Err(i) => i,
    }
}

#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn diagonal_matrix(
        &mut self,
        diag: &VectorOwner,
        p: *const u8,
        desc: *const isize,
        mode: i32,
    ) -> Result<usize, String> {
        guarded(|| self.diagonal_matrix_from(DiagSource::Owner(diag), p, desc, mode))
    }
    #[julia]
    pub fn diagonal_matrix_values(
        &mut self,
        values: *const u8,
        n: usize,
        p: *const u8,
        desc: *const isize,
        mode: i32,
    ) -> Result<usize, String> {
        guarded(|| self.diagonal_matrix_from(DiagSource::Values(values, n), p, desc, mode))
    }
    #[julia]
    pub fn diagonal_update(&mut self, diag: &VectorOwner, mode: i32) -> Result<usize, String> {
        guarded(|| unsafe { scale_matrix(&mut self.matrix, DiagSource::Owner(diag), mode) })
    }
    #[julia]
    pub fn diagonal_update_values(
        &mut self,
        values: *const u8,
        n: usize,
        mode: i32,
    ) -> Result<usize, String> {
        guarded(|| unsafe { scale_matrix(&mut self.matrix, DiagSource::Values(values, n), mode) })
    }
}
impl OwnedMatrix {
    fn diagonal_matrix_from(
        &mut self,
        d: DiagSource<'_>,
        p: *const u8,
        desc: *const isize,
        mode: i32,
    ) -> Result<usize, String> {
        match unsafe { matrix_data(d, p, desc, mode)? } {
            Ok(MatrixData::F64(x)) => {
                self.matrix = x;
                Ok(0)
            }
            Ok(_) => Err("Float64 output requires Float64 operands".into()),
            Err(i) => Ok(i),
        }
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn diagonal_matrix(
        &mut self,
        diag: &VectorOwner,
        p: *const u8,
        desc: *const isize,
        mode: i32,
    ) -> Result<usize, String> {
        guarded(|| {
            let r = unsafe { matrix_data(DiagSource::Owner(diag), p, desc, mode)? };
            Ok(status(r, &mut self.data))
        })
    }
    #[julia]
    pub fn diagonal_matrix_values(
        &mut self,
        values: *const u8,
        n: usize,
        p: *const u8,
        desc: *const isize,
        mode: i32,
    ) -> Result<usize, String> {
        guarded(|| {
            let r = unsafe { matrix_data(DiagSource::Values(values, n), p, desc, mode)? };
            Ok(status(r, &mut self.data))
        })
    }
    #[julia]
    pub fn diagonal_update(&mut self, diag: &VectorOwner, mode: i32) -> Result<usize, String> {
        guarded(|| unsafe { scale_typed(self, DiagSource::Owner(diag), mode) })
    }
    #[julia]
    pub fn diagonal_update_values(
        &mut self,
        values: *const u8,
        n: usize,
        mode: i32,
    ) -> Result<usize, String> {
        guarded(|| unsafe { scale_typed(self, DiagSource::Values(values, n), mode) })
    }
}
#[julia]
impl VectorOwner {
    #[julia]
    pub fn diagonal_vector(
        &mut self,
        diag: &VectorOwner,
        p: *const u8,
        desc: *const isize,
        mode: i32,
    ) -> Result<usize, String> {
        guarded(|| {
            let r = unsafe { vector_data(DiagSource::Owner(diag), p, desc, mode)? };
            Ok(status(r, &mut self.data))
        })
    }
    #[julia]
    pub fn diagonal_vector_values(
        &mut self,
        values: *const u8,
        n: usize,
        p: *const u8,
        desc: *const isize,
        mode: i32,
    ) -> Result<usize, String> {
        guarded(|| {
            let r = unsafe { vector_data(DiagSource::Values(values, n), p, desc, mode)? };
            Ok(status(r, &mut self.data))
        })
    }
    #[julia]
    pub fn diagonal_collection_update(
        &mut self,
        diag: &VectorOwner,
        mode: i32,
    ) -> Result<usize, String> {
        guarded(|| unsafe { scale_owner(self, DiagSource::Owner(diag), mode) })
    }
    #[julia]
    pub fn diagonal_collection_update_values(
        &mut self,
        values: *const u8,
        n: usize,
        mode: i32,
    ) -> Result<usize, String> {
        guarded(|| unsafe { scale_owner(self, DiagSource::Values(values, n), mode) })
    }
    #[julia]
    pub fn diagonal_inverse(&mut self, diag: &VectorOwner) -> Result<usize, String> {
        guarded(|| {
            let r = match diag.data()? {
                VectorData::F32(x) => inverse(x)?.map(VectorData::F32),
                VectorData::F64(x) => inverse(x)?.map(VectorData::F64),
                VectorData::C32(x) => inverse(x)?.map(VectorData::C32),
                VectorData::C64(x) => inverse(x)?.map(VectorData::C64),
            };
            Ok(status(r, &mut self.data))
        })
    }
    #[julia]
    pub fn diagonal_of_view(
        &mut self,
        p: *const u8,
        desc: *const isize,
        kind: i32,
    ) -> Result<(), String> {
        guarded(|| {
            let s = unsafe { Spec::read(desc)? };
            self.data = Some(match unsafe { source(p, s)? } {
                Borrowed::F32(a) => VectorData::F32(extract(a, s.conjugate, kind)?),
                Borrowed::F64(a) => VectorData::F64(extract(a, s.conjugate, kind)?),
                Borrowed::C32(a) => VectorData::C32(extract(a, s.conjugate, kind)?),
                Borrowed::C64(a) => VectorData::C64(extract(a, s.conjugate, kind)?),
            });
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn diag(values: &[f64]) -> VectorOwner {
        VectorOwner {
            data: Some(VectorData::F64(
                VectorStore::new(2, values.len(), |i| values[i]).unwrap(),
            )),
        }
    }
    fn whole(m: usize, n: usize, kind: isize) -> [isize; 9] {
        [kind, 0, 1, m as isize, 0, 1, n as isize, 0, 0]
    }
    #[test]
    fn products_solves_and_sums() {
        let d = diag(&[2.0, -4.0]);
        let mut a = OwnedMatrix::new();
        a.matrix = Mat::from_fn(2, 3, |i, j| (i * 3 + j) as f64 + 1.0);
        let desc = whole(2, 3, 0);
        let p = (&a as *const OwnedMatrix).cast::<u8>();
        let mut out = OwnedMatrix::new();
        assert_eq!(out.diagonal_matrix(&d, p, desc.as_ptr(), 0).unwrap(), 0);
        assert_eq!(out.matrix[(1, 2)], -24.0);
        assert_eq!(out.diagonal_matrix(&d, p, desc.as_ptr(), 2).unwrap(), 0);
        assert_eq!(out.matrix[(1, 2)], -1.5);
        assert!(out.diagonal_matrix(&d, p, desc.as_ptr(), 1).is_err());
        assert!(out.diagonal_matrix(&d, p, desc.as_ptr(), 4).is_err());
        let z = diag(&[2.0, 0.0]);
        let before = out.matrix.clone();
        assert_eq!(out.diagonal_matrix(&z, p, desc.as_ptr(), 2).unwrap(), 2);
        assert_eq!(out.matrix, before);
        let mut sq = OwnedMatrix::new();
        sq.matrix = Mat::from_fn(2, 2, |i, j| (i + 2 * j) as f64);
        let sd = whole(2, 2, 0);
        let q = (&sq as *const OwnedMatrix).cast::<u8>();
        out.diagonal_matrix(&d, q, sd.as_ptr(), 6).unwrap();
        assert_eq!(out.matrix[(0, 0)], 2.0);
        assert_eq!(out.matrix[(1, 0)], -1.0);
        assert_eq!(out.matrix[(1, 1)], -7.0);
        let values = [1.0f64, f64::INFINITY];
        out.diagonal_matrix_values(values.as_ptr().cast(), 2, q, sd.as_ptr(), 4)
            .unwrap();
        assert_eq!(out.matrix[(1, 1)], f64::INFINITY);
        assert_eq!(out.matrix[(0, 1)], 2.0);
    }
    #[test]
    fn in_place_updates_preserve_on_singular() {
        let mut a = OwnedMatrix::new();
        a.matrix = Mat::from_fn(2, 2, |i, j| (i + 2 * j) as f64 + 1.0);
        let before = a.matrix.clone();
        assert_eq!(a.diagonal_update(&diag(&[1.0, 0.0]), 2).unwrap(), 2);
        assert_eq!(a.matrix, before);
        assert_eq!(a.diagonal_update(&diag(&[2.0, 3.0]), 1).unwrap(), 0);
        assert_eq!(a.matrix[(1, 1)], 12.0);
        assert!(a.diagonal_update(&diag(&[2.0]), 0).is_err());
        let mut r = VectorOwner {
            data: Some(VectorData::F64(
                VectorStore::new(1, 2, |i| i as f64 + 1.0).unwrap(),
            )),
        };
        assert!(r.diagonal_collection_update(&diag(&[2.0, 3.0]), 0).is_err());
        r.diagonal_collection_update(&diag(&[2.0, 4.0]), 3).unwrap();
        match &r.data {
            Some(VectorData::F64(x)) => assert_eq!(x.get(1), 0.5),
            _ => panic!(),
        }
        let mut inv = VectorOwner { data: None };
        assert_eq!(inv.diagonal_inverse(&diag(&[2.0, 0.0])).unwrap(), 2);
        assert!(inv.data.is_none());
        assert_eq!(inv.diagonal_inverse(&diag(&[2.0, -4.0])).unwrap(), 0);
        match &inv.data {
            Some(VectorData::F64(x)) => assert_eq!((x.kind(), x.get(1)), (2, -0.25)),
            _ => panic!(),
        }
    }
}
