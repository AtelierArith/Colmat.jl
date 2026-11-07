//! Mutable destinations named by view descriptors, and the single overlap
//! rule every native write into them follows.
//!
//! A destination is any view descriptor `views::source` can read: a native
//! owner (kinds 0/1/2) or a borrowed Julia strided buffer (kind 4). The
//! `MatMut` is built from the owner or the Julia pointer/strides for one call
//! only and never escapes it.
//!
//! Overlap rule: before any `MatMut` exists, the destination is read as a
//! `MatRef` and its byte range is compared with every input's byte range.
//! Inputs that overlap the destination (the destination itself, its
//! transpose, a view of the same owner or the same Julia memory) are copied
//! into an owned snapshot; disjoint inputs are read in place.
use super::{source, span, Borrowed, Spec, RAW_KIND};
use crate::collections::{VectorData, VectorOwner};
use crate::typed::{MatrixData, TypedMatrix};
use crate::{guarded, OwnedMatrix};
use faer::traits::{math_utils::*, ComplexField};
use faer::{Accum, Conj, Mat, MatMut, MatRef, Par};
use rustcall_julia_macros::julia;

/// `MatMut` over a Julia strided buffer of `T` (strides in elements).
///
/// # Safety
/// `p` addresses the first element of a live, injective Julia strided array
/// with this layout, rooted by Julia for the call. Native storage never
/// aliases Julia memory.
pub(crate) unsafe fn raw_mut<'a, T>(
    p: *mut u8,
    m: usize,
    n: usize,
    rs: isize,
    cs: isize,
) -> MatMut<'a, T> {
    let ptr = if m == 0 || n == 0 {
        std::ptr::NonNull::<T>::dangling().as_ptr()
    } else {
        p.cast::<T>()
    };
    MatMut::from_raw_parts_mut(ptr, m, n, rs, cs)
}

/// Whether an `m x n` layout with strides `rs`/`cs` maps distinct indices to
/// distinct elements (Julia validates this too; Rust re-checks cheaply).
pub(crate) fn injective(m: usize, n: usize, rs: isize, cs: isize) -> bool {
    let mut axes = [(rs.unsigned_abs(), m), (cs.unsigned_abs(), n)];
    if m == 0 || n == 0 {
        return true;
    }
    axes.sort();
    let mut extent = 0usize;
    for (s, len) in axes {
        if len <= 1 {
            continue;
        }
        if s == 0 || s <= extent {
            return false;
        }
        match s.checked_mul(len - 1) {
            Some(x) => extent = x,
            None => return false,
        }
    }
    true
}

/// Bounds-checked exclusive borrow of the view descriptor `s` over `base`.
/// Conjugated or index-repeating selections cannot be written through.
pub(crate) fn view_mut<'a, T>(base: MatMut<'a, T>, s: Spec) -> Result<MatMut<'a, T>, String> {
    if s.conjugate {
        return Err("destination cannot be conjugated".into());
    }
    s.check(base.nrows(), base.ncols())?;
    if (s.rs == 0 && s.m > 1) || (s.cs == 0 && s.n > 1) {
        return Err("destination must not repeat elements".into());
    }
    let (rs, cs) = (base.row_stride(), base.col_stride());
    let offset = if s.m == 0 || s.n == 0 {
        0
    } else {
        rs.checked_mul(s.r)
            .and_then(|x| cs.checked_mul(s.c).and_then(|y| x.checked_add(y)))
            .ok_or("view offset overflow")?
    };
    let srs = rs.checked_mul(s.rs).ok_or("view stride overflow")?;
    let scs = cs.checked_mul(s.cs).ok_or("view stride overflow")?;
    // SAFETY: every selected index is inside the owner and the map is
    // injective, so this is a genuine exclusive borrow of `base`.
    let v =
        unsafe { MatMut::from_raw_parts_mut(base.as_ptr_mut().offset(offset), s.m, s.n, srs, scs) };
    Ok(if s.transpose { v.transpose_mut() } else { v })
}

pub(crate) enum BorrowedMut<'a> {
    F32(MatMut<'a, f32>),
    F64(MatMut<'a, f64>),
    C32(MatMut<'a, faer::c32>),
    C64(MatMut<'a, faer::c64>),
}

/// Exclusive borrow of the destination named by `p`/`s` for one call.
///
/// # Safety
/// `p` is a live owner of descriptor kind `s.kind` (or the first element of
/// a rooted Julia buffer for kind 4) that no other live reference writes.
/// Inputs overlapping it must already have been snapshot (see [`Operand`]).
pub(crate) unsafe fn destination<'a>(p: *mut u8, s: Spec) -> Result<BorrowedMut<'a>, String> {
    if let Some(raw) = s.raw.filter(|_| s.kind == RAW_KIND) {
        if !injective(raw.m, raw.n, raw.rs, raw.cs) {
            return Err("destination must not repeat elements".into());
        }
        let (m, n, rs, cs) = (raw.m, raw.n, raw.rs, raw.cs);
        return Ok(match raw.tag {
            0 => BorrowedMut::F32(view_mut(raw_mut(p, m, n, rs, cs), s)?),
            1 => BorrowedMut::F64(view_mut(raw_mut(p, m, n, rs, cs), s)?),
            2 => BorrowedMut::C32(view_mut(raw_mut(p, m, n, rs, cs), s)?),
            3 => BorrowedMut::C64(view_mut(raw_mut(p, m, n, rs, cs), s)?),
            _ => return Err("unknown borrowed scalar".into()),
        });
    }
    Ok(match s.kind {
        0 => BorrowedMut::F64(view_mut(
            (&mut *p.cast::<OwnedMatrix>()).matrix.as_mut(),
            s,
        )?),
        1 => match (&mut *p.cast::<TypedMatrix>())
            .data
            .as_mut()
            .ok_or("matrix is not initialized")?
        {
            MatrixData::F32(x) => BorrowedMut::F32(view_mut(x.as_mut(), s)?),
            MatrixData::F64(x) => BorrowedMut::F64(view_mut(x.as_mut(), s)?),
            MatrixData::C32(x) => BorrowedMut::C32(view_mut(x.as_mut(), s)?),
            MatrixData::C64(x) => BorrowedMut::C64(view_mut(x.as_mut(), s)?),
        },
        2 => match (&mut *p.cast::<VectorOwner>())
            .data
            .as_mut()
            .ok_or("collection is not initialized")?
        {
            VectorData::F32(x) => BorrowedMut::F32(view_mut(x.as_mat_mut(), s)?),
            VectorData::F64(x) => BorrowedMut::F64(view_mut(x.as_mat_mut(), s)?),
            VectorData::C32(x) => BorrowedMut::C32(view_mut(x.as_mat_mut(), s)?),
            VectorData::C64(x) => BorrowedMut::C64(view_mut(x.as_mat_mut(), s)?),
        },
        _ => return Err("unknown view owner".into()),
    })
}

/// Selects one scalar arm of the type-erased borrows.
pub(crate) trait Arm: ComplexField + Sized {
    fn pick_ref(b: Borrowed<'_>) -> Option<MatRef<'_, Self>>;
    fn pick_mut(b: BorrowedMut<'_>) -> Option<MatMut<'_, Self>>;
}
macro_rules! arm {
    ($t:ty, $v:ident) => {
        impl Arm for $t {
            fn pick_ref(b: Borrowed<'_>) -> Option<MatRef<'_, Self>> {
                match b {
                    Borrowed::$v(x) => Some(x),
                    _ => None,
                }
            }
            fn pick_mut(b: BorrowedMut<'_>) -> Option<MatMut<'_, Self>> {
                match b {
                    BorrowedMut::$v(x) => Some(x),
                    _ => None,
                }
            }
        }
    };
}
arm!(f32, F32);
arm!(f64, F64);
arm!(faer::c32, C32);
arm!(faer::c64, C64);

fn erased_span(b: &Borrowed<'_>) -> Option<(usize, usize)> {
    match b {
        Borrowed::F32(x) => span(*x),
        Borrowed::F64(x) => span(*x),
        Borrowed::C32(x) => span(*x),
        Borrowed::C64(x) => span(*x),
    }
}
fn overlaps(a: Option<(usize, usize)>, b: Option<(usize, usize)>) -> bool {
    match (a, b) {
        (Some((a0, a1)), Some((b0, b1))) => a0 < b1 && b0 < a1,
        _ => false,
    }
}

/// An input read in place, or a snapshot of one that overlaps the destination.
pub(crate) enum Operand<'a, T> {
    Direct(MatRef<'a, T>, bool),
    Owned(Mat<T>),
}
impl<T: ComplexField> Operand<'_, T> {
    pub(crate) fn get(&self) -> (MatRef<'_, T>, bool) {
        match self {
            Operand::Direct(a, c) => (*a, *c),
            Operand::Owned(a) => (a.as_ref(), false),
        }
    }
    pub(crate) fn is_snapshot(&self) -> bool {
        matches!(self, Operand::Owned(_))
    }
}

/// The destination's byte range, read before any mutable borrow exists.
///
/// # Safety
/// As for [`source`].
pub(crate) unsafe fn destination_span(
    p: *const u8,
    s: Spec,
) -> Result<Option<(usize, usize)>, String> {
    Ok(erased_span(&source(p, s)?))
}

/// Read input `p`/`desc` of scalar `T`; snapshot it if it overlaps `dest`.
///
/// # Safety
/// As for [`source`]; must be called before the destination is borrowed.
pub(crate) unsafe fn operand<'a, T: Arm>(
    p: *const u8,
    s: Spec,
    dest: Option<(usize, usize)>,
) -> Result<Operand<'a, T>, String> {
    let a = T::pick_ref(source(p, s)?).ok_or("operand scalar types must match")?;
    Ok(if overlaps(span(a), dest) {
        Operand::Owned(if s.conjugate {
            a.conjugate().to_owned()
        } else {
            a.to_owned()
        })
    } else {
        Operand::Direct(a, s.conjugate)
    })
}

fn scale<T: ComplexField>(mut c: MatMut<'_, T>, beta: &T) {
    for j in 0..c.ncols() {
        for i in 0..c.nrows() {
            let x = mul(beta, &c[(i, j)]);
            c[(i, j)] = x;
        }
    }
}
fn conj_of(c: bool) -> Conj {
    if c {
        Conj::Yes
    } else {
        Conj::No
    }
}

/// `C = alpha*A*B + beta*C`. `beta_zero` never reads C (NaN-safe) and
/// `alpha_zero` never reads A or B.
#[allow(clippy::too_many_arguments)]
unsafe fn gemm<T: Arm>(
    dest: *mut u8,
    ds: Spec,
    a: *const u8,
    sa: Spec,
    b: *const u8,
    sb: Spec,
    alpha: *const u8,
    beta: *const u8,
    alpha_zero: bool,
    beta_zero: bool,
) -> Result<(), String> {
    let dspan = destination_span(dest, ds)?;
    let x = operand::<T>(a, sa, dspan)?;
    let y = operand::<T>(b, sb, dspan)?;
    let alpha = copy(&*alpha.cast::<T>());
    let beta = copy(&*beta.cast::<T>());
    let mut c = T::pick_mut(destination(dest, ds)?).ok_or("destination scalar types must match")?;
    let ((a, ca), (b, cb)) = (x.get(), y.get());
    if a.ncols() != b.nrows() {
        return Err("inner dimensions must match".into());
    }
    if c.shape() != (a.nrows(), b.ncols()) {
        return Err("output dimensions must match".into());
    }
    if alpha_zero {
        if beta_zero {
            c.fill(zero::<T>());
        } else {
            scale(c, &beta);
        }
        return Ok(());
    }
    let accum = if beta_zero {
        Accum::Replace
    } else {
        if beta != one::<T>() {
            scale(c.as_mut(), &beta);
        }
        Accum::Add
    };
    faer::linalg::matmul::matmul_with_conj(
        c,
        accum,
        a,
        conj_of(ca),
        b,
        conj_of(cb),
        alpha,
        Par::Seq,
    );
    Ok(())
}

/// Native dense `mul!`: `dest = alpha*A*B + beta*dest` for view descriptors
/// of one native scalar. `alpha`/`beta` point to that scalar.
///
/// # Safety
/// Julia roots every owner, descriptor and coefficient for this call.
#[allow(clippy::too_many_arguments)]
#[julia]
pub fn view_gemm(
    dest: *mut u8,
    dd: *const isize,
    a: *const u8,
    ad: *const isize,
    b: *const u8,
    bd: *const isize,
    alpha: *const u8,
    beta: *const u8,
    alpha_zero: bool,
    beta_zero: bool,
) -> Result<(), String> {
    guarded(|| unsafe {
        let (ds, sa, sb) = (Spec::read(dd)?, Spec::read(ad)?, Spec::read(bd)?);
        macro_rules! go {
            ($t:ty) => {
                gemm::<$t>(dest, ds, a, sa, b, sb, alpha, beta, alpha_zero, beta_zero)
            };
        }
        match source(dest, ds)? {
            Borrowed::F32(_) => go!(f32),
            Borrowed::F64(_) => go!(f64),
            Borrowed::C32(_) => go!(faer::c32),
            Borrowed::C64(_) => go!(faer::c64),
        }
    })
}

unsafe fn assign<T: Arm>(dest: *mut u8, ds: Spec, p: *const u8, s: Spec) -> Result<(), String> {
    let x = operand::<T>(p, s, destination_span(dest, ds)?)?;
    let mut c = T::pick_mut(destination(dest, ds)?).ok_or("assignment scalar types must match")?;
    let (a, conjugate) = x.get();
    if c.shape() != a.shape() {
        return Err("view assignment shape mismatch".into());
    }
    if conjugate {
        for j in 0..a.ncols() {
            for i in 0..a.nrows() {
                c[(i, j)] = conj(&a[(i, j)]);
            }
        }
    } else {
        c.copy_from(a);
    }
    Ok(())
}

/// Copy the view `src`/`sd` into the destination `dest`/`dd` of the same
/// shape and scalar; any destination kind, including borrowed Julia memory.
///
/// # Safety
/// Julia roots both owners and descriptors for this call.
#[julia]
pub fn view_assign_into(
    dest: *mut u8,
    dd: *const isize,
    src: *const u8,
    sd: *const isize,
) -> Result<(), String> {
    guarded(|| unsafe {
        let (ds, s) = (Spec::read(dd)?, Spec::read(sd)?);
        match source(dest, ds)? {
            Borrowed::F32(_) => assign::<f32>(dest, ds, src, s),
            Borrowed::F64(_) => assign::<f64>(dest, ds, src, s),
            Borrowed::C32(_) => assign::<faer::c32>(dest, ds, src, s),
            Borrowed::C64(_) => assign::<faer::c64>(dest, ds, src, s),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocation_probe;

    fn raw(m: usize, n: usize, rs: isize, cs: isize, transpose: bool, tag: isize) -> Vec<isize> {
        vec![
            RAW_KIND as isize,
            0,
            1,
            m as isize,
            0,
            1,
            n as isize,
            transpose as isize,
            0,
            m as isize,
            n as isize,
            rs,
            cs,
            tag,
        ]
    }
    fn owner(m: usize, n: usize, transpose: bool, conjugate: bool) -> Vec<isize> {
        vec![
            0,
            0,
            1,
            m as isize,
            0,
            1,
            n as isize,
            transpose as isize,
            conjugate as isize,
        ]
    }
    fn reference(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> Mat<f64> {
        a * b
    }

    #[test]
    fn injectivity_rejects_repeated_and_interleaved_layouts() {
        assert!(injective(3, 4, 1, 3));
        assert!(injective(3, 4, -1, 3));
        assert!(injective(3, 4, 4, 1));
        assert!(injective(1, 4, 0, 2));
        assert!(!injective(3, 4, 0, 3));
        assert!(!injective(3, 4, 1, 2));
        assert!(injective(0, 4, 0, 0));
    }

    #[test]
    fn gemm_writes_julia_memory_in_place_without_allocating() {
        let (m, k, n) = (5, 4, 3);
        let a = Mat::from_fn(m, k, |i, j| (i + 2 * j) as f64);
        let b = Mat::from_fn(k, n, |i, j| (3 * i) as f64 - j as f64);
        let (oa, ob) = (
            OwnedMatrix { matrix: a.clone() },
            OwnedMatrix { matrix: b.clone() },
        );
        let (ad, bd) = (owner(m, k, false, false), owner(k, n, false, false));
        // Column-major Julia buffer of 5 x 3 with leading dimension 7.
        let mut julia = vec![1.0f64; 7 * n];
        let dd = raw(m, n, 1, 7, false, 1);
        let (alpha, beta) = (2.0f64, 0.5f64);
        let expected = Mat::from_fn(m, n, |i, j| {
            alpha * reference(a.as_ref(), b.as_ref())[(i, j)] + beta
        });
        let count = allocation_probe::count(|| {
            view_gemm(
                julia.as_mut_ptr().cast(),
                dd.as_ptr(),
                (&oa as *const OwnedMatrix).cast(),
                ad.as_ptr(),
                (&ob as *const OwnedMatrix).cast(),
                bd.as_ptr(),
                (&alpha as *const f64).cast(),
                (&beta as *const f64).cast(),
                false,
                false,
            )
            .unwrap();
        });
        assert_eq!(count, 0);
        for j in 0..n {
            for i in 0..m {
                assert!((julia[i + 7 * j] - expected[(i, j)]).abs() < 1e-12);
            }
            // Padding rows are untouched.
            assert_eq!(julia[5 + 7 * j], 1.0);
        }
    }

    #[test]
    fn overlapping_inputs_are_snapshot_and_disjoint_inputs_are_not() {
        let n = 4;
        let mut julia: Vec<f64> = (0..n * n).map(|x| x as f64 * 0.25 - 1.0).collect();
        let before = MatRef::from_column_major_slice(&julia, n, n).to_owned();
        let d = raw(n, n, 1, n as isize, false, 1);
        let t = raw(n, n, 1, n as isize, true, 1);
        let p = julia.as_mut_ptr();
        unsafe {
            let ds = Spec::read(d.as_ptr()).unwrap();
            let dspan = destination_span(p.cast(), ds).unwrap();
            let x = operand::<f64>(p.cast(), Spec::read(t.as_ptr()).unwrap(), dspan).unwrap();
            assert!(x.is_snapshot());
            let other = vec![0.0f64; n * n];
            let y = operand::<f64>(other.as_ptr().cast(), ds, dspan).unwrap();
            assert!(!y.is_snapshot());
        }
        // C = C' * C with C as all three operands.
        let (one, zero) = (1.0f64, 0.0f64);
        view_gemm(
            p.cast(),
            d.as_ptr(),
            p.cast(),
            t.as_ptr(),
            p.cast(),
            d.as_ptr(),
            (&one as *const f64).cast(),
            (&zero as *const f64).cast(),
            false,
            true,
        )
        .unwrap();
        let expected = before.transpose() * &before;
        let got = MatRef::from_column_major_slice(&julia, n, n);
        assert!((got - &expected).norm_max() < 1e-12);
    }

    #[test]
    fn zero_coefficients_never_read_nan_operands() {
        let a = OwnedMatrix {
            matrix: Mat::from_fn(2, 2, |_, _| f64::NAN),
        };
        let mut c = OwnedMatrix {
            matrix: Mat::from_fn(2, 2, |_, _| f64::NAN),
        };
        let d = owner(2, 2, false, false);
        let (alpha, beta) = (0.0f64, 0.0f64);
        let ap = (&a as *const OwnedMatrix).cast::<u8>();
        view_gemm(
            (&mut c as *mut OwnedMatrix).cast(),
            d.as_ptr(),
            ap,
            d.as_ptr(),
            ap,
            d.as_ptr(),
            (&alpha as *const f64).cast(),
            (&beta as *const f64).cast(),
            true,
            true,
        )
        .unwrap();
        assert_eq!(c.matrix, Mat::<f64>::zeros(2, 2));
    }

    #[test]
    fn failures_leave_the_destination_untouched() {
        let a = OwnedMatrix {
            matrix: Mat::from_fn(3, 2, |i, j| (i + j) as f64),
        };
        let mut julia = vec![7.0f64; 9];
        let d = raw(3, 3, 1, 3, false, 1);
        let ad = owner(3, 2, false, false);
        let one = 1.0f64;
        let ap = (&a as *const OwnedMatrix).cast::<u8>();
        let err = view_gemm(
            julia.as_mut_ptr().cast(),
            d.as_ptr(),
            ap,
            ad.as_ptr(),
            ap,
            ad.as_ptr(),
            (&one as *const f64).cast(),
            (&one as *const f64).cast(),
            false,
            false,
        );
        assert!(err.is_err());
        // Wrong scalar tag, repeated layout and conjugated destinations.
        let wrong = raw(3, 3, 1, 3, false, 0);
        assert!(
            view_assign_into(julia.as_mut_ptr().cast(), wrong.as_ptr(), ap, ad.as_ptr()).is_err()
        );
        let repeated = raw(3, 3, 0, 3, false, 1);
        let src = owner(3, 3, false, false);
        let sq = OwnedMatrix {
            matrix: Mat::zeros(3, 3),
        };
        let sp = (&sq as *const OwnedMatrix).cast::<u8>();
        assert!(view_assign_into(
            julia.as_mut_ptr().cast(),
            repeated.as_ptr(),
            sp,
            src.as_ptr()
        )
        .is_err());
        let mut conjugated = raw(3, 3, 1, 3, false, 1);
        conjugated[8] = 1;
        assert!(view_assign_into(
            julia.as_mut_ptr().cast(),
            conjugated.as_ptr(),
            sp,
            src.as_ptr()
        )
        .is_err());
        assert!(julia.iter().all(|&x| x == 7.0));
    }

    #[test]
    fn assignment_copies_native_owners_into_julia_memory_in_place() {
        let a = OwnedMatrix {
            matrix: Mat::from_fn(3, 2, |i, j| (i + 10 * j) as f64),
        };
        let ap = (&a as *const OwnedMatrix).cast::<u8>();
        // Reversed-row Julia destination: first element is the last row.
        let mut julia = vec![0.0f64; 6];
        let d = raw(3, 2, -1, 3, false, 1);
        let first = unsafe { julia.as_mut_ptr().add(2) };
        let sd = owner(3, 2, false, false);
        let count = allocation_probe::count(|| {
            view_assign_into(first.cast(), d.as_ptr(), ap, sd.as_ptr()).unwrap();
        });
        assert_eq!(count, 0);
        assert_eq!(julia, vec![2.0, 1.0, 0.0, 12.0, 11.0, 10.0]);
        // A native destination receiving a transposed native source.
        let mut c = OwnedMatrix {
            matrix: Mat::zeros(2, 3),
        };
        view_assign_into(
            (&mut c as *mut OwnedMatrix).cast(),
            owner(2, 3, false, false).as_ptr(),
            ap,
            owner(3, 2, true, false).as_ptr(),
        )
        .unwrap();
        assert_eq!(c.matrix, a.matrix.transpose().to_owned());
    }
}
