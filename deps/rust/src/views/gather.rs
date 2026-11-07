//! Index-list gather (`A[I, J]`) and scatter (`A[I, J] = B`) between view
//! descriptors, plus the effective strides of a view.
//!
//! Index lists are 0-based positions in the named view (after its offset,
//! steps and transpose). Every index is checked before the destination is
//! written, and inputs overlapping the destination are snapshot first (see
//! [`super::destination`]), so a failure leaves the destination untouched.
use super::destination::{destination, destination_span, operand, Arm};
use super::{source, Borrowed, Spec};
use crate::guarded;
use faer::traits::math_utils::*;
use rustcall_julia_macros::julia;

fn check_indices(indices: &[usize], len: usize) -> Result<(), String> {
    if indices.iter().all(|&i| i < len) {
        Ok(())
    } else {
        Err("index out of bounds".into())
    }
}

unsafe fn gather<T: Arm>(
    dest: *mut u8,
    ds: Spec,
    src: *const u8,
    s: Spec,
    rows: &[usize],
    cols: &[usize],
) -> Result<(), String> {
    let x = operand::<T>(src, s, destination_span(dest, ds)?)?;
    let (a, conjugate) = x.get();
    check_indices(rows, a.nrows())?;
    check_indices(cols, a.ncols())?;
    let mut c = T::pick_mut(destination(dest, ds)?).ok_or("gather scalar types must match")?;
    if c.shape() != (rows.len(), cols.len()) {
        return Err("gather destination shape mismatch".into());
    }
    for (j, &q) in cols.iter().enumerate() {
        for (i, &p) in rows.iter().enumerate() {
            let v = &a[(p, q)];
            c[(i, j)] = if conjugate { conj(v) } else { copy(v) };
        }
    }
    Ok(())
}

unsafe fn scatter<T: Arm>(
    dest: *mut u8,
    ds: Spec,
    rows: &[usize],
    cols: &[usize],
    src: *const u8,
    s: Spec,
) -> Result<(), String> {
    let x = operand::<T>(src, s, destination_span(dest, ds)?)?;
    let (b, conjugate) = x.get();
    if b.shape() != (rows.len(), cols.len()) {
        return Err("scatter source shape mismatch".into());
    }
    let mut c = T::pick_mut(destination(dest, ds)?).ok_or("scatter scalar types must match")?;
    check_indices(rows, c.nrows())?;
    check_indices(cols, c.ncols())?;
    // Column-major order: with repeated indices the last write wins, as in Julia.
    for (j, &q) in cols.iter().enumerate() {
        for (i, &p) in rows.iter().enumerate() {
            let v = &b[(i, j)];
            c[(p, q)] = if conjugate { conj(v) } else { copy(v) };
        }
    }
    Ok(())
}

/// `dest = src[rows, cols]`: gather the selected entries of the view
/// `src`/`sd` into the destination `dest`/`dd` (any destination kind) of
/// shape `(rows.len(), cols.len())` and the same scalar.
///
/// # Safety
/// Julia roots both owners and descriptors for this call.
#[julia]
pub fn view_gather_into(
    dest: *mut u8,
    dd: *const isize,
    src: *const u8,
    sd: *const isize,
    rows: &[usize],
    cols: &[usize],
) -> Result<(), String> {
    guarded(|| unsafe {
        let (ds, s) = (Spec::read(dd)?, Spec::read(sd)?);
        match source(dest, ds)? {
            Borrowed::F32(_) => gather::<f32>(dest, ds, src, s, rows, cols),
            Borrowed::F64(_) => gather::<f64>(dest, ds, src, s, rows, cols),
            Borrowed::C32(_) => gather::<faer::c32>(dest, ds, src, s, rows, cols),
            Borrowed::C64(_) => gather::<faer::c64>(dest, ds, src, s, rows, cols),
        }
    })
}

/// `dest[rows, cols] = src`: scatter the view `src`/`sd` of shape
/// `(rows.len(), cols.len())` into the destination `dest`/`dd`.
///
/// # Safety
/// Julia roots both owners and descriptors for this call.
#[julia]
pub fn view_scatter_into(
    dest: *mut u8,
    dd: *const isize,
    rows: &[usize],
    cols: &[usize],
    src: *const u8,
    sd: *const isize,
) -> Result<(), String> {
    guarded(|| unsafe {
        let (ds, s) = (Spec::read(dd)?, Spec::read(sd)?);
        match source(dest, ds)? {
            Borrowed::F32(_) => scatter::<f32>(dest, ds, rows, cols, src, s),
            Borrowed::F64(_) => scatter::<f64>(dest, ds, rows, cols, src, s),
            Borrowed::C32(_) => scatter::<faer::c32>(dest, ds, rows, cols, src, s),
            Borrowed::C64(_) => scatter::<faer::c64>(dest, ds, rows, cols, src, s),
        }
    })
}

/// Effective element strides `(row, col)` of the view `p`/`desc`, including
/// the owner's column padding, steps and transpose.
///
/// # Safety
/// Julia roots the owner and descriptor for this call.
#[julia]
pub fn view_strides(p: *const u8, desc: *const isize, out: &mut [isize]) -> Result<(), String> {
    guarded(|| unsafe {
        let [rs, cs] = out else {
            return Err("stride output must have two entries".into());
        };
        (*rs, *cs) = match source(p, Spec::read(desc)?)? {
            Borrowed::F32(a) => (a.row_stride(), a.col_stride()),
            Borrowed::F64(a) => (a.row_stride(), a.col_stride()),
            Borrowed::C32(a) => (a.row_stride(), a.col_stride()),
            Borrowed::C64(a) => (a.row_stride(), a.col_stride()),
        };
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocation_probe;
    use crate::OwnedMatrix;
    use faer::Mat;

    fn owner(m: usize, n: usize) -> Vec<isize> {
        vec![0, 0, 1, m as isize, 0, 1, n as isize, 0, 0]
    }

    #[test]
    fn gather_and_scatter_follow_julia_index_semantics() {
        let a = OwnedMatrix {
            matrix: Mat::from_fn(4, 3, |i, j| (10 * i + j) as f64),
        };
        let mut c = OwnedMatrix {
            matrix: Mat::zeros(3, 2),
        };
        let (rows, cols) = ([3usize, 0, 0], [2usize, 1]);
        let ap = (&a as *const OwnedMatrix).cast::<u8>();
        let cp = (&mut c as *mut OwnedMatrix).cast::<u8>();
        let count = allocation_probe::count(|| {
            view_gather_into(
                cp,
                owner(3, 2).as_ptr(),
                ap,
                owner(4, 3).as_ptr(),
                &rows,
                &cols,
            )
            .unwrap();
        });
        // Only the two descriptor vectors built inside the closure.
        assert_eq!(count, 2);
        assert_eq!(c.matrix[(0, 0)], 32.0);
        assert_eq!(c.matrix[(2, 1)], 1.0);

        // Repeated destination rows: the last column-major write wins.
        let mut d = OwnedMatrix {
            matrix: Mat::zeros(4, 3),
        };
        let dp = (&mut d as *mut OwnedMatrix).cast::<u8>();
        view_scatter_into(
            dp,
            owner(4, 3).as_ptr(),
            &rows,
            &cols,
            cp,
            owner(3, 2).as_ptr(),
        )
        .unwrap();
        assert_eq!(d.matrix[(0, 2)], c.matrix[(2, 0)]);
        assert_eq!(d.matrix[(3, 1)], c.matrix[(0, 1)]);

        // Out-of-bounds indices fail before any write.
        let before = d.matrix.clone();
        assert!(view_scatter_into(
            dp,
            owner(4, 3).as_ptr(),
            &[0, 4, 1],
            &cols,
            cp,
            owner(3, 2).as_ptr(),
        )
        .is_err());
        assert_eq!(d.matrix, before);

        // A source aliasing the destination is snapshot: reverse rows in place.
        view_scatter_into(
            dp,
            owner(4, 3).as_ptr(),
            &[3, 2, 1, 0],
            &[0, 1, 2],
            dp,
            owner(4, 3).as_ptr(),
        )
        .unwrap();
        assert_eq!(d.matrix, Mat::from_fn(4, 3, |i, j| before[(3 - i, j)]));
    }

    #[test]
    fn strides_report_padding_steps_and_transposes() {
        let a = OwnedMatrix {
            matrix: Mat::zeros(5, 3),
        };
        let cs = a.matrix.col_stride();
        let ap = (&a as *const OwnedMatrix).cast::<u8>();
        let mut out = [0isize; 2];
        view_strides(ap, [0, 4, -2, 3, 0, 2, 2, 1, 0].as_ptr(), &mut out).unwrap();
        assert_eq!(out, [2 * cs, -2]);
    }
}
