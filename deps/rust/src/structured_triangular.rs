//! Band/packed triangular products and solves using native row products.
//! Rows are gathered only over their stored intervals; structural zeros and
//! unit diagonals never become dense matrix entries that could read NaNs.
use crate::kernels::{flag, Scalar};
use crate::storage_layout::TriangleStorage;
use crate::structured_blas::vector_start;
use colmat::{Accum, Conj, Mat, MatMut, MatRef};
use paste::paste;

#[derive(Clone, Copy)]
enum Storage {
    Band { width: usize, ld: usize },
    Packed,
}
impl Storage {
    fn width(self, n: usize) -> usize {
        match self {
            Self::Band { width, .. } => width,
            Self::Packed => n,
        }
    }
    fn increment_argument(self) -> i32 {
        match self {
            Self::Band { .. } => 9,
            Self::Packed => 7,
        }
    }
    fn index(self, n: usize, lower: bool, i: usize, j: usize) -> usize {
        match self {
            Self::Band { width, ld } => j * ld + if lower { i - j } else { width + i - j },
            Self::Packed => {
                TriangleStorage::Packed.segment(n, lower, j).0 + if lower { i - j } else { i }
            }
        }
    }
}
unsafe fn with_off_diagonal<T: Scalar>(
    storage: Storage,
    lower: bool,
    tr: u8,
    n: usize,
    i: usize,
    a: *const T,
    x: *const T,
    inc: isize,
    action: impl FnOnce(MatRef<'_, T>, usize, Conj),
) {
    let upper = (!lower) == (tr == b'N');
    let width = storage.width(n);
    let (first, end) = if upper {
        (i + 1, i.saturating_add(width).saturating_add(1).min(n))
    } else {
        (i.saturating_sub(width), i)
    };
    if first == end {
        return;
    }
    if tr == b'N' {
        // The no-transpose BLAS algorithm skips zero columns, including all
        // accesses to that column's stored matrix entries.
        if (first..end).all(|j| *x.offset(j as isize * inc) == T::z()) {
            return;
        }
        let row = Mat::from_fn(1, end - first, |_, k| {
            let j = first + k;
            if *x.offset(j as isize * inc) == T::z() {
                T::z()
            } else {
                *a.add(storage.index(n, lower, i, j))
            }
        });
        action(row.as_ref(), first, Conj::No);
    } else {
        // A row of op(A) is a contiguous stored column of A.
        let row = MatRef::from_raw_parts(
            a.add(storage.index(n, lower, first, i)),
            1,
            end - first,
            0,
            1,
        );
        action(row, first, if tr == b'C' { Conj::Yes } else { Conj::No });
    }
}
unsafe fn divide_diagonal<T: Scalar>(a: *const T, x: *mut T, conjugated: bool) {
    let diagonal = MatRef::from_raw_parts(a, 1, 1, 1, 0);
    let rhs = MatMut::from_raw_parts_mut(x, 1, 1, 1, 0);
    crate::threading::with_par(|par| {
        colmat::linalg::triangular_solve::solve_lower_triangular_in_place_with_conj(
            diagonal,
            if conjugated { Conj::Yes } else { Conj::No },
            rhs,
            par,
        );
    });
}
unsafe fn apply<T: Scalar>(
    lower: bool,
    tr: u8,
    unit: bool,
    n: usize,
    a: *const T,
    storage: Storage,
    x: *mut T,
    inc: isize,
    solve: bool,
) {
    if unit && (n == 1 || storage.width(n) == 0) {
        return;
    }
    let Some(start) = vector_start(n as isize, inc) else {
        crate::invalid(storage.increment_argument());
        return;
    };
    let inc = if n == 1 { 1 } else { inc };
    let x = x.offset(start);
    if !solve {
        let input = Mat::from_fn(n, 1, |i, _| *x.offset(i as isize * inc));
        let original = input.as_ref().as_ptr();
        for i in 0..n {
            let value = input[(i, 0)];
            let dst = x.offset(i as isize * inc);
            *dst = if unit || (tr == b'N' && value == T::z()) {
                value
            } else {
                let diagonal = *a.add(storage.index(n, lower, i, i));
                value * if tr == b'C' { diagonal.cj() } else { diagonal }
            };
            with_off_diagonal(
                storage,
                lower,
                tr,
                n,
                i,
                a,
                original,
                1,
                |row, first, conj| {
                    let rhs = input.as_ref().submatrix(first, 0, row.ncols(), 1);
                    let output = MatMut::from_raw_parts_mut(dst, 1, 1, 1, 0);
                    crate::threading::with_par(|par| {
                        colmat::linalg::matmul::matmul_with_conj(
                            output,
                            Accum::Add,
                            row,
                            conj,
                            rhs,
                            Conj::No,
                            T::o(),
                            par,
                        )
                    });
                },
            );
        }
    } else {
        let upper = (!lower) == (tr == b'N');
        for step in 0..n {
            let i = if upper { n - 1 - step } else { step };
            let dst = x.offset(i as isize * inc);
            with_off_diagonal(storage, lower, tr, n, i, a, x, inc, |row, first, conj| {
                // Solved input entries exclude the single mutable output.
                let rhs =
                    MatRef::from_raw_parts(x.offset(first as isize * inc), row.ncols(), 1, inc, 0);
                let output = MatMut::from_raw_parts_mut(dst, 1, 1, 1, 0);
                crate::threading::with_par(|par| {
                    colmat::linalg::matmul::matmul_with_conj(
                        output,
                        Accum::Add,
                        row,
                        conj,
                        rhs,
                        Conj::No,
                        -T::o(),
                        par,
                    )
                });
            });
            if !unit && !(tr == b'N' && *dst == T::z()) {
                divide_diagonal(a.add(storage.index(n, lower, i, i)), dst, tr == b'C');
            }
        }
    }
}
/// Internal LAPACK adapter: an upper band factor with a contiguous RHS.
/// Argument validation and zero-size handling belong to the caller.
pub(crate) unsafe fn solve_upper_band<T: Scalar>(
    n: usize,
    width: usize,
    a: *const T,
    ld: usize,
    x: *mut T,
    tr: u8,
) {
    apply(
        false,
        tr,
        false,
        n,
        a,
        Storage::Band { width, ld },
        x,
        1,
        true,
    );
}

macro_rules! check {($($arg:expr=>$valid:expr),+)=>{$(if !$valid{crate::invalid($arg);return;})+};}
macro_rules! exports {
 ($p:ident,$t:ty,$i:ty,$suffix:tt,$band:ident,$packed:ident,$solve:expr)=>{paste!{
  #[no_mangle]
  pub unsafe extern "C" fn [<$p $band $suffix>](u:*const u8,tr:*const u8,d:*const u8,n:*const $i,k:*const $i,
      a:*const $t,ld:*const $i,x:*mut $t,inc:*const $i) {
    let(u,tr,d,n,k,ld,inc)=(flag(u),flag(tr),flag(d),*n as isize,*k as isize,*ld as isize,*inc as isize);
    check!(1=>matches!(u,b'U'|b'L'),2=>matches!(tr,b'N'|b'T'|b'C'),3=>matches!(d,b'N'|b'U'),4=>n>=0,5=>k>=0,
        7=>k.checked_add(1).is_some_and(|required|ld>=required),9=>inc!=0);
    if n==0 {return;}
    apply(u==b'L',tr,d==b'U',n as usize,a,Storage::Band{width:k as usize,ld:ld as usize},x,inc,$solve);
  }
  #[no_mangle]
  pub unsafe extern "C" fn [<$p $packed $suffix>](u:*const u8,tr:*const u8,d:*const u8,n:*const $i,
      a:*const $t,x:*mut $t,inc:*const $i) {
    let(u,tr,d,n,inc)=(flag(u),flag(tr),flag(d),*n as isize,*inc as isize);
    check!(1=>matches!(u,b'U'|b'L'),2=>matches!(tr,b'N'|b'T'|b'C'),3=>matches!(d,b'N'|b'U'),4=>n>=0,7=>inc!=0);
    if n==0 {return;}
    apply(u==b'L',tr,d==b'U',n as usize,a,Storage::Packed,x,inc,$solve);
  }
 }};
}
macro_rules! common {
    ($p:ident,$t:ty,$i:ty,$suffix:tt) => {
        exports!($p, $t, $i, $suffix, tbmv, tpmv, false);
        exports!($p, $t, $i, $suffix, tbsv, tpsv, true);
    };
}
common!(s, f32, i32, _);
common!(d, f64, i32, _);
common!(c, colmat::c32, i32, _);
common!(z, colmat::c64, i32, _);
common!(s, f32, i64, _64_);
common!(d, f64, i64, _64_);
common!(c, colmat::c32, i64, _64_);
common!(z, colmat::c64, i64, _64_);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn empty_and_unit_diagonal_only_calls_need_no_array_storage() {
        unsafe {
            dtbmv_(
                b"U".as_ptr(),
                b"N".as_ptr(),
                b"N".as_ptr(),
                &0,
                &2,
                std::ptr::null(),
                &3,
                std::ptr::null_mut(),
                &1,
            );
            ztpsv_64_(
                b"L".as_ptr(),
                b"C".as_ptr(),
                b"U".as_ptr(),
                &1,
                std::ptr::null(),
                std::ptr::null_mut(),
                &i64::MIN,
            );
            ztbmv_64_(
                b"U".as_ptr(),
                b"T".as_ptr(),
                b"U".as_ptr(),
                &3,
                &0,
                std::ptr::null(),
                &1,
                std::ptr::null_mut(),
                &1,
            );
        }
    }
    #[test]
    fn no_transpose_zero_columns_do_not_read_matrix_storage() {
        unsafe {
            let mut x = [-0.0; 3];
            dtbmv_(
                b"U".as_ptr(),
                b"N".as_ptr(),
                b"N".as_ptr(),
                &3,
                &1,
                std::ptr::null(),
                &2,
                x.as_mut_ptr(),
                &-1,
            );
            dtbsv_(
                b"L".as_ptr(),
                b"N".as_ptr(),
                b"N".as_ptr(),
                &3,
                &1,
                std::ptr::null(),
                &2,
                x.as_mut_ptr(),
                &1,
            );
            dtpmv_(
                b"L".as_ptr(),
                b"N".as_ptr(),
                b"N".as_ptr(),
                &3,
                std::ptr::null(),
                x.as_mut_ptr(),
                &1,
            );
            dtpsv_(
                b"U".as_ptr(),
                b"N".as_ptr(),
                b"N".as_ptr(),
                &3,
                std::ptr::null(),
                x.as_mut_ptr(),
                &-1,
            );
            assert!(x.iter().all(|v| v.to_bits() == (-0.0_f64).to_bits()));
            let a = [2.0, f64::NAN, f64::NAN, 0.5, 0.5, 2.0];
            x = [1.0, 0.0, 2.0];
            dtpmv_(
                b"U".as_ptr(),
                b"N".as_ptr(),
                b"N".as_ptr(),
                &3,
                a.as_ptr(),
                x.as_mut_ptr(),
                &1,
            );
            assert_eq!(x, [3.0, 1.0, 4.0]);
            dtpsv_(
                b"U".as_ptr(),
                b"N".as_ptr(),
                b"N".as_ptr(),
                &3,
                a.as_ptr(),
                x.as_mut_ptr(),
                &1,
            );
            assert_eq!(x, [1.0, 0.0, 2.0]);
        }
    }
    #[test]
    fn singular_diagonal_follows_blas_floating_point_behavior() {
        unsafe {
            let mut x = [1.0];
            dtbsv_(
                b"L".as_ptr(),
                b"N".as_ptr(),
                b"N".as_ptr(),
                &1,
                &0,
                [0.0].as_ptr(),
                &1,
                x.as_mut_ptr(),
                &1,
            );
            assert_eq!(x, [f64::INFINITY]);
        }
    }
    #[test]
    fn integer_width_and_stride_errors_precede_output_access() {
        unsafe {
            dtpsv_64_(
                b"U".as_ptr(),
                b"N".as_ptr(),
                b"N".as_ptr(),
                &(-(1_i64 << 32) + 3),
                std::ptr::null(),
                std::ptr::null_mut(),
                &1,
            );
            assert_eq!(crate::colmat_blas_last_error(), 4);
            dtbmv_64_(
                b"U".as_ptr(),
                b"N".as_ptr(),
                b"N".as_ptr(),
                &0,
                &i64::MAX,
                std::ptr::null(),
                &i64::MAX,
                std::ptr::null_mut(),
                &1,
            );
            assert_eq!(crate::colmat_blas_last_error(), 7);
            let mut x = [777.0; 3];
            dtpmv_64_(
                b"U".as_ptr(),
                b"N".as_ptr(),
                b"N".as_ptr(),
                &3,
                std::ptr::null(),
                x.as_mut_ptr(),
                &i64::MAX,
            );
            assert_eq!(crate::colmat_blas_last_error(), 7);
            assert_eq!(x, [777.0; 3]);
        }
    }
}
