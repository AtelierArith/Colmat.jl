//! Packed/RFP representations around native LLT and Cholesky solves.
//! All dense work is owned by the adapter; native LLT's opposite-triangle
//! scratch never aliases a caller's packed storage.
use crate::generalized_selfadjoint::GeneralizedScalar;
use crate::storage_layout::TriangleStorage;
use colmat::{Mat, MatMut};

unsafe fn read<T: GeneralizedScalar>(
    n: usize,
    lower: bool,
    layout: TriangleStorage,
    a: *const T,
) -> Mat<T> {
    let mut work = Mat::<T>::zeros(n, n);
    for j in 0..n {
        let (offset, stride, conjugated) = layout.segment(n, lower, j);
        let first = if lower { j } else { 0 };
        let length = if lower { n - j } else { j + 1 };
        for k in 0..length {
            let i = first + k;
            let p = a.offset(offset as isize + k as isize * stride);
            work[(i, j)] = if i == j {
                T::diagonal(p)
            } else if conjugated {
                (*p).cj()
            } else {
                *p
            };
        }
    }
    work
}
unsafe fn write<T: GeneralizedScalar>(
    n: usize,
    lower: bool,
    layout: TriangleStorage,
    a: *mut T,
    work: &Mat<T>,
) {
    for j in 0..n {
        let (offset, stride, conjugated) = layout.segment(n, lower, j);
        let first = if lower { j } else { 0 };
        let length = if lower { n - j } else { j + 1 };
        for k in 0..length {
            let i = first + k;
            let value = if i == j {
                T::real(work[(i, j)].re())
            } else if conjugated {
                work[(i, j)].cj()
            } else {
                work[(i, j)]
            };
            *a.offset(offset as isize + k as isize * stride) = value;
        }
    }
}
pub(crate) unsafe fn factor<T: GeneralizedScalar>(
    n: usize,
    lower: bool,
    layout: TriangleStorage,
    a: *mut T,
) -> usize {
    if n == 0 {
        return 0;
    }
    let mut work = read(n, lower, layout, a);
    let info = if lower {
        crate::lapack::cholesky(work.as_mut())
    } else {
        crate::lapack::cholesky(work.as_mut().transpose_mut())
    };
    write(n, lower, layout, a, &work);
    info
}
pub(crate) unsafe fn solve<T: GeneralizedScalar>(
    n: usize,
    nrhs: usize,
    lower: bool,
    layout: TriangleStorage,
    a: *const T,
    b: *mut T,
    ldb: usize,
) {
    if n == 0 || nrhs == 0 {
        return;
    }
    let work = read(n, lower, layout, a);
    crate::lapack::solve_cholesky(
        work.as_ref(),
        MatMut::from_raw_parts_mut(b, n, nrhs, 1, ldb as isize),
        if lower { b'L' } else { b'U' },
    );
}
pub(crate) unsafe fn inverse<T: GeneralizedScalar>(
    n: usize,
    lower: bool,
    layout: TriangleStorage,
    a: *mut T,
) -> usize {
    if n == 0 {
        return 0;
    }
    // Check all diagonal entries before overwriting any factor, as TPTRI does.
    for j in 0..n {
        let (offset, stride, _) = layout.segment(n, lower, j);
        let p = a.offset(offset as isize + if lower { 0 } else { j as isize * stride });
        if T::diagonal(p) == T::z() {
            return j + 1;
        }
    }
    let work = read(n, lower, layout, a);
    let mut result = Mat::<T>::identity(n, n);
    crate::lapack::solve_cholesky(
        work.as_ref(),
        result.as_mut(),
        if lower { b'L' } else { b'U' },
    );
    write(n, lower, layout, a, &result);
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::Scalar;
    #[test]
    fn upper_packed_factor_solve_and_inverse_are_native() {
        unsafe {
            let mut ap = [4.0f64, 1.0, 3.0];
            assert_eq!(
                factor(2, false, TriangleStorage::Packed, ap.as_mut_ptr()),
                0
            );
            assert_eq!(ap[0], 2.0);
            assert_eq!(ap[1], 0.5);
            assert!((ap[2] - (2.75f64).sqrt()).abs() < 1e-15);
            let mut b = [6.0, 7.0];
            solve(
                2,
                1,
                false,
                TriangleStorage::Packed,
                ap.as_ptr(),
                b.as_mut_ptr(),
                2,
            );
            assert!((b[0] - 1.0).abs() < 1e-14);
            assert!((b[1] - 2.0).abs() < 1e-14);
            assert_eq!(
                inverse(2, false, TriangleStorage::Packed, ap.as_mut_ptr()),
                0
            );
            for (actual, expected) in ap.iter().zip([3.0 / 11.0, -1.0 / 11.0, 4.0 / 11.0]) {
                assert!((actual - expected).abs() < 1e-14);
            }
        }
    }
    #[test]
    fn complex_packed_ignores_imaginary_diagonals_and_conjugates_solves() {
        use colmat::c64;
        unsafe {
            let mut ap = [
                c64::new(4.0, f64::NAN),
                c64::new(1.0, 2.0),
                c64::new(3.0, f64::NAN),
            ];
            assert_eq!(
                factor(2, false, TriangleStorage::Packed, ap.as_mut_ptr()),
                0
            );
            assert_eq!(ap[0], c64::new(2.0, 0.0));
            assert_eq!(ap[2].im, 0.0);
            let x = [c64::new(1.0, 1.0), c64::new(2.0, -1.0)];
            let mut b = [
                c64::new(4.0, 0.0) * x[0] + c64::new(1.0, 2.0) * x[1],
                c64::new(1.0, -2.0) * x[0] + c64::new(3.0, 0.0) * x[1],
            ];
            solve(
                2,
                1,
                false,
                TriangleStorage::Packed,
                ap.as_ptr(),
                b.as_mut_ptr(),
                2,
            );
            for i in 0..2 {
                assert!((b[i] - x[i]).magnitude() < 1e-13);
            }
        }
    }
    #[test]
    fn zero_diagonal_inverse_preserves_the_entire_factor() {
        unsafe {
            let mut ap = [2.0f64, 0.5, 0.0];
            let before = ap;
            assert_eq!(
                inverse(2, false, TriangleStorage::Packed, ap.as_mut_ptr()),
                2
            );
            assert_eq!(ap, before);
            assert_eq!(
                factor::<f64>(0, true, TriangleStorage::Packed, std::ptr::null_mut()),
                0
            );
            solve::<f64>(
                9,
                0,
                true,
                TriangleStorage::Packed,
                std::ptr::null(),
                std::ptr::null_mut(),
                9,
            );
        }
    }
}

#[cfg(test)]
mod rfp_tests {
    use super::*;
    #[test]
    fn two_by_two_rfp_layouts_factor_solve_and_invert() {
        unsafe {
            for lower in [false, true] {
                for transposed in [false, true] {
                    let layout = TriangleStorage::Rfp { transposed };
                    let mut a = if lower {
                        [3.0f64, 4.0, 1.0]
                    } else {
                        [1.0, 3.0, 4.0]
                    };
                    assert_eq!(factor(2, lower, layout, a.as_mut_ptr()), 0);
                    let mut b = [6.0, 7.0];
                    solve(2, 1, lower, layout, a.as_ptr(), b.as_mut_ptr(), 2);
                    assert!((b[0] - 1.0).abs() < 1e-14);
                    assert!((b[1] - 2.0).abs() < 1e-14);
                    assert_eq!(inverse(2, lower, layout, a.as_mut_ptr()), 0);
                    let expected = if lower {
                        [4.0 / 11.0, 3.0 / 11.0, -1.0 / 11.0]
                    } else {
                        [-1.0 / 11.0, 4.0 / 11.0, 3.0 / 11.0]
                    };
                    for i in 0..3 {
                        assert!((a[i] - expected[i]).abs() < 1e-14);
                    }
                }
            }
        }
    }
}

fn fits_dense<T: GeneralizedScalar>(n: usize) -> bool {
    n.checked_mul(n)
        .and_then(|count| count.checked_mul(std::mem::size_of::<T>()))
        .is_some_and(|bytes| bytes <= isize::MAX as usize)
}
fn rfp_flag<T: GeneralizedScalar>(trr: u8) -> bool {
    trr == b'N' || trr == if T::COMPLEX { b'C' } else { b'T' }
}
macro_rules! valid {($info:expr,$($arg:expr=>$test:expr),+)=>{$(if !$test{crate::lapack::bad(&mut *$info,$arg);return;})+};}
macro_rules! packed_factor_export {($p:ident,$family:ident,$op:ident,$t:ty,$i:ty,$s:tt)=>{paste::paste!{
    #[no_mangle]
    pub unsafe extern "C" fn [<$p $family $s>](u:*const u8,n:*const $i,a:*mut $t,info:*mut $i) {
        *info=0;let u=crate::kernels::flag(u);valid!(info,1=>crate::kernels::uplo(u),2=>*n>=0);
        let n=*n as usize;if n==0{return;}valid!(info,2=>fits_dense::<$t>(n));
        *info=$op(n,u==b'L',TriangleStorage::Packed,a)as $i;
    }
}};}
macro_rules! rfp_factor_export {($p:ident,$family:ident,$op:ident,$t:ty,$i:ty,$s:tt)=>{paste::paste!{
    #[no_mangle]
    pub unsafe extern "C" fn [<$p $family $s>](trr:*const u8,u:*const u8,n:*const $i,a:*mut $t,info:*mut $i) {
        *info=0;let(trr,u)=(crate::kernels::flag(trr),crate::kernels::flag(u));
        valid!(info,1=>rfp_flag::<$t>(trr),2=>crate::kernels::uplo(u),3=>*n>=0);
        let n=*n as usize;if n==0{return;}valid!(info,3=>fits_dense::<$t>(n));
        *info=$op(n,u==b'L',TriangleStorage::Rfp{transposed:trr!=b'N'},a)as $i;
    }
}};}
macro_rules! exports {($p:ident,$t:ty,$i:ty,$s:tt)=>{
    packed_factor_export!($p,pptrf,factor,$t,$i,$s);packed_factor_export!($p,pptri,inverse,$t,$i,$s);
    rfp_factor_export!($p,pftrf,factor,$t,$i,$s);rfp_factor_export!($p,pftri,inverse,$t,$i,$s);
    paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p pptrs $s>](u:*const u8,n:*const $i,nrhs:*const $i,a:*const $t,b:*mut $t,ldb:*const $i,info:*mut $i) {
            *info=0;let u=crate::kernels::flag(u);
            valid!(info,1=>crate::kernels::uplo(u),2=>*n>=0,3=>*nrhs>=0,6=>*ldb>=(*n).max(1));
            let(n,nrhs)=(*n as usize,*nrhs as usize);if n==0||nrhs==0{return;}valid!(info,2=>fits_dense::<$t>(n));
            solve(n,nrhs,u==b'L',TriangleStorage::Packed,a,b,*ldb as usize);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p pftrs $s>](trr:*const u8,u:*const u8,n:*const $i,nrhs:*const $i,a:*const $t,b:*mut $t,ldb:*const $i,info:*mut $i) {
            *info=0;let(trr,u)=(crate::kernels::flag(trr),crate::kernels::flag(u));
            valid!(info,1=>rfp_flag::<$t>(trr),2=>crate::kernels::uplo(u),3=>*n>=0,4=>*nrhs>=0,7=>*ldb>=(*n).max(1));
            let(n,nrhs)=(*n as usize,*nrhs as usize);if n==0||nrhs==0{return;}valid!(info,3=>fits_dense::<$t>(n));
            solve(n,nrhs,u==b'L',TriangleStorage::Rfp{transposed:trr!=b'N'},a,b,*ldb as usize);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p ppsv $s>](u:*const u8,n:*const $i,nrhs:*const $i,a:*mut $t,b:*mut $t,ldb:*const $i,info:*mut $i) {
            *info=0;let flag=crate::kernels::flag(u);
            valid!(info,1=>crate::kernels::uplo(flag),2=>*n>=0,3=>*nrhs>=0,6=>*ldb>=(*n).max(1));
            [<$p pptrf $s>](u,n,a,info);if *info==0{[<$p pptrs $s>](u,n,nrhs,a,b,ldb,info);}
        }
    }
};}
exports!(s, f32, i32, _);
exports!(d, f64, i32, _);
exports!(c, colmat::c32, i32, _);
exports!(z, colmat::c64, i32, _);
exports!(s, f32, i64, _64_);
exports!(d, f64, i64, _64_);
exports!(c, colmat::c32, i64, _64_);
exports!(z, colmat::c64, i64, _64_);

#[cfg(test)]
mod abi_tests {
    use super::*;
    use crate::kernels::Scalar;
    #[test]
    fn empty_rhs_and_unrepresentable_workspace_do_not_read_arrays() {
        unsafe {
            let mut info = 777i64;
            dpptrf_64_(b"L".as_ptr(), &0, std::ptr::null_mut(), &mut info);
            assert_eq!(info, 0);
            dpptrs_64_(
                b"U".as_ptr(),
                &((1i64 << 32) + 1),
                &0,
                std::ptr::null(),
                std::ptr::null_mut(),
                &((1i64 << 32) + 1),
                &mut info,
            );
            assert_eq!(info, 0);
            dpftrs_64_(
                b"T".as_ptr(),
                b"L".as_ptr(),
                &((1i64 << 32) + 1),
                &0,
                std::ptr::null(),
                std::ptr::null_mut(),
                &((1i64 << 32) + 1),
                &mut info,
            );
            assert_eq!(info, 0);
            dpptrf_64_(b"L".as_ptr(), &i64::MAX, std::ptr::null_mut(), &mut info);
            assert_eq!(info, -2);
            assert_eq!(crate::colmat_blas_last_error(), 2);
            dpftrf_64_(
                b"N".as_ptr(),
                b"L".as_ptr(),
                &(-(1i64 << 32) + 1),
                std::ptr::null_mut(),
                &mut info,
            );
            assert_eq!(info, -3);
            assert_eq!(crate::colmat_blas_last_error(), 3);
            zpftrf_64_(
                b"T".as_ptr(),
                b"L".as_ptr(),
                &0,
                std::ptr::null_mut(),
                &mut info,
            );
            assert_eq!(info, -1);
            assert_eq!(crate::colmat_blas_last_error(), 1);
        }
    }
    #[test]
    fn complex_rfp_folded_and_conjugate_transposed_layouts_solve() {
        use colmat::c64;
        let x = [c64::new(1.0, 1.0), c64::new(2.0, -1.0)];
        for lower in [false, true] {
            for transposed in [false, true] {
                let off = c64::new(1.0, if lower ^ transposed { -2.0 } else { 2.0 });
                let mut a = if lower {
                    [c64::new(3.0, f64::NAN), c64::new(4.0, f64::NAN), off]
                } else {
                    [off, c64::new(3.0, f64::NAN), c64::new(4.0, f64::NAN)]
                };
                let mut b = [
                    c64::new(4.0, 0.0) * x[0] + c64::new(1.0, 2.0) * x[1],
                    c64::new(1.0, -2.0) * x[0] + c64::new(3.0, 0.0) * x[1],
                ];
                let layout = TriangleStorage::Rfp { transposed };
                unsafe {
                    assert_eq!(factor(2, lower, layout, a.as_mut_ptr()), 0);
                    solve(2, 1, lower, layout, a.as_ptr(), b.as_mut_ptr(), 2);
                }
                for i in 0..2 {
                    assert!((b[i] - x[i]).magnitude() < 1e-13);
                }
            }
        }
    }
}
