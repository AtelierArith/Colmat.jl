//! Standard-form congruences shared by generalized self-adjoint LAPACK drivers.
use crate::kernels::{flag, Scalar};
use colmat::{Conj, Mat, MatRef};

pub(crate) trait GeneralizedScalar: Scalar {
    unsafe fn diagonal(pointer: *const Self) -> Self;
}
macro_rules! real_scalar {
    ($t:ty) => {
        impl GeneralizedScalar for $t {
            unsafe fn diagonal(pointer: *const Self) -> Self {
                *pointer
            }
        }
    };
}
macro_rules! complex_scalar {
    ($t:ty) => {
        impl GeneralizedScalar for $t {
            unsafe fn diagonal(pointer: *const Self) -> Self {
                Self::new(core::ptr::addr_of!((*pointer).re).read(), 0.0)
            }
        }
    };
}
real_scalar!(f32);
real_scalar!(f64);
complex_scalar!(colmat::c32);
complex_scalar!(colmat::c64);

pub(crate) unsafe fn hermitian<T: GeneralizedScalar>(
    u: u8,
    n: usize,
    a: *const T,
    ld: isize,
) -> Mat<T> {
    Mat::from_fn(n, n, |i, j| {
        if i == j {
            T::diagonal(a.offset(i as isize + j as isize * ld))
        } else if (u == b'L' && i > j) || (u == b'U' && i < j) {
            *a.offset(i as isize + j as isize * ld)
        } else {
            (*a.offset(j as isize + i as isize * ld)).cj()
        }
    })
}
// The lower Cholesky factor is L for UPLO=L and U^H for UPLO=U.
// Complex diagonal imaginary parts are not read. Real factors are borrowed;
// complex factors need a selected-triangle representation with real diagonal.
pub(crate) unsafe fn with_lower<T: GeneralizedScalar, R>(
    u: u8,
    n: usize,
    b: *const T,
    ld: isize,
    action: impl FnOnce(MatRef<'_, T>) -> R,
) -> R {
    if T::COMPLEX {
        let factor = Mat::from_fn(n, n, |i, j| {
            if i < j {
                T::z()
            } else if i == j {
                T::diagonal(b.offset(i as isize + j as isize * ld))
            } else if u == b'L' {
                *b.offset(i as isize + j as isize * ld)
            } else {
                (*b.offset(j as isize + i as isize * ld)).cj()
            }
        });
        action(factor.as_ref())
    } else {
        let factor = crate::lapack::input_ref(b, n, n, ld);
        action(if u == b'L' {
            factor
        } else {
            factor.transpose()
        })
    }
}
pub(crate) fn standardize<T: GeneralizedScalar>(
    itype: u8,
    mut matrix: Mat<T>,
    lower: MatRef<'_, T>,
) -> Mat<T> {
    use colmat::linalg::matmul::triangular::{matmul_with_conj, BlockStructure::*};
    use colmat::linalg::triangular_solve::*;
    let n = matrix.nrows();
    crate::threading::with_par(|mut par| {
        if itype == 1 {
            solve_lower_triangular_in_place(lower, matrix.as_mut(), par.reborrow());
            // X L^H = M is equivalent to conj(L) X^T = M^T.
            solve_lower_triangular_in_place_with_conj(
                lower,
                Conj::Yes,
                matrix.as_mut().transpose_mut(),
                par,
            );
        } else {
            let mut intermediate = Mat::<T>::zeros(n, n);
            matmul_with_conj(
                intermediate.as_mut(),
                Rectangular,
                colmat::Accum::Replace,
                matrix.as_ref(),
                Rectangular,
                Conj::No,
                lower,
                TriangularLower,
                Conj::No,
                T::o(),
                par.reborrow(),
            );
            matmul_with_conj(
                matrix.as_mut(),
                Rectangular,
                colmat::Accum::Replace,
                lower.transpose(),
                TriangularUpper,
                Conj::Yes,
                intermediate.as_ref(),
                Rectangular,
                Conj::No,
                T::o(),
                par,
            );
        }
    });
    for i in 0..n {
        matrix[(i, i)] = T::real(matrix[(i, i)].re());
    }
    matrix
}
pub(crate) unsafe fn output<T: GeneralizedScalar>(matrix: &Mat<T>, u: u8, a: *mut T, ld: isize) {
    let n = matrix.nrows();
    for j in 0..n {
        for i in 0..n {
            if (u == b'U' && i <= j) || (u == b'L' && i >= j) {
                *a.offset(i as isize + j as isize * ld) = matrix[(i, j)];
            }
        }
    }
}
use paste::paste;
macro_rules! exports {
    ($p:ident,$blocked:ident,$unblocked:ident,$t:ty,$i:ty,$s:tt)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $blocked $s>](itype:*const $i,uplo:*const u8,n:*const $i,a:*mut $t,lda:*const $i,b:*const $t,ldb:*const $i,info:*mut $i) {
            *info=0;let(kind,u,nn,la,lb)=(*itype as isize,flag(uplo),*n as isize,*lda as isize,*ldb as isize);
            for(arg,valid)in[(1,(1..=3).contains(&kind)),(2,matches!(u,b'U'|b'L')),(3,nn>=0),(5,la>=nn.max(1)),(7,lb>=nn.max(1))] {if !valid {*info=-arg;crate::invalid(arg as i32);return;}}
            if nn==0{return;}
            let size=nn as usize;
            let matrix=hermitian(u,size,a,la);
            let result=with_lower(u,size,b,lb,|lower|standardize(kind as u8,matrix,lower));
            output(&result,u,a,la);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $unblocked $s>](itype:*const $i,uplo:*const u8,n:*const $i,a:*mut $t,lda:*const $i,b:*const $t,ldb:*const $i,info:*mut $i) {
            [<$p $blocked $s>](itype,uplo,n,a,lda,b,ldb,info);
        }
    }};
}
exports!(s, sygst, sygs2, f32, i32, _);
exports!(d, sygst, sygs2, f64, i32, _);
exports!(c, hegst, hegs2, colmat::c32, i32, _);
exports!(z, hegst, hegs2, colmat::c64, i32, _);
exports!(s, sygst, sygs2, f32, i64, _64_);
exports!(d, sygst, sygs2, f64, i64, _64_);
exports!(c, hegst, hegs2, colmat::c32, i64, _64_);
exports!(z, hegst, hegs2, colmat::c64, i64, _64_);
