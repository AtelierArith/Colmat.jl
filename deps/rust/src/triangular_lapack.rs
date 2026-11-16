//! LAPACK triangular solve/inverse/product over native structured kernels.
use crate::kernels::{flag, Scalar};
use colmat::{Conj, Mat, MatMut, MatRef};

unsafe fn singular<T: Scalar>(n: usize, a: *const T, ld: isize, unit: bool) -> usize {
    if !unit {
        for i in 0..n {
            if *a.offset(i as isize * (ld + 1)) == T::z() {
                return i + 1;
            }
        }
    }
    0
}
unsafe fn solve<T: Scalar>(
    u: u8,
    t: u8,
    unit: bool,
    n: usize,
    k: usize,
    a: *const T,
    la: isize,
    b: *mut T,
    lb: isize,
) {
    use colmat::linalg::triangular_solve::*;
    let matrix = MatRef::from_raw_parts(a, n, n, 1, la);
    let matrix = if t == b'N' {
        matrix
    } else {
        matrix.transpose()
    };
    let upper = (u == b'U') == (t == b'N');
    let conj = if t == b'C' { Conj::Yes } else { Conj::No };
    let rhs = MatMut::from_raw_parts_mut(b, n, k, 1, lb);
    crate::threading::with_par(|par| match (upper, unit) {
        (false, false) => solve_lower_triangular_in_place_with_conj(matrix, conj, rhs, par),
        (false, true) => solve_unit_lower_triangular_in_place_with_conj(matrix, conj, rhs, par),
        (true, false) => solve_upper_triangular_in_place_with_conj(matrix, conj, rhs, par),
        (true, true) => solve_unit_upper_triangular_in_place_with_conj(matrix, conj, rhs, par),
    });
}
pub(crate) unsafe fn inverse<T: Scalar>(u: u8, unit: bool, n: usize, a: *mut T, ld: isize) {
    use colmat::linalg::triangular_inverse::*;
    let mut result = Mat::<T>::zeros(n, n);
    {
        let source = MatRef::from_raw_parts(a, n, n, 1, ld);
        crate::threading::with_par(|par| match (u == b'U', unit) {
            (false, false) => invert_lower_triangular(result.as_mut(), source, par),
            (false, true) => invert_unit_lower_triangular(result.as_mut(), source, par),
            (true, false) => invert_upper_triangular(result.as_mut(), source, par),
            (true, true) => invert_unit_upper_triangular(result.as_mut(), source, par),
        });
    }
    for j in 0..n {
        for i in 0..n {
            if ((u == b'U' && i <= j) || (u == b'L' && i >= j)) && (!unit || i != j) {
                *a.offset(i as isize + j as isize * ld) = result[(i, j)];
            }
        }
    }
}
unsafe fn product<T: Scalar>(u: u8, n: usize, a: *mut T, ld: isize) {
    use colmat::linalg::matmul::triangular::{matmul_with_conj, BlockStructure::*};
    let mut result = Mat::<T>::zeros(n, n);
    {
        let source = MatRef::from_raw_parts(a, n, n, 1, ld);
        crate::threading::with_par(|par| {
            if u == b'U' {
                matmul_with_conj(
                    result.as_mut(),
                    TriangularUpper,
                    colmat::Accum::Replace,
                    source,
                    TriangularUpper,
                    Conj::No,
                    source.transpose(),
                    TriangularLower,
                    Conj::Yes,
                    T::o(),
                    par,
                );
            } else {
                matmul_with_conj(
                    result.as_mut(),
                    TriangularLower,
                    colmat::Accum::Replace,
                    source.transpose(),
                    TriangularUpper,
                    Conj::Yes,
                    source,
                    TriangularLower,
                    Conj::No,
                    T::o(),
                    par,
                );
            }
        });
    }
    for j in 0..n {
        for i in 0..n {
            if (u == b'U' && i <= j) || (u == b'L' && i >= j) {
                *a.offset(i as isize + j as isize * ld) = if i == j {
                    T::real(result[(i, j)].re())
                } else {
                    result[(i, j)]
                };
            }
        }
    }
}
use paste::paste;
macro_rules! exports {
    ($p:ident,$t:ty,$i:ty,$s:tt)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p trtrs $s>](uplo:*const u8,trans:*const u8,diag:*const u8,n:*const $i,nrhs:*const $i,a:*const $t,lda:*const $i,b:*mut $t,ldb:*const $i,info:*mut $i) {
            *info=0;let(u,t,d,nn,k,la,lb)=(flag(uplo),flag(trans),flag(diag),*n as isize,*nrhs as isize,*lda as isize,*ldb as isize);
            for(arg,valid)in[(1,matches!(u,b'U'|b'L')),(2,matches!(t,b'N'|b'T'|b'C')),(3,matches!(d,b'N'|b'U')),(4,nn>=0),(5,k>=0),(7,la>=nn.max(1)),(9,lb>=nn.max(1))] { if !valid { *info=-arg;crate::invalid(arg as i32);return; } }
            if nn==0 {return;}
            *info=singular(nn as usize,a,la,d==b'U') as $i;
            if *info!=0||k==0 {return;}
            solve(u,t,d==b'U',nn as usize,k as usize,a,la,b,lb);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p trtri $s>](uplo:*const u8,diag:*const u8,n:*const $i,a:*mut $t,lda:*const $i,info:*mut $i) {
            *info=0;let(u,d,nn,ld)=(flag(uplo),flag(diag),*n as isize,*lda as isize);
            for(arg,valid)in[(1,matches!(u,b'U'|b'L')),(2,matches!(d,b'N'|b'U')),(3,nn>=0),(5,ld>=nn.max(1))] { if !valid { *info=-arg;crate::invalid(arg as i32);return; } }
            if nn==0 {return;}
            *info=singular(nn as usize,a,ld,d==b'U') as $i;
            if *info==0 {inverse(u,d==b'U',nn as usize,a,ld);}
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p trti2 $s>](uplo:*const u8,diag:*const u8,n:*const $i,a:*mut $t,lda:*const $i,info:*mut $i) {
            *info=0;let(u,d,nn,ld)=(flag(uplo),flag(diag),*n as isize,*lda as isize);
            for(arg,valid)in[(1,matches!(u,b'U'|b'L')),(2,matches!(d,b'N'|b'U')),(3,nn>=0),(5,ld>=nn.max(1))] { if !valid { *info=-arg;crate::invalid(arg as i32);return; } }
            // TRTI2 assumes nonsingularity and has no positive INFO contract.
            if nn>0 { inverse(u,d==b'U',nn as usize,a,ld); }
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p lauum $s>](uplo:*const u8,n:*const $i,a:*mut $t,lda:*const $i,info:*mut $i) {
            *info=0;let(u,nn,ld)=(flag(uplo),*n as isize,*lda as isize);
            for(arg,valid)in[(1,matches!(u,b'U'|b'L')),(2,nn>=0),(4,ld>=nn.max(1))] { if !valid { *info=-arg;crate::invalid(arg as i32);return; } }
            if nn>0 {
                if let Some(result)=crate::triangular_product::maybe_saved(0,u,nn as i64,a,ld as i64) {
                    if <$i>::try_from(result).is_err(){crate::invalid(-2);*info=1;}else{*info=result as $i;}return;
                }
                product(u,nn as usize,a,ld);
            }
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p lauu2 $s>](uplo:*const u8,n:*const $i,a:*mut $t,lda:*const $i,info:*mut $i) {
            *info=0;let(u,nn,ld)=(flag(uplo),*n as isize,*lda as isize);
            for(arg,valid)in[(1,matches!(u,b'U'|b'L')),(2,nn>=0),(4,ld>=nn.max(1))] { if !valid { *info=-arg;crate::invalid(arg as i32);return; } }
            if nn>0 {
                if let Some(result)=crate::triangular_product::maybe_saved(1,u,nn as i64,a,ld as i64) {
                    if <$i>::try_from(result).is_err(){crate::invalid(-2);*info=1;}else{*info=result as $i;}return;
                }
                product(u,nn as usize,a,ld);
            }
        }
    }};
}
exports!(s, f32, i32, _);
exports!(d, f64, i32, _);
exports!(c, colmat::c32, i32, _);
exports!(z, colmat::c64, i32, _);
exports!(s, f32, i64, _64_);
exports!(d, f64, i64, _64_);
exports!(c, colmat::c32, i64, _64_);
exports!(z, colmat::c64, i64, _64_);
