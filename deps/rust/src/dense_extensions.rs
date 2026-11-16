//! Dense BLAS extensions using native triangular and rectangular matmul.
use crate::generalized_selfadjoint::GeneralizedScalar;
use crate::kernels::{self, flag, start, Scalar};
use colmat::{Accum, Conj, MatMut, MatRef};
use paste::paste;

unsafe fn scale_triangle<T: Scalar>(u: u8, n: isize, beta: T, c: *mut T, ld: isize) {
    if beta == T::o() {
        return;
    }
    for j in 0..n {
        let rows = if u == b'U' { 0..j + 1 } else { j..n };
        for i in rows {
            let p = c.offset(i + j * ld);
            *p = if beta == T::z() { T::z() } else { beta * *p };
        }
    }
}
unsafe fn triangular_product<T: Scalar>(
    u: u8,
    n: isize,
    alpha: T,
    lhs: MatRef<'_, T>,
    conj_l: Conj,
    rhs: MatRef<'_, T>,
    conj_r: Conj,
    beta: T,
    c: *mut T,
    ld: isize,
) {
    use colmat::linalg::matmul::triangular::{matmul_with_conj, BlockStructure::*};
    let accum = if beta == T::z() {
        Accum::Replace
    } else {
        scale_triangle(u, n, beta, c, ld);
        Accum::Add
    };
    let dst = MatMut::from_raw_parts_mut(c, n as usize, n as usize, 1, ld);
    crate::threading::with_par(|par| {
        matmul_with_conj(
            dst,
            if u == b'U' {
                TriangularUpper
            } else {
                TriangularLower
            },
            accum,
            lhs,
            Rectangular,
            conj_l,
            rhs,
            Rectangular,
            conj_r,
            alpha,
            par,
        )
    });
}
pub(crate) unsafe fn gemmt<T: Scalar>(
    u: u8,
    ta: u8,
    tb: u8,
    n: isize,
    k: isize,
    alpha: T,
    a: *const T,
    la: isize,
    b: *const T,
    lb: isize,
    beta: T,
    c: *mut T,
    lc: isize,
) {
    if n == 0 {
        return;
    }
    if alpha == T::z() || k == 0 {
        scale_triangle(u, n, beta, c, lc);
        return;
    }
    let lhs = MatRef::from_raw_parts(
        a,
        n as usize,
        k as usize,
        if matches!(ta, b'N' | b'R') { 1 } else { la },
        if matches!(ta, b'N' | b'R') { la } else { 1 },
    );
    let rhs = MatRef::from_raw_parts(
        b,
        k as usize,
        n as usize,
        if matches!(tb, b'N' | b'R') { 1 } else { lb },
        if matches!(tb, b'N' | b'R') { lb } else { 1 },
    );
    triangular_product(
        u,
        n,
        alpha,
        lhs,
        if matches!(ta, b'C' | b'R') {
            Conj::Yes
        } else {
            Conj::No
        },
        rhs,
        if matches!(tb, b'C' | b'R') {
            Conj::Yes
        } else {
            Conj::No
        },
        beta,
        c,
        lc,
    );
}
pub(crate) unsafe fn rank2<T: GeneralizedScalar>(
    u: u8,
    n: isize,
    alpha: T,
    x: *const T,
    ix: isize,
    y: *const T,
    iy: isize,
    a: *mut T,
    la: isize,
    hermitian: bool,
) {
    if n == 0 || alpha == T::z() {
        return;
    }
    let x = x.offset(start(n, ix));
    let y = y.offset(start(n, iy));
    if hermitian {
        for i in 0..n {
            let p = a.offset(i + i * la);
            *p = T::diagonal(p);
        }
    }
    let xv = MatRef::from_raw_parts(x, n as usize, 1, ix, 0);
    let yv = MatRef::from_raw_parts(y, n as usize, 1, iy, 0);
    triangular_product(
        u,
        n,
        alpha,
        xv,
        Conj::No,
        yv.transpose(),
        if hermitian { Conj::Yes } else { Conj::No },
        T::o(),
        a,
        la,
    );
    triangular_product(
        u,
        n,
        if hermitian { alpha.cj() } else { alpha },
        yv,
        Conj::No,
        xv.transpose(),
        if hermitian { Conj::Yes } else { Conj::No },
        T::o(),
        a,
        la,
    );
    if hermitian {
        for i in 0..n {
            let p = a.offset(i + i * la);
            *p = T::diagonal(p);
        }
    }
}
macro_rules! check {($($arg:expr=>$valid:expr),+)=>{$(if !$valid{crate::invalid($arg);return;})+};}
macro_rules! triangular_exports {
    ($p:ident,$t:ty,$i:ty,$suffix:tt,$name:ident) => {
        paste! {
            #[no_mangle]
            pub unsafe extern "C" fn [<$p $name $suffix>](uplo:*const u8,ta:*const u8,tb:*const u8,
                n:*const $i,k:*const $i,alpha:*const $t,a:*const $t,lda:*const $i,
                b:*const $t,ldb:*const $i,beta:*const $t,c:*mut $t,ldc:*const $i) {
                let(u,ta,tb,n,k,la,lb,lc)=(flag(uplo),flag(ta),flag(tb),*n as isize,*k as isize,
                    *lda as isize,*ldb as isize,*ldc as isize);
                check!(1=>kernels::uplo(u),2=>kernels::trans(ta),3=>kernels::trans(tb),
                    4=>n>=0,5=>k>=0,8=>la>=1.max(if ta==b'N'{n}else{k}),
                    10=>lb>=1.max(if tb==b'N'{k}else{n}),13=>lc>=n.max(1));
                gemmt(u,ta,tb,n,k,*alpha,a,la,b,lb,*beta,c,lc);
            }
        }
    };
}
macro_rules! rank_exports {
    ($p:ident,$t:ty,$i:ty,$suffix:tt,$name:ident,$hermitian:expr)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $name $suffix>](uplo:*const u8,n:*const $i,alpha:*const $t,
            x:*const $t,incx:*const $i,y:*const $t,incy:*const $i,a:*mut $t,lda:*const $i) {
            let(u,n,ix,iy,la)=(flag(uplo),*n as isize,*incx as isize,*incy as isize,*lda as isize);
            check!(1=>kernels::uplo(u),2=>n>=0,5=>ix!=0,7=>iy!=0,9=>la>=n.max(1));
            rank2(u,n,*alpha,x,ix,y,iy,a,la,$hermitian);
        }
    }};
}
macro_rules! common {
    ($p:ident,$t:ty,$i:ty,$suffix:tt) => {
        triangular_exports!($p, $t, $i, $suffix, gemmt);
        triangular_exports!($p, $t, $i, $suffix, gemmtr);
        rank_exports!($p, $t, $i, $suffix, syr2, false);
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
rank_exports!(c, colmat::c32, i32, _, her2, true);
rank_exports!(z, colmat::c64, i32, _, her2, true);
rank_exports!(c, colmat::c32, i64, _64_, her2, true);
rank_exports!(z, colmat::c64, i64, _64_, her2, true);
macro_rules! three_m {
    ($p:ident,$t:ty,$i:ty,$suffix:tt) => {
        paste! {
            #[no_mangle]
            pub unsafe extern "C" fn [<$p gemm3m $suffix>](ta:*const u8,tb:*const u8,m:*const $i,
                n:*const $i,k:*const $i,alpha:*const $t,a:*const $t,lda:*const $i,
                b:*const $t,ldb:*const $i,beta:*const $t,c:*mut $t,ldc:*const $i) {
                // Arithmetic compatibility only: use ordinary native GEMM, not a
                // claim that the three-real-multiplication algorithm is provided.
                crate::abi::[<$p gemm $suffix>](ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
            }
        }
    };
}
three_m!(c, colmat::c32, i32, _);
three_m!(z, colmat::c64, i32, _);
three_m!(c, colmat::c32, i64, _64_);
three_m!(z, colmat::c64, i64, _64_);

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr::{null, null_mut};
    #[test]
    fn unused_operands_accept_null_without_reading() {
        unsafe {
            let mut c = [f64::NAN; 9];
            gemmt(
                b'U',
                b'N',
                b'N',
                3,
                2,
                0.0,
                null(),
                3,
                null(),
                2,
                0.0,
                c.as_mut_ptr(),
                3,
            );
            for j in 0..3 {
                for i in 0..3 {
                    if i <= j {
                        assert_eq!(c[i + j * 3], 0.0);
                    } else {
                        assert!(c[i + j * 3].is_nan());
                    }
                }
            }
            gemmt::<f64>(
                b'L',
                b'N',
                b'N',
                3,
                2,
                0.0,
                null(),
                3,
                null(),
                2,
                1.0,
                null_mut(),
                3,
            );
            gemmt::<f64>(
                b'L',
                b'N',
                b'N',
                0,
                2,
                1.0,
                null(),
                1,
                null(),
                2,
                0.0,
                null_mut(),
                1,
            );
            rank2::<colmat::c64>(
                b'U',
                3,
                Scalar::z(),
                null(),
                1,
                null(),
                -1,
                null_mut(),
                3,
                true,
            );
        }
    }
}
