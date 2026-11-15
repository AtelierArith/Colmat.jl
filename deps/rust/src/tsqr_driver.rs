//! Public LATSQR and TSQR explicit-Q workspace/ABI contracts.
use crate::householder::ReflectorScalar;
use crate::kernels::{matrix_addressable, Scalar};
use crate::lapack::LapackInt;
use paste::paste;
unsafe fn error<I: LapackInt>(info: *mut I, arg: i64) {
    *info = I::from_i64(-arg);
    crate::invalid(arg as i32);
}
fn slots(m: isize, n: isize, mb: isize) -> isize {
    if mb > n && mb < m {
        (m - n) / (mb - n) + isize::from((m - n) % (mb - n) != 0)
    } else {
        1
    }
}
unsafe fn factor<T: ReflectorScalar, I: LapackInt>(
    m: isize,
    n: isize,
    mb: isize,
    nb: isize,
    a: *mut T,
    lda: isize,
    t: *mut T,
    ldt: isize,
    work: *mut T,
    lwork: isize,
    info: *mut I,
) {
    *info = I::from_i64(0);
    for (arg, ok) in [
        (1, m >= 0),
        (2, n >= 0 && m >= n),
        (3, mb >= 1),
        (4, nb >= 1 && (nb <= n || n == 0)),
        (6, lda >= m.max(1)),
        (8, ldt >= nb),
    ] {
        if !ok {
            error(info, arg);
            return;
        }
    }
    let Some(minimum) = (if n == 0 { Some(1) } else { n.checked_mul(nb) }) else {
        error(info, 10);
        return;
    };
    if lwork != -1 && lwork < minimum {
        error(info, 10);
        return;
    }
    *work = crate::lapack::workspace::<T>(minimum);
    if lwork == -1 || n == 0 {
        return;
    }
    if !matrix_addressable::<T>(m, n, lda) {
        error(info, 6);
        return;
    }
    let Some(columns) = n.checked_mul(slots(m, n, mb)) else {
        error(info, 8);
        return;
    };
    if !matrix_addressable::<T>(nb, columns, ldt) {
        error(info, 8);
        return;
    }
    crate::tsqr::factor(
        m as usize,
        n as usize,
        mb as usize,
        nb as usize,
        a,
        lda,
        t,
        ldt,
    );
}
unsafe fn generate<T: Scalar, I: LapackInt>(
    row: bool,
    m: isize,
    n: isize,
    mb: isize,
    nb: isize,
    a: *mut T,
    lda: isize,
    t: *const T,
    ldt: isize,
    work: *mut T,
    lwork: isize,
    info: *mut I,
) {
    *info = I::from_i64(0);
    for (arg, ok) in [
        (1, m >= 0),
        (2, n >= 0 && m >= n),
        (3, mb > n),
        (4, nb >= 1),
        (6, lda >= m.max(1)),
        (8, ldt >= nb.min(n).max(1)),
    ] {
        if !ok {
            error(info, arg);
            return;
        }
    }
    let block = nb.min(n);
    let Some(optimum) = (if row {
        block.checked_mul(block.max(n - block))
    } else {
        m.checked_add(block).and_then(|x| x.checked_mul(n))
    }) else {
        error(info, 10);
        return;
    };
    let minimum = if row { 1 } else { optimum.max(2) };
    if lwork != -1 && lwork < minimum {
        error(info, 10);
        return;
    }
    *work = crate::lapack::workspace::<T>(optimum.max(if row { 1 } else { 2 }));
    if lwork == -1 || n == 0 {
        return;
    }
    if !matrix_addressable::<T>(m, n, lda) {
        error(info, 6);
        return;
    }
    let Some(columns) = n.checked_mul(slots(m, n, mb)) else {
        error(info, 8);
        return;
    };
    if !matrix_addressable::<T>(block, columns, ldt) {
        error(info, 8);
        return;
    }
    let q = crate::tsqr::thin_q(
        m as usize,
        n as usize,
        mb as usize,
        block as usize,
        a,
        lda,
        t,
        ldt,
    );
    for j in 0..n {
        for i in 0..m {
            *a.offset(i + j * lda) = q[(i + j * m) as usize];
        }
    }
}
macro_rules! exports {($p:ident,$org:ident,$t:ty,$i:ty,$s:tt)=>{paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p latsqr $s>](m:*const $i,n:*const $i,mb:*const $i,nb:*const $i,a:*mut $t,lda:*const $i,t:*mut $t,ldt:*const $i,work:*mut $t,lwork:*const $i,info:*mut $i) {factor(*m as isize,*n as isize,*mb as isize,*nb as isize,a,*lda as isize,t,*ldt as isize,work,*lwork as isize,info);}
    #[no_mangle] pub unsafe extern "C" fn [<$p $org $s>](m:*const $i,n:*const $i,mb:*const $i,nb:*const $i,a:*mut $t,lda:*const $i,t:*const $t,ldt:*const $i,work:*mut $t,lwork:*const $i,info:*mut $i) {generate(false,*m as isize,*n as isize,*mb as isize,*nb as isize,a,*lda as isize,t,*ldt as isize,work,*lwork as isize,info);}
    #[no_mangle] pub unsafe extern "C" fn [<$p $org _row $s>](m:*const $i,n:*const $i,mb:*const $i,nb:*const $i,a:*mut $t,lda:*const $i,t:*const $t,ldt:*const $i,work:*mut $t,lwork:*const $i,info:*mut $i) {generate(true,*m as isize,*n as isize,*mb as isize,*nb as isize,a,*lda as isize,t,*ldt as isize,work,*lwork as isize,info);}
}};}
exports!(s, orgtsqr, f32, i32, _);
exports!(d, orgtsqr, f64, i32, _);
exports!(c, ungtsqr, colmat::c32, i32, _);
exports!(z, ungtsqr, colmat::c64, i32, _);
exports!(s, orgtsqr, f32, i64, _64_);
exports!(d, orgtsqr, f64, i64, _64_);
exports!(c, ungtsqr, colmat::c32, i64, _64_);
exports!(z, ungtsqr, colmat::c64, i64, _64_);
