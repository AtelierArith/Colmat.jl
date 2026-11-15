//! Checked LP64/ILP64 contracts for triangular-pentagonal LAPACK entries.
use crate::householder::ReflectorScalar;
use crate::kernels::{flag, matrix_addressable, Scalar};
use crate::lapack::LapackInt;
use paste::paste;
unsafe fn error<I: LapackInt>(info: *mut I, arg: i64) {
    *info = I::from_i64(-arg);
    crate::invalid(arg as i32);
}
fn trans<T: Scalar>(tr: u8) -> bool {
    tr == b'N' || tr == if T::COMPLEX { b'C' } else { b'T' }
}
fn combined_fits<T>(left: bool, m: isize, n: isize, k: isize) -> bool {
    let Some(order) = k.checked_add(if left { m } else { n }) else {
        return false;
    };
    let (rows, cols) = if left { (order, n) } else { (m, order) };
    matrix_addressable::<T>(rows, cols, rows.max(1))
        && matrix_addressable::<T>(order, k, order.max(1))
}
unsafe fn factor<T: ReflectorScalar, I: LapackInt>(
    lq: bool,
    m: isize,
    n: isize,
    l: isize,
    nb: Option<isize>,
    a: *mut T,
    lda: isize,
    b: *mut T,
    ldb: isize,
    t: *mut T,
    ldt: isize,
    info: *mut I,
) {
    *info = I::from_i64(0);
    let k = if lq { m } else { n };
    let shift = i64::from(nb.is_some());
    for (arg, ok) in [(1, m >= 0), (2, n >= 0), (3, l >= 0 && l <= m.min(n))] {
        if !ok {
            error(info, arg);
            return;
        }
    }
    if let Some(nb) = nb {
        if nb < 1 || (nb > k && k > 0) {
            error(info, 4);
            return;
        }
    }
    let nb = nb.unwrap_or(k.max(1));
    for (arg, ok) in [
        (5 + shift, lda >= k.max(1)),
        (7 + shift, ldb >= m.max(1)),
        (9 + shift, ldt >= if shift == 0 { k.max(1) } else { nb }),
    ] {
        if !ok {
            error(info, arg);
            return;
        }
    }
    if m == 0 || n == 0 {
        return;
    }
    let last = (k - 1) % nb + 1;
    for (arg, ok) in [
        (5 + shift, matrix_addressable::<T>(k, k, lda)),
        (7 + shift, matrix_addressable::<T>(m, n, ldb)),
        (9 + shift, matrix_addressable::<T>(last, k, ldt)),
    ] {
        if !ok {
            error(info, arg);
            return;
        }
    }
    let Some(order) = m.checked_add(n) else {
        error(info, 5 + shift);
        return;
    };
    if !matrix_addressable::<T>(order, k, order) {
        error(info, 5 + shift);
        return;
    }
    crate::pentagonal::factor(
        lq,
        m as usize,
        n as usize,
        l as usize,
        nb as usize,
        a,
        lda,
        b,
        ldb,
        t,
        ldt,
    );
}
unsafe fn apply<T: Scalar, I: LapackInt>(
    lq: bool,
    side: u8,
    tr: u8,
    m: isize,
    n: isize,
    k: isize,
    l: isize,
    nb: isize,
    v: *const T,
    ldv: isize,
    t: *const T,
    ldt: isize,
    a: *mut T,
    lda: isize,
    b: *mut T,
    ldb: isize,
    info: *mut I,
) {
    *info = I::from_i64(0);
    let left = side == b'L';
    let tail = if left { m } else { n };
    for (arg, ok) in [
        (1, matches!(side, b'L' | b'R')),
        (2, trans::<T>(tr)),
        (3, m >= 0),
        (4, n >= 0),
        (5, k >= 0),
        (6, l >= 0 && l <= k && l <= tail),
        (7, nb >= 1 && (nb <= k || k == 0)),
        (9, ldv >= if lq { k } else { tail.max(1) }),
        (11, ldt >= nb),
        (13, lda >= if left { k.max(1) } else { m.max(1) }),
        (15, ldb >= m.max(1)),
    ] {
        if !ok {
            error(info, arg);
            return;
        }
    }
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    let (vr, vc) = if lq { (k, tail) } else { (tail, k) };
    let (ar, ac) = if left { (k, n) } else { (m, k) };
    for (arg, ok) in [
        (9, matrix_addressable::<T>(vr, vc, ldv)),
        (11, matrix_addressable::<T>((k - 1) % nb + 1, k, ldt)),
        (13, matrix_addressable::<T>(ar, ac, lda)),
        (15, matrix_addressable::<T>(m, n, ldb)),
    ] {
        if !ok {
            error(info, arg);
            return;
        }
    }
    if !combined_fits::<T>(left, m, n, k) {
        error(info, 5);
        return;
    }
    crate::pentagonal::apply(
        lq,
        side,
        tr,
        m as usize,
        n as usize,
        k as usize,
        l as usize,
        nb as usize,
        v,
        ldv,
        t,
        ldt,
        a,
        lda,
        b,
        ldb,
    );
}
unsafe fn block<T: Scalar>(
    side: u8,
    tr: u8,
    direct: u8,
    store: u8,
    m: isize,
    n: isize,
    k: isize,
    l: isize,
    v: *const T,
    ldv: isize,
    t: *const T,
    ldt: isize,
    a: *mut T,
    lda: isize,
    b: *mut T,
    ldb: isize,
    ldwork: isize,
) {
    let left = side == b'L';
    let tail = if left { m } else { n };
    for (arg, ok) in [
        (1, matches!(side, b'L' | b'R')),
        (2, trans::<T>(tr)),
        (3, matches!(direct, b'F' | b'B')),
        (4, matches!(store, b'C' | b'R')),
        (5, m >= 0),
        (6, n >= 0),
        (7, k >= 0),
        (8, l >= 0 && l <= k && l <= tail),
        (10, ldv >= if store == b'C' { tail.max(1) } else { k }),
        (12, ldt >= k),
        (14, lda >= if left { k.max(1) } else { m.max(1) }),
        (16, ldb >= m.max(1)),
        (18, ldwork >= if left { k } else { m }),
    ] {
        if !ok {
            crate::invalid(arg);
            return;
        }
    }
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    let (vr, vc) = if store == b'R' { (k, tail) } else { (tail, k) };
    let (ar, ac) = if left { (k, n) } else { (m, k) };
    for (arg, ok) in [
        (10, matrix_addressable::<T>(vr, vc, ldv)),
        (12, matrix_addressable::<T>(k, k, ldt)),
        (14, matrix_addressable::<T>(ar, ac, lda)),
        (16, matrix_addressable::<T>(m, n, ldb)),
    ] {
        if !ok {
            crate::invalid(arg);
            return;
        }
    }
    if !combined_fits::<T>(left, m, n, k) {
        crate::invalid(7);
        return;
    }
    crate::pentagonal::block(
        side, tr, direct, store, m as usize, n as usize, k as usize, l as usize, v, ldv, t, ldt, a,
        lda, b, ldb,
    );
}
macro_rules! factor_exports {($p:ident,$name:ident,$name2:ident,$lq:expr,$t:ty,$i:ty,$s:tt)=>{paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p $name $s>](m:*const $i,n:*const $i,l:*const $i,nb:*const $i,a:*mut $t,lda:*const $i,b:*mut $t,ldb:*const $i,t:*mut $t,ldt:*const $i,_work:*mut $t,info:*mut $i) {factor($lq,*m as isize,*n as isize,*l as isize,Some(*nb as isize),a,*lda as isize,b,*ldb as isize,t,*ldt as isize,info);}
    #[no_mangle] pub unsafe extern "C" fn [<$p $name2 $s>](m:*const $i,n:*const $i,l:*const $i,a:*mut $t,lda:*const $i,b:*mut $t,ldb:*const $i,t:*mut $t,ldt:*const $i,info:*mut $i) {factor($lq,*m as isize,*n as isize,*l as isize,None,a,*lda as isize,b,*ldb as isize,t,*ldt as isize,info);}
}};}
macro_rules! apply_exports {($p:ident,$name:ident,$lq:expr,$t:ty,$i:ty,$s:tt)=>{paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p $name $s>](side:*const u8,tr:*const u8,m:*const $i,n:*const $i,k:*const $i,l:*const $i,nb:*const $i,v:*const $t,ldv:*const $i,t:*const $t,ldt:*const $i,a:*mut $t,lda:*const $i,b:*mut $t,ldb:*const $i,_work:*mut $t,info:*mut $i) {apply($lq,flag(side),flag(tr),*m as isize,*n as isize,*k as isize,*l as isize,*nb as isize,v,*ldv as isize,t,*ldt as isize,a,*lda as isize,b,*ldb as isize,info);}
}};}
macro_rules! exports {($p:ident,$t:ty,$i:ty,$s:tt)=>{paste!{
    factor_exports!($p,tpqrt,tpqrt2,false,$t,$i,$s);
    factor_exports!($p,tplqt,tplqt2,true,$t,$i,$s);
    apply_exports!($p,tpmqrt,false,$t,$i,$s);
    apply_exports!($p,tpmlqt,true,$t,$i,$s);
    #[no_mangle] pub unsafe extern "C" fn [<$p tprfb $s>](side:*const u8,tr:*const u8,direct:*const u8,store:*const u8,m:*const $i,n:*const $i,k:*const $i,l:*const $i,v:*const $t,ldv:*const $i,t:*const $t,ldt:*const $i,a:*mut $t,lda:*const $i,b:*mut $t,ldb:*const $i,_work:*mut $t,ldwork:*const $i) {block(flag(side),flag(tr),flag(direct),flag(store),*m as isize,*n as isize,*k as isize,*l as isize,v,*ldv as isize,t,*ldt as isize,a,*lda as isize,b,*ldb as isize,*ldwork as isize);}
}};}
exports!(s, f32, i32, _);
exports!(d, f64, i32, _);
exports!(c, colmat::c32, i32, _);
exports!(z, colmat::c64, i32, _);
exports!(s, f32, i64, _64_);
exports!(d, f64, i64, _64_);
exports!(c, colmat::c32, i64, _64_);
exports!(z, colmat::c64, i64, _64_);
