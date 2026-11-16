//! Compact-WY QR: logical A/T/V extents and independent caller strides.
use super::{
    ge_nancheck, row_to_column, self_adjoint::zeros, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR,
};
use crate::kernels::Scalar;

fn factor_geometry(
    layout: i32,
    m: i64,
    n: i64,
    nb: Option<i64>,
    lda: i64,
    ldt: i64,
) -> Result<(), i64> {
    if !matches!(layout, ROW_MAJOR | COL_MAJOR) {
        return Err(-1);
    }
    let shift = if nb.is_some() { 1 } else { 0 };
    if layout == ROW_MAJOR {
        if lda < n {
            return Err(-5 - shift);
        }
        if ldt < if nb.is_some() { m.min(n) } else { n } {
            return Err(-7 - shift);
        }
    }
    if let Some(block) = nb {
        if m < 0 {
            return Err(-2);
        }
        if n < 0 {
            return Err(-3);
        }
        if block < 1 || (m.min(n) > 0 && block > m.min(n)) {
            return Err(-4);
        }
    } else {
        if n < 0 {
            return Err(-3);
        }
        if m < n {
            return Err(-2);
        }
    }
    if layout == COL_MAJOR {
        if lda < m.max(1) {
            return Err(-5 - shift);
        }
        if ldt < nb.unwrap_or(n.max(1)) {
            return Err(-7 - shift);
        }
    }
    Ok(())
}
unsafe fn factor_work<T: Scalar>(
    layout: i32,
    m: i64,
    n: i64,
    nb: Option<i64>,
    a: *mut T,
    lda: i64,
    t: *mut T,
    ldt: i64,
    mut call: impl FnMut(*mut T, i64, *mut T, i64) -> i64,
) -> i64 {
    if let Err(e) = factor_geometry(layout, m, n, nb, lda, ldt) {
        return e;
    }
    if layout == COL_MAJOR {
        return call(a, lda, t, ldt);
    }
    let rows = nb.unwrap_or(n.max(1));
    let cols = if nb.is_some() { m.min(n) } else { n };
    if m == 0 || n == 0 {
        return call(a, m.max(1), t, rows);
    }
    let mut at = match row_to_column(m as usize, n as usize, a, lda as usize) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let count = match (rows as usize).checked_mul(cols as usize) {
        Some(v) => v,
        None => return -1011,
    };
    let mut tt = match zeros::<T>(count) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let info = call(at.as_mut_ptr(), m, tt.as_mut_ptr(), rows);
    if info >= 0 {
        for i in 0..m as usize {
            for j in 0..n as usize {
                *a.add(i * lda as usize + j) = at[i + j * m as usize];
            }
        }
        for i in 0..rows as usize {
            for j in 0..cols as usize {
                *t.add(i * ldt as usize + j) = tt[i + j * rows as usize];
            }
        }
    }
    info
}
unsafe fn factor_nan<T: Scalar>(
    layout: i32,
    m: i64,
    n: i64,
    a: *const T,
    lda: i64,
    blocked: bool,
) -> Result<(), i64> {
    if !matches!(layout, ROW_MAJOR | COL_MAJOR) {
        return Err(-1);
    }
    if LAPACKE_get_nancheck() != 0 && ge_nancheck(layout, m, n, a, lda)? {
        return Err(if blocked { -5 } else { -4 });
    }
    Ok(())
}
fn apply_geometry(
    layout: i32,
    side: u8,
    trans: u8,
    m: i64,
    n: i64,
    k: i64,
    nb: i64,
    ldv: i64,
    ldt: i64,
    ldc: i64,
    complex: bool,
) -> Result<(), i64> {
    if !matches!(layout, ROW_MAJOR | COL_MAJOR) {
        return Err(-1);
    }
    if layout == ROW_MAJOR {
        if ldc < n {
            return Err(-13);
        }
        // Logical row-major T is NB-by-K, including partial last blocks.
        if ldt < nb.max(k) {
            return Err(-11);
        }
        if ldv < k {
            return Err(-9);
        }
    }
    let side = side.to_ascii_uppercase();
    let trans = trans.to_ascii_uppercase();
    if !matches!(side, b'L' | b'R') {
        return Err(-2);
    }
    if trans != b'N' && trans != if complex { b'C' } else { b'T' } {
        return Err(-3);
    }
    if m < 0 {
        return Err(-4);
    }
    if n < 0 {
        return Err(-5);
    }
    let order = if side == b'L' { m } else { n };
    if k < 0 || k > order {
        return Err(-6);
    }
    if nb < 1 || (k > 0 && nb > k) {
        return Err(-7);
    }
    if layout == COL_MAJOR {
        if ldv < order.max(1) {
            return Err(-9);
        }
        if ldt < nb {
            return Err(-11);
        }
        if ldc < m.max(1) {
            return Err(-13);
        }
    }
    Ok(())
}
unsafe fn apply_work<T: Scalar>(
    layout: i32,
    side: u8,
    trans: u8,
    m: i64,
    n: i64,
    k: i64,
    nb: i64,
    v: *const T,
    ldv: i64,
    t: *const T,
    ldt: i64,
    c: *mut T,
    ldc: i64,
    mut call: impl FnMut(*const T, i64, *const T, i64, *mut T, i64) -> i64,
) -> i64 {
    if let Err(e) = apply_geometry(layout, side, trans, m, n, k, nb, ldv, ldt, ldc, T::COMPLEX) {
        return e;
    }
    if layout == COL_MAJOR {
        return call(v, ldv, t, ldt, c, ldc);
    }
    let order = if side.to_ascii_uppercase() == b'L' {
        m
    } else {
        n
    };
    if m == 0 || n == 0 || k == 0 {
        return call(v, order.max(1), t, nb, c, m.max(1));
    }
    let vt = match row_to_column(order as usize, k as usize, v, ldv as usize) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let tt = match row_to_column(nb as usize, k as usize, t, ldt as usize) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let mut ct = match row_to_column(m as usize, n as usize, c, ldc as usize) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let info = call(vt.as_ptr(), order, tt.as_ptr(), nb, ct.as_mut_ptr(), m);
    if info >= 0 {
        for i in 0..m as usize {
            for j in 0..n as usize {
                *c.add(i * ldc as usize + j) = ct[i + j * m as usize];
            }
        }
    }
    info
}
unsafe fn apply_nan<T: Scalar>(
    layout: i32,
    side: u8,
    m: i64,
    n: i64,
    k: i64,
    nb: i64,
    v: *const T,
    ldv: i64,
    t: *const T,
    ldt: i64,
    c: *const T,
    ldc: i64,
) -> Result<(), i64> {
    if !matches!(layout, ROW_MAJOR | COL_MAJOR) {
        return Err(-1);
    }
    if LAPACKE_get_nancheck() != 0 {
        let order = if side.to_ascii_uppercase() == b'L' {
            m
        } else {
            n
        };
        if ge_nancheck(layout, m, n, c, ldc)? {
            return Err(-12);
        }
        if ge_nancheck(layout, nb, k, t, ldt)? {
            return Err(-10);
        }
        if ge_nancheck(layout, order, k, v, ldv)? {
            return Err(-8);
        }
    }
    Ok(())
}
macro_rules! plain_exports {
    ($p:ident,$family:ident,$t:ty,$j:ty,[$($suffix:tt)*],$f:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $family _work $($suffix)*>](layout:i32,m:$j,n:$j,a:*mut $t,lda:$j,t:*mut $t,ldt:$j)->$j{
            factor_work(layout,m as i64,n as i64,None,a,lda as i64,t,ldt as i64,|a,lda,t,ldt|{let mut info:$j=0;crate::qr::[<$p $family $f>](&m,&n,a,&(lda as $j),t,&(ldt as $j),&mut info);if info<0{(info-1) as i64}else{info as i64}}) as $j
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $family $($suffix)*>](layout:i32,m:$j,n:$j,a:*mut $t,lda:$j,t:*mut $t,ldt:$j)->$j{
            if let Err(e)=factor_nan(layout,m as i64,n as i64,a,lda as i64,false){return e as $j;}
            [<LAPACKE_ $p $family _work $($suffix)*>](layout,m,n,a,lda,t,ldt)
        }
    }};
}
macro_rules! exports {
    ($p:ident,$t:ty,$j:ty,[$($suffix:tt)*],$f:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p geqrt_work $($suffix)*>](layout:i32,m:$j,n:$j,nb:$j,a:*mut $t,lda:$j,t:*mut $t,ldt:$j,workspace:*mut $t)->$j{
            factor_work(layout,m as i64,n as i64,Some(nb as i64),a,lda as i64,t,ldt as i64,|a,lda,t,ldt|{let mut info:$j=0;crate::qr::[<$p geqrt $f>](&m,&n,&nb,a,&(lda as $j),t,&(ldt as $j),workspace,&mut info);if info<0{(info-1) as i64}else{info as i64}}) as $j
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p geqrt $($suffix)*>](layout:i32,m:$j,n:$j,nb:$j,a:*mut $t,lda:$j,t:*mut $t,ldt:$j)->$j{
            if let Err(e)=factor_nan(layout,m as i64,n as i64,a,lda as i64,true){return e as $j;}
            if let Err(e)=factor_geometry(layout,m as i64,n as i64,Some(nb as i64),lda as i64,ldt as i64){return e as $j;}
            let length=if m==0||n==0{1}else{match (nb as usize).checked_mul(n as usize){Some(v)=>v,None=>return -1010}};
            let mut workspace=match zeros::<$t>(length){Ok(v)=>v,Err(_)=>return -1010};
            [<LAPACKE_ $p geqrt_work $($suffix)*>](layout,m,n,nb,a,lda,t,ldt,workspace.as_mut_ptr())
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p gemqrt_work $($suffix)*>](layout:i32,side:i8,trans:i8,m:$j,n:$j,k:$j,nb:$j,v:*const $t,ldv:$j,t:*const $t,ldt:$j,c:*mut $t,ldc:$j,workspace:*mut $t)->$j{
            apply_work(layout,side as u8,trans as u8,m as i64,n as i64,k as i64,nb as i64,v,ldv as i64,t,ldt as i64,c,ldc as i64,|v,ldv,t,ldt,c,ldc|{let mut info:$j=0;crate::qr::[<$p gemqrt $f>](&(side as u8),&(trans as u8),&m,&n,&k,&nb,v,&(ldv as $j),t,&(ldt as $j),c,&(ldc as $j),workspace,&mut info);if info<0{(info-1) as i64}else{info as i64}}) as $j
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p gemqrt $($suffix)*>](layout:i32,side:i8,trans:i8,m:$j,n:$j,k:$j,nb:$j,v:*const $t,ldv:$j,t:*const $t,ldt:$j,c:*mut $t,ldc:$j)->$j{
            if let Err(e)=apply_nan(layout,side as u8,m as i64,n as i64,k as i64,nb as i64,v,ldv as i64,t,ldt as i64,c,ldc as i64){return e as $j;}
            if let Err(e)=apply_geometry(layout,side as u8,trans as u8,m as i64,n as i64,k as i64,nb as i64,ldv as i64,ldt as i64,ldc as i64,<$t>::COMPLEX){return e as $j;}
            let nw=if (side as u8).to_ascii_uppercase()==b'L'{n}else{m};
            let length=if m==0||n==0||k==0{1}else{match (nw as usize).checked_mul(nb as usize){Some(v)=>v,None=>return -1010}};
            let mut workspace=match zeros::<$t>(length){Ok(v)=>v,Err(_)=>return -1010};
            [<LAPACKE_ $p gemqrt_work $($suffix)*>](layout,side,trans,m,n,k,nb,v,ldv,t,ldt,c,ldc,workspace.as_mut_ptr())
        }
    }
    plain_exports!($p,geqrt2,$t,$j,[$($suffix)*],$f);
    plain_exports!($p,geqrt3,$t,$j,[$($suffix)*],$f);
    };
}
exports!(s, f32, i32, [], _);
exports!(d, f64, i32, [], _);
exports!(c, colmat::c32, i32, [], _);
exports!(z, colmat::c64, i32, [], _);
exports!(s, f32, i64, [64_], _64_);
exports!(d, f64, i64, [64_], _64_);
exports!(c, colmat::c32, i64, [64_], _64_);
exports!(z, colmat::c64, i64, [64_], _64_);
