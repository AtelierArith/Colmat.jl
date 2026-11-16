//! Apply QR/LQ/QL/RQ reflectors with const inputs and independent C strides.
use super::{
    ge_nancheck, row_to_column, self_adjoint::zeros, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR,
};
use crate::kernels::Scalar;

fn geometry(
    layout: i32,
    side: u8,
    trans: u8,
    m: i64,
    n: i64,
    k: i64,
    lda: i64,
    ldc: i64,
    lwork: i64,
    rows: bool,
    complex: bool,
) -> Result<(), i64> {
    if !matches!(layout, ROW_MAJOR | COL_MAJOR) {
        return Err(-1);
    }
    let side = side.to_ascii_uppercase();
    let trans = trans.to_ascii_uppercase();
    let order = if side == b'L' { m } else { n };
    if layout == ROW_MAJOR {
        if lda < if rows { order } else { k } {
            return Err(-8);
        }
        if ldc < n {
            return Err(-11);
        }
    }
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
    if k < 0 || k > order {
        return Err(-6);
    }
    if layout == COL_MAJOR {
        if lda < (if rows { k } else { order }).max(1) {
            return Err(-8);
        }
        if ldc < m.max(1) {
            return Err(-11);
        }
    }
    if lwork != -1 && lwork < (if side == b'L' { n } else { m }).max(1) {
        return Err(-13);
    }
    Ok(())
}
unsafe fn work<T: Scalar>(
    layout: i32,
    side: u8,
    trans: u8,
    m: i64,
    n: i64,
    k: i64,
    a: *const T,
    lda: i64,
    c: *mut T,
    ldc: i64,
    lwork: i64,
    rows: bool,
    mut call: impl FnMut(*const T, i64, *mut T, i64) -> i64,
) -> i64 {
    if let Err(e) = geometry(
        layout,
        side,
        trans,
        m,
        n,
        k,
        lda,
        ldc,
        lwork,
        rows,
        T::COMPLEX,
    ) {
        return e;
    }
    if layout == COL_MAJOR {
        return call(a, lda, c, ldc);
    }
    let order = if side.to_ascii_uppercase() == b'L' {
        m
    } else {
        n
    };
    let (ar, ac) = if rows { (k, order) } else { (order, k) };
    let (la, lc) = (ar.max(1), m.max(1));
    if lwork == -1 || m == 0 || n == 0 || k == 0 {
        return call(a, la, c, lc);
    }
    let at = match row_to_column(ar as usize, ac as usize, a, lda as usize) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let mut ct = match row_to_column(m as usize, n as usize, c, ldc as usize) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let info = call(at.as_ptr(), la, ct.as_mut_ptr(), lc);
    if info >= 0 {
        for i in 0..m as usize {
            for j in 0..n as usize {
                *c.add(i * ldc as usize + j) = ct[i + j * m as usize];
            }
        }
    }
    info
}
unsafe fn high_nan<T: Scalar>(
    layout: i32,
    side: u8,
    m: i64,
    n: i64,
    k: i64,
    a: *const T,
    lda: i64,
    tau: *const T,
    c: *const T,
    ldc: i64,
    rows: bool,
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
        let (ar, ac) = if rows { (k, order) } else { (order, k) };
        if ge_nancheck(layout, ar, ac, a, lda)? {
            return Err(-7);
        }
        if ge_nancheck(layout, m, n, c, ldc)? {
            return Err(-10);
        }
        if k > 0 && !tau.is_null() {
            if (k as usize)
                .checked_mul(core::mem::size_of::<T>())
                .is_none_or(|v| v > isize::MAX as usize)
            {
                return Err(-1011);
            }
            for i in 0..k as usize {
                let x = *tau.add(i);
                if x != x {
                    return Err(-9);
                }
            }
        }
    }
    Ok(())
}
macro_rules! exports {
    ($p:ident,$family:ident,$module:ident,$rows:expr,$t:ty,$j:ty,[$($suffix:tt)*],$f:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $family _work $($suffix)*>](layout:i32,side:i8,trans:i8,m:$j,n:$j,k:$j,a:*const $t,lda:$j,tau:*const $t,c:*mut $t,ldc:$j,workspace:*mut $t,lwork:$j)->$j{
            work(layout,side as u8,trans as u8,m as i64,n as i64,k as i64,a,lda as i64,c,ldc as i64,lwork as i64,$rows,|a,lda,c,ldc|{
                let mut info:$j=0;crate::$module::[<$p $family $f>](&(side as u8),&(trans as u8),&m,&n,&k,a,&(lda as $j),tau,c,&(ldc as $j),workspace,&lwork,&mut info);
                if info<0{(info-1) as i64}else{info as i64}
            }) as $j
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $family $($suffix)*>](layout:i32,side:i8,trans:i8,m:$j,n:$j,k:$j,a:*const $t,lda:$j,tau:*const $t,c:*mut $t,ldc:$j)->$j{
            if let Err(e)=high_nan(layout,side as u8,m as i64,n as i64,k as i64,a,lda as i64,tau,c,ldc as i64,$rows){return e as $j;}
            let mut query=<$t>::z();let info=[<LAPACKE_ $p $family _work $($suffix)*>](layout,side,trans,m,n,k,a,lda,tau,c,ldc,&mut query,-1);if info!=0{return info;}
            let minimum=(if (side as u8).to_ascii_uppercase()==b'L'{n}else{m}).max(1) as f64;
            let requested=query.re();if !requested.is_finite()||requested<1.{return -1010;}let length=requested.max(minimum);if !length.is_finite()||length<1.||length>=isize::MAX as f64||length > <$j>::MAX as f64{return -1010;}
            let mut workspace=match zeros::<$t>(length as usize){Ok(v)=>v,Err(_)=>return -1010};
            [<LAPACKE_ $p $family _work $($suffix)*>](layout,side,trans,m,n,k,a,lda,tau,c,ldc,workspace.as_mut_ptr(),length as $j)
        }
    }};
}
exports!(s, ormqr, qr, false, f32, i32, [], _);
exports!(d, ormqr, qr, false, f64, i32, [], _);
exports!(c, unmqr, qr, false, colmat::c32, i32, [], _);
exports!(z, unmqr, qr, false, colmat::c64, i32, [], _);
exports!(s, ormqr, qr, false, f32, i64, [64_], _64_);
exports!(d, ormqr, qr, false, f64, i64, [64_], _64_);
exports!(c, unmqr, qr, false, colmat::c32, i64, [64_], _64_);
exports!(z, unmqr, qr, false, colmat::c64, i64, [64_], _64_);
exports!(s, ormlq, lq, true, f32, i32, [], _);
exports!(d, ormlq, lq, true, f64, i32, [], _);
exports!(c, unmlq, lq, true, colmat::c32, i32, [], _);
exports!(z, unmlq, lq, true, colmat::c64, i32, [], _);
exports!(s, ormlq, lq, true, f32, i64, [64_], _64_);
exports!(d, ormlq, lq, true, f64, i64, [64_], _64_);
exports!(c, unmlq, lq, true, colmat::c32, i64, [64_], _64_);
exports!(z, unmlq, lq, true, colmat::c64, i64, [64_], _64_);
exports!(s, ormql, reverse_qr, false, f32, i32, [], _);
exports!(d, ormql, reverse_qr, false, f64, i32, [], _);
exports!(c, unmql, reverse_qr, false, colmat::c32, i32, [], _);
exports!(z, unmql, reverse_qr, false, colmat::c64, i32, [], _);
exports!(s, ormql, reverse_qr, false, f32, i64, [64_], _64_);
exports!(d, ormql, reverse_qr, false, f64, i64, [64_], _64_);
exports!(c, unmql, reverse_qr, false, colmat::c32, i64, [64_], _64_);
exports!(z, unmql, reverse_qr, false, colmat::c64, i64, [64_], _64_);
exports!(s, ormrq, reverse_qr, true, f32, i32, [], _);
exports!(d, ormrq, reverse_qr, true, f64, i32, [], _);
exports!(c, unmrq, reverse_qr, true, colmat::c32, i32, [], _);
exports!(z, unmrq, reverse_qr, true, colmat::c64, i32, [], _);
exports!(s, ormrq, reverse_qr, true, f32, i64, [64_], _64_);
exports!(d, ormrq, reverse_qr, true, f64, i64, [64_], _64_);
exports!(c, unmrq, reverse_qr, true, colmat::c32, i64, [64_], _64_);
exports!(z, unmrq, reverse_qr, true, colmat::c64, i64, [64_], _64_);
