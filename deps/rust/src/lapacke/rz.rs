//! RZ uses upper-trapezoidal factors and sparse row-reflector tails.
use super::{
    ge_nancheck, row_to_column, self_adjoint::zeros, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR,
};
use crate::kernels::Scalar;

fn factor_geometry(layout: i32, m: i64, n: i64, lda: i64) -> Result<(), i64> {
    if !matches!(layout, ROW_MAJOR | COL_MAJOR) {
        return Err(-1);
    }
    if layout == ROW_MAJOR && lda < n {
        return Err(-5);
    }
    if m < 0 {
        return Err(-2);
    }
    if n < m {
        return Err(-3);
    }
    if layout == COL_MAJOR && lda < m.max(1) {
        return Err(-5);
    }
    Ok(())
}
unsafe fn factor_work<T: Scalar>(
    layout: i32,
    m: i64,
    n: i64,
    a: *mut T,
    lda: i64,
    lwork: i64,
    mut call: impl FnMut(*mut T, i64) -> i64,
) -> i64 {
    if let Err(e) = factor_geometry(layout, m, n, lda) {
        return e;
    }
    let minimum = if m == n { 1 } else { m.max(1) };
    if layout == COL_MAJOR {
        return call(a, lda);
    }
    if lwork == -1 || lwork < minimum || m == 0 || m == n {
        return call(a, m.max(1));
    }
    let count = match (m as usize).checked_mul(n as usize) {
        Some(v) => v,
        None => return -1011,
    };
    let mut temp = match zeros::<T>(count) {
        Ok(v) => v,
        Err(_) => return -1011,
    };
    // Numerical TZRZF never reads the strict lower triangle.
    for j in 0..n as usize {
        for i in 0..(m as usize).min(j + 1) {
            temp[i + j * m as usize] = *a.add(i * lda as usize + j);
        }
    }
    let info = call(temp.as_mut_ptr(), m);
    if info >= 0 {
        for j in 0..n as usize {
            for i in 0..(m as usize).min(j + 1) {
                *a.add(i * lda as usize + j) = temp[i + j * m as usize];
            }
        }
    }
    info
}
fn apply_geometry(
    layout: i32,
    side: u8,
    trans: u8,
    m: i64,
    n: i64,
    k: i64,
    l: i64,
    lda: i64,
    ldc: i64,
    lwork: i64,
    complex: bool,
) -> Result<(), i64> {
    if !matches!(layout, ROW_MAJOR | COL_MAJOR) {
        return Err(-1);
    }
    let side = side.to_ascii_uppercase();
    let trans = trans.to_ascii_uppercase();
    let order = if side == b'R' { n } else { m };
    if layout == ROW_MAJOR {
        // SIDE=R uses Q order N, independently of C's number of rows.
        if lda < order {
            return Err(-9);
        }
        if ldc < n {
            return Err(-12);
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
    if l < 0 || l > order {
        return Err(-7);
    }
    if layout == COL_MAJOR {
        if lda < k.max(1) {
            return Err(-9);
        }
        if ldc < m.max(1) {
            return Err(-12);
        }
    }
    let nw = if side == b'L' { n } else { m };
    if lwork != -1 && lwork < nw.max(1) {
        return Err(-14);
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
    l: i64,
    a: *const T,
    lda: i64,
    tau: *const T,
    c: *mut T,
    ldc: i64,
    lwork: i64,
    mut call: impl FnMut(*const T, i64, *mut T, i64) -> i64,
) -> i64 {
    if let Err(e) = apply_geometry(layout, side, trans, m, n, k, l, lda, ldc, lwork, T::COMPLEX) {
        return e;
    }
    if layout == COL_MAJOR {
        return call(a, lda, c, ldc);
    }
    if lwork == -1 || m == 0 || n == 0 || k == 0 {
        return call(a, k.max(1), c, m.max(1));
    }
    if (0..k as usize).all(|i| *tau.add(i) == T::z()) {
        return call(a, k.max(1), c, m.max(1));
    }
    let order = if side.to_ascii_uppercase() == b'L' {
        m
    } else {
        n
    };
    let mut at = Vec::<T>::new();
    if l > 0 {
        let count = match (k as usize).checked_mul(order as usize) {
            Some(v) => v,
            None => return -1011,
        };
        at = match zeros::<T>(count) {
            Ok(v) => v,
            Err(_) => return -1011,
        };
        for i in 0..k as usize {
            if *tau.add(i) != T::z() {
                for j in (order - l) as usize..order as usize {
                    at[i + j * k as usize] = *a.add(i * lda as usize + j);
                }
            }
        }
    }
    let mut ct = match row_to_column(m as usize, n as usize, c, ldc as usize) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let info = call(
        if l == 0 { a } else { at.as_ptr() },
        k.max(1),
        ct.as_mut_ptr(),
        m.max(1),
    );
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
    a: *const T,
    lda: i64,
    tau: *const T,
    c: *const T,
    ldc: i64,
) -> Result<(), i64> {
    if !matches!(layout, ROW_MAJOR | COL_MAJOR) {
        return Err(-1);
    }
    if LAPACKE_get_nancheck() != 0 {
        let order = if side.to_ascii_uppercase() == b'R' {
            n
        } else {
            m
        };
        if ge_nancheck(layout, k, order, a, lda)? {
            return Err(-8);
        }
        if ge_nancheck(layout, m, n, c, ldc)? {
            return Err(-11);
        }
        for i in 0..k.max(0) as usize {
            let value = *tau.add(i);
            if value != value {
                return Err(-10);
            }
        }
    }
    Ok(())
}
macro_rules! exports {
    ($p:ident,$t:ty,$j:ty,[$($suffix:tt)*],$f:tt,$apply:ident)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p tzrzf_work $($suffix)*>](layout:i32,m:$j,n:$j,a:*mut $t,lda:$j,tau:*mut $t,workspace:*mut $t,lwork:$j)->$j {
            factor_work(layout,m as i64,n as i64,a,lda as i64,lwork as i64,|a,lda|{let mut info:$j=0;crate::rz::[<$p tzrzf $f>](&m,&n,a,&(lda as $j),tau,workspace,&lwork,&mut info);if info<0{(info-1) as i64}else{info as i64}}) as $j
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p tzrzf $($suffix)*>](layout:i32,m:$j,n:$j,a:*mut $t,lda:$j,tau:*mut $t)->$j {
            if let Err(e)=super::qr::high_nan(layout,m as i64,n as i64,a,lda as i64){return e as $j;}
            let mut query=<$t>::z();let info=[<LAPACKE_ $p tzrzf_work $($suffix)*>](layout,m,n,a,lda,tau,&mut query,-1);if info!=0{return info;}
            let length=query.re();if !length.is_finite()||length<1.||length>=isize::MAX as f64||length>=<$j>::MAX as f64{return -1010;}
            let mut workspace=match zeros::<$t>(length as usize){Ok(v)=>v,Err(_)=>return -1010};
            [<LAPACKE_ $p tzrzf_work $($suffix)*>](layout,m,n,a,lda,tau,workspace.as_mut_ptr(),length as $j)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $apply _work $($suffix)*>](layout:i32,side:i8,trans:i8,m:$j,n:$j,k:$j,l:$j,a:*const $t,lda:$j,tau:*const $t,c:*mut $t,ldc:$j,workspace:*mut $t,lwork:$j)->$j {
            apply_work(layout,side as u8,trans as u8,m as i64,n as i64,k as i64,l as i64,a,lda as i64,tau,c,ldc as i64,lwork as i64,|a,lda,c,ldc|{let mut info:$j=0;crate::rz::[<$p $apply $f>](&(side as u8),&(trans as u8),&m,&n,&k,&l,a,&(lda as $j),tau,c,&(ldc as $j),workspace,&lwork,&mut info);if info<0{(info-1) as i64}else{info as i64}}) as $j
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $apply $($suffix)*>](layout:i32,side:i8,trans:i8,m:$j,n:$j,k:$j,l:$j,a:*const $t,lda:$j,tau:*const $t,c:*mut $t,ldc:$j)->$j {
            if let Err(e)=apply_nan(layout,side as u8,m as i64,n as i64,k as i64,a,lda as i64,tau,c,ldc as i64){return e as $j;}
            let mut query=<$t>::z();let info=[<LAPACKE_ $p $apply _work $($suffix)*>](layout,side,trans,m,n,k,l,a,lda,tau,c,ldc,&mut query,-1);if info!=0{return info;}
            let minimum=if (side as u8).to_ascii_uppercase()==b'L'{n.max(1)}else{m.max(1)};
            let length=query.re().max(minimum as f64);if !length.is_finite()||length<1.||length>=isize::MAX as f64||length>=<$j>::MAX as f64{return -1010;}
            let mut workspace=match zeros::<$t>(length as usize){Ok(v)=>v,Err(_)=>return -1010};
            [<LAPACKE_ $p $apply _work $($suffix)*>](layout,side,trans,m,n,k,l,a,lda,tau,c,ldc,workspace.as_mut_ptr(),length as $j)
        }
    }};
}
exports!(s, f32, i32, [], _, ormrz);
exports!(d, f64, i32, [], _, ormrz);
exports!(c, colmat::c32, i32, [], _, unmrz);
exports!(z, colmat::c64, i32, [], _, unmrz);
exports!(s, f32, i64, [64_], _64_, ormrz);
exports!(d, f64, i64, [64_], _64_, ormrz);
exports!(c, colmat::c32, i64, [64_], _64_, unmrz);
exports!(z, colmat::c64, i64, [64_], _64_, unmrz);
