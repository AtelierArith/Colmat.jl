//! Explicit orthogonal/unitary Q: QR, LQ, QL and RQ storage conventions.
use super::{
    ge_nancheck, qr::work, self_adjoint::zeros, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR,
};
use crate::kernels::Scalar;

fn geometry(layout: i32, m: i64, n: i64, k: i64, lda: i64, rows: bool) -> Result<(), i64> {
    if !matches!(layout, ROW_MAJOR | COL_MAJOR) {
        return Err(-1);
    }
    if layout == ROW_MAJOR && lda < n {
        return Err(-6);
    }
    if m < 0 {
        return Err(-2);
    }
    if n < 0 || (if rows { n < m } else { n > m }) {
        return Err(-3);
    }
    if k < 0 || k > if rows { m } else { n } {
        return Err(-4);
    }
    if layout == COL_MAJOR && lda < m.max(1) {
        return Err(-6);
    }
    Ok(())
}
unsafe fn high_nan<T: Scalar>(
    layout: i32,
    m: i64,
    n: i64,
    k: i64,
    a: *const T,
    lda: i64,
    tau: *const T,
) -> Result<(), i64> {
    if !matches!(layout, ROW_MAJOR | COL_MAJOR) {
        return Err(-1);
    }
    if LAPACKE_get_nancheck() != 0 {
        if ge_nancheck(layout, m, n, a, lda)? {
            return Err(-5);
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
                    return Err(-7);
                }
            }
        }
    }
    Ok(())
}
macro_rules! exports {
    ($p:ident,$family:ident,$module:ident,$rows:expr,$t:ty,$j:ty,[$($suffix:tt)*],$f:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $family _work $($suffix)*>](layout:i32,m:$j,n:$j,k:$j,a:*mut $t,lda:$j,tau:*const $t,workspace:*mut $t,lwork:$j)->$j {
            if let Err(e)=geometry(layout,m as i64,n as i64,k as i64,lda as i64,$rows){return e as $j;}
            work(layout,m as i64,n as i64,a,lda as i64,if lwork== -1{Some(-1)}else{None},|a,lda|{
                let mut info:$j=0;crate::$module::[<$p $family $f>](&m,&n,&k,a,&(lda as $j),tau,workspace,&lwork,&mut info);
                if info<0{(info-1) as i64}else{info as i64}
            }) as $j
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $family $($suffix)*>](layout:i32,m:$j,n:$j,k:$j,a:*mut $t,lda:$j,tau:*const $t)->$j {
            if let Err(e)=high_nan(layout,m as i64,n as i64,k as i64,a,lda as i64,tau){return e as $j;}
            let mut query=<$t>::z();let info=[<LAPACKE_ $p $family _work $($suffix)*>](layout,m,n,k,a,lda,tau,&mut query,-1);if info!=0{return info;}
            let length=query.re();if !length.is_finite()||length<1.||length>=isize::MAX as f64||length > <$j>::MAX as f64{return -1010;}
            let mut workspace=match zeros::<$t>(length as usize){Ok(v)=>v,Err(_)=>return -1010};
            [<LAPACKE_ $p $family _work $($suffix)*>](layout,m,n,k,a,lda,tau,workspace.as_mut_ptr(),length as $j)
        }
    }};
}
exports!(s, orgqr, qr, false, f32, i32, [], _);
exports!(d, orgqr, qr, false, f64, i32, [], _);
exports!(c, ungqr, qr, false, colmat::c32, i32, [], _);
exports!(z, ungqr, qr, false, colmat::c64, i32, [], _);
exports!(s, orgqr, qr, false, f32, i64, [64_], _64_);
exports!(d, orgqr, qr, false, f64, i64, [64_], _64_);
exports!(c, ungqr, qr, false, colmat::c32, i64, [64_], _64_);
exports!(z, ungqr, qr, false, colmat::c64, i64, [64_], _64_);
exports!(s, orglq, lq, true, f32, i32, [], _);
exports!(d, orglq, lq, true, f64, i32, [], _);
exports!(c, unglq, lq, true, colmat::c32, i32, [], _);
exports!(z, unglq, lq, true, colmat::c64, i32, [], _);
exports!(s, orglq, lq, true, f32, i64, [64_], _64_);
exports!(d, orglq, lq, true, f64, i64, [64_], _64_);
exports!(c, unglq, lq, true, colmat::c32, i64, [64_], _64_);
exports!(z, unglq, lq, true, colmat::c64, i64, [64_], _64_);
exports!(s, orgql, reverse_qr, false, f32, i32, [], _);
exports!(d, orgql, reverse_qr, false, f64, i32, [], _);
exports!(c, ungql, reverse_qr, false, colmat::c32, i32, [], _);
exports!(z, ungql, reverse_qr, false, colmat::c64, i32, [], _);
exports!(s, orgql, reverse_qr, false, f32, i64, [64_], _64_);
exports!(d, orgql, reverse_qr, false, f64, i64, [64_], _64_);
exports!(c, ungql, reverse_qr, false, colmat::c32, i64, [64_], _64_);
exports!(z, ungql, reverse_qr, false, colmat::c64, i64, [64_], _64_);
exports!(s, orgrq, reverse_qr, true, f32, i32, [], _);
exports!(d, orgrq, reverse_qr, true, f64, i32, [], _);
exports!(c, ungrq, reverse_qr, true, colmat::c32, i32, [], _);
exports!(z, ungrq, reverse_qr, true, colmat::c64, i32, [], _);
exports!(s, orgrq, reverse_qr, true, f32, i64, [64_], _64_);
exports!(d, orgrq, reverse_qr, true, f64, i64, [64_], _64_);
exports!(c, ungrq, reverse_qr, true, colmat::c32, i64, [64_], _64_);
exports!(z, ungrq, reverse_qr, true, colmat::c64, i64, [64_], _64_);
