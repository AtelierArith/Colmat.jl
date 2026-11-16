//! Column pivot flags use the caller integer ABI; complex work has real RWORK.
use super::{
    qr::{geometry, high_nan, work},
    self_adjoint::zeros,
};
use crate::kernels::Scalar;
fn minimum(m: i64, n: i64, complex: bool) -> Result<i64, i64> {
    if m.min(n) == 0 {
        return Ok(1);
    }
    n.checked_mul(if complex { 1 } else { 3 })
        .and_then(|v| v.checked_add(1))
        .ok_or(-1010)
}
fn count(n: i64, factor: usize) -> Result<usize, i64> {
    (n as usize)
        .checked_mul(factor)
        .map(|v| v.max(1))
        .ok_or(-1010)
}
macro_rules! exports {
    ($p:ident,$t:ty,$j:ty,[$($suffix:tt)*],$f:tt $(,$r:ty)?)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p geqp3_work $($suffix)*>](layout:i32,m:$j,n:$j,a:*mut $t,lda:$j,jpvt:*mut $j,tau:*mut $t,workspace:*mut $t,lwork:$j $(,rwork:*mut $r)?)->$j {
            if let Err(e)=geometry(layout,m as i64,n as i64,lda as i64,None){return e as $j;}
            let needed=match minimum(m as i64,n as i64,<$t>::COMPLEX){Ok(v)=>v,Err(e)=>return e as $j};
            // Query and insufficient work must publish WORK without reading A.
            let mode=if lwork== -1||(lwork as i64)<needed{Some(-1)}else{None};
            work(layout,m as i64,n as i64,a,lda as i64,mode,|a,lda|{
                let mut info:$j=0;crate::qr_pivoted::[<$p geqp3 $f>](&m,&n,a,&(lda as $j),jpvt,tau,workspace,&lwork,$(rwork as *mut $r,)?&mut info);if info<0{(info-1) as i64}else{info as i64}
            }) as $j
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p geqp3 $($suffix)*>](layout:i32,m:$j,n:$j,a:*mut $t,lda:$j,jpvt:*mut $j,tau:*mut $t)->$j {
            if let Err(e)=high_nan(layout,m as i64,n as i64,a,lda as i64){return e as $j;}
            if let Err(e)=geometry(layout,m as i64,n as i64,lda as i64,None){return e as $j;}
            $(let real_count=match count(n as i64,2){Ok(v)=>v,Err(e)=>return e as $j};let mut rwork=match zeros::<$r>(real_count){Ok(v)=>v,Err(_)=>return -1010};)?
            let mut query=<$t>::z();let info=[<LAPACKE_ $p geqp3_work $($suffix)*>](layout,m,n,a,lda,jpvt,tau,&mut query,-1 $(,rwork.as_mut_ptr() as *mut $r)?);if info!=0{return info;}
            let length=query.re();if !length.is_finite()||length<1.||length>=isize::MAX as f64||length>=<$j>::MAX as f64{return -1010;}
            let mut workspace=match zeros::<$t>(length as usize){Ok(v)=>v,Err(_)=>return -1010};
            [<LAPACKE_ $p geqp3_work $($suffix)*>](layout,m,n,a,lda,jpvt,tau,workspace.as_mut_ptr(),length as $j $(,rwork.as_mut_ptr() as *mut $r)?)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p geqpf_work $($suffix)*>](layout:i32,m:$j,n:$j,a:*mut $t,lda:$j,jpvt:*mut $j,tau:*mut $t,workspace:*mut $t $(,rwork:*mut $r)?)->$j {
            work(layout,m as i64,n as i64,a,lda as i64,None,|a,lda|{
                let mut info:$j=0;crate::qr_pivoted::[<$p geqpf $f>](&m,&n,a,&(lda as $j),jpvt,tau,workspace,$(rwork as *mut $r,)?&mut info);if info<0{(info-1) as i64}else{info as i64}
            }) as $j
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p geqpf $($suffix)*>](layout:i32,m:$j,n:$j,a:*mut $t,lda:$j,jpvt:*mut $j,tau:*mut $t)->$j {
            if let Err(e)=high_nan(layout,m as i64,n as i64,a,lda as i64){return e as $j;}
            if let Err(e)=geometry(layout,m as i64,n as i64,lda as i64,None){return e as $j;}
            let length=match count(n as i64,if <$t>::COMPLEX{1}else{3}){Ok(v)=>v,Err(e)=>return e as $j};
            let mut workspace=match zeros::<$t>(length){Ok(v)=>v,Err(_)=>return -1010};
            $(let real_count=match count(n as i64,2){Ok(v)=>v,Err(e)=>return e as $j};let mut rwork=match zeros::<$r>(real_count){Ok(v)=>v,Err(_)=>return -1010};)?
            [<LAPACKE_ $p geqpf_work $($suffix)*>](layout,m,n,a,lda,jpvt,tau,workspace.as_mut_ptr() $(,rwork.as_mut_ptr() as *mut $r)?)
        }
    }};
}
exports!(s, f32, i32, [], _);
exports!(d, f64, i32, [], _);
exports!(c, colmat::c32, i32, [], _, f32);
exports!(z, colmat::c64, i32, [], _, f64);
exports!(s, f32, i64, [64_], _64_);
exports!(d, f64, i64, [64_], _64_);
exports!(c, colmat::c32, i64, [64_], _64_, f32);
exports!(z, colmat::c64, i64, [64_], _64_, f64);
