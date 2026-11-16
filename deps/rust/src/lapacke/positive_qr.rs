//! GEQRFP uses real nonnegative R diagonals and its own empty-work contract.
use super::{
    qr::{geometry, high_nan, work},
    self_adjoint::zeros,
};
use crate::kernels::Scalar;
macro_rules! exports {
    ($p:ident,$t:ty,$j:ty,[$($suffix:tt)*],$f:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p geqrfp_work $($suffix)*>](layout:i32,m:$j,n:$j,a:*mut $t,lda:$j,tau:*mut $t,workspace:*mut $t,lwork:$j)->$j {
            if let Err(e)=geometry(layout,m as i64,n as i64,lda as i64,None){return e as $j;}
            let minimum=if m==0||n==0{1}else{n};
            if lwork!= -1&&lwork<minimum{return -8;}
            work(layout,m as i64,n as i64,a,lda as i64,if lwork== -1{Some(-1)}else{None},|a,lda|{
                let mut info:$j=0;crate::qr::[<$p geqrfp $f>](&m,&n,a,&(lda as $j),tau,workspace,&lwork,&mut info);if info<0{(info-1) as i64}else{info as i64}
            }) as $j
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p geqrfp $($suffix)*>](layout:i32,m:$j,n:$j,a:*mut $t,lda:$j,tau:*mut $t)->$j {
            if let Err(e)=high_nan(layout,m as i64,n as i64,a,lda as i64){return e as $j;}
            let mut query=<$t>::z();let info=[<LAPACKE_ $p geqrfp_work $($suffix)*>](layout,m,n,a,lda,tau,&mut query,-1);if info!=0{return info;}
            let length=query.re();if !length.is_finite()||length<1.||length>=isize::MAX as f64||length > <$j>::MAX as f64{return -1010;}
            let mut workspace=match zeros::<$t>(length as usize){Ok(v)=>v,Err(_)=>return -1010};
            [<LAPACKE_ $p geqrfp_work $($suffix)*>](layout,m,n,a,lda,tau,workspace.as_mut_ptr(),length as $j)
        }
    }};
}
exports!(s, f32, i32, [], _);
exports!(d, f64, i32, [], _);
exports!(c, colmat::c32, i32, [], _);
exports!(z, colmat::c64, i32, [], _);
exports!(s, f32, i64, [64_], _64_);
exports!(d, f64, i64, [64_], _64_);
exports!(c, colmat::c32, i64, [64_], _64_);
exports!(z, colmat::c64, i64, [64_], _64_);
