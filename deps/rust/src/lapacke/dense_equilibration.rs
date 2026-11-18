//! LAPACKE GEEQU/GEEQUB high/work adapters for dense equilibration.
//! Reference: Reference LAPACK v3.12.1
//! LAPACKE/src/lapacke_{s,d,c,z}{geequ,geequb}{,_work}.c.
use super::{ge_nancheck, row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;

macro_rules! exports {
    ($p:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fortran:tt,$fname:ident)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $fname $($suffix)*>](
            layout:i32,m:$i,n:$i,a:*const $t,lda:$i,
            r:*mut $r,c:*mut $r,rowcnd:*mut $r,colcnd:*mut $r,amax:*mut $r,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0
                && ge_nancheck(layout,m as i64,n as i64,a,lda as i64).unwrap_or(false)
            {
                return -4;
            }
            [<LAPACKE_ $p $fname _work $($suffix)*>](layout,m,n,a,lda,r,c,rowcnd,colcnd,amax)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $fname _work $($suffix)*>](
            layout:i32,m:$i,n:$i,a:*const $t,lda:$i,
            r:*mut $r,c:*mut $r,rowcnd:*mut $r,colcnd:*mut $r,amax:*mut $r,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -5;}
            if layout==COL_MAJOR || m==0 || n==0 {
                let mut info:$i=0;
                crate::general_equilibration::[<$p $fname $fortran>](&m,&n,a,&lda,r,c,rowcnd,colcnd,amax,&mut info);
                return info;
            }
            let (mm,nn)=(m as usize,n as usize);
            let at=match row_to_column(mm,nn,a,lda as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let ld=mm.max(1) as $i;
            let mut info:$i=0;
            crate::general_equilibration::[<$p $fname $fortran>](&m,&n,at.as_ptr(),&ld,r,c,rowcnd,colcnd,amax,&mut info);
            if info<0 {info-1} else {info}
        }
    }};
}
exports!(s, f32, f32, i32, [], _, geequ);
exports!(d, f64, f64, i32, [], _, geequ);
exports!(c, colmat::c32, f32, i32, [], _, geequ);
exports!(z, colmat::c64, f64, i32, [], _, geequ);
exports!(s, f32, f32, i64, [64_], _64_, geequ);
exports!(d, f64, f64, i64, [64_], _64_, geequ);
exports!(c, colmat::c32, f32, i64, [64_], _64_, geequ);
exports!(z, colmat::c64, f64, i64, [64_], _64_, geequ);
exports!(s, f32, f32, i32, [], _, geequb);
exports!(d, f64, f64, i32, [], _, geequb);
exports!(c, colmat::c32, f32, i32, [], _, geequb);
exports!(z, colmat::c64, f64, i32, [], _, geequb);
exports!(s, f32, f32, i64, [64_], _64_, geequb);
exports!(d, f64, f64, i64, [64_], _64_, geequb);
exports!(c, colmat::c32, f32, i64, [64_], _64_, geequb);
exports!(z, colmat::c64, f64, i64, [64_], _64_, geequb);
