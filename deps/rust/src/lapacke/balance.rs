//! LAPACKE GEBAL/GEBAK high/work adapters for balancing and back-transform.
//! Reference: Reference LAPACK v3.12.1 LAPACKE/src/lapacke_{s,d,c,z}{gebal,gebak}{,_work}.c.
use super::{ge_nancheck, row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;

unsafe fn nan_vec<T: Scalar>(n: i64, a: *const T) -> bool {
    (0..n.max(0) as usize).any(|i| {
        let v = *a.add(i);
        v != v
    })
}

macro_rules! exports {
    ($p:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p gebal $($suffix)*>](
            layout:i32,job:u8,n:$i,a:*mut $t,lda:$i,ilo:*mut $i,ihi:*mut $i,
            scale:*mut $r,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            let j=job.to_ascii_uppercase();
            if LAPACKE_get_nancheck()!=0
                && matches!(j,b'B'|b'P'|b'S')
                && ge_nancheck(layout,n as i64,n as i64,a,lda as i64).unwrap_or(false)
            {
                return -4;
            }
            [<LAPACKE_ $p gebal_work $($suffix)*>](layout,job,n,a,lda,ilo,ihi,scale)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p gebal_work $($suffix)*>](
            layout:i32,job:u8,n:$i,a:*mut $t,lda:$i,ilo:*mut $i,ihi:*mut $i,
            scale:*mut $r,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -5;}
            if layout==COL_MAJOR || n==0 {
                let mut info:$i=0;
                crate::balance::[<$p gebal $fortran>](&job,&n,a,&lda,ilo,ihi,scale,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let mut at=match row_to_column(nn,nn,a,lda as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let ld=nn.max(1) as $i;
            let mut info:$i=0;
            crate::balance::[<$p gebal $fortran>](&job,&n,at.as_mut_ptr(),&ld,ilo,ihi,scale,&mut info);
            if info>=0 {for i in 0..nn {for j in 0..nn {*a.add(i*lda as usize+j)=at[i+j*nn];}}}
            if info<0 {info-1} else {info}
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p gebak $($suffix)*>](
            layout:i32,job:u8,side:u8,n:$i,ilo:$i,ihi:$i,scale:*const $r,
            m:$i,v:*mut $t,ldv:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if nan_vec(n as i64,scale) {return -7;}
                if ge_nancheck(layout,n as i64,m as i64,v,ldv as i64).unwrap_or(false) {return -9;}
            }
            [<LAPACKE_ $p gebak_work $($suffix)*>](layout,job,side,n,ilo,ihi,scale,m,v,ldv)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p gebak_work $($suffix)*>](
            layout:i32,job:u8,side:u8,n:$i,ilo:$i,ihi:$i,scale:*const $r,
            m:$i,v:*mut $t,ldv:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && ldv<m {return -10;}
            if layout==COL_MAJOR || n==0 || m==0 {
                let mut info:$i=0;
                crate::balance::[<$p gebak $fortran>](&job,&side,&n,&ilo,&ihi,scale,&m,v,&ldv,&mut info);
                return if info<0 {info-1} else {info};
            }
            let (nn,mm)=(n as usize,m as usize);
            let mut vt=match row_to_column(nn,mm,v,ldv as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let ld=nn.max(1) as $i;
            let mut info:$i=0;
            crate::balance::[<$p gebak $fortran>](&job,&side,&n,&ilo,&ihi,scale,&m,vt.as_mut_ptr(),&ld,&mut info);
            if info>=0 {for i in 0..nn {for j in 0..mm {*v.add(i*ldv as usize+j)=vt[i+j*nn];}}}
            if info<0 {info-1} else {info}
        }
    }};
}
exports!(s, f32, f32, i32, [], _);
exports!(d, f64, f64, i32, [], _);
exports!(c, colmat::c32, f32, i32, [], _);
exports!(z, colmat::c64, f64, i32, [], _);
exports!(s, f32, f32, i64, [64_], _64_);
exports!(d, f64, f64, i64, [64_], _64_);
exports!(c, colmat::c32, f32, i64, [64_], _64_);
exports!(z, colmat::c64, f64, i64, [64_], _64_);
