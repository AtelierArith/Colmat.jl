//! LAPACKE GESVDQ high/work adapters for the QR-preconditioned SVD.
//! Reference: Reference LAPACK v3.12.1 LAPACKE/src/lapacke_{s,d,c,z}gesvdq{,_work}.c.
//! Row-major A is transposed and written back (it holds the QR factors); U
//! (M-by-M for JOBU='A', N-by-N for 'F', M-by-N otherwise, NUMRANK columns for
//! 'R') and V (NUMRANK or N rows by N) are written back at their logical
//! sizes. The high-level entry queries IWORK/WORK/RWORK.
use super::{ge_nancheck, row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;
use crate::lapack::LapackInt;

macro_rules! gesvdq_exports {
    ($p:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fort:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p gesvdq $($suffix)*>](
            layout:i32,joba:u8,jobp:u8,jobr:u8,jobu:u8,jobv:u8,m:$i,n:$i,a:*mut $t,lda:$i,
            s:*mut $r,u:*mut $t,ldu:$i,v:*mut $t,ldv:$i,numrank:*mut $i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 && ge_nancheck(layout,m as i64,n as i64,a,lda as i64).unwrap_or(false) {
                return -9;
            }
            let (mut qi,mut qw,mut qr)=(0 as $i,[<$t>::z();2],<$r>::z());
            let r=[<LAPACKE_ $p gesvdq_work $($suffix)*>](layout,joba,jobp,jobr,jobu,jobv,m,n,a,lda,s,u,ldu,v,ldv,numrank,
                &mut qi,-1,qw.as_mut_ptr(),-1,&mut qr,-1);
            if r!=0 {return r;}
            let li=(qi as usize).max(1);
            let lw=(qw[0].re().max(2.0)) as usize;
            let lr=(qr as f64).max(1.0) as usize;
            let mut iwork=vec![0 as $i;li];
            let mut work=vec![<$t>::z();lw.max(2)];
            let mut rwork=vec![<$r>::z();lr.max(2)];
            [<LAPACKE_ $p gesvdq_work $($suffix)*>](layout,joba,jobp,jobr,jobu,jobv,m,n,a,lda,s,u,ldu,v,ldv,numrank,
                iwork.as_mut_ptr(),li as $i,work.as_mut_ptr(),lw as $i,rwork.as_mut_ptr(),lr as $i)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p gesvdq_work $($suffix)*>](
            layout:i32,joba:u8,jobp:u8,jobr:u8,jobu:u8,jobv:u8,m:$i,n:$i,a:*mut $t,lda:$i,
            s:*mut $r,u:*mut $t,ldu:$i,v:*mut $t,ldv:$i,numrank:*mut $i,
            iwork:*mut $i,liwork:$i,work:*mut $t,lwork:$i,rwork:*mut $r,lrwork:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            let (ju,jv)=(jobu.to_ascii_uppercase(),jobv.to_ascii_uppercase());
            let ucols:i64=match ju {b'N'=>1,b'A'=>m as i64,_=>n as i64};
            let vcols:i64=if jv==b'N' {1} else {n as i64};
            if layout==ROW_MAJOR && lda<n {return -10;}
            if layout==ROW_MAJOR && (ldu as i64)<ucols {return -13;}
            if layout==ROW_MAJOR && (ldv as i64)<vcols {return -15;}
            let query=liwork== -1||lwork== -1||lrwork== -1;
            if layout==COL_MAJOR || m<=0 || n<=0 || query {
                let mut info:$i=0;
                let (la,lu,lv)=if layout==ROW_MAJOR {(m.max(1),m.max(n).max(1),n.max(1))} else {(lda,ldu,ldv)};
                crate::advanced_svd::[<$p gesvdq $fort>](&joba,&jobp,&jobr,&jobu,&jobv,&m,&n,a,&la,s,u,&lu,v,&lv,numrank,iwork,&liwork,work,&lwork,rwork,&lrwork,&mut info);
                return if info<0 {info-1} else {info};
            }
            let (mm,nn)=(m as usize,n as usize);
            let mut at=match row_to_column(mm,nn,a,lda as usize){Ok(x)=>x,Err(e)=>return e as $i};
            let side=mm.max(nn);
            let mut ut=vec![<$t>::z();side*side];
            let mut vt=vec![<$t>::z();nn*nn];
            let (la,lu,lv)=(mm.max(1) as $i,side as $i,nn.max(1) as $i);
            let mut info:$i=0;
            crate::advanced_svd::[<$p gesvdq $fort>](&joba,&jobp,&jobr,&jobu,&jobv,&m,&n,at.as_mut_ptr(),&la,s,ut.as_mut_ptr(),&lu,vt.as_mut_ptr(),&lv,numrank,iwork,&liwork,work,&lwork,rwork,&lrwork,&mut info);
            if info>=0 {
                let rank=(*numrank).val().clamp(0,nn as isize) as usize;
                for i in 0..mm {for j in 0..nn {*a.add(i*lda as usize+j)=at[i+j*mm];}}
                let urows=if ju==b'F' {nn} else {mm};
                let ucount=match ju {b'N'=>0,b'R'=>rank,b'A'=>mm,_=>nn};
                for i in 0..urows {for j in 0..ucount {*u.add(i*ldu as usize+j)=ut[i+j*side];}}
                let vrows=match jv {b'N'=>0,b'R'=>rank,_=>nn};
                for i in 0..vrows {for j in 0..nn {*v.add(i*ldv as usize+j)=vt[i+j*nn];}}
            }
            if info<0 {info-1} else {info}
        }
    }};
}
macro_rules! all_abi {
    ($i:ty,[$($suffix:tt)*],$fort:tt)=>{
        gesvdq_exports!(s,f32,f32,$i,[$($suffix)*],$fort);
        gesvdq_exports!(d,f64,f64,$i,[$($suffix)*],$fort);
        gesvdq_exports!(c,colmat::c32,f32,$i,[$($suffix)*],$fort);
        gesvdq_exports!(z,colmat::c64,f64,$i,[$($suffix)*],$fort);
    };
}
all_abi!(i32, [], _);
all_abi!(i64, [64_], _64_);
