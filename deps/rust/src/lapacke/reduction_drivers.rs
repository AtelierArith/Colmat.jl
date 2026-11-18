//! LAPACKE GEHRD, GEBRD and LARFX high/work adapters.
//! Reference: Reference LAPACK v3.12.1
//! LAPACKE/src/lapacke_{s,d,c,z}{gehrd,gebrd,larfx}{,_work}.c.
use super::{ge_nancheck, row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;

unsafe fn nan_scalar<T: Scalar>(v: T) -> bool {
    v != v
}
unsafe fn nan_vec<T: Scalar>(n: i64, a: *const T) -> bool {
    (0..n.max(0) as usize).any(|i| {
        let v = *a.add(i);
        v != v
    })
}

macro_rules! gehrd {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p gehrd $($suffix)*>](
            layout:i32,n:$i,ilo:$i,ihi:$i,a:*mut $t,lda:$i,tau:*mut $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0
                && ge_nancheck(layout,n as i64,n as i64,a,lda as i64).unwrap_or(false)
            {
                return -5;
            }
            if layout==ROW_MAJOR && lda<n {return -6;}
            let mut query=<$t>::z();
            let mut info:$i=0;
            crate::hessenberg::[<$p gehrd $fortran>](&n,&ilo,&ihi,a,&lda,tau,&mut query,&mut (-1 as $i),&mut info);
            if info!=0 {return info;}
            let lw=query.re().max(1.0) as usize;
            let mut work=vec![<$t>::z();lw];
            [<LAPACKE_ $p gehrd_work $($suffix)*>](layout,n,ilo,ihi,a,lda,tau,work.as_mut_ptr(),lw as $i)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p gehrd_work $($suffix)*>](
            layout:i32,n:$i,ilo:$i,ihi:$i,a:*mut $t,lda:$i,tau:*mut $t,work:*mut $t,lwork:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -6;}
            if layout==COL_MAJOR || n==0 {
                let mut info:$i=0;
                crate::hessenberg::[<$p gehrd $fortran>](&n,&ilo,&ihi,a,&lda,tau,work,&lwork,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let mut at=match row_to_column(nn,nn,a,lda as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let ld=nn.max(1) as $i;
            let mut info:$i=0;
            crate::hessenberg::[<$p gehrd $fortran>](&n,&ilo,&ihi,at.as_mut_ptr(),&ld,tau,work,&lwork,&mut info);
            if info>=0 {for i in 0..nn {for j in 0..nn {*a.add(i*lda as usize+j)=at[i+j*nn];}}}
            if info<0 {info-1} else {info}
        }
    }};
}

macro_rules! gebrd {
    ($p:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p gebrd $($suffix)*>](
            layout:i32,m:$i,n:$i,a:*mut $t,lda:$i,d:*mut $r,e:*mut $r,
            tauq:*mut $t,taup:*mut $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0
                && ge_nancheck(layout,m as i64,n as i64,a,lda as i64).unwrap_or(false)
            {
                return -4;
            }
            if layout==ROW_MAJOR && lda<n {return -5;}
            let mut query=<$t>::z();
            let mut info:$i=0;
            crate::bidiagonal::[<$p gebrd $fortran>](&m,&n,a,&lda,d,e,tauq,taup,&mut query,&mut (-1 as $i),&mut info);
            if info!=0 {return info;}
            let lw=query.re().max(1.0) as usize;
            let mut work=vec![<$t>::z();lw];
            [<LAPACKE_ $p gebrd_work $($suffix)*>](layout,m,n,a,lda,d,e,tauq,taup,work.as_mut_ptr(),lw as $i)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p gebrd_work $($suffix)*>](
            layout:i32,m:$i,n:$i,a:*mut $t,lda:$i,d:*mut $r,e:*mut $r,
            tauq:*mut $t,taup:*mut $t,work:*mut $t,lwork:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -5;}
            if layout==COL_MAJOR || m==0 || n==0 {
                let mut info:$i=0;
                crate::bidiagonal::[<$p gebrd $fortran>](&m,&n,a,&lda,d,e,tauq,taup,work,&lwork,&mut info);
                return if info<0 {info-1} else {info};
            }
            let (mm,nn)=(m as usize,n as usize);
            let mut at=match row_to_column(mm,nn,a,lda as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let ld=mm.max(1) as $i;
            let mut info:$i=0;
            crate::bidiagonal::[<$p gebrd $fortran>](&m,&n,at.as_mut_ptr(),&ld,d,e,tauq,taup,work,&lwork,&mut info);
            if info>=0 {for i in 0..mm {for j in 0..nn {*a.add(i*lda as usize+j)=at[i+j*mm];}}}
            if info<0 {info-1} else {info}
        }
    }};
}

macro_rules! larfx {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p larfx $($suffix)*>](
            layout:i32,side:u8,m:$i,n:$i,v:*const $t,tau:$t,c:*mut $t,ldc:$i,work:*mut $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if ge_nancheck(layout,m as i64,n as i64,c,ldc as i64).unwrap_or(false) {return -7;}
                if nan_scalar(tau) {return -6;}
                let lv=if side.to_ascii_uppercase()==b'L' {m} else {n};
                if nan_vec(lv as i64,v) {return -5;}
            }
            [<LAPACKE_ $p larfx_work $($suffix)*>](layout,side,m,n,v,tau,c,ldc,work)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p larfx_work $($suffix)*>](
            layout:i32,side:u8,m:$i,n:$i,v:*const $t,tau:$t,c:*mut $t,ldc:$i,work:*mut $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && ldc<n {return -8;}
            if layout==COL_MAJOR || m==0 || n==0 {
                crate::householder::[<$p larfx $fortran>](&side,&m,&n,v,&tau,c,&ldc,work);
                return 0;
            }
            let (mm,nn)=(m as usize,n as usize);
            let mut ct=match row_to_column(mm,nn,c,ldc as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let ld=mm.max(1) as $i;
            crate::householder::[<$p larfx $fortran>](&side,&m,&n,v,&tau,ct.as_mut_ptr(),&ld,work);
            for i in 0..mm {for j in 0..nn {*c.add(i*ldc as usize+j)=ct[i+j*mm];}}
            0
        }
    }};
}
gehrd!(s, f32, i32, [], _);
gehrd!(d, f64, i32, [], _);
gehrd!(c, colmat::c32, i32, [], _);
gehrd!(z, colmat::c64, i32, [], _);
gehrd!(s, f32, i64, [64_], _64_);
gehrd!(d, f64, i64, [64_], _64_);
gehrd!(c, colmat::c32, i64, [64_], _64_);
gehrd!(z, colmat::c64, i64, [64_], _64_);
gebrd!(s, f32, f32, i32, [], _);
gebrd!(d, f64, f64, i32, [], _);
gebrd!(c, colmat::c32, f32, i32, [], _);
gebrd!(z, colmat::c64, f64, i32, [], _);
gebrd!(s, f32, f32, i64, [64_], _64_);
gebrd!(d, f64, f64, i64, [64_], _64_);
gebrd!(c, colmat::c32, f32, i64, [64_], _64_);
gebrd!(z, colmat::c64, f64, i64, [64_], _64_);
larfx!(s, f32, i32, [], _);
larfx!(d, f64, i32, [], _);
larfx!(c, colmat::c32, i32, [], _);
larfx!(z, colmat::c64, i32, [], _);
larfx!(s, f32, i64, [64_], _64_);
larfx!(d, f64, i64, [64_], _64_);
larfx!(c, colmat::c32, i64, [64_], _64_);
larfx!(z, colmat::c64, i64, [64_], _64_);
