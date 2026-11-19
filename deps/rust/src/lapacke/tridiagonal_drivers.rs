//! LAPACKE STERF, STEQR, STEV, STEVD and STEDC high/work adapters.
//! Reference: Reference LAPACK v3.12.1
//! LAPACKE/src/lapacke_{s,d,c,z}{steqr,stedc}{,_work}.c,
//! lapacke_{s,d}{sterf,stev,stevd}{,_work}.c. STERF has no matrix_layout.
use super::{row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;

unsafe fn nan_vec<T: Scalar>(n: i64, a: *const T) -> bool {
    (0..n.max(0) as usize).any(|i| {
        let v = *a.add(i);
        v != v
    })
}
unsafe fn nan_z<T: Scalar>(layout: i32, n: i64, z: *const T, ldz: i64) -> bool {
    super::ge_nancheck(layout, n, n, z, ldz).unwrap_or(false)
}
/// Column-major working copy of the row-major eigenvector matrix; the input
/// values are read only when the caller supplies a starting Z (COMPZ='V').
unsafe fn z_in<T: Scalar>(read: bool, n: usize, z: *const T, ldz: usize) -> Result<Vec<T>, i64> {
    if read {
        row_to_column(n, n, z, ldz)
    } else {
        Ok(vec![T::z(); n * n])
    }
}
unsafe fn z_out<T: Scalar>(n: usize, z: *mut T, ldz: usize, zt: &[T]) {
    for i in 0..n {
        for j in 0..n {
            *z.add(i * ldz + j) = zt[i + j * n];
        }
    }
}

macro_rules! steqr_exports {
    ($p:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p steqr $($suffix)*>](
            layout:i32,compz:u8,n:$i,d:*mut $r,e:*mut $r,z:*mut $t,ldz:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if nan_vec(n as i64,d) {return -4;}
                if nan_vec((n-1).max(0) as i64,e) {return -5;}
                if compz.to_ascii_uppercase()==b'V' && nan_z(layout,n as i64,z,ldz as i64) {return -6;}
            }
            let lw=if compz.to_ascii_uppercase()==b'N' {1} else {(2*n.max(0) as usize).saturating_sub(2).max(1)};
            let mut work=vec![<$r>::z();lw];
            [<LAPACKE_ $p steqr_work $($suffix)*>](layout,compz,n,d,e,z,ldz,work.as_mut_ptr())
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p steqr_work $($suffix)*>](
            layout:i32,compz:u8,n:$i,d:*mut $r,e:*mut $r,z:*mut $t,ldz:$i,work:*mut $r,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            let cz=compz.to_ascii_uppercase();
            if layout==ROW_MAJOR && ldz<n {return -7;}
            if layout==COL_MAJOR || n==0 {
                let mut info:$i=0;
                crate::tridiagonal_evd::[<$p steqr $fortran>](&compz,&n,d,e,z,&ldz,work,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let mut zt=match z_in(cz==b'V',nn,z,ldz as usize){Ok(v)=>v,Err(err)=>return err as $i};
            let ld=nn.max(1) as $i;
            let mut info:$i=0;
            crate::tridiagonal_evd::[<$p steqr $fortran>](&compz,&n,d,e,zt.as_mut_ptr(),&ld,work,&mut info);
            if info>=0 && cz!=b'N' {z_out(nn,z,ldz as usize,&zt);}
            if info<0 {info-1} else {info}
        }
    }};
}
macro_rules! sterf_exports {
    ($p:ident,$r:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p sterf $($suffix)*>](n:$i,d:*mut $r,e:*mut $r)->$i {
            if LAPACKE_get_nancheck()!=0 {
                if nan_vec(n as i64,d) {return -2;}
                if nan_vec((n-1).max(0) as i64,e) {return -3;}
            }
            [<LAPACKE_ $p sterf_work $($suffix)*>](n,d,e)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p sterf_work $($suffix)*>](n:$i,d:*mut $r,e:*mut $r)->$i {
            let mut info:$i=0;
            crate::tridiagonal_evd::[<$p sterf $fortran>](&n,d,e,&mut info);
            info
        }
    }};
}
macro_rules! stev_exports {
    ($p:ident,$r:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p stev $($suffix)*>](
            layout:i32,jobz:u8,n:$i,d:*mut $r,e:*mut $r,z:*mut $r,ldz:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if nan_vec(n as i64,d) {return -4;}
                if nan_vec((n-1).max(0) as i64,e) {return -5;}
            }
            let lw=(2*n.max(0) as usize).saturating_sub(2).max(1);
            let mut work=vec![<$r>::z();lw];
            [<LAPACKE_ $p stev_work $($suffix)*>](layout,jobz,n,d,e,z,ldz,work.as_mut_ptr())
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p stev_work $($suffix)*>](
            layout:i32,jobz:u8,n:$i,d:*mut $r,e:*mut $r,z:*mut $r,ldz:$i,work:*mut $r,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && ldz<n {return -7;}
            if layout==COL_MAJOR || n==0 {
                let mut info:$i=0;
                crate::tridiagonal_evd::[<$p stev $fortran>](&jobz,&n,d,e,z,&ldz,work,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let mut zt=vec![<$r>::z();nn*nn];
            let ld=nn.max(1) as $i;
            let mut info:$i=0;
            crate::tridiagonal_evd::[<$p stev $fortran>](&jobz,&n,d,e,zt.as_mut_ptr(),&ld,work,&mut info);
            if info>=0 && jobz.to_ascii_uppercase()==b'V' {z_out(nn,z,ldz as usize,&zt);}
            if info<0 {info-1} else {info}
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p stevd $($suffix)*>](
            layout:i32,jobz:u8,n:$i,d:*mut $r,e:*mut $r,z:*mut $r,ldz:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if nan_vec(n as i64,d) {return -4;}
                if nan_vec((n-1).max(0) as i64,e) {return -5;}
            }
            let (mut qw,mut qi)=(<$r>::z(),0 as $i);
            let r=[<LAPACKE_ $p stevd_work $($suffix)*>](layout,jobz,n,d,e,z,ldz,&mut qw,-1,&mut qi,-1);
            if r!=0 {return r;}
            let (lw,li)=(qw.re().max(1.0) as usize,(qi as usize).max(1));
            let mut work=vec![<$r>::z();lw];
            let mut iwork=vec![0 as $i;li];
            [<LAPACKE_ $p stevd_work $($suffix)*>](layout,jobz,n,d,e,z,ldz,work.as_mut_ptr(),lw as $i,iwork.as_mut_ptr(),li as $i)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p stevd_work $($suffix)*>](
            layout:i32,jobz:u8,n:$i,d:*mut $r,e:*mut $r,z:*mut $r,ldz:$i,
            work:*mut $r,lwork:$i,iwork:*mut $i,liwork:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && ldz<n {return -7;}
            if layout==COL_MAJOR || n==0 || lwork==-1 || liwork==-1 {
                let mut info:$i=0;
                let ld=if layout==ROW_MAJOR {n.max(1)} else {ldz};
                crate::tridiagonal_evd::[<$p stevd $fortran>](&jobz,&n,d,e,z,&ld,work,&lwork,iwork,&liwork,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let mut zt=vec![<$r>::z();nn*nn];
            let ld=nn.max(1) as $i;
            let mut info:$i=0;
            crate::tridiagonal_evd::[<$p stevd $fortran>](&jobz,&n,d,e,zt.as_mut_ptr(),&ld,work,&lwork,iwork,&liwork,&mut info);
            if info>=0 && jobz.to_ascii_uppercase()==b'V' {z_out(nn,z,ldz as usize,&zt);}
            if info<0 {info-1} else {info}
        }
    }};
}
macro_rules! stedc_real_exports {
    ($p:ident,$r:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p stedc $($suffix)*>](
            layout:i32,compz:u8,n:$i,d:*mut $r,e:*mut $r,z:*mut $r,ldz:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if nan_vec(n as i64,d) {return -4;}
                if nan_vec((n-1).max(0) as i64,e) {return -5;}
                if compz.to_ascii_uppercase()==b'V' && nan_z(layout,n as i64,z,ldz as i64) {return -6;}
            }
            let (mut qw,mut qi)=(<$r>::z(),0 as $i);
            let r=[<LAPACKE_ $p stedc_work $($suffix)*>](layout,compz,n,d,e,z,ldz,&mut qw,-1,&mut qi,-1);
            if r!=0 {return r;}
            let (lw,li)=(qw.re().max(1.0) as usize,(qi as usize).max(1));
            let mut work=vec![<$r>::z();lw];
            let mut iwork=vec![0 as $i;li];
            [<LAPACKE_ $p stedc_work $($suffix)*>](layout,compz,n,d,e,z,ldz,work.as_mut_ptr(),lw as $i,iwork.as_mut_ptr(),li as $i)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p stedc_work $($suffix)*>](
            layout:i32,compz:u8,n:$i,d:*mut $r,e:*mut $r,z:*mut $r,ldz:$i,
            work:*mut $r,lwork:$i,iwork:*mut $i,liwork:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            let cz=compz.to_ascii_uppercase();
            if layout==ROW_MAJOR && ldz<n {return -7;}
            if layout==COL_MAJOR || n==0 || lwork==-1 || liwork==-1 {
                let mut info:$i=0;
                let ld=if layout==ROW_MAJOR {n.max(1)} else {ldz};
                crate::tridiagonal_evd::[<$p stedc $fortran>](&compz,&n,d,e,z,&ld,work,&lwork,iwork,&liwork,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let mut zt=match z_in(cz==b'V',nn,z,ldz as usize){Ok(v)=>v,Err(err)=>return err as $i};
            let ld=nn.max(1) as $i;
            let mut info:$i=0;
            crate::tridiagonal_evd::[<$p stedc $fortran>](&compz,&n,d,e,zt.as_mut_ptr(),&ld,work,&lwork,iwork,&liwork,&mut info);
            if info>=0 && cz!=b'N' {z_out(nn,z,ldz as usize,&zt);}
            if info<0 {info-1} else {info}
        }
    }};
}
macro_rules! stedc_complex_exports {
    ($p:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p stedc $($suffix)*>](
            layout:i32,compz:u8,n:$i,d:*mut $r,e:*mut $r,z:*mut $t,ldz:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if nan_vec(n as i64,d) {return -4;}
                if nan_vec((n-1).max(0) as i64,e) {return -5;}
                if compz.to_ascii_uppercase()==b'V' && nan_z(layout,n as i64,z,ldz as i64) {return -6;}
            }
            let (mut qw,mut qr,mut qi)=(<$t>::z(),<$r>::z(),0 as $i);
            let r=[<LAPACKE_ $p stedc_work $($suffix)*>](layout,compz,n,d,e,z,ldz,&mut qw,-1,&mut qr,-1,&mut qi,-1);
            if r!=0 {return r;}
            let (lw,lr,li)=(qw.re().max(1.0) as usize,qr.re().max(1.0) as usize,(qi as usize).max(1));
            let mut work=vec![<$t>::z();lw];
            let mut rwork=vec![<$r>::z();lr];
            let mut iwork=vec![0 as $i;li];
            [<LAPACKE_ $p stedc_work $($suffix)*>](layout,compz,n,d,e,z,ldz,work.as_mut_ptr(),lw as $i,rwork.as_mut_ptr(),lr as $i,iwork.as_mut_ptr(),li as $i)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p stedc_work $($suffix)*>](
            layout:i32,compz:u8,n:$i,d:*mut $r,e:*mut $r,z:*mut $t,ldz:$i,
            work:*mut $t,lwork:$i,rwork:*mut $r,lrwork:$i,iwork:*mut $i,liwork:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            let cz=compz.to_ascii_uppercase();
            if layout==ROW_MAJOR && ldz<n {return -7;}
            if layout==COL_MAJOR || n==0 || lwork==-1 || lrwork==-1 || liwork==-1 {
                let mut info:$i=0;
                let ld=if layout==ROW_MAJOR {n.max(1)} else {ldz};
                crate::tridiagonal_evd::[<$p stedc $fortran>](&compz,&n,d,e,z,&ld,work,&lwork,rwork,&lrwork,iwork,&liwork,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let mut zt=match z_in(cz==b'V',nn,z,ldz as usize){Ok(v)=>v,Err(err)=>return err as $i};
            let ld=nn.max(1) as $i;
            let mut info:$i=0;
            crate::tridiagonal_evd::[<$p stedc $fortran>](&compz,&n,d,e,zt.as_mut_ptr(),&ld,work,&lwork,rwork,&lrwork,iwork,&liwork,&mut info);
            if info>=0 && cz!=b'N' {z_out(nn,z,ldz as usize,&zt);}
            if info<0 {info-1} else {info}
        }
    }};
}
steqr_exports!(s, f32, f32, i32, [], _);
steqr_exports!(d, f64, f64, i32, [], _);
steqr_exports!(c, colmat::c32, f32, i32, [], _);
steqr_exports!(z, colmat::c64, f64, i32, [], _);
steqr_exports!(s, f32, f32, i64, [64_], _64_);
steqr_exports!(d, f64, f64, i64, [64_], _64_);
steqr_exports!(c, colmat::c32, f32, i64, [64_], _64_);
steqr_exports!(z, colmat::c64, f64, i64, [64_], _64_);
sterf_exports!(s, f32, i32, [], _);
sterf_exports!(d, f64, i32, [], _);
sterf_exports!(s, f32, i64, [64_], _64_);
sterf_exports!(d, f64, i64, [64_], _64_);
stev_exports!(s, f32, i32, [], _);
stev_exports!(d, f64, i32, [], _);
stev_exports!(s, f32, i64, [64_], _64_);
stev_exports!(d, f64, i64, [64_], _64_);
stedc_real_exports!(s, f32, i32, [], _);
stedc_real_exports!(d, f64, i32, [], _);
stedc_real_exports!(s, f32, i64, [64_], _64_);
stedc_real_exports!(d, f64, i64, [64_], _64_);
stedc_complex_exports!(c, colmat::c32, f32, i32, [], _);
stedc_complex_exports!(z, colmat::c64, f64, i32, [], _);
stedc_complex_exports!(c, colmat::c32, f32, i64, [64_], _64_);
stedc_complex_exports!(z, colmat::c64, f64, i64, [64_], _64_);
