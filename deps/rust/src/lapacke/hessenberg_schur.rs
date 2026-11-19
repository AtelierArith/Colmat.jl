//! LAPACKE HSEQR high/work adapters for the Schur form of a Hessenberg matrix.
//! Reference: Reference LAPACK v3.12.1 LAPACKE/src/lapacke_{s,d,c,z}hseqr{,_work}.c.
use super::{ge_nancheck, row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;
use crate::lapack::LapackInt;

macro_rules! real {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p hseqr $($suffix)*>](
            layout:i32,job:u8,compz:u8,n:$i,ilo:$i,ihi:$i,
            h:*mut $t,ldh:$i,wr:*mut $t,wi:*mut $t,z:*mut $t,ldz:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if ge_nancheck(layout,n as i64,n as i64,h,ldh as i64).unwrap_or(false) {return -7;}
                let c=compz.to_ascii_uppercase();
                if (c==b'I'||c==b'V') && ge_nancheck(layout,n as i64,n as i64,z,ldz as i64).unwrap_or(false) {return -11;}
            }
            if layout==ROW_MAJOR && ldh<n {return -8;}
            if layout==ROW_MAJOR && ldz<n {return -12;}
            let mut query=<$t>::z();
            let mut info=<$i>::from_i64(0);
            crate::schur::[<$p hseqr $fortran>](&job,&compz,&n,&ilo,&ihi,h,&ldh,wr,wi,z,&ldz,&mut query,&mut (-1 as $i),&mut info);
            if info!=0 {return info;}
            let lw=query.re().max(1.0) as usize;
            let mut work=vec![<$t>::z();lw];
            [<LAPACKE_ $p hseqr_work $($suffix)*>](layout,job,compz,n,ilo,ihi,h,ldh,wr,wi,z,ldz,work.as_mut_ptr(),lw as $i)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p hseqr_work $($suffix)*>](
            layout:i32,job:u8,compz:u8,n:$i,ilo:$i,ihi:$i,
            h:*mut $t,ldh:$i,wr:*mut $t,wi:*mut $t,z:*mut $t,ldz:$i,
            work:*mut $t,lwork:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && ldh<n {return -8;}
            if layout==ROW_MAJOR && ldz<n {return -12;}
            if layout==COL_MAJOR || n==0 {
                let mut info:$i=0;
                crate::schur::[<$p hseqr $fortran>](&job,&compz,&n,&ilo,&ihi,h,&ldh,wr,wi,z,&ldz,work,&lwork,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let c=compz.to_ascii_uppercase();
            let want_z=c==b'I'||c==b'V';
            let mut ht=match row_to_column(nn,nn,h,ldh as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut zt=if want_z {match row_to_column(nn,nn,z,ldz as usize){Ok(v)=>v,Err(e)=>return e as $i}} else {Vec::new()};
            let ld=nn.max(1) as $i;
            let mut info:$i=0;
            crate::schur::[<$p hseqr $fortran>](&job,&compz,&n,&ilo,&ihi,ht.as_mut_ptr(),&ld,wr,wi,zt.as_mut_ptr(),&ld,work,&lwork,&mut info);
            if info>=0 {
                for i in 0..nn {for j in 0..nn {
                    *h.add(i*ldh as usize+j)=ht[i+j*nn];
                    if want_z {*z.add(i*ldz as usize+j)=zt[i+j*nn];}
                }}
            }
            if info<0 {info-1} else {info}
        }
    }};
}

macro_rules! complex {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p hseqr $($suffix)*>](
            layout:i32,job:u8,compz:u8,n:$i,ilo:$i,ihi:$i,
            h:*mut $t,ldh:$i,w:*mut $t,z:*mut $t,ldz:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if ge_nancheck(layout,n as i64,n as i64,h,ldh as i64).unwrap_or(false) {return -7;}
                let c=compz.to_ascii_uppercase();
                if (c==b'I'||c==b'V') && ge_nancheck(layout,n as i64,n as i64,z,ldz as i64).unwrap_or(false) {return -11;}
            }
            if layout==ROW_MAJOR && ldh<n {return -8;}
            if layout==ROW_MAJOR && ldz<n {return -12;}
            let mut query=<$t>::z();
            let mut info=<$i>::from_i64(0);
            crate::schur::[<$p hseqr $fortran>](&job,&compz,&n,&ilo,&ihi,h,&ldh,w,z,&ldz,&mut query,&mut (-1 as $i),&mut info);
            if info!=0 {return info;}
            let lw=query.re().max(1.0) as usize;
            let mut work=vec![<$t>::z();lw];
            [<LAPACKE_ $p hseqr_work $($suffix)*>](layout,job,compz,n,ilo,ihi,h,ldh,w,z,ldz,work.as_mut_ptr(),lw as $i)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p hseqr_work $($suffix)*>](
            layout:i32,job:u8,compz:u8,n:$i,ilo:$i,ihi:$i,
            h:*mut $t,ldh:$i,w:*mut $t,z:*mut $t,ldz:$i,
            work:*mut $t,lwork:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && ldh<n {return -8;}
            if layout==ROW_MAJOR && ldz<n {return -12;}
            if layout==COL_MAJOR || n==0 {
                let mut info:$i=0;
                crate::schur::[<$p hseqr $fortran>](&job,&compz,&n,&ilo,&ihi,h,&ldh,w,z,&ldz,work,&lwork,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let c=compz.to_ascii_uppercase();
            let want_z=c==b'I'||c==b'V';
            let mut ht=match row_to_column(nn,nn,h,ldh as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut zt=if want_z {match row_to_column(nn,nn,z,ldz as usize){Ok(v)=>v,Err(e)=>return e as $i}} else {Vec::new()};
            let ld=nn.max(1) as $i;
            let mut info:$i=0;
            crate::schur::[<$p hseqr $fortran>](&job,&compz,&n,&ilo,&ihi,ht.as_mut_ptr(),&ld,w,zt.as_mut_ptr(),&ld,work,&lwork,&mut info);
            if info>=0 {
                for i in 0..nn {for j in 0..nn {
                    *h.add(i*ldh as usize+j)=ht[i+j*nn];
                    if want_z {*z.add(i*ldz as usize+j)=zt[i+j*nn];}
                }}
            }
            if info<0 {info-1} else {info}
        }
    }};
}
real!(s, f32, i32, [], _);
real!(d, f64, i32, [], _);
complex!(c, colmat::c32, i32, [], _);
complex!(z, colmat::c64, i32, [], _);
real!(s, f32, i64, [64_], _64_);
real!(d, f64, i64, [64_], _64_);
complex!(c, colmat::c32, i64, [64_], _64_);
complex!(z, colmat::c64, i64, [64_], _64_);
