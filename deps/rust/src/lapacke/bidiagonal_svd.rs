//! LAPACKE BDSQR, BDSDC and BDSVDX high/work adapters for the singular value
//! decomposition of a real upper/lower bidiagonal matrix.
//! Reference: Reference LAPACK v3.12.1
//! LAPACKE/src/lapacke_{s,d,c,z}bdsqr{,_work}.c and
//! lapacke_{s,d}{bdsdc,bdsvdx}{,_work}.c.
//! D/E hold the bidiagonal (E has N-1 entries); VT, U and C are the matrices
//! multiplied by the computed singular vectors and are transposed for
//! row-major. BDSDC COMPQ='P' follows the native compact driver.
use super::{row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;
use crate::lapack::LapackInt;

unsafe fn nan_vec<T: Scalar>(n: i64, a: *const T) -> bool {
    (0..n.max(0) as usize).any(|i| {
        let v = *a.add(i);
        v != v
    })
}
unsafe fn nan_mat<T: Scalar>(layout: i32, m: i64, n: i64, a: *const T, ld: i64) -> bool {
    super::ge_nancheck(layout, m, n, a, ld).unwrap_or(false)
}
unsafe fn store<T: Scalar>(src: &[T], rows: usize, cols: usize, dst: *mut T, ld: usize) {
    for i in 0..rows {
        for j in 0..cols {
            *dst.add(i * ld + j) = src[i + j * rows];
        }
    }
}

macro_rules! bdsqr_exports {
    ($p:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fort:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p bdsqr $($suffix)*>](
            layout:i32,uplo:u8,n:$i,ncvt:$i,nru:$i,ncc:$i,d:*mut $r,e:*mut $r,
            vt:*mut $t,ldvt:$i,u:*mut $t,ldu:$i,c:*mut $t,ldc:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if nan_vec(n as i64,d) {return -7;}
                if nan_vec((n as i64-1).max(0),e) {return -8;}
                if ncvt>0 && nan_mat(layout,n as i64,ncvt as i64,vt,ldvt as i64) {return -9;}
                if nru>0 && nan_mat(layout,nru as i64,n as i64,u,ldu as i64) {return -11;}
                if ncc>0 && nan_mat(layout,n as i64,ncc as i64,c,ldc as i64) {return -13;}
            }
            let mut work=vec![<$r>::z();(4*n.max(0) as usize).max(1)];
            [<LAPACKE_ $p bdsqr_work $($suffix)*>](layout,uplo,n,ncvt,nru,ncc,d,e,vt,ldvt,u,ldu,c,ldc,work.as_mut_ptr())
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p bdsqr_work $($suffix)*>](
            layout:i32,uplo:u8,n:$i,ncvt:$i,nru:$i,ncc:$i,d:*mut $r,e:*mut $r,
            vt:*mut $t,ldvt:$i,u:*mut $t,ldu:$i,c:*mut $t,ldc:$i,work:*mut $r,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && ldvt<ncvt {return -10;}
            if layout==ROW_MAJOR && ldu<n {return -12;}
            if layout==ROW_MAJOR && ldc<ncc {return -14;}
            if layout==COL_MAJOR || n<=0 {
                let mut info:$i=0;
                let (lv,lu,lc)=if layout==ROW_MAJOR {(n.max(1),nru.max(1),n.max(1))} else {(ldvt,ldu,ldc)};
                crate::bidiagonal_lapack::[<$p bdsqr $fort>](&uplo,&n,&ncvt,&nru,&ncc,d,e,vt,&lv,u,&lu,c,&lc,work,&mut info);
                return if info<0 {info-1} else {info};
            }
            let (nn,cv,ru,cc)=(n as usize,ncvt.max(0) as usize,nru.max(0) as usize,ncc.max(0) as usize);
            let mut vtt=if cv>0 {match row_to_column(nn,cv,vt,ldvt as usize){Ok(v)=>v,Err(e)=>return e as $i}} else {Vec::new()};
            let mut ut=if ru>0 {match row_to_column(ru,nn,u,ldu as usize){Ok(v)=>v,Err(e)=>return e as $i}} else {Vec::new()};
            let mut ct=if cc>0 {match row_to_column(nn,cc,c,ldc as usize){Ok(v)=>v,Err(e)=>return e as $i}} else {Vec::new()};
            let (lv,lu,lc)=(nn.max(1) as $i,ru.max(1) as $i,nn.max(1) as $i);
            let mut info:$i=0;
            crate::bidiagonal_lapack::[<$p bdsqr $fort>](&uplo,&n,&ncvt,&nru,&ncc,d,e,vtt.as_mut_ptr(),&lv,ut.as_mut_ptr(),&lu,ct.as_mut_ptr(),&lc,work,&mut info);
            if info>=0 {
                if cv>0 {store(&vtt,nn,cv,vt,ldvt as usize);}
                if ru>0 {store(&ut,ru,nn,u,ldu as usize);}
                if cc>0 {store(&ct,nn,cc,c,ldc as usize);}
            }
            if info<0 {info-1} else {info}
        }
    }};
}
macro_rules! bdsdc_exports {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fort:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p bdsdc $($suffix)*>](
            layout:i32,uplo:u8,compq:u8,n:$i,d:*mut $t,e:*mut $t,u:*mut $t,ldu:$i,vt:*mut $t,ldvt:$i,q:*mut $t,iq:*mut $i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if nan_vec(n as i64,d) {return -5;}
                if nan_vec((n as i64-1).max(0),e) {return -6;}
            }
            let nn=n.max(0) as usize;
            let cq=compq.to_ascii_uppercase();
            let lw=match cq {b'N'=>4*nn,b'P'=>6*nn,_=>3*nn*nn+4*nn}.max(1);
            let mut work=vec![<$t>::z();lw];
            let mut iwork=vec![0 as $i;(8*nn).max(1)];
            [<LAPACKE_ $p bdsdc_work $($suffix)*>](layout,uplo,compq,n,d,e,u,ldu,vt,ldvt,q,iq,work.as_mut_ptr(),iwork.as_mut_ptr())
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p bdsdc_work $($suffix)*>](
            layout:i32,uplo:u8,compq:u8,n:$i,d:*mut $t,e:*mut $t,u:*mut $t,ldu:$i,vt:*mut $t,ldvt:$i,q:*mut $t,iq:*mut $i,
            work:*mut $t,iwork:*mut $i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            let cq=compq.to_ascii_uppercase();
            if layout==ROW_MAJOR && cq==b'I' && ldu<n {return -8;}
            if layout==ROW_MAJOR && cq==b'I' && ldvt<n {return -10;}
            if layout==COL_MAJOR || n<=0 || cq!=b'I' {
                let mut info:$i=0;
                let (lu,lv)=if layout==ROW_MAJOR {(n.max(1),n.max(1))} else {(ldu,ldvt)};
                crate::bidiagonal_lapack::[<$p bdsdc $fort>](&uplo,&compq,&n,d,e,u,&lu,vt,&lv,q,iq,work,iwork,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let mut ut=vec![<$t>::z();nn*nn];
            let mut vtt=vec![<$t>::z();nn*nn];
            let ld=nn.max(1) as $i;
            let mut info:$i=0;
            crate::bidiagonal_lapack::[<$p bdsdc $fort>](&uplo,&compq,&n,d,e,ut.as_mut_ptr(),&ld,vtt.as_mut_ptr(),&ld,q,iq,work,iwork,&mut info);
            if info>=0 {
                store(&ut,nn,nn,u,ldu as usize);
                store(&vtt,nn,nn,vt,ldvt as usize);
            }
            if info<0 {info-1} else {info}
        }
    }};
}
macro_rules! bdsvdx_exports {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fort:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p bdsvdx $($suffix)*>](
            layout:i32,uplo:u8,jobz:u8,range:u8,n:$i,d:*mut $t,e:*mut $t,vl:$t,vu:$t,il:$i,iu:$i,
            ns:*mut $i,s:*mut $t,z:*mut $t,ldz:$i,superb:*mut $i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if nan_vec(n as i64,d) {return -6;}
                if nan_vec((n as i64-1).max(0),e) {return -7;}
                if range.to_ascii_uppercase()==b'V' {
                    if vl.is_nan() {return -8;}
                    if vu.is_nan() {return -9;}
                }
            }
            let nn=n.max(0) as usize;
            let mut work=vec![<$t>::z();(14*nn).max(1)];
            let mut iwork=vec![0 as $i;(12*nn).max(1)];
            let _=superb;
            [<LAPACKE_ $p bdsvdx_work $($suffix)*>](layout,uplo,jobz,range,n,d,e,vl,vu,il,iu,ns,s,z,ldz,work.as_mut_ptr(),iwork.as_mut_ptr())
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p bdsvdx_work $($suffix)*>](
            layout:i32,uplo:u8,jobz:u8,range:u8,n:$i,d:*mut $t,e:*mut $t,vl:$t,vu:$t,il:$i,iu:$i,
            ns:*mut $i,s:*mut $t,z:*mut $t,ldz:$i,work:*mut $t,iwork:*mut $i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            let want=jobz.to_ascii_uppercase()==b'V';
            let cols:i128=if !want {1} else if matches!(range.to_ascii_uppercase(),b'A'|b'V') {n as i128}
                else if range.to_ascii_uppercase()==b'I' {iu as i128-il as i128+1} else {1};
            if layout==ROW_MAJOR && (ldz as i128)<cols {return -15;}
            if layout==COL_MAJOR || n<=0 {
                let mut info:$i=0;
                crate::bidiagonal_lapack::[<$p bdsvdx $fort>](&uplo,&jobz,&range,&n,d,e,&vl,&vu,&il,&iu,ns,s,z,&ldz,work,iwork,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let rows=2*nn;
            let mut zt=vec![<$t>::z();rows*nn];
            let ld=rows.max(1) as $i;
            let mut info:$i=0;
            crate::bidiagonal_lapack::[<$p bdsvdx $fort>](&uplo,&jobz,&range,&n,d,e,&vl,&vu,&il,&iu,ns,s,zt.as_mut_ptr(),&ld,work,iwork,&mut info);
            if info>=0 && want {
                let cols=(*ns).val().clamp(0,nn as isize) as usize;
                store(&zt,rows,cols,z,ldz as usize);
            }
            if info<0 {info-1} else {info}
        }
    }};
}
macro_rules! all_abi {
    ($i:ty,[$($suffix:tt)*],$fort:tt)=>{
        bdsqr_exports!(s,f32,f32,$i,[$($suffix)*],$fort);
        bdsqr_exports!(d,f64,f64,$i,[$($suffix)*],$fort);
        bdsqr_exports!(c,colmat::c32,f32,$i,[$($suffix)*],$fort);
        bdsqr_exports!(z,colmat::c64,f64,$i,[$($suffix)*],$fort);
        bdsdc_exports!(s,f32,$i,[$($suffix)*],$fort);
        bdsdc_exports!(d,f64,$i,[$($suffix)*],$fort);
        bdsvdx_exports!(s,f32,$i,[$($suffix)*],$fort);
        bdsvdx_exports!(d,f64,$i,[$($suffix)*],$fort);
    };
}
all_abi!(i32, [], _);
all_abi!(i64, [64_], _64_);
