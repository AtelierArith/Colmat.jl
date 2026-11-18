//! LAPACKE HGEQZ high/work adapters.
//! Reference: Reference LAPACK v3.12.1 LAPACKE/src/lapacke_{s,d,c,z}hgeqz{,_work}.c.
use super::{ge_nancheck, row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;

unsafe fn convert<T: Scalar>(
    layout: i32,
    n: i64,
    h: *mut T,
    ldh: i64,
    t: *mut T,
    ldt: i64,
    q: *mut T,
    ldq: i64,
    z: *mut T,
    ldz: i64,
    compq: u8,
    compz: u8,
    ldq_err: i64,
    ldz_err: i64,
    mut call: impl FnMut(*mut T, i64, *mut T, i64, *mut T, i64, *mut T, i64) -> i64,
) -> i64 {
    if layout != ROW_MAJOR && layout != COL_MAJOR {
        return -1;
    }
    if layout == ROW_MAJOR && ldh < n {
        return -9;
    }
    if layout == ROW_MAJOR && ldt < n {
        return -11;
    }
    if layout == ROW_MAJOR && ldq < n {
        return ldq_err;
    }
    if layout == ROW_MAJOR && ldz < n {
        return ldz_err;
    }
    if layout == COL_MAJOR || n == 0 {
        return call(h, ldh, t, ldt, q, ldq, z, ldz);
    }
    let nn = n as usize;
    let mut ht = match row_to_column(nn, nn, h, ldh as usize) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let mut tt = match row_to_column(nn, nn, t, ldt as usize) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let mut qt = if compq == b'V' {
        match row_to_column(nn, nn, q, ldq as usize) {
            Ok(v) => v,
            Err(e) => return e,
        }
    } else {
        Vec::new()
    };
    let mut zt = if compz == b'V' {
        match row_to_column(nn, nn, z, ldz as usize) {
            Ok(v) => v,
            Err(e) => return e,
        }
    } else {
        Vec::new()
    };
    let ld = nn.max(1) as i64;
    let info = call(
        ht.as_mut_ptr(),
        ld,
        tt.as_mut_ptr(),
        ld,
        qt.as_mut_ptr(),
        ld,
        zt.as_mut_ptr(),
        ld,
    );
    if info >= 0 {
        for i in 0..nn {
            for j in 0..nn {
                *h.add(i * ldh as usize + j) = ht[i + j * nn];
                *t.add(i * ldt as usize + j) = tt[i + j * nn];
                if compq == b'V' {
                    *q.add(i * ldq as usize + j) = qt[i + j * nn];
                }
                if compz == b'V' {
                    *z.add(i * ldz as usize + j) = zt[i + j * nn];
                }
            }
        }
    }
    info
}

macro_rules! real {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p hgeqz $($suffix)*>](
            layout:i32,job:u8,compq:u8,compz:u8,n:$i,ilo:$i,ihi:$i,
            h:*mut $t,ldh:$i,t:*mut $t,ldt:$i,ar:*mut $t,ai:*mut $t,be:*mut $t,
            q:*mut $t,ldq:$i,z:*mut $t,ldz:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            let (cq,cz)=(compq.to_ascii_uppercase(),compz.to_ascii_uppercase());
            if LAPACKE_get_nancheck()!=0 {
                if ge_nancheck(layout,n as i64,n as i64,h,ldh as i64).unwrap_or(false) {return -8;}
                if cq==b'V' && ge_nancheck(layout,n as i64,n as i64,q,ldq as i64).unwrap_or(false) {return -15;}
                if ge_nancheck(layout,n as i64,n as i64,t,ldt as i64).unwrap_or(false) {return -10;}
                if cz==b'V' && ge_nancheck(layout,n as i64,n as i64,z,ldz as i64).unwrap_or(false) {return -17;}
            }
            let mut qw=<$t>::z();
            let mut info=[<LAPACKE_ $p hgeqz_work $($suffix)*>](layout,job,compq,compz,n,ilo,ihi,h,ldh,t,ldt,ar,ai,be,q,ldq,z,ldz,&mut qw,-1);
            if info!=0 {return info;}
            let lw=qw.re().max(1.0) as usize;
            let mut work=vec![<$t>::z();lw];
            info=[<LAPACKE_ $p hgeqz_work $($suffix)*>](layout,job,compq,compz,n,ilo,ihi,h,ldh,t,ldt,ar,ai,be,q,ldq,z,ldz,work.as_mut_ptr(),lw as $i);
            info
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p hgeqz_work $($suffix)*>](
            layout:i32,job:u8,compq:u8,compz:u8,n:$i,ilo:$i,ihi:$i,
            h:*mut $t,ldh:$i,t:*mut $t,ldt:$i,ar:*mut $t,ai:*mut $t,be:*mut $t,
            q:*mut $t,ldq:$i,z:*mut $t,ldz:$i,work:*mut $t,lwork:$i,
        )->$i {
            let (cq,cz)=(compq.to_ascii_uppercase(),compz.to_ascii_uppercase());
            convert(layout,n as i64,h,ldh as i64,t,ldt as i64,q,ldq as i64,z,ldz as i64,cq,cz,-16,-18,|h,lh,t,lt,q,lq,z,lz|{
                let mut info:$i=0;
                crate::qz::[<$p hgeqz $fortran>](&job,&cq,&cz,&n,&ilo,&ihi,h,&(lh as $i),t,&(lt as $i),ar,ai,be,q,&(lq as $i),z,&(lz as $i),work,&lwork,&mut info);
                if info<0 {(info-1) as i64} else {info as i64}
            }) as $i
        }
    }};
}

macro_rules! complex {
    ($p:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p hgeqz $($suffix)*>](
            layout:i32,job:u8,compq:u8,compz:u8,n:$i,ilo:$i,ihi:$i,
            h:*mut $t,ldh:$i,t:*mut $t,ldt:$i,alpha:*mut $t,beta:*mut $t,
            q:*mut $t,ldq:$i,z:*mut $t,ldz:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            let (cq,cz)=(compq.to_ascii_uppercase(),compz.to_ascii_uppercase());
            if LAPACKE_get_nancheck()!=0 {
                if ge_nancheck(layout,n as i64,n as i64,h,ldh as i64).unwrap_or(false) {return -8;}
                if cq==b'V' && ge_nancheck(layout,n as i64,n as i64,q,ldq as i64).unwrap_or(false) {return -14;}
                if ge_nancheck(layout,n as i64,n as i64,t,ldt as i64).unwrap_or(false) {return -10;}
                if cz==b'V' && ge_nancheck(layout,n as i64,n as i64,z,ldz as i64).unwrap_or(false) {return -16;}
            }
            let mut qw=<$t>::z();
            let mut rwork=vec![<$r>::z();(n.max(0) as usize).max(1)];
            let mut info=[<LAPACKE_ $p hgeqz_work $($suffix)*>](layout,job,compq,compz,n,ilo,ihi,h,ldh,t,ldt,alpha,beta,q,ldq,z,ldz,&mut qw,-1,rwork.as_mut_ptr());
            if info!=0 {return info;}
            let lw=qw.re().max(1.0) as usize;
            let mut work=vec![<$t>::z();lw];
            info=[<LAPACKE_ $p hgeqz_work $($suffix)*>](layout,job,compq,compz,n,ilo,ihi,h,ldh,t,ldt,alpha,beta,q,ldq,z,ldz,work.as_mut_ptr(),lw as $i,rwork.as_mut_ptr());
            info
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p hgeqz_work $($suffix)*>](
            layout:i32,job:u8,compq:u8,compz:u8,n:$i,ilo:$i,ihi:$i,
            h:*mut $t,ldh:$i,t:*mut $t,ldt:$i,alpha:*mut $t,beta:*mut $t,
            q:*mut $t,ldq:$i,z:*mut $t,ldz:$i,work:*mut $t,lwork:$i,rwork:*mut $r,
        )->$i {
            let (cq,cz)=(compq.to_ascii_uppercase(),compz.to_ascii_uppercase());
            convert(layout,n as i64,h,ldh as i64,t,ldt as i64,q,ldq as i64,z,ldz as i64,cq,cz,-15,-17,|h,lh,t,lt,q,lq,z,lz|{
                let mut info:$i=0;
                crate::qz::[<$p hgeqz $fortran>](&job,&cq,&cz,&n,&ilo,&ihi,h,&(lh as $i),t,&(lt as $i),alpha,beta,q,&(lq as $i),z,&(lz as $i),work,&lwork,rwork,&mut info);
                if info<0 {(info-1) as i64} else {info as i64}
            }) as $i
        }
    }};
}
real!(s, f32, i32, [], _);
real!(d, f64, i32, [], _);
complex!(c, colmat::c32, f32, i32, [], _);
complex!(z, colmat::c64, f64, i32, [], _);
real!(s, f32, i64, [64_], _64_);
real!(d, f64, i64, [64_], _64_);
complex!(c, colmat::c32, f32, i64, [64_], _64_);
complex!(z, colmat::c64, f64, i64, [64_], _64_);
