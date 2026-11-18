//! LAPACKE GGHRD/GGHD3 high/work adapters.
//! Reference: Reference LAPACK v3.12.1 LAPACKE/src/lapacke_{s,d,c,z}{gghrd,gghd3}{,_work}.c.
use super::{ge_nancheck, row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;

unsafe fn convert<T: Scalar>(
    layout: i32,
    n: i64,
    a: *mut T,
    lda: i64,
    b: *mut T,
    ldb: i64,
    q: *mut T,
    ldq: i64,
    z: *mut T,
    ldz: i64,
    compq: u8,
    compz: u8,
    mut call: impl FnMut(*mut T, i64, *mut T, i64, *mut T, i64, *mut T, i64) -> i64,
) -> i64 {
    if layout != ROW_MAJOR && layout != COL_MAJOR {
        return -1;
    }
    if layout == ROW_MAJOR && lda < n {
        return -8;
    }
    if layout == ROW_MAJOR && ldb < n {
        return -10;
    }
    if layout == ROW_MAJOR && ldq < n {
        return -12;
    }
    if layout == ROW_MAJOR && ldz < n {
        return -14;
    }
    if layout == COL_MAJOR || n == 0 {
        return call(a, lda, b, ldb, q, ldq, z, ldz);
    }
    let nn = n as usize;
    let mut at = match row_to_column(nn, nn, a, lda as usize) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let mut bt = match row_to_column(nn, nn, b, ldb as usize) {
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
        at.as_mut_ptr(),
        ld,
        bt.as_mut_ptr(),
        ld,
        qt.as_mut_ptr(),
        ld,
        zt.as_mut_ptr(),
        ld,
    );
    if info >= 0 {
        for i in 0..nn {
            for j in 0..nn {
                *a.add(i * lda as usize + j) = at[i + j * nn];
                *b.add(i * ldb as usize + j) = bt[i + j * nn];
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

macro_rules! exports {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p gghrd $($suffix)*>](
            layout:i32,compq:u8,compz:u8,n:$i,ilo:$i,ihi:$i,a:*mut $t,lda:$i,
            b:*mut $t,ldb:$i,q:*mut $t,ldq:$i,z:*mut $t,ldz:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if ge_nancheck(layout,n as i64,n as i64,a,lda as i64).unwrap_or(false) {return -7;}
                if ge_nancheck(layout,n as i64,n as i64,b,ldb as i64).unwrap_or(false) {return -9;}
                if compq.to_ascii_uppercase()==b'V' && ge_nancheck(layout,n as i64,n as i64,q,ldq as i64).unwrap_or(false) {return -11;}
                if compz.to_ascii_uppercase()==b'V' && ge_nancheck(layout,n as i64,n as i64,z,ldz as i64).unwrap_or(false) {return -13;}
            }
            [<LAPACKE_ $p gghrd_work $($suffix)*>](layout,compq,compz,n,ilo,ihi,a,lda,b,ldb,q,ldq,z,ldz)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p gghrd_work $($suffix)*>](
            layout:i32,compq:u8,compz:u8,n:$i,ilo:$i,ihi:$i,a:*mut $t,lda:$i,
            b:*mut $t,ldb:$i,q:*mut $t,ldq:$i,z:*mut $t,ldz:$i,
        )->$i {
            let (cq,cz)=(compq.to_ascii_uppercase(),compz.to_ascii_uppercase());
            convert(layout,n as i64,a,lda as i64,b,ldb as i64,q,ldq as i64,z,ldz as i64,cq,cz,|a,la,b,lb,q,lq,z,lz|{
                let mut info:$i=0;
                crate::generalized_hessenberg::[<$p gghrd $fortran>](&cq,&cz,&n,&ilo,&ihi,a,&(la as $i),b,&(lb as $i),q,&(lq as $i),z,&(lz as $i),&mut info);
                if info<0 {(info-1) as i64} else {info as i64}
            }) as $i
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p gghd3 $($suffix)*>](
            layout:i32,compq:u8,compz:u8,n:$i,ilo:$i,ihi:$i,a:*mut $t,lda:$i,
            b:*mut $t,ldb:$i,q:*mut $t,ldq:$i,z:*mut $t,ldz:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if ge_nancheck(layout,n as i64,n as i64,a,lda as i64).unwrap_or(false) {return -7;}
                if ge_nancheck(layout,n as i64,n as i64,b,ldb as i64).unwrap_or(false) {return -9;}
                if compq.to_ascii_uppercase()==b'V' && ge_nancheck(layout,n as i64,n as i64,q,ldq as i64).unwrap_or(false) {return -11;}
                if compz.to_ascii_uppercase()==b'V' && ge_nancheck(layout,n as i64,n as i64,z,ldz as i64).unwrap_or(false) {return -13;}
            }
            let mut work=vec![<$t>::z();(3*n.max(0) as usize).max(1)];
            [<LAPACKE_ $p gghd3_work $($suffix)*>](layout,compq,compz,n,ilo,ihi,a,lda,b,ldb,q,ldq,z,ldz,work.as_mut_ptr(),work.len() as $i)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p gghd3_work $($suffix)*>](
            layout:i32,compq:u8,compz:u8,n:$i,ilo:$i,ihi:$i,a:*mut $t,lda:$i,
            b:*mut $t,ldb:$i,q:*mut $t,ldq:$i,z:*mut $t,ldz:$i,
            work:*mut $t,lwork:$i,
        )->$i {
            let (cq,cz)=(compq.to_ascii_uppercase(),compz.to_ascii_uppercase());
            convert(layout,n as i64,a,lda as i64,b,ldb as i64,q,ldq as i64,z,ldz as i64,cq,cz,|a,la,b,lb,q,lq,z,lz|{
                let mut info:$i=0;
                crate::generalized_hessenberg::[<$p gghd3 $fortran>](&cq,&cz,&n,&ilo,&ihi,a,&(la as $i),b,&(lb as $i),q,&(lq as $i),z,&(lz as $i),work,&lwork,&mut info);
                if info<0 {(info-1) as i64} else {info as i64}
            }) as $i
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
