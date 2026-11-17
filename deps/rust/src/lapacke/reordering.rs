//! LAPACKE TREXC high/work adapters.
//! Reference: Reference LAPACK v3.12.1 LAPACKE/src/lapacke_{s,d,c,z}trexc{,_work}.c.
use super::{ge_nancheck, row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;
use crate::lapack::LapackInt;

unsafe fn run<T: Scalar, I: LapackInt>(
    layout: i32,
    compq: u8,
    n: I,
    t: *mut T,
    ldt: I,
    q: *mut T,
    ldq: I,
    mut ifst: I,
    mut ilst: I,
    high: bool,
    mut call: impl FnMut(*mut T, I, *mut T, I, *mut I, *mut I) -> I,
) -> I {
    if layout != ROW_MAJOR && layout != COL_MAJOR {
        return I::from_i64(-1);
    }
    let compq = compq.to_ascii_uppercase();
    if !matches!(compq, b'N' | b'V') {
        return I::from_i64(-2);
    }
    if n.val() < 0 {
        return I::from_i64(-3);
    }
    if high && LAPACKE_get_nancheck() != 0 {
        if ge_nancheck(layout, n.val() as i64, n.val() as i64, t, ldt.val() as i64).unwrap_or(false)
        {
            return I::from_i64(-4);
        }
        if compq == b'V'
            && ge_nancheck(
                compq_n(layout),
                n.val() as i64,
                n.val() as i64,
                q,
                ldq.val() as i64,
            )
            .unwrap_or(false)
        {
            return I::from_i64(-6);
        }
    }
    if layout == COL_MAJOR {
        return call(t, ldt, q, ldq, &mut ifst, &mut ilst);
    }
    let nn = n.val() as usize;
    if ldt.val() < n.val() {
        return I::from_i64(-5);
    }
    if compq == b'V' && ldq.val() < n.val() {
        return I::from_i64(-7);
    }
    if nn == 0 {
        return I::from_i64(0);
    }
    let mut tt = match row_to_column(nn, nn, t, ldt.val() as usize) {
        Ok(v) => v,
        Err(e) => return I::from_i64(e),
    };
    let mut qt = if compq == b'V' {
        match row_to_column(nn, nn, q, ldq.val() as usize) {
            Ok(v) => v,
            Err(e) => return I::from_i64(e),
        }
    } else {
        Vec::new()
    };
    let ld = I::from_i64(nn.max(1) as i64);
    let info = call(
        tt.as_mut_ptr(),
        ld,
        qt.as_mut_ptr(),
        ld,
        &mut ifst,
        &mut ilst,
    );
    if info.val() >= 0 {
        for i in 0..nn {
            for j in 0..nn {
                *t.add(i * ldt.val() as usize + j) = tt[i + j * nn];
                if compq == b'V' {
                    *q.add(i * ldq.val() as usize + j) = qt[i + j * nn];
                }
            }
        }
    }
    info
}
fn compq_n(layout: i32) -> i32 {
    layout
}

macro_rules! real {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p trexc $($suffix)*>](
            layout:i32,compq:u8,n:$i,t:*mut $t,ldt:$i,q:*mut $t,ldq:$i,
            ifst:*mut $i,ilst:*mut $i,
        )->$i {
            let mut work=vec![<$t>::z(); n.max(0) as usize];
            let _=work.as_mut_ptr();
            run::<$t,$i>(layout,compq,n,t,ldt,q,ldq,*ifst,*ilst,true,|t,ldt,q,ldq,ifst,ilst|{
                let mut info:$i=0;
                crate::schur_reorder::[<$p trexc $fortran>](&compq,&n,t,&ldt,q,&ldq,ifst,ilst,work.as_mut_ptr(),&mut info);
                if info<0 {<$i>::from_i64((info-1) as i64)} else {info}
            })
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p trexc_work $($suffix)*>](
            layout:i32,compq:u8,n:$i,t:*mut $t,ldt:$i,q:*mut $t,ldq:$i,
            ifst:*mut $i,ilst:*mut $i,_work:*mut $t,
        )->$i {
            run::<$t,$i>(layout,compq,n,t,ldt,q,ldq,*ifst,*ilst,false,|t,ldt,q,ldq,ifst,ilst|{
                let mut info:$i=0;
                crate::schur_reorder::[<$p trexc $fortran>](&compq,&n,t,&ldt,q,&ldq,ifst,ilst,_work,&mut info);
                if info<0 {<$i>::from_i64((info-1) as i64)} else {info}
            })
        }
    }};
}

macro_rules! complex {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p trexc $($suffix)*>](
            layout:i32,compq:u8,n:$i,t:*mut $t,ldt:$i,q:*mut $t,ldq:$i,
            ifst:$i,ilst:$i,
        )->$i {
            let mut work=vec![<$t>::z(); n.max(0) as usize];
            let _=work.as_mut_ptr();
            run::<$t,$i>(layout,compq,n,t,ldt,q,ldq,ifst,ilst,true,|t,ldt,q,ldq,ifst,ilst|{
                let mut info:$i=0;
                crate::schur_reorder::[<$p trexc $fortran>](&compq,&n,t,&ldt,q,&ldq,ifst,ilst,&mut info);
                if info<0 {<$i>::from_i64((info-1) as i64)} else {info}
            })
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p trexc_work $($suffix)*>](
            layout:i32,compq:u8,n:$i,t:*mut $t,ldt:$i,q:*mut $t,ldq:$i,
            ifst:$i,ilst:$i,_work:*mut $t,
        )->$i {
            run::<$t,$i>(layout,compq,n,t,ldt,q,ldq,ifst,ilst,false,|t,ldt,q,ldq,ifst,ilst|{
                let mut info:$i=0;
                crate::schur_reorder::[<$p trexc $fortran>](&compq,&n,t,&ldt,q,&ldq,ifst,ilst,&mut info);
                if info<0 {<$i>::from_i64((info-1) as i64)} else {info}
            })
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
