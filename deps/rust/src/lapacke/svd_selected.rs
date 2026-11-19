//! LAPACKE GESVDX high/work adapters for selected singular triplets.
//! Reference: Reference LAPACK v3.12.1 LAPACKE/src/lapacke_{s,d,c,z}gesvdx{,_work}.c.
//! The high-level entry queries WORK/LWORK and uses the caller's SUPERB array as
//! the integer workspace (12*min(M,N) entries). Row-major U is M-by-NS and VT is
//! NS-by-N; only the NS selected triplets are written back.
use super::{row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;
use crate::lapack::LapackInt;

fn columns(job: u8, range: u8, k: i64, il: i64, iu: i64) -> i128 {
    if job.to_ascii_uppercase() != b'V' {
        1
    } else if range.to_ascii_uppercase() == b'I' {
        iu as i128 - il as i128 + 1
    } else {
        k as i128
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn svdx_job<T: Scalar, I: LapackInt>(
    layout: i32,
    jobu: u8,
    jobvt: u8,
    range: u8,
    m: i64,
    n: i64,
    a: *mut T,
    lda: i64,
    il: i64,
    iu: i64,
    ns: *const I,
    u: *mut T,
    ldu: i64,
    vt: *mut T,
    ldvt: i64,
    query: bool,
    mut call: impl FnMut(*mut T, i64, *mut T, i64, *mut T, i64) -> i64,
) -> i64 {
    if layout != ROW_MAJOR && layout != COL_MAJOR {
        return -1;
    }
    let k = m.min(n);
    if layout == ROW_MAJOR {
        if lda < n {
            return -8;
        }
        if (ldu as i128) < columns(jobu, range, k, il, iu) {
            return -16;
        }
        if ldvt < n {
            return -18;
        }
    }
    if layout == COL_MAJOR || m <= 0 || n <= 0 {
        return call(a, lda, u, ldu, vt, ldvt);
    }
    if query {
        return call(a, m.max(1), u, m.max(1), vt, k.max(1));
    }
    let (mm, nn, kk) = (m as usize, n as usize, k as usize);
    let mut at = match row_to_column(mm, nn, a, lda as usize) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let mut ut = vec![T::z(); mm * kk];
    let mut vtt = vec![T::z(); kk * nn];
    let info = call(at.as_mut_ptr(), m, ut.as_mut_ptr(), m, vtt.as_mut_ptr(), k);
    if info >= 0 {
        let count = (*ns).val().clamp(0, kk as isize) as usize;
        if jobu.to_ascii_uppercase() == b'V' {
            for i in 0..mm {
                for j in 0..count {
                    *u.add(i * ldu as usize + j) = ut[i + j * mm];
                }
            }
        }
        if jobvt.to_ascii_uppercase() == b'V' {
            for i in 0..count {
                for j in 0..nn {
                    *vt.add(i * ldvt as usize + j) = vtt[i + j * kk];
                }
            }
        }
    }
    info
}

macro_rules! gesvdx_exports {
    ($p:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fort:tt,[$($rw:ident)*])=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p gesvdx $($suffix)*>](
            layout:i32,jobu:u8,jobvt:u8,range:u8,m:$i,n:$i,a:*mut $t,lda:$i,vl:$r,vu:$r,il:$i,iu:$i,
            ns:*mut $i,s:*mut $r,u:*mut $t,ldu:$i,vt:*mut $t,ldvt:$i,superb:*mut $i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if super::ge_nancheck(layout,m as i64,n as i64,a,lda as i64).unwrap_or(false) {return -7;}
                if range.to_ascii_uppercase()==b'V' {
                    if vl.is_nan() {return -9;}
                    if vu.is_nan() {return -10;}
                }
            }
            let mut qw=<$t>::z();
            let r=[<LAPACKE_ $p gesvdx_work $($suffix)*>](layout,jobu,jobvt,range,m,n,a,lda,vl,vu,il,iu,ns,s,u,ldu,vt,ldvt,&mut qw,-1,
                $({let _=stringify!($rw);std::ptr::null_mut::<$r>()},)*superb);
            if r!=0 {return r;}
            let lw=qw.re().max(1.0) as usize;
            let mut work=vec![<$t>::z();lw];
            let k=(m.min(n)).max(0) as usize;
            $(let mut $rw=vec![<$r>::z();(17*k*k).max(1)];)*
            [<LAPACKE_ $p gesvdx_work $($suffix)*>](layout,jobu,jobvt,range,m,n,a,lda,vl,vu,il,iu,ns,s,u,ldu,vt,ldvt,work.as_mut_ptr(),lw as $i,
                $($rw.as_mut_ptr(),)*superb)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p gesvdx_work $($suffix)*>](
            layout:i32,jobu:u8,jobvt:u8,range:u8,m:$i,n:$i,a:*mut $t,lda:$i,vl:$r,vu:$r,il:$i,iu:$i,
            ns:*mut $i,s:*mut $r,u:*mut $t,ldu:$i,vt:*mut $t,ldvt:$i,work:*mut $t,lwork:$i,$($rw:*mut $r,)*iwork:*mut $i,
        )->$i {
            svdx_job::<$t,$i>(layout,jobu,jobvt,range,m as i64,n as i64,a,lda as i64,il as i64,iu as i64,ns,u,ldu as i64,vt,ldvt as i64,
                lwork== -1,|aa,la,uu,lu,vv,lv|{
                let mut info:$i=0;
                crate::advanced_svd::[<$p gesvdx $fort>](&jobu,&jobvt,&range,&m,&n,aa,&(la as $i),&vl,&vu,&il,&iu,ns,s,uu,&(lu as $i),vv,&(lv as $i),
                    work,&lwork,$($rw,)*iwork,&mut info);
                if info<0 {(info-1) as i64} else {info as i64}
            }) as $i
        }
    }};
}
macro_rules! all_abi {
    ($i:ty,[$($suffix:tt)*],$fort:tt)=>{
        gesvdx_exports!(s,f32,f32,$i,[$($suffix)*],$fort,[]);
        gesvdx_exports!(d,f64,f64,$i,[$($suffix)*],$fort,[]);
        gesvdx_exports!(c,colmat::c32,f32,$i,[$($suffix)*],$fort,[rwork]);
        gesvdx_exports!(z,colmat::c64,f64,$i,[$($suffix)*],$fort,[rwork]);
    };
}
all_abi!(i32, [], _);
all_abi!(i64, [64_], _64_);
