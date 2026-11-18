//! LAPACKE GBEQU/GBEQUB high/work adapters for band equilibration.
//! Reference: Reference LAPACK v3.12.1 LAPACKE/src/lapacke_{s,d,c,z}{gbequ,gbequb}{,_work}.c.
//! The band input uses the analysis convention AB(KU+1+i-j, j) with LDAB >= KL+KU+1.
use super::{LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;

fn band_rows(kl: usize, ku: usize) -> usize {
    kl + ku + 1
}
fn index(row_major: bool, ku: usize, ld: usize, i: usize, j: usize) -> usize {
    let r = ku + i - j;
    if row_major {
        r * ld + j
    } else {
        r + j * ld
    }
}
unsafe fn band_to_column<T: Scalar>(
    layout: i32,
    m: usize,
    n: usize,
    kl: usize,
    ku: usize,
    ab: *const T,
    ldab: usize,
) -> Vec<T> {
    let ld = band_rows(kl, ku);
    let mut out = vec![T::z(); ld * n];
    let row_major = layout == ROW_MAJOR;
    for j in 0..n {
        let lo = j.saturating_sub(ku).min(m);
        let hi = (j + kl + 1).min(m);
        for i in lo..hi {
            out[index(false, ku, ld, i, j)] = *ab.add(index(row_major, ku, ldab, i, j));
        }
    }
    out
}
unsafe fn band_nan<T: Scalar>(
    layout: i32,
    m: usize,
    n: usize,
    kl: usize,
    ku: usize,
    ab: *const T,
    ldab: usize,
) -> bool {
    let row_major = layout == ROW_MAJOR;
    let ld = if row_major { ldab } else { band_rows(kl, ku) };
    for j in 0..n {
        let lo = j.saturating_sub(ku).min(m);
        let hi = (j + kl + 1).min(m);
        for i in lo..hi {
            let v = *ab.add(index(row_major, ku, ld, i, j));
            if v != v {
                return true;
            }
        }
    }
    false
}

macro_rules! exports {
    ($p:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fortran:tt,$fname:ident)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $fname $($suffix)*>](
            layout:i32,m:$i,n:$i,kl:$i,ku:$i,ab:*const $t,ldab:$i,
            r:*mut $r,c:*mut $r,rowcnd:*mut $r,colcnd:*mut $r,amax:*mut $r,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0
                && band_nan(layout,m.max(0) as usize,n.max(0) as usize,kl.max(0) as usize,ku.max(0) as usize,ab,ldab.max(0) as usize)
            {
                return -6;
            }
            [<LAPACKE_ $p $fname _work $($suffix)*>](layout,m,n,kl,ku,ab,ldab,r,c,rowcnd,colcnd,amax)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $fname _work $($suffix)*>](
            layout:i32,m:$i,n:$i,kl:$i,ku:$i,ab:*const $t,ldab:$i,
            r:*mut $r,c:*mut $r,rowcnd:*mut $r,colcnd:*mut $r,amax:*mut $r,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && ldab<n {return -7;}
            if layout==COL_MAJOR {
                let mut info:$i=0;
                crate::general_equilibration::[<$p $fname $fortran>](&m,&n,&kl,&ku,ab,&ldab,r,c,rowcnd,colcnd,amax,&mut info);
                return info;
            }
            let (mm,nn,kk,uu)=(m as usize,n as usize,kl.max(0) as usize,ku.max(0) as usize);
            let tmp=band_to_column(layout,mm,nn,kk,uu,ab,ldab as usize);
            let ld=band_rows(kk,uu).max(1) as $i;
            let mut info:$i=0;
            crate::general_equilibration::[<$p $fname $fortran>](&m,&n,&kl,&ku,tmp.as_ptr(),&ld,r,c,rowcnd,colcnd,amax,&mut info);
            if info<0 {info-1} else {info}
        }
    }};
}
exports!(s, f32, f32, i32, [], _, gbequ);
exports!(d, f64, f64, i32, [], _, gbequ);
exports!(c, colmat::c32, f32, i32, [], _, gbequ);
exports!(z, colmat::c64, f64, i32, [], _, gbequ);
exports!(s, f32, f32, i64, [64_], _64_, gbequ);
exports!(d, f64, f64, i64, [64_], _64_, gbequ);
exports!(c, colmat::c32, f32, i64, [64_], _64_, gbequ);
exports!(z, colmat::c64, f64, i64, [64_], _64_, gbequ);
exports!(s, f32, f32, i32, [], _, gbequb);
exports!(d, f64, f64, i32, [], _, gbequb);
exports!(c, colmat::c32, f32, i32, [], _, gbequb);
exports!(z, colmat::c64, f64, i32, [], _, gbequb);
exports!(s, f32, f32, i64, [64_], _64_, gbequb);
exports!(d, f64, f64, i64, [64_], _64_, gbequb);
exports!(c, colmat::c32, f32, i64, [64_], _64_, gbequb);
exports!(z, colmat::c64, f64, i64, [64_], _64_, gbequb);
