//! LAUUM layout conversion preserves selected complex entries without conjugation.
use super::{
    triangular::{nancheck, pack},
    LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR,
};
use crate::kernels::Scalar;
unsafe fn work<T: Scalar>(
    layout: i32,
    u: u8,
    n: i64,
    a: *mut T,
    ld: i64,
    mut call: impl FnMut(*mut T, i64) -> i64,
) -> i64 {
    if !matches!(layout, ROW_MAJOR | COL_MAJOR) {
        return -1;
    }
    let u = u.to_ascii_uppercase();
    if layout == ROW_MAJOR && ld < n {
        return -5;
    }
    if !matches!(u, b'L' | b'U') {
        return -2;
    }
    if n < 0 {
        return -3;
    }
    if layout == COL_MAJOR {
        if ld < n.max(1) {
            return -5;
        }
        return call(a, ld);
    }
    if n == 0 {
        return call(a, 1);
    }
    let nn = n as usize;
    let mut temp = match pack(u, b'N', nn, a, ld as usize) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let info = call(temp.as_mut_ptr(), n);
    if info >= 0 {
        for i in 0..nn {
            for j in 0..nn {
                if (u == b'L' && i >= j) || (u == b'U' && i <= j) {
                    *a.add(i * ld as usize + j) = temp[i + j * nn];
                }
            }
        }
    }
    info
}
macro_rules! exports {
    ($p:ident,$t:ty,$j:ty,[$($suffix:tt)*],$f:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p lauum_work $($suffix)*>](layout:i32,u:u8,n:$j,a:*mut $t,ld:$j)->$j {
            work(layout,u,n as i64,a,ld as i64,|a,ld|{let mut info:$j=0;crate::triangular_lapack::[<$p lauum $f>](&u,&n,a,&(ld as $j),&mut info);if info<0{(info-1) as i64}else{info as i64}}) as $j
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p lauum $($suffix)*>](layout:i32,u:u8,n:$j,a:*mut $t,ld:$j)->$j {
            if !matches!(layout,ROW_MAJOR|COL_MAJOR){return -1;}
            if LAPACKE_get_nancheck()!=0 {match nancheck(layout,u,b'N',n as i64,a,ld as i64){Ok(true)=>return -4,Err(e)=>return e as $j,Ok(false)=>{}}}
            [<LAPACKE_ $p lauum_work $($suffix)*>](layout,u,n,a,ld)
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
