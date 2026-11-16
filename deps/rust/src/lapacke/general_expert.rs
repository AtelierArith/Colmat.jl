//! GEEVX C adapters over the saved expert Fortran dispatch.
use super::{
    ge_nancheck, self_adjoint::zeros, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR,
    TRANSPOSE_MEMORY_ERROR,
};
use crate::kernels::Scalar;

fn span<T>(n: usize, ld: usize) -> Result<(), i64> {
    if n > 0
        && (n - 1)
            .checked_mul(ld)
            .and_then(|x| x.checked_add(n))
            .and_then(|x| x.checked_mul(core::mem::size_of::<T>()))
            .is_none_or(|x| x > isize::MAX as usize)
    {
        return Err(TRANSPOSE_MEMORY_ERROR);
    }
    Ok(())
}

unsafe fn work<T: Scalar>(
    layout: i32,
    bal: u8,
    jl: u8,
    jr: u8,
    sense: u8,
    n: i64,
    a: *mut T,
    lda: i64,
    vl: *mut T,
    ldvl: i64,
    vr: *mut T,
    ldvr: i64,
    workspace: *mut T,
    lwork: i64,
    mut call: impl FnMut(*mut T, i64, *mut T, i64, *mut T, i64) -> i64,
) -> i64 {
    if layout != ROW_MAJOR && layout != COL_MAJOR {
        return -1;
    }
    let (jl, jr) = (jl.to_ascii_uppercase(), jr.to_ascii_uppercase());
    let (bal, sense) = (bal.to_ascii_uppercase(), sense.to_ascii_uppercase());
    let off = if T::COMPLEX { 0 } else { 1 };
    if layout == ROW_MAJOR {
        if lda < n {
            return -8;
        }
        if ldvl < 1 || (jl == b'V' && ldvl < n) {
            return -11 - off;
        }
        if ldvr < 1 || (jr == b'V' && ldvr < n) {
            return -13 - off;
        }
    }
    if !matches!(bal, b'N' | b'P' | b'S' | b'B') {
        return -2;
    }
    if !matches!(jl, b'N' | b'V') {
        return -3;
    }
    if !matches!(jr, b'N' | b'V') {
        return -4;
    }
    if !matches!(sense, b'N' | b'E' | b'V' | b'B')
        || (matches!(sense, b'E' | b'B') && (jl != b'V' || jr != b'V'))
    {
        return -5;
    }
    if n < 0 {
        return -6;
    }
    if layout == COL_MAJOR {
        if lda < n.max(1) {
            return -8;
        }
        if ldvl < 1 || (jl == b'V' && ldvl < n) {
            return -11 - off;
        }
        if ldvr < 1 || (jr == b'V' && ldvr < n) {
            return -13 - off;
        }
    }
    let wide = n as i128;
    let base = wide
        * if T::COMPLEX || (jl == b'N' && jr == b'N') {
            2
        } else {
            3
        };
    let minimum = if n == 0 {
        1
    } else if matches!(sense, b'V' | b'B') {
        base.max(wide * wide + wide * if T::COMPLEX { 2 } else { 6 })
    } else {
        base.max(1)
    };
    if (4i128 * n as i128) > isize::MAX as i128 {
        return TRANSPOSE_MEMORY_ERROR;
    }
    if lwork != -1 && (lwork as i128) < minimum {
        return -21 - off;
    }
    if layout == COL_MAJOR {
        return call(a, lda, vl, ldvl, vr, ldvr);
    }
    let ld = n.max(1);
    if lwork == -1 || n == 0 {
        return call(a, ld, vl, ld, vr, ld);
    }
    let nn = n as usize;
    if let Err(e) = span::<T>(nn, lda as usize) {
        return e;
    }
    if jl == b'V' {
        if let Err(e) = span::<T>(nn, ldvl as usize) {
            return e;
        }
    }
    if jr == b'V' {
        if let Err(e) = span::<T>(nn, ldvr as usize) {
            return e;
        }
    }
    let count = nn.checked_mul(nn).unwrap_or(usize::MAX);
    let mut at = match zeros::<T>(count) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let mut left = if jl == b'V' {
        match zeros::<T>(count) {
            Ok(v) => v,
            Err(e) => return e,
        }
    } else {
        Vec::new()
    };
    let mut right = if jr == b'V' {
        match zeros::<T>(count) {
            Ok(v) => v,
            Err(e) => return e,
        }
    } else {
        Vec::new()
    };
    for i in 0..nn {
        for j in 0..nn {
            at[i + j * nn] = *a.add(i * lda as usize + j);
        }
    }
    let info = call(
        at.as_mut_ptr(),
        ld,
        left.as_mut_ptr(),
        ld,
        right.as_mut_ptr(),
        ld,
    );
    if info < 0 {
        return info;
    }
    for i in 0..nn {
        for j in 0..nn {
            *a.add(i * lda as usize + j) = at[i + j * nn];
            if jl == b'V' {
                *vl.add(i * ldvl as usize + j) = left[i + j * nn];
            }
            if jr == b'V' {
                *vr.add(i * ldvr as usize + j) = right[i + j * nn];
            }
        }
    }
    let _ = workspace;
    info
}

macro_rules! exports {
    ($p:ident,$t:ty,$r:ty,$j:ty,[$($suffix:tt)*],$f:tt,($($wi:ident:$wity:ty,)*),$aux:ty)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p geevx_work $($suffix)*>](
            layout:i32,bal:u8,jl:u8,jr:u8,sense:u8,n:$j,a:*mut $t,lda:$j,w:*mut $t,$($wi:$wity,)*
            vl:*mut $t,ldvl:$j,vr:*mut $t,ldvr:$j,ilo:*mut $j,ihi:*mut $j,scale:*mut $r,
            abnrm:*mut $r,rconde:*mut $r,rcondv:*mut $r,workspace:*mut $t,lwork:$j,aux:*mut $aux,
        )->$j {
            work(layout,bal,jl,jr,sense,n as i64,a,lda as i64,vl,ldvl as i64,vr,ldvr as i64,workspace,lwork as i64,
                |a,la,vl,ll,vr,lr|{
                    let mut info:$j=0;
                    crate::lapack::[<$p geevx $f>](&bal,&jl,&jr,&sense,&n,a,&(la as $j),w,$($wi,)*vl,&(ll as $j),vr,&(lr as $j),ilo,ihi,scale,abnrm,rconde,rcondv,workspace,&lwork,aux,&mut info);
                    if info<0 {(info-1) as i64} else {info as i64}
                }) as $j
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p geevx $($suffix)*>](
            layout:i32,bal:u8,jl:u8,jr:u8,sense:u8,n:$j,a:*mut $t,lda:$j,w:*mut $t,$($wi:$wity,)*
            vl:*mut $t,ldvl:$j,vr:*mut $t,ldvr:$j,ilo:*mut $j,ihi:*mut $j,scale:*mut $r,
            abnrm:*mut $r,rconde:*mut $r,rcondv:*mut $r,
        )->$j {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                match ge_nancheck(layout,n as i64,n as i64,a,lda as i64) {Ok(true)=>return -7,Err(e)=>return e as $j,Ok(false)=>{}}
            }
            let count=if <$t>::COMPLEX {(2i128*(n as i128).max(0)).max(1)} else if matches!(sense.to_ascii_uppercase(),b'V'|b'B') {(2i128*(n as i128).max(0)-2).max(1)} else {1};
            if count>isize::MAX as i128/core::mem::size_of::<$aux>() as i128 {return -1010;}
            let mut aux=Vec::<$aux>::new();
            if aux.try_reserve_exact(count as usize).is_err() {return -1010;}
            aux.resize(count as usize,0 as $aux);
            let mut query=<$t>::z();
            let info=[<LAPACKE_ $p geevx_work $($suffix)*>](layout,bal,jl,jr,sense,n,a,lda,w,$($wi,)*vl,ldvl,vr,ldvr,ilo,ihi,scale,abnrm,rconde,rcondv,&mut query,-1,aux.as_mut_ptr());
            if info!=0 {return info;}
            let length=query.re();
            if !length.is_finite() || length<1. || length>=isize::MAX as f64 || length > <$j>::MAX as f64 {return -1010;}
            let mut workspace=match zeros::<$t>(length as usize) {Ok(v)=>v,Err(_)=>return -1010};
            [<LAPACKE_ $p geevx_work $($suffix)*>](layout,bal,jl,jr,sense,n,a,lda,w,$($wi,)*vl,ldvl,vr,ldvr,ilo,ihi,scale,abnrm,rconde,rcondv,workspace.as_mut_ptr(),length as $j,aux.as_mut_ptr())
        }
    }};
}
exports!(s,f32,f32,i32,[],_,(wi:*mut f32,),i32);
exports!(d,f64,f64,i32,[],_,(wi:*mut f64,),i32);
exports!(c, colmat::c32, f32, i32, [], _, (), f32);
exports!(z, colmat::c64, f64, i32, [], _, (), f64);
exports!(s,f32,f32,i64,[64_],_64_,(wi:*mut f32,),i64);
exports!(d,f64,f64,i64,[64_],_64_,(wi:*mut f64,),i64);
exports!(c, colmat::c32, f32, i64, [64_], _64_, (), f32);
exports!(z, colmat::c64, f64, i64, [64_], _64_, (), f64);

#[cfg(test)]
mod tests {
    use super::*;
    fn layout_contract<T: Scalar>() {
        for jl in [b'N', b'V'] {
            for jr in [b'N', b'V'] {
                for info in [-4, 0, 1] {
                    let sentinel = T::real(777.);
                    let mut a = vec![sentinel; 15];
                    for i in 0..3 {
                        for j in 0..3 {
                            a[i * 5 + j] = T::real((10 * i + j) as f64);
                        }
                    }
                    let saved = a.clone();
                    let mut left = vec![sentinel; 15];
                    let mut right = left.clone();
                    let result = unsafe {
                        work(
                            ROW_MAJOR,
                            b'B',
                            jl,
                            jr,
                            b'N',
                            3,
                            a.as_mut_ptr(),
                            5,
                            left.as_mut_ptr(),
                            5,
                            right.as_mut_ptr(),
                            5,
                            core::ptr::null_mut(),
                            32,
                            |a, la, vl, ll, vr, lr| {
                                assert_eq!((la, ll, lr), (3, 3, 3));
                                for i in 0..3 {
                                    for j in 0..3 {
                                        assert!(*a.add(i + j * 3) == T::real((10 * i + j) as f64));
                                        *a.add(i + j * 3) = T::real((100 + 10 * i + j) as f64);
                                        if jl == b'V' {
                                            *vl.add(i + j * 3) = T::real((200 + 10 * i + j) as f64);
                                        }
                                        if jr == b'V' {
                                            *vr.add(i + j * 3) = T::real((300 + 10 * i + j) as f64);
                                        }
                                    }
                                }
                                info
                            },
                        )
                    };
                    assert_eq!(result, info);
                    for i in 0..3 {
                        for j in 0..5 {
                            assert!(
                                a[i * 5 + j]
                                    == if info < 0 || j >= 3 {
                                        saved[i * 5 + j]
                                    } else {
                                        T::real((100 + 10 * i + j) as f64)
                                    }
                            );
                            assert!(
                                left[i * 5 + j]
                                    == if info < 0 || j >= 3 || jl == b'N' {
                                        sentinel
                                    } else {
                                        T::real((200 + 10 * i + j) as f64)
                                    }
                            );
                            assert!(
                                right[i * 5 + j]
                                    == if info < 0 || j >= 3 || jr == b'N' {
                                        sentinel
                                    } else {
                                        T::real((300 + 10 * i + j) as f64)
                                    }
                            );
                        }
                    }
                }
            }
        }
    }
    fn headers<T: Scalar>() {
        for layout in [ROW_MAJOR, COL_MAJOR] {
            for sense in [b'N', b'E', b'V', b'B'] {
                for jl in [b'N', b'V'] {
                    for jr in [b'N', b'V'] {
                        let legal = !matches!(sense, b'E' | b'B') || (jl == b'V' && jr == b'V');
                        let minimum = if matches!(sense, b'V' | b'B') {
                            if T::COMPLEX {
                                15
                            } else {
                                27
                            }
                        } else if T::COMPLEX || (jl == b'N' && jr == b'N') {
                            6
                        } else {
                            9
                        };
                        for length in [-1, minimum - 1, minimum] {
                            let mut called = false;
                            let mut a = vec![T::o(); 9];
                            let mut vl = vec![T::z(); 9];
                            let mut vr = vl.clone();
                            let result = unsafe {
                                work::<T>(
                                    layout,
                                    b'b',
                                    jl,
                                    jr,
                                    sense,
                                    3,
                                    a.as_mut_ptr(),
                                    3,
                                    vl.as_mut_ptr(),
                                    3,
                                    vr.as_mut_ptr(),
                                    3,
                                    core::ptr::null_mut(),
                                    length,
                                    |_, _, _, _, _, _| {
                                        called = true;
                                        0
                                    },
                                )
                            };
                            assert_eq!(
                                result,
                                if !legal {
                                    -5
                                } else if length != -1 && length < minimum {
                                    if T::COMPLEX {
                                        -21
                                    } else {
                                        -22
                                    }
                                } else {
                                    0
                                }
                            );
                            assert_eq!(called, legal && (length == -1 || length >= minimum));
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn row_copy_real() {
        layout_contract::<f32>();
        layout_contract::<f64>();
    }
    #[test]
    fn row_copy_complex() {
        layout_contract::<colmat::c32>();
        layout_contract::<colmat::c64>();
    }
    #[test]
    fn workspace_real() {
        headers::<f32>();
        headers::<f64>();
    }
    #[test]
    fn workspace_complex() {
        headers::<colmat::c32>();
        headers::<colmat::c64>();
    }
}
