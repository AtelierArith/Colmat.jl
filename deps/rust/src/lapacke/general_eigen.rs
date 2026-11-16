//! GEEV C adapters. Real conjugate pairs retain LAPACK's adjacent-column format.
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
    jl: u8,
    jr: u8,
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
    let off = if T::COMPLEX { 0 } else { 1 };
    if layout == ROW_MAJOR {
        if lda < n {
            return -6;
        }
        if ldvl < 1 || (jl == b'V' && ldvl < n) {
            return -9 - off;
        }
        if ldvr < 1 || (jr == b'V' && ldvr < n) {
            return -11 - off;
        }
    }
    if !matches!(jl, b'N' | b'V') {
        return -2;
    }
    if !matches!(jr, b'N' | b'V') {
        return -3;
    }
    if n < 0 {
        return -4;
    }
    if layout == COL_MAJOR {
        if lda < n.max(1) {
            return -6;
        }
        if ldvl < 1 || (jl == b'V' && ldvl < n) {
            return -9 - off;
        }
        if ldvr < 1 || (jr == b'V' && ldvr < n) {
            return -11 - off;
        }
    }
    let factor = if T::COMPLEX {
        2i128
    } else if jl == b'V' || jr == b'V' {
        4
    } else {
        3
    };
    let minimum = (factor * n as i128).max(1);
    if (4i128 * n as i128) > isize::MAX as i128 {
        return TRANSPOSE_MEMORY_ERROR;
    }
    if lwork != -1 && (lwork as i128) < minimum {
        return -13 - off;
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
    ($p:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$f:tt,($($imag:ident:$imag_ty:ty,)*),[$($extra:ident)*])=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p geev_work $($suffix)*>](
            layout:i32,jl:u8,jr:u8,n:$i,a:*mut $t,lda:$i,values:*mut $t,$($imag:$imag_ty,)*
            vl:*mut $t,ldvl:$i,vr:*mut $t,ldvr:$i,workspace:*mut $t,lwork:$i,$($extra:*mut $r,)*
        )-> $i {
            work(layout,jl,jr,n as i64,a,lda as i64,vl,ldvl as i64,vr,ldvr as i64,workspace,lwork as i64,
                |a,la,vl,ll,vr,lr| {
                    let mut info:$i=0;
                    crate::lapack::[<$p geev $f>](&jl,&jr,&n,a,&(la as $i),values,$($imag,)*vl,&(ll as $i),vr,&(lr as $i),
                        workspace,&lwork,$($extra,)*&mut info);
                    if info<0 {(info-1) as i64} else {info as i64}
                }) as $i
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p geev $($suffix)*>](
            layout:i32,jl:u8,jr:u8,n:$i,a:*mut $t,lda:$i,values:*mut $t,$($imag:$imag_ty,)*
            vl:*mut $t,ldvl:$i,vr:*mut $t,ldvr:$i,
        )-> $i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                match ge_nancheck(layout,n as i64,n as i64,a,lda as i64) {Ok(true)=>return -5,Err(e)=>return e as $i,Ok(false)=>{}}
            }
            let nr=if <$t>::COMPLEX {(n as i128).max(0)*2} else {1};
            if nr>isize::MAX as i128 {return -1010;}
            let mut _rw=match zeros::<$r>((nr as usize).max(1)) {Ok(v)=>v,Err(_)=>return -1010};
            $(let $extra=_rw.as_mut_ptr();)*
            let mut query=<$t>::z();
            let info=[<LAPACKE_ $p geev_work $($suffix)*>](layout,jl,jr,n,a,lda,values,$($imag,)*vl,ldvl,vr,ldvr,&mut query,-1,$($extra,)*);
            if info!=0 {return info;}
            let nw=query.re();
            if !nw.is_finite() || nw<1.0 || nw>=isize::MAX as f64 || nw > <$i>::MAX as f64 {return -1010;}
            let mut w=match zeros::<$t>(nw as usize) {Ok(v)=>v,Err(_)=>return -1010};
            [<LAPACKE_ $p geev_work $($suffix)*>](layout,jl,jr,n,a,lda,values,$($imag,)*vl,ldvl,vr,ldvr,w.as_mut_ptr(),nw as $i,$($extra,)*)
        }
    }};
}
exports!(s,f32,f32,i32,[],_,(wi:*mut f32,),[]);
exports!(d,f64,f64,i32,[],_,(wi:*mut f64,),[]);
exports!(c, colmat::c32, f32, i32, [], _, (), [rwork]);
exports!(z, colmat::c64, f64, i32, [], _, (), [rwork]);
exports!(s,f32,f32,i64,[64_],_64_,(wi:*mut f32,),[]);
exports!(d,f64,f64,i64,[64_],_64_,(wi:*mut f64,),[]);
exports!(c, colmat::c32, f32, i64, [64_], _64_, (), [rwork]);
exports!(z, colmat::c64, f64, i64, [64_], _64_, (), [rwork]);

#[cfg(test)]
mod tests {
    use super::*;
    use colmat::c64;

    fn exercise<T: Scalar>(
        convert: impl Fn(T) -> c64,
        mut call: impl FnMut(
            bool,
            i32,
            u8,
            u8,
            i64,
            *mut T,
            i64,
            *mut T,
            *mut f64,
            *mut T,
            i64,
            *mut T,
            i64,
            *mut T,
            i64,
        ) -> i64,
        tolerance: f64,
    ) {
        for layout in [ROW_MAJOR, COL_MAJOR] {
            for n in [0i64, 1] {
                for high in [false, true] {
                    let sentinel = T::real(777.);
                    let mut a = vec![T::real(7.), sentinel, sentinel];
                    let mut values = vec![sentinel; 4];
                    let mut imag = vec![777.; 4];
                    let mut left = vec![sentinel; 3];
                    let mut right = left.clone();
                    let mut workspace = vec![T::z(); 4];
                    let minimum = if n == 0 {
                        1
                    } else if T::COMPLEX {
                        2
                    } else {
                        4
                    };
                    let info = call(
                        high,
                        layout,
                        b'V',
                        b'V',
                        n,
                        a.as_mut_ptr(),
                        3,
                        values.as_mut_ptr(),
                        imag.as_mut_ptr(),
                        left.as_mut_ptr(),
                        3,
                        right.as_mut_ptr(),
                        3,
                        workspace.as_mut_ptr(),
                        minimum,
                    );
                    assert_eq!(info, 0);
                    if n == 1 {
                        assert!((convert(values[0]) - c64::new(7., 0.)).magnitude() < tolerance);
                        assert!((left[0].magnitude() - 1.).abs() < tolerance);
                        assert!((right[0].magnitude() - 1.).abs() < tolerance);
                    }
                    for j in n as usize..3 {
                        assert_eq!(values[j], sentinel);
                        assert_eq!(left[j], sentinel);
                        assert_eq!(right[j], sentinel);
                    }
                    assert_eq!(a[1], sentinel);
                    assert_eq!(a[2], sentinel);
                }
            }
        }
        for layout in [ROW_MAJOR, COL_MAJOR] {
            for jl in [b'N', b'V'] {
                for jr in [b'N', b'V'] {
                    for high in [false, true] {
                        let n = 3usize;
                        let ld = 5usize;
                        let original = [[1., -2., 0.], [2., 1., 0.], [0., 0., 5.]];
                        let at = |i: usize, j: usize| {
                            if layout == ROW_MAJOR {
                                i * ld + j
                            } else {
                                i + j * ld
                            }
                        };
                        let sentinel = T::real(777.);
                        let mut a = vec![sentinel; ld * n];
                        for i in 0..n {
                            for j in 0..n {
                                a[at(i, j)] = T::real(original[i][j]);
                            }
                        }
                        let saved = a.clone();
                        let mut left = vec![sentinel; ld * n];
                        let mut right = left.clone();
                        let mut values = vec![sentinel; n + 1];
                        let mut imag = vec![777.; n + 1];
                        let minimum = if T::COMPLEX {
                            6
                        } else if jl == b'V' || jr == b'V' {
                            12
                        } else {
                            9
                        };
                        let mut workspace = vec![T::z(); 12];
                        macro_rules! invoke {
                            ($high:expr,$lw:expr) => {
                                call(
                                    $high,
                                    layout,
                                    jl,
                                    jr,
                                    3,
                                    a.as_mut_ptr(),
                                    5,
                                    values.as_mut_ptr(),
                                    imag.as_mut_ptr(),
                                    left.as_mut_ptr(),
                                    5,
                                    right.as_mut_ptr(),
                                    5,
                                    workspace.as_mut_ptr(),
                                    $lw,
                                )
                            };
                        }
                        assert_eq!(invoke!(false, -1), 0);
                        assert_eq!(a, saved);
                        assert!(values.iter().all(|&v| v == sentinel));
                        assert!(left.iter().chain(&right).all(|&v| v == sentinel));
                        assert!(workspace[0].re() >= minimum as f64);
                        assert_eq!(
                            invoke!(false, minimum - 1),
                            if T::COMPLEX { -13 } else { -14 }
                        );
                        assert_eq!(a, saved);
                        assert_eq!(invoke!(high, minimum), 0);
                        assert_eq!(values[n], sentinel);
                        assert_eq!(imag[n], 777.);
                        for k in 0..ld * n {
                            if k % ld >= n {
                                assert_eq!(a[k], saved[k]);
                                assert_eq!(left[k], sentinel);
                                assert_eq!(right[k], sentinel);
                            }
                        }
                        let lambda: Vec<c64> = (0..n)
                            .map(|j| {
                                if T::COMPLEX {
                                    convert(values[j])
                                } else {
                                    c64::new(values[j].re(), imag[j])
                                }
                            })
                            .collect();
                        for expected in [c64::new(1., 2.), c64::new(1., -2.), c64::new(5., 0.)] {
                            assert!(lambda
                                .iter()
                                .any(|&v| (v - expected).magnitude() < tolerance));
                        }
                        for (requested, vectors, is_left) in
                            [(jl, &left, true), (jr, &right, false)]
                        {
                            if requested == b'N' {
                                assert!(vectors.iter().all(|&v| v == sentinel));
                                continue;
                            }
                            for j in 0..n {
                                let vector: Vec<c64> = (0..n)
                                    .map(|i| {
                                        if T::COMPLEX || imag[j] == 0. {
                                            convert(vectors[at(i, j)])
                                        } else if imag[j] > 0. {
                                            c64::new(
                                                vectors[at(i, j)].re(),
                                                vectors[at(i, j + 1)].re(),
                                            )
                                        } else {
                                            c64::new(
                                                vectors[at(i, j - 1)].re(),
                                                -vectors[at(i, j)].re(),
                                            )
                                        }
                                    })
                                    .collect();
                                let norm: f64 = vector.iter().map(|v| v.magnitude().powi(2)).sum();
                                assert!((norm - 1.).abs() < tolerance * 10.);
                                let eigenvalue = if is_left { lambda[j].cj() } else { lambda[j] };
                                for i in 0..n {
                                    let product = (0..n).fold(c64::new(0., 0.), |sum, k| {
                                        sum + vector[k]
                                            * if is_left {
                                                original[k][i]
                                            } else {
                                                original[i][k]
                                            }
                                    });
                                    assert!(
                                        (product - vector[i] * eigenvalue).magnitude()
                                            < tolerance * 20.
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    macro_rules! case {
        ($name:ident,$p:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],[$($wi:ident)*],[$($rw:ident)*],$convert:expr,$tol:expr) => {
            #[test]
            fn $name() { paste::paste! {
                exercise::<$t>($convert, |high,layout,jl,jr,n,a,lda,w,imag,vl,ll,vr,lr,work,lwork| unsafe {
                    let mut _imag: Vec<$r> = vec![0.;3];
                    $(let $wi=_imag.as_mut_ptr();)*
                    let mut _real: Vec<$r> = vec![0.;6];
                    $(let $rw=_real.as_mut_ptr();)*
                    let info = if high {
                        [<LAPACKE_ $p geev $($suffix)*>](layout,jl,jr,n as $i,a,lda as $i,w,$($wi,)*vl,ll as $i,vr,lr as $i)
                    } else {
                        [<LAPACKE_ $p geev_work $($suffix)*>](layout,jl,jr,n as $i,a,lda as $i,w,$($wi,)*vl,ll as $i,vr,lr as $i,work,lwork as $i,$($rw,)*)
                    };
                    if !<$t>::COMPLEX {for j in 0..3 {*imag.add(j)=_imag[j] as f64;}}
                    info as i64
                },$tol);
            }}
        }
    }
    case!(
        s32,
        s,
        f32,
        f32,
        i32,
        [],
        [wi],
        [],
        |v| c64::new(v as f64, 0.),
        2e-5
    );
    case!(
        s64,
        s,
        f32,
        f32,
        i64,
        [64_],
        [wi],
        [],
        |v| c64::new(v as f64, 0.),
        2e-5
    );
    case!(
        d32,
        d,
        f64,
        f64,
        i32,
        [],
        [wi],
        [],
        |v| c64::new(v, 0.),
        1e-12
    );
    case!(
        d64,
        d,
        f64,
        f64,
        i64,
        [64_],
        [wi],
        [],
        |v| c64::new(v, 0.),
        1e-12
    );
    case!(
        c32,
        c,
        colmat::c32,
        f32,
        i32,
        [],
        [],
        [rw],
        |v: colmat::c32| c64::new(v.re as f64, v.im as f64),
        2e-5
    );
    case!(
        c64,
        c,
        colmat::c32,
        f32,
        i64,
        [64_],
        [],
        [rw],
        |v: colmat::c32| c64::new(v.re as f64, v.im as f64),
        2e-5
    );
    case!(z32, z, colmat::c64, f64, i32, [], [], [rw], |v| v, 1e-12);
    case!(z64, z, colmat::c64, f64, i64, [64_], [], [rw], |v| v, 1e-12);
}
