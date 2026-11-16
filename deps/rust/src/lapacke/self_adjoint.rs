//! SYEV/HEEV layout adapters over the existing Fortran numerical contract.
use super::{
    cholesky::po_nancheck, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR, TRANSPOSE_MEMORY_ERROR,
};
use crate::kernels::Scalar;

fn zeros<T: Scalar>(count: usize) -> Result<Vec<T>, i64> {
    if count
        .checked_mul(core::mem::size_of::<T>())
        .is_none_or(|n| n > isize::MAX as usize)
    {
        return Err(TRANSPOSE_MEMORY_ERROR);
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| TRANSPOSE_MEMORY_ERROR)?;
    values.resize(count, T::z());
    Ok(values)
}

unsafe fn work<T: Scalar>(
    layout: i32,
    job: u8,
    uplo: u8,
    n: i64,
    a: *mut T,
    lda: i64,
    workspace: *mut T,
    lwork: i64,
    call: impl FnMut(*mut T, i64, *mut T, i64) -> i64,
) -> i64 {
    layout_work(
        layout,
        job,
        uplo,
        n,
        a,
        lda,
        workspace,
        lwork,
        lwork == -1,
        || {
            let minimum = ((if T::COMPLEX { 2i128 } else { 3i128 }) * n as i128 - 1).max(1);
            if minimum > isize::MAX as i128 {
                return Err(TRANSPOSE_MEMORY_ERROR);
            }
            if lwork != -1 && (lwork as i128) < minimum {
                return Err(-9);
            }
            Ok(())
        },
        call,
    )
}

unsafe fn layout_work<T: Scalar>(
    layout: i32,
    job: u8,
    uplo: u8,
    n: i64,
    a: *mut T,
    lda: i64,
    workspace: *mut T,
    lwork: i64,
    query: bool,
    validate_workspace: impl FnOnce() -> Result<(), i64>,
    mut call: impl FnMut(*mut T, i64, *mut T, i64) -> i64,
) -> i64 {
    if layout != ROW_MAJOR && layout != COL_MAJOR {
        return -1;
    }
    if layout == ROW_MAJOR && lda < n {
        return -6;
    }
    let job = job.to_ascii_uppercase();
    let uplo = uplo.to_ascii_uppercase();
    if !matches!(job, b'N' | b'V') {
        return -2;
    }
    if !matches!(uplo, b'U' | b'L') {
        return -3;
    }
    if n < 0 {
        return -4;
    }
    if layout == COL_MAJOR && lda < n.max(1) {
        return -6;
    }
    if let Err(info) = validate_workspace() {
        return info;
    }
    if layout == COL_MAJOR {
        return call(a, lda, workspace, lwork);
    }
    let ld = n.max(1);
    if query {
        return call(a, ld, workspace, lwork);
    }
    let nn = n as usize;
    let mut packed = match zeros::<T>((ld as usize).checked_mul(ld as usize).unwrap_or(usize::MAX))
    {
        Ok(v) => v,
        Err(e) => return e,
    };
    if nn > 0
        && (nn - 1)
            .checked_mul(lda as usize)
            .and_then(|x| x.checked_add(nn))
            .and_then(|x| x.checked_mul(core::mem::size_of::<T>()))
            .is_none_or(|x| x > isize::MAX as usize)
    {
        return TRANSPOSE_MEMORY_ERROR;
    }
    for i in 0..nn {
        for j in 0..nn {
            if (uplo == b'L' && i >= j) || (uplo == b'U' && i <= j) {
                packed[i + j * ld as usize] = *a.add(i * lda as usize + j);
            }
        }
    }
    let info = call(packed.as_mut_ptr(), ld, workspace, lwork);
    for i in 0..nn {
        for j in 0..nn {
            if job == b'V' || (uplo == b'L' && i >= j) || (uplo == b'U' && i <= j) {
                *a.add(i * lda as usize + j) = packed[i + j * ld as usize];
            }
        }
    }
    info
}

macro_rules! exports {
    ($p:ident, $family:ident, $t:ty, $r:ty, $i:ty, [$($suffix:tt)*], $f:tt, [$($extra:ident)*]) => { paste::paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $family _work $($suffix)*>](
            layout: i32, job: u8, uplo: u8, n: $i, a: *mut $t, lda: $i, values: *mut $r,
            workspace: *mut $t, lwork: $i, $($extra: *mut $r,)*
        ) -> $i {
            work(layout, job, uplo, n as i64, a, lda as i64, workspace, lwork as i64,
                |a, ld, w, lw| {
                    let mut info: $i = 0;
                    crate::lapack_symmetric::[<$p $family $f>](&job, &uplo, &n, a, &(ld as $i),
                        values, w, &(lw as $i), $($extra,)* &mut info);
                    if info < 0 { (info-1) as i64 } else { info as i64 }
                }) as $i
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $family $($suffix)*>](
            layout: i32, job: u8, uplo: u8, n: $i, a: *mut $t, lda: $i, values: *mut $r,
        ) -> $i {
            if layout != ROW_MAJOR && layout != COL_MAJOR { return -1; }
            if LAPACKE_get_nancheck() != 0 {
                match po_nancheck(layout,uplo,n as i64,a,lda as i64) {
                    Ok(true) => return -5, Err(e) => return e as $i, Ok(false) => {},
                }
            }
            let nr = if <$t>::COMPLEX { (3i128*(n as i128)-2).max(1) } else { 1 };
            if nr > isize::MAX as i128 { return -1010; }
            let mut _rw = match zeros::<$r>(nr as usize) { Ok(v) => v, Err(_) => return -1010 };
            $(let $extra = _rw.as_mut_ptr();)*
            let mut query = <$t>::z();
            let info = [<LAPACKE_ $p $family _work $($suffix)*>](layout,job,uplo,n,a,lda,values,&mut query,-1,$($extra,)*);
            if info != 0 { return info; }
            let size = query.re();
            if !size.is_finite() || size < 1.0 || size >= isize::MAX as f64 || size > <$i>::MAX as f64 { return -1010; }
            let mut w = match zeros::<$t>(size as usize) { Ok(v) => v, Err(_) => return -1010 };
            [<LAPACKE_ $p $family _work $($suffix)*>](layout,job,uplo,n,a,lda,values,w.as_mut_ptr(),size as $i,$($extra,)*)
        }
    }};
}
exports!(s, syev, f32, f32, i32, [], _, []);
exports!(d, syev, f64, f64, i32, [], _, []);
exports!(c, heev, colmat::c32, f32, i32, [], _, [rwork]);
exports!(z, heev, colmat::c64, f64, i32, [], _, [rwork]);
exports!(s, syev, f32, f32, i64, [64_], _64_, []);
exports!(d, syev, f64, f64, i64, [64_], _64_, []);
exports!(c, heev, colmat::c32, f32, i64, [64_], _64_, [rwork]);
exports!(z, heev, colmat::c64, f64, i64, [64_], _64_, [rwork]);

fn dd_minima<T: Scalar>(job: u8, n: i64) -> Result<(i64, i64, i64), i64> {
    if n <= 1 {
        return Ok((1, 1, 1));
    }
    let n = n as u128;
    let vectors = job.to_ascii_uppercase() == b'V';
    let w = if T::COMPLEX {
        if vectors {
            2 * n + n * n
        } else {
            n + 1
        }
    } else if vectors {
        1 + 6 * n + 2 * n * n
    } else {
        2 * n + 1
    };
    let rw = if !T::COMPLEX {
        1
    } else if vectors {
        1 + 5 * n + 2 * n * n
    } else {
        n
    };
    let iw = if vectors { 3 + 5 * n } else { 1 };
    if [w, rw, iw].iter().any(|&v| v > isize::MAX as u128) {
        return Err(TRANSPOSE_MEMORY_ERROR);
    }
    Ok((w as i64, rw as i64, iw as i64))
}

macro_rules! dd_exports {
    ($p:ident,$family:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$f:tt,[$($extra:ident)*]) => { paste::paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $family d_work $($suffix)*>](
            layout:i32,job:u8,uplo:u8,n:$i,a:*mut $t,lda:$i,values:*mut $r,
            workspace:*mut $t,lwork:$i,$($extra:*mut $r,[<l $extra>]:$i,)*iwork:*mut $i,liwork:$i,
        )-> $i {
            let query = lwork == -1 || liwork == -1 $(|| [<l $extra>] == -1)*;
            layout_work(layout,job,uplo,n as i64,a,lda as i64,workspace,lwork as i64,query,
                || {
                    let (nw,_nr,ni)=dd_minima::<$t>(job,n as i64)?;
                    if ni > <$i>::MAX as i64 { return Err(TRANSPOSE_MEMORY_ERROR); }
                    if !query {
                        if (lwork as i64) < nw { return Err(-9); }
                        $(if ([<l $extra>] as i64) < _nr { return Err(-11); })*
                        if (liwork as i64) < ni { return Err(if <$t>::COMPLEX {-13} else {-11}); }
                    }
                    Ok(())
                },
                |a,ld,w,lw| {
                    let mut info:$i=0;
                    crate::lapack_symmetric::[<$p $family d $f>](&job,&uplo,&n,a,&(ld as $i),values,
                        w,&(lw as $i),$($extra,&[<l $extra>],)*iwork,&liwork,&mut info);
                    if info < 0 {(info-1) as i64} else {info as i64}
                }) as $i
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $family d $($suffix)*>](
            layout:i32,job:u8,uplo:u8,n:$i,a:*mut $t,lda:$i,values:*mut $r,
        )-> $i {
            if layout != ROW_MAJOR && layout != COL_MAJOR { return -1; }
            if LAPACKE_get_nancheck()!=0 {
                match po_nancheck(layout,uplo,n as i64,a,lda as i64) {
                    Ok(true)=>return -5,Err(e)=>return e as $i,Ok(false)=>{},
                }
            }
            let mut wq=<$t>::z();let mut _rq=0 as $r;let mut iq=0 as $i;
            $(let $extra=&mut _rq as *mut $r;let [<l $extra>]:$i=-1;)*
            let info=[<LAPACKE_ $p $family d_work $($suffix)*>](layout,job,uplo,n,a,lda,values,&mut wq,-1,$($extra,[<l $extra>],)*&mut iq,-1);
            if info!=0 {return info;}
            let nw=wq.re();let nr=if <$t>::COMPLEX {_rq as f64} else {1.0};
            if [nw,nr].iter().any(|&v| !v.is_finite() || v<1.0 || v>=isize::MAX as f64 || v > <$i>::MAX as f64) || iq<1 {return -1010;}
            let mut w=match zeros::<$t>(nw as usize) {Ok(v)=>v,Err(_)=>return -1010};
            let mut _rw=match zeros::<$r>(nr as usize) {Ok(v)=>v,Err(_)=>return -1010};
            let mut iw=Vec::<$i>::new();
            if (iq as usize).checked_mul(core::mem::size_of::<$i>()).is_none_or(|v|v>isize::MAX as usize) || iw.try_reserve_exact(iq as usize).is_err() {return -1010;}
            iw.resize(iq as usize,0);
            $(let $extra=_rw.as_mut_ptr();let [<l $extra>]=nr as $i;)*
            [<LAPACKE_ $p $family d_work $($suffix)*>](layout,job,uplo,n,a,lda,values,w.as_mut_ptr(),nw as $i,$($extra,[<l $extra>],)*iw.as_mut_ptr(),iq)
        }
    }};
}
dd_exports!(s, syev, f32, f32, i32, [], _, []);
dd_exports!(d, syev, f64, f64, i32, [], _, []);
dd_exports!(c, heev, colmat::c32, f32, i32, [], _, [rwork]);
dd_exports!(z, heev, colmat::c64, f64, i32, [], _, [rwork]);
dd_exports!(s, syev, f32, f32, i64, [64_], _64_, []);
dd_exports!(d, syev, f64, f64, i64, [64_], _64_, []);
dd_exports!(c, heev, colmat::c32, f32, i64, [64_], _64_, [rwork]);
dd_exports!(z, heev, colmat::c64, f64, i64, [64_], _64_, [rwork]);

#[cfg(test)]
mod workspace_order_regression {
    macro_rules! check {
        ($test:ident,$p:ident,$t:ty,$r:ty,$i:ty,$suffix:tt) => { paste::paste! {
            #[test]
            fn $test() { unsafe {
                for (lw,lr,li,dd_info,range_info) in [
                    (0,0,0,-8,-18),(32,0,0,-10,-20),(32,128,0,-12,-22),
                    (32,0,64,-10,-20),(0,-1,0,0,0),(0,0,-1,0,0),
                ] {
                    let mut work=vec![<$t as crate::kernels::Scalar>::z();32];
                    let mut rw=vec![0 as $r;128];let mut iw=vec![0 as $i;64];
                    let mut info: $i=0;let mut count: $i=0;
                    crate::lapack_symmetric::[<$p heevd $suffix>](
                        &b'V',&b'L',&3,core::ptr::null_mut(),&3,core::ptr::null_mut(),
                        work.as_mut_ptr(),&lw,rw.as_mut_ptr(),&lr,iw.as_mut_ptr(),&li,&mut info);
                    assert_eq!(info,dd_info);
                    if info==0 {
                        assert_eq!(crate::kernels::Scalar::re(work[0]),15.0);
                        assert_eq!(rw[0],34 as $r);assert_eq!(iw[0],18);
                    }
                    crate::lapack_symmetric::[<$p heevr $suffix>](
                        &b'V',&b'A',&b'L',&3,core::ptr::null_mut(),&3,&(0 as $r),&(0 as $r),
                        &1,&3,&(0 as $r),&mut count,core::ptr::null_mut(),core::ptr::null_mut(),&3,
                        core::ptr::null_mut(),work.as_mut_ptr(),&lw,rw.as_mut_ptr(),&lr,iw.as_mut_ptr(),&li,&mut info);
                    assert_eq!(info,range_info);
                    if info==0 {
                        assert_eq!(crate::kernels::Scalar::re(work[0]),6.0);
                        assert_eq!(rw[0],72 as $r);assert_eq!(iw[0],30);
                    }
                }
            } }
        }};
    }
    check!(complex32_lp64, c, colmat::c32, f32, i32, _);
    check!(complex64_lp64, z, colmat::c64, f64, i32, _);
    check!(complex32_ilp64, c, colmat::c32, f32, i64, _64_);
    check!(complex64_ilp64, z, colmat::c64, f64, i64, _64_);
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe fn numerical<T: Scalar, R: Scalar>(
        off: T,
        tolerance: f64,
        mut call: impl FnMut(bool, i32, u8, u8, i64, *mut T, i64, *mut R, *mut T, i64) -> i64,
    ) {
        for layout in [ROW_MAJOR, COL_MAJOR] {
            for job in [b'N', b'V'] {
                for uplo in [b'L', b'U'] {
                    for n in [0usize, 1, 2] {
                        for high in [false, true] {
                            let ld = n + 2;
                            let index = |i: usize, j: usize| {
                                if layout == ROW_MAJOR {
                                    i * ld + j
                                } else {
                                    i + j * ld
                                }
                            };
                            let mut a = vec![T::real(f64::NAN); ld * (n + 2)];
                            let logical = |i, j| {
                                if i == j {
                                    T::real(if i == 0 { 4.0 } else { 7.0 })
                                } else if i < j {
                                    off
                                } else {
                                    off.cj()
                                }
                            };
                            for i in 0..n {
                                for j in 0..n {
                                    if (uplo == b'L' && i >= j) || (uplo == b'U' && i <= j) {
                                        a[index(i, j)] = logical(i, j);
                                    }
                                }
                            }
                            let bytes = |v: &[T]| {
                                core::slice::from_raw_parts(
                                    v.as_ptr() as *const u8,
                                    core::mem::size_of_val(v),
                                )
                                .to_vec()
                            };
                            let before = bytes(&a);
                            let mut values = vec![R::real(-99.0); n + 2];
                            let mut workspace = vec![T::z(); 1];
                            if !high {
                                assert_eq!(
                                    call(
                                        false,
                                        layout,
                                        job,
                                        uplo,
                                        n as i64,
                                        a.as_mut_ptr(),
                                        ld as i64,
                                        values.as_mut_ptr(),
                                        workspace.as_mut_ptr(),
                                        -1
                                    ),
                                    0
                                );
                                assert_eq!(bytes(&a), before, "query touched matrix");
                                workspace.resize(workspace[0].re() as usize, T::z());
                            }
                            assert_eq!(
                                call(
                                    high,
                                    layout,
                                    job,
                                    uplo,
                                    n as i64,
                                    a.as_mut_ptr(),
                                    ld as i64,
                                    values.as_mut_ptr(),
                                    workspace.as_mut_ptr(),
                                    workspace.len() as i64
                                ),
                                0
                            );
                            for k in 0..n {
                                let expected = if n == 1 {
                                    4.0
                                } else {
                                    (11.0
                                        + if k == 0 { -1.0 } else { 1.0 }
                                            * (9.0 + 4.0 * off.magnitude().powi(2)).sqrt())
                                        / 2.0
                                };
                                assert!((values[k].re() - expected).abs() < tolerance * 10.0);
                            }
                            assert_eq!(values[n].re(), -99.0);
                            assert_eq!(values[n + 1].re(), -99.0);
                            if job == b'V' {
                                for i in 0..n {
                                    for k in 0..n {
                                        let mut product = T::z();
                                        for j in 0..n {
                                            product = product + logical(i, j) * a[index(j, k)];
                                        }
                                        assert!(
                                            (product - a[index(i, k)] * T::real(values[k].re()))
                                                .magnitude()
                                                < tolerance * 30.0
                                        );
                                    }
                                }
                                for i in 0..n {
                                    for j in 0..n {
                                        let mut dot = T::z();
                                        for k in 0..n {
                                            dot = dot + a[index(k, i)].cj() * a[index(k, j)];
                                        }
                                        assert!(
                                            (dot - T::real(if i == j { 1.0 } else { 0.0 }))
                                                .magnitude()
                                                < tolerance * 10.0
                                        );
                                    }
                                }
                            }
                            for q in 0..a.len() {
                                let written = (0..n).any(|i| {
                                    (0..n).any(|j| {
                                        index(i, j) == q
                                            && (job == b'V'
                                                || (uplo == b'L' && i >= j)
                                                || (uplo == b'U' && i <= j))
                                    })
                                });
                                if !written {
                                    assert!(a[q] != a[q], "unused storage changed");
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    macro_rules! check {
        ($test:ident,$p:ident,$family:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],[$($extra:ident)*],$off:expr,$tol:expr) => { paste::paste! {
            #[test]
            fn $test() { unsafe {
                numerical::<$t,$r>($off,$tol,|high,layout,job,uplo,n,a,ld,values,w,lw| {
                    let mut _rw=vec![0 as $r;32];$(let $extra=_rw.as_mut_ptr();)*
                    if high { [<LAPACKE_ $p $family $($suffix)*>](layout,job,uplo,n as $i,a,ld as $i,values) as i64 }
                    else { [<LAPACKE_ $p $family _work $($suffix)*>](layout,job,uplo,n as $i,a,ld as $i,values,w,lw as $i,$($extra,)*) as i64 }
                });
            } }
        }};
    }
    check!(real32_lp64, s, syev, f32, f32, i32, [], [], 1.0, 1e-5);
    check!(real64_lp64, d, syev, f64, f64, i32, [], [], 1.0, 1e-12);
    check!(
        complex32_lp64,
        c,
        heev,
        colmat::c32,
        f32,
        i32,
        [],
        [rw],
        colmat::c32::new(1.0, 2.0),
        1e-5
    );
    check!(
        complex64_lp64,
        z,
        heev,
        colmat::c64,
        f64,
        i32,
        [],
        [rw],
        colmat::c64::new(1.0, 2.0),
        1e-12
    );
    check!(real32_ilp64, s, syev, f32, f32, i64, [64_], [], 1.0, 1e-5);
    check!(real64_ilp64, d, syev, f64, f64, i64, [64_], [], 1.0, 1e-12);
    check!(
        complex32_ilp64,
        c,
        heev,
        colmat::c32,
        f32,
        i64,
        [64_],
        [rw],
        colmat::c32::new(1.0, 2.0),
        1e-5
    );
    check!(
        complex64_ilp64,
        z,
        heev,
        colmat::c64,
        f64,
        i64,
        [64_],
        [rw],
        colmat::c64::new(1.0, 2.0),
        1e-12
    );

    macro_rules! check_dd {
        ($test:ident,$p:ident,$family:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],[$($extra:ident)*],$off:expr,$tol:expr) => { paste::paste! {
            #[test]
            fn $test() { unsafe {
                numerical::<$t,$r>($off,$tol,|high,layout,job,uplo,n,a,ld,values,w,lw| {
                    let mut _rw=vec![0 as $r;128];let mut iw=vec![0 as $i;64];
                    $(let $extra=_rw.as_mut_ptr();let [<l $extra>]=128;)*
                    if high { [<LAPACKE_ $p $family d $($suffix)*>](layout,job,uplo,n as $i,a,ld as $i,values) as i64 }
                    else { [<LAPACKE_ $p $family d_work $($suffix)*>](layout,job,uplo,n as $i,a,ld as $i,values,w,lw as $i,$($extra,[<l $extra>],)*iw.as_mut_ptr(),64) as i64 }
                });
            } }
        }};
    }
    check_dd!(dd_real32_lp64, s, syev, f32, f32, i32, [], [], 1.0, 1e-5);
    check_dd!(dd_real64_lp64, d, syev, f64, f64, i32, [], [], 1.0, 1e-12);
    check_dd!(
        dd_complex32_lp64,
        c,
        heev,
        colmat::c32,
        f32,
        i32,
        [],
        [rw],
        colmat::c32::new(1.0, 2.0),
        1e-5
    );
    check_dd!(
        dd_complex64_lp64,
        z,
        heev,
        colmat::c64,
        f64,
        i32,
        [],
        [rw],
        colmat::c64::new(1.0, 2.0),
        1e-12
    );
    check_dd!(
        dd_real32_ilp64,
        s,
        syev,
        f32,
        f32,
        i64,
        [64_],
        [],
        1.0,
        1e-5
    );
    check_dd!(
        dd_real64_ilp64,
        d,
        syev,
        f64,
        f64,
        i64,
        [64_],
        [],
        1.0,
        1e-12
    );
    check_dd!(
        dd_complex32_ilp64,
        c,
        heev,
        colmat::c32,
        f32,
        i64,
        [64_],
        [rw],
        colmat::c32::new(1.0, 2.0),
        1e-5
    );
    check_dd!(
        dd_complex64_ilp64,
        z,
        heev,
        colmat::c64,
        f64,
        i64,
        [64_],
        [rw],
        colmat::c64::new(1.0, 2.0),
        1e-12
    );
}
