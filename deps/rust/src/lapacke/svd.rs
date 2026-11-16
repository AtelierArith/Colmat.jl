//! SVD layout adapters. Native driver scratch and algorithms remain backend-owned.
use super::{ge_nancheck, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR, TRANSPOSE_MEMORY_ERROR};
use crate::kernels::Scalar;

fn zeros<T: Scalar>(count: usize) -> Result<Vec<T>, i64> {
    if count
        .checked_mul(core::mem::size_of::<T>())
        .is_none_or(|n| n > isize::MAX as usize)
    {
        return Err(TRANSPOSE_MEMORY_ERROR);
    }
    let mut out = Vec::new();
    out.try_reserve_exact(count)
        .map_err(|_| TRANSPOSE_MEMORY_ERROR)?;
    out.resize(count, T::z());
    Ok(out)
}

// The current Fortran driver computes these expressions in isize before its
// argument checks. Reject an unrepresentable workspace before entering it.
fn representable_workspace<T: Scalar>(dd: bool, job: u8, m: i64, n: i64) -> bool {
    let r = m.min(n).max(0) as u128;
    let big = m.max(n).max(0) as u128;
    if dd && job.to_ascii_uppercase() != b'N' && r > 0 && r > isize::MAX as u128 / r {
        return false;
    }
    let size = if dd {
        if T::COMPLEX {
            match job.to_ascii_uppercase() {
                b'N' => 2 * r + big,
                b'O' => 2 * r * r + 2 * r + big,
                b'S' => r * r + 3 * r,
                _ => r * r + 2 * r + big,
            }
        } else {
            match job.to_ascii_uppercase() {
                b'N' => 3 * r + big.max(7 * r),
                b'O' => 3 * r + big.max(5 * r * r + 4 * r),
                b'S' => 4 * r * r + 7 * r,
                _ => 4 * r * r + 6 * r + big,
            }
        }
    } else if T::COMPLEX {
        2 * r + big
    } else {
        (3 * r + big).max(5 * r)
    };
    size <= isize::MAX as u128
}

unsafe fn high<T: Scalar, R: Scalar, I: Copy + Default>(
    dd: bool,
    layout: i32,
    job: u8,
    m: i64,
    n: i64,
    a: *mut T,
    lda: i64,
    superb: *mut R,
    mut call: impl FnMut(*mut T, i64, *mut R, *mut I) -> i64,
) -> i64 {
    if layout != ROW_MAJOR && layout != COL_MAJOR {
        return -1;
    }
    if LAPACKE_get_nancheck() != 0 {
        match ge_nancheck(layout, m, n, a, lda) {
            Ok(true) => return if dd { -5 } else { -6 },
            Err(e) => return e,
            Ok(false) => {}
        }
    }
    // These arrays are initialized even though the current native driver owns
    // its numerical scratch. In particular SUPERB must never expose allocator data.
    let r = m.min(n).max(0) as usize;
    let big = m.max(n).max(0) as usize;
    let nr = if !T::COMPLEX {
        Some(0)
    } else if !dd {
        r.checked_mul(5)
    } else if job.to_ascii_uppercase() == b'N' {
        r.checked_mul(7)
    } else {
        r.checked_mul(5)
            .and_then(|x| x.checked_add(7))
            .zip(
                big.checked_mul(2)
                    .and_then(|x| r.checked_mul(2).and_then(|y| x.checked_add(y)))
                    .and_then(|x| x.checked_add(1)),
            )
            .and_then(|(x, y)| r.checked_mul(x.max(y)))
    };
    let Some(nr) = nr else {
        return -1010;
    };
    let mut rw = match zeros::<R>(nr.max(1)) {
        Ok(v) => v,
        Err(_) => return -1010,
    };
    let Some(ni) = (if dd { r.checked_mul(8) } else { Some(0) }) else {
        return -1010;
    };
    if ni
        .checked_mul(core::mem::size_of::<I>())
        .is_none_or(|x| x > isize::MAX as usize)
    {
        return -1010;
    }
    let mut iw = Vec::<I>::new();
    if iw.try_reserve_exact(ni).is_err() {
        return -1010;
    }
    iw.resize(ni, I::default());
    let mut query = T::z();
    let info = call(&mut query, -1, rw.as_mut_ptr(), iw.as_mut_ptr());
    if info != 0 {
        return info;
    }
    let size = query.re();
    if !size.is_finite() || size < 1.0 || size >= isize::MAX as f64 {
        return -1010;
    }
    let mut w = match zeros::<T>(size as usize) {
        Ok(v) => v,
        Err(_) => return -1010,
    };
    let info = call(
        w.as_mut_ptr(),
        size as i64,
        rw.as_mut_ptr(),
        iw.as_mut_ptr(),
    );
    if !dd {
        for i in 0..r.saturating_sub(1) {
            *superb.add(i) = if T::COMPLEX {
                rw[i]
            } else {
                R::real(w[i + 1].re())
            };
        }
    }
    info
}

/// A query reaches the backend with the reference temporary leading dimensions,
/// without touching matrix data. Execution uses the same validation before packing.
unsafe fn work<T: Scalar>(
    dd: bool,
    layout: i32,
    ju: u8,
    jv: u8,
    m: i64,
    n: i64,
    a: *mut T,
    lda: i64,
    u: *mut T,
    ldu: i64,
    vt: *mut T,
    ldvt: i64,
    workspace: *mut T,
    lwork: i64,
    mut call: impl FnMut(*mut T, i64, *mut T, i64, *mut T, i64, *mut T, i64) -> i64,
) -> i64 {
    if layout != COL_MAJOR && layout != ROW_MAJOR {
        return -1;
    }
    if layout == COL_MAJOR {
        if !representable_workspace::<T>(dd, ju, m, n) {
            return TRANSPOSE_MEMORY_ERROR;
        }
        return call(a, lda, u, ldu, vt, ldvt, workspace, lwork);
    }
    let ju = ju.to_ascii_uppercase();
    let jv = jv.to_ascii_uppercase();
    let r = m.min(n);
    let use_u = matches!(ju, b'A' | b'S') || (dd && ju == b'O' && m < n);
    let use_v = if dd {
        matches!(ju, b'A' | b'S') || (ju == b'O' && m >= n)
    } else {
        matches!(jv, b'A' | b'S')
    };
    let uc = if ju == b'A' || (dd && ju == b'O' && m < n) {
        m
    } else if ju == b'S' {
        r
    } else {
        1
    };
    let vr = if (dd && (ju == b'A' || (ju == b'O' && m >= n))) || (!dd && jv == b'A') {
        n
    } else if (dd && ju == b'S') || (!dd && jv == b'S') {
        r
    } else {
        1
    };
    let vc = if dd || use_v { n } else { 1 };
    if lda < n {
        return if dd { -6 } else { -7 };
    }
    if ldu < uc {
        return if dd { -9 } else { -10 };
    }
    if ldvt < vc {
        return if dd { -11 } else { -12 };
    }
    if !representable_workspace::<T>(dd, ju, m, n) {
        return TRANSPOSE_MEMORY_ERROR;
    }
    let la = m.max(1);
    let lu = if use_u { m.max(1) } else { 1 };
    let lv = vr.max(1);
    if lwork == -1 {
        return call(a, la, u, lu, vt, lv, workspace, -1);
    }
    let mut query = T::z();
    let info = call(a, la, u, lu, vt, lv, &mut query, -1);
    if info != 0 {
        return info;
    }
    // The backend's query validates flags and dimensions. Validate LWORK before
    // allocation so an invalid call cannot touch the caller's matrix.
    if (lwork as f64) < query.re() {
        return if dd { -13 } else { -14 };
    }
    let mut at = match zeros::<T>(
        (la as usize)
            .checked_mul(n.max(1) as usize)
            .unwrap_or(usize::MAX),
    ) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let mut ut = match zeros::<T>(if use_u {
        (lu as usize)
            .checked_mul(uc.max(1) as usize)
            .unwrap_or(usize::MAX)
    } else {
        0
    }) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let mut vtt = match zeros::<T>(if use_v {
        (lv as usize)
            .checked_mul(n.max(1) as usize)
            .unwrap_or(usize::MAX)
    } else {
        0
    }) {
        Ok(v) => v,
        Err(e) => return e,
    };
    // Check the complete referenced row spans before pointer arithmetic.
    for (rows, cols, ld) in [
        (m, n, lda),
        (if use_u { m } else { 0 }, uc, ldu),
        (if use_v { vr } else { 0 }, n, ldvt),
    ] {
        if rows > 0
            && cols > 0
            && (rows as usize - 1)
                .checked_mul(ld as usize)
                .and_then(|x| x.checked_add(cols as usize))
                .and_then(|x| x.checked_mul(core::mem::size_of::<T>()))
                .is_none_or(|x| x > isize::MAX as usize)
        {
            return TRANSPOSE_MEMORY_ERROR;
        }
    }
    for i in 0..m as usize {
        for j in 0..n as usize {
            at[i + j * la as usize] = *a.add(i * lda as usize + j);
        }
    }
    let info = call(
        at.as_mut_ptr(),
        la,
        ut.as_mut_ptr(),
        lu,
        vtt.as_mut_ptr(),
        lv,
        workspace,
        lwork,
    );
    for i in 0..m as usize {
        for j in 0..n as usize {
            *a.add(i * lda as usize + j) = at[i + j * la as usize];
        }
    }
    if use_u {
        for i in 0..m as usize {
            for j in 0..uc as usize {
                *u.add(i * ldu as usize + j) = ut[i + j * lu as usize];
            }
        }
    }
    if use_v {
        for i in 0..vr as usize {
            for j in 0..n as usize {
                *vt.add(i * ldvt as usize + j) = vtt[i + j * lv as usize];
            }
        }
    }
    info
}

macro_rules! exports {
    ($p:ident, $t:ty, $r:ty, $i:ty, [$($suffix:tt)*], $f:tt, [$($extra:ident)*]) => { paste::paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p gesvd_work $($suffix)*>](
            layout: i32, ju: u8, jv: u8, m: $i, n: $i, a: *mut $t, lda: $i,
            s: *mut $r, u: *mut $t, ldu: $i, vt: *mut $t, ldvt: $i,
            workspace: *mut $t, lwork: $i, $($extra: *mut $r,)*
        ) -> $i {
            work(false, layout, ju, jv, m as i64, n as i64, a, lda as i64,
                u, ldu as i64, vt, ldvt as i64, workspace, lwork as i64,
                |a, la, u, lu, vt, lv, w, lw| {
                    let mut info: $i = 0;
                    crate::lapack::[<$p gesvd $f>](&ju, &jv, &m, &n, a, &(la as $i), s,
                        u, &(lu as $i), vt, &(lv as $i), w, &(lw as $i), $($extra,)* &mut info);
                    if info < 0 { (info - 1) as i64 } else { info as i64 }
                }) as $i
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p gesdd_work $($suffix)*>](
            layout: i32, job: u8, m: $i, n: $i, a: *mut $t, lda: $i,
            s: *mut $r, u: *mut $t, ldu: $i, vt: *mut $t, ldvt: $i,
            workspace: *mut $t, lwork: $i, $($extra: *mut $r,)* iwork: *mut $i,
        ) -> $i {
            work(true, layout, job, job, m as i64, n as i64, a, lda as i64,
                u, ldu as i64, vt, ldvt as i64, workspace, lwork as i64,
                |a, la, u, lu, vt, lv, w, lw| {
                    let mut info: $i = 0;
                    crate::lapack::[<$p gesdd $f>](&job, &m, &n, a, &(la as $i), s,
                        u, &(lu as $i), vt, &(lv as $i), w, &(lw as $i), $($extra,)* iwork, &mut info);
                    if info < 0 { (info - 1) as i64 } else { info as i64 }
                }) as $i
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p gesvd $($suffix)*>](
            layout: i32, ju: u8, jv: u8, m: $i, n: $i, a: *mut $t, lda: $i,
            s: *mut $r, u: *mut $t, ldu: $i, vt: *mut $t, ldvt: $i, superb: *mut $r,
        ) -> $i {
            high::<$t, $r, $i>(false, layout, ju, m as i64, n as i64, a, lda as i64, superb,
                |w, lw, _rw, _iw| {
                    if lw > <$i>::MAX as i64 { return -1010; }
                    $(let $extra = _rw;)*
                    [<LAPACKE_ $p gesvd_work $($suffix)*>](layout, ju, jv, m, n, a, lda,
                        s, u, ldu, vt, ldvt, w, lw as $i, $($extra,)*) as i64
                }) as $i
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p gesdd $($suffix)*>](
            layout: i32, job: u8, m: $i, n: $i, a: *mut $t, lda: $i,
            s: *mut $r, u: *mut $t, ldu: $i, vt: *mut $t, ldvt: $i,
        ) -> $i {
            high::<$t, $r, $i>(true, layout, job, m as i64, n as i64, a, lda as i64, core::ptr::null_mut(),
                |w, lw, _rw, iw| {
                    if lw > <$i>::MAX as i64 { return -1010; }
                    $(let $extra = _rw;)*
                    [<LAPACKE_ $p gesdd_work $($suffix)*>](layout, job, m, n, a, lda,
                        s, u, ldu, vt, ldvt, w, lw as $i, $($extra,)* iw) as i64
                }) as $i
        }
    }};
}
exports!(s, f32, f32, i32, [], _, []);
exports!(d, f64, f64, i32, [], _, []);
exports!(c, colmat::c32, f32, i32, [], _, [rwork]);
exports!(z, colmat::c64, f64, i32, [], _, [rwork]);
exports!(s, f32, f32, i64, [64_], _64_, []);
exports!(d, f64, f64, i64, [64_], _64_, []);
exports!(c, colmat::c32, f32, i64, [64_], _64_, [rwork]);
exports!(z, colmat::c64, f64, i64, [64_], _64_, [rwork]);

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! numerical {
        ($test:ident, $p:ident, $t:ty, $r:ty, $i:ty, [$($suffix:tt)*], [$($extra:ident)*], $make:expr, $tol:expr) => { paste::paste! {
            #[test]
            fn $test() {
                unsafe {
                    for dd in [false, true] { for layout in [ROW_MAJOR, COL_MAJOR] {
                        for (m,n) in [(3usize,2usize), (2,3), (0,2), (2,0)] {
                            for job in [b'A', b'S', b'N', b'O'] {
                                let r = m.min(n);
                                let uc = if job == b'A' || (dd && job == b'O' && m<n) {m} else {r};
                                let vr = if job == b'A' || (dd && job == b'O' && m>=n) {n} else {r};
                                let lda = if layout == ROW_MAJOR {n+2} else {m+2};
                                let ldu = if layout == ROW_MAJOR {uc.max(1)+2} else {m+2};
                                let ldvt = if layout == ROW_MAJOR {n+2} else {vr.max(1)+2};
                                let offset = |i:usize,j:usize,ld:usize| if layout == ROW_MAJOR {i*ld+j} else {i+j*ld};
                                let sentinel = <$t>::real(789.0);
                                let mut a = vec![sentinel; lda*(m.max(n)+2)];
                                for i in 0..m { for j in 0..n { a[offset(i,j,lda)] = ($make)(i,j); } }
                                let original = a.clone();
                                let mut u = vec![sentinel; ldu*(m+2)];
                                let mut vt = vec![sentinel; ldvt*(n+2)];
                                let mut values = vec![0 as $r; r.max(1)];
                                let mut superb = vec![0 as $r; r.max(1)];
                                let mut query = <$t>::z();
                                let mut _rw = vec![0 as $r; 128];
                                let mut iw = vec![0 as $i; 8*r.max(1)];
                                $(let $extra = _rw.as_mut_ptr();)*
                                let info = if dd {
                                    [<LAPACKE_ $p gesdd_work $($suffix)*>](layout,job,m as $i,n as $i,a.as_mut_ptr(),lda as $i,
                                        values.as_mut_ptr(),u.as_mut_ptr(),ldu as $i,vt.as_mut_ptr(),ldvt as $i,&mut query,-1,$($extra,)*iw.as_mut_ptr())
                                } else {
                                    [<LAPACKE_ $p gesvd_work $($suffix)*>](layout,job,if job==b'O' {b'N'} else {job},m as $i,n as $i,a.as_mut_ptr(),lda as $i,
                                        values.as_mut_ptr(),u.as_mut_ptr(),ldu as $i,vt.as_mut_ptr(),ldvt as $i,&mut query,-1,$($extra,)*)
                                };
                                assert_eq!(info,0,"query dd={dd} layout={layout} job={job} m={m} n={n}");
                                assert_eq!(a,original,"query touched A");
                                assert!(query.re() >= 1.0);
                                // Run both high and caller-workspace paths against the same input.
                                for high_level in [true, false] {
                                    a.copy_from_slice(&original);
                                    u.fill(sentinel); vt.fill(sentinel);
                                    let mut w = vec![<$t>::z(); query.re() as usize];
                                    let info = if dd {
                                        if high_level { [<LAPACKE_ $p gesdd $($suffix)*>](layout,job,m as $i,n as $i,a.as_mut_ptr(),lda as $i,
                                            values.as_mut_ptr(),u.as_mut_ptr(),ldu as $i,vt.as_mut_ptr(),ldvt as $i) }
                                        else { [<LAPACKE_ $p gesdd_work $($suffix)*>](layout,job,m as $i,n as $i,a.as_mut_ptr(),lda as $i,
                                            values.as_mut_ptr(),u.as_mut_ptr(),ldu as $i,vt.as_mut_ptr(),ldvt as $i,w.as_mut_ptr(),w.len() as $i,$($extra,)*iw.as_mut_ptr()) }
                                    } else {
                                        let jv = if job==b'O' {b'N'} else {job};
                                        if high_level { [<LAPACKE_ $p gesvd $($suffix)*>](layout,job,jv,m as $i,n as $i,a.as_mut_ptr(),lda as $i,
                                            values.as_mut_ptr(),u.as_mut_ptr(),ldu as $i,vt.as_mut_ptr(),ldvt as $i,superb.as_mut_ptr()) }
                                        else { [<LAPACKE_ $p gesvd_work $($suffix)*>](layout,job,jv,m as $i,n as $i,a.as_mut_ptr(),lda as $i,
                                            values.as_mut_ptr(),u.as_mut_ptr(),ldu as $i,vt.as_mut_ptr(),ldvt as $i,w.as_mut_ptr(),w.len() as $i,$($extra,)*) }
                                    };
                                    assert_eq!(info,0,"execute dd={dd} layout={layout} job={job} m={m} n={n}");
                                    let mut norm = 0.0;
                                    for i in 0..m { for j in 0..n { norm += original[offset(i,j,lda)].magnitude().powi(2); } }
                                    let snorm:f64 = values[..r].iter().map(|&v| (v as f64).powi(2)).sum();
                                    assert!((norm-snorm).abs() <= $tol*(1.0+norm));
                                    for k in 0..r { assert!(values[k]>=0 as $r); if k+1<r { assert!(values[k]>=values[k+1]); } }
                                    if matches!(job,b'A'|b'S') {
                                        for i in 0..m { for j in 0..n {
                                            let mut value = <$t>::z();
                                            for k in 0..r { value = value + u[offset(i,k,ldu)]*<$t>::real(values[k] as f64)*vt[offset(k,j,ldvt)]; }
                                            assert!((value-original[offset(i,j,lda)]).magnitude() <= $tol*(1.0+norm.sqrt()));
                                        } }
                                    }
                                    for k in 0..a.len() {
                                        let logical = (0..m).any(|i| (0..n).any(|j| offset(i,j,lda)==k));
                                        if !logical { assert_eq!(a[k],sentinel,"A padding changed"); }
                                    }
                                }
                            }
                        }
                    } }
                }
            }
        }};
    }
    numerical!(
        real32_lp64,
        s,
        f32,
        f32,
        i32,
        [],
        [],
        |i, j| ((i + 1) * 3 + j * j) as f32,
        2e-5
    );
    numerical!(
        real64_lp64,
        d,
        f64,
        f64,
        i32,
        [],
        [],
        |i, j| ((i + 1) * 3 + j * j) as f64,
        1e-12
    );
    numerical!(
        complex32_lp64,
        c,
        colmat::c32,
        f32,
        i32,
        [],
        [rw],
        |i, j| colmat::c32::new(
            ((i + 1) * 3 + j * j) as f32,
            (i as f32 - 2.0 * j as f32) * 0.3
        ),
        3e-5
    );
    numerical!(
        complex64_lp64,
        z,
        colmat::c64,
        f64,
        i32,
        [],
        [rw],
        |i, j| colmat::c64::new(
            ((i + 1) * 3 + j * j) as f64,
            (i as f64 - 2.0 * j as f64) * 0.3
        ),
        1e-12
    );
    numerical!(
        real32_ilp64,
        s,
        f32,
        f32,
        i64,
        [64_],
        [],
        |i, j| ((i + 1) * 3 + j * j) as f32,
        2e-5
    );
    numerical!(
        real64_ilp64,
        d,
        f64,
        f64,
        i64,
        [64_],
        [],
        |i, j| ((i + 1) * 3 + j * j) as f64,
        1e-12
    );
    numerical!(
        complex32_ilp64,
        c,
        colmat::c32,
        f32,
        i64,
        [64_],
        [rw],
        |i, j| colmat::c32::new(
            ((i + 1) * 3 + j * j) as f32,
            (i as f32 - 2.0 * j as f32) * 0.3
        ),
        3e-5
    );
    numerical!(
        complex64_ilp64,
        z,
        colmat::c64,
        f64,
        i64,
        [64_],
        [rw],
        |i, j| colmat::c64::new(
            ((i + 1) * 3 + j * j) as f64,
            (i as f64 - 2.0 * j as f64) * 0.3
        ),
        1e-12
    );
}
