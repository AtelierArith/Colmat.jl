//! Least-squares driver compositions with LAPACK in-place output layouts.
use crate::householder::ReflectorScalar;
use crate::kernels::{trmatrix, Scalar};

pub(crate) unsafe fn full_rank<T: ReflectorScalar>(
    trans: u8,
    m: usize,
    n: usize,
    nrhs: usize,
    a: *mut T,
    lda: isize,
    b: *mut T,
    ldb: isize,
) -> usize {
    let k = m.min(n);
    if k == 0 || nrhs == 0 {
        for j in 0..nrhs {
            for i in 0..m.max(n) {
                *b.offset(i as isize + j as isize * ldb) = T::z();
            }
        }
        return 0;
    }
    let mut tau = vec![T::z(); k];
    let adjoint = if T::COMPLEX { b'C' } else { b'T' };
    if m >= n {
        crate::qr::standard_qr(m, n, a, lda, tau.as_mut_ptr(), false);
        for i in 0..n {
            if *a.offset(i as isize * (lda + 1)) == T::z() {
                return i + 1;
            }
        }
        if trans == b'N' {
            crate::qr::apply_q(b'L', adjoint, m, nrhs, k, a, lda, tau.as_ptr(), b, ldb);
            trmatrix(
                b'L',
                b'U',
                b'N',
                b'N',
                n as isize,
                nrhs as isize,
                T::o(),
                a,
                lda,
                b,
                ldb,
                true,
            );
        } else {
            trmatrix(
                b'L',
                b'U',
                adjoint,
                b'N',
                n as isize,
                nrhs as isize,
                T::o(),
                a,
                lda,
                b,
                ldb,
                true,
            );
            for j in 0..nrhs {
                for i in n..m {
                    *b.offset(i as isize + j as isize * ldb) = T::z();
                }
            }
            crate::qr::apply_q(b'L', b'N', m, nrhs, k, a, lda, tau.as_ptr(), b, ldb);
        }
    } else {
        crate::lq::standard_lq(m, n, a, lda, tau.as_mut_ptr());
        for i in 0..m {
            if *a.offset(i as isize * (lda + 1)) == T::z() {
                return i + 1;
            }
        }
        if trans == b'N' {
            trmatrix(
                b'L',
                b'L',
                b'N',
                b'N',
                m as isize,
                nrhs as isize,
                T::o(),
                a,
                lda,
                b,
                ldb,
                true,
            );
            for j in 0..nrhs {
                for i in m..n {
                    *b.offset(i as isize + j as isize * ldb) = T::z();
                }
            }
            crate::lq::apply_lq(b'L', adjoint, n, nrhs, k, a, lda, tau.as_ptr(), b, ldb);
        } else {
            crate::lq::apply_lq(b'L', b'N', n, nrhs, k, a, lda, tau.as_ptr(), b, ldb);
            trmatrix(
                b'L',
                b'L',
                adjoint,
                b'N',
                m as isize,
                nrhs as isize,
                T::o(),
                a,
                lda,
                b,
                ldb,
                true,
            );
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_tall_and_wide_solutions() {
        unsafe {
            let mut a = [1.0_f64, 0.0, 1.0, 0.0, 1.0, 1.0];
            let mut b = [2.0, 3.0, 5.0];
            assert_eq!(
                full_rank(b'N', 3, 2, 1, a.as_mut_ptr(), 3, b.as_mut_ptr(), 3),
                0
            );
            assert!((b[0] - 2.0).abs() < 1e-12 && (b[1] - 3.0).abs() < 1e-12);
            let mut a = [1.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0];
            let mut b = [2.0, 3.0, f64::NAN];
            assert_eq!(
                full_rank(b'N', 2, 3, 1, a.as_mut_ptr(), 2, b.as_mut_ptr(), 3),
                0
            );
            assert!((b[0] + b[2] - 2.0).abs() < 1e-12 && (b[1] + b[2] - 3.0).abs() < 1e-12);
            assert!((b[0] + b[1] - b[2]).abs() < 1e-12);
        }
    }
}

fn rescale(value: f64, numerator: f64, denominator: f64) -> f64 {
    let ratio = numerator / denominator;
    if ratio.is_finite() && ratio != 0.0 {
        value * ratio
    } else if numerator <= 1.0 {
        (value * numerator) / denominator
    } else {
        (value / denominator) * numerator
    }
}

pub(crate) unsafe fn scaled_full_rank<T: ReflectorScalar>(
    trans: u8,
    m: usize,
    n: usize,
    nrhs: usize,
    a: *mut T,
    lda: isize,
    b: *mut T,
    ldb: isize,
) -> usize {
    if m.min(n) == 0 || nrhs == 0 {
        return full_rank(trans, m, n, nrhs, a, lda, b, ldb);
    }
    let mut ascale = 0.0_f64;
    for j in 0..n {
        for i in 0..m {
            let z = *a.offset(i as isize + j as isize * lda);
            ascale = ascale.max(z.re().abs()).max(z.im().abs());
        }
    }
    if ascale == 0.0 {
        for j in 0..nrhs {
            for i in 0..m.max(n) {
                *b.offset(i as isize + j as isize * ldb) = T::z();
            }
        }
        return 0;
    }
    if !ascale.is_finite() {
        ascale = 1.0;
    }
    let input_rows = if trans == b'N' { m } else { n };
    let solution_rows = if trans == b'N' { n } else { m };
    let mut bscale = 0.0_f64;
    for j in 0..nrhs {
        for i in 0..input_rows {
            let z = *b.offset(i as isize + j as isize * ldb);
            bscale = bscale.max(z.re().abs()).max(z.im().abs());
        }
    }
    if bscale == 0.0 || !bscale.is_finite() {
        bscale = 1.0;
    }
    for j in 0..n {
        for i in 0..m {
            let p = a.offset(i as isize + j as isize * lda);
            let z = *p;
            *p = T::parts(z.re() / ascale, z.im() / ascale);
        }
    }
    for j in 0..nrhs {
        for i in 0..input_rows {
            let p = b.offset(i as isize + j as isize * ldb);
            let z = *p;
            *p = T::parts(z.re() / bscale, z.im() / bscale);
        }
    }
    let info = full_rank(trans, m, n, nrhs, a, lda, b, ldb);
    for j in 0..n {
        for i in 0..m {
            if (m >= n && i <= j) || (m < n && i >= j) {
                let p = a.offset(i as isize + j as isize * lda);
                let z = *p;
                *p = T::parts(z.re() * ascale, z.im() * ascale);
            }
        }
    }
    if info == 0 {
        for j in 0..nrhs {
            for i in 0..m.max(n) {
                let p = b.offset(i as isize + j as isize * ldb);
                let z = *p;
                *p = if i < solution_rows {
                    T::parts(
                        rescale(z.re(), bscale, ascale),
                        rescale(z.im(), bscale, ascale),
                    )
                } else {
                    T::parts(z.re() * bscale, z.im() * bscale)
                };
            }
        }
    }
    info
}

use crate::kernels::flag;
use colmat::{c32, c64};
use paste::paste;
macro_rules! gels_exports {
    ($p:ident,$t:ty,$i:ty,$s:tt)=>{paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p gels $s>](tr:*const u8,m:*const $i,n:*const $i,nrhs:*const $i,a:*mut $t,lda:*const $i,
            b:*mut $t,ldb:*const $i,work:*mut $t,lwork:*const $i,info:*mut $i) {
            *info=0;let(tr,m,n,r,la,lb,lw)=(flag(tr),*m as isize,*n as isize,*nrhs as isize,*lda as isize,*ldb as isize,*lwork as isize);
            let minimum=(m.min(n)+m.min(n).max(r)).max(1);
            for(arg,ok)in [(1,tr==b'N'||tr==if <$t>::COMPLEX {b'C'} else {b'T'}),(2,m>=0),(3,n>=0),(4,r>=0),(6,la>=m.max(1)),(8,lb>=m.max(n).max(1))] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            *work=crate::lapack::workspace::<$t>(minimum);
            if lw != -1 && lw < minimum {*info=-10;crate::invalid(10);return;}
            if lw== -1 {return;}
            *info=scaled_full_rank(tr,m as usize,n as usize,r as usize,a,la,b,lb) as $i;
        }
    }};
}
gels_exports!(s, f32, i32, _);
gels_exports!(d, f64, i32, _);
gels_exports!(c, c32, i32, _);
gels_exports!(z, c64, i32, _);
gels_exports!(s, f32, i64, _64_);
gels_exports!(d, f64, i64, _64_);
gels_exports!(c, c32, i64, _64_);
gels_exports!(z, c64, i64, _64_);

pub(crate) unsafe fn svd_solve<T: ReflectorScalar>(
    m: usize,
    n: usize,
    nrhs: usize,
    a: *mut T,
    lda: isize,
    b: *mut T,
    ldb: isize,
    rcond: f64,
    store_vectors: bool,
) -> Result<(usize, Vec<f64>), ()> {
    let k = m.min(n);
    if k == 0 {
        for j in 0..nrhs {
            for i in 0..m.max(n) {
                *b.offset(i as isize + j as isize * ldb) = T::z();
            }
        }
        return Ok((0, vec![]));
    }
    let mut scale = 0.0_f64;
    for j in 0..n {
        for i in 0..m {
            let z = *a.offset(i as isize + j as isize * lda);
            scale = scale.max(z.re().abs()).max(z.im().abs());
        }
    }
    if scale == 0.0 {
        for j in 0..nrhs {
            for i in 0..m.max(n) {
                *b.offset(i as isize + j as isize * ldb) = T::z();
            }
        }
        return Ok((0, vec![0.0; k]));
    }
    if !scale.is_finite() {
        return Err(());
    }
    // The native SVD normalizes extreme scales internally, so the caller's
    // matrix can be read directly. The right-hand side still needs its own
    // scale to stay finite in the transformed products.
    let matrix = crate::lapack::input_ref(a, m, n, lda);
    let (u, values, v) = crate::lapack::svd(matrix, m, k)?;
    let cutoff = if rcond < 0.0 { T::EPS } else { rcond };
    let rank = values
        .iter()
        .take_while(|&&s| s > 0.0 && s > values[0] * cutoff)
        .count();
    let mut rhs_scale = 0.0_f64;
    for j in 0..nrhs {
        for i in 0..m {
            let z = *b.offset(i as isize + j as isize * ldb);
            rhs_scale = rhs_scale.max(z.re().abs()).max(z.im().abs());
        }
    }
    if rhs_scale == 0.0 {
        rhs_scale = 1.0;
    }
    if !rhs_scale.is_finite() {
        return Err(());
    }
    let mut rhs = vec![T::z(); m * nrhs];
    for j in 0..nrhs {
        for i in 0..m {
            let z = *b.offset(i as isize + j as isize * ldb);
            rhs[i + j * m] = T::parts(z.re() / rhs_scale, z.im() / rhs_scale);
        }
    }
    let mut transformed = vec![T::z(); m * nrhs];
    crate::kernels::gemm(
        b'C',
        b'N',
        m as isize,
        nrhs as isize,
        m as isize,
        T::o(),
        u.as_ref().as_ptr(),
        u.as_ref().col_stride(),
        rhs.as_ptr(),
        m as isize,
        T::z(),
        transformed.as_mut_ptr(),
        m as isize,
    );
    let mut weighted = vec![T::z(); rank * nrhs];
    for j in 0..nrhs {
        for i in 0..rank {
            let z = transformed[i + j * m];
            weighted[i + j * rank] = T::parts(z.re() / values[i], z.im() / values[i]);
        }
    }
    let mut solution = vec![T::z(); n * nrhs];
    crate::kernels::gemm(
        b'N',
        b'N',
        n as isize,
        nrhs as isize,
        rank as isize,
        T::o(),
        v.as_ref().as_ptr(),
        v.as_ref().col_stride(),
        weighted.as_ptr(),
        rank.max(1) as isize,
        T::z(),
        solution.as_mut_ptr(),
        n as isize,
    );
    for j in 0..nrhs {
        for i in 0..n {
            let z = solution[i + j * n];
            *b.offset(i as isize + j as isize * ldb) = T::parts(
                rescale(z.re(), rhs_scale, 1.0),
                rescale(z.im(), rhs_scale, 1.0),
            );
        }
        for i in n..m {
            let z = transformed[i + j * m];
            *b.offset(i as isize + j as isize * ldb) =
                T::parts(z.re() * rhs_scale, z.im() * rhs_scale);
        }
    }
    if store_vectors {
        for j in 0..n {
            for i in 0..k {
                *a.offset(i as isize + j as isize * lda) = v[(j, i)].cj();
            }
        }
    }
    Ok((rank, values))
}

#[cfg(test)]
mod svd_tests {
    use super::*;
    #[test]
    fn rank_deficient_minimum_norm() {
        unsafe {
            let mut a = [1.0_f64, 2.0, 2.0, 4.0, 3.0, 6.0];
            let mut b = [1.0, 2.0, f64::NAN];
            let (rank, s) =
                svd_solve(2, 3, 1, a.as_mut_ptr(), 2, b.as_mut_ptr(), 3, 1e-12, true).unwrap();
            assert_eq!(rank, 1);
            assert!((s[0] - 70.0_f64.sqrt()).abs() < 1e-12);
            for i in 0..3 {
                assert!((b[i] - (i + 1) as f64 / 14.0).abs() < 1e-12);
            }
        }
    }
}

use crate::lapack::LapackInt;
unsafe fn gelss_driver<T: ReflectorScalar, R: Scalar, I: LapackInt>(
    m: I,
    n: I,
    nrhs: I,
    a: *mut T,
    lda: I,
    b: *mut T,
    ldb: I,
    s: *mut R,
    rcond: R,
    rank: *mut I,
    work: *mut T,
    lwork: I,
    info: *mut I,
) {
    *info = I::from_i64(0);
    let (m, n, r, la, lb, lw) = (
        m.val(),
        n.val(),
        nrhs.val(),
        lda.val(),
        ldb.val(),
        lwork.val(),
    );
    let k = m.min(n);
    let minimum = if k == 0 {
        1
    } else if T::COMPLEX {
        2 * k + m.max(n).max(r)
    } else {
        3 * k + (2 * k).max(m.max(n)).max(r)
    };
    for (arg, ok) in [
        (1, m >= 0),
        (2, n >= 0),
        (3, r >= 0),
        (5, la >= m.max(1)),
        (7, lb >= m.max(n).max(1)),
    ] {
        if !ok {
            *info = I::from_i64(-arg);
            crate::invalid(arg as i32);
            return;
        }
    }
    *work = crate::lapack::workspace::<T>(minimum);
    if lw != -1 && lw < minimum {
        *info = I::from_i64(-12);
        crate::invalid(12);
        return;
    }
    if lw == -1 {
        return;
    }
    match svd_solve(
        m as usize,
        n as usize,
        r as usize,
        a,
        la,
        b,
        lb,
        rcond.re(),
        true,
    ) {
        Ok((count, values)) => {
            *rank = I::from_i64(count as i64);
            for (i, value) in values.iter().enumerate() {
                *s.add(i) = R::real(*value);
            }
        }
        Err(()) => {
            *info = I::from_i64(1);
        }
    }
}
macro_rules! gelss_real_exports {
    ($p:ident,$t:ty,$i:ty,$suffix:tt)=>{paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p gelss $suffix>](m:*const $i,n:*const $i,nrhs:*const $i,a:*mut $t,lda:*const $i,b:*mut $t,ldb:*const $i,
            s:*mut $t,rcond:*const $t,rank:*mut $i,work:*mut $t,lwork:*const $i,info:*mut $i) {
            gelss_driver(*m,*n,*nrhs,a,*lda,b,*ldb,s,*rcond,rank,work,*lwork,info);
        }
    }};
}
macro_rules! gelss_complex_exports {
    ($p:ident,$t:ty,$r:ty,$i:ty,$suffix:tt)=>{paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p gelss $suffix>](m:*const $i,n:*const $i,nrhs:*const $i,a:*mut $t,lda:*const $i,b:*mut $t,ldb:*const $i,
            s:*mut $r,rcond:*const $r,rank:*mut $i,work:*mut $t,lwork:*const $i,_rwork:*mut $r,info:*mut $i) {
            gelss_driver(*m,*n,*nrhs,a,*lda,b,*ldb,s,*rcond,rank,work,*lwork,info);
        }
    }};
}
gelss_real_exports!(s, f32, i32, _);
gelss_real_exports!(d, f64, i32, _);
gelss_real_exports!(s, f32, i64, _64_);
gelss_real_exports!(d, f64, i64, _64_);
gelss_complex_exports!(c, c32, f32, i32, _);
gelss_complex_exports!(z, c64, f64, i32, _);
gelss_complex_exports!(c, c32, f32, i64, _64_);
gelss_complex_exports!(z, c64, f64, i64, _64_);

fn gelsd_workspace(m: isize, n: isize, nrhs: isize, complex: bool) -> (isize, isize, isize) {
    let k = m.min(n).max(0);
    if k == 0 {
        return (1, 1, 1);
    }
    let levels = (((k as f64 / 26.0).log2() as isize) + 1).max(0);
    let integer = (3 * k * levels + 11 * k).max(1);
    if complex {
        let real =
            10 * k + 50 * k + 8 * k * levels + 75 * nrhs + (26 * 26).max(n * (1 + nrhs) + 2 * nrhs);
        ((2 * k + k * nrhs).max(1), integer, real.max(1))
    } else {
        (
            (12 * k + 50 * k + 8 * k * levels + k * nrhs + 26 * 26).max(1),
            integer,
            1,
        )
    }
}
unsafe fn gelsd_driver<T: ReflectorScalar, R: Scalar, I: LapackInt>(
    m: I,
    n: I,
    nrhs: I,
    a: *mut T,
    lda: I,
    b: *mut T,
    ldb: I,
    s: *mut R,
    rcond: R,
    rank: *mut I,
    work: *mut T,
    lwork: I,
    rwork: *mut R,
    iwork: *mut I,
    info: *mut I,
) {
    *info = I::from_i64(0);
    let (m, n, r, la, lb, lw) = (
        m.val(),
        n.val(),
        nrhs.val(),
        lda.val(),
        ldb.val(),
        lwork.val(),
    );
    for (arg, ok) in [
        (1, m >= 0),
        (2, n >= 0),
        (3, r >= 0),
        (5, la >= m.max(1)),
        (7, lb >= m.max(n).max(1)),
    ] {
        if !ok {
            *info = I::from_i64(-arg);
            crate::invalid(arg as i32);
            return;
        }
    }
    let (w, iw, rw) = gelsd_workspace(m, n, r, T::COMPLEX);
    *work = crate::lapack::workspace::<T>(w);
    *iwork = I::from_i64(iw as i64);
    if T::COMPLEX {
        *rwork = crate::lapack::workspace::<R>(rw);
    }
    if lw != -1 && lw < w {
        *info = I::from_i64(-12);
        crate::invalid(12);
        return;
    }
    if lw == -1 {
        return;
    }
    match svd_solve(
        m as usize,
        n as usize,
        r as usize,
        a,
        la,
        b,
        lb,
        rcond.re(),
        false,
    ) {
        Ok((count, values)) => {
            *rank = I::from_i64(count as i64);
            for (i, value) in values.iter().enumerate() {
                *s.add(i) = R::real(*value);
            }
        }
        Err(()) => {
            *info = I::from_i64(1);
        }
    }
}
macro_rules! gelsd_real_exports {
    ($p:ident,$t:ty,$i:ty,$suffix:tt)=>{paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p gelsd $suffix>](m:*const $i,n:*const $i,nrhs:*const $i,a:*mut $t,lda:*const $i,b:*mut $t,ldb:*const $i,
            s:*mut $t,rcond:*const $t,rank:*mut $i,work:*mut $t,lwork:*const $i,iwork:*mut $i,info:*mut $i) {
            gelsd_driver(*m,*n,*nrhs,a,*lda,b,*ldb,s,*rcond,rank,work,*lwork,std::ptr::null_mut(),iwork,info);
        }
    }};
}
macro_rules! gelsd_complex_exports {
    ($p:ident,$t:ty,$r:ty,$i:ty,$suffix:tt)=>{paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p gelsd $suffix>](m:*const $i,n:*const $i,nrhs:*const $i,a:*mut $t,lda:*const $i,b:*mut $t,ldb:*const $i,
            s:*mut $r,rcond:*const $r,rank:*mut $i,work:*mut $t,lwork:*const $i,rwork:*mut $r,iwork:*mut $i,info:*mut $i) {
            gelsd_driver(*m,*n,*nrhs,a,*lda,b,*ldb,s,*rcond,rank,work,*lwork,rwork,iwork,info);
        }
    }};
}
gelsd_real_exports!(s, f32, i32, _);
gelsd_real_exports!(d, f64, i32, _);
gelsd_real_exports!(s, f32, i64, _64_);
gelsd_real_exports!(d, f64, i64, _64_);
gelsd_complex_exports!(c, c32, f32, i32, _);
gelsd_complex_exports!(z, c64, f64, i32, _);
gelsd_complex_exports!(c, c32, f32, i64, _64_);
gelsd_complex_exports!(z, c64, f64, i64, _64_);

pub(crate) unsafe fn complete_orthogonal<T: ReflectorScalar>(
    m: usize,
    n: usize,
    nrhs: usize,
    a: *mut T,
    lda: isize,
    b: *mut T,
    ldb: isize,
    fixed: &[bool],
    rcond: f64,
) -> Result<(usize, Vec<usize>), ()> {
    let k = m.min(n);
    let mut tau = vec![T::z(); k];
    let pivot =
        crate::qr_pivoted::pivoted_qr(m, n, 0, a, lda, tau.as_mut_ptr(), fixed, k, -1.0, -1.0);
    if pivot.exception != 0 {
        return Err(());
    }
    let mut rank = 0;
    if k > 0 && *a != T::z() {
        rank = 1;
        for size in 2..=k {
            let triangular = colmat::Mat::from_fn(size, size, |i, j| {
                if i <= j {
                    *a.offset(i as isize + j as isize * lda)
                } else {
                    T::z()
                }
            });
            let (_, s, _) = crate::lapack::svd(triangular.as_ref(), 0, 0)?;
            if s[size - 1] < s[0] * rcond {
                break;
            }
            rank = size;
        }
    }
    if nrhs == 0 {
        return Ok((rank, pivot.permutation));
    }
    crate::qr::apply_q(
        b'L',
        if T::COMPLEX { b'C' } else { b'T' },
        m,
        nrhs,
        k,
        a,
        lda,
        tau.as_ptr(),
        b,
        ldb,
    );
    let mut rz_tau = vec![T::z(); rank];
    if rank < n {
        crate::rz::factor(rank, n, n - rank, a, lda, rz_tau.as_mut_ptr());
    }
    if rank > 0 {
        trmatrix(
            b'L',
            b'U',
            b'N',
            b'N',
            rank as isize,
            nrhs as isize,
            T::o(),
            a,
            lda,
            b,
            ldb,
            true,
        );
    }
    for j in 0..nrhs {
        for i in rank..n {
            *b.offset(i as isize + j as isize * ldb) = T::z();
        }
    }
    if rank < n {
        crate::rz::apply_rz(
            b'L',
            if T::COMPLEX { b'C' } else { b'T' },
            n,
            nrhs,
            rank,
            n - rank,
            a,
            lda,
            rz_tau.as_ptr(),
            b,
            ldb,
        );
    }
    let mut solution = vec![T::z(); n];
    for j in 0..nrhs {
        for i in 0..n {
            solution[pivot.permutation[i]] = *b.offset(i as isize + j as isize * ldb);
        }
        for i in 0..n {
            *b.offset(i as isize + j as isize * ldb) = solution[i];
        }
    }
    Ok((rank, pivot.permutation))
}

#[cfg(test)]
mod complete_tests {
    use super::*;
    #[test]
    fn fixed_rank_deficient_complete_solve() {
        unsafe {
            let mut a = [1.0_f64, 2.0, 2.0, 4.0, 3.0, 6.0];
            let mut b = [1.0, 2.0, f64::NAN];
            let (rank, p) = complete_orthogonal(
                2,
                3,
                1,
                a.as_mut_ptr(),
                2,
                b.as_mut_ptr(),
                3,
                &[false, false, true],
                1e-12,
            )
            .unwrap();
            assert_eq!(rank, 1);
            assert_eq!(p[0], 2);
            for i in 0..3 {
                assert!((b[i] - (i + 1) as f64 / 14.0).abs() < 1e-12);
            }
        }
    }
}

unsafe fn scaled_complete<T: ReflectorScalar>(
    m: usize,
    n: usize,
    r: usize,
    a: *mut T,
    lda: isize,
    b: *mut T,
    ldb: isize,
    fixed: &[bool],
    rcond: f64,
) -> Result<(usize, Vec<usize>), ()> {
    let mut ascale = 0.0_f64;
    let mut bscale = 0.0_f64;
    for j in 0..n {
        for i in 0..m {
            let z = *a.offset(i as isize + j as isize * lda);
            ascale = ascale.max(z.re().abs()).max(z.im().abs());
        }
    }
    if ascale == 0.0 {
        for j in 0..r {
            for i in 0..m.max(n) {
                *b.offset(i as isize + j as isize * ldb) = T::z();
            }
        }
        return Ok((
            0,
            (0..n)
                .filter(|&j| fixed[j])
                .chain((0..n).filter(|&j| !fixed[j]))
                .collect(),
        ));
    }
    if !ascale.is_finite() {
        return Err(());
    }
    for j in 0..r {
        for i in 0..m {
            let z = *b.offset(i as isize + j as isize * ldb);
            bscale = bscale.max(z.re().abs()).max(z.im().abs());
        }
    }
    if bscale == 0.0 {
        bscale = 1.0;
    }
    if !bscale.is_finite() {
        return Err(());
    }
    for j in 0..n {
        for i in 0..m {
            let p = a.offset(i as isize + j as isize * lda);
            let z = *p;
            *p = T::parts(z.re() / ascale, z.im() / ascale);
        }
    }
    for j in 0..r {
        for i in 0..m {
            let p = b.offset(i as isize + j as isize * ldb);
            let z = *p;
            *p = T::parts(z.re() / bscale, z.im() / bscale);
        }
    }
    let (rank, permutation) = complete_orthogonal(m, n, r, a, lda, b, ldb, fixed, rcond)?;
    for j in 0..rank {
        for i in 0..=j {
            let p = a.offset(i as isize + j as isize * lda);
            let z = *p;
            *p = T::parts(z.re() * ascale, z.im() * ascale);
        }
    }
    for j in 0..r {
        for i in 0..m.max(n) {
            let p = b.offset(i as isize + j as isize * ldb);
            let z = *p;
            *p = if i < n {
                T::parts(
                    rescale(z.re(), bscale, ascale),
                    rescale(z.im(), bscale, ascale),
                )
            } else {
                T::parts(z.re() * bscale, z.im() * bscale)
            };
        }
    }
    Ok((rank, permutation))
}
unsafe fn gelsy_driver<T: ReflectorScalar, R: Scalar, I: LapackInt>(
    m: I,
    n: I,
    nrhs: I,
    a: *mut T,
    lda: I,
    b: *mut T,
    ldb: I,
    jpvt: *mut I,
    rcond: R,
    rank: *mut I,
    work: *mut T,
    lwork: Option<I>,
    info: *mut I,
) {
    *info = I::from_i64(0);
    let (m, n, r, la, lb) = (m.val(), n.val(), nrhs.val(), lda.val(), ldb.val());
    let k = m.min(n);
    let minimum = if !T::COMPLEX && (k == 0 || r == 0) {
        1
    } else if T::COMPLEX {
        k + (2 * k).max(n + 1).max(k + r)
    } else {
        (k + 3 * n + 1).max(2 * k + r)
    };
    let query = lwork.map(|lw| lw.val() == -1).unwrap_or(false);
    for (arg, ok) in [
        (1, m >= 0),
        (2, n >= 0),
        (3, r >= 0),
        (5, la >= m.max(1)),
        (7, lb >= m.max(n).max(1)),
    ] {
        if !ok {
            *info = I::from_i64(-arg);
            crate::invalid(arg as i32);
            return;
        }
    }
    if let Some(lw) = lwork {
        *work = crate::lapack::workspace::<T>(minimum);
        if !query && lw.val() < minimum {
            *info = I::from_i64(-12);
            crate::invalid(12);
            return;
        }
    }
    if query {
        return;
    }
    if k == 0 || r == 0 {
        *rank = I::from_i64(0);
        return;
    }
    let fixed: Vec<bool> = (0..n as usize).map(|j| (*jpvt.add(j)).val() != 0).collect();
    match scaled_complete(
        m as usize,
        n as usize,
        r as usize,
        a,
        la,
        b,
        lb,
        &fixed,
        rcond.re(),
    ) {
        Ok((count, permutation)) => {
            *rank = I::from_i64(count as i64);
            for (j, &p) in permutation.iter().enumerate() {
                *jpvt.add(j) = I::from_i64(p as i64 + 1);
            }
        }
        Err(()) => {
            *info = I::from_i64(1);
        }
    }
}

macro_rules! gelsy_real_exports {
    ($p:ident,$t:ty,$i:ty,$suffix:tt)=>{paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p gelsy $suffix>](m:*const $i,n:*const $i,nrhs:*const $i,a:*mut $t,lda:*const $i,b:*mut $t,ldb:*const $i,
            jpvt:*mut $i,rcond:*const $t,rank:*mut $i,work:*mut $t,lwork:*const $i,info:*mut $i) {
            gelsy_driver(*m,*n,*nrhs,a,*lda,b,*ldb,jpvt,*rcond,rank,work,Some(*lwork),info);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p gelsx $suffix>](m:*const $i,n:*const $i,nrhs:*const $i,a:*mut $t,lda:*const $i,b:*mut $t,ldb:*const $i,
            jpvt:*mut $i,rcond:*const $t,rank:*mut $i,work:*mut $t,info:*mut $i) {
            gelsy_driver(*m,*n,*nrhs,a,*lda,b,*ldb,jpvt,*rcond,rank,work,None,info);
        }
    }};
}
macro_rules! gelsy_complex_exports {
    ($p:ident,$t:ty,$r:ty,$i:ty,$suffix:tt)=>{paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p gelsy $suffix>](m:*const $i,n:*const $i,nrhs:*const $i,a:*mut $t,lda:*const $i,b:*mut $t,ldb:*const $i,
            jpvt:*mut $i,rcond:*const $r,rank:*mut $i,work:*mut $t,lwork:*const $i,_rwork:*mut $r,info:*mut $i) {
            gelsy_driver(*m,*n,*nrhs,a,*lda,b,*ldb,jpvt,*rcond,rank,work,Some(*lwork),info);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p gelsx $suffix>](m:*const $i,n:*const $i,nrhs:*const $i,a:*mut $t,lda:*const $i,b:*mut $t,ldb:*const $i,
            jpvt:*mut $i,rcond:*const $r,rank:*mut $i,work:*mut $t,_rwork:*mut $r,info:*mut $i) {
            gelsy_driver(*m,*n,*nrhs,a,*lda,b,*ldb,jpvt,*rcond,rank,work,None,info);
        }
    }};
}
gelsy_real_exports!(s, f32, i32, _);
gelsy_real_exports!(d, f64, i32, _);
gelsy_real_exports!(s, f32, i64, _64_);
gelsy_real_exports!(d, f64, i64, _64_);
gelsy_complex_exports!(c, c32, f32, i32, _);
gelsy_complex_exports!(z, c64, f64, i32, _);
gelsy_complex_exports!(c, c32, f32, i64, _64_);
gelsy_complex_exports!(z, c64, f64, i64, _64_);

macro_rules! full_driver_exports {
    ($p:ident,$t:ty,$i:ty,$suffix:tt)=>{paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p gelst $suffix>](tr:*const u8,m:*const $i,n:*const $i,nrhs:*const $i,a:*mut $t,lda:*const $i,
            b:*mut $t,ldb:*const $i,work:*mut $t,lwork:*const $i,info:*mut $i) {
            [<$p gels $suffix>](tr,m,n,nrhs,a,lda,b,ldb,work,lwork,info);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p getsls $suffix>](tr:*const u8,m:*const $i,n:*const $i,nrhs:*const $i,a:*mut $t,lda:*const $i,
            b:*mut $t,ldb:*const $i,work:*mut $t,lwork:*const $i,info:*mut $i) {
            *info=0;let(tr,m,n,r,la,lb,lw)=(flag(tr),*m as isize,*n as isize,*nrhs as isize,*lda as isize,*ldb as isize,*lwork as isize);
            let k=m.min(n);let minimum=if k==0||r==0 {1} else {k+5+k.max(r)};
            let query=matches!(lw,-1|-2);
            for(arg,ok)in [(1,tr==b'N'||tr==if <$t>::COMPLEX {b'C'} else {b'T'}),(2,m>=0),(3,n>=0),(4,r>=0),(6,la>=m.max(1)),(8,lb>=m.max(n).max(1)),(10,query||lw>=minimum)] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            *work=crate::lapack::workspace::<$t>(minimum);if query {return;}
            *info=scaled_full_rank(tr,m as usize,n as usize,r as usize,a,la,b,lb) as $i;
        }
    }};
}
full_driver_exports!(s, f32, i32, _);
full_driver_exports!(d, f64, i32, _);
full_driver_exports!(c, c32, i32, _);
full_driver_exports!(z, c64, i32, _);
full_driver_exports!(s, f32, i64, _64_);
full_driver_exports!(d, f64, i64, _64_);
full_driver_exports!(c, c32, i64, _64_);
full_driver_exports!(z, c64, i64, _64_);

#[cfg(test)]
mod workspace_publication_tests {
    use super::*;
    type Gels<T, I> = unsafe extern "C" fn(
        *const u8,
        *const I,
        *const I,
        *const I,
        *mut T,
        *const I,
        *mut T,
        *const I,
        *mut T,
        *const I,
        *mut I,
    );

    unsafe fn check<T: ReflectorScalar, R: Scalar, I: LapackInt>(gels: Gels<T, I>) {
        for (m, n, nrhs) in [(0, 0, 2), (0, 3, 2), (3, 0, 2), (3, 5, 0), (3, 2, 1)] {
            for family in 0..3 {
                let (n, r, la, lb) = (
                    I::from_i64(n),
                    I::from_i64(nrhs),
                    I::from_i64(m.max(1)),
                    I::from_i64(m.max(n).max(1)),
                );
                let execute = |m: I, lwork: I, expected: i64, query: Option<T>| {
                    let mut a = [T::real(11.0); 64];
                    let mut b = [T::real(12.0); 64];
                    let mut singular = [R::real(780.0); 16];
                    let mut pivot = [I::from_i64(779); 16];
                    let mut rank = I::from_i64(778);
                    let mut info = I::from_i64(-999);
                    let mut work = [T::real(777.0); 3];
                    match family {
                        0 => gels(
                            &b'N',
                            &m,
                            &n,
                            &r,
                            a.as_mut_ptr(),
                            &la,
                            b.as_mut_ptr(),
                            &lb,
                            work.as_mut_ptr().add(1),
                            &lwork,
                            &mut info,
                        ),
                        1 => gelss_driver(
                            m,
                            n,
                            r,
                            a.as_mut_ptr(),
                            la,
                            b.as_mut_ptr(),
                            lb,
                            singular.as_mut_ptr(),
                            R::real(-1.0),
                            &mut rank,
                            work.as_mut_ptr().add(1),
                            lwork,
                            &mut info,
                        ),
                        _ => gelsy_driver(
                            m,
                            n,
                            r,
                            a.as_mut_ptr(),
                            la,
                            b.as_mut_ptr(),
                            lb,
                            pivot.as_mut_ptr(),
                            R::real(-1.0),
                            &mut rank,
                            work.as_mut_ptr().add(1),
                            Some(lwork),
                            &mut info,
                        ),
                    }
                    assert_eq!(info.val() as i64, expected);
                    assert_eq!(
                        crate::colmat_blas_last_error() as i64,
                        if expected < 0 { -expected } else { 0 }
                    );
                    assert!(a.iter().all(|x| *x == T::real(11.0)));
                    assert!(b.iter().all(|x| *x == T::real(12.0)));
                    assert!(singular.iter().all(|x| *x == R::real(780.0)));
                    assert!(pivot.iter().all(|x| x.val() == 779));
                    assert_eq!(rank.val(), 778);
                    assert!(work[0] == T::real(777.0) && work[2] == T::real(777.0));
                    if let Some(query) = query {
                        assert!(work[1] == query);
                    }
                    work[1]
                };
                let query = execute(I::from_i64(m), I::from_i64(-1), 0, None);
                assert!(query.re() >= 1.0);
                execute(
                    I::from_i64(m),
                    I::from_i64(0),
                    if family == 0 { -10 } else { -12 },
                    Some(query),
                );
                // An earlier invalid header must leave WORK and output arrays untouched.
                execute(
                    I::from_i64(-1),
                    I::from_i64(0),
                    if family == 0 { -2 } else { -1 },
                    Some(T::real(777.0)),
                );
            }
        }
    }
    #[test]
    fn work_is_published_before_insufficient_work_in_both_abis() {
        unsafe {
            check::<f32, f32, i32>(sgels_);
            check::<f64, f64, i32>(dgels_);
            check::<c32, f32, i32>(cgels_);
            check::<c64, f64, i32>(zgels_);
            check::<f32, f32, i64>(sgels_64_);
            check::<f64, f64, i64>(dgels_64_);
            check::<c32, f32, i64>(cgels_64_);
            check::<c64, f64, i64>(zgels_64_);
        }
    }
}
