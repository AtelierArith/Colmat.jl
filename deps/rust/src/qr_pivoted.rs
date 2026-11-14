//! Column-pivoted QR adapter primitives. Pivot choices use fresh residual
//! column norms rather than translating a diagonal-based native rank estimate.
use crate::householder::ReflectorScalar;
use crate::kernels::{nrm2, Scalar};

pub(crate) struct PivotedQr {
    pub permutation: Vec<usize>,
    pub rank: usize,
    pub residual: f64,
    pub initial: f64,
    pub exception: usize,
}

pub(crate) unsafe fn pivoted_qr<T: ReflectorScalar>(
    m: usize,
    n: usize,
    nrhs: usize,
    a: *mut T,
    lda: isize,
    tau: *mut T,
    fixed: &[bool],
    max_rank: usize,
    abs_tol: f64,
    rel_tol: f64,
) -> PivotedQr {
    let mut permutation: Vec<usize> = (0..n).collect();
    // Stable fixed-column order is part of JPVT's public contract.
    let order: Vec<usize> = (0..n)
        .filter(|&j| fixed[j])
        .chain((0..n).filter(|&j| !fixed[j]))
        .collect();
    for (target, &original) in order.iter().enumerate() {
        let source = permutation.iter().position(|&p| p == original).unwrap();
        if source != target {
            for i in 0..m {
                std::ptr::swap(
                    a.offset(i as isize + target as isize * lda),
                    a.offset(i as isize + source as isize * lda),
                );
            }
            permutation.swap(target, source);
        }
    }
    let fixed_count = fixed.iter().filter(|&&x| x).count();
    let mut initial = 0.0_f64;
    let mut exception = 0;
    for j in 0..n {
        let norm = nrm2(m as isize, a.offset(j as isize * lda), 1);
        if norm.is_nan() {
            return PivotedQr {
                permutation,
                rank: 0,
                residual: f64::NAN,
                initial: f64::NAN,
                exception: j + 1,
            };
        }
        if norm.is_infinite() && exception == 0 {
            exception = n + j + 1;
        }
        initial = initial.max(norm);
    }
    let mut result = PivotedQr {
        permutation,
        rank: 0,
        residual: initial,
        initial,
        exception,
    };
    let mut work = vec![T::z(); n + nrhs];
    for j in 0..m.min(n).min(max_rank) {
        let mut pivot = j;
        let mut best = -1.0;
        for col in j..n {
            let norm = nrm2(
                (m - j) as isize,
                a.offset(j as isize + col as isize * lda),
                1,
            );
            if norm.is_nan() {
                result.residual = f64::NAN;
                result.exception = col + 1;
                return result;
            }
            if norm > best {
                best = norm;
                pivot = col;
            }
        }
        result.residual = best;
        if best == 0.0
            || (abs_tol >= 0.0 && best <= abs_tol)
            || (rel_tol >= 0.0 && (if j == 0 { 1.0 } else { best / initial }) <= rel_tol)
        {
            break;
        }
        if j < fixed_count {
            pivot = j;
        }
        if pivot != j {
            for i in 0..m {
                std::ptr::swap(
                    a.offset(i as isize + j as isize * lda),
                    a.offset(i as isize + pivot as isize * lda),
                );
            }
            result.permutation.swap(j, pivot);
        }
        let column = a.offset(j as isize * (lda + 1));
        if best.is_finite() {
            crate::qr::standard_qr(m - j, 1, column, lda, tau.add(j), false);
        } else {
            crate::householder::generate(
                (m - j) as isize,
                column,
                column.add(1),
                1,
                tau.add(j),
                false,
            );
        }
        if (*tau.add(j)).magnitude().is_nan() {
            result.residual = f64::NAN;
            result.exception = j + 1;
            return result;
        }
        let trailing = n + nrhs - j - 1;
        if trailing > 0 {
            let beta = *column;
            *column = T::o();
            crate::householder::apply(
                b'L',
                (m - j) as isize,
                trailing as isize,
                column,
                1,
                (*tau.add(j)).cj(),
                a.offset(j as isize + (j + 1) as isize * lda),
                lda,
                work.as_mut_ptr(),
            );
            *column = beta;
        }
        result.rank = j + 1;
    }
    result.residual = 0.0;
    if result.rank < m.min(n) {
        for col in result.rank..n {
            result.residual = result.residual.max(nrm2(
                (m - result.rank) as isize,
                a.offset(result.rank as isize + col as isize * lda),
                1,
            ));
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixed_columns_and_residual_stop() {
        unsafe {
            let original = [1.0_f64, 2.0, 3.0, 7.0, 0.0, 1.0, 0.0, 4.0, 2.0];
            let mut a = original;
            let mut tau = [0.0; 3];
            let result = pivoted_qr(
                3,
                3,
                0,
                a.as_mut_ptr(),
                3,
                tau.as_mut_ptr(),
                &[false, true, false],
                3,
                -1.0,
                -1.0,
            );
            assert_eq!(result.permutation[0], 1);
            let mut r = a;
            for j in 0..3 {
                for i in j + 1..3 {
                    r[i + j * 3] = 0.0;
                }
            }
            crate::qr::apply_q(
                b'L',
                b'N',
                3,
                3,
                result.rank,
                a.as_ptr(),
                3,
                tau.as_ptr(),
                r.as_mut_ptr(),
                3,
            );
            for j in 0..3 {
                for i in 0..3 {
                    assert!((r[i + j * 3] - original[i + result.permutation[j] * 3]).abs() < 1e-12);
                }
            }
            let mut a = original;
            let mut tau = [0.0; 3];
            let result = pivoted_qr(
                3,
                3,
                0,
                a.as_mut_ptr(),
                3,
                tau.as_mut_ptr(),
                &[false; 3],
                3,
                f64::INFINITY,
                -1.0,
            );
            assert_eq!(result.rank, 0);
            assert_eq!(a, original);
            assert_eq!(result.residual, result.initial);
        }
    }
}

use crate::lapack::LapackInt;
use colmat::{c32, c64};
use paste::paste;
unsafe fn complete<T: ReflectorScalar, I: LapackInt>(
    m: usize,
    n: usize,
    a: *mut T,
    lda: isize,
    jpvt: *mut I,
    tau: *mut T,
) {
    let fixed: Vec<bool> = (0..n).map(|j| (*jpvt.add(j)).val() != 0).collect();
    for j in 0..m.min(n) {
        *tau.add(j) = T::z();
    }
    let result = pivoted_qr(m, n, 0, a, lda, tau, &fixed, m.min(n), -1.0, -1.0);
    for j in 0..n {
        *jpvt.add(j) = I::from_i64(result.permutation[j] as i64 + 1);
    }
}
macro_rules! real_exports {
    ($p:ident,$t:ty,$i:ty,$s:tt)=>{paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p geqp3 $s>](m:*const $i,n:*const $i,a:*mut $t,lda:*const $i,jpvt:*mut $i,tau:*mut $t,work:*mut $t,lwork:*const $i,info:*mut $i) {
            *info=0;let(m,n,la,lw)=(*m as isize,*n as isize,*lda as isize,*lwork as isize);
            let minwork=if m.min(n)==0 {1} else {3*n+1};
            for(arg,ok)in [(1,m>=0),(2,n>=0),(4,la>=m.max(1)),(8,lw== -1||lw>=minwork)] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            *work=crate::lapack::workspace::<$t>(minwork);
            if lw== -1 {return;}
            complete(m as usize,n as usize,a,la,jpvt,tau);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p geqpf $s>](m:*const $i,n:*const $i,a:*mut $t,lda:*const $i,jpvt:*mut $i,tau:*mut $t,_work:*mut $t,info:*mut $i) {
            *info=0;let(m,n,la)=(*m as isize,*n as isize,*lda as isize);
            for(arg,ok)in [(1,m>=0),(2,n>=0),(4,la>=m.max(1))] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            complete(m as usize,n as usize,a,la,jpvt,tau);
        }
    }};
}
macro_rules! complex_exports {
    ($p:ident,$t:ty,$r:ty,$i:ty,$s:tt)=>{paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p geqp3 $s>](m:*const $i,n:*const $i,a:*mut $t,lda:*const $i,jpvt:*mut $i,tau:*mut $t,work:*mut $t,lwork:*const $i,_rwork:*mut $r,info:*mut $i) {
            *info=0;let(m,n,la,lw)=(*m as isize,*n as isize,*lda as isize,*lwork as isize);
            let minwork=if m.min(n)==0 {1} else {n+1};
            for(arg,ok)in [(1,m>=0),(2,n>=0),(4,la>=m.max(1)),(8,lw== -1||lw>=minwork)] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            *work=crate::lapack::workspace::<$t>(minwork);
            if lw== -1 {return;}
            complete(m as usize,n as usize,a,la,jpvt,tau);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p geqpf $s>](m:*const $i,n:*const $i,a:*mut $t,lda:*const $i,jpvt:*mut $i,tau:*mut $t,_work:*mut $t,_rwork:*mut $r,info:*mut $i) {
            *info=0;let(m,n,la)=(*m as isize,*n as isize,*lda as isize);
            for(arg,ok)in [(1,m>=0),(2,n>=0),(4,la>=m.max(1))] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            complete(m as usize,n as usize,a,la,jpvt,tau);
        }
    }};
}
real_exports!(s, f32, i32, _);
real_exports!(d, f64, i32, _);
real_exports!(s, f32, i64, _64_);
real_exports!(d, f64, i64, _64_);
complex_exports!(c, c32, f32, i32, _);
complex_exports!(z, c64, f64, i32, _);
complex_exports!(c, c32, f32, i64, _64_);
complex_exports!(z, c64, f64, i64, _64_);

unsafe fn rank_driver<T: ReflectorScalar, R: Scalar, I: LapackInt>(
    m: I,
    n: I,
    nrhs: I,
    kmax: I,
    abstol: *mut R,
    reltol: *mut R,
    a: *mut T,
    lda: I,
    k: *mut I,
    residual: *mut R,
    relative: *mut R,
    jpiv: *mut I,
    tau: *mut T,
    work: *mut T,
    lwork: I,
    info: *mut I,
) {
    *info = I::from_i64(0);
    let (m, n, r, limit, la, lw) = (
        m.val(),
        n.val(),
        nrhs.val(),
        kmax.val(),
        lda.val(),
        lwork.val(),
    );
    for (arg, ok) in [
        (1, m >= 0),
        (2, n >= 0),
        (3, r >= 0),
        (4, limit >= 0),
        (5, !(*abstol).re().is_nan()),
        (6, !(*reltol).re().is_nan()),
        (8, la >= m.max(1)),
    ] {
        if !ok {
            *info = I::from_i64(-arg);
            crate::invalid(arg as i32);
            return;
        }
    }
    let minimum = if m.min(n) == 0 {
        1
    } else if T::COMPLEX {
        n + r - 1
    } else {
        3 * n + r - 1
    };
    *work = crate::lapack::workspace::<T>(minimum.max(1));
    if lw != -1 && lw < minimum {
        *info = I::from_i64(-15);
        crate::invalid(15);
        return;
    }
    if lw == -1 {
        return;
    }
    let (m, n, r) = (m as usize, n as usize, r as usize);
    for j in 0..m.min(n) {
        *tau.add(j) = T::z();
    }
    let absolute = if (*abstol).re() >= 0.0 {
        (*abstol).re().max(T::SAFE * T::EPS)
    } else {
        -1.0
    };
    let relative_tol = if (*reltol).re() >= 0.0 {
        (*reltol).re().max(T::EPS / 2.0)
    } else {
        -1.0
    };
    let result = pivoted_qr(
        m,
        n,
        r,
        a,
        la,
        tau,
        &vec![false; n],
        limit as usize,
        absolute,
        relative_tol,
    );
    *k = I::from_i64(result.rank as i64);
    *residual = R::real(result.residual);
    *relative = R::real(if result.residual == 0.0 {
        0.0
    } else if result.rank == 0 && !result.initial.is_nan() {
        1.0
    } else {
        result.residual / result.initial
    });
    *info = I::from_i64(result.exception as i64);
    for j in 0..n {
        *jpiv.add(j) = I::from_i64(result.permutation[j] as i64 + 1);
    }
}
macro_rules! rank_real_exports {
    ($p:ident,$t:ty,$i:ty,$s:tt)=>{paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p geqp3rk $s>](m:*const $i,n:*const $i,nrhs:*const $i,kmax:*const $i,abstol:*mut $t,reltol:*mut $t,
            a:*mut $t,lda:*const $i,k:*mut $i,residual:*mut $t,relative:*mut $t,jpiv:*mut $i,tau:*mut $t,work:*mut $t,lwork:*const $i,_iwork:*mut $i,info:*mut $i) {
            rank_driver(*m,*n,*nrhs,*kmax,abstol,reltol,a,*lda,k,residual,relative,jpiv,tau,work,*lwork,info);
        }
    }};
}
macro_rules! rank_complex_exports {
    ($p:ident,$t:ty,$r:ty,$i:ty,$s:tt)=>{paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p geqp3rk $s>](m:*const $i,n:*const $i,nrhs:*const $i,kmax:*const $i,abstol:*mut $r,reltol:*mut $r,
            a:*mut $t,lda:*const $i,k:*mut $i,residual:*mut $r,relative:*mut $r,jpiv:*mut $i,tau:*mut $t,work:*mut $t,lwork:*const $i,_rwork:*mut $r,_iwork:*mut $i,info:*mut $i) {
            rank_driver(*m,*n,*nrhs,*kmax,abstol,reltol,a,*lda,k,residual,relative,jpiv,tau,work,*lwork,info);
        }
    }};
}
rank_real_exports!(s, f32, i32, _);
rank_real_exports!(d, f64, i32, _);
rank_real_exports!(s, f32, i64, _64_);
rank_real_exports!(d, f64, i64, _64_);
rank_complex_exports!(c, c32, f32, i32, _);
rank_complex_exports!(z, c64, f64, i32, _);
rank_complex_exports!(c, c32, f32, i64, _64_);
rank_complex_exports!(z, c64, f64, i64, _64_);
