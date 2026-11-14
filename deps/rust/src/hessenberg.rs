//! Hessenberg reduction with LAPACK reflector gauge and active-range storage.
use crate::householder::ReflectorScalar;
use crate::kernels::Scalar;

pub(crate) unsafe fn reduce<T: ReflectorScalar>(
    n: usize,
    lo: usize,
    hi: usize,
    a: *mut T,
    lda: isize,
    tau: *mut T,
) {
    for i in 0..n.saturating_sub(1) {
        *tau.add(i) = T::z();
    }
    let mut work = vec![T::z(); n];
    for i in lo..hi.saturating_sub(1) {
        let height = hi - i - 1;
        let vector = a.offset((i + 1) as isize + i as isize * lda);
        crate::householder::generate(height as isize, vector, vector.add(1), 1, tau.add(i), false);
        let beta = *vector;
        *vector = T::o();
        crate::householder::apply(
            b'R',
            hi as isize,
            height as isize,
            vector,
            1,
            *tau.add(i),
            a.offset((i + 1) as isize * lda),
            lda,
            work.as_mut_ptr(),
        );
        crate::householder::apply(
            b'L',
            height as isize,
            (n - i - 1) as isize,
            vector,
            1,
            (*tau.add(i)).cj(),
            a.offset((i + 1) as isize * (lda + 1)),
            lda,
            work.as_mut_ptr(),
        );
        *vector = beta;
    }
}

pub(crate) unsafe fn apply<T: Scalar>(
    side: u8,
    tr: u8,
    m: usize,
    n: usize,
    lo: usize,
    hi: usize,
    a: *const T,
    lda: isize,
    tau: *const T,
    c: *mut T,
    ldc: isize,
) {
    if m == 0 || n == 0 || hi <= lo + 1 {
        return;
    }
    let dim = hi - lo - 1;
    let mut basis = vec![T::z(); dim * dim];
    let mut coeff = vec![T::z(); dim];
    for j in 0..dim {
        coeff[j] = *tau.add(lo + j);
        for i in j + 1..dim {
            basis[i + j * dim] = *a.offset((lo + 1 + i) as isize + (lo + j) as isize * lda);
        }
    }
    let (rows, cols, offset) = if side == b'L' {
        (dim, n, (lo + 1) as isize)
    } else {
        (m, dim, (lo + 1) as isize * ldc)
    };
    crate::qr::apply_q(
        side,
        tr,
        rows,
        cols,
        dim,
        basis.as_ptr(),
        dim as isize,
        coeff.as_ptr(),
        c.offset(offset),
        ldc,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    fn reconstruction<T: ReflectorScalar>() {
        unsafe {
            let n = 4;
            let original: Vec<T> = (0..16)
                .map(|i| {
                    T::parts(
                        ((i * 7 + 3) % 17) as f64,
                        if T::COMPLEX { i as f64 / 5.0 } else { 0.0 },
                    )
                })
                .collect();
            let mut a = original.clone();
            let mut tau = vec![T::z(); n - 1];
            reduce(n, 0, n, a.as_mut_ptr(), n as isize, tau.as_mut_ptr());
            let mut q = vec![T::z(); n * n];
            for i in 0..n {
                q[i + i * n] = T::o();
            }
            apply(
                b'L',
                b'N',
                n,
                n,
                0,
                n,
                a.as_ptr(),
                n as isize,
                tau.as_ptr(),
                q.as_mut_ptr(),
                n as isize,
            );
            let mut h = a.clone();
            for j in 0..n {
                for i in j + 2..n {
                    h[i + j * n] = T::z();
                }
            }
            let mut temp = vec![T::z(); n * n];
            let mut result = temp.clone();
            crate::kernels::gemm(
                b'N',
                b'N',
                n as isize,
                n as isize,
                n as isize,
                T::o(),
                q.as_ptr(),
                n as isize,
                h.as_ptr(),
                n as isize,
                T::z(),
                temp.as_mut_ptr(),
                n as isize,
            );
            crate::kernels::gemm(
                b'N',
                b'C',
                n as isize,
                n as isize,
                n as isize,
                T::o(),
                temp.as_ptr(),
                n as isize,
                q.as_ptr(),
                n as isize,
                T::z(),
                result.as_mut_ptr(),
                n as isize,
            );
            for i in 0..n * n {
                assert!((result[i] - original[i]).magnitude() < 1e-10);
            }
        }
    }
    #[test]
    fn real_and_complex_similarity() {
        reconstruction::<f64>();
        reconstruction::<colmat::c64>();
    }
}

pub(crate) unsafe fn explicit<T: Scalar>(
    n: usize,
    lo: usize,
    hi: usize,
    a: *mut T,
    lda: isize,
    tau: *const T,
) {
    if n == 0 {
        return;
    }
    let mut factors = vec![T::z(); n * n];
    for j in lo..hi.saturating_sub(1) {
        for i in j + 2..hi {
            factors[i + j * n] = *a.offset(i as isize + j as isize * lda);
        }
    }
    for j in 0..n {
        for i in 0..n {
            *a.offset(i as isize + j as isize * lda) = if i == j { T::o() } else { T::z() };
        }
    }
    apply(
        b'L',
        b'N',
        n,
        n,
        lo,
        hi,
        factors.as_ptr(),
        n as isize,
        tau,
        a,
        lda,
    );
}
use colmat::{c32, c64};
use paste::paste;
macro_rules! factor_exports {
    ($p:ident,$t:ty,$i:ty,$s:tt)=>{paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p gehrd $s>](n:*const $i,ilo:*const $i,ihi:*const $i,a:*mut $t,lda:*const $i,tau:*mut $t,work:*mut $t,lwork:*const $i,info:*mut $i) {
            *info=0;let(n,lo,hi,la,lw)=(*n as isize,*ilo as isize,*ihi as isize,*lda as isize,*lwork as isize);
            for(arg,ok)in [(1,n>=0),(2,lo>=1&&lo<=n.max(1)),(3,hi>=lo.min(n)&&hi<=n),(5,la>=n.max(1)),(8,lw== -1||lw>=n.max(1))] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            *work=crate::lapack::workspace::<$t>(n.max(1));if lw== -1 {return;}
            reduce(n as usize,(lo-1) as usize,hi as usize,a,la,tau);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p gehd2 $s>](n:*const $i,ilo:*const $i,ihi:*const $i,a:*mut $t,lda:*const $i,tau:*mut $t,_work:*mut $t,info:*mut $i) {
            *info=0;let(n,lo,hi,la)=(*n as isize,*ilo as isize,*ihi as isize,*lda as isize);
            for(arg,ok)in [(1,n>=0),(2,lo>=1&&lo<=n.max(1)),(3,hi>=lo.min(n)&&hi<=n),(5,la>=n.max(1))] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            reduce(n as usize,(lo-1) as usize,hi as usize,a,la,tau);
        }
    }};
}
macro_rules! generate_exports {
    ($p:ident,$t:ty,$i:ty,$s:tt,$name:ident)=>{paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $name $s>](n:*const $i,ilo:*const $i,ihi:*const $i,a:*mut $t,lda:*const $i,tau:*const $t,work:*mut $t,lwork:*const $i,info:*mut $i) {
            *info=0;let(n,lo,hi,la,lw)=(*n as isize,*ilo as isize,*ihi as isize,*lda as isize,*lwork as isize);
            for(arg,ok)in [(1,n>=0),(2,lo>=1&&lo<=n.max(1)),(3,hi>=lo.min(n)&&hi<=n),(5,la>=n.max(1)),(8,lw== -1||lw>=(hi-lo).max(1))] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            *work=crate::lapack::workspace::<$t>((hi-lo).max(1));if lw== -1 {return;}
            explicit(n as usize,(lo-1) as usize,hi as usize,a,la,tau);
        }
    }};
}
factor_exports!(s, f32, i32, _);
factor_exports!(d, f64, i32, _);
factor_exports!(c, c32, i32, _);
factor_exports!(z, c64, i32, _);
factor_exports!(s, f32, i64, _64_);
factor_exports!(d, f64, i64, _64_);
factor_exports!(c, c32, i64, _64_);
factor_exports!(z, c64, i64, _64_);
generate_exports!(s, f32, i32, _, orghr);
generate_exports!(d, f64, i32, _, orghr);
generate_exports!(c, c32, i32, _, unghr);
generate_exports!(z, c64, i32, _, unghr);
generate_exports!(s, f32, i64, _64_, orghr);
generate_exports!(d, f64, i64, _64_, orghr);
generate_exports!(c, c32, i64, _64_, unghr);
generate_exports!(z, c64, i64, _64_, unghr);

use crate::kernels::flag;
macro_rules! apply_exports {
    ($p:ident,$t:ty,$i:ty,$s:tt,$name:ident)=>{paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $name $s>](side:*const u8,tr:*const u8,m:*const $i,n:*const $i,ilo:*const $i,ihi:*const $i,
            a:*const $t,lda:*const $i,tau:*const $t,c:*mut $t,ldc:*const $i,work:*mut $t,lwork:*const $i,info:*mut $i) {
            *info=0;let(side,tr)=(flag(side),flag(tr));
            let(m,n,lo,hi,la,lc,lw)=(*m as isize,*n as isize,*ilo as isize,*ihi as isize,*lda as isize,*ldc as isize,*lwork as isize);
            let order=if side==b'L' {m} else {n};let nw=if side==b'L' {n} else {m};
            for(arg,ok)in [(1,matches!(side,b'L'|b'R')),(2,tr==b'N'||tr==if <$t>::COMPLEX {b'C'} else {b'T'}),
                (3,m>=0),(4,n>=0),(5,lo>=1&&lo<=order.max(1)),(6,hi>=lo.min(order)&&hi<=order),(8,la>=order.max(1)),(11,lc>=m.max(1)),(13,lw== -1||lw>=nw.max(1))] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            *work=crate::lapack::workspace::<$t>(nw.max(1));if lw== -1 {return;}
            apply(side,tr,m as usize,n as usize,(lo-1) as usize,hi as usize,a,la,tau,c,lc);
        }
    }};
}
apply_exports!(s, f32, i32, _, ormhr);
apply_exports!(d, f64, i32, _, ormhr);
apply_exports!(c, c32, i32, _, unmhr);
apply_exports!(z, c64, i32, _, unmhr);
apply_exports!(s, f32, i64, _64_, ormhr);
apply_exports!(d, f64, i64, _64_, ormhr);
apply_exports!(c, c32, i64, _64_, unmhr);
apply_exports!(z, c64, i64, _64_, unmhr);
