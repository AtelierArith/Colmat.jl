//! Homogeneous generalized Schur primitives over the public native QZ APIs.
use crate::householder::ReflectorScalar;
use crate::kernels::Scalar;
use crate::schur::multiply;
use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::{c32, c64, Col, Mat};

pub(crate) trait QzScalar: ReflectorScalar {
    fn native(
        a: &mut Mat<Self>,
        b: &mut Mat<Self>,
        q: &mut Mat<Self>,
        z: &mut Mat<Self>,
    ) -> (Vec<Self>, Vec<f64>, Vec<Self>);
    fn reorder(
        a: &mut Mat<Self>,
        b: &mut Mat<Self>,
        q: &mut Mat<Self>,
        z: &mut Mat<Self>,
        from: usize,
        to: usize,
    ) -> bool;
}
macro_rules! real_qz {
    ($t:ty) => {
        impl QzScalar for $t {
            fn native(
                a: &mut Mat<Self>,
                b: &mut Mat<Self>,
                q: &mut Mat<Self>,
                z: &mut Mat<Self>,
            ) -> (Vec<Self>, Vec<f64>, Vec<Self>) {
                use colmat::linalg::gevd::qz_real::{hessenberg_to_qz, hessenberg_to_qz_scratch};
                let n = a.nrows();
                let mut ar = Col::<Self>::from_fn(n, |_| Self::NAN);
                let mut ai = ar.clone();
                let mut beta = ar.clone();
                crate::threading::with_par(|par| {
                    let params = colmat::auto!(Self);
                    let mut mem =
                        MemBuffer::new(hessenberg_to_qz_scratch::<Self>(n, par.shape(), params));
                    hessenberg_to_qz(
                        a.as_mut(),
                        b.as_mut(),
                        Some(q.as_mut()),
                        Some(z.as_mut()),
                        ar.as_mut(),
                        ai.as_mut(),
                        beta.as_mut(),
                        colmat::linalg::evd::ComputeEigenvectors::Yes,
                        par,
                        params,
                        MemStack::new(&mut mem),
                    );
                });
                (
                    (0..n).map(|i| ar[i]).collect(),
                    (0..n).map(|i| ai[i] as f64).collect(),
                    (0..n).map(|i| beta[i]).collect(),
                )
            }
            fn reorder(
                a: &mut Mat<Self>,
                b: &mut Mat<Self>,
                q: &mut Mat<Self>,
                z: &mut Mat<Self>,
                from: usize,
                to: usize,
            ) -> bool {
                use colmat::linalg::gevd::{reorder_schur_real, reorder_schur_scratch};
                crate::threading::with_par(|par| {
                    let mut mem = MemBuffer::new(reorder_schur_scratch::<Self>(a.nrows()));
                    reorder_schur_real(
                        a.as_mut(),
                        b.as_mut(),
                        Some(q.as_mut()),
                        Some(z.as_mut()),
                        from,
                        to,
                        par,
                        MemStack::new(&mut mem),
                    )
                    .is_ok()
                })
            }
        }
    };
}
macro_rules! complex_qz {
    ($t:ty) => {
        impl QzScalar for $t {
            fn native(
                a: &mut Mat<Self>,
                b: &mut Mat<Self>,
                q: &mut Mat<Self>,
                z: &mut Mat<Self>,
            ) -> (Vec<Self>, Vec<f64>, Vec<Self>) {
                use colmat::linalg::gevd::qz_cplx::{hessenberg_to_qz, hessenberg_to_qz_scratch};
                let n = a.nrows();
                let mut alpha = Col::<Self>::from_fn(n, |_| Self::real(f64::NAN));
                let mut beta = alpha.clone();
                crate::threading::with_par(|par| {
                    let params = colmat::auto!(Self);
                    let mut mem =
                        MemBuffer::new(hessenberg_to_qz_scratch::<Self>(n, par.shape(), params));
                    hessenberg_to_qz(
                        a.as_mut(),
                        b.as_mut(),
                        Some(q.as_mut()),
                        Some(z.as_mut()),
                        alpha.as_mut(),
                        beta.as_mut(),
                        colmat::linalg::evd::ComputeEigenvectors::Yes,
                        par,
                        params,
                        MemStack::new(&mut mem),
                    );
                });
                (
                    (0..n).map(|i| alpha[i]).collect(),
                    vec![0.0; n],
                    (0..n).map(|i| beta[i]).collect(),
                )
            }
            fn reorder(
                a: &mut Mat<Self>,
                b: &mut Mat<Self>,
                q: &mut Mat<Self>,
                z: &mut Mat<Self>,
                from: usize,
                to: usize,
            ) -> bool {
                use colmat::linalg::gevd::{reorder_schur_cplx, reorder_schur_scratch};
                crate::threading::with_par(|par| {
                    let mut mem = MemBuffer::new(reorder_schur_scratch::<Self>(a.nrows()));
                    reorder_schur_cplx(
                        a.as_mut(),
                        b.as_mut(),
                        Some(q.as_mut()),
                        Some(z.as_mut()),
                        from,
                        to,
                        par,
                        MemStack::new(&mut mem),
                    )
                    .is_ok()
                })
            }
        }
    };
}
real_qz!(f32);
real_qz!(f64);
complex_qz!(c32);
complex_qz!(c64);

fn identity<T: Scalar>(n: usize) -> Mat<T> {
    Mat::from_fn(n, n, |i, j| if i == j { T::o() } else { T::z() })
}
fn converged<T: QzScalar>(a: &Mat<T>, b: &Mat<T>, ar: &[T], ai: &[f64], beta: &[T]) -> bool {
    let n = a.nrows();
    for i in 0..n {
        if !ar[i].magnitude().is_finite() || !ai[i].is_finite() || !beta[i].magnitude().is_finite()
        {
            return false;
        }
        for j in 0..n {
            if !a[(i, j)].magnitude().is_finite() || !b[(i, j)].magnitude().is_finite() {
                return false;
            }
            if i > j && b[(i, j)] != T::z() {
                return false;
            }
            if i > j + (if T::COMPLEX { 0 } else { 1 }) && a[(i, j)] != T::z() {
                return false;
            }
        }
        if !T::COMPLEX && i >= 2 && a[(i, i - 1)] != T::z() && a[(i - 1, i - 2)] != T::z() {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    fn check<T: QzScalar>() {
        let n = 8;
        let original_a = Mat::from_fn(n, n, |i, j| {
            if i <= j + 1 {
                T::parts(
                    ((i * 5 + j * 3 + 1) % 13) as f64,
                    if T::COMPLEX {
                        ((i + j) % 3) as f64
                    } else {
                        0.0
                    },
                )
            } else {
                T::z()
            }
        });
        let original_b = Mat::from_fn(n, n, |i, j| {
            if i <= j {
                T::real(if i == j {
                    (i + 2) as f64
                } else {
                    ((i + j) % 4) as f64
                })
            } else {
                T::z()
            }
        });
        let mut a = original_a.clone();
        let mut b = original_b.clone();
        let mut q = identity::<T>(n);
        let mut z = identity::<T>(n);
        let (ar, ai, beta) = T::native(&mut a, &mut b, &mut q, &mut z);
        assert!(converged(&a, &b, &ar, &ai, &beta));
        for (original, result) in [(&original_a, &a), (&original_b, &b)] {
            let reconstructed = multiply(&multiply(&q, b'N', result, b'N'), b'N', &z, b'C');
            for j in 0..n {
                for i in 0..n {
                    assert!((reconstructed[(i, j)] - original[(i, j)]).magnitude() < 1e-10);
                }
            }
        }
    }
    #[test]
    fn native_real_and_complex_pencil() {
        check::<f64>();
        check::<c64>();
    }
}

fn embed<T: Scalar>(basis: &Mat<T>, n: usize, lo: usize, hi: usize) -> Mat<T> {
    Mat::from_fn(n, n, |i, j| {
        if i >= lo && i < hi && j >= lo && j < hi {
            basis[(i - lo, j - lo)]
        } else if i == j {
            T::o()
        } else {
            T::z()
        }
    })
}
fn positive_beta<T: QzScalar>(a: &mut Mat<T>, b: &mut Mat<T>, z: &mut Mat<T>) {
    let n = a.nrows();
    for j in 0..n {
        let diag = b[(j, j)];
        let magnitude = diag.magnitude();
        if magnitude > 0.0 && magnitude.is_finite() {
            let phase = T::parts(diag.re() / magnitude, -diag.im() / magnitude);
            for i in 0..n {
                a[(i, j)] = a[(i, j)] * phase;
                b[(i, j)] = b[(i, j)] * phase;
                z[(i, j)] = z[(i, j)] * phase;
            }
            b[(j, j)] = T::real(magnitude);
        }
    }
}
fn active<T: QzScalar>(
    a: &mut Mat<T>,
    b: &mut Mat<T>,
    q: &mut Mat<T>,
    z: &mut Mat<T>,
    lo: usize,
    hi: usize,
) -> (Vec<T>, Vec<f64>, Vec<T>, usize) {
    let n = a.nrows();
    let size = hi - lo;
    let mut ar: Vec<T> = (0..n).map(|i| a[(i, i)]).collect();
    let mut ai = vec![0.0; n];
    let mut beta: Vec<T> = (0..n).map(|i| b[(i, i)]).collect();
    let mut err = 0;
    if size > 0 {
        let mut h = Mat::from_fn(size, size, |i, j| {
            if i <= j + 1 {
                a[(lo + i, lo + j)]
            } else {
                T::z()
            }
        });
        let mut t = Mat::from_fn(size, size, |i, j| {
            if i <= j {
                b[(lo + i, lo + j)]
            } else {
                T::z()
            }
        });
        let max_a = (0..size)
            .flat_map(|j| (0..size).map(move |i| (i, j)))
            .map(|(i, j)| h[(i, j)].magnitude())
            .fold(0.0_f64, f64::max);
        let max_b = (0..size)
            .flat_map(|j| (0..size).map(move |i| (i, j)))
            .map(|(i, j)| t[(i, j)].magnitude())
            .fold(0.0_f64, f64::max);
        let sa = if max_a > 0.0 && max_a.is_finite() {
            max_a
        } else {
            1.0
        };
        let sb = if max_b > 0.0 && max_b.is_finite() {
            max_b
        } else {
            1.0
        };
        for j in 0..size {
            for i in 0..size {
                h[(i, j)] = h[(i, j)] / T::real(sa);
                t[(i, j)] = t[(i, j)] / T::real(sb);
            }
        }
        let mut u = identity::<T>(size);
        let mut v = identity::<T>(size);
        let (w, imag, be) = T::native(&mut h, &mut t, &mut u, &mut v);
        if !converged(&h, &t, &w, &imag, &be) {
            err = hi;
        }
        let left = embed(&u, n, lo, hi);
        let right = embed(&v, n, lo, hi);
        let updated_a = multiply(&left, b'C', &multiply(a, b'N', &right, b'N'), b'N');
        let updated_b = multiply(&left, b'C', &multiply(b, b'N', &right, b'N'), b'N');
        for j in 0..n {
            for i in 0..n {
                if i >= lo && i < hi && j >= lo && j < hi {
                    a[(i, j)] = h[(i - lo, j - lo)] * T::real(sa);
                    b[(i, j)] = t[(i - lo, j - lo)] * T::real(sb);
                } else if (i >= lo && i < hi) || (j >= lo && j < hi) {
                    a[(i, j)] = updated_a[(i, j)];
                    b[(i, j)] = updated_b[(i, j)];
                }
            }
        }
        *q = multiply(q, b'N', &left, b'N');
        *z = multiply(z, b'N', &right, b'N');
        for i in 0..size {
            ar[lo + i] = w[i] * T::real(sa);
            ai[lo + i] = imag[i] * sa;
            beta[lo + i] = be[i] * T::real(sb);
        }
    }
    positive_beta(a, b, z);
    for i in 0..n {
        if T::COMPLEX || i < lo || i >= hi {
            ar[i] = a[(i, i)];
            beta[i] = b[(i, i)];
        } else if beta[i].re() < 0.0 {
            ar[i] = -ar[i];
            ai[i] = -ai[i];
            beta[i] = -beta[i];
        }
    }
    (ar, ai, beta, err)
}
unsafe fn hgeqz<T: QzScalar>(
    n: usize,
    lo: usize,
    hi: usize,
    h: *mut T,
    ldh: isize,
    t: *mut T,
    ldt: isize,
    q: *mut T,
    ldq: isize,
    z: *mut T,
    ldz: isize,
    cq: u8,
    cz: u8,
) -> (Vec<T>, Vec<f64>, Vec<T>, usize) {
    let mut a = Mat::from_fn(n, n, |i, j| {
        if i <= j + 1 {
            *h.offset(i as isize + j as isize * ldh)
        } else {
            T::z()
        }
    });
    let mut b = Mat::from_fn(n, n, |i, j| {
        if i <= j {
            *t.offset(i as isize + j as isize * ldt)
        } else {
            T::z()
        }
    });
    let mut left = if cq == b'V' {
        crate::lapack::input(q, n, n, ldq)
    } else {
        identity::<T>(n)
    };
    let mut right = if cz == b'V' {
        crate::lapack::input(z, n, n, ldz)
    } else {
        identity::<T>(n)
    };
    let result = active(&mut a, &mut b, &mut left, &mut right, lo, hi);
    crate::lapack::output(&a, h, ldh);
    crate::lapack::output(&b, t, ldt);
    if cq != b'N' {
        crate::lapack::output(&left, q, ldq);
    }
    if cz != b'N' {
        crate::lapack::output(&right, z, ldz);
    }
    result
}
use crate::kernels::flag;
use paste::paste;
macro_rules! hgeqz_real_exports {
    ($p:ident,$t:ty,$i:ty,$s:tt)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p hgeqz $s>](job:*const u8,compq:*const u8,compz:*const u8,n:*const $i,ilo:*const $i,ihi:*const $i,h:*mut $t,ldh:*const $i,t:*mut $t,ldt:*const $i,alphar:*mut $t,alphai:*mut $t,beta:*mut $t,q:*mut $t,ldq:*const $i,z:*mut $t,ldz:*const $i,work:*mut $t,lwork:*const $i,info:*mut $i){
            *info=0;let(job,cq,cz)=(flag(job),flag(compq),flag(compz));let(n,lo,hi,lh,lt,lq,lz,lw)=(*n as isize,*ilo as isize,*ihi as isize,*ldh as isize,*ldt as isize,*ldq as isize,*ldz as isize,*lwork as isize);
            for(arg,ok)in[(1,matches!(job,b'E'|b'S')),(2,matches!(cq,b'N'|b'I'|b'V')),(3,matches!(cz,b'N'|b'I'|b'V')),(4,n>=0),(5,lo>=1&&lo<=n.max(1)),(6,hi>=lo.min(n)&&hi<=n),(8,lh>=n.max(1)),(10,lt>=n.max(1)),(15,lq>=1&&(cq==b'N'||lq>=n)),(17,lz>=1&&(cz==b'N'||lz>=n)),(19,lw== -1||lw>=n.max(1))]{if !ok{*info=-arg;crate::invalid(arg as i32);return;}}
            *work=crate::lapack::workspace::<$t>(n.max(1));if lw== -1{return;}
            let(ar,ai,be,err)=hgeqz(n as usize,(lo-1) as usize,hi as usize,h,lh,t,lt,q,lq,z,lz,cq,cz);
            for j in 0..n as usize{*alphar.add(j)=ar[j];*alphai.add(j)=ai[j] as $t;*beta.add(j)=be[j];}*info=err as $i;
        }
    }};
}
macro_rules! hgeqz_complex_exports {
    ($p:ident,$t:ty,$r:ty,$i:ty,$s:tt)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p hgeqz $s>](job:*const u8,compq:*const u8,compz:*const u8,n:*const $i,ilo:*const $i,ihi:*const $i,h:*mut $t,ldh:*const $i,t:*mut $t,ldt:*const $i,alpha:*mut $t,beta:*mut $t,q:*mut $t,ldq:*const $i,z:*mut $t,ldz:*const $i,work:*mut $t,lwork:*const $i,_rwork:*mut $r,info:*mut $i){
            *info=0;let(job,cq,cz)=(flag(job),flag(compq),flag(compz));let(n,lo,hi,lh,lt,lq,lz,lw)=(*n as isize,*ilo as isize,*ihi as isize,*ldh as isize,*ldt as isize,*ldq as isize,*ldz as isize,*lwork as isize);
            for(arg,ok)in[(1,matches!(job,b'E'|b'S')),(2,matches!(cq,b'N'|b'I'|b'V')),(3,matches!(cz,b'N'|b'I'|b'V')),(4,n>=0),(5,lo>=1&&lo<=n.max(1)),(6,hi>=lo.min(n)&&hi<=n),(8,lh>=n.max(1)),(10,lt>=n.max(1)),(14,lq>=1&&(cq==b'N'||lq>=n)),(16,lz>=1&&(cz==b'N'||lz>=n)),(18,lw== -1||lw>=n.max(1))]{if !ok{*info=-arg;crate::invalid(arg as i32);return;}}
            *work=crate::lapack::workspace::<$t>(n.max(1));if lw== -1{return;}
            let(ar,_,be,err)=hgeqz(n as usize,(lo-1) as usize,hi as usize,h,lh,t,lt,q,lq,z,lz,cq,cz);
            for j in 0..n as usize{*alpha.add(j)=ar[j];*beta.add(j)=be[j];}*info=err as $i;
        }
    }};
}
hgeqz_real_exports!(s, f32, i32, _);
hgeqz_real_exports!(d, f64, i32, _);
hgeqz_real_exports!(s, f32, i64, _64_);
hgeqz_real_exports!(d, f64, i64, _64_);
hgeqz_complex_exports!(c, c32, f32, i32, _);
hgeqz_complex_exports!(z, c64, f64, i32, _);
hgeqz_complex_exports!(c, c32, f32, i64, _64_);
hgeqz_complex_exports!(z, c64, f64, i64, _64_);

unsafe fn general<T: QzScalar>(
    n: usize,
    a: *mut T,
    lda: isize,
    b: *mut T,
    ldb: isize,
    q: *mut T,
    ldq: isize,
    z: *mut T,
    ldz: isize,
    want_q: bool,
    want_z: bool,
    sort: bool,
    mut select: impl FnMut(T, f64, T) -> bool,
) -> (Vec<T>, Vec<f64>, Vec<T>, Vec<bool>, usize, usize) {
    let mut h = crate::lapack::input(a, n, n, lda);
    let mut t = crate::lapack::input(b, n, n, ldb);
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for j in 0..n {
        for i in 0..n {
            sa = sa.max(h[(i, j)].magnitude());
            sb = sb.max(t[(i, j)].magnitude());
        }
    }
    if sa == 0.0 || !sa.is_finite() {
        sa = 1.0;
    }
    if sb == 0.0 || !sb.is_finite() {
        sb = 1.0;
    }
    for j in 0..n {
        for i in 0..n {
            h[(i, j)] = h[(i, j)] / T::real(sa);
            t[(i, j)] = t[(i, j)] / T::real(sb);
        }
    }
    let lh = h.as_ref().col_stride();
    let lt = t.as_ref().col_stride();
    let mut tau = vec![T::z(); n];
    crate::qr::standard_qr(n, n, t.as_mut().as_ptr_mut(), lt, tau.as_mut_ptr(), false);
    crate::qr::apply_q(
        b'L',
        b'C',
        n,
        n,
        n,
        t.as_ref().as_ptr(),
        lt,
        tau.as_ptr(),
        h.as_mut().as_ptr_mut(),
        lh,
    );
    let mut left = identity::<T>(n);
    let mut right = identity::<T>(n);
    let ll = left.as_ref().col_stride();
    let lr = right.as_ref().col_stride();
    crate::qr::apply_q(
        b'L',
        b'N',
        n,
        n,
        n,
        t.as_ref().as_ptr(),
        lt,
        tau.as_ptr(),
        left.as_mut().as_ptr_mut(),
        ll,
    );
    for j in 0..n {
        for i in j + 1..n {
            t[(i, j)] = T::z();
        }
    }
    crate::generalized_hessenberg::reduce(
        n,
        0,
        n,
        h.as_mut().as_ptr_mut(),
        lh,
        t.as_mut().as_ptr_mut(),
        lt,
        left.as_mut().as_ptr_mut(),
        ll,
        right.as_mut().as_ptr_mut(),
        lr,
        b'V',
        b'V',
    );
    let (mut ar, mut ai, mut beta, mut info) = active(&mut h, &mut t, &mut left, &mut right, 0, n);
    let mut chosen = vec![false; n];
    let mut sdim = 0;
    if sort && info == 0 {
        for i in 0..n {
            chosen[i] = select(ar[i] * T::real(sa), ai[i] * sa, beta[i] * T::real(sb));
        }
        let mut i = 0;
        while i < n {
            let size = if !T::COMPLEX && i + 1 < n && h[(i + 1, i)] != T::z() {
                2
            } else {
                1
            };
            let take = chosen[i..i + size].iter().any(|&flag| flag);
            for flag in &mut chosen[i..i + size] {
                *flag = take;
            }
            if take {
                if i != sdim {
                    if !T::reorder(&mut h, &mut t, &mut left, &mut right, i, sdim) {
                        info = n + 3;
                        break;
                    }
                    chosen[sdim..i + size].rotate_right(size);
                }
                sdim += size;
            }
            i += size;
        }
        // Re-evaluate homogeneous values from the reordered pencil, retaining its order.
        let previous_info = info;
        let result = active(&mut h, &mut t, &mut left, &mut right, 0, n);
        ar = result.0;
        ai = result.1;
        beta = result.2;
        info = if previous_info != 0 {
            previous_info
        } else {
            result.3
        };
        if info == 0 {
            let mut actual = vec![false; n];
            for i in 0..n {
                actual[i] = select(ar[i] * T::real(sa), ai[i] * sa, beta[i] * T::real(sb));
            }
            let mut i = 0;
            while i < n {
                let size = if !T::COMPLEX && i + 1 < n && h[(i + 1, i)] != T::z() {
                    2
                } else {
                    1
                };
                let take = actual[i..i + size].iter().any(|&flag| flag);
                for flag in &mut actual[i..i + size] {
                    *flag = take;
                }
                i += size;
            }
            if actual
                .iter()
                .enumerate()
                .any(|(i, &flag)| flag != (i < sdim))
            {
                info = n + 2;
            }
            sdim = actual.iter().filter(|&&flag| flag).count();
            chosen = actual;
        }
    }
    for j in 0..n {
        for i in 0..n {
            h[(i, j)] = h[(i, j)] * T::real(sa);
            t[(i, j)] = t[(i, j)] * T::real(sb);
        }
    }
    for i in 0..n {
        ar[i] = ar[i] * T::real(sa);
        ai[i] *= sa;
        beta[i] = beta[i] * T::real(sb);
    }
    crate::lapack::output(&h, a, lda);
    crate::lapack::output(&t, b, ldb);
    if want_q {
        crate::lapack::output(&left, q, ldq);
    }
    if want_z {
        crate::lapack::output(&right, z, ldz);
    }
    (ar, ai, beta, chosen, sdim, info)
}
macro_rules! gges_real_exports {
    ($p:ident,$family:ident,$t:ty,$i:ty,$s:tt,$blocked:expr)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $family $s>](jobvsl:*const u8,jobvsr:*const u8,sort:*const u8,select:Option<unsafe extern "C" fn(*const $t,*const $t,*const $t)->$i>,n:*const $i,a:*mut $t,lda:*const $i,b:*mut $t,ldb:*const $i,sdim:*mut $i,alphar:*mut $t,alphai:*mut $t,beta:*mut $t,vsl:*mut $t,ldvsl:*const $i,vsr:*mut $t,ldvsr:*const $i,work:*mut $t,lwork:*const $i,bwork:*mut $i,info:*mut $i){
            *info=0;let(jq,jz,sorting)=(flag(jobvsl),flag(jobvsr),flag(sort));let(n,la,lb,lq,lz,lw)=(*n as isize,*lda as isize,*ldb as isize,*ldvsl as isize,*ldvsr as isize,*lwork as isize);
            let minimum=if n==0{1}else if $blocked{6*n+16}else{(8*n).max(6*n+16)};
            for(arg,ok)in[(1,matches!(jq,b'N'|b'V')),(2,matches!(jz,b'N'|b'V')),(3,matches!(sorting,b'N'|b'S')),(5,n>=0),(7,la>=n.max(1)),(9,lb>=n.max(1)),(15,lq>=1&&(jq==b'N'||lq>=n)),(17,lz>=1&&(jz==b'N'||lz>=n)),(19,lw== -1||lw>=minimum)]{if !ok{*info=-arg;crate::invalid(arg as i32);return;}}
            *work=crate::lapack::workspace::<$t>(minimum);if lw== -1{return;}*sdim=0;
            let(ar,ai,be,chosen,count,err)=general(n as usize,a,la,b,lb,vsl,lq,vsr,lz,jq==b'V',jz==b'V',sorting==b'S',|re,im,beta|{let im=im as $t;select.expect("GGES SORT=S requires SELECT")(&re,&im,&beta)!=0});
            for i in 0..n as usize{*alphar.add(i)=ar[i];*alphai.add(i)=ai[i] as $t;*beta.add(i)=be[i];if sorting==b'S'{*bwork.add(i)=chosen[i] as $i;}}
            *sdim=count as $i;*info=err as $i;
        }
    }};
}
macro_rules! gges_complex_exports {
    ($p:ident,$family:ident,$t:ty,$r:ty,$i:ty,$s:tt)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $family $s>](jobvsl:*const u8,jobvsr:*const u8,sort:*const u8,select:Option<unsafe extern "C" fn(*const $t,*const $t)->$i>,n:*const $i,a:*mut $t,lda:*const $i,b:*mut $t,ldb:*const $i,sdim:*mut $i,alpha:*mut $t,beta:*mut $t,vsl:*mut $t,ldvsl:*const $i,vsr:*mut $t,ldvsr:*const $i,work:*mut $t,lwork:*const $i,_rwork:*mut $r,bwork:*mut $i,info:*mut $i){
            *info=0;let(jq,jz,sorting)=(flag(jobvsl),flag(jobvsr),flag(sort));let(n,la,lb,lq,lz,lw)=(*n as isize,*lda as isize,*ldb as isize,*ldvsl as isize,*ldvsr as isize,*lwork as isize);
            let minimum=(2*n).max(1);
            for(arg,ok)in[(1,matches!(jq,b'N'|b'V')),(2,matches!(jz,b'N'|b'V')),(3,matches!(sorting,b'N'|b'S')),(5,n>=0),(7,la>=n.max(1)),(9,lb>=n.max(1)),(14,lq>=1&&(jq==b'N'||lq>=n)),(16,lz>=1&&(jz==b'N'||lz>=n)),(18,lw== -1||lw>=minimum)]{if !ok{*info=-arg;crate::invalid(arg as i32);return;}}
            *work=crate::lapack::workspace::<$t>(minimum);if lw== -1{return;}*sdim=0;
            let(ar,_,be,chosen,count,err)=general(n as usize,a,la,b,lb,vsl,lq,vsr,lz,jq==b'V',jz==b'V',sorting==b'S',|alpha,_,beta|select.expect("GGES SORT=S requires SELECT")(&alpha,&beta)!=0);
            for i in 0..n as usize{*alpha.add(i)=ar[i];*beta.add(i)=be[i];if sorting==b'S'{*bwork.add(i)=chosen[i] as $i;}}
            *sdim=count as $i;*info=err as $i;
        }
    }};
}
gges_real_exports!(s, gges, f32, i32, _, false);
gges_real_exports!(s, gges3, f32, i32, _, true);
gges_real_exports!(s, gges, f32, i64, _64_, false);
gges_real_exports!(s, gges3, f32, i64, _64_, true);
gges_real_exports!(d, gges, f64, i32, _, false);
gges_real_exports!(d, gges3, f64, i32, _, true);
gges_real_exports!(d, gges, f64, i64, _64_, false);
gges_real_exports!(d, gges3, f64, i64, _64_, true);
gges_complex_exports!(c, gges, c32, f32, i32, _);
gges_complex_exports!(c, gges3, c32, f32, i32, _);
gges_complex_exports!(c, gges, c32, f32, i64, _64_);
gges_complex_exports!(c, gges3, c32, f32, i64, _64_);
gges_complex_exports!(z, gges, c64, f64, i32, _);
gges_complex_exports!(z, gges3, c64, f64, i32, _);
gges_complex_exports!(z, gges, c64, f64, i64, _64_);
gges_complex_exports!(z, gges3, c64, f64, i64, _64_);

macro_rules! gegs_real_exports {
    ($p:ident,$t:ty,$i:ty,$s:tt)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p gegs $s>](jobvsl:*const u8,jobvsr:*const u8,n:*const $i,a:*mut $t,lda:*const $i,b:*mut $t,ldb:*const $i,alphar:*mut $t,alphai:*mut $t,beta:*mut $t,vsl:*mut $t,ldvsl:*const $i,vsr:*mut $t,ldvsr:*const $i,work:*mut $t,lwork:*const $i,info:*mut $i){
            *info=0;let(jq,jz)=(flag(jobvsl),flag(jobvsr));let(n,la,lb,lq,lz,lw)=(*n as isize,*lda as isize,*ldb as isize,*ldvsl as isize,*ldvsr as isize,*lwork as isize);
            let minimum=(4*n).max(1);
            for(arg,ok)in[(1,matches!(jq,b'N'|b'V')),(2,matches!(jz,b'N'|b'V')),(3,n>=0),(5,la>=n.max(1)),(7,lb>=n.max(1)),(12,lq>=1&&(jq==b'N'||lq>=n)),(14,lz>=1&&(jz==b'N'||lz>=n)),(16,lw== -1||lw>=minimum)]{if !ok{*info=-arg;crate::invalid(arg as i32);return;}}
            *work=crate::lapack::workspace::<$t>(minimum);if lw== -1{return;}
            let(ar,ai,be,_,_,err)=general(n as usize,a,la,b,lb,vsl,lq,vsr,lz,jq==b'V',jz==b'V',false,|_,_,_|false);
            for i in 0..n as usize{*alphar.add(i)=ar[i];*alphai.add(i)=ai[i] as $t;*beta.add(i)=be[i];}*info=err as $i;
        }
    }};
}
macro_rules! gegs_complex_exports {
    ($p:ident,$t:ty,$r:ty,$i:ty,$s:tt)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p gegs $s>](jobvsl:*const u8,jobvsr:*const u8,n:*const $i,a:*mut $t,lda:*const $i,b:*mut $t,ldb:*const $i,alpha:*mut $t,beta:*mut $t,vsl:*mut $t,ldvsl:*const $i,vsr:*mut $t,ldvsr:*const $i,work:*mut $t,lwork:*const $i,_rwork:*mut $r,info:*mut $i){
            *info=0;let(jq,jz)=(flag(jobvsl),flag(jobvsr));let(n,la,lb,lq,lz,lw)=(*n as isize,*lda as isize,*ldb as isize,*ldvsl as isize,*ldvsr as isize,*lwork as isize);
            let minimum=(2*n).max(1);
            for(arg,ok)in[(1,matches!(jq,b'N'|b'V')),(2,matches!(jz,b'N'|b'V')),(3,n>=0),(5,la>=n.max(1)),(7,lb>=n.max(1)),(11,lq>=1&&(jq==b'N'||lq>=n)),(13,lz>=1&&(jz==b'N'||lz>=n)),(15,lw== -1||lw>=minimum)]{if !ok{*info=-arg;crate::invalid(arg as i32);return;}}
            *work=crate::lapack::workspace::<$t>(minimum);if lw== -1{return;}
            let(ar,_,be,_,_,err)=general(n as usize,a,la,b,lb,vsl,lq,vsr,lz,jq==b'V',jz==b'V',false,|_,_,_|false);
            for i in 0..n as usize{*alpha.add(i)=ar[i];*beta.add(i)=be[i];}*info=err as $i;
        }
    }};
}
gegs_real_exports!(s, f32, i32, _);
gegs_real_exports!(d, f64, i32, _);
gegs_real_exports!(s, f32, i64, _64_);
gegs_real_exports!(d, f64, i64, _64_);
gegs_complex_exports!(c, c32, f32, i32, _);
gegs_complex_exports!(z, c64, f64, i32, _);
gegs_complex_exports!(c, c32, f32, i64, _64_);
gegs_complex_exports!(z, c64, f64, i64, _64_);
