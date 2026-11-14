//! Compact-WY QR adapters over the native public QR factorization.
use crate::householder::ReflectorScalar;
use crate::kernels::Scalar;
use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::{Mat, MatRef};

// Factor one active column at a time so an exactly zero column still occupies
// its LAPACK diagonal position. The native rank-revealing panel may otherwise
// pack subsequent nonzero columns into that position.
pub(crate) unsafe fn compact_qr<T: ReflectorScalar>(
    m: usize,
    n: usize,
    nb: usize,
    a: *mut T,
    lda: isize,
    t: *mut T,
    ldt: isize,
) {
    use colmat::linalg::qr::no_pivoting::factor::{qr_in_place, qr_in_place_scratch};
    let k = m.min(n);
    if k == 0 {
        return;
    }
    let mut tau = vec![T::z(); k];
    for j in 0..k {
        let height = m - j;
        let column = a.offset(j as isize * (lda + 1));
        let mut panel = MatRef::from_raw_parts(column, height, 1, 1, lda).to_owned();
        // Scale the active column before native reflector generation, whose
        // tiny-tail branch deliberately treats subnormals as zero.
        let mut scale = 0.0_f64;
        for i in 0..height {
            let z = panel[(i, 0)];
            scale = scale.max(z.re().abs()).max(z.im().abs());
        }
        if scale > 0.0 && scale.is_finite() {
            for i in 0..height {
                let z = panel[(i, 0)];
                panel[(i, 0)] = T::parts(z.re() / scale, z.im() / scale);
            }
        } else {
            scale = 1.0;
        }
        let mut coeff = Mat::<T>::zeros(1, 1);
        crate::threading::with_par(|par| {
            let mut mem = MemBuffer::new(qr_in_place_scratch::<T>(
                height,
                1,
                1,
                par.shape(),
                Default::default(),
            ));
            qr_in_place(
                panel.as_mut(),
                coeff.as_mut(),
                par,
                MemStack::new(&mut mem),
                Default::default(),
            );
        });
        let beta = panel[(0, 0)];
        panel[(0, 0)] = T::parts(beta.re() * scale, beta.im() * scale);
        // Native H = I - v v^H / coeff; LAPACK H = I - tau v v^H.
        tau[j] = T::real(1.0 / coeff[(0, 0)].re());
        for i in 0..height {
            *column.add(i) = panel[(i, 0)];
        }
        if j + 1 < n && tau[j] != T::z() {
            panel[(0, 0)] = T::o();
            let mut work = vec![T::z(); n - j - 1];
            crate::householder::apply(
                b'L',
                height as isize,
                (n - j - 1) as isize,
                panel.as_ref().as_ptr(),
                1,
                tau[j].cj(),
                a.offset(j as isize + (j + 1) as isize * lda),
                lda,
                work.as_mut_ptr(),
            );
        }
    }
    for j in (0..k).step_by(nb) {
        let block = nb.min(k - j);
        crate::householder::factor(
            b'F',
            b'C',
            m - j,
            block,
            a.offset(j as isize * (lda + 1)),
            lda,
            tau.as_ptr().add(j),
            t.offset(j as isize * ldt),
            ldt,
        );
    }
}

unsafe fn multiply_q<T: Scalar>(
    side: u8,
    tr: u8,
    m: usize,
    n: usize,
    k: usize,
    nb: usize,
    v: *const T,
    ldv: isize,
    t: *const T,
    ldt: isize,
    c: *mut T,
    ldc: isize,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    let mut starts: Vec<usize> = (0..k).step_by(nb).collect();
    if (side == b'L' && tr == b'N') || (side == b'R' && tr != b'N') {
        starts.reverse();
    }
    for j in starts {
        let block = nb.min(k - j);
        let (rows, cols, offset) = if side == b'L' {
            (m - j, n, j as isize)
        } else {
            (m, n - j, j as isize * ldc)
        };
        crate::householder::apply_block(
            side,
            tr,
            b'F',
            b'C',
            rows,
            cols,
            block,
            v.offset(j as isize * (ldv + 1)),
            ldv,
            t.offset(j as isize * ldt),
            ldt,
            c.offset(offset),
            ldc,
        );
    }
}
use crate::kernels::flag;
use colmat::{c32, c64};
use paste::paste;
macro_rules! exports {
    ($p:ident,$t:ty,$i:ty,$s:tt)=>{paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p geqrt $s>](m:*const $i,n:*const $i,nb:*const $i,a:*mut $t,
            lda:*const $i,t:*mut $t,ldt:*const $i,_work:*mut $t,info:*mut $i) {
            *info=0;
            let (m,n,nb,la,lt)=(*m as isize,*n as isize,*nb as isize,*lda as isize,*ldt as isize);
            for (arg,ok) in [(1,m>=0),(2,n>=0),(3,nb>=1 && (nb<=m.min(n)||m.min(n)==0)),(5,la>=m.max(1)),(7,lt>=nb)] {
                if !ok {*info=-arg;return;}
            }
            compact_qr(m as usize,n as usize,nb as usize,a,la,t,lt);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p geqrt2 $s>](m:*const $i,n:*const $i,a:*mut $t,lda:*const $i,
            t:*mut $t,ldt:*const $i,info:*mut $i) {
            *info=0;
            let (m,n,la,lt)=(*m as isize,*n as isize,*lda as isize,*ldt as isize);
            for (arg,ok) in [(1,m>=n && m>=0),(2,n>=0),(4,la>=m.max(1)),(6,lt>=n.max(1))] {
                if !ok {*info=-arg;return;}
            }
            compact_qr(m as usize,n as usize,n.max(1) as usize,a,la,t,lt);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p geqrt3 $s>](m:*const $i,n:*const $i,a:*mut $t,lda:*const $i,
            t:*mut $t,ldt:*const $i,info:*mut $i) {
            [<$p geqrt2 $s>](m,n,a,lda,t,ldt,info);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p gemqrt $s>](side:*const u8,tr:*const u8,m:*const $i,n:*const $i,k:*const $i,
            nb:*const $i,v:*const $t,ldv:*const $i,t:*const $t,ldt:*const $i,c:*mut $t,ldc:*const $i,
            _work:*mut $t,info:*mut $i) {
            *info=0;
            let (side,tr)=(flag(side),flag(tr));
            let (m,n,k,nb,lv,lt,lc)=(*m as isize,*n as isize,*k as isize,*nb as isize,*ldv as isize,*ldt as isize,*ldc as isize);
            let q=if side==b'L' {m} else {n};
            for (arg,ok) in [(1,matches!(side,b'L'|b'R')),(2,tr==b'N'||tr==if <$t>::COMPLEX {b'C'} else {b'T'}),
                (3,m>=0),(4,n>=0),(5,k>=0 && k<=q),(6,nb>=1 && (nb<=k||k==0)),(8,lv>=q.max(1)),(10,lt>=nb),(12,lc>=m.max(1))] {
                if !ok {*info=-arg;return;}
            }
            multiply_q(side,tr,m as usize,n as usize,k as usize,nb as usize,v,lv,t,lt,c,lc);
        }
    }};
}
exports!(s, f32, i32, _);
exports!(d, f64, i32, _);
exports!(c, c32, i32, _);
exports!(z, c64, i32, _);
exports!(s, f32, i64, _64_);
exports!(d, f64, i64, _64_);
exports!(c, c32, i64, _64_);
exports!(z, c64, i64, _64_);

// Standard A/TAU storage shares the compact factorization's block-size-one path.
pub(crate) unsafe fn standard_qr<T: ReflectorScalar>(
    m: usize,
    n: usize,
    a: *mut T,
    lda: isize,
    tau: *mut T,
    positive: bool,
) {
    let k = m.min(n);
    if !positive && !T::COMPLEX {
        compact_qr(m, n, 1, a, lda, tau, 1);
        return;
    }
    let mut work = vec![T::z(); n];
    for j in 0..k {
        let p = a.offset(j as isize * (lda + 1));
        crate::householder::generate((m - j) as isize, p, p.add(1), 1, tau.add(j), positive);
        if j + 1 == n {
            continue;
        }
        let beta = *p;
        *p = T::o();
        crate::householder::apply(
            b'L',
            (m - j) as isize,
            (n - j - 1) as isize,
            p,
            1,
            (*tau.add(j)).cj(),
            a.offset(j as isize + (j + 1) as isize * lda),
            lda,
            work.as_mut_ptr(),
        );
        *p = beta;
    }
}

pub(crate) unsafe fn apply_q<T: Scalar>(
    side: u8,
    tr: u8,
    m: usize,
    n: usize,
    k: usize,
    a: *const T,
    lda: isize,
    tau: *const T,
    c: *mut T,
    ldc: isize,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    let order = if side == b'L' { m } else { n };
    let mut indices: Vec<usize> = (0..k).collect();
    if (side == b'L' && tr == b'N') || (side == b'R' && tr != b'N') {
        indices.reverse();
    }
    let mut work = vec![T::z(); m.max(n)];
    for j in indices {
        let tau = if tr == b'N' {
            *tau.add(j)
        } else {
            (*tau.add(j)).cj()
        };
        if tau == T::z() {
            continue;
        }
        let mut v = vec![T::o(); order - j];
        for i in 1..v.len() {
            v[i] = *a.offset((j + i) as isize + j as isize * lda);
        }
        let (rows, cols, offset) = if side == b'L' {
            (m - j, n, j as isize)
        } else {
            (m, n - j, j as isize * ldc)
        };
        crate::householder::apply(
            side,
            rows as isize,
            cols as isize,
            v.as_ptr(),
            1,
            tau,
            c.offset(offset),
            ldc,
            work.as_mut_ptr(),
        );
    }
}

macro_rules! factor_exports {
    ($p:ident,$t:ty,$i:ty,$s:tt,$name:ident,$positive:expr,blocked)=>{paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $name $s>](m:*const $i,n:*const $i,a:*mut $t,lda:*const $i,
            tau:*mut $t,work:*mut $t,lwork:*const $i,info:*mut $i) {
            *info=0;
            let (m,n,la,lw)=(*m as isize,*n as isize,*lda as isize,*lwork as isize);
            let query=lw== -1;
            for (arg,ok) in [(1,m>=0),(2,n>=0),(4,la>=m.max(1)),(7,query||lw>=n.max(1))] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            *work=crate::lapack::workspace::<$t>(n.max(1));
            if query {return;}
            standard_qr(m as usize,n as usize,a,la,tau,$positive);
        }
    }};
    ($p:ident,$t:ty,$i:ty,$s:tt,$name:ident,$positive:expr,unblocked)=>{paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $name $s>](m:*const $i,n:*const $i,a:*mut $t,lda:*const $i,
            tau:*mut $t,_work:*mut $t,info:*mut $i) {
            *info=0;
            let (m,n,la)=(*m as isize,*n as isize,*lda as isize);
            for (arg,ok) in [(1,m>=0),(2,n>=0),(4,la>=m.max(1))] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            standard_qr(m as usize,n as usize,a,la,tau,$positive);
        }
    }};
}
macro_rules! standard_exports {
    ($p:ident,$t:ty,$i:ty,$s:tt) => {
        factor_exports!($p, $t, $i, $s, geqrf, false, blocked);
        factor_exports!($p, $t, $i, $s, geqrfp, true, blocked);
        factor_exports!($p, $t, $i, $s, geqr2, false, unblocked);
        factor_exports!($p, $t, $i, $s, geqr2p, true, unblocked);
    };
}
standard_exports!(s, f32, i32, _);
standard_exports!(d, f64, i32, _);
standard_exports!(c, c32, i32, _);
standard_exports!(z, c64, i32, _);
standard_exports!(s, f32, i64, _64_);
standard_exports!(d, f64, i64, _64_);
standard_exports!(c, c32, i64, _64_);
standard_exports!(z, c64, i64, _64_);

#[cfg(test)]
mod standard_tests {
    use super::*;
    #[test]
    fn standard_and_positive_qr_reconstruct() {
        for positive in [false, true] {
            let original = [1.0_f64, 2.0, -3.0, 4.0, -1.0, 2.0];
            let mut a = original;
            let mut tau = [0.0; 2];
            unsafe {
                standard_qr(3, 2, a.as_mut_ptr(), 3, tau.as_mut_ptr(), positive);
                if positive {
                    assert!(a[0] >= 0.0 && a[4] >= 0.0);
                }
                let mut r = [a[0], 0.0, 0.0, a[3], a[4], 0.0];
                apply_q(
                    b'L',
                    b'N',
                    3,
                    2,
                    2,
                    a.as_ptr(),
                    3,
                    tau.as_ptr(),
                    r.as_mut_ptr(),
                    3,
                );
                for i in 0..6 {
                    assert!((r[i] - original[i]).abs() < 1e-12);
                }
            }
        }
    }
    #[test]
    fn query_does_not_read_matrix() {
        unsafe {
            let mut info = 99_i32;
            let mut work = 0.0_f64;
            dgeqrf_(
                &3,
                &2,
                std::ptr::null_mut(),
                &3,
                std::ptr::null_mut(),
                &mut work,
                &-1,
                &mut info,
            );
            assert_eq!(info, 0);
            assert_eq!(work, 2.0);
            dgeqrf_(
                &3,
                &2,
                std::ptr::null_mut(),
                &3,
                std::ptr::null_mut(),
                &mut work,
                &1,
                &mut info,
            );
            assert_eq!(info, -7);
        }
    }
}

unsafe fn explicit_q<T: Scalar>(
    m: usize,
    n: usize,
    k: usize,
    a: *mut T,
    lda: isize,
    tau: *const T,
) {
    if n == 0 {
        return;
    }
    let mut reflectors = vec![T::z(); m * k];
    for j in 0..k {
        for i in j + 1..m {
            reflectors[i + j * m] = *a.offset(i as isize + j as isize * lda);
        }
    }
    for j in 0..n {
        for i in 0..m {
            *a.offset(i as isize + j as isize * lda) = if i == j { T::o() } else { T::z() };
        }
    }
    apply_q(
        b'L',
        b'N',
        m,
        n,
        k,
        reflectors.as_ptr(),
        m as isize,
        tau,
        a,
        lda,
    );
}
macro_rules! q_generate_exports {
    ($p:ident,$t:ty,$i:ty,$s:tt,$blocked:ident,$unblocked:ident)=>{paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $blocked $s>](m:*const $i,n:*const $i,k:*const $i,a:*mut $t,lda:*const $i,
            tau:*const $t,work:*mut $t,lwork:*const $i,info:*mut $i) {
            *info=0;
            let (m,n,k,la,lw)=(*m as isize,*n as isize,*k as isize,*lda as isize,*lwork as isize);
            let query=lw== -1;
            for (arg,ok) in [(1,m>=0),(2,n>=0 && n<=m),(3,k>=0 && k<=n),(5,la>=m.max(1)),(8,query||lw>=n.max(1))] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            *work=crate::lapack::workspace::<$t>(n.max(1));
            if query {return;}
            explicit_q(m as usize,n as usize,k as usize,a,la,tau);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $unblocked $s>](m:*const $i,n:*const $i,k:*const $i,a:*mut $t,lda:*const $i,
            tau:*const $t,_work:*mut $t,info:*mut $i) {
            *info=0;
            let (m,n,k,la)=(*m as isize,*n as isize,*k as isize,*lda as isize);
            for (arg,ok) in [(1,m>=0),(2,n>=0 && n<=m),(3,k>=0 && k<=n),(5,la>=m.max(1))] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            explicit_q(m as usize,n as usize,k as usize,a,la,tau);
        }
    }};
}
q_generate_exports!(s, f32, i32, _, orgqr, org2r);
q_generate_exports!(d, f64, i32, _, orgqr, org2r);
q_generate_exports!(c, c32, i32, _, ungqr, ung2r);
q_generate_exports!(z, c64, i32, _, ungqr, ung2r);
q_generate_exports!(s, f32, i64, _64_, orgqr, org2r);
q_generate_exports!(d, f64, i64, _64_, orgqr, org2r);
q_generate_exports!(c, c32, i64, _64_, ungqr, ung2r);
q_generate_exports!(z, c64, i64, _64_, ungqr, ung2r);

macro_rules! q_apply_exports {
    ($p:ident,$t:ty,$i:ty,$s:tt,$blocked:ident,$unblocked:ident)=>{paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $blocked $s>](side:*const u8,tr:*const u8,m:*const $i,n:*const $i,k:*const $i,
            a:*const $t,lda:*const $i,tau:*const $t,c:*mut $t,ldc:*const $i,work:*mut $t,lwork:*const $i,info:*mut $i) {
            *info=0;
            let (side,tr)=(flag(side),flag(tr));
            let (m,n,k,la,lc,lw)=(*m as isize,*n as isize,*k as isize,*lda as isize,*ldc as isize,*lwork as isize);
            let order=if side==b'L' {m} else {n};
            let nw=if side==b'L' {n} else {m};
            let query=lw== -1;
            for (arg,ok) in [(1,matches!(side,b'L'|b'R')),(2,tr==b'N'||tr==if <$t>::COMPLEX {b'C'} else {b'T'}),
                (3,m>=0),(4,n>=0),(5,k>=0 && k<=order),(7,la>=order.max(1)),(10,lc>=m.max(1)),(12,query||lw>=nw.max(1))] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            *work=crate::lapack::workspace::<$t>(nw.max(1));
            if query {return;}
            apply_q(side,tr,m as usize,n as usize,k as usize,a,la,tau,c,lc);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $unblocked $s>](side:*const u8,tr:*const u8,m:*const $i,n:*const $i,k:*const $i,
            a:*const $t,lda:*const $i,tau:*const $t,c:*mut $t,ldc:*const $i,_work:*mut $t,info:*mut $i) {
            let len=if flag(side)==b'L' {*n} else {*m};
            let len=len.max(1);
            let mut work=<$t>::z();
            [<$p $blocked $s>](side,tr,m,n,k,a,lda,tau,c,ldc,&mut work,&len,info);
        }
    }};
}
q_apply_exports!(s, f32, i32, _, ormqr, orm2r);
q_apply_exports!(d, f64, i32, _, ormqr, orm2r);
q_apply_exports!(c, c32, i32, _, unmqr, unm2r);
q_apply_exports!(z, c64, i32, _, unmqr, unm2r);
q_apply_exports!(s, f32, i64, _64_, ormqr, orm2r);
q_apply_exports!(d, f64, i64, _64_, ormqr, orm2r);
q_apply_exports!(c, c32, i64, _64_, unmqr, unm2r);
q_apply_exports!(z, c64, i64, _64_, unmqr, unm2r);

macro_rules! solve_exports {
    ($p:ident,$t:ty,$i:ty,$s:tt)=>{paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p geqrs $s>](m:*const $i,n:*const $i,nrhs:*const $i,a:*const $t,lda:*const $i,
            tau:*const $t,b:*mut $t,ldb:*const $i,_work:*mut $t,lwork:*const $i,info:*mut $i) {
            *info=0;
            let (m,n,r,la,lb,lw)=(*m as isize,*n as isize,*nrhs as isize,*lda as isize,*ldb as isize,*lwork as isize);
            for (arg,ok) in [(1,m>=0),(2,n>=0 && n<=m),(3,r>=0),(5,la>=m.max(1)),(8,lb>=m.max(1)),
                (10,lw>=1 && (m==0||n==0||lw>=r))] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            if m==0 || n==0 || r==0 {return;}
            apply_q(b'L',if <$t>::COMPLEX {b'C'} else {b'T'},m as usize,r as usize,n as usize,a,la,tau,b,lb);
            crate::kernels::trmatrix(b'L',b'U',b'N',b'N',n,r,<$t>::o(),a,la,b,lb,true);
        }
    }};
}
solve_exports!(s, f32, i32, _);
solve_exports!(d, f64, i32, _);
solve_exports!(c, c32, i32, _);
solve_exports!(z, c64, i32, _);
solve_exports!(s, f32, i64, _64_);
solve_exports!(d, f64, i64, _64_);
solve_exports!(c, c32, i64, _64_);
solve_exports!(z, c64, i64, _64_);
