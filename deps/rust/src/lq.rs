//! LQ storage adapters via the adjoint QR identity A = R^H Q^H.
use crate::householder::ReflectorScalar;
use crate::kernels::Scalar;

pub(crate) unsafe fn standard_lq<T: ReflectorScalar>(
    m: usize,
    n: usize,
    a: *mut T,
    lda: isize,
    tau: *mut T,
) {
    if m == 0 || n == 0 {
        return;
    }
    let mut adjoint = vec![T::z(); m * n];
    for j in 0..n {
        for i in 0..m {
            adjoint[j + i * n] = (*a.offset(i as isize + j as isize * lda)).cj();
        }
    }
    crate::qr::standard_qr(n, m, adjoint.as_mut_ptr(), n as isize, tau, false);
    for j in 0..n {
        for i in 0..m {
            *a.offset(i as isize + j as isize * lda) = adjoint[j + i * n].cj();
        }
    }
}

pub(crate) unsafe fn apply_lq<T: Scalar>(
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
    let mut reflectors = vec![T::z(); order * k];
    for j in 0..k {
        for i in j + 1..order {
            reflectors[i + j * order] = (*a.offset(j as isize + i as isize * lda)).cj();
        }
    }
    crate::qr::apply_q(
        side,
        if tr == b'N' {
            if T::COMPLEX {
                b'C'
            } else {
                b'T'
            }
        } else {
            b'N'
        },
        m,
        n,
        k,
        reflectors.as_ptr(),
        order as isize,
        tau,
        c,
        ldc,
    );
}

pub(crate) unsafe fn explicit_lq<T: Scalar>(
    m: usize,
    n: usize,
    k: usize,
    a: *mut T,
    lda: isize,
    tau: *const T,
) {
    if m == 0 || n == 0 {
        return;
    }
    let mut factors = vec![T::z(); k * n];
    for i in 0..k {
        for j in i + 1..n {
            factors[i + j * k] = *a.offset(i as isize + j as isize * lda);
        }
    }
    for j in 0..n {
        for i in 0..m {
            *a.offset(i as isize + j as isize * lda) = if i == j { T::o() } else { T::z() };
        }
    }
    apply_lq(
        b'R',
        b'N',
        m,
        n,
        k,
        factors.as_ptr(),
        k as isize,
        tau,
        a,
        lda,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn adjoint_qr_lq_reconstructs() {
        unsafe {
            let original = [1.0_f64, 2.0, 3.0, -1.0, 4.0, 2.0];
            let mut a = original;
            let mut tau = [0.0; 2];
            standard_lq(2, 3, a.as_mut_ptr(), 2, tau.as_mut_ptr());
            let mut q = [0.0; 9];
            for i in 0..3 {
                q[i + i * 3] = 1.0;
            }
            apply_lq(
                b'R',
                b'N',
                3,
                3,
                2,
                a.as_ptr(),
                2,
                tau.as_ptr(),
                q.as_mut_ptr(),
                3,
            );
            for j in 0..3 {
                for i in 0..2 {
                    let mut value = 0.0;
                    for k in 0..=i {
                        value += a[i + k * 2] * q[k + j * 3];
                    }
                    assert!((value - original[i + j * 2]).abs() < 1e-12);
                }
            }
            let mut explicit = [0.0; 6];
            explicit.copy_from_slice(&a);
            explicit_lq(2, 3, 2, explicit.as_mut_ptr(), 2, tau.as_ptr());
            for j in 0..3 {
                for i in 0..2 {
                    assert!((explicit[i + j * 2] - q[i + j * 3]).abs() < 1e-12);
                }
            }
        }
    }
}

use colmat::{c32, c64};
use paste::paste;
macro_rules! factor_exports {
    ($p:ident,$t:ty,$i:ty,$s:tt)=>{paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p gelqf $s>](m:*const $i,n:*const $i,a:*mut $t,lda:*const $i,tau:*mut $t,work:*mut $t,lwork:*const $i,info:*mut $i) {
            *info=0;let(m,n,la,lw)=(*m as isize,*n as isize,*lda as isize,*lwork as isize);
            let minimum=if n==0 {1}else{m.max(1)};
            for(arg,ok)in [(1,m>=0),(2,n>=0),(4,la>=m.max(1)),(7,lw== -1||lw>=minimum)] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            *work=crate::lapack::workspace::<$t>(if m==0||n==0 {1}else{minimum});if lw== -1 {return;}
            standard_lq(m as usize,n as usize,a,la,tau);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p gelq2 $s>](m:*const $i,n:*const $i,a:*mut $t,lda:*const $i,tau:*mut $t,_work:*mut $t,info:*mut $i) {
            *info=0;let(m,n,la)=(*m as isize,*n as isize,*lda as isize);
            for(arg,ok)in [(1,m>=0),(2,n>=0),(4,la>=m.max(1))] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            standard_lq(m as usize,n as usize,a,la,tau);
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

use crate::kernels::flag;
macro_rules! lq_generate_exports {
    ($p:ident,$t:ty,$i:ty,$s:tt,$blocked:ident,$unblocked:ident)=>{paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $blocked $s>](m:*const $i,n:*const $i,k:*const $i,a:*mut $t,lda:*const $i,
            tau:*const $t,work:*mut $t,lwork:*const $i,info:*mut $i) {
            *info=0;
            let (m,n,k,la,lw)=(*m as isize,*n as isize,*k as isize,*lda as isize,*lwork as isize);
            let query=lw== -1;
            for (arg,ok) in [(1,m>=0),(2,n>=m),(3,k>=0 && k<=m),(5,la>=m.max(1)),(8,query||lw>=m.max(1))] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            *work=crate::lapack::workspace::<$t>(m.max(1));
            if query {return;}
            explicit_lq(m as usize,n as usize,k as usize,a,la,tau);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $unblocked $s>](m:*const $i,n:*const $i,k:*const $i,a:*mut $t,lda:*const $i,
            tau:*const $t,_work:*mut $t,info:*mut $i) {
            *info=0;
            let (m,n,k,la)=(*m as isize,*n as isize,*k as isize,*lda as isize);
            for (arg,ok) in [(1,m>=0),(2,n>=m),(3,k>=0 && k<=m),(5,la>=m.max(1))] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            explicit_lq(m as usize,n as usize,k as usize,a,la,tau);
        }
    }};
}
lq_generate_exports!(s, f32, i32, _, orglq, orgl2);
lq_generate_exports!(d, f64, i32, _, orglq, orgl2);
lq_generate_exports!(c, c32, i32, _, unglq, ungl2);
lq_generate_exports!(z, c64, i32, _, unglq, ungl2);
lq_generate_exports!(s, f32, i64, _64_, orglq, orgl2);
lq_generate_exports!(d, f64, i64, _64_, orglq, orgl2);
lq_generate_exports!(c, c32, i64, _64_, unglq, ungl2);
lq_generate_exports!(z, c64, i64, _64_, unglq, ungl2);

macro_rules! lq_apply_exports {
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
                (3,m>=0),(4,n>=0),(5,k>=0 && k<=order),(7,la>=k.max(1)),(10,lc>=m.max(1)),(12,query||lw>=nw.max(1))] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            *work=crate::lapack::workspace::<$t>(nw.max(1));
            if query {return;}
            apply_lq(side,tr,m as usize,n as usize,k as usize,a,la,tau,c,lc);
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
lq_apply_exports!(s, f32, i32, _, ormlq, orml2);
lq_apply_exports!(d, f64, i32, _, ormlq, orml2);
lq_apply_exports!(c, c32, i32, _, unmlq, unml2);
lq_apply_exports!(z, c64, i32, _, unmlq, unml2);
lq_apply_exports!(s, f32, i64, _64_, ormlq, orml2);
lq_apply_exports!(d, f64, i64, _64_, ormlq, orml2);
lq_apply_exports!(c, c32, i64, _64_, unmlq, unml2);
lq_apply_exports!(z, c64, i64, _64_, unmlq, unml2);

macro_rules! solve_exports {
    ($p:ident,$t:ty,$i:ty,$s:tt)=>{paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p gelqs $s>](m:*const $i,n:*const $i,nrhs:*const $i,a:*const $t,lda:*const $i,tau:*const $t,
            b:*mut $t,ldb:*const $i,_work:*mut $t,lwork:*const $i,info:*mut $i) {
            *info=0;let(m,n,r,la,lb,lw)=(*m as isize,*n as isize,*nrhs as isize,*lda as isize,*ldb as isize,*lwork as isize);
            for(arg,ok)in [(1,m>=0),(2,n>=m),(3,r>=0),(5,la>=m.max(1)),(8,lb>=n.max(1)),(10,lw>=1&&(m==0||n==0||lw>=r))] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            if m==0||n==0||r==0 {return;}
            crate::kernels::trmatrix(b'L',b'L',b'N',b'N',m,r,<$t>::o(),a,la,b,lb,true);
            for j in 0..r {for i in m..n {*b.offset(i+j*lb)=<$t>::z();}}
            apply_lq(b'L',if <$t>::COMPLEX {b'C'} else {b'T'},n as usize,r as usize,m as usize,a,la,tau,b,lb);
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

pub(crate) unsafe fn compact_lq<T: ReflectorScalar>(
    m: usize,
    n: usize,
    mb: usize,
    a: *mut T,
    lda: isize,
    t: *mut T,
    ldt: isize,
) {
    let k = m.min(n);
    if k == 0 {
        return;
    }
    let mut tau = vec![T::z(); k];
    standard_lq(m, n, a, lda, tau.as_mut_ptr());
    for j in (0..k).step_by(mb) {
        crate::householder::factor(
            b'F',
            b'R',
            n - j,
            mb.min(k - j),
            a.offset(j as isize * (lda + 1)),
            lda,
            tau.as_ptr().add(j),
            t.offset(j as isize * ldt),
            ldt,
        );
    }
}

pub(crate) unsafe fn apply_compact_lq<T: Scalar>(
    side: u8,
    tr: u8,
    m: usize,
    n: usize,
    k: usize,
    mb: usize,
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
    let mut starts: Vec<usize> = (0..k).step_by(mb).collect();
    if (side == b'L' && tr != b'N') || (side == b'R' && tr == b'N') {
        starts.reverse();
    }
    let block_tr = if tr == b'N' {
        if T::COMPLEX {
            b'C'
        } else {
            b'T'
        }
    } else {
        b'N'
    };
    for j in starts {
        let (rows, cols, offset) = if side == b'L' {
            (m - j, n, j as isize)
        } else {
            (m, n - j, j as isize * ldc)
        };
        crate::householder::apply_block(
            side,
            block_tr,
            b'F',
            b'R',
            rows,
            cols,
            mb.min(k - j),
            v.offset(j as isize * (ldv + 1)),
            ldv,
            t.offset(j as isize * ldt),
            ldt,
            c.offset(offset),
            ldc,
        );
    }
}

#[cfg(test)]
mod compact_tests {
    use super::*;
    #[test]
    fn compact_row_storage_matches_standard_q() {
        unsafe {
            for mb in [1, 2] {
                let original = [1.0_f64, 2.0, 3.0, -1.0, 4.0, 2.0];
                let mut a = original;
                let mut t = vec![0.0; mb * 2];
                compact_lq(2, 3, mb, a.as_mut_ptr(), 2, t.as_mut_ptr(), mb as isize);
                let mut q = [0.0; 9];
                for i in 0..3 {
                    q[i + i * 3] = 1.0;
                }
                apply_compact_lq(
                    b'R',
                    b'N',
                    3,
                    3,
                    2,
                    mb,
                    a.as_ptr(),
                    2,
                    t.as_ptr(),
                    mb as isize,
                    q.as_mut_ptr(),
                    3,
                );
                for j in 0..3 {
                    for i in 0..2 {
                        let mut value = 0.0;
                        for k in 0..=i {
                            value += a[i + k * 2] * q[k + j * 3];
                        }
                        assert!((value - original[i + j * 2]).abs() < 1e-12);
                    }
                }
            }
        }
    }
}

macro_rules! compact_exports {
    ($p:ident,$t:ty,$i:ty,$s:tt)=>{paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p gelqt $s>](m:*const $i,n:*const $i,mb:*const $i,a:*mut $t,lda:*const $i,t:*mut $t,ldt:*const $i,_work:*mut $t,info:*mut $i) {
            *info=0;let(m,n,mb,la,lt)=(*m as isize,*n as isize,*mb as isize,*lda as isize,*ldt as isize);
            for(arg,ok)in [(1,m>=0),(2,n>=0),(3,mb>=1&&(mb<=m.min(n)||m.min(n)==0)),(5,la>=m.max(1)),(7,lt>=mb)] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            compact_lq(m as usize,n as usize,mb as usize,a,la,t,lt);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p gelqt3 $s>](m:*const $i,n:*const $i,a:*mut $t,lda:*const $i,t:*mut $t,ldt:*const $i,info:*mut $i) {
            *info=0;let(m,n,la,lt)=(*m as isize,*n as isize,*lda as isize,*ldt as isize);
            for(arg,ok)in [(1,m>=0),(2,n>=m),(4,la>=m.max(1)),(6,lt>=m.max(1))] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            compact_lq(m as usize,n as usize,m.max(1) as usize,a,la,t,lt);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p gemlqt $s>](side:*const u8,tr:*const u8,m:*const $i,n:*const $i,k:*const $i,mb:*const $i,
            v:*const $t,ldv:*const $i,t:*const $t,ldt:*const $i,c:*mut $t,ldc:*const $i,_work:*mut $t,info:*mut $i) {
            *info=0;let(side,tr)=(flag(side),flag(tr));
            let(m,n,k,mb,lv,lt,lc)=(*m as isize,*n as isize,*k as isize,*mb as isize,*ldv as isize,*ldt as isize,*ldc as isize);
            let order=if side==b'L' {m} else {n};
            for(arg,ok)in [(1,matches!(side,b'L'|b'R')),(2,tr==b'N'||tr==if <$t>::COMPLEX {b'C'} else {b'T'}),
                (3,m>=0),(4,n>=0),(5,k>=0&&k<=order),(6,mb>=1&&(mb<=k||k==0)),(8,lv>=k.max(1)),(10,lt>=mb),(12,lc>=m.max(1))] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            apply_compact_lq(side,tr,m as usize,n as usize,k as usize,mb as usize,v,lv,t,lt,c,lc);
        }
    }};
}
compact_exports!(s, f32, i32, _);
compact_exports!(d, f64, i32, _);
compact_exports!(c, c32, i32, _);
compact_exports!(z, c64, i32, _);
compact_exports!(s, f32, i64, _64_);
compact_exports!(d, f64, i64, _64_);
compact_exports!(c, c32, i64, _64_);
compact_exports!(z, c64, i64, _64_);

// Short-wide LQ stores one compact factor for the first NB columns, then
// factors of [L B] panels. The upper part of the first M columns retains the
// first panel's reflector vectors; later panels only replace its lower L.
pub(crate) unsafe fn short_wide_lq<T: ReflectorScalar>(
    m: usize,
    n: usize,
    mb: usize,
    nb: usize,
    a: *mut T,
    lda: isize,
    t: *mut T,
    ldt: isize,
) {
    if m == 0 || n == 0 {
        return;
    }
    if n <= m || nb <= m || nb >= n {
        compact_lq(m, n, mb, a, lda, t, ldt);
        return;
    }
    compact_lq(m, nb, mb, a, lda, t, ldt);
    let mut start = nb;
    let mut block = 1;
    while start < n {
        let width = (nb - m).min(n - start);
        let mut panel = vec![T::z(); m * (m + width)];
        for j in 0..m {
            for i in j..m {
                panel[i + j * m] = *a.offset(i as isize + j as isize * lda);
            }
        }
        for j in 0..width {
            for i in 0..m {
                panel[i + (m + j) * m] = *a.offset(i as isize + (start + j) as isize * lda);
            }
        }
        compact_lq(
            m,
            m + width,
            mb,
            panel.as_mut_ptr(),
            m as isize,
            t.offset((block * m) as isize * ldt),
            ldt,
        );
        for j in 0..m {
            for i in j..m {
                *a.offset(i as isize + j as isize * lda) = panel[i + j * m];
            }
        }
        for j in 0..width {
            for i in 0..m {
                *a.offset(i as isize + (start + j) as isize * lda) = panel[i + (m + j) * m];
            }
        }
        start += width;
        block += 1;
    }
}

pub(crate) unsafe fn apply_short_wide_lq<T: Scalar>(
    side: u8,
    tr: u8,
    m: usize,
    n: usize,
    k: usize,
    mb: usize,
    nb: usize,
    a: *const T,
    lda: isize,
    t: *const T,
    ldt: isize,
    c: *mut T,
    ldc: isize,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    let order = if side == b'L' { m } else { n };
    if nb <= k || nb >= order {
        apply_compact_lq(side, tr, m, n, k, mb, a, lda, t, ldt, c, ldc);
        return;
    }
    let mut stages = vec![(0_usize, nb)];
    let mut start = nb;
    while start < order {
        let width = (nb - k).min(order - start);
        stages.push((start, width));
        start += width;
    }
    let mut indices: Vec<usize> = (0..stages.len()).collect();
    // Q = Q_last ... Q_first. Left Q and right Q^H therefore apply
    // panels in chronological order; their adjoints reverse that order.
    if (side == b'L' && tr != b'N') || (side == b'R' && tr == b'N') {
        indices.reverse();
    }
    for stage in indices {
        let (start, width) = stages[stage];
        let coordinates: Vec<usize> = if stage == 0 {
            (0..nb).collect()
        } else {
            (0..k).chain(start..start + width).collect()
        };
        let dim = coordinates.len();
        let mut v = vec![T::z(); k * dim];
        for (j, &original) in coordinates.iter().enumerate() {
            for i in 0..k {
                if stage == 0 || j >= k {
                    v[i + j * k] = *a.offset(i as isize + original as isize * lda);
                }
            }
        }
        let (rows, cols) = if side == b'L' { (dim, n) } else { (m, dim) };
        let mut selected = vec![T::z(); rows * cols];
        for j in 0..cols {
            for i in 0..rows {
                let (ri, cj) = if side == b'L' {
                    (coordinates[i], j)
                } else {
                    (i, coordinates[j])
                };
                selected[i + j * rows] = *c.offset(ri as isize + cj as isize * ldc);
            }
        }
        apply_compact_lq(
            side,
            tr,
            rows,
            cols,
            k,
            mb,
            v.as_ptr(),
            k as isize,
            t.offset((stage * k) as isize * ldt),
            ldt,
            selected.as_mut_ptr(),
            rows as isize,
        );
        for j in 0..cols {
            for i in 0..rows {
                let (ri, cj) = if side == b'L' {
                    (coordinates[i], j)
                } else {
                    (i, coordinates[j])
                };
                *c.offset(ri as isize + cj as isize * ldc) = selected[i + j * rows];
            }
        }
    }
}

#[cfg(test)]
mod short_wide_tests {
    use super::*;
    #[test]
    fn successive_panels_reconstruct_and_adjoint_inverts() {
        unsafe {
            for mb in [1, 2] {
                let original: Vec<f64> = (0..18).map(|i| ((i * 7 + 3) % 19) as f64 - 8.0).collect();
                let mut a = original.clone();
                let mut t = vec![0.0; mb * 2 * 3];
                short_wide_lq(2, 9, mb, 5, a.as_mut_ptr(), 2, t.as_mut_ptr(), mb as isize);
                let mut q = vec![0.0; 81];
                for i in 0..9 {
                    q[i + i * 9] = 1.0;
                }
                apply_short_wide_lq(
                    b'R',
                    b'N',
                    9,
                    9,
                    2,
                    mb,
                    5,
                    a.as_ptr(),
                    2,
                    t.as_ptr(),
                    mb as isize,
                    q.as_mut_ptr(),
                    9,
                );
                for j in 0..9 {
                    for i in 0..2 {
                        let mut value = 0.0;
                        for k in 0..=i {
                            value += a[i + k * 2] * q[k + j * 9];
                        }
                        assert!((value - original[i + j * 2]).abs() < 1e-11);
                    }
                }
                apply_short_wide_lq(
                    b'R',
                    b'T',
                    9,
                    9,
                    2,
                    mb,
                    5,
                    a.as_ptr(),
                    2,
                    t.as_ptr(),
                    mb as isize,
                    q.as_mut_ptr(),
                    9,
                );
                for j in 0..9 {
                    for i in 0..9 {
                        assert!((q[i + j * 9] - if i == j { 1.0 } else { 0.0 }).abs() < 1e-11);
                    }
                }
            }
        }
    }
}

macro_rules! short_wide_exports {
    ($p:ident,$t:ty,$i:ty,$s:tt)=>{paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p laswlq $s>](m:*const $i,n:*const $i,mb:*const $i,nb:*const $i,a:*mut $t,lda:*const $i,
            t:*mut $t,ldt:*const $i,work:*mut $t,lwork:*const $i,info:*mut $i) {
            *info=0;let(m,n,mb,nb,la,lt,lw)=(*m as isize,*n as isize,*mb as isize,*nb as isize,*lda as isize,*ldt as isize,*lwork as isize);
            let minimum=if m.min(n)==0 {1} else {m*mb};
            for(arg,ok)in [(1,m>=0),(2,n>=m),(3,mb>=1&&(mb<=m||m==0)),(4,nb>=0),(6,la>=m.max(1)),(8,lt>=mb),(10,lw== -1||lw>=minimum)] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            *work=crate::lapack::workspace::<$t>(minimum);if lw== -1 {return;}
            short_wide_lq(m as usize,n as usize,mb as usize,nb as usize,a,la,t,lt);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p lamswlq $s>](side:*const u8,tr:*const u8,m:*const $i,n:*const $i,k:*const $i,mb:*const $i,nb:*const $i,
            a:*const $t,lda:*const $i,t:*const $t,ldt:*const $i,c:*mut $t,ldc:*const $i,work:*mut $t,lwork:*const $i,info:*mut $i) {
            *info=0;let(side,tr)=(flag(side),flag(tr));
            let(m,n,k,mb,nb,la,lt,lc,lw)=(*m as isize,*n as isize,*k as isize,*mb as isize,*nb as isize,*lda as isize,*ldt as isize,*ldc as isize,*lwork as isize);
            let order=if side==b'L' {m} else {n};
            let minimum=if m.min(n).min(k)==0 {1} else {(if side==b'L' {n} else {m})*mb};
            for(arg,ok)in [(1,matches!(side,b'L'|b'R')),(2,tr==b'N'||tr==if <$t>::COMPLEX {b'C'} else {b'T'}),
                (3,m>=0),(4,n>=0),(5,k>=0&&k<=order),(6,mb>=1&&mb<=k),(9,la>=k.max(1)),(11,lt>=mb),(13,lc>=m.max(1)),(15,lw== -1||lw>=minimum)] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            *work=crate::lapack::workspace::<$t>(minimum);if lw== -1 {return;}
            apply_short_wide_lq(side,tr,m as usize,n as usize,k as usize,mb as usize,nb.max(0) as usize,a,la,t,lt,c,lc);
        }
    }};
}
short_wide_exports!(s, f32, i32, _);
short_wide_exports!(d, f64, i32, _);
short_wide_exports!(c, c32, i32, _);
short_wide_exports!(z, c64, i32, _);
short_wide_exports!(s, f32, i64, _64_);
short_wide_exports!(d, f64, i64, _64_);
short_wide_exports!(c, c32, i64, _64_);
short_wide_exports!(z, c64, i64, _64_);

macro_rules! header_exports {
    ($p:ident,$t:ty,$i:ty,$s:tt)=>{paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p gelq $s>](m:*const $i,n:*const $i,a:*mut $t,lda:*const $i,t:*mut $t,tsize:*const $i,
            work:*mut $t,lwork:*const $i,info:*mut $i) {
            *info=0;let(m,n,la,ts,lw)=(*m as isize,*n as isize,*lda as isize,*tsize as isize,*lwork as isize);
            let query=matches!(ts,-1|-2)||matches!(lw,-1|-2);
            // Choose the legal block-size-one representation. Optimal and
            // minimum queries coincide for this adapter's chosen algorithm.
            let needed=m+5;let nw=n.max(1);
            for(arg,ok)in [(1,m>=0),(2,n>=0),(4,la>=m.max(1)),(6,query||ts>=needed),(8,query||lw>=nw)] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            *t=crate::lapack::workspace::<$t>(needed);
            *t.add(1)=<$t>::o();*t.add(2)=<$t>::real(n as f64);
            *work=crate::lapack::workspace::<$t>(nw);
            if query {return;}
            compact_lq(m as usize,n as usize,1,a,la,t.add(5),1);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p gemlq $s>](side:*const u8,tr:*const u8,m:*const $i,n:*const $i,k:*const $i,a:*const $t,lda:*const $i,
            t:*const $t,tsize:*const $i,c:*mut $t,ldc:*const $i,work:*mut $t,lwork:*const $i,info:*mut $i) {
            *info=0;let(side,tr)=(flag(side),flag(tr));
            let(m,n,k,la,ts,lc,lw)=(*m as isize,*n as isize,*k as isize,*lda as isize,*tsize as isize,*ldc as isize,*lwork as isize);
            let order=if side==b'L' {m} else {n};
            for(arg,ok)in [(1,matches!(side,b'L'|b'R')),(2,tr==b'N'||tr==if <$t>::COMPLEX {b'C'} else {b'T'}),
                (3,m>=0),(4,n>=0),(5,k>=0&&k<=order),(7,la>=k.max(1)),(9,ts>=5),(11,lc>=m.max(1))] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            let mb=(*t.add(1)).re() as isize;let nb=(*t.add(2)).re() as isize;
            if mb<1 || (k>0 && mb>k) || nb<0 {*info=-8;crate::invalid(8);return;}
            let minimum=if m.min(n).min(k)==0 {1} else {(if side==b'L' {n} else {m})*mb};
            if lw!= -1 && lw<minimum {*info=-13;crate::invalid(13);return;}
            *work=crate::lapack::workspace::<$t>(minimum);
            if lw== -1 || m.min(n).min(k)==0 {return;}
            apply_short_wide_lq(side,tr,m as usize,n as usize,k as usize,mb as usize,nb as usize,a,la,t.add(5),mb,c,lc);
        }
    }};
}
header_exports!(s, f32, i32, _);
header_exports!(d, f64, i32, _);
header_exports!(c, c32, i32, _);
header_exports!(z, c64, i32, _);
header_exports!(s, f32, i64, _64_);
header_exports!(d, f64, i64, _64_);
header_exports!(c, c32, i64, _64_);
header_exports!(z, c64, i64, _64_);
