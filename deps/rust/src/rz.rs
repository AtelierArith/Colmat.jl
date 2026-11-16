//! RZ reflectors act on the first and last L coordinates only.
use crate::householder::ReflectorScalar;
use crate::kernels::{start, Scalar};

pub(crate) unsafe fn apply_sparse<T: Scalar>(
    side: u8,
    m: usize,
    n: usize,
    l: usize,
    v: *const T,
    inc: isize,
    tau: T,
    c: *mut T,
    ldc: isize,
) {
    if m == 0 || n == 0 || tau == T::z() {
        return;
    }
    let order = if side == b'L' { m } else { n };
    if l == order {
        let v = v.offset(start(l as isize, inc));
        let mut vector = vec![T::z(); order];
        for i in 0..order {
            vector[i] = *v.offset(i as isize * inc);
        }
        vector[0] = vector[0] + T::o();
        let mut work = vec![T::z(); m.max(n)];
        crate::householder::apply(
            side,
            m as isize,
            n as isize,
            vector.as_ptr(),
            1,
            tau,
            c,
            ldc,
            work.as_mut_ptr(),
        );
        return;
    }
    let coordinates: Vec<usize> = std::iter::once(0).chain(order - l..order).collect();
    let mut vector = vec![T::o(); l + 1];
    let v = if l == 0 {
        v
    } else {
        v.offset(start(l as isize, inc))
    };
    for i in 0..l {
        vector[i + 1] = *v.offset(i as isize * inc);
    }
    let (rows, cols) = if side == b'L' { (l + 1, n) } else { (m, l + 1) };
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
    let mut work = vec![T::z(); m.max(n)];
    crate::householder::apply(
        side,
        rows as isize,
        cols as isize,
        vector.as_ptr(),
        1,
        tau,
        selected.as_mut_ptr(),
        rows as isize,
        work.as_mut_ptr(),
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

pub(crate) unsafe fn factor<T: ReflectorScalar>(
    m: usize,
    n: usize,
    l: usize,
    a: *mut T,
    lda: isize,
    tau: *mut T,
) {
    if m == n {
        for i in 0..m {
            *tau.add(i) = T::z();
        }
        return;
    }
    for i in (0..m).rev() {
        let mut tail = vec![T::z(); l];
        for j in 0..l {
            tail[j] = (*a.offset(i as isize + (n - l + j) as isize * lda)).cj();
        }
        let mut alpha = (*a.offset(i as isize * (lda + 1))).cj();
        let mut coeff = T::z();
        crate::householder::generate(
            (l + 1) as isize,
            &mut alpha,
            tail.as_mut_ptr(),
            1,
            &mut coeff,
            false,
        );
        *tau.add(i) = coeff.cj();
        for j in 0..l {
            *a.offset(i as isize + (n - l + j) as isize * lda) = tail[j];
        }
        if i > 0 {
            apply_sparse(
                b'R',
                i,
                n - i,
                l,
                tail.as_ptr(),
                1,
                coeff,
                a.offset(i as isize * lda),
                lda,
            );
        }
        *a.offset(i as isize * (lda + 1)) = alpha.cj();
    }
}

pub(crate) unsafe fn apply_rz<T: Scalar>(
    side: u8,
    tr: u8,
    m: usize,
    n: usize,
    k: usize,
    l: usize,
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
    for i in indices {
        let coeff = if tr == b'N' {
            *tau.add(i)
        } else {
            (*tau.add(i)).cj()
        };
        if coeff == T::z() {
            continue;
        }
        if l > order - i {
            // LAPACK permits L up to the full order. In that case the
            // implicit coordinate overlaps the stored tail: combine them.
            let mut vector = vec![T::z(); l];
            for j in 0..l {
                vector[j] = *a.offset(i as isize + (order - l + j) as isize * lda);
            }
            vector[i - (order - l)] = vector[i - (order - l)] + T::o();
            let (rows, cols, offset) = if side == b'L' {
                (l, n, (order - l) as isize)
            } else {
                (m, l, (order - l) as isize * ldc)
            };
            let mut work = vec![T::z(); m.max(n)];
            crate::householder::apply(
                side,
                rows as isize,
                cols as isize,
                vector.as_ptr(),
                1,
                coeff,
                c.offset(offset),
                ldc,
                work.as_mut_ptr(),
            );
            continue;
        }
        let (rows, cols, offset) = if side == b'L' {
            (m - i, n, i as isize)
        } else {
            (m, n - i, i as isize * ldc)
        };
        let tail = if l == 0 {
            a
        } else {
            a.offset(i as isize + (order - l) as isize * lda)
        };
        apply_sparse(side, rows, cols, l, tail, lda, coeff, c.offset(offset), ldc);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn reconstruct<T: ReflectorScalar>() {
        unsafe {
            let m = 3;
            let n = 6;
            let mut original = vec![T::z(); m * n];
            for j in 0..n {
                for i in 0..m.min(j + 1) {
                    original[i + j * m] = T::parts(
                        (i + j * 3 + 1) as f64,
                        if T::COMPLEX { i as f64 - 1.0 } else { 0.0 },
                    );
                }
            }
            let mut a = original.clone();
            let mut tau = vec![T::z(); m];
            factor(m, n, n - m, a.as_mut_ptr(), m as isize, tau.as_mut_ptr());
            let mut r = vec![T::z(); m * n];
            for j in 0..m {
                for i in 0..=j {
                    r[i + j * m] = a[i + j * m];
                }
            }
            apply_rz(
                b'R',
                b'N',
                m,
                n,
                m,
                n - m,
                a.as_ptr(),
                m as isize,
                tau.as_ptr(),
                r.as_mut_ptr(),
                m as isize,
            );
            for i in 0..r.len() {
                assert!((r[i] - original[i]).magnitude() < 1e-11);
            }
            apply_rz(
                b'R',
                if T::COMPLEX { b'C' } else { b'T' },
                m,
                n,
                m,
                n - m,
                a.as_ptr(),
                m as isize,
                tau.as_ptr(),
                r.as_mut_ptr(),
                m as isize,
            );
            for j in 0..n {
                for i in 0..m {
                    let expected = if j < m && i <= j {
                        a[i + j * m]
                    } else {
                        T::z()
                    };
                    assert!((r[i + j * m] - expected).magnitude() < 1e-11);
                }
            }
        }
    }
    #[test]
    fn real_and_complex_rz_reconstruct() {
        reconstruct::<f64>();
        reconstruct::<colmat::c64>();
    }
}

use crate::kernels::flag;
use colmat::{c32, c64};
use paste::paste;
macro_rules! basic_exports {
    ($p:ident,$t:ty,$i:ty,$s:tt)=>{paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p tzrzf $s>](m:*const $i,n:*const $i,a:*mut $t,lda:*const $i,tau:*mut $t,work:*mut $t,lwork:*const $i,info:*mut $i) {
            *info=0;let(m,n,la,lw)=(*m as isize,*n as isize,*lda as isize,*lwork as isize);
            for(arg,ok)in [(1,m>=0),(2,n>=m),(4,la>=m.max(1))] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            let minimum=if m==n {1}else{m.max(1)};
            *work=crate::lapack::workspace::<$t>(minimum);
            if lw!= -1&&lw<minimum {*info=-7;crate::invalid(7);return;}
            if lw== -1 {return;}
            factor(m as usize,n as usize,(n-m) as usize,a,la,tau);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p tzrqf $s>](m:*const $i,n:*const $i,a:*mut $t,lda:*const $i,tau:*mut $t,info:*mut $i) {
            *info=0;let(m,n,la)=(*m as isize,*n as isize,*lda as isize);
            for(arg,ok)in [(1,m>=0),(2,n>=m),(4,la>=m.max(1))] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            factor(m as usize,n as usize,(n-m) as usize,a,la,tau);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p latrz $s>](m:*const $i,n:*const $i,l:*const $i,a:*mut $t,lda:*const $i,tau:*mut $t,_work:*mut $t) {
            let(m,n,l,la)=(*m as isize,*n as isize,*l as isize,*lda as isize);
            for(arg,ok)in [(1,m>=0),(2,n>=m),(3,l>=0&&l<=n-m),(5,la>=m.max(1))] {
                if !ok {crate::invalid(arg);return;}
            }
            factor(m as usize,n as usize,l as usize,a,la,tau);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p larz $s>](side:*const u8,m:*const $i,n:*const $i,l:*const $i,v:*const $t,inc:*const $i,tau:*const $t,c:*mut $t,ldc:*const $i,_work:*mut $t) {
            let(side,m,n,l,inc,lc)=(flag(side),*m as isize,*n as isize,*l as isize,*inc as isize,*ldc as isize);
            let order=if side==b'L' {m} else {n};
            for(arg,ok)in [(1,matches!(side,b'L'|b'R')),(2,m>=0),(3,n>=0),(4,l>=0&&l<=order),(6,inc!=0),(9,lc>=m.max(1))] {
                if !ok {crate::invalid(arg);return;}
            }
            apply_sparse(side,m as usize,n as usize,l as usize,v,inc,*tau,c,lc);
        }
    }};
}
basic_exports!(s, f32, i32, _);
basic_exports!(d, f64, i32, _);
basic_exports!(c, c32, i32, _);
basic_exports!(z, c64, i32, _);
basic_exports!(s, f32, i64, _64_);
basic_exports!(d, f64, i64, _64_);
basic_exports!(c, c32, i64, _64_);
basic_exports!(z, c64, i64, _64_);

macro_rules! apply_exports {
    ($p:ident,$t:ty,$i:ty,$s:tt,$name:ident)=>{paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $name $s>](side:*const u8,tr:*const u8,m:*const $i,n:*const $i,k:*const $i,l:*const $i,
            a:*const $t,lda:*const $i,tau:*const $t,c:*mut $t,ldc:*const $i,work:*mut $t,lwork:*const $i,info:*mut $i) {
            *info=0;let(side,tr)=(flag(side),flag(tr));
            let(m,n,k,l,la,lc,lw)=(*m as isize,*n as isize,*k as isize,*l as isize,*lda as isize,*ldc as isize,*lwork as isize);
            let order=if side==b'L' {m} else {n};let nw=if side==b'L' {n} else {m};
            for(arg,ok)in [(1,matches!(side,b'L'|b'R')),(2,tr==b'N'||tr==if <$t>::COMPLEX {b'C'} else {b'T'}),
                (3,m>=0),(4,n>=0),(5,k>=0&&k<=order),(6,l>=0&&l<=order),(8,la>=k.max(1)),(11,lc>=m.max(1)),(13,lw== -1||lw>=nw.max(1))] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            *work=crate::lapack::workspace::<$t>(if m==0||n==0{1}else{nw.max(1)});if lw== -1 {return;}
            apply_rz(side,tr,m as usize,n as usize,k as usize,l as usize,a,la,tau,c,lc);
        }
    }};
}
apply_exports!(s, f32, i32, _, ormrz);
apply_exports!(d, f64, i32, _, ormrz);
apply_exports!(c, c32, i32, _, unmrz);
apply_exports!(z, c64, i32, _, unmrz);
apply_exports!(s, f32, i64, _64_, ormrz);
apply_exports!(d, f64, i64, _64_, ormrz);
apply_exports!(c, c32, i64, _64_, unmrz);
apply_exports!(z, c64, i64, _64_, unmrz);

pub(crate) unsafe fn block_factor<T: Scalar>(
    n: usize,
    k: usize,
    v: *const T,
    ldv: isize,
    tau: *const T,
    t: *mut T,
    ldt: isize,
) {
    if k == 0 {
        return;
    }
    // Move the implicit identity to the end so the shared backward
    // Householder factor builder sees its usual unit triangular basis.
    let mut basis = vec![T::z(); k * (n + k)];
    for j in 0..n {
        for i in 0..k {
            basis[i + j * k] = *v.offset(i as isize + j as isize * ldv);
        }
    }
    crate::householder::factor(
        b'B',
        b'R',
        n + k,
        k,
        basis.as_ptr(),
        k as isize,
        tau,
        t,
        ldt,
    );
}

pub(crate) unsafe fn block_apply<T: Scalar>(
    side: u8,
    tr: u8,
    m: usize,
    n: usize,
    k: usize,
    l: usize,
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
    let order = if side == b'L' { m } else { n };
    let coordinates: Vec<usize> = (0..k)
        .chain((order - l..order).filter(|&j| j >= k))
        .collect();
    let dim = coordinates.len();
    let mut b = vec![T::z(); dim * k];
    for j in 0..k {
        b[j + j * dim] = T::o();
        for r in 0..l {
            let coordinate = order - l + r;
            let index = coordinates.iter().position(|&x| x == coordinate).unwrap();
            b[index + j * dim] = b[index + j * dim] + *v.offset(j as isize + r as isize * ldv);
        }
    }
    let mut triangular = vec![T::z(); k * k];
    for j in 0..k {
        for i in j..k {
            triangular[i + j * k] = (*t.offset(i as isize + j as isize * ldt)).cj();
        }
    }
    let mut bt = vec![T::z(); dim * k];
    let mut h = vec![T::z(); dim * dim];
    crate::kernels::gemm(
        b'N',
        tr,
        dim as isize,
        k as isize,
        k as isize,
        T::o(),
        b.as_ptr(),
        dim as isize,
        triangular.as_ptr(),
        k as isize,
        T::z(),
        bt.as_mut_ptr(),
        dim as isize,
    );
    crate::kernels::gemm(
        b'N',
        b'C',
        dim as isize,
        dim as isize,
        k as isize,
        -T::o(),
        bt.as_ptr(),
        dim as isize,
        b.as_ptr(),
        dim as isize,
        T::z(),
        h.as_mut_ptr(),
        dim as isize,
    );
    for i in 0..dim {
        h[i + i * dim] = h[i + i * dim] + T::o();
    }
    let (rows, cols) = if side == b'L' { (dim, n) } else { (m, dim) };
    let mut selected = vec![T::z(); rows * cols];
    let mut output = vec![T::z(); rows * cols];
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
    let (a, la, b, lb) = if side == b'L' {
        (h.as_ptr(), dim as isize, selected.as_ptr(), rows as isize)
    } else {
        (selected.as_ptr(), rows as isize, h.as_ptr(), dim as isize)
    };
    crate::kernels::gemm(
        b'N',
        b'N',
        rows as isize,
        cols as isize,
        dim as isize,
        T::o(),
        a,
        la,
        b,
        lb,
        T::z(),
        output.as_mut_ptr(),
        rows as isize,
    );
    for j in 0..cols {
        for i in 0..rows {
            let (ri, cj) = if side == b'L' {
                (coordinates[i], j)
            } else {
                (i, coordinates[j])
            };
            *c.offset(ri as isize + cj as isize * ldc) = output[i + j * rows];
        }
    }
}

macro_rules! block_exports {
    ($p:ident,$t:ty,$i:ty,$s:tt)=>{paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p larzt $s>](d:*const u8,s:*const u8,n:*const $i,k:*const $i,v:*const $t,ldv:*const $i,tau:*const $t,t:*mut $t,ldt:*const $i) {
            let(d,s,n,k,lv,lt)=(flag(d),flag(s),*n as isize,*k as isize,*ldv as isize,*ldt as isize);
            for(arg,ok)in [(1,d==b'B'),(2,s==b'R'),(3,n>=0),(4,k>=0),(6,lv>=k.max(1)),(9,lt>=k.max(1))] {
                if !ok {crate::invalid(arg);return;}
            }
            block_factor(n as usize,k as usize,v,lv,tau,t,lt);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p larzb $s>](side:*const u8,tr:*const u8,d:*const u8,s:*const u8,m:*const $i,n:*const $i,k:*const $i,l:*const $i,
            v:*const $t,ldv:*const $i,t:*const $t,ldt:*const $i,c:*mut $t,ldc:*const $i,_work:*mut $t,ldwork:*const $i) {
            let(side,tr,d,s)=(flag(side),flag(tr),flag(d),flag(s));
            let(m,n,k,l,lv,lt,lc,lw)=(*m as isize,*n as isize,*k as isize,*l as isize,*ldv as isize,*ldt as isize,*ldc as isize,*ldwork as isize);
            let order=if side==b'L' {m} else {n};let nw=if side==b'L' {n} else {m};
            for(arg,ok)in [(1,matches!(side,b'L'|b'R')),(2,tr==b'N'||tr==if <$t>::COMPLEX {b'C'} else {b'T'}),(3,d==b'B'),(4,s==b'R'),
                (5,m>=0),(6,n>=0),(7,k>=0&&k<=order),(8,l>=0&&l<=order),(10,lv>=k.max(1)),(12,lt>=k.max(1)),(14,lc>=m.max(1)),(16,lw>=nw.max(1))] {
                if !ok {crate::invalid(arg);return;}
            }
            block_apply(side,tr,m as usize,n as usize,k as usize,l as usize,v,lv,t,lt,c,lc);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p latzm $s>](side:*const u8,m:*const $i,n:*const $i,v:*const $t,inc:*const $i,tau:*const $t,
            c1:*mut $t,c2:*mut $t,ldc:*const $i,_work:*mut $t) {
            let(side,m,n,inc,lc)=(flag(side),*m as isize,*n as isize,*inc as isize,*ldc as isize);
            for(arg,ok)in [(1,matches!(side,b'L'|b'R')),(2,m>=0),(3,n>=0),(5,inc!=0),(9,lc>=m.max(1))] {
                if !ok {crate::invalid(arg);return;}
            }
            if m==0||n==0||*tau==<$t>::z() {return;}
            let mut joined=vec![<$t>::z();(m*n) as usize];
            for j in 0..n {for i in 0..m {
                joined[(i+j*m) as usize]=if side==b'L' {
                    if i==0 {*c1.offset(j*lc)} else {*c2.offset(i-1+j*lc)}
                } else if j==0 {*c1.offset(i)} else {*c2.offset(i+(j-1)*lc)};
            }}
            apply_sparse(side,m as usize,n as usize,(if side==b'L' {m-1} else {n-1}) as usize,v,inc,*tau,joined.as_mut_ptr(),m);
            for j in 0..n {for i in 0..m {
                let value=joined[(i+j*m) as usize];
                if side==b'L' {if i==0 {*c1.offset(j*lc)=value;} else {*c2.offset(i-1+j*lc)=value;}}
                else if j==0 {*c1.offset(i)=value;} else {*c2.offset(i+(j-1)*lc)=value;}
            }}
        }
    }};
}
block_exports!(s, f32, i32, _);
block_exports!(d, f64, i32, _);
block_exports!(c, c32, i32, _);
block_exports!(z, c64, i32, _);
block_exports!(s, f32, i64, _64_);
block_exports!(d, f64, i64, _64_);
block_exports!(c, c32, i64, _64_);
block_exports!(z, c64, i64, _64_);
