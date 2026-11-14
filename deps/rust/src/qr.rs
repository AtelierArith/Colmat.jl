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
