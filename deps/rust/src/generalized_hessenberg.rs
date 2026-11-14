//! Generalized Hessenberg reduction over the public native triangular-pencil API.
use crate::kernels::Scalar;
use crate::schur::multiply;
use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::{c32, c64, Mat};

fn identity<T: Scalar>(n: usize) -> Mat<T> {
    Mat::from_fn(n, n, |i, j| if i == j { T::o() } else { T::z() })
}
fn embedded<T: Scalar>(q: &Mat<T>, n: usize, lo: usize, hi: usize) -> Mat<T> {
    Mat::from_fn(n, n, |i, j| {
        if i >= lo && i < hi && j >= lo && j < hi {
            q[(i - lo, j - lo)]
        } else if i == j {
            T::o()
        } else {
            T::z()
        }
    })
}
unsafe fn reduce<T: Scalar>(
    n: usize,
    lo: usize,
    hi: usize,
    a: *mut T,
    lda: isize,
    b: *mut T,
    ldb: isize,
    q: *mut T,
    ldq: isize,
    z: *mut T,
    ldz: isize,
    cq: u8,
    cz: u8,
) {
    let mut aa = crate::lapack::input(a, n, n, lda);
    // Strict lower B is outside the input contract and must never be read.
    let mut bb = Mat::from_fn(n, n, |i, j| {
        if i <= j {
            *b.offset(i as isize + j as isize * ldb)
        } else {
            T::z()
        }
    });
    let size = hi - lo;
    let mut h = Mat::from_fn(size, size, |i, j| aa[(lo + i, lo + j)]);
    let mut t = Mat::from_fn(size, size, |i, j| bb[(lo + i, lo + j)]);
    let mut u = identity::<T>(size);
    let mut v = identity::<T>(size);
    if size > 2 {
        use colmat::linalg::gevd::gen_hessenberg::{
            generalized_hessenberg, generalized_hessenberg_scratch, GeneralizedHessenbergParams,
        };
        let params: GeneralizedHessenbergParams = colmat::auto!(T);
        crate::threading::with_par(|par| {
            let mut memory = MemBuffer::new(generalized_hessenberg_scratch::<T>(size, params));
            generalized_hessenberg(
                h.as_mut(),
                t.as_mut(),
                Some(u.as_mut()),
                Some(v.as_mut()),
                par,
                MemStack::new(&mut memory),
                params,
            );
        });
    }
    let left = embedded(&u, n, lo, hi);
    let right = embedded(&v, n, lo, hi);
    let updated_a = multiply(&left, b'C', &multiply(&aa, b'N', &right, b'N'), b'N');
    let updated_b = multiply(&left, b'C', &multiply(&bb, b'N', &right, b'N'), b'N');
    for j in 0..n {
        for i in 0..n {
            if i >= lo && i < hi && j >= lo && j < hi {
                aa[(i, j)] = if i <= j + 1 {
                    h[(i - lo, j - lo)]
                } else {
                    T::z()
                };
                bb[(i, j)] = if i <= j { t[(i - lo, j - lo)] } else { T::z() };
            } else if (i >= lo && i < hi) || (j >= lo && j < hi) {
                aa[(i, j)] = updated_a[(i, j)];
                bb[(i, j)] = if i <= j { updated_b[(i, j)] } else { T::z() };
            }
        }
    }
    crate::lapack::output(&aa, a, lda);
    crate::lapack::output(&bb, b, ldb);
    if cq != b'N' {
        let result = if cq == b'I' {
            left
        } else {
            multiply(&crate::lapack::input(q, n, n, ldq), b'N', &left, b'N')
        };
        crate::lapack::output(&result, q, ldq);
    }
    if cz != b'N' {
        let result = if cz == b'I' {
            right
        } else {
            multiply(&crate::lapack::input(z, n, n, ldz), b'N', &right, b'N')
        };
        crate::lapack::output(&result, z, ldz);
    }
}
use paste::paste;
macro_rules! exports {
    ($p:ident,$t:ty,$i:ty,$s:tt)=>{paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p gghrd $s>](compq:*const u8,compz:*const u8,n:*const $i,ilo:*const $i,ihi:*const $i,a:*mut $t,lda:*const $i,b:*mut $t,ldb:*const $i,q:*mut $t,ldq:*const $i,z:*mut $t,ldz:*const $i,info:*mut $i) {
            *info=0;let(cq,cz)=(crate::kernels::flag(compq),crate::kernels::flag(compz));
            let(n,lo,hi,la,lb,lq,lz)=(*n as isize,*ilo as isize,*ihi as isize,*lda as isize,*ldb as isize,*ldq as isize,*ldz as isize);
            for(arg,ok)in[(1,matches!(cq,b'N'|b'I'|b'V')),(2,matches!(cz,b'N'|b'I'|b'V')),(3,n>=0),(4,lo>=1&&lo<=n.max(1)),(5,hi>=lo.min(n)&&hi<=n),(7,la>=n.max(1)),(9,lb>=n.max(1)),(11,lq>=1&&(cq==b'N'||lq>=n)),(13,lz>=1&&(cz==b'N'||lz>=n))] {
                if !ok{*info=-arg;crate::invalid(arg as i32);return;}
            }
            reduce(n as usize,(lo-1) as usize,hi as usize,a,la,b,lb,q,lq,z,lz,cq,cz);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p gghd3 $s>](compq:*const u8,compz:*const u8,n:*const $i,ilo:*const $i,ihi:*const $i,a:*mut $t,lda:*const $i,b:*mut $t,ldb:*const $i,q:*mut $t,ldq:*const $i,z:*mut $t,ldz:*const $i,work:*mut $t,lwork:*const $i,info:*mut $i) {
            // Validate all preceding arguments before checking WORK, without changing arrays.
            let(cq,cz)=(crate::kernels::flag(compq),crate::kernels::flag(compz));let(nv,lo,hi)=(*n as isize,*ilo as isize,*ihi as isize);
            *info=0;
            for(arg,ok)in[(1,matches!(cq,b'N'|b'I'|b'V')),(2,matches!(cz,b'N'|b'I'|b'V')),(3,nv>=0),(4,lo>=1&&lo<=nv.max(1)),(5,hi>=lo.min(nv)&&hi<=nv),(7,*lda as isize>=nv.max(1)),(9,*ldb as isize>=nv.max(1)),(11,*ldq>=1&&(cq==b'N'||*ldq as isize>=nv)),(13,*ldz>=1&&(cz==b'N'||*ldz as isize>=nv)),(15,*lwork== -1||*lwork>=1)] {
                if !ok{*info=-arg;crate::invalid(arg as i32);return;}
            }
            *work=crate::lapack::workspace::<$t>(1);if *lwork== -1{return;}
            [<$p gghrd $s>](compq,compz,n,ilo,ihi,a,lda,b,ldb,q,ldq,z,ldz,info);
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
