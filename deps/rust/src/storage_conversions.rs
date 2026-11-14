//! TR/TP/TF format conversions using borrowed native copy views.
use crate::kernels::{flag, uplo, Scalar};
use crate::storage_layout::TriangleStorage;
use colmat::{MatMut, MatRef};
use paste::paste;
unsafe fn convert<T: Scalar>(
    u: u8,
    n: usize,
    a: *const T,
    source: TriangleStorage,
    b: *mut T,
    target: TriangleStorage,
) {
    let lower = u == b'L';
    for j in 0..n {
        let length = if lower { n - j } else { j + 1 };
        let (ai, astride, aconj) = source.segment(n, lower, j);
        let (bi, bstride, bconj) = target.segment(n, lower, j);
        let src = MatRef::from_raw_parts(a.add(ai), length, 1, astride, 0);
        let mut dst = MatMut::from_raw_parts_mut(b.add(bi), length, 1, bstride, 0);
        if T::COMPLEX && (aconj ^ bconj) {
            dst.copy_from(src.conjugate());
        } else {
            dst.copy_from(src);
        }
    }
}
macro_rules! check {($info:expr,$($arg:expr=>$valid:expr),+)=>{$(if !$valid{*$info=-$arg;crate::invalid($arg);return;})+};}
macro_rules! exports {
    ($p:ident,$t:ty,$i:ty,$suffix:tt)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p trttp $suffix>](u:*const u8,n:*const $i,a:*const $t,
            lda:*const $i,ap:*mut $t,info:*mut $i) {
            *info=0;let(u,n,ld)=(flag(u),*n as isize,*lda as isize);
            check!(info,1=>uplo(u),2=>n>=0,4=>ld>=n.max(1));
            convert(u,n as usize,a,TriangleStorage::Full(ld),ap,TriangleStorage::Packed);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p tpttr $suffix>](u:*const u8,n:*const $i,ap:*const $t,
            a:*mut $t,lda:*const $i,info:*mut $i) {
            *info=0;let(u,n,ld)=(flag(u),*n as isize,*lda as isize);
            check!(info,1=>uplo(u),2=>n>=0,5=>ld>=n.max(1));
            convert(u,n as usize,ap,TriangleStorage::Packed,a,TriangleStorage::Full(ld));
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p trttf $suffix>](transr:*const u8,u:*const u8,n:*const $i,
            a:*const $t,lda:*const $i,arf:*mut $t,info:*mut $i) {
            *info=0;let(tr,u,n,ld)=(flag(transr),flag(u),*n as isize,*lda as isize);
            check!(info,1=>(tr==b'N'||tr==if <$t>::COMPLEX{b'C'}else{b'T'}),
                2=>uplo(u),3=>n>=0,5=>ld>=n.max(1));
            convert(u,n as usize,a,TriangleStorage::Full(ld),arf,TriangleStorage::Rfp{transposed:tr!=b'N'});
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p tfttr $suffix>](transr:*const u8,u:*const u8,n:*const $i,
            arf:*const $t,a:*mut $t,lda:*const $i,info:*mut $i) {
            *info=0;let(tr,u,n,ld)=(flag(transr),flag(u),*n as isize,*lda as isize);
            check!(info,1=>(tr==b'N'||tr==if <$t>::COMPLEX{b'C'}else{b'T'}),
                2=>uplo(u),3=>n>=0,6=>ld>=n.max(1));
            convert(u,n as usize,arf,TriangleStorage::Rfp{transposed:tr!=b'N'},a,TriangleStorage::Full(ld));
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p tpttf $suffix>](transr:*const u8,u:*const u8,n:*const $i,
            ap:*const $t,arf:*mut $t,info:*mut $i) {
            *info=0;let(tr,u,n)=(flag(transr),flag(u),*n as isize);
            check!(info,1=>(tr==b'N'||tr==if <$t>::COMPLEX{b'C'}else{b'T'}),2=>uplo(u),3=>n>=0);
            convert(u,n as usize,ap,TriangleStorage::Packed,arf,TriangleStorage::Rfp{transposed:tr!=b'N'});
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p tfttp $suffix>](transr:*const u8,u:*const u8,n:*const $i,
            arf:*const $t,ap:*mut $t,info:*mut $i) {
            *info=0;let(tr,u,n)=(flag(transr),flag(u),*n as isize);
            check!(info,1=>(tr==b'N'||tr==if <$t>::COMPLEX{b'C'}else{b'T'}),2=>uplo(u),3=>n>=0);
            convert(u,n as usize,arf,TriangleStorage::Rfp{transposed:tr!=b'N'},ap,TriangleStorage::Packed);
        }
    }};
}
exports!(s, f32, i32, _);
exports!(d, f64, i32, _);
exports!(c, colmat::c32, i32, _);
exports!(z, colmat::c64, i32, _);
exports!(s, f32, i64, _64_);
exports!(d, f64, i64, _64_);
exports!(c, colmat::c32, i64, _64_);
exports!(z, colmat::c64, i64, _64_);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::householder::ReflectorScalar;
    unsafe fn roundtrip<T: ReflectorScalar>() {
        for n in 0..10 {
            for u in [b'U', b'L'] {
                for transposed in [false, true] {
                    let ld = n + 2;
                    let guard = T::real(777.0);
                    let mut original = vec![guard; ld * n];
                    for j in 0..n {
                        for i in 0..n {
                            if (u == b'U' && i <= j) || (u == b'L' && i >= j) {
                                original[i + j * ld] =
                                    T::parts((1 + i + 3 * j) as f64, (2 * i) as f64 - j as f64);
                            }
                        }
                    }
                    let count = n * (n + 1) / 2;
                    let mut rectangular = vec![guard; count];
                    let mut packed = vec![guard; count];
                    let mut reconstructed = vec![guard; ld * n];
                    convert(
                        u,
                        n,
                        original.as_ptr(),
                        TriangleStorage::Full(ld as isize),
                        rectangular.as_mut_ptr(),
                        TriangleStorage::Rfp { transposed },
                    );
                    convert(
                        u,
                        n,
                        rectangular.as_ptr(),
                        TriangleStorage::Rfp { transposed },
                        packed.as_mut_ptr(),
                        TriangleStorage::Packed,
                    );
                    convert(
                        u,
                        n,
                        packed.as_ptr(),
                        TriangleStorage::Packed,
                        reconstructed.as_mut_ptr(),
                        TriangleStorage::Full(ld as isize),
                    );
                    assert!(original == reconstructed);
                }
            }
        }
    }
    #[test]
    fn real_and_complex_format_roundtrips() {
        unsafe {
            roundtrip::<f32>();
            roundtrip::<f64>();
            roundtrip::<colmat::c32>();
            roundtrip::<colmat::c64>();
        }
    }
    #[test]
    fn normal_rfp_folds_a_conjugated_diagonal_block() {
        unsafe {
            use colmat::c64;
            let a = [
                c64::new(1.0, 2.0),
                c64::new(0.0, 0.0),
                c64::new(3.0, 4.0),
                c64::new(5.0, 6.0),
            ];
            let mut rfp = [c64::new(0.0, 0.0); 3];
            convert(
                b'U',
                2,
                a.as_ptr(),
                TriangleStorage::Full(2),
                rfp.as_mut_ptr(),
                TriangleStorage::Rfp { transposed: false },
            );
            assert_eq!(rfp, [a[2], a[3], a[0].cj()]);
        }
    }
}
