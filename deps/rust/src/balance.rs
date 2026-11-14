//! Standalone LAPACK balancing and eigenvector back-transformation.
use crate::lapack::General;
use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::Mat;

pub(crate) fn encode_scale(
    permutation: &[usize],
    mut scale: Vec<f64>,
    lo: usize,
    hi: usize,
) -> Vec<f64> {
    let n = permutation.len();
    let mut current: Vec<usize> = (0..n).collect();
    for i in (hi..n).rev().chain(0..lo) {
        let position = (0..n).find(|&k| current[k] == permutation[i]).unwrap();
        scale[i] = (position + 1) as f64;
        current.swap(i, position);
    }
    scale
}
trait BalanceScalar: General {
    fn back(values: &mut Mat<Self>, permutation: &[usize], scale: &[f64]);
}
macro_rules! scalar {
    ($t:ty,$r:ty) => {
        impl BalanceScalar for $t {
            fn back(values: &mut Mat<Self>, permutation: &[usize], scale: &[f64]) {
                use colmat::linalg::evd::{
                    balance_eigenvectors_in_place, balance_eigenvectors_scratch,
                };
                let scale: Vec<$r> = scale.iter().map(|&x| x as $r).collect();
                let mut memory =
                    MemBuffer::new(balance_eigenvectors_scratch::<Self>(values.nrows()));
                balance_eigenvectors_in_place(
                    values.as_mut(),
                    permutation,
                    &scale,
                    colmat::Par::Seq,
                    MemStack::new(&mut memory),
                );
            }
        }
    };
}
scalar!(f32, f32);
scalar!(f64, f64);
scalar!(colmat::c32, f32);
scalar!(colmat::c64, f64);
unsafe fn back<T: BalanceScalar>(
    job: u8,
    side: u8,
    n: usize,
    lo: usize,
    hi: usize,
    raw: &[f64],
    m: usize,
    v: *mut T,
    ldv: isize,
) {
    let mut permutation: Vec<usize> = (0..n).collect();
    if matches!(job, b'P' | b'B') {
        // Reverse xGEBAK's swaps to recover the native final permutation.
        for i in (hi..n).rev().chain(0..lo) {
            permutation.swap(i, raw[i] as usize - 1);
        }
    }
    let mut scale = vec![1.0; n];
    if matches!(job, b'S' | b'B') {
        for i in lo..hi {
            scale[i] = if side == b'L' { 1.0 / raw[i] } else { raw[i] };
        }
    }
    let mut affected = vec![false; n];
    for i in 0..n {
        if permutation[i] != i || scale[i] != 1.0 {
            affected[i] = true;
            affected[permutation[i]] = true;
        }
    }
    if !affected.iter().any(|&x| x) {
        return;
    }
    let mut values = Mat::from_fn(n, m, |i, j| {
        if affected[i] {
            *v.offset(i as isize + j as isize * ldv)
        } else {
            T::z()
        }
    });
    T::back(&mut values, &permutation, &scale);
    for j in 0..m {
        for i in 0..n {
            if affected[i] {
                *v.offset(i as isize + j as isize * ldv) = values[(i, j)];
            }
        }
    }
}
use crate::kernels::flag;
use paste::paste;
macro_rules! exports{
    ($p:ident,$t:ty,$r:ty,$i:ty,$s:tt)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p gebal $s>](job:*const u8,n:*const $i,a:*mut $t,lda:*const $i,ilo:*mut $i,ihi:*mut $i,scale:*mut $r,info:*mut $i){
            *info=0;let(mode,nn,la)=(flag(job),*n as isize,*lda as isize);
            for(arg,ok)in[(1,matches!(mode,b'N'|b'P'|b'S'|b'B')),(2,nn>=0),(4,la>=nn.max(1))]{if !ok{*info=-arg;crate::invalid(arg as i32);return;}}
            *ilo=1;*ihi=nn as $i;for i in 0..nn as usize{*scale.add(i)=1.0;}
            if mode==b'N'||nn==0{return;}
            let matrix=crate::lapack::input_mut(a,nn as usize,nn as usize,la);
            match <$t as General>::balance(matrix,mode){
                Ok((permutation,factors,lo,hi))=>{
                    let encoded=encode_scale(&permutation,factors,lo,hi);
                    for i in 0..nn as usize{*scale.add(i)=encoded[i] as $r;}
                    *ilo=(lo+1) as $i;*ihi=hi as $i;
                },
                Err(())=>{*info=-3;crate::invalid(3);}
            }
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p gebak $s>](job:*const u8,side:*const u8,n:*const $i,ilo:*const $i,ihi:*const $i,scale:*const $r,m:*const $i,v:*mut $t,ldv:*const $i,info:*mut $i){
            *info=0;let(mode,side,nn,lo,hi,columns,ld)=(flag(job),flag(side),*n as isize,*ilo as isize,*ihi as isize,*m as isize,*ldv as isize);
            for(arg,ok)in[(1,matches!(mode,b'N'|b'P'|b'S'|b'B')),(2,matches!(side,b'L'|b'R')),(3,nn>=0),(4,lo>=1&&lo<=nn.max(1)),(5,hi>=lo.min(nn)&&hi<=nn),(7,columns>=0),(9,ld>=nn.max(1))]{if !ok{*info=-arg;crate::invalid(arg as i32);return;}}
            if mode==b'N'||nn==0||columns==0{return;}
            let mut raw=vec![1.0;nn as usize];
            for i in 0..nn as usize{
                if ((mode==b'P'||mode==b'B')&&(i<(lo-1) as usize||i>=hi as usize))||((mode==b'S'||mode==b'B')&&i>=(lo-1) as usize&&i<hi as usize){raw[i]=*scale.add(i) as f64;}
            }
            back(mode,side,nn as usize,(lo-1) as usize,hi as usize,&raw,columns as usize,v,ld);
        }
    }};
}
exports!(s, f32, f32, i32, _);
exports!(d, f64, f64, i32, _);
exports!(c, colmat::c32, f32, i32, _);
exports!(z, colmat::c64, f64, i32, _);
exports!(s, f32, f32, i64, _64_);
exports!(d, f64, f64, i64, _64_);
exports!(c, colmat::c32, f32, i64, _64_);
exports!(z, colmat::c64, f64, i64, _64_);
