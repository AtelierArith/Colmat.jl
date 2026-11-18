//! LAPACKE PTEQR high/work adapters.
//! Reference: Reference LAPACK v3.12.1 LAPACKE/src/lapacke_{s,d,c,z}pteqr{,_work}.c.
use super::{row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;

unsafe fn nan_vec<T: Scalar>(n: i64, a: *const T) -> bool {
    (0..n.max(0) as usize).any(|i| {
        let v = *a.add(i);
        v != v
    })
}

macro_rules! exports {
    ($p:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p pteqr $($suffix)*>](
            layout:i32,compz:u8,n:$i,d:*mut $r,e:*mut $r,z:*mut $t,ldz:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            let cz=compz.to_ascii_uppercase();
            if LAPACKE_get_nancheck()!=0 {
                if nan_vec(n as i64,d) {return -4;}
                if nan_vec((n-1).max(0) as i64,e) {return -5;}
                if cz==b'V' && super::ge_nancheck(layout,n as i64,n as i64,z,ldz as i64).unwrap_or(false) {return -6;}
            }
            let lw=if cz==b'N' {1} else {(4*n.max(0) as usize).saturating_sub(4).max(1)};
            let mut work=vec![<$r>::z();lw];
            [<LAPACKE_ $p pteqr_work $($suffix)*>](layout,compz,n,d,e,z,ldz,work.as_mut_ptr())
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p pteqr_work $($suffix)*>](
            layout:i32,compz:u8,n:$i,d:*mut $r,e:*mut $r,z:*mut $t,ldz:$i,work:*mut $r,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            let cz=compz.to_ascii_uppercase();
            if layout==ROW_MAJOR && ldz<n {return -7;}
            if layout==COL_MAJOR || n==0 {
                let mut info:$i=0;
                crate::tridiagonal_evd::[<$p pteqr $fortran>](&compz,&n,d,e,z,&ldz,work,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let mut zt=if cz==b'V' {
                match row_to_column(nn,nn,z,ldz as usize){Ok(v)=>v,Err(err)=>return err as $i}
            } else {
                vec![<$t>::z();nn*nn]
            };
            let ld=nn.max(1) as $i;
            let mut info:$i=0;
            crate::tridiagonal_evd::[<$p pteqr $fortran>](&compz,&n,d,e,zt.as_mut_ptr(),&ld,work,&mut info);
            if info>=0 && cz!=b'N' {
                for i in 0..nn {for j in 0..nn {*z.add(i*ldz as usize+j)=zt[i+j*nn];}}
            }
            if info<0 {info-1} else {info}
        }
    }};
}
exports!(s, f32, f32, i32, [], _);
exports!(d, f64, f64, i32, [], _);
exports!(c, colmat::c32, f32, i32, [], _);
exports!(z, colmat::c64, f64, i32, [], _);
exports!(s, f32, f32, i64, [64_], _64_);
exports!(d, f64, f64, i64, [64_], _64_);
exports!(c, colmat::c32, f32, i64, [64_], _64_);
exports!(z, colmat::c64, f64, i64, [64_], _64_);
