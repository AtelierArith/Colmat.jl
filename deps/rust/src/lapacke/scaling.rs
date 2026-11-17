//! LAPACKE LASCL high/work adapters.
//! Reference: Reference LAPACK v3.12.1 LAPACKE/src/lapacke_{s,d,c,z}lascl{,_work}.c.
use super::{ge_nancheck, row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;

fn rows(kind: u8, m: i64, kl: i64, ku: i64) -> i64 {
    match kind.to_ascii_uppercase() {
        b'B' => kl + 1,
        b'Q' => ku + 1,
        b'Z' => 2 * kl + ku + 1,
        _ => m,
    }
}

macro_rules! exports {
    ($p:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p lascl $($suffix)*>](
            layout:i32,kind:u8,kl:$i,ku:$i,cfrom:$r,cto:$r,m:$i,n:$i,a:*mut $t,lda:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                let nr=rows(kind,m as i64,kl as i64,ku as i64);
                if ge_nancheck(layout,nr,n as i64,a,lda as i64).unwrap_or(false) {return -9;}
            }
            [<LAPACKE_ $p lascl_work $($suffix)*>](layout,kind,kl,ku,cfrom,cto,m,n,a,lda)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p lascl_work $($suffix)*>](
            layout:i32,kind:u8,kl:$i,ku:$i,cfrom:$r,cto:$r,m:$i,n:$i,a:*mut $t,lda:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -9;}
            if layout==COL_MAJOR || m==0 || n==0 {
                let mut info:$i=0;
                crate::matrix_scaling::[<$p lascl $fortran>](&kind,&kl,&ku,&cfrom,&cto,&m,&n,a,&lda,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nr=rows(kind,m as i64,kl as i64,ku as i64) as usize;
            let nn=n as usize;
            let mut tmp=match row_to_column(nr,nn,a,lda as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut info:$i=0;
            let ld=nr.max(1) as $i;
            crate::matrix_scaling::[<$p lascl $fortran>](&kind,&kl,&ku,&cfrom,&cto,&m,&n,tmp.as_mut_ptr(),&ld,&mut info);
            if info>=0 {for i in 0..nr {for j in 0..nn {*a.add(i*lda as usize+j)=tmp[i+j*nr];}}}
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
