//! Generalized Schur pencil eigenvector LAPACKE adapters.
//! Reference: Reference LAPACK v3.12.1 LAPACKE/src/lapacke_{s,d,c,z}tgevc{,_work}.c.
//! The Fortran owner consumes the caller's S/T pencil; layouts are converted
//! without reading caller padding.
use super::{ge_nancheck, row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;

fn zeros<T: Scalar>(count: usize) -> Result<Vec<T>, i64> {
    if count
        .checked_mul(core::mem::size_of::<T>())
        .is_none_or(|n| n > isize::MAX as usize)
    {
        return Err(super::TRANSPOSE_MEMORY_ERROR);
    }
    let mut out = Vec::new();
    out.try_reserve_exact(count)
        .map_err(|_| super::TRANSPOSE_MEMORY_ERROR)?;
    out.resize(count, T::z());
    Ok(out)
}

macro_rules! tgevc {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt,$factor:expr)=>{paste::paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p tgevc $($suffix)*>](
            layout:i32,side:u8,howmny:u8,select:*const $i,n:$i,s:*const $t,lds:$i,
            p:*const $t,ldp:$i,vl:*mut $t,ldvl:$i,vr:*mut $t,ldvr:$i,mm:$i,m:*mut $i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            let side_u=side.to_ascii_uppercase();
            if LAPACKE_get_nancheck()!=0 {
                match ge_nancheck(layout,n as i64,n as i64,p,ldp as i64) {
                    Ok(true)=>return -8,Err(info)=>return info as $i,Ok(false)=>{},
                }
                match ge_nancheck(layout,n as i64,n as i64,s,lds as i64) {
                    Ok(true)=>return -6,Err(info)=>return info as $i,Ok(false)=>{},
                }
                if matches!(side_u,b'L'|b'B') {
                    match ge_nancheck(layout,n as i64,mm as i64,vl,ldvl as i64) {
                        Ok(true)=>return -10,Err(info)=>return info as $i,Ok(false)=>{},
                    }
                }
                if matches!(side_u,b'R'|b'B') {
                    match ge_nancheck(layout,n as i64,mm as i64,vr,ldvr as i64) {
                        Ok(true)=>return -12,Err(info)=>return info as $i,Ok(false)=>{},
                    }
                }
            }
            let size = if $factor { 6 } else { 2 };
            let Some(count) = (n.max(0) as usize).checked_mul(size) else {return -1010;};
            let mut work = match zeros::<$t>(count.max(1)) {
                Ok(v)=>v,Err(_)=>return -1010,
            };
            [<LAPACKE_ $p tgevc_work $($suffix)*>](
                layout,side,howmny,select,n,s,lds,p,ldp,vl,ldvl,vr,ldvr,mm,m,work.as_mut_ptr(),
            )
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p tgevc_work $($suffix)*>](
            layout:i32,side:u8,howmny:u8,select:*const $i,n:$i,s:*const $t,lds:$i,
            p:*const $t,ldp:$i,vl:*mut $t,ldvl:$i,vr:*mut $t,ldvr:$i,mm:$i,m:*mut $i,
            _work:*mut $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            let (side_u,how_u)=(side.to_ascii_uppercase(),howmny.to_ascii_uppercase());
            let left=matches!(side_u,b'L'|b'B');
            let right=matches!(side_u,b'R'|b'B');
            if layout==COL_MAJOR {
                let mut info:$i=0;
                crate::tgevc::[<$p tgevc $fortran>](
                    &side,&howmny,select,&n,s,&lds,p,&ldp,vl,&ldvl,vr,&ldvr,&mm,m,_work,&mut info,
                );
                return if info<0 {info-1} else {info};
            }
            if lds<n {return -7;}
            if ldp<n {return -9;}
            if ldvl<mm {return -11;}
            if ldvr<mm {return -13;}
            if n<0 {return 0;}
            if n==0 {
                let mut info:$i=0;
                crate::tgevc::[<$p tgevc $fortran>](
                    &side,&howmny,select,&n,s,&lds,p,&ldp,vl,&ldvl,vr,&ldvr,&mm,m,_work,&mut info,
                );
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let width=mm as usize;
            let mut st=match row_to_column(nn,nn,s,lds as usize) {
                Ok(v)=>v,Err(info)=>return info as $i,
            };
            let mut pt=match row_to_column(nn,nn,p,ldp as usize) {
                Ok(v)=>v,Err(info)=>return info as $i,
            };
            let back=how_u==b'B';
            let mut vlt=if left&&back {
                match row_to_column(nn,width,vl,ldvl as usize) {
                    Ok(v)=>v,Err(info)=>return info as $i,
                }
            } else {zeros::<$t>(nn.checked_mul(width.max(1)).unwrap_or(0)).unwrap_or_default()};
            let mut vrt=if right&&back {
                match row_to_column(nn,width,vr,ldvr as usize) {
                    Ok(v)=>v,Err(info)=>return info as $i,
                }
            } else {zeros::<$t>(nn.checked_mul(width.max(1)).unwrap_or(0)).unwrap_or_default()};
            let ld=n.max(1) as $i;
            let mut info:$i=0;
            crate::tgevc::[<$p tgevc $fortran>](
                &side,&howmny,select,&n,st.as_mut_ptr(),&ld,pt.as_mut_ptr(),&ld,
                vlt.as_mut_ptr(),&ld,vrt.as_mut_ptr(),&ld,&mm,m,_work,&mut info,
            );
            if info>=0 {
                if left {
                    for j in 0..width {for i in 0..nn {
                        *vl.add(i*ldvl as usize+j)=vlt[i+j*nn];
                    }}
                }
                if right {
                    for j in 0..width {for i in 0..nn {
                        *vr.add(i*ldvr as usize+j)=vrt[i+j*nn];
                    }}
                }
            }
            if info<0 {info-1} else {info}
        }
    }};
}
tgevc!(s, f32, i32, [], _, true);
tgevc!(d, f64, i32, [], _, true);
tgevc!(c, colmat::c32, i32, [], _, false);
tgevc!(z, colmat::c64, i32, [], _, false);
tgevc!(s, f32, i64, [64_], _64_, true);
tgevc!(d, f64, i64, [64_], _64_, true);
tgevc!(c, colmat::c32, i64, [64_], _64_, false);
tgevc!(z, colmat::c64, i64, [64_], _64_, false);
