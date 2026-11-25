//! Selected band/packed EVD, using a native full-spectrum dense expansion.
//! Positive ABSTOL and ambiguous value-range boundaries retain an expert
//! driver: filtering approximate full-spectrum eigenvalues cannot claim the
//! accuracy or endpoint-counting contract of a Sturm/bisection implementation.
use crate::generalized_selfadjoint::GeneralizedScalar;
use crate::kernels::{flag, Scalar};
use crate::lapack::LapackInt;
use crate::structured_evd::{compute, Selection, Storage};
use paste::paste;
use std::ffi::c_void;
use std::sync::atomic::{AtomicPtr, Ordering};

pub(crate) static FALLBACKS: [AtomicPtr<c_void>; 8] =
    [const { AtomicPtr::new(std::ptr::null_mut()) }; 8];

unsafe fn prepare<R: Scalar, I: LapackInt>(
    job: u8,
    mode: u8,
    triangle: u8,
    n: isize,
    band: Option<(isize, isize, isize)>,
    vl: *const R,
    vu: *const R,
    il: *const I,
    iu: *const I,
    ldz: isize,
) -> Result<Selection, i64> {
    for (arg, valid) in [
        (1, matches!(job, b'N' | b'V')),
        (2, matches!(mode, b'A' | b'V' | b'I')),
        (3, matches!(triangle, b'U' | b'L')),
        (4, n >= 0),
    ] {
        if !valid {
            return Err(arg);
        }
    }
    if let Some((kd, ldab, ldq)) = band {
        if kd < 0 {
            return Err(5);
        }
        if !kd.checked_add(1).is_some_and(|min| ldab >= min) {
            return Err(7);
        }
        // Unlike LDZ, LDQ is not constrained for JOBZ=N.
        if job == b'V' && ldq < n.max(1) {
            return Err(9);
        }
    }
    let offset = if band.is_some() { 4 } else { 0 };
    let selected = match mode {
        b'A' => Selection::All,
        b'V' if n == 0 => Selection::All,
        b'V' => {
            let (lo, hi) = ((*vl).re(), (*vu).re());
            if hi <= lo {
                return Err(7 + offset);
            }
            Selection::Value(lo, hi)
        }
        _ => {
            let (lo, hi) = ((*il).val(), (*iu).val());
            if lo < 1 || lo > n.max(1) {
                return Err(8 + offset);
            }
            if hi < lo.min(n) || hi > n {
                return Err(9 + offset);
            }
            Selection::Index((lo - 1) as usize, hi as usize)
        }
    };
    if ldz < 1 || (job == b'V' && ldz < n) {
        return Err(14 + offset);
    }
    Ok(selected)
}

fn addressable<T>(n: isize, band: Option<(isize, isize, isize)>, job: u8, ldz: isize) -> bool {
    let bytes = std::mem::size_of::<T>() as isize;
    let fits = |rows: isize, cols: isize| {
        rows.checked_mul(cols)
            .and_then(|x| x.checked_mul(bytes))
            .is_some()
    };
    if !fits(n, n) {
        return false;
    }
    if job == b'V' && !fits(n, ldz) {
        return false;
    }
    match band {
        Some((_, ldab, ldq)) => fits(n, ldab) && (job != b'V' || fits(n, ldq)),
        None => n
            .checked_add(1)
            .and_then(|x| n.checked_mul(x))
            .and_then(|x| (x / 2).checked_mul(bytes))
            .is_some(),
    }
}

unsafe fn native<T: GeneralizedScalar, R: Scalar, I: LapackInt>(
    job: u8,
    triangle: u8,
    n: usize,
    band: Option<(isize, isize, isize)>,
    a: *mut T,
    q: *mut T,
    selected: Selection,
    abstol: *const R,
    m: *mut I,
    w: *mut R,
    z: *mut T,
    ldz: isize,
    ifail: *mut I,
) -> Result<(), ()> {
    let storage = match band {
        Some((kd, ldab, _)) => Storage::Band {
            width: kd as usize,
            ld: ldab as usize,
        },
        None => Storage::Packed,
    };
    *m = I::from_i64(0);
    if n == 0 {
        return Ok(());
    }
    if n == 1 {
        let value = T::diagonal(a.add(storage.index(1, triangle == b'L', 0, 0))).re();
        let included = match selected {
            Selection::Value(lo, hi) => lo < value && value <= hi,
            _ => true,
        };
        if included {
            *m = I::from_i64(1);
            *w = R::real(value);
        }
        // Packed VX writes its scalar Z even for an empty selected interval;
        // band VX writes Z only when selected. Neither quick return writes Q.
        if job == b'V' && (included || band.is_none()) {
            *z = T::o();
        }
        return Ok(());
    }
    if (*abstol).re() > 0.0 {
        return Err(());
    }
    let spectrum = compute(n, triangle == b'L', storage, a, job == b'V')?;
    if let Selection::Value(lo, hi) = selected {
        let scale = spectrum.values.iter().fold(0.0f64, |x, y| x.max(y.abs()));
        let epsilon = if std::mem::size_of::<R>() == 4 {
            f32::EPSILON as f64
        } else {
            f64::EPSILON
        };
        let tolerance = (64.0 * n as f64 * epsilon) * scale;
        if spectrum
            .values
            .iter()
            .any(|&v| (v - lo).abs() <= tolerance || (v - hi).abs() <= tolerance)
        {
            return Err(());
        }
    }
    let indices = spectrum.selected(selected);
    for (column, &index) in indices.iter().enumerate() {
        *w.add(column) = R::real(spectrum.values[index]);
        if job == b'V' {
            *ifail.add(column) = I::from_i64(0);
            for row in 0..n {
                *z.add(row + column * ldz as usize) = spectrum.vectors[(row, index)];
            }
        }
    }
    if job == b'V' {
        if let Some((_, _, ldq)) = band {
            for j in 0..n {
                for i in 0..n {
                    *q.add(i + j * ldq as usize) = spectrum.vectors[(i, j)];
                }
            }
        }
    }
    spectrum.write_diagonal_reduction(triangle == b'L', storage, a);
    *m = I::from_i64(indices.len() as i64);
    Ok(())
}

macro_rules! selected_exports {
    ($p:ident,$bev:ident,$pev:ident,$t:ty,$r:ty,$i:ty,$suffix:tt,$id:expr,
     ($($extra:ident:$extra_ty:ty,)*))=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $bev $suffix>](
            jobz:*const u8,range:*const u8,uplo:*const u8,n:*const $i,kd:*const $i,
            ab:*mut $t,ldab:*const $i,q:*mut $t,ldq:*const $i,vl:*const $r,vu:*const $r,
            il:*const $i,iu:*const $i,abstol:*const $r,m:*mut $i,w:*mut $r,z:*mut $t,ldz:*const $i,
            work:*mut $t,$($extra:$extra_ty,)*iwork:*mut $i,ifail:*mut $i,info:*mut $i) {
            let(job,mode,triangle,nn,lz)=(flag(jobz),flag(range),flag(uplo),*n as isize,*ldz as isize);
            let band=Some((*kd as isize,*ldab as isize,*ldq as isize));
            *info=0;
            let selection=match prepare(job,mode,triangle,nn,band,vl,vu,il,iu,lz) {
                Ok(s)=>s,Err(arg)=>{*info=-arg as $i;crate::invalid(arg as i32);return;}
            };
            if !addressable::<$t>(nn,band,job,lz) {*info=1;return;}
            if native(job,triangle,nn as usize,band,ab,q,selection,abstol,m,w,z,lz,ifail).is_ok() {return;}
            let pointer=FALLBACKS[$id].load(Ordering::Acquire);
            if pointer.is_null() {*info=1;return;}
            let f:unsafe extern "C" fn(*const u8,*const u8,*const u8,*const i64,*const i64,
                *mut $t,*const i64,*mut $t,*const i64,*const $r,*const $r,*const i64,*const i64,
                *const $r,*mut i64,*mut $r,*mut $t,*const i64,*mut $t,$($extra_ty,)*
                *mut i64,*mut i64,*mut i64)=std::mem::transmute(pointer);
            let(mut count,mut status)=(0i64,0i64);
            let mut integers=vec![0i64;5*nn as usize];
            let mut failed=vec![0i64;nn as usize];
            let(lo,hi)=if mode==b'I'{(*il as i64,*iu as i64)}else{(1,nn as i64)};
            f(jobz,range,uplo,&(nn as i64),&(*kd as i64),ab,&(*ldab as i64),q,&(*ldq as i64),
                vl,vu,&lo,&hi,abstol,&mut count,w,z,&(lz as i64),work,$($extra,)*
                integers.as_mut_ptr(),failed.as_mut_ptr(),&mut status);
            *m=count as $i;*info=status as $i;
            for (j,&value) in integers.iter().enumerate() {*iwork.add(j)=value as $i;}
            if job==b'V' {for j in 0..count.max(0).min(nn as i64) as usize {*ifail.add(j)=failed[j] as $i;}}
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $pev $suffix>](
            jobz:*const u8,range:*const u8,uplo:*const u8,n:*const $i,ap:*mut $t,
            vl:*const $r,vu:*const $r,il:*const $i,iu:*const $i,abstol:*const $r,
            m:*mut $i,w:*mut $r,z:*mut $t,ldz:*const $i,work:*mut $t,$($extra:$extra_ty,)*
            iwork:*mut $i,ifail:*mut $i,info:*mut $i) {
            let(job,mode,triangle,nn,lz)=(flag(jobz),flag(range),flag(uplo),*n as isize,*ldz as isize);
            *info=0;
            let selection=match prepare(job,mode,triangle,nn,None,vl,vu,il,iu,lz) {
                Ok(s)=>s,Err(arg)=>{*info=-arg as $i;crate::invalid(arg as i32);return;}
            };
            if !addressable::<$t>(nn,None,job,lz) {*info=1;return;}
            if native(job,triangle,nn as usize,None,ap,std::ptr::null_mut(),selection,abstol,m,w,z,lz,ifail).is_ok() {return;}
            let pointer=FALLBACKS[4+$id].load(Ordering::Acquire);
            if pointer.is_null() {*info=1;return;}
            let f:unsafe extern "C" fn(*const u8,*const u8,*const u8,*const i64,*mut $t,
                *const $r,*const $r,*const i64,*const i64,*const $r,*mut i64,*mut $r,*mut $t,
                *const i64,*mut $t,$($extra_ty,)* *mut i64,*mut i64,*mut i64)=std::mem::transmute(pointer);
            let(mut count,mut status)=(0i64,0i64);
            let mut integers=vec![0i64;5*nn as usize];
            let mut failed=vec![0i64;nn as usize];
            let(lo,hi)=if mode==b'I'{(*il as i64,*iu as i64)}else{(1,nn as i64)};
            f(jobz,range,uplo,&(nn as i64),ap,vl,vu,&lo,&hi,abstol,&mut count,w,z,&(lz as i64),
                work,$($extra,)*integers.as_mut_ptr(),failed.as_mut_ptr(),&mut status);
            *m=count as $i;*info=status as $i;
            for (j,&value) in integers.iter().enumerate() {*iwork.add(j)=value as $i;}
            if job==b'V' {for j in 0..count.max(0).min(nn as i64) as usize {*ifail.add(j)=failed[j] as $i;}}
        }
    }};
}
macro_rules! selected_abi {
    ($i:ty,$suffix:tt)=>{
        selected_exports!(s,sbevx,spevx,f32,f32,$i,$suffix,0,());
        selected_exports!(d,sbevx,spevx,f64,f64,$i,$suffix,1,());
        selected_exports!(c,hbevx,hpevx,colmat::c32,f32,$i,$suffix,2,(rwork:*mut f32,));
        selected_exports!(z,hbevx,hpevx,colmat::c64,f64,$i,$suffix,3,(rwork:*mut f64,));
    };
}
selected_abi!(i32, "_");
selected_abi!(i64, "_64_");

#[cfg(test)]
mod tests {
    use super::*;
    macro_rules! selected_test {
        ($name:ident,$bev:ident,$pev:ident,$t:ty,$r:ty,$i:ty,($($extra:expr,)*))=>{
            #[test]
            fn $name() { unsafe {
                for lower in [false,true] {for job in [b'N',b'V'] {for mode in [b'A',b'I',b'V'] {
                    for empty in [false,true] {
                        if empty&&mode!=b'V' {continue;}
                        let triangle=if lower {b'L'}else{b'U'};
                        let(vl,vu)=if empty {(0.0 as $r,0.5 as $r)}else{(1.5 as $r,4.0 as $r)};
                        let il:$i=2;let iu:$i=2;
                        let(lo,hi)=if mode==b'V'{(&vl as *const $r,&vu as *const $r)}else{(std::ptr::null(),std::ptr::null())};
                        let(first,last)=if mode==b'I'{(&il as *const $i,&iu as *const $i)}else{(std::ptr::null(),std::ptr::null())};
                        for band in [false,true] {
                            let mut a=if band {
                                if lower {vec![<$t>::real(2.0),<$t>::real(1.0),<$t>::real(777.0),
                                    <$t>::real(2.0),<$t>::real(f64::NAN),<$t>::real(777.0)]}
                                else {vec![<$t>::real(f64::NAN),<$t>::real(2.0),<$t>::real(777.0),
                                    <$t>::real(1.0),<$t>::real(2.0),<$t>::real(777.0)]}
                            }else{vec![<$t>::real(2.0),<$t>::real(1.0),<$t>::real(2.0),<$t>::real(777.0)]};
                            let mut q=[<$t>::real(777.0);7];
                            let mut z=[<$t>::real(777.0);7];
                            let mut w=[777.0 as $r;3];
                            let mut failed=[777 as $i;3];
                            let mut count:$i=-99;let mut info:$i=-99;
                            if band {
                                $bev(&job,&mode,&triangle,&2,&1,a.as_mut_ptr(),&3,q.as_mut_ptr(),
                                    &(if job==b'V'{3}else{-99}),lo,hi,first,last,&0.0,&mut count,w.as_mut_ptr(),
                                    z.as_mut_ptr(),&3,std::ptr::null_mut(),$($extra,)*std::ptr::null_mut(),failed.as_mut_ptr(),&mut info);
                            }else{
                                $pev(&job,&mode,&triangle,&2,a.as_mut_ptr(),lo,hi,first,last,&0.0,&mut count,
                                    w.as_mut_ptr(),z.as_mut_ptr(),&3,std::ptr::null_mut(),$($extra,)*std::ptr::null_mut(),failed.as_mut_ptr(),&mut info);
                            }
                            assert_eq!(info,0);
                            let expected=if empty {0}else if mode==b'A'{2}else{1};
                            assert_eq!(count,expected);
                            for j in 0..expected as usize {
                                let value=if mode==b'A'&&j==0{1.0}else{3.0};
                                assert!((w[j] as f64-value).abs()<1e-5);
                                if job==b'V' {
                                    assert_eq!(failed[j],0);
                                    let(x,y)=(z[3*j],z[1+3*j]);
                                    assert!((x*<$t>::real(2.0)+y-x*<$t>::real(value)).magnitude()<1e-5);
                                    assert!((x+y*<$t>::real(2.0)-y*<$t>::real(value)).magnitude()<1e-5);
                                    assert!((x.magnitude().powi(2)+y.magnitude().powi(2)-1.0).abs()<1e-5);
                                    assert_eq!(z[2+3*j],<$t>::real(777.0));
                                }
                            }
                            for j in expected as usize..3 {assert_eq!(w[j],777.0);assert_eq!(failed[j],777);}
                            let written=if job==b'V'{expected as usize}else{0};
                            for j in written..2 {for i in 0..3 {assert_eq!(z[i+3*j],<$t>::real(777.0));}}
                            assert_eq!(z[6],<$t>::real(777.0));
                            if band&&job==b'V' {
                                // Q and the diagonal reduction reconstruct the
                                // original matrix, even when the selected Z is empty.
                                for j in 0..2 {for i in 0..2 {
                                    let value=q[i]*q[j].cj()+q[i+3]*<$t>::real(3.0)*q[j+3].cj();
                                    assert!((value-<$t>::real(if i==j{2.0}else{1.0})).magnitude()<1e-5);
                                }}
                                assert_eq!(q[2],<$t>::real(777.0));assert_eq!(q[5],<$t>::real(777.0));assert_eq!(q[6],<$t>::real(777.0));
                            }else{assert!(q.iter().all(|&x|x==<$t>::real(777.0)));}
                            if band {assert_eq!(a[2],<$t>::real(777.0));assert_eq!(a[5],<$t>::real(777.0));}
                            else{assert_eq!(a[3],<$t>::real(777.0));}
                        }
                    }
                }}}
                let mut count:$i=-99;let mut info:$i=-99;
                $bev(&b'N',&b'A',&b'U',&0,&0,std::ptr::null_mut(),&1,std::ptr::null_mut(),&-1,
                    std::ptr::null(),std::ptr::null(),std::ptr::null(),std::ptr::null(),std::ptr::null(),
                    &mut count,std::ptr::null_mut(),std::ptr::null_mut(),&1,std::ptr::null_mut(),$($extra,)*
                    std::ptr::null_mut(),std::ptr::null_mut(),&mut info);
                assert_eq!((count,info),(0,0));
                $pev(&b'N',&b'V',&b'U',&0,std::ptr::null_mut(),std::ptr::null(),std::ptr::null(),
                    std::ptr::null(),std::ptr::null(),std::ptr::null(),&mut count,std::ptr::null_mut(),
                    std::ptr::null_mut(),&1,std::ptr::null_mut(),$($extra,)*std::ptr::null_mut(),std::ptr::null_mut(),&mut info);
                assert_eq!((count,info),(0,0));
            }}
        };
    }
    selected_test!(s32, ssbevx_, sspevx_, f32, f32, i32, ());
    selected_test!(d32, dsbevx_, dspevx_, f64, f64, i32, ());
    selected_test!(
        c32,
        chbevx_,
        chpevx_,
        colmat::c32,
        f32,
        i32,
        (std::ptr::null_mut(),)
    );
    selected_test!(
        z32,
        zhbevx_,
        zhpevx_,
        colmat::c64,
        f64,
        i32,
        (std::ptr::null_mut(),)
    );
    selected_test!(s64, ssbevx_64_, sspevx_64_, f32, f32, i64, ());
    selected_test!(d64, dsbevx_64_, dspevx_64_, f64, f64, i64, ());
    selected_test!(
        c64,
        chbevx_64_,
        chpevx_64_,
        colmat::c32,
        f32,
        i64,
        (std::ptr::null_mut(),)
    );
    selected_test!(
        z64,
        zhbevx_64_,
        zhpevx_64_,
        colmat::c64,
        f64,
        i64,
        (std::ptr::null_mut(),)
    );

    #[test]
    fn ambiguous_boundaries_leave_the_original_storage_for_expert_recovery() {
        unsafe {
            let mut a = [2.0f64, 1.0, 2.0];
            let before = a;
            let mut m = -99i32;
            let mut w = [777.0; 2];
            let mut z = [777.0; 4];
            let mut fail = [777i32; 2];
            assert!(native(
                b'V',
                b'U',
                2,
                None,
                a.as_mut_ptr(),
                std::ptr::null_mut(),
                Selection::Value(1.0, 3.0),
                &0.0,
                &mut m,
                w.as_mut_ptr(),
                z.as_mut_ptr(),
                2,
                fail.as_mut_ptr()
            )
            .is_err());
            assert_eq!(a, before);
            assert_eq!(m, 0);
            assert_eq!(w, [777.0; 2]);
            assert_eq!(z, [777.0; 4]);
            assert_eq!(fail, [777; 2]);
        }
    }
}
