//! Selected self-adjoint LAPACKE adapters with real by-value range endpoints.
use super::{
    cholesky::po_nancheck, self_adjoint::zeros, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR,
    TRANSPOSE_MEMORY_ERROR,
};
use crate::{kernels::Scalar, lapack::LapackInt};

fn span<T>(rows: usize, cols: usize, ld: usize) -> Result<(), i64> {
    if rows == 0 || cols == 0 {
        return Ok(());
    }
    if (rows - 1)
        .checked_mul(ld)
        .and_then(|x| x.checked_add(cols))
        .and_then(|x| x.checked_mul(core::mem::size_of::<T>()))
        .is_none_or(|x| x > isize::MAX as usize)
    {
        return Err(TRANSPOSE_MEMORY_ERROR);
    }
    Ok(())
}

unsafe fn work<T: Scalar, I: LapackInt>(
    layout: i32,
    job: u8,
    mode: u8,
    uplo: u8,
    n: i64,
    a: *mut T,
    lda: i64,
    vl: f64,
    vu: f64,
    il: i64,
    iu: i64,
    m: *mut I,
    z: *mut T,
    ldz: i64,
    query: bool,
    validate_workspace: impl FnOnce() -> Result<(), i64>,
    mut call: impl FnMut(*mut T, i64, *mut T, i64) -> i64,
) -> i64 {
    if layout != ROW_MAJOR && layout != COL_MAJOR {
        return -1;
    }
    let (job, mode, uplo) = (
        job.to_ascii_uppercase(),
        mode.to_ascii_uppercase(),
        uplo.to_ascii_uppercase(),
    );
    if layout == ROW_MAJOR {
        if lda < n {
            return -7;
        }
        let columns = if job != b'V' {
            1i128
        } else if matches!(mode, b'A' | b'V') {
            n as i128
        } else if mode == b'I' {
            iu as i128 - il as i128 + 1
        } else {
            1
        };
        if columns > i64::MAX as i128 || columns < i64::MIN as i128 {
            return TRANSPOSE_MEMORY_ERROR;
        }
        if (ldz as i128) < columns {
            return -16;
        }
    }
    if !matches!(job, b'N' | b'V') {
        return -2;
    }
    if !matches!(mode, b'A' | b'I' | b'V') {
        return -3;
    }
    if !matches!(uplo, b'U' | b'L') {
        return -4;
    }
    if n < 0 {
        return -5;
    }
    if layout == COL_MAJOR && lda < n.max(1) {
        return -7;
    }
    if mode == b'V' && n > 0 && !(vl < vu) {
        return -9;
    }
    if mode == b'I' {
        if il < 1 || il > n.max(1) {
            return -10;
        }
        if iu < n.min(il) || iu > n {
            return -11;
        }
    }
    if layout == COL_MAJOR && (ldz < 1 || (job == b'V' && ldz < n)) {
        return -16;
    }
    if let Err(e) = validate_workspace() {
        return e;
    }
    if layout == COL_MAJOR {
        return call(a, lda, z, ldz);
    }
    let ld = n.max(1);
    if query {
        return call(a, ld, z, ld);
    }
    let nn = n as usize;
    if let Err(e) = span::<T>(nn, nn, lda as usize) {
        return e;
    }
    let count = (ld as usize).checked_mul(ld as usize).unwrap_or(usize::MAX);
    let mut at = match zeros::<T>(count) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let mut zt = if job == b'V' {
        match zeros::<T>(count) {
            Ok(v) => v,
            Err(e) => return e,
        }
    } else {
        Vec::new()
    };
    for i in 0..nn {
        for j in 0..nn {
            if (uplo == b'L' && i >= j) || (uplo == b'U' && i <= j) {
                at[i + j * ld as usize] = *a.add(i * lda as usize + j);
            }
        }
    }
    let info = call(at.as_mut_ptr(), ld, zt.as_mut_ptr(), ld);
    if info < 0 {
        return info;
    }
    for i in 0..nn {
        for j in 0..nn {
            if (uplo == b'L' && i >= j) || (uplo == b'U' && i <= j) {
                *a.add(i * lda as usize + j) = at[i + j * ld as usize];
            }
        }
    }
    if job == b'V' {
        // Only returned columns are meaningful. The scalar quick return also
        // writes Z=1 when its single eigenvalue was excluded by the interval.
        let cols = if n == 1 {
            1
        } else {
            (*m).val().clamp(0, n as isize) as usize
        };
        if let Err(e) = span::<T>(nn, cols, ldz as usize) {
            return e;
        }
        for i in 0..nn {
            for j in 0..cols {
                *z.add(i * ldz as usize + j) = zt[i + j * ld as usize];
            }
        }
    }
    info
}

macro_rules! exports {
    ($p:ident,$family:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$f:tt,[$($extra:ident)*]) => {paste::paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $family _work $($suffix)*>](
            layout:i32,job:u8,mode:u8,uplo:u8,n:$i,a:*mut $t,lda:$i,vl:$r,vu:$r,il:$i,iu:$i,
            abstol:$r,m:*mut $i,values:*mut $r,z:*mut $t,ldz:$i,support:*mut $i,
            workspace:*mut $t,lwork:$i,$($extra:*mut $r,[<l $extra>]:$i,)*iwork:*mut $i,liwork:$i,
        )-> $i {
            let query=lwork== -1 || liwork== -1 $(|| [<l $extra>]== -1)*;
            work(layout,job,mode,uplo,n as i64,a,lda as i64,vl as f64,vu as f64,il as i64,iu as i64,
                m,z,ldz as i64,query,|| {
                    let nn=n.max(0) as i128;
                    let nw=if n<=1 {1} else {nn*if <$t>::COMPLEX {2} else {26}};
                    let _nr=if n<=1 {1} else {24*nn};let ni=if n<=1 {1} else {10*nn};
                    if nw>isize::MAX as i128 || (<$t>::COMPLEX && _nr>isize::MAX as i128) || ni>(<$i>::MAX as i128) {return Err(TRANSPOSE_MEMORY_ERROR);}
                    if !query {
                        if (lwork as i128)<nw {return Err(-19);}
                        $(if ([<l $extra>] as i128)<_nr {return Err(-21);})*
                        if (liwork as i128)<ni {return Err(if <$t>::COMPLEX {-23} else {-21});}
                    }
                    Ok(())
                },|a,ld,z,lz| {
                    let mut info:$i=0;
                    crate::lapack_symmetric::[<$p $family $f>](&job,&mode,&uplo,&n,a,&(ld as $i),&vl,&vu,&il,&iu,
                        &abstol,m,values,z,&(lz as $i),support,workspace,&lwork,$($extra,&[<l $extra>],)*iwork,&liwork,&mut info);
                    if info<0 {(info-1) as i64} else {info as i64}
                }) as $i
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $family $($suffix)*>](
            layout:i32,job:u8,mode:u8,uplo:u8,n:$i,a:*mut $t,lda:$i,vl:$r,vu:$r,il:$i,iu:$i,
            abstol:$r,m:*mut $i,values:*mut $r,z:*mut $t,ldz:$i,support:*mut $i,
        )-> $i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                match po_nancheck(layout,uplo,n as i64,a,lda as i64) {Ok(true)=>return -6,Err(e)=>return e as $i,Ok(false)=>{}}
                if abstol.is_nan() {return -12;}
                if mode.to_ascii_uppercase()==b'V' {
                    if vl.is_nan() {return -8;}if vu.is_nan() {return -9;}
                }
            }
            let mut wq=<$t>::z();let mut _rq=0 as $r;let mut iq=0 as $i;
            $(let $extra=&mut _rq as *mut $r;let [<l $extra>]:$i=-1;)*
            let info=[<LAPACKE_ $p $family _work $($suffix)*>](layout,job,mode,uplo,n,a,lda,vl,vu,il,iu,abstol,m,values,z,ldz,support,
                &mut wq,-1,$($extra,[<l $extra>],)*&mut iq,-1);
            if info!=0 {return info;}
            let nw=wq.re();let nr=if <$t>::COMPLEX {_rq as f64} else {1.0};
            if [nw,nr].iter().any(|&v|!v.is_finite() || v<1.0 || v>=isize::MAX as f64 || v > <$i>::MAX as f64) || iq<1 {return -1010;}
            let mut w=match zeros::<$t>(nw as usize) {Ok(v)=>v,Err(_)=>return -1010};
            let mut _rw=match zeros::<$r>(nr as usize) {Ok(v)=>v,Err(_)=>return -1010};
            let mut iw=Vec::<$i>::new();
            if (iq as usize).checked_mul(core::mem::size_of::<$i>()).is_none_or(|v|v>isize::MAX as usize) || iw.try_reserve_exact(iq as usize).is_err() {return -1010;}
            iw.resize(iq as usize,0);
            $(let $extra=_rw.as_mut_ptr();let [<l $extra>]=nr as $i;)*
            [<LAPACKE_ $p $family _work $($suffix)*>](layout,job,mode,uplo,n,a,lda,vl,vu,il,iu,abstol,m,values,z,ldz,support,
                w.as_mut_ptr(),nw as $i,$($extra,[<l $extra>],)*iw.as_mut_ptr(),iq)
        }
    }};
}
exports!(s, syevr, f32, f32, i32, [], _, []);
exports!(d, syevr, f64, f64, i32, [], _, []);
exports!(c, heevr, colmat::c32, f32, i32, [], _, [rwork]);
exports!(z, heevr, colmat::c64, f64, i32, [], _, [rwork]);
exports!(s, syevr, f32, f32, i64, [64_], _64_, []);
exports!(d, syevr, f64, f64, i64, [64_], _64_, []);
exports!(c, heevr, colmat::c32, f32, i64, [64_], _64_, [rwork]);
exports!(z, heevr, colmat::c64, f64, i64, [64_], _64_, [rwork]);
