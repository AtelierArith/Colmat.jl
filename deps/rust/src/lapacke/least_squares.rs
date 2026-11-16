//! Least-squares storage: A is M-by-N; B has max(M,N) output rows.
use super::{
    ge_nancheck, row_to_column, self_adjoint::zeros, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR,
};
use crate::kernels::Scalar;

fn gels_geometry(
    layout: i32,
    trans: u8,
    m: i64,
    n: i64,
    nrhs: i64,
    lda: i64,
    ldb: i64,
    complex: bool,
) -> Result<(), i64> {
    if !matches!(layout, ROW_MAJOR | COL_MAJOR) {
        return Err(-1);
    }
    if layout == ROW_MAJOR {
        if lda < n {
            return Err(-7);
        }
        if ldb < nrhs {
            return Err(-9);
        }
    }
    let trans = trans.to_ascii_uppercase();
    if trans != b'N' && trans != if complex { b'C' } else { b'T' } {
        return Err(-2);
    }
    if m < 0 {
        return Err(-3);
    }
    if n < 0 {
        return Err(-4);
    }
    if nrhs < 0 {
        return Err(-5);
    }
    if layout == COL_MAJOR {
        if lda < m.max(1) {
            return Err(-7);
        }
        if ldb < m.max(n).max(1) {
            return Err(-9);
        }
    }
    Ok(())
}

unsafe fn layout_work<T: Scalar>(
    layout: i32,
    m: i64,
    n: i64,
    nrhs: i64,
    input_rows: i64,
    a: *mut T,
    lda: i64,
    b: *mut T,
    ldb: i64,
    skip_packing: bool,
    mut call: impl FnMut(*mut T, i64, *mut T, i64) -> i64,
) -> i64 {
    if layout == COL_MAJOR {
        return call(a, lda, b, ldb);
    }
    let rows = m.max(n);
    if skip_packing {
        return call(a, m.max(1), b, rows.max(1));
    }
    let mut at = match row_to_column(m as usize, n as usize, a, lda as usize) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let count = match (rows as usize).checked_mul(nrhs as usize) {
        Some(v) => v,
        None => return -1011,
    };
    let mut bt = match zeros::<T>(count) {
        Ok(v) => v,
        Err(e) => return e,
    };
    // Empty GELS writes zeros without reading either input. Otherwise read
    // only RHS input rows, independently of its larger solution allocation.
    if m > 0 && n > 0 {
        if !crate::kernels::matrix_addressable::<T>(
            nrhs as isize,
            input_rows as isize,
            ldb as isize,
        ) {
            return -1011;
        }
        for j in 0..nrhs as usize {
            for i in 0..input_rows as usize {
                bt[i + j * rows as usize] = *b.add(i * ldb as usize + j);
            }
        }
    }
    if !crate::kernels::matrix_addressable::<T>(nrhs as isize, rows as isize, ldb as isize) {
        return -1011;
    }
    let info = call(at.as_mut_ptr(), m.max(1), bt.as_mut_ptr(), rows.max(1));
    if info >= 0 {
        for j in 0..n as usize {
            for i in 0..m as usize {
                *a.add(i * lda as usize + j) = at[i + j * m as usize];
            }
        }
        for j in 0..nrhs as usize {
            for i in 0..rows as usize {
                *b.add(i * ldb as usize + j) = bt[i + j * rows as usize];
            }
        }
    }
    info
}

macro_rules! exports {($p:ident,$t:ty,$j:ty,[$($suffix:tt)*],$f:tt)=>{paste::paste!{
    #[no_mangle]
    pub unsafe extern "C" fn [<LAPACKE_ $p gels_work $($suffix)*>](layout:i32,trans:i8,m:$j,n:$j,nrhs:$j,a:*mut $t,lda:$j,b:*mut $t,ldb:$j,workspace:*mut $t,lwork:$j)->$j {
        if let Err(e)=gels_geometry(layout,trans as u8,m as i64,n as i64,nrhs as i64,lda as i64,ldb as i64,<$t>::COMPLEX){return e as $j;}
        let minimum=(m.min(n) as i128+(m.min(n).max(nrhs) as i128)).max(1);
        if minimum>(<$j>::MAX as i128){return -11;}
        let input_rows=if (trans as u8).to_ascii_uppercase()==b'N'{m}else{n};
        layout_work(layout,m as i64,n as i64,nrhs as i64,input_rows as i64,a,lda as i64,b,ldb as i64,lwork== -1||(lwork as i128)<minimum||nrhs==0,|a,lda,b,ldb|{
            let mut info:$j=0;
            crate::least_squares::[<$p gels $f>](&(trans as u8),&m,&n,&nrhs,a,&(lda as $j),b,&(ldb as $j),workspace,&lwork,&mut info);
            if info<0 {(info-1) as i64}else{info as i64}
        }) as $j
    }
    #[no_mangle]
    pub unsafe extern "C" fn [<LAPACKE_ $p gels $($suffix)*>](layout:i32,trans:i8,m:$j,n:$j,nrhs:$j,a:*mut $t,lda:$j,b:*mut $t,ldb:$j)->$j {
        if !matches!(layout,ROW_MAJOR|COL_MAJOR){return -1;}
        if LAPACKE_get_nancheck()!=0 {
            match ge_nancheck(layout,m as i64,n as i64,a,lda as i64){Ok(true)=>return -6,Err(e)=>return e as $j,_=>{}}
            match ge_nancheck(layout,m.max(n) as i64,nrhs as i64,b,ldb as i64){Ok(true)=>return -8,Err(e)=>return e as $j,_=>{}}
        }
        let mut query=<$t>::z();
        let info=[<LAPACKE_ $p gels_work $($suffix)*>](layout,trans,m,n,nrhs,a,lda,b,ldb,&mut query,-1);
        if info!=0 {return info;}
        if !query.re().is_finite() || query.re()<1.0 {return -1010;}
        let count=query.re() as usize;
        if count>(<$j>::MAX as usize){return -1010;}
        let mut workspace=match zeros::<$t>(count){Ok(v)=>v,Err(_)=>return -1010};
        [<LAPACKE_ $p gels_work $($suffix)*>](layout,trans,m,n,nrhs,a,lda,b,ldb,workspace.as_mut_ptr(),count as $j)
    }
}};}
exports!(s, f32, i32, [], _);
exports!(d, f64, i32, [], _);
exports!(c, colmat::c32, i32, [], _);
exports!(z, colmat::c64, i32, [], _);
exports!(s, f32, i64, [64_], _64_);
exports!(d, f64, i64, [64_], _64_);
exports!(c, colmat::c32, i64, [64_], _64_);
exports!(z, colmat::c64, i64, [64_], _64_);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lapack::LapackInt;
    type High<T, I> = unsafe extern "C" fn(i32, i8, I, I, I, *mut T, I, *mut T, I) -> I;
    type Work<T, I> = unsafe extern "C" fn(i32, i8, I, I, I, *mut T, I, *mut T, I, *mut T, I) -> I;
    unsafe fn empty<T: Scalar, I: LapackInt>(high: High<T, I>, work: Work<T, I>) {
        for layout in [101, 102] {
            for high_level in [false, true] {
                for (m, n, r) in [(0, 0, 2), (0, 3, 2), (3, 0, 2), (3, 5, 0)] {
                    let rows = m.max(n);
                    let lda = if layout == 101 { n + 2 } else { m + 2 };
                    let ldb = if layout == 101 { r + 3 } else { rows + 3 };
                    let mut a = [T::real(777.0); 64];
                    let mut b = [T::real(778.0); 64];
                    let mut workspace = [T::real(779.0); 64];
                    let ii = I::from_i64;
                    let info = if high_level {
                        high(
                            layout,
                            b'N' as i8,
                            ii(m),
                            ii(n),
                            ii(r),
                            a.as_mut_ptr().add(1),
                            ii(lda),
                            b.as_mut_ptr().add(1),
                            ii(ldb),
                        )
                    } else {
                        work(
                            layout,
                            b'N' as i8,
                            ii(m),
                            ii(n),
                            ii(r),
                            a.as_mut_ptr().add(1),
                            ii(lda),
                            b.as_mut_ptr().add(1),
                            ii(ldb),
                            workspace.as_mut_ptr().add(1),
                            ii(62),
                        )
                    };
                    assert_eq!(info.val(), 0);
                    assert!(a.iter().all(|x| *x == T::real(777.0)));
                    let mut touched = [false; 64];
                    for j in 0..r as usize {
                        for i in 0..rows as usize {
                            let q = 1 + if layout == 101 {
                                i * ldb as usize + j
                            } else {
                                i + j * ldb as usize
                            };
                            assert!(b[q] == T::z());
                            touched[q] = true;
                        }
                    }
                    assert!(b
                        .iter()
                        .enumerate()
                        .all(|(i, x)| touched[i] || *x == T::real(778.0)));
                    assert!(workspace[0] == T::real(779.0) && workspace[63] == T::real(779.0));
                }
            }
        }
    }
    #[test]
    fn empty_gels_c_layouts_preserve_padding_and_write_zero_rhs() {
        unsafe {
            empty::<f32, i32>(LAPACKE_sgels, LAPACKE_sgels_work);
            empty::<f64, i32>(LAPACKE_dgels, LAPACKE_dgels_work);
            empty::<colmat::c32, i32>(LAPACKE_cgels, LAPACKE_cgels_work);
            empty::<colmat::c64, i32>(LAPACKE_zgels, LAPACKE_zgels_work);
            empty::<f32, i64>(LAPACKE_sgels64_, LAPACKE_sgels_work64_);
            empty::<f64, i64>(LAPACKE_dgels64_, LAPACKE_dgels_work64_);
            empty::<colmat::c32, i64>(LAPACKE_cgels64_, LAPACKE_cgels_work64_);
            empty::<colmat::c64, i64>(LAPACKE_zgels64_, LAPACKE_zgels_work64_);
        }
    }
}

fn rank_geometry(layout: i32, m: i64, n: i64, nrhs: i64, lda: i64, ldb: i64) -> Result<(), i64> {
    if !matches!(layout, ROW_MAJOR | COL_MAJOR) {
        return Err(-1);
    }
    if layout == ROW_MAJOR {
        if lda < n {
            return Err(-6);
        }
        if ldb < nrhs {
            return Err(-8);
        }
    }
    if m < 0 {
        return Err(-2);
    }
    if n < 0 {
        return Err(-3);
    }
    if nrhs < 0 {
        return Err(-4);
    }
    if layout == COL_MAJOR {
        if lda < m.max(1) {
            return Err(-6);
        }
        if ldb < m.max(n).max(1) {
            return Err(-8);
        }
    }
    Ok(())
}
unsafe fn rank_nan<T: Scalar, R: Scalar>(
    layout: i32,
    m: i64,
    n: i64,
    nrhs: i64,
    a: *const T,
    lda: i64,
    b: *const T,
    ldb: i64,
    rcond: R,
) -> Result<(), i64> {
    if !matches!(layout, ROW_MAJOR | COL_MAJOR) {
        return Err(-1);
    }
    if LAPACKE_get_nancheck() != 0 {
        if ge_nancheck(layout, m, n, a, lda)? {
            return Err(-5);
        }
        if ge_nancheck(layout, m.max(n), nrhs, b, ldb)? {
            return Err(-7);
        }
        if rcond.re().is_nan() {
            return Err(-10);
        }
    }
    Ok(())
}
fn allocation_len(value: f64, maximum: usize) -> Result<usize, i64> {
    if !value.is_finite() || value < 1.0 {
        return Err(-1010);
    }
    let length = value as usize;
    if length > maximum {
        return Err(-1010);
    }
    Ok(length)
}
fn gelsy_minimum(m: i64, n: i64, nrhs: i64, complex: bool) -> i128 {
    let k = m.min(n) as i128;
    let n = n as i128;
    let nrhs = nrhs as i128;
    if !complex && (k == 0 || nrhs == 0) {
        1
    } else if complex {
        k + (2 * k).max(n + 1).max(k + nrhs)
    } else {
        (k + 3 * n + 1).max(2 * k + nrhs)
    }
}
macro_rules! gelsy_exports {($p:ident,$t:ty,$r:ty,$j:ty,[$($suffix:tt)*],$f:tt,$rw:ident,[$($rwparam:tt)*],[$($rwarg:tt)*])=>{paste::paste!{
    #[no_mangle]
    pub unsafe extern "C" fn [<LAPACKE_ $p gelsy_work $($suffix)*>](layout:i32,m:$j,n:$j,nrhs:$j,a:*mut $t,lda:$j,b:*mut $t,ldb:$j,jpvt:*mut $j,rcond:$r,rank:*mut $j,workspace:*mut $t,lwork:$j $($rwparam)*)->$j {
        if let Err(e)=rank_geometry(layout,m as i64,n as i64,nrhs as i64,lda as i64,ldb as i64){return e as $j;}
        let minimum=gelsy_minimum(m as i64,n as i64,nrhs as i64,<$t>::COMPLEX);
        if minimum>(<$j>::MAX as i128){return -13;}
        let skip=lwork== -1||(lwork as i128)<minimum||m.min(n)==0||nrhs==0;
        layout_work(layout,m as i64,n as i64,nrhs as i64,m as i64,a,lda as i64,b,ldb as i64,skip,|a,lda,b,ldb|{
            let mut info:$j=0;
            crate::least_squares::[<$p gelsy $f>](&m,&n,&nrhs,a,&(lda as $j),b,&(ldb as $j),jpvt,&rcond,rank,workspace,&lwork $($rwarg)*,&mut info);
            if info<0 {(info-1) as i64}else{info as i64}
        }) as $j
    }
    #[no_mangle]
    pub unsafe extern "C" fn [<LAPACKE_ $p gelsy $($suffix)*>](layout:i32,m:$j,n:$j,nrhs:$j,a:*mut $t,lda:$j,b:*mut $t,ldb:$j,jpvt:*mut $j,rcond:$r,rank:*mut $j)->$j {
        if let Err(e)=rank_nan(layout,m as i64,n as i64,nrhs as i64,a,lda as i64,b,ldb as i64,rcond){return e as $j;}
        let mut query=<$t>::z();let mut real_query=<$r>::z();let $rw=&mut real_query as *mut $r;
        let info=[<LAPACKE_ $p gelsy_work $($suffix)*>](layout,m,n,nrhs,a,lda,b,ldb,jpvt,rcond,rank,&mut query,-1 $($rwarg)*);
        if info!=0{return info;}
        let count=match allocation_len(query.re(),<$j>::MAX as usize){Ok(v)=>v,Err(e)=>return e as $j};
        let mut workspace=match zeros::<$t>(count){Ok(v)=>v,Err(_)=>return -1010};
        let rwcount=if <$t>::COMPLEX {
            match (n.max(0) as usize).checked_mul(2){Some(v)=>v.max(1),None=>return -1010}
        }else{0};
        let mut real_workspace=match zeros::<$r>(rwcount){Ok(v)=>v,Err(_)=>return -1010};
        let $rw=real_workspace.as_mut_ptr();
        [<LAPACKE_ $p gelsy_work $($suffix)*>](layout,m,n,nrhs,a,lda,b,ldb,jpvt,rcond,rank,workspace.as_mut_ptr(),count as $j $($rwarg)*)
    }
}};}
gelsy_exports!(s, f32, f32, i32, [], _, _rwork, [], []);
gelsy_exports!(d, f64, f64, i32, [], _, _rwork, [], []);
gelsy_exports!(c,colmat::c32,f32,i32,[],_,_rwork,[,_rwork:*mut f32],[,_rwork]);
gelsy_exports!(z,colmat::c64,f64,i32,[],_,_rwork,[,_rwork:*mut f64],[,_rwork]);
gelsy_exports!(s, f32, f32, i64, [64_], _64_, _rwork, [], []);
gelsy_exports!(d, f64, f64, i64, [64_], _64_, _rwork, [], []);
gelsy_exports!(c,colmat::c32,f32,i64,[64_],_64_,_rwork,[,_rwork:*mut f32],[,_rwork]);
gelsy_exports!(z,colmat::c64,f64,i64,[64_],_64_,_rwork,[,_rwork:*mut f64],[,_rwork]);

#[derive(Clone, Copy)]
enum SvdDriver {
    Classical,
    Divide,
}
fn svd_workspace(
    m: i64,
    n: i64,
    nrhs: i64,
    complex: bool,
    driver: SvdDriver,
) -> (i128, i128, i128) {
    let k = m.min(n) as i128;
    let m = m as i128;
    let n = n as i128;
    let r = nrhs as i128;
    if k == 0 {
        return (1, 1, 1);
    }
    if matches!(driver, SvdDriver::Classical) {
        let work = if complex {
            2 * k + m.max(n).max(r)
        } else {
            3 * k + (2 * k).max(m.max(n)).max(r)
        };
        return (work, 1, if complex { 5 * k } else { 1 });
    }
    let levels = (((k as f64 / 26.0).log2() as i128) + 1).max(0);
    let integer = (3 * k * levels + 11 * k).max(1);
    if complex {
        let real = (60 * k + 8 * k * levels + 75 * r + (26 * 26).max(n * (1 + r) + 2 * r)).max(1);
        ((2 * k + k * r).max(1), integer, real)
    } else {
        (
            (62 * k + 8 * k * levels + k * r + 26 * 26).max(1),
            integer,
            1,
        )
    }
}
fn integer_workspace<J: crate::lapack::LapackInt>(count: usize) -> Result<Vec<J>, i64> {
    if count
        .checked_mul(core::mem::size_of::<J>())
        .is_none_or(|n| n > isize::MAX as usize)
    {
        return Err(-1010);
    }
    let mut values = Vec::new();
    values.try_reserve_exact(count).map_err(|_| -1010)?;
    values.resize(count, J::from_i64(0));
    Ok(values)
}
macro_rules! svd_exports {($p:ident,$family:ident,$divide:expr,$t:ty,$r:ty,$j:ty,[$($suffix:tt)*],$f:tt,$rw:ident,$iw:ident,[$($rwparam:tt)*],[$($rwarg:tt)*],[$($iwparam:tt)*],[$($iwarg:tt)*])=>{paste::paste!{
    #[no_mangle]
    pub unsafe extern "C" fn [<LAPACKE_ $p $family _work $($suffix)*>](layout:i32,m:$j,n:$j,nrhs:$j,a:*mut $t,lda:$j,b:*mut $t,ldb:$j,s:*mut $r,rcond:$r,rank:*mut $j,workspace:*mut $t,lwork:$j $($rwparam)* $($iwparam)*)->$j {
        if let Err(e)=rank_geometry(layout,m as i64,n as i64,nrhs as i64,lda as i64,ldb as i64){return e as $j;}
        let driver=if $divide {SvdDriver::Divide}else{SvdDriver::Classical};
        let (minimum,integer,real)=svd_workspace(m as i64,n as i64,nrhs as i64,<$t>::COMPLEX,driver);
        if [minimum,integer,real].iter().any(|n|*n>(<$j>::MAX as i128)){return -13;}
        layout_work(layout,m as i64,n as i64,nrhs as i64,m as i64,a,lda as i64,b,ldb as i64,lwork== -1||(lwork as i128)<minimum,|a,lda,b,ldb|{
            let mut info:$j=0;
            crate::least_squares::[<$p $family $f>](&m,&n,&nrhs,a,&(lda as $j),b,&(ldb as $j),s,&rcond,rank,workspace,&lwork $($rwarg)* $($iwarg)*,&mut info);
            if info<0 {(info-1) as i64}else{info as i64}
        }) as $j
    }
    #[no_mangle]
    pub unsafe extern "C" fn [<LAPACKE_ $p $family $($suffix)*>](layout:i32,m:$j,n:$j,nrhs:$j,a:*mut $t,lda:$j,b:*mut $t,ldb:$j,s:*mut $r,rcond:$r,rank:*mut $j)->$j {
        if let Err(e)=rank_nan(layout,m as i64,n as i64,nrhs as i64,a,lda as i64,b,ldb as i64,rcond){return e as $j;}
        let mut query=<$t>::z();let mut real_query=<$r>::z();let mut integer_query:$j=0;
        let $rw=&mut real_query as *mut $r;let $iw=&mut integer_query as *mut $j;
        let info=[<LAPACKE_ $p $family _work $($suffix)*>](layout,m,n,nrhs,a,lda,b,ldb,s,rcond,rank,&mut query,-1 $($rwarg)* $($iwarg)*);
        if info!=0{return info;}
        let count=match allocation_len(query.re(),<$j>::MAX as usize){Ok(v)=>v,Err(e)=>return e as $j};
        let mut workspace=match zeros::<$t>(count){Ok(v)=>v,Err(_)=>return -1010};
        let rwcount=if <$t>::COMPLEX {
            if $divide {match allocation_len(real_query.re(),<$j>::MAX as usize){Ok(v)=>v,Err(e)=>return e as $j}}
            else {match (m.min(n).max(0) as usize).checked_mul(5){Some(v)=>v.max(1),None=>return -1010}}
        }else{0};
        let iwcount=if $divide {if integer_query<1{return -1010;}integer_query as usize}else{0};
        let mut real_workspace=match zeros::<$r>(rwcount){Ok(v)=>v,Err(_)=>return -1010};
        let mut integer_workspace=match integer_workspace::<$j>(iwcount){Ok(v)=>v,Err(e)=>return e as $j};
        let $rw=real_workspace.as_mut_ptr();let $iw=integer_workspace.as_mut_ptr();
        [<LAPACKE_ $p $family _work $($suffix)*>](layout,m,n,nrhs,a,lda,b,ldb,s,rcond,rank,workspace.as_mut_ptr(),count as $j $($rwarg)* $($iwarg)*)
    }
}};}
svd_exports!(
    s,
    gelss,
    false,
    f32,
    f32,
    i32,
    [],
    _,
    _rwork,
    _iwork,
    [],
    [],
    [],
    []
);
svd_exports!(
    d,
    gelss,
    false,
    f64,
    f64,
    i32,
    [],
    _,
    _rwork,
    _iwork,
    [],
    [],
    [],
    []
);
svd_exports!(c,gelss,false,colmat::c32,f32,i32,[],_,_rwork,_iwork,[,_rwork:*mut f32],[,_rwork],[],[]);
svd_exports!(z,gelss,false,colmat::c64,f64,i32,[],_,_rwork,_iwork,[,_rwork:*mut f64],[,_rwork],[],[]);
svd_exports!(
    s,
    gelss,
    false,
    f32,
    f32,
    i64,
    [64_],
    _64_,
    _rwork,
    _iwork,
    [],
    [],
    [],
    []
);
svd_exports!(
    d,
    gelss,
    false,
    f64,
    f64,
    i64,
    [64_],
    _64_,
    _rwork,
    _iwork,
    [],
    [],
    [],
    []
);
svd_exports!(c,gelss,false,colmat::c32,f32,i64,[64_],_64_,_rwork,_iwork,[,_rwork:*mut f32],[,_rwork],[],[]);
svd_exports!(z,gelss,false,colmat::c64,f64,i64,[64_],_64_,_rwork,_iwork,[,_rwork:*mut f64],[,_rwork],[],[]);
svd_exports!(s,gelsd,true,f32,f32,i32,[],_,_rwork,_iwork,[],[],[,_iwork:*mut i32],[,_iwork]);
svd_exports!(d,gelsd,true,f64,f64,i32,[],_,_rwork,_iwork,[],[],[,_iwork:*mut i32],[,_iwork]);
svd_exports!(c,gelsd,true,colmat::c32,f32,i32,[],_,_rwork,_iwork,[,_rwork:*mut f32],[,_rwork],[,_iwork:*mut i32],[,_iwork]);
svd_exports!(z,gelsd,true,colmat::c64,f64,i32,[],_,_rwork,_iwork,[,_rwork:*mut f64],[,_rwork],[,_iwork:*mut i32],[,_iwork]);
svd_exports!(s,gelsd,true,f32,f32,i64,[64_],_64_,_rwork,_iwork,[],[],[,_iwork:*mut i64],[,_iwork]);
svd_exports!(d,gelsd,true,f64,f64,i64,[64_],_64_,_rwork,_iwork,[],[],[,_iwork:*mut i64],[,_iwork]);
svd_exports!(c,gelsd,true,colmat::c32,f32,i64,[64_],_64_,_rwork,_iwork,[,_rwork:*mut f32],[,_rwork],[,_iwork:*mut i64],[,_iwork]);
svd_exports!(z,gelsd,true,colmat::c64,f64,i64,[64_],_64_,_rwork,_iwork,[,_rwork:*mut f64],[,_rwork],[,_iwork:*mut i64],[,_iwork]);
