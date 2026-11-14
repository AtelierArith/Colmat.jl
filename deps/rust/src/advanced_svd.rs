//! Selected and QR-preconditioned SVD adapters over the public native SVD.
use crate::kernels::{flag, Scalar};
use colmat::Mat;
use std::ffi::c_void;
use std::sync::atomic::{AtomicPtr, Ordering};

static FALLBACKS: [AtomicPtr<c_void>; 8] = [const { AtomicPtr::new(std::ptr::null_mut()) }; 8];
#[no_mangle]
pub extern "C" fn colmat_advanced_svd_set_fallback(id: i32, pointer: *mut c_void) {
    if let Some(slot) = FALLBACKS.get(id as usize) {
        slot.store(pointer, Ordering::Release);
    }
}
#[no_mangle]
pub extern "C" fn colmat_advanced_svd_has_fallback(id: i32) -> i32 {
    FALLBACKS
        .get(id as usize)
        .is_some_and(|p| !p.load(Ordering::Acquire).is_null()) as i32
}
fn fallback(id: usize) -> *mut c_void {
    let pointer = FALLBACKS[id].load(Ordering::Acquire);
    assert!(
        !pointer.is_null(),
        "Colmat advanced SVD fallback must be configured before use"
    );
    pointer
}
enum Selection {
    All,
    Value(f64, f64),
    Index(usize, usize),
}
unsafe fn selection<R: Scalar, I: crate::lapack::LapackInt>(
    mode: u8,
    k: isize,
    vl: *const R,
    vu: *const R,
    il: *const I,
    iu: *const I,
) -> Result<Selection, i32> {
    if k == 0 {
        return Ok(Selection::All);
    }
    match mode {
        b'A' => Ok(Selection::All),
        b'V' => {
            let (lo, hi) = ((*vl).re(), (*vu).re());
            if lo < 0.0 {
                Err(8)
            } else if hi <= lo {
                Err(9)
            } else {
                Ok(Selection::Value(lo, hi))
            }
        }
        b'I' => {
            let (lo, hi) = ((*il).val(), (*iu).val());
            if lo < 1 || lo > k.max(1) {
                Err(10)
            } else if hi < lo.min(k) || hi > k {
                Err(11)
            } else {
                Ok(Selection::Index((lo - 1) as usize, hi as usize))
            }
        }
        _ => Err(3),
    }
}
fn x_workspace(m: isize, n: isize, complex: bool) -> isize {
    let k = m.min(n);
    if k <= 0 {
        return 1;
    }
    // The conventional ILAENV(6) crossover is floor(1.6*min(M,N)).
    let tall = (m.max(n) as u128) >= k as u128 * 8 / 5;
    if complex {
        if tall {
            k.saturating_mul(k.saturating_add(5))
        } else {
            k.saturating_mul(3).saturating_add(m.max(n))
        }
    } else if tall {
        k.saturating_mul(k.saturating_mul(3).saturating_add(20))
    } else {
        k.saturating_mul(k.saturating_mul(2).saturating_add(19))
            .max(k.saturating_mul(4).saturating_add(m.max(n)))
    }
}
unsafe fn selected_svd<T: Scalar>(
    m: usize,
    n: usize,
    a: *const T,
    ld: isize,
    left: bool,
    right: bool,
) -> Result<(Mat<T>, Vec<f64>, Mat<T>), ()> {
    let matrix = crate::lapack::input_ref(a, m, n, ld);
    if !matrix.is_all_finite() {
        return Err(());
    }
    let result = crate::lapack::svd(
        matrix,
        if left { m.min(n) } else { 0 },
        if right { m.min(n) } else { 0 },
    )?;
    if !result.0.as_ref().is_all_finite()
        || !result.1.iter().all(|x| x.is_finite())
        || !result.2.as_ref().is_all_finite()
    {
        return Err(());
    }
    Ok(result)
}
unsafe fn x_output<T: Scalar, R: Scalar, I: crate::lapack::LapackInt>(
    result: (Mat<T>, Vec<f64>, Mat<T>),
    selected: Selection,
    left: bool,
    right: bool,
    ns: *mut I,
    s: *mut R,
    u: *mut T,
    lu: isize,
    vt: *mut T,
    lv: isize,
    iwork: *mut I,
) {
    let (ul, values, vr) = result;
    let indices: Vec<usize> = (0..values.len())
        .filter(|&i| match selected {
            Selection::All => true,
            Selection::Value(lo, hi) => lo < values[i] && values[i] <= hi,
            Selection::Index(lo, hi) => lo <= i && i < hi,
        })
        .collect();
    *ns = I::from_i64(indices.len() as i64);
    for (j, &source) in indices.iter().enumerate() {
        *s.add(j) = R::real(values[source]);
        *iwork.add(j) = I::from_i64(0);
        if left {
            for i in 0..ul.nrows() {
                *u.offset(i as isize + j as isize * lu) = ul[(i, source)];
            }
        }
        if right {
            for i in 0..vr.nrows() {
                *vt.offset(j as isize + i as isize * lv) = vr[(i, source)].cj();
            }
        }
    }
}
use paste::paste;
macro_rules! x_real {
    ($p:ident,$r:ty,$i:ty,$s:tt,$id:expr)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p gesvdx $s>](jobu:*const u8,jobvt:*const u8,range:*const u8,m:*const $i,n:*const $i,a:*mut $r,lda:*const $i,vl:*const $r,vu:*const $r,il:*const $i,iu:*const $i,ns:*mut $i,s:*mut $r,u:*mut $r,ldu:*const $i,vt:*mut $r,ldvt:*const $i,work:*mut $r,lwork:*const $i,iwork:*mut $i,info:*mut $i) {
            *ns=0;*info=0;
            let(left,right,mode,mm,nn,la,lu,lv,lw)=(flag(jobu),flag(jobvt),flag(range),*m as isize,*n as isize,*lda as isize,*ldu as isize,*ldvt as isize,*lwork as isize);
            for(arg,valid)in[(1,matches!(left,b'N'|b'V')),(2,matches!(right,b'N'|b'V')),(3,matches!(mode,b'A'|b'I'|b'V')),(4,mm>=0),(5,nn>=0),(7,la>=mm.max(1))]{if !valid{*info=-arg;crate::invalid(arg as i32);return;}}
            let k=mm.min(nn);let selected=match selection(mode,k,vl,vu,il,iu){Ok(s)=>s,Err(arg)=>{*info=-arg as $i;crate::invalid(arg);return;}};
            if k>0 {
                let rows=match selected{Selection::Index(lo,hi)=>(hi-lo)as isize,_=>k};
                for(arg,valid)in[(15,left==b'N'||lu>=mm),(17,right==b'N'||lv>=rows)]{if !valid{*info=-arg;crate::invalid(arg as i32);return;}}
            }
            let minimum=x_workspace(mm,nn,false);*work=crate::lapack::workspace::<$r>(minimum);
            if lw!= -1&&lw<minimum{*info=-19;crate::invalid(19);return;}
            if lw== -1||k==0{return;}
            match selected_svd(mm as usize,nn as usize,a,la,left==b'V',right==b'V') {
                Ok(result)=>x_output(result,selected,left==b'V',right==b'V',ns,s,u,lu,vt,lv,iwork),
                Err(())=>{
                    let f:unsafe extern "C" fn(*const u8,*const u8,*const u8,*const i64,*const i64,*mut $r,*const i64,*const $r,*const $r,*const i64,*const i64,*mut i64,*mut $r,*mut $r,*const i64,*mut $r,*const i64,*mut $r,*const i64,*mut i64,*mut i64)=std::mem::transmute(fallback($id));
                    let(mut count,mut status)=(0i64,0i64);let mut scratch=vec![0i64;12*k as usize];
                    let(lo,hi)=if mode==b'I'{(*il as i64,*iu as i64)}else{(1,k as i64)};
                    f(jobu,jobvt,range,&(mm as i64),&(nn as i64),a,&(la as i64),vl,vu,&lo,&hi,&mut count,s,u,&(lu as i64),vt,&(lv as i64),work,&(lw as i64),scratch.as_mut_ptr(),&mut status);
                    *ns=count as $i;*info=status as $i;for j in 0..scratch.len(){*iwork.add(j)=scratch[j]as $i;}
                }
            }
        }
    }};
}
macro_rules! x_complex {
    ($p:ident,$t:ty,$r:ty,$i:ty,$s:tt,$id:expr)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p gesvdx $s>](jobu:*const u8,jobvt:*const u8,range:*const u8,m:*const $i,n:*const $i,a:*mut $t,lda:*const $i,vl:*const $r,vu:*const $r,il:*const $i,iu:*const $i,ns:*mut $i,s:*mut $r,u:*mut $t,ldu:*const $i,vt:*mut $t,ldvt:*const $i,work:*mut $t,lwork:*const $i,rwork:*mut $r,iwork:*mut $i,info:*mut $i) {
            *ns=0;*info=0;
            let(left,right,mode,mm,nn,la,lu,lv,lw)=(flag(jobu),flag(jobvt),flag(range),*m as isize,*n as isize,*lda as isize,*ldu as isize,*ldvt as isize,*lwork as isize);
            for(arg,valid)in[(1,matches!(left,b'N'|b'V')),(2,matches!(right,b'N'|b'V')),(3,matches!(mode,b'A'|b'I'|b'V')),(4,mm>=0),(5,nn>=0),(7,la>=mm.max(1))]{if !valid{*info=-arg;crate::invalid(arg as i32);return;}}
            let k=mm.min(nn);let selected=match selection(mode,k,vl,vu,il,iu){Ok(s)=>s,Err(arg)=>{*info=-arg as $i;crate::invalid(arg);return;}};
            if k>0 {
                let rows=match selected{Selection::Index(lo,hi)=>(hi-lo)as isize,_=>k};
                for(arg,valid)in[(15,left==b'N'||lu>=mm),(17,right==b'N'||lv>=rows)]{if !valid{*info=-arg;crate::invalid(arg as i32);return;}}
            }
            let minimum=x_workspace(mm,nn,true);*work=crate::lapack::workspace::<$t>(minimum);
            if lw!= -1&&lw<minimum{*info=-19;crate::invalid(19);return;}
            if lw== -1||k==0{return;}
            match selected_svd(mm as usize,nn as usize,a,la,left==b'V',right==b'V') {
                Ok(result)=>x_output(result,selected,left==b'V',right==b'V',ns,s,u,lu,vt,lv,iwork),
                Err(())=>{
                    let f:unsafe extern "C" fn(*const u8,*const u8,*const u8,*const i64,*const i64,*mut $t,*const i64,*const $r,*const $r,*const i64,*const i64,*mut i64,*mut $r,*mut $t,*const i64,*mut $t,*const i64,*mut $t,*const i64,*mut $r,*mut i64,*mut i64)=std::mem::transmute(fallback($id));
                    let(mut count,mut status)=(0i64,0i64);let mut scratch=vec![0i64;12*k as usize];
                    let(lo,hi)=if mode==b'I'{(*il as i64,*iu as i64)}else{(1,k as i64)};
                    f(jobu,jobvt,range,&(mm as i64),&(nn as i64),a,&(la as i64),vl,vu,&lo,&hi,&mut count,s,u,&(lu as i64),vt,&(lv as i64),work,&(lw as i64),rwork,scratch.as_mut_ptr(),&mut status);
                    *ns=count as $i;*info=status as $i;for j in 0..scratch.len(){*iwork.add(j)=scratch[j]as $i;}
                }
            }
        }
    }};
}
x_real!(s, f32, i32, _, 0);
x_real!(d, f64, i32, _, 1);
x_complex!(c, colmat::c32, f32, i32, _, 2);
x_complex!(z, colmat::c64, f64, i32, _, 3);
x_real!(s, f32, i64, _64_, 0);
x_real!(d, f64, i64, _64_, 1);
x_complex!(c, colmat::c32, f32, i64, _64_, 2);
x_complex!(z, colmat::c64, f64, i64, _64_, 3);

// QR is staged because a failed native SVD must leave the original input
// available to the saved expert driver. The committed factor and tau retain
// the LAPACK Householder representation, including in JOBU='F' mode.
struct PreconditionedSvd<T: Scalar> {
    qr: Vec<T>,
    tau: Vec<T>,
    permutation: Vec<usize>,
    row_swaps: Vec<usize>,
    values: Vec<f64>,
    left: Mat<T>,
    right_adjoint: Mat<T>,
    rank: usize,
    zero_count: usize,
}
unsafe fn preconditioned_svd<T: crate::householder::ReflectorScalar>(
    m: usize,
    n: usize,
    a: *const T,
    lda: isize,
    accuracy: u8,
    row_pivot: bool,
    transpose: bool,
    jobu: u8,
    jobv: u8,
) -> Result<PreconditionedSvd<T>, ()> {
    let mut qr = vec![T::z(); m * n];
    let mut maximum = 0.0_f64;
    for j in 0..n {
        for i in 0..m {
            let value = *a.offset(i as isize + j as isize * lda);
            if !value.re().is_finite() || !value.im().is_finite() {
                return Err(());
            }
            maximum = maximum.max(value.re().abs()).max(value.im().abs());
            qr[i + j * m] = value;
        }
    }
    let mut row_swaps = Vec::new();
    if row_pivot {
        let mut norms: Vec<f64> = (0..m)
            .map(|i| {
                (0..n)
                    .map(|j| qr[i + j * m].re().hypot(qr[i + j * m].im()))
                    .fold(0.0, f64::max)
            })
            .collect();
        for i in 0..m.saturating_sub(1) {
            let mut selected = i;
            for p in i + 1..m {
                if norms[p] > norms[selected] {
                    selected = p;
                }
            }
            norms.swap(i, selected);
            for j in 0..n {
                qr.swap(i + j * m, selected + j * m);
            }
            row_swaps.push(selected);
        }
    }
    // Scale only near the overflow boundary to protect the QR column norms.
    let limit = if T::EPS > 1e-10 {
        f32::MAX as f64
    } else {
        f64::MAX
    };
    let scale = if maximum > limit / (m.max(1) as f64).sqrt() / 2.0 {
        0.5 / (m.max(1) as f64).sqrt()
    } else {
        1.0
    };
    if scale != 1.0 {
        for x in &mut qr {
            *x = *x * T::real(scale);
        }
    }
    let mut tau = vec![T::z(); n];
    let factor = crate::qr_pivoted::pivoted_qr(
        m,
        n,
        0,
        qr.as_mut_ptr(),
        m as isize,
        tau.as_mut_ptr(),
        &vec![false; n],
        n,
        -1.0,
        -1.0,
    );
    if factor.exception != 0 {
        return Err(());
    }
    let magnitude = |i: usize| qr[i + i * m].re().hypot(qr[i + i * m].im());
    let mut rank = factor.rank;
    for i in 1..rank {
        let cutoff = match accuracy {
            b'A' => (n as f64).sqrt() * (T::EPS / 2.0) * magnitude(0),
            b'M' => ((T::EPS / 2.0) * magnitude(i - 1)).max(T::SAFE * T::EPS / 2.0),
            _ => 0.0,
        };
        if magnitude(i) < cutoff || magnitude(i) == 0.0 {
            rank = i;
            break;
        }
    }
    let left_requested = jobu != b'N';
    let right_requested = jobv != b'N';
    let (ur, mut values, vr) = if rank == 0 {
        (Mat::<T>::zeros(0, 0), Vec::new(), Mat::<T>::identity(n, n))
    } else if transpose {
        let rt = Mat::<T>::from_fn(
            n,
            rank,
            |i, j| if j <= i { qr[j + i * m].cj() } else { T::z() },
        );
        let (v, s, u) = crate::lapack::svd(
            rt.as_ref(),
            if right_requested { n } else { 0 },
            if left_requested { rank } else { 0 },
        )?;
        (u, s, v)
    } else {
        let r = Mat::<T>::from_fn(rank, n, |i, j| if i <= j { qr[i + j * m] } else { T::z() });
        crate::lapack::svd(
            r.as_ref(),
            if left_requested { rank } else { 0 },
            if right_requested { n } else { 0 },
        )?
    };
    if !ur.as_ref().is_all_finite()
        || !vr.as_ref().is_all_finite()
        || !values.iter().all(|v| v.is_finite())
    {
        return Err(());
    }
    let qr_rank = rank;
    while rank > 0 && values[rank - 1] <= 0.0 {
        rank -= 1;
    }
    for value in &mut values {
        *value /= scale;
    }
    values.resize(n, 0.0);
    let columns = match jobu {
        b'N' => 0,
        b'R' => rank,
        b'A' => m,
        _ => n,
    };
    let rows = if jobu == b'F' { n } else { m };
    let mut left = Mat::<T>::from_fn(rows, columns, |i, j| {
        if i < qr_rank && j < qr_rank {
            ur[(i, j)]
        } else if i == j {
            T::real(1.0)
        } else {
            T::z()
        }
    });
    if left_requested && jobu != b'F' {
        let mut contiguous = vec![T::z(); m * columns];
        for j in 0..columns {
            for i in 0..m {
                contiguous[i + j * m] = left[(i, j)];
            }
        }
        crate::qr::apply_q(
            b'L',
            b'N',
            m,
            columns,
            n,
            qr.as_ptr(),
            m as isize,
            tau.as_ptr(),
            contiguous.as_mut_ptr(),
            m as isize,
        );
        for (i, &p) in row_swaps.iter().enumerate().rev() {
            for j in 0..columns {
                contiguous.swap(i + j * m, p + j * m);
            }
        }
        left = Mat::<T>::from_fn(m, columns, |i, j| contiguous[i + j * m]);
    }
    let right_rows = if jobv == b'N' {
        0
    } else if jobv == b'R' {
        rank
    } else {
        n
    };
    let mut right_adjoint = Mat::<T>::zeros(right_rows, n);
    for j in 0..n {
        for i in 0..right_rows {
            right_adjoint[(i, factor.permutation[j])] = vr[(j, i)].cj();
        }
    }
    Ok(PreconditionedSvd {
        qr,
        tau,
        permutation: factor.permutation,
        row_swaps,
        values,
        left,
        right_adjoint,
        rank,
        zero_count: qr_rank - rank,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::householder::ReflectorScalar;
    unsafe fn check<T: ReflectorScalar>(tolerance: f64) {
        for (m, n) in [(3, 3), (7, 3)] {
            let original: Vec<T> = (0..n)
                .flat_map(|j| {
                    (0..m).map(move |i| T::parts((i + 2 * j + 1) as f64, (i * j) as f64 / 3.0))
                })
                .collect();
            for accuracy in [b'A', b'M', b'H'] {
                for row_pivot in [false, true] {
                    for transpose in [false, true] {
                        for jobu in [b'A', b'S', b'R', b'F'] {
                            let result = preconditioned_svd(
                                m,
                                n,
                                original.as_ptr(),
                                m as isize,
                                accuracy,
                                row_pivot,
                                transpose,
                                jobu,
                                b'R',
                            )
                            .unwrap();
                            let mut left = result.left;
                            if jobu == b'F' {
                                let mut buffer = vec![T::z(); m * n];
                                for j in 0..n {
                                    for i in 0..n {
                                        buffer[i + j * m] = left[(i, j)];
                                    }
                                }
                                crate::qr::apply_q(
                                    b'L',
                                    b'N',
                                    m,
                                    n,
                                    n,
                                    result.qr.as_ptr(),
                                    m as isize,
                                    result.tau.as_ptr(),
                                    buffer.as_mut_ptr(),
                                    m as isize,
                                );
                                for (i, &p) in result.row_swaps.iter().enumerate().rev() {
                                    for j in 0..n {
                                        buffer.swap(i + j * m, p + j * m);
                                    }
                                }
                                left = Mat::<T>::from_fn(m, n, |i, j| buffer[i + j * m]);
                            }
                            for j in 0..n {
                                for i in 0..m {
                                    let mut reconstructed = T::z();
                                    for k in 0..result.rank {
                                        reconstructed = reconstructed
                                            + left[(i, k)]
                                                * T::real(result.values[k])
                                                * result.right_adjoint[(k, j)];
                                    }
                                    let error = reconstructed - original[i + j * m];
                                    assert!(error.re().hypot(error.im())<tolerance,
                                    "reconstruction failed for {}x{}, jobu={}, transpose={}, row_pivot={}",
                                    m,n,jobu as char,transpose,row_pivot);
                                }
                            }
                            assert_eq!(result.permutation.len(), n);
                            assert!(result.zero_count <= n);
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn qr_preconditioned_svd_real_and_complex() {
        unsafe {
            check::<f32>(2e-4);
            check::<f64>(1e-11);
            check::<colmat::c32>(2e-4);
            check::<colmat::c64>(1e-11);
        }
    }
}

fn q_workspace(
    m: isize,
    n: isize,
    complex: bool,
    row: bool,
    estimate: bool,
    jobu: u8,
    jobv: u8,
) -> (isize, isize, isize) {
    let add = |a: isize, b: isize| a.saturating_add(b);
    let mul = |a: isize, b: isize| a.saturating_mul(b);
    let integers = add(
        add(n, if row { m.saturating_sub(1) } else { 0 }),
        if estimate && !complex { n } else { 0 },
    )
    .max(1);
    let real = if complex { mul(5, n) } else { 2 }.max(if row { m } else { 2 });
    let qr = add(mul(if complex { 1 } else { 3 }, n), 1);
    let svd = mul(if complex { 3 } else { 5 }, n).max(1);
    let apply = match jobu {
        b'A' => m.max(1),
        b'S' | b'U' | b'R' => n.max(1),
        _ => 0,
    };
    let condition = if estimate {
        mul(if complex { 2 } else { 3 }, n)
    } else {
        0
    };
    let work = if jobu == b'N' && jobv == b'N' {
        add(n, qr).max(svd).max(condition)
    } else {
        add(n, qr.max(svd).max(apply).max(condition))
    }
    .max(2);
    (integers, work, real)
}
macro_rules! q_driver {
    ($p:ident,$t:ty,$r:ty,$i:ty,$s:tt,$id:expr,$complex:expr)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p gesvdq $s>](joba:*const u8,jobp:*const u8,
            jobr:*const u8,jobu:*const u8,jobv:*const u8,m:*const $i,n:*const $i,
            a:*mut $t,lda:*const $i,s:*mut $r,u:*mut $t,ldu:*const $i,
            v:*mut $t,ldv:*const $i,numrank:*mut $i,iwork:*mut $i,liwork:*const $i,
            work:*mut $t,lwork:*const $i,rwork:*mut $r,lrwork:*const $i,info:*mut $i) {
            let(acc,row,tr,ju,jv)=(flag(joba),flag(jobp),flag(jobr),flag(jobu),flag(jobv));
            let(mm,nn,la,lu,lv,li,lw,lr)=(*m as isize,*n as isize,*lda as isize,
                *ldu as isize,*ldv as isize,*liwork as isize,*lwork as isize,*lrwork as isize);
            let query=li== -1 || lw== -1 || lr== -1;
            *info=0;
            for(arg,valid) in [(1,matches!(acc,b'A'|b'M'|b'H'|b'E')),
                (2,matches!(row,b'P'|b'N')),(3,matches!(tr,b'T'|b'N')),
                (4,matches!(ju,b'A'|b'S'|b'U'|b'R'|b'F'|b'N')),
                (5,matches!(jv,b'A'|b'V'|b'R'|b'N') && !(ju==b'R' && matches!(jv,b'A'|b'V'))),
                (6,mm>=0),(7,nn>=0 && nn<=mm),(9,la>=mm.max(1)),
                (12,lu>=1 && (ju==b'N'||lu>=if ju==b'F'{nn}else{mm})),
                (14,lv>=1 && ((jv==b'N' && acc!=b'E')||lv>=nn))] {
                if !valid {*info=-arg;crate::invalid(arg as i32);return;}
            }
            let(imin,wmin,rmin)=q_workspace(mm,nn,$complex,row==b'P',acc==b'E',ju,jv);
            for(arg,valid) in [(17,query||li>=imin),(19,query||lw>=wmin),(21,query||lr>=rmin)] {
                if !valid {*info=-arg;crate::invalid(arg as i32);return;}
            }
            if query {
                *iwork=imin as $i;*work=crate::lapack::workspace::<$t>(wmin);
                *work.add(1)=crate::lapack::workspace::<$t>(wmin);*rwork=rmin as $r;return;
            }
            if mm==0||nn==0{return;}
            let result=if acc==b'E' {Err(())} else {
                preconditioned_svd(mm as usize,nn as usize,a,la,acc,row==b'P',tr==b'T',ju,jv)
            };
            match result {
                Ok(result)=>{
                    *numrank=result.rank as $i;
                    for j in 0..nn as usize {
                        *s.add(j)=result.values[j] as $r;
                        *iwork.add(j)=(result.permutation[j]+1) as $i;
                        *work.add(j)=result.tau[j];
                        for i in 0..mm as usize {*a.offset(i as isize+j as isize*la)=result.qr[i+j*mm as usize];}
                    }
                    for (i,p) in result.row_swaps.iter().enumerate() {*iwork.add(nn as usize+i)=(p+1) as $i;}
                    for j in 0..result.left.ncols() {for i in 0..result.left.nrows() {
                        *u.offset(i as isize+j as isize*lu)=result.left[(i,j)];
                    }}
                    for j in 0..result.right_adjoint.ncols() {for i in 0..result.right_adjoint.nrows() {
                        *v.offset(i as isize+j as isize*lv)=result.right_adjoint[(i,j)];
                    }}
                    *rwork= -1.0;
                    *rwork.add(1)=if result.values.iter().all(|&s|s==0.0) {-1.0} else {result.zero_count as $r};
                }
                Err(())=>{
                    let f:unsafe extern "C" fn(*const u8,*const u8,*const u8,*const u8,*const u8,
                        *const i64,*const i64,*mut $t,*const i64,*mut $r,*mut $t,*const i64,
                        *mut $t,*const i64,*mut i64,*mut i64,*const i64,*mut $t,*const i64,
                        *mut $r,*const i64,*mut i64)=std::mem::transmute(fallback($id));
                    let mut integers=vec![0i64;li as usize];
                    let(mut rank,mut status)=(0i64,0i64);
                    f(joba,jobp,jobr,jobu,jobv,&(mm as i64),&(nn as i64),a,&(la as i64),s,
                        u,&(lu as i64),v,&(lv as i64),&mut rank,integers.as_mut_ptr(),&(li as i64),
                        work,&(lw as i64),rwork,&(lr as i64),&mut status);
                    *numrank=rank as $i;*info=status as $i;
                    for j in 0..integers.len() {*iwork.add(j)=integers[j] as $i;}
                }
            }
        }
    }};
}
q_driver!(s, f32, f32, i32, _, 4, false);
q_driver!(d, f64, f64, i32, _, 5, false);
q_driver!(c, colmat::c32, f32, i32, _, 6, true);
q_driver!(z, colmat::c64, f64, i32, _, 7, true);
q_driver!(s, f32, f32, i64, _64_, 4, false);
q_driver!(d, f64, f64, i64, _64_, 5, false);
q_driver!(c, colmat::c32, f32, i64, _64_, 6, true);
q_driver!(z, colmat::c64, f64, i64, _64_, 7, true);
