//! Fortran integer ABIs for generalized Schur block movement.
use crate::kernels::{matrix_addressable, Scalar};
use crate::lapack::LapackInt;
use crate::qz_reorder::ReorderScalar;
use paste::paste;
unsafe fn invalid<I: LapackInt>(info: *mut I, arg: i64) {
    *info = I::from_i64(-arg);
    crate::invalid(arg as i32);
}
unsafe fn first<T: Scalar>(n: isize, index: isize, a: *const T, ld: isize) -> Option<isize> {
    if !T::COMPLEX && index > 0 {
        if !matrix_addressable::<T>(index + 1, index, ld) {
            return None;
        }
        if *a.offset(index + (index - 1) * ld) != T::z() {
            return Some(index - 1);
        }
    }
    debug_assert!(index < n);
    Some(index)
}
unsafe fn exchange<T: ReorderScalar, I: LapackInt>(
    wantq: bool,
    wantz: bool,
    n: isize,
    a: *mut T,
    lda: isize,
    b: *mut T,
    ldb: isize,
    q: *mut T,
    ldq: isize,
    z: *mut T,
    ldz: isize,
    ifst: *mut I,
    ilst: *mut I,
    work: Option<(*mut T, isize)>,
    info: *mut I,
) {
    *info = I::from_i64(0);
    let from = (*ifst).val();
    let to = (*ilst).val();
    for (arg, ok) in [
        (3, n >= 0),
        (5, lda >= n.max(1)),
        (7, ldb >= n.max(1)),
        (9, ldq >= if wantq { n.max(1) } else { 1 }),
        (11, ldz >= if wantz { n.max(1) } else { 1 }),
        (12, from >= 1 && from <= n),
        (13, to >= 1 && to <= n),
    ] {
        if !ok {
            invalid(info, arg);
            return;
        }
    }
    if let Some((pointer, length)) = work {
        let Some(need) = (if n <= 1 {
            Some(1)
        } else {
            n.checked_mul(4).and_then(|x| x.checked_add(16))
        }) else {
            invalid(info, 15);
            return;
        };
        *pointer = T::real(need as f64);
        if length == -1 {
            return;
        }
        if length < need {
            invalid(info, 15);
            return;
        }
    }
    if n <= 1 {
        return;
    }
    let (from, to) = (from - 1, to - 1);
    if T::COMPLEX && from == to {
        return;
    }
    let Some(start) = first(n, from, a, lda) else {
        invalid(info, 5);
        return;
    };
    let Some(target) = first(n, to, a, lda) else {
        invalid(info, 5);
        return;
    };
    if start == target {
        *ifst = I::from_i64(start as i64 + 1);
        *ilst = I::from_i64(target as i64 + 1);
        return;
    }
    for (arg, ok) in [
        (5, matrix_addressable::<T>(n, n, lda)),
        (7, matrix_addressable::<T>(n, n, ldb)),
        (9, !wantq || matrix_addressable::<T>(n, n, ldq)),
        (11, !wantz || matrix_addressable::<T>(n, n, ldz)),
    ] {
        if !ok {
            invalid(info, arg);
            return;
        }
    }
    let result = crate::qz_reorder::move_pair(
        n as usize,
        from as usize,
        to as usize,
        a,
        lda,
        b,
        ldb,
        wantq.then_some((q, ldq)),
        wantz.then_some((z, ldz)),
    );
    *ifst = I::from_i64(result.first as i64 + 1);
    // ZTGEXC's upward failure reports the attempted adjacent-swap start;
    // real TGEXC reports the moving block's current first row instead.
    let position = if T::COMPLEX && result.rejected && from > to {
        result.position as i64
    } else {
        result.position as i64 + 1
    };
    *ilst = I::from_i64(position);
    if result.rejected {
        *info = I::from_i64(1);
    }
}
/// Reorder-only TGSEN implementation. Returns false only when homogeneous
/// extraction needs the saved LAPACK implementation; staged changes have then
/// not been committed to the caller's factors.
unsafe fn selected<T: ReorderScalar, I: LapackInt>(
    wantq: bool,
    wantz: bool,
    select: *const I,
    n: isize,
    a: *mut T,
    lda: isize,
    b: *mut T,
    ldb: isize,
    ar: *mut T,
    ai: *mut T,
    beta: *mut T,
    q: *mut T,
    ldq: isize,
    z: *mut T,
    ldz: isize,
    m: *mut I,
    work: *mut T,
    lwork: isize,
    iwork: *mut I,
    liwork: isize,
    safmin: f64,
    info: *mut I,
) -> bool {
    *info = I::from_i64(0);
    let shift = if T::COMPLEX { 1 } else { 0 };
    for (arg, ok) in [
        (5, n >= 0),
        (7, lda >= n.max(1)),
        (9, ldb >= n.max(1)),
        (14 - shift, ldq >= 1 && (!wantq || ldq >= n)),
        (16 - shift, ldz >= 1 && (!wantz || ldz >= n)),
    ] {
        if !ok {
            invalid(info, arg);
            return true;
        }
    }
    *m = I::from_i64(0);
    let need = if T::COMPLEX {
        Some(1)
    } else {
        n.checked_mul(4).and_then(|x| x.checked_add(16))
    };
    let Some(need) = need else {
        invalid(info, 22 - shift);
        return true;
    };
    *work = T::real(need as f64);
    *iwork = I::from_i64(1);
    let query = lwork == -1 || liwork == -1;
    // IJOB=0 queries do not inspect SELECT or any matrix entries.
    if query {
        return true;
    }
    for (arg, ok) in [
        (22 - shift, lwork >= need),
        (24 - shift, liwork >= 1),
        (7, matrix_addressable::<T>(n, n, lda)),
        (9, matrix_addressable::<T>(n, n, ldb)),
        (14 - shift, !wantq || matrix_addressable::<T>(n, n, ldq)),
        (16 - shift, !wantz || matrix_addressable::<T>(n, n, ldz)),
    ] {
        if !ok {
            invalid(info, arg);
            return true;
        }
    }
    if n == 0 {
        return true;
    }
    let size = n as usize;
    let mask: Vec<bool> = (0..size).map(|i| (*select.add(i)).val() != 0).collect();
    let copy = |p: *mut T, ld: isize, lower: usize| -> Vec<T> {
        (0..size * size)
            .map(|index| {
                let (i, j) = (index % size, index / size);
                if i <= j + lower {
                    *p.offset(i as isize + j as isize * ld)
                } else {
                    T::z()
                }
            })
            .collect()
    };
    let mut aa = copy(a, lda, usize::from(!T::COMPLEX));
    let mut bb = copy(b, ldb, 0);
    let mut qq = if wantq {
        copy(q, ldq, size)
    } else {
        Vec::new()
    };
    let mut zz = if wantz {
        copy(z, ldz, size)
    } else {
        Vec::new()
    };
    // Complex reference TGSEN publishes the original diagonals before moving
    // blocks and retains those values on a rejected exchange or trivial mask.
    let original_complex = if T::COMPLEX {
        Some((
            (0..size).map(|i| aa[i + i * size]).collect::<Vec<T>>(),
            (0..size).map(|i| bb[i + i * size]).collect::<Vec<T>>(),
        ))
    } else {
        None
    };
    let (count, rejected) = crate::qz_reorder::select_blocks(
        size,
        &mask,
        aa.as_mut_ptr(),
        n,
        bb.as_mut_ptr(),
        n,
        wantq.then_some((qq.as_mut_ptr(), n)),
        wantz.then_some((zz.as_mut_ptr(), n)),
    );
    // Reference TGSEN normalizes rows and the corresponding Q columns.
    for k in 0..size {
        let singleton = T::COMPLEX
            || ((k == 0 || aa[k + (k - 1) * size] == T::z())
                && (k + 1 == size || aa[k + 1 + k * size] == T::z()));
        if singleton && (!T::COMPLEX || (!rejected && count > 0 && count < size)) {
            let diag = bb[k + k * size];
            let mag = diag.magnitude();
            let phase = if T::COMPLEX {
                if mag > safmin {
                    diag.cj() / T::real(mag)
                } else {
                    T::o()
                }
            } else if diag.re().is_sign_negative() {
                -T::o()
            } else {
                T::o()
            };
            for j in k..size {
                aa[k + j * size] = aa[k + j * size] * phase;
                bb[k + j * size] = bb[k + j * size] * phase;
            }
            if wantq {
                for i in 0..size {
                    qq[i + k * size] = qq[i + k * size] * phase.cj();
                }
            }
            if T::COMPLEX {
                bb[k + k * size] = T::real(if mag > safmin { mag } else { 0.0 });
            }
        }
    }
    let Ok((mut values, imag, mut betas)) =
        crate::qz_reorder::saved_values(size, aa.as_ptr(), n, bb.as_ptr(), n)
    else {
        return false;
    };
    if T::COMPLEX && (rejected || count == 0 || count == size) {
        let (old_values, old_betas) = original_complex.unwrap();
        values = old_values;
        betas = old_betas;
    }
    for j in 0..size {
        for i in 0..size {
            if i <= j + usize::from(!T::COMPLEX) {
                *a.offset(i as isize + j as isize * lda) = aa[i + j * size];
            }
            if i <= j {
                *b.offset(i as isize + j as isize * ldb) = bb[i + j * size];
            }
            if wantq {
                *q.offset(i as isize + j as isize * ldq) = qq[i + j * size];
            }
            if wantz {
                *z.offset(i as isize + j as isize * ldz) = zz[i + j * size];
            }
        }
    }
    for i in 0..size {
        *ar.add(i) = values[i];
        if !T::COMPLEX {
            *ai.add(i) = imag[i];
        }
        *beta.add(i) = betas[i];
    }
    *m = I::from_i64(count as i64);
    *info = I::from_i64(i64::from(rejected));
    true
}

// TGEX2 exchanges exactly two adjacent diagonal blocks. Unlike TGEXC,
// its real workspace argument has no query mode.
unsafe fn adjacent<T: ReorderScalar, I: LapackInt>(
    wantq: bool,
    wantz: bool,
    n: isize,
    a: *mut T,
    lda: isize,
    b: *mut T,
    ldb: isize,
    q: *mut T,
    ldq: isize,
    z: *mut T,
    ldz: isize,
    j1: isize,
    n1: isize,
    n2: isize,
    work: Option<(*mut T, isize)>,
    info: *mut I,
) {
    *info = I::from_i64(0);
    if n <= 1 || n1 <= 0 || n2 <= 0 {
        return;
    }
    if !T::COMPLEX && (n1 > n || j1.checked_add(n1).is_some_and(|x| x > n)) {
        return;
    }
    let Some(m) = n1.checked_add(n2) else {
        invalid(info, 14);
        return;
    };
    if let Some((pointer, length)) = work {
        let need = n
            .checked_mul(m)
            .zip(m.checked_mul(m).and_then(|x| x.checked_mul(2)))
            .map(|(x, y)| x.max(y).max(1));
        let Some(need) = need else {
            invalid(info, 16);
            return;
        };
        if length < need {
            *pointer = T::real(need as f64);
            invalid(info, 16);
            return;
        }
    }
    for (arg, ok) in [
        (5, lda >= n && matrix_addressable::<T>(n, n, lda)),
        (7, ldb >= n && matrix_addressable::<T>(n, n, ldb)),
        (
            9,
            ldq >= 1 && (!wantq || (ldq >= n && matrix_addressable::<T>(n, n, ldq))),
        ),
        (
            11,
            ldz >= 1 && (!wantz || (ldz >= n && matrix_addressable::<T>(n, n, ldz))),
        ),
        (12, j1 >= 1 && j1.checked_add(m - 1).is_some_and(|x| x <= n)),
        (13, n1 <= 2),
        (14, n2 <= 2),
    ] {
        if !ok {
            invalid(info, arg);
            return;
        }
    }
    let start = j1 - 1;
    // The explicit real block sizes must describe the actual Schur blocks;
    // otherwise a multi-step move could partially mutate a requested single swap.
    if !T::COMPLEX {
        let width = |index: isize| -> isize {
            if index + 1 < n && *a.offset(index + 1 + index * lda) != T::z() {
                2
            } else {
                1
            }
        };
        if first(n, start, a, lda) != Some(start) || width(start) != n1 || width(start + n1) != n2 {
            *info = I::from_i64(1);
            return;
        }
    }
    let result = crate::qz_reorder::move_pair(
        n as usize,
        (start + n1) as usize,
        start as usize,
        a,
        lda,
        b,
        ldb,
        wantq.then_some((q, ldq)),
        wantz.then_some((z, ldz)),
    );
    if result.rejected {
        *info = I::from_i64(1);
    }
}
macro_rules! adjacent_real_exports {($p:ident,$t:ty,$i:ty,$s:tt)=>{paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p tgex2 $s>](wq:*const $i,wz:*const $i,n:*const $i,a:*mut $t,lda:*const $i,b:*mut $t,ldb:*const $i,q:*mut $t,ldq:*const $i,z:*mut $t,ldz:*const $i,j1:*const $i,n1:*const $i,n2:*const $i,work:*mut $t,lwork:*const $i,info:*mut $i) {
        adjacent((*wq).val()!=0,(*wz).val()!=0,(*n).val(),a,(*lda).val(),b,(*ldb).val(),q,(*ldq).val(),z,(*ldz).val(),(*j1).val(),(*n1).val(),(*n2).val(),Some((work,(*lwork).val())),info);
    }
}};}
macro_rules! adjacent_complex_exports {($p:ident,$t:ty,$i:ty,$s:tt)=>{paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p tgex2 $s>](wq:*const $i,wz:*const $i,n:*const $i,a:*mut $t,lda:*const $i,b:*mut $t,ldb:*const $i,q:*mut $t,ldq:*const $i,z:*mut $t,ldz:*const $i,j1:*const $i,info:*mut $i) {
        adjacent((*wq).val()!=0,(*wz).val()!=0,(*n).val(),a,(*lda).val(),b,(*ldb).val(),q,(*ldq).val(),z,(*ldz).val(),(*j1).val(),1,1,None,info);
    }
}};}
adjacent_real_exports!(s, f32, i32, _);
adjacent_real_exports!(d, f64, i32, _);
adjacent_real_exports!(s, f32, i64, _64_);
adjacent_real_exports!(d, f64, i64, _64_);
adjacent_complex_exports!(c, colmat::c32, i32, _);
adjacent_complex_exports!(z, colmat::c64, i32, _);
adjacent_complex_exports!(c, colmat::c32, i64, _64_);
adjacent_complex_exports!(z, colmat::c64, i64, _64_);

use std::ffi::c_void;
use std::sync::atomic::{AtomicPtr, Ordering};
static EXPERT: [AtomicPtr<c_void>; 8] = [const { AtomicPtr::new(std::ptr::null_mut()) }; 8];
#[no_mangle]
pub extern "C" fn colmat_qz_reorder_set_fallback(id: i32, pointer: *mut c_void) {
    if let Some(slot) = EXPERT.get(id as usize) {
        slot.store(pointer, Ordering::Release);
    }
}
#[no_mangle]
pub extern "C" fn colmat_qz_reorder_has_fallback(id: i32) -> i32 {
    EXPERT
        .get(id as usize)
        .is_some_and(|slot| !slot.load(Ordering::Acquire).is_null()) as i32
}
macro_rules! selected_real_exports {($p:ident,$t:ty,$r:ty,$i:ty,$s:tt,$id:expr)=>{paste!{
#[no_mangle]
pub unsafe extern "C" fn [<$p tgsen $s>](ijob:*const $i,wq:*const $i,wz:*const $i,select:*const $i,n:*const $i,a:*mut $t,lda:*const $i,b:*mut $t,ldb:*const $i,ar:*mut $t,ai:*mut $t,beta:*mut $t,q:*mut $t,ldq:*const $i,z:*mut $t,ldz:*const $i,m:*mut $i,pl:*mut $r,pr:*mut $r,dif:*mut $r,work:*mut $t,lwork:*const $i,iwork:*mut $i,liwork:*const $i,info:*mut $i) {
    let(job,nn,la,lb,lq,lz,lw,liw)=((*ijob).val(),(*n).val(),(*lda).val(),(*ldb).val(),(*ldq).val(),(*ldz).val(),(*lwork).val(),(*liwork).val());
    let(wantq,wantz)=((*wq).val()!=0,(*wz).val()!=0);
    *info=0;
    for(arg,ok)in[(1,(0..=5).contains(&job)),(5,nn>=0),(7,la>=nn.max(1)),(9,lb>=nn.max(1)),(14,lq>=1&&(!wantq||lq>=nn)),(16,lz>=1&&(!wantz||lz>=nn))] {
        if !ok {invalid(info,arg);return;}
    }
    if job==0 && selected(wantq,wantz,select,nn,a,la,b,lb,ar,ai,beta,q,lq,z,lz,m,work,lw,iwork,liw,<$r>::MIN_POSITIVE as f64,info) {return;}
    // Expert estimates, and native extraction refusal, run the saved ILP64
    // routine from the original factors. SELECT is read-only; LP64 integer
    // workspace copies back only slots actually written by LAPACK.
    for(arg,ok)in[(7,matrix_addressable::<$t>(nn,nn,la)),(9,matrix_addressable::<$t>(nn,nn,lb)),(14,!wantq||matrix_addressable::<$t>(nn,nn,lq)),(16,!wantz||matrix_addressable::<$t>(nn,nn,lz))] {if !ok {invalid(info,arg);return;}}
    let pointer=EXPERT[$id].load(Ordering::Acquire);
    assert!(!pointer.is_null(),"initialize QZ reorder expert fallback before use");
    let(job64,wq64,wz64,n64,la64,lb64,lq64,lz64,lw64,liw64)=(job as i64,i64::from(wantq),i64::from(wantz),nn as i64,la as i64,lb as i64,lq as i64,lz as i64,lw as i64,liw as i64);
    let mut selected64=Vec::new();let mut workspace64=Vec::new();
    let (selptr,iwptr)=if std::mem::size_of::<$i>()==8 {(select.cast::<i64>(),iwork.cast::<i64>())} else {
        selected64.extend((0..nn as usize).map(|k|(*select.add(k)).val() as i64));
        workspace64.resize(liw.max(1) as usize,i64::MIN);
        (selected64.as_ptr(),workspace64.as_mut_ptr())
    };
    let(mut m64,mut info64)=(0i64,0i64);
    let fallback:unsafe extern "C" fn(*const i64,*const i64,*const i64,*const i64,*const i64,*mut $t,*const i64,*mut $t,*const i64,*mut $t,*mut $t,*mut $t,*mut $t,*const i64,*mut $t,*const i64,*mut i64,*mut $r,*mut $r,*mut $r,*mut $t,*const i64,*mut i64,*const i64,*mut i64)=std::mem::transmute(pointer);
    fallback(&job64,&wq64,&wz64,selptr,&n64,a,&la64,b,&lb64,ar,ai,beta,q,&lq64,z,&lz64,&mut m64,pl,pr,dif,work,&lw64,iwptr,&liw64,&mut info64);
    *m=m64 as $i;*info=info64 as $i;
    for(k,value)in workspace64.into_iter().enumerate() {if value!=i64::MIN {*iwork.add(k)=value as $i;}}
}
}};}
macro_rules! selected_complex_exports {($p:ident,$t:ty,$r:ty,$i:ty,$s:tt,$id:expr)=>{paste!{
#[no_mangle]
pub unsafe extern "C" fn [<$p tgsen $s>](ijob:*const $i,wq:*const $i,wz:*const $i,select:*const $i,n:*const $i,a:*mut $t,lda:*const $i,b:*mut $t,ldb:*const $i,ar:*mut $t,beta:*mut $t,q:*mut $t,ldq:*const $i,z:*mut $t,ldz:*const $i,m:*mut $i,pl:*mut $r,pr:*mut $r,dif:*mut $r,work:*mut $t,lwork:*const $i,iwork:*mut $i,liwork:*const $i,info:*mut $i) {
    let(job,nn,la,lb,lq,lz,lw,liw)=((*ijob).val(),(*n).val(),(*lda).val(),(*ldb).val(),(*ldq).val(),(*ldz).val(),(*lwork).val(),(*liwork).val());
    let(wantq,wantz)=((*wq).val()!=0,(*wz).val()!=0);
    *info=0;
    for(arg,ok)in[(1,(0..=5).contains(&job)),(5,nn>=0),(7,la>=nn.max(1)),(9,lb>=nn.max(1)),(13,lq>=1&&(!wantq||lq>=nn)),(15,lz>=1&&(!wantz||lz>=nn))] {
        if !ok {invalid(info,arg);return;}
    }
    if job==0 && selected(wantq,wantz,select,nn,a,la,b,lb,ar,std::ptr::null_mut(),beta,q,lq,z,lz,m,work,lw,iwork,liw,<$r>::MIN_POSITIVE as f64,info) {return;}
    // Expert estimates, and native extraction refusal, run the saved ILP64
    // routine from the original factors. SELECT is read-only; LP64 integer
    // workspace copies back only slots actually written by LAPACK.
    for(arg,ok)in[(7,matrix_addressable::<$t>(nn,nn,la)),(9,matrix_addressable::<$t>(nn,nn,lb)),(13,!wantq||matrix_addressable::<$t>(nn,nn,lq)),(15,!wantz||matrix_addressable::<$t>(nn,nn,lz))] {if !ok {invalid(info,arg);return;}}
    let pointer=EXPERT[$id].load(Ordering::Acquire);
    assert!(!pointer.is_null(),"initialize QZ reorder expert fallback before use");
    let(job64,wq64,wz64,n64,la64,lb64,lq64,lz64,lw64,liw64)=(job as i64,i64::from(wantq),i64::from(wantz),nn as i64,la as i64,lb as i64,lq as i64,lz as i64,lw as i64,liw as i64);
    let mut selected64=Vec::new();let mut workspace64=Vec::new();
    let (selptr,iwptr)=if std::mem::size_of::<$i>()==8 {(select.cast::<i64>(),iwork.cast::<i64>())} else {
        selected64.extend((0..nn as usize).map(|k|(*select.add(k)).val() as i64));
        workspace64.resize(liw.max(1) as usize,i64::MIN);
        (selected64.as_ptr(),workspace64.as_mut_ptr())
    };
    let(mut m64,mut info64)=(0i64,0i64);
    let fallback:unsafe extern "C" fn(*const i64,*const i64,*const i64,*const i64,*const i64,*mut $t,*const i64,*mut $t,*const i64,*mut $t,*mut $t,*mut $t,*const i64,*mut $t,*const i64,*mut i64,*mut $r,*mut $r,*mut $r,*mut $t,*const i64,*mut i64,*const i64,*mut i64)=std::mem::transmute(pointer);
    fallback(&job64,&wq64,&wz64,selptr,&n64,a,&la64,b,&lb64,ar,beta,q,&lq64,z,&lz64,&mut m64,pl,pr,dif,work,&lw64,iwptr,&liw64,&mut info64);
    *m=m64 as $i;*info=info64 as $i;
    for(k,value)in workspace64.into_iter().enumerate() {if value!=i64::MIN {*iwork.add(k)=value as $i;}}
}
}};}
selected_real_exports!(s, f32, f32, i32, _, 0);
selected_real_exports!(s, f32, f32, i64, _64_, 0);
selected_real_exports!(d, f64, f64, i32, _, 1);
selected_real_exports!(d, f64, f64, i64, _64_, 1);
selected_complex_exports!(c, colmat::c32, f32, i32, _, 2);
selected_complex_exports!(c, colmat::c32, f32, i64, _64_, 2);
selected_complex_exports!(z, colmat::c64, f64, i32, _, 3);
selected_complex_exports!(z, colmat::c64, f64, i64, _64_, 3);

macro_rules! real_exports {($p:ident,$t:ty,$i:ty,$s:tt)=>{paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p tgexc $s>](wq:*const $i,wz:*const $i,n:*const $i,a:*mut $t,lda:*const $i,b:*mut $t,ldb:*const $i,q:*mut $t,ldq:*const $i,z:*mut $t,ldz:*const $i,ifst:*mut $i,ilst:*mut $i,work:*mut $t,lwork:*const $i,info:*mut $i) {
        exchange((*wq).val()!=0,(*wz).val()!=0,(*n).val(),a,(*lda).val(),b,(*ldb).val(),q,(*ldq).val(),z,(*ldz).val(),ifst,ilst,Some((work,(*lwork).val())),info);
    }
}};}
macro_rules! complex_exports {($p:ident,$t:ty,$i:ty,$s:tt)=>{paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p tgexc $s>](wq:*const $i,wz:*const $i,n:*const $i,a:*mut $t,lda:*const $i,b:*mut $t,ldb:*const $i,q:*mut $t,ldq:*const $i,z:*mut $t,ldz:*const $i,ifst:*mut $i,ilst:*mut $i,info:*mut $i) {
        exchange((*wq).val()!=0,(*wz).val()!=0,(*n).val(),a,(*lda).val(),b,(*ldb).val(),q,(*ldq).val(),z,(*ldz).val(),ifst,ilst,None,info);
    }
}};}
real_exports!(s, f32, i32, _);
real_exports!(d, f64, i32, _);
real_exports!(s, f32, i64, _64_);
real_exports!(d, f64, i64, _64_);
complex_exports!(c, colmat::c32, i32, _);
complex_exports!(z, colmat::c64, i32, _);
complex_exports!(c, colmat::c32, i64, _64_);
complex_exports!(z, colmat::c64, i64, _64_);

#[cfg(test)]
mod tests {
    use super::*;
    fn abi<T: ReorderScalar, I: LapackInt>(tol: f64) {
        let n = 3;
        let ld = 5;
        let mut a = vec![T::real(777.0); ld * n];
        let mut b = a.clone();
        for j in 0..n {
            for i in 0..n {
                a[i + j * ld] = if i == j {
                    T::real((i + 1) as f64)
                } else {
                    T::z()
                };
                b[i + j * ld] = if i == j { T::o() } else { T::z() };
            }
        }
        let mut q = vec![T::real(777.0); 1];
        let mut z = q.clone();
        let mut from = [I::from_i64(-777), I::from_i64(3), I::from_i64(-777)];
        let mut to = [I::from_i64(-777), I::from_i64(1), I::from_i64(-777)];
        let mut info = [I::from_i64(-777); 3];
        let mut work = [T::real(777.0); 30];
        unsafe {
            exchange(
                false,
                false,
                n as isize,
                a.as_mut_ptr(),
                ld as isize,
                b.as_mut_ptr(),
                ld as isize,
                q.as_mut_ptr(),
                1,
                z.as_mut_ptr(),
                1,
                &mut from[1],
                &mut to[1],
                if T::COMPLEX {
                    None
                } else {
                    Some((work.as_mut_ptr(), 28))
                },
                &mut info[1],
            );
        }
        assert_eq!(info[1].val(), 0);
        assert_eq!(crate::colmat_blas_last_error(), 0);
        assert_eq!(from[1].val(), 3);
        assert_eq!(to[1].val(), 1);
        assert!((a[0] / b[0] - T::real(3.0)).magnitude() < tol);
        for array in [from, to, info] {
            assert_eq!(array[0].val(), -777);
            assert_eq!(array[2].val(), -777);
        }
        assert_eq!(q[0].re(), 777.0);
        assert_eq!(z[0].re(), 777.0);
        for j in 0..n {
            for i in n..ld {
                assert_eq!(a[i + j * ld].re(), 777.0);
                assert_eq!(b[i + j * ld].re(), 777.0);
            }
        }
    }
    fn selected_abi<T: ReorderScalar, I: LapackInt>(tol: f64) {
        let mut a = [T::real(-1.0), T::z(), T::z(), T::real(-3.0)];
        let mut b = [-T::o(), T::z(), T::z(), -T::o()];
        let mut q = [T::o(), T::z(), T::z(), T::o()];
        let mut z = q;
        let mask = [I::from_i64(0), I::from_i64(1)];
        let mut ar = [T::real(777.0); 2];
        let mut ai = ar;
        let mut beta = ar;
        let mut m = I::from_i64(-777);
        let mut iw = I::from_i64(-777);
        let mut work = [T::real(777.0); 24];
        let mut info = I::from_i64(-777);
        let olda = a;
        let oldb = b;
        let ok = unsafe {
            selected(
                true,
                true,
                mask.as_ptr(),
                2,
                a.as_mut_ptr(),
                2,
                b.as_mut_ptr(),
                2,
                ar.as_mut_ptr(),
                ai.as_mut_ptr(),
                beta.as_mut_ptr(),
                q.as_mut_ptr(),
                2,
                z.as_mut_ptr(),
                2,
                &mut m,
                work.as_mut_ptr(),
                24,
                &mut iw,
                1,
                f64::MIN_POSITIVE,
                &mut info,
            )
        };
        assert!(ok);
        assert_eq!(info.val(), 0);
        assert_eq!(m.val(), 1);
        assert_eq!(iw.val(), 1);
        assert_eq!(crate::colmat_blas_last_error(), 0);
        for (i, value) in [3.0, 1.0].into_iter().enumerate() {
            assert!((ar[i] / beta[i] - T::real(value)).magnitude() < tol);
            assert!(beta[i].re() >= 0.0);
        }
        for old in [olda, oldb] {
            let current = if old == olda { a } else { b };
            for j in 0..2 {
                for i in 0..2 {
                    let mut value = T::z();
                    for k in 0..2 {
                        for l in 0..2 {
                            value = value + q[i + k * 2] * current[k + l * 2] * z[j + l * 2].cj();
                        }
                    }
                    assert!((value - old[i + j * 2]).magnitude() < tol);
                }
            }
        }
        if T::COMPLEX {
            assert_eq!(ai[0].re(), 777.0);
        } else {
            assert_eq!(ai[0].re(), 0.0);
        }
    }
    #[test]
    fn selected_all_scalar_and_integer_widths() {
        selected_abi::<f32, i32>(1e-4);
        selected_abi::<f32, i64>(1e-4);
        selected_abi::<f64, i32>(1e-11);
        selected_abi::<f64, i64>(1e-11);
        selected_abi::<colmat::c32, i32>(1e-4);
        selected_abi::<colmat::c32, i64>(1e-4);
        selected_abi::<colmat::c64, i32>(1e-11);
        selected_abi::<colmat::c64, i64>(1e-11);
    }
    #[test]
    fn selected_complex_trivial_mask_keeps_original_phase() {
        type C = colmat::c64;
        for take in [0i64, 1] {
            let mut a = C::new(2.0, 3.0);
            let mut b = C::new(-1.0, 2.0);
            let (olda, oldb) = (a, b);
            let mut ar = C::new(777.0, 0.0);
            let mut beta = ar;
            let mut work = ar;
            let mut m = -777i64;
            let mut iw = -777i64;
            let mut info = -777i64;
            assert!(unsafe {
                selected(
                    false,
                    false,
                    &take,
                    1,
                    &mut a,
                    1,
                    &mut b,
                    1,
                    &mut ar,
                    std::ptr::null_mut(),
                    &mut beta,
                    std::ptr::null_mut(),
                    1,
                    std::ptr::null_mut(),
                    1,
                    &mut m,
                    &mut work,
                    1,
                    &mut iw,
                    1,
                    f64::MIN_POSITIVE,
                    &mut info,
                )
            });
            assert_eq!((a, b, ar, beta), (olda, oldb, olda, oldb));
            assert_eq!((m, info), (take, 0));
        }
    }
    #[test]
    fn selected_query_reads_no_matrices_or_mask() {
        let mut dummy = 777.0f64;
        let mut m = -777i64;
        let mut work = 777.0;
        let mut iw = -777i64;
        let mut info = -777i64;
        assert!(unsafe {
            selected(
                false,
                false,
                std::ptr::null(),
                3,
                std::ptr::null_mut(),
                3,
                std::ptr::null_mut(),
                3,
                &mut dummy,
                &mut dummy,
                &mut dummy,
                std::ptr::null_mut(),
                1,
                std::ptr::null_mut(),
                1,
                &mut m,
                &mut work,
                -1,
                &mut iw,
                0,
                f64::MIN_POSITIVE,
                &mut info,
            )
        });
        assert_eq!((m, iw, info), (0, 1, 0));
        assert_eq!(work, 28.0);
        assert_eq!(dummy, 777.0);
    }
    fn adjacent_abi<T: ReorderScalar, I: LapackInt>(tol: f64) {
        let mut a = [T::real(1.0), T::z(), T::real(0.25), T::real(3.0)];
        let mut b = [T::o(), T::z(), T::z(), T::o()];
        let mut dummy = T::real(777.0);
        let mut work = [T::real(777.0); 8];
        let mut info = I::from_i64(-777);
        unsafe {
            adjacent(
                false,
                false,
                2,
                a.as_mut_ptr(),
                2,
                b.as_mut_ptr(),
                2,
                &mut dummy,
                1,
                &mut dummy,
                1,
                1,
                1,
                1,
                if T::COMPLEX {
                    None
                } else {
                    Some((work.as_mut_ptr(), 8))
                },
                &mut info,
            );
        }
        assert_eq!(info.val(), 0);
        assert_eq!(crate::colmat_blas_last_error(), 0);
        assert!((a[0] / b[0] - T::real(3.0)).magnitude() < tol);
        assert!((a[3] / b[3] - T::real(1.0)).magnitude() < tol);
        assert_eq!(dummy.re(), 777.0);
    }
    #[test]
    fn adjacent_all_scalar_and_integer_widths() {
        adjacent_abi::<f32, i32>(1e-4);
        adjacent_abi::<f32, i64>(1e-4);
        adjacent_abi::<f64, i32>(1e-11);
        adjacent_abi::<f64, i64>(1e-11);
        adjacent_abi::<colmat::c32, i32>(1e-4);
        adjacent_abi::<colmat::c32, i64>(1e-4);
        adjacent_abi::<colmat::c64, i32>(1e-11);
        adjacent_abi::<colmat::c64, i64>(1e-11);
    }
    #[test]
    fn adjacent_real_workspace_is_not_a_query() {
        let mut scalar = 777.0f64;
        let mut work = 777.0;
        let mut info = -777i64;
        unsafe {
            adjacent(
                false,
                false,
                2,
                &mut scalar,
                2,
                &mut scalar,
                2,
                &mut scalar,
                1,
                &mut scalar,
                1,
                1,
                1,
                1,
                Some((&mut work, -1)),
                &mut info,
            );
        }
        assert_eq!(info, -16);
        assert_eq!(crate::colmat_blas_last_error(), 16);
        assert_eq!(work, 8.0);
        assert_eq!(scalar, 777.0);
    }
    #[test]
    fn all_scalar_and_integer_widths() {
        abi::<f32, i32>(1e-4);
        abi::<f32, i64>(1e-4);
        abi::<f64, i32>(1e-11);
        abi::<f64, i64>(1e-11);
        abi::<colmat::c32, i32>(1e-4);
        abi::<colmat::c32, i64>(1e-4);
        abi::<colmat::c64, i32>(1e-11);
        abi::<colmat::c64, i64>(1e-11);
    }
    #[test]
    fn real_query_and_empty_index_validation() {
        let mut scalar = 777.0f64;
        let mut from = 2i64;
        let mut to = 1i64;
        let mut info = -777i64;
        let mut work = 777.0;
        unsafe {
            exchange(
                false,
                false,
                3,
                &mut scalar,
                3,
                &mut scalar,
                3,
                &mut scalar,
                1,
                &mut scalar,
                1,
                &mut from,
                &mut to,
                Some((&mut work, -1)),
                &mut info,
            );
        }
        assert_eq!(info, 0);
        assert_eq!(work, 28.0);
        assert_eq!(scalar, 777.0);
        from = 1;
        to = 1;
        work = 777.0;
        unsafe {
            exchange(
                false,
                false,
                0,
                &mut scalar,
                1,
                &mut scalar,
                1,
                &mut scalar,
                1,
                &mut scalar,
                1,
                &mut from,
                &mut to,
                Some((&mut work, -1)),
                &mut info,
            );
        }
        assert_eq!(info, -12);
        assert_eq!(crate::colmat_blas_last_error(), 12);
        assert_eq!(work, 777.0);
    }
}
