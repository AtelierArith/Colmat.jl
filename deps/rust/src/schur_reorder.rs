//! Ordinary Schur block moves with LAPACK index and rejection contracts.
use crate::kernels::{flag, matrix_addressable, Scalar};
use crate::lapack::LapackInt;
use crate::schur::SchurScalar;
use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::{Mat, MatMut};
use paste::paste;
trait MoveScalar: SchurScalar {
    fn move_block(
        t: MatMut<'_, Self>,
        q: Option<MatMut<'_, Self>>,
        from: usize,
        to: usize,
    ) -> Result<usize, usize>;
}
macro_rules! real_move {
    ($t:ty) => {
        impl MoveScalar for $t {
            fn move_block(
                t: MatMut<'_, Self>,
                q: Option<MatMut<'_, Self>>,
                from: usize,
                to: usize,
            ) -> Result<usize, usize> {
                use colmat::linalg::evd::{reorder_schur_real, reorder_schur_scratch};
                crate::threading::with_par(|par| {
                    let mut memory = MemBuffer::new(reorder_schur_scratch::<Self>(t.nrows()));
                    reorder_schur_real(t, q, from, to, par, MemStack::new(&mut memory))
                        .map_err(|e| e.position)
                })
            }
        }
    };
}
macro_rules! complex_move {
    ($t:ty) => {
        impl MoveScalar for $t {
            fn move_block(
                t: MatMut<'_, Self>,
                q: Option<MatMut<'_, Self>>,
                from: usize,
                to: usize,
            ) -> Result<usize, usize> {
                use colmat::linalg::evd::{reorder_schur_cplx, reorder_schur_scratch};
                crate::threading::with_par(|par| {
                    let mut memory = MemBuffer::new(reorder_schur_scratch::<Self>(t.nrows()));
                    Ok(reorder_schur_cplx(
                        t,
                        q,
                        from,
                        to,
                        par,
                        MemStack::new(&mut memory),
                    ))
                })
            }
        }
    };
}
real_move!(f32);
real_move!(f64);
complex_move!(colmat::c32);
complex_move!(colmat::c64);
unsafe fn fail<I: LapackInt>(info: *mut I, arg: i64) {
    *info = I::from_i64(-arg);
    crate::invalid(arg as i32);
}
unsafe fn stage<T: Scalar>(n: usize, t: *const T, ldt: isize) -> Mat<T> {
    Mat::from_fn(n, n, |i, j| {
        if i <= j || (!T::COMPLEX && i == j + 1) {
            *t.offset(i as isize + j as isize * ldt)
        } else {
            T::z()
        }
    })
}
unsafe fn perform<T: MoveScalar>(
    n: usize,
    t: *mut T,
    ldt: isize,
    q: *mut T,
    ldq: isize,
    want: bool,
    from: usize,
    to: usize,
) -> (usize, bool) {
    let mut tcopy = stage(n, t, ldt);
    let mut qcopy = if want {
        Some(crate::lapack::input_ref(q, n, n, ldq).to_owned())
    } else {
        None
    };
    let result = T::move_block(tcopy.as_mut(), qcopy.as_mut().map(|x| x.as_mut()), from, to);
    // A failed move commits exactly the state of the last successful swap.
    crate::lapack::output(&tcopy, t, ldt);
    if let Some(qcopy) = qcopy {
        crate::lapack::output(&qcopy, q, ldq);
    }
    match result {
        Ok(position) => (position, false),
        Err(position) => (position, true),
    }
}
unsafe fn trexc<T: MoveScalar, I: LapackInt>(
    comp: u8,
    n: isize,
    t: *mut T,
    ldt: isize,
    q: *mut T,
    ldq: isize,
    first: *mut I,
    last: *mut I,
    info: *mut I,
) {
    *info = I::from_i64(0);
    let want = comp == b'V';
    for (arg, ok) in [
        (1, matches!(comp, b'V' | b'N')),
        (2, n >= 0),
        (4, ldt >= n.max(1)),
        (6, ldq >= 1 && (!want || ldq >= n)),
        (7, n == 0 || ((*first).val() >= 1 && (*first).val() <= n)),
        (8, n == 0 || ((*last).val() >= 1 && (*last).val() <= n)),
    ] {
        if !ok {
            fail(info, arg);
            return;
        }
    }
    if n <= 1 {
        return;
    }
    let (mut from, mut to) = (((*first).val() - 1) as usize, ((*last).val() - 1) as usize);
    if T::COMPLEX && from == to {
        return;
    }
    if !matrix_addressable::<T>(n, n, ldt) {
        fail(info, 4);
        return;
    }
    if !T::COMPLEX {
        if from > 0 && *t.offset(from as isize + (from - 1) as isize * ldt) != T::z() {
            from -= 1;
        }
        if to > 0 && *t.offset(to as isize + (to - 1) as isize * ldt) != T::z() {
            to -= 1;
        }
        *first = I::from_i64(from as i64 + 1);
        *last = I::from_i64(to as i64 + 1);
    }
    if from == to {
        return;
    }
    if want && !matrix_addressable::<T>(n, n, ldq) {
        fail(info, 6);
        return;
    }
    let (position, rejected) = perform(n as usize, t, ldt, q, ldq, want, from, to);
    if !T::COMPLEX {
        *last = I::from_i64(position as i64 + 1);
    }
    *info = I::from_i64(rejected as i64);
}
unsafe fn laexc<T: MoveScalar, I: LapackInt>(
    want: bool,
    n: isize,
    t: *mut T,
    ldt: isize,
    q: *mut T,
    ldq: isize,
    j: isize,
    n1: isize,
    n2: isize,
    info: *mut I,
) {
    *info = I::from_i64(0);
    if n == 0 || n1 == 0 || n2 == 0 {
        return;
    }
    for (arg, ok) in [
        (2, n >= 0),
        (4, ldt >= n.max(1)),
        (6, ldq >= 1 && (!want || ldq >= n)),
        (8, (1..=2).contains(&n1)),
        (9, (1..=2).contains(&n2)),
        (
            7,
            j >= 1
                && j.checked_add(n1)
                    .and_then(|v| v.checked_add(n2))
                    .is_some_and(|v| v - 1 <= n),
        ),
    ] {
        if !ok {
            fail(info, arg);
            return;
        }
    }
    if !matrix_addressable::<T>(n, n, ldt) {
        fail(info, 4);
        return;
    }
    if want && !matrix_addressable::<T>(n, n, ldq) {
        fail(info, 6);
        return;
    }
    let (_, rejected) = perform(
        n as usize,
        t,
        ldt,
        q,
        ldq,
        want,
        (j - 1) as usize,
        (j - 1 + n1) as usize,
    );
    *info = I::from_i64(rejected as i64);
}
// Spectrum of an already standardized Schur form. Separate square roots
// preserve a tiny conjugate-pair imaginary part next to a large real part.
unsafe fn values<T: Scalar>(n: usize, t: *const T, ld: isize) -> (Vec<T>, Vec<f64>) {
    let w = (0..n)
        .map(|j| *t.offset(j as isize + j as isize * ld))
        .collect();
    let mut wi = vec![0.0; n];
    if !T::COMPLEX {
        let mut j = 0;
        while j + 1 < n {
            let c = *t.offset((j + 1) as isize + j as isize * ld);
            if c != T::z() {
                let b = *t.offset(j as isize + (j + 1) as isize * ld);
                let beta = b.abs1().sqrt() * c.abs1().sqrt();
                wi[j] = beta;
                wi[j + 1] = -beta;
                j += 2;
            } else {
                j += 1;
            }
        }
    }
    (w, wi)
}
unsafe fn trsen<T: MoveScalar, R: Scalar, I: LapackInt>(
    job: u8,
    comp: u8,
    selected: *const I,
    n: isize,
    t: *mut T,
    ldt: isize,
    q: *mut T,
    ldq: isize,
    w: *mut T,
    wi: *mut R,
    m: *mut I,
    work: *mut T,
    lwork: isize,
    iwork: *mut I,
    liwork: isize,
    info: *mut I,
) -> bool {
    *info = I::from_i64(0);
    let want = comp == b'V';
    for (arg, ok) in [
        (1, matches!(job, b'N' | b'E' | b'V' | b'B')),
        (2, matches!(comp, b'N' | b'V')),
        (4, n >= 0),
        (6, ldt >= n.max(1)),
        (8, ldq >= 1 && (!want || ldq >= n)),
    ] {
        if !ok {
            fail(info, arg);
            return true;
        }
    }
    if job != b'N' {
        return false;
    }
    if !T::COMPLEX && !matrix_addressable::<T>(n, n, ldt) {
        fail(info, 6);
        return true;
    }
    let size = n as usize;
    let mut flags = vec![false; size];
    let mut count = 0;
    let mut j = 0;
    while j < size {
        let pair =
            !T::COMPLEX && j + 1 < size && *t.offset((j + 1) as isize + j as isize * ldt) != T::z();
        let len = if pair { 2 } else { 1 };
        let take = (*selected.add(j)).val() != 0 || (pair && (*selected.add(j + 1)).val() != 0);
        for flag in &mut flags[j..j + len] {
            *flag = take;
        }
        if take {
            count += len;
        }
        j += len;
    }
    *m = I::from_i64(count as i64);
    let minimum = if T::COMPLEX { 1 } else { n.max(1) };
    let query = lwork == -1;
    if !query && lwork < minimum {
        fail(info, if T::COMPLEX { 14 } else { 15 });
        return true;
    }
    // Only LWORK=-1 is a TRSEN query. LIWORK=-1 alone is an error.
    if !T::COMPLEX && !query && liwork < 1 {
        fail(info, 17);
        return true;
    }
    *work = crate::lapack::workspace::<T>(minimum);
    if !T::COMPLEX {
        *iwork = I::from_i64(1);
    }
    if query {
        return true;
    }
    if !matrix_addressable::<T>(n, n, ldt) {
        fail(info, 6);
        return true;
    }
    if count != 0 && count != size {
        if want && !matrix_addressable::<T>(n, n, ldq) {
            fail(info, 8);
            return true;
        }
        let mut a = stage(size, t, ldt);
        let mut basis = if want {
            Some(crate::lapack::input_ref(q, size, size, ldq).to_owned())
        } else {
            None
        };
        let (mut here, mut target) = (0, 0);
        while here < size {
            let len = if !T::COMPLEX && here + 1 < size && a[(here + 1, here)] != T::z() {
                2
            } else {
                1
            };
            if flags[here..here + len].iter().any(|&v| v) {
                if here != target {
                    if T::move_block(a.as_mut(), basis.as_mut().map(|x| x.as_mut()), here, target)
                        .is_err()
                    {
                        *info = I::from_i64(1);
                        break;
                    }
                    flags[target..here + len].rotate_right(len);
                }
                target += len;
            }
            here += len;
        }
        crate::lapack::output(&a, t, ldt);
        if let Some(basis) = basis {
            crate::lapack::output(&basis, q, ldq);
        }
    }
    let (real, imag) = values(size, t, ldt);
    for j in 0..size {
        *w.add(j) = real[j];
        if !T::COMPLEX {
            *wi.add(j) = R::real(imag[j]);
        }
    }
    true
}

use core::ffi::c_void;
use core::sync::atomic::{AtomicPtr, Ordering};
static EXPERT: [AtomicPtr<c_void>; 8] = [const { AtomicPtr::new(core::ptr::null_mut()) }; 8];
#[no_mangle]
pub extern "C" fn colmat_schur_expert_set_fallback(id: i32, p: *mut c_void) {
    if let Some(slot) = EXPERT.get(id as usize) {
        slot.store(p, Ordering::Release);
    }
}
#[no_mangle]
pub extern "C" fn colmat_schur_expert_has_fallback(id: i32) -> i32 {
    EXPERT
        .get(id as usize)
        .is_some_and(|slot| !slot.load(Ordering::Acquire).is_null()) as i32
}
fn expert(id: usize) -> *mut c_void {
    let p = EXPERT[id].load(Ordering::Acquire);
    assert!(
        !p.is_null(),
        "initialize original Schur expert driver before use"
    );
    p
}
unsafe fn select64<I: LapackInt>(n: isize, selected: *const I) -> Vec<i64> {
    (0..n).map(|j| (*selected.offset(j)).val() as i64).collect()
}
macro_rules! trsen_real_exports {($p:ident,$t:ty,$id:expr,$i:ty,$s:tt)=>{paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p trsen $s>](job:*const u8,comp:*const u8,selected:*const $i,n:*const $i,t:*mut $t,ldt:*const $i,q:*mut $t,ldq:*const $i,w:*mut $t,wi:*mut $t,m:*mut $i,condition:*mut $t,sep:*mut $t,work:*mut $t,lwork:*const $i,iwork:*mut $i,liwork:*const $i,info:*mut $i) {
        if trsen(flag(job),flag(comp),selected,*n as isize,t,*ldt as isize,q,*ldq as isize,w,wi,m,work,*lwork as isize,iwork,*liwork as isize,info) {return;}
        let selected=select64(*n as isize,selected);
        let(n,ldt,ldq,lwork,liwork)=(*n as i64,*ldt as i64,*ldq as i64,*lwork as i64,*liwork as i64);
        let mut owned=if core::mem::size_of::<$i>()==4 {vec![i64::MIN;if lwork==-1 {1} else {liwork.max(1) as usize}]} else {Vec::new()};
        let ip=if core::mem::size_of::<$i>()==4 {owned.as_mut_ptr()} else {iwork.cast::<i64>()};
        let(mut count,mut status)=(i64::MIN,0i64);
        let f:unsafe extern "C" fn(*const u8,*const u8,*const i64,*const i64,*mut $t,*const i64,*mut $t,*const i64,*mut $t,*mut $t,*mut i64,*mut $t,*mut $t,*mut $t,*const i64,*mut i64,*const i64,*mut i64,usize,usize)=core::mem::transmute(expert($id));
        f(job,comp,selected.as_ptr(),&n,t,&ldt,q,&ldq,w,wi,&mut count,condition,sep,work,&lwork,ip,&liwork,&mut status,1,1);
        for (j,&v) in owned.iter().enumerate() {if v!=i64::MIN {let value=<$i as LapackInt>::from_i64(v);if value.val() as i64!=v {fail(info,17);return;}*iwork.add(j)=value;}}
        if count!=i64::MIN {*m=count as $i;}*info=status as $i;if status<0 {crate::invalid((-status) as i32);}
    }
}};}
macro_rules! trsen_complex_exports {($p:ident,$t:ty,$r:ty,$id:expr,$i:ty,$s:tt)=>{paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p trsen $s>](job:*const u8,comp:*const u8,selected:*const $i,n:*const $i,t:*mut $t,ldt:*const $i,q:*mut $t,ldq:*const $i,w:*mut $t,m:*mut $i,condition:*mut $r,sep:*mut $r,work:*mut $t,lwork:*const $i,info:*mut $i) {
        if trsen::<$t,$r,$i>(flag(job),flag(comp),selected,*n as isize,t,*ldt as isize,q,*ldq as isize,w,core::ptr::null_mut(),m,work,*lwork as isize,core::ptr::null_mut(),0,info) {return;}
        let selected=select64(*n as isize,selected);
        let(n,ldt,ldq,lwork)=(*n as i64,*ldt as i64,*ldq as i64,*lwork as i64);let(mut count,mut status)=(i64::MIN,0i64);
        let f:unsafe extern "C" fn(*const u8,*const u8,*const i64,*const i64,*mut $t,*const i64,*mut $t,*const i64,*mut $t,*mut i64,*mut $r,*mut $r,*mut $t,*const i64,*mut i64,usize,usize)=core::mem::transmute(expert($id));
        f(job,comp,selected.as_ptr(),&n,t,&ldt,q,&ldq,w,&mut count,condition,sep,work,&lwork,&mut status,1,1);
        if count!=i64::MIN {*m=count as $i;}*info=status as $i;if status<0 {crate::invalid((-status) as i32);}
    }
}};}
macro_rules! trsen_abi {
    ($i:ty,$s:tt) => {
        trsen_real_exports!(s, f32, 0, $i, $s);
        trsen_real_exports!(d, f64, 1, $i, $s);
        trsen_complex_exports!(c, colmat::c32, f32, 2, $i, $s);
        trsen_complex_exports!(z, colmat::c64, f64, 3, $i, $s);
    };
}
trsen_abi!(i32, "_");
trsen_abi!(i64, "_64_");

fn geesx_header<T: Scalar, I: LapackInt>(
    job: u8,
    sort: u8,
    sense: u8,
    n: isize,
    lda: isize,
    ldvs: isize,
    info: &mut I,
) -> bool {
    *info = I::from_i64(0);
    for (arg, ok) in [
        (1, matches!(job, b'N' | b'V')),
        (2, matches!(sort, b'N' | b'S')),
        (
            4,
            matches!(sense, b'N' | b'E' | b'V' | b'B') && (sort == b'S' || sense == b'N'),
        ),
        (5, n >= 0),
        (7, lda >= n.max(1)),
        (
            if T::COMPLEX { 11 } else { 12 },
            ldvs >= 1 && (job == b'N' || ldvs >= n),
        ),
    ] {
        if !ok {
            *info = I::from_i64(-arg);
            crate::invalid(arg as i32);
            return false;
        }
    }
    true
}
macro_rules! real_callback {($t:ty,$guard:ident,$cell:ident,$bridge:ident)=>{
    std::thread_local! {static $cell:std::cell::Cell<Option<unsafe extern "C" fn(*const $t,*const $t)->i32>>=const {std::cell::Cell::new(None)};}
    struct $guard(Option<unsafe extern "C" fn(*const $t,*const $t)->i32>);
    impl $guard {fn new(f:Option<unsafe extern "C" fn(*const $t,*const $t)->i32>)->Self {Self($cell.with(|cell|cell.replace(f)))}}
    impl Drop for $guard {fn drop(&mut self) {$cell.with(|cell|cell.set(self.0));}}
    unsafe extern "C" fn $bridge(a:*const $t,b:*const $t)->i64 {
        $cell.with(|cell|cell.get().expect("active GEESX SELECT callback")(a,b)!=0) as i64
    }
};}
macro_rules! complex_callback {($t:ty,$guard:ident,$cell:ident,$bridge:ident)=>{
    std::thread_local! {static $cell:std::cell::Cell<Option<unsafe extern "C" fn(*const $t)->i32>>=const {std::cell::Cell::new(None)};}
    struct $guard(Option<unsafe extern "C" fn(*const $t)->i32>);
    impl $guard {fn new(f:Option<unsafe extern "C" fn(*const $t)->i32>)->Self {Self($cell.with(|cell|cell.replace(f)))}}
    impl Drop for $guard {fn drop(&mut self) {$cell.with(|cell|cell.set(self.0));}}
    unsafe extern "C" fn $bridge(a:*const $t)->i64 {
        $cell.with(|cell|cell.get().expect("active GEESX SELECT callback")(a)!=0) as i64
    }
};}
real_callback!(f32, SelectS, SELECT_S, select_s);
real_callback!(f64, SelectD, SELECT_D, select_d);
complex_callback!(colmat::c32, SelectC, SELECT_C, select_c);
complex_callback!(colmat::c64, SelectZ, SELECT_Z, select_z);
macro_rules! geesx_real_exports {($p:ident,$t:ty,$id:expr,$guard:ident,$bridge:ident,$i:ty,$s:tt)=>{paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p geesx $s>](job:*const u8,sort:*const u8,select:Option<unsafe extern "C" fn(*const $t,*const $t)->$i>,sense:*const u8,n:*const $i,a:*mut $t,lda:*const $i,sdim:*mut $i,wr:*mut $t,wi:*mut $t,vs:*mut $t,ldvs:*const $i,rconde:*mut $t,rcondv:*mut $t,work:*mut $t,lwork:*const $i,iwork:*mut $i,liwork:*const $i,bwork:*mut $i,info:*mut $i) {
        let(j,s,mode)=(flag(job),flag(sort),flag(sense));let(dim,la,lv,lw,li)=(*n as isize,*lda as isize,*ldvs as isize,*lwork as isize,*liwork as isize);
        if !geesx_header::<$t,$i>(j,s,mode,dim,la,lv,&mut *info) {return;}
        let query=lw==-1||li==-1;
        if mode==b'N' {
            let minimum=match dim.checked_mul(3) {Some(v) if core::mem::size_of::<$i>()!=4||v<=i32::MAX as isize=>v.max(1),_=>{fail(info,5);return;}};
            *work=crate::lapack::workspace::<$t>(minimum);*iwork=1;
            if !query&&lw<minimum {fail(info,16);return;}
            if !query&&li<1 {fail(info,18);return;}
            if query {return;}
            crate::schur::[<$p gees $s>](job,sort,select,n,a,lda,sdim,wr,wi,vs,ldvs,work,lwork,bwork,info);return;
        }
        let(n,lda,ldvs,lwork,liwork)=(dim as i64,la as i64,lv as i64,lw as i64,li as i64);
        let mut integer=if core::mem::size_of::<$i>()==4 {vec![i64::MIN;if query {1} else {li.max(1) as usize}]} else {Vec::new()};
        let ip=if core::mem::size_of::<$i>()==4 {integer.as_mut_ptr()} else {iwork.cast::<i64>()};
        let mut logical=if core::mem::size_of::<$i>()==4 {vec![i64::MIN;if query {1} else {dim.max(1) as usize}]} else {Vec::new()};
        let bp=if core::mem::size_of::<$i>()==4 {logical.as_mut_ptr()} else {bwork.cast::<i64>()};
        let _active=if core::mem::size_of::<$i>()==4 {Some($guard::new(select.map(|p|core::mem::transmute(p))))} else {None};
        let callback:Option<unsafe extern "C" fn(*const $t,*const $t)->i64>=if core::mem::size_of::<$i>()==4 {select.map(|_|$bridge as unsafe extern "C" fn(*const $t,*const $t)->i64)} else {select.map(|p|core::mem::transmute(p))};
        let(mut count,mut status)=(i64::MIN,0i64);
        let f:unsafe extern "C" fn(*const u8,*const u8,Option<unsafe extern "C" fn(*const $t,*const $t)->i64>,*const u8,*const i64,*mut $t,*const i64,*mut i64,*mut $t,*mut $t,*mut $t,*const i64,*mut $t,*mut $t,*mut $t,*const i64,*mut i64,*const i64,*mut i64,*mut i64,usize,usize,usize)=core::mem::transmute(expert(4+$id));
        f(job,sort,callback,sense,&n,a,&lda,&mut count,wr,wi,vs,&ldvs,rconde,rcondv,work,&lwork,ip,&liwork,bp,&mut status,1,1,1);
        for (k,&v) in integer.iter().enumerate() {if v!=i64::MIN {let value=<$i as LapackInt>::from_i64(v);if value.val() as i64!=v {fail(info,18);return;}*iwork.add(k)=value;}}
        for (k,&v) in logical.iter().enumerate() {if v!=i64::MIN {*bwork.add(k)=v as $i;}}
        if count!=i64::MIN {*sdim=count as $i;}*info=status as $i;if status<0 {crate::invalid((-status) as i32);}
    }
}};}
macro_rules! geesx_complex_exports {($p:ident,$t:ty,$r:ty,$id:expr,$guard:ident,$bridge:ident,$i:ty,$s:tt)=>{paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p geesx $s>](job:*const u8,sort:*const u8,select:Option<unsafe extern "C" fn(*const $t)->$i>,sense:*const u8,n:*const $i,a:*mut $t,lda:*const $i,sdim:*mut $i,w:*mut $t,vs:*mut $t,ldvs:*const $i,rconde:*mut $r,rcondv:*mut $r,work:*mut $t,lwork:*const $i,rwork:*mut $r,bwork:*mut $i,info:*mut $i) {
        let(j,s,mode)=(flag(job),flag(sort),flag(sense));let(dim,la,lv,lw)=(*n as isize,*lda as isize,*ldvs as isize,*lwork as isize);
        if !geesx_header::<$t,$i>(j,s,mode,dim,la,lv,&mut *info) {return;}
        let query=lw==-1;
        if mode==b'N' {
            let minimum=match dim.checked_mul(2) {Some(v) if core::mem::size_of::<$i>()!=4||v<=i32::MAX as isize=>v.max(1),_=>{fail(info,5);return;}};
            *work=crate::lapack::workspace::<$t>(minimum);
            if !query&&lw<minimum {fail(info,15);return;}
            if query {return;}
            crate::schur::[<$p gees $s>](job,sort,select,n,a,lda,sdim,w,vs,ldvs,work,lwork,rwork,bwork,info);return;
        }
        let(n,lda,ldvs,lwork)=(dim as i64,la as i64,lv as i64,lw as i64);
        let mut logical=if core::mem::size_of::<$i>()==4 {vec![i64::MIN;if query {1} else {dim.max(1) as usize}]} else {Vec::new()};
        let bp=if core::mem::size_of::<$i>()==4 {logical.as_mut_ptr()} else {bwork.cast::<i64>()};
        let _active=if core::mem::size_of::<$i>()==4 {Some($guard::new(select.map(|p|core::mem::transmute(p))))} else {None};
        let callback:Option<unsafe extern "C" fn(*const $t)->i64>=if core::mem::size_of::<$i>()==4 {select.map(|_|$bridge as unsafe extern "C" fn(*const $t)->i64)} else {select.map(|p|core::mem::transmute(p))};
        let(mut count,mut status)=(i64::MIN,0i64);
        let f:unsafe extern "C" fn(*const u8,*const u8,Option<unsafe extern "C" fn(*const $t)->i64>,*const u8,*const i64,*mut $t,*const i64,*mut i64,*mut $t,*mut $t,*const i64,*mut $r,*mut $r,*mut $t,*const i64,*mut $r,*mut i64,*mut i64,usize,usize,usize)=core::mem::transmute(expert(4+$id));
        f(job,sort,callback,sense,&n,a,&lda,&mut count,w,vs,&ldvs,rconde,rcondv,work,&lwork,rwork,bp,&mut status,1,1,1);
        for (k,&v) in logical.iter().enumerate() {if v!=i64::MIN {*bwork.add(k)=v as $i;}}
        if count!=i64::MIN {*sdim=count as $i;}*info=status as $i;if status<0 {crate::invalid((-status) as i32);}
    }
}};}
macro_rules! geesx_abi {
    ($i:ty,$s:tt) => {
        geesx_real_exports!(s, f32, 0, SelectS, select_s, $i, $s);
        geesx_real_exports!(d, f64, 1, SelectD, select_d, $i, $s);
        geesx_complex_exports!(c, colmat::c32, f32, 2, SelectC, select_c, $i, $s);
        geesx_complex_exports!(z, colmat::c64, f64, 3, SelectZ, select_z, $i, $s);
    };
}
geesx_abi!(i32, "_");
geesx_abi!(i64, "_64_");

macro_rules! real_exports {($p:ident,$t:ty,$i:ty,$s:tt)=>{paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p trexc $s>](comp:*const u8,n:*const $i,t:*mut $t,ldt:*const $i,q:*mut $t,ldq:*const $i,first:*mut $i,last:*mut $i,_work:*mut $t,info:*mut $i) {trexc(flag(comp),*n as isize,t,*ldt as isize,q,*ldq as isize,first,last,info);}
    #[no_mangle] pub unsafe extern "C" fn [<$p laexc $s>](want:*const $i,n:*const $i,t:*mut $t,ldt:*const $i,q:*mut $t,ldq:*const $i,j:*const $i,n1:*const $i,n2:*const $i,_work:*mut $t,info:*mut $i) {laexc((*want).val()!=0,*n as isize,t,*ldt as isize,q,*ldq as isize,*j as isize,*n1 as isize,*n2 as isize,info);}
}};}
macro_rules! complex_exports {($p:ident,$t:ty,$i:ty,$s:tt)=>{paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p trexc $s>](comp:*const u8,n:*const $i,t:*mut $t,ldt:*const $i,q:*mut $t,ldq:*const $i,first:*mut $i,last:*mut $i,info:*mut $i) {trexc(flag(comp),*n as isize,t,*ldt as isize,q,*ldq as isize,first,last,info);}
}};}
macro_rules! abi {
    ($i:ty,$s:tt) => {
        real_exports!(s, f32, $i, $s);
        real_exports!(d, f64, $i, $s);
        complex_exports!(c, colmat::c32, $i, $s);
        complex_exports!(z, colmat::c64, $i, $s);
    };
}
abi!(i32, "_");
abi!(i64, "_64_");

#[cfg(test)]
mod tests {
    use super::*;
    fn verify<T: MoveScalar, I: LapackInt>(imag: T, tol: f64) {
        unsafe {
            for n in [0usize, 1, 5] {
                for from in 0..n.max(1) {
                    for to in 0..n.max(1) {
                        let mut a = Mat::from_fn(n, n, |i, j| {
                            if i > j {
                                T::z()
                            } else if i == j {
                                T::real((i + 2) as f64) + imag * T::real(i as f64 / 7.0)
                            } else {
                                T::real((i + j + 1) as f64 / 11.0)
                                    + imag * T::real((i + 2 * j) as f64 / 13.0)
                            }
                        });
                        if !T::COMPLEX && n == 5 {
                            a[(0, 0)] = T::real(2.0);
                            a[(1, 1)] = T::real(2.0);
                            a[(0, 1)] = T::real(3.0);
                            a[(1, 0)] = T::real(-2.0);
                            a[(3, 3)] = T::real(7.0);
                            a[(4, 4)] = T::real(7.0);
                            a[(3, 4)] = T::real(-4.0);
                            a[(4, 3)] = T::real(1.0);
                        }
                        let canonical = |j: usize| {
                            if !T::COMPLEX && j > 0 && a[(j, j - 1)] != T::z() {
                                j - 1
                            } else {
                                j
                            }
                        };
                        let (first, destination) = if n > 0 {
                            (canonical(from), canonical(to))
                        } else {
                            (0, 0)
                        };
                        let block_size = |j: usize| {
                            if !T::COMPLEX && j + 1 < n && a[(j + 1, j)] != T::z() {
                                2
                            } else {
                                1
                            }
                        };
                        let expected = if first < destination {
                            destination + block_size(destination) - block_size(first)
                        } else {
                            destination
                        };
                        for want in [false, true] {
                            let ld = n + 2;
                            let mut t = vec![T::real(777.0); ld * n];
                            let mut q = t.clone();
                            for j in 0..n {
                                for i in 0..n {
                                    t[i + j * ld] = a[(i, j)];
                                    q[i + j * ld] = if i == j { T::o() } else { T::z() };
                                }
                            }
                            let old_q = q.clone();
                            let mut ifst = I::from_i64(if n == 0 { -777 } else { from as i64 + 1 });
                            let mut ilst = I::from_i64(if n == 0 { 888 } else { to as i64 + 1 });
                            let mut info = I::from_i64(-777);
                            let old_first = ifst.val();
                            let old_last = ilst.val();
                            trexc(
                                if want { b'V' } else { b'N' },
                                n as isize,
                                t.as_mut_ptr(),
                                ld as isize,
                                if want {
                                    q.as_mut_ptr()
                                } else {
                                    core::ptr::null_mut()
                                },
                                if want { ld as isize } else { 1 },
                                &mut ifst,
                                &mut ilst,
                                &mut info,
                            );
                            assert_eq!(info.val(), 0);
                            if !T::COMPLEX && n > 1 {
                                assert_eq!(ifst.val(), first as isize + 1);
                                assert_eq!(ilst.val(), expected as isize + 1);
                            } else {
                                assert_eq!(ifst.val(), old_first);
                                assert_eq!(ilst.val(), old_last);
                            }
                            for j in 0..n {
                                for i in n..ld {
                                    assert_eq!(t[i + j * ld], T::real(777.0));
                                    assert_eq!(q[i + j * ld], T::real(777.0));
                                }
                            }
                            if !want {
                                assert_eq!(q, old_q);
                                continue;
                            }
                            for j in 0..n {
                                for i in 0..n {
                                    let mut rebuilt = T::z();
                                    let mut gram = T::z();
                                    for k in 0..n {
                                        gram = gram + q[k + i * ld].cj() * q[k + j * ld];
                                        for l in 0..n {
                                            rebuilt = rebuilt
                                                + q[i + k * ld]
                                                    * t[k + l * ld]
                                                    * q[j + l * ld].cj();
                                        }
                                    }
                                    assert!(
                                        (rebuilt - a[(i, j)]).abs1() < tol,
                                        "n={n} from={from} to={to}"
                                    );
                                    assert!(
                                        (gram - if i == j { T::o() } else { T::z() }).abs1() < tol
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn trexc_positions_similarity_and_ignored_q_all_scalar_integer_types() {
        verify::<f32, i32>(0.0, 3e-4);
        verify::<f32, i64>(0.0, 3e-4);
        verify::<f64, i32>(0.0, 3e-11);
        verify::<f64, i64>(0.0, 3e-11);
        verify::<colmat::c32, i32>(colmat::c32::new(0.0, 1.0), 3e-4);
        verify::<colmat::c32, i64>(colmat::c32::new(0.0, 1.0), 3e-4);
        verify::<colmat::c64, i32>(colmat::c64::new(0.0, 1.0), 3e-11);
        verify::<colmat::c64, i64>(colmat::c64::new(0.0, 1.0), 3e-11);
    }
    fn auxiliary<T: MoveScalar, I: LapackInt>(tol: f64) {
        unsafe {
            for n1 in [1usize, 2] {
                for n2 in [1usize, 2] {
                    for first in [0usize, 1] {
                        for want in [false, true] {
                            let n = first + n1 + n2 + 1;
                            let ld = n + 2;
                            let mut a = Mat::from_fn(n, n, |i, j| {
                                if i > j {
                                    T::z()
                                } else if i == j {
                                    T::real((i + 2) as f64)
                                } else {
                                    T::real((i + j + 1) as f64 / 13.0)
                                }
                            });
                            for (start, size, value) in [(first, n1, 10.0), (first + n1, n2, 20.0)]
                            {
                                for i in start..start + size {
                                    a[(i, i)] = T::real(value);
                                }
                                if size == 2 {
                                    a[(start, start + 1)] = T::real(3.0);
                                    a[(start + 1, start)] = T::real(-2.0);
                                }
                            }
                            let mut t = vec![T::real(777.0); ld * n];
                            let mut q = t.clone();
                            for j in 0..n {
                                for i in 0..n {
                                    t[i + j * ld] = a[(i, j)];
                                    q[i + j * ld] = if i == j { T::o() } else { T::z() };
                                }
                            }
                            let old_q = q.clone();
                            let mut info = I::from_i64(-777);
                            laexc(
                                want,
                                n as isize,
                                t.as_mut_ptr(),
                                ld as isize,
                                if want {
                                    q.as_mut_ptr()
                                } else {
                                    core::ptr::null_mut()
                                },
                                if want { ld as isize } else { 1 },
                                first as isize + 1,
                                n1 as isize,
                                n2 as isize,
                                &mut info,
                            );
                            assert_eq!(info.val(), 0);
                            assert!((t[first + first * ld] - T::real(20.0)).abs1() < tol);
                            for j in 0..n {
                                for i in n..ld {
                                    assert_eq!(t[i + j * ld], T::real(777.0));
                                    assert_eq!(q[i + j * ld], T::real(777.0));
                                }
                            }
                            if !want {
                                assert_eq!(q, old_q);
                                continue;
                            }
                            for j in 0..n {
                                for i in 0..n {
                                    let mut rebuilt = T::z();
                                    let mut gram = T::z();
                                    for k in 0..n {
                                        gram = gram + q[k + i * ld].cj() * q[k + j * ld];
                                        for l in 0..n {
                                            rebuilt = rebuilt
                                                + q[i + k * ld]
                                                    * t[k + l * ld]
                                                    * q[j + l * ld].cj();
                                        }
                                    }
                                    assert!((rebuilt - a[(i, j)]).abs1() < tol);
                                    assert!(
                                        (gram - if i == j { T::o() } else { T::z() }).abs1() < tol
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn adjacent_all_real_block_sizes_and_integer_abis() {
        auxiliary::<f32, i32>(4e-4);
        auxiliary::<f32, i64>(4e-4);
        auxiliary::<f64, i32>(4e-11);
        auxiliary::<f64, i64>(4e-11);
    }
    #[test]
    fn auxiliary_quick_returns_do_not_read_matrices() {
        unsafe {
            let mut info = 777i64;
            let nil = core::ptr::null_mut::<f64>();
            for (n, n1, n2) in [(0, 1, 2), (4, 0, 2), (4, 2, 0)] {
                laexc::<f64, i64>(true, n, nil, 0, nil, 0, -777, n1, n2, &mut info);
                assert_eq!(info, 0);
            }
        }
    }
}

#[cfg(test)]
mod rejection_tests {
    use super::*;
    fn check<T: MoveScalar, I: LapackInt>(rows: [[f64; 4]; 4]) {
        unsafe {
            for want in [false, true] {
                let ld = 6;
                let mut t = vec![T::real(777.0); ld * 4];
                let mut q = t.clone();
                for j in 0..4 {
                    for i in 0..4 {
                        t[i + j * ld] = T::real(rows[i][j]);
                        q[i + j * ld] = if i == j { T::o() } else { T::z() };
                    }
                }
                let old_t = t.clone();
                let old_q = q.clone();
                let mut info = [I::from_i64(-777); 3];
                laexc(
                    want,
                    4,
                    t.as_mut_ptr(),
                    ld as isize,
                    if want {
                        q.as_mut_ptr()
                    } else {
                        core::ptr::null_mut()
                    },
                    if want { ld as isize } else { 1 },
                    1,
                    2,
                    2,
                    info.as_mut_ptr().add(1),
                );
                assert_eq!(info[1].val(), 1);
                assert_eq!(info[0].val(), -777);
                assert_eq!(info[2].val(), -777);
                assert_eq!(
                    t.iter().map(|x| x.re().to_bits()).collect::<Vec<_>>(),
                    old_t.iter().map(|x| x.re().to_bits()).collect::<Vec<_>>()
                );
                assert_eq!(
                    q.iter().map(|x| x.re().to_bits()).collect::<Vec<_>>(),
                    old_q.iter().map(|x| x.re().to_bits()).collect::<Vec<_>>()
                );
                for from in [1, 2] {
                    for to in [3, 4] {
                        let mut first = I::from_i64(from);
                        let mut last = I::from_i64(to);
                        trexc(
                            if want { b'V' } else { b'N' },
                            4,
                            t.as_mut_ptr(),
                            ld as isize,
                            if want {
                                q.as_mut_ptr()
                            } else {
                                core::ptr::null_mut()
                            },
                            if want { ld as isize } else { 1 },
                            &mut first,
                            &mut last,
                            info.as_mut_ptr().add(1),
                        );
                        assert_eq!((info[1].val(), first.val(), last.val()), (1, 1, 1));
                        assert_eq!(
                            t.iter().map(|x| x.re().to_bits()).collect::<Vec<_>>(),
                            old_t.iter().map(|x| x.re().to_bits()).collect::<Vec<_>>()
                        );
                        assert_eq!(
                            q.iter().map(|x| x.re().to_bits()).collect::<Vec<_>>(),
                            old_q.iter().map(|x| x.re().to_bits()).collect::<Vec<_>>()
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn finite_rejected_swaps_preserve_t_q_and_return_current_position() {
        // These finite standardized windows were independently executed with
        // original OpenBLAS DLAEXC/SLAEXC: INFO=1 and bitwise unchanged T/Q.
        let real64 = [
            [
                1.0,
                7.1362384635298e+44,
                -3.7414441915671115e+50,
                -3.622271631530685e-71,
            ],
            [
                -7.24454326306137e-71,
                1.0,
                -4.7477838728798994e-66,
                -5.795634610449096e-70,
            ],
            [0.0, 0.0, 1.5, 2.535301200456459e+30],
            [0.0, 0.0, -2.8698592549372254e-42, 1.5],
        ];
        let real32 = [
            [1.0, 0.5, 8.881784e-16, 536870900.0],
            [-3.3881318e-21, 1.0, 128.0, 5.421011e-20],
            [0.0, 0.0, 1.000061, 5.684342e-14],
            [0.0, 0.0, -67108864.0, 1.000061],
        ];
        check::<f64, i32>(real64);
        check::<f64, i64>(real64);
        check::<f32, i32>(real32);
        check::<f32, i64>(real32);
    }
}

#[cfg(test)]
mod trsen_tests {
    use super::*;
    fn selected<T: MoveScalar, I: LapackInt>(imag: T, tol: f64) {
        unsafe {
            let n = 5;
            let ld = 7;
            let mut a = Mat::from_fn(n, n, |i, j| {
                if i > j {
                    T::z()
                } else if i == j {
                    T::real((i + 2) as f64) + imag * T::real(i as f64 / 7.0)
                } else {
                    T::real((i + j + 1) as f64 / 11.0) + imag * T::real((i + 2 * j) as f64 / 13.0)
                }
            });
            if !T::COMPLEX {
                a[(0, 0)] = T::real(2.0);
                a[(1, 1)] = T::real(2.0);
                a[(0, 1)] = T::real(3.0);
                a[(1, 0)] = T::real(-2.0);
                a[(3, 3)] = T::real(7.0);
                a[(4, 4)] = T::real(7.0);
                a[(3, 4)] = T::real(-4.0);
                a[(4, 3)] = T::real(1.0);
            }
            let mut layout = Vec::new();
            let mut j = 0;
            while j < n {
                let len = if !T::COMPLEX && j + 1 < n && a[(j + 1, j)] != T::z() {
                    2
                } else {
                    1
                };
                layout.push((j, len));
                j += len;
            }
            for pattern in [
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [0, -3, 0, 0, 0],
                [0, 0, 1, 0, 1],
            ] {
                for want in [false, true] {
                    for query in [false, true] {
                        let flags: Vec<I> = pattern.into_iter().map(I::from_i64).collect();
                        let old_flags: Vec<_> = flags.iter().map(|x| x.val()).collect();
                        let chosen: Vec<_> = layout
                            .iter()
                            .copied()
                            .filter(|&(j, len)| pattern[j..j + len].iter().any(|&x| x != 0))
                            .collect();
                        let others: Vec<_> = layout
                            .iter()
                            .copied()
                            .filter(|&(j, len)| pattern[j..j + len].iter().all(|&x| x == 0))
                            .collect();
                        let count: usize = chosen.iter().map(|x| x.1).sum();
                        let order: Vec<_> = chosen
                            .into_iter()
                            .chain(others)
                            .flat_map(|(j, len)| j..j + len)
                            .collect();
                        let mut t = vec![T::real(777.0); ld * n];
                        let mut q = t.clone();
                        for j in 0..n {
                            for i in 0..n {
                                t[i + j * ld] = a[(i, j)];
                                q[i + j * ld] = if i == j { T::o() } else { T::z() };
                            }
                        }
                        let old_t = t.clone();
                        let old_q = q.clone();
                        let mut work = vec![
                            T::real(777.0);
                            if query {
                                3
                            } else {
                                if T::COMPLEX {
                                    3
                                } else {
                                    n + 2
                                }
                            }
                        ];
                        let mut iwork = [I::from_i64(-777); 3];
                        let mut w = [T::real(777.0); 7];
                        let mut wi = w;
                        let mut m = [I::from_i64(-777); 3];
                        let mut info = m;
                        let minimum = if T::COMPLEX { 1 } else { n as isize };
                        assert!(trsen::<T, T, I>(
                            b'N',
                            if want { b'V' } else { b'N' },
                            flags.as_ptr(),
                            n as isize,
                            t.as_mut_ptr(),
                            ld as isize,
                            if want {
                                q.as_mut_ptr()
                            } else {
                                core::ptr::null_mut()
                            },
                            if want { ld as isize } else { 1 },
                            w.as_mut_ptr().add(1),
                            wi.as_mut_ptr().add(1),
                            m.as_mut_ptr().add(1),
                            work.as_mut_ptr().add(1),
                            if query { -1 } else { minimum },
                            if T::COMPLEX {
                                core::ptr::null_mut()
                            } else {
                                iwork.as_mut_ptr().add(1)
                            },
                            if query { -9 } else { 1 },
                            info.as_mut_ptr().add(1)
                        ));
                        assert_eq!(info[1].val(), 0);
                        assert_eq!(m[1].val(), count as isize);
                        assert_eq!(flags.iter().map(|x| x.val()).collect::<Vec<_>>(), old_flags);
                        assert_eq!(work[1], T::real(minimum as f64));
                        assert_eq!(work[0], T::real(777.0));
                        assert_eq!(*work.last().unwrap(), T::real(777.0));
                        assert_eq!(iwork[0].val(), -777);
                        assert_eq!(iwork[2].val(), -777);
                        assert_eq!(m[0].val(), -777);
                        assert_eq!(m[2].val(), -777);
                        assert_eq!(info[0].val(), -777);
                        assert_eq!(info[2].val(), -777);
                        if query {
                            assert_eq!(t, old_t);
                            assert_eq!(q, old_q);
                            assert!(w.iter().all(|&x| x == T::real(777.0)));
                            continue;
                        }
                        for j in 0..n {
                            assert!((w[j + 1] - a[(order[j], order[j])]).abs1() < tol);
                            for i in n..ld {
                                assert_eq!(t[i + j * ld], T::real(777.0));
                                assert_eq!(q[i + j * ld], T::real(777.0));
                            }
                        }
                        assert_eq!(w[0], T::real(777.0));
                        assert_eq!(w[6], T::real(777.0));
                        if !want {
                            assert_eq!(q, old_q);
                            continue;
                        }
                        for j in 0..n {
                            for i in 0..n {
                                let mut rebuilt = T::z();
                                let mut gram = T::z();
                                for k in 0..n {
                                    gram = gram + q[k + i * ld].cj() * q[k + j * ld];
                                    for l in 0..n {
                                        rebuilt = rebuilt
                                            + q[i + k * ld] * t[k + l * ld] * q[j + l * ld].cj();
                                    }
                                }
                                assert!((rebuilt - a[(i, j)]).abs1() < tol);
                                assert!((gram - if i == j { T::o() } else { T::z() }).abs1() < tol);
                            }
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn stable_native_selection_and_minimal_queries_all_types_and_abis() {
        selected::<f32, i32>(0.0, 4e-4);
        selected::<f32, i64>(0.0, 4e-4);
        selected::<f64, i32>(0.0, 4e-11);
        selected::<f64, i64>(0.0, 4e-11);
        selected::<colmat::c32, i32>(colmat::c32::new(0.0, 1.0), 4e-4);
        selected::<colmat::c32, i64>(colmat::c32::new(0.0, 1.0), 4e-4);
        selected::<colmat::c64, i32>(colmat::c64::new(0.0, 1.0), 4e-11);
        selected::<colmat::c64, i64>(colmat::c64::new(0.0, 1.0), 4e-11);
    }
    #[test]
    fn real_integer_work_query_is_not_a_collective_query() {
        unsafe {
            let flags = [1i32];
            let mut t = [2.0];
            let mut w = [777.0];
            let mut wi = [777.0];
            let mut work = [777.0];
            let mut iwork = [-777i32];
            let mut m = -777;
            let mut info = -777;
            assert!(trsen::<f64, f64, i32>(
                b'N',
                b'N',
                flags.as_ptr(),
                1,
                t.as_mut_ptr(),
                1,
                core::ptr::null_mut(),
                1,
                w.as_mut_ptr(),
                wi.as_mut_ptr(),
                &mut m,
                work.as_mut_ptr(),
                1,
                iwork.as_mut_ptr(),
                -1,
                &mut info
            ));
            assert_eq!((info, m), (-17, 1));
            assert_eq!(work, [777.0]);
            assert_eq!(iwork, [-777]);
            assert_eq!(crate::colmat_blas_last_error(), 17);
        }
    }
}

#[cfg(test)]
mod geesx_tests {
    use super::*;
    fn residual<T: Scalar>(a: &[T], t: &[T], q: &[T], ld: usize, tol: f64) {
        for j in 0..2 {
            for i in 0..2 {
                let mut rebuilt = T::z();
                let mut gram = T::z();
                for k in 0..2 {
                    gram = gram + q[k + i * ld].cj() * q[k + j * ld];
                    for l in 0..2 {
                        rebuilt = rebuilt + q[i + k * ld] * t[k + l * ld] * q[j + l * ld].cj();
                    }
                }
                assert!((rebuilt - a[i + j * ld]).abs1() < tol);
                assert!((gram - if i == j { T::o() } else { T::z() }).abs1() < tol);
            }
        }
    }
    macro_rules! real_case {
        ($t:ty,$i:ty,$entry:ident,$tol:expr) => {{
            unsafe extern "C" fn choose(re: *const $t, _im: *const $t) -> $i {
                (*re < 1.5) as $i
            }
            let mut a = [777.0 as $t; 8];
            a[0] = 2.0;
            a[1] = 0.0;
            a[4] = 0.3;
            a[5] = 1.0;
            let old = a;
            let mut q = [777.0 as $t; 8];
            let mut work = [777.0 as $t; 8];
            let mut w = [777.0 as $t; 4];
            let mut wi = w;
            let mut integer = [-777 as $i; 3];
            let mut logical = [-777 as $i; 4];
            let (mut count, mut status) = (-777 as $i, -777 as $i);
            let (mut ce, mut cv) = (111.0 as $t, 222.0 as $t);
            $entry(
                &b'V',
                &b'S',
                Some(choose),
                &b'N',
                &2,
                a.as_mut_ptr(),
                &4,
                &mut count,
                w.as_mut_ptr().add(1),
                wi.as_mut_ptr().add(1),
                q.as_mut_ptr(),
                &4,
                &mut ce,
                &mut cv,
                work.as_mut_ptr().add(1),
                &6,
                integer.as_mut_ptr().add(1),
                &1,
                logical.as_mut_ptr().add(1),
                &mut status,
            );
            assert_eq!((status, count), (0, 1));
            assert_eq!((ce, cv), (111.0, 222.0));
            assert!((w[1] - 1.0).abs() < $tol);
            assert!((w[2] - 2.0).abs() < $tol);
            assert_eq!(logical, [-777, 1, 0, -777]);
            assert_eq!(integer, [-777, 1, -777]);
            assert_eq!(work[0], 777.0);
            assert_eq!(work[7], 777.0);
            residual(&old, &a, &q, 4, $tol as f64);
            for j in 0..2 {
                for i in 2..4 {
                    assert_eq!(a[i + j * 4], 777.0);
                    assert_eq!(q[i + j * 4], 777.0);
                }
            }
            let old_a = a;
            let old_q = q;
            count = -777;
            // LIWORK=-1 alone queries both workspaces, without calling SELECT.
            $entry(
                &b'V',
                &b'S',
                None,
                &b'N',
                &2,
                a.as_mut_ptr(),
                &4,
                &mut count,
                w.as_mut_ptr().add(1),
                wi.as_mut_ptr().add(1),
                q.as_mut_ptr(),
                &4,
                &mut ce,
                &mut cv,
                work.as_mut_ptr().add(1),
                &0,
                integer.as_mut_ptr().add(1),
                &-1,
                logical.as_mut_ptr().add(1),
                &mut status,
            );
            assert_eq!((status, count), (0, -777));
            assert_eq!(a, old_a);
            assert_eq!(q, old_q);
        }};
    }
    macro_rules! complex_case {
        ($t:ty,$r:ty,$i:ty,$entry:ident,$tol:expr) => {{
            unsafe extern "C" fn choose(re: *const $t) -> $i {
                ((*re).re < 1.5) as $i
            }
            let mut a = [<$t>::real(777.0); 8];
            a[0] = <$t>::real(2.0);
            a[1] = <$t>::z();
            a[4] = <$t>::new(0.3, 0.25);
            a[5] = <$t>::real(1.0);
            let old = a;
            let mut q = [<$t>::real(777.0); 8];
            let mut work = [<$t>::real(777.0); 6];
            let mut w = [<$t>::real(777.0); 4];
            let mut rw = [777.0 as $r; 4];
            let mut logical = [-777 as $i; 4];
            let (mut count, mut status) = (-777 as $i, -777 as $i);
            let (mut ce, mut cv) = (111.0 as $r, 222.0 as $r);
            $entry(
                &b'V',
                &b'S',
                Some(choose),
                &b'N',
                &2,
                a.as_mut_ptr(),
                &4,
                &mut count,
                w.as_mut_ptr().add(1),
                q.as_mut_ptr(),
                &4,
                &mut ce,
                &mut cv,
                work.as_mut_ptr().add(1),
                &4,
                rw.as_mut_ptr().add(1),
                logical.as_mut_ptr().add(1),
                &mut status,
            );
            assert_eq!((status, count), (0, 1));
            assert_eq!((ce, cv), (111.0, 222.0));
            assert!((w[1] - <$t>::o()).abs1() < $tol);
            assert!((w[2] - <$t>::real(2.0)).abs1() < $tol);
            assert_eq!(logical, [-777, 1, 0, -777]);
            assert_eq!(work[0], <$t>::real(777.0));
            assert_eq!(work[5], <$t>::real(777.0));
            assert_eq!(rw[0], 777.0);
            assert_eq!(rw[3], 777.0);
            residual(&old, &a, &q, 4, $tol);
            for j in 0..2 {
                for i in 2..4 {
                    assert_eq!(a[i + j * 4], <$t>::real(777.0));
                    assert_eq!(q[i + j * 4], <$t>::real(777.0));
                }
            }
            let old_a = a;
            let old_q = q;
            count = -777;
            $entry(
                &b'V',
                &b'S',
                None,
                &b'N',
                &2,
                a.as_mut_ptr(),
                &4,
                &mut count,
                w.as_mut_ptr().add(1),
                q.as_mut_ptr(),
                &4,
                &mut ce,
                &mut cv,
                work.as_mut_ptr().add(1),
                &-1,
                rw.as_mut_ptr().add(1),
                logical.as_mut_ptr().add(1),
                &mut status,
            );
            assert_eq!((status, count), (0, -777));
            assert_eq!(a, old_a);
            assert_eq!(q, old_q);
        }};
    }
    #[test]
    fn native_geesx_sorting_and_queries_both_integer_abis() {
        unsafe {
            real_case!(f32, i32, sgeesx_, 4e-4);
            real_case!(f32, i64, sgeesx_64_, 4e-4);
            real_case!(f64, i32, dgeesx_, 4e-11);
            real_case!(f64, i64, dgeesx_64_, 4e-11);
            complex_case!(colmat::c32, f32, i32, cgeesx_, 4e-4);
            complex_case!(colmat::c32, f32, i64, cgeesx_64_, 4e-4);
            complex_case!(colmat::c64, f64, i32, zgeesx_, 4e-11);
            complex_case!(colmat::c64, f64, i64, zgeesx_64_, 4e-11);
        }
    }
    #[test]
    fn nested_and_thread_local_callback_guards_restore_previous_select() {
        unsafe extern "C" fn yes(_a: *const f64, _b: *const f64) -> i32 {
            -3
        }
        unsafe extern "C" fn no(_a: *const f64, _b: *const f64) -> i32 {
            0
        }
        unsafe {
            assert!(SELECT_D.with(|cell| cell.get().is_none()));
            let outer = SelectD::new(Some(yes));
            assert_eq!(select_d(&1.0, &0.0), 1);
            {
                let _inner = SelectD::new(Some(no));
                assert_eq!(select_d(&1.0, &0.0), 0);
            }
            assert_eq!(select_d(&1.0, &0.0), 1);
            std::thread::spawn(|| {
                assert!(SELECT_D.with(|cell| cell.get().is_none()));
                let _guard = SelectD::new(Some(no));
                assert_eq!(select_d(&1.0, &0.0), 0);
            })
            .join()
            .unwrap();
            assert_eq!(select_d(&1.0, &0.0), 1);
            drop(outer);
            assert!(SELECT_D.with(|cell| cell.get().is_none()));
        }
    }
}
