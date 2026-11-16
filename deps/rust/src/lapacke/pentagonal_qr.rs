//! Triangular-pentagonal QR factors preserve structurally ignored storage.
use super::{ge_nancheck, self_adjoint::zeros, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;

fn factor_geometry(
    layout: i32,
    m: i64,
    n: i64,
    l: i64,
    nb: Option<i64>,
    lda: i64,
    ldb: i64,
    ldt: i64,
) -> Result<(), i64> {
    if !matches!(layout, ROW_MAJOR | COL_MAJOR) {
        return Err(-1);
    }
    let blocked = i64::from(nb.is_some());
    if layout == ROW_MAJOR {
        for (arg, ld) in [
            (-5 - 2 * blocked, lda),
            (-7 - 2 * blocked, ldb),
            (-9 - 2 * blocked, ldt),
        ] {
            if ld < n {
                return Err(arg);
            }
        }
    }
    if m < 0 {
        return Err(-2);
    }
    if n < 0 {
        return Err(-3);
    }
    if l < 0 || l > m.min(n) {
        return Err(-4);
    }
    if let Some(nb) = nb {
        if nb < 1 || (n > 0 && nb > n) {
            return Err(-5);
        }
    }
    if layout == COL_MAJOR {
        for (arg, ld, minimum) in [
            (-6 - blocked, lda, n.max(1)),
            (-8 - blocked, ldb, m.max(1)),
            (-10 - blocked, ldt, nb.unwrap_or(n.max(1))),
        ] {
            if ld < minimum {
                return Err(arg);
            }
        }
    }
    Ok(())
}

unsafe fn factor_nan<T: Scalar>(
    layout: i32,
    m: i64,
    n: i64,
    a: *const T,
    lda: i64,
    b: *const T,
    ldb: i64,
    blocked: bool,
) -> Result<(), i64> {
    if !matches!(layout, ROW_MAJOR | COL_MAJOR) {
        return Err(-1);
    }
    if LAPACKE_get_nancheck() != 0 {
        if ge_nancheck(layout, n, n, a, lda)? {
            return Err(if blocked { -6 } else { -4 });
        }
        if ge_nancheck(layout, m, n, b, ldb)? {
            return Err(if blocked { -8 } else { -6 });
        }
    }
    Ok(())
}

fn active(m: usize, l: usize, i: usize, j: usize) -> bool {
    i < m - l || i - (m - l) <= j
}

unsafe fn factor_work<T: Scalar>(
    layout: i32,
    m: i64,
    n: i64,
    l: i64,
    nb: Option<i64>,
    a: *mut T,
    lda: i64,
    b: *mut T,
    ldb: i64,
    t: *mut T,
    ldt: i64,
    mut call: impl FnMut(*mut T, i64, *mut T, i64, *mut T, i64) -> i64,
) -> i64 {
    if let Err(e) = factor_geometry(layout, m, n, l, nb, lda, ldb, ldt) {
        return e;
    }
    if layout == COL_MAJOR {
        return call(a, lda, b, ldb, t, ldt);
    }
    let rows = nb.unwrap_or(n.max(1));
    if m == 0 || n == 0 {
        return call(a, n.max(1), b, m.max(1), t, rows);
    }
    // Check caller spans independently of temporary allocation dimensions.
    for (r, c, ld) in [(n, n, lda), (m, n, ldb), (rows, n, ldt)] {
        if !crate::kernels::matrix_addressable::<T>(c as isize, r as isize, ld as isize) {
            return -1011;
        }
    }
    let allocate = |r: i64, c: i64| -> Result<Vec<T>, i64> {
        zeros::<T>((r as usize).checked_mul(c as usize).ok_or(-1011)?)
    };
    let mut at = match allocate(n, n) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let mut bt = match allocate(m, n) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let mut tt = match allocate(rows, n) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let (mu, nu, lu) = (m as usize, n as usize, l as usize);
    for j in 0..nu {
        for i in 0..=j {
            at[i + j * nu] = *a.add(i * lda as usize + j);
        }
        for i in 0..mu {
            if active(mu, lu, i, j) {
                bt[i + j * mu] = *b.add(i * ldb as usize + j);
            }
        }
    }
    let info = call(
        at.as_mut_ptr(),
        n,
        bt.as_mut_ptr(),
        m,
        tt.as_mut_ptr(),
        rows,
    );
    if info >= 0 {
        for j in 0..nu {
            for i in 0..=j {
                *a.add(i * lda as usize + j) = at[i + j * nu];
            }
            for i in 0..mu {
                if active(mu, lu, i, j) {
                    *b.add(i * ldb as usize + j) = bt[i + j * mu];
                }
            }
            // Each block's T is upper triangular, including a partial last block.
            for i in 0..=j % (rows as usize) {
                *t.add(i * ldt as usize + j) = tt[i + j * rows as usize];
            }
        }
    }
    info
}

macro_rules! exports {($p:ident,$t:ty,$j:ty,[$($suffix:tt)*],$f:tt)=>{paste::paste!{
    #[no_mangle]
    pub unsafe extern "C" fn [<LAPACKE_ $p tpqrt_work $($suffix)*>](layout:i32,m:$j,n:$j,l:$j,nb:$j,a:*mut $t,lda:$j,b:*mut $t,ldb:$j,t:*mut $t,ldt:$j,workspace:*mut $t)->$j {
        factor_work(layout,m as i64,n as i64,l as i64,Some(nb as i64),a,lda as i64,b,ldb as i64,t,ldt as i64,|a,lda,b,ldb,t,ldt|{let mut info:$j=0;crate::pentagonal_driver::[<$p tpqrt $f>](&m,&n,&l,&nb,a,&(lda as $j),b,&(ldb as $j),t,&(ldt as $j),workspace,&mut info);if info<0 {(info-1) as i64}else {info as i64}}) as $j
    }
    #[no_mangle]
    pub unsafe extern "C" fn [<LAPACKE_ $p tpqrt $($suffix)*>](layout:i32,m:$j,n:$j,l:$j,nb:$j,a:*mut $t,lda:$j,b:*mut $t,ldb:$j,t:*mut $t,ldt:$j)->$j {
        if let Err(e)=factor_nan(layout,m as i64,n as i64,a,lda as i64,b,ldb as i64,true) {return e as $j;}
        if let Err(e)=factor_geometry(layout,m as i64,n as i64,l as i64,Some(nb as i64),lda as i64,ldb as i64,ldt as i64) {return e as $j;}
        let length=match (nb.max(1) as usize).checked_mul(n.max(1) as usize) {Some(v)=>v,None=>return -1010};
        let mut workspace=match zeros::<$t>(length) {Ok(v)=>v,Err(_)=>return -1010};
        [<LAPACKE_ $p tpqrt_work $($suffix)*>](layout,m,n,l,nb,a,lda,b,ldb,t,ldt,workspace.as_mut_ptr())
    }
    #[no_mangle]
    pub unsafe extern "C" fn [<LAPACKE_ $p tpqrt2_work $($suffix)*>](layout:i32,m:$j,n:$j,l:$j,a:*mut $t,lda:$j,b:*mut $t,ldb:$j,t:*mut $t,ldt:$j)->$j {
        factor_work(layout,m as i64,n as i64,l as i64,None,a,lda as i64,b,ldb as i64,t,ldt as i64,|a,lda,b,ldb,t,ldt|{let mut info:$j=0;crate::pentagonal_driver::[<$p tpqrt2 $f>](&m,&n,&l,a,&(lda as $j),b,&(ldb as $j),t,&(ldt as $j),&mut info);if info<0 {(info-1) as i64}else {info as i64}}) as $j
    }
    #[no_mangle]
    pub unsafe extern "C" fn [<LAPACKE_ $p tpqrt2 $($suffix)*>](layout:i32,m:$j,n:$j,l:$j,a:*mut $t,lda:$j,b:*mut $t,ldb:$j,t:*mut $t,ldt:$j)->$j {
        if let Err(e)=factor_nan(layout,m as i64,n as i64,a,lda as i64,b,ldb as i64,false) {return e as $j;}
        [<LAPACKE_ $p tpqrt2_work $($suffix)*>](layout,m,n,l,a,lda,b,ldb,t,ldt)
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

#[derive(Clone, Copy)]
enum Application {
    Q { nb: i64 },
    Block { direct: u8, store: u8, ldwork: i64 },
}
impl Application {
    fn direct(self) -> u8 {
        match self {
            Self::Q { .. } => b'F',
            Self::Block { direct, .. } => direct.to_ascii_uppercase(),
        }
    }
    fn store(self) -> u8 {
        match self {
            Self::Q { .. } => b'C',
            Self::Block { store, .. } => store.to_ascii_uppercase(),
        }
    }
    fn rows_t(self, k: i64) -> i64 {
        match self {
            Self::Q { nb } => nb,
            Self::Block { .. } => k,
        }
    }
}

fn application_geometry(
    layout: i32,
    side: u8,
    trans: u8,
    m: i64,
    n: i64,
    k: i64,
    l: i64,
    mode: Application,
    ldv: i64,
    ldt: i64,
    lda: i64,
    ldb: i64,
    complex: bool,
) -> Result<bool, i64> {
    if !matches!(layout, ROW_MAJOR | COL_MAJOR) {
        return Err(-1);
    }
    let side = side.to_ascii_uppercase();
    let trans = trans.to_ascii_uppercase();
    let left = side == b'L';
    let tail = if left { m } else { n };
    let block = matches!(mode, Application::Block { .. });
    if layout == ROW_MAJOR {
        if !block && !matches!(side, b'L' | b'R') {
            return Err(-2);
        }
        let ac = if left { n } else { k };
        let vc = if mode.store() == b'R' { tail } else { k };
        let shift = i64::from(block);
        for (arg, ld, minimum) in [
            (-14 - shift, lda, ac),
            (-16 - shift, ldb, n),
            (-12 - shift, ldt, k),
            (-10 - shift, ldv, vc),
        ] {
            if ld < minimum {
                return Err(arg);
            }
        }
    }
    let valid_flags = matches!(side, b'L' | b'R')
        && (trans == b'N' || trans == if complex { b'C' } else { b'T' });
    if block {
        let Application::Block { ldwork, .. } = mode else {
            unreachable!()
        };
        // The underlying Fortran subroutine is void. Invalid numerical headers
        // still return C INFO=0; dispatch them without packing or array reads.
        return Ok(valid_flags
            && matches!(mode.direct(), b'F' | b'B')
            && matches!(mode.store(), b'C' | b'R')
            && m >= 0
            && n >= 0
            && k >= 0
            && l >= 0
            && l <= k.min(tail)
            && ldwork >= if left { k } else { m });
    }
    if !matches!(side, b'L' | b'R') {
        return Err(-2);
    }
    if trans != b'N' && trans != if complex { b'C' } else { b'T' } {
        return Err(-3);
    }
    for (arg, valid) in [
        (-4, m >= 0),
        (-5, n >= 0),
        (-6, k >= 0),
        (-7, l >= 0 && l <= k.min(tail)),
    ] {
        if !valid {
            return Err(arg);
        }
    }
    let nb = mode.rows_t(k);
    if nb < 1 || (k > 0 && nb > k) {
        return Err(-8);
    }
    if layout == COL_MAJOR {
        for (arg, ld, minimum) in [
            (-10, ldv, tail.max(1)),
            (-12, ldt, nb),
            (-14, lda, if left { k.max(1) } else { m.max(1) }),
            (-16, ldb, m.max(1)),
        ] {
            if ld < minimum {
                return Err(arg);
            }
        }
    }
    Ok(true)
}

unsafe fn application_nan<T: Scalar>(
    layout: i32,
    side: u8,
    m: i64,
    n: i64,
    k: i64,
    mode: Application,
    v: *const T,
    ldv: i64,
    t: *const T,
    ldt: i64,
    a: *const T,
    lda: i64,
    b: *const T,
    ldb: i64,
) -> Result<(), i64> {
    if !matches!(layout, ROW_MAJOR | COL_MAJOR) {
        return Err(-1);
    }
    if LAPACKE_get_nancheck() == 0 {
        return Ok(());
    }
    let side = side.to_ascii_uppercase();
    let (ar, ac, tail) = match side {
        b'L' => (k, n, m),
        b'R' => (m, k, n),
        _ => (0, 0, 0),
    };
    let (vr, vc) = match mode.store() {
        b'C' => (tail, k),
        b'R' => (k, tail),
        _ => (0, 0),
    };
    let shift = i64::from(matches!(mode, Application::Block { .. }));
    for (arg, rows, cols, pointer, ld) in [
        (-13 - shift, ar, ac, a, lda),
        (-15 - shift, m, n, b, ldb),
        (-11 - shift, mode.rows_t(k), k, t, ldt),
        (-9 - shift, vr, vc, v, ldv),
    ] {
        if ge_nancheck(layout, rows, cols, pointer, ld)? {
            return Err(arg);
        }
    }
    Ok(())
}

unsafe fn application_work<T: Scalar>(
    layout: i32,
    side: u8,
    trans: u8,
    m: i64,
    n: i64,
    k: i64,
    l: i64,
    mode: Application,
    v: *const T,
    ldv: i64,
    t: *const T,
    ldt: i64,
    a: *mut T,
    lda: i64,
    b: *mut T,
    ldb: i64,
    mut call: impl FnMut(*const T, i64, *const T, i64, *mut T, i64, *mut T, i64) -> i64,
) -> i64 {
    let valid = match application_geometry(
        layout,
        side,
        trans,
        m,
        n,
        k,
        l,
        mode,
        ldv,
        ldt,
        lda,
        ldb,
        T::COMPLEX,
    ) {
        Ok(v) => v,
        Err(e) => return e,
    };
    if layout == COL_MAJOR {
        return call(v, ldv, t, ldt, a, lda, b, ldb);
    }
    let left = side.to_ascii_uppercase() == b'L';
    let tail = if left { m } else { n };
    let (vr, vc) = if mode.store() == b'R' {
        (k, tail)
    } else {
        (tail, k)
    };
    let (ar, ac) = if left { (k, n) } else { (m, k) };
    let tr = mode.rows_t(k);
    if !valid || m == 0 || n == 0 || k == 0 {
        return call(v, vr.max(1), t, tr.max(1), a, ar.max(1), b, m.max(1));
    }
    for (rows, cols, ld) in [(vr, vc, ldv), (tr, k, ldt), (ar, ac, lda), (m, n, ldb)] {
        if !crate::kernels::matrix_addressable::<T>(cols as isize, rows as isize, ld as isize) {
            return -1011;
        }
    }
    let allocate = |r: i64, c: i64| -> Result<Vec<T>, i64> {
        zeros::<T>((r as usize).checked_mul(c as usize).ok_or(-1011)?)
    };
    let mut vt = match allocate(vr, vc) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let mut tt = match allocate(tr, k) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let mut at = match super::row_to_column(ar as usize, ac as usize, a, lda as usize) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let mut bt = match super::row_to_column(m as usize, n as usize, b, ldb as usize) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let forward = mode.direct() == b'F';
    for j in 0..k as usize {
        for i in 0..tail as usize {
            let used = if forward {
                active(tail as usize, l as usize, i, j)
            } else {
                i >= l as usize || j <= k as usize - l as usize + i
            };
            if used {
                let (r, c) = if mode.store() == b'R' { (j, i) } else { (i, j) };
                vt[r + c * vr as usize] = *v.add(r * ldv as usize + c);
            }
        }
        let local = match mode {
            Application::Q { nb } => j % nb as usize,
            Application::Block { .. } => j,
        };
        for i in 0..tr as usize {
            if if forward { i <= local } else { i >= local } {
                tt[i + j * tr as usize] = *t.add(i * ldt as usize + j);
            }
        }
    }
    let info = call(
        vt.as_ptr(),
        vr,
        tt.as_ptr(),
        tr,
        at.as_mut_ptr(),
        ar,
        bt.as_mut_ptr(),
        m,
    );
    if info >= 0 {
        for j in 0..ac as usize {
            for i in 0..ar as usize {
                *a.add(i * lda as usize + j) = at[i + j * ar as usize];
            }
        }
        for j in 0..n as usize {
            for i in 0..m as usize {
                *b.add(i * ldb as usize + j) = bt[i + j * m as usize];
            }
        }
    }
    info
}

macro_rules! application_exports {($p:ident,$t:ty,$j:ty,[$($suffix:tt)*],$f:tt)=>{paste::paste!{
    #[no_mangle]
    pub unsafe extern "C" fn [<LAPACKE_ $p tpmqrt_work $($suffix)*>](layout:i32,side:i8,trans:i8,m:$j,n:$j,k:$j,l:$j,nb:$j,v:*const $t,ldv:$j,t:*const $t,ldt:$j,a:*mut $t,lda:$j,b:*mut $t,ldb:$j,workspace:*mut $t)->$j {
        application_work(layout,side as u8,trans as u8,m as i64,n as i64,k as i64,l as i64,Application::Q{nb:nb as i64},v,ldv as i64,t,ldt as i64,a,lda as i64,b,ldb as i64,|v,ldv,t,ldt,a,lda,b,ldb|{let mut info:$j=0;crate::pentagonal_driver::[<$p tpmqrt $f>](&(side as u8),&(trans as u8),&m,&n,&k,&l,&nb,v,&(ldv as $j),t,&(ldt as $j),a,&(lda as $j),b,&(ldb as $j),workspace,&mut info);if info<0{(info-1) as i64}else{info as i64}}) as $j
    }
    #[no_mangle]
    pub unsafe extern "C" fn [<LAPACKE_ $p tpmqrt $($suffix)*>](layout:i32,side:i8,trans:i8,m:$j,n:$j,k:$j,l:$j,nb:$j,v:*const $t,ldv:$j,t:*const $t,ldt:$j,a:*mut $t,lda:$j,b:*mut $t,ldb:$j)->$j {
        let mode=Application::Q{nb:nb as i64};
        if let Err(e)=application_nan(layout,side as u8,m as i64,n as i64,k as i64,mode,v,ldv as i64,t,ldt as i64,a,lda as i64,b,ldb as i64){return e as $j;}
        if let Err(e)=application_geometry(layout,side as u8,trans as u8,m as i64,n as i64,k as i64,l as i64,mode,ldv as i64,ldt as i64,lda as i64,ldb as i64,<$t>::COMPLEX){return e as $j;}
        let nw=if (side as u8).to_ascii_uppercase()==b'L'{n}else{m};
        let count=match (nw.max(1) as usize).checked_mul(nb.max(1) as usize){Some(v)=>v,None=>return -1010};
        let mut workspace=match zeros::<$t>(count){Ok(v)=>v,Err(_)=>return -1010};
        [<LAPACKE_ $p tpmqrt_work $($suffix)*>](layout,side,trans,m,n,k,l,nb,v,ldv,t,ldt,a,lda,b,ldb,workspace.as_mut_ptr())
    }
    #[no_mangle]
    pub unsafe extern "C" fn [<LAPACKE_ $p tprfb_work $($suffix)*>](layout:i32,side:i8,trans:i8,direct:i8,store:i8,m:$j,n:$j,k:$j,l:$j,v:*const $t,ldv:$j,t:*const $t,ldt:$j,a:*mut $t,lda:$j,b:*mut $t,ldb:$j,workspace:*mut $t,ldwork:$j)->$j {
        application_work(layout,side as u8,trans as u8,m as i64,n as i64,k as i64,l as i64,Application::Block{direct:direct as u8,store:store as u8,ldwork:ldwork as i64},v,ldv as i64,t,ldt as i64,a,lda as i64,b,ldb as i64,|v,ldv,t,ldt,a,lda,b,ldb|{crate::pentagonal_driver::[<$p tprfb $f>](&(side as u8),&(trans as u8),&(direct as u8),&(store as u8),&m,&n,&k,&l,v,&(ldv as $j),t,&(ldt as $j),a,&(lda as $j),b,&(ldb as $j),workspace,&ldwork);0}) as $j
    }
    #[no_mangle]
    pub unsafe extern "C" fn [<LAPACKE_ $p tprfb $($suffix)*>](layout:i32,side:i8,trans:i8,direct:i8,store:i8,m:$j,n:$j,k:$j,l:$j,v:*const $t,ldv:$j,t:*const $t,ldt:$j,a:*mut $t,lda:$j,b:*mut $t,ldb:$j)->$j {
        let left=(side as u8).to_ascii_uppercase()==b'L';let ldwork=if left{k}else{m};let cols=if left{n}else{k};
        let mode=Application::Block{direct:direct as u8,store:store as u8,ldwork:ldwork as i64};
        if let Err(e)=application_nan(layout,side as u8,m as i64,n as i64,k as i64,mode,v,ldv as i64,t,ldt as i64,a,lda as i64,b,ldb as i64){return e as $j;}
        if let Err(e)=application_geometry(layout,side as u8,trans as u8,m as i64,n as i64,k as i64,l as i64,mode,ldv as i64,ldt as i64,lda as i64,ldb as i64,<$t>::COMPLEX){return e as $j;}
        let count=match (ldwork.max(1) as usize).checked_mul(cols.max(1) as usize){Some(v)=>v,None=>return -1010};
        let mut workspace=match zeros::<$t>(count){Ok(v)=>v,Err(_)=>return -1010};
        [<LAPACKE_ $p tprfb_work $($suffix)*>](layout,side,trans,direct,store,m,n,k,l,v,ldv,t,ldt,a,lda,b,ldb,workspace.as_mut_ptr(),ldwork)
    }
}};}
application_exports!(s, f32, i32, [], _);
application_exports!(d, f64, i32, [], _);
application_exports!(c, colmat::c32, i32, [], _);
application_exports!(z, colmat::c64, i32, [], _);
application_exports!(s, f32, i64, [64_], _64_);
application_exports!(d, f64, i64, [64_], _64_);
application_exports!(c, colmat::c32, i64, [64_], _64_);
application_exports!(z, colmat::c64, i64, [64_], _64_);
