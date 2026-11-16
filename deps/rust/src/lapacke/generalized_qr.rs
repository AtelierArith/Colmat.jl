//! Generalized QR/RQ and constrained least-squares C storage adapters.
use super::{
    ge_nancheck, row_to_column, self_adjoint::zeros, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR,
};
use crate::kernels::Scalar;

#[derive(Clone, Copy)]
enum Kind {
    Qr,
    Rq,
    Lse,
    Glm,
}
impl Kind {
    fn extents(self, first: i64, second: i64, third: i64) -> (i64, i64, i64, i64) {
        match self {
            Self::Qr | Self::Glm => (first, second, first, third),
            Self::Rq => (first, third, second, third),
            Self::Lse => (first, second, third, second),
        }
    }
    fn pair(self) -> bool {
        matches!(self, Self::Qr | Self::Rq)
    }
    fn headers(self, f: i64, s: i64, p: i64) -> bool {
        f >= 0
            && s >= 0
            && p >= 0
            && match self {
                Self::Lse => p <= s && p >= s - f,
                Self::Glm => s <= f && p >= f - s,
                _ => true,
            }
    }
}

unsafe fn work<T: Scalar>(
    kind: Kind,
    layout: i32,
    f: i64,
    s: i64,
    p: i64,
    a: *mut T,
    la: i64,
    b: *mut T,
    lb: i64,
    lwork: i64,
    mut call: impl FnMut(*mut T, i64, *mut T, i64) -> i64,
) -> i64 {
    if !matches!(layout, ROW_MAJOR | COL_MAJOR) {
        return -1;
    }
    let (ar, ac, br, bc) = kind.extents(f, s, p);
    if layout == ROW_MAJOR {
        if la < ac {
            return -6;
        }
        if lb < bc {
            return if kind.pair() { -9 } else { -8 };
        }
    }
    if layout == COL_MAJOR {
        return call(a, la, b, lb);
    }
    let minimum = if kind.pair() {
        f.max(s).max(p).max(1) as i128
    } else if (matches!(kind, Kind::Lse) && s == 0) || (matches!(kind, Kind::Glm) && f == 0) {
        1
    } else {
        f as i128 + s as i128 + p as i128
    };
    // Header failures must reach the native boundary: generalized pair
    // routines publish their own WORK value even before header errors.
    // Queries and insufficient-WORK calls must not read numerical arrays.
    if !kind.headers(f, s, p) || lwork == -1 || (lwork as i128) < minimum {
        return call(a, ar.max(1), b, br.max(1));
    }
    let mut at = match row_to_column(ar as usize, ac as usize, a, la as usize) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let mut bt = match row_to_column(br as usize, bc as usize, b, lb as usize) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let info = call(at.as_mut_ptr(), ar.max(1), bt.as_mut_ptr(), br.max(1));
    if info >= 0 {
        for j in 0..ac as usize {
            for i in 0..ar as usize {
                *a.add(i * la as usize + j) = at[i + j * ar as usize];
            }
        }
        for j in 0..bc as usize {
            for i in 0..br as usize {
                *b.add(i * lb as usize + j) = bt[i + j * br as usize];
            }
        }
    }
    info
}
unsafe fn nan<T: Scalar>(
    kind: Kind,
    layout: i32,
    f: i64,
    s: i64,
    p: i64,
    a: *const T,
    la: i64,
    b: *const T,
    lb: i64,
    c: *const T,
    d: *const T,
) -> Result<(), i64> {
    if !matches!(layout, ROW_MAJOR | COL_MAJOR) {
        return Err(-1);
    }
    if LAPACKE_get_nancheck() == 0 {
        return Ok(());
    }
    let (ar, ac, br, bc) = kind.extents(f, s, p);
    if ge_nancheck(layout, ar, ac, a, la)? {
        return Err(-5);
    }
    if ge_nancheck(layout, br, bc, b, lb)? {
        return Err(if kind.pair() { -8 } else { -7 });
    }
    if !kind.pair() {
        if ge_nancheck(COL_MAJOR, f, 1, c, f.max(1))? {
            return Err(-9);
        }
        if matches!(kind, Kind::Lse) && ge_nancheck(COL_MAJOR, p, 1, d, p.max(1))? {
            return Err(-10);
        }
    }
    Ok(())
}
fn workspace<T: Scalar>(query: T, maximum: usize) -> Result<Vec<T>, i64> {
    let value = query.re();
    if !value.is_finite() || value < 1.0 {
        return Err(-1010);
    }
    let count = value as usize;
    if count > maximum {
        return Err(-1010);
    }
    zeros::<T>(count).map_err(|_| -1010)
}
macro_rules! pair_exports {($p:ident,$family:ident,$kind:ident,$t:ty,$j:ty,[$($suffix:tt)*],$f:tt)=>{paste::paste!{
    #[no_mangle]
    pub unsafe extern "C" fn [<LAPACKE_ $p $family _work $($suffix)*>](layout:i32,first:$j,second:$j,third:$j,a:*mut $t,lda:$j,taua:*mut $t,b:*mut $t,ldb:$j,taub:*mut $t,workspace:*mut $t,lwork:$j)->$j{
        work(Kind::$kind,layout,first as i64,second as i64,third as i64,a,lda as i64,b,ldb as i64,lwork as i64,|a,lda,b,ldb|{
            let mut info:$j=0;
            crate::generalized_qr::[<$p $family $f>](&first,&second,&third,a,&(lda as $j),taua,b,&(ldb as $j),taub,workspace,&lwork,&mut info);
            if info<0{(info-1) as i64}else{info as i64}
        }) as $j
    }
    #[no_mangle]
    pub unsafe extern "C" fn [<LAPACKE_ $p $family $($suffix)*>](layout:i32,first:$j,second:$j,third:$j,a:*mut $t,lda:$j,taua:*mut $t,b:*mut $t,ldb:$j,taub:*mut $t)->$j{
        if let Err(e)=nan(Kind::$kind,layout,first as i64,second as i64,third as i64,a,lda as i64,b,ldb as i64,core::ptr::null(),core::ptr::null()){return e as $j;}
        let mut query=<$t>::z();
        let info=[<LAPACKE_ $p $family _work $($suffix)*>](layout,first,second,third,a,lda,taua,b,ldb,taub,&mut query,-1);
        if info!=0{return info;}
        let mut w=match workspace(query,<$j>::MAX as usize){Ok(v)=>v,Err(e)=>return e as $j};
        [<LAPACKE_ $p $family _work $($suffix)*>](layout,first,second,third,a,lda,taua,b,ldb,taub,w.as_mut_ptr(),w.len() as $j)
    }
}};}
macro_rules! constrained_exports {($p:ident,$family:ident,$kind:ident,$t:ty,$j:ty,[$($suffix:tt)*],$f:tt)=>{paste::paste!{
    #[no_mangle]
    pub unsafe extern "C" fn [<LAPACKE_ $p $family _work $($suffix)*>](layout:i32,first:$j,second:$j,third:$j,a:*mut $t,lda:$j,b:*mut $t,ldb:$j,c:*mut $t,d:*mut $t,x:*mut $t,workspace:*mut $t,lwork:$j)->$j{
        work(Kind::$kind,layout,first as i64,second as i64,third as i64,a,lda as i64,b,ldb as i64,lwork as i64,|a,lda,b,ldb|{
            let mut info:$j=0;
            crate::constrained_least_squares::[<$p $family $f>](&first,&second,&third,a,&(lda as $j),b,&(ldb as $j),c,d,x,workspace,&lwork,&mut info);
            if info<0{(info-1) as i64}else{info as i64}
        }) as $j
    }
    #[no_mangle]
    pub unsafe extern "C" fn [<LAPACKE_ $p $family $($suffix)*>](layout:i32,first:$j,second:$j,third:$j,a:*mut $t,lda:$j,b:*mut $t,ldb:$j,c:*mut $t,d:*mut $t,x:*mut $t)->$j{
        if let Err(e)=nan(Kind::$kind,layout,first as i64,second as i64,third as i64,a,lda as i64,b,ldb as i64,c,d){return e as $j;}
        let mut query=<$t>::z();
        let info=[<LAPACKE_ $p $family _work $($suffix)*>](layout,first,second,third,a,lda,b,ldb,c,d,x,&mut query,-1);
        if info!=0{return info;}
        let mut w=match workspace(query,<$j>::MAX as usize){Ok(v)=>v,Err(e)=>return e as $j};
        [<LAPACKE_ $p $family _work $($suffix)*>](layout,first,second,third,a,lda,b,ldb,c,d,x,w.as_mut_ptr(),w.len() as $j)
    }
}};}
macro_rules! exports {($p:ident,$t:ty,$j:ty,[$($suffix:tt)*],$f:tt)=>{
    pair_exports!($p,ggqrf,Qr,$t,$j,[$($suffix)*],$f);
    pair_exports!($p,ggrqf,Rq,$t,$j,[$($suffix)*],$f);
    constrained_exports!($p,gglse,Lse,$t,$j,[$($suffix)*],$f);
    constrained_exports!($p,ggglm,Glm,$t,$j,[$($suffix)*],$f);
};}
exports!(s, f32, i32, [], _);
exports!(d, f64, i32, [], _);
exports!(c, colmat::c32, i32, [], _);
exports!(z, colmat::c64, i32, [], _);
exports!(s, f32, i64, [64_], _64_);
exports!(d, f64, i64, [64_], _64_);
exports!(c, colmat::c32, i64, [64_], _64_);
exports!(z, colmat::c64, i64, [64_], _64_);
