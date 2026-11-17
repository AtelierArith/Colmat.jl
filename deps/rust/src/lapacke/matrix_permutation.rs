//! LAPACKE LAPMR/LAPMT row/column permutation adapters.
//! Reference: Reference LAPACK v3.12.1 LAPACKE/src/lapacke_{s,d,c,z}
//! {lapmr,lapmt}{,_work}.c. Row-major storage is converted only over the
//! logical rectangle, so padding is never read or written.
use super::{row_to_column, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;
use crate::lapack::LapackInt;

unsafe fn apply<T: Scalar, I: crate::lapack::LapackInt>(
    layout: i32,
    forward: I,
    m: I,
    n: I,
    a: *mut T,
    lda: I,
    k: *mut I,
    rows: bool,
    mut call: impl FnMut(*mut T, I) -> I,
) -> I {
    if layout != ROW_MAJOR && layout != COL_MAJOR {
        return I::from_i64(-1);
    }
    let (m, n, ld) = (m.val(), n.val(), lda.val());
    if layout == ROW_MAJOR && ld < n {
        return I::from_i64(-6);
    }
    if m < 0 {
        return I::from_i64(-3);
    }
    if n < 0 {
        return I::from_i64(-4);
    }
    if layout == COL_MAJOR && ld < m.max(1) {
        return I::from_i64(-6);
    }
    if m == 0 || n == 0 {
        return I::from_i64(0);
    }
    if layout == COL_MAJOR {
        return call(a, lda);
    }
    let (mm, nn) = (m as usize, n as usize);
    let mut temp = match row_to_column(mm, nn, a, ld as usize) {
        Ok(v) => v,
        Err(info) => return I::from_i64(info),
    };
    let info = call(temp.as_mut_ptr(), I::from_i64(m as i64));
    if info.val() >= 0 {
        for j in 0..nn {
            for i in 0..mm {
                *a.add(i * ld as usize + j) = temp[i + j * mm];
            }
        }
    }
    let _ = rows;
    let _ = forward;
    info
}

macro_rules! export {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt,$fname:ident,$ffamily:ident,$rows:expr)=>{paste::paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $fname $($suffix)*>](
            layout:i32,forward:$i,m:$i,n:$i,a:*mut $t,lda:$i,k:*mut $i,
        )->$i {
            apply::<$t,$i>(layout,forward,m,n,a,lda,k,$rows,|a,ld| {
                let mut fwd=forward;
                let (mm,nn,ll)=(m.val() as $i,n.val() as $i,ld as $i);
                crate::matrix_permutation::[<$p $ffamily $fortran>](
                    &mut fwd,&mm,&nn,a,&ll,k,
                );
                $i::from_i64(0)
            })
        }
    }};
}

macro_rules! all {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{
        export!($p,$t,$i,[$($suffix)*],$fortran,lapmr,lapmr,true);
        export!($p,$t,$i,[$($suffix)*],$fortran,lapmr_work,lapmr,true);
        export!($p,$t,$i,[$($suffix)*],$fortran,lapmt,lapmt,false);
        export!($p,$t,$i,[$($suffix)*],$fortran,lapmt_work,lapmt,false);
    };
}
all!(s, f32, i32, [], _);
all!(d, f64, i32, [], _);
all!(c, colmat::c32, i32, [], _);
all!(z, colmat::c64, i32, [], _);
all!(s, f32, i64, [64_], _64_);
all!(d, f64, i64, [64_], _64_);
all!(c, colmat::c32, i64, [64_], _64_);
all!(z, colmat::c64, i64, [64_], _64_);
