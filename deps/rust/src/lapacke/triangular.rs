//! Triangular solve/inverse C adapters; unit diagonal storage is never read.
use super::{
    ge_nancheck, self_adjoint::zeros, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR,
    TRANSPOSE_MEMORY_ERROR,
};
use crate::kernels::Scalar;

fn selected(u: u8, d: u8, i: usize, j: usize) -> bool {
    (if u == b'L' { i >= j } else { i <= j }) && (d != b'U' || i != j)
}
fn span<T>(rows: usize, cols: usize, ld: usize) -> Result<(), i64> {
    if rows == 0 || cols == 0 {
        return Ok(());
    }
    let bytes = (rows - 1)
        .checked_mul(ld)
        .and_then(|v| v.checked_add(cols))
        .and_then(|v| v.checked_mul(core::mem::size_of::<T>()));
    if bytes.is_none_or(|v| v > isize::MAX as usize) {
        Err(TRANSPOSE_MEMORY_ERROR)
    } else {
        Ok(())
    }
}
pub(super) unsafe fn nancheck<T: Scalar>(
    layout: i32,
    u: u8,
    d: u8,
    n: i64,
    a: *const T,
    ld: i64,
) -> Result<bool, i64> {
    let (u, d) = (u.to_ascii_uppercase(), d.to_ascii_uppercase());
    if n <= 0 || ld <= 0 || a.is_null() || !matches!(u, b'L' | b'U') {
        return Ok(false);
    }
    let n = n as usize;
    let ld = ld as usize;
    span::<T>(n, n.min(ld), ld)?;
    for outer in 0..n {
        for inner in 0..n.min(ld) {
            let (i, j) = if layout == ROW_MAJOR {
                (outer, inner)
            } else {
                (inner, outer)
            };
            if selected(u, d, i, j) {
                let v = *a.add(outer * ld + inner);
                if v != v {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}
pub(super) unsafe fn pack<T: Scalar>(
    u: u8,
    d: u8,
    n: usize,
    a: *const T,
    ld: usize,
) -> Result<Vec<T>, i64> {
    span::<T>(n, n, ld)?;
    let mut out = zeros::<T>(n.checked_mul(n).ok_or(TRANSPOSE_MEMORY_ERROR)?)?;
    for i in 0..n {
        for j in 0..n {
            if selected(u, d, i, j) {
                out[i + j * n] = *a.add(i * ld + j);
            } else if i == j && d == b'U' {
                out[i + j * n] = T::o();
            }
        }
    }
    Ok(out)
}
unsafe fn inverse<T: Scalar>(
    layout: i32,
    u: u8,
    d: u8,
    n: i64,
    a: *mut T,
    ld: i64,
    mut call: impl FnMut(*mut T, i64) -> i64,
) -> i64 {
    if !matches!(layout, ROW_MAJOR | COL_MAJOR) {
        return -1;
    }
    let (u, d) = (u.to_ascii_uppercase(), d.to_ascii_uppercase());
    if layout == ROW_MAJOR && ld < n {
        return -6;
    }
    if !matches!(u, b'L' | b'U') {
        return -2;
    }
    if !matches!(d, b'N' | b'U') {
        return -3;
    }
    if n < 0 {
        return -4;
    }
    if layout == COL_MAJOR {
        if ld < n.max(1) {
            return -6;
        }
        return call(a, ld);
    }
    if n == 0 {
        return call(a, 1);
    }
    let nn = n as usize;
    let mut temp = match pack(u, d, nn, a, ld as usize) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let info = call(temp.as_mut_ptr(), n);
    if info >= 0 {
        for i in 0..nn {
            for j in 0..nn {
                if selected(u, d, i, j) {
                    *a.add(i * ld as usize + j) = temp[i + j * nn];
                }
            }
        }
    }
    info
}
unsafe fn solve<T: Scalar>(
    layout: i32,
    u: u8,
    t: u8,
    d: u8,
    n: i64,
    k: i64,
    a: *const T,
    la: i64,
    b: *mut T,
    lb: i64,
    mut call: impl FnMut(*const T, i64, *mut T, i64) -> i64,
) -> i64 {
    if !matches!(layout, ROW_MAJOR | COL_MAJOR) {
        return -1;
    }
    let (u, t, d) = (
        u.to_ascii_uppercase(),
        t.to_ascii_uppercase(),
        d.to_ascii_uppercase(),
    );
    if layout == ROW_MAJOR {
        if la < n {
            return -8;
        }
        if lb < k {
            return -10;
        }
    }
    if !matches!(u, b'L' | b'U') {
        return -2;
    }
    if !matches!(t, b'N' | b'T' | b'C') {
        return -3;
    }
    if !matches!(d, b'N' | b'U') {
        return -4;
    }
    if n < 0 {
        return -5;
    }
    if k < 0 {
        return -6;
    }
    if layout == COL_MAJOR {
        if la < n.max(1) {
            return -8;
        }
        if lb < n.max(1) {
            return -10;
        }
        return call(a, la, b, lb);
    }
    if n == 0 {
        return call(a, 1, b, 1);
    }
    let (nn, kk) = (n as usize, k as usize);
    let temp = match pack(u, d, nn, a, la as usize) {
        Ok(v) => v,
        Err(e) => return e,
    };
    if let Err(e) = span::<T>(nn, kk, lb as usize) {
        return e;
    }
    let mut rhs = match nn
        .checked_mul(kk)
        .ok_or(TRANSPOSE_MEMORY_ERROR)
        .and_then(zeros::<T>)
    {
        Ok(v) => v,
        Err(e) => return e,
    };
    for i in 0..nn {
        for j in 0..kk {
            rhs[i + j * nn] = *b.add(i * lb as usize + j);
        }
    }
    let info = call(temp.as_ptr(), n, rhs.as_mut_ptr(), n);
    if info >= 0 {
        for i in 0..nn {
            for j in 0..kk {
                *b.add(i * lb as usize + j) = rhs[i + j * nn];
            }
        }
    }
    info
}
macro_rules! exports {
    ($p:ident,$t:ty,$j:ty,[$($suffix:tt)*],$f:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p trtri_work $($suffix)*>](layout:i32,u:u8,d:u8,n:$j,a:*mut $t,ld:$j)->$j {
            inverse(layout,u,d,n as i64,a,ld as i64,|a,ld|{let mut info:$j=0;crate::triangular_lapack::[<$p trtri $f>](&u,&d,&n,a,&(ld as $j),&mut info);if info<0{(info-1) as i64}else{info as i64}}) as $j
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p trtri $($suffix)*>](layout:i32,u:u8,d:u8,n:$j,a:*mut $t,ld:$j)->$j {
            if !matches!(layout,ROW_MAJOR|COL_MAJOR){return -1;}
            if LAPACKE_get_nancheck()!=0 {match nancheck(layout,u,d,n as i64,a,ld as i64){Ok(true)=>return -5,Err(e)=>return e as $j,Ok(false)=>{}}}
            [<LAPACKE_ $p trtri_work $($suffix)*>](layout,u,d,n,a,ld)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p trtrs_work $($suffix)*>](layout:i32,u:u8,t:u8,d:u8,n:$j,k:$j,a:*const $t,la:$j,b:*mut $t,lb:$j)->$j {
            solve(layout,u,t,d,n as i64,k as i64,a,la as i64,b,lb as i64,|a,la,b,lb|{let mut info:$j=0;crate::triangular_lapack::[<$p trtrs $f>](&u,&t,&d,&n,&k,a,&(la as $j),b,&(lb as $j),&mut info);if info<0{(info-1) as i64}else{info as i64}}) as $j
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p trtrs $($suffix)*>](layout:i32,u:u8,t:u8,d:u8,n:$j,k:$j,a:*const $t,la:$j,b:*mut $t,lb:$j)->$j {
            if !matches!(layout,ROW_MAJOR|COL_MAJOR){return -1;}
            if LAPACKE_get_nancheck()!=0 {
                match nancheck(layout,u,d,n as i64,a,la as i64){Ok(true)=>return -7,Err(e)=>return e as $j,Ok(false)=>{}}
                match ge_nancheck(layout,n as i64,k as i64,b,lb as i64){Ok(true)=>return -9,Err(e)=>return e as $j,Ok(false)=>{}}
            }
            [<LAPACKE_ $p trtrs_work $($suffix)*>](layout,u,t,d,n,k,a,la,b,lb)
        }
    }};
}
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
    fn numerical<T: Scalar>(
        make: impl Fn(f64, f64) -> T,
        mut inv: impl FnMut(bool, i32, u8, u8, i64, *mut T, i64) -> i64,
        mut trs: impl FnMut(bool, i32, u8, u8, u8, i64, i64, *const T, i64, *mut T, i64) -> i64,
        tol: f64,
    ) {
        for layout in [ROW_MAJOR, COL_MAJOR] {
            for high in [false, true] {
                let mut a = vec![T::z(); 15];
                let ld = 5;
                for i in 0..3 {
                    a[i * (ld + 1)] = T::real(if i == 1 { 0. } else { 2. });
                }
                let saved = a.clone();
                assert_eq!(inv(high, layout, b'L', b'N', 3, a.as_mut_ptr(), 5), 2);
                assert!(a == saved);
                for k in [0, 2] {
                    let mut b = vec![T::real(777.); 15];
                    let old = b.clone();
                    assert_eq!(
                        trs(
                            high,
                            layout,
                            b'U',
                            b'N',
                            b'N',
                            3,
                            k,
                            a.as_ptr(),
                            5,
                            b.as_mut_ptr(),
                            5
                        ),
                        2
                    );
                    assert!(a == saved && b == old);
                }
            }
        }
        let null = core::ptr::null_mut();
        assert_eq!(inv(false, 0, b'L', b'N', 0, null, 1), -1);
        assert_eq!(inv(false, COL_MAJOR, b'?', b'N', 0, null, 1), -2);
        assert_eq!(inv(false, COL_MAJOR, b'L', b'?', 0, null, 1), -3);
        assert_eq!(inv(false, COL_MAJOR, b'L', b'N', -1, null, 1), -4);
        assert_eq!(inv(false, ROW_MAJOR, b'?', b'?', 3, null, 2), -6);
        assert_eq!(inv(false, COL_MAJOR, b'L', b'N', 3, null, 2), -6);
        assert_eq!(trs(false, 0, b'L', b'N', b'N', 0, 0, null, 1, null, 1), -1);
        assert_eq!(
            trs(false, COL_MAJOR, b'?', b'N', b'N', 0, 0, null, 1, null, 1),
            -2
        );
        assert_eq!(
            trs(false, COL_MAJOR, b'L', b'?', b'N', 0, 0, null, 1, null, 1),
            -3
        );
        assert_eq!(
            trs(false, COL_MAJOR, b'L', b'N', b'?', 0, 0, null, 1, null, 1),
            -4
        );
        assert_eq!(
            trs(false, COL_MAJOR, b'L', b'N', b'N', -1, 0, null, 1, null, 1),
            -5
        );
        assert_eq!(
            trs(false, COL_MAJOR, b'L', b'N', b'N', 0, -1, null, 1, null, 1),
            -6
        );
        assert_eq!(
            trs(false, ROW_MAJOR, b'?', b'?', b'?', 3, 2, null, 2, null, 1),
            -8
        );
        assert_eq!(
            trs(false, ROW_MAJOR, b'?', b'?', b'?', 3, 2, null, 3, null, 1),
            -10
        );
        assert_eq!(
            trs(false, COL_MAJOR, b'L', b'N', b'N', 3, 2, null, 2, null, 3),
            -8
        );
        assert_eq!(
            trs(false, COL_MAJOR, b'L', b'N', b'N', 3, 2, null, 3, null, 2),
            -10
        );
        for layout in [ROW_MAJOR, COL_MAJOR] {
            for n in [0usize, 1, 3] {
                for u in [b'l', b'u'] {
                    for d in [b'n', b'u'] {
                        for high in [false, true] {
                            let ld = n + 2;
                            let unit = d == b'u';
                            let upper = u == b'u';
                            let at = |i: usize, j: usize| {
                                if layout == ROW_MAJOR {
                                    i * ld + j
                                } else {
                                    i + j * ld
                                }
                            };
                            let mut a = vec![T::real(f64::NAN); ld * n.max(1)];
                            let mut matrix = vec![T::z(); n * n];
                            for i in 0..n {
                                for j in 0..n {
                                    if (upper && i <= j) || (!upper && i >= j) {
                                        let value = if i == j {
                                            T::real(if unit { 1. } else { (i + 2) as f64 })
                                        } else {
                                            make(
                                                (i + j + 1) as f64 * 0.1,
                                                (i as f64 - j as f64) * 0.05,
                                            )
                                        };
                                        matrix[i + j * n] = value;
                                        if !unit || i != j {
                                            a[at(i, j)] = value;
                                        }
                                    }
                                }
                            }
                            let saved = a.clone();
                            assert_eq!(
                                inv(high, layout, u, d, n as i64, a.as_mut_ptr(), ld as i64),
                                0
                            );
                            for i in 0..n {
                                for j in 0..n {
                                    let product = (0..n).fold(T::z(), |sum, k| {
                                        let value = if unit && k == j {
                                            T::o()
                                        } else if (upper && k <= j) || (!upper && k >= j) {
                                            a[at(k, j)]
                                        } else {
                                            T::z()
                                        };
                                        sum + matrix[i + k * n] * value
                                    });
                                    assert!(
                                        (product - if i == j { T::o() } else { T::z() })
                                            .magnitude()
                                            < tol
                                    );
                                }
                            }
                            for k in 0..a.len() {
                                let (i, j) = if layout == ROW_MAJOR {
                                    (k / ld, k % ld)
                                } else {
                                    (k % ld, k / ld)
                                };
                                if i >= n
                                    || j >= n
                                    || !selected(
                                        u.to_ascii_uppercase(),
                                        d.to_ascii_uppercase(),
                                        i,
                                        j,
                                    )
                                {
                                    assert!(a[k] != a[k]);
                                }
                            }
                            for t in [b'n', b't', b'c'] {
                                for nrhs in [0usize, 2] {
                                    let lb = if layout == ROW_MAJOR { nrhs + 2 } else { n + 2 };
                                    let bt = |i: usize, j: usize| {
                                        if layout == ROW_MAJOR {
                                            i * lb + j
                                        } else {
                                            i + j * lb
                                        }
                                    };
                                    let size = if layout == ROW_MAJOR {
                                        n.max(1) * lb
                                    } else {
                                        nrhs.max(1) * lb
                                    };
                                    let mut b = vec![T::real(f64::NAN); size];
                                    let solution = |i: usize, j: usize| {
                                        make((i + j + 1) as f64 * 0.3, (i + 1) as f64 * 0.1)
                                    };
                                    for i in 0..n {
                                        for j in 0..nrhs {
                                            b[bt(i, j)] = (0..n).fold(T::z(), |sum, k| {
                                                let value = if t == b'n' {
                                                    matrix[i + k * n]
                                                } else {
                                                    let v = matrix[k + i * n];
                                                    if t == b'c' {
                                                        v.cj()
                                                    } else {
                                                        v
                                                    }
                                                };
                                                sum + value * solution(k, j)
                                            });
                                        }
                                    }
                                    assert_eq!(
                                        trs(
                                            high,
                                            layout,
                                            u,
                                            t,
                                            d,
                                            n as i64,
                                            nrhs as i64,
                                            saved.as_ptr(),
                                            ld as i64,
                                            b.as_mut_ptr(),
                                            lb as i64
                                        ),
                                        0
                                    );
                                    for i in 0..n {
                                        for j in 0..nrhs {
                                            assert!(
                                                (b[bt(i, j)] - solution(i, j)).magnitude() < tol
                                            );
                                        }
                                    }
                                    for k in 0..b.len() {
                                        let (i, j) = if layout == ROW_MAJOR {
                                            (k / lb, k % lb)
                                        } else {
                                            (k % lb, k / lb)
                                        };
                                        if i >= n || j >= nrhs {
                                            assert!(b[k] != b[k]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    macro_rules! case {
        ($name:ident,$p:ident,$t:ty,$j:ty,[$($suffix:tt)*],$make:expr,$tol:expr)=>{
            #[test]fn $name(){paste::paste!{numerical::<$t>($make,
                |high,layout,u,d,n,a,ld|unsafe{if high{[<LAPACKE_ $p trtri $($suffix)*>](layout,u,d,n as $j,a,ld as $j) as i64}else{[<LAPACKE_ $p trtri_work $($suffix)*>](layout,u,d,n as $j,a,ld as $j) as i64}},
                |high,layout,u,t,d,n,k,a,la,b,lb|unsafe{if high{[<LAPACKE_ $p trtrs $($suffix)*>](layout,u,t,d,n as $j,k as $j,a,la as $j,b,lb as $j) as i64}else{[<LAPACKE_ $p trtrs_work $($suffix)*>](layout,u,t,d,n as $j,k as $j,a,la as $j,b,lb as $j) as i64}},$tol);}}
        }
    }
    case!(s32, s, f32, i32, [], |r, _| r as f32, 2e-5);
    case!(s64, s, f32, i64, [64_], |r, _| r as f32, 2e-5);
    case!(d32, d, f64, i32, [], |r, _| r, 1e-12);
    case!(d64, d, f64, i64, [64_], |r, _| r, 1e-12);
    case!(
        c32,
        c,
        colmat::c32,
        i32,
        [],
        |r, i| colmat::c32::new(r as f32, i as f32),
        2e-5
    );
    case!(
        c64,
        c,
        colmat::c32,
        i64,
        [64_],
        |r, i| colmat::c32::new(r as f32, i as f32),
        2e-5
    );
    case!(z32, z, colmat::c64, i32, [], colmat::c64::new, 1e-12);
    case!(z64, z, colmat::c64, i64, [64_], colmat::c64::new, 1e-12);
}
