//! GEQRF/GEQR2 preserve compact Householder storage and complex TAU.
use super::{
    ge_nancheck, row_to_column, self_adjoint::zeros, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR,
};
use crate::kernels::Scalar;
fn geometry(layout: i32, m: i64, n: i64, lda: i64, lwork: Option<i64>) -> Result<(), i64> {
    if !matches!(layout, ROW_MAJOR | COL_MAJOR) {
        return Err(-1);
    }
    if layout == ROW_MAJOR && lda < n {
        return Err(-5);
    }
    if m < 0 {
        return Err(-2);
    }
    if n < 0 {
        return Err(-3);
    }
    if layout == COL_MAJOR && lda < m.max(1) {
        return Err(-5);
    }
    if lwork.is_some_and(|v| v != -1 && v < n.max(1)) {
        return Err(-8);
    }
    Ok(())
}
unsafe fn work<T: Scalar>(
    layout: i32,
    m: i64,
    n: i64,
    a: *mut T,
    lda: i64,
    lwork: Option<i64>,
    mut call: impl FnMut(*mut T, i64) -> i64,
) -> i64 {
    if let Err(e) = geometry(layout, m, n, lda, lwork) {
        return e;
    }
    if layout == COL_MAJOR {
        return call(a, lda);
    }
    if lwork == Some(-1) || m == 0 || n == 0 {
        return call(a, m.max(1));
    }
    let (mm, nn) = (m as usize, n as usize);
    let mut temp = match row_to_column(mm, nn, a, lda as usize) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let info = call(temp.as_mut_ptr(), m);
    if info >= 0 {
        for i in 0..mm {
            for j in 0..nn {
                *a.add(i * lda as usize + j) = temp[i + j * mm];
            }
        }
    }
    info
}
unsafe fn high_nan<T: Scalar>(
    layout: i32,
    m: i64,
    n: i64,
    a: *const T,
    lda: i64,
) -> Result<(), i64> {
    if !matches!(layout, ROW_MAJOR | COL_MAJOR) {
        return Err(-1);
    }
    if LAPACKE_get_nancheck() != 0 && ge_nancheck(layout, m, n, a, lda)? {
        return Err(-4);
    }
    Ok(())
}
macro_rules! exports {
    ($p:ident,$t:ty,$j:ty,[$($suffix:tt)*],$f:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p geqrf_work $($suffix)*>](layout:i32,m:$j,n:$j,a:*mut $t,lda:$j,tau:*mut $t,workspace:*mut $t,lwork:$j)->$j {
            work(layout,m as i64,n as i64,a,lda as i64,Some(lwork as i64),|a,lda|{let mut info:$j=0;crate::qr::[<$p geqrf $f>](&m,&n,a,&(lda as $j),tau,workspace,&lwork,&mut info);if info<0{(info-1) as i64}else{info as i64}}) as $j
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p geqr2_work $($suffix)*>](layout:i32,m:$j,n:$j,a:*mut $t,lda:$j,tau:*mut $t,workspace:*mut $t)->$j {
            work(layout,m as i64,n as i64,a,lda as i64,None,|a,lda|{let mut info:$j=0;crate::qr::[<$p geqr2 $f>](&m,&n,a,&(lda as $j),tau,workspace,&mut info);if info<0{(info-1) as i64}else{info as i64}}) as $j
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p geqrf $($suffix)*>](layout:i32,m:$j,n:$j,a:*mut $t,lda:$j,tau:*mut $t)->$j {
            if let Err(e)=high_nan(layout,m as i64,n as i64,a,lda as i64){return e as $j;}
            let mut query=<$t>::z();let info=[<LAPACKE_ $p geqrf_work $($suffix)*>](layout,m,n,a,lda,tau,&mut query,-1);
            if info!=0{return info;}
            let length=query.re();if !length.is_finite()||length<1.||length>=isize::MAX as f64||length > <$j>::MAX as f64{return -1010;}
            let mut workspace=match zeros::<$t>(length as usize){Ok(v)=>v,Err(_)=>return -1010};
            [<LAPACKE_ $p geqrf_work $($suffix)*>](layout,m,n,a,lda,tau,workspace.as_mut_ptr(),length as $j)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p geqr2 $($suffix)*>](layout:i32,m:$j,n:$j,a:*mut $t,lda:$j,tau:*mut $t)->$j {
            if let Err(e)=high_nan(layout,m as i64,n as i64,a,lda as i64){return e as $j;}
            if let Err(e)=geometry(layout,m as i64,n as i64,lda as i64,None){return e as $j;}
            let mut workspace=match zeros::<$t>((n as usize).max(1)){Ok(v)=>v,Err(_)=>return -1010};
            [<LAPACKE_ $p geqr2_work $($suffix)*>](layout,m,n,a,lda,tau,workspace.as_mut_ptr())
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
        mut call: impl FnMut(bool, bool, i32, i64, i64, *mut T, i64, *mut T, *mut T, i64) -> i64,
        tol: f64,
    ) {
        for (m, n) in [
            (0usize, 0usize),
            (0, 3),
            (3, 0),
            (1, 1),
            (2, 4),
            (4, 2),
            (3, 3),
        ] {
            for layout in [ROW_MAJOR, COL_MAJOR] {
                for blocked in [false, true] {
                    for high in [false, true] {
                        for deficient in [false, true] {
                            let ld = if layout == ROW_MAJOR { n + 2 } else { m + 2 };
                            let k = m.min(n);
                            let at = |i: usize, j: usize| {
                                if layout == ROW_MAJOR {
                                    i * ld + j
                                } else {
                                    i + j * ld
                                }
                            };
                            let size = if layout == ROW_MAJOR {
                                m.max(1) * ld
                            } else {
                                n.max(1) * ld
                            };
                            let mut a = vec![T::real(f64::NAN); size];
                            let mut original = vec![T::z(); m * n];
                            for i in 0..m {
                                for j in 0..n {
                                    let value = if deficient && j == 0 {
                                        T::z()
                                    } else {
                                        make(
                                            ((i + 2 * j + 1) as f64).cos()
                                                + if i == j { 2. } else { 0. },
                                            ((2 * i + j + 1) as f64).sin() * 0.2,
                                        )
                                    };
                                    a[at(i, j)] = value;
                                    original[i + j * m] = value;
                                }
                            }
                            let saved = a.clone();
                            let mut tau = vec![T::real(777.); k + 2];
                            let mut workspace = vec![T::real(777.); n.max(1) + 2];
                            let same = |x: &[T], y: &[T]| {
                                x.iter().zip(y).all(|(&x, &y)| x == y || (x != x && y != y))
                            };
                            if blocked && !high {
                                assert_eq!(
                                    call(
                                        blocked,
                                        false,
                                        layout,
                                        m as i64,
                                        n as i64,
                                        a.as_mut_ptr(),
                                        ld as i64,
                                        unsafe { tau.as_mut_ptr().add(1) },
                                        unsafe { workspace.as_mut_ptr().add(1) },
                                        -1
                                    ),
                                    0
                                );
                                assert!(
                                    same(&a, &saved) && tau.iter().all(|&v| v == T::real(777.))
                                );
                                assert!(workspace[1].re() >= n.max(1) as f64);
                                assert_eq!(
                                    call(
                                        blocked,
                                        false,
                                        layout,
                                        m as i64,
                                        n as i64,
                                        a.as_mut_ptr(),
                                        ld as i64,
                                        unsafe { tau.as_mut_ptr().add(1) },
                                        unsafe { workspace.as_mut_ptr().add(1) },
                                        n.max(1) as i64 - 1
                                    ),
                                    -8
                                );
                                assert!(same(&a, &saved));
                            }
                            assert_eq!(
                                call(
                                    blocked,
                                    high,
                                    layout,
                                    m as i64,
                                    n as i64,
                                    a.as_mut_ptr(),
                                    ld as i64,
                                    unsafe { tau.as_mut_ptr().add(1) },
                                    unsafe { workspace.as_mut_ptr().add(1) },
                                    n.max(1) as i64
                                ),
                                0
                            );
                            assert!(tau[0] == T::real(777.) && tau[k + 1] == T::real(777.));
                            assert!(
                                workspace[0] == T::real(777.)
                                    && workspace[n.max(1) + 1] == T::real(777.)
                            );
                            if blocked && !high && (m == 0 || n == 0) {
                                assert!(workspace[1] == T::o());
                            }
                            let mut q = vec![T::z(); m * m];
                            for i in 0..m {
                                q[i + i * m] = T::o();
                            }
                            for j in 0..k {
                                let mut v = vec![T::z(); m];
                                v[j] = T::o();
                                for i in j + 1..m {
                                    v[i] = a[at(i, j)];
                                }
                                let mut qv = vec![T::z(); m];
                                for i in 0..m {
                                    for l in 0..m {
                                        qv[i] = qv[i] + q[i + l * m] * v[l];
                                    }
                                }
                                for i in 0..m {
                                    for l in 0..m {
                                        q[i + l * m] =
                                            q[i + l * m] - qv[i] * tau[j + 1] * v[l].cj();
                                    }
                                }
                            }
                            for i in 0..m {
                                for j in 0..n {
                                    let value = (0..m).fold(T::z(), |sum, l| {
                                        sum + if l <= j {
                                            q[i + l * m] * a[at(l, j)]
                                        } else {
                                            T::z()
                                        }
                                    });
                                    assert!((value - original[i + j * m]).magnitude() < tol * 20.);
                                }
                            }
                            for i in 0..m {
                                for j in 0..m {
                                    let value = (0..m).fold(T::z(), |sum, l| {
                                        sum + q[l + i * m].cj() * q[l + j * m]
                                    });
                                    assert!(
                                        (value - if i == j { T::o() } else { T::z() }).magnitude()
                                            < tol * 20.
                                    );
                                }
                            }
                            for pos in 0..a.len() {
                                let (i, j) = if layout == ROW_MAJOR {
                                    (pos / ld, pos % ld)
                                } else {
                                    (pos % ld, pos / ld)
                                };
                                if i >= m || j >= n {
                                    assert!(a[pos] != a[pos]);
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
            #[test]fn $name(){paste::paste!{numerical::<$t>($make,|blocked,high,layout,m,n,a,ld,tau,work,lwork|unsafe{
                if blocked {if high{[<LAPACKE_ $p geqrf $($suffix)*>](layout,m as $j,n as $j,a,ld as $j,tau) as i64}else{[<LAPACKE_ $p geqrf_work $($suffix)*>](layout,m as $j,n as $j,a,ld as $j,tau,work,lwork as $j) as i64}}
                else if high{[<LAPACKE_ $p geqr2 $($suffix)*>](layout,m as $j,n as $j,a,ld as $j,tau) as i64}else{[<LAPACKE_ $p geqr2_work $($suffix)*>](layout,m as $j,n as $j,a,ld as $j,tau,work) as i64}
            },$tol);}}
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
