//! Selected generalized EVD: full native spectrum plus guarded selection.
//! Expert absolute accuracy and ambiguous value bounds retain original LAPACK.
use crate::generalized_selfadjoint::GeneralizedScalar;
use crate::householder::ReflectorScalar;
use crate::kernels::{flag, matrix_addressable, Scalar};
use crate::lapack::LapackInt;
use crate::storage_layout::TriangleStorage;
use crate::structured_evd::{Selection, Spectrum};

pub(crate) unsafe fn range<R: Scalar, I: LapackInt>(
    mode: u8,
    n: isize,
    vl: *const R,
    vu: *const R,
    il: *const I,
    iu: *const I,
    args: [i64; 4],
) -> Result<Selection, i64> {
    match mode {
        b'A' => Ok(Selection::All),
        b'V' if n == 0 => Ok(Selection::All),
        b'V' => {
            let (lo, hi) = ((*vl).re(), (*vu).re());
            if hi <= lo {
                Err(args[1])
            } else {
                Ok(Selection::Value(lo, hi))
            }
        }
        b'I' => {
            let (lo, hi) = ((*il).val(), (*iu).val());
            if lo < 1 || lo > n.max(1) {
                Err(args[2])
            } else if hi < lo.min(n) || hi > n {
                Err(args[3])
            } else {
                Ok(Selection::Index((lo - 1) as usize, hi as usize))
            }
        }
        _ => Err(args[0]),
    }
}
pub(crate) fn ambiguous<T: ReflectorScalar>(values: &[f64], selection: Selection) -> bool {
    if let Selection::Value(lo, hi) = selection {
        if lo.is_nan() || hi.is_nan() {
            return true;
        }
        let scale = values.iter().fold(0.0f64, |s, v| s.max(v.abs()));
        let guard = 64.0 * values.len() as f64 * T::EPS * scale;
        values
            .iter()
            .any(|&v| (v - lo).abs() <= guard || (v - hi).abs() <= guard)
    } else {
        false
    }
}
unsafe fn packed<T: GeneralizedScalar + ReflectorScalar, R: Scalar, I: LapackInt>(
    kind: isize,
    job: u8,
    mode: u8,
    u: u8,
    n: isize,
    ap: *mut T,
    bp: *mut T,
    vl: *const R,
    vu: *const R,
    il: *const I,
    iu: *const I,
    abstol: *const R,
    m: *mut I,
    w: *mut R,
    z: *mut T,
    ldz: isize,
    ifail: *mut I,
    info: *mut I,
) -> bool {
    *info = I::from_i64(0);
    for (arg, ok) in [
        (1, (1..=3).contains(&kind)),
        (2, matches!(job, b'N' | b'V')),
        (3, matches!(mode, b'A' | b'V' | b'I')),
        (4, matches!(u, b'U' | b'L')),
        (5, n >= 0),
    ] {
        if !ok {
            *info = I::from_i64(-arg);
            crate::invalid(arg as i32);
            return true;
        }
    }
    let selected = match range(mode, n, vl, vu, il, iu, [3, 9, 10, 11]) {
        Ok(s) => s,
        Err(arg) => {
            *info = I::from_i64(-arg);
            crate::invalid(arg as i32);
            return true;
        }
    };
    if ldz < 1 || (job == b'V' && ldz < n) {
        *info = I::from_i64(-16);
        crate::invalid(16);
        return true;
    }
    *m = I::from_i64(0);
    if n == 0 {
        return true;
    }
    if !matrix_addressable::<T>(n, n, n) {
        *info = I::from_i64(-5);
        crate::invalid(5);
        return true;
    }
    if !((*abstol).re() <= 0.0) {
        return false;
    }
    let size = n as usize;
    let lower = u == b'L';
    let mut staged = vec![T::z(); size * (size + 1) / 2];
    for j in 0..size {
        let (offset, _, _) = TriangleStorage::Packed.segment(size, lower, j);
        let first = if lower { j } else { 0 };
        let count = if lower { size - j } else { j + 1 };
        for k in 0..count {
            staged[offset + k] = if first + k == j {
                T::diagonal(bp.add(offset + k))
            } else {
                *bp.add(offset + k)
            };
        }
    }
    // The caller's factor remains untouched until selection is accepted. Both
    // EVD refusal and a nonpositive B recover through the original packed driver.
    let Ok(result) = crate::structured_generalized::packed_eigen(
        kind as u8,
        job,
        lower,
        size,
        ap,
        staged.as_mut_ptr(),
    ) else {
        return false;
    };
    if ambiguous::<T>(&result.values, selected) {
        return false;
    }
    let spectrum = Spectrum {
        values: result.values,
        vectors: result.vectors,
    };
    let indices = spectrum.selected(selected);
    let columns = if size == 1 && job == b'V' {
        1
    } else {
        indices.len()
    };
    if job == b'V' && !matrix_addressable::<T>(n, columns as isize, ldz) {
        *info = I::from_i64(-16);
        crate::invalid(16);
        return true;
    }
    for (column, &source) in indices.iter().enumerate() {
        *w.add(column) = R::real(spectrum.values[source]);
        if job == b'V' {
            for row in 0..size {
                *z.offset(row as isize + column as isize * ldz) = spectrum.vectors[(row, source)];
            }
            // PEVX's singleton quick return does not reference IFAIL.
            if size > 1 {
                *ifail.add(column) = I::from_i64(0);
            }
        }
    }
    // PEVX writes the unit scalar even for an empty value interval; the
    // generalized driver then performs no back-transform because M=0.
    if size == 1 && job == b'V' && indices.is_empty() {
        *z = T::o();
    }
    crate::structured_generalized::pack(size, lower, ap, &result.reduced);
    for (k, value) in staged.into_iter().enumerate() {
        *bp.add(k) = value;
    }
    *m = I::from_i64(indices.len() as i64);
    true
}

macro_rules! real_exports {($p:ident,$t:ty,$r:ty,$i:ty,$s:tt,$id:expr)=>{paste::paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p spgvx $s>](kind:*const $i,job:*const u8,mode:*const u8,u:*const u8,n:*const $i,ap:*mut $t,bp:*mut $t,vl:*const $r,vu:*const $r,il:*const $i,iu:*const $i,abstol:*const $r,m:*mut $i,w:*mut $r,z:*mut $t,ldz:*const $i,work:*mut $t,iwork:*mut $i,ifail:*mut $i,info:*mut $i) {
        if packed((*kind).val(),flag(job),flag(mode),flag(u),(*n).val(),ap,bp,vl,vu,il,iu,abstol,m,w,z,(*ldz).val(),ifail,info) {return;}
        let f:unsafe extern "C" fn(*const i64,*const u8,*const u8,*const u8,*const i64,*mut $t,*mut $t,*const $r,*const $r,*const i64,*const i64,*const $r,*mut i64,*mut $r,*mut $t,*const i64,*mut $t,*mut i64,*mut i64,*mut i64,usize,usize,usize)=std::mem::transmute(crate::structured_generalized::fallback(8+$id));
        let nn=(*n).val() as usize;let vectors=flag(job)==b'V';let narrow=std::mem::size_of::<$i>()==4;
        let mut iw=Vec::new();let mut failed=Vec::new();
        let(ip,fp)=if narrow {iw.resize(5*nn,i64::MIN);failed.resize(if vectors {nn} else {0},i64::MIN);(iw.as_mut_ptr(),failed.as_mut_ptr())} else {(iwork.cast::<i64>(),ifail.cast::<i64>())};
        let (lo,hi)=if flag(mode)==b'I' {((*il).val() as i64,(*iu).val() as i64)} else {(1,nn as i64)};
        let(mut count,mut status)=(0i64,0i64);
        f(&((*kind).val() as i64),job,mode,u,&(nn as i64),ap,bp,vl,vu,&lo,&hi,abstol,&mut count,w,z,&((*ldz).val() as i64),work,ip,fp,&mut status,1,1,1);
        *m=count as $i;*info=status as $i;
        for(k,value)in iw.into_iter().enumerate() {if value!=i64::MIN {*iwork.add(k)=value as $i;}}
        for(k,value)in failed.into_iter().enumerate() {if value!=i64::MIN {*ifail.add(k)=value as $i;}}
    }
}};}

macro_rules! complex_exports {($p:ident,$t:ty,$r:ty,$i:ty,$s:tt,$id:expr)=>{paste::paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p hpgvx $s>](kind:*const $i,job:*const u8,mode:*const u8,u:*const u8,n:*const $i,ap:*mut $t,bp:*mut $t,vl:*const $r,vu:*const $r,il:*const $i,iu:*const $i,abstol:*const $r,m:*mut $i,w:*mut $r,z:*mut $t,ldz:*const $i,work:*mut $t,rwork:*mut $r,iwork:*mut $i,ifail:*mut $i,info:*mut $i) {
        if packed((*kind).val(),flag(job),flag(mode),flag(u),(*n).val(),ap,bp,vl,vu,il,iu,abstol,m,w,z,(*ldz).val(),ifail,info) {return;}
        let f:unsafe extern "C" fn(*const i64,*const u8,*const u8,*const u8,*const i64,*mut $t,*mut $t,*const $r,*const $r,*const i64,*const i64,*const $r,*mut i64,*mut $r,*mut $t,*const i64,*mut $t,*mut $r,*mut i64,*mut i64,*mut i64,usize,usize,usize)=std::mem::transmute(crate::structured_generalized::fallback(8+$id));
        let nn=(*n).val() as usize;let vectors=flag(job)==b'V';let narrow=std::mem::size_of::<$i>()==4;
        let mut iw=Vec::new();let mut failed=Vec::new();
        let(ip,fp)=if narrow {iw.resize(5*nn,i64::MIN);failed.resize(if vectors {nn} else {0},i64::MIN);(iw.as_mut_ptr(),failed.as_mut_ptr())} else {(iwork.cast::<i64>(),ifail.cast::<i64>())};
        let (lo,hi)=if flag(mode)==b'I' {((*il).val() as i64,(*iu).val() as i64)} else {(1,nn as i64)};
        let(mut count,mut status)=(0i64,0i64);
        f(&((*kind).val() as i64),job,mode,u,&(nn as i64),ap,bp,vl,vu,&lo,&hi,abstol,&mut count,w,z,&((*ldz).val() as i64),work,rwork,ip,fp,&mut status,1,1,1);
        *m=count as $i;*info=status as $i;
        for(k,value)in iw.into_iter().enumerate() {if value!=i64::MIN {*iwork.add(k)=value as $i;}}
        for(k,value)in failed.into_iter().enumerate() {if value!=i64::MIN {*ifail.add(k)=value as $i;}}
    }
}};}
real_exports!(s, f32, f32, i32, _, 0);
real_exports!(s, f32, f32, i64, _64_, 0);
real_exports!(d, f64, f64, i32, _, 1);
real_exports!(d, f64, f64, i64, _64_, 1);
complex_exports!(c, colmat::c32, f32, i32, _, 2);
complex_exports!(c, colmat::c32, f32, i64, _64_, 2);
complex_exports!(z, colmat::c64, f64, i32, _, 3);
complex_exports!(z, colmat::c64, f64, i64, _64_, 3);

#[cfg(test)]
mod tests {
    use super::*;
    use colmat::Mat;
    fn native_ranges<T: GeneralizedScalar + ReflectorScalar, R: Scalar, I: LapackInt>(tol: f64) {
        for lower in [false, true] {
            for kind in 1..=3 {
                for (mode, empty) in [(b'A', false), (b'I', false), (b'V', false), (b'V', true)] {
                    let n = 3;
                    let mut a = vec![T::real(777.0); 8];
                    let mut b = a.clone();
                    let aa = Mat::<T>::from_fn(n, n, |i, j| {
                        if i == j {
                            T::real(((i + 1) * (i + 1)) as f64)
                        } else {
                            T::z()
                        }
                    });
                    let bb = Mat::<T>::from_fn(n, n, |i, j| {
                        if i == j {
                            T::real((i + 1) as f64)
                        } else {
                            T::z()
                        }
                    });
                    unsafe {
                        crate::structured_generalized::pack(n, lower, a.as_mut_ptr().add(1), &aa);
                        crate::structured_generalized::pack(n, lower, b.as_mut_ptr().add(1), &bb);
                    }
                    let (lo, hi) = if empty {
                        (-5.0, -1.0)
                    } else if kind == 1 {
                        (1.5, 2.5)
                    } else {
                        (1.5, 26.5)
                    };
                    let (lo, hi) = (R::real(lo), R::real(hi));
                    let il = I::from_i64(2);
                    let iu = I::from_i64(3);
                    let abstol = R::z();
                    let mut m = [I::from_i64(-777); 3];
                    let mut info = I::from_i64(-777);
                    let mut w = [R::real(777.0); 5];
                    let mut z = vec![T::real(777.0); 17];
                    let mut fail = [I::from_i64(-777); 5];
                    assert!(unsafe {
                        packed(
                            kind,
                            b'V',
                            mode,
                            if lower { b'L' } else { b'U' },
                            3,
                            a.as_mut_ptr().add(1),
                            b.as_mut_ptr().add(1),
                            &lo,
                            &hi,
                            &il,
                            &iu,
                            &abstol,
                            &mut m[1],
                            w.as_mut_ptr().add(1),
                            z.as_mut_ptr().add(1),
                            5,
                            fail.as_mut_ptr().add(1),
                            &mut info,
                        )
                    });
                    assert_eq!(info.val(), 0);
                    assert_eq!(crate::colmat_blas_last_error(), 0);
                    let chosen: Vec<usize> = if empty {
                        Vec::new()
                    } else if mode == b'A' {
                        vec![0, 1, 2]
                    } else if mode == b'I' {
                        vec![1, 2]
                    } else {
                        vec![1]
                    };
                    assert_eq!(m[1].val(), chosen.len() as isize);
                    assert_eq!(m[0].val(), -777);
                    assert_eq!(m[2].val(), -777);
                    for (col, &index) in chosen.iter().enumerate() {
                        let lambda = if kind == 1 {
                            (index + 1) as f64
                        } else {
                            ((index + 1) * (index + 1) * (index + 1)) as f64
                        };
                        assert!((w[col + 1].re() - lambda).abs() < tol);
                        assert_eq!(fail[col + 1].val(), 0);
                        for row in 0..3 {
                            let v = z[1 + row + col * 5];
                            let ad = ((row + 1) * (row + 1)) as f64;
                            let bd = (row + 1) as f64;
                            let coefficient = if kind == 1 {
                                ad - bd * lambda
                            } else {
                                ad * bd - lambda
                            };
                            assert!((v * T::real(coefficient)).magnitude() < tol);
                        }
                    }
                    for col in 0..3 {
                        for row in 0..5 {
                            if col >= chosen.len() || row >= 3 {
                                assert_eq!(z[1 + row + col * 5].re(), 777.0);
                            }
                        }
                    }
                    assert_eq!(z[0].re(), 777.0);
                    assert_eq!(z[16].re(), 777.0);
                    for k in chosen.len() + 1..5 {
                        assert_eq!(w[k].re(), 777.0);
                        assert_eq!(fail[k].val(), -777);
                    }
                    assert_eq!(w[0].re(), 777.0);
                    assert_eq!(fail[0].val(), -777);
                    assert_eq!(a[0].re(), 777.0);
                    assert_eq!(a[7].re(), 777.0);
                    assert_eq!(b[0].re(), 777.0);
                    assert_eq!(b[7].re(), 777.0);
                }
            }
        }
    }
    #[test]
    fn selected_packed_ranges_all_scalar_and_integer_types() {
        native_ranges::<f32, f32, i32>(1e-3);
        native_ranges::<f32, f32, i64>(1e-3);
        native_ranges::<f64, f64, i32>(1e-11);
        native_ranges::<f64, f64, i64>(1e-11);
        native_ranges::<colmat::c32, f32, i32>(1e-3);
        native_ranges::<colmat::c32, f32, i64>(1e-3);
        native_ranges::<colmat::c64, f64, i32>(1e-11);
        native_ranges::<colmat::c64, f64, i64>(1e-11);
    }
    #[test]
    fn expert_recovery_sees_original_packed_inputs() {
        for (tol, lo, hi, bad_b) in [
            (1e-12, 1.5, 2.5, false),
            (0.0, 2.0, 2.5, false),
            (0.0, 1.5, 2.5, true),
        ] {
            let mut a = [1.0f64, 0.0, 4.0, 0.0, 0.0, 9.0];
            let mut b = [1.0f64, 0.0, if bad_b { -1.0 } else { 2.0 }, 0.0, 0.0, 3.0];
            let (olda, oldb) = (a, b);
            let mut w = [777.0f64; 3];
            let mut z = [777.0f64; 9];
            let mut fail = [-777i64; 3];
            let (mut m, mut info) = (-777i64, -777i64);
            assert!(!unsafe {
                packed(
                    1,
                    b'V',
                    b'V',
                    b'U',
                    3,
                    a.as_mut_ptr(),
                    b.as_mut_ptr(),
                    &lo,
                    &hi,
                    &1i64,
                    &3i64,
                    &tol,
                    &mut m,
                    w.as_mut_ptr(),
                    z.as_mut_ptr(),
                    3,
                    fail.as_mut_ptr(),
                    &mut info,
                )
            });
            assert_eq!(a, olda);
            assert_eq!(b, oldb);
            assert_eq!((m, info), (0, 0));
            assert_eq!(w, [777.0; 3]);
            assert_eq!(z, [777.0; 9]);
            assert_eq!(fail, [-777; 3]);
        }
    }
    #[test]
    fn empty_singleton_keeps_ifail_and_writes_unscaled_unit_z() {
        let (mut a, mut b) = (8.0f64, 2.0f64);
        let (mut m, mut info) = (-777i64, -777i64);
        let (mut w, mut z) = (777.0f64, 777.0f64);
        let mut fail = -777i64;
        assert!(unsafe {
            packed(
                1,
                b'V',
                b'V',
                b'L',
                1,
                &mut a,
                &mut b,
                &5.0,
                &6.0,
                std::ptr::null(),
                std::ptr::null(),
                &0.0,
                &mut m,
                &mut w,
                &mut z,
                isize::MAX,
                &mut fail,
                &mut info,
            )
        });
        assert_eq!((m, info, fail), (0, 0, -777));
        assert_eq!((w, z), (777.0, 1.0));
    }
}
