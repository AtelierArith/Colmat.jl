//! Band/packed self-adjoint matrices adapted to native full-spectrum EVD.
//! This is a dense expansion, not a dedicated band/packed eigensolver.
use crate::generalized_selfadjoint::GeneralizedScalar;
use crate::kernels::flag;
use crate::kernels::Scalar;
use crate::lapack::LapackInt;
use crate::storage_layout::TriangleStorage;
use colmat::Mat;
use paste::paste;

#[derive(Clone, Copy)]
pub(crate) enum Storage {
    Band { width: usize, ld: usize },
    Packed,
}
impl Storage {
    fn rows(self, n: usize, lower: bool, j: usize) -> (usize, usize) {
        match self {
            Self::Band { width, .. } => {
                if lower {
                    (j, n.min(j + width + 1))
                } else {
                    (j.saturating_sub(width), j + 1)
                }
            }
            Self::Packed => {
                if lower {
                    (j, n)
                } else {
                    (0, j + 1)
                }
            }
        }
    }
    fn index(self, n: usize, lower: bool, i: usize, j: usize) -> usize {
        match self {
            Self::Band { width, ld } => j * ld + if lower { i - j } else { width + i - j },
            Self::Packed => {
                TriangleStorage::Packed.segment(n, lower, j).0 + if lower { i - j } else { i }
            }
        }
    }
}
#[derive(Clone, Copy)]
pub(crate) enum Selection {
    All,
    Value(f64, f64),
    Index(usize, usize),
}
pub(crate) struct Spectrum<T: Scalar> {
    pub(crate) values: Vec<f64>,
    pub(crate) vectors: Mat<T>,
}
impl<T: Scalar> Spectrum<T> {
    pub(crate) fn selected(&self, range: Selection) -> Vec<usize> {
        self.values
            .iter()
            .enumerate()
            .filter_map(|(i, &value)| {
                let selected = match range {
                    Selection::All => true,
                    Selection::Value(lo, hi) => lo < value && value <= hi,
                    Selection::Index(lo, hi) => lo <= i && i < hi,
                };
                selected.then_some(i)
            })
            .collect()
    }
    /// A diagonal matrix is a valid tridiagonal reduction. With eigenvectors
    /// as the full reduction basis Q, A=Q*diag(values)*Q^H holds, including
    /// selected drivers whose final Z contains only a subset of Q.
    pub(crate) unsafe fn write_diagonal_reduction(&self, lower: bool, storage: Storage, a: *mut T) {
        let n = self.values.len();
        for j in 0..n {
            let (first, end) = storage.rows(n, lower, j);
            for i in first..end {
                *a.add(storage.index(n, lower, i, j)) = if i == j {
                    T::real(self.values[j])
                } else {
                    T::z()
                };
            }
        }
    }
}
pub(crate) unsafe fn compute<T: GeneralizedScalar>(
    n: usize,
    lower: bool,
    storage: Storage,
    a: *const T,
    vectors: bool,
) -> Result<Spectrum<T>, ()> {
    if n == 0 {
        return Ok(Spectrum {
            values: vec![],
            vectors: Mat::zeros(0, 0),
        });
    }
    let mut matrix = Mat::<T>::zeros(n, n);
    for j in 0..n {
        let (first, end) = storage.rows(n, lower, j);
        for i in first..end {
            let p = a.add(storage.index(n, lower, i, j));
            let value = if i == j { T::diagonal(p) } else { *p };
            matrix[(i, j)] = value;
            if i != j {
                matrix[(j, i)] = value.cj();
            }
        }
    }
    if !matrix.as_ref().is_all_finite() {
        return Err(());
    }
    let (values, vectors) = crate::lapack::selfadj(matrix.as_ref(), vectors)?;
    if !values.iter().all(|x| x.is_finite()) || !vectors.as_ref().is_all_finite() {
        return Err(());
    }
    Ok(Spectrum { values, vectors })
}

fn validate(
    job: u8,
    triangle: u8,
    n: isize,
    band: Option<(isize, isize)>,
    ldz: isize,
) -> Result<(), i64> {
    let zarg = if band.is_some() { 9 } else { 7 };
    let mut checks = vec![
        (1, matches!(job, b'N' | b'V')),
        (2, matches!(triangle, b'U' | b'L')),
        (3, n >= 0),
    ];
    if let Some((width, ld)) = band {
        checks.push((4, width >= 0));
        checks.push((6, width.checked_add(1).is_some_and(|min| ld >= min)));
    }
    checks.push((zarg, ldz >= 1 && (job != b'V' || ldz >= n)));
    for (arg, valid) in checks {
        if !valid {
            return Err(arg);
        }
    }
    Ok(())
}

unsafe fn argument_error<I: LapackInt>(info: *mut I, arg: i64) {
    *info = I::from_i64(-arg);
    crate::invalid(arg as i32);
}

// Common BEV/PEV contract. Caller work arrays are not borrowed by the
// full-spectrum native algorithm, which owns its dense expansion and scratch.
unsafe fn ordinary<T: GeneralizedScalar, R: Scalar, I: LapackInt>(
    job: u8,
    triangle: u8,
    n: isize,
    band: Option<(isize, isize)>,
    a: *mut T,
    w: *mut R,
    z: *mut T,
    ldz: isize,
    info: *mut I,
) {
    *info = I::from_i64(0);
    if let Err(arg) = validate(job, triangle, n, band, ldz) {
        argument_error(info, arg);
        return;
    }
    if n == 0 {
        return;
    }
    let storage = match band {
        Some((width, ld)) => Storage::Band {
            width: width as usize,
            ld: ld as usize,
        },
        None => Storage::Packed,
    };
    if n == 1 {
        *w = R::real(T::diagonal(a.add(storage.index(1, triangle == b'L', 0, 0))).re());
        if job == b'V' {
            *z = T::o();
        }
        return;
    }
    // Reject unrepresentable address arithmetic before allocating or borrowing.
    let element = std::mem::size_of::<T>() as isize;
    let dense = n.checked_mul(n).and_then(|x| x.checked_mul(element));
    let source = match band {
        Some((_, ld)) => n.checked_mul(ld),
        None => n
            .checked_add(1)
            .and_then(|x| n.checked_mul(x))
            .map(|x| x / 2),
    }
    .and_then(|x| x.checked_mul(element));
    let output = if job == b'V' {
        n.checked_mul(ldz).and_then(|x| x.checked_mul(element))
    } else {
        Some(0)
    };
    if dense.is_none() || source.is_none() || output.is_none() {
        *info = I::from_i64(1);
        return;
    }
    let Ok(spectrum) = compute(n as usize, triangle == b'L', storage, a, job == b'V') else {
        *info = I::from_i64(1);
        return;
    };
    for j in 0..n as usize {
        *w.add(j) = R::real(spectrum.values[j]);
        if job == b'V' {
            for i in 0..n as usize {
                *z.add(i + j * ldz as usize) = spectrum.vectors[(i, j)];
            }
        }
    }
    spectrum.write_diagonal_reduction(triangle == b'L', storage, a);
}

macro_rules! ordinary_exports {
    ($p:ident,$bev:ident,$pev:ident,$t:ty,$r:ty,$i:ty,$suffix:tt,($($extra:ident:$extra_ty:ty,)*)) => {paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $bev $suffix>](
            jobz:*const u8,uplo:*const u8,n:*const $i,kd:*const $i,ab:*mut $t,
            ldab:*const $i,w:*mut $r,z:*mut $t,ldz:*const $i,
            _work:*mut $t,$($extra:$extra_ty,)*info:*mut $i) {
            ordinary(flag(jobz),flag(uplo),*n as isize,Some((*kd as isize,*ldab as isize)),
                ab,w,z,*ldz as isize,info);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $pev $suffix>](
            jobz:*const u8,uplo:*const u8,n:*const $i,ap:*mut $t,w:*mut $r,
            z:*mut $t,ldz:*const $i,_work:*mut $t,$($extra:$extra_ty,)*info:*mut $i) {
            ordinary(flag(jobz),flag(uplo),*n as isize,None,ap,w,z,*ldz as isize,info);
        }
    }};
}
macro_rules! ordinary_abi {
    ($i:ty,$suffix:tt) => {
        ordinary_exports!(s,sbev,spev,f32,f32,$i,$suffix,());
        ordinary_exports!(d,sbev,spev,f64,f64,$i,$suffix,());
        ordinary_exports!(c,hbev,hpev,colmat::c32,f32,$i,$suffix,(_rwork:*mut f32,));
        ordinary_exports!(z,hbev,hpev,colmat::c64,f64,$i,$suffix,(_rwork:*mut f64,));
    };
}
ordinary_abi!(i32, "_");
ordinary_abi!(i64, "_64_");

// Preserve LAPACK's public minimum-work contract even though the native
// solver owns its scratch. BEVD/PEVD share the native full-spectrum solver;
// the suffix does not imply a dedicated structured divide-and-conquer kernel.
fn workspace_minima(
    n: isize,
    vectors: bool,
    band: bool,
    complex: bool,
) -> Option<(isize, isize, isize)> {
    if n <= 1 {
        return Some((1, 1, 1));
    }
    let nn = n as u128;
    let (work, real, integer) = if !vectors {
        (if complex { nn } else { 2 * nn }, nn, 1)
    } else if complex {
        (
            if band { 2 * nn * nn } else { 2 * nn },
            1 + 5 * nn + 2 * nn * nn,
            3 + 5 * nn,
        )
    } else {
        (
            if band {
                1 + 5 * nn + 2 * nn * nn
            } else {
                1 + 6 * nn + nn * nn
            },
            1,
            3 + 5 * nn,
        )
    };
    Some((
        isize::try_from(work).ok()?,
        isize::try_from(real).ok()?,
        isize::try_from(integer).ok()?,
    ))
}

unsafe fn with_workspace<T: GeneralizedScalar, R: Scalar, I: LapackInt>(
    job: u8,
    triangle: u8,
    n: isize,
    band: Option<(isize, isize)>,
    a: *mut T,
    w: *mut R,
    z: *mut T,
    ldz: isize,
    work: *mut T,
    lwork: isize,
    real_work: Option<(*mut R, isize)>,
    iwork: *mut I,
    liwork: isize,
    info: *mut I,
) {
    *info = I::from_i64(0);
    if let Err(arg) = validate(job, triangle, n, band, ldz) {
        argument_error(info, arg);
        return;
    }
    let work_arg = if band.is_some() { 11 } else { 9 };
    let integer_arg = work_arg + if T::COMPLEX { 4 } else { 2 };
    let Some((lw, lr, li)) = workspace_minima(n, job == b'V', band.is_some(), T::COMPLEX) else {
        argument_error(info, work_arg);
        return;
    };
    // IWORK(1) must itself fit the caller's integer ABI. Do not truncate a
    // valid ILP64 recommendation when exposing the LP64 entry point.
    if I::from_i64(li as i64).val() != li {
        argument_error(info, integer_arg);
        return;
    }
    *work = crate::lapack::workspace::<T>(lw);
    *iwork = I::from_i64(li as i64);
    if let Some((rwork, _)) = real_work {
        *rwork = crate::lapack::workspace::<R>(lr);
    }
    let query = lwork == -1 || liwork == -1 || real_work.is_some_and(|(_, len)| len == -1);
    if !query {
        if lwork < lw {
            argument_error(info, work_arg);
            return;
        }
        if real_work.is_some_and(|(_, len)| len < lr) {
            argument_error(info, work_arg + 2);
            return;
        }
        if liwork < li {
            argument_error(info, integer_arg);
            return;
        }
    }
    if query {
        return;
    }
    ordinary(job, triangle, n, band, a, w, z, ldz, info);
}

macro_rules! workspace_exports {
    ($p:ident,$bev:ident,$pev:ident,$t:ty,$r:ty,$i:ty,$suffix:tt,
     ($($extra:ident:$extra_ty:ty,)*),$real:expr) => {paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $bev $suffix>](
            jobz:*const u8,uplo:*const u8,n:*const $i,kd:*const $i,ab:*mut $t,
            ldab:*const $i,w:*mut $r,z:*mut $t,ldz:*const $i,
            work:*mut $t,lwork:*const $i,$($extra:$extra_ty,)*iwork:*mut $i,liwork:*const $i,info:*mut $i) {
            with_workspace(flag(jobz),flag(uplo),*n as isize,Some((*kd as isize,*ldab as isize)),
                ab,w,z,*ldz as isize,work,*lwork as isize,$real,iwork,*liwork as isize,info);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $pev $suffix>](
            jobz:*const u8,uplo:*const u8,n:*const $i,ap:*mut $t,w:*mut $r,z:*mut $t,ldz:*const $i,
            work:*mut $t,lwork:*const $i,$($extra:$extra_ty,)*iwork:*mut $i,liwork:*const $i,info:*mut $i) {
            with_workspace(flag(jobz),flag(uplo),*n as isize,None,ap,w,z,*ldz as isize,
                work,*lwork as isize,$real,iwork,*liwork as isize,info);
        }
    }};
}
macro_rules! workspace_abi {
    ($i:ty,$suffix:tt) => {
        workspace_exports!(s,sbevd,spevd,f32,f32,$i,$suffix,(),None);
        workspace_exports!(d,sbevd,spevd,f64,f64,$i,$suffix,(),None);
        workspace_exports!(c,hbevd,hpevd,colmat::c32,f32,$i,$suffix,
            (rwork:*mut f32,lrwork:*const $i,),Some((rwork,*lrwork as isize)));
        workspace_exports!(z,hbevd,hpevd,colmat::c64,f64,$i,$suffix,
            (rwork:*mut f64,lrwork:*const $i,),Some((rwork,*lrwork as isize)));
    };
}
workspace_abi!(i32, "_");
workspace_abi!(i64, "_64_");

#[cfg(test)]
mod tests {
    use super::*;
    macro_rules! call_workspace {
        (real,$f:ident,($($arg:expr,)*),$rw:expr,$lr:expr,$iw:expr,$li:expr,$info:expr) => {
            {let _=($rw,$lr); $f($($arg,)*$iw,$li,$info)}
        };
        (complex,$f:ident,($($arg:expr,)*),$rw:expr,$lr:expr,$iw:expr,$li:expr,$info:expr) => {
            $f($($arg,)*$rw,$lr,$iw,$li,$info)
        };
    }
    macro_rules! workspace_test {
        ($name:ident,$bev:ident,$pev:ident,$t:ty,$r:ty,$i:ty,$kind:tt) => {
            #[test]
            fn $name() {
                unsafe {
                    for band in [false, true] {
                        for job in [b'N', b'V'] {
                            for n in [0, 1, 2, 3] {
                                let (lw, lr, li) =
                                    workspace_minima(n, job == b'V', band, <$t>::COMPLEX).unwrap();
                                let ldz = n.max(1) as $i;
                                // Each individual -1 triggers a joint query. All
                                // matrix/output pointers are deliberately null.
                                for which in 0..if <$t>::COMPLEX { 3 } else { 2 } {
                                    let mut work = [<$t>::real(777.0); 2];
                                    let mut rwork = [777.0 as $r; 2];
                                    let mut iwork = [777 as $i; 2];
                                    let mut info: $i = -99;
                                    let lwork: $i = if which == 0 { -1 } else { 0 };
                                    let liwork: $i = if which == 1 { -1 } else { 0 };
                                    let lrwork: $i = if which == 2 { -1 } else { 0 };
                                    if band {
                                        call_workspace!(
                                            $kind,
                                            $bev,
                                            (
                                                &job,
                                                &b'U',
                                                &(n as $i),
                                                &1,
                                                std::ptr::null_mut(),
                                                &2,
                                                std::ptr::null_mut(),
                                                std::ptr::null_mut(),
                                                &ldz,
                                                work.as_mut_ptr(),
                                                &lwork,
                                            ),
                                            rwork.as_mut_ptr(),
                                            &lrwork,
                                            iwork.as_mut_ptr(),
                                            &liwork,
                                            &mut info
                                        );
                                    } else {
                                        call_workspace!(
                                            $kind,
                                            $pev,
                                            (
                                                &job,
                                                &b'U',
                                                &(n as $i),
                                                std::ptr::null_mut(),
                                                std::ptr::null_mut(),
                                                std::ptr::null_mut(),
                                                &ldz,
                                                work.as_mut_ptr(),
                                                &lwork,
                                            ),
                                            rwork.as_mut_ptr(),
                                            &lrwork,
                                            iwork.as_mut_ptr(),
                                            &liwork,
                                            &mut info
                                        );
                                    }
                                    assert_eq!(info, 0);
                                    assert_eq!(work[0], <$t>::real(lw as f64));
                                    assert_eq!(work[1], <$t>::real(777.0));
                                    assert_eq!(iwork, [li as $i, 777]);
                                    if <$t>::COMPLEX {
                                        assert_eq!(rwork, [lr as $r, 777.0]);
                                    }
                                }
                                // Identity needs no reference eigensolver and
                                // permits checking the full V/N result at minimum work.
                                let mut a = vec![
                                    <$t>::z();
                                    if band {
                                        2 * n as usize
                                    } else {
                                        (n * (n + 1) / 2) as usize
                                    }
                                ];
                                let storage = if band {
                                    Storage::Band { width: 1, ld: 2 }
                                } else {
                                    Storage::Packed
                                };
                                for j in 0..n as usize {
                                    a[storage.index(n as usize, false, j, j)] = <$t>::o();
                                }
                                let mut values = vec![777.0 as $r; n as usize + 1];
                                let mut vectors =
                                    vec![<$t>::real(777.0); ldz as usize * n as usize + 1];
                                let mut work = vec![<$t>::real(777.0); lw as usize + 1];
                                let mut rwork = vec![777.0 as $r; lr as usize + 1];
                                let mut iwork = vec![777 as $i; li as usize + 1];
                                let mut info: $i = -99;
                                if band {
                                    call_workspace!(
                                        $kind,
                                        $bev,
                                        (
                                            &job,
                                            &b'U',
                                            &(n as $i),
                                            &1,
                                            a.as_mut_ptr(),
                                            &2,
                                            values.as_mut_ptr(),
                                            vectors.as_mut_ptr(),
                                            &ldz,
                                            work.as_mut_ptr(),
                                            &(lw as $i),
                                        ),
                                        rwork.as_mut_ptr(),
                                        &(lr as $i),
                                        iwork.as_mut_ptr(),
                                        &(li as $i),
                                        &mut info
                                    );
                                } else {
                                    call_workspace!(
                                        $kind,
                                        $pev,
                                        (
                                            &job,
                                            &b'U',
                                            &(n as $i),
                                            a.as_mut_ptr(),
                                            values.as_mut_ptr(),
                                            vectors.as_mut_ptr(),
                                            &ldz,
                                            work.as_mut_ptr(),
                                            &(lw as $i),
                                        ),
                                        rwork.as_mut_ptr(),
                                        &(lr as $i),
                                        iwork.as_mut_ptr(),
                                        &(li as $i),
                                        &mut info
                                    );
                                }
                                assert_eq!(info, 0);
                                for j in 0..n as usize {
                                    assert!((values[j] as f64 - 1.0).abs() < 1e-5);
                                    if job == b'V' {
                                        for k in 0..n as usize {
                                            let mut dot = <$t>::z();
                                            for i in 0..n as usize {
                                                dot = dot
                                                    + vectors[i + j * ldz as usize].cj()
                                                        * vectors[i + k * ldz as usize];
                                            }
                                            assert!(
                                                (dot - <$t>::real(if j == k { 1.0 } else { 0.0 }))
                                                    .magnitude()
                                                    < 1e-5
                                            );
                                        }
                                    }
                                }
                                assert_eq!(values[n as usize], 777.0);
                                assert_eq!(*vectors.last().unwrap(), <$t>::real(777.0));
                                if job == b'N' {
                                    assert!(vectors.iter().all(|&v| v == <$t>::real(777.0)));
                                }
                                assert_eq!(*work.last().unwrap(), <$t>::real(777.0));
                                assert_eq!(*iwork.last().unwrap(), 777);
                                assert_eq!(*rwork.last().unwrap(), 777.0);
                                for short in 0..if <$t>::COMPLEX { 3 } else { 2 } {
                                    let lwork = lw as $i - if short == 0 { 1 } else { 0 };
                                    let liwork = li as $i - if short == 1 { 1 } else { 0 };
                                    let lrwork = lr as $i - if short == 2 { 1 } else { 0 };
                                    if band {
                                        call_workspace!(
                                            $kind,
                                            $bev,
                                            (
                                                &job,
                                                &b'U',
                                                &(n as $i),
                                                &1,
                                                std::ptr::null_mut(),
                                                &2,
                                                std::ptr::null_mut(),
                                                std::ptr::null_mut(),
                                                &ldz,
                                                work.as_mut_ptr(),
                                                &lwork,
                                            ),
                                            rwork.as_mut_ptr(),
                                            &lrwork,
                                            iwork.as_mut_ptr(),
                                            &liwork,
                                            &mut info
                                        );
                                    } else {
                                        call_workspace!(
                                            $kind,
                                            $pev,
                                            (
                                                &job,
                                                &b'U',
                                                &(n as $i),
                                                std::ptr::null_mut(),
                                                std::ptr::null_mut(),
                                                std::ptr::null_mut(),
                                                &ldz,
                                                work.as_mut_ptr(),
                                                &lwork,
                                            ),
                                            rwork.as_mut_ptr(),
                                            &lrwork,
                                            iwork.as_mut_ptr(),
                                            &liwork,
                                            &mut info
                                        );
                                    }
                                    let arg = if band { 11 } else { 9 };
                                    let expected = match short {
                                        0 => arg,
                                        1 => arg + if <$t>::COMPLEX { 4 } else { 2 },
                                        _ => arg + 2,
                                    };
                                    assert_eq!(info, -expected);
                                }
                            }
                        }
                    }
                }
            }
        };
    }
    workspace_test!(workspace_s32, ssbevd_, sspevd_, f32, f32, i32, real);
    workspace_test!(workspace_d32, dsbevd_, dspevd_, f64, f64, i32, real);
    workspace_test!(
        workspace_c32,
        chbevd_,
        chpevd_,
        colmat::c32,
        f32,
        i32,
        complex
    );
    workspace_test!(
        workspace_z32,
        zhbevd_,
        zhpevd_,
        colmat::c64,
        f64,
        i32,
        complex
    );
    workspace_test!(workspace_s64, ssbevd_64_, sspevd_64_, f32, f32, i64, real);
    workspace_test!(workspace_d64, dsbevd_64_, dspevd_64_, f64, f64, i64, real);
    workspace_test!(
        workspace_c64,
        chbevd_64_,
        chpevd_64_,
        colmat::c32,
        f32,
        i64,
        complex
    );
    workspace_test!(
        workspace_z64,
        zhbevd_64_,
        zhpevd_64_,
        colmat::c64,
        f64,
        i64,
        complex
    );
    #[test]
    fn workspace_reference_minima_and_integer_boundaries() {
        // Reference-LAPACK 3.12.1 formulas evaluated at concrete sizes.
        for (band, complex, expected2, expected3) in [
            (true, false, (19, 1, 13), (34, 1, 18)),
            (false, false, (17, 1, 13), (28, 1, 18)),
            (true, true, (8, 19, 13), (18, 34, 18)),
            (false, true, (4, 19, 13), (6, 34, 18)),
        ] {
            assert_eq!(workspace_minima(2, true, band, complex), Some(expected2));
            assert_eq!(workspace_minima(3, true, band, complex), Some(expected3));
        }
        assert_eq!(workspace_minima(isize::MAX, true, true, true), None);
        unsafe {
            let mut work = 777.0f32;
            let mut iwork = 777i32;
            let mut info = -99i32;
            sspevd_(
                b"V".as_ptr(),
                b"U".as_ptr(),
                &4096,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &4096,
                &mut work,
                &-1,
                &mut iwork,
                &0,
                &mut info,
            );
            assert_eq!(info, 0);
            assert!(work as f64 >= 16_801_793.0);
            assert_eq!(iwork, 20_483);
            work = 777.0;
            iwork = 777;
            sspevd_(
                b"V".as_ptr(),
                b"U".as_ptr(),
                &430_000_000,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &430_000_000,
                &mut work,
                &-1,
                &mut iwork,
                &0,
                &mut info,
            );
            assert_eq!(info, -11);
            assert_eq!(work, 777.0);
            assert_eq!(iwork, 777);
        }
    }
    macro_rules! ordinary_test {
        ($name:ident,$band:ident,$packed:ident,$t:ty,$r:ty,$i:ty,($($extra:expr,)*)) => {
            #[test]
            fn $name() { unsafe {
                for lower in [false,true] {
                    let triangle = if lower { b'L' } else { b'U' };
                    let mut ab = if lower {
                        [<$t>::real(2.0),<$t>::real(1.0),<$t>::real(777.0),
                         <$t>::real(2.0),<$t>::real(f64::NAN),<$t>::real(777.0)]
                    } else {
                        [<$t>::real(f64::NAN),<$t>::real(2.0),<$t>::real(777.0),
                         <$t>::real(1.0),<$t>::real(2.0),<$t>::real(777.0)]
                    };
                    let mut w = [0.0 as $r;3]; w[2]=777.0;
                    let mut z = [<$t>::real(777.0);6];
                    let mut info: $i = -99;
                    $band(&b'V',&triangle,&2,&1,ab.as_mut_ptr(),&3,w.as_mut_ptr(),
                        z.as_mut_ptr(),&3,std::ptr::null_mut(),$($extra,)* &mut info);
                    assert_eq!(info,0);
                    assert!((w[0] as f64-1.0).abs()<1e-5);
                    assert!((w[1] as f64-3.0).abs()<1e-5);
                    assert_eq!(w[2],777.0);
                    for j in 0..2 {
                        assert_eq!(ab[2+3*j],<$t>::real(777.0));
                        assert_eq!(z[2+3*j],<$t>::real(777.0));
                        let (x,y)=(z[3*j],z[1+3*j]);
                        assert!((x*<$t>::real(2.0)+y-x*<$t>::real(w[j] as f64)).magnitude()<1e-5);
                        assert!((x+y*<$t>::real(2.0)-y*<$t>::real(w[j] as f64)).magnitude()<1e-5);
                        assert!((x.magnitude().powi(2)+y.magnitude().powi(2)-1.0).abs()<1e-5);
                    }
                    let mut ap = [<$t>::real(2.0),<$t>::real(1.0),<$t>::real(2.0),<$t>::real(777.0)];
                    $packed(&b'N',&triangle,&2,ap.as_mut_ptr(),w.as_mut_ptr(),
                        std::ptr::null_mut(),&1,std::ptr::null_mut(),$($extra,)* &mut info);
                    assert_eq!(info,0);
                    assert!((w[0] as f64-1.0).abs()<1e-5);
                    assert!((w[1] as f64-3.0).abs()<1e-5);
                    assert_eq!(ap[3],<$t>::real(777.0));
                }
                let mut info: $i = -99;
                $band(&b'N',&b'U',&0,&0,std::ptr::null_mut(),&1,std::ptr::null_mut(),
                    std::ptr::null_mut(),&1,std::ptr::null_mut(),$($extra,)* &mut info);
                assert_eq!(info,0);
                $packed(&b'N',&b'U',&0,std::ptr::null_mut(),std::ptr::null_mut(),
                    std::ptr::null_mut(),&1,std::ptr::null_mut(),$($extra,)* &mut info);
                assert_eq!(info,0);
                // Invalid leading dimensions must be rejected before reading arrays.
                $band(&b'V',&b'L',&2,&1,std::ptr::null_mut(),&2,std::ptr::null_mut(),
                    std::ptr::null_mut(),&1,std::ptr::null_mut(),$($extra,)* &mut info);
                assert_eq!(info,-9);
                $packed(&b'V',&b'L',&2,std::ptr::null_mut(),std::ptr::null_mut(),
                    std::ptr::null_mut(),&1,std::ptr::null_mut(),$($extra,)* &mut info);
                assert_eq!(info,-7);
            }}
        };
    }
    ordinary_test!(s32, ssbev_, sspev_, f32, f32, i32, ());
    ordinary_test!(d32, dsbev_, dspev_, f64, f64, i32, ());
    ordinary_test!(
        c32,
        chbev_,
        chpev_,
        colmat::c32,
        f32,
        i32,
        (std::ptr::null_mut(),)
    );
    ordinary_test!(
        z32,
        zhbev_,
        zhpev_,
        colmat::c64,
        f64,
        i32,
        (std::ptr::null_mut(),)
    );
    ordinary_test!(s64, ssbev_64_, sspev_64_, f32, f32, i64, ());
    ordinary_test!(d64, dsbev_64_, dspev_64_, f64, f64, i64, ());
    ordinary_test!(
        c64,
        chbev_64_,
        chpev_64_,
        colmat::c32,
        f32,
        i64,
        (std::ptr::null_mut(),)
    );
    ordinary_test!(
        z64,
        zhbev_64_,
        zhpev_64_,
        colmat::c64,
        f64,
        i64,
        (std::ptr::null_mut(),)
    );
    fn residual<T: Scalar>(a: &[T], s: &Spectrum<T>, tol: f64) {
        let n = s.values.len();
        for j in 0..n {
            for i in 0..n {
                let mut value = T::z();
                let mut dot = T::z();
                for k in 0..n {
                    value = value + a[i + k * n] * s.vectors[(k, j)];
                    dot = dot + s.vectors[(k, i)].cj() * s.vectors[(k, j)];
                }
                assert!((value - s.vectors[(i, j)] * T::real(s.values[j])).magnitude() < tol);
                assert!((dot - if i == j { T::o() } else { T::z() }).magnitude() < tol);
            }
        }
    }
    #[test]
    fn band_triangles_ignore_padding_and_produce_a_reduction_basis() {
        unsafe {
            let original = [2.0f64, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0];
            for lower in [false, true] {
                let mut ab = if lower {
                    [2.0, 1.0, 777.0, 2.0, 1.0, 777.0, 2.0, f64::NAN, 777.0]
                } else {
                    [f64::NAN, 2.0, 777.0, 1.0, 2.0, 777.0, 1.0, 2.0, 777.0]
                };
                let before = ab;
                let storage = Storage::Band { width: 1, ld: 3 };
                let s = compute(3, lower, storage, ab.as_ptr(), true).unwrap();
                residual(&original, &s, 1e-12);
                assert!((s.values[0] - (2.0 - 2.0f64.sqrt())).abs() < 1e-12);
                assert_eq!(s.selected(Selection::All), [0, 1, 2]);
                assert_eq!(s.selected(Selection::Index(1, 2)), [1]);
                assert_eq!(s.selected(Selection::Value(s.values[1], s.values[2])), [2]);
                for i in 0..9 {
                    assert!(ab[i] == before[i] || (ab[i].is_nan() && before[i].is_nan()));
                }
                s.write_diagonal_reduction(lower, storage, ab.as_mut_ptr());
                for j in 0..3 {
                    assert_eq!(ab[2 + 3 * j], 777.0);
                    assert_eq!(ab[(if lower { 0 } else { 1 }) + 3 * j], s.values[j]);
                }
            }
        }
    }
    #[test]
    fn packed_hermitian_ignores_imaginary_diagonals() {
        use colmat::c64;
        unsafe {
            let a = [
                c64::new(2.0, f64::NAN),
                c64::new(0.0, 1.0),
                c64::new(2.0, f64::NAN),
            ];
            let s = compute(2, false, Storage::Packed, a.as_ptr(), true).unwrap();
            residual(
                &[
                    c64::new(2.0, 0.0),
                    c64::new(0.0, -1.0),
                    c64::new(0.0, 1.0),
                    c64::new(2.0, 0.0),
                ],
                &s,
                1e-12,
            );
            assert!((s.values[0] - 1.0).abs() < 1e-12);
            assert!((s.values[1] - 3.0).abs() < 1e-12);
        }
    }
    #[test]
    fn single_precision_and_values_only_paths() {
        unsafe {
            let a = [2.0f32, 0.0, 3.0];
            let s = compute(2, false, Storage::Packed, a.as_ptr(), false).unwrap();
            assert_eq!(s.values, [2.0, 3.0]);
            assert_eq!(s.vectors.ncols(), 0);
            let a = [
                colmat::c32::new(2.0, f32::NAN),
                colmat::c32::new(0.0, 1.0),
                colmat::c32::new(2.0, f32::NAN),
            ];
            let s = compute(2, false, Storage::Packed, a.as_ptr(), true).unwrap();
            assert!((s.values[0] - 1.0).abs() < 1e-5);
            assert!((s.values[1] - 3.0).abs() < 1e-5);
        }
    }
    #[test]
    fn extreme_scales_preserve_relative_eigenvalues() {
        unsafe {
            for scale in [1e-300f64, 1e300] {
                let a = [2.0 * scale, scale, 2.0 * scale];
                let s = compute(2, false, Storage::Packed, a.as_ptr(), true).unwrap();
                assert!((s.values[0] / scale - 1.0).abs() < 1e-12);
                assert!((s.values[1] / scale - 3.0).abs() < 1e-12);
            }
            for scale in [1e-35f32, 1e35] {
                let a = [2.0 * scale, scale, 2.0 * scale];
                let s = compute(2, false, Storage::Packed, a.as_ptr(), true).unwrap();
                assert!((s.values[0] / scale as f64 - 1.0).abs() < 1e-5);
                assert!((s.values[1] / scale as f64 - 3.0).abs() < 1e-5);
            }
            for scale in [1e-300f64, 1e300] {
                let a = [
                    colmat::c64::new(2.0 * scale, f64::NAN),
                    colmat::c64::new(0.0, scale),
                    colmat::c64::new(2.0 * scale, f64::NAN),
                ];
                let s = compute(2, false, Storage::Packed, a.as_ptr(), true).unwrap();
                assert!((s.values[0] / scale - 1.0).abs() < 1e-12);
                assert!((s.values[1] / scale - 3.0).abs() < 1e-12);
                let normalized = Spectrum {
                    values: s.values.iter().map(|x| x / scale).collect(),
                    vectors: s.vectors,
                };
                residual(
                    &[
                        colmat::c64::new(2.0, 0.0),
                        colmat::c64::new(0.0, -1.0),
                        colmat::c64::new(0.0, 1.0),
                        colmat::c64::new(2.0, 0.0),
                    ],
                    &normalized,
                    1e-12,
                );
            }
            for scale in [1e-35f32, 1e35] {
                let a = [
                    colmat::c32::new(2.0 * scale, f32::NAN),
                    colmat::c32::new(0.0, scale),
                    colmat::c32::new(2.0 * scale, f32::NAN),
                ];
                let s = compute(2, false, Storage::Packed, a.as_ptr(), true).unwrap();
                assert!((s.values[0] / scale as f64 - 1.0).abs() < 1e-5);
                assert!((s.values[1] / scale as f64 - 3.0).abs() < 1e-5);
            }
        }
    }
    #[test]
    fn empty_and_nonfinite_problems_do_not_mutate_input() {
        unsafe {
            let s = compute::<f64>(0, false, Storage::Packed, std::ptr::null(), true).unwrap();
            assert!(s.values.is_empty());
            let a = [f64::NAN, 0.0, 2.0];
            assert!(compute(2, false, Storage::Packed, a.as_ptr(), true).is_err());
            assert!(a[0].is_nan());
        }
    }
}
