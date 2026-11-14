//! LAPACK matrix-norm contracts over native real reductions and complex l2.
//! Storage definitions follow Reference-LAPACK SRC/{lange,lantr,lansf,trttf}.
//! RFP block coordinates are validated with independent TRTTF packing.
use crate::generalized_selfadjoint::GeneralizedScalar;
use crate::householder::ReflectorScalar;
use crate::kernels::Scalar;
use colmat::{traits::RealField, MatRef};

#[derive(Clone, Copy)]
enum Kind {
    Maximum,
    One,
    Infinity,
    Frobenius,
}
impl Kind {
    fn parse(value: u8) -> Option<Self> {
        match value.to_ascii_uppercase() {
            b'M' => Some(Self::Maximum),
            b'1' | b'O' => Some(Self::One),
            b'I' => Some(Self::Infinity),
            b'F' | b'E' => Some(Self::Frobenius),
            _ => None,
        }
    }
}
struct Reduction<T: ReflectorScalar<Real = R>, R: Scalar + RealField> {
    kind: Kind,
    elements: Vec<T>,
    magnitudes: Vec<R>,
    groups: Vec<Vec<R>>,
    nan: bool,
    infinite: bool,
}
impl<T: ReflectorScalar<Real = R>, R: Scalar + RealField> Reduction<T, R> {
    fn new(kind: Kind, m: usize, n: usize) -> Self {
        let count = match kind {
            Kind::One => n,
            Kind::Infinity => m,
            _ => 0,
        };
        Self {
            kind,
            elements: Vec::new(),
            magnitudes: Vec::new(),
            groups: (0..count).map(|_| Vec::new()).collect(),
            nan: false,
            infinite: false,
        }
    }
    fn push(&mut self, i: usize, j: usize, value: T) {
        if matches!(self.kind, Kind::Frobenius) {
            if value.re().is_nan() || value.im().is_nan() {
                self.nan = true;
                return;
            }
            if value.re().is_infinite() || value.im().is_infinite() {
                self.infinite = true;
                return;
            }
            self.elements.push(value);
            return;
        }
        // LAPACK uses complex modulus, whereas native complex norm_max and
        // norm_l1 use component maxima and abs1 respectively. In M/1/I modes,
        // hypot(inf,NaN)=inf; Frobenius follows component-wise LASSQ semantics.
        let magnitude = R::real(value.magnitude());
        if magnitude.re().is_nan() {
            self.nan = true;
            return;
        }
        if magnitude.re().is_infinite() {
            self.infinite = true;
            return;
        }
        match self.kind {
            Kind::Maximum => self.magnitudes.push(magnitude),
            Kind::One => self.groups[j].push(magnitude),
            Kind::Infinity => self.groups[i].push(magnitude),
            Kind::Frobenius => unreachable!(),
        }
    }
    fn finish(self) -> R {
        if self.nan {
            return R::real(f64::NAN);
        }
        if self.infinite {
            return R::real(f64::INFINITY);
        }
        unsafe {
            match self.kind {
                Kind::Frobenius => {
                    if self.elements.is_empty() {
                        return R::z();
                    }
                    // The pinned native three-scale squared accumulators lose
                    // subnormal or near-MAX norms. See the standalone crate at
                    // reproducers/native-norm-l2; keep normalization until the
                    // native range defect is fixed and independently rechecked.
                    let scale = self
                        .elements
                        .iter()
                        .map(|value| value.re().abs().max(value.im().abs()))
                        .fold(0.0, f64::max);
                    if scale == 0.0 {
                        return R::z();
                    }
                    let mut elements = self.elements;
                    for value in &mut elements {
                        *value = T::parts(value.re() / scale, value.im() / scale);
                    }
                    let normalized =
                        MatRef::from_raw_parts(elements.as_ptr(), elements.len(), 1, 1, 0)
                            .norm_l2();
                    R::real(normalized.re() * scale)
                }
                Kind::Maximum => real_max(&self.magnitudes),
                Kind::One | Kind::Infinity => {
                    let sums: Vec<R> = self
                        .groups
                        .iter()
                        .map(|values| {
                            if values.is_empty() {
                                R::z()
                            } else {
                                MatRef::from_raw_parts(values.as_ptr(), values.len(), 1, 1, 0).sum()
                            }
                        })
                        .collect();
                    real_max(&sums)
                }
            }
        }
    }
}
unsafe fn real_max<R: Scalar + RealField>(values: &[R]) -> R {
    if values.is_empty() {
        R::z()
    } else {
        MatRef::from_raw_parts(values.as_ptr(), values.len(), 1, 1, 0).norm_max()
    }
}
unsafe fn stored<T: GeneralizedScalar>(
    a: *const T,
    index: isize,
    diagonal: bool,
    hermitian: bool,
) -> T {
    if diagonal && hermitian {
        T::diagonal(a.offset(index))
    } else {
        *a.offset(index)
    }
}
// Enumerate a triangular RFP array without accessing a full n-by-n expansion.
// Conjugation in complex TRANSR='C' does not affect any norm or real diagonal.
fn visit_rfp(n: usize, lower: bool, transposed: bool, mut visit: impl FnMut(usize, usize, usize)) {
    if n == 0 {
        return;
    }
    let even = n % 2 == 0;
    let rows = if even { n + 1 } else { n };
    let columns = (n + 1) / 2;
    let half = n / 2;
    for j in 0..n {
        let indices = if lower { j..n } else { 0..j + 1 };
        for i in indices {
            let (row, column) = if lower {
                if j < n - half {
                    (i + usize::from(even), j)
                } else {
                    (j - (n - half), i - half)
                }
            } else if j >= half {
                (i, j - half)
            } else {
                (half + j + 1, i)
            };
            let index = if transposed {
                column + row * columns
            } else {
                row + column * rows
            };
            visit(i, j, index);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    #[test]
    fn rectangular_packed_mapping_is_bijective() {
        for n in 0..14 {
            for lower in [false, true] {
                for transposed in [false, true] {
                    let mut positions = HashSet::new();
                    let mut entries = HashSet::new();
                    visit_rfp(n, lower, transposed, |i, j, index| {
                        assert!(i < n && j < n);
                        assert!(if lower { i >= j } else { i <= j });
                        assert!(positions.insert(index));
                        assert!(entries.insert((i, j)));
                    });
                    let count = n * (n + 1) / 2;
                    assert_eq!(entries.len(), count);
                    assert_eq!(positions, (0..count).collect());
                }
            }
        }
    }
    #[test]
    fn frobenius_range_and_mixed_nonfinite_semantics() {
        for scale in [f64::from_bits(1), f64::MAX / 8.0] {
            let mut norm = Reduction::<f64, f64>::new(Kind::Frobenius, 7, 7);
            for j in 0..7 {
                for i in 0..7 {
                    norm.push(i, j, scale);
                }
            }
            let actual = norm.finish();
            assert!(actual.is_finite() && actual > 0.0);
            assert!((actual / (scale * 7.0) - 1.0).abs() < 1e-14);
        }
        for kind in [Kind::Maximum, Kind::One, Kind::Infinity] {
            let mut norm = Reduction::<colmat::c64, f64>::new(kind, 1, 1);
            norm.push(0, 0, colmat::c64::new(f64::INFINITY, f64::NAN));
            assert!(norm.finish().is_infinite());
        }
        let mut norm = Reduction::<colmat::c64, f64>::new(Kind::Frobenius, 1, 1);
        norm.push(0, 0, colmat::c64::new(f64::INFINITY, f64::NAN));
        assert!(norm.finish().is_nan());
    }
    #[test]
    fn empty_and_unit_storage_need_no_input_memory() {
        unsafe {
            use std::ptr::{null, null_mut};
            for mode in [b'M', b'1', b'I', b'F'] {
                for u in [b'U', b'L'] {
                    assert_eq!(dlantp_64_(&mode, &u, &b'U', &1, null(), null_mut()), 1.0);
                    assert_eq!(
                        dlantb_64_(&mode, &u, &b'U', &1, &0, null(), &1, null_mut()),
                        1.0
                    );
                    assert_eq!(
                        dlantr_64_(&mode, &u, &b'U', &1, &1, null(), &1, null_mut()),
                        1.0
                    );
                }
            }
            assert_eq!(
                dlange_64_(&b'I', &i64::MAX, &0, null(), &i64::MAX, null_mut()),
                0.0
            );
            assert_eq!(dlangt_64_(&b'F', &0, null(), null(), null()), 0.0);
        }
    }
    #[test]
    fn complex_modulus_and_induced_norms_differ_from_component_reductions() {
        for (kind, expected) in [
            (Kind::Maximum, 5.0),
            (Kind::One, 7.0),
            (Kind::Infinity, 6.0),
        ] {
            let mut norm = Reduction::<colmat::c64, f64>::new(kind, 2, 2);
            norm.push(0, 0, colmat::c64::new(3.0, 4.0));
            norm.push(1, 0, colmat::c64::new(2.0, 0.0));
            norm.push(0, 1, colmat::c64::new(1.0, 0.0));
            assert_eq!(norm.finish(), expected);
        }
        assert!(Kind::parse(b'?').is_none());
    }
}

unsafe fn dense<T: GeneralizedScalar + ReflectorScalar<Real = R>, R: Scalar + RealField>(
    kind: Kind,
    m: usize,
    n: usize,
    a: *const T,
    ld: isize,
    hessenberg: bool,
) -> R {
    if m == 0 || n == 0 {
        return R::z();
    }
    let mut norm = Reduction::new(kind, m, n);
    for j in 0..n {
        for i in 0..if hessenberg {
            m.min(j.saturating_add(2))
        } else {
            m
        } {
            norm.push(i, j, *a.offset(i as isize + j as isize * ld));
        }
    }
    norm.finish()
}
unsafe fn symmetric<T: GeneralizedScalar + ReflectorScalar<Real = R>, R: Scalar + RealField>(
    kind: Kind,
    u: u8,
    n: usize,
    a: *const T,
    ld: isize,
    hermitian: bool,
) -> R {
    let mut norm = Reduction::new(kind, n, n);
    for j in 0..n {
        let rows = if u == b'U' { 0..j + 1 } else { j..n };
        for i in rows {
            let value = stored(a, i as isize + j as isize * ld, i == j, hermitian);
            norm.push(i, j, value);
            if i != j {
                norm.push(j, i, value);
            }
        }
    }
    norm.finish()
}
unsafe fn band<T: GeneralizedScalar + ReflectorScalar<Real = R>, R: Scalar + RealField>(
    kind: Kind,
    n: usize,
    kl: usize,
    ku: usize,
    a: *const T,
    ld: isize,
) -> R {
    let mut norm = Reduction::new(kind, n, n);
    for j in 0..n {
        for i in j.saturating_sub(ku)..n.min(j.saturating_add(kl).saturating_add(1)) {
            norm.push(i, j, *a.offset((ku + i - j) as isize + j as isize * ld));
        }
    }
    norm.finish()
}
unsafe fn selected_band<T: GeneralizedScalar + ReflectorScalar<Real = R>, R: Scalar + RealField>(
    kind: Kind,
    u: u8,
    n: usize,
    k: usize,
    a: *const T,
    ld: isize,
    symmetric: bool,
    hermitian: bool,
    unit: bool,
) -> R {
    let mut norm = Reduction::new(kind, n, n);
    for j in 0..n {
        let rows = if u == b'U' {
            j.saturating_sub(k)..j + 1
        } else {
            j..n.min(j.saturating_add(k).saturating_add(1))
        };
        for i in rows {
            let row = if u == b'U' { k + i - j } else { i - j };
            let value = if unit && i == j {
                T::o()
            } else {
                stored(a, row as isize + j as isize * ld, i == j, hermitian)
            };
            norm.push(i, j, value);
            if symmetric && i != j {
                norm.push(j, i, value);
            }
        }
    }
    norm.finish()
}
unsafe fn packed<T: GeneralizedScalar + ReflectorScalar<Real = R>, R: Scalar + RealField>(
    kind: Kind,
    u: u8,
    n: usize,
    a: *const T,
    symmetric: bool,
    hermitian: bool,
    unit: bool,
) -> R {
    let mut norm = Reduction::new(kind, n, n);
    let mut index = 0;
    for j in 0..n {
        let rows = if u == b'U' { 0..j + 1 } else { j..n };
        for i in rows {
            let value = if unit && i == j {
                T::o()
            } else {
                stored(a, index, i == j, hermitian)
            };
            norm.push(i, j, value);
            if symmetric && i != j {
                norm.push(j, i, value);
            }
            index += 1;
        }
    }
    norm.finish()
}
unsafe fn triangular<T: GeneralizedScalar + ReflectorScalar<Real = R>, R: Scalar + RealField>(
    kind: Kind,
    u: u8,
    unit: bool,
    m: usize,
    n: usize,
    a: *const T,
    ld: isize,
) -> R {
    if m == 0 || n == 0 {
        return R::z();
    }
    let mut norm = Reduction::new(kind, m, n);
    for j in 0..n {
        let rows = if u == b'U' {
            0..m.min(j.saturating_add(1))
        } else {
            j.min(m)..m
        };
        for i in rows {
            let value = if unit && i == j {
                T::o()
            } else {
                *a.offset(i as isize + j as isize * ld)
            };
            norm.push(i, j, value);
        }
    }
    norm.finish()
}
unsafe fn rfp<T: GeneralizedScalar + ReflectorScalar<Real = R>, R: Scalar + RealField>(
    kind: Kind,
    u: u8,
    n: usize,
    a: *const T,
    transposed: bool,
    hermitian: bool,
) -> R {
    let mut norm = Reduction::new(kind, n, n);
    visit_rfp(n, u == b'L', transposed, |i, j, index| {
        let value = stored(a, index as isize, i == j, hermitian);
        norm.push(i, j, value);
        if i != j {
            norm.push(j, i, value);
        }
    });
    norm.finish()
}
use crate::kernels::{diag, flag, uplo};
use paste::paste;
macro_rules! check {($($arg:expr=>$valid:expr),+)=>{$(if !$valid{crate::invalid($arg);return 0.0;})+};}
macro_rules! norm_kind {
    ($pointer:expr) => {
        match Kind::parse(flag($pointer)) {
            Some(kind) => kind,
            None => {
                crate::invalid(1);
                return 0.0;
            }
        }
    };
}
macro_rules! symmetric_exports {
    ($p:ident,$t:ty,$r:ty,$i:ty,$suffix:tt,$dense:ident,$band:ident,$packed:ident,$h:expr)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $dense $suffix>](norm:*const u8,u:*const u8,n:*const $i,
            a:*const $t,lda:*const $i,_work:*mut $r)->$r {
            let kind=norm_kind!(norm);let(u,n,ld)=(flag(u),*n as isize,*lda as isize);
            check!(2=>uplo(u),3=>n>=0,5=>ld>=n.max(1));
            symmetric(kind,u,n as usize,a,ld,$h)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $band $suffix>](norm:*const u8,u:*const u8,n:*const $i,k:*const $i,
            a:*const $t,lda:*const $i,_work:*mut $r)->$r {
            let kind=norm_kind!(norm);let(u,n,k,ld)=(flag(u),*n as isize,*k as isize,*lda as isize);
            check!(2=>uplo(u),3=>n>=0,4=>k>=0,6=>ld>=k.saturating_add(1));
            selected_band(kind,u,n as usize,k as usize,a,ld,true,$h,false)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $packed $suffix>](norm:*const u8,u:*const u8,n:*const $i,
            a:*const $t,_work:*mut $r)->$r {
            let kind=norm_kind!(norm);let(u,n)=(flag(u),*n as isize);
            check!(2=>uplo(u),3=>n>=0);packed(kind,u,n as usize,a,true,$h,false)
        }
    }};
}
macro_rules! common_exports {
    ($p:ident,$t:ty,$r:ty,$i:ty,$suffix:tt)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p lange $suffix>](norm:*const u8,m:*const $i,n:*const $i,
            a:*const $t,lda:*const $i,_work:*mut $r)->$r {
            let kind=norm_kind!(norm);let(m,n,ld)=(*m as isize,*n as isize,*lda as isize);
            check!(2=>m>=0,3=>n>=0,5=>ld>=m.max(1));dense(kind,m as usize,n as usize,a,ld,false)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p lanhs $suffix>](norm:*const u8,n:*const $i,a:*const $t,
            lda:*const $i,_work:*mut $r)->$r {
            let kind=norm_kind!(norm);let(n,ld)=(*n as isize,*lda as isize);
            check!(2=>n>=0,4=>ld>=n.max(1));dense(kind,n as usize,n as usize,a,ld,true)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p langb $suffix>](norm:*const u8,n:*const $i,kl:*const $i,ku:*const $i,
            a:*const $t,lda:*const $i,_work:*mut $r)->$r {
            let kind=norm_kind!(norm);let(n,kl,ku,ld)=(*n as isize,*kl as isize,*ku as isize,*lda as isize);
            check!(2=>n>=0,3=>kl>=0,4=>ku>=0,6=>ld>=kl.saturating_add(ku).saturating_add(1));
            band(kind,n as usize,kl as usize,ku as usize,a,ld)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p langt $suffix>](norm:*const u8,n:*const $i,dl:*const $t,
            d:*const $t,du:*const $t)->$r {
            let kind=norm_kind!(norm);let n=*n as isize;check!(2=>n>=0);
            let mut result=Reduction::<$t,$r>::new(kind,n as usize,n as usize);
            for j in 0..n {result.push(j as usize,j as usize,*d.offset(j));
                if j+1<n {result.push((j+1)as usize,j as usize,*dl.offset(j));
                    result.push(j as usize,(j+1)as usize,*du.offset(j));}}
            result.finish()
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p lantr $suffix>](norm:*const u8,u:*const u8,diagonal:*const u8,
            m:*const $i,n:*const $i,a:*const $t,lda:*const $i,_work:*mut $r)->$r {
            let kind=norm_kind!(norm);let(u,d,m,n,ld)=(flag(u),flag(diagonal),*m as isize,*n as isize,*lda as isize);
            check!(2=>uplo(u),3=>diag(d),4=>m>=0,5=>n>=0,7=>ld>=m.max(1));
            triangular(kind,u,d==b'U',m as usize,n as usize,a,ld)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p lantb $suffix>](norm:*const u8,u:*const u8,diagonal:*const u8,
            n:*const $i,k:*const $i,a:*const $t,lda:*const $i,_work:*mut $r)->$r {
            let kind=norm_kind!(norm);let(u,d,n,k,ld)=(flag(u),flag(diagonal),*n as isize,*k as isize,*lda as isize);
            check!(2=>uplo(u),3=>diag(d),4=>n>=0,5=>k>=0,7=>ld>=k.saturating_add(1));
            selected_band(kind,u,n as usize,k as usize,a,ld,false,false,d==b'U')
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p lantp $suffix>](norm:*const u8,u:*const u8,diagonal:*const u8,
            n:*const $i,a:*const $t,_work:*mut $r)->$r {
            let kind=norm_kind!(norm);let(u,d,n)=(flag(u),flag(diagonal),*n as isize);
            check!(2=>uplo(u),3=>diag(d),4=>n>=0);
            packed(kind,u,n as usize,a,false,false,d==b'U')
        }
    }
    symmetric_exports!($p,$t,$r,$i,$suffix,lansy,lansb,lansp,false);
    };
}
macro_rules! extra_exports {
    ($p:ident,$t:ty,$r:ty,$i:ty,$suffix:tt,$rfp:ident,$tri:ident,$h:expr)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $rfp $suffix>](norm:*const u8,transr:*const u8,u:*const u8,
            n:*const $i,a:*const $t,_work:*mut $r)->$r {
            let kind=norm_kind!(norm);let(tr,u,n)=(flag(transr),flag(u),*n as isize);
            check!(2=>(tr==b'N'||tr==if <$t>::COMPLEX {b'C'}else{b'T'}),3=>uplo(u),4=>n>=0);
            rfp(kind,u,n as usize,a,tr!=b'N',$h)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $tri $suffix>](norm:*const u8,n:*const $i,d:*const $r,e:*const $t)->$r {
            let kind=norm_kind!(norm);let n=*n as isize;check!(2=>n>=0);
            let mut result=Reduction::<$t,$r>::new(kind,n as usize,n as usize);
            for j in 0..n {result.push(j as usize,j as usize,<$t>::real(*d.offset(j)as f64));
                if j+1<n {let value=*e.offset(j);result.push((j+1)as usize,j as usize,value);
                    result.push(j as usize,(j+1)as usize,value);}}
            result.finish()
        }
    }};
}
common_exports!(s, f32, f32, i32, _);
common_exports!(d, f64, f64, i32, _);
common_exports!(c, colmat::c32, f32, i32, _);
common_exports!(z, colmat::c64, f64, i32, _);
common_exports!(s, f32, f32, i64, _64_);
common_exports!(d, f64, f64, i64, _64_);
common_exports!(c, colmat::c32, f32, i64, _64_);
common_exports!(z, colmat::c64, f64, i64, _64_);
extra_exports!(s, f32, f32, i32, _, lansf, lanst, false);
extra_exports!(d, f64, f64, i32, _, lansf, lanst, false);
extra_exports!(s, f32, f32, i64, _64_, lansf, lanst, false);
extra_exports!(d, f64, f64, i64, _64_, lansf, lanst, false);
extra_exports!(c, colmat::c32, f32, i32, _, lanhf, lanht, true);
extra_exports!(z, colmat::c64, f64, i32, _, lanhf, lanht, true);
extra_exports!(c, colmat::c32, f32, i64, _64_, lanhf, lanht, true);
extra_exports!(z, colmat::c64, f64, i64, _64_, lanhf, lanht, true);
symmetric_exports!(c, colmat::c32, f32, i32, _, lanhe, lanhb, lanhp, true);
symmetric_exports!(z, colmat::c64, f64, i32, _, lanhe, lanhb, lanhp, true);
symmetric_exports!(c, colmat::c32, f32, i64, _64_, lanhe, lanhb, lanhp, true);
symmetric_exports!(z, colmat::c64, f64, i64, _64_, lanhe, lanhb, lanhp, true);
