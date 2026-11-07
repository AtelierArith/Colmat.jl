//! Dot products, norms, sums, extrema and predicates read in place from
//! native owners, strided/reversed views and borrowed Julia arrays. Inputs are
//! view descriptors decoded by `views::source`; no reference outlives a call.
use crate::guarded;
use crate::statistics::{expected_len, widens, Native, ALL, COLS, NONE, ROWS};
use crate::typed::norm_output;
use crate::views::{source, Borrowed, Spec};
use faer::traits::math_utils::*;
use faer::{ColRef, Conj, Mat, MatRef};
use rustcall_julia_macros::julia;

const SUM: i32 = 0;
const MINIMUM: i32 = 1;
const MAXIMUM: i32 = 2;
const EXTREMA: i32 = 3;
const NORM: i32 = 4;

fn shape(a: &Borrowed<'_>) -> (usize, usize) {
    match a {
        Borrowed::F32(a) => a.shape(),
        Borrowed::F64(a) => a.shape(),
        Borrowed::C32(a) => a.shape(),
        Borrowed::C64(a) => a.shape(),
    }
}

unsafe fn sums<T: Native>(a: MatRef<'_, T>, out: *mut u8, axis: i32) -> Result<(), String> {
    let out = out.cast::<T>();
    let m = a.nrows();
    match axis {
        ALL => out.write(a.sum()),
        ROWS => (0..a.ncols()).for_each(|j| out.add(j).write(a.col(j).sum())),
        COLS => (0..m).for_each(|i| out.add(i).write(a.row(i).sum())),
        NONE => {
            (0..a.ncols()).for_each(|j| (0..m).for_each(|i| out.add(i + j * m).write(a[(i, j)])))
        }
        _ => return Err("unknown reduction axis".into()),
    }
    Ok(())
}

/// Real scalars ordered like Julia's `min`/`max`: NaN wins, `-0.0 < 0.0`.
trait Ordered: Copy {
    fn nan(self) -> bool;
    fn less(self, other: Self) -> bool;
}
macro_rules! ordered {
    ($t:ty) => {
        impl Ordered for $t {
            #[inline(always)]
            fn nan(self) -> bool {
                self.is_nan()
            }
            #[inline(always)]
            fn less(self, other: Self) -> bool {
                self.total_cmp(&other).is_lt()
            }
        }
    };
}
ordered!(f32);
ordered!(f64);

/// Running `(minimum, maximum)` of one slice; the first NaN seen wins both.
struct Extremum<R> {
    lo: R,
    hi: R,
    nan: Option<R>,
}
impl<R: Ordered> Extremum<R> {
    fn new(first: R) -> Self {
        Self {
            lo: first,
            hi: first,
            nan: first.nan().then_some(first),
        }
    }
    #[inline(always)]
    fn push(&mut self, x: R) {
        if x.nan() && self.nan.is_none() {
            self.nan = Some(x);
        }
        if x.less(self.lo) {
            self.lo = x;
        }
        if self.hi.less(x) {
            self.hi = x;
        }
    }
    fn column(&mut self, x: ColRef<'_, R>) {
        match x.try_as_col_major() {
            Some(x) => x.as_slice().iter().for_each(|&x| self.push(x)),
            None => x.iter().for_each(|&x| self.push(x)),
        }
    }
    fn get(&self) -> (R, R) {
        self.nan.map_or((self.lo, self.hi), |x| (x, x))
    }
}
const EMPTY: &str = "reducing over an empty collection is not allowed";

unsafe fn extrema<R: Ordered>(
    a: MatRef<'_, R>,
    out: *mut u8,
    axis: i32,
    code: i32,
) -> Result<(), String> {
    let out = out.cast::<R>();
    let write = |k: usize, (lo, hi): (R, R)| match code {
        MINIMUM => out.add(k).write(lo),
        MAXIMUM => out.add(k).write(hi),
        _ => {
            out.add(2 * k).write(lo);
            out.add(2 * k + 1).write(hi);
        }
    };
    let (m, n) = a.shape();
    match axis {
        ALL => {
            if m == 0 || n == 0 {
                return Err(EMPTY.into());
            }
            let mut acc = Extremum::new(a[(0, 0)]);
            (0..n).for_each(|j| acc.column(a.col(j)));
            write(0, acc.get());
        }
        ROWS => {
            if m == 0 && n > 0 {
                return Err(EMPTY.into());
            }
            for j in 0..n {
                let mut acc = Extremum::new(a[(0, j)]);
                acc.column(a.col(j));
                write(j, acc.get());
            }
        }
        COLS => {
            if n == 0 && m > 0 {
                return Err(EMPTY.into());
            }
            for i in 0..m {
                let mut acc = Extremum::new(a[(i, 0)]);
                acc.column(a.row(i).transpose());
                write(i, acc.get());
            }
        }
        NONE => (0..n)
            .for_each(|j| (0..m).for_each(|i| write(i + j * m, Extremum::new(a[(i, j)]).get()))),
        _ => return Err("unknown reduction axis".into()),
    }
    Ok(())
}

/// `LinearAlgebra.norm(A, p)` for every `p` but NaN, as Julia computes it:
/// p = 0 counts nonzeros, ±Inf propagate NaN, other p follow `generic_normp`
/// (accumulation in Float64, rescaling by the extreme magnitude).
unsafe fn norm<T: Native>(a: MatRef<'_, T>, p: f64, out: *mut u8) -> Result<(), String> {
    if a.nrows() == 0 || a.ncols() == 0 {
        out.cast::<T::Real>().write(T::real_from_f64(0.0));
        return Ok(());
    }
    if p == 2.0 || p == 1.0 || p == f64::INFINITY {
        let code = if p == 2.0 {
            0
        } else if p == 1.0 {
            1
        } else {
            2
        };
        return norm_output(a, out, code);
    }
    let len = (a.nrows() * a.ncols()) as f64;
    // Entries in column order, through contiguous slices when possible.
    let each = |f: &mut dyn FnMut(&T)| {
        let a = if a.row_stride() < 0 {
            a.reverse_rows()
        } else {
            a
        };
        for j in 0..a.ncols() {
            let c = a.col(j);
            match c.try_as_col_major() {
                Some(c) => c.as_slice().iter().for_each(&mut *f),
                None => c.iter().for_each(&mut *f),
            }
        }
    };
    let fold = |init: f64, g: &dyn Fn(f64, f64) -> f64| {
        let mut acc = init;
        each(&mut |x| acc = g(acc, T::real_to_f64(&abs(x))));
        acc
    };
    // mapreduce(norm, min/max): NaN propagates from any position.
    let extreme = |largest: bool| {
        let init = if largest {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
        fold(init, &|v, x| {
            if v.is_nan() || x.is_nan() {
                f64::NAN
            } else if largest {
                v.max(x)
            } else {
                v.min(x)
            }
        })
    };
    let value = if p == 0.0 {
        let mut count = 0usize;
        each(&mut |x| count += usize::from(x.parts() != (0.0, 0.0)));
        count as f64
    } else if p == f64::NEG_INFINITY {
        extreme(false)
    } else {
        let mut maxabs = f64::NAN;
        if p > 1.0 || p < -1.0 {
            maxabs = extreme(p > 1.0);
            if maxabs == 0.0 || maxabs.is_infinite() {
                out.cast::<T::Real>().write(T::real_from_f64(maxabs));
                return Ok(());
            }
        }
        let scaled = maxabs.powf(p);
        if (-1.0..=1.0).contains(&p) || ((len * scaled).is_finite() && scaled != 0.0) {
            fold(0.0, &|s, x| s + x.powf(p)).powf(1.0 / p)
        } else {
            // norm(v)/maxabs is computed in the data's real type.
            let sum = fold(0.0, &|s, x| {
                s + T::real_to_f64(&T::real_from_f64(x / maxabs)).powf(p)
            });
            maxabs * sum.powf(1.0 / p)
        }
    };
    out.cast::<T::Real>().write(T::real_from_f64(value));
    Ok(())
}

/// Sum (`T`), minimum/maximum (`T`), extrema (`(T, T)` pairs, real only) along
/// `axis`, or the `p`-norm (`T::Real`, full reduction only) of a native owner,
/// view or borrowed Julia array. `output_len` counts output slots.
#[julia]
pub fn view_reduce(
    owner: *const u8,
    desc: *const isize,
    output: *mut u8,
    output_len: usize,
    code: i32,
    axis: i32,
    p: f64,
) -> Result<(), String> {
    guarded(|| unsafe {
        let spec = Spec::read(desc)?;
        if spec.conjugate {
            return Err("reductions do not accept conjugated views".into());
        }
        let a = source(owner, spec)?;
        let (m, n) = shape(&a);
        if output_len != expected_len(m, n, axis)? {
            return Err("reduction output length mismatch".into());
        }
        if code == NORM && axis != ALL {
            return Err("norms reduce the whole array".into());
        }
        match (code, a) {
            (SUM, Borrowed::F32(a)) => sums(a, output, axis),
            (SUM, Borrowed::F64(a)) => sums(a, output, axis),
            (SUM, Borrowed::C32(a)) => sums(a, output, axis),
            (SUM, Borrowed::C64(a)) => sums(a, output, axis),
            (MINIMUM..=EXTREMA, Borrowed::F32(a)) => extrema(a, output, axis, code),
            (MINIMUM..=EXTREMA, Borrowed::F64(a)) => extrema(a, output, axis, code),
            (MINIMUM..=EXTREMA, _) => Err("extrema require real scalars".into()),
            (NORM, Borrowed::F32(a)) => norm(a, p, output),
            (NORM, Borrowed::F64(a)) => norm(a, p, output),
            (NORM, Borrowed::C32(a)) => norm(a, p, output),
            (NORM, Borrowed::C64(a)) => norm(a, p, output),
            _ => Err("unknown reduction".into()),
        }
    })
}

/// `all(isfinite, A)` (code 0) or `any(isnan, A)` (code 1).
#[julia]
pub fn view_predicate(owner: *const u8, desc: *const isize, code: i32) -> Result<bool, String> {
    guarded(|| unsafe {
        let a = source(owner, Spec::read(desc)?)?;
        macro_rules! test {
            ($a:expr) => {
                match code {
                    0 => Ok($a.is_all_finite()),
                    1 => Ok($a.has_nan()),
                    _ => Err("unknown predicate".into()),
                }
            };
        }
        match a {
            Borrowed::F32(a) => test!(a),
            Borrowed::F64(a) => test!(a),
            Borrowed::C32(a) => test!(a),
            Borrowed::C64(a) => test!(a),
        }
    })
}

enum Staged<'a, W> {
    Borrowed(MatRef<'a, W>),
    Converted(Mat<W>),
}
impl<W> Staged<'_, W> {
    fn get(&self) -> MatRef<'_, W> {
        match self {
            Staged::Borrowed(a) => *a,
            Staged::Converted(a) => a.as_ref(),
        }
    }
}
/// Borrow `a` as `W` in place, or stage one converted copy when promoted.
unsafe fn staged<'a, S: Native, W: Native>(a: MatRef<'a, S>) -> Result<Staged<'a, W>, String> {
    if S::TAG == W::TAG {
        return Ok(Staged::Borrowed(MatRef::from_raw_parts(
            a.as_ptr().cast::<W>(),
            a.nrows(),
            a.ncols(),
            a.row_stride(),
            a.col_stride(),
        )));
    }
    if !widens(S::TAG, W::TAG) {
        return Err("dot working scalar must promote both operands".into());
    }
    Ok(Staged::Converted(Mat::from_fn(
        a.nrows(),
        a.ncols(),
        |i, j| {
            let (re, im) = a[(i, j)].parts();
            W::from_parts(re, im)
        },
    )))
}
unsafe fn stage<'a, W: Native>(a: Borrowed<'a>) -> Result<Staged<'a, W>, String> {
    match a {
        Borrowed::F32(a) => staged(a),
        Borrowed::F64(a) => staged(a),
        Borrowed::C32(a) => staged(a),
        Borrowed::C64(a) => staged(a),
    }
}
/// A vector-shaped (`k x 1` or `1 x k`) operand as a column.
fn column<W>(a: MatRef<'_, W>) -> Option<ColRef<'_, W>> {
    if a.ncols() == 1 {
        Some(a.col(0))
    } else if a.nrows() == 1 {
        Some(a.row(0).transpose())
    } else {
        None
    }
}
fn conj_flag(yes: bool) -> Conj {
    if yes {
        Conj::Yes
    } else {
        Conj::No
    }
}
/// Julia's `dot(x, y)`: `sum(conj(x[i]) * y[i])` in column-major order, with
/// faer's `inner_prod` per column.
fn dot<W: Native>(
    a: MatRef<'_, W>,
    conj_a: bool,
    b: MatRef<'_, W>,
    conj_b: bool,
) -> Result<W, String> {
    let (ca, cb) = (conj_flag(conj_a), conj_flag(conj_b));
    if let (Some(x), Some(y)) = (column(a), column(b)) {
        if x.nrows() != y.nrows() {
            return Err("dot product arguments must have the same length".into());
        }
        return Ok(faer::linalg::matmul::dot::inner_prod(
            x.transpose(),
            ca,
            y,
            cb,
        ));
    }
    if a.shape() != b.shape() {
        return Err("dot product arguments must have the same shape".into());
    }
    Ok((0..a.ncols()).fold(zero::<W>(), |s, j| {
        add(
            &s,
            &faer::linalg::matmul::dot::inner_prod(a.col(j).transpose(), ca, b.col(j), cb),
        )
    }))
}
unsafe fn dot_into<W: Native>(
    a: Borrowed<'_>,
    conj_a: bool,
    b: Borrowed<'_>,
    conj_b: bool,
    out: *mut u8,
) -> Result<(), String> {
    let (a, b) = (stage::<W>(a)?, stage::<W>(b)?);
    out.cast::<W>()
        .write(dot(a.get(), conj_a, b.get(), conj_b)?);
    Ok(())
}

/// `dot(x, y)` of two native owners, views or borrowed Julia arrays in the
/// working scalar `work`, conjugating `x`. An operand whose scalar `work`
/// promotes is staged once; other operands are read in place.
#[julia]
pub fn view_dot(
    lhs: *const u8,
    lhs_desc: *const isize,
    rhs: *const u8,
    rhs_desc: *const isize,
    output: *mut u8,
    work: i32,
) -> Result<(), String> {
    guarded(|| unsafe {
        let (sa, sb) = (Spec::read(lhs_desc)?, Spec::read(rhs_desc)?);
        let (a, b) = (source(lhs, sa)?, source(rhs, sb)?);
        let (ca, cb) = (!sa.conjugate, sb.conjugate);
        match work {
            0 => dot_into::<f32>(a, ca, b, cb, output),
            1 => dot_into::<f64>(a, ca, b, cb, output),
            2 => dot_into::<faer::c32>(a, ca, b, cb, output),
            3 => dot_into::<faer::c64>(a, ca, b, cb, output),
            _ => Err("unsupported dot scalar".into()),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    fn reduce<T: Native>(a: MatRef<'_, T>, code: i32, axis: i32) -> Vec<T> {
        let len =
            expected_len(a.nrows(), a.ncols(), axis).unwrap() * if code == EXTREMA { 2 } else { 1 };
        let mut out = vec![T::from_parts(0.0, 0.0); len];
        unsafe {
            match code {
                SUM => sums(a, out.as_mut_ptr().cast(), axis).unwrap(),
                _ => unreachable!(),
            }
        }
        out
    }
    fn extreme(a: MatRef<'_, f64>, code: i32, axis: i32) -> Result<Vec<f64>, String> {
        let len = expected_len(a.nrows(), a.ncols(), axis)? * if code == EXTREMA { 2 } else { 1 };
        let mut out = vec![0.0; len];
        unsafe { extrema(a, out.as_mut_ptr().cast(), axis, code)? };
        Ok(out)
    }
    fn normp<T: Native>(a: MatRef<'_, T>, p: f64) -> T::Real {
        let mut out = std::mem::MaybeUninit::<T::Real>::uninit();
        unsafe {
            norm(a, p, out.as_mut_ptr().cast()).unwrap();
            out.assume_init()
        }
    }

    #[test]
    fn axis_sums_follow_slice_order_on_reversed_views() {
        let a = Mat::<f64>::from_fn(3, 2, |i, j| (1 + i + 10 * j) as f64);
        let r = a.as_ref().reverse_rows_and_cols();
        assert_eq!(reduce(r, SUM, ROWS), vec![36.0, 6.0]);
        assert_eq!(reduce(r, SUM, COLS), vec![16.0, 14.0, 12.0]);
        assert_eq!(reduce(r, SUM, ALL), vec![42.0]);
        assert_eq!(reduce(r, SUM, NONE), vec![13.0, 12.0, 11.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn extrema_match_julia_min_max() {
        let a = Mat::<f64>::from_fn(2, 2, |i, j| [[0.0, -0.0], [1.0, f64::NAN]][j][i]);
        let lo = extreme(a.as_ref(), MINIMUM, ROWS).unwrap();
        assert!(lo[0].is_sign_negative() && lo[0] == 0.0);
        assert!(lo[1].is_nan());
        let both = extreme(a.as_ref(), EXTREMA, COLS).unwrap();
        assert_eq!(both[0], 0.0);
        assert!(!both[1].is_sign_negative());
        assert!(both[2].is_nan() && both[3].is_nan());
        assert!(extreme(Mat::<f64>::zeros(0, 2).as_ref(), MAXIMUM, ROWS).is_err());
        assert_eq!(
            extreme(Mat::<f64>::zeros(2, 0).as_ref(), MAXIMUM, ROWS).unwrap(),
            vec![]
        );
    }

    #[test]
    fn general_norms_follow_generic_normp() {
        let a = Mat::<f64>::from_fn(3, 1, |i, _| [3.0, -4.0, 0.0][i]);
        let a = a.as_ref();
        assert!((normp(a, 3.0) - 91f64.powf(1.0 / 3.0)).abs() < 1e-14);
        assert_eq!(normp(a, 0.0), 2.0);
        assert_eq!(normp(a, f64::NEG_INFINITY), 0.0);
        assert_eq!(normp(a, -2.0), 0.0);
        assert!((normp(a, 0.5) - (3f64.sqrt() + 2.0).powi(2)).abs() < 1e-12);
        // Rescaling keeps large magnitudes finite.
        let big = Mat::<f32>::from_fn(2, 1, |_, _| 2.0e38);
        assert!((normp(big.as_ref(), 3.0) - 2.0e38 * 2f32.powf(1.0 / 3.0)).abs() < 1e32);
        let nan = Mat::<f64>::from_fn(2, 1, |i, _| [1.0, f64::NAN][i]);
        for p in [3.0, -3.0, f64::NEG_INFINITY, 0.5] {
            assert!(normp(nan.as_ref(), p).is_nan());
        }
        assert_eq!(normp(nan.as_ref(), 0.0), 2.0);
        let z = Mat::<faer::c64>::from_fn(1, 1, |_, _| faer::c64::new(3.0, 4.0));
        assert!((normp(z.as_ref(), 7.0) - 5.0).abs() < 1e-14);
    }

    #[test]
    fn dot_conjugates_the_first_argument_across_layouts() {
        let x = Mat::<faer::c64>::from_fn(3, 1, |i, _| faer::c64::new(i as f64, 1.0));
        let y = Mat::<faer::c64>::from_fn(1, 3, |_, j| faer::c64::new(1.0, j as f64));
        let expected = (0..3).fold(faer::c64::new(0.0, 0.0), |s, i| {
            s + faer::c64::new(i as f64, 1.0).conj() * faer::c64::new(1.0, (2 - i) as f64)
        });
        let yr = y.as_ref().reverse_cols();
        assert_eq!(dot(x.as_ref(), true, yr, false).unwrap(), expected);
        assert!(dot(
            x.as_ref(),
            true,
            Mat::<faer::c64>::zeros(2, 1).as_ref(),
            false
        )
        .is_err());
    }

    // Rust-side counts: same-type kernels read every layout in place; a
    // promoted dot operand is staged once.
    #[test]
    fn allocations_are_limited_to_promotion_staging() {
        use crate::allocation_probe::count;
        let a = Mat::<f64>::from_fn(64, 48, |i, j| (i * 3 + j) as f64 - 50.0);
        let reversed = a.as_ref().reverse_rows().reverse_cols();
        let strided = a.as_ref().subrows(0, 32).subcols(0, 24);
        let mut out = vec![0.0f64; 2 * 64 * 48];
        let out = out.as_mut_ptr().cast::<u8>();
        for view in [a.as_ref(), reversed, strided] {
            for axis in [NONE, ROWS, COLS, ALL] {
                for code in [MINIMUM, MAXIMUM, EXTREMA] {
                    assert_eq!(
                        count(|| unsafe { extrema(view, out, axis, code).unwrap() }),
                        0
                    );
                }
                assert_eq!(count(|| unsafe { sums(view, out, axis).unwrap() }), 0);
            }
            for p in [
                1.0,
                2.0,
                f64::INFINITY,
                0.0,
                f64::NEG_INFINITY,
                3.0,
                0.5,
                -2.5,
            ] {
                assert_eq!(count(|| unsafe { norm(view, p, out).unwrap() }), 0);
            }
            assert_eq!(
                count(|| assert!(view.is_all_finite() && !view.has_nan())),
                0
            );
            let row = view.row(0).as_mat();
            assert_eq!(
                count(|| unsafe {
                    dot_into::<f64>(Borrowed::F64(view), true, Borrowed::F64(view), false, out)
                        .unwrap()
                }),
                0
            );
            assert_eq!(
                count(|| unsafe {
                    dot_into::<f64>(
                        Borrowed::F64(row),
                        true,
                        Borrowed::F64(view.row(1).transpose().as_mat()),
                        false,
                        out,
                    )
                    .unwrap()
                }),
                0
            );
            let narrow =
                Mat::<f32>::from_fn(view.nrows(), view.ncols(), |i, j| view[(i, j)] as f32);
            assert_eq!(
                count(|| unsafe {
                    dot_into::<f64>(
                        Borrowed::F32(narrow.as_ref()),
                        true,
                        Borrowed::F64(view),
                        false,
                        out,
                    )
                    .unwrap()
                }),
                1
            );
        }
    }
}
