//! Numerical adoption control for faer PR #282; this is not a benchmark.
use faer::linalg::solvers::{Llt, PartialPivLu};
use faer::traits::ComplexField;
use faer::{c32, c64, Mat, MatRef, Par, Side};

trait Entry: ComplexField {
    const NAME: &'static str;
    fn at(i: usize, j: usize) -> Self;
    fn sentinel() -> Self;
    fn conjugated(value: &Self) -> Self;
    fn same(a: &Self, b: &Self) -> bool;
    fn spd(n: usize, i: usize, j: usize) -> Self;
}

fn f32_at(i: usize, j: usize) -> f32 {
    match (i + 17 * j) % 19 {
        0 => -0.0,
        1 => f32::from_bits(0x7fc1_2345),
        2 => f32::from_bits(0xffc2_3456),
        _ => (i as f32 + 1.0) * 0.125 - (j as f32 + 1.0) * 0.03125,
    }
}
fn f64_at(i: usize, j: usize) -> f64 {
    match (i + 17 * j) % 19 {
        0 => -0.0,
        1 => f64::from_bits(0x7ff8_0000_0001_2345),
        2 => f64::from_bits(0xfff8_0000_0002_3456),
        _ => (i as f64 + 1.0) * 0.125 - (j as f64 + 1.0) * 0.03125,
    }
}
macro_rules! real_entry {
    ($t:ty, $at:ident, $nan:expr) => {
        impl Entry for $t {
            const NAME: &'static str = stringify!($t);
            fn at(i: usize, j: usize) -> Self {
                $at(i, j)
            }
            fn sentinel() -> Self {
                Self::from_bits($nan)
            }
            fn conjugated(value: &Self) -> Self {
                *value
            }
            fn same(a: &Self, b: &Self) -> bool {
                a.to_bits() == b.to_bits()
            }
            fn spd(n: usize, i: usize, j: usize) -> Self {
                if i == j {
                    n as Self + 3.0
                } else {
                    ((i + j) % 13 + 1) as Self / 512.0
                }
            }
        }
    };
}
real_entry!(f32, f32_at, 0x7fc5_4321);
real_entry!(f64, f64_at, 0x7ff8_0000_0005_4321);
macro_rules! complex_entry {
    ($t:ty, $r:ty, $at:ident) => {
        impl Entry for $t {
            const NAME: &'static str = stringify!($t);
            fn at(i: usize, j: usize) -> Self {
                Self::new($at(i, j), $at(i + 7, j + 13))
            }
            fn sentinel() -> Self {
                Self::new(<$r as Entry>::sentinel(), -0.0)
            }
            fn conjugated(value: &Self) -> Self {
                Self::new(value.re, -value.im)
            }
            fn same(a: &Self, b: &Self) -> bool {
                a.re.to_bits() == b.re.to_bits() && a.im.to_bits() == b.im.to_bits()
            }
            fn spd(n: usize, i: usize, j: usize) -> Self {
                Self::new(<$r as Entry>::spd(n, i, j), (i as $r - j as $r) / 16384.0)
            }
        }
    };
}
complex_entry!(c32, f32, f32_at);
complex_entry!(c64, f64, f64_at);

fn copies<T: Entry>() -> usize {
    let mut cases = 0;
    for (m, n) in [
        (0, 0),
        (0, 3),
        (3, 0),
        (1, 1),
        (2, 3),
        (17, 300),
        (300, 17),
        (64, 64),
        (65, 65),
        (129, 129),
        (100, 257),
        (257, 100),
        (257, 257),
    ] {
        let storage: Vec<T> = (0..m)
            .flat_map(|i| (0..n).map(move |j| T::at(i, j)))
            .collect();
        for reversed in 0..4 {
            let base = if m == 0 || n == 0 {
                0
            } else {
                (if reversed & 1 != 0 { (m - 1) * n } else { 0 })
                    + (if reversed & 2 != 0 { n - 1 } else { 0 })
            };
            let rs = if reversed & 1 != 0 {
                -(n as isize)
            } else {
                n as isize
            };
            let cs = if reversed & 2 != 0 { -1 } else { 1 };
            // SAFETY: signed strides visit exactly the preserved row-major
            // allocation, starting at its corresponding corner; no aliasing
            // destination is constructed. Empty views are never dereferenced.
            let src = unsafe { MatRef::from_raw_parts(storage.as_ptr().add(base), m, n, rs, cs) };
            for conjugated in [false, true] {
                for lower in [false, true] {
                    let mut dst = Mat::from_fn(m, n, |_, _| T::sentinel());
                    match (lower, conjugated) {
                        (true, true) => dst.copy_from_triangular_lower(src.conjugate()),
                        (true, false) => dst.copy_from_triangular_lower(src),
                        (false, true) => dst.copy_from(src.conjugate()),
                        (false, false) => dst.copy_from(src),
                    }
                    for j in 0..n {
                        for i in 0..m {
                            let expected = if lower && i < j {
                                T::sentinel()
                            } else if conjugated {
                                T::conjugated(&src[(i, j)])
                            } else {
                                src[(i, j)].clone()
                            };
                            assert!(T::same(&dst[(i,j)],&expected),
                            "{} {m}x{n} reversed={reversed} conjugated={conjugated} lower={lower} at({i},{j})",T::NAME);
                        }
                    }
                    cases += 1;
                }
            }
        }
    }
    cases
}

fn same<T: Entry>(a: MatRef<'_, T>, b: MatRef<'_, T>, lower: bool) {
    assert_eq!(a.shape(), b.shape());
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            if !lower || i >= j {
                assert!(
                    T::same(&a[(i, j)], &b[(i, j)]),
                    "{} factor differs at({i},{j})",
                    T::NAME
                );
            }
        }
    }
}
fn factors<T: Entry>(mut par: Par<'_>) {
    for n in [65, 129, 257] {
        let cm: Vec<T> = (0..n)
            .flat_map(|j| (0..n).map(move |i| T::spd(n, i, j)))
            .collect();
        let rm: Vec<T> = (0..n)
            .flat_map(|i| (0..n).map(move |j| T::spd(n, i, j)))
            .collect();
        let cm = MatRef::from_column_major_slice(&cm, n, n);
        let rm = MatRef::from_row_major_slice(&rm, n, n);
        let a = Llt::new(cm, Side::Lower, par.reborrow()).unwrap();
        let b = Llt::new(rm, Side::Lower, par.reborrow()).unwrap();
        same(a.L(), b.L(), true);
        let a = PartialPivLu::new(cm, par.reborrow());
        let b = PartialPivLu::new(rm, par.reborrow());
        same(a.L(), b.L(), false);
        same(a.U(), b.U(), false);
        assert_eq!(a.P().arrays(), b.P().arrays());
    }
}
fn check<T: Entry>() {
    let cases = copies::<T>();
    factors::<T>(Par::Seq);
    faer::team::with_team(2, |par| factors::<T>(par));
    println!(
        "{}: {cases} copy cases; 3 sizes x Seq/Team2 Llt/LU bitwise controls passed",
        T::NAME
    );
}
fn main() {
    check::<f32>();
    check::<f64>();
    check::<c32>();
    check::<c64>();
}
