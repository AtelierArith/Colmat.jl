use faer::{c64, mat, Col, Mat};
fn main() {
    let a = mat![[1.0f64, f64::NAN]];
    let v = Col::<f64>::from_fn(2, |i| [1.0, f64::NAN][i]);
    let z = Col::<c64>::from_fn(2, |i| [c64::new(1.0, 0.0), c64::new(f64::NAN, 0.0)][i]);
    let zi = Col::<c64>::from_fn(2, |i| [c64::new(1.0, 0.0), c64::new(0.0, f64::NAN)][i]);
    let m = Mat::<f64>::from_fn(2, 3, |i, j| if (i, j) == (1, 1) { f64::NAN } else { 1.0 });
    let only = Col::<f64>::from_fn(1, |_| f64::NAN);
    let big = Col::<f64>::from_fn(100, |i| if i == 57 { f64::NAN } else { i as f64 });
    println!("{} {} {} {}", a.norm_max(), v.norm_max(), v.norm_l1(), v.norm_l2());
    println!("{} {}", z.norm_max(), zi.norm_max());
    println!("{} {}", m.row(1).norm_max(), m.norm_max());
    println!("{} {}", only.norm_max(), big.norm_max());
}
