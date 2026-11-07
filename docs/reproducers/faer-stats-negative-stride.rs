use faer::stats::{row_mean, row_varm, NanHandling};
use faer::{Mat, Row};
fn main() {
    // Column j holds [0, 2(j+1)]: means 1, 2, 3, 4 and variances 2, 8, 18, 32.
    let a = Mat::<f64>::from_fn(2, 4, |i, j| (j + 1) as f64 * if i == 0 { 0.0 } else { 2.0 });
    let r = a.reverse_cols(); // expected means 4, 3, 2, 1
    for nan in [NanHandling::Propagate, NanHandling::Ignore] {
        let mut m = Row::<f64>::zeros(4);
        row_mean(m.as_mut(), r, nan);
        let owned = r.to_owned();
        let mut mo = Row::<f64>::zeros(4);
        row_mean(mo.as_mut(), owned.as_ref(), nan);
        let (mut v, mut vo) = (Row::<f64>::zeros(4), Row::<f64>::zeros(4));
        row_varm(v.as_mut(), r, mo.as_ref(), nan);
        row_varm(vo.as_mut(), owned.as_ref(), mo.as_ref(), nan);
        println!("{nan:?}: mean view {:?} owned {:?}", m, mo);
        println!("{nan:?}: varm view {:?} owned {:?}", v, vo);
    }
}
