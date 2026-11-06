use faer::diag::Diag;
use faer::traits::{math_utils::*, ComplexField};
use faer::{c32, c64, Col, Mat, Row};

fn control<T: ComplexField>() {
    let mut a = Mat::<T>::with_capacity(11, 13);
    assert_eq!(a.shape(), (0, 0));
    a.resize_with(3, 4, |_, _| from_f64::<T>(2.0));
    a.try_reserve(17, 19).unwrap();
    let pointer = a.as_ptr();
    a.truncate(1, 2);
    assert_eq!(a.shape(), (1, 2));
    a.resize_with(17, 19, |_, _| from_f64::<T>(3.0));
    assert_eq!(a.as_ptr(), pointer);
    assert_eq!(a[(0, 0)], from_f64::<T>(2.0));
    assert_eq!(a[(16, 18)], from_f64::<T>(3.0));
    let mut column = Col::<T>::full(3, from_f64::<T>(2.0));
    column.try_reserve(17).unwrap();
    column.truncate(1);
    column.resize_with(17, |_| from_f64::<T>(3.0));
    assert_eq!(column.nrows(), 17);
    let mut row = Row::<T>::full(3, from_f64::<T>(2.0));
    row.try_reserve(19).unwrap();
    row.truncate(1);
    row.resize_with(19, |_| from_f64::<T>(3.0));
    assert_eq!(row.ncols(), 19);
    let diagonal = Diag::<T>::zeros(5);
    assert_eq!(diagonal.dim(), 5);
    #[cfg(feature = "probe-capacity")]
    {
        let _: usize = a.row_capacity();
        let _: usize = a.col_capacity();
        let _: usize = column.capacity();
        let _: usize = row.capacity();
        let _: usize = diagonal.capacity();
    }
    println!(
        "{} reserve/resize/truncate control passed",
        std::any::type_name::<T>()
    );
}
fn main() {
    control::<f32>();
    control::<f64>();
    control::<c32>();
    control::<c64>();
}
