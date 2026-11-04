use rustcall_julia_macros::julia;
#[julia]
pub struct Counter { value: i32 }
#[julia]
impl Counter {
    #[julia]
    pub fn new() -> Self { Self { value: 42 } }
}
#[julia]
pub struct Destination { value: i32 }
#[julia]
impl Destination {
    #[julia]
    pub fn new() -> Self { Self { value: 0 } }
}
mod operations;
#[test]
fn rust_answer() { assert_eq!(Destination::new().read(&Counter::new()), 42); }
