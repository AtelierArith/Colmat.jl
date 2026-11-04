use super::*;
#[julia]
impl Destination {
    #[julia]
    pub fn read(&mut self, source: &Counter) -> i32 { self.value = source.value; self.value }
}
