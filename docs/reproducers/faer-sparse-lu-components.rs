use faer::sparse::linalg::lu::LuRef;
fn inspect(f: LuRef<'_, usize, f64>) {
    let _ = f.l_factor();
    let _ = f.u_factor();
}
fn main() {}
