# faer complex magnitude underflow

Reported in https://github.com/tensor4all/faer/issues/274.

- [x] I have read the [fork review rules](https://github.com/tensor4all/faer/blob/main/docs/review-rules.md).

## Observed failure

`faer_traits::math_utils::abs` returns zero for nonzero subnormal `c32` and `c64` values even when the magnitude is exactly representable. Pure-imaginary inputs make the oracle exact: `abs(0 + i*x)` must be `abs(x)`. The standard real `hypot` control returns that nonzero value.

Executed on clean fork HEAD `44945d7c200f7a3cc8ae995a6cc82c82b8678e73` (t4a-faer 0.30.0, t4a-faer-traits 0.24.2), checked against the remote HEAD on 2026-11-06 JST. Linux x86_64, `rustc 1.99.1 (58cf8f9c3 2026-10-13)`, debug profile, `default-features=false, features=["std"]`. The Git-pinned manifest below was executed; no Julia, RustCall or numerical matrix kernel is involved.

## Complete reproduction

Create `Cargo.toml`:
```toml
[package]
name = "faer-traits-abs-mwe"
version = "0.1.0"
edition = "2021"
[dependencies]
faer_traits = { package = "t4a-faer-traits", git = "https://github.com/tensor4all/faer.git", rev = "44945d7c200f7a3cc8ae995a6cc82c82b8678e73", default-features = false, features = ["std"] }
```

Create `src/main.rs`:
```rust
use faer_traits::{c32,c64,math_utils::abs};
fn main() {
 let mut bad=false;
 for x in [f32::from_bits(1 << 19),f32::from_bits(1 << 9)] {
  let z=c32::new(0.0,x); let actual=abs(&z); let expected=z.re.hypot(z.im);
  println!("c32 x={x:e} faer_abs={actual:e} hypot={expected:e} ratio={}",actual/x);
  bad |= actual!=expected;
 }
 for x in [f64::from_bits(1 << 44),f64::from_bits(1 << 4)] {
  let z=c64::new(0.0,x); let actual=abs(&z); let expected=z.re.hypot(z.im);
  println!("c64 x={x:e} faer_abs={actual:e} hypot={expected:e} ratio={}",actual/x);
  bad |= actual!=expected;
 }
 assert!(!bad,"faer complex magnitude disagrees with the exactly representable pure-axis magnitude");
}
```

Run from that directory:
```sh
cargo run --manifest-path Cargo.toml
```

Actual output (exit 101 from the final assertion):
```text
c32 x=7.34684e-40 faer_abs=7.34684e-40 hypot=7.34684e-40 ratio=1
c32 x=7.17e-43 faer_abs=0e0 hypot=7.17e-43 ratio=0
c64 x=8.691694759794e-311 faer_abs=8.691694759794e-311 hypot=8.691694759794e-311 ratio=1
c64 x=8e-323 faer_abs=0e0 hypot=8e-323 ratio=0
```

The inputs are constructed with `from_bits`, so no power/exponent calculation can underflow before the function is called. The first value for each type is a subnormal control that still succeeds; the smaller value returns zero. All four expected magnitudes are representable and nonzero.

## Cause and proposed correction

The private [`abs_impl`](https://github.com/tensor4all/faer/blob/44945d7c200f7a3cc8ae995a6cc82c82b8678e73/faer-traits/src/lib.rs#L337) scales small components by the fixed `sqrt_max_positive`, then squares them. For the deepest subnormals, that lift is insufficient: both squares underflow to zero before the final rescaling. The public `ComplexField` implementation routes its magnitude through this function.

Proposed, not implemented or benchmarked here: normalize finite nonzero components by their own maximum magnitude and evaluate `hi * sqrt(1 + (lo/hi)^2)`, preserving the existing NaN/Inf policy explicitly. This avoids premature square underflow while keeping generic `RealField` support. An equivalent verified full-range hypot implementation is also appropriate. Review the scalar hot path and applicable performance acceptance before adopting a fix; this report supplies correctness evidence only.

Regression tests should cover both complex precisions, pure real/pure imaginary axes including the exact `from_bits` cases, mixed components, zero, normal/subnormal transitions and large finite values. Compare with `f32/f64::hypot` and the bound `magnitude >= max(abs(re),abs(im))` whenever the mathematical result is representable.

This is separate from #74's triangular-solve reciprocal issue. Native Householder also has a deliberate `head_norm < min_positive` flush; this report isolates the scalar magnitude behavior and does not claim that correcting `abs_impl` alone changes that separate kernel policy.
