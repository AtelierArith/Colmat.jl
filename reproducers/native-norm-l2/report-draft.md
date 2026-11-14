# Unpublished follow-up draft

Submission to tensor4all/t4a-faer#36 failed: the repository is archived and read-only. No comment or new native issue was published. The following retained text is a draft; recheck repository state, permission, evidence and provenance before any future submission.

---

Follow-up to the fix in #46: the moderate-scale strided fallback case is fixed, but the current identical main revision still loses finite, representable norms at more extreme scales, including ordinary contiguous owned matrices. I am adding this executed evidence here rather than opening a duplicate report.

The safe public `Mat::norm_l2()` returns zero or infinity for small constant matrices whose Frobenius norms are finite and representable. This blocks LAPACK Frobenius-norm adapters without an additional normalization step.

For a 7x7 matrix filled with a real value `v`, the exact mathematical norm is `7*abs(v)`. The reproducer covers f32/f64/c32/c64, using the smallest positive subnormal and `MAX/8`. Normal-size positive controls pass; all eight extreme cases fail.

## Executed versus inspected state

- Executed: an independent Rust crate using `tensor4all/colmat-rs` at `e1c83482f5a61449d30e989808c3ba5f751ec391`, with no Julia, RustCall, C ABI, local source patches, or unsafe code.
- Inspected: `tensor4all/faer` (redirects to `tensor4all/t4a-faer`) main is the identical commit `e1c83482f5a61449d30e989808c3ba5f751ec391`; these are separate repository IDs. I have not claimed a separate execution from that repository URL.
- Toolchain: rustc 1.99.1 (`58cf8f9c35b355199727627d07ae34a4e2846d67`, LLVM 23.1.1), x86_64-apple-darwin.
- Features: default features disabled; `std`, `rayon`, `threadteam` enabled, matching the adapter dependency.
- Profile: release. The command exits 101 at the final assertion; it is not a timeout or compilation error.

## Complete reproducer

Create the following files in an empty directory. Cargo.lock is retained with the caller reproduction; the numerical dependency is pinned by the git revision below.

`Cargo.toml`:

```toml
[package]
name = "colmat-native-norm-l2-repro"
version = "0.1.0"
edition = "2021"
publish = false

[dependencies]
colmat = { git = "https://github.com/tensor4all/colmat-rs.git", rev = "e1c83482f5a61449d30e989808c3ba5f751ec391", default-features = false, features = ["std", "rayon", "threadteam"] }

[workspace]
```

`src/main.rs`:

```rust
use colmat::{c32, c64, Mat};

fn main() {
    let tiny32 = f32::from_bits(1);
    let tiny64 = f64::from_bits(1);
    let large32 = f32::MAX / 8.0;
    let large64 = f64::MAX / 8.0;
    let mut failed = 0;
    macro_rules! check {
        ($name:expr,$value:expr,$expected:expr,$tol:expr) => {{
            let actual = $value;
            let expected = $expected;
            let valid = actual.is_finite()
                && actual > 0.0
                && ((actual as f64) / (expected as f64) - 1.0).abs() <= $tol;
            println!(
                "{}: actual={:?}, expected={:?}, valid={}",
                $name, actual, expected, valid
            );
            if !valid {
                failed += 1;
            }
        }};
    }
    // A 7x7 constant matrix has Frobenius norm exactly 7*abs(value).
    // All expected results are finite; large targets use the scalar-rounded value.
    check!(
        "f32 positive control",
        Mat::<f32>::from_fn(7, 7, |_, _| 1.0).norm_l2(),
        7.0f32,
        1e-6
    );
    check!(
        "f64 positive control",
        Mat::<f64>::from_fn(7, 7, |_, _| 1.0).norm_l2(),
        7.0f64,
        1e-14
    );
    check!(
        "f32 subnormal",
        Mat::<f32>::from_fn(7, 7, |_, _| tiny32).norm_l2(),
        7.0 * tiny32,
        1e-6
    );
    check!(
        "f64 subnormal",
        Mat::<f64>::from_fn(7, 7, |_, _| tiny64).norm_l2(),
        7.0 * tiny64,
        1e-14
    );
    check!(
        "f32 large",
        Mat::<f32>::from_fn(7, 7, |_, _| large32).norm_l2(),
        7.0 * large32,
        1e-6
    );
    check!(
        "f64 large",
        Mat::<f64>::from_fn(7, 7, |_, _| large64).norm_l2(),
        7.0 * large64,
        1e-14
    );
    check!(
        "c32 subnormal",
        Mat::<c32>::from_fn(7, 7, |_, _| c32::new(tiny32, 0.0)).norm_l2(),
        7.0 * tiny32,
        1e-6
    );
    check!(
        "c64 subnormal",
        Mat::<c64>::from_fn(7, 7, |_, _| c64::new(tiny64, 0.0)).norm_l2(),
        7.0 * tiny64,
        1e-14
    );
    check!(
        "c32 large",
        Mat::<c32>::from_fn(7, 7, |_, _| c32::new(large32, 0.0)).norm_l2(),
        7.0 * large32,
        1e-6
    );
    check!(
        "c64 large",
        Mat::<c64>::from_fn(7, 7, |_, _| c64::new(large64, 0.0)).norm_l2(),
        7.0 * large64,
        1e-14
    );
    assert_eq!(
        failed, 0,
        "native norm_l2 violates finite representable Frobenius norms"
    );
}
```

Run `cargo run --release` (or `cargo run --release --locked` after retaining the generated lockfile).

Observed output:

```text
f32 positive control: actual=7.0, expected=7.0, valid=true
f64 positive control: actual=7.0, expected=7.0, valid=true
f32 subnormal: actual=0.0, expected=1e-44, valid=false
f64 subnormal: actual=0.0, expected=3.5e-323, valid=false
f32 large: actual=inf, expected=2.9774705e38, valid=false
f64 large: actual=inf, expected=1.5729814930045262e308, valid=false
c32 subnormal: actual=0.0, expected=1e-44, valid=false
c64 subnormal: actual=0.0, expected=3.5e-323, valid=false
c32 large: actual=inf, expected=2.9774705e38, valid=false
c64 large: actual=inf, expected=1.5729814930045262e308, valid=false
```

## Suspected cause and proposed correction

In [norm_l2.rs at the executed revision](https://github.com/tensor4all/faer/blob/e1c83482f5a61449d30e989808c3ba5f751ec391/faer/src/linalg/reductions/norm_l2.rs), `norm_l2_x3` accumulates squared values at three fixed scales and `norm_l2` reconstructs the norm from those accumulators. At these extremes the boosted tiny accumulator still underflows to zero, or the reduced large accumulator overflows although the norm itself fits. The generic-stride fallback also passes its hypot accumulator through squared intermediate representations.

A proposed correction is to normalize by a per-input maximum component (with explicit zero/NaN/Inf semantics) before accumulating squares, or use a range-aware scaled sum-of-squares representation. In particular, changing the final choice among three already-zero/infinite accumulators cannot recover the lost information. This proposal has not been applied to or verified as a fix in the native fork.

Please add bounded real/complex regression cases for these finite targets, including contiguous and generic-stride views. The caller workaround normalizes finite components before native norm_l2 and rescales the result; it retains the pinned dependency and does not resolve this native defect. Caller context: [Colmat.jl #279](https://github.com/AtelierArith/Colmat.jl/issues/279), under [#263](https://github.com/AtelierArith/Colmat.jl/issues/263).

**Provenance** — Prepared by the Codex AI agent using GitHub account `terasakisatoshi` while implementing the human-authorized Colmat.jl #263 goal and the `reporting-faer-bugs` skill. There was no separate human request for this specific report, and its contents were not human-reviewed. Model: GPT-6 (exact deployment identifier unavailable in this runtime). Execution: 2026-11-14 15:29 UTC. Review it before relying on it.
