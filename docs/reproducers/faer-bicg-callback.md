# BiCGSTAB terminal callback

Reported as [faer #242](https://github.com/tensor4all/faer/issues/242); fix and
controlled-callback extension: [PR #243](https://github.com/tensor4all/faer/pull/243).

On clean faer main `12843515e07f8d6f27fc525668227088658ffe0f`, Rust 1.99.1,
Linux x86_64, std/linalg without default features, the adjacent standalone Rust
source reports `iter_count=1, callback_count=0, residual=0` and exits 101 at its
callback-count assertion. No Julia or RustCall is involved.

Create a sibling Cargo binary project and copy `faer-bicg-callback.rs` into its
`src/main.rs`. Its complete manifest is:

```toml
[package]
name = "iteration-callback-mwe"
version = "0.1.0"
edition = "2021"
[dependencies]
faer = { package = "t4a-faer", path = "../faer/faer", default-features = false, features = ["std", "linalg"] }
```

Clone tensor4all/faer into `faer`, select the revision above, and run
`cargo run --manifest-path mwe/Cargo.toml`. The successful half-step return in
BiCGSTAB precedes its callback site. The fix invokes the callback on that return;
the expected corrected output is `iter_count=1, callback_count=1, residual=0`.

Candidate `778afacc1a1b723a744a3994bc423c942a70c5bf` was executed with the
same MWE and produced the corrected output with exit status 0.
