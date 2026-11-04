# LSMR scale-sensitive false convergence

[faer #245](https://github.com/tensor4all/faer/issues/245) reports a standalone
Rust failure on main at reporting time `12843515e07f8d6f27fc525668227088658ffe0f`,
t4a-faer 0.28.2, Rust 1.99.1 / Linux x86_64, std/linalg without default features.
The adjacent source is the executed MWE. It reports success and zero estimated
residual after one update for both `1e-300 I` and `1e300 I`, but returns Inf and
zero respectively; true dense residuals are NaN and 2. The final assertion exits
101. The finite reference `b/scale` passes its independent residual check.

Create a Cargo binary beside a checkout named `faer`, copy
`faer-lsmr-scale.rs` to `src/main.rs`, and run `cargo run`. Complete manifest:

```toml
[package]
name = "lsmr-scale-mwe"
version = "0.1.0"
edition = "2021"
[dependencies]
faer = { package = "t4a-faer", path = "../faer/faer", default-features = false, features = ["std", "linalg"] }
```

An executed control replaces identity `m` with a dense diagonal `1/scale I`,
passed as `&m`. It returns finite solutions with true residuals about 2.22e-16 and
4.44e-16. The report proposes scaled recurrence blocks to avoid materializing
two tiny/huge triangular inverses before applying the solution coefficient.
That numerical correction is not implemented here.

Colmat's independent true-residual verification returns `:residual_mismatch`.
The observer bridge now preserves nonfinite failed native iterates by borrowing
its GC-preserved Julia snapshot for an internal operator product, with explicit
shape/type/alignment/range checks and no pointer retained after the call.

The same executable also fails on clean main `46154ca9ad1df24eeb2abeee76a7537415d4b9a0`
(t4a-faer 0.28.3, including #243), with identical numerical output and exit 101.
