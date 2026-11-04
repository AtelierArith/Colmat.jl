# LSMR initial-convergence residual diagnostics

Reported as [faer #248](https://github.com/tensor4all/faer/issues/248); correction
in [PR #249](https://github.com/tensor4all/faer/pull/249).

The adjacent executable on clean faer main `46154ca9ad1df24eeb2abeee76a7537415d4b9a0`
(t4a-faer 0.28.3, Rust 1.99.1 / Linux x86_64) reports initial normal residual
sqrt(2) but returns info with zero absolute/relative residuals. The independent
info assertion exits 101. Public candidate `106d870fa40545223ced29e5c8e0034c9459ad84`
returns sqrt(2), 1, iter_count=0 and stopped=false; the same MWE exits 0.

Create a Cargo binary next to a checkout named `faer`, copy the adjacent source
to `src/main.rs`, and run `cargo run`. Complete manifest:

```toml
[package]
name = "lsmr-initial-info-mwe"
version = "0.1.0"
edition = "2021"
[dependencies]
faer = { package = "t4a-faer", path = "../faer/faer", default-features = false, features = ["std", "linalg"] }
```

The correction preserves the computed `norm` and `norm/norm_ref` on the initial
tolerance return. The preceding zero-reference return is unchanged. The original
[absolute-tolerance MWE](faer-lsmr-atol.rs) also succeeds on the selected upstream,
so Colmat no longer converts atol into a relative bound for the fixed #207.
