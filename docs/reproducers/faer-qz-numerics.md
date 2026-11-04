# Additional QZ numerical reproductions

All original failures were executed on clean faer
`19053d8c22993851c032d5373b4ec188998f2b5c`, version 0.29.0, Rust 1.99.1,
Linux x86_64, with `default-features=false, features=["std","linalg"]`.
Julia and RustCall are not involved.

Create sibling `faer` and `mwe` directories, checkout the pinned revision in
`faer`, and place one source below at `mwe/src/main.rs` with this manifest:

```toml
[package]
name = "qz-numerical-mwe"
version = "0.0.0"
edition = "2021"
[dependencies]
faer = { package = "t4a-faer", path = "../faer/faer", default-features = false, features = ["std", "linalg"] }
```

- [faer-qz-aed-sentinel.rs](faer-qz-aed-sentinel.rs): `cargo run` exits 101 at
  qz_real/mod.rs:2301 with unsigned subtraction overflow. Its n=78 default-tuning
  fixture has the queried memory plus 1 MiB to isolate this failure from #252.
  Report: [faer #254](https://github.com/tensor4all/faer/issues/254), proposed fix
  [PR #255](https://github.com/tensor4all/faer/pull/255).
- [faer-hessenberg-subnormal.rs](faer-hessenberg-subnormal.rs): `cargo run`
  reports `all_finite=false` for f32/f64/c32/c64 and exits 101. The 3x3 pencil
  is finite and has identity B, with no unrepresentable norm. Tiny norm/cosine
  reciprocals in the rotation helper produce the nonfinite factors. Report:
  [faer #256](https://github.com/tensor4all/faer/issues/256). A local range-safety
  candidate makes all four finite and passes independent reconstruction/unitarity
  regressions; it is not yet adopted in Colmat.
- [faer-qz-accuracy.rs](faer-qz-accuracy.rs): use `cargo run --release` to
  isolate numerical quality from the debug overflow above. With a valid maximum
  AED window its DGET54-style coupled residual ratio is 436.6112, Q/Z orthogonality
  errors are 1.7249e-11 and 1.2511e-11. The assertion uses threshold 20. LAPACK on
  the identical pencil gives ratio 0.7724 and orthogonality errors near 3e-14.
  Report: [faer #257](https://github.com/tensor4all/faer/issues/257). The local
  rotation candidate improves but does not resolve this gap (ratio 178.5679).

None of the reports implies that every QZ numerical issue has been fixed.
Colmat's selected dependency and existing scratch workaround remain unchanged
until merged candidates and the original plus wrapper regressions are verified.
