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

## AED Sylvester index scratch after the rounded-window fix

[faer-qz-aed-index-scratch.rs](faer-qz-aed-index-scratch.rs) independently
reproduces [faer #260](https://github.com/tensor4all/faer/issues/260). Use the
manifest above, but select faer main `3cc63cdb312e9f81eb6ac194bfea7fcda1d70d50`
(PR #253) plus the exact sentinel correction from
`d862cd76245c09c1058fcff3417ac83c565e96bf` (PR #255). This isolates #254;
no Givens/range changes are present. Both commits can be fetched from the fork;
cherry-picking the latter onto the former produces the tested source combination.

`cargo run` with rustc 1.99.1 on Linux x86_64 queries 2,112 bytes for a valid
18x18 Hessenberg/triangular pencil, `blocking_threshold=15`, default window.
It exits 101 in full-pivot LU: zero workspace bytes remain for a requested
24-byte `(usize, usize, f64)` record. The corresponding public regression also
fails for f32. The rounded-window fix alone therefore does not complete #252's
broader workspace audit.

Changing only AED's one `[usize;8]` reservation to four arrays changes the
query to 2,304 bytes and completes the QZ call without an allocation panic.
The reproducer retains its independent numerical gate and still exits 101
there: coupled residual ratio 61.6469 exceeds 20, with Q/Z orthogonality errors
3.74745e-13 / 4.63070e-13. This is additional evidence for #257, not numerical
success. Draft [faer PR #258](https://github.com/tensor4all/faer/pull/258)
contains the array correction and focused public/private regressions. Colmat's
construction scratch correction remains in place pending verified adoption.
