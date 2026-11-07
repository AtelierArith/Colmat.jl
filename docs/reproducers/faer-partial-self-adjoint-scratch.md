# faer partial self-adjoint eigen scratch under-declaration

Reported as [tensor4all/faer#301](https://github.com/tensor4all/faer/issues/301).

`partial_self_adjoint_eigen_scratch` declares one `n x n` matrix for the full-subspace branch, but
`partial_full_spectrum` allocates two (the dense operator copy and the eigenvectors `u`). With an
exact-size buffer `partial_self_adjoint_eigen` panics inside `self_adjoint_evd`.

Executed on t4a-faer `57e0f375a9b722332787373de9f91822232e5c8a`, rustc 1.99.1, Darwin x86_64, with a
standalone crate depending on that git revision (`features = ["std", "linalg"]`) and
[`faer-partial-self-adjoint-scratch.rs`](faer-partial-self-adjoint-scratch.rs) as `src/main.rs`:

- `cargo run --release -- exact`: exit status 101, `buffer is not large enough to allocate an array of
  type f64 of the requested length` (remaining 11264 bytes, requested 32768).
- `cargo run --release -- padded` (one extra `temp_mat_scratch::<f64>(n, n)`): exit status 0, four
  eigenpairs with residual 8.3e-14.

Colmat adds the missing `n x n` buffer to its partial self-adjoint EVD scratch query as a workaround;
remove it after adopting a faer revision that fixes #301 and re-running this reproducer.

Related API requests from the same work: [#302](https://github.com/tensor4all/faer/issues/302)
(run-time `SchurParams`/`GeneralizedSchurParams` controls; Colmat uses scoped globals in
`with_schur_tuning`), [#303](https://github.com/tensor4all/faer/issues/303) (`ColPivQr` rank and
truncation; Colmat implements `rank(F)` and `ldiv!(F, B, rcond)` itself) and
[#304](https://github.com/tensor4all/faer/issues/304) (balancing job selection and EVD-driver
balancing; Colmat rejects `permute != scale`).
