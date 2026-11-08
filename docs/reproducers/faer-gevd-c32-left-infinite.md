# faer gevd_cplx: wrong c32 left eigenvectors with an infinite eigenvalue

Reported as [tensor4all/faer#324](https://github.com/tensor4all/faer/issues/324) (Colmat #239).

For a 12×12 `c32` pencil whose `B` is exactly singular (one infinite eigenvalue), `gevd_cplx` returns
wrong left eigenvectors. Every column has a relative residual of 0.12–0.47. The right
eigenvectors, the eigenvalues and the same pencil in `c64` are correct. Deterministic pencils of the
same structure are fine, so the failure depends on the data. In Colmat's test suite the outcome also
depended on whether other work ran first in the process.

Executed on t4a-faer `d374ec0`, rustc 1.99.1, Darwin x86_64, `--release`, with
[`faer-gevd-c32-left-infinite.rs`](faer-gevd-c32-left-infinite.rs) as `src/main.rs` (the pencil is
embedded as f32 bit patterns):

```
c32: max relative residual left 1.962e-1, right 1.889e-7
c64 (same pencil): max relative residual left 3.903e-16, right 2.810e-16
```

Until a faer revision fixes #324, Colmat computes the left (and both-side) eigenvectors of
`ComplexF32` pencils in `ComplexF64` and rounds the results back to `ComplexF32`. Right-only and
values-only calls are unchanged.
