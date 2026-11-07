# Self-adjoint factorization parameters in faer

Standalone Rust reproduction (no Julia or RustCall) for the dense LLᴴ, LDLᴴ
and LBLᴴ parameters Colmat connects in `cholesky`/`ldlt`/`bunchkaufman`
(issue #104). Not filed upstream; drafted here for a later report.

From this directory, fetch the exact tested fork revision into the ignored
`faer/` directory and run:

```sh
git init faer
git -C faer remote add origin https://github.com/tensor4all/faer.git
git -C faer fetch --depth=1 origin d374ec022b76bd99089af396964945fa03d8d29c
git -C faer checkout --detach FETCH_HEAD
cargo run --locked --release
```

Executed on t4a-faer `d374ec022b76bd99089af396964945fa03d8d29c`, rustc 1.99.1,
Darwin arm64, `--release`, features as in the Colmat consumer. The program exits
with code 1 when every defect below reproduces. Output:

```
llt n=2: count=1 L[0,0]=-3 expected sqrt(delta)=1, max|offdiag(L Lᴴ - A)|=2.000e0
llt n=64: count=1 L[0,0]=-3 expected sqrt(delta)=1, max|offdiag(L Lᴴ - A)|=2.000e0
llt n=200: NonPositivePivot { index: 128 } (expected success)
ldlt no signs: count=0 D[0]=1 (pivot 0 was replaced by delta)
ldlt signs [+1,-1]: count=1
lblt solvers::Lblt::new perm: [5, 1, 2, 4, 3, 0]
lblt low-level Partial perm: [1, 2, 0, 3, 4, 5]
lblt low-level PartialDiag perm: [5, 1, 2, 4, 3, 0]
lblt low-level Rook perm: [1, 2, 0, 3, 4, 5]
lblt low-level RookDiag perm: [5, 1, 2, 4, 3, 0]
lblt low-level Full perm: [5, 1, 2, 4, 3, 0]
complex symmetric LBL: max|reconstruct - A| = 2.000e0 (reconstruct = [
[1.0 + 0.0i, 2.0 + 1.0i],
[2.0 - 1.0i, 3.0 + 0.0i],
])
defects reproduced: [true, true, true, true, true]
```

1. **Regularized LLᴴ stores a wrong diagonal (bug).** With
   `LltRegularization { delta: 1, epsilon: 1e-8 }`, the input
   `A = 3.5 I + 0.5 11ᵀ` with `A[0,0] = -3` has a positive definite
   regularized matrix `A + 4 e0 e0ᵀ`, so `L[0,0]` must be `sqrt(delta) = 1`.
   `cholesky_block_left_looking` scales the whole pivot column, diagonal
   included, by `1/sqrt(delta)` (`ldlt/factor.rs`, `simd_cholesky_row_batch`
   and `cholesky_fallback`), so the diagonal holds the original pivot over
   `sqrt(delta)` (`-3`). `llt::factor::cholesky_in_place` does not copy the
   temporary D back (the LDLᴴ entry point does). Beyond the recursion
   threshold the blocked recursion then solves with that diagonal and the
   factorization fails (`n = 200`). Proposed fix: write `D[j]` (the square
   root used) to `A[j,j]` in LLᴴ mode, as LDLᴴ does; regression test: the
   off-diagonal of `L Lᴴ` equals `A` for this matrix at 2, 64 and 200.
   Colmat's workaround (`deps/rust/src/selfadjoint_tuning.rs`) runs
   regularized LLᴴ as the same kernel's LDLᴴ mode with every expected sign
   `+1` (identical replacement rule and count) and scales the unit factor by
   `sqrt(D)`.
2. **LDLᴴ regularization count misses replaced pivots.** `count` increments
   only on the `sign == 1 && diag <= epsilon` branch: corrections towards an
   expected `-1` and replacements without signs (`sign == 0`) are not
   counted, although `LdltInfo` documents "number of pivots whose value or
   sign had to be corrected". Proposed fix: increment on every branch that
   changes `diag`. Colmat documents the count as faer's.
3. **`solvers::Lblt::new` takes no `LbltParams` (API request).** Only the
   low-level `lblt::factor::cholesky_in_place` reaches strategies other than
   the default `PartialDiag` (above, `Partial`/`Rook` give a different
   permutation), the block size and the parallel floor; likewise
   `solvers::Llt::new`/`Ldlt::new` take neither params nor regularization, and
   the solver types cannot be assembled from an existing factor. Proposed API:
   `Lblt::new_with_params(A, side, par, params)`,
   `Llt::new_with(A, side, regularization, par, params)` and
   `Ldlt::new_with(...)` returning the info. Colmat owns equivalent factor
   types (`deps/rust/src/lblt.rs`, `deps/rust/src/cholesky_factors.rs`).
4. **No transposed-symmetric LBLᵀ (API request).** The self-adjoint kernels
   read a complex symmetric, non-Hermitian matrix as the Hermitian matrix of
   its lower triangle (the reconstruction above conjugates the upper triangle
   and drops the diagonal's imaginary part); LAPACK `zsytrf`/`zsytrf_rook`
   have no counterpart. Colmat rejects such input for `ldlt`/`bunchkaufman`
   with `ArgumentError`.
