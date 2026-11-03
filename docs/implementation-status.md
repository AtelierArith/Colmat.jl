# Issue #1 implementation status

The complete objective is the feature inventory in
<https://github.com/AtelierArith/Colmat.jl/issues/1>. This checklist records
progress; completing one phase does not complete the issue.

- [ ] Owned matrix/vector/diagonal types, scalar types, views and capacity management
- [ ] Native arithmetic, reductions, structured products, permutations and Kronecker product
- [ ] Reusable LU, full-pivot LU, QR, pivoted QR, LLT, LDLT and block LDLT factorizations
- [ ] Dense SVD, eigen, generalized eigen, Schur and QZ
- [ ] Reusable solves, transposed/adjoint/right solves, least squares, inverses and pseudoinverses
- [ ] Sparse storage, arithmetic, symbolic/numeric factorizations and ordering
- [ ] Matrix-free operators, preconditioners, CG, BiCGSTAB, LSMR and partial spectra
- [ ] Low-level destructive factorizations, scratch, reflectors, rotations and Pfaffian
- [ ] Caller-owned thread teams, statistics, random generation and NPY
- [ ] Documentation, tests and feature-by-feature final audit

Existing before this goal: Float64 FaerMatrix arithmetic and matrix/vector
solves; native thin/full `svd` and `svdvals`; Julia array compatibility; lifecycle
and aliasing checks (361 tests). Most array-generic Julia methods are not native
faer implementations and do not count as completing the corresponding row.

## Verified dense Float64 checkpoint

The following now call Rust directly rather than Julia's generic elementwise
factorizations. They do not complete the scalar-type, sparse or parallel scope.

- Reusable `FaerFactorization` for partial/full pivot LU, QR/column-pivot QR,
  Cholesky, LDLT and block LDLT. Independent factor storage, clone, close,
  factor accessors, permutation conventions, reconstruction and inverse.
- Factor solves, transpose/adjoint solves, alias-safe `ldiv!`, right solves and
  `rdiv!`; rectangular QR least squares and SVD minimum-norm fallback for wide
  or numerically rank-deficient QR systems.
- Native triangular and unit-triangular solves and inverses, ignoring the
  unused triangle and implicit unit diagonal.
- General/self-adjoint and generalized eigenpairs/eigenvalues. Selected
  triangle handling and independent Julia factor outputs.
- Native sum (including axes 1/2), entrywise norms, inner product, determinant,
  trace, extrema, finite/NaN/identity/symmetry predicates, operator norms and
  Kronecker product. Factor logabsdet/logdet and positive-definiteness checks.
- Pfaffian with skew-symmetry validation.
- Test dependencies migrated to a Julia 1.13 workspace.

Validation: `Pkg.test(; julia_args=["--threads=2"])` passes 808 tests, including
factor lifecycle/reuse, rectangular and empty systems, residuals, generalized
eigenvector normalization and Pfaffian determinant identities.

Still required: the remaining Issue #1 inventory, notably non-Float64 storage,
owned vectors/diagonals/borrowed views, permutations and capacity APIs, native
pseudoinverse/rank/condition interfaces, Schur/QZ, sparse and matrix-free APIs,
destructive/scratch/rotation/reflector APIs, teams, statistics, random and NPY.
Generalized eigenvalues with infinite/indeterminate ratios and all advanced
failure/keyword conventions need further coverage. No whole-issue completion
is claimed.

## Verified typed-storage checkpoint

`FaerMatrix{T}` and `FaerFactorization{T,...}` now preserve Float32, Float64,
ComplexF32 and ComplexF64. Float64 keeps its original owner ABI; a separate
typed Rust module owns the other scalar types. Promotion happens before an
unfactorized mixed-type solve. Reusing a lower-precision factorization retains
the numerical precision of those factors.

Validated across all four types:

- Owned construction, conversion, mutation, clone/deepcopy, close, scalar
  promotion, arithmetic, matrix products, transpose versus adjoint and Kronecker
  products. Complex dot products conjugate the first operand.
- Thin/full SVD with real singular values, LU/full-pivot LU, QR/pivoted QR,
  LLT/LDLT/block LDLT, factor reuse, inverse, factor access and cloning.
- General/self-adjoint/generalized eigenpairs, triangle selection (including
  ignored imaginary Hermitian diagonals), logabsdet with complex phase,
  triangular solves/inverses and complex Pfaffian identities.
- Native sum, axis sums, norms and dot products. Julia complex entrywise norms
  use complex magnitudes, rather than faer's componentwise BLAS-like metrics.
- Identity/undef constructors and native `reserve!`, `sizehint!`, `resize!` and
  `truncate!`, with preserved overlap and caller-specified fill values.
- Singular LU/block-LDLT status and singular QR-inverse rejection.

Validation: `Pkg.test(; julia_args=["--threads=2"])` passes **1868 tests**.
The README's runnable Julia examples are checked separately.

The frozen 76-row inventory remains the scope. Still incomplete: owned vector
and diagonal types, native borrowed views, permutation operations, extended
precision aliases/custom scalar traits, full pseudoinverse/rank/condition
interfaces, Schur/QZ, sparse, matrix-free algorithms, destructive factorization
and scratch interfaces, rotations/reflectors, teams, statistics, random and NPY.
Complex symmetric (non-Hermitian) structured decompositions and advanced
keyword/error coverage also remain to be audited. This checkpoint does not
complete Issue #1.
