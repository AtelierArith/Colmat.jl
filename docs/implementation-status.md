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

## Owned collections and borrowed views checkpoint

- `FaerVector`, `FaerRow` and `FaerDiagonal` own faer `Col`, `Row` and `Diag`
  storage in all four native scalar types. Construction, independent copies,
  scalar promotion, arithmetic, sums, norms, conjugation, dot products and
  lifecycle checks use native calls. Stored diagonal entries can be changed via
  `filldiagonal!` or `diagonalview`; off-diagonal values remain implicit zeros.
- Matrix/vector products and matrix/factor solves with Rust-owned vector RHS
  return Rust-owned vectors. Vector `mul!` and `ldiv!` handle overlapping input
  and output. Row/column capacity and length management are available.
- Range-based matrix, row and column views retain their owner. Nested,
  reversed, strided and repeated-index ranges are supported. Native products
  and copies create genuine MatRef/MatMut only within synchronous calls;
  references never escape a retained owner lifetime. Equal-type view products
  borrow input buffers directly.
- Capacity reservation preserves views. Shape changes permanently invalidate
  earlier views; closing an owner also rejects further view accesses. Closing
  an intermediate view does not close its owner or nested independent view.
- View assignment stages its source before borrowing the destination mutably.
  Repeated indices use ordered scalar writes, avoiding aliased mutable refs.
  Broadcast and zero-dimensional nested views retain Julia array behavior.
  `deepcopy(view)` clones its parent and preserves the view type.

Validation: the final `Pkg.test(; julia_args=["--threads=2"])` run passes
**2666 tests**, including view-copy and array-integration checks.
README Julia examples pass, Julia method ambiguity detection is empty, and
`cargo fmt --check` passes.

Still required: the complete remaining 76-row inventory. In particular, generic
integer-vector indexing is not a native range view; arbitrary borrowed Julia
buffers, specialized diagonal/matrix products and all row/column operation
combinations are not finished. Permutations, extended precision/custom fields,
pseudoinverse/rank/condition interfaces, Schur/QZ, sparse, matrix-free,
destructive/scratch/reflector/rotation APIs, teams, statistics, random and NPY
remain. This checkpoint does not complete Issue #1.

## Native permutations and pseudoinverse checkpoint

- `FaerPermutation` owns a validated faer `Perm<usize>`, with Julia 1-based
  indices, independent clone/inverse/composition and explicit lifecycle. Gather
  semantics match `A[p,:]` and `A[:,p]`; the composition `p*q` has indices
  `Vector(p)[Vector(q)]`. Factor permutations can be copied into this owner.
- Native `permute_rows`/`permute_cols`, their mutating variants and row/column
  swaps support all four matrix scalar types. Matrix owners retain their shape
  and existing views. View destinations use staged independent native results.
  Native vector `permute!`/`invpermute!` and row-column operations are included.
- Native SVD pseudoinverse reconstruction now returns Rust-owned matrices for
  `pinv`, including absolute/relative cutoffs, tall/wide/empty matrices and
  rank-deficient inputs. Hermitian and real Symmetric wrappers use native
  self-adjoint EVD pseudoinverse reconstruction, honoring the selected triangle
  and ignoring imaginary Hermitian diagonals.
- Dedicated `rank` and `cond` interfaces use native spectra and factors;
  condition numbers support p=1/2/Inf. Typed matrix operator 1/Inf norms now
  reduce complex magnitudes directly in Rust. Julia's min-dimension default
  tolerance is used rather than faer's different max-dimension default.
- Float32 cutoff handling preserves comparison against Float64 tolerances,
  avoiding upward rounding that could incorrectly discard a native value.

Validation: `Pkg.test(; julia_args=["--threads=2"])` passes **3973 tests**.
All README Julia examples, method ambiguity checks and `cargo fmt --check`
pass. Closed empty matrices also reject operator-norm access in all four types.
Unfinished scope still includes specialized diagonal/structured products,
all row/column combinations, extended precision/custom scalar fields,
Schur/QZ, sparse storage and solvers, matrix-free algorithms, destructive
factorization and caller-owned scratch, reflectors/rotations, thread teams,
statistics, random generation and NPY. Advanced factor and algorithm options
remain subject to the frozen 76-row inventory; this is not issue completion.


## Native Hessenberg and Schur checkpoint

- All four native scalar types now support `hessenberg` / `hessenberg!`,
  `schur` / `schur!`, and `ordschur` / `ordschur!`. Schur outputs use Julia's
  standard `LinearAlgebra.Schur` with Rust-owned T/Z matrices. Hessenberg
  outputs use `FaerHessenberg` with explicit native H/Q matrices.
- Hessenberg reduction constructs Q using faer's block Householder application;
  `schur(H::FaerHessenberg)` reuses H/Q and performs only the Schur QR stage.
  Schur eigenvectors use faer's dedicated real/complex Schur back-substitution,
  preserving value order and converting real pair columns to complex vectors.
- Real conjugate pairs remain standardized 2x2 blocks, and selection cannot
  split a pair. Reordering stages independent native T/Z and commits only after
  success, including eigenvalue element-type conversion. Existing native matrix
  views remain valid when a factor or input matrix is updated.
- Power-of-two scaling follows faer's EVD strategy for very small/large inputs.
  Tests include blocked 257x257 reductions, repeated eigenvalues, defective
  Jordan blocks, selected conjugate pairs and independent lifecycle/copies.
- Native FFI eigenvalue buffers always use the matrix's matching complex scalar
  type, even for manually constructed Julia Schur factors whose cached values
  have another element type. `deepcopy` preserves custom value-view types.

Validation: `Pkg.test(; julia_args=["--threads=2"])` passes **5583 tests**.
All README Julia examples and `cargo fmt --check` pass; method ambiguity
detection is empty. This does not complete
Issue #1. Generalized Schur/QZ, balancing, nondefault algorithm controls, sparse,
matrix-free, caller-owned scratch, independent reflectors/rotations, thread
teams, statistics, random and NPY remain. The use of Householder routines inside
Hessenberg reduction is not a standalone public reflector API. The mutating
interfaces still allocate independent work results and internal scratch.


## Native generalized Hessenberg and QZ checkpoint

- `schur(A,B)` returns Julia `GeneralizedSchur` with native owned S/T/Q/Z,
  satisfying A=Q*S*Z' and B=Q*T*Z'. All four scalar types are supported, including
  native mixed-type promotion, native range views and mixed Julia/native arrays.
- `hessenberg(A,B)` returns `FaerGeneralizedHessenberg` with explicit native
  H/T/Q/Z. B is first reduced by native QR, then faer's generalized Hessenberg
  routine reduces the pencil. `schur(H)` reuses this reduction and accumulated
  Q/Z, running only the native real/complex QZ stage.
- Native homogeneous alpha/beta vectors remain available, including infinite
  eigenvalues (nonzero alpha, zero beta) and indeterminate 0/0 pairs. `eigvals(F)`
  follows Julia's ratio convention. Nonconvergence is checked with initialized
  sentinels, finite homogeneous outputs and converged Schur-band structure.
- Matrix-pair `schur!`/`hessenberg!` and `schur!(H)` commit independent native
  work results after success. Existing views and owners are preserved. Mutating
  input pairs require distinct owners and matching scalar types.
- Input matrices are scaled independently by powers of two for safe kernel
  ranges; S/T and homogeneous alpha/beta are rescaled to the original pencil.
  Tests include extreme relative scales, singular pencils, all scalar promotion
  pairs, blocked QZ (96x96) and blocked generalized Hessenberg (257x257).
- The inspected faer real-QZ scratch helper undersizes odd maximum AED windows:
  its floor window differs from the kernel's even-rounded window (e.g. 31 vs
  32 at n=96). Its local Sylvester workspace also lists one index array while
  the solver allocates four. The wrapper computes the recursive live bound with
  these sizes and takes its maximum with the public recommendation. Blocked
  QZ remains enabled; the dependency checkout is not patched. Regressions cover
  n=78/96/258, including nested window sizing.
- Factor copy/deepcopy, explicit close, dimension checks and invalid-reduction
  failure paths are covered. Source pointers remain retained and serialized;
  all mutable output handles are fresh owners during native computation.

Validation: `Pkg.test(; julia_args=["--threads=2"])` passes **7957 tests**.
README Julia examples, method ambiguity checks and `cargo fmt --check` pass.
The faer dependency checkout remains clean. Issue #1 remains incomplete:
generalized Schur reordering, nondefault tuning, balancing, specialized
structured products, sparse storage/solvers, matrix-free algorithms,
caller-owned scratch and remaining destructive factorizations, standalone
reflectors/rotations, teams, statistics, random and NPY still require work.
Generalized Schur reordering has no corresponding public faer entry point in
the inspected revision and is not counted as implemented by returning the
standard Julia factor type. Work results and internal scratch are allocated.
