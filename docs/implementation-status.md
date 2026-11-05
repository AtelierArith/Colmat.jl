# Issue #1 implementation status

The complete objective is the feature inventory in
<https://github.com/AtelierArith/Colmat.jl/issues/1>. This checklist records
progress; completing one phase does not complete the issue.

The remaining implementation work and acceptance criteria are tracked in
[follow-up Issue #22](https://github.com/AtelierArith/Colmat.jl/issues/22).
Earlier Issue #1/#2 records and the [handoff plan](issue-1-follow-up.md) remain
historical evidence; the full 76-row inventory is still the overall scope.

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
At this historical checkpoint generalized Schur reordering had no public faer
entry point. It is implemented by the Issue #22 reorder extension recorded below;
work results and internal scratch remain allocated.


## Native sparse storage, patterns and arithmetic checkpoint

- `FaerSparseMatrix` owns actual faer CSC/CSR matrices in all four native scalar
  types, as a Julia `AbstractSparseMatrix{T,Int}`. Construction, duplicate-triplet
  summation, format conversion, independent copies and Julia CSC snapshots
  preserve sparse storage. Explicit zeros are retained until a structural
  operation removes them; `dropzeros!` is provided.
- Scalar access and mutation, structural insertion, sparse copy assignment and
  live `nonzeros` views are connected. Value-only changes retain views, while
  structural changes invalidate them. Value views retain owners, support
  alias-safe copying/broadcast, and reject stale/closed access.
- `FaerSparsity` owns value-free symbolic CSC/CSR structures independently of
  numeric owners. Symbolic union calls faer directly. New numeric matrices can
  copy the structure with supplied values or zero-initialized storage.
- Native sparse +/-, scalar arithmetic, conjugation, transpose/adjoint,
  sparse*sparse, sparse*dense and dense*sparse products are connected, including
  four-type promotion and CSC/CSR combinations. Matrix/vector `mul!` handles
  aliasing and alpha/beta; products with owned vectors return owned vectors.
- Native sum and entrywise p=1/2/Inf norms operate only on stored values.
  CSC index accessors return snapshots, and CSR compressed storage has explicit
  physical row-order snapshots. Sparse inputs are densified only when a dense
  conversion or dense output interface is explicitly requested by these APIs.
- Tests cover a 2-by-10^9 CSR matrix with only two stored entries, including
  copying, arithmetic, promotion, transpose, reduction and symbolic reuse.
- Loading SparseArrays revealed intersections with existing triangular solves
  and view copy methods. Explicit methods resolve compressed-vector RHS and
  CHOLMOD/CSC view copies without changing their native numerical paths.

Validation: `Pkg.test()` now uses ParallelTestRunner.jl 2.10.0 and passes
**11548 assertions across 16 independent files and 16 worker processes**.
The runner discovers tests only under `test/`; the original compatibility PoC
is in `test/poc.jl`. Every test file imports its own dependencies. README Julia
examples and `cargo fmt --check` pass; method ambiguity detection is empty.
Issue #1 is still incomplete:
sparse factorizations/solves/ordering, specialized sparse broadcast/map and
borrowed sparse views, matrix-free algorithms, remaining structured products,
extended/custom scalar types, destructive/scratch APIs, balancing, independent
reflectors/rotations, teams, statistics, random and NPY remain. General array
fallbacks may return dense Julia arrays and are not counted as native sparse
implementations. CSC conversion necessarily allocates column pointers; CSR
callers should use physical `compressed_storage` when preserving row layout.


## Native sparse triangular solve checkpoint

- The four standard triangular wrappers over native CSC/CSR owners, including
  transposed and adjoint parents, dispatch to faer's public sparse triangular
  solve routines for Float32/Float64/ComplexF32/ComplexF64.
- Unused halves are removed in sparse storage; explicit diagonal entries supply
  the first/last-position invariant required by faer. Unit triangles ignore
  stored diagonals and supply ones even when no diagonal was stored.
- Native diagonal validation reports Julia SingularException with a 1-based
  index. No sparse coefficient matrix is densified. A 50,000-square diagonal
  regression exercises a size that would require 20 GB if made dense.
- Matrix/vector RHS, structured and sparse RHS, promotion, owned outputs,
  alias-safe two/three-argument ldiv!, empty shapes, and closed-owner errors
  are covered. Structured RHS dispatch intersections are resolved explicitly.
- Sparse LLT/LU/QR, symbolic analysis reuse, ordering and the remaining Issue #1
  categories are still pending. Existing low-level triangular scratch controls
  are not exposed.

Validation: default `Pkg.test()` with ParallelTestRunner.jl passed **13027
assertions across 17 independent files using 16 workers**. The new triangular
file contributes 1479 assertions. All README Julia examples, JuliaFormatter,
`cargo fmt --check`, and `git diff --check` pass. No dependency source changes.


## Native sparse symbolic and numeric factorization checkpoint

- `FaerSparseSymbolic` owns faer's SymbolicLu/SymbolicQr/SymbolicLlt plus a
  canonical compressed pattern snapshot. `factorize(plan,A)` validates exact
  stored structure before invoking numeric LU/QR/LLT. CSC/CSR storage and scalar
  type can differ; stored-zero positions remain part of the pattern contract.
- `lu`, `qr`, `cholesky`, `factorize` and sparse matrix backslash use these native
  sparse factors. A separate `FaerSparseFactorization` owner avoids casting
  sparse handles to the existing dense factor ABI.
- Independent source snapshots and plan clones keep numeric factors usable
  after original inputs/plans are changed or closed. Copy/deepcopy and close
  are supported. Mixed precision solves reuse symbolic analysis and refactor
  a sparse snapshot in the promoted scalar type.
- Matrix/vector/owned/view RHS, LU/LLT transpose/adjoint solves, alias-safe
  `ldiv!`, singular/nonpositive-pivot status and `check=false` are implemented.
  Cholesky wrappers access only the selected triangle; ignored-half NaNs are
  accepted. Native Hermitian/symmetric checks visit only stored entries.
- `lu!(F,A)`, `qr!(F,A)` and `cholesky!(F,A)` reuse the analysis and replace numeric
  factors transactionally, preserving F when a checked update fails. These
  updates require matching scalar type and pattern; `factorize(plan,A)` allows
  a different native scalar type.
- Sparse QR currently handles full-column-rank square/tall inputs. faer does
  not provide transpose QR solves through this interface. Wide/rank-deficient
  solves, explicit L/U/Q/R/permutation access, independent ordering controls,
  low-level sparse solver/scratch controls, and remaining Issue #1 categories
  remain incomplete. QR success currently records construction success, not a
  numerical rank guarantee.
- Regressions cover 50,000-square sparse factors without dense coefficient
  conversion, all four scalars and both storage formats, repeated numeric
  updates, failed-update preservation, empty matrices and zero-column RHS.

Validation: default `Pkg.test()` passed **15212 assertions across 18 independent
files using ParallelTestRunner.jl and 16 worker processes**. The new sparse
factor suite contributes 2185 assertions. README Julia examples pass, and
method ambiguity detection is empty. JuliaFormatter, `cargo fmt --check` and
`git diff --check` pass. The t4a-faer checkout remains unmodified.


## Independent sparse ordering and permutation checkpoint

- `amd` and `colamd` call faer's public AMD/COLAMD ordering routines and own
  the resulting forward/inverse permutation as FaerPermutation. AMD requires
  a square pattern; COLAMD accepts rectangular patterns. Dense thresholds and
  aggressive absorption controls are passed through, and optional AMD
  statistics expose all three public FlopCount estimates.
- Native numeric/symbolic CSC/CSR, Julia CSC, transposed/adjoint sparse owners,
  and native Hermitian/Symmetric wrappers are accepted. Wrapper patterns mirror
  only the selected half. Numeric values and stored zeros do not change the
  ordering graph. Empty shapes return the identity permutation.
- Native sparse gather rebuilds only triplets/index pairs, preserves stored
  zeros and CSC/CSR orientation, and supports `A[p,:]`, `A[:,p]`, `A[p,q]`,
  `permute_rows/cols` on numeric and symbolic owners. Numeric in-place variants
  use a staged sparse result and invalidate value views only when the stored
  pattern changes.
- Tests cover deterministic orders, valid inverses, tuning controls, independent
  lifetimes, empty/rectangular patterns, ignored halves, native sparse gathers,
  value-view invalidation, and reduced Cholesky fill for an arrow graph.
  A 50,000-square case and a 2-by-10^9 CSR row gather verify that coefficient
  matrices are not densified or converted to CSC for permutation.
- Direct ordering selection inside sparse factorizations, caller-owned ordering
  scratch, factor components, wide/rank-deficient QR, borrowed sparse views,
  matrix-free and all remaining Issue #1 categories are still pending.

Validation: default `Pkg.test()` passed **20273 assertions across 19 independent
files using ParallelTestRunner.jl and 16 worker processes**. Ordering contributes
5061 assertions. README Julia examples, JuliaFormatter, `cargo fmt --check`,
`git diff --check`, and method ambiguity checks pass. The dependency checkout
has no modifications.


## Sparse Cholesky direct ordering and algorithm checkpoint

- Cholesky symbolic/numeric owners now use faer's public low-level
  SymbolicCholesky/LLT interfaces, keeping immutable analyses in Arc and
  independent numeric values. LU/QR retain their prior high-level backends.
- `FaerSparseSymbolic(...; kind=:cholesky, ordering=...)` and `cholesky` accept
  AMD/default, identity or a validated custom permutation. `cholesky` also
  accepts Julia's sparse `perm` keyword. Simultaneous perm/ordering is rejected.
- AMD dense/aggressive controls and auto/forced simplicial/forced supernodal
  selection are exposed. Plan/factor `.p` returns an independent permutation,
  `.algorithm` reports the actual backend, and `.factor_storage` counts numeric
  scalar slots, including supernodal padding. Numeric updates, copies and
  promoted solves reuse these analyses without losing the selected controls.
- An identity-order regression revealed that faer's numeric path passes a
  both-triangle matrix directly into a triangle-only kernel. It corrupts the
  simplicial workspace (a 7-by-7 SPD case gave residual 0.558). Colmat explicitly
  filters to the selected stored triangle before calling the numeric kernel,
  retaining sparse storage and the original pattern validation contract.
  The t4a-faer checkout is not patched.
- faer's simplicial failure pivot is one-based, while its supernodal path
  forwards a zero-based dense pivot. Native conversion now normalizes both to
  Julia's one-based info; regressions exercise every pivot with both algorithms.
- Tests cover all four scalars, CSC/CSR, both triangles, all ordering/algorithm
  combinations, source/plan/permutation closure, numeric refactorization,
  transpose/adjoint and mixed precision solves, ignored-half NaNs, empty shapes,
  exact custom permutation retention, and reduced allocated factor storage on
  an arrow graph. No coefficient matrix is densified.
- LU/QR direct custom ordering, factor components, wide/rank-deficient QR,
  external scratch and all other remaining Issue #1 categories are incomplete.

Validation: default `Pkg.test()` passed **25409 assertions across 20 independent
files using ParallelTestRunner.jl and 16 workers**. The new sparse Cholesky suite
contributes 5136 assertions. README Julia examples, JuliaFormatter,
`cargo fmt --check`, `git diff --check` and method ambiguity checks pass.
The dependency checkout remains unmodified.


## Sparse Cholesky component extraction and reconstruction checkpoint

- `F.L` and `F.U` export independent Rust-owned CSC matrices from the actual
  native numeric factors. Simplicial export reads compressed column indices;
  supernodal export walks public node views and excludes unused upper block
  slots. Stored lower-block zeros are preserved.
- `L*U` reconstructs the symmetrically permuted coefficients. `FaerSparseMatrix(F)`
  uses native sparse multiplication and inverse row/column permutations to
  return original ordering. `sparse(F)`/`SparseMatrixCSC(F)` return Julia CSC;
  `Matrix(F)`/`FaerMatrix(F)` explicitly request dense storage. Typed conversions
  preserve the numerical precision of the original factorization.
- `det`, `logdet` and `logabsdet` export only the numeric diagonal from Rust,
  then perform scalar log reductions in Julia, without reconstructing or
  densifying the matrix. Log reductions avoid intermediate determinant
  overflow/underflow. Failed factorizations raise PosDefException before any
  component or diagonal access, and closed factors raise ArgumentError.
- Exported factors remain valid after numeric updates or close, and modifying
  an exported factor cannot modify the original factorization. Reconstruction
  may introduce fill/stored zeros and roundoff through sparse multiplication;
  it does not return the source snapshot.
- Tests compare against Julia dense Cholesky, check `L*U ≈ A[p,p]`, original-order
  reconstruction, all four scalar types/CSC/CSR/both triangles/orderings/native
  algorithms, several supernodes and fill, empty matrices, failed/closed factors,
  ignored-half NaNs, mixed-type conversions with source-precision tolerances,
  and large/small determinant scales. A 50,000-square diagonal case verifies
  sparse component extraction/reconstruction and diagonal-only logdet.
- Sparse LU/QR component extraction, wide/rank-deficient QR, matrix-free,
  external scratch and all other remaining Issue #1 categories remain open.

Validation: default `Pkg.test()` passed **28732 assertions across 21 independent
files using ParallelTestRunner.jl and 16 workers**. The component suite contributes
3323 assertions. All README Julia examples, JuliaFormatter, `cargo fmt --check`,
`git diff --check` and method ambiguity checks pass. Dependency sources are clean.


## Fixed-pattern sparse arithmetic checkpoint

- `add_assign!`, `sub_assign!`, `add_into!` and `sub_into!` connect directly to
  faer's corresponding fixed-pattern sparse kernels. Native CSC/CSR receivers
  accept either native orientation, native transpose/adjoint, and Julia CSC
  sources. CSR calls use transposed CSC views without CSC coefficient copies.
- Every stored input position must exist in the destination, including stored
  zeros. Native subset checks complete before mutation because faer's kernels
  may detect an invalid position only after writing earlier entries. Invalid
  shapes, patterns, closed inputs and failed conversions preserve destination
  values and indices.
- Independent input snapshots make destination/source alias combinations safe.
  Inputs are converted to destination scalar type and orientation before
  arithmetic; the output type is fixed. Destinations retain existing value
  storage, indices, format, nnz and epoch, keeping live nonzeros views valid.
- Overwrite forms zero all stored destination values before running faer's
  ternary kernel. The inspected upstream implementation omits its final tail
  after both input columns end, leaving old output values there otherwise.
  Snapshots preserve aliased inputs during initialization. Dependency sources
  remain unmodified, and the initialization also works if upstream fixes the
  omitted tail.
- Regressions cover all four scalars, CSC/CSR combinations, input subpatterns,
  empty shapes, cancellation/stored zeros, live views, alias combinations,
  pre-mutation rejection, mixed-type conversion, complex-to-real failure,
  transpose/adjoint and overwrite tails. A 50,000-square diagonal case and a
  2-by-10^9 CSR case exercise sparse storage bounds.
- General native sparse broadcast/map and arbitrary callbacks, sparse LU/QR
  components, matrix-free, scratch interfaces and all other remaining Issue #1
  categories are still incomplete.

Validation: default `Pkg.test()` passed **30552 assertions across 22 independent
files using ParallelTestRunner.jl and 16 workers**. The new fixed-update suite
contributes 1820 assertions. README Julia examples, JuliaFormatter,
`cargo fmt --check`, `git diff --check` and method ambiguity checks pass.
The t4a-faer checkout remains clean.


## Native Julia Givens / faer Jacobi checkpoint

- `givens` on native matrices/vectors and their views calls faer's rotg and
  returns standard Julia Givens plus r. Scalar-only `givens` remains Julia's
  implementation; no stdlib-only signature is overwritten.
- Left/right `*`, `lmul!` and `rmul!` adapt Julia's `[c s; -conj(s) c]` convention
  to faer's Jacobi kernels. Matrix/vector/row owners update native storage;
  views stage native copies and copy back through existing view validation.
  In-place operations convert coefficients before mutation and require real c;
  allocating products promote scalar types and return independent native data.
- Standard adjoint and composed rotations interoperate through these methods.
  Indices are validated for both axes, and closed/invalidated views fail before
  mutation. Cosine with a nonzero imaginary component is explicitly rejected
  because faer's SIMD kernels multiply using its real part.
- rotg compatibility fixes stay in the bridge: the inspected complex zero-q
  branch returns r=1 instead of p; real sign normalization differs when q
  dominates; and real coefficients computed from a rounded subnormal r can
  produce c=s=1 (norm squared 2). The wrapper corrects the zero-q result,
  normalizes the real sign convention, and scales real inputs before rotg so
  the coefficient computation stays away from the subnormal range.
- Tests compare coefficients and results against Julia for random values,
  reversed indices, zero inputs, large/subnormal values and complex phases,
  four scalar types, row/column application, adjoints/composition, mixed-type
  promotion, strided/reversed views, failed conversions and empty axes.
- Householder, standalone Jacobi objects and their dedicated
  transpose/conjugate/make_givens/2x2 interfaces, matrix-free, scratch and all
  other remaining Issue #1 categories are still incomplete.

Validation: default `Pkg.test()` passed **32718 assertions across 23 independent
files using ParallelTestRunner.jl and 16 workers**. The new rotation suite
contributes 2166 assertions. README Julia examples, JuliaFormatter,
`cargo fmt --check`, `git diff --check` and method ambiguity checks pass.
The t4a-faer dependency checkout remains unmodified.


## Owned elementary Householder checkpoint

- `FaerHouseholder` / `householder(x)` own faer's elementary reflection in
  compact O(n) native storage. Generation calls make_householder_in_place after
  componentwise real scaling; complex division by a real scale avoids squared
  denominator underflow. The source is copied, all four scalars are retained,
  and nonfinite inputs are rejected. Empty/zero-tail inputs yield identity.
- The public convention is faer's `H = I - v*v'/tau`, with real denominator tau.
  Independent native `H.v` and beta/tau/input_norm/head_with_beta_inv metadata
  are exposed. Clone/deepcopy/close and native conjugation support independent
  adjoint, transpose and inverse objects; indexing is read-only.
- Left/right application uses faer's block-Householder routines with block
  width one and queried scratch. Native matrices/vectors/rows update owned
  storage; native views stage independent results and copy back. Julia strided
  arrays are supported, with in-place output limited to the four native types.
- Allocating products promote compact coefficients and source arrays. In-place
  calls convert coefficients/metadata before mutation. Mixed-precision tests
  use the promoted compact formula as oracle, avoiding a lower-precision dense
  materialization as an assumed higher-precision reference.
- Three/five-argument mul! supports left matrix/vector and right matrix forms,
  alpha/beta and aliases. It stages native products and converts the entire
  result buffer to the destination type before copying, preserving outputs on
  conversion failure. Application does not form an n-by-n reflection; explicit
  Matrix/FaerMatrix conversion and composition of two reflections do.
- Tests cover unitary/Hermitian properties and annihilation, metadata,
  independent lifetimes and exported vectors, four scalars, mixed promotion,
  both sides, stdlib strided buffers, strided/reversed native views, stale/closed
  views, failed conversions and shapes, empty/identity cases, large/subnormal
  scales, alpha/beta and aliases. A 100,000-element application verifies linear
  reflection storage without a dense coefficient matrix.
- Arbitrary block basis/factor construction, upgrade and block-sequence APIs,
  external scratch, standalone Jacobi APIs, matrix-free and all other remaining
  Issue #1 categories remain incomplete.

Validation: default `Pkg.test()` passed **35486 assertions across 24 independent
files using ParallelTestRunner.jl and 16 workers**. The Householder suite
contributes 2768 assertions. README Julia examples, JuliaFormatter,
`cargo fmt --check`, `git diff --check` and method ambiguity checks pass.
The dependency checkout has no modifications.


## Compact block Householder sequence checkpoint

- `FaerHouseholderSequence` owns a canonical m-by-k basis and blocksize-by-k
  factor. Strict-lower essentials are copied; diagonal ones/upper zeros are
  implicit. Positive real denominators are checked against basis norms to
  ensure elementary unitary reflections. Infinity encodes identity; unused
  identity essentials are canonicalized to zero.
- Factors are constructed with faer's upgrade_householder_factor, including
  partial final blocks. The explicit-factor constructor validates relevant
  upper blocks and retains supplied coefficients. Lower/padding entries are
  ignored. Basis/factor exports are independent native matrices; tau exports
  an independent real Julia vector.
- Reblock/reblock! rebuild factors with the native upgrade API, transactionally
  for in-place changes. Copy/deepcopy/close and transformed owners preserve
  independent lifetimes. Transform flags select all left/right, normal/transpose
  and conjugated sequence variants, with the matching scratch recommendation.
- Shared Householder application/product/mul! helpers now accept both elementary
  and sequence operators through a private abstract matrix type. Source-type
  promotion retains compact coefficients and rebuilds factors; views and typed
  destination conversion retain the existing staging rules. Left/right solves
  and ldiv!/rdiv! use the unitary adjoint without materializing the operator.
- A new edge regression exposed faer's rank-one SIMD return when its tail has
  zero rows: an explicitly supplied tau=0.5 must flip sign rather than act as
  identity. The bridge factors out a final scalar reflection when it is a
  single-column final block, placing it before/after the prefix according to
  side and transpose. Scratch allocation precedes any scalar mutation. Native
  dependencies are not patched.
- Tests compare against independent products of elementary reflection matrices,
  covering all four scalars, zero/full/partial ranks, block widths, Inf identity
  factors, explicit factor loading, ignored upper/padding data, all transform
  variants, promotion, views, alpha/beta/aliases, copies, failed reblocking and
  unitary solves. A 100,000-row, three-reflection case uses compact storage.
- Direct prev_block_size upgrade controls and external scratch, standalone
  Jacobi APIs, matrix-free and the other remaining Issue #1 categories remain
  incomplete. Scalar indexing and explicit dense conversion may materialize
  temporary columns/matrices; array application uses compact coefficients.

Validation: default `Pkg.test()` passed **48918 assertions across 25 independent
files using ParallelTestRunner.jl and 16 workers**. The sequence suite contributes
13432 assertions. Existing elementary Householder tests pass after shared helper
integration. README Julia examples, JuliaFormatter, `cargo fmt --check`,
`git diff --check` and method ambiguity checks pass. Dependency sources are clean.

### Native operator adapters and CG

Added owned `FaerLinearOperator` adapters for dense, CSC/CSR, diagonal and
implicit identity storage, native `LinOp`/`BiLinOp` products, transpose/adjoint,
`mul!`, and native in-place preconditioner application. Added inverse-diagonal
and identity preconditioners and native block conjugate gradients through
`cg`/`cg!`, with retained failure iterates and independently recomputed residuals.
Arbitrary Julia callbacks, composed operators, other iterative solvers and
partial spectra remain unimplemented. Focused operator tests pass 348 assertions;
the full ParallelTestRunner suite passes 49,266 assertions across 26 files.

### Native block BiCGSTAB

Added `bicgstab` and `bicgstab!`, directly calling faer's native block solver
with independently optional left and right preconditioner actions (`M`/`N`).
Dense, CSC/CSR, diagonal and identity operators retain their storage. All four
native scalar types, mixed precision, initial guesses, tolerances and iteration
limits are supported. `FaerBicgstabResult` retains native termination diagnostics
and independently recomputed true residuals. Checked failures throw
`FaerBicgstabError` containing the last iterate and leave in-place destinations
unchanged; unchecked calls return/copy that iterate. Shared CG input validation
now rejects nonfinite RHS/initial values before native conversion.

The dedicated suite covers nonsymmetric real/complex systems, left/right/both
preconditioners, multiple RHS, aliases, native vectors, promotion, zero RHS,
exact initial guesses, failure preservation, invalid/closed inputs and a
50,000-dimensional sparse system. Arbitrary callbacks, iteration histories,
external scratch and other remaining Issue #1 categories are still incomplete.

Validation: `Pkg.test()` passes **49,747 assertions across 27 independent files
with ParallelTestRunner.jl and 16 workers**; BiCGSTAB contributes 481 assertions.
All README Julia examples, method ambiguity checks, `cargo check`, Rust/Julia
formatting and `git diff --check` pass. Dependency sources remain unchanged.

### Native block LSMR

Added `lsmr`/`lsmr!`, `FaerLsmrResult` and `FaerLsmrError`, directly calling
faer's block LSMR for rectangular least squares with optional right
preconditioning. Dense, CSC/CSR, diagonal and identity operators retain their
storage; neither coefficient densification nor normal equations are needed.
All four native scalar types and promotions, multiple RHS, initial guesses,
iteration limits and checked/unchecked failure handling are supported.

Convergence uses the true preconditioned normal residual
`norm(M' * A' * (b-A*x))`, relative to `norm(M' * A' * b)`. The bridge recomputes
both normal and primal residuals independently and rejects false convergence.
Upstream's current LSMR ignores `abs_tolerance`; the bridge converts it to an
effective relative tolerance using that reference norm. Empty dimensions bypass
upstream scratch sizing, which otherwise rounds RHS width to a multiple of the
number of unknowns. Checked failures retain the last iterate and preserve
in-place destinations. Nonzero initial guesses can retain nullspace components;
right preconditioning changes the minimum-norm metric.

Tests cover over/underdetermined and inconsistent problems, rank-deficient and
zero operators, native sparse formats and vectors, more RHS than unknowns,
real/complex preconditioners, mixed types, exact/zero initial conditions,
absolute tolerance, failure preservation, invalid/closed inputs and a sparse
30,000-column system. Arbitrary callbacks, histories, external scratch, partial
spectra and other remaining Issue #1 categories are still incomplete.

Validation: `Pkg.test()` passes **50,186 assertions across 28 independent files
with ParallelTestRunner.jl and 16 workers**; LSMR contributes 439 assertions.
All README Julia examples, method ambiguity checks, `cargo check`, Rust/Julia
formatting and `git diff --check` pass. Dependency sources remain unchanged.

### Standalone native Jacobi rotations

Added the immutable 2×2 `FaerJacobiRotation`, native `rotg`/real `make_givens`,
transpose/conjugate/adjoint, left/right 2×2 helpers, multiplication and in-place
application to Julia/native arrays and views. `LinearAlgebra.Givens(J,i,j)`
embeds a rotation in a larger array through existing native kernels. Native
scalar promotion and staged destination conversion preserve outputs on failure.

The bridge corrects two upstream details without modifying the dependency:
complex `rotg(p,0)` returns the original p as r (the native branch returns 1),
and native complex 2×2 right application is conjugated before use to compensate
for its adjoint-then-transpose implementation. Real rotg inputs are scaled
before the native call to preserve unit coefficients for subnormal inputs.
Tests compare against independent 2×2 matrix multiplication for all four types,
transforms, embedding, views, zero/sign/complex cases and failure preservation.

Partial spectra remain unimplemented. A separate native probe reproduces an
upstream panic for a 4×4 identity matrix in `partial_self_adjoint_eigen` at
`operator/self_adjoint_eigen/mod.rs:290` (full projection-space branch).
The probe and invocation instructions are in `docs/reproducers/`. This is a faer
limitation, not a RustCall defect, and does not reduce the remaining Issue #1 scope.

Validation: `Pkg.test()` passes **50,698 assertions across 29 independent files
with ParallelTestRunner.jl and 16 workers**; standalone Jacobi contributes 512
assertions. All README Julia examples, method ambiguity checks, `cargo check`,
Rust/Julia formatting and `git diff --check` pass. Dependency sources are clean.
The separate upstream probe terminates with the expected panic/exit code 101.

### Native mean and variance

Connected `Statistics.mean`, `var` and `varm` for native matrices/vectors/rows
and views to faer's `row_mean/col_mean/row_varm/col_varm` kernels. Calls snapshot
native data to exact primitive buffers without using finite-only matrix loaders,
so NaNs and infinities retain statistical meaning. Results are Julia scalars or
arrays with standard `dims` shapes, including tuples, empty/no-op reductions
and dimensions beyond the input rank. All four scalar types are supported;
complex variances are real, and explicitly supplied means promote working type.

`nan=:propagate` is the default; `nan=:ignore` excludes entries containing a NaN
component and uses each slice's included count. The bridge adapts faer's fixed
N-1 denominator and singleton-zero behavior to Julia's `corrected` option and
supplied-mean arithmetic, including empty/singleton/all-NaN slices. Explicit
reduced means accept trailing singleton dimensions. No dependency sources change.

The focused suite compares against Statistics for all scalar types, empty and
wide/tall SIMD-sized inputs, reductions, sample/population variance, supplied
means, NaN/Inf handling, native rows/vectors, views and closed inputs. Third-party
Statistics-only method ambiguities are excluded from the ownership check;
ambiguities involving Colmat methods are checked. Random generation, NPY,
parallel teams and the other remaining Issue #1 categories remain incomplete.

Validation: full `Pkg.test()` passes **53,316 assertions across 30 independent
files with ParallelTestRunner.jl and 16 workers**. After adding closed-empty-mean
validation and imaginary-component NaN regression cases, the complete focused
statistics suite passes 2,621 assertions. All README Julia examples, owned-method
ambiguity checks, `cargo check`, Rust/Julia formatting and `git diff --check`
pass. Dependency sources remain clean.

### Native balancing and right eigenvector back-transforms

Added `balance`/`balance!` and owned `FaerBalance` metadata (`B`, 1-based `perm`,
positive power-of-two `scale`, Julia `active` range), directly calling faer's
balancing routine. The similarity convention is
`B[i,j] = A[perm[i],perm[j]] * scale[j] / scale[i]`. Nonmutating calls snapshot
the input; `balance!` commits a staged result to its existing native matrix and
returns a result sharing that owner.

Added `balance_eigenvectors`/`balance_eigenvectors!`, directly calling faer's
right-vector back-transform, including complex vectors after real balancing,
multiple columns, native vectors and Julia/native arrays/views. Columns are not
renormalized. Metadata validation, native scratch allocation and complete output
conversion occur before destination mutation; nonfinite input or overflowing
back-transforms fail without changing the destination. No dependency source is
modified. External scratch and compact Hessenberg control remain unimplemented.

The focused suite covers all four scalars, empty/singleton/random/isolation cases,
power-of-two similarity reconstruction, eigenpair residuals in original
coordinates, metadata/copy/close ownership, shared in-place results, vector and
view applications, invalid metadata and overflow preservation. Other outstanding
Issue #1 categories retain their scope.

Validation: `Pkg.test()` passes **53,950 assertions across 31 independent files
with ParallelTestRunner.jl and 16 workers**; balancing contributes 631 assertions.
The previous statistics follow-up regressions are included in this full run.
All README Julia examples, method ambiguity checks, `cargo check`, Rust/Julia
formatting and `git diff --check` pass. Dependency sources remain clean.

### Native self-adjoint tridiagonalization and band EVD

Connected faer's `tridiag_in_place` through `tridiagonalize`/`tridiagonalize!`
and `hessenberg`/`hessenberg!` on native Hermitian and real symmetric wrappers.
`FaerTridiagonalization` owns a real `SymTridiagonal` H and explicit native Q,
with `A ≈ Q*H*Q'`. For complex inputs, off-diagonal phases are absorbed into Q.
Blocksize can be selected, while compact reflector storage remains internal.
The existing Householder sequence application is reused, including its native
scalar-tail correction. Nonmutating inputs are preserved; destructive facades
stage the full result before storing explicit H and retain independent Q.

Connected `tridiagonal_self_adjoint_evd` through `tridiagonal_eigen` and
`tridiagonal_eigvals`, eigen/eigvals on a native-vector SymTridiagonal, and
reuse through eigen/eigvals on the new reduction. Hermitian band inputs accept
real-valued diagonals and complex subdiagonals; upper entries are conjugates.
All four scalar types/promotions, ordering and QR/divide-and-conquer threshold
selection are supported. Coefficients/work remain O(n) for eigenvalues-only
calls. Finite extreme inputs are safely scaled before native kernels and
values restored afterward. Eigenvectors require O(n²) storage.

The dedicated suite covers empty/singleton/blocked reductions, all scalar types,
complex phase orientation, original-coordinate eigenpair residuals, orthogonality,
native structured wrappers and views, selected triangles including ignored NaNs,
in-place preservation, closed inputs, tiny/huge scales, both band EVD paths and
a 50,000-dimensional eigenvalues-only problem. External scratch, direct compact
reflector control and other remaining Issue #1 requirements remain incomplete.

Validation: `Pkg.test()` passes **55,571 assertions across 32 independent files
with ParallelTestRunner.jl and 16 workers**; the new suite contributes 1,621
assertions. All README Julia examples, method ambiguity checks, `cargo check`,
Rust/Julia formatting and `git diff --check` pass. Dependency sources are clean.

### Native bidiagonalization and bidiagonal SVD

Added `bidiagonalize`/`bidiagonalize!` and owned `FaerBidiagonalization` with
native U/V and a Julia Bidiagonal core. Tall/square problems use upper B;
wide problems reduce the adjoint and return lower B. Thin bases are the default,
with optional full bases and configurable Householder blocksize. Existing
compact Householder application, including scalar-tail correction, forms the
bases. Destructive facades stage an independent result before storing B's
rectangular embedding. Input matrices and compact metadata are not aliased.

Added `bidiagonal_svd`, `svd/svdvals` on the reduction, and native-vector
Bidiagonal overloads. The public native square band kernel handles upper/lower,
real/complex coefficients, real descending singular values and its small/large
recursion threshold. All four scalar types and promotion are supported. The
native kernel computes dense factors/workspace even for svdvals; no linear-space
values-only claim is made. Thin rectangular reduction avoids full-square bases
for high-aspect matrices.

Enabled faer's rayon feature to work around a confirmed blocked-reduction bug:
the essential Y update is incorrectly cfg-gated on rayon. Numerical calls still
use Par::Seq. A pure-Rust, zero-scratch-initialized MWE gives relative residual
0.6180742 without rayon and 9.783102e-16 with it. Reported as tensor4all/faer#203,
with removal of the erroneous cfg and feature-matrix regression tests proposed.
No dependency sources are patched. Additional native issues are reported with
executed Rust MWEs as #204 (small partial eigen panic), #205 (complex zero-q
rotg), #206 (complex right 2×2 Jacobi) and #207 (LSMR absolute tolerance).

Validation on the isolated baseline: Pkg.test(allow_reresolve=false) passes
58,236 assertions in 33 independent files using ParallelTestRunner and 16 workers;
the new suite contributes 2,665 assertions. It covers empty/rectangular/wide,
complex phases, thin/full bases and SVD, both native band paths, blocked sizes
above 128, tiny/huge scales, aliases/close/failure behavior, and a 20,000×2 thin
problem. All README examples pass. External scratch, compact controls and other
remaining Issue #1 requirements retain their scope. The initial isolated run was followed by integrated RustCall migration
verification below.


### Integrated RustCall 0.9.2 / macros 0.6 validation

Integrated the owner-binding migration with the bidiagonal implementation.
RustCall compat is the caret range "0.9.2"; Cargo uses rustcall_julia_macros
0.6.0. Native typed owner arguments use checked &T/&mut T borrows; explicit raw
owner paths remain only for type-erased or optional arguments. FaerMatrix copy
uses duplicate(&self) -> Result<Self,String> directly. Regression tests verify
independent storage, errors without allocation, aliased mutable borrows, wrong
owner types, finalized/closed owners and allowed shared aliases.

Canonical integrated validation: Pkg.test(allow_reresolve=false) passes
**58,271 assertions across 34 independent files using ParallelTestRunner.jl
2.10.0 and 16 workers**, on Julia 1.13.3 / RustCall 0.9.2. The owner-binding
suite contributes 35 assertions and the bidiagonal suite contributes 2,665.
All README Julia examples, cargo check, Rust/Julia formatting and git diff
--check pass. All five faer reports (#203–#207) were checked after publication
to contain Cargo.toml, complete Rust main and invocation instructions. Dependency
sources remain at d5be1473eed9314fca6f80ad8b4072d86282e890 without local changes.
Other Issue #1 categories remain incomplete.

### Native RNG, distribution sampling and orthogonal/unitary matrices

Enabled faer's rand feature and connected CwiseMatDistribution,
CwiseColDistribution, CwiseRowDistribution, ComplexDistribution and UnitaryMat.
`rand`/`randn` construct native matrices/vectors/rows; `rand!`/`randn!` stage fills
for native owners and views. All four scalar types, integer/tuple dimensions,
empty shapes and default RNG conveniences are supported. Complex uniforms have
independent [0,1) components; complex normals use independent variance-1/2
components, matching Julia's expected squared modulus of one.

Added the independently owned `FaerRNG <: Random.AbstractRNG` over Rust StdRng.
It supports UInt64 bit sampling (and Julia's derived scalar/array algorithms),
seeding, independent copy/deepcopy and close/GC lifecycle. Native constructors
use the continuous Rust stream directly through checked mutable owner borrows.
Other Julia RNGs provide exactly one UInt64 seed per nonempty native call;
empty calls consume none. Invalid dimensions/closed destinations are checked
before drawing. Reproducibility is for matching dependency versions and call
sequences; native sampling does not promise Julia's generated value sequence.

`randunitary` calls faer's sequential QR and diagonal-phase correction to produce
real orthogonal or complex unitary matrices. The native RNG and output owners
are independent. Arbitrary Rust Distribution instances/Julia callbacks remain
unexposed, and the rest of the original Issue #1 scope stays open.

Validation: `Pkg.test(allow_reresolve=false)` passes **59,023 assertions across
35 independent files using ParallelTestRunner 2.10.0 and 16 workers**, on Julia
1.13.3 / RustCall 0.9.2. Random tests contribute 752 assertions, covering stream
clones/reseed/close, scalar/Julia-array RNG use, all native shapes/types/layouts,
seed consumption, empty/invalid cases, views, uniform ranges, normal moments
and complex normalization, and orthogonality/unitarity. All README Julia blocks,
method ambiguity checks, cargo check, Rust/Julia formatting and git diff --check
pass. Dependency sources remain unchanged.

### NPY import and owned read-only views

Enabled faer's npy feature. `loadnpy`, `npyview` and `npyinfo` accept bytes,
IO streams and file paths for Float32/Float64/ComplexF32/ComplexF64, NPY versions
1–3, C/Fortran order and either byte order. Scalar/vector input maps to 1×1/n×1;
original shape, version, order and byte order remain available as metadata.
Malformed headers, unsupported dtypes/ranks, dimension overflow, truncated
payloads and trailing bytes are rejected before native reads. Native copies
preserve NaN payloads, Inf and signed zeros.

Read-only views own a 64-byte-aligned native-endian snapshot. Their owner stays
alive across GC; source mutation cannot alter it. `copy` materializes an
independent mutable FaerMatrix and `deepcopy` duplicates the read-only owner
without changing its concrete Julia type. Close, bounds and native dtype/shape
checks protect reads and exports, including private-owner reload regressions.
These views are not memory maps or aliases of the input bytes.

Standalone Rust MWEs demonstrated nonnative-endian misreading and unaligned
C-order copying panics in faer. Reports #210/#211 include executed complete
crates, observed results and proposed corrections; source is retained under
`docs/reproducers/faer-npy.rs`. The adapter normalizes endian and alignment
without modifying the dependency checkout. The header parser also runs on a
scoped Rust OS thread with a 2 MiB stack: pest's stack guard otherwise mistakes
Julia Task switched stacks for exhausted OS stacks during parallel tests. The
stack guard remains enabled for malformed input. Parsing/thread startup adds
cost to view reads; materializing once is appropriate for repeated arithmetic.

Validation: `Pkg.test(allow_reresolve=false)` passes **64,775 assertions across
36 files with ParallelTestRunner 2.10.0 and 16 workers**, on Julia 1.13.3 /
RustCall 0.9.2. NPY contributes 5,752 assertions, including an independently
encoded golden fixture and an explicit Julia Task regression. All README Julia
examples, cargo check, Rust/Julia formatting and git diff --check pass. The
dependency checkout remains unchanged. Other Issue #1 work,
including matrix-free callbacks/partial spectra, teams and caller-owned scratch,
remains open.

### Native partial eigen and singular spectra

Added `partial_eigen`, `partial_self_adjoint_eigen` and `partial_svd` for
`FaerLinearOperator`. All four scalar types use faer's restarted projection
methods with configurable start, tolerance, projection dimensions and restart
limit. Native owner dtype and dimensions are rechecked before exporting primitive
buffers. Julia returns sorted reported converged columns, requested/converged
counts and independently recomputed residuals. SVD checks both singular equations.
The self-adjoint entry point assumes Hermitian input; the chosen initial vector
must have a component in the desired invariant subspace.

faer's full-projection self-adjoint branch panics (#204); partial SVD likewise
asserts max_dim<min(m,n) although its public driver selects equality for small
matrices and full-spectrum requests. The latter was executed in standalone Rust
and reported with complete crate, observed assertion and proposed correction in
faer #213. Both boundaries use explicitly reported `:dense_fallback` through
native EVD/SVD. Other requests retain `:partial` and do not materialize the
operator. Zero requested count/restarts return empty outputs. Callback operators,
iteration histories and caller-owned scratch remain unimplemented.

Validation: `Pkg.test(allow_reresolve=false)` passes **65,116 assertions across
37 files with ParallelTestRunner 2.10.0 and 16 workers**, on Julia 1.13.3 /
RustCall 0.9.2. Partial spectrum tests contribute 341 assertions across all four
types, diagonal/sparse/rectangular operators, complex eigenvalues, full-space
general eigen, small/native-dense boundaries, empty shapes, zero restarts,
initial vectors, closed owners and invalid controls. All 29 README Julia blocks,
method ambiguity checks, cargo check, Julia/Rust formatting and git diff --check
pass. Dependency sources remain unchanged. The full Issue #1 goal stays open.

### Native triangular and diagonal products

Connected faer's `matmul::triangular::matmul_with_conj` through Base `*` for
lower/upper/unit triangular wrappers and native dense matrices/views, including
native triangular pairs. Exact primitive snapshots carry physical parent data;
block-structure flags ignore unused halves and stored unit diagonals, including
NaNs there. All four scalar types and mixed-type promotion are supported.
Triangular-pair results are dense FaerMatrix objects.

Native diagonal multiplication handles FaerDiagonal/Julia Diagonal with native
dense matrices, plus native diagonal/triangular pairs. The native adapter masks
ignored triangular entries before the diagonal kernel. Inputs are currently
snapshotted densely, even diagonal storage; zero-copy and allocation-free paths
remain work to do. Rust operations produce independently owned native outputs.

Three/five-argument `mul!` supports native matrices and strided/reversed native
views as outputs, alpha/beta and either-input aliasing. The operation locks the
owner lifecycle, checks shapes/open owners even for zero alpha, skips ignored
input values for alpha=0 and avoids reading old output for beta=0. A temporary
result precedes all writes. External-array/vector combinations and low-level
structured destination controls are not yet fully connected.

Validation: `Pkg.test(allow_reresolve=false)` passes **65,974 assertions across
38 files with ParallelTestRunner 2.10.0 and 16 workers**, on Julia 1.13.3 /
RustCall 0.9.2. Structured products contribute 858 assertions: four types,
empty/rectangular/unit/triangular/diagonal combinations, poisoned unused entries,
coefficients including complex alpha on real operands, ignored zero coefficients,
input aliasing, reversed views, mixed precision, closed owners and invalid shapes.
All 30 README Julia blocks, method ambiguity checks, cargo check, Julia/Rust
formatting and git diff --check pass. Dependency sources remain unchanged.
Remaining Issue #1 categories and limitations listed above remain open.

### Pivoted Cholesky LLT

Added `cholesky(A, RowMaximum(); tol, check)` for native matrices and their
Hermitian/real Symmetric wrappers. Standard `CholeskyPivoted` holds independently
owned FaerMatrix factors, so L/U/p/P/rank access follows Julia conventions.
Selected triangles are materialized without reading poisoned unused entries.
Rank-deficient factors are returned with check=false; full-rank checks protect
solves and inverses. Copy owns separate factor storage; close invalidates it.

The negative-tolerance automatic-threshold path calls faer's public pivoted LLT
factor kernel. Its fixed threshold cannot express Julia's nonnegative absolute
tol, so that case uses a complete diagonal-pivot implementation in the Rust
adapter. The Julia default remains tol=0. Both paths use faer's native pivoted
solve, inverse and reconstruct kernels, with validated permutations and promoted
primitive snapshots. Output triangles are completed by conjugation. Explicit
workspaces and snapshots are allocated; block_size, destructive entry points
and caller-owned scratch remain unexposed.

Validation: `Pkg.test(allow_reresolve=false)` passes **66,393 assertions across
39 files with ParallelTestRunner 2.10.0 and 16 workers**, on Julia 1.13.3 /
RustCall 0.9.2. Pivoted Cholesky contributes 419 assertions across four types,
automatic/zero/positive tolerances, empty/full/rank-deficient cases, selected
triangles with ignored NaNs, permutation reconstruction, native solve/inverse,
copy/close, source independence and invalid inputs. All 31 README Julia blocks,
method ambiguity checks, cargo check, Julia/Rust formatting and git diff --check
pass. Dependency sources remain unchanged. Other Issue #1 work remains open.

### Staged dense lu!, qr! and cholesky!

Added native-matrix `lu!` (including RowMaximum/allowsingular), `qr!` (NoPivot/
ColumnNorm), and `cholesky!` (ordinary/pivoted, selected Hermitian/real Symmetric
triangles). These allocate independent native factors and write back only once
the factorization is available. LU combines upper U with strict-lower L; QR
combines upper R with strict-lower Householder basis; Cholesky writes L or U
according to the selected triangle. Same-shape views observe these updates;
returned factors remain independent of later input changes/close. A failed
ordinary check=false Cholesky returns failure metadata without modifying input.
Singular LU explicitly allowed by the caller writes its available packed factors.

`FaerFactorization.packed` exports independent native packed storage for partial/
full LU, QR/column-pivot QR and Cholesky. QR `Qbasis`/`Qcoeff` accessors directly
export faer's compact representation, including exact native block dimensions.
Tests reconstruct `FaerHouseholderSequence` from these components and compare
with native Q. This is faer's block representation, not LAPACK's tau format.

The implementation deliberately retains independent factors and staged commits;
it does not yet expose zero-allocation direct input-buffer factorization or
caller-owned workspace. LDLT/block LDLT, full-pivot LU bang and SVD/EVD bang
entry points remain open, along with other outstanding Issue #1 requirements.

Validation: `Pkg.test(allow_reresolve=false)` passes **66,970 assertions across
40 files with ParallelTestRunner 2.10.0 and 16 workers**, on Julia 1.13.3 /
RustCall 0.9.2. Destructive tests contribute 577 assertions across four types,
empty/tall/wide/square shapes, pivoting, packed writeback, live views, independent
factor lifetimes, failure transactions and allowed singular LU. Compact QR
components independently reconstruct native Q through FaerHouseholderSequence.
All 32 README Julia blocks, method ambiguity checks, cargo check, Julia/Rust
formatting and git diff --check pass. Dependency sources remain unchanged.
The complete Issue #1 scope remains active.

### Public simplicial sparse LU components and column orderings

Added an explicit `algorithm=:simplicial` sparse LU path using faer's public
SimplicialLu storage. It owns the actual numeric factors and row/column
permutations used for solves, rather than recomputing dense surrogate factors.
`F.L/F.U` export independent sorted native CSC factors and `F.p/F.q` return
independent FaerPermutation objects with A[p,q]=L*U. The adapter's native solve
uses the same factors for normal/transpose/adjoint systems. CSC/CSR and all four
scalar types are supported; symbolic plan copies and numeric lu! updates retain
column ordering and validate the original sparse pattern.

For this explicit algorithm, :default/:identity mean identity column order;
an arbitrary permutation (including a COLAMD result) selects another ordering.
The default :auto path retains the preexisting high-level faer symbolic/numeric
backend and its fill-reducing selection. Its opaque auto/supernodal factors
still lack component getters here, and those remain part of the full objective.
No dependency private-field casts or source patches are used.

Native Rust tests directly verify L/U reconstruction and coherent permutations
and solve transforms for all four scalar types. The Julia API additionally
checks nonreal complex values, ownership and sparse update behavior.

Validation: `Pkg.test(allow_reresolve=false)` passes **67,244 assertions across
41 files with ParallelTestRunner 2.10.0 and 16 workers**, on Julia 1.13.3 /
RustCall 0.9.2. The new sparse LU suite contributes 274 assertions, covering
CSC/CSR, four types including nonreal coefficients, identity/custom columns,
L/U and p/q reconstruction, normal/transpose/adjoint solves, symbolic reuse,
lu! updates, clone/close, empty/singular systems and invalid controls. An old
negative test was updated to accept the newly supported explicit LU algorithm;
QR's unsupported selection remains rejected. Four native Rust component tests,
all 33 README Julia blocks, method ambiguity checks, cargo check, Julia/Rust
formatting and git diff --check pass. Dependency sources remain unchanged.
Other Issue #1 requirements remain open.

### Caller-owned reusable parallel teams

Enabled faer's threadteam feature and added `FaerTeam(width)` over a retained
threadteam 0.3 pool/reservation. The native owner lends a Par::Team only during
each execution, preserving the exclusive token's lifetime; width one uses
Par::Seq. Close immediately releases the reservation, GC owns the handle, and
the do-block form closes on success/error. Copy/deepcopy build independent
reservations rather than cloning a borrowed authority. Calls and lifecycle are
serialized with the existing matrix owner lock.

`team_mul(team,A,B)` and `mul!(dest,team,A,B[,alpha,beta])` use this authority
for native dense multiplication, with four types, promotion, conversion/view/
transpose snapshots, independent output, alias-safe writes and ignored zero
coefficients. A Rust test directly dispatches onto distinct caller/worker thread
IDs twice using the same reservation; Julia also verifies a 512×512 blocked
product. This proves actual worker dispatch, without making speedup claims.

Only the explicit multiplication entry points use teams so far. Other numeric
methods remain sequential; broader team integration, external pool borrowing
and the full wait/cost tuning API remain part of Issue #1. Small workloads may
use a narrower internal region. Snapshot/output allocations are still present.

Validation: `Pkg.test(allow_reresolve=false)` passes **67,557 assertions across
42 files with ParallelTestRunner 2.10.0 and 16 workers**, on Julia 1.13.3 /
RustCall 0.9.2. Team tests contribute 313 assertions for width one/two, repeated
calls, four types, empty/rectangular/blocked products, promotion, transposes,
views, coefficients, aliasing, task calls, copies, closed/invalid owners and
exception cleanup. The native team test verifies two distinct thread IDs on
two calls through one reservation. All 34 README Julia blocks, method ambiguity
checks, cargo check, Julia/Rust formatting and git diff --check pass. The faer
checkout remains unchanged. Broader team connections and the other Issue #1
requirements remain open.

### Caller-owned aligned scratch and Pfaffian team execution

Added `FaerScratch` over dyn-stack MemBuffer with explicit byte/alignment
queries, monotone reserve!, independent copy/deepcopy, immediate close and GC
ownership. `scratch_requirements(pfaffian,A)` calls faer's exact requirements;
`FaerScratch(pfaffian,A)` prepares a matching buffer. `pfaffian(A;scratch=s)`
mutably borrows this same buffer without implicit growth. Julia and native
checks reject insufficient capacity/alignment before kernel execution; matrix
shape, finiteness and skew-symmetry remain validated. Matrix input is preserved.
A native regression checks pointer identity over repeated calls.

`pfaffian(A;team=t,scratch=s)` also borrows a reusable FaerTeam. With only team
specified, a temporary native workspace is closed in a finally block. Existing
calls without these keywords retain their original path. Tests cover all four
scalar types, complex skew-symmetry, empty/odd cases, changed/reused buffers,
capacity growth, alignment, close/copy, blocked Pfaffian and cross-type reuse.
This extends team integration beyond multiplication.

Pfaffian is currently the connected caller-owned workspace consumer. Other
kernels' scratch queries/consumers and lower-level tuning still require work.
Julia/FFI bookkeeping and the numerical input/output ownership may allocate;
workspace reuse does not claim whole-call zero allocation.

Validation: `Pkg.test(allow_reresolve=false)` passes **67,956 assertions across
43 files with ParallelTestRunner 2.10.0 and 16 workers**, on Julia 1.13.3 /
RustCall 0.9.2. Workspace tests contribute 399 assertions, including exact
requirements, stable capacity, alignment failure, cross-type reuse, independent
copies, error cleanup, odd/invalid inputs, team reuse and a blocked Pfaffian
with independently known value. The Rust test verifies native buffer pointer
identity over repeated calculations. All 35 README Julia blocks, method ambiguity
checks, cargo check, Julia/Rust formatting and git diff --check pass. Dependency
sources remain unchanged. Other workspace consumers and remaining Issue #1
requirements stay open.

### Staged LDLT, block LDLT and full-pivot LU writeback

Added `ldlt!` and `bunchkaufman!` for native matrices and selected Hermitian/real
Symmetric triangles, plus `full_piv_lu!` for rectangular/square native matrices.
All four scalar types retain independent native factor objects before packed
writeback. Views observe updates; closing/changing input leaves returned factors
usable. Failed LDLT without available factors preserves input for check=false;
explicitly allowed singular block LDLT/full LU write their available factors.

`F.packed` now also exports LDLT/block LDLT: strict lower L plus D's diagonal,
with upper zeros. Upper wrappers receive its adjoint. Block off-diagonals remain
in F.D and permutations in F.p, matching faer's split block representation;
input packed storage alone is insufficient without that returned metadata.
Tests independently reconstruct L*D*L' from written input and returned metadata,
including forced complex 2×2 blocks and permutations. Full LU retains F.p/F.q.

These are staged, allocating entry points. Direct mutation of existing native
storage without staging, external scratch for these factorizations, and remaining
SVD/EVD destructive entry points remain work in the full Issue #1 objective.

Validation: `Pkg.test(allow_reresolve=false)` passes **68,453 assertions across
44 files with ParallelTestRunner 2.10.0 and 16 workers**, on Julia 1.13.3 /
RustCall 0.9.2. New tests contribute 497 assertions: four types, empty/rectangular
LU, lower/upper indefinite decompositions, forced nonreal 2×2 blocks, independent
reconstruction from written factors, solve residuals, live views, factor lifetimes
and singular/failure transactions. All 36 README Julia blocks, method ambiguity
checks, cargo check, Julia/Rust formatting and git diff --check pass. Dependency
sources remain unchanged. The remaining Issue #1 requirements stay open.

### SVD low-level workspace, teams and destructive Sigma writeback

Connected faer's low-level SVD for all four scalar types with thin/full/value-only
modes, ephemeral team authority and caller-owned FaerScratch. Requirements include
scratch-resident temporary singular values for complex inputs; the adapter exports
real Julia singular values and independently allocated Julia U/V factors. Queries
with a team borrow its actual Par shape/costs rather than estimating conditions.
The native boundary checks current owner shape and scalar tag before buffer writes.

`svd`/`svdvals` accept scratch/team keywords while unchanged calls retain their
existing route. `svd!`/`svdvals!` compute first, then replace the input by rectangular
Sigma on success. Views observe this writeback; factor outputs survive input close.
This is an explicit staged convention, not a LAPACK temporary-factor layout.
Failed capacity checks, invalid input and native errors preserve input. Temporary
workspaces are closed in finally blocks. Queries/factories/reserve also accept the
bang-function aliases with the same computational mode.

Input/output staging, factor arrays and FFI bookkeeping allocate. Caller-owned
scratch removes implicit workspace growth, not every allocation. EVD destructive
interfaces, other scratch consumers and remaining Issue #1 requirements remain open.

Validation: `Pkg.test(allow_reresolve=false)` passes **69,337 assertions across
45 files with ParallelTestRunner 2.10.0 and 16 workers**, on Julia 1.13.3 /
RustCall 0.9.2. New SVD tests contribute 884 assertions across four types,
empty/tall/wide/square, thin/full/value-only modes, reconstruction/orthogonality,
real singular values, exact Sigma writeback/views, factor independence, team
execution, query aliases, insufficient scratch, invalid inputs and native shape
guards. All six native Rust tests, 37 README Julia blocks, method ambiguity
checks, cargo check, Julia/Rust formatting and git diff --check pass. Dependency
sources remain unchanged. The full Issue #1 scope stays open.

### Staged ordinary and generalized eigen mutation

Added `eigen!`/`eigvals!` for four native scalar types, general matrices and
selected Hermitian/real Symmetric wrappers, including generalized pairs. General
single-matrix calls reuse native Schur/eigenvectors-from-Schur, normalize right
vectors and sort independent Julia outputs before Schur-form writeback. Real
inputs retain real quasi-triangular storage even for complex conjugate pairs.
Self-adjoint eigen! writes eigenvectors; values-only writes a diagonal matrix.

Generalized general pairs use existing native eigen and QZ passes, then write
QZ S/T. Self-adjoint pairs write eigenvectors (or diagonal values for values-only)
to the first parent and diagonal eigenvalues to the second. Distinct parent
storage and matching scalar types are required. These explicitly defined staged
formats do not claim LAPACK work-buffer compatibility. Sort/convergence failures
before writeback preserve inputs; output arrays are independent of later changes.
Temporary Schur/QZ owners are closed in finally blocks.

Checks cover residuals, structural Schur/QZ invariants, specified triangles with
ignored NaNs, orthogonality, sorting, aliases/type rejection and failure input
preservation. EVD caller-owned scratch/team integration, direct allocation-free
updates and other Issue #1 limitations remain open.

Validation: `Pkg.test(allow_reresolve=false)` passes **69,703 assertions across
46 files with ParallelTestRunner 2.10.0 and 16 workers**, on Julia 1.13.3 /
RustCall 0.9.2. Eigen mutation contributes 366 assertions across four types,
empty/scalar/general/self-adjoint/generalized systems, real complex-pair Schur
forms, QZ structure, residuals, normalization/orthogonality, selected triangles,
values-only pair policies, sorting and failure/alias/type guards. All 38 README
Julia blocks, method ambiguity checks, Julia formatting and git diff --check pass.
Native dependency sources remain unchanged. The complete Issue #1 objective is
still active with the remaining limitations above.

### Self-adjoint EVD workspace and teams

Connected faer's low-level self_adjoint_evd for Hermitian/real Symmetric native
wrappers. eigen/eigvals/eigen!/eigvals! accept caller-owned scratch and team
keywords while calls without these options keep their existing route. Queries,
FaerScratch factories and reserve! support both value/vector and bang aliases;
team queries use the actual borrowed Par shape/costs.

The exact requirement includes scratch-resident canonical selected-triangle data,
temporary eigenvalues and kernel workspace. Diagonal imaginary parts and ignored
triangles are discarded; nonfinite active coefficients are rejected. Native owner
shape and scalar tag are checked before writes into Julia output buffers. Outputs
have real eigenvalues and independent vectors, with existing sorting semantics.
Temporary workspace is closed in finally blocks; mutable scratch/ephemeral team
authority remain within the native call. Existing bang writeback policies apply.

Tests include nonreal upper/lower triangles, ignored imaginary NaNs, blocked
analytic tridiagonal spectra, real Symmetric wrappers, input preservation and
shape guards. General/generalized EVD scratch/team, other kernel consumers and
remaining Issue #1 controls still need implementation. Outputs, canonicalization
and bookkeeping do not claim whole-call allocation freedom.

Validation: `Pkg.test(allow_reresolve=false)` passes **70,308 assertions across
47 files with ParallelTestRunner 2.10.0 and 16 workers**, on Julia 1.13.3 /
RustCall 0.9.2. Self-adjoint EVD workspace adds 605 assertions: four types,
nonreal lower/upper data, width one/two, empty/scalar/blocked systems, independent
spectra/residuals/orthogonality, query aliases, reserve, bang policies, ignored
imaginary diagonal NaNs, sorting/failure transactions and native shape guards.
All six native Rust tests, 39 README Julia blocks, method ambiguity checks,
cargo check, Julia/Rust formatting and git diff --check pass. Dependency sources
remain unchanged. Other Issue #1 requirements remain open.

### General EVD workspace and team execution

Added low-level general real/complex EVD adapters behind eigen/eigvals scratch/team
keywords. Real kernels use scratch-resident real/imaginary values and paired real
vectors, then export matching complex conjugate eigenpairs; complex kernels write
complex outputs directly. Native normalization, scalar precision and sorting are
preserved. Requirements/factories/reserve support bang aliases and actual team
conditions. Shape/tag checks precede writes into Julia output arrays.

General eigen!/eigvals! use these options for their eigen pass, then independently
compute the existing sequential native Schur form for staged writeback. The query
describes the eigen pass; it does not cover the additional Schur computation's
internal workspace. Output arrays, sorting and writeback factors allocate.
Unchanged calls retain their former route. Temporary workspace and Schur owners
are closed on errors. Generalized-pair scratch/team and broader controls remain
part of the full Issue #1 objective.

Tests cover four types, real conjugate pairs, normalized right vectors, zero/
empty matrices, values-only, query modes, repeated workspaces, staged writes,
error transactions and shape guards. A cyclic matrix with analytically known
roots tests the larger real/complex paths and orthogonality.

Validation: `Pkg.test(allow_reresolve=false)` passes **70,675 assertions across
48 files with ParallelTestRunner 2.10.0 and 16 workers**, on Julia 1.13.3 /
RustCall 0.9.2. General EVD workspace contributes 367 assertions across four
types, width one/two, empty/scalar/zero/nonnormal/conjugate-pair systems, values
only, normalization, cyclic spectra, query aliases, workspaces, staged writes,
sorting/error transactions and native shape guards. All six native Rust tests,
40 README Julia blocks, method ambiguity checks, cargo check, Julia/Rust
formatting and git diff --check pass. Dependency sources remain unchanged.
The complete Issue #1 objective remains active.

## Definite self-adjoint generalized EVD workspace and teams

Hermitian/real Symmetric pairs now accept `scratch=` and `team=` for
`eigen/eigvals` and their bang variants, for all four native scalar types.
Requirements/factories/reserve use both matrix arguments and the promoted scalar
type. Native owner shapes/tags are checked before writing output buffers.

The adapter composes faer's public LLT, triangular solves and self-adjoint EVD:
canonical selected-triangle A and B copies reside in caller scratch, along with
temporary eigenvalues. LLT and EVD share the maximum of their kernel workspace
requirements through a sequential reborrow. Every numerical stage receives the
explicit team's authority. B must be positive definite; eigenvectors satisfy
A V = B V diagonal(values) and V' B V = I. Imaginary diagonal parts and unused
triangles are ignored independently. Default calls use temporary sequential
workspace with the same pipeline.

Sorting, output arrays, promotion and staged bang writeback still allocate.
B-factorization and sorting errors preserve both bang inputs. General
nonsymmetric pairs, remaining kernels and tuning controls still require further
Issue #1 work. The dependency checkout is unchanged.

Validation: `Pkg.test(;allow_reresolve=false)` passes **71,229 assertions across
49 files**, using ParallelTestRunner 2.10.0 with 16 workers on Julia 1.13.3 /
RustCall 0.9.2. The new generalized workspace tests contribute 554 assertions:
four scalar types, empty/scalar/dense/64-dimensional pairs, independently
selected triangles, B-orthonormality, values-only, staged writeback, promotion,
sorting/error transactions and native shape guards. All six native Rust tests
and all 41 README Julia blocks pass. Julia/Rust formatting, method ambiguity
checks and git diff --check pass. The full Issue #1 goal remains incomplete.

## Generalized EVD scratch and explicit teams

General native matrix pairs now support `scratch=` and `team=` for
`eigen/eigvals/eigen!/eigvals!`, with four scalar types and scalar promotion.
Requirement queries, factories and reserve accept both inputs. Native shape/tag
checks guard output writes. Writable A/B copies and packed real eigenvalues,
beta and vectors are allocated within caller scratch. The adapter composes the
public generalized EVD driver with the existing real-QZ scratch correction.
Right vectors are normalized; real conjugate pairs become complex Julia arrays.
Default calls use the same pipeline with temporary sequential workspace.

Bang options apply to the eigen pass; a separate sequential QZ computes S/T
writeback. The query excludes this second decomposition's internal workspace,
output arrays, sorting and promotion. Invalid finite-eigenvalue eigenvectors
caused by faer's repeated-shift defect are recovered from native sequential SVD
nullspaces, with sufficient-nullity checks. This exceptional path allocates
additional matrices and internal workspace. Defective/nonfinite-eigenvalue
vector recovery and homogeneous alpha/beta exposure remain unfinished.

The standalone identity-pencil Rust MWE in
`docs/reproducers/faer-gevd-repeated.rs` exits 101 with NaN right vectors using
an isolated pinned git Cargo dependency, independently of Julia/RustCall.
[faer #217](https://github.com/tensor4all/faer/issues/217) contains the exact
manifest/source, measured output, toolchain, concrete protected-denominator
fix proposal and regression cases. Latest remote HEAD f817d003 was inspected
and retains the same helper, but was not executed. No dependency sources were
modified. No new RustCall defect was observed.

Validation: canonical `Pkg.test(;allow_reresolve=false)` passes **71,577
assertions across 50 files** on Julia 1.13.3 / RustCall 0.9.2, using
ParallelTestRunner 2.10.0 with 16 workers. The generalized workspace file adds
348 assertions across four types, width one/two, empty/scalar/nonsymmetric
conjugate-pair systems, dense 64-dimensional known spectra, repeated identity
pencils, promotion, shape guards, sorting/error transactions and staged QZ
writeback. The test includes method ambiguity checks. All six native tests
and 42 README Julia blocks pass, as do Julia/Rust formatting, cargo check and
git diff --check. The frozen 76-row inventory and dependency sources are
unchanged. The full Issue #1 objective remains active and incomplete.

## Homogeneous generalized eigenvalues and infinite-vector recovery

General `FaerMatrix` pairs now accept `homogeneous=true` on `eigen/eigvals`.
The native adapter exports alpha and beta before division: eigen returns
independent Julia `(α, β, vectors)` arrays and values-only returns `(α, β)`.
The homogeneous default uses native order (`sortby=nothing`) without forming
quotient eigenvalues. Explicit sorting uses the quotient and applies a common
permutation to alpha/beta/vectors. Ordinary calls keep GeneralizedEigen and
quotient-vector return types. Four scalar types, promotion and scratch/team
options are preserved; scratch requirements are unchanged.

Homogeneous residuals beta*A*v = alpha*B*v cover finite and infinite eigenvalues.
Beta zero with nonzero alpha denotes infinity, while alpha=beta=0 remains
indeterminate and is retained by values-only queries. Default quotient queries
produce NaN for 0/0. The option applies to general pairs, not the specialized
self-adjoint pair or bang APIs.

The faer #217 workaround now groups projectively equal normalized alpha/beta
pairs and forms beta*A-alpha*B before native SVD recovery. This includes repeated
infinite eigenvalues. Requested teams propagate into the recovery SVD, whose
additional temporary workspace still lies outside the primary caller buffer.
Insufficient nullity or invalid indeterminate-vector recovery is rejected.
No new RustCall or faer defect was found in this phase; dependency sources remain
unchanged. Further extreme-scale/defective-pencil coverage and the broader
Issue #1 requirements remain unfinished.

Validation: `Pkg.test(;allow_reresolve=false)` passes **71,950 assertions across
51 files**, using ParallelTestRunner 2.10.0 with 16 workers on Julia 1.13.3 /
RustCall 0.9.2. The new file contributes 373 assertions covering all four types,
width one/two, empty/scalar/conjugate-pair systems, sorting correspondence,
promotion, repeated finite/infinite eigenvalues, singular B, indeterminate
values, homogeneous residuals and owner errors. Method ambiguity checks pass.
All six native Rust tests and 43 README Julia blocks pass. Cargo check,
Julia/Rust formatting and git diff --check pass. The frozen inventory retains
76 rows and dependency sources are unchanged. Issue #1 remains incomplete.

## Left generalized eigenvectors

General native matrix pairs now accept `eigen(A,B;side=:left)`, with right
remaining the default. Left vectors are normalized and satisfy V' A = Λ V' B,
or diagonal(beta) V' A = diagonal(alpha) V' B in homogeneous mode. Four scalar
types, promotion, sorting, caller scratch and explicit teams are supported.
The adapter connects the public driver's left output; packed real conjugate
pairs require conjugating the packed imaginary component for this convention.
The current backend has common one-sided scratch requirements.

The direct left-only faer path corrupts a 64-dimensional regular pencil's
spectrum: right-only and both-output yield 64/64 finite values, while left-only
yields 4/64 finite values and nonfinite vectors. The independent safe-Rust MWE
`docs/reproducers/faer-gevd-left.rs` uses matching query/output flags and fresh
zero buffers. It exits 101 on both pinned d5be1473 (0.28.1) and latest f817d003
(0.28.2), actually executed in isolated git-dependent projects. Both-output
left homogeneous max residual is 3.180292834636944e-11.

[faer #219](https://github.com/tensor4all/faer/issues/219) includes exact
Cargo.toml/source, outputs, distinctness from #37/#217, and a concrete interim
native fix: allocate a temporary right output for left-only requests, route
through both-output accumulation, and increase the left-only scratch query.
The exact underlying optional-output update is not yet isolated; the report
makes that distinction and does not claim an untested patch was verified.

Colmat detects nonfinite direct-left output and recomputes the mathematically
equivalent adjoint pencil with the native right driver, then conjugates alpha
and beta. It reuses the caller buffer/team after the first native borrow ends;
extra input/output copies allocate separately. Owners are closed on errors.
Invalid repeated vectors can still use homogeneous adjoint nullspace recovery.
Simultaneous left/right output, single-matrix side selection, bang side
selection and broader Issue #1 controls remain unfinished.

Validation: canonical `Pkg.test(;allow_reresolve=false)` passes **72,155
assertions across 52 files**, using ParallelTestRunner 2.10.0 with 16 workers
on Julia 1.13.3 / RustCall 0.9.2. The new file contributes 205 assertions for
four types, width one/two, empty/scalar/nonsymmetric/conjugate-pair systems,
homogeneous sorting, repeated finite/infinite eigenvalues, mixed promotion,
errors and dense 64-dimensional analytic spectra. Method ambiguity checks,
all six native tests and all 44 README Julia blocks pass. Cargo check,
Julia/Rust formatting and git diff --check pass. The frozen inventory retains
76 rows and dependency sources remain unchanged. Issue #1 remains incomplete.

## Simultaneous generalized left/right eigenvectors

General native pairs now support `eigen(A,B;side=:both)`, returning independent
Julia arrays `(values,left,right)`, or `(α,β,left,right)` with homogeneous mode.
Both sides are produced by the same public native generalized EVD driver call,
not two independently ordered decompositions. Four scalar types, promotion,
sorting and explicit teams are supported. Left/right columns are independently
normalized and receive the same eigenvalue permutation; biorthogonal scaling
is not imposed.

Workspace queries, factories and reserve accept `side=:right/:left/:both`.
Real simultaneous output needs a second packed eigenvector matrix in caller
scratch; complex output writes directly to the two Julia matrices. Native
owner shape/tag validation precedes output writes. The scratch requirement
uses the requested mode and actual team conditions; caller storage does not
grow implicitly. Output arrays and exceptional repeated-value SVD recovery
still allocate separately. Single-sided output retains its prior API.

Packed real left vectors use conjugated imaginary components, while right
vectors use the original packed imaginary components. Both native outputs are
converted against the same alpha/beta sequence. The existing homogeneous
nullspace recovery applies independently to invalid left/right columns. The
known #219 left-only driver defect does not affect the tested simultaneous
path, including the same dense 64-dimensional analytic pencil.

Side selection for single matrices and destructive routines, remaining
workspace/tuning controls and broader Issue #1 features remain unfinished.
No new RustCall or faer defect was observed in this phase. Dependency sources
and the frozen inventory are unchanged.

Validation: `Pkg.test(;allow_reresolve=false)` passes **72,543 assertions
across 53 files**, using ParallelTestRunner 2.10.0 and 16 workers on Julia
1.13.3 / RustCall 0.9.2. The new file contributes 388 assertions covering four
types, width one/two, empty/scalar/conjugate-pair systems, simultaneous left/
right residuals, homogeneous sorting, repeated finite/infinite eigenvalues,
dense 64-dimensional analytic spectra, promotion, insufficient workspace,
reserve and owner errors. Method ambiguity checks pass. All six native Rust
tests and 45 README Julia blocks pass. Cargo check, Julia/Rust formatting and
git diff --check pass. The frozen inventory retains 76 rows and dependency
sources are unchanged. The full Issue #1 goal remains active and incomplete.

## Single-matrix left and simultaneous EVD

General native matrices now accept `eigen(A;side=:left/:right/:both)` across
four scalar types. The public native EVD driver's left/right output flags are
connected directly. Simultaneous output uses one EVD call and a common spectrum;
real input stores a second packed vector matrix in caller scratch, while
complex output writes directly to two independent Julia arrays. EVD's packed
real left-vector convention differs from generalized EVD; no imaginary-sign
flip is applied for the single-matrix driver. Independent left/right residuals
verify that conversion rather than assuming the two drivers share a layout.

The default right return remains `Eigen`. Left-only quotient results use named
`(values,vectors)` tuples and both-side output uses `(values,left,right)`.
Generalized left-only quotient output follows the same named-tuple convention,
preserving its values/vectors fields and homogeneous mode. Left output data
therefore does not expose ordinary right-factorization solve/reconstruction
operations. Self-adjoint wrappers and bang routine return types are unchanged.

Queries, factories and reserve accept the requested side and actual team;
real simultaneous mode needs additional packed scratch. All selected vector
columns are normalized; sorting applies a common column permutation. Inputs
remain unchanged and native shape/tag checks guard output writes. Julia output
arrays and sorting allocate independently. No new upstream RustCall or faer
defect was observed. The dependency checkout is unchanged.

Single-matrix tuning/balancing controls, destructive side selection, remaining
workspace kernels and the broader Issue #1 requirements remain unfinished.

Validation: canonical `Pkg.test(;allow_reresolve=false)` passes **73,002
assertions across 54 files** on Julia 1.13.3 / RustCall 0.9.2, using
ParallelTestRunner 2.10.0 and 16 workers. The new sided-EVD file contributes
435 assertions across four types, width one/two, empty/scalar/conjugate-pair
systems, both residual equations, shared sorting, dense nonnormal
64-dimensional analytic spectra, capacity/reserve and owner guards. A further
24 assertions verify the generalized left result's named-tuple contract.
Method ambiguity checks, all six native tests and all 46 README Julia blocks
pass. Cargo check, Julia/Rust formatting and git diff --check pass. The frozen
inventory retains 76 rows and dependency sources remain unchanged. The full
Issue #1 objective remains active and incomplete.

## Factor reconstruction, typed conversion and native matrix pairs

`reconstruct(F)` now returns independent native storage for the supported dense
owned LU/QR/Cholesky/LDLT factors, Schur/Hessenberg and tridiagonal/bidiagonal
reductions. Typed `Matrix{T}(F)` and `FaerMatrix{T}(F)` conversions are available
for these formats. Same-type native conversion returns a freshly reconstructed
owner; different scalar types follow Julia conversion rules. Temporary owners
are closed when conversion fails, preserving the factorization.

Explicit simplicial sparse LU adds `reconstruct/Matrix/FaerMatrix/
FaerSparseMatrix/SparseMatrixCSC/sparse`, including typed conversions. The
implementation multiplies actual native L/U and applies independent inverse
row/column permutations through native sparse kernels. It never copies the
retained input snapshot. Tests poison that snapshot and verify reconstruction
still reflects the numeric factors, then verify lu! updates. Reconstruction
returns CSC storage and does not promise the original stored-zero pattern.
Auto/supernodal sparse LU and sparse QR reconstruction remain unexposed.

Generalized Schur/Hessenberg return `(A,B)` named fields through native products
Q*S*Z' (or Q*H*Z') and Q*T*Z'. Both outputs own independent native storage and
remain usable after closing F. Pair shape/owner checks precede products and
partial outputs are closed if the second reconstruction fails. Existing Schur/
Hessenberg reconstruction was already present; the mapping no longer lists it
as entirely missing. No new upstream RustCall or faer defect was observed.

Validation: canonical `Pkg.test(;allow_reresolve=false)` passes **74,442
assertions across 55 files**, using ParallelTestRunner 2.10.0 with 16 workers
on Julia 1.13.3 / RustCall 0.9.2. The new reconstruction file contributes 1,440
assertions across four scalar types, empty/3-dimensional factors, nine dense
factorization/reduction families, CSC/CSR, custom row/column permutations,
actual-factor provenance, numeric updates, typed CSC index conversion,
closed-owner independence, conversion errors and pair shape guards. Method
ambiguity checks pass. All six native tests and 47 README Julia blocks pass.
Julia/Rust formatting and git diff --check pass. The frozen inventory retains
76 rows and dependency sources are unchanged. Issue #1 remains incomplete.

## Retained native SVD / self-adjoint EVD and saved-factor pseudoinverses

`svd(A;native=true)` and `eigen(Hermitian(A);native=true)` now return a
`FaerSpectralFactor` containing actual Rust faer Svd/SelfAdjointEigen objects,
not just copied Julia components or an input snapshot. Four scalar types,
thin/full rectangular and empty SVD, selected triangles, real Symmetric
wrappers, EVD sorting and explicit teams are supported. Ordinary calls retain
their prior Julia result types. Construction uses backend-owned workspace;
caller scratch and native bang construction remain unexposed.

`pinv(F;atol,rtol,team)` calls the public pseudoinverse-from-saved-factors kernel
without decomposing again. Reconstruction uses the retained object's
DenseSolveCore implementation. Matrix/native typed conversions, rank,
svdvals or self-adjoint eigvals/eigvecs reuse the same owner. Component getters
return independent native matrices and real Julia vectors. Eigen order is a
common values/vectors permutation; pinv/reconstruction are order invariant.
Component iteration, copy/deepcopy, issuccess and idempotent immediate native
close are supported. Extracted parts and cloned factors survive source/owner/
team closure; changing a component copy does not mutate the retained factors.

Native dtype/kind/length checks precede pointer writes. Shape checks guard
matrix results. Native unit coverage rejects wrong dtype/length and closed
owners without modifying output sentinels, using sufficiently sized/aligned
buffers so validation failures do not rely on undefined behavior. Another
native test mutates the source after factorization and verifies reconstruction
and thresholded pseudoinverse still use the saved factors.

Float32 cutoff promotion reuses the existing downward-rounded cutoff helper.
A boundary test initially compared against an input eigenvalue that the EVD
rounded by one ulp. The final test positions the Float64 threshold immediately
below the actual saved value, isolating cutoff conversion from decomposition
rounding; the pseudoinverse correctly follows saved singular/eigenvalues.
No new upstream RustCall or faer defect was established in this phase.

Validation: canonical `Pkg.test(;allow_reresolve=false)` passes **75,626
assertions across 56 files**, using ParallelTestRunner 2.10.0 with 16 workers
on Julia 1.13.3 / RustCall 0.9.2. The new file contributes 1,184 assertions for
all four types, thin/full empty/tall/wide SVD, selected triangles, native-owner
reuse, Moore-Penrose identities, negative/zero eigenvalues, tolerances, sorting,
copy/deepcopy, extracted component independence, team lifetime, shape/owner
checks, Float32 boundary cutoffs, Symmetric wrappers and iteration. All eight
native tests and all 48 README Julia blocks pass. Cargo check, Julia/Rust
formatting and git diff --check pass. The frozen 76-row inventory and dependency
sources are unchanged. Remaining spectral/workspace controls and the full
Issue #1 objective are still incomplete.

## Caller-owned scratch for saved spectral pseudoinverses

`pinv(F::FaerSpectralFactor;scratch=...,atol,rtol,team)` now reuses caller-owned
aligned workspace for both retained SVD and self-adjoint EVD. Requirement
queries, factories and reserve accept `pinv,F` and the actual team. The native
query validates owner dtype/kind/shape before writing byte/alignment metadata;
execution repeats that validation and borrows OwnedScratch mutably. Kernel
workspace comes directly from its MemStack. No SVD/EVD is repeated, no implicit
reserve occurs, and the independent native result matrix allocates separately.

The existing backend-owned path and new caller-owned path share the same
pseudoinverse kernel/tolerance implementation. Empty factors skip the kernel
but still require an open initialized workspace. Copy/deepcopy workspaces are
independent and reusable; source/factor and scratch ownership remain separate.
Native factor construction and direct matrix pinv(A) retain their existing
workspace interfaces and are not claimed covered by this addition.

A native test verifies repeated saved-SVD/EVD pseudoinverse calls preserve the
workspace address. Insufficient capacity preserves the preexisting output,
and forged factor dimensions do not modify query sentinels. No new upstream
RustCall or faer defect was observed; dependency sources remain unchanged.

Validation: canonical `Pkg.test(;allow_reresolve=false)` passes **76,619
assertions across 57 files**, using ParallelTestRunner 2.10.0 with 16 workers
on Julia 1.13.3 / RustCall 0.9.2. The new file contributes 993 assertions for
four scalar types, thin/full empty/tall/wide/rank-deficient SVD, selected
triangles and indefinite/zero self-adjoint spectra, repeated buffers, workspace
copy/deepcopy, capacity/alignment errors, explicit reserve, tolerance and
owner/shape checks, and 64-dimensional retained-factor reconstruction kernels.
Method ambiguity checks, all nine native tests and 49 README Julia blocks
pass. Cargo check, Julia/Rust formatting and git diff --check pass. The frozen
inventory retains 76 rows. The full Issue #1 objective remains incomplete.

### Retained spectral factor solves

Retained native SVD and self-adjoint EVD now support left/right solves,
transpose/adjoint factors, three-argument `ldiv!`/`rdiv!`, square two-argument
mutation, and square nonsingular `inv`. Native matrix products apply the
truncated minimum-norm solution to each RHS without constructing a full
pseudoinverse or decomposing again. Signed eigenvalues are retained; scalar
promotion affects computation while clipping follows the factor's precision.
The same native factor can be used after closing its source matrix.

Projection/result arrays allocate internally. Right solves stage adjointed
RHS/results. Destination scalar conversion precedes writeback. Caller-owned
solve scratch and allocation-free mutation remain unimplemented. The new focused
file checks 2,476 assertions including all four scalar types and RHS promotions,
thin/full tall/wide SVD, transpose/adjoint, signed/singular EVD, tolerance/team,
closed owners, destination conversion failure, aliasing and empty dimensions.

Validation: Julia 1.13.3 / RustCall 0.9.2, canonical
`Pkg.test(; allow_reresolve=false)` passed 79,095 assertions across 58 files with
16 ParallelTestRunner workers. All nine native tests and all 49 README Julia
blocks passed. Method ambiguity detection returned none. Julia/Rust formatting
and `git diff --check` passed. The dependency checkout remains untouched. The
full Issue #1 objective remains incomplete.

### Explicit simplicial sparse QR components

`qr(A; algorithm=:simplicial, ordering=:identity|permutation)` now retains
public native simplicial symbolic and numeric QR components. The default
ordering for this path is identity. `F.Q` returns an independent thin native
m-by-n matrix, `F.R` an independent sparse CSC n-by-n factor, and `F.pcol`
(or `F.p`) the column permutation satisfying `Q*R = A[:,pcol]`. Q extraction
reverses the retained Householder reflectors with m-by-n storage; it does not
materialize an m-by-m identity. Reconstruction multiplies the current native
factors and undoes the column permutation entirely in Rust. It does not reuse
the stored source snapshot or refactorize. Typed Matrix/FaerMatrix conversions,
copy/deepcopy and symbolic numeric updates are supported.

The new independent test file passes 1,078 assertions across all four native
scalar types, CSC/CSR, empty/tall/square shapes, custom column ordering, Q
orthogonality and Q*R reconstruction, least-squares solutions, source mutation,
closed owners, copied factors, numeric updates, and typed reconstruction.
Method ambiguity detection is empty. Auto/supernodal components, full Q,
rank-deficient minimum-norm and transpose/adjoint QR solves, and caller-owned
QR scratch remain unimplemented. The full Issue #1 objective remains active.

Final validation on Julia 1.13.3 / RustCall 0.9.2: canonical
`Pkg.test(; allow_reresolve=false)` passed 80,173 assertions across 59 files with
16 ParallelTestRunner workers. The old sparse-QR unsupported-algorithm assertion
was updated to verify the newly supported simplicial analysis. All nine native
Rust tests and 50 README Julia blocks passed; method ambiguity detection,
Julia/Rust formatting and `git diff --check` passed. Both inventories retain 76
feature rows. The dependency checkout remains clean at d5be1473.

### Sparse simplicial QR transpose and adjoint solves

Explicit simplicial full-column-rank QR now solves transposed/adjoint systems
with the minimum-norm solution. It gathers the RHS by the retained column
permutation, solves the sparse transposed/conjugated R triangular system and
applies the saved Householder reflectors in reverse. Thin Q extraction shares
the same native reflector application. The solve does not construct Q, a full
pseudoinverse or a new factorization when scalar types match. The underdetermined
output has m rows for an original m-by-n QR. A zero R diagonal is rejected.
Three-argument `ldiv!` supports rectangular factors and two-argument mutation
requires square factors. Auto QR still rejects transpose/adjoint operations.

The new file passes 1,179 focused assertions across four scalar types, CSC/CSR,
identity/custom order, square/tall/empty dimensions, complex transpose versus
adjoint, vector/matrix/empty RHS, mutation, RHS scalar promotions, owner closure,
source snapshot mutation and singular rejection. All 1,078 existing sparse QR
component assertions and ambiguity detection pass. Mixed scalar solves continue
to refactor the retained source snapshot using the existing sparse promotion
path; native mixed-factor reuse, rank thresholds, rank-deficient solves and
caller scratch remain unimplemented.

Final validation on Julia 1.13.3 / RustCall 0.9.2: canonical
`Pkg.test(; allow_reresolve=false)` passed 81,352 assertions across 60 files with
16 ParallelTestRunner workers. All nine native Rust tests and all 51 README
Julia blocks passed. Method ambiguity detection, Julia/Rust formatting and
`git diff --check` passed. The dependency checkout remains untouched. The full
Issue #1 objective remains incomplete.

### Direct COLAMD ordering for simplicial sparse LU/QR

`lu`/`qr` and `FaerSparseSymbolic` now accept `ordering=:colamd` for explicit
`algorithm=:simplicial`. The existing native COLAMD adapter computes the column
permutation from the stored pattern once during analysis, including stored
zeros. `dense_row`, `dense_col` and `aggressive` controls are forwarded unchanged.
The native symbolic plan owns a copy; the temporary permutation is closed after
analysis, including on errors. Numeric `lu!`/`qr!` updates and copied factors
retain the chosen column ordering. Default identity and explicit permutation
behavior remain available. Auto/supernodal ordering controls are still missing.

Focused validation passes 1,304 assertions across all four scalar types,
CSC/CSR, square/tall/empty shapes, default/custom COLAMD controls, nonidentity
orders, reconstruction from native factors, normal/adjoint solves, numeric
updates, independent input/plan lifetime, copied factors and invalid/closed
inputs. No dependency defect was observed in this change.

Final validation on Julia 1.13.3 / RustCall 0.9.2: canonical
`Pkg.test(; allow_reresolve=false)` passed 82,656 assertions across 61 files with
16 ParallelTestRunner workers. All nine native Rust tests and 52 README Julia
blocks passed. Method ambiguity detection, Julia formatting and
`git diff --check` passed. The dependency checkout remains untouched at d5be1473.
The full Issue #1 objective remains incomplete.

### Low-level auto/supernodal sparse QR ownership

The active feature branch replaces opaque high-level QR owners with the public
low-level SymbolicQr/QrRef API, an Arc-owned symbolic analysis and private numeric
index/value buffers. Checked buffer initialization precedes every QrRef borrow;
copying clones numeric data and shares immutable analysis. This prepares the
remaining workspace interface without inspecting private upstream storage.
The existing sparse factor focused file passed 2,185 assertions after migration.
An explicit `algorithm=:supernodal` selector and focused four-type CSC/CSR tests
pass 128 assertions. Caller-owned scratch and component
extraction are not yet exposed, and Issues #1/#2 remain incomplete.

Final validation: canonical Pkg.test passed 82,784 assertions across 62 files
with 16 workers. All 10 native tests, 52 README blocks, method ambiguity detection,
Julia/Rust formatting and git diff --check passed. The first full run mixed an
updated test with already loaded code; the next identified a Matrix(Vector)
conversion error in the new test. The final fixed-source run passed completely.

### Retained auto/supernodal QR solve workspace

`scratch_requirements(ldiv!, F, B)`, `FaerScratch(ldiv!, F, B)` and
`reserve!(s, ldiv!, F, B)` query and reserve native sparse QR solve workspace.
`ldiv!(X, F, B; scratch=s)` supplies the caller's aligned buffer to the retained
QrRef solve. This interface supports normal auto/supernodal QR and matrix RHS
matching the factor scalar type. Native RHS/result storage and destination copy
allocate outside scratch. No source refactorization is performed. Factor
construction and simplicial/transpose/adjoint solves retain their previous paths.

Focused verification passed 587 assertions: all four types, CSC/CSR, empty and
rectangular inputs, repeated workspace reuse, source mutation, close, invalid
inputs, reserve and insufficient capacity preserving the destination. Canonical Pkg.test passed 83,371 assertions / 63 files with 16 workers.
All 10 native tests and 53 README blocks passed, as did method ambiguity
detection, Julia/Rust formatting and git diff --check. Issues #1/#2 remain incomplete.

### Auto/supernodal QR COLAMD controls and permutation

Auto and explicit supernodal sparse QR now forward `dense_row`, `dense_col`
and `aggressive` to the public native symbolic COLAMD analysis. Both
`ordering=:default` and `ordering=:colamd` select that path. `F.p/F.pcol`
returns an independent copy of the actual retained column permutation for
these factors. Numeric updates preserve immutable symbolic analysis and its
ordering. Identity/custom permutations remain explicit simplicial features.

Focused verification passed 576 assertions across four types, CSC/CSR,
auto/supernodal, empty/single-column/tall inputs and aggressive absorption
on/off. Permutations match standalone native COLAMD, and solves/copies/numeric
updates preserve the expected results. Canonical Pkg.test passed 83,947 assertions / 64 files with 16 workers.
All 10 native tests and 53 README blocks passed; ambiguity detection,
Julia/Rust formatting and git diff --check passed. This does not complete Issues #1/#2.

### Sparse QR numeric factor construction workspace

`factorize(plan,A;scratch=s)` now accepts caller-owned workspace for native
auto/supernodal QR numeric construction. The matching requirement/factory/reserve
APIs dispatch on `factorize`, symbolic plan and numeric matrix. Pattern validation
is shared with ordinary factorization before the kernel reads buffers. Native
numeric index/value buffers remain independently owned; scratch, input and plan
may be closed after construction. Symbolic analysis, snapshots and owned factor
storage allocate outside scratch. Explicit simplicial construction is unchanged.

Focused verification passed 393 assertions across four scalar types, CSC/CSR,
empty/square/tall inputs, repeated workspace reuse, independent factors built
from different numeric values, reserve, insufficient capacity, pattern mismatch,
shape mismatch and owner closure. Canonical Pkg.test passed 84,340 assertions / 65 files with 16 workers.
All 10 native tests and 54 README blocks passed, together with ambiguity
detection, Julia/Rust formatting and git diff --check.
Issues #1/#2 remain incomplete.

### Native matrix-free operator composition

`FaerLinearOperator * FaerLinearOperator` retains independent native copies of
both operands and applies them sequentially without forming the coefficient
product. Transpose/adjoint reverse the operand application order, conjugation
propagates to both, and nested compositions use recursive native scratch
requirements. Intermediate RHS storage is carved from the existing MemStack,
without a per-application intermediate heap allocation. Matrix conversion is
explicit. `.coefficients` rejects compositions because there is no single base
storage. Existing compositions cannot yet be promoted to a new scalar type.

Focused tests passed 226 assertions: all four scalar types, dense/sparse inputs,
empty/rectangular dimensions, conjugate/transpose/adjoint, copies and closed
operands, nesting, CG on a composed normal operator, product application and
invalid shapes/owners. Canonical Pkg.test passed 84,566 assertions / 66 files with 16 workers.
All 10 native tests and 55 README blocks passed, along with ambiguity detection,
Julia/Rust formatting and git diff --check. Julia callbacks and the
remaining Issue #2 work are still pending.

Upstream API feedback: tensor4all/faer PR #228 adds a read-only public
SymbolicQr::raw accessor needed for safe QR component inspection. The standalone
MWE fails on b4fb784 with E0599; the patch passes both forced-variant integration
coverage and an executable doctest. This is an API extension, not a numerical
bug. See docs/reproducers/faer-sparse-qr-symbolic-accessor.md. The dependency
checkout remains unchanged; the upstream PR has not been adopted into Colmat.
The PR was merged upstream as fac26dc2a346f2be41f2011fba2bc8e4d7ff22f5
on 2026-11-03; Colmat adoption still requires a dependency update and validation.

### Native operator tree scalar promotion

Mixed scalar operator application and composition now widen native operator
leaves recursively, preserving dense/CSC/CSR/diagonal/identity representation,
composition structure and transpose/conjugate flags. No coefficient product is
materialized for conversion. Matching scalar inputs retain their existing owner.
Narrowing and complex-to-real conversions are rejected at the native boundary.

Focused verification passed 400 assertions across all 16 scalar pairs and five
leaf representations, including composed mixed RHS, nested mixed composition,
independent operand ownership and closure. Canonical Pkg.test passed 84,966
assertions / 67 files with 16 workers. All 10 native tests and 55 README blocks
passed, as did ambiguity detection, Julia/Rust formatting and git diff --check. Julia
callbacks and remaining Issue #1/#2 functionality remain pending.

### Retained LU inverse-action preconditioners

`FaerLinearOperator(F)` accepts square nonsingular native partial-pivot LU
factors and owns copies of L/U and pivots. Native LinOp/BiLinOp application calls
the public low-level LU solve kernels with supplied scratch. No inverse matrix
or refactorization is performed. Copy, composition, conjugation, transpose and
adjoint retain the inverse-action contract. Scalar promotion converts saved
L/U directly and preserves pivots, including inside composed operators. Source
factors and inputs can be closed independently. CG requires a positive-definite
inverse action; this requirement does not hold for arbitrary LU.

Focused verification passed 167 assertions across four types, empty/singleton/
square shapes, independent copy/close, CG, all scalar promotion pairs, transpose
versus adjoint, singular and rectangular rejection. Canonical Pkg.test passed 85,133 assertions / 68 files with 16 workers.
All 10 native tests and 56 README blocks passed, along with ambiguity detection,
Julia/Rust formatting and git diff --check. An additional standalone 12-assertion
check used A=[0 2;3 1] (complex variant adds 0.2im to A[1,2]) to verify
row-pivoted normal/transpose/adjoint actions across all four types.
Other dense and sparse decomposition preconditioners,
callbacks and the remaining Issue #1/#2 work are still pending.

### Retained Cholesky preconditioners

Native Cholesky factors can now construct FaerLinearOperator inverse actions.
The operator independently owns lower/upper triangular factors and applies
native nonunit triangular solves. It shares ownership, composition and saved
factor scalar-promotion machinery with LU preconditioners; Cholesky uses no
LU-specific solve workspace. Normal/conjugate/transpose/adjoint actions do not
form an inverse or refactor. Failed factors raise PosDefException.

Focused verification passed 737 assertions across four factor/RHS scalar types,
both input triangle choices, empty/singleton/square shapes, copies, closed source
factors, mixed promotion, transpose/adjoint, CG and failed-factor rejection.
Canonical Pkg.test passed 85,870 assertions / 69 files with 16 workers.
All 10 native tests and 57 README blocks passed, along with ambiguity detection,
Julia/Rust formatting and git diff --check. Other factor types, sparse
preconditioners, callbacks and remaining Issue #1/#2 scope are pending.

### Retained full-pivot LU preconditioners

FaerLinearOperator accepts native full-pivot LU factors, independently retaining
L/U plus both row and column permutations. Normal/transpose/adjoint application
calls the corresponding native full-pivot solve kernels and scratch queries.
Scalar promotion converts saved factor values and preserves both permutations,
without refactorization. This shares inverse-action copy/composition machinery
with partial-pivot LU and Cholesky preconditioners.

Focused verification passed 337 assertions across four factor/RHS scalar types,
empty/singleton/square inputs, copies and source closure, transpose versus adjoint
and singular rejection. The nontrivial fixture includes A=[0 2;3 100] (complex
variant adds 0.3im at A[1,2]), forcing both row and column pivoting. Canonical Pkg.test passed 86,207 assertions / 70 files with 16 workers.
All 10 native tests and 58 README blocks passed, along with ambiguity detection,
Julia/Rust formatting and git diff --check. Issues #1/#2 remain incomplete.

### Retained QR inverse-action preconditioners

FaerLinearOperator accepts square native QR and column-pivoted QR factors. It
owns compact Householder basis/coefficients, R and column permutation and applies
native column-pivoted QR solve kernels (identity permutation for unpivoted QR).
No explicit Q or inverse is constructed, and factors are never recomputed.
Mixed scalar promotion converts retained components; transpose/adjoint and
independent copy/close reuse the same native representation. Empty square
factors are handled without entering kernels requiring nonzero block size.
Zero R pivots and rectangular factors are rejected; rank tolerance and
rank-deficient pseudoinverse preconditioning remain unsupported.

Focused verification passed 677 assertions across all factor/RHS scalar pairs,
pivoted/unpivoted, empty/singleton/square, source closure, copies, transpose versus
adjoint, CG and zero-pivot rejection. Canonical Pkg.test passed 86,884 assertions / 71 files with 16 workers.
All 10 native tests and 59 README blocks passed, along with ambiguity detection,
Julia/Rust formatting and git diff --check. Remaining Issue #1/#2 work is still pending.

### Retained LDLT inverse-action preconditioners

FaerLinearOperator accepts native self-adjoint LDLT factors, independently
owning L and D. Native low-level LDLT solve uses the operator's scratch without
inverse construction or refactorization. Transpose applies the conjugated action
and adjoint the ordinary action for these self-adjoint factors. Copy/composition
and mixed scalar promotion reuse retained components. Zero D pivots are rejected;
indefinite actions are supported with general iterative solvers, not assumed to
satisfy CG's positive-definiteness contract.

Focused verification passed 675 assertions across factor/RHS scalar pairs, both
input triangles, empty/singleton/square, copy/close, transpose versus adjoint,
BiCGSTAB on an indefinite system and singular/closed-factor rejection.
Canonical Pkg.test passed 87,559 assertions / 72 files with 16 workers.
All 10 native tests and 60 README blocks passed, along with ambiguity detection,
Julia/Rust formatting and git diff --check. Block-pivoted LDLT, sparse factor
preconditioners, callbacks and remaining Issue #1/#2 work remain pending.

### Retained Bunch–Kaufman preconditioners

FaerLinearOperator accepts native block-pivoted LDLT factors, retaining L,
block diagonal and subdiagonal components and the pivot permutation. Native
Bunch–Kaufman solves use operator scratch. Valid 2x2 blocks with zero individual
diagonal entries are preserved, unlike the unpivoted LDLT zero-pivot rejection.
Mixed scalar promotion converts both block components and preserves pivots;
normal/conjugate/transpose/adjoint, copies and composition reuse saved factors.

Focused verification passed 899 assertions across all factor/RHS scalar pairs,
both triangles, empty/singleton/2x2/larger shapes, zero-diagonal invertible blocks,
permuted blocks, source closure, copy, transpose/adjoint, indefinite BiCGSTAB and
singular/closed-factor rejection. Canonical Pkg.test passed 88,458 assertions /
73 files with 16 workers. All 10 native tests and 61 README blocks passed,
along with ambiguity detection, Julia/Rust formatting and git diff --check.
Sparse and spectral preconditioners, callbacks and remaining Issue #1/#2 scope
are pending.

### Saved native SVD/EVD inverse actions

FaerLinearOperator(F;atol,rtol) accepts saved native SVD/self-adjoint EVD factors
and independently owns spectral components and truncated inverse weights.
Application uses native projection/scaling with a scratch-backed intermediate,
without an explicit pseudoinverse or decomposition. Threshold decisions reuse
the saved-factor solve helper in original factor precision and remain fixed
through later scalar promotion. Eigenvalue signs are retained. Rectangular and
rank-truncated actions have reverse(size(F)); square/nonsingular/positive-definite
requirements for particular iterative preconditioners still apply.

Focused verification passed 1,062 assertions across all factor/RHS scalar pairs,
empty/rectangular/square SVD, truncation, negative self-adjoint spectra,
transpose/adjoint, independent close/copy, CG and invalid tolerance/owner checks.
Canonical Pkg.test passed 89,520 assertions / 74 files with 16 workers.
All 10 native tests and 62 README blocks passed, along with ambiguity detection,
Julia/Rust formatting and git diff --check. Sparse factor preconditioners,
callbacks and remaining Issue #1/#2 work remain pending.

### Native simplicial sparse LU preconditioners

FaerLinearOperator accepts explicit simplicial sparse LU factors, exporting
current native sparse L/U and both permutations into an independent operator.
Application gathers the RHS, performs sparse triangular solves and scatters the
solution using a scratch-backed RHS-sized intermediate. Normal/conjugate/
transpose/adjoint and mixed scalar promotion preserve sparse factor storage.
No source snapshot, dense coefficient matrix, inverse or refactorization is used.
Construction validates square components, diagonal positions, triangular entries
and nonzero U pivots at the native boundary.

Focused verification passed 867 assertions across scalar pairs, CSC/CSR,
empty/singleton/pivoted systems, forced column ordering, source snapshot zeroing
before operator construction, close/copy, all transforms and BiCGSTAB. A 20,000
order sparse diagonal case also passes a promoted complex-vector application.
Canonical Pkg.test passed 90,387 assertions / 75 files with 16 workers.
All 10 native tests and 63 README blocks passed, along with ambiguity detection,
Julia/Rust formatting and git diff --check. Auto/supernodal sparse LU and
other sparse factor preconditioners, callbacks and remaining Issue #1/#2 scope
remain pending.

### Native sparse Cholesky preconditioners

FaerLinearOperator accepts sparse Cholesky factors from auto/simplicial/
supernodal analysis, retaining independent sparse L/U and symmetric permutation.
The shared sparse-factor action distinguishes unit-lower LU from nonunit-lower
Cholesky solves. Native sparse triangular kernels and RHS-sized scratch avoid
dense coefficients, inverses and source-snapshot refactorization. Mixed promotion
preserves sparse factors, ordering and the lower-diagonal convention.

Focused verification passed 5,192 assertions across all scalar pairs, CSC/CSR,
three algorithms, both input triangles, reverse ordering, empty/singleton/square,
source snapshot zeroing, copy/close, conjugate/transpose/adjoint, CG, invalid
factors and a 20,000-order supernodal promoted complex RHS. Canonical Pkg.test passed 95,579 assertions / 76 files with 16 workers.
All 10 native tests and 64 README blocks passed, along with ambiguity detection,
Julia/Rust formatting and git diff --check. Auto/supernodal LU, sparse QR preconditioners,
callbacks and remaining Issue #1/#2 scope remain pending.

### Native simplicial sparse QR inverse actions

FaerLinearOperator accepts full-column-rank explicit simplicial sparse QR,
retaining sparse Householder components, R, tau and immutable symbolic analysis/
column permutation. Normal/conjugate actions use native saved QR solve; transpose/
adjoint use sparse transposed triangular solve and reverse reflector application
into the caller's result. RHS intermediates use native scratch. No dense Q,
pseudoinverse, source snapshot or refactorization is used. Scalar promotion
copies raw symbolic arrays without sorting Householder pivot order. The action
has reverse(size(F)), including tall least-squares/minimum-norm dual actions.
Missing/zero R pivots are rejected; no numerical rank tolerance is exposed.

Focused verification passed 1,287 assertions across scalar pairs, CSC/CSR,
empty/tall/square, custom column ordering, source snapshot zeroing, copy/close,
all transforms, CG, invalid factors and 20,000-order complex normal/adjoint
application. Canonical Pkg.test passed 96,866 assertions / 77 files with 16
workers. All 10 native tests and 65 README blocks passed, along with ambiguity
detection, Julia/Rust formatting and git diff --check. Auto/supernodal
sparse factor interfaces, callbacks and remaining Issue #1/#2 scope are pending.

### Mixed sparse solves reuse saved native factors

Mixed-type RHS for explicit simplicial LU/QR and all sparse Cholesky algorithms
now use the saved-factor native inverse action, including vector/matrix RHS,
transpose/adjoint and three-argument ldiv!. Converted factor components retain
the original factorization precision; the input is never refactorized. Temporary
operators and fallback promoted factors/source matrices are closed explicitly.
Auto/supernodal LU/QR still refactorize source snapshots for mixed RHS.

Focused tests passed 4,848 assertions, including closed source snapshots, all
16 scalar pairs, CSC/CSR, custom ordering, empty/square/tall matrices and
transposed minimum-norm QR against Julia's independent pseudoinverse. Existing
mixed Cholesky comparisons now use the original factor precision for tolerance,
rather than assuming a new decomposition in the RHS precision. The initial full
run exposed those stale precision assumptions; they were corrected before the
final full validation. No upstream defect was identified in this change.

Final canonical Pkg.test passed 101,714 assertions / 78 files with 16 workers.
All 10 native tests and 66 README blocks passed, as did ambiguity detection,
Julia/Rust formatting and git diff --check. The dependency remains the clean
d5be1473eed9314fca6f80ad8b4072d86282e890 checkout.

### Owned auto/supernodal sparse QR components and transpose solves

The build script fetched upstream 0bf88db6e70c29488df91a128f5e2562f8859730
(t4a-faer 0.28.2), including merged PR #228's SymbolicQr::raw accessor.
The dependency checkout is clean. Adapter-owned numeric buffers are split using
public symbolic lengths and borrowed for component extraction. Simplicial and
supernodal automatic selections both expose thin Q, sparse CSC R, permutations
and reconstruction from current factors, independently of the retained input.
Supernodal R exports only its mathematical triangular entries. Transpose/adjoint
minimum-norm solves extract sparse R and apply saved Householder blocks in reverse
order using their actual numeric dimensions and native Householder kernels.
No dense Q or pseudoinverse is formed by the transpose solve. Extraction and
transpose solves allocate native work arrays; mixed auto/supernodal QR RHS still
refactorize snapshots. Full Q, rank-deficient solves, inverse operators and
remaining Issue #1/#2 requirements are pending.

Focused verification passed 1,640 assertions: four scalar types, CSC/CSR,
auto/supernodal, empty/square/tall matrices, disconnected patterns and dense
130-by-80 block cases, complex transpose/adjoint, numeric updates, retained
components, closed source snapshots and invalid factors/owners.

The first latest-upstream full run exposed a duplicate Jacobi-right correction:
upstream #206 is fixed in this revision. The retained standalone Rust MWE passed
with norm(I*J - actual)=0; the obsolete adapter conjugation was removed.

Final Pkg.test passed 103,354 assertions / 79 files / 16 workers. Native Rust
11 tests, README 67 blocks, ambiguity detection, formatting and diff checks
passed on upstream 0bf88db.


## Issue #22: saved auto/supernodal sparse LU components

Native L/U and row/column permutation export is connected for all sparse LU
algorithms, enabling existing saved-factor reconstruction. Forced supernodal
selection and native COLAMD controls are connected. Upstream API: faer #235 /
PR #238; separate no-Rayon numeric fix: #236 / PR #237.

Validation against candidate faer 55d54a1: canonical
`Pkg.test(;allow_reresolve=false)` passed 103,851 assertions across 80 files
(16 workers). Focused lifecycle/component/COLAMD checks passed 497 assertions.
README passed 68 executable blocks; Rust passed 11 tests. Julia/Rust formatting
and diff checks passed.

Upstream PRs #237, #238 and #240 are merged. `deps/build.jl` fetched clean
faer main `12843515e07f8d6f27fc525668227088658ffe0f` (t4a-faer 0.28.2).
This includes the no-Rayon numeric solve fix, saved LU public accessors and
explicit sparse column ordering. Native tests pass 11 cases on this revision.
Canonical Pkg.test on merged main also passes 103,851 assertions / 80 files
with 16 workers. All 68 README blocks pass on the merged dependency.

## Issue #22: LU mixed RHS saved-factor reuse (merged)

All sparse LU algorithms now use the native saved-factor inverse action for
mixed RHS. Source snapshot closure is covered. Focused tests pass 6,768
assertions for mixed reuse and 1,731 for auto/supernodal inverse/preconditioner
actions. Canonical validation passes 107,502 assertions across 81 files (16 workers). Auto/supernodal QR remains pending.

LU mixed RHS follow-up README validation: 69 executable blocks passed.

## Issue #22: auto/supernodal QR mixed RHS and inverse actions (merged)

Saved numeric buffers are converted directly for scalar promotion; all sparse
algorithms now reuse native factors for mixed RHS. The source snapshot
refactorization branch is removed. Auto/supernodal QR inverse actions retain
native Householder/R/permutation buffers and allocate intermediates per apply.
Focused integration checks pass 12,535 assertions. Canonical validation passes
111,982 assertions across 82 files (16 workers). Rust tests pass 11 tests.
README passes 70 executable blocks.

## Issue #22: native explicit LU/QR ordering (merged)

Identity and supplied column permutations now reach native symbolic LU/QR
analysis for auto/supernodal algorithms. Upstream faer #239 / PR #240 provides
the API. Candidate faer 3889bd1 passes seven focused tests and three doctests.
Colmat focused integration passes 2,016 assertions. Canonical validation passes
113,998 assertions / 83 files / 16 workers; README passes 71 executable blocks.
Other native tuning remains pending. Merged upstream adoption is recorded below.

## Issue #22: full Q from saved sparse QR factors (merged)

All algorithms expose `Qfull` and the explicit thin alias `Qthin`, while `Q`
retains its existing thin contract. Missing pivots receive unused coordinate
basis vectors before reflector application, preserving orthogonality for zero
and rank-deficient inputs. Focused tests pass 1,224 assertions. Canonical validation passes 115,222 assertions / 84 files / 16 workers.
Rust passes 11 tests; README passes 72 executable blocks. Rank-deficient/wide solves and rank tolerance remain
separate unfinished contracts.

## Issue #22: sparse QR rank and explicit tolerance solves (merged)

Saved R supports native singular values, SVD-based rank and pseudoinverse;
`ldiv!(X,F,B;atol,rtol)` supplies minimum-norm least squares for rank-deficient
inputs. This path materializes dense R/thin Q and allocates native SVD workspace.
Default backslash still requires full column rank; originally wide QR is pending.
Focused tests pass 3,960 assertions. Canonical validation passes 119,182
assertions across 85 files (16 workers); README passes 73 executable blocks.
Native Rust code is unchanged from the validated full-Q branch (11 tests).

## Issue #22: originally wide sparse QR (merged)

Native prefix QR plus per-column Householder projections stores Q/R/full column
ordering. Wide normal/transpose/adjoint solves and inverse actions use native
SVD of saved R for minimum norm, with the same default cutoff as rank/pinv.
This path allocates dense R/SVD workspace. Caller-owned wide scratch and tall/
square default rank-deficient backslash remain pending.
Focused wide integration passes 7,488 assertions; closed-snapshot copy/deepcopy
regression passes 180 assertions for LU/QR/Cholesky. Native tests pass 20 tests.
Canonical validation passes 126,858 assertions / 87 files / 16 workers;
README passes 74 executable blocks.

## Issue #22: default rank-aware sparse QR (merged)

Tall/square sparse QR backslash and inverse operators now use saved-factor
minimum-norm solves for rank-deficient inputs. A conservative sparse R norm
certificate preserves the existing sparse solve for certified full-rank cases;
inconclusive cases use native dense SVD of R and thin Q. The default threshold
uses the factor's scalar precision. This path allocates dense intermediates.
Numerically rank-deficient caller-owned scratch solves reject before output
mutation; their SVD scratch integration remains open.

Canonical Julia tests pass 128,781 assertions across 88 files (16 workers).
Focused coverage includes all four scalar types, mixed RHS, CSC/CSR, all QR
algorithms, zero/structural/numerical rank deficiency, transpose/adjoint,
closed source snapshots, and independent inverse operator ownership.
Native Rust tests pass 22 cases, including large Float32 diagonal certificates,
cutoff boundaries and an inconclusive nonsingular triangular factor. All 75
README Julia blocks pass. Julia/Rust formatting and diff checks pass.

## Issue #22: main integration and dependency adoption

PRs #23–#30 are integrated into main (QR mixed-factor PR #25 entered main
through #24). The integrated code/test tree at `78576d4` matches the validated
`a9b60b8`; only implementation-status documentation differs. Canonical tests
pass 128,781 assertions / 88 files / 16 workers, Rust passes 22 tests, and all
75 executable README blocks pass. Frozen issue-1 inventory is unchanged.

The clean dependency is t4a-faer 0.28.2, main
`12843515e07f8d6f27fc525668227088658ffe0f`, including merged upstream PRs
#237 (no-Rayon solve), #238 (saved LU components), and #240 (column ordering).
Its tree matches the ordering candidate used for final QR integration tests.
LU and combined LU/QR canonical suites also passed on merged upstream main
(103,851 and 111,982 assertions respectively).

Native LU components/reconstruction, all-algorithm LU/QR mixed RHS and inverse
actions, explicit column ordering, full Q, rank/pseudoinverse/tolerances, wide
QR, and default rank-deficient backslash are integrated. Dense R/SVD fallback
allocations and incomplete rank-deficient/wide caller-owned scratch remain
explicit limitations. Further native tuning, matrix-free callbacks/history,
generalized Schur ordering, remaining workspace/in-place/team/view/scalar
coverage and the full 76-row review remain open under Issue #22.

## Issue #22: iterative observers, histories and stopping (in progress)

CG/BiCGSTAB/LSMR and their in-place forms now accept `callback` and `history`.
Callbacks receive independent Julia iterate snapshots and independently
recomputed residuals. Returning false requests a real native stop, reported as
`:stopped` without treating it as a checked numerical failure. Histories retain
residual diagnostics without full iterate copies. Callback exceptions safely
return through RustCall and preserve in-place destinations. Default unobserved
solves make no snapshot copies or extra operator applications.

Upstream main skipped BiCGSTAB's terminal half-step callback (#242). A native
MWE failed on `1284351` and passed on public candidate `778afac`, which also
extends the three canonical callbacks with unit/ControlFlow continuation and
explicit stop info. PR tensor4all/faer#243 passed required CI and merged as `51d7584`.
The dependency build now selects clean faer main `46154ca9ad1df24eeb2abeee76a7537415d4b9a0`
(t4a-faer 0.28.3). Its faer/src and faer-traits trees match the validated candidate;
additional upstream changes update release metadata and PATCHSET records.

Final canonical validation passes 134,984 assertions / 89 files / 16 workers.
Focused observer coverage passes 6,203 assertions across four scalar types,
mixed RHS, native/Julia vectors and block matrices, dense/CSC/CSR, independent
callback snapshots, nested callbacks, Julia Tasks, native destinations/views,
RHS/output aliasing, checked failures and nonfinite failed iterates. Rust passes
22 tests; README passes 76 executable blocks. Formatting and diff checks pass.

An additional standalone Rust MWE confirms LSMR false convergence on tiny/huge
scaled identity systems (#245); Colmat rejects it through true residual checks,
and observation preserves its failed iterate. The upstream numerical correction
remains open. The now-fixed #207 absolute-tolerance workaround still needs
cleanup. Julia-defined operator/preconditioner callbacks, partial-spectrum
callbacks, caller-owned iterative scratch and teams remain separate work.

Merged-dependency adoption is verified: `deps/build.jl` built clean main
`46154ca` / t4a-faer 0.28.3 and updated the Cargo lockfile. The canonical suite
again passes all 134,984 assertions / 89 files / 16 workers on this revision;
Rust 22 tests and README 76 blocks also pass on merged upstream main.

## Issue #22: direct native LSMR tolerances (in progress)

Removed the old #207 conversion of absolute tolerance into a relative bound.
LSMR receives atol and rtol independently; Colmat retains independent true
normal-residual verification. The original standalone Rust atol MWE succeeds on
selected upstream main `46154ca`.

That validation exposed wrong zero initial-convergence diagnostic fields (#248).
The standalone MWE fails on main and passes on public candidate `106d870` from
PR #249. Its four focused Rust tests cover real/complex single/double precision,
zero/nonzero initial guesses, strided block RHS and right preconditioning.
PR #249 passed required CI and merged as `fc52fc9`; integration now selects
clean main `fc52fc9b5b025a9d4b75110415bb026584885e4e` (t4a-faer 0.29.0).
Relative to the tested candidate, changes are release/ledger/workflow metadata
and the BiCGSTAB breakdown doc clarification; the LSMR computation is unchanged.

Focused Julia LSMR validation passes 5,691 assertions. It covers independent
absolute/relative and combined bounds, native diagnostics, mixed RHS, dense and
CSC/CSR, native matrix RHS, right preconditioning, zero references, retained
initial guesses, callback/history zero-update behavior and checked in-place
failures. Coordinate RHS provide exact cutoff equality fixtures; dense block
RHS use bounds with a clear margin because QR norm estimates can round by 1 ulp.

Canonical candidate validation passes 140,236 assertions / 89 files / 16 workers.
All 77 executable README blocks and 22 Rust tests pass. Julia/Rust formatting
and diff checks pass. Merged upstream 0.29.0 adoption and final checks are recorded below.

Merged faer main `fc52fc9` / t4a-faer 0.29.0 passes the canonical suite again:
140,236 assertions / 89 files / 16 workers, README 77 blocks and Rust 22 tests.
The original diagnostic MWE also passes on this merged revision. The Cargo
lockfile now records 0.29.0. Upstream #248 is fixed and the #207 workaround is
removed; the extreme-scale issue #245 remains open.


## Issue #22: saved generalized Schur reordering

`ordschur` and `ordschur!` operate on the saved native S/T/Q/Z factors through
faer #250 / PR #251. Both keep stable order within selected/unselected groups.
Selecting either member of a real conjugate block selects the whole pair,
matching Julia's generalized Schur convention. The four native scalar types,
empty/singleton factors, regular infinity and unchanged 0/0 blocks are covered.
Crossing a nonregular block or another numerically unsafe move throws.

All four work factors are independent native copies; no original input snapshot
or QZ refactorization is used. Homogeneous alpha/beta are refreshed from current
S/T, ignoring stale or modified metadata. The new faer real-block API performs
joint equilibration and verifies its characteristic polynomial. Complex values
copy the reordered diagonals. The native boundary checks current factor
shapes before forming raw selection slices and validates nonnull/aligned spans
and disjoint homogeneous output buffers. Explicit requested scalar tags are
checked against the current erased native variants; both QZ construction and
reordering reject stale Julia dtype/shape metadata before output writes.

`ordschur!` stages every native result and metadata conversion before writeback.
Owners and existing views remain live; failures preserve all four factors and
both metadata vectors. Current native owner addresses detect aliasing even
through distinct Julia wrappers; the same guard also protects existing matrix-
pair and generalized-Hessenberg mutation. Homogeneous vectors can be disjoint
contiguous, interleaved or reversed views into one parent; overlapping element
byte intervals reject. Empty shared metadata is valid. Closed owners, malformed
factor structure, stale cached dimensions and invalid lengths reject before
input writeback.

Focused validation passes 2,409 assertions, including 4-type random QZ results,
conjugate masks, infinity, stale metadata, saved-owner lifetime, Julia Tasks,
owner/view preservation, overlapping/disjoint metadata, conversion failures,
nonregular rejection and current-owner guards. Native tests pass 25 cases,
including raw buffer length/alignment/span/overlap validation. Canonical validation passes 142,645 assertions / 90 files / 16 workers;
all 78 README blocks, 25 native tests and Julia/Rust formatting/diff checks pass.

Caller-owned reorder scratch, nondefault QZ tuning and remaining workspace/team
connections are still incomplete. The frozen inventory and full 76-row scope
are retained.

Merged upstream adoption selects clean faer main
`19053d8c22993851c032d5373b4ec188998f2b5c` (0.29.0). PR #251 passed all hosted
checks and merged with a tree identical to the tested candidate. The build
fetched and recorded this merged revision. The real-QZ construction scratch
workaround remains: its underlying query defect is now independently reproduced
on this latest main and reported as [faer #252](https://github.com/tensor4all/faer/issues/252).
The 96x96 public-query MWE and proposed corrections are retained in
[the reproducer](reproducers/faer-qz-scratch.md); no upstream patch is claimed.


## Issue #22: additional upstream QZ numerical findings

Three further Rust-only reproductions are retained in
[the QZ numerical record](reproducers/faer-qz-numerics.md). They execute against
clean faer 19053d8, the currently selected Colmat dependency:

- [faer #254](https://github.com/tensor4all/faer/issues/254): a completely
  deflated window starting at row zero uses an unsigned sentinel. Computing
  its count panics with overflow checks; an ample-workspace n=78 MWE isolates
  this from #252. PR #255 proposes the explicit count branch.
- [faer #256](https://github.com/tensor4all/faer/issues/256): the generalized-
  Hessenberg Givens helper produces nonfinite factors from finite subnormal
  entries, for all four native scalar types. The 3x3 pencil has identity B
  and no unrepresentable norm. Colmat's finite-output check rejects the result;
  the proposed stable-rotation correction is in draft faer PR #258.
- [faer #257](https://github.com/tensor4all/faer/issues/257): a valid maximum
  AED window produces a DGET54-style coupled residual ratio 436.6 and Q/Z
  orthogonality errors near 1e-11 on a deterministic n=78 fixture. LAPACK on
  identical data gives ratio 0.7724. Release mode isolates it from #254;
  extra valid workspace isolates it from #252. A range-safety candidate improves
  the ratio to 178.6 but still fails the diagnostic threshold 20.

PR #253 addresses #252's even-rounded window. Its code and PR #255 match the
independently checked local corrections; required hosted checks must complete
before adopting their merged revisions. Draft PR #258 stacks on #255, covers
#256 plus the four-array local Sylvester workspace audit, and adds broader
property tests. It does not resolve #257. The wrapper correction is retained
and the selected dependency is unchanged by this report-only checkpoint.

## Issue #22: caller-owned generalized Schur reorder workspace

Saved-factor `ordschur` / `ordschur!` accept `scratch=FaerScratch(...)`.
`scratch_requirements` and `reserve!` cover selection flags, native block moves
and real homogeneous-value extraction. Native flags and temporary real alpha/
beta columns now live in the supplied buffer. Four independent work factors and
Julia output/conversion vectors still allocate, preserving atomic writeback and
existing owners/views. Default calls create and close a temporary workspace.
Insufficient bytes/alignment, closed or invalid scratch reject before input
writeback. Shared workspace access uses the existing matrix lock.

Validation on clean faer 19053d8 (0.29.0): canonical `Pkg.test` passes 143,422
assertions / 91 files / 16 workers, native tests pass 25, and all 78 executable
README blocks pass. The new workspace file passes 777 assertions, including a
final focused run with precision-specific residual/orthogonality bounds. Its
coverage includes all four scalar types, empty/singleton factors, real pairs,
exact queried capacity, reserve, repeated/Task reuse, preserved owners/views,
capacity/alignment/close errors and reuse after a numerical rejection. Existing
QZ/reorder tests also pass. Formatting and diff checks pass; frozen inventory
is unchanged. The final candidate initializes flags directly from the selection, avoiding a
redundant zero-fill before copying.

QZ construction scratch, nondefault tuning and team integration remain open.
Upstream PR #253 merged after this candidate's dependency was selected. The
wrapper's existing QZ construction bound is retained until the remaining
upstream fixes and merged-dependency adoption checks complete.

## Issue #22: independently reproduced AED index scratch shortfall

[faer #260](https://github.com/tensor4all/faer/issues/260) now has an executed
18x18 public-query reproduction, separately from #252's merged rounded-window
fix. On PR #253 plus the exact #255 sentinel correction, 2,112 queried bytes
panic in full-pivot LU. Changing only AED's one permutation-array reservation
to four yields 2,304 bytes and completes the QZ call. Its retained numerical
assertion still fails at ratio 61.65; #257 remains open. Source, baseline and
run instructions are retained in [the numerical record](reproducers/faer-qz-numerics.md).
The wider public tests also expose this capacity failure for f32. Draft #258
contains the correction. No dependency checkout or selected revision is changed
by this report-only update.

## Issue #22: dense native factor construction with caller-owned teams

Dense partial/full-pivot LU, QR with/without column pivoting, unpivoted Cholesky,
LDLT and Bunch–Kaufman construction accept `team=FaerTeam(...)`, including their
staged bang forms and the owned-matrix `factorize` dispatcher. The native owner
borrows the team only while constructing the decomposition and forwards its
`Par` to the actual faer constructors and their scratch queries. Width one uses
sequential execution. Returned factors store owned numerical data and remain
usable after both source and team close; copies are independent.

The same guarded initialization helper serves default and team calls. Requested
scalar/dimensions are compared with the current native owner before construction,
preventing stale Julia metadata from producing a wrongly typed/sized factor.
Team arguments use RustCall's checked references and the existing matrix lock;
no new raw-owner exception or per-element dispatch is introduced. The existing
selected-triangle, unsuccessful `check=false` and atomic staged-writeback
contracts are retained. Construction still allocates native factors/workspace.

Focused tests cover all four scalars, team widths one/two, empty/singleton,
rectangular QR, 129x129 blocked inputs, selected upper/lower triangles with NaNs
in unused storage, 2x2 indefinite blocks, singular/nonpositive failures, existing
views, Tasks, independent copies, close/lifetime and corrupted native metadata.
The new file passes 3,470 assertions with precision/dimension-aware bounds;
existing factorization and destructive suites also pass. Native checks pass 27
tests, including closed-team rejection and native type/shape guards. Canonical
`Pkg.test(;allow_reresolve=false)` passes 146,892 assertions / 92 files / 16
workers; all 79 executable README blocks and formatting/diff checks pass.

This connects the construction phase, not retained-factor solve/property calls.
Pivoted Cholesky, sparse/iterative teams, caller-owned construction scratch and
other kernels remain tracked. In particular, faer's current public QZ reordering
accepts an ignored `_par`; no QZ parallelism or speed improvement is claimed.
The clean selected dependency remains faer 19053d8 (0.29.0). The complete 76-row
scope and frozen inventory are preserved.

## Issue #22: caller resources for native operator application

Native operator products support explicit `*(op,B; scratch,team)` and the
right-operator form. `mul!` supports both orientations, vectors/matrices and
alpha/beta with the same resource keywords. `scratch_requirements`,
`FaerScratch` and `reserve!` resolve the actual working scalar, orientation,
RHS count and team; mul! includes destination/alpha/beta promotion. Composition
and saved-factor inverse actions reuse the LinOp-declared scratch and borrow a
caller-owned team without storing its token. Default multiplication preserves
its existing sequential allocation route.

The new native entry points validate current operator/RHS scalar and dimensions
against requested Julia metadata before application. Default products and
operator widening now check the original native metadata too, before a
conversion can mask a stale type. All output matrices stage errors; resource
capacity/alignment/close and conversion failures preserve destinations.
Zero-alpha calls validate supplied resources but do not read RHS elements or
old output when beta is zero. Native query output spans are checked and all
owners use RustCall's checked references. Shared reuse is serialized by the
existing owner lock.

The new focused file passes 4,655 assertions, covering four scalars, team widths
one/two, dense/CSC/CSR/diagonal/identity, all four orientations, composition,
saved dense/sparse inverse and spectral actions, rectangular spectral actions,
scalar promotion, matrix/vector/native vector/view output, overlap, repeat/Task
reuse, reserve, capacity/alignment/close failures, atomic conversion failure,
and corrupted native dtype/shape in default, explicit and promoted paths.
Existing operator/composition/QR-action suites pass; native checks pass 29
tests. Final canonical `Pkg.test(;allow_reresolve=false)` passes 151,547
assertions / 93 files / 16 workers; all 80 executable README blocks and
formatting/diff checks pass.

This is LinOp-query workspace: independent result matrices, Julia arithmetic/
conversion buffers and promoted operator snapshots still allocate. Existing
auto/supernodal and wide sparse-QR inverse actions still allocate application
temporaries and run sequentially outside caller scratch. Those native wrapper
fallbacks remain explicit incomplete work, as do direct retained-factor solve
keywords, dense rectangular QR inverse actions and arbitrary Julia callbacks.
No universal allocation-free or parallel-speed claim is made. The selected
clean faer dependency is unchanged at 19053d8 (0.29.0); full 76-row scope and
frozen inventory remain intact.

## Prepared owned/wide sparse QR inverse workspace

Owned auto/supernodal inverse actions now prepare the saved sparse R, column
permutation and Householder projection layout once. Wide inverse actions cache
the dense pseudoinverse of saved R once. Applications, conjugates, transposes
and adjoints use declared caller scratch and propagate the caller team through
native Householder, triangular-solve and matrix-product kernels. No dense Q or
source refactorization is introduced. Scalar promotion converts the cached
wide inverse and preserves its original rank cutoff.

Eight native regression tests cover all four scalars, forced symbolic algorithms,
permutations and an entirely empty wide prefix. After warmup, repeated Seq and
two-thread Team applications perform zero Rust heap allocations on the calling
thread. This counter does not measure worker-thread allocations. Julia regression
coverage includes all orientations, CSC/CSR, zero shapes, large blocked factors,
exact queries, shortage/reserve, factor lifetime, copies and shared Task reuse.

Construction, direct factor solves, result staging and Julia promotion buffers
still allocate separately. This closes the specific owned/wide action fallback
recorded in the preceding checkpoint; remaining factor solve/kernel resources
and arbitrary Julia callbacks stay open under Issue #22. The faer dependency
remains 19053d8 (0.29.0).

Final canonical validation passes 158,626 assertions / 94 files / 16 workers
(Julia 1.13.3 / RustCall 0.9.2), native 37 tests and 80 executable README blocks.
The new Julia file contributes 7,079 assertions. Formatting/diff checks pass;
the frozen inventory is unchanged.

## Retained dense factor solve teams

`ldiv!(X,F,B;team)` and `rdiv!(X,B,F;team)` now borrow a caller-owned native
team for saved dense LU/full LU/QR/pivoted QR/Cholesky/LDLT/Bunch–Kaufman.
Both two-argument square forms and three-argument rectangular forms support
transpose/adjoint, all four native scalars and existing mixed-scalar promotion.
The native regular, least-squares, reconstruction, SVD/pseudoinverse and final
product paths all receive the borrowed Par. Tokens never escape execution.
Construction and solve teams have independent lifetimes.

A shared Julia staging path replaces the scattered mutating factor solve
methods, snapshots RHS before output writeback and stages destination conversion.
New checked native team entrypoints compare current dtype, shape, factor kind
and RHS column count against requested metadata before assigning a result.
Native singular status is checked independently of Julia's cached status.
Native outputs and Julia destinations survive failed validation or conversion.

Mixed-scalar promotion retains the existing reconstruction/refactorization
policy; component extraction remains sequential, while promoted products and
new factorization use the team. Ordinary backslash remains sequential; caller
scratch, direct native in-place storage, component-extraction teams, sparse
factor solve teams and the remaining 76-row/callback scope stay open. Native
workspace and result staging still allocate. The selected faer remains the
clean verified 19053d8 (0.29.0).

Final canonical validation passes 170,823 assertions / 95 files / 16 workers
(Julia 1.13.3 / RustCall 0.9.2), native 39 tests and 81 executable README blocks.
The new dense-solve file contributes 12,197 assertions. Closed/invalid teams,
unsuccessful factors, nonfinite RHS, conversion failure, current native dtype/
shape/kind, copies, shared Tasks, zero shapes and blocked 129-order factors
are covered. Formatting/diff checks pass; the frozen inventory is unchanged.

## Julia-defined native operator and preconditioner callbacks

A new FaerLinearOperator constructor accepts forward/adjoint in-place Julia
functions and matrix dimensions. Native operators retain integer identities,
not Julia pointers. Julia owners retain function state across copies, transforms,
promotion and compositions. A synchronous native scope installs a process-lived
static dispatcher only on the original calling thread and restores prior state
on every exit; native workers cannot invoke Julia through a foreign-thread scope.
The calling task is temporarily pinned while native frames are live, preserving
RustCall's same-thread callback borrow even if user code yields. Its prior
scheduling state is restored on success and failure. Same-task nested calls
remain reentrant; other tasks serialize on the existing owner lock.
RustCall's scoped body callback keeps the native call on the Julia task. The
separate static action dispatcher also composes with the solver's existing
RustCall iteration callback slots without replacing them.

The action dispatcher copies native input into owned Julia matrices and provides
an owned output initialized with NaN, then checks complete finite overwrite
before copying back. Escaped arrays remain owned and valid. User exceptions never
unwind through C/Rust; the scope returns before the original Julia exception is
rethrown. Failed actions cannot commit staged products or in-place solver output.
Native conjugate/transpose applications use declared scratch for conjugating
intermediates. Callback marshaling, Julia function work and result staging still
allocate outside native scratch. Captured state is shared by reference even in
deepcopy; native operator ownership remains independent.

CG/BiCGSTAB/LSMR and partial spectra use their native algorithms with these
LinOp/BiLinOp/Precond/BiPrecond actions. Existing explicit small/full-projection
partial-spectrum dense fallbacks retain their reported algorithm labels.
Iteration histories/stopping callbacks can run alongside action callbacks.
Remaining iterative workspace/team, direct factor scratch, dense rectangular QR
inverse actions and other incomplete 76-row features remain open. No new faer or
RustCall defect is implied by the adapter; the selected dependency is unchanged
at clean 19053d8 (0.29.0).

Final canonical validation passes 172,304 assertions / 96 files / 16 workers
(Julia 1.13.3 / RustCall 0.9.2), native 41 tests and 82 executable README blocks.
The callback file contributes 1,481 assertions. A separate four-thread probe
passes eight assertions for migratable tasks, yields and scheduling restoration
on success/failure. Original thrown values, including nothing, survive both
action and observer boundaries. Formatting/diff checks pass and the frozen
inventory is unchanged.

## Rectangular and truncated dense QR inverse actions

Saved QR/column-pivoted QR now supplies a minimum-norm action for rectangular
inputs, including rank deficiency. Nonzero atol or explicit rtol selects the
same truncated action for square QR; the square default retains its existing
nonsingular triangular-solve contract. This preserves existing preconditioner
behavior while making rank truncation explicit for square factors.

Construction computes the SVD of saved R once, then retains its singular
vectors/weights, compact Householder basis/coefficients and column permutation.
The original matrix is not reconstructed or refactorized; no dense Q or full
pseudoinverse of A is formed. Normal/conjugate/transpose/adjoint application uses
native scratch for projections/permutations and propagates the caller team into
Householder and spectral products. Promotion converts saved values and preserves
rank-cutoff decisions. Construction, promoted snapshots and result staging
allocate separately; construction uses internal sequential SVD workspace.

faer stores infinity as an inverse-Householder coefficient for an identity
reflector; validation accepts this representation and rejects NaNs. A reproduced
all-truncated action with finite floatmax RHS returned NaN from an overflowing
projection. Cached zero actions now return zeros before projection, and discarded
spectral rows are set to zero without multiplying infinity by zero. This fixes
the owning shared spectral-action helper too, with a direct partial-truncation
regression and four-scalar QR/SVD zero-action tests. No upstream defect is claimed.

Remaining direct-factor/iteration scratch, other kernel team/tuning and the
full 76-row scope remain open. The selected dependency stays clean 19053d8
(0.29.0); faer #258 is tracked separately.

Final canonical validation passes 179,505 assertions / 97 files / 16 workers
(Julia 1.13.3 / RustCall 0.9.2), native 47 tests and 83 executable README blocks.
The new file contributes 7,201 assertions, including empty-prefix wide QR and
floatmax RHS with zero truncation. Native tests measure zero calling-thread Rust
heap allocations during warmed Seq and two-thread applications (worker threads
are not measured). Formatting/diff checks pass; frozen inventory is unchanged.

## Householder application scratch and teams (PR #45)

Baseline: main `814fd1d`, faer `19053d8` (0.29.0), canonical
179,505 assertions / 97 files, Rust 47 tests, README 83 blocks.

Elementary reflection and compact-sequence application now take caller-owned
scratch and a scoped native team for left/right application, mul!, and inverse
actions. Native query/compute use the same existing Householder kernels and
scalar-tail handling; elementary 1x1 factors live on the Rust stack. The kernel
receives the actual `Par`, not a retained token. Julia result staging, view
writeback, coefficient promotion and scalar conversion allocate separately.
Native owner scalar/shape/block metadata is validated before promoted clones;
capacity/alignment and resource lifetimes are checked before destination writes.
Construction/reblock scratch/team and direct retained dense solve scratch remain
separate pending items within the complete 76-row objective.

RustCall 0.9.2 glob-imported borrowed owner handling was independently reproduced
without faer and reported as [RustCall #569](https://github.com/AtelierArith/RustCall.jl/issues/569),
with complete executable Rust/Julia sources in
`docs/reproducers/rustcall-glob-owner/`. Standalone Rust passes; Julia rejects
`&Counter` as by-value, while the explicit-import control succeeds. Colmat uses
explicit imports and checked owner handles, with no raw pointer bypass.
Latest upstream resolver source was inspected (byte-identical to installed),
not executed; the report distinguishes the conservative limitation from unsafe
execution and proposes scoped unambiguous resolution plus accurate diagnostics.

Final validation: canonical `Pkg.test(;allow_reresolve=false)` passed
184,477 assertions / 98 files / 16 workers (5m29.0s); new focused coverage
4,972 assertions, native 55 tests, executable README 84 blocks. Format/diff
checks passed and frozen inventory is unchanged. The dependency remains clean
at `19053d8`. No full-suite result is inferred from focused tests.

## Retained dense solve scratch (PR #46)

Baseline main `bd9d3d4`, faer clean `19053d8` (0.29.0), canonical
184,477 assertions / 98 files, Rust 55 tests, README 84 blocks.

Connect caller-owned solve scratch and exact capacity/alignment queries/reserve
to dense LU/full LU/QR/pivoted QR/Cholesky/LDLT/Bunch–Kaufman left/right solves,
all native scalars and orientations, scalar promotion and vector/matrix/view
destinations. Native queries/compute use the actual scoped team. Results stage
before destination writeback. Rectangular/rank-deficient QR uses saved R SVD
and compact Householder inside the buffer; no original matrix or dense Q/pinv
is formed. R SVD is recomputed per fallback solve, unlike the cached LinOp.
Retained legacy SVD also consumes saved U/S/V in this native workspace path.

New mixed-rank tests exposed a pre-existing wrapper issue: Float32 factor
reconstruction noise could be retained as rank after Float64 promotion, changing
the minimum-norm answer. Keep source-precision cutoffs through reconstruction
and refactorization in both explicit-workspace and default promoted QR solves.
This is a Colmat policy defect, not a RustCall/faer report. New tests retain an
independent source-matrix pseudoinverse reference rather than accepting the
perturbed promoted matrix's spurious rank. Mixed-scalar factor construction and
Julia/native result staging allocate outside solve scratch; no allocation-free
whole-API claim is made. Sparse solves and native direct in-place remain open.

Executed baseline main `bd9d3d4` confirms Float32/ComplexF32 diagonal
`[1,1e-8]` QR promoted to ComplexF64 returns approximately `[1,1e8]`
(4 before-regression checks). Current default and explicit-scratch solves return
`[1,0]` with the source cutoff (20 after-regression checks). No upstream
dependency checkout was modified.

A final shape audit found the existing wide/transformed LU reconstruction/SVD
paths also require scratch coverage. Connect low-level saved-L/U/permutation
reconstruction and SVD to the same buffer (query includes both native queries
and their matrix/projection temporaries). Extend rectangular LU/full LU normal
left solves to minimum-norm solutions as well; transformed/right/empty/native
and mixed-scalar cases are covered. Nonzero-info LU keeps its existing singular
error contract. The original input snapshot is never consulted. This final
extension requires a fresh full-suite run after the earlier QR-only checkpoint;
do not reuse that checkpoint as final validation.

Array owner metadata is validated before Julia-side raw-buffer conversions,
including native matrices, vectors/rows/diagonals, views and transforms. Source
and destination scalar/shape/layout mismatches are rejected before reads or
writeback; the negative tests construct mismatched wrappers but never call an
unchecked buffer conversion. This check also protects default dense-factor RHS
promotion. It is scoped to dense solves, not a package-wide metadata audit.

Final exact-tree validation: canonical `Pkg.test(;allow_reresolve=false)` passed
222,089 assertions / 99 files / 16 workers (7m09.5s), including 37,612 assertions
in the new dense-solve workspace coverage; Rust 61 tests and all 85 executable
README blocks passed. Native boundary/array-owner guards, source precision,
rectangular LU/QR, blocked/empty shapes, all scalar/orientation/mixed cases,
views/vector/row/diagonal layouts, aliases, Tasks and conversion/resource failure
staging are covered. JuliaFormatter/rustfmt/diff checks pass; frozen inventory
and faer checkout remain unchanged. Public FaerSpectralFactor solve scratch
remains separate from this FaerFactorization API and legacy SVD backend.

## Public retained spectral solve scratch (PR #47)

Baseline main `be6b3c5`, faer clean `19053d8` (0.29.0), canonical
222,089 assertions / 99 files, Rust 61 tests, README 85 blocks.

Connect caller scratch for public saved SVD/self-adjoint EVD left/right solves
and inverses, preserving all tolerances, source precision and scalar promotion.
Native storage includes reciprocal weights, projection and promoted U/V.
Direct right projection removes the owned adjoint RHS/result temporaries.
No decomposition, source matrix or intermediate pseudoinverse is required.
The inverse output itself is naturally a full matrix. Default and explicit
workspace routes share native kernels, including legacy internal FFI methods.
Current factor/array owners are validated before conversion and workspace use;
closed/insufficient resources fail before staged destination writeback. Native
result ownership and Julia/RHS conversion still allocate outside kernel scratch;
native-owner construction resources remain separate.

Exact threshold tests straddle the actual stored Float32 singular/eigenvalue.
The baseline EVD of diag(1,0.1f0) stores 0.100000024, within Float32 rounding;
comparing to a new Julia decomposition of the input literal would incorrectly
require clipping that retained value. Baseline stored-factor pinv/solve both
retain it. This is a fixture correction, not an upstream numerical defect.
General matrix-reference tests remain independent and unchanged.

Final exact-tree validation: canonical `Pkg.test(;allow_reresolve=false)` passed
249,959 assertions / 100 files / 16 workers (7m26.9s), including 27,870 new
assertions. Rust 67 tests and all 86 executable README blocks passed. Focused
solve coverage 27,824 plus 22 owner/conversion and 24 stored-threshold assertions
passed. Preallocated projection probes measured zero calling-thread Rust heap
allocations after warmup for four scalars and both sides with Seq/Team(2);
result ownership, coefficient promotion setup and worker heaps are outside that
measurement. No whole-API allocation-free claim is made. Format/diff checks pass;
frozen inventory and clean faer `19053d8` remain unchanged.


## Retained sparse LU/Cholesky solve scratch and teams

Public backslash, right division, ldiv!/rdiv! (two/three arguments), scratch
queries, FaerScratch construction and reserve now cover all sparse LU/Cholesky
algorithms. Same-scalar solves borrow native numeric factors and invoke their
matching scratch queries and kernels with the live team. LLT transpose/conjugate
uses the Hermitian factor convention. Mixed scalars retain the existing
saved-component inverse-action conversion policy without snapshot refactorization.
Its preparation allocates outside solve scratch, as do output ownership, RHS
conversion and staged writeback. No whole-API allocation-free claim is made.

Testing closed source and symbolic owners exposed a Colmat bug: Cholesky p used
the external analysis plan even though numeric LLT retained its own symbolic Arc.
The getter now copies that numeric-owned permutation, keeping mixed-scalar
solves independent of the closed plan. This is a wrapper correction, not a new
RustCall/faer defect.

Sparse QR workspace restrictions, sparse construction resources, iterative
resources and the rest of the complete 76-row inventory remain pending.

Validation at Julia 1.13.3 / RustCall.jl 0.9.2 and unchanged t4a-faer 0.29.0
revision `19053d8c22993851c032d5373b4ec188998f2b5c`: canonical Pkg.test
296,398 assertions / 101 files / 16 workers passed in 7m46.3s; the new
workspace file contributes 46,439 assertions. Native cargo test passed 71 tests;
all 87 README blocks passed. JuliaFormatter DefaultStyle, rustfmt for the new
native module, diff and frozen-inventory checks passed. Earlier focused runs
found the closed-plan permutation bug and method ambiguities; the final
canonical run includes their fixes and all validation tests.


## Sparse QR R-SVD/projection solve resources

All sparse QR algorithms now connect caller scratch/team to normal/transposed/
adjoint left/right solves, including vector/matrix/native/view RHS, promotion,
empty/rectangular/wide/rank-deficient factors and explicit atol/rtol. Default
rank-deficient/tolerance and wide solves share the new kernel instead of forming
an explicit dense pseudoinverse. Native scratch includes dense R staging, its
source-precision SVD, source Q*U_R, promoted coefficients/weights and projection.
Cutoff comparison remains in the source precision, while reciprocals are formed
in the working precision, consistent with public retained spectral solves.

Same-type normal solves without explicit tolerance use a sparse rank certificate
and borrow native factors when it proves full column rank. Their query and
kernel receive the actual team. This preserves linear-size solve work for large
well-conditioned sparse factors rather than forcing a dense R SVD. Default tall
full-rank calls retain their established fast path.

Q/R/permutation materialization for the spectral path is independent preparation
outside solve scratch; Q extraction remains sequential. This change does not
claim allocation-free whole API calls or finish Q extraction/construction
resources. RHS/result ownership and staged destination conversion also allocate
outside scratch. Inverse-operator construction retains its separate policy.
Other sparse factor construction, iterative resources and all 76 inventory rows
remain in scope. Existing rejection tests for unsupported QR workspace forms
are updated to verify newly supported behavior; insufficient-buffer checks remain.

Final validation at Julia 1.13.3 / RustCall.jl 0.9.2 and unchanged t4a-faer
0.29.0 revision `19053d8c22993851c032d5373b4ec188998f2b5c`: canonical
Pkg.test passed 372,115 assertions / 102 files / 16 workers in 7m59.7s.
The new file contributes 75,717 assertions. Native cargo test passed all 75
tests; all 88 README blocks passed. An additional executed 1,440-assertion
check covered all-zero maps, zero-column RHS and all-clipped tolerances.
JuliaFormatter DefaultStyle, rustfmt for the new native module, diff and
frozen-inventory checks passed. No new RustCall/faer defect was identified.


## Right solves with transposed/adjoint vector operands

An executed main-branch reproducer found that sparse LU/QR/Cholesky right
solves rejected `adjoint(vector)` with a FaerMatrix constructor MethodError.
The same normalization pattern existed in retained dense factor rdiv!.
Although these row operands are AbstractMatrix, adjoint of an adjoint or
transpose vector can be one-dimensional. The three frontends now explicitly
reshape the normalized RHS to `(size(B,2),size(B,1))` before native conversion.
This preserves row orientation and conjugation for vector-backed rows and
ordinary matrix operands. Native Rust code and dependency revisions are unchanged;
this was a Colmat wrapper defect, not a RustCall/faer issue.

Regression tests cover all four scalar types and mixed complex RHS, source
transpose/adjoint, transposed/adjoint row operands, empty/scalar/square and
rectangular factors, all sparse algorithms, native destinations, source/plan
closure, scratch/team, writable row destinations and failed conversions.
The focused run passed 26,980 assertions. Final canonical/README results follow.

Final validation: canonical Pkg.test passed 399,095 assertions / 103 files /
16 workers in 8m16.4s, including 26,980 row-operand regression assertions. All
88 README blocks passed. Additional executed FaerVector-backed row RHS checks
passed 36 assertions. JuliaFormatter DefaultStyle, diff and frozen-inventory
checks passed. The Rust tree is byte-identical to PR #49's 75-test validated
tree; this Julia-only correction does not change native code or faer revision.


## Borrowed sparse QR solve component preparation

The R-SVD solve no longer prepares a full dense Q, a sparse R export or copied
column permutations. R is staged directly from retained simplicial/supernodal/
wide numeric storage into the declared buffer. Saved reflectors apply Q directly
to U_R, avoiding full-Q materialization. Pivot completion and supernodal block
metadata use stack arrays; per-node gather and blocked Householder work share
the tail of the buffer with the R SVD. Query and execution use matching layouts
and actual team conditions. Indexed simplicial reflector traversal is sequential;
blocked Householder calls borrow the caller team.

The existing conservative rank bound is prepared once per numeric factor
revision, before installation, and copied/refreshed with factor clones/updates.
This preserves its original source-precision criterion while eliminating R
exports and row/column-sum allocations from repeated selection queries. One-time
rank-bound preparation remains part of the separately allocating construction
policy, including numeric construction calls that accept caller scratch. It is
not included in the numeric factorization kernel's scratch query yet.

Native comparison tests cover all four scalars and all three algorithms,
empty/tall/wide/square/96-by-64 blocked shapes, missing prefix pivots and deficient
factors. They check R, permutations, Q application and independent reconstruction
against the input. After warming, component staging/application and cached rank
queries made zero Rust heap allocations on the calling thread with Seq/Team2.
Worker-thread allocations, factor construction, native result ownership and
Julia conversions are not covered by that measurement; no whole-API
allocation-free claim is made. The full 76-row construction/iterative/kernel
scope remains open.

The no-copy statement concerns the R-SVD workspace kernel and mode query. Tall
full-rank defaults without explicit resources retain legacy mixed-type and
transpose/adjoint component preparation; forcing SVD there would regress large
full-rank sparse solves. Their further workspace/temporary control remains in
the full-scope audit alongside explicit Q/R output construction.

Final validation at Julia 1.13.3 / RustCall.jl 0.9.2 and unchanged t4a-faer
0.29.0 / `19053d8c22993851c032d5373b4ec188998f2b5c`: canonical Pkg.test
passed 399,529 assertions / 104 files / 16 workers in 8m16.1s. The new revision
tests contribute 434 assertions. Native cargo test passed all 80 tests; all 88
README blocks passed. JuliaFormatter DefaultStyle, rustfmt for the new native
modules, diff and frozen-inventory checks passed. No new RustCall/faer defect
was identified.


## QR numeric construction rank-bound preparation in caller scratch

The auto/supernodal QR construction query now includes the existing conservative
source-precision rank bound. Three f64 accumulation arrays share a buffer with
the numeric QR kernel, since the stages execute sequentially. An allocation-free
borrowed R visitor replaces the temporary CSC R export. Ordinary construction
uses the same preparation through a temporary native buffer. Numeric factors and
the rank hint are installed only after both stages succeed; short buffers and
pattern failures preserve an existing native revision.

Construction/query boundaries now validate current native matrix dtype, dimensions,
storage and stored count, plus plan kind, dimensions and triangle, before Julia
snapshot value reads. The old query accepted an executed Float32 wrapper around
Float64 native storage and returned a requirement; the new guard rejects this
metadata mismatch. No unsafe snapshot operation was run for that baseline check.
This is wrapper validation, not a RustCall/faer defect. Native output-pointer
query alignment/null/span checks were also added.

Native tests compare borrowed rank preparation to the previous materialized-R
criterion across all four scalar types, algorithms and full/deficient/empty/wide/
blocked fixtures. Prepared accumulation uses zero Rust heap allocations on the
calling thread. This excludes numeric output, input conversion and other stages.
The public construction scratch interface still targets auto/supernodal QR;
simplicial/wide construction, construction teams, symbolic/input normalization/
snapshot/finite checks and persistent numeric output allocation remain separate.
The complete 76-row scope remains open.

Final canonical `Pkg.test(;allow_reresolve=false)` passed **400,854 assertions /
105 files / 16 workers in 8m09.1s**. The new construction/metadata tests contribute
1,325 assertions; 434 existing numeric-revision/source-precision and 393 existing
construction assertions also passed in focused runs. Native cargo test passed all
85 tests and all 88 README blocks passed. JuliaFormatter DefaultStyle, rustfmt
for changed native modules, diff and frozen-inventory checks passed. The selected
clean t4a-faer remains 0.29.0 / `19053d8`; no new upstream defect was confirmed.

## Auto/supernodal sparse QR numeric construction teams

Numeric factorization now accepts `team=` together with optional `scratch=`.
Queries enter the reserved team scope to use the actual parallelism shape, and
execution passes that same shape/token to faer's numeric QR kernel. The existing
sequential path shares the same implementation. Team-only construction reserves
a temporary buffer sized for the team and releases it after the native call.
`qr(A;scratch,team)` forwards both resources to numeric factorization. Rank-bound
preparation remains sequential and reuses the buffer after the numeric kernel.

Invalid/closed team or scratch and insufficient capacity are rejected before
snapshot reads. Saved numeric factors own their output and remain usable after
resource/source/plan close. Symbolic analysis, simplicial/wide construction
resources, sparse LU/Cholesky construction resources, input snapshots and owned
output allocation remain pending; the full 76-row scope is unchanged.

Final canonical `Pkg.test(;allow_reresolve=false)` passed **405,573 assertions /
106 files / 16 workers in 8m16.7s**. New focused team checks passed 4,719
assertions, plus 1,325 existing construction/metadata checks. All 89 native tests
and 88 README blocks passed. JuliaFormatter DefaultStyle, rustfmt for the changed
native module/owned constructor, diff and frozen-inventory checks passed.
Julia 1.13.3 / RustCall.jl 0.9.2 and clean t4a-faer 0.29.0 / `19053d8` remain
unchanged. No new upstream defect was confirmed.

## Explicit simplicial sparse QR numeric construction scratch

The explicit simplicial QR adapter now separates owned numeric output from the
faer kernel's temporary workspace. `new_in` uses a supplied MemStack; ordinary
construction reserves a temporary buffer and calls the same implementation.
The public construction query combines kernel scratch and the rank-bound
accumulators sequentially, and resource execution installs the numeric revision
only after both stages succeed. `factorize(plan,A;scratch,team)` and
`qr(A;algorithm=:simplicial,scratch,team)` use the existing common resource path.
The simplicial numeric kernel is sequential: faer's API takes no Par argument.
A supplied team is checked/entered, but does not parallelize this kernel or
change its scratch requirement. This is documented rather than claiming
parallel speedup. Wide construction, symbolic analysis, input/snapshot and
persistent output allocation remain separate work. All 76 inventory rows remain
in scope. No new upstream defect was confirmed.

Final canonical `Pkg.test(;allow_reresolve=false)` passed **410,092 assertions /
107 files / 16 workers in 8m27.3s**. New focused construction checks passed
4,519 assertions, plus 6,044 existing construction/metadata checks. All 93 native
tests and 89 README blocks passed. JuliaFormatter DefaultStyle, rustfmt for
changed native modules, diff and frozen-inventory checks passed. The selected
clean t4a-faer remains 0.29.0 / `19053d8`, Julia 1.13.3 and RustCall.jl 0.9.2.

## Wide sparse QR numeric construction scratch and teams

Wide construction now uses caller scratch for prefix QR and saved full-Q-adjoint
column projections, sequentially reusing one buffer. The symbolic query bounds
pivot/used indices, column/result/gather matrices and block-Householder work per
node; numeric block sizes cannot exceed their retained symbolic tau layout. The
prefix kernel and blocked Householder calls use the same scoped caller team.
Indexed simplicial reflection traversal remains sequential. R assembly visits
retained prefix R directly and does not export another sparse R. Default wide
construction uses the same implementation with a temporary buffer.

Full-Q-adjoint application also shares the new scratch kernel in the default
wide solve path, preserving the old source-precision minimum-norm policy. Native
tests retain the old implementation as a test-only reference and compare both
it and an independently formed full Q. Prepared application is measured with
preallocated output and scratch; this does not include setup or worker heaps.
Input normalization, prefix CSC, output triplet/CSC assembly, symbolic analysis
and persistent numeric output allocate separately. All 76 rows remain in scope;
no whole-API allocation-free construction claim is made.

Final canonical `Pkg.test(;allow_reresolve=false)` passed **423,060 assertions /
108 files / 16 workers in 8m25.1s**. Focused wide construction checks passed
12,968 new assertions plus 4,719 existing construction-team checks. All 97 native
tests and 90 executable README blocks passed. The prepared Q-adjoint kernel
uses zero calling-thread Rust heap allocations after warmup with sequential and
team2 execution; setup and worker heaps are excluded. JuliaFormatter DefaultStyle,
rustfmt for changed native modules, diff and frozen-inventory checks passed.
Julia 1.13.3 / RustCall.jl 0.9.2 and clean t4a-faer 0.29.0 / `19053d8` are
unchanged; no new upstream defect was confirmed. Staged `qr!` resource keywords
and separately allocating sparse output assembly remain tracked.

## Retained sparse QR numeric update resources

`qr!(F,A)` accepts optional construction scratch/team and forwards them through
the existing staged numeric refactorization. Queries and scratch/reserve helpers
validate the factor handle and use its retained plan with the same selected
team. Every shape/algorithm is supported, including wide prefix/projection and
explicit simplicial's sequential kernel. Symbolic controls are preserved.
The update builds the next numeric revision before replacing handle, dimensions,
status, plan and source, then releases the former owned plan/snapshot. Failed
capacity/resource/pattern/value/metadata checks preserve the existing revision.
Native Rust code and dependency revision are unchanged; this wires the Julia
update API to the previously validated construction kernels.

The operation remains staged, with snapshots, sparse assembly and persistent
output allocation separate from kernel scratch. Direct native in-place updates
and the remaining construction/iterative/kernel resources remain in the full
76-row scope. No new upstream defect was confirmed.

Final canonical `Pkg.test(;allow_reresolve=false)` passed **433,194 assertions /
109 files / 16 workers in 8m17.0s**. Focused update checks passed 10,134 new
assertions plus 434 existing numeric-revision/source-precision checks. All 91
README blocks passed. JuliaFormatter DefaultStyle, diff and frozen-inventory
checks passed. The Rust tree is identical to PR #55's 97-test-validated tree;
no native source or dependency checkout was changed. Julia 1.13.3 / RustCall.jl
0.9.2 and clean t4a-faer 0.29.0 / `19053d8` remain selected.

Upstream main advanced to `c099d72` through PR #261, changing only PATCHSET.md's
release wording (0.28.3 to the actually published 0.29.0). Production code and
the held #258 numerical fixes/author condition are unchanged, so the validated
dependency is retained.

## Sparse Cholesky numeric construction scratch and execution teams

A common sparse construction query/execute facade now covers QR and LLT. The
Cholesky path uses the retained symbolic LLT requirement for the selected team
and passes the supplied MemStack/Par to numeric factorization. Default
construction uses a temporary native buffer through the same kernel. The
selected triangle is still filtered before entering the native kernel so
ignored values, including NaNs, cannot contaminate its workspace.

Public plan-based factorize, Hermitian/Symmetric/plain sparse cholesky and
staged cholesky! accept resources; query/FaerScratch/reserve support factorize
and cholesky!. The retained plan fixes ordering and triangle. Checked failed
updates preserve the prior revision; check=false installs unsuccessful status
and supports recovery by a later update. Simplicial kernels remain sequential.
The shared Julia resource file is now sparse_construction_workspace.jl.

Input snapshots/CSC normalization, triangle triplet/CSC assembly, symbolic
analysis and persistent output allocate separately. This is kernel workspace
control, not whole-API allocation-free or direct native in-place construction.
LU construction and the full 76-row iterative/kernel/tuning work remain open.

Final canonical `Pkg.test(;allow_reresolve=false)` passed **449,861 assertions /
110 files / 16 workers in 8m23.3s**. Focused construction/update checks passed
16,667 new assertions plus 10,134 existing QR-update assertions. All 102 native
tests and 92 executable README blocks passed. Prepared LLT kernel execution
uses zero calling-thread Rust heap allocations after warmup with width1/width2
(all scalar types, algorithms and triangles); setup and worker heaps are excluded.
JuliaFormatter DefaultStyle, changed native-module rustfmt, diff and frozen
inventory checks passed. No new upstream defect was confirmed.
Julia 1.13.3 / RustCall.jl 0.9.2 and clean t4a-faer 0.29.0 / `19053d8` remain
selected. Upstream main `c099d72` and the draft/open #258 hold are unchanged.

## Sparse LU numeric construction scratch and execution teams

The retained LU plan now owns faer's public low-level SymbolicLu and an owned
NumericLu adapter instead of a high-level owner with private symbolic storage.
The adapter pairs those exact public outputs in LuRef for existing component,
permutation and solve paths. Symbolic ordering/COLAMD controls are computed once
and retained; no symbolic reanalysis is performed by numeric workspace queries.
Default and caller-scratch construction use the same numeric kernel and actual
Par. Explicit simplicial LU also has a provided-stack constructor; it remains
sequential. No private faer fields, transmute or dependency-local edits are used.

The common construction query/execute path covers LU/QR/Cholesky. LU and staged
lu! accept scratch/team, with query/FaerScratch/reserve using the saved plan.
Checked singular updates preserve the prior revision; check=false records an
unsuccessful status and supports recovery. The low-level LU routine owns dynamic
factor/permutation buffers, so output allocations can occur inside the kernel
separately from its declared temporary scratch. Input normalization, symbolic/
output assembly, direct native in-place updates and the full 76-row scope remain
open. No whole-API allocation-free construction claim is made.

Final canonical `Pkg.test(;allow_reresolve=false)` passed **466,838 assertions /
111 files / 16 workers in 8m19.8s**. Focused LU construction/update checks
passed 16,977 new assertions plus 21,386 existing QR/Cholesky construction
checks. All 107 native tests and 93 executable README blocks passed.
JuliaFormatter DefaultStyle, changed native-module rustfmt, diff and frozen
inventory checks passed. Julia 1.13.3 / RustCall.jl 0.9.2 and clean t4a-faer
0.29.0 / `19053d8` remain selected. No new upstream defect was confirmed.
Upstream main `0693fae` (PR #263 after `c099d72`) changes only LU square-input
panic documentation, with no behavior change; the #258 adoption hold remains.

## Iterative CG/BiCGSTAB/LSMR resources (Issue #22)

The three iterative solvers and their staged bang variants accept caller-owned
`scratch` and scoped `team`, with matching `scratch_requirements`, `FaerScratch`
and `reserve!`. The query uses the promoted scalar type, RHS columns, checked
operator/preconditioner owners and actual team shape; it does not apply operators
or invoke callbacks. Missing preconditioners become checked identity owners on
the new resource boundary; no new optional raw-owner arguments are introduced.

Native iteration, the LSMR reference normal residual, and independently recomputed
final primal/normal residuals reuse one buffer sequentially. Default sequential
entry points use the same core. Final diagnostics no longer clone adjoint
operators or allocate temporary heap matrices. Warmed native diagnostic kernels
are checked for zero calling-thread Rust heap allocations; result staging,
operator/RHS conversion and promotion, Julia actions, snapshots and history
storage remain separate allocations, and worker allocations are outside that
measurement. No whole-call allocation-free or direct native in-place claim is made.

The Julia task is pinned during every native iterative frame, including a native
operator with a yielding Julia iteration callback, and its prior sticky state is
restored on success or error. Callback snapshots remain independent, stop requests
retain the stopped iterate, and callback errors (including `throw(nothing)`)
return through Rust before propagation. Staged bang destinations remain unchanged
on errors. Capacity validation precedes observer construction/applications; native
failure tests preserve result owners and statistic buffers on insufficient scratch
or a closed team. Existing LSMR true-normal-residual protection for faer #245 is
preserved.

Focused Julia tests pass 23,217 assertions: 22,651 basic/resource/metadata cases
and 566 callback/shared-resource cases. They cover all four scalar types, native
dense/CSC/CSR and Julia-defined operators, vectors/block RHS, empty and rectangular
LSMR shapes, blocked dimensions, initial/native destinations, promotion, views,
short/reserve/closed resources, history/stops, exception recovery, independent
snapshots, task sharing and sticky-state restoration. The same 566 callback
assertions also pass in a separate four-Julia-thread run. Native tests pass 112,
including four scalar-specific solver/query/diagnostic allocation suites and an
invalid-resource preservation test. An independent 96-assertion probe covers
composed/transposed/adjoint/conjugated operators and retained Cholesky inverse
preconditioners after their source/factor owners close.

The full 76-row inventory remains frozen. Partial-spectral resources, remaining
construction/kernel scratch/team/tuning, direct in-place and other limitations
continue in the canonical mapping. No new upstream defect was confirmed.

Upstream main advanced to `0f9330f5df129e9e7a4777b078a88741a711e8a7`:
the change after previously inspected `0693fae` adds an internal real-QZ
scratch-query regression test for #252/#253, with no production behavior change.
The #258 draft remains open at `e475baf`, and its maintainer approval condition
still applies; #256/#260 fixes remain candidates and #257 remains unresolved.
The selected clean dependency remains t4a-faer 0.29.0 / `19053d8`.

Final canonical `Pkg.test(;allow_reresolve=false,test_args=["--jobs=16"])`
passes **490,055 assertions / 113 files / 16 workers in 8m57.2s** on Julia
1.13.3 / RustCall.jl 0.9.2 and clean faer `19053d8` (0.29.0). All **94 README
blocks** pass. JuliaFormatter, the new native module rustfmt, diff checks and
frozen inventory verification pass. Runtime/test checksums remained unchanged
throughout these final gates.

## Partial eigen/SVD resources (Issue #22)

`partial_eigen`, `partial_self_adjoint_eigen` and `partial_svd` now accept
caller-owned `scratch` and scoped `team`, with matching `scratch_requirements`,
`FaerScratch` and `reserve!`. Default calls share the same resource core. Queries
validate native dtype/shape before sizing or reading start buffers, account for
projection options and actual team shape, and do not apply operators or callbacks.
Zero requested columns or zero restarts require no kernel scratch.

The native boundary keeps the existing restarted partial methods and the explicit
small/full-projection self-adjoint/SVD fallback for faer #204/#213. Fallback
materializes operator columns in caller scratch, then runs low-level EVD/SVD in
the remaining stack with the actual Par. Self-adjoint fallback preserves Julia
Hermitian's upper-triangle convention. Real general eigen requests still return
complex vectors; the composite query includes scalar-promoted final residual
applications and both forward/adjoint applications for SVD. Each stage reuses the
buffer sequentially. Native output staging, start/result storage, sorting,
operator promotion and Julia callback buffers allocate separately. No whole-call
allocation-free claim is made.

Every new native entry point uses checked operator/scratch/team owners. Start and
output buffer alignment, byte span and dimensions are checked before native
access; staged outputs are exported only after successful computation. Native
failure tests preserve result arrays on insufficient scratch, closed teams and
metadata mismatch. Julia task pinning covers the native frame, including yielding
Julia operator actions, and restores prior state on exceptions. Existing true
residuals, output sorting, returned column counts, start/projection validation and
`:partial`/`:dense_fallback` reporting are retained.

Focused resource tests pass **8,212 assertions**, covering four scalar types; dense/CSC/CSR and Julia-defined
operators; empty/square/tall/wide shapes; partial and forced fallback paths;
capacity/reserve; idle calls; invalid/closed owners; start-buffer and forged
operator metadata; non-diagonal 129-order decompositions; composed and transformed
operators. A separate four-Julia-thread run passes **2,968 assertions**, covering yield/thread identity,
sticky restoration, `throw(nothing)` and ordinary errors, recovery, independent
retained buffers and Tasks sharing scratch/team. Retained Cholesky inverse actions
remain valid after source/factor owners close. Native tests pass **117**, including
four scalar-specific native query/solve suites and resource-failure preservation.

The complete 76-row mapping and frozen inventory are retained. Other constructor,
primitive-kernel, symbolic/output-assembly scratch/team/tuning, direct native
in-place, view and scalar limitations remain open. No new upstream defect was
confirmed. Upstream main remains `0f9330f` and #258 remains draft/open at
`e475baf`, with its approval condition unchanged; verified clean t4a-faer 0.29.0 /
`19053d8` remains selected.

All **95 executable README blocks** pass on the final runtime tree.
JuliaFormatter, new native module rustfmt, diff checks and the frozen inventory
verification pass; the mapping retains all 76 rows.

Final canonical `Pkg.test(;allow_reresolve=false,test_args=["--jobs=16"])`
passes **501,235 assertions / 115 files / 16 workers in 9m08.1s**, on Julia
1.13.3 / RustCall.jl 0.9.2 / clean faer `19053d8` (0.29.0). The final runtime/test
checksums remained unchanged throughout the canonical, README and focused gates.

## Householder sequence construction teams and saved-factor reblock (Issue #22)

Sequence construction from basis/denominators or basis/block factors accepts
`team`. Factor-input construction also accepts a target `blocksize` and
`prev_block_size`. `reblock`/`reblock!` now create the replacement directly from
the saved native basis/factor owner, instead of exporting basis/tau through Julia
and regenerating from scalar blocks. They preserve orientation and commit the
replacement handle/blocksize only after success. Exported component matrices
remain independent.

The previous width must positively divide both source and target block sizes;
the default is their gcd. That ensures every copied previous block is contained
in a valid source block and a target block. Its strict upper coefficients and
diagonal denominators are copied into the new native output, then faer's upgrade
receives that actual previous width and the scoped Par. Arbitrary positive target
block sizes remain supported through the default. This interface describes
reuse of already valid previous blocks, not unchecked claims that missing cross-
block coefficients exist. Output owners still allocate independently.

Native input spans/alignment/size are checked before reads, and Julia validates
native source owners before snapshot conversion. Native source dtype, dimensions,
reflection count and block size are checked before reblock allocation. Invalid
widths, forged metadata, null input and closed teams preserve the existing native
output; the staged Julia bang form preserves its original owner. Factor-matrix
validation keeps the existing basis/denominator and upper-block checks, with
validation owners closed in finally blocks. A validated provided factor is no
longer redundantly upgraded just before its coefficients overwrite the result.

Focused tests pass **45,674 assertions** (45,648 numerical/lifetime/orientation/
block combinations plus 26 validation/atomic/shared-team checks), over all four
scalars, no team/width 1/2, empty/rectangular/blocked/1024-row bases, identity
reflectors, common previous widths, arbitrary new target widths, component
snapshots and source/factor close. Independent sequential reflector applications
check products without building a large dense Q. Existing sequence tests pass
**13,432 assertions**. Native **122 tests** include four scalar suites with
independent Gram/denominator references at up to 8192-by-32, width 1/2 and
previous-width reuse, plus native failed-output preservation.

Construction/reblock caller-owned scratch remains open. The native upgrade API
has no scratch parameter/query and its deep parallel branch allocates a partial
buffer on every call. On actually executed latest main `cd516af`, a standalone
release MWE measures one allocation per warmed width-2 call (8192/16384/16384/
32768 requested bytes for f32/f64/c32/c64), versus zero at width 1. Both paths
pass independent strict-upper Gram checks. This is an already recorded site in
[faer #93](https://github.com/tensor4all/faer/issues/93), so no duplicate issue
was opened: [executed evidence and a concrete canonical scratch API proposal](https://github.com/tensor4all/faer/issues/93#issuecomment-6648218975)
were added there. The retained, locked standalone crate is
`docs/reproducers/householder-upgrade-allocation/`. No performance/scaling or
whole-call allocation-free claim is made. The proposal has not been implemented.

Upstream `0f9330f..cd516af` changes migration/owned-team documentation and examples
only; the Householder kernel is identical to selected clean `19053d8` (0.29.0).
The selected dependency is unchanged. #258 remains draft/open at `e475baf` and
its approval condition still applies. The full 76-row scope, frozen inventory and
remaining construction/kernel/symbolic/output workspace/tuning, direct in-place,
view and scalar limitations remain tracked.

Final canonical `Pkg.test(;allow_reresolve=false,test_args=["--jobs=16"])`
passes **546,909 assertions / 116 files / 16 workers in 9m22.5s**, on Julia
1.13.3 / RustCall.jl 0.9.2 / clean faer `19053d8` (0.29.0). All **96 README
blocks** pass. JuliaFormatter, new native module rustfmt, diff and frozen inventory
checks pass; final runtime/test checksums remained unchanged through the gates.

## AMD/COLAMD caller-owned workspace

Standalone `amd`/`colamd` accept `scratch=` and provide exact requirements,
`FaerScratch` construction and explicit `reserve!`, using the existing faer
AMD/COLAMD queries and kernels. Four scalar numeric CSC/CSR inputs, symbolic
patterns, Julia CSC and existing sparse wrappers retain their ordering and AMD
statistics. Native shape metadata, workspace capacity/alignment and lifetime
are checked before execution. Controls retain faer's existing semantics and do
not change scratch size; the kernels are sequential. Temporary pattern
conversion, query buffers and the owning output permutation allocate separately.
Symbolic factor analysis's internal ordering workspace and the other incomplete
entries in the 76-row mapping remain in scope.

## Merged QZ fix dependency update

The validated faer 0.29.0 selection is main
`2d4ea4eab8d8f724269508e16b4aaafaf5c67ae7`: the even-window query and
fully-deflated row-zero sentinel fixes are present. The native Float32/Float64
n=78 regression fails on 19053d8 and passes with bounded independent
reconstruction/orthogonality on the new revision. The related Julia regression
and full package checks validate the integrated update. The wrapper's #260
scratch correction stays in place until its separate upstream fix is merged
and verified. See [the retained reproducers](reproducers/faer-qz-numerics.md).
Householder/LLT constructor scratch proposals #267/#269 and #258's range/index
changes remain unselected; the full 76-row inventory remains in scope.

Final validation on the selected source: 556,244 assertions / 117 files /
16 workers (9m06.0s), 125 native tests and 97 executable README blocks pass.
Runtime/test file hashes and the clean dependency revision are unchanged
through the gates. Formatting and the 76-row/frozen-inventory checks pass.

## Generalized Hessenberg and QZ construction resources

Matrix-pair `hessenberg`/`schur`, their staged bang forms, and saved generalized
Hessenberg `schur`/`schur!` accept caller scratch and a scoped native team.
Requirements, FaerScratch construction and reserve cover the same operation,
promoted scalar and actual team shape. QR factor coefficients and Q formation
use low-level routines; the coefficient phase, projection and Hessenberg/QZ
phases reuse one queried stack. Homogeneous-value temporaries are initialized
stack arrays. The real-QZ correction for #260 keeps its four index arrays and
now receives the actual token shape, including generalized EVD queries.

The new native execution path uses checked owner references for all four
optional saved factors (no new raw-owner exception). Original native owner,
view and common structured-wrapper metadata is checked before promotion;
capacity/alignment and closed resources are rejected before conversion or
output setup. Owning matrices and Julia outputs/promotion/staging allocate
separately. The selected faer QR's internal deep Householder buffer can still
allocate; its external control awaits #267. No whole-API allocation/scaling
claim follows from the caller-thread zero-allocation test of the warmed QZ
step with precreated pair owners (four scalars, Seq/Team(2); workers/setup are
outside that counter).

Public tests cover construction/reuse/bang, four scalars, empty/singleton and
two-block-plus-tail shapes, views/mixed sources, capacity/alignment/reserve,
Tasks/lifetime/close, native metadata before conversion, and failed writeback.
A singular-pencil fixture initially assumed an invariant finite spectrum;
the unchanged parent produces the same valid Schur blocks. Its checks now
verify returned-block homogeneous consistency plus reconstruction/unitarity,
and a separate regular pencil checks finite and infinite eigenvalues. No
numerical tolerance was loosened and no upstream defect was inferred.
Early preflight retains QZ-specific current-native-dimension and requested-scalar
diagnostics, rather than masking them with the generic dense-solve validator.
The existing execution regressions and 28 added scratch-query checks cover both.
The 76-row inventory and the other kernel/tuning/in-place work remain open.

Final canonical `Pkg.test(;allow_reresolve=false,test_args=["--jobs=16"])`
passes **559,647 assertions / 118 files / 16 workers in 9m24.1s**, with
**8,544 focused assertions**, **127 native tests** and **98 executable README
blocks** passing. JuliaFormatter, native rustfmt, diff and frozen-inventory
checks pass. All 247 runtime/test file hashes and clean faer `2d4ea4e` remain
unchanged through the final gates. The 76 mapping rows are retained.

## Single-matrix Hessenberg/Schur construction resources

Single-matrix `hessenberg`/`schur`, staged bang forms and Schur conversion from
saved native Hessenberg factors accept caller scratch and an actual scoped team.
The common query/execution path retains current-native metadata checks before
view materialization or output setup. Capacity/alignment and closed resources
are checked before native computation; saved H/Q use checked owner references.
Hessenberg Householder coefficients and real/complex Schur temporary eigenvalue
columns use stack storage. Sequential convenience calls use the same kernels
with automatically owned workspace. Source scaling, structure validation,
independent output ownership and staged bang writeback remain intact.

Owned output matrices, view copies, Julia eigenvalue arrays and staged results
allocate separately. The selected faer may allocate the deep Householder partial
buffer; its external control awaits unmerged #267. The warmed Schur-step
calling-thread allocation test excludes setup, owning output creation and worker
heaps; it does not imply a whole-API allocation or scaling bound. Tuning, Schur
reordering/eigenvector kernel resources, direct in-place and all other incomplete
76-row inventory entries remain in scope.

A review probe found a candidate-only dispatch omission for square matrix views
whose retained owner is a FaerRow. Parent main supports that input. The checked
collection-metadata validator is reused before view copies, and new four-scalar
Seq/Team tests cover query, execution, stale owner metadata and close. The initial
frozen validation processes were deliberately stopped before this source change;
final gates use the corrected frozen source. No upstream defect was inferred.

Final canonical `Pkg.test(;allow_reresolve=false,test_args=["--jobs=16"])`
passes **562,600 assertions / 119 files / 16 workers in 9m23.6s**, with
**4,563 focused assertions** (2,953 new), **129 native tests** and **99 executable
README blocks** passing. JuliaFormatter, native rustfmt and diff checks pass.
All 250 runtime/test file hashes and clean faer `2d4ea4e` remained unchanged
through the corrected final gates. Frozen inventory and all 76 mapping rows
are retained.

## Saved single Schur reordering resources

Saved `ordschur`/`ordschur!` accept caller scratch, scoped teams, declared queries,
FaerScratch construction and reserve. Native selection flags and block-swap
workspace use the caller stack. The default path uses the same loop and kernels
with automatically owned workspace. Original current-native T/Z metadata and
resource capacity/alignment are checked before selection coefficient access or
output setup. New native boundaries use checked owner references and validate
selection/value spans, non-overlap and binary flags before use. Whole real
conjugate-pair selection, independent working factors and staged bang conversion
preserve caller state on failure; temporary bang results are explicitly closed.

Selected faer `reorder_schur_real/cplx` accept but ignore Par and perform sequential
work. Requirements therefore do not depend on team width; passing and validating
a scoped token is not a scaling claim. Owning work factors and Julia selection,
eigenvalue and conversion arrays allocate separately. The warmed n=6 native
calling-thread allocation probe measures exactly the two owning outputs for
four scalars at Seq/Team(2), excluding setup, scratch acquisition and worker heaps.
No whole-call allocation or performance bound is implied. Saved Schur eigenvector
kernel resources, tuning, other incomplete workspace/team/direct-in-place items
and all 76 mapping rows remain in scope.

The first independent Julia eigenspectrum check failed because lexicographic
sorting by numerically equal real parts reordered complex conjugates after
roundoff. It now matches eigenvalue multisets with the same numerical tolerances;
reconstruction and orthogonality bounds are unchanged. No dependency defect was
inferred from that test expectation.

Final canonical `Pkg.test(;allow_reresolve=false,test_args=["--jobs=16"])`
passes **565,569 assertions / 120 files / 16 workers in 9m12.3s**; **4,579 focused
assertions** (2,969 new), **131 native tests** and **100 executable README blocks**
pass. JuliaFormatter, native rustfmt and diff checks pass. All 254 frozen
validation inputs (253 runtime/test files and README), the clean selected faer
`2d4ea4e`, frozen inventory and all 76 mapping rows remain unchanged through
the final gates.

## Saved Schur right eigenvector declared resources

Saved native `eigvecs(F)`/`eigen(F)` accept caller scratch, requirements,
FaerScratch/reserve and an actual scoped team. Current native T/Z metadata and
resources/capacity/alignment are checked before output setup. The old raw-owner
`schur_eigenvectors_from` entry and its exception are removed; new native methods
borrow checked OwnedMatrix or TypedMatrix inputs with a TypedMatrix complex
output. Real packed output uses caller stack alongside faer's public kernel
query; complex output owns only its returned matrix. Execution passes the actual
token through Schur back substitution and Q back transformation. Default calls
share this path. Values remain independent Julia copies and all results outlive
input factors and resources.

The selected faer internally allocates a range-scaled Schur copy outside its
public query. A pinned standalone release MWE executed latest main `2d4ea4e`
for f32/f64/c32/c64: n=32 Seq, three repetitions at each ordinary/large/small
scale, zero allocations ordinarily and one n×n matrix for extreme scales;
independent normalized residuals and unit-column checks pass. Complete source,
observations and a canonical stack/lifetime/query proposal are in
[faer #270](https://github.com/tensor4all/faer/issues/270) and the retained
[locked reproducer](reproducers/schur-eigenvectors-allocation/README.md).
This is a workspace coverage gap, not a wrong-result report. The binding retains
that numerically correct kernel and its allocation. Prescaling the whole public
call can change original real-pair classification after tiny subdiagonal
underflow, so no such numerical workaround is introduced. Internal #270 storage
remains unfinished upstream/downstream work.

The warmed n=6 native compute probe sees one owning output allocation ordinarily,
and two including the known scaled copy at extremes, for four scalars at
Seq/Team(2); setup/scratch acquisition/worker heaps are excluded. No whole-call
allocation or speedup/scaling claim follows. Tuning, other workspace/team kernels,
native parallelization of sequential Schur reorder, direct in-place and all
incomplete entries in the 76-row inventory remain in scope.

Final canonical `Pkg.test(;allow_reresolve=false,test_args=["--jobs=16"])`
passes **572,512 assertions / 121 files / 16 workers in 9m26.0s**; **8,553 focused
assertions** (6,943 new), **133 native tests**, **101 executable README blocks**
and **36 standalone release probes** pass. JuliaFormatter, native/MWE rustfmt,
diff checks, frozen inventory and all 76 mapping rows pass. All 261 frozen
validation input hashes and clean selected faer `2d4ea4e` remained unchanged
through the final gates. Internal scaling workspace #270 remains unresolved.


## Native real extrema (PR #68)

Float32 `minimum/maximum/extrema` now use a checked RustCall reference to the
native TypedMatrix rather than generic Julia scalar iteration. Float64 uses
the same real ordering rule through its existing reduction boundary. The
adapter explicitly preserves Julia NaN propagation and orders negative zero
below positive zero; it cannot use an unspecified equal-zero choice from the
native primitive min/max operation. No raw output pointer or copied Julia
matrix is required. The source lock and existing current native shape/type
validator cover the empty-input decision and computation. Empty/closed inputs
and corrupt shape metadata are rejected. Complex ordering, dims/map keywords
and borrowed-view specializations retain their existing generic behavior.

Focused tests: 240 new Julia assertions plus 107 existing reduction assertions
pass; two new native tests pass. The first focused run attempted to construct a
Float32 matrix containing Inf, which correctly failed the existing finite-only
constructor. The test now constructs finite storage and writes Inf through the
supported element setter before checking extrema. No production constructor or
upstream behavior was changed for that fixture correction.

The official `deps/build.jl` rebuilt the binding with 33 functions and 18 structs
against unchanged clean faer `2d4ea4e`. Final canonical `Pkg.test(;allow_reresolve=false,test_args=["--jobs=16"])`
passes **572,752 assertions / 122 files / 16 workers in 9m34.2s**. All **135
native tests** and **102 executable README blocks** pass. All 263 frozen
validation inputs remain unchanged through these gates, and the frozen
inventory and all 76 mapping rows are preserved. The tested runtime commit is
`8fe8fa6`; subsequent result-record changes are documentation only. This change
advances the reductions row without completing the full inventory or its
remaining options.


## Tridiagonal EVD resources (PR #69)

The dedicated band EVD now queries and consumes caller-owned scratch for the
scaled n-by-2 band, temporary native eigenvalues and the faer kernel. Its actual
Par token/shape are used consistently for query and execution. Direct APIs,
saved FaerTridiagonalization and native SymTridiagonal dispatch share this path;
saved-factor Q back-transformation uses the supplied team separately. The
legacy native entry point delegates to the same computation with automatic
scratch. RustCall 0.9.2 checks OwnedScratch/OwnedTeam references; no raw-owner
exceptions were added. Shared scaling accepts a borrowed MatMut with unchanged
arithmetic, so the band need not be an owning matrix. Output Julia staging is
borrowed only during the call; conversion, output ownership and sorting still
allocate separately. Tridiagonal reduction resources and compact factors remain
unfinished work in the same inventory row.

Capacity/alignment and current source metadata are checked before coefficient
conversion/output setup. Native pointer spans, alignment and input/output
overlap are checked before reference construction; immutable coefficient inputs
may alias. Rejection from insufficient scratch preserves output sentinels.

Focused tests pass 987 new assertions and 1,621 existing assertions, including
four scalars, Seq/Team2, empty/singleton, QR/divide-and-conquer, complex phases,
2^±100 (f32) and 2^±1000 (f64), repeated/reserved/short/misaligned/closed resources,
Tasks, saved factors, native coefficients and stale metadata. A test initially
compared whole eigenvalue vectors with only a fixed absolute tolerance; it now
uses the same precision-based relative coefficient (1024 eps) alongside the
absolute floor, matching the scale-aware residual oracle. Production numerical
code was not changed in response. A closed-team fixture initially used Any[];
it was corrected to Float64[] so the intended numeric method is tested.

All 139 native tests pass. Four new tests check insufficient scratch output
preservation, invalid pointers/overlap and warmed n=6 Seq calling-thread heap
counts of zero. Setup, Julia staging, worker heaps and larger upstream kernel
allocation gaps are outside that observation; no whole-API allocation or
performance/scaling claim is made.

Final canonical `Pkg.test(;allow_reresolve=false,test_args=["--jobs=16"])`
passes **573,739 assertions / 123 files / 16 workers in 9m24.3s**. All **103
executable README blocks** pass, and all 266 frozen validation input hashes,
the frozen inventory and all 76 mapping rows are preserved. Tested runtime is
`bc20ec4`; subsequent result-record changes are documentation only.
