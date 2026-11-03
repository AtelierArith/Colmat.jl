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
