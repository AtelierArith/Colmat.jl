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
