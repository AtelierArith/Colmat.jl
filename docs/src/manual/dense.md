# 密行列の分解と求解

保存した分解を再利用して、求解・再構成・疑似逆行列を計算します。

## 保存した密分解

密行列の分解は Rust に保持して再利用できます。

```julia
using Colmat, LinearAlgebra

A = FaerMatrix([4.0 1; 1 3])
B = FaerMatrix([1.0 2; 3 4])
F = lu(A)
X = F \ B
@assert Matrix(X) ≈ Matrix(A) \ Matrix(B)
@assert Matrix(F.L * F.U) ≈ Matrix(A)[F.p, :]
ldiv!(X, F, B)
Q = qr(A, ColumnNorm())
H = cholesky(Hermitian(A))
E = eigen(Hermitian(A))
@assert Matrix(A) * E.vectors ≈ E.vectors * Diagonal(E.values)
```

dense LU・完全ピボット LU・QR（列ピボットを含む）・Cholesky・LDLT・
Bunch–Kaufman の構築と対応する `!` 版には `team=` を指定できます。
`factorize(A; team)` も対応します。構築後の因子は team や入力を閉じても独立に使えます。
保存因子からの通常の求解・成分取り出しは引き続き逐次経路です。

```julia
using Colmat, LinearAlgebra

a = [4.0 1; 1 3]
A = FaerMatrix(a)
F = FaerTeam(2) do team
    Q = qr(A, ColumnNorm(); team)
    @assert Matrix(Q) ≈ a
    lu(A; team)
end
close(A)
@assert Matrix(F) ≈ a
@assert F \ [1.0, 2.0] ≈ a \ [1.0, 2.0]
```

## Structured products

```julia
using Colmat, LinearAlgebra
A = FaerMatrix([2.0 99; 3 4])
L = LowerTriangular(A)
B = FaerMatrix([1.0 2; 3 4])
C = L * B
@assert Matrix(C) == [2.0 4; 15 22]
D = FaerDiagonal([2.0, 3.0])
@assert Matrix(D * B) == [2.0 4; 9 12]
mul!(B, L, B, 2.0, 0.0)
@assert Matrix(B) == 2Matrix(C)
```

Lower/upper and unit-triangular wrappers use faer's structured multiplication;
These methods return a `FaerMatrix`, including triangular×triangular
products. Native destination `mul!` supports coefficients and aliasing. Operand
snapshots and a temporary result are allocated. Unused triangles and
implicit unit diagonals are ignored, including nonfinite values stored there.

`FaerDiagonal` or Julia `Diagonal` times a `FaerMatrix`, a range view or a
triangular wrapper of a `FaerMatrix` (on either side) scales rows or columns in
Rust: the stored diagonal is borrowed as a column (a Julia `Diagonal` crosses
as its contiguous buffer for one call) and never expanded to a dense matrix,
and the dense operand is borrowed in place. `mul!(C, D, B, α, β)` with
a `FaerMatrix` or view destination folds `α` into an O(n) copy of the diagonal,
builds the result in one Rust buffer and writes it into `C`; `β == 0` never reads
`C`, `α == 0` never reads the operands, and an operand sharing the destination's
owner is first copied natively. Only the diagonal is promoted on this path; a
dense operand of another scalar type, a triangular wrapper of a Julia matrix,
or a destination narrower than the promoted type falls back to the dense
snapshot path. `D * v::FaerVector` and `r::FaerRow * D` return native vectors.

## Pivoted Cholesky

```julia
using Colmat, LinearAlgebra
A = FaerMatrix([4.0 2; 2 1])
F = cholesky(Hermitian(A), RowMaximum(); tol=-1, check=false)
@assert F isa CholeskyPivoted
@assert F.rank == 1
@assert Matrix(F) ≈ Matrix(A)
B = FaerMatrix([4.0 1; 1 3])
G = cholesky(B, RowMaximum(); tol=-1)
@assert Matrix(B) * (G \ [1.0, 2.0]) ≈ [1.0, 2.0]
```

Pivoted factors use independently owned native storage. `F.L`, `F.U`, `F.p`,
`F.P`, `rank(F)` and reconstruction follow `CholeskyPivoted` conventions.
Negative `tol` selects faer's automatic threshold; nonnegative `tol` selects
an absolute cutoff in the Rust adapter's complete diagonal-pivot method because
faer's factor kernel does not expose a tolerance parameter. The Julia default
is `tol=0`. Reconstruction, full-rank solves and inverses use faer's pivoted LLT
kernels. Rank-deficient factors require `check=false`; their solves/inverses
throw `RankDeficientException`. Native snapshots and workspace are allocated.
`close(F)` closes its native factors; a copied factorization remains usable.

### Pivoted Cholesky workspace and blocking

`cholesky[!](A, RowMaximum(); scratch, team, block_size=128)` supports native
matrices and views, including stored-triangle wrappers. A positive `block_size`
controls both faer's automatic-tolerance kernel (`tol<0`) and the adapter's
absolute-tolerance panel algorithm (`tol>=0`). The latter uses native matrix
products for its trailing updates. A team selects the native execution context;
small or sequential parts need not use every worker.
`block_size=0` is rejected before query/compute: the selected faer revision otherwise
loops without advancing its panel index ([faer #283](https://github.com/tensor4all/faer/issues/283)).

The same `tol`, `block_size` and team must be used for query, reserve and compute.
Declared caller scratch includes permutation arrays, faer's kernel temporaries,
and the Hermitian working matrix for explicit tolerance. Returned factors,
view copies, factor/RHS promotion, upper-factor conversion, owned results and
Julia arrays allocate separately.

Saved factors accept `scratch`/`team` in `ldiv!`, `rdiv!`, `inv`, and
`reconstruct`; native RHS `\` and `/` return native matrices. Queries always
put F before B, including right solves. The stored factors are reused across
promoted RHS types. Bang construction and in-place solves stage their output;
destination aliases of factor storage are rejected. Rank-deficient factors
support reconstruction, while solve/inverse still require full rank.

```julia
using Colmat, LinearAlgebra
A = FaerMatrix(ComplexF64[4 im; -im 3])
original = Matrix(A)
team = FaerTeam(2)
work = FaerScratch(0)
try
    reserve!(work, cholesky, Hermitian(A), RowMaximum(); tol=0, block_size=1, team)
    F = cholesky(Hermitian(A), RowMaximum(); tol=0, block_size=1, scratch=work, team)
    B = ComplexF64[1 2; 3 4]
    rhs = copy(B)
    reserve!(work, ldiv!, F, rhs; team)
    ldiv!(F, rhs; scratch=work, team)
    @assert original * rhs ≈ B
    right = copy(B)
    reserve!(work, rdiv!, F, right; team)
    rdiv!(right, F; scratch=work, team)
    @assert right * original ≈ B
    reserve!(work, inv, F; team)
    inverse = inv(F; scratch=work, team)
    @assert Matrix(inverse) * original ≈ I
    reserve!(work, reconstruct, F; team)
    rebuilt = reconstruct(F; scratch=work, team)
    @assert Matrix(rebuilt) ≈ original
    close(rebuilt)
    close(inverse)
    close(F)
finally
    close(work)
    close(team)
    close(A)
end
```

## Destructive factorization entry points

```julia
using Colmat, LinearAlgebra
original = [4.0 1; 1 3]
A = FaerMatrix(original)
F = lu!(A)
@assert Matrix(A) == Matrix(F.packed)
@assert Matrix(F) ≈ original
B = FaerMatrix(original)
Q = qr!(B)
sequence = FaerHouseholderSequence(Q.Qbasis, Q.Qcoeff)
@assert Matrix(sequence) ≈ Matrix(Q.Q)
@assert Matrix(B) == Matrix(Q.packed)
```

`lu!`, `qr!` (including `ColumnNorm()`) and `cholesky!` (including `RowMaximum()`)
compute independent native factors and then overwrite the input after the result
is available. Existing same-shape views observe the packed entries. The returned
factorization remains independent if the input changes or closes. These methods
allocate factors and staging buffers; errors before writeback preserve the input.
Failed ordinary `cholesky!(...; check=false)` returns an unsuccessful factor
without changing the input.

LU packed storage combines U with the strict lower part of L. QR combines R
with the strict lower Householder basis; its block coefficients are exposed as
`F.Qcoeff` and its basis as `F.Qbasis`. This is faer's block format, rather than
LAPACK's tau layout. `F.packed` returns an independent copy, with lower storage
for Cholesky. Upper-triangle Cholesky wrappers receive U in the input instead.

## Direct in-place LU, QR and Cholesky

```julia
using Colmat, LinearAlgebra
A = FaerMatrix([4.0 1; 1 3])
work = FaerScratch(lu!, A)          # scratch_requirements(lu!, A; team)
F = lu!(A; scratch = work)          # faer lu_in_place on A's own storage
@assert F isa FaerInPlaceFactorization && F.factors === A
B = FaerMatrix([1.0 2; 3 4])
solve = FaerScratch(ldiv!, F, B)
ldiv!(F, B; scratch = solve)        # overwrites B with A \ B
C = FaerMatrix([4.0 1; 1 3])
G = cholesky!(Hermitian(C, :U); scratch = FaerScratch(cholesky!, C))
```

Passing `scratch=` to `lu!`, `qr!`/`qr!(A, NoPivot())` or `cholesky!`
(including `Hermitian`/`Symmetric` wrappers) selects the direct path. faer's
`lu_in_place`, `qr_in_place` or `cholesky_in_place` runs on the input's native
storage, and the result is a `FaerInPlaceFactorization` that aliases it, as
Julia's `lu!` does. The packed layout is the same as above. The LU permutation
and the QR block coefficients live in Julia-owned buffers of F. Writes to A
change F. Closing, resizing or truncating A invalidates F. Shape, scalar,
scratch, team, finiteness and Hermitian checks all run before the first write.
A numerical failure (a singular LU with `check=true`, or a non-positive-definite
Cholesky) leaves A partially factored, as LAPACK does.

`ldiv!(F, B; scratch, team)` solves in place on a `FaerMatrix` B of the same
scalar type, including through `transpose(F)` and `adjoint(F)`. Tall QR writes
the least-squares solution into `B[1:n, :]`. `\`, the factor properties,
`Matrix(F)`, `inv`, and `det`/`logabsdet` (LU and Cholesky) are also available.

With caller scratch, Colmat allocates no native memory for the factorization
or the solve. On x86_64 without AVX2/FMA and on other architectures, the
products inside faer's blocked kernels still allocate inside the `gemm` crate
(t4a-faer `upstream-reports/gemm-001`).

The direct factor does not support `rdiv!`, operator or preconditioner use,
pseudoinverse, mixed-scalar promotion of factors, rectangular LU or wide-QR
solves. Those need the default result without `scratch=`. Full-pivot LU,
LDLT, Bunch–Kaufman, pivoted Cholesky and column-pivoted QR have no direct
path yet.

## Destructive indefinite and full-pivot LU decompositions

```julia
using Colmat, LinearAlgebra
original = [1.0 2; 2 -1]
A = FaerMatrix(original)
F = ldlt!(Hermitian(A, :L))
@assert Matrix(A) == Matrix(F.packed)
@assert Matrix(F) ≈ original
B = FaerMatrix([0.0 2; 2 0])
G = bunchkaufman!(Hermitian(B))
@assert Matrix(G.L * G.D * G.L') ≈ [0.0 2; 2 0][G.p, G.p]
C = FaerMatrix(original)
H = full_piv_lu!(C)
@assert Matrix(H.L * H.U) ≈ original[H.p, H.q]
```

`ldlt!`, `bunchkaufman!` and `full_piv_lu!` stage independent native factors
before writing packed entries into the input. Same-shape views see those
entries; returned factors remain independent of later changes or close.
LDLT/block LDLT store D's diagonal plus strict-lower L (or its adjoint for an
upper wrapper). Block LDLT keeps its block off-diagonals in the returned `F.D`
and its permutation in `F.p`; retain those to reconstruct/solve. This is faer's
split block representation. `F.packed` is always lower-oriented. Failed
`ldlt!(...; check=false)` leaves input unchanged because no factor is available;
allowed singular full-pivot LU/block LDLT write the available packed factors.
These entry points allocate factors and writeback buffers.

## 自己随伴分解の Julia 規約

```julia
using Colmat, LinearAlgebra
S = [1e-3 1 0; 1 1e-3 100; 0 100 1e-3]
F = bunchkaufman(Hermitian(FaerMatrix(S)), true)   # rook ピボット
@assert F.rook && Matrix(F.L * F.D * F.L') ≈ S[F.p, F.p]
@assert inertia(F) == inertia(bunchkaufman(Hermitian(S), true))
G = cholesky(FaerMatrix([4.0 1; 2 3]); check = false)  # 非対称入力
@assert G.info == -1 && !issuccess(G)
C = FaerMatrix(ComplexF64[4 1+2im; 1+2im 3])
@assert cholesky(Symmetric(C); check = false).info == -1
```

- `bunchkaufman[!](A, rook=false; check, team)` は Julia と同じ位置引数 `rook` を受け、
  `true` で faer の rook ピボット（`PivotingStrategy::Rook`）を使います。`false` は従来どおり
  faer 既定の `PartialDiag` で、LAPACK の古典的 Bunch–Kaufman（faer の `Partial`）とは
  ピボット列が異なり得ます。`F.rook` は選んだ戦略を返し、`copy` と昇格求解の再分解でも保持されます。
- `inertia(F; atol, rtol)` は `F.D` の 1×1/2×2 ブロックから Julia と同じ許容規則で
  `(正, 負, 零)` 固有値数を数えます。`F.D` を密に取り出すため O(n²) の一時領域を使います。
- `cholesky(A; check=false)` は非 Hermitian 入力に対して Julia と同様 `info == -1` の
  失敗因子を返します（`check=true` は `PosDefException(-1)`）。`cholesky!` はこのとき入力を変更しません。
  失敗因子は数値因子を保持しないため、成分取り出しはエラーになります。`F.UL` は `uplo` 側の三角因子です。
- 複素 `Symmetric` ラッパーは、格納三角（対角を含む）が実数のとき Hermitian として分解します。
  それ以外の複素対称（非 Hermitian）行列は、`cholesky` では Julia と同じく `info == -1`、
  `ldlt`/`bunchkaufman` では `ArgumentError` です。t4a-faer の LDLᴴ/LBLᴴ は自己随伴専用で、
  LAPACK `zsytrf` 相当の転置対称分解を提供しないためです。
- `F.uplo` は Julia の `Char` ではなく `Symbol`（`:L`/`:U`）です。LDLT/Bunch–Kaufman は
  常に下三角向きの `F.L` を返し、Julia の `uplo='U'` における `F.U`（`A[p,p] = U D Uᴴ`）は
  提供しません。密な `ldlt(A)` は Julia 標準には無く（`SymTridiagonal`・疎のみ）、Colmat では
  ピボットなし LDLᴴ として Hermitian 入力を受けます。`cholesky[!](A, NoPivot())` も同じ native 経路です。

## SVD workspace, teams and destructive entry points

```julia
using Colmat, LinearAlgebra
original = [1.0 2; 3 4; 5 7]
A = FaerMatrix(original)
FaerTeam(2) do team
    workspace = FaerScratch(svd, A; team=team)
    F = svd!(A; scratch=workspace, team=team)
    @assert F.U * Diagonal(F.S) * F.Vt ≈ original
    sigma = zeros(size(original))
    for i in eachindex(F.S)
        sigma[i, i] = F.S[i]
    end
    @assert Matrix(A) == sigma
    close(workspace)
end
B = FaerMatrix(original)
values = svdvals!(B)
@assert diag(Matrix(B)) ≈ values
```

`svd`/`svdvals` accept `scratch=` and `team=` for low-level native SVD; existing
calls without these keywords retain their prior route. `scratch_requirements`
and `FaerScratch` accept `svd`, `svdvals`, `svd!` and `svdvals!`; specify matching
`full` and `team` options. The query uses the actual team's execution conditions.
Singular values are real for all four input types. Thin/full and empty factors
follow Julia's SVD conventions. Factor output arrays are Julia allocations;
temporary diagonal values and algorithm workspace use the supplied native buffer.

`svd!` and `svdvals!` replace input by rectangular Sigma only after success.
Same-shape views observe Sigma; returned factors remain independent. This is an
explicit writeback convention, not a LAPACK work matrix. Insufficient scratch,
invalid input or native failure leaves input unchanged. Source/factor bookkeeping
and writeback allocate; neither entry point promises whole-call zero allocation.
Temporary scratch created for a call is closed in a finally block.

## Reconstructing native factorizations

```julia
using Colmat, LinearAlgebra, SparseArrays
A0 = [0.0 2 1; 3 1 0; 0 1 4]
F = lu(FaerSparseMatrix(sparse(A0)); algorithm=:simplicial, ordering=[3,1,2])
R = reconstruct(F)
@assert R isa FaerSparseMatrix
@assert Matrix(R) ≈ A0
@assert Matrix{ComplexF64}(F) ≈ A0
close(F)
@assert Matrix(R) ≈ A0
close(R)
B0 = [2.0 0 0; 0 1 0; 0 0 3]
QZ = schur(FaerMatrix(A0), FaerMatrix(B0))
pair = reconstruct(QZ)
close(QZ)
@assert Matrix(pair.A) ≈ A0
@assert Matrix(pair.B) ≈ B0
close(pair.A)
close(pair.B)
```

`reconstruct(F)` returns independent native storage. Dense owned LU/QR/
Cholesky/LDLT, Schur/Hessenberg and tridiagonal/bidiagonal reductions return
`FaerMatrix`; sparse Cholesky and every sparse LU algorithm return `FaerSparseMatrix`
in original row/column order. Generalized Schur/Hessenberg return `(A,B)` named
fields, reconstructing both members through Q/S-or-H/T/Z. The sparse LU path
uses current native factors and permutations, including numeric updates; it
never copies the retained input snapshot. Its default output is CSC. All sparse QR
algorithms reconstruct a dense native matrix from their saved Q/R factors.

`Matrix{T}(F)` and `FaerMatrix{T}(F)` also support the dense owned decompositions
above; sparse LU adds these conversions and typed sparse CSC output.
Numeric conversion follows Julia rules, including errors for nonrepresentable
complex-to-real values. Conversion temporaries are closed on errors. Reconstructed
owners remain valid after the original factors are updated or closed.

## Retained dense solves with reusable scratch and teams

`ldiv!(X,F,B; scratch,team)` and `rdiv!(X,B,F; scratch,team)` borrow native resources for
retained LU/full LU/QR/pivoted QR/Cholesky/LDLT/Bunch–Kaufman solves, including
transpose/adjoint, rectangular LU/full LU/QR and rank-deficient QR minimum-norm paths.
Two-argument forms require square factors. Solve teams are independent of the
team used to construct the factor; the factor remains usable after either team
closes. Native solve work, including the QR fallback R SVD and projections,
uses caller scratch; the result and Julia staging allocate separately. Mixed
native scalar types can reconstruct/refactor a factor outside this solve query.
QR rank cutoffs retain the source precision after promotion so reconstruction
roundoff does not become new solution directions. Component extraction
remains sequential; promoted matrix products and factorization borrow the team.

```julia
using Colmat, LinearAlgebra
A0=ComplexF64[4 im; 1 3]
F=lu(FaerMatrix(A0))
FaerTeam(2) do team
    B=ones(ComplexF64,2,2)
    X=zeros(ComplexF64,2,2)
    @assert ldiv!(X,F,B;team)===X
    @assert X≈A0\B
    @assert rdiv!(X,B,F';team)===X
    @assert X≈B/A0'
end
@assert F\ones(2)≈A0\ones(2)
close(F)
```

```julia
using Colmat, LinearAlgebra
A0=Float32[1 0; 0 1e-8; 0 0]
F=qr(FaerMatrix(A0),ColumnNorm())
B=ones(ComplexF64,3,2)
X=zeros(ComplexF64,2,2)
team=FaerTeam(2)
work=FaerScratch(ldiv!,X,F,B;team)
ldiv!(X,F,B;scratch=work,team)
@assert X ≈ ComplexF64[1 1; 0 0]
@assert Matrix(F\FaerMatrix(B)) ≈ X
reserve!(work,ldiv!,X,F,B;team)
close(work)
close(team)
close(F)
```

`scratch_requirements(ldiv!,X,F,B;team)` and the corresponding right-solve
argument order describe the working scalar and actual team. Two-argument
`ldiv!(F,B)` / `rdiv!(B,F)` have matching queries for square factors. Capacity,
alignment, closed-resource, conversion and alias failures preserve destinations.
Wide/rank-deficient QR uses the saved Householder/R and column permutation;
the R SVD is computed inside scratch on each fallback solve. Rectangular LU/full
LU reconstructs from saved L/U and permutations and computes its SVD inside
the same workspace; the original input snapshot is not used. `FaerLinearOperator(F;rtol=...)` offers a separately configurable cached
QR R SVD action for repeated applications. Float64 factors split
complex RHS into two real solves that reuse one workspace. Factor construction
and its mixed-scalar reconstruction/refactorization still allocate independently.

## Reusing Rust-owned spectral factors

```julia
using Colmat, LinearAlgebra
A0 = [1.0 2; 3 4; 5 6]
A = FaerMatrix(A0)
F = svd(A; native=true)
close(A)
P = pinv(F)
@assert A0 * Matrix(P) * A0 ≈ A0
B = ones(3, 2)
X = F \ B
@assert Matrix(X) ≈ pinv(A0) * B
close(X)
@assert Matrix(reconstruct(F)) ≈ A0
U, S, V = F
close(F)
@assert Matrix(U) * Diagonal(S) * Matrix(V)' ≈ A0
H0 = [4.0 1; 1 3]
E = eigen(Hermitian(FaerMatrix(H0)); native=true)
@assert Matrix(pinv(E; atol=0, rtol=1e-12)) ≈ inv(H0)
close(E)
close(U)
close(V)
close(P)
```

`native=true` retains actual faer `Svd` / `SelfAdjointEigen` objects in a
`FaerSpectralFactor`, across four scalar types. Ordinary calls keep their current
Julia result types. Native SVD supports thin/full and rectangular/empty shapes;
native self-adjoint EVD honors each selected triangle and `sortby`. Real
`Symmetric` wrappers are supported too. Construction accepts `team=` and uses
backend-owned workspace; caller `scratch=` is not supported for construction.

`pinv(F; atol, rtol, team)` reuses stored factors through faer's pseudoinverse
reconstruction kernel. `reconstruct(F; team)`, `Matrix(F)`, typed conversions,
`rank`, `svdvals` or self-adjoint `eigvals/eigvecs` also reuse the factorization.
`copy/deepcopy` create independent native owners and `close` releases the retained
factors immediately. Components are independent copies: U/V/vectors are native
matrices, S/values are real Julia vectors. Eigen sorting applies a common
permutation to values/vectors. Extracted components survive closing F; modifying
them does not update F. General EVD native ownership, native bang construction
and caller-controlled spectral-owner construction workspace remain unexposed.

Saved factors also support `F \ B`, `B / F`, `ldiv!`, `rdiv!`, and square
`inv(F)`, including transpose/adjoint and mixed scalar types. Solves project the
RHS through the retained native factors, applying a truncated minimum-norm
solution without building a full pseudoinverse. `atol`, `rtol`, `scratch`, and `team` are
accepted; cutoff precision follows the stored factor even when the RHS promotes
the computation. Signed self-adjoint eigenvalues are preserved. Three-argument
mutation supports rectangular factors; two-argument mutation requires square
factors. Destination conversion is staged before writeback. Caller scratch holds
projection, weights and promoted U/V copies; native results and Julia/RHS
conversions allocate separately. Right solves avoid an owned adjoint RHS copy.
Singular inverses throw
`SingularException`.

## Reusable workspace for saved-factor pseudoinverses

```julia
using Colmat, LinearAlgebra
A0 = [1.0 2; 3 4; 5 6]
F = svd(FaerMatrix(A0); native=true)
FaerTeam(2) do team
    workspace = FaerScratch(pinv, F; team=team)
    P = pinv(F; scratch=workspace, team=team)
    @assert A0 * Matrix(P) * A0 ≈ A0
    @assert scratch_capacity(workspace) == scratch_requirements(pinv, F; team=team)
    reserve!(workspace, pinv, F; team=team)
    @assert Matrix(pinv(F; scratch=workspace, team=team)) ≈ Matrix(P)
    close(P)
    close(workspace)
end
close(F)
```

`pinv(F; scratch=workspace, atol, rtol, team)` consumes caller-owned aligned
storage for retained native SVD and self-adjoint EVD factors. The query reads
actual factor metadata and team conditions. `FaerScratch(pinv,F)` and
`reserve!(workspace,pinv,F)` use the same requirements as execution; insufficient
capacity/alignment is rejected and the workspace never grows implicitly.
Repeated calls reuse the allocated buffer and do not repeat SVD/EVD. Thresholds
apply to the stored singular/eigenvalues. The independent result matrix allocates
outside scratch. Empty factors need no kernel workspace but still require an
open buffer. This interface controls saved-factor pseudoinverse workspace;
native factor construction and direct `pinv(A)` keep their existing interfaces.
