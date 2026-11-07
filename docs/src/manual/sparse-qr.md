# 疎 QR と最小ノルム求解

疎 QR の成分・ランク・横長行列・転置求解と作業領域を説明します。

## Sparse QR components

```julia
using Colmat, LinearAlgebra, SparseArrays
A0 = [1.0 0; 2 3; 0 4]
A = FaerSparseMatrix(sparse(A0))
F = qr(A; algorithm=:simplicial, ordering=[2, 1])
Q, R, p = F.Q, F.R, F.pcol
@assert Matrix(Q) * Matrix(R) ≈ A0[:, Vector(p)]
@assert Matrix(reconstruct(F)) ≈ A0
close(F)
@assert Matrix(Q)' * Matrix(Q) ≈ Matrix{Float64}(I, 2, 2)
close(Q)
close(R)
close(p)
close(A)
```

Every sparse QR algorithm retains faer's native Householder/R factors and
symbolic analysis. `F.Q` is an independent thin `FaerMatrix` with dimensions
`(m,n)`; `F.R` is an independent sparse CSC `(n,n)` factor. `F.pcol` (also `F.p`)
is the column permutation satisfying `Q*R ≈ A[:,pcol]`. Extraction, reconstruction,
copy and numeric updates use retained factors. `reconstruct(F)`, `Matrix(F)` and
`FaerMatrix(F)` recover the original column order without using the stored input
snapshot. Four scalar types and CSC/CSR inputs are supported. For explicit
simplicial QR, ordering is identity by default; `ordering=:colamd` or an explicit permutation selects another column
order. COLAMD accepts `dense_row`, `dense_col` and `aggressive` controls and is
retained across symbolic reuse and numeric updates. Auto/supernodal QR uses
COLAMD ordering and exposes the same thin Q, sparse R and reconstruction.
Full Q and rank-deficient minimum-norm solves are described below. AMD ordering
remains unavailable. Caller scratch
is supported for auto/supernodal normal solves and numeric construction as
described below; component extraction and transpose solves allocate workspace.

## Minimum-norm solves with sparse QR transpose/adjoint

```julia
using Colmat, LinearAlgebra, SparseArrays
A0 = ComplexF64[1 0; 2im 3; 0 4]
A = FaerSparseMatrix(sparse(A0))
F = qr(A; algorithm=:simplicial, ordering=[2, 1])
B = ones(ComplexF64, 2, 2)
X = adjoint(F) \ B
Y = transpose(F) \ B
@assert Matrix(X) ≈ pinv(A0') * B
@assert Matrix(Y) ≈ pinv(transpose(A0)) * B
close(X)
close(Y)
close(F)
close(A)
```

For every full-column-rank sparse QR algorithm, `transpose(F) \ B` and
`adjoint(F) \ B` return the minimum-norm solution of the corresponding
underdetermined system. They reuse the native R/Householder factors and column
permutation, without constructing Q or a pseudoinverse. Three-argument `ldiv!`
supports rectangular factors; two-argument `ldiv!` requires square factors.
Native work arrays and sparse R extraction/conversion allocate per solve.
Rank-deficient factors use the SVD minimum-norm path described below. Mixed-type RHS reuse saved native factors for every sparse LU/QR/Cholesky algorithm. Factor values are promoted without
refactorizing the input, so accuracy remains limited by the original factor
precision. Mixed-type QR transpose/adjoint solves also reuse the saved reflectors and R.

```julia
using Colmat, LinearAlgebra, SparseArrays
A0 = Float32[4 1; 1 3]
A = FaerSparseMatrix(sparse(A0))
F = lu(A; algorithm=:simplicial)
B = ComplexF64[1 + im, 2 - im]
x = F \ B
@assert eltype(x) == ComplexF64
@assert A0 * x ≈ B rtol=1e-6
close(F)
close(A)
```

## Auto/supernodal sparse QR components and minimum-norm solves

```julia
using Colmat, LinearAlgebra, SparseArrays
A0 = ComplexF64[4 0; 1im 3; 0 1]
F = qr(FaerSparseMatrix(sparse(A0)); algorithm=:supernodal)
Q, R, p = F.Q, F.R, F.pcol
@assert Matrix(Q) * Matrix(R) ≈ A0[:, Vector(p)]
@assert Matrix(F) ≈ A0
B = ones(ComplexF64, 2, 2)
X = adjoint(F) \ B
@assert Matrix(X) ≈ pinv(A0') * B
close(F)
@assert Matrix(Q)' * Matrix(Q) ≈ Matrix{ComplexF64}(I, 2, 2)
close(Q)
close(R)
close(p)
close(X)
```

The native transpose solve extracts sparse R and applies saved Householder
blocks in reverse order. It does not form a dense Q or pseudoinverse. Extracted
components own their storage and remain valid after numeric updates or close.

## Explicit supernodal sparse QR

`qr(A; algorithm=:supernodal)` and
`FaerSparseSymbolic(A; kind=:qr, algorithm=:supernodal)` force faer's native
supernodal QR. Both retain public low-level symbolic analysis and adapter-owned
numeric buffers. Normal least-squares solves, independent copies, symbolic
reuse and numeric `qr!` updates support all four scalar types and CSC/CSR.
This path uses native COLAMD ordering, with optional `dense_row`, `dense_col`
and `aggressive` controls. `F.p/F.pcol` returns an independent column permutation.
Q/R component getters, transpose/adjoint solves and solve workspace are
available for every algorithm through the interfaces below.

## Reuse workspace for retained sparse QR solves

```julia
using Colmat, LinearAlgebra, SparseArrays
A0 = [2.0 0; 0 3; 1 0]
F = qr(FaerSparseMatrix(sparse(A0)); algorithm=:supernodal)
B = ones(3, 2)
s = FaerScratch(ldiv!, F, B)
X = zeros(2, 2)
ldiv!(X, F, B; scratch=s)
@assert X ≈ A0 \ B
@assert scratch_capacity(s).bytes >= scratch_requirements(ldiv!, F, B).bytes
close(s)
close(F)
```

This interface supports all QR algorithms, mixed vector/matrix RHS, transpose/
adjoint, right solves, rank-deficient and wide factors. Tolerances may be passed
to the query, constructor/reserve and solve. The source precision determines rank
decisions, including after promotion. `reserve!` explicitly grows an existing
buffer; insufficient or closed scratch fails before writeback. Normal full-rank
calls without resources retain their existing fast path. Normal same-type
scratch/team calls also borrow the sparse numeric factors when the rank bound
certifies full column rank. Other explicit resource calls, wide defaults and
rank/tolerance fallbacks use an R SVD, coefficient promotion and projection in
caller scratch without constructing a dense pseudoinverse or reconstructing the
input. In this R-SVD workspace path, saved R and column permutations are borrowed,
and saved reflectors apply
Q directly to the R singular vectors inside scratch. No independent Q or R copy
is prepared for an R-SVD workspace solve. Blocked Householder calls borrow the team; indexed
simplicial traversal remains sequential. Rank certificates are prepared once
per numeric factor revision and retained through copies. RHS/result ownership
and destination staging allocate separately. The numeric QR
construction workspace below also includes rank-certificate preparation. Input
normalization/snapshot/finite checks and owned factor storage allocate separately.


For auto/supernodal QR, `ordering=:default` and `ordering=:colamd` both select
native COLAMD. For example, `qr(A; algorithm=:supernodal, ordering=:colamd,
dense_row=-1.0, dense_col=0.0, aggressive=false)` retains the resulting order
through copies and numeric updates. Identity/custom ordering for this path
is supported for every QR algorithm.

## Reuse workspace for sparse QR factor construction

```julia
using Colmat, LinearAlgebra, SparseArrays
A = FaerSparseMatrix(sparse([2.0 0; 0 3; 1 0]))
p = FaerSparseSymbolic(A; kind=:qr, algorithm=:supernodal)
team = FaerTeam(2)
s = FaerScratch(factorize, p, A; team)
F = factorize(p, A; scratch=s, team)
G = factorize(p, A; scratch=s, team)
close(team)
@assert F \ ones(3) ≈ Matrix(A) \ ones(3)
close(p)
close(s)
@assert G \ ones(3) ≈ F \ ones(3)
close(F)
close(G)
close(A)
```

`factorize(plan,A;scratch=s)` supplies caller-owned workspace to auto/supernodal
QR numeric construction. `scratch_requirements(factorize,plan,A)` queries the
scalar-dependent requirement and `reserve!(s,factorize,plan,A)` grows capacity
explicitly. Add `team=team` consistently to the query, scratch constructor,
reserve and factorization to use a retained `FaerTeam`. `factorize(plan,A;team)`
allocates temporary scratch; `qr(A;scratch=s,team)` forwards both resources to
numeric construction. The symbolic analysis remains sequential. The numeric matrix must match the plan's stored pattern exactly.
Symbolic analysis, sparse snapshots and owned factor buffers allocate outside
scratch. Returned factors do not borrow the workspace and remain valid after
it or the input plan is closed. Explicit simplicial QR uses the same public
scratch interface with a sequential native numeric kernel.


Explicit simplicial QR also accepts numeric construction scratch. Its native
numeric kernel is sequential, including when a `FaerTeam` is supplied. The query
includes both numeric-kernel work and source-precision rank-bound preparation;
output indices/values, symbolic analysis and input snapshots allocate separately.
Wide QR uses the same construction interface; see the wide example below.

```julia
using Colmat, LinearAlgebra, SparseArrays
A0 = ComplexF32[4 0.1im; 0.2 5; 0.1 0.3]
A = FaerSparseMatrix(sparse(A0); format=:csr)
p = FaerSparseSymbolic(A; kind=:qr, algorithm=:simplicial, ordering=[2,1])
team = FaerTeam(2)
s = FaerScratch(factorize, p, A; team)
F = factorize(p, A; scratch=s, team)
@assert scratch_requirements(factorize,p,A;team) == scratch_requirements(factorize,p,A)
close(s)
close(team)
close(p)
close(A)
@assert F \ ones(ComplexF32,3) ≈ A0 \ ones(ComplexF32,3)
close(F)
```


Wide QR also accepts construction scratch and teams for all three algorithms.
The prefix QR and subsequent one-column Q-adjoint projections reuse the buffer.
Blocked Householder operations use the caller team; indexed simplicial traversal
is sequential. Input normalization, prefix CSC, output triplets/CSC assembly and
owned numeric output allocate separately.

```julia
using Colmat, LinearAlgebra, SparseArrays
A0 = ComplexF64[0 0 4im 0; 0 0 0 5]
A = FaerSparseMatrix(sparse(A0); format=:csr)
p = FaerSparseSymbolic(A; kind=:qr, algorithm=:supernodal, ordering=:identity)
team = FaerTeam(2)
s = FaerScratch(factorize,p,A;team)
F = factorize(p,A;scratch=s,team)
@assert Matrix(F) ≈ A0
close(s)
close(team)
close(p)
close(A)
@assert F \ ones(ComplexF64,2) ≈ pinv(A0)*ones(ComplexF64,2)
close(F)
```


Retained sparse QR factors accept the same construction resources during a
staged numeric update. Query with `scratch_requirements(qr!,F,A;team)`, construct
`FaerScratch(qr!,F,A;team)`, or grow it with `reserve!(s,qr!,F,A;team)`. The input
must have the same stored pattern and scalar type as the retained plan/factor.
The new numeric revision is installed after successful construction; errors
preserve the previous factors, plan and source. Symbolic ordering is retained.
Snapshot and owned-output allocation remain separate; this is a staged update.

```julia
using Colmat, LinearAlgebra, SparseArrays
A0 = [4.0 0.1 0.2; 0.2 5 0.3]
A = FaerSparseMatrix(sparse(A0))
B = FaerSparseMatrix(sparse(2A0); format=:csr)
F = qr(A; algorithm=:supernodal)
team = FaerTeam(2)
s = FaerScratch(qr!,F,B;team)
@assert qr!(F,B;scratch=s,team) === F
@assert Matrix(F) ≈ 2A0
close(s)
close(team)
close(A)
close(B)
@assert F \ ones(2) ≈ pinv(2A0)*ones(2)
close(F)
```


Sparse Cholesky construction and staged updates also accept scratch/team for
all three algorithms. Query with `scratch_requirements(factorize,plan,A;team)`
and `scratch_requirements(cholesky!,F,A;team)`; the corresponding `FaerScratch`
and `reserve!` helpers use the same requirement. The retained plan selects the
triangle and ordering. `check=true` preserves existing factors after a failed
update; `check=false` installs the unsuccessful status, which can be replaced
by a later successful update. Simplicial numeric kernels remain sequential.
Input snapshots/CSC conversion, selected-triangle assembly and owned output
allocate separately from native kernel scratch.

```julia
using Colmat, LinearAlgebra, SparseArrays
A0 = ComplexF64[4 0.1im; -0.1im 5]
A = FaerSparseMatrix(sparse(A0); format=:csr)
p = FaerSparseSymbolic(A;kind=:cholesky,uplo=:U,algorithm=:supernodal,ordering=[2,1])
team = FaerTeam(2)
s = FaerScratch(factorize,p,A;team)
F = factorize(p,A;scratch=s,team)
B = FaerSparseMatrix(sparse(2A0))
reserve!(s,cholesky!,F,B;team)
@assert cholesky!(F,B;scratch=s,team) === F
close(s)
close(team)
close(p)
close(A)
close(B)
@assert F \ ones(ComplexF64,2) ≈ (2A0) \ ones(ComplexF64,2)
close(F)
```


Sparse LU construction and staged updates accept the same resources for all
three algorithms. `scratch_requirements(factorize,plan,A;team)` and
`scratch_requirements(lu!,F,A;team)` query the native numeric kernel;
`FaerScratch`/`reserve!` use the corresponding signature. Symbolic ordering and
COLAMD controls are retained across updates. The explicit simplicial kernel is
sequential. LU numeric factor/permutation buffers are owned output and may
allocate inside the native numeric routine, separately from caller scratch.
Checked singular updates preserve the old factors; `check=false` installs the
unsuccessful status and permits a later successful update.

```julia
using Colmat, LinearAlgebra, SparseArrays
A0 = ComplexF64[0.1 4; 5 0.2im]
A = FaerSparseMatrix(sparse(A0))
p = FaerSparseSymbolic(A;kind=:lu,algorithm=:supernodal,ordering=[2,1])
team = FaerTeam(2)
s = FaerScratch(factorize,p,A;team)
F = factorize(p,A;scratch=s,team)
B = FaerSparseMatrix(sparse(2A0);format=:csr)
reserve!(s,lu!,F,B;team)
@assert lu!(F,B;scratch=s,team) === F
@assert Matrix(F.L)*Matrix(F.U) ≈ (2A0)[Vector(F.p),Vector(F.q)]
close(s)
close(team)
close(p)
close(A)
close(B)
@assert F \ ones(ComplexF64,2) ≈ (2A0) \ ones(ComplexF64,2)
close(F)
```

## Native inverse actions from sparse QR

```julia
using Colmat, LinearAlgebra, SparseArrays
A0 = ComplexF64[2 0; 1im 3; 0 1]
F = qr(FaerSparseMatrix(sparse(A0)); algorithm=:simplicial, ordering=[2, 1])
M = FaerLinearOperator(F)
close(F)
@assert size(M) == (2, 3)
@assert M * ones(ComplexF64, 3) ≈ A0 \ ones(ComplexF64, 3)
@assert adjoint(M) * ones(ComplexF64, 2) ≈ pinv(A0') * ones(ComplexF64, 2)
close(M)
```

The action retains sparse Householder components, R and column permutation;
no dense Q or pseudoinverse is formed. Full-column-rank tall and square factors
produce an action of shape `reverse(size(F))`. Transpose/adjoint apply minimum-norm
inverse actions using saved reflectors and sparse triangular solves. All native
scalar promotions preserve Householder pivot ordering. Explicit simplicial and
auto/supernodal action intermediates use native scratch; caller scratch and team
are supported for all orientations.
Every algorithm reuses saved factors without refactorizing source snapshots.
Rank-deficient factors instead build an independent native dense pseudoinverse
action, with the allocation costs described below.


```julia
using Colmat, LinearAlgebra, SparseArrays
A0=Float32[4 1; 1 3]
F=lu(FaerSparseMatrix(sparse(A0)); algorithm=:supernodal)
close(F.source)
b=ComplexF64[1+im,2-im]
@assert isapprox(A0*(F\b), b; rtol=1e-5)
M=FaerLinearOperator(F)
close(F)
@assert isapprox(A0*(M*b), b; rtol=1e-5)
close(M)
```


```julia
using Colmat, LinearAlgebra, SparseArrays
A0=ComplexF32[2 0; im 3; 0 1]
F=qr(FaerSparseMatrix(sparse(A0)); algorithm=:supernodal)
close(F.source)
b=ComplexF64[1+im,2,3-im]
@assert isapprox(Matrix(F\reshape(b,:,1)), reshape(ComplexF64.(A0)\b,:,1); rtol=1e-5)
M=FaerLinearOperator(F)
close(F)
@assert isapprox(M*b, ComplexF64.(A0)\b; rtol=1e-5)
close(M)
```


```julia
using Colmat, LinearAlgebra, SparseArrays
A0=ComplexF64[4 0; im 3; 0 1]
F=qr(FaerSparseMatrix(sparse(A0)); algorithm=:supernodal, ordering=[2,1])
@assert Vector(F.pcol)==[2,1]
@assert Matrix(F.Q)*Matrix(F.R)≈A0[:,[2,1]]
@assert Matrix(F)≈A0
close(F)
```

## Full orthogonal bases from sparse QR

`F.Q` and `F.Qthin` return the independent thin basis; `F.Qfull` returns an
independent square orthogonal/unitary basis for every sparse QR algorithm.
The complement is formed from saved native Householder factors, including
structurally missing pivots. No source snapshot is refactorized.

```julia
using Colmat, LinearAlgebra, SparseArrays
A0=ComplexF64[2 0; im 3; 0 1; 0 0]
F=qr(FaerSparseMatrix(sparse(A0)); algorithm=:supernodal)
Q=Matrix(F.Qfull)
@assert Q'*Q≈Matrix{ComplexF64}(I,4,4)
@assert Q[:,1:2]≈Matrix(F.Qthin)
@assert Q*vcat(Matrix(F.R),zeros(ComplexF64,2,2))≈A0[:,Vector(F.pcol)]
close(F)
@assert Q'*Q≈Matrix{ComplexF64}(I,4,4)
```

## Lazy sparse QR Q operator

`F.Qoperator` corresponds to SPQR's lazy `F.Q`. It returns an independent m×m
`FaerLinearOperator` that applies the saved native Householder reflectors for
every algorithm (auto, simplicial, supernodal and wide) and all four scalar
types. Dense Q is never formed unless you explicitly call `Matrix(Qop)`, which
matches `Matrix(F.Qfull)`. This differs from SPQR: there `Matrix(F.Q)` returns
the thin m×min(m,n) basis, but `Matrix(F.Qoperator)` is square. Use `F.Qthin`
when porting code that expects the thin basis. `Qop*B`, `Qop'*B`, `transpose(Qop)*B`, `conj(Qop)*B`
and `mul!(X,Qop,B)` accept vectors, matrices, views and mixed-type right-hand
sides. `mul!(X,Qop,B; scratch,team)` runs on caller-owned scratch and a team;
use `scratch_requirements(mul!,X,Qop,B; team)` to query the size. With
sufficient scratch, the native application does not allocate. Constructing the
operator copies the saved reflectors, so the operator stays usable after
`close(F)`.
Colmat stores Q in the original row order, so `F.prow` is the identity
permutation and `F.Q*F.R ≈ A[F.prow,F.pcol]` holds as in SPQR.
`F.R` is assembled directly as a sorted CSC matrix from the saved factor. No
triplet list or extra sort pass is used.

The following faer sparse QR controls are not exposed yet:
- an arbitrary `supernodal_flop_ratio_threshold` (only `algorithm=:auto`,
  `:simplicial` and `:supernodal` are available);
- the supernodal `relax` amalgamation thresholds;
- the numeric `blocking_threshold` for dense front QR.

faer's symbolic QR analysis takes neither a `Par` nor a caller stack. It always
runs sequentially and allocates its own workspace.

```julia
using Colmat, LinearAlgebra, SparseArrays
A0=ComplexF64[2 0; im 3; 0 1; 0 0]
F=qr(FaerSparseMatrix(sparse(A0)); algorithm=:supernodal)
Qop=F.Qoperator
@assert size(Qop)==(4,4)
@assert Vector(F.prow)==1:4
R=vcat(Matrix(F.R),zeros(ComplexF64,2,2))
@assert Matrix(Qop*R)≈A0[Vector(F.prow),Vector(F.pcol)]
b=ComplexF64[1,2,3,4]
x=zeros(ComplexF64,4)
work=FaerScratch(mul!,x,Qop',b)
mul!(x,Qop',b;scratch=work)
@assert x≈Matrix(F.Qfull)'*b
close(F)
@assert Vector(Qop*x)≈b
close(work)
close(Qop)
```

## Rank and tolerance-aware solves from saved sparse QR

`svdvals(F)`, `rank(F;atol,rtol)` and `pinv(F;atol,rtol)` use saved sparse QR
factors. Singular values are computed by native dense SVD of the n×n R factor.
The rank threshold is `max(atol,rtol*maximum(svdvals(F)))`; the default relative
tolerance is `min(size(F)...)*eps(real(eltype(F)))` when atol is zero.
`pinv(F)` also materializes thin Q and returns an independent native dense result.
Three-argument `ldiv!(X,F,B;atol,rtol)` computes the corresponding minimum-norm
least-squares solution, including transpose/adjoint factors and mixed RHS types.
This explicit tolerance path accepts caller-owned scratch and a team as described in the workspace sections on this page.
Ordinary `F \ B` also supports rank-deficient factors; wide sparse QR uses the native construction described below.

```julia
using Colmat,LinearAlgebra,SparseArrays
A0=[1.0 2;0 0;0 0]
F=qr(FaerSparseMatrix(sparse(A0));algorithm=:supernodal)
close(F.source)
@assert rank(F;rtol=1e-8)==1
P=pinv(F;rtol=1e-8)
@assert Matrix(P)≈pinv(A0;rtol=1e-8)
X=zeros(2,1)
ldiv!(X,F,ones(3,1);rtol=1e-8)
@assert X≈pinv(A0)*ones(3,1)
close(F)
@assert Matrix(P)≈pinv(A0)
close(P)
```

## Native QR for originally wide sparse inputs

Wide inputs use a native square-prefix QR and direct saved Householder adjoint
applications to project the remaining columns. Column projection processes one
column at a time without materializing a dense Q or the complete tail. `F.Q` and
`F.Qfull` are m×m; `F.R` is m×n. The full input column permutation is retained
by symbolic reuse and numeric updates.
Direct normal and transpose/adjoint solves use native dense SVD of saved R
for minimum-norm least squares, including rank-deficient wide inputs. Each direct
solve materializes m×n dense R and allocates SVD workspace; caller-owned scratch and teams are available through the workspace interface described on this page. A `FaerLinearOperator(F)` instead
caches the pseudoinverse of R during construction and reuses caller scratch/team
for subsequent applications. Scalar promotion preserves the cached rank cutoff.
The default cutoff matches rank/pinv:
`min(size(F)...)*eps(real(eltype(F)))*maximum(svdvals(F))`.

```julia
using Colmat,LinearAlgebra,SparseArrays
A0=ComplexF64[0 0 2 im;0 0 0 3]
F=qr(FaerSparseMatrix(sparse(A0));algorithm=:supernodal,ordering=[4,2,1,3])
@assert Matrix(F.Q)*Matrix(F.R)≈A0[:,Vector(F.pcol)]
close(F.source)
b=ComplexF64[1+im,2]
@assert F\b≈pinv(A0)*b
G=copy(F)
close(F)
@assert Matrix(G)≈A0
@assert G\b≈pinv(A0)*b
close(G)
```

## Rank-aware sparse QR backslash

For all sparse QR algorithms, ordinary `F \ B`, transpose/adjoint solves,
and `FaerLinearOperator(F)` apply the default numerical-rank cutoff to saved
factors, including zero and rank-deficient inputs. A sparse norm bound first
certifies well-conditioned tall/square R factors; inconclusive cases use native
SVD of dense R and materialize thin Q. This fallback allocates and can be costly
for large sparse factors. It does not read the source snapshot.
Caller-owned scratch supports numerically rank-deficient factors through the R-SVD workspace path described on this page.
Explicit tolerances remain available through `ldiv!(X,F,B;atol,rtol)` and `pinv`.

```julia
using Colmat, SparseArrays, LinearAlgebra
A = [1.0 2; 2 4; 0 0]
F = qr(FaerSparseMatrix(sparse(A)))
b = ones(3)
@assert isapprox(Vector(F \ b), pinv(A)*b; atol=1e-12)
close(F)
```

## Sparse QR minimum-norm solves with scratch and a team

```julia
using Colmat,LinearAlgebra,SparseArrays
A0 = Float32[1 2 1; 2 4 2]
a = FaerSparseMatrix(sparse(A0))
F = qr(a;algorithm=:simplicial,ordering=[3,1,2])
team = FaerTeam(2)
B = ones(ComplexF64,2,2)
s = FaerScratch(ldiv!,F,B;team,rtol=1e-5)
close(a)
close(F.source)
close(F.plan)
X = zeros(ComplexF64,3,2)
ldiv!(X,F,B;scratch=s,team,rtol=1e-5)
@assert isapprox(X,pinv(ComplexF64.(A0);rtol=1e-5)*B;rtol=1e-4)
C = ones(ComplexF64,2,3)
reserve!(s,rdiv!,C,F;team,rtol=1e-5)
Y = zeros(ComplexF64,2,2)
rdiv!(Y,C,F;scratch=s,team,rtol=1e-5)
@assert isapprox(Y,C*pinv(ComplexF64.(A0);rtol=1e-5);rtol=1e-4)
close(s)
close(team)
close(F)
```


The rank certificate used to select the sparse QR solve kernel is refreshed by
`qr!` and numeric `factorize` calls. It belongs to the retained numeric revision;
closing or changing a source matrix does not change it. Reusing scratch across a
numeric update can require more capacity when a deficient factor switches to the
R-SVD path; query again and call `reserve!` before solving.


For auto/supernodal QR, `scratch_requirements(factorize,plan,A)` covers both the
numeric QR kernel and the source-precision rank-bound accumulators. These stages
reuse one buffer; R is visited directly without a temporary sparse export.
Construction validates current native matrix and plan metadata before reading
snapshot values. Symbolic analysis, input conversion/snapshot/finite checks and
owned numeric output remain separate allocations, so this is not a whole-API
allocation-free construction interface. Explicit simplicial QR shares this interface with a sequential native kernel.
Wide construction uses prefix QR and one-column saved-reflector projections
in the supplied buffer, with the same team for both stages. Sparse output
assembly and input normalization still allocate separately.
