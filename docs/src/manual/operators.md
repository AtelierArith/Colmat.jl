# 線形作用素・反復法

行列を明示しない作用素、前処理、CG・BiCGSTAB・LSMR を使います。

## Native linear operators and conjugate gradients

`FaerLinearOperator` owns a snapshot of a dense, CSC/CSR sparse, diagonal, or
identity operator. Sparse coefficients stay sparse during multiplication and CG.
A preconditioner applies an approximate inverse:

```julia
using Colmat, LinearAlgebra, SparseArrays
A = spdiagm(-1 => fill(-1.0, 99), 0 => fill(4.0, 100), 1 => fill(-1.0, 99))
op = FaerLinearOperator(A)
b = ones(100)
result = cg(op, b; M = diagonal_preconditioner(diag(A)))
@assert issuccess(result)
@assert A * result.x ≈ b
@assert op * b ≈ A * b
```

CG requires a Hermitian positive definite operator and preconditioner action.
`cg!` uses its destination as the initial guess. Failed checked solves throw
`FaerCgError`, whose `result` contains the last iterate and residual diagnostics;
`check=false` returns that result directly. Convergence includes an independently
recomputed true residual check. Iteration callbacks and residual histories are
available as described on this page, including Julia-defined operator actions.

For a nonsymmetric square matrix, use `bicgstab` or `bicgstab!`. `M` and `N`
apply left and right approximate inverses, respectively:

```julia
using Colmat, LinearAlgebra, SparseArrays
A = sparse([5.0 2 0; -1 4 1; 0 -1 3])
b = [1.0, 2, 3]
result = bicgstab(A, b; N = diagonal_preconditioner(diag(A)))
@assert issuccess(result)
@assert A * result.x ≈ b
x = zeros(3)
@assert bicgstab!(x, A, b).x === x
@assert A * x ≈ b
```

`FaerBicgstabResult` reports the true and estimated residuals, iteration count
and termination reason. Checked failures throw `FaerBicgstabError` retaining
the result; `check=false` returns the last iterate. Multiple right-hand sides
use faer's block algorithm. Unlike CG, BiCGSTAB does not require Hermitian
positive definite coefficients.

For rectangular least squares, `lsmr` and `lsmr!` use the native block LSMR
algorithm without forming normal equations:

```julia
using Colmat, LinearAlgebra, SparseArrays
A = sparse([1.0 0; 0 2; 1 1; 2 -1])
b = [1.0, 2, 4, -1]
result = lsmr(A, b)
@assert issuccess(result)
@assert result.x ≈ A \ b
@assert result.normal_abs_residual ≈ norm(A' * (b - A * result.x)) atol=1e-12
x = zeros(size(A, 2))
@assert lsmr!(x, A, b).x === x
@assert x ≈ result.x
```

`M` supplies a nonsingular right preconditioner action. Convergence uses
`norm(M' * A' * (b-A*x)) ≤ max(atol, rtol * norm(M' * A' * b))`, taking `M=I`
when omitted. `FaerLsmrResult` separates ordinary residuals from these normal
residuals; an inconsistent least squares problem can converge with a nonzero
ordinary residual. Checked failures throw `FaerLsmrError` retaining the last
iterate. A nonzero initial guess can retain nullspace components, and right
preconditioning changes the minimum-norm metric.

Absolute and relative LSMR tolerances are passed independently to faer. The
absolute bound can accept a nonzero initial normal residual with no updates:

```julia
using Colmat, LinearAlgebra
A = Matrix{Float64}(I, 2, 2)
b = [1.0, 0.0]
result = lsmr(A, b; atol=1.0, rtol=0, maxiter=0)
@assert issuccess(result) && result.iterations == 0
@assert iszero(result.x)
@assert result.normal_abs_residual == result.estimated_abs_residual == 1.0
@assert result.normal_rel_residual == result.estimated_rel_residual == 1.0
```

## Reuse operator application scratch and teams

`mul!(X, op, B[, alpha, beta]; scratch, team)` and
`*(op, B; scratch, team)` accept caller-owned resources for native LinOp
application, including conjugate/transpose/adjoint and right-operator products.
Query the exact working scalar and orientation with `scratch_requirements`;
alpha/beta and output type participate in a `mul!` query.

```julia
using Colmat, LinearAlgebra

a = [4.0 1; 1 3]
A = FaerMatrix(a)
F = lu(A)
op = FaerLinearOperator(F)
close(F)
close(A)
b = [1.0 3; 2 4]
x = zeros(2, 2)
FaerTeam(2) do team
    work = FaerScratch(mul!, x, op, b; team)
    req = scratch_requirements(mul!, x, op, b; team)
    @assert scratch_capacity(work) == req
    mul!(x, op, b; scratch=work, team)
    @assert a * x ≈ b
    first = copy(x)
    reserve!(work, mul!, x, op, b, 2.0, 1.0; team)
    mul!(x, op, b, 2.0, 1.0; scratch=work, team)
    @assert x ≈ 3first
    close(work)
end
close(op)
```

Kernel scratch covers the native LinOp query, including composition and saved
factor inverse-action intermediates where implemented. Result staging, Julia
scaling/conversion buffers and promoted operator snapshots allocate separately.
Prepared auto/supernodal and wide sparse-QR inverse actions use this buffer and
propagate the team to their kernels. Wide inverse construction caches the dense
pseudoinverse of R once; construction and direct factor solves have separate
allocation costs. Invalid/closed resources and insufficient
capacity/alignment reject before destination writeback. Shared use is serialized
by the owner lock.

## Compose native matrix-free operators

```julia
using Colmat, LinearAlgebra, SparseArrays
A0 = [3.0 0; 1 2; 0 1]
A = FaerLinearOperator(sparse(A0))
N = adjoint(A) * A
close(A)
b = ones(2)
result = cg(N, b)
@assert result.x ≈ (A0' * A0) \ b
@assert Matrix(N) ≈ A0' * A0
close(N)
```

Operator multiplication retains independent copies of both operands and applies
them in sequence without materializing their coefficient product. Composition
supports nested products, conjugation, transpose and adjoint; intermediate RHS
storage comes from native scratch. Explicit `Matrix(op)` materializes the result.
A composition has no single `.coefficients` representation. Mixed native scalar types are promoted recursively through the operator tree.
Dense, CSC/CSR, diagonal and identity leaves preserve their representations;
composition never forms a dense coefficient product during promotion.

## Use retained LU factors as a native preconditioner

```julia
using Colmat, LinearAlgebra
A0 = [4.0 1; 1 3]
A = FaerMatrix(A0)
F = lu(A)
M = FaerLinearOperator(F)
close(F)
b = ones(2)
@assert cg(A, b; M=M).x ≈ A0 \ b
@assert M * b ≈ A0 \ b
close(M)
close(A)
```

`FaerLinearOperator(F)` for a square nonsingular native partial-pivot LU owns
copies of L/U and the pivot permutation. It applies the inverse action using
native triangular solves and scratch, without forming an inverse or repeating
factorization. Normal/conjugate/transpose/adjoint application, independent copies,
composition and mixed native scalar promotion reuse those saved factors. There
is no single `.coefficients` matrix. CG requires a positive-definite inverse
action; general LU actions can be used with BiCGSTAB. Other retained decompositions and sparse factor actions are described on this page.

## Use retained Cholesky factors as a preconditioner

```julia
using Colmat, LinearAlgebra
A0 = ComplexF64[4 1im; -1im 3]
A = FaerMatrix(A0)
F = cholesky(Hermitian(A, :U))
M = FaerLinearOperator(F)
close(F)
b = ones(ComplexF64, 2)
@assert cg(A, b; M=M).x ≈ A0 \ b
@assert transpose(M) * b ≈ transpose(A0) \ b
close(M)
close(A)
```

A native Cholesky preconditioner owns independent triangular factors and applies
them with lower/upper triangular solves. It supports the selected input triangle,
all four native scalar types, mixed RHS promotion, copy/composition and
transpose/adjoint without inverse construction or refactorization. Failed
Cholesky factors are rejected with `PosDefException`. Other dense and sparse factor actions are described on this page.

## Retained full-pivot LU inverse actions

```julia
using Colmat, LinearAlgebra
A0 = [0.0 2; 3 100]
F = full_piv_lu(FaerMatrix(A0))
M = FaerLinearOperator(F)
close(F)
b = ones(2)
@assert M * b ≈ A0 \ b
@assert transpose(M) * b ≈ transpose(A0) \ b
N = M * FaerLinearOperator(A0)
@assert Matrix(N) ≈ Matrix{Float64}(I, 2, 2)
close(M)
close(N)
```

Full-pivot LU inverse actions retain both row and column permutations with their
triangular factors. Normal, transpose and adjoint application use native
full-pivot solves and the corresponding scratch requirement. Mixed scalar
promotion copies factor values and preserves both permutations, without
refactorization. Singular and rectangular factors are rejected.

## Retained QR inverse-action preconditioners

```julia
using Colmat, LinearAlgebra
A0 = ComplexF64[4 1im; 1 3]
F = qr(FaerMatrix(A0), ColumnNorm())
M = FaerLinearOperator(F)
close(F)
b = ones(ComplexF64, 2)
@assert M * b ≈ A0 \ b
@assert transpose(M) * b ≈ transpose(A0) \ b
@assert adjoint(M) * b ≈ adjoint(A0) \ b
close(M)
```

Square native QR and column-pivoted QR factors can provide inverse actions.
Their compact Householder basis/coefficients, R and column permutation are copied
into an independent native owner. No explicit Q, inverse or refactorization is
needed. Native QR solve kernels consume operator scratch, and mixed scalar
promotion converts the saved components. Empty square factors are supported.
Without a nonzero atol or explicit rtol, square QR keeps this nonsingular
inverse contract and rejects zero R pivots.

Rectangular QR instead provides a minimum-norm action of shape `reverse(size(F))`,
including rank-deficient inputs. `atol`/`rtol` select singular values of saved R
using `max(atol,rtol*maximum(svdvals(R)))`. The default rtol is
`min(size(F)...)*eps(real(eltype(F)))` when atol is zero, and zero otherwise.
Nonzero atol or explicit rtol also select this truncated action for square QR.
Construction computes an SVD of R once and stores its singular factors, compact
Householder factors and column permutation. Applications form neither dense Q
nor a pseudoinverse of A and reuse native scratch/team. Promotion converts the
cache without recomputing rank; construction and result staging allocate.

```julia
using Colmat, LinearAlgebra
A0=ComplexF64[1 im 2;0 2 1]
F=qr(FaerMatrix(A0),ColumnNorm())
M=FaerLinearOperator(F)
close(F)
B=ones(ComplexF64,2,2)
FaerTeam(2) do team
    X=zeros(ComplexF64,3,2)
    scratch=FaerScratch(mul!,X,M,B;team)
    @assert mul!(X,M,B;scratch,team)===X
    @assert X≈pinv(A0)*B
    close(scratch)
end
@assert Matrix(M')≈pinv(A0)'
close(M)
# Explicit rtol enables a truncated action for singular square QR too.
F=qr(FaerMatrix([1.0 0;0 0]))
M=FaerLinearOperator(F;rtol=1e-12)
@assert Matrix(M)≈[1.0 0;0 0]
close(M)
close(F)
```

## Retained LDLT preconditioners for indefinite systems

```julia
using Colmat, LinearAlgebra
A0 = ComplexF64[4 1im; -1im -3]
A = FaerMatrix(A0)
F = ldlt(Hermitian(A, :L))
M = FaerLinearOperator(F)
close(F)
b = ones(ComplexF64, 2)
@assert M * b ≈ A0 \ b
@assert transpose(M) * b ≈ transpose(A0) \ b
@assert bicgstab(A, b; M=M).x ≈ A0 \ b
close(M)
close(A)
```

The native LDLT inverse action owns L and D and applies the low-level saved-factor
solve without an inverse or refactorization. Both selected input triangles,
normal/conjugate/transpose/adjoint, copy/composition and mixed native scalar
promotion are supported. Zero diagonal pivots are rejected. Indefinite inverse
actions are suitable for general solvers such as BiCGSTAB; they do not satisfy
CG's positive-definiteness requirement. Sparse factor actions are described on this page.

## Retained Bunch–Kaufman preconditioners

```julia
using Colmat, LinearAlgebra
A0 = ComplexF64[0 1im; -1im 0]
F = bunchkaufman(Hermitian(FaerMatrix(A0), :U))
M = FaerLinearOperator(F)
close(F)
b = ones(ComplexF64, 2)
@assert M * b ≈ A0 \ b
@assert transpose(M) * b ≈ transpose(A0) \ b
@assert adjoint(M) * b ≈ adjoint(A0) \ b
close(M)
```

Block-pivoted LDLT inverse actions retain L, the block diagonal's diagonal and
subdiagonal, and the pivot permutation. Native Bunch–Kaufman solves consume
operator scratch; valid 2×2 blocks with zero diagonal entries are preserved.
Mixed promotion copies both block components and the permutation, without an
inverse or refactorization. These indefinite actions can be used with BiCGSTAB;
CG's positive-definiteness requirement still applies. Sparse factor actions are described on this page.

## Matrix-free saved SVD/EVD pseudoinverse actions

```julia
using Colmat, LinearAlgebra
A0 = [3.0 0; 0 1e-6; 0 0]
F = svd(FaerMatrix(A0); native=true)
M = FaerLinearOperator(F; rtol=1e-3)
close(F)
@assert size(M) == (2, 3)
@assert M * ones(3) ≈ [1/3, 0]
@assert Matrix(M) ≈ pinv(A0; rtol=1e-3)
close(M)
```

`FaerLinearOperator(F;atol,rtol)` for a saved native SVD or self-adjoint EVD
owns spectral components and applies a truncated inverse action using native
projection/scaling kernels and scratch. The operator shape is `reverse(size(F))`;
rectangular and rank-truncated actions are supported. Thresholds follow `pinv(F)`
and are chosen in the original factor precision before any RHS promotion.
Negative eigenvalues retain their sign. Copy/composition, mixed native scalar
promotion, conjugation and transpose/adjoint do not recompute the decomposition
or materialize a pseudoinverse. Iterative preconditioning requires a suitable
square nonsingular action; CG additionally requires positive definiteness.

## Julia-defined operator and preconditioner actions

`FaerLinearOperator(apply!,adjoint_apply!,m,n;T)` supplies native LinOp/BiLinOp
traits with Julia functions. Each function receives owned Julia matrices `(Y,X)`
and must overwrite all of Y with the forward or adjoint product. The functions
must accept batched RHS and promoted native scalar types. Captured state is held
by reference and shared by copies, transforms and compositions. Callback arrays
may be retained after a call; they never borrow native temporary storage.

```julia
using Colmat, LinearAlgebra
A0=[4.0 1;1 3]
apply! = (Y,X)->mul!(Y,A0,X)
op=FaerLinearOperator(apply!,apply!,2,2)
jacobi! = (Y,X)->copyto!(Y,X ./ diag(A0))
M=FaerLinearOperator(jacobi!,jacobi!,2,2)
b=[1.0,2.0]
@assert Vector(op*b)≈A0*b
result=cg(op,b;M,history=true)
@assert result.converged
@assert result.x≈A0\b
@assert Matrix(op)≈A0
close(M)
close(op)
```

CG/BiCGSTAB/LSMR, preconditioner application and partial spectra still run their
native algorithms. They invoke Julia actions synchronously on the calling Julia
thread. The task is temporarily pinned for this call. Same-task nested Colmat
calls are supported; calls from other tasks serialize on the owner lock until
the callback returns. User exceptions return through the native boundary before being rethrown;
failed actions preserve staged destinations. Outputs must be finite and fully
written. Empty-domain/range and zero-column products are treated as zero maps.
Native composition/conjugation intermediates can reuse `scratch=`/`team=` as
for other operators; Julia callback buffers and result staging allocate separately.

`buffers=:reuse` keeps one `(Y, X)` pair per scalar type, orientation and shape
inside the operator and reuses it for every later application, so an iterative
solve no longer allocates two Julia arrays per operator application. The
callback must then not retain `Y` or `X` after it returns; a recursive
application of the same operator from inside its callback receives fresh
arrays. The default `buffers=:owned` keeps passing independent arrays.

```julia
using Colmat, LinearAlgebra
d = collect(range(2.0, 3.0; length=1000))
apply! = (Y, X) -> (Y .= d .* X; Y)
op = FaerLinearOperator(apply!, apply!, 1000, 1000; buffers=:reuse)
r = cg(op, FaerVector(ones(1000)))
@assert Vector(r.x) ≈ 1 ./ d
close(op)
```


## Iterative histories and stopping callbacks

`cg`, `bicgstab`, `lsmr` and their in-place forms accept `history=true` and
`callback=state -> ...`. Each callback receives an independent Julia iterate
`state.x`, the completed update number `state.iteration`, true primal residual
norms, and (for LSMR) true preconditioned normal residual norms. Return `nothing`
or `true` to continue and `false` to stop. A requested stop returns
`reason=:stopped`, `converged=false`, including with `check=true`.

```julia
using Colmat, LinearAlgebra
A = Diagonal(collect(1.0:6.0))
b = ones(6)
result = cg(A, b; history=true, callback=state -> state.iteration < 2)
@assert result.reason == :stopped
@assert result.iterations == length(result.history) == 2
@assert isapprox(last(result.history).abs_residual, norm(b-A*result.x); atol=1e-12)
@assert !result.converged
```

History retains residual records without full iterates. Observation allocates
copies and makes extra native operator applications to recompute residuals.
With both options disabled, these copies and extra applications are absent.
Callbacks run synchronously on the calling Julia task; retained iterate copies
stay valid after the solver returns. Initial convergence or zero update limits
produce no callback/history entries. Callback exceptions propagate after safely
returning through Rust; in-place destinations remain unchanged. A deliberate
stop copies the stopped iterate into an in-place destination.

Julia-defined operator/preconditioner actions are available through the constructor
above. The three iterative solvers and their in-place forms also accept
`scratch` and `team`, with `scratch_requirements`, `FaerScratch` and `reserve!`.
Queries account for the promoted scalar type, RHS columns, preconditioners and
actual team shape without applying operators or invoking callbacks. The buffer
covers the native iteration and independently recomputed final residuals.
Result staging, operator promotion and Julia callback/history storage allocate
separately. Short or closed resources fail before updating destinations.

```julia
using Colmat, LinearAlgebra
A = Diagonal(collect(4.0:9.0))
b = ones(6)
team = FaerTeam(2)
work = FaerScratch(cg, A, b; team)
r = cg(A, b; scratch=work, team, history=true)
@assert r.converged && r.x ≈ A\b
x = zeros(6)
reserve!(work, lsmr!, x, A, b; team)
r2 = lsmr!(x, A, b; scratch=work, team)
@assert r2.x === x && x ≈ A\b
close(work)
close(team)
```

LSMR can falsely report native convergence on extreme uniform scales
([faer #245](https://github.com/tensor4all/faer/issues/245)). Colmat's true-residual
check rejects those results. Observing a nonfinite failed iterate preserves the
same failure diagnostics; the history bridge accepts its nonfinite values.
