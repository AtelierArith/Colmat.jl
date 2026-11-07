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

## SVD algorithms and tuning

```julia
using Colmat, LinearAlgebra
A = FaerMatrix([sin(i + 3j) for i in 1:40, j in 1:30])
F = svd(A; alg=LinearAlgebra.QRIteration())
@assert F.U * Diagonal(F.S) * F.Vt ≈ Matrix(A)
@assert svdvals(A; alg=LinearAlgebra.DivideAndConquer()) ≈ F.S
G = with_svd_tuning(; recursion_threshold=8, qr_ratio_threshold=1.5) do
    svd(A)
end
@assert G.S ≈ F.S
close(A)
```

`alg=DivideAndConquer()` (the default) uses faer's driver, which runs the
bidiagonal implicit QR algorithm below `recursion_threshold` (128) and divide
and conquer above it; `alg=QRIteration()` uses the implicit QR algorithm at
every size. `with_svd_tuning(f; recursion_threshold, qr_ratio_threshold,
bidiag_par_threshold, qr_blocking_threshold)` overrides faer's `SvdParams` (the
last is the blocking threshold of the QR that precedes bidiagonalization for
tall inputs; faer's QR workspace does not depend on it) for every SVD, scratch query,
`FaerScratch` and `reserve!` in the scope (unspecified values keep the enclosing
scope's), and calls without `scratch`/`team` then use the parameterized
workspace driver. The scope holds Colmat's native-call lock.
## Column-pivoted QR rank and truncated solves

```julia
using Colmat, LinearAlgebra
a = [1.0 2 3; 2 4 6; 1 0 1; 0 1 1]
a[:, 2] = 2a[:, 1]                       # rank two
F = qr(FaerMatrix(a), ColumnNorm())
@assert rank(F) == 2
@assert rank(F; atol=1e3) == 0
B = reshape([1.0, 2, 3, 4], 4, 1)
B, r = ldiv!(F, B, 1e-10)               # minimum-norm, rank-truncated
@assert r == 2
@assert B[1:3, :] ≈ pinv(a) * [1.0, 2, 3, 4]
X = zeros(3, 1)
ldiv!(X, F, [1.0 2 3 4]'; rtol=1e-10)   # keyword form, same kernel
@assert X == B[1:3, :]
x = ldiv!(zeros(3), F, [1.0, 2, 3, 4]; atol=1e3)
@assert iszero(x)                       # rank zero writes zeros
close(F)
```

`rank(F; atol, rtol)` follows `rank(::QRPivoted)`: the first stored
`|R[i,i]| <= max(atol, rtol*|R[1,1]|)`, with the same default `rtol`. It reads
the native R diagonal without reconstruction. `ldiv!(F, B, rcond; team)`
matches `ldiv!(::QRPivoted, B, rcond)`. `B` needs at least `max(m,n)` rows,
its first `n` rows receive the minimum-norm solution, and the rank is
returned. The rank is decided with `rtol=rcond`, whereas LAPACK `xgelsy`
estimates it incrementally, so the two can differ for pivots near the
threshold. The leading rank rows of R are completed orthogonally on the
native side. faer's `ColPivQr` has no rank or tolerance API of its own.

`ldiv!(X, F, B; atol, rtol, team)` (vector or matrix `B` with `m` rows, `X` of
size `(n, size(B, 2))`) and the square `ldiv!(F, B; atol, rtol, team)` truncate to
`rank(F; atol, rtol)` and run the same kernel, so `atol=0, rtol=rcond` reproduces
the positional result exactly. Defaults follow `pinv` and the sparse-QR
minimum-norm solve: `atol=nothing` is zero and `rtol=nothing` is
`min(m,n)*eps(real(T))` when `atol` is zero. The rank uses the source precision;
a real factor solves a complex `B` (both forms) by parts. `X` may alias `B`, and
validation errors, including a non-finite `B`, leave the destination unchanged.
Without `atol`/`rtol` the ordinary dense solve runs. Tolerances are rejected
with `scratch=` and on transposed or adjoint factors.
## Determinant and trace

```julia
using Colmat, LinearAlgebra
A = FaerMatrix([4.0 1 2; 1 3 0; 2 0 5])
@assert det(A) ≈ 43
@assert det(view(A, 3:-1:1, :)) ≈ -43        # borrowed view, no copy
team = FaerTeam(2)
@assert det(A'; team) ≈ 43
close(team)
@assert det(LowerTriangular(A)) == 60       # product of the stored diagonal
@assert logabsdet(FaerDiagonal([2.0, -3.0])) == (log(6.0), -1.0)
@assert tr(view(A, 1:2, 1:2)) == 7
```

`det` of a native matrix, range view, `transpose` or `adjoint` calls faer's
`determinant` on the borrowed view; `team=` runs it on a caller-owned
`FaerTeam`. Triangular wrappers and `FaerDiagonal` use only the stored
diagonal for `det`, `tr` and `logabsdet` (a zero entry gives `(-Inf, 0)`).
Non-square operands throw `DimensionMismatch`, empty operands give a
determinant of one, and a real negative determinant makes `logdet` throw
`DomainError`.

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
v = FaerVector([1.0, 1.0])
@assert Vector(L * v) == [2.0, 7.0]           # a native FaerVector
E = FaerMatrix(fill(-1.0, 2, 2))
structured_mul!(E, L, FaerMatrix([1.0 0; 0 1]); dest=:strict_lower)
@assert Matrix(E) == [-1.0 -1; 3 -1]      # only the strict lower block is written
```

Lower/upper and unit-triangular wrappers use faer's `matmul::triangular`.
`FaerDiagonal` and Julia `Diagonal` use native diagonal products. These
methods return a `FaerMatrix`, including triangular×triangular products, and a
triangular wrapper times a `FaerVector` returns a `FaerVector`. Operands are
borrowed rather than snapshotted: native parents, ranges and their
`transpose`/`adjoint` through view descriptors, and strided Julia parents of
the product type directly from their buffers (other Julia inputs and scalar
conversions are copied once). `mul!` into a native matrix, range view or
`FaerVector` writes the destination in place with faer's accumulation, so no
temporary result is allocated; an operand sharing the destination's storage is
snapshotted first. A zero `β` does not read the destination and a zero `α`
does not read the operands. When the promoted type is wider than the
destination, the product is computed out of place and converted.
`structured_mul!(C, A, B, α, β; dest)` additionally selects the destination
block (`:full`, `:lower`, `:upper`, `:strict_lower`, `:strict_upper`) that is
computed and written. Unused triangles and implicit unit diagonals are
ignored, including nonfinite values stored there.

Structured wrappers mixed with Julia arrays follow the same rule as other
mixed products: a native operand gives a native result. A triangular wrapper
of a `FaerMatrix`, a range view or their `transpose`/`adjoint` (so
`UpperTriangular(A)'` too) times a strided Julia matrix, its lazy
transpose/adjoint, a `faerview` or a Julia vector (on either side for
matrices) borrows the Julia operand and returns a `FaerMatrix`/`FaerVector`.
`mul!` with these operands, and with a Julia-parent wrapper or a Julia
`Diagonal` and a native operand, writes a strided Julia destination (or a
`faerview`) in place through its buffer; operands sharing the destination's
memory are snapshotted first. `Symmetric`/`Hermitian` wrappers (native or
strided Julia parents) with native operands, Julia operands of a native
wrapper, triangular wrappers or each other are expanded once, natively, from
the stored triangle (a `Hermitian` diagonal keeps its real part) because faer
has no self-adjoint product kernel; the product then takes the dense gemm path.
`Bidiagonal`, `Tridiagonal` and `SymTridiagonal` with a native matrix copy the
native operand to Julia once and use LinearAlgebra's O(n²) banded kernel,
returning a `FaerMatrix` (expanding the band for faer's gemm would make the
product O(n³)). `FaerMatrix ± Diagonal` (and view operands) passes only the
diagonal's n entries to Rust. Without these routes LinearAlgebra's generic
kernels read the native operand element by element (hundreds of MB at n=64).

```julia
using Colmat, LinearAlgebra
A = FaerMatrix([2.0 1; 3 4])
J = [1.0 2; 3 4]
@assert UpperTriangular(A) * J isa FaerMatrix
@assert Matrix(Symmetric(A) * J) ≈ Symmetric([2.0 1; 3 4]) * J
C = zeros(2, 2)
mul!(C, UpperTriangular(A), J)              # written in C's buffer
@assert C ≈ UpperTriangular([2.0 1; 3 4]) * J
@assert Matrix(A + Diagonal([1.0, 1.0])) == [3.0 1; 3 5]
```

`FaerDiagonal` or Julia `Diagonal` times a `FaerMatrix`, a range view, their
`transpose`/`adjoint`, or a triangular wrapper of a `FaerMatrix` (on either
side) scales rows or columns in
Rust: the stored diagonal is borrowed as a column (a Julia `Diagonal` crosses
as its contiguous buffer for one call) and never expanded to a dense matrix,
and the dense operand is borrowed in place. `mul!(C, D, B, α, β)` with
a `FaerMatrix` or view destination folds `α` into an O(n) copy of the diagonal,
builds the result in one Rust buffer and writes it into `C`; `β == 0` never reads
`C`, `α == 0` never reads the operands, and an operand sharing the destination's
owner is first copied natively. A native dense operand of a narrower scalar
type is converted once in Rust (applying its transpose or conjugation) rather
than through a Julia snapshot. A destination narrower than the promoted type
is intentionally computed in the promoted type and converted on assignment, as
LinearAlgebra does (a nonreal result for a real destination throws
`InexactError`). `D * v::FaerVector` and `r::FaerRow * D` return native vectors,
and `mul!(y, D, x, α, β)` / `mul!(z, r, D, α, β)` write native column (or
column view) and row destinations in place. `transpose(D)` and `D'` take the
same native paths (`D'` of a complex diagonal is one O(n) conjugated copy).

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

## Direct in-place LU, QR, Cholesky, LDLᴴ and Bunch–Kaufman

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

The same `scratch=` keyword selects the direct path for `full_piv_lu!(A)`
(faer `lu::full_pivoting::factor::lu_in_place`; `F.p` and `F.q` hold both
permutations), `ldlt!(A)` (`cholesky::ldlt::factor::cholesky_in_place`; L below
the diagonal and D on it) and `bunchkaufman!(A, rook)`
(`cholesky::lblt::factor::cholesky_in_place` with the same `LbltParams` as the
retained factor: `Rook` for `rook=true`, faer's default `PartialDiag`
otherwise). LDLᴴ and Bunch–Kaufman take a Hermitian `FaerMatrix` or a
`Hermitian`/`Symmetric` wrapper and, for an upper wrapper, store the adjoint
factor in the upper triangle. Bunch–Kaufman keeps the block diagonal's
off-diagonal and the symmetric permutation in Julia-owned buffers of F and
provides `F.D`, `F.p`, `F.P`, `F.rook` and `inertia(F)`. Size the workspace with
`scratch_requirements(full_piv_lu!, A)`, `scratch_requirements(ldlt!, A)` or
`scratch_requirements(bunchkaufman!, A[, rook])`.

```julia
H = FaerMatrix([1e-8 2 0; 2 -3 1; 0 1 4])
K = bunchkaufman!(Hermitian(H, :U), true; scratch = FaerScratch(bunchkaufman!, H, true))
K \ [1.0, 2.0, 3.0]
```

`ldiv!(F, B; scratch, team)` solves in place on a `FaerMatrix` B of the same
scalar type, including through `transpose(F)` and `adjoint(F)`. Tall QR writes
the least-squares solution into `B[1:n, :]`. `\`, the factor properties,
`Matrix(F)`, `inv`, and `det`/`logabsdet`/`logdet` (all but QR; computed
natively from the stored pivots with the diagonal log-measure) are also
available.

With caller scratch, Colmat allocates no native memory for the factorization
or the solve. Small shapes make no native allocation at all. Blocked shapes
still allocate inside faer's matrix products (also inside its blocked
triangular solves). On aarch64 with `Par::Seq`:

| case | direct | default path |
| :--- | ---: | ---: |
| LU 70×70 | 9 | 14 |
| full-pivot LU 70×70 | 0 | 7 |
| LDLᴴ 70×70 / 140×140 | 1 / 18 | 4 / 21 |
| Bunch–Kaufman 70×70 / 140×140 | 0 / 22 | 6 / 28 |
| solves up to 70×70 | 0 | — |
| LDLᴴ/Bunch–Kaufman solve 140×140, 3 RHS | 8 | — |

A lone faer 70×35×70 product allocates twice and a lone 140×140 unit-triangular
solve four times (t4a-faer `upstream-reports/gemm-001`). Allocations under a team
were not measured.

The direct factor does not support `rdiv!`, operator or preconditioner use,
pseudoinverse, mixed-scalar promotion of factors, rectangular LU or wide-QR
solves. Those need the default result without `scratch=`. Pivoted Cholesky
and column-pivoted QR have no direct path yet. `cholesky!(A, RowMaximum();
scratch)` keeps its existing meaning (caller scratch for the independent
`CholeskyPivoted` factor), and faer's in-place pivoted kernel covers only the
negative-tolerance path.

## Direct triangular solves and destructive triangular inverses

```julia
using Colmat, LinearAlgebra
A = FaerMatrix([2.0 0; 1 4])
B = FaerMatrix([2.0 4; 9 8])
ldiv!(LowerTriangular(A), B)            # solved in B's own storage
@assert Matrix(B) ≈ LowerTriangular([2.0 0; 1 4]) \ [2.0 4; 9 8]
X = [1.0 2; 3 4]
rdiv!(X, UpperTriangular(transpose(A))) # X / transpose(L), in place in X
Z = FaerMatrix(ComplexF64[1 2; 3 4])
work = FaerScratch(ldiv!, LowerTriangular(A), Z) # promoted triangle copy
ldiv!(LowerTriangular(A), Z; scratch = work)
T = UpperTriangular(FaerMatrix([2.0 1; 0 4]))
LinearAlgebra.inv!(T; scratch = FaerScratch(LinearAlgebra.inv!, T))
```

Lower/upper/unit triangular wrappers of a `FaerMatrix`, a native matrix view or
the `transpose`/`adjoint` (or elementwise conjugate) of either solve with faer's
`solve_{unit_}{lower,upper}_triangular_in_place_with_conj`. The triangle is
borrowed through its view descriptor, so transposed, adjoint and strided parents
are never copied. `ldiv!(A, B)`, `ldiv!(Y, A, B)` and `rdiv!(B, A)` run the
kernel directly on the destination when it is a `FaerMatrix`/`FaerVector`, a
native view, a `faerview` borrow or an injective Julia strided array holding
`promote_type(eltype(A), eltype(B))`. `rdiv!` solves `transpose(op(A)) *
transpose(X) = transpose(B)` on the same memory. Narrower or non-strided
destinations stage one promoted copy. A destination that shares the triangle's
owner solves from a snapshot of the triangle. `A \ B` and `B / A` return a
`FaerMatrix` (a vector RHS returns a vector, and a row vector stays a row
vector).

faer's dense triangular kernels need no workspace. `scratch` holds only the
promoted copy of a triangle whose scalar is narrower than the promoted type
(for example, a real triangle with a complex RHS), so
`scratch_requirements(ldiv!, A, B)` is zero without promotion. It is also
defined for `\`, `rdiv!` and `ldiv!(Y, A, B)`, with `FaerScratch` and
`reserve!` for each. `team=` runs the kernel on a caller-owned `FaerTeam`.
Zero diagonals throw `SingularException(i)`. Closed owners, closed teams and
undersized scratch throw before any write.

`LinearAlgebra.inv!(A; scratch, team)` overwrites the selected triangle of the
parent (a `FaerMatrix`, a native view or a `transpose` of either) with the
inverse triangle and returns `A`, as Julia's `inv!` does for strided triangles.
The opposite strict triangle is untouched, and so is the diagonal of a unit
wrapper. faer's `triangular_inverse::invert_*` kernels write into a separate
destination, so the source triangle is first copied into scratch, sized by
`scratch_requirements(inv!, A)`. Adjoint parents keep using `inv(A)`.

With caller scratch, Colmat allocates no native memory on these paths. The
direct solve made no native allocation through n = 70 (aarch64, `Par::Seq`).
At n = 300, faer's blocked kernel allocates matmul packing buffers: 12 per
solve. The in-place inverse is allocation-free at n = 5, but faer's matrix
products allocate at n = 70. See t4a-faer `upstream-reports/gemm-001`.
Allocations under a team were not measured.

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
above; sparse LU adds these conversions and typed sparse CSC output. The in-place
`lu!/qr!/cholesky!(A; scratch)` results (`FaerInPlaceFactorization`) also support
`reconstruct(F)` and the typed conversions; they read A's packed storage and
throw `ArgumentError` once F is closed or A is closed, resized or truncated.
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
