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

### 借用 Julia 配列の分解と in-place 求解

`faerview(J)` で借用した Julia 行列（4 型、負・非単位 stride を含む）は、
`lu`・`full_piv_lu`・`qr`（`NoPivot`/`ColumnNorm`）・`cholesky`・`ldlt`・
`bunchkaufman`（`Hermitian`/実 `Symmetric` ラッパーを含む）・`svd`/`svdvals`・
`eigen`/`eigvals` の入力として faer が親配列の `MatRef` を直接読みます。
結果は `FaerMatrix` 入力と同じ型（`FaerFactorization`・`SVD`・`Eigen`）で、
Julia や native の入力 snapshot は作りません（因子自身の格納は出力として確保）。
以前は LinearAlgebra の汎用経路（LAPACK と `Matrix` へのコピー）でした。
LAPACK を使うには親配列 `parent(B)` を渡します。`det`・`inv`・`B\b`・
`factorize`・`cond`・`rank`・`pinv` など LinearAlgebra の汎用関数もこの native 分解を
通ります。`scratch`・`team`（SVD/EVD）・`alg`・`native=true`・`side`・平衡化・
チューニング中の呼び出しは入力を native へ1回コピーして `FaerMatrix` の経路を使います。
読み取り専用の NPY 借用も分解の入力にできます。ラッパーなしの Hermitian な借用行列の
`eigen`/`eigvals` は、Julia 配列と同様に自己随伴ドライバで実固有値を返します
（`eigmax`/`eigmin` も使えます。`FaerMatrix` の `eigen` は一般ドライバで複素数です）。

同じ要素型の正方・ランク落ちしていない保存因子による `ldiv!(F, B)` は、
`B` が `Matrix`・strided `SubArray`・`faerview`・`FaerVector`・native ビューなら
faer の `solve_in_place` を `B` の記憶域で直接実行し、右辺のコピー・新しい結果・
書き戻しを作りません。`ldiv!(X, F, B)` は `B` を `X` へ native に1回代入して
（重なる入力は snapshot）から `X` で同じ in-place 求解をします。`F \ B`
（`B` が Julia strided 配列・`faerview`）は右辺の native コピー1つで解いて返します。
2引数 `rdiv!(B, F)` も同じ条件で `transpose(B)` に在位置求解します。`scratch=` を
渡すと因子種別ごとの低水準 kernel が caller scratch 上で動き、在位置求解は native
割当なしになります。長方形・ランク落ちした QR・SVD 因子、型昇格は従来の新規結果の
経路で、Julia の書き込み先へは結果を native に1回代入します。

```julia
using Colmat, LinearAlgebra

J = [4.0 1; 1 3]
F = lu(faerview(J))                  # J を直接読む native LU
@assert F isa Colmat.FaerFactorization
B = [1.0 2; 3 4]
ldiv!(F, B)                          # B の記憶域で直接求解
@assert B ≈ J \ [1.0 2; 3 4]
@assert eigvals(Hermitian(faerview(J))) ≈ eigvals(Hermitian(J))
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

## Pseudoinverses of saved QR factors

```julia
using Colmat, LinearAlgebra
a = [1.0 2 3; 2 4 6; 1 0 1; 0 1 1]       # rank two
F = qr(FaerMatrix(a), ColumnNorm())
P = pinv(F; rtol=1e-10)
@assert Matrix(P) ≈ pinv(a; rtol=1e-10)
FaerTeam(2) do team
    work = FaerScratch(pinv, F; team)    # SVD of saved R
    Q = pinv(F; scratch=work, team)
    @assert Matrix(Q) ≈ pinv(a)
    close(Q)
    close(work)
end
close(P)
close(F)
```

`pinv(F; atol, rtol, team, scratch)` of a saved `qr(A)` or `qr(A, ColumnNorm())`
returns an independent native `FaerMatrix` without refactorizing A. With
`A·P = Q·R`, R has the singular values of A, so the cutoff
`max(atol, rtol*maximum(svdvals(A)))` and the defaults (`rtol =
min(m,n)*eps(real(T))` when `atol` is zero, otherwise zero) are those of
`pinv(A; atol, rtol)`. Tall, wide, square and rank-deficient factors use the same
path. The SVD of R runs in caller scratch (`scratch_requirements(pinv, F;
team)`, `FaerScratch`, `reserve!`), and the retained singular columns are applied
through the saved Householder reflectors, so no dense Q is formed. A direct
`qr!(A; scratch)` factor (`FaerInPlaceFactorization`) uses the native SVD
pseudoinverse of `F.R` and its thin Q on `team`, without a scratch form.

Julia has no `pinv` for LU, Cholesky, LDLᵀ or Bunch–Kaufman factors. These
factors are only useful for nonsingular input, where `inv(F)` is the
pseudoinverse; for singular input they carry no rank-revealing information, so
Colmat does not add them. Schur and Hessenberg factors would need the same SVD
as `pinv(A)`.
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

`qr!(A, ColumnNorm(); scratch, team)` runs faer's column-pivoted
`qr::col_pivoting::factor::qr_in_place` on A and returns a
`FaerInPlaceFactorization{T,:pivoted_qr}`. A holds the packed QR of `A*P`, and
the column permutation (`F.p`, and `F.P` with `A*P == Q*R`) and the block
coefficients live in Julia-owned buffers of F. The packed factor, `F.p` and
`F.Qcoeff` are bit-identical to those of `qr(A, ColumnNorm())`. Besides the
solves, right solves, `inv`/`inv!` and parts above, `rank(F; atol, rtol)`,
`ldiv!(F, B, rcond)` and `ldiv!(X, F, B; atol, rtol)` use the retained factor's
rank rule and rank-truncated minimum-norm kernel (see the column-pivoted QR
section), and `pinv(F; atol, rtol)` is available. The plain `F \ b` and
`ldiv!(F, B)` run faer's least-squares solve without a rank test, so use a
tolerance form for rank-deficient factors.

```julia
X = [1.0 2 3; 2 4 6; 1 2 1; 0 0 1]                  # rank two
P = FaerMatrix(X)
F = qr!(P, ColumnNorm(); scratch = FaerScratch(qr!, P, ColumnNorm()))
@assert rank(F) == 2 && F.factors === P
x = zeros(3)
ldiv!(x, F, [1.0, 2, 3, 4]; rtol = 1e-10)           # minimum-norm solution
@assert x ≈ pinv(X) * [1.0, 2, 3, 4]
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
solve four times (t4a-faer `upstream-reports/gemm-001`). With a `FaerTeam` of
width 2, the direct LU, full-pivot LU, QR, column-pivoted QR and Cholesky
factorizations and their solves made no allocation on the calling thread at
n = 6 and n = 200 (aarch64). The test allocation probe counts the calling
thread only, so allocations on the team's workers are not visible in these
figures.

Square factors also solve from the right in B's own storage:
`rdiv!(B, F; scratch, team)` (and `transpose(F)`, `adjoint(F)`) runs faer's
in-place left solve on the transposed view of B, as `Aᵀ Xᵀ = Bᵀ` for `X A = B`
and as the conjugate-only solve `conj(A) Xᵀ = Bᵀ` for `X Aᴴ = B`. Size its
workspace with `scratch_requirements(rdiv!, B, F)`. `B / F` returns a
`FaerMatrix` solved in one native copy of B.

`LinearAlgebra.inv!(F; scratch, team)` overwrites A (`F.factors`) with `inv(A)`
and returns it, like Julia's `inv!(::LU)` and `inv!(::Cholesky)`. For LDLᴴ,
Bunch–Kaufman and Cholesky both triangles hold the Hermitian inverse. faer's
inverse kernels read the factor while they write their output, so the packed
factor (and Bunch–Kaufman's off-diagonal) is first copied into scratch
(`scratch_requirements(inv!, F)`). F no longer holds a factor afterwards, so
`inv!` closes it. Singular, failed or non-square factors throw before any
write. `LinearAlgebra.inv!(A::FaerMatrix; scratch, team)` inverts a square
matrix in place through its direct LU, with one workspace sized by
`scratch_requirements(inv!, A)`.

```julia
A = FaerMatrix([4.0 1; 2 3])
F = lu!(A; scratch = FaerScratch(lu!, A))
X = FaerMatrix([1.0 2; 3 4])
rdiv!(X, F; scratch = FaerScratch(rdiv!, X, F))   # X / A, in X's storage
LinearAlgebra.inv!(F; scratch = FaerScratch(LinearAlgebra.inv!, F))
@assert Matrix(A) ≈ inv([4.0 1; 2 3]) && !isopen(F)
```

With caller scratch, right solves and `inv!` made no native allocation for any
kind (LU, full LU, QR, Cholesky, LDLᴴ, both Bunch–Kaufman pivotings, both
storage triangles) at n = 6 and n = 70 (complex double, aarch64, `Par::Seq`).
The zero-pivot check of a QR factor runs natively before any write, so it does
not allocate either.

`ldiv!(F, b)` of a square direct factor also overwrites Julia strided arrays
and `FaerVector`s, and a real factor solves a complex `b` as two real solves.
`FaerLinearOperator(F)` (and `M = F` in `cg`/`bicgstab`/`lsmr`) is an
inverse-action operator over an independent copy of the direct factor.
The direct factor does not support
promotion of the factor to another scalar type, rectangular LU or wide-QR solves (`pinv(F;
atol, rtol, team)` of a direct QR is supported, see below). Those need the
default result without `scratch=`. Pivoted Cholesky
has no direct path yet. `cholesky!(A, RowMaximum();
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

Lower/upper/unit triangular wrappers of a `FaerMatrix`, a native matrix view, a
`faerview` borrow of Julia memory (including its strided views) or the
`transpose`/`adjoint` (or elementwise conjugate) of any of them solve with faer's
`solve_{unit_}{lower,upper}_triangular_in_place_with_conj`. The triangle is
borrowed through its view descriptor, so transposed, adjoint and strided parents
are never copied. `ldiv!(A, B)`, `ldiv!(Y, A, B)` and `rdiv!(B, A)` run the
kernel directly on the destination when it is a `FaerMatrix`/`FaerVector`, a
native view, a `faerview` borrow or an injective Julia strided array holding
`promote_type(eltype(A), eltype(B))`. `rdiv!` solves `transpose(op(A)) *
transpose(X) = transpose(B)` on the same memory. Narrower or non-strided
destinations stage one promoted copy. A destination that shares the triangle's
owner, or for a `faerview` parent the same Julia memory, solves from a
snapshot of the triangle. `A \ B` and `B / A` return a `FaerMatrix` (a vector
RHS returns a vector, and a row vector stays a row vector).

A plain Julia array parent (`LowerTriangular(::Matrix)`) intentionally keeps
LinearAlgebra's path: with a Julia RHS every type involved belongs to Base, so a
Colmat method would be type piracy, and LinearAlgebra already solves strided
`BLAS` triangles in place with `trsm`/`trsv`. Wrap the parent with `faerview(A)`
to select the native kernel without copying it.

faer's dense triangular kernels need no workspace. `scratch` holds only the
promoted copy of a triangle whose scalar is narrower than the promoted type
(for example, a real triangle with a complex RHS), so
`scratch_requirements(ldiv!, A, B)` is zero without promotion. It is also
defined for `\`, `rdiv!` and `ldiv!(Y, A, B)`, with `FaerScratch` and
`reserve!` for each. `team=` runs the kernel on a caller-owned `FaerTeam`.
Zero diagonals throw `SingularException(i)`. Closed owners, closed teams and
undersized scratch throw before any write.

`LinearAlgebra.inv!(A; scratch, team)` overwrites the selected triangle of the
parent (any of the parents above, including `adjoint`, conjugated and
`faerview` parents) with the inverse triangle and returns `A`, as Julia's
`inv!` does for strided triangles. A conjugated parent needs no extra pass:
the stored triangle is inverted unconjugated, since `inv(Xᴴ) = inv(X)ᴴ`. The
opposite strict triangle is untouched, and so is the diagonal of a unit
wrapper. faer's `triangular_inverse::invert_*` kernels write into a separate
destination, so the source triangle is first copied into scratch, sized by
`scratch_requirements(inv!, A)`.

With caller scratch, Colmat allocates no native memory on these paths. The
direct solve made no native allocation through n = 70 (aarch64, `Par::Seq`).
At n = 300, faer's blocked kernel can allocate matmul packing buffers: from 0
to 12 per solve, depending on warm-up. The in-place inverse is allocation-free at n = 5, but faer's matrix
products allocate at n = 70. See t4a-faer `upstream-reports/gemm-001`. Under a
two-worker `FaerTeam`, the calling thread made no native allocation for left or
right solves at n = 6, 70 and 300. The allocation probe observes only the
calling thread, so allocations on the team's workers are not counted.

`cond(A, 1)` and `cond(A, Inf)` of these wrappers are estimates, as Julia's
LAPACK `trcon` path is for strided triangles. `opnorm(A, p)` is exact, and
`‖A⁻¹‖ₚ` is the deterministic Higham–Tisseur block estimate (a lower bound,
usually exact) from a few native solves with A and Aᴴ, so A⁻¹ is never formed.
A zero diagonal gives `Inf`. `cond(A, 2)` uses the singular values. A native
`Hermitian` (or real `Symmetric`) matrix computes `cond(A, 2)` as
`maximum(abs.(λ))/minimum(abs.(λ))` from faer's self-adjoint eigenvalues instead
of an SVD. At n = 1000 on aarch64 macOS, a triangular `cond(A, 1)` took 2.7 ms
(1.0 s through the generic `inv`), and a Hermitian `cond(A)` took 94 ms (2.2 s
through the generic SVD).

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
  `eigen`/`eigvals` も同じ規則で、格納三角が実数なら自己随伴 EVD（実固有値）、それ以外は一般 native EVD
  （複素固有値）に渡します（spectral マニュアルの self-adjoint EVD algorithms 節）。
- `F.uplo` は Julia の `Char` ではなく `Symbol`（`:L`/`:U`）です。LDLT/Bunch–Kaufman は
  常に下三角向きの `F.L`（`A[p,p] = L D Lᴴ`、上三角ラッパーは格納三角の随伴を分解）を返し、
  Julia の `uplo='U'` における `F.U`（`A[p,p] = U D Uᴴ`）は提供しません。`F.L'` は
  `Uᴴ D U` を満たすだけで Julia の `U` にならず、反転による上三角への付け替え
  （`U = J L J`）は LAPACK `sytrf` の上三角因子と一致せず `F.D`/`F.p` の意味も変えるためです。
  密な `ldlt(A)` は Julia 標準には無く（`SymTridiagonal`・疎のみ）、Colmat では
  ピボットなし LDLᴴ として Hermitian 入力を受けます。`cholesky[!](A, NoPivot())` も同じ native 経路です。

## 自己随伴分解のパラメータと動的正則化

```julia
using Colmat, LinearAlgebra
A = FaerMatrix([1.0 2; 2 1])                      # 不定値
F = cholesky(A; regularization_delta = 1, regularization_epsilon = 1e-8)
@assert issuccess(F) && F.regularization_count == 1
L = Matrix(F.L)
@assert L * L' ≈ [1 2; 2 5]                       # A + diag(0, 4)
G = ldlt(FaerMatrix([0.0 1; 1 0]); regularization_delta = 1,
         regularization_epsilon = 1e-8, regularization_signs = [1, -1])
@assert G.regularization_count == 1
H = bunchkaufman(Hermitian(FaerMatrix([1e-8 1; 1 -3])); pivoting = :full)
K = with_cholesky_tuning(; ldlt_block_size = 32, lblt_block_size = 16) do
    ldlt(FaerMatrix([4.0 1; 1 3]))
end
```

- 結果を変える faer の設定は呼び出しごとのキーワードです。`cholesky`・`ldlt`（`!`、
  `scratch=` の直接経路、`team=`、`faerview`・ラッパー入力を含む）の
  `regularization_delta`/`regularization_epsilon` は faer の `LltRegularization`/
  `LdltRegularization` を有効にします（両方正のときのみ。片方だけ・負・非有限は `ArgumentError`）。
  分解される行列が `A` に対角摂動を加えたものに変わるため、意図した場合（前処理など）に限って使います。
  `ldlt` の `regularization_signs`（各行 -1/0/1）は期待する符号です。`F.regularization_count` は
  faer の `dynamic_regularization_count` です。
- 固定版 faer の LLᴴ 正則化は置換したピボットの対角を誤って格納し、ブロック化経路では因子全体が
  壊れます。Colmat は同じ置換規則・カウントを持つ faer の LDLᴴ カーネル（全符号 +1）で分解し、
  `sqrt(D)` を掛けて L を作ります。LDLᴴ のカウントは期待符号 +1 のピボットだけを数え、
  -1 方向の補正や符号なしの置換は数えません（どちらも
  [実行済み再現例](https://github.com/AtelierArith/Colmat.jl/tree/main/docs/reproducers/faer-selfadjoint-factor-params)）。
- `bunchkaufman[!](A, rook; pivoting)` は `LbltParams.pivoting` の全戦略
  `:partial`・`:partial_diag`（既定）・`:rook`・`:rook_diag`・`:full` を選べます。`F.rook` は
  `:rook`/`:rook_diag` で `true`、`rook=true` と他の戦略の組み合わせは `ArgumentError` です。
  `:partial` は LAPACK の古典的 Bunch–Kaufman と同じ探索規則ですが、ピボット列が一致するとは限りません
  （9×9 の乱数対称行列 200 例で `bunchkaufman(Symmetric(A, :L)).p` と一致したのは 159 例）。
  非推奨の `Diagonal` は提供しません。`scratch_requirements(bunchkaufman!, A, rook; pivoting)` も同じ戦略を受けます。
- 性能だけに効く `LltParams`/`LdltParams` の `recursion_threshold`・`block_size` と
  `LbltParams` の `block_size`・`par_threshold`（`:full` のみが参照）は
  `with_cholesky_tuning(f; llt_recursion_threshold, llt_block_size, ldlt_recursion_threshold,
  ldlt_block_size, lblt_block_size, lblt_par_threshold)` のスコープで指定します（`nothing` は faer 既定）。
  保持因子・型付き・借用入力・team・staged/直接の `!` と scratch 照会がすべて従います。LBLᴴ の直接経路の
  作業領域は `lblt_block_size` で変わるため、照会もスコープ内で行います。faer の `LbltInfo.transposition_count`
  は公開しません（置換そのものは `F.p` で得られ、対称置換は行列式に影響しないため `det` は `B` のブロックだけから求めます）。

## LU・QR 分解のパラメータ

```julia
using Colmat, LinearAlgebra
A = FaerMatrix([4.0 1 2; 1 3 0; 2 0 5])
F = with_lu_tuning(; partial_recursion_threshold = 2,
                   full_par_threshold = FaerParThreshold(:never)) do
    lu(A), full_piv_lu(A)
end
Q = with_qr_tuning(; blocking_threshold = 0,
                   pivoted_par_threshold = FaerParThreshold(:never)) do
    qr(A), qr!(copy(A), ColumnNorm(); scratch = FaerScratch(qr!, A, ColumnNorm()))
end
```

- `with_lu_tuning(f; partial_recursion_threshold, partial_par_threshold, full_par_threshold)` は
  faer の `PartialPivLuParams.{recursion_threshold, par_threshold}` と `FullPivLuParams.par_threshold`、
  `with_qr_tuning(f; blocking_threshold, pivoted_par_threshold)` は `QrParams.blocking_threshold` と
  `ColPivQrParams.par_threshold` をスコープ内のすべての `lu`/`full_piv_lu`/`qr`/`qr(A, ColumnNorm())`
  構築に適用します（保存因子・型付き・借用入力・`team=`・staged/直接の `!`、`inv!(A)`、
  混合スカラー求解の昇格再分解、scratch 照会・`FaerScratch`・`reserve!`）。`nothing` は faer 既定で、
  2 つのスコープは同じ native 状態の別の欄を持つため、どちらの順にも入れ子にできます。
  これらは再帰・ブロック化・並列化だけを変え、分解は丸め誤差の範囲で同じです。
- faer の `PartialPivLu::new` などは既定パラメータしか渡さず、既存の因子から組み立てることもできないため、
  Colmat は保存 LU/QR 因子を自前の型で持ち、低水準の `*_in_place` kernel を呼びます。既定値のままなら
  faer の型と bit 一致します（Rust の単体テストで確認）。
- `PartialPivLuParams`/`FullPivLuParams` の rank 打切り（`max_rank`・`abs_tol`・`rel_tol`）は分解を途中で止め、
  faer の求解・逆行列・再構成はそのような因子を受け付けないため公開しません（列ピボット QR の rank は
  `rank(F; atol, rtol)` と許容誤差付き `ldiv!` で扱います）。`PartialPivLuParams.block_size` と
  `ColPivQrParams.blocking_threshold` は pin した faer の kernel が読まないため公開しません。
  `with_svd_tuning` の `qr_blocking` は SVD 内部の QR 用で、このスコープとは独立です。
  保存因子のない `A \ B`・`det(A)`・QZ/Schur・疎分解の内部 LU/QR は faer 既定のままです。

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
Two-argument forms require square factors. When B holds the factor's scalar in
writable memory (a `FaerMatrix`/`FaerVector`, a native view, a writable
`faerview` borrow or an injective Julia strided array), `ldiv!(F, B)` and
`rdiv!(B, F)` (also through `transpose(F)` and `adjoint(F)`) run faer's
per-kind `solve_in_place` directly in B's storage through its view descriptor
(a right solve on B's transposed view). No RHS copy or native result is made.
With caller scratch the solve makes no native allocation of its own (all seven
kinds; Float64 at n = 6 and 70, complex at n = 6; aarch64, `Par::Seq`).
Retained SVDs and rank-deficient QR (whose minimum-norm solution needs work
matrices), scalar promotion, read-only or non-strided B and three-argument
`rdiv!` keep the out-of-place path below; three-argument `ldiv!(X, F, B)`
first assigns B into X natively and then solves in X. Solve teams are independent of the
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
