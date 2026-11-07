# 固有値・Schur・QZ

単一行列と行列対の固有値問題、Schur 分解、左右固有ベクトルを扱います。

## Hessenberg と Schur 分解

Hessenberg 化と Schur 分解も native 経路を使います。

```julia
using Colmat, LinearAlgebra

A = FaerMatrix([1.0 2 0; -2 1 0; 0 0 4])
H = hessenberg(A)                # 明示的な Rust 所有 H・Q
@assert Matrix(H.Q * H.H * H.Q') ≈ Matrix(A)
S = schur(H)                     # Hessenberg 化を再実行せず Schur 段階を実行
@assert S isa Schur
@assert Matrix(S.Z * S.T * S.Z') ≈ Matrix(A)
G = ordschur(S, real.(S.values) .> 2)
@assert real(G.values[1]) ≈ 4
V = eigvecs(G)                   # Rust 所有の複素固有ベクトル行列
@assert Matrix(A) * Matrix(V) ≈ Matrix(V) * Diagonal(G.values)
ordschur!(S, real.(S.values) .> 2)
B = copy(A)
SB = schur!(B)
@assert SB.T === B
```

`schur` は Rust 所有の `T`・`Z` と Julia の固有値ベクトルを持つ
`LinearAlgebra.Schur` を返します。実数では共役対を 2×2 ブロックに保ち、
`ordschur` の選択では対の両方を含めるか、両方を除外します。
`hessenberg` の既定の結果は `FaerHessenberg` で、Julia の compact Householder 表現と
異なり `H`・`Q` の両方を明示的な行列として所有します。
`compact=true` は native reflector を保持する `FaerCompactHessenberg` を返し、
保存因子からの Schur でも Q を暗黙に適用します。所有権と bang の Q 置換は
[compact Hessenberg](reductions.md#Compact-general-Hessenberg-Q) を参照してください。
`hessenberg!`・`schur!` は独立した作業結果を計算してから入力の所有バッファを
更新し、形状と既存ビューを保ちます。`schur!(H)` は `H.H`・`H.Q` の両方を
更新し、返す Schur 分解とストレージを共有します。並べ替えも成功後に更新する
ため、失敗時は元の分解を保ちます。これらは内部の一時領域を割り当てます。

### Tuning the multishift QR and QZ drivers

```julia
using Colmat, LinearAlgebra
A = FaerMatrix([sin(i + j^2) for i in 1:80, j in 1:80])
F = with_schur_tuning(; shift_count=8, deflation_window=10, blocking_threshold=20) do
    schur(A)
end
@assert Matrix(F.Z) * Matrix(F.T) * Matrix(F.Z)' ≈ Matrix(A)
close(A)
```

`with_schur_tuning(f; shift_count, deflation_window, blocking_threshold,
nibble_threshold, shift_chase_cost)` applies faer's `SchurParams` and
`GeneralizedSchurParams` controls to every Schur and QZ factorization, scratch
query, `FaerScratch` and `reserve!` inside the scope, so query and execution
agree. Zero keeps faer's default. Because faer takes the shift count,
deflation window and shift-chase cost as `extern "C"` function pointers, the
overrides are stored natively for the scope, which holds Colmat's native-call
lock: other tasks wait, `f` must call Colmat on the current task, and the
previous settings return even when `f` throws. The general EVD driver's
internal Schur step keeps faer's defaults.

### Tuning the EVD drivers

```julia
using Colmat, LinearAlgebra
S = Hermitian(FaerMatrix([1.0 / (i + j) + (i == j) for i in 1:60, j in 1:60]))
F = with_evd_tuning(; self_adjoint_recursion_threshold=8,
                    tridiagonal_par_threshold=FaerParThreshold(:never)) do
    eigen(S)
end
@assert Matrix(S) * F.vectors ≈ F.vectors * Diagonal(F.values)
close(parent(S))
```

`with_evd_tuning(f; ...)` overrides the parameters inside faer's EVD drivers
for every general and self-adjoint `eigen`/`eigvals` call, scratch query,
`FaerScratch` and `reserve!` in the scope: the general driver's Hessenberg
`hessenberg_par_threshold`/`hessenberg_blocking_threshold` and
`evd_recursion_threshold` (eigenvectors from Schur), and the self-adjoint
driver's `tridiagonal_par_threshold` and `self_adjoint_recursion_threshold`.
`nothing` keeps faer's default. Inside the scope, calls without
`scratch`/`team` use the parameterized workspace driver. The general driver's
multishift QR step follows `with_schur_tuning`, and both scopes nest.
Destructive `eigen!` writeback keeps faer's defaults.

## 一般化 Schur／QZ

一般化 Schur／QZ は行列対に適用します。

```julia
using Colmat, LinearAlgebra

A = FaerMatrix([1.0 2; -2 1])
B = FaerMatrix([2.0 1; 1 3])
F = schur(A, B)
@assert F isa GeneralizedSchur
@assert Matrix(F.Q * F.S * F.Z') ≈ Matrix(A)
@assert Matrix(F.Q * F.T * F.Z') ≈ Matrix(B)
@assert eigvals(F) ≈ F.α ./ F.β
H = hessenberg(A, B)              # Rust 所有の H・T・Q・Z
G = schur(H)                      # 一般化 Hessenberg 化を再実行せず QZ 段階へ
@assert Matrix(G.Q * G.S * G.Z') ≈ Matrix(A)
AA, BB = copy(A), copy(B)
FM = schur!(AA, BB)
@assert FM.S === AA && FM.T === BB
```

`GeneralizedSchur` の `S`・`T`・`Q`・`Z` はすべて Rust 所有行列です。
`hessenberg(A,B)` は `FaerGeneralizedHessenberg` を返し、
`A ≈ Q*H*Z'`・`B ≈ Q*T*Z'` を満たします。`hessenberg!`・`schur!` は
計算成功後に既存の所有バッファを更新し、既存ビューを維持します。
更新する行列対は同じスカラー型で、別々の所有者にする必要があります。
`schur!(H)` は H/T/Q/Z のすべてを更新し、結果とストレージを共有します。

固有値は同次形式の `α`・`β` でも保持します。`β == 0, α != 0` は無限固有値、
`α == β == 0` は不定な固有値を表します。比に直す前の値を調べることで区別できます。
実数・複素数とも native QZ を使い、混合型の通常呼び出しでは型を昇格します。
一般化 `ordschur`／`ordschur!` は保存済みの S/T を native で並べ替え、
選択したブロックを先頭へ集めます。選択群・非選択群の内部の順序は保ちます。
実共役対の片方だけを選ぶと両方が対象になります。α/β は保存因子から更新します。

```julia
using Colmat, LinearAlgebra

A = FaerMatrix([4.0 0 0; 0 0 -1; 0 1 0])
B = FaerMatrix(Matrix{Float64}(I, 3, 3))
F = schur(A, B)
select = imag.(F.values) .> 0       # 実共役対の片方を選択
G = ordschur(F, select)
@assert all(abs.(real.(G.values[1:2])) .< 1e-12)
@assert all(isapprox.(abs.(imag.(G.values[1:2])), 1))
@assert G.values[3] ≈ 4
@assert Matrix(G.Q * G.S * G.Z') ≈ Matrix(A)
@assert Matrix(G.Q * G.T * G.Z') ≈ Matrix(B)
work = FaerScratch(ordschur, F)
req = scratch_requirements(ordschur!, F)
@assert scratch_capacity(work) == req
H = ordschur(F, select; scratch=work)
@assert H.values ≈ G.values
Sview = view(F.S, :, :)
@assert ordschur!(F, select; scratch=work) === F
@assert Matrix(Sview) == Matrix(F.S)
@assert F.values ≈ G.values
@assert scratch_capacity(work) == req
close(work)
```

`ordschur!` は全計算・固有値の変換に成功してから既存の S/T/Q/Z・α/β を
更新し、元の所有者とビューを保ちます。行列は別々の native 所有者、α/β は
要素領域の重ならない mutable strided vector が必要です。同じ Julia 配列の
互いに重ならないビューも使えます。数値的に安全な交換や固有値評価ができない
場合は例外を返し、元の分解を保ちます。`scratch=` では `FaerScratch` を再利用でき、
`reserve!(work, ordschur, F)` で容量を増やせます。容量・alignment の不足や閉じた
scratch は入力を変更せずに拒否します。作業因子と Julia の出力・変換配列は別に確保します。
`values`／`eigvals(F)` は Julia の α/β の比に従い、複素ゼロ除算では NaN を
含む場合があるため、無限・不定値の判定には同次対を使ってください。
QZ 構築の `scratch`・`team` は、このページの構築用リソースの節を参照してください。

## Balancing before eigenvalue computations

`balance` calls faer's permutation and power-of-two scaling routine:

```julia
using Colmat, LinearAlgebra
A = [1.0 1024; 1/1024 2]
F = balance(A)
@assert Matrix(F.B) ≈ A[F.perm, F.perm] .* reshape(F.scale, 1, :) ./ F.scale
E = eigen(F.B)
X = balance_eigenvectors(F, E.vectors)
@assert A * Matrix(X) ≈ Matrix(X) * Diagonal(E.values)
```

`F.perm` uses 1-based indices, and `F.active` is the active Julia range.
`balance!(A::FaerMatrix)` replaces A after a successful computation and returns
a result sharing A as `B`. `balance_eigenvectors!` restores right eigenvector
columns in place after staging the full result. Back-transforms do not normalize
columns; invalid metadata, conversion failure or overflow leave the destination
unchanged.

The general EVD can balance in one call. faer's EVD driver itself does not
balance, so `eigen`/`eigvals(A; permute=true, scale=true)` balances natively
first, runs the EVD on `B` with the same `side`, `scratch` and `team`, maps the
right (`P*S*v`) or left (`P*S⁻¹*w`) eigenvectors back and renormalizes each
column to unit 2-norm:

```julia
using Colmat, LinearAlgebra
a = [1.0 1e4 0; 1e-4 2 1e3; 0 1e-3 3]
A = FaerMatrix(a)
E = eigen(A; permute=true, scale=true)
@assert norm(a * E.vectors - E.vectors * Diagonal(E.values)) < 1e-10
L = eigen(A; side=:left, permute=true, scale=true)
@assert norm(L.vectors' * a - Diagonal(L.values) * L.vectors') < 1e-10
close(A)
```

faer's `balance_in_place` always permutes and scales together, so `permute`
and `scale` must be equal (otherwise `ArgumentError`). Balancing is opt-in:
the default `false` keeps the unbalanced native driver, unlike LinearAlgebra's
LAPACK default. The Hessenberg, Schur and EVD-from-Schur parameters inside
faer's EVD driver are tracked with the Hessenberg (#123) and Schur (#110)
controls.

## Partial eigenvalues and singular values

```julia
using Colmat, LinearAlgebra
op = FaerLinearOperator(Diagonal(collect(1.0:96)))
F = partial_self_adjoint_eigen(op, 2; tol=1e-8)
@assert F.nconverged == 2
@assert F.values ≈ [96.0, 95.0]
@assert maximum(F.residuals) < 1e-5
S = partial_svd(op, 2; tol=1e-8)
@assert S.S ≈ [96.0, 95.0]
```

`partial_eigen`, `partial_self_adjoint_eigen` and `partial_svd` call native
restarted projection methods and return only reported converged columns, with
independently recomputed residuals. Set `v0`, `min_dim`, `max_dim`, `tol` and
`max_restarts` to control the search. The self-adjoint variant assumes Hermitian
input. A start orthogonal to a desired invariant subspace can miss it. Inspect
`nconverged` and residuals before using results. Small/full-projection
self-adjoint and SVD requests (`max(max_dim, 64, 2k) >= min(size(op)...)`) take
faer's native full-subspace branch, which materializes the operator and runs a
dense EVD/SVD; they report `algorithm == :dense_fallback`, and iterative results
report `:partial`. The self-adjoint dense branch reads the lower triangle of the
materialized operator, so results are defined only for Hermitian operators. If
the branch fails, for example because the materialized operator is not finite,
it returns `nconverged == 0` instead of throwing. faer emits the converged
results sorted by decreasing magnitude, and they are written directly into the
returned Julia arrays: there is no Rust or Julia staging copy, no sort
permutation copy, and truncation to `nconverged` resizes the arrays in place.
`partial_svd` writes the right singular vectors straight into the rows of the
returned `Vt::Matrix` (conjugated in place), so no transpose copy is made. The
residuals are recomputed natively in the same call; a real operator acts on the
real and imaginary parts of complex `partial_eigen` vectors instead of being
promoted to a complex copy. A `v0` given as a `FaerVector`, a one-dimensional
`FaerView`, a [`faerview`](@ref) borrow or a strided Julia vector of the
operator's scalar type is read in place; other start vectors are converted once.

Measured Julia `@allocated` for `k = 6` and a dense `Float64` operator,
before → after: `partial_eigen` 567 KB → 66 KB (n = 400) and 1.84 MB → 181 KB
(n = 1600); `partial_self_adjoint_eigen` 973 KB → 99 KB and `partial_svd`
(1600×800) 1.56 MB → 148 KB at n = 1600. What remains is the returned arrays
plus a fixed FFI marshalling overhead of roughly 16–30 KB; a Julia callback
operator adds about 100 B per application for its trampoline.

All three methods accept `scratch` and `team`. Query or reserve with the same
projection controls and team used for execution. The buffer covers the partial
kernel, including its full-subspace operator materialization and dense EVD/SVD,
plus independent residual applications. For the self-adjoint full-subspace
branch Colmat also adds the `n×n` eigenvector buffer that faer's own scratch
query omits. Querying does not apply operators. Julia callback operators reuse
their per-operator marshalling arrays; callbacks run on the calling Julia task,
including when a team executes the native kernels.

```julia
using Colmat, LinearAlgebra
op = FaerLinearOperator(Diagonal(collect(1.0:96)))
team = FaerTeam(2)
work = FaerScratch(partial_self_adjoint_eigen, op, 2; team)
F = partial_self_adjoint_eigen(op, 2; scratch=work, team, tol=1e-8)
@assert F.values ≈ [96.0,95.0]
reserve!(work, partial_svd, op, 2; max_dim=96, team)
S = partial_svd(op, 2; max_dim=96, scratch=work, team)
@assert S.algorithm == :dense_fallback && S.S ≈ [96.0,95.0]
close(work)
close(team)
close(op)
```

## Destructive eigenvalue decompositions

```julia
using Colmat, LinearAlgebra
original = [2.0 -1; -1 2]
A = FaerMatrix(original)
F = eigen!(Hermitian(A, :L))
@assert Matrix(A) == F.vectors
@assert original * F.vectors ≈ F.vectors * Diagonal(F.values)
rotation = [0.0 -1; 1 0]
B = FaerMatrix(rotation)
G = eigen!(B)
@assert rotation * G.vectors ≈ G.vectors * Diagonal(G.values)
@assert eltype(B) == Float64
```

`eigen!` and `eigvals!` support four native types and preserve the input scalar
type. General matrices receive their real quasi-triangular or complex triangular
Schur form. Right eigenvectors are normalized and copied to independent Julia
arrays; `sortby` follows the existing eigen API. Native Hermitian/real Symmetric
wrappers read the selected triangle. Their `eigen!` writes eigenvectors to the
parent, while `eigvals!` writes a diagonal matrix of sorted eigenvalues.

For generalized pairs, native general eigen and QZ passes produce the result
and S/T writeback. Self-adjoint pairs write eigenvectors (or diagonal values for
`eigvals!`) to the first parent and a diagonal eigenvalue matrix to the second.
These are explicit staged formats; they do not expose LAPACK work buffers.
Pair parents must be distinct with matching scalar types. Convergence/sort errors
before writeback preserve inputs. Returned arrays remain independent of later
input changes/close. Factorization, sorting and writeback allocate. Single-matrix and generalized EVD support scratch/team keywords below.

## Self-adjoint EVD with reusable workspace and teams

```julia
using Colmat, LinearAlgebra
original = ComplexF64[4 1im; -1im 3]
A = Hermitian(FaerMatrix(original), :L)
FaerTeam(2) do team
    workspace = FaerScratch(eigen, A; team=team)
    F = eigen(A; scratch=workspace, team=team)
    @assert original * F.vectors ≈ F.vectors * Diagonal(F.values)
    reserve!(workspace, eigvals, A; team=team)
    @assert eigvals(A; scratch=workspace, team=team) ≈ F.values
    close(workspace)
end
```

Native Hermitian and real Symmetric single-matrix `eigen`, `eigvals`, `eigen!`
and `eigvals!` accept `scratch=` and `team=`. Query matching mode/team options
through `scratch_requirements` or construct using `FaerScratch(function, A)`.
The buffer includes a canonical copy of the selected triangle and temporary
values. Unused triangles and complex diagonal imaginary parts are ignored;
nonfinite values in the active data are rejected. Eigenvalues are real.
Sorting and independent eigenvector output arrays allocate separately. Caller
workspace never grows implicitly, and temporary workspace is closed on errors.
Other kernels retain their existing workspace and execution interfaces.

## General EVD with caller-owned scratch and teams

```julia
using Colmat, LinearAlgebra
original = [0.0 -1; 1 0]
A = FaerMatrix(original)
FaerTeam(2) do team
    workspace = FaerScratch(eigen, A; team=team)
    F = eigen(A; scratch=workspace, team=team)
    @assert original * F.vectors ≈ F.vectors * Diagonal(F.values)
    @assert imag.(F.values) ≈ [-1.0, 1.0]
    reserve!(workspace, eigvals, A; team=team)
    @assert eigvals(A; scratch=workspace, team=team) ≈ F.values
    close(workspace)
end
```

General native `eigen`/`eigvals` now accept `scratch=` and `team=`. Outputs remain
complex for real input, with normalized right eigenvectors. Real conjugate-pair
conversion uses the provided scratch; complex outputs are written directly.
Workspace queries/factories/reserve support the ordinary and bang aliases, and
use actual team conditions. Calls without options retain their original path.

For `eigen!`/`eigvals!`, these options apply to the eigen computation. The staged
Schur-form writeback performs an additional sequential native Schur factorization
with its own temporary storage. Queries describe the eigen pass, so caller-owned
scratch does not cover that additional writeback computation. Generalized-pair
scratch/team interfaces remain unexposed. Output arrays, sorting and temporary
writeback factors allocate; input is preserved on errors before writeback.

## Definite self-adjoint generalized EVD workspace

```julia
using Colmat, LinearAlgebra
A0 = ComplexF64[4 1im; -1im 3]
B0 = ComplexF64[2 0; 0 1]
A, B = Hermitian(FaerMatrix(A0), :L), Hermitian(FaerMatrix(B0), :U)
FaerTeam(2) do team
    workspace = FaerScratch(eigen, A, B; team=team)
    F = eigen(A, B; scratch=workspace, team=team)
    @assert A0 * F.vectors ≈ B0 * F.vectors * Diagonal(F.values)
    @assert F.vectors' * B0 * F.vectors ≈ I
    reserve!(workspace, eigvals, A, B; team=team)
    @assert eigvals(A, B; scratch=workspace, team=team) ≈ F.values
    close(workspace)
end
```

Four native types and real `Symmetric` pairs support `eigen/eigvals` and their
bang variants with `scratch=` and `team=`. B must be positive definite. The
workspace contains two selected-triangle copies, temporary eigenvalues, and
reused LLT/EVD kernel storage. The native pipeline reduces through Cholesky,
solves a self-adjoint EVD, and back-transforms B-orthonormal eigenvectors.
Unused triangles and imaginary diagonal parts are ignored independently.
Output arrays, scalar promotion, sorting and staged bang writeback allocate
separately. Nonpositive B or sorting failure preserves bang inputs.

A B that is not positive definite throws `PosDefException(i)` whose `info` is
the one-based failing pivot, matching `cholesky(Hermitian(B); check=false).info`.
Non-finite values in a selected triangle throw `ArgumentError`. A diagonal B
(`FaerDiagonal` or `Diagonal`) with positive entries uses the same native
driver; a zero entry throws `ArgumentError`, and an indefinite diagonal falls
back to LinearAlgebra's general path.

```julia
using Colmat, LinearAlgebra
A = Hermitian(FaerMatrix([4.0 1; 1 3]))
try
    eigvals(A, Hermitian(FaerMatrix([1.0 0; 0 -1])))
catch e
    @assert e isa PosDefException && e.info == 2
end
@assert eigvals(A, FaerDiagonal([1.0, 2.0])) ≈ eigvals(Hermitian([4.0 1; 1 3]), Diagonal([1.0, 2.0]))
close(parent(A))
```

## Generalized EVD for general matrix pairs

```julia
using Colmat, LinearAlgebra
A = FaerMatrix([0.0 -1; 1 0])
B = FaerMatrix([2.0 0; 0 1])
FaerTeam(2) do team
    workspace = FaerScratch(eigen, A, B; team=team)
    F = eigen(A, B; scratch=workspace, team=team)
    @assert Matrix(A) * F.vectors ≈ Matrix(B) * F.vectors * Diagonal(F.values)
    reserve!(workspace, eigvals, A, B; team=team)
    @assert eigvals(A, B; scratch=workspace, team=team) ≈ F.values
    close(workspace)
end
```

General `FaerMatrix` pairs support four types and scalar promotion, with complex
values and normalized right eigenvectors. The workspace includes writable input
copies and packed real outputs as needed. `scratch_requirements` and `reserve!`
support eigen/eigvals and their bang aliases. Bang calls use the options for the
eigen pass, then compute a separate sequential QZ for staged S/T writeback;
that QZ workspace is outside this query. Sorting, output arrays and writeback
allocate separately.

Native output is returned without post-processing. The pinned faer revision
returns finite vectors for repeated finite and infinite eigenvalues
([faer #217](https://github.com/tensor4all/faer/issues/217), fixed upstream), so
Colmat no longer recomputes nullspaces. Regression tests cover identity pencils,
defective Jordan blocks (finite and infinite), singular pencils with
indeterminate α=β=0 pairs and pencils scaled near the floating-point range
limits. Defective pencils yield finite but linearly dependent vectors, as in
LAPACK.

## Homogeneous generalized eigenvalues

```julia
using Colmat, LinearAlgebra
A = FaerMatrix([2.0 0 0; 0 3 0; 0 0 4])
B = FaerMatrix([1.0 0 0; 0 0 0; 0 0 0])
F = eigen(A, B; homogeneous=true)
@assert count(iszero, F.β) == 2
@assert all(isfinite, F.α) && all(isfinite, F.β)
@assert Matrix(A) * F.vectors * Diagonal(F.β) ≈
        Matrix(B) * F.vectors * Diagonal(F.α)
values = eigvals(A, B; homogeneous=true)
@assert values.α ≈ F.α && values.β ≈ F.β
```

General native pairs accept `homogeneous=true`: `eigen` returns independent Julia
arrays `(α, β, vectors)`, while `eigvals` returns `(α, β)`. Four types, promotion,
`scratch=` and `team=` are supported. The default homogeneous order is the native
order, with no division or sorting. An explicit `sortby` uses α/β to select a
common permutation of both arrays and the vectors. Ordinary calls still return
`GeneralizedEigen` or quotient eigenvalues in their usual sorted order.

A nonzero α with β=0 denotes infinity. The pair α=β=0 remains indeterminate;
values-only homogeneous queries preserve it, while quotient queries return NaN.
The homogeneous residual above stays meaningful for infinite eigenvalues. This option is for general pairs;
self-adjoint wrappers and destructive eigen routines keep their current APIs.

## Left generalized eigenvectors

```julia
using Colmat, LinearAlgebra
A = FaerMatrix(ComplexF64[1 -3im; 2 4])
B = FaerMatrix(ComplexF64[2 0.2; 0 1])
FaerTeam(2) do team
    workspace = FaerScratch(eigen, A, B; team=team)
    F = eigen(A, B; side=:left, scratch=workspace, team=team)
    @assert F.vectors' * Matrix(A) ≈ Diagonal(F.values) * F.vectors' * Matrix(B)
    H = eigen(A, B; side=:left, homogeneous=true, scratch=workspace, team=team)
    @assert Diagonal(H.β) * H.vectors' * Matrix(A) ≈
            Diagonal(H.α) * H.vectors' * Matrix(B)
    close(workspace)
end
```

`side=:right` remains the default. General native pairs also accept `side=:left`
to compute normalized left eigenvectors through faer's left-vector driver,
using four scalar types, promotion, scratch and explicit teams. Sorting keeps
vectors aligned with their eigenvalues or α/β pairs. The returned `vectors`
field contains the selected side. Left-only quotient output is a named tuple
`(values,vectors)`, rather than a factorization for subsequent solves. Requirements are common to the two one-sided
modes in the current backend; the same `FaerScratch(eigen, A, B; team=team)`
works for either. The left-only driver result is returned directly; the pinned
faer revision preserves the spectrum when only left vectors are requested
([faer #219](https://github.com/tensor4all/faer/issues/219), fixed upstream), so
no adjoint-pencil recomputation or extra copies occur. Simultaneous output is available with `side=:both` below. Side selection for
single matrices is described below; destructive routines retain their current API.

## Simultaneous left and right generalized eigenvectors

```julia
using Colmat, LinearAlgebra
A = FaerMatrix([1.0 -3; 2 4])
B = FaerMatrix([2.0 0.2; 0 1])
FaerTeam(2) do team
    workspace = FaerScratch(eigen, A, B; side=:both, team=team)
    F = eigen(A, B; side=:both, scratch=workspace, team=team)
    @assert Matrix(A) * F.right ≈ Matrix(B) * F.right * Diagonal(F.values)
    @assert F.left' * Matrix(A) ≈ Diagonal(F.values) * F.left' * Matrix(B)
    H = eigen(A, B; side=:both, homogeneous=true, scratch=workspace, team=team)
    @assert Matrix(A) * H.right * Diagonal(H.β) ≈
            Matrix(B) * H.right * Diagonal(H.α)
    @assert Diagonal(H.β) * H.left' * Matrix(A) ≈
            Diagonal(H.α) * H.left' * Matrix(B)
    close(workspace)
end
```

`side=:both` computes the two sides in one native generalized EVD call. It
returns `(values,left,right)` for quotient values or `(α,β,left,right)` with
`homogeneous=true`. Four scalar types, promotion, sorting and explicit teams
are supported. Both sets of columns are independently normalized and share the
same eigenvalue order; biorthogonal scaling is not imposed. Queries and
constructors must include `side=:both`: real input needs an extra packed matrix
inside the caller workspace. `reserve!(workspace, eigen, A, B; side=:both, team=team)`
can enlarge an existing buffer. Julia output arrays and exceptional repeated-value
SVD recovery allocate separately. Inputs are preserved.

## Left and simultaneous eigenvectors of one matrix

```julia
using Colmat, LinearAlgebra
A = FaerMatrix([1.0 -3; 2 4])
FaerTeam(2) do team
    workspace = FaerScratch(eigen, A; side=:both, team=team)
    F = eigen(A; side=:both, scratch=workspace, team=team)
    @assert Matrix(A) * F.right ≈ F.right * Diagonal(F.values)
    @assert F.left' * Matrix(A) ≈ Diagonal(F.values) * F.left'
    G = eigen(A; side=:left, scratch=workspace, team=team)
    @assert G.vectors' * Matrix(A) ≈ Diagonal(G.values) * G.vectors'
    close(workspace)
end
```

General native matrices accept `side=:left/:right/:both` for four scalar types.
The default right result remains `Eigen`; left-only returns `(values,vectors)`,
and simultaneous output returns `(values,left,right)` as named tuples. These
left output containers carry eigenvectors without exposing ordinary right-vector
factorization solve/reconstruction operations. Generalized left-only quotient
output follows the same named-tuple convention; homogeneous output is unchanged.

The left columns are normalized and satisfy the adjoint equations above. Both
sides are computed in one native EVD call, then share the same sorting
permutation. Query or construct workspace with the requested side; real
simultaneous output includes two packed matrices. Output arrays and sorting
allocate independently. Explicit teams are propagated through the EVD driver.
Self-adjoint wrappers and destructive routines retain their current APIs.

## Reusable workspace for saved SVD/EVD solves

```julia
using Colmat, LinearAlgebra
A0=Float32[3 0;0 0.1;0 0]
F=svd(FaerMatrix(A0);native=true)
B=ones(ComplexF64,3,2)
X=zeros(ComplexF64,2,2)
team=FaerTeam(2)
work=FaerScratch(ldiv!,X,F,B;rtol=0.2,team)
ldiv!(X,F,B;rtol=0.2,scratch=work,team)
@assert X≈ComplexF64[1/3 1/3;0 0]
C=ones(ComplexF64,2,2)
Y=zeros(ComplexF64,2,3)
reserve!(work,rdiv!,Y,C,F;rtol=0.2,team)
rdiv!(Y,C,F;rtol=0.2,scratch=work,team)
@assert Y≈ComplexF64[1/3 0 0;1/3 0 0]
close(work)
close(team)
close(F)
H0=[4.0 1;1 3]
E=eigen(Hermitian(FaerMatrix(H0));native=true)
work=FaerScratch(inv,E)
inverse=inv(E;scratch=work)
@assert Matrix(inverse)≈inv(H0)
close(inverse)
close(work)
close(E)
```

`scratch_requirements` accepts the same operand order for backslash, `/`,
`ldiv!`, `rdiv!` and `inv`. Optional tolerances are validated without changing
capacity; scalar promotion changes the coefficient/projection buffer sizes.
`reserve!` grows explicitly, and operations reject insufficient/already-closed
resources before destination writeback. Factor construction remains separate.

## Reuse workspace for generalized Hessenberg and QZ construction

```julia
using Colmat, LinearAlgebra
A = FaerMatrix([4.0 1; 2 3])
B = FaerMatrix([2.0 0; 0 2])
team = FaerTeam(2)
work = FaerScratch(hessenberg, A, B; team)
H = hessenberg(A, B; scratch=work, team)
reserve!(work, schur, H; team)
F = schur(H; scratch=work, team)
@assert Matrix(F.Q*F.S*F.Z') ≈ Matrix(A)
@assert Matrix(F.Q*F.T*F.Z') ≈ Matrix(B)
reserve!(work, schur!, A, B; team)
G = schur!(copy(A), copy(B); scratch=work, team)
@assert isapprox(sort(real.(G.values)), sort(real.(F.values)))
close(work)
close(team)
```

`schur`/`hessenberg` on matrix pairs, their staged bang methods and
`schur(H)`/`schur!(H)` accept `scratch` and `team`. Query the same operation,
source and team with `scratch_requirements`; computations reject insufficient
capacity/alignment instead of growing the buffer. Native metadata is checked
before scalar promotion. Views, mixed native scalars and a native matrix paired
with a Julia matrix are supported. Resource and native failures precede bang
writeback; existing factor owners and views are preserved on successful updates.

The workspace holds QR coefficients, projection storage, generalized Hessenberg
work, QZ temporary homogeneous columns and the declared QZ kernel scratch
(including the existing correction for faer #260). Source copies, owned
factor outputs, staging and Julia conversion/output arrays allocate separately.
The selected faer QR can still allocate its deep Householder partial buffer;
caller control of that buffer awaits faer #267. This is not a whole-API
zero-allocation or scaling claim. The actual scoped team token reaches QR,
Householder apply, projection, generalized Hessenberg and QZ kernels.

## Single-matrix Hessenberg and Schur construction resources

Single-matrix reductions and Schur iteration from a saved Hessenberg factor
accept reusable scratch and a scoped team. Queries and `reserve!` use the same
operation and team as execution. Owned factors, view copies, Julia eigenvalue
arrays and staged bang results allocate separately. The selected faer may also
allocate its internal deep Householder partial buffer; external control awaits
faer #267. These interfaces do not imply a whole-call allocation or scaling bound.

```julia
using Colmat, LinearAlgebra
A = FaerMatrix(ComplexF64[4 1+im 2; 2-im 3 0; 1 0 5])
team = FaerTeam(2)
work = FaerScratch(hessenberg, A; team)
H = hessenberg(A; scratch=work, team)
reserve!(work, schur, H; team)
F = schur(H; scratch=work, team)
@assert Matrix(F) ≈ Matrix(A)
reserve!(work, schur!, A; team)
G = schur!(copy(A); scratch=work, team)
@assert Matrix(G) ≈ Matrix(A)
close(G)
close(F)
close(H)
close(work)
close(team)
close(A)
```

## Saved Schur reordering resources

`ordschur` and `ordschur!` accept reusable scratch for native selection flags
and block swaps. Working factors and Julia selection/eigenvalue/conversion
vectors allocate separately. A supplied team is validated and lends its token;
the selected faer reorder kernels currently perform sequential work, so no
parallel speedup or whole-call allocation bound is implied.

```julia
using Colmat, LinearAlgebra
A = FaerMatrix([1.0 0.2 0.1; 0 2 0.3; 0 0 3])
F = schur(A)
team = FaerTeam(2)
work = FaerScratch(ordschur, F; team)
selected = Bool[false, false, true]
G = ordschur(F, selected; scratch=work, team)
@assert real(G.values[1]) ≈ 3
@assert Matrix(G) ≈ Matrix(A)
H = copy(F)
reserve!(work, ordschur!, H; team)
ordschur!(H, selected; scratch=work, team)
@assert Matrix(H) ≈ Matrix(A)
close(H)
close(G)
close(F)
close(work)
close(team)
close(A)
```

## Eigenvectors from saved Schur factors

`eigvecs(F)` and `eigen(F)` reuse saved native Schur factors with caller scratch
and an actual scoped team. Queries cover faer's declared kernel temporary space
and packed real output; the returned complex matrix and Julia eigenvalue copies
allocate separately. Extreme Schur entry magnitudes can still trigger an internal
scaled copy in selected faer, tracked in [faer #270](https://github.com/tensor4all/faer/issues/270).
No whole-call allocation or parallel scaling bound is implied.

```julia
using Colmat, LinearAlgebra
A = FaerMatrix([1.0 0.2 0.1; 0 2 0.3; 0 0 3])
F = schur(A)
team = FaerTeam(2)
work = FaerScratch(eigvecs, F; team)
V = eigvecs(F; scratch=work, team)
@assert Matrix(A)*Matrix(V) ≈ Matrix(V)*Diagonal(F.values)
reserve!(work, eigen, F; team)
E = eigen(F; scratch=work, team)
@assert Matrix(A)*Matrix(E.vectors) ≈ Matrix(E.vectors)*Diagonal(E.values)
close(E.vectors)
close(V)
close(work)
close(team)
close(F)
close(A)
```
