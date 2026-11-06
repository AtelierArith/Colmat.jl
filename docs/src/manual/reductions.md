# 反射・回転・帯行列への縮約

Householder、Jacobi、三重対角化、二重対角化を扱います。

## Givens 回転

`LinearAlgebra.Givens` と消去後の値を返します。

```julia
using Colmat, LinearAlgebra

A = FaerMatrix([3.0 2; 4 1])
G, r = givens(A, 1, 2, 1)
B = G * A
@assert B[1, 1] ≈ r
@assert abs(B[2, 1]) < 1e-12
@assert Matrix(A) == [3.0 2; 4 1]
lmul!(G, A)                       # faer の Jacobi カーネルで更新
@assert Matrix(A) ≈ Matrix(B)
rmul!(A, G')                      # 右からの随伴回転にも対応

v = FaerVector([3.0, 4.0, 5.0])
H, _ = givens(v, 1, 2)
@assert Vector(H * v) ≈ [5.0, 0.0, 5.0]
V = view(A, 2:-1:1, :)
lmul!(G, V)                       # ビューは native の一時結果を書き戻す
```

4数値型の行列・ベクトル・範囲ビューに対応します。右からの回転には `FaerRow` も
使えます。`*` は型を昇格した独立した native 配列を返し、`lmul!`・`rmul!` は
係数を出力の型に変換してから更新します。変換に失敗すれば出力は変わりません。
回転の cosine は実数が必要です。生成対象の2要素は有限値に限定しています。
スカラー形式の `givens(f,g,i,j)` は Julia 標準の実装です。
独立した Jacobi オブジェクトは、このページの Jacobi rotations を参照してください。

Householder 反射は `householder(x)` または `FaerHouseholder(x)` で生成できます。

## Householder 反射と列

faer と同じ `H = I - v*v'/tau` の規約で、`tau` は実数の分母です。

```julia
using Colmat, LinearAlgebra

x = FaerVector([3.0, 4.0, 5.0])
H = householder(x)
y = H * x
@assert Vector(y) ≈ [H.beta, 0.0, 0.0]
@assert Matrix(H)' ≈ Matrix(H)
@assert Matrix(H) * Matrix(H) ≈ Matrix{Float64}(I, 3, 3)
@assert Matrix(H) ≈ I - Vector(H.v) * Vector(H.v)' / H.tau

A = FaerMatrix([1.0 2; 3 4; 5 6])
B = H * A                        # 反射の密行列を作らずに適用
lmul!(H, A)
@assert Matrix(A) ≈ Matrix(B)
C = FaerMatrix(zeros(3, 2))
mul!(C, H, A, 2.0, 0.0)
@assert Matrix(C) ≈ 2Matrix(H) * Matrix(A)
R = FaerRow([1.0, 2.0, 3.0])
rmul!(R, H)
```

生成は入力をコピーしてスケーリングしてから行い、4数値型を保持します。
入力は有限値に限定し、空の入力と末尾がゼロの場合は恒等反射になります。
`H.v` は独立した native ベクトル、`H.beta`・`H.tau`・`H.input_norm`・
`H.head_with_beta_inv` は faer の生成情報です。`tau` は LAPACK の乗数とは異なります。
`H'`・`transpose(H)`・`conj(H)`・`inv(H)`・コピーは独立した所有者を返します。
`H * A`・`A * H`・`H \ b`、左右の in-place 適用、3引数・5引数の `mul!` に対応します。
通常の Julia strided 配列も使え、in-place の出力は4数値型に限定します。
係数・生成情報を出力型に変換できない場合は更新前に失敗します。ビューは一時結果を
書き戻します。`Matrix(H)` と反射同士の積は密行列を作ります。
単一反射・block／sequence の左右適用は `scratch=` と `team=` に対応します。
必要量照会・reserve は以下の API を使います。block／sequence は次の所有型で扱います。

複数の反射は `FaerHouseholderSequence` に compact な basis／factor として保持できます。

```julia
using Colmat, LinearAlgebra

V = [1.0 0; 0.3 1; 0.4 0.5]
tau = [sum(abs2, V[:, j]) / 2 for j in axes(V, 2)]
Q = FaerHouseholderSequence(V, tau; blocksize=2)
A = FaerMatrix([1.0 2; 3 4; 5 6])
B = Q * A
@assert Matrix(B) ≈ Matrix(Q) * Matrix(A)
@assert Matrix(Q' * B) ≈ Matrix(A)
@assert Matrix(Q \ B) ≈ Matrix(A)

restored = FaerHouseholderSequence(Q.basis, Q.factor)
@assert Matrix(restored) ≈ Matrix(Q)
small_blocks = reblock(Q, 1)      # faer の factor upgrade を使って再構築
@assert Matrix(small_blocks) ≈ Matrix(Q)
reblock!(Q, 3)
@assert Q.blocksize == 3
```

```julia
using Colmat, LinearAlgebra

V = [1.0 0; 0.3 1; 0.4 0.5]
tau = [sum(abs2, V[:, j]) / 2 for j in axes(V, 2)]
Q = FaerHouseholderSequence(V, tau; blocksize=2)
A = [1.0 2; 3 4; 5 6]
expected = Matrix(Q) * A
team = FaerTeam(2)
work = FaerScratch(lmul!, Q, A; team)
lmul!(Q, A; scratch=work, team)
@assert A ≈ expected
reserve!(work, ldiv!, Q, A; team)
ldiv!(Q, A; scratch=work, team)
@assert A ≈ [1.0 2; 3 4; 5 6]
close(work)
close(team)
```

`lmul!/rmul!/mul!/ldiv!/rdiv!` で作業領域を再利用できます。左右積の照会は
演算と同じ引数順序を使い、`mul!` の照会は alpha/beta と型昇格も反映します。
容量・alignment 不足や閉じた resource は出力更新前に拒否します。native kernel は
caller-owned team を借用します。結果の staging・係数の型変換・Julia 配列変換は
scratch の外で割り当てます。sequence 構築・reblock の team は指定できます。

`Q = H₁*H₂*…*Hₖ` の順序で、m×k の basis には k≤m が必要です。
対角は暗黙の 1、上三角は無視し、strict lower の値だけを使います。
有限の `tau[j]` は `vⱼ'vⱼ/2` と一致する必要があります。`Inf` は恒等反射を表し、
その列の未使用部分はゼロに正規化します。`blocksize=k` なら単一ブロックです。
`Q.basis`・`Q.factor` は独立した native 行列、`Q.tau` は Julia の実数ベクトルです。
factor は blocksize×k で、各ブロックの上三角を連結した faer の形式です。

`Q'`・`transpose(Q)`・`conj(Q)` は独立した所有者を返します。これらの `.basis`・`.factor`
は変換前の格納表現を返し、`.transposed`・`.conjugated` が適用方向を表します。
積・左右の in-place 適用・`mul!`・左右の求解は密な Q を作りません。
`Matrix(Q)` と演算子同士の積は明示的に密行列を作ります。
sequence の構築と `reblock/reblock!` は `team=` を受け取ります。
factor 行列からの構築は `blocksize=` で変更後のサイズを指定でき、`prev_block_size=`
（既定は元・先 blocksize の gcd）が両方を割り切る場合に保存済みの上三角 block を再利用します。
reblock は basis/tau を Julia に戻さず native owner から構築し、変換方向と失敗時の旧 owner を保持します。

```julia
using Colmat, LinearAlgebra
V = [1.0 0 0; 0.2 1 0; 0.3 0.4 1; 0.1 -0.2 0.3]
tau = [sum(abs2, V[:,j])/2 for j in 1:3]
team = FaerTeam(2)
Q = FaerHouseholderSequence(V, tau; blocksize=2, team)
R = reblock(Q, 4; prev_block_size=2, team)
@assert Matrix(R) ≈ Matrix(Q)
S = FaerHouseholderSequence(Q.basis, Q.factor; blocksize=4, prev_block_size=2, team)
@assert Matrix(S) ≈ Matrix(Q)
reblock!(R, 3; team)
@assert Matrix(R) ≈ Matrix(Q)
close(S); close(R); close(Q); close(team)
```

native upgrade は scratch 引数・必要量照会を持たず、深い並列分岐の内部 buffer を毎回確保します。
[faer #93 の実行済み MWE](https://github.com/tensor4all/faer/issues/93#issuecomment-6648218975)
と修正案で追跡し、構築/reblock の caller-owned scratch は未完了です。

`nonzeros(S)` は格納値の変更に使えます。要素の挿入、`dropzeros!`、異なる構造の

## Standalone Jacobi rotations

`rotg` returns faer's 2×2 unitary rotation, using its native sign convention:

```julia
using Colmat, LinearAlgebra
J, r = rotg(3.0, 4.0)
@assert J * [3.0, 4.0] ≈ [r, 0.0]
@assert Matrix(J') * Matrix(J) ≈ Matrix{Float64}(I, 2, 2)
A = FaerMatrix([1.0 2; 3 4])
lmul!(J, A)
@assert Matrix(A) ≈ Matrix(J) * [1.0 2; 3 4]
G = LinearAlgebra.Givens(J, 2, 4) # embed the same rotation in a larger array
```

`make_givens(p,q)` exposes faer's real constructor. `transpose`, `conj` and
`adjoint` use its native transforms. `apply_on_the_left_2x2` and
`apply_on_the_right_2x2` accept four entries and return them in row order.
Standalone rotations require an axis of length two; embedded `Givens` rotations
use the existing native array kernels.

## Hermitian tridiagonalization and band eigensolvers

`hessenberg` on a native Hermitian (or real symmetric) matrix calls faer's
self-adjoint tridiagonalization. `tridiagonalize` also accepts ordinary arrays:

```julia
using Colmat, LinearAlgebra
A = FaerMatrix([4.0 1 2; 1 3 1; 2 1 5])
F = hessenberg(Hermitian(A))
@assert F.H isa SymTridiagonal
@assert Matrix(F) ≈ Matrix(A)
E = eigen(F) # reuse the reduction and call the native tridiagonal eigensolver
@assert Matrix(A) * Matrix(E.vectors) ≈ Matrix(E.vectors) * Diagonal(E.values)
S = SymTridiagonal(FaerVector([1.0, 2, 3]), FaerVector([0.2, 0.4]))
@assert eigvals(S) ≈ tridiagonal_eigvals([1.0, 2, 3], [0.2, 0.4])
```

By default `FaerTridiagonalization` owns explicit native `Q` and a real `SymTridiagonal` H;
complex phases are absorbed into Q. `tridiagonalize!` and `hessenberg!` on native
self-adjoint wrappers stage the reduction and then store explicit H in the input.
They preserve the input on failure. Compact Householder storage stays internal.

`tridiagonal_eigen(d,e)` and `tridiagonal_eigvals(d,e)` call faer's dedicated
band kernels directly, treating e as the **subdiagonal** and `conj(e)` as the
superdiagonal. Values are real and ascending; eigenvectors are native matrices.
`recursion_threshold` selects QR versus divide-and-conquer for vectors, and
`sortby` changes ordering. Eigenvalues-only calls use linear coefficient/work
storage and do not form a dense matrix. All four native types, complex
subdiagonals and scaling for extreme finite inputs are supported.

## Bidiagonal reduction and SVD

`bidiagonalize` uses faer's native reduction for square or rectangular matrices:

```julia
using Colmat, LinearAlgebra
A = [1.0 2; 3 4; 5 6]
F = bidiagonalize(A)
@assert Matrix(F) ≈ A
@assert Matrix(F.U) * Matrix(F.B) * Matrix(F.V)' ≈ A
S = svd(F) # reuse the reduction and call the dedicated bidiagonal SVD
@assert Matrix(S.U) * Diagonal(S.S) * Matrix(S.Vt) ≈ A
B = Bidiagonal(FaerVector([3.0, -2]), FaerVector([0.5]), :U)
@assert svdvals(B) ≈ bidiagonal_svd([3.0, -2], [0.5]).S
```

The default factors are thin; `bidiagonalize(A; full=true)` retains complete
bases for `svd(F; full=true)`. B is upper for tall/square inputs and lower for
wide inputs. `bidiagonalize!` stages its result before replacing a native input
by the rectangular embedding of B. Compact reflectors stay internal.

Real power-of-two scaling is restored componentwise, including complex inputs
near the floating-point exponent limits:

```julia
using Colmat, LinearAlgebra
A = ComplexF64[1 0.4-0.1im; 0.2+0.3im 1]
scale = 2.0^-1000
F = bidiagonalize(A .* scale)
try
    @assert A ≈ Matrix(F.U) * (Matrix(F.B) ./ scale) * Matrix(F.V)'
finally
    close(F)
end
```

`bidiagonal_svd(d,e; uplo=:U)` supports real or complex diagonals and either
selected offdiagonal, returning native U/Vt and nonnegative descending real S.
`recursion_threshold` selects the native small/large SVD paths. Inputs remain
unchanged. The public native bidiagonal SVD computes dense factors/workspace,
including when `svdvals` requests only values.

## Reusing bidiagonal SVD workspace

`bidiagonal_svd`, saved reduction `svd/svdvals`, and native `Bidiagonal`
dispatch accept `scratch=` and `team=`. The query includes coefficient copies
and the declared kernel workspace at the actual team shape; U/V are written
directly into preserved staging. Saved-factor back-transform products also use
the team, with their own work/output storage. Julia conversions and result
ownership allocate separately. Reduction construction resources are described below.

```julia
using Colmat, LinearAlgebra
A = ComplexF64[1 2; 3+im 4; 5 6-im]
F = bidiagonalize(A)
team = FaerTeam(2)
work = FaerScratch(svd, F; team, recursion_threshold=2)
try
    E = svd(F; scratch=work, team, recursion_threshold=2)
    @assert Matrix(E.U)*Diagonal(E.S)*Matrix(E.Vt) ≈ A
    reserve!(work, svdvals, F; team)
    @assert svdvals(F; scratch=work, team) ≈ svdvals(A)
    close(E.U)
    close(E.Vt)
finally
    close(work)
    close(team)
    close(F)
end
```

## Reusing bidiagonal reduction workspace

`bidiagonalize` and the native matrix's staged `bidiagonalize!` accept `scratch`
and `team`, with matching query/FaerScratch/reserve. Workspace holds the canonical
work matrix and compact factors; reduction and both basis applications reuse
one kernel region and the actual scoped team token. U/V are formed in preserved
Julia staging, borrowing the compact basis/factors without owning copies.
Conversion, returned owners and bang writeback allocate separately. The native
deep Householder upgrade buffer remains subject to [faer #93](https://github.com/tensor4all/faer/issues/93);
declared kernel scratch does not imply whole-call zero allocations.

```julia
using Colmat, LinearAlgebra
A = FaerMatrix(ComplexF64[1 2; 3+im 4; 5 6-im])
team = FaerTeam(2)
work = FaerScratch(bidiagonalize, A; full=true, team)
try
    F = bidiagonalize(A; full=true, scratch=work, team)
    @assert Matrix(F) ≈ Matrix(A)
    reserve!(work, svd, F; team)
    E = svd(F; scratch=work, team)
    @assert Matrix(E.U)*Diagonal(E.S)*Matrix(E.Vt) ≈ Matrix(A)
    close(E.U)
    close(E.Vt)
    close(F)
finally
    close(work)
    close(team)
    close(A)
end
```

## Bidiagonal parallel policy

Use `FaerParThreshold` to select faer's derived, fixed or never-parallel policy.
The default remains derived. Reuse the same policy for workspace planning and
computation; changing it may change the required capacity. A fixed floor counts
elements of the reduction's updated trailing block, while `:never` also keeps
explicit U/V formation sequential. Supplied teams are validated for every policy.

```julia
using Colmat, LinearAlgebra
A = FaerMatrix(ComplexF64[1 2; 3+im 4; 5 6-im])
original = Matrix(A)
team = FaerTeam(2)
policy = FaerParThreshold(:never)
work = FaerScratch(bidiagonalize, A; team, par_threshold=policy)
try
    F = bidiagonalize(A; scratch=work, team, par_threshold=policy)
    @assert Matrix(F) ≈ original
    close(F)
    policy = FaerParThreshold(0)
    reserve!(work, bidiagonalize!, A; team, par_threshold=policy)
    F = bidiagonalize!(A; scratch=work, team, par_threshold=policy)
    @assert Matrix(F) ≈ original
    @assert Matrix(A)[1:2, :] ≈ Matrix(F.B)
    @assert iszero(Matrix(A)[3, :])
    close(F)
finally
    close(work)
    close(team)
    close(A)
end
```

## Reusing tridiagonal eigensolver workspace

Dedicated tridiagonal eigenvectors and eigenvalues accept `scratch` and `team`.
The query includes the scaled band and native temporary eigenvalues. Coefficient
conversion, result storage and sorting allocate separately. Saved tridiagonal
factors and native `SymTridiagonal` inputs forward these resources; saved-factor
Q back-transformation also uses the supplied team. Tridiagonal reduction
construction accepts its own reusable workspace as described below.

```julia
using Colmat, LinearAlgebra
d = ComplexF64[1, 2, 3]
e = ComplexF64[0.25 + 0.125im, 0.25 - 0.125im]
team = FaerTeam(2)
work = FaerScratch(tridiagonal_eigen, d, e; team, recursion_threshold=2)
try
    F = tridiagonal_eigen(d, e; scratch=work, team, recursion_threshold=2)
    X = Matrix(F.vectors)
    A = Matrix(Tridiagonal(e, d, conj.(e)))
    @assert A*X ≈ X*Diagonal(F.values)
    close(F.vectors)
    reserve!(work, tridiagonal_eigvals, d, e; team)
    values = tridiagonal_eigvals(d, e; scratch=work, team)
    @assert values ≈ eigvals(Hermitian(A))
finally
    close(work)
    close(team)
end
```

## Extreme-scale complex tridiagonal reduction

Tridiagonal reduction restores its real power-of-two scale componentwise, so a
finite complex result does not depend on squaring a tiny or huge complex divisor.
Check reconstruction after removing the scale to keep the residual meaningful.

```julia
using Colmat, LinearAlgebra
A = ComplexF64[1 0.2-0.3im; 0.2+0.3im 1]
scale = 2.0^1000
F = tridiagonalize(A .* scale)
try
    Q = Matrix(F.Q)
    @assert A ≈ Q * (Matrix(F.H) ./ scale) * Q'
finally
    close(F)
end
```

## Tiny relative complex tridiagonal phases

A small nonzero complex off-diagonal can require a finite unit phase even when
it is negligible in a global residual. Phase normalization uses real norms,
without squaring a complex-typed tiny divisor. Check that component separately:

```julia
using Colmat, LinearAlgebra
x = 1e-300
A = ComplexF64[1 -im*x; im*x 2]
F = tridiagonalize(A)
try
    Q = Matrix(F.Q)
    restored = Q * Matrix(F.H) * adjoint(Q)
    @assert all(isfinite, Q)
    @assert adjoint(Q)*Q ≈ Matrix{ComplexF64}(I, 2, 2)
    @assert abs(real(restored[2,1])/x) <= 128eps(Float64)
    @assert abs(imag(restored[2,1])/x-1) <= 128eps(Float64)
finally
    close(F)
end
```

Native Householder reduction deliberately discards subnormal heads in its
working scale; componentwise retention of such relative subnormal entries is
not promised. Whole-matrix scaling protects globally subnormal inputs. The
separate scalar magnitude underflow is reported in
[faer #274](https://github.com/tensor4all/faer/issues/274).

## Reusing tridiagonal reduction workspace

`tridiagonalize` and its staged bang form accept `scratch` and `team`, including
native Hermitian/Symmetric `hessenberg` dispatch. Workspace covers the canonical
work matrix, compact coefficients and a region reused by reduction and explicit
Q formation. Native Q and basis copies are avoided; Julia conversion, returned
owners and bang writeback still allocate separately.

```julia
using Colmat, LinearAlgebra
A = FaerMatrix(ComplexF64[4 1-0.5im; 1+0.5im 3])
H = Hermitian(A)
team = FaerTeam(2)
work = FaerScratch(tridiagonalize, H; team)
try
    F = hessenberg(H; scratch=work, team)
    @assert Matrix(F) ≈ Matrix(H)
    close(F)
    reserve!(work, tridiagonalize!, A; team)
    F = tridiagonalize!(A; scratch=work, team)
    @assert Matrix(A) ≈ Matrix(F.H)
    close(F)
finally
    close(work)
    close(team)
    close(A)
end
```

## Tridiagonal parallel policy

`tridiagonalize[!]` and native Hermitian/real Symmetric `hessenberg[!]` accept
`par_threshold=FaerParThreshold()`. Fixed thresholds count `rem*rem÷2` elements
for the updated triangle. Explicit Q applications keep their own parallel rule;
`FaerParThreshold(:never)` makes both reduction and Q formation sequential.
Use the same policy for workspace planning and execution. A supplied team must
still be open, including for empty matrices and Never.

```julia
using Colmat, LinearAlgebra
A = FaerMatrix(ComplexF64[4 1-im 2; 1+im 5 0.5im; 2 -0.5im 6])
original = Matrix(A)
H = Hermitian(A)
team = FaerTeam(2)
policy = FaerParThreshold(:never)
work = FaerScratch(hessenberg, H; team, par_threshold=policy)
try
    F = hessenberg(H; scratch=work, team, par_threshold=policy)
    @assert Matrix(F.Q)*Matrix(F.H)*Matrix(F.Q)' ≈ original
    close(F)
    policy = FaerParThreshold(0)
    reserve!(work, hessenberg!, H; team, par_threshold=policy)
    F = hessenberg!(H; scratch=work, team, par_threshold=policy)
    @assert Matrix(F.Q)*Matrix(F.H)*Matrix(F.Q)' ≈ original
    @assert Matrix(A) ≈ Matrix(F.H)
    close(F)
finally
    close(work)
    close(team)
    close(A)
end
```

## Compact tridiagonal reflectors

Use `compact=true` to retain Q as `FaerTridiagonalQ`. Products and in-place
Householder actions apply saved native reflectors and phases without forming a
dense Q. `Matrix(F.Q)` explicitly materializes it. `F.Q.basis`, `.factor` and
`.phases` return independent Julia arrays describing the base representation;
the factor can contain positive infinite denominators denoting identity.
For n rows, the Householder product acts on rows 2:n, followed by the diagonal
phase factor. Transpose/adjoint/conj record orientation and share immutable
native storage; closing one wrapper leaves another open. Copy and deepcopy
make independent native owners. Input arrays and native source owners can be
changed or closed after construction.

```julia
using Colmat, LinearAlgebra
A = ComplexF64[4 1+im 0; 1-im 3 0.5im; 0 -0.5im 2]
team = FaerTeam(2)
work = FaerScratch(tridiagonalize, A; compact=true, team)
try
    F = tridiagonalize(A; compact=true, scratch=work, team)
    @assert F.Q isa FaerTridiagonalQ
    X = ones(ComplexF64, 3, 2)
    reserve!(work, lmul!, F.Q, X; team)
    lmul!(F.Q, X; scratch=work, team)
    @assert X ≈ Matrix(F.Q) * ones(ComplexF64, 3, 2)
    reserve!(work, eigen, F; team)
    E = eigen(F; scratch=work, team)
    @assert A * Matrix(E.vectors) ≈ Matrix(E.vectors) * Diagonal(E.values)
    close(F)
finally
    close(work)
    close(team)
end
```

Construction scratch covers only the reduction kernel in compact mode. The
returned packed matrix, block factors and phases allocate their own storage;
Julia input conversion and staged bang writeback also allocate separately.
Use `compact=true` consistently in `scratch_requirements`, `FaerScratch` and
`reserve!` for construction. Later Q applications have their own scratch/team
contract. Saved-factor `eigen` reuses one workspace for the band eigensolver
and implicit Q back-transformation; its query takes the maximum requirement.
A construction `FaerParThreshold(:never)` governs the reduction and does not
restrict separately requested Q applications. No speedup or worker utilization
is claimed by this storage option.

## Compact bidiagonal bases

`bidiagonalize(A; compact=true)` returns implicit `FaerBidiagonalBasis` U/V
with the same thin shapes as the explicit default. `full=true` retains square
bases. The two wrappers share immutable native packed storage, and each can be
closed independently. Copying a factor duplicates its storage once, with the
new U/V sharing that duplicate. Inputs can be changed or closed afterward.

Products, transpose/adjoint/conj, rectangular `mul!`, and saved-factor SVD apply
reflectors by embedding/projecting operands rather than forming a square Q.
Shape-preserving `lmul!`/`rmul!` require a square basis; rectangular actions use
a correctly shaped separate destination. Since each basis is a partial isometry,
`Q \ A` and `A / Q` apply its adjoint (least-squares/minimum-norm as appropriate).
`Matrix(Q)` explicitly materializes its exact thin/full shape. Independent Julia
arrays `Q.basis` and `Q.factor` expose the base Householder representation;
`Q.offset` gives its leading identity coordinates and `Q.basis_conjugated`
indicates conjugation of that product. The usual orientation flags apply after
embedding and selecting the retained columns. Complex band phases stay in B.

```julia
using Colmat, LinearAlgebra
A = ComplexF64[1 im 2; 3 4-im 5; 6 7 8+im; 9 10 11]
team = FaerTeam(2)
work = FaerScratch(bidiagonalize, A; compact=true, team)
try
    F = bidiagonalize(A; compact=true, scratch=work, team)
    @assert size(F.U) == (4, 3)
    X = ones(ComplexF64, 3, 2)
    Y = zeros(ComplexF64, 4, 2)
    reserve!(work, mul!, Y, F.U, X; team)
    mul!(Y, F.U, X; scratch=work, team)
    @assert Y ≈ Matrix(F.U) * X
    reserve!(work, svd, F; team)
    S = svd(F; scratch=work, team)
    @assert Matrix(S.U) * Diagonal(S.S) * Matrix(S.Vt) ≈ A
    close(F)
finally
    close(work)
    close(team)
end
```

Compact construction scratch covers only the reduction kernel; the returned
packed matrix and factors allocate their own storage. Application scratch
includes operand embedding and the Householder kernel, while Julia input/output
conversion allocates separately. Saved SVD queries the maximum of the band
solver and successive U/V application requirements and reuses one workspace/team.
A full SVD requires a reduction made with `full=true`, as for explicit factors.
Values-only still uses the existing native full-factor band-SVD path. Construction
parallel policy does not constrain later separately requested applications. No
speedup is claimed; native internal Householder buffers remain subject to faer #267.

## Compact general Hessenberg Q

For a general native square matrix, `hessenberg(A; compact=true)` returns
`FaerCompactHessenberg` with native H and an implicit `FaerHessenbergQ`. The
explicit `FaerHessenberg` default is unchanged. The implicit Q uses saved native
Householder factors acting on rows 2:n. Its `.basis` and `.factor` properties
return independent arrays; `.phases` returns ones because this reduction has no
additional diagonal phase factor. Products, left/right actions, solves and
transpose/adjoint/conjugation apply those factors without forming dense Q.
`Matrix(F.Q)` explicitly materializes it.

```julia
using Colmat, LinearAlgebra
A = FaerMatrix(ComplexF64[4 im 2; 1 3-im 0; 2 1 5])
original = Matrix(A)
team = FaerTeam(2)
work = FaerScratch(hessenberg, A; compact=true, team)
try
    F = hessenberg(A; compact=true, scratch=work, team)
    @assert F.Q isa FaerHessenbergQ
    @assert Matrix(F) ≈ original
    X = ones(ComplexF64, 3, 2)
    reserve!(work, lmul!, F.Q, X; team)
    lmul!(F.Q, X; scratch=work, team)
    @assert X ≈ Matrix(F.Q) * ones(ComplexF64, 3, 2)
    reserve!(work, schur, F; team)
    S = schur(F; scratch=work, team)
    @assert Matrix(S.Z) * Matrix(S.T) * Matrix(S.Z)' ≈ original
    oldQ = F.Q
    savedQ = Matrix(oldQ)
    schur!(F; scratch=work, team)
    @assert F.Q isa FaerMatrix
    @assert Matrix(oldQ) ≈ savedQ
    @assert Matrix(F) ≈ original
    close(oldQ)
    close(S)
    close(F)
finally
    close(work)
    close(team)
    close(A)
end
```

Construction owns the packed reflectors independently of its input. In
`hessenberg!`, H shares the input matrix while Q retains separate packed storage.
Transpose/adjoint/conj wrappers share immutable reflector storage and close
independently; copy/deepcopy duplicate it and preserve the concrete wrapper type.

Saved `schur(F)` starts the Hessenberg QR stage with identity and applies the
saved Q directly to the resulting native Schur basis. It preserves F and avoids
materializing the original Q. `schur!(F)` stages the entire result, updates H,
and replaces the implicit Q with the explicit Schur basis. Its result shares
H and Q with F. A previously retained implicit Q remains a valid independent
wrapper; close it separately. Repeated `schur!` keeps the existing explicit Q
owner, as for the explicit default factor.

Use `compact=true` for construction queries, `FaerScratch` and `reserve!`.
Saved Schur queries the maximum of the QR-stage and implicit-Q application
requirements and reuses that one workspace/team. Packed factors, owned outputs,
Julia conversions, eigenvalue arrays and staged bang writeback allocate
separately. No allocation-free API, performance improvement, or new Hessenberg
tuning is claimed; native internal Householder buffers remain subject to faer #267.

## General Hessenberg reduction controls

Both explicit and compact general Hessenberg factors accept faer's public
`HessenbergParams` controls. `par_threshold=FaerParThreshold()` uses the derived
cost model; an integer floor counts squared trailing order, and `:never` keeps
reduction and explicit Q formation sequential. A supplied team must remain open.
`blocking_threshold=65536` is the upstream default: orders satisfying
`n*n < blocking_threshold` use unblocked reduction. Zero permits the blocked
path at every order; `typemax(UInt)` selects the unblocked path for representable
matrix dimensions. These choices control reduction, not saved Schur iteration.

Use identical options for query, reserve and computation. The bang method writes
H only after successful computation and retains an independent Q. Caller scratch
covers declared kernel temporaries; output owners, packed reflectors and view
copies allocate separately. Compact construction now reads the native input and
returns a checked owned Q directly, without staging either input or H in Julia.

```julia
using Colmat, LinearAlgebra
A = FaerMatrix(ComplexF64[4 im 2; 1 3-im 0; 2 1 5])
original = Matrix(A)
team = FaerTeam(2)
work = FaerScratch(0)
options = (; compact=true, team,
    par_threshold=FaerParThreshold(:never), blocking_threshold=0)
try
    req = scratch_requirements(hessenberg!, A; options...)
    reserve!(work, hessenberg!, A; options...)
    @assert scratch_capacity(work).bytes >= req.bytes
    F = hessenberg!(A; options..., scratch=work)
    @assert F.H === A
    @assert Matrix(F) ≈ original
    close(F)
finally
    close(work)
    close(team)
    close(A)
end
```
