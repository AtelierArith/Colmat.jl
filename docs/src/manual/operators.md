# 線形作用素・反復法

行列を明示しない作用素、前処理、CG・BiCGSTAB・LSMRを使います。

## 反復法の選び方と最初の計算

反復法は近似解を少しずつ改善して `A*x ≈ b`を目指す方法です。大きな疎行列など、分解を保存する費用が大きい問題で役立ちます。残差 `b - A*x`がどれだけ小さくなったかを確認します。

| 方法 | 入力の条件・用途 |
| :--- | :--- |
| `cg` | Hermitian 正定値の正方行列と正定値の前処理 |
| `bicgstab` | 対称とは限らない正方行列 |
| `lsmr` | 長方形も含む最小二乗問題 |

正定値とは、ゼロでない xに対して `real(x' * A * x) > 0`となる性質です。CGは条件を満たすと分かっているときに使ってください。前処理は、解きやすい近似逆行列を作用させて収束を助ける仕組みです。BiCGSTABの `M`は左、`N`は右の前処理です。

`FaerLinearOperator(A)`は行列とベクトルの積を表すオブジェクトです。行列入力は構築時にコピーし、疎行列の係数は疎のまま保持します。元の行列を後で変更しても作用素は変わりません。

返り値の `converged`と残差を確認します。既定の `check=true`では収束失敗時に対応する `FaerCgError`、`FaerBicgstabError`、`FaerLsmrError`を投げ、例外の `result`に最終結果を保持します。`check=false` なら失敗結果を直接返します。`cg!` などは出力先の現在値を初期値にします。

LSMRは通常の残差がゼロにならなくても、最小二乗の条件を満たせば収束します。判定には `norm(M' * A' * (b-A*x))`を使い、絶対・相対許容誤差を独立に適用します。非ゼロの初期値や右前処理を使うと、元の問題の最小ノルム解になるとは限りません。

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

## 作用素の積と作業領域

`mul!(X, op, B; scratch, team)`で出力先を再利用できます。転置・随伴や係数付きの積にも対応します。数値型・向き・係数をそろえて必要量を照会してください。同じ数値型の Julia 配列やネイティブ配列には直接書き込めます。重なる入力や型変換が必要な場合は一時領域を使います。

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
```

## 行列を作らず作用素を組み合わせる

作用素は積、和、転置、随伴を組み合わせられます。`op' * op`は、まず op、次にその随伴を適用する演算です。全要素を持つ行列を作りません。ただし正規方程式は条件数を悪化させることがあるため、最小二乗が目的なら LSMR も検討してください。

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
```

## LU 分解を前処理に使う

`FaerLinearOperator(F)`に保存した正則な正方 LU 因子を渡すと、逆行列を明示的に作らず `F \ x`を適用する作用素になります。元の分解から独立した因子を保持します。

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
```

## Cholesky 分解を前処理に使う

正定値行列の Cholesky 分解から近似逆作用を作れます。CGに使う前処理も正定値である必要があります。元の分解を閉じても、構築済みの作用素は使えます。

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
```

## 完全ピボット LUによる逆作用

完全ピボット LUでは行と列の置換を保存します。作用素はそれらも保持して求解し、転置・随伴でも対応する置換を使います。特異な因子と長方形の因子はこの逆作用には使えません。

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
```

## QR 分解による逆作用・最小ノルム作用

正方 QRの既定は通常の逆作用で、ゼロの対角を持つ Rは拒否します。長方形の QR、または `atol`が非ゼロ・`rtol`を明示した QRでは、小さな特異値を切り捨てた最小ノルム作用を作ります。

構築時に Rの SVDを計算して保存し、適用時には再分解しません。しきい値は `max(atol, rtol*maximum(svdvals(R)))`です。作業領域を構築にも渡す場合は `FaerScratch(FaerLinearOperator, F; atol, rtol, team)`と同じ設定を使います。

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
```

```julia
using Colmat, LinearAlgebra
A0=[1.0 2 0;0 1 1]
F=qr(FaerMatrix(A0))
FaerTeam(2) do team
    scratch=FaerScratch(FaerLinearOperator,F;team)
    M=FaerLinearOperator(F;scratch,team)
    @assert Matrix(M)≈pinv(A0)
    close(M)
    close(scratch)
end
```

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
# rtol を明示すると特異な正方 QR も切り捨てた逆作用として扱う
F=qr(FaerMatrix([1.0 0;0 0]))
M=FaerLinearOperator(F;rtol=1e-12)
@assert Matrix(M)≈[1.0 0;0 0]
```

## LDLᴴ 分解による前処理

Lと Dを保存して求解します。ゼロの対角ピボットは拒否します。不定値（正と負の方向を持つ）行列の逆作用は BiCGSTAB などに使えますが、CGの正定値条件を満たしません。

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
```

## Bunch–Kaufman 分解による前処理

L、Dの 1×1・2×2 ブロック、置換を保持します。2×2 ブロックが正則なら、個々の対角要素がゼロでも使えます。不定値の作用素を CGの前処理には使わないでください。

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
```

## SVD・固有値分解による疑似逆作用

`FaerLinearOperator(F; atol, rtol)`は保存した SVD または自己随伴固有値分解を使います。小さい特異値・固有値を切り捨て、全体の疑似逆行列を作らず適用します。作用素の形状は `reverse(size(F))`です。しきい値は元の因子の精度で決め、負の固有値の符号は保ちます。

```julia
using Colmat, LinearAlgebra
A0 = [3.0 0; 0 1e-6; 0 0]
F = svd(FaerMatrix(A0); native=true)
M = FaerLinearOperator(F; rtol=1e-3)
close(F)
@assert size(M) == (2, 3)
@assert M * ones(3) ≈ [1/3, 0]
@assert Matrix(M) ≈ pinv(A0; rtol=1e-3)
```

## Julia 関数で作用素を定義する

行列を保存せず計算式だけで積を求めたい場合は、`FaerLinearOperator(apply!, adjoint_apply!, m, n; T)`を使います。2 つの関数は通常の Julia 行列 `(Y, X)`を受け取り、順方向と随伴の積で Y 全体を上書きします。複数列の右辺と昇格した数値型にも対応してください。

コールバックは呼び出し元の Julia タスクで同期的に実行します。入出力の受け渡し用配列は再利用するため、関数内で X や Yを保存しないでください。後で値が必要なら `copy` します。出力は有限値で全要素を書き込みます。例外は Rustの境界を戻ってから Juliaに伝えます。

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
```

## 収束履歴と途中停止

`history=true`は各反復の残差を記録します。`callback=state -> ...`は反復番号 `iteration`、独立した近似解 `x`、残差を受け取ります。`true` または `nothing`で続行し、`false`で停止します。意図的な停止では `reason=:stopped`、`converged=false`となり、`check=true`でも例外にはしません。

観測用のコピーと残差再計算には追加の費用がかかります。初期値ですでに収束した場合や更新回数がゼロの場合は記録・コールバックがありません。コールバックが例外を投げた場合、更新形式の出力先は保持します。

3 種類の反復法はいずれも `scratch`と `team`に対応します。必要量の照会は作用素やコールバックを実行しません。最終的な真の残差も確認するため、内部推定だけによる誤った収束は受け入れません。

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
```
