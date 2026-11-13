# 疎 QRと最小ノルム求解

疎 QRの成分・ランク・横長行列・転置求解と作業領域を説明します。

## 疎 QRの因子と最初の計算

QRは `A[:, p] ≈ Q*R`と表す分解です。Qは列どうしが直交する行列、Rは上三角行列、pは列の並べ替えです。最小二乗問題を解く際に使います。

縦長の m×n 入力では `F.Q`は m×nの密な `FaerMatrix`、`F.R`は n×nの疎行列です。`F.pcol`（`F.p`と同じ）は列置換です。取り出した成分は独立し、Fの更新・終了後も使えます。復元は保存因子を使い、元の列順に戻します。

`algorithm`は `:auto`、`:simplicial`、`:supernodal`を選べます。後者は列群をまとめて処理する方式です。まず既定の `:auto`を使ってください。数値構築・求解の作業領域は全方式で指定できます。

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
```

## 転置・随伴の最小ノルム解

未知数の方が多い場合は解が複数あり得ます。その中で `norm(x)`が最小の解を最小ノルム解と呼びます。`transpose(F) \ B`と `adjoint(F) \ B`は対応する問題を保存因子から解きます。複素数では転置と随伴（転置して共役を取る操作）は異なります。

3 引数 `ldiv!`は長方形の因子に対応し、2 引数形式は正方に限ります。右辺の型が広がっても保存因子を再利用するため、元の因子の精度が上がるわけではありません。

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
```

```julia
using Colmat, LinearAlgebra, SparseArrays
A0 = Float32[4 1; 1 3]
A = FaerSparseMatrix(sparse(A0))
F = lu(A; algorithm=:simplicial)
B = ComplexF64[1 + im, 2 - im]
x = F \ B
@assert eltype(x) == ComplexF64
@assert A0 * x ≈ B rtol=1e-6
```

## 自動選択・列群方式の最小ノルム解

`:auto`と `:supernodal`でも同じ成分と転置・随伴求解を使えます。列が独立な場合は保存した Rと反射で解き、密な Q や疑似逆行列全体を作る必要はありません。ランク不足の場合は後述の SVD 経路を使います。

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
```

## 列群方式を明示的に選ぶ

`qr(A; algorithm=:supernodal)` または構造解析の同じ指定で列群方式を選びます。CSC/CSRと 4 数値型に対応します。`ordering=:default`と `:colamd`は COLAMD、`:identity` や明示的な置換も指定できます。順序と設定は数値更新でも保持します。


## 疎 QR 求解の作業領域

すべての方式で `scratch`と `team`を指定できます。転置・随伴、右からの求解、混合型、横長・ランク不足にも対応します。許容誤差とチームを照会・構築・実行で合わせてください。

通常の縦長・正則な Rは疎な求解を使います。必要な場合は保存 Rの SVDと Qの適用に作業領域を使います。出力や型変換は別のメモリです。因子更新後は必要量が変わることがあるため再照会してください。

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
```

## 構造解析を再利用した分解・更新

`factorize(plan, A; scratch, team)`は構造解析を再利用して数値因子を作ります。Aの格納構造は計画と完全に一致する必要があります。返された因子は入力・計画・作業領域から独立しています。

`qr!(F, A; scratch, team)`は成功後に新しい数値因子へ置き換えます。型と構造が一致する入力が必要です。同様の資源指定は疎 LU・Choleskyの構築と更新にも対応します。`check=false`の LU・Choleskyは失敗状態を保存する点に注意してください。構造解析と返り値は作業領域とは別に確保します。疎な要素を順に処理する `:simplicial`の数値カーネルは、チームを渡しても逐次実行です。

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
```

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
```

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
```

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
```

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
```

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
```

## 疎 QRを逆作用として使う

`FaerLinearOperator(F)`は保存因子を使う逆作用を作ります。縦長で列が独立なら、Q や疑似逆行列全体を作らず最小ノルム作用を適用します。転置・随伴、型の昇格、`scratch`と `team`に対応します。ランク不足では密な疑似逆作用の保存が必要となる場合があります。

```julia
using Colmat, LinearAlgebra, SparseArrays
A0 = ComplexF64[2 0; 1im 3; 0 1]
F = qr(FaerSparseMatrix(sparse(A0)); algorithm=:simplicial, ordering=[2, 1])
M = FaerLinearOperator(F)
close(F)
@assert size(M) == (2, 3)
@assert M * ones(ComplexF64, 3) ≈ A0 \ ones(ComplexF64, 3)
@assert adjoint(M) * ones(ComplexF64, 2) ≈ pinv(A0') * ones(ComplexF64, 2)
```

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
```

```julia
using Colmat, LinearAlgebra, SparseArrays
A0=ComplexF64[4 0; im 3; 0 1]
F=qr(FaerSparseMatrix(sparse(A0)); algorithm=:supernodal, ordering=[2,1])
@assert Vector(F.pcol)==[2,1]
@assert Matrix(F.Q)*Matrix(F.R)≈A0[:,[2,1]]
@assert Matrix(F)≈A0
```

## 完全な直交基底を取り出す

`F.Q`と `F.Qthin`は必要な列だけの基底、`F.Qfull`は m×mの完全な直交基底（複素数ではユニタリ基底）を返します。どの方式でも保存した反射から作り、入力を再分解しません。完全な基底は密行列なので、大きな入力では次の作用素形式を検討してください。

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

## Q 全体を作らず適用する

`F.Qoperator`は m×mの `FaerLinearOperator`を返します。積・転置・随伴を保存した反射で計算し、`Matrix(Qop)`を呼ぶと完全な Qを作ります。必要な列だけが欲しい場合は `F.Qthin`を使います。SPQRの `Matrix(F.Q)`と形状が異なる点に注意してください。

作用素は反射を独立して保存するため、Fを閉じても使えます。`mul!(X, Qop, B; scratch, team)`に対応します。`F.prow`は恒等置換で、元の行順を保ちます。

列群方式の調整には `supernodal_threshold`（自動選択のみ）、`relax`、`blocking_threshold`があります。`:simplicial`では列群の設定を指定できません。自動・列群方式の構造解析そのものは逐次実行で、呼び出し元の作業領域には対応しません。

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
```

## ランクと許容誤差付き求解

`svdvals(F)`、`rank(F; atol, rtol)`、`pinv(F; atol, rtol)`は保存した Rの密な SVDを使います。しきい値は `max(atol, rtol*maximum(svdvals(F)))`です。`atol == 0`のとき既定の `rtol`は `min(size(F)...)*eps(real(eltype(F)))`です。

`pinv`は密な結果を返します。解だけなら `ldiv!(X, F, B; atol, rtol)`を使え、ランク不足でも最小ノルム最小二乗解を求められます。

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
```

## 横長の疎行列

横長の m×n 入力（m < n）では `F.Q`は m×m、`F.R`は m×nです。先頭の正方部分の QRと保存した反射を使い、残りの列を 1 列ずつ処理します。

求解は保存した Rの密な SVDで最小ノルム解を求めるため、元の入力が疎でも密な作業領域が必要です。反復して適用する場合は `FaerLinearOperator(F)`が Rの疑似逆作用を保存します。

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
```

## 通常の求解とランク不足

`F \ B` も全方式で数値ランクを確認します。良条件の縦長・正方の Rは疎な評価で判定し、確定できない場合は密な Rの SVDを使います。この切り替えは大きな疎因子では費用がかかる場合があります。許容誤差を変えるには `ldiv!` または `pinv`を使います。

```julia
using Colmat, SparseArrays, LinearAlgebra
A = [1.0 2; 2 4; 0 0]
F = qr(FaerSparseMatrix(sparse(A)))
b = ones(3)
@assert isapprox(Vector(F \ b), pinv(A)*b; atol=1e-12)
```

## 最小ノルム求解で資源を再利用する

数値更新でランクが変わると、通常の疎求解から Rの SVD 経路へ切り替わり、必要な作業領域が増える場合があります。`qr!`の後は必要量を照会し直し、`reserve!`で拡張します。元の入力を閉じても保存した数値因子のランク判定は変わりません。

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
```
