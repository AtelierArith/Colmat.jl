# 固有値・Schur・QZ

固有値 λと固有ベクトル vは `A*v = λ*v`を満たします。まずは `eigen(A)`、値だけが必要なら `eigvals(A)`を使います。各列と固有値が対応しているかは残差で確認できます。

```julia
using Colmat, LinearAlgebra
A = FaerMatrix([4.0 1; 1 3])
F = eigen(Hermitian(A))
@assert Matrix(A) * F.vectors ≈ F.vectors * Diagonal(F.values)
```

`Hermitian(A)`は自己随伴であることを明示します。一般の行列では固有値が複素数になることもあります。Schur 分解は行列を三角または小さなブロックの三角形に変形する方法、Hessenberg 化はその準備、QZは行列対のための方法です。これらの詳細は必要になったときに参照してください。

## Hessenbergと Schur 分解

Hessenberg 化と Schur 分解も ネイティブ 経路を使います。

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

`schur`は Rust 所有の `T`・`Z`と Juliaの固有値ベクトルを持つ
`LinearAlgebra.Schur`を返します。実数では共役対を 2×2 ブロックに保ち、
`ordschur`の選択では対の両方を含めるか、両方を除外します。
`hessenberg`の既定の結果は `FaerHessenberg`で、Juliaの 省メモリ形式の Householder 表現と
異なり `H`・`Q`の両方を明示的な行列として所有します。
`compact=true`は ネイティブ reflectorを保持する `FaerCompactHessenberg`を返し、
保存因子からの Schurでも Qを暗黙に適用します。所有権と 更新形式の Q 置換は
[省メモリ形式の Hessenberg](reductions.md#compact-hessenberg-q)を参照してください。
`hessenberg!`・`schur!`は独立した作業結果を計算してから入力の所有バッファを
更新し、形状と既存ビューを保ちます。`schur!(H)`は `H.H`・`H.Q`の両方を
更新し、返す Schur 分解とストレージを共有します。並べ替えも成功後に更新する
ため、失敗時は元の分解を保ちます。これらは内部の一時領域を割り当てます。

### Schur・QZの計算方法の調整

`with_schur_tuning`は Schur・QZと一般化 Hessenberg 化の性能設定を一時的に変えます。初めて使う場合は既定値で構いません。シフト数、収束した部分を取り除く窓の大きさ、ブロック化などを調整できます。

必要量の照会と実行を同じスコープ内で行います。設定は例外時も元に戻ります。スコープは Colmatの呼び出しロックを保持するため、同じタスクから Colmatを呼んでください。一般の固有値分解内部の QR・QZ 計算にも適用します。

```julia
using Colmat, LinearAlgebra
A = FaerMatrix([sin(i + j^2) for i in 1:80, j in 1:80])
F = with_schur_tuning(; shift_count=8, deflation_window=10, blocking_threshold=20) do
    schur(A)
end
@assert Matrix(F.Z) * Matrix(F.T) * Matrix(F.Z)' ≈ Matrix(A)
```

### 固有値分解の計算方法の調整

EVDは固有値分解の略です。`with_evd_tuning`で Hessenberg 化・三重対角化や再帰のしきい値を一時的に変えます。`nothing`は既定値です。`with_schur_tuning`と入れ子にでき、作業領域の照会も同じ設定に従います。通常の利用では指定不要です。

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

## 一般化 Schur／QZ

一般化 Schur／QZは行列対に適用します。

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

`GeneralizedSchur`の `S`・`T`・`Q`・`Z`はすべて Rust 所有行列です。
`hessenberg(A,B)`は `FaerGeneralizedHessenberg`を返し、
`A ≈ Q*H*Z'`・`B ≈ Q*T*Z'`を満たします。`hessenberg!`・`schur!`は
計算成功後に既存の所有バッファを更新し、既存ビューを維持します。
更新する行列対は同じスカラー型で、別々の所有者にする必要があります。
`schur!(H)`は H/T/Q/Zのすべてを更新し、結果とストレージを共有します。

固有値は同次形式の `α`・`β`でも保持します。`β == 0, α != 0`は無限固有値、
`α == β == 0`は不定な固有値を表します。比に直す前の値を調べることで区別できます。
実数・複素数とも ネイティブ QZを使い、混合型の通常呼び出しでは型を昇格します。
一般化 `ordschur`／`ordschur!`は保存済みの S/Tを ネイティブで並べ替え、
選択したブロックを先頭へ集めます。選択群・非選択群の内部の順序は保ちます。
実共役対の片方だけを選ぶと両方が対象になります。α/βは保存因子から更新します。

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
```

`ordschur!`は全計算・固有値の変換に成功してから既存の S/T/Q/Z・α/βを
更新し、元の所有者とビューを保ちます。行列は別々の Rust 所有者、α/βは
要素領域の重ならない mutable ストライドd vectorが必要です。同じ Julia 配列の
互いに重ならないビューも使えます。数値的に安全な交換や固有値評価ができない
場合は例外を返し、元の分解を保ちます。`scratch=`では `FaerScratch`を再利用でき、
`reserve!(work, ordschur, F)`で容量を増やせます。容量・アラインメントの不足や閉じた
scratchは入力を変更せずに拒否します。作業因子と Juliaの出力・変換配列は別に確保します。
`values`／`eigvals(F)`は Juliaの α/βの比に従い、複素ゼロ除算では NaNを
含む場合があるため、無限・不定値の判定には同次対を使ってください。
QZ 構築の `scratch`・`team`は、このページの構築用リソースの節を参照してください。

## 計算前に行列のスケールを整える

要素の大きさが極端に異なる場合、`balance`で行・列の並べ替えと 2のべき乗による拡大縮小を行えます。固有値は保たれますが、固有ベクトルは元の座標へ戻す必要があります。`balance_eigenvectors!`はその変換を行います。

`eigen(A; permute=true, scale=true)`は調整、固有値分解、元の座標への変換、列の長さの正規化をまとめて行います。両設定は同じ値にしてください。Colmatの既定は両方 `false`で、Juliaの LAPACK 経路の既定とは異なります。行列対にはこの設定は使えません。

```julia
using Colmat, LinearAlgebra
A = [1.0 1024; 1/1024 2]
F = balance(A)
@assert Matrix(F.B) ≈ A[F.perm, F.perm] .* reshape(F.scale, 1, :) ./ F.scale
E = eigen(F.B)
X = balance_eigenvectors(F, E.vectors)
@assert A * Matrix(X) ≈ Matrix(X) * Diagonal(E.values)
```

```julia
using Colmat, LinearAlgebra
a = [1.0 1e4 0; 1e-4 2 1e3; 0 1e-3 3]
A = FaerMatrix(a)
E = eigen(A; permute=true, scale=true)
@assert norm(a * E.vectors - E.vectors * Diagonal(E.values)) < 1e-10
L = eigen(A; side=:left, permute=true, scale=true)
@assert norm(L.vectors' * a - Diagonal(L.values) * L.vectors') < 1e-10
```

## 一部の固有値・特異値だけを求める

大きな問題で少数の値だけが必要なら、`partial_eigen`、`partial_self_adjoint_eigen`、`partial_svd`を使えます。自己随伴版は Hermitian な入力に限ります。初期ベクトル `v0`、探索空間の大きさ `min_dim`・`max_dim`、許容誤差 `tol`、再開始回数 `max_restarts`を指定できます。

必ず `nconverged`と残差を確認してください。返るのは収束が確認された列だけで、初期ベクトルによって目的の部分空間を見逃す場合があります。

自己随伴版と SVDは、探索空間が問題全体に達する小さな問題では密行列を作る経路に切り替わります。`algorithm == :dense_fallback`で判別でき、常に少量のメモリで済むわけではありません。反復経路は `:partial`です。`scratch`と `team`に対応し、照会時には探索設定もそろえます。

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
```

## 入力を書き換える固有値計算

`eigen!`と `eigvals!`は成功後に入力を書き換えます。一般の行列には Schur 形を書き、自己随伴行列の `eigen!`は固有ベクトル、`eigvals!`は固有値の対角行列を書きます。返り値の配列は入力から独立しています。

行列対は別々の所有者かつ同じ数値型が必要です。一般の行列対は S/T、自己随伴の対は第 1 入力に固有ベクトル（値だけの形式では対角行列）、第 2 入力に固有値の対角行列を書きます。収束や並べ替えの失敗時は書き戻しません。

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

## 自己随伴行列の固有値と作業領域

自己随伴とは `A == A'`となる性質で、実数では対称行列です。`Hermitian(A, :L)`は下三角、`:U`は上三角を正しい値として扱います。参照しない側と複素対角の虚部は無視します。

`eigen`、`eigvals`と更新形式は `scratch`と `team`に対応します。固有値は実数です。選択した三角領域に非有限値があれば拒否します。返り値や並べ替え用の配列は作業領域とは別に確保します。

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

## 計算方法・範囲選択・複素対称行列

`alg=DivideAndConquer()`は既定の分割統治法、`QRIteration()`は QR 反復を選びます。`RobustRepresentations()` も既定経路を選び、MRRR 法を提供するわけではありません。値だけの計算は QRを使います。`native=true`の保存形式では `QRIteration()`を指定できません。

`eigen(A, irange)`は昇順の添字範囲、`eigen(A, vl, vu)`は `(vl, vu]`の値を選びます。現在は全固有値を計算してから抽出するため、少数の選択でも全分解の費用がかかります。

複素数では `Symmetric`（転置で不変）と `Hermitian`（随伴で不変）は別です。複素 `Symmetric`の参照三角がすべて実数なら自己随伴版、それ以外は一般の固有値計算を使います。後者には `scratch`を指定できず、複素 `Symmetric`の `eigen!` も定義していません。

```julia
using Colmat, LinearAlgebra
H = Hermitian(FaerMatrix([2.0 -1 0; -1 2 -1; 0 -1 2]))
λ = eigvals(H)
@assert eigen(H; alg=LinearAlgebra.QRIteration()).values ≈ λ
@assert eigvals(H, 2:3) ≈ λ[2:3]
@assert eigen(H, 1.0, 4.0).values ≈ λ[1.0 .< λ .<= 4.0]
@assert eigmin(H) ≈ λ[1] && eigmax(H) ≈ λ[end]
S = Symmetric(FaerMatrix(ComplexF64[1 2im; 2im 3]))
@assert eigen(S).values ≈ eigen(Symmetric(ComplexF64[1 2im; 2im 3])).values
@assert eigvals(Symmetric(FaerMatrix(ComplexF64[2 1; 1 2]))) isa Vector{Float64}
```

## 一般の固有値計算と作業領域

一般の `eigen`・`eigvals`は `scratch`と `team`を受け取ります。実数入力でも固有値・固有ベクトルは複素数です。固有ベクトルの列の 2 ノルムは 1に正規化します。

更新形式の設定は固有値計算に適用しますが、入力へ Schur 形を書き戻すために別の逐次 Schur 分解も実行します。その追加作業は照会した `scratch`の範囲外です。行列対にも専用の作業領域インターフェースがあります。

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

## 正定値の行列対の固有値問題

一般化固有値問題は `A*v = λ*B*v`を解く問題です。Aと Bが自己随伴で Bが正定値なら、Cholesky 分解を使った専用経路があります。固有ベクトルは `V' * B * V ≈ I`となるよう正規化します。

`eigen`、`eigvals`と更新形式は `scratch`と `team`に対応します。選択した三角領域だけを読みます。Bが正定値でなければ `PosDefException(i)`、非有限値があれば `ArgumentError`です。正の対角行列 B も使えます。

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

## 一般の行列対の固有値問題

一般の行列対にも `eigen(A, B)`と `eigvals(A, B)`を使えます。4 種類の対応数値型、型の昇格、`scratch`、`team`に対応します。実数入力でも結果は複素数です。

更新形式は固有値計算後、別の逐次 QZ 分解で S/Tを書き戻します。この追加の作業領域は固有値計算用の照会に含まれません。重複した固有値では固有ベクトルが独立とは限りません。無限固有値や不定な対を区別するには、以下の同次形式を使ってください。

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

### Julia 標準との主な違い

| 項目 | Colmatの規約 |
| :--- | :--- |
| 一般の行列対の型 | 固有値・ベクトルは常に複素数 |
| 固有ベクトルの長さ | 左右とも列ごとの 2 ノルムを 1にする。符号・位相は一意でない |
| 自己随伴の専用経路 | `Hermitian` / `Symmetric`で明示する |
| 並べ替え | 通常は Juliaと同じ `sortby`。同次形式の既定は並べ替えなし |
| 無限・不定値 | 比の NaN / Inf だけで判定せず `α`と `β`を使う |
| スケール調整 | 行列対は内部で 2のべき乗による値域調整を行い、α・βを元へ戻す |
| 更新形式 | A/Bに別計算の一般化 Schur 形 S/Tを書く |

固有ベクトルは定数倍でも同じ解です。他のライブラリとの比較では要素の一致ではなく、固有値と方程式の残差を確認してください。


## 無限固有値も扱う同次形式

`homogeneous=true`は λを割り算で求めず、`β*A*v = α*B*v`の組 `(α, β)`を返します。`β == 0` かつ `α != 0`は無限固有値、両方ゼロは不定な対です。

`eigen`は `(α, β, vectors)`、`eigvals`は `(α, β)`の独立した Julia 配列を返します。既定では元の順序を保ち、`sortby`を明示すると比に基づいて共通の並べ替えを行います。対象は一般の行列対で、自己随伴ラッパーや更新形式には指定しません。

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

## 行列対の左固有ベクトル

通常の固有ベクトルは右から掛ける列ベクトルです。`side=:left`は左固有ベクトルを列として返します。返り値は `(values, vectors)`、同次形式では `(α, β, vectors)`です。左固有ベクトルは右とは一般に異なるため、下の例の随伴の方程式で残差を確認します。

`side=:right`が既定です。どちらの片側計算も `scratch`と `team`を使えます。破壊的更新形式にはこの切り替えを指定しません。

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

## 行列対の左右を同時に求める

`side=:both`は `(values, left, right)`、同次形式では `(α, β, left, right)`を返します。左右の列は同じ固有値順に並び、個別に長さを正規化します。左右の内積を 1にする正規化は行いません。

作業領域の照会・構築にも `side=:both`を渡してください。片側の場合より多くの領域が必要になることがあります。

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

## 単一行列の左・左右固有ベクトル

一般の単一行列も `side=:left`と `:both`に対応します。左だけなら `(values, vectors)`、左右なら `(values, left, right)`です。通常の右側は `Eigen`のままです。

左側を含む名前付きタプルは固有ベクトルを運ぶための返り値で、右側の分解と同じ求解・再構成機能を持つわけではありません。必要量の照会にも向きを指定します。自己随伴ラッパーと更新形式には指定しません。

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

## 保存した SVD・固有値分解で解く作業領域

`scratch_requirements`、`FaerScratch`、`reserve!`は求解と同じ引数順を使います。許容誤差、型の昇格、右辺の列数、チームを実行と合わせてください。容量不足は更新前に拒否します。因子そのものの構築費用は別です。

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
```

## 一般化 Hessenberg・QZの作業領域

行列対の `hessenberg`、`schur`、保存した Hessenberg からの `schur`と各更新形式は `scratch`と `team`に対応します。入力や出力の所有メモリは別に確保します。更新形式は計算成功後に書き戻し、既存のビューを保ちます。必要量は同じ演算・チームで照会してください。

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
```

## 単一行列の Hessenberg・Schurの作業領域

単一行列の縮約と、保存した Hessenberg からの Schur 計算でも一時メモリとチームを再利用できます。返される因子、固有値配列、更新時の書き戻しや内部の追加領域は別に確保される場合があります。

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
```

## Schur 形の並べ替えと作業領域

`ordschur`と `ordschur!`は選択した固有値のブロックを先頭に移します。作業メモリを再利用できますが、作業因子や Juliaの配列は別に確保します。現在の並べ替えカーネルは、チームを渡しても逐次実行です。

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
```

## 保存した Schur 分解から固有ベクトルを求める

`eigvecs(F)`と `eigen(F)`は保存した Schur 因子を再利用します。`scratch`と `team`に対応します。返り値の複素行列や固有値は独立した配列です。極端な大きさの要素を持つ場合など、Rust 内部で追加コピーが必要になる場合があります。

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
```
