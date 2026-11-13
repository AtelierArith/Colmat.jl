# 反射・回転・帯行列への縮約

このページは分解の構成要素を直接扱いたい場合の詳細です。通常の連立方程式や固有値計算には `lu`、`qr`、`svd`、`eigen`を使えば、ここでの操作を手で組み合わせる必要はありません。

反射・回転は長さや直交性を保ちながら成分をゼロにする変換です。「縮約」は、それらを使って元の問題を同じ情報を持つ計算しやすい帯状の行列へ変形することです。三重対角化は固有値、二重対角化は SVDの準備に使います。

## Givens 回転

`LinearAlgebra.Givens`と消去後の値を返します。

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
使えます。`*`は型を昇格した独立した Rust 所有配列を返し、`lmul!`・`rmul!`は
係数を出力の型に変換してから更新します。変換に失敗すれば出力は変わりません。
回転の cosineは実数が必要です。生成対象の2要素は有限値に限定しています。
スカラー形式の `givens(f,g,i,j)`は Julia 標準の実装です。
独立した Jacobi オブジェクトは、このページの Jacobi rotationsを参照してください。

Householder 反射は `householder(x)` または `FaerHouseholder(x)`で生成できます。

## Householder 反射と列

faerと同じ `H = I - v*v'/tau`の規約で、`tau`は実数の分母です。

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
`H.v`は独立した ネイティブ ベクトル、`H.beta`・`H.tau`・`H.input_norm`・
`H.head_with_beta_inv`は faerの生成情報です。`tau`は LAPACKの乗数とは異なります。
`H'`・`transpose(H)`・`conj(H)`・`inv(H)`・コピーは独立した所有者を返します。
`H * A`・`A * H`・`H \ b`、左右の 格納領域内での更新 適用、3引数・5引数の `mul!`に対応します。
通常の Julia ストライドd 配列も使え、格納領域内での更新の出力は4数値型に限定します。
係数・生成情報を出力型に変換できない場合は更新前に失敗します。単一反射・
`FaerHouseholderSequence`・省メモリ形式の Hessenberg／三重対角 Qの `lmul!`・`rmul!`・
`mul!(C, H, B[, α, β])`は、要素型が同じなら Rust 所有型・ビュー・`faerview`・
Julia ストライドd 配列の Cに faerが直接書き込みます（一時結果・書き戻しなし。Bが Cと
重なる場合だけ スナップショット、`β == 0`では Cを、`α == 0`では Bを読みません。`β ≠ 0`
は Cと同形の ネイティブ 積バッファを1つ使います）。省メモリ形式の二重対角 U/Vの `mul!` も
宣言済み scratchの作業行列だけで Cに直接書き込みます。`Matrix(H)`と反射同士の積は密行列を作ります。
単一反射・ブロック／sequenceの左右適用は `scratch=`と `team=`に対応します。
必要量照会・reserveは以下の APIを使います。ブロック／sequenceは次の所有型で扱います。

複数の反射は `FaerHouseholderSequence`に 省メモリ形式の 基底／係数として保持できます。

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
```

`lmul!/rmul!/mul!/ldiv!/rdiv!`で作業領域を再利用できます。左右積の照会は
演算と同じ引数順序を使い、`mul!`の照会は alpha/betaと型昇格も反映します。
容量・アラインメント 不足や閉じた 資源は出力更新前に拒否します。Rust 側のカーネルは
呼び出し元が保持するチームを借用します。結果の 書き戻す前の一時保存・係数の型変換・Julia 配列変換は
scratchの外で割り当てます。sequence 構築・reblockの teamは指定できます。

`Q = H₁*H₂*…*Hₖ`の順序で、m×kの 基底には k≤mが必要です。
対角は暗黙の 1、上三角は無視し、狭義下三角の値だけを使います。
有限の `tau[j]`は `vⱼ'vⱼ/2`と一致する必要があります。`Inf`は恒等反射を表し、
その列の未使用部分はゼロに正規化します。`blocksize=k` なら単一ブロックです。
`Q.basis`・`Q.factor`は独立した Rust 所有行列、`Q.tau`は Juliaの実数ベクトルです。
因子は ブロックサイズ×kで、各ブロックの上三角を連結した faerの形式です。

`Q'`・`transpose(Q)`・`conj(Q)`は独立した所有者を返します。これらの `.basis`・`.factor`
は変換前の格納表現を返し、`.transposed`・`.conjugated`が適用方向を表します。
積・左右の 格納領域内での更新 適用・`mul!`・左右の求解は密な Qを作りません。
`Matrix(Q)`と演算子同士の積は明示的に密行列を作ります。
sequenceの構築と `reblock/reblock!`は `team=`を受け取ります。
因子 行列からの構築は `blocksize=`で変更後のサイズを指定でき、`prev_block_size=`
（既定は元・先 ブロックサイズの gcd）が両方を割り切る場合に保存済みの上三角 ブロックを再利用します。
reblockは 基底/tauを Juliaに戻さず Rust 側の所有者 から構築し、変換方向と失敗時の旧 所有者を保持します。

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

ネイティブ upgradeは scratch 引数・必要量照会を持たず、深い並列分岐の内部 バッファを毎回確保します。
[faer #93の実行済み MWE](https://github.com/tensor4all/faer/issues/93#issuecomment-6648218975)
と修正案で追跡し、構築/reblockの 呼び出し元が保持する作業領域は未完了です。

## Jacobi 回転

`rotg`は 2×2の回転と変換後の値を返します。回転は長さを保ちながら、指定した成分をゼロにするために使います。単独の回転を配列に適用する場合は対象の軸が長さ 2である必要があります。大きな配列の 2 行・2 列に埋め込むには `LinearAlgebra.Givens(J, i, j)`を使います。`make_givens`は実数の生成関数です。

```julia
using Colmat, LinearAlgebra
J, r = rotg(3.0, 4.0)
@assert J * [3.0, 4.0] ≈ [r, 0.0]
@assert Matrix(J') * Matrix(J) ≈ Matrix{Float64}(I, 2, 2)
A = FaerMatrix([1.0 2; 3 4])
lmul!(J, A)
@assert Matrix(A) ≈ Matrix(J) * [1.0 2; 3 4]
G = LinearAlgebra.Givens(J, 2, 4) # 大きな配列の指定した位置に回転を適用
```

## 三重対角化と固有値計算

自己随伴行列を、対角とその両隣だけが非ゼロの三重対角行列 Hに変形します。`A ≈ Q*H*Q'`が成り立ち、固有値計算の準備に使います。通常は `eigen(A)`に任せて構いません。

`tridiagonalize`の既定は明示的な Qと実数の `SymTridiagonal`を保持します。更新形式は成功後に入力を Hに置き換えます。`tridiagonal_eigen(d, e)`の eは下側の副対角で、上側はその複素共役です。値は実数の昇順で、値だけの計算は密行列を作りません。

```julia
using Colmat, LinearAlgebra
A = FaerMatrix([4.0 1 2; 1 3 1; 2 1 5])
F = hessenberg(Hermitian(A))
@assert F.H isa SymTridiagonal
@assert Matrix(F) ≈ Matrix(A)
E = eigen(F) # 縮約を再利用して三重対角の固有値を計算
@assert Matrix(A) * Matrix(E.vectors) ≈ Matrix(E.vectors) * Diagonal(E.values)
S = SymTridiagonal(FaerVector([1.0, 2, 3]), FaerVector([0.2, 0.4]))
@assert eigvals(S) ≈ tridiagonal_eigvals([1.0, 2, 3], [0.2, 0.4])
```

## 二重対角化と SVD

二重対角化は、対角と片側の隣だけを残す変形です。SVDの準備に使います。`bidiagonalize(A)`の既定は必要な列だけの基底を持ち、`full=true`は完全な基底を保存します。縦長・正方では上二重対角、横長では下二重対角になります。

更新形式は成功後に入力を Bの長方形への埋め込みで置き換えます。`bidiagonal_svd(d, e; uplo=:U)`は帯成分から直接 SVDを計算します。特異値は非負の降順です。現在、値だけを求める場合も内部では因子用の領域を確保します。

```julia
using Colmat, LinearAlgebra
A = [1.0 2; 3 4; 5 6]
F = bidiagonalize(A)
@assert Matrix(F) ≈ A
@assert Matrix(F.U) * Matrix(F.B) * Matrix(F.V)' ≈ A
S = svd(F) # 縮約を再利用して二重対角 SVD を計算
@assert Matrix(S.U) * Diagonal(S.S) * Matrix(S.Vt) ≈ A
B = Bidiagonal(FaerVector([3.0, -2]), FaerVector([0.5]), :U)
@assert svdvals(B) ≈ bidiagonal_svd([3.0, -2], [0.5]).S
```

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

## 二重対角 SVDの作業領域

`bidiagonal_svd`、保存した縮約の `svd`・`svdvals`、ネイティブな `Bidiagonal`は `scratch`と `team`に対応します。係数の一時コピーとカーネルの一時領域を照会します。返り値と Juliaへの変換は別に確保します。

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

## 二重対角化の作業領域

`bidiagonalize`と更新形式は作業領域を再利用できます。`full`・`compact`・`team`を照会と実行で合わせてください。縮約と基底の適用は同じ一時領域を使い回します。入力変換、返り値、書き戻しや Rust 内部の追加バッファは別に確保する場合があります。

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

## 二重対角化の並列化設定

`FaerParThreshold()`は既定の計算量モデル、整数は更新する残りの領域の要素数の下限、`:never`は縮約と明示的な U/Vの構築を逐次実行にします。照会と実行で同じ設定を使います。どの設定でも、渡したチームは開いている必要があります。

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

## 三重対角固有値計算の作業領域

固有値だけ、または固有ベクトルを含む計算で `scratch`と `team`を使えます。保存済みの三重対角化やネイティブな `SymTridiagonal` からも渡せます。保存した Qによる元の座標への変換もチームを使います。出力と並べ替え用配列は別に確保します。

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

## 極端な大きさの複素入力

非常に大きい・小さい有限値は、2のべき乗でスケールを調整してから縮約し、成分ごとに戻します。残差を調べる際にも元のスケールを外すと、比較自体のオーバーフローを避けられます。

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

## 小さな複素成分の位相

全体の残差では見えにくい小さな副対角成分も、位相（複素数の向き）の調整を必要とします。重要な成分は個別にも確認してください。作業スケールで非正規化数となるほど相対的に小さい成分を、すべて保存する保証はありません。

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

## 三重対角化の作業領域

`tridiagonalize`と更新形式、自己随伴入力の `hessenberg`は `scratch`と `team`を受け取ります。作業行列、反射の係数、縮約と Q 構築の一時領域を再利用します。返り値と入力変換、書き戻しのメモリは別です。

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

## 三重対角化の並列化設定

固定しきい値は、残りの次数 `rem`に対する `rem*rem÷2`を基準にします。`:never`は縮約と明示的な Qの構築を逐次実行にします。照会と実行で同じ設定を使い、空の問題でも渡すチームは開いておいてください。

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

## 三重対角化の Qを省メモリで保存する

`compact=true`は密な Qの代わりに `FaerTridiagonalQ`として反射と位相を保存します。積や求解は Q 全体を作らず適用し、`Matrix(F.Q)`で明示的な行列にできます。

転置・随伴・共役は変更しない保存領域を共有し、それぞれ個別に閉じられます。`copy`と `deepcopy`は独立した所有者を作ります。`basis`・`factor`・`phases`は独立した Julia 配列です。入力を変更・終了しても保存した Qは使えます。

実数の帯成分が表現可能である必要があり、絶対値がオーバーフローする場合は先に入力を縮小します。照会にも `compact=true`を渡します。保存領域や出力は一時作業領域とは別に確保します。

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

## 二重対角化の U/Vを省メモリで保存する

`bidiagonalize(A; compact=true)`は `FaerBidiagonalBasis`に反射を保存します。既定は必要な列だけの基底、`full=true`は正方の基底です。Uと Vは変更しない保存領域を共有し、それぞれ個別に閉じられます。

`Matrix(Q)`は保存した形状の行列を作ります。長方形の基底への適用は適切な形状の別の出力先を使い、`lmul!`・`rmul!`は正方基底に限ります。`Q \ A`と `A / Q`は随伴による最小二乗・最小ノルム作用です。完全な SVDには構築時にも `full=true`が必要です。作業領域は構築と適用で別に照会します。

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

## 一般の Hessenberg 化で Q を省メモリ保存する {#compact-hessenberg-q}

`hessenberg(A; compact=true)`は `FaerCompactHessenberg`を返し、Hと反射を保持する `FaerHessenbergQ`を保存します。`Matrix(F.Q)`を呼ぶまで密な Qを作りません。積・求解・転置・随伴で保存した反射を使います。

`hessenberg!`では Hが入力を共有し、Qは独立した反射を保持します。`schur(F)`は元の Qを明示的に作らず利用します。`schur!(F)`は成功後に Hを更新し、Qを明示的な Schur 基底に置き換えます。以前に取り出した暗黙の Qは引き続き使えます。構築用の照会にも `compact=true`を渡してください。

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

## 一般の Hessenberg 化の設定

`par_threshold`は並列化、`blocking_threshold`はブロック化の判断を制御します。既定の `blocking_threshold=65536`では `n*n`がそれ未満なら非ブロック経路です。`0`はブロック経路を許し、十分大きい値は非ブロック経路を選びます。保存後の Schur 反復の設定とは別です。

明示的・省メモリ形式とも照会と実行を同じ設定にします。更新形式は計算成功後に Hを書き、独立した Qを保持します。

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
