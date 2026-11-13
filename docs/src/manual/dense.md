# 密行列の分解と求解

密行列は全要素を格納する行列です。まずは `A \ b`で連立方程式を解き、右辺だけを変えて繰り返す場合に分解を保存します。「分解」「最小二乗」「ランク」の意味は[計算の選び方と用語](../getting-started.md)にあります。

このページの前半では通常の分解、後半では入力を書き換える形式や性能調整を説明します。まず `lu(A)`、`qr(A)`、`svd(A)`の必要なものから試してください。

## 保存した密分解

密行列の分解は Rustに保持して再利用できます。

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

密 LU・完全ピボット LU・QR（列ピボットを含む）・Cholesky・LDLT・
Bunch–Kaufmanの構築と対応する `!` 版には `team=`を指定できます。
`factorize(A; team)` も対応します。通常の保存因子は team や入力を閉じても独立に使えます。入力を共有する直接分解は例外です。
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

### Julia 配列を借用した分解・直接求解

`faerview(J)`で借用した Julia 行列（4 型、負・非単位 ストライドを含む）は、
`lu`・`full_piv_lu`・`qr`（`NoPivot`/`ColumnNorm`）・`cholesky`・`ldlt`・
`bunchkaufman`（`Hermitian`/実 `Symmetric` ラッパーを含む）・`svd`/`svdvals`・
`eigen`/`eigvals`の入力として faerが親配列の `MatRef`を直接読みます。
結果は `FaerMatrix` 入力と同じ型（`FaerFactorization`・`SVD`・`Eigen`）で、
Julia や Rust 側の入力 スナップショットは作りません（因子自身の格納は出力として確保）。
以前は LinearAlgebraの汎用経路（LAPACKと `Matrix`へのコピー）でした。
LAPACKを使うには親配列 `parent(B)`を渡します。`det`・`inv`・`B\b`・
`factorize`・`cond`・`rank`・`pinv` など LinearAlgebraの汎用関数もこの Rust 側の分解を
通ります。`scratch`・`team`（SVD/EVD）・`alg`・`native=true`・`side`・平衡化・
チューニング中の呼び出しは入力を ネイティブへ1回コピーして `FaerMatrix`の経路を使います。
読み取り専用の NPY 借用も分解の入力にできます。ラッパーなしの Hermitian な借用行列の
`eigen`/`eigvals`は、Julia 配列と同様に自己随伴ドライバで実固有値を返します
（`eigmax`/`eigmin` も使えます。`FaerMatrix`の `eigen`は一般ドライバで複素数です）。

同じ要素型の正方・ランク落ちしていない保存因子による `ldiv!(F, B)`は、
`B`が `Matrix`・ストライドd `SubArray`・`faerview`・`FaerVector`・ネイティブビューなら
faerの `solve_in_place`を `B`の記憶域で直接実行し、右辺のコピー・新しい結果・
書き戻しを作りません。`ldiv!(X, F, B)`は `B`を `X`へ Rust 側で1回代入して
（重なる入力は スナップショット）から `X`で同じ 格納領域内での更新 求解をします。`F \ B`
（`B`が Julia ストライドd 配列・`faerview`）は右辺の ネイティブ コピー1つで解いて返します。
2引数 `rdiv!(B, F)` も同じ条件で `transpose(B)`に在位置求解します。`scratch=`を
渡すと因子種別ごとの低水準 カーネルが 呼び出し元の作業領域 上で動き、在位置求解は ネイティブ
割当なしになります。長方形・ランク落ちした QR・SVD 因子、型昇格は従来の新規結果の
経路で、Juliaの書き込み先へは結果を Rust 側で1回代入します。

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

## 特異値分解（SVD）と計算方法

SVDは `A ≈ U * Diagonal(S) * Vt`と表す分解です。`S`の値（特異値）は、行列が各方向をどれだけ拡大するかを表します。小さな特異値は、入力の違いを結果から区別しにくい方向に対応します。`svdvals(A)`は特異値だけを返します。

`alg=LinearAlgebra.QRIteration()`で計算方法を選べます。性能調整の `with_svd_tuning`は通常不要です。設定を変える場合は作業領域の照会と実行を同じスコープ内で行ってください。

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
```

## 列を並べ替える QRと数値ランク

ランクは独立な列の数です。丸め誤差があるため、数値計算では小さな特異値をゼロとみなす許容誤差を指定します。`qr(A, ColumnNorm())`は列を並べ替えて分解し、`rank(F; atol, rtol)` や許容誤差付きの `ldiv!`で保存因子を再利用できます。

しきい値は絶対許容誤差 `atol`と相対許容誤差 `rtol` から決まります。しきい値を大きくすると、解に使う方向を減らします。ランク不足の最小ノルム解では、保存した Rの SVDを使います。

```julia
using Colmat, LinearAlgebra
a = [1.0 2 3; 2 4 6; 1 0 1; 0 1 1]
a[:, 2] = 2a[:, 1]                       # ランクは 2
F = qr(FaerMatrix(a), ColumnNorm())
@assert rank(F) == 2
@assert rank(F; atol=1e3) == 0
B = reshape([1.0, 2, 3, 4], 4, 1)
B, r = ldiv!(F, B, 1e-10)               # 小さい特異値を切り捨てた最小ノルム解
@assert r == 2
@assert B[1:3, :] ≈ pinv(a) * [1.0, 2, 3, 4]
X = zeros(3, 1)
ldiv!(X, F, [1.0 2 3 4]'; rtol=1e-10)   # キーワード形式も同じ計算を使う
@assert X == B[1:3, :]
x = ldiv!(zeros(3), F, [1.0, 2, 3, 4]; atol=1e3)
@assert iszero(x)                       # ランクが 0 ならゼロを書き込む
```

## 保存した QR から疑似逆行列を作る

疑似逆行列は、逆行列が存在しない場合や長方形の行列でも最小二乗解を表すための行列です。`pinv(F; atol, rtol)`は保存した QRの Rと反射を使い、元の入力を再分解しません。結果は独立した密行列なので、大きな問題では必要なメモリに注意してください。解だけが必要なら `ldiv!`を使えます。

```julia
using Colmat, LinearAlgebra
a = [1.0 2 3; 2 4 6; 1 0 1; 0 1 1]       # ランクは 2
F = qr(FaerMatrix(a), ColumnNorm())
P = pinv(F; rtol=1e-10)
@assert Matrix(P) ≈ pinv(a; rtol=1e-10)
FaerTeam(2) do team
    work = FaerScratch(pinv, F; team)    # 保存した R の SVD 用
    Q = pinv(F; scratch=work, team)
    @assert Matrix(Q) ≈ pinv(a)
    close(Q)
    close(work)
end
```

## 行列式とトレース

`det(A)`は行列式、`tr(A)`は対角成分の和（トレース）です。`logabsdet`は行列式の絶対値の対数と符号を返し、積が極端に大きい・小さい場合に便利です。特異かどうかの数値判定には行列式の大きさだけを使わず、ランクや特異値も確認してください。

```julia
using Colmat, LinearAlgebra
A = FaerMatrix([4.0 1 2; 1 3 0; 2 0 5])
@assert det(A) ≈ 43
@assert det(view(A, 3:-1:1, :)) ≈ -43        # ビューを借用し、コピーしない
team = FaerTeam(2)
@assert det(A'; team) ≈ 43
close(team)
@assert det(LowerTriangular(A)) == 60       # 格納された対角成分の積
@assert logabsdet(FaerDiagonal([2.0, -3.0])) == (log(6.0), -1.0)
@assert tr(view(A, 1:2, 1:2)) == 7
```

## 三角行列・対角行列の積

`LowerTriangular(A)`は下三角だけを使う行列、`UpperTriangular(A)`は上三角だけを使う行列です。`UnitLowerTriangular` などは対角を 1とみなします。無視される側に値があっても計算には使いません。

`mul!`は出力先へ積を書き込みます。`structured_mul!`の `dest`で出力する三角領域を指定できます。対角行列との積は行または列を拡大・縮小する操作であり、密な対角行列を作る必要はありません。

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
@assert Vector(L * v) == [2.0, 7.0]           # Rust 所有の FaerVector
E = FaerMatrix(fill(-1.0, 2, 2))
structured_mul!(E, L, FaerMatrix([1.0 0; 0 1]); dest=:strict_lower)
@assert Matrix(E) == [-1.0 -1; 3 -1]      # 対角を除く下三角だけを書き換える
```

```julia
using Colmat, LinearAlgebra
A = FaerMatrix([2.0 1; 3 4])
J = [1.0 2; 3 4]
@assert UpperTriangular(A) * J isa FaerMatrix
@assert Matrix(Symmetric(A) * J) ≈ Symmetric([2.0 1; 3 4]) * J
C = zeros(2, 2)
mul!(C, UpperTriangular(A), J)              # C の格納領域へ直接書く
@assert C ≈ UpperTriangular([2.0 1; 3 4]) * J
@assert Matrix(A + Diagonal([1.0, 1.0])) == [3.0 1; 3 5]
```

## 並べ替え付き Cholesky 分解

Cholesky 分解は正定値行列を三角因子に分けます。`RowMaximum()`を指定する形式は対角の大きさで並べ替え、正の半定値行列のランクを扱えます。`check=false`では状態を自分で確認してください。分解を得たことと、任意の右辺に対して通常の逆行列が存在することは別です。

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

### 並べ替え付き Choleskyの作業領域

`FaerScratch`と `team`を使えます。許容誤差やブロックの設定を含め、必要量照会と実行で同じ引数を渡してください。入力を更新する形式では、元の値が必要なら先にコピーします。

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

## 入力を書き換える分解

`lu!` や `qr!`の `!`は入力が更新されることを示します。結果の保存形式は元の行列とは異なります。通常の形式では独立した因子を計算してから入力に書き戻しますが、`scratch`を渡す直接分解では入力と返り値の因子が格納領域を共有します。次の節の使い分けを確認してください。

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

## 入力の格納領域を直接使う分解

`lu!(A; scratch)` などは `FaerInPlaceFactorization`を返し、入力 `A`に保存した因子を参照します。`A`を変更・サイズ変更・`close` すると因子にも影響するため、分解の利用中は入力を保ってください。

対象は LU、完全ピボット LU、QR、列ピボット QR、Cholesky、LDLᴴ、Bunch–Kaufmanです。入力の形状・数値型・設定を合わせて作業領域を用意します。直接書き換えるため、数値的な失敗後も元の入力を保つ保証はありません。入力を残したいときは `lu(A)` などを使います。メモリ確保の削減は処理や大きさに依存し、Rust 内部の確保までゼロとは限りません。

```julia
using Colmat, LinearAlgebra
A = FaerMatrix([4.0 1; 1 3])
work = FaerScratch(lu!, A)          # scratch_requirements(lu!, A; team)
F = lu!(A; scratch = work)          # A 自身の格納領域で LU 分解
@assert F isa FaerInPlaceFactorization && F.factors === A
B = FaerMatrix([1.0 2; 3 4])
solve = FaerScratch(ldiv!, F, B)
ldiv!(F, B; scratch = solve)        # B を A \ B で上書き
C = FaerMatrix([4.0 1; 1 3])
G = cholesky!(Hermitian(C, :U); scratch = FaerScratch(cholesky!, C))
```

```julia
H = FaerMatrix([1e-8 2 0; 2 -3 1; 0 1 4])
K = bunchkaufman!(Hermitian(H, :U), true; scratch = FaerScratch(bunchkaufman!, H, true))
K \ [1.0, 2.0, 3.0]
```

```julia
X = [1.0 2 3; 2 4 6; 1 2 1; 0 0 1]                  # ランクは 2
P = FaerMatrix(X)
F = qr!(P, ColumnNorm(); scratch = FaerScratch(qr!, P, ColumnNorm()))
@assert rank(F) == 2 && F.factors === P
x = zeros(3)
ldiv!(x, F, [1.0, 2, 3, 4]; rtol = 1e-10)           # 最小ノルム解
@assert x ≈ pinv(X) * [1.0, 2, 3, 4]
```

```julia
A = FaerMatrix([4.0 1; 2 3])
F = lu!(A; scratch = FaerScratch(lu!, A))
X = FaerMatrix([1.0 2; 3 4])
rdiv!(X, F; scratch = FaerScratch(rdiv!, X, F))   # X 自身の格納領域で X / A を計算
LinearAlgebra.inv!(F; scratch = FaerScratch(LinearAlgebra.inv!, F))
@assert Matrix(A) ≈ inv([4.0 1; 2 3]) && !isopen(F)
```

## 三角行列を使った求解

三角行列なら、一般の LU 分解をせず順に未知数を求められます。`ldiv!`は左から、`rdiv!`は右からの求解です。ネイティブ配列・ビュー・借用配列を親にした三角ラッパーに対応します。普通の Julia 行列を親にした場合は Julia 標準の処理を使います。コピーなしで Colmatの処理を選ぶには `LowerTriangular(faerview(A))` などを使います。

ゼロの対角成分は `SingularException`になります。型の昇格や入出力の重なりがある場合は一時コピーが必要です。`inv!`は選択した三角領域だけを逆行列で置き換え、反対側は変更しません。単位三角ラッパーの格納対角も変更しません。

`cond(A, 1)`と `cond(A, Inf)`は逆行列を明示的に作らない推定値です。`cond(A, 2)`は特異値を使います。

```julia
using Colmat, LinearAlgebra
A = FaerMatrix([2.0 0; 1 4])
B = FaerMatrix([2.0 4; 9 8])
ldiv!(LowerTriangular(A), B)            # B 自身の格納領域で求解
@assert Matrix(B) ≈ LowerTriangular([2.0 0; 1 4]) \ [2.0 4; 9 8]
X = [1.0 2; 3 4]
rdiv!(X, UpperTriangular(transpose(A))) # X を X / transpose(L) で更新
Z = FaerMatrix(ComplexF64[1 2; 3 4])
work = FaerScratch(ldiv!, LowerTriangular(A), Z) # 数値型を昇格した三角行列のコピー用
ldiv!(LowerTriangular(A), Z; scratch = work)
T = UpperTriangular(FaerMatrix([2.0 1; 0 4]))
LinearAlgebra.inv!(T; scratch = FaerScratch(LinearAlgebra.inv!, T))
```

## 不定値分解と完全ピボット LUの更新形式

`ldlt!`、`bunchkaufman!`、`full_piv_lu!`は通常、独立した因子を作ってから入力へ書き戻します。`scratch` 付きの直接分解とは所有権が異なります。

LDLᴴは `L * D * L'`の形です。Bunch–Kaufmanでは Dに 2×2 ブロックがあるため、入力に書かれた対角・下三角だけでは復元できません。返された `F.D`と置換 `F.p` も保持してください。

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

- `bunchkaufman[!](A, rook=false; check, team)`は Juliaと同じ位置引数 `rook`を受け、
  `true`で faerの rook ピボット（`PivotingStrategy::Rook`）を使います。`false`は従来どおり
  faer 既定の `PartialDiag`で、LAPACKの古典的 Bunch–Kaufman（faerの `Partial`）とは
  ピボット列が異なり得ます。`F.rook`は選んだ戦略を返し、`copy`と昇格求解の再分解でも保持されます。
- `inertia(F; atol, rtol)`は `F.D`の 1×1/2×2 ブロックから Juliaと同じ許容規則で
  `(正, 負, 零)` 固有値数を数えます。`F.D`を密に取り出すため O(n²)の一時領域を使います。
- `cholesky(A; check=false)`は非 Hermitian 入力に対して Juliaと同様 `info == -1`の
  失敗因子を返します（`check=true`は `PosDefException(-1)`）。`cholesky!`はこのとき入力を変更しません。
  失敗因子は数値因子を保持しないため、成分取り出しはエラーになります。`F.UL`は `uplo` 側の三角因子です。
- 複素 `Symmetric` ラッパーは、格納三角（対角を含む）が実数のとき Hermitianとして分解します。
  それ以外の複素対称（非 Hermitian）行列は、`cholesky`では Juliaと同じく `info == -1`、
  `ldlt`/`bunchkaufman`では `ArgumentError`です。t4a-faerの LDLᴴ/LBLᴴは自己随伴専用で、
  LAPACK `zsytrf` 相当の転置対称分解を提供しないためです。
  `eigen`/`eigvals` も同じ規則で、格納三角が実数なら自己随伴 EVD（実固有値）、それ以外は一般 ネイティブ EVD
  （複素固有値）に渡します（spectral マニュアルの 自己随伴固有値計算の節）。
- `F.uplo`は Juliaの `Char`ではなく `Symbol`（`:L`/`:U`）です。LDLT/Bunch–Kaufmanは
  常に下三角向きの `F.L`（`A[p,p] = L D Lᴴ`、上三角ラッパーは格納三角の随伴を分解）を返し、
  Juliaの `uplo='U'`における `F.U`（`A[p,p] = U D Uᴴ`）は提供しません。`F.L'`は
  `Uᴴ D U`を満たすだけで Juliaの `U`にならず、反転による上三角への付け替え
  （`U = J L J`）は LAPACK `sytrf`の上三角因子と一致せず `F.D`/`F.p`の意味も変えるためです。
  密な `ldlt(A)`は Julia 標準には無く（`SymTridiagonal`・疎のみ）、Colmatでは
  ピボットなし LDLᴴとして Hermitian 入力を受けます。`cholesky[!](A, NoPivot())` も同じ ネイティブ 経路です。

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

- 結果を変える faerの設定は呼び出しごとのキーワードです。`cholesky`・`ldlt`（`!`、
  `scratch=`の直接経路、`team=`、`faerview`・ラッパー入力を含む）の
  `regularization_delta`/`regularization_epsilon`は faerの `LltRegularization`/
  `LdltRegularization`を有効にします（両方正のときのみ。片方だけ・負・非有限は `ArgumentError`）。
  分解される行列が `A`に対角摂動を加えたものに変わるため、意図した場合（前処理など）に限って使います。
  `ldlt`の `regularization_signs`（各行 -1/0/1）は期待する符号です。`F.regularization_count`は
  faerの `dynamic_regularization_count`です。
- 固定版 faerの LLᴴ 正則化は置換したピボットの対角を誤って格納し、ブロック化経路では因子全体が
  壊れます。Colmatは同じ置換規則・カウントを持つ faerの LDLᴴ カーネル（全符号 +1）で分解し、
  `sqrt(D)`を掛けて Lを作ります。LDLᴴのカウントは期待符号 +1のピボットだけを数え、
  -1 方向の補正や符号なしの置換は数えません（どちらも
  [実行済み再現例](https://github.com/AtelierArith/Colmat.jl/tree/main/docs/reproducers/faer-selfadjoint-factor-params)）。
- `bunchkaufman[!](A, rook; pivoting)`は `LbltParams.pivoting`の全戦略
  `:partial`・`:partial_diag`（既定）・`:rook`・`:rook_diag`・`:full`を選べます。`F.rook`は
  `:rook`/`:rook_diag`で `true`、`rook=true`と他の戦略の組み合わせは `ArgumentError`です。
  `:partial`は LAPACKの古典的 Bunch–Kaufmanと同じ探索規則ですが、ピボット列が一致するとは限りません
  （9×9の乱数対称行列 200 例で `bunchkaufman(Symmetric(A, :L)).p`と一致したのは 159 例）。
  非推奨の `Diagonal`は提供しません。`scratch_requirements(bunchkaufman!, A, rook; pivoting)` も同じ戦略を受けます。
- 性能だけに効く `LltParams`/`LdltParams`の `recursion_threshold`・`block_size`と
  `LbltParams`の `block_size`・`par_threshold`（`:full`のみが参照）は
  `with_cholesky_tuning(f; llt_recursion_threshold, llt_block_size, ldlt_recursion_threshold,
  ldlt_block_size, lblt_block_size, lblt_par_threshold)` のスコープで指定します（`nothing`は faer 既定）。
  保持因子・型付き・借用入力・team・一時保存／直接の `!`と scratch 照会がすべて従います。LBLᴴの直接経路の
  作業領域は `lblt_block_size`で変わるため、照会もスコープ内で行います。faerの `LbltInfo.transposition_count`
 は公開しません（置換そのものは `F.p`で得られ、対称置換は行列式に影響しないため `det`は `B`のブロックだけから求めます）。

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

- `with_lu_tuning(f; partial_recursion_threshold, partial_par_threshold, full_par_threshold)`は
  faerの `PartialPivLuParams.{recursion_threshold, par_threshold}`と `FullPivLuParams.par_threshold`、
  `with_qr_tuning(f; blocking_threshold, pivoted_par_threshold)`は `QrParams.blocking_threshold`と
  `ColPivQrParams.par_threshold`をスコープ内のすべての `lu`/`full_piv_lu`/`qr`/`qr(A, ColumnNorm())`
  構築に適用します（保存因子・型付き・借用入力・`team=`・一時保存／直接の `!`、`inv!(A)`、
  混合スカラー求解の昇格再分解、scratch 照会・`FaerScratch`・`reserve!`）。`nothing`は faer 既定で、
  2 つのスコープは同じ ネイティブ 状態の別の欄を持つため、どちらの順にも入れ子にできます。
  これらは再帰・ブロック化・並列化だけを変え、分解は丸め誤差の範囲で同じです。
- faerの `PartialPivLu::new` などは既定パラメータしか渡さず、既存の因子から組み立てることもできないため、
  Colmatは保存 LU/QR 因子を自前の型で持ち、低水準の `*_in_place` カーネルを呼びます。既定値のままなら
  faerの型と bit 一致します（Rustの単体テストで確認）。
- `PartialPivLuParams`/`FullPivLuParams`の ランクの打ち切り（`max_rank`・`abs_tol`・`rel_tol`）は分解を途中で止め、
  faerの求解・逆行列・再構成はそのような因子を受け付けないため公開しません（列ピボット QRの ランクは
  `rank(F; atol, rtol)`と許容誤差付き `ldiv!`で扱います）。`PartialPivLuParams.block_size`と
  `ColPivQrParams.blocking_threshold`は 固定した faerの カーネルが読まないため公開しません。
  `with_svd_tuning`の `qr_blocking`は SVD 内部の QR 用で、このスコープとは独立です。
  保存因子のない `A \ B`・`det(A)`・QZ/Schur・疎分解の内部 LU/QRは faer 既定のままです。

## SVDの作業領域と更新形式

`svd`、`svdvals`、`svd!`、`svdvals!`は `scratch`と `team`に対応します。`full`の設定も必要量の照会に合わせます。返り値の配列は作業領域とは別に確保されます。

`svd!`と `svdvals!`は成功後に入力を長方形の対角行列 Σへ置き換えます。Juliaの LAPACKの作業行列をそのまま公開する形式ではありません。容量不足や計算失敗時は書き戻しません。

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

## 分解から元の行列を復元する

`reconstruct(F)`は保存因子から、入力時の行・列の順序で独立した行列を作ります。密分解は `FaerMatrix`、疎 LUと疎 Choleskyは `FaerSparseMatrix`、疎 QRは密な `FaerMatrix`を返します。一般化 Schur／Hessenbergは名前付きの `(A, B)`を返します。

`Matrix(F)` や型を指定した変換も利用できます。直接分解の因子は入力の格納領域を読むため、その入力が閉じられたりサイズ変更されたりすると例外になります。復元済みの独立した行列は、その後に因子を閉じても使えます。

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
```

## 保存した分解で作業領域を再利用する

同じ Aに対して右辺だけを変えるなら、分解 `F`を保存して求解を繰り返します。`ldiv!(X, F, B; scratch, team)`と `rdiv!(X, B, F; scratch, team)`に対応します。2 引数の更新形式は正方の因子に限られます。

同じ数値型の書き込み可能な右辺では、その格納領域を直接使う経路があります。型の昇格、ランク不足、長方形の因子などでは別の作業配列が必要になります。構築時と求解時のチームは別に選べ、どちらを閉じても独立した保存因子は使えます。

必要量は演算と同じ引数順で照会します。入力と出力の型や許容誤差も合わせてください。

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
```

## SVD・固有値分解を Rust 側に保存する

`svd(A; native=true)`と自己随伴行列の `eigen(A; native=true)`は `FaerSpectralFactor`を返します。分解を保存して `pinv`、`rank`、`reconstruct`、求解を繰り返すための形式です。通常の呼び出しの返り値は変わりません。

`F \ B`は保存因子を使い、小さな特異値・固有値を許容誤差で切り捨てた最小ノルム解を求めます。疑似逆行列全体を作る必要はありません。取り出した成分は独立したコピーで、変更しても Fは変わりません。構築には `team`を指定できますが、呼び出し元の `scratch`は指定できません。求解・疑似逆行列の作業領域は別途指定できます。

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
```

## 保存因子の疑似逆行列と作業領域

`FaerScratch(pinv, F; team)`と `pinv(F; scratch, atol, rtol, team)`で作業領域を再利用します。SVD や固有値分解はやり直しません。しきい値は保存した値に適用し、返される行列は独立して確保します。容量不足なら `reserve!`で拡張してください。

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
```
