# 疎行列と直接法

CSC/CSR、疎構造、ordering、symbolic analysis と数値分解を扱います。

## 疎行列の格納と演算

疎行列は CSC／CSR の実体を Rust に所有させます。

```julia
using Colmat, LinearAlgebra, SparseArrays

S = FaerSparseMatrix(sparse([1.0 0 2; 0 3 0]); format=:csr)
T = FaerSparseMatrix(sparse([0.0 1 0; 4 0 0]))
@assert Matrix(S + T) ≈ Matrix(S) + Matrix(T)
C = S * S'                       # 結果も Rust 所有の疎行列
@assert SparseMatrixCSC(C) ≈ sparse(S) * sparse(S)'
x = FaerVector([1.0, 2.0, 3.0])
y = S * x
@assert Vector(y) ≈ sparse(S) * Vector(x)

P = FaerSparsity(S)               # 数値を持たない独立した疎構造
R = FaerSparseMatrix{Float64}(undef, P)
fill!(nonzeros(R), 5)             # 格納位置だけを更新
@assert nnz(R) == nnz(P)
U = pattern_union(P, FaerSparsity(T))
@assert nnz(U) == 5
v = nonzeros(S)                   # CSR では行順の live view
v[1] = 7
@assert S[1, 1] == 7
storage = compressed_storage(S)  # 1-based の独立した配列スナップショット
julia_csc = SparseMatrixCSC(S)    # Julia CSC への明示変換
```

構築・コピー・CSC/CSR 変換・転置／随伴・加減算・スカラー倍・疎×疎の積は
疎構造を維持します。疎×密・密×疎の積も、疎入力を密行列化せず native
ルーチンで計算します。通常の `sum` と entrywise `norm(A,1/2/Inf)` も接続済みです。
`FaerSparseMatrix(I,J,V,m,n)` では重複する triplet を加算し、格納ゼロを保持します。
`Matrix(S)`・`FaerMatrix(S)` は明示的に密行列を作ります。

### Julia CSC の借用 (`faerview`)

`faerview(S::SparseMatrixCSC{T,Ti})`（`T` は 4 型、`Ti` は任意の整数型）は
`FaerBorrowedSparseMatrix` を返し、各ネイティブ呼び出しの間だけ faer の
`SparseColMatRef<usize,T>` を渡します。格納値 `nonzeros(S)` はコピーせず読みます。
faer の疎構造は常に 0-based なので、Julia の 1-based な `colptr`/`rowval` は
そのままでは借用できません。呼び出しごとにビューが持つ再利用バッファ
（`n+1` 語と `nnz` 語の `UInt`）へ変換し、単調性・範囲・行の昇順一意を検査してから
Rust に渡します（容量に達した後は割り当てなし）。何もキャッシュしないため、
値の変更も `S[i,j]=x` による構造変更も次の呼び出しに反映されます。

接続済みの経路は、密な native 行列・ビュー・転置／随伴・`faerview` した Julia
配列・Julia 配列との積（`*`、`FaerMatrix` への5引数 `mul!`、左右両方）、
`copy`/`FaerSparseMatrix`/`FaerSparsity`（1回の所有コピー）、`lu`・`qr`・
`cholesky`・`factorize(plan,B)`・`FaerSparseSymbolic(B)`・`B \ x`・
`lu!`/`qr!`/`cholesky!` です。分解は従来どおり所有 snapshot を保持するため、
後で `S` を書き換えても分解済みの因子は変わりません。

```julia
using Colmat, LinearAlgebra, SparseArrays

S = sparse([4.0 1 0; 1 5 2; 0 2 6])
B = faerview(S)                    # 値はコピーしない
X = FaerMatrix(rand(3, 2))
C = FaerMatrix(zeros(3, 2))
mul!(C, B, X, 2, 0)                # native 積を C に直接書き込む
@assert Matrix(C) ≈ 2 * S * Matrix(X)
F = lu(B)
S.nzval .*= 2                      # 借用ビューには反映、F は不変
lu!(F, B)                          # 現在の値で再分解
@assert F \ [1.0, 2, 3] ≈ (2 * Matrix(sparse([4.0 1 0; 1 5 2; 0 2 6]))) \ [1.0, 2, 3]
```

CSR の借用（`transpose(S)` 由来）、借用を代入先にする `SparseColMatMut` 経路、
`u32` index での faer 呼び出し、非圧縮形式（faer の `col_nnz`）は未接続です。
Julia の `SparseMatrixCSC` は常に圧縮形式なので、非圧縮形式は Julia 入力では
生じません。

### 疎×疎積の計画再利用 (`FaerSparseProductPlan`)

`A * B` は呼び出しごとに faer の symbolic 段（結果の疎構造と作業分割）と
numeric 段を実行し、結果の疎行列を新しく割り当てます。疎構造が固定で値だけが
変わる反復では、`FaerSparseProductPlan(A, B)` で
`sparse_sparse_matmul_symbolic` を1回だけ実行し、
`mul!(C, plan, A, B, α=1, β=0; scratch, team)` で
`sparse_sparse_matmul_numeric` だけを繰り返せます。

- plan は `A`・`B` の格納パターン（格納ゼロを含む）の独立コピーを持ちます。
  毎回の `mul!` は、`A`・`B` のパターンと格納形式、および `C` のパターンが
  plan と完全に一致するかを割り当てなしで検査します。不一致なら `C` に
  書き込む前に `ArgumentError` を投げます。`C` のパターンが結果パターンより
  小さいと faer の作業列が汚れるため、`C` は結果パターンと完全一致が必要です。
- `C` には `FaerSparseMatrix{T}(plan)`（ゼロ値）か `A * B` を使います。
  `mul!` は `C` のパターン・nnz・`epoch`・`nonzeros(C)` の live view を維持します。
  `β == 0` は上書き、それ以外は `C` を `β` 倍して加算します。
- `A`・`B`・`C` は同じ CSC か CSR です（CSR は `Cᵀ = Bᵀ Aᵀ` の列優先 plan）。
  `faerview(S)` の借用 Julia CSC 同士なら値も index 変換バッファ経由でコピーせずに
  読みます。借用と native CSC の混在では借用側を1回コピーします。
- `eltype(C)` は `promote_type(eltype(A), eltype(B))` を保持できる型です。
  型の異なる入力は `eltype(C)` へ変換したコピー（同じパターン）を作ります。
  `C === A`／`C === B` は `ArgumentError` です。
- numeric 段の作業領域はスレッドごとに `eltype(C)` の密な1列です。
  `scratch_requirements(mul!, C, plan, A, B; team)`、
  `FaerScratch(mul!, C, plan, A, B; team)`、
  `reserve!(work, mul!, C, plan, A, B; team)` で用意します。`scratch` を省略すると
  一時領域を割り当てて解放します。`FaerTeam` を渡すと faer の費用モデルに従って
  並列実行します（小さな積は逐次のままです）。容量不足も書き込み前に検出します。

```julia
using Colmat, LinearAlgebra, SparseArrays

A = FaerSparseMatrix(sprand(200, 200, 0.02))
B = FaerSparseMatrix(sprand(200, 200, 0.02))
plan = FaerSparseProductPlan(A, B)       # symbolic 段は1回だけ
C = FaerSparseMatrix{Float64}(plan)      # 結果パターンのゼロ行列
work = FaerScratch(mul!, C, plan, A, B)
for k in 1:3
    fill!(nonzeros(A), k)                # 値だけを変更
    mul!(C, plan, A, B; scratch=work)    # numeric 段だけを実行
end
@assert Matrix(C) ≈ Matrix(A) * Matrix(B)
```

同型の入力と caller の `scratch` を使う場合、Rust 側の numeric 呼び出しは
割り当てを行いません（Rust の allocation probe で検証）。Julia 側には FFI 呼び出し
1回分の固定 bookkeeping だけが残ります。400×400・密度 2% の測定では、
Julia ヒープ上の割り当ては従来の `mul!(C, A, B)` の 7712 バイト（これに加えて
native 側で symbolic 構造・結果値・新しい owner を毎回割り当て）に対し、
plan 経由は 1648 バイトで native 側の割り当てはありません。
借用 Julia CSC の plan 経由では 0-based index バッファの検査・変換が加わります。

疎三角ラッパーの `\` と2引数・3引数の `ldiv!` は native の疎求解を使います。
通常・単位・下三角・上三角、転置／随伴、型の promotion に対応します。
単位三角では未格納の対角を 1 として扱い、通常の三角行列で対角がゼロなら
`SingularException` を投げます。右辺が `FaerMatrix`／`FaerVector` なら結果も Rust 所有です。

```julia
using Colmat, LinearAlgebra, SparseArrays

A = FaerSparseMatrix(sparse([3.0 99; 2 4]); format=:csr)
b = FaerVector([3.0, 10.0])
x = LowerTriangular(A) \ b
@assert Vector(x) ≈ [1.0, 2.0]
ldiv!(UnitLowerTriangular(A), b)
@assert Vector(b) ≈ [3.0, 4.0]
```

`ldiv!(T, B)` は、`B` が `promote_type(eltype(T), eltype(B))` の要素型を持つ
`FaerMatrix`／`FaerVector`・native view・`faerview` 借用・重複要素のない Julia strided
配列なら、faer の `sp_solve_{unit_}{lower,upper}_triangular_in_place` を `B` のメモリ上で
直接実行します。結果用の行列や右辺の複製は作りません。CSR 格納と `transpose`／`adjoint`
は列優先の kernel とその転置版・共役 flag に写像するため、疎行列も複製しません。
格納済みの疎行列は、すべての列が faer の前提を満たす場合にそのまま借用されます。
前提は、選択した半分だけが格納され、対角が下三角なら先頭、上三角なら末尾にあることです。
単位三角でも対角の格納が必要です。両半分を格納した行列や対角のない単位三角では、
三角部分を caller の `scratch` 上に組み直します。実数の疎行列と複素の右辺のように
型を昇格する場合は、昇格した値だけを `scratch` に置きます。
`scratch_requirements(ldiv!, T, B)` は、この組み直しと昇格に必要な byte 数を返します。
前提を満たす同型の場合は 0 です。`FaerScratch(ldiv!, T, B)` と
`reserve!(work, ldiv!, T, B)` も使えます。caller の scratch を渡し、`Y === B` の場合、
Rust 側の求解は割り当てを行いません。Julia 側には n に依存しない数 KB の固定 bookkeeping
（lock・view 記述子・必要量と容量の照会・FFI 引数）だけが残ります。`ldiv!(Y, T, B)` で `Y !== B` なら `B` を `Y` に
複製してから解きます。要素型の狭い出力や strided でない出力では、昇格型の作業コピーを
1つ作り、書き戻すときに変換します。`team=` は検証しますが、faer の疎三角 kernel は逐次実行です。
対角のゼロ、close 済みの行列・view・scratch・team、容量不足はいずれも出力に書き込む前に検出します。

```julia
using Colmat, LinearAlgebra, SparseArrays

L = FaerSparseMatrix(sparse([2.0 0; 1 4]))       # 対角が先頭の下三角
B = FaerMatrix([2.0 4; 9 6])
@assert scratch_requirements(ldiv!, LowerTriangular(L), B).bytes == 0
work = FaerScratch(ldiv!, UpperTriangular(transpose(L)), B)
ldiv!(UpperTriangular(transpose(L)), B; scratch=work)   # Lᵀ x = B を B 上で解く
@assert Matrix(B) ≈ UpperTriangular([2.0 1; 0 4]) \ [2.0 4; 9 6]
```

疎行列の `lu`・`cholesky`・`qr` は再利用可能な `FaerSparseFactorization` を返します。

### 疎行列全体の broadcast・`map`

`FaerSparseMatrix` を含む broadcast と `map`／`map!` は、SparseArrays と同じ結果規約に従います。
結果は疎行列で、`f` が生成したゼロは格納しません（入力の格納ゼロも同様です）。
結果の数値型が4型のいずれかなら `FaerSparseMatrix` を返し、CSC／CSR 形式は最初の疎入力に合わせます。

```julia
S = FaerSparseMatrix(sparse([1.0 0 -2; 0 0 3]); format=:csr)
A = abs.(S)                       # 値だけを変換。構造は S のコピー
B = 2 .* S .^ 2                   # 融合した式も同じ経路
C = map(x -> x / 4, S)
v = nonzeros(S)
S .*= 0.5                         # 構造が変わらなければ値を一括で書き戻す
@assert isopen(v)                 # 既存の値ビューも有効なまま
D = S .+ 1                        # f(0) ≠ 0。SparseArrays と同じく全要素を格納
mask = S .> 0                     # Bool は native 型ではないため SparseMatrixCSC
```

次の条件をすべて満たす場合は、native の高速経路で計算します。

- 疎入力が1つの `FaerSparseMatrix` だけ（同じ行列の複数回参照は可）
- ほかの引数が `Number`／`Ref`／0次元配列のスカラー
- `f(0, スカラー...)` がゼロ（`S .* Inf` は `0*Inf` が NaN になるため対象外）
- 結果の数値型が native 型

この経路では、格納値を Julia に1回だけ転送し、Julia 側で `f` を適用します。
その値を元の疎構造のコピーに付けて新しい owner を作ります。
出力先が入力自身の `S .= f.(S)`／`S .*= α` では値を一括で書き戻します。
このとき nnz・格納順・値ビューは維持されます。
ゼロが生じた場合は `dropzeros!` 相当の処理で構造を変え、以前の値ビューを失効させます。
出力の型への変換に失敗した場合（実数行列に複素数を書く等）は、書き込み前に例外を投げます。
400×400・密度 2% の `abs.(S)` では、Julia ヒープの割り当てが約 230 MB から約 33 KB になりました。

上記以外（`f(0)≠0`、複数の疎入力、密配列・`SparseMatrixCSC` との混在）では、
各 native 入力を `SparseMatrixCSC` に1回変換して SparseArrays の broadcast を使います。
結果の数値型が native なら `FaerSparseMatrix` に戻し、それ以外（`Bool`・`BigFloat` 等）は `SparseMatrixCSC` のまま返します。
`S .= x` のようなスカラー代入は Base の規約どおり `fill!` になります。
`x == 0` では構造を保ち、非ゼロでは暗黙のゼロ位置を埋めることになるため `ArgumentError` です。
密な出力先への `D .+= S` は、疎入力を変換してから Julia の broadcast で計算します。
`map(f, A, B...)` は全引数が同じサイズである必要があり、異なると `DimensionMismatch` です。
疎行列の転置／随伴ラッパーの broadcast は、この経路の対象外です。

## symbolic analysis と数値分解

`A \ b` は正方行列なら LU、縦長なら QR による最小二乗求解を選びます。

```julia
using Colmat, LinearAlgebra, SparseArrays

A = FaerSparseMatrix(sparse([4.0 1; 1 3]))
b = FaerVector([1.0, 2.0])
F = lu(A)
@assert Vector(F \ b) ≈ [4.0 1; 1 3] \ Vector(b)
H = cholesky(Hermitian(A, :L))
@assert Vector(H \ b) ≈ Vector(F \ b)

plan = FaerSparseSymbolic(FaerSparsity(A); kind=:lu)
B = 2A
G = factorize(plan, B)           # symbolic analysis を再利用
lu!(F, B)                        # F の数値分解を更新
@assert Vector(F \ b) ≈ Vector(G \ b)

R = FaerSparseMatrix(sparse([1.0 0; 1 1; 0 1]))
y = qr(R) \ [1.0, 2.0, 2.0]
@assert y ≈ [1.0 0; 1 1; 0 1] \ [1.0, 2.0, 2.0]
```

`FaerSparseSymbolic` の `kind` は `:lu`・`:qr`・`:cholesky` です。
`factorize(plan, A)` と `lu!(F,A)`・`qr!(F,A)`・`cholesky!(F,A)` では、
格納ゼロを含むパターンが完全に一致する必要があります。CSC／CSR と数値型は
変更できますが、更新先の分解を渡す `!` 形式では数値型を維持します。
分解は元の行列と plan に依存せず使え、`close` にも対応します。
混合精度の求解は保存済みの疎因子を用います。作業領域と型変換の詳細は、各分解の求解の節を参照してください。`ldiv!` は一時結果を経由して更新します。
LU・Cholesky は転置／随伴の求解にも対応します。
疎 QR は横長・ランク落ち・転置／随伴の求解にも対応します。
成分取得、ordering、最小ノルム求解の詳細は [疎 QR](sparse-qr.md) を参照してください。

## ordering

fill-in を減らすための ordering を単独で計算できます。
`amd` は正方行列の `A + transpose(A)` のパターン、`colamd` は長方形も含む
列パターンを使い、どちらも `FaerPermutation` を返します。

```julia
using Colmat, LinearAlgebra, SparseArrays

A = FaerSparseMatrix(sparse([4.0 0 1; 0 3 1; 1 1 5]); format=:csr)
p = amd(A)
B = A[p, p]                      # CSC/CSR を維持する native の置換
@assert sparse(B) == sparse(A)[Vector(p), Vector(p)]
info = amd(FaerSparsity(A); dense=10.0, aggressive=true, statistics=true)
@assert Vector(info.permutation) == Vector(p)
@assert info.n_div >= 0
q = colamd(A; dense_row=0.5, dense_col=0.5, aggressive=true)
C = permute_cols(A, q)
@assert sparse(C) == sparse(A)[:, Vector(q)]
```

ordering は値に依存せず、格納ゼロもパターンに含めます。数値を持たない
`FaerSparsity`、通常の Julia CSC、転置／随伴にも対応します。
`Hermitian(A,:L)` などでは選択した三角部分のパターンを反映します。
AMD の `statistics=true` は置換と faer の演算数見積もりを返します。
`permute_rows`・`permute_cols` は疎数値行列と疎パターンに対応し、数値行列は
`permute_rows!`・`permute_cols!` でも更新できます。構造が変わる更新では以前の
`nonzeros` ビューが失効します。疎 LU／QR への恒等・任意 ordering の直接指定にも対応します。
`amd` / `colamd` は逐次 native kernel の caller-owned scratch に対応します。
query は格納パターン（格納ゼロを含む）から必要量を計算し、数値には依存しません。
容量・alignment 不足や閉じた workspace は拒否し、自動拡張はしません。
CSC の複製／CSR 変換、Julia からのパターン構築、返却する置換は別途確保します。
symbolic analysis 内の ordering 用 scratch は `FaerSparseSymbolic(A; scratch=s)` に渡します（下記）。

```julia
using Colmat, SparseArrays
A = FaerSparseMatrix(sparse([4.0 0 1; 0 3 1; 1 1 5]); format=:csr)
work = FaerScratch(amd, A)
info = amd(A; scratch=work, statistics=true)
@assert isperm(Vector(info.permutation))
reserve!(work, colamd, A)
q = colamd(A; scratch=work)
@assert Vector(q) == Vector(colamd(A))
close(work)
```

`scratch_requirements(FaerSparseSymbolic, A; kwargs...)`・`FaerScratch`・`reserve!` は
`FaerSparseSymbolic` と同じキーワード（`kind`・`uplo`・`ordering`・`algorithm`・
`dense`・`dense_row`・`dense_col`・`aggressive`）で、symbolic analysis 内の ordering
kernel 用 workspace を照会します。対象は次の経路です。

- Cholesky の `:default`／`:amd`: faer の symbolic analysis と同じ入力・設定の AMD を
  caller scratch で実行し、得た置換を faer に明示指定します。`algorithm=:auto` では
  simplicial／supernodal の選択を scratch なしの場合と一致させるため、因子の列数も
  同じ scratch で計算します。置換・方式・`factor_storage` は scratch なしと同一です。
  この列数計算の分だけ `:auto` は scratch なしより遅くなります（測定例で約 15–30%）。
  `:simplicial`／`:supernodal` を明示すると追加計算はありません。
- 横長 QR の既定 COLAMD と、明示 simplicial LU／QR の `ordering=:colamd`: COLAMD kernel の
  必要量です。
- 恒等・任意 ordering と、正方／縦長の auto／supernodal LU／QR は 0 バイトです。
  後者の COLAMD は caller stack を受け取らない faer の symbolic API 内で確保されます。
  明示 ordering を渡しても faer の内部要求量に COLAMD 分が残るため、事前計算では減りません。

ordering kernel は逐次実行です。容量・alignment 不足や閉じた workspace は拒否します。
CSC への変換・複製、faer の ordering 後の symbolic 用内部バッファ、保持する置換と
symbolic 構造は必要な出力として別途確保します。一括の `cholesky`／`lu`／`qr` の
`scratch` は数値分解用のままです。symbolic 側の workspace も管理する場合は
`FaerSparseSymbolic` と `factorize(plan, A)` の二段階で呼び出します。

```julia
using Colmat, LinearAlgebra, SparseArrays
A = FaerSparseMatrix(sparse([4.0 0 1; 0 3 1; 1 1 5]))
s = FaerScratch(FaerSparseSymbolic, A; kind=:cholesky)
plan = FaerSparseSymbolic(A; kind=:cholesky, scratch=s)
@assert Vector(plan.p) == Vector(FaerSparseSymbolic(A; kind=:cholesky).p)
F = factorize(plan, A)
@assert F \ ones(3) ≈ [4.0 0 1; 0 3 1; 1 1 5] \ ones(3)
close(s)
```

## 疎 Cholesky と分解成分

疎 Cholesky には ordering と分解方式を直接指定できます。

```julia
using Colmat, LinearAlgebra, SparseArrays

A = FaerSparseMatrix(sparse([4.0 0 1; 0 3 1; 1 1 5]))
p = amd(A)
F = cholesky(Hermitian(A, :L); perm=Vector(p), algorithm=:simplicial)
@assert Vector(F.p) == Vector(p)
@assert F.algorithm == :simplicial
@assert F.factor_storage >= size(A, 1)
@assert F \ ones(3) ≈ [4.0 0 1; 0 3 1; 1 1 5] \ ones(3)

plan = FaerSparseSymbolic(A; kind=:cholesky, ordering=:identity,
                         algorithm=:supernodal)
G = factorize(plan, A)
@assert Vector(G.p) == 1:3
@assert G.algorithm == :supernodal
```

`ordering=:default`／`:amd` は AMD、`:identity` は並べ替えなしです。
任意の置換も渡せます。`cholesky` では Julia 標準の疎行列 API と同じ
`perm=p` も使えます。`perm` と `ordering` の同時指定はエラーです。
`dense`・`aggressive` は AMD の設定、`algorithm` は `:auto`・`:simplicial`・
`:supernodal` です。`F.p`／`plan.p` は独立した置換、`.algorithm` は実際の分解方式、
`.factor_storage` は確保する数値スロット数（supernodal の padding を含む）を返します。
コピー・数値更新・混合精度の求解でも symbolic analysis の設定を引き継ぎます。
LU／QR の ordering と数値分解の `scratch`・`team` については、このページおよび [疎 QR](sparse-qr.md) を参照してください。

疎 Cholesky の `F.L`・`F.U` は独立した Rust 所有の CSC 行列です。
元の分解を更新・close しても、取り出した行列は使えます。

```julia
using Colmat, LinearAlgebra, SparseArrays

A = FaerSparseMatrix(sparse([4.0 0 1; 0 3 1; 1 1 5]); format=:csr)
F = cholesky(A; perm=[3, 1, 2], algorithm=:supernodal)
L, U, p = F.L, F.U, Vector(F.p)
@assert sparse(L * U) ≈ sparse(A)[p, p]
@assert Matrix(F) ≈ Matrix(A)     # 元の行・列の順序で再構成
R = FaerSparseMatrix(F)           # native の疎積・置換で再構成
@assert sparse(R) ≈ sparse(A)
@assert logdet(F) ≈ logdet(cholesky(Hermitian(Matrix(A))))
@assert det(F) ≈ det(Matrix(A))
@assert logabsdet(F)[2] == 1
```

`FaerSparseMatrix(F)`・`sparse(F)` は疎行列を返します。再構成は分解と同じ精度で行い、
積によって追加の格納位置や丸め誤差が生じる場合があります。
`Matrix(F)`・`FaerMatrix(F)` は明示的に密行列を作ります。
`logdet`・`logabsdet` は Rust の分解バッファから対角だけを取得して計算し、
密行列化しません。失敗した分解の要素取得・再構成・行列式は `PosDefException` になります。

## 疎構造を保つ更新

既存の疎構造を保った加減算には `add_assign!`・`sub_assign!`・
`add_into!`・`sub_into!` を使えます。

```julia
using Colmat, SparseArrays

D = FaerSparseMatrix(sparse([1.0 2; 3 4]); format=:csr)
S = FaerSparseMatrix(sparse([2.0 0; 1 0]))
values = nonzeros(D)
add_assign!(D, S)                 # D += S、格納位置は変更しない
@assert Matrix(D) == [3.0 2; 4 4]
@assert Vector(values) == compressed_storage(D).values
sub_assign!(D, S)
@assert Matrix(D) == [1.0 2; 3 4]
add_into!(D, S, S)                # D = S + S、未使用の格納位置も 0 に更新
@assert Matrix(D) == [4.0 0; 2 0]
@assert nnz(D) == 4
sub_into!(D, D, S)                # 入出力の共有にも対応
@assert Matrix(D) == Matrix(S)
```

入力の全格納位置は出力の格納位置に含まれている必要があります。格納ゼロも対象で、
不足すると更新前に `ArgumentError` を投げます。サイズ・型変換の失敗でも出力は変わりません。
入力は出力の数値型と CSC／CSR 形式にコピー・変換してから native カーネルで計算し、
出力の型・構造・格納数を維持します。値ビューも失効しません。
通常の Julia CSC と native の転置／随伴を入力に使えます。
一般の broadcast（`D .+= S` 等）はこの専用 API とは別の実装で、SparseArrays と同じくゼロを除去し構造を変えることがあります。

native 配列に対する `givens` は faer で係数を計算し、Julia 標準の

## 格納値ビューと変換

`copyto!` は以前の値ビューを失効させます。`FaerSparsity` は元の行列の変更・close に
影響されません。`FaerSparseMatrix{T}(P)` は Boolean パターンを数値の 1 に変換し、
`FaerSparseMatrix{T}(undef,P)` は同じ格納位置を 0 で初期化します。
`FaerSparseMatrix(P, values)` で格納順の値をまとめて設定できます。

`rowvals`・`getcolptr` は CSC のみ対応し、返す添字配列は独立したコピーです。
CSR の行ポインタ・列添字には `compressed_storage` を使ってください。
疎分解の残りの機能・低水準制御は引き続き実装対象です。broadcast/`map` の結果規約は
「疎行列全体の broadcast・`map`」を参照してください。

`full_piv_lu`、`ldlt`、`bunchkaufman`、一般・自己随伴・一般化の
`eigen`／`eigvals`、三角行列の求解・逆行列、`det`／`logabsdet`／`logdet`、

## Sparse LU components

```julia
using Colmat, LinearAlgebra, SparseArrays
A = FaerSparseMatrix(sparse([0.0 2 1; 3 1 0; 0 1 4]); format=:csr)
F = lu(A; algorithm=:simplicial, ordering=[3, 1, 2])
p, q = Vector(F.p), Vector(F.q)
@assert Matrix(F.L * F.U) ≈ Matrix(A)[p, q]
@assert Matrix(A) * (F \ ones(3)) ≈ ones(3)
```

Every sparse LU algorithm retains native saved factors and exposes
independent native CSC `F.L`/`F.U` and row/column `FaerPermutation` objects
`F.p`/`F.q`. They are the factors used by normal, transpose and adjoint solves.
CSC/CSR input and all four native scalar types are supported, including symbolic
plan reuse and numeric `lu!` updates. `ordering=:default` or `:identity` selects
identity columns for this explicit algorithm. Select `ordering=:colamd` to run
native COLAMD once during symbolic analysis, or supply an explicit permutation.
`dense_row=0.5`, `dense_col=0.5` and `aggressive=true` control COLAMD when selected.
Numeric `lu!` updates retain that column ordering.

The default `algorithm=:auto` uses faer's fill-reducing selection;
`algorithm=:supernodal` forces its supernodal path. Both expose saved components
and accept default/COLAMD ordering with `dense_row`, `dense_col`, `aggressive`.
Every LU algorithm accepts `ordering=:identity` or a supplied column permutation.
Sparse component extraction allocates independent CSC storage.

```julia
using Colmat, LinearAlgebra, SparseArrays
A0 = ComplexF64[0 2+im 1; 3 1 0; 0 1 4-im]
F = lu(FaerSparseMatrix(sparse(A0)); algorithm=:supernodal)
L, U, p, q = F.L, F.U, Vector(F.p), Vector(F.q)
@assert Matrix(L * U) ≈ A0[p, q]
@assert Matrix(F) ≈ A0
close(F)
@assert Matrix(L * U) ≈ A0[p, q]
```

## COLAMD ordering in sparse LU/QR

```julia
using Colmat, LinearAlgebra, SparseArrays
A0 = [8.0 1 2; 1 5 0; 2 0 6; 1 0 1]
A = FaerSparseMatrix(sparse(A0); format=:csr)
plan = FaerSparseSymbolic(A; kind=:qr, algorithm=:simplicial,
                          ordering=:colamd, dense_row=-1.0, aggressive=false)
F = factorize(plan, A)
p = Vector(F.pcol)
@assert Matrix(F.Q) * Matrix(F.R) ≈ A0[:, p]
@assert Matrix(F) ≈ A0
B = FaerSparseMatrix(sparse(2A0))
qr!(F, B)
@assert Vector(F.pcol) == p
@assert Matrix(F) ≈ 2A0
close(plan)
close(F)
close(A)
close(B)
```

`lu(A; algorithm=:simplicial, ordering=:colamd, ...)` and `qr` accept the
same COLAMD controls as `FaerSparseSymbolic`. Stored zeros participate in the
ordering; numeric values do not. The generated permutation is copied into
Rust-owned analysis, so no temporary ordering handle needs to remain open.
Every sparse LU/QR algorithm accepts this explicit ordering option.

## Native preconditioning from sparse LU

```julia
using Colmat, LinearAlgebra, SparseArrays
A0 = [0.0 2 1; 3 1 0; 0 1 4]
A = FaerSparseMatrix(sparse(A0); format=:csr)
F = lu(A; algorithm=:simplicial, ordering=[3, 1, 2])
M = FaerLinearOperator(F)
close(F)
b = ones(3)
@assert M * b ≈ A0 \ b
@assert bicgstab(A, b; M=M).x ≈ A0 \ b
close(M)
close(A)
```

The inverse-action operator retains sparse CSC L/U and both permutations from
the current native factorization for every LU algorithm. It never uses the source snapshot
or a dense coefficient matrix, including when promoting factors for a mixed RHS.
Normal/conjugate/transpose/adjoint application uses sparse triangular solves
and an RHS-sized native scratch intermediate. Copies remain independent after
the original factors are updated or closed. All sparse LU algorithms expose the retained component interface used here.

## Native preconditioning from sparse Cholesky

```julia
using Colmat, LinearAlgebra, SparseArrays
A0 = ComplexF64[4 1im; -1im 3]
A = FaerSparseMatrix(sparse(A0); format=:csr)
F = cholesky(Hermitian(A, :U); algorithm=:supernodal, ordering=[2, 1])
M = FaerLinearOperator(F)
close(F)
b = ones(ComplexF64, 2)
@assert M * b ≈ A0 \ b
@assert cg(A, b; M=M).x ≈ A0 \ b
close(M)
close(A)
```

Sparse Cholesky inverse actions support auto/simplicial/supernodal factors,
both input triangles and retained symmetric ordering. Their independently owned
sparse L/U factors use nonunit triangular solves; mixed scalar promotion,
conjugation, transpose/adjoint and copy/composition preserve sparse storage.
The operator does not read or refactor the source snapshot. All sparse LU and QR algorithms also provide saved-factor inverse actions,
as described in their sections.

## Reuse workspace and teams for sparse LU/Cholesky solves

All three sparse LU/Cholesky algorithms support retained left/right solves with
`scratch` and `team`, including transpose/adjoint, vectors, native arrays and
mixed scalar RHS. Queries use the working scalar type and team; `reserve!`
explicitly grows a buffer when either changes.

```julia
using Colmat, LinearAlgebra, SparseArrays
A0 = ComplexF64[5 1+0.2im; 1-0.2im 4]
a = FaerSparseMatrix(sparse(A0))
F = cholesky(Hermitian(a); algorithm=:supernodal, ordering=[2,1])
team = FaerTeam(2)
B = ones(ComplexF64,2,3)
s = FaerScratch(ldiv!, F, B; team)
close(a)
close(F.source)
close(F.plan)
X = similar(B)
ldiv!(X,F,B; scratch=s,team)
@assert X ≈ A0 \ B
C = ones(Float32,3,2)
reserve!(s,rdiv!,C,transpose(F);team)
Y = similar(C,ComplexF64)
rdiv!(Y,C,transpose(F);scratch=s,team)
@assert Y ≈ C / transpose(A0)
close(s)
close(team)
close(F)
```

Same-type solves borrow the numeric factor directly. Mixed-type solves convert
saved sparse factors into an inverse operator without refactorizing the source;
preparing that operator allocates separately from solve scratch. Native output,
RHS conversion and destination staging also allocate separately. Overlapping
inputs are supported; conversion and resource failures leave destinations intact.
Sparse QR keeps its separate workspace interface and remaining restrictions.
