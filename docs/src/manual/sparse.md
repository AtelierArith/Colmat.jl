# 疎行列と直接法

ゼロが多い行列では、非ゼロの位置と値だけを保存する疎行列を使うとメモリを減らせます。`using SparseArrays`で Juliaの `sparse`を読み込み、`FaerSparseMatrix`へ渡します。

```julia
using Colmat, LinearAlgebra, SparseArrays
A = FaerSparseMatrix(sparse([4.0 1 0; 1 4 1; 0 1 4]))
b = ones(3)
x = lu(A) \ b
@assert Matrix(A) * x ≈ b
```

CSCは列ごと、CSRは行ごとに非ゼロを保存する形式です。初めは既定の CSCを使って構いません。「直接法」は LU などの分解を作って解く方法で、分解中に非ゼロが増える場合があります。構造を再利用する計画や並べ替えは、繰り返し計算の性能を改善したいときに使います。

## 疎行列の格納と演算

疎行列は CSC／CSRの実体を Rustに所有させます。

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
疎構造を維持します。疎×密・密×疎の積も、疎入力を密行列化せず ネイティブ
ルーチンで計算します。通常の `sum`と 要素ごとの `norm(A,1/2/Inf)` も接続済みです。
`FaerSparseMatrix(I,J,V,m,n)`では重複する 行・列・値の組を加算し、格納ゼロを保持します。
`Matrix(S)`・`FaerMatrix(S)`は明示的に密行列を作ります。

`kron`は `FaerSparseMatrix`・借用 `faerview(::SparseMatrixCSC)`・それらの転置／随伴の
いずれかを含むと、CSC（または転置した被演算子を通じて CSR）の圧縮構造を Rustで
直接組み立てた `FaerSparseMatrix`を返します（格納数 `nnz(A)*nnz(B)`、各列は整列済み）。
形式は最初の疎被演算子の実効 格納順（CSCの転置は CSR）に従い、形式の異なる
被演算子だけを変換します。SparseArraysと同じく、密な被演算子（ネイティブ・ビュー・
ベクトル・Julia 配列）は疎化して非ゼロだけを格納し、疎被演算子の格納ゼロは保持します。

```julia
using Colmat, LinearAlgebra, SparseArrays

s = sparse([1.0 0; 2 3])
K = kron(FaerSparseMatrix(s), faerview(s))
@assert K isa FaerSparseMatrix && nnz(K) == nnz(s)^2
@assert SparseMatrixCSC(K) == kron(s, s)
@assert SparseMatrixCSC(kron(FaerSparseMatrix(s), [1.0 0; 0 2])) == kron(s, sparse([1.0 0; 0 2]))
```

### Julia CSCの借用 (`faerview`)

`faerview(S::SparseMatrixCSC{T,Ti})`（`T`は 4 型、`Ti`は任意の整数型）は
`FaerBorrowedSparseMatrix`を返し、各ネイティブ呼び出しの間だけ faerの
`SparseColMatRef<usize,T>`を渡します。格納値 `nonzeros(S)`はコピーせず読みます。
faerの疎構造は常に 0 始まり なので、Juliaの 1 始まり な `colptr`/`rowval`は
そのままでは借用できません。呼び出しごとにビューが持つ再利用バッファ
（`n+1` 語と `nnz` 語の `UInt`）へ変換し、単調性・範囲・行の昇順一意を検査してから
Rustに渡します（容量に達した後は割り当てなし）。何もキャッシュしないため、
値の変更も `S[i,j]=x`による構造変更も次の呼び出しに反映されます。

接続済みの経路は、密な Rust 所有行列・ビュー・転置／随伴・`faerview` した Julia
配列・Julia 配列との積（`*`、`FaerMatrix`への5引数 `mul!`、左右両方）、
`copy`/`FaerSparseMatrix`/`FaerSparsity`（1回の所有コピー）、`lu`・`qr`・
`cholesky`・`factorize(plan,B)`・`FaerSparseSymbolic(B)`・`B \ x`・
`lu!`/`qr!`/`cholesky!`、疎三角求解（`LowerTriangular(B) \ x`・`ldiv!` など）、
計画を使わない疎×疎 `*`・疎加減算、`FaerLinearOperator(B)`です。分解は従来どおり
所有 スナップショットを保持するため、後で `S`を書き換えても分解済みの因子は変わりません。

- `FaerSparseSymbolic(B)`（`lu(B)` などの内部も同じ）は、呼び出しごとの 0 始まり
  バッファから直接 構造解析を行い、一時的な `FaerSparsity`を作りません。
  所有 CSCの `FaerSparseMatrix` も同様に直接解析します。計画が保持する 非ゼロ配置は
  1 つで、分解ごとの 計画のコピー（`copy(plan)`）はそれを共有します。したがって
  `lu(B)`の 非ゼロ配置は 計画 用と分解の スナップショット 用の 2 つだけです。両者は寿命の異なる
  所有物です: 計画は `factorize(plan, B2)`・`lu!(F, B2)`で後の入力の 非ゼロ配置を検査する
  （faerの数値分解は 非ゼロ配置 一致を前提とし検査しない）ために保持し、スナップショットは分解の
  値の出所です。faerの所有疎行列型は 非ゼロ配置を共有できないため、これ以上は減らせません。
  CSR 所有と、明示 simplicialの `ordering=:colamd`（Colmat 側で COLAMDを先に実行）は
  CSCの `FaerSparsity`を 1 回作ります。
- 疎三角求解は借用配列を faerの三角 カーネルにそのまま渡します（転置・随伴は列優先 カーネルの
  転置版・共役 フラグに写像）。scratch・昇格・前提を満たさない 格納順の扱いは所有行列と同じです。
  出力先が `S`の配列とメモリを共有する場合は書き込み前に `ArgumentError`です。
- 疎×疎 `*`・`+`・`-`は、借用（転置・随伴を含む）同士、借用と `FaerSparseMatrix`
  （転置・随伴を含む）の混在で、どの被演算子も所有コピーせずに読みます。結果は
  `FaerSparseMatrix`で、形式は左被演算子 `op(A)`の向きに従います（`faerview(S)`は CSC、
  その転置・随伴は CSR）。向きの異なる右被演算子は 1 回変換し、加減算で共役が必要な
  被演算子は共役コピーを作ります。
- `FaerLinearOperator(B)`（`transpose(B)`・`B'` も。`Hermitian(B)`は Julia CSCと同じく
  Julia 側で三角部分を展開してから）は構築時に 1 回だけ スナップショットを取ります。`cg`・
  `bicgstab`・`lsmr`に `B`を渡した場合も同じ スナップショットを使います。Rust 側の 適用・合成・反復法は Rustの中で 1 回の呼び出しを越えて
  作用素を使うため、呼び出し中しか有効でない借用ポインタを保持できません。以後の `S`への
  書き込みは 作用素に反映されません。
- `FaerSparseMatrix(transpose(B))`・`copy(B')`は借用 CSR から所有 CSRを直接作ります
  （CSCの配列は `Sᵀ`の CSR 配列そのものなので、転置の並べ替えなしに 1 回コピー）。
  `format=:csc`を指定すると CSCに変換します。

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

`faerview(transpose(S))`・`faerview(S')`は `transpose(faerview(S))`・
`adjoint(faerview(S))`を返します。借用 CSCの転置は同じ配列を CSRとして読む faerの
`SparseRowMatRef`（随伴は共役型）なので、密な行列・ビュー・Julia 配列との `*`と
`FaerMatrix`への `mul!`（左右両方）は転置や共役のコピーを作りません。

借用を値の書き込み先（faerの `SparseColMatMut`）にもできます。パターンは固定で、
`nonzeros(S)`は同じベクトルのまま書き換わります。

- `add_assign!(faerview(S), src)`・`sub_assign!`・`add_into!`・`sub_into!`:
  所有の出力先と同じ規約（入力は スナップショット なので別名も安全、出力先にない位置は
  書き込み前に `ArgumentError`）。
- `mul!(faerview(C), plan, A, B, α, β; scratch, team)`: `C`は 計画の結果
  パターンと完全一致が必要です。`C`と借用入力が格納値を共有する場合は
  `ArgumentError`です。
- `rmul!(B, α)`・`lmul!(α, B)`・`fill!(B, x)`・`B .*= α` などの ブロードキャストは
  親 `S`に転送します（Juliaの `SparseMatrixCSC`と同じ意味論）。

`u32` indexでの faer 呼び出しは意図的に対応しません。Juliaの indexは 1 始まり なので、
`Ti`の幅や符号（`Int32`・`UInt32` なども受け付けます）に関係なく呼び出しごとの
0 始まり バッファが必要です。所有疎行列・
非ゼロ配置・すべての疎 カーネルは `usize`に単相化しており、`u32`を加えると疎の全入口の
単相化が倍になる一方、利点は index バッファの縮小だけです。非圧縮形式（faerの
`col_nnz`）も未接続です。Juliaの `SparseMatrixCSC`は常に圧縮形式なので、非圧縮形式は
Julia 入力では生じません。

### 転置・随伴の疎行列積と疎×密の `mul!`

Rust 側の疎行列 `A`は、どちらの格納形式でも faerの列優先ビューを1つコピーなしで持ちます
（CSCはそのまま、CSRは転置として読みます）。`transpose(A)`・`A'`はこのビューの向きと
共役 フラグ（faerの `Conjugate` 型）に写像するため、積の入力として転置や共役のコピーを作りません。

- 疎×密: `op(A) * X`・`X * op(A)`（`op`は恒等・`transpose`・`adjoint`、`X`は
  `FaerMatrix`・ネイティブ ビュー・転置／随伴・`faerview` 借用）と、`FaerMatrix`への
  `mul!(C, op(A), X, α, β)`・`mul!(C, X, op(A), α, β)`は、faerの
  `sparse_dense_matmul`/`dense_sparse_matmul`で `C`に直接書き込みます。
  以前の `mul!`は積を新しく作り、`β*C`と足してから代入していました。
  `X`が `C`を参照する場合は、書き込み前に `X`を複製します。
- 疎×疎: `op(A) * op(B)`は faerの `sparse_sparse_matmul`に共役 フラグ 付きで渡します。
  両方が列優先として読める場合は、そのまま CSCの結果を返します。
  両方が行列優先の場合は `Cᵀ = op(B)ᵀ op(A)ᵀ`を計算し、CSRの結果を返します。
  どちらもコピーしません。向きが異なる組（CSC×CSR など）では、faerに CSR×CSCの
  カーネルがないため、右の入力だけを1回変換します。結果の形式は `op(A)`に従い、
  `A`（CSC）・`transpose(A)`／`A'`（CSR）は CSC、それ以外は CSRです。

この faerの版の疎×密 カーネルは `MemStack`を受け取らず、`Par` も使わずに逐次実行します。
そのため疎×密の `mul!`には callerの `scratch`／`team`を渡す引数がなく、Rust 側の
作業領域も必要ありません。並列化には faer 側の対応が必要です。

```julia
using Colmat, LinearAlgebra, SparseArrays

A = FaerSparseMatrix(sprand(6, 4, 0.5); format=:csr)
X = FaerMatrix(rand(6, 3))
C = FaerMatrix(rand(4, 3))
C0 = Matrix(C)
mul!(C, A', X, 2.0, 1.0)              # Aᴴ を共役ビューとして読み、C に直接書く
@assert Matrix(C) ≈ 2 * Matrix(A)' * Matrix(X) + C0
B = FaerSparseMatrix(sprand(3, 6, 0.5); format=:csr)
G = A' * B'                           # 両方を共役の列優先ビューとして読む（コピーなし）
@assert G.format === :csc && Matrix(G) ≈ Matrix(A)' * Matrix(B)'
```

### 疎×疎積の計画再利用 (`FaerSparseProductPlan`)

`A * B`は呼び出しごとに faerの symbolic 段（結果の疎構造と作業分割）と
numeric 段を実行し、結果の疎行列を新しく割り当てます。疎構造が固定で値だけが
変わる反復では、`FaerSparseProductPlan(A, B)`で
`sparse_sparse_matmul_symbolic`を1回だけ実行し、
`mul!(C, plan, A, B, α=1, β=0; scratch, team)`で
`sparse_sparse_matmul_numeric` だけを繰り返せます。

- 計画は `A`・`B`の格納パターン（格納ゼロを含む）の独立コピーを持ちます。
  毎回の `mul!`は、`A`・`B`のパターンと格納形式、および `C`のパターンが
  計画と完全に一致するかを割り当てなしで検査します。不一致なら `C`に
  書き込む前に `ArgumentError`を投げます。`C`のパターンが結果パターンより
  小さいと faerの作業列が汚れるため、`C`は結果パターンと完全一致が必要です。
- `C`には `FaerSparseMatrix{T}(plan)`（ゼロ値）か `A * B`を使います。
  `mul!`は `C`のパターン・nnz・`epoch`・`nonzeros(C)`の 変更を共有するビューを維持します。
  `β == 0`は上書き、それ以外は `C`を `β` 倍して加算します。
- 計画と `C`の形式は `A`に従います（CSRは `Cᵀ = Bᵀ Aᵀ`の列優先 計画）。
  `B`の形式が `A`と異なる場合は、計画の作成時に `B`の転置パターンと値の置換を
  1回だけ計算して保持します。毎回の `mul!`は `B`の値をこの置換で callerの
  `scratch` 上に並べ替えるだけで、形式変換の割り当ては行いません。
  `faerview(S)`の借用 Julia CSC 同士なら値も index 変換バッファ経由でコピーせずに
  読みます。借用と Rust 側の混在でもどちらもコピーせずに読みます（ネイティブ CSRは列優先の
  転置として読みます）。出力先 `C`には借用 `faerview(C)` も使えます。
- `eltype(C)`は `promote_type(eltype(A), eltype(B))`を保持できる型です。
  型の異なる入力は `eltype(C)`へ変換したコピー（同じパターン）を作ります。
  `C === A`／`C === B`は `ArgumentError`です。
- numeric 段の作業領域はスレッドごとに `eltype(C)`の密な1列です。`B`の形式が
  `A`と異なる 計画では、`B`の値を並べ替える `nnz(B)` 個分の領域が加わります。
  `scratch_requirements(mul!, C, plan, A, B; team)`、
  `FaerScratch(mul!, C, plan, A, B; team)`、
  `reserve!(work, mul!, C, plan, A, B; team)`で用意します。`scratch`を省略すると
  一時領域を割り当てて解放します。`FaerTeam`を渡すと faerの費用モデルに従って
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

同型の入力と callerの `scratch`を使う場合、Rust 側の numeric 呼び出しは
割り当てを行いません（Rustの allocation probeで検証）。Julia 側には FFI 呼び出し
1回分の固定 bookkeeping だけが残ります。400×400・密度 2%の測定では、
Julia ヒープ上の割り当ては従来の `mul!(C, A, B)`の 7712 バイト（これに加えて
ネイティブ 側で symbolic 構造・結果値・新しい 所有者を毎回割り当て）に対し、
計画 経由は 1648 バイトで ネイティブ 側の割り当てはありません。
借用 Julia CSCの 計画 経由では 0 始まり index バッファの検査・変換が加わります。

疎三角ラッパーの `\`と2引数・3引数の `ldiv!`は Rust 側の疎求解を使います。
通常・単位・下三角・上三角、転置／随伴、型の promotionに対応します。
単位三角では未格納の対角を 1として扱い、通常の三角行列で対角がゼロなら
`SingularException`を投げます。右辺が `FaerMatrix`／`FaerVector` なら結果も Rust 所有です。

```julia
using Colmat, LinearAlgebra, SparseArrays

A = FaerSparseMatrix(sparse([3.0 99; 2 4]); format=:csr)
b = FaerVector([3.0, 10.0])
x = LowerTriangular(A) \ b
@assert Vector(x) ≈ [1.0, 2.0]
ldiv!(UnitLowerTriangular(A), b)
@assert Vector(b) ≈ [3.0, 4.0]
```

`ldiv!(T, B)`は、`B`が `promote_type(eltype(T), eltype(B))`の要素型を持つ
`FaerMatrix`／`FaerVector`・ネイティブ ビュー・`faerview` 借用・重複要素のない Julia ストライドd
配列なら、faerの `sp_solve_{unit_}{lower,upper}_triangular_in_place`を `B`のメモリ上で
直接実行します。結果用の行列や右辺の複製は作りません。CSR 格納と `transpose`／`adjoint`
は列優先の カーネルとその転置版・共役 フラグに写像するため、疎行列も複製しません。
格納済みの疎行列は、すべての列が faerの前提を満たす場合にそのまま借用されます。
前提は、選択した半分だけが格納され、対角が下三角なら先頭、上三角なら末尾にあることです。
単位三角でも対角の格納が必要です。両半分を格納した行列や対角のない単位三角では、
三角部分を callerの `scratch` 上に組み直します。実数の疎行列と複素の右辺のように
型を昇格する場合は、昇格した値だけを `scratch`に置きます。
`scratch_requirements(ldiv!, T, B)`は、この組み直しと昇格に必要な byte 数を返します。
前提を満たす同型の場合は 0です。`FaerScratch(ldiv!, T, B)`と
`reserve!(work, ldiv!, T, B)` も使えます。callerの scratchを渡し、`Y === B`の場合、
Rust 側の求解は割り当てを行いません。Julia 側には nに依存しない数 KBの固定 bookkeeping
（lock・ビュー 記述子・必要量と容量の照会・FFI 引数）だけが残ります。`ldiv!(Y, T, B)`で `Y !== B` なら `B`を `Y`に
複製してから解きます。要素型の狭い出力や ストライドdでない出力では、昇格型の作業コピーを
1つ作り、書き戻すときに変換します。`team=`は検証しますが、faerの疎三角 カーネルは逐次実行です。
対角のゼロ、close 済みの行列・ビュー・scratch・team、容量不足はいずれも出力に書き込む前に検出します。

```julia
using Colmat, LinearAlgebra, SparseArrays

L = FaerSparseMatrix(sparse([2.0 0; 1 4]))       # 対角が先頭の下三角
B = FaerMatrix([2.0 4; 9 6])
@assert scratch_requirements(ldiv!, LowerTriangular(L), B).bytes == 0
work = FaerScratch(ldiv!, UpperTriangular(transpose(L)), B)
ldiv!(UpperTriangular(transpose(L)), B; scratch=work)   # Lᵀ x = B を B 上で解く
@assert Matrix(B) ≈ UpperTriangular([2.0 1; 0 4]) \ [2.0 4; 9 6]
```

疎行列の `lu`・`cholesky`・`qr`は再利用可能な `FaerSparseFactorization`を返します。

### 疎行列全体の ブロードキャスト・`map`

`FaerSparseMatrix`を含む ブロードキャストと `map`／`map!`は、SparseArraysと同じ結果規約に従います。
結果は疎行列で、`f`が生成したゼロは格納しません（入力の格納ゼロも同様です）。
結果の数値型が4型のいずれかなら `FaerSparseMatrix`を返し、CSC／CSR 形式は最初の疎入力に合わせます。

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

次の条件をすべて満たす場合は、Rust 側の高速経路で計算します。

- 疎入力が `FaerSparseMatrix` またはその `transpose`／`adjoint` だけで、
  すべて結果と同じ大きさ（ブロードキャストによる次元の拡張なし）
- ほかの引数が `Number`／`Ref`／0次元配列のスカラー
- `f(0, …, スカラー...)`がゼロ（`S .* Inf`は `0*Inf`が NaNになるため対象外）
- 結果の数値型が ネイティブ 型

この経路では、格納値を入力ごとに Juliaへ1回だけ転送し、Julia 側で `f`を適用します。

- 疎入力が1つ（同じ行列の複数回参照を含む）: 値を元の疎構造のコピーに付けて
  新しい 所有者を作ります。`transpose(S)`・`S'`は `S`の配列を他方の格納形式として
  読むため、結果の形式は `copy(S')`と同じく `S`と逆になります（随伴は値を共役にしてから `f`を適用）。
- 疎入力が複数: 転置／随伴の入力は ネイティブで1回コピーし、最初の入力と形式の異なる
  入力も1回変換します。全入力の疎構造が一致すれば、値を位置ごとにそのまま組み合わせます。
  一致しなければ、faerの `union_symbolic`で和集合の構造を作ります。各入力の値はその
  構造上に Rust 側で並べ（未格納の位置は 0）、Julia 側で `f`を適用します。
  400×400・密度 2%の `S .+ T`では、Julia ヒープの割り当てが約 600 KB から約 220 KBに、
  同じ構造の場合は約 870 KB から約 100 KBになりました。

`f`は Rust 側では実行しません。Juliaの関数を Rustの カーネル から呼ぶには `f` ごとの
`@cfunction`と GC・スレッドの制約が必要です。一方、値の転送は入力ごとに1回で済むため、
Rust 側での callback 実行には利点がありません。

出力先が入力自身の `S .= f.(S)`／`S .*= α`では値を一括で書き戻します。
このとき nnz・格納順・値ビューは維持されます。
ゼロが生じた場合は `dropzeros!` 相当の処理で構造を変え、以前の値ビューを失効させます。
出力の型への変換に失敗した場合（実数行列に複素数を書く等）は、書き込み前に例外を投げます。
400×400・密度 2%の `abs.(S)`では、Julia ヒープの割り当てが約 230 MB から約 33 KBになりました。

上記以外（`f(0)≠0`、次元の拡張、密配列・`SparseMatrixCSC`との混在）では、
各 ネイティブ 入力を `SparseMatrixCSC`に1回変換して SparseArraysの ブロードキャストを使います。
結果の数値型が ネイティブ なら `FaerSparseMatrix`に戻し、それ以外（`Bool`・`BigFloat` 等）は `SparseMatrixCSC`のまま返します。
`S .= x`のようなスカラー代入は Baseの規約どおり `fill!`になります。
`x == 0`では構造を保ち、非ゼロでは暗黙のゼロ位置を埋めることになるため `ArgumentError`です。
密な出力先への `D .+= S`は、疎入力を変換してから Juliaの ブロードキャストで計算します。
`map(f, A, B...)`は全引数が同じサイズである必要があり、異なると `DimensionMismatch`です。

## 構造解析と数値分解

`A \ b`は正方行列なら LU、縦長なら QRによる最小二乗求解を選びます。

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

`FaerSparseSymbolic`の `kind`は `:lu`・`:qr`・`:cholesky`です。
`factorize(plan, A)`と `lu!(F,A)`・`qr!(F,A)`・`cholesky!(F,A)`では、
格納ゼロを含むパターンが完全に一致する必要があります。CSC／CSRと数値型は
変更できますが、更新先の分解を渡す `!` 形式では数値型を維持します。
分解は元の行列と 計画に依存せず使え、`close`にも対応します。
混合精度の求解は保存済みの疎因子を用います。作業領域と型変換の詳細は、各分解の求解の節を参照してください。`ldiv!`は一時結果を経由して更新します。
LU・Choleskyは転置／随伴の求解にも対応します。
疎 QRは横長・ランク落ち・転置／随伴の求解にも対応します。
成分取得、並べ替え、最小ノルム求解の詳細は [疎 QR](sparse-qr.md)を参照してください。

## 並べ替え

fill-inを減らすための 並べ替えを単独で計算できます。
`amd`は正方行列の `A + transpose(A)`のパターン、`colamd`は長方形も含む
列パターンを使い、どちらも `FaerPermutation`を返します。

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

並べ替えは値に依存せず、格納ゼロもパターンに含めます。数値を持たない
`FaerSparsity`、通常の Julia CSC、転置／随伴にも対応します。
`Hermitian(A,:L)` などでは選択した三角部分のパターンを反映します。
AMDの `statistics=true`は置換と faerの演算数見積もりを返します。
`permute_rows`・`permute_cols`は疎数値行列と疎パターンに対応し、数値行列は
`permute_rows!`・`permute_cols!`でも更新できます。構造が変わる更新では以前の
`nonzeros` ビューが失効します。疎 LU／QRへの恒等・任意 並べ替えの直接指定にも対応します。
`amd` / `colamd`は逐次 Rust 側のカーネルの 呼び出し元が保持する作業領域に対応します。
queryは格納パターン（格納ゼロを含む）から必要量を計算し、数値には依存しません。
容量・アラインメント 不足や閉じた 作業領域は拒否し、自動拡張はしません。
CSCの複製／CSR 変換、Julia からのパターン構築、返却する置換は別途確保します。
構造解析 内の 並べ替え用の作業領域は `FaerSparseSymbolic(A; scratch=s)`に渡します（下記）。

```julia
using Colmat, SparseArrays
A = FaerSparseMatrix(sparse([4.0 0 1; 0 3 1; 1 1 5]); format=:csr)
work = FaerScratch(amd, A)
info = amd(A; scratch=work, statistics=true)
@assert isperm(Vector(info.permutation))
reserve!(work, colamd, A)
q = colamd(A; scratch=work)
@assert Vector(q) == Vector(colamd(A))
```

`scratch_requirements(FaerSparseSymbolic, A; kwargs...)`・`FaerScratch`・`reserve!`は
`FaerSparseSymbolic`と同じキーワード（`kind`・`uplo`・`ordering`・`algorithm`・
`dense`・`dense_row`・`dense_col`・`aggressive`）で、構造解析 内の 並べ替え
カーネル 用 作業領域を照会します。対象は次の経路です。

- Choleskyの `:default`／`:amd`: faerの 構造解析と同じ入力・設定の AMDを
  呼び出し元の作業領域で実行し、得た置換を faerに明示指定します。`algorithm=:auto`では
  simplicial／supernodalの選択を scratch なしの場合と一致させるため、因子の列数も
  同じ scratchで計算します。置換・方式・`factor_storage`は scratch なしと同一です。
  この列数計算の分だけ `:auto`は scratch なしより遅くなります（測定例で約 15–30%）。
  `:simplicial`／`:supernodal`を明示すると追加計算はありません。
- 横長 QRの既定 COLAMDと、明示 simplicial LU／QRの `ordering=:colamd`: COLAMD カーネルの
  必要量です。
- 縦長／正方の明示 simplicial QR（全 並べ替え）: faerの公開 カーネル（列消去木・postorder・
  列数・simplicial symbolic）と転置パターンの一時領域です。`:colamd`では COLAMDと同じ
  バッファを順に使い、必要量は大きい方です。
- 恒等・任意 並べ替えの Cholesky／LUと、正方／縦長の auto／supernodal LU／QRは 0 バイトです。
  後者の COLAMDは caller stackを受け取らない faerの symbolic API 内で確保されます。
  明示 並べ替えを渡しても faerの内部要求量に COLAMD 分が残るため、事前計算では減りません
  （[tensor4all/faer#322](https://github.com/tensor4all/faer/issues/322)、
  実行済みの再現例 `docs/reproducers/faer-symbolic-colamd-scratch.md`）。
  faerの symbolic QRが caller stack・`Par`を受け取らない点は 上流 未起票で、実行済みの再現例は
  [sparse-symbolic-qr-作業領域](https://github.com/AtelierArith/Colmat.jl/tree/main/docs/reproducers/sparse-symbolic-qr-workspace)
 にあります。

並べ替えカーネルは逐次実行です。容量・アラインメント 不足や閉じた 作業領域は拒否します。
CSCへの変換・複製、faerの 並べ替え 後の symbolic 用内部バッファ、保持する置換と
symbolic 構造は必要な出力として別途確保します。一括の `cholesky`／`lu`／`qr`の
`scratch`は数値分解用のままです。symbolic 側の 作業領域 も管理する場合は
`FaerSparseSymbolic`と `factorize(plan, A)`の二段階で呼び出します。

```julia
using Colmat, LinearAlgebra, SparseArrays
A = FaerSparseMatrix(sparse([4.0 0 1; 0 3 1; 1 1 5]))
s = FaerScratch(FaerSparseSymbolic, A; kind=:cholesky)
plan = FaerSparseSymbolic(A; kind=:cholesky, scratch=s)
@assert Vector(plan.p) == Vector(FaerSparseSymbolic(A; kind=:cholesky).p)
F = factorize(plan, A)
@assert F \ ones(3) ≈ [4.0 0 1; 0 3 1; 1 1 5] \ ones(3)
```

## 疎 Choleskyと分解成分

疎 Choleskyには 並べ替えと分解方式を直接指定できます。

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

`ordering=:default`／`:amd`は AMD、`:identity`は並べ替えなしです。
任意の置換も渡せます。`cholesky`では Julia 標準の疎行列 APIと同じ
`perm=p` も使えます。`perm`と `ordering`の同時指定はエラーです。
`dense`・`aggressive`は AMDの設定、`algorithm`は `:auto`・`:simplicial`・
`:supernodal`です。`F.p`／`plan.p`は独立した置換、`.algorithm`は実際の分解方式、
`.factor_storage`は確保する数値スロット数（supernodalの paddingを含む）を返します。
コピー・数値更新・混合精度の求解でも 構造解析の設定を引き継ぎます。

faerの supernodal 制御も指定できます（疎 Choleskyと、明示 simplicial 以外の疎 LU／QR）。
疎 QRでは横長入力の head 解析にも適用され、`plan.algorithm`／`F.algorithm`で
選ばれた方式を確認できます。

- `supernodal_threshold=r`: faerの `supernodal_flop_ratio_threshold`。`algorithm=:auto`
 でのみ有効な非負の実数で、既定は faerと同じ 1.0です。`0`はほぼ常に supernodal、
  `Inf`は常に simplicialを選びます。`FaerSparseSymbolic(A; scratch)`の AMD 経路でも
  同じ選択になります。
- `relax=[(n, z), ...]`: faerの `SymbolicSupernodalParams.relax`。隣接 supernodeの
  合計列数が `n` 以下で、併合後のゼロの割合が `z`（`0 ≤ z ≤ 1`）未満なら併合します。
  省略時は faerの既定値です。空の配列を渡すと併合しません。`algorithm=:simplicial`
 とは併用できません。
- `blocking_threshold=b`（疎 QRのみ）: supernodal frontの dense QRに使う faerの
  `QrParams.blocking_threshold`（非負整数、既定 48²）。計画に保持され、symbolic 再利用・
  `qr!`でも使われます。数値分解の scratch 量は変わりません。

```julia
using Colmat, LinearAlgebra, SparseArrays

A = FaerSparseMatrix(sparse([4.0 1 0; 1 4 1; 0 1 4]))
F = cholesky(Hermitian(A); supernodal_threshold=0)
@assert F.algorithm == :supernodal
G = cholesky(Hermitian(A); algorithm=:supernodal, relax=Tuple{Int,Float64}[])
@assert G.factor_storage <= F.factor_storage
```

疎 Choleskyの数値分解は、CSC 入力をコピーせずに借用します（CSR 入力は CSCへの変換が
1 回必要です。faerの疎 LU・QR・Choleskyの数値分解は `SparseColMatRef` しか受け取らず、
構造解析 も Aの CSCの非ゼロ配置に対して作るためです。CSRを Aᵀとして読むには別の
解析と転置した因子の管理が必要で、複素 Hermitianの Choleskyでは共役した値のコピーも
必要になるため、変換 1 回を意図的に残しています）。

faerの数値 `LltRegularization`（動的正則化）と `LltParams`は意図的に公開しません。
この pinの `SymbolicCholesky::factorize_numeric_llt`は、カーネルが返す正則化した pivotの
個数（`LltInfo`）を捨てるため、正則化で摂動した分解と厳密な分解を区別できず、
`check`・`PosDefException`・`issuccess`の規約（Juliaの CHOLMODと同じ）と両立しません。
正定値でない行列には `A + δI`のような shift か疎 LUを使ってください。`LltParams`
（supernode 内の密 ブロックの再帰閾値と ブロック 幅）は結果を変えない性能用の値で、他の密
カーネルと同様に faerの `Auto` 既定値を使います。

faerの 構造解析（`factorize_symbolic_{cholesky,lu,qr}`）は callerの stackを
受け取らず内部で割り当てます。`scratch`を渡すと Choleskyの AMDと wide QRの COLAMDは 呼び出し元の作業領域 上で
実行しますが、残りの内部割当は faerの API 上の制約です。入力が選択した三角部分だけを格納していれば、そのまま faerに渡します。
両方の三角部分を格納している場合は、選択した三角部分を faerの数値分解用 作業領域と
同じ stack 上に組み直します。`scratch`を渡したときの割当は返却する数値バッファ 1 つだけで、
`scratch_requirements(factorize, plan, A)`はこの組み直し分を含みます。
`F.L`・`F.U`は 行・列・値の組を経由せず、直接ソート済み CSCとして組み立てます。

LU／QRの 並べ替えと数値分解の `scratch`・`team`については、このページおよび [疎 QR](sparse-qr.md)を参照してください。

疎 Choleskyの `F.L`・`F.U`は独立した Rust 所有の CSC 行列です。
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

`FaerSparseMatrix(F)`・`sparse(F)`は疎行列を返します。再構成は分解と同じ精度で行い、
積によって追加の格納位置や丸め誤差が生じる場合があります。
`Matrix(F)`・`FaerMatrix(F)`は明示的に密行列を作ります。
`logdet`・`logabsdet`は Rustの分解バッファから対角だけを取得して計算し、
密行列化しません。失敗した分解の要素取得・再構成・行列式は `PosDefException`になります。

## 疎構造を保つ更新

既存の疎構造を保った加減算には `add_assign!`・`sub_assign!`・
`add_into!`・`sub_into!`を使えます。

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
不足すると更新前に `ArgumentError`を投げます。サイズ・型変換の失敗でも出力は変わりません。
入力は出力の数値型と CSC／CSR 形式にコピー・変換してから Rust 側のカーネルで計算し、
出力の型・構造・格納数を維持します。値ビューも失効しません。
通常の Julia CSCと Rust 側の転置／随伴を入力に使えます。
一般の ブロードキャスト（`D .+= S` 等）はこの専用 APIとは別の実装で、SparseArraysと同じくゼロを除去し構造を変えることがあります。

Rust 所有配列に対する `givens`は faerで係数を計算し、Julia 標準の

## 格納値ビューと変換

`copyto!`は以前の値ビューを失効させます。`FaerSparsity`は元の行列の変更・closeに
影響されません。`FaerSparseMatrix{T}(P)`は Boolean パターンを数値の 1に変換し、
`FaerSparseMatrix{T}(undef,P)`は同じ格納位置を 0で初期化します。
`FaerSparseMatrix(P, values)`で格納順の値をまとめて設定できます。

`rowvals`・`getcolptr`は CSCのみ対応し、返す添字配列は独立したコピーです。
CSRの行ポインタ・列添字には `compressed_storage`を使ってください。
疎分解の残りの機能・低水準制御は引き続き実装対象です。ブロードキャスト/`map`の結果規約は
「疎行列全体の ブロードキャスト・`map`」を参照してください。

`full_piv_lu`、`ldlt`、`bunchkaufman`、一般・自己随伴・一般化の
`eigen`／`eigvals`、三角行列の求解・逆行列、`det`／`logabsdet`／`logdet`、

## 疎 LUの成分と列の順序

疎 LUは正方行列を Lと Uの三角因子に分けます。`F.L`、`F.U`、行置換 `F.p`、列置換 `F.q`は独立したコピーです。`L*U ≈ A[p,q]`を満たします。すべての方式で、構造解析の再利用と数値更新に対応します。長方形の問題には疎 QRを使ってください。

`:auto`は方式を自動選択し、`:supernodal`は列群をまとめる方式です。`:simplicial`は列ごとの方式で、既定は列を並べ替えません。どの方式も `ordering=:identity`、`:colamd`、明示的な列置換を使えます。COLAMDは非ゼロ配置から計算量や新しい非ゼロの増加を減らす順序を探します。

`supernodal_threshold`は自動選択、`relax`は列群のまとめ方を調整します。これらは `:simplicial`には指定できません。CSCの数値入力は直接読み、CSRは CSCに変換します。因子の取り出しは独立した疎行列を確保するため、繰り返し取り出すより保存因子を直接使って求解すると効率的です。

```julia
using Colmat, LinearAlgebra, SparseArrays
A = FaerSparseMatrix(sparse([0.0 2 1; 3 1 0; 0 1 4]); format=:csr)
F = lu(A; algorithm=:simplicial, ordering=[3, 1, 2])
p, q = Vector(F.p), Vector(F.q)
@assert Matrix(F.L * F.U) ≈ Matrix(A)[p, q]
@assert Matrix(A) * (F \ ones(3)) ≈ ones(3)
```

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

## 条件数の推定

条件数は、入力の小さな誤差が解にどれだけ影響し得るかの目安です。大きいほど解が敏感になります。`cond(A, 1)`と `cond(A, Inf)`は疎 LUと少数回の求解で推定し、逆行列全体を作りません。推定なので厳密値とは限りません。

開始値は固定の種から作り、利用者の乱数生成器は変更しません。空の入力は 0、特異な行列は `Inf`です。疎行列の `cond(A)`（2 ノルム）は未対応で、必要ならメモリ量を確認して `cond(FaerMatrix(A), 2)`を使います。

```julia
using Colmat, LinearAlgebra, SparseArrays
A0 = spdiagm(0 => [4.0, 5, 6, 7], 1 => [1.0, 1, 1], -1 => [-1.0, -1, -1])
A = FaerSparseMatrix(A0)
@assert opnorm(A, 1) == opnorm(A0, 1)
κ = cond(A, 1)
@assert κ <= cond(Matrix(A0), 1) * (1 + sqrt(eps()))
@assert cond(A, Inf) > 0
```

## LU・QRの列順序を COLAMDで選ぶ

`ordering=:colamd`は格納構造だけを使って列の順序を選びます。格納されたゼロも構造に含み、数値そのものは参照しません。`dense_row`、`dense_col`、`aggressive`で探索を調整できます。生成した順序は計画へ保存され、数値更新でも保持します。

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
```

## 疎 LUを前処理に使う

`FaerLinearOperator(F)`は L/Uと行・列の置換を独立して保持し、疎な三角求解で逆作用を適用します。元の入力を再分解したり密行列にしたりしません。因子を更新・終了しても作用素は使えます。前処理の考え方は[線形作用素・反復法](operators.md)を参照してください。

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
```

## 疎 Choleskyを前処理に使う

疎 Choleskyの保存因子からも独立した逆作用を作れます。各方式、選択した三角領域、置換、転置・随伴、数値型の昇格に対応し、疎な格納を保ちます。元の入力を読み直しません。

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
```

## 疎 LU・Cholesky 求解の資源再利用

全方式で、左・右の求解、転置・随伴、ベクトル・行列・混合型の右辺に `scratch`と `team`を指定できます。照会と実行で数値型とチームをそろえ、変更時は `reserve!`で拡張します。

同じ型の求解は保存した数値因子を直接使います。混合型では保存因子を変換しますが、元の入力を再分解しません。出力・変換・書き戻し用のメモリは作業領域の外で確保します。疎 QRにも専用の[作業領域インターフェース](sparse-qr.md)があります。

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
```
