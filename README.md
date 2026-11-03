# Colmat.jl

RustCall.jl **v0.9.1** で動作確認した、t4a-faer を Julia から呼び出す PoC。
Tenferro.jl と同じく `#[julia]` と `@rust_crate` を使用します。

```sh
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.build("Colmat")'
julia --project examples/poc.jl
julia --project -e 'using Pkg; Pkg.test()'
```

`Pkg.test()` は ParallelTestRunner.jl でテストファイルを自動検出し、
独立したプロセスで並列実行します。並列数と対象一覧は次のように指定できます。

```sh
julia --project -e 'using Pkg; Pkg.test(; test_args=["--jobs=4"])'
julia --project -e 'using Pkg; Pkg.test(; test_args=["--list"])'
```

`julia --project deps/build.jl` でもビルドできます。Git とネットワーク接続、
Rust 1.89 以降が必要です（RustCall の Rust ツールチェーンも利用できます）。

`deps/build.jl` は https://github.com/tensor4all/faer のデフォルトブランチの
最新 HEAD を `deps/t4a-faer` にクローンし、再実行時も最新 HEAD に更新します。
ローカル変更がある場合は上書きせず停止します。使用したコミット SHA は
ビルド出力に表示します。最新版を追うため、将来の API 変更でビルドが失敗する
可能性があります。

`deps/rust` の橋渡し用 cdylib がこのクローンを Cargo の path dependency として
参照し、RustCall のキャッシュにコンパイルされます。`extern/t4a-faer` は不要です。
初回の `using Colmat` の前にビルドしてください。通常の読み込みでは Git の更新は
行わず、最後にビルドしたソースを使用します。

```julia
using Colmat, LinearAlgebra
A = FaerMatrix([4.0 1; 1 3])
B = FaerMatrix([1.0 2; 3 4])
C = A * B
x = A \ [1.0, 2.0]
F = svd(A)
@assert F.U * Diagonal(F.S) * F.Vt ≈ Matrix(A)
```

`FaerMatrix{T}` は Rust の `faer::Mat<T>` を所有する `AbstractMatrix{T}`
です。構築時に入力をコピーし、その後の行列同士の加減算・スカラー倍・行列積・
LU 求解は Rust 内のバッファを使います。これらの結果も独立した `FaerMatrix` になります。
`svd(A)`・`svdvals(A)` も Rust 側の行列を直接使い、分解結果を Julia 配列で返します。

```julia
using Colmat, LinearAlgebra

A = FaerMatrix([4.0 1; 1 3])
B = FaerMatrix([1.0 2; 3 4])
C = 2A + B - A / 2
D = A * B
X = A \ B
x = A \ [1.0, 2.0]         # ベクトルの右辺には Julia Vector を返す
@assert Matrix(X) ≈ Matrix(A) \ Matrix(B)

C[1, 2] = 5
mul!(C, A, B)              # C を更新。C === A / C === B も可
mul!(C, A, B, 2.0, 3.0)   # C = 2A*B + 3C
copyto!(C, A)
Z = zero(A)
E = copy(A)               # Rust の行列を複製
snapshot = Matrix(A)      # Julia へ独立したコピーを取得
```

Rust が所有する列ベクトル・行ベクトル・対角行列も使えます。

```julia
using Colmat, LinearAlgebra

A = FaerMatrix([4.0 1; 1 3])
v = FaerVector([1.0, 2.0])
w = A * v                       # Rust の Col を所有する FaerVector
x = lu(A) \ v                    # 結果も FaerVector
@assert Vector(w) ≈ Matrix(A) * Vector(v)
@assert Vector(x) ≈ Matrix(A) \ Vector(v)
@assert v' * v ≈ dot(Vector(v), Vector(v))
r = FaerRow([1.0, 2.0])          # 共役しない 1×2 行列
D = FaerDiagonal([2.0, 3.0])
filldiagonal!(D, 4)
@assert Matrix(D) == Matrix(Diagonal([4.0, 4.0]))

V = view(A, 2:-1:1, :)           # 逆順・間引きの範囲を扱える
@assert Matrix(V' * V) ≈ Matrix(V)' * Matrix(V)
copyto!(view(A, :, 1), v)         # 重複する領域からのコピーにも対応
fill!(diagonalview(D), 5)         # 対角の格納領域を借用
reserve!(A, 8, 8)                # 容量確保後も V は有効
@assert isopen(V)
close(V)                        # A 自体は閉じない
```

`view` は親の所有者を保持し、ネイティブ呼び出しの間だけ Rust の
`MatRef` / `MatMut` を作ります。部分行列・行・列・入れ子の範囲ビューに
対応します。親の形状を変える `resize!` / `truncate!` は既存のビューを
失効させます。親を閉じた場合もアクセスは例外になります。
`copy(view(...))` は独立した Rust 所有行列・ベクトルを返します。
ビューへの `copyto!` は入力を一時コピーし、代入先と重なる場合も値を保ちます。
`FaerDiagonal` の非対角要素は暗黙のゼロで、非ゼロの代入はできません。

置換も Rust に保持できます。添字は Julia と同じ 1-based です。

```julia
using Colmat, LinearAlgebra

A = FaerMatrix([1.0 2 3; 4 5 6])
p = FaerPermutation([2, 1])
q = FaerPermutation([3, 1, 2])
@assert Matrix(A[p, q]) == Matrix(A)[Vector(p), Vector(q)]
permute_rows!(A, p)
permute_cols!(A, q)
swap_cols!(A, 1, 3)
@assert Vector(p * invperm(p)) == [1, 2]

B = FaerMatrix([1.0 2; 2 4; 3 6])
P = pinv(B; rtol=1e-12)
@assert Matrix(B * P * B) ≈ Matrix(B)
@assert rank(B; rtol=1e-12) == 1
@assert cond(FaerMatrix([4.0 1; 1 3]), 1) ≈ cond([4.0 1; 1 3], 1)
```

`permute_rows` / `permute_cols` は独立した行列を返し、`!` 付きの関数は
入力を更新します。置換・交換は形状を変えないため、既存のビューも有効です。
`p * q` の添字列は `Vector(p)[Vector(q)]` です。`permute!(v,p)` /
`invpermute!(v,p)` で所有ベクトルにも適用できます。
`pinv` は faer の SVD と疑似逆再構成を使い、Hermitian・実 Symmetric では
自己随伴固有値分解を使います。`atol` / `rtol` は Julia と同じ判定式を使い、
省略時の `rtol` は `min(size(A)...)*eps` です。`cond` は 1・2・Inf ノルムに対応します。

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
`hessenberg` の結果は `FaerHessenberg` で、Julia の compact Householder 表現と
異なり `H`・`Q` の両方を明示的な行列として所有します。
`hessenberg!`・`schur!` は独立した作業結果を計算してから入力の所有バッファを
更新し、形状と既存ビューを保ちます。`schur!(H)` は `H.H`・`H.Q` の両方を
更新し、返す Schur 分解とストレージを共有します。並べ替えも成功後に更新する
ため、失敗時は元の分解を保ちます。これらは内部の一時領域を割り当てます。

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
一般化 Schur の `ordschur`、既定値以外のアルゴリズム設定、外部から渡す
作業領域は現在未対応です。

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

疎行列の `lu`・`cholesky`・`qr` は再利用可能な `FaerSparseFactorization` を返します。
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
混合精度の右辺では、内部の疎スナップショットを昇格し、symbolic analysis を
再利用して数値分解を作り直します。`ldiv!` は一時結果を経由して更新します。
LU・Cholesky は転置／随伴の求解にも対応します。
疎 QR は現在、列フルランクで行数が列数以上の行列が対象です。
横長・ランク落ちへの対応、QR の転置／随伴求解、LU／QR の分解要素の取り出し、
LU／QR への ordering 指定・追加の低水準制御は引き続き実装対象です。

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
`nonzeros` ビューが失効します。LU／QR への ordering の直接指定と外部 scratch は未対応です。

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
LU／QR への ordering の直接指定と、外部 scratch・追加の低水準制御は未対応です。

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
一般の `D .+= S` などの broadcast は、この専用 API と同一の実装ではありません。

native 配列に対する `givens` は faer で係数を計算し、Julia 標準の
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
独立した Jacobi オブジェクト・transpose/conjugate 専用 API は未対応です。

Householder 反射は `householder(x)` または `FaerHouseholder(x)` で生成できます。
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
単一反射の外部 scratch は未対応です。block／sequence の操作は次の所有型で扱います。

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
外部 scratch と `prev_block_size` を直接渡す低水準 upgrade は未対応です。

`nonzeros(S)` は格納値の変更に使えます。要素の挿入、`dropzeros!`、異なる構造の
`copyto!` は以前の値ビューを失効させます。`FaerSparsity` は元の行列の変更・close に
影響されません。`FaerSparseMatrix{T}(P)` は Boolean パターンを数値の 1 に変換し、
`FaerSparseMatrix{T}(undef,P)` は同じ格納位置を 0 で初期化します。
`FaerSparseMatrix(P, values)` で格納順の値をまとめて設定できます。

`rowvals`・`getcolptr` は CSC のみ対応し、返す添字配列は独立したコピーです。
CSR の行ポインタ・列添字には `compressed_storage` を使ってください。
疎分解の残りの機能・低水準制御は引き続き実装対象です。汎用 broadcast/map は
専用の疎 native 経路ではなく、Julia の汎用処理や密配列の結果になる場合があります。

`full_piv_lu`、`ldlt`、`bunchkaufman`、一般・自己随伴・一般化の
`eigen`／`eigvals`、三角行列の求解・逆行列、`det`／`logabsdet`／`logdet`、
`sum`／`norm`／`dot`／`tr`／`kron`、`pfaffian` も Rust に接続しています。
Float32・Float64・ComplexF32・ComplexF64 の密行列に対応します。
Issue #1 全体の進捗と残りの対象は
[実装台帳](docs/implementation-status.md) に記録しています。

```julia
using Colmat, LinearAlgebra

A = FaerMatrix(ComplexF32[1+2im 3; 4 5-im])
F = svd(A)
@assert eltype(A) === ComplexF32
@assert eltype(F.S) === Float32
@assert F.U * Diagonal(F.S) * F.Vt ≈ Matrix(A)
@assert Matrix(copy(A')) ≈ Matrix(A)'

B = FaerMatrix{Float32}(I, 2, 2)
reserve!(B, 8, 8)
resize!(B, 3, 3; fill=0)
truncate!(B, 2, 2)
@assert Matrix(B) == Matrix{Float32}(I, 2, 2)
```

同じ型の演算・分解は型を保持し、異なる型の行列同士の演算は
`promote_type` に従います。混合型の `A \ B` は係数行列を昇格してから
分解します。作成済みの `F = lu(A)` を再利用する場合は、元の分解精度が
残る点に注意してください。複素行列の内積は第1引数を共役化し、
`transpose(A)` と `A'` を区別します。

`size`、`axes`、インデックスによる読み書き、`similar`、`fill!`、`copy`、
`deepcopy`、ブロードキャストにも対応します。`A'` / `transpose(A)` は Julia の
遅延ラッパーで、`copy(A')` は Rust 側で転置行列を作ります。汎用の配列演算や
ブロードキャストは要素ごとに FFI を呼ぶ場合があります。

RustCall の所有ハンドルにより、GC 時に Rust の行列も解放されます。
`close(A)` は参照を外して以後のアクセスを禁止し、実際の解放は GC に委ねます。
複製した行列は元の行列を閉じた後も利用できます。Rust の可変参照が重複しないよう、
行列へのネイティブ呼び出しはロックで直列化しています。

- `faer_mul(A, B)`: 行列積。
- `faer_solve(A, B)`: 正方行列の partial-pivot LU による求解。右辺は行列またはベクトル。
- `svd(A::FaerMatrix; full=false)`: SVD。Julia 標準の `LinearAlgebra.SVD` を返します。`full=true` にも対応します。
- `svdvals(A::FaerMatrix)`: 特異値のみ計算します。
- `faer_svd(A)`: 互換 API。`FaerMatrix` では `svd(A)` と同じです。

通常の Julia 行列を受け取る上記の関数は、入力を Float64 の列優先バッファにコピーします。ビュー・転置も利用でき、
入力は変更しません。処理は `Par::Seq` で実行します。NaN/Inf、特異な LU、
SVD の失敗は例外になります。Julia バッファは `GC.@preserve` で保持し、Rust 側に
ポインタを残しません。この PoC はコピーと割り当てを行うため、性能測定用の
ゼロコピー・並列実装ではありません。互換 API に通常の実数配列を渡す経路は
Float64 に変換します。複素数を含め型を保持する場合は `FaerMatrix` を使ってください。
疎分解の残りの機能などの API は Issue #1 で管理しています。

RustCall v0.9.1 の制限として、受信オブジェクトを持つメソッドの `Result<Self, String>`
返却と、引数の `&OwnedMatrix` の変換に問題がありました。最小再現例と修正案を
[RustCall #564](https://github.com/AtelierArith/RustCall.jl/issues/564)・
[#565](https://github.com/AtelierArith/RustCall.jl/issues/565) に報告しています。
現在は新しい出力オブジェクトに結果を書き込み、入力の所有者を `GC.@preserve` で
保持した上でポインタを借用する方法を使用しています。

### Native linear operators and conjugate gradients

`FaerLinearOperator` owns a snapshot of a dense, CSC/CSR sparse, diagonal, or
identity operator. Sparse coefficients stay sparse during multiplication and CG.
A preconditioner applies an approximate inverse:

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

CG requires a Hermitian positive definite operator and preconditioner action.
`cg!` uses its destination as the initial guess. Failed checked solves throw
`FaerCgError`, whose `result` contains the last iterate and residual diagnostics;
`check=false` returns that result directly. Convergence includes an independently
recomputed true residual check. Julia function callbacks are not yet supported.

For a nonsymmetric square matrix, use `bicgstab` or `bicgstab!`. `M` and `N`
apply left and right approximate inverses, respectively:

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

`FaerBicgstabResult` reports the true and estimated residuals, iteration count
and termination reason. Checked failures throw `FaerBicgstabError` retaining
the result; `check=false` returns the last iterate. Multiple right-hand sides
use faer's block algorithm. Unlike CG, BiCGSTAB does not require Hermitian
positive definite coefficients.

For rectangular least squares, `lsmr` and `lsmr!` use the native block LSMR
algorithm without forming normal equations:

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

`M` supplies a nonsingular right preconditioner action. Convergence uses
`norm(M' * A' * (b-A*x)) ≤ max(atol, rtol * norm(M' * A' * b))`, taking `M=I`
when omitted. `FaerLsmrResult` separates ordinary residuals from these normal
residuals; an inconsistent least squares problem can converge with a nonzero
ordinary residual. Checked failures throw `FaerLsmrError` retaining the last
iterate. A nonzero initial guess can retain nullspace components, and right
preconditioning changes the minimum-norm metric.

### Standalone Jacobi rotations

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

### Native statistics

`Statistics.mean`, `var`, and `varm` use faer's row/column kernels for native
matrices, vectors, rows and views. Dimension reductions return Julia arrays;
full reductions return scalars:

```julia
using Colmat, Statistics
A = FaerMatrix([1.0 2; 3 4; 5 6])
@assert mean(A; dims=1) ≈ [3.0 4.0]
@assert var(A; dims=1) ≈ [4.0 4.0]
@assert varm(A, mean(A; dims=1); dims=1) ≈ var(A; dims=1)
A[1, 1] = NaN
@assert isnan(mean(A))
@assert mean(A; dims=1, nan=:ignore) ≈ [4.0 4.0]
```

The default `nan=:propagate` retains NaNs; `nan=:ignore` excludes entries with
any NaN component and adjusts each slice's count. `corrected=false` uses
population variance. Complex variances are real, empty slices return NaN, and
singleton sample variances follow Julia's NaN convention. Supplied means promote
the working scalar type. Calls snapshot native data to primitive buffers before
running faer's kernels; they leave the input unchanged.
