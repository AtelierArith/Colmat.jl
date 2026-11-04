# Colmat.jl

RustCall.jl **v0.9.2** で動作確認した、t4a-faer を Julia から呼び出す PoC。
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

dense LU・完全ピボット LU・QR（列ピボットを含む）・Cholesky・LDLT・
Bunch–Kaufman の構築と対応する `!` 版には `team=` を指定できます。
`factorize(A; team)` も対応します。構築後の因子は team や入力を閉じても独立に使えます。
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
QZ 構築の外部作業領域・既定値以外のアルゴリズム設定・team は未対応です。

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
`nonzeros` ビューが失効します。疎 LU／QR への恒等・任意 ordering の直接指定にも対応します。
ordering 自体の caller-owned scratch は未対応です。

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
scratch の外で割り当てます。構築・reblock の scratch/team は別の未対応項目です。

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

RustCall v0.9.2 で [#564](https://github.com/AtelierArith/RustCall.jl/issues/564)・
[#565](https://github.com/AtelierArith/RustCall.jl/issues/565) が修正されました。
`copy(::FaerMatrix)` は `duplicate(&self) -> Result<Self, String>` から独立した所有ハンドルを
直接受け取ります。行列・分解など型が決まっている所有者の引数には Rust の `&T` / `&mut T` を
使い、RustCall に型、生存状態、ライブラリ世代、可変借用の重複を検査させます。
型を実行時に切り替えるビューなどと省略可能な所有者の引数には、明示的な生ポインタ経路を残しています。
Rust 側の `rustcall_julia_macros` は 0.6 系が必要です。既存環境の更新時には
`Pkg.update("RustCall")` を実行してください。更新後の Rust クレートは次回ロード時に再ビルドされます。

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
recomputed true residual check. Iteration callbacks and residual histories are
available as described below; Julia-defined operator actions remain pending.

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

Absolute and relative LSMR tolerances are passed independently to faer. The
absolute bound can accept a nonzero initial normal residual with no updates:

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

### Balancing before eigenvalue computations

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

### Hermitian tridiagonalization and band eigensolvers

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

`FaerTridiagonalization` owns explicit native `Q` and a real `SymTridiagonal` H;
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

### Bidiagonal reduction and SVD

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

`bidiagonal_svd(d,e; uplo=:U)` supports real or complex diagonals and either
selected offdiagonal, returning native U/Vt and nonnegative descending real S.
`recursion_threshold` selects the native small/large SVD paths. Inputs remain
unchanged. The public native bidiagonal SVD computes dense factors/workspace,
including when `svdvals` requests only values.

### Native random matrices and unitary samples

`FaerRNG` owns a Rust random stream. Native array sampling uses faer's matrix,
column and row distributions directly:

```julia
using Colmat, Random, LinearAlgebra
rng = FaerRNG(42)
A = randn(rng, FaerMatrix{ComplexF64}, 3, 2)
snapshot = copy(rng)
v = rand(rng, FaerVector{Float64}, 4)
@assert Vector(v) == Vector(rand(snapshot, FaerVector{Float64}, 4))
Q = randunitary(rng, ComplexF64, 4)
@assert adjoint(Matrix(Q)) * Matrix(Q) ≈ Matrix{ComplexF64}(I, 4, 4)
Random.seed!(rng, 42)
@assert Matrix(randn(rng, FaerMatrix{ComplexF64}, 3, 2)) == Matrix(A)
row = rand(MersenneTwister(1), FaerRow{Float32}, (8,))
@assert size(row) == (1, 8)
randn!(rng, view(A, 1:2, 1:2))
```

`rand` gives independent uniform real/imaginary components in [0,1). `randn`
gives real standard normals; complex components each have variance 1/2, matching
Julia's unit expected squared modulus. `rand!`/`randn!` fill native matrices,
vectors, rows and views after staging the result. `randunitary(Float64,n)` gives
an orthogonal sample; complex types give unitary samples through faer's
sequential QR and phase correction.

Another Julia RNG can supply a UInt64 seed for each nonempty native sampling
call. Empty calls consume no seed. Use `FaerRNG` to keep a continuous native
stream; copy and reseed it independently, and close it when no longer needed.
Native sampling is reproducible for the same dependency versions and call
sequence, without promising Julia's random sequence. Scalar/ordinary Julia-array
draws from `FaerRNG` use Julia's sampling algorithms over its native random bits.

### NPY loading and read-only views

`loadnpy` reads NPY bytes, an IO or a file path into an independent native matrix.
This example uses a small fixture included with the package:

```julia
using Colmat
path = joinpath(pkgdir(Colmat), "test", "fixtures", "npy_c_f64.npy")
A = loadnpy(path)
@assert Matrix(A) == [1.0 2 3; 4 5 6]
V = npyview(path)
@assert npyinfo(V).shape == (2, 3)
@assert V[2, 3] == 6.0
B = copy(V) # independent, mutable FaerMatrix
close(V)
@assert Matrix(B) == Matrix(A)
```

Float32/Float64 and ComplexF32/ComplexF64, NPY versions 1–3, C/Fortran order and
both byte orders are supported. Scalars become 1×1 matrices and vectors become
n×1 matrices; `npyinfo` retains original shape/version/order metadata. NaN/Inf,
NaN payloads and signed zero are preserved. Higher ranks and other dtypes are
rejected, as are inconsistent lengths and trailing bytes. IO inputs are read
from their current position to EOF as one array.

`npyview` (or `loadnpy(...; view=true)`) owns an aligned, native-endian snapshot
of the bytes and uses faer's aligned-reference API for reads. It does not alias
the input bytes or memory-map a file. Entries are read-only. `copy` materializes
a native matrix; `deepcopy` preserves the view type with independent storage.
Closing a view invalidates its reads, while materialized/deep copies remain valid.

### Partial eigenvalues and singular values

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
self-adjoint and SVD requests use native dense decompositions to avoid upstream
panics, with `algorithm == :dense_fallback`; iterative results report `:partial`.

### Structured products

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
```

Lower/upper and unit-triangular wrappers use faer's structured multiplication;
`FaerDiagonal` and Julia `Diagonal` with native matrices use native diagonal
products. These methods return a `FaerMatrix`, including triangular×triangular
products. Native destination `mul!` supports coefficients and aliasing. Operand
snapshots and a temporary result are allocated; even diagonal operands are
currently snapshotted through a dense representation. Unused triangles and
implicit unit diagonals are ignored, including nonfinite values stored there.

### Pivoted Cholesky

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

Pivoted factors use independently owned native storage. `F.L`, `F.U`, `F.p`,
`F.P`, `rank(F)` and reconstruction follow `CholeskyPivoted` conventions.
Negative `tol` selects faer's automatic threshold; nonnegative `tol` selects
an absolute cutoff in the Rust adapter's complete diagonal-pivot method because
faer's factor kernel does not expose a tolerance parameter. The Julia default
is `tol=0`. Reconstruction, full-rank solves and inverses use faer's pivoted LLT
kernels. Rank-deficient factors require `check=false`; their solves/inverses
throw `RankDeficientException`. Native snapshots and workspace are allocated.
`close(F)` closes its native factors; a copied factorization remains usable.

### Destructive factorization entry points

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

`lu!`, `qr!` (including `ColumnNorm()`) and `cholesky!` (including `RowMaximum()`)
compute independent native factors and then overwrite the input after the result
is available. Existing same-shape views observe the packed entries. The returned
factorization remains independent if the input changes or closes. These methods
allocate factors and staging buffers; errors before writeback preserve the input.
Failed ordinary `cholesky!(...; check=false)` returns an unsuccessful factor
without changing the input.

LU packed storage combines U with the strict lower part of L. QR combines R
with the strict lower Householder basis; its block coefficients are exposed as
`F.Qcoeff` and its basis as `F.Qbasis`. This is faer's block format, rather than
LAPACK's tau layout. `F.packed` returns an independent copy, with lower storage
for Cholesky. Upper-triangle Cholesky wrappers receive U in the input instead.

### Sparse LU components

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

### Caller-owned parallel teams

```julia
using Colmat, LinearAlgebra
FaerTeam(2) do team
    A = FaerMatrix([1.0 2; 3 4])
    B = team_mul(team, A, A)
    @assert Matrix(B) == [7.0 10; 15 22]
    mul!(A, team, A, A, 2.0, 0.0)
    @assert Matrix(A) == 2Matrix(B)
end # releases the reservation, including when the body throws
```

`FaerTeam(width)` retains a reusable native pool/team. Width includes the caller
and `width-1` reserved workers; width one is sequential. Each multiplication
borrows a `Par::Team` only during the native call. Four scalar types, promotion,
views, transposes, coefficients and aliased destinations are supported. Copies
reserve independent teams. Close teams promptly; GC also owns their handles.

`team_mul`, `mul!(dest, team, A, B[, alpha, beta])`, `pfaffian(A; team=team)`
and SVD/EVD entry points with `team=` use this explicit team. Other routines retain sequential execution. Calls are serialized
by Colmat's owner lock, and faer may choose a narrower region for small work.
The API controls worker reservations; it does not promise a speedup for every
matrix size. Inputs needing conversion and destination writes use temporaries.

### Reusable native scratch

```julia
using Colmat
A = FaerMatrix([0.0 2; -2 0])
requirements = scratch_requirements(pfaffian, A)
workspace = FaerScratch(pfaffian, A)
@assert scratch_capacity(workspace) == requirements
@assert pfaffian(A; scratch=workspace) == 2.0
FaerTeam(2) do team
    @assert pfaffian(A; scratch=workspace, team=team) == 2.0
end
reserve!(workspace, requirements.bytes; alignment=requirements.alignment)
close(workspace)
```

`FaerScratch` owns an aligned native MemBuffer. Capacity/alignment are explicit,
and insufficient workspace raises an error before kernel execution. Grow it
using `reserve!(workspace, pfaffian, A)` for a new operation size, or supply byte
capacity/alignment directly. Calls never grow the workspace implicitly; copy
and deepcopy allocate independent storage. Contents are scratch, not user data.
Pfaffian validates shape, finiteness and skew-symmetry and leaves A unchanged.
`team=` works with supplied scratch or with a temporary workspace released after
the call. Pfaffian, SVD and single/generalized EVD are connected consumers; other kernels still
allocate their own workspaces. Julia/FFI bookkeeping may still allocate even
when the native workspace is reused.

### Destructive indefinite and full-pivot LU decompositions

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

`ldlt!`, `bunchkaufman!` and `full_piv_lu!` stage independent native factors
before writing packed entries into the input. Same-shape views see those
entries; returned factors remain independent of later changes or close.
LDLT/block LDLT store D's diagonal plus strict-lower L (or its adjoint for an
upper wrapper). Block LDLT keeps its block off-diagonals in the returned `F.D`
and its permutation in `F.p`; retain those to reconstruct/solve. This is faer's
split block representation. `F.packed` is always lower-oriented. Failed
`ldlt!(...; check=false)` leaves input unchanged because no factor is available;
allowed singular full-pivot LU/block LDLT write the available packed factors.
These entry points allocate factors and writeback buffers.

### SVD workspace, teams and destructive entry points

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

`svd`/`svdvals` accept `scratch=` and `team=` for low-level native SVD; existing
calls without these keywords retain their prior route. `scratch_requirements`
and `FaerScratch` accept `svd`, `svdvals`, `svd!` and `svdvals!`; specify matching
`full` and `team` options. The query uses the actual team's execution conditions.
Singular values are real for all four input types. Thin/full and empty factors
follow Julia's SVD conventions. Factor output arrays are Julia allocations;
temporary diagonal values and algorithm workspace use the supplied native buffer.

`svd!` and `svdvals!` replace input by rectangular Sigma only after success.
Same-shape views observe Sigma; returned factors remain independent. This is an
explicit writeback convention, not a LAPACK work matrix. Insufficient scratch,
invalid input or native failure leaves input unchanged. Source/factor bookkeeping
and writeback allocate; neither entry point promises whole-call zero allocation.
Temporary scratch created for a call is closed in a finally block.

### Destructive eigenvalue decompositions

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

### Self-adjoint EVD with reusable workspace and teams

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

### General EVD with caller-owned scratch and teams

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

### Definite self-adjoint generalized EVD workspace

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

### Generalized EVD for general matrix pairs

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

For repeated finite eigenvalues, upstream can return nonfinite vectors
([faer #217](https://github.com/tensor4all/faer/issues/217)). Colmat recovers
invalid columns using native SVD nullspaces with the requested team, checking sufficient
nullity. This exceptional path allocates additional matrices and internal SVD
workspace outside the supplied buffer. Recovery also supports repeated infinite eigenvalues. Defective eigenspaces
with insufficient nullity and invalid indeterminate vectors are rejected. The dependency checkout is left unchanged.

### Homogeneous generalized eigenvalues

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
The homogeneous residual above stays meaningful for infinite eigenvalues. Vector
recovery uses normalized pairs and native SVD, propagating the requested team,
and may allocate additional workspace. This option is for general pairs;
self-adjoint wrappers and destructive eigen routines keep their current APIs.

### Left generalized eigenvectors

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
works for either. If the direct left driver produces nonfinite outputs
([faer #219](https://github.com/tensor4all/faer/issues/219)), Colmat recomputes the
equivalent adjoint pencil with the native right driver. This reuses the selected
scratch/team and allocates additional input/output copies. Invalid repeated left
vectors can also use adjoint homogeneous nullspaces, including infinite values. Simultaneous output is available with `side=:both` below. Side selection for
single matrices is described below; destructive routines retain their current API.

### Simultaneous left and right generalized eigenvectors

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

### Left and simultaneous eigenvectors of one matrix

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

### Reconstructing native factorizations

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
close(pair.A)
close(pair.B)
```

`reconstruct(F)` returns independent native storage. Dense owned LU/QR/
Cholesky/LDLT, Schur/Hessenberg and tridiagonal/bidiagonal reductions return
`FaerMatrix`; sparse Cholesky and every sparse LU algorithm return `FaerSparseMatrix`
in original row/column order. Generalized Schur/Hessenberg return `(A,B)` named
fields, reconstructing both members through Q/S-or-H/T/Z. The sparse LU path
uses current native factors and permutations, including numeric updates; it
never copies the retained input snapshot. Its default output is CSC. All sparse QR
algorithms reconstruct a dense native matrix from their saved Q/R factors.

`Matrix{T}(F)` and `FaerMatrix{T}(F)` also support the dense owned decompositions
above; sparse LU adds these conversions and typed sparse CSC output.
Numeric conversion follows Julia rules, including errors for nonrepresentable
complex-to-real values. Conversion temporaries are closed on errors. Reconstructed
owners remain valid after the original factors are updated or closed.

### Retained dense solves with reusable scratch and teams

`ldiv!(X,F,B; scratch,team)` and `rdiv!(X,B,F; scratch,team)` borrow native resources for
retained LU/full LU/QR/pivoted QR/Cholesky/LDLT/Bunch–Kaufman solves, including
transpose/adjoint, rectangular LU/full LU/QR and rank-deficient QR minimum-norm paths.
Two-argument forms require square factors. Solve teams are independent of the
team used to construct the factor; the factor remains usable after either team
closes. Native solve work, including the QR fallback R SVD and projections,
uses caller scratch; the result and Julia staging allocate separately. Mixed
native scalar types can reconstruct/refactor a factor outside this solve query.
QR rank cutoffs retain the source precision after promotion so reconstruction
roundoff does not become new solution directions. Component extraction
remains sequential; promoted matrix products and factorization borrow the team.

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
close(F)
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
close(work)
close(team)
close(F)
```

`scratch_requirements(ldiv!,X,F,B;team)` and the corresponding right-solve
argument order describe the working scalar and actual team. Two-argument
`ldiv!(F,B)` / `rdiv!(B,F)` have matching queries for square factors. Capacity,
alignment, closed-resource, conversion and alias failures preserve destinations.
Wide/rank-deficient QR uses the saved Householder/R and column permutation;
the R SVD is computed inside scratch on each fallback solve. Rectangular LU/full
LU reconstructs from saved L/U and permutations and computes its SVD inside
the same workspace; the original input snapshot is not used. `FaerLinearOperator(F;rtol=...)` offers a separately configurable cached
QR R SVD action for repeated applications. Float64 factors split
complex RHS into two real solves that reuse one workspace. Factor construction
and its mixed-scalar reconstruction/refactorization still allocate independently.

### Reusing Rust-owned spectral factors

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
close(E)
close(U)
close(V)
close(P)
```

`native=true` retains actual faer `Svd` / `SelfAdjointEigen` objects in a
`FaerSpectralFactor`, across four scalar types. Ordinary calls keep their current
Julia result types. Native SVD supports thin/full and rectangular/empty shapes;
native self-adjoint EVD honors each selected triangle and `sortby`. Real
`Symmetric` wrappers are supported too. Construction accepts `team=` and uses
backend-owned workspace; caller `scratch=` is not supported for construction.

`pinv(F; atol, rtol, team)` reuses stored factors through faer's pseudoinverse
reconstruction kernel. `reconstruct(F; team)`, `Matrix(F)`, typed conversions,
`rank`, `svdvals` or self-adjoint `eigvals/eigvecs` also reuse the factorization.
`copy/deepcopy` create independent native owners and `close` releases the retained
factors immediately. Components are independent copies: U/V/vectors are native
matrices, S/values are real Julia vectors. Eigen sorting applies a common
permutation to values/vectors. Extracted components survive closing F; modifying
them does not update F. General EVD native ownership, native bang construction
and caller-controlled spectral-owner construction workspace remain unexposed.

Saved factors also support `F \ B`, `B / F`, `ldiv!`, `rdiv!`, and square
`inv(F)`, including transpose/adjoint and mixed scalar types. Solves project the
RHS through the retained native factors, applying a truncated minimum-norm
solution without building a full pseudoinverse. `atol`, `rtol`, `scratch`, and `team` are
accepted; cutoff precision follows the stored factor even when the RHS promotes
the computation. Signed self-adjoint eigenvalues are preserved. Three-argument
mutation supports rectangular factors; two-argument mutation requires square
factors. Destination conversion is staged before writeback. Caller scratch holds
projection, weights and promoted U/V copies; native results and Julia/RHS
conversions allocate separately. Right solves avoid an owned adjoint RHS copy.
Singular inverses throw
`SingularException`.

### Reusable workspace for saved SVD/EVD solves

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

### Reusable workspace for saved-factor pseudoinverses

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
close(F)
```

`pinv(F; scratch=workspace, atol, rtol, team)` consumes caller-owned aligned
storage for retained native SVD and self-adjoint EVD factors. The query reads
actual factor metadata and team conditions. `FaerScratch(pinv,F)` and
`reserve!(workspace,pinv,F)` use the same requirements as execution; insufficient
capacity/alignment is rejected and the workspace never grows implicitly.
Repeated calls reuse the allocated buffer and do not repeat SVD/EVD. Thresholds
apply to the stored singular/eigenvalues. The independent result matrix allocates
outside scratch. Empty factors need no kernel workspace but still require an
open buffer. This interface controls saved-factor pseudoinverse workspace;
native factor construction and direct `pinv(A)` keep their existing interfaces.

### Sparse QR components

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
close(Q)
close(R)
close(p)
close(A)
```

Every sparse QR algorithm retains faer's native Householder/R factors and
symbolic analysis. `F.Q` is an independent thin `FaerMatrix` with dimensions
`(m,n)`; `F.R` is an independent sparse CSC `(n,n)` factor. `F.pcol` (also `F.p`)
is the column permutation satisfying `Q*R ≈ A[:,pcol]`. Extraction, reconstruction,
copy and numeric updates use retained factors. `reconstruct(F)`, `Matrix(F)` and
`FaerMatrix(F)` recover the original column order without using the stored input
snapshot. Four scalar types and CSC/CSR inputs are supported. For explicit
simplicial QR, ordering is identity by default; `ordering=:colamd` or an explicit permutation selects another column
order. COLAMD accepts `dense_row`, `dense_col` and `aggressive` controls and is
retained across symbolic reuse and numeric updates. Auto/supernodal QR uses
COLAMD ordering and exposes the same thin Q, sparse R and reconstruction.
Full Q and rank-deficient minimum-norm solves are described below. AMD ordering
remains unavailable. Caller scratch
is supported for auto/supernodal normal solves and numeric construction as
described below; component extraction and transpose solves allocate workspace.

### Minimum-norm solves with sparse QR transpose/adjoint

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
close(X)
close(Y)
close(F)
close(A)
```

For every full-column-rank sparse QR algorithm, `transpose(F) \ B` and
`adjoint(F) \ B` return the minimum-norm solution of the corresponding
underdetermined system. They reuse the native R/Householder factors and column
permutation, without constructing Q or a pseudoinverse. Three-argument `ldiv!`
supports rectangular factors; two-argument `ldiv!` requires square factors.
Native work arrays and sparse R extraction/conversion allocate per solve.
Rank-deficient factors use the SVD minimum-norm path described below. Mixed-type RHS reuse saved native factors for every sparse LU/QR/Cholesky algorithm. Factor values are promoted without
refactorizing the input, so accuracy remains limited by the original factor
precision. Mixed-type QR transpose/adjoint solves also reuse the saved reflectors and R.

```julia
using Colmat, LinearAlgebra, SparseArrays
A0 = Float32[4 1; 1 3]
A = FaerSparseMatrix(sparse(A0))
F = lu(A; algorithm=:simplicial)
B = ComplexF64[1 + im, 2 - im]
x = F \ B
@assert eltype(x) == ComplexF64
@assert A0 * x ≈ B rtol=1e-6
close(F)
close(A)
```


### Auto/supernodal sparse QR components and minimum-norm solves

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
close(Q)
close(R)
close(p)
close(X)
```

The native transpose solve extracts sparse R and applies saved Householder
blocks in reverse order. It does not form a dense Q or pseudoinverse. Extracted
components own their storage and remain valid after numeric updates or close.

### COLAMD ordering in sparse LU/QR

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


### Explicit supernodal sparse QR

`qr(A; algorithm=:supernodal)` and
`FaerSparseSymbolic(A; kind=:qr, algorithm=:supernodal)` force faer's native
supernodal QR. Both retain public low-level symbolic analysis and adapter-owned
numeric buffers. Normal least-squares solves, independent copies, symbolic
reuse and numeric `qr!` updates support all four scalar types and CSC/CSR.
This path uses native COLAMD ordering, with optional `dense_row`, `dense_col`
and `aggressive` controls. `F.p/F.pcol` returns an independent column permutation.
Q/R component getters and transpose/adjoint solves are not yet exposed for
supernodal QR. Solve workspace is available through the interface below. The explicit simplicial path retains its existing
component and transpose/adjoint interfaces.


### Reuse workspace for retained sparse QR solves

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
close(s)
close(F)
```

This interface accepts matrix RHS matching the factor's scalar type for normal
auto/supernodal QR solves. `reserve!(s,ldiv!,F,B)` explicitly grows an existing
buffer. Insufficient capacity/alignment is rejected before writing the result;
the buffer never grows implicitly. Native RHS/result storage and destination
copying allocate outside scratch. Factor construction has a separate workspace interface below. Simplicial QR and
transpose/adjoint solves retain their existing workspace paths.


For auto/supernodal QR, `ordering=:default` and `ordering=:colamd` both select
native COLAMD. For example, `qr(A; algorithm=:supernodal, ordering=:colamd,
dense_row=-1.0, dense_col=0.0, aggressive=false)` retains the resulting order
through copies and numeric updates. Identity/custom ordering for this path
is supported for every QR algorithm.

### Reuse workspace for sparse QR factor construction

```julia
using Colmat, LinearAlgebra, SparseArrays
A = FaerSparseMatrix(sparse([2.0 0; 0 3; 1 0]))
p = FaerSparseSymbolic(A; kind=:qr, algorithm=:supernodal)
s = FaerScratch(factorize, p, A)
F = factorize(p, A; scratch=s)
G = factorize(p, A; scratch=s)
@assert F \ ones(3) ≈ Matrix(A) \ ones(3)
close(p)
close(s)
@assert G \ ones(3) ≈ F \ ones(3)
close(F)
close(G)
close(A)
```

`factorize(plan,A;scratch=s)` supplies caller-owned workspace to auto/supernodal
QR numeric construction. `scratch_requirements(factorize,plan,A)` queries the
scalar-dependent requirement and `reserve!(s,factorize,plan,A)` grows capacity
explicitly. The numeric matrix must match the plan's stored pattern exactly.
Symbolic analysis, sparse snapshots and owned factor buffers allocate outside
scratch. Returned factors do not borrow the workspace and remain valid after
it or the input plan is closed. Explicit simplicial factor construction retains
its previous workspace path.

### Reuse operator application scratch and teams

`mul!(X, op, B[, alpha, beta]; scratch, team)` and
`*(op, B; scratch, team)` accept caller-owned resources for native LinOp
application, including conjugate/transpose/adjoint and right-operator products.
Query the exact working scalar and orientation with `scratch_requirements`;
alpha/beta and output type participate in a `mul!` query.

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
close(op)
```

Kernel scratch covers the native LinOp query, including composition and saved
factor inverse-action intermediates where implemented. Result staging, Julia
scaling/conversion buffers and promoted operator snapshots allocate separately.
Prepared auto/supernodal and wide sparse-QR inverse actions use this buffer and
propagate the team to their kernels. Wide inverse construction caches the dense
pseudoinverse of R once; construction and direct factor solves have separate
allocation costs. Invalid/closed resources and insufficient
capacity/alignment reject before destination writeback. Shared use is serialized
by the owner lock.

### Compose native matrix-free operators

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
close(N)
```

Operator multiplication retains independent copies of both operands and applies
them in sequence without materializing their coefficient product. Composition
supports nested products, conjugation, transpose and adjoint; intermediate RHS
storage comes from native scratch. Explicit `Matrix(op)` materializes the result.
A composition has no single `.coefficients` representation. Mixed native scalar types are promoted recursively through the operator tree.
Dense, CSC/CSR, diagonal and identity leaves preserve their representations;
composition never forms a dense coefficient product during promotion.

### Use retained LU factors as a native preconditioner

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
close(M)
close(A)
```

`FaerLinearOperator(F)` for a square nonsingular native partial-pivot LU owns
copies of L/U and the pivot permutation. It applies the inverse action using
native triangular solves and scratch, without forming an inverse or repeating
factorization. Normal/conjugate/transpose/adjoint application, independent copies,
composition and mixed native scalar promotion reuse those saved factors. There
is no single `.coefficients` matrix. CG requires a positive-definite inverse
action; general LU actions can be used with BiCGSTAB. Other decomposition and
sparse factor preconditioners remain pending.

### Use retained Cholesky factors as a preconditioner

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
close(M)
close(A)
```

A native Cholesky preconditioner owns independent triangular factors and applies
them with lower/upper triangular solves. It supports the selected input triangle,
all four native scalar types, mixed RHS promotion, copy/composition and
transpose/adjoint without inverse construction or refactorization. Failed
Cholesky factors are rejected with `PosDefException`. Sparse and other dense
factor preconditioners remain pending.

### Retained full-pivot LU inverse actions

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
close(M)
close(N)
```

Full-pivot LU inverse actions retain both row and column permutations with their
triangular factors. Normal, transpose and adjoint application use native
full-pivot solves and the corresponding scratch requirement. Mixed scalar
promotion copies factor values and preserves both permutations, without
refactorization. Singular and rectangular factors are rejected.

### Retained QR inverse-action preconditioners

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
close(M)
```

Square native QR and column-pivoted QR factors can provide inverse actions.
Their compact Householder basis/coefficients, R and column permutation are copied
into an independent native owner. No explicit Q, inverse or refactorization is
needed. Native QR solve kernels consume operator scratch, and mixed scalar
promotion converts the saved components. Empty square factors are supported.
Without a nonzero atol or explicit rtol, square QR keeps this nonsingular
inverse contract and rejects zero R pivots.

Rectangular QR instead provides a minimum-norm action of shape `reverse(size(F))`,
including rank-deficient inputs. `atol`/`rtol` select singular values of saved R
using `max(atol,rtol*maximum(svdvals(R)))`. The default rtol is
`min(size(F)...)*eps(real(eltype(F)))` when atol is zero, and zero otherwise.
Nonzero atol or explicit rtol also select this truncated action for square QR.
Construction computes an SVD of R once and stores its singular factors, compact
Householder factors and column permutation. Applications form neither dense Q
nor a pseudoinverse of A and reuse native scratch/team. Promotion converts the
cache without recomputing rank; construction and result staging allocate.

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
# Explicit rtol enables a truncated action for singular square QR too.
F=qr(FaerMatrix([1.0 0;0 0]))
M=FaerLinearOperator(F;rtol=1e-12)
@assert Matrix(M)≈[1.0 0;0 0]
close(M)
close(F)
```

### Retained LDLT preconditioners for indefinite systems

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
close(M)
close(A)
```

The native LDLT inverse action owns L and D and applies the low-level saved-factor
solve without an inverse or refactorization. Both selected input triangles,
normal/conjugate/transpose/adjoint, copy/composition and mixed native scalar
promotion are supported. Zero diagonal pivots are rejected. Indefinite inverse
actions are suitable for general solvers such as BiCGSTAB; they do not satisfy
CG's positive-definiteness requirement. Sparse factor preconditioners remain pending.

### Retained Bunch–Kaufman preconditioners

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
close(M)
```

Block-pivoted LDLT inverse actions retain L, the block diagonal's diagonal and
subdiagonal, and the pivot permutation. Native Bunch–Kaufman solves consume
operator scratch; valid 2×2 blocks with zero diagonal entries are preserved.
Mixed promotion copies both block components and the permutation, without an
inverse or refactorization. These indefinite actions can be used with BiCGSTAB;
CG's positive-definiteness requirement still applies. Sparse factor
preconditioners remain pending.

### Matrix-free saved SVD/EVD pseudoinverse actions

```julia
using Colmat, LinearAlgebra
A0 = [3.0 0; 0 1e-6; 0 0]
F = svd(FaerMatrix(A0); native=true)
M = FaerLinearOperator(F; rtol=1e-3)
close(F)
@assert size(M) == (2, 3)
@assert M * ones(3) ≈ [1/3, 0]
@assert Matrix(M) ≈ pinv(A0; rtol=1e-3)
close(M)
```

`FaerLinearOperator(F;atol,rtol)` for a saved native SVD or self-adjoint EVD
owns spectral components and applies a truncated inverse action using native
projection/scaling kernels and scratch. The operator shape is `reverse(size(F))`;
rectangular and rank-truncated actions are supported. Thresholds follow `pinv(F)`
and are chosen in the original factor precision before any RHS promotion.
Negative eigenvalues retain their sign. Copy/composition, mixed native scalar
promotion, conjugation and transpose/adjoint do not recompute the decomposition
or materialize a pseudoinverse. Iterative preconditioning requires a suitable
square nonsingular action; CG additionally requires positive definiteness.

### Native preconditioning from sparse LU

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

### Native preconditioning from sparse Cholesky

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

### Native inverse actions from sparse QR

```julia
using Colmat, LinearAlgebra, SparseArrays
A0 = ComplexF64[2 0; 1im 3; 0 1]
F = qr(FaerSparseMatrix(sparse(A0)); algorithm=:simplicial, ordering=[2, 1])
M = FaerLinearOperator(F)
close(F)
@assert size(M) == (2, 3)
@assert M * ones(ComplexF64, 3) ≈ A0 \ ones(ComplexF64, 3)
@assert adjoint(M) * ones(ComplexF64, 2) ≈ pinv(A0') * ones(ComplexF64, 2)
close(M)
```

The action retains sparse Householder components, R and column permutation;
no dense Q or pseudoinverse is formed. Full-column-rank tall and square factors
produce an action of shape `reverse(size(F))`. Transpose/adjoint apply minimum-norm
inverse actions using saved reflectors and sparse triangular solves. All native
scalar promotions preserve Householder pivot ordering. Explicit simplicial and
auto/supernodal action intermediates use native scratch; caller scratch and team
are supported for all orientations.
Every algorithm reuses saved factors without refactorizing source snapshots.
Rank-deficient factors instead build an independent native dense pseudoinverse
action, with the allocation costs described below.


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
close(M)
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
close(M)
```


```julia
using Colmat, LinearAlgebra, SparseArrays
A0=ComplexF64[4 0; im 3; 0 1]
F=qr(FaerSparseMatrix(sparse(A0)); algorithm=:supernodal, ordering=[2,1])
@assert Vector(F.pcol)==[2,1]
@assert Matrix(F.Q)*Matrix(F.R)≈A0[:,[2,1]]
@assert Matrix(F)≈A0
close(F)
```


### Full orthogonal bases from sparse QR

`F.Q` and `F.Qthin` return the independent thin basis; `F.Qfull` returns an
independent square orthogonal/unitary basis for every sparse QR algorithm.
The complement is formed from saved native Householder factors, including
structurally missing pivots. No source snapshot is refactorized.

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


### Rank and tolerance-aware solves from saved sparse QR

`svdvals(F)`, `rank(F;atol,rtol)` and `pinv(F;atol,rtol)` use saved sparse QR
factors. Singular values are computed by native dense SVD of the n×n R factor.
The rank threshold is `max(atol,rtol*maximum(svdvals(F)))`; the default relative
tolerance is `min(size(F)...)*eps(real(eltype(F)))` when atol is zero.
`pinv(F)` also materializes thin Q and returns an independent native dense result.
Three-argument `ldiv!(X,F,B;atol,rtol)` computes the corresponding minimum-norm
least-squares solution, including transpose/adjoint factors and mixed RHS types.
This explicit tolerance path allocates and cannot use caller-owned scratch.
Ordinary `F \ B` also supports rank-deficient factors; wide sparse QR uses the native construction described below.

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
close(P)
```


### Native QR for originally wide sparse inputs

Wide inputs use a native square-prefix QR and direct saved Householder adjoint
applications to project the remaining columns. Column projection processes one
column at a time without materializing a dense Q or the complete tail. `F.Q` and
`F.Qfull` are m×m; `F.R` is m×n. The full input column permutation is retained
by symbolic reuse and numeric updates.
Direct normal and transpose/adjoint solves use native dense SVD of saved R
for minimum-norm least squares, including rank-deficient wide inputs. Each direct
solve materializes m×n dense R and allocates SVD workspace; caller-owned scratch
is not connected for these factor solves. A `FaerLinearOperator(F)` instead
caches the pseudoinverse of R during construction and reuses caller scratch/team
for subsequent applications. Scalar promotion preserves the cached rank cutoff.
The default cutoff matches rank/pinv:
`min(size(F)...)*eps(real(eltype(F)))*maximum(svdvals(F))`.

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
close(G)
```

### Rank-aware sparse QR backslash

For all sparse QR algorithms, ordinary `F \ B`, transpose/adjoint solves,
and `FaerLinearOperator(F)` apply the default numerical-rank cutoff to saved
factors, including zero and rank-deficient inputs. A sparse norm bound first
certifies well-conditioned tall/square R factors; inconclusive cases use native
SVD of dense R and materialize thin Q. This fallback allocates and can be costly
for large sparse factors. It does not read the source snapshot.
Caller-owned scratch currently rejects numerically rank-deficient factors before
changing the output; omit `scratch` for the allocating minimum-norm solve.
Explicit tolerances remain available through `ldiv!(X,F,B;atol,rtol)` and `pinv`.

```julia
using Colmat, SparseArrays, LinearAlgebra
A = [1.0 2; 2 4; 0 0]
F = qr(FaerSparseMatrix(sparse(A)))
b = ones(3)
@assert isapprox(Vector(F \ b), pinv(A)*b; atol=1e-12)
close(F)
```

### Julia-defined operator and preconditioner actions

`FaerLinearOperator(apply!,adjoint_apply!,m,n;T)` supplies native LinOp/BiLinOp
traits with Julia functions. Each function receives owned Julia matrices `(Y,X)`
and must overwrite all of Y with the forward or adjoint product. The functions
must accept batched RHS and promoted native scalar types. Captured state is held
by reference and shared by copies, transforms and compositions. Callback arrays
may be retained after a call; they never borrow native temporary storage.

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
close(M)
close(op)
```

CG/BiCGSTAB/LSMR, preconditioner application and partial spectra still run their
native algorithms. They invoke Julia actions synchronously on the calling Julia
thread. The task is temporarily pinned for this call. Same-task nested Colmat
calls are supported; calls from other tasks serialize on the owner lock until
the callback returns. User exceptions return through the native boundary before being rethrown;
failed actions preserve staged destinations. Outputs must be finite and fully
written. Empty-domain/range and zero-column products are treated as zero maps.
Native composition/conjugation intermediates can reuse `scratch=`/`team=` as
for other operators; Julia callback buffers and result staging allocate separately.
This interface does not promise allocation-free Julia actions.

### Iterative histories and stopping callbacks

`cg`, `bicgstab`, `lsmr` and their in-place forms accept `history=true` and
`callback=state -> ...`. Each callback receives an independent Julia iterate
`state.x`, the completed update number `state.iteration`, true primal residual
norms, and (for LSMR) true preconditioned normal residual norms. Return `nothing`
or `true` to continue and `false` to stop. A requested stop returns
`reason=:stopped`, `converged=false`, including with `check=true`.

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

History retains residual records without full iterates. Observation allocates
copies and makes extra native operator applications to recompute residuals.
With both options disabled, these copies and extra applications are absent.
Callbacks run synchronously on the calling Julia task; retained iterate copies
stay valid after the solver returns. Initial convergence or zero update limits
produce no callback/history entries. Callback exceptions propagate after safely
returning through Rust; in-place destinations remain unchanged. A deliberate
stop copies the stopped iterate into an in-place destination.

Julia-defined operator/preconditioner actions are available through the constructor
above. Iteration workspace/team control remains separate work under Issue #22.

LSMR can falsely report native convergence on extreme uniform scales
([faer #245](https://github.com/tensor4all/faer/issues/245)). Colmat's true-residual
check rejects those results. Observing a nonfinite failed iterate preserves the
same failure diagnostics; the history bridge accepts its nonfinite values.


### Reuse workspace and teams for sparse LU/Cholesky solves

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
