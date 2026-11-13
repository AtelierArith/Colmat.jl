# 統計・乱数・NPY

統計量、乱数生成、NPY ファイルを扱います。

## 平均・分散・標準偏差

`using Statistics`を追加すると `mean`、`var`、`varm`、`std`を使えます。`dims=1`は各列、`dims=2`は各行について集計します。`dims`を省略すると全要素をまとめた数値を返し、指定すると Julia 配列を返します。

既定の `nan=:propagate`は NaNを結果に伝えます。`nan=:ignore`は NaNを含む要素を除いて個数も補正します。`corrected=false`は個数そのもので割る母分散です。複素数の分散は実数になります。空の集計結果は NaN、1 要素の標本分散も Juliaと同じく NaNです。

ビューや逆順のビューも直接読めます。`varm`に渡す平均によって計算型が広がる場合は変換用コピーを作ります。全要素の集計と `dims` 付き集計の結果型は `Statistics.varm`の規約に従います。`BigFloat` など未対応の計算型になる平均は `ArgumentError`です。

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

```julia
using Colmat, Statistics
A = FaerMatrix(Float32[1 2; 3 5])
@assert varm(A, 2.0) isa Float64
@assert eltype(varm(A, [2.0 3.0]; dims=1)) === Float32
@assert var(view(A, 2:-1:1, :); dims=2) ≈ var(Float32[3 5; 1 2]; dims=2)
```

## 乱数とランダムな直交行列

`using Random`を追加します。`FaerRNG(seed)`は Rust 側の乱数列を保持します。`rand`は実部・虚部それぞれが `[0,1)`の一様乱数、`randn`は標準正規乱数です。複素正規乱数は実部・虚部の分散が各 1/2になります。

`randunitary`は、実数型なら `Q'Q ≈ I`を満たす直交行列、複素数型なら同じ関係を満たすユニタリ行列を生成します。`rand!`と `randn!`は既存の配列を更新します。

Juliaの他の乱数生成器を渡した場合、空でない各呼び出しで `UInt64`の種を 1 個取り出して一時的な Rustの乱数列を作ります。空の呼び出しは種を消費しません。同じ状態・依存関係・呼び出し順なら再現できますが、Julia 配列を直接生成したときと同じ数列ではありません。連続した Rustの乱数列が必要なら `FaerRNG`を使ってください。

`rand!(rng, A, X)`は範囲・コレクション・Juliaの分布オブジェクトなど、`rand(rng, X)`に対応する値から標本を生成します。同じ型・形状の Julia 配列に生成してから一括代入するため、標本生成や型変換に失敗しても `A`は変わりません。一様分布の区間や正規分布の平均・標準偏差は、下の例のように平行移動と拡大で指定できます。乱数生成に `team`の並列版はありません。

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

```julia
using Colmat, Random
A = zeros(FaerMatrix{Float64}, 2, 3)
rand!(Xoshiro(1), A, 1:6)
@assert Matrix(A) == rand!(Xoshiro(1), Matrix{Float64}(undef, 2, 3), 1:6)
B = 2 .+ 3 .* Matrix(randn(FaerRNG(1), FaerMatrix{Float64}, 2, 2))  # 平均 2、標準偏差 3 の正規分布
```

## NumPyの NPY ファイルを読む

NPYは NumPyの配列を保存するファイル形式です。`loadnpy(path)`はファイルを読み、独立した変更可能な `FaerMatrix`を返します。バイト列と IO も入力にできます。下の例ではパッケージ付属の小さなファイルを使います。

対応する型は `Float32`、`Float64`、`ComplexF32`、`ComplexF64`、形式は NPY 1〜3です。行優先・列優先と両方のバイト順に対応します。スカラーは 1×1、ベクトルは n×1 行列になります。元の形状などは `npyinfo`で確認できます。3 次元以上や未対応の型、不整合な長さ、余分な末尾データは拒否します。IOは現在位置から末尾までを 1 配列として読みます。

`npyview`（または `loadnpy(...; view=true)`）は読み取り専用のコピーを保持します。入力とメモリを共有せず、ファイルのメモリマップでもありません。`copy(V)`は変更可能な行列、`deepcopy(V)`は独立した読み取り専用ビューを作ります。`close(V)` 後も、これらの独立したコピーは使えます。

```julia
using Colmat
path = joinpath(pkgdir(Colmat), "test", "fixtures", "npy_c_f64.npy")
A = loadnpy(path)
@assert Matrix(A) == [1.0 2 3; 4 5 6]
V = npyview(path)
@assert npyinfo(V).shape == (2, 3)
@assert V[2, 3] == 6.0
B = copy(V) # 独立した変更可能な FaerMatrix
close(V)
@assert Matrix(B) == Matrix(A)
```

### ファイルやバイト列をコピーせずに読む

`npyborrow(path)`と `npyborrow(io::IOStream)`はファイルをメモリに対応付け、`npyborrow(bytes)`はバイトベクトルを共有します。返り値は `FaerBorrowedMatrix`です。`sum`、`mean`、積などは元のデータを直接読みます。`npyinfo`は借用時に記録した形状・格納順・書き込み可否を返します。

既定は読み取り専用で、代入や `mul!`の出力先にはできません。`writable=true` ならファイルまたはバイト列へ直接書き込みます。ファイルは書き込み権限が必要です。`Colmat.Mmap.sync!(W)`で変更を同期できます。

コピーしない読み取りには、実行環境と同じバイト順と要素に合った配置が必要です。条件を満たさない入力は既定で `loadnpy`によるコピーに切り替わります。`fallback=false`は代わりに例外にします。`writable=true`ではコピーへの切り替えはしません。

同じプロセス内で同じファイルを複数回借用すると、同じ対応付けを共有します。入力と出力が重なる演算では入力を一時コピーしてから更新します。別プロセスや外部で作ったメモリマップによる更新との順序は管理しません。

借用は元のバイト列やメモリマップを保持し、参照がなくなると GCによって解放されます。明示的な `close`はありません。ストリームを閉じても対応付けは残ります。ファイルサイズ変更時には次の借用で対応付けを更新しますが、読み取り中のファイル切り詰めは OSのエラー（`SIGBUS`）を起こし得ます。利用中は切り詰めないでください。

```julia
M = npyborrow(path)      # 読み取り専用のメモリマップ。データのコピーなし
@assert M == [1.0 2 3; 4 5 6]
@assert strides(M) == (3, 1) # 行ストライドを使って C の行優先順で読む
C = copy(M)              # 独立した FaerMatrix

tmp = tempname()
cp(path, tmp)
W = npyborrow(tmp; writable = true) # 読み書き可能な共有メモリマップ
mul!(W, FaerMatrix([0.0 1; 1 0]), npyborrow(tmp)) # ファイルに対応する領域の行を交換
Colmat.Mmap.sync!(W)     # 同期。重なる入力は更新前にコピー済み
@assert loadnpy(tmp) == [4.0 5 6; 1 2 3]
```

## 最小値・最大値

`minimum`、`maximum`、`extrema`は実数配列の最小値・最大値を返します。`Float32`の入力なら結果も `Float32`です。NaNは結果に伝わり、負のゼロは正のゼロより小さいものとして扱います。

`dims`は Baseと同じ規約で、空の次元を集計すると `ArgumentError`です。`init` や `identity` 以外の関数を先頭に渡す形式は Juliaの汎用処理を使います。複素数の順序付けも Juliaの規約に従います。

```julia
using Colmat
A = FaerMatrix(Float32[3 -2; 1 4])
@assert minimum(A) === -2f0
@assert maximum(A) === 4f0
@assert extrema(A) === (-2f0, 4f0)
A[1, 1] = -0f0
A[1, 2] = 0f0
A[2, 1] = 0f0
A[2, 2] = 0f0
@assert signbit(minimum(A))
@assert !signbit(maximum(A))
@assert extrema(A; dims = 1) == [(0f0, 0f0) (0f0, 0f0)]
```

## ビューの集計とノルム

集計は配列から 1 個または少数の値を得る操作です。`sum`は和、`dot`は内積、`norm`は要素の大きさをまとめた尺度です。ビューや `faerview` も直接読めます。`dot`は第 1 引数を複素共役にし、形状が違っても同じ要素数なら列優先順に対応付けます。

`norm(A)`は全要素の絶対値の二乗和の平方根です。行列がベクトルをどれだけ拡大するかを調べる場合は `opnorm`を使います。`norm(A, 0)`は非ゼロ数、`norm(A, Inf)`は最大絶対値、`norm(A, -Inf)`は最小絶対値です。NaN 以外の実数 `p`に対応します。和の計算順序が Juliaと異なるため、丸め誤差による差が生じる場合があります。

`sum!`、`minimum!`、`maximum!`、`extrema!`は出力先の長さが 1の次元に沿って集計します。`init=false`は出力先の現在値と結果を組み合わせます。`all(isfinite, A)`、`any(isnan, A)`、`istriu`、`istril`、`isdiag` などで値や構造も調べられます。

```julia
using Colmat, LinearAlgebra
J = [3.0 -4.0; 0.0 1.0]
B = faerview(J)
x = FaerVector([1.0, 2.0, 3.0])
@assert dot(view(x, 3:-1:1), faerview([1.0, 0.0, 1.0])) == 4.0
@assert norm(B, 0) == 3.0
@assert norm(B, 3) ≈ norm(J, 3)
@assert norm(B, -Inf) == 0.0
@assert sum(view(x, 1:2:3)) == 4.0
@assert minimum(B; dims = 2) == [-4.0; 0.0;;]
@assert all(isfinite, B) && !any(isnan, view(x, :))
r = zeros(1, 2)
@assert sum!(r, B) == [3.0 -3.0]
@assert minimum!(r, view(FaerMatrix(J), 2:-1:1, :); init = false) == [0.0 -4.0]
@assert istriu(B) && !issymmetric(B)
```
