# Colmat.jl

RustCall.jl **v0.9.1** で動作確認した、t4a-faer を Julia から呼び出す PoC。
Tenferro.jl と同じく `#[julia]` と `@rust_crate` を使用します。

```sh
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.build("Colmat")'
julia --project examples/poc.jl
julia --project -e 'using Pkg; Pkg.test()'
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

`FaerMatrix` は Rust の `faer::Mat<f64>` を所有する `AbstractMatrix{Float64}`
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

`full_piv_lu`、`ldlt`、`bunchkaufman`、一般・自己随伴・一般化の
`eigen`／`eigvals`、三角行列の求解・逆行列、`det`／`logabsdet`／`logdet`、
`sum`／`norm`／`dot`／`tr`／`kron`、`pfaffian` も Rust に接続しています。
現在は Float64 の密行列が対象です。Issue #1 全体の進捗と残りの対象は
[実装台帳](docs/implementation-status.md) に記録しています。

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
ゼロコピー・並列実装ではありません。複素数・疎行列は対象外です。

RustCall v0.9.1 の制限として、受信オブジェクトを持つメソッドの `Result<Self, String>`
返却と、引数の `&OwnedMatrix` の変換に問題がありました。最小再現例と修正案を
[RustCall #564](https://github.com/AtelierArith/RustCall.jl/issues/564)・
[#565](https://github.com/AtelierArith/RustCall.jl/issues/565) に報告しています。
現在は新しい出力オブジェクトに結果を書き込み、入力の所有者を `GC.@preserve` で
保持した上でポインタを借用する方法を使用しています。
