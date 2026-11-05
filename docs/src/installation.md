# インストールと最初の計算

Julia 1.13、Git、ネットワーク接続、Rust 1.89 以降が必要です。RustCall の Rust ツールチェーンも利用できます。
リポジトリをクローンし、そのルートで次を実行してください。

```sh
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.build("Colmat")'
julia --project examples/poc.jl
julia --project -e 'using Pkg; Pkg.test()'
```

## ネイティブライブラリのビルド

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

分野別の使い方は [配列・型・所有権](manual/arrays.md) と [密行列の分解と求解](manual/dense.md) へ進んでください。
