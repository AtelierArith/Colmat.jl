# インストールと最初の計算

Julia 1.13、Git、ネットワーク接続、Rust 1.89 以降が必要です。RustCallの Rust ツールチェーンも利用できます。
次の手順でリポジトリを用意し、依存関係とネイティブライブラリを準備します。

```sh
git clone https://github.com/AtelierArith/Colmat.jl.git
cd Colmat.jl
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.build("Colmat")'
```

## 最初の計算

リポジトリのルートで `julia --project` を実行して Julia を起動し、次を試してください。

```julia
using Colmat, LinearAlgebra
A = FaerMatrix([4.0 1; 1 3])
B = FaerMatrix([1.0 2; 3 4])
C = A * B
x = A \ [1.0, 2.0]
F = svd(A)
@assert F.U * Diagonal(F.S) * F.Vt ≈ Matrix(A)
```

`x` はおよそ `[0.090909, 0.636364]` です。`@assert` は条件を満たすと何も表示せず、満たさないと例外を投げます。テスト全体を実行したい場合は[開発ガイド](development.md)を参照してください。

分野別の使い方は [配列・型・所有権](manual/arrays.md)と [密行列の分解と求解](manual/dense.md)へ進んでください。

## ビルドの詳細

`julia --project deps/build.jl`でもビルドできます。Gitとネットワーク接続、
Rust 1.89 以降が必要です（RustCallの Rust ツールチェーンも利用できます）。

`deps/build.jl`は https://github.com/tensor4all/faerのデフォルトブランチの
最新 HEAD（Rust クレート `colmat`）を `deps/t4a-faer`にクローンし、再実行時も最新 HEADに更新します。
ローカル変更がある場合は上書きせず停止します。使用したコミット SHAは
ビルド出力に表示します。最新版を追うため、将来の API 変更でビルドが失敗する
可能性があります。

`deps/rust`の橋渡し用 `cdylib`がこのクローンを Cargoの ローカルパス依存として
参照し、RustCallのキャッシュにコンパイルされます。`extern/t4a-faer`は不要です。
クローン内の `faer/`が `colmat` クレートです。橋渡しコードでは Cargoの別名 `faer`を使います。
初回の `using Colmat`の前にビルドしてください。通常の読み込みでは Gitの更新は
行わず、最後にビルドしたソースを使用します。
