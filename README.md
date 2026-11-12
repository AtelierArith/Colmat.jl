# Colmat.jl

Rust の [colmat](https://github.com/tensor4all/faer) を Julia から利用する線形代数パッケージです。
RustCall.jl の `#[julia]` と `@rust_crate` を使い、Rust が所有する密・疎行列、
分解、反復法、作業領域を Julia の配列・LinearAlgebra API から扱います。

Julia 1.13、Git、ネットワーク接続、Rust 1.89 以降が必要です。
リポジトリのルートで実行してください。

```sh
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.build("Colmat")'
julia --project examples/poc.jl
```

```julia
using Colmat, LinearAlgebra

A = FaerMatrix([4.0 1; 1 3])
F = lu(A)
x = F \ [1.0, 2.0]
@assert x ≈ [4.0 1; 1 3] \ [1.0, 2.0]
```

ビルドは colmat のデフォルトブランチの最新 HEAD を取得します。
使用したコミット SHA はビルド出力に表示されます。初回の `using Colmat` 前にビルドしてください。

- [マニュアルの目次](docs/src/index.md) / [導入ガイド](docs/src/installation.md)
- [API 案内](docs/src/reference/api.md)
- [ドキュメントのビルド・開発手順](docs/README.md)
- [実装状況と既知の課題](docs/implementation-status.md)

```sh
julia --project -e 'using Pkg; Pkg.test()'
```

テストの並列数や対象一覧の確認方法は [開発ガイド](docs/src/development.md) を参照してください。
