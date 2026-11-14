# Colmat.jl

Rust の [colmat](https://github.com/tensor4all/colmat-rs) を使う Julia の BLAS バックエンドです。
`using Colmat` で libblastrampoline の転送先を登録します。通常の `Matrix` と
`LinearAlgebra` をそのまま使えます。専用の行列型・RustCall は不要です。

```julia
using Colmat, LinearAlgebra

A = randn(100, 80)
B = randn(80, 60)
C = A * B
mul!(C, A, B)

BLAS.get_config()           # Colmat と既存のフォールバックを表示
BLAS.set_num_threads(2)     # Colmat と既存バックエンドの両方を設定
```

Julia 1.13、Git、Rust 1.89 以降が必要です。初回ビルドにはネットワーク接続が必要です。

```sh
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.build("Colmat")'
julia --project examples/poc.jl
julia --project -e 'using Pkg; Pkg.test()'
```

ビルドは `deps/upstream.toml` に記録した Rust ソースと `Cargo.lock` を使用します。
ビルド中に upstream の HEAD を自動更新しません。共有ライブラリの再ビルド後は
Julia を再起動してください。

## 提供範囲

LP64 / ILP64、Float32 / Float64 / ComplexF32 / ComplexF64 を提供します。

| 分類 | Colmat に転送するルーチン |
|---|---|
| Level 1 | copy, swap, scal, axpy, axpby, dot / dotu / dotc, nrm2, asum, iamax, rot |
| Level 2 | gemv, symv, hemv, ger / geru / gerc, syr, her, trmv, trsv |
| Level 3 | gemm, symm, hemm, syrk, syr2k, herk, her2k, trmm, trsm |

行列積は Julia のメモリを借用して Rust colmat のカーネルで実行します。
対称・Hermitian・三角行列積などは必要に応じて一時領域を確保します。
一部のベクトル演算と三角求解はこのパッケージの Rust 互換層で実装しています。
初期スレッド数は1です。設定した並列実行幅は行列積に適用されます。

**未提供の BLAS（帯・packed 格納、rotg / rotmg / rotm など）と LAPACK は、
ロード前のバックエンドに残します。** `lu`・`svd`・`eigen` 自体の実装を
Colmat に置き換えるものではありません。フォールバック内部の BLAS 呼び出しが
LBT を経由するかは、そのライブラリのリンク構成に依存します。

ロードによる切り替えは Julia プロセス全体に適用されます。ほかのバックエンドを
後からロードした場合は `Colmat.load_backend()` で Colmat を再選択できます。
バックエンドの切り替えは並列計算を始める前に行ってください。

0.2 は設計を全面変更しました。0.1 の `FaerMatrix` などの API は提供しません。

[マニュアル](docs/src/index.md) / [導入例](docs/src/getting-started.md) /
[BLAS 対応範囲](docs/src/reference/coverage.md) / [API](docs/src/reference/api.md) /
[開発](docs/src/development.md) / [トラブル対処](docs/src/troubleshooting.md)

旧版の資料を参照する方法は [0.2 への設計変更](docs/src/migration.md) に記載しています。
