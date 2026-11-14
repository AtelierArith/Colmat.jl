# インストール

## 必要な環境

- Julia 1.13。
- Rust 1.89 以降と Cargo。
- Git。
- 初回の Rust ソース・Cargo 依存関係取得に必要なネットワーク接続。

Rust のコンパイルに必要な、その OS の開発ツールも用意してください。
現在の CI 設定は Linux と macOS を対象にしています。

## リポジトリからビルドする

リポジトリのルートで実行します。

```sh
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.build("Colmat")'
julia --project examples/poc.jl
```

別の Julia プロジェクトからこの checkout を利用する場合は、そのプロジェクトで
ローカルパッケージとして登録します。

```julia
using Pkg
Pkg.develop(path="/absolute/path/to/Colmat.jl")
Pkg.build("Colmat")
```

ビルド後、新しい Julia プロセスで `using Colmat, LinearAlgebra` を実行します。
モジュールのロード時に Rust のダウンロードやコンパイルは行いません。

## ビルドで生成するもの

`deps/build.jl` は次の順に処理します。

1. Rust ソースがなければ `deps/t4a-faer` に取得する。
2. checkout のコミットと `deps/upstream.toml` の revision を照合する。
3. `cargo build --release --locked` で BLAS 共有ライブラリを作る。
4. ライブラリを `deps/usr/lib` に配置する。

Rust ソースと Cargo 依存関係は `deps/upstream.toml` と `deps/rust/Cargo.lock`
で固定します。ビルド中に upstream の HEAD へ自動更新しません。
既存 checkout のコミットが一致しない場合や追跡ファイルに変更がある場合は停止します。

ビルド用 Julia 依存関係は `deps/Project.toml`、文書用は `docs/Project.toml` に
分離しています。RustCall のインストールは不要です。

共有ライブラリを再ビルドした後は Julia を再起動してください。
診断方法は[トラブルシューティング](@ref)を参照してください。
