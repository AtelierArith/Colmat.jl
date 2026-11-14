# 開発

## 構成

- `src/Colmat.jl`: モジュールとロード時の初期化。
- `src/backend.jl`: LBT 登録、ライブラリの場所、スレッド設定。
- `deps/rust/src/abi.rs`: LP64 / ILP64 の Fortran BLAS と CBLAS 内積の入口。
- `deps/rust/src/kernels.rs`: 配列の借用と Rust 計算処理。
- `deps/rust/src/threading.rs`: OS スレッドごとのプールとスレッド設定。
- `deps/upstream.toml` / `deps/rust/Cargo.lock`: ソースと依存関係の固定。

## 検証

```sh
cargo fmt --manifest-path deps/rust/Cargo.toml --check
cargo test --manifest-path deps/rust/Cargo.toml --locked
cargo clippy --manifest-path deps/rust/Cargo.toml --locked --no-deps -- -D warnings
julia --project -e 'using Pkg; Pkg.build("Colmat"); Pkg.test()'
```

テストは転送先を LP64 / ILP64 の両方で検査し、スカラーのループによる参照計算、
転置・複素共役、パディング、負ストライド、ゼロ係数、空配列、三角演算、
スレッド設定、LAPACK フォールバックを確認します。Julia 標準の BLAS テストも
同じプロセスの転送設定で実行します。

旧版の台帳・再現コードは現在の文書から削除しました。
[0.2 への設計変更](@ref)に Git 履歴の参照方法を記載しています。

## upstream の更新

`deps/colmat-rs` を目的のコミットに checkout し、`deps/upstream.toml` の revision を
同じ値に更新します。必要に応じて Cargo.lock を更新し、ビルドとテストを実行します。
依存元の checkout にパッケージ側の修正を混ぜないでください。

共有ライブラリを書き換えた後は新しい Julia プロセスで検証してください。


## BLAS シンボルを追加する

`abi.rs` で LP64 と ILP64 の入口を揃え、`kernels.rs` の計算処理に接続します。
複素数の係数・戻り値、負ストライド、ゼロ係数、未参照の三角部分や単位対角を
読み出さない条件は、ルーチン固有の規約に従ってください。

Julia 配列の借用は呼び出し中に限定し、所有権を Rust 側へ移したりポインタを
次の呼び出しまで保存したりしません。LBT とネイティブ ABI の両方で転送先と
計算結果を確認し、[BLAS 対応範囲](@ref)も同時に更新します。

## 文書の検証

```sh
julia --project=docs -e 'using Pkg; Pkg.instantiate(; workspace=true)'
julia --project=docs docs/make.jl
```

文書ビルドは公開 API の掲載漏れと doctest を検査します。
対応シンボル数の doctest は、文書と LBT の実際の登録内容を照合します。
公開例は `examples/poc.jl` でも実行できます。


## 実測対応表の更新

[Julia API の実測対応表](@ref)は、通常配列の実行結果と LBT 登録先を照合して生成します。

```sh
julia --project=test test/support_matrix.jl --output=docs/src/reference/support-matrix.md
```

生成前に `Pkg.build("Colmat")` を実行し、新しい Julia プロセスを使ってください。
プローブは `Pkg.test()` にも含まれますが、通常のテストでは文書を上書きしません。
生成版は検証した Julia・プラットフォーム・ソースコミットを記録したスナップショットです。
Rust に機能があること、Julia から呼べること、Colmat が担当することを分けて更新します。


## LBT 全シンボルの監査

```sh
python3 docs/generate_lbt_inventory.py
julia --project=test test/lbt_inventory.jl
```

Python 3.11 以降はこの監査を再生成する場合だけ必要です。通常のビルド・ロードには不要です。
`docs/lbt-capabilities.json` で公開 API の根拠と制約を管理し、LBT の全 canonical 名を
TSV に出力します。未分類の補助関数も独立公開 API なしの一覧へ記録し、省略しません。
`test/lbt_inventory.jl` は実際の LBT シンボル集合と記録の一致・重複の不存在を検査します。

`deps/generate_symmetric.py` は自己随伴 EVD の精度・整数 ABI ごとの具体的な C 署名を
`lapack_symmetric.rs` に生成します。変更時は生成後に Rustfmt を実行します。
