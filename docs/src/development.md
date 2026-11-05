# 開発ガイド

## テスト

リポジトリのルートでパッケージの依存関係とネイティブライブラリを準備します。

```sh
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.build("Colmat")'
julia --project -e 'using Pkg; Pkg.test()'
julia --project -e 'using Pkg; Pkg.test(; test_args=["--jobs=4"])'
julia --project -e 'using Pkg; Pkg.test(; test_args=["--list"])'
```

ParallelTestRunner.jl がテストファイルを自動検出し、独立したプロセスで並列実行します。

## ドキュメントのビルド

[Documenter.jl](https://documenter.juliadocs.org/) と
[DocumenterVitepress.jl](https://github.com/LuxDL/DocumenterVitepress.jl) を使います。
リポジトリのルートで次を実行します。

```sh
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
python3 -m http.server 8000 --directory docs/build/1
```

ブラウザーで `http://localhost:8000` を開きます。
DocumenterVitepress が Node.js とフロントエンド依存を用意するため、初回はネットワーク接続が必要です。
マニュアルのコードフェンスは表示用であり、このビルドでは実行しません。
Colmat のロードや Rust のビルドも不要です。数値動作はパッケージのテストで確認してください。

本文は `docs/src/`、ページ順は `docs/make.jl` の `pages` で管理します。
新しい利用例は該当分野のページに追加し、README は概要と入口を保ちます。
API 案内は手動で管理する主要 API の索引です。

## CI と公開

GitHub Actions は pull request、main への push、`v*` タグでドキュメントをビルドします。
main とタグの push 時に `DocumenterVitepress.deploydocs` で `gh-pages` へ公開します。
プレビューは各実行の `documentation` artifact から取得できます。
GitHub Pages の公開元はリポジトリ設定で **Deploy from a branch → gh-pages / (root)** にしてください。
公開先は `https://atelierarith.github.io/Colmat.jl/dev/` です（初回の公開が成功した後に利用可能）。

## 実装記録

履歴・監査用の記録は既存の場所に残しています。

- [実装台帳](https://github.com/AtelierArith/Colmat.jl/blob/main/docs/implementation-status.md)
- [公開 API 対応表](https://github.com/AtelierArith/Colmat.jl/blob/main/docs/public-api-mapping.md)
- [Issue #1 の棚卸し](https://github.com/AtelierArith/Colmat.jl/blob/main/docs/issue-1-inventory.md)
- [再現例](https://github.com/AtelierArith/Colmat.jl/tree/main/docs/reproducers)

これらは過去の時点の制約も含むため、現在の利用方法はマニュアルとソースを参照してください。
