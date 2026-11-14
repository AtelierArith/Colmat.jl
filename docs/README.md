# ドキュメントの開発

現在の公開マニュアルは [docs/src/index.md](src/index.md) が入口です。
Documenter.jl と DocumenterVitepress.jl でビルドします。
リポジトリのルートで実行してください。

```sh
julia --project=docs -e 'using Pkg; Pkg.instantiate(; workspace=true); Pkg.build("Colmat")'
julia --project=docs docs/make.jl
python3 -m http.server 8000 --bind 127.0.0.1 --directory docs/build/1
```

ブラウザーで `http://127.0.0.1:8000` を開きます。
文書生成は公開 API を読み込むため、Colmat のネイティブビルドが必要です。
文書環境はルートの Julia workspace に含まれ、ルートの Manifest を使用します。

本文は `src/`、ページ構成は `make.jl`、表示設定は `src/.vitepress/config.mts` で
管理します。`jldoctest` のコード例と公開 API の docstring は文書ビルド時に検証します。

- [実装状況](implementation-status.md)は現行マニュアルへの案内です。
- [API 対応表](public-api-mapping.md)も同じ公開マニュアルを参照します。
- 旧 API 台帳・再現コードは [設計変更](src/migration.md)に記載した Git 履歴を参照します。

GitHub Pages へのデプロイは `.github/workflows/documentation.yml` で管理します。
ローカルの `docs/make.jl` は公開せず、`--deploy` 指定時だけデプロイ処理を呼びます。
