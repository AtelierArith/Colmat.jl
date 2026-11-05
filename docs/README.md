# ドキュメントの開発

[マニュアル](src/index.md) は Documenter.jl + DocumenterVitepress.jl でビルドします。
リポジトリのルートで次を実行してください。

```sh
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
python3 -m http.server 8000 --directory docs/build/1
```

`http://localhost:8000` を開くと確認できます。Colmat や Rust のビルドは不要です。
本文は `src/`、ナビゲーションは `make.jl` で編集します。
コードフェンスは表示用で、ドキュメントビルドでは実行しません。
テスト・CI・公開設定の詳細は [開発ガイド](src/development.md) を参照してください。

このディレクトリ直下の実装台帳・API 対応表・Issue 記録と `reproducers/` は
開発履歴として維持し、サイトのビルド対象には含めません。
