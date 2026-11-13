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
本文・見出し・コメントは日本語で書き、初出の線形代数用語は用途と一緒に説明します。
基本例では GC に管理を任せ、解放時期を示す例では `with_resource` などの `do` 構文を使います。
GC と明示的な解放の違いは [GC と資源の寿命](src/manual/lifetimes.md) を参照してください。
テスト・CI・公開設定の詳細は [開発ガイド](src/development.md) を参照してください。

このディレクトリ直下の実装台帳・API 対応表・Issue 記録と `reproducers/` は
開発履歴として維持し、サイトのビルド対象には含めません。
