# 0.2 への設計変更

0.2 では Julia–Rust バインディングから BLAS バックエンドへ設計を変更しました。
旧版との後方互換性はありません。

| 旧版の使い方 | 現在の使い方 |
|---|---|
| `FaerMatrix` / `FaerVector` を作る | 通常の `Matrix` / `Vector` を使う |
| 専用型に対して乗算や `mul!` を呼ぶ | 同じ Base / LinearAlgebra API を通常配列に使う |
| `FaerTeam` で乗算する | `BLAS.set_num_threads` または `Colmat.set_num_threads` を使う |
| `close` / `with_resource` で Rust ハンドルを解放する | 通常配列は Julia が所有。専用ハンドルの管理は不要 |
| RustCall でバインディングを生成する | Cargo で BLAS ABI 共有ライブラリをビルドする |
| 専用の分解・疎行列・反復法・NPY API を使う | これらの Colmat API は削除。利用側で対応する Julia API を選ぶ |

通常配列の LU・SVD・一般／自己随伴固有値・Cholesky は LAPACK ABI を経由して
colmat-rs の分解を使います。旧専用型 API の互換層を提供する変更ではありません。

## 旧資料

旧版の機能台帳、Issue #1 の計画、RustCall や旧 API の再現コードは、現在の
ドキュメントから削除しました。旧コードと一緒に参照する場合は、設計変更前の
Git commit `ced1e13bf765f18b44e92317bc053b8e212fed4e` を使用してください。

```sh
git show ced1e13bf765f18b44e92317bc053b8e212fed4e:docs/implementation-status.md
git ls-tree -r --name-only ced1e13bf765f18b44e92317bc053b8e212fed4e docs/reproducers
```

旧記録のテスト件数や依存 revision は、その旧実装を検証した履歴です。
現在の対応範囲や検証結果には転用しません。
