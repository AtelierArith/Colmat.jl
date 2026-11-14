# Colmat 0.2 の実装状況

現在の Colmat は libblastrampoline に登録する BLAS / LAPACK バックエンドです。

- [LBT 全ルーチンの接続候補・未対応一覧](src/reference/lbt-inventory.md)
- [Julia API の実測対応表](src/reference/support-matrix.md)
- [対応する型・整数 ABI・ルーチン](src/reference/coverage.md)
- [転送とフォールバックの仕組み](src/manual/backend.md)
- [ソース構成・依存更新・検証方法](src/development.md)

対応範囲は Rust colmat 全体の機能一覧ではなく、`deps/rust/src/abi.rs` が公開する
BLAS シンボルを基準にします。未提供の BLAS と LAPACK の転送先は保持します。

旧版の機能台帳、Issue #1 の計画、バグ再現コードは現在の設計には適用しません。
[設計変更と旧資料の参照方法](src/migration.md)を参照してください。
