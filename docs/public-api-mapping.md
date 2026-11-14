# 公開 API と BLAS 対応範囲

Colmat 0.2 の API と対応範囲は次のページで管理します。

- [Julia 側の公開 API](src/reference/api.md)
- [BLAS の型・整数 ABI・シンボル一覧](src/reference/coverage.md)
- [通常の Julia 配列による使用例](src/getting-started.md)

Rust colmat の分解・疎行列・反復法などの全機能を Julia に公開する対応表では
ありません。Julia 側の行列演算は Base / LinearAlgebra の API を使用します。
