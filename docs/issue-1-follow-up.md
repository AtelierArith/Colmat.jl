# Issue #1 の継続実装

[Follow-up Issue #2](https://github.com/AtelierArith/Colmat.jl/issues/2) で残作業と
受け入れ条件を追跡する。[Issue #1](https://github.com/AtelierArith/Colmat.jl/issues/1)
は全体の対応表として継続し、全 76 機能行の完了まで閉じない。

## 引き継ぎ時点

実装基準は `7d2de5b`、対応表基準は `e72b115`。
simplicial 疎 QR の成分・再構成・転置/随伴求解、および simplicial 疎 LU/QR の
`ordering=:colamd` と調整値に対応している。これらの実装は main に直接 push 済み。

Julia 1.13.3 / RustCall.jl 0.9.2 で、`Pkg.test(;allow_reresolve=false)` の
82,656 assertions / 61 files（ParallelTestRunner 16 workers）、native Rust 9 tests、
README の 52 Julia blocks が成功した。この検証の faer checkout は
`d5be1473eed9314fca6f80ad8b4072d86282e890`（0.28.1）。

## 残作業

| 対象 | 次の実装・確認 |
|---|---|
| auto/supernodal 疎 LU | native L/U・置換の公開、保持因子からの再構成 |
| auto/supernodal 疎 QR | 薄い native Q・疎 R・列置換・再構成・転置/随伴求解は実装済み。full Q・rank-deficient 求解などを継続 |
| auto/supernodal LU/QR ordering | ordering と native 制御値の指定 |
| 横長・rank-deficient 疎 QR | rank 判定・許容誤差・最小ノルム求解の契約 |
| mixed-type 疎求解 | simplicial LU/QR と全 Cholesky は保持因子を再利用済み。auto/supernodal LU/QR の snapshot 再分解をなくす |
| matrix-free | Julia callback、作用素合成、分解による前処理、反復履歴・停止 callback |
| 一般化 Schur | 公開 upstream API の調査、並べ替え、実共役対・無限固有値の規約 |
| workspace | native 因子構築・求解・残りの kernel の scratch 照会・reserve・不足時の契約 |
| 破壊的演算 | 実現可能な native 直接 in-place 経路への接続 |
| 並列 team | 未接続の分解・疎演算・反復法への拡張と owner lifetime の検証 |
| その他の部分対応 | 借用ビュー、非標準スカラー、shape/stride、構築形式・低水準 tuning |

詳細と優先順位の更新は Issue #2 で行う。全対象の判定には
[canonical 対応表](public-api-mapping.md) の未実装・制約列を用いる。

## 実装ごとの完了条件

- Julia/Base/LinearAlgebra の dispatch を整え、native・Julia fallback・一部対応を区別する。
- 対応するスカラー型・CSC/CSR・形状・複素転置/随伴・昇格・alias・copy/update/close を検証する。
- 保持因子の再利用を主張する場合、source snapshot の変更後にも正しく動作することを検証する。
- 集中テスト、canonical `Pkg.test()`、README 例、必要な native checks とフォーマットを完了する。
- [凍結した調査一覧](issue-1-inventory.md) を保ち、対応表と Issue #1 を同期する。
- 依存 checkout を変更しない。upstream の不具合は実行済み MWE・実出力・具体的修正案とともに適切な repository に報告する。

実装・検証の詳細は [implementation-status.md](implementation-status.md) を参照。

## 公開 API の追加調査（d5be1473）

- 高水準 `sparse::solvers::{SymbolicLu,Lu,SymbolicQr,Qr}` の所有データは private。
  成分公開・ordering・caller scratch の実装では、メモリ配置を推測せず、
  公開低水準 symbolic/numeric API を保持する adapter への移行が必要。
- LU の公開 `LuRef` は行・列置換と通常/転置求解を公開するが、
  `NumericLu.raw` と `NumericLuRaw` は private。低水準 generic LU へ移すだけでは
  L/U 成分は抽出できない。明示的 simplicial の `l_factor_unsorted` /
  `u_factor_unsorted` は現在の adapter から利用済み。
- QR は `SymbolicQr::factorize_numeric_qr` と `QrRef` により、所有する index/value
  buffers と caller scratch を使える。高水準 `Qr` の private buffers に依存せず
  numeric 更新と求解を維持できるため、auto QR の workspace 接続の実装候補とする。
- supernodal LU の内部 storage への accessor と一般化 QZ の reorder は、
  公開可否を個別に確認する。private API の存在を公開機能の実装済み扱いにしない。
