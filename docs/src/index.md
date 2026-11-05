# Colmat.jl マニュアル

Colmat.jl は Rust の [t4a-faer](https://github.com/tensor4all/faer) を Julia から呼び出す線形代数パッケージです。
`FaerMatrix` などの Rust 所有配列を Julia の `AbstractArray` として扱い、
`LinearAlgebra` の分解・求解や疎行列・反復法を利用できます。

初めて使う場合は [インストール](installation.md) の後、
[配列・型・所有権](manual/arrays.md) と [密行列の分解と求解](manual/dense.md) を読んでください。

| 目的 | ページ |
| :--- | :--- |
| 配列の構築・変換、ビューと寿命を理解する | [配列・型・所有権](manual/arrays.md) |
| 分解を保存して繰り返し解く | [密行列の分解と求解](manual/dense.md) |
| 固有値・左右固有ベクトル、Schur・QZ を計算する | [固有値・Schur・QZ](manual/spectral.md) |
| CSC/CSR、ordering、symbolic analysis を使う | [疎行列と直接法](manual/sparse.md) |
| 疎な最小二乗・最小ノルム問題を解く | [疎 QR](manual/sparse-qr.md) |
| 作用素や前処理を使って反復法で解く | [線形作用素・反復法](manual/operators.md) |
| Householder・Jacobi・帯行列への縮約を使う | [反射・回転・縮約](manual/reductions.md) |
| scratch とスレッドチームを再利用する | [作業領域とチーム](manual/resources.md) |
| 統計・乱数・NPY を使う | [統計・乱数・NPY](manual/utilities.md) |

## 共通の規約

密配列のネイティブ数値型は `Float32`・`Float64`・`ComplexF32`・`ComplexF64` です。
添字と置換は Julia と同じ 1-based です。構築・明示変換は通常コピーを作り、
ビューは元の所有者を保持します。`!` 付き操作のストレージ共有や失敗時の挙動は各節で説明します。

`scratch=` はカーネルの一時領域を制御します。返り値・型変換・更新の staging などは
別途メモリを確保するため、呼び出し全体が allocation-free になるとは限りません。
`team=` を受け取る場合も、カーネルによっては逐次実行になります。

このマニュアルは README の利用例を分野別に整理したものです。
API を探す場合は [API 案内](reference/api.md)、開発やビルドは [開発ガイド](development.md) を参照してください。
実装の履歴・未完了項目は [実装台帳](https://github.com/AtelierArith/Colmat.jl/blob/main/docs/implementation-status.md) に記録しています。
