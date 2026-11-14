# Julia API の実測対応表

通常配列の呼び出しが成功するかと、その入口を Colmat が提供するかを分けて記録します。
入口は Julia 1.13 のラッパー実装から選び、LBT の登録先を照合します。動的な呼び出し回数の計測やライブラリ内部を追跡したトレースではありません。

## 検証条件

- Julia: `1.13.3` / `x86_64-linux-gnu`。
- Colmat ソース: `bc0dce2af48590dec20f99976ad7aee8b999a447`。
- Rust colmat ソース: `e1c83482f5a61449d30e989808c3ba5f751ec391`。
- Julia の BLAS 整数 ABI: `ilp64`。
- Colmat 実行幅: `1`。
- 要素型は Float32 / Float64 / ComplexF32 / ComplexF64 の順。

## 実行結果

| Julia API | Float32 | Float64 | ComplexF32 | ComplexF64 | 主な入口の登録先 | Rust 側の実装・接続 |
|---|---|---|---|---|---|---|
| `A * B` | 成功 | 成功 | 成功 | 成功 | Colmat | 行列積 |
| `mul!(C, A, B)` | 成功 | 成功 | 成功 | 成功 | Colmat | 行列積 |
| `A * x` | 成功 | 成功 | 成功 | 成功 | Colmat | 行列積 |
| `dot(x, x)` | 成功 | 成功 | 成功 | 成功 | Colmat | Rust 互換層 |
| `norm(x)` | 成功 | 成功 | 成功 | 成功 | Colmat | Rust 互換層 |
| `BLAS.axpy!` | 成功 | 成功 | 成功 | 成功 | Colmat | Rust 互換層 |
| `UpperTriangular(A) \ x` | 成功 | 成功 | 成功 | 成功 | Colmat | Rust 互換層 |
| `UpperTriangular(A) \ B` | 成功 | 成功 | 成功 | 成功 | Colmat | Rust 互換層 |
| `lu(A)` | 成功 | 成功 | 成功 | 成功 | Colmat | partial_piv_lu（接続済み） |
| `lu(A) \ x` | 成功 | 成功 | 成功 | 成功 | Colmat | LU solve（接続済み） |
| `qr(A)` | 成功 | 成功 | 成功 | 成功 | 既存バックエンド | qr（未接続） |
| `qr(A, ColumnNorm())` | 成功 | 成功 | 成功 | 成功 | 既存バックエンド | col_piv_qr（未接続） |
| `svd(A)` | 成功 | 成功 | 成功 | 成功 | Colmat | svd（接続済み） |
| `svd(A; alg=LinearAlgebra.QRIteration())` | 成功 | 成功 | 成功 | 成功 | Colmat | svd（接続済み） |
| `svdvals(A)` | 成功 | 成功 | 成功 | 成功 | Colmat | SVD（接続済み） |
| `eigen(A)` | 成功 | 成功 | 成功 | 成功 | Colmat | eigen（接続済み） |
| `eigen(Hermitian(A))` | 成功 | 成功 | 成功 | 成功 | Colmat | self_adjoint_eigen（接続済み） |
| `cholesky(Hermitian(A))` | 成功 | 成功 | 成功 | 成功 | Colmat | llt（接続済み） |

## 入口シンボルと実際のライブラリ

各行は上表と同じ API を確認しています。シンボルは上記4型の順です。

| Julia API | 確認した入口 | LBT の登録先 |
|---|---|---|
| `A * B` | `sgemm_`, `dgemm_`, `cgemm_`, `zgemm_` | `libcolmat_blas.so` |
| `mul!(C, A, B)` | `sgemm_`, `dgemm_`, `cgemm_`, `zgemm_` | `libcolmat_blas.so` |
| `A * x` | `sgemv_`, `dgemv_`, `cgemv_`, `zgemv_` | `libcolmat_blas.so` |
| `dot(x, x)` | `cblas_sdot`, `cblas_ddot`, `cblas_cdotc_sub`, `cblas_zdotc_sub` | `libcolmat_blas.so` |
| `norm(x)` | `snrm2_`, `dnrm2_`, `scnrm2_`, `dznrm2_` | `libcolmat_blas.so` |
| `BLAS.axpy!` | `saxpy_`, `daxpy_`, `caxpy_`, `zaxpy_` | `libcolmat_blas.so` |
| `UpperTriangular(A) \ x` | `strsv_`, `dtrsv_`, `ctrsv_`, `ztrsv_` | `libcolmat_blas.so` |
| `UpperTriangular(A) \ B` | `strsm_`, `dtrsm_`, `ctrsm_`, `ztrsm_` | `libcolmat_blas.so` |
| `lu(A)` | `sgetrf_`, `dgetrf_`, `cgetrf_`, `zgetrf_` | `libcolmat_blas.so` |
| `lu(A) \ x` | `sgetrs_`, `dgetrs_`, `cgetrs_`, `zgetrs_` | `libcolmat_blas.so` |
| `qr(A)` | `sgeqrt_`, `dgeqrt_`, `cgeqrt_`, `zgeqrt_` | `libopenblas64_.so` |
| `qr(A, ColumnNorm())` | `sgeqp3_`, `dgeqp3_`, `cgeqp3_`, `zgeqp3_` | `libopenblas64_.so` |
| `svd(A)` | `sgesdd_`, `dgesdd_`, `cgesdd_`, `zgesdd_` | `libcolmat_blas.so` |
| `svd(A; alg=LinearAlgebra.QRIteration())` | `sgesvd_`, `dgesvd_`, `cgesvd_`, `zgesvd_` | `libcolmat_blas.so` |
| `svdvals(A)` | `sgesdd_`, `dgesdd_`, `cgesdd_`, `zgesdd_` | `libcolmat_blas.so` |
| `eigen(A)` | `sgeevx_`, `dgeevx_`, `cgeevx_`, `zgeevx_` | `libcolmat_blas.so` |
| `eigen(Hermitian(A))` | `ssyevr_`, `dsyevr_`, `cheevr_`, `zheevr_` | `libcolmat_blas.so` |
| `cholesky(Hermitian(A))` | `spotrf_`, `dpotrf_`, `cpotrf_`, `zpotrf_` | `libcolmat_blas.so` |

## 判定の範囲

- 「成功」は、このプローブの小さな有限値入力で参照計算または分解の残差を満たしたことを表します。
- 行列積・求解・分解再構成の残差は、BLAS を呼ばないスカラーの積和で確認します。
- 「既存バックエンド」での成功は、Rust colmat の該当分解が接続済みであるという意味ではありません。
- Rust 側の機能名は固定 revision の `faer/src/linalg/solvers.rs` で確認した対応候補です。未接続の Rust 関数をこのプローブが直接実行したとは主張しません。
- この表は全 BLAS / LAPACK の網羅検証ではありません。特異行列、非有限値、一般化固有値、他 OS の結果も含みません。
- LP64 の直接 ABI テストや負ストライド等の既存回帰テストとは別の、Julia 高水準 API の確認です。

## 再生成

リポジトリのルートで、ビルド後の新しい Julia プロセスから実行します。

```sh
julia --project=test test/support_matrix.jl --output=docs/src/reference/support-matrix.md
```

検証が失敗したときは表を上書きしません。呼び出しケースと検証方法は `test/support_matrix.jl` にあります。

## Householder ABI 接続（#264）

この追記は既存の Linux 実測表とは別の検証です。Julia 1.13.3 /
`x86_64-apple-darwin`、colmat-rs
`e1c83482f5a61449d30e989808c3ba5f751ec391` を使用しました。
`test/householder.jl` は `larf/larfx/larfy/larft/larfb/larfg/larfgp` の
4型・LP64/ILP64 の共有ライブラリ入口を具体的な C 署名で実行し、
28 canonical シンボルの LBT 登録先が Colmat であることを確認します。

生成は LAPACK の `I - tau*v*v'`、実数 beta、LARFGP の非負 beta に合わせた
Rust adapter です。colmat の実数係数・複素位相を保持する生成 API の単なる
逆数変換ではありません。適用は既存の native GEMV/SYMV/GEMM を使用します。
F/B、C/R の implicit unit diagonal を展開し、LARFT の三角因子を構成します。
LARFB は LAPACK 形式の外部因子も受け入れます。

検証は反射積・ユニタリ性・極小/極大尺度・複素 head・ゼロ係数・
長さ1・正負 stride・左右・上下三角・外部 OpenBLAS の LARFT 因子を含みます。
LARFX の小次数では WORK を参照せず、LARFY は選択した三角領域だけを更新します。
これらの補助ルーチンには workspace query や INFO 引数はありません。
不正引数の診断は既存の `colmat_blas_last_error` 規約を使います。

LARFB は内部で basis と次数の二乗の明示的な反射行列を確保します。
呼び出し側の最小合法 WORK で動作しますが、作業領域だけを使う実装や
LAPACK と同じ計算量・割り当て量を保証するものではありません。
この接続は Fortran ABI に限定し、LAPACKE は別タスクです。

## Compact QR 接続（#290）

Julia 1.13.3 / x86_64-apple-darwin と上記固定 colmat-rs revision で、
通常の `qr(A)`、`Matrix(F.Q)`、Q/Q' の左右適用を Colmat の
GEQRT/GEMQRT 入口に接続しました。既存 Linux snapshot の未接続という
記録は、その時点の結果として保持しています。

`test/qr_compact.jl` は4型、LP64/ILP64 の具体的な C 署名、NB=1/2/4、
GEQRT2/GEQRT3、外部 OpenBLAS の因子との双方向相互運用、leading dimension、
空行列、ゼロ列・従属列、極小尺度、不正引数を検証します。
これらの入口には LWORK query はありません。WORK は標準の最小長で実行します。

native QR は各 active column に適用し、実数の native 係数の逆数から
LAPACK の T を構成します。これにより、native の rank 処理がゼロ列を
詰める場合も LAPACK の対角位置を保ちます。複素 R の対角位相を保持し、
Q と R の積・外部 LAPACK による反射適用を検証します。
GEQRT3 も共通の数値経路を使い、Netlib の再帰算法とは異なります。
内部配列を確保し、GEMQRT は各ブロックを明示的に展開します。

## 通常 QR 接続（#289）

Julia 1.13.3 / x86_64-apple-darwin と同じ固定 colmat-rs revision で、
GEQRF/GEQR2、非負対角版、ORGQR/UNGQR/ORG2R/UNG2R、
ORMQR/UNMQR/ORM2R/UNM2R、GEQRS の36シンボルを両整数 ABI に接続しました。
`test/qr_standard.jl` は Julia ラッパー経由の再構成・Q の左右適用に加え、
両 ABI の直接実行、外部 OpenBLAS 因子の受け入れと native 因子の外部適用、
非負対角、workspace query と最小合法作業領域、padded leading dimension、
空・矩形・ゼロ列、GEQRS の最小二乗残差を検証します。

実数通常版は native QR の block-size-one 係数から A/TAU を作ります。
複素通常版と非負対角版は LAPACK gauge reflector 生成を使い、
実数 beta と複素 TAU を Julia の標準規約に合わせます。
Q の適用は native GEMV、GEQRS の三角 solve は native 三角 kernel に接続します。
内部配列を確保するため、呼び出し側の WORK 以外にもメモリを使用します。
GEQRS は deprecated LAPACK 契約に従い、workspace query を提供しません。

## 列ピボット QR 接続（#299）

Julia 1.13.3 / x86_64-apple-darwin と同じ固定 colmat-rs revision で、
GEQP3/GEQPF/GEQP3RK の12シンボルを4型・両整数 ABI に接続しました。
`qr(A, ColumnNorm())` は GEQP3 から native QR/reflector 適用へ到達します。
固定 JPVT 列は入力順を保って先頭へ移し、出力は元の列番号を1始まりで返します。

GEQP3RK は KMAX と、残差部分の最大列2ノルムに対する ABSTOL/RELTOL で停止します。
各段階で列ノルムを再計算するため、native ColPivQr の対角値による rank 判定や
Netlib の部分ノルム更新とは異なる計算量・丸め経路です。付随 NRHS 列は
ピボット選択から除外し、Q' を適用します。NaN/Inf の INFO、ゼロ・極小 tolerance、
rank と残差ノルムの返却を別々に扱います。内部配列を確保します。

`test/qr_pivoted.jl` で両 ABI の具体的な C 署名、固定列、外部 OpenBLAS 因子、
途中停止した分解の再構成、独立計算した残差ノルム、付随右辺、
query と最小合法 workspace、leading dimension、空行列・1×1・ゼロ行列、
NaN/Inf 診断、LBT 登録を検証します。空行列の Julia GEQP3 ラッパーは
native 入口を呼ばないため、直接 ABI の空行列ケースと区別します。

## LQ 接続（#291）

Julia 1.13.3 / x86_64-apple-darwin と同じ固定 colmat-rs revision で、
通常・ブロック・short-wide・ヘッダー形式の LQ、Q の生成・左右適用、
GELQS の56シンボルを4型・両整数 ABI に接続しました。
通常 LQ は随伴 QR の恒等式を使い、A/TAU と行保存のブロック T を構成します。
short-wide 形式は最初の NB 列の因子と、後続の [L B] パネルの因子を
LAPACK と同じ順序で格納します。外部の T とヘッダーを受け入れます。

GELQ は合法な MB=1・NB=N の形式を選ぶため、最適・最小 query の推奨値が
一致します。GELQT3 は共通の分解経路であり Netlib の再帰算法ではありません。
内部で随伴行列・反射・選択パネルなどを確保します。GELQS は deprecated
LAPACK 契約に従い workspace query を提供しません。

`test/lq.jl` は4型・両 ABI、通常 Julia ラッパー、最小合法 workspace、
最小ヘッダー query、ブロック端、外部 OpenBLAS 因子との双方向相互運用、
左右・転置・共役、leading dimension の padding、空・矩形・階数落ち、
GELQS の残差と最小ノルム性、56シンボルの LBT 登録を検証します。
