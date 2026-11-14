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
