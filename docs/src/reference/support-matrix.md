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

## RZ 接続（#304）

Julia 1.13.3 / x86_64-apple-darwin と同じ固定 colmat-rs revision で、
TZRZF/TZRQF/LATRZ、ORMRZ/UNMRZ、LARZ/LATZM、LARZT/LARZB の
32シンボルを4型・両整数 ABI に接続しました。upper trapezoid の
`A = [R 0] Z` を後ろの行から構成し、疎な反射を native GEMV/GEMM で適用します。
LARZT/LARZB は LAPACK の backward・行保存を扱い、複素 T の共役も変換します。
単反射・ブロック反射とも作用しない中央領域を保持し、内部配列を確保します。

`test/rz.jl` は両 ABI の直接呼び出し、Julia ラッパー、外部 OpenBLAS の
因子・適用との相互運用、query と最小合法 workspace、padding、正負 stride、
分離された C1/C2、L=0 と暗黙成分に重なる尾部、中央領域を NaN にした入力、
直交性・再構成・共役適用と LBT 登録を検証します。

## 最小二乗 driver 接続（#310）

Julia 1.13.3 / x86_64-apple-darwin と同じ固定 colmat-rs revision で、
GELS/GELST/GETSLS、GELSS/GELSD、GELSY/GELSX の28シンボルを4型・両整数 ABI に接続しました。
full-rank 系は QR/LQ と三角 solve で tall/wide と転置・共役を処理し、
A/B を尺度調整して解・残差・因子の尺度を戻します。
GELSS/GELSD は共通 native SVD で RCOND と rank を判定します。
GELSS の A には行方向の右特異ベクトルを返します。GELSD の query は
WORK に加え IWORK/RWORK の必要長を返します。

GELSY/GELSX は固定 JPVT の列ピボット QR、leading R の条件判定、RZ、
Q/Z 適用、逆置換による完全直交分解です。条件判定には leading R の
特異値を使うため、Netlib の増分条件推定とは境界付近の rank が異なる可能性があります。
GELST/GETSLS は共通 full-rank 経路で、Netlib の compact-WY/TS 算法と同一ではありません。
GETSLS は最適・最小 query の値が一致する構成です。SVD の full U や中間行列などを
内部で確保し、GELSD という名称でも native divide-and-conquer の実行を保証しません。

`test/least_squares.jl` は両 ABI の具体的な C 署名、外部 OpenBLAS との比較、
固定列、rank-deficient と過少決定の最小ノルム解、特異値・右特異ベクトル、
TRANS/共役、残差配置、極小/極大尺度、padding、最小 workspace と query、
空・ゼロ入力、28シンボルの LBT 登録を検証します。

## Hessenberg 接続（#293）

Julia 1.13.3 / x86_64-apple-darwin と同じ固定 colmat-rs revision で、
GEHRD/GEHD2、ORGHR/UNGHR、ORMHR/UNMHR の16シンボルを4型・両整数 ABI に接続しました。
LAPACK gauge の reflector 生成と native GEMV による左右適用で A/TAU を構成し、
複素 subdiagonal を実数にします。GEHRD/GEHD2 は共通経路で、ブロック化の差はありません。
ILO/IHI の部分範囲に対応し、Q はその範囲外で恒等変換です。
内部で反射・Q 生成用の配列を確保します。

`test/hessenberg.jl` は Julia ラッパー、両 ABI の具体的な C 署名、
外部 OpenBLAS の因子と Q との相互運用、A=Q*H*Q'、直交性、active range 外の
保存、空・1×1、左右・転置・共役、padding、最小 workspace と query、
不正範囲と LBT 登録を検証します。

## Schur driver 接続（#300）

Julia 1.13.3 / x86_64-apple-darwin と同じ固定 colmat-rs revision で、
HSEQR/LAHQR/LAQR0/LAQR4/GEES の20シンボルを4型・両整数 ABI に接続しました。
Hessenberg の active block を公開 native Schur API へ渡し、範囲外との結合と
既存 Z を native GEMM で更新します。COMPZ=N/I/V、JOB=E/S、
補助入口の WANTT/WANTZ と ILOZ/IHIZ を扱います。JOB=E/WANTT=false でも
Schur 形式まで計算します。各入口は同じ native QR 経路を使い、Netlib の
LAHQR/LAQR0/LAQR4 の算法の切り替えを再現するものではありません。

GEES は Hessenberg 還元、native Schur、native block reorder を合成します。
尺度を正規化してから還元・反復・並べ替えを行い、SELECT には元の単位の
固有値を渡します。SELECT の Fortran LOGICAL は整数 ABI と同じ幅で扱い、
実数共役対は片方が選択されれば両方を選択します。SDIM と選択された
スペクトルの前方配置を返し、並べ替え拒否と再選択の不整合はそれぞれ
INFO=N+1/N+2 です。SORT=N と workspace query では callback を呼びません。
GEESX/TRSEN の条件推定はこの接続に含みません。

公開 native Schur API は非収束の部分 index を返しません。その場合は
active range 全体を未収束として INFO=IHI を返し、部分収束を偽って報告しません。
内部で行列・反射係数・native scratch を確保し、呼び出し元の最小合法 WORK で
実行できます。query はその必要長を返します。

`test/schur.jl` は4型・両 ABI の具体的な C 署名、Julia の通常ラッパー、
A=Z*T*Z'、直交性、active range、既存 Z の更新と行範囲、実 Schur 共役対、
SELECT/SDIM、極小・極大尺度、padding、未参照領域の NaN、空・1×1、
query、不正引数、非収束 INFO と20シンボルの LBT 登録を検証します。

## 一般化 Hessenberg 接続（#269）

Julia 1.13.3 / x86_64-apple-darwin と同じ固定 colmat-rs revision で、
GGHRD/GGHD3 の8シンボルを4型・両整数 ABI に接続しました。
公開 `generalized_hessenberg` が返す active block の左右基底を全体へ埋め込み、
範囲外との結合と既存 Q/Z を native GEMM で更新します。
COMPQ/COMPZ=N/I/V と ILO/IHI を扱い、B の厳密下三角は入力で読みません。
GGHRD/GGHD3 は同じ native 経路で、内部の行列・scratch を確保します。
GGHD3 は呼び出し元の WORK を1要素だけ必要とし、query でも1を返します。
Netlib GGHRD/GGHD3 の別々の算法や外部 WORK の利用方法を再現しません。

`test/generalized_hessenberg.jl` は両 ABI の具体的な C 署名、4型、
Q'*A*Z=H と Q'*B*Z=T、Hessenberg/三角構造、ユニタリ性、
空・1×1・2×2、部分範囲、Q/Z の初期化・既存値への更新・未使用、
40×40 のブロック経路、padding、未参照 B/Q/Z の NaN、query、
不正引数と8シンボルの LBT 登録を検証します。

## QZ driver 接続（#294）

Julia 1.13.3 / x86_64-apple-darwin と同じ固定 colmat-rs revision で、
HGEQZ/GGES/GGES3/GEGS の16シンボルを4型・両整数 ABI に接続しました。
HGEQZ は active block の native QZ と基底変換を合成し、
COMPQ/COMPZ=N/I/V と ILO/IHI を扱います。
JOB=E でも内部では Schur 形式まで計算します。
β の対角を非負の実数に揃え、Q/Z と pencil の変換を維持します。
α/β は homogeneous な組で返し、β=0 を有限の固有値へ置き換えません。

GGES/GGES3/GEGS は QR による B の三角化、一般化 Hessenberg 還元、native QZ を
合成します。A/B は独立に尺度を正規化し、α/β と S/T の単位を戻します。
GGES/GGES3 の SELECT は LP64/ILP64 の LOGICAL 幅で呼び、実数共役対の
片方が選択されれば両方を選択します。選択された block を native reorder で
前方へ移し、並べ替えた pencil の QZ を再評価して α/β を更新します。
query と SORT=N では callback を呼びません。並べ替え拒否は INFO=N+3、
再選択の不整合は INFO=N+2 です。GGESX/TGSEN の条件推定は対象外です。

公開 native QZ API は非収束状態・部分 index を返しません。
返却された Schur 構造と homogeneous な α/β の各成分の有限性を確認し、不完全な結果は
active range 全体を未収束として INFO=IHI とします。
内部行列と scratch を確保し、入口ごとに LAPACK の最小合法 WORK と query を
提供します。GGES/GGES3/GEGS は共通経路で、Netlib の各 driver の
ブロック化や permutation balancing の実装を再現するものではありません。

`test/qz.jl` は両 ABI の具体的な C 署名、4型、pencil の再構成、
Schur 構造、ユニタリ性、active range、Q/Z の初期化・更新・未使用、
40×40 の入力、homogeneous 値の独立 OpenBLAS 比較、β=0、実数共役対、
SELECT/SDIM、独立した極小・極大尺度、padding、未参照領域の NaN、
空・1×1・2×2、query と最小 WORK、不正引数、非収束 INFO、
Julia GGES ラッパーと16シンボルの LBT 登録を検証します。

## 一般化 EVD driver 接続（#301）

Julia 1.13.3 / x86_64-apple-darwin と同じ固定 colmat-rs revision で、
GGEV/GGEV3/GEGV/GGEVX の16シンボルを4型・両整数 ABI に接続しました。
通常 driver は公開 `gevd_real/cplx` の homogeneous α/β と左右固有ベクトルを
変換します。β=0 と 0/0 の特異 pencil を保持し、実数共役対を隣接する実部・虚部列で
返します。native の実数 transpose-left 規約は虚部列を反転して LAPACK の
adjoint-left 規約へ変換し、各ベクトルの最大の |Re|+|Im| を1に正規化します。
各入口は同じ native 経路と内部 scratch を使い、入口ごとの最小 WORK/query を返します。
native GEVD が非収束を返す場合は、部分 index がないため INFO=N とします。

GGEVX の `BALANC=N/SENSE=N` は native です。ILO/IHI=1/N、LSCALE/RSCALE=1 と
元の行列の ABNRM/BBNRM を返します。A/B の Schur 出力は、固定 revision の native
実装の2冪 prescale を追跡して元の単位へ戻します。依存更新時にはこの変換を
再検証する必要があります。

その他の BALANC/SENSE は既存 ILP64 expert backend へ委譲します。
LP64 の呼び出しでは整数スカラーを64 bitへ変換し、IWORK/BWORK は呼び出し内で
確保するため、32 bit の出力領域へ64 bitで書き込みません。ILO/IHI/INFO を元の幅へ戻し、
query 時には ILO/IHI を変更しません。balance と条件推定を native 対応済みとは扱いません。
Colmat のロード前に4型の ILP64 GGEVX が必要です。通常の Julia は OpenBLAS を提供します。
不足時は LBT 登録の変更前に明示的な設定エラーを返し、合法なモードを不正引数として
報告したり、未計算の条件推定を成功扱いにしたりしません。
再登録では Colmat 自身を expert backend として保存しません。

`test/generalized_eigen.jl` は4型・両 ABI・全4入口、左右・片側・values-only、
homogeneous な左右残差、β=0、0/0、実数共役対と正規化、独立 OpenBLAS 比較、
GGEVX の全 BALANC/SENSE 組合せ、metadata、query、整数幅の guard、padding、
元の尺度の Schur 出力、空・1×1、不十分な workspace と LBT 登録を検証します。
別プロセスで expert backend 欠落時の登録前診断と再登録も確認します。

## ピボット付き Cholesky 接続（#265）

Julia 1.13.3 / x86_64-apple-darwin と同じ固定 colmat-rs revision で、
PSTRF/PSTF2 の8シンボルを4型・両整数 ABI に接続しました。
adapter が最大対角 pivot と TOL による停止を制御し、各 pivot の native LLT と
native GEMM による Schur complement 更新を合成します。公開の一括 pivoted LLT は
固定の内部許容値を使うため、この構成で任意の TOL と TOL=0 を扱います。

TOL<0 は N*DLAMCH(E)*最大対角を使い、最初の正 pivot を取った後の候補を
TOL と比較します。RANK<n は INFO=1、空入力は RANK=0/INFO=0 です。
PIV は置換後の各位置にある元の行番号を1始まりで返します。
初期対角が非正なら因子化せず RANK=0/INFO=1 とします。計算途中で停止した場合は、
完了した因子、未計算の置換済み三角部分、停止対角を返します。
UPLO=U では下三角の内部因子を共役転置して出力します。
反対側の三角と padding は変更せず、複素対角の虚部も入力では読みません。

PSTRF/PSTF2 は共通の逐次 pivot 経路で、内部の密行列と native scratch を確保します。
呼び出し元 WORK の内容には依存しません。これらの LAPACK 入口には workspace query が
ありません。許容値の近傍では丸めによって pivot や rank が変化し得るため、
浮動小数点 Gram 行列の代数的 rank と数値的 rank を区別します。

`test/pivoted_cholesky.jl` は4型・両 ABI・UPLO の直接署名、PIV と rank、
PSD/full-rank/厳密な rank-deficient/ゼロ・負の対角、部分因子の再構成、
OpenBLAS の因子・停止状態との比較、TOL の等号境界と自動許容値、
大きな動的範囲での TOL=0、空・1×1、padding と未参照三角、
複素対角の虚部、整数 guard、不正引数、Julia の通常 API と LBT 登録を検証します。

## Bunch–Kaufman 接続（#266）

Julia 1.13.3 / x86_64-apple-darwin と同じ固定 colmat-rs revision で、
実数 SY / 複素 Hermitian HE の TRF/TF2、TRS/TRS2、TRI/TRI2/TRI2X、SV の
32シンボルを両整数 ABI に接続しました。複素対称 SY 系は既存バックエンドに残します。
native `cholesky::lblt` の Partial pivoting と分離された対角・副対角・最終置換を使い、
LAPACK の classic IPIV と、途中の置換を含む A の保存形式へ変換します。
native L の後続の行交換を戻して列を保存し、外部の LAPACK A/IPIV からも
逆変換して native solve/inverse を呼び出します。
UPLO=U は行・列の順序を反転した基底で処理します。

TRF は native の既定 panel、TF2 は native の非ブロック経路を使用します。
TRS/TRS2 は共通 solve、TRI/TRI2/TRI2X は共通 inverse です。
内部で行列と native scratch を確保します。TRF/SV の query は1、
共通 inverse 経路の TRI2 query は max(1,N) を返します。
TRI2X の NB は blocking hint として受け付け、同じ inverse を使います。
反対側の三角、padding、複素対角の虚部は入力では読みません。
出力も指定された三角だけを更新します。

factorization/inverse の1×1 のゼロ D は LAPACK の位置で正の INFO を返し、
SV はその場合に solve を実行しません。
対角がゼロでも非特異な2×2 block を誤って特異と判定しません。
入力 factor の形式と INFO は LAPACK に従い、native の最終置換を
そのまま IPIV として返す扱いにはしません。

`test/bunch_kaufman.jl` は4型・両 ABI・UPLO、1×1/2×2、80×80 の panel 経路、
PAP' と LDL'/UDU' の再構成、solve/inverse residual、native と OpenBLAS の
A/IPIV の双方向相互運用、各 inverse と driver、特異 INFO と B の保存、query、
最小 WORK、不十分な workspace、不正引数、padding・整数 guard・未参照領域、
極小・極大尺度、Julia の通常 API と32シンボルの LBT 登録を検証します。
複素対称問題が HE へ誤接続されないことも回帰確認します。


## GEBAL/GEBAK の接続（#267）

4型の GEBAL/GEBAK、8 Fortran シンボルを LP64/ILP64 に接続しました。
GEBAL は native `balance_in_place_with`、GEBAK は
`balance_eigenvectors_in_place` を使います。JOB=N/P/S/B と SIDE=L/R を扱い、
最終置換を LAPACK の ILO/IHI/SCALE の swap encoding に変換します。
全固有値が孤立する場合も、非空行列には LAPACK の1要素 active range を返します。
この encoding と範囲の処理は既存 GEEVX と共有します。

GEBAK は更新対象の行だけを読み書きし、選択した JOB が使わない SCALE、
未参照の行、leading dimension の padding を読みません。
内部では行列・置換・尺度と native scratch を確保し、固有ベクトルを再正規化しません。

`test/balance.jl` は4型・両 ABI、相似変換の再構成、左右の逆変換、
孤立固有値・非自明な置換・尺度、OpenBLAS との metadata の双方向相互運用、
空・1要素・矩形の固有ベクトル、未参照領域、不正引数・整数幅、
Julia の LAPACK wrapper と LBT 登録を検証します。


## 三重対角 EVD driver の接続（#268）

STEQR/STEDC/PTEQR/STEGR/STEMR の4型、STERF/STEV/STEVD/STEVR/STEVX の実数2型、
計30 Fortran シンボルを LP64/ILP64 に接続しました。
通常の経路は native `tridiagonal_self_adjoint_evd` を呼び出します。
STEQR と STERF は QR 経路を選び、STEDC/STEVD は native の既定閾値128で
QR と divide-and-conquer を使い分けます。COMPZ=V は既存 Z に固有ベクトルを後乗算します。
PTEQR は正定値三重対角の bidiagonal Cholesky 因子を作り、
native `bidiagonal_svd` の QR 経路を使って小さな正固有値の相対精度を保ちます。
PTEQR の固有値は降順、他の driver は昇順です。

範囲選択 driver の通常経路は、全スペクトルから A/I/V を選択します。
専用 MRRR や bisection/inverse iteration の実装を提供するという意味ではありません。
STEMR の TRYRAC=true、STEVR/STEVX の正の ABSTOL は、保存した既存 LAPACK に委譲します。
STEGR の ABSTOL は Reference LAPACK と同じく使いません。
ISUPPZ は得られた固有ベクトルの support から作り、STEVR の部分範囲では更新しません。
STEMR の NZC=-1 query は必要な列数を Z(1,1) に返します。

公開 native API は非収束時の途中の D/E を返さないため、native が失敗した場合も
元の入力から保存した LAPACK を呼び、各 driver の INFO と部分結果の契約を保ちます。
このためロード前に24入口の ILP64 LAPACK バックエンドが必要です。通常の Julia は
OpenBLAS を提供します。LP64 呼び出しには整数引数・出力と整数 scratch を変換し、
再登録時には自分自身を転送先として保存しません。

内部で行列・固有ベクトル・native scratch を確保します。
STEVR/STEMR/STEGR の選択範囲が狭くても全スペクトルを計算するため、
専用の範囲選択実装と同じ計算量・割り当て量にはなりません。
STEDC の caller workspace は小問題の QR 閾値25と LAPACK の D&C の最小量を使い、
native scratch は別に確保します。query は入力行列と固有ベクトルを変更しません。
ただし STEMR の value-range の列数 query は D/E を読みます。

`test/tridiagonal_evd.jl` は4型・両 ABI、0/1/2次元、重複・split・区間端点、
既存 Z の後乗算、support、各 query と最小 workspace、guard と整数幅、
160次元の D&C、expert fallback、通常 Julia API と30シンボルの登録を検証します。
隔離プロセスでは fallback 不足の診断と通常経路が fallback を使わないことを確認します。
STEBZ/STEIN はこの接続に含めません。


## 三角行列の LAPACK solve/inverse/product（#270）

TRTRS/TRTRI/TRTI2/LAUUM/LAUU2 の4型、計20 Fortran シンボルを両整数 ABI に接続しました。
TRTRS は借用した A と B に native triangular solve を適用し、UPLO/TRANS/DIAG を変換します。
TRTRI/TRTI2 は native triangular inverse、LAUUM/LAUU2 は native triangular matmul を使います。
TRTI2 は TRTRI と共通の inverse kernel、LAUU2 は LAUUM と共通の product kernel です。
名前ごとの独立した blocked/unblocked 算法を提供するという意味ではありません。

A の反対側の三角と padding は入力で参照せず、出力でも保持します。
DIAG=U は実際の対角を読まず、inverse 出力でもその対角を変更しません。
TRTRS/TRTRI は最初のゼロ対角を正の INFO で返し、B/A を変更しません。
TRTRS は NRHS=0 でもこの特異対角検査を行います。
TRTI2 は Reference LAPACK と同じく非特異性を仮定し、正の INFO を返しません。

LAUUM/LAUU2 は上三角なら U Uᴴ、下三角なら Lᴴ L を計算します。
複素対角も含めて数学的な三角積を計算し、出力の対角の虚部はゼロにします。
solve は入力と出力を借用します。inverse/product は借用した入力から内部の
n×n 出力行列を作って指定三角だけを書き戻し、入力全体のコピーは作りません。
これらの入口に caller workspace や workspace query はありません。

`test/triangular_lapack.jl` は4型・両 ABI・UPLO/TRANS/DIAG、空・1要素・80次元、
複数/ゼロ右辺、unit diagonal と反対三角の NaN、padding、外部 inverse との相互運用、
左右の inverse residual、三角積、特異 INFO、不正引数・整数幅、通常 Julia API と
20シンボルの LBT 登録を検証します。既存 BLAS TRSV/TRSM の経路はこの接続の対象外です。
