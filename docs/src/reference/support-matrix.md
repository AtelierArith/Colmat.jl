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


## 一般化自己随伴問題の標準形変換（#275）

実数 SYGST/SYGS2 と複素 HEGST/HEGS2 の8 Fortran シンボルを両整数 ABI に接続しました。
入力 B は既に Cholesky 因子化されている必要があり、この入口では因子化しません。
UPLO=L の L、UPLO=U の Uᴴ を共通の下三角因子として扱い、
ITYPE=1 は L⁻¹ A L⁻ᴴ、ITYPE=2/3 は Lᴴ A L を構成します。
前者は native triangular solve、後者は native triangular matmul を使います。
SYGS2/HEGS2 と SYGST/HEGST は共通の kernel を使い、名前ごとの
独立した blocked/unblocked 算法を提供するという意味ではありません。

A/B の指定された三角だけを入力として読みます。A は対称/Hermitian 行列へ展開し、
複素対角の実部だけを読みます。B の実数因子は借用し、複素因子は実対角を持つ
下三角表現へ変換するため内部行列を確保します。ITYPE=2/3 は積のための
一時行列も確保します。B は変更せず、A の反対側の三角と padding も保持します。
これらの入口に workspace 引数や query はありません。
B の正定値性を再検査せず、LAPACK と同じく INFO は不正引数だけを報告します。

`test/generalized_reduction.jl` は4型・両 ABI・ITYPE・UPLO、0/1/3/40/80次元、
非単位対角因子、参照 LAPACK の両算法との比較、元の3種類の一般化問題の残差と
B/B⁻¹ による正規化、複素対角虚部と反対三角の NaN、未参照領域の保持、
不正引数・整数幅、LBT 経由の通常配列、外部 Cholesky 因子と広い尺度を検証します。
標準形変換の共通処理は一般化自己随伴 EVD driver でも利用できます。


## 一般化自己随伴正定値 EVD driver（#292）

実数 SYGV/SYGVD/SYGVX と複素 HEGV/HEGVD/HEGVX の12 Fortran シンボルを
両整数 ABI に接続しました。native LLT、#275 の標準形変換、native self-adjoint EVD、
三角 solve/matmul による固有ベクトルの逆変換を合成します。
ITYPE=1 は A X = B X Λ、2 は A B X = X Λ、3 は B A X = X Λ を扱い、
1/2 は Xᴴ B X = I、3 は Xᴴ B⁻¹ X = I に正規化します。
UPLO の両方と JOBZ=N/V を扱います。B は指定された三角に Cholesky 因子を返します。
正定値性の失敗は INFO=N+k とし、A と固有値・固有ベクトルの出力を更新しません。

GV/GVD は共通の native EVD を使い、既定の閾値128で QR と D&C を選びます。
legacy driver 名ごとの独立した QR/D&C 実装を提供するという意味ではありません。
GVX は全スペクトルから A/I/V を選択し、正の ABSTOL は保存した ILP64 LAPACK に委譲します。
専用 bisection/inverse iteration による選択計算の実装を提供するという意味ではありません。
通常の計算では固有値が昇順となり、GV/GVD は A、GVX は Z に固有ベクトルを返します。
IFAIL は native 成功時に最初の M 要素をゼロにします。

複素 native LLT は反対三角を作業用に使うため、内部行列で因子化し、
指定三角だけを B へ書き戻します。複素対角の虚部は入力で読みません。
実数 B の因子化は借用バッファで実行します。A の標準形、固有ベクトル、
native scratch と、native 失敗時の再実行に備えた B の指定三角の控えを内部で確保します。
native が途中の固有値・固有ベクトルを返さず失敗した場合、B を復元して
保存した LAPACK を呼び、各 driver の INFO と途中結果を維持します。
このためロード前に12入口の ILP64 バックエンドが必要です。通常の Julia は OpenBLAS を提供します。
LP64 の転送は整数引数・整数出力を変換し、整数 scratch を内部で確保します。
再登録時には自分自身を fallback として保存しません。

query は入力と値・ベクトルの出力を変更せず、この実装で受け付ける最小 WORK を返します。
GVD の複数 workspace query と、各 caller workspace の最小量・不十分な量を検証します。
caller workspace だけで全処理が完結するという意味ではありません。
A/B の反対三角と padding は入力で参照しません。B の反対三角は出力でも保持します。
JOBZ=V の GV/GVD は A 全体が固有ベクトル出力となり、GVX の Z は最初の M 列だけを更新します。

`test/generalized_selfadjoint.jl` は4型・両 ABI・ITYPE・UPLO・JOBZ、正規化と各残差、
参照 LAPACK、範囲端点と重複値、空・1要素・160次元 D&C、query と最小 work、
正定値性 INFO、不正引数・整数幅、未参照領域、精度指定と native 失敗の fallback、
通常 Julia の `eigen(Hermitian(A), Hermitian(B))` と12シンボルの登録を検証します。
隔離プロセスでは backend 不足を LBT 登録前に検出し、通常経路の native 実行を確認します。


## 完全ピボット LU と scaled solve（#272）

GETC2/GESC2 の4型、計8 Fortran シンボルを両整数 ABI に接続しました。
GETC2 は SMIN を abs_tol に指定して native full-pivot LU を実行します。
その最終置換を LAPACK の逐次 IPIV/JPIV に変換し、残る Schur complement には
完全ピボット選択・小ピボット置換を行い、native triangular solve と matmul で消去を継続します。
因子化を最後まで実行してから対角だけを置換する方式ではありません。

SMIN は max(eps × 最初の最大絶対値, floatmin/eps)、1要素の場合は floatmin/eps です。
小ピボットは正の実数 SMIN に置き換え、INFO は最後に摂動した位置を返します。
native prefix と参照 LAPACK は、同値候補の選び方や複素数の pivot score が異なるため、
IPIV/JPIV の値そのものの一致は保証しません。因子形式と再構成・solve の契約を保ちます。
Bignum の定数と最終 SMIN は対応する実数精度です。

GESC2 は行置換、unit lower solve、upper solve、逆順の列置換を行い、
元の行列に対する A x = SCALE b を満たす scaled solution を返します。
前進代入の更新前と後退代入の除算・積の前で右辺全体を縮小して overflow を防ぎます。
後退代入では U の行を対角で正規化してから native の積を使い、
未正規化の列との積だけが巨大になる不要な SCALE の縮小を避けます。
SCALE は実数精度で毎回更新し、積がゼロへ underflow した場合は右辺もゼロにします。
SCALE は参照 LAPACK と数値的に同一とは限らず、0≤SCALE≤1 と残差を検証します。

A は借用して因子化し、GESC2 は入力因子を変更しません。
内部で native scratch・行列サイズに比例する置換配列・正規化した U 行を確保し、
入力行列全体の控えは作りません。caller workspace と query はありません。
N=0 は配列に触れずに戻り、GESC2 の SCALE は1です。
アダプタは不正な N/LDA を検査し、GETC2 は負の INFO、
INFO 引数を持たない GESC2 は `colmat_blas_last_error` で引数番号を報告します。
GESC2 は最後の IPIV/JPIV 要素を参照しません。

`test/complete_pivot_lu.jl` は4型・両 ABI、行/列置換の再構成、
非特異・特異・階数落ち・subnormal・SMIN 境界、外部因子と solve の双方向相互運用、
極小/極大尺度、巨大右辺、112次元の非ゼロ解と300次元の不可避な SCALE underflow、
BigFloat による残差/解の比較、padding・論理部分行列・整数 guard、
未参照 pivot、不正引数と LBT 経由の通常配列を検証します。

## 選択 SVD と QR 前処理 SVD（#273）

GESVDX/GESVDQ の4型、計8 Fortran シンボルを両整数 ABI に接続しました。
GESVDX は native SVD から降順の特異値と U/VT を生成し、RANGE=A/I/V を
処理します。値区間は `(VL,VU]` で、計算した特異値をそのまま比較します。
区間の端の数学的特異値と丸め後の特異値は一致しない場合があります。
NS 個の出力だけを書き、VT の複素共役と選択列数に応じた LDVT を扱います。

GESVDQ は列ピボット QR、rank 判定、rank 個の R の行に対する native SVD、
Q 適用と置換の逆変換を組み合わせます。JOBA=A/M/H、JOBP=N/P、JOBR=N/T、
JOBU=A/S/U/R/F/N、JOBV=A/V/R/N を扱い、JOBU=R と JOBV=A/V の組合せは
参照実装と同じく不正引数です。JOBU=F では A の Householder 表現と
WORK の先頭 N 個の tau を保持し、Q 適用前の左ベクトルを返します。
IWORK の先頭 N 個は列置換で、行ピボット時は続く M-1 個が逐次行交換です。
JOBA=E の条件推定は保存した ILP64 driver に委譲します。

GEJSV/GESVJ は独立した Jacobi 算法が必要なため、今回のアダプタでは
export せず既存バックエンドに残します。通常の SVD で代用しません。
GESVDX/GESVDQ は native 計算が失敗した場合も、変更前の入力から保存した
ILP64 driver に委譲します。LP64 入口では整数配列を変換します。
ロード前に既存 ILP64 の8 driver が必要で、再登録時の自己参照を防ぎます。

workspace query と最小合法 workspace を提供しますが、caller workspace
だけで計算するという保証はありません。native SVD と QR の一時領域を
内部確保し、GESVDQ は失敗時の再実行に備えて QR 入力を一時領域に保存します。
入力 A の余分な leading-dimension 領域と未使用のベクトル出力は保持します。

`test/advanced_svd.jl` は4型・両 ABI の矩形残差、特異値の重複・ゼロ・
区間端、query・最小 workspace、整数幅、rank 判定、factored Q の適用、
JOBA=E の参照バックエンドとの一致を検証します。

## SYEVX/HEEVX の範囲選択（#274）

実数 SYEVX と複素数 HEEVX の4シンボルを LP64/ILP64 に接続しました。
非正の ABSTOL では native 自己随伴 EVD の全スペクトルから RANGE=A/I/V
を選択し、昇順の W と対応する Z を返します。値区間は `(VL,VU]` です。
Sturm count、MRRR、inverse iteration の独立算法を提供するという意味では
ありません。正の ABSTOL と native 計算失敗は、ロード前に保存した ILP64
参照 driver に委譲し、LP64 の整数 workspace と出力を変換します。

UPLO の指定部分だけを一時行列へ読み、複素対角の虚部は読みません。
native 経路では入力 A を保持し、caller の leading-dimension の余白と
未使用の出力は保持します。参照 driver への委譲時は LAPACK が許す範囲で
A の指定三角部分を上書きします。成功時、JOBZ=V の先頭 M 個の IFAIL は
0 です。N=1 の Z(1,1) は範囲に入らない場合も参照実装と同じく1を書きます。
query は M、行列、固有値・固有ベクトルを変更しません。

最小 LWORK は N<=1 で1、他は実数8N・複素数2Nです。query はこの合法な
最小値を推奨値として返します。native の全固有値計算には別途内部確保を
使用するため、caller workspace が全一時メモリを制限するものではありません。

`test/selected_selfadjoint.jl` で4型・両 ABI、UPLO、A/I/V、重複・ゼロ・
区間端、workspace・INFO・IFAIL、不正引数、空配列、整数幅、極小/極大
スケール、複素対角の無視領域、LBT 登録と数値呼び出しを検証します。

## dense BLAS 拡張と rank-2 update（#276）

GEMMT/GEMMTR と SYR2 の4型、複素 HER2 と GEMM3M の2型、計16 Fortran
シンボルを両整数 ABI に接続しました。GEMMT/GEMMTR は native の三角出力
matmul を使い、指定三角部分だけを更新します。全行列の積を確保してから
切り出す方法ではなく、反対側の三角部分と leading-dimension の余白を保持します。
N/T/C を独立に扱い、beta=0 では出力の古い値を読みません。
alpha=0 または K=0 では A/B を読まず、指定三角部分の beta 倍だけを行います。

SYR2/HER2 は native の rank-1 三角積を2回適用し、正負のベクトル stride
を直接借用します。SYR2 は複素数でも転置積であり、HER2 だけが共役を使います。
非ゼロ alpha の HER2 は古い対角の実部だけを読み、出力の虚部を0にします。
alpha=0 のときは入力ベクトルも A も読み書きせず、対角も保持します。
不正引数は既存 BLAS と同じ `colmat_blas_last_error()` に引数位置を記録します。

GEMM3M は通常の native GEMM に接続して同じ積和の算術契約を提供します。
3回の実数乗算から構成する固有算法、他実装との演算順序や bit 一致を提供する
という意味ではありません。CBLAS 共通入口の接続は別 Issue #271 です。

`test/dense_extensions.jl` は4型・両 ABI、独立したスカラー積和、N/T/C、
複素係数、空/矩形、正負 stride、alpha/beta=0、無視領域の NaN、実対角、
引数診断、整数幅、LBT 登録先と実呼び出しを検証します。Rust の null pointer
テストで、不要なオペランドや alpha=0/beta=1 の C を参照しない経路も確認します。

## BLAS の値還元と IAMIN（#277）

実数 AMAX/AMIN/MAX/MIN/SUM の2型と IAMIN の4型、計14 Fortran シンボルを
両整数 ABI に接続しました。MAX/MIN/SUM は正 stride のベクトルを native
`MatRef::max/min/sum` へ借用します。絶対値の極値と IAMIN は実数比較値を
一時配列に作り、native min/max で還元します。IAMIN は最小値の最初の位置を
探し、物理オフセットではなく Fortran の1-based 論理 index を返します。

複素 IAMIN は modulus ではなく `abs(real(x))+abs(imag(x))` を比較します。
加算結果を元の実数精度へ丸めてから比較するため、ComplexF32 で丸めにより
同率になる場合も先頭を選びます。空入力は値・index とも0です。
正の N に対する stride の規約は次のとおりです。

| INCX | MAX/MIN/AMAX/AMIN | SUM | IAMIN |
| :--- | :--- | :--- | :--- |
| 正 | native 還元 | native 和 | abs1 の最初の最小 index |
| 0 | 先頭の値／絶対値 | 0、X を読まない | 0、X を読まない |
| 負 | 0、X を読まない | 0、X を読まない | 0、X を読まない |

負 stride は [OpenBLAS 0.3.30 汎用 AMIN kernel](https://github.com/OpenMathLib/OpenBLAS/blob/v0.3.30/kernel/arm/amin.c)
などと同じ非参照の規約に固定します。一部の最適化 backend で負 stride が
先頭より前を読む動作は再現しません。外部 backend との差分検証は正 stride と
定義済みの zero-stride 動作を対象とし、負 stride は独立期待値と null pointer
テストで検証します。SUM の演算順序や signed zero、NaN payload の bit 一致を
他 backend と保証するものではありません。

`test/reduction_extensions.jl` は両実数精度・該当する複素型、両 ABI、stride、
同率・abs1 と modulus の差・Float32 の丸め、空入力、非有限の scalar、
巨大配列を確保しない整数幅検査、LBT 登録と実呼び出しを検証します。

## Givens 回転の生成と配列適用（#278）

ROTG/LARTG/LARGV/LARTV/LAR2V/LASR の4型と、実数 LARTGP/LARTGS の2型、
計28 Fortran シンボルを両整数 ABI に接続しました。native JacobiRotation の
sine の符号・共役を LAPACK の規約に変換します。real ROTG/LARGV は大きい
入力の符号、LARTG は非ゼロ F の符号、LARTGP は非負 R を採用します。
real ROTG は A に R、B に補助値 Z を返し、complex ROTG は B を保持します。

real 生成系は入力を正規化してから native ROTG を呼ぶため、R 自体が型の
範囲を超える場合にも有限で単位長の係数を保持します。LARTGS はシフト列を
構成して LARTGP の順序を変えた回転を生成します。中間積が overflow する
場合は指数を分離した積・商で比を求め、表現可能な小さい sine を失わない
ようにします。小さい X とゼロシフトに関する LAPACK の分岐も維持します。

LARGV は正 stride の X/Y から配列係数と R を生成します。LARTV は個別の
係数を native の2成分適用へ、LAR2V は Hermitian 2x2 行列の両側適用へ
変換します。complex LAR2V は X/Y の実部だけを読み、実対角を出力します。
LASR は native の行/列 view による左右適用を使い、V/T/B pivot と F/B 順序を
扱います。恒等回転は A を参照せず、leading-dimension の余白も保持します。

配列係数ルーチンの増分は正値が利用条件です。正の N に対する不正な増分と
LASR の不正引数は `colmat_blas_last_error()` に引数位置を記録します。
配列係数ルーチンの N<=0 は係数・配列・増分を参照せず戻ります。
workspace は不要で、入力の借用は呼び出し中に限定します。
modified Givens の ROTM/ROTMG を提供するという意味ではありません。

`test/givens.jl` は両 ABI・4型、BigFloat の独立した生成式、極小/極大値・
ゼロ・複素位相、R の overflow と subnormal の出力量子化、配列係数・stride、
両側 Hermitian 適用、LASR の左右/pivot/順序、guard・引数診断・整数幅、
全28シンボルの LBT 数値呼び出しを検証します。通常領域では既存 LAPACK とも
照合します。既存 OpenBLAS の complex ROTG に単位長条件を満たさない極端な
入力があるため、その領域の正否判定は BigFloat と残差に基づきます。
Rust テストでは空/恒等操作の null pointer と、小さい sine の保持も検証します。

## 通常・構造付き格納の行列ノルム（#279）

LANGE/LANGB/LANGT/LANHS、対称・Hermitian の dense/band/packed/RFP/三重対角、
三角・台形・三角 band/packed の計54 Fortran シンボルを両整数 ABI に接続しました。
M、1/O、I、F/E と小文字の同義指定を扱います。1 ノルムは最大列和、I は最大行和
であり、行列全体の elementwise l1 ではありません。複素数の各要素には modulus
を使います。native の複素 norm_max（成分最大）・norm_l1（abs1）をそのまま
LAPACK の定義と同一視せず、実数絶対値への変換・native 和・最大値を合成します。

F/E は複素成分を保ち、最大成分で正規化した値を native norm_l2 で還元して
から戻します。pinned native の固定された3段階の二乗和では、最小 subnormal
や MAX 近傍で表現可能なノルムが0/∞になるため、この正規化を行います。
[単独再現 crate](https://github.com/AtelierArith/Colmat.jl/tree/main/reproducers/native-norm-l2)
は依存の git revision と Cargo.lock を保持し、native の8件の失敗を exit 101 で
示す診断です。通常の package test が失敗しているという意味ではありません。
関連する [native 報告 #36](https://github.com/tensor4all/t4a-faer/issues/36) への
追記は archived/read-only により拒否され、未公開草稿を再現 crate に残しています。

参照する格納要素だけを走査し、unit diagonal は1を合成して格納値を読みません。
Hermitian の対角は実部だけを読み、対称形式の複素対角は両成分を扱います。
M/1/I の complex `Inf+NaN*i` は modulus の∞、F/E は成分ごとの NaN を返す
LAPACK の規約に合わせます。別要素の NaN は各ノルムへ伝播します。
RFP は偶数/奇数、UPLO、実数 N/T と複素数 N/C の対応を変換します。

入力配列と WORK は保持します。非空入力の内部一時領域は格納要素数に比例し、
band/三重対角を dense 行列へ展開しません。1/I の group 管理には行/列数に比例
する領域も使います。caller WORK が全一時メモリを制限するという保証はなく、
LWORK query を持たない入口です。空の m または n は配列を読まず0を返します。
不正引数は0を返し、`colmat_blas_last_error()` に引数位置を記録します。
LAPACK の INFO 引数を新設するものではありません。

`test/matrix_norms.jl` は4型・両 ABI、独立 scalar/BigFloat norm、全格納形式、
無視領域の NaN、workspace guard、unit/Hermitian 対角、極小/極大、NaN/Inf、
引数診断・整数幅、全54入口の LBT 数値呼び出しを検証します。RFP は既存
TRTTF が独立に pack したデータと照合します。Rust は空/暗黙対角の null pointer、
RFP 全要素の重複なしの配置、複素 modulus、Frobenius の値域も検証します。
