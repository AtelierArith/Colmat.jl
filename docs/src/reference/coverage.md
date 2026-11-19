# BLAS 対応範囲

通常の Julia API を実行した結果と登録先の照合は
[Julia API の実測対応表](@ref)に記載します。

## 型と整数 ABI

| Julia の要素型 | BLAS 接頭辞 |
|---|---|
| `Float32` | `s` |
| `Float64` | `d` |
| `ComplexF32` | `c` |
| `ComplexF64` | `z` |

同じ共有ライブラリが LP64（32 bit 整数）と ILP64（64 bit 整数）を提供します。
LP64 の Fortran シンボルは `dgemm_`、ILP64 は `dgemm_64_` のように公開します。
LBT が Julia 側の整数 ABI に対応する入口へ転送します。
配列の要素精度と整数引数の幅は独立したものです。

## 提供ルーチン

`deps/rust/src/abi.rs` の実装を基準にした一覧です。
「全4型」は上表の4型を意味します。

| Level | 系列 | 対象 |
|---|---|---|
| 1 | `copy`, `swap`, `scal`, `axpy`, `axpby`, `iamax`, `rot`, `rotg` | 全4型 |
| 1 | `dot` | 実数2型 |
| 1 | `dotu`, `dotc` | 複素数2型 |
| 1 | `nrm2`, `asum` | 全4型、戻り値は対応する実数型 |
| 1 | `csscal`, `zdscal`, `csrot`, `zdrot` | 複素配列と実数係数 |
| 1 | `amax`, `amin`, `max`, `min`, `sum` | 実数2型 |
| 1 | `iamin` | 全4型、複素数は abs1、Fortran 1-based index |
| 2 | `gemv`, `symv`, `syr`, `syr2`, `trmv`, `trsv` | 全4型 |
| 2 | `ger` | 実数2型 |
| 2 | `geru`, `gerc`, `hemv`, `her`, `her2` | 複素数2型 |
| 3 | `gemm`, `gemmt`, `gemmtr`, `symm`, `syrk`, `syr2k`, `trmm`, `trsm` | 全4型 |
| 3 | `gemm3m`, `hemm`, `herk`, `her2k` | 複素数2型 |

`iamax` の実際のシンボル名は `isamax_` / `idamax_` / `icamax_` / `izamax_`、
複素ノルム・絶対値和は `scnrm2_` / `dznrm2_` / `scasum_` / `dzasum_` です。
CBLAS は既存の内積6入口に加え、#271 の172入口を LP64／ILP64 に接続します。
row/column-major、Cint の enum、複素係数ポインタ、0-based の `size_t`
インデックスを変換します。LBT の全 CBLAS extension を提供するという意味ではありません。
CBLAS GEMM3M は公開 native 実数 GEMM 3回の合成で、一時領域を確保します。
中間和の overflow・桁落ちがあり、最適化済み3M kernel や速度向上は主張しません。
Fortran GEMM3M は引き続き通常の native 複素 GEMM による算術互換です。

この構成では LP64・ILP64 それぞれ1480シンボル（BLAS 374、LAPACK 1106）を LBT に登録します。
登録を調べるには次を実行します。

```jldoctest
julia> using Colmat, LinearAlgebra

julia> cfg = BLAS.get_config();

julia> libs = filter(l -> realpath(l.libname) == realpath(Colmat.backend_library()), cfg.loaded_libs);

julia> Set(l.interface for l in libs) == Set([:lp64, :ilp64])
true

julia> all(length(BLAS.lbt_forwarded_funcs(cfg, l)) == 2692 for l in libs)
true
```

## フォールバックに残すもの

このページの表は主要系列の概要です。追加の band/packed 系列や LAPACK を含む
現在の接続状況は [LBT と colmat-rs の全ルーチン対応表](@ref) で確認できます。
未登録のルーチンは既存バックエンドを使います。

## 計算経路と一時領域

| 演算 | 現在の経路 |
|---|---|
| GEMM | Julia バッファを借用して Rust colmat の行列積を実行 |
| GEMV | Rust colmat の行列積を実行し、結果を一時ベクトルから出力へ転送 |
| SYMM / HEMM / TRMM | 構造付き入力を一時領域へ展開して行列積を実行 |
| SYRK / SYR2K / HERK / HER2K | 一時行列で積を計算し、指定された三角部分へ書き戻す |
| ベクトル演算、SYMV / HEMV、GER / SYR / HER | Rust 互換層のループ |
| TRSV / TRSM | Rust 互換層の三角求解。右側 TRSM は係数行列の一時展開を使用 |
| GETRF / POTRF | Julia バッファを借用し、因数分解を in-place で実行 |
| GETRS / GETRI / POTRS / POTRI | 因子を借用し、求解・逆行列の結果だけを書き戻す |
| GESDD / GESVD | 入力を借用して読み取り、特異値・特異ベクトルを出力へ書き込む |
| GEEV / GEEVX / SYEV / HEEV 系 | 入力を借用して固有値分解を実行。balance と Schur 形の生成も呼び出し側バッファで実行 |
| GGEV / GGHRD 系 | 入力を借用し、Schur 形・Hessenberg 形を呼び出し側バッファへ書き戻す |
| HSEQR / GEES / HGEQZ / GGES 系 | 入力の Hessenberg 行列・三角行列を借用し、スケーリングとブロック変換を in-place で実行。Q/Z は出力領域へ書き込む |
| TREXC / LAEXC / TRSEN / GEESX | 公開 Schur block move と既存 GEES を使用し、block index・選択順・Q 更新・拒否時の途中状態を変換。S/SEP・RCONDE/RCONDV は元の ILP64 LAPACK、整数配列・callback は ABI 変換 |
| TREVC / TREVC3 | 公開 Schur 固有ベクトル helper、反転した転置/随伴形式による左側、選択列と back-transform を合成。native 拒否は保存した元の LAPACK を使用 |
| TGEXC / TGEX2 / TGSEN / GGESX | public QZ reorder と保存因子の固有値抽出。条件・分離推定は保存した既存 ILP64 LAPACK。内部行列・scratch と整数 ABI 変換用領域を確保 |
| band/packed 三重対角化・Q 生成/適用 | native 三重対角化と packed reflector 適用を合成。AP/TAU は外部 LAPACK と相互利用でき、band は既存 X への Q の右乗算も実行 |
| SBG/HBG/SPG/HPG の GST/GV/GVD/GVX | band/packed を所有行列へ展開し、三角・split Cholesky、合同変換、全 EVD と範囲選択を合成。正の ABSTOL・不確かな値域境界・native 失敗は保存した元の LAPACK を使用 |
| GELS / GELSS / GELSD 系 | 係数行列を借用して SVD・QR に渡す。右辺だけをスケーリング用の作業領域に展開する |
| PSTRF / PSTF2 | 呼び出し側の三角部分を借用する。残余行列の更新用ワークは別途確保する |

現時点で全ルーチンが割り当てなしに動く、または既存バックエンドより高速である
という保証はありません。性能比較では転送先・型・行列サイズ・スレッド数を揃えて
測定してください。

## ネイティブ境界

BLAS ABI は有効なポインタ、形状、leading dimension、ストライド、出力領域を
呼び出し側が渡す契約です。Julia の通常のラッパーによる検査に加え、互換層は
オプション・次元・leading dimension・Level 2/3 のゼロストライドなどを検査します。
Level 1 の引数規約はルーチンごとに異なります。

引数検査で不正が見つかったときは計算せずに戻ります。
ネイティブ診断関数 `colmat_blas_last_error()` は呼び出し OS スレッドに記録した
最後の診断を読み取り、記録をクリアします。0は診断なし、正の値は不正引数番号です。
GEEVX の低水準呼び出しでは、必要な元の実装が未設定なら -1、外部実装の整数出力が
呼び出し側 ABI の範囲外なら -2 を記録します。これらは LAPACK の INFO 値ではありません。
Rust の例外を Julia の例外へ
変換する API や、任意の生ポインタの安全性を検査する API ではありません。


## 接続する LAPACK

全4型・LP64/ILP64 に、次の19系列（計76 canonical シンボル）を提供します。

| 系列 | native 処理 |
|---|---|
| GETRF / GETRF2 / GETF2 / GETRS / GETRI / GESV | 部分ピボット LU・三角求解 |
| POTRF / POTRF2 / POTF2 / POTRS / POTRI / POSV | LLT・三角求解 |
| GESDD / GESVD | SVD、full/thin/overwrite の変換 |
| GEEV / GEEVX | 一般 EVD・balance・左右固有ベクトル・Schur |
| GGEV / GGEV3 / GEGV / GGEVX | 一般化 EVD・homogeneous α/β・左右固有ベクトル |
| SYEV / HEEV / SYEVD / HEEVD | 自己随伴 EVD |
| SYEVR / HEEVR | 全自己随伴 EVD からの範囲選択 |

実数の SY 系と複素数の HE 系は型ごとの同じ系列として数えます。
GEEVX の条件数推定は保存した元の backend に委譲し、LP64 の元の実装がなければ
ILP64 へ整数を変換します。登録前に4型の元の ILP64 GEEVX を要求します。native 対応は SENSE=N です。
GGEVX は BALANC=N/SENSE=N が native で、その他のモードは既存 ILP64 backend へ委譲します。
全 LBT シンボルについての対応候補・公開 API 不足・契約の差は
[LBT と colmat-rs の全ルーチン対応表](@ref)に列挙します。


## ROOK 因子の接続（#372）

実数 SY / 複素 Hermitian HE の `TRF_ROOK`、`TF2_ROOK`、`TRS_ROOK`、
`TRI_ROOK`、`SV_ROOK` の20シンボルを両整数 ABI に接続します。
公開 native Rook pivoting と、2つの swap index を持つ compact IPIV の変換を使い、
外部 LAPACK 因子との solve/inverse の相互利用を検証しています。
TRF/SV の最小 caller WORK と query は1で、内部 scratch は別途確保します。
subnormal 行には対称な合同スケーリング、subnormal D と右辺には block ごとの共通
スケーリングを適用します。これらは共有する通常・packed BK adapter にも適用されます。
専用 Fortran 算法・同一 pivot 列・性能の同等性を表すものではありません。
