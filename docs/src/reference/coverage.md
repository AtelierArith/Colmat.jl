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
| 1 | `copy`, `swap`, `scal`, `axpy`, `axpby`, `iamax`, `rot` | 全4型 |
| 1 | `dot` | 実数2型 |
| 1 | `dotu`, `dotc` | 複素数2型 |
| 1 | `nrm2`, `asum` | 全4型、戻り値は対応する実数型 |
| 1 | `csscal`, `zdscal`, `csrot`, `zdrot` | 複素配列と実数係数 |
| 2 | `gemv`, `symv`, `syr`, `trmv`, `trsv` | 全4型 |
| 2 | `ger` | 実数2型 |
| 2 | `geru`, `gerc`, `hemv`, `her` | 複素数2型 |
| 3 | `gemm`, `symm`, `syrk`, `syr2k`, `trmm`, `trsm` | 全4型 |
| 3 | `hemm`, `herk`, `her2k` | 複素数2型 |

`iamax` の実際のシンボル名は `isamax_` / `idamax_` / `icamax_` / `izamax_`、
複素ノルム・絶対値和は `scnrm2_` / `dznrm2_` / `scasum_` / `dzasum_` です。
実数内積と複素内積には CBLAS の `cblas_sdot` / `cblas_ddot` と
`cblas_cdot{u,c}_sub` / `cblas_zdot{u,c}_sub` も提供します。
CBLAS 全体の互換ライブラリを提供するものではありません。

この構成では LP64・ILP64 それぞれ112シンボルを LBT に登録します。
登録を調べるには次を実行します。

```jldoctest
julia> using Colmat, LinearAlgebra

julia> cfg = BLAS.get_config();

julia> libs = filter(l -> realpath(l.libname) == realpath(Colmat.backend_library()), cfg.loaded_libs);

julia> Set(l.interface for l in libs) == Set([:lp64, :ilp64])
true

julia> all(length(BLAS.lbt_forwarded_funcs(cfg, l)) == 112 for l in libs)
true
```

## フォールバックに残すもの

帯・packed 格納の演算（`gbmv`, `sbmv`, `hbmv`, `spmv`, `hpmv`, `spr` など）、
`rotg`, `rotmg`, `rotm`, `syr2`, `her2`, `gemmt` など、この一覧にないルーチンは
Colmat が提供しません。LAPACK のシンボルも提供しません。
`lu`, `qr`, `svd`, `eigen` などを Rust colmat の分解へ直接接続する API はありません。

## 計算経路と一時領域

| 演算 | 現在の経路 |
|---|---|
| GEMM | Julia バッファを借用して Rust colmat の行列積を実行 |
| GEMV | Rust colmat の行列積を実行し、結果を一時ベクトルから出力へ転送 |
| SYMM / HEMM / TRMM | 構造付き入力を一時領域へ展開して行列積を実行 |
| SYRK / SYR2K / HERK / HER2K | 一時行列で積を計算し、指定された三角部分へ書き戻す |
| ベクトル演算、SYMV / HEMV、GER / SYR / HER | Rust 互換層のループ |
| TRSV / TRSM | Rust 互換層の三角求解。右側 TRSM は係数行列の一時展開を使用 |

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
最後の不正引数番号を読み取り、記録をクリアします。Rust の例外を Julia の例外へ
変換する API や、任意の生ポインタの安全性を検査する API ではありません。
