# t4a-faer / Julia / Colmat 機能対応表

## 目的・調査基準

t4a-faer の公開機能をカテゴリ別に整理し、Julia / LinearAlgebra の対応 API と Colmat の実装状況を追跡する。

- 調査日: 2026-11-04
- t4a-faer: `0.28.1`、コミット [`d5be1473eed9314fca6f80ad8b4072d86282e890`](https://github.com/tensor4all/faer/tree/d5be1473eed9314fca6f80ad8b4072d86282e890)。実際のビルド対象 `deps/t4a-faer` を調査。
- Colmat の判定基準: 検証済み実装コミット `bdd39e3`。以下は main ブランチの実装状況。
- 直近の検証記録: Julia 1.13.3 / RustCall.jl 0.9.2、83371 件のテスト成功。`Pkg.test(; allow_reresolve=false)` は ParallelTestRunner.jl 2.10.0 により 63 ファイルを 16 プロセスで検証。
- 所有コレクション・範囲ビュー・置換・疑似逆・Hessenberg/Schur/QZ などを実装済み。全 76 項目の完了は意味しない。
- 公開 API を**機能単位**で列挙する。型パラメータ違い、可変・不変ビュー、同族の `*_scratch`、内部実装用の doc-hidden 補助シンボルを一件ずつ数える一覧ではない。
- Julia 欄は意味上の対応。型、分解の規約、引数、アルゴリズム選択が完全一致するという意味ではない。標準外のものはその旨を記す。

### 実装状況の凡例

- **R**: Colmat の専用メソッドから t4a-faer に直接接続。
- **J**: Julia の AbstractArray / LinearAlgebra 汎用処理で利用可能。**t4a-faer の同名ルーチンを接続済みという意味ではない**。
- **一部**: 一部の型・操作・呼び出し形式のみ対応。
- **未**: Colmat の専用 API / Rust 接続がない。
- J の動作確認は小さな Float64 行列によるスモーク確認。サイズ・全型・全オプションの保証ではない。

## 1. データ型・構築・ビュー

| 機能 | t4a-faer public API | Julia / LinearAlgebra 対応 | Colmat 実装済 | Colmat 未実装・制約 |
|---|---|---|---|---|
| 密行列の所有・構築 | `Mat<T>`, `MatRef/MatMut/MatUninitMut`, `from_fn`, `zeros`, `ones`, `identity`, `full` | `Matrix`, `zeros`, `ones`, `Matrix(I,m,n)`, `fill` | R: 4 型の `FaerMatrix{T}(A)`、`undef`（ゼロ初期化）、単位行列、`similar/zero/fill!` | 全構築形式・非標準スカラー型 |
| 行・列ベクトル | `Row / Col`, `RowRef / ColRef`, `RowMut / ColMut` | `Vector`, 行ベクトル、`view` | R: `FaerVector` / `FaerRow`、4 型の所有・コピー・演算・容量管理 | 全 Row/Col 演算組合せ、任意の借用 Julia バッファ |
| 対角行列 | `diag::{Diag, DiagRef, DiagMut}` | `Diagonal`, `diag`, `diagind` | R: `FaerDiagonal`、格納対角の演算・`filldiagonal!`・`diagonalview`、密/三角との左右の native 積 | 全 DiagRef/DiagMut 操作。対角積は現在 dense snapshot 経由 |
| 読み書き・形状 | `nrows/ncols/shape`, 添字、`row/col` | `size/axes/getindex/setindex!` | R: shape・要素読み書き、行・列の範囲ビュー | 全形状・stride trait の公開 |
| コピー・Julia との変換 | `clone/to_owned/copy_from` | `copy/deepcopy/copyto!/Matrix` | R: 所有型の独立コピー・Julia 変換・範囲ビューの一括コピー/代入 | 借用 Julia バッファの恒久的ゼロコピー型。ビュー代入は入力を一時コピー |
| 部分行列・分割・逆順 | `submatrix/subrows/subcols/split_at/reverse_rows/reverse_cols` | `view`, `@view`, slicing, `reverse` | R: `view` の範囲・間引き・逆順・入れ子、native copy/積/代入 | 任意の整数ベクトル添字は汎用処理。独立した split API |
| 転置・共役・随伴 | `transpose/conjugate/adjoint` | `transpose/conj/adjoint`, `A'` | J: 遅延ラッパー。R: `conj`、転置・随伴のコピー、複素演算 | 積や求解で転置ラッパーを一時所有行列へコピーする経路あり |
| 動的容量・サイズ変更 | `with_capacity/try_reserve/reserve/resize_with/truncate` | `sizehint!` 等に類する容量管理。Matrix に直接同等の公開 API はない | R: `reserve!`、`sizehint!`、`resize!(;fill)`、`truncate!` | 容量の照会・全低水準制御 |
| 置換・行列交換 | `Perm/PermRef`, `permute_rows/cols`, `swap_rows/cols` | 置換ベクトル、`A[p,:]`, `A[:,p]` | R: `FaerPermutation`、逆置換・合成、`permute_rows/cols` と `!`、`swap_rows!/swap_cols!`、`A[p,:]/A[:,p]`、疎数値行列・疎パターンの native 置換 | ビューの更新は一時結果を経由。全低水準 PermRef 形式・caller-owned scratch |
| スカラー型 | `f32/f64`, `c32/c64`、`fx128/cx128` の公開型エイリアス、generic field traits | `Float32/Float64/ComplexF32/ComplexF64` 等 | R: Float32 / Float64 / ComplexF32 / ComplexF64 の所有・演算・分解 | fx128/cx128、任意の field traits |

## 2. 基本演算・集約・積

| 機能 | t4a-faer public API | Julia / LinearAlgebra 対応 | Colmat 実装済 | Colmat 未実装・制約 |
|---|---|---|---|---|
| 加算・減算・符号 | 行列演算、要素操作 | `A+B`, `A-B`, `-A` | R: 4 型の加減算・符号と mixed-type promotion | 全種類の外部配列との組合せの専用最適化 |
| スカラー倍・除算 | `Scale`、行列スカラー演算 | `α*A`, `A*α`, `A/α` | R: 実数・複素数スカラーと型 promotion | 非標準スカラー型 |
| 行列積 | `linalg::matmul::{matmul,matmul_with_conj}` | `A*B`, `mul!` | R: 4 型の行列積、3/5 引数 `mul!`、alias 対応 | `mul!` は一時結果を割り当てる。構造化積・全 mixed-operand の最適化 |
| 行列ベクトル積 | 同上、Row/Col ビュー | `A*x`, `mul!(y,A,x)` | R: `FaerMatrix * FaerVector` と範囲ビュー、alias-safe ベクトル `mul!` | 全行ベクトル・構造化型・外部配列との組合せ |
| 内積 | `matmul::dot::inner_prod` | `dot(x,y)` | R: 行列・所有ベクトル・行・対角の `dot`、複素数の第 1 引数を共役 | 借用ベクトルビューの専用 dot 経路 |
| 要素ごとの map / zip | `map/for_each`, `zip!/unzip!/Zip!` | `map`, `map!`, broadcast `f.(A)` | J: broadcast 動作確認 | Rust バルク処理の専用接続。要素単位 FFI、Julia Matrix の出力になる場合がある |
| ノルム | `norm_l2/squared_norm_l2/norm_l1/norm_max` | `norm(A)`, `sum(abs2,A)`, `norm(A,1/Inf)` | R: 4 型の `norm(A,1/2/Inf)`。複素数は Julia の絶対値規約 | その他の p の専用経路。Frobenius norm と `opnorm(A,2)` は別 |
| 和・最大・最小・検査 | `sum/max/min/is_all_finite/has_nan/is_identity` | `sum/maximum/minimum`, `all(isfinite,A)`, `any(isnan,A)`, `isone` 相当 | R: 和・軸別和、Float64 の extrema・finite/NaN/identity/symmetry 検査 | 集約・検査の全型・全オプションは要個別照合 |
| 行列式 | `MatRef::determinant(par)` | `det`, 関連: `logdet/logabsdet` | R: `det`、分解を使う `logabsdet/logdet`、複素位相対応 | 高度な引数・エラー規約の網羅 |
| トレース | diagonal ビュー＋集約で構成 | `tr(A)` | R: `tr(A)` | 全特殊構造の最適化 |
| Kronecker 積 | `MatRef::kron`, `linalg::kron::kron` | `kron(A,B)` | R: 4 型・型 promotion の `kron(A,B)` | 疎行列・ビューの専用経路 |
| 三角・構造指定の行列積 | `matmul::triangular` | 三角ラッパーとの積、BLAS の構造化積に相当 | R: 下/上/単位三角と native 密行列・三角ペア、4 型・promotion、3/5 引数 `mul!`、alias・無視要素の NaN | 入力 snapshot と出力一時領域を確保。全外部配列・vector 組合せと destination structure の低水準指定 |

## 3. 密行列の分解・固有値

| 機能 | t4a-faer public API | Julia / LinearAlgebra 対応 | Colmat 実装済 | Colmat 未実装・制約 |
|---|---|---|---|---|
| 部分ピボット LU | `partial_piv_lu`, `PartialPivLu::{L,U,P}` | `lu(A)` | R: 再利用可能な `lu/lu!`、L/U/p/P、packed、求解・逆行列・再構成 | 全オプション・無割当の直接 in-place 経路 |
| 完全ピボット LU | `full_piv_lu`, `FullPivLu::{L,U,P,Q}` | 完全ピボット LU（標準 `lu(A)` と同一ではない） | R: `full_piv_lu/full_piv_lu!`、L/U/p/q/P・packed、求解・再構成 | 全低水準オプション・無割当の直接 in-place 経路 |
| QR | `qr`, `Qr`, `Q_basis/Q_coeff/R/thin_R`, `compute_Q/compute_thin_Q` | `qr(A)`, `Matrix(F.Q)`, `F.R` | R: `qr/qr!`、Q/Qthin/R・Qbasis/Qcoeff/packed、分解の再利用・再構成 | 無割当の直接 in-place 経路 |
| 列ピボット QR | `col_piv_qr`, `ColPivQr`, `P` | `qr(A,ColumnNorm())` | R: `qr/qr!(A,ColumnNorm())`、置換・因子・compact basis/coeff・求解 | 独立したランク判定・全許容誤差インターフェース |
| Cholesky LLᴴ | `llt`, `Llt`, `L` | `cholesky(Hermitian(A))` | R: `cholesky/cholesky!`、Hermitian/Symmetric の指定三角、packed・分解の再利用 | 複素対称非 Hermitian と全オプションは要照合 |
| ピボット付き Cholesky | `cholesky::llt_pivoting::{factor,solve,inverse,reconstruct}` | `cholesky(Hermitian(A),RowMaximum())` に相当 | R: native factors の標準 `CholeskyPivoted`、4 型、指定三角・置換・ランク・再構成・求解・逆・copy/close | 負の tol は faer 固定閾値、非負 tol は Rust adapter の対角ピボット法で補完。`cholesky!` は staged writeback。外部 scratch・block_size は未公開 |
| LDLᴴ | `ldlt`, `Ldlt::{L,D}` | `ldlt` に相当。ただし Julia 標準の対象構造・ピボット規約と要照合 | R: `ldlt/ldlt!`、L/D・packed・求解・再構成 | 複素対称非 Hermitian、全オプション |
| ピボット付きブロック LDLᴴ | `lblt`, `Lblt::{L,B_diag,B_subdiag,P}` | `bunchkaufman(Hermitian(A))` に相当 | R: `bunchkaufman/bunchkaufman!`、ブロック D・置換・packed・求解・再構成 | 全 Julia 規約・オプションの網羅 |
| thin / full SVD | `thin_svd/svd`, `Svd::{U,S,V}` | `svd(A;full=false/true)` | R: 4 型の `svd/svd!`、thin/full、空行列、`scratch=`/`team=`、互換 `faer_svd`、`native=true` の Rust Svd 保持・成分コピー・再構成・copy/close | 既定の因子は Julia 配列。native owner の caller-owned scratch、Julia の alg/tolerance キーワードは未対応 |
| 特異値のみ | `singular_values` | `svdvals(A)` | R: 4 型の `svdvals/svdvals!`、実数出力、`scratch=`/`team=` | alg キーワード |
| 自己随伴固有値分解・固有値のみ | `self_adjoint_eigen/self_adjoint_eigenvalues`, `SelfAdjointEigen` | `eigen/eigvals(Hermitian(A))` | R: `eigen/eigvals/eigen!/eigvals!(Hermitian(A))`、指定三角・4 型、`scratch=`/`team=`、self-adjoint の requirements/reserve、`native=true` の Rust SelfAdjointEigen 保持・再利用 | native owner の caller-owned scratch、複素対称非 Hermitian、全キーワード |
| 一般固有値分解・固有値のみ | `eigen/eigenvalues`, `Eigen` | `eigen/eigvals(A)` | R: 4 型の `eigen/eigvals/eigen!/eigvals!(A)`、`scratch=`/`team=`、`eigen(;side=:left/:right/:both)` と両側の作業領域照会、bang は別途 Schur 形の書き戻し | EVD driver 内の balancing 等の全制御項目。bang の Schur 書き戻し分解は別途逐次・内部 workspace |
| 自己随伴一般化固有値 | `self_adjoint_generalized_eigen/self_adjoint_generalized_eigenvalues` | `eigen/eigvals(Hermitian(A),Hermitian(B))` | R: Hermitian/実 Symmetric ペアの `eigen/eigvals/eigen!/eigvals!(A,B)`、4 型・指定三角・B 正定値、`scratch=`/`team=`、LLT/EVD 再利用領域 | 全構造・高度な失敗規約の網羅 |
| 一般化固有値 | `generalized_eigen`, `GeneralizedEigen::{U,S_a,S_b}` | `eigen(A,B)`, `eigvals(A,B)` | R: `eigen/eigvals/eigen!/eigvals!(A,B)`、4 型・promotion・`scratch=`/`team=`、`homogeneous=true` で α/β、`side=:left/:right/:both` で左/右/同時ベクトル、両側の作業領域照会、bang は別途逐次 QZ S/T 書き戻し | 高度な規約・全極端スケール・defective pencil の網羅。左 driver の非有限出力は随伴ペアの右 driver へ回復（追加コピー、[faer #219](https://github.com/tensor4all/faer/issues/219)）。重複有限/無限固有値の非有限ベクトルは native SVD 零空間で回復（追加割当、[faer #217](https://github.com/tensor4all/faer/issues/217)） |
| Schur・並べ替え | `evd::{hessenberg_schur_real/cplx,reorder_schur_real/cplx,eigenvectors_from_schur_*}` | `schur`, `ordschur`, Schur からの固有ベクトル | R: 4 型の `schur/schur!`、`ordschur/ordschur!`、`eigvecs/eigen(::Schur)`、Rust 所有 T/Z | 既定アルゴリズムのみ。全 SchurParams・並列 team・caller-owned scratch |
| 一般化 Schur / QZ | `gevd::{gen_hessenberg,qz_real,qz_cplx}`, `hessenberg_to_qz` | `schur(A,B)` | R: 4 型の `schur(A,B)` / `schur!`、native S/T/Q/Z、α/β、`hessenberg(A,B)` と再利用 | 一般化 `ordschur`、全チューニング項目・caller-owned scratch。作業領域はラッパー内で補正・確保 |

## 4. 求解・逆行列・分解の再利用

| 機能 | t4a-faer public API | Julia / LinearAlgebra 対応 | Colmat 実装済 | Colmat 未実装・制約 |
|---|---|---|---|---|
| 正方連立方程式 | `Solve::solve/solve_in_place` | `A\B`, `ldiv!` | R: `A\B`、再利用可能な `F\B`、保存済み native SVD/EVD の混合型・許容誤差付き求解、alias-safe `ldiv!` | 真の無割当 in-place 求解・全引数形式 |
| 共役・転置・随伴の求解 | `solve_conjugate/solve_transpose/solve_adjoint` と in-place 版 | `conj(A)\B`, `transpose(A)\B`, `A'\B` | R: 転置・随伴分解の求解、行列ラッパーの求解 | 行列ラッパーはコピー経由。全共役 in-place 形式 |
| 右側からの求解 | `rsolve_*_in_place` | `B/A`, `rdiv!` | R: `B/A`、`B/F`、保存済み native SVD/EVD の転置/随伴・混合型を含む右求解、`rdiv!` | 全型・全入力組合せの専用最適化 |
| 最小二乗 | `SolveLstsq::{solve_lstsq,solve_lstsq_in_place,...}`（QR/SVD 等） | 長方形 `A\b`, `ldiv!` | R: 長方形の求解、QR と広幅・ランク不足時の SVD 経路、保存済み native SVD の `F\B`/3 引数 `ldiv!`（`atol/rtol/team`） | 全アルゴリズム選択・許容誤差設定 |
| 三角連立方程式 | `solve_{unit_}{lower,upper}_triangular_in_place`, `linalg::triangular_solve` | `LowerTriangular/UpperTriangular/UnitLowerTriangular/UnitUpperTriangular` の `\` | R: lower/upper/unit の三角ラッパーの求解 | 全 mixed-operand と低水準 scratch 公開 |
| 逆行列 | `DenseSolveCore::inverse`, `triangular_inverse::invert_*` | `inv(A)`, `inv(F)` | R: `inv(A)`、`inv(F)`（保存済み native SVD/EVD・転置/随伴を含む。数値的特異性を検査）、三角 inverse | 破壊的 inverse・全オプション |
| 疑似逆 | `Svd::pseudoinverse`, `SelfAdjointEigen::pseudoinverse`, `pseudoinverse_from_*[_with_tolerance]` | `pinv(A;atol,rtol)` | R: 4 型の `pinv(A;atol,rtol)`、native SVD 再構成、Hermitian/実 Symmetric の EVD 再構成、保存済み native SVD/EVD の `pinv(F;atol,rtol,team,scratch)`・必要量照会/reserve | 残りの分解形式、非標準スカラー・高度なアルゴリズム制御 |
| 分解からの再構成 | `DenseSolveCore::reconstruct`、各分解の accessor | `Matrix(F)`, 因子の積 | R: 所有する LU/QR/Cholesky/LDLT・Schur/Hessenberg・三重/二重対角化の `reconstruct/Matrix/FaerMatrix`・型指定変換、疎 Cholesky/simplicial LU の native 疎再構成、一般化 Schur/Hessenberg の `(A,B)` | auto/supernodal 疎 LU・疎 QR 等の残りの形式 |
| ランク・条件数 | SVD/QR などを組み合わせて構成（同名の高水準 API としては扱わない） | `rank/cond/opnorm` | R: native スペクトル・分解を使う `rank(;atol,rtol)`、`cond(A,1/2/Inf)`、4 型の `opnorm` | 推定条件数・全構造化入力の最適化。faer 同名の高水準 API としての主張ではない |

## 5. 疎行列

| 機能 | t4a-faer public API | Julia / LinearAlgebra 対応 | Colmat 実装済 | Colmat 未実装・制約 |
|---|---|---|---|---|
| CSC・CSR、所有・借用・構築 | `SparseColMat/SparseRowMat` と Ref/Mut、`Triplet/Pair` | SparseArrays の `SparseMatrixCSC/sparse`。CSR は直接同じ標準型ではない | R: `FaerSparseMatrix`、4 型の native CSC/CSR、triplet 構築・重複加算、Julia CSC 変換 | 借用 Ref/Mut、zero-copy、非圧縮形式・index 型の選択 |
| 数値を持たない疎構造 | `SymbolicSparseColMat/RowMat` | sparsity pattern、symbolic analysis | R: `FaerSparsity`、独立した symbolic CSC/CSR、数値行列への再利用、LU/QR/Cholesky の `FaerSparseSymbolic` | 借用 symbolic view |
| 疎加減算・構造の結合 | `sparse::ops::{add,sub,add_assign,sub_assign,add_into,sub_into,union_symbolic,...}` | 疎行列の `+/-` | R: native `+/-`、`pattern_union`、固定構造の `add_assign!/sub_assign!/add_into!/sub_into!` | 任意 callback、全体 broadcast/map の native 経路 |
| 疎×疎・疎×密・密×疎 | `sparse::linalg::matmul::{sparse_sparse_matmul,sparse_dense_matmul,dense_sparse_matmul}` | sparse/dense の `*/mul!` | R: 4 型・型 promotion、CSC/CSR の native 積、`mul!`、ベクトルとの積 | symbolic/numeric 積計画の再利用、低水準制御 |
| 疎 Cholesky | `sp_cholesky`, `SymbolicLlt/Llt`、低水準の疎 Cholesky | `cholesky(sparse(Hermitian(A)))` | R: CSC/CSR・4 型、`cholesky`、選択三角、`FaerSparseSymbolic`、`cholesky!`、求解、AMD/恒等/任意 ordering、分解方式指定、`p/algorithm/factor_storage`、疎 `L/U` 抽出・再構成、分解対角からの `det/logdet/logabsdet` | 外部 scratch・追加の低水準制御 |
| 疎 LU | `sp_lu`, `SymbolicLu/Lu` | `lu(sparse(A))` | R: CSC/CSR・4 型、`lu`、symbolic 再利用、`lu!`、転置/随伴・混合精度の求解。`algorithm=:simplicial` の native L/U・p/q、元順序への疎再構成・型指定変換、恒等/任意/COLAMD column ordering（dense_row/dense_col/aggressive） | auto/supernodal の L/U・置換成分と低水準制御。明示的 simplicial の既定列順序は恒等 |
| 疎 QR・最小二乗 | `sp_qr`, `SymbolicQr/Qr`, `SolveLstsq` | `qr(sparse(A))`, sparse `A\b` | R: CSC/CSR・4 型、縦長/正方の列フルランク最小二乗、明示的 `algorithm=:supernodal` 選択、`qr!`、symbolic 再利用、`ldiv!`。auto/supernodal の同型 matrix RHS 向け `ldiv!(X,F,B;scratch)` と作業領域照会/reserve。明示的 `algorithm=:simplicial` の薄い native Q・疎 R・p/pcol・元列順序への native 再構成・恒等/任意/COLAMD 列順序（dense_row/dense_col/aggressive）・転置/随伴の最小ノルム求解 | 元の入力が横長・rank-deficient の求解、auto/supernodal の Q/R 成分・転置/随伴求解、完全 Q、低水準制御 |
| 疎三角求解 | `sp_solve_{unit_}{lower,upper}_triangular_in_place` | sparse 三角ラッパーの `\` | R: CSC/CSR・4 型、通常/単位の上下三角、転置/随伴、matrix/vector RHS、`ldiv!` | 作業領域の外部指定、割り当てを伴わない直接 in-place 経路 |
| Fill-reducing ordering | `sparse::linalg::{amd,colamd}` | 疎分解の並べ替え・ordering に相当 | R: `amd/colamd`、numeric/symbolic CSC/CSR、長方形 COLAMD、制御値・AMD 演算数見積もり、`FaerPermutation` 返却、Cholesky と明示的 simplicial LU/QR への直接 ordering 指定 | auto/supernodal LU/QR への ordering 指定、caller-owned scratch |

## 6. Matrix-free・反復法

| 機能 | t4a-faer public API | Julia / LinearAlgebra 対応 | Colmat 実装済 | Colmat 未実装・制約 |
|---|---|---|---|---|
| 線形作用素・双方向作用素 | `matrix_free::{LinOp,BiLinOp}` | `mul!` を持つ作用素型相当。LinearAlgebra に同一の trait はない | R: `FaerLinearOperator`、dense/CSC/CSR/diagonal/identity、積・共役・転置・随伴 | 任意の Julia callback・作用素合成 |
| 前処理 | `Precond/BiPrecond/IdentityPrecond` | 反復法の前処理（標準外） | R: native in-place 適用、`identity_preconditioner`、`diagonal_preconditioner` | callback・分解を用いる前処理 |
| 共役勾配法 | `conjugate_gradient::{conjugate_gradient,CgParams,CgInfo,CgError}` | CG solver（標準外） | R: `cg/cg!`、block RHS、4 型、初期値・許容誤差・反復上限、失敗 iterate・独立した真の残差 | callback・履歴・独立 scratch API |
| BiCGSTAB | `bicgstab::{bicgstab,BicgParams,BicgInfo,BicgError}` | BiCGSTAB solver（標準外） | R: `bicgstab/bicgstab!`、左右前処理・block RHS・4 型・初期値・許容誤差・反復上限、失敗 iterate・真の残差 | callback・履歴・独立 scratch API |
| LSMR | `lsmr::{lsmr,LsmrParams,LsmrInfo,LsmrError}` | 最小二乗反復 solver（標準外） | R: `lsmr/lsmr!`、長方形・block RHS・右前処理・4 型・初期値・許容誤差・反復上限、真の前処理付き正規残差 | callback・履歴・独立 scratch API。upstream の絶対許容誤差は有効な相対値へ換算（[faer #207](https://github.com/tensor4all/faer/issues/207)） |
| 部分固有値・部分 SVD | `matrix_free::eigen::{partial_eigen,partial_self_adjoint_eigen,partial_svd}` と params/info/scratch | 部分スペクトルを求める Krylov 法（標準外） | R: `partial_eigen/partial_self_adjoint_eigen/partial_svd`、4 型・native operator・射影パラメータ・初期値・収束数・真の残差 | 小規模/full-projection の自己随伴・SVD は native dense fallback（[faer #204](https://github.com/tensor4all/faer/issues/204)、[#213](https://github.com/tensor4all/faer/issues/213)）。callback と外部 scratch は未対応 |

## 7. 低水準の数値プリミティブ

| 機能 | t4a-faer public API | Julia / LinearAlgebra 対応 | Colmat 実装済 | Colmat 未実装・制約 |
|---|---|---|---|---|
| 破壊的分解・作業領域管理 | 各分解モジュールの `*_in_place / *_scratch`、`temp_mat_scratch/temp_mat_zeroed` | `lu!/qr!/cholesky!/svd!/eigen!`、LAPACK workspace 相当 | 一部 R: `lu!/full_piv_lu!/qr!/cholesky!/ldlt!/bunchkaufman!/svd!/svdvals!/eigen!/eigvals!`（pivoting 含む）、単一行列・行列対の `hessenberg! / schur!` は成功後に入力を更新 | 直接 in-place の低水準経路・全 tuning。caller-owned scratch は Pfaffian/SVD/単一行列・一般化 EVD/保存済みスペクトル因子の疑似逆に接続、他 kernel は未。現行の ! は独立因子・一時結果を割り当てる |
| Householder | `householder::make_householder_in_place/upgrade_householder_factor/apply_block_householder_*/apply_block_householder_sequence_*` | reflector・QR の Q 適用相当 | R: `FaerHouseholder/householder`、`FaerHouseholderSequence`、4 型の basis/tau/factor 構築、block factor upgrade/reblock、左右の sequence 適用・求解・`mul!`、随伴/転置/共役・ビュー | `prev_block_size` の直接制御、外部 scratch |
| Jacobi/Givens 回転 | `JacobiRotation::{make_givens,rotg,apply_*,transpose,adjoint}` | `givens`, `Givens`, `lmul!/rmul!` | R: native 配列の `givens`、標準 Givens の左/右適用・`*`、4 型・昇格・ベクトル・範囲ビュー、随伴/合成回転。`FaerJacobiRotation`、`rotg/make_givens`、2×2左右適用、transpose/conj/adjoint、Givens 埋め込み | — |
| Hessenberg 化・balancing | `evd::hessenberg::hessenberg_in_place`, `evd::{balance_in_place,balance_eigenvectors_in_place,...}` | `hessenberg/hessenberg!`、LAPACK balancing 相当 | R: 4 型の `hessenberg/hessenberg!`、明示的な Rust 所有 H/Q、Schur 段階への再利用。`balance/balance!`、`FaerBalance` の B/perm/scale/active、右固有ベクトルの `balance_eigenvectors[!]` | compact Householder 形式・全パラメータ・外部 scratch |
| 対称三重対角化・三重対角 EVD | `evd::tridiag::tridiag_in_place`, `tridiagonal_self_adjoint_evd` | `SymTridiagonal`、構造化固有値分解 | R: `tridiagonalize[!]`、native self-adjoint wrapper の `hessenberg[!]`、実 H/明示 Q、`eigen/eigvals(F)`、native ベクトルを持つ SymTridiagonal、`tridiagonal_eigen/eigvals`、4 型・複素位相・QR/分割統治 | compact reflector の直接公開、外部 scratch・全 tuning |
| 二重対角化・二重対角 SVD | `svd::bidiag::bidiag_in_place`, `bidiagonal_svd` | `Bidiagonal`、LAPACK bidiagonal SVD 相当 | R: `bidiagonalize[!]`、thin/full U/B/V・縦長/横長・4 型、`bidiagonal_svd`、`svd/svdvals(F)` と native ベクトルを持つ Bidiagonal | compact reflector の直接公開・外部 scratch・全 tuning。values-only も native の dense factors/workspace を使用 |
| Pfaffian | `linalg::pfaffian::{pfaffian,pfaffian_scratch,PfaffianError}` | 反対称行列の Pfaffian。LinearAlgebra に直接同名の高水準 API はない | R: `pfaffian`、反対称性の検査・複素対応、`FaerScratch`/容量・alignment 照会/reserve/copy/close、`scratch=`/`team=` | 他 kernel の caller-owned scratch、低水準アルゴリズム制御 |

## 8. 並列実行・統計・I/O・補助 API

| 機能 | t4a-faer public API | Julia / LinearAlgebra 対応 | Colmat 実装済 | Colmat 未実装・制約 |
|---|---|---|---|---|
| 逐次実行 | `Par::Seq` | 同期・逐次の数値処理 | R: 既定の native 数値処理、幅1の FaerTeam | 明示的 team 行列積/Pfaffian/SVD/単一行列・一般化 EVD 以外は逐次 |
| 呼び出し側所有の並列 team | `Par::Team`, `team::with_team`, `Par::reborrow`（threadteam feature） | Julia task/thread や BLAS thread 設定と関心は近いが、直接同一ではない | 一部 R: `FaerTeam` の pool/予約所有・再利用・copy/close/do、`team_mul`・team 指定 `mul!`・`pfaffian/svd/svdvals` と単一行列・一般化ペアの `eigen/eigvals(;team)`、4 型・promotion・ビュー・alias | 行列積/Pfaffian/SVD/単一行列・一般化 EVD 以外の team 接続、外部 pool の借用、全 wait/cost tuning は未対応。owner lock により呼び出しを直列化 |
| 行・列の平均・分散 | `stats::{col_mean,row_mean,col_varm,row_varm,NanHandling}` | Statistics の `mean/varm` と dims/NaN 規約の対応 | R: `Statistics.mean/var/varm`、4 型・行/列/全体・dims・ビュー・NaN propagate/ignore・標本/母分散 | native データを primitive buffer にコピーして faer kernel を適用。指定平均は native 型へ昇格 |
| 乱数・直交/ユニタリ行列の生成 | `CwiseMat/Col/RowDistribution`, `ComplexDistribution`, `UnitaryMat`（rand feature） | Random の `rand/randn`、ランダム直交行列の生成相当 | R: `FaerRNG`、`rand/randn` の native Mat/Col/Row、`rand!/randn!`・ビュー、`randunitary`、4 型・複素成分正規化・copy/seed/close | 任意の Rust Distribution/Julia callback の直接公開は未対応。Julia RNG は非空の生成ごとに UInt64 シードを1個渡す |
| NPY 読み込み・ビュー | `io::npy::{Npy,FromNpy,...}`, `as_aligned_ref/to_mat`（npy feature） | NPY 対応 I/O（Julia 標準外） | R: `loadnpy/npyview/npyinfo`、4 型・versions 1–3・C/Fortran・両 endian、独立 Mat と owned read-only aligned view、metadata/copy/deepcopy/close | 真の mmap/元バッファ別名は未対応。rank>2・他 dtype・連結配列は拒否。aligned/native-endian 正規化で [faer #210](https://github.com/tensor4all/faer/issues/210)・[#211](https://github.com/tensor4all/faer/issues/211) を回避 |
| 型・形状・stride・共役・蓄積指定 | `Shape/Stride/Idx/IdxInc/MaybeIdx`, `ContiguousFwd/Bwd`, `Conj/Accum/Side`, `Spec/Auto` | axes/strides、BLAS の共役・αβ・上下三角指定に相当 | 一部: サイズ、1-based 添字への変換、mul! の αβ | Rust の型レベル shape/stride、全制御項目の公開 |
| リテラル・連結・zip マクロ | `mat!/row!/col!`, `concat!`, `zip!/unzip!/Zip!` | 配列リテラル、`hcat/vcat`、broadcast/zip | Julia 配列から FaerMatrix を構築可能 | Rust マクロ自体を Julia に移植する API は未提供 |

## 判定上の注意

- 「R」は専用 Rust 経路、「J」は Julia 汎用実装であり、区別する。一部の操作は一時所有行列へコピーする。
- `FaerMatrix` は上記 4 型を保持する。疎入力を密行列化できても疎行列サポートには数えない。
- `rayon` はブロック二重対角化で必要な Y 更新が feature gate 内にある不具合の回避（[faer #203](https://github.com/tensor4all/faer/issues/203)）。既定の数値呼び出しは `Par::Seq`。明示的 team 行列積/Pfaffian/SVD/単一行列・一般化 EVD は `Par::Team` を借用する。
- Cargo は `default-features=false, features=["std","linalg","rayon","rand","npy","threadteam"]`。public API の存在・feature の有効化・Julia 接続は別々に判定する。
- SVD の Rust V に対して Julia `F.Vt` は Vᴴ。置換添字は Julia 側で 1-based にする。
- 範囲ビューは親を保持し、native 呼び出し中にだけ MatRef/MatMut を作る。容量確保では維持し、親の形状変更・close では失効する。
- auto/supernodal 疎 LU と疎 QR の成分公開・横長/ランク落ち QR・auto LU/QR への ordering 指定、matrix-free、残りの破壊的分解・scratch、並列 team は今後の実装対象。一般化 Schur の並べ替えにも未対応。
- `pinv/rank` の既定 rtol は Julia の `min(size(A)...)*eps`。faer の既定値と異なるため明示して渡す。Float32 と Float64 許容誤差の境界比較も検証した。
- 実 Schur の共役対は 2×2 ブロックとして扱い、選択では対を分割しない。Hessenberg の Q は明示行列で、compact reflector 形式とは区別する。
- QZ の α/β は無限固有値と 0/0 を保つ同次形式。検査した faer real-QZ の scratch 推奨量には偶数丸め window・index 配列数の不足があるため、依存クローンを変更せずラッパーで再帰的な割当上限を補正した。blocked 経路もテスト済み。
- Jacobi の複素 zero-q r と 2×2 右適用の補正は [faer #205](https://github.com/tensor4all/faer/issues/205)、[#206](https://github.com/tensor4all/faer/issues/206) に Rust MWE 付きで報告。
- RustCall [#564](https://github.com/AtelierArith/RustCall.jl/issues/564)・[#565](https://github.com/AtelierArith/RustCall.jl/issues/565) は v0.9.2 で解消。`FaerMatrix` のコピーは `Result<Self, String>` を返すメソッドを使用し、型が決まっている所有者の引数は `&T` / `&mut T` として検査する。型を実行時に切り替える引数と省略可能な所有者には明示的な生ポインタ経路を残す。

## ソース

- [一般化 Hessenberg](https://github.com/tensor4all/faer/blob/d5be1473eed9314fca6f80ad8b4072d86282e890/faer/src/linalg/gevd/gen_hessenberg/mod.rs)
- [Real QZ と scratch](https://github.com/tensor4all/faer/blob/d5be1473eed9314fca6f80ad8b4072d86282e890/faer/src/linalg/gevd/qz_real/mod.rs)
- [Complex QZ](https://github.com/tensor4all/faer/blob/d5be1473eed9314fca6f80ad8b4072d86282e890/faer/src/linalg/gevd/qz_cplx/mod.rs)

- [Hessenberg reduction](https://github.com/tensor4all/faer/blob/d5be1473eed9314fca6f80ad8b4072d86282e890/faer/src/linalg/evd/hessenberg.rs)
- [Schur・並べ替え・Schur 固有ベクトル](https://github.com/tensor4all/faer/blob/d5be1473eed9314fca6f80ad8b4072d86282e890/faer/src/linalg/evd/mod.rs)

- [置換・行列交換](https://github.com/tensor4all/faer/blob/d5be1473eed9314fca6f80ad8b4072d86282e890/faer/src/perm/mod.rs)
- [SVD 疑似逆の許容誤差付き再構成](https://github.com/tensor4all/faer/blob/d5be1473eed9314fca6f80ad8b4072d86282e890/faer/src/linalg/svd/mod.rs)
- [自己随伴 EVD 疑似逆](https://github.com/tensor4all/faer/blob/d5be1473eed9314fca6f80ad8b4072d86282e890/faer/src/linalg/evd/mod.rs)

- [Julia との公式移行対応表（調査コミット固定）](https://github.com/tensor4all/faer/blob/d5be1473eed9314fca6f80ad8b4072d86282e890/docs/migration/julia.md)
- [高水準 dense solver API](https://github.com/tensor4all/faer/blob/d5be1473eed9314fca6f80ad8b4072d86282e890/faer/src/linalg/solvers.rs)
- [MatRef の変換・reduction・determinant・kron](https://github.com/tensor4all/faer/blob/d5be1473eed9314fca6f80ad8b4072d86282e890/faer/src/mat/matref.rs)
- [低水準 linalg 公開モジュール](https://github.com/tensor4all/faer/blob/d5be1473eed9314fca6f80ad8b4072d86282e890/faer/src/linalg/mod.rs)
- [疎行列 solver](https://github.com/tensor4all/faer/blob/d5be1473eed9314fca6f80ad8b4072d86282e890/faer/src/sparse/solvers.rs)
- [matrix-free 公開モジュール](https://github.com/tensor4all/faer/blob/d5be1473eed9314fca6f80ad8b4072d86282e890/faer/src/operator/mod.rs)
- [stats](https://github.com/tensor4all/faer/blob/d5be1473eed9314fca6f80ad8b4072d86282e890/faer/src/stats/mod.rs)
- Colmat 検証済みコミットの調査対象: `src/`, `deps/rust/src/`, `deps/rust/Cargo.toml`, `test/`。

- [AMD ordering](https://github.com/tensor4all/faer/blob/d5be1473eed9314fca6f80ad8b4072d86282e890/faer/src/sparse/linalg/amd.rs)
- [COLAMD ordering](https://github.com/tensor4all/faer/blob/d5be1473eed9314fca6f80ad8b4072d86282e890/faer/src/sparse/linalg/colamd.rs)

- [Cholesky symbolic ordering・simplicial/supernodal・数値分解](https://github.com/tensor4all/faer/blob/d5be1473eed9314fca6f80ad8b4072d86282e890/faer/src/sparse/linalg/cholesky.rs)

- [Jacobi/Givens 回転](https://github.com/tensor4all/faer/blob/d5be1473eed9314fca6f80ad8b4072d86282e890/faer/src/linalg/jacobi.rs)

- [Householder 生成・block/sequence 適用](https://github.com/tensor4all/faer/blob/d5be1473eed9314fca6f80ad8b4072d86282e890/faer/src/linalg/householder.rs)
