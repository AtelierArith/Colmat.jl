# Frozen scope for Issue #1

The following inventory is a snapshot of the requested scope, captured before
implementation. Historical implementation statuses below are not current
completion claims. See implementation-status.md for ongoing work.

## 目的・調査基準

t4a-faer の公開機能をカテゴリ別に整理し、Julia / LinearAlgebra の対応 API と Colmat の実装状況を追跡する。

- 調査日: 2026-11-03
- t4a-faer: `0.28.1`、コミット [`d5be1473eed9314fca6f80ad8b4072d86282e890`](https://github.com/tensor4all/faer/tree/d5be1473eed9314fca6f80ad8b4072d86282e890)。実際のビルド対象 `deps/t4a-faer` を調査。
- Colmat main の基準: [`d4f5a51`](https://github.com/AtelierArith/Colmat.jl/commit/d4f5a519d2ec9ef1d7268d79a6b860e554a46567)。
- **実装状況には、このセッションの未コミット作業ツリーを含む。FaerMatrix とその演算子・svd・svdvals はまだ main に未反映。** main にあるのは Julia 配列を渡す `faer_mul / faer_solve / faer_svd` の PoC。以下の FaerMatrix の記述を main の公開済み機能と取り違えないこと。
- 現行作業ツリー: Julia 1.13.3 / RustCall.jl 0.9.1。361 件のテストが成功。
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
| 密行列の所有・構築 | `Mat<T>`, `MatRef/MatMut/MatUninitMut`, `from_fn`, `zeros`, `ones`, `identity`, `full` | `Matrix`, `zeros`, `ones`, `Matrix(I,m,n)`, `fill` | R: `FaerMatrix(A)`, `similar`, `zero`, `fill!` | 専用コンストラクタ群、Rust の全型への対応 |
| 行・列ベクトル | `Row / Col`, `RowRef / ColRef`, `RowMut / ColMut` | `Vector`, 行ベクトル、`view` | 一部: Julia Vector を行列積・求解の入出力に利用 | Rust 所有の独立した FaerVector / 行ベクトル型 |
| 対角行列 | `diag::{Diag, DiagRef, DiagMut}` | `Diagonal`, `diag`, `diagind` | — | 未: Rust 対角行列のラッパーと専用演算 |
| 読み書き・形状 | `nrows/ncols/shape`, 添字、`row/col` | `size/axes/getindex/setindex!` | R: shape・要素読み書き、J: 線形添字など | 行・列をまとめて借用する Rust ビュー |
| コピー・Julia との変換 | `clone/to_owned/copy_from` | `copy/deepcopy/copyto!/Matrix` | R: 独立したコピー、`Matrix(A)`、FaerMatrix 同士の `copyto!` | 借用 Julia バッファを使う恒久的なゼロコピー型 |
| 部分行列・分割・逆順 | `submatrix/subrows/subcols/split_at/reverse_rows/reverse_cols` | `view`, `@view`, slicing, `reverse` | J: `view` は SubArray として動作 | Rust MatRef/MatMut を保持する部分行列型、専用一括接続 |
| 転置・共役・随伴 | `transpose/conjugate/adjoint` | `transpose/conj/adjoint`, `A'` | J: 遅延転置/随伴、R: `copy(transpose(A)) / copy(A')` | 複素行列と共役処理、Rust の転置ビューを使う分解・積 |
| 動的容量・サイズ変更 | `with_capacity/try_reserve/reserve/resize_with/truncate` | `sizehint!` 等に類する容量管理。Matrix に直接同等の公開 API はない | — | 未: サイズ変更・容量管理の公開 API |
| 置換・行列交換 | `Perm/PermRef`, `permute_rows/cols`, `swap_rows/cols` | 置換ベクトル、`A[p,:]`, `A[:,p]` | J: 添字処理による操作の基盤 | 未: Rust 置換型、置換・交換ルーチンの専用接続 |
| スカラー型 | `f32/f64`, `c32/c64`、`fx128/cx128` の公開型エイリアス、generic field traits | `Float32/Float64/ComplexF32/ComplexF64` 等 | 一部: 実数入力を Float64 に変換 | Float32・複素数などの保持・演算。全ルーチンが全型対応との主張はしない |

## 2. 基本演算・集約・積

| 機能 | t4a-faer public API | Julia / LinearAlgebra 対応 | Colmat 実装済 | Colmat 未実装・制約 |
|---|---|---|---|---|
| 加算・減算・符号 | 行列演算、要素操作 | `A+B`, `A-B`, `-A` | R: FaerMatrix 同士の加減算 | 多型・複素数の promotion |
| スカラー倍・除算 | `Scale`、行列スカラー演算 | `α*A`, `A*α`, `A/α` | R: Real を Float64 に変換して実行 | 複素スカラー・型を保持する演算 |
| 行列積 | `linalg::matmul::{matmul,matmul_with_conj}` | `A*B`, `mul!` | R: FaerMatrix 同士の積、3/5 引数 `mul!`、alias 対応 | 複素共役指定、全 mixed-operand 組合せの専用接続。現行 `mul!` は一時結果を割り当てる |
| 行列ベクトル積 | 同上、Row/Col ビュー | `A*x`, `mul!(y,A,x)` | R: `A*x`。Vector は一時 Rust 行列へコピー、結果は Julia Vector | Rust 所有ベクトル、専用ベクトル `mul!` |
| 内積 | `matmul::dot::inner_prod` | `dot(x,y)` | J: `dot(A,B)` 動作確認 | 未: Rust inner_prod 直接接続、複素共役の検証 |
| 要素ごとの map / zip | `map/for_each`, `zip!/unzip!/Zip!` | `map`, `map!`, broadcast `f.(A)` | J: broadcast 動作確認 | Rust バルク処理の専用接続。要素単位 FFI、Julia Matrix の出力になる場合がある |
| ノルム | `norm_l2/squared_norm_l2/norm_l1/norm_max` | `norm(A)`, `sum(abs2,A)`, `norm(A,1/Inf)` | J: `norm(A)` 動作確認 | 未: 各 Rust reduction の接続。行列の `norm_l2` は Frobenius 系で `opnorm(A,2)` と区別 |
| 和・最大・最小・検査 | `sum/max/min/is_all_finite/has_nan/is_identity` | `sum/maximum/minimum`, `all(isfinite,A)`, `any(isnan,A)`, `isone` 相当 | J: `sum(A)` 動作確認、要素アクセス基盤あり | Rust 集約/検査の専用接続。他の汎用呼び出しは未検証 |
| 行列式 | `MatRef::determinant(par)` | `det`, 関連: `logdet/logabsdet` | J: `det/logabsdet` 動作確認 | 未: Rust determinant の接続。`logdet/logabsdet` が Rust に同形であるという意味ではない |
| トレース | diagonal ビュー＋集約で構成 | `tr(A)` | J: `tr(A)` 動作確認 | 専用 Rust バルク経路なし |
| Kronecker 積 | `MatRef::kron`, `linalg::kron::kron` | `kron(A,B)` | J: `kron(A,A)` 動作確認 | 未: Rust kron の接続 |
| 三角・構造指定の行列積 | `matmul::triangular` | 三角ラッパーとの積、BLAS の構造化積に相当 | — | 未: 専用接続 |

## 3. 密行列の分解・固有値

| 機能 | t4a-faer public API | Julia / LinearAlgebra 対応 | Colmat 実装済 | Colmat 未実装・制約 |
|---|---|---|---|---|
| 部分ピボット LU | `partial_piv_lu`, `PartialPivLu::{L,U,P}` | `lu(A)` | R: 正方行列の `A\B` 内部で使用。J: `lu(A)` も動作 | 未: Rust LU 分解を保持・返却する `lu(::FaerMatrix)` 専用実装 |
| 完全ピボット LU | `full_piv_lu`, `FullPivLu::{L,U,P,Q}` | 完全ピボット LU（標準 `lu(A)` と同一ではない） | — | 未: 分解型・専用 API |
| QR | `qr`, `Qr`, `Q_basis/Q_coeff/R/thin_R`, `compute_Q/compute_thin_Q` | `qr(A)`, `Matrix(F.Q)`, `F.R` | J: `qr(A)` 動作確認 | 未: Rust QR と Householder データを所有する分解型 |
| 列ピボット QR | `col_piv_qr`, `ColPivQr`, `P` | `qr(A,ColumnNorm())` | — | 未: Rust 分解・置換・ランク判定の専用 API |
| Cholesky LLᴴ | `llt`, `Llt`, `L` | `cholesky(Hermitian(A))` | J: SPD の `cholesky(A)` 動作確認 | 未: Rust LLT 接続、分解オブジェクト |
| ピボット付き Cholesky | `cholesky::llt_pivoting::{factor,solve,inverse,reconstruct}` | `cholesky(Hermitian(A),RowMaximum())` に相当 | — | 未 |
| LDLᴴ | `ldlt`, `Ldlt::{L,D}` | `ldlt` に相当。ただし Julia 標準の対象構造・ピボット規約と要照合 | — | 未 |
| ピボット付きブロック LDLᴴ | `lblt`, `Lblt::{L,B_diag,B_subdiag,P}` | `bunchkaufman(Hermitian(A))` に相当 | — | 未: ブロック D・置換規約も設計が必要 |
| thin / full SVD | `thin_svd/svd`, `Svd::{U,S,V}` | `svd(A;full=false/true)` | R: `svd(::FaerMatrix)`、thin/full、空行列、互換 `faer_svd` | 分解因子は Julia 配列。Rust 分解オブジェクトの保持、Julia の alg/tolerance キーワードは未対応 |
| 特異値のみ | `singular_values` | `svdvals(A)` | R: 特異ベクトルを作らず計算 | Float64 / Par::Seq に限定 |
| 自己随伴固有値分解・固有値のみ | `self_adjoint_eigen/self_adjoint_eigenvalues`, `SelfAdjointEigen` | `eigen/eigvals(Hermitian(A))` | — | 未: 専用接続・Hermitian/Symmetric 入力規約 |
| 一般固有値分解・固有値のみ | `eigen/eigenvalues`, `Eigen` | `eigen/eigvals(A)` | — | 未: 現行 FaerMatrix の通常呼び出しは MethodError を確認 |
| 自己随伴一般化固有値 | `self_adjoint_generalized_eigen/self_adjoint_generalized_eigenvalues` | `eigen/eigvals(Hermitian(A),Hermitian(B))` | — | 未 |
| 一般化固有値 | `generalized_eigen`, `GeneralizedEigen::{U,S_a,S_b}` | `eigen(A,B)`, `eigvals(A,B)` | — | 未: α/β 表現・無限固有値の取り扱いを含む |
| Schur・並べ替え | `evd::{hessenberg_schur_real/cplx,reorder_schur_real/cplx,eigenvectors_from_schur_*}` | `schur`, `ordschur`, Schur からの固有ベクトル | — | 未: `schur(A)` は現行 MethodError |
| 一般化 Schur / QZ | `gevd::{gen_hessenberg,qz_real,qz_cplx}`, `hessenberg_to_qz` | `schur(A,B)` | — | 未 |

## 4. 求解・逆行列・分解の再利用

| 機能 | t4a-faer public API | Julia / LinearAlgebra 対応 | Colmat 実装済 | Colmat 未実装・制約 |
|---|---|---|---|---|
| 正方連立方程式 | `Solve::solve/solve_in_place` | `A\B`, `ldiv!` | R: 正方 FaerMatrix の `\`、Julia 行列/Vector RHS、互換 `faer_solve` | 未: 専用 `ldiv!`、分解の再利用。毎回 LU を作る |
| 共役・転置・随伴の求解 | `solve_conjugate/solve_transpose/solve_adjoint` と in-place 版 | `conj(A)\B`, `transpose(A)\B`, `A'\B` | — | 未: 対応する Rust Solve 経路 |
| 右側からの求解 | `rsolve_*_in_place` | `B/A`, `rdiv!` | — | 未: 右除算の専用接続 |
| 最小二乗 | `SolveLstsq::{solve_lstsq,solve_lstsq_in_place,...}`（QR/SVD 等） | 長方形 `A\b`, `ldiv!` | — | 未: FaerMatrix 専用 `\` は正方行列のみ |
| 三角連立方程式 | `solve_{unit_}{lower,upper}_triangular_in_place`, `linalg::triangular_solve` | `LowerTriangular/UpperTriangular/UnitLowerTriangular/UnitUpperTriangular` の `\` | — | 未: Rust 三角 solver 接続 |
| 逆行列 | `DenseSolveCore::inverse`, `triangular_inverse::invert_*` | `inv(A)`, `inv(F)` | J: `inv(A)` 動作確認 | 未: Rust 分解型の inverse、三角 inverse の専用接続 |
| 疑似逆 | `Svd::pseudoinverse`, `SelfAdjointEigen::pseudoinverse`, `pseudoinverse_from_*[_with_tolerance]` | `pinv(A;atol,rtol)` | J: `pinv(A)` 動作確認 | Rust pseudoinverse 専用経路、許容誤差規約の対応 |
| 分解からの再構成 | `DenseSolveCore::reconstruct`、各分解の accessor | `Matrix(F)`, 因子の積 | 一部: SVD は Julia SVD を返す | Rust 分解を保持する LU/QR/Cholesky 等の再構成 |
| ランク・条件数 | SVD/QR などを組み合わせて構成（同名の高水準 API としては扱わない） | `rank/cond/opnorm` | J: 3 呼び出しとも小行列で動作確認 | 専用最適化・全 p/許容誤差オプションの検証 |

## 5. 疎行列

| 機能 | t4a-faer public API | Julia / LinearAlgebra 対応 | Colmat 実装済 | Colmat 未実装・制約 |
|---|---|---|---|---|
| CSC・CSR、所有・借用・構築 | `SparseColMat/SparseRowMat` と Ref/Mut、`Triplet/Pair` | SparseArrays の `SparseMatrixCSC/sparse`。CSR は直接同じ標準型ではない | — | 未: Rust の疎行列ラッパー・無 densify の変換 |
| 数値を持たない疎構造 | `SymbolicSparseColMat/RowMat` | sparsity pattern、symbolic analysis | — | 未 |
| 疎加減算・構造の結合 | `sparse::ops::{add,sub,add_assign,sub_assign,union_symbolic,...}` | 疎行列の `+/-` | — | 未 |
| 疎×疎・疎×密・密×疎 | `sparse::linalg::matmul::{sparse_sparse_matmul,sparse_dense_matmul,dense_sparse_matmul}` | sparse/dense の `*/mul!` | — | 未 |
| 疎 Cholesky | `sp_cholesky`, `SymbolicLlt/Llt`、低水準の疎 Cholesky | `cholesky(sparse(Hermitian(A)))` | — | 未: symbolic/numeric の分離と再利用 |
| 疎 LU | `sp_lu`, `SymbolicLu/Lu` | `lu(sparse(A))` | — | 未 |
| 疎 QR・最小二乗 | `sp_qr`, `SymbolicQr/Qr`, `SolveLstsq` | `qr(sparse(A))`, sparse `A\b` | — | 未 |
| 疎三角求解 | `sp_solve_{unit_}{lower,upper}_triangular_in_place` | sparse 三角ラッパーの `\` | — | 未 |
| Fill-reducing ordering | `sparse::linalg::{amd,colamd}` | 疎分解の並べ替え・ordering に相当 | — | 未: ordering を独立に公開する API |

## 6. Matrix-free・反復法

| 機能 | t4a-faer public API | Julia / LinearAlgebra 対応 | Colmat 実装済 | Colmat 未実装・制約 |
|---|---|---|---|---|
| 線形作用素・双方向作用素 | `matrix_free::{LinOp,BiLinOp}` | `mul!` を持つ作用素型相当。LinearAlgebra に同一の trait はない | — | 未 |
| 前処理 | `Precond/BiPrecond/IdentityPrecond` | 反復法の前処理（標準外） | — | 未 |
| 共役勾配法 | `conjugate_gradient::{conjugate_gradient,CgParams,CgInfo,CgError}` | CG solver（標準外） | — | 未 |
| BiCGSTAB | `bicgstab::{bicgstab,BicgParams,BicgInfo,BicgError}` | BiCGSTAB solver（標準外） | — | 未 |
| LSMR | `lsmr::{lsmr,LsmrParams,LsmrInfo,LsmrError}` | 最小二乗反復 solver（標準外） | — | 未 |
| 部分固有値・部分 SVD | `matrix_free::eigen::{partial_eigen,partial_self_adjoint_eigen,partial_svd}` と params/info/scratch | 部分スペクトルを求める Krylov 法（標準外） | — | 未 |

## 7. 低水準の数値プリミティブ

| 機能 | t4a-faer public API | Julia / LinearAlgebra 対応 | Colmat 実装済 | Colmat 未実装・制約 |
|---|---|---|---|---|
| 破壊的分解・作業領域管理 | 各分解モジュールの `*_in_place / *_scratch`、`temp_mat_scratch/temp_mat_zeroed` | `lu!/qr!/cholesky!/svd!/eigen!`、LAPACK workspace 相当 | — | 未: 真に入力ストレージを書き換える分解と caller-owned scratch |
| Householder | `householder::make_householder_in_place/apply_block_householder_*/apply_block_householder_sequence_*` | reflector・QR の Q 適用相当 | — | 未 |
| Jacobi/Givens 回転 | `JacobiRotation::{make_givens,rotg,apply_*,transpose,adjoint}` | `givens`, `Givens`, `lmul!/rmul!` | — | 未 |
| Hessenberg 化・balancing | `evd::hessenberg::hessenberg_in_place`, `evd::{balance_in_place,balance_eigenvectors_in_place,...}` | `hessenberg/hessenberg!`、LAPACK balancing 相当 | — | 未 |
| 対称三重対角化・三重対角 EVD | `evd::tridiag::tridiag_in_place`, `tridiagonal_self_adjoint_evd` | `SymTridiagonal`、構造化固有値分解 | — | 未 |
| 二重対角化・二重対角 SVD | `svd::bidiag::bidiag_in_place`, `bidiagonal_svd` | `Bidiagonal`、LAPACK bidiagonal SVD 相当 | — | 未 |
| Pfaffian | `linalg::pfaffian::{pfaffian,pfaffian_scratch,PfaffianError}` | 反対称行列の Pfaffian。LinearAlgebra に直接同名の高水準 API はない | — | 未 |

## 8. 並列実行・統計・I/O・補助 API

| 機能 | t4a-faer public API | Julia / LinearAlgebra 対応 | Colmat 実装済 | Colmat 未実装・制約 |
|---|---|---|---|---|
| 逐次実行 | `Par::Seq` | 同期・逐次の数値処理 | R: 全 native 数値処理 | thread budget の公開設定なし |
| 呼び出し側所有の並列 team | `Par::Team`, `team::with_team`, `Par::reborrow`（threadteam feature） | Julia task/thread や BLAS thread 設定と関心は近いが、直接同一ではない | — | 未: team 所有・再利用。現行 Cargo features では threadteam 無効 |
| 行・列の平均・分散 | `stats::{col_mean,row_mean,col_varm,row_varm,NanHandling}` | Statistics の `mean/varm` と dims/NaN 規約の対応 | — | 未: native 統計接続 |
| 乱数・直交/ユニタリ行列の生成 | `CwiseMat/Col/RowDistribution`, `ComplexDistribution`, `UnitaryMat`（rand feature） | Random の `rand/randn`、ランダム直交行列の生成相当 | — | 未: native 乱数生成。rand feature 無効 |
| NPY 読み込み・ビュー | `io::npy::{Npy,FromNpy,...}`, `as_aligned_ref/to_mat`（npy feature） | NPY 対応 I/O（Julia 標準外） | — | 未: I/O 接続。npy feature 無効 |
| 型・形状・stride・共役・蓄積指定 | `Shape/Stride/Idx/IdxInc/MaybeIdx`, `ContiguousFwd/Bwd`, `Conj/Accum/Side`, `Spec/Auto` | axes/strides、BLAS の共役・αβ・上下三角指定に相当 | 一部: サイズ、1-based 添字への変換、mul! の αβ | Rust の型レベル shape/stride、全制御項目の公開 |
| リテラル・連結・zip マクロ | `mat!/row!/col!`, `concat!`, `zip!/unzip!/Zip!` | 配列リテラル、`hcat/vcat`、broadcast/zip | Julia 配列から FaerMatrix を構築可能 | Rust マクロ自体を Julia に移植する API は未提供 |

## 確認結果からの注意

- `lu/qr/cholesky/inv/det/logabsdet/pinv/rank/cond/tr/norm/opnorm/dot/sum/kron/view/broadcast` は小さな Float64 FaerMatrix で動作するものを確認した。ただし **R と J を混同しないこと**。特に Julia 汎用 LU/QR/Cholesky は Rust の faer 分解ではなく、要素単位の FFI 呼び出しを伴い得る。
- `eigen/eigvals/schur` は同条件で MethodError を確認。既存 AbstractMatrix 継承だけで全 LinearAlgebra が動くわけではない。
- 現行 Rust dependency 設定は `default-features=false, features=["std","linalg"]`。API の public 性と、optional feature の有効化と、Julia 接続の有無は別の軸。
- `FaerMatrix` は Float64 のみ。入力の実数型を保持せず変換する。疎行列を FaerMatrix に渡せても密行列化されるため、疎行列サポートとは数えない。
- SVD の Rust `V()` は V、Julia の `F.Vt` は Vᴴ。置換は Rust が 0-based、Julia が 1-based。分解の追加時に規約を明示する。
- RustCall の阻害要因と回避策: [#564](https://github.com/AtelierArith/RustCall.jl/issues/564)、[#565](https://github.com/AtelierArith/RustCall.jl/issues/565)。

## 対応を進める際の候補

- [ ] 現行 FaerMatrix / svd / svdvals の変更をコミットし、main の状態と表を同期する
- [ ] native LU / QR / Cholesky の分解オブジェクトと `lu/qr/cholesky` を実装する
- [ ] `ldiv!`・分解の再利用・長方形最小二乗を実装する
- [ ] native `eigen/eigvals`、Hermitian / Symmetric、一般化固有値へ拡張する
- [ ] Float32 / ComplexF32 / ComplexF64 を追加する
- [ ] reduction / dot / norm / determinant / kron の native 経路を追加する
- [ ] sparse と matrix-free を別の所有型・作用素型として設計する
- [ ] threadteam と scratch の所有・再利用を公開する

## ソース

- [Julia との公式移行対応表（調査コミット固定）](https://github.com/tensor4all/faer/blob/d5be1473eed9314fca6f80ad8b4072d86282e890/docs/migration/julia.md)
- [高水準 dense solver API](https://github.com/tensor4all/faer/blob/d5be1473eed9314fca6f80ad8b4072d86282e890/faer/src/linalg/solvers.rs)
- [MatRef の変換・reduction・determinant・kron](https://github.com/tensor4all/faer/blob/d5be1473eed9314fca6f80ad8b4072d86282e890/faer/src/mat/matref.rs)
- [低水準 linalg 公開モジュール](https://github.com/tensor4all/faer/blob/d5be1473eed9314fca6f80ad8b4072d86282e890/faer/src/linalg/mod.rs)
- [疎行列 solver](https://github.com/tensor4all/faer/blob/d5be1473eed9314fca6f80ad8b4072d86282e890/faer/src/sparse/solvers.rs)
- [matrix-free 公開モジュール](https://github.com/tensor4all/faer/blob/d5be1473eed9314fca6f80ad8b4072d86282e890/faer/src/operator/mod.rs)
- [stats](https://github.com/tensor4all/faer/blob/d5be1473eed9314fca6f80ad8b4072d86282e890/faer/src/stats/mod.rs)
- Colmat 作業ツリーの調査対象: `src/Colmat.jl`, `src/matrix.jl`, `deps/rust/src/lib.rs`, `deps/rust/Cargo.toml`, `test/runtests.jl`, `test/matrix.jl`。
