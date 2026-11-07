# API 案内

Julia 標準の演算は `using LinearAlgebra`、疎行列は `using SparseArrays`、
統計は `using Statistics`、乱数は `using Random` と組み合わせます。
以下は主要な入口の一覧です。引数、制約、実行例はリンク先のマニュアルにあります。
実装に付属する docstring は Julia のヘルプモード（例: `?FaerMatrix`）でも確認できます。

| 分野 | 主な型・関数 | 詳細 |
| :--- | :--- | :--- |
| 密配列 | `FaerMatrix`, `FaerVector`, `FaerRow`, `FaerDiagonal` | [配列](../manual/arrays.md) |
| ビュー・容量 | `view`, `split_at_row`, `split_at_col`, `split_at`, `strides`, `diagonalview`, `faerview`, `FaerBorrowedMatrix`, `FaerBorrowedSparseMatrix`, `capacity`, `with_capacity`, `reserve!`, `resize!`, `resize_with!`, `truncate!`, `close`, `isopen` | [所有権](../manual/arrays.md) |
| 置換 | `FaerPermutation`, `permute_rows!`, `permute_cols!`, `permute!`, `invpermute!`, `swap_rows!`, `swap_cols!`, `scratch_requirements(permute_rows!, A, p)` | [置換](../manual/arrays.md) |
| 密分解 | `lu`, `full_piv_lu`, `qr`, `cholesky`, `ldlt`, `bunchkaufman`, `factorize` と対応する `!` 形式 | [密分解](../manual/dense.md) |
| 求解・再構成 | `ldiv!`, `rdiv!`, `reconstruct`, `qr_q`, `inv`, `pinv`, `rank`, `cond`, `FaerSpectralFactor` | [求解](../manual/dense.md) |
| SVD | `svd`, `svd!`, `svdvals`, `svdvals!` | [SVD](../manual/dense.md) |
| 固有値問題 | `eigen`, `eigvals`, `eigvecs`, `hessenberg`, `schur`, `ordschur`, `balance` | [スペクトル](../manual/spectral.md) |
| 部分スペクトル | `partial_eigen`, `partial_self_adjoint_eigen`, `partial_svd` | [部分スペクトル](../manual/spectral.md) |
| 疎配列・構造 | `FaerSparseMatrix`, `FaerSparsity`, `compressed_storage`, `pattern_union`, `nonzeros`, `FaerSparseProductPlan` | [疎行列](../manual/sparse.md) |
| 疎分解・ordering | `FaerSparseSymbolic`, `FaerSparseFactorization`, `amd`, `colamd`, `factorize` | [疎直接法](../manual/sparse.md) |
| 疎 QR | `qr`, `qr!`, `ldiv!`, `rdiv!`, `svdvals`, `rank`, `pinv` | [疎 QR](../manual/sparse-qr.md) |
| 反復法 | `cg`, `bicgstab`, `lsmr` と対応する `!` 形式 | [反復法](../manual/operators.md) |
| 作用素・前処理 | `FaerLinearOperator`, `identity_preconditioner`, `diagonal_preconditioner` | [線形作用素](../manual/operators.md) |
| Householder | `FaerHouseholder`, `householder`, `FaerHouseholderSequence`, `reblock`, `reblock!` | [反射](../manual/reductions.md) |
| 回転・縮約 | `FaerJacobiRotation`, `rotg`, `make_givens`, `tridiagonalize`, `tridiagonal_eigen`, `tridiagonal_eigvals`, `bidiagonalize`, `bidiagonal_svd` | [回転・縮約](../manual/reductions.md) |
| リソース | `FaerScratch`, `scratch_requirements`, `scratch_capacity`, `FaerTeam`, `team_mul`, `FaerParThreshold` | [作業領域](../manual/resources.md) |
| 統計・乱数 | `mean`, `var`, `std`, `FaerRNG`, `randunitary`, `rand!(rng, A, X)` | [統計・乱数](../manual/utilities.md) |
| NPY | `FaerNpyMatrix`, `loadnpy`, `npyview`, `npyborrow`, `npyinfo` | [NPY](../manual/utilities.md) |
| 構造化積 | `structured_mul!`, 三角ラッパー・`Diagonal` との `*`/`mul!` | [密分解](../manual/dense.md) |
| 互換 API | `faer_mul`, `faer_solve`, `faer_svd` | [互換 API](../manual/arrays.md) |

`FaerFactorization` と `FaerSparseFactorization` はネイティブ因子を保持します。
`lu!/qr!/cholesky!(A; scratch)` が返す `FaerInPlaceFactorization` は入力 `A` の native 格納を packed 因子として参照します。
通常の Julia 配列を返す分解と Rust 所有因子を再利用する経路は、各ページの戻り値の説明で区別してください。
