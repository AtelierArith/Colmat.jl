# はじめに

[インストール](@ref)のビルドを済ませ、新しい Julia プロセスで実行してください。

## 通常の配列で計算する

```jldoctest
julia> using Colmat, LinearAlgebra

julia> A = [1.0 2; 3 4]; B = [2.0 1; 1 3];

julia> A * B == [4.0 7; 10 15]
true

julia> C = zeros(2, 2); mul!(C, A, B); C == A * B
true

julia> x = [2.0, 3.0]; A * x == [8.0, 18.0]
true

julia> dot(x, x) == 13.0
true
```

入力と結果は通常の `Matrix` / `Vector` です。行列を Colmat 専用型へ変換する
操作はありません。行列のメモリは引き続き Julia が所有します。

## 転送先を確認する

`BLAS.get_config()` はロード済みのライブラリを表示します。フォールバックを
保持するため、OpenBLAS などが Colmat と一緒に表示されるのは正常です。
ルーチンごとの転送先は次のように確認できます。

```jldoctest
julia> using Colmat, LinearAlgebra

julia> interface = sizeof(LinearAlgebra.BlasInt) == 8 ? :ilp64 : :lp64;

julia> lib = BLAS.lbt_find_backing_library("dgemm_", interface);

julia> realpath(lib.libname) == realpath(Colmat.backend_library())
true
```

この確認は倍精度 GEMM の転送先を調べます。他のルーチンの対応状況は
[BLAS 対応範囲](@ref)を参照してください。

## LAPACK を使う演算

```jldoctest
julia> using Colmat, LinearAlgebra

julia> A = [4.0 1; 1 3]; b = [1.0, 2.0];

julia> A * (A \ b) ≈ b
true

julia> F = svd(A); F.U * Diagonal(F.S) * F.Vt ≈ A
true
```

これらの LU・SVD ドライバーは colmat-rs の分解・求解へ接続します。
転送の境界は[バックエンドとフォールバック](@ref)で説明します。
