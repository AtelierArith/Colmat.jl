# Colmat.jl

Colmat は Rust の colmat を使う Julia の BLAS / LAPACK バックエンドです。
`using Colmat` で libblastrampoline (LBT) に転送先を登録し、通常の Julia 配列と
`LinearAlgebra` の API から計算します。

```jldoctest
julia> using Colmat, LinearAlgebra

julia> A = [4.0 1; 1 3]; B = [2.0 0; 0 2];

julia> A * B == [8.0 2; 2 6]
true

julia> C = similar(A); mul!(C, A, B); C == A * B
true
```

## 読み進め方

- [インストール](@ref): 必要な環境と初回ビルド。
- [はじめに](@ref): 行列積、ベクトル演算、転送先の確認。
- [バックエンドとフォールバック](@ref): プロセス全体の切り替えと LAPACK の扱い。
- [スレッド設定](@ref): Colmat と既存バックエンドの並列実行幅。
- [LBT と colmat-rs の全ルーチン対応表](@ref): 全シンボルの接続候補と未対応 API。
- [BLAS 対応範囲](@ref): 型・整数 ABI・提供ルーチン。
- [Julia API の実測対応表](@ref): 実行結果と Colmat / 既存バックエンドの区別。
- [トラブルシューティング](@ref): ビルドやロードで困ったときの確認。

## 対象とする計算

Float32 / Float64 / ComplexF32 / ComplexF64、LP64 / ILP64 に対応します。
GEMM・GEMV と行列積を利用する Level 3 演算では Rust colmat のカーネルを使い、
ベクトル演算と三角求解の一部はこのパッケージの Rust 互換層で実行します。

`lu`・`svd`・一般／自己随伴の `eigen`・`cholesky` は LAPACK ABI を通じて
colmat-rs の分解へ接続します。未提供の BLAS / LAPACK は既存バックエンドを使います。
QR・一般化固有値・Schur など、Rust 側に機能があってもまだ未接続の系列があります。
公開 API の存在と実際の ABI 接続は、全ルーチン対応表で区別します。

[0.2 への設計変更](@ref)で専用行列型による旧 API を廃止しました。
