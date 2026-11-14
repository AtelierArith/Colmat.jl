# トラブルシューティング

## 共有ライブラリが見つからない

`Colmat's BLAS library is missing` と表示されたら、対象の Julia プロジェクトで
ビルドし、Julia を再起動してください。

```julia
using Pkg
Pkg.build("Colmat")
```

ビルドが失敗する場合は、リポジトリ利用時の `deps/build.log` を確認します。
Git・Cargo・OS の開発ツールが実行可能であることも確認してください。

## Rust ソースのコミットが一致しない

ビルドは checkout の実際の HEAD と `deps/upstream.toml` を照合します。
次のコマンドは状態の確認だけを行います。

```sh
git -C deps/colmat-rs rev-parse HEAD
git -C deps/colmat-rs status --short
cat deps/upstream.toml
```

ローカル変更がある場合は保存してから、指定 revision のクリーンな checkout を
用意してください。意図的な依存更新の手順は[開発](@ref)を参照してください。

## OpenBLAS も表示される

`BLAS.get_config()` に複数のライブラリが出るのは、未提供の BLAS と LAPACK の
転送先を残しているためです。GEMM の実際の転送先は[はじめに](@ref)の方法で
確認できます。ライブラリの表示順だけで全ルーチンの実行先は判断できません。

## 別のバックエンドをロードした後に戻したい

並列計算を止めてから `Colmat.load_backend()` を呼びます。
[バックエンドとフォールバック](@ref)に切り替え範囲と順序の説明があります。

## スレッド数が一致しない

`Colmat.get_num_threads()` は Colmat の設定値、`BLAS.get_num_threads()` は LBT が
認識するライブラリ全体の値です。他のバックエンドも揃える場合は
`BLAS.set_num_threads(n)` を使います。詳細は[スレッド設定](@ref)を参照してください。

## 旧版の名前が見つからない

`FaerMatrix`・`FaerScratch`・`FaerTeam`・`with_resource` などは 0.2 で削除しました。
通常の `Matrix` / `Vector` と `LinearAlgebra` を使用してください。
旧 API の対応方針は[0.2 への設計変更](@ref)を参照してください。
