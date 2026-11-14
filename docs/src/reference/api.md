# API

Colmat は名前を export しません。行列演算には Base / LinearAlgebra を使用します。
設定用の公開関数は `Colmat.` を付けて呼び出します。

| 関数 | 用途 |
|---|---|
| `Colmat.load_backend(; verbose=false)` | 提供 BLAS を LBT に登録し直す |
| `Colmat.backend_library()` | BLAS 共有ライブラリの絶対パスを返す |
| `Colmat.get_num_threads()` | Colmat の設定済み実行幅を返す |
| `Colmat.set_num_threads(n)` | Colmat の行列積の実行幅を設定する |

ロード時に `load_backend()` が実行されるため、通常の利用では明示的な登録は不要です。
[バックエンドとフォールバック](@ref)と[スレッド設定](@ref)も参照してください。

```@docs
Colmat
Colmat.load_backend
Colmat.backend_library
Colmat.get_num_threads
Colmat.set_num_threads
```
