"""
    with_resource(f, constructor, args...; kwargs...)

`constructor(args...; kwargs...)` で資源を作り、`f(resource)` の終了時に
`close(resource)` を呼びます。`do` ブロックを使えます。正常終了時はブロックの
返り値を返し、例外時も `finally` によって後始末します。構築自体が失敗した場合は
ブロックを実行しません。

通常の配列は GC に任せられます。作業メモリやスレッド予約の解放時期を
指定したいときに使ってください。解放の意味は資源の `close` に従います。
行列の `close` は参照を外す操作で、Rust メモリの即時回収を保証しません。

```julia
using Colmat, LinearAlgebra
A = FaerMatrix([4.0 1; 1 3])
F = with_resource(FaerScratch, svd, A) do scratch
    svd(A; scratch)
end
x = F \\ [1.0, 2.0]  # 分解は作業メモリから独立している
```

ブロック外で使う返り値は、閉じられる資源から独立している必要があります。
ビューや入力を共有する分解を返す場合は、必要に応じて `copy` してください。
複数の資源はブロックを入れ子にすると、構築と逆の順序で閉じられます。
"""
function with_resource(f, constructor, args...; kwargs...)
    resource = constructor(args...; kwargs...)
    try
        return f(resource)
    finally
        close(resource)
    end
end
