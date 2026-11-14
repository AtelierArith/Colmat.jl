# スレッド設定

## Colmat とフォールバックをまとめて設定する

```julia
using Colmat, LinearAlgebra
BLAS.set_num_threads(2)
```

LBT に Colmat のスレッド設定関数を登録しているため、この標準 API から Colmat と
スレッド設定に対応するロード済みライブラリを設定できます。

## Colmat だけを設定する

```jldoctest
julia> using Colmat

julia> saved = Colmat.get_num_threads();

julia> Colmat.set_num_threads(2); Colmat.get_num_threads()
2

julia> Colmat.set_num_threads(saved)
```

初期値は1です。`Colmat.set_num_threads(n)` は `1:typemax(Cint)` の整数を受け取り、
範囲外には `ArgumentError` を返します。
`Colmat.get_num_threads()` は Colmat の設定値を返します。
`BLAS.get_num_threads()` はロード済みライブラリ全体を対象にするため、
Colmat だけを変更した場合は値が一致しないことがあります。

## 実行幅とプール

設定値は行列積の並列実行幅で、呼び出し側を含む参加者数です。
ベクトル演算や三角求解のすべてを並列化する設定ではありません。

各呼び出し元 OS スレッドがプールを遅延生成します。プールの実行幅が設定値と
異なれば、次の並列経路で古いプールを解放して作り直します。
実行幅1では逐次経路を使います。プール作成やチーム取得が失敗した場合も逐次経路へ
戻るため、設定値が常に実際の並列参加数になるとは限りません。

複数の Julia スレッドから同時に行列積を呼ぶと、各呼び出し元がそれぞれ参加者を
確保します。外側の Julia 並列処理と組み合わせる際は、Colmat の実行幅も考慮して
設定してください。設定変更は進行中の計算を作り直す操作ではありません。
