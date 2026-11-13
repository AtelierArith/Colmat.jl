# GCと資源の寿命

## 普通の計算では GCに任せる

Colmatの行列や保存因子は、Juliaの GC（不要なオブジェクトを回収する仕組み）に任せられます。RustCallが Rust 所有オブジェクトに finalizer（回収時の後始末）を登録しているため、単に `close`を書かなかったことが恒久的なメモリリークになるわけではありません。

```julia
using Colmat, LinearAlgebra
function solve_once(b)
    A = FaerMatrix([4.0 1; 1 3])
    A \ b
end
x = solve_once([1.0, 2.0])
```

ただし、参照が不要になった直後に GCが動く保証はありません。Rust 側の大きなメモリやスレッド予約が、Juliaの小さなラッパーの見た目に対応した時期に回収されるとも限りません。`GC.gc()`を各計算の後に呼ぶことを、通常の資源管理の代わりにはしません。

## 解放時期を決めたいときは `do` ブロック

`with_resource(構築関数, 引数...; キーワード...) do resource`は、資源を作ってブロックに渡し、正常終了・例外・途中の `return`のいずれでも `close`を呼びます。ブロックの返り値が全体の返り値になります。構築が失敗した場合はブロックを実行しません。

```julia
using Colmat, LinearAlgebra
A = FaerMatrix([4.0 1; 1 3])
F = with_resource(FaerScratch, svd, A) do scratch
    svd(A; scratch)
end
@assert F.U * Diagonal(F.S) * F.Vt ≈ Matrix(A)
```

上の例では SVDの返り値は作業領域から独立しています。作業領域だけを片付け、結果はそのまま使えます。ブロックから閉じられる資源そのものやそのビューを返すと、返したものは使えなくなります。入力を共有する直接分解も同様です。必要ならブロック内で独立したコピーを作ってください。

`FaerTeam`には専用の `do` 構文もあります。

```julia
using Colmat, LinearAlgebra
A = FaerMatrix([4.0 1; 1 3])
F = FaerTeam(2) do team
    lu(A; team)
end
@assert Matrix(A) * (F \ [1.0, 2.0]) ≈ [1.0, 2.0]
```

## 明示的な後始末が必要になる場面

**後続の処理へ予約スレッドを確実に譲る場合**は GC 待ちにできません。サーバーで複数のジョブを処理する例では、ジョブごとにチームを閉じます。`do` ブロックなら `close`を列挙する必要はありません。

```julia
using Colmat, LinearAlgebra
function solve_job(A, b)
    FaerTeam(2) do team
        F = lu(A; team)
        x = similar(b)
        ldiv!(x, F, b; team)
        x
    end  # 正常終了・例外のどちらでも、このジョブのスレッド予約を解放
end
```

**大きな一時メモリを次の処理の前に確実に返す場合**も、作業領域を明示的に閉じます。特に資源を参照する変数やキャッシュが残る処理では GCは回収できません。

```julia
using Colmat, LinearAlgebra
function singular_values_then_continue(A)
    values = with_resource(FaerScratch, svdvals, A) do scratch
        svdvals(A; scratch)
    end  # 一時メモリを解放してから、次の処理へ進める
    sum(values)
end
```

| 資源 | 現在の `close`の効果 |
| :--- | :--- |
| `FaerScratch` | Rustの作業バッファを解放し、以後の使用を禁止 |
| `FaerTeam` | スレッド予約を返し、以後の使用を禁止 |
| `FaerMatrix`・疎行列・疎分解など | Rust 所有者への参照を外し、以後の使用を禁止。所有者の実際の破棄は GCに委ねる |
| `npyborrow`の借用 | 明示的な `close`はなく、親のメモリマップなどを参照がなくなるまで保持 |

したがって、行列を `with_resource`で囲んでも「この行で必ず Rustの全メモリを OSに返す」保証はありません。作業メモリと予約スレッドの解放時期を制御できることとは区別してください。RustCallに GC 連携がないための制約ではなく、Colmatの各 `close`の実装と GCのタイミングによる違いです。

## 入力・計画・作業領域を閉じた後も分解を使う

次の例は、構造解析の再利用、CSR 入力での数値更新、資源の後始末を組み合わせています。仕組みを確認するための詳細例なので、最初の求解では `lu(A) \ b` から始めて構いません。

```julia
using Colmat, LinearAlgebra, SparseArrays
A0 = ComplexF64[0.1 4; 5 0.2im]
F = with_resource(FaerSparseMatrix, sparse(A0)) do A
    with_resource(FaerSparseSymbolic, A;
                  kind=:lu, algorithm=:supernodal, ordering=[2, 1]) do plan
        FaerTeam(2) do team
            with_resource(FaerScratch, factorize, plan, A; team) do scratch
                factor = factorize(plan, A; scratch, team)
                with_resource(FaerSparseMatrix, sparse(2A0); format=:csr) do B
                    reserve!(scratch, lu!, factor, B; team)
                    lu!(factor, B; scratch, team)
                end
                factor
            end
        end
    end
end
# 保存因子は入力・外部の計画・作業領域・チームから独立している
@assert F \ ones(ComplexF64, 2) ≈ (2A0) \ ones(ComplexF64, 2)
```

複数の資源は入れ子にすると内側から閉じます。各構築を対応するブロック内で行うため、途中の構築が失敗しても、すでに作った外側の資源は後始末されます。

`factorize(plan, A)`が返す疎分解は、因子と必要な計画・入力の情報を自身で保持します。外側の `plan` や Aを閉じても Fは利用できます。一方、`lu!(A; scratch)` などが返す直接分解は Aを共有するため、同じ使い方はできません。[密分解](dense.md)の所有権の説明を確認してください。
