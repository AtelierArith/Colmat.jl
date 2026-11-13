# 作業領域とスレッドチーム

このページは、同じ規模の計算を繰り返す際の性能調整を説明します。`scratch`は一時メモリ、`team`は計算スレッドの指定です。どちらも初めは省略して構いません。

これらの寿命を指定するときは [GCと資源の寿命](lifetimes.md)の `with_resource`と `FaerTeam`の `do` 構文を使えます。

## 再利用できるスレッドチーム

`FaerTeam(width)`は計算に使うスレッドを予約します。`width`は呼び出し元を含む人数で、1 なら逐次実行です。まずは通常の計算を使い、大きな計算を繰り返すときに検討してください。

`team_mul` や `mul!(C, A, B; team)`、対応する分解・求解の `team=`に同じチームを渡せます。`A * B` や `F \ B`、`F.Q`のような記法ではチームを指定できません。並列化したい場合は `mul!`、`ldiv!`、`rdiv!`、`qr_q`を使います。小さい問題や並列化に対応しない処理では、チームを渡しても逐次実行します。Colmat 内の呼び出しは所有者を守るロックでも直列化されます。

チームは Juliaのスレッドプールとは別に管理されます。`copy(team)`は独立した予約を作るため、単に再利用する場合は元のチームを渡してください。`do` ブロックは例外時も予約を解放します。

| 設定 | 意味 |
| :--- | :--- |
| `acquire_timeout` | 予約を待つ秒数。既定は 30 秒 |
| `spin_polls` | 待機時に空回りする回数の上限。`0`で早く休止する |
| `qos` | macOSの作業スレッドの優先度区分 |
| `pin` | Linuxで CPU 配置を固定するか。既定は `true` |
| `dispatch_cost` | 並列化の判断に使う起動コスト（秒）。大きいほど分割を控える |

`team_diagnostics(team)`で予約や待機の状況を調べられます。速度は行列の大きさや環境に依存するため、実際の処理で測定してください。

```julia
using Colmat, LinearAlgebra
FaerTeam(2) do team
    A = FaerMatrix([1.0 2; 3 4])
    B = team_mul(team, A, A)
    @assert Matrix(B) == [7.0 10; 15 22]
    mul!(A, team, A, A, 2.0, 0.0)
    @assert Matrix(A) == 2Matrix(B)
    C = zeros(2, 2)
    mul!(C, A, A; team)             # チームを使い C を直接更新
    F = lu(A; team)
    @assert Matrix(F; team) ≈ Matrix(A)
end # 例外時もスレッド予約を解放
```

```julia
using Colmat

FaerTeam(2; spin_polls=0, qos=:utility, pin=false, acquire_timeout=5,
         dispatch_cost=[2 => 2e-6, 4 => 4e-6]) do team
    A = FaerMatrix(rand(200, 200))
    team_mul(team, A, A)
    d = team_diagnostics(team)
    @assert d.width <= 2 && d.spin_polls == 0 && !d.pin && d.dispatch_cost > 0
end
```

## 並列化のしきい値

`FaerParThreshold()` または `FaerParThreshold(:derived)`は計算量から並列化を判断します。`FaerParThreshold(n)`は固定の下限、`:never`は縮約と明示的な基底の構築を逐次実行にします。

二重対角化では更新する残りの領域の要素数、三重対角化では残りの次数を `rem`として `rem*rem÷2`を下限と比較します。整数は `0 <= n < typemax(UInt)`が必要です。一般の Hessenberg 化にも対応する設定があります。

これ自体はチームを作りません。並列実行には開いた `FaerTeam`が必要です。作業領域の照会と実行では同じ設定を使い、変更時は必要量を照会し直してください。具体例は[反射・回転・縮約](reductions.md)にあります。


## 作業メモリの再利用

`FaerScratch`は Rustの計算に使う一時メモリです。同じ規模の計算を繰り返すときに、毎回の確保を減らすために使います。最初は `scratch`を省略して構いません。

`FaerScratch(関数, 引数...; 設定...)`で必要量を確保し、同じ設定で実行します。容量不足なら自動拡張せず、計算開始前に例外になります。入力や設定を変えた場合は `reserve!`で明示的に拡張してください。`copy`と `deepcopy`は独立したメモリを作ります。

この領域に返り値は保存されません。Julia 配列への変換、返り値、Rust 内部の追加メモリなどは別に確保されるため、呼び出し全体のメモリ確保がゼロになる保証はありません。下の例の `pfaffian`は歪対称行列に対する Pfaffianを計算し、入力を変更しません。

```julia
using Colmat
A = FaerMatrix([0.0 2; -2 0])
requirements = scratch_requirements(pfaffian, A)
workspace = FaerScratch(pfaffian, A)
@assert scratch_capacity(workspace) == requirements
@assert pfaffian(A; scratch=workspace) == 2.0
FaerTeam(2) do team
    @assert pfaffian(A; scratch=workspace, team=team) == 2.0
end
reserve!(workspace, requirements.bytes; alignment=requirements.alignment)
```

## 操作別のリソース対応

必要量の照会・構築・実行では同じ操作、入力、数値型、`team`、許容誤差を使います。
入力や teamを変えた場合は必要量を再照会し、必要なら `reserve!`で拡張してください。

| 操作 | 詳細 |
| :--- | :--- |
| 密分解・保存因子の求解・SVD、`lu!/qr!/cholesky!(A; scratch)`の格納領域を直接使う分解 | [密行列の分解と求解](dense.md) |
| 固有値・Schur・QZ・並べ替え | [固有値問題](spectral.md) |
| 疎 LU・Choleskyの構築・更新・求解、並べ替え | [疎直接法](sparse.md) |
| 疎 QRの構築・更新・最小ノルム求解 | [疎 QR](sparse-qr.md) |
| 作用素・前処理・CG・BiCGSTAB・LSMR | [線形作用素・反復法](operators.md) |
| Householder 適用・三重対角化・帯行列固有値 | [反射・回転・縮約](reductions.md) |
