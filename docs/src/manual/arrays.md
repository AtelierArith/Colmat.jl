# 配列・型・所有権

まず、行列には `FaerMatrix`、列ベクトルには `FaerVector`を使います。`Matrix(A)`と `Vector(v)`で通常の Julia 配列へ戻せます。これらは独立したコピーです。`view(A, :, 1)`はコピーせず元の値を参照するため、変更も共有します。

| 型 | Julia から見た形状 | 用途 |
| :--- | :--- | :--- |
| `FaerMatrix` | 2 次元の `AbstractMatrix` | 通常の行列 |
| `FaerVector` | 1 次元の `AbstractVector` | 列ベクトル |
| `FaerRow` | 1×nの行列 | 共役を取らない行ベクトル |
| `FaerDiagonal` | 対角行列 | 対角成分だけを保存 |
| `faerview(J)` | 元の Julia 配列を借用 | 入力のコピーを避けたい場合 |

普通の計算は GCに任せられます。後始末の `do` 構文と入力共有の注意点は [GCと資源の寿命](lifetimes.md)を参照してください。以下は基本操作の後に、混合演算・格納形式・性能上の詳細を説明します。

## 密行列と基本演算

`FaerMatrix{T}`は Rustの `faer::Mat<T>`を所有する `AbstractMatrix{T}`
です。構築時に入力をコピーし、その後の行列同士の加減算・スカラー倍・行列積・
LU 求解は Rust 内のバッファを使います。これらの結果も独立した `FaerMatrix`になります。
`svd(A)`・`svdvals(A)` も Rust 側の行列を直接使い、分解結果を Julia 配列で返します。

`FaerMatrix{T}(A)`は入力を `Matrix{T}`に一度変換してコピーします。
ユーザー定義の変換が異なる格納型や入力と異なる形状を返した場合は、
Rustへデータを渡す前に例外にします。

```julia
using Colmat, LinearAlgebra

A = FaerMatrix([4.0 1; 1 3])
B = FaerMatrix([1.0 2; 3 4])
C = 2A + B - A / 2
D = A * B
X = A \ B
x = A \ [1.0, 2.0]         # ベクトルの右辺には Julia Vector を返す
y = A * [1.0, 2.0]         # 行列ベクトル積は FaerVector
@assert Matrix(X) ≈ Matrix(A) \ Matrix(B)

C[1, 2] = 5
mul!(C, A, B)              # C を更新。C === A / C === B も可
mul!(C, A, B, 2.0, 3.0)   # C = 2A*B + 3C
copyto!(C, A)
Z = zero(A)
E = copy(A)               # Rust の行列を複製
snapshot = Matrix(A)      # Julia へ独立したコピーを取得
```

### Julia 配列・転置・ビューとの混合演算

Rust 側の行列・範囲ビューとその `transpose` / `adjoint`が一方に含まれる `+` / `-` / `*`
は、もう一方が ネイティブ・`faerview`の借用・要素重複のない Julia ストライドd 配列
（`Matrix`・ストライドd `SubArray`・その転置／随伴）のいずれでも、Julia 側を借用配列として
その場で読み、結果を新しい Rust 側の所有者に書きます（入力の スナップショット・Julia 経由の
一時配列なし）。結果の型は「ネイティブが関与すれば ネイティブ」で、`A * J`・`A' * J`・
`A * J'`は `FaerMatrix`、`FaerMatrix`・ビュー・`A'`と Julia ベクトルの積 `A * x`は
`FaerVector`、`r::FaerRow * B`は `FaerRow`（`r * x`は1要素の `FaerVector`）、
`v' * B` / `transpose(v) * B`（`v::FaerVector`）は `FaerVector`の随伴／転置です。
異なるスカラー型は `promote_type`に従い、狭い側だけを Rustの `view_convert`が
記述子 から直接昇格コピーします。ビュー・遅延転置のスカラー倍 `x * V` / `V / x` /
`-V` も Rust 側でコピーしてからスケールします。昇格型が ネイティブでない（`BigFloat`
など）場合と、Julia 配列だけ・`faerview`と Julia 配列だけの組合せは従来どおり Juliaの
結果（`Matrix` など）です。

`mul!(y, A, x[, α, β])`（`y`が `FaerVector`・ビュー・Julia／借用ベクトル）と
`mul!(r, x, B[, α, β])`（`r`が `FaerRow`、`x`が `FaerRow`・`v'`・`transpose(v)`）は、
行列積と同じく faerが `y` / `r`へ直接書きます（`A`の転置／随伴、Julia 行列・
ベクトルの混在を含む。`y` / `r`と重なる入力だけ スナップショット。`team=` も同じ）。

```julia
J = rand(3, 3)
A = FaerMatrix(rand(3, 3))
@assert A * J isa FaerMatrix
@assert A' + J' isa FaerMatrix
@assert A * rand(3) isa FaerVector
r = FaerRow(rand(3))
@assert r * J isa FaerRow
mul!(r, r, A)                         # r = r * A（r は先に snapshot）
```

## ベクトル・行・対角行列とビュー

Rustが所有する列ベクトル・行ベクトル・対角行列も使えます。

```julia
using Colmat, LinearAlgebra

A = FaerMatrix([4.0 1; 1 3])
v = FaerVector([1.0, 2.0])
w = A * v                       # Rust の Col を所有する FaerVector
x = lu(A) \ v                    # 結果も FaerVector
@assert Vector(w) ≈ Matrix(A) * Vector(v)
@assert Vector(x) ≈ Matrix(A) \ Vector(v)
@assert v' * v ≈ dot(Vector(v), Vector(v))
r = FaerRow([1.0, 2.0])          # 共役しない 1×2 行列
D = FaerDiagonal([2.0, 3.0])
filldiagonal!(D, 4)
@assert Matrix(D) == Matrix(Diagonal([4.0, 4.0]))

V = view(A, 2:-1:1, :)           # 逆順・間引きの範囲を扱える
@assert Matrix(V' * V) ≈ Matrix(V)' * Matrix(V)
copyto!(view(A, :, 1), v)         # 重複する領域からのコピーにも対応
fill!(diagonalview(D), 5)         # 対角の格納領域を借用
@assert Matrix(A[[2, 1, 2], :]) == Matrix(A)[[2, 1, 2], :]  # 指定位置を Rust 側で一括取得
A[[2, 1], [true, false]] = [7.0, 8.0]                         # 指定位置へ Rust 側で一括代入
top, bottom = split_at_row(A, 1)  # faer の split_at_row に対応する native ビュー
@assert Matrix(bottom) == Matrix(A)[2:2, :]
@assert strides(A)[1] == 1 && stride(A, 2) >= size(A, 1)  # padding を含む列 stride
reserve!(A, 8, 8)                # 容量確保後も V は有効
@assert all(capacity(A) .>= (8, 8)) # alignment padding を含む native 容量
reserve!(v, 8)
@assert capacity(v) >= 8
@assert isopen(V)
close(V)                        # A 自体は閉じない

reserved = with_capacity(FaerMatrix{ComplexF64}, 8, 9)
@assert size(reserved) == (0, 0)
@assert all(capacity(reserved) .>= (8, 9))
resize!(reserved, 2, 3; fill=1+im)
@assert Matrix(reserved) == fill(1+im, 2, 3)
resize_with!(reserved, 4, 5) do i, j
    complex(i, j)
end
@assert Matrix(reserved)[1:2, 1:3] == fill(1+im, 2, 3)
@assert reserved[4, 5] == 4+5im
close(reserved)

from_function = FaerMatrix{ComplexF64}((i,j)->complex(i,j), 2, 3)
from_function_vector = FaerVector{Float32}(i->i^2, 4)
from_function_row = FaerRow(i->2i, 3)
@assert Matrix(from_function) == [complex(i,j) for i in 1:2, j in 1:3]
@assert Vector(from_function_vector) == Float32[1,4,9,16]
@assert Matrix(from_function_row) == [2.0 4.0 6.0]
close(from_function)
close(from_function_vector)
close(from_function_row)

native_zeros = zeros(FaerMatrix{Float32}, 2, 3)
native_ones = ones(FaerRow{ComplexF64}, (3,))
native_full = FaerDiagonal{ComplexF32}(3; fill=1-2im)
@assert Matrix(native_zeros) == zeros(Float32, 2, 3)
@assert Matrix(native_ones) == ones(ComplexF64, 1, 3)
@assert Vector(native_full) == fill(ComplexF32(1-2im), 3)
```

`view`は親の所有者を保持し、ネイティブ呼び出しの間だけ Rustの
`MatRef` / `MatMut`を作ります。部分行列・行・列・入れ子の範囲ビューに
対応します。親の形状を変える `resize!` / `truncate!`は既存のビューを
失効させます。親を閉じた場合もアクセスは例外になります。
`copy(view(...))`は独立した Rust 所有行列・ベクトルを返します。
ビューへの `copyto!`は Rust 所有型・ビューの入力を一時コピーし、代入先と重なる
場合も値を保ちます。同じ要素型の Julia ストライドd 配列（`Matrix`、範囲・逆順の
`view` など）は一時コピーせず、Juliaのメモリを直接読みます（形状が一致する
所有型への `copyto!` も同様）。
`FaerMatrix` / `FaerRow`への Julia 配列・ビュー・疎行列などからの `copyto!` も同じ
一括代入経路を使い、要素毎の `setindex!`は行いません。形状が異なっても要素数が
収まれば Baseと同じく線形順にコピーし、長すぎる入力は `BoundsError`です。ただし
`FaerMatrix` 同士のネイティブコピーは従来どおり形状一致が必要で、`DimensionMismatch`
になります。要素型の変換失敗・`close` 済みの代入先はいずれも書き込み前に例外となり、
代入先は変わりません。
ネイティブ 型の入力をビューへ代入する `copyto!(view, native)`は、代入先が重複しない
ビューなら入力を一時コピーせず直接読みます（同じ所有者で重なる場合のみ Rustが
スナップショット します）。

`FaerMatrix`・`FaerRow`・`FaerVector`とそのビューの `A[I, J]` / `A[I, J] = B`は、
整数ベクトル・範囲・`:`・Bool マスク（重複添字を含む）を Rustの1回の
指定位置の一括取得／代入で処理します。結果の型は Baseと同じ規則（行列結果は ネイティブ
所有型、行列から取り出したベクトルは Julia `Vector`）です。代入は Baseと同じ形状
検査を行い、入力は書き込み前に要素型へ変換し、代入先と重なる ネイティブ 入力
（`A[:, p] = A` など）は先に スナップショット します。重複添字への代入は列優先で後の値が
残ります。行列の線形添字（`A[mask]`、`A[[1, 5]]`）は1回の一括読み出し後に Julia
で添字付けします。整数ベクトル・マスクの `view(A, I, J)`は Juliaの `SubArray`
のままですが、`copy`・`Array`・`copyto!`・`S .= B`・`fill!`は同じ ネイティブ
指定位置の一括取得／代入を使います。

`split_at_row(A, k)`・`split_at_col(A, k)`・`split_at(A, i, j)`は faerの同名 API
に対応し、`k` 行（列）目までと残りの ネイティブビューを返します（`0 <= k <= size`）。
ベクトルは `split_at_row(v, k)`で分割します。`strides(A)` / `stride(A, d)`は
faerが報告する実際の ストライドを返します。所有型は列優先で、列 ストライドは
配置調整用の余白を含む行容量です（`reserve!`で変わります）。ビューは間引き・
逆順・行ビューの ストライドを反映します。Rust 側の記憶域は Julia から直接アドレス
指定できないため、`pointer`と `Base.elsize`は定義しません。

`faerview(A)`は Juliaの ストライドd ベクトル・行列を借用する `FaerBorrowedMatrix` /
`FaerBorrowedVector`を返します。`A`を保持し、各ネイティブ呼び出しの間だけ
`pointer(A)`・`size(A)`・`strides(A)` から faerの `MatRef`を作るため、負や
非単位の ストライド もコピーせずに読みます。ポインタ・形状は保存しないので、親の
`Vector`を `resize!` しても次の呼び出しは現在の長さとアドレスを使います。
要素型は Float32 / Float64 / ComplexF32 / ComplexF64、親は `Array` / `Memory`に
限り、ストライド 0 や重なる ストライドは `ArgumentError`です。`copy` / `FaerMatrix`、
`+` / `-` / `*`（転置・随伴、Rust 所有行列やビューとの混在、型昇格を含む）、
`FaerVector`への `mul!`、`mean` / `var`、`dot` / `norm(A,1/2/Inf)` / `sum` /
`all(isfinite,·)` / `any(isnan,·)`、`copyto!`の入力が借用のまま動作し、
`view` も借用を保ちます。添字・`strides`・`pointer`は親へ転送します。

借用配列と、要素型が同じで要素の重複しない Julia ストライドd 配列は Rust 側の
書き込み先にもなります。`mul!(C, A, B[, α, β])`（A/Bの少なくとも一方が ネイティブ か
借用、転置・随伴を含む）、ネイティブ 入力からの `copyto!`、Householder 反射・列・省メモリ形式の
Hessenberg／三重対角／二重対角基底の `lmul!` / `rmul!` / `mul!`、`FaerLinearOperator`の
`mul!(Y, op, X[, α, β])`は、呼び出し中だけ `pointer`/`strides` から faerの `MatMut`を
作って結果を直接書き込み、一時結果や Juliaへの書き戻しを作りません。入力と
書き込み先のアドレス範囲が重なる場合（`mul!(C, C', C)`、同じ配列のビューなど）は
その入力だけを Rust 側で スナップショット してから書き込みます。`β == 0`では Cを読まず、
`α == 0`では A/Bを読みません。要素型が異なる場合は入力を昇格し、Cが昇格型を
表せない場合は結果を全て計算・変換してから書き込みます（`InexactError` なら Cは
変わりません）。ストライド 0のビューなど要素が重複する書き込み先は Juliaの経路です。

借用配列は密分解（`lu`・`qr`・`cholesky`・`ldlt`・`bunchkaufman`・`svd`・`eigen` など）の
入力にもなり、faerが親配列を直接読んで Rust 側の分解を返します。保存因子による
`ldiv!(F, B)`は借用配列・Julia ストライドd 配列の `B`で直接 格納領域内での更新に解きます
（[密行列の分解と求解](dense.md)を参照）。

```julia
J = rand(4, 3)
B = faerview(J)                       # コピーしない
P = B' * FaerMatrix(J)                # Julia のメモリを直接読む native 積
J[1, 1] = 0                           # 以後の呼び出しに反映
@assert copy(B)[1, 1] == 0
copyto!(view(FaerMatrix{Float64}(undef, 4, 3), :, :), view(J, :, :))
C = zeros(3, 3)
mul!(C, B', FaerMatrix(J), 2.0, 0.0)  # C に直接書き込む
copyto!(J, FaerMatrix(ones(4, 3)))    # Julia のメモリへ1回で代入
```

`FaerDiagonal`の非対角要素は暗黙のゼロで、非ゼロの代入はできません。

対角との積・除算・和は faerの `DiagRef`を借用して Rust 側で計算し、
n×nの密な スナップショットを作りません。`D*A` / `A*D` / `D\A` / `A/D` / `D±A`、
`D*v` / `D\v`、`r*D` / `r/D`（結果は `FaerRow`）、`inv(D)`、
`lmul!` / `rmul!` / `ldiv!` / `rdiv!`に対応します。Juliaの `Diagonal` も、
格納された n 個の要素だけを渡して同じ経路を使います。
`FaerDiagonal`と Julia ストライドd 行列・ベクトル（`Matrix`・ストライドd `SubArray`・その
転置／随伴）の `D*M` / `M*D` / `D\M` / `M/D` / `D±M` / `D*x` / `D\x` も Julia 側を
その場で読み、結果は `FaerMatrix` / `FaerVector`です（以前は `Matrix` / `Vector`）。
Juliaの `Diagonal`と Julia 配列だけの組合せは Juliaの結果です。
ゼロの対角要素で割ると `SingularException`を投げ、入力も代入先も変更しません。
`diag(A)`と `FaerDiagonal(A)`は密行列・範囲ビューの主対角を Rust 側でコピーし、
`v * w'`と `v * r::FaerRow`は Rust 所有の `FaerMatrix`を返します。

`FaerDiagonal`と `FaerDiagonal` / Julia `Diagonal`の `+`・`-`・`*`・`\`・`/`は
`FaerDiagonal`を返します（Juliaの `Diagonal` 同士の演算が `Diagonal`を返すのと同じ）。
`transpose(D)`と `D'`は `D`（複素数では O(n)の共役コピー）として同じ ネイティブ 経路を
使い、`v'*D`・`transpose(v)*D`・`v'/D`は Rust 側の `FaerVector`を包む
`Adjoint` / `Transpose`を返します。`mul!(y, D, x)`と `mul!(z, r, D)`は
`FaerVector`・ベクトルビュー・`FaerRow`の出力先へ直接書き込みます。
`all(isfinite, x)` / `any(isnan, x)`は格納要素に faerの `is_all_finite` /
`has_nan`を使います。`map(f, D)`は格納要素を1回で読み、`f(0)==0` なら
`FaerDiagonal`、そうでなければ密な `FaerMatrix`を返します（`Diagonal`と同じ規則）。
`FaerRow ± v'`（`transpose(v)` も）は `FaerRow`を返します。

```julia
using Colmat, LinearAlgebra

A = FaerMatrix([1.0 2; 3 4])
D = FaerDiagonal([2.0, 4.0])
@assert Matrix(D * A) == Diagonal([2.0, 4.0]) * [1.0 2; 3 4]
@assert Matrix(FaerRow([1.0, 1.0]) / D) == [0.5 0.25]
ldiv!(D, A)                      # A の格納領域を直接更新
@assert Matrix(A) == [0.5 1; 0.75 1]
@assert Vector(diag(inv(D))) == [0.5, 0.25]
@assert Vector(diag(A)) == [0.5, 1.0]
@assert (D * Diagonal([1.0, 2.0])) isa FaerDiagonal
@assert Vector(diag(D \ Diagonal([4.0, 4.0]))) == [2.0, 1.0]
@assert collect(FaerVector([1.0, 1.0])' * D) == [2.0 4.0]
@assert map(x -> 2x, D) isa FaerDiagonal && all(isfinite, D)
try
    FaerDiagonal([1.0, 0.0]) \ A
catch e
    @assert e isa SingularException && e.info == 2
end
```

`FaerMatrix{T}(m,n; fill=value)`と `FaerVector{T}(n; fill=value)` /
`FaerRow{T}(n; fill=value)` / `FaerDiagonal{T}(n; fill=value)`は定数値から
Rust 側で直接構築し、Juliaの密配列を実体化しません。fillの既定値はゼロです。
`zeros` / `ones`に所有型を指定する形式もあり、行列は `(m,n)`、コレクションは
`(n,)`の タプル 指定にも対応します。型パラメータを省略すると Float64です。
`ones(FaerDiagonal{T},n)`は格納対角を1にし、非対角成分は暗黙のゼロです。

`FaerMatrix{T}(f,m,n)` / `FaerVector{T}(f,n)` / `FaerRow{T}(f,n)`は
関数または 呼び出し可能 オブジェクト から構築します。1 始まりの添字を渡し、要素ごとに
1回だけ、行列では列優先順に評価します。空の領域では関数を呼びません。
`{T}`を省略すると Float64 なので、複素数の結果には複素型を明示してください。

`capacity(A)`は所有行列の実際の行・列容量を組で返し、`capacity(v)` /
`capacity(r)` / `capacity(D)`は格納要素の容量を整数で返します。
予約した最小値より 配置調整用の余白の分だけ大きい場合があります。
ビューには容量照会を定義していません。対角の容量を照会しても、固定の次元を
変更できるようにはなりません。容量は Rust 側の所有者 から読み取り、Julia 側で
予約値や表示サイズから推測しません。

容量管理の対応範囲は次のとおりです。

| faer API | Colmatの対応・制約 |
|---|---|
| `Mat::with_capacity` | `with_capacity(FaerMatrix{T}, rows, cols)`は次元 `(0,0)`の所有行列を予約済み容量で作ります。`FaerVector{T}` / `FaerRow{T}`と1個の容量を指定する形式もあり、`{T}`を省略すると Float64です。 |
| `try_reserve` / `reserve` | `reserve!` / `sizehint!`は `try_reserve`を使い、失敗を例外として返します。 |
| `resize_with` | `resize!(...; fill=value)`は固定値で初期化します。`resize_with!(f,A,rows,cols)` / `resize_with!(f,x,length)`は新規要素だけを添字から生成し、重なる領域を保持します。 |
| `truncate` | `truncate!`は縮小のみを受け付け、容量を保持します。 |
| `row_capacity` / `col_capacity` / `capacity` | 所有型の `capacity`で実際の容量を照会します。 |
| unsafe `set_dims` | 初期化済み領域の証明を呼び出し側に要求するため、Juliaの公開 APIとしては提供していません。安全な `resize!`は新しい要素を初期化します。 |

`resize_with!`の関数は1 始まりの添字を受け取り、新規要素だけを列優先順に
評価します。do-ブロック 形式も使えます。縮小のみ・形状不変なら関数を呼びません。
新規値を Juliaの一時 プリミティブ型 バッファに評価・型変換してから Rust 側で渡すため、
関数の例外や変換失敗では入力を更新しません。関数自身による 所有者の変更・`close`は
禁止で、関数自身の副作用は巻き戻せません。形状・所有者が変わった場合は拒否します。
ネイティブ 側も元形状・スカラー型・Row/Colの種類・バッファ 長を検証し、`truncate!` 前に
必要容量を確保します。成功時の形状変更は既存ビューを失効させます。

`Row` / `Col`は予約とサイズ変更に対応します。faer `Diag`に予約・サイズ変更の
公開メソッドはなく、Colmatでも対角の次元は固定です。対応範囲と検証は
[Issue #90](https://github.com/AtelierArith/Colmat.jl/issues/90)で追跡します。

### Kronecker 積

`kron(A,B)`は `FaerMatrix`・ストライドd／逆順の `FaerMatrixView`・`FaerVector`／
`FaerRow`とそのビュー・`faerview`の借用 Julia 配列・それらの転置／随伴、
および ネイティブと混在する Julia ストライドd 配列を ビュー記述子でその場で読み、
faer `linalg::kron::kron`で新しい Rust 側の所有者に書きます（入力の スナップショット なし）。
形状は LinearAlgebraに合わせ、`kron(v,w)`は `FaerVector`、`kron(v',w')`は
`kron(v,w)'`、`kron(FaerRow,FaerRow)`は `FaerRow`、それ以外は `FaerMatrix`です。
異なるスカラー型は 型の昇格 し、変換が必要な入力だけをコピーします。
`kron!(C,A,B)`は `FaerMatrix`・`FaerVector`・それらのビューへ積を直接書き、
`C`と 所有者を共有する入力だけを先に スナップショット します。faerの `kron`は `Par`
を取らないため Seq/Teamの指定はありません。

```julia
using Colmat, LinearAlgebra

A = FaerMatrix([1.0 2; 3 4])
v = FaerVector([1.0, 2.0])
@assert kron(view(A, :, 2:-1:1), [1.0 0; 0 1]) isa FaerMatrix
@assert Vector(kron(v, v)) == [1.0, 2.0, 2.0, 4.0]
C = FaerMatrix(zeros(4, 4))
kron!(C, A, A')
@assert Matrix(C) == kron(Matrix(A), Matrix(A)')
```

## 置換・ランク・疑似逆行列

置換も Rustに保持できます。添字は Juliaと同じ 1 始まりです。

```julia
using Colmat, LinearAlgebra

A = FaerMatrix([1.0 2 3; 4 5 6])
p = FaerPermutation([2, 1])
q = FaerPermutation([3, 1, 2])
@assert Matrix(A[p, q]) == Matrix(A)[Vector(p), Vector(q)]
permute_rows!(A, p)
permute_cols!(A, q)
swap_cols!(A, 1, 3)
@assert Vector(p * invperm(p)) == [1, 2]

B = FaerMatrix([1.0 2; 2 4; 3 6])
P = pinv(B; rtol=1e-12)
@assert Matrix(B * P * B) ≈ Matrix(B)
@assert rank(B; rtol=1e-12) == 1
@assert cond(FaerMatrix([4.0 1; 1 3]), 1) ≈ cond([4.0 1; 1 3], 1)
```

`permute_rows` / `permute_cols`は独立した行列を返し、`!` 付きの関数は
入力を更新します。置換・交換は形状を変えないため、既存のビューも有効です。
`p * q`の添字列は `Vector(p)[Vector(q)]`です。`permute!(v,p)` /
`invpermute!(v,p)`で所有ベクトルとベクトルビューにも適用できます。
`invpermute!`は faerの借用逆置換を使い、新しい置換を作りません。

ストライド付き・逆順のビューは所有者から直接 `MatMut`として借用され、
`permute_rows!` / `permute_cols!` / `swap_rows!` / `swap_cols!` /
`permute!`は一時結果を経由せずに faerの 格納領域内での更新 経路で更新します。
`permute_rows(view, p)` もビューを直接読み、中間コピーを作りません。
格納領域内での更新 置換は対象を作業領域へコピーするため、呼び出し側の
`FaerScratch`を `scratch=`で渡せます。

```julia
using Colmat

A = FaerMatrix(reshape(collect(1.0:30.0), 5, 6))
W = view(A, 5:-2:1, 2:2:6)
p = FaerPermutation([3, 1, 2])
expected = Matrix(W)[Vector(p), :]
work = FaerScratch(permute_rows!, W, p)
reserve!(work, permute_cols!, W, p)
permute_rows!(W, p; scratch=work)
@assert Matrix(W) == expected
permute_cols!(W, p; scratch=work)
@assert scratch_requirements(permute_rows!, W, p).bytes <= scratch_capacity(work).bytes
```

`scratch`を省略すると、呼び出しごとに同じ要件の ネイティブ 領域を確保します。
容量・アラインメント不足、閉じた引数、無効化されたビュー、長さ不一致は
書き込み前に失敗します。添字が重複するビュー（step 0）は可変借用が別名になるため、
独立した一時結果と順序付き書き込みで更新します。faerの置換カーネルは逐次処理で
`Par`を取らないため `team=`はありません。

faer `PermRef`の形式のうち、Julia から意味を持つものは上記で接続済みです。
`Perm * Mat` / `Mat * Perm` / `Perm * Col` / `Row * Perm`はそれぞれ
`permute_rows(A,p)`、`permute_cols(A,invperm(p))`、`permute!(copy(v),p)`、
`permute_cols!(copy(r),invperm(p))`と同じ結果です。`FaerPermutation`は
`AbstractVector{Int}` なので、`p * A` や `A * p`は Julia 標準のベクトル積
（`A * Vector(p)` など）の意味を保ち、置換行列の作用には上書きしません。
同じ理由で `transpose(p)` / `adjoint(p)` も逆置換にせず、`invperm(p)`を使います。
添字型 `I`の選択（`u32` など）・`canonicalized`・`as_shape`/`bound_arrays`の
次元トークン・`new_unchecked`・`Reborrow`/`IntoConst`は Rustの型・所有権の
仕組みで、Juliaでは `Int` 添字の `FaerPermutation`に固定するため公開しません。
別々の行列の行・列を入れ替える `swap_rows`/`swap_cols`（`RowMut`/`ColMut` 対）も、
同一行列内の `swap_rows!`/`swap_cols!` 以外は公開していません。
`pinv`は faerの SVDと疑似逆再構成を使い、Hermitian・実 Symmetricでは
自己随伴固有値分解を使います。`atol` / `rtol`は Juliaと同じ判定式を使い、
省略時の `rtol`は `min(size(A)...)*eps`です。`cond`は 1・2・Inf ノルムに対応します。

## 数値型・変換・ライフサイクル

`sum`／`norm`／`dot`／`tr`／`kron`／`kron!`、`pfaffian` も Rustに接続しています。
Float32・Float64・ComplexF32・ComplexF64の密行列に対応します。
Issue #1 全体の進捗と残りの対象は
[実装台帳](https://github.com/AtelierArith/Colmat.jl/blob/main/docs/implementation-status.md)に記録しています。

```julia
using Colmat, LinearAlgebra

A = FaerMatrix(ComplexF32[1+2im 3; 4 5-im])
F = svd(A)
@assert eltype(A) === ComplexF32
@assert eltype(F.S) === Float32
@assert F.U * Diagonal(F.S) * F.Vt ≈ Matrix(A)
@assert Matrix(copy(A')) ≈ Matrix(A)'

B = FaerMatrix{Float32}(I, 2, 2)
reserve!(B, 8, 8)
resize!(B, 3, 3; fill=0)
truncate!(B, 2, 2)
@assert Matrix(B) == Matrix{Float32}(I, 2, 2)
```

同じ型の演算・分解は型を保持し、異なる型の行列同士の演算は
`promote_type`に従います。混合型の `A \ B`は係数行列を昇格してから
分解します。作成済みの `F = lu(A)`を再利用する場合は、元の分解精度が
残る点に注意してください。複素行列の内積は第1引数を共役化し、
`transpose(A)`と `A'`を区別します。

### 非有限値（Inf・NaN）

構築（Julia 配列・ビュー・転置から）・型変換・`copy`／`copyto!`・
`+`／`-`／スカラー積・除算は、Juliaの `Matrix`と同様に Inf・NaN・符号付きゼロを
そのまま保持します。実数のスカラーは複素行列の実部・虚部を成分ごとに
スケールするため（Juliaの `Real*Complex`と同じ）、`-A` や `2A`で無限大の成分が
もう一方の成分を NaNにすることはありません。実行列と複素スカラーの積、
複素スカラーでの除算は複素数の積として計算するため、Juliaと比べてゼロの符号だけが
異なることがあります。

有限性は分解・求解などの数値カーネルの入口で検査します。`lu`・`qr`・`cholesky`・
`ldlt`・`bunchkaufman`・`svd`／`svdvals`・`eigen`／`eigvals`（一般化を含む）・
`schur`・`hessenberg`・`inv`・`pinv`・`\`・`logabsdet`・`rank`・`opnorm`・`cond`・
疎行列の分解は、非有限な入力に対して `ArgumentError` または `RustError`を投げます
（Juliaの `lu` などは NaNを伝播しますが、faerのカーネルは有限入力を前提とするため）。
行列積・`tr`・`det` などの要素演算は Juliaと同様に非有限値を伝播します。

```julia
using Colmat, LinearAlgebra

A = FaerMatrix(ComplexF64[Inf 1; NaN -0.0])
@assert isequal(Matrix(-A), -Matrix(A))
@assert isequal(Matrix(FaerMatrix{ComplexF64}(FaerMatrix(Float32[Inf -0.0]))), ComplexF64[Inf -0.0])
@assert try lu(A); false catch; true end
```

### 対応スカラー型の範囲

Rust 所有型（`FaerMatrix`・`FaerVector`・`FaerRow`・`FaerDiagonal`・
`FaerSparseMatrix`）と分解は Float32・Float64・ComplexF32・ComplexF64の4型だけを
扱います。型指定のない構築では、その他の実数入力（整数・`Bool`・`Rational`・
`Float16`・`BigFloat` など）を Float64に、その他の複素数入力を ComplexF64に
変換します。`BigFloat` などの精度はこの変換で失われます。
配列からの構築で4型以外を明示した場合（`FaerMatrix{T}(A)`、
`FaerVector{T}(v)`・`FaerRow{T}(v)`・`FaerDiagonal{T}(v)`、
`FaerSparseMatrix{T}(A)`）は `ArgumentError`です。`undef`・関数・`fill`・
三つ組などの他の構築形式も4型だけを定義しています。
スカラー倍・除算（`x*A`・`A*x`・`A/x`・`x\A`）の結果型は
`promote_type(typeof(x), eltype(A))`に従い、それが4型以外なら `ArgumentError`
にします。Float64 行列でも `BigFloat`のスカラーを暗黙に丸めません。

```julia
using Colmat

@assert eltype(FaerMatrix([1 2; 3 4])) === Float64
@assert eltype(FaerVector([1+2im, 3])) === ComplexF64
A = FaerMatrix(Float32[1 2; 3 4])
@assert eltype(2A) === Float32
@assert eltype(A / 2.0) === Float64
@assert eltype(A * im) === ComplexF32
rejected = try
    big"0.1" * A
catch err
    err
end
@assert rejected isa ArgumentError
```

t4a-faerの `fx128`／`cx128`（`qd::Quad`、2つの f64による 倍精度数を 2 個組み合わせた拡張精度と
その複素数）と任意の フィールド トレイトは意図的に接続していません。Rust 側の所有型・
分解は4型の列挙で ディスパッチ しており、新しいスカラー型ごとにすべての ネイティブ 経路の
実体化が要ります。Julia 標準には対応する型がなく、Juliaで定義された型を Rustの
コンパイル時 トレイトとして渡すこともできません。拡張精度が必要な場合は Juliaの
汎用 `Matrix{BigFloat}` などの経路を使ってください。

`size`、`axes`、インデックスによる読み書き、`similar`、`fill!`、`copy`、
`deepcopy`、ブロードキャストにも対応します。`A'` / `transpose(A)`は Juliaの
遅延ラッパーで、`copy(A')`は Rust 側で転置行列を作ります。ブロードキャスト・
`map`・連結・`foreach`・`mapreduce`は次節の一括経路を使います。それ以外の汎用の
配列演算（`collect`、`for`による反復等）は要素ごとに FFIを呼ぶ場合があります。

### 密配列全体の ブロードキャスト・`map`・連結

`FaerMatrix`・`FaerVector`・`FaerRow`・ネイティブビューとその `'`／`transpose`を含む
ブロードキャストは、専用の `FaerArrayStyle`で処理します。各 ネイティブ 入力を Julia 配列へ
1回ずつ一括転送し（同じ入力の複数回参照は1回）、`faerview`の借用配列は親配列を
そのまま読みます。その上で Julia 自身の ブロードキャストを実行するため、次元の拡張・
0次元配列・`Ref`・タプル・空配列・結果の数値型は Juliaと同じ規則になります。

```julia
using Colmat
A = FaerMatrix([1.0 2; 3 4])
v = FaerVector([10.0, 20.0])
r = FaerRow([1.0, -1.0])
B = sin.(A) .+ 2 .* v          # FaerMatrix（列ベクトルを各列に拡張）
w = v ./ 0                     # Inf を含む結果も FaerVector
s = r .* 3                     # 行だけの式は FaerRow
mask = A .> 2                  # Bool は native 型ではないため BitMatrix
A .+= 1                        # Julia の作業配列で計算し、1回で書き戻す
view(A, 2:2, :) .= view(A, 1:1, :) .* 2  # 同じ owner の重なりも旧値を読む
C = map(+, A, B)               # FaerMatrix
D = [A B; B A]                 # hvcat も一括で FaerMatrix
@assert Matrix(D) == [Matrix(A) Matrix(B); Matrix(B) Matrix(A)]
```

結果の規約:

- Juliaが `Float32`・`Float64`・`ComplexF32`・`ComplexF64`の密な
  `Vector`／`Matrix`を返す場合は、`FaerVector`／`FaerMatrix`を返します。
  結果が1行で、ネイティブ 入力がすべて `FaerRow`の場合は `FaerRow`です。
  Julia から ネイティブへの転送は1回の一括書き込みで、`Inf`／`NaN` も保持します。
- それ以外は Juliaの結果をそのまま返します（比較の `BitMatrix`、
  `Int`／`BigFloat` 等の配列、3次元以上の配列、`SparseMatrixCSC`との積の疎行列等）。
- `FaerSparseMatrix`を含む式は疎行列の規約に従います（[疎行列](sparse.md)）。
  `FaerDiagonal`は格納要素を1回読んだ `Diagonal`として扱い、Juliaが対角のまま
  返す結果（`D .* 2`・`abs.(D)`・`D .* D`）は `FaerDiagonal`、それ以外（`D .+ 1`・
  `D .+ A`）は密な Rust 側の所有者です。`D .= 式`は `Diagonal`の作業領域で計算し
  （非零の非対角要素は書き込み前に例外）、格納要素を1回で書き戻します。
- 借用配列だけの ブロードキャストは従来どおり Julia 配列を返します。

同じ形状・同じ数値型の `FaerMatrix` 同士の `A .+ B`／`A .- B`と、実数行列の
型を広げない実スカラー倍・複素行列の同型スカラー倍（`2 .* A` 等）は、Rustの
行列全体の加算・スカラー倍をそのまま使います。丸めと `Inf`／`NaN`の扱いは
Juliaのスカラー演算と一致する組合せに限っています（`false .* A` や実行列への
複素スカラー倍は一般経路）。

`dest .= 式`（`.+=` 等を含む）は、各入力を読み終えてから `eltype(dest)`の Julia
作業配列に計算し、最後に1回の一括書き込みで `dest`を更新します。変換の失敗
（実数行列への複素数等の `InexactError`）や形状不一致では `dest`は変わりません。
`dest .= x`（スカラー）は `fill!`、同じ型の `FaerMatrix` や同じ数値型・同じ軸の
Julia 配列からの `dest .= B`は直接の一括 `copyto!`です（他の組合せも作業配列経由で
`Inf`／`NaN`を保持します）。Julia 配列や借用配列が出力先の場合は、ネイティブ 入力を転送した後に Juliaの
`materialize!`で書き込み、借用配列は親配列として別名解析されます。

`map(f, A, Bs...)`／`map!(f, dest, A, Bs...)`は、最初の配列（`map!`では出力先）が
Rust 側の場合に同じ転送を行い、Juliaの `map`を実行します。Julia 配列の後に ネイティブ
配列が続く `map(f, J, A)`（ネイティブ より前の Julia 配列は3個まで）と、Julia 配列が
出力先の `map!(f, J, A)` も同じ経路です。

`hcat`・`vcat`・`hvcat`（`[A B; C D]`）・`hvncat`（`[A;; B]`、`[A;;; B]`）・
`cat(...; dims)`と型指定リテラル（`T[A B]`・`T[A; B]`・`T[A B; C D]`・`T[A;; B]`）は、
どの位置の引数が ネイティブでも全 ネイティブ 引数を1回ずつ一括転送し、Julia／SparseArraysの
アルゴリズムで連結します。

```julia
J = [5.0 6; 7 8]
hcat(J, A)                # FaerMatrix（Julia 配列が先頭でも native）
[J A; A J]                # FaerMatrix
vcat(1.0, v)              # FaerVector（スカラーが先頭）
ComplexF64[J A]           # FaerMatrix{ComplexF64}
Int[J round.(Int, J)]     # Int は native 型ではないため Matrix{Int}
[A;;; J]                  # 3次元の結果は Array{Float64,3}
S = FaerSparseMatrix(sparse([1.0 0; 0 2]))
[S J; A S]                # FaerSparseMatrix（CSC）
reduce(hcat, [v, v])      # FaerMatrix
```

結果の規約は ブロードキャストと同じで、`vcat`によるベクトル同士の連結は `FaerVector`、
全 ネイティブ 引数が `FaerRow`の1行結果は `FaerRow`、それ以外の ネイティブ 型の2次元結果は
`FaerMatrix`、3次元以上や ネイティブ 型以外（`Int`・`BigFloat` 等）は Julia 配列です。
疎行列は SparseArraysの規約（疎行列を含む `hcat`／`vcat`／`hvcat`は疎行列）に従い、
`FaerSparseMatrix`が含まれる疎の結果は CSCの `FaerSparseMatrix`になります。Juliaの
`SparseMatrixCSC`と ネイティブ 密配列だけの連結は、ブロードキャストと同じく `SparseMatrixCSC`の
ままです。借用配列（`faerview`）や Julia 配列だけの連結は Base／SparseArraysの結果の
ままです。

ネイティブ 引数より前に置ける引数は、スカラー・`Array`・`BitArray`・範囲・Julia 配列の
`SubArray`／`'`／`transpose`・`SparseMatrixCSC`・`SparseVector`・借用配列で、15個まで
（4×4のブロックリテラルの全位置）です。これ以外の型（`Diagonal`・`UniformScaling`
等）の後に ネイティブ 引数が来る場合や16個目以降の場合は、従来の要素単位の Base 経路で
正しい結果を返します。`reduce(hcat, vs)`／`reduce(vcat, vs)`は、要素型が Rust 側の
ベクトルでは Baseの一括経路で Rust 側の所有者を返し、Julia 配列との混在ベクトルでは
上記の2引数連結の繰り返しになります。

`foreach(f, A, Bs...)`は、`A`（ネイティブ 密配列またはその `'`／`transpose`）と他の
ネイティブ 引数を反復前に1回ずつ一括転送します。`f`が反復中に `A`を変更しても反復する
値は呼び出し時点のものです。`mapreduce(f, op, A; dims, init)`（`sum(f, A)`・
`prod(f, A)`・`maximum(f, A)`・`minimum(f, A)`を含む）も1回の転送の後に Juliaの
`mapreduce`を実行し、`dims` 付きの ネイティブ 型の結果は Rust 側の所有者です。`sum(A)` や
`sum(abs2, A)` 等の既存の ネイティブ リダクションはそのままです。

faerの `mat!`／`row!`／`col!`・`concat!`・`zip!`／`unzip!` マクロは Rustの
コンパイル時構文で、Juliaには移植しません。それぞれ配列リテラルからの構築
（`FaerMatrix([1.0 2; 3 4])`、`FaerRow([1.0, 2.0])`、`FaerVector([1.0, 2.0])`）、
上記の連結、ブロードキャスト／`map`が同じ用途を担います。

`dest .= A .± B`／`dest .*= α` も同じ条件で Rustの結果を1回の ネイティブ 代入で書き込みます。

RustCallの所有ハンドルにより、GC 時に Rustの行列も解放されます。
`close(A)`は参照を外して以後のアクセスを禁止し、実際の解放は GCに委ねます。
複製した行列は元の行列を閉じた後も利用できます。Rustの可変参照が重複しないよう、
行列へのネイティブ呼び出しはロックで直列化しています。

## 互換 API と RustCall

新しいコードでは `FaerMatrix` と `*`、`\`、`svd` などの Julia 標準の記法を使ってください。既存コード向けに次の入口もあります。

| 関数 | 用途 |
| :--- | :--- |
| `faer_mul(A, B)` | 行列積 |
| `faer_solve(A, B)` | 正方行列の LU 求解 |
| `faer_svd(A)` | 特異値分解 |

通常の Julia 実数配列をこれらの互換 API に直接渡す場合は、`Float64` のバッファへコピーして逐次計算します。複素数や元の数値型を保持したい場合は `FaerMatrix` と通常の演算を使ってください。ネイティブ配列を受け取る互換メソッドは、対応する演算へ委譲します。

RustCall は Julia と Rust の間の型変換や所有オブジェクトの寿命を管理します。GC と `close` の違い、および `with_resource` の使い方は [GC と資源の寿命](lifetimes.md)にあります。Rust の借用型やポインタを意識しなくても、通常の配列操作を使えます。

依存関係は `Project.toml` の `[sources]` で RustCall の `main` を指定し、互換範囲は `0.10` です。Rust 側のマクロは Cargo で crates.io の `0.7` 系を指定しています。特定の同一 Git コミットへ両者を固定する構成ではありません。Julia 側で解決した版は `Manifest.toml` を参照してください。詳細は[開発ガイド](../development.md)にあります。
