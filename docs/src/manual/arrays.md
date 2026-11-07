# 配列・型・所有権

密行列、ベクトル、ビュー、置換の基本操作と、Rust 側の所有権を説明します。

## 密行列と基本演算

`FaerMatrix{T}` は Rust の `faer::Mat<T>` を所有する `AbstractMatrix{T}`
です。構築時に入力をコピーし、その後の行列同士の加減算・スカラー倍・行列積・
LU 求解は Rust 内のバッファを使います。これらの結果も独立した `FaerMatrix` になります。
`svd(A)`・`svdvals(A)` も Rust 側の行列を直接使い、分解結果を Julia 配列で返します。

`FaerMatrix{T}(A)` は入力を `Matrix{T}` に一度変換してコピーします。
ユーザー定義の変換が異なる格納型や入力と異なる形状を返した場合は、
Rust へデータを渡す前に例外にします。

```julia
using Colmat, LinearAlgebra

A = FaerMatrix([4.0 1; 1 3])
B = FaerMatrix([1.0 2; 3 4])
C = 2A + B - A / 2
D = A * B
X = A \ B
x = A \ [1.0, 2.0]         # ベクトルの右辺には Julia Vector を返す
@assert Matrix(X) ≈ Matrix(A) \ Matrix(B)

C[1, 2] = 5
mul!(C, A, B)              # C を更新。C === A / C === B も可
mul!(C, A, B, 2.0, 3.0)   # C = 2A*B + 3C
copyto!(C, A)
Z = zero(A)
E = copy(A)               # Rust の行列を複製
snapshot = Matrix(A)      # Julia へ独立したコピーを取得
```

## ベクトル・行・対角行列とビュー

Rust が所有する列ベクトル・行ベクトル・対角行列も使えます。

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
close(native_zeros)
close(native_ones)
close(native_full)
```

`view` は親の所有者を保持し、ネイティブ呼び出しの間だけ Rust の
`MatRef` / `MatMut` を作ります。部分行列・行・列・入れ子の範囲ビューに
対応します。親の形状を変える `resize!` / `truncate!` は既存のビューを
失効させます。親を閉じた場合もアクセスは例外になります。
`copy(view(...))` は独立した Rust 所有行列・ベクトルを返します。
ビューへの `copyto!` は native 所有型・ビューの入力を一時コピーし、代入先と重なる
場合も値を保ちます。同じ要素型の Julia strided 配列（`Matrix`、範囲・逆順の
`view` など）は一時コピーせず、Julia のメモリを直接読みます（形状が一致する
所有型への `copyto!` も同様）。
`FaerMatrix` / `FaerRow` への Julia 配列・ビュー・疎行列などからの `copyto!` も同じ
一括代入経路を使い、要素毎の `setindex!` は行いません。形状が異なっても要素数が
収まれば Base と同じく線形順にコピーし、長すぎる入力は `BoundsError` です。ただし
`FaerMatrix` 同士のネイティブコピーは従来どおり形状一致が必要で、`DimensionMismatch`
になります。要素型の変換失敗・close 済みの代入先はいずれも書き込み前に例外となり、
代入先は変わりません。

`faerview(A)` は Julia の strided ベクトル・行列を借用する `FaerBorrowedMatrix` /
`FaerBorrowedVector` を返します。`A` を保持し、各ネイティブ呼び出しの間だけ
`pointer(A)`・`size(A)`・`strides(A)` から faer の `MatRef` を作るため、負や
非単位の stride もコピーせずに読みます。ポインタ・形状は保存しないので、親の
`Vector` を `resize!` しても次の呼び出しは現在の長さとアドレスを使います。
要素型は Float32 / Float64 / ComplexF32 / ComplexF64、親は `Array` / `Memory` に
限り、stride 0 や重なる stride は `ArgumentError` です。`copy` / `FaerMatrix`、
`+` / `-` / `*`（転置・随伴、native 行列やビューとの混在、型昇格を含む）、
`FaerVector` への `mul!`、`mean` / `var`、`copyto!` の入力が借用のまま動作し、
`view` も借用を保ちます。添字・`strides`・`pointer` は親へ転送します。

```julia
J = rand(4, 3)
B = faerview(J)                       # コピーしない
P = B' * FaerMatrix(J)                # Julia のメモリを直接読む native 積
J[1, 1] = 0                           # 以後の呼び出しに反映
@assert copy(B)[1, 1] == 0
copyto!(view(FaerMatrix{Float64}(undef, 4, 3), :, :), view(J, :, :))
close(P)
```

`FaerDiagonal` の非対角要素は暗黙のゼロで、非ゼロの代入はできません。

対角との積・除算・和は faer の `DiagRef` を借用して native に計算し、
n×n の密な snapshot を作りません。`D*A` / `A*D` / `D\A` / `A/D` / `D±A`、
`D*v` / `D\v`、`r*D` / `r/D`（結果は `FaerRow`）、`inv(D)`、
`lmul!` / `rmul!` / `ldiv!` / `rdiv!` に対応します。Julia の `Diagonal` も、
格納された n 個の要素だけを渡して同じ経路を使います。
ゼロの対角要素で割ると `SingularException` を投げ、入力も代入先も変更しません。
`diag(A)` と `FaerDiagonal(A)` は密行列・範囲ビューの主対角を native にコピーし、
`v * w'` は Rust 所有の `FaerMatrix` を返します。

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
try
    FaerDiagonal([1.0, 0.0]) \ A
catch e
    @assert e isa SingularException && e.info == 2
end
close(A)
close(D)
```

`FaerMatrix{T}(m,n; fill=value)` と `FaerVector{T}(n; fill=value)` /
`FaerRow{T}(n; fill=value)` / `FaerDiagonal{T}(n; fill=value)` は定数値から
native に直接構築し、Julia の密配列を実体化しません。fill の既定値はゼロです。
`zeros` / `ones` に所有型を指定する形式もあり、行列は `(m,n)`、collection は
`(n,)` の tuple 指定にも対応します。型パラメータを省略すると Float64 です。
`ones(FaerDiagonal{T},n)` は格納対角を1にし、非対角成分は暗黙のゼロです。

`FaerMatrix{T}(f,m,n)` / `FaerVector{T}(f,n)` / `FaerRow{T}(f,n)` は
関数または callable object から構築します。1-based の添字を渡し、要素ごとに
1回だけ、行列では列優先順に評価します。空の領域では関数を呼びません。
`{T}` を省略すると Float64 なので、複素数の結果には複素型を明示してください。

`capacity(A)` は所有行列の実際の行・列容量を組で返し、`capacity(v)` /
`capacity(r)` / `capacity(D)` は格納要素の容量を整数で返します。
予約した最小値より alignment padding の分だけ大きい場合があります。
ビューには容量照会を定義していません。対角の容量を照会しても、固定の次元を
変更できるようにはなりません。容量は native owner から読み取り、Julia 側で
予約値や表示サイズから推測しません。

容量管理の対応範囲は次のとおりです。

| faer API | Colmat の対応・制約 |
|---|---|
| `Mat::with_capacity` | `with_capacity(FaerMatrix{T}, rows, cols)` は次元 `(0,0)` の所有行列を予約済み容量で作ります。`FaerVector{T}` / `FaerRow{T}` と1個の容量を指定する形式もあり、`{T}` を省略すると Float64 です。 |
| `try_reserve` / `reserve` | `reserve!` / `sizehint!` は `try_reserve` を使い、失敗を例外として返します。 |
| `resize_with` | `resize!(...; fill=value)` は固定値で初期化します。`resize_with!(f,A,rows,cols)` / `resize_with!(f,x,length)` は新規要素だけを添字から生成し、重なる領域を保持します。 |
| `truncate` | `truncate!` は縮小のみを受け付け、容量を保持します。 |
| `row_capacity` / `col_capacity` / `capacity` | 所有型の `capacity` で実際の容量を照会します。 |
| unsafe `set_dims` | 初期化済み領域の証明を呼び出し側に要求するため、Julia の公開 API としては提供していません。安全な `resize!` は新しい要素を初期化します。 |

`resize_with!` の関数は1-based の添字を受け取り、新規要素だけを列優先順に
評価します。do-block 形式も使えます。縮小のみ・形状不変なら関数を呼びません。
新規値を Julia の一時 primitive buffer に評価・型変換してから native に渡すため、
関数の例外や変換失敗では入力を更新しません。関数自身による owner の変更・close は
禁止で、関数自身の副作用は巻き戻せません。形状・owner が変わった場合は拒否します。
native 側も元形状・スカラー型・Row/Col の種類・buffer 長を検証し、truncate 前に
必要容量を確保します。成功時の形状変更は既存ビューを失効させます。

`Row` / `Col` は予約とサイズ変更に対応します。faer `Diag` に予約・サイズ変更の
公開メソッドはなく、Colmat でも対角の次元は固定です。対応範囲と検証は
[Issue #90](https://github.com/AtelierArith/Colmat.jl/issues/90) で追跡します。

## 置換・ランク・疑似逆行列

置換も Rust に保持できます。添字は Julia と同じ 1-based です。

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

`permute_rows` / `permute_cols` は独立した行列を返し、`!` 付きの関数は
入力を更新します。置換・交換は形状を変えないため、既存のビューも有効です。
`p * q` の添字列は `Vector(p)[Vector(q)]` です。`permute!(v,p)` /
`invpermute!(v,p)` で所有ベクトルとベクトルビューにも適用できます。
`invpermute!` は faer の借用逆置換を使い、新しい置換を作りません。

ストライド付き・逆順のビューは所有者から直接 `MatMut` として借用され、
`permute_rows!` / `permute_cols!` / `swap_rows!` / `swap_cols!` /
`permute!` は一時結果を経由せずに faer の in-place 経路で更新します。
`permute_rows(view, p)` もビューを直接読み、中間コピーを作りません。
in-place 置換は対象を作業領域へコピーするため、呼び出し側の
`FaerScratch` を `scratch=` で渡せます。

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

`scratch` を省略すると、呼び出しごとに同じ要件の native 領域を確保します。
容量・アラインメント不足、閉じた引数、無効化されたビュー、長さ不一致は
書き込み前に失敗します。添字が重複するビュー（step 0）は可変借用が別名になるため、
独立した一時結果と順序付き書き込みで更新します。faer の置換カーネルは逐次処理で
`Par` を取らないため `team=` はありません。

faer `PermRef` の形式のうち、Julia から意味を持つものは上記で接続済みです。
`Perm * Mat` / `Mat * Perm` / `Perm * Col` / `Row * Perm` はそれぞれ
`permute_rows(A,p)`、`permute_cols(A,invperm(p))`、`permute!(copy(v),p)`、
`permute_cols!(copy(r),invperm(p))` と同じ結果です。`FaerPermutation` は
`AbstractVector{Int}` なので、`p * A` や `A * p` は Julia 標準のベクトル積
（`A * Vector(p)` など）の意味を保ち、置換行列の作用には上書きしません。
同じ理由で `transpose(p)` / `adjoint(p)` も逆置換にせず、`invperm(p)` を使います。
添字型 `I` の選択（`u32` など）・`canonicalized`・`as_shape`/`bound_arrays` の
次元トークン・`new_unchecked`・`Reborrow`/`IntoConst` は Rust の型・所有権の
仕組みで、Julia では `Int` 添字の `FaerPermutation` に固定するため公開しません。
別々の行列の行・列を入れ替える `swap_rows`/`swap_cols`（`RowMut`/`ColMut` 対）も、
同一行列内の `swap_rows!`/`swap_cols!` 以外は公開していません。
`pinv` は faer の SVD と疑似逆再構成を使い、Hermitian・実 Symmetric では
自己随伴固有値分解を使います。`atol` / `rtol` は Julia と同じ判定式を使い、
省略時の `rtol` は `min(size(A)...)*eps` です。`cond` は 1・2・Inf ノルムに対応します。

## 数値型・変換・ライフサイクル

`sum`／`norm`／`dot`／`tr`／`kron`、`pfaffian` も Rust に接続しています。
Float32・Float64・ComplexF32・ComplexF64 の密行列に対応します。
Issue #1 全体の進捗と残りの対象は
[実装台帳](https://github.com/AtelierArith/Colmat.jl/blob/main/docs/implementation-status.md) に記録しています。

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
`promote_type` に従います。混合型の `A \ B` は係数行列を昇格してから
分解します。作成済みの `F = lu(A)` を再利用する場合は、元の分解精度が
残る点に注意してください。複素行列の内積は第1引数を共役化し、
`transpose(A)` と `A'` を区別します。

### 対応スカラー型の範囲

native 所有型（`FaerMatrix`・`FaerVector`・`FaerRow`・`FaerDiagonal`・
`FaerSparseMatrix`）と分解は Float32・Float64・ComplexF32・ComplexF64 の4型だけを
扱います。型指定のない構築では、その他の実数入力（整数・`Bool`・`Rational`・
`Float16`・`BigFloat` など）を Float64 に、その他の複素数入力を ComplexF64 に
変換します。`BigFloat` などの精度はこの変換で失われます。
配列からの構築で4型以外を明示した場合（`FaerMatrix{T}(A)`、
`FaerVector{T}(v)`・`FaerRow{T}(v)`・`FaerDiagonal{T}(v)`、
`FaerSparseMatrix{T}(A)`）は `ArgumentError` です。`undef`・関数・`fill`・
三つ組などの他の構築形式も4型だけを定義しています。
スカラー倍・除算（`x*A`・`A*x`・`A/x`・`x\A`）の結果型は
`promote_type(typeof(x), eltype(A))` に従い、それが4型以外なら `ArgumentError`
にします。Float64 行列でも `BigFloat` のスカラーを暗黙に丸めません。

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

t4a-faer の `fx128`／`cx128`（`qd::Quad`、2つの f64 による double-double と
その複素数）と任意の field trait は意図的に接続していません。Rust 側の所有型・
分解は4型の列挙で dispatch しており、新しいスカラー型ごとにすべての native 経路の
実体化が要ります。Julia 標準には対応する型がなく、Julia で定義された型を Rust の
コンパイル時 trait として渡すこともできません。拡張精度が必要な場合は Julia の
汎用 `Matrix{BigFloat}` などの経路を使ってください。

`size`、`axes`、インデックスによる読み書き、`similar`、`fill!`、`copy`、
`deepcopy`、ブロードキャストにも対応します。`A'` / `transpose(A)` は Julia の
遅延ラッパーで、`copy(A')` は Rust 側で転置行列を作ります。汎用の配列演算や
ブロードキャストは要素ごとに FFI を呼ぶ場合があります。

RustCall の所有ハンドルにより、GC 時に Rust の行列も解放されます。
`close(A)` は参照を外して以後のアクセスを禁止し、実際の解放は GC に委ねます。
複製した行列は元の行列を閉じた後も利用できます。Rust の可変参照が重複しないよう、
行列へのネイティブ呼び出しはロックで直列化しています。

## 互換 API と RustCall

- `faer_mul(A, B)`: 行列積。
- `faer_solve(A, B)`: 正方行列の partial-pivot LU による求解。右辺は行列またはベクトル。
- `svd(A::FaerMatrix; full=false)`: SVD。Julia 標準の `LinearAlgebra.SVD` を返します。`full=true` にも対応します。
- `svdvals(A::FaerMatrix)`: 特異値のみ計算します。
- `faer_svd(A)`: 互換 API。`FaerMatrix` では `svd(A)` と同じです。

通常の Julia 行列を受け取る上記の関数は、入力を Float64 の列優先バッファにコピーします。ビュー・転置も利用でき、
入力は変更しません。処理は `Par::Seq` で実行します。NaN/Inf、特異な LU、
SVD の失敗は例外になります。Julia バッファは `GC.@preserve` で保持し、Rust 側に
ポインタを残しません。この PoC はコピーと割り当てを行うため、性能測定用の
ゼロコピー・並列実装ではありません。互換 API に通常の実数配列を渡す経路は
Float64 に変換します。複素数を含め型を保持する場合は `FaerMatrix` を使ってください。
残機能は Issue #22 と機能対応表で追跡しています。

RustCall v0.9.2 で [#564](https://github.com/AtelierArith/RustCall.jl/issues/564)・
[#565](https://github.com/AtelierArith/RustCall.jl/issues/565) が修正されました。
`copy(::FaerMatrix)` は `duplicate(&self) -> Result<Self, String>` から独立した所有ハンドルを
直接受け取ります。行列・分解など型が決まっている所有者の引数には Rust の `&T` / `&mut T` を
使い、RustCall に型、生存状態、ライブラリ世代、可変借用の重複を検査させます。
型を実行時に切り替えるビューなどの既存 API には、明示的な生ポインタ経路を残しています。

現在は未公開 RustCall 0.10 / macros 0.7 のマージ済みコミット
[`b53c9b9`](https://github.com/AtelierArith/RustCall.jl/commit/b53c9b963084231d784e68760001f15cc27d1ab1)
を `Project.toml` の `[sources]` と Cargo の Git 依存で同一 revision に固定しています。
Hessenberg 化の新しい API は `Option<&OwnedTeam>` による checked な省略可能 team と、
`#[rustcall_owned_return(OwnedCompactTridiagonal)]` による別所有型の直接返却を使います。
必要量照会の UInt 配列も checked な可変 slice として渡し、手動の raw pointer 処理を除いています。
これにより Seq/Team の wrapper 重複と compact reduction の Julia 入出力 staging を削っています。
crate の registry 公開は行っていません。checkout 後は `Pkg.instantiate()` で固定した依存を
解決してください。更新後の native クレートは次回ロード時に再ビルドされます。
