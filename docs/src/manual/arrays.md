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
ビューへの `copyto!` は入力を一時コピーし、代入先と重なる場合も値を保ちます。
`FaerDiagonal` の非対角要素は暗黙のゼロで、非ゼロの代入はできません。

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
`invpermute!(v,p)` で所有ベクトルにも適用できます。
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
