# 統計・乱数・NPY

統計量、乱数生成、NPY ファイルを扱います。

## Native statistics

`Statistics.mean`, `var`, and `varm` use faer's row/column kernels for native
matrices, vectors, rows and views. Dimension reductions return Julia arrays;
full reductions return scalars:

```julia
using Colmat, Statistics
A = FaerMatrix([1.0 2; 3 4; 5 6])
@assert mean(A; dims=1) ≈ [3.0 4.0]
@assert var(A; dims=1) ≈ [4.0 4.0]
@assert varm(A, mean(A; dims=1); dims=1) ≈ var(A; dims=1)
A[1, 1] = NaN
@assert isnan(mean(A))
@assert mean(A; dims=1, nan=:ignore) ≈ [4.0 4.0]
```

The default `nan=:propagate` retains NaNs; `nan=:ignore` excludes entries with
any NaN component and adjusts each slice's count. `corrected=false` uses
population variance. Complex variances are real, empty slices return NaN, and
singleton sample variances follow Julia's NaN convention. faer's kernels read
native arrays and strided or reversed views in place, so no Julia copy of the
data is made; inputs are left unchanged.

Supplied means follow `Statistics.varm` result types: a full reduction returns
the real type of the promoted scalar, while a `dims` reduction keeps the data's
real type. A mean that widens the data scalar (for example a `Float64` or
complex mean for `Float32` data) makes Rust stage one converted copy, as does a
full reduction of a view whose entries do not form one strided column. Means
whose promotion is not a native scalar, such as `BigFloat`, throw an
`ArgumentError`:

```julia
using Colmat, Statistics
A = FaerMatrix(Float32[1 2; 3 5])
@assert varm(A, 2.0) isa Float64
@assert eltype(varm(A, [2.0 3.0]; dims=1)) === Float32
@assert var(view(A, 2:-1:1, :); dims=2) ≈ var(Float32[3 5; 1 2]; dims=2)
```

## Native random matrices and unitary samples

`FaerRNG` owns a Rust random stream. Native array sampling uses faer's matrix,
column and row distributions directly:

```julia
using Colmat, Random, LinearAlgebra
rng = FaerRNG(42)
A = randn(rng, FaerMatrix{ComplexF64}, 3, 2)
snapshot = copy(rng)
v = rand(rng, FaerVector{Float64}, 4)
@assert Vector(v) == Vector(rand(snapshot, FaerVector{Float64}, 4))
Q = randunitary(rng, ComplexF64, 4)
@assert adjoint(Matrix(Q)) * Matrix(Q) ≈ Matrix{ComplexF64}(I, 4, 4)
Random.seed!(rng, 42)
@assert Matrix(randn(rng, FaerMatrix{ComplexF64}, 3, 2)) == Matrix(A)
row = rand(MersenneTwister(1), FaerRow{Float32}, (8,))
@assert size(row) == (1, 8)
randn!(rng, view(A, 1:2, 1:2))
```

`rand` gives independent uniform real/imaginary components in [0,1). `randn`
gives real standard normals; complex components each have variance 1/2, matching
Julia's unit expected squared modulus. `rand!`/`randn!` fill native matrices,
vectors, rows and views after staging the result. `randunitary(Float64,n)` gives
an orthogonal sample; complex types give unitary samples through faer's
sequential QR and phase correction.

Seed contract for other Julia RNGs: each nonempty native `rand`, `randn`,
`rand!(rng, A)`, `randn!(rng, A)` or `randunitary` call draws exactly one
`rand(rng, UInt64)`, seeds a temporary native stream with it, and samples from
that stream. Empty calls consume no seed. The result is reproducible for the
same Julia RNG state, dependency versions and call sequence, but it is not
Julia's own `rand(rng, T, dims)`/`randn` sequence, and consecutive calls are
seeded independently rather than continuing one native stream. Use `FaerRNG` to
keep a continuous native stream; copy and reseed it independently, and close it
when no longer needed. Scalar/ordinary Julia-array draws from `FaerRNG` use
Julia's sampling algorithms over its native 64-bit words, including narrower
and 128-bit integer, `Float16`/`Float32` and complex draws.

`rand!(rng, A, X)` fills a native matrix, vector, row or view with Julia samples
from `X`: a range, collection, scalar type, `Random.Sampler`, or any
distribution object that implements Julia's `rand(rng, X)` (Distributions.jl
included). It stages Julia's `rand!` on an `Array` of the same eltype and shape,
so `rng` is consumed exactly as for that array, then performs one bulk native
assignment. A closed target fails before sampling; sampling or conversion
failures leave the target unchanged:

```julia
using Colmat, Random
A = zeros(FaerMatrix{Float64}, 2, 3)
rand!(Xoshiro(1), A, 1:6)
@assert Matrix(A) == rand!(Xoshiro(1), Matrix{Float64}(undef, 2, 3), 1:6)
B = 2 .+ 3 .* Matrix(randn(FaerRNG(1), FaerMatrix{Float64}, 2, 2))  # N(2, 3²)
```

Parameterized faer/rand_distr distributions are not exposed as separate native
APIs: uniform on `[a,b)` and normal with mean `μ` and standard deviation `σ` are
affine maps of native `rand`/`randn`, and other distributions are available
through `rand!(rng, A, X)` or the per-element function constructors such as
`FaerMatrix{T}((i, j) -> f(i, j), m, n)`. Rust's `Distribution<T>` is a
compile-time generic trait with no runtime object a Julia value could name;
wrapping a Julia callback as one would cost one FFI round trip per element
while holding the native RNG borrow, which is slower than staging in Julia.
faer's distributions take no `Par`, so there is no Seq/Team variant.

## NPY loading and read-only views

`loadnpy` reads NPY bytes, an IO or a file path into an independent native matrix.
This example uses a small fixture included with the package:

```julia
using Colmat
path = joinpath(pkgdir(Colmat), "test", "fixtures", "npy_c_f64.npy")
A = loadnpy(path)
@assert Matrix(A) == [1.0 2 3; 4 5 6]
V = npyview(path)
@assert npyinfo(V).shape == (2, 3)
@assert V[2, 3] == 6.0
B = copy(V) # independent, mutable FaerMatrix
close(V)
@assert Matrix(B) == Matrix(A)
```

Float32/Float64 and ComplexF32/ComplexF64, NPY versions 1–3, C/Fortran order and
both byte orders are supported. Scalars become 1×1 matrices and vectors become
n×1 matrices; `npyinfo` retains original shape/version/order metadata. NaN/Inf,
NaN payloads and signed zero are preserved. Higher ranks and other dtypes are
rejected, as are inconsistent lengths and trailing bytes. IO inputs are read
from their current position to EOF as one array.

The default `loadnpy` copies the payload directly from the source bytes through
faer's `Npy::to_mat`, which reads unaligned and foreign-endian inputs without an
intermediate normalized buffer; contiguous byte vectors (including contiguous
`view`s) are read in place without a Julia-side snapshot.

`npyview` (or `loadnpy(...; view=true)`) owns an aligned, native-endian snapshot
of the bytes and uses faer's aligned-reference API for reads; that API requires
a 64-byte aligned buffer in native byte order, so the view normalizes its
snapshot. It does not alias the input bytes or memory-map a file (aliasing a
borrowed buffer is tracked separately). Entries are read-only. `copy` materializes
a native matrix; `deepcopy` preserves the view type with independent storage.
Closing a view invalidates its reads, while materialized/deep copies remain valid.

## Real matrix extrema

`minimum`, `maximum` and `extrema` use native storage directly for Float32 and
Float64 matrices and return the input scalar type. NaNs propagate; negative zero
sorts below positive zero. Empty matrices throw `ArgumentError`. Complex matrices
retain Julia's generic ordering contract.

```julia
using Colmat
A = FaerMatrix(Float32[3 -2; 1 4])
@assert minimum(A) === -2f0
@assert maximum(A) === 4f0
@assert extrema(A) === (-2f0, 4f0)
A[1, 1] = -0f0
A[1, 2] = 0f0
A[2, 1] = 0f0
A[2, 2] = 0f0
@assert signbit(minimum(A))
@assert !signbit(maximum(A))
close(A)
```
