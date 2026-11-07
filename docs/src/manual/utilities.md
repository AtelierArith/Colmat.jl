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
singleton sample variances follow Julia's NaN convention. Supplied means promote
the working scalar type. Calls snapshot native data to primitive buffers before
running faer's kernels; they leave the input unchanged.

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

Another Julia RNG can supply a UInt64 seed for each nonempty native sampling
call. Empty calls consume no seed. Use `FaerRNG` to keep a continuous native
stream; copy and reseed it independently, and close it when no longer needed.
Native sampling is reproducible for the same dependency versions and call
sequence, without promising Julia's random sequence. Scalar/ordinary Julia-array
draws from `FaerRNG` use Julia's sampling algorithms over its native random bits.

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
