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
snapshot. It does not alias the input bytes or memory-map a file (use
`npyborrow` for that). Entries are read-only. `copy` materializes
a native matrix; `deepcopy` preserves the view type with independent storage.
Closing a view invalidates its reads, while materialized/deep copies remain valid.

### Zero-copy borrows: memory-mapped files and aliased bytes

`npyborrow(path)` and `npyborrow(io::IOStream)` memory-map the file, and
`npyborrow(bytes)` aliases a `Vector{UInt8}` in place. All return a
`FaerBorrowedMatrix` whose parent addresses the payload directly. C-order arrays
are exposed through row strides `(n, 1)`, so neither order is copied. Native
consumers such as `copy`, `FaerMatrix`, `+`/`-`/`*` (including transposes and
native operands), `sum`, `mean` and `var` read the mapped pages in place, as
they do for `faerview`. `npyinfo(M)` returns the metadata recorded at borrow
time (shape, order, version, byte order, plus `writable`) without rereading the
source.

```julia
M = npyborrow(path)      # read-only mmap, no payload copy
@assert M == [1.0 2 3; 4 5 6]
@assert strides(M) == (3, 1) # C order read through row strides
C = copy(M)              # independent FaerMatrix

tmp = tempname()
cp(path, tmp)
W = npyborrow(tmp; writable = true) # read-write MAP_SHARED mapping
mul!(W, FaerMatrix([0.0 1; 1 0]), npyborrow(tmp)) # rows swapped in the file pages
Colmat.Mmap.sync!(W)     # flush; the aliased input was snapshot first
@assert loadnpy(tmp) == [4.0 5 6; 1 2 3]
```

- **When aliasing applies:** the payload must be native-endian and its data
  offset a multiple of the element alignment. NumPy pads headers to 16 or 64 bytes,
  so its files qualify. Otherwise (foreign endian, odd header padding, or a
  `SubArray` byte source) the copying `loadnpy` result is returned. Pass
  `fallback=false` to get an `ArgumentError` instead. `writable=true` never
  falls back: a copy would silently not write back.
- **Read-only by default:** `setindex!`, broadcasts, `copyto!`, `fill!`, `mul!`
  and `ldiv!` into the borrow throw `ArgumentError`, and native kernels never
  write it. Views of it are plain Julia views.
- **Writable borrows:** `writable=true` opens a path `"r+"` (a stream must be
  writable) and maps it read-write, or allows writes into the byte vector.
  `setindex!`, `.=`, `copyto!`, `fill!`, `mul!` and `ldiv!`/`rdiv!` then write the
  payload in place; dense `mul!`, `copyto!` and triangular solves write it
  natively through a faer `MatMut`. `Mmap.sync!(W)` flushes the mapping, and
  pages written before the finalizer unmaps it stay in the file either way.
- **Same-file aliasing:** every borrow of one file in a process (identified by
  device and inode, so hard links too) shares a single mapping. That mapping is
  read-write once any borrow of the file is writable, and a read-only borrow
  sharing it still rejects writes. Two borrows of one file therefore alias at
  the same addresses. The byte-range overlap checks of native destinations, and
  Julia's `mightalias` for broadcasts, snapshot an overlapping input before
  writing (`mul!(W, R, R)` or `copyto!(W, transpose(R))` with `R` a second
  borrow of the file is correct). Separate mappings would alias at different
  virtual addresses, which no range check can relate. Copies of a file have
  their own inode and do not alias. Writes by other processes, or through
  `Mmap.mmap` arrays created outside `npyborrow`, are visible but neither
  tracked nor ordered with native calls.
- **Streams:** `npyborrow(io)` maps the whole file behind an `IOStream` and
  borrows the NPY data from the stream's current position to the end of the
  file. The position is not advanced. If the shared mapping is read-write and
  must be remapped (the file size changed), the stream must be writable.
- **Lifetime:** the borrow keeps the byte vector or mapping alive. Closing or
  unlinking the file does not invalidate the mapping, which is unmapped by its
  finalizer once every borrow of the file and its views are unreachable; there
  is no explicit `close`.
- **Changing the source:** shrinking the source vector with `resize!` is
  detected before every access. When a file's size has changed, the next borrow
  of it remaps the file and all its borrows move to the new mapping; a borrow
  whose payload no longer fits then throws `ArgumentError` on access.
  Truncating a mapped file while it is accessed raises `SIGBUS`, as for any
  `mmap`.

## Real matrix extrema

`minimum`, `maximum` and `extrema` use native storage directly for Float32 and
Float64 matrices, vectors, rows and views and return the input scalar type
(`faerview` arrays keep Julia's vectorized reductions over the parent, without a
copy). NaNs propagate; negative zero sorts below positive zero. `dims`
follows Base (an integer or tuple; dimensions past `ndims(A)` reduce nothing),
and reducing an empty dimension throws `ArgumentError`. `init` and function-first
forms other than `identity` run Julia's generic implementation on one snapshot
(the parent array of a `faerview`). Complex inputs retain Julia's generic ordering
contract.

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
@assert extrema(A; dims = 1) == [(0f0, 0f0) (0f0, 0f0)]
close(A)
```

## Reductions of views and borrowed arrays

`dot`, `norm`, `sum`, `minimum`/`maximum`/`extrema`, `all(isfinite, A)` and
`any(isnan, A)` read native views (strided and reversed ranges) in place, without
a Julia or Rust copy; `dot`, `sum`, the predicates and `norm(A, 1/2/Inf)` do the
same for `faerview` arrays. `dot` conjugates its first argument
with faer's `inner_prod` for any pair of native vectors, rows, matrices, views and
borrowed arrays; vector-shaped operands pair entries in order, and mixed scalar
types promote (Rust stages one converted copy of a promoted operand).
`norm(A, p)` is the entrywise norm for every real `p` except NaN: 2/1/Inf use
faer's reductions, 0 counts nonzeros, -Inf is the NaN-propagating smallest
magnitude, and other `p` follow `LinearAlgebra.generic_normp` (Float64
accumulation, rescaling by the extreme magnitude); for `faerview` arrays those
`p` use Julia's own `norm` of the parent. Use `opnorm` for operator norms. Summation order is faer's, so sums may differ from Julia's by rounding.

```julia
using Colmat, LinearAlgebra
J = [3.0 -4.0; 0.0 1.0]
B = faerview(J)
x = FaerVector([1.0, 2.0, 3.0])
@assert dot(view(x, 3:-1:1), faerview([1.0, 0.0, 1.0])) == 4.0
@assert norm(B, 0) == 3.0
@assert norm(B, 3) ≈ norm(J, 3)
@assert norm(B, -Inf) == 0.0
@assert sum(view(x, 1:2:3)) == 4.0
@assert minimum(B; dims = 2) == [-4.0; 0.0;;]
@assert all(isfinite, B) && !any(isnan, view(x, :))
close(x)
```
