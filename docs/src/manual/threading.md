# Thread settings

## Setting Colmat and the fallbacks together

```julia
using Colmat, LinearAlgebra
BLAS.set_num_threads(2)
```

Because Colmat registers its thread-setting function with LBT, this standard API sets
both Colmat and every loaded library that supports thread settings.

## Setting only Colmat

```jldoctest
julia> using Colmat

julia> saved = Colmat.get_num_threads();

julia> Colmat.set_num_threads(2); Colmat.get_num_threads()
2

julia> Colmat.set_num_threads(saved)
```

The initial value is the value of `BLAS.get_num_threads()` at the time `using Colmat` runs.
`Colmat.set_num_threads(n)` accepts integers in `1:typemax(Cint)` and raises
`ArgumentError` for values outside that range.
`Colmat.get_num_threads()` returns Colmat's own configuration value.
`BLAS.get_num_threads()` returns the value held by LBT, so the two are different.
Calling only `Colmat.set_num_threads(n)` does not update the value recorded by LBT.

```julia
BLAS.set_num_threads(8)     # colmat=8, BLAS.get_num_threads() == 8
Colmat.set_num_threads(3)   # colmat=3, BLAS.get_num_threads() stays 8
```

The reverse direction, `BLAS.set_num_threads(n)`, calls the setter of every backend
registered with LBT, so Colmat's configuration value also becomes `n`. Use
`BLAS.set_num_threads(n)` when you want everything aligned.

## Execution width and pools

The configuration value is the parallel execution width of matrix multiplication, that
is, the number of participants including the caller. It does not make every vector
operation or triangular solve parallel.

Each calling OS thread lazily creates a pool. If the pool's execution width differs from
the configuration value, the next parallel path releases the old pool and recreates it.
Width one uses the sequential path. If pool creation or team acquisition fails, execution
also falls back to the sequential path, so the configuration value is not always the
actual number of parallel participants.

When multiple Julia threads call matrix multiplication at the same time, each caller
acquires its own participants. When combining this with outer Julia parallelism, choose
the configuration with Colmat's execution width in mind. Changing the setting does not
recreate computations that are already in progress.
