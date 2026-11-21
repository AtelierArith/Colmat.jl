# Troubleshooting

## The shared library is not found

If `Colmat's BLAS library is missing` appears, build in the target Julia project and
restart Julia.

```julia
using Pkg
Pkg.build("Colmat")
```

If the build fails, check `deps/build.log` when using the repository. Also make sure that
Git, Cargo, and the OS development tools are available.

## The Rust source commit does not match

The build compares the actual checkout HEAD against `deps/upstream.toml`.
The following commands only inspect the state.

```sh
git -C deps/colmat-rs rev-parse HEAD
git -C deps/colmat-rs status --short
cat deps/upstream.toml
```

If there are local changes, save them and then prepare a clean checkout at the specified
revision. For the procedure for an intentional dependency update, see [Development](@ref).

## OpenBLAS also appears

Multiple libraries appearing in `BLAS.get_config()` is expected because the forwarding
targets for unsupported BLAS and LAPACK routines are retained. You can check the actual
GEMM forwarding target with the method in [Getting started](@ref). The display order of
libraries alone does not tell you where each routine runs.

## Returning after another backend was loaded

Stop parallel computation, then call `Colmat.load_backend()`.
[Backends and fallbacks](@ref) explains the scope and ordering of switching.

## Thread counts do not match

`Colmat.get_num_threads()` returns Colmat's configuration value, while
`BLAS.get_num_threads()` returns the value held by LBT. `Colmat.set_num_threads(n)` does
not update the value recorded by LBT, so changing only Colmat can leave the two out of
sync. To align every registered backend at once, use `BLAS.set_num_threads(n)`.
See [Thread settings](@ref) for details.

## An old name is not found

`FaerMatrix`, `FaerScratch`, `FaerTeam`, `with_resource`, and similar names were removed
in 0.2. Use ordinary `Matrix` / `Vector` and `LinearAlgebra`.
See [The 0.2 redesign](@ref) for the policy on the old API.
