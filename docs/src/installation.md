# Installation

## Requirements

- Julia 1.13.
- Rust 1.89 or later and Cargo.
- Git.
- A network connection for the initial download of the Rust source and Cargo dependencies.

Also install the development tools for your OS that Rust compilation requires.
The current CI configuration targets Linux and macOS.

## Building from the repository

Run the following from the repository root.

```sh
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.build("Colmat")'
julia --project examples/poc.jl
```

To use this checkout from another Julia project, register it there as a local package.

```julia
using Pkg
Pkg.develop(path="/absolute/path/to/Colmat.jl")
Pkg.build("Colmat")
```

After building, run `using Colmat, LinearAlgebra` in a new Julia process.
Loading the module does not download or compile Rust.

## What the build produces

`deps/build.jl` performs the following steps in order.

1. If the Rust source is absent, fetch it into `deps/colmat-rs`.
2. Verify the checkout commit against the revision in `deps/upstream.toml`.
3. Build the BLAS shared library with `cargo build --release --locked`.
4. Install the library into `deps/usr/lib`.

The Rust source and Cargo dependencies are pinned by `deps/upstream.toml` and
`deps/rust/Cargo.lock`. The build does not automatically update the upstream HEAD.
It stops if the commit of an existing checkout does not match or if tracked files have
local changes.

Julia dependencies for the build live in `deps/Project.toml`, and those for the
documentation in `docs/Project.toml`. RustCall does not need to be installed.

Restart Julia after rebuilding the shared library.
See [Troubleshooting](@ref) for diagnostics.
