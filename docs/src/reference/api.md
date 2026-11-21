# API

Colmat exports no names. Use Base / LinearAlgebra for matrix operations.
Call configuration functions with the `Colmat.` prefix.

| Function | Purpose |
|---|---|
| `Colmat.load_backend(; verbose=false)` | Re-register the provided BLAS with LBT |
| `Colmat.backend_library()` | Return the absolute path of the BLAS shared library |
| `Colmat.get_num_threads()` | Return Colmat's configured execution width |
| `Colmat.set_num_threads(n)` | Set Colmat's execution width for matrix multiplication |

Because `load_backend()` runs on load, ordinary use does not require explicit
registration. See also [Backends and fallbacks](@ref) and [Thread settings](@ref).

```@docs
Colmat
Colmat.load_backend
Colmat.backend_library
Colmat.get_num_threads
Colmat.set_num_threads
```
