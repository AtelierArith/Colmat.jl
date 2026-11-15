# Standalone GTSV diagnostic; Colmat is not loaded.
# Run: julia --startup-file=no reproducers/openblas-gtsv-zero-rhs.jl
# OpenBLAS 0.3.30 performs back substitution into column one at NRHS=0.
# All accesses below remain inside the allocated vector. Passing an empty
# Julia matrix instead would let this reference path access invalid memory.
# A correct zero-RHS call can factor D but must leave every B entry unchanged.
using LinearAlgebra, Libdl
lib=only(
    filter(
        l->l.interface==:ilp64&&occursin("openblas", lowercase(basename(l.libname))),
        BLAS.get_config().loaded_libs,
    ),
).libname
f=Libdl.dlsym(Libdl.dlopen(lib), :dgtsv_64_)
dl=[1.0];
d=[2.0, 2.0];
du=[1.0];
b=[777.0, 3.0, 4.0, 777.0, 777.0];
info=Ref{Int64}(-1)
ccall(
    f,
    Cvoid,
    (
        Ref{Int64},
        Ref{Int64},
        Ptr{Float64},
        Ptr{Float64},
        Ptr{Float64},
        Ptr{Float64},
        Ref{Int64},
        Ref{Int64},
    ),
    2,
    0,
    dl,
    d,
    du,
    view(b, 2:5),
    4,
    info,
)
println((; info=info[], d, b))
