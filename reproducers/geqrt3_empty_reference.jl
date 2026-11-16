# Run as an isolated subprocess: some reference builds recurse at N=0.
using Colmat, LinearAlgebra, Libdl
reference=only(
    filter(
        l->l.interface==:ilp64&&occursin("openblas", lowercase(l.libname)),
        BLAS.get_config().loaded_libs,
    ),
).libname
for library in (Colmat.backend_library(), reference)
    println("Calling GEQRT3 N=0: ", basename(library))
    flush(stdout)
    p=Libdl.dlsym(Libdl.dlopen(library), :dgeqrt3_64_)
    a=[777.0]
    t=[778.0]
    info=Ref{Int64}(-99)
    ccall(
        p,
        Cvoid,
        (
            Ref{Int64},
            Ref{Int64},
            Ptr{Float64},
            Ref{Int64},
            Ptr{Float64},
            Ref{Int64},
            Ref{Int64},
        ),
        0,
        0,
        a,
        1,
        t,
        1,
        info,
    )
    println((; library=basename(library), info=info[], a, t))
    flush(stdout)
end
