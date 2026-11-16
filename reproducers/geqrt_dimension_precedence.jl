using Colmat, LinearAlgebra, Libdl
reference=only(
    filter(
        l->l.interface==:ilp64&&occursin("openblas", lowercase(l.libname)),
        BLAS.get_config().loaded_libs,
    ),
).libname
for family in ("geqrt2", "geqrt3"), library in (Colmat.backend_library(), reference)
    p=Libdl.dlsym(Libdl.dlopen(library), Symbol("d"*family*"_64_"))
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
        -1,
        -2,
        a,
        1,
        t,
        1,
        info,
    )
    println((; family, library=basename(library), info=info[], a, t))
end
