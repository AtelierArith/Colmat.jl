using Colmat, LinearAlgebra, Libdl
reference=only(
    filter(
        l->l.interface==:ilp64&&occursin("openblas", lowercase(l.libname)),
        BLAS.get_config().loaded_libs,
    ),
).libname
for library in (Colmat.backend_library(), reference)
    p=Libdl.dlsym(Libdl.dlopen(library), :dgeqrf_64_)
    work=fill(777.0, 3)
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
            Ptr{Float64},
            Ref{Int64},
            Ptr{Int64},
        ),
        0,
        3,
        C_NULL,
        1,
        C_NULL,
        work,
        3,
        info,
    )
    println((; library=basename(library), info=info[], work))
end
