using Colmat, LinearAlgebra, Libdl
reference=only(
    filter(
        l->l.interface==:ilp64&&occursin("openblas", lowercase(l.libname)),
        BLAS.get_config().loaded_libs,
    ),
).libname
for library in (Colmat.backend_library(), reference), lwork in (-1, 1)
    p=Libdl.dlsym(Libdl.dlopen(library), :dgelqf_64_)
    work=[777.0]
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
        3,
        0,
        C_NULL,
        3,
        C_NULL,
        work,
        lwork,
        info,
    )
    println((; library=basename(library), lwork, info=info[], work))
end
