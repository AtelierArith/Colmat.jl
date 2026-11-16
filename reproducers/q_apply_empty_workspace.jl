using Colmat, LinearAlgebra, Libdl
reference=only(
    filter(
        l->l.interface==:ilp64&&occursin("openblas", lowercase(l.libname)),
        BLAS.get_config().loaded_libs,
    ),
).libname
for T in (Float64, ComplexF64)
    @eval function empty_q_application(::Type{$T}, p, work, lwork)
        info=Ref{Int64}(-777)
        ccall(
            p,
            Cvoid,
            (
                Ref{UInt8},
                Ref{UInt8},
                Ref{Int64},
                Ref{Int64},
                Ref{Int64},
                Ptr{$T},
                Ref{Int64},
                Ptr{$T},
                Ptr{$T},
                Ref{Int64},
                Ptr{$T},
                Ref{Int64},
                Ref{Int64},
            ),
            UInt8('L'),
            UInt8('N'),
            0,
            3,
            0,
            C_NULL,
            1,
            C_NULL,
            C_NULL,
            1,
            work,
            lwork,
            info,
        )
        info[]
    end
end
for (T, prefix, stem) in ((Float64, "d", "orm"), (ComplexF64, "z", "unm")),
    orientation in ("qr", "lq", "ql", "rq"),
    library in (Colmat.backend_library(), reference),
    lwork in (-1, 3)

    name=prefix*stem*orientation*"_64_"
    p=Libdl.dlsym(Libdl.dlopen(library), name)
    work=fill(T(777), 4)
    info=empty_q_application(T, p, work, lwork)
    println((; name, library=basename(library), lwork, info, work1=work[1], guard=work[4]))
end
