using Colmat, LinearAlgebra, Libdl
reference=only(
    filter(
        l->l.interface==:ilp64&&occursin("openblas", lowercase(l.libname)),
        BLAS.get_config().loaded_libs,
    ),
).libname
for (T, prefix) in ((Float64, "d"), (ComplexF64, "z"))
    @eval function geqp3_probe(::Type{$T}, p)
        a=fill($T(2), 1)
        tau=fill($T(778), 1)
        pivot=Int64[0]
        work=fill($T(777), 1)
        rwork=fill(Float64(779), 2)
        info=Ref{Int64}(-99)
        if $T<:Complex
            ccall(
                p,
                Cvoid,
                (
                    Ref{Int64},
                    Ref{Int64},
                    Ptr{$T},
                    Ref{Int64},
                    Ptr{Int64},
                    Ptr{$T},
                    Ptr{$T},
                    Ref{Int64},
                    Ptr{Float64},
                    Ptr{Int64},
                ),
                1,
                1,
                a,
                1,
                pivot,
                tau,
                work,
                0,
                rwork,
                info,
            )
        else
            ccall(
                p,
                Cvoid,
                (
                    Ref{Int64},
                    Ref{Int64},
                    Ptr{$T},
                    Ref{Int64},
                    Ptr{Int64},
                    Ptr{$T},
                    Ptr{$T},
                    Ref{Int64},
                    Ptr{Int64},
                ),
                1,
                1,
                a,
                1,
                pivot,
                tau,
                work,
                0,
                info,
            )
        end
        (; info=info[], work, a, tau, pivot, rwork)
    end
    for library in (Colmat.backend_library(), reference)
        p=Libdl.dlsym(Libdl.dlopen(library), prefix*"geqp3_64_")
        println((; library=basename(library), scalar=T, result=geqp3_probe(T, p)))
    end
end
