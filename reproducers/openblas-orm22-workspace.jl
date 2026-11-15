# Run without Colmat. Reference-LAPACK 3.12.1 ORM22/UNM22 prose specifies
# minimum N for SIDE=L and M for SIDE=R; executable code instead requires
# NQ (M for L, N for R), except when one block is empty.
using LinearAlgebra, Libdl
reference=only(
    filter(
        l->l.interface==:ilp64&&occursin("openblas", lowercase(basename(l.libname))),
        BLAS.get_config().loaded_libs,
    ),
).libname
println("Julia ", VERSION, "; ", Sys.MACHINE, "; ", reference)
for (T, prefix, family) in (
    (Float32, "s", "orm22"),
    (Float64, "d", "orm22"),
    (ComplexF32, "c", "unm22"),
    (ComplexF64, "z", "unm22"),
)
    symbol=Symbol(prefix*family*"_64_")
    @eval let
        f=Libdl.dlsym(Libdl.dlopen(reference), $(QuoteNode(symbol)))
        for side in ('L', 'R'), lw in (2, 5)
            m, n=side=='L' ? (5, 2) : (2, 5)
            q=Matrix{$T}(I, 5, 5)
            c=ones($T, m, n)
            work=zeros($T, lw)
            info=Ref{Int64}(-99)
            ccall(
                f,
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{UInt8},
                    Ref{Int64},
                    Ref{Int64},
                    Ref{Int64},
                    Ref{Int64},
                    Ptr{$T},
                    Ref{Int64},
                    Ptr{$T},
                    Ref{Int64},
                    Ptr{$T},
                    Ref{Int64},
                    Ref{Int64},
                ),
                side,
                'N',
                m,
                n,
                2,
                3,
                q,
                5,
                c,
                m,
                work,
                lw,
                info,
            )
            println($(string(T)), " SIDE=", side, " LWORK=", lw, " INFO=", info[])
            @assert info[]==(lw==2 ? -12 : 0)
        end
    end
end
