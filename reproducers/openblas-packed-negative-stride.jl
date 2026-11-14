# Standalone OpenBLAS diagnostic; Colmat is deliberately not loaded.
# Run: julia --startup-file=no reproducers/openblas-packed-negative-stride.jl
# Julia 1.13.3 / OpenBLAS 0.3.30 on x86_64 macOS produces mismatches.
# In the 0.3.30 complex SPR/SPR2 interfaces, a FLOAT* is advanced by
# -(n-1)*inc rather than twice that count. With inc=-1 and this guard,
# all accesses remain within allocated storage, even in the broken path.
# https://github.com/OpenMathLib/OpenBLAS/blob/v0.3.30/interface/zspr.c#L119
# https://github.com/OpenMathLib/OpenBLAS/blob/v0.3.30/interface/zspr2.c#L121
using LinearAlgebra, Libdl

const reference_library=only(
    filter(
        l->l.interface==:ilp64 && occursin("openblas", lowercase(basename(l.libname))),
        BLAS.get_config().loaded_libs,
    ),
).libname
for (T, p) in ((ComplexF32, "c"), (ComplexF64, "z"))
    @eval function reference_update!(
        family, ap::Vector{$T}, x::Vector{$T}, y::Vector{$T}, inc
    )
        f=Libdl.dlsym(Libdl.dlopen(reference_library), Symbol($p*family*"_64_"))
        if family=="spr"
            ccall(
                f,
                Cvoid,
                (Ref{UInt8}, Ref{Int64}, Ref{$T}, Ptr{$T}, Ref{Int64}, Ptr{$T}),
                'U',
                3,
                one($T),
                pointer(x, 2),
                inc,
                ap,
            )
        else
            ccall(
                f,
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{Int64},
                    Ref{$T},
                    Ptr{$T},
                    Ref{Int64},
                    Ptr{$T},
                    Ref{Int64},
                    Ptr{$T},
                ),
                'U',
                3,
                one($T),
                pointer(x, 2),
                inc,
                pointer(y, 2),
                inc,
                ap,
            )
        end
    end
end
println("Reference: ", reference_library)
for T in (ComplexF32, ComplexF64), family in ("spr", "spr2")
    x=T[T(10, 11), T(1, 2), T(3, 4), T(5, 6), T(12, 13)]
    y=T[T(7, 8), T(-1, 2), T(2, -3), T(4, 5), T(9, 10)]
    logical_x=reverse(x[2:4])
    logical_y=reverse(y[2:4])
    expected=T[
        if family=="spr"
            logical_x[i]*logical_x[j]
        else
            logical_x[i]*logical_y[j]+logical_y[i]*logical_x[j]
        end for j in 1:3 for i in 1:j
    ]
    negative=zeros(T, 6)
    positive=zeros(T, 6)
    reference_update!(family, negative, x, y, -1)
    reference_update!(
        family, positive, vcat(T(0), logical_x, T(0)), vcat(T(0), logical_y, T(0)), 1
    )
    println(
        T,
        " ",
        family,
        ": negative agrees=",
        negative==expected,
        ", equivalent positive agrees=",
        positive==expected,
        ", first negative=",
        negative[1],
        ", expected=",
        expected[1],
    )
end
