# Run without Colmat: julia --startup-file=no reproducers/openblas-bevd-singleton-upper.jl
# A=[2], UPLO=U, KD=1 stores its diagonal in AB(2,1). AB(1,1) is unused.
using LinearAlgebra, Libdl
reference=only(
    filter(
        l->l.interface==:ilp64&&occursin("openblas", lowercase(basename(l.libname))),
        BLAS.get_config().loaded_libs,
    ),
).libname
println("Julia ", VERSION, "; ", Sys.MACHINE, "; ", reference)
for (T, prefix, family) in (
    (Float32, "s", "sbevd"),
    (Float64, "d", "sbevd"),
    (ComplexF32, "c", "hbevd"),
    (ComplexF64, "z", "hbevd"),
)
    R=real(T)
    signature=Any[
        Ref{UInt8},
        Ref{UInt8},
        Ref{Int64},
        Ref{Int64},
        Ptr{T},
        Ref{Int64},
        Ptr{R},
        Ptr{T},
        Ref{Int64},
        Ptr{T},
        Ref{Int64},
    ]
    arguments=Any[:('V'), :('U'), 1, 1, :ab, 2, :w, :z, 1, :work, 1]
    if T<:Complex
        append!(signature, [Ptr{R}, Ref{Int64}])
        append!(arguments, [:rwork, 1])
    end
    append!(signature, [Ptr{Int64}, Ref{Int64}, Ref{Int64}])
    append!(arguments, [:iwork, 1, :info])
    symbol=Symbol(prefix*family*"_64_")
    @eval let
        ab=$T[NaN, 2]
        w=zeros($R, 1)
        z=zeros($T, 1)
        work=zeros($T, 1)
        rwork=zeros($R, 1)
        iwork=zeros(Int64, 1)
        info=Ref{Int64}(-99)
        f=Libdl.dlsym(Libdl.dlopen(reference), $(QuoteNode(symbol)))
        ccall(f, Cvoid, $(Expr(:tuple, signature...)), $(arguments...))
        println($(string(T)), ": INFO=", info[], " W=", w[1], "; expected 2")
    end
end
