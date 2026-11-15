# Standalone provider diagnostic; Colmat is not loaded.
# Run: julia --startup-file=no reproducers/openblas-sfrk-zero-product.jl
# Linux CI with Julia 1.13.4 observed NaNs in the off-diagonal RFP block
# for alpha=0, beta=0.5 and unused A containing NaNs. macOS can agree with
# the finite control, so this script reports the behavior without requiring
# a provider-specific failure. Both problems must mathematically give C/2.
using LinearAlgebra, Libdl

const reference_library=only(
    filter(
        l->l.interface==:ilp64 && occursin("openblas", lowercase(basename(l.libname))),
        BLAS.get_config().loaded_libs,
    ),
).libname

for (T, p) in ((Float32, "s"), (Float64, "d"))
    @eval function reference_sfrk(a::Matrix{$T})
        f=Libdl.dlsym(Libdl.dlopen(reference_library), Symbol($p*"sfrk_64_"))
        c=ones($T, 3)
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{UInt8},
                Ref{UInt8},
                Ref{Int64},
                Ref{Int64},
                Ref{$T},
                Ptr{$T},
                Ref{Int64},
                Ref{$T},
                Ptr{$T},
            ),
            'N',
            'U',
            'N',
            2,
            1,
            zero($T),
            a,
            2,
            $T(0.5),
            c,
        )
        return c
    end
end

println("Reference: ", reference_library)
for T in (Float32, Float64)
    poisoned=reference_sfrk(fill(T(NaN), 2, 1))
    finite=reference_sfrk(zeros(T, 2, 1))
    @assert finite==fill(T(0.5), 3)
    println(T, ": unused NaN A -> ", poisoned, "; finite A -> ", finite)
end
