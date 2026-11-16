# HEEV assumes imaginary diagonal components are zero, including stored NaNs.
using Colmat, LinearAlgebra, Libdl

function execute(pointer)
    a=ComplexF64[
        i==j ? complex(4+i, NaN) : complex(0.2(i+j), 0.3(i-j)) for i in 1:3, j in 1:3
    ]
    w=fill(-99.0, 3)
    work=zeros(ComplexF64, 32)
    rwork=zeros(32)
    info=Ref{Int64}(0)
    ccall(
        pointer,
        Cvoid,
        (
            Ref{UInt8},
            Ref{UInt8},
            Ref{Int64},
            Ptr{ComplexF64},
            Ref{Int64},
            Ptr{Float64},
            Ptr{ComplexF64},
            Ref{Int64},
            Ptr{Float64},
            Ref{Int64},
        ),
        UInt8('V'),
        UInt8('L'),
        3,
        a,
        3,
        w,
        work,
        length(work),
        rwork,
        info,
    )
    return (info=info[], values=w, finite=all(isfinite, a))
end

native=Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :zheev_64_)
reference=only(
    filter(
        l->l.interface==:ilp64 && occursin("openblas", lowercase(l.libname)),
        BLAS.get_config().loaded_libs,
    ),
)
external=Libdl.dlsym(Libdl.dlopen(reference.libname), :zheev_64_)
@assert native!=external
println("Colmat: ", execute(native))
println("OpenBLAS: ", execute(external))
