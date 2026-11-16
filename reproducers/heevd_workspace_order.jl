# Multiple invalid workspaces must report the first invalid argument.
using Colmat, LinearAlgebra, Libdl

function execute(pointer)
    work=zeros(ComplexF64, 32)
    rw=zeros(1)
    iw=zeros(Int64, 1)
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
            Ptr{Int64},
            Ref{Int64},
            Ref{Int64},
        ),
        UInt8('V'),
        UInt8('L'),
        3,
        C_NULL,
        3,
        C_NULL,
        work,
        32,
        rw,
        0,
        iw,
        0,
        info,
    )
    return info[]
end

native=Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :zheevd_64_)
reference=only(
    filter(
        l->l.interface==:ilp64 && occursin("openblas", lowercase(l.libname)),
        BLAS.get_config().loaded_libs,
    ),
)
external=Libdl.dlsym(Libdl.dlopen(reference.libname), :zheevd_64_)
@assert native!=external
println((colmat=execute(native), openblas=execute(external)))
