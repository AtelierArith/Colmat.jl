# Exact eigenvalues are 0, 2, 4. The interval (2, 10] must select only 4.
using Colmat, LinearAlgebra, Libdl

function execute(pointer)
    a=ComplexF64[2 1+im 0; 1-im 2 1+im; 0 1-im 2]
    w=zeros(3)
    z=zeros(ComplexF64, 3, 3)
    support=zeros(Int64, 6)
    work=zeros(ComplexF64, 128)
    rw=zeros(128)
    iw=zeros(Int64, 64)
    m=Ref{Int64}(-1)
    info=Ref{Int64}(0)
    ccall(
        pointer,
        Cvoid,
        (
            Ref{UInt8},
            Ref{UInt8},
            Ref{UInt8},
            Ref{Int64},
            Ptr{ComplexF64},
            Ref{Int64},
            Ref{Float64},
            Ref{Float64},
            Ref{Int64},
            Ref{Int64},
            Ref{Float64},
            Ref{Int64},
            Ptr{Float64},
            Ptr{ComplexF64},
            Ref{Int64},
            Ptr{Int64},
            Ptr{ComplexF64},
            Ref{Int64},
            Ptr{Float64},
            Ref{Int64},
            Ptr{Int64},
            Ref{Int64},
            Ref{Int64},
        ),
        UInt8('V'),
        UInt8('V'),
        UInt8('L'),
        3,
        a,
        3,
        2.0,
        10.0,
        1,
        3,
        0.0,
        m,
        w,
        z,
        3,
        support,
        work,
        length(work),
        rw,
        length(rw),
        iw,
        length(iw),
        info,
    )
    return (info=info[], m=m[], values=w[1:max(0, m[])])
end

native=Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :zheevr_64_)
reference=only(
    filter(
        l->l.interface==:ilp64 && occursin("openblas", lowercase(l.libname)),
        BLAS.get_config().loaded_libs,
    ),
)
external=Libdl.dlsym(Libdl.dlopen(reference.libname), :zheevr_64_)
@assert native!=external
println("Colmat: ", execute(native))
expected=execute(external)
@assert expected.info==0 && expected.m==1
println("OpenBLAS: ", expected)
