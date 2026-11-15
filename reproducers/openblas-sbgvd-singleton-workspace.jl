# Run in a fresh process without importing Colmat:
# julia --startup-file=no --project reproducers/openblas-sbgvd-singleton-workspace.jl
# Reproduced with Julia 1.13.3 / OpenBLAS_jll 0.3.30 on macOS x86_64.
# Reference-LAPACK v3.12.1 SRC/dsbgvd.f declares LWORK=1 for N=1,
# but uses WORK(INDWRK=2) and WORK(INDWK2=3) in the vector path.
using LinearAlgebra, Libdl
provider=BLAS.lbt_find_backing_library("dsbgvd_", :ilp64)
@assert occursin("openblas", lowercase(basename(provider.libname)))
println((julia=VERSION, backend=provider.libname))
f=BLAS.lbt_get_forward("dsbgvd_", :ilp64)
a=reshape([3.0], 1, 1);
b=reshape([2.0], 1, 1);
w=[777.0];
z=fill(777.0, 1, 1)
# Three allocated payload elements make the probe safe even if the routine
# writes beyond its declared LWORK=1. The outer elements are separate guards.
work=fill(777.0, 5);
iw=fill(Int64(-777), 3);
info=Ref{Int64}(-777)
ccall(
    f,
    Cvoid,
    (
        Ref{UInt8},
        Ref{UInt8},
        Ref{Int64},
        Ref{Int64},
        Ref{Int64},
        Ptr{Float64},
        Ref{Int64},
        Ptr{Float64},
        Ref{Int64},
        Ptr{Float64},
        Ptr{Float64},
        Ref{Int64},
        Ptr{Float64},
        Ref{Int64},
        Ptr{Int64},
        Ref{Int64},
        Ref{Int64},
        Csize_t,
        Csize_t,
    ),
    UInt8(86),
    UInt8(76),
    1,
    0,
    0,
    a,
    1,
    b,
    1,
    w,
    z,
    1,
    view(work, 2:4),
    1,
    view(iw, 2:2),
    1,
    info,
    1,
    1,
)
println((info=info[], values=w, vectors=z, work, iw))
@assert work[1]==work[5]==777

@assert info[]==0 && isapprox(w[1], 1.5)
@assert any(!=(777.0), work[3:4]) # Allocated slack beyond declared LWORK=1.
@assert !isapprox(2*z[1, 1]^2, 1.0) # The original B=[2] normalization fails.
println("Reproduced: write beyond declared LWORK and invalid singleton normalization")
