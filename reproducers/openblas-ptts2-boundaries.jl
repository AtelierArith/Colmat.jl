# Standalone PTTS2 diagnostic; Colmat is not loaded.
# Run: julia --startup-file=no reproducers/openblas-ptts2-boundaries.jl
# All zero-RHS probes allocate a dummy column explicitly. The one-element
# minimum-subnormal case has the representable exact solution one; reciprocal
# scaling in the reference can instead produce infinity.
using LinearAlgebra, Libdl
const reference_library=only(
    filter(
        l->l.interface==:ilp64&&occursin("openblas", lowercase(basename(l.libname))),
        BLAS.get_config().loaded_libs,
    ),
).libname
println("Reference: ", reference_library)
f=Libdl.dlsym(Libdl.dlopen(reference_library), :dptts2_64_)
d=[2.0, 3.0];
e=[0.25];
b=[777.0, 2.0, 7.0, 777.0, 777.0];
before=copy(b)
ccall(
    f,
    Cvoid,
    (Ref{Int64}, Ref{Int64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{Int64}),
    2,
    0,
    d,
    e,
    view(b, 2:5),
    4,
)
println((; case="real zero RHS", before, b))
f=Libdl.dlsym(Libdl.dlopen(reference_library), :zptts2_64_)
e=ComplexF64[0.25 + 0.5im];
b=ComplexF64[777, 2 + im, 7 - im, 777, 777];
before=copy(b)
ccall(
    f,
    Cvoid,
    (
        Ref{Int64},
        Ref{Int64},
        Ref{Int64},
        Ptr{Float64},
        Ptr{ComplexF64},
        Ptr{ComplexF64},
        Ref{Int64},
    ),
    0,
    2,
    0,
    d,
    e,
    view(b, 2:5),
    4,
)
println((; case="complex zero RHS", before, b))
f=Libdl.dlsym(Libdl.dlopen(reference_library), :dptts2_64_)
tiny=nextfloat(0.0);
d=[tiny];
b=[tiny]
ccall(
    f,
    Cvoid,
    (Ref{Int64}, Ref{Int64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{Int64}),
    1,
    1,
    d,
    Float64[],
    b,
    1,
)
println((; case="one tiny pivot", expected=1.0, b))
