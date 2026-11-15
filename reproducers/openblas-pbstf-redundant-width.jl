# Standalone PBSTF diagnostic; Colmat is not loaded.
# Run: julia --startup-file=no reproducers/openblas-pbstf-redundant-width.jl
# Reference PBSTF sets m=(n+kd)/2 without bounding m by n. With n=1,kd=4,
# it attempts column two and can return INFO=2, outside the declared problem.
# Allocate that dummy column explicitly so all provider accesses stay within
# allocated storage. Only the first column belongs to the logical matrix.
using LinearAlgebra, Libdl
lib=only(
    filter(
        l->l.interface==:ilp64&&occursin("openblas", lowercase(basename(l.libname))),
        BLAS.get_config().loaded_libs,
    ),
).libname
f=Libdl.dlsym(Libdl.dlopen(lib), :dpbstf_64_)
a=zeros(7, 2);
a[5, 1]=4;
a[5, 2]=-1;
info=Ref{Int64}(-777)
ccall(
    f,
    Cvoid,
    (Ref{UInt8}, Ref{Int64}, Ref{Int64}, Ptr{Float64}, Ref{Int64}, Ref{Int64}),
    'U',
    1,
    4,
    a,
    7,
    info,
)
println((; n=1, kd=4, info=info[], diagonals=a[5, :]))
