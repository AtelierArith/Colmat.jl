# Julia --startup-file=no reproducers/potrf_ignored_storage.jl [path/to/library]
# Own Fortran adapter regression: ignored triangle/padding and diagonal imaginary parts.
using Libdl

library = if isempty(ARGS)
    joinpath(@__DIR__, "..", "deps", "usr", "lib", "libcolmat_blas."*Libdl.dlext)
else
    only(ARGS)
end
entry = Libdl.dlsym(Libdl.dlopen(library), :zpotrf_)
a = fill(ComplexF64(NaN), 15)
a[1] = ComplexF64(4, NaN)
a[2] = 2+1.4im
a[3] = 4-0.8im
a[7] = ComplexF64(10.49, NaN)
a[8] = -1.28+0.6im
a[13] = ComplexF64(21.8, NaN)
saved = copy(a)
selected = Set([1, 2, 3, 7, 8, 13])
info = Ref{Int32}(-99)
ccall(
    entry,
    Cvoid,
    (Ref{UInt8}, Ref{Int32}, Ptr{ComplexF64}, Ref{Int32}, Ref{Int32}),
    UInt8('L'),
    3,
    a,
    5,
    info,
)
changed = [i for i in eachindex(a) if i∉selected && !isequal(a[i], saved[i])]
finite = all(z -> isfinite(real(z)) && isfinite(imag(z)), a[collect(selected)])
println(
    "INFO=",
    info[],
    "; changed ignored positions=",
    changed,
    "; finite factor entries=",
    finite,
)
@assert info[]==0
@assert isempty(changed)
@assert finite
