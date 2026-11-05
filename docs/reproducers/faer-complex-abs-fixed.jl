# Consumer checks for tensor4all/faer#274, fixed by #275 at 1621d94.
# Run after julia --project deps/build.jl selects that revision or a later fix.
using Colmat, LinearAlgebra

for R in (Float32, Float64)
    values=R===Float32 ?
           [
        reinterpret(Float32, UInt32(1)),
        reinterpret(Float32, UInt32(1<<9)),
        floatmax(Float32),
    ] :
           [
        reinterpret(Float64, UInt64(1)),
        reinterpret(Float64, UInt64(1<<4)),
        floatmax(Float64),
    ]
    for x in values, z in (complex(x, zero(R)), complex(zero(R), x))
        a=FaerMatrix(reshape([z], 1, 1))
        try
            @assert norm(a, 1)==x
            @assert norm(a, Inf)==x
        finally
            close(a)
        end
    end
end
println("24 public complex norm endpoint checks passed")
