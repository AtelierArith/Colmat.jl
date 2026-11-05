# Run from the package root: julia --project docs/reproducers/bidiagonal-rescaling.jl
using Colmat, LinearAlgebra

A=Matrix{ComplexF64}(I, 4, 4)
A[2, 1]=0.2+0.3im
A[1, 2]=0.4-0.1im
for exponent in (1000, -1000)
    scale=2.0^exponent
    F=bidiagonalize(A .* scale)
    try
        U, V=Matrix(F.U), Matrix(F.V)
        residual=norm(A-U*(Matrix(F.B) ./ scale)*V')/norm(A)
        println((exponent = exponent, residual = residual))
        @assert residual<=4096eps(Float64)
    finally
        close(F)
    end
end
