# Run with: julia --project docs/reproducers/tridiagonal-rescaling.jl
# At Colmat main 5d63ec5: large input throws; small input has residual ~2.064.
# With real power-of-two restoration: both dimensionless residuals are small.
using Colmat,LinearAlgebra
A=Matrix{ComplexF64}(I,4,4)
A[2,1]=0.2+0.3im
A[1,2]=conj(A[2,1])
for scale in (2.0^1000,2.0^(-1000))
    F=tridiagonalize(A.*scale)
    try
        Q=Matrix(F.Q)
        residual=norm(A-Q*(Matrix(F.H)./scale)*Q')
        println("scale=",scale," dimensionless residual=",residual)
        @assert residual<=2048eps(Float64)*norm(A)
    finally
        close(F)
    end
end
