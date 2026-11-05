# Run: julia --startup-file=no --project docs/reproducers/tridiagonal-relative-phase.jl
# Main dc1df07 throws "tridiagonalization overflowed the working type" despite
# finite Hermitian inputs: the bridge divides by a complex-typed tiny norm.
using Colmat, LinearAlgebra

scaled(z, x) = complex(real(z)/x, imag(z)/x)

for T in (ComplexF32, ComplexF64)
    R=typeof(real(zero(T)))
    scales=R===Float32 ? (R(1e-30),) : (R(1e-300),)
    for x in scales
        z=T(0, x)
        A=T[1 conj(z); z 2]
        F=tridiagonalize(A)
        try
            Q=Matrix(F.Q)
            reconstructed=Q*Matrix(F.H)*adjoint(Q)
            @assert all(isfinite, Q)
            @assert norm(adjoint(Q)*Q-I)<=128eps(R)
            @assert abs(scaled(reconstructed[2, 1], x)-scaled(z, x))<=128eps(R)
            println((
                type = T,
                scale = x,
                scaled_offdiagonal_error = abs(scaled(reconstructed[2, 1], x)-scaled(z, x)),
            ))
        finally
            close(F)
        end
    end
end
