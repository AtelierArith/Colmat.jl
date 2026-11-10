using Test, Colmat, LinearAlgebra
@testset "Tridiagonal real rescaling" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        R = typeof(real(zero(T)))
        exponent = R === Float32 ? 120 : 1000
        a = Matrix{T}(I, 4, 4)
        a[2, 1] = T <: Complex ? T(0.2, 0.3) : T(0.2)
        a[1, 2] = conj(a[2, 1])
        for scale in (one(R), R(2)^exponent, R(2)^(-exponent)), bs in (0, 1, 2)
            source = a .* scale
            F = tridiagonalize(source; blocksize=bs)
            try
                Q = Matrix(F.Q)
                h = Matrix(F.H) ./ scale
                @test all(isfinite, F.H.dv)
                @test all(isfinite, F.H.ev)
                @test all(isfinite, Q)
                @test norm(a - Q * h * Q') <= 2048eps(R) * norm(a)
                @test Q'Q ≈ Matrix{T}(I, 4, 4) atol = 2048eps(R)
                @test source == a .* scale
            finally
                close(F)
            end
        end
    end
end
