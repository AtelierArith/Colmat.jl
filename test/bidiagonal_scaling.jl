using Test, Colmat, LinearAlgebra

@testset "Bidiagonal real rescaling" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        R = typeof(real(zero(T)))
        exponent = R === Float32 ? 120 : 1000
        for (m, n) in ((4, 4), (5, 3), (3, 5))
            a = Matrix{T}(I, m, n)
            a[2, 1] = T <: Complex ? T(0.2, 0.3) : T(0.2)
            a[1, 2] = T <: Complex ? T(0.4, -0.1) : T(0.4)
            for scale in (one(R), R(2)^exponent, R(2)^(-exponent)),
                bs in (0, 1, 2),
                full in (false, true)

                source = a .* scale
                F = bidiagonalize(source; blocksize=bs, full)
                try
                    U = Matrix(F.U)
                    V = Matrix(F.V)
                    k = min(m, n)
                    b = Matrix(F.B) ./ scale
                    @test all(isfinite, F.B.dv)
                    @test all(isfinite, F.B.ev)
                    @test all(isfinite, U)
                    @test all(isfinite, V)
                    @test norm(a - U[:, 1:k] * b * V[:, 1:k]') <= 4096eps(R) * norm(a)
                    @test U'U ≈ Matrix{T}(I, size(U, 2), size(U, 2)) atol = 4096eps(R)
                    @test V'V ≈ Matrix{T}(I, size(V, 2), size(V, 2)) atol = 4096eps(R)
                    @test source == a .* scale
                finally
                    close(F)
                end
            end
        end
    end
end
