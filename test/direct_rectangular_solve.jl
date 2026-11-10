using Colmat, Test, LinearAlgebra, Random

_drs_factor(kind, A) =
    if kind === :lu
        lu!(A; scratch=FaerScratch(lu!, A))
    elseif kind === :full_lu
        full_piv_lu!(A; scratch=FaerScratch(full_piv_lu!, A))
    elseif kind === :qr
        qr!(A; scratch=FaerScratch(qr!, A))
    else
        qr!(A, ColumnNorm(); scratch=FaerScratch(qr!, A, ColumnNorm()))
    end

@testset "Rectangular direct factors solve through the pseudoinverse" begin
    rng = MersenneTwister(4)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        (m, n) in ((7, 4), (4, 7)),
        kind in (:lu, :full_lu, :qr, :pivoted_qr)

        X = randn(rng, T, m, n)
        F = _drs_factor(kind, FaerMatrix(copy(X)))
        tol = sqrt(eps(real(T)))
        b = randn(rng, T, m)
        B = randn(rng, T, m, 3)
        bt = randn(rng, T, n)
        @test isapprox(F \ b, X \ b; rtol=tol, atol=tol)
        @test (F \ b) isa Vector{T}
        @test isapprox(Matrix(F \ B), X \ B; rtol=tol, atol=tol)
        @test isapprox(F' \ bt, X' \ bt; rtol=tol, atol=tol)
        @test isapprox(transpose(F) \ bt, transpose(X) \ bt; rtol=tol, atol=tol)
        @test_throws DimensionMismatch F \ zeros(T, m + 1)
        # In-place ldiv! still needs a square factor.
        @test_throws Union{ArgumentError,DimensionMismatch} ldiv!(F, zeros(T, m))
    end
end
