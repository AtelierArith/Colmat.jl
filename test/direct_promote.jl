using Colmat, Test, LinearAlgebra, Random

function _dp_factor(kind, A)
    if kind === :lu
        lu!(A; scratch=FaerScratch(lu!, A))
    elseif kind === :full_lu
        full_piv_lu!(A; scratch=FaerScratch(full_piv_lu!, A))
    elseif kind === :qr
        qr!(A; scratch=FaerScratch(qr!, A))
    elseif kind === :pivoted_qr
        qr!(A, ColumnNorm(); scratch=FaerScratch(qr!, A, ColumnNorm()))
    elseif kind === :cholesky
        cholesky!(Hermitian(A); scratch=FaerScratch(cholesky!, Hermitian(A)))
    elseif kind === :ldlt
        ldlt!(Hermitian(A); scratch=FaerScratch(ldlt!, Hermitian(A)))
    else
        bunchkaufman!(Hermitian(A); scratch=FaerScratch(bunchkaufman!, Hermitian(A)))
    end
end

@testset "Direct factors promote to the right-hand side's scalar type" begin
    rng = MersenneTwister(6)
    n = 6
    for (T, S) in (
            (Float32, Float64),
            (Float32, ComplexF64),
            (Float32, ComplexF32),
            (ComplexF32, ComplexF64),
            (ComplexF32, Float64),
            (Float64, ComplexF64),
        ),
        kind in (:lu, :full_lu, :qr, :pivoted_qr, :cholesky, :ldlt, :bunchkaufman)

        X = randn(rng, T, n, n)
        M = kind in (:cholesky, :ldlt, :bunchkaufman) ? X * X' + n * I : X + n * I
        A = FaerMatrix(copy(M))
        F = _dp_factor(kind, A)
        packed = Matrix(A)
        P = promote_type(T, S)
        tol = 500eps(real(T))
        ref = Matrix{P}(M)
        for b in (randn(rng, S, n), randn(rng, S, n, 2))
            x = F \ b
            @test eltype(x) === P
            @test isapprox(Array(x), ref \ b; rtol=tol, atol=tol)
            @test isapprox(Array(F' \ b), ref' \ b; rtol=tol, atol=tol)
            @test isapprox(Array(transpose(F) \ b), transpose(ref) \ b; rtol=tol, atol=tol)
        end
        # The direct factor's storage is not touched by the promoted solve.
        @test Matrix(A) == packed
    end
    # A narrower right-hand side keeps the factor's type.
    A = FaerMatrix(randn(rng, n, n) + n * I)
    F = lu!(A; scratch=FaerScratch(lu!, A))
    @test eltype(F \ randn(rng, Float32, n)) === Float64
end
