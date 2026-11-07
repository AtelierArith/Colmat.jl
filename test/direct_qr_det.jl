using Colmat, Test, LinearAlgebra, Random

@testset "Determinants of direct in-place QR factors" begin
    rng=MersenneTwister(5)
    for T in (Float32, Float64, ComplexF32, ComplexF64), n in (1, 2, 5, 40, 70), piv in (false, true)
        A0=randn(rng, T, n, n)
        A=FaerMatrix(copy(A0))
        F=piv ? qr!(A, ColumnNorm(); scratch = FaerScratch(qr!, A, ColumnNorm())) :
          qr!(A; scratch = FaerScratch(qr!, A))
        la, ph=logabsdet(F)
        ra, rp=logabsdet(A0)
        tol=200*n*eps(real(T))
        @test la≈ra rtol=tol atol=tol
        @test ph≈rp rtol=sqrt(eps(real(T)))
        @test typeof(ph)===typeof(rp)
        n<=40 && @test det(F)≈det(A0) rtol=tol
        # The storage is read in place: A still holds the packed factor.
        @test F.factors===A
    end
    for T in (Float64, ComplexF64)
        Z=FaerMatrix(zeros(T, 3, 3))
        F=qr!(Z; scratch = FaerScratch(qr!, Z))
        @test logabsdet(F)==(-Inf, zero(T))
        @test iszero(det(F))
        E=FaerMatrix(Matrix{T}(I, 4, 4))
        @test det(qr!(E; scratch = FaerScratch(qr!, E)))≈one(T)
        # A permutation matrix has determinant ±1.
        P=FaerMatrix(Matrix{T}(I, 4, 4)[[2, 1, 3, 4], :])
        @test det(qr!(P, ColumnNorm(); scratch = FaerScratch(qr!, P, ColumnNorm())))≈-one(T)
    end
    R=FaerMatrix(randn(rng, 5, 3))
    F=qr!(R; scratch = FaerScratch(qr!, R))
    @test_throws DimensionMismatch det(F)
end
