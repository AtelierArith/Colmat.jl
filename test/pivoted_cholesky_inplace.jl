using Colmat, Test, LinearAlgebra, Random

@testset "Pivoted Cholesky factors A in place with inplace=true" begin
    rng=MersenneTwister(12)
    n=9
    for T in (Float32, Float64, ComplexF32, ComplexF64), tol in (-1.0, 0.0)
        X=randn(rng, T, n, n)
        S=X*X'+I
        tolr=1000eps(real(T))
        for wrap in (identity, A->Hermitian(A, :L), A->(T<:Real ? Symmetric(A, :L) : Hermitian(A, :L)))
            A=FaerMatrix(copy(S))
            ref=cholesky(wrap(FaerMatrix(copy(S))), RowMaximum(); tol)
            F=cholesky!(wrap(A), RowMaximum(); inplace = true, tol)
            @test F isa LinearAlgebra.CholeskyPivoted
            @test F.factors===A
            @test F.rank==n
            @test F.p==ref.p
            # Same kernel as the retained factor: identical L.
            @test Matrix(F.factors)≈Matrix(ref.factors) rtol = tolr
            @test Matrix(F)≈S rtol = tolr
            b=randn(rng, T, n)
            @test F\b≈S\b rtol = sqrt(eps(real(T)))
            close(ref.factors)
        end
        # Caller scratch and a team for the automatic-threshold kernel.
        if tol<0
            A=FaerMatrix(copy(S))
            s=FaerScratch(cholesky!, A, RowMaximum(); tol)
            FaerTeam(2) do team
                F=cholesky!(A, RowMaximum(); inplace = true, tol, scratch = s, team)
                @test F.factors===A
                @test Matrix(F)≈S rtol = tolr
            end
            close(s)
        end
    end
    # Rank-deficient input: check=false keeps the rank, check=true throws.
    Y=randn(rng, 8, 3)
    D=Y*Y'
    A=FaerMatrix(copy(D))
    F=cholesky!(A, RowMaximum(); inplace = true, tol = -1.0, check = false)
    @test F.rank==3
    @test Matrix(F)≈D rtol = 1e-8 atol = 1e-8
    @test_throws LinearAlgebra.RankDeficientException cholesky!(
        FaerMatrix(copy(D)),
        RowMaximum();
        inplace = true,
        tol = -1.0,
    )
    # The upper triangle cannot be read in place.
    @test_throws ArgumentError cholesky!(
        Hermitian(FaerMatrix(copy(D)), :U),
        RowMaximum();
        inplace = true,
    )
    # inplace=false (default) keeps the staged-writeback meaning.
    A=FaerMatrix(copy(D+I))
    F=cholesky!(A, RowMaximum())
    @test F.factors!==A
end

@testset "In-place pivoted Cholesky allocates no n×n Julia staging" begin
    n=300
    X=randn(n, n)
    S=X*X'+n*I
    A=FaerMatrix(copy(S))
    s=FaerScratch(cholesky!, A, RowMaximum(); tol = -1.0)
    f=()->(copyto!(A, S); cholesky!(A, RowMaximum(); inplace = true, tol = -1.0, scratch = s))
    f()
    @test @allocated(f())<n*n*sizeof(Float64)÷4
    close(s)
end
