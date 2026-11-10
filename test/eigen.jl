using Test, LinearAlgebra, Random, Colmat
@testset "Native eigen and Pfaffian" begin
    rng = MersenneTwister(31)
    for a in ([0.0 -1; 1 0], [1.0 2 0; 0 3 4; 0 0 5], randn(rng, 5, 5), zeros(0, 0))
        A = FaerMatrix(a)
        f = eigen(A)
        @test a * f.vectors ≈ f.vectors * Diagonal(f.values)
        @test eigvals(A) ≈ f.values
        @test f.values ≈ eigvals(a)
        @test Matrix(A) == a
        @test eigen(A; sortby=x -> (-real(x), imag(x))).values ≈
            sort(f.values; by=x -> (-real(x), imag(x)))
    end
    for n in (0, 1, 4), uplo in (:L, :U)
        a = randn(rng, n, n)
        a = a + a'
        A = FaerMatrix(a)
        H = Hermitian(A, uplo)
        f = eigen(H)
        @test f.values ≈ eigvals(Hermitian(a))
        @test a * f.vectors ≈ f.vectors * Diagonal(f.values)
        @test f.vectors' * f.vectors ≈ Matrix{Float64}(I, n, n)
        @test eigvals(H) ≈ f.values
        b = randn(rng, n, n)
        b = b * b' + Matrix{Float64}(I, n, n)
        B = FaerMatrix(b)
        g = eigen(H, Hermitian(B, uplo))
        @test a * g.vectors ≈ b * g.vectors * Diagonal(g.values)
        @test g.vectors' * b * g.vectors ≈ Matrix{Float64}(I, n, n)
        @test eigvals(H, Hermitian(B, uplo)) ≈ g.values
    end
    a = [1.0 2; 3 4]
    b = [2.0 0; 0 3]
    f = eigen(FaerMatrix(a), FaerMatrix(b))
    @test a * f.vectors ≈ b * f.vectors * Diagonal(f.values)
    @test eigvals(FaerMatrix(a), FaerMatrix(b)) ≈ f.values
    A = FaerMatrix([4.0 1; 1 3])
    A[1, 2] = NaN
    @test eigvals(Hermitian(A, :L)) ≈ eigvals(Hermitian([4.0 1; 1 3]))
    @test_throws Exception eigen(A)
    @test_throws DimensionMismatch eigen(FaerMatrix(ones(2, 3)))
    @test_throws DimensionMismatch eigen(FaerMatrix(ones(2, 2)), FaerMatrix(ones(3, 3)))
    @test pfaffian(FaerMatrix([0.0 -3; 3 0])) ≈ -3
    @test pfaffian(FaerMatrix(zeros(0, 0))) == 1
    for n in (2, 4, 8)
        a = randn(rng, n, n)
        a = a - a'
        p = pfaffian(FaerMatrix(a))
        @test p^2 ≈ det(a)
    end
    @test_throws Exception pfaffian(FaerMatrix(ones(3, 3)))
    @test_throws Exception pfaffian(FaerMatrix(ones(2, 2)))
    @test isempty(Test.detect_ambiguities(Colmat; recursive=true))
end
