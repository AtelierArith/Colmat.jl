using Test, LinearAlgebra, Random, Colmat
@testset "Native reductions" begin
    rng = MersenneTwister(23)
    for (m, n) in ((3, 4), (3, 3), (1, 1), (0, 3), (3, 0), (0, 0))
        a = randn(rng, m, n)
        A = FaerMatrix(a)
        @test sum(A) ≈ sum(a)
        @test Matrix(sum(A; dims=1)) ≈ sum(a; dims=1)
        @test Matrix(sum(A; dims=2)) ≈ sum(a; dims=2)
        @test dot(A, A) ≈ dot(a, a)
        for p in (1, 2, Inf)
            @test norm(A, p) ≈ norm(a, p)
            @test opnorm(A, p) ≈ opnorm(a, p)
        end
        @test all(isfinite, A)
        @test !any(isnan, A)
        if !isempty(a)
            @test minimum(A) ≈ minimum(a)
            @test maximum(A) ≈ maximum(a)
            @test all(isapprox.(extrema(A), extrema(a)))
        end
        if m == n
            @test det(A) ≈ det(a)
            @test tr(A) ≈ tr(a)
        end
        @test Matrix(kron(A, A)) ≈ kron(a, a)
    end
    A = FaerMatrix([1.0 0; 0 1])
    @test isone(A)
    @test issymmetric(A)
    @test ishermitian(A)
    A[1, 2] = NaN
    @test !all(isfinite, A)
    @test any(isnan, A)
    @test isnan(maximum(A))
    @test isnan(minimum(A))
    @test !isone(A)
    @test !issymmetric(A)
    @test_throws DimensionMismatch det(FaerMatrix(ones(2, 3)))
    @test_throws DimensionMismatch tr(FaerMatrix(ones(2, 3)))
    # Julia pairs equal-length arrays in column-major order.
    @test dot(FaerMatrix(ones(2, 3)), FaerMatrix(ones(3, 2))) == 6.0
    @test_throws DimensionMismatch dot(FaerMatrix(ones(2, 3)), FaerMatrix(ones(3, 3)))
    @test_throws ArgumentError minimum(FaerMatrix(zeros(0, 2)))
    @test_throws ArgumentError maximum(FaerMatrix(zeros(2, 0)))
end
