using Test, LinearAlgebra, Random, Colmat

@testset "t4a-faer via RustCall" begin
    @test v"0.9.1" <= pkgversion(Colmat.RustCall) < v"0.10"
    rng = MersenneTwister(42)
    @testset "matrix multiplication" begin
        for (m, k, n) in ((4, 3, 5), (0, 3, 2), (3, 0, 2), (3, 2, 0))
            a, b = randn(rng, m, k), randn(rng, k, n)
            @test faer_mul(a, b) ≈ a * b
        end
        a, b = randn(rng, 6, 4), randn(rng, 2, 3)
        @test faer_mul(@view(a[1:2:6, 1:2:4]), b) ≈ a[1:2:6, 1:2:4] * b
        @test faer_mul(transpose(a), a) ≈ a' * a
        @test faer_mul([1 2; 3 4], ones(Float32, 2, 1)) == [3.0; 7.0;;]
        @test_throws DimensionMismatch faer_mul(ones(2, 3), ones(2, 2))
        @test_throws Exception faer_mul(fill(NaN, 2, 2), ones(2, 2))
    end
    @testset "LU solve" begin
        a = [0.0 2 1; 3 1 2; 1 0 4] # Requires row pivoting.
        for b in (randn(rng, 3), randn(rng, 3, 2))
            original = copy(b)
            x = faer_solve(a, b)
            @test x ≈ a \ b
            @test a * x ≈ b
            @test b == original
        end
        @test size(faer_solve(zeros(0, 0), zeros(0, 2))) == (0, 2)
        @test size(faer_solve(a, zeros(3, 0))) == (3, 0)
        @test_throws Exception faer_solve([1.0 2; 2 4], ones(2))
        @test_throws Exception faer_solve([Inf 0.0; 0 1], ones(2))
        @test_throws DimensionMismatch faer_solve(ones(2, 3), ones(2))
        @test_throws DimensionMismatch faer_solve(a, ones(2))
    end
    @testset "thin SVD" begin
        for (m, n) in ((5, 3), (3, 5), (3, 3), (1, 1), (0, 3), (3, 0))
            a = randn(rng, m, n)
            original = copy(a)
            f = faer_svd(a)
            k = min(m, n)
            @test f isa SVD
            @test size(f.U) == (m, k)
            @test size(f.Vt) == (k, n)
            @test f.U * Diagonal(f.S) * f.Vt ≈ a
            @test f.S ≈ svdvals(a)
            @test f.U' * f.U ≈ Matrix{Float64}(I, k, k)
            @test f.Vt * f.Vt' ≈ Matrix{Float64}(I, k, k)
            @test a == original
        end
        @test faer_svd(zeros(4, 3)).S == zeros(3)
        @test_throws Exception faer_svd(fill(NaN, 2, 2))
    end
end
