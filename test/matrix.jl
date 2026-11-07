using Test, LinearAlgebra, Random, Colmat

@testset "Rust-owned FaerMatrix" begin
    a = [4.0 1 2; 1 5 1; 2 1 6]
    b = [1.0 2 3; 0 2 1; 3 1 4]
    A, B = FaerMatrix(a), FaerMatrix(b)
    @test A isa AbstractMatrix{Float64}
    @test isempty(Test.detect_ambiguities(Colmat; recursive = true))
    @test size(A) == size(a)
    @test axes(A) == axes(a)
    @test eltype(A) === Float64
    @test Matrix(A) == a
    @test Array(A) == a
    @test Matrix{Float32}(A) == Float32.(a)
    @test collect(A) == a
    @test norm(A) ≈ norm(a)
    @test dot(A, B) ≈ dot(a, b)
    @test A[2, 3] == a[2, 3]
    @test A[4] == a[4]
    @test_throws BoundsError A[0, 1]
    @test_throws BoundsError A[4, 1]
    @test_throws BoundsError (A[1, 4] = 2)
    @test_throws BoundsError A[-1]

    @testset "ownership and mutation" begin
        source = copy(a)
        C = FaerMatrix(source)
        source[1] = 99
        @test C[1] == a[1]
        snapshot = Matrix(C)
        snapshot[1] = 98
        @test C[1] == a[1]
        for D in (copy(C), deepcopy(C), FaerMatrix(C))
            @test D !== C
            D[1] = 97
            @test C[1] == a[1]
            @test D[1] == 97
        end
        C[1, 2] = 42
        @test C[1, 2] == 42
        @test copyto!(C, C) === C
        @test copyto!(C, A) === C
        @test Matrix(C) == a
        @test fill!(C, 2) === C
        @test Matrix(C) == fill(2.0, size(a))
        copyto!(C, b)
        @test Matrix(C) == b
        @test Matrix(FaerMatrix(@view a[1:2:3, :])) == a[1:2:3, :]
        @test Matrix(FaerMatrix(a')) == a'
        @test Matrix(FaerMatrix([1 2; 3 4])) == [1 2; 3 4]
        @test all(isnan, Matrix(FaerMatrix(fill(NaN, 2, 2))))
        @test similar(A) isa FaerMatrix
        @test similar(A, Float32) isa FaerMatrix{Float32}
        @test similar(A, Float64, (2,)) isa Vector{Float64}
        @test size(similar(A, Float64, (2, 4))) == (2, 4)
        @test_throws ArgumentError similar(A, Float64, (-1, 2))
    end

    @testset "vector space operations" begin
        for (actual, expected) in (
            (A+B, a+b),
            (A-B, a-b),
            (-A, -a),
            (+A, a),
            (2A, 2a),
            (A*3, a*3),
            (A/2, a/2),
            (2\A, a/2),
            (zero(A), zero(a)),
            (A+A, 2a),
        )
            @test actual isa FaerMatrix
            @test Matrix(actual) ≈ expected
        end
        C = FaerMatrix(reverse(a; dims = 1))
        @test Matrix((A+B)+C) ≈ Matrix(A+(B+C))
        @test Matrix(A+B) ≈ Matrix(B+A)
        @test Matrix(A+zero(A)) ≈ a
        @test Matrix(A+(-A)) ≈ zero(a)
        @test Matrix(2*(A+B)) ≈ Matrix(2A+2B)
        @test Matrix((2+3)*A) ≈ Matrix(2A+3A)
        @test Matrix(A) == a
        @test Matrix(B) == b
        @test_throws DimensionMismatch A + FaerMatrix(ones(2, 3))
        @test_throws DimensionMismatch A - FaerMatrix(ones(3, 2))
        @test Matrix(A .+ 2 .* B) ≈ a .+ 2 .* b
        @test Matrix(copy(A')) == a'
        @test Matrix(copy(transpose(A))) == transpose(a)
    end

    @testset "products and solves" begin
        @test A*B isa FaerMatrix
        @test Matrix(A*B) ≈ a*b
        @test Matrix(A*A) ≈ a*a
        @test A*[1, 2, 3] ≈ a*[1, 2, 3]
        @test_throws DimensionMismatch A*FaerMatrix(ones(2, 3))
        @test_throws DimensionMismatch A*ones(2)
        @test A\B isa FaerMatrix
        @test Matrix(A\B) ≈ a\b
        @test Matrix(A\b) ≈ a\b
        @test A\[1, 2, 3] ≈ a\[1, 2, 3]
        @test Matrix(A\A) ≈ Matrix{Float64}(I, 3, 3)
        @test_throws DimensionMismatch A\ones(2)
        @test FaerMatrix(ones(2, 3))\ones(2) ≈ fill(1/3, 3)
        @test_throws Exception FaerMatrix([1.0 2; 2 4])\ones(2)
        @test Matrix(faer_mul(A, B)) ≈ a*b
        @test Matrix(faer_solve(A, B)) ≈ a\b
        @test faer_svd(A).S ≈ svdvals(a)
    end

    @testset "mul! and aliasing" begin
        C = copy(A)
        @test mul!(C, A, B) === C
        @test Matrix(C) ≈ a*b
        fill!(C, NaN)
        mul!(C, A, B, 2.0, 0.0)
        @test Matrix(C) ≈ 2a*b
        C = copy(A)
        mul!(C, A, B, 2.0, 3.0)
        @test Matrix(C) ≈ 2a*b+3a
        C = copy(A)
        mul!(C, C, B)
        @test Matrix(C) ≈ a*b
        C = copy(B)
        mul!(C, A, C, 2.0, 3.0)
        @test Matrix(C) ≈ 2a*b+3b
        C = copy(A)
        mul!(C, C, C, 2.0, 3.0)
        @test Matrix(C) ≈ 2a*a+3a
        N = fill!(copy(A), NaN)
        C = copy(B)
        mul!(C, N, N, 0.0, 2.0)
        @test Matrix(C) == 2b
        mul!(C, N, N, 0.0, 0.0)
        @test Matrix(C) == zeros(3, 3)
        @test_throws DimensionMismatch mul!(similar(A, Float64, (2, 2)), A, B)
    end

    @testset "empty matrices" begin
        for (m, k, n) in ((0, 3, 2), (3, 0, 2), (3, 2, 0))
            x, y = zeros(m, k), zeros(k, n)
            X, Y = FaerMatrix(x), FaerMatrix(y)
            @test Matrix(X+X) == x+x
            @test Matrix(2X) == 2x
            @test Matrix(X*Y) == x*y
        end
        @test size(FaerMatrix(zeros(0, 0))\FaerMatrix(zeros(0, 2))) == (0, 2)
    end

    @testset "lifecycle and concurrent access" begin
        C = copy(A)
        D = copy(C)
        @test isopen(C)
        @test close(C) === nothing
        @test !isopen(C)
        @test close(C) === nothing
        @test_throws ArgumentError Matrix(C)
        @test_throws ArgumentError C[1, 1]
        @test_throws ArgumentError (C[1, 1] = 1)
        @test_throws ArgumentError copy(C)
        @test_throws ArgumentError C+A
        @test_throws ArgumentError A+C
        @test_throws ArgumentError copyto!(C, C)
        @test Matrix(D) == a
        # Temporary native owners must stay alive for nested operations and GC.
        for _ = 1:10
            R = (FaerMatrix(a)+FaerMatrix(b))*FaerMatrix(a)
            GC.gc()
            @test Matrix(R) ≈ (a+b)*a
        end
        tasks = [Threads.@spawn Matrix(A*B+2A) for _ = 1:8]
        @test all(r -> r ≈ a*b+2a, fetch.(tasks))
    end
end

@testset "LinearAlgebra SVD dispatch" begin
    rng = MersenneTwister(7)
    for (m, n) in ((5, 3), (3, 5), (3, 3), (1, 1), (0, 3), (3, 0), (0, 0))
        a = randn(rng, m, n)
        A = FaerMatrix(a)
        k = min(m, n)
        for full in (false, true)
            f = svd(A; full)
            @test f isa SVD
            @test size(f.U) == (m, full ? m : k)
            @test size(f.Vt) == (full ? n : k, n)
            @test f.S ≈ svdvals(a)
            @test f.U[:, 1:k] * Diagonal(f.S) * f.Vt[1:k, :] ≈ a
            @test f.U' * f.U ≈ Matrix{Float64}(I, size(f.U, 2), size(f.U, 2))
            @test f.Vt * f.Vt' ≈ Matrix{Float64}(I, size(f.Vt, 1), size(f.Vt, 1))
            @test Matrix(A) == a
        end
        @test svdvals(A) ≈ svdvals(a)
        @test faer_svd(A).S ≈ svd(A).S
        f = svd(A)
        close(A)
        @test f.S ≈ svdvals(a) # Factors do not borrow the input owner.
        @test_throws ArgumentError svd(A)
        @test_throws ArgumentError svdvals(A)
    end
    A = FaerMatrix(zeros(4, 3))
    @test svd(A).S == zeros(3)
    @test svdvals(A) == zeros(3)
    fill!(A, NaN)
    @test_throws Exception svd(A)
    @test_throws Exception svdvals(A)
    @test isempty(Test.detect_ambiguities(Colmat; recursive = true))
end
