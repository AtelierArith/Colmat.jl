using Test, Colmat, LinearAlgebra, SparseArrays
@testset "Simultaneous generalized eigenvectors" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), n in (0, 1, 4), width in (1, 2)
        A = Matrix{T}(Diagonal(T.(1:n)))
        B = Matrix{T}(I, n, n)
        if n > 1
            A[1, 2] = -3
            A[2, 1] = 2
            B[1, 1] = 2
            B[1, 2] = T(0.2)
        end
        T <: Complex && (A .*= T(1 + 0.3im))
        a, b = FaerMatrix(A), FaerMatrix(B)
        FaerTeam(width) do team
            s = FaerScratch(eigen, a, b; side=:both, team)
            @test scratch_capacity(s) == scratch_requirements(eigen, a, b; side=:both, team)
            F = eigen(a, b; side=:both, scratch=s, team)
            @test keys(F) == (:values, :left, :right)
            @test A * F.right ≈ B * F.right * Diagonal(F.values) atol = 1e-3
            @test F.left' * A ≈ Diagonal(F.values) * F.left' * B atol = 1e-3
            @test all(v -> isapprox(norm(v), 1; atol=1e-4), eachcol(F.left))
            @test all(v -> isapprox(norm(v), 1; atol=1e-4), eachcol(F.right))
            H = eigen(a, b; side=:both, homogeneous=true, scratch=s, team)
            @test keys(H) == (:α, :β, :left, :right)
            @test A * H.right * Diagonal(H.β) ≈ B * H.right * Diagonal(H.α) atol = 1e-3
            @test Diagonal(H.β) * H.left' * A ≈ Diagonal(H.α) * H.left' * B atol = 1e-3
            G = eigen(
                a, b; side=:both, homogeneous=true, scratch=s, team, sortby=x -> -real(x)
            )
            @test issorted(real.(G.α ./ G.β); rev=true)
            @test A * G.right * Diagonal(G.β) ≈ B * G.right * Diagonal(G.α) atol = 1e-3
            @test Diagonal(G.β) * G.left' * A ≈ Diagonal(G.α) * G.left' * B atol = 1e-3
            @test Matrix(a) == A
            @test Matrix(b) == B
            close(s)
        end
    end
end
@testset "Both repeated, infinite and larger pairs" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        for (A, B) in (
            (Matrix{T}(I, 3, 3), Matrix{T}(I, 3, 3)),
            (Matrix{T}(Diagonal(T[2, 3, 4])), Matrix{T}(Diagonal(T[1, 0, 0]))),
        )
            a, b = FaerMatrix(A), FaerMatrix(B)
            FaerTeam(2) do team
                H = eigen(a, b; side=:both, homogeneous=true, team)
                @test all(isfinite, H.left)
                @test all(isfinite, H.right)
                @test A * H.right * Diagonal(H.β) ≈ B * H.right * Diagonal(H.α) atol = 1e-3
                @test Diagonal(H.β) * H.left' * A ≈ Diagonal(H.α) * H.left' * B atol = 1e-3
            end
        end
        n = 64
        C = Matrix{T}(SymTridiagonal(fill(2.0, n), fill(-1.0, n - 1)))
        X = reshape(T.(1:(n * n)), n, n) / (n * n)
        T <: Complex && (X .+= T(0.1im))
        B = X * X' + Matrix{T}(I, n, n)
        A = B * C
        a, b = FaerMatrix(A), FaerMatrix(B)
        FaerTeam(2) do team
            s = FaerScratch(eigen, a, b; side=:both, team)
            F = eigen(a, b; side=:both, scratch=s, team)
            @test real.(F.values) ≈ [2 - 2cos(k * pi / (n + 1)) for k in 1:n] atol = 3e-3
            @test A * F.right ≈ B * F.right * Diagonal(F.values) rtol = 1e-3 atol = 1e-3
            @test F.left' * A ≈ Diagonal(F.values) * F.left' * B rtol = 1e-3 atol = 1e-3
            close(s)
        end
    end
end
@testset "Both workspace and owner guards" begin
    a, b = FaerMatrix(Float32[1 -2; 3 1]), FaerMatrix([2.0 0; 0 1])
    F = eigen(a, b; side=:both)
    @test eltype(F.values) == ComplexF64
    @test F.left' * Matrix(a) ≈ Diagonal(F.values) * F.left' * Matrix(b)
    @test Matrix(a) * F.right ≈ Matrix(b) * F.right * Diagonal(F.values)
    s = FaerScratch(eigen, a, b)
    @test_throws ArgumentError eigen(a, b; side=:both, scratch=s)
    reserve!(s, eigen, a, b; side=:both)
    @test eigen(a, b; side=:both, scratch=s).values ≈ F.values
    @test_throws ArgumentError scratch_requirements(eigen, a, b; side=:invalid)
    close(a)
    @test_throws ArgumentError eigen(a, b; side=:both, scratch=s)
    close(s)
    @test isempty(
        Test.detect_ambiguities(Colmat, Base, LinearAlgebra, SparseArrays; recursive=true)
    )
end
