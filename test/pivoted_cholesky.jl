using Test, Colmat, LinearAlgebra, SparseArrays
@testset "Pivoted native Cholesky" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        tol in (-1.0, 0.0, 1e-4),
        n in (0, 1, 4)

        X = reshape(T.(1:(n * n)), n, n)
        T <: Complex && (X .+= T(0.5im))
        A = X * X' + Matrix{T}(I, n, n)
        F = cholesky(Hermitian(FaerMatrix(A)), RowMaximum(); tol)
        @test F isa CholeskyPivoted
        @test F.factors isa FaerMatrix{T}
        @test F.rank == n
        @test Matrix(F) ≈ A rtol = 1e-4
        @test Matrix(F.L) * Matrix(F.L)' ≈ A[F.p, F.p] rtol = 1e-4
        @test Matrix(inv(F)) * A ≈ Matrix{T}(I, n, n) atol = 1e-3
        b = ones(T, n)
        @test A * (F \ b) ≈ b atol = 1e-3
        C = copy(F)
        close(F)
        @test isopen(C)
        @test !isopen(F)
    end
    for tol in (-1.0, 0.0, 1e-4)
        A = FaerMatrix([4.0 2; 2 1])
        F = cholesky(A, RowMaximum(); tol, check=false)
        @test F.rank == 1
        @test Matrix(F) ≈ Matrix(A)
        @test_throws RankDeficientException cholesky(A, RowMaximum(); tol)
        @test_throws RankDeficientException inv(F)
    end
    @test isempty(
        Test.detect_ambiguities(Colmat, Base, LinearAlgebra, SparseArrays; recursive=true)
    )
end
@testset "Pivoted tolerance, selected triangles and ownership" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        A = Matrix(Diagonal(T[1, 1e-8, 4]))
        F = cholesky(FaerMatrix(A), RowMaximum(); tol=1e-6, check=false)
        @test F.rank == 2
        @test F.p[1] == 3
        @test Matrix(F) ≈ Matrix(Diagonal(T[1, 0, 4]))
        G = cholesky(FaerMatrix(A), RowMaximum(); tol=0)
        @test G.rank == 3
        @test Matrix(G) ≈ A
        Z = cholesky(FaerMatrix(zeros(T, 3, 3)), RowMaximum(); tol=-1, check=false)
        @test Z.rank == 0
        @test Matrix(Z) == zeros(T, 3, 3)
        # Only the selected half is authoritative; nonfinite unused entries are ignored.
        M = T[4 1; 1 3]
        T <: Complex && (M[2, 1] += T(0.5im); M[1, 2]=conj(M[2, 1]))
        for uplo in (:L, :U)
            native = FaerMatrix(M)
            native[uplo === :L ? 1 : 2, uplo === :L ? 2 : 1] = T(NaN)
            H = cholesky(Hermitian(native, uplo), RowMaximum(); tol=-1)
            @test Matrix(H) ≈ M
            b = ones(T, 2, 2)
            @test M * (H \ b) ≈ b
            @test Matrix(inv(H)) * M ≈ I atol = 1e-5
            native[1, 1] = T(99)
            @test Matrix(H) ≈ M
            close(native)
            @test Matrix(H) ≈ M
        end
        F = cholesky(FaerMatrix(T[4 1; 1 3]), RowMaximum())
        B = FaerMatrix(ones(T, 2, 2))
        @test ldiv!(F, B) === B
        @test T[4 1; 1 3] * Matrix(B) ≈ ones(T, 2, 2)
        close(F)
        @test_throws ArgumentError Matrix(F)
    end
    @test_throws ArgumentError cholesky(FaerMatrix([1.0 2; 3 4]), RowMaximum())
    @test_throws ArgumentError cholesky(FaerMatrix(ones(2, 2)), RowMaximum(); tol=NaN)
end
