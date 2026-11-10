using Test, Colmat, LinearAlgebra, SparseArrays
@testset "Staged native destructive factorizations" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        dims in ((0, 0), (0, 3), (3, 0), (1, 1), (4, 3), (3, 4), (4, 4))

        m, n = dims
        X = reshape(T.(1:(m * n)), dims)
        for i in 1:min(m, n)
            X[i, i] += T(10)
        end
        T <: Complex && (X .+= T(0.25im))
        for pivot in (NoPivot(), ColumnNorm())
            A = FaerMatrix(X)
            v = view(A, :, :)
            F = qr!(A, pivot)
            @test Matrix(F) ≈ X
            @test Matrix(A) == Matrix(F.packed)
            @test Matrix(v) == Matrix(A)
            if min(m, n) > 0
                sequence = FaerHouseholderSequence(F.Qbasis, F.Qcoeff)
                @test Matrix(sequence) ≈ Matrix(F.Q)
            end
            @test size(F.Qbasis) == (m, min(m, n))
            @test size(F.Qcoeff, 2) == min(m, n)
            saved = Matrix(F)
            fill!(A, zero(T))
            close(A)
            @test Matrix(F) ≈ saved
        end
        A = FaerMatrix(X)
        F = lu!(A; check=false)
        @test Matrix(F) ≈ X
        @test Matrix(A) == Matrix(F.packed)
    end
    for T in (Float32, Float64, ComplexF32, ComplexF64), n in (0, 1, 4), uplo in (:L, :U)
        X = reshape(T.(1:(n * n)), n, n)
        T <: Complex && (X .+= T(0.25im))
        S = X * X' + Matrix{T}(I, n, n)
        A = FaerMatrix(S)
        F = cholesky!(Hermitian(A, uplo))
        @test Matrix(F) ≈ S
        @test Matrix(A) ≈ (uplo === :L ? Matrix(F.L) : Matrix(F.U))
        B = FaerMatrix(S)
        G = cholesky!(Hermitian(B, uplo), RowMaximum(); tol=-1)
        @test Matrix(G) ≈ S rtol = 1e-4
        @test Matrix(B) ≈ (uplo === :L ? Matrix(G.L) : Matrix(G.U))
        close(A)
        close(B)
        @test Matrix(F) ≈ S
        @test Matrix(G) ≈ S rtol = 1e-4
    end
    A = FaerMatrix([1.0 2; 2 1])
    saved = Matrix(A)
    @test_throws PosDefException cholesky!(A)
    @test Matrix(A) == saved
    F = cholesky!(A; check=false)
    @test !issuccess(F)
    @test Matrix(A) == saved
    A = FaerMatrix([1.0 2; 2 4])
    saved = Matrix(A)
    @test_throws SingularException lu!(A)
    @test Matrix(A) == saved
    F = lu!(A; allowsingular=true)
    @test !issuccess(F)
    @test Matrix(A) == Matrix(F.packed)
    @test isempty(
        Test.detect_ambiguities(Colmat, Base, LinearAlgebra, SparseArrays; recursive=true)
    )
end
