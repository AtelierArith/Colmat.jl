using Test, Colmat, LinearAlgebra, SparseArrays
@testset "Native indefinite destructive decompositions" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), n in (0, 1, 4), uplo in (:L, :U)
        X = reshape(T.(1:(n * n)), n, n)
        T <: Complex && (X .+= T(0.25im))
        S = X * X' + Matrix{T}(I, n, n)
        for fn in (ldlt!, bunchkaufman!)
            A = FaerMatrix(S)
            v = view(A, :, :)
            F = fn(Hermitian(A, uplo))
            @test Matrix(F) ≈ S rtol = 1e-4
            @test Matrix(A) ≈ (uplo === :L ? Matrix(F.packed) : Matrix(F.packed)')
            @test Matrix(v) == Matrix(A)
            @test diag(Matrix(A)) ≈ diag(Matrix(F.D))
            L = if uplo === :L
                Matrix(UnitLowerTriangular(Matrix(A)))
            else
                Matrix(UnitUpperTriangular(Matrix(A)))'
            end
            reconstructed = L * Matrix(F.D) * L'
            expected = fn === bunchkaufman! ? S[F.p, F.p] : S
            @test reconstructed ≈ expected rtol = 1e-4
            b = ones(T, n)
            @test S * (F \ b) ≈ b atol = 2e-3
            fill!(A, zero(T))
            close(A)
            @test Matrix(F) ≈ S rtol = 1e-4
        end
    end
    for T in (Float32, Float64, ComplexF32, ComplexF64), uplo in (:L, :U)
        x = T(2) + (T <: Complex ? T(0.5im) : zero(T))
        S = T[0 x; conj(x) 0]
        A = FaerMatrix(S)
        F = bunchkaufman!(Hermitian(A, uplo))
        @test Matrix(F.D)[1, 2] != 0
        @test Matrix(F) ≈ S
        @test Matrix(F.L * F.D * F.L') ≈ S[F.p, F.p]
        @test Matrix(A) ≈ (uplo === :L ? Matrix(F.packed) : Matrix(F.packed)')
    end
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        for fn in (ldlt!, bunchkaufman!)
            A = FaerMatrix(T[0 0; 0 0])
            saved = Matrix(A)
            @test_throws SingularException fn(A)
            @test Matrix(A) == saved
            F = fn(A; check=false)
            @test !issuccess(F)
            if fn === ldlt!
                @test Matrix(A) == saved
            else
                @test Matrix(A) == Matrix(F.packed)
            end
        end
    end
    @test isempty(
        Test.detect_ambiguities(Colmat, Base, LinearAlgebra, SparseArrays; recursive=true)
    )
end
@testset "Full pivot LU writeback" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        dims in ((0, 0), (0, 3), (3, 0), (3, 4), (4, 3), (4, 4))

        A = reshape(T.(1:prod(dims)), dims)
        for i in 1:min(dims...)
            A[i, i] += T(10)
        end
        T <: Complex && (A .+= T(0.5im))
        native = FaerMatrix(A)
        F = full_piv_lu!(native; check=false)
        @test Matrix(native) == Matrix(F.packed)
        @test Matrix(F.L * F.U) ≈ A[F.p, F.q]
        @test Matrix(F) ≈ A
        close(native)
        @test Matrix(F) ≈ A
    end
end
