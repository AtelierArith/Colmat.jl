using Test, Colmat, LinearAlgebra, SparseArrays

function _indefinite_hermitian(::Type{T}, n) where {T}
    X = reshape(T.(1:(n * n)) ./ n^2, n, n)
    T <: Complex && (X .+= T(0.3im) .* reshape(T.((n * n):-1:1), n, n) ./ n^2)
    H = X + X'
    for i in 1:n
        H[i, i] = real(H[i, i]) + (isodd(i) ? 3 : -2)
    end
    return Matrix(Hermitian(H))
end

@testset "Rook-pivoted Bunch-Kaufman" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), n in (0, 1, 5), uplo in (:L, :U)
        S = _indefinite_hermitian(T, n)
        tol = T <: Union{Float32,ComplexF32} ? 1e-4 : 1e-10
        b = ones(T, n)
        for rook in (false, true)
            A = FaerMatrix(S)
            F = bunchkaufman(Hermitian(A, uplo), rook)
            @test F isa FaerFactorization{T,:bunchkaufman}
            @test F.rook === rook
            @test :rook in propertynames(F)
            @test issuccess(F)
            @test Matrix(F) ≈ S rtol = tol
            @test Matrix(F.L * F.D * F.L') ≈ S[F.p, F.p] rtol = tol atol = tol
            @test S * (F \ b) ≈ b rtol = tol atol = tol
            @test inertia(F) == inertia(bunchkaufman(Hermitian(S), rook))
            @test copy(F).rook === rook
            if uplo === :L
                G = bunchkaufman(FaerMatrix(S), rook)
                @test G isa FaerFactorization && G.rook === rook
                @test Matrix(G) ≈ S rtol = tol
            end
            B = FaerMatrix(S)
            v = view(B, :, :)
            W = bunchkaufman!(Hermitian(B, uplo), rook)
            @test W isa FaerFactorization && W.rook === rook
            @test Matrix(B) ≈ (uplo === :L ? Matrix(W.packed) : Matrix(W.packed)')
            @test Matrix(v) == Matrix(B)
            close(B)
            @test Matrix(W) ≈ S rtol = tol
            if T === Float32
                # Promoted solves refactor with the saved pivoting strategy.
                @test S * (F \ ones(Float64, n)) ≈ ones(n) rtol = 1e-4 atol = 1e-4
            end
        end
        n == 0 && continue
        Z = FaerMatrix(zeros(T, n, n))
        @test_throws SingularException bunchkaufman(Z, true)
        @test_throws SingularException bunchkaufman!(Z, true)
        @test Matrix(Z) == zeros(T, n, n)
        F = bunchkaufman(Z, true; check=false)
        @test !issuccess(F) && F.rook && inertia(F) == (0, 0, n)
    end
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        # faer's rook search moves the large off-diagonal pivot forward here,
        # while the default partial-diagonal strategy keeps the identity order.
        S = T[1e-3 1 0; 1 1e-3 100; 0 100 1e-3]
        @test bunchkaufman(FaerMatrix(S)).p == [1, 2, 3]
        @test bunchkaufman(FaerMatrix(S), true).p == [2, 3, 1]
        @test bunchkaufman!(FaerMatrix(S), true).p == [2, 3, 1]
    end
    FaerTeam(2) do team
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            S = _indefinite_hermitian(T, 6)
            F = bunchkaufman(Hermitian(FaerMatrix(S)), true; team)
            @test F.rook
            @test Matrix(F) ≈ S rtol = 1e-4
            G = bunchkaufman!(FaerMatrix(S), true; team)
            @test G.rook
            @test Matrix(G) ≈ S rtol = 1e-4
        end
    end
    S = _indefinite_hermitian(Float64, 6)
    F = bunchkaufman(Symmetric(FaerMatrix(S)), true)
    @test F isa FaerFactorization && F.rook
    @test inertia(F) == inertia(bunchkaufman(Symmetric(S), true))
    @test inertia(F; atol=100.0) == (0, 0, 6)
end

@testset "Non-Hermitian Cholesky check=false" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        N = T[4 1; 2 3]
        @test cholesky(N; check=false).info == -1
        A = FaerMatrix(N)
        @test_throws PosDefException cholesky(A)
        F = cholesky(A; check=false)
        @test F isa FaerFactorization{T,:cholesky}
        @test F.info == -1 && !issuccess(F) && isopen(F)
        @test !isposdef(A)
        saved = Matrix(A)
        @test cholesky!(A; check=false).info == -1
        @test Matrix(A) == saved
        @test_throws DimensionMismatch cholesky(FaerMatrix(ones(T, 2, 3)); check=false)
        @test cholesky(A, NoPivot(); check=false).info == -1
        for uplo in (:L, :U)
            P = T[4 1; 1 3]
            @test cholesky(Hermitian(FaerMatrix(P), uplo), NoPivot()) isa FaerFactorization
            D = FaerMatrix(P)
            H = cholesky!(Hermitian(D, uplo), NoPivot())
            @test H isa FaerFactorization && Matrix(H) ≈ P
            @test cholesky!(FaerMatrix(P), NoPivot()) isa FaerFactorization
            G = cholesky(Hermitian(FaerMatrix(P), uplo))
            @test Matrix(G.UL) ≈ Matrix(uplo === :U ? G.U : G.L)
            @test :UL in propertynames(G)
        end
    end
end

@testset "Complex Symmetric wrappers" begin
    for T in (ComplexF32, ComplexF64), uplo in (:L, :U)
        S = T[4 1 + 2im; 1 + 2im 3]
        A = FaerMatrix(S)
        saved = Matrix(A)
        @test_throws PosDefException cholesky(Symmetric(A, uplo))
        F = cholesky(Symmetric(A, uplo); check=false)
        @test F.info == -1 && !issuccess(F)
        @test cholesky(Symmetric(S, uplo); check=false).info == -1
        @test !isposdef(Symmetric(A, uplo))
        @test cholesky!(Symmetric(A, uplo); check=false).info == -1
        @test cholesky!(Symmetric(A, uplo), NoPivot(); check=false).info == -1
        @test Matrix(A) == saved
        @test_throws ArgumentError ldlt(Symmetric(A, uplo))
        @test_throws ArgumentError bunchkaufman(Symmetric(A, uplo))
        @test_throws ArgumentError bunchkaufman(Symmetric(A, uplo), true)
        @test_throws ArgumentError ldlt!(Symmetric(A, uplo))
        @test_throws ArgumentError bunchkaufman!(Symmetric(A, uplo))
        @test Matrix(A) == saved
        @test_throws ArgumentError bunchkaufman(A)
        @test_throws ArgumentError ldlt(A)
        # Real stored triangle: the wrapper is also Hermitian, as in Julia.
        R = T[4 1; 1 3]
        garbage = uplo === :L ? T[0 5im; 0 0] : T[0 0; 5im 0]
        B = FaerMatrix(R + garbage)
        @test isposdef(Symmetric(B, uplo))
        C = cholesky(Symmetric(B, uplo))
        @test C isa FaerFactorization && Matrix(C) ≈ R
        @test Matrix(ldlt(Symmetric(B, uplo))) ≈ R
        K = bunchkaufman(Symmetric(B, uplo), true)
        @test K isa FaerFactorization && K.rook && Matrix(K) ≈ R
        @test inertia(K) == (2, 0, 0)
        cholesky!(Symmetric(B, uplo))
        @test Matrix(B) ≈ (uplo === :L ? Matrix(C.packed) : Matrix(C.packed)')
    end
    @test isempty(
        Test.detect_ambiguities(Colmat, Base, LinearAlgebra, SparseArrays; recursive=true)
    )
end
