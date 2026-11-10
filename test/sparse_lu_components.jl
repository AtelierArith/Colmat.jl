using Test, Colmat, LinearAlgebra, SparseArrays
@testset "Public simplicial sparse LU components" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        format in (:csc, :csr),
        ordering in (:identity, [3, 1, 2])

        A = T[0 2 1; 3 1 0; 0 1 4]
        T <: Complex && (A[1, 2] += T(0.5im); A[3, 3] += T(0.25im))
        native = FaerSparseMatrix(sparse(A); format)
        F = lu(native; algorithm=:simplicial, ordering)
        L, U, p, q = F.L, F.U, Vector(F.p), Vector(F.q)
        @test L isa FaerSparseMatrix{T}
        @test U isa FaerSparseMatrix{T}
        @test Matrix(L * U) ≈ A[p, q]
        @test istril(Matrix(L))
        @test istriu(Matrix(U))
        @test diag(Matrix(L)) == ones(T, 3)
        b = ones(T, 3, 2)
        @test A * Matrix(F \ b) ≈ b
        @test transpose(A) * Matrix(transpose(F) \ b) ≈ b
        @test A' * Matrix(adjoint(F) \ b) ≈ b
        plan = FaerSparseSymbolic(native; kind=:lu, algorithm=:simplicial, ordering)
        G = factorize(plan, native)
        @test Matrix(G.L * G.U) ≈ A[Vector(G.p), Vector(G.q)]
        C = copy(F)
        close(F)
        @test Matrix(C.L * C.U) ≈ A[Vector(C.p), Vector(C.q)]
        @test_throws ArgumentError F.L
        close(native)
        @test Matrix(C.L * C.U) ≈ A[Vector(C.p), Vector(C.q)]
    end
    @test isempty(
        Test.detect_ambiguities(Colmat, Base, LinearAlgebra, SparseArrays; recursive=true)
    )
end
@testset "Simplicial LU empty, singular and updates" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), format in (:csc, :csr)
        A = FaerSparseMatrix(spzeros(T, 0, 0); format)
        F = lu(A; algorithm=:simplicial)
        @test size(F.L) == (0, 0)
        @test size(F.U) == (0, 0)
        @test isempty(Vector(F.p))
        X = T[0 2 1; 3 1 0; 0 1 4]
        A = FaerSparseMatrix(sparse(X); format)
        F = lu(A; algorithm=:simplicial, ordering=[3, 1, 2])
        B = FaerSparseMatrix(sparse(2X); format)
        @test lu!(F, B) === F
        @test Matrix(F.L * F.U) ≈ (2X)[Vector(F.p), Vector(F.q)]
        S = FaerSparseMatrix(sparse(T[1 2; 2 4]); format)
        @test_throws SingularException lu(S; algorithm=:simplicial)
        G = lu(S; algorithm=:simplicial, check=false)
        @test !issuccess(G)
        @test_throws SingularException G.L
    end
    @test_throws ArgumentError lu(
        FaerSparseMatrix(sparse([1.0 0; 0 1])); algorithm=:simplicial, ordering=:amd
    )
end
