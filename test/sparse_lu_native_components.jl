using Test, Colmat, LinearAlgebra, SparseArrays
@testset "Saved auto/supernodal sparse LU components" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        format in (:csc, :csr),
        algorithm in (:auto, :supernodal)

        A = T[0 2 1; 3 1 0; 0 1 4]
        T <: Complex && (A[1, 2] += T(0.5im); A[3, 3] += T(0.25im))
        native = FaerSparseMatrix(sparse(A); format)
        F = lu(native; algorithm)
        L, U, p, q = F.L, F.U, Vector(F.p), Vector(F.q)
        @test L isa FaerSparseMatrix{T}
        @test U isa FaerSparseMatrix{T}
        @test Matrix(L * U) ≈ A[p, q]
        @test Matrix(F) ≈ A
        @test Matrix(reconstruct(F)) ≈ A
        @test istril(Matrix(L))
        @test istriu(Matrix(U))
        @test diag(Matrix(L)) == ones(T, 3)
        b = ones(T, 3, 2)
        @test A * Matrix(F \ b) ≈ b
        @test transpose(A) * Matrix(transpose(F) \ b) ≈ b
        @test A' * Matrix(adjoint(F) \ b) ≈ b
        plan = FaerSparseSymbolic(native; kind=:lu, algorithm)
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

@testset "Native sparse LU empty and updates" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        format in (:csc, :csr),
        algorithm in (:auto, :supernodal)

        E = FaerSparseMatrix(spzeros(T, 0, 0); format)
        F = lu(E; algorithm)
        @test size(F.L) == (0, 0)
        @test size(F.U) == (0, 0)
        @test size(Matrix(F)) == (0, 0)
        @test isempty(Vector(F.p))
        @test isempty(Vector(F.q))
        A = T[0 2 1; 3 1 0; 0 1 4]
        native = FaerSparseMatrix(sparse(A); format)
        F = lu(native; algorithm)
        L, U = F.L, F.U
        p, q = Vector(F.p), Vector(F.q)
        replacement = FaerSparseMatrix(sparse(2A); format)
        @test lu!(F, replacement) === F
        @test Matrix(F) ≈ 2A
        @test Matrix(L * U) ≈ A[p, q]
        close(native)
        close(replacement)
        @test Matrix(F) ≈ 2A
        close(F)
        @test Matrix(L * U) ≈ A[p, q]
    end
end

@testset "Native sparse LU multiple supernodes and COLAMD controls" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        format in (:csc, :csr),
        algorithm in (:auto, :supernodal)

        n = 40
        A = T[
            if i == j
                0.1
            else
                (
                    if (i + j) % 3 == 0 || abs(i - j) == 1
                        mod(17(i - 1) + 13(j - 1), 19) - 9
                    else
                        0
                    end
                )
            end for i in 1:n, j in 1:n
        ]
        if T <: Complex
            for j in 1:n, i in 1:n
                i != j && !iszero(A[i, j]) && (A[i, j] += T(0.1im * mod(i + j, 5)))
            end
        end
        native = FaerSparseMatrix(sparse(A); format)
        F = lu(
            native;
            algorithm,
            ordering=:colamd,
            dense_row=0.8,
            dense_col=0.7,
            aggressive=false,
        )
        tolerance = T in (Float32, ComplexF32) ? 1e-3 : 1e-10
        @test isapprox(
            Matrix(F.L * F.U), A[Vector(F.p), Vector(F.q)]; atol=tolerance, rtol=tolerance
        )
        @test isapprox(Matrix(F), A; atol=tolerance, rtol=tolerance)
        b = ones(T, n, 2)
        @test isapprox(A * Matrix(F \ b), b; atol=tolerance, rtol=tolerance)
        @test isapprox(
            transpose(A) * Matrix(transpose(F) \ b), b; atol=tolerance, rtol=tolerance
        )
        @test isapprox(A' * Matrix(adjoint(F) \ b), b; atol=tolerance, rtol=tolerance)
        close(native)
        @test isapprox(Matrix(F), A; atol=tolerance, rtol=tolerance)
    end
end
