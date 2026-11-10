using Colmat, LinearAlgebra, SparseArrays, Test
@testset "Retained sparse LU preconditioners without source refactor" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        R in (Float32, Float64, ComplexF32, ComplexF64), format in (:csc, :csr),
        n in (0, 1, 3)

        A = n == 3 ? T[0 2 1; 3 1 0; 0 1 4] : Matrix{T}(I, n, n) * T(4)
        T <: Complex && n > 0 && (A[n, n] += T(0.3im))
        a = FaerSparseMatrix(sparse(A); format)
        F = lu(a; algorithm=:simplicial, ordering=reverse(collect(1:n)))
        fill!(nonzeros(F.source), 0)
        M = FaerLinearOperator(F)
        close(F)
        close(a)
        B = ones(R, n, 2)
        P = promote_type(T, R)
        for (op, ref) in (
            (M, A),
            (conj(M), conj(A)),
            (transpose(M), transpose(A)),
            (adjoint(M), adjoint(A)),
        )
            X = op * B
            @test eltype(X) === P
            @test Matrix(X) ≈ P.(ref) \ P.(B) atol = 1e-3 rtol = 1e-3
            close(X)
        end
        G = copy(M)
        close(M)
        @test Matrix(G * B) ≈ P.(A) \ P.(B) atol = 1e-3 rtol = 1e-3
        close(G)
    end
end
@testset "Sparse LU preconditioning and sparse scaling" begin
    A = FaerSparseMatrix(sparse([0.0 2 1; 3 1 0; 0 1 4]))
    F = lu(A; algorithm=:simplicial, ordering=[3, 1, 2])
    M = FaerLinearOperator(F)
    @test bicgstab(A, ones(3); M).x ≈ Matrix(A) \ ones(3)
    close(M)
    close(F)
    close(A)
    n = 20_000
    A = FaerSparseMatrix(spdiagm(0 => fill(2.0, n)); format=:csr)
    F = lu(A; algorithm=:simplicial)
    M = FaerLinearOperator(F)
    close(F)
    close(A)
    @test size(M) == (n, n)
    @test M * ones(ComplexF64, n) ≈ fill(0.5 + 0im, n)
    close(M)
end
