using Colmat, LinearAlgebra, Test
@testset "Retained Cholesky inverse actions" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        R in (Float32, Float64, ComplexF32, ComplexF64), uplo in (:L, :U),
        n in (0, 1, 4)

        A = Matrix{T}(I, n, n) * T(4)
        n > 1 && (A[1, 2]=T(0.3); A[2, 1]=T(0.3))
        T <: Complex && n > 1 && (A[1, 2] += T(0.2im); A[2, 1] -= T(0.2im))
        F = cholesky(Hermitian(FaerMatrix(A), uplo))
        M = FaerLinearOperator(F)
        close(F)
        B = ones(R, n, 2)
        P = promote_type(T, R)
        for (op, ref) in ((M, A), (transpose(M), transpose(A)), (adjoint(M), adjoint(A)))
            X = op * B
            @test eltype(X) === P
            @test Matrix(X) ≈ P.(ref) \ P.(B) atol = 1e-3 rtol = 1e-3
            close(X)
        end
        G = copy(M)
        close(M)
        @test Matrix(G * B) ≈ P.(A) \ P.(B) atol = 1e-3 rtol = 1e-3
        n > 0 && @test cg(FaerMatrix(A), ones(T, n); M=G).x ≈ A \ ones(T, n) atol = 1e-3
        close(G)
    end
    F = cholesky(Hermitian(FaerMatrix([-1.0 0; 0 1])); check=false)
    @test_throws PosDefException FaerLinearOperator(F)
    close(F)
end
