using Colmat, LinearAlgebra, Test
@testset "Retained LU inverse-action preconditioner" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), n in (0, 1, 4)
        A = Matrix{T}(I, n, n) * T(4)
        n > 1 && (A[1, 2]=T(0.3); A[2, 1]=T(0.3))
        T <: Complex && n > 1 && (A[1, 2] += T(0.2im); A[2, 1] -= T(0.2im))
        a = FaerMatrix(A)
        F = lu(a)
        M = FaerLinearOperator(F)
        close(a)
        close(F)
        B = ones(T, n, 2)
        @test Matrix(M * B) ≈ A \ B atol = 1e-3
        @test Matrix(transpose(M) * B) ≈ transpose(A) \ B atol = 1e-3
        @test Matrix(adjoint(M) * B) ≈ adjoint(A) \ B atol = 1e-3
        @test_throws ArgumentError M.coefficients
        G = copy(M)
        close(M)
        @test Matrix(G * B) ≈ A \ B atol = 1e-3
        n > 0 && @test cg(FaerMatrix(A), ones(T, n); M=G).x ≈ A \ ones(T, n) atol = 1e-3
        close(G)
    end
end
@testset "LU preconditioner scalar promotion and rejection" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        R in (Float32, Float64, ComplexF32, ComplexF64)

        A = T[4 1; 2 3]
        T <: Complex && (A[1, 2] += T(0.2im))
        F = lu(FaerMatrix(A))
        M = FaerLinearOperator(F)
        close(F)
        B = R[1 2; 3 4]
        P = promote_type(T, R)
        for (op, ref) in ((M, A), (transpose(M), transpose(A)), (adjoint(M), adjoint(A)))
            X = op * B
            @test eltype(X) === P
            @test Matrix(X) ≈ P.(ref) \ P.(B) atol = 1e-3 rtol = 1e-3
            close(X)
        end
        close(M)
    end
    F = lu(FaerMatrix([1.0 2; 2 4]); check=false)
    @test_throws SingularException FaerLinearOperator(F)
    close(F)
    @test_throws ArgumentError FaerLinearOperator(F)
    F = lu(FaerMatrix(ones(3, 2)); check=false)
    @test_throws DimensionMismatch FaerLinearOperator(F)
    close(F)
end
