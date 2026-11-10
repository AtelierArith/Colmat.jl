using Colmat, LinearAlgebra, Test
@testset "Retained LDLT inverse-action preconditioners" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        R in (Float32, Float64, ComplexF32, ComplexF64), uplo in (:L, :U),
        n in (0, 1, 4)

        A = zeros(T, n, n)
        for i in 1:n
            A[i, i] = T(isodd(i) ? 4 : -4)
        end
        if n > 1
            A[1, 2] = T(0.3)
            A[2, 1] = T(0.3)
            T <: Complex && (A[1, 2] += T(0.2im); A[2, 1] -= T(0.2im))
        end
        F = ldlt(Hermitian(FaerMatrix(A), uplo))
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
        close(G)
    end
end
@testset "LDLT preconditioned indefinite solve" begin
    A = FaerMatrix([4.0 1; 1 -3])
    F = ldlt(Hermitian(A))
    M = FaerLinearOperator(F)
    @test bicgstab(A, ones(2); M).x ≈ Matrix(A) \ ones(2)
    close(F)
    close(M)
    close(A)
end
@testset "LDLT invalid preconditioner factor" begin
    F = ldlt(FaerMatrix(zeros(2, 2)); check=false)
    @test_throws SingularException FaerLinearOperator(F)
    close(F)
    @test_throws ArgumentError FaerLinearOperator(F)
end
