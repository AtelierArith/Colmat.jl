using Colmat, LinearAlgebra, Test
@testset "Retained QR inverse-action preconditioners" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        R in (Float32, Float64, ComplexF32, ComplexF64), pivoted in (false, true),
        n in (0, 1, 4)

        A = Matrix{T}(I, n, n) * T(4)
        if n > 1
            A[1, 1] = T(0.1)
            A[1, 2] = T(0.3)
            A[2, 2] = 100
            T <: Complex && (A[1, 2] += T(0.2im))
        end
        a = FaerMatrix(A)
        F = pivoted ? qr(a, ColumnNorm()) : qr(a)
        M = FaerLinearOperator(F)
        close(a)
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
    F = qr(FaerMatrix(ones(3, 2)))
    M = FaerLinearOperator(F)
    @test Matrix(M) ≈ pinv(ones(3, 2))
    close(M)
    close(F)
end
@testset "QR preconditioner iterative use and zero pivot" begin
    for pivoted in (false, true)
        A = FaerMatrix([4.0 1; 1 3])
        F = pivoted ? qr(A, ColumnNorm()) : qr(A)
        M = FaerLinearOperator(F)
        close(F)
        @test cg(A, ones(2); M).x ≈ Matrix(A) \ ones(2)
        close(M)
        close(A)
        F = if pivoted
            qr(FaerMatrix(zeros(2, 2)), ColumnNorm())
        else
            qr(FaerMatrix(zeros(2, 2)))
        end
        @test_throws Colmat.RustCall.RustError FaerLinearOperator(F)
        close(F)
    end
end
