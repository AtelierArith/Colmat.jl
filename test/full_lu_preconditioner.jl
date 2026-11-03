using Colmat, LinearAlgebra, Test
@testset "Full-pivot LU preconditioner" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        R in (Float32, Float64, ComplexF32, ComplexF64),
        n in (0, 1, 4)

        A=Matrix{T}(I, n, n)*T(4)
        if n>1
            A[1, 1]=0
            A[1, 2]=2
            A[2, 1]=3
            A[2, 2]=100
            T<:Complex && (A[1, 2]+=T(0.3im))
        end
        F=full_piv_lu(FaerMatrix(A))
        M=FaerLinearOperator(F)
        close(F)
        B=ones(R, n, 2)
        P=promote_type(T, R)
        for (op, ref) in ((M, A), (transpose(M), transpose(A)), (adjoint(M), adjoint(A)))
            X=op*B
            @test eltype(X)===P
            @test Matrix(X)≈P.(ref)\P.(B) atol=1e-3 rtol=1e-3
            close(X)
        end
        G=copy(M)
        close(M)
        @test Matrix(G*B)≈P.(A)\P.(B) atol=1e-3 rtol=1e-3
        close(G)
    end
    F=full_piv_lu(FaerMatrix([1.0 2; 2 4]); check = false)
    @test_throws SingularException FaerLinearOperator(F)
    close(F)
end
