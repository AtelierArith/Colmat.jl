using Colmat, LinearAlgebra, Test
@testset "Retained block LDLT inverse-action preconditioners" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        R in (Float32, Float64, ComplexF32, ComplexF64),
        uplo in (:L, :U),
        n in (0, 1, 2, 4)

        A=Matrix{T}(I, n, n)*T(4)
        if n>1
            A[1, 1]=0
            A[2, 2]=0
            A[1, 2]=1
            A[2, 1]=1
            T<:Complex && (A[1, 2]+=T(0.2im); A[2, 1]-=T(0.2im))
            n>2 && (A[1, 4] = T(3); A[4, 1] = T(3))
        end
        F=bunchkaufman(Hermitian(FaerMatrix(A), uplo))
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
end
@testset "Block LDLT preconditioned indefinite solve" begin
    A=FaerMatrix([0.0 1; 1 0])
    F=bunchkaufman(Hermitian(A))
    M=FaerLinearOperator(F)
    @test bicgstab(A, ones(2); M).x≈Matrix(A)\ones(2)
    close(F)
    close(M)
    close(A)
end
@testset "Block LDLT invalid factor" begin
    F=bunchkaufman(FaerMatrix(zeros(2, 2)); check = false)
    @test_throws SingularException FaerLinearOperator(F)
    close(F)
    @test_throws ArgumentError FaerLinearOperator(F)
end
