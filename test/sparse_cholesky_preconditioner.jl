using Colmat, LinearAlgebra, SparseArrays, Test
@testset "Retained sparse Cholesky preconditioners" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        R in (Float32, Float64, ComplexF32, ComplexF64),
        format in (:csc, :csr),
        algorithm in (:auto, :simplicial, :supernodal),
        uplo in (:L, :U),
        n in (0, 1, 4)

        A=Matrix{T}(I, n, n)*T(5)
        if n>1
            A[1, 2]=T(0.3)
            A[2, 1]=T(0.3)
            T<:Complex && (A[1, 2]+=T(0.2im); A[2, 1]-=T(0.2im))
        end
        a=FaerSparseMatrix(sparse(A); format)
        F=cholesky(Hermitian(a, uplo); algorithm, ordering = reverse(collect(1:n)))
        fill!(nonzeros(F.source), 0)
        M=FaerLinearOperator(F)
        close(F)
        close(a)
        B=ones(R, n, 2)
        P=promote_type(T, R)
        for (op, ref) in (
            (M, A),
            (conj(M), conj(A)),
            (transpose(M), transpose(A)),
            (adjoint(M), adjoint(A)),
        )
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
@testset "Sparse Cholesky preconditioned CG" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        A=FaerSparseMatrix(sparse(T[4 1; 1 3]))
        F=cholesky(Hermitian(A); algorithm = :supernodal)
        M=FaerLinearOperator(F)
        close(F)
        @test cg(A, ones(T, 2); M).x≈Matrix(A)\ones(T, 2) atol=1e-3
        close(A)
        close(M)
    end
end
@testset "Sparse Cholesky scale and invalid owners" begin
    n=20_000
    A=FaerSparseMatrix(spdiagm(0=>fill(2.0, n)); format = :csr)
    F=cholesky(Hermitian(A); algorithm = :supernodal)
    M=FaerLinearOperator(F)
    close(F)
    close(A)
    @test size(M)==(n, n)
    @test M*ones(ComplexF64, n)≈fill(0.5+0im, n)
    close(M)
    F=cholesky(Hermitian(FaerSparseMatrix(spdiagm(0=>[-1.0, 1.0]))); check = false)
    @test_throws PosDefException FaerLinearOperator(F)
    close(F)
    @test_throws ArgumentError FaerLinearOperator(F)
end
