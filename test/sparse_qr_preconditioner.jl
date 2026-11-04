using Colmat, LinearAlgebra, SparseArrays, Test
@testset "Retained sparse QR inverse actions" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        R in (Float32, Float64, ComplexF32, ComplexF64),
        format in (:csc, :csr),
        (m, n) in ((0, 0), (3, 0), (3, 3), (5, 3))

        A=zeros(T, m, n)
        for j = 1:n
            A[j, j]=T(3+j)
            j<m && (A[j+1, j]=T(0.2))
            T<:Complex && (A[j, j]+=T(0.3im))
        end
        F=qr(
            FaerSparseMatrix(sparse(A); format);
            algorithm = :simplicial,
            ordering = reverse(collect(1:n)),
        )
        fill!(nonzeros(F.source), 0)
        M=FaerLinearOperator(F)
        close(F)
        expected=pinv(A)
        @test size(M)==(n, m)
        P=promote_type(T, R)
        for (op, ref) in (
            (M, expected),
            (conj(M), conj(expected)),
            (transpose(M), transpose(expected)),
            (adjoint(M), adjoint(expected)),
        )
            B=ones(R, size(ref, 2), 2)
            X=op*B
            @test eltype(X)===P
            @test Matrix(X)≈P.(ref)*P.(B) atol=1e-3 rtol=1e-3
            close(X)
        end
        G=copy(M)
        close(M)
        @test Matrix(G)≈expected atol=1e-3 rtol=1e-3
        close(G)
    end
end
@testset "Sparse QR action scaling and validation" begin
    n=20_000
    A=FaerSparseMatrix(spdiagm(0=>fill(2.0, n)); format = :csr)
    F=qr(A; algorithm = :simplicial)
    M=FaerLinearOperator(F)
    close(F)
    close(A)
    @test size(M)==(n, n)
    @test M*ones(ComplexF64, n)≈fill(0.5+0im, n)
    @test adjoint(M)*ones(ComplexF64, n)≈fill(0.5+0im, n)
    close(M)
    A=FaerSparseMatrix(sparse([4.0 1; 1 3]))
    F=qr(A; algorithm = :simplicial)
    M=FaerLinearOperator(F)
    @test cg(A, ones(2); M).x≈Matrix(A)\ones(2)
    close(F)
    @test_throws ArgumentError FaerLinearOperator(F)
    close(M)
    F=qr(A)
    @test Matrix(FaerLinearOperator(F))≈inv(Matrix(A))
    close(F)
    close(A)
    A=FaerSparseMatrix(sparse([1, 2], [1, 2], [1.0, 0.0], 2, 2))
    F=qr(A; algorithm = :simplicial)
    N=FaerLinearOperator(F)
    @test Matrix(N)≈pinv(Matrix(A))
    close(N)
    close(F)
    close(A)
end
