using Colmat, LinearAlgebra, SparseArrays, Test
@testset "Retained auto/supernodal QR inverse actions" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        R in (Float32, Float64, ComplexF32, ComplexF64),
        format in (:csc, :csr),
        (m, n) in ((0, 0), (3, 0), (3, 3), (5, 3)),
        algorithm in (:auto, :supernodal)

        A=zeros(T, m, n)
        for j = 1:n
            A[j, j]=T(3+j)
            j<m && (A[j+1, j]=T(0.2))
            T<:Complex && (A[j, j]+=T(0.3im))
        end
        F=qr(FaerSparseMatrix(sparse(A); format); algorithm)
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
