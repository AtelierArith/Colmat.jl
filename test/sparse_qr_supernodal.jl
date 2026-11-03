using Colmat, LinearAlgebra, SparseArrays, Test

@testset "Explicit supernodal sparse QR" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        format in (:csc, :csr),
        (m, n) in ((0, 0), (3, 0), (4, 4), (7, 4))

        A=zeros(T, m, n)
        for j = 1:n
            A[j, j]=T(3+j)
            j<m && (A[j+1, j]=T(0.4))
            T<:Complex && (A[j, j]+=T(0.2im))
        end
        a=FaerSparseMatrix(sparse(A); format)
        plan=FaerSparseSymbolic(a; kind = :qr, algorithm = :supernodal)
        F=factorize(plan, a)
        close(plan)
        close(a)
        B=reshape(T.(1:2m), m, 2)
        @test Matrix(F\B)≈A\B atol=1e-3 rtol=1e-3
        G=copy(F)
        close(F)
        @test Matrix(G\B)≈A\B atol=1e-3 rtol=1e-3
        b=FaerSparseMatrix(sparse(2A); format)
        @test qr!(G, b)===G
        @test Matrix(G\B)≈(2A)\B atol=1e-3 rtol=1e-3
        close(b)
        close(G)
    end
end
