using Colmat, LinearAlgebra, SparseArrays, Test
@testset "Saved sparse QR full orthogonal basis" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        format in (:csc, :csr),
        algorithm in (:auto, :simplicial, :supernodal),
        (m, n) in ((0, 0), (5, 0), (1, 1), (5, 3), (6, 6), (40, 20))

        A=zeros(T, m, n)
        offset=m-n
        for j = 1:n
            A[offset+j, j]=T(3+j)
            j<n && (A[offset+j+1, j]=T(0.2))
            T<:Complex && (A[offset+j, j]+=T(0.3im))
        end
        F=qr(FaerSparseMatrix(sparse(A); format); algorithm)
        Q=Matrix(F.Qfull)
        thin=Matrix(F.Qthin)
        @test size(Q)==(m, m)
        @test thin≈Matrix(F.Q)
        @test Q[:, 1:n]≈thin atol=1e-3 rtol=1e-3
        @test Q'*Q≈Matrix{T}(I, m, m) atol=1e-3 rtol=1e-3
        @test Q*Q'≈Matrix{T}(I, m, m) atol=1e-3 rtol=1e-3
        padded=vcat(Matrix(F.R), zeros(T, m-n, n))
        @test Q*padded≈A[:, Vector(F.pcol)] atol=1e-3 rtol=1e-3
        G=copy(F)
        close(F.source)
        @test Matrix(F.Qfull)≈Q
        close(F)
        @test Matrix(G.Qfull)≈Q
        close(G)
    end
end

@testset "Rank-deficient QR full bases" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        algorithm in (:auto, :simplicial, :supernodal),
        A in (zeros(T, 5, 3), T[1 2 0; 0 0 0; 0 0 0; 0 0 0; 0 0 0])

        F=qr(FaerSparseMatrix(sparse(A)); algorithm)
        Q=Matrix(F.Qfull)
        @test adjoint(Q)*Q≈Matrix{T}(I, 5, 5) atol=1e-3 rtol=1e-3
        @test Q[:, 1:3]≈Matrix(F.Qthin) atol=1e-3 rtol=1e-3
        @test Q[:, 1:3]*Matrix(F.R)≈A[:, Vector(F.pcol)] atol=1e-3 rtol=1e-3
        close(F)
    end
end
