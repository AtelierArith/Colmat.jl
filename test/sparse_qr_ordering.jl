using Colmat, LinearAlgebra, SparseArrays, Test
@testset "Native COLAMD controls and retained QR permutations" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        algorithm in (:auto, :supernodal), format in (:csc, :csr), n in (0, 1, 9),
        aggressive in (true, false)

        A = zeros(T, n + 3, n)
        for j in 1:n
            A[j, j] = T(4 + j)
            A[1, j] += T(0.2)
            A[n + 3, j] = T(0.1j)
            T <: Complex && (A[j, j] += T(0.2im))
        end
        a = FaerSparseMatrix(sparse(A); format)
        p = Vector(colamd(a; dense_row=-1.0, dense_col=0.0, aggressive))
        F = qr(a; algorithm, ordering=:colamd, dense_row=-1.0, dense_col=0.0, aggressive)
        @test Vector(F.pcol) == p
        B = ones(T, n + 3, 2)
        @test Matrix(F \ B) ≈ A \ B atol = 1e-3 rtol = 1e-3
        G = copy(F)
        close(F)
        @test Vector(G.p) == p
        b = FaerSparseMatrix(sparse(2A); format)
        @test qr!(G, b) === G
        @test Vector(G.pcol) == p
        @test Matrix(G \ B) ≈ (2A) \ B atol = 1e-3 rtol = 1e-3
        close(a)
        close(b)
        close(G)
    end
end
