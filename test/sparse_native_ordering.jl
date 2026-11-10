using Colmat, LinearAlgebra, SparseArrays, Test
@testset "Explicit native sparse LU/QR column ordering" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        format in (:csc, :csr),
        kind in (:lu, :qr),
        algorithm in (:auto, :supernodal),
        n in (0, 1, 3),
        order in (:identity, reverse(collect(1:n)), circshift(collect(1:n), 1))

        m = kind === :qr && n > 1 ? n + 2 : n
        A = zeros(T, m, n)
        for j in 1:n
            A[j, j] = T(4 + j)
            j < m && (A[j + 1, j] = T(0.3))
            T <: Complex && (A[j, j] += T(0.2im))
        end
        a = FaerSparseMatrix(sparse(A); format)
        plan = FaerSparseSymbolic(a; kind, algorithm, ordering=order)
        F = factorize(plan, a)
        expected = order === :identity ? collect(1:n) : order
        q = kind === :lu ? Vector(F.q) : Vector(F.pcol)
        @test q == expected
        @test Matrix(F) ≈ A atol = 1e-3 rtol = 1e-3
        if kind === :lu
            @test Matrix(F.L * F.U) ≈ A[Vector(F.p), q] atol = 1e-3 rtol = 1e-3
        else
            @test Matrix(F.Q) * Matrix(F.R) ≈ A[:, q] atol = 1e-3 rtol = 1e-3
        end
        G = copy(F)
        close(plan)
        close(a)
        close(F.source)
        B = ones(ComplexF64, m, 2)
        @test Matrix(F \ B) ≈ pinv(ComplexF64.(A)) * B atol = 1e-3 rtol = 1e-3
        for (f, ref) in ((transpose(F), transpose(A)), (adjoint(F), adjoint(A)))
            b = ones(ComplexF64, n, 2)
            @test Matrix(f \ b) ≈ pinv(ComplexF64.(ref)) * b atol = 1e-3 rtol = 1e-3
        end
        close(F)
        @test Matrix(G) ≈ A atol = 1e-3 rtol = 1e-3
        close(G)
    end
end
