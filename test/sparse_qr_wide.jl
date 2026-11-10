using Colmat, LinearAlgebra, SparseArrays, Test
@testset "Native wide sparse QR factors and minimum norm solves" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        format in (:csc, :csr),
        algorithm in (:auto, :simplicial, :supernodal),
        (m, n) in ((0, 3), (1, 3), (3, 5), (5, 12)),
        empty_head in (false, true),
        reversed in (false, true)

        A = zeros(T, m, n)
        for j in 1:n, i in 1:m
            empty_head && j <= m && continue
            A[i, j] = T(mod(j - 1, max(m, 1)) + 1 == i ? 3 + j : 0.1)
            T <: Complex && (A[i, j] += T(0.05im * (i + j)))
        end
        order = reversed ? reverse(collect(1:n)) : :default
        native = FaerSparseMatrix(sparse(A); format)
        F = qr(native; algorithm, ordering=order)
        Q, R = Matrix(F.Q), Matrix(F.R)
        @test size(Q) == (m, m)
        @test size(R) == (m, n)
        @test Q * R ≈ A[:, Vector(F.pcol)] atol = 1e-3 rtol = 1e-3
        @test adjoint(Q) * Q ≈ Matrix{T}(I, m, m) atol = 1e-3 rtol = 1e-3
        @test Matrix(F) ≈ A atol = 1e-3 rtol = 1e-3
        reversed && (@test Vector(F.pcol) == order)
        close(native)
        close(F.source)
        for (f, ref) in ((F, A), (transpose(F), transpose(A)), (adjoint(F), adjoint(A)))
            for S in (T, ComplexF64)
                B = ones(S, size(ref, 1), 2)
                @test Matrix(f \ B) ≈ pinv(promote_type(T, S).(ref)) * B atol = 1e-3 rtol =
                    1e-3
            end
        end
        M = FaerLinearOperator(F)
        @test Matrix(M) ≈ pinv(A) atol = 1e-3 rtol = 1e-3
        @test rank(F) == rank(A)
        P = pinv(F)
        @test Matrix(P) ≈ pinv(A) atol = 1e-3 rtol = 1e-3
        G = copy(F)
        close(F)
        @test Matrix(G) ≈ A atol = 1e-3 rtol = 1e-3
        replacement = FaerSparseMatrix(sparse(2A); format)
        saved_columns = Vector(G.pcol)
        @test qr!(G, replacement) === G
        @test Vector(G.pcol) == saved_columns
        @test Matrix(G) ≈ 2A atol = 1e-3 rtol = 1e-3
        close(replacement)
        @test Matrix(M) ≈ pinv(A) atol = 1e-3 rtol = 1e-3
        close(G)
        close(M)
        close(P)
    end
end
