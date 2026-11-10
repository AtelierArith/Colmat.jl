using Test, Colmat, LinearAlgebra, SparseArrays
@testset "Simplicial sparse QR components" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        dims in ((0, 0), (3, 0), (3, 3), (5, 3), (8, 4)), format in (:csc, :csr),
        custom in (false, true)

        m, n = dims
        A = zeros(T, m, n)
        for j in 1:n
            A[j, j] = T(3 + j)
            for i in (j + 1):m
                isodd(i + j) && (A[i, j] = T(0.2i + j))
            end
        end
        T <: Complex && n > 0 && (A[m, 1] += T(0.3im))
        a = FaerSparseMatrix(sparse(A); format)
        order = custom ? reverse(collect(1:n)) : :identity
        plan = FaerSparseSymbolic(a; kind=:qr, algorithm=:simplicial, ordering=order)
        F = factorize(plan, a)
        close(plan)
        close(a)
        Q, R, p = F.Q, F.R, F.p
        @test Q isa FaerMatrix{T}
        @test R isa FaerSparseMatrix{T}
        @test size(Q) == (m, n)
        @test size(R) == (n, n)
        @test Vector(p) == (custom ? reverse(collect(1:n)) : collect(1:n))
        @test Matrix(Q) * Matrix(R) ≈ A[:, Vector(p)] atol = 1e-3
        @test Matrix(Q)' * Matrix(Q) ≈ Matrix{T}(I, n, n) atol = 1e-3
        @test Matrix(F) ≈ A atol = 1e-3
        B = reshape(T.(1:(m * 2)), m, 2)
        X = F \ B
        @test Matrix(X) ≈ A \ B atol = 1e-3 rtol = 1e-3
        G = deepcopy(F)
        close(F)
        @test_throws ArgumentError F.Q
        @test_throws ArgumentError reconstruct(F)
        @test Matrix(G) ≈ A atol = 1e-3
        @test Matrix(Q) * Matrix(R) ≈ A[:, Vector(p)] atol = 1e-3
        close(G)
        close(Q)
        close(R)
        close(p)
        close(X)
    end
end
@testset "QR controls and retained components" begin
    a = FaerSparseMatrix(sparse([1.0 0; 2 3; 0 4]))
    F = qr(a; algorithm=:simplicial)
    @test Matrix(F) ≈ Matrix(a)
    @test_throws ArgumentError qr(a; algorithm=:simplicial, ordering=:amd)
    @test (qr(a; algorithm=:supernodal) \ ones(3)) ≈ Matrix(a) \ ones(3)
    @test_throws ArgumentError qr(a; algorithm=:simplicial, ordering=[1, 1])
    Q, R = F.Q, F.R
    fill!(nonzeros(F.source), 0)
    @test Matrix(F) ≈ [1.0 0; 2 3; 0 4]
    @test Matrix(Q) * Matrix(R) ≈ [1.0 0; 2 3; 0 4]
    close(Q)
    close(R)
    close(F)
    close(a)
end

@testset "QR update and typed reconstruction" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        A = T[1 0; 2 3; 0 4]
        a = FaerSparseMatrix(sparse(A))
        F = qr(a; algorithm=:simplicial, ordering=[2, 1])
        oldQ, oldR = F.Q, F.R
        q = F.pcol
        @test Vector(q) == [2, 1]
        close(q)
        @test :pcol in propertynames(F)
        @test Matrix{ComplexF64}(F) ≈ A atol = 1e-3
        C = FaerMatrix{ComplexF64}(F)
        @test Matrix(C) ≈ A atol = 1e-3
        close(C)
        b = FaerSparseMatrix(sparse(2A))
        @test qr!(F, b) === F
        @test Matrix(F) ≈ 2A atol = 1e-3
        @test Matrix(oldQ) * Matrix(oldR) ≈ A[:, [2, 1]] atol = 1e-3
        @test Matrix(F \ ones(T, 3, 1)) ≈ (2A) \ ones(T, 3, 1) atol = 1e-3
        close(oldQ)
        close(oldR)
        close(F)
        close(a)
        close(b)
    end
end
