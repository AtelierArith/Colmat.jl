using Colmat, LinearAlgebra, SparseArrays, Test
@testset "Sparse QR factor construction workspace" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        algorithm in (:auto, :supernodal), format in (:csc, :csr),
        (m, n) in ((0, 0), (3, 0), (3, 3), (5, 3))

        A = zeros(T, m, n)
        for j in 1:n
            A[j, j] = T(4 + j)
            j < m && (A[j + 1, j] = T(0.3))
            T <: Complex && (A[j, j] += T(0.2im))
        end
        a = FaerSparseMatrix(sparse(A); format)
        p = FaerSparseSymbolic(a; kind=:qr, algorithm)
        req = scratch_requirements(factorize, p, a)
        s = FaerScratch(factorize, p, a)
        B = ones(T, m, 2)
        for _ in 1:2
            F = factorize(p, a; scratch=s)
            @test Matrix(F \ B) ≈ A \ B atol = 1e-3 rtol = 1e-3
            @test scratch_capacity(s).bytes >= req.bytes
            close(F)
        end
        @test reserve!(s, factorize, p, a) === s
        close(s)
        @test_throws ArgumentError factorize(p, a; scratch=s)
        close(p)
        close(a)
    end
end

@testset "Factor workspace validation and independent ownership" begin
    a = FaerSparseMatrix(sparse([2.0 0; 0 3; 1 0]))
    p = FaerSparseSymbolic(a; kind=:qr, algorithm=:supernodal)
    s = FaerScratch(0; alignment=1)
    @test_throws ArgumentError factorize(p, a; scratch=s)
    @test reserve!(s, factorize, p, a) === s
    F = factorize(p, a; scratch=s)
    b = FaerSparseMatrix(sparse([4.0 0; 0 6; 2 0]))
    G = factorize(p, b; scratch=s)
    @test (F \ ones(3)) ≈ Matrix(a) \ ones(3)
    @test (G \ ones(3)) ≈ Matrix(b) \ ones(3)
    other = FaerSparseMatrix(sparse([2.0 1; 0 3; 1 0]))
    @test_throws ArgumentError factorize(p, other; scratch=s)
    wrong = FaerSparseMatrix(spzeros(4, 2))
    @test_throws DimensionMismatch scratch_requirements(factorize, p, wrong)
    close(s)
    close(p)
    close(a)
    close(b)
    @test F \ ones(3) ≈ [0.6, 1 / 3]
    @test G \ ones(3) ≈ [0.3, 1 / 6]
    @test_throws ArgumentError scratch_requirements(factorize, p, other)
    close(F)
    close(G)
    close(other)
    close(wrong)
end
