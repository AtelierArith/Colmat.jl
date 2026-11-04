using Colmat, LinearAlgebra, SparseArrays, Test
@testset "Sparse factor copies survive closed source snapshots" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        kind in (:lu, :qr, :cholesky),
        algorithm in (:auto, :simplicial, :supernodal)

        A=T[4 1; 1 3]
        a=FaerSparseMatrix(sparse(A))
        F=kind===:lu ? lu(a; algorithm) :
          kind===:qr ? qr(a; algorithm) : cholesky(Hermitian(a); algorithm)
        close(F.source)
        G=copy(F)
        D=deepcopy(F)
        @test typeof(G)===typeof(F)
        @test typeof(D)===typeof(F)
        @test !isopen(G.source)
        close(F)
        @test G\ones(T, 2)≈A\ones(T, 2)
        @test isapprox(
            D\ones(ComplexF64, 2),
            A\ones(ComplexF64, 2);
            atol = 10eps(typeof(real(zero(T)))),
            rtol = 10eps(typeof(real(zero(T)))),
        )
        close(G)
        close(D)
        close(a)
    end
end
