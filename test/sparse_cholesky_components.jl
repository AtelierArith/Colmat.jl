using Colmat, Test, LinearAlgebra, SparseArrays, Random

@testset "Native sparse Cholesky components and reconstruction" begin
    rng=MersenneTwister(553)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        format in (:csc, :csr),
        algorithm in (:simplicial, :supernodal),
        uplo in (:L, :U),
        mode in (:amd, :identity, :custom)

        raw=randn(rng, T, 9, 9)
        a=raw*raw'+9I
        # Exercise both triangles in the owner; the wrapper selects one.
        source=FaerSparseMatrix(sparse(a); format)
        ordering=mode===:custom ? randperm(rng, 9) : mode
        f=cholesky(Hermitian(source, uplo); ordering, algorithm)
        l, u, p=f.L, f.U, f.p
        @test l isa FaerSparseMatrix{T}
        @test u isa FaerSparseMatrix{T}
        @test l.format===:csc && u.format===:csc
        @test istril(Matrix(l)) && istriu(Matrix(u))
        @test Matrix(u) ≈ Matrix(l)'
        @test Matrix(l*u) ≈ a[Vector(p), Vector(p)]
        @test Matrix(l) ≈ Matrix(cholesky(Hermitian(a[Vector(p), Vector(p)])).L)
        @test Matrix(f) ≈ a
        @test Matrix(FaerMatrix(f)) ≈ a
        @test sparse(f) ≈ a
        @test SparseMatrixCSC(f) ≈ a
        @test FaerSparseMatrix(f) isa FaerSparseMatrix{T}
        @test Matrix(FaerSparseMatrix(f)) ≈ a
        @test Matrix{ComplexF64}(f) ≈ ComplexF64.(a) rtol=50eps(real(T))
        @test Matrix(FaerSparseMatrix{ComplexF64}(f)) ≈ ComplexF64.(a) rtol=50eps(real(T))
        @test Matrix(FaerMatrix{ComplexF64}(f)) ≈ ComplexF64.(a) rtol=50eps(real(T))
        @test logdet(f) ≈ logdet(cholesky(Hermitian(a)))
        @test det(f) ≈ det(cholesky(Hermitian(a)))
        @test logabsdet(f)[1] ≈ logabsdet(cholesky(Hermitian(a)))[1]
        @test logabsdet(f)[2]==1
        @test logdet(f) isa real(T)
        @test det(f) isa real(T)
        @test hasproperty(f, :L) && hasproperty(f, :U)
        expectedl=Matrix(l)
        expectedu=Matrix(u)
        l[1, 1]*=2
        @test Matrix(f.L) ≈ expectedl
        @test Matrix(f) ≈ a
        cholesky!(f, 2source)
        @test Matrix(f) ≈ 2a
        @test Matrix(u)==expectedu
        close(f)
        close(source)
        @test Matrix(u)==expectedu
        @test_throws ArgumentError f.L
        @test_throws ArgumentError f.U
        @test_throws ArgumentError Matrix(f)
        @test_throws ArgumentError logdet(f)
    end
end
@testset "Sparse factors with several supernodes and fill" begin
    rng=MersenneTwister(54)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        n in (0, 1, 3, 31, 103),
        algorithm in (:simplicial, :supernodal)

        raw=sprand(rng, T, n, n, 0.03)
        a=raw*raw'+spdiagm(0=>fill(T(3), n))
        s=FaerSparseMatrix(a)
        f=cholesky(s; algorithm)
        @test sparse(f.L*f.U) ≈ a[Vector(f.p), Vector(f.p)]
        @test sparse(f) ≈ a
        @test nnz(f.L)<=f.factor_storage
        @test logdet(f) ≈ logdet(cholesky(Hermitian(Matrix(a))))
        @test det(f) ≈ det(cholesky(Hermitian(Matrix(a))))
    end
end
@testset "Selected triangle, failed factors and stable log determinants" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        algorithm in (:simplicial, :supernodal)

        a=T[5 NaN NaN; 1 4 NaN; 0 1 3]
        f=cholesky(Hermitian(FaerSparseMatrix(sparse(a)), :L); algorithm, perm = [3, 1, 2])
        @test Matrix(f) ≈ Matrix(Hermitian(a, :L))
        bad=cholesky(
            FaerSparseMatrix(spdiagm(0=>T[1, -1]));
            algorithm,
            ordering = :identity,
            check = false,
        )
        @test_throws PosDefException bad.L
        @test_throws PosDefException bad.U
        @test_throws PosDefException Matrix(bad)
        @test_throws PosDefException logdet(bad)
    end
    for d in ([1e300, 1e-300, 2.0], [1e300, 1e300], [1e-300, 1e-300])
        f=cholesky(FaerSparseMatrix(spdiagm(0=>d)); ordering = :identity)
        @test logdet(f) ≈ sum(log, d)
        @test logabsdet(f)[2]==1.0
    end
    @test isempty(
        detect_ambiguities(Colmat, Base, LinearAlgebra, SparseArrays; recursive = true),
    )
end
@testset "Sparse Cholesky component extraction scales with stored factors" begin
    n=50_000
    f=cholesky(FaerSparseMatrix(spdiagm(0=>fill(4.0, n))); algorithm = :supernodal)
    l, u=f.L, f.U
    @test nnz(l)==n && nnz(u)==n
    @test l[1, 1]==2 && l[n, n]==2
    @test isapprox(logdet(f), n*log(4.0))
    @test nnz(FaerSparseMatrix(f))==n
end
