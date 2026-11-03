using Colmat, Test, LinearAlgebra, SparseArrays, Random

@testset "Sparse Cholesky direct ordering and algorithm selection" begin
    rng=MersenneTwister(243)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        format in (:csc, :csr),
        uplo in (:L, :U),
        algorithm in (:auto, :simplicial, :supernodal),
        mode in (:amd, :identity, :custom)

        raw=randn(rng, T, 7, 7)
        a=raw*raw'+7I
        source=FaerSparseMatrix(sparse(a); format)
        custom=randperm(rng, 7)
        order=mode===:custom ? FaerPermutation(custom) : mode
        plan=FaerSparseSymbolic(source; kind = :cholesky, uplo, ordering = order, algorithm)
        @test plan.algorithm in (:simplicial, :supernodal)
        @test algorithm===:auto || plan.algorithm===algorithm
        @test plan.factor_storage>=7
        p=plan.p
        @test p isa FaerPermutation
        @test isperm(Vector(p))
        mode===:custom && @test Vector(p)==custom
        mode===:identity && @test Vector(p)==1:7
        copied=copy(plan)
        @test Vector(copied.p)==Vector(p)
        @test copied.algorithm===plan.algorithm
        @test copied.factor_storage==plan.factor_storage
        f=factorize(plan, source)
        @test f.algorithm===plan.algorithm
        @test f.factor_storage==plan.factor_storage
        @test Vector(f.p)==Vector(p)
        rhs=randn(rng, T, 7, 2)
        @test Matrix(f\rhs) ≈ a\rhs
        @test Matrix(f'\rhs) ≈ a'\rhs
        @test Matrix(transpose(f)\rhs) ≈ transpose(a)\rhs
        @test Vector(FaerPermutation(f))==Vector(p)
        direct=cholesky(Hermitian(source, uplo); ordering = order, algorithm)
        @test Vector(direct.p)==Vector(p)
        @test Matrix(direct\rhs) ≈ a\rhs
        update=2source
        @test cholesky!(f, update)===f
        @test Vector(f.p)==Vector(p)
        @test f.algorithm===plan.algorithm
        @test Matrix(f\rhs) ≈ (2a)\rhs
        mixed=randn(rng, ComplexF64, 7, 2)
        @test Matrix(f\mixed) ≈ (2ComplexF64.(a))\mixed
        close(plan)
        close(source)
        mode===:custom && close(order)
        @test Matrix(f\rhs) ≈ (2a)\rhs
        clone=copy(f)
        @test Vector(clone.p)==Vector(p)
        @test clone.algorithm===f.algorithm
        @test clone.factor_storage==f.factor_storage
        close(f)
        @test_throws ArgumentError f.p
        @test_throws ArgumentError f.algorithm
        @test_throws ArgumentError f.factor_storage
        @test_throws ArgumentError plan.p
        @test_throws ArgumentError plan.algorithm
        @test_throws ArgumentError plan.factor_storage
        @test Matrix(clone\rhs) ≈ (2a)\rhs
    end
end
@testset "Cholesky order is applied inside symbolic analysis" begin
    n=80
    a=spdiagm(0=>fill(2.0, n))
    a[1, :].=0.1
    a[:, 1].=0.1
    a[1, 1]=n
    source=FaerSparseMatrix(a)
    identity=cholesky(source; ordering = :identity, algorithm = :simplicial)
    reduced=cholesky(source; ordering = :amd, algorithm = :simplicial)
    custom=cholesky(source; ordering = Vector(reduced.p), algorithm = :simplicial)
    @test identity.factor_storage>reduced.factor_storage
    @test reduced.factor_storage==custom.factor_storage
    @test Vector(identity.p)==1:n
    @test Vector(custom.p)==Vector(reduced.p)
    for f in (identity, reduced, custom)
        @test norm(a*(f\ones(n))-ones(n))<1e-12
    end
    # Ignore stored entries in the other half, including nonfinite values.
    raw=[5.0 NaN NaN; 1 4 NaN; 0 1 3]
    s=FaerSparseMatrix(sparse(raw); format = :csr)
    f=cholesky(Hermitian(s, :L); ordering = [3, 1, 2], algorithm = :supernodal)
    @test f\ones(3) ≈ Hermitian(raw, :L)\ones(3)
    @test Vector(f.p)==[3, 1, 2]
end
@testset "Cholesky ordering validation, tuning and empty shapes" begin
    source=FaerSparseMatrix(sparse([4.0 1; 1 3]))
    for ordering in (:bad, [1, 1], [1], [0, 2])
        @test_throws (ordering==[1] ? DimensionMismatch : ArgumentError) cholesky(
            source;
            ordering,
        )
    end
    @test_throws ArgumentError cholesky(source; algorithm = :bad)
    for dense in (Inf, -Inf, NaN)
        @test_throws ArgumentError cholesky(source; dense)
    end
    for dense in (-1.0, 0.0, 10.0), aggressive in (false, true)
        f=cholesky(source; dense, aggressive)
        @test f\ones(2) ≈ [4.0 1; 1 3]\ones(2)
        @test isperm(Vector(f.p))
    end
    p=FaerPermutation([2, 1])
    close(p)
    @test_throws ArgumentError cholesky(source; ordering = p)
    for kind in (:lu, :qr)
        @test_throws ArgumentError FaerSparseSymbolic(source; kind, ordering = :identity)
        @test FaerSparseSymbolic(source; kind, algorithm = :simplicial) isa
              FaerSparseSymbolic{kind}
        @test_throws ArgumentError FaerSparseSymbolic(source; kind, dense = 2.0)
    end
    for n in (0, 1),
        algorithm in (:auto, :simplicial, :supernodal),
        ordering in (:amd, :identity, collect(n:-1:1))

        a=FaerSparseMatrix(sparse(1:n, 1:n, fill(2.0, n), n, n))
        f=cholesky(a; algorithm, ordering)
        @test f\ones(n) ≈ fill(0.5, n)
        @test Vector(f.p)==1:n
        @test f.factor_storage>=n
    end
    @test isempty(
        detect_ambiguities(Colmat, Base, LinearAlgebra, SparseArrays; recursive = true),
    )
end

@testset "Julia sparse Cholesky perm keyword" begin
    source=FaerSparseMatrix(sparse([4.0 1; 1 3]))
    f=cholesky(source; perm = [2, 1], algorithm = :simplicial)
    @test Vector(f.p)==[2, 1]
    @test f\ones(2) ≈ [4.0 1; 1 3]\ones(2)
    @test_throws ArgumentError cholesky(source; perm = [2, 1], ordering = :amd)
    @test hasproperty(f, :p) &&
          hasproperty(f, :algorithm) &&
          hasproperty(f, :factor_storage)
    @test hasproperty(f.plan, :p) && hasproperty(f.plan, :algorithm)
end

@testset "Sparse Cholesky reports one-based failed pivots" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        format in (:csc, :csr),
        algorithm in (:simplicial, :supernodal),
        pivot = 1:4

        d=fill(T(2), 4)
        d[pivot]=T(-1)
        source=FaerSparseMatrix(spdiagm(0=>d); format)
        f=cholesky(source; algorithm, ordering = :identity, check = false)
        @test !issuccess(f)
        @test f.info==pivot
        @test_throws PosDefException f\ones(T, 4)
    end
end
