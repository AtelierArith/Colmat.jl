using Colmat, LinearAlgebra, SparseArrays, Test
function ccr_fixture(T, n)
    X = T[0.01sin(i + 2j) for i in 1:n, j in 1:n]
    T <: Complex && (X .+= T(0.01im) .* T[cos(2i + j) for i in 1:n, j in 1:n])
    A = X + X'
    for i in 1:n
        A[i, i] += T(5 + i / max(n, 1))
    end
    return A
end
function ccr_source(A, side, format)
    stored = copy(A)
    for j in 1:size(A, 2), i in 1:size(A, 1)
        (side == :L ? i < j : i > j) && (stored[i, j] = eltype(A)(NaN))
    end
    return FaerSparseMatrix(sparse(stored); format)
end
@testset "Sparse Cholesky construction and update resources" begin
    for width in (0, 1, 2)
        team = width == 0 ? nothing : FaerTeam(width)
        for T in (Float32, Float64, ComplexF32, ComplexF64),
            algorithm in (:auto, :simplicial, :supernodal),
            format in (:csc, :csr),
            side in (:L, :U),
            n in (0, 1, 4, 64),
            reversed in (false, true)

            A = ccr_fixture(T, n)
            a = ccr_source(A, side, format)
            ordering = reversed ? reverse(collect(1:n)) : :identity
            p = FaerSparseSymbolic(a; kind=:cholesky, uplo=side, algorithm, ordering)
            req = scratch_requirements(factorize, p, a; team)
            work = FaerScratch(factorize, p, a; team)
            cap = scratch_capacity(work)
            f = factorize(p, a; scratch=work, team)
            @test issuccess(f)
            @test scratch_capacity(work) == cap
            @test Matrix(f) ≈ A atol = 1e-3 rtol = 1e-3
            @test Matrix(f \ ones(T, n, 2)) ≈ A \ ones(T, n, 2) atol = 1e-3 rtol = 1e-3
            g = cholesky(Hermitian(a, side); algorithm, ordering, scratch=work, team)
            @test Matrix(g) ≈ A atol = 1e-3 rtol = 1e-3
            @test scratch_requirements(cholesky!, f, a; team) == req
            update = FaerScratch(cholesky!, f, a; team)
            if req.bytes > 0
                short = FaerScratch(req.bytes - 1; alignment=req.alignment)
                @test_throws ArgumentError factorize(p, a; scratch=short, team)
                @test_throws ArgumentError cholesky!(f, a; scratch=short, team)
                @test reserve!(short, cholesky!, f, a; team) === short
                close(short)
            end
            D = 2A
            d = ccr_source(D, side, format == :csc ? :csr : :csc)
            independent = deepcopy(f)
            oldplan = f.plan
            oldsource = f.source
            @test cholesky!(f, d; scratch=update, team) === f
            @test !isopen(oldplan) && !isopen(oldsource)
            @test Matrix(f) ≈ D atol = 1e-3 rtol = 1e-3
            @test Matrix(independent) ≈ A atol = 1e-3 rtol = 1e-3
            close(a)
            close(d)
            close(p)
            close(work)
            close(update)
            close(f.source)
            close(f.plan)
            @test Matrix(f \ ones(T, n, 2)) ≈ D \ ones(T, n, 2) atol = 1e-3 rtol = 1e-3
            close(f)
            close(g)
            close(independent)
        end
        isnothing(team) || close(team)
    end
end
@testset "Cholesky non-positive pivots, staging and recovery" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        algorithm in (:auto, :simplicial, :supernodal), side in (:L, :U),
        pivot in (1, 3)

        good = FaerSparseMatrix(spdiagm(0 => fill(T(2), 4)))
        diag = fill(T(2), 4)
        diag[pivot] = T(-1)
        bad = FaerSparseMatrix(spdiagm(0 => diag); format=:csr)
        p = FaerSparseSymbolic(
            good; kind=:cholesky, uplo=side, algorithm, ordering=:identity
        )
        team = FaerTeam(2)
        work = FaerScratch(factorize, p, good; team)
        f = factorize(p, good; scratch=work, team)
        before = (f.handle, f.plan, f.source, f.info, f.dims)
        err = try
            cholesky!(f, bad; scratch=work, team)
            nothing
        catch e
            e
        end
        @test err isa PosDefException
        @test err.info == pivot
        @test (f.handle, f.plan, f.source, f.info, f.dims) == before
        @test Matrix(f) ≈ 2Matrix{T}(I, 4, 4)
        failed = factorize(p, bad; check=false, scratch=work, team)
        @test !issuccess(failed)
        @test failed.info == pivot
        @test_throws PosDefException failed \ ones(T, 4)
        @test cholesky!(f, bad; check=false, scratch=work, team) === f
        @test !issuccess(f) && f.info == pivot
        @test cholesky!(f, good; scratch=work, team) === f
        @test issuccess(f)
        close(failed)
        close(f)
        close(team)
        close(work)
        close(p)
        close(good)
        close(bad)
    end
end
@testset "Cholesky construction resource validation and shared Tasks" begin
    a = FaerSparseMatrix(sparse([4.0 0.1; 0.1 5]))
    p = FaerSparseSymbolic(a; kind=:cholesky)
    team = FaerTeam(2)
    work = FaerScratch(factorize, p, a; team)
    f = factorize(p, a; team)
    before = (f.handle, f.plan, f.source, f.info, f.dims)
    @test_throws ArgumentError factorize(p, a; team=:invalid)
    @test_throws ArgumentError cholesky!(f, a; scratch=:invalid, team)
    bad = FaerSparseMatrix(spdiagm(0 => [4.0, 5]))
    @test_throws ArgumentError cholesky!(f, bad; scratch=work, team)
    @test (f.handle, f.plan, f.source, f.info, f.dims) == before
    invalid = copy(a)
    invalid[1, 1] = Inf
    @test_throws ArgumentError cholesky!(f, invalid; scratch=work, team)
    @test (f.handle, f.plan, f.source, f.info, f.dims) == before
    jobs = [@async begin
        yield()
        cholesky!(f, a; scratch=work, team)
        Matrix(f)
    end for _ in 1:3]
    @test all(x -> x ≈ Matrix(a), fetch.(jobs))
    close(team)
    @test_throws ArgumentError scratch_requirements(factorize, p, a; team)
    @test_throws ArgumentError cholesky!(f, a; scratch=work, team)
    team = FaerTeam(2)
    close(work)
    @test_throws ArgumentError cholesky!(f, a; scratch=work, team)
    @test Matrix(cholesky(a; team)) ≈ Matrix(a)
    close(f)
    close(a)
    close(p)
    close(team)
    close(bad)
    close(invalid)
end
