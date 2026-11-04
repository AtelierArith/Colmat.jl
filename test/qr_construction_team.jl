using Colmat, LinearAlgebra, SparseArrays, Test
function qct_fixture(T, m, n, deficient)
    A=T[0.02sin(i+2j) for i = 1:m, j = 1:n]
    for j = 1:min(m, n)
        A[j, j]+=T(4+j/max(n, 1))
    end
    T<:Complex && (A .+= T(0.03im) .* T[cos(2i+j) for i = 1:m, j = 1:n])
    deficient && n>0 && (A[:, 1].=0)
    A
end
@testset "Sparse QR numeric construction teams" begin
    for width in (1, 2)
        team=FaerTeam(width)
        for T in (Float32, Float64, ComplexF32, ComplexF64),
            algorithm in (:auto, :supernodal),
            format in (:csc, :csr),
            (m, n) in ((0, 0), (3, 0), (1, 1), (7, 4), (96, 64)),
            deficient in (false, true)

            A=qct_fixture(T, m, n, deficient)
            a=FaerSparseMatrix(sparse(A); format)
            p=FaerSparseSymbolic(a; kind = :qr, algorithm, ordering = reverse(collect(1:n)))
            req=scratch_requirements(factorize, p, a; team)
            work=FaerScratch(factorize, p, a; team)
            cap=scratch_capacity(work)
            @test cap.bytes>=req.bytes && cap.alignment>=req.alignment
            f=factorize(p, a; scratch = work, team)
            g=factorize(p, a; team)
            h=qr(a; algorithm, ordering = reverse(collect(1:n)), scratch = work, team)
            @test scratch_capacity(work)==cap
            for result in (f, g, h)
                @test Matrix(result)≈A atol=1e-3 rtol=1e-3
                @test Matrix(result\ones(T, m, 2))≈pinv(A; rtol = 1e-5)*ones(T, m, 2) atol=1e-3 rtol=1e-3
            end
            if req.bytes>0
                short=FaerScratch(req.bytes-1; alignment = req.alignment)
                @test_throws ArgumentError factorize(p, a; scratch = short, team)
                @test reserve!(short, factorize, p, a; team)===short
                r=factorize(p, a; scratch = short, team)
                @test Matrix(r)≈A atol=1e-3 rtol=1e-3
                close(r)
                close(short)
            end
            independent=deepcopy(f)
            close(a)
            close(p)
            close(work)
            for result in (f, g, h)
                close(result.source)
                close(result.plan)
                @test Matrix(result)≈A atol=1e-3 rtol=1e-3
                close(result)
            end
            @test Matrix(independent)≈A atol=1e-3 rtol=1e-3
            close(independent)
        end
        close(team)
    end
end
@testset "Construction resource failures and reuse" begin
    team=FaerTeam(2)
    A=[4.0 0.2; 0.1 5; 0.2 0.3]
    a=FaerSparseMatrix(sparse(A))
    p=FaerSparseSymbolic(a; kind = :qr, algorithm = :supernodal)
    work=FaerScratch(factorize, p, a; team)
    @test_throws ArgumentError factorize(p, a; team = :invalid)
    @test_throws ArgumentError scratch_requirements(factorize, p, a; team = :invalid)
    @test_throws ArgumentError factorize(p, a; scratch = :invalid, team)
    bad=FaerSparseMatrix(sparse([4.0 0; 0.1 5; 0.2 0.3]))
    @test_throws ArgumentError factorize(p, bad; scratch = work, team)
    @test Matrix(a)==A
    close(bad)
    # Shared scratch/team are serialized by the package lock across Julia Tasks.
    jobs=[@async begin
        yield()
        f=factorize(p, a; scratch = work, team)
        actual=Matrix(f)
        close(f)
        actual
    end for _ = 1:4]
    @test all(x->x≈A, fetch.(jobs))
    close(team)
    @test_throws ArgumentError factorize(p, a; team)
    @test_throws ArgumentError factorize(p, a; scratch = work, team)
    @test_throws ArgumentError scratch_requirements(factorize, p, a; team)
    team=FaerTeam(2)
    close(work)
    @test_throws ArgumentError factorize(p, a; scratch = work, team)
    @test Matrix(qr(a; team))≈A
    for algorithm in (:simplicial, :supernodal)
        wide=FaerSparseMatrix(sparse(ones(2, 3)))
        @test_throws Colmat.RustCall.RustError qr(wide; algorithm, team)
        close(wide)
    end
    simp=FaerSparseSymbolic(a; kind = :qr, algorithm = :simplicial)
    @test_throws Colmat.RustCall.RustError factorize(simp, a; team)
    close(simp)
    square=FaerSparseMatrix(sparse([3.0 0; 0 4]))
    luplan=FaerSparseSymbolic(square; kind = :lu)
    @test_throws ArgumentError factorize(luplan, square; team)
    close(luplan)
    close(square)
    close(p)
    close(a)
    close(team)
end
