using Colmat, LinearAlgebra, SparseArrays, Test
function wqc_fixture(T, m, n, mode)
    A = T[0.02sin(i + 2j) for i in 1:m, j in 1:n]
    for j in 1:n
        m > 0 && (A[mod1(j, m), j] += T(4 + j / max(n, 1)))
    end
    T <: Complex && (A .+= T(0.01im) .* T[cos(2i + j) for i in 1:m, j in 1:n])
    mode == 1 && m > 0 && (A[:, 1:m] .= 0)
    mode == 2 && m > 0 && (A[1, :] .= 0)
    return A
end
@testset "Wide sparse QR construction resources" begin
    for width in (0, 1, 2)
        team = width == 0 ? nothing : FaerTeam(width)
        for T in (Float32, Float64, ComplexF32, ComplexF64),
            algorithm in (:auto, :simplicial, :supernodal),
            format in (:csc, :csr),
            (m, n) in ((0, 3), (1, 3), (4, 7), (64, 96)),
            mode in 0:2,
            reversed in (false, true)

            A = wqc_fixture(T, m, n, mode)
            a = FaerSparseMatrix(sparse(A); format)
            ordering = reversed ? reverse(collect(1:n)) : :identity
            p = FaerSparseSymbolic(a; kind=:qr, algorithm, ordering)
            req = scratch_requirements(factorize, p, a; team)
            work = FaerScratch(factorize, p, a; team)
            cap = scratch_capacity(work)
            f = factorize(p, a; scratch=work, team)
            @test scratch_capacity(work) == cap
            @test Matrix(f) ≈ A atol = 2e-3 rtol = 2e-3
            b = ones(T, m, 2)
            gold = pinv(A; rtol=1e-5) * b
            @test Matrix(f \ b) ≈ gold atol = 2e-3 rtol = 2e-3
            if req.bytes > 0
                short = FaerScratch(req.bytes - 1; alignment=req.alignment)
                @test_throws ArgumentError factorize(p, a; scratch=short, team)
                @test reserve!(short, factorize, p, a; team) === short
                g = qr(a; algorithm, ordering, scratch=short, team)
                @test Matrix(g) ≈ A atol = 2e-3 rtol = 2e-3
                close(g)
                close(short)
            end
            g = deepcopy(f)
            close(a)
            close(p)
            close(work)
            close(f.source)
            close(f.plan)
            @test Matrix(f \ b) ≈ gold atol = 2e-3 rtol = 2e-3
            close(f)
            @test Matrix(g \ b) ≈ gold atol = 2e-3 rtol = 2e-3
            close(g)
        end
        isnothing(team) || close(team)
    end
end
@testset "Wide construction failures, team-only path and shared resources" begin
    A = [4.0 0.1 0.2; 0.2 5 0.3]
    a = FaerSparseMatrix(sparse(A))
    p = FaerSparseSymbolic(a; kind=:qr, algorithm=:supernodal)
    team = FaerTeam(2)
    work = FaerScratch(factorize, p, a; team)
    f = factorize(p, a; team)
    @test Matrix(f) ≈ A
    bad = FaerSparseMatrix(sparse([4.0 0 0.2; 0.2 5 0.3]))
    @test_throws ArgumentError factorize(p, bad; scratch=work, team)
    @test Matrix(f) ≈ A
    jobs = [@async begin
        yield()
        g = factorize(p, a; scratch=work, team)
        x = Matrix(g)
        close(g)
        x
    end for _ in 1:3]
    @test all(x -> x ≈ A, fetch.(jobs))
    close(team)
    @test Vector(f \ ones(2)) ≈ pinv(A) * ones(2)
    @test_throws ArgumentError factorize(p, a; scratch=work, team)
    @test_throws ArgumentError scratch_requirements(factorize, p, a; team)
    team = FaerTeam(2)
    close(work)
    @test_throws ArgumentError factorize(p, a; scratch=work, team)
    close(team)
    close(p)
    close(a)
    close(bad)
    close(f)
end
