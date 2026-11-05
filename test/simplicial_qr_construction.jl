using Colmat, LinearAlgebra, SparseArrays, Test
function sqc_fixture(T, m, n, deficient)
    A=T[0.01sin(i+2j) for i = 1:m, j = 1:n]
    for j = 1:min(m, n)
        A[j, j]+=T(5+j/max(n, 1))
    end
    T<:Complex && (A .+= T(0.02im) .* T[cos(2i+j) for i = 1:m, j = 1:n])
    deficient && n>0 && (A[:, 1].=0)
    return A
end
@testset "Explicit simplicial QR construction scratch" begin
    for width in (0, 1, 2)
        team=width==0 ? nothing : FaerTeam(width)
        for T in (Float32, Float64, ComplexF32, ComplexF64),
            format in (:csc, :csr),
            (m, n) in ((0, 0), (3, 0), (1, 1), (7, 4), (96, 64)),
            deficient in (false, true),
            reversed in (false, true)

            A=sqc_fixture(T, m, n, deficient)
            a=FaerSparseMatrix(sparse(A); format)
            ordering=reversed ? reverse(collect(1:n)) : :identity
            p=FaerSparseSymbolic(a; kind = :qr, algorithm = :simplicial, ordering)
            req=scratch_requirements(factorize, p, a; team)
            @test req==scratch_requirements(factorize, p, a) # kernel has no parallel parameter
            work=FaerScratch(factorize, p, a; team)
            cap=scratch_capacity(work)
            f=factorize(p, a; scratch = work, team)
            @test scratch_capacity(work)==cap
            @test Matrix(f)≈A atol=1e-3 rtol=1e-3
            b=ones(T, m, 2)
            gold=pinv(A; rtol = 1e-5)*b
            @test Matrix(f\b)≈gold atol=1e-3 rtol=1e-3
            g=qr(a; algorithm = :simplicial, ordering, scratch = work, team)
            @test Matrix(g)≈A atol=1e-3 rtol=1e-3
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
            close(f.source)
            close(f.plan)
            @test Matrix(f\b)≈gold atol=1e-3 rtol=1e-3
            close(f)
            @test Matrix(independent\b)≈gold atol=1e-3 rtol=1e-3
            close(independent)
            close(g)
        end
        isnothing(team)||close(team)
    end
end
@testset "Simplicial construction resource errors and team-only path" begin
    team=FaerTeam(2)
    A=[4.0 0.2; 0.1 5; 0.2 0.3]
    a=FaerSparseMatrix(sparse(A))
    p=FaerSparseSymbolic(a; kind = :qr, algorithm = :simplicial)
    work=FaerScratch(factorize, p, a; team)
    g=factorize(p, a; team)
    close(team)
    @test Matrix(g)≈A
    @test_throws ArgumentError factorize(p, a; scratch = work, team)
    @test_throws ArgumentError reserve!(work, factorize, p, a; team)
    team=FaerTeam(2)
    bad=FaerSparseMatrix(sparse([4.0 0; 0.1 5; 0.2 0.3]))
    @test_throws ArgumentError factorize(p, bad; scratch = work, team)
    @test Matrix(g)≈A
    f=factorize(p, a; scratch = work, team)
    @test Matrix(f)≈A
    close(work)
    @test_throws ArgumentError factorize(p, a; scratch = work, team)
    close(f)
    close(g)
    close(team)
    close(p)
    close(a)
    close(bad)
end
