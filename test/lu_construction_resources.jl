using Colmat, LinearAlgebra, SparseArrays, Test
function lcr_fixture(T, n)
    B = T[0.01sin(i + 2j) for i in 1:n, j in 1:n]
    T <: Complex && (B .+= T(0.01im) .* T[cos(2i + j) for i in 1:n, j in 1:n])
    for i in 1:n
        B[i, i] += T(5 + i / max(n, 1))
    end
    return B[reverse(collect(1:n)), :]
end
function lcr_stored(A, D, format)
    s = sparse(A)
    for j in 1:size(A, 2), p in nzrange(s, j)
        nonzeros(s)[p] = D[rowvals(s)[p], j]
    end
    return FaerSparseMatrix(s; format)
end
@testset "Sparse LU construction and staged update resources" begin
    for width in (0, 1, 2)
        team = width == 0 ? nothing : FaerTeam(width)
        for T in (Float32, Float64, ComplexF32, ComplexF64),
            algorithm in (:auto, :simplicial, :supernodal),
            format in (:csc, :csr),
            n in (0, 1, 4, 64),
            order in (:identity, :reverse, :colamd)

            A = lcr_fixture(T, n)
            a = FaerSparseMatrix(sparse(A); format)
            ordering = order == :reverse ? reverse(collect(1:n)) : order
            controls = if order == :colamd
                (dense_row=-1.0, dense_col=0.0, aggressive=false)
            else
                (dense_row=0.5, dense_col=0.5, aggressive=true)
            end
            p = FaerSparseSymbolic(a; kind=:lu, algorithm, ordering, controls...)
            req = scratch_requirements(factorize, p, a; team)
            work = FaerScratch(factorize, p, a; team)
            cap = scratch_capacity(work)
            f = factorize(p, a; scratch=work, team)
            @test issuccess(f)
            @test scratch_capacity(work) == cap
            @test Matrix(f) ≈ A atol = 1e-3 rtol = 1e-3
            @test Matrix(f.L) * Matrix(f.U) ≈ A[Vector(f.p), Vector(f.q)] atol = 1e-3 rtol =
                1e-3
            b = ones(T, n, 2)
            @test Matrix(f \ b) ≈ A \ b atol = 1e-3 rtol = 1e-3
            @test Matrix(transpose(f) \ b) ≈ transpose(A) \ b atol = 1e-3 rtol = 1e-3
            @test Matrix(adjoint(f) \ b) ≈ adjoint(A) \ b atol = 1e-3 rtol = 1e-3
            mixed = ones(ComplexF64, n, 2)
            @test Matrix(f \ mixed) ≈ ComplexF64.(A) \ mixed atol = 1e-3 rtol = 1e-3
            g = lu(a; algorithm, ordering, controls..., scratch=work, team)
            @test Matrix(g) ≈ A atol = 1e-3 rtol = 1e-3
            @test scratch_requirements(lu!, f, a; team) == req
            update = FaerScratch(lu!, f, a; team)
            if req.bytes > 0
                short = FaerScratch(req.bytes - 1; alignment=req.alignment)
                @test_throws ArgumentError factorize(p, a; scratch=short, team)
                @test_throws ArgumentError lu!(f, a; scratch=short, team)
                @test reserve!(short, lu!, f, a; team) === short
                close(short)
            end
            d = lcr_stored(A, 2A, format == :csc ? :csr : :csc)
            independent = deepcopy(f)
            columns = Vector(f.q)
            oldplan = f.plan
            oldsource = f.source
            @test lu!(f, d; scratch=update, team) === f
            @test !isopen(oldplan) && !isopen(oldsource)
            @test Vector(f.q) == columns
            @test Matrix(f) ≈ 2A atol = 1e-3 rtol = 1e-3
            @test Matrix(independent) ≈ A atol = 1e-3 rtol = 1e-3
            close(a)
            close(d)
            close(p)
            close(work)
            close(update)
            close(f.source)
            close(f.plan)
            @test Matrix(f \ b) ≈ (2A) \ b atol = 1e-3 rtol = 1e-3
            @test Matrix(f \ mixed) ≈ ComplexF64.(2A) \ mixed atol = 1e-3 rtol = 1e-3
            close(f)
            close(g)
            close(independent)
        end
        isnothing(team) || close(team)
    end
end
@testset "Singular LU construction, atomic update and recovery" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        algorithm in (:auto, :simplicial, :supernodal)

        A = lcr_fixture(T, 4)
        a = FaerSparseMatrix(sparse(A))
        bad = lcr_stored(A, zeros(T, 4, 4), :csr)
        p = FaerSparseSymbolic(a; kind=:lu, algorithm, ordering=:identity)
        team = FaerTeam(2)
        work = FaerScratch(factorize, p, a; team)
        f = factorize(p, a; scratch=work, team)
        before = (f.handle, f.plan, f.source, f.info, f.dims)
        err = try
            lu!(f, bad; scratch=work, team)
            nothing
        catch e
            e
        end
        @test err isa SingularException
        @test err.info == 1
        @test (f.handle, f.plan, f.source, f.info, f.dims) == before
        @test Matrix(f) ≈ A atol = 1e-3 rtol = 1e-3
        failed = factorize(p, bad; check=false, scratch=work, team)
        @test !issuccess(failed) && failed.info == 1
        @test_throws SingularException failed \ ones(T, 4)
        @test lu!(f, bad; check=false, scratch=work, team) === f
        @test !issuccess(f) && f.info == 1
        @test lu!(f, a; scratch=work, team) === f
        @test issuccess(f)
        close(failed)
        close(f)
        close(p)
        close(team)
        close(work)
        close(a)
        close(bad)
    end
end
@testset "LU resource errors and shared owner reuse" begin
    a = FaerSparseMatrix(sparse([4.0 0.1; 0.2 5]))
    p = FaerSparseSymbolic(a; kind=:lu, algorithm=:supernodal)
    team = FaerTeam(2)
    work = FaerScratch(factorize, p, a; team)
    f = factorize(p, a; team)
    before = (f.handle, f.plan, f.source, f.info, f.dims)
    @test_throws ArgumentError lu!(f, a; scratch=:invalid, team)
    @test_throws ArgumentError scratch_requirements(factorize, p, a; team=:invalid)
    bad = FaerSparseMatrix(spdiagm(0 => [4.0, 5]))
    @test_throws ArgumentError lu!(f, bad; scratch=work, team)
    invalid = copy(a)
    invalid[1, 1] = Inf
    @test_throws ArgumentError lu!(f, invalid; scratch=work, team)
    @test (f.handle, f.plan, f.source, f.info, f.dims) == before
    jobs = [@async begin
        yield()
        lu!(f, a; scratch=work, team)
        Matrix(f)
    end for _ in 1:3]
    @test all(x -> x ≈ Matrix(a), fetch.(jobs))
    close(team)
    @test_throws ArgumentError reserve!(work, lu!, f, a; team)
    @test_throws ArgumentError lu!(f, a; scratch=work, team)
    team = FaerTeam(2)
    close(work)
    @test_throws ArgumentError lu!(f, a; scratch=work, team)
    close(f)
    close(team)
    close(a)
    close(p)
    close(bad)
    close(invalid)
end
