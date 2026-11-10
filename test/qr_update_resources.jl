using Colmat, LinearAlgebra, SparseArrays, Test
function qur_fixture(T, m, n)
    A = T[0.01sin(i + 2j) for i in 1:m, j in 1:n]
    for j in 1:min(m, n)
        A[j, j] += T(5 + j / max(n, 1))
    end
    T <: Complex && (A .+= T(0.01im) .* T[cos(2i + j) for i in 1:m, j in 1:n])
    return A
end
function qur_with_pattern(A, D, format)
    s = sparse(A)
    for j in 1:size(A, 2), p in nzrange(s, j)
        nonzeros(s)[p] = D[rowvals(s)[p], j]
    end
    return FaerSparseMatrix(s; format)
end
@testset "Staged sparse QR updates with construction resources" begin
    for width in (0, 1, 2)
        team = width == 0 ? nothing : FaerTeam(width)
        for T in (Float32, Float64, ComplexF32, ComplexF64),
            algorithm in (:auto, :simplicial, :supernodal), format in (:csc, :csr),
            (m, n) in ((0, 0), (3, 0), (0, 3), (1, 1), (7, 4), (4, 7), (96, 64), (64, 96))

            A = qur_fixture(T, m, n)
            a = FaerSparseMatrix(sparse(A); format)
            ordering = reverse(collect(1:n))
            f = qr(a; algorithm, ordering)
            independent = deepcopy(f)
            req = scratch_requirements(qr!, f, a; team)
            @test req == scratch_requirements(factorize, f.plan, a; team)
            work = FaerScratch(qr!, f, a; team)
            cap = scratch_capacity(work)
            D = copy(A)
            if m >= n
                n > 0 && (D[:, 1] .= 0)
            else
                m > 0 && (D[:, 1:m] .= 0)
            end
            d = qur_with_pattern(A, D, format == :csc ? :csr : :csc)
            oldplan = f.plan
            oldsource = f.source
            @test qr!(f, d; scratch=work, team) === f
            @test !isopen(oldplan) && !isopen(oldsource)
            @test scratch_capacity(work) == cap
            @test Vector(f.pcol) == ordering
            @test Matrix(f) ≈ D atol = 2e-3 rtol = 2e-3
            b = ones(T, m, 2)
            @test Matrix(f \ b) ≈ pinv(D; rtol=1e-5) * b atol = 2e-3 rtol = 2e-3
            @test Matrix(independent) ≈ A atol = 2e-3 rtol = 2e-3
            if req.bytes > 0
                short = FaerScratch(req.bytes - 1; alignment=req.alignment)
                before = (f.handle, f.plan, f.source, f.info, f.dims)
                @test_throws ArgumentError qr!(f, a; scratch=short, team)
                @test (f.handle, f.plan, f.source, f.info, f.dims) == before
                @test Matrix(f) ≈ D atol = 2e-3 rtol = 2e-3
                @test reserve!(short, qr!, f, a; team) === short
                @test qr!(f, a; scratch=short, team) === f
                close(short)
            else
                @test qr!(f, a; scratch=work, team) === f
            end
            @test Matrix(f) ≈ A atol = 2e-3 rtol = 2e-3
            # A retained snapshot may itself be the update input; the new copy outlives it.
            snapshot = f.source
            @test qr!(f, snapshot; scratch=work, team) === f
            @test !isopen(snapshot)
            @test Matrix(f) ≈ A atol = 2e-3 rtol = 2e-3
            close(a)
            close(d)
            close(work)
            close(f.source)
            close(f.plan)
            @test Matrix(f \ b) ≈ pinv(A; rtol=1e-5) * b atol = 2e-3 rtol = 2e-3
            close(f)
            close(independent)
        end
        isnothing(team) || close(team)
    end
end
@testset "QR update failure atomicity and resource reuse" begin
    for algorithm in (:auto, :simplicial, :supernodal), (m, n) in ((5, 3), (3, 5))
        A = qur_fixture(ComplexF64, m, n)
        a = FaerSparseMatrix(sparse(A))
        f = qr(a; algorithm)
        team = FaerTeam(2)
        work = FaerScratch(qr!, f, a; team)
        before = (f.handle, f.plan, f.source, f.info, f.dims)
        badpattern = sparse(A)
        badpattern[1, 1] = 0
        dropzeros!(badpattern)
        bad = FaerSparseMatrix(badpattern)
        @test_throws ArgumentError qr!(f, bad; scratch=work, team)
        @test (f.handle, f.plan, f.source, f.info, f.dims) == before
        @test Matrix(f) ≈ A
        invalid = copy(a)
        invalid[1, 1] = ComplexF64(Inf)
        @test_throws ArgumentError qr!(f, invalid; scratch=work, team)
        @test (f.handle, f.plan, f.source, f.info, f.dims) == before
        @test Matrix(f) ≈ A
        close(invalid)
        closed = copy(a)
        close(closed)
        @test_throws ArgumentError qr!(f, closed; scratch=work, team)
        @test (f.handle, f.plan, f.source, f.info, f.dims) == before
        count = a.stored
        a.stored += 1
        @test_throws Colmat.RustCall.RustError qr!(f, a; scratch=work, team)
        @test (f.handle, f.plan, f.source, f.info, f.dims) == before
        a.stored = count
        @test_throws ArgumentError qr!(f, a; scratch=:bad, team)
        @test_throws ArgumentError qr!(f, a; scratch=work, team=:bad)
        @test_throws ArgumentError scratch_requirements(qr!, f, a; team=:bad)
        close(team)
        @test_throws ArgumentError qr!(f, a; scratch=work, team)
        @test_throws ArgumentError reserve!(work, qr!, f, a; team)
        @test (f.handle, f.plan, f.source, f.info, f.dims) == before
        team = FaerTeam(2)
        close(work)
        @test_throws ArgumentError qr!(f, a; scratch=work, team)
        @test (f.handle, f.plan, f.source, f.info, f.dims) == before
        # Team-only updates allocate temporary construction scratch.
        @test qr!(f, a; team) === f
        work = FaerScratch(qr!, f, a; team)
        jobs = [@async begin
            yield()
            qr!(f, a; scratch=work, team)
            Matrix(f)
        end for _ in 1:3]
        @test all(x -> x ≈ A, fetch.(jobs))
        close(f.plan)
        before = (f.handle, f.plan, f.source, f.info, f.dims)
        @test_throws ArgumentError qr!(f, a; scratch=work, team)
        @test (f.handle, f.plan, f.source, f.info, f.dims) == before
        @test Matrix(f) ≈ A
        close(f)
        @test_throws ArgumentError scratch_requirements(qr!, f, a; team)
        @test_throws ArgumentError qr!(f, a; scratch=work, team)
        close(team)
        close(work)
        close(a)
        close(bad)
    end
end
