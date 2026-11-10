using Colmat, Test, LinearAlgebra, SparseArrays, Random

function _symbolic_ordering_cases(rng)
    tridiagonal = spdiagm(-1 => fill(-1.0, 39), 0 => fill(4.0, 40), 1 => fill(-1.0, 39))
    # Dense enough for faer's auto heuristic to select supernodal.
    block = sprand(rng, 150, 150, 0.5)
    block = block + block' + 150I
    grid = spdiagm(-1 => fill(-1.0, 11), 0 => fill(2.0, 12), 1 => fill(-1.0, 11))
    laplacian = kron(grid, sparse(1.0I, 12, 12)) + kron(sparse(1.0I, 12, 12), grid)
    # Stored entries outside the selected triangle still participate in faer's AMD.
    lopsided = sparse(
        [1, 2, 3, 4, 5, 1, 1, 2, 5, 5],
        [1, 2, 3, 4, 5, 2, 5, 4, 3, 1],
        [4.0, 4, 4, 4, 4, 1, 1, 1, 1, 1],
        5,
        5,
    )
    return (tridiagonal, block, laplacian, lopsided, sparse(4.0I, 5, 5), spzeros(0, 0))
end

@testset "Cholesky symbolic AMD uses caller scratch with identical analysis" begin
    rng = MersenneTwister(118)
    seen = Set{Symbol}()
    for a in _symbolic_ordering_cases(rng),
        format in (:csc, :csr), uplo in (:L, :U),
        algorithm in (:auto, :simplicial, :supernodal)

        source = FaerSparseMatrix(a; format)
        pattern = FaerSparsity(source)
        for input in (source, pattern),
            ordering in (:default, :amd),
            (dense, aggressive) in ((10.0, true), (-1.0, false))

            opts = (; kind=:cholesky, uplo, ordering, algorithm, dense, aggressive)
            req = scratch_requirements(FaerSparseSymbolic, input; opts...)
            @test req.bytes >= 0 && ispow2(req.alignment)
            size(a, 1) > 0 && @test req.bytes > 0
            s = FaerScratch(FaerSparseSymbolic, input; opts...)
            @test scratch_capacity(s) == req
            ref = FaerSparseSymbolic(input; opts...)
            plan = FaerSparseSymbolic(input; scratch=s, opts...)
            @test Vector(plan.p) == Vector(ref.p)
            @test plan.algorithm == ref.algorithm
            @test plan.factor_storage == ref.factor_storage
            @test scratch_capacity(s) == req
            algorithm === :auto && push!(seen, ref.algorithm)
            if size(a, 1) > 0 && input === source && format === :csc
                F = factorize(plan, source)
                G = factorize(ref, source)
                b = collect(1.0:size(a, 1))
                @test F \ b ≈ G \ b
            end
            if req.bytes > 0
                small = FaerScratch(req.bytes - 1; alignment=req.alignment)
                @test_throws ArgumentError FaerSparseSymbolic(input; scratch=small, opts...)
                @test reserve!(small, FaerSparseSymbolic, input; opts...) === small
                @test Vector(FaerSparseSymbolic(input; scratch=small, opts...).p) ==
                    Vector(ref.p)
                close(small)
            end
            close(s)
            @test_throws ArgumentError FaerSparseSymbolic(input; scratch=s, opts...)
            @test_throws ArgumentError FaerSparseSymbolic(input; scratch=ones(3), opts...)
        end
        @test sparse(source) == a
        close(source)
        close(pattern)
    end
    @test seen == Set((:simplicial, :supernodal))
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        a = sparse(T[4 1 0; 1 4 1; 0 1 4])
        source = FaerSparseMatrix(a)
        s = FaerScratch(FaerSparseSymbolic, source; kind=:cholesky)
        plan = FaerSparseSymbolic(source; kind=:cholesky, scratch=s)
        F = factorize(plan, source)
        @test F \ T[1, 2, 3] ≈ Matrix(a) \ T[1, 2, 3]
        close(s)
        close(source)
    end
end

@testset "Borrowed Julia CSC symbolic scratch" begin
    a = sparse([4.0 1 0 1; 1 4 1 0; 0 1 4 1; 1 0 1 4])
    b = faerview(a)
    for kind in (:cholesky, :lu)
        s = FaerScratch(FaerSparseSymbolic, b; kind)
        @test scratch_capacity(s) == scratch_requirements(FaerSparseSymbolic, b; kind)
        plan = FaerSparseSymbolic(b; kind, scratch=s)
        F = factorize(plan, b)
        @test F \ ones(4) ≈ Matrix(a) \ ones(4)
        close(s)
    end
    @test a == sparse([4.0 1 0 1; 1 4 1 0; 0 1 4 1; 1 0 1 4])
end

@testset "Explicit Cholesky orderings need no symbolic ordering scratch" begin
    a = FaerSparseMatrix(sparse([4.0 1 0; 1 4 1; 0 1 4]))
    for ordering in (:identity, [3, 1, 2])
        req = scratch_requirements(FaerSparseSymbolic, a; kind=:cholesky, ordering)
        @test req.bytes == 0
        s = FaerScratch(req.bytes; alignment=req.alignment)
        plan = FaerSparseSymbolic(a; kind=:cholesky, ordering, scratch=s)
        ordering isa Vector && @test Vector(plan.p) == ordering
        close(s)
    end
    close(a)
end

@testset "LU/QR symbolic COLAMD scratch" begin
    rng = MersenneTwister(1180)
    for (m, n) in ((9, 9), (13, 7), (7, 13), (0, 0)),
        kind in (:lu, :qr), algorithm in (:auto, :simplicial, :supernodal),
        format in (:csc, :csr)

        kind === :lu && m != n && continue
        a = sprand(rng, m, n, 0.35)
        if m == n
            a += sparse(3.0I, m, n)
        end
        source = FaerSparseMatrix(a; format)
        pattern = FaerSparsity(source)
        ordering = algorithm === :simplicial ? :colamd : :default
        for input in (source, pattern), aggressive in (true, false)
            opts = (; kind, algorithm, ordering, aggressive, dense_row=0.4, dense_col=0.6)
            req = scratch_requirements(FaerSparseSymbolic, input; opts...)
            # Square/tall auto/supernodal COLAMD runs inside faer's allocating symbolic API.
            internal = algorithm !== :simplicial && m >= n
            colamd_req = scratch_requirements(
                colamd, input; dense_row=0.4, dense_col=0.6, aggressive
            )
            if kind === :qr && algorithm === :simplicial && m >= n
                # COLAMD, then the simplicial symbolic kernels, share the buffer.
                @test req.bytes >= colamd_req.bytes && req.bytes > 0
                @test req.alignment >= colamd_req.alignment
            else
                @test req == (internal ? (bytes=0, alignment=1) : colamd_req)
            end
            s = FaerScratch(FaerSparseSymbolic, input; opts...)
            plan = FaerSparseSymbolic(input; scratch=s, opts...)
            ref = FaerSparseSymbolic(input; opts...)
            if input === source && m > 0
                F = factorize(plan, source)
                G = factorize(ref, source)
                if kind === :lu
                    @test Vector(F.q) == Vector(G.q)
                    @test Vector(F.p) == Vector(G.p)
                else
                    @test Vector(F.p) == Vector(G.p)
                    @test Matrix(F.R) ≈ Matrix(G.R)
                end
            end
            if req.bytes > 0
                small = FaerScratch(req.bytes - 1; alignment=req.alignment)
                @test_throws ArgumentError FaerSparseSymbolic(input; scratch=small, opts...)
                close(small)
            end
            close(s)
            @test_throws ArgumentError FaerSparseSymbolic(input; scratch=s, opts...)
        end
        close(source)
        close(pattern)
    end
end

@testset "Explicit simplicial QR symbolic analysis in caller scratch" begin
    rng = MersenneTwister(1160)
    for (m, n) in ((40, 20), (25, 25), (1, 1)), format in (:csc, :csr)
        a = sprand(rng, m, n, 0.2) + sparse(1:n, 1:n, 2.0, m, n)
        source = FaerSparseMatrix(a; format)
        b = collect(1.0:m)
        for ordering in (:default, :identity, reverse(collect(1:n)), :colamd)
            opts = (; kind=:qr, algorithm=:simplicial, ordering)
            req = scratch_requirements(FaerSparseSymbolic, source; opts...)
            @test req.bytes > 0
            s = FaerScratch(FaerSparseSymbolic, source; opts...)
            plan = FaerSparseSymbolic(source; scratch=s, opts...)
            ref = FaerSparseSymbolic(source; opts...)
            F, G = factorize(plan, source), factorize(ref, source)
            @test Vector(F.p) == Vector(G.p)
            @test Matrix(F.R) == Matrix(G.R)
            @test F \ b == G \ b
            small = FaerScratch(req.bytes - 1; alignment=req.alignment)
            @test_throws ArgumentError FaerSparseSymbolic(source; scratch=small, opts...)
            foreach(close, (small, s, plan, ref, F, G))
        end
        close(source)
    end
end

@testset "Symbolic scratch query validation" begin
    a = FaerSparseMatrix(sparse([4.0 1; 1 3]))
    @test_throws ArgumentError scratch_requirements(FaerSparseSymbolic, a; kind=:bogus)
    @test_throws ArgumentError scratch_requirements(
        FaerSparseSymbolic, a; kind=:cholesky, dense=NaN
    )
    @test_throws ArgumentError scratch_requirements(FaerSparseSymbolic, a; ordering=:amd)
    @test_throws DimensionMismatch scratch_requirements(
        FaerSparseSymbolic, FaerSparseMatrix(sprand(3, 2, 0.5)); kind=:cholesky
    )
    p = FaerSparsity(a)
    close(a)
    @test_throws ArgumentError scratch_requirements(FaerSparseSymbolic, a; kind=:cholesky)
    s = FaerScratch(FaerSparseSymbolic, p; kind=:cholesky)
    tasks = [Threads.@spawn(FaerSparseSymbolic(p; kind=:cholesky, scratch=s)) for _ in 1:4]
    expected = Vector(FaerSparseSymbolic(p; kind=:cholesky).p)
    for t in tasks
        @test Vector(fetch(t).p) == expected
    end
    close(s)
    close(p)
end
