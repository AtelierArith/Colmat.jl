using Colmat, Test, LinearAlgebra, SparseArrays, Random

@testset "AMD/COLAMD exact external workspace and controls" begin
    rng = MersenneTwister(22771)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        format in (:csc, :csr),
        (m, n) in ((0, 0), (0, 5), (5, 0), (1, 1), (9, 9), (13, 7), (7, 13))

        a = sprand(rng, T, m, n, 0.3)
        source = FaerSparseMatrix(a; format)
        pattern = FaerSparsity(source)
        for input in (source, pattern, a), fun in (amd, colamd)
            if fun === amd && m != n
                @test_throws DimensionMismatch scratch_requirements(fun, input)
                @test_throws DimensionMismatch FaerScratch(fun, input)
                continue
            end
            req = scratch_requirements(fun, input)
            s = FaerScratch(req.bytes; alignment=req.alignment)
            @test scratch_capacity(s) == req
            before = scratch_capacity(s)
            for aggressive in (true, false), threshold in (-1.0, 0.5, 10.0)
                opts = if fun === amd
                    (; dense=threshold, aggressive)
                else
                    (; dense_row=threshold, dense_col=threshold, aggressive)
                end
                @test req == scratch_requirements(fun, input; opts...)
                p = fun(input; scratch=s, opts...)
                ref = fun(input; opts...)
                @test Vector(p) == Vector(ref)
                @test isperm(Vector(p))
                @test scratch_capacity(s) == before
                if fun === amd
                    stats = amd(input; scratch=s, statistics=true, opts...)
                    reference = amd(input; statistics=true, opts...)
                    @test Vector(stats.permutation) == Vector(reference.permutation)
                    @test stats.n_div == reference.n_div
                    @test stats.n_mult_subs_ldl == reference.n_mult_subs_ldl
                    @test stats.n_mult_subs_lu == reference.n_mult_subs_lu
                    close(stats.permutation)
                    close(reference.permutation)
                end
                close(p)
                close(ref)
            end
            if req.bytes > 0
                small = FaerScratch(req.bytes - 1; alignment=req.alignment)
                @test_throws ArgumentError fun(input; scratch=small)
                @test reserve!(small, fun, input) === small
                @test isperm(Vector(fun(input; scratch=small)))
                close(small)
            end
            if req.alignment > 1
                weak = FaerScratch(req.bytes; alignment=1)
                @test_throws ArgumentError fun(input; scratch=weak)
                reserve!(weak, fun, input)
                @test isperm(Vector(fun(input; scratch=weak)))
                close(weak)
            end
            close(s)
            @test_throws ArgumentError fun(input; scratch=s)
            @test_throws ArgumentError fun(input; scratch=ones(3))
        end
        @test sparse(source) == a
        @test Matrix(pattern) == Matrix(a .!= 0)
        close(source)
        close(pattern)
    end
end

@testset "Ordering wrappers, stored zeros, ownership, and Tasks" begin
    a = sparse([1, 2, 3, 4, 1], [1, 1, 2, 4, 4], [0.0, NaN, Inf, 3, 2], 4, 4)
    source = FaerSparseMatrix(a; format=:csr)
    for input in (
            source,
            transpose(source),
            adjoint(source),
            Hermitian(source, :L),
            Hermitian(source, :U),
            Symmetric(source, :L),
        ),
        fun in (amd, colamd)

        s = FaerScratch(fun, input)
        expected = Vector(fun(input))
        for task in [Threads.@spawn(fun(input; scratch=s)) for _ in 1:4]
            p = fetch(task)
            @test Vector(p) == expected
            close(p)
        end
        @test nnz(source) == 5
        p = fun(input; scratch=s)
        close(s)
        @test isperm(Vector(p))
        close(p)
    end
    pat = FaerSparsity(source)
    s = FaerScratch(amd, pat)
    p = amd(pat; scratch=s)
    close(source)
    close(pat)
    @test isperm(Vector(p))
    for fun in (amd, colamd)
        @test_throws ArgumentError scratch_requirements(fun, source)
        @test_throws ArgumentError scratch_requirements(fun, pat)
        @test_throws ArgumentError fun(pat; scratch=s)
    end
    close(p)
    close(s)
end

@testset "Ordering metadata and option validation" begin
    a = FaerSparseMatrix(sparse([4.0 1; 2 3]))
    p = FaerSparsity(a)
    work = FaerScratch(amd, p)
    p.dims = (3, 3)
    for fun in (amd, colamd)
        @test_throws Colmat.RustCall.RustError scratch_requirements(fun, p)
        @test_throws Colmat.RustCall.RustError fun(p; scratch=work)
    end
    p.dims = (2, 2)
    for value in (Inf, -Inf, NaN)
        @test_throws ArgumentError scratch_requirements(amd, a; dense=value)
        @test_throws ArgumentError scratch_requirements(colamd, a; dense_row=value)
        @test_throws ArgumentError scratch_requirements(colamd, a; dense_col=value)
    end
    @test isperm(Vector(amd(p; scratch=work)))
    close(a)
    close(p)
    close(work)
end
