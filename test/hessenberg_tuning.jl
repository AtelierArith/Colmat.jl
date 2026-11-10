using Colmat, Test, LinearAlgebra, Random

@testset "Hessenberg reduction controls and matching scratch" begin
    rng = MersenneTwister(8500)
    team = FaerTeam(2)
    for T in (Float32, Float64, ComplexF32, ComplexF64), n in (0, 1, 2, 17, 65)
        a = randn(rng, T, n, n)
        source = FaerMatrix(a)
        tol = 1000max(n, 1) * eps(real(T))
        for compact in (false, true),
            resource in (nothing, team),
            policy in (
                FaerParThreshold(),
                FaerParThreshold(1),
                FaerParThreshold(typemax(UInt) - 1),
                FaerParThreshold(:never),
            ),
            blocking in (0, 256 * 256, typemax(UInt))

            options = (;
                compact, team=resource, par_threshold=policy, blocking_threshold=blocking
            )
            req = scratch_requirements(hessenberg, source; options...)
            @test scratch_requirements(hessenberg!, source; options...) == req
            if policy.value == typemax(UInt)
                @test req ==
                    scratch_requirements(hessenberg, source; options..., team=nothing)
            end
            work = FaerScratch(hessenberg, source; options...)
            @test scratch_capacity(work).bytes >= req.bytes
            f = hessenberg(source; options..., scratch=work)
            @test (f isa FaerCompactHessenberg) == compact
            @test Matrix(source) == a
            @test istriu(Matrix(f.H), -1)
            @test Matrix(f) ≈ a atol = tol rtol = tol
            q = Matrix(f.Q)
            @test q'q ≈ Matrix{T}(I, n, n) atol = tol rtol = tol
            close(source)
            @test Matrix(f) ≈ a atol = tol rtol = tol
            source = FaerMatrix(a)
            close(f)
            close(work)
        end
        close(source)
    end
    close(team)
end

@testset "Hessenberg controls: staged output, views and resources" begin
    rng = MersenneTwister(8501)
    team = FaerTeam(2)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        compact in (false, true),
        blocking in (0, typemax(UInt))

        a = randn(rng, T, 17, 17)
        source = FaerMatrix(a)
        options = (;
            compact, team, par_threshold=FaerParThreshold(1), blocking_threshold=blocking
        )
        work = FaerScratch(0)
        req = scratch_requirements(hessenberg, source; options...)
        @test req.bytes > 0
        for op in (hessenberg, hessenberg!)
            @test_throws ArgumentError op(source; options..., scratch=work)
            @test Matrix(source) == a
        end
        reserve!(work, hessenberg!, source; options...)
        saved = view(source, :, :)
        f = hessenberg!(source; options..., scratch=work)
        @test f.H === source
        @test Matrix(saved) == Matrix(f.H)
        @test Matrix(f) ≈ a rtol = 0.002 atol = 0.002
        close(f.Q)
        close(source)
        source = FaerMatrix(a)
        old = source.dims
        source.dims = (16, 16)
        @test_throws Exception hessenberg(source; options..., scratch=work)
        @test_throws Exception scratch_requirements(hessenberg, source; options...)
        source.dims = old
        @test Matrix(source) == a
        for bad in (-1, big(typemax(UInt)) + 1)
            @test_throws ArgumentError hessenberg(
                source; options..., blocking_threshold=bad
            )
            @test_throws ArgumentError scratch_requirements(
                hessenberg, source; options..., blocking_threshold=bad
            )
        end
        @test_throws ArgumentError hessenberg(source; options..., par_threshold=:never)
        source[1, 1] = T(NaN)
        invalid = Matrix(source)
        @test_throws Colmat.RustCall.RustError hessenberg!(source; options..., scratch=work)
        @test isequal(Matrix(source), invalid)
        close(source)
        close(work)
        @test_throws ArgumentError hessenberg(FaerMatrix(a); options..., scratch=work)
    end
    source = FaerMatrix(randn(rng, ComplexF64, 17, 17))
    for compact in (false, true),
        view_a in (view(source, 17:-1:1, 17:-1:1), view(source, 1:2:17, 1:2:17))

        options = (;
            compact, team, par_threshold=FaerParThreshold(:never), blocking_threshold=0
        )
        work = FaerScratch(hessenberg, view_a; options...)
        result = fetch(Threads.@spawn hessenberg(view_a; options..., scratch=work))
        GC.gc()
        @test Matrix(result) ≈ Matrix(view_a)
        # Saved transformations remain valid with a different reduction policy.
        spec = schur(result; team)
        @test Matrix(spec.Z) * Matrix(spec.T) * Matrix(spec.Z)' ≈ Matrix(view_a)
        close(result)
        close(spec.T)
        close(spec.Z)
        close(work)
    end
    close(team)
    @test_throws ArgumentError hessenberg(
        source; team, par_threshold=FaerParThreshold(:never)
    )
    @test_throws ArgumentError scratch_requirements(
        hessenberg, source; team, par_threshold=FaerParThreshold(:never)
    )
    close(source)
end

@testset "Hessenberg tuned scaling and checked owner arguments" begin
    team = FaerTeam(2)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        compact in (false, true),
        blocking in (0, typemax(UInt))

        R = real(T)
        base = T[4 1 2 0; 2 3 0 1; 1 0 5 0; 0 2 0 6]
        T <: Complex && (base[1, 3] += T(im))
        for exponent in (T <: Union{Float32,ComplexF32} ? (-100, 100) : (-900, 900))
            scale = ldexp(one(R), exponent)
            source = FaerMatrix(base * scale)
            options = (;
                compact,
                team,
                par_threshold=FaerParThreshold(1),
                blocking_threshold=blocking,
            )
            work = FaerScratch(hessenberg, source; options...)
            f = hessenberg(source; options..., scratch=work)
            @test Matrix(f) ./ scale ≈ base rtol = 1000eps(R) atol = 1000eps(R)
            @test all(isfinite, Matrix(f.H))
            @test Matrix(source) == base * scale
            close(f)
            close(work)
            close(source)
        end
        source = FaerMatrix(base)
        work = FaerScratch(hessenberg, source; compact, team, blocking_threshold=blocking)
        fn = if compact
            Colmat.RustCrate.compact_hessenberg_tuned
        else
            Colmat.RustCrate.hessenberg_tuned_workspace
        end
        args = compact ? (source, source, work) : (source, source, source, work)
        # A source cannot also be a mutable output. RustCall checks the handles
        # before native entry, including the optional team borrow.
        @test_throws Exception Colmat._native_call(
            fn, args..., UInt(4), Colmat._tag(T), UInt(0), UInt(blocking), team
        )
        @test Matrix(source) == base
        close(work)
        close(source)
    end
    close(team)
end

@testset "Hessenberg scratch query uses checked mutable slices" begin
    source = FaerMatrix([4.0 1; 2 3])
    team = FaerTeam(2)
    for compact in (false, true), resource in (nothing, team)
        fn = if compact
            Colmat.RustCrate.compact_hessenberg_tuned_requirements
        else
            Colmat.RustCrate.hessenberg_tuned_requirements
        end
        call =
            (out; tag=Int32(1), n=UInt(2)) ->
                Colmat._native_call(fn, n, tag, UInt(0), UInt(0), resource, out)
        backing = fill(UInt(42), 4)
        call(view(backing, 2:3))
        req = scratch_requirements(
            hessenberg, source; compact, team=resource, blocking_threshold=0
        )
        @test backing[2:3] == UInt[req.bytes, req.alignment]
        @test backing[[1, 4]] == UInt[42, 42]
        for len in (0, 1, 3)
            out = fill(UInt(43), len)
            @test_throws Colmat.RustCall.RustError call(out)
            @test out == fill(UInt(43), len)
        end
        out = fill(UInt(44), 2)
        @test_throws Colmat.RustCall.RustError call(out; tag=Int32(9))
        @test out == UInt[44, 44]
        @test_throws Colmat.RustCall.RustError call(out; n=typemax(UInt))
        @test out == UInt[44, 44]
        @test_throws ArgumentError call(zeros(Float64, 2))
    end
    close(team)
    close(source)
end
