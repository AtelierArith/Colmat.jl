using Test, Colmat, LinearAlgebra, Random
sro_snapshot(f) = (Matrix(f.T), Matrix(f.Z), copy(f.values))
# Match eigenvalue multisets without ordering numerically equal real parts.
function sro_same_spectrum(a, b; rtol, atol)
    length(a) == length(b) || return false
    remaining = collect(b)
    for value in a
        j = argmin(abs.(remaining .- value))
        isapprox(value, remaining[j]; rtol, atol) || return false
        deleteat!(remaining, j)
    end
    return true
end
function sro_factor(T, n, rng)
    R = typeof(real(zero(T)))
    t = triu(T(0.01) * randn(rng, T, n, n))
    for i in 1:n
        t[i, i] = T(i)
        T <: Complex && (t[i, i] += T(0.1im * i))
    end
    q = Matrix(qr(randn(rng, T, n, n)).Q)
    f = LinearAlgebra.Schur(FaerMatrix(t), FaerMatrix(q), Complex{R}.(diag(t)))
    return f, q * t * q'
end
function sro_check(g, a, expected, selected)
    n = size(a, 1)
    R = typeof(real(zero(eltype(a))))
    h, q = Matrix(g.T), Matrix(g.Z)
    @test norm(q * h * q' - a) / (max(n, 1) * eps(R) * max(norm(a), one(R))) < 20
    @test norm(q' * q - I) / (max(n, 1) * eps(R)) < 20
    @test h == triu(h, eltype(a) <: Real ? -1 : 0)
    @test sro_same_spectrum(
        g.values, expected; rtol=100 * max(n, 1) * eps(R), atol=100 * max(n, 1) * eps(R)
    )
    k = count(selected)
    @test sro_same_spectrum(
        g.values[1:k],
        expected[selected];
        rtol=100 * max(n, 1) * eps(R),
        atol=100 * max(n, 1) * eps(R),
    )
end
@testset "Saved Schur reorder scratch and scoped team" begin
    rng = MersenneTwister(66104)
    for T in (Float32, Float64, ComplexF32, ComplexF64), width in (1, 2)
        team = FaerTeam(width)
        for n in (0, 1, 6, 17, 65)
            f, a = sro_factor(T, n, rng)
            old = sro_snapshot(f)
            req = scratch_requirements(ordschur, f; team)
            @test req == scratch_requirements(ordschur!, f; team)
            @test req == scratch_requirements(ordschur, f)
            work = FaerScratch(ordschur, f; team)
            cap = scratch_capacity(work)
            for select in (falses(n), trues(n), isodd.(1:n), iseven.(1:n))
                original = copy(select)
                g = ordschur(f, select; scratch=work, team)
                sro_check(g, a, old[3], select)
                @test sro_snapshot(f) == old
                @test select == original
                @test scratch_capacity(work) == cap
                x = copy(f)
                tx, qx, vx = x.T, x.Z, x.values
                tv, qv = view(x.T, :, :), view(x.Z, :, :)
                @test ordschur!(x, Vector{Bool}(select); scratch=work, team) === x
                @test x.T === tx && x.Z === qx && x.values === vx
                @test Matrix(tv) == Matrix(x.T) && Matrix(qv) == Matrix(x.Z)
                sro_check(x, a, old[3], select)
                close(x)
                close(g)
            end
            close(work)
            close(f)
        end
        close(team)
    end
end
@testset "Conjugate blocks, independent spectrum, and reusable factor owners" begin
    rng = MersenneTwister(66105)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        R = typeof(real(zero(T)))
        expected = Complex{R}[-3 + 2im, -3 - 2im, 1, 2, 4 + im, 4 - im]
        t = triu(T(0.01) * randn(rng, T, 6, 6))
        if T <: Real
            for (i, re, im) in ((1, -3, 2), (5, 4, 1))
                t[i, i] = t[i + 1, i + 1] = T(re)
                t[i, i + 1] = T(im)
                t[i + 1, i] = T(-im)
            end
            t[3, 3] = 1
            t[4, 4] = 2
        else
            for i in 1:6
                t[i, i] = T(expected[i])
            end
        end
        q = Matrix(qr(randn(rng, T, 6, 6)).Q)
        a = q * t * q'
        @test sro_same_spectrum(eigvals(a), expected; rtol=100 * eps(R), atol=100 * eps(R))
        f = LinearAlgebra.Schur(FaerMatrix(t), FaerMatrix(q), copy(expected))
        old = sro_snapshot(f)
        team = FaerTeam(2)
        work = FaerScratch(ordschur, f; team)
        for select in
            (Bool[0, 0, 1, 0, 1, 1], Bool[0, 0, 0, 1, 0, 0], Bool[1, 1, 1, 0, 0, 0])
            g = ordschur(f, select; scratch=work, team)
            sro_check(g, a, expected, select)
            @test sro_snapshot(f) == old
            close(g)
        end
        if T <: Real
            for op in (ordschur, ordschur!)
                @test_throws ArgumentError op(f, Bool[1, 0, 0, 0, 0, 0]; scratch=work, team)
                @test sro_snapshot(f) == old
            end
        end
        select = Bool[0, 0, 1, 0, 1, 1]
        for task in [Threads.@spawn ordschur(f, select; scratch=work, team) for _ in 1:3]
            g = fetch(task)
            sro_check(g, a, expected, select)
            close(g)
        end
        g = ordschur(f, select; scratch=work, team)
        close(f)
        close(work)
        close(team)
        GC.gc()
        sro_check(g, a, expected, select)
        close(g)
    end
    @test isempty(detect_ambiguities(Colmat, Base, LinearAlgebra; recursive=true))
end
@testset "Saved Schur capacity and validation failures preserve factors" begin
    rng = MersenneTwister(66106)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        f, a = sro_factor(T, 6, rng)
        old = sro_snapshot(f)
        team = FaerTeam(2)
        req = scratch_requirements(ordschur, f; team)
        work = FaerScratch(req.bytes - 1; alignment=req.alignment)
        for op in (ordschur, ordschur!)
            @test_throws ArgumentError op(f, Bool[0, 0, 0, 1, 1, 1]; scratch=work, team)
            @test sro_snapshot(f) == old
            @test_throws ArgumentError op(f, trues(6); scratch=1)
            @test_throws ArgumentError op(f, trues(6); team=1)
        end
        reserve!(work, ordschur!, f; team)
        close(ordschur(f, trues(6); scratch=work, team))
        if req.alignment > 1
            unaligned = FaerScratch(req.bytes; alignment=1)
            @test_throws ArgumentError ordschur(f, trues(6); scratch=unaligned, team)
            close(unaligned)
        end
        for op in (ordschur, ordschur!)
            @test_throws DimensionMismatch op(f, trues(5); scratch=work, team)
            @test sro_snapshot(f) == old
        end
        f.T[6, 1] = one(T)
        broken = sro_snapshot(f)
        for op in (ordschur, ordschur!)
            @test_throws Colmat.RustCall.RustError op(f, trues(6); scratch=work, team)
            @test sro_snapshot(f) == broken
        end
        f.T[6, 1] = zero(T)
        f.T[1, 1] = T(NaN)
        nonfinite = sro_snapshot(f)
        for op in (ordschur, ordschur!)
            @test_throws Colmat.RustCall.RustError op(f, trues(6); scratch=work, team)
            @test isequal(sro_snapshot(f), nonfinite)
        end
        close(work)
        @test_throws ArgumentError ordschur(f, trues(6); scratch=work)
        close(team)
        @test_throws ArgumentError scratch_requirements(ordschur, f; team)
        @test_throws ArgumentError ordschur(f, trues(6); team)
        close(f)
    end
    f = LinearAlgebra.Schur(
        FaerMatrix([1.5 0; 0 2.5]), FaerMatrix(Matrix{Float64}(I, 2, 2)), [1, 2]
    )
    old = sro_snapshot(f)
    team = FaerTeam(2)
    work = FaerScratch(ordschur!, f; team)
    @test_throws InexactError ordschur!(f, Bool[false, true]; scratch=work, team)
    @test sro_snapshot(f) == old
    close(f)
    close(work)
    close(team)
end
@testset "Saved Schur metadata precedes coefficient and selection access" begin
    rng = MersenneTwister(66107)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        f, a = sro_factor(T, 3, rng)
        for m in (f.T, f.Z)
            m.dims = (2, 2)
            for op in (ordschur, ordschur!)
                @test_throws r"current native dimensions" scratch_requirements(op, f)
                @test_throws r"current native dimensions" op(f, trues(3))
            end
            m.dims = (3, 3)
        end
        resize!(f.values, 2)
        @test_throws DimensionMismatch scratch_requirements(ordschur, f)
        @test_throws DimensionMismatch ordschur(f, trues(3))
        close(f)
    end
    for T in (Float32, ComplexF32, ComplexF64)
        f, a = sro_factor(T, 3, rng)
        U = T === Float32 ? ComplexF64 : Float32
        replacement = Matrix{U}(I, 3, 3)
        for m in (f.T, f.Z)
            original = Matrix(m)
            GC.@preserve replacement Colmat._native_call(
                Colmat.RustCrate.load,
                m,
                Ptr{UInt8}(pointer(replacement)),
                UInt(3),
                UInt(3),
                Colmat._tag(U),
            )
            for op in (ordschur, ordschur!)
                @test_throws r"requested scalar type" scratch_requirements(op, f)
                @test_throws r"requested scalar type" op(f, trues(3))
            end
            GC.@preserve original Colmat._native_call(
                Colmat.RustCrate.load,
                m,
                Ptr{UInt8}(pointer(original)),
                UInt(3),
                UInt(3),
                Colmat._tag(T),
            )
        end
        close(f)
    end
    f, a = sro_factor(Float64, 3, rng)
    close(f.Z)
    @test_throws ArgumentError scratch_requirements(ordschur, f)
    @test_throws ArgumentError ordschur(f, trues(3))
    close(f)
end
