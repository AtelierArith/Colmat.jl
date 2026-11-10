using Test, Colmat, LinearAlgebra, Random
sve_snapshot(f) = (Matrix(f.T), Matrix(f.Z), copy(f.values))
function sve_factor(T, n, rng; scale=one(typeof(real(zero(T)))))
    R = typeof(real(zero(T)))
    t = triu(T(0.01) * randn(rng, T, n, n))
    for i in 1:n
        t[i, i] = T(i)
        T <: Complex && (t[i, i] += T(0.1im * i))
    end
    values = Complex{R}.(diag(t))
    if T <: Real && n >= 2
        t[1, 1] = t[2, 2] = T(2)
        t[1, 2] = one(T)
        t[2, 1] = -one(T)
        values[1] = Complex{R}(2, 1)
        values[2] = Complex{R}(2, -1)
    end
    q = Matrix(qr(randn(rng, T, n, n)).Q)
    f = LinearAlgebra.Schur(FaerMatrix(t .* scale), FaerMatrix(q), values .* scale)
    return f, q * t * q', values
end
function sve_check(v, a, values)
    x = Matrix(v)
    n = size(a, 1)
    R = typeof(real(zero(eltype(a))))
    @test eltype(v) == Complex{R}
    @test size(v) == (n, n)
    @test all(isfinite, x)
    @test norm(a * x - x * Diagonal(values)) / (max(n, 1) * eps(R) * max(norm(a), one(R))) <
        40
    for j in 1:n
        @test abs(norm(x[:, j]) - 1) < 100 * eps(R)
    end
end
@testset "Saved Schur eigenvectors declared scratch and actual team" begin
    rng = MersenneTwister(67101)
    for T in (Float32, Float64, ComplexF32, ComplexF64), width in (1, 2)
        team = FaerTeam(width)
        for n in (0, 1, 6, 17, 65, 257)
            f, a, values = sve_factor(T, n, rng)
            old = sve_snapshot(f)
            req = scratch_requirements(eigvecs, f; team)
            @test req == scratch_requirements(eigen, f; team)
            @test req == scratch_requirements(eigvecs, f)
            work = FaerScratch(eigvecs, f; team)
            cap = scratch_capacity(work)
            v = eigvecs(f; scratch=work, team)
            sve_check(v, a, values)
            @test sve_snapshot(f) == old
            @test scratch_capacity(work) == cap
            reserve!(work, eigen, f; team)
            e = eigen(f; scratch=work, team)
            @test e.values == f.values && e.values !== f.values
            sve_check(e.vectors, a, values)
            @test sve_snapshot(f) == old
            close(v)
            close(e.vectors)
            close(work)
            close(f)
        end
        close(team)
    end
end
@testset "Saved Schur extreme scales, reuse, lifetime and tasks" begin
    rng = MersenneTwister(67102)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        R = typeof(real(zero(T)))
        team = FaerTeam(2)
        for scale in (
            one(R), R === Float32 ? R(1e30) : R(1e300), R === Float32 ? R(1e-30) : R(1e-300)
        )
            f, a, values = sve_factor(T, 6, rng; scale)
            old = sve_snapshot(f)
            work = FaerScratch(eigen, f; team)
            for task in [Threads.@spawn eigvecs(f; scratch=work, team) for _ in 1:3]
                v = fetch(task)
                sve_check(v, a, values)
                close(v)
            end
            @test sve_snapshot(f) == old
            e = eigen(f; scratch=work, team)
            @test e.values == f.values
            close(f)
            close(work)
            GC.gc()
            sve_check(e.vectors, a, values)
            close(e.vectors)
        end
        close(team)
    end
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        for t in (zeros(T, 3, 3), Matrix{T}(I, 3, 3), T[1 1 0; 0 1 1; 0 0 1])
            f = schur(FaerMatrix(t))
            w = FaerScratch(eigvecs, f)
            v = eigvecs(f; scratch=w)
            sve_check(v, t, f.values)
            close(v)
            close(w)
            close(f)
        end
    end
    @test isempty(detect_ambiguities(Colmat, Base, LinearAlgebra; recursive=true))
end
@testset "Saved Schur eigenvector resource and native failure contracts" begin
    rng = MersenneTwister(67103)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        f, a, values = sve_factor(T, 6, rng)
        old = sve_snapshot(f)
        team = FaerTeam(2)
        req = scratch_requirements(eigvecs, f; team)
        w = FaerScratch(req.bytes - 1; alignment=req.alignment)
        for fn in (eigvecs, eigen)
            @test_throws ArgumentError fn(f; scratch=w, team)
            @test sve_snapshot(f) == old
            @test_throws ArgumentError fn(f; scratch=1)
            @test_throws ArgumentError fn(f; team=1)
        end
        reserve!(w, eigvecs, f; team)
        v = eigvecs(f; scratch=w, team)
        close(v)
        if req.alignment > 1
            bad = FaerScratch(req.bytes; alignment=1)
            @test_throws ArgumentError eigvecs(f; scratch=bad, team)
            close(bad)
        end
        f.T[6, 1] = one(T)
        invalid = sve_snapshot(f)
        for fn in (eigvecs, eigen)
            @test_throws Colmat.RustCall.RustError fn(f; scratch=w, team)
            @test sve_snapshot(f) == invalid
        end
        f.T[6, 1] = zero(T)
        f.Z[1, 1] = T(NaN)
        invalid = sve_snapshot(f)
        for fn in (eigvecs, eigen)
            @test_throws Colmat.RustCall.RustError fn(f; scratch=w, team)
            @test isequal(sve_snapshot(f), invalid)
        end
        close(w)
        close(team)
        @test_throws ArgumentError eigvecs(f; scratch=w)
        @test_throws ArgumentError scratch_requirements(eigen, f; team)
        @test_throws ArgumentError eigvecs(f; team)
        close(f)
    end
end
@testset "Saved Schur eigenvector current metadata and closed owners" begin
    rng = MersenneTwister(67104)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        f, a, values = sve_factor(T, 3, rng)
        for m in (f.T, f.Z)
            m.dims = (2, 2)
            for fn in (eigvecs, eigen)
                @test_throws r"current native dimensions" scratch_requirements(fn, f)
                @test_throws r"current native dimensions" fn(f)
            end
            m.dims = (3, 3)
        end
        resize!(f.values, 2)
        @test_throws DimensionMismatch scratch_requirements(eigvecs, f)
        @test_throws DimensionMismatch eigen(f)
        close(f)
    end
    for T in (Float32, ComplexF32, ComplexF64)
        f, a, values = sve_factor(T, 3, rng)
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
            for fn in (eigvecs, eigen)
                @test_throws r"requested scalar type" scratch_requirements(fn, f)
                @test_throws r"requested scalar type" fn(f)
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
    f, a, values = sve_factor(Float64, 3, rng)
    close(f.Z)
    @test_throws ArgumentError eigvecs(f)
    @test_throws ArgumentError scratch_requirements(eigen, f)
    close(f)
end
