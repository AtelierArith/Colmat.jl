using Colmat, LinearAlgebra, Random, Test
@testset "Compact bidiagonal thin/full bases and saved SVD" begin
    rng = MersenneTwister(8024)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        (m, n) in ((0, 0), (0, 3), (3, 0), (1, 1), (4, 3), (3, 4), (5, 5)),
        full in (false, true)

        a = randn(rng, T, m, n)
        k = min(m, n)
        f = bidiagonalize(a; compact=true, full)
        @test f.U isa FaerBidiagonalBasis
        @test size(f.U) == (m, full ? m : k)
        @test size(f.V) == (n, full ? n : k)
        @test Matrix(f) ≈ a atol = 500eps(real(T))
        for q in (f.U, f.V), h in (q, transpose(q), adjoint(q), conj(q))
            d = Matrix(h)
            rhs = randn(rng, T, size(h, 2), 2)
            lhs = randn(rng, T, 2, size(h, 1))
            @test Matrix(h * rhs) ≈ d * rhs atol = 500eps(real(T))
            @test Matrix(lhs * h) ≈ lhs * d atol = 500eps(real(T))
            if h === q
                v = randn(rng, T, size(q, 2))
                owned = FaerVector(v)
                @test q * v ≈ d * v atol = 500eps(real(T))
                @test Vector(q * owned) ≈ d * v atol = 500eps(real(T))
                b = randn(rng, T, size(q, 1))
                @test q \ b ≈ d' * b atol = 500eps(real(T))
                row = FaerRow(b)
                @test Matrix(row * q) ≈ reshape(b, 1, :) * d atol = 500eps(real(T))
                close(owned)
                close(row)
            end
        end
        s = svd(f; full)
        @test Matrix(s.U[:, 1:k]) * Diagonal(s.S) * Matrix(s.Vt[1:k, :]) ≈ a atol =
            500eps(real(T))
        @test Colmat._handle(f.U) === Colmat._handle(f.V)
        copied = copy(f)
        @test Colmat._handle(copied.U) === Colmat._handle(copied.V)
        @test Colmat._handle(copied.U) !== Colmat._handle(f.U)
        c = copy(f.U)
        held = f.V'
        close(f)
        @test isopen(c)
        @test isopen(held)
        close(c)
        close(held)
        close(copied)
    end
end

@testset "Compact bidiagonal resources and rectangular mul" begin
    rng = MersenneTwister(8025)
    for T in (Float32, Float64, ComplexF32, ComplexF64), (m, n) in ((7, 4), (4, 7), (4, 4))
        a = randn(rng, T, m, n)
        k = min(m, n)
        team = FaerTeam(2)
        work = FaerScratch(bidiagonalize, a; compact=true, team, blocksize=2)
        f = bidiagonalize(a; compact=true, team, scratch=work, blocksize=2)
        for q in (f.U, f.V), h in (q, q', transpose(q), conj(q))
            rhs = randn(rng, T, size(h, 2), 3)
            dest = zeros(T, size(h, 1), 3)
            reserve!(work, mul!, dest, h, rhs; team)
            cap = scratch_capacity(work)
            mul!(dest, h, rhs; scratch=work, team)
            @test dest ≈ Matrix(h) * rhs atol = 1000eps(real(T))
            @test scratch_capacity(work) == cap
            rhsr = randn(rng, T, 3, size(h, 1))
            destr = zeros(T, 3, size(h, 2))
            reserve!(work, mul!, destr, rhsr, h; team)
            mul!(destr, rhsr, h, T(2), T(0); scratch=work, team)
            @test destr ≈ T(2) * rhsr * Matrix(h) atol = 1000eps(real(T))
        end
        reserve!(work, svd, f; team)
        s = svd(f; scratch=work, team)
        @test Matrix(s.U) * Diagonal(s.S) * Matrix(s.Vt) ≈ a atol = 1000eps(real(T))
        @test svdvals(f; scratch=work, team) ≈ svdvals(a) atol = 1000eps(real(T))
        thin = m > n ? f.U : f.V
        if size(thin, 1) != size(thin, 2)
            rhs = ones(T, size(thin, 2), 2)
            @test_throws DimensionMismatch lmul!(thin, rhs)
            @test rhs == ones(T, size(thin, 2), 2)
            @test_throws ArgumentError svd(f; full=true)
        end
        q = f.U
        old = q.m
        q.m += 1
        @test_throws Exception q.basis
        q.m = old
        basis = q.basis
        factor = q.factor
        fill!(basis, zero(T))
        fill!(factor, zero(T))
        @test Matrix(f) ≈ a atol = 1000eps(real(T))
        close(team)
        @test_throws ArgumentError bidiagonalize(
            a; compact=true, team, par_threshold=FaerParThreshold(:never)
        )
        close(work)
        close(f)
    end
end

@testset "Compact bidiagonal alias, promotion and extreme scaling" begin
    rng = MersenneTwister(8026)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        R = real(T)
        tol = 1000eps(R)
        a = randn(rng, T, 6, 4)
        for scale in
            (floatmin(R) / R(16), T <: Union{Float32,ComplexF32} ? R(1e30) : R(1e300))
            f = bidiagonalize(a .* scale; compact=true, blocksize=2)
            restored = Matrix(f)
            scaled = complex.(real.(restored) ./ scale, imag.(restored) ./ scale)
            @test scaled ≈ a atol = tol rtol = tol
            close(f)
        end
        f = bidiagonalize(a; compact=true, full=true)
        q = f.U
        d = Matrix(q)
        rhs = randn(rng, ComplexF64, 6, 3)
        @test Matrix(q * rhs) ≈ ComplexF64.(d) * rhs atol = max(tol, 1e-12)
        backing = randn(rng, T, 8, 5)
        dest = view(backing, 2:7, 2:4)
        before = copy(dest)
        work = FaerScratch(mul!, dest, q, dest, T(2), T(-1))
        mul!(dest, q, dest, T(2), T(-1); scratch=work)
        @test dest ≈ T(2) * d * before - before atol = tol
        short = FaerScratch(0)
        req = scratch_requirements(mul!, dest, q, dest)
        if req.bytes > 0
            before = copy(dest)
            @test_throws Exception mul!(dest, q, dest; scratch=short)
            @test dest == before
        end
        reserve!(work, svd, f; full=true)
        s = svd(f; full=true, scratch=work)
        @test size(s.U) == (6, 6)
        @test size(s.Vt) == (4, 4)
        @test Matrix(s.U[:, 1:4]) * Diagonal(s.S) * Matrix(s.Vt) ≈ a atol = tol
        close(f.U)
        @test all(isfinite, Matrix(f.V))
        close(f)
        close(work)
        close(short)
        for policy in (FaerParThreshold(), FaerParThreshold(0), FaerParThreshold(:never))
            team = FaerTeam(2)
            f = bidiagonalize(a; compact=true, team, par_threshold=policy)
            @test Matrix(f) ≈ a atol = tol
            close(f)
            close(team)
        end
    end
end

@testset "Compact bidiagonal exported representation and owner metadata" begin
    rng = MersenneTwister(8027)
    for T in (Float32, Float64, ComplexF32, ComplexF64), (m, n) in ((6, 4), (4, 6), (0, 3))
        source = FaerMatrix(randn(rng, T, m, n))
        f = bidiagonalize(source; compact=true)
        close(source)
        GC.gc()
        for q in (f.U, f.V)
            seq = FaerHouseholderSequence(q.basis, q.factor)
            embedded = Matrix{T}(I, Colmat._basis_dim(q), Colmat._basis_dim(q))
            if size(seq, 1) > 0
                hh = Matrix(seq)
                q.basis_conjugated && (hh = conj.(hh))
                embedded[(q.offset + 1):end, (q.offset + 1):end] = hh
            end
            @test embedded[:, 1:q.cols] ≈ Matrix(q) atol = 1000eps(real(T))
            close(seq)
        end
        old = f.U.handle
        other = bidiagonalize(zeros(T === Float64 ? Float32 : Float64, m, n); compact=true)
        f.U.handle = other.U.handle
        @test_throws Exception Matrix(f.U)
        f.U.handle = old
        held = f.V'
        reference = Matrix(held)
        close(f)
        @test Matrix(held) ≈ reference
        close(held)
        close(other)
    end
end
