using Test, Colmat, LinearAlgebra
function ssw_fixture(T, m, n)
    a = T[0.04sin(i + 2j) for i in 1:m, j in 1:n]
    for i in 1:min(m, n)
        a[i, i] += T(3 + i)
    end
    T <: Complex && (a .+= T(0.03im) .* T[cos(i + 3j) for i in 1:m, j in 1:n])
    return a
end
function ssw_check(f, a, T, S, team; atol=0, rtol=min(size(a)...) * eps(real(T)))
    tol = 1500eps(real(T)) * max(size(a)..., 1)
    for (g, ref) in ((f, a), (transpose(f), transpose(a)), (adjoint(f), adjoint(a))),
        right in (false, true)

        b = fill(
            S <: Complex ? S(1 + 0.2im) : one(S),
            right ? (2, size(ref, 2)) : (size(ref, 1), 2),
        )
        inverse = pinv(Matrix(ref); atol, rtol)
        gold = right ? b * inverse : inverse * b
        P = promote_type(T, S)
        out = zeros(P, size(gold))
        fun = right ? rdiv! : ldiv!
        args = right ? (out, b, g) : (out, g, b)
        work = FaerScratch(fun, args...; atol, rtol, team)
        req = scratch_requirements(fun, args...; atol, rtol, team)
        cap = scratch_capacity(work)
        for _ in 1:2
            @test fun(args...; scratch=work, atol, rtol, team) === out
            @test out ≈ gold atol = tol * max(norm(gold), 1) rtol = tol
            @test scratch_capacity(work) == cap
        end
        division = right ? (/) : (\)
        divargs = right ? (b, g) : (g, b)
        @test scratch_requirements(division, divargs...; atol, rtol, team) == req
        native = division(divargs...; scratch=work, atol, rtol, team)
        @test native isa FaerMatrix{P}
        @test Matrix(native) ≈ gold atol = tol * max(norm(gold), 1) rtol = tol
        close(native)
        rhs = FaerMatrix(b)
        dest = FaerMatrix(zeros(P, size(gold)))
        nativeargs = right ? (dest, rhs, g) : (dest, g, rhs)
        @test fun(nativeargs...; scratch=work, atol, rtol, team) === dest
        @test Matrix(dest) ≈ gold atol = tol * max(norm(gold), 1) rtol = tol
        close(rhs)
        close(dest)
        if req.bytes > 0
            short = FaerScratch(req.bytes - 1; alignment=req.alignment)
            before = copy(out)
            @test_throws ArgumentError fun(args...; scratch=short, atol, rtol, team)
            @test out == before
            reserve!(short, fun, args...; atol, rtol, team)
            @test fun(args...; scratch=short, atol, rtol, team) === out
            @test out ≈ gold atol = tol * max(norm(gold), 1) rtol = tol
            close(short)
        end
        if size(ref, 1) == size(ref, 2)
            aliased = P.(b)
            args = right ? (aliased, g) : (g, aliased)
            @test fun(args...; scratch=work, atol, rtol, team) === aliased
            @test aliased ≈ gold atol = tol * max(norm(gold), 1) rtol = tol
        end
        close(work)
    end
end
@testset "saved SVD solve caller workspace" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        S in (Float32, Float64, ComplexF32, ComplexF64), full in (false, true),
        (m, n) in ((0, 0), (0, 3), (3, 0), (1, 1), (7, 3), (3, 7), (4, 4))

        a = ssw_fixture(T, m, n)
        source = FaerMatrix(a)
        f = svd(source; native=true, full)
        close(source)
        team = FaerTeam(2)
        ssw_check(f, a, T, S, team)
        close(f)
        close(team)
    end
end
@testset "saved self-adjoint EVD solve caller workspace" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        S in (Float32, Float64, ComplexF32, ComplexF64),
        n in (0, 1, 5)

        a = ssw_fixture(T, n, n)
        a = a + a'
        n > 1 && (a[1, 1] = -abs(a[1, 1]))
        source = FaerMatrix(a)
        f = eigen(Hermitian(source); native=true)
        close(source)
        ssw_check(f, a, T, S, nothing)
        close(f)
    end
end
@testset "clipping, rank deficiency and blocked spectral workspace" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        kind in (:svd, :eigen),
        threaded in (false, true)

        a = Matrix(Diagonal(T[4, -0.01, 0]))
        source = FaerMatrix(a)
        f = if kind === :svd
            svd(source; native=true, full=true)
        else
            eigen(Hermitian(source); native=true)
        end
        team = threaded ? FaerTeam(2) : nothing
        ssw_check(f, a, T, ComplexF64, team; atol=0.1, rtol=0)
        ssw_check(f, a, T, ComplexF64, team; atol=0, rtol=0.1)
        close(f)
        close(source)
        isnothing(team) || close(team)
    end
    for T in (Float32, Float64, ComplexF32, ComplexF64), (m, n) in ((129, 65), (33, 70))
        a = ssw_fixture(T, m, n)
        a[:, end] .= a[:, 1]
        f = svd(FaerMatrix(a); native=true)
        team = FaerTeam(2)
        ssw_check(f, a, T, ComplexF64, team)
        close(f)
        close(team)
    end
end
@testset "spectral workspace vectors, views, inverse and lifetimes" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), kind in (:svd, :eigen)
        a = ssw_fixture(T, 5, 5)
        a = a * a' + I
        source = FaerMatrix(a)
        f = kind === :svd ? svd(source; native=true) : eigen(Hermitian(source); native=true)
        close(source)
        team = FaerTeam(2)
        b = fill(ComplexF64(1 + 0.2im), 5, 2)
        gold = a \ b
        owner = zeros(ComplexF64, 10, 4)
        dest = view(owner, 1:2:10, 1:2:4)
        work = FaerScratch(ldiv!, dest, f, b; team)
        @test ldiv!(dest, f, b; scratch=work, team) === dest
        @test dest ≈ gold rtol = 2000eps(real(T)) atol = 2000eps(real(T))
        x = copy(b[:, 1])
        y = zeros(ComplexF64, 5)
        reserve!(work, ldiv!, y, f, x; team)
        @test ldiv!(y, f, x; scratch=work, team) === y
        @test y ≈ gold[:, 1] rtol = 2000eps(real(T)) atol = 2000eps(real(T))
        nx = FaerVector(x)
        ny = FaerVector(y)
        @test ldiv!(ny, f, nx; scratch=work, team) === ny
        @test Vector(ny) ≈ gold[:, 1] rtol = 2000eps(real(T)) atol = 2000eps(real(T))
        for g in (f, transpose(f), adjoint(f))
            reserve!(work, inv, g; team)
            result = inv(g; scratch=work, team)
            reference = if g === f
                a
            elseif g isa LinearAlgebra.TransposeFactorization
                transpose(a)
            else
                adjoint(a)
            end
            @test Matrix(result) ≈ inv(reference) rtol = 2000eps(real(T)) atol =
                2000eps(real(T))
            close(result)
        end
        copied = deepcopy(f)
        close(f)
        @test_throws ArgumentError scratch_requirements(\, f, b)
        reserve!(work, ldiv!, dest, copied, b; team)
        tasks = [@async begin
            out = zeros(ComplexF64, 5, 2)
            ldiv!(out, copied, b; scratch=work, team)
            out
        end for _ in 1:3]
        @test all(
            x -> isapprox(x, gold; rtol=2000eps(real(T)), atol=2000eps(real(T))),
            fetch.(tasks),
        )
        before = copy(dest)
        @test_throws ArgumentError ldiv!(dest, copied, b; scratch=:bad, team)
        @test dest == before
        req = scratch_requirements(ldiv!, dest, copied, b; team)
        if req.alignment > 1
            weak = FaerScratch(req.bytes; alignment=1)
            @test_throws ArgumentError ldiv!(dest, copied, b; scratch=weak, team)
            @test dest == before
            close(weak)
        end
        close(team)
        @test_throws ArgumentError scratch_requirements(ldiv!, dest, copied, b; team)
        @test_throws ArgumentError ldiv!(dest, copied, b; scratch=work, team)
        @test dest == before
        close(work)
        @test_throws ArgumentError ldiv!(dest, copied, b; scratch=work)
        @test dest == before
        close(nx)
        close(ny)
        close(copied)
    end
end
@testset "spectral source precision and zero maps at extreme RHS" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), kind in (:svd, :eigen)
        a = Matrix(Diagonal(T[4, -0.01, 0]))
        f = if kind === :svd
            svd(FaerMatrix(a); native=true)
        else
            eigen(Hermitian(FaerMatrix(a)); native=true)
        end
        for g in (f, transpose(f), adjoint(f)), right in (false, true)
            b = fill(T(floatmax(real(T))), right ? (2, 3) : (3, 2))
            out = zeros(T, size(b))
            fun = right ? rdiv! : ldiv!
            args = right ? (out, b, g) : (out, g, b)
            work = FaerScratch(fun, args...; rtol=Inf)
            @test fun(args...; scratch=work, rtol=Inf) === out
            @test iszero(out)
            result = right ? (/)(b, g; rtol=Inf) : (\)(g, b; rtol=Inf)
            @test iszero(Matrix(result))
            close(result)
            close(work)
        end
        @test_throws SingularException inv(f; scratch=FaerScratch(inv, f))
        close(f)
    end
    for T in (Float32, ComplexF32)
        f = svd(FaerMatrix(Matrix(Diagonal(T[1, 1e-9]))); native=true)
        b = ones(ComplexF64, 2, 1)
        out = zeros(ComplexF64, 2, 1)
        work = FaerScratch(ldiv!, out, f, b)
        @test ldiv!(out, f, b; scratch=work) === out
        @test out ≈ ComplexF64[1; 0;;]
        @test_throws ArgumentError scratch_requirements(ldiv!, out, f, b; rtol=-1)
        @test_throws ArgumentError scratch_requirements(ldiv!, out, f, b; atol=NaN)
        close(work)
        close(f)
    end
end
@testset "spectral solve owner validation and failed conversion" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        f = svd(FaerMatrix(Matrix{T}(I, 4, 4)); native=true)
        f.dims = (3, 3)
        @test_throws Colmat.RustCall.RustError scratch_requirements(\, f, ones(T, 3, 2))
        @test_throws Colmat.RustCall.RustError (\)(f, ones(T, 3, 2); scratch=FaerScratch())
        @test_throws Colmat.RustCall.RustError inv(f; scratch=FaerScratch())
        close(f)
    end
    f = eigen(Hermitian(FaerMatrix(Matrix{ComplexF64}(I, 3, 3))); native=true)
    for (T, K) in ((Float32, :eigen), (ComplexF64, :svd))
        invalid = FaerSpectralFactor{T,K,typeof(f.handle)}(
            f.handle, size(f), f.ucols, f.vcols, copy(f.order)
        )
        @test_throws Colmat.RustCall.RustError scratch_requirements(
            \, invalid, ones(T, 3, 2)
        )
        @test_throws Colmat.RustCall.RustError (\)(
            invalid, ones(T, 3, 2); scratch=FaerScratch()
        )
        invalid.handle = nothing
    end
    b = FaerMatrix(ones(ComplexF64, 3, 2))
    invalid = FaerMatrix{Float32,typeof(b.handle)}(b.handle, size(b))
    @test_throws Colmat.RustCall.RustError scratch_requirements(\, f, invalid)
    @test_throws Colmat.RustCall.RustError (\)(f, invalid; scratch=FaerScratch())
    close(invalid)
    close(b)
    close(f)
    for T in (ComplexF32, ComplexF64)
        a = T[4 im; -im 3]
        f = eigen(Hermitian(FaerMatrix(a)); native=true)
        b = ones(T, 2, 1)
        out = fill(42.0, 2, 1)
        work = FaerScratch(ldiv!, out, f, b)
        @test_throws InexactError ldiv!(out, f, b; scratch=work)
        @test out == fill(42.0, 2, 1)
        close(work)
        close(f)
    end
end
@testset "spectral exact stored-value tolerance thresholds" begin
    for T in (Float32, ComplexF32), kind in (:svd, :eigen)
        a = Matrix(Diagonal(T[1, 0.1]))
        f = if kind === :svd
            svd(FaerMatrix(a); native=true)
        else
            eigen(Hermitian(FaerMatrix(a)); native=true)
        end
        values = kind === :svd ? f.S : f.values
        small = minimum(abs, values)
        large = maximum(abs, values)
        b = ones(ComplexF64, 2, 1)
        out = zeros(ComplexF64, 2, 1)
        # EVD rounding can move the stored small eigenvalue several Float32 ulps.
        # Thresholds therefore straddle that actual value, not the input literal.
        for at in (small, prevfloat(Float64(small)))
            work = FaerScratch(ldiv!, out, f, b; atol=at, rtol=0)
            gold =
                ComplexF64[inv(Float64(large)); at isa Float32 ? 0 : inv(Float64(small));;]
            @test ldiv!(out, f, b; scratch=work, atol=at, rtol=0) === out
            @test out ≈ gold rtol = 500eps(Float32) atol = 500eps(Float32)
            result = (\)(f, b; atol=at, rtol=0)
            @test Matrix(result) ≈ gold rtol = 500eps(Float32) atol = 500eps(Float32)
            close(result)
            close(work)
        end
        close(f)
    end
end
