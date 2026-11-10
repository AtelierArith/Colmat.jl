using Test, LinearAlgebra, Colmat

function qzw_factor(T, n)
    R = typeof(real(zero(T)))
    s = Matrix{T}(Diagonal(T.(1:n)))
    t = Matrix{T}(I, n, n)
    return GeneralizedSchur(
        FaerMatrix(s),
        FaerMatrix(t),
        zeros(Complex{R}, n),
        zeros(T, n),
        FaerMatrix(t),
        FaerMatrix(t),
    )
end
qzw_snapshot(f) = (map(Matrix, (f.S, f.T, f.Q, f.Z)), copy(f.α), copy(f.β))
function qzw_check(f, expected, a, b, tol)
    n = size(f, 1)
    @test f.values ≈ expected
    @test Matrix(f.Q * f.S * f.Z') ≈ a atol = tol rtol = tol
    @test Matrix(f.Q * f.T * f.Z') ≈ b atol = tol rtol = tol
    @test Matrix(f.Q)' * Matrix(f.Q) ≈ Matrix{eltype(f)}(I, n, n) atol = tol rtol = tol
    @test Matrix(f.Z)' * Matrix(f.Z) ≈ Matrix{eltype(f)}(I, n, n) atol = tol rtol = tol
end

@testset "Caller-owned generalized Schur reorder scratch" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), n in (0, 1, 4, 9)
        tol = typeof(real(zero(T))) === Float32 ? 3e-5 : 3e-12
        f = qzw_factor(T, n)
        a, b = Matrix(f.S), Matrix(f.T)
        select = BitVector(iseven.(1:n))
        expected = vcat(collect(1:n)[select], collect(1:n)[.!select])
        req = scratch_requirements(ordschur, f)
        @test req == scratch_requirements(ordschur!, f)
        work = FaerScratch(ordschur, f)
        @test scratch_capacity(work) == req
        cap = scratch_capacity(work)
        saved = qzw_snapshot(f)
        for mask in (select, copy(select))
            g = ordschur(f, mask; scratch=work)
            qzw_check(g, expected, a, b, tol)
            @test qzw_snapshot(f) == saved
            @test scratch_capacity(work) == cap
            close(g)
        end
        g = copy(f)
        owners = (g.S, g.T, g.Q, g.Z)
        v = view(g.S, :, :)
        @test ordschur!(g, select; scratch=work) === g
        @test all(x === y for (x, y) in zip(owners, (g.S, g.T, g.Q, g.Z)))
        @test Matrix(v) == Matrix(g.S)
        qzw_check(g, expected, a, b, tol)
        @test scratch_capacity(work) == cap
        close(g)
        # Shared reuse is serialized by the package lock, including Julia Tasks.
        results = fetch.([@async ordschur(f, select; scratch=work) for _ in 1:2])
        foreach(g -> qzw_check(g, expected, a, b, tol), results)
        foreach(close, results)
        if req.bytes > 0
            short = FaerScratch(req.bytes - 1; alignment=req.alignment)
            for fun in (ordschur, ordschur!)
                @test_throws ArgumentError fun(f, select; scratch=short)
                @test qzw_snapshot(f) == saved
            end
            reserve!(short, ordschur!, f)
            h = ordschur(f, select; scratch=short)
            qzw_check(h, expected, a, b, tol)
            close(h)
            close(short)
        end
        if req.alignment > 1
            unaligned = FaerScratch(req.bytes; alignment=1)
            @test_throws ArgumentError ordschur!(f, select; scratch=unaligned)
            @test qzw_snapshot(f) == saved
            close(unaligned)
        end
        close(work)
        @test_throws ArgumentError ordschur!(f, select; scratch=work)
        @test_throws ArgumentError ordschur(f, select; scratch=zeros(UInt8, req.bytes))
        @test qzw_snapshot(f) == saved
        close(f)
        @test_throws ArgumentError scratch_requirements(ordschur, f)
    end
    # Selecting one member of a real block moves the pair, using exact query.
    for T in (Float32, Float64)
        tol = T === Float32 ? 3e-5 : 3e-12
        f = qzw_factor(T, 4)
        copyto!(f.S, T[1 0 0 0; 0 0 -2 0; 0 2 0 0; 0 0 0 4])
        a, b = Matrix(f.S), Matrix(f.T)
        work = FaerScratch(ordschur, f)
        g = ordschur(f, Bool[0, 0, 1, 0]; scratch=work)
        qzw_check(g, Complex{T}[2im, -2im, 1, 4], a, b, tol)
        @test Matrix(g.S)[2, 1] != 0
        close(g)
        close(work)
        close(f)
    end
end

@testset "Reorder scratch remains reusable after rejection" begin
    f = qzw_factor(Float64, 2)
    copyto!(f.S, [1.0 0; 0 0])
    copyto!(f.T, [1.0 0; 0 0])
    work = FaerScratch(ordschur, f)
    saved = qzw_snapshot(f)
    cap = scratch_capacity(work)
    @test_throws Exception ordschur!(f, Bool[0, 1]; scratch=work)
    @test qzw_snapshot(f) == saved
    @test scratch_capacity(work) == cap
    close(f)
    g = qzw_factor(Float64, 2)
    h = ordschur(g, Bool[0, 1]; scratch=work)
    qzw_check(h, [2, 1], Matrix(g.S), Matrix(g.T), 1e-12)
    @test scratch_capacity(work) == cap
    close(h)
    close(g)
    close(work)
end
