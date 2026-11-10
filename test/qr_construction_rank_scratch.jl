using Colmat, LinearAlgebra, SparseArrays, Test
function qcr_fixture(T, m, n, deficient)
    A = T[0.01sin(i + 2j) for i in 1:m, j in 1:n]
    for i in 1:min(m, n)
        A[i, i] += T(5 + i / n)
    end
    T <: Complex && (A .+= T(0.01im) .* T[cos(2i + j) for i in 1:m, j in 1:n])
    deficient && n > 0 && (A[:, 1] .= 0)
    return A
end
@testset "QR construction scratch includes rank preparation" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        algorithm in (:auto, :supernodal),
        format in (:csc, :csr),
        (m, n) in ((0, 0), (3, 0), (1, 1), (5, 3), (32, 24), (96, 64)),
        deficient in (false, true)

        A = qcr_fixture(T, m, n, deficient)
        a = FaerSparseMatrix(sparse(A); format)
        p = FaerSparseSymbolic(a; kind=:qr, algorithm, ordering=reverse(collect(1:n)))
        req = scratch_requirements(factorize, p, a)
        work = FaerScratch(factorize, p, a)
        if req.bytes > 0
            short = FaerScratch(req.bytes - 1; alignment=req.alignment)
            @test_throws ArgumentError factorize(p, a; scratch=short)
            @test reserve!(short, factorize, p, a) === short
            close(short)
        end
        cap = scratch_capacity(work)
        f = factorize(p, a; scratch=work)
        g = copy(f)
        @test scratch_capacity(work) == cap
        @test Matrix(f) ≈ A atol = 1e-3 rtol = 1e-3
        b = ones(T, m, 2)
        gold = pinv(A; rtol=1e-5) * b
        @test Matrix(f \ b) ≈ gold atol = 1e-3 rtol = 1e-3
        close(work)
        close(p)
        close(a)
        close(f.source)
        close(f.plan)
        @test Matrix(f \ b) ≈ gold atol = 1e-3 rtol = 1e-3
        close(f)
        @test Matrix(g \ b) ≈ gold atol = 1e-3 rtol = 1e-3
        close(g)
    end
end
@testset "sparse construction validates native metadata before snapshot reads" begin
    A = [4.0 0.1; 0.2 5]
    a = FaerSparseMatrix(sparse(A))
    p = FaerSparseSymbolic(a; kind=:qr)
    original_dims = a.dims
    original_count = a.stored
    original_format = a.format
    a.stored += 1
    @test_throws Colmat.RustCall.RustError scratch_requirements(factorize, p, a)
    @test_throws Colmat.RustCall.RustError factorize(p, a)
    a.stored = original_count
    a.format = :csr
    @test_throws Colmat.RustCall.RustError scratch_requirements(factorize, p, a)
    @test_throws Colmat.RustCall.RustError factorize(p, a)
    a.format = original_format
    a.dims = (3, 2)
    p.dims = (3, 2)
    @test_throws Colmat.RustCall.RustError scratch_requirements(factorize, p, a)
    @test_throws Colmat.RustCall.RustError factorize(p, a)
    a.dims = original_dims
    p.dims = original_dims
    p.uplo = :U
    @test_throws Colmat.RustCall.RustError scratch_requirements(factorize, p, a)
    @test_throws Colmat.RustCall.RustError factorize(p, a)
    p.uplo = :L
    fake = FaerSparseMatrix{Float32,typeof(a.handle)}(
        a.handle, a.dims, a.stored, a.format, a.epoch
    )
    @test_throws Colmat.RustCall.RustError scratch_requirements(factorize, p, fake)
    @test_throws Colmat.RustCall.RustError factorize(p, fake)
    fake.handle = nothing
    @test Matrix(factorize(p, a)) ≈ A
    luplan = FaerSparseSymbolic(a; kind=:lu)
    fakeplan = FaerSparseSymbolic{:qr,typeof(luplan.handle)}(
        luplan.handle, luplan.dims, luplan.uplo
    )
    @test_throws Colmat.RustCall.RustError scratch_requirements(factorize, fakeplan, a)
    @test_throws Colmat.RustCall.RustError factorize(fakeplan, a)
    fakeplan.handle = nothing
    close(luplan)
    close(p)
    close(a)
end
