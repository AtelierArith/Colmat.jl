using Colmat, Test, LinearAlgebra, SparseArrays

function _tuning_laplacian(k)
    n = k^2
    a = spzeros(n, n)
    for i in 1:k, j in 1:k
        v = (i - 1) * k + j
        a[v, v] = 4.5
        i < k && (a[v, v + k] = a[v + k, v] = -1)
        j < k && (a[v, v + 1] = a[v + 1, v] = -1)
    end
    return a
end

@testset "Sparse supernodal threshold and relaxation controls" begin
    a = _tuning_laplacian(20)
    n = size(a, 1)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        x = T.(a)
        b = ones(T, n)
        source = FaerSparseMatrix(x)
        f = cholesky(Hermitian(source); supernodal_threshold=Inf)
        @test f.algorithm === :simplicial
        @test Matrix(f.L * f.U) ≈ Matrix(x)[Vector(f.p), Vector(f.p)] rtol = 50eps(real(T))
        g = cholesky(Hermitian(source); supernodal_threshold=0)
        @test g.algorithm === :supernodal
        @test f \ b ≈ g \ b rtol = 100eps(real(T))
        # Relaxation trades padded storage for wider supernodes.
        tight = cholesky(
            Hermitian(source); algorithm=:supernodal, relax=Tuple{Int,Float64}[]
        )
        default = cholesky(Hermitian(source); algorithm=:supernodal)
        wide = cholesky(
            Hermitian(source); algorithm=:supernodal, relax=[(typemax(Int), 1.0)]
        )
        @test tight.factor_storage < default.factor_storage < wide.factor_storage
        for h in (tight, wide)
            @test Matrix(h) ≈ Matrix(x) rtol = 50eps(real(T))
            @test h \ b ≈ Matrix(x) \ b rtol = 100eps(real(T))
        end
        # Caller scratch and symbolic reuse keep the tuned analysis.
        s = FaerScratch(FaerSparseSymbolic, source; kind=:cholesky, supernodal_threshold=0)
        plan = FaerSparseSymbolic(source; kind=:cholesky, supernodal_threshold=0, scratch=s)
        @test plan.algorithm === :supernodal
        @test factorize(plan, source) \ b ≈ g \ b rtol = 100eps(real(T))
        for (alg, kw) in (
            (:auto, (; supernodal_threshold=0.0)),
            (:auto, (; supernodal_threshold=Inf, relax=[(4, 0.5)])),
            (:supernodal, (; relax=Tuple{Int,Float64}[])),
            (:supernodal, (; relax=[(typemax(Int), 1.0)])),
        )
            F = lu(source; algorithm=alg, kw...)
            @test F \ b ≈ Matrix(x) \ b rtol = 100eps(real(T))
            @test Matrix(F.L * F.U) ≈ Matrix(x)[Vector(F.p), Vector(F.q)] rtol =
                50eps(real(T))
        end
        close(source)
    end
    m = FaerSparseMatrix(a)
    h = Hermitian(m)
    @test_throws ArgumentError cholesky(h; supernodal_threshold=-1)
    @test_throws ArgumentError cholesky(h; supernodal_threshold=NaN)
    @test_throws ArgumentError cholesky(h; supernodal_threshold=1, algorithm=:supernodal)
    @test_throws ArgumentError cholesky(h; relax=[(4, 0.5)], algorithm=:simplicial)
    @test_throws ArgumentError cholesky(h; relax=[(4, 1.5)])
    @test_throws ArgumentError cholesky(h; relax=[(-1, 0.5)])
    @test_throws ArgumentError cholesky(h; relax=[4])
    @test_throws ArgumentError lu(m; algorithm=:simplicial, relax=[(4, 0.5)])
    @test_throws ArgumentError lu(m; algorithm=:simplicial, supernodal_threshold=1)
    @test_throws ArgumentError qr(m; algorithm=:simplicial, supernodal_threshold=1)
    @test_throws ArgumentError FaerSparseSymbolic(
        m; kind=:qr, algorithm=:simplicial, relax=[(4, 0.5)]
    )
    close(m)
end

function _tuning_qr_pattern(m, n)
    rows = Int[]
    cols = Int[]
    for j in 1:n, r in (j - 1, j, j * 7 + 2, j + n ÷ 2)
        push!(rows, mod(r, m) + 1)
        push!(cols, j)
    end
    return sparse(rows, cols, [1.0 + mod(3i, 7) / 4 for i in eachindex(rows)], m, n)
end

@testset "Sparse QR supernodal threshold, relaxation and blocking" begin
    for (m, n) in ((300, 150), (150, 300)), T in (Float32, Float64, ComplexF32, ComplexF64)
        x = T.(_tuning_qr_pattern(m, n))
        T <: Complex && (x += im * T.(_tuning_qr_pattern(m, n)) ./ 3)
        source = FaerSparseMatrix(x)
        b = ones(T, m)
        reference = qr(source)
        tol = (T <: Union{Float32,ComplexF32} ? 1e3 : 1e2) * eps(real(T))
        variants = (
            (; supernodal_threshold=Inf),
            (; supernodal_threshold=0),
            (; supernodal_threshold=0, relax=Tuple{Int,Float64}[]),
            (; algorithm=:supernodal, relax=[(typemax(Int), 1.0)]),
            (; algorithm=:supernodal, blocking_threshold=0),
            (; supernodal_threshold=0, blocking_threshold=typemax(Int)),
        )
        algorithms = Symbol[]
        for kw in variants
            F = qr(source; kw...)
            @test F \ b ≈ reference \ b rtol = tol
            @test Matrix(F) ≈ Matrix(x) rtol = tol
            push!(algorithms, F.algorithm)
            # Symbolic reuse and in-place updates keep the controls.
            plan = FaerSparseSymbolic(source; kind=:qr, kw...)
            G = factorize(plan, source)
            @test plan.algorithm === G.algorithm === F.algorithm
            qr!(G, source)
            @test G \ b ≈ reference \ b rtol = tol
            r = scratch_requirements(factorize, plan, source)
            s = FaerScratch(r.bytes; alignment=r.alignment)
            H = factorize(plan, source; scratch=s)
            @test H \ b ≈ reference \ b rtol = tol
            foreach(close, (F, G, H, plan, s))
        end
        # Inf forces the simplicial structure and 0 the supernodal one (wide QR
        # reports its square head); the native tests check relaxation storage.
        @test algorithms == [:simplicial; fill(:supernodal, 5)]
        close(source)
    end
    m = FaerSparseMatrix(_tuning_qr_pattern(40, 20))
    @test_throws ArgumentError qr(m; supernodal_threshold=-1)
    @test_throws ArgumentError qr(m; algorithm=:supernodal, supernodal_threshold=1)
    @test_throws ArgumentError qr(m; algorithm=:simplicial, relax=[(4, 0.5)])
    @test_throws ArgumentError qr(m; algorithm=:simplicial, blocking_threshold=16)
    @test_throws ArgumentError qr(m; blocking_threshold=-1)
    @test_throws ArgumentError qr(m; blocking_threshold=1.5)
    s = FaerSparseMatrix(sparse(4.0I, 6, 6))
    @test_throws ArgumentError FaerSparseSymbolic(s; kind=:lu, blocking_threshold=16)
    @test_throws ArgumentError FaerSparseSymbolic(s; kind=:cholesky, blocking_threshold=16)
    foreach(close, (m, s))
end

@testset "Sparse Cholesky triangle selection shares caller scratch" begin
    a = _tuning_laplacian(6)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        uplo in (:L, :U),
        algorithm in (:simplicial, :supernodal)

        x = T.(a)
        both = FaerSparseMatrix(x)
        one = FaerSparseMatrix(uplo === :L ? tril(x) : triu(x))
        hb, ho = Hermitian(both, uplo), Hermitian(one, uplo)
        pb = FaerSparseSymbolic(both; kind=:cholesky, uplo, algorithm)
        po = FaerSparseSymbolic(one; kind=:cholesky, uplo, algorithm)
        rb = scratch_requirements(factorize, pb, both)
        ro = scratch_requirements(factorize, po, one)
        # Inputs holding both triangles reserve room for the selected one.
        @test rb.bytes > ro.bytes
        b = ones(T, size(x, 1))
        for (h, r) in ((hb, rb), (ho, ro))
            s = FaerScratch(r.bytes; alignment=r.alignment)
            f = cholesky(h; algorithm, scratch=s)
            @test f \ b ≈ Matrix(x) \ b rtol = 100eps(real(T))
            @test Matrix(f) ≈ Matrix(x) rtol = 50eps(real(T))
        end
        foreach(close, (pb, po, both, one))
    end
end
