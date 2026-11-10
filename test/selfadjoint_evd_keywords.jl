using Colmat, Test, LinearAlgebra

const SA_EVD_TYPES = (Float32, Float64, ComplexF32, ComplexF64)
function _sa_sample(::Type{T}, n) where {T}
    return T.([sin(i + 2j) + (T <: Complex ? im * cos(i * j) : 0) for i in 1:n, j in 1:n])
end
function _sa_state()
    v = zeros(UInt, 5)
    f = zeros(UInt8, 5)
    Colmat._native_call(Colmat.RustCrate.evd_tuning_state, v, f)
    return v, Int.(f)
end
function _sa_wrap(::Type{T}, a, uplo) where {T}
    return if T <: Complex
        Hermitian(a, uplo)
    else
        (uplo === :U ? Symmetric(a, uplo) : Hermitian(a, uplo))
    end
end

@testset "Complex Symmetric eigen follows Julia" begin
    for T in (ComplexF32, ComplexF64), uplo in (:L, :U), n in (0, 1, 6)
        tol = sqrt(eps(real(T))) * 10
        # Non-Hermitian complex symmetric: general (non-normal) EVD.
        a = _sa_sample(T, n)
        S = Symmetric(FaerMatrix(a), uplo)
        J = Symmetric(a, uplo)
        E = eigen(S)
        # Julia's `eigen` shortcut for diagonal input (n <= 1) is not mirrored.
        n > 0 && @test E isa Eigen{T,T}
        n > 1 && @test typeof(E) == typeof(eigen(J))
        @test Matrix(J) * E.vectors ≈ E.vectors * Diagonal(E.values) atol = tol
        @test E.values ≈ eigen(J).values atol = tol
        v = eigvals(S)
        @test typeof(v) == typeof(eigvals(J))
        @test v ≈ eigvals(J) atol = tol
        @test typeof(eigvals(S; sortby=nothing)) == typeof(v)
        if n > 0
            team = FaerTeam(2)
            @test eigvals(S; team) ≈ v atol = tol
            @test eigen(S; permute=true, scale=true).values ≈ v atol = tol
        end
        # A real stored triangle is Hermitian: Julia's ishermitian branch.
        r = T.(real.(a .+ transpose(a)))
        R = Symmetric(FaerMatrix(r), uplo)
        JR = Symmetric(r, uplo)
        ER = eigen(R)
        n > 1 && @test typeof(ER) == typeof(eigen(JR))
        @test ER isa Eigen{T,real(T)}
        @test ER.values ≈ eigen(JR).values atol = tol
        @test Matrix(JR) * ER.vectors ≈ ER.vectors * Diagonal(ER.values) atol = tol
        @test ER.vectors' * ER.vectors ≈ I atol = tol
        @test eigvals(R) isa Vector{real(T)}
        @test eigvals(R) ≈ eigvals(JR) atol = tol
    end
end

@testset "Self-adjoint alg keyword" begin
    for T in SA_EVD_TYPES, n in (0, 5, 140), uplo in (:L, :U)
        a = _sa_sample(T, n)
        H = _sa_wrap(T, FaerMatrix(a + a'), uplo)
        tol = sqrt(eps(real(T))) * max(n, 1)
        E0 = eigen(H)
        for alg in (
            LinearAlgebra.QRIteration(),
            LinearAlgebra.DivideAndConquer(),
            LinearAlgebra.RobustRepresentations(),
        )
            E = eigen(H; alg)
            @test E isa typeof(E0)
            @test E.values ≈ E0.values atol = tol
            @test Matrix(H) * E.vectors ≈ E.vectors * Diagonal(E.values) atol = tol
            @test eigvals(H; alg) ≈ E0.values atol = tol
            @test eigvals(H; alg, sortby=x -> -x) ≈ reverse(E0.values) atol = tol
            team = FaerTeam(2)
            @test eigen(H; alg, team).values ≈ E0.values atol = tol
            B = FaerMatrix(a + a')
            W = eigen!(_sa_wrap(T, B, uplo); alg)
            @test W.values ≈ E0.values atol = tol
            @test Matrix(B) ≈ W.vectors
            C = FaerMatrix(a + a')
            @test eigvals!(_sa_wrap(T, C, uplo); alg) ≈ E0.values atol = tol
        end
        @test eigen(H; native=true, alg=LinearAlgebra.DivideAndConquer()).values ≈ E0.values atol =
            tol
        @test_throws ArgumentError eigen(H; native=true, alg=LinearAlgebra.QRIteration())
        @test_throws ArgumentError eigen(H; alg=:qr)
        @test_throws ArgumentError eigvals(H; alg=1)
    end
    # QRIteration drops the divide-and-conquer workspace and keeps an outer scope.
    H = Hermitian(FaerMatrix(_sa_sample(Float64, 200) + _sa_sample(Float64, 200)'))
    full = scratch_requirements(eigen, H).bytes
    with_evd_tuning(tridiagonal_par_threshold=FaerParThreshold(:never)) do
        outer = _sa_state()
        inner = Ref{Any}()
        bytes = Ref(0)
        Colmat._selfadjoint_alg(LinearAlgebra.QRIteration()) do
            inner[] = _sa_state()
            bytes[] = scratch_requirements(eigen, H).bytes
        end
        @test bytes[] < full
        @test inner[][2] == [0, 0, 0, 1, 1]
        @test inner[][1][5] == outer[1][5]
        @test _sa_state() == outer
    end
    @test _sa_state() == (zeros(UInt, 5), zeros(Int, 5))
end

@testset "Self-adjoint irange and value interval" begin
    for T in SA_EVD_TYPES, n in (1, 6), uplo in (:L, :U)
        a = _sa_sample(T, n)
        h = a + a'
        H = _sa_wrap(T, FaerMatrix(h), uplo)
        J = _sa_wrap(T, h, uplo)
        tol = sqrt(eps(real(T))) * 10
        all = eigvals(J)
        for r in (1:n, 1:1, n:n, (n > 2 ? (2:(n - 1),) : ())...)
            E = eigen(H, r)
            @test E isa Eigen{T,real(T)}
            @test E.values ≈ all[r] atol = tol
            @test size(E.vectors) == (n, length(r))
            @test Matrix(J) * E.vectors ≈ E.vectors * Diagonal(E.values) atol = tol
            v = eigvals(H, r)
            @test v isa Vector{real(T)}
            @test v ≈ all[r] atol = tol
            B = FaerMatrix(h)
            @test eigen!(_sa_wrap(T, B, uplo), r).values ≈ all[r] atol = tol
            C = FaerMatrix(h)
            @test eigvals!(_sa_wrap(T, C, uplo), r) ≈ all[r] atol = tol
        end
        vl, vu = real(T)(all[1] - 1), real(T)(all[end] / 2 + all[1] / 2)
        expected = all[vl .< all .<= vu]
        E = eigen(H, vl, vu)
        @test E.values ≈ expected atol = tol
        @test size(E.vectors) == (n, length(expected))
        @test eigvals(H, vl, vu) ≈ expected atol = tol
        @test eigvals!(_sa_wrap(T, FaerMatrix(h), uplo), vl, vu) ≈ expected atol = tol
        @test eigen!(_sa_wrap(T, FaerMatrix(h), uplo), vl, vu).values ≈ expected atol = tol
        @test eigvals(H, 1, 2) isa Vector{real(T)}
        @test eigmin(H) ≈ all[1] atol = tol
        @test eigmax(H) ≈ all[end] atol = tol
        @test_throws ArgumentError eigvals(H, 0:1)
        @test_throws ArgumentError eigen(H, 1:(n + 1))
        @test_throws ArgumentError eigvals(H, 2:1)
        @test_throws ArgumentError eigen(H, 1.0, 1.0)
        @test_throws ArgumentError eigvals(H, 2.0, 1.0)
    end
    # Half-open (vl, vu], as LAPACK syevr.
    D = Hermitian(FaerMatrix(diagm([1.0, 2.0, 3.0, 4.0])))
    @test eigvals(D, 2.0, 3.0) == [3.0]
    @test eigen(D, 1.0, 4.0).values == [2.0, 3.0, 4.0]
    @test eigvals(D, 2:3) == [2.0, 3.0]
end
