using Colmat, Test, LinearAlgebra

const EVD_TUNING_TYPES = (Float32, Float64, ComplexF32, ComplexF64)
function _evd_state()
    v = zeros(UInt, 5)
    f = zeros(UInt8, 5)
    Colmat._native_call(Colmat.RustCrate.evd_tuning_state, v, f)
    return Int.(v), Int.(f)
end
function _evd_sample(::Type{T}, n) where {T}
    return T.([sin(i + 2j) + (T <: Complex ? im * cos(i * j) : 0) for i in 1:n, j in 1:n])
end
_ekey(x) = (round(real(x); digits=3), round(imag(x); digits=3))

@testset "Tuned general and self-adjoint EVD drivers" begin
    general = (
        hessenberg_par_threshold=FaerParThreshold(:never),
        hessenberg_blocking_threshold=16,
        evd_recursion_threshold=4,
    )
    selfadj = (
        tridiagonal_par_threshold=FaerParThreshold(:never),
        self_adjoint_recursion_threshold=4,
    )
    team = FaerTeam(2)
    for T in EVD_TUNING_TYPES, n in (0, 7, 70)
        a = _evd_sample(T, n)
        A = FaerMatrix(a)
        tol = 500eps(real(T)) * max(n, 1)
        E0 = eigen(A)
        for kw in ((;), (; team))
            E = with_evd_tuning(() -> eigen(A; kw...); general...)
            @test norm(a * E.vectors - E.vectors * Diagonal(E.values)) <=
                tol * max(norm(a), 1)
            @test sort(E.values; by=_ekey) ≈ sort(E0.values; by=_ekey)
            v = with_evd_tuning(() -> eigvals(A; kw...); general...)
            @test sort(v; by=_ekey) ≈ sort(E0.values; by=_ekey)
        end
        h = a + a'
        H = Hermitian(FaerMatrix(h))
        F = with_evd_tuning(() -> eigen(H); selfadj...)
        @test norm(h * F.vectors - F.vectors * Diagonal(F.values)) <= tol * max(norm(h), 1)
        @test with_evd_tuning(() -> eigvals(H; team); selfadj...) ≈ eigvals(Hermitian(h))
        # Queried scratch inside the scope serves the tuned call.
        with_evd_tuning(; selfadj...) do
            work = FaerScratch(eigen, H)
            @test eigen(H; scratch=work).values ≈ eigvals(Hermitian(h))
            close(work)
        end
        @test Matrix(A) == a
        foreach(close, (A, parent(H)))
    end
    close(team)
end

@testset "EVD tuning reaches the drivers and is scoped" begin
    S = Symmetric(FaerMatrix(_evd_sample(Float64, 120) + _evd_sample(Float64, 120)'))
    base = scratch_requirements(eigen, S)
    tuned = with_evd_tuning(
        () -> scratch_requirements(eigen, S); self_adjoint_recursion_threshold=4
    )
    @test tuned.bytes != base.bytes
    @test _evd_state() == (zeros(Int, 5), zeros(Int, 5))
    with_evd_tuning(; hessenberg_blocking_threshold=8) do
        @test _evd_state() == ([0, 8, 0, 0, 0], [0, 1, 0, 0, 0])
        with_evd_tuning(; evd_recursion_threshold=3) do
            @test _evd_state() == ([0, 0, 3, 0, 0], [0, 0, 1, 0, 0])
        end
        @test _evd_state() == ([0, 8, 0, 0, 0], [0, 1, 0, 0, 0])
    end
    @test_throws ErrorException with_evd_tuning(
        () -> error("boom"); evd_recursion_threshold=2
    )
    @test _evd_state() == (zeros(Int, 5), zeros(Int, 5))
    @test Colmat._evd_tuning_depth[] == 0
    for bad in (
        (; evd_recursion_threshold=0),
        (; self_adjoint_recursion_threshold=-1),
        (; hessenberg_blocking_threshold=-1),
        (; hessenberg_par_threshold=3),
        (; tridiagonal_par_threshold=:never),
    )
        @test_throws ArgumentError with_evd_tuning(() -> nothing; bad...)
    end
    # Schur tuning composes with the general driver's multishift QR step.
    A = FaerMatrix(_evd_sample(Float64, 90))
    E = with_schur_tuning(; shift_count=6, deflation_window=8) do
        with_evd_tuning(() -> eigen(A); evd_recursion_threshold=4)
    end
    @test norm(Matrix(A) * E.vectors - E.vectors * Diagonal(E.values)) <
        1e-10 * norm(Matrix(A))
    foreach(close, (A, parent(S)))
end

@testset "Destructive eigen! and eigvals! follow with_evd_tuning" begin
    general = (
        hessenberg_par_threshold=FaerParThreshold(:never),
        hessenberg_blocking_threshold=16,
        evd_recursion_threshold=4,
    )
    team = FaerTeam(2)
    for T in EVD_TUNING_TYPES, n in (0, 7, 70)
        a = _evd_sample(T, n)
        A = FaerMatrix(a)
        for kw in ((;), (; team))
            with_evd_tuning(; general...) do
                # Returned pairs are those of the tuned driver.
                R = eigen(A; kw...)
                B = copy(A)
                E = eigen!(B; kw...)
                @test E.values == R.values && E.vectors == R.vectors
                C = copy(A)
                @test eigvals!(C; kw...) == eigvals(A; kw...)
                # The written Schur form uses the scope's Hessenberg settings.
                h = hessenberg(
                    A; par_threshold=FaerParThreshold(:never), blocking_threshold=16
                )
                S = schur(h)
                @test Matrix(B) == Matrix(S.T) == Matrix(C)
                foreach(close, (B, C, h, S))
            end
        end
        @test Matrix(A) == a
        close(A)
    end
    # Only the Hessenberg overrides that are set change the written form.
    A = FaerMatrix(_evd_sample(Float64, 70))
    B = copy(A)
    with_evd_tuning(() -> eigvals!(B); evd_recursion_threshold=4)
    S = schur(A)
    @test Matrix(B) == Matrix(S.T)
    foreach(close, (A, B, S))
    close(team)
end

@testset "Self-adjoint eigen! writeback follows with_evd_tuning" begin
    selfadj = (self_adjoint_recursion_threshold=4,)
    for T in EVD_TUNING_TYPES, n in (0, 7, 120)
        a = _evd_sample(T, n)
        h = a + a'
        H = Hermitian(FaerMatrix(h))
        untuned = eigen(H)
        with_evd_tuning(; selfadj...) do
            R = eigen(H)
            X = Hermitian(copy(parent(H)))
            E = eigen!(X)
            @test E.values == R.values && E.vectors == R.vectors
            @test Matrix(parent(X)) == R.vectors
            Y = Hermitian(copy(parent(H)))
            @test eigvals!(Y) == eigvals(H)
            # Divide and conquer below faer's default recursion threshold
            # (128) is not the untuned QR path.
            n == 120 && @test R.vectors != untuned.vectors
            foreach(close, (parent(X), parent(Y)))
        end
        close(parent(H))
    end
end
