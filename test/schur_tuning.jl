using Colmat, Test, LinearAlgebra, Random

const TUNING_TYPES=(Float32, Float64, ComplexF32, ComplexF64)
_tuning_state() = (
    s = zeros(UInt, 7);
    Colmat._native_call(Colmat.RustCrate.schur_tuning_state, s);
    Int.(s)
)
_sample(::Type{T}, n, seed) where {T} =
    T.([sin(seed*i+j^2) + (T<:Complex ? im*cos(i*j+seed) : 0) for i = 1:n, j = 1:n])
_key(x) = (round(real(x); digits = 3), round(imag(x); digits = 3))

@testset "Tuned Schur and QZ factorizations" begin
    tuning=(
        shift_count = 6,
        deflation_window = 8,
        blocking_threshold = 12,
        nibble_threshold = 25,
    )
    for T in TUNING_TYPES, n in (0, 5, 90)
        a=_sample(T, n, 1)
        A=FaerMatrix(a)
        tol=200eps(real(T))*max(n, 1)
        F0=schur(A)
        F=with_schur_tuning(()->schur(A); tuning...)
        @test norm(Matrix(F.Z)*Matrix(F.T)*Matrix(F.Z)'-a) <= tol*max(norm(a), 1)
        @test sort(F.values; by = _key) ≈ sort(F0.values; by = _key)
        H=hessenberg(A)
        G=with_schur_tuning(()->schur(H); tuning...)
        @test sort(G.values; by = _key) ≈ sort(F0.values; by = _key)
        B=copy(A)
        S=with_schur_tuning(()->schur!(B); tuning...)
        @test sort(S.values; by = _key) ≈ sort(F0.values; by = _key)
        b=_sample(T, n, 2)+T(n)*I
        Bq=FaerMatrix(b)
        Q=with_schur_tuning(()->schur(A, Bq); tuning..., shift_chase_cost = 3)
        @test norm(Matrix(Q.Q)*Matrix(Q.S)*Matrix(Q.Z)'-a) <= tol*max(norm(a), 1)
        @test norm(Matrix(Q.Q)*Matrix(Q.T)*Matrix(Q.Z)'-b) <= tol*max(norm(b), 1)
        # Scratch queried and built inside the scope serves the tuned call.
        with_schur_tuning(; shift_count = 8) do
            work=FaerScratch(schur, A, Bq)
            @test schur(A, Bq; scratch = work) isa GeneralizedSchur
            close(work)
        end
        foreach(close, (A, B, Bq))
    end
end

@testset "Tuning changes QZ requirements and is scoped" begin
    A=FaerMatrix(_sample(Float64, 300, 3))
    B=FaerMatrix(_sample(Float64, 300, 4)+300I)
    base=scratch_requirements(schur, A, B)
    big=with_schur_tuning(
        ()->scratch_requirements(schur, A, B);
        shift_count = 200,
        deflation_window = 280,
    )
    @test big.bytes > base.bytes
    @test _tuning_state() == zeros(Int, 7)
    with_schur_tuning(; shift_count = 4, nibble_threshold = 10) do
        @test _tuning_state() == [4, 0, 0, 10, 0, 0, 0]
        with_schur_tuning(; deflation_window = 6) do
            @test _tuning_state() == [0, 6, 0, 0, 0, 0, 0]
        end
        @test _tuning_state() == [4, 0, 0, 10, 0, 0, 0]
    end
    @test_throws ErrorException with_schur_tuning(()->error("boom"); shift_count = 4)
    @test _tuning_state() == zeros(Int, 7)
    @test with_schur_tuning(()->42) == 42
    for bad in (
        (; shift_count = 1),
        (; deflation_window = 1),
        (; nibble_threshold = 101),
        (; blocking_threshold = -1),
        (; generalized_hessenberg_block_size = -1),
        (; generalized_hessenberg_blocking_threshold = -1),
    )
        @test_throws ArgumentError with_schur_tuning(()->nothing; bad...)
    end
    @test _tuning_state() == zeros(Int, 7)
    @test Colmat._schur_tuning_depth[] == 0
    foreach(close, (A, B))
end

_qz_close(f, a, b) = (
    q=Matrix(f.Q);
    z=Matrix(f.Z);
    first=f isa FaerGeneralizedHessenberg ? f.H : f.S;
    (norm(q*Matrix(first)*z'-a)+norm(q*Matrix(f.T)*z'-b))/max(norm(a)+norm(b), 1)
)

@testset "Generalized Hessenberg block size reaches QZ construction" begin
    A=FaerMatrix(_sample(Float64, 90, 5))
    B=FaerMatrix(_sample(Float64, 90, 6)+90I)
    base=scratch_requirements(hessenberg, A, B)
    req(bs)=with_schur_tuning(
        ()->scratch_requirements(hessenberg, A, B);
        generalized_hessenberg_block_size = bs,
    )
    # faer's blocked reduction needs an n×6·block_size panel.
    @test req(2).bytes < base.bytes < req(64).bytes
    @test req(1) == req(90) == req(2)
    @test with_schur_tuning(
        ()->scratch_requirements(schur, A, B);
        generalized_hessenberg_block_size = 64,
    ).bytes >= req(64).bytes
    # The forwarded blocking threshold is not read by the pinned faer kernel.
    @test with_schur_tuning(
        ()->scratch_requirements(hessenberg, A, B);
        generalized_hessenberg_blocking_threshold = 4,
    ) == base
    foreach(close, (A, B))
    team=FaerTeam(2)
    rng=MersenneTwister(1107)
    for T in TUNING_TYPES, n in (0, 5, 40, 90), bs in (1, 2, 8, 37, 64)
        a=randn(rng, T, n, n)
        b=randn(rng, T, n, n)
        A, B=FaerMatrix(a), FaerMatrix(b)
        tol=1000eps(real(T))*max(n, 1)
        with_schur_tuning(; generalized_hessenberg_block_size = bs) do
            for kw in ((;), (; team)), fn in (hessenberg, schur)
                # Exactly the queried workspace must serve the tuned kernel.
                r=scratch_requirements(fn, A, B; kw...)
                work=FaerScratch(r.bytes; alignment = r.alignment)
                F=fn(A, B; scratch = work, kw...)
                @test _qz_close(F, a, b) <= tol
                close(F)
                bang=fn===schur ? schur! : hessenberg!
                x, y=copy(A), copy(B)
                G=bang(x, y; scratch = work, kw...)
                @test _qz_close(G, a, b) <= tol
                close(G)
                close(work)
                H=fn(A, B; kw...)
                @test _qz_close(H, a, b) <= tol
                close(H)
            end
        end
        @test Matrix(A) == a && Matrix(B) == b
        foreach(close, (A, B))
    end
    close(team)
end

@testset "with_schur_tuning reaches the general EVD driver without resources" begin
    tuning=(shift_count = 6, deflation_window = 8, blocking_threshold = 12)
    for T in TUNING_TYPES, n in (0, 7, 90)
        a=_sample(T, n, 9)
        A=FaerMatrix(a)
        with_schur_tuning(; tuning...) do
            @test !Colmat._general_evd_default_path(nothing, nothing)
            @test Colmat._evd_default_path(nothing, nothing)
            work=FaerScratch(eigen, A)
            # The default call runs the same tuned workspace driver.
            E=eigen(A)
            R=eigen(A; scratch = work)
            @test E.values == R.values && E.vectors == R.vectors
            values=eigvals(A; scratch = FaerScratch(eigvals, A))
            @test eigvals(A) == values
            B=copy(A)
            F=eigen!(B)
            @test F.values == R.values && F.vectors == R.vectors
            S=schur(A)
            @test Matrix(B) == Matrix(S.T)
            C=copy(A)
            @test eigvals!(C) == values
            @test Matrix(C) == Matrix(S.T)
            foreach(close, (work, B, C, S))
        end
        @test Colmat._schur_tuning_depth[] == 0
        @test Colmat._general_evd_default_path(nothing, nothing)
        close(A)
    end
end
