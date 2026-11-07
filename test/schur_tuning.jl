using Colmat, Test, LinearAlgebra

const TUNING_TYPES=(Float32, Float64, ComplexF32, ComplexF64)
_tuning_state() = (
    s = zeros(UInt, 5);
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
    @test _tuning_state() == zeros(Int, 5)
    with_schur_tuning(; shift_count = 4, nibble_threshold = 10) do
        @test _tuning_state() == [4, 0, 0, 10, 0]
        with_schur_tuning(; deflation_window = 6) do
            @test _tuning_state() == [0, 6, 0, 0, 0]
        end
        @test _tuning_state() == [4, 0, 0, 10, 0]
    end
    @test_throws ErrorException with_schur_tuning(()->error("boom"); shift_count = 4)
    @test _tuning_state() == zeros(Int, 5)
    @test with_schur_tuning(()->42) == 42
    for bad in (
        (; shift_count = 1),
        (; deflation_window = 1),
        (; nibble_threshold = 101),
        (; blocking_threshold = -1),
    )
        @test_throws ArgumentError with_schur_tuning(()->nothing; bad...)
    end
    @test _tuning_state() == zeros(Int, 5)
    foreach(close, (A, B))
end
