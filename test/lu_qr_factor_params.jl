using Test, Colmat, LinearAlgebra

_lq_state() = (
    v = zeros(UInt, 5);
    f = zeros(UInt8, 5);
    Colmat._native_call(Colmat.RustCrate.lu_qr_tuning_state, v, f);
    (v, Int.(f))
)
_lq_tol(::Type{T}) where {T} = real(T)===Float32 ? 1e-3 : 1e-10
function _lq_sample(::Type{T}, m, n) where {T}
    X=T[sin(i+2j)+(T<:Complex ? im*cos(i*j) : 0) for i = 1:m, j = 1:n]
    for i = 1:min(m, n)
        X[i, i]+=T(3)
    end
    return X
end
_packed(F) = Matrix(F.packed)
_direct(f!, A, p...) = f!(A, p...; scratch = FaerScratch(f!, A, p...))

@testset "with_lu_tuning / with_qr_tuning" begin
    @test _lq_state()==(zeros(Int, 5), zeros(Int, 5))
    lu_tuned=(
        partial_recursion_threshold = 1,
        partial_par_threshold = FaerParThreshold(:never),
        full_par_threshold = FaerParThreshold(7),
    )
    qr_tuned=(blocking_threshold = 0, pivoted_par_threshold = FaerParThreshold(:never))
    for T in (Float32, Float64, ComplexF32, ComplexF64), (m, n) in ((0, 0), (5, 5), (130, 130), (150, 90))
        X=_lq_sample(T, m, n)
        tol=_lq_tol(T)
        ref_lu=_packed(lu(FaerMatrix(X); check = false))
        ref_qr=_packed(qr(FaerMatrix(X)))
        results=with_lu_tuning(; lu_tuned...) do
            @test _lq_state()==([1, typemax(UInt), 7, 0, 0], [1, 1, 1, 0, 0])
            # Retained, borrowed, staged and direct paths use the same parameters.
            F=lu(FaerMatrix(X); check = false)
            @test _packed(lu(faerview(X); check = false))==_packed(F)
            A=FaerMatrix(X)
            lu!(A; check = false)
            @test Matrix(A)==_packed(F)
            @test Matrix(_direct(lu!, FaerMatrix(X)).factors)==_packed(F)
            G=full_piv_lu(FaerMatrix(X))
            @test Matrix(_direct(full_piv_lu!, FaerMatrix(X)).factors)==_packed(G)
            F, G
        end
        F, G=results
        @test _lq_state()==(zeros(Int, 5), zeros(Int, 5))
        @test Matrix(F)≈X rtol=tol atol=tol
        @test Matrix(G)≈X rtol=tol atol=tol
        # Recursion threshold one changes the blocking of the default LU.
        m==n==130 && @test _packed(F)!=ref_lu
        # The QR scope nests inside the LU scope and keeps its fields.
        Q, P=with_lu_tuning(; partial_recursion_threshold = 3) do
            with_qr_tuning(; qr_tuned...) do
                @test _lq_state()==([3, 0, 0, 0, typemax(UInt)], [1, 0, 0, 1, 1])
                Q=qr(FaerMatrix(X))
                @test _packed(qr(faerview(X)))==_packed(Q)
                @test Matrix(_direct(qr!, FaerMatrix(X)).factors)==_packed(Q)
                P=qr(FaerMatrix(X), ColumnNorm())
                @test Matrix(_direct(qr!, FaerMatrix(X), ColumnNorm()).factors)==_packed(P)
                Q, P
            end
        end
        @test _lq_state()==(zeros(Int, 5), zeros(Int, 5))
        @test Matrix(Q)≈X rtol=tol atol=tol
        @test Matrix(P)≈X rtol=tol atol=tol
        m==n==130 && T===Float64 && @test _packed(Q)!=ref_qr
        if m==n>0
            b=ones(T, n)
            @test X*(F\b)≈b rtol=sqrt(tol)
            @test X*(Q\b)≈b rtol=sqrt(tol)
            @test X*(P\b)≈b rtol=sqrt(tol)
        end
    end

    @testset "team, scratch queries and :never" begin
        X=_lq_sample(Float64, 160, 120)
        team=FaerTeam(2)
        seq=qr(FaerMatrix(X), ColumnNorm())
        with_qr_tuning(; pivoted_par_threshold = FaerParThreshold(:never)) do
            # `:never` keeps the column-pivoted kernel sequential on a team.
            P=qr(FaerMatrix(X), ColumnNorm(); team)
            @test _packed(P)==_packed(seq)
            A=FaerMatrix(X)
            work=FaerScratch(qr!, A, ColumnNorm(); team)
            @test reserve!(work, qr!, A, ColumnNorm(); team)===work
            F=qr!(A, ColumnNorm(); scratch = work, team)
            @test Matrix(F.factors)==_packed(seq)
        end
        with_lu_tuning(; full_par_threshold = FaerParThreshold(:never)) do
            S=FaerMatrix(X[1:120, :])
            @test _packed(full_piv_lu(S; team))==_packed(full_piv_lu(FaerMatrix(X[1:120, :])))
            req=scratch_requirements(lu!, S; team)
            @test req.bytes>=0
        end
        close(team)
    end

    @testset "validation and restoration" begin
        @test_throws ArgumentError with_lu_tuning(() -> nothing; partial_recursion_threshold = 0)
        @test_throws ArgumentError with_lu_tuning(() -> nothing; partial_recursion_threshold = 1.5)
        @test_throws ArgumentError with_lu_tuning(() -> nothing; partial_par_threshold = 3)
        @test_throws ArgumentError with_lu_tuning(() -> nothing; full_par_threshold = :never)
        @test_throws ArgumentError with_qr_tuning(() -> nothing; blocking_threshold = -1)
        @test_throws ArgumentError with_qr_tuning(() -> nothing; pivoted_par_threshold = 0)
        @test_throws MethodError with_qr_tuning(() -> nothing; block_size = 4)
        @test_throws ErrorException with_qr_tuning(; blocking_threshold = 5) do
            with_lu_tuning(; partial_recursion_threshold = 2) do
                @test _lq_state()==([2, 0, 0, 5, 0], [1, 0, 0, 1, 0])
                error("boom")
            end
        end
        @test _lq_state()==(zeros(Int, 5), zeros(Int, 5))
        @test with_lu_tuning(() -> 42)==42
    end
end
