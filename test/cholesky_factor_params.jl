using Test, Colmat, LinearAlgebra

const PARAM_TYPES=(Float32, Float64, ComplexF32, ComplexF64)
_ptol(::Type{T}) where {T} = real(T)===Float32 ? 1e-3 : 1e-10
_tuning_state() = (
    v = zeros(UInt, 6);
    f = zeros(UInt8, 6);
    Colmat._native_call(Colmat.RustCrate.selfadjoint_tuning_state, v, f);
    (v, Int.(f))
)
function _hpd(::Type{T}, n) where {T}
    X=T[sin(i+2j)+(T<:Complex ? im*cos(i*j) : 0) for i = 1:n, j = 1:n]
    return Matrix(Hermitian(X*X'+T(n)*I))
end
function _hindef(::Type{T}, n) where {T}
    H=_hpd(T, n)
    for i = 1:3:n
        H[i, i]=-H[i, i]
    end
    return H
end
# Diagonally dominant except for a negative first pivot: regularization
# replaces exactly that pivot (faer's own LLᴴ regularization breaks here).
function _first_negative(::Type{T}, n) where {T}
    H=fill(T(0.5), n, n)
    for i = 1:n
        H[i, i]=i==1 ? -3 : 4
    end
    return H
end
_offdiag(A) = A-Diagonal(diag(A))

@testset "with_cholesky_tuning" begin
    @test _tuning_state()==(zeros(Int, 6), zeros(Int, 6))
    for T in PARAM_TYPES, n in (0, 5, 150)
        S=_hpd(T, n)
        Si=_hindef(T, n)
        A=FaerMatrix(S)
        Ai=FaerMatrix(Si)
        tol=_ptol(T)
        b=ones(T, n)
        tuned=(
            llt_recursion_threshold = 3,
            llt_block_size = 5,
            ldlt_recursion_threshold = 2,
            ldlt_block_size = 7,
            lblt_block_size = 4,
            lblt_par_threshold = FaerParThreshold(:never),
        )
        F, G, H=with_cholesky_tuning(; tuned...) do
            @test _tuning_state()==([3, 5, 2, 7, 4, typemax(UInt)], ones(Int, 6))
            cholesky(Hermitian(A, :U)), ldlt(A), bunchkaufman(Ai)
        end
        @test _tuning_state()==(zeros(Int, 6), zeros(Int, 6))
        @test Matrix(F)≈S rtol=tol
        @test Matrix(F.L)≈Matrix(cholesky(A).L) rtol=tol atol=tol
        @test Matrix(G)≈S rtol=tol
        @test Matrix(H)≈Si rtol=tol atol=tol
        @test S*(F\b)≈b rtol=tol atol=tol
        # Staged and direct in-place paths, with scratch queried in the scope.
        with_cholesky_tuning(; tuned...) do
            B=copy(A)
            W=cholesky!(B)
            @test Matrix(W)≈S rtol=tol
            for (fn, input, args) in (
                (cholesky!, S, ()),
                (ldlt!, S, ()),
                (bunchkaufman!, Si, (true,)),
            )
                C=FaerMatrix(input)
                work=FaerScratch(fn, C, args...)
                D=fn(C, args...; scratch = work)
                @test D isa FaerInPlaceFactorization
                @test Matrix(D)≈input rtol=tol atol=tol
            end
            team=FaerTeam(2)
            try
                @test Matrix(ldlt(A; team))≈S rtol=tol
            finally
                close(team)
            end
        end
    end
    # The LBLᴴ block size changes the direct workspace; LLᴴ/LDLᴴ's do not.
    A=FaerMatrix(_hindef(Float64, 200))
    base=scratch_requirements(bunchkaufman!, A).bytes
    blocked=with_cholesky_tuning(() -> scratch_requirements(bunchkaufman!, A).bytes; lblt_block_size = 128)
    unblocked=with_cholesky_tuning(() -> scratch_requirements(bunchkaufman!, A).bytes; lblt_block_size = 1)
    @test unblocked<base<blocked
    @test with_cholesky_tuning(() -> scratch_requirements(ldlt!, A); ldlt_block_size = 1)==
          scratch_requirements(ldlt!, A)
    # Nested scopes restore the enclosing settings, also after an error.
    with_cholesky_tuning(; llt_block_size = 9) do
        @test_throws ErrorException with_cholesky_tuning(; ldlt_block_size = 3) do
            # A scope replaces, rather than merges, the enclosing settings.
            @test _tuning_state()[1][1:4]==[0, 0, 0, 3]
            error("boom")
        end
        @test _tuning_state()==([0, 9, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0])
    end
    @test _tuning_state()==(zeros(Int, 6), zeros(Int, 6))
    for bad in (
        (; llt_block_size = 0),
        (; ldlt_recursion_threshold = 0),
        (; llt_recursion_threshold = -1),
        (; lblt_block_size = -1),
        (; lblt_par_threshold = 3),
        (; ldlt_block_size = 2.5),
    )
        @test_throws ArgumentError with_cholesky_tuning(() -> nothing; bad...)
    end
    @test _tuning_state()==(zeros(Int, 6), zeros(Int, 6))
end

@testset "Cholesky dynamic regularization" begin
    for T in PARAM_TYPES, n in (2, 64, 200)
        S=_first_negative(T, n)
        tol=_ptol(T)*n
        A=FaerMatrix(S)
        @test !issuccess(cholesky(A; check = false))
        @test cholesky(A; check = false).regularization_count==0
        reg=(regularization_delta = 1, regularization_epsilon = 1e-6)
        F=cholesky(A; reg...)
        @test issuccess(F)
        @test F.regularization_count==1
        @test :regularization_count in propertynames(F)
        @test copy(F).regularization_count==1
        L=Matrix(F.L)
        @test L[1, 1]≈1
        R=L*L'
        @test _offdiag(R)≈_offdiag(S) rtol=tol atol=tol
        @test diag(R)≈[1; diag(S)[2:end]] rtol=tol
        @test Matrix(F)≈R rtol=tol atol=tol
        @test cholesky(Hermitian(FaerMatrix(Matrix(S')), :U); reg...).regularization_count==1
        # Direct and staged in-place paths and borrowed input agree.
        B=FaerMatrix(S)
        D=cholesky!(B; scratch = FaerScratch(cholesky!, B), reg...)
        @test D.regularization_count==1
        @test Matrix(D.L)≈L rtol=tol atol=tol
        C=FaerMatrix(S)
        @test cholesky!(Hermitian(C, :L); reg...).regularization_count==1
        @test Matrix(C)≈L rtol=tol atol=tol
        @test Matrix(cholesky(faerview(copy(S)); reg...).L)≈L rtol=tol atol=tol
        team=FaerTeam(2)
        try
            @test cholesky(A; team, reg...).regularization_count==1
        finally
            close(team)
        end
        # Without regularization the next factor reports zero again.
        @test cholesky(FaerMatrix(_hpd(T, n))).regularization_count==0
    end
    A=FaerMatrix(_first_negative(Float64, 4))
    @test_throws ArgumentError cholesky(A; regularization_delta = 1)
    @test_throws ArgumentError cholesky(A; regularization_epsilon = 1e-8)
    @test_throws ArgumentError cholesky(A; regularization_delta = -1, regularization_epsilon = 1)
    @test_throws ArgumentError cholesky(A; regularization_delta = Inf, regularization_epsilon = 1)
    @test_throws MethodError cholesky(A; regularization_signs = [1, 1, 1, 1])
    @test_throws MethodError cholesky(FaerMatrix(ComplexF64.(Matrix(A))); bogus = 1)
end

@testset "LDLᴴ dynamic regularization" begin
    for T in PARAM_TYPES
        S=T[0 1; 1 0]
        A=FaerMatrix(S)
        @test !issuccess(ldlt(A; check = false))
        reg=(regularization_delta = 1, regularization_epsilon = 1e-6)
        # Without signs faer replaces the zero pivot but does not count it.
        F=ldlt(A; reg...)
        @test issuccess(F)
        @test real(Matrix(F.D)[1, 1])≈1
        @test F.regularization_count==0
        R=Matrix(F.L)*Matrix(F.D)*Matrix(F.L)'
        @test _offdiag(R)≈_offdiag(S)
        G=ldlt(A; reg..., regularization_signs = [1, -1])
        @test G.regularization_count==1
        @test real.(diag(Matrix(G.D)))≈[1, -1]
        D=ldlt!(FaerMatrix(S); scratch = FaerScratch(ldlt!, A), reg..., regularization_signs = Int8[1, -1])
        @test D.regularization_count==1
        @test Matrix(D.D)≈Matrix(G.D)
        @test ldlt!(Hermitian(FaerMatrix(S), :U); reg..., regularization_signs = [1, -1]).regularization_count==1
    end
    A=FaerMatrix([0.0 1; 1 0])
    reg=(regularization_delta = 1, regularization_epsilon = 1e-6)
    @test_throws DimensionMismatch ldlt(A; reg..., regularization_signs = [1])
    @test_throws ArgumentError ldlt(A; reg..., regularization_signs = [1, 2])
    @test_throws ArgumentError ldlt(A; regularization_signs = [1, -1])
    # Invalid signs are rejected before the direct path writes A.
    B=FaerMatrix([0.0 1; 1 0])
    @test_throws DimensionMismatch ldlt!(B; scratch = FaerScratch(ldlt!, B), reg..., regularization_signs = [1])
    @test Matrix(B)==[0.0 1; 1 0]
end

@testset "Bunch-Kaufman pivoting strategies" begin
    strategies=(:partial, :partial_diag, :rook, :rook_diag, :full)
    for T in PARAM_TYPES, n in (0, 1, 7, 90)
        S=_hindef(T, n)
        tol=_ptol(T)*max(n, 1)
        A=FaerMatrix(S)
        b=ones(T, n)
        for p in strategies
            F=bunchkaufman(Hermitian(A, :U); pivoting = p)
            @test F.rook===(p in (:rook, :rook_diag))
            @test copy(F).rook===F.rook
            @test Matrix(F)≈S rtol=tol atol=tol
            @test Matrix(F.L*F.D*F.L')≈S[F.p, F.p] rtol=tol atol=tol
            @test S*(F\b)≈b rtol=tol atol=tol
            @test inertia(F)==inertia(bunchkaufman(Hermitian(S), true))
            work=FaerScratch(bunchkaufman!, A; pivoting = p)
            B=FaerMatrix(S)
            D=bunchkaufman!(B, p===:rook; scratch = work, pivoting = p)
            @test D.rook===F.rook
            @test D.p==F.p
            @test Matrix(D)≈S rtol=tol atol=tol
            C=FaerMatrix(S)
            @test bunchkaufman!(C; pivoting = p).p==F.p
            @test bunchkaufman(faerview(copy(S)); pivoting = p).p==F.p
        end
        @test bunchkaufman(A; pivoting = :partial_diag).p==bunchkaufman(A).p
        @test bunchkaufman(A; pivoting = :rook).p==bunchkaufman(A, true).p
        @test bunchkaufman(A, true; pivoting = :rook_diag).rook
    end
    A=FaerMatrix(_hindef(Float64, 200))
    # The query takes the strategy (faer's scratch takes the whole LbltParams).
    @test scratch_requirements(bunchkaufman!, A; pivoting = :full).bytes>=0
    @test_throws ArgumentError bunchkaufman(A, true; pivoting = :partial)
    @test_throws ArgumentError bunchkaufman(A; pivoting = :diagonal)
    @test_throws ArgumentError bunchkaufman(A; pivoting = "rook")
    @test_throws ArgumentError scratch_requirements(bunchkaufman!, A; pivoting = :bogus)
    # The lower-oriented factor has no Julia `uplo='U'` `F.U`.
    @test :U ∉ propertynames(bunchkaufman(Hermitian(A, :U)))
end

@testset "Promoted solves of parameterized factors" begin
    # Mixed-scalar solves re-factor the reconstruction (the regularized matrix)
    # at the promoted type; the LBLᴴ strategy is not carried, the solution is.
    n=40
    b=Float32.(1:n)
    S=_first_negative(Float32, n)
    reg=(regularization_delta = 1, regularization_epsilon = 1e-6)
    for F in (
        cholesky(FaerMatrix(S); reg...),
        ldlt(FaerMatrix(S); reg..., regularization_signs = ones(Int, n)),
        bunchkaufman(FaerMatrix(_hindef(Float32, n)); pivoting = :full),
    )
        x=Vector(F\b)
        @test Vector(F\ComplexF32.(b))≈x rtol=1e-3
    end
end
