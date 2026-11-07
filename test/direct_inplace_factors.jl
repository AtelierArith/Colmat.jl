using Test, Colmat, LinearAlgebra, SparseArrays

_sample(T, m, n) = begin
    X = T.(reshape(1:(m*n), m, n)) ./ 4
    for i = 1:min(m, n)
        X[i, i] += T(10)
    end
    T <: Complex && (X .+= T(0.25im) .* reshape(T.(n*m:-1:1), m, n) ./ (m * n))
    X
end
_tol(T) = sqrt(eps(real(T)))
_hermitian(T, n) = begin
    X = _sample(T, n, n)
    X * X' + n * Matrix{T}(I, n, n)
end
# Hermitian indefinite with a tiny leading pivot (Bunch–Kaufman 2×2 blocks).
_indefinite(T, n) = begin
    H = (_sample(T, n, n) + _sample(T, n, n)') / 8
    for i = 1:n
        H[i, i] = T((1e-6, -3, 2, 0, -1, 4)[mod1(i, 6)])
    end
    H
end
# The selected triangle of S with NaN garbage in the other (S itself for :plain).
_stored(S, uplo) =
    uplo === :plain ? copy(S) :
    uplo === :L ? tril(S) + triu(fill(eltype(S)(NaN), size(S)), 1) :
    triu(S) + tril(fill(eltype(S)(NaN), size(S)), -1)

@testset "Direct in-place dense factorizations" begin
    @testset "LU $T $dims" for T in (Float32, Float64, ComplexF32, ComplexF64),
        dims in ((0, 0), (1, 1), (4, 4), (5, 3), (3, 5))

        m, n = dims
        X = _sample(T, m, n)
        A = FaerMatrix(X)
        v = view(A, :, :)
        work = FaerScratch(lu!, A)
        @test scratch_requirements(lu!, A) == scratch_requirements(lu!, A; team = nothing)
        F = lu!(A; scratch = work)
        @test F isa FaerInPlaceFactorization{T,:lu}
        @test F.factors === A
        @test issuccess(F)
        @test size(F) == dims
        old = lu(FaerMatrix(X))
        @test Matrix(A) ≈ Matrix(old.packed) rtol = _tol(T)
        @test Matrix(v) == Matrix(A)
        @test F.p == old.p
        @test Matrix(F.L) ≈ Matrix(old.L) rtol = _tol(T)
        @test Matrix(F.U) ≈ Matrix(old.U) rtol = _tol(T)
        @test Matrix(F.P) == Matrix(old.P)
        @test Matrix(F.packed) == Matrix(A)
        @test Matrix(F) ≈ X rtol = _tol(T) atol = _tol(T)
        if m == n
            b = T.(collect(1:n))
            B = T.(reshape(1:(2n), n, 2))
            @test F \ b ≈ X \ b rtol = _tol(T)
            @test Matrix(F \ B) ≈ X \ B rtol = _tol(T)
            @test transpose(F) \ b ≈ transpose(X) \ b rtol = _tol(T)
            @test F' \ b ≈ X' \ b rtol = _tol(T)
            n > 0 && @test det(F) ≈ det(X) rtol = _tol(T)
            n > 0 && @test logabsdet(F)[1] ≈ logabsdet(X)[1] rtol = _tol(T)
            @test Matrix(inv(F)) ≈ inv(X) rtol = _tol(T) atol = _tol(T)
            # Solve in place with a caller workspace and a team.
            R = FaerMatrix(B)
            sw = FaerScratch(ldiv!, F, R)
            @test ldiv!(F, R; scratch = sw) === R
            @test Matrix(R) ≈ X \ B rtol = _tol(T)
            team = FaerTeam(2)
            R2 = FaerMatrix(B)
            reserve!(sw, ldiv!, transpose(F), R2; team)
            ldiv!(transpose(F), R2; scratch = sw, team)
            @test Matrix(R2) ≈ transpose(X) \ B rtol = _tol(T)
            close(team)
        end
        G = copy(F)
        @test G.factors !== A
        @test Matrix(G) ≈ Matrix(F)
        close(A)
        @test !isopen(F)
        @test_throws ArgumentError F.L
        @test Matrix(G) ≈ X rtol = _tol(T) atol = _tol(T)
        close(G)
        @test_throws ArgumentError Matrix(G)
    end

    @testset "QR $T $dims" for T in (Float32, Float64, ComplexF32, ComplexF64),
        dims in ((0, 0), (1, 1), (4, 4), (6, 3), (3, 5))

        m, n = dims
        X = _sample(T, m, n)
        A = FaerMatrix(X)
        for F in (
            qr!(A; scratch = FaerScratch(qr!, A)),
            qr!(FaerMatrix(X), NoPivot(); scratch = FaerScratch(qr!, A, NoPivot())),
        )
            @test F isa FaerInPlaceFactorization{T,:qr}
            old = qr(FaerMatrix(X))
            @test Matrix(F.factors) ≈ Matrix(old.packed) rtol = _tol(T)
            @test Matrix(F.Qcoeff) ≈ Matrix(old.Qcoeff) rtol = _tol(T)
            @test Matrix(F.R) ≈ Matrix(old.R) rtol = _tol(T)
            @test Matrix(F.Qbasis) ≈ Matrix(old.Qbasis) rtol = _tol(T)
            @test Matrix(F.Q) ≈ Matrix(old.Q) rtol = _tol(T) atol = _tol(T)
            @test Matrix(F.Qthin) ≈ Matrix(old.Qthin) rtol = _tol(T) atol = _tol(T)
            @test Matrix(F) ≈ X rtol = _tol(T) atol = _tol(T)
            if m >= n
                b = T.(collect(1:m))
                @test F \ b ≈ X \ b rtol = _tol(T) atol = _tol(T)
                B = FaerMatrix(T.(reshape(1:(2m), m, 2)))
                ldiv!(F, B; scratch = FaerScratch(ldiv!, F, B))
                @test Matrix(B)[1:n, :] ≈ X \ T.(reshape(1:(2m), m, 2)) rtol = _tol(T) atol =
                    _tol(T)
            else
                # Wide QR: the minimum-norm solution through the pseudoinverse.
                b = T.(collect(1:m))
                @test F \ b ≈ X \ b rtol = sqrt(eps(real(T))) atol = sqrt(eps(real(T)))
            end
            if m == n
                b = T.(collect(1:n))
                @test transpose(F) \ b ≈ transpose(X) \ b rtol = _tol(T)
                @test F' \ b ≈ X' \ b rtol = _tol(T)
                @test Matrix(inv(F)) ≈ inv(X) rtol = _tol(T) atol = _tol(T)
            end
        end
    end

    @testset "column-pivoted QR $T $dims" for T in (Float32, Float64, ComplexF32, ComplexF64),
        dims in ((0, 0), (1, 1), (4, 4), (6, 3), (3, 5))

        m, n = dims
        X = _sample(T, m, n)
        A = FaerMatrix(X)
        work = FaerScratch(qr!, A, ColumnNorm())
        @test scratch_requirements(qr!, A, ColumnNorm()) ==
              scratch_requirements(qr!, A, ColumnNorm(); team = nothing)
        F = qr!(A, ColumnNorm(); scratch = work)
        @test F isa FaerInPlaceFactorization{T,:pivoted_qr}
        @test F.factors === A
        @test issuccess(F)
        old = qr(FaerMatrix(X), ColumnNorm())
        # The same faer kernel and parameters as the retained factor.
        @test Matrix(F.factors) == Matrix(old.packed)
        @test Matrix(F.Qcoeff) == Matrix(old.Qcoeff)
        @test F.p == old.p
        @test Matrix(F.P) == Matrix{T}(I, n, n)[:, old.p]
        @test Matrix(F.R) == Matrix(old.R)
        @test Matrix(F.Qbasis) == Matrix(old.Qbasis)
        @test Matrix(F.Q) ≈ Matrix(old.Q) rtol = _tol(T) atol = _tol(T)
        @test Matrix(F.Qthin) ≈ Matrix(old.Qthin) rtol = _tol(T) atol = _tol(T)
        @test Matrix(qr_q(F; thin = true)) ≈ Matrix(old.Qthin) rtol = _tol(T) atol = _tol(T)
        @test Matrix(F.Qthin) * Matrix(F.R) ≈ X[:, F.p] rtol = _tol(T) atol = _tol(T)
        @test Matrix(F) ≈ X rtol = _tol(T) atol = _tol(T)
        @test Matrix(reconstruct(F; team = nothing)) ≈ X rtol = _tol(T) atol = _tol(T)
        @test rank(F) == rank(old)
        @test rank(F; atol = 1e3) == 0
        if m >= n
            b = T.(collect(1:m))
            @test F \ b ≈ X \ b rtol = _tol(T) atol = _tol(T)
            B = FaerMatrix(T.(reshape(1:(2m), m, 2)))
            ldiv!(F, B; scratch = FaerScratch(ldiv!, F, B))
            @test Matrix(B)[1:n, :] ≈ X \ T.(reshape(1:(2m), m, 2)) rtol = _tol(T) atol =
                _tol(T)
        else
            b = T.(collect(1:m))
            @test F \ b ≈ X \ b rtol = sqrt(eps(real(T))) atol = sqrt(eps(real(T)))
        end
        if m == n
            b = T.(collect(1:n))
            @test transpose(F) \ b ≈ transpose(X) \ b rtol = _tol(T)
            @test F' \ b ≈ X' \ b rtol = _tol(T)
            @test Matrix(inv(F)) ≈ inv(X) rtol = _tol(T) atol = _tol(T)
        end
        # Determinants come from R's diagonal and the reflection count.
        if m == n
            @test det(F) ≈ det(X) rtol = _tol(T)
        else
            @test_throws DimensionMismatch det(F)
        end
        # Rank-truncated solves run the retained factor's kernel.
        Bm = T.(reshape(1:(2m), m, 2))
        for (atol, rtol) in ((0, 1e-3), (1e3, 0), (0, 0))
            Xd = zeros(T, n, 2)
            Xr = zeros(T, n, 2)
            ldiv!(Xd, F, Bm; atol, rtol)
            ldiv!(Xr, old, Bm; atol, rtol)
            @test Xd == Xr
        end
        Bd = vcat(Bm, zeros(T, max(n - m, 0), 2))
        Br = copy(Bd)
        _, rd = ldiv!(F, Bd, 1e-3)
        _, rr = ldiv!(old, Br, 1e-3)
        @test rd == rr
        @test Bd == Br
        @test Matrix(pinv(F)) ≈ pinv(X) rtol = sqrt(_tol(T)) atol = sqrt(_tol(T))
        G = copy(F)
        @test G.factors !== A
        @test G.p == F.p
        close(A)
        @test !isopen(F)
        @test_throws ArgumentError F.R
        @test_throws ArgumentError rank(F)
        @test Matrix(G) ≈ X rtol = _tol(T) atol = _tol(T)
    end

    @testset "column-pivoted QR rank deficiency and teams" begin
        for T in (Float64, ComplexF32)
            # Rank two: the second column is twice the first.
            X = T[1 2 3; 2 4 6; 1 2 1; 0 0 1; 1 2 0]
            A = FaerMatrix(X)
            F = qr!(A, ColumnNorm(); scratch = FaerScratch(qr!, A, ColumnNorm()))
            old = qr(FaerMatrix(X), ColumnNorm())
            @test rank(F) == rank(old) == 2
            b = T.(collect(1:5))
            x = zeros(T, 3)
            ldiv!(x, F, b; rtol = sqrt(eps(real(T))))
            @test x ≈ pinv(X) * b rtol = sqrt(_tol(T))
            @test Matrix(pinv(F)) ≈ pinv(X) rtol = sqrt(_tol(T)) atol = sqrt(_tol(T))
            @test Matrix(pinv(F; atol = 1e3)) == zeros(T, 3, 5)
            @test_throws ArgumentError ldiv!(x, transpose(F), b; rtol = 1e-3)
            @test_throws ArgumentError ldiv!(x, F, b; rtol = 1e-3, scratch = FaerScratch(64))
            @test_throws ArgumentError rank(F; atol = -1)
            @test_throws DimensionMismatch ldiv!(F, zeros(T, 2), 1e-3)
            # A team lends its workers to construction, solves and parts.
            team = FaerTeam(2)
            Y = _sample(T, 70, 50)
            A = FaerMatrix(Y)
            work = FaerScratch(qr!, A, ColumnNorm(); team)
            @test reserve!(work, qr!, A, ColumnNorm(); team) === work
            F = qr!(A, ColumnNorm(); scratch = work, team)
            old = qr(FaerMatrix(Y), ColumnNorm())
            @test F.p == old.p
            @test Matrix(F.R) ≈ Matrix(old.R) rtol = _tol(T)
            C = T.(reshape(1:140, 70, 2))
            B = FaerMatrix(C)
            ldiv!(F, B; scratch = FaerScratch(ldiv!, F, B; team), team)
            @test Matrix(B)[1:50, :] ≈ Y \ C rtol = _tol(T)
            @test Matrix(FaerMatrix(F; team)) ≈ Y rtol = _tol(T)
            close(team)
        end
        # Validation precedes any write.
        X = rand(4, 3)
        A = FaerMatrix(X)
        @test_throws ArgumentError qr!(A, ColumnNorm(); scratch = FaerScratch(0))
        @test_throws ArgumentError qr!(A, ColumnNorm(); scratch = :bad)
        @test Matrix(A) == X
        Y = copy(X)
        Y[1] = NaN
        B = FaerMatrix(Y)
        @test_throws Exception qr!(B, ColumnNorm(); scratch = FaerScratch(qr!, B, ColumnNorm()))
        @test isequal(Matrix(B), Y)
        # Without scratch the staged path still returns a retained factor.
        @test qr!(FaerMatrix(X), ColumnNorm()) isa FaerFactorization{Float64,:pivoted_qr}
    end

    @testset "Cholesky $T $n $uplo" for T in (Float32, Float64, ComplexF32, ComplexF64),
        n in (0, 1, 5),
        uplo in (:L, :U)

        X = _sample(T, n, n)
        S = X * X' + Matrix{T}(I, n, n)
        A = FaerMatrix(S)
        work = FaerScratch(cholesky!, Hermitian(A, uplo))
        F = cholesky!(Hermitian(A, uplo); scratch = work)
        @test F isa FaerInPlaceFactorization{T,:cholesky}
        @test F.factors === A
        @test F.uplo === uplo
        ref = cholesky(Hermitian(S, uplo))
        @test Matrix(A) ≈ (uplo === :L ? Matrix(ref.L) : Matrix(ref.U)) rtol = _tol(T)
        @test Matrix(F.L) ≈ Matrix(ref.L) rtol = _tol(T)
        @test Matrix(F.U) ≈ Matrix(ref.U) rtol = _tol(T)
        @test Matrix(F.UL) ≈ (uplo === :L ? Matrix(ref.L) : Matrix(ref.U)) rtol = _tol(T)
        @test Matrix(F.packed) ≈ Matrix(ref.L) rtol = _tol(T)
        @test Matrix(F) ≈ S rtol = _tol(T) atol = _tol(T)
        b = T.(collect(1:n))
        @test F \ b ≈ S \ b rtol = _tol(T)
        @test transpose(F) \ b ≈ transpose(S) \ b rtol = _tol(T)
        @test F' \ b ≈ S \ b rtol = _tol(T)
        n > 0 && @test logdet(F) ≈ logdet(S) rtol = _tol(T)
        @test Matrix(inv(F)) ≈ inv(S) rtol = _tol(T) atol = _tol(T)
        @test isposdef(F)
        if n > 0
            G = cholesky!(FaerMatrix(S); scratch = work)
            @test Matrix(G) ≈ S rtol = _tol(T)
        end
    end

    @testset "team construction $T" for T in (Float64, ComplexF32)
        team = FaerTeam(2)
        X = _sample(T, 70, 70)
        S = X * X' + 70 * Matrix{T}(I, 70, 70)
        A, B = FaerMatrix(X), FaerMatrix(X)
        F = lu!(A; scratch = FaerScratch(lu!, A))
        G = lu!(B; scratch = FaerScratch(lu!, B; team), team)
        @test G.p == F.p
        @test Matrix(B) ≈ Matrix(A) rtol = _tol(T)
        A, B = FaerMatrix(X[:, 1:40]), FaerMatrix(X[:, 1:40])
        qr!(A; scratch = FaerScratch(qr!, A))
        work = FaerScratch(0)
        reserve!(work, qr!, B; team)
        qr!(B; scratch = work, team)
        @test Matrix(B) ≈ Matrix(A) rtol = _tol(T)
        for uplo in (:L, :U)
            A, B = FaerMatrix(S), FaerMatrix(S)
            cholesky!(Hermitian(A, uplo); scratch = FaerScratch(cholesky!, A))
            H = cholesky!(
                Hermitian(B, uplo);
                scratch = FaerScratch(cholesky!, Hermitian(B, uplo); team),
                team,
            )
            @test Matrix(B) ≈ Matrix(A) rtol = _tol(T)
            @test Matrix(H) ≈ S rtol = _tol(T)
        end
        I70 = _indefinite(T, 70)
        for (f, M, args, input) in (
            (full_piv_lu!, X, (), identity),
            (ldlt!, S, (), M -> Hermitian(M, :U)),
            (bunchkaufman!, I70, (true,), M -> Hermitian(M, :L)),
        )
            A, B = FaerMatrix(M), FaerMatrix(M)
            F = f(input(A), args...; scratch = FaerScratch(f, input(A), args...))
            work = FaerScratch(0)
            reserve!(work, f, input(B), args...; team)
            G = f(input(B), args...; scratch = work, team)
            @test Matrix(B) ≈ Matrix(A) rtol = _tol(T)
            f === ldlt! || @test G.p == F.p
            f === full_piv_lu! && @test G.q == F.q
            @test Matrix(G) ≈ M rtol = _tol(T) atol = _tol(T)
            seq = scratch_requirements(f, input(B), args...)
            par = scratch_requirements(f, input(B), args...; team)
            if par.bytes > seq.bytes
                C = FaerMatrix(M)
                small = FaerScratch(f, input(C), args...)
                @test_throws ArgumentError f(input(C), args...; scratch = small, team)
                @test Matrix(C) == M
            end
        end
        close(team)
        @test_throws ArgumentError lu!(FaerMatrix(X); scratch = FaerScratch(lu!, A), team)
    end

    @testset "complex right-hand sides of a real factor" begin
        X = _sample(Float64, 4, 4)
        F = lu!(FaerMatrix(X); scratch = FaerScratch(lu!, FaerMatrix(X)))
        b = ComplexF64[1 + 2im, 3, -1im, 4]
        B = ComplexF64.(reshape(1:8, 4, 2)) .* (1 + 1im)
        @test F \ b ≈ X \ b
        @test Matrix(F \ B) ≈ X \ B
        @test transpose(F) \ b ≈ transpose(X) \ b
        @test F' \ b ≈ X' \ b
    end

    @testset "aliasing, failures and validation" begin
        X = _sample(Float64, 4, 4)
        A = FaerMatrix(X)
        work = FaerScratch(lu!, A)
        F = lu!(A; scratch = work)
        b = collect(1.0:4.0)
        x = F \ b
        # F aliases A: writes to A change F, as for Julia's lu!.
        A[1, 1] = A[1, 1] * 2
        @test !(F \ b ≈ x)
        # A right-hand side sharing F's storage is rejected.
        @test_throws ArgumentError ldiv!(F, A)
        # Resizing A invalidates F.
        resize!(A, 5, 5)
        @test !isopen(F)
        @test_throws ArgumentError F \ b

        # Validation failures leave A untouched.
        A = FaerMatrix(X)
        empty = FaerScratch(0)
        if scratch_requirements(lu!, A).bytes > 0
            @test_throws ArgumentError lu!(A; scratch = empty)
            @test Matrix(A) == X
        end
        @test_throws ArgumentError lu!(A; scratch = :bad)
        @test_throws ArgumentError lu!(A; scratch = work, team = :bad)
        @test Matrix(A) == X
        closed = FaerScratch(lu!, A)
        close(closed)
        @test_throws ArgumentError lu!(A; scratch = closed)
        @test Matrix(A) == X

        # Singular LU follows Julia: check throws, allowsingular returns info.
        Z = FaerMatrix([1.0 2; 2 4])
        @test_throws SingularException lu!(Z; scratch = FaerScratch(lu!, Z))
        Z = FaerMatrix([1.0 2; 2 4])
        G = lu!(Z; allowsingular = true, scratch = FaerScratch(lu!, Z))
        @test !issuccess(G)
        @test logabsdet(G)[1] == -Inf
        @test_throws SingularException G \ [1.0, 2.0]

        # Non-Hermitian input is rejected before any write.
        N = FaerMatrix([1.0 2; 3 4])
        nw = FaerScratch(cholesky!, N)
        @test_throws PosDefException cholesky!(N; scratch = nw)
        H = cholesky!(N; check = false, scratch = nw)
        @test H.info == -1
        @test Matrix(N) == [1.0 2; 3 4]
        # Indefinite input: numerical failure, as with LAPACK.
        P = FaerMatrix([1.0 2; 2 1])
        @test_throws PosDefException cholesky!(P; scratch = nw)
        P = FaerMatrix([1.0 2; 2 1])
        H = cholesky!(P; check = false, scratch = nw)
        @test !issuccess(H)
        @test H.info > 0

        # Complex Symmetric with a real stored triangle is Hermitian.
        S = ComplexF64[4 1; 1 3]
        C = FaerMatrix(S)
        K = cholesky!(Symmetric(C); scratch = FaerScratch(cholesky!, C))
        @test K isa FaerInPlaceFactorization
        @test Matrix(K) ≈ S

        # Without scratch the retained, independent path is unchanged.
        A = FaerMatrix(X)
        R = lu!(A)
        @test R isa FaerFactorization
        close(A)
        @test Matrix(R) ≈ X
        @test qr!(FaerMatrix(X), ColumnNorm()) isa FaerFactorization
    end

    @testset "full-pivot LU $T $dims" for T in (Float32, Float64, ComplexF32, ComplexF64),
        dims in ((0, 0), (1, 1), (5, 5), (6, 3), (3, 5))

        m, n = dims
        X = _sample(T, m, n)
        A = FaerMatrix(X)
        work = FaerScratch(full_piv_lu!, A)
        F = full_piv_lu!(A; scratch = work)
        @test F isa FaerInPlaceFactorization{T,:full_lu}
        @test F.factors === A
        @test issuccess(F)
        old = full_piv_lu(FaerMatrix(X))
        @test Matrix(A) ≈ Matrix(old.packed) rtol = _tol(T)
        @test F.p == old.p
        @test F.q == old.q
        @test Matrix(F.P) == Matrix(old.P)
        @test Matrix(F.L) ≈ Matrix(old.L) rtol = _tol(T)
        @test Matrix(F.U) ≈ Matrix(old.U) rtol = _tol(T)
        @test Matrix(F) ≈ X rtol = _tol(T) atol = _tol(T)
        @test X[F.p, F.q] ≈ Matrix(F.L) * Matrix(F.U) rtol = _tol(T) atol = _tol(T)
        @test :q in propertynames(F)
        if m == n
            b = T.(collect(1:n))
            @test F \ b ≈ X \ b rtol = _tol(T)
            @test transpose(F) \ b ≈ transpose(X) \ b rtol = _tol(T)
            @test F' \ b ≈ X' \ b rtol = _tol(T)
            if n > 0
                @test det(F) ≈ det(X) rtol = _tol(T)
                @test all(logabsdet(F) .≈ logabsdet(X))
            end
            @test Matrix(inv(F)) ≈ inv(X) rtol = _tol(T) atol = _tol(T)
            R = FaerMatrix(T.(reshape(1:(2n), n, 2)))
            team = FaerTeam(2)
            sw = FaerScratch(ldiv!, F', R; team)
            @test ldiv!(F', R; scratch = sw, team) === R
            @test Matrix(R) ≈ X' \ T.(reshape(1:(2n), n, 2)) rtol = _tol(T)
            close(team)
        end
        @test Matrix(copy(F)) ≈ X rtol = _tol(T) atol = _tol(T)
    end

    @testset "LDLᴴ $T n=$n $uplo" for T in (Float32, Float64, ComplexF32, ComplexF64),
        n in (0, 1, 6),
        uplo in (:L, :U, :plain)

        S = _hermitian(T, n)
        # Garbage in the unread triangle must be ignored by the wrappers.
        A = FaerMatrix(_stored(S, uplo))
        input = uplo === :plain ? A : Hermitian(A, uplo)
        work = FaerScratch(ldlt!, input)
        @test scratch_requirements(ldlt!, input) == scratch_requirements(ldlt!, A)
        F = ldlt!(input; scratch = work)
        @test F isa FaerInPlaceFactorization{T,:ldlt}
        @test F.factors === A
        ref = ldlt(Hermitian(FaerMatrix(S), uplo === :U ? :U : :L))
        @test Matrix(F.L) ≈ Matrix(ref.L) rtol = _tol(T)
        @test Matrix(F.D) ≈ Matrix(ref.D) rtol = _tol(T)
        @test Matrix(F.packed) ≈ Matrix(ref.packed) rtol = _tol(T)
        packed = Matrix(ref.packed)
        @test Matrix(A) ≈ (uplo === :U ? copy(packed') : packed) rtol = _tol(T)
        @test Matrix(F) ≈ S rtol = _tol(T) atol = _tol(T)
        b = T.(collect(1:n))
        @test F \ b ≈ S \ b rtol = _tol(T)
        @test transpose(F) \ b ≈ transpose(S) \ b rtol = _tol(T)
        @test F' \ b ≈ S \ b rtol = _tol(T)
        n > 0 && @test logdet(F) ≈ logdet(S) rtol = _tol(T)
        n > 0 && @test det(F) ≈ det(S) rtol = _tol(T)
        @test Matrix(inv(F)) ≈ inv(S) rtol = _tol(T) atol = _tol(T)
    end

    @testset "Bunch–Kaufman $T rook=$rook $uplo" for T in (
            Float32,
            Float64,
            ComplexF32,
            ComplexF64,
        ),
        rook in (false, true),
        uplo in (:L, :U, :plain)

        n = 7
        S = _indefinite(T, n)
        A = FaerMatrix(_stored(S, uplo))
        input = uplo === :plain ? A : Hermitian(A, uplo)
        work =
            rook ? FaerScratch(bunchkaufman!, input, true) :
            FaerScratch(bunchkaufman!, input)
        F = bunchkaufman!(input, rook; scratch = work)
        @test F isa FaerInPlaceFactorization{T,:bunchkaufman}
        @test F.factors === A
        @test F.rook == rook
        ref = bunchkaufman(Hermitian(FaerMatrix(S), uplo === :U ? :U : :L), rook)
        @test F.p == ref.p
        @test Matrix(F.P) == Matrix(ref.P)
        @test Matrix(F.L) ≈ Matrix(ref.L) rtol = _tol(T)
        @test Matrix(F.D) ≈ Matrix(ref.D) rtol = _tol(T) atol = _tol(T)
        @test Matrix(F.packed) ≈ Matrix(ref.packed) rtol = _tol(T)
        @test Matrix(F) ≈ S rtol = _tol(T) atol = _tol(T)
        @test inertia(F) == inertia(ref)
        b = T.(collect(1:n))
        @test F \ b ≈ S \ b rtol = _tol(T)
        @test transpose(F) \ b ≈ transpose(S) \ b rtol = _tol(T)
        @test F' \ b ≈ S \ b rtol = _tol(T)
        R = FaerMatrix(T.(reshape(1:(2n), n, 2)))
        @test ldiv!(F, R; scratch = FaerScratch(ldiv!, F, R)) === R
        @test Matrix(R) ≈ S \ T.(reshape(1:(2n), n, 2)) rtol = _tol(T)
        @test det(F) ≈ det(S) rtol = _tol(T)
        @test all(logabsdet(F) .≈ logabsdet(S))
        @test Matrix(inv(F)) ≈ inv(S) rtol = _tol(T) atol = _tol(T)
        G = copy(F)
        @test G.rook == rook
        @test Matrix(G) ≈ S rtol = _tol(T) atol = _tol(T)
    end

    @testset "self-adjoint direct failures and validation" begin
        # Non-Hermitian plain input is rejected before any write.
        N = [1.0 2; 3 4]
        A = FaerMatrix(N)
        @test_throws ArgumentError ldlt!(A; scratch = FaerScratch(ldlt!, A))
        @test_throws ArgumentError bunchkaufman!(A; scratch = FaerScratch(bunchkaufman!, A))
        @test Matrix(A) == N
        # Insufficient scratch leaves A untouched.
        S = _hermitian(Float64, 6)
        A = FaerMatrix(S)
        for (f, input) in ((ldlt!, A), (bunchkaufman!, A), (full_piv_lu!, A))
            scratch_requirements(f, input).bytes > 0 || continue
            @test_throws ArgumentError f(input; scratch = FaerScratch(0))
            @test Matrix(A) == S
        end
        @test_throws DimensionMismatch ldlt!(Hermitian(FaerMatrix(ones(2, 3)));
            scratch = FaerScratch(0))
        # Singular inputs follow the default paths' `check`.
        Z = [1.0 2; 2 4]
        @test_throws SingularException full_piv_lu!(FaerMatrix(Z);
            scratch = FaerScratch(full_piv_lu!, FaerMatrix(Z)))
        zw = FaerScratch(full_piv_lu!, FaerMatrix(Z))
        G = full_piv_lu!(FaerMatrix(Z); check = false, scratch = zw)
        @test !issuccess(G)
        @test logabsdet(G) == (-Inf, 0.0)
        Zb = zeros(3, 3)
        @test_throws SingularException bunchkaufman!(FaerMatrix(Zb);
            scratch = FaerScratch(bunchkaufman!, FaerMatrix(Zb)))
        H = bunchkaufman!(FaerMatrix(Zb); check = false,
            scratch = FaerScratch(bunchkaufman!, FaerMatrix(Zb)))
        @test H.info == bunchkaufman(FaerMatrix(Zb); check = false).info > 0
        @test det(H) == 0
        lw = FaerScratch(ldlt!, FaerMatrix(Zb))
        @test_throws SingularException ldlt!(FaerMatrix(Zb); scratch = lw)
        L = ldlt!(FaerMatrix(Zb); check = false, scratch = lw)
        @test L.info == ldlt(FaerMatrix(Zb); check = false).info > 0
        @test_throws SingularException det(L)
        # Complex Symmetric with a real stored triangle is Hermitian; otherwise rejected.
        C = FaerMatrix(ComplexF64[4 1; 1 -3])
        K = bunchkaufman!(Symmetric(C); scratch = FaerScratch(bunchkaufman!, C))
        @test Matrix(K) ≈ ComplexF64[4 1; 1 -3]
        Cn = FaerMatrix(ComplexF64[4 1im; 1im 3])
        @test_throws ArgumentError ldlt!(Symmetric(Cn); scratch = FaerScratch(ldlt!, Cn))
        # Without scratch the retained path is unchanged.
        @test ldlt!(FaerMatrix(S)) isa FaerFactorization
        @test bunchkaufman!(FaerMatrix(S), true) isa FaerFactorization
        @test full_piv_lu!(FaerMatrix(S)) isa FaerFactorization
    end

    # Right solves and the destructive inverse of every direct factor kind.
    _direct_cases(T, n) = (
        (:lu, _sample(T, n, n), A -> lu!(A; scratch = FaerScratch(lu!, A))),
        (:full_lu, _sample(T, n, n), A -> full_piv_lu!(A; scratch = FaerScratch(full_piv_lu!, A))),
        (:qr, _sample(T, n, n), A -> qr!(A; scratch = FaerScratch(qr!, A))),
        (:pivoted_qr, _sample(T, n, n),
            A -> qr!(A, ColumnNorm(); scratch = FaerScratch(qr!, A, ColumnNorm()))),
        (:cholesky_L, _hermitian(T, n),
            A -> cholesky!(Hermitian(A, :L); scratch = FaerScratch(cholesky!, A))),
        (:cholesky_U, _hermitian(T, n),
            A -> cholesky!(Hermitian(A, :U); scratch = FaerScratch(cholesky!, A))),
        (:ldlt, _hermitian(T, n),
            A -> ldlt!(Hermitian(A, :U); scratch = FaerScratch(ldlt!, A))),
        (:bunchkaufman, _indefinite(T, n),
            A -> bunchkaufman!(Hermitian(A, :U); scratch = FaerScratch(bunchkaufman!, A))),
        (:bunchkaufman_rook, _indefinite(T, n),
            A -> bunchkaufman!(Hermitian(A, :L), true;
                scratch = FaerScratch(bunchkaufman!, A, true))),
    )
    @testset "right solves and inv! $name $T" for T in (Float32, Float64, ComplexF32, ComplexF64),
        n in (0, 5), (name, S, make) in _direct_cases(T, 5)

        S = S[1:n, 1:n]
        stored =
            name in (:cholesky_U, :ldlt, :bunchkaufman) ? _stored(S, :U) :
            name in (:cholesky_L, :bunchkaufman_rook) ? _stored(S, :L) : S
        A = FaerMatrix(stored)
        F = make(A)
        B = T.(reshape(1:(3n), 3, n)) ./ 3
        for (G, M) in ((F, S), (transpose(F), transpose(S)), (F', S'))
            X = FaerMatrix(B)
            work = FaerScratch(rdiv!, X, G)
            @test scratch_requirements(rdiv!, X, G) ==
                  scratch_requirements(rdiv!, X, G; team = nothing)
            @test rdiv!(X, G; scratch = work) === X
            @test Matrix(X) ≈ B / M rtol = _tol(T) atol = _tol(T)
            Y = B / G
            @test Y isa FaerMatrix{T}
            @test Matrix(Y) ≈ B / M rtol = _tol(T) atol = _tol(T)
            team = FaerTeam(2)
            Z = FaerMatrix(B)
            @test reserve!(work, rdiv!, Z, G; team) === work
            rdiv!(Z, G; scratch = work, team)
            @test Matrix(Z) ≈ B / M rtol = _tol(T) atol = _tol(T)
            close(team)
            # Without scratch a temporary workspace is used.
            @test Matrix(rdiv!(FaerMatrix(B), G)) ≈ B / M rtol = _tol(T) atol = _tol(T)
        end
        # A right-hand side sharing F's storage or of another eltype is rejected.
        @test_throws ArgumentError rdiv!(A, F)
        @test_throws ArgumentError rdiv!(FaerMatrix(zeros(T === Float64 ? Float32 : Float64, 3, n)), F)
        @test_throws DimensionMismatch rdiv!(FaerMatrix(zeros(T, 2, n + 1)), F)
        # inv! overwrites F.factors with inv(A) (both triangles for the
        # self-adjoint kinds, as Julia's inv!(::Cholesky)) and invalidates F.
        expected = n == 0 ? zeros(T, 0, 0) : inv(S)
        work = FaerScratch(LinearAlgebra.inv!, F)
        @test scratch_requirements(LinearAlgebra.inv!, F).bytes >= n * n * sizeof(T)
        @test LinearAlgebra.inv!(F; scratch = work) === A
        @test Matrix(A) ≈ expected rtol = _tol(T) atol = _tol(T)
        @test !isopen(F)
        @test_throws ArgumentError F \ ones(T, n)
        @test_throws ArgumentError LinearAlgebra.inv!(F)
        # The temporary-workspace and team forms.
        A2 = FaerMatrix(stored)
        F2 = make(A2)
        team = FaerTeam(2)
        reserve!(work, LinearAlgebra.inv!, F2; team)
        @test LinearAlgebra.inv!(F2; scratch = work, team) === A2
        @test Matrix(A2) ≈ expected rtol = _tol(T) atol = _tol(T)
        close(team)
        A3 = FaerMatrix(Matrix(A2))
        LinearAlgebra.inv!(A3)
        @test Matrix(A3) ≈ S rtol = sqrt(_tol(T)) atol = sqrt(_tol(T))
    end
    @testset "inv! of a general matrix and failures" begin
        for T in (Float64, ComplexF32)
            X = _sample(T, 4, 4)
            A = FaerMatrix(X)
            work = FaerScratch(LinearAlgebra.inv!, A)
            @test scratch_requirements(LinearAlgebra.inv!, A).bytes >=
                  scratch_requirements(lu!, A).bytes
            @test LinearAlgebra.inv!(A; scratch = work) === A
            @test Matrix(A) ≈ inv(X) rtol = _tol(T)
        end
        # Singular inputs throw before any write.
        Z = [1.0 2; 2 4]
        A = FaerMatrix(Z)
        @test_throws SingularException LinearAlgebra.inv!(A)
        G = lu!(FaerMatrix(Z); allowsingular = true, scratch = FaerScratch(lu!, FaerMatrix(Z)))
        packed = Matrix(G.factors)
        @test_throws SingularException LinearAlgebra.inv!(G)
        @test Matrix(G.factors) == packed
        @test isopen(G)
        Q = qr!(FaerMatrix(Z); scratch = FaerScratch(qr!, FaerMatrix(Z)))
        @test_throws SingularException LinearAlgebra.inv!(Q)
        W = FaerMatrix(rand(3, 4))
        Fw = lu!(W; scratch = FaerScratch(lu!, W))
        @test_throws DimensionMismatch LinearAlgebra.inv!(Fw)
        @test_throws DimensionMismatch rdiv!(FaerMatrix(rand(2, 4)), Fw)
        @test_throws DimensionMismatch LinearAlgebra.inv!(FaerMatrix(rand(3, 4)))
        X = _sample(Float64, 4, 4)
        A = FaerMatrix(X)
        F = lu!(A; scratch = FaerScratch(lu!, A))
        @test_throws ArgumentError LinearAlgebra.inv!(F; scratch = FaerScratch(0))
        @test_throws ArgumentError rdiv!(FaerMatrix(ones(2, 4)), F; scratch = :bad)
        @test isopen(F)
        @test Matrix(F) ≈ X
    end

    @test isempty(
        Test.detect_ambiguities(Colmat, Base, LinearAlgebra, SparseArrays; recursive = true),
    )
end
