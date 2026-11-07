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
                @test_throws DimensionMismatch F \ T.(collect(1:m))
            end
            if m == n
                b = T.(collect(1:n))
                @test transpose(F) \ b ≈ transpose(X) \ b rtol = _tol(T)
                @test F' \ b ≈ X' \ b rtol = _tol(T)
                @test Matrix(inv(F)) ≈ inv(X) rtol = _tol(T) atol = _tol(T)
            end
        end
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

    @test isempty(
        Test.detect_ambiguities(Colmat, Base, LinearAlgebra, SparseArrays; recursive = true),
    )
end
