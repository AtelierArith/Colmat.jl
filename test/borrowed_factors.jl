using Colmat, LinearAlgebra, SparseArrays, Statistics, Test

const BF_TYPES = (Float32, Float64, ComplexF32, ComplexF64)
function _bf_matrix(::Type{T}, n; hermitian=false) where {T}
    a = T.([
        1 / (i + j) + (i == j) * n + 0.3(i > j) + (T <: Complex ? im * sin(i - 2j) : 0) for
        i in 1:n, j in 1:n
    ])
    return hermitian ? (a + a') / 2 : a
end
_bf_tol(T) = 1000eps(real(T))

@testset "Borrowed inputs factor natively without staging" begin
    for T in BF_TYPES
        n = 6
        a = _bf_matrix(T, n)
        h = _bf_matrix(T, n; hermitian=true)
        b = T.(reshape(1:(2n), n, 2)) .+ T(0.25)
        before = copy(a)
        B = faerview(a)
        H = faerview(h)
        tol = _bf_tol(T)
        for (F, ref) in (
            (lu(B), a),
            (lu(B, RowMaximum()), a),
            (full_piv_lu(B), a),
            (qr(B), a),
            (qr(B, NoPivot()), a),
            (qr(B, ColumnNorm()), a),
            (cholesky(Hermitian(H)), h),
            (cholesky(Hermitian(H, :U)), h),
            (cholesky(H), h),
            (ldlt(Hermitian(H)), h),
            (ldlt(H), h),
            (bunchkaufman(Hermitian(H)), h),
            (bunchkaufman(Hermitian(H, :U), true), h),
            (bunchkaufman(H), h),
        )
            @test F isa Colmat.FaerFactorization{T}
            @test isapprox(Matrix(F), ref; rtol=tol)
            @test isapprox(F \ b, ref \ b; rtol=tol)
        end
        # The borrowed parent is only read.
        @test a == before
        # Same factor kind and results as a native input.
        @test typeof(lu(B)) == typeof(lu(FaerMatrix(a)))
        @test typeof(cholesky(Hermitian(H))) == typeof(cholesky(Hermitian(FaerMatrix(h))))
        @test lu(B).p == lu(FaerMatrix(a)).p
        # Negative and non-unit strides are read in place.
        r = faerview(view(a, n:-1:1, :))
        @test isapprox(Matrix(lu(r)), a[n:-1:1, :]; rtol=tol)
        big = zeros(T, 2n, 2n)
        big[1:2:end, 1:2:end] = a
        s = faerview(view(big, 1:2:(2n - 1), 1:2:(2n - 1)))
        @test isapprox(Matrix(qr(s)), a; rtol=tol)
        # Rectangular and empty inputs.
        w = faerview(a[:, 1:4])
        @test isapprox(Matrix(qr(w)), a[:, 1:4]; rtol=tol)
        @test size(lu(faerview(zeros(T, 0, 0)))) == (0, 0)
        # Construction checks keep the native-input conventions.
        @test_throws LinearAlgebra.PosDefException cholesky(B)
        c = cholesky(B; check=false)
        @test !issuccess(c) && c.info == -1
        @test_throws ArgumentError ldlt(B)
        @test_throws DimensionMismatch cholesky(Hermitian(faerview(zeros(T, 3, 3)))[1:2, :])
        z = faerview(zeros(T, 3, 3))
        @test_throws LinearAlgebra.SingularException lu(z)
        @test !issuccess(lu(z; check=false))
        @test_throws Exception lu(faerview(T[1 NaN; 0 1]))
        @test_throws Exception svd(faerview(T[1 NaN; 0 1]))
    end
end

@testset "Borrowed spectral inputs" begin
    for T in BF_TYPES
        n = 5
        a = _bf_matrix(T, n)
        h = _bf_matrix(T, n; hermitian=true)
        B = faerview(a)
        H = faerview(h)
        tol = _bf_tol(T)
        A = FaerMatrix(a)
        for full in (false, true)
            S = svd(B; full)
            R = svd(A; full)
            @test typeof(S) == typeof(R)
            @test isapprox(S.S, R.S; rtol=tol)
            @test isapprox(S.U * Diagonal(S.S) * S.Vt[1:n, :], a; rtol=tol)
        end
        w = faerview(a[:, 1:3])
        @test isapprox(svdvals(w), svdvals(a[:, 1:3]); rtol=tol)
        @test isapprox(svdvals(B), svdvals(a); rtol=tol)
        E = eigen(B)
        @test typeof(E) == typeof(eigen(A))
        @test isapprox(a * E.vectors, E.vectors * Diagonal(E.values); rtol=10tol)
        @test isapprox(eigvals(B), eigvals(A); rtol=tol)
        for uplo in (:U, :L)
            Eh = eigen(Hermitian(H, uplo))
            @test typeof(Eh) == typeof(eigen(Hermitian(FaerMatrix(h), uplo)))
            @test isapprox(h * Eh.vectors, Eh.vectors * Diagonal(Eh.values); rtol=10tol)
            @test isapprox(eigvals(Hermitian(H, uplo)), eigvals(Hermitian(h)); rtol=tol)
        end
        # Options without a borrowed driver take one native copy of the input.
        @test isapprox(svd(B; alg=LinearAlgebra.QRIteration()).S, svdvals(a); rtol=tol)
        L = eigen(B; side=:left)
        @test isapprox(L.vectors' * a, Diagonal(L.values) * L.vectors'; rtol=10tol)
        @test isapprox(
            sort(eigvals(B; permute=true, scale=true); by=Colmat._eigensort),
            eigvals(A);
            rtol=10tol,
        )
    end
end

@testset "Generic LinearAlgebra on borrowed inputs" begin
    for T in BF_TYPES
        n = 5
        a = _bf_matrix(T, n)
        h = _bf_matrix(T, n; hermitian=true)
        B = faerview(a)
        H = faerview(h)
        tol = _bf_tol(T)
        v = T.(1:n)
        @test isapprox(det(B), det(a); rtol=tol)
        @test isapprox(logabsdet(B)[1], logabsdet(a)[1]; rtol=tol)
        @test isapprox(Matrix(inv(B)), inv(a); rtol=tol)
        @test isapprox(B \ v, a \ v; rtol=tol)
        @test isapprox(Matrix(B \ reshape(v, :, 1)), a \ reshape(v, :, 1); rtol=tol)
        @test isapprox(Hermitian(H) \ v, h \ v; rtol=tol)
        @test isposdef(Hermitian(H)) == isposdef(Hermitian(h))
        @test isapprox(cond(B), cond(a); rtol=10tol)
        @test isapprox(opnorm(B), opnorm(a); rtol=tol)
        @test rank(B) == rank(a)
        @test isapprox(pinv(B), pinv(a); rtol=10tol)
        @test isapprox(factorize(B) \ v, a \ v; rtol=tol)
        # Hermitian borrows: factorize picks Cholesky/Bunch–Kaufman, and eigen
        # keeps LinearAlgebra's real self-adjoint result as for Julia arrays.
        @test isapprox(H \ v, h \ v; rtol=tol)
        @test isapprox(factorize(H) \ v, h \ v; rtol=tol)
        @test eigvals(H) isa Vector{real(T)}
        @test isapprox(eigvals(H), eigvals(Hermitian(h)); rtol=tol)
        @test eigen(H).values isa Vector{real(T)}
        @test isapprox(eigmax(H), maximum(eigvals(Hermitian(h))); rtol=tol)
        @test isapprox(eigmin(H), minimum(eigvals(Hermitian(h))); rtol=tol)
        @test eltype(eigvals(B)) <: Complex
    end
end

@testset "Failed borrowed solves keep inputs and destinations" begin
    for T in BF_TYPES
        n = 4
        a = _bf_matrix(T, n)
        F = lu(FaerMatrix(a))
        bad = ones(T, n, 2)
        bad[2, 1] = T(NaN)
        X = copy(bad)
        @test_throws Exception ldiv!(F, X)
        @test isequal(X, bad)
        Y = fill(T(7), n, 2)
        @test_throws Exception ldiv!(Y, F, bad)
        @test Y == fill(T(7), n, 2)
        team = FaerTeam(2)
        close(team)
        @test_throws ArgumentError ldiv!(Y, F, ones(T, n, 2); team)
        @test Y == fill(T(7), n, 2)
        close(F)
        @test_throws ArgumentError ldiv!(Y, F, ones(T, n, 2))
        @test Y == fill(T(7), n, 2)
    end
end

@testset "Borrowed factor inputs: teams and read-only borrows" begin
    a = _bf_matrix(Float64, 8)
    team = FaerTeam(2)
    F = lu(faerview(a); team)
    @test isapprox(Matrix(F), a; rtol=_bf_tol(Float64))
    G = qr(faerview(_bf_matrix(ComplexF32, 8)); team)
    @test G isa Colmat.FaerFactorization{ComplexF32}
    close(team)
    @test_throws ArgumentError lu(faerview(a); team)
    @test_throws ArgumentError lu(faerview(a); team=1)
    # An NPY borrow is read-only but may be factored (construction only reads).
    header = "{'descr': '<f8', 'fortran_order': True, 'shape': $(size(a)), }"
    header *= " "^(63 - mod(10 + length(header), 64)) * "\n"
    bytes = UInt8[0x93; codeunits("NUMPY"); 0x01; 0x00]
    append!(bytes, reinterpret(UInt8, [htol(UInt16(length(header)))]))
    append!(bytes, codeunits(header))
    append!(bytes, reinterpret(UInt8, vec(a)))
    N = npyborrow(bytes)
    @test N isa FaerBorrowedMatrix{Float64}
    @test isapprox(Matrix(lu(N)), a; rtol=_bf_tol(Float64))
    # ... but not written by an in-place solve, natively or through a fallback.
    @test_throws ArgumentError ldiv!(F, N)
    @test_throws ArgumentError ldiv!(N, F, ones(8, 8))
    @test N == a
    # A writable NPY borrow is a native in-place destination (bytes in place).
    W = npyborrow(copy(bytes); writable=true)
    rhs = Matrix(W)
    @test ldiv!(F, W) === W
    @test isapprox(Matrix(W), a \ rhs; rtol=_bf_tol(Float64))
    W2 = npyborrow(copy(bytes); writable=true)
    ldiv!(W2, F, a)
    @test isapprox(Matrix(W2), Matrix(I, 8, 8); atol=1e-10)
end

@testset "Closed native operands keep Julia destinations" begin
    for T in BF_TYPES
        a = _bf_matrix(T, 4)
        F = lu(faerview(a))
        R = FaerMatrix(ones(T, 4, 2))
        close(R)
        Y = fill(T(3), 4, 2)
        @test_throws ArgumentError ldiv!(Y, F, R)
        @test Y == fill(T(3), 4, 2)
        @test_throws ArgumentError ldiv!(F, R)
        X = FaerMatrix(a)
        close(X)
        @test_throws ArgumentError lu(X)
    end
end

@testset "Retained-factor solves in place in Julia and borrowed destinations" begin
    for T in BF_TYPES, kind in (:lu, :full_lu, :qr, :pqr, :cholesky, :ldlt, :bk)
        n = 6
        h = kind in (:cholesky, :ldlt, :bk)
        a = _bf_matrix(T, n; hermitian=h)
        A = FaerMatrix(a)
        F = if kind === :lu
            lu(A)
        elseif kind === :full_lu
            full_piv_lu(A)
        elseif kind === :qr
            qr(A)
        elseif kind === :pqr
            qr(A, ColumnNorm())
        elseif kind === :cholesky
            cholesky(Hermitian(A))
        elseif kind === :ldlt
            ldlt(Hermitian(A))
        else
            bunchkaufman(Hermitian(A))
        end
        b = T.(reshape(1:(3n), n, 3)) .+ T(0.5)
        tol = _bf_tol(T)
        for (G, ref) in ((F, a), (F', a'), (transpose(F), transpose(a)))
            x = ref \ b
            X = copy(b)
            @test ldiv!(G, X) === X
            @test isapprox(X, x; rtol=tol)
            V = faerview(copy(b))
            @test ldiv!(G, V) === V
            @test isapprox(parent(V), x; rtol=tol)
            y = b[:, 1]
            ldiv!(G, y)
            @test isapprox(y, x[:, 1]; rtol=tol)
            Y = zeros(T, n, 3)
            @test ldiv!(Y, G, b) === Y
            @test isapprox(Y, x; rtol=tol)
            @test isapprox(ldiv!(zeros(T, n, 3), G, faerview(b)), x; rtol=tol)
            # Reversed destination view.
            R = zeros(T, n, 3)
            ldiv!(faerview(view(R, n:-1:1, :)), G, b)
            @test isapprox(R[n:-1:1, :], x; rtol=tol)
            # `F \ B` returns a native result for borrowed and Julia inputs.
            for rhs in (b, faerview(copy(b)))
                Z = G \ rhs
                @test Z isa FaerMatrix{T}
                @test isapprox(Matrix(Z), x; rtol=tol)
            end
            @test isapprox(G \ b[:, 1], x[:, 1]; rtol=tol)
            # Overlapping input and destination share Julia memory.
            M = zeros(T, n, 4)
            M[:, 1:3] = b
            ldiv!(view(M, :, 2:4), G, view(M, :, 1:3))
            @test isapprox(M[:, 2:4], x; rtol=tol)
        end
    end
end

@testset "Rank-deficient and rectangular factors keep the fresh-result path" begin
    for T in BF_TYPES
        n = 5
        a = _bf_matrix(T, n)
        a[:, 3] = a[:, 2]
        F = qr(FaerMatrix(a), ColumnNorm())
        b = T.(reshape(1:(2n), n, 2))
        ref = F \ FaerMatrix(b)
        X = copy(b)
        ldiv!(F, X)
        @test isapprox(X, Matrix(ref); rtol=_bf_tol(T))
        Y = copy(b)
        ldiv!(Y, F, Y)
        @test isapprox(Y, Matrix(ref); rtol=_bf_tol(T))
        @test isapprox(Matrix(F \ b), Matrix(ref); rtol=_bf_tol(T))
        w = _bf_matrix(T, 7)[:, 1:4]
        Q = qr(FaerMatrix(w))
        rhs = T.(reshape(1:14, 7, 2))
        D = zeros(T, 4, 2)
        ldiv!(D, Q, rhs)
        @test isapprox(D, w \ rhs; rtol=10_bf_tol(T))
    end
end

@testset "In-place solves allocate no right-hand-side copies" begin
    n = 400
    a = _bf_matrix(Float64, n)
    F = lu(FaerMatrix(a))
    H = cholesky(Hermitian(FaerMatrix((a + a') / 2)))
    b = reshape(sin.(1:(16n)), n, 16)
    X = copy(b)
    Y = similar(b)
    V = faerview(X)
    rhs = sizeof(b)
    for f in (
        () -> ldiv!(F, X),
        () -> ldiv!(F, V),
        () -> ldiv!(H, X),
        () -> ldiv!(Y, F, b),
        () -> ldiv!(F', X),
    )
        f()
        # Only a constant per-call overhead (descriptors, call wrappers).
        @test @allocated(f()) < rhs ÷ 4
    end
    v = b[:, 1]
    ldiv!(F, v)
    @test @allocated(ldiv!(F, v)) < 8192
    copyto!(X, b)
    @test isapprox(ldiv!(F, X), a \ b; rtol=1e-10)
end

@testset "Borrowed factor methods add no ambiguities" begin
    ambiguities = Test.detect_ambiguities(
        Colmat, Base, LinearAlgebra, SparseArrays, Statistics; recursive=true
    )
    @test isempty(
        filter(a -> a[1].module === Colmat || a[2].module === Colmat, ambiguities)
    )
end
