using Colmat, Test, LinearAlgebra, SparseArrays

const PIVQR_TYPES = (Float32, Float64, ComplexF32, ComplexF64)
# Rank-deficient samples: (tall, wide) with a dependent column and row.
function _pivqr_samples(::Type{T}) where {T}
    a = T[1 2 3; 2 4 6; 1 0 1; 0 1 1]
    T <: Complex && (a = a .+ T(0.5im) .* T[1 2 0; 2 4 0; 0 0 1; 1 2 1])
    a[:, 2] = 2a[:, 1]
    w = T[1 2 3 4; 2 4 6 8]
    T <: Complex && (w = w .* T(1 + 0.25im))
    return (a, w, T[2 1; 1 3; 0 1], zeros(T, 3, 2), zeros(T, 0, 3))
end

@testset "Column-pivoted QR rank matches QRPivoted" begin
    for T in PIVQR_TYPES, a in _pivqr_samples(T)
        F = qr(FaerMatrix(a), ColumnNorm())
        G = qr(a, ColumnNorm())
        @test rank(F) == rank(G)
        for (atol, rtol) in ((0, 0.1), (0, 0.9), (1.0, 0), (1e3, 0), (1e-3, 1e-3))
            @test rank(F; atol, rtol) == rank(G; atol, rtol)
        end
        @test rank(F; atol=1.0) == rank(G; atol=1.0)
        @test_throws ArgumentError rank(F; atol=-1)
        @test_throws ArgumentError rank(F; rtol=-1)
        close(F)
    end
end

@testset "Rank-truncated minimum-norm pivoted QR solves" begin
    team = FaerTeam(2)
    for T in PIVQR_TYPES, a in _pivqr_samples(T)[1:3], k in (1, 3)
        m, n = size(a)
        F = qr(FaerMatrix(a), ColumnNorm())
        G = qr(a, ColumnNorm())
        rows = max(m, n)
        b = zeros(T, rows, k)
        b[1:m, :] = T.(reshape(1:(m * k), m, k))
        tol = sqrt(eps(real(T)))
        ref, rref = ldiv!(G, copy(b), tol)
        for t in (nothing, team)
            B = copy(b)
            out, r = ldiv!(F, B, tol; team=t)
            @test out === B
            @test r == rref
            @test isapprox(B[1:n, :], ref[1:n, :]; rtol=50tol)
            N = FaerMatrix(b)
            @test ldiv!(F, N, tol; team=t)[2] == rref
            @test isapprox(Matrix(N)[1:n, :], ref[1:n, :]; rtol=50tol)
            @test Matrix(N)[(n + 1):end, :] == b[(n + 1):end, :]
        end
        if k == 1
            v = copy(vec(b))
            @test ldiv!(F, v, tol)[2] == rref
            @test isapprox(v[1:n], ref[1:n, 1]; rtol=50tol)
            fv = FaerVector(vec(b))
            ldiv!(F, fv, tol)
            @test isapprox(Vector(fv)[1:n], ref[1:n, 1]; rtol=50tol)
        end
        @test_throws DimensionMismatch ldiv!(F, zeros(T, rows - 1, k), tol)
        @test_throws ArgumentError ldiv!(F, copy(b), -1)
        close(F)
    end
    # A zero matrix has rank zero and a zero solution.
    F = qr(FaerMatrix(zeros(3, 2)), ColumnNorm())
    B = ones(3, 1)
    @test ldiv!(F, B, 1e-8) == (B, 0)
    @test B[1:2, :] == zeros(2, 1)
    closed = FaerTeam(2)
    close(closed)
    @test_throws ArgumentError ldiv!(F, ones(3, 1), 1e-8; team=closed)
    close(F)
    @test_throws ArgumentError rank(F)
    close(team)
end

# Keyword-tolerance solves: ldiv!(X, F, B; atol, rtol, team) and ldiv!(F, B; ...).
# Low-rank product with distinct column norms so faer and LAPACK pivot alike.
function pqr_lowrank(T, m, n, r)
    u = T[sin(i + 2j) + (i == j) for i in 1:m, j in 1:r]
    v =
        T[cos(2i + j) + 2(i == j) for i in 1:r, j in 1:n] .* T[1 + j / n for _ in 1:r, j in 1:n]
    T <: Complex && (u .+= T(0.3im) .* T[cos(i - j) for i in 1:m, j in 1:r])
    return u * v
end
# Orthogonal columns scaled by a graded spectrum: |R_ii| equal the singular values.
function pqr_graded(T, m, d)
    q = Matrix(qr(T[sin(i + 3j) + (i == j) for i in 1:m, j in 1:length(d)]).Q)[
        :, 1:length(d)
    ]
    return q * Diagonal(T.(d))
end
pqr_tol(T) = 2000eps(real(T))
@testset "pivoted QR rank follows the saved R diagonal" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        d = [1, 1e-2, 1e-4, 1e-6, 1e-9]
        a = pqr_graded(T, 6, d)
        f = qr(FaerMatrix(a), ColumnNorm())
        reference = qr(a, ColumnNorm())
        @test rank(f) == rank(reference) == (T <: Union{Float32,ComplexF32} ? 4 : 5)
        for kw in (
            (; atol=1e-3),
            (; atol=1e-5),
            (; rtol=1e-5),
            (; atol=0.5, rtol=0),
            (; atol=2.0),
            (; atol=1e-12, rtol=1e-3),
            (; atol=Float32(1e-3)),
        )
            @test rank(f; kw...) == rank(reference; kw...)
        end
        @test rank(f; atol=1e-3) == 2
        @test rank(f; atol=1e-5) == 3
        for (m, n, r) in ((7, 4, 2), (4, 7, 2), (5, 5, 3), (6, 6, 6), (3, 8, 3))
            b = pqr_lowrank(T, m, n, r)
            g = qr(FaerMatrix(b), ColumnNorm())
            @test rank(g) == rank(qr(b, ColumnNorm())) == r
            close(g)
        end
        for (m, n) in ((0, 3), (3, 0), (0, 0))
            g = qr(FaerMatrix(zeros(T, m, n)), ColumnNorm())
            @test rank(g) == 0
            close(g)
        end
        z = qr(FaerMatrix(zeros(T, 3, 2)), ColumnNorm())
        @test rank(z) == 0
        close(z)
        @test_throws ArgumentError rank(f; atol=-1)
        @test_throws ArgumentError rank(f; rtol=NaN)
        close(f)
        @test_throws ArgumentError rank(f)
    end
end
@testset "Positional rcond and keyword rtol give identical solves" begin
    team = FaerTeam(2)
    for T in PIVQR_TYPES, a in _pivqr_samples(T)[1:4], k in (1, 3), t in (nothing, team)
        m, n = size(a)
        F = qr(FaerMatrix(a), ColumnNorm())
        rhs = T.(reshape(1:(m * k), m, k))
        padded = zeros(T, max(m, n), k)
        padded[1:m, :] = rhs
        for rcond in (sqrt(eps(real(T))), 0.1, 0.9, 0.0)
            out, r = ldiv!(F, copy(padded), rcond; team=t)
            @test r == rank(F; atol=0, rtol=rcond)
            X = fill(T(NaN), n, k)
            ldiv!(X, F, rhs; rtol=rcond, team=t)
            @test X == out[1:n, :]
            ldiv!(X, F, rhs; atol=0, rtol=rcond, team=t)
            @test X == out[1:n, :]
        end
        close(F)
    end
    close(team)
    # Both forms reject a non-finite right-hand side and keep B intact.
    F = qr(FaerMatrix([2.0 1; 1 3; 0 1]), ColumnNorm())
    # A real factor solves a complex B by parts, positionally and by keyword.
    z = ComplexF64[1 + 2im 3; -1im 2 + im; 4 5im]
    re, _ = ldiv!(F, real(z), 1e-8)
    im_, _ = ldiv!(F, imag(z), 1e-8)
    zc, r = ldiv!(F, copy(z), 1e-8)
    @test r == 2
    @test zc[1:2, :] == complex.(re[1:2, :], im_[1:2, :])
    @test ldiv!(zeros(ComplexF64, 2, 2), F, z; rtol=1e-8) == zc[1:2, :]
    bad = [Inf; 1.0; 2.0;;]
    @test_throws Colmat.RustCall.RustError ldiv!(F, bad, 1e-8)
    @test isequal(bad, [Inf; 1.0; 2.0;;])
    close(F)
end
@testset "Keyword-tolerance pivoted QR solves match pinv" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        (m, n, r) in ((7, 4, 2), (4, 7, 2), (5, 5, 3), (6, 4, 4), (3, 6, 3), (1, 1, 1)),
        S in unique((T, Float64, ComplexF64)),
        team in (nothing, FaerTeam(2))

        a = pqr_lowrank(T, m, n, r)
        source = FaerMatrix(a)
        f = qr(source, ColumnNorm())
        close(source)
        P = promote_type(T, S)
        b = P[cos(i + 2j) + (i == j) for i in 1:m, j in 1:3]
        rt = sqrt(eps(real(T)))
        @test rank(f; rtol=rt) == r
        reference = pinv(Matrix{P}(a); rtol=rt) * b
        x = fill(P(NaN), n, 3)
        @test ldiv!(x, f, b; rtol=rt, team) === x
        @test x ≈ reference rtol = pqr_tol(T) atol = pqr_tol(T)
        # The solve and rank share the same cutoff on the R diagonal.
        x2 = zeros(P, n, 3)
        ldiv!(x2, f, b; atol=0, rtol=rt, team)
        @test x2 ≈ x
        # Vectors and views of larger parents.
        xv = zeros(P, n)
        ldiv!(xv, f, b[:, 2]; rtol=rt, team)
        @test xv ≈ reference[:, 2] rtol = pqr_tol(T) atol = pqr_tol(T)
        bigb = zeros(P, m + 2, 5)
        bigb[2:(m + 1), 2:4] .= b
        bigx = fill(P(7), n + 3, 4)
        ldiv!(view(bigx, 3:(n + 2), 1:3), f, view(bigb, 2:(m + 1), 2:4); rtol=rt, team)
        @test bigx[3:(n + 2), 1:3] ≈ reference rtol = pqr_tol(T) atol = pqr_tol(T)
        @test all(==(P(7)), bigx[1:2, :]) && all(==(P(7)), bigx[:, 4])
        if m == n
            # Aliased destination and the two-argument square form.
            y = copy(b)
            @test ldiv!(y, f, y; rtol=rt, team) === y
            @test y ≈ reference rtol = pqr_tol(T) atol = pqr_tol(T)
            y = copy(b)
            @test ldiv!(f, y; rtol=rt, team) === y
            @test y ≈ reference rtol = pqr_tol(T) atol = pqr_tol(T)
            yv = b[:, 1]
            @test ldiv!(f, yv; rtol=rt, team) === yv
            @test yv ≈ reference[:, 1] rtol = pqr_tol(T) atol = pqr_tol(T)
        else
            @test_throws DimensionMismatch ldiv!(f, copy(b); rtol=rt, team)
        end
        isnothing(team) || close(team)
        close(f)
    end
end
@testset "pivoted QR truncation flips with atol on a graded spectrum" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        d = [1, 1e-1, 1e-2, 1e-4, 1e-6]
        a = pqr_graded(T, 7, d)
        f = qr(FaerMatrix(a), ColumnNorm())
        b = T[sin(3i + j) for i in 1:7, j in 1:2]
        for (atol, r) in ((5e-2, 2), (5e-3, 3), (5e-5, 4), (0.5, 1), (2.0, 0))
            @test rank(f; atol) == r
            x = ldiv!(zeros(T, 5, 2), f, b; atol)
            @test x ≈ pinv(a; atol) * b rtol = sqrt(eps(real(T))) atol = sqrt(eps(real(T)))
        end
        @test iszero(ldiv!(ones(T, 5, 2), f, b; atol=2.0))
        close(f)
    end
end
@testset "pivoted QR truncated solve validates before writing" begin
    for T in (Float64, ComplexF32)
        a = pqr_lowrank(T, 5, 4, 2)
        f = qr(FaerMatrix(a), ColumnNorm())
        b = ones(T, 5, 2)
        x = fill(T(3), 4, 2)
        @test_throws ArgumentError ldiv!(x, f, b; atol=-1)
        @test_throws ArgumentError ldiv!(x, f, b; rtol=NaN)
        @test_throws DimensionMismatch ldiv!(zeros(T, 3, 2), f, b; rtol=1e-6)
        @test_throws DimensionMismatch ldiv!(x, f, ones(T, 4, 2); rtol=1e-6)
        @test_throws ArgumentError ldiv!(x, transpose(f), ones(T, 4, 2); rtol=1e-6)
        @test_throws ArgumentError ldiv!(x, f', ones(T, 4, 2); rtol=1e-6)
        @test_throws ArgumentError ldiv!(x, f, b; rtol=1e-6, scratch=Colmat.FaerScratch(0))
        bad = copy(b)
        bad[1] = T(Inf)
        @test_throws Exception ldiv!(x, f, bad; rtol=1e-6)
        @test all(==(T(3)), x)
        @test b == ones(T, 5, 2)
        team = FaerTeam(2)
        close(team)
        @test_throws ArgumentError ldiv!(x, f, b; rtol=1e-6, team)
        @test all(==(T(3)), x)
        close(f)
        @test_throws ArgumentError ldiv!(x, f, b; rtol=1e-6)
        @test all(==(T(3)), x)
        # Without tolerances the existing solve contract is unchanged.
        g = qr(FaerMatrix(a), ColumnNorm())
        @test ldiv!(zeros(T, 4, 2), g, b) ≈ g \ b
        close(g)
    end
end

@testset "Pivoted QR rank solves have no method ambiguities" begin
    @test isempty(
        Test.detect_ambiguities(Colmat, Base, LinearAlgebra, SparseArrays; recursive=true)
    )
end
