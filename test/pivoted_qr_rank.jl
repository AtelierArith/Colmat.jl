using Colmat, Test, LinearAlgebra

const PIVQR_TYPES=(Float32, Float64, ComplexF32, ComplexF64)
# Rank-deficient samples: (tall, wide) with a dependent column and row.
function _pivqr_samples(::Type{T}) where {T}
    a=T[1 2 3; 2 4 6; 1 0 1; 0 1 1]
    T<:Complex && (a=a .+ T(0.5im) .* T[1 2 0; 2 4 0; 0 0 1; 1 2 1])
    a[:, 2]=2a[:, 1]
    w=T[1 2 3 4; 2 4 6 8]
    T<:Complex && (w=w .* T(1+0.25im))
    return (a, w, T[2 1; 1 3; 0 1], zeros(T, 3, 2), zeros(T, 0, 3))
end

@testset "Column-pivoted QR rank matches QRPivoted" begin
    for T in PIVQR_TYPES, a in _pivqr_samples(T)
        F=qr(FaerMatrix(a), ColumnNorm())
        G=qr(a, ColumnNorm())
        @test rank(F) == rank(G)
        for (atol, rtol) in ((0, 0.1), (0, 0.9), (1.0, 0), (1e3, 0), (1e-3, 1e-3))
            @test rank(F; atol, rtol) == rank(G; atol, rtol)
        end
        @test rank(F; atol = 1.0) == rank(G; atol = 1.0)
        @test_throws ArgumentError rank(F; atol = -1)
        @test_throws ArgumentError rank(F; rtol = -1)
        close(F)
    end
end

@testset "Rank-truncated minimum-norm pivoted QR solves" begin
    team=FaerTeam(2)
    for T in PIVQR_TYPES, a in _pivqr_samples(T)[1:3], k in (1, 3)
        m, n=size(a)
        F=qr(FaerMatrix(a), ColumnNorm())
        G=qr(a, ColumnNorm())
        rows=max(m, n)
        b=zeros(T, rows, k)
        b[1:m, :]=T.(reshape(1:(m*k), m, k))
        tol=sqrt(eps(real(T)))
        ref, rref=ldiv!(G, copy(b), tol)
        for t in (nothing, team)
            B=copy(b)
            out, r=ldiv!(F, B, tol; team = t)
            @test out === B
            @test r == rref
            @test isapprox(B[1:n, :], ref[1:n, :]; rtol = 50tol)
            N=FaerMatrix(b)
            @test ldiv!(F, N, tol; team = t)[2] == rref
            @test isapprox(Matrix(N)[1:n, :], ref[1:n, :]; rtol = 50tol)
            @test Matrix(N)[(n+1):end, :] == b[(n+1):end, :]
        end
        if k==1
            v=copy(vec(b))
            @test ldiv!(F, v, tol)[2] == rref
            @test isapprox(v[1:n], ref[1:n, 1]; rtol = 50tol)
            fv=FaerVector(vec(b))
            ldiv!(F, fv, tol)
            @test isapprox(Vector(fv)[1:n], ref[1:n, 1]; rtol = 50tol)
        end
        @test_throws DimensionMismatch ldiv!(F, zeros(T, rows-1, k), tol)
        @test_throws ArgumentError ldiv!(F, copy(b), -1)
        close(F)
    end
    # A zero matrix has rank zero and a zero solution.
    F=qr(FaerMatrix(zeros(3, 2)), ColumnNorm())
    B=ones(3, 1)
    @test ldiv!(F, B, 1e-8) == (B, 0)
    @test B[1:2, :] == zeros(2, 1)
    closed=FaerTeam(2)
    close(closed)
    @test_throws ArgumentError ldiv!(F, ones(3, 1), 1e-8; team = closed)
    close(F)
    @test_throws ArgumentError rank(F)
    close(team)
end
