using Colmat, Test, LinearAlgebra

const SAGEVD_TYPES = (Float32, Float64, ComplexF32, ComplexF64)
function _sagevd_pair(::Type{T}) where {T}
    a = T[4 1 0; 1 3 1; 0 1 2]
    b = T[2 0.5 0; 0.5 2 0; 0 0 1]
    if T <: Complex
        a = a + T(0.5im) * T[0 1 0; -1 0 1; 0 -1 0]
        b = b + T(0.25im) * T[0 1 0; -1 0 0; 0 0 0]
    end
    return a, b
end
_wrap(W, x) = W === Symmetric ? Symmetric(x) : Hermitian(x)

@testset "Non-positive-definite B throws PosDefException with the pivot" begin
    team = FaerTeam(2)
    for T in SAGEVD_TYPES, pivot in (1, 2, 3)
        W = T <: Complex ? Hermitian : Symmetric
        a, b = _sagevd_pair(T)
        nb = copy(b)
        nb[pivot, pivot] = -T(5)
        ref = cholesky(Hermitian(nb); check=false).info
        A, B = FaerMatrix(a), FaerMatrix(nb)
        calls = (
            () -> eigen(_wrap(W, A), _wrap(W, B)),
            () -> eigvals(_wrap(W, A), _wrap(W, B)),
            () -> eigen(_wrap(W, A), _wrap(W, B); team),
            () -> eigvals(_wrap(W, A), _wrap(W, B); sortby=nothing),
            () -> eigen!(_wrap(W, A), _wrap(W, B)),
            () -> eigvals!(_wrap(W, A), _wrap(W, B); team),
        )
        for f in calls
            err = try
                f()
                nothing
            catch e
                e
            end
            @test err isa PosDefException
            @test err.info == ref
        end
        # Failures never modify the inputs, including bang forms.
        @test Matrix(A) == a
        @test Matrix(B) == nb
        foreach(close, (A, B))
    end
    close(team)
end

@testset "Non-finite selected triangles throw ArgumentError" begin
    for T in SAGEVD_TYPES
        a, b = _sagevd_pair(T)
        for (x, y) in ((true, false), (false, true))
            A, B = FaerMatrix(a), FaerMatrix(b)
            x && (A[3, 1] = T(NaN))
            y && (B[2, 2] = T(Inf))
            before = (Matrix(A), Matrix(B))
            @test_throws ArgumentError eigvals(Hermitian(A, :L), Hermitian(B, :L))
            @test_throws ArgumentError eigen!(Hermitian(A, :L), Hermitian(B, :L))
            @test isequal((Matrix(A), Matrix(B)), before)
            foreach(close, (A, B))
        end
        # A non-finite value in the unused triangle is ignored.
        A, B = FaerMatrix(a), FaerMatrix(b)
        A[1, 3] = T(NaN)
        @test eigvals(Hermitian(A, :L), Hermitian(B)) ≈
            eigvals(Hermitian(a, :L), Hermitian(b))
        foreach(close, (A, B))
    end
end

@testset "Diagonal right-hand sides" begin
    for T in SAGEVD_TYPES
        a, _ = _sagevd_pair(T)
        A = FaerMatrix(a)
        H = Hermitian(A)
        d = real(T)[1, 2, 3]
        ref = eigen(Hermitian(a), Diagonal(d))
        for D in (Diagonal(d), FaerDiagonal(T.(d)))
            vals = eigvals(H, D)
            @test vals ≈ ref.values
            F = eigen(H, D)
            @test F.values ≈ ref.values
            @test a * F.vectors ≈ Matrix(Diagonal(T.(d))) * F.vectors * Diagonal(F.values)
            @test eigvals(H, D; sortby=x -> -x) ≈ reverse(ref.values)
        end
        @test_throws ArgumentError eigvals(H, Diagonal(real(T)[1, 0, 3]))
        @test_throws DimensionMismatch eigvals(H, Diagonal(real(T)[1, 2]))
        # Indefinite diagonals follow LinearAlgebra's general path.
        nd = real(T)[1, -2, 3]
        @test sort(eigvals(H, Diagonal(nd)); by=real) ≈
            sort(eigvals(Hermitian(a), Diagonal(nd)); by=real)
        @test Matrix(A) == a
        close(A)
    end
end
