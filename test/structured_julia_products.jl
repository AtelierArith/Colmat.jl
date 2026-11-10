using Colmat, Test, LinearAlgebra

const SJP_TYPES = (Float32, Float64, ComplexF32, ComplexF64)
const SJP_TRIANGULAR = (
    LowerTriangular, UpperTriangular, UnitLowerTriangular, UnitUpperTriangular
)
function _sjp(::Type{T}, m, n, s=1) where {T}
    return T.([
        sin(s * i + j) + (T <: Complex ? im * cos(i - s * j) : 0) for i in 1:m, j in 1:n
    ])
end
_sjp_tol(::Type{T}) where {T} = sqrt(eps(real(T))) * 100
_sjp_close(x, y, T) = isapprox(Array(x), y; rtol=_sjp_tol(T))
# Julia-side operand forms: a plain matrix, lazy transposes and a borrowed view.
_sjp_julia_forms(j) = (j, transpose(copy(transpose(j))), copy(j')', faerview(copy(j)))

@testset "Native-parent triangular wrappers with Julia operands" begin
    for T in SJP_TYPES, W in SJP_TRIANGULAR
        n = 5
        a = _sjp(T, n, n)
        A = FaerMatrix(a)
        for j in _sjp_julia_forms(_sjp(T, n, n, 2))
            ref = Array(j)
            x = W(A) * j
            @test x isa FaerMatrix{T}
            @test _sjp_close(x, W(a) * ref, T)
            y = j * W(A)
            @test y isa FaerMatrix{T}
            @test _sjp_close(y, ref * W(a), T)
            # Adjoints and transposes of the wrapper (parents are lazy wrappers).
            @test _sjp_close(W(A)' * j, W(a)' * ref, T)
            @test _sjp_close(j * transpose(W(A)), ref * transpose(W(a)), T)
        end
        # A view parent and a Julia-parent wrapper with a native operand.
        V = view(A, 1:n, 1:n)
        j = _sjp(T, n, n, 3)
        @test _sjp_close(W(V) * j, W(a) * j, T)
        @test _sjp_close(W(j) * A, W(j) * a, T)
        # Mixed scalar types promote natively.
        j64 = _sjp(ComplexF64, n, n, 4)
        z = W(A) * j64
        @test z isa FaerMatrix{ComplexF64}
        @test _sjp_close(z, W(a) * j64, ComplexF64)
        # Vectors.
        v = _sjp(T, n, 1, 5)[:, 1]
        u = W(A) * v
        @test u isa FaerVector{T}
        @test _sjp_close(u, W(a) * v, T)
        close(A)
    end
end

@testset "Triangular mul! into Julia and native destinations" begin
    for T in SJP_TYPES, W in SJP_TRIANGULAR
        n = 4
        a = _sjp(T, n, n)
        j = _sjp(T, n, n, 2)
        c0 = _sjp(T, n, n, 3)
        A = FaerMatrix(a)
        α, β = T(2), T(-1)
        for (x, y, rx, ry) in (
            (W(A), j, W(a), j),
            (j, W(A), j, W(a)),
            (W(j), A, W(j), a),
            (A, W(j), a, W(j)),
            (W(A), A, W(a), a),
            (W(A), W(j), W(a), W(j)),
            (W(A), transpose(j), W(a), transpose(j)),
        )
            C = copy(c0)
            @test mul!(C, x, y) === C
            @test isapprox(C, rx * ry; rtol=_sjp_tol(T))
            C = copy(c0)
            @test mul!(C, x, y, α, β) === C
            @test isapprox(C, α * (rx * ry) + β * c0; rtol=_sjp_tol(T))
            D = FaerMatrix(c0)
            @test mul!(D, x, y, α, β) === D
            @test _sjp_close(D, α * (rx * ry) + β * c0, T)
            close(D)
        end
        # A borrowed destination and a NaN destination with β = 0.
        C = fill(T(NaN), n, n)
        mul!(faerview(C), W(A), j)
        @test isapprox(C, W(a) * j; rtol=_sjp_tol(T))
        # The destination aliases an operand.
        J = copy(j)
        mul!(J, W(A), J)
        @test isapprox(J, W(a) * j; rtol=_sjp_tol(T))
        J = copy(j)
        mul!(J, W(J), A)
        @test isapprox(J, W(j) * a; rtol=_sjp_tol(T))
        # Diagonals into Julia destinations.
        d = Diagonal(_sjp(T, n, 1, 6)[:, 1])
        C = copy(c0)
        mul!(C, d, A, α, β)
        @test isapprox(C, α * (d * a) + β * c0; rtol=_sjp_tol(T))
        C = copy(c0)
        mul!(C, A, d)
        @test isapprox(C, a * d; rtol=_sjp_tol(T))
        close(A)
    end
end

@testset "Symmetric and Hermitian wrappers" begin
    for T in SJP_TYPES, S in (Symmetric, Hermitian), uplo in (:U, :L)
        n = 5
        a = _sjp(T, n, n)
        # Hermitian reads only the real part of a complex diagonal.
        A = FaerMatrix(a)
        sa = S(a, uplo)
        j = _sjp(T, n, n, 2)
        J = FaerMatrix(j)
        for (x, ref) in (
            (S(A, uplo) * j, sa * j),
            (j * S(A, uplo), j * sa),
            (S(A, uplo) * A, sa * a),
            (A * S(A, uplo), a * sa),
            (S(A, uplo) * j', sa * j'),
            (S(A, uplo) * faerview(copy(j)), sa * j),
            (S(j, uplo) * A, S(j, uplo) * a),
            (A * S(j, uplo), a * S(j, uplo)),
            (S(A, uplo) * S(J, uplo), sa * S(j, uplo)),
            (S(A, uplo) * UpperTriangular(A), sa * UpperTriangular(a)),
            (LowerTriangular(A) * S(A, uplo), LowerTriangular(a) * sa),
            (S(view(A, 1:n, 1:n), uplo) * j, sa * j),
        )
            @test x isa FaerMatrix{T}
            @test _sjp_close(x, ref, T)
        end
        v = _sjp(T, n, 1, 3)[:, 1]
        y = S(A, uplo) * v
        @test y isa FaerVector{T}
        @test _sjp_close(y, sa * v, T)
        for dest in (zeros(T, n), FaerVector(zeros(T, n))), x in (v, FaerVector(v))
            @test mul!(dest, S(A, uplo), x, T(2), T(0)) === dest
            @test _sjp_close(dest, 2 * (sa * v), T)
        end
        w = FaerVector(zeros(T, n))
        mul!(w, S(a, uplo), FaerVector(v))
        @test _sjp_close(w, sa * v, T)
        c0 = _sjp(T, n, n, 4)
        α, β = T(2), T(-1)
        for (x, y, rx, ry) in (
            (S(A, uplo), j, sa, j),
            (j, S(A, uplo), j, sa),
            (S(j, uplo), A, S(j, uplo), a),
            (A, S(j, uplo), a, S(j, uplo)),
        )
            C = copy(c0)
            @test mul!(C, x, y, α, β) === C
            @test isapprox(C, α * (rx * ry) + β * c0; rtol=_sjp_tol(T))
            D = FaerMatrix(c0)
            @test mul!(D, x, y) === D
            @test _sjp_close(D, rx * ry, T)
            close(D)
        end
        # The destination is the wrapped parent.
        B = FaerMatrix(a)
        mul!(B, S(B, uplo), J)
        @test _sjp_close(B, sa * j, T)
        foreach(close, (A, J, B))
    end
end

@testset "Banded wrappers with native operands" begin
    for T in SJP_TYPES
        n = 5
        a = _sjp(T, n, n)
        A = FaerMatrix(a)
        dv = _sjp(T, n, 1, 2)[:, 1]
        ev = _sjp(T, n - 1, 1, 3)[:, 1]
        for M in (
            Bidiagonal(dv, ev, :U),
            Bidiagonal(dv, ev, :L),
            Tridiagonal(ev, dv, reverse(ev)),
            SymTridiagonal(dv, ev),
        )
            T <: Complex && M isa SymTridiagonal && continue
            x = M * A
            @test x isa FaerMatrix{T}
            @test _sjp_close(x, M * a, T)
            y = A * M
            @test y isa FaerMatrix{T}
            @test _sjp_close(y, a * M, T)
            v = FaerVector(dv)
            z = M * v
            @test z isa FaerVector{T}
            @test _sjp_close(z, M * dv, T)
            C = zeros(T, n, n)
            @test mul!(C, M, A) === C
            @test isapprox(C, M * a; rtol=_sjp_tol(T))
        end
        close(A)
    end
end

@testset "Native matrices ± Julia Diagonal" begin
    for T in SJP_TYPES
        n = 4
        a = _sjp(T, n, n)
        A = FaerMatrix(a)
        d = Diagonal(_sjp(T, n, 1, 2)[:, 1])
        V = view(A, 1:n, 1:n)
        for (x, ref) in (
            (A + d, a + d),
            (d + A, d + a),
            (A - d, a - d),
            (d - A, d - a),
            (V + d, a + d),
            (d - V, d - a),
        )
            @test x isa FaerMatrix{T}
            @test _sjp_close(x, ref, T)
        end
        @test_throws DimensionMismatch A + Diagonal(ones(T, n + 1))
        # Promotion: native for native scalars, Julia's generic result otherwise.
        x = A + Diagonal(ones(ComplexF64, n))
        @test x isa FaerMatrix{ComplexF64}
        @test _sjp_close(x, a + I, ComplexF64)
        y = A + Diagonal(ones(BigFloat, n))
        @test eltype(y) == promote_type(T, BigFloat)
        @test y ≈ a + I
        close(A)
    end
end

@testset "Structured Julia products allocate O(n²) at most" begin
    n = 64
    a = _sjp(Float64, n, n)
    A = FaerMatrix(a)
    j = _sjp(Float64, n, n, 2)
    jv = j[:, 1]
    C = zeros(n, n)
    y = zeros(n)
    Y = FaerVector(zeros(n))
    bound = 16 * 1024
    for f in (
        () -> UpperTriangular(A) * j,
        () -> j * LowerTriangular(A),
        () -> UnitUpperTriangular(A) * j',
        () -> UpperTriangular(A)' * j,
        () -> UpperTriangular(A) * faerview(j),
        () -> UpperTriangular(A) * jv,
        () -> Symmetric(A) * j,
        () -> A * Symmetric(j),
        () -> Symmetric(A) * A,
        () -> Hermitian(A) * A,
        () -> Symmetric(j) * A,
        () -> Symmetric(A) * Symmetric(A),
        () -> Symmetric(A) * UpperTriangular(A),
        () -> Symmetric(A) * jv,
        () -> A + Diagonal(jv),
        () -> Diagonal(jv) - A,
        () -> mul!(C, UpperTriangular(A), j),
        () -> mul!(C, UpperTriangular(A), A),
        () -> mul!(C, UpperTriangular(A), UpperTriangular(j)),
        () -> mul!(C, Symmetric(A), j),
        () -> mul!(C, A, Symmetric(j)),
        () -> mul!(C, Diagonal(jv), A),
        () -> mul!(C, A, Diagonal(jv)),
        () -> mul!(y, UpperTriangular(A), jv),
        () -> mul!(y, Symmetric(A), jv),
        () -> mul!(Y, Hermitian(A), jv),
    )
        f()
        @test @allocated(f()) < bound
    end
    # Banded products copy the native operand to Julia once (no banded kernel
    # in faer) and return a native result.
    for f in (
        () -> Bidiagonal(jv, jv[1:(end - 1)], :U) * A,
        () -> A * Tridiagonal(jv[2:end], jv, jv[2:end]),
        () -> SymTridiagonal(jv, jv[2:end]) * A,
        () -> mul!(C, Bidiagonal(jv, jv[1:(end - 1)], :U), A),
        () -> Bidiagonal(jv, jv[1:(end - 1)], :U) * Y,
    )
        f()
        @test @allocated(f()) < 4 * n * n * sizeof(Float64)
    end
    close(A)
    close(Y)
end
