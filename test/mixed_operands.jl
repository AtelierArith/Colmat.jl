using Colmat, Test, LinearAlgebra

const MIXED_TYPES = (Float32, Float64, ComplexF32, ComplexF64)
function _mx(::Type{T}, m, n, s=1) where {T}
    return T.([
        sin(s * i + j) + (T <: Complex ? im * cos(i - s * j) : 0) for i in 1:m, j in 1:n
    ])
end

@testset "Native ± and * with Julia strided matrices" begin
    for T in MIXED_TYPES
        a = _mx(T, 5, 4)
        j = _mx(T, 5, 4, 2)
        k = _mx(T, 4, 3, 3)
        A = FaerMatrix(a)
        for (x, ref) in (
            (A + j, a + j),
            (j + A, j + a),
            (A - j, a - j),
            (j - A, j - a),
            (A * k, a * k),
            (permutedims(j) * A, permutedims(j) * a),
        )
            @test x isa FaerMatrix{T}
            @test Matrix(x) ≈ ref
        end
        # Strided views of Julia arrays are borrowed too; others still work.
        big = _mx(T, 10, 8, 4)
        sv = view(big, 1:2:9, 2:2:8)
        @test Matrix(A + sv) ≈ a + sv
        @test Matrix(A + view(big, [1, 3, 5, 7, 9], 1:4)) ≈ a + big[[1, 3, 5, 7, 9], 1:4]
        @test_throws DimensionMismatch A + _mx(T, 4, 4)
        @test Matrix(A) == a
        close(A)
    end
end

@testset "Mixed-precision promotion converts natively" begin
    for (S, T) in (
        (Float32, Float64),
        (Float32, ComplexF64),
        (Float64, ComplexF32),
        (ComplexF32, ComplexF64),
    )
        a = _mx(S, 6, 6)
        b = _mx(T, 6, 6, 2)
        A, B = FaerMatrix(a), FaerMatrix(b)
        P = promote_type(S, T)
        for (x, ref) in ((A + B, a + b), (B - A, b - a), (A * B, a * b), (A' * B, a' * b))
            @test x isa FaerMatrix{P}
            @test Matrix(x) ≈ ref
        end
        foreach(close, (A, B))
    end
    # Complex-to-real promotion is not native; it keeps the checked path.
    C = FaerMatrix(ComplexF64[1 2; 3 4])
    @test Matrix(Colmat._coerce(C, Float64)) == [1.0 2; 3 4]
    @test_throws InexactError Colmat._coerce(FaerMatrix(ComplexF64[1im 0; 0 1]), Float64)
    n = 500
    A = FaerMatrix(_mx(Float64, n, n))
    B = FaerMatrix(_mx(Float32, n, n, 2))
    A + B
    @test @allocated(A + B) < 64 * 1024
    j = _mx(Float64, n, n, 3)
    A + j
    A * j
    @test @allocated(A + j) < 64 * 1024
    @test @allocated(A * j) < 64 * 1024
    foreach(close, (A, B, C))
end

@testset "mul! into a native vector with Julia operands" begin
    for T in MIXED_TYPES
        a = _mx(T, 7, 5)
        x = _mx(T, 5, 1, 2)[:, 1]
        A = FaerMatrix(a)
        y = FaerVector(ones(T, 7))
        @test mul!(y, A, x) === y
        @test Vector(y) ≈ a * x
        mul!(y, A, x, T(2), T(1))
        @test Vector(y) ≈ a * x .+ 2 .* (a * x)
        mul!(y, a, FaerVector(x))
        @test Vector(y) ≈ a * x
        foreach(close, (A, y))
    end
    n = 2000
    A = FaerMatrix(_mx(Float64, n, n))
    x = _mx(Float64, n, 1, 2)[:, 1]
    y = FaerVector(zeros(n))
    mul!(y, A, x)
    @test @allocated(mul!(y, A, x)) < 64 * 1024
    foreach(close, (A, y))
end
