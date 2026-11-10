using Colmat, Test, LinearAlgebra, SparseArrays, Random
using LinearAlgebra: Givens

@testset "Native Givens construction and normalization" begin
    rng = MersenneTwister(613)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        R = real(T)
        pairs = [(randn(rng, T), randn(rng, T)) for _ in 1:12]
        if T <: Complex
            tiny = R(floatmin(R) / 2)
            append!(
                pairs,
                [
                    (T(2 + 3im), zero(T)),
                    (zero(T), T(-2 + im)),
                    (T(tiny + tiny * im), T(-tiny + tiny * im)),
                ],
            )
        end
        append!(
            pairs,
            [
                (T(0), T(0)),
                (T(-2), T(0)),
                (T(0), T(-3)),
                (T(-2), T(5)),
                (T(2), T(-5)),
                (T(floatmin(R)), T(floatmin(R))),
                (T(nextfloat(zero(R))), T(nextfloat(zero(R)))),
                (T(floatmax(R) / 4), T(floatmax(R) / 4)),
            ],
        )
        for (p, q) in pairs, (i, j) in ((1, 2), (2, 1))
            v = T[p, q]
            native = FaerVector(v)
            g, r = givens(native, i, j)
            reference, expected = givens(v, i, j)
            @test g isa Givens{T}
            @test g.i1 == reference.i1 && g.i2 == reference.i2
            @test g.c ≈ reference.c
            @test g.s ≈ reference.s
            @test r ≈ expected
            @test abs2(g.c) + abs2(g.s) ≈ one(R)
            scale = max(abs(p), abs(q))
            if !iszero(scale)
                normalized = v / scale
                out = g * normalized
                @test abs(out[j]) <= 20eps(R)
                @test abs(out[i]) ≈ norm(normalized)
            end
            @test Vector(native) == v
        end
        a = randn(rng, T, 5, 4)
        native = FaerMatrix(a)
        for (i, j) in ((1, 4), (4, 1)), col in 1:4
            g, r = givens(native, i, j, col)
            result = Matrix(g * native)
            @test result[i, col] ≈ r
            @test abs(result[j, col]) <= 20eps(R) * norm(a)
            @test Matrix(native) == a
        end
    end
end
@testset "Jacobi kernels implement standard Givens application" begin
    rng = MersenneTwister(83)
    for T in (Float32, Float64, ComplexF32, ComplexF64), shape in ((2, 2), (7, 9), (9, 7))
        m, n = shape
        a = randn(rng, T, m, n)
        g, _ = givens(randn(rng, T), randn(rng, T), 1, 2)
        native = FaerMatrix(a)
        @test Matrix(g * native) ≈ g * a
        @test Matrix(g' * native) ≈ g' * a
        @test Matrix(native * g) ≈ a * g
        @test Matrix(native * g') ≈ a * g'
        @test Matrix(native) == a
        left = copy(native)
        @test lmul!(g, left) === left
        @test Matrix(left) ≈ g * a
        @test lmul!(g', left) === left
        @test Matrix(left) ≈ a
        right = copy(native)
        @test rmul!(right, g) === right
        @test Matrix(right) ≈ a * g
        @test rmul!(right, g') === right
        @test Matrix(right) ≈ a
        vector = randn(rng, T, m)
        v = FaerVector(vector)
        @test g * v isa FaerVector
        @test Vector(g * v) ≈ g * vector
        @test lmul!(g, v) === v
        @test Vector(v) ≈ g * vector
        row = FaerRow(vector)
        @test row * g isa FaerRow
        @test Matrix(row * g) ≈ reshape(vector, 1, :) * g
        @test rmul!(row, g) === row
        @test Matrix(row) ≈ reshape(vector, 1, :) * g
        # Standard composed rotations reuse our Givens apply methods.
        composed = g * g'
        @test Matrix(composed * native) ≈ a
        @test Matrix(native * composed) ≈ a
    end
end
@testset "Native Givens views, conversion, and validation" begin
    rng = MersenneTwister(37)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        a = randn(rng, T, 8, 10)
        native = FaerMatrix(a)
        v = view(native, 8:-2:2, 10:-2:2)
        g, _ = givens(v, 1, 3, 2)
        expected = copy(a)
        selected = view(expected, 8:-2:2, 10:-2:2)
        lmul!(g, selected)
        @test lmul!(g, v) === v
        @test Matrix(native) ≈ expected
        rmul!(selected, g)
        @test rmul!(v, g) === v
        @test Matrix(native) ≈ expected
        @test Matrix(g * v) ≈ g * Matrix(v)
        vector = FaerVector(randn(rng, T, 11))
        sub = view(vector, 11:-2:1)
        g, _ = givens(sub, 2, 5)
        expected = Vector(vector)
        lmul!(g, view(expected, 11:-2:1))
        @test lmul!(g, sub) === sub
        @test Vector(vector) ≈ expected
        @test Vector(g * sub) ≈ g * Vector(sub)
        resize!(vector, 12)
        @test_throws InvalidStateException lmul!(g, sub)
        close(v)
        @test_throws ArgumentError g * v
        @test_throws ArgumentError givens(v, 1, 2, 1)
        @test_throws BoundsError givens(native, 0, 2, 1)
        @test_throws ArgumentError givens(native, 1, 1, 1)
        @test_throws BoundsError givens(vector, 1, 50)
        invalid = FaerVector(T[0, 1])
        invalid[1] = T(Inf)
        @test_throws ArgumentError givens(invalid, 1, 2)
        before = Matrix(native)
        bad = Givens(0, 2, one(T), zero(T))
        @test_throws ArgumentError lmul!(bad, native)
        bad = Givens(1, 100, one(T), zero(T))
        @test_throws DimensionMismatch rmul!(native, bad)
        @test Matrix(native) == before
        close(native)
        @test_throws ArgumentError g * native
        @test_throws ArgumentError lmul!(g, native)
    end
    g, _ = givens(1 + im, 2 - im, 1, 2)
    a = FaerMatrix(Float32[1 2; 3 4])
    @test eltype(g * a) === ComplexF64
    @test Matrix(g * a) ≈ g * Matrix(a)
    before = Matrix(a)
    @test_throws InexactError lmul!(g, a)
    @test Matrix(a) == before
    bad = Givens(1, 2, 1 + im, 0 + im)
    @test_throws ArgumentError bad * a
    @test isempty(
        detect_ambiguities(Colmat, Base, LinearAlgebra, SparseArrays; recursive=true)
    )
end

@testset "Rotation promotion and empty axes" begin
    rng = MersenneTwister(745)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        S in (Float32, Float64, ComplexF32, ComplexF64)

        a = randn(rng, T, 4, 5)
        g, _ = givens(randn(rng, S), randn(rng, S), 1, 3)
        native = FaerMatrix(a)
        @test eltype(g * native) === promote_type(T, S)
        @test Matrix(g * native) ≈ g * a
        @test eltype(native * g) === promote_type(T, S)
        @test Matrix(native * g) ≈ a * g
        v = FaerVector(a[:, 1])
        @test eltype(g * v) === promote_type(T, S)
        @test Vector(g * v) ≈ g * a[:, 1]
        row = FaerRow(a[:, 1])
        @test eltype(row * g) === promote_type(T, S)
        @test Matrix(row * g) ≈ reshape(a[:, 1], 1, :) * g
        reversed = Givens(3, 1, g.c, g.s)
        @test Matrix(reversed * native) ≈ reversed * a
        @test Matrix(native * reversed) ≈ a * reversed
    end
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        g, _ = givens(T(1), T(2), 1, 3)
        a = FaerMatrix(zeros(T, 3, 0))
        @test size(g * a) == (3, 0)
        @test lmul!(g, a) === a
        b = FaerMatrix(zeros(T, 0, 3))
        @test size(b * g) == (0, 3)
        @test rmul!(b, g) === b
    end
end
