using Test, Colmat, LinearAlgebra, SparseArrays, Random

@testset "Standalone Jacobi rotations" begin
    rng = MersenneTwister(42)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        R = typeof(real(zero(T)))
        tol = 256eps(R)
        for (p, q) in (
            (T(3), T(4)),
            (T(-3), T(4)),
            (zero(T), T(4)),
            (T(3), zero(T)),
            (zero(T), zero(T)),
            (randn(rng, T), randn(rng, T)),
        )
            J, r = rotg(p, q)
            D = Matrix(J)
            @test J isa FaerJacobiRotation{T}
            @test D * [p, q] ≈ [r, zero(T)] atol = tol * max(abs(p), abs(q), one(R))
            @test D' * D ≈ Matrix{T}(I, 2, 2) atol = tol
            @test Matrix(transpose(J)) == transpose(D)
            @test Matrix(conj(J)) == conj(D)
            @test Matrix(J') == D'
            @test Matrix(inv(J)) == D'
            @test J * [p, q] ≈ D * [p, q] atol = tol
            A = randn(rng, T, 2, 3)
            @test Matrix(J * A) ≈ D * A atol = tol
            B = randn(rng, T, 3, 2)
            @test Matrix(B * J) ≈ B * D atol = tol
            native = FaerMatrix(A)
            @test lmul!(J, native) === native
            @test Matrix(native) ≈ D * A atol = tol
            host = copy(A)
            @test lmul!(J, host) === host
            @test host ≈ D * A atol = tol
            host = copy(B)
            @test rmul!(host, J) === host
            @test host ≈ B * D atol = tol
            block = randn(rng, T, 2, 2)
            left = apply_on_the_left_2x2(
                J, block[1, 1], block[1, 2], block[2, 1], block[2, 2]
            )
            right = apply_on_the_right_2x2(
                J, block[1, 1], block[1, 2], block[2, 1], block[2, 2]
            )
            @test [left[1] left[2]; left[3] left[4]] ≈ D * block atol = tol
            @test [right[1] right[2]; right[3] right[4]] ≈ block * D atol = tol
            larger = FaerMatrix(randn(rng, T, 4, 3))
            expected = Matrix(larger)
            G = LinearAlgebra.Givens(J, 2, 4)
            lmul!(G, expected)
            lmul!(G, larger)
            @test Matrix(larger) ≈ expected atol = tol
        end
        @test_throws ArgumentError FaerJacobiRotation(T(2), zero(T))
        @test_throws ArgumentError rotg(T(NaN), one(T))
        J = first(rotg(T(3), T(4)))
        @test_throws DimensionMismatch J * zeros(T, 3)
        @test_throws ArgumentError LinearAlgebra.Givens(J, 1, 1)
        @test_throws BoundsError J[3, 1]
        closeme = FaerMatrix(zeros(T, 2, 0))
        close(closeme)
        @test_throws ArgumentError J * closeme
    end
    for T in (Float32, Float64)
        for (p, q) in (
            (T(3), T(4)), (T(-3), T(4)), (T(3), T(-4)), (zero(T), one(T)), (one(T), zero(T))
        )
            J = make_givens(p, q)
            @test abs((J * [p, q])[2]) <= 256eps(T) * max(abs(p), abs(q))
            @test Matrix(J)' * Matrix(J) ≈ Matrix{T}(I, 2, 2)
        end
        p = nextfloat(zero(T))
        J, r = rotg(p, p)
        @test all(isfinite, Matrix(J))
        @test Matrix(J)' * Matrix(J) ≈ Matrix{T}(I, 2, 2)
        @test isfinite(r)
    end
    J = first(rotg(3.0 + 2im, 4.0 - im))
    @test_throws ArgumentError FaerJacobiRotation(1im, 0im)
    dest = Matrix{Float64}(I, 2, 2)
    old = copy(dest)
    @test_throws InexactError lmul!(J, dest)
    @test dest == old
    parent = FaerMatrix(reshape(collect(1.0:16), 4, 4))
    region = view(parent, 1:2, 2:4)
    original = Matrix(region)
    realJ = make_givens(3.0, 4.0)
    @test lmul!(realJ, region) === region
    @test Matrix(region) ≈ Matrix(realJ) * original
    @test isempty(
        Test.detect_ambiguities(Colmat, Base, LinearAlgebra, SparseArrays; recursive=true)
    )
end
