using Test
using LinearAlgebra
using Random
using LinearAlgebra: LAPACK

# The adapter factors Cholesky's diagonal panels in blocks (16 complex / 32 real
# on AVX2) and saves only the diagonal blocks of the unowned triangle. These
# sizes cross every internal boundary so a block-size change cannot silently
# break the factor or the untouched-triangle contract.
@testset "factorizations across internal block boundaries" begin
    rng = MersenneTwister(0x5eed)
    sizes = (1, 15, 16, 17, 31, 32, 33, 47, 48, 63, 64, 65, 95, 96, 97, 127, 128, 129)
    for T in (Float64, ComplexF64), n in sizes
        A = randn(rng, T, n, n)
        M = A * A' + n * I
        sentinel = T <: Complex ? T(123 + 456im) : T(123)

        # UPLO='U': factor correct, unowned strict lower untouched.
        buf = Matrix(M)
        for j in 2:n, i in 1:j-1
            buf[j, i] = sentinel * (i + j)
        end
        lower = [buf[j, i] for j in 2:n for i in 1:j-1]
        LAPACK.potrf!('U', buf)
        U = UpperTriangular(buf)
        @test U' * U ≈ M
        @test [buf[j, i] for j in 2:n for i in 1:j-1] == lower

        # UPLO='L': factor correct, unowned strict upper untouched.
        buf = Matrix(M)
        for j in 2:n, i in 1:j-1
            buf[i, j] = sentinel * (i + j)
        end
        upper = [buf[i, j] for j in 2:n for i in 1:j-1]
        LAPACK.potrf!('L', buf)
        L = LowerTriangular(buf)
        @test L * L' ≈ M
        @test [buf[i, j] for j in 2:n for i in 1:j-1] == upper

        # LU: Julia's pivot vector reproduces the final permutation.
        F = lu(A)
        @test F.L * F.U ≈ A[F.p, :]
        @test sort(F.p) == collect(1:n)

        # A non-positive pivot past the first panel reports its global index.
        diagonal = Matrix{T}(I, 200, 200)
        diagonal[150, 150] = -one(T)
        for uplo in ('U', 'L')
            _, info = LAPACK.potrf!(uplo, copy(diagonal))
            @test info == 150
        end

        # QR factor and explicit Q across the panel boundaries.
        Fqr = qr(A)
        @test Matrix(Fqr.Q) * Fqr.R ≈ A
        Q = Matrix(Fqr.Q)
        @test Q' * Q ≈ Matrix{T}(I, n, n)

        # Thin SVD: the square 'A' job takes the thin native route.
        U2, s, V = svd(A)
        @test U2 * Diagonal(s) * V' ≈ A
    end
end
