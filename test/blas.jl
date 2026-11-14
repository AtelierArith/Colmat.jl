# Explicit scalar loops are independent of both Colmat and fallback BLAS.
function product(A, B)
    return [
        sum(
            (A[i, k] * B[k, j] for k in axes(A, 2));
            init=zero(promote_type(eltype(A), eltype(B))),
        ) for i in axes(A, 1), j in axes(B, 2)
    ]
end
op(A, t) =
    if t == 'N'
        A
    elseif t == 'T'
        transpose(A)
    else
        adjoint(A)
    end

Random.seed!(782)
@testset "Ordinary Julia arrays" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset "$T GEMM" begin
            for ta in ('N', 'T', 'C'), tb in ('N', 'T', 'C')
                m, n, k = 9, 7, 5
                A = randn(T, ta == 'N' ? (m, k) : (k, m))
                B = randn(T, tb == 'N' ? (k, n) : (n, k))
                for alpha in (zero(T), T(1.3)), beta in (zero(T), T(-0.7), one(T))
                    C = randn(T, m, n)
                    old = copy(C)
                    if beta == 0
                        fill!(C, T(NaN))
                    end
                    BLAS.gemm!(ta, tb, alpha, A, B, beta, C)
                    @test C ≈ alpha * product(op(A, ta), op(B, tb)) + beta * old
                end
            end
            # Padded column-major storage and subarrays.
            A = view(randn(T, 14, 10), 2:10, 2:6)
            B = view(randn(T, 10, 12), 2:6, 3:9)
            storage = fill(T(93), 14, 12)
            C = view(storage, 2:10, 3:9)
            mul!(C, A, B)
            @test C ≈ product(A, B)
            @test all(==(T(93)), storage[1, :])
            @test A * B ≈ product(A, B)
            @test A * B isa Matrix{T}
            @test zeros(T, 3, 0) * zeros(T, 0, 4) == zeros(T, 3, 4)
        end
        @testset "$T vectors and triangles" begin
            for t in ('N', 'T', 'C')
                A = randn(T, 8, 5)
                M = op(A, t)
                x = randn(T, size(M, 2))
                y = randn(T, size(M, 1))
                old = copy(y)
                BLAS.gemv!(t, T(2), A, x, T(-1), y)
                @test y ≈ vec(2product(M, reshape(x, :, 1))) - old
                # BLAS wrappers compute correct starts for negative strides.
                xs = reverse(x)
                ys = reverse(old)
                BLAS.gemv!(
                    t, T(2), A, view(xs, length(xs):-1:1), T(-1), view(ys, length(ys):-1:1)
                )
                @test reverse(ys) ≈ y
            end
            for u in ('U', 'L'), t in ('N', 'T', 'C'), d in ('N', 'U'), side in ('L', 'R')
                A = randn(T, 6, 6)
                for i in 1:6
                    A[i, i] += T(8)
                end
                tri = if u == 'U'
                    (d == 'U' ? UnitUpperTriangular(A) : UpperTriangular(A))
                else
                    (d == 'U' ? UnitLowerTriangular(A) : LowerTriangular(A))
                end
                M = Matrix(op(tri, t))
                B = randn(T, 6, 6)
                old = copy(B)
                alpha = T(0.7)
                BLAS.trmm!(side, u, t, d, alpha, A, B)
                @test B ≈ alpha * (side == 'L' ? product(M, old) : product(old, M))
                X = randn(T, 6, 6)
                rhs = side == 'L' ? product(M, X) : product(X, M)
                BLAS.trsm!(side, u, t, d, alpha, A, rhs)
                @test rhs ≈ alpha * X
            end
            x = randn(T, 15)
            y = randn(T, 15)
            if T <: Real
                @test BLAS.dot(x, y) ≈ sum(x[i] * y[i] for i in eachindex(x))
            else
                @test BLAS.dotc(x, y) ≈ sum(conj(x[i]) * y[i] for i in eachindex(x))
                @test BLAS.dotu(x, y) ≈ sum(x[i] * y[i] for i in eachindex(x))
            end
        end
        @testset "$T decomposition interoperability" begin
            A = randn(T, 9, 9) + 9I
            b = randn(T, 9)
            @test norm(A * (A \ b) - b) < 1e-4
            F = lu(A)
            @test F.L * F.U ≈ A[F.p, :]
            S = svd(A)
            @test S.U * Diagonal(S.S) * S.Vt ≈ A
        end
    end
end
