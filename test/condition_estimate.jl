using Test, Colmat, LinearAlgebra, SparseArrays

function _cond_sample(T, n)
    A = spdiagm(0 => T.(4 .+ (1:n)), 1 => T.(fill(1, n - 1)), -2 => T.(fill(-2, n - 2)))
    A[1, n] = T(3)
    T <: Complex &&
        (A .+= T(0.5im) .* spdiagm(-1 => T.(1:(n - 1)), 2 => T.(fill(1, n - 2))))
    return A
end

@testset "Estimated sparse 1/Inf condition numbers" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        format in (:csc, :csr),
        n in (1, 2, 6, 40)

        A = n == 1 ? sparse(T[3;;]) : _cond_sample(T, n)
        a = FaerSparseMatrix(A; format)
        R = real(T)
        for p in (1, Inf)
            @test opnorm(a, p) isa R
            @test opnorm(a, p) ≈ opnorm(A, p) rtol = sqrt(eps(R))
            exact = cond(Matrix(A), p)
            est = cond(a, p)
            @test est isa R
            # Higham–Tisseur estimates are lower bounds of the exact value.
            @test est <= exact * (1 + sqrt(eps(R)))
            @test est >= exact / n
            # Tiny inputs are estimated exactly; larger ones may underestimate
            # (as SparseArrays does for some random starts).
            n <= 2 && @test est ≈ exact rtol = sqrt(eps(R))
            # Deterministic: repeated estimates agree.
            @test cond(a, p) == est
        end
        @test Matrix(a) == Matrix(A)
        close(a)
    end
end
@testset "Sparse condition number edge cases follow Colmat dense conventions" begin
    for T in (Float64, ComplexF32)
        R = real(T)
        e = FaerSparseMatrix(spzeros(T, 0, 0))
        @test cond(e, 1) === zero(R)
        @test cond(e, Inf) === zero(R)
        s = FaerSparseMatrix(sparse(T[1 2; 2 4]))
        @test cond(s, 1) === R(Inf)
        @test cond(s, Inf) === R(Inf)
        @test Matrix(s) == T[1 2; 2 4]
        z = FaerSparseMatrix(spzeros(T, 3, 3))
        @test cond(z, 1) === R(Inf)
        a = FaerSparseMatrix(_cond_sample(T, 4))
        # As for SparseArrays: no sparse 2-norm condition number or operator norm.
        @test_throws ArgumentError cond(a)
        @test_throws ArgumentError cond(a, 2)
        @test_throws ArgumentError cond(a, 3)
        @test_throws ArgumentError opnorm(a, 2)
        r = FaerSparseMatrix(sparse(T[1 2 3; 4 5 6]))
        @test_throws DimensionMismatch cond(r, 1)
        @test opnorm(r, 1) ≈ opnorm(sparse(T[1 2 3; 4 5 6]), 1)
        @test opnorm(r, Inf) ≈ opnorm(sparse(T[1 2 3; 4 5 6]), Inf)
        foreach(close, (e, s, z, a, r))
        @test_throws ArgumentError cond(a, 1)
        @test_throws ArgumentError opnorm(a, 1)
    end
end

function _tri_sample(T, n)
    a = T[(i == j ? 4 + i : 0.3sin(i + 2j)) for i in 1:n, j in 1:n]
    T <: Complex && (a .+= T(0.2im) .* T[cos(i + 3j) for i in 1:n, j in 1:n])
    return a
end

@testset "Estimated dense triangular 1/Inf condition numbers" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        W in (UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular),
        n in (1, 2, 7, 40),
        wrap in (identity, transpose, adjoint, x -> view(x, :, :))

        R = real(T)
        a = _tri_sample(T, n)
        source = FaerMatrix(a)
        parent = wrap(source)
        w = W(parent)
        dense = Matrix(W(wrap(a)))
        for p in (1, Inf)
            exact = cond(dense, p)
            est = cond(w, p)
            @test est isa R
            @test est <= exact * (1 + sqrt(eps(R)))
            @test est >= exact / n
            n <= 2 && @test est ≈ exact rtol = sqrt(eps(R))
            @test cond(w, p) == est
        end
        @test cond(w) ≈ cond(dense) rtol = sqrt(eps(R))
        @test cond(w, 2) ≈ cond(dense, 2) rtol = sqrt(eps(R))
        @test Matrix(source) == a
        close(source)
    end
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        R = real(T)
        singular = FaerMatrix(T[1 2; 0 0])
        @test cond(UpperTriangular(singular), 1) === R(Inf)
        @test cond(UpperTriangular(singular), Inf) === R(Inf)
        @test cond(UnitUpperTriangular(singular), 1) ≈ cond(T[1 2; 0 1], 1)
        e = FaerMatrix(zeros(T, 0, 0))
        @test cond(UpperTriangular(e), 1) === zero(R)
        @test cond(LowerTriangular(e), Inf) === zero(R)
        @test_throws ArgumentError cond(UpperTriangular(singular), 3)
        close(singular)
        close(e)
    end
end

@testset "Hermitian 2-norm condition numbers from native eigenvalues" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        S in (Hermitian, Symmetric), uplo in (:U, :L),
        n in (1, 3, 9)

        R = real(T)
        a = _tri_sample(T, n)
        # Complex Symmetric is not Hermitian: |λ| are not its singular values.
        a = T <: Complex && S === Symmetric ? a + transpose(a) : a + a'
        a[1, 1] = -a[1, 1]
        source = FaerMatrix(a)
        h = S(source, uplo)
        dense = Matrix(S(a, uplo))
        @test cond(h) isa R
        @test cond(h) ≈ cond(dense) rtol = sqrt(eps(R))
        @test cond(h, 2) ≈ cond(dense, 2) rtol = sqrt(eps(R))
        # Complex Symmetric `inv` has no native LDLᵀ (generic path throws).
        for p in (1, Inf)
            T <: Complex && S === Symmetric && continue
            @test cond(h, p) ≈ cond(dense, p) rtol = sqrt(eps(R))
        end
        close(source)
    end
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        R = real(T)
        z = FaerMatrix(zeros(T, 2, 2))
        @test cond(Hermitian(z)) === R(Inf)
        s = FaerMatrix(T[1 0; 0 0])
        @test cond(Hermitian(s)) === R(Inf)
        e = FaerMatrix(zeros(T, 0, 0))
        @test cond(Hermitian(e)) === zero(R)
        @test_throws ArgumentError cond(Hermitian(s), 3)
        foreach(close, (z, s, e))
    end
end
