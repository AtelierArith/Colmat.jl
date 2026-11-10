using Test, LinearAlgebra, Random, Colmat
@testset "Typed Rust matrices" begin
    rng = MersenneTwister(81)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        R = typeof(real(zero(T)))
        tol = 100eps(R)
        @testset "$T storage and arithmetic" begin
            a = randn(rng, T, 3, 3)
            b = randn(rng, T, 3, 3)
            A, B = FaerMatrix(a), FaerMatrix(b)
            @test eltype(A) === T
            @test A isa FaerMatrix{T}
            @test Matrix(A) == a
            @test Array(A) == a
            @test Matrix{ComplexF64}(A) == ComplexF64.(a)
            @test A[2, 1] == a[2, 1]
            A[2, 1] = T(4)
            @test A[2, 1] == T(4)
            A = FaerMatrix(a)
            @test Matrix(copy(A)) == a
            @test Matrix(deepcopy(A)) == a
            @test Matrix(+A) == a
            @test eltype(similar(A)) === T
            @test Matrix(zero(A)) == zero(a)
            C = copy(A)
            copyto!(C, B)
            @test Matrix(C) == b
            @test copyto!(C, C) === C
            @test Matrix(fill!(C, 2)) == fill(T(2), 3, 3)
            @test Matrix(A + B) ≈ a + b
            @test Matrix(A - B) ≈ a - b
            @test Matrix(A * B) ≈ a * b
            @test Matrix(2A) ≈ 2a
            @test Matrix(A / 2) ≈ a / 2
            @test eltype(2A) === T
            @test eltype(A / 2) === T
            @test Matrix(copy(transpose(A))) ≈ transpose(a)
            @test Matrix(copy(adjoint(A))) ≈ adjoint(a)
            @test Matrix(conj(A)) ≈ conj(a)
            @test Matrix(A' * B) ≈ a' * b
            @test Matrix(transpose(A) * B) ≈ transpose(a) * b
            @test sum(A) ≈ sum(a)
            @test sum(A) isa T
            @test norm(A) ≈ norm(a)
            @test norm(A) isa R
            @test norm(A, 1) ≈ norm(a, 1)
            @test norm(A, Inf) ≈ norm(a, Inf)
            @test det(A) ≈ det(a)
            @test tr(A) ≈ tr(a)
            @test Matrix(kron(A, B)) ≈ kron(a, b)
            @test all(isfinite, A)
            @test !any(isnan, A)
            C = copy(A)
            mul!(C, A, B, T(2), T(3))
            @test Matrix(C) ≈ 2a * b + 3a
            C = copy(A)
            mul!(C, C, B)
            @test Matrix(C) ≈ a * b
            @test_throws BoundsError A[4, 1]
            close(C)
            @test_throws ArgumentError Matrix(C)
        end
        @testset "$T decompositions" begin
            a = randn(rng, T, 4, 4) + T(4) * I
            b = randn(rng, T, 4, 2)
            A, B = FaerMatrix(a), FaerMatrix(b)
            for ctor in (lu, full_piv_lu, qr)
                F = ctor(A)
                @test F isa FaerFactorization{T}
                @test Matrix(F) ≈ a
                @test eltype(F \ B) === T
                @test Matrix(F \ B) ≈ a \ b
                @test Matrix(F' \ B) ≈ a' \ b
                @test Matrix(transpose(F) \ B) ≈ transpose(a) \ b
                @test F \ b[:, 1] ≈ a \ b[:, 1]
                @test Matrix(inv(F)) ≈ inv(a)
                @test Matrix(copy(F)) ≈ a
                @test Matrix(deepcopy(F)) ≈ a
                if ctor === lu
                    @test Matrix(F.L * F.U) ≈ a[F.p, :]
                elseif ctor === full_piv_lu
                    @test Matrix(F.L * F.U) ≈ a[F.p, F.q]
                else
                    @test Matrix(F.Qthin * F.R) ≈ a
                end
                C = copy(B)
                ldiv!(C, F, C)
                @test Matrix(C) ≈ a \ b
                close(F)
                @test_throws ArgumentError Matrix(F)
            end
            h = a * a' + I
            H = FaerMatrix(h)
            @test ishermitian(H)
            for ctor in (cholesky, ldlt, bunchkaufman)
                F = ctor(Hermitian(H))
                @test F isa FaerFactorization{T}
                @test Matrix(F) ≈ h
                @test Matrix(F \ B) ≈ h \ b
                if ctor === cholesky
                    @test Matrix(F.L) * Matrix(F.U) ≈ h
                elseif ctor === ldlt
                    @test Matrix(F.L * F.D * copy(F.L')) ≈ h
                else
                    @test Matrix(F.L * F.D * copy(F.L')) ≈ h[F.p, F.p]
                end
            end
            for (m, n) in ((5, 3), (3, 5), (0, 3), (3, 0), (0, 0))
                a = randn(rng, T, m, n)
                A = FaerMatrix(a)
                for full in (false, true)
                    F = svd(A; full)
                    k = min(m, n)
                    @test eltype(F.U) === T
                    @test eltype(F.S) === R
                    @test F.U[:, 1:k] * Diagonal(F.S) * F.Vt[1:k, :] ≈ a
                    @test F.U' * F.U ≈ Matrix{T}(I, size(F.U, 2), size(F.U, 2)) atol = tol
                    @test F.Vt * F.Vt' ≈ Matrix{T}(I, size(F.Vt, 1), size(F.Vt, 1)) atol =
                        tol
                    @test F.S ≈ svdvals(a)
                end
                @test svdvals(A) ≈ svdvals(a)
                if m > 0 && n > 0
                    b = randn(rng, T, m, 2)
                    @test Matrix(qr(A, ColumnNorm()) \ FaerMatrix(b)) ≈ pinv(a) * b
                    @test Matrix(A \ FaerMatrix(b)) ≈ pinv(a) * b
                end
            end
        end
    end
    @testset "scalar promotion" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64),
            S in (Float32, Float64, ComplexF32, ComplexF64)

            a = randn(rng, T, 2, 2) + T(4) * I
            b = randn(rng, S, 2, 2)
            A, B = FaerMatrix(a), FaerMatrix(b)
            P = promote_type(T, S)
            @test eltype(A + B) === P
            @test Matrix(A + B) ≈ a + b
            @test Matrix(A * B) ≈ a * b
            @test eltype(A \ B) === P
            @test Matrix(A \ B) ≈ Matrix{P}(a) \ Matrix{P}(b)
            @test Matrix(lu(A) \ B) ≈ a \ b rtol = 100eps(typeof(real(zero(T))))
            C = FaerMatrix{P}(A)
            @test eltype(C) === P
            @test Matrix(C) ≈ a
        end
    end
    @test isempty(Test.detect_ambiguities(Colmat; recursive=true))
end

@testset "Typed eigen, structured algebra and inner products" begin
    rng = MersenneTwister(513)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        R = typeof(real(zero(T)))
        tol = 200eps(R)
        @testset "$T" begin
            a = randn(rng, T, 4, 4)
            A = FaerMatrix(a)
            @test dot(A, A) ≈ dot(a, a)
            @test dot(A, A) isa T
            @test Matrix(sum(A; dims=1)) ≈ sum(a; dims=1)
            @test Matrix(sum(A; dims=2)) ≈ sum(a; dims=2)
            @test sum(abs2, A) ≈ sum(abs2, a)
            @test sum(abs, A) ≈ sum(abs, a)
            F = eigen(A)
            @test eltype(F.values) === Complex{R}
            @test a * F.vectors ≈ F.vectors * Diagonal(F.values) atol = tol
            @test eigvals(A) ≈ F.values
            h = a * a' + I
            H = FaerMatrix(h)
            F = eigen(Hermitian(H))
            @test eltype(F.values) === R
            @test eltype(F.vectors) === T
            @test h * F.vectors ≈ F.vectors * Diagonal(F.values) atol = tol
            @test F.vectors' * F.vectors ≈ Matrix{T}(I, 4, 4) atol = tol
            b = randn(rng, T, 4, 4)
            b = b * b' + I
            B = FaerMatrix(b)
            F = eigen(A, B)
            @test a * F.vectors ≈ b * F.vectors * Diagonal(F.values) atol = tol
            F = eigen(Hermitian(H), Hermitian(B))
            @test h * F.vectors ≈ b * F.vectors * Diagonal(F.values) atol = tol
            @test F.vectors' * b * F.vectors ≈ Matrix{T}(I, 4, 4) atol = tol
            @test eigvals(Hermitian(H), Hermitian(B)) ≈ F.values
            expected, phase = logabsdet(a)
            value, actualphase = logabsdet(A)
            @test value ≈ expected
            @test actualphase ≈ phase
            @test det(lu(A)) ≈ det(a)
            for wrapper in
                (LowerTriangular, UpperTriangular, UnitLowerTriangular, UnitUpperTriangular)
                tri = wrapper(FaerMatrix(a + T(4) * I))
                rhs = randn(rng, T, 4, 2)
                @test Matrix(tri \ FaerMatrix(rhs)) ≈ wrapper(a + T(4) * I) \ rhs
                @test Matrix(inv(tri)) ≈ inv(wrapper(a + T(4) * I))
            end
            skew = a - transpose(a)
            @test pfaffian(FaerMatrix(skew))^2 ≈ det(skew) atol = tol
            @test eltype(copy(transpose(A))) === T
            @test eltype(copy(adjoint(A))) === T
            if T <: Complex
                H[1, 1] = T(real(H[1, 1]), NaN)
                @test eigvals(Hermitian(H)) ≈ eigvals(Hermitian(h))
                @test Matrix(cholesky(Hermitian(H))) ≈ h
            end
        end
    end
    @test isempty(Test.detect_ambiguities(Colmat; recursive=true))
end

@testset "Typed failure and selected-triangle semantics" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        @test_throws SingularException lu(FaerMatrix(T[1 2; 2 4]))
        @test !issuccess(lu(FaerMatrix(T[1 2; 2 4]); check=false))
        F = bunchkaufman(Hermitian(FaerMatrix(T[0 1; 1 0])))
        @test Matrix(F.L * F.D * copy(F.L')) ≈ T[0 1; 1 0][F.p, F.p]
        @test F \ T[1, 2] ≈ T[2, 1]
        @test !issuccess(bunchkaufman(Hermitian(FaerMatrix(zeros(T, 2, 2))); check=false))
        @test_throws Exception inv(qr(FaerMatrix(zeros(T, 2, 2))))
        for (m, n) in ((0, 3), (3, 0), (0, 0))
            A = FaerMatrix(zeros(T, m, n))
            @test size(Matrix(A)) == (m, n)
            @test norm(A) == 0
            @test dot(A, A) == 0
        end
    end
end
