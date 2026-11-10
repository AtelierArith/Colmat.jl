using Test, Colmat, LinearAlgebra, SparseArrays
@testset "Native partial spectra" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        R = typeof(real(zero(T)))
        tolerance = R === Float32 ? 2e-4 : 1e-8
        diagonal = T.(1:96)
        op = FaerLinearOperator(Diagonal(diagonal))
        for f in (partial_eigen, partial_self_adjoint_eigen, partial_svd)
            F = f(op, 2; tol=tolerance, max_restarts=20)
            @test F.requested == 2
            @test F.algorithm == :partial
            @test F.nconverged == 2
            @test maximum(F.residuals; init=0.0) < 100tolerance * 96
            if f === partial_svd
                @test F.S ≈ R[96, 95] rtol = 100tolerance
                @test F.U' * F.U ≈ I atol = 100tolerance
                @test F.Vt * F.Vt' ≈ I atol = 100tolerance
            else
                @test F.values ≈ T[96, 95] rtol = 100tolerance
                @test size(F.vectors) == (96, 2)
            end
            empty = f(op, 0)
            @test empty.nconverged == 0
            @test isempty(empty.residuals)
            stopped = f(op, 2; max_restarts=0)
            @test stopped.nconverged == 0
            @test_throws ArgumentError f(op, -1)
            @test_throws ArgumentError f(op, 97)
            @test_throws ArgumentError f(op, 1; tol=NaN)
            @test_throws ArgumentError f(op, 1; max_dim=-1)
            @test_throws DimensionMismatch f(op, 1; v0=ones(T, 95))
            @test_throws ArgumentError f(op, 1; v0=zeros(T, 96))
        end
        small = FaerLinearOperator(Diagonal(T[-2, 1, 4, 3]))
        F = partial_self_adjoint_eigen(small, 2)
        @test F.algorithm == :dense_fallback
        @test F.values ≈ R[4, 3]
        @test maximum(F.residuals) < 100eps(R)
        close(op)
        @test_throws ArgumentError partial_eigen(op, 1)
    end
end
@testset "Partial spectrum layouts and fallback" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        R = typeof(real(zero(T)))
        tol = R === Float32 ? 2e-4 : 1e-8
        for dims in ((5, 3), (3, 5), (1, 1), (0, 0), (0, 3), (3, 0))
            A = reshape(T.(1:prod(dims)), dims)
            op = FaerLinearOperator(A)
            k = min(dims...)
            F = partial_svd(op, k)
            @test F.nconverged == k
            @test F.U * Diagonal(F.S) * F.Vt ≈ A atol = 100tol
            @test eltype(F.S) == R
            @test size(F.Vt) == (k, dims[2])
        end
        # Non-Hermitian diagonal with nonreal eigenvalues exercises complex outputs.
        vals = T <: Complex ? T.(1:96) .+ T(0.5im) : T.(1:96)
        op = FaerLinearOperator(sparse(Diagonal(vals)))
        F = partial_eigen(op, 3; tol, max_restarts=20)
        @test F.nconverged == 3
        @test F.values ≈ reverse(vals[(end - 2):end]) rtol = 100tol
        @test maximum(F.residuals) < 100tol * 96
        @test eltype(F.vectors) == Complex{R}
        # A rectangular sparse operator stays on the native partial SVD path.
        A = vcat(Matrix(Diagonal(T.(1:96))), zeros(T, 8, 96))
        S = partial_svd(FaerLinearOperator(sparse(A)), 2; tol, max_restarts=20)
        @test S.algorithm == :partial
        @test S.nconverged == 2
        @test S.S ≈ R[96, 95] rtol = 100tol
        @test maximum(S.residuals) < 100tol * 96
    end
end
@testset "Full-space general eigen and parameter validation" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        op = FaerLinearOperator(Diagonal(T[1, 2, 3, 4]))
        F = partial_eigen(op, 2)
        @test F.values ≈ T[4, 3]
        @test F.nconverged == 2
        @test maximum(F.residuals) < 1e-4
    end
    op = FaerLinearOperator(I, 96)
    @test_throws ArgumentError partial_eigen(op, 2; min_dim=90, max_dim=64)
end
@testset "Native full-subspace branch failure and triangle convention" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        R = typeof(real(zero(T)))
        # Finite inputs whose composition overflows materialize a non-finite operator;
        # faer's full-subspace branch reports that as zero converged pairs.
        big = FaerLinearOperator(fill(T(floatmax(R)), 4, 4))
        bad = big * big
        close(big)
        for f in (partial_self_adjoint_eigen, partial_svd)
            F = f(bad, 2)
            @test F.algorithm == :dense_fallback
            @test F.requested == 2
            @test F.nconverged == 0
            @test isempty(F.residuals)
            @test isempty(f === partial_svd ? F.S : F.values)
        end
        close(bad)
        # The full-subspace self-adjoint branch reads the lower triangle of the operator.
        H = T[4 1 0 0; 1 3 1 0; 0 1 -2 1; 0 0 1 1]
        T <: Complex && (H[2, 1] += T(0.5im); H[1, 2]=conj(H[2, 1]))
        lower = FaerLinearOperator(tril(H))
        F = partial_self_adjoint_eigen(lower, 4)
        @test F.algorithm == :dense_fallback
        @test F.nconverged == 4
        @test F.values ≈ sort(eigvals(Hermitian(H, :L)); by=abs, rev=true) rtol = 100eps(R)
        close(lower)
    end
end
