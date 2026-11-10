using Test, Colmat, LinearAlgebra, SparseArrays, Random

@testset "Native self-adjoint tridiagonalization" begin
    rng = MersenneTwister(91)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        R = typeof(real(zero(T)))
        tol = 2048eps(R)
        for n in (0, 1, 2, 3, 5, 17, 65)
            raw = randn(rng, T, n, n)
            A = Matrix(Hermitian(raw))
            for bs in unique([0, 1, min(2, max(1, n - 1))])
                f = tridiagonalize(A; blocksize=bs)
                @test f isa FaerTridiagonalization
                @test f.H isa SymTridiagonal{R}
                @test eltype(f.Q) === T
                @test size(f) == (n, n)
                @test Matrix(f) ≈ A rtol = tol atol = tol
                Q = Matrix(f.Q)
                @test Q' * Q ≈ Matrix{T}(I, n, n) atol = tol
                E = eigen(f)
                @test E.values ≈ eigvals(Hermitian(A)) rtol = tol atol = tol
                X = Matrix(E.vectors)
                @test norm(A * X - X * Diagonal(E.values)) <= tol * max(norm(A), one(R))
                @test eigvals(f) ≈ E.values rtol = tol atol = tol
                copied = copy(f)
                @test copied.Q !== f.Q
                @test copied.H.dv !== f.H.dv
                @test Matrix(copied) ≈ A rtol = tol atol = tol
                close(f)
                @test !isopen(f)
                @test isopen(copied)
                @test_throws ArgumentError eigen(f)
            end
            native = FaerMatrix(A)
            F = hessenberg(Hermitian(native))
            @test F isa FaerTridiagonalization
            @test Matrix(F) ≈ A rtol = tol atol = tol
            @test Matrix(native) == A
            changed = tridiagonalize!(native)
            @test Matrix(native) == Matrix(changed.H)
            @test Matrix(changed) ≈ A rtol = tol atol = tol
        end
        for n in (0, 1, 2, 7, 33, 130)
            d = T.(randn(rng, R, n))
            e = randn(rng, T, max(0, n - 1))
            A = Matrix(Tridiagonal(e, d, conj.(e)))
            reference = eigvals(Hermitian(A))
            for threshold in (16, 256)
                E = tridiagonal_eigen(d, e; recursion_threshold=threshold)
                @test E.values ≈ reference rtol = tol atol = tol
                @test issorted(E.values)
                @test eltype(E.values) === R
                @test eltype(E.vectors) === T
                X = Matrix(E.vectors)
                @test norm(A * X - X * Diagonal(E.values)) <= tol * max(norm(A), one(R))
                @test X' * X ≈ Matrix{T}(I, n, n) atol = tol
                @test tridiagonal_eigvals(d, e; recursion_threshold=threshold) ≈ reference rtol =
                    tol atol = tol
            end
        end
        for scale in (sqrt(floatmin(R)), sqrt(floatmax(R)) / R(100))
            base = T[4 1 2; 1 3 1; 2 1 5]
            f = tridiagonalize(scale .* base)
            @test Matrix(f) ./ scale ≈ base rtol = tol atol = tol
            d = T[1, 2, 3] .* scale
            e = T[0.2, 0.4] .* scale
            @test tridiagonal_eigvals(d, e) ./ scale ≈
                eigvals(SymTridiagonal(R[1, 2, 3], R[0.2, 0.4])) rtol = tol atol = tol
        end
        @test_throws ArgumentError tridiagonalize(T[1 2; 3 4])
        @test_throws DimensionMismatch tridiagonalize(zeros(T, 2, 3))
        @test_throws ArgumentError tridiagonalize(Matrix{T}(I, 2, 2); blocksize=-1)
        @test_throws DimensionMismatch tridiagonal_eigen(T[1, 2], T[])
        @test_throws ArgumentError tridiagonal_eigen(T[NaN], T[])
        @test_throws ArgumentError tridiagonal_eigen(T[1], T[]; recursion_threshold=1)
        source = FaerMatrix(Matrix{T}(I, 3, 3))
        source[1, 1] = T(NaN)
        saved = Matrix(source)
        @test_throws ArgumentError tridiagonalize!(source)
        @test isequal(Matrix(source), saved)
    end
    for T in (Float32, Float64)
        d = FaerVector(T[1, 2, 3])
        e = FaerVector(T[0.2, 0.4])
        S = SymTridiagonal(d, e)
        @test eigen(S).values ≈ eigvals(SymTridiagonal(Vector(d), Vector(e)))
        @test eigvals(S) ≈ eigen(S).values
    end
    A = FaerMatrix([4.0 1 2; 1 3 1; 2 1 5])
    F = hessenberg!(Symmetric(A))
    @test Matrix(A) == Matrix(F.H)
    @test Matrix(F) ≈ [4.0 1 2; 1 3 1; 2 1 5]
    upper = FaerMatrix([4.0 1 2; 99 3 1; 99 99 5])
    @test Matrix(hessenberg(Symmetric(upper, :U))) ≈ [4.0 1 2; 1 3 1; 2 1 5]
    parent = FaerMatrix(Matrix{Float64}(I, 5, 5))
    region = view(parent, 2:4, 2:4)
    @test Matrix(hessenberg(Hermitian(region))) ≈ Matrix{Float64}(I, 3, 3)
    original = Matrix(region)
    result = hessenberg!(Hermitian(region))
    @test Matrix(region) == Matrix(result.H)
    @test Matrix(result) ≈ original
    close(parent)
    @test_throws ArgumentError hessenberg(Hermitian(region))
    @test_throws ArgumentError tridiagonal_eigen(ComplexF64[1 + im], ComplexF64[])
    @test tridiagonal_eigvals([1.0, 3, 2], zeros(2); sortby=x -> -x) == [3.0, 2, 1]
    # Eigenvalues-only storage stays linear in the band dimension.
    n = 50_000
    d = collect(1.0:n)
    @test tridiagonal_eigvals(d, zeros(n - 1)) ≈ d
    ignored = FaerMatrix([4.0 1 2; 1 3 1; 2 1 5])
    ignored[3, 1] = NaN
    @test Matrix(hessenberg(Hermitian(ignored, :U))) ≈ [4.0 1 2; 1 3 1; 2 1 5]
    closed = FaerVector(Float64[])
    close(closed)
    @test_throws ArgumentError tridiagonal_eigvals(closed, Float64[])
    @test isempty(
        Test.detect_ambiguities(Colmat, Base, LinearAlgebra, SparseArrays; recursive=true)
    )
end
