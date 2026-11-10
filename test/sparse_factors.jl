using Colmat, Test, LinearAlgebra, SparseArrays, Random

@testset "Reusable native sparse LU QR and Cholesky" begin
    rng = MersenneTwister(431)
    for T in (Float32, Float64, ComplexF32, ComplexF64), format in (:csc, :csr)
        for kind in (:lu, :qr, :cholesky), uplo in (:L, :U)
            raw = randn(rng, T, 6, 6)
            a = kind === :cholesky ? raw * raw' + 6I : raw + 6I
            source = FaerSparseMatrix(sparse(a); format)
            pattern = FaerSparsity(source)
            plan = FaerSparseSymbolic(pattern; kind, uplo)
            f = factorize(plan, source)
            @test f isa FaerSparseFactorization{T}
            @test issuccess(f)
            @test size(f) == size(a)
            @test size(f, 3) == 1
            @test isopen(f) && isopen(plan)
            rhs = randn(rng, T, 6, 3)
            for b in (rhs, FaerMatrix(rhs), view(FaerMatrix(rhs), :, :))
                @test Matrix(f \ b) ≈ a \ rhs
            end
            v = FaerVector(rhs[:, 1])
            @test f \ rhs[:, 1] ≈ a \ rhs[:, 1]
            @test f \ sparse(rhs[:, 1]) ≈ a \ rhs[:, 1]
            @test Vector(f \ v) ≈ a \ rhs[:, 1]
            @test f \ v isa FaerVector
            out = FaerMatrix(zeros(T, 6, 3))
            @test ldiv!(out, f, FaerMatrix(rhs)) === out
            @test Matrix(out) ≈ a \ rhs
            original = FaerMatrix(rhs)
            @test ldiv!(f, original) === original
            @test Matrix(original) ≈ a \ rhs
            if kind !== :qr
                @test Matrix(f' \ rhs) ≈ a' \ rhs
                @test Matrix(transpose(f) \ rhs) ≈ transpose(a) \ rhs
            else
                @test Matrix(f' \ rhs) ≈ a' \ rhs
            end
            # Symbolic analysis is shared only by faer's immutable Arc; numeric
            # owners and source snapshots stay independent.
            changed = FaerSparseMatrix(sparse(2a); format=format === :csr ? :csc : :csr)
            g = factorize(plan, changed)
            @test Matrix(g \ rhs) ≈ (2a) \ rhs
            source[1, 1] += 1
            close(source)
            close(pattern)
            close(plan)
            @test Matrix(f \ rhs) ≈ a \ rhs
            mixed = randn(rng, ComplexF64, 6, 2)
            @test Matrix(f \ mixed) ≈ ComplexF64.(a) \ mixed rtol = sqrt(eps(real(T)))
            @test eltype(f \ mixed) === promote_type(T, ComplexF64)
            copied = copy(f)
            deep = deepcopy(f)
            close(f)
            @test !isopen(f)
            @test_throws ArgumentError f \ rhs
            @test Matrix(copied \ rhs) ≈ a \ rhs
            @test Matrix(deep \ rhs) ≈ a \ rhs
            @test Matrix(copied \ mixed) ≈ ComplexF64.(a) \ mixed rtol = sqrt(eps(real(T)))
        end
        a = randn(rng, T, 8, 4)
        s = FaerSparseMatrix(sparse(a); format)
        f = qr(s)
        b = randn(rng, T, 8, 2)
        @test size(f \ b) == (4, 2)
        @test Matrix(f \ b) ≈ a \ b
        @test Matrix(s \ b) ≈ a \ b
        @test norm(a' * (a * Matrix(f \ b) - b)) < 100eps(real(T)) * norm(a) * norm(b)
        dest = FaerMatrix(zeros(T, 4, 2))
        @test ldiv!(dest, f, FaerMatrix(b)) === dest
        @test Matrix(dest) ≈ a \ b
        @test_throws DimensionMismatch ldiv!(f, FaerMatrix(b))
        @test_throws DimensionMismatch f \ zeros(T, 7, 2)
        @test_throws DimensionMismatch ldiv!(zeros(T, 3, 2), f, b)
    end
end

@testset "Sparse analysis validation and failures" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), format in (:csc, :csr)
        a = FaerSparseMatrix(sparse(T[4 1; 1 3]); format)
        for f in (
            lu(a),
            qr(a),
            cholesky(a),
            cholesky(Hermitian(a, :U)),
            cholesky(Symmetric(FaerSparseMatrix(real.(sparse(a))), :L)),
        )
            @test Matrix(f \ Matrix{T}(I, 2, 2)) ≈ inv(T[4 1; 1 3])
        end
        p = FaerSparseSymbolic(a)
        @test_throws ArgumentError factorize(
            p, FaerSparseMatrix(sparse(T[4 0; 1 3]); format)
        )
        @test_throws DimensionMismatch factorize(
            p, FaerSparseMatrix(spzeros(T, 3, 3); format)
        )
        q = copy(p)
        close(p)
        @test_throws ArgumentError factorize(p, a)
        @test factorize(q, a) \ T[1, 2] ≈ T[4 1; 1 3] \ T[1, 2]
        close(q)
        @test_throws ArgumentError FaerSparseSymbolic(a; kind=:bad)
        @test_throws ArgumentError FaerSparseSymbolic(a; uplo=:bad)
        W = qr(FaerSparseMatrix(spzeros(T, 2, 3); format))
        @test Matrix(W) == zeros(T, 2, 3)
        @test Matrix(W \ ones(T, 2, 1)) == zeros(T, 3, 1)
        close(W)
        @test_throws DimensionMismatch lu(FaerSparseMatrix(spzeros(T, 3, 2); format))
        for kind in (:lu, :cholesky)
            bad = FaerSparseMatrix(sparse(T[1 2; 2 1]); format)
            kind === :lu && (bad = FaerSparseMatrix(sparse(T[1 2; 2 4]); format))
            plan = FaerSparseSymbolic(bad; kind)
            @test_throws (kind === :lu ? SingularException : PosDefException) factorize(
                plan, bad
            )
            f = factorize(plan, bad; check=false)
            @test !issuccess(f)
            @test_throws (kind === :lu ? SingularException : PosDefException) f \ ones(T, 2)
            @test !issuccess(copy(f))
        end
        bad = FaerSparseMatrix(sparse(T[1 2; 0 3]); format)
        @test_throws ArgumentError cholesky(bad)
        bad[1, 1] = T(Inf)
        @test_throws ArgumentError lu(bad)
    end
    @test isempty(
        detect_ambiguities(Colmat, Base, LinearAlgebra, SparseArrays; recursive=true)
    )
end

@testset "Sparse Cholesky triangle selection and factor transforms" begin
    rng = MersenneTwister(531)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        format in (:csc, :csr),
        uplo in (:L, :U)

        a = randn(rng, T, 5, 5)
        h = a * a' + 5I
        stored = copy(h)
        for j in 1:5, i in 1:5
            (uplo === :L ? i < j : i > j) && (stored[i, j] = T(NaN))
        end
        source = FaerSparseMatrix(sparse(stored); format)
        f = cholesky(Hermitian(source, uplo))
        b = randn(rng, T, 5, 2)
        @test Matrix(f \ b) ≈ h \ b
        @test Matrix(f' \ b) ≈ h' \ b
        @test Matrix(transpose(f) \ b) ≈ transpose(h) \ b
        complex = randn(rng, ComplexF64, 5, 2)
        @test Matrix(f \ complex) ≈ ComplexF64.(h) \ complex rtol = sqrt(eps(real(T)))
        @test Matrix(f' \ complex) ≈ ComplexF64.(h)' \ complex rtol = sqrt(eps(real(T)))
        for g in (f, f', transpose(f))
            mat = copy(b)
            @test ldiv!(g, mat) === mat
            @test mat ≈ Matrix(g \ b)
            output = zeros(T, 5, 2)
            @test ldiv!(output, g, b) === output
            @test output ≈ Matrix(g \ b)
            v = copy(b[:, 1])
            @test ldiv!(g, v) === v
            @test v ≈ g \ b[:, 1]
            out = zeros(T, 5)
            @test ldiv!(out, g, b[:, 1]) === out
            @test out ≈ g \ b[:, 1]
        end
        @test_throws DimensionMismatch f \ ones(T, 4)
        @test_throws DimensionMismatch ldiv!(zeros(T, 4), f, ones(T, 5))
    end
end
@testset "Sparse factors preserve sparse scaling and empty RHS" begin
    n = 50_000
    for kind in (:lu, :qr, :cholesky), format in (:csc, :csr)
        a = FaerSparseMatrix(sparse(1:n, 1:n, fill(2.0, n), n, n); format)
        @test ishermitian(a) && issymmetric(a)
        plan = FaerSparseSymbolic(a; kind)
        f = factorize(plan, a)
        close(a)
        close(plan)
        @test f \ ones(n) ≈ fill(0.5, n)
        @test size(f \ zeros(n, 0)) == (n, 0)
    end
    for kind in (:lu, :qr, :cholesky)
        a = FaerSparseMatrix(spzeros(0, 0))
        f = factorize(FaerSparseSymbolic(a; kind), a)
        @test size(f \ zeros(0, 2)) == (0, 2)
        @test isempty(f \ Float64[])
    end
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        a = T[2 1; 0 3]
        source = FaerSparseMatrix(sparse(a))
        @test !ishermitian(source) && !issymmetric(source)
        f = lu(source)
        b = Complex{real(T)}[1 + im, 2 - im]
        @test f \ b ≈ a \ b
        @test f' \ b ≈ a' \ b
        @test transpose(f) \ b ≈ transpose(a) \ b
        @test issymmetric(FaerSparseMatrix(sparse(T[1 2; 2 3])))
    end
end

@testset "Sparse numeric refactorization reuses analysis" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), kind in (:lu, :qr, :cholesky)
        a = FaerSparseMatrix(sparse(T[4 1; 1 3]))
        f = factorize(FaerSparseSymbolic(a; kind), a)
        refactor! = if kind === :lu
            lu!
        elseif kind === :qr
            qr!
        else
            cholesky!
        end
        b = FaerSparseMatrix(sparse(T[8 2; 2 6]); format=:csr)
        @test refactor!(f, b) === f
        @test f \ T[1, 2] ≈ T[8 2; 2 6] \ T[1, 2]
        @test_throws ArgumentError refactor!(f, FaerSparseMatrix(sparse(T[8 0; 2 6])))
        @test f \ T[1, 2] ≈ T[8 2; 2 6] \ T[1, 2]
        if kind !== :qr
            bad = FaerSparseMatrix(sparse(T[1 2; 2 (kind === :lu ? 4 : 1)]))
            @test_throws (kind === :lu ? SingularException : PosDefException) refactor!(
                f, bad
            )
            @test f \ T[1, 2] ≈ T[8 2; 2 6] \ T[1, 2]
            @test refactor!(f, bad; check=false) === f
            @test !issuccess(f)
            @test refactor!(f, b) === f
            @test issuccess(f)
        end
        close(f)
        @test_throws ArgumentError refactor!(f, b)
    end
end
