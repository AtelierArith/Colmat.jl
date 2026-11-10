using Colmat, Test, LinearAlgebra, SparseArrays, Random

@testset "Native AMD COLAMD and independent ownership" begin
    rng = MersenneTwister(745)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        format in (:csc, :csr),
        n in (0, 1, 4, 17, 96)

        a = sprand(rng, T, n, n, 0.12)
        source = FaerSparseMatrix(a; format)
        pattern = FaerSparsity(source)
        for aggressive in (true, false), threshold in (-1.0, 0.0, 0.5, 10.0)
            stats = amd(pattern; dense=threshold, aggressive, statistics=true)
            p = stats.permutation
            @test p isa FaerPermutation
            @test length(p) == n
            @test isperm(Vector(p))
            @test Vector(invperm(p))[Vector(p)] == 1:n
            @test Vector(p) == Vector(amd(source; dense=threshold, aggressive))
            @test Vector(p) == Vector(amd(a; dense=threshold, aggressive))
            @test all(
                x -> isfinite(x) && x >= 0,
                (stats.n_div, stats.n_mult_subs_ldl, stats.n_mult_subs_lu),
            )
            q = colamd(pattern; dense_row=threshold, dense_col=threshold, aggressive)
            @test q isa FaerPermutation
            @test isperm(Vector(q))
            @test Vector(q) == Vector(
                colamd(source; dense_row=threshold, dense_col=threshold, aggressive)
            )
            @test Vector(q) ==
                Vector(colamd(a; dense_row=threshold, dense_col=threshold, aggressive))
        end
        @test sparse(source) == a
        p = amd(source)
        q = colamd(pattern)
        close(source)
        close(pattern)
        @test isperm(Vector(p)) && isperm(Vector(q))
        @test_throws ArgumentError amd(source)
        @test_throws ArgumentError colamd(pattern)
    end
end
@testset "COLAMD rectangular and empty patterns" begin
    rng = MersenneTwister(184)
    for (m, n) in ((0, 0), (0, 7), (7, 0), (3, 8), (8, 3), (51, 37), (37, 51)),
        format in (:csc, :csr)

        a = sprand(rng, m, n, 0.2)
        s = FaerSparseMatrix(a; format)
        p = colamd(s)
        @test isperm(Vector(p))
        @test length(p) == n
        @test Vector(p) == Vector(colamd(a))
        @test Matrix(s[:, p]) == Matrix(a[:, Vector(p)])
        m == n || @test_throws DimensionMismatch amd(s)
    end
    a = FaerSparseMatrix(sparse([1.0 2; 0 3]))
    for x in (Inf, -Inf, NaN)
        @test_throws ArgumentError amd(a; dense=x)
        @test_throws ArgumentError colamd(a; dense_row=x)
        @test_throws ArgumentError colamd(a; dense_col=x)
    end
    @test isempty(
        detect_ambiguities(Colmat, Base, LinearAlgebra, SparseArrays; recursive=true)
    )
end
@testset "Native sparse permutation gather and mutation" begin
    rng = MersenneTwister(411)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        format in (:csc, :csr),
        (m, n) in ((0, 0), (1, 1), (6, 9), (9, 6))

        a = sprand(rng, T, m, n, 0.3)
        s = FaerSparseMatrix(a; format)
        p = FaerPermutation(randperm(rng, m))
        q = FaerPermutation(randperm(rng, n))
        pattern = FaerSparsity(s)
        for owner in (s, pattern)
            expected = owner === s ? Matrix(a) : Matrix(sparse(pattern))
            for (result, ref) in (
                (owner[p, :], expected[Vector(p), :]),
                (owner[:, q], expected[:, Vector(q)]),
                (owner[p, q], expected[Vector(p), Vector(q)]),
            )
                @test typeof(result) === typeof(owner)
                @test result.format === format
                @test Matrix(result) == ref
                @test nnz(result) == nnz(owner)
            end
            @test Matrix(permute_rows(owner, Vector(p))) == expected[Vector(p), :]
            @test Matrix(permute_cols(owner, Vector(q))) == expected[:, Vector(q)]
            @test_throws DimensionMismatch permute_rows(owner, FaerPermutation(1:(m + 1)))
            @test_throws DimensionMismatch permute_cols(owner, FaerPermutation(1:(n + 1)))
        end
        values = nonzeros(s)
        original = copy(s)
        expected = a[Vector(p), Vector(q)]
        @test permute_rows!(s, p) === s
        @test permute_cols!(s, q) === s
        @test sparse(s) == expected
        @test sparse(original) == a
        @test nnz(s) == nnz(a)
        @test_throws ArgumentError permute_rows(s, FaerPermutation([2, 1, 1]))
        close(p)
        @test_throws ArgumentError original[p, :]
    end
    # Stored zeros participate in permutation and ordering; values do not.
    s = FaerSparseMatrix(sparse([1, 2, 3], [1, 2, 3], [0.0, NaN, Inf], 3, 3))
    @test Vector(amd(s)) == Vector(amd(FaerSparsity(s)))
    @test Vector(colamd(s)) == Vector(colamd(FaerSparsity(s)))
    @test nnz(s[FaerPermutation([3, 1, 2]), :]) == 3
    @test isequal(Matrix(s[:, FaerPermutation([3, 1, 2])]), Matrix(s)[:, [3, 1, 2]])
end
@testset "Ordering honors selected Hermitian triangle" begin
    rng = MersenneTwister(971)
    for format in (:csc, :csr), uplo in (:L, :U), n in (3, 23)
        a = sprand(rng, n, n, 0.15)
        s = FaerSparseMatrix(a; format)
        h = Hermitian(s, uplo)
        reference = sparse(Hermitian(a, uplo))
        @test Vector(amd(h)) == Vector(amd(reference))
        @test Vector(colamd(h)) == Vector(colamd(reference))
        @test Vector(amd(Symmetric(s, uplo))) == Vector(amd(reference))
    end
end
@testset "Ordering reduces fill on an arrow graph" begin
    n = 80
    a = spdiagm(0 => fill(2.0, n))
    a[1, :] .= 0.1
    a[:, 1] .= 0.1
    a[1, 1] = n
    p = Vector(amd(a))
    before = nnz(sparse(cholesky(Hermitian(a); perm=1:n).L))
    after = nnz(sparse(cholesky(Hermitian(a[p, p]); perm=1:n).L))
    @test after < before
    @test after == 2n - 1
end
@testset "Ordering and sparse permutation scale with stored data" begin
    n = 50_000
    s = FaerSparseMatrix(sparse(1:n, 1:n, ones(n), n, n); format=:csr)
    for order in (amd, colamd)
        p = order(s)
        @test isperm(Vector(p))
        @test nnz(s[p, p]) == n
    end
    # Row permutation must preserve CSR without allocating 10^9 column pointers.
    wide = FaerSparseMatrix([1, 2], [1, 10^9], [2.0, 3.0], 2, 10^9; format=:csr)
    p = FaerPermutation([2, 1])
    @test size(wide[p, :]) == (2, 10^9)
    @test wide[p, :][2, 1] == 2
    @test wide[p, :][1, 10 ^ 9] == 3
    pattern = FaerSparsity(wide)
    @test size(pattern[p, :]) == (2, 10^9)
    @test pattern[p, :][2, 1]
end

@testset "Ordering transpose patterns and value-view invalidation" begin
    rng = MersenneTwister(23)
    for format in (:csc, :csr)
        a = sprand(rng, ComplexF64, 17, 17, 0.15)
        s = FaerSparseMatrix(a; format)
        @test Vector(amd(s')) == Vector(amd(sparse(a')))
        @test Vector(colamd(s')) == Vector(colamd(sparse(a')))
        @test Vector(amd(transpose(s))) == Vector(amd(sparse(transpose(a))))
        @test Vector(colamd(transpose(s))) == Vector(colamd(sparse(transpose(a))))
        diagonal = FaerSparseMatrix(sparse([1.0 0; 0 2]); format)
        values = nonzeros(diagonal)
        p = FaerPermutation([2, 1])
        permute_rows!(diagonal, p)
        @test_throws InvalidStateException values[1]
        @test Matrix(diagonal) == [0.0 2; 1 0]
        full = FaerSparseMatrix(sparse([1.0 2; 3 4]); format)
        values = nonzeros(full)
        permute_rows!(full, p)
        @test Vector(values) == compressed_storage(full).values
        @test Matrix(full) == [3.0 4; 1 2]
    end
end
