using Test, LinearAlgebra, Random, SparseArrays, Colmat

@testset "Native CSC/CSR ownership and conversion" begin
    rng = MersenneTwister(608)
    for T in (Float32, Float64, ComplexF32, ComplexF64), fmt in (:csc, :csr)
        for (m, n) in ((5, 4), (0, 4), (5, 0), (0, 0), (1, 1))
            source = sprand(rng, T, m, n, 0.4)
            a = FaerSparseMatrix(source; format=fmt)
            @test a isa AbstractSparseMatrix{T,Int}
            @test issparse(a)
            @test size(a) == (m, n)
            @test nnz(a) == nnz(source)
            @test Matrix(a) == Matrix(source)
            @test SparseMatrixCSC(a) == source
            @test sparse(a) == source
            @test findnz(a) == findnz(source)
            @test SparseMatrixCSC{ComplexF64,Int}(a) ==
                SparseMatrixCSC{ComplexF64,Int}(source)
            @test Matrix(FaerMatrix(a)) == Matrix(source)
            @test sum(a) ≈ sum(source)
            for p in (1, 2, Inf, 3)
                @test norm(a, p) ≈ norm(source, p)
            end
            for g in (copy(a), deepcopy(a), copy(a; format=fmt === :csc ? :csr : :csc))
                @test Matrix(g) == Matrix(source)
                @test isopen(g)
            end
            @test size(zero(a)) == size(a) && nnz(zero(a)) == 0
            @test nnz(similar(a, T, (m, n))) == 0
            @test Matrix(conj(a)) == conj(Matrix(source))
            @test Matrix(copy(transpose(a))) == transpose(Matrix(source))
            @test Matrix(copy(adjoint(a))) == Matrix(source)'
            @test copy(transpose(a)).format === (fmt === :csc ? :csr : :csc)
            if fmt === :csc
                @test rowvals(a) == rowvals(source)
                @test SparseArrays.getcolptr(a) == SparseArrays.getcolptr(source)
                for j in 1:n
                    @test nzrange(a, j) == nzrange(source, j)
                end
            else
                @test_throws ArgumentError rowvals(a)
                @test_throws ArgumentError SparseArrays.getcolptr(a)
            end
            for j in 1:n, i in 1:m
                @test a[i, j] == source[i, j]
            end
            data = compressed_storage(a)
            @test data.format === fmt
            @test length(data.pointers) == (fmt === :csc ? n : m) + 1
            @test length(data.indices) == length(data.values) == nnz(source)
            @test data.values == Vector(nonzeros(a))
            nvals = nonzeros(a)
            saved = copy(a)
            close(a)
            @test !isopen(a) && !isopen(nvals)
            @test_throws ArgumentError Vector(nvals)
            @test_throws ArgumentError sparse(a)
            @test_throws ArgumentError sum(a)
            @test_throws ArgumentError nnz(a)
            @test Matrix(saved) == Matrix(source)
        end
        i = [3, 1, 3, 2, 1]
        j = [2, 1, 2, 4, 3]
        v = T[1, 2, -1, 3, 0]
        source = sparse(i, j, v, 3, 4)
        a = FaerSparseMatrix(i, j, v, 3, 4; format=fmt)
        @test SparseMatrixCSC(a) == source
        @test nnz(a) == nnz(source)
        @test any(iszero, Vector(nonzeros(a)))
        @test_throws DimensionMismatch FaerSparseMatrix(i, j, v[1:2], 3, 4; format=fmt)
        @test_throws ArgumentError FaerSparseMatrix([0], [1], T[1], 3, 4; format=fmt)
        @test_throws ArgumentError FaerSparseMatrix([1], [5], T[1], 3, 4; format=fmt)
        @test_throws ArgumentError FaerSparseMatrix(Int[], Int[], T[], -1, 4; format=fmt)
        @test_throws ArgumentError FaerSparseMatrix(source; format=:bad)
    end
end

@testset "Sparse value views, structural mutation and aliasing" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), fmt in (:csc, :csr)
        a = FaerSparseMatrix(sparse([1, 2], [1, 3], T[2, 4], 3, 4); format=fmt)
        v = nonzeros(a)
        @test Vector(v) == T[2, 4]
        v[1] = T(5)
        @test a[1, 1] == 5
        a[2, 3] = T(6)
        @test Vector(v) == T[5, 6]
        @test isopen(v)
        a[3, 4] = zero(T)
        @test isopen(v) && nnz(a) == 2
        a[3, 4] = T(7)
        @test !isopen(v) && nnz(a) == 3
        @test_throws InvalidStateException v[1]
        v = nonzeros(a)
        a[3, 4] = zero(T)
        @test isopen(v) && nnz(a) == 3
        dropzeros!(a)
        @test !isopen(v) && nnz(a) == 2
        v = nonzeros(a)
        dropzeros!(a)
        @test isopen(v)
        original = Vector(v)
        w = deepcopy(v)
        @test typeof(w) === typeof(v) && Vector(w) == original
        fill!(w, T(9))
        @test Vector(v) == original
        fill!(v, T(8))
        @test Vector(v) == fill(T(8), 2)
        fill!(a, 0)
        @test nnz(a) == 2 && all(iszero, Vector(v))
        @test_throws ArgumentError fill!(a, 1)
        copyto!(a, FaerSparseMatrix(sparse([1, 2], [1, 3], T[2, 4], 3, 4); format=fmt))
        @test isopen(v) && Vector(v) == T[2, 4]
        copyto!(a, FaerSparseMatrix(sparse([1], [2], T[7], 3, 4)))
        @test !isopen(v) && nnz(a) == 1
        @test a[1, 2] == 7
        @test copyto!(a, a) === a
        b = FaerSparseMatrix(sparse([1, 2, 3], [1, 2, 3], T[1, 2, 3], 3, 3); format=fmt)
        v, w = nonzeros(b), nonzeros(b)
        copyto!(v, view(w, 3:-1:1))
        @test Vector(v) == T[3, 2, 1]
        data = compressed_storage(b)
        data.values[1] = 99
        data.indices[1] = 3
        @test b[1, 1] == 3
        if fmt === :csc
            rows = rowvals(b)
            rows[1] = 3
            ptr = SparseArrays.getcolptr(b)
            ptr[2] = 1
            @test b[1, 1] == 3
        end
        held = let c = FaerSparseMatrix(sparse([1], [1], T[4], 1, 1); format=fmt)
            nonzeros(c)
        end
        GC.gc()
        @test held[1] == 4
        close(held)
        @test !isopen(held)
        @test_throws ArgumentError held[1]
        @test_throws BoundsError b[0, 1]
        @test_throws BoundsError b[4, 1] = 1
    end
end

@testset "Native sparse arithmetic and products" begin
    rng = MersenneTwister(710)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        afmt in (:csc, :csr),
        bfmt in (:csc, :csr)

        tol = typeof(real(zero(T))) === Float32 ? 5e-4 : 3e-11
        for (m, k, n) in ((5, 4, 3), (0, 4, 3), (5, 0, 3), (5, 4, 0))
            a, b = sprand(rng, T, m, k, 0.4), sprand(rng, T, k, n, 0.4)
            aa, bb = FaerSparseMatrix(a; format=afmt), FaerSparseMatrix(b; format=bfmt)
            c = aa * bb
            @test c isa FaerSparseMatrix{T}
            @test c.format === afmt
            @test SparseMatrixCSC(c) ≈ a * b rtol = tol atol = tol
            @test Matrix(aa * FaerMatrix(Matrix(b))) ≈ Matrix(a * b) rtol = tol atol = tol
            @test Matrix(FaerMatrix(Matrix(a)) * bb) ≈ Matrix(a * b) rtol = tol atol = tol
            @test Matrix(aa * Matrix(b)) ≈ Matrix(a * b) rtol = tol atol = tol
            @test Matrix(Matrix(a) * bb) ≈ Matrix(a * b) rtol = tol atol = tol
            @test Matrix(aa * copy(adjoint(aa))) ≈ Matrix(a * a') rtol = tol atol = tol
            @test Matrix(adjoint(aa) * aa) ≈ Matrix(a' * a) rtol = tol atol = tol
            @test Matrix(transpose(aa) * aa) ≈ Matrix(transpose(a) * a) rtol = tol atol =
                tol
            x = randn(rng, T, k)
            v = FaerVector(x)
            @test Vector(aa * v) ≈ a * x rtol = tol atol = tol
            @test aa * x ≈ a * x rtol = tol atol = tol
            @test Vector(aa * view(v, :)) ≈ a * x rtol = tol atol = tol
            dest = zero(c)
            mul!(dest, aa, bb)
            @test SparseMatrixCSC(dest) ≈ a * b rtol = tol atol = tol
            mul!(dest, aa, bb, T(2), T(3))
            @test SparseMatrixCSC(dest) ≈ 5(a * b) rtol = tol atol = tol
            dense = FaerMatrix{T}(undef, m, n)
            mul!(dense, aa, bb)
            @test Matrix(dense) ≈ Matrix(a * b) rtol = tol atol = tol
        end
        source = sprand(rng, T, 4, 4, 0.4)
        other = sprand(rng, T, 4, 4, 0.4)
        a, b = FaerSparseMatrix(source; format=afmt), FaerSparseMatrix(other; format=bfmt)
        @test SparseMatrixCSC(a + b) ≈ source + other
        @test SparseMatrixCSC(a - b) ≈ source - other
        @test SparseMatrixCSC(a + other) ≈ source + other
        @test SparseMatrixCSC(source - b) ≈ source - other
        @test SparseMatrixCSC(source * b) ≈ source * other
        @test SparseMatrixCSC(a * other) ≈ source * other
        @test SparseMatrixCSC(2a) ≈ 2source
        @test SparseMatrixCSC(a / 2) ≈ source / 2
        @test SparseMatrixCSC(-a) ≈ -source
        orig = copy(source)
        mul!(a, a, b)
        @test SparseMatrixCSC(a) ≈ orig * other rtol = tol atol = tol
        x = FaerVector(randn(rng, T, 4))
        before = Vector(x)
        mul!(x, b, x, T(2), T(3))
        @test Vector(x) ≈ 2other * before + 3before rtol = tol atol = tol
    end
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        S in (Float32, Float64, ComplexF32, ComplexF64)

        a, b = sprand(rng, T, 3, 3, 0.7), sprand(rng, S, 3, 3, 0.7)
        aa, bb = FaerSparseMatrix(a; format=:csr), FaerSparseMatrix(b)
        @test eltype(aa + bb) === promote_type(T, S)
        @test SparseMatrixCSC(aa + bb) ≈ a + b
        @test SparseMatrixCSC(aa * bb) ≈ a * b
        @test Matrix(aa * FaerMatrix(Matrix(b))) ≈ Matrix(a * b)
        @test Matrix(FaerMatrix(Matrix(a)) * bb) ≈ Matrix(a * b)
    end
end

@testset "Large sparse shapes retain sparse storage" begin
    a = FaerSparseMatrix([1, 2], [1, 10^9], [2.0, 3.0], 2, 10^9; format=:csr)
    @test size(a) == (2, 10^9) && nnz(a) == 2
    @test a[1, 1] == 2 && a[2, 10 ^ 9] == 3 && a[1, 10 ^ 9] == 0
    @test sum(a) == 5 && norm(a) ≈ sqrt(13)
    @test sum(copy(a)) == 5
    @test sum(a + a) == 10
    @test sum(im * a) == 5im
    @test size(copy(transpose(a))) == (10^9, 2)
    @test compressed_storage(a).pointers == [1, 2, 3]
    v = nonzeros(a)
    @test Vector(v) == [2, 3]
    @test isempty(
        detect_ambiguities(Colmat, Base, LinearAlgebra, SparseArrays; recursive=true)
    )
end

@testset "SparseArrays dispatch intersections" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        data = T[3 1; 2 4]
        a = FaerMatrix(data)
        b = sparsevec([1], T[2], 2)
        for C in
            (LowerTriangular, UpperTriangular, UnitLowerTriangular, UnitUpperTriangular)
            @test C(a) \ b ≈ C(data) \ Vector(b)
        end
        dest = FaerMatrix{T}(undef, 4, 4)
        v = view(dest, 2:3, 2:3)
        copyto!(v, sparse(data))
        @test Matrix(v) == data
        d = SparseArrays.CHOLMOD.Dense(data)
        copyto!(v, d)
        @test Matrix(v) == data
        before = Matrix(v)
        @test_throws DimensionMismatch copyto!(view(dest, :, :), d)
        @test Matrix(v) == before
    end
    @test isempty(
        detect_ambiguities(Colmat, Base, LinearAlgebra, SparseArrays; recursive=true)
    )
end
