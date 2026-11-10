using Test, Colmat, Statistics, LinearAlgebra, SparseArrays, Random

function _same_stat(x, y, tol)
    return isnan(x) && isnan(y) || isequal(x, y) || isapprox(x, y; rtol=tol, atol=tol)
end
function _same_stats(x, y, tol)
    size(x) == size(y) || return false
    return if x isa Number
        _same_stat(x, y, tol)
    else
        all(_same_stat(a, b, tol) for (a, b) in zip(x, y))
    end
end
@testset "Native Statistics" begin
    rng = MersenneTwister(17)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        R = typeof(real(zero(T)))
        tol = 256eps(R)
        for shape in ((3, 4), (7, 1027), (1027, 7), (1, 3), (3, 1), (0, 3), (3, 0), (0, 0))
            A = randn(rng, T, shape...)
            native = FaerMatrix(A)
            for dims in (:, 1, 2, (1, 2), (), 3, (2, 3))
                @test _same_stats(mean(native; dims), mean(A; dims), tol)
                @test eltype(mean(native; dims)) === T
                for corrected in (true, false)
                    @test _same_stats(
                        var(native; dims, corrected), var(A; dims, corrected), tol
                    )
                    @test eltype(var(native; dims, corrected)) === R
                    m = mean(A; dims)
                    @test _same_stats(
                        varm(native, m; dims, corrected),
                        (
                            if dims isa Colon
                                varm(A, m; corrected)
                            else
                                varm(A, m; dims, corrected)
                            end
                        ),
                        tol,
                    )
                    @test _same_stats(
                        var(native; mean=m, dims, corrected),
                        var(A; mean=m, dims, corrected),
                        tol,
                    )
                end
            end
        end
        for values in (T[1, 2, 4], T[2], T[], T[Inf], T[1, Inf, 2])
            native = FaerVector(zeros(T, length(values)))
            for i in eachindex(values)
                native[i] = values[i]
            end
            for dims in (:, 1, 2, ())
                @test _same_stats(mean(native; dims), mean(values; dims), tol)
                for corrected in (true, false)
                    @test _same_stats(
                        var(native; dims, corrected), var(values; dims, corrected), tol
                    )
                end
            end
        end
        native = FaerMatrix(zeros(T, 3, 3))
        A = T[1 2 3; 4 5 6; 7 8 9]
        A[1, 1] = T(NaN)
        A[:, 2] .= T(NaN)
        for j in 1:3, i in 1:3
            native[i, j] = A[i, j]
        end
        for dims in (1, 2)
            for (i, slice) in enumerate(eachslice(A; dims=dims == 1 ? 2 : 1))
                filtered = filter(x -> !isnan(x), slice)
                @test _same_stat(mean(native; dims, nan=:ignore)[i], mean(filtered), tol)
                for corrected in (true, false)
                    @test _same_stat(
                        var(native; dims, nan=:ignore, corrected)[i],
                        var(filtered; corrected),
                        tol,
                    )
                end
            end
            @test _same_stats(mean(native; dims), mean(A; dims), tol)
            @test _same_stats(var(native; dims), var(A; dims), tol)
        end
        @test _same_stat(
            mean(native; nan=:ignore), mean(filter(x -> !isnan(x), vec(A))), tol
        )
        @test_throws ArgumentError mean(native; nan=:invalid)
        @test_throws ArgumentError var(native; dims=0)
        @test_throws DimensionMismatch varm(native, zeros(T, 3); dims=1)
        @test_throws DimensionMismatch varm(native, zeros(T, 1, 1))
        singleton = FaerVector(T[2])
        @test _same_stat(varm(singleton, T(1)), varm(T[2], T(1)), tol)
        @test varm(singleton, T(1); corrected=false) == one(R)
        parent = FaerMatrix(randn(rng, T, 5, 6))
        region = view(parent, 2:4, 2:5)
        @test _same_stats(mean(region; dims=2), mean(Matrix(region); dims=2), tol)
        @test _same_stats(var(region; dims=1), var(Matrix(region); dims=1), tol)
        close(parent)
        @test_throws ArgumentError mean(region)
        @test_throws ArgumentError var(parent)
    end
    A = FaerMatrix(Float32[1 2; 3 4])
    M = mean(Matrix(A); dims=1)
    @test varm(A, reshape(M, 1, 2, 1); dims=1) ≈ varm(Matrix(A), M; dims=1)
    @test varm(A, 1.0 + 2im) ≈ varm(ComplexF64.(Matrix(A)), 1.0 + 2im)
    @test eltype(varm(A, 1.0 + 2im)) === Float64
    # Supplied means follow Statistics.varm promotion: full reductions use the
    # promoted scalar's real type; dims reductions keep the data's real type.
    data = Float32[1 2; 3 5; 4 9]
    A = FaerMatrix(data)
    for m in (2, 2.0, 2.0f0, 2 + 1im, 2.0f0 + 1im, 1//3, true, 2.0 + 1im)
        expected = varm(data, m)
        @test typeof(varm(A, m)) === typeof(expected)
        @test varm(A, m) ≈ expected
        @test typeof(var(A; mean=m)) === typeof(var(data; mean=m))
        @test var(A; mean=m, corrected=false) ≈ var(data; mean=m, corrected=false)
    end
    for m in ([2 3], [2.0 3.0], [2 + 1im 3], Float16[2 3], [1//3 1], ComplexF64[2 3 + im])
        expected = varm(data, m; dims=1)
        @test eltype(varm(A, m; dims=1)) === eltype(expected)
        @test varm(A, m; dims=1) ≈ expected
    end
    @test typeof(varm(FaerMatrix(zeros(Float32, 0, 2)), 1.0)) === Float32
    @test isnan(varm(FaerMatrix(zeros(Float32, 0, 2)), 1.0))
    @test_throws ArgumentError varm(A, big(2.0))
    @test_throws ArgumentError varm(A, Any[2 3]; dims=1)
    # A mean borrowed from the same owner is copied before the native call.
    @test varm(A, view(A, 1:1, :); dims=1) ≈ varm(data, data[1:1, :]; dims=1)
    @test Matrix(A) == data
    # Strided, reversed and vector views are reduced in place by the native kernels.
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        R = typeof(real(zero(T)))
        tol = 256eps(R)
        B = randn(MersenneTwister(5), T, 7, 9)
        B[2, 3] = T(NaN)
        B[5, 8] = T(NaN)
        # Constructors reject NaN; write them through native element access.
        parent = FaerMatrix(zeros(T, size(B)))
        for j in axes(B, 2), i in axes(B, 1)
            parent[i, j] = B[i, j]
        end
        for (rows, cols) in
            ((7:-1:1, 9:-2:1), (2:3:7, 1:9), (6:-1:2, 8:-1:3), (1:7, 9:-1:1))
            region = view(parent, rows, cols)
            dense = B[rows, cols]
            for dims in (:, 1, 2, (1, 2), ()), corrected in (true, false)
                @test _same_stats(mean(region; dims), mean(dense; dims), tol)
                @test _same_stats(
                    var(region; dims, corrected), var(dense; dims, corrected), tol
                )
                for nan in (:propagate, :ignore)
                    m = mean(region; dims, nan)
                    @test _same_stats(
                        var(region; dims, corrected, nan, mean=m),
                        var(region; dims, corrected, nan),
                        tol,
                    )
                end
            end
            for dims in (1, 2)
                means = mean(region; dims, nan=:ignore)
                variances = var(region; dims, nan=:ignore)
                for (i, slice) in enumerate(eachslice(dense; dims=dims == 1 ? 2 : 1))
                    filtered = filter(x -> !isnan(x), slice)
                    @test _same_stat(means[i], mean(filtered), tol)
                    @test _same_stat(variances[i], var(filtered), tol)
                end
            end
            @test _same_stat(
                var(region; nan=:ignore), var(filter(x -> !isnan(x), vec(dense))), tol
            )
        end
        column = view(parent, 7:-2:1, 4)
        row = view(parent, 3, 9:-1:1)
        for (v, dense) in ((column, B[7:-2:1, 4]), (row, B[3, 9:-1:1]))
            @test _same_stats(mean(v; dims=1), mean(dense; dims=1), tol)
            @test _same_stat(var(v), var(dense), tol)
        end
        @test isequal(Matrix(parent), B)
    end
    # Native inputs are not snapshotted on the Julia heap.
    large = FaerMatrix(randn(256, 256))
    largeview = view(large, 256:-1:1, 1:2:255)
    for f in (
        () -> mean(large; dims=1),
        () -> var(large; dims=2),
        () -> var(large; nan=:ignore),
        () -> mean(largeview; dims=2),
        () -> var(largeview),
    )
        f()
        @test @allocated(f()) < 256 * 256 * 8 ÷ 8
    end
    row = FaerRow([1.0, 2, 4])
    @test mean(row) ≈ mean([1.0, 2, 4])
    @test var(row) ≈ var([1.0, 2, 4])
    closedmean = FaerMatrix(zeros(1, 0))
    close(closedmean)
    @test_throws ArgumentError varm(FaerMatrix(zeros(3, 0)), closedmean; dims=1)
    complex_nan = FaerVector(ComplexF64[1 + im, 2 + im])
    complex_nan[1] = ComplexF64(1, NaN)
    @test mean(complex_nan; nan=:ignore) ≈ 2 + im
    @test isnan(var(complex_nan; nan=:ignore))
    @test isempty(
        filter(
            pair -> any(m -> m.module === Colmat, pair),
            Test.detect_ambiguities(
                Colmat, Base, LinearAlgebra, SparseArrays, Statistics; recursive=true
            ),
        ),
    )
end
