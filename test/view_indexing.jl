using Test, LinearAlgebra, SparseArrays, Colmat

function _indexing_data(T, m, n)
    data = reshape(T.(1:(m * n)), m, n)
    T <: Complex && (data .+= im .* reverse(data; dims=1))
    return data
end

@testset "Native gather A[I, J] with integer vectors and masks" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        data = _indexing_data(T, 6, 5)
        a = FaerMatrix(data)
        rows = [5, 1, 1, 6, 2]
        cols = Int32[4, 2, 4]
        rmask = BitVector([1, 0, 1, 1, 0, 1])
        cmask = [false, true, true, false, true]
        for (I, J) in (
            (rows, :),
            (:, cols),
            (rows, cols),
            (rmask, :),
            (rmask, cmask),
            (rows, 2:2:4),
            (5:-2:1, cols),
            (Int[], cols),
            (rows, Int[]),
            (3, cols),
            (rows, 2),
            (1:3, 2:4),
            (:, :),
        )
            expected = data[I, J]
            got = a[I, J]
            @test got == expected
            # Result containers follow Base's `similar` rule.
            @test got isa (ndims(expected) == 2 ? FaerMatrix{T} : Vector{T})
        end
        # Views are read through their own strided/reversed descriptor.
        v = view(a, 6:-1:2, 1:2:5)
        dv = data[6:-1:2, 1:2:5]
        @test v[[1, 5, 1], [3, 1]] == dv[[1, 5, 1], [3, 1]]
        @test v[BitVector([1, 0, 0, 1, 1]), :] == dv[BitVector([1, 0, 0, 1, 1]), :]
        @test v[[2, 4], 3] == dv[[2, 4], 3]
        x = FaerVector(data[:, 1])
        @test x[[6, 1, 1]] == data[[6, 1, 1], 1]
        @test x[rmask] == data[rmask, 1]
        rv = view(a, 2, :)
        @test rv[[5, 1, 3]] == data[2, [5, 1, 3]]
        r = FaerRow(data[1, :])
        @test r[1, [5, 2]] == data[1, [5, 2]]
        # Linear indexing reads the owner once in bulk; containers follow Base.
        base(x, I...) = invoke(getindex, Tuple{AbstractArray,Vararg{Any}}, x, I...)
        lmask = real.(data) .> 10
        for (x, I) in (
            (a, (lmask,)),
            (a, ([1, 30, 7],)),
            (a, (:,)),
            (a, ([1 2; 3 4],)),
            (r, (2:3,)),
            (v, ([1, 5],)),
            (a, (rows, cols)),
            (v, (rmask[1:5], 2)),
            (x, (rmask,)),
        )
            expected = base(x, I...)
            got = x[I...]
            @test got == expected
            @test typeof(got) == typeof(expected)
        end
        @test_throws BoundsError a[[0, 1], :]
        @test_throws BoundsError a[[1, 7], :]
        @test_throws BoundsError a[:, [6]]
        @test_throws BoundsError a[trues(5), :]
        @test_throws BoundsError v[[6], :]
    end
    a = FaerMatrix(rand(200, 200))
    I = collect(1:4:200)
    a[I, :]
    # One native call writes the result owner: no per-element boxing (which
    # would allocate about 160 KB for these 10_000 elements). The fixed part
    # measured 8224 B on macOS x86_64, above the earlier 8 KB bound.
    @test @allocated(a[I, :]) < 16_000
    @test @allocated(a[I, I]) < 16_000
end

@testset "Native scatter A[I, J] = B" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        data = _indexing_data(T, 6, 5)
        a = FaerMatrix(data)
        B = T.(reshape(100:111, 4, 3))
        I = [6, 2, 3, 1]
        J = [5, 1, 3]
        a[I, J] = B
        data[I, J] = B
        @test Matrix(a) == data
        # Native and borrowed sources, vectors into a dropped dimension.
        a[[1, 4], :] = FaerMatrix(data[2:3, :])
        data[[1, 4], :] = data[2:3, :]
        @test Matrix(a) == data
        a[[2, 5], 4] = T[-1, -2]
        data[[2, 5], 4] = T[-1, -2]
        @test Matrix(a) == data
        a[3, [1, 5]] = FaerVector(T[7, 8])
        data[3, [1, 5]] = T[7, 8]
        @test Matrix(a) == data
        a[BitVector([1, 0, 0, 0, 0, 1]), [true, false, false, false, true]] = faerview(
            T[1 2; 3 4]
        )
        data[BitVector([1, 0, 0, 0, 0, 1]), [true, false, false, false, true]] = T[1 2; 3 4]
        @test Matrix(a) == data
        # Repeated destination indices: the last write in column-major order wins.
        a[[2, 2], [1, 1]] = T[1 3; 2 4]
        data[[2, 2], [1, 1]] = T[1 3; 2 4]
        @test a[2, 1] == T(4)
        @test Matrix(a) == data
        # Sources aliasing the destination are read before any write.
        a[[2, 1, 3], :] = view(a, 1:3, :)
        data[[2, 1, 3], :] = data[1:3, :]
        @test Matrix(a) == data
        a[:, [5, 4, 3, 2, 1]] = a
        data[:, [5, 4, 3, 2, 1]] = copy(data)
        @test Matrix(a) == data
        # Destinations through views and vectors.
        v = view(a, 6:-1:2, :)
        v[[1, 3], [2, 4]] = T[9 8; 7 6]
        data[[6, 4], [2, 4]] = T[9 8; 7 6]
        @test Matrix(a) == data
        xd = data[:, 2]
        x = FaerVector(xd)
        x[[3, 1]] = T[11, 12]
        xd[[3, 1]] = T[11, 12]
        @test Vector(x) == xd
        # Conversion and shape failures leave the destination unchanged.
        before = Matrix(a)
        @test_throws DimensionMismatch a[[1, 2], [1, 2]] = ones(T, 3, 2)
        @test_throws BoundsError a[[1, 7], [1]] = ones(T, 2, 1)
        if T <: Real
            @test_throws InexactError a[[1, 2], [1]] = ComplexF64[1 + im, 2]
        end
        @test Matrix(a) == before
    end
    a = FaerMatrix(rand(200, 200))
    I = collect(1:4:200)
    B = rand(length(I), 200)
    a[I, :] = B
    @test @allocated(a[I, :]=B) < 8_000
end

@testset "Integer-vector views are SubArrays with bulk copies" begin
    for T in (Float32, ComplexF64)
        data = _indexing_data(T, 6, 5)
        a = FaerMatrix(data)
        s = view(a, [5, 1, 2], :)
        @test s isa SubArray
        @test copy(s) == data[[5, 1, 2], :]
        @test Matrix(s) == data[[5, 1, 2], :]
        @test Array(view(a, [2, 4], 3)) == data[[2, 4], 3]
        B = T.(reshape(1:15, 3, 5))
        copyto!(s, B)
        data[[5, 1, 2], :] = B
        @test Matrix(a) == data
        s .= 2 .* B
        data[[5, 1, 2], :] = 2 .* B
        @test Matrix(a) == data
        fill!(view(a, [6, 3], [true, false, true, false, false]), T(-5))
        data[[6, 3], [1, 3]] .= T(-5)
        @test Matrix(a) == data
        view(a, [1, 2], :) .= 0
        data[[1, 2], :] .= 0
        @test Matrix(a) == data
    end
    a = FaerMatrix(rand(200, 200))
    s = view(a, collect(1:4:200), :)
    copy(s)
    # One bulk native copy: the fixed part measured 8224 B on macOS x86_64;
    # per-element copies would allocate about 160 KB.
    @test @allocated(copy(s)) < 16_000
end

@testset "Native assignment from another native owner" begin
    for T in (Float32, ComplexF64)
        data = _indexing_data(T, 6, 5)
        a = FaerMatrix(data)
        src = FaerMatrix(T.(reshape(1:10, 2, 5)))
        copyto!(view(a, 2:3, :), src)
        data[2:3, :] = Matrix(src)
        @test Matrix(a) == data
        # Same owner, disjoint and overlapping ranges.
        copyto!(view(a, 1:2, :), view(a, 5:6, :))
        data[1:2, :] = data[5:6, :]
        @test Matrix(a) == data
        copyto!(view(a, 2:4, :), view(a, 1:3, :))
        data[2:4, :] = data[1:3, :]
        @test Matrix(a) == data
    end
    a = FaerMatrix(rand(200, 200))
    b = FaerMatrix(rand(50, 200))
    dest = view(a, 1:50, :)
    copyto!(dest, b)
    # No snapshot of the 80 KB source.
    @test @allocated(copyto!(dest, b)) < 4_000
    other = view(a, 101:150, :)
    copyto!(dest, other)
    @test @allocated(copyto!(dest, other)) < 4_000
end

@testset "split_at_row / split_at_col / split_at" begin
    for T in (Float64, ComplexF32)
        data = _indexing_data(T, 6, 5)
        a = FaerMatrix(data)
        for k in 0:6
            top, bottom = split_at_row(a, k)
            @test top isa FaerMatrixView && bottom isa FaerMatrixView
            @test Matrix(top) == data[1:k, :]
            @test Matrix(bottom) == data[(k + 1):end, :]
        end
        for k in 0:5
            left, right = split_at_col(a, k)
            @test Matrix(left) == data[:, 1:k]
            @test Matrix(right) == data[:, (k + 1):end]
        end
        tl, tr, bl, br = split_at(a, 2, 3)
        @test Matrix(tl) == data[1:2, 1:3]
        @test Matrix(tr) == data[1:2, 4:5]
        @test Matrix(bl) == data[3:6, 1:3]
        @test Matrix(br) == data[3:6, 4:5]
        # Splits of views compose and write through to the owner.
        v = view(a, 6:-1:1, 1:2:5)
        p, q = split_at_row(v, 4)
        @test Matrix(q) == data[2:-1:1, 1:2:5]
        fill!(q, T(0))
        data[2:-1:1, 1:2:5] .= 0
        @test Matrix(a) == data
        x = FaerVector(data[:, 1])
        h, t = split_at_row(x, 2)
        @test h isa FaerVectorView
        @test Vector(h) == data[1:2, 1] && Vector(t) == data[3:6, 1]
        h2, t2 = split_at_row(t, 4)
        @test Vector(h2) == data[3:6, 1] && isempty(t2)
        r = FaerRow(data[1, :])
        l, rr = split_at_col(r, 1)
        @test Matrix(l) == data[1:1, 1:1] && Matrix(rr) == data[1:1, 2:5]
        l, rr = split_at_row(view(a, 3, :), 3)
        @test Vector(l) == data[3, 1:3] && Vector(rr) == data[3, 4:5]
        @test_throws BoundsError split_at_row(a, 7)
        @test_throws BoundsError split_at_row(a, -1)
        @test_throws BoundsError split_at_col(a, 6)
        @test_throws BoundsError split_at(a, 1, 6)
        @test_throws BoundsError split_at_row(x, 7)
        close(a)
        @test_throws ArgumentError split_at_row(a, 1)
    end
end

@testset "Native strides" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        a = FaerMatrix(_indexing_data(T, 6, 5))
        s = strides(a)
        @test s[1] == 1
        @test s[2] >= size(a, 1)
        @test stride(a, 2) == s[2]
        reserve!(a, 40, 5)
        cs = stride(a, 2)
        @test cs >= 40
        @test cs == capacity(a)[1]
        v = view(a, 5:-2:1, 2:2:4)
        @test strides(v) == (-2, 2cs)
        @test strides(view(a, 2, :)) == (cs,)
        @test strides(view(a, :, 3)) == (1,)
        x = FaerVector(T[1, 2, 3])
        @test strides(x) == (1,)
        @test stride(x, 1) == 1
        @test strides(view(x, 3:-1:1)) == (-1,)
        r = FaerRow(T[1, 2, 3])
        @test stride(r, 2) == 1
        # Native storage is not pointer-backed for Julia code.
        @test_throws MethodError Base.elsize(a)
        @test_throws ErrorException pointer(a)
        resize!(a, 2, 2)
        @test_throws InvalidStateException strides(v)
        close(a)
        @test_throws ArgumentError strides(a)
    end
end

@testset "Indexing methods are unambiguous" begin
    @test isempty(
        Test.detect_ambiguities(Colmat, Base, LinearAlgebra, SparseArrays; recursive=true)
    )
end
