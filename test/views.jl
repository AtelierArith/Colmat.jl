using Test, LinearAlgebra, Colmat

@testset "Native borrowed views" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        data = reshape(T.(1:30), 5, 6)
        T <: Complex && (data .+= im .* reverse(data; dims=1))
        a = FaerMatrix(data)
        for (rr, cc) in ((1:5, 1:6), (5:-2:1, 6:-2:1), (2:2:4, 1:3:6), (1:0, 1:6))
            v = view(a, rr, cc)
            @test v isa FaerMatrixView
            @test Matrix(v) == data[rr, cc]
            @test Matrix(copy(v)) == data[rr, cc]
            @test Matrix(v + v) == 2data[rr, cc]
            @test Matrix(v - v) == zero(data[rr, cc])
            @test Matrix(v' * v) ≈ data[rr, cc]'data[rr, cc]
            @test Matrix(transpose(v) * v) ≈ transpose(data[rr, cc]) * data[rr, cc]
            @test Matrix(view(v, :, :)) == data[rr, cc]
            close(v)
            @test !isopen(v)
            @test_throws ArgumentError copy(v)
        end
        v = view(a, 5:-2:1, 6:-2:1)
        w = view(v, 2:3, 2:-1:1)
        @test Matrix(w) == data[3:-2:1, 4:2:6]
        close(v)
        @test isopen(w)
        w[1, 2] = T(77)
        data[3, 6] = T(77)
        @test Matrix(a) == data
        fill!(w, T(8))
        data[3:-2:1, 4:2:6] .= T(8)
        @test Matrix(a) == data
        for v in (view(a, 2, :), view(a, :, 3), view(a, 5:-2:1, 3))
            expected = if v.dropped == 1
                vec(data[2, :])
            elseif v.rows == 1:5
                data[:, 3]
            else
                data[5:-2:1, 3]
            end
            @test v isa FaerVectorView
            @test Vector(v) == expected
            @test Vector(copy(v)) == expected
            @test Vector(view(v, length(v):-1:1)) == reverse(expected)
        end
        # Native matrix/vector products preserve Rust ownership.
        v = FaerVector(T.(1:6))
        vv = view(v, 6:-1:1)
        @test a * v isa FaerVector
        @test Vector(a * v) ≈ data * T.(1:6)
        @test Vector(a * vv) ≈ data * reverse(T.(1:6))
        @test Vector(view(a, 1:2:5, :) * v) ≈ data[1:2:5, :] * T.(1:6)
        @test Vector(a' * FaerVector(T.(1:5))) ≈ data' * T.(1:5)
        # Snapshot overlapping assignments before a mutable native borrow.
        b = FaerMatrix(data)
        copyto!(view(b, 2:5, :), view(b, 1:4, :))
        expected = copy(data)
        expected[2:5, :] = data[1:4, :]
        @test Matrix(b) == expected
        copyto!(view(b, 1, :), T.(11:16))
        expected[1, :] = T.(11:16)
        @test Matrix(b) == expected
        copyto!(view(b, :, 2), T.(21:25))
        expected[:, 2] = T.(21:25)
        @test Matrix(b) == expected
        # Noninjective ranges use defined ordered writes, never an aliased MatMut.
        repeat = StepRangeLen(2, 0, 3)
        z = view(b, repeat, 1:2)
        @test Matrix(z) == expected[fill(2, 3), 1:2]
        copyto!(z, reshape(T.(1:6), 3, 2))
        expected[2, 1] = T(3)
        expected[2, 2] = T(6)
        @test Matrix(b) == expected
        # Capacity changes retain views, shape changes invalidate permanently.
        w = view(b, 1:2, 1:2)
        reserve!(b, 30, 30)
        @test Matrix(w) == expected[1:2, 1:2]
        resize!(b, 5, 6)
        @test isopen(w)
        resize!(b, 6, 6)
        @test !isopen(w)
        @test_throws InvalidStateException copy(w)
        resize!(b, 5, 6)
        @test !isopen(w)
        w = view(b, :, :)
        close(b)
        @test !isopen(w)
        @test_throws ArgumentError fill!(w, 0)
        # Views retain owners over GC, including stored diagonal buffers.
        held = let p = FaerMatrix(data)
            view(p, 1:2, 1:2)
        end
        GC.gc()
        @test Matrix(held) == data[1:2, 1:2]
        d = FaerDiagonal(T[1, 2, 3])
        dv = diagonalview(d)
        fill!(dv, T(4))
        @test Matrix(d) == Matrix(Diagonal(fill(T(4), 3)))
        copyto!(view(dv, 3:-1:1), T[5, 6, 7])
        @test Vector(d) == T[7, 6, 5]
        row = FaerRow(T[1, 2, 3])
        rv = view(row, 1, 3:-1:1)
        fill!(rv, 9)
        @test Vector(row) == T[9, 9, 9]
        x = FaerVector(T[1, 2, 3])
        xv = view(x, :)
        reserve!(x, 20)
        @test Vector(xv) == T[1, 2, 3]
        resize!(x, 4)
        @test !isopen(xv)
    end
end

@testset "Owned vector products and solves" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        data = T[4 1; 1 3]
        a = FaerMatrix(data)
        v = FaerVector(T[1, 2])
        f = lu(a)
        for b in (v, view(v, 2:-1:1))
            x = a \ b
            y = f \ b
            @test x isa FaerVector
            @test y isa FaerVector
            @test Vector(x) ≈ data \ Vector(b)
            @test Vector(y) ≈ data \ Vector(b)
        end
        dest = copy(v)
        ldiv!(dest, f, v)
        @test Vector(dest) ≈ data \ Vector(v)
        ldiv!(f, dest)
        @test Vector(dest) ≈ data \ (data \ Vector(v))
        dest = copy(v)
        mul!(dest, a, dest)
        @test Vector(dest) ≈ data * Vector(v)
        before = Vector(dest)
        mul!(dest, a, view(dest, :), 2, 3)
        @test Vector(dest) ≈ 2data * before + 3before
        copyto!(v, reverse(Vector(v)))
        @test Vector(v) == T[2, 1]
        @test Vector(FaerVector(FaerMatrix(v))) == Vector(v)
    end
    a = FaerMatrix(Float32[4 1; 1 3])
    b = FaerVector(ComplexF64[1 + im, 2 - im])
    @test Vector(a * b) ≈ Matrix(a) * Vector(b)
    @test Vector(a \ b) ≈ Matrix(a) \ Vector(b)
    @test eltype(a \ b) === ComplexF64
    @test Vector(lu(a) \ b) ≈ Matrix(a) \ Vector(b)
    @test_throws DimensionMismatch mul!(FaerVector([1.0]), a, FaerVector([1.0, 2]))
    @test isempty(detect_ambiguities(Colmat, Base, LinearAlgebra; recursive=true))
end

@testset "View copies and generic array integration" begin
    a = FaerMatrix(reshape(collect(1.0:12), 3, 4))
    v = view(a, 3:-1:1, 2:4)
    d = deepcopy(v)
    @test d isa FaerMatrixView
    @test Matrix(d) == Matrix(v)
    v[1, 1] = 99
    @test d[1, 1] != 99
    s = view(v, 2, 2)
    @test size(s) == ()
    @test s[] == a[2, 3]
    s[] = 21
    @test a[2, 3] == 21
    @test_throws BoundsError view(v, 4, 1)
    col = view(a, :, 2)
    snapshot = Vector(col)
    col .= 2 .* col
    @test Vector(col) == 2snapshot
    child = view(col, 3:-1:1)
    close(col)
    @test Vector(child) == reverse(2snapshot)
end
