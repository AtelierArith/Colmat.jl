using Colmat, Test, LinearAlgebra

@testset "Direct constant native matrix construction" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        dims in ((0, 0), (0, 5), (4, 0), (2, 3)),
        value in (T(2), T(NaN), T(Inf), T(-0.0))

        for tuple in (false, true)
            a=tuple ? FaerMatrix{T}(dims; fill = value) :
              FaerMatrix{T}(dims...; fill = value)
            try
                @test eltype(a)===T
                @test size(a)==dims
                @test isequal(Matrix(a), fill(value, dims...))
            finally
                close(a)
            end
        end
    end
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        f in (zeros, ones),
        tuple in (false, true)

        a=tuple ? f(FaerMatrix{T}, (2, 3)) : f(FaerMatrix{T}, 2, 3)
        @test Matrix(a)==fill(f===zeros ? zero(T) : one(T), 2, 3)
        @test eltype(a)===T
        close(a)
    end
    for T in (ComplexF32, ComplexF64)
        a=FaerMatrix{T}(2, 3; fill = 1-2im)
        @test Matrix(a)==fill(T(1-2im), 2, 3)
        close(a)
    end
end

@testset "Direct constant native collections" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        C in (FaerVector, FaerRow, FaerDiagonal),
        n in (0, 1, 5),
        value in (T(2), T(NaN), T(Inf), T(-0.0))

        a=C{T}(n; fill = value)
        try
            @test eltype(a)===T
            @test isequal(Vector(a), fill(value, n))
            if C===FaerDiagonal
                @test isequal(Matrix(a), Matrix(Diagonal(fill(value, n))))
            elseif C===FaerRow
                @test size(a)==(1, n)
            end
        finally
            close(a)
        end
    end
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        C in (FaerVector, FaerRow, FaerDiagonal),
        f in (zeros, ones),
        tuple in (false, true)

        a=tuple ? f(C{T}, (4,)) : f(C{T}, 4)
        @test Vector(a)==fill(f===zeros ? zero(T) : one(T), 4)
        @test eltype(a)===T
        close(a)
    end
end

@testset "Constant constructor defaults and validation" begin
    for f in (zeros, ones)
        a=f(FaerMatrix, 2, 3)
        @test eltype(a)===Float64
        close(a)
        a=f(FaerMatrix, (2, 3))
        @test eltype(a)===Float64
        close(a)
    end
    for C in (FaerVector, FaerRow, FaerDiagonal), f in (zeros, ones), tuple in (false, true)
        a=tuple ? f(C, (3,)) : f(C, 3)
        @test eltype(a)===Float64
        close(a)
    end
    a=FaerMatrix(2, 3; fill = 4)
    @test Matrix(a)==fill(4.0, 2, 3)
    close(a)
    a=FaerMatrix((2, 3); fill = 4)
    @test Matrix(a)==fill(4.0, 2, 3)
    close(a)
    for C in (FaerVector, FaerRow, FaerDiagonal)
        a=C(3; fill = 4)
        @test Vector(a)==fill(4.0, 3)
        close(a)
        @test_throws ArgumentError C(-1)
        @test_throws InexactError C(typemax(UInt))
        @test_throws Exception C{Float32}(3; fill = "invalid")
    end
    @test_throws ArgumentError FaerMatrix(-1, 3)
    @test_throws ArgumentError FaerMatrix(2, -1)
    @test_throws OverflowError FaerMatrix(typemax(Int), 2)
    @test_throws InexactError FaerMatrix(typemax(UInt), 0)
    @test_throws Exception FaerMatrix{Float32}(2, 3; fill = "invalid")
end

@testset "Complex constant component preservation" begin
    for T in (ComplexF32, ComplexF64),
        C in (FaerMatrix, FaerVector, FaerRow, FaerDiagonal),
        value in (T(1-2im), T(-0.0, -0.0))

        dims=C===FaerMatrix ? (2, 3) : (3,)
        a=C{T}(dims...; fill = value)
        try
            actual=C===FaerMatrix ? Matrix(a) : Vector(a)
            @test isequal(actual, fill(value, dims...))
            @test eltype(a)===T
        finally
            close(a)
        end
    end
end
