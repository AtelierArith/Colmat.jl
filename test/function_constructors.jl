using Colmat, Test, LinearAlgebra

struct ConstructorConstant{T}
    value::T
end
(f::ConstructorConstant)(args...) = f.value

@testset "Native arrays from Julia functions" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        dims in ((0, 0), (0, 5), (4, 0), (3, 5))

        calls = Tuple{Int,Int}[]
        f(i, j) = (push!(calls, (i, j)); T <: Complex ? T(i, j) : T(i + 2j))
        a = FaerMatrix{T}(f, dims...)
        try
            @test eltype(a) === T
            @test size(a) == dims
            @test calls == [(i, j) for j in 1:dims[2] for i in 1:dims[1]]
            @test Matrix(a) ==
                [T <: Complex ? T(i, j) : T(i + 2j) for i in 1:dims[1], j in 1:dims[2]]
        finally
            close(a)
        end
    end
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        C in (FaerVector, FaerRow),
        n in (0, 5)

        calls = Int[]
        f(i) = (push!(calls, i); T <: Complex ? T(i, 2i) : T(2i))
        a = C{T}(f, n)
        try
            @test eltype(a) === T
            @test calls == collect(1:n)
            @test Vector(a) == [T <: Complex ? T(i, 2i) : T(2i) for i in 1:n]
        finally
            close(a)
        end
    end
    for C in (FaerMatrix, FaerVector, FaerRow)
        dims = C === FaerMatrix ? (2, 3) : (4,)
        a = C(ConstructorConstant(3), dims...)
        @test eltype(a) === Float64
        @test all(==(3), a)
        close(a)
        calls = Ref(0)
        bad(args...) = (calls[] += 1; error("constructor initializer failed"))
        @test_throws ErrorException C(bad, dims...)
        @test calls[] == 1
        invalid = C === FaerMatrix ? (-1, 3) : (-1,)
        calls[] = 0
        @test_throws ArgumentError C(bad, invalid...)
        @test calls[] == 0
    end
    # Existing undef and identity constructors remain more specific methods.
    a = FaerMatrix{Float32}(undef, 2, 3)
    @test size(a) == (2, 3)
    close(a)
    a = FaerMatrix{ComplexF64}(I, 2, 2)
    @test Matrix(a) == Matrix{ComplexF64}(I, 2, 2)
    close(a)
end

@testset "Function constructors preserve native scalar values" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        C in (FaerMatrix, FaerVector, FaerRow),
        value in (T(NaN), T(Inf), T(-0.0))

        dims = C === FaerMatrix ? (2, 3) : (4,)
        a = C{T}(ConstructorConstant(value), dims...)
        try
            actual = C === FaerMatrix ? Matrix(a) : Vector(a)
            @test isequal(actual, fill(value, dims...))
        finally
            close(a)
        end
    end
    for C in (FaerMatrix, FaerVector, FaerRow)
        dims = C === FaerMatrix ? (2, 3) : (4,)
        @test_throws Exception C{Float32}(ConstructorConstant("invalid scalar"), dims...)
    end
    @test_throws OverflowError FaerMatrix((i, j) -> error("must not run"), typemax(Int), 2)
end
