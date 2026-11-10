using Colmat, Test, LinearAlgebra

struct DenseConversionInput{T,F} <: AbstractMatrix{T}
    dims::Tuple{Int,Int}
    convert_dense::F
end
Base.size(a::DenseConversionInput) = a.dims
Base.Matrix{T}(a::DenseConversionInput) where {T} = a.convert_dense(T)

@testset "Dense construction validates converted storage before native loading" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        for construct in (FaerMatrix, FaerMatrix{T})
            for result in (zeros(UInt8, 2, 3), zeros(T, 6), transpose(zeros(T, 3, 2)))
                calls = Ref(0)
                convert_dense = S -> begin
                    calls[] += 1
                    result
                end
                source = DenseConversionInput{T,typeof(convert_dense)}(
                    (2, 3), convert_dense
                )
                @test_throws ArgumentError construct(source)
                @test calls[] == 1
            end
            for dims in ((3, 2), (1, 6), (0, 3))
                convert_dense = S -> zeros(S, dims)
                source = DenseConversionInput{T,typeof(convert_dense)}(
                    (2, 3), convert_dense
                )
                @test_throws DimensionMismatch construct(source)
            end
            calls = Ref(0)
            data = reshape(T.(1:6), 2, 3)
            convert_dense = S -> begin
                calls[] += 1
                Matrix{S}(data)
            end
            source = DenseConversionInput{T,typeof(convert_dense)}((2, 3), convert_dense)
            native = construct(source)
            @test calls[] == 1
            @test native isa FaerMatrix{T}
            @test Matrix(native) == data
            fill!(data, zero(T))
            @test Matrix(native) == reshape(T.(1:6), 2, 3)
            close(native)
        end
        for dims in ((0, 0), (0, 3), (3, 0), (2, 3))
            data = reshape(T.(1:prod(dims)), dims)
            for source in
                (data, view(data, size(data, 1):-1:1, :), transpose(data), adjoint(data))
                native = FaerMatrix{T}(source)
                @test size(native) == size(source)
                @test Matrix(native) == Matrix(source)
                close(native)
            end
        end
        source = FaerMatrix(reshape(T.(1:6), 2, 3))
        native = FaerMatrix{T}(source)
        close(source)
        @test Matrix(native) == reshape(T.(1:6), 2, 3)
        close(native)
    end
    calls = Ref(0)
    convert_dense = T -> begin
        calls[] += 1
        @test T === Float64
        ones(T, 2, 3)
    end
    source = DenseConversionInput{Int,typeof(convert_dense)}((2, 3), convert_dense)
    native = FaerMatrix(source)
    @test calls[] == 1
    @test Matrix(native) == ones(2, 3)
    close(native)
end
