using Colmat, Test

@testset "Empty owned arrays with native capacity" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        (m, n) in ((0, 0), (0, 7), (9, 0), (3, 5))

        a = with_capacity(FaerMatrix{T}, m, n)
        try
            @test size(a) == (0, 0)
            @test eltype(a) === T
            reserved = capacity(a)
            @test all(reserved .>= (m, n))
            resize!(a, m, n; fill=T(2))
            @test Matrix(a) == fill(T(2), m, n)
            @test capacity(a) == reserved
        finally
            close(a)
        end
    end
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        C in (FaerVector, FaerRow),
        n in (0, 7)

        a = with_capacity(C{T}, n)
        try
            @test size(a) == (C === FaerVector ? (0,) : (1, 0))
            @test eltype(a) === T
            reserved = capacity(a)
            @test reserved >= n
            resize!(a, n; fill=T(3))
            @test Vector(a) == fill(T(3), n)
            @test capacity(a) == reserved
        finally
            close(a)
        end
    end
    for C in (FaerMatrix, FaerVector, FaerRow)
        args = C === FaerMatrix ? (2, 3) : (5,)
        a = with_capacity(C, args...)
        @test eltype(a) === Float64
        close(a)
    end
    @test_throws ArgumentError with_capacity(FaerMatrix, -1, 2)
    @test_throws ArgumentError with_capacity(FaerMatrix, 2, -1)
    @test_throws OverflowError with_capacity(FaerMatrix, typemax(Int), 2)
    @test_throws InexactError with_capacity(FaerMatrix, typemax(UInt), 1)
    for C in (FaerVector, FaerRow)
        @test_throws ArgumentError with_capacity(C, -1)
        @test_throws InexactError with_capacity(C, typemax(UInt))
    end
    @test_throws MethodError with_capacity(FaerDiagonal, 5)
end
