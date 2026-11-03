using Test, Colmat, LinearAlgebra
@testset "Native matrix storage and capacity" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        A=FaerMatrix{T}(undef, 2, 3)
        @test eltype(A)===T
        @test size(A)==(2, 3)
        @test Matrix(A)==zeros(T, 2, 3)
        @test Matrix(FaerMatrix{T}(I, 2, 3))==Matrix{T}(I, 2, 3)
        @test Matrix(FaerMatrix{T}(2I, 3, 2))==T(2)*Matrix{T}(I, 3, 2)
        A[1, 2]=T(3)
        a=Matrix(A)
        @test reserve!(A, 8, 9)===A
        @test size(A)==size(a)
        @test Matrix(A)==a
        @test sizehint!(A, (10, 10))===A
        @test resize!(A, 4, 5; fill = 2)===A
        expected=fill(T(2), 4, 5)
        expected[1:2, 1:3]=a
        @test Matrix(A)==expected
        @test truncate!(A, 1, 2)===A
        @test Matrix(A)==expected[1:1, 1:2]
        @test resize!(A, 0, 3)===A
        @test size(A)==(0, 3)
        resize!(A, 2, 2)
        @test Matrix(A)==zeros(T, 2, 2)
        @test_throws ArgumentError resize!(A, -1, 2)
        @test_throws ArgumentError reserve!(A, 2, -1)
        @test_throws DimensionMismatch truncate!(A, 4, 4)
        @test size(A)==(2, 2)
        close(A)
        @test_throws ArgumentError resize!(A, 3, 3)
        @test_throws ArgumentError reserve!(A, 3, 3)
    end
    @test eltype(FaerMatrix(undef, 1, 1))===Float64
    @test Matrix(FaerMatrix(I, 2, 2))==Matrix{Float64}(I, 2, 2)
end
