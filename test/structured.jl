using Test, LinearAlgebra, Random, Colmat
@testset "Native structured solves" begin
    rng=MersenneTwister(92)
    a=[3.0 1 2; 4 5 1; 2 3 6]
    b=randn(rng, 3, 2)
    for wrapper in
        (LowerTriangular, UpperTriangular, UnitLowerTriangular, UnitUpperTriangular)
        A=wrapper(FaerMatrix(a))
        @test Matrix(A\FaerMatrix(b))≈wrapper(a)\b
        @test A\b[:, 1]≈wrapper(a)\b[:, 1]
        @test Matrix(inv(A))≈inv(wrapper(a))
        B=FaerMatrix(b)
        @test ldiv!(A, B)===B
        @test Matrix(B)≈wrapper(a)\b
        B=FaerMatrix(b)
        @test ldiv!(B, A, B)===B
        @test Matrix(B)≈wrapper(a)\b
    end
    # Unit-diagonal and opposite triangles must not be read.
    A=FaerMatrix([1.0 2; 3 4])
    A[1, 1]=NaN
    A[2, 2]=Inf
    A[1, 2]=NaN
    @test UnitLowerTriangular(A)\[1.0, 2]≈[1.0, -1]
    @test Matrix(inv(UnitLowerTriangular(A)))≈[1.0 0; -3 1]
    A=FaerMatrix(a)
    B=FaerMatrix(b)
    @test Matrix(A'\B)≈a'\b
    @test transpose(A)\b[:, 1]≈transpose(a)\b[:, 1]
    @test Matrix(B'/A)≈b'/a
    @test Matrix(FaerMatrix(b')/A)≈b'/a
    f=lu(A)
    C=FaerMatrix(b')
    @test rdiv!(C, f)===C
    @test Matrix(C)≈b'/a
    x=copy(b[:, 1])
    ldiv!(f, x)
    @test x≈a\b[:, 1]
    c=copy(b)
    ldiv!(f, c)
    @test c≈a\b
    @test isempty(Test.detect_ambiguities(Colmat; recursive = true))
end
