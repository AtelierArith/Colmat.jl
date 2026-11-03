using Test, Colmat, Random, LinearAlgebra, SparseArrays, Statistics

@testset "Native random distributions" begin
    @testset "owned RNG stream" begin
        r=FaerRNG(123)
        s=copy(r)
        @test r.handle.ptr != s.handle.ptr
        @test rand(r, UInt64)==rand(s, UInt64)
        @test rand(r, 10)==rand(s, 10)
        @test randn(r, 10)==randn(s, 10)
        @test Random.seed!(r, 789)===r
        @test rand(r, UInt64)==rand(FaerRNG(789), UInt64)
        s=deepcopy(r)
        close(r)
        GC.gc(true)
        @test !isopen(r)
        @test isopen(s)
        @test rand(s, UInt64) isa UInt64
        @test_throws ArgumentError rand(r, UInt64)
        @test_throws ArgumentError rand(r, FaerMatrix, 0, 0)
        @test_throws ArgumentError copy(r)
        @test_throws ArgumentError Random.seed!(r, 1)
        @test_throws ArgumentError Random.seed!(r)
        @test Random.seed!(s)===s
        @test FaerRNG() isa FaerRNG
        @test_throws ArgumentError FaerRNG(-1)
        @test_throws ArgumentError FaerRNG(big(typemax(UInt64))+1)
    end
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        R=typeof(real(zero(T)))
        tol=1024eps(R)
        for sampler in (rand, randn),
            (m, n) in ((0, 0), (0, 3), (3, 0), (1, 1), (3, 7), (33, 17))

            r=FaerRNG(45)
            s=copy(r)
            A=sampler(r, FaerMatrix{T}, m, n)
            @test A isa FaerMatrix{T}
            @test size(A)==(m, n)
            @test Matrix(A)==Matrix(sampler(s, FaerMatrix{T}, (m, n)))
            @test Matrix(sampler(FaerRNG(7), typeof(A), m, n))==Matrix(
                sampler(FaerRNG(7), FaerMatrix{T}, m, n),
            )
            @test all(isfinite, Matrix(A))
            if sampler===rand
                @test all(x->0<=real(x)<1 && 0<=imag(x)<1, Matrix(A))
            end
        end
        for sampler in (rand, randn), n in (0, 1, 9, 65)
            seed=78
            v=sampler(FaerRNG(seed), FaerVector{T}, n)
            row=sampler(FaerRNG(seed), FaerRow{T}, n)
            @test v isa FaerVector{T}
            @test row isa FaerRow{T}
            @test size(row)==(1, n)
            @test Vector(v)==vec(Matrix(row))
            @test Vector(v)==vec(Matrix(sampler(FaerRNG(seed), FaerMatrix{T}, n, 1)))
            @test Vector(v)==Vector(sampler(FaerRNG(seed), FaerVector{T}, (n,)))
            @test Matrix(row)==Matrix(sampler(FaerRNG(seed), FaerRow{T}, (n,)))
        end
        for n in (0, 1, 2, 7, 33)
            r=FaerRNG(15)
            s=copy(r)
            Q=randunitary(r, T, n)
            n>1 && (@test Matrix(Q)!=Matrix(randunitary(FaerRNG(16), T, n)))
            @test Q isa FaerMatrix{T}
            @test size(Q)==(n, n)
            @test Matrix(Q)==Matrix(randunitary(s, T, n))
            @test adjoint(Matrix(Q))*Matrix(Q) ≈ Matrix{T}(I, n, n) atol=tol
        end
        for fillfn in (rand!, randn!)
            sample=fillfn===rand! ? rand : randn
            r=FaerRNG(31)
            A=FaerMatrix(zeros(T, 4, 5))
            @test fillfn(r, A)===A
            @test Matrix(A)==Matrix(sample(FaerRNG(31), FaerMatrix{T}, 4, 5))
            v=FaerVector(zeros(T, 8))
            @test fillfn(FaerRNG(31), v)===v
            @test Vector(v)==Vector(sample(FaerRNG(31), FaerVector{T}, 8))
            row=FaerRow(zeros(T, 8))
            @test fillfn(FaerRNG(31), row)===row
            @test Matrix(row)==Matrix(sample(FaerRNG(31), FaerRow{T}, 8))
            parent=FaerMatrix(zeros(T, 6, 7))
            region=view(parent, 2:4, 3:6)
            @test fillfn(FaerRNG(31), region)===region
            @test Matrix(region)==Matrix(sample(FaerRNG(31), FaerMatrix{T}, 3, 4))
            @test all(iszero, Matrix(parent)[[1, 5, 6], :])
            close(parent)
            r=FaerRNG(19)
            s=copy(r)
            @test_throws ArgumentError fillfn(r, region)
            @test rand(r, UInt64)==rand(s, UInt64)
        end
        # Julia RNG bridge consumes exactly one UInt64 for a nonempty call.
        r=MersenneTwister(321)
        s=copy(r)
        seed=rand(s, UInt64)
        @test Vector(randn(r, FaerVector{T}, 5))==Vector(
            randn(FaerRNG(seed), FaerVector{T}, 5),
        )
        @test rand(r, UInt64)==rand(s, UInt64)
        r=MersenneTwister(321)
        s=copy(r)
        @test isempty(rand(r, FaerMatrix{T}, 0, 3))
        @test rand(r, UInt64)==rand(s, UInt64)
        r=FaerRNG(321)
        s=copy(r)
        @test isempty(randunitary(r, T, 0))
        @test rand(r, UInt64)==rand(s, UInt64)
        normal=Vector(randn(FaerRNG(44), FaerVector{T}, 50_000))
        @test abs(mean(normal))<0.04
        @test abs(mean(abs2, normal)-1)<0.08
        uniform=Vector(rand(FaerRNG(44), FaerVector{T}, 50_000))
        @test abs(mean(real, uniform)-0.5)<0.04
        if T<:Complex
            @test abs(mean(imag, uniform)-0.5)<0.04
            @test abs(var(real.(normal))-0.5)<0.06
            @test abs(var(imag.(normal))-0.5)<0.06
        end
    end
    for sampler in (rand, randn)
        @test sampler(FaerMatrix{Float64}, 2, 3) isa FaerMatrix{Float64}
        @test sampler(FaerVector{Float32}, 3) isa FaerVector{Float32}
        @test sampler(FaerRow{ComplexF64}, 3) isa FaerRow{ComplexF64}
        @test sampler(FaerMatrix{Float32}, (2, 3)) isa FaerMatrix{Float32}
        @test sampler(FaerMatrix, (2, 3)) isa FaerMatrix{Float64}
        @test sampler(FaerVector, (3,)) isa FaerVector{Float64}
        @test sampler(FaerRow, (3,)) isa FaerRow{Float64}
        r=FaerRNG(9)
        s=copy(r)
        @test_throws ArgumentError sampler(r, FaerMatrix, -1, 2)
        @test_throws OverflowError sampler(r, FaerMatrix, typemax(Int), 2)
        @test rand(r, UInt64)==rand(s, UInt64)
    end
    @test randunitary(3) isa FaerMatrix{Float64}
    @test randunitary(Float32, 3) isa FaerMatrix{Float32}
    @test randunitary(FaerRNG(1), 3) isa FaerMatrix{Float64}
    @test isempty(
        Test.detect_ambiguities(
            Colmat,
            Base,
            Random,
            LinearAlgebra,
            SparseArrays;
            recursive = true,
        ),
    )
end
