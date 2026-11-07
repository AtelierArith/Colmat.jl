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
        # Julia scalar sampling below 64/above 64 bits derives from native UInt64 words.
        for U in (Bool, Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64)
            @test rand(FaerRNG(5), U)===rand(FaerRNG(5), UInt64)%U
        end
        for U in (Int128, UInt128)
            r=FaerRNG(5)
            hi, lo=rand(r, UInt64), rand(r, UInt64)
            @test rand(FaerRNG(5), U)===((UInt128(hi)<<64)|lo)%U
        end
        for U in (Float16, Float32, ComplexF32, Char, UInt32(1):UInt32(9))
            r=FaerRNG(6)
            s=copy(r)
            @test rand(r, U, 4)==rand(s, U, 4)
        end
        @test all(x->0<=x<1, rand(FaerRNG(7), Float32, 100))
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
    @testset "Julia sampler fills" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            collections = Any[1:5, T[2, 3.5, -1], T, Random.Sampler(Xoshiro, 0:3)]
            T<:Complex && push!(collections, T[1+2im, -3im])
            for X in collections, make in (MersenneTwister, Xoshiro, FaerRNG)
                for (target, dims) in (
                    (zeros(FaerMatrix{T}, 4, 5), (4, 5)),
                    (zeros(FaerVector{T}, 7), (7,)),
                    (zeros(FaerRow{T}, 6), (1, 6)),
                    (zeros(FaerMatrix{T}, 0, 3), (0, 3)),
                )
                    r=make(55)
                    s=copy(r)
                    @test rand!(r, target, X)===target
                    expected=rand!(s, Array{T}(undef, dims), X)
                    @test reshape(Array(target), dims)==expected
                    @test rand(r, UInt64)==rand(s, UInt64)
                    close(target)
                end
                parent=FaerMatrix(zeros(T, 6, 7))
                region=view(parent, 2:4, 3:6)
                r=make(56)
                s=copy(r)
                @test rand!(r, region, X)===region
                @test Matrix(region)==rand!(s, Matrix{T}(undef, 3, 4), X)
                @test all(iszero, Matrix(parent)[[1, 5, 6], :])
                @test all(iszero, Matrix(parent)[:, [1, 2, 7]])
                column=view(parent, 1:3, 1)
                @test rand!(r, column, X)===column
                @test Vector(column)==rand!(s, Vector{T}(undef, 3), X)
                r=make(57)
                s=copy(r)
                close(parent)
                @test_throws ArgumentError rand!(r, region, X)
                @test rand(r, UInt64)==rand(s, UInt64)
            end
            Random.seed!(1)
            a=rand!(zeros(FaerVector{T}, 4), 1:3)
            Random.seed!(1)
            @test Vector(a)==rand!(Vector{T}(undef, 4), 1:3)
            # Sampling from the target (or its parent) reads a pre-write state.
            values=T[1 2 3; 4 5 6; 7 8 9]
            A=FaerMatrix(values)
            r=Xoshiro(58)
            s=copy(r)
            @test rand!(r, A, A)===A
            @test Matrix(A)==rand!(s, Matrix{T}(undef, 3, 3), values)
            P=FaerMatrix(values)
            region=view(P, 1:2, 2:3)
            rand!(r, region, P)
            expected=copy(values)
            expected[1:2, 2:3]=rand!(s, Matrix{T}(undef, 2, 2), values)
            @test Matrix(P)==expected
            # Sampling and conversion finish before the native target changes.
            A=zeros(FaerMatrix{T}, 4, 5)
            @test_throws MethodError rand!(Xoshiro(1), A, Any[1, "x"])
            @test Matrix(A)==zeros(T, 4, 5)
            if T<:Real
                @test_throws InexactError rand!(Xoshiro(1), A, [0.5im])
                @test Matrix(A)==zeros(T, 4, 5)
            end
        end
    end
    @testset "bulk staging allocations" begin
        # Staged output and the native snapshot are required; per-element writes are not.
        for T in (Float32, ComplexF64)
            r=Xoshiro(2)
            A=zeros(FaerMatrix{T}, 64, 64)
            bound=8*sizeof(T)*length(A)
            for fill in (a->rand!(r, a, 1:5), a->rand!(r, a), a->randn!(r, a))
                fill(A)
                @test @allocated(fill(A))<bound
            end
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
