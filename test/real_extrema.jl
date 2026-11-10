using Test, Colmat, Random

@testset "Native real extrema" begin
    rng = MersenneTwister(271)
    for T in (Float32, Float64)
        for dims in ((1, 1), (3, 5), (17, 7)),
            source in (randn(rng, T, dims...), fill(T(3), dims...))

            a = FaerMatrix(source)
            try
                @test minimum(a) isa T
                @test maximum(a) isa T
                @test isequal(minimum(a), minimum(source))
                @test isequal(maximum(a), maximum(source))
                @test isequal(extrema(a), extrema(source))
                @test Matrix(a) == source
                @test fetch(@async extrema(a)) == extrema(source)
                old = a.dims
                a.dims = (old[1] + 1, old[2])
                try
                    @test_throws Exception minimum(a)
                    @test_throws Exception maximum(a)
                finally
                    a.dims = old
                end
                @test extrema(a) == extrema(source)
            finally
                close(a)
            end
            @test_throws ArgumentError minimum(a)
            @test_throws ArgumentError maximum(a)
            @test_throws ArgumentError extrema(a)
        end
        for source in (T[0 -0.0], T[-0.0 0], T[-Inf Inf], T[-0.0], T[0.0])
            data = reshape(source, 1, :)
            a = FaerMatrix(zeros(T, size(data)))
            for i in eachindex(data)
                a[i] = data[i]
            end
            try
                @test isequal(minimum(a), minimum(data))
                @test isequal(maximum(a), maximum(data))
                @test isequal(extrema(a), extrema(data))
            finally
                close(a)
            end
        end
        for dims in ((0, 0), (0, 3), (3, 0))
            a = FaerMatrix(zeros(T, dims...))
            try
                @test_throws ArgumentError minimum(a)
                @test_throws ArgumentError maximum(a)
                @test_throws ArgumentError extrema(a)
            finally
                close(a)
            end
        end
        for position in 1:6
            a = FaerMatrix(reshape(T.(1:6), 2, 3))
            try
                a[position] = T(NaN)
                @test isnan(minimum(a))
                @test isnan(maximum(a))
                @test all(isnan, extrema(a))
            finally
                close(a)
            end
        end
    end
end
