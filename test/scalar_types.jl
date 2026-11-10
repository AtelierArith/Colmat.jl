using Test, LinearAlgebra, SparseArrays, Colmat

# Colmat owns exactly four native scalar types. Untyped constructors convert
# other real inputs to Float64 and other complex inputs to ComplexF64; scalar
# arithmetic follows promote_type and rejects non-native results.
@testset "Native scalar type scope" begin
    @testset "untyped constructors" begin
        for (input, T) in (
            ([1 2; 3 4], Float64),
            ([true false; false true], Float64),
            (Float16[1 2; 3 4], Float64),
            ([1//3 2; 3 4], Float64),
            (big.([1.0 2; 3 4]), Float64),
            ([1 + 2im 2; 3 4], ComplexF64),
            (ComplexF16[1 + im 2; 3 4], ComplexF64),
            (Complex{BigFloat}[1 + im 2; 3 4], ComplexF64),
            (Complex{Rational{Int}}[1//2 + im 2; 3 4], ComplexF64),
        )
            A = FaerMatrix(input)
            @test eltype(A) === T
            @test Matrix(A) == T.(input)
            v = vec(input)
            for C in (FaerVector, FaerRow, FaerDiagonal)
                x = C(v)
                @test eltype(x) === T
                @test Vector(x) == T.(v)
            end
            S = FaerSparseMatrix(input)
            @test eltype(S) === T
            @test Matrix(S) == T.(input)
        end
    end

    @testset "explicit unsupported element types" begin
        for T in (BigFloat, Float16, Int, Rational{Int}, Complex{BigFloat}, ComplexF16)
            @test_throws ArgumentError FaerMatrix{T}([1 2; 3 4])
            @test_throws ArgumentError FaerVector{T}([1, 2])
            @test_throws ArgumentError FaerRow{T}([1, 2])
            @test_throws ArgumentError FaerDiagonal{T}([1, 2])
            @test_throws ArgumentError FaerSparseMatrix{T}(sparse([1 0; 0 2]))
        end
        err = try
            FaerMatrix{BigFloat}([1 2; 3 4])
        catch e
            e
        end
        @test occursin(
            "Float32, Float64, ComplexF32 and ComplexF64", sprint(showerror, err)
        )
    end

    @testset "scalar multiplication and division" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            a = T[1 2; 3 4]
            owners = (
                FaerMatrix(a),
                FaerVector(vec(a)),
                FaerRow(vec(a)),
                FaerDiagonal(vec(a)),
                FaerSparseMatrix(a),
            )
            for A in owners,
                x in (2, true, Float16(2), 1//2, π, 2.0f0, 2.0, 2im, 1.5f0 + im)

                P = promote_type(typeof(x), T)
                @test eltype(x * A) === P
                @test eltype(A * x) === P
                @test Array(x * A) ≈ x * Array(A)
                @test Array(A / x) ≈ Array(A) / x
                @test eltype(A / x) === promote_type(typeof(one(T) / x), T)
                A isa FaerMatrix && @test x \ A isa FaerMatrix{eltype(A / x)}
                A isa FaerMatrix && @test Array(x \ A) ≈ x \ Array(A)
            end
            for A in owners, x in (big(2.0), big(2), Complex(big(2.0)))
                P = promote_type(typeof(x), T)
                P in (Float32, Float64, ComplexF32, ComplexF64) && continue
                @test_throws ArgumentError x * A
                @test_throws ArgumentError A * x
                @test_throws ArgumentError A / x
                A isa FaerMatrix && @test_throws ArgumentError x \ A
            end
        end
        # Float64 owners used to round BigFloat scalars silently.
        A = FaerMatrix([1.0 2; 3 4])
        @test_throws ArgumentError big"0.1" * A
        @test_throws ArgumentError A / big(3)
        @test_throws ArgumentError mul!(copy(A), A, A, big(2.0), 0)
        @test Matrix(A) == [1.0 2; 3 4]
    end
end
