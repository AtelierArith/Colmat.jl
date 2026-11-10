using Colmat, Test, LinearAlgebra

@testset "Owned matrix allocation capacity" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        dims in ((0, 0), (0, 7), (9, 0), (3, 4))

        value = T <: Complex ? T(2 - im) : T(2)
        fillvalue = T <: Complex ? T(-3 + 4im) : T(3)
        data = fill(value, dims)
        a = FaerMatrix(data)
        initial = capacity(a)
        @test initial isa Tuple{Int,Int}
        @test all(initial .>= dims)
        v = view(a, :, :)
        @test reserve!(a, 17, 19) === a
        capacities = capacity(a)
        @test all(capacities .>= (17, 19))
        @test size(a) == dims
        @test Matrix(a) == data
        @test isopen(v)
        @test Matrix(v) == data
        @test_throws MethodError capacity(v)
        close(v)
        @test sizehint!(a, (0, 0)) === a
        @test capacity(a) == capacities
        expanded = fill(fillvalue, capacities)
        expanded[1:dims[1], 1:dims[2]] = data
        @test resize!(a, capacities...; fill=fillvalue) === a
        @test capacity(a) == capacities
        @test Matrix(a) == expanded
        @test truncate!(a, dims...) === a
        @test capacity(a) == capacities
        @test Matrix(a) == data
        b = copy(a)
        close(a)
        @test_throws ArgumentError capacity(a)
        @test all(capacity(b) .>= dims)
        @test Matrix(b) == data
        close(b)
    end
end

@testset "Owned collection allocation capacity" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        C in (FaerVector, FaerRow, FaerDiagonal),
        n in (0, 3, 17)

        value = T <: Complex ? T(2 - im) : T(2)
        fillvalue = T <: Complex ? T(-3 + 4im) : T(3)
        data = fill(value, n)
        x = C(data)
        @test capacity(x) isa Int
        @test capacity(x) >= n
        @test Vector(x) == data
        if C !== FaerDiagonal
            @test reserve!(x, 19) === x
            reserved = capacity(x)
            @test reserved >= 19
            @test Vector(x) == data
            @test sizehint!(x, 0) === x
            @test capacity(x) == reserved
            @test resize!(x, reserved; fill=fillvalue) === x
            @test capacity(x) == reserved
            expected = vcat(data, fill(fillvalue, reserved - n))
            @test Vector(x) == expected
            @test truncate!(x, n) === x
            @test capacity(x) == reserved
            @test Vector(x) == data
        else
            @test size(x) == (n, n)
        end
        y = copy(x)
        close(x)
        @test_throws ArgumentError capacity(x)
        @test capacity(y) >= n
        @test Vector(y) == data
        close(y)
    end
end

@testset "Checked capacity outputs and conversion bounds" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        a = FaerMatrix(ones(T, 2, 3))
        for n in (0, 1, 3)
            out = fill(UInt(91), n)
            @test_throws Colmat.RustCall.RustError Colmat._native_call(
                Colmat.RustCrate.storage_capacities, a, out
            )
            @test out == fill(UInt(91), n)
        end
        out = zeros(UInt, 2)
        Colmat._native_call(Colmat.RustCrate.storage_capacities, a, out)
        @test Tuple(Int.(out)) == capacity(a)
        close(a)
    end
    uninitialized = Colmat.RustCrate.TypedMatrix()
    out = fill(UInt(91), 2)
    @test_throws Colmat.RustCall.RustError Colmat._native_call(
        Colmat.RustCrate.storage_capacities, uninitialized, out
    )
    @test out == fill(UInt(91), 2)
    collection = Colmat.RustCrate.VectorOwner()
    @test_throws Colmat.RustCall.RustError Colmat._native_call(
        Colmat.RustCrate.collection_capacity, collection
    )
    @test Colmat._capacity_int(UInt(typemax(Int))) == typemax(Int)
    @test_throws OverflowError Colmat._capacity_int(UInt(typemax(Int)) + UInt(1))
end
