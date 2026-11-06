using Colmat, Test

initializer_value(T, i, j = 0) = T<:Complex ? T(i+2j, i-j) : T(i+2j)

@testset "Function initialization preserves matrix overlap and order" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        old in ((2, 3), (0, 0), (0, 3), (2, 0)),
        dims in ((4, 5), (1, 5), (4, 1), (1, 1), (0, 4), (4, 0), (2, 3))

        data=fill(T(7), old)
        a=FaerMatrix(data)
        v=view(a, :, :)
        calls=Tuple{Int,Int}[]
        expected=[
            i<=old[1]&&j<=old[2] ? data[i, j] : initializer_value(T, i, j) for
            i = 1:dims[1], j = 1:dims[2]
        ]
        expected_calls=[(i, j) for j = 1:dims[2] for i = 1:dims[1] if i>old[1]||j>old[2]]
        reserve!(a, max(old[1], dims[1]), max(old[2], dims[2]))
        reserved=capacity(a)
        try
            result=resize_with!(a, dims...) do i, j
                push!(calls, (i, j))
                initializer_value(T, i, j)
            end
            @test result===a
            @test calls==expected_calls
            @test size(a)==dims
            @test Matrix(a)==expected
            @test capacity(a)==reserved
            @test isopen(v)==(dims==old)
        finally
            close(v)
            close(a)
        end
    end
end

@testset "Function initialization of columns and rows" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        C in (FaerVector, FaerRow),
        old in (0, 3),
        n in (0, 2, 3, 7)

        data=fill(T(7), old)
        a=C(data)
        calls=Int[]
        try
            @test resize_with!(i->(push!(calls, i); initializer_value(T, i)), a, n)===a
            @test calls==collect((old+1):n)
            @test Vector(a)==[i<=old ? data[i] : initializer_value(T, i) for i = 1:n]
        finally
            close(a)
        end
    end
end

@testset "Initializer errors precede native mutation" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        a=FaerMatrix(fill(T(7), 2, 3))
        try
            calls=Ref(0)
            bad(i, j) = (calls[]+=1; calls[]==2 ? error("initializer failed") : T(1))
            @test_throws ErrorException resize_with!(bad, a, 4, 5)
            @test calls[]==2
            @test Matrix(a)==fill(T(7), 2, 3)
            @test size(a)==(2, 3)
            @test_throws Exception resize_with!((i, j)->"not a scalar", a, 4, 5)
            @test Matrix(a)==fill(T(7), 2, 3)
            @test_throws ArgumentError resize_with!((i, j)->error("must not run"), a, -1, 5)
            @test_throws OverflowError resize_with!(
                (i, j)->error("must not run"),
                a,
                typemax(Int),
                2,
            )
            native=real(T)===Float32 ? Colmat.RustCrate.resize_matrix_values_f32 :
                   Colmat.RustCrate.resize_matrix_values_f64
            @test_throws Colmat.RustCall.RustError Colmat._native_call(
                native,
                a,
                UInt(1),
                UInt(1),
                UInt(2),
                UInt(3),
                Int32(99),
                real(T)[],
            )
            @test Matrix(a)==fill(T(7), 2, 3)
            values=zeros(real(T), 1)
            native=real(T)===Float32 ? Colmat.RustCrate.resize_matrix_values_f32 :
                   Colmat.RustCrate.resize_matrix_values_f64
            @test_throws Colmat.RustCall.RustError Colmat._native_call(
                native,
                a,
                UInt(4),
                UInt(5),
                UInt(2),
                UInt(3),
                Colmat._tag(T),
                values,
            )
            @test Matrix(a)==fill(T(7), 2, 3)
            @test_throws Colmat.RustCall.RustError Colmat._native_call(
                native,
                a,
                UInt(2),
                UInt(3),
                UInt(1),
                UInt(3),
                Colmat._tag(T),
                real(T)[],
            )
            @test Matrix(a)==fill(T(7), 2, 3)
        finally
            close(a)
        end
        @test_throws ArgumentError resize_with!((i, j)->error("must not run"), a, 3, 3)
        for C in (FaerVector, FaerRow)
            a=C(fill(T(7), 3))
            try
                @test_throws ErrorException resize_with!(
                    i->error("initializer failed"),
                    a,
                    7,
                )
                @test Vector(a)==fill(T(7), 3)
                @test_throws Exception resize_with!(i->"not a scalar", a, 7)
                @test Vector(a)==fill(T(7), 3)
                native=real(T)===Float32 ? Colmat.RustCrate.resize_collection_values_f32 :
                       Colmat.RustCrate.resize_collection_values_f64
                @test_throws Colmat.RustCall.RustError Colmat._native_call(
                    native,
                    a,
                    UInt(2),
                    UInt(3),
                    Int32(1-Colmat._collection_kind(a)),
                    Colmat._tag(T),
                    real(T)[],
                )
                @test Vector(a)==fill(T(7), 3)
                @test_throws Colmat.RustCall.RustError Colmat._native_call(
                    native,
                    a,
                    UInt(2),
                    UInt(3),
                    Colmat._collection_kind(a),
                    Int32(99),
                    real(T)[],
                )
                @test Vector(a)==fill(T(7), 3)
            finally
                close(a)
            end
            @test_throws ArgumentError resize_with!(i->error("must not run"), a, 7)
        end
    end
end

@testset "Owner mutation inside initializer is rejected" begin
    a=FaerMatrix(ones(2, 2))
    try
        @test_throws ArgumentError resize_with!((i, j)->(resize!(a, 1, 1); 1.0), a, 3, 3)
        @test size(a)==(1, 1)
    finally
        close(a)
    end
    a=FaerVector([1.0])
    @test_throws ArgumentError resize_with!(i->(close(a); 1.0), a, 3)
    @test !isopen(a)
end
