using Colmat, Test, LinearAlgebra

@testset "RustCall checked owner bindings" begin
    @testset "fallible copies own independent storage" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            data = T[4 1; 2 3]
            a = FaerMatrix(data)
            b = copy(a)
            @test typeof(b) === typeof(a)
            @test b.handle.ptr != a.handle.ptr
            @test Matrix(b) == data
            b[1, 1] = T(10)
            @test a[1, 1] == T(4)
            close(a)
            GC.gc(true)
            @test Matrix(b) == T[10 1; 2 3]
            @test_throws ArgumentError copy(a)
        end
        # TypedMatrix::duplicate returns Err without allocating a new owner.
        h = Colmat.RustCrate.TypedMatrix()
        @test_throws Colmat.RustCall.RustError Colmat.RustCall.result_to_exception(
            Colmat.RustCrate.duplicate(h),
        )
    end

    @testset "checked references reach the generated wrappers" begin
        a = FaerMatrix([4.0 1; 2 3])
        b = FaerMatrix([2.0 0; 0 2])
        @test Matrix(a + a) == 2Matrix(a) # Shared borrows may alias.
        @test dot(a, a) == sum(abs2, Matrix(a))
        @test_throws ArgumentError Colmat._native_call(Colmat.RustCrate.assign, a, a)
        @test Matrix(a) == [4.0 1; 2 3]
        @test copyto!(a, a) === a # The public self-copy avoids an aliased borrow.
        typed = FaerMatrix(ones(Float32, 2, 2))
        @test_throws ArgumentError Colmat._native_call(Colmat.RustCrate.assign, a, typed)
        @test Matrix(a * b) == Matrix(a) * Matrix(b)

        # A finalized non-receiver argument must be rejected before entering Rust.
        freed = Colmat.RustCrate.OwnedMatrix()
        finalize(freed)
        @test_throws Colmat.RustCall.RustError Colmat.RustCrate.assign(a.handle, freed)
        @test Matrix(a) == [4.0 1; 2 3]
        close(b)
        @test_throws ArgumentError a + b
    end
end
