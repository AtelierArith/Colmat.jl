using Test, LinearAlgebra, SparseArrays, Colmat

_bulk_copyto_bytes(dest, src) = @allocated copyto!(dest, src)

@testset "Bulk copyto! into native owners" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        data=reshape(T.(1:30), 5, 6)
        T<:Complex && (data .+= im .* reverse(data; dims = 1))
        # Julia sources take one native view assignment, not per-element setindex!.
        big=FaerMatrix{T}(undef, 100, 100)
        buffer=rand(T, 100, 100)
        _bulk_copyto_bytes(big, buffer)
        @test _bulk_copyto_bytes(big, buffer)<4*sizeof(buffer)
        @test Matrix(big)==buffer
        row=FaerRow{T}(undef, 1000)
        rowbuffer=rand(T, 1, 1000)
        _bulk_copyto_bytes(row, rowbuffer)
        @test _bulk_copyto_bytes(row, rowbuffer)<4*sizeof(rowbuffer)
        @test Matrix(row)==rowbuffer

        a=FaerMatrix(zero(data))
        @test copyto!(a, data)===a
        @test Matrix(a)==data
        # Sources aliasing the destination are snapshotted before the write.
        a=FaerMatrix(data)
        copyto!(a, view(a, 5:-1:1, :))
        @test Matrix(a)==data[5:-1:1, :]
        s=FaerMatrix(data[1:5, 1:5])
        copyto!(s, transpose(s))
        @test Matrix(s)==transpose(data[1:5, 1:5])
        r=FaerRow(T.(1:6))
        copyto!(r, view(r, :, 6:-1:1))
        @test Matrix(r)==reshape(T.(6:-1:1), 1, 6)

        # Base linear-order semantics for differently shaped sources.
        a=FaerMatrix(zero(data))
        copyto!(a, reshape(data, 6, 5))
        @test Matrix(a)==data
        copyto!(a, T.(1:30))
        @test vec(Matrix(a))==T.(1:30)
        a=FaerMatrix(data)
        copyto!(a, T[-1, -2, -3])
        expected=copy(data)
        expected[1:3]=T[-1, -2, -3]
        @test Matrix(a)==expected
        r=FaerRow(zeros(T, 4))
        copyto!(r, T[1, 2, 3, 4])
        @test Matrix(r)==reshape(T[1, 2, 3, 4], 1, 4)
        @test copyto!(r, T[])===r
        # Too-long sources and failed conversions leave the destination intact.
        a=FaerMatrix(data)
        @test_throws BoundsError copyto!(a, ones(T, 31))
        @test_throws BoundsError copyto!(r, ones(T, 2, 3))
        @test Matrix(a)==data
        @test Matrix(r)==reshape(T[1, 2, 3, 4], 1, 4)
        # Native matrix-to-matrix copies keep their shape-strict contract.
        @test_throws DimensionMismatch copyto!(a, FaerMatrix(permutedims(data)))
        @test_throws DimensionMismatch copyto!(a, FaerMatrix{ComplexF64}(permutedims(data)))
        @test Matrix(a)==data
        bad=Any[fill(T(1), 29); "x"]
        @test_throws MethodError copyto!(a, reshape(bad, 5, 6))
        @test_throws MethodError copyto!(r, Any[1, 2, 3, "x"])
        @test Matrix(a)==data
        @test Matrix(r)==reshape(T[1, 2, 3, 4], 1, 4)
        if T<:Real
            @test_throws InexactError copyto!(a, fill(complex(T(1), T(1)), 5, 6))
            @test Matrix(a)==data
        end
        # Sparse sources stay on the bulk path.
        sp=sparse(data)
        a=FaerMatrix(zero(data))
        copyto!(a, sp)
        @test Matrix(a)==data
        # Closed destinations reject writes.
        close(a)
        @test_throws ArgumentError copyto!(a, data)
        close(r)
        @test_throws ArgumentError copyto!(r, T[1, 2, 3, 4])
        @test_throws ArgumentError copyto!(r, T[])
    end
    @test isempty(
        Test.detect_ambiguities(Colmat, Base, LinearAlgebra, SparseArrays; recursive = true),
    )
end
