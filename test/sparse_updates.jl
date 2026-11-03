using Colmat, Test, LinearAlgebra, SparseArrays, Random

@testset "Fixed-pattern native sparse arithmetic" begin
    rng=MersenneTwister(514)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        destformat in (:csc, :csr),
        srcformat in (:csc, :csr)

        a=randn(rng, T, 5, 7)
        b=sprand(rng, T, 5, 7, 0.4)
        c=sprand(rng, T, 5, 7, 0.35)
        src=FaerSparseMatrix(b; format = srcformat)
        rhs=FaerSparseMatrix(c; format = srcformat)
        for source in (src, b), (op, sign) in ((add_assign!, 1), (sub_assign!, -1))
            dst=FaerSparseMatrix(sparse(a); format = destformat)
            storage=compressed_storage(dst)
            values=nonzeros(dst)
            epoch=dst.epoch
            @test op(dst, source)===dst
            @test Matrix(dst) ≈ a+sign*Matrix(b)
            @test dst.epoch==epoch
            @test nnz(dst)==length(a)
            @test compressed_storage(dst).pointers==storage.pointers
            @test compressed_storage(dst).indices==storage.indices
            @test Vector(values)==compressed_storage(dst).values
            @test sparse(src)==b
        end
        for (op, sign) in ((add_into!, 1), (sub_into!, -1)),
            left in (src, b),
            right in (rhs, c)

            dst=FaerSparseMatrix(sparse(a); format = destformat)
            values=nonzeros(dst)
            @test op(dst, left, right)===dst
            @test Matrix(dst) ≈ Matrix(b)+sign*Matrix(c)
            @test nnz(dst)==length(a)
            @test Vector(values)==compressed_storage(dst).values
            @test sparse(src)==b && sparse(rhs)==c
        end
        for (op, sign) in ((add_assign!, 1), (sub_assign!, -1))
            dst=FaerSparseMatrix(sparse(a); format = destformat)
            @test op(dst, dst)===dst
            @test Matrix(dst) ≈ (1+sign)*a
            @test nnz(dst)==length(a)
        end
        for (op, sign) in ((add_into!, 1), (sub_into!, -1))
            for side in (:left, :right, :both)
                dst=FaerSparseMatrix(sparse(a); format = destformat)
                lhs=side in (:left, :both) ? dst : src
                rhs=side in (:right, :both) ? dst : src
                expected=Matrix(lhs)+sign*Matrix(rhs)
                @test op(dst, lhs, rhs)===dst
                @test Matrix(dst) ≈ expected
            end
        end
    end
end
@testset "Fixed updates preserve patterns and fail before mutation" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), format in (:csc, :csr)
        a=T[1 0 0; 2 3 0; 0 4 5]
        s=FaerSparseMatrix(sparse(a); format)
        good=FaerSparseMatrix(sparse(T[0 0 0; 1 2 0; 0 0 0]))
        # A missing source position still fails when its stored value is zero.
        bad=FaerSparseMatrix([1, 3], [1, 1], T[8, 0], 3, 3)
        values=nonzeros(s)
        before=compressed_storage(s)
        for op in (add_assign!, sub_assign!)
            @test_throws ArgumentError op(s, bad)
            @test compressed_storage(s)==before
            @test Vector(values)==before.values
        end
        for op in (add_into!, sub_into!), (lhs, rhs) in ((bad, good), (good, bad))
            @test_throws ArgumentError op(s, lhs, rhs)
            @test compressed_storage(s)==before
            @test Vector(values)==before.values
        end
        for op in (add_assign!, sub_assign!)
            @test_throws DimensionMismatch op(s, FaerSparseMatrix(spzeros(T, 2, 2)))
        end
        @test_throws DimensionMismatch add_into!(
            s,
            good,
            FaerSparseMatrix(spzeros(T, 2, 2)),
        )
        @test_throws DimensionMismatch sub_into!(
            s,
            FaerSparseMatrix(spzeros(T, 2, 2)),
            good,
        )
        close(bad)
        @test_throws ArgumentError add_assign!(s, bad)
        @test compressed_storage(s)==before
        close(s)
        @test_throws ArgumentError sub_assign!(s, good)
        empty=FaerSparseMatrix(spzeros(T, 0, 7); format)
        @test add_assign!(empty, copy(empty))===empty
        @test add_into!(empty, empty, empty)===empty
        @test nnz(empty)==0
        # Explicit zeros are retained on cancellation; views remain live.
        s=FaerSparseMatrix(sparse(a); format)
        values=nonzeros(s)
        @test sub_assign!(s, s)===s
        @test nnz(s)==nnz(sparse(a))
        @test all(iszero, values)
    end
end
@testset "Fixed update conversion and sparse transforms" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), format in (:csc, :csr)
        a=T[1 2; 3 4]
        b=ComplexF64[2+im 0; 1-im 5]
        for source in (FaerSparseMatrix(sparse(a))', transpose(FaerSparseMatrix(sparse(a))))
            d=FaerSparseMatrix(sparse(a); format)
            expected=a+Matrix(source)
            @test Matrix(add_assign!(d, source)) ≈ expected
        end
        d=FaerSparseMatrix(sparse(a); format)
        if T<:Real
            before=compressed_storage(d)
            @test_throws InexactError add_assign!(d, FaerSparseMatrix(sparse(b)))
            @test compressed_storage(d)==before
            @test_throws InexactError add_into!(
                d,
                FaerSparseMatrix(sparse(a)),
                FaerSparseMatrix(sparse(b)),
            )
            @test compressed_storage(d)==before
        else
            @test Matrix(add_assign!(d, FaerSparseMatrix(sparse(b)))) ≈ a+T.(b)
        end
        source=FaerSparseMatrix(sparse(Float64[1/3 1/7; 1/9 1/11]))
        d=FaerSparseMatrix(sparse(a); format)
        @test Matrix(add_assign!(d, source)) ≈ a+T.(Matrix(source))
        @test eltype(d)===T
    end
    @test isempty(
        detect_ambiguities(Colmat, Base, LinearAlgebra, SparseArrays; recursive = true),
    )
end
@testset "Fixed updates retain sparse scaling" begin
    n=50_000
    for format in (:csc, :csr)
        a=FaerSparseMatrix(spdiagm(0=>fill(2.0, n)); format)
        b=FaerSparseMatrix(spdiagm(0=>ones(n)); format)
        values=nonzeros(a)
        @test add_assign!(a, b)===a
        @test all(==(3.0), values)
        @test sub_into!(a, a, b)===a
        @test all(==(2.0), values)
        @test nnz(a)==n
    end
    a=FaerSparseMatrix([1, 2], [1, 10^9], [2.0, 3.0], 2, 10^9; format = :csr)
    b=copy(a)
    @test add_assign!(a, b)===a
    @test a[1, 1]==4 && a[2, 10^9]==6
    @test sub_into!(a, a, b)===a
    @test a[1, 1]==2 && a[2, 10^9]==3
    @test nnz(a)==2
end

@testset "Overwrite zeros after both input columns end" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        format in (:csc, :csr),
        op in (add_into!, sub_into!)

        dest=FaerSparseMatrix(sparse(fill(T(7), 3, 3)); format)
        lhs=FaerSparseMatrix(sparse(T[1 0 0; 0 0 0; 0 0 0]); format)
        rhs=copy(lhs)
        expected=op===add_into! ? 2Matrix(lhs) : zeros(T, 3, 3)
        @test Matrix(op(dest, lhs, rhs))==expected
        @test nnz(dest)==9
        empty=FaerSparseMatrix(spzeros(T, 3, 3); format)
        @test Matrix(op(dest, empty, empty))==zeros(T, 3, 3)
        @test nnz(dest)==9
    end
end
