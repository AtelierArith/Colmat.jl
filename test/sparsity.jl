using Test, LinearAlgebra, SparseArrays, Colmat

@testset "Native value-free sparse patterns" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), fmt in (:csc, :csr)
        a=FaerSparseMatrix([1, 3, 2], [1, 2, 4], T[2, 0, 3], 3, 4; format = fmt)
        p=FaerSparsity(a)
        mask=falses(3, 4)
        mask[1, 1]=mask[3, 2]=mask[2, 4]=true
        @test p isa AbstractSparseMatrix{Bool,Int}
        @test nnz(p)==3
        @test Matrix(p)==mask
        @test p[3, 2]
        @test !p[1, 4]
        @test Matrix(copy(p))==mask
        @test Matrix(deepcopy(p))==mask
        @test Matrix(copy(p; format = fmt===:csc ? :csr : :csc))==mask
        @test Matrix(copy(transpose(p)))==transpose(mask)
        @test Matrix(copy(adjoint(p)))==transpose(mask)
        @test !hasproperty(compressed_storage(p), :values)
        b=FaerSparseMatrix{T}(undef, p)
        @test nnz(b)==3 && all(iszero, Vector(nonzeros(b)))
        @test Matrix(FaerSparseMatrix{T}(p))==T.(mask)
        values=Vector(nonzeros(a))
        b=FaerSparseMatrix(p, values)
        @test Matrix(b)==Matrix(a)
        fill!(nonzeros(b), T(7))
        @test Matrix(p)==mask
        dropzeros!(a)
        a[1, 4]=T(9)
        @test Matrix(p)==mask
        for otherfmt in (:csc, :csr)
            q=FaerSparsity(
                FaerSparseMatrix([1, 2], [4, 2], T[0, 4], 3, 4; format = otherfmt),
            )
            u=pattern_union(p, q)
            othermask=falses(3, 4)
            othermask[1, 4]=othermask[2, 2]=true
            @test Matrix(u)==(mask .| othermask)
            @test nnz(u)==5
            @test Matrix(pattern_union(p, q; format = :csr))==Matrix(u)
        end
        saved=copy(p)
        close(p)
        @test !isopen(p)
        @test_throws ArgumentError p[1, 1]
        @test_throws ArgumentError FaerSparseMatrix{T}(p)
        @test Matrix(saved)==mask
        @test_throws DimensionMismatch FaerSparseMatrix(saved, T[1])
        close(a)
        @test Matrix(saved)==mask
    end
    a=FaerSparseMatrix([1, 2], [1, 10^9], [2.0, 0.0], 2, 10^9; format = :csr)
    p=FaerSparsity(a)
    close(a)
    @test p[2, 10^9] && nnz(p)==2
    @test compressed_storage(p).pointers==[1, 2, 3]
    @test sum(FaerSparseMatrix(p, [3.0, 4.0]))==7
    @test nnz(pattern_union(p, p))==2
end
