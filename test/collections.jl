using Test, LinearAlgebra, Colmat

@testset "Rust-owned columns, rows and diagonals" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        values=T<:Complex ? T[1+2im, -3+im, 4-2im] : T[1, -3, 4]
        v, r, d=FaerVector(values), FaerRow(values), FaerDiagonal(values)
        @test eltype(v)===T
        @test size(v)==(3,)
        @test Matrix(r)==reshape(values, 1, :)
        @test Matrix(d)==Matrix(Diagonal(values))
        @test Vector(v)==values
        @test Array(v)==values
        @test Matrix(FaerMatrix(v))==reshape(values, :, 1)
        for x in (v, r, d)
            @test Array(copy(x))==Array(x)
            @test Array(deepcopy(x))==Array(x)
            @test Array(x+x)≈2Array(x)
            @test Array(x-x)==zero(Array(x))
            @test Array(-x)==-Array(x)
            @test Array(2x)==2Array(x)
            @test Array(x/2)==Array(x)/2
            @test Array(conj(x))==conj(Array(x))
            @test sum(x)≈sum(Array(x))
            for p in (1, 2, Inf, 3)
                @test norm(x, p)≈norm(Array(x), p)
            end
            @test dot(x, x)≈dot(Array(x), Array(x))
            @test Array(zero(x))==zero(Array(x))
            y=copy(x)
            close(y)
            @test !isopen(y)
            @test_throws ArgumentError copy(y)
            @test_throws ArgumentError Array(y)
        end
        @test v'v≈values'values
        @test transpose(v)*v≈transpose(values)*values
        @test Matrix(copy(v'))==reshape(conj(values), 1, :)
        @test Matrix(copy(transpose(v)))==reshape(values, 1, :)
        @test Matrix(copy(r'))==reshape(conj(values), :, 1)
        @test Matrix(d*d)≈Matrix(Diagonal(values)^2)
        @test Vector(diag(d))==values
        v[2]=T(9)
        r[1, 2]=T(8)
        d[2, 2]=T(7)
        @test v[2]==9
        @test r[1, 2]==8
        @test d[2, 2]==7
        @test d[1, 2]==0
        @test_throws ArgumentError d[1, 2]=1
        @test_throws BoundsError v[0]
        @test_throws BoundsError d[4, 4]
        fill!(v, T(3))
        fill!(r, T(4))
        filldiagonal!(d, T(5))
        @test Vector(v)==fill(T(3), 3)
        @test Vector(r)==fill(T(4), 3)
        @test Vector(d)==fill(T(5), 3)
        @test_throws ArgumentError fill!(d, 1)
        fill!(d, 0)
        @test iszero(Matrix(d))
        for x in (v, r)
            before=Vector(x)
            reserve!(x, 16)
            sizehint!(x, 20)
            @test Vector(x)==before
            resize!(x, 5; fill = T(6))
            @test Vector(x)==vcat(before, T[6, 6])
            truncate!(x, 2)
            @test Vector(x)==before[1:2]
            @test_throws DimensionMismatch truncate!(x, 3)
            @test_throws ArgumentError resize!(x, -1)
        end
        for C in (FaerVector, FaerRow, FaerDiagonal)
            x=C{T}(undef, 0)
            @test isempty(x)
            @test isempty(Vector(copy(x)))
            @test sum(x)==zero(T)
            @test norm(x)==0
        end
    end
    @test eltype(FaerVector([1, 2]))===Float64
    @test eltype(FaerVector(Float32[1])+FaerVector(ComplexF64[2im]))===ComplexF64
    @test_throws DimensionMismatch FaerVector([1.0])+FaerVector([1.0, 2])
end
