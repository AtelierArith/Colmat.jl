using Test, LinearAlgebra, Random, Colmat

@testset "Native permutation ownership and conventions" begin
    rng=MersenneTwister(749)
    for n in (0, 1, 2, 5, 13)
        indices=randperm(rng, n)
        other=randperm(rng, n)
        p, q=FaerPermutation(indices), FaerPermutation(other)
        @test p isa AbstractVector{Int}
        @test Vector(p)==indices
        @test collect(p)==indices
        @test Vector(copy(p))==indices
        @test Vector(deepcopy(p))==indices
        @test Vector(invperm(p))==invperm(indices)
        @test Vector(inv(p))==invperm(indices)
        @test Vector(p*q)==indices[other]
        @test Vector(p*invperm(p))==collect(1:n)
        @test isperm(p)
        @test Matrix(p)==Matrix{Float64}(I, n, n)[indices, :]
        @test Matrix{Float32}(p)==Matrix{Float32}(I, n, n)[indices, :]
        @test Matrix(FaerMatrix(p))==Matrix(p)
        saved=copy(p)
        ip=invperm(p)
        close(p)
        @test !isopen(p)
        @test isopen(saved)
        @test Vector(saved)==indices
        @test Vector(ip)==invperm(indices)
        @test_throws ArgumentError Vector(p)
        @test_throws ArgumentError copy(p)
        @test_throws ArgumentError isperm(p)
        if n>0
            @test saved[1]==indices[1]
            @test_throws ArgumentError p[1]
        end
        @test_throws BoundsError saved[0]
        @test_throws BoundsError saved[n+1]
    end
    for bad in ([0, 1], [1, 1], [1, 3], [-1, 2], [2], [1, 2, 4])
        @test_throws ArgumentError FaerPermutation(bad)
    end
    @test_throws DimensionMismatch FaerPermutation([1])*FaerPermutation([2, 1])
end

@testset "Native row and column permutation" begin
    rng=MersenneTwister(209)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        (m, n) in ((4, 3), (0, 3), (4, 0), (0, 0), (1, 1))

        data=randn(rng, T, m, n)
        a=FaerMatrix(data)
        pi, qi=randperm(rng, m), randperm(rng, n)
        p, q=FaerPermutation(pi), FaerPermutation(qi)
        @test Matrix(permute_rows(a, p))==data[pi, :]
        @test Matrix(permute_cols(a, q))==data[:, qi]
        @test Matrix(a[p, :])==data[pi, :]
        @test Matrix(a[:, q])==data[:, qi]
        @test Matrix(a[p, q])==data[pi, qi]
        @test Matrix(permute_rows(a, pi))==data[pi, :]
        @test Matrix(permute_cols(a, qi))==data[:, qi]
        @test eltype(a[p, q])===T
        @test Matrix(a)==data
        held=view(a, :, :)
        @test permute_rows!(a, p)===a
        @test Matrix(a)==data[pi, :]
        @test Matrix(held)==data[pi, :]
        @test permute_cols!(a, q)===a
        @test Matrix(a)==data[pi, qi]
        permute_rows!(a, invperm(p))
        permute_cols!(a, invperm(q))
        @test Matrix(a)==data
        if m>0
            swap_rows!(a, 1, m)
            expected=copy(data)
            expected[[1, m], :]=data[[m, 1], :]
            @test Matrix(a)==expected
            swap_rows!(a, 1, m)
            @test Matrix(a)==data
        end
        if n>0
            swap_cols!(a, 1, n)
            expected=copy(data)
            expected[:, [1, n]]=data[:, [n, 1]]
            @test Matrix(a)==expected
            swap_cols!(a, 1, n)
            @test Matrix(a)==data
        end
        @test_throws BoundsError swap_rows!(a, 0, 1)
        @test_throws BoundsError swap_cols!(a, 1, n+1)
        @test_throws DimensionMismatch permute_rows!(a, collect(1:(m+1)))
        @test_throws DimensionMismatch permute_cols(a, collect(1:(n+1)))
        @test Matrix(a)==data
        close(a)
        @test_throws ArgumentError permute_rows!(a, p)
    end
end

@testset "Permuting collections, views and factor permutations" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        data=T[10, 20, 30, 40]
        p=FaerPermutation([3, 1, 4, 2])
        v=FaerVector(data)
        held=view(v, :)
        @test permute!(v, p)===v
        @test Vector(v)==data[Vector(p)]
        @test Vector(held)==Vector(v)
        @test invpermute!(v, p)===v
        @test Vector(v)==data
        permute!(v, [4, 3, 2, 1])
        @test Vector(v)==reverse(data)
        row=FaerRow(data)
        permute_cols!(row, p)
        @test Vector(row)==data[Vector(p)]
        swap_cols!(row, 1, 4)
        @test Vector(row)==data[Vector(p)[[4, 2, 3, 1]]]
        @test_throws BoundsError swap_cols!(row, 1, 5)
        @test_throws DimensionMismatch permute!(v, [2, 1])
        @test_throws ArgumentError permute!(v, [1, 1, 1, 1])
        a=FaerMatrix(reshape(T.(1:30), 5, 6))
        snapshot=Matrix(a)
        w=view(a, 5:-2:1, 2:2:6)
        expected=snapshot[5:-2:1, 2:2:6]
        pp=FaerPermutation([3, 1, 2])
        @test Matrix(permute_rows(w, pp))==expected[Vector(pp), :]
        permute_rows!(w, pp)
        @test Matrix(w)==expected[Vector(pp), :]
        expected=expected[Vector(pp), :]
        permute_cols!(w, pp)
        @test Matrix(w)==expected[:, Vector(pp)]
        expected=expected[:, Vector(pp)]
        swap_rows!(w, 1, 3)
        expected[[1, 3], :]=expected[[3, 1], :]
        swap_cols!(w, 1, 3)
        expected[:, [1, 3]]=expected[:, [3, 1]]
        @test Matrix(w)==expected
        snapshot[5:-2:1, 2:2:6]=expected
        @test Matrix(a)==snapshot
        f=lu(FaerMatrix(T[0 2 1; 3 1 2; 1 0 4]))
        fp=FaerPermutation(f)
        @test Vector(fp)==f.p
        @test Matrix(f.L*f.U)≈Matrix(FaerMatrix(f))[fp, :]
        g=full_piv_lu(FaerMatrix(T[0 2 1; 3 1 2; 1 0 4]))
        @test Vector(FaerPermutation(g; column = true))==g.q
        close(f)
        @test isopen(fp)
    end
    @test isempty(detect_ambiguities(Colmat, Base, LinearAlgebra; recursive = true))
end
