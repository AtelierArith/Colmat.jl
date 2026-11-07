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
end

@testset "Native in-place view permutation and caller-owned scratch" begin
    rng=MersenneTwister(95)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        data=randn(rng, T, 6, 7)
        a=FaerMatrix(data)
        held=view(a, :, :)
        w=view(a, 6:-2:1, 2:2:7)
        expected=copy(data)
        sub=expected[6:-2:1, 2:2:7]
        p, q=FaerPermutation([2, 3, 1]), FaerPermutation([3, 1, 2])
        @test permute_rows(w, p)==sub[Vector(p), :]
        @test Matrix(a)==data
        @test permute_rows!(w, p)===w
        sub=sub[Vector(p), :]
        @test permute_cols!(w, q)===w
        sub=sub[:, Vector(q)]
        expected[6:-2:1, 2:2:7]=sub
        @test Matrix(a)==expected
        @test Matrix(held)==expected
        rq=scratch_requirements(permute_rows!, w, p)
        cq=scratch_requirements(permute_cols!, w, q)
        @test rq.bytes>0 && ispow2(rq.alignment)
        @test cq.bytes>0 && ispow2(cq.alignment)
        work=FaerScratch(permute_rows!, w, p)
        @test scratch_capacity(work).bytes>=rq.bytes
        reserve!(work, permute_cols!, w, q)
        cap=scratch_capacity(work)
        for _ in 1:2
            permute_rows!(w, p; scratch = work)
            sub=sub[Vector(p), :]
            permute_cols!(w, q; scratch = work)
            sub=sub[:, Vector(q)]
        end
        @test scratch_capacity(work)==cap
        expected[6:-2:1, 2:2:7]=sub
        @test Matrix(a)==expected
        @test swap_rows!(w, 1, 3)===w
        @test swap_cols!(w, 2, 3)===w
        sub[[1, 3], :]=sub[[3, 1], :]
        sub[:, [2, 3]]=sub[:, [3, 2]]
        expected[6:-2:1, 2:2:7]=sub
        @test Matrix(a)==expected
        # Whole-owner permutation also consumes caller-owned scratch.
        full=FaerScratch(permute_rows!, a, FaerPermutation(collect(6:-1:1)))
        permute_rows!(a, collect(6:-1:1); scratch = full)
        expected=expected[6:-1:1, :]
        reserve!(full, permute_cols!, a, collect(7:-1:1))
        permute_cols!(a, collect(7:-1:1); scratch = full)
        expected=expected[:, 7:-1:1]
        @test Matrix(a)==expected
        # Failures happen before any destination write.
        small=FaerScratch(0)
        @test_throws ArgumentError permute_rows!(w, p; scratch = small)
        @test_throws ArgumentError permute_rows!(a, collect(1:6); scratch = small)
        @test_throws ArgumentError permute_rows!(w, p; scratch = "work")
        @test_throws DimensionMismatch permute_rows!(w, [2, 1]; scratch = work)
        @test_throws DimensionMismatch permute_cols!(w, [2, 1])
        @test_throws DimensionMismatch scratch_requirements(permute_rows!, w, [1, 2])
        @test_throws BoundsError swap_rows!(w, 0, 1)
        @test_throws BoundsError swap_cols!(w, 1, 4)
        closedp=FaerPermutation([3, 2, 1])
        close(closedp)
        @test_throws ArgumentError permute_rows!(w, closedp)
        closeds=FaerScratch(permute_rows!, w, p)
        close(closeds)
        @test_throws ArgumentError permute_rows!(w, p; scratch = closeds)
        gone=view(a, 1:2, 1:2)
        close(gone)
        @test_throws ArgumentError permute_rows!(gone, [2, 1])
        @test_throws ArgumentError swap_rows!(gone, 1, 2)
        @test_throws ArgumentError scratch_requirements(permute_rows!, gone, [2, 1])
        @test Matrix(a)==expected
        # Repeated-index views keep ordered staged writes instead of an aliased borrow.
        rep=view(a, StepRangeLen(2, 0, 3), 1:7)
        permute_rows!(rep, [3, 1, 2])
        swap_rows!(rep, 1, 2)
        @test Matrix(a)==expected
        empty=view(a, 1:0, 1:7)
        @test permute_rows!(empty, Int[])===empty
        close(a)
        @test_throws ArgumentError permute_rows!(w, p)
        @test_throws ArgumentError permute_rows!(w, p; scratch = work)
        # Two-dimensional views of an owned row.
        row=FaerRow(T.(1:6))
        rv=view(row, 1:1, 6:-1:1)
        permute_cols!(rv, [2, 1, 3, 4, 5, 6])
        @test Vector(row)==T[1, 2, 3, 4, 6, 5]
        rw=FaerScratch(permute_cols!, rv, collect(1:6))
        permute_cols!(rv, [2, 1, 3, 4, 5, 6]; scratch = rw)
        swap_cols!(rv, 1, 6)
        @test Vector(row)==T[6, 2, 3, 4, 5, 1]
        permute_cols!(row, [6, 2, 3, 4, 5, 1]; scratch = rw)
        @test Vector(row)==T[1, 2, 3, 4, 5, 6]
    end
end

@testset "Native permutation of vector views and inverse application" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        data=T.(1:7)
        p=FaerPermutation([3, 1, 4, 2])
        v=FaerVector(data)
        held=view(v, :)
        vv=view(v, 7:-2:1)
        sub=data[7:-2:1]
        @test permute!(vv, p)===vv
        sub=sub[Vector(p)]
        @test Vector(vv)==sub
        @test invpermute!(vv, p)===vv
        sub=data[7:-2:1]
        @test Vector(vv)==sub
        work=FaerScratch(permute!, vv, p)
        @test scratch_requirements(permute!, vv, p).bytes<=scratch_capacity(work).bytes
        permute!(vv, p; scratch = work)
        invpermute!(vv, p; scratch = work)
        permute!(vv, [4, 3, 2, 1]; scratch = work)
        sub=sub[[4, 3, 2, 1]]
        @test Vector(vv)==sub
        @test Vector(held)[7:-2:1]==sub
        whole=FaerScratch(permute!, v, collect(1:7))
        permute!(v, collect(7:-1:1); scratch = whole)
        invpermute!(v, collect(7:-1:1); scratch = whole)
        @test Vector(v)[7:-2:1]==sub
        @test_throws DimensionMismatch permute!(vv, [2, 1])
        @test_throws DimensionMismatch invpermute!(vv, [2, 1])
        @test_throws ArgumentError permute!(vv, p; scratch = FaerScratch(0))
        @test Vector(vv)==sub
        # A row-vector view of a matrix borrows the transposed descriptor.
        m=reshape(T.(1:12), 3, 4)
        a=FaerMatrix(m)
        rv=view(a, 2, 4:-1:1)
        permute!(rv, [2, 1, 3, 4])
        m[2, 4:-1:1]=m[2, 4:-1:1][[2, 1, 3, 4]]
        @test Matrix(a)==m
        invpermute!(rv, [2, 3, 4, 1])
        m[2, 4:-1:1]=m[2, 4:-1:1][invperm([2, 3, 4, 1])]
        @test Matrix(a)==m
        # Invalidated vector views fail before writing.
        before=Vector(v)
        resize!(v, 8)
        @test_throws InvalidStateException permute!(vv, p)
        @test_throws InvalidStateException scratch_requirements(permute!, vv, p)
        @test Vector(v)[1:7]==before
    end
end

@testset "Permutation ambiguities" begin
    @test isempty(detect_ambiguities(Colmat, Base, LinearAlgebra; recursive = true))
end
