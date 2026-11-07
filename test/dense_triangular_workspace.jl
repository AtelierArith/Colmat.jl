using Test, LinearAlgebra, Random, Colmat

const DTW_WRAPPERS=(LowerTriangular, UpperTriangular, UnitLowerTriangular, UnitUpperTriangular)
function dtw_matrix(rng, T, n)
    a=randn(rng, T, n, n)./n
    a[diagind(a)].+=T(2)
    return a
end
# Native parents of the triangle with the Julia matrix they represent.
function dtw_parents(a)
    big=FaerMatrix(zeros(eltype(a), size(a, 1)+1, size(a, 2)+2))
    copyto!(view(big, 2:size(a, 1)+1, 3:size(a, 2)+2), a)
    s=FaerMatrix(a)
    return (
        (s, a),
        (transpose(s), transpose(a)),
        (adjoint(s), adjoint(a)),
        (view(big, 2:size(a, 1)+1, 3:size(a, 2)+2), a),
        (transpose(FaerMatrix(a)), transpose(a)),
        # faerview borrows of Julia memory, also strided and adjoint.
        (faerview(copy(a)), a),
        (adjoint(faerview(copy(adjoint(a)))), a),
        (view(faerview(dtw_padded(a)), 2:size(a, 1)+1, 3:size(a, 2)+2), a),
    )
end
function dtw_padded(a)
    big=zeros(eltype(a), size(a, 1)+1, size(a, 2)+2)
    big[2:end, 3:end]=a
    return big
end

@testset "Direct in-place dense triangular solves" begin
    rng=MersenneTwister(111)
    n=6
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        tol=200eps(real(T))
        a=dtw_matrix(rng, T, n)
        for (p, dense) in dtw_parents(a), C in DTW_WRAPPERS
            tri, ref=C(p), C(Matrix(dense))
            b=randn(rng, T, n, 3)
            gold=ref\b
            @test scratch_requirements(ldiv!, tri, b).bytes==0
            # Native owner, in place.
            x=FaerMatrix(b)
            @test ldiv!(tri, x)===x
            @test Matrix(x)≈gold rtol=tol
            # Native vector and strided/reversed native views.
            v=FaerVector(b[:, 1])
            @test ldiv!(tri, v)===v
            @test Vector(v)≈gold[:, 1] rtol=tol
            big=FaerMatrix(zeros(T, 2n, 5))
            copyto!(view(big, 2:2:2n, 2:4), b)
            w=view(big, 2:2:2n, 2:4)
            @test ldiv!(tri, w)===w
            @test Matrix(big)[2:2:2n, 2:4]≈gold rtol=tol
            @test iszero(Matrix(big)[1:2:2n, :])
            r=FaerMatrix(b[n:-1:1, :])
            @test Matrix(ldiv!(tri, view(r, n:-1:1, :)))≈gold rtol=tol
            # Julia memory: arrays, strided views and faerview borrows.
            j=copy(b)
            @test ldiv!(tri, j)===j
            @test j≈gold rtol=tol
            jv=b[:, 2]
            @test ldiv!(tri, jv)===jv
            @test jv≈gold[:, 2] rtol=tol
            reversed=b[n:-1:1, :]
            @test ldiv!(tri, view(reversed, n:-1:1, :))≈gold rtol=tol
            borrowed=copy(b)
            @test ldiv!(tri, faerview(borrowed)) isa Colmat.FaerBorrowedArray
            @test borrowed≈gold rtol=tol
            # Three-argument form copies the RHS first and preserves it.
            y=FaerMatrix(zeros(T, n, 3))
            nb=FaerMatrix(b)
            @test ldiv!(y, tri, nb)===y
            @test Matrix(y)≈gold rtol=tol
            @test Matrix(nb)==b
            # Out-of-place solves.
            out=tri\nb
            @test out isa FaerMatrix{T}
            @test Matrix(out)≈gold rtol=tol
            @test tri\b[:, 1]≈gold[:, 1] rtol=tol
            @test Vector(tri\FaerVector(b[:, 1]))≈gold[:, 1] rtol=tol
            # Right solves X op(A) = B.
            c=randn(rng, T, 3, n)
            rgold=c/ref
            xr=FaerMatrix(c)
            @test rdiv!(xr, tri)===xr
            @test Matrix(xr)≈rgold rtol=tol
            jr=copy(c)
            @test rdiv!(jr, tri)===jr
            @test jr≈rgold rtol=tol
            bigr=FaerMatrix(zeros(T, 5, 2n))
            copyto!(view(bigr, 2:4, 2:2:2n), c)
            @test Matrix(rdiv!(view(bigr, 2:4, 2:2:2n), tri))≈rgold rtol=tol
            @test Matrix(FaerMatrix(c)/tri)≈rgold rtol=tol
            @test Matrix(c/tri)≈rgold rtol=tol
        end
        @test Matrix(FaerMatrix(a))==a
    end
end

@testset "Mixed-scalar dense triangular solves" begin
    rng=MersenneTwister(7)
    n=5
    for (TA, TB) in (
        (Float32, Float64),
        (Float64, ComplexF64),
        (Float32, ComplexF32),
        (ComplexF32, ComplexF64),
        (Float64, Int),
    )
        a=dtw_matrix(rng, TA, n)
        b=TB<:Integer ? rand(rng, -3:3, n, 2) : randn(rng, TB, n, 2)
        P=promote_type(TA, TB<:Integer ? Float64 : TB)
        for C in DTW_WRAPPERS, (p, dense) in dtw_parents(a)[1:3]
            tri, ref=C(p), C(Matrix{P}(dense))
            gold=ref\Matrix{P}(b)
            tol=500eps(real(P))*n
            req=scratch_requirements(ldiv!, tri, b)
            TA===P || @test req.bytes>=n*n*sizeof(P)
            TA===P && @test req.bytes==0
            work=FaerScratch(ldiv!, tri, b)
            out=tri\b
            @test out isa FaerMatrix{P}
            @test Matrix(out)≈gold rtol=tol
            if TB===P
                x=FaerMatrix(b)
                @test ldiv!(tri, x; scratch = work)===x
                @test Matrix(x)≈gold rtol=tol
                j=copy(b)
                @test ldiv!(tri, j; scratch = work)===j
                @test j≈gold rtol=tol
                r=copy(transpose(b))
                @test rdiv!(r, transpose(tri); scratch = work)===r
                @test r≈transpose(gold) rtol=tol
            else
                # Narrower destinations stage one promoted copy.
                y=zeros(P, n, 2)
                @test ldiv!(y, tri, b)===y
                @test y≈gold rtol=tol
            end
            close(work)
        end
    end
    # A real destination for a real triangle and a narrower RHS type.
    a=dtw_matrix(rng, Float64, 4)
    x=Float32.(randn(rng, 4, 2))
    gold=LowerTriangular(a)\Float64.(x)
    y=copy(x)
    @test ldiv!(LowerTriangular(FaerMatrix(a)), y)===y
    @test y≈Float32.(gold) rtol=1e-5
end

@testset "Dense triangular resources, aliasing and failures" begin
    rng=MersenneTwister(3)
    n=5
    a=dtw_matrix(rng, Float64, n)
    A=FaerMatrix(a)
    tri=LowerTriangular(A)
    b=FaerMatrix(randn(rng, n, 2))
    snapshot=Matrix(b)
    # Promotion scratch: query, create, reserve, and capacity checks.
    cb=FaerMatrix(randn(rng, ComplexF64, n, 2))
    csnap=Matrix(cb)
    req=scratch_requirements(ldiv!, tri, cb)
    @test req.bytes>=n*n*sizeof(ComplexF64)
    @test scratch_requirements(\, tri, cb)==req
    @test scratch_requirements(rdiv!, FaerMatrix(zeros(ComplexF64, 2, n)), tri)==req
    @test scratch_requirements(ldiv!, FaerMatrix(zeros(ComplexF64, n, 2)), tri, cb)==req
    @test_throws DimensionMismatch scratch_requirements(
        ldiv!,
        FaerMatrix(zeros(n, 3)),
        tri,
        cb,
    )
    small=FaerScratch(0)
    @test_throws ArgumentError ldiv!(tri, cb; scratch = small)
    @test Matrix(cb)==csnap
    @test reserve!(small, ldiv!, tri, cb)===small
    @test ldiv!(tri, cb; scratch = small)===cb
    @test Matrix(cb)≈LowerTriangular(a)\csnap
    @test FaerScratch(\, tri, cb) isa FaerScratch
    @test FaerScratch(rdiv!, FaerMatrix(zeros(ComplexF64, 2, n)), tri) isa FaerScratch
    @test reserve!(FaerScratch(0), rdiv!, FaerMatrix(zeros(ComplexF64, 2, n)), tri) isa
          FaerScratch
    close(small)
    @test_throws ArgumentError ldiv!(tri, cb; scratch = small)
    @test_throws ArgumentError ldiv!(tri, b; scratch = 1)
    # Teams run faer's parallel kernels on the same destination.
    team=FaerTeam(2)
    x=copy(b)
    @test ldiv!(tri, x; team)===x
    @test Matrix(x)≈LowerTriangular(a)\snapshot
    xr=FaerMatrix(copy(transpose(snapshot)))
    @test Matrix(rdiv!(xr, UpperTriangular(A); team))≈transpose(snapshot)/UpperTriangular(a)
    close(team)
    @test_throws ArgumentError ldiv!(tri, b; team)
    @test_throws ArgumentError ldiv!(tri, b; team = 1)
    @test Matrix(b)==snapshot
    # Singular triangles throw before any write, also for Y !== B.
    s=copy(a)
    s[3, 3]=0
    S=LowerTriangular(FaerMatrix(s))
    @test_throws SingularException(3) ldiv!(S, b)
    @test Matrix(b)==snapshot
    y=FaerMatrix(fill(7.0, n, 2))
    @test_throws SingularException(3) ldiv!(y, S, b)
    @test Matrix(y)==fill(7.0, n, 2)
    j=copy(snapshot)
    @test_throws SingularException(3) rdiv!(copy(transpose(j)), S)
    @test_throws SingularException(3) S\b
    # Unit triangles ignore the diagonal and the other half.
    U=FaerMatrix(tril(a, -1))
    for i in 1:n, k in i:n
        U[i, k]=i==k ? 0.0 : NaN
    end
    @test Matrix(UnitLowerTriangular(U)\b)≈UnitLowerTriangular(a)\snapshot
    # Row vectors keep LinearAlgebra's wrapped-vector result.
    r=randn(rng, n)
    @test transpose(r)/tri isa Transpose
    @test transpose(r)/tri≈transpose(r)/LowerTriangular(a)
    @test r'/UpperTriangular(A)≈r'/UpperTriangular(a)
    # A destination sharing the triangle's storage is solved from a snapshot.
    for (dest, ref) in (
        (A, () -> LowerTriangular(a)\a),
        (view(A, :, 2:3), () -> LowerTriangular(a)\a[:, 2:3]),
    )
        gold=ref()
        @test ldiv!(LowerTriangular(A), dest)===dest
        @test Matrix(dest)≈gold
        copyto!(A, a)
    end
    @test Matrix(rdiv!(A, UpperTriangular(A)))≈a/UpperTriangular(a)
    copyto!(A, a)
    # A faerview parent sharing Julia memory with the destination, too.
    J=copy(a)
    @test ldiv!(LowerTriangular(faerview(J)), J)===J
    @test J≈LowerTriangular(a)\a
    J=copy(a)
    @test ldiv!(LowerTriangular(faerview(J)), view(J, :, 2:3))==view(J, :, 2:3)
    @test J[:, 2:3]≈LowerTriangular(a)\a[:, 2:3]
    J=copy(a)
    @test rdiv!(faerview(J), UpperTriangular(faerview(J))) isa Colmat.FaerBorrowedArray
    @test J≈a/UpperTriangular(a)
    # A plain Julia parent stays on LinearAlgebra's (BLAS) path.
    @test !hasmethod(
        scratch_requirements,
        Tuple{typeof(ldiv!),LowerTriangular{Float64,Matrix{Float64}},Matrix{Float64}},
    )
    # Closed owners and views fail before any write.
    closedrhs=FaerMatrix(zeros(n, 2))
    close(closedrhs)
    @test_throws ArgumentError ldiv!(tri, closedrhs)
    v=view(b, :, 1:1)
    close(v)
    @test_throws ArgumentError ldiv!(tri, v)
    gone=copy(A)
    close(gone)
    @test_throws ArgumentError ldiv!(LowerTriangular(gone), b)
    @test_throws ArgumentError ldiv!(y, UpperTriangular(transpose(gone)), b)
    @test_throws DimensionMismatch ldiv!(tri, FaerMatrix(zeros(n+1, 1)))
    @test_throws DimensionMismatch rdiv!(FaerMatrix(zeros(1, n+1)), tri)
    @test Matrix(b)==snapshot
    @test isempty(Test.detect_ambiguities(Colmat; recursive = true))
end

@testset "Destructive dense triangular inverse" begin
    rng=MersenneTwister(44)
    n=6
    for T in (Float32, Float64, ComplexF32, ComplexF64), C in DTW_WRAPPERS
        tol=200eps(real(T))
        a=dtw_matrix(rng, T, n)
        gold=Matrix(inv(C(copy(a))))
        A=FaerMatrix(a)
        work=FaerScratch(LinearAlgebra.inv!, C(A))
        @test scratch_requirements(LinearAlgebra.inv!, C(A)).bytes>=n*n*sizeof(T)
        tri=C(A)
        @test LinearAlgebra.inv!(tri; scratch = work)===tri
        @test Matrix(tri)≈gold rtol=tol
        # The opposite strict triangle is untouched.
        lower=C in (LowerTriangular, UnitLowerTriangular)
        @test all(Matrix(A)[i, j]==a[i, j] for i in 1:n, j in 1:n if i!=j && (i>j)!=lower)
        # Views and transposed parents invert in place too.
        big=FaerMatrix(zeros(T, n+2, n+2))
        copyto!(view(big, 2:n+1, 2:n+1), a)
        tv=C(view(big, 2:n+1, 2:n+1))
        @test Matrix(LinearAlgebra.inv!(tv))≈gold rtol=tol
        At=FaerMatrix(copy(transpose(a)))
        tt=C(transpose(At))
        @test Matrix(LinearAlgebra.inv!(tt))≈gold rtol=tol
        # Adjoint and elementwise-conjugate parents invert the stored triangle
        # unconjugated: inv(Xᴴ) = inv(X)ᴴ, inv(conj(X)) = conj(inv(X)).
        Aa=FaerMatrix(copy(adjoint(a)))
        ta=C(adjoint(Aa))
        @test scratch_requirements(LinearAlgebra.inv!, ta).bytes>=n*n*sizeof(T)
        @test LinearAlgebra.inv!(ta; scratch = work)===ta
        @test Matrix(ta)≈gold rtol=tol
        @test all(
            Matrix(Aa)'[i, j]==a[i, j] for i in 1:n, j in 1:n if i!=j && (i>j)!=lower
        )
        tc=C(transpose(adjoint(FaerMatrix(conj(a)))))
        @test Matrix(LinearAlgebra.inv!(tc))≈gold rtol=tol
        tr=C(adjoint(transpose(FaerMatrix(conj(a)))))
        @test Matrix(LinearAlgebra.inv!(tr))≈gold rtol=tol
        # faerview parents invert the Julia memory in place.
        J=copy(a)
        tb=C(faerview(J))
        @test scratch_requirements(LinearAlgebra.inv!, tb).bytes>=n*n*sizeof(T)
        @test LinearAlgebra.inv!(tb; scratch = work)===tb
        @test C(J)≈gold rtol=tol
        @test all(J[i, j]==a[i, j] for i in 1:n, j in 1:n if i!=j && (i>j)!=lower)
        Ja=copy(adjoint(a))
        @test Matrix(LinearAlgebra.inv!(C(adjoint(faerview(Ja)))))≈gold rtol=tol
        close(work)
    end
    # A read-only borrow (an NPY borrow of a byte buffer) is solved with but
    # never written by inv!.
    a=dtw_matrix(rng, Float64, 3)
    header="{'descr': '<f8', 'fortran_order': True, 'shape': (3, 3), }"
    header*=" "^(63-mod(10+length(header), 64))*"\n"
    bytes=vcat(
        UInt8[0x93],
        Vector{UInt8}("NUMPY"),
        UInt8[1, 0],
        reinterpret(UInt8, [htol(UInt16(length(header)))]),
        Vector{UInt8}(header),
        reinterpret(UInt8, htol.(vec(a))),
    )
    R=npyborrow(bytes)
    @test R==a
    @test_throws ArgumentError LinearAlgebra.inv!(UpperTriangular(R))
    @test_throws ArgumentError LinearAlgebra.inv!(LowerTriangular(R'))
    @test R==a
    rb=ones(3, 2)
    @test ldiv!(UpperTriangular(R), rb)===rb
    @test rb≈UpperTriangular(a)\ones(3, 2)
    s=Matrix(Diagonal([1.0, 0, 2]))
    S=FaerMatrix(s)
    @test_throws SingularException(2) LinearAlgebra.inv!(UpperTriangular(S))
    @test Matrix(S)==s
    small=FaerScratch(0)
    @test_throws ArgumentError LinearAlgebra.inv!(
        LowerTriangular(FaerMatrix(rand(3, 3)+3I));
        scratch = small,
    )
    team=FaerTeam(2)
    a=dtw_matrix(rng, Float64, 4)
    A=FaerMatrix(a)
    @test reserve!(small, LinearAlgebra.inv!, UpperTriangular(A); team)===small
    @test Matrix(LinearAlgebra.inv!(UpperTriangular(A); scratch = small, team)) ≈
          inv(UpperTriangular(a))
    close(team)
    close(small)
end
