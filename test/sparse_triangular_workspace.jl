using Test, LinearAlgebra, SparseArrays, Random, Colmat

const STW_WRAPPERS=(LowerTriangular, UpperTriangular, UnitLowerTriangular, UnitUpperTriangular)
function stw_matrix(rng, T, n, half)
    a=randn(rng, T, n, n)
    a[diagind(a)].+=T(n+2)
    half===:lower && return tril(a)
    half===:upper && return triu(a)
    return a
end
# Dense wrapper of op(A) as Julia defines it, for every operation applied.
stw_ops(s, a) = ((s, a), (transpose(s), transpose(a)), (adjoint(s), adjoint(a)))

@testset "Direct in-place sparse triangular solves" begin
    rng=MersenneTwister(117)
    n=6
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        format in (:csc, :csr),
        half in (:full, :lower, :upper)

        tol=200eps(real(T))
        a=stw_matrix(rng, T, n, half)
        s=FaerSparseMatrix(sparse(a); format)
        for (sp, dense) in stw_ops(s, a), C in STW_WRAPPERS
            tri, ref=C(sp), C(Matrix(dense))
            b=randn(rng, T, n, 3)
            gold=ref\b
            work=FaerScratch(ldiv!, tri, b)
            # Native owner, in place with caller scratch.
            x=FaerMatrix(b)
            @test ldiv!(tri, x; scratch = work)===x
            @test Matrix(x)≈gold rtol=tol
            # Native vector and strided/reversed native views.
            v=FaerVector(b[:, 1])
            @test ldiv!(tri, v; scratch = work)===v
            @test Vector(v)≈gold[:, 1] rtol=tol
            big=FaerMatrix(zeros(T, 2n, 5))
            copyto!(view(big, 2:2:2n, 2:4), b)
            w=view(big, 2:2:2n, 2:4)
            @test ldiv!(tri, w; scratch = work)===w
            @test Matrix(big)[2:2:2n, 2:4]≈gold rtol=tol
            @test iszero(Matrix(big)[1:2:2n, :])
            r=FaerMatrix(b[n:-1:1, :])
            @test Matrix(ldiv!(tri, view(r, n:-1:1, :); scratch = work))≈gold rtol=tol
            # Julia memory: arrays, strided views and faerview borrows.
            j=copy(b)
            @test ldiv!(tri, j; scratch = work)===j
            @test j≈gold rtol=tol
            jv=b[:, 2]
            @test ldiv!(tri, jv)===jv
            @test jv≈gold[:, 2] rtol=tol
            parentj=zeros(T, n+1, 4)
            parentj[2:end, 2:end]=b
            @test ldiv!(tri, view(parentj, 2:n+1, 2:4))≈gold rtol=tol
            reversed=b[n:-1:1, :]
            @test ldiv!(tri, view(reversed, n:-1:1, :))≈gold rtol=tol
            borrowed=copy(b)
            @test ldiv!(tri, faerview(borrowed)) isa Colmat.FaerBorrowedArray
            @test borrowed≈gold rtol=tol
            # Three-argument form copies the RHS first and preserves it.
            y=FaerMatrix(zeros(T, n, 3))
            nb=FaerMatrix(b)
            @test ldiv!(y, tri, nb; scratch = work)===y
            @test Matrix(y)≈gold rtol=tol
            @test Matrix(nb)==b
            # Out-of-place solves share the kernel and accept scratch.
            out=\(tri, nb; scratch = work)
            @test out isa FaerMatrix{T}
            @test Matrix(out)≈gold rtol=tol
            @test Matrix(nb)==b
            close(work)
        end
        @test Matrix(s)==a
    end
end

@testset "Sparse triangular scratch requirements and resources" begin
    T=Float64
    rng=MersenneTwister(5)
    lower=FaerSparseMatrix(sparse(stw_matrix(rng, T, 5, :lower)))
    full=FaerSparseMatrix(sparse(stw_matrix(rng, T, 5, :full)); format = :csr)
    b=FaerMatrix(randn(rng, T, 5, 2))
    # Stored layout already matches faer's kernel: nothing to stage.
    @test scratch_requirements(ldiv!, LowerTriangular(lower), b).bytes==0
    @test scratch_requirements(ldiv!, UpperTriangular(transpose(lower)), b).bytes==0
    @test scratch_requirements(ldiv!, UnitUpperTriangular(adjoint(lower)), b).bytes==0
    @test scratch_requirements(\, LowerTriangular(lower), b).bytes==0
    # Both halves stored, or the other half requested: rebuild in scratch.
    req=scratch_requirements(ldiv!, UpperTriangular(lower), b)
    @test req.bytes>0
    @test scratch_requirements(ldiv!, LowerTriangular(full), b).bytes>0
    # Width independent; mixed RHS stages promoted values only.
    @test scratch_requirements(ldiv!, UpperTriangular(lower), FaerMatrix(zeros(5, 9)))==req
    creq=scratch_requirements(ldiv!, LowerTriangular(lower), FaerMatrix(zeros(ComplexF64, 5, 1)))
    @test 0<creq.bytes
    y=FaerMatrix(zeros(5, 2))
    @test scratch_requirements(ldiv!, y, UpperTriangular(lower), b)==req
    @test_throws DimensionMismatch scratch_requirements(
        ldiv!,
        FaerMatrix(zeros(5, 3)),
        UpperTriangular(lower),
        b,
    )
    work=FaerScratch(ldiv!, UpperTriangular(lower), b)
    @test scratch_capacity(work).bytes>=req.bytes
    small=FaerScratch(0)
    snapshot=Matrix(b)
    @test_throws ArgumentError ldiv!(UpperTriangular(lower), b; scratch = small)
    @test Matrix(b)==snapshot
    @test reserve!(small, ldiv!, UpperTriangular(lower), b)===small
    @test ldiv!(UpperTriangular(lower), b; scratch = small)===b
    @test Matrix(b)≈UpperTriangular(Matrix(lower))\snapshot
    copyto!(b, snapshot)
    @test FaerScratch(\, UpperTriangular(lower), b) isa FaerScratch
    @test reserve!(FaerScratch(0), ldiv!, y, UpperTriangular(lower), b) isa FaerScratch
    @test FaerScratch(ldiv!, y, UpperTriangular(lower), b) isa FaerScratch
    # Teams are validated; faer's sparse triangular kernels are sequential.
    team=FaerTeam(2)
    reserve!(work, ldiv!, LowerTriangular(full), b; team)
    @test Matrix(ldiv!(LowerTriangular(full), copy(b); team, scratch = work)) ≈
          LowerTriangular(Matrix(full))\snapshot
    close(team)
    @test_throws ArgumentError ldiv!(LowerTriangular(full), b; team)
    @test_throws ArgumentError ldiv!(LowerTriangular(full), b; team = 1)
    @test_throws ArgumentError ldiv!(LowerTriangular(full), b; scratch = 1)
    close(work)
    @test_throws ArgumentError ldiv!(UpperTriangular(lower), b; scratch = work)
    @test Matrix(b)==snapshot
    # Closed owners and views fail before any write.
    closedrhs=FaerMatrix(zeros(5, 2))
    close(closedrhs)
    @test_throws ArgumentError ldiv!(LowerTriangular(lower), closedrhs)
    v=view(b, :, 1:1)
    close(v)
    @test_throws ArgumentError ldiv!(LowerTriangular(lower), v)
    gone=copy(lower)
    close(gone)
    @test_throws ArgumentError ldiv!(LowerTriangular(gone), b)
    @test_throws ArgumentError ldiv!(y, LowerTriangular(transpose(gone)), b)
    @test Matrix(b)==snapshot
    @test iszero(Matrix(y))
end

@testset "Sparse triangular failures, aliasing and mixed types" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), format in (:csc, :csr)
        a=T[2 0 0; 1 0 0; 4 5 3]
        s=FaerSparseMatrix(sparse(a); format)
        b=FaerMatrix(T[1 2; 3 4; 5 6])
        y=FaerMatrix(fill(T(7), 3, 2))
        j=T[1, 2, 3]
        for C in (LowerTriangular, UpperTriangular)
            for tri in (C(s), C(transpose(s)), C(adjoint(s)))
                e=try
                    ldiv!(tri, b)
                    nothing
                catch err
                    err
                end
                @test e isa SingularException && e.info==2
                @test_throws SingularException ldiv!(y, tri, b)
                @test_throws SingularException ldiv!(tri, j)
                @test Matrix(b)==T[1 2; 3 4; 5 6]
                @test Matrix(y)==fill(T(7), 3, 2)
                @test j==T[1, 2, 3]
            end
        end
        # Unit wrappers never read the (zero) diagonal.
        @test Matrix(ldiv!(UnitLowerTriangular(s), copy(b))) ≈
              UnitLowerTriangular(a)\T[1 2; 3 4; 5 6]
        # Overlapping views of one owner: the RHS is read before the solve.
        g=FaerSparseMatrix(sparse(T[2 0 0; 1 3 0; 4 5 6]); format)
        m=FaerMatrix(T[1 2 3; 4 5 6; 7 8 9])
        rhs=Matrix(m)[:, 2:3]
        ldiv!(view(m, :, 1:2), LowerTriangular(g), view(m, :, 2:3))
        @test Matrix(m)[:, 1:2]≈LowerTriangular(T[2 0 0; 1 3 0; 4 5 6])\rhs
        @test Matrix(m)[:, 3]==T[3, 6, 9]
        @test_throws DimensionMismatch ldiv!(LowerTriangular(g), zeros(T, 2, 2))
        @test_throws DimensionMismatch ldiv!(zeros(T, 3, 1), LowerTriangular(g), zeros(T, 3, 2))
    end
    # Promoted RHS: real sparse, complex destination solves directly.
    g=FaerSparseMatrix(sparse([2.0 0 0; 1 3 0; 4 5 6]))
    ref=LowerTriangular([2.0 0 0; 1 3 0; 4 5 6])
    cb=ComplexF64[1+im 2; 3 4im; 5 6]
    x=FaerMatrix(cb)
    @test ldiv!(LowerTriangular(g), x)===x
    @test Matrix(x)≈ref\cb
    f32=FaerSparseMatrix(sparse(Float32[2 0 0; 1 3 0; 4 5 6]))
    xv=[1.0, 2.0, 3.0]
    @test ldiv!(LowerTriangular(f32), xv)===xv
    @test xv≈ref\[1.0, 2.0, 3.0]
    # Narrower destinations stage one promoted copy and convert on writeback.
    narrow=Float32[1, 2, 3]
    @test ldiv!(LowerTriangular(g), narrow)===narrow
    @test narrow≈Float32.(ref\[1.0, 2.0, 3.0])
    nested=FaerMatrix(Float32[1 2; 3 4; 5 6])
    @test ldiv!(LowerTriangular(g), nested)===nested
    @test Matrix(nested)≈Float32.(ref\[1.0 2; 3 4; 5 6])
    # Non-strided Julia destinations take the staged path too.
    q=zeros(6)
    idx=[1, 3, 5]
    q[idx]=[1.0, 2.0, 3.0]
    @test ldiv!(LowerTriangular(g), view(q, idx)) == ref\[1.0, 2.0, 3.0]
    @test q[idx]≈ref\[1.0, 2.0, 3.0]
    @test isempty(
        detect_ambiguities(Colmat, Base, LinearAlgebra, SparseArrays; recursive = true),
    )
end

@testset "Sparse triangular in-place allocation bound" begin
    function stw_bytes(n, half)
        rng=MersenneTwister(n)
        a=sprandn(rng, n, n, min(1.0, 4/n))+(n+2)*I
        s=FaerSparseMatrix(half===:lower ? tril(a) : a)
        tri=LowerTriangular(s)
        x=FaerMatrix(randn(rng, n, 4))
        jv=randn(rng, n)
        work=FaerScratch(ldiv!, tri, x)
        ldiv!(tri, x; scratch = work)
        ldiv!(tri, jv; scratch = work)
        native=@allocated ldiv!(tri, x; scratch = work)
        julia=@allocated ldiv!(tri, jv; scratch = work)
        close(work)
        return native, julia
    end
    for half in (:lower, :full)
        small=stw_bytes(8, half)
        large=stw_bytes(4000, half)
        # Bookkeeping only (lock, view descriptor, requirement and capacity
        # queries, FFI argument boxing): no RHS, result or sparse operand copy,
        # so the cost does not grow with n beyond boxing integers above 512.
        @test all(large .<= small .+ 512)
        @test all(<(8192), large)
        @test all(<(4000*8), large)
    end
end
