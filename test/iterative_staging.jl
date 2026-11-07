using Colmat, Test, LinearAlgebra, SparseArrays

const STAGING_TYPES=(Float32, Float64, ComplexF32, ComplexF64)
function _staging_operator(::Type{T}, n) where {T}
    a=spdiagm(-1=>fill(T(-1), n-1), 0=>fill(T(4), n), 1=>fill(T(-1), n-1))
    return FaerLinearOperator(FaerSparseMatrix(a)), a
end

@testset "Iterative solvers do not stage native operands in Julia" begin
    n=20_000
    bytes=n*sizeof(Float64)
    op, a=_staging_operator(Float64, n)
    b=collect(range(1.0, 2.0; length = n))
    fb=FaerVector(b)
    B=FaerMatrix(hcat(b, reverse(b)))
    x=FaerVector(zeros(n))
    for (name, f) in (
        ("cg vector", ()->cg(op, fb)),
        ("cg block", ()->cg(op, B)),
        ("bicgstab", ()->bicgstab(op, fb)),
        ("lsmr", ()->lsmr(op, fb)),
        ("cg!", ()->cg!(x, op, fb)),
        ("bicgstab!", ()->bicgstab!(x, op, fb)),
        ("lsmr!", ()->lsmr!(x, op, fb)),
    )
        f()
        @test @allocated(f()) < bytes ÷ 4
    end
    # A Julia vector RHS allocates only its returned Julia solution.
    cg(op, b)
    @test @allocated(cg(op, b)) < 1.5*bytes
    @test Vector(cg(op, fb).x) ≈ a\b
    @test Matrix(cg(op, B).x) ≈ a\Matrix(B)
    @test Vector(cg!(x, op, fb).x) ≈ a\b
    @test Vector(fb) == b
    foreach(close, (fb, B, x))
end

@testset "Staging changes keep solver results and validation" begin
    for T in STAGING_TYPES
        op, a=_staging_operator(T, 64)
        b=T.(1:64)
        for rhs in (b, FaerVector(b), view(FaerVector(b), :), FaerMatrix(reshape(b, :, 1)))
            r=cg(op, rhs)
            @test r.converged
            @test vec(Array(r.x)) ≈ a\b
            @test bicgstab(op, rhs).converged
            @test lsmr(op, rhs).converged
        end
        x=FaerVector(zeros(T, 64))
        @test cg!(x, op, FaerVector(b)).x === x
        @test Vector(x) ≈ a\b
        bad=copy(b)
        bad[3]=T(NaN)
        @test_throws ArgumentError cg(op, bad)
        nb=FaerVector(b)
        nb[3]=T(Inf)
        @test_throws ArgumentError cg(op, nb)
        @test_throws ArgumentError bicgstab(op, nb)
        @test_throws ArgumentError cg!(x, op, nb)
        @test Vector(x) ≈ a\b
    end
end

@testset "Native copyto! sources are read in place and alias safely" begin
    for T in STAGING_TYPES
        a=T.(reshape(1:20, 4, 5))
        A=FaerMatrix(a)
        # Overlapping source and destination views of one owner.
        copyto!(view(A, 2:4, :), view(A, 1:3, :))
        ref=copy(a)
        ref[2:4, :]=a[1:3, :]
        @test Matrix(A) == ref
        B=FaerMatrix(zeros(T, 4, 5))
        copyto!(B, A)
        @test Matrix(B) == ref
        v=FaerVector(T.(1:5))
        w=FaerVector(zeros(T, 5))
        copyto!(w, v)
        @test Vector(w) == T.(1:5)
        copyto!(view(v, 2:5), view(v, 1:4))
        @test Vector(v) == T[1, 1, 2, 3, 4]
        # A different scalar type still converts.
        copyto!(w, FaerVector(Float64.(1:5)))
        @test Vector(w) == T.(1:5)
        foreach(close, (A, B, v, w))
    end
end
