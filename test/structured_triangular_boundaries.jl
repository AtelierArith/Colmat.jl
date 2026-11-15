using Test, LinearAlgebra, Colmat
if !isdefined(@__MODULE__, :direct_st_solve!)
    include("structured_triangular_lapack.jl")
end
@testset "Implicit unit triangles preserve unused huge strides" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), u in ('L', 'U'),
        trans in ('N', 'T', 'C')

        huge=Int(typemax(J))
        # No coefficient array is referenced for a diagonal unit band.
        for (n, kd) in ((5, 0), (1, 3))
            unused=Matrix{T}(undef, 0, 0)
            b=fill(T(NaN), n, 2)
            before=copy(b)
            r=direct_st_solve!(
                unused, b, J; band=true, u, trans, diag=('U'), n, kd, lda=huge
            )
            @test r.info==r.diagnostic==0
            @test r.guard==fill(J(-777), 2)
            @test isequal(b, before)
        end
        # Order-one unit solves are identities. Neither the empty factor
        # payload nor any RHS element may be accessed through huge LDB.
        unused=fill(T(777), 2)
        b=fill(T(NaN), 1, 2)
        before=copy(b)
        r=direct_st_solve!(unused, b, J; u, trans, diag=('U'), n=1, ldb=huge)
        @test r.info==r.diagnostic==0
        @test r.guard==fill(J(-777), 2)
        @test isequal(b, before)
        for trr in (T<:Complex ? ('N', 'C') : ('N', 'T'))
            @test direct_st_rfp_solve!(
                unused, b, J; trr, u, trans, diag=('U'), m=1, n=2, alpha=one(T), ldb=huge
            )==0
            @test isequal(b, before)
        end
        @test unused==fill(T(777), 2)
    end
end
