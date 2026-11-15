using Test, LinearAlgebra, Libdl, Colmat
isdefined(Main, :direct_bdsqr!)||include("bidiagonal_svd.jl")
@testset "BDSQR retained factors and positive INFO count on range failure" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        library in (Colmat.backend_library(), BLAS.libblastrampoline)

        R=real(T)
        large=floatmax(R)
        ds=R[777, large, large, large, 777]
        es=R[777, large, large, 777]
        d=view(ds, 2:4)
        e=view(es, 2:3)
        u=fill(T(777), 5, 3)
        vt=copy(u)
        c=fill(T(777), 5, 2)
        r=direct_bdsqr!(d, e, vt, u, c, J; nru=3, library)
        @test r.info==count(!iszero, e)==2
        @test r.diagnostic==0
        @test r.guard==fill(J(-777), 2)
        @test ds==R[777, large, large, large, 777]&&es==R[777, large, large, 777]
        @test all(==(T(777)), u)&&all(==(T(777)), vt)&&all(==(T(777)), c)
        @test all(==(R(777)), r.work)
    end
end
@testset "BDSQR normal and subnormal scaling with independent reconstruction" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), n in (2, 7),
        uplo in ('U', 'L')

        R=real(T)
        tol=3000eps(R)
        scales=R===Float32 ? R[1e-30, 1e30, 1e-40] : R[1e-300, 1e300, 1e-320]
        for scale in scales,
            (library, K) in
            ((Colmat.backend_library(), J), (bidiagonal_svd_reference(), Int64))

            initial_d=R[isodd(i) ? -1-i/10 : 1+i/10 for i in 1:n]*scale
            initial_e=fill(R(0.125)*scale, n-1)
            original=Matrix(Bidiagonal(initial_d, initial_e, uplo))
            # Reference-LAPACK v3.12.1 SRC/dbdsqr.f, THRESH (lines 416/421),
            # uses an absolute safe-minimum floor. Normalize the reference
            # oracle and rescale its singular values, preserving the same
            # mathematical bidiagonal. Native receives the original input.
            factor_scale=library==Colmat.backend_library() ? one(R) : scale
            d=initial_d ./ factor_scale
            e=initial_e ./ factor_scale
            u=fill(T(777), n+2, n)
            vt=copy(u)
            c=fill(T(777), n+2, 2)
            u[1:n, :]=Matrix{T}(I, n, n)
            vt[1:n, :]=Matrix{T}(I, n, n)
            c[1:n, :].=one(T)
            r=direct_bdsqr!(d, e, vt, u, c, K; uplo, library)
            @test r.info==r.diagnostic==0
            d .*= factor_scale
            left=u[1:n, :]
            rightt=vt[1:n, :]
            atol=max(tol*norm(original), R(20*n)*nextfloat(zero(R)))
            @test left*Diagonal(d)*rightt≈original atol=atol rtol=tol
            @test left'*left≈Matrix{T}(I, n, n) atol=tol rtol=tol
            @test rightt*rightt'≈Matrix{T}(I, n, n) atol=tol rtol=tol
            @test all(>=(0), d)&&issorted(d; rev=true)
            @test r.guard==fill(K(-777), 2)
            @test r.work[[1, end]]==fill(R(777), 2)
            @test all(==(T(777)), u[(n + 1):end, :])&&all(==(T(777)), vt[(n + 1):end, :])&&all(
                ==(T(777)), c[(n + 1):end, :]
            )
        end
    end
end
