using Test, LinearAlgebra, Libdl, Colmat
isdefined(Main, :direct_bdsqr!)||include("bidiagonal_svd.jl")
isdefined(Main, :direct_bdsdc!)||include("bidiagonal_dc.jl")
isdefined(Main, :direct_bdsvdx!)||include("bidiagonal_selected.jl")
isdefined(Main, :direct_lasq1!)||include("bidiagonal_values.jl")
@testset "Bidiagonal ABI diagnostics and unreferenced singleton arrays" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        R=real(T)
        defaults=(uplo=('U'), n=3, ncvt=3, nru=3, ncc=2, ldvt=5, ldu=5, ldc=5)
        for (name, value, arg) in (
            (:uplo, '?', 1),
            (:n, -1, 2),
            (:ncvt, -1, 3),
            (:nru, -1, 4),
            (:ncc, -1, 5),
            (:ldvt, 2, 9),
            (:ldu, 2, 11),
            (:ldc, 2, 13),
        )
            d=fill(R(777), 3)
            e=copy(d)
            vt=fill(T(777), 5, 3)
            u=copy(vt)
            c=copy(vt)
            kw=merge(defaults, NamedTuple{(name,)}((value,)))
            r=direct_bdsqr!(d, e, vt, u, c, J; kw...)
            @test r.info==-arg&&r.diagnostic==arg
            @test r.guard==fill(J(-777), 2)
            @test all(==(R(777)), d)&&all(==(R(777)), e)
            @test all(==(T(777)), vt)&&all(==(T(777)), u)&&all(==(T(777)), c)
            @test all(==(R(777)), r.work)
        end
        huge=typemax(J)
        for sign in (1, -1)
            d=R[sign]
            e=R[777]
            u=ones(T, 3, 1)
            vt=ones(T, 3, sign==1 ? 2 : 1)
            c=ones(T, 3, 2)
            r=direct_bdsqr!(
                d,
                e,
                vt,
                u,
                c,
                J;
                n=1,
                nru=1,
                ncvt=size(vt, 2),
                ncc=2,
                ldvt=huge,
                ldu=huge,
                ldc=huge,
            )
            @test r.info==r.diagnostic==0
            @test d==R[1]
            @test all(==(one(T)), u)&&all(==(one(T)), c)
            @test vt[1, 1]==T(sign)
        end
    end
end
@testset "BDSDC and BDSVDX ABI argument ordering" begin
    for R in (Float32, Float64), J in (Int32, Int64)
        defaults=(uplo=('U'), comp=('I'), n=3, ldu=5, ldvt=5)
        for (name, value, arg) in
            ((:uplo, '?', 1), (:comp, '?', 2), (:n, -1, 3), (:ldu, 2, 7), (:ldvt, 2, 9))
            d=fill(R(777), 3)
            e=copy(d)
            u=fill(R(777), 5, 3)
            vt=copy(u)
            q=R[777]
            iq=J[-777]
            kw=merge(defaults, NamedTuple{(name,)}((value,)))
            r=direct_bdsdc!(d, e, u, vt, q, iq, J; kw...)
            @test r.info==-arg&&r.diagnostic==arg
            @test r.guard==fill(J(-777), 2)
            @test all(==(R(777)), d)&&all(==(R(777)), e)&&all(==(R(777)), u)&&all(
                ==(R(777)), vt
            )
            @test all(==(R(777)), r.work)&&all(==(J(-777)), r.iw)
            @test q==R[777]&&iq==J[-777]
        end
        defaults=(
            uplo=('U'), job=('V'), range=('A'), n=3, vl=R(0), vu=R(1), il=1, iu=3, ldz=8
        )
        cases=(
            (uplo=('?'),)=>1,
            (job=('?'),)=>2,
            (range=('?'),)=>3,
            (n=-1,)=>4,
            (range=('V'), vl=R(-1))=>7,
            (range=('V'), vu=R(0))=>8,
            (range=('I'), il=0)=>9,
            (range=('I'), iu=4)=>10,
            (ldz=5,)=>14,
        )
        for (overrides, arg) in cases
            d=fill(R(777), 3)
            e=copy(d)
            r=direct_bdsvdx!(d, e, J; merge(defaults, overrides)...)
            @test r.info==-arg&&r.diagnostic==arg
            @test r.guard==fill(J(-777), 4)
            @test all(==(R(777)), d)&&all(==(R(777)), e)
            @test all(==(R(777)), r.s)&&all(==(R(777)), r.z)&&all(==(R(777)), r.work)
            @test all(==(J(-777)), r.iw)
        end
    end
end
@testset "All bidiagonal SVD entry providers and actual LBT calls" begin
    library=BLAS.libblastrampoline
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64)

        R=real(T)
        families=T<:Complex ? ("bdsqr",) : ("bdsqr", "bdsdc", "bdsvdx", "lasq1")
        for family in families
            provider=BLAS.lbt_find_backing_library(
                prefix*family*"_", J===Int32 ? :lp64 : :ilp64
            )
            @test provider!==nothing
            @test realpath(provider.libname)==realpath(Colmat.backend_library())
        end
        d=R[3, 2]
        e=R[0, 777]
        u=ones(T, 3, 2)
        vt=ones(T, 3, 2)
        c=ones(T, 3, 2)
        r=direct_bdsqr!(d, e, vt, u, c, J; nru=1, library)
        @test r.info==r.diagnostic==0
        @test d≈R[3, 2]
        if T<:Real
            d=R[3, 2]
            e=R[0, 777]
            q=R[777]
            iq=J[-777]
            r=direct_bdsdc!(
                d, e, zeros(R, 4, 2), zeros(R, 4, 2), q, iq, J; comp=('N'), library
            )
            @test r.info==r.diagnostic==0
            @test d≈R[3, 2]
            r=direct_bdsvdx!(R[3, 2], R[0, 777], J; library)
            @test r.info==r.diagnostic==0
            @test r.ns==2&&r.s[2:3]≈R[3, 2]
            d=R[3, 2]
            r=direct_lasq1!(d, R[0, 777], J; library)
            @test r.info==r.diagnostic==0
            @test d≈R[3, 2]
        end
    end
end
