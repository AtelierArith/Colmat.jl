using Test, LinearAlgebra, Libdl, Colmat
isdefined(@__MODULE__, :direct_ordinary_trsen!) || include("ordinary_schur_selection.jl")
isdefined(@__MODULE__, :direct_ordinary_geesx!) || include("ordinary_schur_expert.jl")
@testset "Ordinary Schur headers use LAPACK argument numbers" begin
    lib=Libdl.dlopen(
        only(
            filter(
                l->l.interface==:ilp64&&occursin(
                    "openblas", lowercase(basename(l.libname))
                ),
                BLAS.get_config().loaded_libs,
            ),
        ).libname,
    )
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64),
        n in (0, 1, 5)

        initial=ordinary_schur_fixture(T, n, :pairs)
        t=vcat(initial, fill(T(777), 3, n))
        old=copy(t)
        q=fill(T(777), n+3, n)
        oldq=copy(q)
        cases=[
            (; comp=('?'), expected=-1),
            (; n=-1, expected=-2),
            (; ldt=max(0, n-1), expected=-4),
            (; ldq=0, expected=-6),
        ]
        if n>0
            push!(cases, (; first=0, expected=-7))
            push!(cases, (; last=n+1, expected=-8))
        end
        for case in cases
            expected=case.expected
            options=Base.structdiff(case, (; expected))
            kwargs=merge((; first=1, last=1), options)
            tr=copy(t)
            qr=copy(q)
            result=direct_ordinary_trexc!(t, q, J; kwargs...)
            oracle=direct_ordinary_trexc!(
                tr,
                qr,
                Int64;
                kwargs...,
                pointer=Libdl.dlsym(lib, Symbol(prefix*"trexc_64_")),
            )
            @test result.info==oracle.info==expected&&result.diagnostic==-expected
            @test isequal(t, old)&&isequal(tr, old)&&isequal(q, oldq)&&isequal(qr, oldq)
        end
        for job in ('N', 'E', 'V', 'B')
            flags=J.(iseven.(1:n))
            mask=vcat(J(-777), flags, J(-777))
            cases=[
                (; job=('?'), expected=-1),
                (; comp=('?'), expected=-2),
                (; n=-1, expected=-4),
                (; ldt=max(0, n-1), expected=-6),
                (; ldq=0, expected=-8),
            ]
            for case in cases
                expected=case.expected
                options=Base.structdiff(case, (; expected))
                kwargs=merge((; job, lwork=-1, liwork=-9), options)
                tr=copy(t)
                qr=copy(q)
                mr=Int64.(mask)
                oldmask=copy(mask)
                result=direct_ordinary_trsen!(t, q, mask, J; kwargs...)
                oracle=direct_ordinary_trsen!(
                    tr,
                    qr,
                    mr,
                    Int64;
                    kwargs...,
                    pointer=Libdl.dlsym(lib, Symbol(prefix*"trsen_64_")),
                )
                @test result.info==oracle.info==expected&&result.diagnostic==-expected
                @test isequal(t, old)&&isequal(tr, old)&&isequal(q, oldq)&&isequal(qr, oldq)
                @test mask==mr==oldmask
            end
        end
        for sense in ('N', 'E', 'V', 'B')
            minimum=max(1, (T<:Real ? 3 : 2)*n)
            cases=[
                (; job=('?'), expected=-1),
                (; sort=('?'), expected=-2),
                (; sense=('?'), expected=-4),
                (; sort=('N'), sense=('E'), expected=-4),
                (; n=-1, expected=-5),
                (; lda=max(0, n-1), expected=-7),
                (; ldvs=0, expected=T<:Real ? -12 : -11),
                (; lwork=minimum-1, expected=T<:Real ? -16 : -15),
            ]
            if T<:Real
                push!(cases, (; liwork=0, expected=-18))
            end
            for case in cases
                expected=case.expected
                options=Base.structdiff(case, (; expected))
                kwargs=merge((; sense), options)
                tr=copy(t)
                qr=copy(q)
                result=direct_ordinary_geesx!(t, q, J; kwargs...)
                oracle=direct_ordinary_geesx!(
                    tr,
                    qr,
                    Int64;
                    kwargs...,
                    pointer=Libdl.dlsym(lib, Symbol(prefix*"geesx_64_")),
                )
                @test result.info==oracle.info==expected&&result.diagnostic==-expected
                @test result.calls==oracle.calls==0
                @test isequal(t, old)&&isequal(tr, old)&&isequal(q, oldq)&&isequal(qr, oldq)
            end
        end
    end
end
@testset "TRSEN rejection retains M and the last successful selected move" begin
    lib=Libdl.dlopen(
        only(
            filter(
                l->l.interface==:ilp64&&occursin(
                    "openblas", lowercase(basename(l.libname))
                ),
                BLAS.get_config().loaded_libs,
            ),
        ).libname,
    )
    for (T, prefix) in ((Float32, "s"), (Float64, "d")),
        J in (Int32, Int64), comp in ('N', 'V'), partial in (false, true),
        job in ('N', 'E', 'V', 'B')

        initial=partial ? ordinary_partial_rejection(T) : ordinary_rejected_window(T)
        n=size(initial, 2)
        flags=partial ? J[0, 0, 1, 1, 1] : J[0, 0, 1, 1]
        count=partial ? 3 : 2
        lw, li=ordinary_trsen_minimum(T, job, n, count)
        t=vcat(initial, fill(T(777), 3, n))
        tr=copy(t)
        expected=copy(t)
        expected_ref=copy(t)
        q=comp=='V' ? vcat(Matrix{T}(I, n, n), fill(T(777), 3, n)) : fill(T(777), 1, 1)
        qr=copy(q)
        expected_q=copy(q)
        expected_qr=copy(q)
        if partial
            first_move=direct_ordinary_trexc!(
                expected, expected_q, J; first=3, last=1, comp
            )
            reference_move=direct_ordinary_trexc!(
                expected_ref,
                expected_qr,
                Int64;
                first=3,
                last=1,
                comp,
                pointer=Libdl.dlsym(lib, Symbol(prefix*"trexc_64_")),
            )
            @test first_move.info==first_move.diagnostic==reference_move.info==0
        end
        if job!='N'
            expected=copy(expected_ref)
            expected_q=copy(expected_qr)
        end
        mask=vcat(J(-777), flags, J(-777))
        mr=Int64.(mask)
        oldmask=copy(mask)
        result=direct_ordinary_trsen!(t, q, mask, J; job, comp, lwork=lw, liwork=li)
        oracle=direct_ordinary_trsen!(
            tr,
            qr,
            mr,
            Int64;
            job,
            comp,
            lwork=lw,
            liwork=li,
            pointer=Libdl.dlsym(lib, Symbol(prefix*"trsen_64_")),
        )
        @test result.info==oracle.info==1&&result.diagnostic==0
        @test result.count==oracle.count==count
        @test mask==mr==oldmask
        @test isequal(t, expected)&&isequal(q, expected_q)
        @test isequal(tr, expected_ref)&&isequal(qr, expected_qr)
        @test result.values≈ordinary_schur_values(expected)
        @test result.condition==oracle.condition&&result.sep==oracle.sep
    end
end
const ordinary_nested_geesx_runs=Ref(0)
const ordinary_nested_geesx_ok=Ref(true)
function ordinary_nested_geesx_select(re::Ptr{Float64}, im::Ptr{Float64})::Int32
    child=reshape([0.5], 1, 1)
    vectors=fill(777.0, 1, 1)
    result=direct_ordinary_geesx!(
        child,
        vectors,
        Int32;
        job=('N'),
        sort=('S'),
        sense=('E'),
        selection=ordinary_geesx_callback(Float64, Int32, :imag),
    )
    ordinary_nested_geesx_runs[]+=1
    ordinary_nested_geesx_ok[]&=result.info==0&&result.count==0
    return Int32(unsafe_load(re)<1.25)
end
const ordinary_nested_geesx_pointer=@cfunction(
    ordinary_nested_geesx_select, Int32, (Ptr{Float64}, Ptr{Float64})
)
@testset "GEESX original expert callback restores an outer LP64 selection" begin
    ordinary_nested_geesx_runs[]=0
    ordinary_nested_geesx_ok[]=true
    initial=[-2.0 0.5; 0.0 3.0]
    a=vcat(initial, fill(777.0, 3, 2))
    vs=fill(777.0, 5, 2)
    result=direct_ordinary_geesx!(
        a, vs, Int32; sense=('E'), selection=ordinary_nested_geesx_pointer
    )
    @test result.info==result.diagnostic==0&&result.count==1
    @test ordinary_nested_geesx_runs[]>0&&ordinary_nested_geesx_ok[]
    @test vs[1:2, :]*a[1:2, :]*vs[1:2, :]'≈initial
    @test all(==(777.0), vs[3:end, :])&&all(==(777.0), a[3:end, :])
end
