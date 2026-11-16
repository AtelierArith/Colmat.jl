using Test, Colmat, Libdl
isdefined(@__MODULE__, :lapacke288_tp_application_call) ||
    include("lapacke_pentagonal_application.jl")
@testset "Triangular-pentagonal C headers, NaN policy and void diagnostics" begin
    native=Libdl.dlopen(Colmat.backend_library())
    getnan=Libdl.dlsym(native, :LAPACKE_get_nancheck)
    setnan=Libdl.dlsym(native, :LAPACKE_set_nancheck)
    previous=ccall(getnan, Cint, ())
    ccall(setnan, Cvoid, (Cint,), 1)
    try
        for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
            a=zeros(T, 128)
            b=zeros(T, 128)
            v=zeros(T, 128)
            t=zeros(T, 128)
            work=fill(T(779), 128)
            for blocked in (false, true)
                function factor(;
                    high=false, layout=102, m=3, n=3, l=1, nb=2, lda=8, ldb=8, ldt=8
                )
                    lapacke288_tp_factor_call(
                        T,
                        J,
                        Val(blocked),
                        Val(high),
                        layout,
                        m,
                        n,
                        l,
                        nb,
                        a,
                        lda,
                        b,
                        ldb,
                        t,
                        ldt,
                        work,
                    )
                end
                cases=[
                    ((; layout=0), -1),
                    ((; m=-1), -2),
                    ((; n=-1), -3),
                    ((; l=4), -4),
                    ((; lda=2), blocked ? -7 : -6),
                    ((; ldb=2), blocked ? -9 : -8),
                    ((; ldt=0), blocked ? -11 : -10),
                    ((; layout=101, m=-1, lda=2), blocked ? -7 : -5),
                    ((; layout=101, ldb=2), blocked ? -9 : -7),
                    ((; layout=101, ldt=2), blocked ? -11 : -9),
                ]
                blocked && append!(cases, [((; nb=0), -5), ((; nb=4), -5)])
                for (kwargs, expected) in cases
                    saved=deepcopy((a, b, t, work))
                    @test factor(; kwargs...)==expected
                    @test isequal((a, b, t, work), saved)
                end
                if J===Int64
                    @test factor(m=-4294967293, n=0, l=0)==-2
                end
                for layout in (101, 102)
                    fill!(a, 0)
                    fill!(b, 0)
                    fill!(t, 0)
                    a[1]=T(NaN)
                    b[1]=T(NaN)
                    @test factor(; high=true, layout)==(blocked ? -6 : -4)
                    a[1]=0
                    @test factor(; high=true, layout)==(blocked ? -8 : -6)
                    fill!(b, 0)
                    # Numerical factors ignore lower triangular A/B; high scans do not.
                    ignored=1+(layout==101 ? 2*8 : 2)
                    a[ignored]=T(NaN)
                    b[ignored]=T(NaN)
                    @test factor(; layout, l=3)==0
                    @test isnan(a[ignored]) && isnan(b[ignored])
                    @test factor(; high=true, layout, l=3)==(blocked ? -6 : -4)
                    fill!(a, 0)
                    fill!(b, 0)
                    fill!(t, 0)
                end
            end
            function application(
                family;
                high=false,
                layout=102,
                side=('L'),
                trans=('N'),
                direct=('F'),
                store=('C'),
                m=3,
                n=5,
                k=2,
                l=1,
                nb=2,
                ldv=8,
                ldt=8,
                lda=8,
                ldb=8,
                ldwork=8,
            )
                lapacke288_tp_application_call(
                    T,
                    J,
                    Val(family),
                    Val(high),
                    layout,
                    side,
                    trans,
                    direct,
                    store,
                    m,
                    n,
                    k,
                    l,
                    nb,
                    v,
                    ldv,
                    t,
                    ldt,
                    a,
                    lda,
                    b,
                    ldb,
                    work,
                    ldwork,
                )
            end
            cases=[
                ((; layout=0), -1),
                ((; side=('?')), -2),
                ((; trans=('?')), -3),
                ((; m=-1), -4),
                ((; n=-1), -5),
                ((; k=-1), -6),
                ((; l=3), -7),
                ((; nb=0), -8),
                ((; ldv=2), -10),
                ((; ldt=1), -12),
                ((; lda=1), -14),
                ((; ldb=2), -16),
                ((; layout=101, side=('?'), lda=0), -2),
                ((; layout=101, lda=4), -14),
                ((; layout=101, ldb=4), -16),
                ((; layout=101, ldt=1), -12),
                ((; layout=101, ldv=1), -10),
            ]
            for (kwargs, expected) in cases
                saved=deepcopy((v, t, a, b, work))
                @test application(:tpmqrt; kwargs...)==expected
                @test isequal((v, t, a, b, work), saved)
            end
            for (kwargs, diagnostic) in [
                ((; side=('?')), 1),
                ((; trans=('?')), 2),
                ((; direct=('?')), 3),
                ((; store=('?')), 4),
                ((; m=-1), 5),
                ((; n=-1), 6),
                ((; k=-1), 7),
                ((; l=3), 8),
                ((; ldv=2), 10),
                ((; ldt=1), 12),
                ((; lda=1), 14),
                ((; ldb=2), 16),
                ((; ldwork=1), 18),
            ]
                saved=deepcopy((v, t, a, b, work))
                @test application(:tprfb; kwargs...)==0
                @test pentagonal_diagnostic()==diagnostic
                @test isequal((v, t, a, b, work), saved)
            end
            for (kwargs, expected) in [
                ((; layout=101, lda=4), -15),
                ((; layout=101, ldb=4), -17),
                ((; layout=101, ldt=1), -13),
                ((; layout=101, ldv=1), -11),
            ]
                @test application(:tprfb; kwargs...)==expected
            end
            for family in (:tpmqrt, :tprfb), layout in (101, 102)
                shift=family===:tprfb ? 1 : 0
                for (array, error) in
                    ((a, -13-shift), (b, -15-shift), (t, -11-shift), (v, -9-shift))
                    fill!(a, 0)
                    fill!(b, 0)
                    fill!(t, 0)
                    fill!(v, 0)
                    array[1]=T(NaN)
                    saved=deepcopy((v, t, a, b, work))
                    @test application(family; high=true, layout)==error
                    @test isequal((v, t, a, b, work), saved)
                end
            end
            for side in ('L', 'R'), layout in (101, 102)
                fill!(a, 0)
                fill!(b, 0)
                fill!(t, 0)
                fill!(v, 0)
                m, n=side=='L' ? (3, 5) : (5, 3)
                ai, aj=side=='L' ? (0, 4) : (4, 0)
                a[1 + (layout == 101 ? ai * 8 + aj : ai + aj * 8)]=T(NaN)
                @test application(:tprfb; high=true, layout, side, m, n)==-14
                fill!(a, 0)
                pi, pj=side=='L' ? (4, 0) : (0, 4)
                index=1+(layout==101 ? pi*8+pj : pi+pj*8)
                a[index]=T(NaN)
                @test application(:tprfb; high=true, layout, side, m, n)==0
                @test isnan(a[index])
                @test pentagonal_diagnostic()==0
            end
            for family in (:tpmqrt, :tprfb),
                layout in (101, 102), high in (false, true), side in ('L', 'R'),
                (m, n, k) in ((0, 0, 0), (0, 3, 0), (3, 0, 0))

                @test lapacke288_tp_application_call(
                    T,
                    J,
                    Val(family),
                    Val(high),
                    layout,
                    side,
                    'N',
                    'F',
                    'C',
                    m,
                    n,
                    k,
                    0,
                    1,
                    C_NULL,
                    8,
                    C_NULL,
                    8,
                    C_NULL,
                    8,
                    C_NULL,
                    8,
                    C_NULL,
                    8,
                )==0
                @test pentagonal_diagnostic()==0
            end
            if J===Int64
                @test application(:tpmqrt; m=-4294967293)==-4
                @test application(:tprfb; m=-4294967293)==0
                @test pentagonal_diagnostic()==5
            end
        end
    finally
        ccall(setnan, Cvoid, (Cint,), previous)
    end
end
