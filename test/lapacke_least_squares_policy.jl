using Test, Colmat, LinearAlgebra, Libdl
isdefined(@__MODULE__, :lapacke288_svd_ls_call) || include("lapacke_least_squares_svd.jl")
@testset "Least-squares C diagnostic priorities, NaN policy and integer boundaries" begin
    native=Libdl.dlopen(Colmat.backend_library())
    getnan=Libdl.dlsym(native, :LAPACKE_get_nancheck)
    setnan=Libdl.dlsym(native, :LAPACKE_set_nancheck)
    previous=ccall(getnan, Cint, ())
    ccall(setnan, Cvoid, (Cint,), 1)
    diagnostic() = ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
    try
        for T in (Float32, Float64, ComplexF32, ComplexF64),
            J in (Int32, Int64), family in (:gels, :gelsy, :gelss, :gelsd),
            route in (:dll, :lbt_trampoline)

            R=real(T)
            a=zeros(T, 128)
            b=zeros(T, 128)
            pivot=zeros(J, 10)
            rank=J[777, 778, 777]
            singular=fill(R(782), 8)
            work=fill(T(779), 34)
            rwork=fill(R(780), 34)
            iwork=fill(J(781), 34)
            pv=view(pivot, 2:9)
            rv=view(rank, 2:2)
            sv=view(singular, 2:7)
            wv=view(work, 2:33)
            rw=view(rwork, 2:33)
            iw=view(iwork, 2:33)
            function call(;
                high=false,
                layout=102,
                trans=('N'),
                m=3,
                n=4,
                r=2,
                lda=8,
                ldb=8,
                lwork=-1,
                rcond=R(0.001),
                ap=a,
                bp=b,
                pp=pv,
                sp=sv,
                rp=rv,
            )
                if family==:gels
                    lapacke288_gels_call(
                        T,
                        J,
                        high,
                        layout,
                        trans,
                        m,
                        n,
                        r,
                        ap,
                        lda,
                        bp,
                        ldb,
                        wv,
                        lwork;
                        route,
                    )
                elseif family==:gelsy
                    lapacke288_gelsy_call(
                        T,
                        J,
                        high,
                        layout,
                        m,
                        n,
                        r,
                        ap,
                        lda,
                        bp,
                        ldb,
                        pp,
                        rcond,
                        rp,
                        wv,
                        lwork,
                        rw;
                        route,
                    )
                else
                    lapacke288_svd_ls_call(
                        T,
                        J,
                        Val(family),
                        high,
                        layout,
                        m,
                        n,
                        r,
                        ap,
                        lda,
                        bp,
                        ldb,
                        sp,
                        rcond,
                        rp,
                        wv,
                        lwork,
                        rw,
                        iw;
                        route,
                    )
                end
            end
            shift=family==:gels ? 1 : 0
            cases=[
                ((; layout=0), -1),
                ((; m=-1), -2-shift),
                ((; n=-1), -3-shift),
                ((; r=-1), -4-shift),
                ((; lda=2), -6-shift),
                ((; ldb=3), -8-shift),
                ((; layout=101, m=-1, lda=3), -6-shift),
                ((; layout=101, ldb=1), -8-shift),
            ]
            family==:gels && push!(cases, ((; trans=('?')), -2))
            for (kwargs, expected) in cases
                saved=deepcopy((a, b, pivot, rank, singular, work, rwork, iwork))
                @test call(; kwargs...)==expected
                @test isequal((a, b, pivot, rank, singular, work, rwork, iwork), saved)
                @test diagnostic()==0
            end
            # Null numerical arrays prove query/error paths do not pack inputs.
            saved=deepcopy((a, b, pivot, rank, singular))
            @test call(ap=C_NULL, bp=C_NULL, pp=C_NULL, sp=C_NULL, rp=C_NULL)==0
            query=work[2]
            @test real(query)>=1
            @test isequal((a, b, pivot, rank, singular), saved)
            work[2]=T(779)
            @test call(ap=C_NULL, bp=C_NULL, pp=C_NULL, sp=C_NULL, rp=C_NULL, lwork=0)==(
                family==:gels ? -11 : -13
            )
            @test work[2]==query
            @test diagnostic()==(family==:gels ? 10 : 12)
            @test isequal((a, b, pivot, rank, singular), saved)
            if J===Int64
                @test call(m=-4294967293, ap=C_NULL, bp=C_NULL)==-2-shift
                @test call(r=-4294967293, ap=C_NULL, bp=C_NULL)==-4-shift
                @test call(
                    layout=101,
                    m=0,
                    n=4294967296,
                    r=0,
                    lda=4294967296,
                    ldb=0,
                    ap=C_NULL,
                    bp=C_NULL,
                    pp=C_NULL,
                    sp=C_NULL,
                    rp=C_NULL,
                )==0
                if family==:gelsy && T<:Complex
                    @test real(work[2])>4294967296
                end
                @test rank[[1, end]]==J[777, 777]
            end
            for layout in (101, 102)
                fill!(a, 0)
                fill!(b, 0)
                a[1]=T(NaN)
                b[1]=T(NaN)
                saved=deepcopy((a, b, pivot, rank, singular))
                @test call(; high=true, layout, rcond=R(NaN))==(family==:gels ? -6 : -5)
                @test isequal((a, b, pivot, rank, singular), saved)
                a[1]=0
                @test call(; high=true, layout, rcond=R(NaN))==(family==:gels ? -8 : -7)
                fill!(b, 0)
                # The high-level B scan includes max(M,N) logical rows.
                unused=1+(layout==101 ? 3*8 : 3)
                b[unused]=T(NaN)
                @test call(; high=true, layout)==(family==:gels ? -8 : -7)
                fill!(b, 0)
                if family!=:gels
                    saved=deepcopy((a, b, pivot, rank, singular))
                    @test call(; high=true, layout, rcond=R(NaN))==-10
                    @test isequal((a, b, pivot, rank, singular), saved)
                    @test call(; layout, rcond=R(NaN), ap=C_NULL, bp=C_NULL)==0
                    # S is output storage and must not be scanned for NaNs.
                    fill!(singular, R(NaN))
                    @test call(; high=true, layout)==0
                    @test rank[2]==0
                    @test all(iszero, singular[2:(1 + min(3, 4))]) || family==:gelsy
                    @test all(isnan, singular[[1, end]])
                    fill!(singular, R(782))
                    ccall(setnan, Cvoid, (Cint,), 0)
                    @test call(;
                        high=true, layout, m=0, n=0, r=0, rcond=R(NaN), ap=C_NULL, bp=C_NULL
                    )==0
                    ccall(setnan, Cvoid, (Cint,), 1)
                end
                @test diagnostic()==0
            end
            @test work[[1, end]]==T[779, 779] &&
                rwork[[1, end]]==R[780, 780] &&
                iwork[[1, end]]==J[781, 781]
            @test rank[[1, end]]==J[777, 777]
        end
    finally
        ccall(setnan, Cvoid, (Cint,), previous)
    end
end
