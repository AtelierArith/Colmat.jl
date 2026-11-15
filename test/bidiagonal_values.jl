using Test, LinearAlgebra, Libdl, Colmat
function lasq_diagnostic()
    return ccall(
        Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error),
        Cint,
        (),
    )
end
function lasq_reference()
    return only(
        filter(
            l->l.interface==:ilp64&&occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
end
for (R, prefix) in ((Float32, "s"), (Float64, "d")), J in (Int32, Int64)
    symbol=Symbol(prefix*"lasq1_"*(J===Int64 ? "64_" : ""))
    @eval function direct_lasq1!(
        d::AbstractVector{$R},
        e::AbstractVector{$R},
        ::Type{$J};
        n=length(d),
        library=Colmat.backend_library(),
    )
        work=fill($R(777), max(1, 4*max(0, n))+2)
        wp=view(work, 2:(length(work) - 1))
        info=fill($J(-777), 3)
        ip=view(info, 2:2)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol)))
        ccall(f, Cvoid, (Ref{$J}, Ptr{$R}, Ptr{$R}, Ptr{$R}, Ptr{$J}), n, d, e, wp, ip)
        (; info=info[2], guard=info[[1, 3]], work, diagnostic=lasq_diagnostic())
    end
end
@testset "LASQ1 native dqds, reference recovery and integer guards" begin
    reference=lasq_reference()
    for R in (Float32, Float64),
        J in (Int32, Int64), n in (0, 1, 2, 3, 8, 40), deficient in (false, true),
        scale in (R===Float32 ? R[1, 1e-10, 1e10] : R[1, 1e-150, 1e150])

        d0=R[(isodd(i) ? -1 : 1)*(1+i/10) for i in 1:n]*scale
        e0=R[0.125 for i in 1:max(0, n - 1)]*scale
        deficient&&n>0&&(d0[1]=0)
        results=[]
        for (library, K) in ((Colmat.backend_library(), J), (reference, Int64))
            ds=vcat(R(777), d0, R(777))
            es=vcat(R(777), e0, R(777), R(777))
            d=view(ds, 2:(length(ds) - 1))
            e=view(es, 2:(length(es) - 1))
            r=direct_lasq1!(d, e, K; library)
            @test r.info==r.diagnostic==0
            @test r.guard==fill(K(-777), 2)
            @test ds[[1, end]]==es[[1, end]]==r.work[[1, end]]==fill(R(777), 2)
            @test all(>=(0), d)&&issorted(d; rev=true)
            push!(results, (d=copy(d), e=copy(e), work=r.work))
        end
        @test all(isapprox.(results[1].d, results[2].d; rtol=2000eps(R), atol=0))
        if deficient&&n>1
            # Native dqds explicitly rejects a zero diagonal. Its recovery
            # path must preserve reference D/E and all WORK diagnostics.
            @test results[1].d==results[2].d
            @test results[1].e==results[2].e
            @test results[1].work==results[2].work
        elseif n>1
            @test all(==(R(777)), results[1].work)
        end
    end
    for R in (Float32, Float64), J in (Int32, Int64)
        d=R[3, 4]
        e=R[1, 777]
        r=direct_lasq1!(d, e, J)
        @test r.info==r.diagnostic==0
        @test d≈sqrt.(R[18, 8])
        @test e==R[1, 777]
        d=R[777]
        e=copy(d)
        r=direct_lasq1!(d, e, J; n=-1)
        @test r.info==-1&&r.diagnostic==1
        @test r.guard==fill(J(-777), 2)
        @test d==e==R[777]
        @test all(==(R(777)), r.work)
    end
end

@testset "LASQ1 recovery retains relative accuracy beyond squared normal range" begin
    for (R, small) in ((Float32, Float32(1e-22)), (Float64, 1e-161)), J in (Int32, Int64)
        expected=R[hypot(one(R), R(0.25)), small / hypot(one(R), R(0.25))]
        outcomes=[]
        for (library, K) in ((Colmat.backend_library(), J), (lasq_reference(), Int64))
            d=R[small, 1]
            e=R[0.25, 777]
            r=direct_lasq1!(d, e, K; library)
            @test r.info==r.diagnostic==0
            @test all(isapprox.(d, expected; rtol=20eps(R), atol=0))
            @test r.guard==fill(K(-777), 2)
            @test e==R[0.25, 777]
            @test r.work[[1, end]]==fill(R(777), 2)
            push!(outcomes, (d=copy(d), e=copy(e), work=r.work))
        end
        @test outcomes[1]==outcomes[2]
    end
end
