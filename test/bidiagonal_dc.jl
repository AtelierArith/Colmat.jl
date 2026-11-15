using Test, LinearAlgebra, Random, Libdl, Colmat
function bdsdc_diagnostic()
    return ccall(
        Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error),
        Cint,
        (),
    )
end
function bdsdc_reference()
    return only(
        filter(
            l->l.interface==:ilp64&&occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
end
for (R, prefix) in ((Float32, "s"), (Float64, "d")), J in (Int32, Int64)
    symbol=Symbol(prefix*"bdsdc_"*(J===Int64 ? "64_" : ""))
    @eval function direct_bdsdc!(
        d::AbstractVector{$R},
        e::AbstractVector{$R},
        u::Matrix{$R},
        vt::Matrix{$R},
        q::AbstractVector{$R},
        iq::AbstractVector{$J},
        ::Type{$J};
        uplo=('U'),
        comp=('I'),
        n=length(d),
        ldu=size(u, 1),
        ldvt=size(vt, 1),
        library=Colmat.backend_library(),
    )
        nw=max(
            1,
            if comp=='I'
                3*n*n+4*n
            elseif comp=='P'
                6*n
            else
                4*n
            end,
        )
        work=fill($R(777), nw+2)
        wp=view(work, 2:(length(work) - 1))
        iw=fill($J(-777), max(1, 8*n)+2)
        ipw=view(iw, 2:(length(iw) - 1))
        info=fill($J(-777), 3)
        ip=view(info, 2:2)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol)))
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{UInt8},
                Ref{$J},
                Ptr{$R},
                Ptr{$R},
                Ptr{$R},
                Ref{$J},
                Ptr{$R},
                Ref{$J},
                Ptr{$R},
                Ptr{$J},
                Ptr{$R},
                Ptr{$J},
                Ptr{$J},
            ),
            uplo,
            comp,
            n,
            d,
            e,
            u,
            ldu,
            vt,
            ldvt,
            q,
            iq,
            wp,
            ipw,
            ip,
        )
        (; info=info[2], guard=info[[1, 3]], work, iw, diagnostic=bdsdc_diagnostic())
    end
end
@testset "BDSDC explicit factors and preserved compact reference tree" begin
    rng=MersenneTwister(30602)
    reference=bdsdc_reference()
    for R in (Float32, Float64),
        J in (Int32, Int64), n in (0, 1, 3, 30, 70), uplo in ('U', 'L'),
        comp in ('N', 'I', 'P')

        d0=randn(rng, R, n)
        e0=randn(rng, R, max(0, n-1))/3
        original=Matrix(Bidiagonal(d0, e0, uplo))
        tol=3000eps(R)
        # Overallocate compact owner arrays independently of ILAENV tuning.
        # Compare precisely the slots the reference actually writes.
        qlen=max(1, n*(80+8*ceil(Int, log2(max(1, n)))))
        ilen=max(1, n*(9+3*ceil(Int, log2(max(1, n)))))
        results=[]
        for (library, K) in ((Colmat.backend_library(), J), (reference, Int64))
            ds=vcat(R(777), d0, R(777))
            es=vcat(R(777), e0, R(777))
            d=view(ds, 2:(length(ds) - 1))
            e=view(es, 2:(length(es) - 1))
            u=fill(R(777), n+2, n)
            vt=copy(u)
            qs=fill(R(777), qlen+2)
            q=view(qs, 2:(length(qs) - 1))
            iqs=fill(K(-777), ilen+2)
            iq=view(iqs, 2:(length(iqs) - 1))
            r=direct_bdsdc!(d, e, u, vt, q, iq, K; uplo, comp, library)
            @test r.info==r.diagnostic==0
            @test r.guard==fill(K(-777), 2)
            @test ds[[1, end]]==es[[1, end]]==qs[[1, end]]==fill(R(777), 2)
            @test iqs[[1, end]]==r.iw[[1, end]]==fill(K(-777), 2)
            @test r.work[[1, end]]==fill(R(777), 2)
            @test all(>=(0), d)&&issorted(d; rev=true)
            if comp=='I'
                left=u[1:n, :]
                rightt=vt[1:n, :]
                @test left*Diagonal(d)*rightt≈original atol=tol*max(
                    norm(original), floatmin(R)
                ) rtol=tol
                @test left'*left≈Matrix{R}(I, n, n) atol=tol rtol=tol
                @test rightt*rightt'≈Matrix{R}(I, n, n) atol=tol rtol=tol
            else
                @test all(==(R(777)), u)&&all(==(R(777)), vt)
            end
            @test all(==(R(777)), u[(n + 1):end, :])&&all(==(R(777)), vt[(n + 1):end, :])
            comp!='P'&&(@test all(==(R(777)), q)&&all(==(K(-777)), iq))
            push!(results, (d=copy(d), q=copy(q), iq=copy(iq)))
        end
        @test results[1].d≈results[2].d atol=tol*max(norm(original), floatmin(R)) rtol=tol
        if comp=='P'
            @test results[1].q≈results[2].q atol=tol rtol=tol
            @test results[1].iq==results[2].iq
        end
    end
end
