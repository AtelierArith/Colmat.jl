using Test, LinearAlgebra, Random, Libdl, Colmat
function bdsvdx_diagnostic()
    return ccall(
        Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error),
        Cint,
        (),
    )
end
function bdsvdx_reference()
    return only(
        filter(
            l->l.interface==:ilp64&&occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
end
for (R, prefix) in ((Float32, "s"), (Float64, "d")), J in (Int32, Int64)
    symbol=Symbol(prefix*"bdsvdx_"*(J===Int64 ? "64_" : ""))
    @eval function direct_bdsvdx!(
        d::Vector{$R},
        e::Vector{$R},
        ::Type{$J};
        uplo=('U'),
        job=('V'),
        range=('A'),
        n=length(d),
        vl=($R(0)),
        vu=($R(1)),
        il=1,
        iu=max(1, n),
        ldz=max(1, 2*n)+2,
        library=Colmat.backend_library(),
    )
        s=fill($R(777), max(1, n)+2)
        sp=view(s, 2:(length(s) - 1))
        z=fill($R(777), ldz, max(1, n+1))
        work=fill($R(777), max(1, 14*n)+2)
        wp=view(work, 2:(length(work) - 1))
        iw=fill($J(-777), max(1, 12*n)+2)
        iwp=view(iw, 2:(length(iw) - 1))
        ns=fill($J(-777), 3)
        np=view(ns, 2:2)
        info=fill($J(-777), 3)
        ip=view(info, 2:2)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol)))
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{UInt8},
                Ref{UInt8},
                Ref{$J},
                Ptr{$R},
                Ptr{$R},
                Ref{$R},
                Ref{$R},
                Ref{$J},
                Ref{$J},
                Ptr{$J},
                Ptr{$R},
                Ptr{$R},
                Ref{$J},
                Ptr{$R},
                Ptr{$J},
                Ptr{$J},
            ),
            uplo,
            job,
            range,
            n,
            d,
            e,
            vl,
            vu,
            il,
            iu,
            np,
            sp,
            z,
            ldz,
            wp,
            iwp,
            ip,
        )
        (;
            info=info[2],
            ns=ns[2],
            guard=vcat(info[[1, 3]], ns[[1, 3]]),
            s,
            z,
            work,
            iw,
            diagnostic=bdsvdx_diagnostic(),
        )
    end
end
@testset "BDSVDX selected triplets and reference range contracts" begin
    rng=MersenneTwister(30603)
    reference=bdsvdx_reference()
    for R in (Float32, Float64),
        J in (Int32, Int64), n in (0, 1, 3, 7), uplo in ('U', 'L'), job in ('N', 'V'),
        range in ('A', 'V', 'I')

        d=randn(rng, R, n)
        e=randn(rng, R, max(0, n-1))/3
        b=Matrix(Bidiagonal(d, e, uplo))
        allvalues=svdvals(b)
        tol=3000eps(R)
        lo=n>1 ? (allvalues[end]+allvalues[end - 1])/2 : R(0)
        hi=n>1 ? allvalues[1]+one(R) : R(10)
        il=1
        iu=max(1, min(2, n))
        selected=if range=='A'
            allvalues
        elseif range=='I'
            allvalues[1:min(iu, n)]
        else
            filter(s->lo<s<hi, allvalues)
        end
        for (library, K) in ((Colmat.backend_library(), J), (reference, Int64))
            r=direct_bdsvdx!(
                copy(d), copy(e), K; uplo, job, range, vl=lo, vu=hi, il, iu, library
            )
            @test r.info==r.diagnostic==0
            @test r.guard==fill(K(-777), 4)
            @test r.ns==length(selected)
            values=r.s[2:(1 + r.ns)]
            @test values≈selected atol=tol*max(norm(b), floatmin(R)) rtol=tol
            @test r.s[[1, end]]==r.work[[1, end]]==fill(R(777), 2)
            @test r.iw[[1, end]]==fill(K(-777), 2)
            if job=='V'
                u=r.z[1:n, 1:r.ns]
                v=r.z[(n + 1):(2 * n), 1:r.ns]
                @test b*v≈u*Diagonal(values) atol=tol*max(norm(b), floatmin(R)) rtol=tol
                @test b'*u≈v*Diagonal(values) atol=tol*max(norm(b), floatmin(R)) rtol=tol
                @test u'*u≈Matrix{R}(I, r.ns, r.ns) atol=tol rtol=tol
                @test v'*v≈Matrix{R}(I, r.ns, r.ns) atol=tol rtol=tol
                # N=1 reference quick return does not initialize IWORK.
                if n>1
                    @test all(==(0), r.iw[2:(1 + r.ns)])
                end
            else
                @test all(==(R(777)), r.z)
            end
            @test all(==(R(777)), r.z[(2 * n + 1):end, :])
        end
    end
    # Reference general path selects [VL,VU); its singleton shortcut uses
    # (VL,VU]. Pin both behaviors, independently of rounded random spectra.
    for R in (Float32, Float64), J in (Int32, Int64), n in (1, 3)
        d=n==1 ? R[-4] : R[4, 2, 1]
        e=zeros(R, n-1)
        for (library, K) in ((Colmat.backend_library(), J), (reference, Int64))
            r=direct_bdsvdx!(d, e, K; range=('V'), vl=R(2), vu=R(4), library)
            @test r.info==r.diagnostic==0
            @test r.ns==1
            @test r.s[2]≈R(n==1 ? 4 : 2)
        end
    end
end
