using Test, LinearAlgebra, Random, Libdl, Colmat
function tsqr_diagnostic()
    return ccall(
        Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error),
        Cint,
        (),
    )
end
function tsqr_reference()
    return only(
        filter(
            l->l.interface==:ilp64&&occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
end
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    suffix=J===Int64 ? "64_" : ""
    factor=Symbol(prefix*"latsqr_"*suffix)
    org=prefix*(T<:Complex ? "ungtsqr" : "orgtsqr")
    @eval function direct_tsqr_factor!(
        a::Matrix{$T},
        ::Type{$J};
        m=size(a, 1),
        n=size(a, 2),
        mb,
        nb,
        lda=size(a, 1),
        ldt=nb+2,
        lwork=max(1, n*nb),
        library=Colmat.backend_library(),
    )
        slots=mb>n&&mb<m ? cld(m-n, mb-n) : 1
        t=fill($T(777), ldt, max(1, n*slots))
        work=fill($T(777), max(1, lwork))
        info=fill($J(-777), 3)
        ip=view(info, 2:2)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(factor)))
        ccall(
            f,
            Cvoid,
            (
                Ref{$J},
                Ref{$J},
                Ref{$J},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$J},
            ),
            m,
            n,
            mb,
            nb,
            a,
            lda,
            t,
            ldt,
            work,
            lwork,
            ip,
        )
        (; t, work, info=info[2], guard=info[[1, 3]], diagnostic=tsqr_diagnostic())
    end
    @eval function direct_tsqr_generate!(
        a::Matrix{$T},
        t::Matrix{$T},
        ::Type{$J};
        row=false,
        m=size(a, 1),
        n=size(a, 2),
        mb,
        nb,
        lda=size(a, 1),
        ldt=size(t, 1),
        lwork=max(2, (m+min(nb, n))*n),
        library=Colmat.backend_library(),
    )
        work=fill($T(777), max(1, lwork))
        info=fill($J(-777), 3)
        ip=view(info, 2:2)
        f=Libdl.dlsym(Libdl.dlopen(library), Symbol($org*(row ? "_row_" : "_")*$suffix))
        ccall(
            f,
            Cvoid,
            (
                Ref{$J},
                Ref{$J},
                Ref{$J},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$J},
            ),
            m,
            n,
            mb,
            nb,
            a,
            lda,
            t,
            ldt,
            work,
            lwork,
            ip,
        )
        (; work, info=info[2], guard=info[[1, 3]], diagnostic=tsqr_diagnostic())
    end
end
@testset "TSQR row blocks and external factor interoperability" begin
    rng=MersenneTwister(302)
    reference=tsqr_reference()
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        (m, n) in ((0, 0), (1, 1), (5, 2), (12, 3), (17, 4)),
        mb in unique((n+1, n+4, m+2)),
        nb in unique((1, min(2, max(1, n)), max(1, n))),
        mode in (:dense, :zero, :rankone)

        original=if mode==:zero
            zeros(T, m, n)
        elseif mode==:rankone
            randn(rng, T, m)*randn(rng, T, 1, n)
        else
            randn(rng, T, m, n)
        end
        for (library, K) in ((Colmat.backend_library(), J), (reference, Int64))
            a=fill(T(777), m+2, n)
            a[1:m, :]=original
            factor=direct_tsqr_factor!(a, K; m, n, mb, nb, library)
            @test factor.info==factor.diagnostic==0
            @test factor.guard==fill(K(-777), 2)
            @test all(==(T(777)), a[(m + 1):end, :])
            @test all(==(T(777)), factor.t[(nb + 1):end, :])
            R=triu(a[1:n, :])
            for (qlibrary, L) in ((Colmat.backend_library(), J), (reference, Int64)),
                row in (false, true)

                q=copy(a)
                t=copy(factor.t)
                generated=direct_tsqr_generate!(
                    q, t, L; row, m, n, mb, nb, library=qlibrary
                )
                @test generated.info==generated.diagnostic==0
                @test generated.guard==fill(L(-777), 2)
                @test all(==(T(777)), q[(m + 1):end, :])
                @test isequal(t, factor.t)
                Q=q[1:m, :]
                tol=1500eps(real(T))
                @test Q'*Q≈Matrix{T}(I, n, n) atol=tol rtol=tol
                @test Q*R≈original atol=tol rtol=tol
            end
        end
    end
end
