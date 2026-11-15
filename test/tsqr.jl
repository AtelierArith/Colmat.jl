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

for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    suffix=J===Int64 ? "64_" : ""
    geqr=Symbol(prefix*"geqr_"*suffix)
    gemqr=Symbol(prefix*"gemqr_"*suffix)
    hr=Symbol(prefix*(T<:Complex ? "unhr_col_" : "orhr_col_")*suffix)
    getsqr=Symbol(prefix*"getsqrhrt_"*suffix)
    @eval function direct_geqr!(
        a::Matrix{$T},
        ::Type{$J};
        m=size(a, 1),
        n=size(a, 2),
        lda=size(a, 1),
        tsize=65536,
        lwork=65536,
        library=Colmat.backend_library(),
    )
        storage=fill($T(777), max(5, tsize)+2)
        t=view(storage, 2:(length(storage) - 1))
        work=fill($T(777), max(1, lwork))
        info=fill($J(-777), 3)
        ip=view(info, 2:2)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(geqr)))
        ccall(
            f,
            Cvoid,
            (
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
            a,
            lda,
            t,
            tsize,
            work,
            lwork,
            ip,
        )
        (; t, storage, work, info=info[2], guard=info[[1, 3]], diagnostic=tsqr_diagnostic())
    end
    @eval function direct_gemqr!(
        a::Matrix{$T},
        t::AbstractVector{$T},
        c::Matrix{$T},
        ::Type{$J};
        side=('L'),
        trans=('N'),
        m=size(c, 1),
        n=size(c, 2),
        k,
        lda=size(a, 1),
        tsize=length(t),
        ldc=max(1, size(c, 1)),
        lwork=65536,
        library=Colmat.backend_library(),
    )
        work=fill($T(777), max(1, lwork))
        info=fill($J(-777), 3)
        ip=view(info, 2:2)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(gemqr)))
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{UInt8},
                Ref{$J},
                Ref{$J},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$J},
            ),
            side,
            trans,
            m,
            n,
            k,
            a,
            lda,
            t,
            tsize,
            c,
            ldc,
            work,
            lwork,
            ip,
        )
        (; work, info=info[2], guard=info[[1, 3]], diagnostic=tsqr_diagnostic())
    end
    @eval function direct_hr!(
        a::Matrix{$T},
        ::Type{$J};
        m=size(a, 1),
        n=size(a, 2),
        nb,
        lda=size(a, 1),
        ldt=max(1, min(nb, n))+2,
        library=Colmat.backend_library(),
    )
        t=fill($T(777), ldt, max(1, n))
        d=fill($T(777), max(1, n)+2)
        dp=view(d, 2:(length(d) - 1))
        info=fill($J(-777), 3)
        ip=view(info, 2:2)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(hr)))
        ccall(
            f,
            Cvoid,
            (
                Ref{$J},
                Ref{$J},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$T},
                Ptr{$J},
            ),
            m,
            n,
            nb,
            a,
            lda,
            t,
            ldt,
            dp,
            ip,
        )
        (; t, d, info=info[2], guard=info[[1, 3]], diagnostic=tsqr_diagnostic())
    end
    @eval function direct_getsqrhrt!(
        a::Matrix{$T},
        ::Type{$J};
        m=size(a, 1),
        n=size(a, 2),
        mb,
        nb1,
        nb2,
        lda=size(a, 1),
        ldt=max(1, min(nb2, n))+2,
        lwork=65536,
        library=Colmat.backend_library(),
    )
        t=fill($T(777), ldt, max(1, n))
        work=fill($T(777), max(1, lwork))
        info=fill($J(-777), 3)
        ip=view(info, 2:2)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(getsqr)))
        ccall(
            f,
            Cvoid,
            (
                Ref{$J},
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
            nb1,
            nb2,
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
end
function tsqr_metadata(t, mb, nb)
    T=eltype(t)
    metadata=fill(T(777), 5+nb*size(t, 2))
    metadata[1]=T(length(metadata))
    metadata[2]=T(mb)
    metadata[3]=T(nb)
    for j in axes(t, 2), i in 1:nb
        metadata[5 + i + (j - 1) * nb]=t[i, j]
    end
    return metadata
end
@testset "GEQR metadata, minimal workspace and external GEMQR" begin
    rng=MersenneTwister(6302)
    reference=tsqr_reference()
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        (m, n) in ((0, 4), (7, 0), (1, 1), (5, 8), (9, 3), (70, 3)),
        minimal in (false, true),
        mode in (:dense, :zero, :rankone)

        original=if mode==:zero
            zeros(T, m, n)
        elseif mode==:rankone
            randn(rng, T, m)*randn(rng, T, 1, n)
        else
            randn(rng, T, m, n)
        end
        k=min(m, n)
        tol=2000eps(real(T))
        for (library, K) in ((Colmat.backend_library(), J), (reference, Int64))
            a=fill(T(777), m+2, n)
            a[1:m, :]=original
            before=copy(a)
            query=direct_geqr!(
                a, K; m, n, tsize=minimal ? -2 : -1, lwork=minimal ? -2 : -1, library
            )
            @test query.info==query.diagnostic==0
            @test isequal(a, before)
            @test real(query.t[1])>=n+5
            @test real(query.work[1])>=max(1, n)
            factor=direct_geqr!(
                a,
                K;
                m,
                n,
                tsize=Int(real(query.t[1])),
                lwork=Int(real(query.work[1])),
                library,
            )
            @test factor.info==factor.diagnostic==0
            @test factor.guard==fill(K(-777), 2)
            @test factor.storage[[1, end]]==fill(T(777), 2)
            @test factor.t[4:5]==fill(T(777), 2)
            @test all(==(T(777)), a[(m + 1):end, :])
            R=triu(a[1:k, :])
            for (qlibrary, L) in ((Colmat.backend_library(), J), (reference, Int64))
                q=Matrix{T}(I, m, m)
                r=direct_gemqr!(a, factor.t, q, L; k, m, n=m, library=qlibrary)
                @test r.info==r.diagnostic==0
                @test q'*q≈Matrix{T}(I, m, m) atol=tol rtol=tol
                @test q[:, 1:k]*R≈original atol=tol rtol=tol
                for side in ('L', 'R'), trans in ('N', T<:Complex ? 'C' : 'T')
                    rows, cols=side=='L' ? (m, 3) : (3, m)
                    c=fill(T(777), rows+2, cols)
                    c[1:rows, :]=randn(rng, T, rows, cols)
                    before=copy(c)
                    op=trans=='N' ? q : q'
                    expected=side=='L' ? op*before[1:rows, :] : before[1:rows, :]*op
                    applied=direct_gemqr!(
                        a, factor.t, c, L; side, trans, m=rows, n=cols, k, library=qlibrary
                    )
                    @test applied.info==applied.diagnostic==0
                    @test applied.guard==fill(L(-777), 2)
                    @test c[1:rows, :]≈expected atol=tol rtol=tol
                    @test all(==(T(777)), c[(rows + 1):end, :])
                end
            end
        end
    end
end
@testset "Householder column reconstruction and TSQR reconstruction driver" begin
    rng=MersenneTwister(8302)
    reference=tsqr_reference()
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        (m, n) in ((0, 0), (1, 1), (8, 3), (17, 4)),
        nb in (1, 2, 5),
        identity in (false, true)

        tol=2500eps(real(T))
        original=identity ? Matrix{T}(I, m, n) : Matrix(qr(randn(rng, T, m, n)).Q)[:, 1:n]
        for (library, K) in ((Colmat.backend_library(), J), (reference, Int64))
            a=fill(T(777), m+2, n)
            a[1:m, :]=original
            r=direct_hr!(a, K; m, n, nb, library)
            @test r.info==r.diagnostic==0
            @test r.guard==fill(K(-777), 2)
            @test r.d[[1, end]]==fill(T(777), 2)
            @test all(x->x==1||x==-1, r.d[2:(n + 1)])
            @test all(==(T(777)), a[(m + 1):end, :])
            metadata=tsqr_metadata(r.t, m, max(1, min(nb, n)))
            for (qlibrary, L) in ((Colmat.backend_library(), J), (reference, Int64))
                q=Matrix{T}(I, m, n)
                applied=direct_gemqr!(a, metadata, q, L; m, n, k=n, library=qlibrary)
                @test applied.info==applied.diagnostic==0
                @test q*Diagonal(r.d[2:(n + 1)])≈original atol=tol rtol=tol
            end
        end
    end
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), (m, n) in ((0, 0), (1, 1), (9, 3), (18, 4)), nb1 in (1, 3),
        nb2 in (1, 2, 5)

        original=randn(rng, T, m, n)
        mb=n+3
        tol=2500eps(real(T))
        for (library, K) in ((Colmat.backend_library(), J), (reference, Int64))
            a=fill(T(777), m+2, n)
            a[1:m, :]=original
            query=direct_getsqrhrt!(a, K; m, n, mb, nb1, nb2, lwork=-1, library)
            @test query.info==query.diagnostic==0
            r=direct_getsqrhrt!(
                a, K; m, n, mb, nb1, nb2, lwork=Int(real(query.work[1])), library
            )
            @test r.info==r.diagnostic==0
            @test r.guard==fill(K(-777), 2)
            @test all(==(T(777)), a[(m + 1):end, :])
            R=triu(a[1:n, :])
            metadata=tsqr_metadata(r.t, m, max(1, min(nb2, n)))
            for (qlibrary, L) in ((Colmat.backend_library(), J), (reference, Int64))
                q=Matrix{T}(I, m, n)
                applied=direct_gemqr!(a, metadata, q, L; m, n, k=n, library=qlibrary)
                @test applied.info==applied.diagnostic==0
                @test q'*q≈Matrix{T}(I, n, n) atol=tol rtol=tol
                @test q*R≈original atol=tol rtol=tol
            end
        end
    end
end

@testset "TSQR workspace queries and argument diagnostics" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        initial=fill(T(NaN), 7, 3)
        for (kw, arg) in (
            ((; m=-1), 1),
            ((; n=-1), 2),
            ((; mb=0), 3),
            ((; nb=0), 4),
            ((; lda=4), 6),
            ((; ldt=1), 8),
            ((; lwork=5), 10),
        )
            a=copy(initial)
            r=direct_tsqr_factor!(a, J; m=5, n=3, mb=4, nb=2, kw...)
            @test r.info == -arg
            @test r.diagnostic == arg
            @test r.guard==fill(J(-777), 2)
            @test isequal(a, initial)
        end
        a=copy(initial)
        r=direct_tsqr_factor!(a, J; m=5, n=3, mb=4, nb=2, lwork=-1)
        @test r.info==r.diagnostic==0
        @test real(r.work[1])==6
        @test isequal(a, initial)
        t=fill(T(NaN), 4, 9)
        for row in (false, true)
            for (kw, arg) in (
                ((; m=-1), 1),
                ((; n=-1), 2),
                ((; mb=3), 3),
                ((; nb=0), 4),
                ((; lda=4), 6),
                ((; ldt=1), 8),
                ((; lwork=row ? 0 : 20), 10),
            )
                a=copy(initial)
                r=direct_tsqr_generate!(a, t, J; row, m=5, n=3, mb=4, nb=2, kw...)
                @test r.info == -arg
                @test r.diagnostic == arg
                @test r.guard==fill(J(-777), 2)
                @test isequal(a, initial)
            end
            r=direct_tsqr_generate!(
                copy(initial), t, J; row, m=5, n=3, mb=4, nb=2, lwork=-1
            )
            @test r.info==r.diagnostic==0
            @test real(r.work[1])==(row ? 4 : 21)
        end
        for (kw, arg) in (
            ((; m=-1), 1), ((; n=-1), 2), ((; lda=4), 4), ((; tsize=7), 6), ((; lwork=2), 8)
        )
            a=copy(initial)
            r=direct_geqr!(a, J; m=5, n=3, kw...)
            @test r.info == -arg
            @test r.diagnostic == arg
            @test r.guard==fill(J(-777), 2)
            @test isequal(a, initial)
        end
        for (kw, arg) in
            (((; m=-1), 1), ((; n=-1), 2), ((; nb=0), 3), ((; lda=4), 5), ((; ldt=1), 7))
            a=copy(initial)
            r=direct_hr!(a, J; m=5, n=3, nb=2, kw...)
            @test r.info == -arg
            @test r.diagnostic == arg
            @test r.guard==fill(J(-777), 2)
            @test isequal(a, initial)
        end
        for (kw, arg) in (
            ((; m=-1), 1),
            ((; n=-1), 2),
            ((; mb=3), 3),
            ((; nb1=0), 4),
            ((; nb2=0), 5),
            ((; lda=4), 7),
            ((; ldt=1), 9),
            ((; lwork=1), 11),
        )
            a=copy(initial)
            r=direct_getsqrhrt!(a, J; m=5, n=3, mb=4, nb1=2, nb2=2, kw...)
            @test r.info == -arg
            @test r.diagnostic == arg
            @test r.guard==fill(J(-777), 2)
            @test isequal(a, initial)
        end
        a=Matrix{T}(I, 5, 3)
        factor=direct_geqr!(a, J; m=5, n=3)
        @test factor.info==factor.diagnostic==0
        c=fill(T(NaN), 7, 2)
        for (kw, arg) in (
            ((; side=('?')), 1),
            ((; trans=('?')), 2),
            ((; m=-1), 3),
            ((; n=-1), 4),
            ((; k=-1), 5),
            ((; lda=4), 7),
            ((; tsize=4), 9),
            ((; ldc=4), 11),
            ((; lwork=1), 13),
        )
            r=direct_gemqr!(a, factor.t, c, J; m=5, n=2, k=3, kw...)
            @test r.info == -arg
            @test r.diagnostic == arg
            @test r.guard==fill(J(-777), 2)
            @test all(isnan, c)
        end
        r=direct_gemqr!(a, factor.t, c, J; m=5, n=2, k=3, lwork=-1)
        @test r.info==r.diagnostic==0
        @test real(r.work[1])==6
        @test all(isnan, c)
        if J===Int64
            r=direct_geqr!(copy(initial), J; m=(-(Int64(1)<<32)), n=3)
            @test r.info == -1
            @test r.diagnostic==1
        end
    end
end
@testset "All TSQR family symbols through LBT" begin
    rng=MersenneTwister(10302)
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64)

        org=T<:Complex ? "ungtsqr" : "orgtsqr"
        hr=T<:Complex ? "unhr_col" : "orhr_col"
        for family in ("geqr", "gemqr", "getsqrhrt", "latsqr", org, org*"_row", hr)
            provider=BLAS.lbt_find_backing_library(
                prefix*family*"_", J===Int32 ? :lp64 : :ilp64
            )
            @test provider!==nothing
            @test realpath(provider.libname)==realpath(Colmat.backend_library())
        end
        original=randn(rng, T, 12, 3)
        tol=2000eps(real(T))
        library=BLAS.libblastrampoline
        a=copy(original)
        factor=direct_geqr!(a, J; m=12, n=3, library)
        @test factor.info==factor.diagnostic==0
        q=Matrix{T}(I, 12, 3)
        r=direct_gemqr!(a, factor.t, q, J; m=12, n=3, k=3, library)
        @test r.info==r.diagnostic==0
        @test q*triu(a[1:3, :])≈original atol=tol rtol=tol
        a=copy(original)
        factor=direct_tsqr_factor!(a, J; m=12, n=3, mb=5, nb=2, library)
        @test factor.info==factor.diagnostic==0
        for row in (false, true)
            q=copy(a)
            r=direct_tsqr_generate!(q, factor.t, J; row, m=12, n=3, mb=5, nb=2, library)
            @test r.info==r.diagnostic==0
            @test q*triu(a[1:3, :])≈original atol=tol rtol=tol
        end
        q=Matrix(qr(original).Q)[:, 1:3]
        a=copy(q)
        r=direct_hr!(a, J; m=12, n=3, nb=2, library)
        @test r.info==r.diagnostic==0
        metadata=tsqr_metadata(r.t, 12, 2)
        reconstructed=Matrix{T}(I, 12, 3)
        applied=direct_gemqr!(a, metadata, reconstructed, J; m=12, n=3, k=3, library)
        @test applied.info==applied.diagnostic==0
        @test reconstructed*Diagonal(r.d[2:4])≈q atol=tol rtol=tol
        a=copy(original)
        r=direct_getsqrhrt!(a, J; m=12, n=3, mb=5, nb1=2, nb2=2, library)
        @test r.info==r.diagnostic==0
        metadata=tsqr_metadata(r.t, 12, 2)
        q=Matrix{T}(I, 12, 3)
        applied=direct_gemqr!(a, metadata, q, J; m=12, n=3, k=3, library)
        @test applied.info==applied.diagnostic==0
        @test q*triu(a[1:3, :])≈original atol=tol rtol=tol
    end
end

@testset "TSQR minimal row workspace, scale range and unused strides" begin
    rng=MersenneTwister(12302)
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        R=real(T)
        scales=R===Float32 ? (R(1e-30), one(R), R(1e30)) : (R(1e-200), one(R), R(1e200))
        original=randn(rng, T, 17, 3)
        tol=2500eps(R)
        for scale in scales
            a=original*scale
            factor=direct_tsqr_factor!(a, J; m=17, n=3, mb=5, nb=2)
            @test factor.info==factor.diagnostic==0
            r=triu(a[1:3, :])/scale
            q=copy(a)
            generated=direct_tsqr_generate!(
                q, factor.t, J; row=true, m=17, n=3, mb=5, nb=2, lwork=1
            )
            @test generated.info==generated.diagnostic==0
            @test length(generated.work)==1
            @test q'*q≈Matrix{T}(I, 3, 3) atol=tol rtol=tol
            @test q*r≈original atol=tol rtol=tol
        end
        huge=Int(typemax(J))
        a=fill(T(2), 1, 1)
        factor=direct_tsqr_factor!(a, J; m=1, n=1, mb=2, nb=1, lda=huge)
        @test factor.info==factor.diagnostic==0
        generated=direct_tsqr_generate!(
            a, factor.t, J; row=true, m=1, n=1, mb=2, nb=1, lda=huge, lwork=1
        )
        @test generated.info==generated.diagnostic==0
        @test a'*a≈ones(T, 1, 1)
        hr=direct_hr!(a, J; m=1, n=1, nb=1, lda=huge)
        @test hr.info==hr.diagnostic==0
        @test hr.d[2] in (one(T), -one(T))
        empty=Matrix{T}(undef, 0, 0)
        factor=direct_tsqr_factor!(empty, J; m=huge, n=0, mb=1, nb=1, lda=huge)
        @test factor.info==factor.diagnostic==0
        factor=direct_geqr!(empty, J; m=huge, n=0, lda=huge, tsize=5, lwork=1)
        @test factor.info==factor.diagnostic==0
        hr=direct_hr!(empty, J; m=huge, n=0, nb=1, lda=huge)
        @test hr.info==hr.diagnostic==0
        r=direct_getsqrhrt!(empty, J; m=huge, n=0, mb=1, nb1=1, nb2=1, lda=huge, lwork=1)
        @test r.info==r.diagnostic==0
    end
end
