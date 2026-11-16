using Test, LinearAlgebra, Random, Libdl, Colmat
function reverse_qr_diagnostic()
    return ccall(
        Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error),
        Cint,
        (),
    )
end
function reverse_qr_reference()
    return only(
        filter(
            l->l.interface==:ilp64&&occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
end
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64), rq in (false, true),
    blocked in (false, true)

    suffix=J===Int64 ? "64_" : ""
    factor=Symbol(
        prefix*(rq ? (blocked ? "gerqf" : "gerq2") : (blocked ? "geqlf" : "geql2"))*"_"*suffix,
    )
    org=T<:Complex ? "ung" : "org"
    orm=T<:Complex ? "unm" : "orm"
    generator=Symbol(
        prefix*org*(rq ? (blocked ? "rq" : "r2") : (blocked ? "ql" : "2l"))*"_"*suffix
    )
    apply=Symbol(
        prefix*orm*(rq ? (blocked ? "rq" : "r2") : (blocked ? "ql" : "2l"))*"_"*suffix
    )
    fsig=Any[Ref{J}, Ref{J}, Ptr{T}, Ref{J}, Ptr{T}, Ptr{T}]
    fargs=Any[:m, :n, :a, :lda, :tp, :wp]
    gsig=Any[Ref{J}, Ref{J}, Ref{J}, Ptr{T}, Ref{J}, Ptr{T}, Ptr{T}]
    gargs=Any[:m, :n, :k, :a, :lda, :tau, :wp]
    asig=Any[
        Ref{UInt8},
        Ref{UInt8},
        Ref{J},
        Ref{J},
        Ref{J},
        Ptr{T},
        Ref{J},
        Ptr{T},
        Ptr{T},
        Ref{J},
        Ptr{T},
    ]
    aargs=Any[:side, :trans, :m, :n, :k, :a, :lda, :tau, :c, :ldc, :wp]
    for (sig, args) in ((fsig, fargs), (gsig, gargs), (asig, aargs))
        blocked&&(push!(sig, Ref{J}); push!(args, :lw))
        push!(sig, Ptr{J})
        push!(args, :ip)
    end
    @eval function direct_reverse_factor!(
        a::Matrix{$T},
        ::Type{$J},
        ::Val{$rq},
        ::Val{$blocked};
        m,
        n,
        lda=size(a, 1),
        lwork=nothing,
        library=Colmat.backend_library(),
    )
        lw=isnothing(lwork) ? max(1, $rq ? m : n) : lwork
        tau=fill($T(777), max(1, min(m, n))+2)
        work=fill($T(777), max(1, lw)+2)
        info=fill($J(-777), 3)
        tp=view(tau, 2:(length(tau) - 1))
        wp=view(work, 2:(length(work) - 1))
        ip=view(info, 2:2)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(factor)))
        ccall(f, Cvoid, $(Expr(:tuple, fsig...)), $(fargs...))
        (; tau, work, info=info[2], guard=info[[1, 3]], diagnostic=reverse_qr_diagnostic())
    end
    @eval function direct_reverse_generate!(
        a::Matrix{$T},
        tau::AbstractVector{$T},
        ::Type{$J},
        ::Val{$rq},
        ::Val{$blocked};
        m,
        n,
        k,
        lda=size(a, 1),
        lwork=nothing,
        library=Colmat.backend_library(),
    )
        lw=isnothing(lwork) ? max(1, $rq ? m : n) : lwork
        work=fill($T(777), max(1, lw)+2)
        info=fill($J(-777), 3)
        wp=view(work, 2:(length(work) - 1))
        ip=view(info, 2:2)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(generator)))
        ccall(f, Cvoid, $(Expr(:tuple, gsig...)), $(gargs...))
        (; work, info=info[2], guard=info[[1, 3]], diagnostic=reverse_qr_diagnostic())
    end
    @eval function direct_reverse_apply!(
        a::Matrix{$T},
        tau::AbstractVector{$T},
        c::Matrix{$T},
        ::Type{$J},
        ::Val{$rq},
        ::Val{$blocked};
        side=('L'),
        trans=('N'),
        m,
        n,
        k,
        lda=size(a, 1),
        ldc=size(c, 1),
        lwork=nothing,
        library=Colmat.backend_library(),
    )
        lw=isnothing(lwork) ? max(1, side=='L' ? n : m) : lwork
        work=fill($T(777), max(1, lw)+2)
        info=fill($J(-777), 3)
        wp=view(work, 2:(length(work) - 1))
        ip=view(info, 2:2)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(apply)))
        ccall(f, Cvoid, $(Expr(:tuple, asig...)), $(aargs...))
        (; work, info=info[2], guard=info[[1, 3]], diagnostic=reverse_qr_diagnostic())
    end
end
function reverse_qr_storage(a, rq, m, n, k, rows, cols)
    T=eltype(a)
    out=fill(T(777), rows+2, cols)
    out[1:rows, :].=T(NaN)
    order=rq ? n : m
    for i in 1:k
        head=order-k+i
        if rq
            for j in 1:(head - 1)
                out[rows - k + i, j]=a[m - k + i, j]
            end
        else
            for j in 1:(head - 1)
                out[j, cols - k + i]=a[j, n - k + i]
            end
        end
    end
    return out
end
@testset "QL/RQ factor storage and external Q interoperability" begin
    reference=reverse_qr_reference()
    rng=MersenneTwister(303)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        rq in (false, true),
        (m, n) in ((0, 4), (7, 0), (1, 1), (2, 2), (8, 3), (3, 8), (5, 5)),
        blocked in (false, true),
        mode in (:dense, :zero, :rankone)

        original=if mode==:zero
            zeros(T, m, n)
        elseif mode==:rankone
            randn(rng, T, m)*randn(rng, T, 1, n)
        else
            randn(rng, T, m, n)
        end
        k=min(m, n)
        order=rq ? n : m
        tol=2000eps(real(T))
        for (library, K) in ((Colmat.backend_library(), J), (reference, Int64))
            a=fill(T(777), m+2, n)
            a[1:m, :]=original
            factor=direct_reverse_factor!(a, K, Val(rq), Val(blocked); m, n, library)
            @test factor.info==factor.diagnostic==0
            @test factor.guard==fill(K(-777), 2)
            @test factor.tau[[1, end]]==fill(T(777), 2)
            @test factor.work[[1, end]]==fill(T(777), 2)
            @test all(==(T(777)), a[(m + 1):end, :])
            tau=view(factor.tau, 2:(length(factor.tau) - 1))
            triangular=zeros(T, m, n)
            for j in 1:n, i in 1:m
                if rq ? j-i>=n-m : i-j>=m-n
                    triangular[i, j]=a[i, j]
                end
            end
            opstorage=reverse_qr_storage(a, rq, m, n, k, rq ? k : order, rq ? order : k)
            for (qlibrary, L) in ((Colmat.backend_library(), J), (reference, Int64)),
                gblocked in (false, true)

                qstorage=reverse_qr_storage(a, rq, m, n, k, order, order)
                generated=direct_reverse_generate!(
                    qstorage,
                    tau,
                    L,
                    Val(rq),
                    Val(gblocked);
                    m=order,
                    n=order,
                    k,
                    library=qlibrary,
                )
                @test generated.info==generated.diagnostic==0
                @test generated.guard==fill(L(-777), 2)
                @test generated.work[[1, end]]==fill(T(777), 2)
                @test all(==(T(777)), qstorage[(order + 1):end, :])
                q=qstorage[1:order, :]
                @test q'*q≈Matrix{T}(I, order, order) atol=tol rtol=tol
                @test (rq ? triangular*q : q*triangular)≈original atol=tol rtol=tol
                rows, cols=rq ? (k, n) : (m, k)
                thin=reverse_qr_storage(a, rq, m, n, k, rows, cols)
                generated=direct_reverse_generate!(
                    thin,
                    tau,
                    L,
                    Val(rq),
                    Val(gblocked);
                    m=rows,
                    n=cols,
                    k,
                    library=qlibrary,
                )
                @test generated.info==generated.diagnostic==0
                @test thin[1:rows, :]≈(
                    rq ? q[(order - k + 1):order, :] : q[:, (order - k + 1):order]
                ) atol=tol rtol=tol
                @test all(==(T(777)), thin[(rows + 1):end, :])
                for side in ('L', 'R'), trans in ('N', T<:Complex ? 'C' : 'T')
                    rows, cols=side=='L' ? (order, 3) : (3, order)
                    c=fill(T(777), rows+2, cols)
                    c[1:rows, :]=randn(rng, T, rows, cols)
                    before=copy(c)
                    op=trans=='N' ? q : q'
                    expected=side=='L' ? op*before[1:rows, :] : before[1:rows, :]*op
                    r=direct_reverse_apply!(
                        opstorage,
                        tau,
                        c,
                        L,
                        Val(rq),
                        Val(gblocked);
                        side,
                        trans,
                        m=rows,
                        n=cols,
                        k,
                        library=qlibrary,
                    )
                    @test r.info==r.diagnostic==0
                    @test r.guard==fill(L(-777), 2)
                    @test r.work[[1, end]]==fill(T(777), 2)
                    @test c[1:rows, :]≈expected atol=tol rtol=tol
                    @test all(==(T(777)), c[(rows + 1):end, :])
                end
            end
        end
    end
end

@testset "QL/RQ queries, argument diagnostics and unused prefixes" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), rq in (false, true),
        blocked in (false, true)

        a=fill(T(NaN), 7, 3)
        minimum=rq ? 5 : 3
        cases=[((; m=-1), 1), ((; n=-1), 2), ((; lda=4), 4)]
        blocked&&push!(cases, ((; lwork=minimum-1), 7))
        for (kw, arg) in cases
            r=direct_reverse_factor!(copy(a), J, Val(rq), Val(blocked); m=5, n=3, kw...)
            @test r.info == -arg
            @test r.diagnostic == arg
            @test r.guard==fill(J(-777), 2)
        end
        if blocked
            before=copy(a)
            r=direct_reverse_factor!(a, J, Val(rq), Val(true); m=5, n=3, lwork=-1)
            @test r.info==r.diagnostic==0
            @test real(r.work[2])==minimum
            @test isequal(a, before)
        end
        m, n=rq ? (3, 5) : (5, 3)
        tau=zeros(T, 2)
        source=fill(T(NaN), m+2, n)
        cases=[((; m=-1), 1), ((; n=-1), 2), ((; k=-1), 3), ((; k=4), 3), ((; lda=m-1), 5)]
        blocked&&push!(cases, ((; lwork=2), 8))
        for (kw, arg) in cases
            r=direct_reverse_generate!(
                copy(source), tau, J, Val(rq), Val(blocked); m, n, k=2, kw...
            )
            @test r.info == -arg
            @test r.diagnostic == arg
            @test r.guard==fill(J(-777), 2)
        end
        if blocked
            before=copy(source)
            r=direct_reverse_generate!(
                source, tau, J, Val(rq), Val(true); m, n, k=2, lwork=-1
            )
            @test r.info==r.diagnostic==0
            @test real(r.work[2])==3
            @test isequal(source, before)
        end
        factors=fill(T(NaN), rq ? 4 : 7, rq ? 5 : 2)
        c=fill(T(NaN), 7, 3)
        cases=[
            ((; side=('?')), 1),
            ((; trans=('?')), 2),
            ((; m=-1), 3),
            ((; n=-1), 4),
            ((; k=-1), 5),
            ((; k=6), 5),
            ((; lda=rq ? 1 : 4), 7),
            ((; ldc=4), 10),
        ]
        blocked&&push!(cases, ((; lwork=2), 12))
        for (kw, arg) in cases
            before=copy(c)
            r=direct_reverse_apply!(
                factors, tau, c, J, Val(rq), Val(blocked); m=5, n=3, k=2, kw...
            )
            @test r.info == -arg
            @test r.diagnostic == arg
            @test r.guard==fill(J(-777), 2)
            @test isequal(c, before)
        end
        if blocked
            r=direct_reverse_apply!(
                factors, tau, c, J, Val(rq), Val(true); m=5, n=3, k=2, lwork=-1
            )
            @test r.info==r.diagnostic==0
            @test real(r.work[2])==3
        end
        huge=Int(typemax(J))
        before=copy(c)
        r=direct_reverse_apply!(
            factors, tau, c, J, Val(rq), Val(blocked); m=5, n=3, k=2, lda=huge, ldc=huge
        )
        @test r.info==r.diagnostic==0
        @test isequal(c, before)
        # Only a length-one prefix reflector is active. Neither A nor
        # later C columns are referenced, so both large strides are unused.
        factors=fill(T(NaN), 7, 5)
        coefficients=zeros(T, 5)
        coefficients[1]=T(2)
        c=ones(T, 3, 5)
        r=direct_reverse_apply!(
            factors,
            coefficients,
            c,
            J,
            Val(rq),
            Val(blocked);
            side=('R'),
            m=3,
            n=5,
            k=5,
            lda=huge,
            ldc=huge,
        )
        @test r.info==r.diagnostic==0
        @test c[:, 1]==fill(T(-1), 3)
        @test c[:, 2:end]==ones(T, 3, 4)
    end
end
@testset "All QL/RQ symbols through LBT" begin
    rng=MersenneTwister(6303)
    library=BLAS.libblastrampoline
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64), rq in (false, true),
        blocked in (false, true)

        org=T<:Complex ? "ung" : "org"
        orm=T<:Complex ? "unm" : "orm"
        families=(
            rq ? (blocked ? "gerqf" : "gerq2") : (blocked ? "geqlf" : "geql2"),
            org*(rq ? (blocked ? "rq" : "r2") : (blocked ? "ql" : "2l")),
            orm*(rq ? (blocked ? "rq" : "r2") : (blocked ? "ql" : "2l")),
        )
        for family in families
            provider=BLAS.lbt_find_backing_library(
                prefix*family*"_", J===Int32 ? :lp64 : :ilp64
            )
            @test provider!==nothing
            @test realpath(provider.libname)==realpath(Colmat.backend_library())
        end
        m, n=8, 3
        k=min(m, n)
        order=rq ? n : m
        tol=2000eps(real(T))
        original=randn(rng, T, m, n)
        a=copy(original)
        factor=direct_reverse_factor!(a, J, Val(rq), Val(blocked); m, n, library)
        @test factor.info==factor.diagnostic==0
        tau=view(factor.tau, 2:(length(factor.tau) - 1))
        triangular=[(rq ? j-i>=n-m : i-j>=m-n) ? a[i, j] : zero(T) for i in 1:m, j in 1:n]
        q=reverse_qr_storage(a, rq, m, n, k, order, order)
        r=direct_reverse_generate!(
            q, tau, J, Val(rq), Val(blocked); m=order, n=order, k, library
        )
        @test r.info==r.diagnostic==0
        q=q[1:order, :]
        @test (rq ? triangular*q : q*triangular)≈original atol=tol rtol=tol
        opstorage=reverse_qr_storage(a, rq, m, n, k, rq ? k : order, rq ? order : k)
        c=Matrix{T}(I, order, order)
        r=direct_reverse_apply!(
            opstorage, tau, c, J, Val(rq), Val(blocked); m=order, n=order, k, library
        )
        @test r.info==r.diagnostic==0
        @test c≈q atol=tol rtol=tol
    end
end

# Independent empty-factor WORK contract, including unused output guards.
for family in ("geqlf", "gerqf"),
    (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    I in (Int32, Int64)

    symbol=Symbol(p*family*"_"*(I===Int64 ? "64_" : ""))
    @eval function empty_reverse_qr_work_probe(
        ::Type{$T}, ::Type{$I}, ::Val{$(QuoteNode(Symbol(family)))}, library, m, n, lwork
    )
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol)))
        a=fill($T(777), 4)
        tau=fill($T(778), 4)
        work=fill($T(779), 4)
        info=Ref{$I}(-99)
        ccall(
            f,
            Cvoid,
            (Ref{$I}, Ref{$I}, Ptr{$T}, Ref{$I}, Ptr{$T}, Ptr{$T}, Ref{$I}, Ref{$I}),
            m,
            n,
            a,
            max(1, m),
            tau,
            work,
            lwork,
            info,
        )
        (info[], a, tau, work)
    end
end
@testset "Independent empty QL/RQ workspace and unused-output guards" begin
    reference=only(
        filter(
            l->l.interface==:ilp64&&occursin("openblas", lowercase(l.libname)),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
    for family in ("geqlf", "gerqf"),
        T in (Float32, Float64, ComplexF32, ComplexF64),
        I in (Int32, Int64),
        (m, n) in ((0, 0), (0, 3), (3, 0)),
        lw in (-1, 0, 1, 3)

        own=empty_reverse_qr_work_probe(
            T, I, Val(Symbol(family)), Colmat.backend_library(), m, n, lw
        )
        external=empty_reverse_qr_work_probe(
            T, Int64, Val(Symbol(family)), reference, m, n, lw
        )
        @test own==external
        @test own[1]==(lw==0 ? -7 : 0)
        @test own[2]==fill(T(777), 4)
        @test own[3]==fill(T(778), 4)
        @test own[4][1]==one(T)
        @test own[4][2:end]==fill(T(779), 3)
    end
end
