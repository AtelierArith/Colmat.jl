using Test, LinearAlgebra, Random, Libdl, Colmat

function bidiagonal_diagnostic()
    return ccall(
        Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error),
        Cint,
        (),
    )
end
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    R=real(T)
    suffix=J===Int64 ? "64_" : ""
    for family in (:gebd2, :gebrd)
        symbol=Symbol(prefix*string(family)*"_"*suffix)
        signature=Any[
            Ref{J}, Ref{J}, Ptr{T}, Ref{J}, Ptr{R}, Ptr{R}, Ptr{T}, Ptr{T}, Ptr{T}
        ]
        args=Any[:m, :n, :a, :lda, :dp, :ep, :qp, :pp, :wp]
        family==:gebrd&&(push!(signature, Ref{J}); push!(args, :lw))
        push!(signature, Ptr{J})
        push!(args, :ip)
        @eval function direct_bidiagonal!(
            a::Matrix{$T},
            ::Type{$J},
            ::Val{$(QuoteNode(family))};
            m,
            n,
            lda=size(a, 1),
            lwork=nothing,
            library=Colmat.backend_library(),
        )
            k=max(0, min(m, n))
            minimum=k==0 ? 1 : max(m, n)
            lw=isnothing(lwork) ? minimum : lwork
            d=fill($R(777), max(1, k)+2)
            e=copy(d)
            tq=fill($T(777), max(1, k)+2)
            tp=copy(tq)
            work=fill($T(777), max(1, lw)+2)
            info=fill($J(-777), 3)
            dp=view(d, 2:(length(d) - 1))
            ep=view(e, 2:(length(e) - 1))
            qp=view(tq, 2:(length(tq) - 1))
            pp=view(tp, 2:(length(tp) - 1))
            wp=view(work, 2:(length(work) - 1))
            ip=view(info, 2:2)
            f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol)))
            ccall(f, Cvoid, $(Expr(:tuple, signature...)), $(args...))
            return (;
                d,
                e,
                tq,
                tp,
                work,
                info=info[2],
                guard=info[[1, 3]],
                diagnostic=bidiagonal_diagnostic(),
            )
        end
    end
    symbol=Symbol(prefix*(T<:Complex ? "ungbr_" : "orgbr_")*suffix)
    @eval function direct_bidiagonal_generate!(
        a::Matrix{$T},
        tau::AbstractVector{$T},
        ::Type{$J};
        vect=('Q'),
        m,
        n,
        k,
        lda=size(a, 1),
        lwork=nothing,
        library=Colmat.backend_library(),
    )
        lw=isnothing(lwork) ? max(1, min(m, n)) : lwork
        work=fill($T(777), max(1, lw)+2)
        info=fill($J(-777), 3)
        wp=view(work, 2:(length(work) - 1))
        ip=view(info, 2:2)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol)))
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{$J},
                Ref{$J},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$T},
                Ptr{$T},
                Ref{$J},
                Ptr{$J},
            ),
            vect,
            m,
            n,
            k,
            a,
            lda,
            tau,
            wp,
            lw,
            ip,
        )
        return (;
            work, info=info[2], guard=info[[1, 3]], diagnostic=bidiagonal_diagnostic()
        )
    end
    symbol=Symbol(prefix*(T<:Complex ? "unmbr_" : "ormbr_")*suffix)
    @eval function direct_bidiagonal_apply!(
        a::Matrix{$T},
        tau::AbstractVector{$T},
        c::Matrix{$T},
        ::Type{$J};
        vect=('Q'),
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
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol)))
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{UInt8},
                Ref{UInt8},
                Ref{$J},
                Ref{$J},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$T},
                Ptr{$T},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$J},
            ),
            vect,
            side,
            trans,
            m,
            n,
            k,
            a,
            lda,
            tau,
            c,
            ldc,
            wp,
            lw,
            ip,
        )
        return (;
            work, info=info[2], guard=info[[1, 3]], diagnostic=bidiagonal_diagnostic()
        )
    end
    symbol=Symbol(prefix*"labrd_"*suffix)
    @eval function direct_bidiagonal_panel!(
        a::Matrix{$T},
        ::Type{$J};
        m,
        n,
        nb,
        lda=size(a, 1),
        ldx=max(1, m+2),
        ldy=max(1, n+3),
        library=Colmat.backend_library(),
    )
        count=max(1, nb)
        d=fill($R(777), count+2)
        e=copy(d)
        tq=fill($T(777), count+2)
        tp=copy(tq)
        xrows=nb<=1 ? max(1, m+2) : max(1, ldx)
        yrows=nb<=1 ? max(1, n+3) : max(1, ldy)
        xb=fill($T(777), xrows*count+2)
        yb=fill($T(777), yrows*count+2)
        dp=view(d, 2:(length(d) - 1))
        ep=view(e, 2:(length(e) - 1))
        qp=view(tq, 2:(length(tq) - 1))
        pp=view(tp, 2:(length(tp) - 1))
        xp=view(xb, 2:(length(xb) - 1))
        yp=view(yb, 2:(length(yb) - 1))
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol)))
        ccall(
            f,
            Cvoid,
            (
                Ref{$J},
                Ref{$J},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$R},
                Ptr{$R},
                Ptr{$T},
                Ptr{$T},
                Ptr{$T},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
            ),
            m,
            n,
            nb,
            a,
            lda,
            dp,
            ep,
            qp,
            pp,
            xp,
            ldx,
            yp,
            ldy,
        )
        return (;
            d,
            e,
            tq,
            tp,
            xb,
            yb,
            x=reshape(xp, xrows, count),
            y=reshape(yp, yrows, count),
            diagnostic=bidiagonal_diagnostic(),
        )
    end
end

function bidiagonal_product(a, b)
    T=promote_type(eltype(a), eltype(b))
    return [
        sum((a[i, k]*b[k, j] for k in axes(a, 2)); init=zero(T)) for
        i in axes(a, 1), j in axes(b, 2)
    ]
end
function bidiagonal_storage(original)
    m, n=size(original)
    a=fill(eltype(original)(777), max(1, m+3), max(1, n))
    a[1:m, 1:n]=original
    return a
end
function bidiagonal_vectors(a, r, m, n, J; library=Colmat.backend_library())
    T=eltype(a)
    q=fill(T(777), m+3, max(1, m))
    q[1:m, 1:m].=T(NaN)
    ph=fill(T(777), n+3, max(1, n))
    ph[1:n, 1:n].=T(NaN)
    q[1:m, 1:min(m, n)]=a[1:m, 1:min(m, n)]
    ph[1:min(m, n), 1:n]=a[1:min(m, n), 1:n]
    rq=direct_bidiagonal_generate!(
        q, view(r.tq, 2:(length(r.tq) - 1)), J; vect=('Q'), m, n=m, k=n, library
    )
    rp=direct_bidiagonal_generate!(
        ph, view(r.tp, 2:(length(r.tp) - 1)), J; vect=('P'), m=n, n, k=m, library
    )
    @test rq.info==rp.info==0
    @test rq.guard==rp.guard==J[-777, -777]
    @test all(==(T(777)), q[(m + 1):end, :])
    @test all(==(T(777)), ph[(n + 1):end, :])
    return q[1:m, 1:m], ph[1:n, 1:n]
end
function bidiagonal_reference()
    return only(
        filter(
            l->l.interface==:ilp64&&occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
end

@testset "Native bidiagonal factors and external Q/P interoperability" begin
    reference=bidiagonal_reference()
    rng=MersenneTwister(295)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        (m, n) in ((0, 4), (4, 0), (1, 1), (1, 4), (4, 1), (4, 7), (7, 4), (5, 5)),
        family in (:gebd2, :gebrd),
        mode in (:dense, :rankone, :zero, :diagonal)

        R=real(T)
        tol=R===Float32 ? 5e-4 : 5e-11
        k=min(m, n)
        original=if mode==:dense
            randn(rng, T, m, n)
        elseif mode==:rankone
            bidiagonal_product(randn(rng, T, m, 1), randn(rng, T, 1, n))
        elseif mode==:diagonal
            [i==j ? T(i+1) : zero(T) for i in 1:m, j in 1:n]
        else
            zeros(T, m, n)
        end
        a=bidiagonal_storage(original)
        external_a=copy(a)
        own=direct_bidiagonal!(a, J, Val(family); m, n)
        external=direct_bidiagonal!(external_a, Int64, Val(family); m, n, library=reference)
        @test own.info==external.info==0
        @test own.diagnostic==0
        @test own.guard==J[-777, -777]
        @test own.d[2:(k + 1)]≈external.d[2:(k + 1)] atol=tol rtol=tol
        @test own.e[2:k]≈external.e[2:k] atol=tol rtol=tol
        @test all(==(T(777)), a[(m + 1):end, :])
        @test own.d[[1, end]]==R[777, 777]
        @test own.e[[1, end]]==R[777, 777]
        @test own.tq[[1, end]]==T[777, 777]
        @test own.tp[[1, end]]==T[777, 777]
        @test own.work[[1, end]]==T[777, 777]
        for (factor, storage) in ((own, a), (external, external_a))
            q, ph=bidiagonal_vectors(storage, factor, m, n, J)
            reference_q, reference_ph=bidiagonal_vectors(
                storage, factor, m, n, Int64; library=reference
            )
            @test q≈reference_q atol=tol rtol=tol
            @test ph≈reference_ph atol=tol rtol=tol
            if m>n
                thin=copy(storage)
                thin_result=direct_bidiagonal_generate!(
                    thin,
                    view(factor.tq, 2:(length(factor.tq) - 1)),
                    J;
                    vect=('Q'),
                    m,
                    n,
                    k=n,
                )
                @test thin_result.info==0
                @test thin[1:m, 1:n]≈q[:, 1:n] atol=tol rtol=tol
                @test all(==(T(777)), thin[(m + 1):end, :])
            elseif m<n
                thin=copy(storage)
                thin_result=direct_bidiagonal_generate!(
                    thin,
                    view(factor.tp, 2:(length(factor.tp) - 1)),
                    J;
                    vect=('P'),
                    m,
                    n,
                    k=m,
                )
                @test thin_result.info==0
                @test thin[1:m, 1:n]≈ph[1:m, :] atol=tol rtol=tol
                @test all(==(T(777)), thin[(m + 1):end, :])
            end
            @test bidiagonal_product(q', q)≈Matrix{T}(I, m, m) atol=tol rtol=tol
            @test bidiagonal_product(ph, ph')≈Matrix{T}(I, n, n) atol=tol rtol=tol
            b=zeros(T, m, n)
            for i in 1:k
                b[i, i]=factor.d[i + 1]
            end
            for i in 1:(k - 1)
                if m>=n
                    b[i, i + 1]=factor.e[i + 1]
                else
                    b[i + 1, i]=factor.e[i + 1]
                end
            end
            @test bidiagonal_product(bidiagonal_product(q, b), ph)≈original atol=tol rtol=tol
        end
    end
end

@testset "Bidiagonal Q/P left right transpose and external factors" begin
    reference=bidiagonal_reference()
    rng=MersenneTwister(1295)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), (rows, cols) in ((5, 3), (3, 5), (4, 4), (1, 3), (3, 1)),
        source in (:native, :external)

        R=real(T)
        tol=R===Float32 ? 5e-4 : 5e-11
        original=randn(rng, T, rows, cols)
        a=bidiagonal_storage(original)
        factor=direct_bidiagonal!(
            a,
            Int64,
            Val(:gebrd);
            m=rows,
            n=cols,
            library=source==:native ? Colmat.backend_library() : reference,
        )
        q, ph=bidiagonal_vectors(a, factor, rows, cols, Int64; library=reference)
        for vect in ('Q', 'P'),
            side in ('L', 'R'),
            trans in (T<:Complex ? ('N', 'C') : ('N', 'T'))

            order=vect=='Q' ? rows : cols
            k=vect=='Q' ? cols : rows
            operator=vect=='Q' ? q : ph'
            op=trans=='N' ? operator : operator'
            m, n=side=='L' ? (order, 3) : (3, order)
            input=randn(rng, T, m, n)
            c=bidiagonal_storage(input)
            external_c=copy(c)
            tau=view(vect=='Q' ? factor.tq : factor.tp, 2:(length(factor.tq) - 1))
            result=direct_bidiagonal_apply!(a, tau, c, J; vect, side, trans, m, n, k)
            external=direct_bidiagonal_apply!(
                a, tau, external_c, Int64; vect, side, trans, m, n, k, library=reference
            )
            expected=if side=='L'
                bidiagonal_product(op, input)
            else
                bidiagonal_product(input, op)
            end
            @test result.info==external.info==0
            @test result.diagnostic==0
            @test result.guard==J[-777, -777]
            @test c[1:m, 1:n]≈expected atol=tol rtol=tol
            @test c[1:m, 1:n]≈external_c[1:m, 1:n] atol=tol rtol=tol
            @test all(==(T(777)), c[(m + 1):end, :])
            @test result.work[[1, end]]==T[777, 777]
        end
    end
end

@testset "LABRD preserves A22 and exposes interoperable deferred updates" begin
    reference=bidiagonal_reference()
    rng=MersenneTwister(2295)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        (m, n) in ((0, 4), (4, 0), (1, 1), (1, 4), (4, 1), (4, 6), (6, 4), (5, 5)),
        nb in 0:min(m, n),
        mode in (:dense, :zero)

        R=real(T)
        tol=R===Float32 ? 8e-4 : 8e-11
        original=mode==:dense ? randn(rng, T, m, n) : zeros(T, m, n)
        a=bidiagonal_storage(original)
        external_a=copy(a)
        result=direct_bidiagonal_panel!(a, J; m, n, nb)
        external=direct_bidiagonal_panel!(external_a, Int64; m, n, nb, library=reference)
        @test result.diagnostic==0
        @test result.d[2:(nb + 1)]≈external.d[2:(nb + 1)] atol=tol rtol=tol
        active_e=min(nb, max(0, min(m, n)-1))
        @test result.e[2:(active_e + 1)]≈external.e[2:(active_e + 1)] atol=tol rtol=tol
        @test a[(nb + 1):m, (nb + 1):n]==original[(nb + 1):m, (nb + 1):n]
        @test all(==(T(777)), a[(m + 1):end, :])
        for values in (result.d, result.e)
            @test values[[1, end]]==R[777, 777]
        end
        for values in (result.tq, result.tp, result.xb, result.yb)
            @test values[[1, end]]==T[777, 777]
        end
        @test all(==(T(777)), result.x[(m + 1):end, :])
        @test all(==(T(777)), result.y[(n + 1):end, :])
        for column in 1:nb
            if m<n||column<n
                @test result.x[(column + 1):m, column]≈external.x[(column + 1):m, column] atol=tol rtol=tol
            end
            if m>=n||column<m
                @test result.y[(column + 1):n, column]≈external.y[(column + 1):n, column] atol=tol rtol=tol
            end
        end
        q=Matrix{T}(I, m, m)
        p=Matrix{T}(I, n, n)
        for column in 1:nb
            start=m>=n ? column : column+1
            if start<=m
                v=zeros(T, m)
                v[start]=one(T)
                v[(start + 1):m]=a[(start + 1):m, column]
                h=[
                    (i==j ? one(T) : zero(T))-result.tq[column + 1]*v[i]*conj(v[j]) for
                    i in 1:m, j in 1:m
                ]
                q=bidiagonal_product(q, h)
            end
            start=m>=n ? column+1 : column
            if start<=n
                u=zeros(T, n)
                u[start]=one(T)
                u[(start + 1):n]=conj.(a[column, (start + 1):n])
                g=[
                    (i==j ? one(T) : zero(T))-result.tp[column + 1]*u[i]*conj(u[j]) for
                    i in 1:n, j in 1:n
                ]
                p=bidiagonal_product(p, g)
            end
        end
        transformed=bidiagonal_product(bidiagonal_product(q', original), p)
        expected=copy(original)
        for column in 1:n, row in 1:m
            if row<=nb||column<=nb
                expected[row, column]=if row==column
                    T(result.d[row + 1])
                elseif m>=n&&column==row+1
                    T(result.e[row + 1])
                elseif m<n&&row==column+1
                    T(result.e[column + 1])
                else
                    zero(T)
                end
            else
                for stage in 1:nb
                    vhead=m>=n ? stage : stage+1
                    uhead=m>=n ? stage+1 : stage
                    v=row==vhead ? one(T) : a[row, stage]
                    uh=column==uhead ? one(T) : a[stage, column]
                    expected[row, column]-=v*conj(result.y[column, stage])+result.x[
                        row, stage
                    ]*uh
                end
            end
        end
        @test transformed≈expected atol=tol rtol=tol
    end
end

@testset "Bidiagonal workspace queries INFO and unused reflector storage" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        a=fill(T(NaN), 7, 4)
        tau=zeros(T, 4)
        R=real(T)
        for family in (:gebd2, :gebrd)
            for (options, status) in (((; m=-1), -1), ((; n=-1), -2), ((; lda=3), -4))
                result=direct_bidiagonal!(a, J, Val(family); (; m=4, n=3, options...)...)
                @test result.info==status
                @test result.diagnostic == -status
                @test result.guard==J[-777, -777]
                @test all(isnan, a)
            end
        end
        result=direct_bidiagonal!(a, J, Val(:gebrd); m=4, n=3, lwork=-1)
        @test result.info==0
        @test result.diagnostic==0
        @test real(result.work[2])>=4
        @test all(isnan, a)
        @test all(==(R(777)), result.d)
        @test all(==(T(777)), result.tq)
        result=direct_bidiagonal!(a, J, Val(:gebrd); m=4, n=3, lwork=3)
        @test result.info == -10
        @test result.diagnostic==10
        for vect in ('Q', 'P')
            result=direct_bidiagonal_generate!(a, tau, J; vect, m=4, n=4, k=2, lwork=-1)
            @test result.info==0
            @test real(result.work[2])>=4
            @test all(isnan, a)
            cases=[
                ((; vect=('?')), -1),
                ((; m=-1), -2),
                ((; n=-1), -3),
                ((; k=-1), -4),
                ((; lda=3), -6),
                ((; lwork=3), -9),
            ]
            push!(cases, vect=='Q' ? ((; n=5), -3) : ((; m=5), -3))
            for (options, status) in cases
                result=direct_bidiagonal_generate!(
                    a, tau, J; (; vect, m=4, n=4, k=2, options...)...
                )
                @test result.info==status
                @test result.diagnostic == -status
                @test result.guard==J[-777, -777]
                @test all(isnan, a)
            end
            generated=copy(a)
            result=direct_bidiagonal_generate!(generated, tau, J; vect, m=4, n=4, k=2)
            @test result.info==0
            @test generated[1:4, 1:4]==Matrix{T}(I, 4, 4)
            for side in ('L', 'R')
                c=fill(T(NaN), 7, 3)
                query=direct_bidiagonal_apply!(
                    a, tau, c, J; vect, side, m=4, n=3, k=2, lwork=-1
                )
                @test query.info==0
                @test all(isnan, c)
                @test all(isnan, a)
                for (options, status) in (
                    ((; vect=('?')), -1),
                    ((; side=('?')), -2),
                    ((; trans=('?')), -3),
                    ((; m=-1), -4),
                    ((; n=-1), -5),
                    ((; k=-1), -6),
                    ((; lda=1), -8),
                    ((; ldc=3), -11),
                    ((; lwork=0), -13),
                )
                    result=direct_bidiagonal_apply!(
                        a, tau, c, J; (; vect, side, m=4, n=3, k=2, options...)...
                    )
                    @test result.info==status
                    @test result.diagnostic == -status
                    @test result.guard==J[-777, -777]
                    @test all(isnan, c)
                end
                c=fill(T(2), 7, 3)
                before=copy(c)
                result=direct_bidiagonal_apply!(a, tau, c, J; vect, side, m=4, n=3, k=2)
                @test result.info==0
                @test c==before
            end
        end
        for (options, status) in (
            ((; m=-1), 1),
            ((; n=-1), 2),
            ((; nb=-1), 3),
            ((; nb=5), 3),
            ((; lda=3), 5),
            ((; ldx=3), 11),
            ((; ldy=2), 13),
        )
            result=direct_bidiagonal_panel!(a, J; (; m=4, n=3, nb=2, options...)...)
            @test result.diagnostic==status
            @test all(isnan, a)
            @test all(==(R(777)), result.d)
        end
        huge=typemax(J)
        query=direct_bidiagonal!(a, J, Val(:gebrd); m=huge, n=1, lda=huge, lwork=-1)
        @test query.info==0
        @test real(query.work[2])>=big(huge)
        @test all(isnan, a)
    end
end

@testset "All 20 bidiagonal symbols compute through both LBT integer ABIs" begin
    rng=MersenneTwister(3295)
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        prefix=if T===Float32
            "s"
        elseif T===Float64
            "d"
        elseif T===ComplexF32
            "c"
        else
            "z"
        end
        for family in (
            "gebd2",
            "gebrd",
            "labrd",
            T<:Complex ? "ungbr" : "orgbr",
            T<:Complex ? "unmbr" : "ormbr",
        )
            provider=BLAS.lbt_find_backing_library(
                prefix*family*"_", J===Int32 ? :lp64 : :ilp64
            )
            @test provider!==nothing
            @test realpath(provider.libname)==realpath(Colmat.backend_library())
        end
        original=randn(rng, T, 4, 3)
        for family in (:gebd2, :gebrd)
            a=bidiagonal_storage(original)
            factor=direct_bidiagonal!(
                a, J, Val(family); m=4, n=3, library=BLAS.libblastrampoline
            )
            @test factor.info==0
            q, ph=bidiagonal_vectors(a, factor, 4, 3, J; library=BLAS.libblastrampoline)
            c=bidiagonal_storage(Matrix{T}(I, 4, 4))
            r=direct_bidiagonal_apply!(
                a,
                view(factor.tq, 2:4),
                c,
                J;
                vect=('Q'),
                side=('L'),
                m=4,
                n=4,
                k=3,
                library=BLAS.libblastrampoline,
            )
            @test r.info==0
            @test c[1:4, 1:4]≈q
        end
        panel=direct_bidiagonal_panel!(
            bidiagonal_storage(original), J; m=4, n=3, nb=1, library=BLAS.libblastrampoline
        )
        @test panel.diagnostic==0
        @test isfinite(panel.d[2])
    end
end

@testset "Bidiagonal scaled factors against BigFloat reconstruction" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), family in (:gebd2, :gebrd),
        wide in (false, true)

        R=real(T)
        tol=R===Float32 ? 8e-4 : 8e-11
        imaginary=T<:Complex ? T(im) : zero(T)
        base=T[2 0.5; -1 3; 0.25 -0.75]+imaginary .* R[1 0.25; 0.5 -1; 0.75 0.5]
        for scale in (R===Float32 ? (R(1e-38), R(1e30)) : (R(1e-310), R(1e300)))
            original=Matrix((wide ? base' : base) .* scale)
            m, n=size(original)
            k=min(m, n)
            a=bidiagonal_storage(original)
            factor=direct_bidiagonal!(a, J, Val(family); m, n)
            @test factor.info==0
            @test all(isfinite, factor.d[2:(k + 1)])
            @test all(isfinite, factor.e[2:k])
            q, ph=bidiagonal_vectors(a, factor, m, n, J)
            B=fill(big(zero(T)), m, n)
            for i in 1:k
                B[i, i]=big(factor.d[i + 1])/big(scale)
            end
            for i in 1:(k - 1)
                if m>=n
                    B[i, i + 1]=big(factor.e[i + 1])/big(scale)
                else
                    B[i + 1, i]=big(factor.e[i + 1])/big(scale)
                end
            end
            rebuilt=bidiagonal_product(bidiagonal_product(big.(q), B), big.(ph))
            @test maximum(abs, rebuilt-big.(original) ./ big(scale))<BigFloat(tol)
        end
    end
end

@testset "Single-column bidiagonal operations ignore unused huge leading dimensions" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        huge=typemax(J)
        for family in (:gebd2, :gebrd)
            original=reshape(T[3, 4], 2, 1)
            a=bidiagonal_storage(original)
            r=direct_bidiagonal!(a, J, Val(family); m=2, n=1, lda=huge)
            @test r.info==0
            @test r.diagnostic==0
            @test r.d[2]==-5
            @test all(==(T(777)), a[3:end, :])
            generated=copy(a)
            g=direct_bidiagonal_generate!(
                generated, view(r.tq, 2:2), J; vect=('Q'), m=2, n=1, k=1, lda=huge
            )
            @test g.info==0
            @test generated[1:2, 1]≈T[-0.6, -0.8]
            c=bidiagonal_storage(reshape(T[2, 3], 2, 1))
            applied=direct_bidiagonal_apply!(
                a, view(r.tq, 2:2), c, J; vect=('Q'), m=2, n=1, k=1, lda=huge, ldc=huge
            )
            @test applied.info==0
            @test c[1:2, 1]≈T[-3.6, 0.2]
            @test all(==(T(777)), c[3:end, :])
        end
        a=bidiagonal_storage(reshape(T[3, 4], 2, 1))
        panel=direct_bidiagonal_panel!(a, J; m=2, n=1, nb=1, lda=huge, ldx=huge, ldy=huge)
        @test panel.diagnostic==0
        @test panel.d[2]==-5
    end
end
