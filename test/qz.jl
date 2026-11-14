using Test, LinearAlgebra, Random, Libdl, Colmat
for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    R=typeof(real(zero(T)))
    symbol=Symbol(p*"hgeqz_"*(J===Int64 ? "64_" : ""))
    eigtypes=T<:Real ? [:(Ptr{$T}), :(Ptr{$T}), :(Ptr{$T})] : [:(Ptr{$T}), :(Ptr{$T})]
    eigargs=T<:Real ? [:ar, :ai, :beta] : [:ar, :beta]
    realtypes=T<:Real ? [] : [:(Ptr{$R})]
    realargs=T<:Real ? [] : [:rw]
    signature=Expr(
        :tuple,
        :(Ref{UInt8}),
        :(Ref{UInt8}),
        :(Ref{UInt8}),
        :(Ref{$J}),
        :(Ref{$J}),
        :(Ref{$J}),
        :(Ptr{$T}),
        :(Ref{$J}),
        :(Ptr{$T}),
        :(Ref{$J}),
        eigtypes...,
        :(Ptr{$T}),
        :(Ref{$J}),
        :(Ptr{$T}),
        :(Ref{$J}),
        :(Ptr{$T}),
        :(Ref{$J}),
        realtypes...,
        :(Ref{$J}),
    )
    @eval function direct_hgeqz!(
        a::Matrix{$T},
        b::Matrix{$T},
        q::Matrix{$T},
        z::Matrix{$T},
        ::Type{$J};
        lo=1,
        hi=size(a, 2),
        cq=('I'),
        cz=('I'),
        job=('S'),
        query=false,
        library=Colmat.backend_library(),
    )
        n=size(a, 2)
        ar=zeros($T, n)
        ai=zeros($T, n)
        beta=zeros($T, n)
        work=zeros($T, max(1, n))
        rw=zeros($R, n)
        info=Ref{$J}(0)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol)))
        ccall(
            f,
            Cvoid,
            $signature,
            job,
            cq,
            cz,
            n,
            lo,
            hi,
            a,
            max(1, size(a, 1)),
            b,
            max(1, size(b, 1)),
            $(eigargs...),
            q,
            max(1, size(q, 1)),
            z,
            max(1, size(z, 1)),
            work,
            query ? -1 : length(work),
            $(realargs...),
            info,
        )
        alpha=$T<:Real ? complex.(ar, ai) : ar
        return alpha, beta, info[], work
    end
end
@testset "HGEQZ both ABIs and active pencils" begin
    Random.seed!(294)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        (n, lo, hi) in ((0, 1, 0), (1, 1, 1), (2, 1, 2), (7, 1, 7), (7, 2, 6), (40, 1, 40))

        original=triu(randn(T, n, n), -1)
        triangular=triu(randn(T, n, n))
        lo>1 && (original[lo, lo - 1]=0)
        hi<n && (original[hi + 1, hi]=0)
        for comp in ('I', 'V')
            a=copy(original)
            b=copy(triangular)
            q0=Matrix(qr(randn(T, n, n)).Q)
            z0=Matrix(qr(randn(T, n, n)).Q)
            q=comp=='V' ? copy(q0) : zeros(T, n, n)
            z=comp=='V' ? copy(z0) : zeros(T, n, n)
            alpha, beta, info, _=direct_hgeqz!(a, b, q, z, J; lo, hi, cq=comp, cz=comp)
            @test info==0
            left=comp=='V' ? q0'*q : q
            right=comp=='V' ? z0'*z : z
            @test left*a*right' ≈ original
            @test left*b*right' ≈ triangular
            @test q'*q ≈ Matrix{T}(I, n, n)
            @test z'*z ≈ Matrix{T}(I, n, n)
            @test norm(tril(a, T<:Real ? -2 : -1))==0
            @test norm(tril(b, -1))==0
            @test all(real.(diag(b)) .>= 0)
            @test all(real.(beta) .>= 0)
            @test all(isfinite, alpha) && all(isfinite, beta)
        end
    end
end

const qz_callback_calls=Ref(0)
for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    R=typeof(real(zero(T)))
    name=Symbol("select_qz_", p, "_", J)
    callback=Symbol(name, "_pointer")
    if T<:Real
        @eval function $name(ar::Ptr{$T}, ai::Ptr{$T}, beta::Ptr{$T})::$J
            qz_callback_calls[]+=1
            return unsafe_load(ar)*unsafe_load(beta)<0 ? typemax($J) : 0
        end
        @eval const $callback=@cfunction($(name), $J, (Ptr{$T}, Ptr{$T}, Ptr{$T}))
    else
        @eval function $name(alpha::Ptr{$T}, beta::Ptr{$T})::$J
            qz_callback_calls[]+=1
            return real(unsafe_load(alpha)*conj(unsafe_load(beta)))<0 ? typemax($J) : 0
        end
        @eval const $callback=@cfunction($(name), $J, (Ptr{$T}, Ptr{$T}))
    end
    for family in (:gges, :gges3, :gegs)
        symbol=Symbol(p*string(family)*"_"*(J===Int64 ? "64_" : ""))
        eigtypes=T<:Real ? [:(Ptr{$T}), :(Ptr{$T}), :(Ptr{$T})] : [:(Ptr{$T}), :(Ptr{$T})]
        eigargs=T<:Real ? [:ar, :ai, :beta] : [:ar, :beta]
        sorttypes=family===:gegs ? [] : [:(Ref{UInt8}), :(Ptr{Cvoid})]
        sortargs=if family===:gegs
            []
        else
            [:(sorting ? 'S' : 'N'), :(use_callback ? selection : C_NULL)]
        end
        dimtypes=family===:gegs ? [] : [:(Ref{$J})]
        dimargs=family===:gegs ? [] : [:sdim]
        rwtypes=T<:Real ? [] : [:(Ptr{$R})]
        rwargs=T<:Real ? [] : [:rw]
        bwtypes=family===:gegs ? [] : [:(Ptr{$J})]
        bwargs=family===:gegs ? [] : [:bw]
        signature=Expr(
            :tuple,
            :(Ref{UInt8}),
            :(Ref{UInt8}),
            sorttypes...,
            :(Ref{$J}),
            :(Ptr{$T}),
            :(Ref{$J}),
            :(Ptr{$T}),
            :(Ref{$J}),
            dimtypes...,
            eigtypes...,
            :(Ptr{$T}),
            :(Ref{$J}),
            :(Ptr{$T}),
            :(Ref{$J}),
            :(Ptr{$T}),
            :(Ref{$J}),
            rwtypes...,
            bwtypes...,
            :(Ref{$J}),
        )
        minwork=T<:Real ? (
            if family===:gegs
                :(max(1, 4n))
            elseif family===:gges
                :(n==0 ? 1 : max(8n, 6n+16))
            else
                :(n==0 ? 1 : 6n+16)
            end
        ) : :(max(1, 2n))
        @eval function direct_gges!(
            a::Matrix{$T},
            b::Matrix{$T},
            q::Matrix{$T},
            z::Matrix{$T},
            ::Type{$J},
            ::Val{$(QuoteNode(family))};
            jq=('V'),
            jz=('V'),
            sorting=false,
            query=false,
            use_callback=sorting,
            selection=($callback),
            library=Colmat.backend_library(),
        )
            n=size(a, 2)
            ar=zeros($T, n)
            ai=zeros($T, n)
            beta=zeros($T, n)
            work=zeros($T, $minwork)
            rw=zeros($R, 8n)
            bw=zeros($J, n)
            sdim=Ref{$J}(0)
            info=Ref{$J}(0)
            f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol)))
            ccall(
                f,
                Cvoid,
                $signature,
                jq,
                jz,
                $(sortargs...),
                n,
                a,
                max(1, size(a, 1)),
                b,
                max(1, size(b, 1)),
                $(dimargs...),
                $(eigargs...),
                q,
                max(1, size(q, 1)),
                z,
                max(1, size(z, 1)),
                work,
                query ? -1 : length(work),
                $(rwargs...),
                $(bwargs...),
                info,
            )
            alpha=$T<:Real ? complex.(ar, ai) : ar
            return alpha, beta, sdim[], info[], work
        end
    end
end

@testset "GGES/GGES3/GEGS direct ABI and selection" begin
    Random.seed!(2941)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), family in (:gges, :gges3, :gegs), n in (0, 1, 6),
        sorting in (false, true)

        family===:gegs && sorting && continue
        original=randn(T, n, n)
        original_b=randn(T, n, n)
        a=copy(original)
        b=copy(original_b)
        q=zeros(T, n, n)
        z=copy(q)
        calls=qz_callback_calls[]
        alpha, beta, sdim, info, _=direct_gges!(a, b, q, z, J, Val(family); sorting)
        @test info==0
        @test q*a*z' ≈ original
        @test q*b*z' ≈ original_b
        @test q'*q ≈ Matrix{T}(I, n, n)
        @test z'*z ≈ Matrix{T}(I, n, n)
        @test all(real.(beta) .>= 0)
        @test norm(tril(a, T<:Real ? -2 : -1))==0
        @test norm(tril(b, -1))==0
        if sorting
            selected=real.(alpha .* conj.(beta)) .< 0
            @test sdim==sum(selected)
            @test all(selected[1:sdim]) && !any(selected[(sdim + 1):end])
        else
            @test sdim==0 && qz_callback_calls[]==calls
        end
        calls=qz_callback_calls[]
        saved=map(copy, (a, b, q, z))
        _, _, _, info, work=direct_gges!(
            a, b, q, z, J, Val(family); sorting, query=true, use_callback=false
        )
        @test info==0 && real(work[1])>=1
        @test qz_callback_calls[]==calls
        @test all(isequal.(saved, (a, b, q, z)))
    end
end
