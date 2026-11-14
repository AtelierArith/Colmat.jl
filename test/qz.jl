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
        lwork_override=nothing,
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
            if query
                -1
            elseif isnothing(lwork_override)
                length(work)
            else
                lwork_override
            end,
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
            lwork_override=nothing,
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
                if query
                    -1
                elseif isnothing(lwork_override)
                    length(work)
                else
                    lwork_override
                end,
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

function qz_pair_distance(a, b, c, d)
    x=hypot(abs(a), abs(b))
    y=hypot(abs(c), abs(d))
    if x==0 || y==0
        return x==y ? 0.0 : 1.0
    end
    return abs((a/x)*(d/y)-(b/x)*(c/y))
end
@testset "QZ independent homogeneous spectrum and infinite values" begin
    library=only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
    Random.seed!(2942)
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        R=typeof(real(zero(T)))
        for singular in (false, true)
            n=7
            original=triu(randn(T, n, n), -1)
            triangular=triu(randn(T, n, n))
            singular && (triangular[3, 3]=0)
            a=copy(original)
            b=copy(triangular)
            q=zeros(T, n, n)
            z=copy(q)
            alpha, beta, info, _=direct_hgeqz!(a, b, q, z, J)
            refa=copy(original)
            refb=copy(triangular)
            refq=zeros(T, n, n)
            refz=copy(refq)
            ar, be, ri, _=direct_hgeqz!(refa, refb, refq, refz, Int64; library)
            @test info==0 && ri==0
            pool=collect(zip(ar, be))
            for (value, denominator) in zip(alpha, beta)
                distances=[qz_pair_distance(value, denominator, c, d) for (c, d) in pool]
                index=argmin(distances)
                @test distances[index]<=5000eps(R)
                deleteat!(pool, index)
            end
            if singular
                @test any(iszero, beta)
            end
        end
        original=T[2 1 0 0; 0 -1 -3 1; 0 3 -1 1; 0 0 0 4]
        original_b=Matrix{T}(I, 4, 4)
        original_b[4, 4]=0
        for family in (:gges, :gges3, :gegs)
            a=copy(original)
            b=copy(original_b)
            q=zeros(T, 4, 4)
            z=copy(q)
            alpha, beta, sdim, info, _=direct_gges!(
                a, b, q, z, J, Val(family); sorting=family!==:gegs
            )
            @test info==0
            @test count(iszero, beta)==1
            @test q*a*z' ≈ original
            @test q*b*z' ≈ original_b
            if family!==:gegs
                @test sdim==2
                @test all(real.(alpha[1:2] ./ beta[1:2]) .< 0)
            end
            if T<:Real
                indices=findall(!iszero, imag.(alpha))
                @test length(indices)==2
                @test imag(alpha[first(indices)])>0 && imag(alpha[last(indices)])<0
            end
        end
    end
end

@testset "QZ guards, workspace limits, unused factors and Julia wrappers" begin
    Random.seed!(2943)
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        n=5
        original=randn(T, n, n)
        original_b=randn(T, n, n)
        for family in (:gges, :gges3, :gegs)
            a=fill(T(789), n+2, n)
            b=copy(a)
            a[1:n, :]=original
            b[1:n, :]=original_b
            q=fill(T(NaN), n+2, n)
            z=copy(q)
            _, _, sdim, info, _=direct_gges!(a, b, q, z, J, Val(family); jq=('N'), jz=('N'))
            @test info==0 && sdim==0
            @test all(isnan, q) && all(isnan, z)
            @test all(==(T(789)), a[(n + 1):end, :]) && all(==(T(789)), b[(n + 1):end, :])
            saved=map(copy, (a, b, q, z))
            _, _, _, info, _=direct_gges!(a, b, q, z, J, Val(family); lwork_override=0)
            expected=T<:Real ? (family===:gegs ? -16 : -19) : (family===:gegs ? -15 : -18)
            @test info==expected
            @test all(isequal.(saved, (a, b, q, z)))
            _, _, _, info, _=direct_gges!(a, b, q, z, J, Val(family); jq=('?'))
            @test info == -1
            @test all(isequal.(saved, (a, b, q, z)))
        end
        a=fill(T(789), n+2, n)
        b=copy(a)
        a[1:n, :]=triu(original, -1)
        b[1:n, :]=triu(original_b)
        for j in 1:n, i in (j + 2):n
            a[i, j]=T(NaN)
        end
        for j in 1:n, i in (j + 1):n
            b[i, j]=T(NaN)
        end
        q=fill(T(NaN), n+2, n)
        z=copy(q)
        _, _, info, _=direct_hgeqz!(a, b, q, z, J; cq=('N'), cz=('N'), job=('E'))
        @test info==0 && all(isnan, q) && all(isnan, z)
        @test all(==(T(789)), a[(n + 1):end, :]) && all(==(T(789)), b[(n + 1):end, :])
        saved=map(copy, (a, b, q, z))
        _, _, info, work=direct_hgeqz!(a, b, q, z, J; query=true)
        @test info==0 && real(work[1])>=n
        @test all(isequal.(saved, (a, b, q, z)))
        _, _, info, _=direct_hgeqz!(a, b, q, z, J; lo=0)
        @test info == -5
        _, _, info, _=direct_hgeqz!(a, b, q, z, J; lwork_override=0)
        @test info==(T<:Real ? -19 : -18)
        @test all(isequal.(saved, (a, b, q, z)))
        a=fill(T(NaN), 1, 1)
        b=ones(T, 1, 1)
        q=zeros(T, 1, 1)
        z=copy(q)
        _, _, info, _=direct_hgeqz!(a, b, q, z, J)
        @test info==1
        a, b, alpha, beta, q, z=LAPACK.gges!('V', 'V', copy(original), copy(original_b))
        @test q*a*z' ≈ original
        @test q*b*z' ≈ original_b
        prefix=if T===Float32
            "s"
        elseif T===Float64
            "d"
        elseif T===ComplexF32
            "c"
        else
            "z"
        end
        for family in ("hgeqz", "gges", "gges3", "gegs")
            backing=BLAS.lbt_find_backing_library(
                prefix*family*"_", J===Int32 ? :lp64 : :ilp64
            )
            @test backing!==nothing &&
                realpath(backing.libname)==realpath(Colmat.backend_library())
        end
    end
end

@testset "QZ independent scales and real conjugate blocks" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        R=typeof(real(zero(T)))
        base=T[-1 -3 0; 3 -1 1; 0 0 2]
        base_b=Matrix{T}(I, 3, 3)
        for (sa, sb) in (
            (sqrt(floatmin(R)), sqrt(floatmax(R))/10),
            (sqrt(floatmax(R))/10, sqrt(floatmin(R))),
        )
            a=base*sa
            b=base_b*sb
            q=zeros(T, 3, 3)
            z=copy(q)
            alpha, beta, _, info, _=direct_gges!(a, b, q, z, J, Val(:gges))
            @test info==0
            @test q*(a/sa)*z' ≈ base
            @test q*(b/sb)*z' ≈ base_b
            @test all(isfinite, alpha) && all(isfinite, beta)
            @test all(!iszero, beta)
        end
    end
end
