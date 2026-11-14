using Test, LinearAlgebra, Random, Libdl, Colmat
for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64),
    family in (:ggev, :ggev3, :gegv)

    R=typeof(real(zero(T)))
    symbol=Symbol(p*string(family)*"_"*(J===Int64 ? "64_" : ""))
    eigtypes=T<:Real ? [:(Ptr{$T}), :(Ptr{$T}), :(Ptr{$T})] : [:(Ptr{$T}), :(Ptr{$T})]
    eigargs=T<:Real ? [:ar, :ai, :beta] : [:ar, :beta]
    rwtypes=T<:Real ? [] : [:(Ptr{$R})]
    rwargs=T<:Real ? [] : [:rw]
    signature=Expr(
        :tuple,
        :(Ref{UInt8}),
        :(Ref{UInt8}),
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
        rwtypes...,
        :(Ref{$J}),
    )
    @eval function direct_generalized_eigen!(
        a::Matrix{$T},
        b::Matrix{$T},
        vl::Matrix{$T},
        vr::Matrix{$T},
        ::Type{$J},
        ::Val{$(QuoteNode(family))};
        jl=('V'),
        jr=('V'),
        query=false,
        lwork_override=nothing,
        library=Colmat.backend_library(),
    )
        n=size(a, 2)
        ar=zeros($T, n)
        ai=zeros($T, n)
        beta=zeros($T, n)
        work=zeros($T, max(1, ($T<:Real ? 8 : 2)*n))
        rw=zeros($R, 8n)
        info=Ref{$J}(0)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol)))
        ccall(
            f,
            Cvoid,
            $signature,
            jl,
            jr,
            n,
            a,
            max(1, size(a, 1)),
            b,
            max(1, size(b, 1)),
            $(eigargs...),
            vl,
            max(1, size(vl, 1)),
            vr,
            max(1, size(vr, 1)),
            work,
            if query
                -1
            elseif isnothing(lwork_override)
                length(work)
            else
                lwork_override
            end,
            $(rwargs...),
            info,
        )
        alpha=$T<:Real ? complex.(ar, ai) : ar
        return alpha, beta, info[], work
    end
end
function generalized_eigenvectors(v::AbstractMatrix{T}, alpha) where {T}
    T<:Complex && return copy(v)
    result=complex.(v)
    j=1
    while j<=size(v, 2)
        if imag(alpha[j])>0
            result[:, j]=v[:, j]+im*v[:, j + 1]
            result[:, j + 1]=conj.(result[:, j])
            j+=2
        else
            j+=1
        end
    end
    return result
end
function check_generalized_vectors(
    original, original_b, alpha, beta, vl, vr; left=true, right=true
)
    T=eltype(original)
    R=typeof(real(zero(T)))
    n=size(original, 2)
    for (requested, v, isleft) in ((left, vl, true), (right, vr, false))
        if requested
            vectors=generalized_eigenvectors(v[1:n, :], alpha)
            for j in 1:n
                x=vectors[:, j]
                residual=if isleft
                    conj(beta[j])*original'*x-conj(alpha[j])*original_b'*x
                else
                    beta[j]*original*x-alpha[j]*original_b*x
                end
                bound=1000eps(R)*(
                    abs(beta[j])*norm(original)+abs(alpha[j])*norm(original_b)
                )*max(1, norm(x))
                @test norm(residual)<=max(floatmin(R), bound)
                @test maximum(abs.(real.(x))+abs.(imag.(x)); init=zero(R)) ≈ one(R) rtol=100eps(
                    R
                )
            end
        else
            @test all(isnan, v[1:n, :])
        end
    end
end
@testset "Generalized EVD drivers and both ABIs" begin
    Random.seed!(301)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), family in (:ggev, :ggev3, :gegv), n in (0, 1, 6),
        (jl, jr) in (('N', 'N'), ('V', 'N'), ('N', 'V'), ('V', 'V'))

        original=randn(T, n, n)
        original_b=randn(T, n, n)
        a=copy(original)
        b=copy(original_b)
        vl=fill(T(NaN), n, n)
        vr=copy(vl)
        alpha, beta, info, _=direct_generalized_eigen!(a, b, vl, vr, J, Val(family); jl, jr)
        @test info==0
        @test all(isfinite, alpha) && all(isfinite, beta)
        check_generalized_vectors(
            original, original_b, alpha, beta, vl, vr; left=jl=='V', right=jr=='V'
        )
        saved=map(copy, (a, b, vl, vr))
        _, _, info, work=direct_generalized_eigen!(
            a, b, vl, vr, J, Val(family); jl, jr, query=true
        )
        @test info==0 && real(work[1])>=max(1, (T<:Real ? 8 : 2)*n)
        @test all(isequal.(saved, (a, b, vl, vr)))
    end
end

for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    R=typeof(real(zero(T)))
    symbol=Symbol(p*"ggevx_"*(J===Int64 ? "64_" : ""))
    eigtypes=T<:Real ? [:(Ptr{$T}), :(Ptr{$T}), :(Ptr{$T})] : [:(Ptr{$T}), :(Ptr{$T})]
    eigargs=T<:Real ? [:ar, :ai, :beta] : [:ar, :beta]
    rwtypes=T<:Real ? [] : [:(Ptr{$R})]
    rwargs=T<:Real ? [] : [:rw]
    signature=Expr(
        :tuple,
        :(Ref{UInt8}),
        :(Ref{UInt8}),
        :(Ref{UInt8}),
        :(Ref{UInt8}),
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
        :(Ptr{$J}),
        :(Ptr{$J}),
        fill(:(Ptr{$R}), 6)...,
        :(Ptr{$T}),
        :(Ref{$J}),
        rwtypes...,
        :(Ptr{$J}),
        :(Ptr{$J}),
        :(Ptr{$J}),
    )
    @eval function direct_ggevx!(
        a::Matrix{$T},
        b::Matrix{$T},
        vl::Matrix{$T},
        vr::Matrix{$T},
        ::Type{$J};
        bal=('N'),
        sense=('N'),
        jl=('V'),
        jr=('V'),
        query=false,
        library=Colmat.backend_library(),
    )
        n=size(a, 2)
        ar=zeros($T, n)
        ai=zeros($T, n)
        beta=zeros($T, n)
        minimum=if $T<:Real
            v=jl=='V'||jr=='V' ? 6n : 2n
            sense in ('E', 'B') && (v=max(v, 10n))
            sense in ('V', 'B') && (v=max(v, 2n*(n+4)+16))
            n==0 ? 1 : max(1, v)
        else
            if n==0
                1
            elseif sense in ('V', 'B')
                2n*(n+1)
            elseif sense=='E'
                4n
            else
                2n
            end
        end
        work=zeros($T, minimum)
        rw=zeros($R, 8n)
        iw=fill($J(777), n+8)
        bw=fill($J(777), n+2)
        ilo=fill($J(777), 3)
        ihi=copy(ilo)
        info=copy(ilo)
        ls=fill($R(-5), n)
        rs=copy(ls)
        an=[$R(-5)]
        bn=copy(an)
        ce=copy(ls)
        cv=copy(ls)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol)))
        ccall(
            f,
            Cvoid,
            $signature,
            bal,
            jl,
            jr,
            sense,
            n,
            a,
            max(1, size(a, 1)),
            b,
            max(1, size(b, 1)),
            $(eigargs...),
            vl,
            max(1, size(vl, 1)),
            vr,
            max(1, size(vr, 1)),
            view(ilo, 2:2),
            view(ihi, 2:2),
            ls,
            rs,
            an,
            bn,
            ce,
            cv,
            work,
            query ? -1 : length(work),
            $(rwargs...),
            view(iw, 2:(length(iw) - 1)),
            view(bw, 2:(length(bw) - 1)),
            view(info, 2:2),
        )
        alpha=$T<:Real ? complex.(ar, ai) : ar
        return (;
            alpha,
            beta,
            info=info[2],
            ilo=ilo[2],
            ihi=ihi[2],
            ls,
            rs,
            an=an[1],
            bn=bn[1],
            ce,
            cv,
            work,
            integer_guards=(
                ilo[[1, 3]], ihi[[1, 3]], info[[1, 3]], iw[[1, end]], bw[[1, end]]
            ),
        )
    end
end
@testset "GGEVX native basic mode and expert ABI bridge" begin
    Random.seed!(3011)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), bal in ('N', 'P', 'S', 'B'),
        sense in ('N', 'E', 'V', 'B')

        n=4
        original=randn(T, n, n)
        original_b=randn(T, n, n)
        a=copy(original)
        b=copy(original_b)
        vl=zeros(T, n, n)
        vr=copy(vl)
        result=direct_ggevx!(a, b, vl, vr, J; bal, sense)
        @test result.info==0
        check_generalized_vectors(original, original_b, result.alpha, result.beta, vl, vr)
        @test 1<=result.ilo<=result.ihi<=n
        @test all(g->all(==(J(777)), g), result.integer_guards)
        @test isfinite(result.an) && result.an>0
        @test isfinite(result.bn) && result.bn>0
        @test norm(tril(a, T<:Real ? -2 : -1))==0
        @test norm(tril(b, -1))==0
        if bal=='N'
            @test result.ilo==1 && result.ihi==n
            @test result.ls==ones(n) && result.rs==ones(n)
            @test result.an ≈ opnorm(original, 1)
            @test result.bn ≈ opnorm(original_b, 1)
        end
        if sense in ('E', 'B')
            @test all(isfinite, result.ce) && all(>=(0), result.ce)
        end
        if sense in ('V', 'B')
            @test all(isfinite, result.cv) && all(>=(0), result.cv)
        end
        saved=map(copy, (a, b, vl, vr))
        queryresult=direct_ggevx!(a, b, vl, vr, J; bal, sense, query=true)
        @test queryresult.info==0 && real(queryresult.work[1])>=1
        @test queryresult.ilo==777 && queryresult.ihi==777
        @test all(isequal.(saved, (a, b, vl, vr)))
        @test all(g->all(==(J(777)), g), queryresult.integer_guards)
    end
end

@testset "Generalized EVD singular pencils, conjugate pairs and padding" begin
    Random.seed!(3012)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        family in (:ggev, :ggev3, :gegv)

        inputs=(
            (T[0 -1; 1 0], Matrix{T}(I, 2, 2)),
            (zeros(T, 3, 3), zeros(T, 3, 3)),
            (Matrix{T}(I, 3, 3), zeros(T, 3, 3)),
            (T[1 1; 0 1], Matrix{T}(I, 2, 2)),
        )
        for (original, original_b) in inputs
            n=size(original, 2)
            a=fill(T(456), n+2, n)
            b=copy(a)
            vl=copy(a)
            vr=copy(a)
            a[1:n, :]=original
            b[1:n, :]=original_b
            alpha, beta, info, _=direct_generalized_eigen!(a, b, vl, vr, J, Val(family))
            @test info==0
            check_generalized_vectors(original, original_b, alpha, beta, vl, vr)
            @test all(x->all(==(T(456)), x[(n + 1):end, :]), (a, b, vl, vr))
            if all(iszero, original_b)
                @test all(iszero, beta)
            end
            saved=map(copy, (a, b, vl, vr))
            _, _, info, _=direct_generalized_eigen!(
                a, b, vl, vr, J, Val(family); lwork_override=0
            )
            @test info==(T<:Real ? -16 : -15)
            @test all(isequal.(saved, (a, b, vl, vr)))
        end
    end
end

@testset "Generalized EVD independent OpenBLAS homogeneous spectrum" begin
    library=only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
    Random.seed!(3013)
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        n=7
        original=randn(T, n, n)
        original_b=randn(T, n, n)
        vl=zeros(T, n, n)
        vr=copy(vl)
        alpha, beta, info, _=direct_generalized_eigen!(
            copy(original), copy(original_b), vl, vr, J, Val(:ggev)
        )
        ar, be, refinfo, _=direct_generalized_eigen!(
            copy(original), copy(original_b), copy(vl), copy(vr), Int64, Val(:ggev); library
        )
        @test info==0 && refinfo==0
        R=typeof(real(zero(T)))
        pool=collect(zip(ar, be))
        for (a, b) in zip(alpha, beta)
            scale=hypot(abs(a), abs(b))
            a/=scale
            b/=scale
            distances=[
                abs(a*(d/hypot(abs(c), abs(d)))-b*(c/hypot(abs(c), abs(d)))) for
                (c, d) in pool
            ]
            i=argmin(distances)
            @test distances[i]<=5000eps(R)
            deleteat!(pool, i)
        end
    end
end

@testset "GGEVX restored Schur scales, empty input and registration" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        R=typeof(real(zero(T)))
        base=T[-1 -3; 3 -1]
        base_b=T[2 0; 0 1]
        for (sa, sb) in (
            (sqrt(floatmin(R)), sqrt(floatmax(R))/10),
            (sqrt(floatmax(R))/10, sqrt(floatmin(R))),
        )
            original=base*sa
            original_b=base_b*sb
            a=copy(original)
            b=copy(original_b)
            vl=zeros(T, 2, 2)
            vr=copy(vl)
            result=direct_ggevx!(a, b, vl, vr, J)
            @test result.info==0
            @test norm(a/sa) ≈ norm(base)
            @test norm(b/sb) ≈ norm(base_b)
            @test all(isfinite, result.alpha) && all(isfinite, result.beta)
            @test all(!iszero, result.beta)
            check_generalized_vectors(base, base_b, result.alpha/sa, result.beta/sb, vl, vr)
        end
        for bal in ('N', 'B'), sense in ('N', 'B')
            a=zeros(T, 0, 0)
            vl=copy(a)
            vr=copy(a)
            result=direct_ggevx!(a, copy(a), vl, vr, J; bal, sense)
            @test result.info==0
        end
        prefix=if T===Float32
            "s"
        elseif T===Float64
            "d"
        elseif T===ComplexF32
            "c"
        else
            "z"
        end
        for family in ("ggev", "ggev3", "gegv", "ggevx")
            backing=BLAS.lbt_find_backing_library(
                prefix*family*"_", J===Int32 ? :lp64 : :ilp64
            )
            @test backing!==nothing &&
                realpath(backing.libname)==realpath(Colmat.backend_library())
        end
    end
end

@testset "GGEVX missing expert configuration is diagnosed before registration" begin
    worker=joinpath(@__DIR__, "generalized_eigen_fallback_worker.jl")
    project=dirname(Base.active_project())
    @test success(`$(Base.julia_cmd()) --startup-file=no --project=$project $worker`)
end
