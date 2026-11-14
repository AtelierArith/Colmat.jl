using Test, LinearAlgebra, Random, Libdl, Colmat

norm_guard(values::Vector{T}) where {T} = vcat(T(777), values, T(777))
norm_pointer(a::Vector) = view(a, 2:(length(a) - 1))
norm_pointer(a::Matrix) = a
function norms_reference()
    return only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
end
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    R=real(T)
    suffix=J===Int64 ? "64_" : ""
    @eval function direct_matrix_norm(
        family,
        a::Array{$T},
        ::Type{$J};
        norm=('M'),
        m=size(a, 1),
        n=ndims(a)==2 ? size(a, 2) : length(a)-2,
        uplo=('L'),
        diag=('N'),
        transr=('N'),
        k=0,
        kl=0,
        ku=0,
        lda=size(a, 1),
        dl=($T[777, 777]),
        du=($T[777, 777]),
        d=($R[777, 777]),
        library=Colmat.backend_library(),
    )
        fn=Libdl.dlsym(Libdl.dlopen(library), Symbol($prefix*family*"_"*$suffix))
        ap=norm_pointer(a)
        lp=norm_pointer(dl)
        up=norm_pointer(du)
        dp=norm_pointer(d)
        workspace=fill($R(777), max(1, length(a))+4)
        work=view(workspace, 2:(length(workspace) - 1))
        if family=="lange"
            value=ccall(
                fn,
                $R,
                (Ref{UInt8}, Ref{$J}, Ref{$J}, Ptr{$T}, Ref{$J}, Ptr{$R}),
                norm,
                m,
                n,
                ap,
                lda,
                work,
            )
        elseif family=="lanhs"
            value=ccall(
                fn,
                $R,
                (Ref{UInt8}, Ref{$J}, Ptr{$T}, Ref{$J}, Ptr{$R}),
                norm,
                n,
                ap,
                lda,
                work,
            )
        elseif family=="langb"
            value=ccall(
                fn,
                $R,
                (Ref{UInt8}, Ref{$J}, Ref{$J}, Ref{$J}, Ptr{$T}, Ref{$J}, Ptr{$R}),
                norm,
                n,
                kl,
                ku,
                ap,
                lda,
                work,
            )
        elseif family=="langt"
            value=ccall(
                fn,
                $R,
                (Ref{UInt8}, Ref{$J}, Ptr{$T}, Ptr{$T}, Ptr{$T}),
                norm,
                n,
                lp,
                ap,
                up,
            )
        elseif family=="lansy"
            value=ccall(
                fn,
                $R,
                (Ref{UInt8}, Ref{UInt8}, Ref{$J}, Ptr{$T}, Ref{$J}, Ptr{$R}),
                norm,
                uplo,
                n,
                ap,
                lda,
                work,
            )
        elseif family=="lansb"
            value=ccall(
                fn,
                $R,
                (Ref{UInt8}, Ref{UInt8}, Ref{$J}, Ref{$J}, Ptr{$T}, Ref{$J}, Ptr{$R}),
                norm,
                uplo,
                n,
                k,
                ap,
                lda,
                work,
            )
        elseif family=="lansp"
            value=ccall(
                fn,
                $R,
                (Ref{UInt8}, Ref{UInt8}, Ref{$J}, Ptr{$T}, Ptr{$R}),
                norm,
                uplo,
                n,
                ap,
                work,
            )
        elseif family=="lantr"
            value=ccall(
                fn,
                $R,
                (
                    Ref{UInt8},
                    Ref{UInt8},
                    Ref{UInt8},
                    Ref{$J},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$R},
                ),
                norm,
                uplo,
                diag,
                m,
                n,
                ap,
                lda,
                work,
            )
        elseif family=="lantb"
            value=ccall(
                fn,
                $R,
                (
                    Ref{UInt8},
                    Ref{UInt8},
                    Ref{UInt8},
                    Ref{$J},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$R},
                ),
                norm,
                uplo,
                diag,
                n,
                k,
                ap,
                lda,
                work,
            )
        elseif family=="lantp"
            value=ccall(
                fn,
                $R,
                (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{$J}, Ptr{$T}, Ptr{$R}),
                norm,
                uplo,
                diag,
                n,
                ap,
                work,
            )
        elseif family=="lansf"
            value=ccall(
                fn,
                $R,
                (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{$J}, Ptr{$T}, Ptr{$R}),
                norm,
                transr,
                uplo,
                n,
                ap,
                work,
            )
        elseif family=="lanhf"
            value=ccall(
                fn,
                $R,
                (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{$J}, Ptr{$T}, Ptr{$R}),
                norm,
                transr,
                uplo,
                n,
                ap,
                work,
            )
        elseif family=="lanst"
            value=ccall(fn, $R, (Ref{UInt8}, Ref{$J}, Ptr{$R}, Ptr{$T}), norm, n, dp, ap)
        elseif family=="lanht"
            value=ccall(fn, $R, (Ref{UInt8}, Ref{$J}, Ptr{$R}, Ptr{$T}), norm, n, dp, ap)
        elseif family=="lanhe"
            value=ccall(
                fn,
                $R,
                (Ref{UInt8}, Ref{UInt8}, Ref{$J}, Ptr{$T}, Ref{$J}, Ptr{$R}),
                norm,
                uplo,
                n,
                ap,
                lda,
                work,
            )
        elseif family=="lanhb"
            value=ccall(
                fn,
                $R,
                (Ref{UInt8}, Ref{UInt8}, Ref{$J}, Ref{$J}, Ptr{$T}, Ref{$J}, Ptr{$R}),
                norm,
                uplo,
                n,
                k,
                ap,
                lda,
                work,
            )
        elseif family=="lanhp"
            value=ccall(
                fn,
                $R,
                (Ref{UInt8}, Ref{UInt8}, Ref{$J}, Ptr{$T}, Ptr{$R}),
                norm,
                uplo,
                n,
                ap,
                work,
            )
        else
            error("unknown norm family: $family")
        end
        (; value, workspace)
    end
end
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z"))
    symbol=Symbol(prefix*"trttf_64_")
    @eval function reference_rfp(a::Matrix{$T}, transr, uplo)
        n=size(a, 2)
        out=fill($T(777), n*(n+1)÷2+2)
        ap=view(out, 2:(length(out) - 1))
        info=Ref{Int64}(-777)
        fn=Libdl.dlsym(Libdl.dlopen(norms_reference()), $(QuoteNode(symbol)))
        ccall(
            fn,
            Cvoid,
            (Ref{UInt8}, Ref{UInt8}, Ref{Int64}, Ptr{$T}, Ref{Int64}, Ptr{$T}, Ref{Int64}),
            transr,
            uplo,
            n,
            a,
            size(a, 1),
            ap,
            info,
        )
        @assert info[]==0
        out
    end
end
function scalar_matrix_norm(a, mode)
    R=real(eltype(a))
    m, n=size(a)
    min(m, n)==0 && return zero(R)
    any(v->isnan(real(v))||isnan(imag(v)), a) && return R(NaN)
    values=[hypot(BigFloat(real(a[i, j])), BigFloat(imag(a[i, j]))) for i in 1:m, j in 1:n]
    result=if uppercase(mode)=='M'
        maximum(values)
    elseif uppercase(mode) in ('1', 'O')
        maximum(sum(values; dims=1))
    elseif uppercase(mode)=='I'
        maximum(sum(values; dims=2))
    else
        sqrt(sum(abs2, values))
    end
    return R(result)
end
function norm_matrix_storage(a)
    m, n=size(a)
    out=fill(eltype(a)(NaN), max(1, m+3), n)
    out[1:m, :]=a
    return out
end
@testset "LANGE dense matrix norm contracts and complex modulus" begin
    reference=norms_reference()
    rng=MersenneTwister(279)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        (m, n) in ((0, 0), (0, 4), (4, 0), (1, 1), (3, 7), (7, 3), (9, 9)),
        norm in ('M', '1', 'O', 'I', 'F', 'E', 'm', 'o', 'i', 'f')

        R=real(T)
        original=randn(rng, T, m, n)
        a=norm_matrix_storage(original)
        prior=copy(a)
        result=direct_matrix_norm("lange", a, J; m, n, norm)
        external=direct_matrix_norm("lange", a, Int64; m, n, norm, library=reference)
        expected=scalar_matrix_norm(original, norm)
        @test isapprox(result.value, expected; rtol=100eps(R), atol=nextfloat(zero(R)))
        @test isapprox(
            result.value, external.value; rtol=100eps(R), atol=nextfloat(zero(R))
        )
        @test isequal(a, prior)
        @test all(==(R(777)), result.workspace)
    end
    for T in (ComplexF32, ComplexF64), J in (Int32, Int64)
        a=T[3+4im 1; 2 0]
        @test direct_matrix_norm("lange", a, J; norm=('M')).value==5
        @test direct_matrix_norm("lange", a, J; norm=('1')).value==7
        @test direct_matrix_norm("lange", a, J; norm=('I')).value==6
    end
end
@testset "RFP norm layout against independent TRTTF packing" begin
    reference=norms_reference()
    rng=MersenneTwister(281)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        n in 0:9,
        uplo in ('L', 'U'),
        transr in (T<:Complex ? ('N', 'C') : ('N', 'T')),
        norm in ('M', '1', 'O', 'I', 'F', 'E')

        R=real(T)
        raw=randn(rng, T, n, n)
        original=Matrix(Hermitian(raw, uplo=='L' ? :L : :U))
        source=norm_matrix_storage(original)
        if T<:Complex
            for j in 1:n
                source[j, j]=T(real(original[j, j]), NaN)
            end
        end
        a=reference_rfp(source, transr, uplo)
        prior=copy(a)
        family=T<:Complex ? "lanhf" : "lansf"
        result=direct_matrix_norm(family, a, J; n, norm, uplo, transr)
        external=direct_matrix_norm(
            family, a, Int64; n, norm, uplo, transr, library=reference
        )
        expected=scalar_matrix_norm(original, norm)
        @test isapprox(result.value, expected; rtol=100eps(R), atol=nextfloat(zero(R)))
        @test isapprox(
            result.value, external.value; rtol=100eps(R), atol=nextfloat(zero(R))
        )
        @test isequal(a, prior)
        @test all(==(R(777)), result.workspace)
    end
end

function norm_families(T)
    common=(
        "lange",
        "langb",
        "langt",
        "lanhs",
        "lansy",
        "lansb",
        "lansp",
        "lantr",
        "lantb",
        "lantp",
    )
    return if T<:Complex
        (common..., "lanhe", "lanhb", "lanhp", "lanhf", "lanht")
    else
        (common..., "lansf", "lanst")
    end
end
function norm_case(T, family, n; m=n, uplo=('L'), diag=('N'), k=2, kl=2, ku=1, transr=('N'))
    R=real(T)
    raw=[T<:Complex ? T((3i+2j)%7-3, (2i+5j)%9-4) : T((3i+2j)%7-3) for i in 1:m, j in 1:n]
    expected=zeros(T, m, n)
    dl=norm_guard(T[])
    du=norm_guard(T[])
    d=norm_guard(R[])
    hermitian=family in ("lanhe", "lanhb", "lanhp", "lanhf")
    symmetric=family in
              ("lansy", "lansb", "lansp", "lanhe", "lanhb", "lanhp", "lansf", "lanhf")
    unit=diag=='U' && family in ("lantr", "lantb", "lantp")
    if family in ("lange", "lanhs", "lantr")
        a=fill(T(NaN), max(1, m+3), n)
        for j in 1:n, i in 1:m
            selected=family=="lange" || (
                if family=="lanhs"
                    i<=j+1
                elseif uplo=='U'
                    i<=j
                else
                    i>=j
                end
            )
            if selected
                expected[i, j]=unit&&i==j ? one(T) : raw[i, j]
                a[i, j]=unit&&i==j ? T(NaN) : raw[i, j]
            end
        end
    elseif family=="langb"
        a=fill(T(NaN), kl+ku+4, n)
        for j in 1:n, i in max(1, j - ku):min(n, j + kl)
            a[ku + 1 + i - j, j]=raw[i, j]
            expected[i, j]=raw[i, j]
        end
    elseif family=="langt"
        a=norm_guard(T[raw[i, i] for i in 1:n])
        dl=norm_guard(T[raw[i + 1, i] for i in 1:(n - 1)])
        du=norm_guard(T[raw[i, i + 1] for i in 1:(n - 1)])
        for i in 1:n
            expected[i, i]=raw[i, i]
            if i<n
                expected[i + 1, i]=raw[i + 1, i]
                expected[i, i + 1]=raw[i, i + 1]
            end
        end
    elseif family in ("lanst", "lanht")
        d=norm_guard(R[real(raw[i, i]) for i in 1:n])
        a=norm_guard(T[raw[i + 1, i] for i in 1:(n - 1)])
        for i in 1:n
            expected[i, i]=real(raw[i, i])
            if i<n
                expected[i + 1, i]=raw[i + 1, i]
                expected[i, i + 1]=conj(raw[i + 1, i])
            end
        end
    else
        banded=family in ("lansb", "lanhb", "lantb")
        full=family in ("lansy", "lanhe")
        rectangular=family in ("lansf", "lanhf")
        if banded
            a=fill(T(NaN), k+4, n)
        elseif full || rectangular
            a=fill(T(NaN), max(1, n+3), n)
        else
            values=T[]
        end
        for j in 1:n, i in (uplo=='U' ? (1:j) : (j:n))
            banded && abs(i-j)>k && continue
            value=raw[i, j]
            logical=if unit&&i==j
                one(T)
            elseif hermitian&&i==j
                T(real(value))
            else
                value
            end
            stored=if unit&&i==j
                T(NaN)
            elseif hermitian&&i==j
                T(real(value), NaN)
            else
                value
            end
            expected[i, j]=logical
            if symmetric && i!=j
                expected[j, i]=hermitian ? conj(logical) : logical
            end
            if banded
                a[uplo=='U' ? k+1+i-j : 1+i-j, j]=stored
            elseif full || rectangular
                a[i, j]=stored
            else
                push!(values, stored)
            end
        end
        if rectangular
            a=reference_rfp(a, transr, uplo)
        elseif !banded && !full
            a=norm_guard(values)
        end
    end
    return (; a, expected, kwargs=(; m, n, uplo, diag, k, kl, ku, transr, dl, du, d))
end
@testset "Structured matrix norms with poisoned unused storage" begin
    reference=norms_reference()
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        family in norm_families(T),
        n in (0, 1, 2, 5, 9),
        uplo in ('L', 'U'),
        norm in ('M', '1', 'O', 'I', 'F', 'E')

        family in ("lange", "lansf", "lanhf") && continue
        R=real(T)
        for diag in (family in ("lantr", "lantb", "lantp") ? ('N', 'U') : ('N',)),
            k in (family in ("lansb", "lanhb", "lantb") ? (0, 2, n+2) : (2,))

            case=norm_case(T, family, n; uplo, diag, k)
            prior=copy(case.a)
            dp=copy(case.kwargs.d)
            lp=copy(case.kwargs.dl)
            up=copy(case.kwargs.du)
            result=direct_matrix_norm(family, case.a, J; norm, case.kwargs...)
            external=direct_matrix_norm(
                family, case.a, Int64; norm, case.kwargs..., library=reference
            )
            expected=scalar_matrix_norm(case.expected, norm)
            @test isapprox(result.value, expected; rtol=100eps(R), atol=nextfloat(zero(R)))
            @test isapprox(
                result.value, external.value; rtol=100eps(R), atol=nextfloat(zero(R))
            )
            @test isequal(case.a, prior)
            @test all(==(R(777)), result.workspace)
            @test isequal(case.kwargs.d, dp)
            @test isequal(case.kwargs.dl, lp)
            @test isequal(case.kwargs.du, up)
        end
    end
end
@testset "Rectangular triangular norms and asymmetric bandwidths" begin
    reference=norms_reference()
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        norm in ('M', '1', 'I', 'F'),
        (uplo, m, n) in (('U', 0, 4), ('U', 3, 7), ('L', 4, 0), ('L', 7, 3)),
        diag in ('N', 'U')

        R=real(T)
        case=norm_case(T, "lantr", n; m, uplo, diag)
        result=direct_matrix_norm("lantr", case.a, J; norm, case.kwargs...)
        expected=scalar_matrix_norm(case.expected, norm)
        external=direct_matrix_norm(
            "lantr", case.a, Int64; norm, case.kwargs..., library=reference
        )
        @test result.value≈expected rtol=100eps(R)
        @test result.value≈external.value rtol=100eps(R)
    end
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        norm in ('M', '1', 'I', 'F'),
        n in (0, 1, 5),
        kl in (0, 1, 7),
        ku in (0, 2, 7)

        R=real(T)
        case=norm_case(T, "langb", n; kl, ku)
        result=direct_matrix_norm("langb", case.a, J; norm, case.kwargs...)
        expected=scalar_matrix_norm(case.expected, norm)
        @test result.value≈expected rtol=100eps(R)
    end
end

function norm_scale_array(a, scale)
    T=eltype(a)
    map(a) do value
        value==T(777) && return value
        return T<:Complex ? T(real(value)*scale, imag(value)*scale) : value*scale
    end
end
@testset "All matrix norm formats at tiny and large scales" begin
    reference=norms_reference()
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), family in norm_families(T), norm in ('M', '1', 'I', 'F'),
        scale in (nextfloat(zero(real(T))), 16floatmin(real(T)), floatmax(real(T))/128)

        R=real(T)
        case=norm_case(T, family, 5)
        a=norm_scale_array(case.a, scale)
        kwargs=merge(
            case.kwargs,
            (;
                d=norm_scale_array(case.kwargs.d, scale),
                dl=norm_scale_array(case.kwargs.dl, scale),
                du=norm_scale_array(case.kwargs.du, scale),
            ),
        )
        expected=scalar_matrix_norm(norm_scale_array(case.expected, scale), norm)
        result=direct_matrix_norm(family, a, J; norm, kwargs...)
        external=direct_matrix_norm(family, a, Int64; norm, kwargs..., library=reference)
        quantum=(norm in ('1', 'I') ? 5 : 1)*nextfloat(zero(R))
        @test isapprox(result.value, expected; rtol=100eps(R), atol=quantum)
        @test isapprox(result.value, external.value; rtol=100eps(R), atol=quantum)
        @test all(==(R(777)), result.workspace)
    end
end
@testset "Frobenius finite targets at the limits of each real precision" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        R=real(T)
        for value in (T(nextfloat(zero(R))), T(floatmax(R)/8))
            a=fill(value, 7, 7)
            result=direct_matrix_norm("lange", a, J; norm=('F')).value
            expected=scalar_matrix_norm(a, 'F')
            @test isfinite(result)
            @test result>0
            @test isapprox(result, expected; rtol=100eps(R), atol=nextfloat(zero(R)))
        end
        if T<:Complex
            value=T(nextfloat(zero(R)), nextfloat(zero(R)))
            a=fill(value, 7, 7)
            result=direct_matrix_norm("lange", a, J; norm=('F')).value
            @test result==scalar_matrix_norm(a, 'F')
        end
    end
end
@testset "Selected NaN/Inf and complex modulus versus component semantics" begin
    reference=norms_reference()
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), family in norm_families(T), norm in ('M', '1', 'I', 'F'),
        value in (T(NaN), T(Inf))

        case=norm_case(T, family, 3)
        a=copy(case.a)
        index=findfirst(v->!isnan(real(v)) && v!=T(777), a)
        @test index!==nothing
        a[index]=value
        result=direct_matrix_norm(family, a, J; norm, case.kwargs...).value
        external=direct_matrix_norm(
            family, a, Int64; norm, case.kwargs..., library=reference
        ).value
        @test isnan(value) ? isnan(result) : isinf(result)
        @test isnan(external) ? isnan(result) : result==external
    end
    for T in (ComplexF32, ComplexF64),
        J in (Int32, Int64), norm in ('M', '1', 'I', 'F'),
        value in (T(Inf, NaN), T(NaN, Inf), T(Inf, Inf))

        result=direct_matrix_norm("lange", fill(value, 1, 1), J; norm).value
        external=direct_matrix_norm(
            "lange", fill(value, 1, 1), Int64; norm, library=reference
        ).value
        nan_expected=norm=='F' && (isnan(real(value))||isnan(imag(value)))
        @test nan_expected ? isnan(result) : isinf(result)
        @test isnan(external) ? isnan(result) : result==external
    end
end
function matrix_norm_last_error()
    fn=Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error)
    return ccall(fn, Cint, ())
end
@testset "Matrix norm argument diagnostics and integer width" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        family in norm_families(T)

        case=norm_case(T, family, 3)
        R=real(T)
        narg=if family=="lange"
            3
        elseif family in ("lantr",)
            5
        elseif family in ("lantb", "lantp", "lansf", "lanhf")
            4
        elseif family in ("lansy", "lansb", "lansp", "lanhe", "lanhb", "lanhp")
            3
        else
            2
        end
        options=[((; norm=('?')), 1), ((; n=-1), narg)]
        if family in
            ("lansy", "lansb", "lansp", "lanhe", "lanhb", "lanhp", "lantr", "lantb", "lantp")
            push!(options, ((; uplo=('?')), 2))
        end
        if family in ("lantr", "lantb", "lantp")
            push!(options, ((; diag=('?')), 3))
        end
        if family in ("lansf", "lanhf")
            push!(options, ((; transr=('?')), 2))
            push!(options, ((; uplo=('?')), 3))
        end
        if family in ("lansb", "lanhb", "lantb")
            push!(options, ((; k=-1), family=="lantb" ? 5 : 4))
        end
        if family=="langb"
            push!(options, ((; kl=-1), 3))
            push!(options, ((; ku=-1), 4))
        end
        ldarg=if family=="lange"
            5
        elseif family=="lanhs"
            4
        elseif family=="langb"
            6
        elseif family in ("lansy", "lanhe")
            5
        elseif family in ("lansb", "lanhb")
            6
        elseif family in ("lantr", "lantb")
            7
        else
            0
        end
        ldarg>0 && push!(options, ((; lda=0), ldarg))
        for (opts, status) in options
            kwargs=merge(case.kwargs, opts)
            matrix_norm_last_error()
            result=direct_matrix_norm(family, case.a, J; kwargs...)
            @test result.value==zero(R)
            @test matrix_norm_last_error()==status
        end
        if J===Int64
            kwargs=merge(case.kwargs, (; n=(-(Int64(2)^32)+3)))
            matrix_norm_last_error()
            result=direct_matrix_norm(family, case.a, J; kwargs...)
            @test result.value==0
            @test matrix_norm_last_error()==narg
        end
    end
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        a=fill(T(NaN), 1, 0)
        large=Int64(2)^32+3
        matrix_norm_last_error()
        result=direct_matrix_norm("lange", a, Int64; m=large, n=0, lda=large, norm=('I'))
        @test result.value==0
        @test matrix_norm_last_error()==0
        matrix_norm_last_error()
        result=direct_matrix_norm("lange", a, Int64; m=large, n=0, lda=1, norm=('I'))
        @test result.value==0
        @test matrix_norm_last_error()==5
    end
end
@testset "All 54 matrix norm symbols through numerical LBT entry points" begin
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64), family in norm_families(T),
        norm in ('M', '1', 'I', 'F')

        backing=BLAS.lbt_find_backing_library(prefix*family*"_", J===Int32 ? :lp64 : :ilp64)
        @test backing!==nothing
        @test realpath(backing.libname)==realpath(Colmat.backend_library())
        case=norm_case(T, family, 3)
        result=direct_matrix_norm(
            family, case.a, J; norm, case.kwargs..., library=BLAS.libblastrampoline
        )
        expected=scalar_matrix_norm(case.expected, norm)
        @test result.value≈expected rtol=100eps(real(T))
    end
end
