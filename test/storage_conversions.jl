using Test, LinearAlgebra, Random, Libdl, Colmat

function conversion_reference()
    return only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
end
conversion_pointer(a::Matrix) = a
conversion_pointer(a::Vector) = view(a, 2:(length(a) - 1))
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    suffix=J===Int64 ? "64_" : ""
    @eval function direct_storage_conversion!(
        family,
        source::Array{$T},
        target::Array{$T},
        ::Type{$J};
        n,
        uplo=('L'),
        transr=('N'),
        lda=ndims(source)==2 ? size(source, 1) : size(target, 1),
        library=Colmat.backend_library(),
    )
        fn=Libdl.dlsym(Libdl.dlopen(library), Symbol($prefix*family*"_"*$suffix))
        a=conversion_pointer(source)
        b=conversion_pointer(target)
        info=Ref{$J}(-777)
        if family=="trttp"
            ccall(
                fn,
                Cvoid,
                (Ref{UInt8}, Ref{$J}, Ptr{$T}, Ref{$J}, Ptr{$T}, Ref{$J}),
                uplo,
                n,
                a,
                lda,
                b,
                info,
            )
        elseif family=="tpttr"
            ccall(
                fn,
                Cvoid,
                (Ref{UInt8}, Ref{$J}, Ptr{$T}, Ptr{$T}, Ref{$J}, Ref{$J}),
                uplo,
                n,
                a,
                b,
                lda,
                info,
            )
        elseif family=="trttf"
            ccall(
                fn,
                Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{$J}, Ptr{$T}, Ref{$J}, Ptr{$T}, Ref{$J}),
                transr,
                uplo,
                n,
                a,
                lda,
                b,
                info,
            )
        elseif family=="tfttr"
            ccall(
                fn,
                Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{$J}, Ptr{$T}, Ptr{$T}, Ref{$J}, Ref{$J}),
                transr,
                uplo,
                n,
                a,
                b,
                lda,
                info,
            )
        else
            ccall(
                fn,
                Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{$J}, Ptr{$T}, Ptr{$T}, Ref{$J}),
                transr,
                uplo,
                n,
                a,
                b,
                info,
            )
        end
        info[]
    end
end
function conversion_destination(T, format, n)
    return format=="tr" ? fill(T(777), max(1, n+3), n) : fill(T(777), n*(n+1)÷2+2)
end
function conversion_fixture(T, n, uplo, transr, special)
    original=[T<:Complex ? T(2i+3j, i-2j) : T(2i+3j) for i in 1:n, j in 1:n]
    if special && n>0
        original[1, 1]=T<:Complex ? T(-0.0, -0.0) : T(-0.0)
        if n>1
            original[n, n]=T<:Complex ? T(NaN, -0.0) : T(NaN)
        end
    end
    tr=fill(T(NaN), max(1, n+3), n)
    for j in 1:n, i in (uplo=='U' ? (1:j) : (j:n))
        tr[i, j]=original[i, j]
    end
    tp=conversion_destination(T, "tp", n)
    tf=conversion_destination(T, "tf", n)
    @assert direct_storage_conversion!(
        "trttp", tr, tp, Int64; n, uplo, library=conversion_reference()
    )==0
    @assert direct_storage_conversion!(
        "trttf", tr, tf, Int64; n, uplo, transr, library=conversion_reference()
    )==0
    return (; tr, tp, tf, original)
end
@testset "TR/TP/TF conversions match independent packing and roundtrip" begin
    reference=conversion_reference()
    inverses=Dict(
        "trttp"=>"tpttr",
        "tpttr"=>"trttp",
        "trttf"=>"tfttr",
        "tfttr"=>"trttf",
        "tpttf"=>"tfttp",
        "tfttp"=>"tpttf",
    )
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        n in 0:9,
        uplo in ('L', 'U'),
        transr in (T<:Complex ? ('N', 'C') : ('N', 'T')),
        special in (false, true),
        family in ("trttp", "tpttr", "trttf", "tfttr", "tpttf", "tfttp")

        fixture=conversion_fixture(T, n, uplo, transr, special)
        inputformat=family[1:2]
        outputformat=family[4:5]
        source=copy(getproperty(fixture, Symbol(inputformat)))
        prior=copy(source)
        target=conversion_destination(T, outputformat, n)
        expected=copy(target)
        @test direct_storage_conversion!(family, source, target, J; n, uplo, transr)==0
        @test direct_storage_conversion!(
            family, source, expected, Int64; n, uplo, transr, library=reference
        )==0
        @test isequal(target, expected)
        @test isequal(source, prior)
        restored=conversion_destination(T, inputformat, n)
        @test direct_storage_conversion!(
            inverses[family], target, restored, J; n, uplo, transr
        )==0
        if inputformat=="tr"
            for j in 1:n, i in 1:n
                if (uplo=='U' && i<=j)||(uplo=='L' && i>=j)
                    @test isequal(restored[i, j], fixture.original[i, j])
                else
                    @test restored[i, j]==T(777)
                end
            end
            @test all(==(T(777)), restored[(n + 1):end, :])
        else
            @test isequal(restored, prior)
        end
        if outputformat=="tr"
            @test all(==(T(777)), target[(n + 1):end, :])
        else
            @test target[[1, end]]==T[777, 777]
        end
    end
end
@testset "Format conversion diagnostics and integer width" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        family in ("trttp", "tpttr", "trttf", "tfttr", "tpttf", "tfttp")

        fixture=conversion_fixture(T, 3, 'L', 'N', false)
        source=copy(getproperty(fixture, Symbol(family[1:2])))
        target=conversion_destination(T, family[4:5], 3)
        prior=copy(target)
        with_trans=family in ("trttf", "tfttr", "tpttf", "tfttp")
        cases=Tuple{NamedTuple,Int}[
            ((; uplo=('?')), with_trans ? -2 : -1), ((; n=-1), with_trans ? -3 : -2)
        ]
        if with_trans
            push!(cases, ((; transr=('?')), -1))
            push!(cases, ((; transr=T<:Complex ? 'T' : 'C'), -1))
        end
        ldarg=if family=="trttp"
            4
        elseif family in ("tpttr", "trttf")
            5
        elseif family=="tfttr"
            6
        else
            0
        end
        ldarg>0 && push!(cases, ((; lda=2), -ldarg))
        for (options, status) in cases
            @test direct_storage_conversion!(family, source, target, J; n=3, options...)==status
            @test isequal(target, prior)
        end
        if J===Int64
            @test direct_storage_conversion!(
                family, source, target, J; n=(-(Int64(2)^32)+3)
            )==(with_trans ? -3 : -2)
            @test isequal(target, prior)
        end
    end
end
@testset "All 24 storage-conversion symbols through LBT" begin
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64),
        family in ("trttp", "tpttr", "trttf", "tfttr", "tpttf", "tfttp")

        backing=BLAS.lbt_find_backing_library(prefix*family*"_", J===Int32 ? :lp64 : :ilp64)
        @test backing!==nothing
        @test realpath(backing.libname)==realpath(Colmat.backend_library())
        fixture=conversion_fixture(T, 4, 'U', 'N', false)
        source=copy(getproperty(fixture, Symbol(family[1:2])))
        target=conversion_destination(T, family[4:5], 4)
        expected=copy(target)
        @test direct_storage_conversion!(
            family, source, target, J; n=4, uplo=('U'), library=BLAS.libblastrampoline
        )==0
        @test direct_storage_conversion!(
            family, source, expected, Int64; n=4, uplo=('U'), library=conversion_reference()
        )==0
        @test isequal(target, expected)
    end
end
