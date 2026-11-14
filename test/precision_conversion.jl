using Test, LinearAlgebra, Libdl, Colmat

for (family, S, D) in (
        ("slag2d", Float32, Float64),
        ("dlag2s", Float64, Float32),
        ("clag2z", ComplexF32, ComplexF64),
        ("zlag2c", ComplexF64, ComplexF32),
        ("dlat2s", Float64, Float32),
        ("zlat2c", ComplexF64, ComplexF32),
    ),
    J in (Int32, Int64)

    suffix=J===Int64 ? "64_" : ""
    family_key=QuoteNode(Symbol(family))
    @eval function direct_precision!(
        ::Val{$family_key},
        a::Matrix{$S},
        b::Matrix{$D},
        ::Type{$J};
        m,
        n,
        uplo=('U'),
        library=Colmat.backend_library(),
        initial=-777,
    )
        f=Libdl.dlsym(Libdl.dlopen(library), Symbol($family*"_"*$suffix))
        info=fill($J(-777), 3)
        info[2]=initial
        if occursin("lat", $family)
            ccall(
                f,
                Cvoid,
                (Ref{UInt8}, Ref{$J}, Ptr{$S}, Ref{$J}, Ptr{$D}, Ref{$J}, Ptr{$J}),
                uplo,
                n,
                a,
                size(a, 1),
                b,
                size(b, 1),
                pointer(info, 2),
            )
        else
            ccall(
                f,
                Cvoid,
                (Ref{$J}, Ref{$J}, Ptr{$S}, Ref{$J}, Ptr{$D}, Ref{$J}, Ptr{$J}),
                m,
                n,
                a,
                size(a, 1),
                b,
                size(b, 1),
                pointer(info, 2),
            )
        end
        @test info[1]==info[3]==-777
        if info[2]<0
            diagnostic=Libdl.dlsym(
                Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error
            )
            @test ccall(diagnostic, Cint, ()) == -info[2]
        end
        return info[2]
    end
end

@testset "Precision conversion boundaries, selected storage, and ABIs" begin
    reference=only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
    for (family, S, D) in (
            ("slag2d", Float32, Float64),
            ("dlag2s", Float64, Float32),
            ("clag2z", ComplexF32, ComplexF64),
            ("zlag2c", ComplexF64, ComplexF32),
            ("dlat2s", Float64, Float32),
            ("zlat2c", ComplexF64, ComplexF32),
        ),
        J in (Int32, Int64), m0 in 0:5, n in 0:5,
        u in ('U', 'L')

        triangle=occursin("lat", family)
        m=triangle ? n : m0
        narrow=sizeof(S)>sizeof(D)
        R=typeof(real(zero(S)))
        limit=R(floatmax(Float32))
        values=R[
            0, -0.0, nextfloat(R(0)), floatmin(R), -floatmin(R), 1, -1, limit, -limit, NaN
        ]
        narrow || append!(values, R[Inf, -Inf])
        for scenario in 1:7
            a=fill(S(Inf), m+2, n+1)
            b=fill(D(777), m+3, n+1)
            expected=copy(b)
            selected=[
                (i, j) for j in 1:n for i in 1:m if !triangle || (u=='U' ? i<=j : i>=j)
            ]
            for (index, (i, j)) in enumerate(selected)
                x=values[mod1(index, length(values))]
                y=values[mod1(index+3, length(values))]
                a[i, j]=S <: Complex ? S(x, y) : S(x)
            end
            if narrow && scenario>1 && !isempty(selected)
                bad=if scenario==2
                    first(selected)
                elseif scenario==3
                    selected[cld(length(selected), 2)]
                else
                    last(selected)
                end
                overflow = if scenario in (2, 3)
                    nextfloat(limit)
                elseif scenario in (4, 5)
                    -nextfloat(limit)
                elseif scenario == 6
                    R(Inf)
                else
                    R(-Inf)
                end
                a[bad...] = if S <: Complex
                    (isodd(scenario) ? S(0, overflow) : S(overflow, 0))
                else
                    S(overflow)
                end
            end
            info_expected=0
            for (i, j) in selected
                value=a[i, j]
                if narrow && (
                    real(value)<-limit ||
                    real(value)>limit ||
                    imag(value)<-limit ||
                    imag(value)>limit
                )
                    info_expected=1
                    break
                end
                expected[i, j]=D(value)
            end
            before=copy(a)
            @test direct_precision!(Val(Symbol(family)), a, b, J; m, n, uplo=u)==info_expected
            @test isequal(a, before)
            @test isequal(b, expected)
            if J===Int64
                external=fill(D(777), size(b))
                # LAT2 helpers in LAPACK 3.12.1 leave INFO unchanged on success;
                # initialize it to zero for differential execution. Our entry
                # always writes the documented successful INFO=0.
                @test direct_precision!(
                    Val(Symbol(family)),
                    a,
                    external,
                    J;
                    m,
                    n,
                    uplo=u,
                    library=reference,
                    initial=0,
                )==info_expected
                @test isequal(b, external)
            end
        end
    end
end
