using Test, LinearAlgebra, Random, Libdl, Colmat
for (T, p, kind) in (
        (Float32, "s", "sy"),
        (Float64, "d", "sy"),
        (ComplexF32, "c", "he"),
        (ComplexF64, "z", "he"),
    ),
    J in (Int32, Int64)

    suffix=J===Int64 ? "64_" : ""
    factor=Symbol(p*kind*"trf_"*suffix)
    plain=Symbol(p*kind*"tf2_"*suffix)
    solve=Symbol(p*kind*"trs_"*suffix)
    solve2=Symbol(p*kind*"trs2_"*suffix)
    @eval function direct_bk_factor!(
        a::Matrix{$T},
        ::Type{$J};
        uplo=('L'),
        unblocked=false,
        query=false,
        library=Colmat.backend_library(),
    )
        n=size(a, 2)
        piv=fill($J(777), n+2)
        work=zeros($T, 1)
        info=Ref{$J}(0)
        f=Libdl.dlsym(
            Libdl.dlopen(library), unblocked ? $(QuoteNode(plain)) : $(QuoteNode(factor))
        )
        if unblocked
            ccall(
                f,
                Cvoid,
                (Ref{UInt8}, Ref{$J}, Ptr{$T}, Ref{$J}, Ptr{$J}, Ref{$J}),
                uplo,
                n,
                a,
                max(1, size(a, 1)),
                view(piv, 2:(n + 1)),
                info,
            )
        else
            ccall(
                f,
                Cvoid,
                (Ref{UInt8}, Ref{$J}, Ptr{$T}, Ref{$J}, Ptr{$J}, Ptr{$T}, Ref{$J}, Ref{$J}),
                uplo,
                n,
                a,
                max(1, size(a, 1)),
                view(piv, 2:(n + 1)),
                work,
                query ? -1 : 1,
                info,
            )
        end
        return piv[2:(n + 1)], info[], work, piv[[1, end]]
    end
    @eval function direct_bk_solve!(
        a::Matrix{$T},
        piv::Vector{$J},
        b::Matrix{$T};
        uplo=('L'),
        work_variant=false,
        library=Colmat.backend_library(),
    )
        n=size(a, 2)
        nrhs=size(b, 2)
        work=zeros($T, n)
        info=Ref{$J}(0)
        f=Libdl.dlsym(
            Libdl.dlopen(library), work_variant ? $(QuoteNode(solve2)) : $(QuoteNode(solve))
        )
        if work_variant
            ccall(
                f,
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{$J},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                ),
                uplo,
                n,
                nrhs,
                a,
                max(1, size(a, 1)),
                piv,
                b,
                max(1, size(b, 1)),
                work,
                info,
            )
        else
            ccall(
                f,
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{$J},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ref{$J},
                ),
                uplo,
                n,
                nrhs,
                a,
                max(1, size(a, 1)),
                piv,
                b,
                max(1, size(b, 1)),
                info,
            )
        end
        return info[]
    end
end
function bk_input(T, n)
    x=randn(T, n, n)/max(n, 1)
    a=x+x'
    for i in 1:n
        a[i, i]=isodd(i) ? i+3 : -i-3
    end
    if n>=2
        a[1, 1]=0
        a[2, 2]=0
        a[2, 1]=T<:Real ? T(3) : T(3+0.2im)
        a[1, 2]=conj(a[2, 1])
    end
    if n>=5
        a[4, 4]=0
        a[5, 5]=7
        a[5, 4]=2
        a[4, 5]=2
    end
    return a
end
@testset "Bunch-Kaufman factor/solve direct ABI and external factors" begin
    Random.seed!(266)
    library=only(
        filter(
            l->l.interface==:ilp64&&occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), uplo in ('L', 'U'), unblocked in (false, true),
        n in (0, 1, 5, 80)

        original=bk_input(T, n)
        a=fill(T(999), n+2, n)
        a[1:n, :]=original
        for j in 1:n, i in 1:n
            if (uplo=='L'&&i<j)||(uplo=='U'&&i>j)
                a[i, j]=T(NaN)
            end
        end
        before=copy(a)
        piv, info, _, guards=direct_bk_factor!(a, J; uplo, unblocked)
        @test info==0
        @test guards==fill(J(777), 2)
        @test all(==(T(999)), a[(n + 1):end, :])
        n>=2 && @test any(<(0), piv)
        @test isequal(
            uplo=='L' ? triu(a[1:n, :], 1) : tril(a[1:n, :], -1),
            uplo=='L' ? triu(before[1:n, :], 1) : tril(before[1:n, :], -1),
        )
        rhs=randn(T, n, 3)
        for work_variant in (false, true)
            b=fill(T(888), n+2, 3)
            b[1:n, :]=rhs
            saved=copy(a)
            @test direct_bk_solve!(a, piv, b; uplo, work_variant)==0
            @test original*b[1:n, :] ≈ rhs
            @test all(==(T(888)), b[(n + 1):end, :])
            @test isequal(a, saved)
        end
        # Colmat factors consumed by the independent LAPACK solver.
        external_solution=copy(rhs)
        @test direct_bk_solve!(a, Int64.(piv), external_solution; uplo, library)==0
        @test original*external_solution ≈ rhs
        # Independently produced A/IPIV consumed by Colmat, including its row swaps.
        reference=copy(original)
        rp, ri, _, _=direct_bk_factor!(reference, Int64; uplo, unblocked=true, library)
        @test ri==0
        solution=copy(rhs)
        @test direct_bk_solve!(reference, J.(rp), solution; uplo)==0
        @test original*solution ≈ rhs
    end
end

for (T, p, kind) in (
        (Float32, "s", "sy"),
        (Float64, "d", "sy"),
        (ComplexF32, "c", "he"),
        (ComplexF64, "z", "he"),
    ),
    J in (Int32, Int64)

    suffix=J===Int64 ? "64_" : ""
    for family in (:tri, :tri2, :tri2x)
        symbol=Symbol(p*kind*string(family)*"_"*suffix)
        @eval function direct_bk_inverse!(
            a::Matrix{$T},
            piv::Vector{$J},
            ::Val{$(QuoteNode(family))};
            uplo=('L'),
            query=false,
            nb=2,
            lwork_override=nothing,
            library=Colmat.backend_library(),
        )
            n=size(a, 2)
            count=$(family===:tri2x) ? max(1, (n+nb+1)*(nb+3)) : max(1, n)
            work=zeros($T, count)
            info=Ref{$J}(0)
            f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol)))
            if $(family===:tri)
                ccall(
                    f,
                    Cvoid,
                    (Ref{UInt8}, Ref{$J}, Ptr{$T}, Ref{$J}, Ptr{$J}, Ptr{$T}, Ref{$J}),
                    uplo,
                    n,
                    a,
                    max(1, size(a, 1)),
                    piv,
                    work,
                    info,
                )
            else
                hint=if $(family===:tri2x)
                    nb
                elseif query
                    -1
                elseif isnothing(lwork_override)
                    length(work)
                else
                    lwork_override
                end
                ccall(
                    f,
                    Cvoid,
                    (
                        Ref{UInt8},
                        Ref{$J},
                        Ptr{$T},
                        Ref{$J},
                        Ptr{$J},
                        Ptr{$T},
                        Ref{$J},
                        Ref{$J},
                    ),
                    uplo,
                    n,
                    a,
                    max(1, size(a, 1)),
                    piv,
                    work,
                    hint,
                    info,
                )
            end
            return info[], work
        end
    end
    symbol=Symbol(p*kind*"sv_"*suffix)
    @eval function direct_bk_driver!(
        a::Matrix{$T},
        b::Matrix{$T},
        ::Type{$J};
        uplo=('L'),
        query=false,
        lwork_override=nothing,
        library=Colmat.backend_library(),
    )
        n=size(a, 2)
        nrhs=size(b, 2)
        piv=fill($J(777), n+2)
        work=zeros($T, 1)
        info=Ref{$J}(0)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol)))
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{$J},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ref{$J},
            ),
            uplo,
            n,
            nrhs,
            a,
            max(1, size(a, 1)),
            view(piv, 2:(n + 1)),
            b,
            max(1, size(b, 1)),
            work,
            if query
                -1
            elseif isnothing(lwork_override)
                1
            else
                lwork_override
            end,
            info,
        )
        return piv[2:(n + 1)], info[], work, piv[[1, end]]
    end
end

@testset "Bunch-Kaufman inverse families and drivers" begin
    Random.seed!(2661)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), uplo in ('L', 'U'),
        n in (0, 1, 5, 80)

        original=bk_input(T, n)
        factor=fill(T(999), n+2, n)
        factor[1:n, :]=original
        piv, info, _, _=direct_bk_factor!(factor, J; uplo)
        @test info==0
        for family in (:tri, :tri2, :tri2x)
            inverse=copy(factor)
            before=copy(inverse)
            info, _=direct_bk_inverse!(inverse, piv, Val(family); uplo)
            @test info==0
            result=Matrix(Hermitian(inverse[1:n, :], Symbol(uplo)))
            @test original*result ≈ Matrix{T}(I, n, n)
            @test all(==(T(999)), inverse[(n + 1):end, :])
            @test isequal(
                uplo=='L' ? triu(inverse[1:n, :], 1) : tril(inverse[1:n, :], -1),
                uplo=='L' ? triu(before[1:n, :], 1) : tril(before[1:n, :], -1),
            )
            @test all(iszero, imag.(diag(inverse[1:n, :])))
        end
        rhs=randn(T, n, 2)
        a=copy(original)
        b=copy(rhs)
        piv, info, _, guards=direct_bk_driver!(a, b, J; uplo)
        @test info==0
        @test original*b ≈ rhs
        @test guards==fill(J(777), 2)
        saved=map(copy, (a, b))
        qp, info, work, _=direct_bk_driver!(a, b, J; uplo, query=true)
        @test info==0 && real(work[1])>=1
        @test all(==(J(777)), qp)
        @test all(isequal.(saved, (a, b)))
    end
end
@testset "Bunch-Kaufman singular INFO, query and independent inverse" begin
    library=only(
        filter(
            l->l.interface==:ilp64&&occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        uplo in ('L', 'U')

        original=T[0 2 0; 2 0 1; 0 1 3]
        a=copy(original)
        piv, info, _, _=direct_bk_factor!(a, J; uplo)
        @test info==0
        reference=copy(a)
        @test first(direct_bk_inverse!(reference, Int64.(piv), Val(:tri); uplo, library))==0
        for family in (:tri, :tri2, :tri2x)
            ours=copy(a)
            @test first(direct_bk_inverse!(ours, piv, Val(family); uplo))==0
            @test (uplo=='L' ? tril(ours) : triu(ours)) ≈
                (uplo=='L' ? tril(reference) : triu(reference))
        end
        saved=copy(a)
        info, work=direct_bk_inverse!(a, piv, Val(:tri2); uplo, query=true)
        @test info==0 && real(work[1])>=3
        @test a==saved
        info, _=direct_bk_inverse!(a, piv, Val(:tri2); uplo, lwork_override=2)
        @test info == -7 && a==saved
        for original in (zeros(T, 3, 3), Matrix(Diagonal(T[1, 0, 2])))
            a=copy(original)
            rhs=ones(T, 3, 2)
            b=copy(rhs)
            piv, info, _, _=direct_bk_driver!(a, b, J; uplo)
            @test info>0
            @test b==rhs
            reference=copy(original)
            _, expected, _, _=direct_bk_factor!(
                reference, Int64; uplo, unblocked=true, library
            )
            @test info==expected
            for family in (:tri, :tri2, :tri2x)
                before=copy(a)
                inverse_info, _=direct_bk_inverse!(a, piv, Val(family); uplo)
                @test inverse_info==info
                @test isequal(a, before)
            end
        end
    end
end

@testset "Bunch-Kaufman queries, invalid arguments and ignored regions" begin
    Random.seed!(2662)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        uplo in ('L', 'U')

        a=fill(T(NaN), 4, 4)
        before=copy(a)
        piv, info, work, guards=direct_bk_factor!(a, J; uplo, query=true)
        @test info==0 && real(work[1])>=1
        @test all(==(J(777)), piv) && guards==fill(J(777), 2)
        @test isequal(a, before)
        _, info, _, _=direct_bk_factor!(a, J; uplo=('?'))
        @test info == -1 && isequal(a, before)
        rectangle=ones(T, 2, 3)
        @test direct_bk_factor!(rectangle, J; uplo)[2] == -4
        original=bk_input(T, 5)
        a=copy(original)
        if T<:Complex
            for i in 1:5
                a[i, i]=T(real(a[i, i]), NaN)
            end
        end
        piv, info, _, _=direct_bk_factor!(a, J; uplo)
        @test info==0 && all(iszero, imag.(diag(a)))
        b=ones(T, 5, 1)
        saved=copy(b)
        @test direct_bk_solve!(a, piv, b; uplo=('?')) == -1
        @test b==saved
        @test direct_bk_solve!(a, piv, ones(T, 4, 1); uplo) == -8
        for family in (:tri, :tri2, :tri2x)
            before=copy(a)
            @test first(direct_bk_inverse!(a, piv, Val(family); uplo=('?'))) == -1
            @test a==before
        end
        @test direct_bk_driver!(copy(original), copy(b), J; uplo, lwork_override=0)[2] ==
            -10
        prefix=if T===Float32
            "s"
        elseif T===Float64
            "d"
        elseif T===ComplexF32
            "c"
        else
            "z"
        end
        kind=T<:Real ? "sy" : "he"
        for family in ("trf", "tf2", "trs", "trs2", "tri", "tri2", "tri2x", "sv")
            backing=BLAS.lbt_find_backing_library(
                prefix*kind*family*"_", J===Int32 ? :lp64 : :ilp64
            )
            @test backing!==nothing &&
                realpath(backing.libname)==realpath(Colmat.backend_library())
        end
    end
end
@testset "Bunch-Kaufman Julia factors and scaled two-column pivots" begin
    Random.seed!(2663)
    for T in (Float32, Float64, ComplexF32, ComplexF64), uplo in ('L', 'U')
        original=bk_input(T, 7)
        f=bunchkaufman(Hermitian(original, Symbol(uplo)), false)
        unit=uplo=='L' ? f.L : f.U
        @test unit*f.D*unit' ≈ original[f.p, f.p]
        rhs=randn(T, 7, 2)
        @test original*(f\rhs) ≈ rhs
        @test original*inv(f) ≈ Matrix{T}(I, 7, 7)
    end
    for T in (ComplexF32, ComplexF64)
        original=Matrix(Symmetric(randn(T, 5, 5)))
        f=bunchkaufman(Symmetric(original), false)
        rhs=randn(T, 5, 2)
        @test original*(f\rhs) ≈ rhs
        backing=BLAS.lbt_find_backing_library(
            T===ComplexF32 ? "csytrf_" : "zsytrf_", :ilp64
        )
        @test backing!==nothing &&
            realpath(backing.libname)!=realpath(Colmat.backend_library())
    end
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        uplo in ('L', 'U')

        R=typeof(real(zero(T)))
        base=T<:Real ? T[0 2; 2 0] : T[0 2+im; 2-im 0]
        scales=R===Float32 ? (R(1e-20), R(1e20)) : (R(1e-200), R(1e200))
        for scale in scales
            a=base*scale
            b=ones(T, 2, 1)*scale
            piv, info, _, _=direct_bk_driver!(a, b, J; uplo)
            @test info==0
            @test base*b ≈ ones(T, 2, 1)
            @test any(<(0), piv)
            @test first(direct_bk_inverse!(a, piv, Val(:tri); uplo))==0
            result=Matrix(Hermitian(a, Symbol(uplo)))*scale
            @test base*result ≈ Matrix{T}(I, 2, 2)
        end
    end
end
