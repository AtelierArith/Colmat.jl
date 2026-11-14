using Test, LinearAlgebra, Random, Libdl, Colmat
for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    R=typeof(real(zero(T)))
    suffix=J===Int64 ? "64_" : ""
    bal=Symbol(p*"gebal_"*suffix)
    bak=Symbol(p*"gebak_"*suffix)
    @eval function direct_balance!(
        a::Matrix{$T},
        ::Type{$J};
        job=('B'),
        n_override=nothing,
        library=Colmat.backend_library(),
    )
        n=size(a, 2)
        ilo=Ref{$J}(-777)
        ihi=Ref{$J}(-777)
        scale=fill($R(777), n+2)
        info=Ref{$J}(0)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(bal)))
        ccall(
            f,
            Cvoid,
            (Ref{UInt8}, Ref{$J}, Ptr{$T}, Ref{$J}, Ref{$J}, Ref{$J}, Ptr{$R}, Ref{$J}),
            job,
            isnothing(n_override) ? n : n_override,
            a,
            max(1, size(a, 1)),
            ilo,
            ihi,
            view(scale, 2:(n + 1)),
            info,
        )
        return ilo[], ihi[], scale[2:(n + 1)], info[], scale[[1, end]]
    end
    @eval function direct_backbalance!(
        v::Matrix{$T},
        scale::Vector{$R},
        ::Type{$J};
        job=('B'),
        side=('R'),
        lo=1,
        hi=length(scale),
        m_override=nothing,
        library=Colmat.backend_library(),
    )
        n=length(scale)
        info=Ref{$J}(0)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(bak)))
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{UInt8},
                Ref{$J},
                Ref{$J},
                Ref{$J},
                Ptr{$R},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ref{$J},
            ),
            job,
            side,
            n,
            lo,
            hi,
            scale,
            isnothing(m_override) ? size(v, 2) : m_override,
            v,
            max(1, size(v, 1)),
            info,
        )
        return info[]
    end
end
function balance_fixture(T)
    a=zeros(T, 6, 6)
    block=randn(T, 4, 4)
    weights=2.0 .^ [-12, 3, 8, -6]
    for j in 1:4, i in 1:4
        a[i + 1, j + 1]=block[i, j]*weights[i]/weights[j]
    end
    a[1, 1]=11
    a[6, 6]=13
    a[1, 2:5].=1
    a[2:5, 6].=1
    order=[4, 1, 5, 2, 6, 3]
    return a[order, order]
end
@testset "GEBAL/GEBAK similarity and metadata interchange" begin
    Random.seed!(267)
    library=only(
        filter(
            l->l.interface==:ilp64&&occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        job in ('N', 'P', 'S', 'B')

        for original in
            (zeros(T, 0, 0), fill(T(2), 1, 1), T[1 0 0; 2 3 0; 4 5 6], balance_fixture(T))
            n=size(original, 2)
            a=fill(T(999), n+2, n)
            a[1:n, :]=original
            lo, hi, scale, info, guards=direct_balance!(a, J; job)
            @test info==0
            @test (n==0 ? (lo, hi)==(1, 0) : 1<=lo<=hi<=n)
            @test guards==fill(typeof(real(zero(T)))(777), 2)
            @test all(==(T(999)), a[(n + 1):end, :])
            right=Matrix{T}(I, n, n)
            left=copy(right)
            @test direct_backbalance!(right, scale, J; job, side=('R'), lo, hi)==0
            @test direct_backbalance!(left, scale, J; job, side=('L'), lo, hi)==0
            @test left'*original*right ≈ a[1:n, :]
            @test left'*right ≈ Matrix{T}(I, n, n)
            if job=='N'
                @test a[1:n, :]==original && scale==ones(n)
            end
            for side in ('L', 'R')
                values=randn(T, n, 3)
                ours=copy(values)
                reference=copy(values)
                @test direct_backbalance!(ours, scale, J; job, side, lo, hi)==0
                @test direct_backbalance!(
                    reference, scale, Int64; job, side, lo, hi, library
                )==0
                @test ours ≈ reference
                # Independently encoded SCALE and active range consumed by Colmat.
                external=copy(original)
                rlo, rhi, rscale, ri, _=direct_balance!(external, Int64; job, library)
                @test ri==0
                ours=copy(values)
                reference=copy(values)
                @test direct_backbalance!(ours, rscale, J; job, side, lo=rlo, hi=rhi)==0
                @test direct_backbalance!(
                    reference, rscale, Int64; job, side, lo=rlo, hi=rhi, library
                )==0
                @test ours ≈ reference
            end
        end
    end
end

@testset "GEBAK referenced regions and argument validation" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        R=typeof(real(zero(T)))
        for side in ('L', 'R')
            v=fill(T(NaN), 4, 2)
            v[2:3, :]=T[1 2; 3 4]
            before=copy(v)
            scale=R[NaN, 2, 0.5, NaN]
            @test direct_backbalance!(v, scale, J; job=('S'), side, lo=2, hi=3)==0
            factors=side=='R' ? R[2, 0.5] : R[0.5, 2]
            @test v[2:3, :]==before[2:3, :] .* factors
            @test isequal(v[[1, 4], :], before[[1, 4], :])
            v=fill(T(NaN), 4, 2)
            v[1, :]=T[1, 2]
            v[4, :]=T[7, 8]
            before=copy(v)
            @test direct_backbalance!(
                v, R[4, NaN, NaN, 4], J; job=('P'), side, lo=2, hi=3
            )==0
            @test v[[1, 4], :]==before[[4, 1], :]
            @test isequal(v[2:3, :], before[2:3, :])
        end
        v=fill(T(NaN), 4, 2)
        before=copy(v)
        scale=fill(R(NaN), 4)
        @test direct_backbalance!(v, scale, J; job=('N'))==0
        @test isequal(v, before)
        @test direct_backbalance!(zeros(T, 4, 0), scale, J; job=('B'))==0
        @test direct_backbalance!(v, scale, J; job=('?')) == -1
        @test direct_backbalance!(v, scale, J; side=('?')) == -2
        @test direct_backbalance!(v, scale, J; lo=0) == -4
        @test direct_backbalance!(v, scale, J; hi=5) == -5
        @test direct_backbalance!(v, scale, J; m_override=-1) == -7
        if J===Int64
            @test direct_backbalance!(
                v, scale, J; job=('N'), m_override=(-(Int64(1)<<32))
            ) == -7
        end
        @test direct_backbalance!(zeros(T, 3, 2), scale, J) == -9
        @test isequal(v, before)
        a=fill(T(NaN), 4, 4)
        before=copy(a)
        lo, hi, s, info, _=direct_balance!(a, J; job=('N'))
        @test info==0 && (lo, hi)==(1, 4) && s==ones(4)
        @test isequal(a, before)
        @test direct_balance!(a, J; job=('S'))[4] == -3
        @test direct_balance!(a, J; job=('?'))[4] == -1
        @test direct_balance!(a, J; n_override=-1)[4] == -2
        if J===Int64
            @test direct_balance!(a, J; n_override=(-(Int64(1)<<32)))[4] == -2
        end
        @test direct_balance!(zeros(T, 3, 4), J)[4] == -4
        prefix=if T===Float32
            "s"
        elseif T===Float64
            "d"
        elseif T===ComplexF32
            "c"
        else
            "z"
        end
        for family in ("gebal", "gebak")
            backing=BLAS.lbt_find_backing_library(
                prefix*family*"_", J===Int32 ? :lp64 : :ilp64
            )
            @test backing!==nothing &&
                realpath(backing.libname)==realpath(Colmat.backend_library())
        end
    end
end
@testset "Balancing Julia wrappers and singleton GEEVX range" begin
    Random.seed!(2671)
    for T in (Float32, Float64, ComplexF32, ComplexF64), job in ('N', 'P', 'S', 'B')
        original=balance_fixture(T)
        balanced=copy(original)
        lo, hi, scale=LAPACK.gebal!(job, balanced)
        right=Matrix{T}(I, 6, 6)
        left=copy(right)
        LAPACK.gebak!(job, 'R', lo, hi, scale, right)
        LAPACK.gebak!(job, 'L', lo, hi, scale, left)
        @test left'*original*right ≈ balanced
        @test left'*right ≈ Matrix{T}(I, 6, 6)
    end
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        result=LAPACK.geevx!('B', 'N', 'N', 'N', fill(T(2), 1, 1))
        index=T<:Real ? 6 : 5
        @test (result[index], result[index + 1])==(1, 1)
    end
end
