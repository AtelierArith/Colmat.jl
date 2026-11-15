using Test, LinearAlgebra, Libdl, Colmat
if !isdefined(@__MODULE__, :direct_storage_conversion!)
    include("storage_conversions.jl")
end
for (T, R, p, family) in (
        (Float32, Float32, "s", "sfrk"),
        (Float64, Float64, "d", "sfrk"),
        (ComplexF32, Float32, "c", "hfrk"),
        (ComplexF64, Float64, "z", "hfrk"),
    ),
    J in (Int32, Int64)

    suffix=J===Int64 ? "64_" : ""
    @eval function direct_rfp_rank!(
        a::Matrix{$T},
        c::Vector{$T},
        ::Type{$J};
        trr,
        u,
        tr,
        n,
        k,
        alpha,
        beta,
        lda=size(a, 1),
        library=Colmat.backend_library(),
    )
        f=Libdl.dlsym(Libdl.dlopen(library), Symbol($p*$family*"_"*$suffix))
        target=view(c, 2:(length(c) - 1))
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{UInt8},
                Ref{UInt8},
                Ref{$J},
                Ref{$J},
                Ref{$R},
                Ptr{$T},
                Ref{$J},
                Ref{$R},
                Ptr{$T},
            ),
            trr,
            u,
            tr,
            n,
            k,
            alpha,
            a,
            lda,
            beta,
            target,
        )
    end
end

function rfp_rank_fixture(T, n, k, u, tr, trr, alpha, beta)
    R=typeof(real(zero(T)))
    hermitian=T <: Complex
    rows, cols=tr=='N' ? (n, k) : (k, n)
    a=fill(T(NaN), rows+2, cols+1)
    op=zeros(T, n, k)
    if alpha!=0&&k>0
        for j in 1:cols, i in 1:rows
            a[i, j]=T <: Complex ? T((i+2j)/16, (2i-j)/32) : T((i+2j)/16)
        end
        for j in 1:k, i in 1:n
            op[i, j]=if tr=='N'
                a[i, j]
            elseif tr=='T'
                a[j, i]
            else
                conj(a[j, i])
            end
        end
    end
    base=zeros(T, n, n)
    source=fill(T(777), n+2, n)
    for j in 1:n
        indices=u=='U' ? (1:j) : (j:n)
        for i in indices
            value=T <: Complex ? T((2i+j)/16, (i-j)/8) : T((2i+j)/16)
            source[i, j]=hermitian&&i==j ? T(real(value), NaN) : value
            base[i, j]=hermitian&&i==j ? T(real(value), 0) : value
            base[j, i]=hermitian ? conj(base[i, j]) : base[i, j]
        end
    end
    c=fill(T(777), n*(n+1)÷2+2)
    @test direct_storage_conversion!(
        "trttf", source, c, Int64; n, uplo=u, transr=trr, library=conversion_reference()
    )==0
    if beta==0
        c[2:(end - 1)].=T <: Complex ? T(NaN, NaN) : T(NaN)
    end
    return (; a, c, base, op)
end

@testset "RFP symmetric/Hermitian rank-k storage and scalar contracts" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        n in 0:9,
        k in (0, 1, 3, 7),
        u in ('U', 'L'),
        trr in (T <: Complex ? ('N', 'C') : ('N', 'T')),
        tr in (T <: Complex ? ('N', 'C') : ('N', 'T')),
        coefficients in 1:5

        R=typeof(real(zero(T)))
        alpha=if coefficients in (1, 2, 5)
            R(0)
        elseif coefficients==3
            R(1)
        else
            R(0.5)
        end
        beta=if coefficients==1
            R(1)
        elseif coefficients in (2, 3)
            R(0)
        elseif coefficients==4
            R(-0.25)
        else
            R(0.5)
        end
        fixture=rfp_rank_fixture(T, n, k, u, tr, trr, alpha, beta)
        a=fixture.a
        c=copy(fixture.c)
        beforea=copy(a)
        direct_rfp_rank!(a, c, J; trr, u, tr, n, k, alpha, beta)
        @test isequal(a, beforea)
        @test c[1]==c[end]==777
        if n==0 || ((alpha==0||k==0)&&beta==1)
            @test isequal(c, fixture.c)
        else
            actual=fill(T(777), n+2, n)
            @test direct_storage_conversion!(
                "tfttr",
                c,
                actual,
                Int64;
                n,
                uplo=u,
                transr=trr,
                library=conversion_reference(),
            )==0
            for j in 1:n, i in 1:(n + 2)
                selected=i<=n && (u=='U' ? i<=j : i>=j)
                if selected
                    product=if alpha==0||k==0
                        zero(T)
                    else
                        alpha*sum(
                        fixture.op[i, l]*(
                            T <: Complex ? conj(fixture.op[j, l]) : fixture.op[j, l]
                        ) for l in 1:k
                    )
                    end
                    expected=(beta==0 ? zero(T) : beta*fixture.base[i, j])+product
                    @test isapprox(actual[i, j], expected; rtol=200eps(R), atol=200eps(R))
                    if T <: Complex && i==j
                        @test imag(actual[i, j])==0
                    end
                else
                    @test actual[i, j]==777
                end
            end
            if alpha==0&&beta==0
                @test all(x->isequal(x, zero(T)), c[2:(end - 1)])
            end
        end
        if J===Int64
            external=copy(fixture.c)
            direct_rfp_rank!(
                a,
                external,
                J;
                trr,
                u,
                tr,
                n,
                k,
                alpha,
                beta,
                library=conversion_reference(),
            )
            @test isapprox(c, external; rtol=200eps(R), atol=200eps(R), nans=true)
            @test external[1]==external[end]==777
        end
    end
end

@testset "All four RFP rank-k symbols numerically through LBT" begin
    for (T, p, family) in (
            (Float32, "s", "sfrk"),
            (Float64, "d", "sfrk"),
            (ComplexF32, "c", "hfrk"),
            (ComplexF64, "z", "hfrk"),
        ),
        J in (Int32, Int64)

        R=typeof(real(zero(T)))
        n=5
        k=3
        tr=T <: Complex ? 'C' : 'T'
        trr=tr
        fixture=rfp_rank_fixture(T, n, k, 'U', tr, trr, R(0.5), R(-0.25))
        c=copy(fixture.c)
        expected=copy(c)
        backing=BLAS.lbt_find_backing_library(p*family*"_", J===Int32 ? :lp64 : :ilp64)
        @test backing!==nothing
        @test realpath(backing.libname)==realpath(Colmat.backend_library())
        direct_rfp_rank!(
            fixture.a,
            c,
            J;
            trr,
            u=('U'),
            tr,
            n,
            k,
            alpha=R(0.5),
            beta=R(-0.25),
            library=BLAS.libblastrampoline,
        )
        direct_rfp_rank!(
            fixture.a,
            expected,
            Int64;
            trr,
            u=('U'),
            tr,
            n,
            k,
            alpha=R(0.5),
            beta=R(-0.25),
            library=conversion_reference(),
        )
        @test c≈expected
    end
end

@testset "RFP rank-k invalid arguments preserve output" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        R=typeof(real(zero(T)))
        a=fill(T(NaN), 4, 3)
        c=fill(T(777), 8)
        base=(trr=('N'), u=('U'), tr=('N'), n=3, k=2, alpha=R(1), beta=R(1))
        for (override, arg) in (
            ((trr=('?'),), 1),
            ((u=('?'),), 2),
            ((tr=('?'),), 3),
            ((n=-1,), 4),
            ((k=-1,), 5),
            ((lda=1,), 8),
        )
            before=copy(c)
            direct_rfp_rank!(a, c, J; merge(base, override)...)
            f=Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error)
            @test ccall(f, Cint, ())==arg
            @test isequal(c, before)
        end
    end
end
