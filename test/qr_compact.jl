using Test, LinearAlgebra, Random, Colmat
@testset "Native compact QR Julia entry points" begin
    Random.seed!(290)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        (m, n) in ((6, 4), (4, 6), (4, 4), (35, 34), (0, 3), (3, 0)),
        degenerate in (false, true)

        a=randn(T, m, n)
        if degenerate && n>0
            a[:, 1].=0
            n>2 && (a[:, 3].=a[:, 2])
        end
        f=qr(a)
        q=Matrix(f.Q)
        @test q*f.R ≈ a
        @test q'*q ≈ Matrix{T}(I, min(m, n), min(m, n))
        fullq=f.Q*Matrix{T}(I, m, m)
        @test fullq'*fullq ≈ Matrix{T}(I, m, m)
        c=randn(T, m, 3)
        @test f.Q*c ≈ fullq*c
        @test f.Q'*c ≈ fullq'*c
        c=randn(T, 3, m)
        @test c*f.Q ≈ c*fullq
        @test c*f.Q' ≈ c*fullq'
    end
    for p in ("s", "d", "c", "z"),
        family in ("geqrt", "geqrt2", "geqrt3", "gemqrt"),
        abi in (:lp64, :ilp64)

        @test realpath(BLAS.lbt_find_backing_library(p*family*"_", abi).libname)==realpath(
            Colmat.backend_library()
        )
    end
end

using Libdl
for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    I in (Int32, Int64)

    suffix=I===Int64 ? "64_" : ""
    syms=Tuple(Symbol(p*f*"_"*suffix) for f in ("geqrt", "geqrt2", "geqrt3", "gemqrt"))
    @eval function direct_geqrt!(
        a::Matrix{$T}, m, n, nb, ::Type{$I}; variant=1, library=Colmat.backend_library()
    )
        t=fill($T(NaN), nb+1, n)
        work=zeros($T, max(1, nb*n))
        info=Ref{$I}(0)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(syms))[variant])
        if variant==1
            ccall(
                f,
                Cvoid,
                (
                    Ref{$I},
                    Ref{$I},
                    Ref{$I},
                    Ptr{$T},
                    Ref{$I},
                    Ptr{$T},
                    Ref{$I},
                    Ptr{$T},
                    Ref{$I},
                ),
                m,
                n,
                nb,
                a,
                size(a, 1),
                t,
                size(t, 1),
                work,
                info,
            )
        else
            ccall(
                f,
                Cvoid,
                (Ref{$I}, Ref{$I}, Ptr{$T}, Ref{$I}, Ptr{$T}, Ref{$I}, Ref{$I}),
                m,
                n,
                a,
                size(a, 1),
                t,
                size(t, 1),
                info,
            )
        end
        t, info[]
    end
    @eval function direct_gemqrt!(
        side,
        tr,
        m,
        n,
        k,
        nb,
        v::Matrix{$T},
        t::Matrix{$T},
        c::Matrix{$T},
        ::Type{$I};
        library=Colmat.backend_library(),
    )
        work=zeros($T, max(1, nb*max(m, n)))
        info=Ref{$I}(0)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(syms[4])))
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{UInt8},
                Ref{$I},
                Ref{$I},
                Ref{$I},
                Ref{$I},
                Ptr{$T},
                Ref{$I},
                Ptr{$T},
                Ref{$I},
                Ptr{$T},
                Ref{$I},
                Ptr{$T},
                Ref{$I},
            ),
            side,
            tr,
            m,
            n,
            k,
            nb,
            v,
            size(v, 1),
            t,
            size(t, 1),
            c,
            size(c, 1),
            work,
            info,
        )
        c, info[]
    end
end
@testset "Compact QR direct integer ABIs and external factors" begin
    external=BLAS.lbt_find_backing_library("dgeqrf_", :ilp64).libname
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        I in (Int32, Int64), nb in (1, 2, 4),
        variant in 1:3

        m, n=7, 4
        variant>1 && nb!=n && continue
        original=randn(T, m, n)
        a=fill(T(NaN), m+2, n)
        a[1:m, :]=original
        t, info=direct_geqrt!(a, m, n, nb, I; variant)
        @test info==0
        @test all(isnan, a[(m + 1):end, :])
        @test all(isnan, t[nb + 1, :])
        q, info=direct_gemqrt!(
            'L', 'N', m, m, n, nb, a, t, Matrix{T}(LinearAlgebra.I, m, m), I
        )
        @test info==0
        @test q[:, 1:n]*triu(a[1:n, :]) ≈ original
        @test q'*q ≈ Matrix{T}(LinearAlgebra.I, m, m)
        # Apply native factors using the independently loaded ILP64 OpenBLAS.
        externalq, info=direct_gemqrt!(
            'L',
            'N',
            m,
            m,
            n,
            nb,
            a,
            t,
            Matrix{T}(LinearAlgebra.I, m, m),
            Int64;
            library=external,
        )
        @test info==0
        @test externalq ≈ q
        ea=copy(original)
        et, info=direct_geqrt!(ea, m, n, nb, Int64; variant, library=external)
        @test info==0
        eq, info=direct_gemqrt!(
            'L', 'N', m, m, n, nb, ea, et, Matrix{T}(LinearAlgebra.I, m, m), I
        )
        @test info==0
        @test eq[:, 1:n]*triu(ea[1:n, :]) ≈ original
        for side in ('L', 'R'), tr in ('N', T<:Real ? 'T' : 'C')
            rows, cols=side=='L' ? (m, 3) : (3, m)
            c=randn(T, rows, cols)
            padded=fill(T(NaN), rows+2, cols)
            padded[1:rows, :]=c
            result, info=direct_gemqrt!(side, tr, rows, cols, n, nb, a, t, padded, I)
            op=tr=='N' ? q : q'
            @test info==0
            @test result[1:rows, :] ≈ (side=='L' ? op*c : c*op)
            @test all(isnan, result[(rows + 1):end, :])
        end
        _, info=direct_geqrt!(copy(a), -1, n, nb, I; variant)
        @test info==-1
        _, info=direct_gemqrt!('?', 'N', m, m, n, nb, a, t, copy(q), I)
        @test info==-1
    end
end

@testset "Compact QR subnormal scale" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        R=real(T)
        scale=floatmin(R)/16
        normalized=T[2 1; 1 3; -1 2]
        a=normalized*scale
        f=qr(a)
        @test Matrix(f.Q)*(f.R/scale) ≈ normalized rtol=100eps(R)
        @test Matrix(f.Q)'*Matrix(f.Q) ≈ Matrix{T}(I, 2, 2)
    end
end
