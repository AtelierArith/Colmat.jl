using Test, LinearAlgebra, Random, Colmat
@testset "Standard QR Julia LAPACK wrappers" begin
    Random.seed!(289)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        (m, n) in ((7, 4), (4, 7), (4, 4), (0, 3), (3, 0)),
        degenerate in (false, true)

        a=randn(T, m, n)
        if degenerate && n>0
            a[:, 1].=0
        end
        factors, tau=LAPACK.geqrf!(copy(a))
        k=min(m, n)
        packed=zeros(T, m, m)
        packed[:, 1:k]=factors[:, 1:k]
        q=LAPACK.orgqr!(packed, tau, k)
        r=zeros(T, m, n)
        r[1:k, :]=triu(factors[1:k, :])
        @test q*r ≈ a
        @test q'*q ≈ Matrix{T}(I, m, m)
        for side in ('L', 'R'), tr in ('N', T<:Real ? 'T' : 'C')
            c=randn(T, side=='L' ? m : 3, side=='L' ? 3 : m)
            actual=LAPACK.ormqr!(side, tr, factors, tau, copy(c))
            op=tr=='N' ? q : q'
            @test actual ≈ (side=='L' ? op*c : c*op)
        end
    end
end

using Libdl
for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    I in (Int32, Int64)

    suffix=I===Int64 ? "64_" : ""
    names=Tuple(
        Symbol(p*name*"_"*suffix) for name in ("geqrf", "geqrfp", "geqr2", "geqr2p")
    )
    @eval function direct_standard_qr!(
        a::Matrix{$T},
        m,
        n,
        ::Type{$I};
        variant=1,
        query=false,
        library=Colmat.backend_library(),
    )
        tau=zeros($T, min(max(m, 0), max(n, 0)))
        work=zeros($T, max(1, n))
        info=Ref{$I}(0)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(names))[variant])
        if variant<=2
            ccall(
                f,
                Cvoid,
                (Ref{$I}, Ref{$I}, Ptr{$T}, Ref{$I}, Ptr{$T}, Ptr{$T}, Ref{$I}, Ref{$I}),
                m,
                n,
                a,
                size(a, 1),
                tau,
                work,
                query ? -1 : length(work),
                info,
            )
        else
            ccall(
                f,
                Cvoid,
                (Ref{$I}, Ref{$I}, Ptr{$T}, Ref{$I}, Ptr{$T}, Ptr{$T}, Ref{$I}),
                m,
                n,
                a,
                size(a, 1),
                tau,
                work,
                info,
            )
        end
        tau, work[1], info[]
    end
end
@testset "Standard QR factorization direct ABIs" begin
    external=BLAS.lbt_find_backing_library("dgelqf_", :ilp64).libname
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        I in (Int32, Int64), variant in 1:4,
        (m, n) in ((7, 4), (4, 7), (4, 4))

        original=randn(T, m, n)
        k=min(m, n)
        a=fill(T(NaN), m+2, n)
        a[1:m, :]=original
        if variant<=2
            saved=copy(a)
            _, work, info=direct_standard_qr!(a, m, n, I; variant, query=true)
            @test info==0
            @test real(work)>=max(1, n)
            @test isequal(a, saved)
        end
        tau, _, info=direct_standard_qr!(a, m, n, I; variant)
        @test info==0
        @test all(isnan, a[(m + 1):end, :])
        qinput=zeros(T, m, m)
        qinput[:, 1:k]=a[1:m, 1:k]
        q=LAPACK.orgqr!(qinput, tau, k)
        r=zeros(T, m, n)
        r[1:k, :]=triu(a[1:k, :])
        @test q*r ≈ original
        @test q'*q ≈ Matrix{T}(LinearAlgebra.I, m, m)
        if iseven(variant)
            @test all(real(a[j, j])>=0 && imag(a[j, j])==0 for j in 1:k)
        end
        # OpenBLAS and native factors need not choose the same diagonal phase.
        ea=copy(original)
        etau, _, info=direct_standard_qr!(ea, m, n, Int64; variant, library=external)
        @test info==0
        eqinput=zeros(T, m, m)
        eqinput[:, 1:k]=ea[:, 1:k]
        eq=LAPACK.orgqr!(eqinput, etau, k)
        er=zeros(T, m, n)
        er[1:k, :]=triu(ea[1:k, :])
        @test eq*er ≈ original
    end
end

for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    I in (Int32, Int64)

    sym=Symbol(p*"geqrs_"*(I===Int64 ? "64_" : ""))
    @eval function direct_geqrs!(
        a::Matrix{$T}, tau::Vector{$T}, b::Matrix{$T}, ::Type{$I}; lwork=max(1, size(b, 2))
    )
        m, n=size(a)
        nrhs=size(b, 2)
        work=zeros($T, max(1, lwork))
        info=Ref{$I}(0)
        f=Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), $(QuoteNode(sym)))
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
                Ptr{$T},
                Ref{$I},
                Ptr{$T},
                Ref{$I},
                Ref{$I},
            ),
            m,
            n,
            nrhs,
            a,
            m,
            tau,
            b,
            size(b, 1),
            work,
            lwork,
            info,
        )
        b, info[]
    end
end
@testset "GEQRS direct least squares" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), I in (Int32, Int64)
        a=randn(T, 7, 4)
        b=randn(T, 7, 2)
        factor=copy(a)
        tau, _, info=direct_standard_qr!(factor, 7, 4, I)
        @test info==0
        result, info=direct_geqrs!(factor, tau, copy(b), I)
        @test info==0
        @test a'*(a*result[1:4, :]-b) ≈ zeros(T, 4, 2) atol=100eps(real(T))*norm(a)*norm(b)
        _, info=direct_geqrs!(factor, tau, copy(b), I; lwork=1)
        @test info==-10
        # GEQRS is a deprecated driver without workspace-query support.
        _, info=direct_geqrs!(factor, tau, copy(b), I; lwork=-1)
        @test info==-10
    end
end

for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    I in (Int32, Int64)

    suffix=I===Int64 ? "64_" : ""
    qnames=if T<:Real
        ("orgqr", "org2r", "ormqr", "orm2r")
    else
        ("ungqr", "ung2r", "unmqr", "unm2r")
    end
    syms=Tuple(Symbol(p*x*"_"*suffix) for x in qnames)
    @eval function direct_generate_q!(
        a::Matrix{$T},
        m,
        n,
        tau::Vector{$T},
        ::Type{$I};
        unblocked=false,
        query=false,
        library=Colmat.backend_library(),
    )
        k=length(tau)
        work=zeros($T, max(1, n))
        info=Ref{$I}(0)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(syms))[unblocked ? 2 : 1])
        if unblocked
            ccall(
                f,
                Cvoid,
                (Ref{$I}, Ref{$I}, Ref{$I}, Ptr{$T}, Ref{$I}, Ptr{$T}, Ptr{$T}, Ref{$I}),
                m,
                n,
                k,
                a,
                size(a, 1),
                tau,
                work,
                info,
            )
        else
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
                    Ptr{$T},
                    Ref{$I},
                    Ref{$I},
                ),
                m,
                n,
                k,
                a,
                size(a, 1),
                tau,
                work,
                query ? -1 : length(work),
                info,
            )
        end
        a, work[1], info[]
    end
    @eval function direct_apply_q!(
        side,
        tr,
        a::Matrix{$T},
        tau::Vector{$T},
        c::Matrix{$T},
        m,
        n,
        ::Type{$I};
        unblocked=false,
        query=false,
        library=Colmat.backend_library(),
    )
        k=length(tau)
        work=zeros($T, max(1, side=='L' ? n : m))
        info=Ref{$I}(0)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(syms))[unblocked ? 4 : 3])
        if unblocked
            ccall(
                f,
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{UInt8},
                    Ref{$I},
                    Ref{$I},
                    Ref{$I},
                    Ptr{$T},
                    Ref{$I},
                    Ptr{$T},
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
                a,
                size(a, 1),
                tau,
                c,
                size(c, 1),
                work,
                info,
            )
        else
            ccall(
                f,
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{UInt8},
                    Ref{$I},
                    Ref{$I},
                    Ref{$I},
                    Ptr{$T},
                    Ref{$I},
                    Ptr{$T},
                    Ptr{$T},
                    Ref{$I},
                    Ptr{$T},
                    Ref{$I},
                    Ref{$I},
                ),
                side,
                tr,
                m,
                n,
                k,
                a,
                size(a, 1),
                tau,
                c,
                size(c, 1),
                work,
                query ? -1 : length(work),
                info,
            )
        end
        c, work[1], info[]
    end
end
@testset "Standard Q direct ABIs and external application" begin
    external=BLAS.lbt_find_backing_library("dgelqf_", :ilp64).libname
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        I in (Int32, Int64),
        unblocked in (false, true)

        m, n=7, 4
        original=randn(T, m, n)
        a=copy(original)
        tau, _, _=direct_standard_qr!(a, m, n, I)
        packed=fill(T(NaN), m+2, m)
        packed[1:m, 1:n]=a
        if !unblocked
            _, work, info=direct_generate_q!(packed, m, m, tau, I; query=true)
            @test info==0 && real(work)>=m
            @test isequal(packed[1:m, 1:n], a)
        end
        q, _, info=direct_generate_q!(packed, m, m, tau, I; unblocked)
        @test info==0
        @test all(isnan, q[(m + 1):end, :])
        q=q[1:m, :]
        @test q[:, 1:n]*triu(a[1:n, :]) ≈ original
        ep=zeros(T, m, m)
        ep[:, 1:n]=a
        eq, _, info=direct_generate_q!(ep, m, m, tau, Int64; unblocked, library=external)
        @test info==0 && eq ≈ q
        for side in ('L', 'R'), tr in ('N', T<:Real ? 'T' : 'C')
            rows, cols=side=='L' ? (m, 3) : (3, m)
            c=randn(T, rows, cols)
            padded=fill(T(NaN), rows+2, cols)
            padded[1:rows, :]=c
            if !unblocked
                _, work, info=direct_apply_q!(
                    side, tr, a, tau, padded, rows, cols, I; query=true
                )
                @test info==0 && real(work)>=min(rows, cols)
                @test isequal(padded[1:rows, :], c)
            end
            result, _, info=direct_apply_q!(
                side, tr, a, tau, padded, rows, cols, I; unblocked
            )
            op=tr=='N' ? q : q'
            @test info==0
            @test result[1:rows, :] ≈ (side=='L' ? op*c : c*op)
            @test all(isnan, result[(rows + 1):end, :])
            er, _, info=direct_apply_q!(
                side, tr, a, tau, copy(c), rows, cols, Int64; unblocked, library=external
            )
            @test info==0 && er ≈ result[1:rows, :]
        end
    end
end

@testset "Standard QR LBT registration" begin
    for p in ("s", "d", "c", "z"), abi in (:lp64, :ilp64)
        families=(
            "geqrf",
            "geqrfp",
            "geqr2",
            "geqr2p",
            "geqrs",
            p in ("s", "d") ? "orgqr" : "ungqr",
            p in ("s", "d") ? "org2r" : "ung2r",
            p in ("s", "d") ? "ormqr" : "unmqr",
            p in ("s", "d") ? "orm2r" : "unm2r",
        )
        for family in families
            @test realpath(BLAS.lbt_find_backing_library(p*family*"_", abi).libname)==realpath(
                Colmat.backend_library()
            )
        end
    end
end
