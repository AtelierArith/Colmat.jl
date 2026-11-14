using Test, LinearAlgebra, Random, Colmat
@testset "Standard LQ Julia wrappers" begin
    Random.seed!(291)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        (m, n) in ((4, 7), (7, 4), (4, 4), (0, 3), (3, 0)),
        degenerate in (false, true)

        a=randn(T, m, n)
        if degenerate && m>0
            a[1, :].=0
        end
        factors, tau=LAPACK.gelqf!(copy(a))
        k=min(m, n)
        packed=zeros(T, n, n)
        packed[1:k, :]=factors[1:k, :]
        q=LAPACK.orglq!(packed, tau, k)
        l=zeros(T, m, n)
        l[:, 1:k]=tril(factors[:, 1:k])
        @test l*q ≈ a
        @test q*q' ≈ Matrix{T}(I, n, n)
        for side in ('L', 'R'), tr in ('N', T<:Real ? 'T' : 'C')
            c=randn(T, side=='L' ? n : 3, side=='L' ? 3 : n)
            result=LAPACK.ormlq!(side, tr, factors, tau, copy(c))
            op=tr=='N' ? q : q'
            @test result ≈ (side=='L' ? op*c : c*op)
        end
    end
end

using Libdl
for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    I in (Int32, Int64)

    suffix=I===Int64 ? "64_" : ""
    ft=Symbol(p*"gelqt_"*suffix)
    f3=Symbol(p*"gelqt3_"*suffix)
    ap=Symbol(p*"gemlqt_"*suffix)
    @eval function direct_lq_block!(
        a::Matrix{$T},
        m,
        n,
        mb,
        ::Type{$I};
        recursive=false,
        library=Colmat.backend_library(),
    )
        t=fill($T(NaN), mb+1, min(m, n))
        work=zeros($T, max(1, mb*m))
        info=Ref{$I}(0)
        f=Libdl.dlsym(
            Libdl.dlopen(library), recursive ? $(QuoteNode(f3)) : $(QuoteNode(ft))
        )
        if recursive
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
                    Ref{$I},
                    Ptr{$T},
                    Ref{$I},
                ),
                m,
                n,
                mb,
                a,
                size(a, 1),
                t,
                size(t, 1),
                work,
                info,
            )
        end
        t, info[]
    end
    @eval function direct_lq_apply!(
        side,
        tr,
        m,
        n,
        k,
        mb,
        v::Matrix{$T},
        t::Matrix{$T},
        c::Matrix{$T},
        ::Type{$I};
        library=Colmat.backend_library(),
    )
        work=zeros($T, max(1, mb*(side=='L' ? n : m)))
        info=Ref{$I}(0)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(ap)))
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
            mb,
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
@testset "Block LQ direct integer ABIs" begin
    external=BLAS.lbt_find_backing_library("dlaswlq_", :ilp64).libname
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        I in (Int32, Int64), mb in (1, 2, 4),
        recursive in (false, true)

        m, n=4, 7
        recursive && mb!=m && continue
        original=randn(T, m, n)
        a=fill(T(NaN), m+2, n)
        a[1:m, :]=original
        t, info=direct_lq_block!(a, m, n, mb, I; recursive)
        @test info==0
        @test all(isnan, a[(m + 1):end, :])
        @test all(isnan, t[mb + 1, :])
        q, info=direct_lq_apply!(
            'R', 'N', n, n, m, mb, a, t, Matrix{T}(LinearAlgebra.I, n, n), I
        )
        @test info==0
        @test tril(a[1:m, 1:m])*q[1:m, :] ≈ original
        @test q*q' ≈ Matrix{T}(LinearAlgebra.I, n, n)
        eq, info=direct_lq_apply!(
            'R',
            'N',
            n,
            n,
            m,
            mb,
            a,
            t,
            Matrix{T}(LinearAlgebra.I, n, n),
            Int64;
            library=external,
        )
        @test info==0 && eq ≈ q
        for side in ('L', 'R'), tr in ('N', T<:Real ? 'T' : 'C')
            rows, cols=side=='L' ? (n, 3) : (3, n)
            c=randn(T, rows, cols)
            result, info=direct_lq_apply!(side, tr, rows, cols, m, mb, a, t, copy(c), I)
            op=tr=='N' ? q : q'
            @test info==0
            @test result ≈ (side=='L' ? op*c : c*op)
        end
    end
end

for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    I in (Int32, Int64)

    suffix=I===Int64 ? "64_" : ""
    sw=Symbol(p*"laswlq_"*suffix)
    gelq=Symbol(p*"gelq_"*suffix)
    @eval function direct_ts_lq!(
        a::Matrix{$T},
        m,
        n,
        mb,
        nb,
        ::Type{$I};
        header=false,
        query=false,
        query_mode=-1,
        tsize=4096,
        library=Colmat.backend_library(),
    )
        t=fill($T(NaN), max(5, tsize))
        work=zeros($T, header ? max(1, n) : max(1, m*mb))
        info=Ref{$I}(0)
        f=Libdl.dlsym(Libdl.dlopen(library), header ? $(QuoteNode(gelq)) : $(QuoteNode(sw)))
        if header
            ccall(
                f,
                Cvoid,
                (
                    Ref{$I},
                    Ref{$I},
                    Ptr{$T},
                    Ref{$I},
                    Ptr{$T},
                    Ref{$I},
                    Ptr{$T},
                    Ref{$I},
                    Ref{$I},
                ),
                m,
                n,
                a,
                size(a, 1),
                t,
                query ? query_mode : tsize,
                work,
                query ? query_mode : length(work),
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
                    Ref{$I},
                    Ptr{$T},
                    Ref{$I},
                    Ptr{$T},
                    Ref{$I},
                    Ptr{$T},
                    Ref{$I},
                    Ref{$I},
                ),
                m,
                n,
                mb,
                nb,
                a,
                size(a, 1),
                t,
                mb,
                work,
                query ? -1 : length(work),
                info,
            )
        end
        t, work[1], info[]
    end
end

for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    I in (Int32, Int64)

    suffix=I===Int64 ? "64_" : ""
    sw=Symbol(p*"lamswlq_"*suffix)
    gemlq=Symbol(p*"gemlq_"*suffix)
    @eval function direct_ts_apply!(
        side,
        tr,
        m,
        n,
        k,
        mb,
        nb,
        a::Matrix{$T},
        t::Vector{$T},
        c::Matrix{$T},
        ::Type{$I};
        header=false,
        query=false,
        library=Colmat.backend_library(),
    )
        actual_mb=header ? Int(real(t[2])) : mb
        work=zeros($T, max(1, actual_mb*(side=='L' ? n : m)))
        info=Ref{$I}(0)
        f=Libdl.dlsym(
            Libdl.dlopen(library), header ? $(QuoteNode(gemlq)) : $(QuoteNode(sw))
        )
        if header
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
                    Ref{$I},
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
                t,
                length(t),
                c,
                size(c, 1),
                work,
                query ? -1 : length(work),
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
                    Ref{$I},
                ),
                side,
                tr,
                m,
                n,
                k,
                mb,
                nb,
                a,
                size(a, 1),
                t,
                mb,
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
@testset "Short-wide and header LQ ABI interoperability" begin
    external=BLAS.lbt_find_backing_library("dgeqlf_", :ilp64).libname
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        I in (Int32, Int64), header in (false, true),
        mb in (1, 2)

        m, n, nb=2, 9, 5
        original=randn(T, m, n)
        a=copy(original)
        _, _, info=direct_ts_lq!(a, m, n, mb, nb, I; header, query=true)
        @test info==0 && a==original
        t, _, info=direct_ts_lq!(a, m, n, mb, nb, I; header)
        @test info==0
        q, _, info=direct_ts_apply!(
            'R', 'N', n, n, m, mb, nb, a, t, Matrix{T}(LinearAlgebra.I, n, n), I; header
        )
        @test info==0
        @test tril(a[:, 1:m])*q[1:m, :] ≈ original
        @test q*q' ≈ Matrix{T}(LinearAlgebra.I, n, n)
        eq, _, info=direct_ts_apply!(
            'R',
            'N',
            n,
            n,
            m,
            mb,
            nb,
            a,
            t,
            Matrix{T}(LinearAlgebra.I, n, n),
            Int64;
            header,
            library=external,
        )
        @test info==0 && eq ≈ q
        ea=copy(original)
        et, _, info=direct_ts_lq!(ea, m, n, mb, nb, Int64; header, library=external)
        @test info==0
        eq, _, info=direct_ts_apply!(
            'R', 'N', n, n, m, mb, nb, ea, et, Matrix{T}(LinearAlgebra.I, n, n), I; header
        )
        @test info==0
        @test tril(ea[:, 1:m])*eq[1:m, :] ≈ original
        for side in ('L', 'R'), tr in ('N', T<:Real ? 'T' : 'C')
            rows, cols=side=='L' ? (n, 3) : (3, n)
            c=randn(T, rows, cols)
            result, _, info=direct_ts_apply!(
                side, tr, rows, cols, m, mb, nb, a, t, copy(c), I; header
            )
            op=tr=='N' ? q : q'
            @test info==0
            @test result ≈ (side=='L' ? op*c : c*op)
        end
    end
end

for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    I in (Int32, Int64)

    suffix=I===Int64 ? "64_" : ""
    fb=Symbol(p*"gelqf_"*suffix)
    fu=Symbol(p*"gelq2_"*suffix)
    @eval function direct_standard_lq!(
        a::Matrix{$T},
        m,
        n,
        ::Type{$I};
        unblocked=false,
        query=false,
        library=Colmat.backend_library(),
    )
        tau=zeros($T, min(m, n))
        work=zeros($T, max(1, m))
        info=Ref{$I}(0)
        f=Libdl.dlsym(
            Libdl.dlopen(library), unblocked ? $(QuoteNode(fu)) : $(QuoteNode(fb))
        )
        if unblocked
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
        else
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
        end
        tau, work[1], info[]
    end
end
@testset "Standard LQ direct factorization ABIs" begin
    external=BLAS.lbt_find_backing_library("dgeqlf_", :ilp64).libname
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        I in (Int32, Int64), unblocked in (false, true),
        (m, n) in ((4, 7), (7, 4), (4, 4))

        original=randn(T, m, n)
        k=min(m, n)
        a=fill(T(NaN), m+2, n)
        a[1:m, :]=original
        if !unblocked
            _, work, info=direct_standard_lq!(a, m, n, I; query=true)
            @test info==0 && real(work)>=m && a[1:m, :]==original
        end
        tau, _, info=direct_standard_lq!(a, m, n, I; unblocked)
        @test info==0
        q=LAPACK.orglq!(copy(a[1:k, :]), tau, k)
        @test tril(a[1:m, 1:k])*q ≈ original
        @test all(isnan, a[(m + 1):end, :])
        ea=copy(original)
        etau, _, info=direct_standard_lq!(ea, m, n, Int64; unblocked, library=external)
        @test info==0
        eq=LAPACK.orglq!(copy(ea[1:k, :]), etau, k)
        @test tril(ea[:, 1:k])*eq ≈ original
    end
end

for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    I in (Int32, Int64)

    suffix=I===Int64 ? "64_" : ""
    qnames=if T<:Real
        ("orglq", "orgl2", "ormlq", "orml2")
    else
        ("unglq", "ungl2", "unmlq", "unml2")
    end
    syms=Tuple(Symbol(p*x*"_"*suffix) for x in qnames)
    @eval function direct_generate_lq!(
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
        work=zeros($T, max(1, m))
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
    @eval function direct_apply_lq!(
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

@testset "LQ Q generation and application direct ABIs" begin
    external=BLAS.lbt_find_backing_library("dgeqlf_", :ilp64).libname
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        I in (Int32, Int64),
        unblocked in (false, true)

        m, n=3, 7
        original=randn(T, m, n)
        a=copy(original)
        tau, _, _=direct_standard_lq!(a, m, n, I)
        packed=fill(T(NaN), n+2, n)
        packed[1:m, :]=a
        q, _, info=direct_generate_lq!(packed, n, n, tau, I; unblocked)
        @test info==0 && all(isnan, q[(n + 1):end, :])
        q=q[1:n, :]
        @test tril(a[:, 1:m])*q[1:m, :] ≈ original
        ep=zeros(T, n, n)
        ep[1:m, :]=a
        eq, _, info=direct_generate_lq!(ep, n, n, tau, Int64; unblocked, library=external)
        @test info==0 && eq ≈ q
        for side in ('L', 'R'), tr in ('N', T<:Real ? 'T' : 'C')
            rows, cols=side=='L' ? (n, 4) : (4, n)
            c=randn(T, rows, cols)
            actual, _, info=direct_apply_lq!(
                side, tr, a, tau, copy(c), rows, cols, I; unblocked
            )
            op=tr=='N' ? q : q'
            @test info==0 && actual ≈ (side=='L' ? op*c : c*op)
            reference, _, info=direct_apply_lq!(
                side, tr, a, tau, copy(c), rows, cols, Int64; unblocked, library=external
            )
            @test info==0 && actual ≈ reference
        end
    end
end

for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    I in (Int32, Int64)

    sym=Symbol(p*"gelqs_"*(I===Int64 ? "64_" : ""))
    @eval function direct_gelqs!(
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
@testset "GELQS minimum norm direct ABIs" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), I in (Int32, Int64)
        m, n=3, 7
        a=randn(T, m, n)
        rhs=randn(T, m, 2)
        factor=copy(a)
        tau, _, _=direct_standard_lq!(factor, m, n, I)
        b=fill(T(NaN), n, 2)
        b[1:m, :]=rhs
        result, info=direct_gelqs!(factor, tau, b, I)
        @test info==0
        @test a*result ≈ rhs
        @test result ≈ a'*((a*a')\rhs)
        @test direct_gelqs!(factor, tau, copy(b), I; lwork=-1)[2]==-10
    end
end

@testset "LQ minimal header query and registration" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), I in (Int32, Int64)
        a=randn(T, 2, 9)
        saved=copy(a)
        t, w, info=direct_ts_lq!(
            a, 2, 9, 1, 9, I; header=true, query=true, query_mode=-2, tsize=7
        )
        @test info==0 && real(t[1])==7 && real(w)==9
        @test a==saved
        t, _, info=direct_ts_lq!(a, 2, 9, 1, 9, I; header=true, tsize=7)
        @test info==0
        q, _, info=direct_ts_apply!(
            'R', 'N', 9, 9, 2, 1, 9, a, t, Matrix{T}(LinearAlgebra.I, 9, 9), I; header=true
        )
        @test info==0 && tril(a[:, 1:2])*q[1:2, :] ≈ saved
    end
    for p in ("s", "d", "c", "z"), abi in (:lp64, :ilp64)
        qnames=if p in ("s", "d")
            ("orgl2", "orglq", "orml2", "ormlq")
        else
            ("ungl2", "unglq", "unml2", "unmlq")
        end
        for family in (
            "gelq",
            "gelq2",
            "gelqf",
            "gelqs",
            "gelqt",
            "gelqt3",
            "gemlq",
            "gemlqt",
            "lamswlq",
            "laswlq",
            qnames...,
        )
            @test realpath(BLAS.lbt_find_backing_library(p*family*"_", abi).libname)==realpath(
                Colmat.backend_library()
            )
        end
    end
end
