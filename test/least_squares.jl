using Test, LinearAlgebra, Random, Libdl, Colmat
for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    I in (Int32, Int64)

    sym=Symbol(p*"gels_"*(I===Int64 ? "64_" : ""))
    @eval function direct_gels!(
        tr,
        a::Matrix{$T},
        b::Matrix{$T},
        m,
        n,
        nrhs,
        ::Type{$I};
        query=false,
        library=Colmat.backend_library(),
    )
        work=zeros($T, max(1, min(m, n)+max(min(m, n), nrhs)))
        info=Ref{$I}(0)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(sym)))
        ccall(
            f,
            Cvoid,
            (
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
                Ref{$I},
            ),
            tr,
            m,
            n,
            nrhs,
            a,
            size(a, 1),
            b,
            size(b, 1),
            work,
            query ? -1 : length(work),
            info,
        )
        b, work[1], info[]
    end
end
@testset "GELS full-rank direct integer ABIs" begin
    Random.seed!(310)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        I in (Int32, Int64), (m, n) in ((7, 4), (4, 7), (4, 4)),
        tr in ('N', T<:Real ? 'T' : 'C')

        original=randn(T, m, n)
        op=tr=='N' ? original : original'
        rows, cols=size(op)
        rhs=randn(T, rows, 2)
        expected=rows>=cols ? (op'*op)\(op'*rhs) : op'*((op*op')\rhs)
        a=fill(T(NaN), m+2, n)
        a[1:m, :]=original
        b=fill(T(NaN), max(m, n)+2, 2)
        b[1:rows, :]=rhs
        saved_a=copy(a)
        saved_b=copy(b)
        _, w, info=direct_gels!(tr, a, b, m, n, 2, I; query=true)
        @test info==0 && real(w)>=min(m, n)+max(min(m, n), 2)
        @test isequal(a, saved_a)&&isequal(b, saved_b)
        result, _, info=direct_gels!(tr, a, b, m, n, 2, I)
        @test info==0
        @test result[1:cols, :] ≈ expected
        @test all(isnan, a[(m + 1):end, :]) && all(isnan, b[(max(m, n) + 1):end, :])
        if rows>cols
            @test norm(result[(cols + 1):rows, :]) ≈ norm(op*result[1:cols, :]-rhs)
        end
    end
end
@testset "GELS extreme shared scales" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        I in (Int32, Int64),
        (m, n) in ((5, 3), (3, 5))

        R=real(T)
        original=randn(T, m, n)
        rhs=randn(T, m, 2)
        expected=if m>=n
            (original'*original)\(original'*rhs)
        else
            original'*((original*original')\rhs)
        end
        for scale in (floatmin(R)/16, floatmax(R)/64)
            a=original*scale
            b=fill(T(NaN), max(m, n), 2)
            b[1:m, :]=rhs*scale
            result, _, info=direct_gels!('N', a, b, m, n, 2, I)
            @test info==0
            @test result[1:n, :] ≈ expected rtol=200eps(R)
        end
    end
end

for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    I in (Int32, Int64)

    R=real(T)
    sym=Symbol(p*"gelss_"*(I===Int64 ? "64_" : ""))
    @eval function direct_gelss!(
        a::Matrix{$T},
        b::Matrix{$T},
        m,
        n,
        nrhs,
        ::Type{$I};
        rcond=-1,
        query=false,
        library=Colmat.backend_library(),
    )
        k=min(m, n)
        s=zeros($R, k)
        rank=Ref{$I}(-1)
        info=Ref{$I}(0)
        minimum=k==0 ? 1 : ($T <: Real ? 3k+max(2k, m, n, nrhs) : 2k+max(m, n, nrhs))
        work=zeros($T, minimum)
        rwork=zeros($R, max(1, 5k))
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(sym)))
        if $T <: Real
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
                    Ptr{$R},
                    Ref{$R},
                    Ref{$I},
                    Ptr{$T},
                    Ref{$I},
                    Ref{$I},
                ),
                m,
                n,
                nrhs,
                a,
                size(a, 1),
                b,
                size(b, 1),
                s,
                rcond,
                rank,
                work,
                query ? -1 : minimum,
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
                    Ptr{$R},
                    Ref{$R},
                    Ref{$I},
                    Ptr{$T},
                    Ref{$I},
                    Ptr{$R},
                    Ref{$I},
                ),
                m,
                n,
                nrhs,
                a,
                size(a, 1),
                b,
                size(b, 1),
                s,
                rcond,
                rank,
                work,
                query ? -1 : minimum,
                rwork,
                info,
            )
        end
        (; rank=Int(rank[]), s, info=Int(info[]), work=work[1])
    end
end
@testset "GELSS rank and minimum norm direct ABIs" begin
    external=BLAS.lbt_find_backing_library("dgelsd_", :ilp64).libname
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        I in (Int32, Int64), (m, n) in ((7, 4), (4, 7), (4, 4)), rcond in (-1.0, 1e-3),
        deficient in (false, true)

        original=randn(T, m, n)
        deficient && (original[:, end]=original[:, 1])
        rhs=randn(T, m, 2)
        a=fill(T(NaN), m+2, n)
        a[1:m, :]=original
        b=fill(T(NaN), max(m, n)+2, 2)
        b[1:m, :]=rhs
        saved_a=copy(a)
        saved_b=copy(b)
        query=direct_gelss!(a, b, m, n, 2, I; query=true)
        @test query.info==0 && isequal(a, saved_a) && isequal(b, saved_b)
        result=direct_gelss!(a, b, m, n, 2, I; rcond)
        ea=copy(original)
        eb=zeros(T, max(m, n), 2)
        eb[1:m, :]=rhs
        reference=direct_gelss!(ea, eb, m, n, 2, Int64; rcond, library=external)
        @test result.info==0 && reference.info==0
        @test result.rank==reference.rank
        @test result.s ≈ reference.s
        @test b[1:n, :] ≈ eb[1:n, :]
        @test norm(original'*(original*b[1:n, :]-rhs)) <=
            500eps(real(T))*norm(original)^2*max(norm(b[1:n, :]), norm(rhs))
        vh=a[1:min(m, n), :]
        @test vh*vh' ≈ Matrix{T}(LinearAlgebra.I, min(m, n), min(m, n))
        @test all(isnan, a[(m + 1):end, :]) && all(isnan, b[(max(m, n) + 1):end, :])
    end
end

for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    I in (Int32, Int64)

    R=real(T)
    sym=Symbol(p*"gelsd_"*(I===Int64 ? "64_" : ""))
    @eval function direct_gelsd!(
        a::Matrix{$T},
        b::Matrix{$T},
        m,
        n,
        nrhs,
        ::Type{$I};
        rcond=-1,
        query=false,
        library=Colmat.backend_library(),
    )
        s=zeros($R, min(m, n))
        rank=Ref{$I}(-1)
        info=Ref{$I}(0)
        work=zeros($T, 1)
        rwork=zeros($R, 1)
        iwork=zeros($I, 1)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(sym)))
        function invoke!(lw)
            if $T <: Real
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
                        Ptr{$R},
                        Ref{$R},
                        Ref{$I},
                        Ptr{$T},
                        Ref{$I},
                        Ptr{$I},
                        Ref{$I},
                    ),
                    m,
                    n,
                    nrhs,
                    a,
                    size(a, 1),
                    b,
                    size(b, 1),
                    s,
                    rcond,
                    rank,
                    work,
                    lw,
                    iwork,
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
                        Ptr{$R},
                        Ref{$R},
                        Ref{$I},
                        Ptr{$T},
                        Ref{$I},
                        Ptr{$R},
                        Ptr{$I},
                        Ref{$I},
                    ),
                    m,
                    n,
                    nrhs,
                    a,
                    size(a, 1),
                    b,
                    size(b, 1),
                    s,
                    rcond,
                    rank,
                    work,
                    lw,
                    rwork,
                    iwork,
                    info,
                )
            end
        end
        invoke!(-1)
        sizes=(
            max(1, ceil(Int, real(work[1]))),
            max(1, Int(iwork[1])),
            max(1, ceil(Int, rwork[1])),
        )
        if !query && info[]==0
            resize!(work, sizes[1])
            resize!(iwork, sizes[2])
            resize!(rwork, sizes[3])
            invoke!(length(work))
        end
        (; rank=Int(rank[]), s, info=Int(info[]), sizes)
    end
end
@testset "GELSD queried workspaces and solutions" begin
    Random.seed!(31004)
    external=BLAS.lbt_find_backing_library("dgelsy_", :ilp64).libname
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        I in (Int32, Int64), (m, n) in ((7, 4), (4, 7), (4, 4)),
        deficient in (false, true)

        original=randn(T, m, n)
        deficient && (original[:, end]=original[:, 1])
        rhs=randn(T, m, 2)
        rcond=deficient ? 1e-3 : -1.0
        a=fill(T(NaN), m+2, n)
        a[1:m, :]=original
        b=fill(T(NaN), max(m, n)+2, 2)
        b[1:m, :]=rhs
        beforea=copy(a)
        beforeb=copy(b)
        query=direct_gelsd!(a, b, m, n, 2, I; query=true, rcond)
        @test query.info==0 && all(>(0), query.sizes)
        @test isequal(a, beforea)&&isequal(b, beforeb)
        result=direct_gelsd!(a, b, m, n, 2, I; rcond)
        ea=copy(original)
        eb=zeros(T, max(m, n), 2)
        eb[1:m, :]=rhs
        reference=direct_gelsd!(ea, eb, m, n, 2, Int64; rcond, library=external)
        @test result.info==0 && reference.info==0
        @test result.rank==reference.rank
        @test result.s ≈ reference.s
        @test b[1:n, :] ≈ eb[1:n, :]
        @test all(isnan, a[(m + 1):end, :]) && all(isnan, b[(max(m, n) + 1):end, :])
    end
end

for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    I in (Int32, Int64)

    R=real(T)
    suffix=I===Int64 ? "64_" : ""
    fy=Symbol(p*"gelsy_"*suffix)
    fx=Symbol(p*"gelsx_"*suffix)
    @eval function direct_gelsy!(
        a::Matrix{$T},
        b::Matrix{$T},
        m,
        n,
        nrhs,
        piv::Vector{$I};
        rcond=1e-3,
        legacy=false,
        query=false,
        library=Colmat.backend_library(),
    )
        k=min(m, n)
        minimum=if $T <: Real
            ((k==0||nrhs==0) ? 1 : max(k+3n+1, 2k+nrhs))
        else
            k+max(2k, n+1, k+nrhs)
        end
        work=zeros($T, max(minimum, k+3n, k+nrhs, 1))
        rwork=zeros($R, max(1, 2n))
        rank=Ref{$I}(-1)
        info=Ref{$I}(0)
        f=Libdl.dlsym(Libdl.dlopen(library), legacy ? $(QuoteNode(fx)) : $(QuoteNode(fy)))
        if $T <: Real
            if legacy
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
                        Ptr{$I},
                        Ref{$R},
                        Ref{$I},
                        Ptr{$T},
                        Ref{$I},
                    ),
                    m,
                    n,
                    nrhs,
                    a,
                    size(a, 1),
                    b,
                    size(b, 1),
                    piv,
                    rcond,
                    rank,
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
                        Ref{$I},
                        Ptr{$I},
                        Ref{$R},
                        Ref{$I},
                        Ptr{$T},
                        Ref{$I},
                        Ref{$I},
                    ),
                    m,
                    n,
                    nrhs,
                    a,
                    size(a, 1),
                    b,
                    size(b, 1),
                    piv,
                    rcond,
                    rank,
                    work,
                    query ? -1 : minimum,
                    info,
                )
            end
        else
            if legacy
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
                        Ptr{$I},
                        Ref{$R},
                        Ref{$I},
                        Ptr{$T},
                        Ptr{$R},
                        Ref{$I},
                    ),
                    m,
                    n,
                    nrhs,
                    a,
                    size(a, 1),
                    b,
                    size(b, 1),
                    piv,
                    rcond,
                    rank,
                    work,
                    rwork,
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
                        Ptr{$I},
                        Ref{$R},
                        Ref{$I},
                        Ptr{$T},
                        Ref{$I},
                        Ptr{$R},
                        Ref{$I},
                    ),
                    m,
                    n,
                    nrhs,
                    a,
                    size(a, 1),
                    b,
                    size(b, 1),
                    piv,
                    rcond,
                    rank,
                    work,
                    query ? -1 : minimum,
                    rwork,
                    info,
                )
            end
        end
        (; rank=Int(rank[]), piv=Int.(piv), info=Int(info[]))
    end
end
@testset "GELSY GELSX complete orthogonal direct ABIs" begin
    Random.seed!(31005)
    external=BLAS.lbt_find_backing_library("dgeqlf_", :ilp64).libname
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        I in (Int32, Int64),
        legacy in (false, true),
        (m, n) in ((7, 4), (4, 7), (4, 4)),
        deficient in (false, true),
        fixed in (false, true)

        original=randn(T, m, n)
        deficient && (original[:, end]=original[:, 1])
        rhs=randn(T, m, 2)
        flags=zeros(I, n)
        fixed && (flags[2]=3)
        a=fill(T(NaN), m+2, n)
        a[1:m, :]=original
        b=fill(T(NaN), max(m, n)+2, 2)
        b[1:m, :]=rhs
        if !legacy
            saved_a=copy(a)
            saved_b=copy(b)
            saved_p=copy(flags)
            result=direct_gelsy!(a, b, m, n, 2, flags; query=true)
            @test result.info==0 && isequal(a, saved_a)&&isequal(b, saved_b)&&flags==saved_p
        end
        result=direct_gelsy!(a, b, m, n, 2, flags; legacy)
        ea=copy(original)
        eb=zeros(T, max(m, n), 2)
        eb[1:m, :]=rhs
        ep=zeros(Int64, n)
        fixed && (ep[2]=3)
        reference=direct_gelsy!(ea, eb, m, n, 2, ep; legacy, library=external)
        @test result.info==0 && reference.info==0
        @test result.rank==reference.rank
        @test sort(result.piv)==collect(1:n)
        @test !fixed || result.piv[1]==2
        @test b[1:n, :] ≈ eb[1:n, :]
        @test all(isnan, a[(m + 1):end, :])&&all(isnan, b[(max(m, n) + 1):end, :])
    end
end

for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    I in (Int32, Int64)

    suffix=I===Int64 ? "64_" : ""
    st=Symbol(p*"gelst_"*suffix)
    ts=Symbol(p*"getsls_"*suffix)
    @eval function direct_full_driver!(
        tr,
        a::Matrix{$T},
        b::Matrix{$T},
        m,
        n,
        nrhs,
        ::Type{$I};
        tall_short=false,
        query=false,
        query_mode=-1,
    )
        k=min(m, n)
        minimum=if tall_short
            ((k==0||nrhs==0) ? 1 : k+5+max(k, nrhs))
        else
            max(1, k+max(k, nrhs))
        end
        work=zeros($T, minimum)
        info=Ref{$I}(0)
        f=Libdl.dlsym(
            Libdl.dlopen(Colmat.backend_library()),
            tall_short ? $(QuoteNode(ts)) : $(QuoteNode(st)),
        )
        ccall(
            f,
            Cvoid,
            (
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
                Ref{$I},
            ),
            tr,
            m,
            n,
            nrhs,
            a,
            size(a, 1),
            b,
            size(b, 1),
            work,
            query ? query_mode : minimum,
            info,
        )
        info[], work[1]
    end
end
@testset "GELST GETSLS full-rank direct ABIs" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        I in (Int32, Int64), tall_short in (false, true), (m, n) in ((7, 4), (4, 7)),
        tr in ('N', T<:Real ? 'T' : 'C')

        a=randn(T, m, n)
        op=tr=='N' ? a : a'
        rows, cols=size(op)
        rhs=randn(T, rows, 2)
        expected=rows>=cols ? (op'*op)\(op'*rhs) : op'*((op*op')\rhs)
        b=fill(T(NaN), max(m, n), 2)
        b[1:rows, :]=rhs
        beforea=copy(a)
        beforeb=copy(b)
        info, work=direct_full_driver!(
            tr, a, b, m, n, 2, I; tall_short, query=true, query_mode=tall_short ? -2 : -1
        )
        @test info==0 && real(work)>=1 && a==beforea && isequal(b, beforeb)
        info, _=direct_full_driver!(tr, a, b, m, n, 2, I; tall_short)
        @test info==0 && b[1:cols, :] ≈ expected
    end
end

@testset "Least-squares zero and empty boundaries" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        I in (Int32, Int64),
        (m, n) in ((0, 3), (3, 0), (0, 0), (3, 3))

        a=zeros(T, max(1, m), n)
        b=ones(T, max(1, max(m, n)), 2)
        _, _, info=direct_gels!('N', copy(a), copy(b), m, n, 2, I)
        @test info==0
        rhs=copy(b)
        result=direct_gelss!(copy(a), rhs, m, n, 2, I)
        @test result.info==0 && result.rank==0 && all(iszero, result.s)
        @test all(iszero, rhs[1:n, :])
        rhs=copy(b)
        result=direct_gelsd!(copy(a), rhs, m, n, 2, I)
        @test result.info==0 && result.rank==0 && all(iszero, result.s)
        @test all(iszero, rhs[1:n, :])
        rhs=copy(b)
        result=direct_gelsy!(copy(a), rhs, m, n, 2, zeros(I, n))
        @test result.info==0 && result.rank==0
        if m>0&&n>0
            @test all(iszero, rhs)
        end
    end
    for p in ("s", "d", "c", "z"),
        family in ("gels", "gelsd", "gelss", "gelst", "gelsx", "gelsy", "getsls"),
        abi in (:lp64, :ilp64)

        @test realpath(BLAS.lbt_find_backing_library(p*family*"_", abi).libname)==realpath(
            Colmat.backend_library()
        )
    end
end
