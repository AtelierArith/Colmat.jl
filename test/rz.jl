using Test, LinearAlgebra, Random, Colmat
@testset "RZ Julia wrappers" begin
    Random.seed!(304)
    for T in (Float32, Float64, ComplexF32, ComplexF64), (m, n) in ((3, 7), (3, 3), (0, 4))
        original=triu(randn(T, m, n))
        a, tau=LAPACK.tzrzf!(copy(original))
        z=LAPACK.ormrz!('R', 'N', a, tau, Matrix{T}(I, n, n))
        @test hcat(triu(a[:, 1:m]), zeros(T, m, n-m))*z ≈ original
        @test z*z' ≈ Matrix{T}(I, n, n)
        for side in ('L', 'R'), tr in ('N', T<:Real ? 'T' : 'C')
            c=randn(T, side=='L' ? n : 2, side=='L' ? 2 : n)
            result=LAPACK.ormrz!(side, tr, a, tau, copy(c))
            op=tr=='N' ? z : z'
            @test result ≈ (side=='L' ? op*c : c*op)
        end
    end
end

using Libdl
for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    I in (Int32, Int64)

    suffix=I===Int64 ? "64_" : ""
    ft=Symbol(p*"larzt_"*suffix)
    fb=Symbol(p*"larzb_"*suffix)
    @eval function direct_rz_t(
        v::Matrix{$T}, tau::Vector{$T}, ::Type{$I}; library=Colmat.backend_library()
    )
        k, l=size(v)
        t=fill($T(NaN), k+1, k)
        for j in 1:k, i in j:k
            t[i, j]=zero($T)
        end
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(ft)))
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{UInt8},
                Ref{$I},
                Ref{$I},
                Ptr{$T},
                Ref{$I},
                Ptr{$T},
                Ptr{$T},
                Ref{$I},
            ),
            'B',
            'R',
            l,
            k,
            v,
            k,
            tau,
            t,
            size(t, 1),
        )
        t
    end
    @eval function direct_rz_block!(
        side,
        tr,
        v::Matrix{$T},
        t::Matrix{$T},
        c::Matrix{$T},
        ::Type{$I};
        library=Colmat.backend_library(),
    )
        m, n=size(c)
        k, l=size(v)
        ldwork=max(1, side=='L' ? n : m)
        work=zeros($T, ldwork, k)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(fb)))
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{UInt8},
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
            'B',
            'R',
            m,
            n,
            k,
            l,
            v,
            k,
            t,
            size(t, 1),
            c,
            m,
            work,
            ldwork,
        )
        c
    end
end
@testset "RZ block reflector direct ABIs" begin
    external=BLAS.lbt_find_backing_library("dgeqlf_", :ilp64).libname
    for T in (Float32, Float64, ComplexF32, ComplexF64), I in (Int32, Int64), l in (0, 2, 4)
        k, order=2, 8
        v=randn(T, k, l)
        tau=randn(T, k)/4
        tau[2]=0
        t=direct_rz_t(v, tau, I)
        reference=direct_rz_t(copy(v), tau, Int64; library=external)
        @test Matrix(LowerTriangular(t[1:k, :])) ≈
            Matrix(LowerTriangular(reference[1:k, :]))
        @test all(isnan, t[k + 1, :])
        b=zeros(T, order, k)
        b[1:k, :]=Matrix{T}(LinearAlgebra.I, k, k)
        b[(order - l + 1):order, :]=transpose(v)
        h=Matrix{T}(LinearAlgebra.I, order, order)-b*conj.(
            Matrix(LowerTriangular(t[1:k, :]))
        )*b'
        for side in ('L', 'R'), tr in ('N', T<:Real ? 'T' : 'C')
            c=randn(T, side=='L' ? order : 3, side=='L' ? 3 : order)
            actual=direct_rz_block!(side, tr, v, t, copy(c), I)
            op=tr=='N' ? h : h'
            @test actual ≈ (side=='L' ? op*c : c*op)
            @test actual ≈ direct_rz_block!(
                side, tr, copy(v), copy(t), copy(c), Int64; library=external
            )
            middle=(k + 1):(order - l)
            @test if side=='L'
                actual[middle, :]==c[middle, :]
            else
                actual[:, middle]==c[:, middle]
            end
        end
    end
end

for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    I in (Int32, Int64)

    suffix=I===Int64 ? "64_" : ""
    fz=Symbol(p*"tzrzf_"*suffix)
    fq=Symbol(p*"tzrqf_"*suffix)
    fl=Symbol(p*"latrz_"*suffix)
    ap=Symbol(p*(T<:Real ? "ormrz_" : "unmrz_")*suffix)
    @eval function direct_rz_factor!(
        a::Matrix{$T},
        m,
        n,
        ::Type{$I};
        variant=1,
        l=n-m,
        query=false,
        library=Colmat.backend_library(),
    )
        tau=zeros($T, m)
        work=zeros($T, max(1, m))
        info=Ref{$I}(0)
        f=Libdl.dlsym(
            Libdl.dlopen(library),
            ($(QuoteNode(fz)), $(QuoteNode(fq)), $(QuoteNode(fl)))[variant],
        )
        if variant==1
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
        elseif variant==2
            ccall(
                f,
                Cvoid,
                (Ref{$I}, Ref{$I}, Ptr{$T}, Ref{$I}, Ptr{$T}, Ref{$I}),
                m,
                n,
                a,
                size(a, 1),
                tau,
                info,
            )
        else
            ccall(
                f,
                Cvoid,
                (Ref{$I}, Ref{$I}, Ref{$I}, Ptr{$T}, Ref{$I}, Ptr{$T}, Ptr{$T}),
                m,
                n,
                l,
                a,
                size(a, 1),
                tau,
                work,
            )
        end
        tau, work[1], info[]
    end
    @eval function direct_rz_apply!(
        side,
        tr,
        a::Matrix{$T},
        tau::Vector{$T},
        c::Matrix{$T},
        m,
        n,
        l,
        ::Type{$I};
        query=false,
        library=Colmat.backend_library(),
    )
        k=length(tau)
        work=zeros($T, max(1, side=='L' ? n : m))
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
            l,
            a,
            size(a, 1),
            tau,
            c,
            size(c, 1),
            work,
            query ? -1 : length(work),
            info,
        )
        c, work[1], info[]
    end
end
@testset "RZ factorization and application direct ABIs" begin
    external=BLAS.lbt_find_backing_library("dgeqlf_", :ilp64).libname
    for T in (Float32, Float64, ComplexF32, ComplexF64), I in (Int32, Int64), variant in 1:3
        m, n=3, 7
        original=triu(randn(T, m, n))
        a=fill(T(NaN), m+2, n)
        a[1:m, :]=original
        if variant==1
            _, work, info=direct_rz_factor!(a, m, n, I; query=true)
            @test info==0 && real(work)>=m && a[1:m, :]==original
        end
        tau, _, info=direct_rz_factor!(a, m, n, I; variant)
        @test info==0 && all(isnan, a[(m + 1):end, :])
        z, _, info=direct_rz_apply!(
            'R', 'N', a, tau, Matrix{T}(LinearAlgebra.I, n, n), n, n, n-m, I
        )
        @test info==0
        @test hcat(triu(a[1:m, 1:m]), zeros(T, m, n-m))*z ≈ original
        ez, _, info=direct_rz_apply!(
            'R',
            'N',
            a,
            tau,
            Matrix{T}(LinearAlgebra.I, n, n),
            n,
            n,
            n-m,
            Int64;
            library=external,
        )
        @test info==0 && ez ≈ z
        ea=copy(original)
        etau, _, info=direct_rz_factor!(ea, m, n, Int64; variant, library=external)
        @test info==0
        ez, _, info=direct_rz_apply!(
            'R', 'N', ea, etau, Matrix{T}(LinearAlgebra.I, n, n), n, n, n-m, I
        )
        @test info==0 && hcat(triu(ea[:, 1:m]), zeros(T, m, n-m))*ez ≈ original
        for side in ('L', 'R'), tr in ('N', T<:Real ? 'T' : 'C')
            rows, cols=side=='L' ? (n, 4) : (4, n)
            c=randn(T, rows, cols)
            padded=fill(T(NaN), rows+2, cols)
            padded[1:rows, :]=c
            _, _, info=direct_rz_apply!(
                side, tr, a, tau, padded, rows, cols, n-m, I; query=true
            )
            @test info==0 && padded[1:rows, :]==c
            actual, _, info=direct_rz_apply!(side, tr, a, tau, padded, rows, cols, n-m, I)
            op=tr=='N' ? z : z'
            @test info==0 && actual[1:rows, :] ≈ (side=='L' ? op*c : c*op)
            @test all(isnan, actual[(rows + 1):end, :])
        end
    end
end

for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    I in (Int32, Int64)

    suffix=I===Int64 ? "64_" : ""
    larz=Symbol(p*"larz_"*suffix)
    latzm=Symbol(p*"latzm_"*suffix)
    @eval function direct_sparse_rz!(
        side, v::Vector{$T}, inc, tau::$T, c::Matrix{$T}, m, n, l, ::Type{$I}
    )
        work=zeros($T, max(1, side=='L' ? n : m))
        f=Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), $(QuoteNode(larz)))
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
                Ref{$T},
                Ptr{$T},
                Ref{$I},
                Ptr{$T},
            ),
            side,
            m,
            n,
            l,
            v,
            inc,
            tau,
            c,
            size(c, 1),
            work,
        )
        c
    end
    @eval function direct_split_rz!(
        side, v::Vector{$T}, inc, tau::$T, c::Matrix{$T}, ::Type{$I}
    )
        m, n=size(c)
        ld=m+2
        first=fill($T(NaN), ld, max(1, n))
        rest=fill($T(NaN), ld, max(1, n))
        if side=='L'
            first[1, :]=c[1, :]
            rest[1:(m - 1), :]=c[2:m, :]
        else
            first[1:m, 1]=c[:, 1]
            rest[1:m, 1:(n - 1)]=c[:, 2:n]
        end
        work=zeros($T, max(1, side=='L' ? n : m))
        f=Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), $(QuoteNode(latzm)))
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{$I},
                Ref{$I},
                Ptr{$T},
                Ref{$I},
                Ref{$T},
                Ptr{$T},
                Ptr{$T},
                Ref{$I},
                Ptr{$T},
            ),
            side,
            m,
            n,
            v,
            inc,
            tau,
            first,
            rest,
            ld,
            work,
        )
        if side=='L'
            result=vcat(first[1:1, :], rest[1:(m - 1), :])
            guard=all(isnan, first[2:end, :])&&all(isnan, rest[m:end, :])
        else
            result=hcat(first[1:m, 1:1], rest[1:m, 1:(n - 1)])
            guard=all(isnan, first[(m + 1):end, :])&&all(isnan, rest[(m + 1):end, :])
        end
        result, guard
    end
end
@testset "Sparse RZ single reflectors direct ABIs" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        I in (Int32, Int64), side in ('L', 'R'), inc in (1, 2, -1, -2),
        l in (0, 2, 5)

        order=5
        m, n=side=='L' ? (order, 3) : (3, order)
        v=randn(T, l)
        storage=fill(T(NaN), max(0, 1+(l-1)*abs(inc)))
        if l>0
            storage[inc>0 ? (1:inc:length(storage)) : (length(storage):inc:1)]=v
        end
        tau=T<:Real ? T(0.2) : T(0.2+0.1im)
        full=zeros(T, order)
        full[1]=1
        full[(order - l + 1):order].+=v
        h=Matrix{T}(LinearAlgebra.I, order, order)-tau*full*full'
        c=randn(T, m, n)
        padded=fill(T(NaN), m+2, n)
        padded[1:m, :]=c
        actual=direct_sparse_rz!(side, storage, inc, tau, padded, m, n, l, I)
        @test actual[1:m, :] ≈ (side=='L' ? h*c : c*h)
        @test all(isnan, actual[(m + 1):end, :])
        if l<order-1
            middle=2:(order - l)
            poisoned=copy(c)
            if side=='L'
                poisoned[middle, :].=T(NaN)
            else
                poisoned[:, middle].=T(NaN)
            end
            actual=direct_sparse_rz!(side, storage, inc, tau, poisoned, m, n, l, I)
            @test side=='L' ? all(isnan, actual[middle, :]) : all(isnan, actual[:, middle])
            selected=vcat(1, collect((order - l + 1):order))
            reference=side=='L' ? h*c : c*h
            @test if side=='L'
                actual[selected, :] ≈ reference[selected, :]
            else
                actual[:, selected] ≈ reference[:, selected]
            end
        end
    end
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        I in (Int32, Int64), side in ('L', 'R'),
        inc in (1, -2)

        order=5
        m, n=side=='L' ? (order, 3) : (3, order)
        v=randn(T, order-1)
        storage=fill(T(NaN), 1+(order-2)*abs(inc))
        storage[inc>0 ? (1:inc:length(storage)) : (length(storage):inc:1)]=v
        tau=T(0.2)
        full=vcat(one(T), v)
        h=Matrix{T}(LinearAlgebra.I, order, order)-tau*full*full'
        c=randn(T, m, n)
        result, guard=direct_split_rz!(side, storage, inc, tau, c, I)
        @test result ≈ (side=='L' ? h*c : c*h)
        @test guard
    end
end

@testset "RZ overlapping tail and registration" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), I in (Int32, Int64), l in (3, 4, 5)
        order, k=5, 2
        a=randn(T, k, order)
        tau=randn(T, k)/5
        z=Matrix{T}(LinearAlgebra.I, order, order)
        for i in 1:k
            v=zeros(T, order)
            v[i]=1
            v[(order - l + 1):order].+=a[i, (order - l + 1):order]
            z=z*(LinearAlgebra.I-tau[i]*v*v')
        end
        for side in ('L', 'R'), tr in ('N', T<:Real ? 'T' : 'C')
            m, n=side=='L' ? (order, 3) : (3, order)
            c=randn(T, m, n)
            result, _, info=direct_rz_apply!(side, tr, a, tau, copy(c), m, n, l, I)
            op=tr=='N' ? z : z'
            @test info==0 && result ≈ (side=='L' ? op*c : c*op)
        end
    end
    for p in ("s", "d", "c", "z"), abi in (:lp64, :ilp64)
        for family in (
            "larz",
            "larzb",
            "larzt",
            "latrz",
            "latzm",
            "tzrqf",
            "tzrzf",
            p in ("s", "d") ? "ormrz" : "unmrz",
        )
            @test realpath(BLAS.lbt_find_backing_library(p*family*"_", abi).libname)==realpath(
                Colmat.backend_library()
            )
        end
    end
end
