using Test, LinearAlgebra, Random, Libdl, Colmat
@testset "Hessenberg Julia wrappers and active ranges" begin
    Random.seed!(293)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        (n, lo, hi) in ((6, 1, 6), (6, 2, 5), (1, 1, 1), (0, 1, 0))

        original=randn(T, n, n)
        for j in 1:(lo - 1), i in (j + 1):n
            original[i, j]=0
        end
        for i in (hi + 1):n, j in 1:(i - 1)
            original[i, j]=0
        end
        a, tau=LAPACK.gehrd!(lo, hi, copy(original))
        q=LAPACK.orghr!(lo, hi, copy(a), tau)
        h=triu(a, -1)
        @test q*h*q' ≈ original
        @test q'*q ≈ Matrix{T}(I, n, n)
        @test a[(hi + 1):n, :]==original[(hi + 1):n, :]
        @test a[:, 1:(lo - 1)]==original[:, 1:(lo - 1)]
        for side in ('L', 'R'), tr in ('N', T<:Real ? 'T' : 'C')
            c=randn(T, side=='L' ? n : 3, side=='L' ? 3 : n)
            result=LAPACK.ormhr!(side, tr, lo, hi, a, tau, copy(c))
            op=tr=='N' ? q : q'
            @test result ≈ (side=='L' ? op*c : c*op)
        end
    end
end

for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    I in (Int32, Int64)

    suffix=I===Int64 ? "64_" : ""
    f=Symbol(p*"gehrd_"*suffix)
    f2=Symbol(p*"gehd2_"*suffix)
    g=Symbol(p*(T<:Real ? "orghr_" : "unghr_")*suffix)
    ap=Symbol(p*(T<:Real ? "ormhr_" : "unmhr_")*suffix)
    @eval function direct_hessenberg!(
        a::Matrix{$T},
        n,
        lo,
        hi,
        ::Type{$I};
        unblocked=false,
        query=false,
        library=Colmat.backend_library(),
    )
        tau=fill($T(NaN), max(0, n-1))
        work=zeros($T, max(1, n))
        info=Ref{$I}(0)
        ptr=Libdl.dlsym(
            Libdl.dlopen(library), unblocked ? $(QuoteNode(f2)) : $(QuoteNode(f))
        )
        if unblocked
            ccall(
                ptr,
                Cvoid,
                (Ref{$I}, Ref{$I}, Ref{$I}, Ptr{$T}, Ref{$I}, Ptr{$T}, Ptr{$T}, Ref{$I}),
                n,
                lo,
                hi,
                a,
                size(a, 1),
                tau,
                work,
                info,
            )
        else
            ccall(
                ptr,
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
                n,
                lo,
                hi,
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
    @eval function direct_hess_q!(
        a::Matrix{$T},
        tau::Vector{$T},
        n,
        lo,
        hi,
        ::Type{$I};
        query=false,
        library=Colmat.backend_library(),
    )
        work=zeros($T, max(1, hi-lo))
        info=Ref{$I}(0)
        ptr=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(g)))
        ccall(
            ptr,
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
            n,
            lo,
            hi,
            a,
            size(a, 1),
            tau,
            work,
            query ? -1 : length(work),
            info,
        )
        a, work[1], info[]
    end
    @eval function direct_hess_apply!(
        side,
        tr,
        a::Matrix{$T},
        tau::Vector{$T},
        c::Matrix{$T},
        m,
        n,
        lo,
        hi,
        ::Type{$I};
        query=false,
        library=Colmat.backend_library(),
    )
        work=zeros($T, max(1, side=='L' ? n : m))
        info=Ref{$I}(0)
        ptr=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(ap)))
        ccall(
            ptr,
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
            lo,
            hi,
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
@testset "Hessenberg direct integer ABIs and external factors" begin
    external=BLAS.lbt_find_backing_library("dgees_", :ilp64).libname
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        I in (Int32, Int64), unblocked in (false, true),
        (n, lo, hi) in ((6, 1, 6), (6, 2, 5), (1, 1, 1), (0, 1, 0))

        original=randn(T, n, n)
        for j in 1:(lo - 1), i in (j + 1):n
            original[i, j]=0
        end
        for i in (hi + 1):n, j in 1:(i - 1)
            original[i, j]=0
        end
        a=fill(T(NaN), n+2, n)
        a[1:n, :]=original
        if !unblocked
            _, _, info=direct_hessenberg!(a, n, lo, hi, I; query=true)
            @test info==0 && a[1:n, :]==original
        end
        tau, _, info=direct_hessenberg!(a, n, lo, hi, I; unblocked)
        @test info==0 && all(isnan, a[(n + 1):end, :])
        qinput=copy(a)
        saved=copy(qinput)
        _, w, info=direct_hess_q!(qinput, tau, n, lo, hi, I; query=true)
        @test info==0 && real(w)>=max(1, hi-lo) && isequal(saved, qinput)
        q, _, info=direct_hess_q!(qinput, tau, n, lo, hi, I)
        @test info==0 && all(isnan, q[(n + 1):end, :])
        q=q[1:n, :]
        @test q*triu(a[1:n, :], -1)*q' ≈ original
        if n>0
            eq, _, info=direct_hess_q!(copy(a), tau, n, lo, hi, Int64; library=external)
            @test info==0 && eq[1:n, :] ≈ q
            ea=copy(original)
            etau, _, info=direct_hessenberg!(
                ea, n, lo, hi, Int64; unblocked, library=external
            )
            @test info==0
            eq, _, info=direct_hess_q!(copy(ea), etau, n, lo, hi, I)
            @test info==0 && eq*triu(ea, -1)*eq' ≈ original
        end
        for side in ('L', 'R'), tr in ('N', T<:Real ? 'T' : 'C')
            rows, cols=side=='L' ? (n, 3) : (3, n)
            c=randn(T, rows, cols)
            padded=fill(T(NaN), rows+2, cols)
            padded[1:rows, :]=c
            saved=copy(padded)
            _, _, info=direct_hess_apply!(
                side, tr, a, tau, padded, rows, cols, lo, hi, I; query=true
            )
            @test info==0 && isequal(saved, padded)
            result, _, info=direct_hess_apply!(
                side, tr, a, tau, padded, rows, cols, lo, hi, I
            )
            op=tr=='N' ? q : q'
            @test info==0 && result[1:rows, :] ≈ (side=='L' ? op*c : c*op)
            @test all(isnan, result[(rows + 1):end, :])
        end
    end
end

@testset "Hessenberg argument boundaries and registration" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), I in (Int32, Int64)
        a=randn(T, 4, 4)
        saved=copy(a)
        @test direct_hessenberg!(a, 4, 0, 4, I)[3]==-2
        @test a==saved
        tau=zeros(T, 3)
        @test direct_hess_q!(a, tau, 4, 1, 5, I)[3]==-3
        @test direct_hess_apply!('L', '?', a, tau, copy(a), 4, 4, 1, 4, I)[3]==-2
    end
    for p in ("s", "d", "c", "z"), abi in (:lp64, :ilp64)
        for family in (
            "gehrd",
            "gehd2",
            p in ("s", "d") ? "orghr" : "unghr",
            p in ("s", "d") ? "ormhr" : "unmhr",
        )
            @test realpath(BLAS.lbt_find_backing_library(p*family*"_", abi).libname)==realpath(
                Colmat.backend_library()
            )
        end
    end
end
