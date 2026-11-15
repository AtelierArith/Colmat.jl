using Test, LinearAlgebra, Random, Libdl, Colmat
if !isdefined(@__MODULE__, :structured_split_materialize)
    include("structured_generalized_band_gv.jl")
end
function structured_band_dc_sizes(T, n, job)
    n<=1 && return T<:Real ? (1, 1, 1) : (1+n, 1+n, 1)
    if job=='V'
        (T<:Real ? 1+5n+2n^2 : 2n^2, T<:Real ? 1 : 1+5n+2n^2, 3+5n)
    else
        T<:Real ? (2n, 1, 1) : (n, n, 1)
    end
end
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    R=typeof(real(zero(T)))
    symbol=Symbol(prefix*(T<:Real ? "sbgvd_" : "hbgvd_")*(J===Int64 ? "64_" : ""))
    rwtypes=T<:Real ? [] : [:(Ptr{$R}), :(Ref{$J})]
    rwargs=T<:Real ? [] : [:(view(rw, 2:(length(rw) - 1))), :(query===:real ? -1 : lr)]
    signature=Expr(
        :tuple,
        :(Ref{UInt8}),
        :(Ref{UInt8}),
        :(Ref{$J}),
        :(Ref{$J}),
        :(Ref{$J}),
        :(Ptr{$T}),
        :(Ref{$J}),
        :(Ptr{$T}),
        :(Ref{$J}),
        :(Ptr{$R}),
        :(Ptr{$T}),
        :(Ref{$J}),
        :(Ptr{$T}),
        :(Ref{$J}),
        rwtypes...,
        :(Ptr{$J}),
        :(Ref{$J}),
        :(Ptr{$J}),
        :(Csize_t),
        :(Csize_t),
    )
    @eval function direct_structured_bandgvd!(
        ab::Matrix{$T},
        bb::Matrix{$T},
        z::Matrix{$T},
        ::Type{$J};
        ka,
        kb,
        u,
        job=('V'),
        query=:none,
        lengths=nothing,
        pointer=nothing,
    )
        n=size(ab, 2)
        nw, nr, ni=structured_band_dc_sizes($T, n, job)
        lw, lr, li=isnothing(lengths) ? (nw, nr, ni) : lengths
        payloads=query===:none ? (max(1, lw), max(1, lr), max(1, li)) : (1, 1, 1)
        work=fill($T(777), payloads[1]+2)
        rw=fill($R(777), payloads[2]+2)
        iw=fill($J(-777), payloads[3]+2)
        w=fill($R(777), n+2)
        info=fill($J(-777), 3)
        f=if isnothing(pointer)
            Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), $(QuoteNode(symbol)))
        else
            pointer
        end
        ccall(
            f,
            Cvoid,
            $signature,
            job,
            u,
            n,
            ka,
            kb,
            ab,
            size(ab, 1),
            bb,
            size(bb, 1),
            view(w, 2:(n + 1)),
            z,
            max(1, size(z, 1)),
            view(work, 2:(length(work) - 1)),
            query===:work ? -1 : lw,
            $(rwargs...),
            view(iw, 2:(length(iw) - 1)),
            query===:integer ? -1 : li,
            view(info, 2:2),
            1,
            1,
        )
        diagnostic=ccall(
            Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error),
            Cint,
            (),
        )
        @test info[[1, 3]]==iw[[1, end]]==[-777, -777]
        @test work[[1, end]]==fill($T(777), 2)
        @test w[[1, end]]==rw[[1, end]]==fill($R(777), 2)
        return (;
            info=info[2],
            diagnostic,
            values=w[2:(n + 1)],
            work=work[2:(end - 1)],
            rw=rw[2:(end - 1)],
            iw=iw[2:(end - 1)],
        )
    end
end
@testset "Band GVD exact workspaces queries and normalized eigenpairs" begin
    Random.seed!(3085)
    external=only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    )
    reference=Libdl.dlopen(external.libname)
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64),
        n in (0, 1, 4, 9),
        ka in (0, 1, 3),
        kb in unique((0, min(1, ka))),
        u in ('L', 'U'),
        job in ('N', 'V')

        R=typeof(real(zero(T)))
        tol=R===Float32 ? 4e-3 : 5e-10
        a=randn(T, n, n)/max(1, n)
        a=a+a'
        b=randn(T, n, n)/max(1, n)
        b=b+b'
        for j in 1:n, i in 1:n
            abs(i-j)>ka && (a[i, j]=0)
            abs(i-j)>kb && (b[i, j]=0)
        end
        for i in 1:n
            b[i, i]=1+sum(abs, b[i, :])
        end
        original=structured_band_pack(a, ka, u)
        originalb=structured_band_pack(b, kb, u)
        nw, nr, ni=structured_band_dc_sizes(T, n, job)
        pointer=Libdl.dlsym(reference, Symbol(prefix*(T<:Real ? "sbgvd_64_" : "hbgvd_64_")))
        for query in (T<:Real ? (:work, :integer) : (:work, :real, :integer))
            ab=copy(original)
            bb=copy(originalb)
            z=fill(T(777), n+3, n)
            result=direct_structured_bandgvd!(ab, bb, z, J; ka, kb, u, job, query)
            oracle=direct_structured_bandgvd!(
                copy(original),
                copy(originalb),
                copy(z),
                Int64;
                ka,
                kb,
                u,
                job,
                query,
                pointer,
            )
            @test result.info==result.diagnostic==oracle.info==0
            @test real(result.work[1])==real(oracle.work[1])==nw &&
                result.iw[1]==oracle.iw[1]==ni
            T<:Complex && (@test result.rw[1]==oracle.rw[1]==nr)
            @test isequal(ab, original) && isequal(bb, originalb)
            @test all(==(T(777)), z) && all(==(R(777)), result.values)
        end
        ab=copy(original)
        bb=copy(originalb)
        z=job=='V' ? fill(T(777), n+3, n) : fill(T(777), 1, 1)
        result=direct_structured_bandgvd!(ab, bb, z, J; ka, kb, u, job)
        @test result.info==result.diagnostic==0
        @test length(result.work)==nw && length(result.iw)==ni
        @test structured_band_ignored(ab, original, ka, u) &&
            structured_band_ignored(bb, originalb, kb, u)
        factor=structured_split_materialize(bb, kb, u)
        @test factor'*factor≈b rtol=tol atol=tol
        if job=='V'
            x=z[1:n, :]
            d=Diagonal(result.values)
            @test a*x≈b*x*d rtol=tol atol=tol
            @test x'*b*x≈Matrix{T}(I, n, n) rtol=tol atol=tol
            @test all(==(T(777)), z[(n + 1):end, :])
        else
            @test z==fill(T(777), 1, 1)
        end
        if ka<n && kb<n
            # Real Reference-LAPACK SBGVD N=1/JOBZ=V accesses WORK(2:3)
            # despite advertising LWORK=1, and leaves an unnormalized vector.
            # Give only the oracle extra storage, compare its eigenvalues, and
            # verify native vectors independently with the unchanged tolerance.
            # See reproducers/openblas-sbgvd-singleton-workspace.jl.
            lengths=n==1 && T<:Real && job=='V' ? (3, nr, ni) : (nw, nr, ni)
            oracle=direct_structured_bandgvd!(
                copy(original),
                copy(originalb),
                copy(z),
                Int64;
                ka,
                kb,
                u,
                job,
                pointer,
                lengths,
            )
            @test oracle.info==0
            @test result.values≈oracle.values rtol=tol atol=tol
        end
    end
end
@testset "Band GVD insufficient workspaces preserve input" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64), n in (0, 1, 3)
        nw, nr, ni=structured_band_dc_sizes(T, n, 'V')
        ab=fill(T(777), 3, n)
        bb=copy(ab)
        z=fill(T(777), n+3, n)
        cases=if T<:Real
            (((nw-1, nr, ni), 14), ((nw, nr, ni-1), 16))
        else
            (((nw-1, nr, ni), 14), ((nw, nr-1, ni), 16), ((nw, nr, ni-1), 18))
        end
        for (lengths, arg) in cases
            result=direct_structured_bandgvd!(ab, bb, z, J; ka=0, kb=0, u=('L'), lengths)
            @test result.info == -arg && result.diagnostic==arg
            @test all(==(T(777)), ab) && all(==(T(777)), bb) && all(==(T(777)), z)
        end
    end
end
