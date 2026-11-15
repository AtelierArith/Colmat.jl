using Test, LinearAlgebra, Random, Libdl, Colmat
if !isdefined(@__MODULE__, :structured_gv_pack)
    include("structured_generalized.jl")
end

function structured_packed_dc_sizes(T, n, job)
    n<=1 && return (1, 1, 1)
    if T<:Real
        job=='V' ? (1+6n+2n^2, 1, 3+5n) : (2n, 1, 1)
    else
        job=='V' ? (2n, 1+5n+2n^2, 3+5n) : (n, n, 1)
    end
end
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    R=typeof(real(zero(T)))
    symbol=Symbol(prefix*(T<:Real ? "spgvd_" : "hpgvd_")*(J===Int64 ? "64_" : ""))
    rwtypes=T<:Real ? [] : [:(Ptr{$R}), :(Ref{$J})]
    rwargs=T<:Real ? [] : [:(view(rw, 2:(length(rw) - 1))), :(query===:real ? -1 : lr)]
    signature=Expr(
        :tuple,
        :(Ref{$J}),
        :(Ref{UInt8}),
        :(Ref{UInt8}),
        :(Ref{$J}),
        :(Ptr{$T}),
        :(Ptr{$T}),
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
    @eval function direct_structured_gvd!(
        ap::Vector{$T},
        bp::Vector{$T},
        z::Matrix{$T},
        ::Type{$J};
        n,
        kind=1,
        u=('L'),
        job=('V'),
        query=:none,
        lengths=nothing,
        pointer=nothing,
    )
        nw, nr, ni=structured_packed_dc_sizes($T, n, job)
        lw, lr, li=isnothing(lengths) ? (nw, nr, ni) : lengths
        f=if isnothing(pointer)
            Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), $(QuoteNode(symbol)))
        else
            pointer
        end
        payloads=query===:none ? (max(1, lw), max(1, lr), max(1, li)) : (1, 1, 1)
        work=fill($T(777), payloads[1]+2)
        rw=fill($R(777), payloads[2]+2)
        iw=fill($J(-777), payloads[3]+2)
        w=fill($R(777), n+2)
        info=fill($J(-777), 3)
        ccall(
            f,
            Cvoid,
            $signature,
            kind,
            job,
            u,
            n,
            view(ap, 2:(length(ap) - 1)),
            view(bp, 2:(length(bp) - 1)),
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
        @test ap[[1, end]]==bp[[1, end]]==work[[1, end]]==fill($T(777), 2)
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

@testset "Packed GVD minimum payloads residuals and all workspace queries" begin
    Random.seed!(3083)
    external=only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    )
    reference=Libdl.dlopen(external.libname)
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64), n in (0, 1, 4, 9), job in ('N', 'V'), kind in 1:3,
        u in ('L', 'U')

        R=typeof(real(zero(T)))
        tol=R===Float32 ? 3e-3 : 3e-10
        a=randn(T, n, n)/max(1, n)
        a=a+a'
        raw=randn(T, n, n)/max(1, n)
        b=raw*raw'+Matrix{T}(I, n, n)
        original=structured_gv_pack(a, u; poison=true)
        originalb=structured_gv_pack(b, u; poison=true)
        nw, nr, ni=structured_packed_dc_sizes(T, n, job)
        for query in (T<:Real ? (:work, :integer) : (:work, :real, :integer))
            ap=copy(original)
            bp=copy(originalb)
            z=fill(T(777), n+3, n)
            result=direct_structured_gvd!(ap, bp, z, J; n, kind, u, job, query)
            oracle=direct_structured_gvd!(
                copy(original),
                copy(originalb),
                copy(z),
                Int64;
                n,
                kind,
                u,
                job,
                query,
                pointer=Libdl.dlsym(
                    reference, Symbol(prefix*(T<:Real ? "spgvd_64_" : "hpgvd_64_"))
                ),
            )
            @test result.info==result.diagnostic==oracle.info==0
            @test real(result.work[1])==real(oracle.work[1])==nw
            @test result.iw[1]==oracle.iw[1]==ni
            T<:Complex && (@test result.rw[1]==oracle.rw[1]==nr)
            @test isequal(ap, original) && isequal(bp, originalb)
            @test all(==(T(777)), z) && all(==(R(777)), result.values)
        end
        ap=copy(original)
        bp=copy(originalb)
        z=fill(T(777), n+3, n)
        result=direct_structured_gvd!(ap, bp, z, J; n, kind, u, job)
        oracle=direct_structured_gvd!(
            copy(original),
            copy(originalb),
            copy(z),
            Int64;
            n,
            kind,
            u,
            job,
            pointer=Libdl.dlsym(
                reference, Symbol(prefix*(T<:Real ? "spgvd_64_" : "hpgvd_64_"))
            ),
        )
        @test result.info==result.diagnostic==oracle.info==0
        @test length(result.work)==nw && length(result.iw)==ni
        @test result.values≈oracle.values rtol=tol atol=tol
        @test all(==(T(777)), z[(n + 1):end, :])
        if job=='V'
            x=z[1:n, :]
            d=Diagonal(result.values)
            @test (
                if kind==1
                    a*x
                elseif kind==2
                    a*b*x
                else
                    b*a*x
                end
            )≈(kind==1 ? b*x*d : x*d) rtol=tol atol=tol
            @test (kind==3 ? x'*(b\x) : x'*b*x)≈Matrix{T}(I, n, n) rtol=tol atol=tol
        else
            @test all(==(T(777)), z)
        end
    end
end
@testset "Packed GVD insufficient workspace does not read inputs" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        n=3
        nw, nr, ni=structured_packed_dc_sizes(T, n, 'V')
        for (lengths, expected) in
            (((nw-1, nr, ni), 11), ((nw, nr, ni-1), T<:Real ? 13 : 15))
            ap=fill(T(777), n*(n+1)÷2+2)
            bp=copy(ap)
            z=fill(T(777), n+3, n)
            result=direct_structured_gvd!(ap, bp, z, J; n, lengths)
            @test result.info == -expected && result.diagnostic==expected
            @test all(==(T(777)), ap) && all(==(T(777)), bp) && all(==(T(777)), z)
        end
        if T<:Complex
            ap=fill(T(777), n*(n+1)÷2+2)
            bp=copy(ap)
            z=fill(T(777), n+3, n)
            result=direct_structured_gvd!(ap, bp, z, J; n, lengths=(nw, nr-1, ni))
            @test result.info == -13 && result.diagnostic==13
        end
    end
end
