using Test, LinearAlgebra, Random, Libdl, Colmat
if !isdefined(@__MODULE__, :structured_split_materialize)
    include("structured_generalized_band_gv.jl")
end
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    R=typeof(real(zero(T)))
    symbol=Symbol(prefix*(T<:Real ? "sbgvx_" : "hbgvx_")*(J===Int64 ? "64_" : ""))
    rwtypes=T<:Real ? [] : [:(Ptr{$R})]
    rwargs=T<:Real ? [] : [:(view(rw, 2:(length(rw) - 1)))]
    signature=Expr(
        :tuple,
        :(Ref{UInt8}),
        :(Ref{UInt8}),
        :(Ref{UInt8}),
        :(Ref{$J}),
        :(Ref{$J}),
        :(Ref{$J}),
        :(Ptr{$T}),
        :(Ref{$J}),
        :(Ptr{$T}),
        :(Ref{$J}),
        :(Ptr{$T}),
        :(Ref{$J}),
        :(Ref{$R}),
        :(Ref{$R}),
        :(Ref{$J}),
        :(Ref{$J}),
        :(Ref{$R}),
        :(Ptr{$J}),
        :(Ptr{$R}),
        :(Ptr{$T}),
        :(Ref{$J}),
        :(Ptr{$T}),
        rwtypes...,
        :(Ptr{$J}),
        :(Ptr{$J}),
        :(Ptr{$J}),
        :(Csize_t),
        :(Csize_t),
        :(Csize_t),
    )
    @eval function direct_structured_bandgvx!(
        ab::Matrix{$T},
        bb::Matrix{$T},
        q::Matrix{$T},
        z::Matrix{$T},
        ::Type{$J};
        ka,
        kb,
        u,
        job=('V'),
        mode=('A'),
        vl=NaN,
        vu=NaN,
        il=1,
        iu=size(ab, 2),
        abstol=0,
        pointer=nothing,
    )
        n=size(ab, 2)
        f=if isnothing(pointer)
            Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), $(QuoteNode(symbol)))
        else
            pointer
        end
        work=fill($T(777), max(1, ($T<:Real ? 7 : 1)*n)+2)
        rw=fill($R(777), max(1, 7n)+2)
        iw=fill($J(-777), max(1, 5n)+2)
        failed=fill($J(-777), max(1, n)+2)
        m=fill($J(-777), 3)
        info=fill($J(-777), 3)
        w=fill($R(777), n+2)
        ccall(
            f,
            Cvoid,
            $signature,
            job,
            mode,
            u,
            n,
            ka,
            kb,
            ab,
            size(ab, 1),
            bb,
            size(bb, 1),
            q,
            max(1, size(q, 1)),
            vl,
            vu,
            il,
            iu,
            abstol,
            view(m, 2:2),
            view(w, 2:(n + 1)),
            z,
            max(1, size(z, 1)),
            view(work, 2:(length(work) - 1)),
            $(rwargs...),
            view(iw, 2:(length(iw) - 1)),
            view(failed, 2:(length(failed) - 1)),
            view(info, 2:2),
            1,
            1,
            1,
        )
        diagnostic=ccall(
            Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error),
            Cint,
            (),
        )
        @test m[[1, 3]]==info[[1, 3]]==iw[[1, end]]==failed[[1, end]]==[-777, -777]
        @test work[[1, end]]==fill($T(777), 2)
        @test w[[1, end]]==rw[[1, end]]==fill($R(777), 2)
        return (;
            info=info[2], m=m[2], diagnostic, values=w[2:(n + 1)], ifail=failed[2:(end - 1)]
        )
    end
end
@testset "Band GVX complete reduction basis selected vectors and expert accuracy" begin
    Random.seed!(3087)
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
        n in (0, 1, 4, 8),
        ka in (0, 1, 3),
        kb in unique((0, min(1, ka))),
        u in ('L', 'U'),
        job in ('N', 'V')

        R=typeof(real(zero(T)))
        tol=R===Float32 ? 5e-3 : 5e-10
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
        full=direct_structured_bandgv!(
            copy(original), copy(originalb), fill(T(777), 1, 1), J; ka, kb, u, job=('N')
        ).values
        for choice in (:all, :index, :value, :empty, :endpoint),
            abstol in (zero(R), 2floatmin(R), sqrt(eps(R)))

            mode=if choice===:all
                'A'
            elseif choice===:index
                'I'
            else
                'V'
            end
            il, iu=n==0 ? (1, 0) : (min(2, n), n)
            vl, vu=if mode!='V'
                (R(NaN), R(NaN))
            elseif n==0
                (R(-1), R(1))
            elseif choice===:empty
                (full[end]+R(2), full[end]+R(3))
            elseif choice===:endpoint
                (full[1]-R(1), full[end])
            elseif n==1
                (full[1]-R(1), full[1]+R(1))
            else
                ((full[1]+full[2])/2, full[end]+R(1))
            end
            ab=copy(original)
            bb=copy(originalb)
            q=job=='V' ? fill(T(777), n+3, n) : fill(T(777), 1, 1)
            z=copy(q)
            qr=copy(q)
            zr=copy(z)
            result=direct_structured_bandgvx!(
                ab, bb, q, z, J; ka, kb, u, job, mode, vl, vu, il, iu, abstol
            )
            oracle=direct_structured_bandgvx!(
                copy(original),
                copy(originalb),
                qr,
                zr,
                Int64;
                ka,
                kb,
                u,
                job,
                mode,
                vl,
                vu,
                il,
                iu,
                abstol,
                pointer=Libdl.dlsym(
                    reference, Symbol(prefix*(T<:Real ? "sbgvx_64_" : "hbgvx_64_"))
                ),
            )
            @test result.info==result.diagnostic==oracle.info==0
            @test result.m==oracle.m
            m=Int(result.m)
            @test result.values[1:m]≈oracle.values[1:m] rtol=tol atol=tol
            @test all(==(R(777)), result.values[(m + 1):end])
            @test result.ifail==oracle.ifail
            @test structured_band_ignored(ab, original, ka, u) &&
                structured_band_ignored(bb, originalb, kb, u)
            factor=structured_split_materialize(bb, kb, u)
            @test factor'*factor≈b rtol=tol atol=tol
            if job=='V'
                qq=q[1:n, :]
                x=z[1:n, 1:m]
                d=Diagonal(result.values[1:m])
                # Q is the full generalized reduction basis, even when M=0.
                @test qq'*b*qq≈Matrix{T}(I, n, n) rtol=tol atol=tol
                reduced=qq'*a*qq
                @test norm(tril(reduced, -2))+norm(triu(reduced, 2))<=tol*max(
                    1, norm(reduced)
                )
                @test qr[1:n, :]'*b*qr[1:n, :]≈Matrix{T}(I, n, n) rtol=tol atol=tol
                for vectors in (x, zr[1:n, 1:m])
                    lhs=a*vectors
                    rhs=b*vectors*d
                    if abstol<=tol
                        @test lhs≈rhs rtol=tol atol=tol
                    else
                        bound=max(tol, tol*max(norm(lhs), norm(rhs)))+abstol*norm(b*vectors)
                        @test norm(lhs-rhs)<=bound
                    end
                end
                @test x'*b*x≈Matrix{T}(I, m, m) rtol=tol atol=tol
                @test all(==(T(777)), q[(n + 1):end, :]) &&
                    all(==(T(777)), z[(n + 1):end, :])
                @test all(==(T(777)), z[:, (m + 1):end])
                if abstol==0 && choice!==:endpoint
                    @test norm(reduced-Diagonal(diag(reduced)))<=tol*max(1, norm(reduced))
                end
            else
                @test q==z==fill(T(777), 1, 1)
            end
        end
    end
end
