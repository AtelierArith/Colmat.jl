using Test, LinearAlgebra, Random, Libdl, Colmat
if !isdefined(@__MODULE__, :structured_gv_pack)
    include("structured_generalized.jl")
end
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    R=typeof(real(zero(T)))
    symbol=Symbol(prefix*(T<:Real ? "spgvx_" : "hpgvx_")*(J===Int64 ? "64_" : ""))
    rwtypes=T<:Real ? [] : [:(Ptr{$R})]
    rwargs=T<:Real ? [] : [:(view(rw, 2:(length(rw) - 1)))]
    signature=Expr(
        :tuple,
        :(Ref{$J}),
        :(Ref{UInt8}),
        :(Ref{UInt8}),
        :(Ref{UInt8}),
        :(Ref{$J}),
        :(Ptr{$T}),
        :(Ptr{$T}),
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
    @eval function direct_structured_gvx!(
        ap::Vector{$T},
        bp::Vector{$T},
        z::Matrix{$T},
        ::Type{$J};
        n,
        kind=1,
        u=('L'),
        job=('V'),
        mode=('A'),
        vl=NaN,
        vu=NaN,
        il=1,
        iu=n,
        abstol=0,
        pointer=nothing,
    )
        f=if isnothing(pointer)
            Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), $(QuoteNode(symbol)))
        else
            pointer
        end
        work=fill($T(777), max(1, ($T<:Real ? 8 : 2)*n)+2)
        rw=fill($R(777), max(1, 7n)+2)
        iw=fill($J(-777), max(1, 5n)+2)
        failed=fill($J(-777), max(1, n)+2)
        w=fill($R(777), n+2)
        m=fill($J(-777), 3)
        info=fill($J(-777), 3)
        ccall(
            f,
            Cvoid,
            $signature,
            kind,
            job,
            mode,
            u,
            n,
            view(ap, 2:(length(ap) - 1)),
            view(bp, 2:(length(bp) - 1)),
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
        @test ap[[1, end]]==bp[[1, end]]==work[[1, end]]==fill($T(777), 2)
        @test w[[1, end]]==rw[[1, end]]==fill($R(777), 2)
        return (;
            info=info[2],
            m=m[2],
            diagnostic,
            values=w[2:(n + 1)],
            ifail=failed[2:(end - 1)],
            iwork=iw[2:(end - 1)],
        )
    end
end

@testset "Packed GVX range selection expert accuracy and output bounds" begin
    Random.seed!(3086)
    external=only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    )
    reference=Libdl.dlopen(external.libname)
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64), n in (0, 1, 4, 9), u in ('L', 'U'), kind in 1:3,
        job in ('N', 'V')

        R=typeof(real(zero(T)))
        tol=R===Float32 ? 5e-3 : 5e-10
        raw=randn(T, n, n)/max(1, n)
        a=raw+raw'
        raw=randn(T, n, n)/max(1, n)
        b=raw*raw'+Matrix{T}(I, n, n)
        original=structured_gv_pack(a, u; poison=true)
        originalb=structured_gv_pack(b, u; poison=true)
        gv=Libdl.dlsym(reference, Symbol(prefix*(T<:Real ? "spgv_64_" : "hpgv_64_")))
        full=direct_structured_gv!(
            copy(original),
            copy(originalb),
            fill(T(777), 1, 1),
            Int64;
            n,
            kind,
            u,
            job=('N'),
            pointer=gv,
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
            ap=copy(original)
            bp=copy(originalb)
            z=fill(T(777), n+3, n)
            result=direct_structured_gvx!(
                ap, bp, z, J; n, kind, u, job, mode, vl, vu, il, iu, abstol
            )
            oracle_z=fill(T(777), n+3, n)
            oracle=direct_structured_gvx!(
                copy(original),
                copy(originalb),
                oracle_z,
                Int64;
                n,
                kind,
                u,
                job,
                mode,
                vl,
                vu,
                il,
                iu,
                abstol,
                pointer=Libdl.dlsym(
                    reference, Symbol(prefix*(T<:Real ? "spgvx_64_" : "hpgvx_64_"))
                ),
            )
            @test result.info==result.diagnostic==oracle.info==0
            # Reference HPGVX leaves M untouched on its N=0 quick return.
            # The adapter supplies the defined empty selection count instead.
            if n==0
                @test result.m==0
                @test oracle.m in (0, -777)
            else
                @test result.m==oracle.m
            end
            m=Int(result.m)
            x=z[1:n, 1:m]
            @test result.values[1:m]≈oracle.values[1:m] rtol=tol atol=tol
            @test all(==(R(777)), result.values[(m + 1):end])
            @test result.ifail==oracle.ifail
            @test all(==(T(777)), z[(n + 1):end, :])
            if job=='V'
                d=Diagonal(result.values[1:m])
                for vectors in (x, oracle_z[1:n, 1:m])
                    lhs=if kind==1
                        a*vectors
                    elseif kind==2
                        a*b*vectors
                    else
                        b*a*vectors
                    end
                    rhs=kind==1 ? b*vectors*d : vectors*d
                    if abstol<=tol
                        @test lhs≈rhs rtol=tol atol=tol
                    else
                        # Positive ABSTOL deliberately permits coarse eigenvalues.
                        # Keep the machine-accuracy tolerance unchanged and add
                        # the requested absolute eigenvalue error in this pencil.
                        basis=kind==1 ? b*vectors : vectors
                        bound=max(tol, tol*max(norm(lhs), norm(rhs)))+abstol*norm(basis)
                        @test norm(lhs-rhs)<=bound
                    end
                end
                @test (kind==3 ? x'*(b\x) : x'*b*x)≈Matrix{T}(I, m, m) rtol=tol atol=tol
                if n==1 && m==0
                    @test z[1, 1]==one(T)
                else
                    @test all(==(T(777)), z[:, (m + 1):end])
                end
            else
                @test all(==(T(777)), z)
            end
        end
    end
end
