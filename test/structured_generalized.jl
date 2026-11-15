using Test, LinearAlgebra, Random, Libdl, Colmat

# Other LAPACK argument tests may leave a diagnostic on this calling thread.
# Start this independent group clean; each helper consumes its own diagnostic.
ccall(
    Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error), Cint, ()
)

function structured_gv_pack(a, u; poison=false)
    T=eltype(a)
    n=size(a, 2)
    p=fill(T(777), n*(n+1)÷2+2)
    k=2
    for j in 1:n, i in (u=='L' ? (j:n) : (1:j))
        p[k]=poison && T<:Complex && i==j ? T(real(a[i, j]), NaN) : a[i, j]
        k+=1
    end
    return p
end
function structured_gv_unpack(p, n, u)
    T=eltype(p)
    a=zeros(T, n, n)
    k=2
    for j in 1:n, i in (u=='L' ? (j:n) : (1:j))
        a[i, j]=i==j ? T(real(p[k])) : p[k]
        a[j, i]=conj(a[i, j])
        k+=1
    end
    return a
end
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    R=typeof(real(zero(T)))
    suffix=J===Int64 ? "64_" : ""
    gst=Symbol(prefix*(T<:Real ? "spgst_" : "hpgst_")*suffix)
    gv=Symbol(prefix*(T<:Real ? "spgv_" : "hpgv_")*suffix)
    @eval function direct_structured_gst!(
        ap::Vector{$T}, bp::Vector{$T}, ::Type{$J}; n, kind=1, u=('L'), pointer=nothing
    )
        f=if isnothing(pointer)
            Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), $(QuoteNode(gst)))
        else
            pointer
        end
        info=fill($J(-777), 3)
        ccall(
            f,
            Cvoid,
            (Ref{$J}, Ref{UInt8}, Ref{$J}, Ptr{$T}, Ptr{$T}, Ptr{$J}, Csize_t),
            kind,
            u,
            n,
            view(ap, 2:(length(ap) - 1)),
            view(bp, 2:(length(bp) - 1)),
            view(info, 2:2),
            1,
        )
        diagnostic=ccall(
            Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error),
            Cint,
            (),
        )
        @test info[[1, 3]]==[-777, -777]
        @test ap[[1, end]]==bp[[1, end]]==fill($T(777), 2)
        return (; info=info[2], diagnostic)
    end
    rwtypes=T<:Real ? [] : [:(Ptr{$R})]
    rwargs=T<:Real ? [] : [:rw]
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
        rwtypes...,
        :(Ptr{$J}),
        :(Csize_t),
        :(Csize_t),
    )
    @eval function direct_structured_gv!(
        ap::Vector{$T},
        bp::Vector{$T},
        z::Matrix{$T},
        ::Type{$J};
        n,
        kind=1,
        u=('L'),
        job=('V'),
        pointer=nothing,
    )
        f=if isnothing(pointer)
            Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), $(QuoteNode(gv)))
        else
            pointer
        end
        info=fill($J(-777), 3)
        w=fill($R(777), n+2)
        work=fill($T(777), max(1, ($T<:Real ? 3 : 2)*n))
        rw=fill($R(777), max(1, 3n))
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
            work,
            $(rwargs...),
            view(info, 2:2),
            1,
            1,
        )
        diagnostic=ccall(
            Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error),
            Cint,
            (),
        )
        @test info[[1, 3]]==[-777, -777]
        @test ap[[1, end]]==bp[[1, end]]==fill($T(777), 2)
        @test w[[1, end]]==fill($R(777), 2)
        return (; info=info[2], diagnostic, values=w[2:(n + 1)], work, rw)
    end
end

@testset "Packed generalized standard form and external factors" begin
    Random.seed!(308)
    external=only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    )
    reference=Libdl.dlopen(external.libname)
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64), u in ('U', 'L'), kind in 1:3,
        n in (0, 1, 5, 12)

        R=typeof(real(zero(T)))
        tol=R===Float32 ? 1e-3 : 1e-11
        raw=randn(T, n, n)/max(1, n)
        a=raw+raw'
        lower=tril(randn(T, n, n)/max(1, n))
        for i in 1:n
            lower[i, i]=1+i/max(1, n)
        end
        factor=u=='L' ? lower : Matrix(lower')
        ap=structured_gv_pack(a, u; poison=true)
        bp=structured_gv_pack(factor, u; poison=true)
        ar=copy(ap)
        # HPGST expects a Cholesky factor as returned by PPTRF; its packed
        # triangular BLAS calls read the full complex diagonal. Keep the
        # independent oracle factor valid while probing native diagonal reads.
        br=structured_gv_pack(factor, u)
        oldb=copy(bp)
        result=direct_structured_gst!(ap, bp, J; n, kind, u)
        oracle=direct_structured_gst!(
            ar,
            br,
            Int64;
            n,
            kind,
            u,
            pointer=Libdl.dlsym(
                reference, Symbol(prefix*(T<:Real ? "spgst_64_" : "hpgst_64_"))
            ),
        )
        @test result.info==result.diagnostic==oracle.info==0
        @test isequal(bp, oldb)
        c=structured_gv_unpack(ap, n, u)
        cr=structured_gv_unpack(ar, n, u)
        @test c≈cr rtol=tol atol=tol
        if kind==1
            @test lower*c*lower'≈a rtol=tol atol=tol
        else
            @test c≈lower'*a*lower rtol=tol atol=tol
        end
    end
end
@testset "Packed generalized EVD residuals and B normalization" begin
    Random.seed!(3081)
    external=only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    )
    reference=Libdl.dlopen(external.libname)
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64), u in ('U', 'L'), kind in 1:3, n in (0, 1, 5, 12),
        job in ('N', 'V')

        R=typeof(real(zero(T)))
        tol=R===Float32 ? 2e-3 : 2e-10
        raw=randn(T, n, n)/max(1, n)
        a=raw+raw'
        raw=randn(T, n, n)/max(1, n)
        b=raw*raw'+Matrix{T}(I, n, n)
        ap=structured_gv_pack(a, u; poison=true)
        bp=structured_gv_pack(b, u; poison=true)
        ar=copy(ap)
        br=copy(bp)
        z=fill(T(777), n+3, n)
        zr=copy(z)
        result=direct_structured_gv!(ap, bp, z, J; n, kind, u, job)
        oracle=direct_structured_gv!(
            ar,
            br,
            zr,
            Int64;
            n,
            kind,
            u,
            job,
            pointer=Libdl.dlsym(
                reference, Symbol(prefix*(T<:Real ? "spgv_64_" : "hpgv_64_"))
            ),
        )
        @test result.info==result.diagnostic==oracle.info==0
        @test result.values≈oracle.values rtol=tol atol=tol
        @test all(==(T(777)), z[(n + 1):end, :])
        if job=='V'
            x=z[1:n, :]
            diagonal=Diagonal(result.values)
            if kind==1
                @test a*x≈b*x*diagonal rtol=tol atol=tol
            elseif kind==2
                @test a*b*x≈x*diagonal rtol=tol atol=tol
            else
                @test b*a*x≈x*diagonal rtol=tol atol=tol
            end
            gram=kind==3 ? x'*(b\x) : x'*b*x
            @test gram≈Matrix{T}(I, n, n) rtol=tol atol=tol
        else
            @test all(==(T(777)), z)
        end
        packedfactor=structured_gv_unpack(bp, n, u)
        # Only the stored triangle is a factor; its reflected half is ignored.
        f=u=='L' ? tril(packedfactor) : triu(packedfactor)
        @test (u=='L' ? f*f' : f'*f)≈b rtol=tol atol=tol
    end
end
