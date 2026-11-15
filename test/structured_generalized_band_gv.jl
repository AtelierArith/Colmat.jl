using Test, LinearAlgebra, Random, Libdl, Colmat
if !isdefined(@__MODULE__, :structured_band_pack)
    include("structured_generalized_band.jl")
end
function structured_split_materialize(bb, k, u)
    T=eltype(bb)
    n=size(bb, 2)
    middle=min(n, (n+k)÷2)
    s=zeros(T, n, n)
    for j in 1:n, i in (u=='L' ? (j:min(n, j + k)) : (max(1, j - k):j))
        row=u=='L' ? 1+i-j : k+1+i-j
        value=i==j ? T(real(bb[row, j])) : bb[row, j]
        if u=='L'
            if i>middle
                s[i, j]=value
            else
                s[j, i]=conj(value)
            end
        elseif j>middle
            s[j, i]=conj(value)
        else
            s[i, j]=value
        end
    end
    return s
end
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    R=typeof(real(zero(T)))
    symbol=Symbol(prefix*(T<:Real ? "sbgv_" : "hbgv_")*(J===Int64 ? "64_" : ""))
    rwtypes=T<:Real ? [] : [:(Ptr{$R})]
    rwargs=T<:Real ? [] : [:(view(rw, 2:(length(rw) - 1)))]
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
        rwtypes...,
        :(Ptr{$J}),
        :(Csize_t),
        :(Csize_t),
    )
    @eval function direct_structured_bandgv!(
        ab::Matrix{$T},
        bb::Matrix{$T},
        z::Matrix{$T},
        ::Type{$J};
        ka,
        kb,
        u,
        job=('V'),
        pointer=nothing,
    )
        f=if isnothing(pointer)
            Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), $(QuoteNode(symbol)))
        else
            pointer
        end
        n=size(ab, 2)
        work=fill($T(777), max(1, ($T<:Real ? 3 : 1)*n)+2)
        rw=fill($R(777), max(1, 3n)+2)
        w=fill($R(777), n+2)
        info=fill($J(-777), 3)
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
        @test w[[1, end]]==rw[[1, end]]==fill($R(777), 2)
        @test work[[1, end]]==fill($T(777), 2)
        return (; info=info[2], diagnostic, values=w[2:(n + 1)])
    end
end
@testset "Band generalized EVD direct ABIs and split-factor outputs" begin
    Random.seed!(3084)
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
        n in (0, 1, 5, 9),
        ka in unique((0, 1, 3, n+1)),
        kb in unique((0, min(1, ka), min(2, ka))),
        u in ('U', 'L'),
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
        ab=structured_band_pack(a, ka, u)
        bb=structured_band_pack(b, kb, u)
        olda=copy(ab)
        oldb=copy(bb)
        z=job=='V' ? fill(T(777), n+3, n) : fill(T(777), 1, 1)
        result=direct_structured_bandgv!(ab, bb, z, J; ka, kb, u, job)
        @test result.info==result.diagnostic==0
        @test issorted(result.values)
        @test structured_band_ignored(ab, olda, ka, u) &&
            structured_band_ignored(bb, oldb, kb, u)
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
            ar=copy(olda)
            br=copy(oldb)
            zr=copy(z)
            oracle=direct_structured_bandgv!(
                ar,
                br,
                zr,
                Int64;
                ka,
                kb,
                u,
                job,
                pointer=Libdl.dlsym(
                    reference, Symbol(prefix*(T<:Real ? "sbgv_64_" : "hbgv_64_"))
                ),
            )
            @test oracle.info==0
            @test result.values≈oracle.values rtol=tol atol=tol
        end
    end
end
@testset "Band nonpositive B preserves partial split-factor state" begin
    external=only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    )
    reference=Libdl.dlopen(external.libname)
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64), kb in 0:2, u in ('L', 'U'),
        bad in 1:5

        n=5
        ka=kb
        a=Matrix{T}(I, n, n)
        b=Matrix(Diagonal(T.(5:9)))
        b[bad, bad]=-1
        ab=structured_band_pack(a, ka, u)
        bb=structured_band_pack(b, kb, u)
        olda=copy(ab)
        ar=copy(ab)
        br=copy(bb)
        z=fill(T(777), n+3, n)
        zr=copy(z)
        result=direct_structured_bandgv!(ab, bb, z, J; ka, kb, u)
        oracle=direct_structured_bandgv!(
            ar,
            br,
            zr,
            Int64;
            ka,
            kb,
            u,
            pointer=Libdl.dlsym(
                reference, Symbol(prefix*(T<:Real ? "sbgv_64_" : "hbgv_64_"))
            ),
        )
        @test result.info==oracle.info==n+bad && result.diagnostic==0
        @test isequal(ab, olda)
        @test all(==(T(777)), z) && all(==(real(T(777))), result.values)
        @test all(isequal(x, y)||isapprox(x, y) for (x, y) in zip(bb, br))
    end
end
