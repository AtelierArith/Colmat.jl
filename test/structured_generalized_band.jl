using Test, LinearAlgebra, Random, Libdl, Colmat
ccall(
    Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error), Cint, ()
)

function structured_band_pack(a, k, u)
    T=eltype(a)
    n=size(a, 2)
    ab=fill(T(NaN), k+3, n)
    ab[(k + 2):(k + 3), :].=777
    for j in 1:n, i in (u=='L' ? (j:min(n, j + k)) : (max(1, j - k):j))
        row=u=='L' ? 1+i-j : k+1+i-j
        ab[row, j]=T<:Complex && i==j ? T(real(a[i, j]), NaN) : a[i, j]
    end
    return ab
end
function structured_band_unpack(ab, k, u)
    T=eltype(ab)
    n=size(ab, 2)
    a=zeros(T, n, n)
    for j in 1:n, i in (u=='L' ? (j:min(n, j + k)) : (max(1, j - k):j))
        row=u=='L' ? 1+i-j : k+1+i-j
        value=i==j ? T(real(ab[row, j])) : ab[row, j]
        a[i, j]=value
        a[j, i]=conj(value)
    end
    return a
end
function structured_band_ignored(ab, old, k, u)
    n=size(ab, 2)
    for j in 1:n, row in axes(ab, 1)
        i=u=='L' ? j+row-1 : j+row-k-1
        selected=row<=k+1 && 1<=i<=n && (u=='L' ? i>=j : i<=j)
        !selected && !isequal(ab[row, j], old[row, j]) && return false
    end
    return true
end
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    R=typeof(real(zero(T)))
    suffix=J===Int64 ? "64_" : ""
    factor=Symbol(prefix*"pbstf_"*suffix)
    gst=Symbol(prefix*(T<:Real ? "sbgst_" : "hbgst_")*suffix)
    @eval function direct_structured_split!(
        bb::Matrix{$T}, ::Type{$J}; kb, u, pointer=nothing
    )
        f=if isnothing(pointer)
            Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), $(QuoteNode(factor)))
        else
            pointer
        end
        info=Ref{$J}(-777)
        ccall(
            f,
            Cvoid,
            (Ref{UInt8}, Ref{$J}, Ref{$J}, Ptr{$T}, Ref{$J}, Ref{$J}, Csize_t),
            u,
            size(bb, 2),
            kb,
            bb,
            size(bb, 1),
            info,
            1,
        )
        diagnostic=ccall(
            Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error),
            Cint,
            (),
        )
        return (; info=info[], diagnostic)
    end
    rwtypes=T<:Real ? [] : [:(Ptr{$R})]
    rwargs=T<:Real ? [] : [:rw]
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
        :(Ptr{$T}),
        :(Ref{$J}),
        :(Ptr{$T}),
        rwtypes...,
        :(Ptr{$J}),
        :(Csize_t),
        :(Csize_t),
    )
    @eval function direct_structured_bandgst!(
        ab::Matrix{$T},
        bb::Matrix{$T},
        x::Matrix{$T},
        ::Type{$J};
        ka,
        kb,
        u,
        vect=('V'),
        pointer=nothing,
    )
        f=if isnothing(pointer)
            Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), $(QuoteNode(gst)))
        else
            pointer
        end
        n=size(ab, 2)
        work=fill($T(777), max(1, ($T<:Real ? 2 : 1)*n))
        rw=fill($R(777), max(1, n))
        info=fill($J(-777), 3)
        ccall(
            f,
            Cvoid,
            $signature,
            vect,
            u,
            n,
            ka,
            kb,
            ab,
            size(ab, 1),
            bb,
            size(bb, 1),
            x,
            max(1, size(x, 1)),
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
        return (; info=info[2], diagnostic, work, rw)
    end
end

@testset "Band generalized standard form with external split factors" begin
    Random.seed!(3082)
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
        external_factor in (false, true)

        R=typeof(real(zero(T)))
        tol=R===Float32 ? 3e-3 : 5e-10
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
        oldab=copy(ab)
        factor=if external_factor
            direct_structured_split!(
                bb, Int64; kb, u, pointer=Libdl.dlsym(reference, Symbol(prefix*"pbstf_64_"))
            )
        else
            direct_structured_split!(bb, J; kb, u)
        end
        @test factor.info==factor.diagnostic==0
        oldbb=copy(bb)
        x=fill(T(777), n+3, n)
        result=direct_structured_bandgst!(ab, bb, x, J; ka, kb, u)
        @test result.info==result.diagnostic==0
        @test isequal(bb, oldbb)
        @test structured_band_ignored(ab, oldab, ka, u)
        @test all(==(T(777)), x[(n + 1):end, :])
        c=structured_band_unpack(ab, ka, u)
        xx=x[1:n, :]
        @test xx'*a*xx≈c rtol=tol atol=tol
        @test xx'*b*xx≈Matrix{T}(I, n, n) rtol=tol atol=tol
        @test all(iszero, c[i, j] for i in 1:n, j in 1:n if abs(i-j)>ka)
        without=copy(oldab)
        unused=fill(T(777), 1, 1)
        result=direct_structured_bandgst!(without, bb, unused, J; ka, kb, u, vect=('N'))
        @test result.info==result.diagnostic==0
        @test unused==fill(T(777), 1, 1)
        @test structured_band_unpack(without, ka, u)≈c rtol=tol atol=tol
        @test isequal(bb, oldbb)
        # The reference chooses a different admissible unitary Q: compare
        # pencil residuals rather than requiring identical transformed entries.
        if ka<n && kb<n
            ar=copy(oldab)
            xr=fill(T(777), n+3, n)
            oracle=direct_structured_bandgst!(
                ar,
                bb,
                xr,
                Int64;
                ka,
                kb,
                u,
                pointer=Libdl.dlsym(
                    reference, Symbol(prefix*(T<:Real ? "sbgst_64_" : "hbgst_64_"))
                ),
            )
            @test oracle.info==0
            cr=structured_band_unpack(ar, ka, u)
            qr=xr[1:n, :]
            @test qr'*a*qr≈cr rtol=tol atol=tol
            @test qr'*b*qr≈Matrix{T}(I, n, n) rtol=tol atol=tol
        end
    end
end
