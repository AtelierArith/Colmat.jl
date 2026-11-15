using Test, LinearAlgebra, Random, Libdl, Colmat
if !isdefined(@__MODULE__, :direct_storage_conversion!)
    include("storage_conversions.jl")
end
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    suffix=J===Int64 ? "64_" : ""
    @eval function direct_pc_factor!(
        a::Vector{$T},
        ::Type{$J};
        n,
        u=('U'),
        trr=('N'),
        family="pptrf",
        library=Colmat.backend_library(),
    )
        f=Libdl.dlsym(Libdl.dlopen(library), Symbol($prefix*family*"_"*$suffix))
        info=fill($J(777), 3)
        if startswith(family, "pf")
            ccall(
                f,
                Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{$J}, Ptr{$T}, Ptr{$J}),
                trr,
                u,
                n,
                view(a, 2:(length(a) - 1)),
                view(info, 2:2),
            )
        else
            ccall(
                f,
                Cvoid,
                (Ref{UInt8}, Ref{$J}, Ptr{$T}, Ptr{$J}),
                u,
                n,
                view(a, 2:(length(a) - 1)),
                view(info, 2:2),
            )
        end
        @test info[[1, 3]]==[777, 777]
        @test a[[1, end]]==[777, 777]
        info[2]
    end
    @eval function direct_pc_solve!(
        a::Vector{$T},
        b::Matrix{$T},
        ::Type{$J};
        n,
        u=('U'),
        trr=('N'),
        family="pptrs",
        nrhs=size(b, 2),
        ldb=size(b, 1),
        library=Colmat.backend_library(),
    )
        f=Libdl.dlsym(Libdl.dlopen(library), Symbol($prefix*family*"_"*$suffix))
        info=fill($J(777), 3)
        if family=="pftrs"
            ccall(
                f,
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{UInt8},
                    Ref{$J},
                    Ref{$J},
                    Ptr{$T},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$J},
                ),
                trr,
                u,
                n,
                nrhs,
                view(a, 2:(length(a) - 1)),
                view(b, 2:size(b, 1), :),
                ldb,
                view(info, 2:2),
            )
        else
            ccall(
                f,
                Cvoid,
                (Ref{UInt8}, Ref{$J}, Ref{$J}, Ptr{$T}, Ptr{$T}, Ref{$J}, Ptr{$J}),
                u,
                n,
                nrhs,
                view(a, 2:(length(a) - 1)),
                view(b, 2:size(b, 1), :),
                ldb,
                view(info, 2:2),
            )
        end
        @test info[[1, 3]]==[777, 777]
        @test a[[1, end]]==[777, 777]
        info[2]
    end
end
function pc_matrix(T, n, rng)
    upper=zeros(T, n, n)
    for j in 1:n, i in 1:j
        upper[i, j]=i==j ? T(2+i/16) : randn(rng, T)/8
    end
    a=zeros(T, n, n)
    for j in 1:n, i in 1:n, k in 1:n
        a[i, j]+=conj(upper[k, i])*upper[k, j]
    end
    return a
end
function pc_pack(a, u, trr, rfp; poison=true)
    T=eltype(a)
    n=size(a, 1)
    source=fill(T(NaN), n+2, n)
    for j in 1:n
        for i in (u=='U' ? (1:j) : (j:n))
            source[i, j]=poison&&T<:Complex&&i==j ? T(real(a[i, j]), NaN) : a[i, j]
        end
    end
    packed=fill(T(777), n*(n+1)÷2+2)
    @test direct_storage_conversion!(
        rfp ? "trttf" : "trttp",
        source,
        packed,
        Int64;
        n,
        uplo=u,
        transr=trr,
        library=conversion_reference(),
    )==0
    return packed
end
function pc_unpack(a, n, u, trr, rfp)
    T=eltype(a)
    out=fill(T(777), n+2, n)
    @test direct_storage_conversion!(
        rfp ? "tfttr" : "tpttr",
        a,
        out,
        Int64;
        n,
        uplo=u,
        transr=trr,
        library=conversion_reference(),
    )==0
    for j in 1:n, i in 1:(n + 2)
        if i>n||(u=='U' ? i>j : i<j)
            @test out[i, j]==777
        end
    end
    result=zeros(T, n, n)
    for j in 1:n, i in (u=='U' ? (1:j) : (j:n))
        result[i, j]=out[i, j]
    end
    return result
end
function pc_product(a, b)
    c=zeros(promote_type(eltype(a), eltype(b)), size(a, 1), size(b, 2))
    for j in axes(b, 2), i in axes(a, 1), k in axes(a, 2)
        c[i, j]+=a[i, k]*b[k, j]
    end
    return c
end
function pc_hermitian(a, u)
    n=size(a, 1)
    out=copy(a)
    for j in 1:n, i in (u=='U' ? (1:(j - 1)) : ((j + 1):n))
        out[j, i]=conj(out[i, j])
    end
    return out
end
function pc_rhs(a, nrhs, rng)
    n=size(a, 1)
    x=randn(rng, eltype(a), n, nrhs)
    b=fill(eltype(a)(777), n+3, nrhs)
    b[2:(n + 1), :]=pc_product(a, x)
    return (; b, x)
end

@testset "Packed and all RFP Cholesky layouts reconstruct independent matrices" begin
    reference=conversion_reference()
    rng=MersenneTwister(284)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), n in (0:10..., 35), u in ('U', 'L'), rfp in (false, true),
        trr in (rfp ? (T<:Complex ? ('N', 'C') : ('N', 'T')) : ('N',))

        a=pc_matrix(T, n, rng)
        initial=pc_pack(a, u, trr, rfp)
        packed=copy(initial)
        external=copy(initial)
        family=rfp ? "pftrf" : "pptrf"
        @test direct_pc_factor!(packed, J; n, u, trr, family)==0
        @test direct_pc_factor!(external, Int64; n, u, trr, family, library=reference)==0
        R=typeof(real(zero(T)))
        tol=1000eps(R)
        @test isapprox(packed, external; rtol=tol, atol=tol, nans=true)
        factor=pc_unpack(packed, n, u, trr, rfp)
        actual=if u=='U'
            pc_product(adjoint(factor), factor)
        else
            pc_product(factor, adjoint(factor))
        end
        @test actual≈a rtol=tol atol=tol
        for i in 1:n
            @test imag(factor[i, i])==0
        end
    end
end

@testset "Packed and RFP solve inverse and bidirectional factor reuse" begin
    reference=conversion_reference()
    rng=MersenneTwister(1284)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        n in (0, 1, 2, 3, 6, 9, 35),
        u in ('U', 'L'),
        rfp in (false, true),
        trr in (rfp ? (T<:Complex ? ('N', 'C') : ('N', 'T')) : ('N',))

        a=pc_matrix(T, n, rng)
        initial=pc_pack(a, u, trr, rfp)
        packed=copy(initial)
        external=copy(initial)
        ff=rfp ? "pftrf" : "pptrf"
        sf=rfp ? "pftrs" : "pptrs"
        invf=rfp ? "pftri" : "pptri"
        @test direct_pc_factor!(packed, J; n, u, trr, family=ff)==0
        @test direct_pc_factor!(external, Int64; n, u, trr, family=ff, library=reference)==0
        R=typeof(real(zero(T)))
        tol=2000eps(R)
        for nrhs in (0, 1, 3)
            fixture=pc_rhs(a, nrhs, rng)
            saved=copy(packed)
            for (factors, provider) in (
                (packed, Colmat.backend_library()),
                (external, Colmat.backend_library()),
                (packed, reference),
                (external, reference),
            )
                b=copy(fixture.b)
                K=provider==reference ? Int64 : J
                @test direct_pc_solve!(
                    factors, b, K; n, u, trr, family=sf, library=provider
                )==0
                @test b[2:(n + 1), :]≈fixture.x rtol=tol atol=tol
                @test all(==(777), b[[1, n+2, n+3], :])
            end
            @test isequal(packed, saved)
            if !rfp
                driver=copy(initial)
                b=copy(fixture.b)
                @test direct_pc_solve!(driver, b, J; n, u, family="ppsv")==0
                @test b[2:(n + 1), :]≈fixture.x rtol=tol atol=tol
                @test all(==(777), b[[1, n+2, n+3], :])
            end
        end
        nativeinv=copy(packed)
        referenceinv=copy(external)
        @test direct_pc_factor!(nativeinv, J; n, u, trr, family=invf)==0
        @test direct_pc_factor!(
            referenceinv, Int64; n, u, trr, family=invf, library=reference
        )==0
        @test nativeinv≈referenceinv rtol=tol atol=tol
        inverse=pc_hermitian(pc_unpack(nativeinv, n, u, trr, rfp), u)
        @test pc_product(a, inverse)≈Matrix{T}(I, n, n) rtol=tol atol=tol
        for (factors, provider, K) in
            ((external, Colmat.backend_library(), J), (packed, reference, Int64))
            cross=copy(factors)
            @test direct_pc_factor!(cross, K; n, u, trr, family=invf, library=provider)==0
            crossinverse=pc_hermitian(pc_unpack(cross, n, u, trr, rfp), u)
            @test pc_product(a, crossinverse)≈Matrix{T}(I, n, n) rtol=tol atol=tol
        end
    end
end

@testset "Packed and RFP nonpositive factor INFO and singular inverse" begin
    reference=conversion_reference()
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        n in (1, 2, 5, 8),
        u in ('U', 'L'),
        rfp in (false, true),
        trr in (rfp ? (T<:Complex ? ('N', 'C') : ('N', 'T')) : ('N',)),
        position in 1:n,
        bad in (0, -1)

        a=Matrix{T}(I, n, n)
        a[position, position]=T(bad)
        packed=pc_pack(a, u, trr, rfp)
        external=copy(packed)
        before=copy(packed)
        ff=rfp ? "pftrf" : "pptrf"
        invf=rfp ? "pftri" : "pptri"
        @test direct_pc_factor!(packed, J; n, u, trr, family=ff)==position
        @test direct_pc_factor!(external, Int64; n, u, trr, family=ff, library=reference)==position
        # A completed leading principal factor remains meaningful at INFO>0.
        partial=pc_unpack(packed, n, u, trr, rfp)
        for i in 1:(position - 1)
            @test partial[i, i]==1
        end
        if !rfp
            b=fill(T(777), n+3, 2)
            driver=copy(before)
            @test direct_pc_solve!(driver, b, J; n, u, family="ppsv")==position
            @test all(==(777), b)
        end
        a[position, position]=0
        singular=pc_pack(a, u, trr, rfp; poison=false)
        saved=copy(singular)
        @test direct_pc_factor!(singular, J; n, u, trr, family=invf)==position
        @test isequal(singular, saved)
    end
end

@testset "All 28 packed and RFP Cholesky symbols compute through LBT" begin
    rng=MersenneTwister(2284)
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64)

        abi=J===Int64 ? :ilp64 : :lp64
        suffix=J===Int64 ? "64_" : ""
        for family in ("pptrf", "pptrs", "pptri", "ppsv", "pftrf", "pftrs", "pftri")
            native=Libdl.dlsym(
                Libdl.dlopen(Colmat.backend_library()), Symbol(prefix*family*"_"*suffix)
            )
            @test BLAS.lbt_get_forward(prefix*family*"_", abi)==native
        end
        n=5
        u='L'
        a=pc_matrix(T, n, rng)
        R=typeof(real(zero(T)))
        tol=2000eps(R)
        fixture=pc_rhs(a, 2, rng)
        for rfp in (false, true)
            trr=rfp ? (T<:Complex ? 'C' : 'T') : 'N'
            ff=rfp ? "pftrf" : "pptrf"
            sf=rfp ? "pftrs" : "pptrs"
            invf=rfp ? "pftri" : "pptri"
            initial=pc_pack(a, u, trr, rfp)
            packed=copy(initial)
            @test direct_pc_factor!(
                packed, J; n, u, trr, family=ff, library=BLAS.libblastrampoline
            )==0
            b=copy(fixture.b)
            @test direct_pc_solve!(
                packed, b, J; n, u, trr, family=sf, library=BLAS.libblastrampoline
            )==0
            @test b[2:(n + 1), :]≈fixture.x rtol=tol atol=tol
            @test direct_pc_factor!(
                packed, J; n, u, trr, family=invf, library=BLAS.libblastrampoline
            )==0
            inverse=pc_hermitian(pc_unpack(packed, n, u, trr, rfp), u)
            @test pc_product(a, inverse)≈Matrix{T}(I, n, n) rtol=tol atol=tol
            if !rfp
                b=copy(fixture.b)
                @test direct_pc_solve!(
                    initial, b, J; n, u, family="ppsv", library=BLAS.libblastrampoline
                )==0
                @test b[2:(n + 1), :]≈fixture.x rtol=tol atol=tol
            end
        end
    end
end

@testset "Packed and RFP argument precedence INFO width and empty RHS" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        packed=fill(T(777), 8)
        b=fill(T(777), 6, 1)
        for family in ("pptrf", "pptri")
            @test direct_pc_factor!(packed, J; n=-1, u=('X'), family)==-1
            f=Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error)
            @test ccall(f, Cint, ())==1
            @test direct_pc_factor!(packed, J; n=-1, family)==-2
            @test ccall(f, Cint, ())==2
        end
        for family in ("pftrf", "pftri"),
            (changes, arg) in
            (((; trr=('X'), u=('X'), n=-1), 1), ((; u=('X'), n=-1), 2), ((; n=-1), 3))

            args=merge((; n=3, u=('U'), trr=('N')), changes)
            @test direct_pc_factor!(packed, J; args..., family)==-arg
            f=Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error)
            @test ccall(f, Cint, ())==arg
        end
        for family in ("pptrs", "ppsv"),
            (changes, arg) in
            (((; u=('X'), n=-1), 1), ((; n=-1), 2), ((; nrhs=-1), 3), ((; ldb=2), 6))

            args=merge((; n=3, u=('U'), nrhs=1, ldb=6), changes)
            @test direct_pc_solve!(packed, b, J; args..., family)==-arg
            f=Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error)
            @test ccall(f, Cint, ())==arg
        end
        for (changes, arg) in (
            ((; trr=('X'), u=('X'), n=-1), 1),
            ((; u=('X'), n=-1), 2),
            ((; n=-1), 3),
            ((; nrhs=-1), 4),
            ((; ldb=2), 7),
        )
            args=merge((; n=3, u=('U'), trr=('N'), nrhs=1, ldb=6), changes)
            @test direct_pc_solve!(packed, b, J; args..., family="pftrs")==-arg
            f=Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error)
            @test ccall(f, Cint, ())==arg
        end
        @test all(==(777), packed)
        @test all(==(777), b)
        if J===Int64
            large=(Int64(1)<<32)+1
            @test direct_pc_solve!(packed, b, J; n=large, nrhs=0, ldb=large)==0
            @test direct_pc_solve!(
                packed, b, J; n=large, nrhs=0, ldb=large, family="pftrs"
            )==0
        end
    end
end

@testset "Packed driver factors with zero RHS without touching B" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        u in ('U', 'L')

        a=4*Matrix{T}(I, 3, 3)
        packed=pc_pack(a, u, 'N', false)
        b=fill(T(NaN), 6, 1)
        before=copy(b)
        @test direct_pc_solve!(packed, b, J; n=3, u, nrhs=0, family="ppsv")==0
        @test isequal(b, before)
        factor=pc_unpack(packed, 3, u, 'N', false)
        @test diag(factor)==fill(T(2), 3)
        illegal=T<:Complex ? 'T' : 'C'
        @test direct_pc_factor!(packed, J; n=0, u, trr=illegal, family="pftrf")==-1
        diagnostic=Libdl.dlsym(
            Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error
        )
        @test ccall(diagnostic, Cint, ())==1
    end
end
