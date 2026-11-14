using Test, LinearAlgebra, Random, Libdl, Colmat

for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    R = typeof(real(zero(T)))
    symbol = Symbol(prefix * "steqr_" * (J === Int64 ? "64_" : ""))
    @eval function direct_steqr!(
        d::Vector{$R},
        e::Vector{$R},
        z::Matrix{$T},
        ::Type{$J};
        compz=('I'),
        n=length(d),
        ldz=max(1, size(z, 1)),
        library=Colmat.backend_library(),
    )
        status = Ref{$J}(-777)
        work = zeros($R, max(1, 2length(d)-2))
        f = Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol)))
        ccall(
            f,
            Cvoid,
            (Ref{UInt8}, Ref{$J}, Ptr{$R}, Ptr{$R}, Ptr{$T}, Ref{$J}, Ptr{$R}, Ref{$J}),
            compz,
            n,
            d,
            e,
            z,
            ldz,
            work,
            status,
        )
        status[]
    end
end
for (R, prefix) in ((Float32, "s"), (Float64, "d")), J in (Int32, Int64)
    symbol = Symbol(prefix * "sterf_" * (J === Int64 ? "64_" : ""))
    @eval function direct_sterf!(
        d::Vector{$R},
        e::Vector{$R},
        ::Type{$J};
        n=length(d),
        library=Colmat.backend_library(),
    )
        status=Ref{$J}(-777)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol)))
        ccall(f, Cvoid, (Ref{$J}, Ptr{$R}, Ptr{$R}, Ref{$J}), n, d, e, status)
        status[]
    end
end

@testset "Tridiagonal QR direct ABI and accumulation" begin
    Random.seed!(268)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        n in (0, 1, 2, 7, 40)

        R=typeof(real(zero(T)))
        tolerance=R===Float32 ? 3e-5 : 3e-13
        d=R.(2 .+ (1:n) ./ 9)
        e=fill(R(-0.25), max(0, n-1))
        if n>4
            e[4]=0
        end
        a=Matrix(SymTridiagonal(d, e))
        for compz in ('N', 'I', 'V')
            original=compz=='V' ? Matrix(qr(randn(T, n, n)).Q) : fill(T(NaN), n, n)
            z=fill(T(777), n+3, max(1, n))
            z[1:n, 1:n]=original
            dc, ec=copy(d), copy(e)
            @test direct_steqr!(dc, ec, z, J; compz)==0
            @test issorted(dc)
            @test dc ≈ eigvals(Symmetric(a)) rtol=tolerance
            @test all(iszero, ec)
            @test all(==(T(777)), z[(n + 1):end, :])
            if compz=='N'
                @test all(isnan, z[1:n, 1:n])
            else
                vectors=z[1:n, 1:n]
                @test vectors'*vectors ≈ Matrix{T}(I, n, n) atol=tolerance
                target=compz=='V' ? original*a*original' : a
                @test target*vectors ≈ vectors*Diagonal(dc) atol=tolerance
            end
        end
        if T<:Real
            dc, ec=copy(d), copy(e)
            @test direct_sterf!(dc, ec, J)==0
            @test dc ≈ eigvals(Symmetric(a)) rtol=tolerance
        end
    end
end

@testset "Tridiagonal QR argument widths and nonconvergence fallback" begin
    reference=only(
        filter(
            l->l.interface==:ilp64&&occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        R=typeof(real(zero(T)))
        d=R[2, 3]
        e=R[0.25]
        z=zeros(T, 2, 2)
        @test direct_steqr!(d, e, z, J; compz=('?'))==-1
        @test direct_steqr!(d, e, z, J; n=-1)==-2
        @test direct_steqr!(d, e, z, J; ldz=1)==-6
        if J===Int64
            @test direct_steqr!(d, e, z, J; n=(-(Int64(1)<<32)))==-2
        end
        # NaN reaches the saved implementation, whose partial-output semantics
        # are compared directly rather than assigning an invented native INFO.
        ours_d=R[NaN, 2, 3]
        ours_e=R[0.2, 0.3]
        ours_z=zeros(T, 3, 3)
        ref_d, ref_e, ref_z=copy(ours_d), copy(ours_e), copy(ours_z)
        actual=direct_steqr!(ours_d, ours_e, ours_z, J)
        expected=direct_steqr!(ref_d, ref_e, ref_z, Int64; library=reference)
        @test actual==expected
        @test isequal(ours_d, ref_d)
        @test isequal(ours_e, ref_e)
        @test isequal(ours_z, ref_z)
    end
end

for (R, prefix) in ((Float32, "s"), (Float64, "d")), J in (Int32, Int64)
    symbol=Symbol(prefix*"stevd_"*(J===Int64 ? "64_" : ""))
    stev=Symbol(prefix*"stev_"*(J===Int64 ? "64_" : ""))
    @eval function direct_stevd!(
        d::Vector{$R},
        e::Vector{$R},
        z::Matrix{$R},
        ::Type{$J};
        jobz=('V'),
        lwork=nothing,
        liwork=nothing,
        library=Colmat.backend_library(),
    )
        n=length(d)
        minimum=n>1&&jobz=='V' ? 1+4n+n*n : 1
        integers=n>1&&jobz=='V' ? 3+5n : 1
        lw=isnothing(lwork) ? minimum : lwork
        li=isnothing(liwork) ? integers : liwork
        work=fill($R(777), max(1, lw)+2)
        iwork=fill($J(777), max(1, li)+2)
        info=Ref{$J}(-777)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol)))
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{$J},
                Ptr{$R},
                Ptr{$R},
                Ptr{$R},
                Ref{$J},
                Ptr{$R},
                Ref{$J},
                Ptr{$J},
                Ref{$J},
                Ref{$J},
            ),
            jobz,
            n,
            d,
            e,
            z,
            max(1, size(z, 1)),
            view(work, 2:(length(work) - 1)),
            lw,
            view(iwork, 2:(length(iwork) - 1)),
            li,
            info,
        )
        (info[], work, iwork)
    end
    @eval function direct_stev!(
        d::Vector{$R}, e::Vector{$R}, z::Matrix{$R}, ::Type{$J}; jobz=('V')
    )
        info=Ref{$J}(-777)
        work=zeros($R, max(1, 2length(d)-2))
        f=Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), $(QuoteNode(stev)))
        ccall(
            f,
            Cvoid,
            (Ref{UInt8}, Ref{$J}, Ptr{$R}, Ptr{$R}, Ptr{$R}, Ref{$J}, Ptr{$R}, Ref{$J}),
            jobz,
            length(d),
            d,
            e,
            z,
            max(1, size(z, 1)),
            work,
            info,
        )
        info[]
    end
end

@testset "Tridiagonal STEV/STEVD drivers and workspace" begin
    for R in (Float32, Float64),
        J in (Int32, Int64), n in (0, 1, 2, 7, 40),
        jobz in ('N', 'V')

        d=R.(2 .+ (1:n) ./ 9)
        e=fill(R(-0.25), max(0, n-1))
        z=fill(R(777), n+2, max(1, n))
        a=Matrix(SymTridiagonal(d, e))
        tolerance=R===Float32 ? 5e-5 : 5e-13
        for driver in (:stev, :stevd)
            dc, ec, zc=copy(d), copy(e), copy(z)
            if driver==:stevd
                before=(copy(dc), copy(ec), copy(zc))
                status, w, iw=direct_stevd!(dc, ec, zc, J; jobz, lwork=-1, liwork=-1)
                @test status==0
                @test isequal(before, (dc, ec, zc))
                @test w[[1, end]]==R[777, 777]
                @test iw[[1, end]]==J[777, 777]
                lw, li=Int(w[2]), Int(iw[2])
                @test direct_stevd!(dc, ec, zc, J; jobz, lwork=lw-1)[1]==-8
                @test direct_stevd!(dc, ec, zc, J; jobz, liwork=li-1)[1]==-10
                status, w, iw=direct_stevd!(dc, ec, zc, J; jobz)
                @test status==0
                @test w[[1, end]]==R[777, 777]
                @test iw[[1, end]]==J[777, 777]
            else
                @test direct_stev!(dc, ec, zc, J; jobz)==0
            end
            @test dc ≈ eigvals(Symmetric(a)) rtol=tolerance
            @test zc[(n + 1):end, :]==z[(n + 1):end, :]
            if jobz=='V'
                vectors=zc[1:n, 1:n]
                @test a*vectors ≈ vectors*Diagonal(dc) atol=tolerance
                @test vectors'*vectors ≈ Matrix{R}(I, n, n) atol=tolerance
            else
                @test zc==z
            end
        end
    end
end

for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    R=typeof(real(zero(T)))
    symbol=Symbol(prefix*"pteqr_"*(J===Int64 ? "64_" : ""))
    @eval function direct_pteqr!(
        d::Vector{$R},
        e::Vector{$R},
        z::Matrix{$T},
        ::Type{$J};
        compz=('I'),
        n=length(d),
        ldz=max(1, size(z, 1)),
        library=Colmat.backend_library(),
    )
        info=Ref{$J}(-777)
        work=zeros($R, max(1, 4length(d)))
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol)))
        ccall(
            f,
            Cvoid,
            (Ref{UInt8}, Ref{$J}, Ptr{$R}, Ptr{$R}, Ptr{$T}, Ref{$J}, Ptr{$R}, Ref{$J}),
            compz,
            n,
            d,
            e,
            z,
            ldz,
            work,
            info,
        )
        info[]
    end
end
@testset "Positive tridiagonal SVD path and factorization errors" begin
    reference=only(
        filter(
            l->l.interface==:ilp64&&occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        R=typeof(real(zero(T)))
        tolerance=R===Float32 ? 5e-5 : 5e-13
        for n in (0, 1, 2, 7, 40), compz in ('N', 'I', 'V')
            d=R.(2 .+ (1:n) ./ 9)
            e=fill(R(-0.25), max(0, n-1))
            a=Matrix(SymTridiagonal(d, e))
            original=Matrix(qr(randn(T, n, n)).Q)
            if n==1
                original[1]=one(T)
            end
            z=fill(T(777), n+2, max(1, n))
            z[1:n, 1:n]=compz=='V' ? original : fill(T(NaN), n, n)
            @test direct_pteqr!(d, e, z, J; compz)==0
            @test d ≈ reverse(eigvals(Symmetric(a))) rtol=tolerance
            @test all(==(T(777)), z[(n + 1):end, :])
            if compz!='N'
                vectors=z[1:n, 1:n]
                target=compz=='V' ? original*a*original' : a
                @test target*vectors ≈ vectors*Diagonal(d) atol=tolerance
                @test vectors'*vectors ≈ Matrix{T}(I, n, n) atol=tolerance
            else
                @test all(isnan, z[1:n, 1:n])
            end
        end
        tiny=R===Float32 ? R[1, 1e-10, 1e-20] : R[1, 1e-100, 1e-200]
        values=copy(tiny)
        @test direct_pteqr!(values, zeros(R, 2), zeros(T, 3, 3), J)==0
        @test values ./ tiny ≈ ones(R, 3) rtol=tolerance
        for compz in ('N', 'I', 'V')
            d=R[1, 1, 3]
            e=R[2, 0.5]
            z=fill(T(4), 5, 3)
            rd, re, rz=copy(d), copy(e), copy(z)
            @test direct_pteqr!(d, e, z, J; compz)==2
            @test direct_pteqr!(rd, re, rz, Int64; compz, library=reference)==2
            @test d==rd
            @test e==re
            @test z==rz
        end
    end
end
