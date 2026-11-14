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

function tridiagonal_dc_sizes(T, n, compz)
    (n<=1||compz=='N')&&return (1, 1, 1)
    n<=25&&return T<:Complex ? (1, 2(n-1), 1) : (2(n-1), 1, 1)
    lg=ceil(Int, log2(n))
    if compz=='V'
        real=1+3n+2n*lg+4n*n
        return (T<:Complex ? n*n : real, T<:Complex ? real : 1, 6+6n+5n*lg)
    end
    return T<:Complex ? (1, 1+4n+2n*n, 3+5n) : (1+4n+n*n, 1, 3+5n)
end
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    R=typeof(real(zero(T)))
    symbol=Symbol(prefix*"stedc_"*(J===Int64 ? "64_" : ""))
    @eval function direct_stedc!(
        d::Vector{$R},
        e::Vector{$R},
        z::Matrix{$T},
        ::Type{$J};
        compz=('I'),
        lwork=nothing,
        lrwork=nothing,
        liwork=nothing,
        n=length(d),
        ldz=max(1, size(z, 1)),
        library=Colmat.backend_library(),
    )
        sizes=tridiagonal_dc_sizes($T, max(n, 0), compz)
        lw=isnothing(lwork) ? sizes[1] : lwork
        lr=isnothing(lrwork) ? sizes[2] : lrwork
        li=isnothing(liwork) ? sizes[3] : liwork
        work=fill($T(777), max(1, lw)+2)
        rwork=fill($R(777), max(1, lr)+2)
        iwork=fill($J(777), max(1, li)+2)
        info=Ref{$J}(-777)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol)))
        if $T<:Complex
            ccall(
                f,
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{$J},
                    Ptr{$R},
                    Ptr{$R},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$R},
                    Ref{$J},
                    Ptr{$J},
                    Ref{$J},
                    Ref{$J},
                ),
                compz,
                n,
                d,
                e,
                z,
                ldz,
                view(work, 2:(length(work) - 1)),
                lw,
                view(rwork, 2:(length(rwork) - 1)),
                lr,
                view(iwork, 2:(length(iwork) - 1)),
                li,
                info,
            )
        else
            ccall(
                f,
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{$J},
                    Ptr{$R},
                    Ptr{$R},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$J},
                    Ref{$J},
                    Ref{$J},
                ),
                compz,
                n,
                d,
                e,
                z,
                ldz,
                view(work, 2:(length(work) - 1)),
                lw,
                view(iwork, 2:(length(iwork) - 1)),
                li,
                info,
            )
        end
        (; info=info[], work, rwork, iwork)
    end
end
@testset "STEDC real/complex accumulation and workspace" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), n in (0, 1, 2, 7, 40),
        compz in ('N', 'I', 'V')

        R=typeof(real(zero(T)))
        tolerance=R===Float32 ? 6e-5 : 6e-13
        d=R.(2 .+ (1:n) ./ 9)
        e=fill(R(-0.25), max(0, n-1))
        a=Matrix(SymTridiagonal(d, e))
        original=Matrix(qr(randn(T, n, n)).Q)
        if n==1
            original[1]=one(T)
        end
        z=fill(T(777), n+3, max(1, n))
        z[1:n, 1:n]=compz=='V' ? original : fill(T(NaN), n, n)
        before=(copy(d), copy(e), copy(z))
        query=direct_stedc!(d, e, z, J; compz, lwork=-1, lrwork=-1, liwork=-1)
        @test query.info==0
        @test isequal(before, (d, e, z))
        sizes=tridiagonal_dc_sizes(T, n, compz)
        @test real(query.work[2])>=sizes[1]
        @test query.iwork[2]==sizes[3]
        if T<:Complex
            @test query.rwork[2]>=sizes[2]
        end
        @test direct_stedc!(d, e, z, J; compz, lwork=sizes[1]-1).info==-8
        @test direct_stedc!(d, e, z, J; compz, liwork=sizes[3]-1).info==(
            T<:Complex ? -12 : -10
        )
        if T<:Complex
            @test direct_stedc!(d, e, z, J; compz, lrwork=sizes[2]-1).info==-10
        end
        result=direct_stedc!(d, e, z, J; compz)
        @test result.info==0
        @test result.work[[1, end]]==T[777, 777]
        @test result.rwork[[1, end]]==R[777, 777]
        @test result.iwork[[1, end]]==J[777, 777]
        @test d ≈ eigvals(Symmetric(a)) rtol=tolerance
        @test all(==(T(777)), z[(n + 1):end, :])
        if compz=='N'
            @test all(isnan, z[1:n, 1:n])
        else
            vectors=z[1:n, 1:n]
            target=compz=='V' ? original*a*original' : a
            @test target*vectors ≈ vectors*Diagonal(d) atol=tolerance
            @test vectors'*vectors ≈ Matrix{T}(I, n, n) atol=tolerance
        end
    end
end

for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    R=typeof(real(zero(T)))
    suffix=J===Int64 ? "64_" : ""
    symbols=Dict(
        f=>Symbol(prefix*String(f)*"_"*suffix) for f in (:stemr, :stegr, :stevr, :stevx)
    )
    @eval function direct_tridiagonal_range!(
        d::Vector{$R},
        e::Vector{$R},
        z::Matrix{$T},
        ::Type{$J};
        family=:stemr,
        jobz=('V'),
        range=('A'),
        vl=($R(NaN)),
        vu=($R(NaN)),
        il=-777,
        iu=-777,
        abstol=zero($R),
        nzc=size(z, 2),
        tryrac=false,
        lwork=nothing,
        liwork=nothing,
        n=length(d),
        ldz=max(1, size(z, 1)),
        library=Colmat.backend_library(),
    )
        minimum=if family==:stevr
            max(1, 20max(n, 0))
        elseif family==:stevx
            max(1, 5max(n, 0))
        else
            (jobz=='V' ? 18 : 12)*max(n, 0)
        end
        integers=if family==:stevx
            max(1, 5max(n, 0))
        elseif family==:stevr
            max(1, 10max(n, 0))
        else
            (jobz=='V' ? 10 : 8)*max(n, 0)
        end
        lw=isnothing(lwork) ? minimum : lwork
        li=isnothing(liwork) ? integers : liwork
        work=fill($R(777), max(1, lw)+2)
        iwork=fill($J(777), max(1, li)+2)
        metadata=fill($J(777), 2max(1, length(d))+2)
        values=fill($R(777), length(d)+2)
        count=Ref{$J}(-777)
        info=Ref{$J}(-777)
        relative=Ref{$J}(tryrac)
        # STEMR/STEGR define E(N) as scratch. Poison it to catch an input read.
        eb=vcat(e, fill($R(NaN), max(0, length(d)-length(e))))
        f=Libdl.dlsym(Libdl.dlopen(library), $symbols[family])
        wp=view(work, 2:(length(work) - 1))
        ip=view(iwork, 2:(length(iwork) - 1))
        mp=view(metadata, 2:(length(metadata) - 1))
        vp=view(values, 2:(length(values) - 1))
        if family==:stemr
            ccall(
                f,
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{UInt8},
                    Ref{$J},
                    Ptr{$R},
                    Ptr{$R},
                    Ref{$R},
                    Ref{$R},
                    Ref{$J},
                    Ref{$J},
                    Ref{$J},
                    Ptr{$R},
                    Ptr{$T},
                    Ref{$J},
                    Ref{$J},
                    Ptr{$J},
                    Ref{$J},
                    Ptr{$R},
                    Ref{$J},
                    Ptr{$J},
                    Ref{$J},
                    Ref{$J},
                ),
                jobz,
                range,
                n,
                d,
                eb,
                vl,
                vu,
                il,
                iu,
                count,
                vp,
                z,
                ldz,
                nzc,
                mp,
                relative,
                wp,
                lw,
                ip,
                li,
                info,
            )
        elseif family==:stegr||family==:stevr
            ccall(
                f,
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{UInt8},
                    Ref{$J},
                    Ptr{$R},
                    Ptr{$R},
                    Ref{$R},
                    Ref{$R},
                    Ref{$J},
                    Ref{$J},
                    Ref{$R},
                    Ref{$J},
                    Ptr{$R},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$J},
                    Ptr{$R},
                    Ref{$J},
                    Ptr{$J},
                    Ref{$J},
                    Ref{$J},
                ),
                jobz,
                range,
                n,
                d,
                eb,
                vl,
                vu,
                il,
                iu,
                abstol,
                count,
                vp,
                z,
                ldz,
                mp,
                wp,
                lw,
                ip,
                li,
                info,
            )
        elseif family==:stevx
            ccall(
                f,
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{UInt8},
                    Ref{$J},
                    Ptr{$R},
                    Ptr{$R},
                    Ref{$R},
                    Ref{$R},
                    Ref{$J},
                    Ref{$J},
                    Ref{$R},
                    Ref{$J},
                    Ptr{$R},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$R},
                    Ptr{$J},
                    Ptr{$J},
                    Ref{$J},
                ),
                jobz,
                range,
                n,
                d,
                eb,
                vl,
                vu,
                il,
                iu,
                abstol,
                count,
                vp,
                z,
                ldz,
                wp,
                ip,
                mp,
                info,
            )
        end
        (; info=info[], m=count[], values, metadata, work, iwork, relative=relative[], e=eb)
    end
end

@testset "Tridiagonal range selection, support and queries" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        family in (T<:Real ? (:stemr, :stegr, :stevr, :stevx) : (:stemr, :stegr)),
        jobz in ('N', 'V'),
        n in (0, 1, 2, 7, 40),
        range in ('A', 'I', 'V')

        R=typeof(real(zero(T)))
        tolerance=R===Float32 ? 6e-5 : 6e-13
        # Exact repeated eigenvalues exercise interval endpoints and splits.
        d=R.(1 .+ (0:(n - 1)) .÷ 2)
        e=zeros(R, max(0, n-1))
        if n>3
            e[1]=R(0.125)
        end
        a=Matrix(SymTridiagonal(d, e))
        expected=eigvals(Symmetric(a))
        lo, hi=n==0 ? (1, 0) : (max(1, n÷3), max(1, n-n÷3))
        if range=='I'&&n==0&&family in (:stemr, :stegr)
            continue
        end
        vl, vu=R(1), R(3)
        chosen=if range=='A'
            collect(1:n)
        elseif range=='I'
            collect(lo:hi)
        else
            findall(x->vl<x<=vu, expected)
        end
        z=fill(T(777), n+3, max(1, n))
        options=(;
            family,
            jobz,
            range,
            vl=range=='V' ? vl : R(NaN),
            vu=range=='V' ? vu : R(NaN),
            il=range=='I' ? lo : -777,
            iu=range=='I' ? hi : -777,
        )
        if family!=:stevx
            before=(copy(d), copy(e), copy(z))
            query=direct_tridiagonal_range!(d, e, z, J; options..., lwork=-1, liwork=-1)
            @test query.info==0
            @test query.m==-777
            @test isequal(before, (d, e, z))
            @test all(==(J(777)), query.metadata)
        end
        if family==:stemr
            query=direct_tridiagonal_range!(d, e, z, J; options..., nzc=-1)
            @test query.info==0
            @test real(z[1, 1])==(jobz=='V' ? length(chosen) : 0)
            fill!(z, T(777))
            @test direct_tridiagonal_range!(
                d, e, z, J; options..., nzc=(jobz=='V' ? length(chosen) : 0)-1
            ).info==(jobz=='V'&&length(chosen)==0||jobz=='N' ? 0 : -14)
            fill!(z, T(777))
        end
        result=direct_tridiagonal_range!(d, e, z, J; options...)
        @test result.info==0
        @test result.m==length(chosen)
        m=Int(result.m)
        @test result.values[2:(m + 1)] ≈ expected[chosen] rtol=tolerance
        @test result.values[[1, end]]==R[777, 777]
        @test all(==(R(777)), result.values[(m + 2):end])
        @test result.metadata[[1, end]]==J[777, 777]
        @test result.work[[1, end]]==R[777, 777]
        @test result.iwork[[1, end]]==J[777, 777]
        @test all(==(T(777)), z[(n + 1):end, :])
        if jobz=='V'
            vectors=z[1:n, 1:m]
            @test a*vectors ≈ vectors*Diagonal(result.values[2:(m + 1)]) atol=tolerance
            @test vectors'*vectors ≈ Matrix{T}(I, m, m) atol=tolerance
            if family==:stevx
                @test all(iszero, result.metadata[2:(m + 1)])
            elseif family==:stevr&&!(range=='A'||(range=='I'&&lo==1&&hi==n))
                @test all(==(J(777)), result.metadata)
            else
                for column in 1:m
                    first, last=result.metadata[2column:(2column + 1)]
                    @test 1<=first<=last<=n
                    @test all(iszero, vectors[1:(first - 1), column])
                    @test all(iszero, vectors[(last + 1):end, column])
                end
            end
        else
            @test all(==(T(777)), z)
            @test all(==(J(777)), result.metadata)
        end
    end
end

@testset "Tridiagonal expert fallback and LAPACK metadata" begin
    reference=only(
        filter(
            l->l.interface==:ilp64&&occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        family in (T<:Real ? (:stemr, :stevr, :stevx) : (:stemr,)),
        jobz in ('N', 'V'),
        range in ('A', 'I', 'V')

        R=typeof(real(zero(T)))
        tolerance=R===Float32 ? 5e-5 : 5e-13
        d=R[2, 3, 4, 5]
        e=R[0.2, -0.3, 0.1]
        z=fill(T(777), 6, 4)
        rz=copy(z)
        rd=copy(d)
        options=(;
            family,
            jobz,
            range,
            vl=R(2.5),
            vu=R(4.5),
            il=2,
            iu=3,
            tryrac=true,
            abstol=eps(R)/2,
        )
        actual=direct_tridiagonal_range!(d, e, z, J; options...)
        expected=direct_tridiagonal_range!(rd, e, rz, Int64; options..., library=reference)
        @test actual.info==expected.info==0
        @test actual.m==expected.m
        @test actual.values ≈ expected.values rtol=tolerance
        @test actual.relative==expected.relative
        @test d==rd
        @test isequal(actual.e, expected.e)
        @test z ≈ rz rtol=tolerance
        @test actual.metadata==expected.metadata
        @test actual.values[[1, end]]==R[777, 777]
        @test actual.metadata[[1, end]]==J[777, 777]
    end
end

@testset "Tridiagonal range argument indices and integer widths" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        family in (T<:Real ? (:stemr, :stegr, :stevr, :stevx) : (:stemr, :stegr))

        R=typeof(real(zero(T)))
        d=R[2, 3]
        e=R[0.2]
        z=zeros(T, 2, 2)
        @test direct_tridiagonal_range!(d, e, z, J; family, jobz=('?')).info==-1
        @test direct_tridiagonal_range!(d, e, z, J; family, range=('?')).info==-2
        @test direct_tridiagonal_range!(d, e, z, J; family, n=-1).info==-3
        @test direct_tridiagonal_range!(d, e, z, J; family, range=('V'), vl=R(1), vu=R(1)).info==-7
        @test direct_tridiagonal_range!(d, e, z, J; family, range=('I'), il=0, iu=1).info==-8
        @test direct_tridiagonal_range!(d, e, z, J; family, range=('I'), il=1, iu=3).info==-9
        @test direct_tridiagonal_range!(d, e, z, J; family, ldz=1).info==(
            family in (:stemr, :stegr) ? -13 : -14
        )
        if family!=:stevx
            @test direct_tridiagonal_range!(d, e, z, J; family, lwork=0).info==-17
            @test direct_tridiagonal_range!(d, e, z, J; family, liwork=0).info==-19
        end
        if J===Int64
            @test direct_tridiagonal_range!(d, e, z, J; family, n=(-(Int64(1)<<32))).info==-3
        end
    end
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        R=typeof(real(zero(T)))
        d=R[2, 3]
        e=R[0.2]
        z=zeros(T, 2, 2)
        @test direct_stedc!(d, e, z, J; compz=('?')).info==-1
        @test direct_stedc!(d, e, z, J; n=-1).info==-2
        @test direct_stedc!(d, e, z, J; ldz=1).info==-6
        @test direct_pteqr!(d, e, z, J; compz=('?'))==-1
        @test direct_pteqr!(d, e, z, J; n=-1)==-2
        @test direct_pteqr!(d, e, z, J; ldz=1)==-6
    end
end

@testset "Native divide-and-conquer and all tridiagonal LBT entries" begin
    for R in (Float32, Float64), J in (Int32, Int64)
        n=160
        d=R.(2 .+ (1:n) ./ 9)
        e=fill(R(-0.25), n-1)
        a=SymTridiagonal(d, e)
        z=zeros(R, n, n)
        tolerance=R===Float32 ? 3e-4 : 3e-12
        result=direct_stedc!(copy(d), copy(e), z, J)
        @test result.info==0
        @test z'*z ≈ Matrix{R}(I, n, n) atol=tolerance
        @test a*z ≈ z*Diagonal(eigvals(a)) atol=tolerance
        result=direct_tridiagonal_range!(copy(d), copy(e), z, J; range=('I'), il=11, iu=27)
        @test result.info==0
        @test result.m==17
        @test a*z[:, 1:17] ≈ z[:, 1:17]*Diagonal(result.values[2:18]) atol=tolerance
    end
    cfg=BLAS.get_config()
    for prefix in ("s", "d", "c", "z"),
        family in
        (:pteqr, :stedc, :stegr, :stemr, :steqr, :sterf, :stev, :stevd, :stevr, :stevx),
        interface in (:lp64, :ilp64)

        if prefix in ("c", "z")&&family in (:sterf, :stev, :stevd, :stevr, :stevx)
            continue
        end
        symbol=prefix*String(family)*"_"
        library=BLAS.lbt_find_backing_library(symbol, interface)
        @test library!==nothing
        @test realpath(library.libname)==realpath(Colmat.backend_library())
    end
    for R in (Float32, Float64)
        d=R[2, 3, 4]
        e=R[0.2, 0.3]
        a=SymTridiagonal(d, e)
        w, z=LAPACK.stegr!('V', copy(d), copy(e))
        @test a*z ≈ z*Diagonal(w)
        w, z=LAPACK.stev!('V', copy(d), copy(e))
        @test a*z ≈ z*Diagonal(w)
    end
end

@testset "Isolated tridiagonal fallback configuration" begin
    worker=joinpath(@__DIR__, "tridiagonal_fallback_worker.jl")
    project=dirname(Base.active_project())
    @test success(`$(Base.julia_cmd()) --startup-file=no --project=$project $worker`)
end
