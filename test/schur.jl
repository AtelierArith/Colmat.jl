using Test, LinearAlgebra, Random, Libdl, Colmat

for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    symbol=Symbol(p*"hseqr_"*(J===Int64 ? "64_" : ""))
    if T<:Real
        @eval function direct_schur!(
            h::Matrix{$T},
            z::Matrix{$T},
            ::Type{$J};
            lo=1,
            hi=size(h, 2),
            comp=('I'),
            query=false,
            library=Colmat.backend_library(),
        )
            n=size(h, 2)
            wr=zeros($T, n)
            wi=zeros($T, n)
            work=zeros($T, max(1, n))
            info=Ref{$J}(0)
            f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol)))
            ccall(
                f,
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{UInt8},
                    Ref{$J},
                    Ref{$J},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$T},
                    Ptr{$T},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ref{$J},
                ),
                'S',
                comp,
                n,
                lo,
                hi,
                h,
                max(1, size(h, 1)),
                wr,
                wi,
                z,
                max(1, size(z, 1)),
                work,
                query ? -1 : length(work),
                info,
            )
            return complex.(wr, wi), info[], work
        end
    else
        @eval function direct_schur!(
            h::Matrix{$T},
            z::Matrix{$T},
            ::Type{$J};
            lo=1,
            hi=size(h, 2),
            comp=('I'),
            query=false,
            library=Colmat.backend_library(),
        )
            n=size(h, 2)
            w=zeros($T, n)
            work=zeros($T, max(1, n))
            info=Ref{$J}(0)
            f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol)))
            ccall(
                f,
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{UInt8},
                    Ref{$J},
                    Ref{$J},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$T},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ref{$J},
                ),
                'S',
                comp,
                n,
                lo,
                hi,
                h,
                max(1, size(h, 1)),
                w,
                z,
                max(1, size(z, 1)),
                work,
                query ? -1 : length(work),
                info,
            )
            return w, info[], work
        end
    end
end

@testset "HSEQR direct ABI, active ranges and vector accumulation" begin
    Random.seed!(300)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        (n, lo, hi) in ((7, 1, 7), (7, 2, 6), (1, 1, 1), (0, 1, 0))

        h=triu(randn(T, n, n), -1)
        lo>1 && (h[lo, lo - 1]=0)
        hi<n && (h[hi + 1, hi]=0)
        original=copy(h)
        z=Matrix{T}(I, n, n)
        values, info, _=direct_schur!(h, z, J; lo, hi)
        @test info==0
        @test z*h*z' ≈ original
        @test z'*z ≈ Matrix{T}(I, n, n)
        @test all(isfinite, values)
        @test norm(tril(h, T<:Real ? -2 : -1))==0
        q=Matrix(qr(randn(T, n, n)).Q)
        z=copy(q)
        h=copy(original)
        _, info, _=direct_schur!(h, z, J; lo, hi, comp=('V'))
        @test info==0
        @test z*h*z' ≈ q*original*q'
        before=copy(h)
        oldz=copy(z)
        _, info, work=direct_schur!(h, z, J; lo, hi, query=true)
        @test info==0
        @test real(work[1])>=max(1, n)
        @test h==before && z==oldz
    end
end

const schur_callback_calls=Ref(0)
for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    R=typeof(real(zero(T)))
    name=Symbol("select_schur_", p, "_", J)
    callback=Symbol(name, "_pointer")
    symbol=Symbol(p*"gees_"*(J===Int64 ? "64_" : ""))
    if T<:Real
        @eval begin
            function $name(re::Ptr{$T}, im::Ptr{$T})::$J
                schur_callback_calls[]+=1
                return unsafe_load(re)<0 ? 1 : 0
            end
        end
        @eval begin
            const $callback=@cfunction($(name), $J, (Ptr{$T}, Ptr{$T}))
            function direct_gees!(
                a::Matrix{$T},
                vs::Matrix{$T},
                ::Type{$J};
                sorting=true,
                query=false,
                job=('V'),
                use_callback=true,
                selection=($callback),
                library=Colmat.backend_library(),
            )
                n=size(a, 2)
                wr=zeros($T, n)
                wi=zeros($T, n)
                work=zeros($T, max(1, 3n))
                bwork=zeros($J, n)
                sdim=Ref{$J}(-1)
                info=Ref{$J}(0)
                f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol)))
                ccall(
                    f,
                    Cvoid,
                    (
                        Ref{UInt8},
                        Ref{UInt8},
                        Ptr{Cvoid},
                        Ref{$J},
                        Ptr{$T},
                        Ref{$J},
                        Ref{$J},
                        Ptr{$T},
                        Ptr{$T},
                        Ptr{$T},
                        Ref{$J},
                        Ptr{$T},
                        Ref{$J},
                        Ptr{$J},
                        Ref{$J},
                    ),
                    job,
                    sorting ? 'S' : 'N',
                    use_callback ? selection : C_NULL,
                    n,
                    a,
                    max(1, size(a, 1)),
                    sdim,
                    wr,
                    wi,
                    vs,
                    max(1, size(vs, 1)),
                    work,
                    query ? -1 : length(work),
                    bwork,
                    info,
                )
                return complex.(wr, wi), sdim[], info[], work
            end
        end
    else
        @eval begin
            function $name(w::Ptr{$T})::$J
                schur_callback_calls[]+=1
                return real(unsafe_load(w))<0 ? 1 : 0
            end
        end
        @eval begin
            const $callback=@cfunction($(name), $J, (Ptr{$T},))
            function direct_gees!(
                a::Matrix{$T},
                vs::Matrix{$T},
                ::Type{$J};
                sorting=true,
                query=false,
                job=('V'),
                use_callback=true,
                selection=($callback),
                library=Colmat.backend_library(),
            )
                n=size(a, 2)
                w=zeros($T, n)
                work=zeros($T, max(1, 2n))
                rw=zeros($R, n)
                bwork=zeros($J, n)
                sdim=Ref{$J}(-1)
                info=Ref{$J}(0)
                f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol)))
                ccall(
                    f,
                    Cvoid,
                    (
                        Ref{UInt8},
                        Ref{UInt8},
                        Ptr{Cvoid},
                        Ref{$J},
                        Ptr{$T},
                        Ref{$J},
                        Ref{$J},
                        Ptr{$T},
                        Ptr{$T},
                        Ref{$J},
                        Ptr{$T},
                        Ref{$J},
                        Ptr{$R},
                        Ptr{$J},
                        Ref{$J},
                    ),
                    job,
                    sorting ? 'S' : 'N',
                    use_callback ? selection : C_NULL,
                    n,
                    a,
                    max(1, size(a, 1)),
                    sdim,
                    w,
                    vs,
                    max(1, size(vs, 1)),
                    work,
                    query ? -1 : length(work),
                    rw,
                    bwork,
                    info,
                )
                return w, sdim[], info[], work
            end
        end
    end
end

@testset "GEES callback ABI and selected Schur spectrum" begin
    Random.seed!(3001)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), n in (0, 1, 6),
        sorting in (false, true)

        original=randn(T, n, n)
        a=copy(original)
        vs=zeros(T, n, n)
        calls=schur_callback_calls[]
        w, count, info, _=direct_gees!(a, vs, J; sorting, use_callback=sorting)
        @test info==0
        @test vs*a*vs' ≈ original
        @test vs'*vs ≈ Matrix{T}(I, n, n)
        @test count==(sorting ? sum(real.(w) .< 0) : 0)
        if sorting
            @test all(real.(w[1:count]) .< 0)
            @test all(real.(w[(count + 1):end]) .>= 0)
        else
            @test schur_callback_calls[]==calls
        end
        before=copy(a)
        oldvs=copy(vs)
        calls=schur_callback_calls[]
        _, _, info, work=direct_gees!(a, vs, J; sorting, query=true, use_callback=false)
        @test info==0
        @test schur_callback_calls[]==calls
        @test a==before && vs==oldvs
        @test real(work[1])>=max(1, (T<:Real ? 3 : 2)*n)
    end
end

for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64),
    family in (:lahqr, :laqr0, :laqr4)

    symbol=Symbol(p*string(family)*"_"*(J===Int64 ? "64_" : ""))
    eigtypes=T<:Real ? :((Ptr{$T}, Ptr{$T})) : :((Ptr{$T},))
    eigargs=T<:Real ? [:wr, :wi] : [:wr]
    worktypes=family===:lahqr ? [] : [:(Ptr{$T}), :(Ref{$J})]
    workargs=family===:lahqr ? [] : [:work, :(query ? -1 : length(work))]
    ctypes=Expr(
        :tuple,
        :(Ref{$J}),
        :(Ref{$J}),
        :(Ref{$J}),
        :(Ref{$J}),
        :(Ref{$J}),
        :(Ptr{$T}),
        :(Ref{$J}),
        eigtypes.args...,
        :(Ref{$J}),
        :(Ref{$J}),
        :(Ptr{$T}),
        :(Ref{$J}),
        worktypes...,
        :(Ref{$J}),
    )
    @eval function direct_aux_schur!(
        h::Matrix{$T},
        z::Matrix{$T},
        ::Type{$J},
        ::Val{$(QuoteNode(family))};
        n=size(h, 2),
        lo=1,
        hi=n,
        zlo=1,
        zhi=n,
        wantz=true,
        query=false,
        library=Colmat.backend_library(),
    )
        wr=zeros($T, n)
        wi=zeros($T, n)
        work=zeros($T, max(1, n))
        info=Ref{$J}(0)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol)))
        ccall(
            f,
            Cvoid,
            $ctypes,
            1,
            wantz,
            n,
            lo,
            hi,
            h,
            max(1, size(h, 1)),
            $(eigargs...),
            zlo,
            zhi,
            z,
            max(1, size(z, 1)),
            $(workargs...),
            info,
        )
        return info[], work
    end
end

@testset "Schur auxiliary entry points and row-limited Z" begin
    Random.seed!(3002)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        family in (:lahqr, :laqr0, :laqr4)

        n=6
        original=triu(randn(T, n, n), -1)
        original[2, 1]=0
        original[6, 5]=0
        reference=copy(original)
        q=Matrix{T}(I, n, n)
        _, info, _=direct_schur!(reference, q, J; lo=2, hi=5)
        @test info==0
        h=fill(T(987), n+2, n)
        h[1:n, :]=original
        # The lower triangle is not part of the Hessenberg input.
        for j in 1:n, i in (j + 2):n
            h[i, j]=T(NaN)
        end
        z=randn(T, n+2, n)
        oldz=copy(z)
        info, _=direct_aux_schur!(h, z, J, Val(family); lo=2, hi=5, zlo=2, zhi=5)
        @test info==0
        @test h[1:n, :] ≈ reference
        @test z[2:5, :] ≈ oldz[2:5, :]*q
        @test z[[1, 6, 7, 8], :]==oldz[[1, 6, 7, 8], :]
        @test all(==(T(987)), h[(n + 1):end, :])
        if family!==:lahqr
            before=copy(h)
            oldz=copy(z)
            info, work=direct_aux_schur!(h, z, J, Val(family); query=true)
            @test info==0 && real(work[1])>=n
            @test h==before && z==oldz
        end
    end
end

@testset "GEES real conjugate pairs, scaling and Julia wrappers" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        R=typeof(real(zero(T)))
        base=T[2 1 0 0; 0 -1 -3 1; 0 3 -1 1; 0 0 0 -4]
        for scale in (one(R), sqrt(floatmin(R)), sqrt(floatmax(R))/100)
            original=base*scale
            a=copy(original)
            vs=zeros(T, 4, 4)
            w, count, info, _=direct_gees!(a, vs, J)
            @test info==0
            @test count==3
            @test all(real.(w[1:count]) .< 0)
            @test vs*(a/scale)*vs' ≈ base
            @test vs'*vs ≈ Matrix{T}(I, 4, 4)
        end
        a, vs, w=LAPACK.gees!('V', copy(base))
        @test vs*a*vs' ≈ base
        h, z, w=LAPACK.hseqr!(copy(base))
        @test z*h*z' ≈ base
    end
end

for T in (Float32, Float64), J in (Int32, Int64)
    name=Symbol("select_positive_imag_", T, "_", J)
    callback=Symbol(name, "_pointer")
    @eval function $name(re::Ptr{$T}, im::Ptr{$T})::$J
        return unsafe_load(im)>0 ? 1 : 0
    end
    @eval const $callback=@cfunction($(name), $J, (Ptr{$T}, Ptr{$T}))
    @eval function positive_imag_callback(::Type{$T}, ::Type{$J})
        return $callback
    end
end
@testset "Schur argument checks, unused outputs and conjugate selection" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        base=T[2 1 0 0; 0 -1 -3 1; 0 3 -1 1; 0 0 0 -4]
        vs=fill(T(NaN), 4, 4)
        a=copy(base)
        w, sdim, info, _=direct_gees!(
            a, vs, J; sorting=false, job=('N'), use_callback=false
        )
        @test info==0 && sdim==0
        @test all(isnan, vs)
        @test sum(real.(w) .< 0)==3
        a=copy(base)
        vs=zeros(T, 4, 4)
        calls=schur_callback_calls[]
        _, _, info, _=direct_gees!(a, vs, J; job=('?'))
        @test info == -1
        @test a==base && schur_callback_calls[]==calls
        h=copy(base)
        z=fill(T(NaN), 4, 4)
        _, info, _=direct_schur!(h, z, J; comp=('N'))
        @test info==0 && all(isnan, z)
        _, info, _=direct_schur!(h, z, J; lo=0)
        @test info == -4
        for family in (:lahqr, :laqr0, :laqr4)
            h=copy(base)
            z=fill(T(NaN), 4, 4)
            info, _=direct_aux_schur!(h, z, J, Val(family); wantz=false, zlo=0, zhi=0)
            @test info==0 && all(isnan, z)
            info, _=direct_aux_schur!(h, z, J, Val(family); lo=0)
            @test info == -4
        end
        for family in ("gees", "hseqr", "lahqr", "laqr0", "laqr4")
            p=if T===Float32
                "s"
            elseif T===Float64
                "d"
            elseif T===ComplexF32
                "c"
            else
                "z"
            end
            backing=BLAS.lbt_find_backing_library(p*family*"_", J===Int32 ? :lp64 : :ilp64)
            @test backing!==nothing &&
                realpath(backing.libname)==realpath(Colmat.backend_library())
        end
        if T<:Real
            a=copy(base)
            vs=zeros(T, 4, 4)
            w, count, info, _=direct_gees!(a, vs, J; selection=positive_imag_callback(T, J))
            @test info==0 && count==2
            @test imag(w[1])>0 && imag(w[2])<0
            @test vs*a*vs' ≈ base
        end
        # Public native API reports no convergence but does not expose a partial index.
        h=copy(base)
        h[1, 1]=T(NaN)
        z=zeros(T, 4, 4)
        _, info, _=direct_schur!(h, z, J)
        @test info==4
    end
end

@testset "GEES independent OpenBLAS spectrum" begin
    library=BLAS.lbt_find_backing_library("dgges_", :ilp64).libname
    Random.seed!(3003)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        original=randn(T, 8, 8)
        ours=copy(original)
        q=zeros(T, 8, 8)
        w, _, info, _=direct_gees!(ours, q, Int64; sorting=false, use_callback=false)
        reference=copy(original)
        z=zeros(T, 8, 8)
        expected, _, refinfo, _=direct_gees!(
            reference, z, Int64; sorting=false, use_callback=false, library
        )
        @test info==0 && refinfo==0
        pool=collect(expected)
        R=typeof(real(zero(T)))
        for value in w
            i=argmin(abs.(pool .- value))
            @test abs(value-pool[i])<=500eps(R)*max(1, abs(value))
            deleteat!(pool, i)
        end
    end
end
