using Test, Colmat, LinearAlgebra, Libdl
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
    R=typeof(real(zero(T)))
    base=[Cint, Cchar, Cchar, J, Ptr{T}, J, Ptr{R}]
    high_signature=Expr(:tuple, base...)
    work_signature=Expr(:tuple, base..., Ptr{T}, J, (T<:Complex ? [Ptr{R}] : [])...)
    extra=T<:Complex ? [:rwork] : []
    @eval function lapacke288_eigen_call(
        ::Type{$T},
        ::Type{$J},
        p,
        high,
        layout,
        job,
        uplo,
        n,
        a,
        ld,
        w,
        work,
        lwork,
        rwork,
        name,
        route,
    )
        lapacke288_selected_entry(name, $J, route) do
            if high
                ccall(p, $J, $high_signature, layout, job, uplo, n, a, ld, w)
            else
                ccall(
                    p,
                    $J,
                    $work_signature,
                    layout,
                    job,
                    uplo,
                    n,
                    a,
                    ld,
                    w,
                    work,
                    lwork,
                    $(extra...),
                )
            end
        end
    end
end

for T in (Float32, Float64, ComplexF32, ComplexF64)
    R=typeof(real(zero(T)))
    signature=Expr(
        :tuple,
        Ref{UInt8},
        Ref{UInt8},
        Ref{Int64},
        Ptr{T},
        Ref{Int64},
        Ptr{R},
        Ptr{T},
        Ref{Int64},
        (T<:Complex ? [Ptr{R}] : [])...,
        Ref{Int64},
    )
    extra=T<:Complex ? [:rwork] : []
    @eval function lapacke288_eigen_reference(::Type{$T}, p, original, uplo)
        n=size(original, 1)
        a=copy(original)
        w=zeros($R, n)
        work=zeros($T, 1)
        rwork=zeros($R, max(1, 3n-2))
        info=Ref{Int64}(0)
        ccall(
            p,
            Cvoid,
            $signature,
            UInt8('N'),
            UInt8(uplo),
            n,
            a,
            max(1, n),
            w,
            work,
            -1,
            $(extra...),
            info,
        )
        @test info[]==0
        work=zeros($T, Int(real(work[1])))
        ccall(
            p,
            Cvoid,
            $signature,
            UInt8('N'),
            UInt8(uplo),
            n,
            a,
            max(1, n),
            w,
            work,
            length(work),
            $(extra...),
            info,
        )
        @test info[]==0
        w
    end
end

function lapacke288_eigen(::Type{T}, ::Type{J}) where {T,J}
    R=typeof(real(zero(T)))
    prefix=Dict(Float32=>"s", Float64=>"d", ComplexF32=>"c", ComplexF64=>"z")[T]
    family=T<:Complex ? "heev" : "syev"
    suffix=J===Int64 ? "64_" : ""
    native=Libdl.dlopen(Colmat.backend_library())
    trampoline=Libdl.dlopen(BLAS.libblastrampoline)
    reference=only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(l.libname)),
            BLAS.get_config().loaded_libs,
        ),
    )
    refp=Libdl.dlsym(Libdl.dlopen(reference.libname), prefix*family*"_64_")
    @test refp!=Libdl.dlsym(native, prefix*family*"_64_")
    for n in (0, 1, 3), uplo in ('L', 'U', 'l', 'u')
        original=T[
            i==j ? complex(4+i, 0) : complex(0.2(i+j), T<:Complex ? 0.3(i-j) : 0) for
            i in 1:n, j in 1:n
        ]
        expected=lapacke288_eigen_reference(T, refp, original, uppercase(uplo))
        for high in (false, true),
            route in (:dll, :lbt_trampoline),
            layout in (101, 102),
            job in ('N', 'V', 'n', 'v'),
            imaginary_nan in (T<:Complex && !high ? (false, true) : (false,))

            name="LAPACKE_"*prefix*family*(high ? "" : "_work")
            @test BLAS.lbt_get_forward(name, J===Int64 ? :ilp64 : :lp64)==Libdl.dlsym(
                native, name*suffix
            )
            p=Libdl.dlsym(route===:dll ? native : trampoline, name*suffix)
            ld=n+2
            offset(i, j) = layout==101 ? (i-1)*ld+j : i+(j-1)*ld
            a=fill(T(NaN), ld*(n+2))
            mask=falses(length(a))
            for i in 1:n, j in 1:n
                selected=uppercase(uplo)=='L' ? i>=j : i<=j
                if selected
                    a[offset(i, j)]=if imaginary_nan && i==j
                        T(complex(real(original[i, j]), NaN))
                    else
                        original[i, j]
                    end
                end
                mask[offset(i, j)]=uppercase(job)=='V'||selected
            end
            before=copy(a)
            w=fill(R(-99), n+2)
            work=zeros(T, 1)
            rwork=zeros(R, max(1, 3n-2))
            call(work, lw) = lapacke288_eigen_call(
                T,
                J,
                p,
                high,
                layout,
                job,
                uplo,
                n,
                a,
                ld,
                w,
                work,
                lw,
                rwork,
                name,
                route,
            )
            if !high
                @test call(work, -1)==0
                @test isequal(a, before)
                @test all(==(R(-99)), w)
                work=zeros(T, Int(real(work[1])))
                @test call(work, length(work)-1)==-9
                @test isequal(a, before)
            end
            @test call(work, length(work))==0
            tol=R===Float32 ? 5e-5 : 2e-12
            @test w[1:n]≈expected rtol=tol atol=tol
            @test issorted(w[1:n])
            @test w[(n + 1):end]==fill(R(-99), 2)
            @test isequal(a[.!mask], before[.!mask])
            if uppercase(job)=='V'
                vectors=T[a[offset(i, j)] for i in 1:n, j in 1:n]
                @test original*vectors≈vectors*Diagonal(w[1:n]) rtol=tol atol=tol
                @test vectors'*vectors≈Matrix{T}(I, n, n) rtol=tol atol=tol
            end
        end
    end
end

@testset "LAPACKE SYEV/HEEV: independent reference and selected storage" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        lapacke288_eigen(T, J)
    end
end
