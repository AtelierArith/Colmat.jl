using Test, Colmat, LinearAlgebra, Libdl
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
    R=real(T)
    extra=T<:Real ? [:wi] : []
    base=Any[Cint, Cchar, Cchar, J, Ptr{T}, J, Ptr{T}]
    T<:Real && push!(base, Ptr{R})
    append!(base, [Ptr{T}, J, Ptr{T}, J])
    highsig=Expr(:tuple, base...)
    worksig=Expr(:tuple, base..., Ptr{T}, J, (T<:Complex ? [Ptr{R}] : [])...)
    realwork=T<:Complex ? [:rwork] : []
    @eval function lapacke288_geev_call(
        ::Type{$T},
        ::Type{$J},
        p,
        high,
        layout,
        jl,
        jr,
        n,
        a,
        ld,
        w,
        wi,
        vl,
        vr,
        work,
        lwork,
        rwork,
        name,
        route,
    )
        lapacke288_selected_entry(name, $J, route) do
            if high
                ccall(
                    p,
                    $J,
                    $highsig,
                    layout,
                    jl,
                    jr,
                    n,
                    a,
                    ld,
                    w,
                    $(extra...),
                    vl,
                    ld,
                    vr,
                    ld,
                )
            else
                ccall(
                    p,
                    $J,
                    $worksig,
                    layout,
                    jl,
                    jr,
                    n,
                    a,
                    ld,
                    w,
                    $(extra...),
                    vl,
                    ld,
                    vr,
                    ld,
                    work,
                    lwork,
                    $(realwork...),
                )
            end
        end
    end
end

for T in (Float32, Float64, ComplexF32, ComplexF64)
    R=real(T)
    signature=Expr(
        :tuple,
        Ref{UInt8},
        Ref{UInt8},
        Ref{Int64},
        Ptr{T},
        Ref{Int64},
        Ptr{T},
        (T<:Real ? [Ptr{R}] : [])...,
        Ptr{T},
        Ref{Int64},
        Ptr{T},
        Ref{Int64},
        Ptr{T},
        Ref{Int64},
        (T<:Complex ? [Ptr{R}] : [])...,
        Ptr{Int64},
        Clong,
        Clong,
    )
    wiargs=T<:Real ? [:wi] : []
    rwargs=T<:Complex ? [:rwork] : []
    @eval function lapacke288_geev_reference(::Type{$T}, p, original)
        n=size(original, 1)
        ld=max(1, n)
        a=zeros($T, ld, max(1, n))
        a[1:n, 1:n].=original
        w=zeros($T, max(1, n))
        wi=zeros($R, max(1, n))
        rwork=zeros($R, max(1, 2n))
        vl=zeros($T, 1)
        vr=zeros($T, 1)
        work=zeros($T, 1)
        info=Ref{Int64}(-99)
        function invoke(lwork)
            ccall(
                p,
                Cvoid,
                $signature,
                'N',
                'N',
                n,
                a,
                ld,
                w,
                $(wiargs...),
                vl,
                1,
                vr,
                1,
                work,
                lwork,
                $(rwargs...),
                info,
                1,
                1,
            )
        end
        invoke(-1)
        @test info[]==0
        resize!(work, Int(real(work[1])))
        invoke(length(work))
        @test info[]==0
        return $T<:Real ? complex.(w[1:n], wi[1:n]) : w[1:n]
    end
end

@testset "LAPACKE general eigenvalues and left/right vectors" begin
    native=Libdl.dlopen(Colmat.backend_library())
    trampoline=Libdl.dlopen(BLAS.libblastrampoline)
    reference=Libdl.dlopen(
        only(
            filter(
                l->l.interface==:ilp64 && occursin("openblas", lowercase(l.libname)),
                BLAS.get_config().loaded_libs,
            ),
        ).libname,
    )
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64),
        n in (0, 1, 3, 5)

        R=real(T)
        original = if n==0
            zeros(T, 0, 0)
        elseif n==1
            fill(T(7), 1, 1)
        elseif n==3
            T[1 -2 1; 2 1 2; 0 0 5]
        else
            T[sin(i+2j)+(i==j ? i : 0) for i in 1:n, j in 1:n]
        end
        T<:Complex && n>1 && (original .+= T(im)*I(n))
        refp=Libdl.dlsym(reference, prefix*"geev_64_")
        @test refp!=Libdl.dlsym(native, prefix*"geev_64_")
        expected=lapacke288_geev_reference(T, refp, original)
        tol=R(100)*eps(R)
        for route in (:dll, :lbt_trampoline),
            layout in (101, 102), jl in ('N', 'V'), jr in ('N', 'V'),
            high in (false, true)

            name="LAPACKE_"*prefix*"geev"*(high ? "" : "_work")
            suffix=J==Int64 ? "64_" : ""
            @test BLAS.lbt_get_forward(name, J==Int64 ? :ilp64 : :lp64)==Libdl.dlsym(
                native, name*suffix
            )
            p=Libdl.dlsym(route==:dll ? native : trampoline, name*suffix)
            ld=n+2
            offset(i, j) = layout==101 ? (i-1)*ld+j : i+(j-1)*ld
            a=fill(T(777), ld*max(1, n))
            vl=copy(a)
            vr=copy(a)
            for i in 1:n, j in 1:n
                a[offset(i, j)]=original[i, j]
            end
            saved=copy(a)
            w=fill(T(777), n+1)
            wi=fill(R(777), n+1)
            work=zeros(T, max(1, 4n))
            rwork=zeros(R, max(1, 2n))
            call(high, lw) = lapacke288_geev_call(
                T,
                J,
                p,
                high,
                layout,
                jl,
                jr,
                n,
                a,
                ld,
                w,
                wi,
                vl,
                vr,
                work,
                lw,
                rwork,
                name,
                route,
            )
            if !high
                @test call(false, -1)==0
                @test a==saved && all(==(T(777)), vl) && all(==(T(777)), vr)
                @test all(==(T(777)), w)
            end
            minimum=max(1, (
                if T<:Complex
                    2
                elseif jl=='V'||jr=='V'
                    4
                else
                    3
                end
            )*n)
            @test call(high, minimum)==0
            values=T<:Complex ? Complex{R}.(w[1:n]) : complex.(w[1:n], wi[1:n])
            unmatched=collect(expected)
            for value in values
                index=findfirst(e->abs(value-e)<=tol*max(1, abs(e)), unmatched)
                @test index!==nothing
                index===nothing || deleteat!(unmatched, index)
            end
            @test isempty(unmatched)
            @test w[end]==T(777) && wi[end]==R(777)
            for (job, data, left) in ((jl, vl, true), (jr, vr, false))
                if job=='N'
                    @test all(==(T(777)), data)
                    continue
                end
                vectors=Complex{R}[data[offset(i, j)] for i in 1:n, j in 1:n]
                if T<:Real
                    for j in 1:n
                        if wi[j]>0
                            vectors[:, j]=complex.(
                                [data[offset(i, j)] for i in 1:n],
                                [data[offset(i, j+1)] for i in 1:n],
                            )
                        elseif wi[j]<0
                            vectors[:, j]=conj.(vectors[:, j - 1])
                        end
                    end
                end
                @test (left ? original' : original)*vectors≈vectors*Diagonal(
                    left ? conj.(values) : values
                ) atol=tol*20
                for j in 1:n
                    @test norm(vectors[:, j])≈1 atol=tol
                end
            end
            for k in eachindex(a)
                if k>ld*n || (k-1)%ld>=n
                    @test a[k]==saved[k] && vl[k]==T(777) && vr[k]==T(777)
                end
            end
        end
    end
end
