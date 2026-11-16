using Test, Colmat, LinearAlgebra, Libdl
isdefined(@__MODULE__, :lapacke288_qr_form_q)||include("lapacke_qr.jl")
for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
    R=real(T)
    @eval function lapacke288_pivot_call(
        ::Type{$T},
        ::Type{$J},
        p,
        blocked,
        high,
        layout,
        m,
        n,
        a,
        la,
        pivot,
        tau,
        work,
        lw,
        rwork,
        name,
        route,
    )
        lapacke288_selected_entry(name, $J, route) do
            if high
                ccall(
                    p,
                    $J,
                    (Cint, $J, $J, Ptr{$T}, $J, Ptr{$J}, Ptr{$T}),
                    layout,
                    m,
                    n,
                    a,
                    la,
                    pivot,
                    tau,
                )
            elseif $T<:Complex
                if blocked
                    ccall(
                        p,
                        $J,
                        (Cint, $J, $J, Ptr{$T}, $J, Ptr{$J}, Ptr{$T}, Ptr{$T}, $J, Ptr{$R}),
                        layout,
                        m,
                        n,
                        a,
                        la,
                        pivot,
                        tau,
                        work,
                        lw,
                        rwork,
                    )
                else
                    ccall(
                        p,
                        $J,
                        (Cint, $J, $J, Ptr{$T}, $J, Ptr{$J}, Ptr{$T}, Ptr{$T}, Ptr{$R}),
                        layout,
                        m,
                        n,
                        a,
                        la,
                        pivot,
                        tau,
                        work,
                        rwork,
                    )
                end
            elseif blocked
                ccall(
                    p,
                    $J,
                    (Cint, $J, $J, Ptr{$T}, $J, Ptr{$J}, Ptr{$T}, Ptr{$T}, $J),
                    layout,
                    m,
                    n,
                    a,
                    la,
                    pivot,
                    tau,
                    work,
                    lw,
                )
            else
                ccall(
                    p,
                    $J,
                    (Cint, $J, $J, Ptr{$T}, $J, Ptr{$J}, Ptr{$T}, Ptr{$T}),
                    layout,
                    m,
                    n,
                    a,
                    la,
                    pivot,
                    tau,
                    work,
                )
            end
        end
    end
end
for T in (Float32, Float64, ComplexF32, ComplexF64)
    R=real(T)
    @eval function lapacke288_pivot_reference(::Type{$T}, p, blocked, original, flags)
        m, n=size(original)
        k=min(m, n)
        la=m+2
        a=fill($T(777), la, n)
        a[1:m, :].=original
        pivot=fill(Int64(-777), n+2)
        pivot[2:(n + 1)].=flags
        tau=fill($T(778), k+2)
        lw=max(1, 3n+1)
        work=fill($T(779), lw+2)
        rwork=fill($R(780), max(1, 2n)+2)
        info=fill(Int64(-777), 3)
        if $T<:Complex
            if blocked
                ccall(
                    p,
                    Cvoid,
                    (
                        Ref{Int64},
                        Ref{Int64},
                        Ptr{$T},
                        Ref{Int64},
                        Ptr{Int64},
                        Ptr{$T},
                        Ptr{$T},
                        Ref{Int64},
                        Ptr{$R},
                        Ptr{Int64},
                    ),
                    m,
                    n,
                    a,
                    la,
                    view(pivot, 2:(n + 1)),
                    view(tau, 2:(k + 1)),
                    view(work, 2:(lw + 1)),
                    lw,
                    view(rwork, 2:(length(rwork) - 1)),
                    view(info, 2:2),
                )
            else
                ccall(
                    p,
                    Cvoid,
                    (
                        Ref{Int64},
                        Ref{Int64},
                        Ptr{$T},
                        Ref{Int64},
                        Ptr{Int64},
                        Ptr{$T},
                        Ptr{$T},
                        Ptr{$R},
                        Ptr{Int64},
                    ),
                    m,
                    n,
                    a,
                    la,
                    view(pivot, 2:(n + 1)),
                    view(tau, 2:(k + 1)),
                    view(work, 2:(lw + 1)),
                    view(rwork, 2:(length(rwork) - 1)),
                    view(info, 2:2),
                )
            end
        elseif blocked
            ccall(
                p,
                Cvoid,
                (
                    Ref{Int64},
                    Ref{Int64},
                    Ptr{$T},
                    Ref{Int64},
                    Ptr{Int64},
                    Ptr{$T},
                    Ptr{$T},
                    Ref{Int64},
                    Ptr{Int64},
                ),
                m,
                n,
                a,
                la,
                view(pivot, 2:(n + 1)),
                view(tau, 2:(k + 1)),
                view(work, 2:(lw + 1)),
                lw,
                view(info, 2:2),
            )
        else
            ccall(
                p,
                Cvoid,
                (
                    Ref{Int64},
                    Ref{Int64},
                    Ptr{$T},
                    Ref{Int64},
                    Ptr{Int64},
                    Ptr{$T},
                    Ptr{$T},
                    Ptr{Int64},
                ),
                m,
                n,
                a,
                la,
                view(pivot, 2:(n + 1)),
                view(tau, 2:(k + 1)),
                view(work, 2:(lw + 1)),
                view(info, 2:2),
            )
        end
        @test info==[-777, 0, -777]
        @test pivot[[1, end]]==[-777, -777]&&tau[[1, end]]==fill($T(778), 2)
        @test work[[1, end]]==fill($T(779), 2)&&rwork[[1, end]]==fill($R(780), 2)
        @test all(==($T(777)), a[(m + 1):end, :])
        (; a=a[1:m, :], pivot=pivot[2:(n + 1)], tau=tau[2:(k + 1)])
    end
end
@testset "LAPACKE pivoted QR fixed columns and independent Q consumers" begin
    native=Libdl.dlopen(Colmat.backend_library())
    trampoline=Libdl.dlopen(BLAS.libblastrampoline)
    reference=Libdl.dlopen(
        only(
            filter(
                l->l.interface==:ilp64&&occursin("openblas", lowercase(l.libname)),
                BLAS.get_config().loaded_libs,
            ),
        ).libname,
    )
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64),
        blocked in (false, true),
        (m, n) in ((0, 0), (0, 3), (3, 0), (1, 1), (2, 4), (4, 2), (7, 5)),
        pattern in 0:2,
        deficient in (false, true),
        exponent in (-1, 0, 1)

        k=min(m, n)
        R=real(T)
        tol=R(4000)*eps(R)
        original=T[cos(i+2j)+(i==j ? 2 : 0) for i in 1:m, j in 1:n]
        T<:Complex&&(original .+= T(im) .* T[sin(2i+j)/5 for i in 1:m, j in 1:n])
        deficient&&n>0&&(original[:, 1].=0)
        original .*= R(2)^(exponent*(R==Float32 ? 40 : 400))
        flags=[
            if pattern==0
                0
            elseif pattern==1
                (isodd(j) ? -2 : 0)
            else
                9
            end for j in 1:n
        ]
        fixed=findall(!iszero, flags)
        family=blocked ? "geqp3" : "geqpf"
        fp=Libdl.dlsym(reference, prefix*family*"_64_")
        qname=prefix*(T<:Complex ? "ungqr" : "orgqr")*"_64_"
        qref=Libdl.dlsym(reference, qname)
        qown=Libdl.dlsym(native, qname)
        @test fp!=Libdl.dlsym(native, prefix*family*"_64_")&&qref!=qown
        ext=lapacke288_pivot_reference(T, fp, blocked, original, flags)
        eq=lapacke288_qr_form_q(T, qown, ext.a, ext.tau)
        denom=max(norm(original), floatmin(R))
        @test sort(ext.pivot)==collect(1:n)&&ext.pivot[1:length(fixed)]==fixed
        @test norm(eq*triu(ext.a)-original[:, ext.pivot])/denom<=tol*20
        @test eq'*eq≈Matrix{T}(I, m, m) atol=tol
        for route in (:dll, :lbt_trampoline), layout in (101, 102), high in (false, true)
            name="LAPACKE_"*prefix*family*(high ? "" : "_work")
            suffix=J==Int64 ? "64_" : ""
            p=Libdl.dlsym(route==:dll ? native : trampoline, name*suffix)
            @test BLAS.lbt_get_forward(name, J==Int64 ? :ilp64 : :lp64)==Libdl.dlsym(
                native, name*suffix
            )
            la=layout==101 ? n+2 : m+2
            ai(i, j) = layout==101 ? (i-1)*la+j : i+(j-1)*la
            a=fill(T(NaN), layout==101 ? max(1, m)*la : max(1, n)*la)
            mask=falses(length(a))
            for i in 1:m, j in 1:n
                a[ai(i, j)]=original[i, j]
                mask[ai(i, j)]=true
            end
            saved=copy(a)
            pivot=fill(J(-777), n+2)
            pivot[2:(n + 1)].=flags
            tau=fill(T(778), k+2)
            lw=max(1, 3n+1)
            w=fill(T(779), lw+2)
            rw=fill(R(780), max(1, 2n)+2)
            call(lwork) = lapacke288_pivot_call(
                T,
                J,
                p,
                blocked,
                high,
                layout,
                m,
                n,
                a,
                la,
                view(pivot, 2:(n + 1)),
                view(tau, 2:(k + 1)),
                view(w, 2:(lw + 1)),
                lwork,
                view(rw, 2:(length(rw) - 1)),
                name,
                route,
            )
            if blocked&&!high
                @test call(-1)==0
                @test isequal(a, saved)&&pivot[2:(n + 1)]==flags&&all(==(T(778)), tau)
                @test real(w[2])>0
            end
            @test call(lw)==0
            @test pivot[[1, end]]==fill(J(-777), 2)&&tau[[1, end]]==fill(T(778), 2)
            @test w[[1, end]]==fill(T(779), 2)&&rw[[1, end]]==fill(R(780), 2)
            @test isequal(a[.!mask], saved[.!mask])
            permutation=Int.(pivot[2:(n + 1)])
            @test sort(permutation)==collect(1:n)&&permutation[1:length(fixed)]==fixed
            own=T[a[ai(i, j)] for i in 1:m, j in 1:n]
            q=lapacke288_qr_form_q(T, qref, own, tau[2:(k + 1)])
            @test norm(q*triu(own)-original[:, permutation])/denom<=tol*20
            @test q'*q≈Matrix{T}(I, m, m) atol=tol
        end
    end
end
