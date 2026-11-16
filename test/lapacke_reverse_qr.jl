using Test, Colmat, LinearAlgebra, Libdl
isdefined(@__MODULE__, :lapacke288_selected_entry)||include("lapacke_calls.jl")
for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
    @eval function lapacke288_reverse_qr_call(
        ::Type{$T},
        ::Type{$J},
        p,
        blocked,
        high,
        layout,
        m,
        n,
        a,
        ld,
        tau,
        work,
        lwork,
        name,
        route,
    )
        lapacke288_selected_entry(name, $J, route) do
            if high
                ccall(p, $J, (Cint, $J, $J, Ptr{$T}, $J, Ptr{$T}), layout, m, n, a, ld, tau)
            elseif blocked
                ccall(
                    p,
                    $J,
                    (Cint, $J, $J, Ptr{$T}, $J, Ptr{$T}, Ptr{$T}, $J),
                    layout,
                    m,
                    n,
                    a,
                    ld,
                    tau,
                    work,
                    lwork,
                )
            else
                ccall(
                    p,
                    $J,
                    (Cint, $J, $J, Ptr{$T}, $J, Ptr{$T}, Ptr{$T}),
                    layout,
                    m,
                    n,
                    a,
                    ld,
                    tau,
                    work,
                )
            end
        end
    end
end
for T in (Float32, Float64, ComplexF32, ComplexF64)
    @eval function lapacke288_reverse_qr_reference(::Type{$T}, p, blocked, original, rq)
        m, n=size(original)
        ld=m+2
        k=min(m, n)
        a=fill($T(NaN), ld, n)
        a[1:m, :].=original
        tau=fill($T(777), k+2)
        work=fill($T(777), (rq ? max(1, m) : max(1, n))+2)
        info=fill(Int64(-777), 3)
        if blocked
            ccall(
                p,
                Cvoid,
                (
                    Ref{Int64},
                    Ref{Int64},
                    Ptr{$T},
                    Ref{Int64},
                    Ptr{$T},
                    Ptr{$T},
                    Ref{Int64},
                    Ptr{Int64},
                ),
                m,
                n,
                a,
                ld,
                view(tau, 2:(k + 1)),
                view(work, 2:(length(work) - 1)),
                (rq ? (n==0 ? 1 : max(1, m)) : (m==0 ? 1 : max(1, n))),
                view(info, 2:2),
            )
        else
            ccall(
                p,
                Cvoid,
                (Ref{Int64}, Ref{Int64}, Ptr{$T}, Ref{Int64}, Ptr{$T}, Ptr{$T}, Ptr{Int64}),
                m,
                n,
                a,
                ld,
                view(tau, 2:(k + 1)),
                view(work, 2:(length(work) - 1)),
                view(info, 2:2),
            )
        end
        @test info==[-777, 0, -777]
        @test tau[[1, end]]==work[[1, end]]==fill($T(777), 2)
        (; a=a[1:m, :], tau=tau[2:(k + 1)])
    end
    @eval function lapacke288_reverse_qr_form_q(::Type{$T}, p, factors, tau, rq)
        m, n=size(factors)
        k=min(m, n)
        order=rq ? n : m
        ld=order+2
        q=fill($T(NaN), ld, order)
        for j in 1:k, i in 1:(order - k + j - 1)
            if rq
                q[order - k + j, i]=factors[m - k + j, i]
            else
                q[i, order - k + j]=factors[i, n - k + j]
            end
        end
        work=zeros($T, 1)
        info=Ref{Int64}(-777)
        function invoke(lwork)
            ccall(
                p,
                Cvoid,
                (
                    Ref{Int64},
                    Ref{Int64},
                    Ref{Int64},
                    Ptr{$T},
                    Ref{Int64},
                    Ptr{$T},
                    Ptr{$T},
                    Ref{Int64},
                    Ptr{Int64},
                ),
                order,
                order,
                k,
                q,
                ld,
                tau,
                work,
                lwork,
                info,
            )
        end
        invoke(-1)
        @test info[]==0
        resize!(work, max(1, Int(real(work[1]))))
        invoke(length(work))
        @test info[]==0
        @test all(isnan, q[(order + 1):end, :])
        q[1:order, :]
    end
end
@testset "LAPACKE GEQLF/GERQF independent factor consumers" begin
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
    shapes=((0, 0), (0, 3), (3, 0), (1, 1), (2, 4), (4, 2), (3, 3), (7, 5), (5, 7))
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64),
        rq in (false, true),
        (m, n) in shapes,
        blocked in (true,),
        deficient in (false, true),
        exponent in (-1, 0, 1)

        R=real(T)
        scale=R(2)^(exponent*(R==Float32 ? 40 : 400))
        tol=R(500)*eps(R)
        k=min(m, n)
        order=rq ? n : m
        minimum=rq ? (n==0 ? 1 : max(1, m)) : (m==0 ? 1 : max(1, n))
        original=T[cos(i+2j)+(i==j ? 2 : 0) for i in 1:m, j in 1:n]
        T<:Complex&&(original .+= T(im) .* T[sin(2i+j)/5 for i in 1:m, j in 1:n])
        deficient&&n>0&&(original[:, 1].=0)
        original .*= scale
        family=rq ? (blocked ? "gerqf" : "gerq2") : (blocked ? "geqlf" : "geql2")
        refp=Libdl.dlsym(reference, prefix*family*"_64_")
        qname=prefix*((T<:Complex ? "ung" : "org")*(rq ? "rq" : "ql"))*"_64_"
        qref=Libdl.dlsym(reference, qname)
        qown=Libdl.dlsym(native, qname)
        @test refp!=Libdl.dlsym(native, prefix*family*"_64_")&&qref!=qown
        expected=lapacke288_reverse_qr_reference(T, refp, blocked, original, rq)
        independent_q=lapacke288_reverse_qr_form_q(T, qown, expected.a, expected.tau, rq)
        denom=max(norm(original), floatmin(R))
        triangle=T[
            (rq ? j-i>=n-m : i-j>=m-n) ? expected.a[i, j] : zero(T) for i in 1:m, j in 1:n
        ]
        @test norm((rq ? triangle*independent_q : independent_q*triangle)-original)/denom<=tol*20
        @test independent_q'*independent_q≈Matrix{T}(I, order, order) atol=tol
        for route in (:dll, :lbt_trampoline), layout in (101, 102), high in (false, true)
            name="LAPACKE_"*prefix*family*(high ? "" : "_work")
            suffix=J==Int64 ? "64_" : ""
            @test BLAS.lbt_get_forward(name, J==Int64 ? :ilp64 : :lp64)==Libdl.dlsym(
                native, name*suffix
            )
            p=Libdl.dlsym(route==:dll ? native : trampoline, name*suffix)
            ld=layout==101 ? n+2 : m+2
            at(i, j) = layout==101 ? (i-1)*ld+j : i+(j-1)*ld
            a=fill(T(NaN), layout==101 ? max(1, m)*ld : max(1, n)*ld)
            mask=falses(length(a))
            for i in 1:m, j in 1:n
                a[at(i, j)]=original[i, j]
                mask[at(i, j)]=true
            end
            before=copy(a)
            tau=fill(T(777), k+2)
            work=fill(T(777), minimum+2)
            call(lw) = lapacke288_reverse_qr_call(
                T,
                J,
                p,
                blocked,
                high,
                layout,
                m,
                n,
                a,
                ld,
                view(tau, 2:(k + 1)),
                view(work, 2:(length(work) - 1)),
                lw,
                name,
                route,
            )
            if blocked&&!high
                @test call(-1)==0
                @test isequal(a, before)&&all(==(T(777)), tau)
                @test real(work[2])>=minimum
                @test call(minimum-1)==-8
                @test isequal(a, before)&&all(==(T(777)), tau)
            end
            @test call(minimum)==0
            @test isequal(a[.!mask], before[.!mask])
            @test tau[[1, end]]==work[[1, end]]==fill(T(777), 2)
            if blocked&&!high&&(m==0||n==0)
                @test work[2]==one(T)
            end
            factors=T[a[at(i, j)] for i in 1:m, j in 1:n]
            q=lapacke288_reverse_qr_form_q(T, qref, factors, tau[2:(k + 1)], rq)
            triangle=T[
                (rq ? j-i>=n-m : i-j>=m-n) ? factors[i, j] : zero(T) for i in 1:m, j in 1:n
            ]
            @test norm((rq ? triangle*q : q*triangle)-original)/denom<=tol*20
            @test q'*q≈Matrix{T}(I, order, order) atol=tol
        end
    end
end
