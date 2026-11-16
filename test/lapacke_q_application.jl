using Test, Colmat, LinearAlgebra, Libdl
isdefined(@__MODULE__, :lapacke288_selected_entry)||include("lapacke_calls.jl")
for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
    @eval function lapacke288_qapply_call(
        ::Type{$T},
        ::Type{$J},
        p,
        high,
        layout,
        side,
        trans,
        m,
        n,
        k,
        a,
        lda,
        tau,
        c,
        ldc,
        work,
        lwork,
        name,
        route,
    )
        lapacke288_selected_entry(name, $J, route) do
            if high
                ccall(
                    p,
                    $J,
                    (Cint, Int8, Int8, $J, $J, $J, Ptr{$T}, $J, Ptr{$T}, Ptr{$T}, $J),
                    layout,
                    Int8(side),
                    Int8(trans),
                    m,
                    n,
                    k,
                    a,
                    lda,
                    tau,
                    c,
                    ldc,
                )
            else
                ccall(
                    p,
                    $J,
                    (
                        Cint,
                        Int8,
                        Int8,
                        $J,
                        $J,
                        $J,
                        Ptr{$T},
                        $J,
                        Ptr{$T},
                        Ptr{$T},
                        $J,
                        Ptr{$T},
                        $J,
                    ),
                    layout,
                    Int8(side),
                    Int8(trans),
                    m,
                    n,
                    k,
                    a,
                    lda,
                    tau,
                    c,
                    ldc,
                    work,
                    lwork,
                )
            end
        end
    end
end
for T in (Float32, Float64, ComplexF32, ComplexF64)
    @eval function lapacke288_qapply_reference(::Type{$T}, p, side, trans, a0, tau, c0, k)
        m, n=size(c0)
        ar, ac=size(a0)
        la=ar+2
        lc=m+2
        a=fill($T(777), la, ac)
        a[1:ar, :].=a0
        saved=copy(a)
        c=fill($T(777), lc, n)
        c[1:m, :].=c0
        work=fill($T(778), max(1, side=='L' ? n : m)+2)
        info=fill(Int64(-777), 3)
        savedtau=copy(tau)
        ccall(
            p,
            Cvoid,
            (
                Ref{UInt8},
                Ref{UInt8},
                Ref{Int64},
                Ref{Int64},
                Ref{Int64},
                Ptr{$T},
                Ref{Int64},
                Ptr{$T},
                Ptr{$T},
                Ref{Int64},
                Ptr{$T},
                Ref{Int64},
                Ptr{Int64},
            ),
            UInt8(side),
            UInt8(trans),
            m,
            n,
            k,
            a,
            la,
            tau,
            c,
            lc,
            view(work, 2:(length(work) - 1)),
            length(work)-2,
            view(info, 2:2),
        )
        @test info==[-777, 0, -777]
        @test a==saved&&tau==savedtau
        @test all(==($T(777)), c[(m + 1):end, :])
        @test work[[1, end]]==fill($T(778), 2)
        c[1:m, :]
    end
    @eval function lapacke288_qapply_form_q(::Type{$T}, p, a0, tau, k, orientation)
        rows=orientation in ("lq", "rq")
        reverse=orientation in ("ql", "rq")
        order=rows ? size(a0, 2) : size(a0, 1)
        ld=order+2
        q=fill($T(777), ld, order)
        q[1:order, :].=0
        if rows
            offset=reverse ? order-k : 0
            q[(offset + 1):(offset + k), :].=a0
        else
            offset=reverse ? order-k : 0
            q[1:order, (offset + 1):(offset + k)].=a0
        end
        info=fill(Int64(-777), 3)
        work=fill($T(778), max(1, order)+2)
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
            view(work, 2:(length(work) - 1)),
            length(work)-2,
            view(info, 2:2),
        )
        @test info==[-777, 0, -777]
        @test all(==($T(777)), q[(order + 1):end, :])
        @test work[[1, end]]==fill($T(778), 2)
        q[1:order, :]
    end
end
@testset "LAPACKE Q application: independent products and const storage" begin
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
        orientation in ("qr", "lq", "ql", "rq"),
        (m, n) in ((0, 0), (0, 3), (3, 0), (1, 1), (2, 4), (4, 2), (3, 3)),
        side in ('L', 'R')

        rows=orientation in ("lq", "rq")
        reverse=orientation in ("ql", "rq")
        order=side=='L' ? m : n
        family=(T<:Complex ? "unm" : "orm")*orientation
        refp=Libdl.dlsym(reference, prefix*family*"_64_")
        qref=Libdl.dlsym(reference, prefix*(T<:Complex ? "ung" : "org")*orientation*"_64_")
        @test refp!=Libdl.dlsym(native, prefix*family*"_64_")
        R=real(T)
        tol=R(2000)*eps(R)
        for k in unique((0, min(1, order), order))
            ar, ac=rows ? (k, order) : (order, k)
            a0=T[cos(i+2j)/5 for i in 1:ar, j in 1:ac]
            T<:Complex&&(a0 .+= T(im) .* T[sin(2i+j)/7 for i in 1:ar, j in 1:ac])
            tau=fill(T(779), k+2)
            for h in 1:k
                head=reverse ? order-k+h : h
                active=reverse ? (1:(head - 1)) : ((head + 1):order)
                norm2=one(R)
                for i in active
                    norm2+=abs2(rows ? a0[h, i] : a0[i, h])
                end
                tau[h + 1]=T<:Complex ? (one(T)-T(cis(0.7+h/5)))/norm2 : T(2/norm2)
            end
            savedtau=copy(tau)
            tv=view(tau, 2:(k + 1))
            q=lapacke288_qapply_form_q(T, qref, a0, tv, k, orientation)
            @test q'*q≈Matrix{T}(I, order, order) atol=tol
            for trans in ('N', T<:Complex ? 'C' : 'T'), exponent in (-1, 0, 1)
                c0=T[sin(i+2j)+(i==j ? 1 : 0) for i in 1:m, j in 1:n]
                T<:Complex&&(c0 .+= T(im) .* T[cos(2i+j)/3 for i in 1:m, j in 1:n])
                c0 .*= R(2)^(exponent*(R==Float32 ? 40 : 400))
                expected=lapacke288_qapply_reference(T, refp, side, trans, a0, tv, c0, k)
                op=trans=='N' ? q : q'
                product=side=='L' ? op*c0 : c0*op
                denom=max(norm(c0), floatmin(R))
                @test norm(expected-product)/denom<=tol*20
                for route in (:dll, :lbt_trampoline),
                    layout in (101, 102),
                    high in (false, true)

                    name="LAPACKE_"*prefix*family*(high ? "" : "_work")
                    suffix=J==Int64 ? "64_" : ""
                    p=Libdl.dlsym(route==:dll ? native : trampoline, name*suffix)
                    @test BLAS.lbt_get_forward(name, J==Int64 ? :ilp64 : :lp64)==Libdl.dlsym(
                        native, name*suffix
                    )
                    la=layout==101 ? ac+2 : ar+2
                    lc=layout==101 ? n+2 : m+2
                    ai(i, j) = layout==101 ? (i-1)*la+j : i+(j-1)*la
                    ci(i, j) = layout==101 ? (i-1)*lc+j : i+(j-1)*lc
                    a=fill(T(777), layout==101 ? max(1, ar)*la : max(1, ac)*la)
                    c=fill(T(777), layout==101 ? max(1, m)*lc : max(1, n)*lc)
                    mask=falses(length(c))
                    for i in 1:ar, j in 1:ac
                        a[ai(i, j)]=a0[i, j]
                    end
                    for i in 1:m, j in 1:n
                        c[ci(i, j)]=c0[i, j]
                        mask[ci(i, j)]=true
                    end
                    saved_a=copy(a)
                    before=copy(c)
                    minimum=max(1, side=='L' ? n : m)
                    work=fill(T(778), minimum+2)
                    call(lw) = lapacke288_qapply_call(
                        T,
                        J,
                        p,
                        high,
                        layout,
                        side,
                        trans,
                        m,
                        n,
                        k,
                        a,
                        la,
                        tv,
                        c,
                        lc,
                        view(work, 2:(minimum + 1)),
                        lw,
                        name,
                        route,
                    )
                    if !high
                        @test call(-1)==0
                        @test a==saved_a&&c==before&&tau==savedtau
                        @test real(work[2])>=minimum
                        @test call(minimum-1)==-13
                        @test a==saved_a&&c==before&&tau==savedtau
                    end
                    @test call(minimum)==0
                    actual=T[c[ci(i, j)] for i in 1:m, j in 1:n]
                    @test norm(actual-expected)/denom<=tol*20
                    @test norm(actual-product)/denom<=tol*20
                    @test a==saved_a&&tau==savedtau
                    @test c[.!mask]==before[.!mask]
                    @test work[[1, end]]==fill(T(778), 2)
                end
            end
        end
    end
end
