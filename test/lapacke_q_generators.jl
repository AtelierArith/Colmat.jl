using Test, Colmat, LinearAlgebra, Libdl
isdefined(@__MODULE__, :lapacke288_selected_entry)||include("lapacke_calls.jl")
for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
    @eval function lapacke288_qgen_call(
        ::Type{$T}, ::Type{$J}, p, high, layout, m, n, k, a, ld, tau, work, lw, name, route
    )
        lapacke288_selected_entry(name, $J, route) do
            if high
                ccall(
                    p,
                    $J,
                    (Cint, $J, $J, $J, Ptr{$T}, $J, Ptr{$T}),
                    layout,
                    m,
                    n,
                    k,
                    a,
                    ld,
                    tau,
                )
            else
                ccall(
                    p,
                    $J,
                    (Cint, $J, $J, $J, Ptr{$T}, $J, Ptr{$T}, Ptr{$T}, $J),
                    layout,
                    m,
                    n,
                    k,
                    a,
                    ld,
                    tau,
                    work,
                    lw,
                )
            end
        end
    end
end
@testset "LAPACKE Q generation diagnostics and NaN policy" begin
    native=Libdl.dlopen(Colmat.backend_library())
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64), orientation in ("qr", "lq", "ql", "rq"),
        layout in (101, 102)

        family=(T<:Complex ? "ung" : "org")*orientation
        suffix=J==Int64 ? "64_" : ""
        name="LAPACKE_"*prefix*family
        high=Libdl.dlsym(native, name*suffix)
        workp=Libdl.dlsym(native, name*"_work"*suffix)
        a=zeros(T, 16)
        tau=T[1, NaN, 777]
        work=fill(T(778), 4)
        low(l, m, n, k, ld, lw) = lapacke288_qgen_call(
            T, J, workp, false, l, m, n, k, a, ld, tau, work, lw, name*"_work", :dll
        )
        top(k) = lapacke288_qgen_call(
            T, J, high, true, layout, 2, 2, k, a, 4, tau, work, 2, name, :dll
        )
        @test low(0, 2, 2, 1, 4, 2)==-1
        @test low(102, -1, 2, 1, 4, 2)==-2
        @test low(102, 2, -1, 1, 4, 2)==-3
        @test low(102, 2, 2, -1, 4, 2)==-4
        @test low(102, 2, 2, 3, 4, 2)==-4
        @test low(101, -1, 2, 1, 1, 2)==-6
        @test low(102, 2, 2, 1, 1, 2)==-6
        @test low(layout, 2, 2, 1, 4, 1)==-9
        @test a==zeros(T, 16)
        # Every generator ignores the stored diagonal numerically, but its high
        # interface scans the full logical A before querying.
        a[1]=T(NaN)
        before=copy(a)
        savedtau=copy(tau)
        @test top(1)==-5
        @test isequal(a, before)&&isequal(tau, savedtau)
        @test low(layout, 2, 2, 1, 4, 2)==0
        @test all(isfinite, T[a[1], a[2], a[5], a[6]])
        fill!(a, 0)
        tau[1]=T(NaN)
        before=copy(a)
        @test top(1)==-7
        @test a==before
        # NaN beyond K is ignored and const TAU remains intact.
        tau[1]=one(T)
        savedtau=copy(tau)
        @test top(1)==0
        @test isequal(tau, savedtau)
        fill!(a, 0)
        tau[1]=T(NaN)
        savedtau=copy(tau)
        @test top(0)==0
        @test isequal(tau, savedtau)
    end
end
for T in (Float32, Float64, ComplexF32, ComplexF64)
    @eval function lapacke288_qgen_reference(::Type{$T}, p, original, tau, k, rows)
        m, n=size(original)
        ld=m+2
        a=fill($T(777), ld, n)
        a[1:m, :].=original
        work=fill($T(778), max(1, rows ? m : n)+2)
        info=fill(Int64(-777), 3)
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
            m,
            n,
            k,
            a,
            ld,
            tau,
            view(work, 2:(length(work) - 1)),
            length(work)-2,
            view(info, 2:2),
        )
        @test info==[-777, 0, -777]
        @test work[[1, end]]==fill($T(778), 2)
        @test all(==($T(777)), a[(m + 1):end, :])
        a[1:m, :]
    end
end
@testset "LAPACKE explicit Q: independent reflectors and thin/full outputs" begin
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
        J in (Int32, Int64), orientation in ("qr", "lq", "ql", "rq"),
        (short, long) in ((0, 0), (0, 3), (1, 1), (2, 4), (3, 3), (3, 5))

        rows=orientation in ("lq", "rq")
        reverse=orientation in ("ql", "rq")
        m, n=rows ? (short, long) : (long, short)
        order=rows ? n : m
        extent=min(m, n)
        family=(T<:Complex ? "ung" : "org")*orientation
        refp=Libdl.dlsym(reference, prefix*family*"_64_")
        @test refp!=Libdl.dlsym(native, prefix*family*"_64_")
        tol=real(T)(1000)*eps(real(T))
        for k in unique((0, min(1, extent), extent))
            original=T[cos(i+2j)/5 for i in 1:m, j in 1:n]
            T<:Complex&&(original .+= T(im) .* T[sin(2i+j)/7 for i in 1:m, j in 1:n])
            tau=fill(T(777), k+2)
            for h in 1:k
                head=reverse ? order-k+h : h
                slot=reverse ? extent-k+h : h
                active=reverse ? (1:(head - 1)) : ((head + 1):order)
                norm2=one(real(T))
                for i in active
                    norm2+=abs2(rows ? original[slot, i] : original[i, slot])
                end
                tau[h + 1]=T<:Complex ? (one(T)-T(cis(0.7+h/5)))/norm2 : T(2/norm2)
            end
            savedtau=copy(tau)
            expected=lapacke288_qgen_reference(
                T, refp, original, view(tau, 2:(k + 1)), k, rows
            )
            @test tau==savedtau
            @test (rows ? expected*expected' : expected'*expected)≈Matrix{T}(
                I, extent, extent
            ) atol=tol
            for route in (:dll, :lbt_trampoline),
                layout in (101, 102),
                high in (false, true)

                name="LAPACKE_"*prefix*family*(high ? "" : "_work")
                suffix=J==Int64 ? "64_" : ""
                p=Libdl.dlsym(route==:dll ? native : trampoline, name*suffix)
                @test BLAS.lbt_get_forward(name, J==Int64 ? :ilp64 : :lp64)==Libdl.dlsym(
                    native, name*suffix
                )
                ld=layout==101 ? n+2 : m+2
                at(i, j) = layout==101 ? (i-1)*ld+j : i+(j-1)*ld
                a=fill(T(NaN), layout==101 ? max(1, m)*ld : max(1, n)*ld)
                mask=falses(length(a))
                for i in 1:m, j in 1:n
                    a[at(i, j)]=original[i, j]
                    mask[at(i, j)]=true
                end
                before=copy(a)
                minimum=max(1, rows ? m : n)
                work=fill(T(778), minimum+2)
                call(lw) = lapacke288_qgen_call(
                    T,
                    J,
                    p,
                    high,
                    layout,
                    m,
                    n,
                    k,
                    a,
                    ld,
                    view(tau, 2:(k + 1)),
                    view(work, 2:(minimum + 1)),
                    lw,
                    name,
                    route,
                )
                if !high
                    @test call(-1)==0
                    @test isequal(a, before)&&tau==savedtau
                    @test real(work[2])>=minimum
                    @test call(minimum-1)==-9
                    @test isequal(a, before)&&tau==savedtau
                end
                @test call(minimum)==0
                actual=T[a[at(i, j)] for i in 1:m, j in 1:n]
                @test actual≈expected atol=tol rtol=tol
                @test (rows ? actual*actual' : actual'*actual)≈Matrix{T}(I, extent, extent) atol=tol
                @test tau==savedtau
                @test isequal(a[.!mask], before[.!mask])
                @test work[[1, end]]==fill(T(778), 2)
            end
        end
    end
end
