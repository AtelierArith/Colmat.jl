using Test, Colmat, LinearAlgebra, Libdl
isdefined(@__MODULE__, :lapacke288_selected_entry)||include("lapacke_calls.jl")
for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
    @eval function lapacke288_compact_factor(
        ::Type{$T},
        ::Type{$J},
        p,
        variant,
        high,
        layout,
        m,
        n,
        nb,
        a,
        la,
        t,
        lt,
        w,
        name,
        route,
    )
        lapacke288_selected_entry(name, $J, route) do
            if variant==1
                if high
                    ccall(
                        p,
                        $J,
                        (Cint, $J, $J, $J, Ptr{$T}, $J, Ptr{$T}, $J),
                        layout,
                        m,
                        n,
                        nb,
                        a,
                        la,
                        t,
                        lt,
                    )
                else
                    ccall(
                        p,
                        $J,
                        (Cint, $J, $J, $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$T}),
                        layout,
                        m,
                        n,
                        nb,
                        a,
                        la,
                        t,
                        lt,
                        w,
                    )
                end
            else
                ccall(
                    p,
                    $J,
                    (Cint, $J, $J, Ptr{$T}, $J, Ptr{$T}, $J),
                    layout,
                    m,
                    n,
                    a,
                    la,
                    t,
                    lt,
                )
            end
        end
    end
    @eval function lapacke288_compact_apply(
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
        nb,
        v,
        lv,
        t,
        lt,
        c,
        lc,
        w,
        name,
        route,
    )
        lapacke288_selected_entry(name, $J, route) do
            if high
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
                        $J,
                        Ptr{$T},
                        $J,
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
                    nb,
                    v,
                    lv,
                    t,
                    lt,
                    c,
                    lc,
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
                        $J,
                        Ptr{$T},
                        $J,
                        Ptr{$T},
                        $J,
                        Ptr{$T},
                        $J,
                        Ptr{$T},
                    ),
                    layout,
                    Int8(side),
                    Int8(trans),
                    m,
                    n,
                    k,
                    nb,
                    v,
                    lv,
                    t,
                    lt,
                    c,
                    lc,
                    w,
                )
            end
        end
    end
end
for T in (Float32, Float64, ComplexF32, ComplexF64)
    @eval function lapacke288_compact_ref_factor(::Type{$T}, p, variant, original, nb)
        m, n=size(original)
        k=min(m, n)
        la=m+2
        lt=nb+2
        a=fill($T(777), la, n)
        a[1:m, :].=original
        t=zeros($T, lt, k)
        t[(nb + 1):end, :].=$T(778)
        w=fill($T(779), max(1, nb*max(1, n))+2)
        info=fill(Int64(-777), 3)
        if variant==1
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
                    Ref{Int64},
                    Ptr{$T},
                    Ptr{Int64},
                ),
                m,
                n,
                nb,
                a,
                la,
                t,
                lt,
                view(w, 2:(length(w) - 1)),
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
                    Ptr{$T},
                    Ref{Int64},
                    Ptr{Int64},
                ),
                m,
                n,
                a,
                la,
                t,
                lt,
                view(info, 2:2),
            )
        end
        @test info==[-777, 0, -777]
        @test all(==($T(777)), a[(m + 1):end, :])&&all(==($T(778)), t[(nb + 1):end, :])
        @test w[[1, end]]==fill($T(779), 2)
        block=t[1:nb, :]
        # The strict lower parts of block T are undefined factor output; use
        # finite values there when exercising high-level full-matrix NaN scans.
        for j in 1:k, i in 1:nb
            i>(j-1)%nb+1&&(block[i, j]=zero($T))
        end
        (; a=a[1:m, :], t=block)
    end
    @eval function lapacke288_compact_ref_apply(
        ::Type{$T}, p, side, trans, v0, t0, c0, k, nb
    )
        m, n=size(c0)
        order=side=='L' ? m : n
        lv=order+2
        lt=nb+2
        lc=m+2
        v=fill($T(777), lv, k)
        v[1:order, :].=v0[:, 1:k]
        t=fill($T(778), lt, k)
        t[1:nb, :].=t0
        savedv=copy(v)
        savedt=copy(t)
        c=fill($T(779), lc, n)
        c[1:m, :].=c0
        w=fill($T(780), max(1, nb*max(1, side=='L' ? n : m))+2)
        info=fill(Int64(-777), 3)
        ccall(
            p,
            Cvoid,
            (
                Ref{UInt8},
                Ref{UInt8},
                Ref{Int64},
                Ref{Int64},
                Ref{Int64},
                Ref{Int64},
                Ptr{$T},
                Ref{Int64},
                Ptr{$T},
                Ref{Int64},
                Ptr{$T},
                Ref{Int64},
                Ptr{$T},
                Ptr{Int64},
            ),
            UInt8(side),
            UInt8(trans),
            m,
            n,
            k,
            nb,
            v,
            lv,
            t,
            lt,
            c,
            lc,
            view(w, 2:(length(w) - 1)),
            view(info, 2:2),
        )
        @test info==[-777, 0, -777]
        @test v==savedv&&t==savedt
        @test all(==($T(779)), c[(m + 1):end, :])&&w[[1, end]]==fill($T(780), 2)
        c[1:m, :]
    end
end
@testset "LAPACKE compact QR factors and independent block consumers" begin
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
    shapes=((0, 0), (0, 3), (3, 0), (1, 1), (2, 4), (4, 2), (7, 2), (7, 5))
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64), variant in 1:3,
        (m, n) in shapes

        variant>1&&m<n&&continue
        k=min(m, n)
        R=real(T)
        tol=R(3000)*eps(R)
        blocks=variant==1 ? unique((1, max(1, min(2, k)), max(1, k))) : (max(1, n),)
        for nb in blocks, deficient in (false, true), exponent in (-1, 0, 1)
            original=T[cos(i+2j)+(i==j ? 2 : 0) for i in 1:m, j in 1:n]
            T<:Complex&&(original .+= T(im) .* T[sin(2i+j)/5 for i in 1:m, j in 1:n])
            deficient&&n>0&&(original[:, 1].=0)
            original .*= R(2)^(exponent*(R==Float32 ? 40 : 400))
            family=("geqrt", "geqrt2", "geqrt3")[variant]
            # Isolated evidence records reference GEQRT3 recursion at N=0.
            oracle=variant==3&&n==0 ? "geqrt2" : family
            fp=Libdl.dlsym(reference, prefix*oracle*"_64_")
            gp=Libdl.dlsym(reference, prefix*"gemqrt_64_")
            @test fp!=Libdl.dlsym(native, prefix*oracle*"_64_")
            @test gp!=Libdl.dlsym(native, prefix*"gemqrt_64_")
            external=lapacke288_compact_ref_factor(T, fp, variant, original, nb)
            denom=max(norm(original), floatmin(R))
            for route in (:dll, :lbt_trampoline),
                layout in (101, 102),
                high in (false, true)

                name="LAPACKE_"*prefix*family*(high ? "" : "_work")
                suffix=J==Int64 ? "64_" : ""
                p=Libdl.dlsym(route==:dll ? native : trampoline, name*suffix)
                @test BLAS.lbt_get_forward(name, J==Int64 ? :ilp64 : :lp64)==Libdl.dlsym(
                    native, name*suffix
                )
                la=layout==101 ? n+2 : m+2
                lt=layout==101 ? k+2 : nb+2
                ai(i, j) = layout==101 ? (i-1)*la+j : i+(j-1)*la
                ti(i, j) = layout==101 ? (i-1)*lt+j : i+(j-1)*lt
                a=fill(T(777), layout==101 ? max(1, m)*la : max(1, n)*la)
                am=falses(length(a))
                t=fill(T(778), layout==101 ? nb*lt : max(1, k)*lt)
                tm=falses(length(t))
                for i in 1:m, j in 1:n
                    a[ai(i, j)]=original[i, j]
                    am[ai(i, j)]=true
                end
                for i in 1:nb, j in 1:k
                    t[ti(i, j)]=zero(T)
                    tm[ti(i, j)]=true
                end
                beforea=copy(a)
                beforet=copy(t)
                w=fill(T(779), max(1, nb*max(1, n))+2)
                @test lapacke288_compact_factor(
                    T,
                    J,
                    p,
                    variant,
                    high,
                    layout,
                    m,
                    n,
                    nb,
                    a,
                    la,
                    t,
                    lt,
                    view(w, 2:(length(w) - 1)),
                    name,
                    route,
                )==0
                @test a[.!am]==beforea[.!am]&&t[.!tm]==beforet[.!tm]
                @test w[[1, end]]==fill(T(779), 2)
                own_a=T[a[ai(i, j)] for i in 1:m, j in 1:n]
                own_t=T[t[ti(i, j)] for i in 1:nb, j in 1:k]
                q=lapacke288_compact_ref_apply(
                    T, gp, 'L', 'N', own_a, own_t, Matrix{T}(I, m, m), k, nb
                )
                @test q'*q≈Matrix{T}(I, m, m) atol=tol
                @test norm(q*triu(own_a)-original)/denom<=tol*20
                # Reverse interoperability: own C GEMQRT consumes reference factors.
                vn="LAPACKE_"*prefix*"gemqrt"*(high ? "" : "_work")
                vp=Libdl.dlsym(route==:dll ? native : trampoline, vn*suffix)
                @test BLAS.lbt_get_forward(vn, J==Int64 ? :ilp64 : :lp64)==Libdl.dlsym(
                    native, vn*suffix
                )
                lv=layout==101 ? k+2 : m+2
                lc=m+2
                vi(i, j) = layout==101 ? (i-1)*lv+j : i+(j-1)*lv
                ci(i, j) = layout==101 ? (i-1)*lc+j : i+(j-1)*lc
                v=fill(T(777), layout==101 ? max(1, m)*lv : max(1, k)*lv)
                rt=fill(T(778), length(t))
                c=fill(T(780), max(1, m)*lc)
                cm=falses(length(c))
                for i in 1:m, j in 1:k
                    v[vi(i, j)]=external.a[i, j]
                end
                for i in 1:nb, j in 1:k
                    rt[ti(i, j)]=external.t[i, j]
                end
                for i in 1:m, j in 1:m
                    c[ci(i, j)]=T(i==j)
                    cm[ci(i, j)]=true
                end
                savedv=copy(v)
                savedrt=copy(rt)
                savedc=copy(c)
                cw=fill(T(781), max(1, m*nb)+2)
                @test lapacke288_compact_apply(
                    T,
                    J,
                    vp,
                    high,
                    layout,
                    'L',
                    'N',
                    m,
                    m,
                    k,
                    nb,
                    v,
                    lv,
                    rt,
                    lt,
                    c,
                    lc,
                    view(cw, 2:(length(cw) - 1)),
                    vn,
                    route,
                )==0
                @test v==savedv&&rt==savedrt&&c[.!cm]==savedc[.!cm]
                @test cw[[1, end]]==fill(T(781), 2)
                oq=T[c[ci(i, j)] for i in 1:m, j in 1:m]
                @test oq'*oq≈Matrix{T}(I, m, m) atol=tol
                @test norm(oq*triu(external.a)-original)/denom<=tol*20
            end
        end
    end
end

@testset "GEMQRT rectangular left/right and transpose interoperability" begin
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
        side in ('L', 'R'),
        trans in (T<:Complex ? ('N', 'C') : ('N', 'T')),
        (m, n) in ((3, 7), (7, 3), (0, 5), (5, 0)),
        k0 in (0, 1, 2),
        nb0 in (1, 2)

        order=side=='L' ? m : n
        k=min(order, k0)
        nb=max(1, min(nb0, k))
        original=T[cos(i+2j)+(i==j ? 2 : 0) for i in 1:order, j in 1:k]
        T<:Complex&&(original .+= T(im) .* T[sin(2i+j)/5 for i in 1:order, j in 1:k])
        fp=Libdl.dlsym(reference, prefix*"geqrt_64_")
        gp=Libdl.dlsym(reference, prefix*"gemqrt_64_")
        factors=lapacke288_compact_ref_factor(T, fp, 1, original, nb)
        c0=T[sin(i+3j) for i in 1:m, j in 1:n]
        T<:Complex&&(c0 .+= T(im) .* T[cos(3i+j)/7 for i in 1:m, j in 1:n])
        expected=lapacke288_compact_ref_apply(
            T, gp, side, trans, factors.a, factors.t, c0, k, nb
        )
        q=lapacke288_compact_ref_apply(
            T, gp, 'L', 'N', factors.a, factors.t, Matrix{T}(I, order, order), k, nb
        )
        op=trans=='N' ? q : q'
        @test expected≈(side=='L' ? op*c0 : c0*op) atol=real(T)(3000)*eps(real(T))
        for route in (:dll, :lbt_trampoline), layout in (101, 102), high in (false, true)
            name="LAPACKE_"*prefix*"gemqrt"*(high ? "" : "_work")
            suffix=J==Int64 ? "64_" : ""
            p=Libdl.dlsym(route==:dll ? native : trampoline, name*suffix)
            lv=layout==101 ? k+2 : order+2
            lt=layout==101 ? max(k, nb)+2 : nb+2
            lc=layout==101 ? n+2 : m+2
            vi(i, j) = layout==101 ? (i-1)*lv+j : i+(j-1)*lv
            ti(i, j) = layout==101 ? (i-1)*lt+j : i+(j-1)*lt
            ci(i, j) = layout==101 ? (i-1)*lc+j : i+(j-1)*lc
            v=fill(T(NaN), layout==101 ? max(1, order)*lv : max(1, k)*lv)
            t=fill(T(NaN), layout==101 ? nb*lt : max(1, k)*lt)
            c=fill(T(NaN), layout==101 ? max(1, m)*lc : max(1, n)*lc)
            mask=falses(length(c))
            for i in 1:order, j in 1:k
                v[vi(i, j)]=factors.a[i, j]
            end
            for i in 1:nb, j in 1:k
                t[ti(i, j)]=factors.t[i, j]
            end
            for i in 1:m, j in 1:n
                c[ci(i, j)]=c0[i, j]
                mask[ci(i, j)]=true
            end
            sv, st, sc=copy(v), copy(t), copy(c)
            nw=side=='L' ? n : m
            w=fill(T(781), max(1, nw*nb)+2)
            @test lapacke288_compact_apply(
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
                nb,
                v,
                lv,
                t,
                lt,
                c,
                lc,
                view(w, 2:(length(w) - 1)),
                name,
                route,
            )==0
            @test isequal(v, sv)&&isequal(t, st)&&isequal(c[.!mask], sc[.!mask])
            @test w[[1, end]]==fill(T(781), 2)
            actual=T[c[ci(i, j)] for i in 1:m, j in 1:n]
            @test actual≈expected atol=real(T)(3000)*eps(real(T))
        end
    end
end
